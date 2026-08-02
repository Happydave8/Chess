"""Load a Leela Chess Zero .pb.gz network into PyTorch (SE architecture).

Implements the exact forward graph of the released Maia networks:

    NETWORK_SE_WITH_HEADFORMAT + INPUT_CLASSICAL_112_PLANE
    + POLICY_CONVOLUTION (3x3 -> 80ch, then the fixed 5120x1858 policy map)
    + VALUE_WDL (3 outputs)

mirroring `move_prediction/maia_chess_backend/maia/tfprocess.py` from
CSSLab/maia-chess and lc0's weight semantics:

  batch norm :  y = gamma * (x - mean) / sqrt(var + 1e-5) + beta
  SE unit    :  y = sigmoid(gamma_se) * x + beta_se,
                [gamma_se; beta_se] = W2(relu(W1(pool) + b1)) + b2
  final      :  relu(y + residual)

Weights are parsed with maia_v1/pb_reader.py (pure Python, no protobuf).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .pb_reader import NetArrays, parse_net


class MaiaNetError(RuntimeError):
    pass


class Lc0SeNet:
    """Torch inference engine for a Maia/Lc0 SE network."""

    def __init__(self, path: str, device: str = "cpu"):
        a: NetArrays = parse_net(path)
        if a.conv1_w is None:
            raise MaiaNetError(f"{path}: no input conv weights found")
        if a.pol2_w is None or a.pol2_b is None:
            raise MaiaNetError(f"{path}: no convolution policy head (unsupported)")

        def t(arr):
            return torch.from_numpy(arr).to(device)

        self.num_residual = len(a.res)

        self.conv1_w = t(a.conv1_w)
        self.conv1_s = t(a.conv1_scale).view(1, -1, 1, 1)
        self.conv1_b = t(a.conv1_shift).view(1, -1, 1, 1)

        self.res = []
        for (c1w, c1s, c1b, c2w, c2s, c2b,
             se_w1, se_b1, se_w2, se_b2) in a.res:
            self.res.append((
                t(c1w), t(c1s).view(1, -1, 1, 1), t(c1b).view(1, -1, 1, 1),
                t(c2w), t(c2s).view(1, -1, 1, 1), t(c2b).view(1, -1, 1, 1),
                t(se_w1), t(se_b1), t(se_w2), t(se_b2),
            ))

        self.pol1_w = t(a.pol1_w)
        self.pol1_s = t(a.pol1_scale).view(1, -1, 1, 1)
        self.pol1_b = t(a.pol1_shift).view(1, -1, 1, 1)
        self.pol2_w = t(a.pol2_w)
        self.pol2_b = t(a.pol2_b)

        from .lc0_az_policy_map import make_map
        self.policy_map = torch.from_numpy(make_map()).to(device)  # (5120, 1858)

        self.val_w = t(a.val_w)
        self.val_s = t(a.val_scale).view(1, -1, 1, 1)
        self.val_b = t(a.val_shift).view(1, -1, 1, 1)
        self.ip1_w = t(a.ip1_w)
        self.ip1_b = t(a.ip1_b)
        self.ip2_w = t(a.ip2_w)
        self.ip2_b = t(a.ip2_b)

    def __call__(self, x: torch.Tensor):
        """x: (1, 112, 8, 8) float32 tensor.

        Returns (policy_logits (1,1858), value_logits (1,3)).
        """
        x = F.relu(F.conv2d(x, self.conv1_w, padding=1) * self.conv1_s + self.conv1_b)

        for (c1w, c1s, c1b, c2w, c2s, c2b,
             se_w1, se_b1, se_w2, se_b2) in self.res:
            h = F.relu(F.conv2d(x, c1w, padding=1) * c1s + c1b)
            h = F.conv2d(h, c2w, padding=1) * c2s + c2b
            pooled = h.mean(dim=(2, 3))                       # (1, C)
            fc1 = F.relu(pooled @ se_w1.t() + se_b1)          # (1, C/8)
            excited = fc1 @ se_w2.t() + se_b2                 # (1, 2C)
            gamma_se, beta_se = excited.chunk(2, dim=1)
            x = F.relu(torch.sigmoid(gamma_se).unsqueeze(-1).unsqueeze(-1) * h
                       + beta_se.unsqueeze(-1).unsqueeze(-1) + x)

        # policy
        h = F.relu(F.conv2d(x, self.pol1_w, padding=1) * self.pol1_s + self.pol1_b)
        h = F.conv2d(h, self.pol2_w, padding=1, bias=self.pol2_b)   # (1, 80, 8, 8)
        flat = h.reshape(1, 80 * 64)
        policy_logits = flat @ self.policy_map                      # (1, 1858)

        # value
        v = F.relu(F.conv2d(x, self.val_w) * self.val_s + self.val_b)  # (1, 32, 8, 8)
        v = v.reshape(1, 32 * 64)
        v = F.relu(v @ self.ip1_w.t() + self.ip1_b)
        value_logits = v @ self.ip2_w.t() + self.ip2_b               # (1, 3)

        return policy_logits, value_logits
