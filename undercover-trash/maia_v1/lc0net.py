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

Weight layout in the file follows lc0's canonical order ([out, in, y, x]
convolution kernels, LINEAR16 quantization, denormalized to float32).
"""

from __future__ import annotations

import gzip

import numpy as np
import torch
import torch.nn.functional as F

from .net_pb2 import Net as _NetPb

EPS = 1e-5

# NetworkFormat enums (proto/net.proto)
NETWORK_SE_WITH_HEADFORMAT = 4
INPUT_CLASSICAL_112_PLANE = 1
POLICY_CONVOLUTION = 2
VALUE_WDL = 2
LINEAR16 = 1


class MaiaNetError(RuntimeError):
    pass


def _denorm(layer, shape, device):
    a = np.frombuffer(layer.params, np.uint16).astype(np.float32) / 0xFFFF
    a = a * (layer.max_val - layer.min_val) + layer.min_val
    return torch.from_numpy(a.reshape(shape)).to(device)


def _bn(layer, channels, device):
    gamma = _denorm(layer.bn_gammas, (channels,), device)
    beta = _denorm(layer.bn_betas, (channels,), device)
    mean = _denorm(layer.bn_means, (channels,), device)
    var = _denorm(layer.bn_stddivs, (channels,), device)
    scale = gamma / torch.sqrt(var + EPS)
    shift = beta - mean * scale
    return (scale.view(1, channels, 1, 1), shift.view(1, channels, 1, 1))


class Lc0SeNet:
    """Torch inference engine for a Maia/Lc0 SE network."""

    def __init__(self, path: str, device: str = "cpu"):
        with gzip.open(path, "rb") as f:
            pb = _NetPb()
            pb.ParseFromString(f.read())

        fmt = pb.format
        if fmt.network_format.network != NETWORK_SE_WITH_HEADFORMAT:
            raise MaiaNetError(
                f"{path}: unsupported network structure "
                f"{fmt.network_format.network} (need SE_WITH_HEADFORMAT)")
        if fmt.network_format.input != INPUT_CLASSICAL_112_PLANE:
            raise MaiaNetError(
                f"{path}: unsupported input format {fmt.network_format.input}")
        if fmt.network_format.policy != POLICY_CONVOLUTION:
            raise MaiaNetError(
                f"{path}: unsupported policy head {fmt.network_format.policy} "
                f"(need POLICY_CONVOLUTION)")
        if fmt.weights_encoding != LINEAR16:
            raise MaiaNetError(
                f"{path}: unsupported weight encoding {fmt.weights_encoding}")
        if fmt.network_format.value != VALUE_WDL:
            raise MaiaNetError(
                f"{path}: unsupported value head {fmt.network_format.value} "
                f"(need VALUE_WDL)")

        w = pb.weights
        self.num_residual = len(w.residual)
        # Architecture constants for the released Maia nets:
        # 64 filters, 6 SE-residual blocks, se_ratio 8, policy 3x3->80,
        # value 1x1->32 -> 128 -> 3.
        C = 64

        self.conv1_w = _denorm(w.input.weights, (C, 112, 3, 3), device)
        self.conv1_s, self.conv1_b = _bn(w.input, C, device)

        self.res = []
        for r in w.residual:
            c1w = _denorm(r.conv1.weights, (C, C, 3, 3), device)
            c1s, c1b = _bn(r.conv1, C, device)
            c2w = _denorm(r.conv2.weights, (C, C, 3, 3), device)
            c2s, c2b = _bn(r.conv2, C, device)
            se_w1 = _denorm(r.se.w1, (C // 8, C), device)
            se_b1 = _denorm(r.se.b1, (C // 8,), device)
            se_w2 = _denorm(r.se.w2, (2 * C, C // 8), device)
            se_b2 = _denorm(r.se.b2, (2 * C,), device)
            self.res.append((c1w, c1s, c1b, c2w, c2s, c2b,
                             se_w1, se_b1, se_w2, se_b2))

        # Convolution policy head: conv 3x3 -> 64 (BN+ReLU), conv 3x3 -> 80
        # (plain, with bias), then the fixed az->lc0 policy map.
        self.pol1_w = _denorm(w.policy1.weights, (C, C, 3, 3), device)
        self.pol1_s, self.pol1_b = _bn(w.policy1, C, device)
        self.pol2_w = _denorm(w.policy.weights, (80, C, 3, 3), device)
        self.pol2_b = _denorm(w.policy.biases, (80,), device)

        from .lc0_az_policy_map import make_map
        self.policy_map = torch.from_numpy(make_map()).to(device)  # (5120, 1858)

        # WDL value head: conv 1x1 -> 32 (BN+ReLU), 2048 -> 128 -> 3.
        self.val_w = _denorm(w.value.weights, (32, C, 1, 1), device)
        self.val_s, self.val_b = _bn(w.value, 32, device)
        self.ip1_w = _denorm(w.ip1_val_w, (128, 32 * 64), device)
        self.ip1_b = _denorm(w.ip1_val_b, (128,), device)
        self.ip2_w = _denorm(w.ip2_val_w, (3, 128), device)
        self.ip2_b = _denorm(w.ip2_val_b, (3,), device)

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
