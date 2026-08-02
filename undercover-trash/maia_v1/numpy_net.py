"""Numpy-only inference for a Maia/Lc0 SE network.

Same forward graph as maia_v1/lc0net.py (which uses PyTorch), implemented
with plain numpy + im2col convolutions so it runs on platforms where torch
is not available (Termux/Android, minimal containers, ...).

Weights come from maia_v1/pb_reader.parse_net() — no protobuf dependency.
"""

from __future__ import annotations

import numpy as np

from .pb_reader import NetArrays, parse_net


def conv2d(x: np.ndarray, w: np.ndarray, padding: int = 1) -> np.ndarray:
    """x: (C, H, W) float32, w: (O, C, K, K) -> (O, H, W).

    Patch columns are ordered (c, i, j) lexicographically to match torch's
    conv2d kernel flattening (w[o, c, i, j] with index c*K*K + i*K + j).
    """
    C, H, W = x.shape
    O, _, K, _ = w.shape
    if padding:
        xp = np.pad(x, ((0, 0), (padding, padding), (padding, padding)))
    else:
        xp = x
    # im2col: (C*K*K, H*W)
    cols = np.zeros((C * K * K, H * W), dtype=np.float32)
    n = 0
    for c in range(C):
        for i in range(K):
            for j in range(K):
                cols[n] = xp[c, i:i + H, j:j + W].reshape(H * W)
                n += 1
    out = w.reshape(O, C * K * K) @ cols
    return out.reshape(O, H, W)


def conv2d_1x1(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """x: (C, H, W), w: (O, C, 1, 1) -> (O, H, W)."""
    O = w.shape[0]
    return (w.reshape(O, -1) @ x.reshape(x.shape[0], -1)).reshape(O, x.shape[1], x.shape[2])


class NumpyNet:
    """Numpy inference engine for a Maia/Lc0 SE network."""

    def __init__(self, path: str):
        self.arr: NetArrays = parse_net(path)
        self._validate()

    def _validate(self):
        a = self.arr
        if a.conv1_w is None:
            raise RuntimeError("net has no input conv weights")
        if a.pol2_b is None or a.pol2_w is None:
            raise RuntimeError("net has no convolution policy head (unsupported)")

    def __call__(self, x: np.ndarray):
        """x: (112, 8, 8) float32 -> (policy_logits (1858,), value_logits (3,))"""
        a = self.arr
        x = np.maximum(conv2d(x, a.conv1_w) * a.conv1_scale[:, None, None]
                       + a.conv1_shift[:, None, None], 0.0)

        for (c1w, c1s, c1b, c2w, c2s, c2b,
             se_w1, se_b1, se_w2, se_b2) in a.res:
            h = np.maximum(conv2d(x, c1w) * c1s[:, None, None]
                           + c1b[:, None, None], 0.0)
            h = conv2d(h, c2w) * c2s[:, None, None] + c2b[:, None, None]
            pooled = h.mean(axis=(1, 2))
            fc1 = np.maximum(pooled @ se_w1.T + se_b1, 0.0)
            excited = fc1 @ se_w2.T + se_b2
            gamma_se, beta_se = excited[:len(excited) // 2], excited[len(excited) // 2:]
            x = np.maximum(1.0 / (1.0 + np.exp(-gamma_se))[:, None, None] * h
                           + beta_se[:, None, None] + x, 0.0)

        # policy
        h = np.maximum(conv2d(x, a.pol1_w) * a.pol1_scale[:, None, None]
                       + a.pol1_shift[:, None, None], 0.0)
        h = conv2d(h, a.pol2_w) + a.pol2_b[:, None, None]
        from .lc0_az_policy_map import make_map
        policy_logits = (h.reshape(80 * 64) @ make_map())

        # value
        v = np.maximum(conv2d_1x1(x, a.val_w) * a.val_scale[:, None, None]
                       + a.val_shift[:, None, None], 0.0)
        v = v.reshape(32 * 64)
        v = np.maximum(v @ a.ip1_w.T + a.ip1_b, 0.0)
        value_logits = v @ a.ip2_w.T + a.ip2_b

        return policy_logits, value_logits
