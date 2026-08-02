"""Pure-Python reader for Leela Chess Zero network files (.pb.gz).

A minimal protobuf wire-format reader for exactly the messages used by the
released Maia network files (see `net.proto` in lc0 / CSSLab/maia-chess).
This lets the engine load Maia weights with **no protobuf dependency at all**
(important on platforms like Termux/Android where building the protobuf C
extension is painful).

Supported subset of the schema (field numbers from lc0's net.proto):

    Net { magic=1, license=2, min_version=3, format=4, weights=10 }
    Format { weights_encoding=1, network_format=2 }
    NetworkFormat { input=1, output=2, network=3, policy=4, value=5 }
    Weights {
      input=1 (ConvBlock), residual=2 (repeated Residual),
      policy=3 (ConvBlock), ip_pol_w=4, ip_pol_b=5,
      value=6 (ConvBlock), ip1_val_w=7, ip1_val_b=8,
      ip2_val_w=9, ip2_val_b=10, policy1=11 (ConvBlock)
    }
    ConvBlock { weights=1, biases=2, bn_means=3, bn_stddivs=4,
                bn_gammas=5, bn_betas=6 }
    SEunit { w1=1, b1=2, w2=3, b2=4 }
    Residual { conv1=1, conv2=2, se=3 }
    Layer { min_val=1 (float), max_val=2 (float), params=3 (bytes) }

Wire types used: 0 (varint), 2 (length-delimited), 5 (32-bit float).
"""

from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass, field

import numpy as np

# --- wire format -----------------------------------------------------------

WIRE_VARINT = 0
WIRE_I64 = 1
WIRE_LEN = 2
WIRE_I32 = 5


class PbError(RuntimeError):
    pass


def _read_varint(data: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        if pos >= len(data):
            raise PbError("truncated varint")
        b = data[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7
        if shift > 63:
            raise PbError("varint too long")


def _fields(data: bytes):
    """Yield (field_number, wire_type, value) for a message body."""
    pos = 0
    while pos < len(data):
        key, pos = _read_varint(data, pos)
        fnum = key >> 3
        wire = key & 7
        if fnum == 0:
            raise PbError("invalid field number 0")
        if wire == WIRE_VARINT:
            val, pos = _read_varint(data, pos)
        elif wire == WIRE_I64:
            if pos + 8 > len(data):
                raise PbError("truncated 64-bit value")
            val = data[pos:pos + 8]
            pos += 8
        elif wire == WIRE_LEN:
            ln, pos = _read_varint(data, pos)
            if pos + ln > len(data):
                raise PbError("truncated length-delimited value")
            val = data[pos:pos + ln]
            pos += ln
        elif wire == WIRE_I32:
            if pos + 4 > len(data):
                raise PbError("truncated 32-bit value")
            val = data[pos:pos + 4]
            pos += 4
        else:
            raise PbError(f"unsupported wire type {wire}")
        yield fnum, wire, val


def _take(data: bytes, wanted: int):
    """Return the value of the first field with number `wanted`, or None."""
    for fnum, _, val in _fields(data):
        if fnum == wanted:
            return val
    return None


def _take_float(data: bytes, wanted: int) -> float:
    raw = _take(data, wanted)
    if raw is None:
        return 0.0
    if isinstance(raw, bytes) and len(raw) == 4:
        return struct.unpack("<f", raw)[0]
    raise PbError(f"field {wanted} is not a 32-bit float")


def _take_bytes(data: bytes, wanted: int) -> bytes:
    raw = _take(data, wanted)
    if raw is None:
        return b""
    if isinstance(raw, bytes):
        return raw
    raise PbError(f"field {wanted} is not length-delimited")


# --- schema-specific extraction ---------------------------------------------

def _layer(data: bytes):
    """Layer { min_val=1, max_val=2, params=3 } -> np.ndarray float32."""
    params = _take_bytes(data, 3)
    if not params:
        return None
    if len(params) % 2:
        raise PbError("params length is not a multiple of 2")
    a = np.frombuffer(params, dtype="<u2").astype(np.float32) / 0xFFFF
    lo = _take_float(data, 1)
    hi = _take_float(data, 2)
    return a * (hi - lo) + lo


def _conv_block(data: bytes):
    """ConvBlock -> dict of np.ndarray (missing arrays -> None)."""
    return {
        "weights": _layer(_take(data, 1)) if isinstance(_take(data, 1), bytes) else None,
        "biases": _layer(_take(data, 2)) if isinstance(_take(data, 2), bytes) else None,
        "bn_means": _layer(_take(data, 3)) if isinstance(_take(data, 3), bytes) else None,
        "bn_stddivs": _layer(_take(data, 4)) if isinstance(_take(data, 4), bytes) else None,
        "bn_gammas": _layer(_take(data, 5)) if isinstance(_take(data, 5), bytes) else None,
        "bn_betas": _layer(_take(data, 6)) if isinstance(_take(data, 6), bytes) else None,
    }


def _se_unit(data: bytes):
    return {
        "w1": _layer(_take(data, 1)) if isinstance(_take(data, 1), bytes) else None,
        "b1": _layer(_take(data, 2)) if isinstance(_take(data, 2), bytes) else None,
        "w2": _layer(_take(data, 3)) if isinstance(_take(data, 3), bytes) else None,
        "b2": _layer(_take(data, 4)) if isinstance(_take(data, 4), bytes) else None,
    }


# --- public API --------------------------------------------------------------

@dataclass
class NetArrays:
    """All weights of a Maia/Lc0 SE network as flat numpy arrays.

    Shapes match the lc0 canonical layout: conv kernels are [out, in, y, x],
    fully-connected weights are [out, in].
    """
    network_format: int = 0
    input_format: int = 0
    policy_format: int = 0
    value_format: int = 0
    weights_encoding: int = 0

    conv1_w: np.ndarray = None          # (C, 112, 3, 3)
    conv1_scale: np.ndarray = None      # (C,)
    conv1_shift: np.ndarray = None      # (C,)
    res: list = field(default_factory=list)  # (conv1, conv2, se) tuples
    pol1_w: np.ndarray = None           # (C, C, 3, 3)
    pol1_scale: np.ndarray = None
    pol1_shift: np.ndarray = None
    pol2_w: np.ndarray = None           # (80, C, 3, 3)
    pol2_b: np.ndarray = None           # (80,)
    val_w: np.ndarray = None            # (32, C, 1, 1)
    val_scale: np.ndarray = None
    val_shift: np.ndarray = None
    ip1_w: np.ndarray = None            # (128, 2048)
    ip1_b: np.ndarray = None
    ip2_w: np.ndarray = None            # (3, 128)
    ip2_b: np.ndarray = None


def _bn_from_block(block) -> tuple[np.ndarray, np.ndarray]:
    """Return (scale, shift) for the block's batch norm."""
    eps = 1e-5
    gamma = block["bn_gammas"]
    beta = block["bn_betas"]
    mean = block["bn_means"]
    var = block["bn_stddivs"]
    if gamma is None:
        gamma = np.ones_like(mean)
    scale = gamma / np.sqrt(var + eps)
    shift = beta - mean * scale
    return scale.astype(np.float32), shift.astype(np.float32)


def parse_net(path: str) -> NetArrays:
    """Parse a .pb.gz Lc0 network file into NetArrays (no protobuf needed)."""
    with gzip.open(path, "rb") as f:
        data = f.read()

    net = _take(data, 10)          # Weights
    if not isinstance(net, bytes):
        raise PbError("no weights message in net file")
    fmt = _take(data, 4)           # Format
    if isinstance(fmt, bytes):
        nf = _take(fmt, 2)         # NetworkFormat
    else:
        nf = None

    out = NetArrays()
    if isinstance(nf, bytes):
        out.network_format = _take(nf, 3) or 0
        out.input_format = _take(nf, 1) or 0
        out.policy_format = _take(nf, 4) or 0
        out.value_format = _take(nf, 5) or 0
    if isinstance(fmt, bytes):
        out.weights_encoding = _take(fmt, 1) or 0

    # --- input conv --------------------------------------------------------
    inp = _take(net, 1)
    if isinstance(inp, bytes):
        block = _conv_block(inp)
        out.conv1_w = block["weights"]
        out.conv1_scale, out.conv1_shift = _bn_from_block(block)

    # --- residual tower ------------------------------------------------------
    for raw in (val for fnum, _, val in _fields(net) if fnum == 2):
        conv1 = _conv_block(_take(raw, 1)) if isinstance(_take(raw, 1), bytes) else {}
        conv2 = _conv_block(_take(raw, 2)) if isinstance(_take(raw, 2), bytes) else {}
        se = _se_unit(_take(raw, 3)) if isinstance(_take(raw, 3), bytes) else {}
        c1s, c1b = _bn_from_block(conv1)
        c2s, c2b = _bn_from_block(conv2)
        out.res.append((
            conv1.get("weights"), c1s, c1b,
            conv2.get("weights"), c2s, c2b,
            se.get("w1"), se.get("b1"), se.get("w2"), se.get("b2"),
        ))

    # --- policy head ---------------------------------------------------------
    pol1 = _take(net, 11)
    if isinstance(pol1, bytes):
        block = _conv_block(pol1)
        out.pol1_w = block["weights"]
        out.pol1_scale, out.pol1_shift = _bn_from_block(block)
    pol = _take(net, 3)
    if isinstance(pol, bytes):
        block = _conv_block(pol)
        out.pol2_w = block["weights"]
        out.pol2_b = block.get("biases")

    # --- value head ------------------------------------------------------------
    val = _take(net, 6)
    if isinstance(val, bytes):
        block = _conv_block(val)
        out.val_w = block["weights"]
        out.val_scale, out.val_shift = _bn_from_block(block)
    out.ip1_w = _layer(_take(net, 7)) if isinstance(_take(net, 7), bytes) else None
    out.ip1_b = _layer(_take(net, 8)) if isinstance(_take(net, 8), bytes) else None
    out.ip2_w = _layer(_take(net, 9)) if isinstance(_take(net, 9), bytes) else None
    out.ip2_b = _layer(_take(net, 10)) if isinstance(_take(net, 10), bytes) else None

    _apply_shapes(out)
    return out


def _apply_shapes(a: NetArrays):
    """Reshape flat arrays into the network's canonical tensor layout.

    The wire format stores every tensor flattened; the architecture is
    recovered from the released Maia nets (SE-ResNet: 64 filters, 6 blocks,
    policy conv 3x3->80, value 1x1->32 -> 128 -> 3).
    """
    def shaped(arr, shape):
        if arr is None:
            return None
        n = arr.size
        want = 1
        for s in shape:
            want *= s
        if n != want:
            raise PbError(f"tensor of {n} elements does not fit shape {shape}")
        return arr.reshape(shape)

    if a.conv1_w is None:
        return
    C = a.conv1_w.size // (112 * 3 * 3)
    if C * 112 * 9 != a.conv1_w.size:
        raise PbError("input conv has unexpected size")
    a.conv1_w = shaped(a.conv1_w, (C, 112, 3, 3))

    shaped_res = []
    for (c1w, c1s, c1b, c2w, c2s, c2b,
         se_w1, se_b1, se_w2, se_b2) in a.res:
        shaped_res.append((
            shaped(c1w, (C, C, 3, 3)), c1s, c1b,
            shaped(c2w, (C, C, 3, 3)), c2s, c2b,
            shaped(se_w1, (C // 8, C)), se_b1,
            shaped(se_w2, (2 * C, C // 8)), se_b2,
        ))
    a.res = shaped_res

    a.pol1_w = shaped(a.pol1_w, (C, C, 3, 3))
    if a.pol2_w is not None:
        P = a.pol2_w.size // (C * 9)
        a.pol2_w = shaped(a.pol2_w, (P, C, 3, 3))
        a.pol2_b = shaped(a.pol2_b, (P,))

    if a.val_w is not None:
        V = a.val_w.size // C          # value head channels (32 for Maia)
        a.val_w = shaped(a.val_w, (V, C, 1, 1))
    if a.ip1_w is not None:
        H = a.ip1_w.size // (V * 64)   # 128 for Maia
        a.ip1_w = shaped(a.ip1_w, (H, V * 64))
        a.ip1_b = shaped(a.ip1_b, (H,))
    a.ip2_w = shaped(a.ip2_w, (3, 128))
    a.ip2_b = shaped(a.ip2_b, (3,))
