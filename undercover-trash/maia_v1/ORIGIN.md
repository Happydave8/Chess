# Vendored files and attribution

The following files were copied verbatim from
https://github.com/CSSLab/maia-chess (GPL-3.0 licensed, see `LICENSE`):

| File | Upstream path | Purpose |
|------|---------------|---------|
| `net_pb2.py` | `move_prediction/maia_chess_backend/maia/proto/net_pb2.py` | Protobuf classes for the Lc0 network file format |
| `policy_index.py` | `move_prediction/maia_chess_backend/maia/policy_index.py` | The 1858-entry policy move list (white-perspective UCI) |
| `lc0_az_policy_map.py` | `move_prediction/maia_chess_backend/maia/lc0_az_policy_map.py` | Fixed (5120 x 1858) matrix mapping the convolution policy head to the 1858 move logits |

`encoder.py` and `lc0net.py` in this directory are original code written for
this project. They are faithful re-implementations of:

- lc0 `src/neural/encoder.cc` (INPUT_CLASSICAL_112_PLANE encoding), and
- the Maia training graph in `move_prediction/maia_chess_backend/maia/tfprocess.py`
  (SE network + convolution policy head + WDL value head),
  plus lc0's weight semantics (`src/neural/network_legacy.cc`: batch norm is
  `gamma / sqrt(var + 1e-5)`, SE is `sigmoid(gamma) * x + beta`).

Model weights (`maia-1100.pb.gz` ... `maia-1900.pb.gz`) are released by the
Maia team under the GPL-3.0 and are downloadable from
https://github.com/CSSLab/maia-chess/releases/tag/v1.0 or the `maia_weights/`
folder of the repository. See `scripts/download_weights.sh`.
