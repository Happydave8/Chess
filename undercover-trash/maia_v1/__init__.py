"""Maia-v1 (Leela Chess Zero .pb.gz) support package.

This package loads the released CSSLab/maia-chess networks ("maia-1100" ...
"maia-1900", stored as Lc0 network files) into PyTorch so their move policy
can be sampled without needing an lc0 binary.

Files net_pb2.py, policy_index.py and lc0_az_policy_map.py are copied
verbatim from https://github.com/CSSLab/maia-chess (GPL-3.0) and are used to
parse the network file and map the 1858-dim policy output to UCI moves.
encoder.py and lc0net.py are original implementations that faithfully mirror
the reference pipeline (lc0's INPUT_CLASSICAL_112_PLANE encoder and the
SE-network forward graph in maia-chess's tfprocess.py).
"""
