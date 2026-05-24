"""Convert MLIR text to graph tensors."""

from __future__ import annotations

import re
from pathlib import Path

import torch

try:
    from torch_geometric.data import Data as PyGData

    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False
    PyGData = None

MLIR_OPS = [
    "func", "return", "arith", "linalg", "scf", "cf", "gpu", "memref",
    "tensor", "llvm", "other",
]
OP_TO_IDX = {op: i for i, op in enumerate(MLIR_OPS)}


def mlir_to_pyg(mlir_path: Path):
    text = mlir_path.read_text(encoding="utf-8", errors="replace")
    features: list[list[float]] = []
    edges: list[list[int]] = []
    op_pat = re.compile(r"(\w+)\.(\w+)")

    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("//"):
            continue
        m = op_pat.search(s)
        if not m:
            continue
        dialect = m.group(1)
        key = dialect if dialect in OP_TO_IDX else "other"
        feat = [0.0] * len(MLIR_OPS)
        feat[OP_TO_IDX[key]] = 1.0
        idx = len(features)
        features.append(feat)
        if len(features) > 1:
            edges.append([idx - 1, idx])

    if not features:
        x = torch.zeros((1, len(MLIR_OPS)), dtype=torch.float)
        num_nodes = 1
    else:
        x = torch.tensor(features, dtype=torch.float)
        num_nodes = x.size(0)

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    if _HAS_PYG and PyGData is not None:
        return PyGData(x=x, edge_index=edge_index, num_nodes=num_nodes)
    from rl_uco.graph.fallback import SimpleGraphData

    return SimpleGraphData(x=x, edge_index=edge_index, num_nodes=num_nodes)
