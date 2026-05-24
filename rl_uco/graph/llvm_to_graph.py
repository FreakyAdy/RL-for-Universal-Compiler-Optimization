"""Convert LLVM IR text to graph tensors."""

from __future__ import annotations

from pathlib import Path

import torch

from rl_uco.graph.parse import OP_TO_IDX, OPCODES, parse_instructions

try:
    from torch_geometric.data import Data as PyGData

    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False
    PyGData = None


def _build_graph(instrs: list[tuple[str, str, list[str]]]):
    if not instrs:
        x = torch.zeros((1, len(OPCODES)), dtype=torch.float)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        num_nodes = 1
    else:
        var_to_idx: dict[str, int] = {}
        features: list[list[float]] = []
        edges: list[list[int]] = []
        for var, op, uses in instrs:
            if var and var not in var_to_idx:
                var_to_idx[var] = len(var_to_idx)
            idx = len(features)
            feat = [0.0] * len(OPCODES)
            feat[OP_TO_IDX.get(op, OP_TO_IDX["other"])] = 1.0
            features.append(feat)
            if var:
                var_to_idx[var] = idx
            for u in uses:
                key = f"%{u}" if not u.startswith("%") else u
                if key in var_to_idx:
                    edges.append([var_to_idx[key], idx])
        x = torch.tensor(features, dtype=torch.float)
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        num_nodes = x.size(0)

    if _HAS_PYG and PyGData is not None:
        return PyGData(x=x, edge_index=edge_index, num_nodes=num_nodes)
    from rl_uco.graph.fallback import SimpleGraphData

    return SimpleGraphData(x=x, edge_index=edge_index, num_nodes=num_nodes)


def llvm_to_pyg(ir_path: Path):
    text = ir_path.read_text(encoding="utf-8", errors="replace")
    return _build_graph(parse_instructions(text))


def save_graph(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)


def llvm_to_graph(ir_path: Path):
    return llvm_to_pyg(ir_path)
