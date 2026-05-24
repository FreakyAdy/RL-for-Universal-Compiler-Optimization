"""PyG-free graph representation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from rl_uco.graph.parse import OP_TO_IDX, OPCODES, parse_instructions


@dataclass
class SimpleGraphData:
    x: torch.Tensor
    edge_index: torch.Tensor
    num_nodes: int

    def to(self, device: torch.device | str) -> SimpleGraphData:
        return SimpleGraphData(
            self.x.to(device),
            self.edge_index.to(device),
            self.num_nodes,
        )


def llvm_to_simple(ir_path: Path) -> SimpleGraphData:
    text = ir_path.read_text(encoding="utf-8", errors="replace")
    instrs = parse_instructions(text)
    if not instrs:
        x = torch.zeros((1, len(OPCODES)), dtype=torch.float)
        return SimpleGraphData(x=x, edge_index=torch.zeros((2, 0), dtype=torch.long), num_nodes=1)
    hist = [0.0] * len(OPCODES)
    for _, op, _ in instrs:
        hist[OP_TO_IDX.get(op, OP_TO_IDX["other"])] += 1.0
    total = sum(hist) or 1.0
    hist = [h / total for h in hist]
    x = torch.tensor([hist], dtype=torch.float)
    return SimpleGraphData(x=x, edge_index=torch.zeros((2, 0), dtype=torch.long), num_nodes=1)
