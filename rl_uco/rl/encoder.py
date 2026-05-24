"""GNN graph encoder + ISA embedding."""

from __future__ import annotations

import torch
import torch.nn as nn

from rl_uco.config import GRAPH_HIDDEN_DIM, GRAPH_NUM_LAYERS, ISA_EMBED_DIM, ISA_REGISTRY, NUM_ISAS

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GATConv, global_mean_pool

    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False
    Data = object  # type: ignore[misc, assignment]


class ISAEmbedding(nn.Module):
    def __init__(self, num_isas: int = NUM_ISAS, dim: int = ISA_EMBED_DIM):
        super().__init__()
        self.embed = nn.Embedding(num_isas, dim)
        self.attr_proj = nn.Linear(4, dim)

    def forward(self, isa_indices: torch.Tensor, attrs: torch.Tensor | None = None) -> torch.Tensor:
        e = self.embed(isa_indices)
        if attrs is not None:
            e = e + self.attr_proj(attrs)
        return e

    @staticmethod
    def isa_to_index(isa_key: str) -> int:
        info = ISA_REGISTRY.get(isa_key)
        return info.embed_index if info else 0


class GraphEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = GRAPH_HIDDEN_DIM,
        num_layers: int = GRAPH_NUM_LAYERS,
        out_dim: int = GRAPH_HIDDEN_DIM,
    ):
        super().__init__()
        self._use_pyg = _HAS_PYG
        if _HAS_PYG:
            self.convs = nn.ModuleList()
            self.convs.append(GATConv(in_dim, hidden_dim, heads=2, concat=False))
            for _ in range(num_layers - 1):
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=2, concat=False))
            self.out_proj = nn.Linear(hidden_dim, out_dim)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, data: Data) -> torch.Tensor:
        if not self._use_pyg:
            x = data.x
            if x.dim() > 2:
                x = x.mean(dim=0)
            if x.dim() == 2 and x.size(0) > 1:
                x = x.mean(dim=0, keepdim=True)
            return self.mlp(x.squeeze(0) if x.dim() == 2 and x.size(0) == 1 else x.float())
        x, edge_index, batch = data.x, data.edge_index, getattr(data, "batch", None)
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        g = global_mean_pool(x, batch)
        return self.out_proj(g)


class StateEncoder(nn.Module):
    """Graph + ISA -> state vector."""

    def __init__(self, graph_in_dim: int, state_dim: int = GRAPH_HIDDEN_DIM):
        super().__init__()
        self.graph_enc = GraphEncoder(graph_in_dim, out_dim=state_dim)
        self.isa_emb = ISAEmbedding()
        self.fuse = nn.Linear(state_dim + ISA_EMBED_DIM, state_dim)

    def forward(
        self,
        data: Data,
        isa_index: torch.Tensor,
        isa_attrs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        g = self.graph_enc(data)
        isa = self.isa_emb(isa_index, isa_attrs)
        if g.dim() == 1:
            g = g.unsqueeze(0)
        if isa.dim() == 1:
            isa = isa.unsqueeze(0)
        return self.fuse(torch.cat([g, isa], dim=-1))
