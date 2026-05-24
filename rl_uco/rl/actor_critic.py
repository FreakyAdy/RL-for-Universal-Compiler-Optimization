"""Autoregressive pass policy and critic for offline RL."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_uco.config import GRAPH_HIDDEN_DIM
from rl_uco.rl.encoder import StateEncoder


class PassCritic(nn.Module):
    def __init__(self, state_dim: int = GRAPH_HIDDEN_DIM, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class PassPolicy(nn.Module):
    """Autoregressive policy over pass IDs."""

    def __init__(
        self,
        state_dim: int,
        num_passes: int,
        pass_embed_dim: int = 64,
        hidden: int = 256,
        max_steps: int = 12,
    ):
        super().__init__()
        self.num_passes = num_passes
        self.max_steps = max_steps
        self.pass_embed = nn.Embedding(num_passes, pass_embed_dim)
        self.gru = nn.GRU(pass_embed_dim, hidden, batch_first=True)
        self.state_proj = nn.Linear(state_dim, hidden)
        self.head = nn.Linear(hidden, num_passes)

    def forward(
        self,
        state: torch.Tensor,
        pass_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits (B, T, num_passes) or (B, num_passes) for single step."""
        h0 = self.state_proj(state).unsqueeze(0)
        if pass_ids is None:
            logits = self.head(h0.squeeze(0))
            return logits
        emb = self.pass_embed(pass_ids)
        out, _ = self.gru(emb, h0)
        return self.head(out)

    def sample(
        self,
        state: torch.Tensor,
        max_steps: int | None = None,
        temperature: float = 1.0,
    ) -> list[int]:
        max_steps = max_steps or self.max_steps
        ids: list[int] = []
        if state.dim() == 1:
            state = state.unsqueeze(0)
        h0 = self.state_proj(state)  # (batch, hidden)
        hx = h0.unsqueeze(0)  # (num_layers=1, batch, hidden)
        for t in range(max_steps):
            if t == 0:
                logits = self.head(h0)
            else:
                emb = self.pass_embed(inp).unsqueeze(1)  # (batch, 1, emb)
                out, hx = self.gru(emb, hx)
                logits = self.head(out[:, -1, :])
            probs = F.softmax(logits / temperature, dim=-1)
            idx = int(torch.multinomial(probs, 1).item())
            if idx == 0:
                break
            ids.append(idx)
            inp = torch.tensor([idx], device=state.device, dtype=torch.long)
        return ids


class ActorCriticAgent(nn.Module):
    def __init__(
        self,
        graph_in_dim: int,
        num_passes: int,
        state_dim: int = GRAPH_HIDDEN_DIM,
        max_steps: int = 12,
    ):
        super().__init__()
        self.encoder = StateEncoder(graph_in_dim, state_dim)
        self.policy = PassPolicy(state_dim, num_passes, max_steps=max_steps)
        self.critic = PassCritic(state_dim)
        self.num_passes = num_passes

    def encode(
        self,
        data,
        isa_index: torch.Tensor,
        isa_attrs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encoder(data, isa_index, isa_attrs)

    def bc_loss(
        self,
        state: torch.Tensor,
        target_pass_ids: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.policy(state, target_pass_ids)
        return F.cross_entropy(
            logits.reshape(-1, self.num_passes),
            target_pass_ids.reshape(-1),
        )
