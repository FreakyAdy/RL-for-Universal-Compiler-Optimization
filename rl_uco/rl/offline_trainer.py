"""Offline IQL training on Parquet datasets."""

from __future__ import annotations

import json
from pathlib import Path

import click
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from rl_uco.config import CHECKPOINT_DIR, GRAPH_DIR
from rl_uco.data.schema import load_parquet
from rl_uco.graph.parse import OPCODES
from rl_uco.passes.registry import load_registry
from rl_uco.rl.actor_critic import ActorCriticAgent, PassCritic
from rl_uco.rl.encoder import ISAEmbedding


class OfflinePassDataset(Dataset):
    def __init__(self, parquet_path: Path, graph_dir: Path):
        self.df = load_parquet(parquet_path)
        self.df = self.df[self.df["correct"] == True]  # noqa: E712
        self.graph_dir = graph_dir
        self.registry = load_registry()
        self.pass_id_to_idx = {0: 0}
        for pid in list(self.registry.passes.keys()) + list(self.registry.adaptors.keys()):
            self.pass_id_to_idx[pid] = len(self.pass_id_to_idx)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        gpath = self.graph_dir / row["graph_path"]
        if not gpath.exists():
            gpath = GRAPH_DIR / row["graph_path"]
        data = torch.load(gpath, weights_only=False)
        seq = row["pass_sequence"]
        if isinstance(seq, str):
            seq = json.loads(seq)
        pass_ids = [self.pass_id_to_idx.get(int(s.get("pass_id", 0)), 0) for s in seq]
        if not pass_ids:
            pass_ids = [0]
        isa_idx = ISAEmbedding.isa_to_index(row["isa"])
        return {
            "data": data,
            "pass_ids": torch.tensor(pass_ids, dtype=torch.long),
            "reward": torch.tensor(row["reward"], dtype=torch.float),
            "isa_index": torch.tensor(isa_idx, dtype=torch.long),
        }


def collate_batch(batch):
    return {
        "data": batch[0]["data"],
        "pass_ids": batch[0]["pass_ids"],
        "reward": torch.stack([b["reward"] for b in batch]),
        "isa_index": torch.stack([b["isa_index"] for b in batch]),
    }


class IQLTrainer:
    """Implicit Q-Learning style offline actor-critic."""

    def __init__(
        self,
        agent: ActorCriticAgent,
        expectile: float = 0.7,
        beta: float = 3.0,
        lr: float = 3e-4,
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        self.agent = agent.to(device)
        state_dim = agent.critic.net[0].in_features
        self.target_critic = PassCritic(state_dim).to(device)
        self.target_critic.load_state_dict(agent.critic.state_dict())
        self.opt = torch.optim.Adam(agent.parameters(), lr=lr)
        self.expectile = expectile
        self.beta = beta
        self.gamma = gamma
        self.device = device

    def expectile_loss(self, diff: torch.Tensor) -> torch.Tensor:
        w = torch.where(diff > 0, self.expectile, 1 - self.expectile)
        return (w * diff.pow(2)).mean()

    def train_step(self, batch: dict) -> dict[str, float]:
        data = batch["data"].to(self.device)
        isa_index = batch["isa_index"].to(self.device)
        rewards = batch["reward"].to(self.device)
        pass_ids = batch["pass_ids"].to(self.device)

        state = self.agent.encode(data, isa_index)
        v = self.agent.critic(state)
        with torch.no_grad():
            v_tgt = self.target_critic(state)
        adv = rewards - v_tgt
        v_loss = self.expectile_loss(rewards - v)

        q_tgt = rewards
        q_loss = F.mse_loss(v, q_tgt)

        bc = self.agent.bc_loss(state, pass_ids.unsqueeze(0))
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=100.0)
        actor_loss = (exp_adv * bc).mean()

        loss = v_loss + q_loss + actor_loss
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return {
            "loss": float(loss.item()),
            "v_loss": float(v_loss.item()),
            "bc_loss": float(bc.item()),
            "actor_loss": float(actor_loss.item()),
        }


def train_bc_baseline(
    agent: ActorCriticAgent,
    loader: DataLoader,
    epochs: int = 5,
    device: str = "cpu",
) -> None:
    opt = torch.optim.Adam(agent.parameters(), lr=3e-4)
    agent.to(device)
    for _ in range(epochs):
        for batch in loader:
            data = batch["data"].to(device)
            isa_index = batch["isa_index"].to(device)
            pass_ids = batch["pass_ids"].to(device)
            state = agent.encode(data, isa_index)
            loss = agent.bc_loss(state, pass_ids.unsqueeze(0))
            opt.zero_grad()
            loss.backward()
            opt.step()


@click.command()
@click.option("--dataset", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output", type=click.Path(path_type=Path), default=CHECKPOINT_DIR)
@click.option("--epochs", default=20)
@click.option("--batch-size", default=8)
@click.option("--bc-only", is_flag=True, help="Behavior cloning warmup only")
@click.option("--device", default="cpu")
def main(
    dataset: Path,
    output: Path,
    epochs: int,
    batch_size: int,
    bc_only: bool,
    device: str,
) -> None:
    graph_dir = dataset.parent / "graphs"
    if not graph_dir.exists():
        graph_dir = GRAPH_DIR
    ds = OfflinePassDataset(dataset, graph_dir)
    if len(ds) == 0:
        click.echo("No training rows; collect dataset first.")
        return
    loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_batch)
    registry = load_registry()
    num_actions = registry.num_actions
    agent = ActorCriticAgent(len(OPCODES), num_actions)
    if bc_only:
        train_bc_baseline(agent, loader, epochs=epochs, device=device)
    else:
        trainer = IQLTrainer(agent, device=device)
        for ep in range(epochs):
            metrics = trainer.train_step(next(iter(loader)))
            click.echo(f"epoch {ep}: {metrics}")
    output.mkdir(parents=True, exist_ok=True)
    ckpt = output / "best.pt"
    torch.save({"model": agent.state_dict(), "num_actions": num_actions}, ckpt)
    click.echo(f"Saved {ckpt}")


if __name__ == "__main__":
    main()
