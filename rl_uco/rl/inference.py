"""Inference: propose pass sequences from checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import click
import torch

from rl_uco.config import Toolchain
from rl_uco.graph.parse import OPCODES
from rl_uco.graph.llvm_to_graph import llvm_to_pyg
from rl_uco.passes.executor import PassExecutor
from rl_uco.passes.registry import PassAction, load_registry
from rl_uco.passes.validator import PassSequenceValidator
from rl_uco.rl.actor_critic import ActorCriticAgent
from rl_uco.rl.encoder import ISAEmbedding


class InferenceEngine:
    def __init__(self, checkpoint: Path, device: str = "cpu"):
        self.registry = load_registry()
        self.validator = PassSequenceValidator(self.registry)
        self.executor = PassExecutor(self.registry, Toolchain.discover())
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        num_actions = ckpt.get("num_actions", self.registry.num_actions)
        self.agent = ActorCriticAgent(len(OPCODES), num_actions)
        self.agent.load_state_dict(ckpt["model"])
        self.agent.to(device)
        self.agent.eval()
        self.device = device

    def propose_sequence(self, ir_path: Path, isa: str = "x86_64_v3") -> list[PassAction]:
        data = llvm_to_pyg(ir_path).to(self.device)
        isa_idx = torch.tensor([ISAEmbedding.isa_to_index(isa)], device=self.device)
        with torch.no_grad():
            state = self.agent.encode(data, isa_idx)
            ids = self.agent.policy.sample(state.squeeze(0))
        sequence: list[PassAction] = []
        for pid in ids:
            action = self.registry.get_by_id(pid)
            if action and action.name != "STOP":
                sequence.append(
                    PassAction(action.pass_id, action.name, action.pipeline, action.kind),
                )
        return sequence

    def optimize_ir(
        self,
        ir_path: Path,
        output_ir: Path,
        isa: str = "x86_64_v3",
        fallback: str = "O3",
    ) -> bool:
        seq = self.propose_sequence(ir_path, isa=isa)
        vr = self.validator.validate(seq)
        if not vr.valid:
            seq = self.registry.baseline_sequence(fallback)
        result = self.executor.apply_passes(ir_path, seq, output_ir)
        return result.success


@click.command()
@click.option("--ir", "ir_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--checkpoint", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output", type=click.Path(path_type=Path), default=None)
@click.option("--isa", default="x86_64_v3")
@click.option("--json", "emit_json", is_flag=True)
def main(
    ir_path: Path,
    checkpoint: Path,
    output: Path | None,
    isa: str,
    emit_json: bool,
) -> None:
    engine = InferenceEngine(checkpoint)
    seq = engine.propose_sequence(ir_path, isa=isa)
    if emit_json:
        click.echo(json.dumps([a.to_dict() for a in seq], indent=2))
    else:
        names = [a.name for a in seq]
        click.echo(f"Proposed passes: {names}")
    if output:
        ok = engine.optimize_ir(ir_path, output, isa=isa)
        click.echo(f"Wrote {output}: {ok}")


if __name__ == "__main__":
    main()
