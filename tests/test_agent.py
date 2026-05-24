"""RL agent smoke tests."""

import torch

from rl_uco.graph.llvm_to_graph import llvm_to_pyg
from rl_uco.graph.parse import OPCODES
from rl_uco.rl.actor_critic import ActorCriticAgent
from rl_uco.rl.encoder import ISAEmbedding


def test_agent_forward(tmp_path):
    ir = tmp_path / "t.ll"
    ir.write_text(
        "define i32 @f(i32 %x) { %a = add i32 %x, 1 ret i32 %a }",
        encoding="utf-8",
    )
    data = llvm_to_pyg(ir)
    agent = ActorCriticAgent(len(OPCODES), 50)
    isa = torch.tensor([0])
    state = agent.encode(data, isa)
    assert state.shape[-1] > 0
    ids = agent.policy.sample(state.squeeze(0))
    assert isinstance(ids, list)
