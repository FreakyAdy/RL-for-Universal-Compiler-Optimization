"""Graph conversion tests."""

from pathlib import Path

import torch

from rl_uco.graph.llvm_to_graph import llvm_to_pyg
from rl_uco.graph.mlir_to_graph import mlir_to_pyg


FIXTURES = Path(__file__).parent / "fixtures"


def test_llvm_to_pyg_from_text(tmp_path):
    ir = tmp_path / "t.ll"
    ir.write_text(
        """
define i32 @foo(i32 %x) {
entry:
  %0 = add i32 %x, 1
  ret i32 %0
}
""",
        encoding="utf-8",
    )
    data = llvm_to_pyg(ir)
    assert data.num_nodes >= 1
    assert data.x.shape[1] > 0


def test_mlir_to_pyg():
    mlir = FIXTURES / "sample_kernel.mlir"
    if mlir.exists():
        data = mlir_to_pyg(mlir)
        assert data.num_nodes >= 1
