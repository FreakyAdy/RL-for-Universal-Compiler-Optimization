"""MLIR lowering and CUDA path."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from rl_uco.config import COMPILE_TIMEOUT_S, Toolchain
from rl_uco.passes.registry import PassAction, PassRegistry, load_registry


@dataclass
class MLIRArtifact:
    mlir_path: Path
    kernel_name: str
    isa: str = "cuda_sm80"


class MLIRAdapter:
    """Apply MLIR passes and lower toward NVVM/CUDA."""

    def __init__(self, toolchain: Toolchain | None = None, registry: PassRegistry | None = None):
        self.toolchain = toolchain or Toolchain.discover()
        self.registry = registry or load_registry()

    def apply_passes(
        self,
        input_mlir: Path,
        sequence: list[PassAction],
        output_mlir: Path,
    ) -> bool:
        pipeline = self.registry.mlir_pipeline_string(sequence)
        if not pipeline:
            output_mlir.write_text(input_mlir.read_text(encoding="utf-8"))
            return True
        cmd = [
            self.toolchain.mlir_opt,
            str(input_mlir),
            f"-pass-pipeline={pipeline}",
            "-o",
            str(output_mlir),
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=COMPILE_TIMEOUT_S,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError):
            return False
        return proc.returncode == 0 and output_mlir.exists()

    def lower_to_ptx(self, mlir_path: Path, ptx_path: Path) -> bool:
        """Best-effort GPU lowering; requires full pipeline in practice."""
        cmd = [
            self.toolchain.mlir_opt,
            str(mlir_path),
            "--gpu-module-to-binary=format=isa",
            "-o",
            str(ptx_path),
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=COMPILE_TIMEOUT_S,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError):
            return False
        return proc.returncode == 0

    def compile_cuda_kernel(
        self,
        cuda_source: Path,
        output_bin: Path,
        arch: str = "sm_80",
    ) -> bool:
        cmd = [
            "nvcc",
            str(cuda_source),
            "-o",
            str(output_bin),
            f"-arch={arch}",
            "-O0",
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=COMPILE_TIMEOUT_S,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError):
            return False
        return proc.returncode == 0 and output_bin.exists()
