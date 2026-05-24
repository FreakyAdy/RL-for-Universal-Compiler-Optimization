"""Execute LLVM/MLIR passes and correctness checks."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from rl_uco.config import COMPILE_TIMEOUT_S, Toolchain
from rl_uco.passes.registry import PassAction, PassRegistry
from rl_uco.passes.validator import PassSequenceValidator


@dataclass
class CorrectnessResult:
    correct: bool
    reason: str = ""
    optimized_ir: Path | None = None


@dataclass
class CompileResult:
    success: bool
    ir_path: Path | None = None
    stderr: str = ""


class PassExecutor:
    def __init__(
        self,
        registry: PassRegistry | None = None,
        toolchain: Toolchain | None = None,
    ):
        self.registry = registry or __import__(
            "rl_uco.passes.registry", fromlist=["load_registry"]
        ).load_registry()
        self.toolchain = toolchain or Toolchain.discover()
        self.validator = PassSequenceValidator(self.registry)

    def apply_passes(
        self,
        input_ir: Path,
        sequence: list[PassAction],
        output_ir: Path,
        ir_kind: str = "llvm",
    ) -> CompileResult:
        vr = self.validator.validate(sequence, ir_kind=ir_kind)
        if not vr.valid:
            return CompileResult(False, stderr=vr.reason)

        pipeline = vr.pipeline
        if not pipeline:
            # No-op: copy input
            output_ir.write_text(input_ir.read_text(encoding="utf-8"))
            return CompileResult(True, ir_path=output_ir)

        if ir_kind == "mlir":
            cmd = [self.toolchain.mlir_opt, input_ir, f"-pass-pipeline={pipeline}", "-o", str(output_ir)]
        else:
            cmd = [self.toolchain.opt, input_ir, f"-passes={pipeline}", "-S", "-o", str(output_ir)]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=COMPILE_TIMEOUT_S,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return CompileResult(False, stderr="opt timeout")
        except OSError as e:
            return CompileResult(False, stderr=str(e))

        if proc.returncode != 0:
            return CompileResult(False, stderr=proc.stderr or proc.stdout)

        if not output_ir.exists():
            return CompileResult(False, stderr="opt produced no output")

        return CompileResult(True, ir_path=output_ir)

    def llvm_diff(self, reference_ir: Path, optimized_ir: Path) -> CorrectnessResult:
        try:
            proc = subprocess.run(
                [self.toolchain.llvm_diff, str(reference_ir), str(optimized_ir)],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            return CorrectnessResult(False, reason=str(e))

        if proc.returncode == 0:
            return CorrectnessResult(True, optimized_ir=optimized_ir)
        # llvm-diff returns 1 when different — not always incorrect (optimization)
        # For strict mode we require identical semantics via run check instead
        return CorrectnessResult(
            proc.returncode == 0,
            reason=proc.stderr or "llvm-diff: IR differs",
            optimized_ir=optimized_ir,
        )

    def check_correctness(
        self,
        reference_ir: Path,
        sequence: list[PassAction],
        ir_kind: str = "llvm",
    ) -> CorrectnessResult:
        with tempfile.TemporaryDirectory(prefix="rl_uco_") as tmp:
            tmp_path = Path(tmp)
            opt_ir = tmp_path / "optimized.ll"
            cr = self.apply_passes(reference_ir, sequence, opt_ir, ir_kind=ir_kind)
            if not cr.success:
                return CorrectnessResult(False, reason=cr.stderr)
            return self.llvm_diff(reference_ir, opt_ir)
