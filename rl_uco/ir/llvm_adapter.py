"""LLVM IR generation and manipulation."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from rl_uco.config import COMPILE_TIMEOUT_S, ISA_FLAGS, Toolchain


@dataclass
class IRArtifact:
    source_path: Path
    ir_path: Path
    function_name: str
    isa: str


class LLVMAdapter:
    def __init__(self, toolchain: Toolchain | None = None):
        self.toolchain = toolchain or Toolchain.discover()

    def compile_to_ir(
        self,
        source: Path,
        output_ir: Path,
        isa: str = "x86_64_v3",
        opt_level: str = "O0",
    ) -> bool:
        flags = ISA_FLAGS.get(isa, [])
        cmd = [
            self.toolchain.clang,
            f"-{opt_level}",
            "-emit-llvm",
            "-S",
            *flags,
            "-c",
            str(source),
            "-o",
            str(output_ir),
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
        return proc.returncode == 0 and output_ir.exists()

    def codegen(
        self,
        ir_path: Path,
        output_obj: Path,
        isa: str = "x86_64_v3",
    ) -> bool:
        flags = ISA_FLAGS.get(isa, [])
        cmd = [
            self.toolchain.clang,
            "-O0",
            *flags,
            "-c",
            str(ir_path),
            "-o",
            str(output_obj),
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
        return proc.returncode == 0 and output_obj.exists()

    def link_executable(
        self,
        objects: list[Path],
        output_bin: Path,
        extra_sources: list[Path] | None = None,
        isa: str = "x86_64_v3",
    ) -> bool:
        flags = ISA_FLAGS.get(isa, [])
        cmd = [
            self.toolchain.clang,
            *flags,
            *[str(o) for o in objects],
            *[str(s) for s in (extra_sources or [])],
            "-o",
            str(output_bin),
            "-lm",
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

    def emit_function_ir(
        self,
        source: Path,
        function_name: str,
        work_dir: Path,
        isa: str = "x86_64_v3",
    ) -> IRArtifact | None:
        work_dir.mkdir(parents=True, exist_ok=True)
        ir_path = work_dir / f"{function_name}.ll"
        if not self.compile_to_ir(source, ir_path, isa=isa):
            return None
        return IRArtifact(source, ir_path, function_name, isa)

    def count_instructions(self, ir_path: Path) -> int:
        """Count LLVM IR instructions (both `%var = op` and bare terminal forms)."""
        import re

        assign_pat = re.compile(r"^\s*%[\w.]+ = (\w+)")
        bare_ops = {"br", "ret", "store", "call", "switch", "unreachable", "invoke"}
        count = 0
        for line in ir_path.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.strip()
            if not s or s.startswith(";"):
                continue
            m = assign_pat.match(s)
            if m:
                count += 1
            elif s.split()[0] in bare_ops:
                count += 1
        return count
