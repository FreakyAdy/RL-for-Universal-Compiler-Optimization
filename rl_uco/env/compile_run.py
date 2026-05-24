"""Compile-run environment: IR -> passes -> binary -> profile."""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from rl_uco.config import RUN_ITERATIONS, Toolchain
from rl_uco.env.reward import compute_reward
from rl_uco.hardware import get_profiler
from rl_uco.ir.llvm_adapter import LLVMAdapter
from rl_uco.passes.executor import PassExecutor
from rl_uco.passes.registry import PassAction, PassRegistry, load_registry


HARNESS_C = """
#include <stdint.h>
extern int {fn}(void);
int main(void) {{
    volatile int acc = 0;
    for (int i = 0; i < {iters}; i++) {{
        acc ^= {fn}();
    }}
    return acc & 1;
}}
"""


@dataclass
class CompileRunResult:
    success: bool
    correct: bool
    reward: float
    wall_time_ns: float = 0.0
    energy_j: float = 0.0
    baseline_wall_time_ns: float = 0.0
    baseline_energy_j: float = 0.0
    pass_sequence: list[dict] = field(default_factory=list)
    logs: str = ""


class CompileRunEnv:
    def __init__(
        self,
        registry: PassRegistry | None = None,
        toolchain: Toolchain | None = None,
    ):
        self.registry = registry or load_registry()
        self.toolchain = toolchain or Toolchain.discover()
        self.llvm = LLVMAdapter(self.toolchain)
        self.executor = PassExecutor(self.registry, self.toolchain)
        self._baseline_cache: dict[tuple[str, str], tuple[float, float]] = {}

    def _write_harness(self, work_dir: Path, function_name: str) -> Path:
        harness = work_dir / "harness.c"
        harness.write_text(
            HARNESS_C.format(fn=function_name, iters=RUN_ITERATIONS),
            encoding="utf-8",
        )
        return harness

    def _run_output_check(
        self,
        candidate: Path,
        reference: Path,
        isa: str,
    ) -> bool:
        try:
            rc = subprocess.run(
                [str(candidate)],
                capture_output=True,
                timeout=30,
                check=False,
            )
            rr = subprocess.run(
                [str(reference)],
                capture_output=True,
                timeout=30,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError):
            return False
        return rc.returncode == rr.returncode

    def _build_binary(
        self,
        ir_path: Path,
        work_dir: Path,
        function_name: str,
        isa: str,
        label: str,
    ) -> Path | None:
        obj = work_dir / f"{label}.o"
        if not self.llvm.codegen(ir_path, obj, isa=isa):
            return None
        harness = self._write_harness(work_dir, function_name)
        bin_path = work_dir / label
        if not self.llvm.link_executable([obj], bin_path, extra_sources=[harness], isa=isa):
            return None
        return bin_path

    def measure_baseline(
        self,
        ir_path: Path,
        function_name: str,
        isa: str,
    ) -> tuple[float, float]:
        key = (str(ir_path), isa)
        if key in self._baseline_cache:
            return self._baseline_cache[key]
        with tempfile.TemporaryDirectory(prefix="rl_uco_base_") as tmp:
            work = Path(tmp)
            bin_path = self._build_binary(ir_path, work, function_name, isa, "baseline")
            if bin_path is None:
                return float("inf"), 0.0
            prof = get_profiler(isa).profile(bin_path)
            self._baseline_cache[key] = (prof.wall_time_ns, prof.energy_j)
            return self._baseline_cache[key]

    def run(
        self,
        ir_path: Path,
        function_name: str,
        pass_sequence: list[PassAction],
        isa: str = "x86_64_v3",
        skip_output_check: bool = False,
    ) -> CompileRunResult:
        seq_dicts = [a.to_dict() for a in pass_sequence]
        base_time, base_energy = self.measure_baseline(ir_path, function_name, isa)

        with tempfile.TemporaryDirectory(prefix="rl_uco_run_") as tmp:
            work = Path(tmp)
            opt_ir = work / "optimized.ll"
            cr = self.executor.apply_passes(ir_path, pass_sequence, opt_ir)
            if not cr.success:
                return CompileRunResult(
                    False,
                    False,
                    compute_reward(0, 0, base_time, base_energy, correct=False),
                    logs=cr.stderr,
                    pass_sequence=seq_dicts,
                )

            ref_bin = self._build_binary(ir_path, work, function_name, isa, "ref")
            cand_bin = self._build_binary(opt_ir, work, function_name, isa, "cand")
            if cand_bin is None:
                return CompileRunResult(
                    False,
                    False,
                    compute_reward(0, 0, base_time, base_energy, correct=False),
                    logs="link failed",
                    pass_sequence=seq_dicts,
                )

            correct = True
            if not skip_output_check and ref_bin:
                correct = self._run_output_check(cand_bin, ref_bin, isa)

            prof = get_profiler(isa).profile(cand_bin)
            reward = compute_reward(
                prof.wall_time_ns,
                prof.energy_j,
                base_time,
                base_energy,
                correct=correct,
            )
            return CompileRunResult(
                success=True,
                correct=correct,
                reward=reward,
                wall_time_ns=prof.wall_time_ns,
                energy_j=prof.energy_j,
                baseline_wall_time_ns=base_time,
                baseline_energy_j=base_energy,
                pass_sequence=seq_dicts,
            )

    def bootstrap_policies(self) -> dict[str, float]:
        return {
            "O3": 0.07,
            "Oz": 0.07,
            "Os": 0.06,
            "random": 0.40,
            "mutate": 0.20,
            "greedy": 0.20,
        }

    def sample_pass_sequence(
        self,
        policy_tag: str,
        seed_sequence: list[PassAction] | None = None,
        rng=None,
    ) -> list[PassAction]:
        import random

        rng = rng or random.Random()
        if policy_tag in ("O3", "Oz", "Os"):
            return self.registry.baseline_sequence(policy_tag)
        if policy_tag == "random":
            return self.registry.random_sequence(rng=rng)
        if policy_tag == "mutate" and seed_sequence:
            return self.registry.mutate_sequence(seed_sequence, rng=rng)
        if policy_tag == "mutate":
            return self.registry.random_sequence(rng=rng)
        if policy_tag == "greedy":
            return self.registry.random_sequence(length=1, rng=rng)
        return self.registry.random_sequence(rng=rng)
