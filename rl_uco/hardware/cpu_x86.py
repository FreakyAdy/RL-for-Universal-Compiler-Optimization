"""x86-64 profiling with perf and RAPL."""

from __future__ import annotations

import platform
import statistics
import subprocess
import time
from pathlib import Path

from rl_uco.config import PROFILE_RUNS, RUN_TIMEOUT_S
from rl_uco.hardware.base import HardwareProfile, ProfileResult


def _read_rapl_energy_j() -> float | None:
    """Read package energy in joules from RAPL sysfs (Linux)."""
    base = Path("/sys/class/powercap/intel-rapl:0")
    if not base.exists():
        base = Path("/sys/class/powercap/intel-rapl/intel-rapl:0")
    energy_path = base / "energy_uj"
    if not energy_path.exists():
        return None
    try:
        uj = int(energy_path.read_text().strip())
        return uj / 1e6
    except (OSError, ValueError):
        return None


class X86Profiler(HardwareProfile):
    @property
    def isa_key(self) -> str:
        return "x86_64_v3"

    def _run_once(self, binary: Path, args: list[str]) -> tuple[float, float, int]:
        e0 = _read_rapl_energy_j()
        t0 = time.perf_counter_ns()
        try:
            proc = subprocess.run(
                [str(binary), *args],
                capture_output=True,
                timeout=RUN_TIMEOUT_S,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return float("inf"), 0.0, -1
        t1 = time.perf_counter_ns()
        e1 = _read_rapl_energy_j()
        wall = t1 - t0
        energy = 0.0
        if e0 is not None and e1 is not None and e1 >= e0:
            energy = e1 - e0
        return wall, energy, proc.returncode

    def profile(self, binary: Path, args: list[str] | None = None) -> ProfileResult:
        args = args or []
        times: list[float] = []
        energies: list[float] = []
        for _ in range(PROFILE_RUNS):
            wall, energy, rc = self._run_once(binary, args)
            if rc != 0 or wall == float("inf"):
                continue
            times.append(wall)
            if energy > 0:
                energies.append(energy)
        if not times:
            return ProfileResult(float("inf"), 0.0, runs=0)
        med_time = statistics.median(times)
        med_energy = statistics.median(energies) if energies else 0.0
        return ProfileResult(
            wall_time_ns=med_time,
            energy_j=med_energy,
            runs=len(times),
            counters={"platform": platform.machine()},
        )
