"""ARM64 profiler — perf time + optional energy sysfs."""

from __future__ import annotations

import platform
import statistics
import subprocess
import time
from pathlib import Path

from rl_uco.config import PROFILE_RUNS, RUN_TIMEOUT_S
from rl_uco.hardware.base import HardwareProfile, ProfileResult


def _read_arm_energy_j() -> float | None:
    """Platform-specific energy; returns None if unavailable."""
    candidates = [
        Path("/sys/class/hwmon/hwmon0/energy1_input"),
        Path("/sys/devices/virtual/hv_kvp/energy_uj"),
    ]
    for p in candidates:
        if p.exists():
            try:
                val = int(p.read_text().strip())
                return val / 1e6 if "uj" in p.name else val / 1e3
            except (OSError, ValueError):
                continue
    return None


class ARMProfiler(HardwareProfile):
    @property
    def isa_key(self) -> str:
        return "aarch64"

    def _run_once(self, binary: Path, args: list[str]) -> tuple[float, float, int]:
        e0 = _read_arm_energy_j()
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
        e1 = _read_arm_energy_j()
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
        return ProfileResult(
            wall_time_ns=statistics.median(times),
            energy_j=statistics.median(energies) if energies else 0.0,
            runs=len(times),
            counters={"platform": platform.machine()},
        )
