"""NVIDIA CUDA profiling with NVML power integration."""

from __future__ import annotations

import statistics
import subprocess
import time
from pathlib import Path

from rl_uco.config import PROFILE_RUNS, RUN_TIMEOUT_S
from rl_uco.hardware.base import HardwareProfile, ProfileResult

try:
    import pynvml

    _NVML = True
except ImportError:
    _NVML = False


class CUDAProfiler(HardwareProfile):
    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self._initialized = False

    @property
    def isa_key(self) -> str:
        return "cuda_sm80"

    def _init_nvml(self) -> None:
        if self._initialized or not _NVML:
            return
        pynvml.nvmlInit()
        self._initialized = True

    def _read_power_mw(self) -> float:
        if not _NVML:
            return 0.0
        self._init_nvml()
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        return float(pynvml.nvmlDeviceGetPowerUsage(handle))

    def _run_once(self, binary: Path, args: list[str]) -> tuple[float, float, int]:
        samples: list[float] = []
        t0 = time.perf_counter_ns()
        try:
            proc = subprocess.Popen(
                [str(binary), *args],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            while proc.poll() is None:
                samples.append(self._read_power_mw())
                time.sleep(0.001)
            proc.wait(timeout=RUN_TIMEOUT_S)
        except (subprocess.TimeoutExpired, OSError):
            return float("inf"), 0.0, -1
        t1 = time.perf_counter_ns()
        wall_s = (t1 - t0) / 1e9
        avg_mw = statistics.mean(samples) if samples else 0.0
        energy_j = (avg_mw / 1000.0) * wall_s
        return t1 - t0, energy_j, proc.returncode if proc.returncode is not None else -1

    def profile(self, binary: Path, args: list[str] | None = None) -> ProfileResult:
        args = args or []
        times: list[float] = []
        energies: list[float] = []
        for _ in range(PROFILE_RUNS):
            wall, energy, rc = self._run_once(binary, args)
            if rc != 0 or wall == float("inf"):
                continue
            times.append(wall)
            energies.append(energy)
        if not times:
            return ProfileResult(float("inf"), 0.0, runs=0)
        return ProfileResult(
            wall_time_ns=statistics.median(times),
            energy_j=statistics.median(energies),
            runs=len(times),
            counters={"nvml": _NVML},
        )
