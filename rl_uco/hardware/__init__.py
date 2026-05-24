from rl_uco.hardware.base import HardwareProfile, ProfileResult
from rl_uco.hardware.cpu_x86 import X86Profiler
from rl_uco.hardware.cpu_arm import ARMProfiler
from rl_uco.hardware.gpu_cuda import CUDAProfiler

__all__ = [
    "HardwareProfile",
    "ProfileResult",
    "X86Profiler",
    "ARMProfiler",
    "CUDAProfiler",
    "get_profiler",
]


def get_profiler(isa: str) -> HardwareProfile:
    if isa.startswith("x86"):
        return X86Profiler()
    if isa.startswith("aarch") or isa == "arm64":
        return ARMProfiler()
    if isa.startswith("cuda"):
        return CUDAProfiler()
    raise ValueError(f"Unknown ISA: {isa}")
