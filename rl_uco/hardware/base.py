"""Abstract hardware profiling interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProfileResult:
    wall_time_ns: float
    energy_j: float
    cycles: int = 0
    counters: dict[str, float] = field(default_factory=dict)
    runs: int = 0


class HardwareProfile(ABC):
    @abstractmethod
    def profile(self, binary: Path, args: list[str] | None = None) -> ProfileResult:
        """Run binary multiple times; return median time and energy."""

    @property
    @abstractmethod
    def isa_key(self) -> str:
        pass
