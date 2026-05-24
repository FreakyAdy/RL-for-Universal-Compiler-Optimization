"""Global configuration and toolchain discovery."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

# Reward weights
REWARD_TIME_WEIGHT = float(os.environ.get("RL_UCO_W_TIME", "0.7"))
REWARD_ENERGY_WEIGHT = float(os.environ.get("RL_UCO_W_ENERGY", "0.3"))
FAILURE_REWARD = float(os.environ.get("RL_UCO_FAILURE_REWARD", "-10.0"))

# Episode / passes
MAX_PASS_STEPS = int(os.environ.get("RL_UCO_MAX_PASS_STEPS", "12"))
COMPILE_TIMEOUT_S = int(os.environ.get("RL_UCO_COMPILE_TIMEOUT", "120"))
RUN_TIMEOUT_S = int(os.environ.get("RL_UCO_RUN_TIMEOUT", "30"))
RUN_ITERATIONS = int(os.environ.get("RL_UCO_RUN_ITERATIONS", "100000"))
PROFILE_RUNS = int(os.environ.get("RL_UCO_PROFILE_RUNS", "5"))

# Corpus limits
MAX_IR_INSTRUCTIONS = int(os.environ.get("RL_UCO_MAX_IR_INSTRUCTIONS", "5000"))
MAX_FUNCTIONS_PER_FILE = int(os.environ.get("RL_UCO_MAX_FUNCS_PER_FILE", "32"))

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path(os.environ.get("RL_UCO_DATA", PROJECT_ROOT / "data"))
CORPUS_DIR = DEFAULT_DATA_DIR / "corpus"
DATASET_DIR = DEFAULT_DATA_DIR / "datasets"
GRAPH_DIR = DEFAULT_DATA_DIR / "graphs"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# ISA codegen flags
ISA_FLAGS: dict[str, list[str]] = {
    "x86_64_v3": ["-march=x86-64-v3"],
    "aarch64": ["-mcpu=neoverse-n1"],
    "cuda_sm80": [],  # handled by MLIR/CUDA path
}

ISA_EMBED_DIM = 32
GRAPH_HIDDEN_DIM = 256
GRAPH_NUM_LAYERS = 4


@dataclass
class Toolchain:
    clang: str = "clang"
    opt: str = "opt"
    llvm_diff: str = "llvm-diff"
    mlir_opt: str = "mlir-opt"
    llvm_version: str | None = None

    @classmethod
    def discover(cls) -> Toolchain:
        tc = cls(
            clang=shutil.which("clang") or "clang",
            opt=shutil.which("opt") or "opt",
            llvm_diff=shutil.which("llvm-diff") or "llvm-diff",
            mlir_opt=shutil.which("mlir-opt") or "mlir-opt",
        )
        try:
            import subprocess

            out = subprocess.run(
                [tc.clang, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            for line in out.stdout.splitlines():
                if "version" in line.lower():
                    tc.llvm_version = line.strip()
                    break
        except OSError:
            pass
        return tc


@dataclass
class ISAInfo:
    key: str
    display_name: str
    embed_index: int
    codegen_flags: list[str] = field(default_factory=list)


ISA_REGISTRY: dict[str, ISAInfo] = {
    "x86_64_v3": ISAInfo("x86_64_v3", "x86-64-v3", 0, ISA_FLAGS["x86_64_v3"]),
    "aarch64": ISAInfo("aarch64", "AArch64 Neoverse", 1, ISA_FLAGS["aarch64"]),
    "cuda_sm80": ISAInfo("cuda_sm80", "CUDA SM80", 2, []),
}

NUM_ISAS = len(ISA_REGISTRY)
