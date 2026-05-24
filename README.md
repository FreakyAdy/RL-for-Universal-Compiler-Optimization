# RL for Universal Compiler Optimization (RL-UCO)

> A production-ready offline reinforcement learning framework for learning compiler pass selection strategies that minimize **runtime and energy consumption** across diverse hardware architectures.

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue?style=flat&logo=python)](https://www.python.org/downloads/)
[![LLVM 18+](https://img.shields.io/badge/LLVM-18+-orange?style=flat)](https://llvm.org/)

## Overview

**RL-UCO** learns to automatically optimize compiler pass sequences for unseen functions and target ISAs by leveraging offline reinforcement learning. Rather than relying on hand-crafted heuristics or static pass orderings, the system trains an agent on historical compile-run traces to predict high-performance pass pipelines tailored to specific program structures and hardware targets.

```
Traditional Compiler Optimization         RL-UCO Approach
═════════════════════════════════════════ ════════════════════════════════════════

  Hand-crafted pass orderings               Learned pass selection from data
  Static -O0 / -O2 / -O3 levels             Dynamic, context-aware optimization
  Fixed per-architecture tuning             Cross-ISA generalization
  Binary outcomes (compile/fail)            Multi-objective (time + energy)
```

### Key Innovations

┌─────────────────────────────────────────────────────────────────────────────┐
│ ✓ Multi-Objective Optimization                                              │
│   Joint minimization of wall-time and energy (real hardware profiling via   │
│   RAPL, NVML, ARM counters)                                                 │
│                                                                             │
│ ✓ Cross-ISA Generalization                                                  │
│   Single trained model handles x86-64, ARM64, and NVIDIA CUDA targets      │
│                                                                             │
│ ✓ Offline Learning                                                          │
│   Trains on pre-collected datasets without online interaction; suitable     │
│   for regulated production environments                                     │
│                                                                             │
│ ✓ Graph-Based State Representation                                          │
│   Encodes LLVM/MLIR IR as heterogeneous program dependence graphs (PDGs)   │
│   for robust feature extraction                                             │
│                                                                             │
│ ✓ Correctness Guarantee                                                     │
│   Built-in validation against functional equivalence and runtime output    │
│   correctness                                                               │
└─────────────────────────────────────────────────────────────────────────────┘

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Docker Setup](#docker-setup-recommended)
- [Pipeline Overview](#pipeline-overview)
- [Configuration & Tuning](#configuration--tuning)
- [Advanced Usage](#advanced-usage)
  - [Distributed Data Collection](#distributed-data-collection)
  - [Custom Pass Sequences](#custom-pass-sequences)
  - [Multi-Hardware Evaluation](#multi-hardware-evaluation)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

---

## Features

### Core Capabilities

```
┌─────────────────────────────────────────────────────────────┐
│                  System Architecture                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input (Function.ll)                                        │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────────────┐                                  │
│  │  Graph Encoder       │  ◄─── ISA Embedding             │
│  │  (LLVM/MLIR → PyG)   │                                  │
│  └──────────┬───────────┘                                  │
│             │                                              │
│             ▼                                              │
│  ┌──────────────────────┐                                  │
│  │  State Embedding     │                                  │
│  │  (GNN + ISA concat)  │                                  │
│  └──────────┬───────────┘                                  │
│             │                                              │
│    ┌────────┴────────┐                                     │
│    ▼                 ▼                                      │
│  Actor            Critic                                   │
│  (Policy)         (Value)                                  │
│    │                 │                                     │
│    └────────┬────────┘                                     │
│             ▼                                              │
│  Pass Sequence: [inline, O2, sroa, ...]                   │
│             │                                              │
│             ▼                                              │
│  Validation & Execution                                    │
│             │                                              │
│             ▼                                              │
│  Optimized IR  +  Reward Score                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Hardware Profiling Capabilities

╔════════════════════════════════════════════════════════════════════════════╗
║                     Platform-Specific Profiling                            ║
╠═══════════════╦══════════════╦═════════════════╦══════════════════════════╣
║   Platform    ║ Wall Time    ║ Energy          ║ Status                   ║
╠═══════════════╬══════════════╬═════════════════╬══════════════════════════╣
║ x86-64        ║ perf         ║ Intel RAPL      ║ ✓ Production Ready       ║
║ ARM64         ║ perf         ║ SPE/sysfs       ║ ✓ Supported              ║
║ NVIDIA CUDA   ║ CUDA API     ║ NVML            ║ ✓ GPU-optimized          ║
║ Windows       ║ QueryPC      ║ N/A (time only) ║ ◐ Partial (Dev mode)     ║
║ macOS         ║ mach_time    ║ N/A (time only) ║ ◐ Time-based rewards     ║
╚═══════════════╩══════════════╩═════════════════╩══════════════════════════╝

### Supported IR & Frameworks

╔════════════════════════════════════════════════════════════════════════════╗
║ IR Type │ Framework      │ Status            │ Details                    ║
╠═════════╬════════════════╬═══════════════════╬════════════════════════════╣
║ LLVM IR │ LLVM 18+       │ ✓ Primary        │ 100+ passes, NPM support   ║
║ MLIR    │ MLIR (dialect) │ ◑ Experimental   │ GPU kernels, unified graph ║
╚════════════════════════════════════════════════════════════════════════════╝

### Reward Function Composition

```
Normalized Reward
       │
       ├─ (0.7 default) ────────────┐
       │                             ▼
       │              wall_time_improvement
       │                             │
       │                             ├─ [1.0 ... 0.0]
       │                             │
       │              ┌──────────────┘
       │              │
       ├─ (0.3 default) ────────────┐
       │                             ▼
       │              energy_improvement
       │                             │
       │                             ├─ [1.0 ... 0.0]
       │                             │
       └──────────────────────────────┘
                    │
                    ▼
          Weighted Sum
          [0.0 ... 1.0]
          
(Configurable via environment variables)
```

---

## Quick Start

### Prerequisites

- **Python:** 3.11 or later
- **LLVM Toolchain:** 18.0+
  - `clang`, `opt`, `llvm-diff` on PATH (or via Docker)
- **System:** Linux recommended for full RAPL/`perf` support
  - WSL2 on Windows: Partial support (time-only reward if RAPL unavailable)
  - macOS: Limited energy profiling (time-based rewards)
- **Optional:** 
  - NVIDIA GPU + CUDA 12.0+ for MLIR/CUDA compilation
  - Docker & Docker Compose for isolated toolchain
  - Ray >= 2.9 for distributed data collection

### Installation

#### Local Environment (Recommended for Development)

```bash
# Clone repository
git clone https://github.com/yourusername/RL-for-Universal-Compiler-Optimization.git
cd RL-for-Universal-Compiler-Optimization

# Create virtual environment
python -m venv .venv

# Activate
# On macOS / Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install core dependencies
pip install -e "."

# Install development tools (pytest, linting)
pip install -e ".[dev]"
```

#### Optional Extensions

```bash
# For distributed data collection (Linux recommended)
pip install -e ".[distributed]"

# For advanced graph neural network encoder (PyTorch Geometric)
pip install -e ".[gnn]"

# For GPU energy profiling (NVIDIA only)
pip install -e ".[gpu]"

# Install all optional features
pip install -e ".[dev,distributed,gnn,gpu]"
```

### Docker Setup (Recommended)

Containerized environment with pre-configured LLVM 18 and all tools:

```bash
# Build Docker image
cd infra/docker
docker compose build

# Enter development container
docker compose run --rm rl-uco-dev bash

# Inside container, verify tools
clang --version  # LLVM 18.x
opt --version    # LLVM 18.x
```

For GPU support, configure Docker with NVIDIA runtime:

```yaml
# In docker-compose.yml, add to rl-uco-dev service:
runtime: nvidia
environment:
  - NVIDIA_VISIBLE_DEVICES=all
```

---

## Pipeline Overview

The complete workflow from raw source code to optimized inference:

```
╔════════════════════════════════════════════════════════════════════════════╗
║                         RL-UCO COMPLETE PIPELINE                          ║
╚════════════════════════════════════════════════════════════════════════════╝

    PHASE 1: DATA PREPARATION
    ═══════════════════════════════════════════════════════════════════════
    
    ┌─────────────┐
    │Source Code  │  C/C++ functions from open-source or custom codebase
    │   (*.c)     │
    └──────┬──────┘
           │
           │ rl-uco-extract
           │
           ▼
    ┌──────────────────────────────┐
    │ 1. Corpus Extraction         │  ✓ Extract functions
    │                              │  ✓ Normalize to LLVM IR
    │ rl-uco-extract               │  ✓ Flatten nested functions
    │ --source src/               │  ✓ Remove undefined references
    │ --output data/corpus        │  ✓ Emit .ll bitcode
    │                              │  ✓ Validate IR size & complexity
    └──────┬───────────────────────┘
           │
           └─────────────────────┐
                                 │
    PHASE 2: DATA COLLECTION      │
    ═══════════════════════════════════════════════════════════════════════
                                 │
           ┌─────────────────────┘
           │
           ▼
    ┌──────────────────────────────────────────────┐
    │  2. Offline Data Collection                  │  ┌─ Bootstrap Policies:
    │     (Compile + Profile on Hardware)          │  │   • -O3 (baseline)
    │                                              │  │   • Random passes
    │  rl-uco-collect                             │  │   • Mutations
    │  --corpus data/corpus                       │  │   • Custom sequences
    │  --output data/datasets/v1                  │  └─
    │  --isa x86_64_v3                            │
    │  --bootstrap-policies O3,random             │  Per-sample flow:
    │  --workers 8                                │  1. Select pass sequence
    │                                              │  2. Compile via opt
    │  ╔═══════════════════════════════════════╗  │  3. Codegen with clang
    │  ║  CompileRunEnv + HardwareProfiler   ║  │  4. Run benchmark harness
    │  ║  ┌──────────────────────────────┐   ║  │  5. Record wall-time
    │  ║  │ opt ──► clang ──► execute     │   ║  │  6. Record energy (RAPL)
    │  ║  │         │          │          │   ║  │  7. Validate correctness
    │  ║  │         └─►perf    │          │   ║  │     (llvm-diff, output)
    │  ║  │         └─►RAPL    │          │   ║  │  8. Calculate reward
    │  ║  └──────────────────────────────┘   ║  │  9. Write to Parquet
    │  ╚═══════════════════════════════════════╝  │
    │                                              │
    │  Output: Parquet dataset (100s-1000s rows)  │
    └──────┬───────────────────────────────────────┘
           │
           │ Parquet + PyG graphs + Metadata
           │
    PHASE 3: TRAINING
    ═══════════════════════════════════════════════════════════════════════
           │
           ▼
    ┌──────────────────────────────────────────────┐
    │  3. Offline RL Training (IQL)                │  Load fixed dataset
    │                                              │  (no online interaction)
    │  rl-uco-train                               │
    │  --dataset data/datasets/v1                 │  ╔═════════════════════╗
    │  --output checkpoints/                      │  ║  Actor (Policy)     ║
    │  --batch-size 128                           │  ║  ├─ Pass Embedding  ║
    │  --epochs 100                               │  ║  ├─ GRU Encoder     ║
    │  --learning-rate 3e-4                       │  ║  └─ Output Logits   ║
    │                                              │  ║                     ║
    │  ╔══════════════════════════════════════╗   │  ║  Critic (Value)     ║
    │  ║ IQL Update Loop (1000s of steps)     ║   │  ║  ├─ State MLP       ║
    │  ║                                      ║   │  ║  └─ Value Estimate  ║
    │  ║ Sample batch from offline dataset    ║   │  ╚═════════════════════╝
    │  ║ │                                    ║   │
    │  ║ ├─ Update Critic:                    ║   │  Min: (Q(s,a) - target)²
    │  ║ │  Q(s,a) ← reward + γ*max_a Q(s',a)║   │
    │  ║ │                                    ║   │
    │  ║ ├─ Update Actor:                     ║   │  Max: Q(s, π(s))
    │  ║ │  π(a|s) ← improved policy          ║   │
    │  ║ │                                    ║   │
    │  ║ └─ Repeat (conservative updates)     ║   │
    │  ╚══════════════════════════════════════╝   │
    │                                              │
    │  Checkpoint: best.pt (lowest validation    │
    │             loss)                           │
    └──────┬───────────────────────────────────────┘
           │
           │ best.pt (actor + critic + encoder)
           │
    PHASE 4: EVALUATION
    ═══════════════════════════════════════════════════════════════════════
           │
           ▼
    ┌──────────────────────────────────────────────┐
    │  4. Evaluation & Reporting                   │
    │                                              │
    │  rl-uco-eval                                │
    │  --dataset data/datasets/v1                 │
    │  --checkpoint checkpoints/best.pt           │
    │  --baseline O3                              │
    │  --output reports/                          │
    │                                              │
    │  Results:                                    │
    │  ├─ Mean speedup vs -O3: 1.23x              │
    │  ├─ Energy reduction: 15%                    │
    │  ├─ Statistical significance: p < 0.01      │
    │  └─ Failure rate: 0.2% (with fallback)      │
    └──────┬───────────────────────────────────────┘
           │
           │ Validated checkpoint
           │
    PHASE 5: DEPLOYMENT / INFERENCE
    ═══════════════════════════════════════════════════════════════════════
           │
           ▼
    ┌──────────────────────────────────────────────┐
    │  5. Inference on Unseen Functions            │
    │                                              │
    │  rl-uco-infer                               │
    │  --ir function.ll                           │
    │  --checkpoint checkpoints/best.pt           │
    │  --isa x86_64_v3                            │
    │  --output optimized.ll                      │
    │                                              │
    │  Inference flow:                             │
    │  1. Encode IR to PyG graph                   │
    │  2. Get ISA embedding (x86_64_v3)           │
    │  3. Forward pass: state ──► logits           │
    │  4. Sample sequence: [pass1, pass2, ...]    │
    │  5. Validate against registry.yaml          │
    │  6. Execute via opt (or fallback to -O3)    │
    │  7. Write optimized IR                       │
    │                                              │
    │  ┌─────────────────────────────────────┐    │
    │  │ Fallback Strategy                   │    │
    │  │ ├─ Invalid sequence? ──► -O3        │    │
    │  │ ├─ Compilation error? ──► -O2      │    │
    │  │ ├─ Timeout? ──► -O1                 │    │
    │  │ └─ All fail? ──► -O0 (bail-out)    │    │
    │  └─────────────────────────────────────┘    │
    │                                              │
    │  Output: Optimized IR + metadata            │
    └──────────────────────────────────────────────┘

```

### Detailed Example Commands

Complete walkthrough from source to inference:

```bash
╔════════════════════════════════════════════════════════════════════════════╗
║                         PHASE 1: EXTRACT CORPUS                           ║
╚════════════════════════════════════════════════════════════════════════════╝

rl-uco-extract \
  --source tests/fixtures \
  --output data/corpus \
  --max-ir-size 5000 \
  --max-functions 32

# Result: data/corpus/
#   ├─ manifest.json
#   └─ synth_0000/ to synth_0099/
#      ├─ fn.c (original)
#      └─ fn.ll (normalized LLVM IR)


╔════════════════════════════════════════════════════════════════════════════╗
║                    PHASE 2: COLLECT DATASET (x86-64)                      ║
╚════════════════════════════════════════════════════════════════════════════╝

rl-uco-collect \
  --corpus data/corpus \
  --output data/datasets/demo_v1 \
  --isa x86_64_v3 \
  --bootstrap-policies O3,random,mutations \
  --num-samples 5000 \
  --workers 8

# Result: data/datasets/demo_v1/
#   ├─ manifest.json (metadata)
#   ├─ data.parquet (5000 rows × 15 columns)
#   └─ graphs/
#      ├─ synth_0000.pt (PyG graph)
#      └─ synth_0001.pt ...


╔════════════════════════════════════════════════════════════════════════════╗
║                         PHASE 3: TRAIN POLICY                             ║
╚════════════════════════════════════════════════════════════════════════════╝

rl-uco-train \
  --dataset data/datasets/demo_v1 \
  --output checkpoints/ \
  --batch-size 128 \
  --epochs 100 \
  --learning-rate 3e-4 \
  --encoder gnn \
  --seed 42

# Training progress:
#   Epoch   1/100  │ Train Loss: 2.34 │ Val Loss: 2.41 ✓
#   Epoch  50/100  │ Train Loss: 0.87 │ Val Loss: 0.95 ✓
#   Epoch 100/100  │ Train Loss: 0.42 │ Val Loss: 0.51 ✓
#
# Result: checkpoints/
#   ├─ best.pt (lowest val loss)
#   ├─ epoch_050.pt
#   └─ final.pt


╔════════════════════════════════════════════════════════════════════════════╗
║                         PHASE 4: EVALUATE                                 ║
╚════════════════════════════════════════════════════════════════════════════╝

rl-uco-eval \
  --dataset data/datasets/demo_v1 \
  --checkpoint checkpoints/best.pt \
  --baseline O3 \
  --output reports/

# Evaluation report: reports/eval_report.md
#
#   ┌─────────────────────────────────────┐
#   │ RL-UCO vs -O3 Baseline              │
#   ├─────────────────────────────────────┤
#   │ Mean Speedup:        1.27x (±0.12)  │
#   │ Energy Reduction:    18.3% (±2.1%)  │
#   │ Success Rate:        99.8%          │
#   │ Statistical Sig:     p < 0.001      │
#   └─────────────────────────────────────┘


╔════════════════════════════════════════════════════════════════════════════╗
║                    PHASE 5: INFERENCE ON NEW CODE                         ║
╚════════════════════════════════════════════════════════════════════════════╝

# Optimize a single function
rl-uco-infer \
  --ir /path/to/unseen_function.ll \
  --checkpoint checkpoints/best.pt \
  --isa x86_64_v3 \
  --output optimized.ll

# Batch inference on corpus
for ll in data/corpus/*/*.ll; do
  rl-uco-infer \
    --ir "$ll" \
    --checkpoint checkpoints/best.pt \
    --isa x86_64_v3 \
    --output "out/$(basename $ll)"
done
```

---

## Configuration & Tuning

All configuration is controlled via environment variables or [rl_uco/config.py](rl_uco/config.py). 

### Key Parameters

╔════════════════════════════════════════════════════════════════════════════╗
║ REWARD CONFIGURATION                                                       ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  RL_UCO_W_TIME=0.7           ─┐                                           ║
║                               ├─► Weight distribution in reward            ║
║  RL_UCO_W_ENERGY=0.3         ─┘   (must sum to 1.0)                       ║
║                                                                            ║
║  RL_UCO_FAILURE_REWARD=-10.0      Penalty for failed compilations         ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

╔════════════════════════════════════════════════════════════════════════════╗
║ EXECUTION CONSTRAINTS                                                      ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  RL_UCO_MAX_PASS_STEPS=12         Max passes per sequence (episode length) ║
║  RL_UCO_COMPILE_TIMEOUT=120       Compilation timeout (seconds)            ║
║  RL_UCO_RUN_TIMEOUT=30            Execution timeout (seconds)              ║
║  RL_UCO_RUN_ITERATIONS=100000     Microbenchmark loop count                ║
║  RL_UCO_PROFILE_RUNS=5            Profiling runs (median taken)            ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

╔════════════════════════════════════════════════════════════════════════════╗
║ CORPUS FILTERS                                                             ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  RL_UCO_MAX_IR_INSTRUCTIONS=5000  Max IR instructions per function         ║
║  RL_UCO_MAX_FUNCS_PER_FILE=32     Max functions extracted per file         ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

╔════════════════════════════════════════════════════════════════════════════╗
║ MODEL ARCHITECTURE                                                         ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  ISA_EMBED_DIM=32                 ISA embedding dimension                  ║
║  GRAPH_HIDDEN_DIM=256             Graph encoder hidden size                ║
║  GRAPH_NUM_LAYERS=4               GNN stacking depth                       ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

### Configuration Examples

#### Scenario 1: Prioritize Runtime Performance

```bash
export RL_UCO_W_TIME=0.9
export RL_UCO_W_ENERGY=0.1
export RL_UCO_MAX_PASS_STEPS=8

rl-uco-train --dataset data/datasets/demo_v1 --output checkpoints/speed-opt
```

#### Scenario 2: Energy-Aware Optimization (Mobile / Edge)

```bash
export RL_UCO_W_TIME=0.2
export RL_UCO_W_ENERGY=0.8
export RL_UCO_MAX_PASS_STEPS=20  # longer sequences for more exploration

rl-uco-train --dataset data/datasets/demo_v1 --output checkpoints/energy-opt
```

#### Scenario 3: Balanced Multi-Objective with Strict Timeouts

```bash
export RL_UCO_W_TIME=0.5
export RL_UCO_W_ENERGY=0.5
export RL_UCO_COMPILE_TIMEOUT=60    # strict compilation deadline
export RL_UCO_MAX_PASS_STEPS=6      # shorter sequences = faster compilation

rl-uco-train --dataset data/datasets/demo_v1 --output checkpoints/balanced
```

### ISA-Specific Configurations

Codegen flags by ISA (from `config.py`):

```python
ISA_FLAGS = {
    "x86_64_v3":   ["-march=x86-64-v3"],           # AVX-2 capable
    "aarch64":     ["-mcpu=neoverse-n1"],          # ARM Neoverse N1
    "cuda_sm80":   [],                              # CUDA (handled separately)
}
```

Target-specific optimizations:

```bash
# x86-64: Focus on SIMD passes
rl-uco-collect --corpus data/corpus --isa x86_64_v3 --output data/x86

# ARM64: Energy-efficient passes
rl-uco-collect --corpus data/corpus --isa aarch64 --output data/arm
```
export RL_UCO_MAX_PASS_STEPS=20

rl-uco-train --dataset data/datasets/demo_v1 --output checkpoints/energy-opt
```

---

## Advanced Usage

### Distributed Data Collection

For large-scale corpus collection, use Ray distributed collection (Linux only):

```bash
pip install -e ".[distributed]"

rl-uco-collect \
  --corpus data/corpus \
  --output data/datasets/large_v1 \
  --isa x86_64_v3 \
  --use-ray \
  --ray-workers 16 \
  --num-samples 100000
```

Requires Ray cluster setup; for local multi-core, Ray auto-detects CPU count.

### Custom Pass Sequences

Extend `rl_uco/passes/registry.yaml` to add custom passes or change baseline policies:

```yaml
passes:
  strip-debug:
    pass_name: strip-debug
    category: metadata
    required_ir: llvm
    description: "Remove debug metadata"
    
  custom-inline:
    pass_name: inline
    category: ipo
    params: ["-inline-threshold=200"]
    description: "Custom inlining threshold"

policies:
  custom_baseline:
    - O2
    - strip-debug
    - custom-inline
```

### Multi-Hardware Evaluation

To train a single model generalizing across ISAs:

```bash
# 1. Collect data on each target
rl-uco-collect --corpus data/corpus --output data/datasets/x86_v1 --isa x86_64_v3
rl-uco-collect --corpus data/corpus --output data/datasets/arm_v1 --isa aarch64

# 2. Merge datasets
rl-uco-merge-datasets \
  --inputs data/datasets/x86_v1 data/datasets/arm_v1 \
  --output data/datasets/multi_v1

# 3. Train (ISA embedding automatically conditions policy)
rl-uco-train --dataset data/datasets/multi_v1 --output checkpoints/multi-isa
```

---

## Project Structure

```
RL-for-Universal-Compiler-Optimization/
├── README.md                          # This file
├── LICENSE                            # Apache-2.0 license
├── pyproject.toml                     # Project metadata & dependencies
│
├── rl_uco/                            # Core package
│   ├── __init__.py
│   ├── config.py                      # Global configuration, toolchain discovery
│   │
│   ├── corpus/                        # Function extraction & normalization
│   │   ├── extract.py                 # Main corpus extraction CLI
│   │   ├── models.py                  # LLVM/C AST models
│   │   └── normalize.py               # IR normalization passes
│   │
│   ├── passes/                        # Compiler pass management
│   │   ├── registry.yaml              # Pass definitions & validation rules
│   │   ├── executor.py                # Pass execution via opt
│   │   ├── registry.py                # Pass registry data structures
│   │   └── validator.py               # Sequence validation
│   │
│   ├── env/                           # Compile-run environment
│   │   ├── compile_run.py             # Compilation & execution harness
│   │   └── reward.py                  # Multi-objective reward calculation
│   │
│   ├── hardware/                      # Hardware profilers
│   │   ├── base.py                    # Abstract profiler interface
│   │   ├── cpu_x86.py                 # Intel RAPL + perf
│   │   ├── cpu_arm.py                 # ARM energy counters
│   │   └── gpu_cuda.py                # NVIDIA NVML profiler
│   │
│   ├── graph/                         # IR → Graph conversion
│   │   ├── llvm_to_graph.py           # LLVM IR → PyG graph
│   │   ├── mlir_to_graph.py           # MLIR → PyG graph
│   │   └── parse.py                   # IR parsing utilities
│   │
│   ---

## Project Structure

```
RL-for-Universal-Compiler-Optimization/
│
├─ Configuration & Build
│  ├─ README.md                                    ◄ You are here
│  ├─ LICENSE                                      Apache-2.0
│  ├─ pyproject.toml                              Project metadata & dependencies
│  └─ .gitignore
│
├─ Core Package: rl_uco/
│  │
│  ├─ __init__.py
│  ├─ config.py                                    ◄ Global config + toolchain discovery
│  │
│  ├─ corpus/                                      Function extraction & normalization
│  │  ├─ extract.py                               ◄ CLI: rl-uco-extract
│  │  ├─ models.py                                LLVM/C AST representations
│  │  ├─ normalize.py                             IR normalization passes
│  │  └─ __init__.py
│  │
│  ├─ passes/                                      Compiler pass orchestration
│  │  ├─ registry.yaml                            ◄ Pass definitions & validation
│  │  ├─ executor.py                              Pass execution via opt
│  │  ├─ registry.py                              Registry data structures
│  │  ├─ validator.py                             Sequence validation
│  │  └─ __init__.py
│  │
│  ├─ env/                                         Compile-run environment
│  │  ├─ compile_run.py                           Compilation & execution harness
│  │  ├─ reward.py                                Multi-objective reward calculation
│  │  └─ __init__.py
│  │
│  ├─ hardware/                                    Hardware profilers (multi-ISA)
│  │  ├─ base.py                                  Abstract profiler interface
│  │  ├─ cpu_x86.py                               Intel RAPL + perf
│  │  ├─ cpu_arm.py                               ARM energy counters
│  │  ├─ gpu_cuda.py                              NVIDIA NVML profiler
│  │  └─ __init__.py
│  │
│  ├─ graph/                                       IR → PyG graph conversion
│  │  ├─ llvm_to_graph.py                         LLVM IR → PyG graph
│  │  ├─ mlir_to_graph.py                         MLIR → PyG graph
│  │  ├─ parse.py                                 IR parsing utilities
│  │  └─ __init__.py
│  │
│  ├─ ir/                                          IR handling adapters
│  │  ├─ llvm_adapter.py                          LLVM IR interface
│  │  ├─ mlir_adapter.py                          MLIR interface
│  │  └─ __init__.py
│  │
│  ├─ data/                                        Dataset management
│  │  ├─ collector.py                             ◄ CLI: rl-uco-collect
│  │  ├─ schema.py                                Parquet schema definitions
│  │  ├─ versioning.py                            Dataset versioning logic
│  │  ├─ export_parquet.py                        Parquet export utilities
│  │  └─ __init__.py
│  │
│  ├─ rl/                                          Reinforcement learning
│  │  ├─ actor_critic.py                          PassPolicy + PassCritic
│  │  ├─ encoder.py                               StateEncoder (graph → embedding)
│  │  ├─ offline_trainer.py                       ◄ CLI: rl-uco-train (IQL training)
│  │  ├─ inference.py                             ◄ CLI: rl-uco-infer
│  │  └─ __init__.py
│  │
│  └─ eval/                                        Evaluation & reporting
│     ├─ report.py                                ◄ CLI: rl-uco-eval
│     └─ __init__.py
│
├─ Infrastructure & Deployment: infra/
│  │
│  ├─ docker/
│  │  ├─ Dockerfile                               LLVM 18 development container
│  │  └─ docker-compose.yml                       Multi-container orchestration
│  │
│  └─ inference/
│     └─ opt_driver.py                            Standalone inference driver
│                                                  (Python wrapper for opt)
│
├─ Data Directory (Generated): data/
│  │
│  ├─ corpus/                                      Extracted function corpus
│  │  └─ demo/
│  │     ├─ manifest.json
│  │     ├─ synth_0000/ ─► fn.c, fn.ll
│  │     ├─ synth_0001/ ─► fn.c, fn.ll
│  │     └─ ... (up to N functions)
│  │
│  ├─ datasets/                                    Versioned Parquet datasets
│  │  └─ demo_v1/
│  │     ├─ manifest.json                         (metadata: LLVM version, config)
│  │     ├─ data.parquet                          (N rows × 15 columns)
│  │     └─ graphs/
│  │        ├─ synth_0000.pt                      PyG graph tensors
│  │        ├─ synth_0001.pt
│  │        └─ ...
│  │
│  └─ graphs/ (cache)
│     └─ (PyG serialized graphs)
│
├─ Checkpoints: checkpoints/
│  └─ best.pt                                      Trained model checkpoint
│
├─ Documentation: docs/
│  ├─ architecture.md                             System design, data flow, IR strategies
│  ├─ dataset_schema.md                           Parquet schema specification
│  ├─ deployment.md                               Production deployment guide
│  └─ hardware_setup.md                           Platform-specific profiling setup
│
├─ Utilities & Examples: scripts/
│  ├─ generate_demo_data.py
│  └─ scale_corpus.py
│
└─ Tests: tests/
   ├─ test_corpus.py
   ├─ test_graph.py
   ├─ test_passes.py
   ├─ test_reward.py
   ├─ test_agent.py
   ├─ test_schema.py
   └─ fixtures/
      ├─ sample.c
      ├─ sample_kernel.mlir
      └─ sample.ll
```

### Module Dependency Graph

```
            ┌──────────────────────────────────────────┐
            │  User Input (IR, Checkpoint, Config)     │
            └──────────────────┬───────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
    ┌────────────┐      ┌────────────┐       ┌──────────────┐
    │  corpus/   │      │   graph/   │       │  passes/     │
    │  extract   │      │ llvm_to_   │       │  registry    │
    │            │      │ graph      │       │              │
    └────┬───────┘      └────┬───────┘       └──────┬───────┘
         │                   │                       │
         │                   └───────────┬───────────┘
         │                               │
         ▼                               ▼
    ┌────────────┐              ┌──────────────────┐
    │  data/     │              │  rl/encoder      │
    │  schema    │              │  StateEncoder    │
    └────┬───────┘              └──────┬───────────┘
         │                             │
         │                             ▼
         │                      ┌──────────────────┐
         │                      │  rl/actor_critic │
         │                      │  Policy + Critic │
         │                      └──────┬───────────┘
         │                             │
         │        ┌────────────────────┘
         │        │
         ▼        ▼
    ┌────────────────────┐
    │  env/              │
    │  compile_run +     │
    │  reward            │
    └────┬───────────────┘
         │
         ▼
    ┌────────────────────┐
    │  hardware/         │
    │  profilers         │
    └────────────────────┘
```

---

## Documentation

Comprehensive guides for each component:

┌────────────────────────────────────────────────────────────────────────────┐
│ Document                    │ Coverage                                     │
├────────────────────────────────────────────────────────────────────────────┤
│ [Architecture](docs/architecture.md)         │ System design, data flow,    │
│                                 │ IR strategies, ISA conditioning          │
│ [Dataset Schema](docs/dataset_schema.md)     │ Parquet specification, field │
│                                 │ descriptions, versioning scheme          │
│ [Deployment](docs/deployment.md)             │ Production inference, TorchScript  │
│                                 │ export, external driver, CI/batch        │
│ [Hardware Setup](docs/hardware_setup.md)     │ x86 RAPL, ARM, CUDA, Windows │
│                                 │ profiling configuration                  │
└────────────────────────────────────────────────────────────────────────────┘

---

## Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork & Branch:** Create a feature branch (`feature/your-feature`)
2. **Code Style:** Follow PEP 8; use `ruff format` and `ruff check` for linting
   ```bash
   pip install ruff
   ruff format rl_uco/ tests/
   ruff check rl_uco/ tests/
   ```
3. **Tests:** Add unit tests for new functionality; maintain >80% coverage
   ```bash
   pytest tests/ -v --cov=rl_uco
   ```
4. **Documentation:** Update docstrings (Google-style) and relevant `.md` files
5. **Commit:** Use clear, descriptive commit messages
6. **Pull Request:** Link to issue, provide rationale & test results

---

## Citation

If you use RL-UCO in your research, please cite:

```bibtex
@software{rluco2024,
  title={RL-UCO: Offline Reinforcement Learning for Universal Compiler Optimization},
  author={Contributors, RL-UCO},
  year={2024},
  url={https://github.com/yourusername/RL-for-Universal-Compiler-Optimization}
}
```

---

## Troubleshooting

### ❯ LLVM Tools Not Found

**Problem:** Commands like `clang`, `opt`, or `llvm-diff` not found on PATH

**Diagnosis:**
```bash
clang --version    # Should show LLVM 18+
opt --version      # Should show LLVM 18+
which llvm-diff    # Should return path
```

**Solutions:**

┌────────────────────────────────────────────────────────────────────────────┐
│ Option A: Add LLVM to PATH (Linux/macOS)                                  │
├────────────────────────────────────────────────────────────────────────────┤
│ export PATH="/path/to/llvm-18/bin:$PATH"                                   │
│ export LD_LIBRARY_PATH="/path/to/llvm-18/lib:$LD_LIBRARY_PATH"            │
│                                                                            │
│ Option B: Install via package manager                                     │
├────────────────────────────────────────────────────────────────────────────┤
│ # Ubuntu/Debian                                                            │
│ sudo apt-get install -y llvm-18                                            │
│                                                                            │
│ # macOS (Homebrew)                                                         │
│ brew install llvm@18                                                       │
│                                                                            │
│ Option C: Use Docker (Recommended)                                         │
├────────────────────────────────────────────────────────────────────────────┤
│ cd infra/docker && docker compose run --rm rl-uco-dev bash               │
│ # Inside container, clang/opt/llvm-diff already on PATH                   │
└────────────────────────────────────────────────────────────────────────────┘

---

### ❯ Energy Profiling Not Working

**Problem:** Energy values are 0 or missing; only wall-time optimized

**Platform-Specific Diagnostics:**

╔════════════════════════════════════════════════════════════════════════════╗
║ x86-64 (Intel RAPL)                                                        ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║ Check 1: RAPL readable                                                    ║
║   $ cat /sys/class/powercap/intel-rapl:0/energy_uj                       ║
║   ✓ Should show energy in microjoules                                     ║
║                                                                            ║
║ Check 2: msr module loaded                                                ║
║   $ lsmod | grep msr                                                      ║
║   $ sudo modprobe msr  # if not loaded                                    ║
║                                                                            ║
║ Check 3: Permissions fixed                                                ║
║   $ sudo chmod +r /sys/class/powercap/intel-rapl:0/energy_uj             ║
║   $ sudo chmod +r /sys/class/powercap/intel-rapl:1/energy_uj             ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

╔════════════════════════════════════════════════════════════════════════════╗
║ ARM64 (Platform Energy Counters)                                          ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║ Check 1: SPE (Statistical Profiling Extension) available                 ║
║   $ perf list | grep spe                                                  ║
║   ✓ Should show energy_* events if supported                              ║
║                                                                            ║
║ Status: Falls back gracefully to time-only if unavailable                ║
║         Warnings printed to stderr                                        ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

╔════════════════════════════════════════════════════════════════════════════╗
║ NVIDIA CUDA (NVML)                                                         ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║ Check 1: Driver installed                                                 ║
║   $ nvidia-smi                                                             ║
║   ✓ Should show GPU info                                                  ║
║                                                                            ║
║ Check 2: CUDA installed                                                   ║
║   $ nvcc --version  # CUDA Toolkit                                        ║
║   ✓ Should show CUDA version >= 12.0                                     ║
║                                                                            ║
║ Check 3: pynvml installed                                                 ║
║   $ pip install -e ".[gpu]"                                              ║
║   $ python -c "import pynvml; print('OK')"                               ║
║                                                                            ║
║ Check 4: GPU visibility                                                   ║
║   $ CUDA_VISIBLE_DEVICES=0 python -c "import torch; print(torch.cuda.is_available())" │
║   ✓ Should print True                                                     ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

---

### ❯ Slow Data Collection

**Problem:** `rl-uco-collect` takes hours to complete

**Cause:** Single-process collection on large corpus

**Solution 1: Ray Distributed Collection (Linux recommended)**

```bash
# Install Ray support
pip install -e ".[distributed]"

# Enable Ray with 16 workers
rl-uco-collect \
  --corpus data/corpus \
  --output data/datasets/v1 \
  --isa x86_64_v3 \
  --use-ray \
  --ray-workers 16  # Adjust based on CPU count

# Speedup: ~10-14x on 16-core machine
```

**Solution 2: Reduce Corpus Size**

```bash
# Use smaller corpus
rl-uco-collect \
  --corpus data/corpus \
  --output data/datasets/small_v1 \
  --isa x86_64_v3 \
  --num-samples 100  # instead of 1000

# Later scale up with more data
```

---

### ❯ Policy Performance Degradation

**Problem:** Trained policy performs worse than -O3 baseline

**Diagnostics:**

```bash
# 1. Check reward weight balance
echo "Current weights:"
echo "  W_TIME=${RL_UCO_W_TIME:-0.7}"
echo "  W_ENERGY=${RL_UCO_W_ENERGY:-0.3}"

# 2. Inspect dataset quality
python3 << 'EOF'
import pandas as pd
df = pd.read_parquet('data/datasets/v1/data.parquet')
print(df[['reward', 'wall_time_ns', 'energy_j']].describe())
print("\nPass sequence distribution:")
print(df['pass_sequence'].apply(len).value_counts())
EOF

# 3. Check training convergence
# Look for training loss plateauing early
```

**Common Causes & Fixes:**

┌────────────────────────────────────────────────────────────────────────────┐
│ Issue                         │ Fix                                        │
├────────────────────────────────────────────────────────────────────────────┤
│ Imbalanced reward weights     │ Adjust RL_UCO_W_TIME & RL_UCO_W_ENERGY    │
│ Insufficient training data    │ Increase num-samples during collection    │
│ Training stopped too early    │ Increase --epochs (try 200, 500)          │
│ Biased bootstrap policies     │ Diversify with more random/mutation       │
│ Dataset distribution skewed   │ Check for missing ISA target or regime    │
│ Hardware variability          │ Increase PROFILE_RUNS for median timing   │
└────────────────────────────────────────────────────────────────────────────┘

**Recommended Retrain:**

```bash
# Energy-optimized policy
export RL_UCO_W_TIME=0.3
export RL_UCO_W_ENERGY=0.7

rl-uco-train \
  --dataset data/datasets/v1 \
  --output checkpoints/energy-focus \
  --batch-size 64 \
  --epochs 500 \
  --learning-rate 5e-4
```

---

## License

Apache License 2.0 — See [LICENSE](LICENSE) for full details.

---

## Acknowledgments

Built with [LLVM](https://llvm.org/), [PyTorch](https://pytorch.org/), [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/), and [Ray](https://www.ray.io/).

---

**Questions?** Open an issue or contact the maintainers.
