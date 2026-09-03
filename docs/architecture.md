# Architecture

## Data flow

1. **Corpus mining** extracts single-function translation units (C/C++) and emits LLVM bitcode/IR.
2. **Bootstrap policies** (`-O3`, random valid passes, mutations) generate diverse pass sequences.
3. **CompileRunEnv** applies passes via `opt`, codegen with `clang`, runs a micro-benchmark harness.
4. **HardwareProfiler** records median wall time and energy; **correctness gate** rejects bad rows.
5. **Graph export** precomputes PyG graphs; rows stored in versioned Parquet datasets.
6. **Offline IQL** trains actor (pass policy) and critic on fixed logs.
7. **Inference** loads TorchScript/checkpoint, proposes pass sequence, validates, falls back to `-O3`.

## IR strategy

- **Primary:** LLVM IR with New Pass Manager (`-passes=...` pipeline strings built from structured JSON actions).
- **Secondary:** MLIR for GPU kernels; unified node/edge schema in `graph/` for shared encoder.

## ISA conditioning

A learned embedding concatenates with the graph embedding. Supported ISAs: `x86_64_v3`, `aarch64`, `cuda_sm80`.

## Correctness

Every sample must compile and pass `llvm-diff` (when reference IR exists) or runtime output check against `-O0` baseline.

## Offline RL Paradigm (vs. Interactive Compiler Gyms)

Traditional compiler RL environments operate as interactive Gyms where an agent executes step-by-step actions through a `step()`/`reset()` loop, invoking compiler passes, compiling to native binary, and profiling hardware at each step during training.

RL-UCO adopts an **offline reinforcement learning** design:
1. **Decoupled Data Mining:** Exploration policies (-O3 seeds, random valid passes, pass mutations) run once during collection (`rl-uco-collect`), recording execution traces, hardware metrics, and functional correctness into versioned Parquet logs and precomputed PyG graphs.
2. **Fast Offline Policy Optimization:** Implicit Q-Learning (IQL) trains the actor-critic network purely over static datasets without touching the compiler during gradient descent.
3. **One-Shot Inference with Fallback:** At inference time (`rl-uco-infer`), the trained policy evaluates the program graph in a single forward pass to propose a complete pass sequence, validated through a correctness gate before execution.

