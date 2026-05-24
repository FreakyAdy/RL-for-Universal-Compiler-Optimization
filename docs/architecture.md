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
