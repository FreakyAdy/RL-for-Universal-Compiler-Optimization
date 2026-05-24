# Dataset schema

Parquet datasets use the schema defined in `rl_uco/data/schema.py`.

## Row fields

| Field | Type | Description |
|-------|------|-------------|
| `function_id` | string | Stable hash ID |
| `ir_kind` | string | `llvm` or `mlir` |
| `graph_path` | string | Relative path to `.pt` PyG graph |
| `isa` | string | Target ISA key |
| `pass_sequence` | JSON list | Structured pass actions |
| `wall_time_ns` | float64 | Median execution time |
| `energy_j` | float64 | Median energy (joules) |
| `baseline_wall_time_ns` | float64 | Baseline for normalization |
| `baseline_energy_j` | float64 | Baseline energy |
| `reward` | float64 | Computed reward |
| `correct` | bool | Passed correctness gate |
| `dataset_version` | string | e.g. `v1` |
| `policy_tag` | string | Bootstrap policy name |

## Versioning

Datasets live under `data/datasets/<version>/`. Metadata in `manifest.json` records LLVM version, corpus hash, and collection config.
