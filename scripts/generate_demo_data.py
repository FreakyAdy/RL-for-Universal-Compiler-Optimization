#!/usr/bin/env python3
"""Generate synthetic corpus + mock dataset for offline training without hardware."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl_uco.corpus.extract import _synthetic_corpus
from rl_uco.data.export_parquet import export_dataset
from rl_uco.data.schema import DatasetManifest, DatasetRow
from rl_uco.graph.llvm_to_graph import llvm_to_pyg, save_graph


def main() -> None:
    corpus_dir = ROOT / "data" / "corpus" / "demo"
    dataset_dir = ROOT / "data" / "datasets" / "demo_v1"
    graph_dir = dataset_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    records = _synthetic_corpus(corpus_dir, count=20)
    if not records:
        print("Warning: clang unavailable; no IR files generated.")
    rows: list[DatasetRow] = []
    for i, rec in enumerate(records):
        if not rec.ir_path:
            continue
        gpath = graph_dir / f"{rec.function_id}.pt"
        save_graph(llvm_to_pyg(Path(rec.ir_path)), gpath)
        base_t = 1000.0 + i * 10
        wall = base_t * (0.85 if i % 2 == 0 else 0.95)
        rows.append(
            DatasetRow(
                function_id=rec.function_id,
                ir_kind="llvm",
                graph_path=gpath.name,
                isa="x86_64_v3",
                pass_sequence=[
                    {"pass_id": 1, "name": "instcombine", "pipeline": "instcombine", "kind": "transform"},
                    {"pass_id": 3, "name": "gvn", "pipeline": "gvn", "kind": "transform"},
                ],
                wall_time_ns=wall,
                energy_j=wall / 1e6,
                baseline_wall_time_ns=base_t,
                baseline_energy_j=base_t / 1e6,
                reward=-(0.7 * wall / base_t + 0.3 * wall / base_t),
                correct=True,
                dataset_version="demo_v1",
                policy_tag="synthetic",
                function_name=rec.function_name,
            ),
        )

    manifest = DatasetManifest(
        version="demo_v1",
        llvm_version="synthetic",
        corpus_path=str(corpus_dir),
        num_rows=len(rows),
        isas=["x86_64_v3"],
    )
    export_dataset(rows, dataset_dir, version="demo_v1", manifest=manifest)
    print(f"Wrote {len(rows)} rows to {dataset_dir}")


if __name__ == "__main__":
    main()
