#!/usr/bin/env python3
"""RL-UCO Quickstart Demo — generate data, train, and run inference in one command.

Usage:
    python scripts/demo.py

No LLVM toolchain required. Generates synthetic LLVM IR, creates a mock
Parquet dataset, trains an IQL agent for a few epochs, and runs inference
on a sample function.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    import torch

    from rl_uco.corpus.extract import _synthetic_corpus
    from rl_uco.data.export_parquet import export_dataset
    from rl_uco.data.schema import DatasetManifest, DatasetRow
    from rl_uco.graph.llvm_to_graph import llvm_to_pyg, save_graph
    from rl_uco.graph.parse import OPCODES
    from rl_uco.passes.registry import load_registry
    from rl_uco.rl.actor_critic import ActorCriticAgent
    from rl_uco.rl.offline_trainer import IQLTrainer, OfflinePassDataset, collate_batch

    corpus_dir = ROOT / "data" / "corpus" / "demo"
    dataset_dir = ROOT / "data" / "datasets" / "demo_v1"
    graph_dir = dataset_dir / "graphs"
    ckpt_dir = ROOT / "checkpoints"

    # ── Step 1: Generate synthetic corpus ──────────────────────────────────
    print("=" * 60)
    print("Step 1/3: Generating synthetic corpus...")
    print("=" * 60)
    graph_dir.mkdir(parents=True, exist_ok=True)
    records = _synthetic_corpus(corpus_dir, count=20)
    print(f"  Created {len(records)} synthetic functions")

    # ── Step 2: Build mock dataset and train ───────────────────────────────
    print()
    print("=" * 60)
    print("Step 2/3: Building dataset and training IQL agent...")
    print("=" * 60)
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
                    {
                        "pass_id": 1,
                        "name": "instcombine",
                        "pipeline": "instcombine",
                        "kind": "transform",
                    },
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
    pq_path = export_dataset(rows, dataset_dir, version="demo_v1", manifest=manifest)
    print(f"  Wrote {len(rows)} rows to {pq_path}")

    # Train
    from torch.utils.data import DataLoader

    ds = OfflinePassDataset(pq_path, graph_dir)
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_batch)
    registry = load_registry()
    agent = ActorCriticAgent(len(OPCODES), registry.num_actions)
    trainer = IQLTrainer(agent, lr=3e-4)

    epochs = 10
    for ep in range(epochs):
        metrics = trainer.train_epoch(loader)
        print(f"  epoch {ep + 1}/{epochs}: loss={metrics['loss']:.4f}")

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / "best.pt"
    torch.save({"model": agent.state_dict(), "num_actions": registry.num_actions}, ckpt)
    print(f"  Saved checkpoint: {ckpt}")

    # ── Step 3: Inference ──────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Step 3/3: Running inference on a sample function...")
    print("=" * 60)
    from rl_uco.rl.inference import InferenceEngine

    engine = InferenceEngine(ckpt)
    sample_ir = Path(records[0].ir_path)
    seq = engine.propose_sequence(sample_ir, isa="x86_64_v3")
    pass_names = [a.name for a in seq]
    print(f"  Input:  {sample_ir.name}")
    print(f"  Proposed pass sequence ({len(seq)} passes): {pass_names}")
    print(f"  Pipeline string: {registry.pipeline_string(seq)}")

    print()
    print("=" * 60)
    print("[OK] Demo complete!")
    print()
    print("Next steps:")
    print("  - With LLVM 18+: use real corpus with `rl-uco-extract` + `rl-uco-collect`")
    print(
        "  - Evaluate: `rl-uco-eval --dataset data/datasets/demo_v1 --checkpoint checkpoints/best.pt`"
    )
    print("  - Inference: `rl-uco-infer --ir <file.ll> --checkpoint checkpoints/best.pt`")
    print("=" * 60)


if __name__ == "__main__":
    main()
