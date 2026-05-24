"""Evaluation: baselines, geo-mean speedup, energy reduction."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import click
import pandas as pd

from rl_uco.data.schema import load_parquet
from rl_uco.env.compile_run import CompileRunEnv
from rl_uco.passes.registry import PassAction, load_registry
from rl_uco.rl.inference import InferenceEngine


def geo_mean(values: list[float]) -> float:
    if not values:
        return 1.0
    log_sum = sum(math.log(max(v, 1e-9)) for v in values)
    return math.exp(log_sum / len(values))


def speedup_from_row(row: pd.Series) -> float:
    if row["wall_time_ns"] <= 0 or row["baseline_wall_time_ns"] <= 0:
        return 1.0
    return row["baseline_wall_time_ns"] / row["wall_time_ns"]


def energy_ratio_from_row(row: pd.Series) -> float:
    if row["energy_j"] <= 0 or row["baseline_energy_j"] <= 0:
        return 1.0
    return row["baseline_energy_j"] / row["energy_j"]


def eval_o3_baseline(env: CompileRunEnv, ir_path: Path, fn: str, isa: str):
    seq = env.registry.baseline_sequence("O3")
    return env.run(ir_path, fn, seq, isa=isa)


def eval_random_baseline(env: CompileRunEnv, ir_path: Path, fn: str, isa: str, rng: random.Random):
    seq = env.registry.random_sequence(rng=rng)
    return env.run(ir_path, fn, seq, isa=isa)


def eval_coreset_bc(df: pd.DataFrame) -> dict[str, float]:
    """Reproduce coreset-style BC: pick best logged sequence per function."""
    speedups: list[float] = []
    grouped = df.groupby("function_id")
    for _, group in grouped:
        best = group.loc[group["reward"].idxmax()]
        speedups.append(speedup_from_row(best))
    return {"geo_mean_speedup": geo_mean(speedups), "method": "coreset_bc"}


def run_evaluation(
    dataset_path: Path,
    checkpoint: Path | None = None,
    corpus_dir: Path | None = None,
    isa: str = "x86_64_v3",
    max_functions: int = 50,
) -> dict:
    df = load_parquet(dataset_path)
    df = df[df["correct"] == True]  # noqa: E712
    if max_functions:
        df = df.head(max_functions)

    results: dict[str, dict] = {}

    # Dataset oracle / coreset BC
    results["coreset_bc"] = eval_coreset_bc(df)

    # Logged policy tags
    for tag in df["policy_tag"].unique():
        sub = df[df["policy_tag"] == tag]
        speedups = [speedup_from_row(r) for _, r in sub.iterrows()]
        results[f"logged_{tag}"] = {
            "geo_mean_speedup": geo_mean(speedups),
            "n": len(speedups),
        }

    # O3 from live runs (subset)
    if corpus_dir and corpus_dir.exists():
        manifest = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
        env = CompileRunEnv()
        o3_speedups: list[float] = []
        policy_speedups: list[float] = []
        rng = random.Random(42)
        for entry in manifest[: min(10, len(manifest))]:
            ir = Path(entry["ir_path"])
            fn = entry["function_name"]
            if not ir.exists():
                continue
            r = eval_o3_baseline(env, ir, fn, isa)
            if r.success and r.correct:
                o3_speedups.append(r.baseline_wall_time_ns / max(r.wall_time_ns, 1))
            rr = eval_random_baseline(env, ir, fn, isa, rng)
            if rr.success and rr.correct:
                policy_speedups.append(rr.baseline_wall_time_ns / max(rr.wall_time_ns, 1))
        results["live_O3"] = {"geo_mean_speedup": geo_mean(o3_speedups), "n": len(o3_speedups)}
        results["live_random"] = {"geo_mean_speedup": geo_mean(policy_speedups), "n": len(policy_speedups)}

    if checkpoint and checkpoint.exists():
        engine = InferenceEngine(checkpoint)
        inf_speedups: list[float] = []
        for _, row in df.head(10).iterrows():
            gdir = dataset_path.parent / "graphs"
            # Need ir from corpus — skip if unavailable
        results["learned_policy"] = {"note": "run rl-uco-infer on corpus IR files"}

    summary = {
        "dataset": str(dataset_path),
        "isa": isa,
        "results": results,
    }
    return summary


@click.command()
@click.option("--dataset", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--checkpoint", type=click.Path(path_type=Path), default=None)
@click.option("--corpus", type=click.Path(path_type=Path), default=None)
@click.option("--isa", default="x86_64_v3")
@click.option("--output", type=click.Path(path_type=Path), default=None)
def main(
    dataset: Path,
    checkpoint: Path | None,
    corpus: Path | None,
    isa: str,
    output: Path | None,
) -> None:
    summary = run_evaluation(dataset, checkpoint, corpus, isa=isa)
    text = json.dumps(summary, indent=2)
    click.echo(text)
    if output:
        output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
