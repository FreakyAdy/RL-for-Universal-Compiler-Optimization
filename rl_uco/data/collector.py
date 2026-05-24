"""Distributed dataset collection with Ray."""

from __future__ import annotations

import json
import random
from pathlib import Path

import click

from rl_uco.config import GRAPH_DIR, Toolchain
from rl_uco.corpus.models import FunctionRecord
from rl_uco.data.export_parquet import export_dataset
from rl_uco.data.schema import DatasetManifest, DatasetRow
from rl_uco.env.compile_run import CompileRunEnv
from rl_uco.graph.llvm_to_graph import llvm_to_pyg, save_graph
from rl_uco.passes.registry import PassAction, load_registry


def _load_corpus(corpus_dir: Path) -> list[FunctionRecord]:
    manifest = corpus_dir / "manifest.json"
    if not manifest.exists():
        return []
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return [
        FunctionRecord(**{k: v for k, v in row.items() if k in FunctionRecord.__dataclass_fields__})
        for row in data
    ]


def _collect_one(
  function_id: str,
  ir_path: str,
  function_name: str,
  isa: str,
  policy_tag: str,
  graph_dir: str,
  version: str,
  seed_seq: list[dict] | None,
) -> DatasetRow | None:
    env = CompileRunEnv()
    registry = load_registry()
    rng = random.Random(hash(function_id) % (2**32))
    seed_actions = None
    if seed_seq:
        seed_actions = [PassAction.from_dict(d) for d in seed_seq]
    seq = env.sample_pass_sequence(policy_tag, seed_sequence=seed_actions, rng=rng)
    result = env.run(Path(ir_path), function_name, seq, isa=isa)
    if not result.success:
        return None

    gpath = Path(graph_dir) / f"{function_id}.pt"
    if not gpath.exists():
        save_graph(llvm_to_pyg(Path(ir_path)), gpath)

    rel_graph = str(gpath.name)
    return DatasetRow(
        function_id=function_id,
        ir_kind="llvm",
        graph_path=rel_graph,
        isa=isa,
        pass_sequence=result.pass_sequence,
        wall_time_ns=result.wall_time_ns,
        energy_j=result.energy_j,
        baseline_wall_time_ns=result.baseline_wall_time_ns,
        baseline_energy_j=result.baseline_energy_j,
        reward=result.reward,
        correct=result.correct,
        dataset_version=version,
        policy_tag=policy_tag,
        function_name=function_name,
    )


def collect_dataset(
    corpus_dir: Path,
    output_dir: Path,
    isa: str = "x86_64_v3",
    version: str = "v1",
    num_workers: int = 4,
    max_rows: int | None = None,
    graph_dir: Path | None = None,
) -> list[DatasetRow]:
    graph_dir = graph_dir or GRAPH_DIR
    graph_dir.mkdir(parents=True, exist_ok=True)
    records = _load_corpus(corpus_dir)
    if not records:
        return []

    env = CompileRunEnv()
    policies = list(env.bootstrap_policies().keys())
    weights = list(env.bootstrap_policies().values())
    jobs: list[tuple] = []
    top_sequences: list[list[dict]] = []

    for rec in records:
        if not rec.ir_path:
            continue
        policy_tag = random.choices(policies, weights=weights, k=1)[0]
        seed = random.choice(top_sequences) if policy_tag == "mutate" and top_sequences else None
        jobs.append(
            (rec.function_id, rec.ir_path, rec.function_name, isa, policy_tag, str(graph_dir), version, seed),
        )

    if max_rows:
        jobs = jobs[:max_rows]

    rows: list[DatasetRow] = []

    if num_workers > 1:
        try:
            import ray

            ray.init(ignore_reinit_error=True, num_cpus=num_workers)

            @ray.remote
            def remote_collect(job):
                return _collect_one(*job)

            futures = [remote_collect.remote(j) for j in jobs]
            for fut in ray.get(futures):
                if fut and fut.correct:
                    rows.append(fut)
                    if fut.reward > -0.5:
                        top_sequences.append(fut.pass_sequence)
            ray.shutdown()
        except Exception:
            num_workers = 1

    if num_workers <= 1 or not rows:
        for job in jobs:
            row = _collect_one(*job)
            if row:
                rows.append(row)
                if row.correct and row.reward > -0.5:
                    top_sequences.append(row.pass_sequence)

    tc = Toolchain.discover()
    manifest = DatasetManifest(
        version=version,
        llvm_version=tc.llvm_version,
        corpus_path=str(corpus_dir),
        num_rows=len(rows),
        isas=[isa],
        policy_mix=env.bootstrap_policies(),
    )
    if rows:
        export_dataset(rows, output_dir, version=version, manifest=manifest)
    else:
        manifest.save(output_dir / "manifest.json")
    return rows


@click.command()
@click.option("--corpus", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--isa", default="x86_64_v3")
@click.option("--version", default="v1")
@click.option("--num-workers", default=4)
@click.option("--max-rows", default=None, type=int)
def main(
    corpus: Path,
    output: Path,
    isa: str,
    version: str,
    num_workers: int,
    max_rows: int | None,
) -> None:
    rows = collect_dataset(
        corpus,
        output,
        isa=isa,
        version=version,
        num_workers=num_workers,
        max_rows=max_rows,
    )
    click.echo(f"Collected {len(rows)} rows -> {output}")


if __name__ == "__main__":
    main()
