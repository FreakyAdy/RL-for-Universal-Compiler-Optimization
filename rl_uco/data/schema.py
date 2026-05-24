"""Dataset schema and manifest."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class DatasetRow:
    function_id: str
    ir_kind: str
    graph_path: str
    isa: str
    pass_sequence: list[dict]
    wall_time_ns: float
    energy_j: float
    baseline_wall_time_ns: float
    baseline_energy_j: float
    reward: float
    correct: bool
    dataset_version: str = "v1"
    policy_tag: str = "random"
    function_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["pass_sequence"] = json.dumps(self.pass_sequence)
        return d


@dataclass
class DatasetManifest:
    version: str
    llvm_version: str | None
    corpus_path: str
    num_rows: int
    isas: list[str]
    policy_mix: dict[str, float] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> DatasetManifest:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)


PARQUET_COLUMNS = [
    "function_id",
    "ir_kind",
    "graph_path",
    "isa",
    "pass_sequence",
    "wall_time_ns",
    "energy_j",
    "baseline_wall_time_ns",
    "baseline_energy_j",
    "reward",
    "correct",
    "dataset_version",
    "policy_tag",
    "function_name",
]


def rows_to_dataframe(rows: list[DatasetRow]) -> pd.DataFrame:
    return pd.DataFrame([r.to_dict() for r in rows], columns=PARQUET_COLUMNS)


def load_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["pass_sequence"] = df["pass_sequence"].apply(
        lambda s: json.loads(s) if isinstance(s, str) else s,
    )
    return df
