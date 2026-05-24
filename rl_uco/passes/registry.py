"""Load and query pass registry."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REGISTRY_PATH = Path(__file__).with_name("registry.yaml")


@dataclass
class PassAction:
    """Structured pass action (JSON-serializable)."""

    pass_id: int
    name: str
    pipeline: str
    kind: str = "transform"
    adaptor_inner: list[PassAction] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "pass_id": self.pass_id,
            "name": self.name,
            "pipeline": self.pipeline,
            "kind": self.kind,
        }
        if self.adaptor_inner:
            d["adaptor_inner"] = [a.to_dict() for a in self.adaptor_inner]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PassAction:
        inner = [cls.from_dict(x) for x in d.get("adaptor_inner", [])]
        return cls(
            pass_id=int(d["pass_id"]),
            name=str(d["name"]),
            pipeline=str(d["pipeline"]),
            kind=str(d.get("kind", "transform")),
            adaptor_inner=inner,
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class PassRegistry:
    raw: dict[str, Any]
    passes: dict[int, PassAction]
    adaptors: dict[int, PassAction]
    mlir_passes: dict[int, PassAction]
    stop_action_id: int = 0
    max_episode_steps: int = 12
    baselines: dict[str, str] = field(default_factory=dict)

    @property
    def num_actions(self) -> int:
        return 1 + len(self.passes) + len(self.adaptors)  # +STOP

    def get_by_id(self, pass_id: int) -> PassAction | None:
        if pass_id == self.stop_action_id:
            return PassAction(self.stop_action_id, "STOP", "", "stop")
        return self.passes.get(pass_id) or self.adaptors.get(pass_id)

    def pipeline_string(self, sequence: list[PassAction]) -> str:
        parts: list[str] = []
        for action in sequence:
            if action.pass_id == self.stop_action_id or action.name == "STOP":
                break
            if action.kind == "adaptor" and action.adaptor_inner:
                inner = ",".join(a.pipeline for a in action.adaptor_inner)
                parts.append(action.pipeline.format(inner=inner))
            else:
                parts.append(action.pipeline)
        return ",".join(parts)

    def random_sequence(
        self,
        length: int | None = None,
        rng: random.Random | None = None,
    ) -> list[PassAction]:
        rng = rng or random.Random()
        length = length or rng.randint(1, self.max_episode_steps)
        ids = list(self.passes.keys())
        seq: list[PassAction] = []
        for _ in range(length):
            pid = rng.choice(ids)
            p = self.passes[pid]
            seq.append(
                PassAction(p.pass_id, p.name, p.pipeline, p.kind),
            )
        return seq

    def mutate_sequence(
        self,
        sequence: list[PassAction],
        rng: random.Random | None = None,
    ) -> list[PassAction]:
        rng = rng or random.Random()
        seq = [
            PassAction(a.pass_id, a.name, a.pipeline, a.kind, list(a.adaptor_inner))
            for a in sequence
        ]
        if not seq:
            return self.random_sequence(rng=rng)
        op = rng.choice(["swap", "insert", "delete", "replace"])
        ids = list(self.passes.keys())
        if op == "swap" and len(seq) >= 2:
            i, j = rng.sample(range(len(seq)), 2)
            seq[i], seq[j] = seq[j], seq[i]
        elif op == "insert":
            p = self.passes[rng.choice(ids)]
            seq.insert(rng.randint(0, len(seq)), PassAction(p.pass_id, p.name, p.pipeline, p.kind))
        elif op == "delete" and len(seq) > 1:
            del seq[rng.randint(0, len(seq) - 1)]
        else:
            i = rng.randint(0, len(seq) - 1)
            p = self.passes[rng.choice(ids)]
            seq[i] = PassAction(p.pass_id, p.name, p.pipeline, p.kind)
        return seq[: self.max_episode_steps]

    def baseline_sequence(self, level: str) -> list[PassAction]:
        pipe = self.baselines.get(level, self.baselines.get("O3", "default<O3>"))
        return [
            PassAction(-1, f"baseline_{level}", pipe, "baseline"),
        ]

    def mlir_pipeline_string(self, sequence: list[PassAction]) -> str:
        parts = []
        for action in sequence:
            if action.name == "STOP":
                break
            parts.append(action.pipeline)
        return ",".join(parts)


def load_registry(path: Path | None = None) -> PassRegistry:
    path = path or REGISTRY_PATH
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    def _load_entries(entries: list[dict]) -> dict[int, PassAction]:
        out: dict[int, PassAction] = {}
        for e in entries:
            out[int(e["id"])] = PassAction(
                int(e["id"]),
                str(e["name"]),
                str(e["pipeline"]),
                str(e.get("kind", "transform")),
            )
        return out

    return PassRegistry(
        raw=raw,
        passes=_load_entries(raw.get("passes", [])),
        adaptors=_load_entries(raw.get("adaptors", [])),
        mlir_passes=_load_entries(raw.get("mlir_passes", [])),
        stop_action_id=int(raw.get("stop_action_id", 0)),
        max_episode_steps=int(raw.get("max_episode_steps", 12)),
        baselines=dict(raw.get("default_baselines", {})),
    )
