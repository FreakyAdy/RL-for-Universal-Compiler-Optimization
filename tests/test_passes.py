"""Tests for pass registry and validator."""

from rl_uco.passes.registry import load_registry, PassAction
from rl_uco.passes.validator import PassSequenceValidator


def test_registry_loads():
    reg = load_registry()
    assert len(reg.passes) >= 30
    assert reg.stop_action_id == 0


def test_pipeline_string():
    reg = load_registry()
    seq = [
        PassAction(1, "instcombine", "instcombine", "transform"),
        PassAction(3, "gvn", "gvn", "transform"),
    ]
    pipe = reg.pipeline_string(seq)
    assert "instcombine" in pipe
    assert "gvn" in pipe


def test_validator_accepts_transforms():
    reg = load_registry()
    v = PassSequenceValidator(reg)
    seq = [PassAction(1, "instcombine", "instcombine", "transform")]
    r = v.validate(seq)
    assert r.valid
    assert r.pipeline == "instcombine"


def test_random_sequence():
    reg = load_registry()
    seq = reg.random_sequence(length=3)
    assert len(seq) == 3
