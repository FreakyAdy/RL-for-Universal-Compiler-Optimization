"""Corpus extraction tests."""

from pathlib import Path

from rl_uco.corpus.extract import _synthetic_corpus
from rl_uco.corpus.normalize import dedupe_records


def test_synthetic_corpus(tmp_path):
    records = _synthetic_corpus(tmp_path / "corp", count=5)
    assert len(records) == 5
    assert all(r.ir_path for r in records)


def test_dedupe(tmp_path):
    from rl_uco.corpus.models import FunctionRecord

    ir = tmp_path / "f.ll"
    ir.write_text("define i32 @f() { ret i32 0 }\n", encoding="utf-8")
    r1 = FunctionRecord("a", str(ir), "f", str(ir))
    r2 = FunctionRecord("b", str(ir), "f", str(ir))
    out = dedupe_records([r1, r2])
    assert len(out) == 1
