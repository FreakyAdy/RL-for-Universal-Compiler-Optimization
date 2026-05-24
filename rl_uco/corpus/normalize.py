"""Corpus normalization and deduplication."""

from __future__ import annotations

import hashlib
from pathlib import Path

from rl_uco.corpus.models import FunctionRecord
from rl_uco.config import MAX_IR_INSTRUCTIONS


def _bitcode_hash(ir_path: Path) -> str:
    text = ir_path.read_text(encoding="utf-8", errors="replace")
    # Strip comments and whitespace for approximate semantic dedupe
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(";"):
            continue
        lines.append(s)
    normalized = "\n".join(lines)
    return hashlib.sha256(normalized.encode()).hexdigest()


def normalize_record(rec: FunctionRecord) -> FunctionRecord:
    if rec.ir_path and rec.approx_ir_size == 0:
        from rl_uco.ir.llvm_adapter import LLVMAdapter

        rec.approx_ir_size = LLVMAdapter().count_instructions(Path(rec.ir_path))
    if rec.approx_ir_size > MAX_IR_INSTRUCTIONS:
        rec.ir_path = None
    return rec


def dedupe_records(records: list[FunctionRecord]) -> list[FunctionRecord]:
    seen: set[str] = set()
    out: list[FunctionRecord] = []
    for rec in records:
        if not rec.ir_path:
            continue
        h = _bitcode_hash(Path(rec.ir_path))
        if h in seen:
            continue
        seen.add(h)
        out.append(rec)
    return out


def normalize_corpus(records: list[FunctionRecord]) -> list[FunctionRecord]:
    return dedupe_records([normalize_record(r) for r in records])
