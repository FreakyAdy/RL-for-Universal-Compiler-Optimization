"""Corpus data models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FunctionRecord:
    function_id: str
    source_path: str
    function_name: str
    ir_path: str | None = None
    lang: str = "c"
    approx_ir_size: int = 0
    has_loops: bool = False
    is_gpu_candidate: bool = False
    license_tag: str = "unknown"
