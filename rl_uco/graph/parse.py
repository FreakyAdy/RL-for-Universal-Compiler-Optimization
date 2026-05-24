"""LLVM IR parsing helpers."""

from __future__ import annotations

import re

OPCODES = [
    "alloca", "load", "store", "add", "sub", "mul", "udiv", "sdiv",
    "urem", "srem", "shl", "lshr", "ashr", "and", "or", "xor",
    "icmp", "fcmp", "phi", "call", "br", "ret", "getelementptr",
    "bitcast", "zext", "sext", "trunc", "select", "other",
]
OP_TO_IDX = {op: i for i, op in enumerate(OPCODES)}


def parse_instructions(ir_text: str) -> list[tuple[str, str, list[str]]]:
    """Return list of (var, opcode, use_vars)."""
    instrs: list[tuple[str, str, list[str]]] = []
    var_pat = re.compile(r"%([a-zA-Z0-9_.]+)")
    for line in ir_text.splitlines():
        s = line.strip()
        if not s or s.startswith(";"):
            continue
        if s.endswith(":") and not s.startswith("%"):
            continue
        m = re.match(r"(%[\w.]+)\s*=\s*(\w+)", s)
        if m:
            var, op = m.group(1), m.group(2)
            uses = var_pat.findall(s)
            instrs.append((var, op if op in OP_TO_IDX else "other", uses))
        elif s.startswith(("br ", "ret ", "store ", "call ")):
            op = s.split()[0]
            instrs.append(("", op if op in OP_TO_IDX else "other", var_pat.findall(s)))
    return instrs
