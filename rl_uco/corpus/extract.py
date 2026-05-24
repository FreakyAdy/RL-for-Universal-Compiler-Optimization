"""Extract functions from C/C++ sources into corpus entries."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import asdict
from pathlib import Path

import click

from rl_uco.config import MAX_FUNCTIONS_PER_FILE, MAX_IR_INSTRUCTIONS, Toolchain
from rl_uco.corpus.models import FunctionRecord
from rl_uco.corpus.normalize import dedupe_records, normalize_record
from rl_uco.ir.llvm_adapter import LLVMAdapter

# Simple regex fallback when libclang unavailable
_FUNC_RE = re.compile(
    r"^(?:static\s+)?(?:inline\s+)?(?:\w+\s+)+(\w+)\s*\([^;]*\)\s*\{",
    re.MULTILINE,
)


def _clang_extract(source: Path) -> list[str]:
    try:
        import clang.cindex as cx

        index = cx.Index.create()
        tu = index.parse(str(source), args=["-std=c11", "-Wno-everything"])
        names: list[str] = []
        for node in tu.cursor.walk():
            if node.kind == cx.CursorKind.FUNCTION_DECL and node.is_definition():
                if node.spelling and node.location.file and Path(
                    node.location.file.name,
                ).resolve() == source.resolve():
                    names.append(node.spelling)
        return names[:MAX_FUNCTIONS_PER_FILE]
    except Exception:
        return []


def _regex_extract(source: Path) -> list[str]:
    text = source.read_text(encoding="utf-8", errors="replace")
    names = _FUNC_RE.findall(text)
  # filter main
    return [n for n in names if n != "main"][:MAX_FUNCTIONS_PER_FILE]


def _make_single_function_tu(source: Path, fn: str, out: Path) -> bool:
    """Wrap source to expose only one function (best-effort)."""
    text = source.read_text(encoding="utf-8", errors="replace")
    wrapped = f"/* extracted: {fn} */\n{text}\n"
    out.write_text(wrapped, encoding="utf-8")
    return True


def _function_id(source: Path, fn: str) -> str:
    h = hashlib.sha256(f"{source.resolve()}:{fn}".encode()).hexdigest()[:16]
    return f"fn_{h}"


def extract_from_file(
    source: Path,
    output_dir: Path,
    llvm: LLVMAdapter,
    isa: str = "x86_64_v3",
) -> list[FunctionRecord]:
    output_dir.mkdir(parents=True, exist_ok=True)
    names = _clang_extract(source)
    if not names:
        names = _regex_extract(source)
    records: list[FunctionRecord] = []
    lang = "cpp" if source.suffix in (".cpp", ".cc", ".cxx") else "c"
    for fn in names:
        fid = _function_id(source, fn)
        work = output_dir / fid
        work.mkdir(exist_ok=True)
        tu = work / f"{fn}_tu{source.suffix}"
        _make_single_function_tu(source, fn, tu)
        ir_path = work / f"{fn}.ll"
        if not llvm.compile_to_ir(tu, ir_path, isa=isa):
            continue
        n_inst = llvm.count_instructions(ir_path)
        if n_inst > MAX_IR_INSTRUCTIONS or n_inst == 0:
            continue
        ir_text = ir_path.read_text(encoding="utf-8", errors="replace")
        has_loops = "br i1" in ir_text and "loop" in ir_text.lower()
        gpu = "__global__" in source.read_text(encoding="utf-8", errors="replace")
        rec = FunctionRecord(
            function_id=fid,
            source_path=str(source),
            function_name=fn,
            ir_path=str(ir_path),
            lang=lang,
            approx_ir_size=n_inst,
            has_loops=has_loops,
            is_gpu_candidate=gpu,
        )
        records.append(normalize_record(rec))
    return records


def extract_corpus(
    source_root: Path,
    output_dir: Path,
    isa: str = "x86_64_v3",
    extensions: tuple[str, ...] = (".c", ".cpp", ".cc"),
) -> list[FunctionRecord]:
    llvm = LLVMAdapter(Toolchain.discover())
    all_records: list[FunctionRecord] = []
    sources = sorted(
        p for p in source_root.rglob("*") if p.suffix in extensions and p.is_file()
    )
    for src in sources:
        all_records.extend(extract_from_file(src, output_dir, llvm, isa=isa))
    all_records = dedupe_records(all_records)
    manifest = output_dir / "manifest.json"
    manifest.write_text(
        json.dumps([asdict(r) for r in all_records], indent=2),
        encoding="utf-8",
    )
    return all_records


def _synthetic_corpus(output_dir: Path, count: int = 50) -> list[FunctionRecord]:
    """Generate synthetic functions when clang is unavailable."""
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[FunctionRecord] = []
    for i in range(count):
        fid = f"synth_{i:04d}"
        work = output_dir / fid
        work.mkdir(exist_ok=True)
        src = work / "fn.c"
        ir = work / "fn.ll"
        body = f"""
int synth_{i}(int n) {{
    int s = 0;
    for (int j = 0; j < n; j++) s += j * {i % 7 + 1};
    return s;
}}
"""
        src.write_text(body, encoding="utf-8")
        llvm = LLVMAdapter()
        ok = llvm.compile_to_ir(src, ir)
        if not ok:
            # Fallback minimal IR when clang is not installed
            ir.write_text(
                f"define i32 @synth_{i}(i32 %n) {{\n"
                f"entry:\n"
                f"  %s = add i32 %n, {i}\n"
                f"  ret i32 %s\n"
                f"}}\n",
                encoding="utf-8",
            )
            ok = True
        if ok:
            records.append(
                FunctionRecord(
                    function_id=fid,
                    source_path=str(src),
                    function_name=f"synth_{i}",
                    ir_path=str(ir),
                    approx_ir_size=llvm.count_instructions(ir),
                    has_loops=True,
                    license_tag="synthetic",
                )
            )
    manifest = output_dir / "manifest.json"
    manifest.write_text(json.dumps([asdict(r) for r in records], indent=2), encoding="utf-8")
    return records


@click.command()
@click.option("--source", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--isa", default="x86_64_v3")
@click.option("--synthetic", is_flag=True, help="Generate synthetic corpus if extraction fails")
@click.option("--synthetic-count", default=50)
def main(source: Path, output: Path, isa: str, synthetic: bool, synthetic_count: int) -> None:
    records = extract_corpus(source, output, isa=isa)
    if not records and synthetic:
        records = _synthetic_corpus(output, synthetic_count)
    click.echo(f"Extracted {len(records)} functions -> {output}")


if __name__ == "__main__":
    main()
