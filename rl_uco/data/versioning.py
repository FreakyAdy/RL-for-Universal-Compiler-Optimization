"""Dataset versioning utilities."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from rl_uco.data.schema import DatasetManifest


def bump_version(base_dir: Path, new_version: str) -> Path:
    """Create a new dataset directory version with manifest stub."""
    out = base_dir / new_version
    out.mkdir(parents=True, exist_ok=True)
    manifest = DatasetManifest(
        version=new_version,
        llvm_version=None,
        corpus_path="",
        num_rows=0,
        isas=[],
    )
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent": str(base_dir),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    manifest.save(out / "manifest.json")
    return out


def copy_dataset(src: Path, dst_version: str) -> Path:
    """Copy Parquet + graphs to a new versioned folder."""
    dst = src.parent / dst_version
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    manifest_path = dst / "manifest.json"
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        m["version"] = dst_version
        manifest_path.write_text(json.dumps(m, indent=2), encoding="utf-8")
    return dst
