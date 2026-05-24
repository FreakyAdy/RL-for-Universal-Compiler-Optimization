"""Export dataset rows to versioned Parquet."""

from __future__ import annotations

from pathlib import Path

from rl_uco.data.schema import DatasetManifest, rows_to_dataframe, DatasetRow


def export_dataset(
    rows: list[DatasetRow],
    output_dir: Path,
    version: str = "v1",
    manifest: DatasetManifest | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"dataset_{version}.parquet"
    df = rows_to_dataframe(rows)
    df.to_parquet(out, index=False)
    if manifest:
        manifest.num_rows = len(rows)
        manifest.save(output_dir / "manifest.json")
    return out
