"""Read-only EMG discovery and deliberately permissive diagnostic input."""

from __future__ import annotations

import csv
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..pipeline.meta import find_session_root
from ..pipeline.retime import METADATA_COLUMNS


def discover(patterns: tuple[str, ...]) -> list[Path]:
    found: set[Path] = set()
    for pattern in patterns:
        matches = (
            [Path(pattern)]
            if Path(pattern).exists()
            else [Path(p) for p in glob.glob(pattern, recursive=True)]
        )
        if not matches:
            raise ValueError(f"no inputs match: {pattern}")
        for path in matches:
            candidates = path.rglob("*") if path.is_dir() else [path]
            for item in candidates:
                if not item.is_file() or item.suffix.lower() not in {".csv", ".parquet", ".pq"}:
                    continue
                if ".denoised." in item.name or ".mask." in item.name:
                    continue
                if path.is_dir() and "emg" not in item.stem.lower():
                    continue
                found.add(item.resolve())
    if not found:
        raise ValueError("no EMG source files found")
    return sorted(found)


def read_columns(path: Path) -> dict[str, Any]:
    """Keep null/nonfinite values for diagnosis; do not invoke processing validators."""
    if ".parquet" in path.suffixes or ".pq" in path.suffixes:
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise ValueError("reading parquet requires emg-denoise[parquet]") from error
        table: Any = pq.read_table(path)  # type: ignore[no-untyped-call]
        if len(set(table.column_names)) != len(table.column_names):
            raise ValueError("duplicate column names")
        return {name: table[name].to_numpy(zero_copy_only=False) for name in table.column_names}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        if not header or len(set(header)) != len(header) or any(not h.strip() for h in header):
            raise ValueError("missing or duplicate CSV columns")
        rows = [row for row in reader if row]
        if any(len(row) != len(header) for row in rows):
            raise ValueError("CSV row width differs from header")
        return {
            name: np.asarray([row[i] if row[i] else None for row in rows], dtype=object)
            for i, name in enumerate(header)
        }


def channel_names(columns: dict[str, Any]) -> list[str]:
    names = [n for n in columns if n not in METADATA_COLUMNS and n != "sample_index"]
    if names and all(n.startswith("dim_") and n[4:].isdigit() for n in names):
        names.sort(key=lambda n: int(n[4:]))
    return names


def metadata(path: Path) -> tuple[dict[str, Any], dict[str, Any], Path | None]:
    root = find_session_root(path)
    if root is None:
        return {}, {}, None
    info = json.loads((root / "meta/info.json").read_text())
    feature = info.get("features", {}).get(path.stem, {})
    return info, feature, root
