"""Validated recording I/O and atomic file replacement at the application boundary."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import numpy as np

from ._typing import FloatArray
from .dsp.filters import FS, sample_matrix

REPORT_COLUMNS = frozenset({"valid", "segment_id"})


@dataclass(frozen=True, slots=True)
class Recording:
    """An in-memory, contiguous recording used for offline helpers."""

    timestamps: FloatArray
    samples: FloatArray
    channel_names: tuple[str, ...]


def read_header(reader: Iterator[list[str]]) -> list[str]:
    try:
        header = next(reader)
    except StopIteration as error:
        raise ValueError("CSV is empty; expected timestamp and channel columns") from error
    if len(header) < 2 or header[0] != "timestamp" or any(not name.strip() for name in header):
        raise ValueError("CSV header must be timestamp,channel1[,channel2,...]")
    if len(set(header)) != len(header):
        raise ValueError("CSV column names must be unique")
    if REPORT_COLUMNS.intersection(header):
        raise ValueError("input channel names cannot use reserved output metadata columns")
    return header


def parse_row(row: list[str], columns: int, line_number: int) -> list[float]:
    if len(row) != columns:
        raise ValueError(f"CSV line {line_number}: expected {columns} columns, received {len(row)}")
    try:
        values = [float(value) for value in row]
    except ValueError as error:
        raise ValueError(f"CSV line {line_number}: all values must be numeric") from error
    if not np.isfinite(values).all():
        raise ValueError(f"CSV line {line_number}: all values must be finite")
    return values


def load_recording(path: str | Path, sample_rate: float = FS) -> Recording:
    with Path(path).open(newline="", encoding="utf-8-sig") as source:
        reader = csv.reader(source)
        header = read_header(reader)
        rows = [parse_row(row, len(header), reader.line_num) for row in reader if row]
    if not rows:
        raise ValueError("CSV contains no samples")
    data = np.asarray(rows, dtype=np.float64)
    timestamps = data[:, 0]
    period = 1 / float(sample_rate)
    if not np.all(np.abs(np.diff(timestamps) - period) <= 1e-6):
        raise ValueError(f"recording must have contiguous {sample_rate:g} Hz timestamps")
    return Recording(timestamps, sample_matrix(data[:, 1:]), tuple(header[1:]))


def validate_output_paths(inputs: Sequence[Path], outputs: Sequence[Path]) -> None:
    """Prevent source clobbering, including symlink and existing hardlink aliases."""
    for i, output in enumerate(outputs):
        for other in [*inputs, *outputs[:i]]:
            same = output.resolve() == other.resolve()
            if output.exists() and other.exists():
                same = same or output.samefile(other)
            if same:
                raise ValueError("input, output, report and feature paths must be distinct")


@contextmanager
def atomic_text_writer(path: str | Path) -> Iterator[IO[str]]:
    """A failed write leaves any existing destination intact; temp files are removed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def jsonable(value: object) -> Any:
    """Convert a heterogeneous report into strict JSON, replacing nonfinite numbers."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("JSON report keys must be strings")
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    raise TypeError(f"unsupported JSON report value: {type(value).__name__}")


def dump_json(path: str | Path, data: Mapping[str, Any]) -> None:
    payload = json.dumps(jsonable(data), ensure_ascii=False, indent=2, allow_nan=False)
    with atomic_text_writer(path) as handle:
        handle.write(payload + "\n")


def load_json_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON document must contain an object")
    return value
