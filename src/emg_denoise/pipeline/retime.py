"""Application service for stage one: rebuild a recording's time axis.

Reads an acquisition parquet (or CSV), replaces the tick-quantized timestamps
with endpoint-preserving, piecewise interpolated intervals, and writes either:

- a ``timestamp,channel...`` CSV or parquet for stage two (``emg-denoise process``), or
- a parquet that keeps the acquisition schema (``--inplace`` or ``.parquet`` output).

In-place mode renames the source to ``<source>.bak`` and writes the rebuilt file
back under the original name.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .._typing import FloatArray
from ..dsp.filters import FS, assert_emg_recording, sample_matrix
from ..io import atomic_text_writer, dump_json, validate_output_paths
from .timebase import TimebaseReport, rebuild_timestamps

#: Acquisition files name the time column this way; channels are ``dim_0``, ``dim_1``, ...
TIMESTAMP_COLUMN = "timestamp"
CHANNEL_PREFIX = "dim_"

#: Columns carried by acquisition parquet that are not signal channels.
#: ``valid`` / ``segment_id`` are reserved denoise QC columns (legacy outputs);
#: never treat them as signal channels for upstream LeRobot readers.
METADATA_COLUMNS = frozenset(
    {
        TIMESTAMP_COLUMN,
        "episode_index",
        "frame_index",
        "index",
        "valid",
        "segment_id",
    }
)


@dataclass(frozen=True, slots=True)
class RetimeOptions:
    """Paths and parameters for one stage-one run; sample values are never altered."""

    source: Path
    output: Path | None = None
    rate_hz: float = float(FS)
    method: str = "anchor_interpolation"
    tick_us: float | None = None
    non_negative: bool = True
    #: Legacy uniform mode only: stitch fixed-period centroid segments.
    stitch: bool = True
    #: Rename ``source`` to ``source.bak`` and write the rebuild as ``source``.
    inplace: bool = False
    #: Allow overwriting an existing ``.bak`` refusal when re-running in place,
    #: and allow replacing an existing non-inplace output.
    force: bool = False
    #: Write ``<output>.timebase.json`` beside the recording (off by default).
    sidecar: bool = False
    #: Merge diagnostics into the enclosing LeRobot ``meta/info.json`` when present.
    update_meta: bool = True

    @property
    def destination(self) -> Path:
        if self.inplace:
            return Path(self.source)
        if self.output is None:
            raise ValueError("output is required unless --inplace is set")
        return Path(self.output)

    @property
    def backup_path(self) -> Path:
        return Path(f"{self.source}.bak")

    @property
    def report_path(self) -> Path:
        return Path(f"{self.destination}.timebase.json")

    @property
    def period_s(self) -> float:
        if self.rate_hz <= 0.0:
            raise ValueError("rate must be positive")
        return 1.0 / self.rate_hz

    def validate_paths(self) -> None:
        source = Path(self.source)
        if not source.is_file():
            raise ValueError(f"input not found: {source}")
        if self.inplace:
            if source.suffix.lower() not in {".parquet", ".pq"}:
                raise ValueError("--inplace requires a parquet source")
            if self.backup_path.exists() and not self.force:
                raise ValueError(
                    f"backup exists; pass --force to rebuild without replacing it: {self.backup_path}"
                )
            return
        if self.output is None:
            raise ValueError("output is required unless --inplace is set")
        outputs = [self.destination]
        if self.sidecar:
            outputs.append(self.report_path)
        validate_output_paths([source], outputs)


@dataclass(frozen=True, slots=True)
class Acquisition:
    """Raw recording as stored by the acquisition stack, before retiming."""

    timestamps: FloatArray
    samples: FloatArray
    channel_names: tuple[str, ...]
    #: Full column order for parquet rewrite; CSV fills this with timestamp+channels.
    column_names: tuple[str, ...] = ()
    #: Non-channel columns preserved for parquet rewrite (numpy arrays).
    extras: dict[str, Any] = field(default_factory=dict)
    #: Arrow/numpy dtype hint per channel name for parquet rewrite.
    channel_dtypes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.column_names:
            object.__setattr__(
                self,
                "column_names",
                (TIMESTAMP_COLUMN, *self.channel_names),
            )


def _channel_order(names: list[str]) -> list[str]:
    """Signal columns in acquisition order, with ``dim_N`` sorted numerically."""
    channels = [name for name in names if name not in METADATA_COLUMNS]
    if not channels:
        raise ValueError("recording has no channel columns")
    dims = [name for name in channels if name.startswith(CHANNEL_PREFIX)]
    if len(dims) == len(channels):
        # dim_10 must follow dim_9, which lexicographic order would not do.
        return sorted(dims, key=lambda name: int(name.removeprefix(CHANNEL_PREFIX)))
    return channels


def _is_parquet_path(path: Path) -> bool:
    """True for ``.parquet`` / ``.pq``, including a ``.bak`` sibling of either."""
    name = path.name.lower()
    if name.endswith(".bak"):
        name = name[: -len(".bak")]
    return name.endswith(".parquet") or name.endswith(".pq")


def load_acquisition(path: str | Path) -> Acquisition:
    """Load a parquet or CSV recording without validating its sample spacing.

    Unlike :func:`emg_denoise.io.load_recording`, this accepts the non-uniform
    timestamps that stage one exists to repair.
    """
    source = Path(path)
    reader = _read_parquet if _is_parquet_path(source) else _read_csv
    acquisition = reader(source)
    if acquisition.timestamps.size < 2:
        raise ValueError("recording must contain at least two samples")
    return acquisition


def _read_parquet(source: Path) -> Acquisition:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ValueError(
            "reading parquet requires pyarrow; install emg-denoise[parquet]"
        ) from error
    # pyarrow ships partial annotations; read_table is untyped upstream.
    table: Any = pq.read_table(source)  # type: ignore[no-untyped-call]
    names: list[str] = list(table.column_names)
    if TIMESTAMP_COLUMN not in names:
        raise ValueError(f"parquet must contain a {TIMESTAMP_COLUMN!r} column")
    channels = _channel_order(names)

    def column(name: str) -> Any:
        values: Any = table[name].to_numpy(zero_copy_only=False)
        return values

    stamps = np.asarray(column(TIMESTAMP_COLUMN), dtype=np.float64).reshape(-1)
    samples = np.column_stack(
        [np.asarray(column(name), dtype=np.float64).reshape(-1) for name in channels]
    )
    extras = {
        name: np.asarray(column(name))
        for name in names
        if name != TIMESTAMP_COLUMN and name not in channels
    }
    channel_dtypes = {name: table.schema.field(name).type for name in channels}
    return Acquisition(
        stamps,
        sample_matrix(samples),
        tuple(channels),
        column_names=tuple(names),
        extras=extras,
        channel_dtypes=channel_dtypes,
    )


def _read_csv(source: Path) -> Acquisition:
    raw: Any = np.genfromtxt(
        source, delimiter=",", names=True, dtype=np.float64, encoding="utf-8-sig"
    )
    names: list[str] = list(raw.dtype.names or ())
    if TIMESTAMP_COLUMN not in names:
        raise ValueError(f"CSV must contain a {TIMESTAMP_COLUMN!r} column")
    channels = _channel_order(names)

    def column(name: str) -> FloatArray:
        return np.asarray(raw[name], dtype=np.float64).reshape(-1)

    return Acquisition(
        column(TIMESTAMP_COLUMN),
        sample_matrix(np.column_stack([column(c) for c in channels])),
        tuple(channels),
        column_names=(TIMESTAMP_COLUMN, *channels),
    )


def _verify_uniform(timestamps: FloatArray, period_s: float) -> None:
    """Assert the rebuilt axis satisfies what stage two requires of its input."""
    gaps = np.diff(timestamps)
    worst = float(np.abs(gaps - period_s).max())
    # load_recording and replay both admit 1e-6 s of slack; stay well inside it.
    if worst > 1e-9:
        raise ValueError(f"rebuilt timestamps deviate from the period by {worst * 1e6:.6f} us")
    if np.any(gaps <= 0.0):
        raise ValueError("rebuilt timestamps must be strictly increasing")


def _write_csv(path: Path, timestamps: FloatArray, acquisition: Acquisition) -> None:
    header = [TIMESTAMP_COLUMN, *acquisition.channel_names]
    with atomic_text_writer(path) as destination:
        destination.write(",".join(header) + "\n")
        for index in range(timestamps.size):
            values = ",".join(format(value, ".17g") for value in acquisition.samples[index])
            destination.write(f"{format(timestamps[index], '.17g')},{values}\n")


def _write_parquet(path: Path, timestamps: FloatArray, acquisition: Acquisition) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ValueError(
            "writing parquet requires pyarrow; install emg-denoise[parquet]"
        ) from error

    arrays: dict[str, Any] = {}
    for name in acquisition.column_names:
        if name == TIMESTAMP_COLUMN:
            arrays[name] = pa.array(np.asarray(timestamps, dtype=np.float64), type=pa.float64())
            continue
        if name in acquisition.channel_names:
            index = acquisition.channel_names.index(name)
            values = np.asarray(acquisition.samples[:, index])
            dtype = acquisition.channel_dtypes.get(name)
            arrays[name] = pa.array(values, type=dtype) if dtype is not None else pa.array(values)
            continue
        arrays[name] = pa.array(acquisition.extras[name])

    table = pa.table(arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        write_table: Any = pq.write_table
        write_table(table, temporary)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_recording(path: Path, timestamps: FloatArray, acquisition: Acquisition) -> None:
    if path.suffix.lower() in {".parquet", ".pq"}:
        _write_parquet(path, timestamps, acquisition)
    else:
        _write_csv(path, timestamps, acquisition)


def _install_inplace(source: Path, rebuilt: Path, backup: Path, *, force: bool) -> None:
    """Publish a rebuilt parquet: first run creates ``.bak``; force keeps the bak."""
    if not backup.exists():
        os.replace(source, backup)
    elif not force:
        raise ValueError(f"backup exists; pass --force to rebuild without replacing it: {backup}")
    os.replace(rebuilt, source)


def run_retime(options: RetimeOptions) -> dict[str, Any]:
    """Rebuild one recording's time axis; a failed write leaves outputs intact."""
    assert_emg_recording(options.source)
    options.validate_paths()
    acquisition = load_acquisition(options.source)
    timestamps, report = rebuild_timestamps(
        acquisition.timestamps,
        period_s=options.period_s,
        method=options.method,
        tick_s=None if options.tick_us is None else options.tick_us * 1e-6,
        stitch=options.stitch,
        non_negative=options.non_negative,
    )
    if options.method == "uniform":
        _verify_uniform(timestamps, options.period_s)

    destination = options.destination
    if options.inplace:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        try:
            _write_parquet(temporary, timestamps, acquisition)
            _install_inplace(
                Path(options.source), temporary, options.backup_path, force=options.force
            )
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    else:
        _write_recording(destination, timestamps, acquisition)

    payload = _payload(options, acquisition, report)
    if options.sidecar:
        dump_json(options.report_path, payload)
    if options.update_meta:
        from .meta import register_recording_meta

        info = register_recording_meta(options.source, retime_payload=payload)
        if info is not None:
            payload["info"] = str(info)
    return payload


def _payload(
    options: RetimeOptions, acquisition: Acquisition, report: TimebaseReport
) -> dict[str, Any]:
    """Assemble the stage-one report, mirroring the units used in diagnostics."""
    return {
        "output": str(options.destination),
        "backup": str(options.backup_path) if options.inplace else None,
        "report": str(options.report_path) if options.sidecar else None,
        "samples": report.samples,
        "channels": len(acquisition.channel_names),
        "channel_names": list(acquisition.channel_names),
        "rate_hz": options.rate_hz,
        "measured_rate_hz": report.measured_rate_hz,
        "tick_us": report.tick_s * 1e6,
        "period_us": report.period_s * 1e6,
        "measured_period_us": report.measured_period_s * 1e6,
        "period_mismatch_us": report.period_mismatch_s * 1e6,
        "resync_segments": report.segments,
        "shortest_segment_samples": report.shortest_segment,
        "longest_segment_samples": report.longest_segment,
        "expected_segment_samples": report.expected_segment_samples,
        "max_offset_us": report.max_offset_s * 1e6,
        "first_timestamp_s": report.first_timestamp_s,
        "last_timestamp_s": report.last_timestamp_s,
        "method": report.method,
        "anchor_indices": list(report.anchor_indices),
        "interpolated_intervals": list(report.interpolated_intervals),
        "skipped_intervals": list(report.skipped_intervals),
        "edges_unchanged": report.method == "anchor_interpolation",
        "stitched": options.stitch if report.method == "uniform" else False,
        "inplace": options.inplace,
        "warnings": list(report.warnings()),
        "note": (
            "Sample values are unchanged; only the time axis is rebuilt. "
            "Only bounded anchor intervals are interpolated; uncovered edges remain unchanged."
        ),
    }
