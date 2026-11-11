"""File denoise service: viewer-style offline filtfilt by default (CSV/parquet)."""

from __future__ import annotations

import csv
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._typing import FloatArray, IntArray
from ..dsp.core import DenoiserConfig, RealtimeDenoiser
from ..dsp.filters import (
    Modality,
    assert_emg_recording,
    estimate_rate_hz,
    modality_preset,
    resolved_preset,
)
from ..dsp.offline import offline_denoise
from ..io import atomic_text_writer, dump_json, parse_row, read_header, validate_output_paths
from .retime import TIMESTAMP_COLUMN, Acquisition, load_acquisition


@dataclass(frozen=True, slots=True)
class ReplayOptions:
    """Paths and session options for offline file denoise; input units are unchanged."""

    source: Path
    output: Path | None = None
    modality: Modality | None = None
    #: Explicit sample rate; otherwise prefer ``meta/info.json`` fps, then measured gaps.
    sample_rate_hz: float | None = None
    #: Write ``<output>.report.json`` (and optional ``.mask.parquet``) beside the recording.
    sidecar: bool = False
    #: Merge diagnostics into the enclosing LeRobot ``meta/info.json`` when present.
    update_meta: bool = True
    #: Use streaming causal filters instead of viewer-style filtfilt (debug / realtime parity).
    causal: bool = False

    @property
    def destination(self) -> Path:
        return Path(self.output) if self.output is not None else default_process_output(self.source)

    @property
    def report_path(self) -> Path:
        return Path(f"{self.destination}.report.json")

    @property
    def mask_path(self) -> Path:
        return Path(f"{self.destination}.mask.parquet")

    @property
    def resolved_modality(self) -> Modality:
        chosen = "emg" if self.modality is None else self.modality
        return modality_preset(chosen).modality

    def validate_paths(self) -> None:
        outputs = [self.destination]
        if self.sidecar:
            outputs.append(self.report_path)
            if _is_parquet_path(self.destination):
                outputs.append(self.mask_path)
        validate_output_paths([self.source], outputs)


@dataclass(slots=True)
class ReplayReport:
    modality: str = "emg"
    sample_rate_hz: int = 2000
    channels: int = 0
    input_samples: int = 0
    output_samples: int = 0
    valid_samples: int = 0
    segments: int = 0
    max_hop_processing_ms: float = 0.0
    max_estimated_digital_latency_ms: float = 0.0
    observed_end_to_end_ms: float | None = None
    unemitted_samples: int = 0
    resampled_segments: int = 0
    unfiltered_samples: int = 0
    method: str = "offline_filtfilt"
    note: str = "Offline file denoise: mirror pad + filtfilt + period-extended comb."
    stages: list[str] = field(default_factory=list)


def default_process_output(source: str | Path) -> Path:
    """Sibling ``<stem>.denoised.<ext>`` next to the input recording."""
    path = Path(source)
    suffix = path.suffix if path.suffix else ".parquet"
    return path.with_name(f"{path.stem}.denoised{suffix}")


def _is_parquet_path(path: Path) -> bool:
    name = path.name.lower()
    if name.endswith(".bak"):
        name = name[: -len(".bak")]
    return name.endswith(".parquet") or name.endswith(".pq")


def _segment_bounds(timestamps: FloatArray, period: float) -> list[tuple[int, int]]:
    """Half-open contiguous spans where consecutive gaps equal ``period``."""
    if timestamps.size == 0:
        return []
    bounds: list[tuple[int, int]] = []
    start = 0
    for index in range(1, timestamps.size):
        if abs(timestamps[index] - timestamps[index - 1] - period) > 1e-6:
            bounds.append((start, index))
            start = index
    bounds.append((start, timestamps.size))
    return bounds


def _feature_fps(source: Path) -> float | None:
    """Read ``features.<key>.fps`` from the enclosing session ``meta/info.json``."""
    import json

    from .meta import feature_key_from_path, find_session_root

    root = find_session_root(source)
    if root is None:
        return None
    info_path = root / "meta" / "info.json"
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    key = feature_key_from_path(source)
    if key.endswith(".denoised"):
        key = key[: -len(".denoised")]
    feature = (info.get("features") or {}).get(key)
    if not isinstance(feature, dict):
        return None
    fps = feature.get("fps")
    if isinstance(fps, (int, float)) and float(fps) > 0.0:
        return float(fps)
    return None


def resolve_process_rate(
    source: Path,
    modality: Modality,
    *,
    override: float | None = None,
    timestamps: FloatArray | None = None,
) -> int:
    """Pick the denoise sample rate: ``--rate`` > feature fps > measured > preset."""
    measured = estimate_rate_hz(timestamps) if timestamps is not None else None
    declared: float | None = float(override) if override is not None else _feature_fps(source)
    if declared is None:
        declared = float(resolved_preset(modality).sample_rate)

    if measured is not None and abs(measured - declared) / declared > 0.02:
        if override is not None:
            raise ValueError(
                f"timestamps ≈ {measured:.3f} Hz but --rate {override:g} Hz (>2% relative mismatch)"
            )
        if _feature_fps(source) is not None:
            raise ValueError(
                f"timestamps ≈ {measured:.3f} Hz but feature fps is {declared:g} Hz "
                f"(pass --rate to override after checking the recording)"
            )
        # Preset 2000 Hz disagreed with the measured recording rate.
        declared = measured

    preset = resolved_preset(modality, declared)
    return preset.sample_rate


def run_replay(options: ReplayOptions) -> dict[str, Any]:
    """Denoise a recording; a failed write leaves any existing output intact."""
    assert_emg_recording(options.source)
    options.validate_paths()
    if _is_parquet_path(Path(options.source)):
        return _run_replay_parquet(options)
    return _run_replay_csv(options)


def _denoise_arrays(
    timestamps: FloatArray,
    samples: FloatArray,
    *,
    modality: Modality,
    causal: bool,
    sample_rate_hz: int,
) -> tuple[FloatArray, FloatArray, NDArray[np.bool_], IntArray, ReplayReport]:
    preset = resolved_preset(modality, sample_rate_hz)
    period = 1 / preset.sample_rate
    report = ReplayReport(
        modality=modality,
        sample_rate_hz=preset.sample_rate,
        channels=int(samples.shape[1]),
        method="causal_stream" if causal else "offline_filtfilt",
    )
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError("processing requires finite, strictly increasing timestamps")
    if causal and np.any(
        (np.diff(timestamps) <= 3 * period) & (np.abs(np.diff(timestamps) - period) > 1e-6)
    ):
        raise ValueError(
            "causal processing requires a uniform timebase; use offline process for anchor-interpolated data"
        )
    if causal:
        report.note = "Causal streaming denoise (RealtimeDenoiser); edges may ring until settled."
        engine = RealtimeDenoiser(
            DenoiserConfig(
                channels=samples.shape[1],
                modality=modality,
                sample_rate_hz=preset.sample_rate,
            )
        )
        out_stamps: list[FloatArray] = []
        out_samples: list[FloatArray] = []
        out_valid: list[NDArray[np.bool_]] = []
        out_segments: list[IntArray] = []
        for start, stop in _segment_bounds(timestamps, period):
            chunk_t = timestamps[start:stop]
            chunk_x = samples[start:stop]
            report.input_samples += stop - start
            for offset in range(0, stop - start, preset.block_size):
                hop_t = chunk_t[offset : offset + preset.block_size]
                hop_x = chunk_x[offset : offset + preset.block_size]
                for block in engine.process(hop_x, hop_t):
                    out_stamps.append(block.timestamps)
                    out_samples.append(block.samples)
                    out_valid.append(block.valid)
                    out_segments.append(
                        np.full(len(block.samples), block.segment_id, dtype=np.int64)
                    )
                    _accumulate_block(report, block)
        if not out_stamps:
            raise ValueError("recording contains no samples")
        return (
            np.concatenate(out_stamps),
            np.vstack(out_samples),
            np.concatenate(out_valid),
            np.concatenate(out_segments),
            report,
        )

    out_stamps = []
    out_samples = []
    out_valid = []
    out_segments = []
    edges = [
        0,
        *(np.flatnonzero(np.diff(timestamps) > 3 * period) + 1).tolist(),
        len(timestamps),
    ]
    bounds = list(zip(edges[:-1], edges[1:], strict=True))
    for segment_id, (start, stop) in enumerate(bounds):
        chunk_t = timestamps[start:stop]
        chunk_x = samples[start:stop]
        report.input_samples += stop - start
        started = perf_counter()
        irregular = np.any(np.abs(np.diff(chunk_t) - period) > 1e-6)
        segment_valid = True
        if len(chunk_t) < 32:
            filtered = chunk_x.copy()
            segment_valid = False
            report.unfiltered_samples += len(chunk_t)
            diagnostics = {"note": "Short segments preserved unfiltered and marked invalid."}
        elif irregular:
            report.resampled_segments += 1
            # Interpolate onto an internal uniform grid, filter, then map back.
            # The stored time axis and sample count stay exactly unchanged.
            grid = np.linspace(
                chunk_t[0], chunk_t[-1], max(32, round((chunk_t[-1] - chunk_t[0]) / period) + 1)
            )
            grid_rate = (len(grid) - 1) / (grid[-1] - grid[0])
            regular = np.column_stack([np.interp(grid, chunk_t, channel) for channel in chunk_x.T])
            denoised, diagnostics = offline_denoise(
                regular, modality=modality, sample_rate=float(grid_rate)
            )
            filtered = np.column_stack(
                [np.interp(chunk_t, grid, channel) for channel in denoised.T]
            )
            diagnostics["note"] = (
                "Irregular EMG: linear interpolation to an internal uniform grid, offline filtering, then mapping back; original timestamps retained."
            )
        else:
            filtered, diagnostics = offline_denoise(
                chunk_x, modality=modality, sample_rate=float(preset.sample_rate)
            )
        cost_ms = (perf_counter() - started) * 1000.0
        report.max_hop_processing_ms = max(report.max_hop_processing_ms, cost_ms)
        report.stages = list(diagnostics.get("stages") or [])
        report.note = str(diagnostics.get("note") or report.note)
        count = stop - start
        out_stamps.append(chunk_t)
        out_samples.append(filtered)
        out_valid.append(np.full(count, segment_valid, dtype=np.bool_))
        out_segments.append(np.full(count, segment_id, dtype=np.int64))
        report.output_samples += count
        report.valid_samples += count if segment_valid else 0
        report.segments = segment_id + 1
    if not out_stamps:
        raise ValueError("recording contains no samples")
    return (
        np.concatenate(out_stamps),
        np.vstack(out_samples),
        np.concatenate(out_valid),
        np.concatenate(out_segments),
        report,
    )


def _run_replay_csv(options: ReplayOptions) -> dict[str, Any]:
    modality = options.resolved_modality
    with Path(options.source).open(newline="", encoding="utf-8-sig") as source:
        reader = csv.reader(source)
        header = read_header(reader)
        rows = [parse_row(row, len(header), reader.line_num) for row in reader if row]
    if not rows:
        raise ValueError("CSV contains no samples")
    data = np.asarray(rows, dtype=np.float64)
    timestamps = data[:, 0]
    rate = resolve_process_rate(
        Path(options.source),
        modality,
        override=options.sample_rate_hz,
        timestamps=timestamps,
    )
    stamps, samples, _valid, _segments, report = _denoise_arrays(
        timestamps,
        data[:, 1:],
        modality=modality,
        causal=options.causal,
        sample_rate_hz=rate,
    )
    with atomic_text_writer(options.destination) as handle:
        writer = csv.writer(handle)
        # Signal schema only — never emit valid/segment_id as pseudo-channels.
        writer.writerow(header)
        for index in range(stamps.size):
            writer.writerow([format(stamps[index], ".17g"), *samples[index]])
    return _finish_report(options, report, samples=samples)


def _signal_column_names(acquisition: Acquisition) -> tuple[str, ...]:
    """Keep timestamp + metadata + declared channels; drop legacy QC columns."""
    names = [name for name in acquisition.column_names if name not in {"valid", "segment_id"}]
    if TIMESTAMP_COLUMN not in names:
        names = [TIMESTAMP_COLUMN, *names]
    for channel in acquisition.channel_names:
        if channel not in names:
            names.append(channel)
    return tuple(names)


def _run_replay_parquet(options: ReplayOptions) -> dict[str, Any]:
    modality = options.resolved_modality
    acquisition = load_acquisition(options.source)
    timestamps = np.asarray(acquisition.timestamps, dtype=np.float64)
    samples = np.asarray(acquisition.samples, dtype=np.float64)
    if timestamps.size == 0:
        raise ValueError("parquet contains no samples")
    rate = resolve_process_rate(
        Path(options.source),
        modality,
        override=options.sample_rate_hz,
        timestamps=timestamps,
    )
    stamps, filtered, valid, segments, report = _denoise_arrays(
        timestamps,
        samples,
        modality=modality,
        causal=options.causal,
        sample_rate_hz=rate,
    )
    keep_meta = stamps.size == timestamps.size
    extras = (
        {
            name: np.asarray(values)
            for name, values in acquisition.extras.items()
            if name not in {"valid", "segment_id"}
        }
        if keep_meta
        else {}
    )
    columns = (
        _signal_column_names(acquisition)
        if keep_meta
        else (TIMESTAMP_COLUMN, *acquisition.channel_names)
    )
    rebuilt = Acquisition(
        timestamps=stamps,
        samples=filtered,
        channel_names=acquisition.channel_names,
        column_names=columns,
        extras=extras,
        channel_dtypes=acquisition.channel_dtypes,
    )
    _write_denoised_parquet(options.destination, rebuilt)
    if options.sidecar:
        _write_mask_parquet(
            options.mask_path,
            timestamps=stamps,
            valid=valid,
            segment_id=segments,
            episode_index=extras.get("episode_index"),
        )
    return _finish_report(
        options,
        report,
        samples=filtered,
        episode_index=extras.get("episode_index"),
    )


def _accumulate_block(report: ReplayReport, block: Any) -> None:
    report.output_samples += len(block.samples)
    report.valid_samples += int(block.valid.sum())
    report.segments = block.segment_id + 1
    report.max_hop_processing_ms = max(report.max_hop_processing_ms, block.processing_ms)
    report.max_estimated_digital_latency_ms = max(
        report.max_estimated_digital_latency_ms, block.estimated_digital_latency_ms
    )


def _finish_report(
    options: ReplayOptions,
    report: ReplayReport,
    *,
    samples: FloatArray,
    episode_index: Any | None = None,
) -> dict[str, Any]:
    report.unemitted_samples = report.input_samples - report.output_samples
    payload = asdict(report)
    payload["output"] = str(options.destination)
    payload["report"] = str(options.report_path) if options.sidecar else None
    payload["mask"] = (
        str(options.mask_path)
        if options.sidecar and _is_parquet_path(options.destination)
        else None
    )
    payload["causal"] = options.causal
    from .meta import channel_stats, register_recording_meta

    payload["stats"] = channel_stats(samples)
    if options.sidecar:
        dump_json(options.report_path, payload)
    if options.update_meta:
        info = register_recording_meta(
            options.source,
            denoise_payload=payload,
            samples=samples,
            episode_index=episode_index,
        )
        if info is not None:
            payload["info"] = str(info)
    return payload


def _write_denoised_parquet(path: Path, acquisition: Acquisition) -> None:
    """Write signal parquet without ``valid`` / ``segment_id`` pseudo-channels."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ValueError(
            "writing parquet requires pyarrow; install emg-denoise[parquet]"
        ) from error

    arrays: dict[str, Any] = {}
    for name in acquisition.column_names:
        if name in {"valid", "segment_id"}:
            continue
        if name == TIMESTAMP_COLUMN:
            arrays[name] = pa.array(
                np.asarray(acquisition.timestamps, dtype=np.float64), type=pa.float64()
            )
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


def _write_mask_parquet(
    path: Path,
    *,
    timestamps: FloatArray,
    valid: NDArray[np.bool_],
    segment_id: IntArray,
    episode_index: Any | None = None,
) -> None:
    """Optional QC sidecar: sample-aligned mask, never mixed into the signal feature."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover
        raise ValueError(
            "writing parquet requires pyarrow; install emg-denoise[parquet]"
        ) from error

    arrays: dict[str, Any] = {
        TIMESTAMP_COLUMN: pa.array(np.asarray(timestamps, dtype=np.float64), type=pa.float64()),
        "sample_index": pa.array(np.arange(timestamps.size, dtype=np.int64)),
        "valid": pa.array(np.asarray(valid, dtype=np.bool_)),
        "segment_id": pa.array(np.asarray(segment_id, dtype=np.int64)),
    }
    if episode_index is not None:
        arrays["episode_index"] = pa.array(np.asarray(episode_index))

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
        write_table(pa.table(arrays), temporary)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
