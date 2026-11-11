"""Read-only EMG quality checks. Input times are seconds; amplitude units are preserved."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..io import jsonable, validate_output_paths
from ..pipeline.meta import find_session_root
from .discovery import channel_names, discover, metadata, read_columns
from .report import write_reports
from .signal import db_ratio, metrics, runs, spectrum, windows

DEFAULTS: dict[str, Any] = {
    "rate_warn_fraction": 0.01,
    "rate_error_fraction": 0.05,
    "gap_periods": 3.0,
    "uniform_tolerance_s": 1e-6,
    "flatline_s": 1.0,
    "rms_mad_multiplier": 8.0,
    "line_fraction_warn": 0.20,
    "adc_min": None,
    "adc_max": None,
    "clipping_fraction_warn": 0.01,
}


@dataclass(frozen=True)
class QcOptions:
    """Sources/globs, optional nominal rate, window seconds, and report destination."""

    inputs: tuple[str, ...]
    output_dir: Path | None = None
    rate: float | None = None
    window_s: float = 2.0
    step_s: float = 1.0
    config: Path | None = None
    require_denoised: bool = False
    strict: bool = False


def _config(options: QcOptions) -> dict[str, Any]:
    config = DEFAULTS.copy()
    if options.config:
        custom = json.loads(options.config.read_text())
        if not isinstance(custom, dict) or custom.keys() - config.keys():
            raise ValueError("QC config must be an object containing known threshold keys")
        config.update(custom)
    for key, value in config.items():
        if key.startswith("adc_") and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"invalid threshold: {key}")
        if not key.startswith("adc_") and value <= 0:
            raise ValueError(f"threshold must be positive: {key}")
    if (config["adc_min"] is None) != (config["adc_max"] is None) or (
        config["adc_min"] is not None and config["adc_min"] >= config["adc_max"]
    ):
        raise ValueError("ADC bounds must both be supplied, with adc_min < adc_max")
    if config["rate_warn_fraction"] >= config["rate_error_fraction"]:
        raise ValueError("rate warning must be below error threshold")
    for key in ("line_fraction_warn", "clipping_fraction_warn"):
        if config[key] > 1:
            raise ValueError(f"{key} must be <= 1")
    for value in (options.window_s, options.step_s, options.rate or 2000):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("rate, window and step must be positive finite numbers")
    if options.rate == 0:
        raise ValueError("rate must be positive")
    return config


class _Scan:
    def __init__(self, options: QcOptions, config: dict[str, Any]) -> None:
        self.options = options
        self.config = config
        self.findings: list[dict[str, Any]] = []
        self.channels: list[dict[str, Any]] = []
        self.protected: list[Path] = []
        self.incomplete = False

    def finding(
        self,
        path: Path,
        stage: str,
        rule: str,
        severity: str,
        message: str,
        *,
        status: str = "FAIL",
        channel: str | None = None,
        start: int | None = None,
        stop: int | None = None,
        timestamps: Any = None,
        observed: Any = None,
        threshold: Any = None,
    ) -> None:
        self.findings.append(
            dict(
                file=str(path),
                stage=stage,
                rule=rule,
                severity=severity,
                status=status,
                channel=channel,
                start_sample=start,
                stop_sample=stop,
                start_s=timestamps[start]
                if timestamps is not None and start is not None and start < len(timestamps)
                else None,
                end_s=timestamps[stop - 1]
                if timestamps is not None and stop is not None and stop > 0
                else None,
                observed=observed,
                threshold=threshold,
                message=message,
            )
        )

    def inspect(
        self, path: Path, stage: str, fs: float, feature: dict[str, Any]
    ) -> dict[str, Any] | None:
        self.protected.append(path)
        try:
            columns = read_columns(path)
        except (OSError, ValueError, TypeError) as error:
            self.incomplete = True
            self.finding(path, stage, "read", "ERROR", str(error), status="NOT_VERIFIED")
            return None
        if "timestamp" not in columns:
            self.finding(path, stage, "schema", "ERROR", "missing timestamp")
            return None
        names = channel_names(columns)
        for name in ("valid", "segment_id"):
            if name in columns:
                self.finding(
                    path,
                    stage,
                    "legacy_qc_column",
                    "WARN",
                    f"{name} is a metadata column; legacy consumers may read it as signal",
                )
        numeric: dict[str, Any] = {}
        for name in ["timestamp", *names]:
            values = columns[name]
            try:
                numeric[name] = np.asarray(values, dtype=np.float64)
            except (ValueError, TypeError):
                converted = []
                for i, value in enumerate(values):
                    try:
                        converted.append(float(value) if value is not None else float("nan"))
                    except (ValueError, TypeError):
                        converted.append(float("nan"))
                        self.finding(
                            path,
                            stage,
                            "nonnumeric",
                            "ERROR",
                            "non-numeric value",
                            channel=name,
                            start=i,
                            stop=i + 1,
                        )
                numeric[name] = np.asarray(converted)
        t = numeric["timestamp"]
        if not len(t) or not names:
            self.finding(path, stage, "empty", "ERROR", "empty recording or no signal channels")
            return None
        shape = feature.get("shape")
        if shape is not None and shape != [len(names)]:
            self.finding(
                path,
                stage,
                "shape",
                "ERROR",
                "channel count differs from feature.shape",
                observed=len(names),
                threshold=shape,
            )
        for a, b in runs(~np.isfinite(t)):
            self.finding(
                path,
                stage,
                "timestamp_finite",
                "ERROR",
                "null/nonfinite timestamp",
                start=a,
                stop=b,
            )
        dt = np.diff(t)
        for a, b in runs(np.isfinite(dt) & (dt <= 0)):
            self.finding(
                path,
                stage,
                "time_order",
                "ERROR",
                "duplicate/backward timestamp",
                start=a,
                stop=b + 1,
                timestamps=t,
            )
        for a, b in runs(dt > self.config["gap_periods"] / fs):
            self.finding(
                path,
                stage,
                "time_gap",
                "WARN",
                "suspected gap; missing sample count is unknown",
                start=a,
                stop=b + 1,
                timestamps=t,
                observed=float(np.max(dt[a:b])),
                threshold=self.config["gap_periods"] / fs,
            )
        measured = (
            (len(t) - 1) / (t[-1] - t[0])
            if len(t) > 1 and np.isfinite(t).all() and np.all(dt > 0)
            else None
        )
        if measured is not None:
            deviation = abs(measured / fs - 1)
            if deviation > self.config["rate_warn_fraction"]:
                self.finding(
                    path,
                    stage,
                    "sample_rate",
                    "ERROR" if deviation > self.config["rate_error_fraction"] else "WARN",
                    "measured and nominal rate differ",
                    observed=measured,
                    threshold=fs,
                )
        if (
            stage == "current"
            and feature.get("timebase", {}).get("rebuilt")
            and feature.get("timebase", {}).get("method", "uniform") == "uniform"
        ):
            for a, b in runs(
                ~np.isfinite(dt) | (np.abs(dt - 1 / fs) > self.config["uniform_tolerance_s"])
            ):
                self.finding(
                    path,
                    stage,
                    "rebuilt_uniform",
                    "ERROR",
                    "rebuilt timebase is not uniform",
                    start=a,
                    stop=b + 1,
                    timestamps=t,
                    threshold=self.config["uniform_tolerance_s"],
                )
        for name in names:
            x = numeric[name]
            for a, b in runs(~np.isfinite(x)):
                self.finding(
                    path,
                    stage,
                    "signal_finite",
                    "ERROR",
                    "null/nonfinite signal",
                    channel=name,
                    start=a,
                    stop=b,
                    timestamps=t,
                )
            m, rows = metrics(t, x, fs, self.config, self.options.window_s, self.options.step_s)
            self.channels.append(
                dict(
                    file=str(path),
                    stage=stage,
                    channel=name,
                    nominal_rate_hz=fs,
                    measured_rate_hz=measured,
                    dt_quantiles_s=np.quantile(dt[np.isfinite(dt)], [0, 0.5, 0.95, 1]).tolist()
                    if np.isfinite(dt).any()
                    else None,
                    **m,
                )
            )
            if np.isfinite(x).all() and np.all(x == x[0]):
                self.finding(
                    path,
                    stage,
                    "constant",
                    "ERROR",
                    "entire channel is constant",
                    channel=name,
                    start=0,
                    stop=len(x),
                    timestamps=t,
                )
            else:
                for a, b in m["constant_intervals"]:
                    self.finding(
                        path,
                        stage,
                        "flatline",
                        "WARN",
                        "sustained identical samples",
                        channel=name,
                        start=a,
                        stop=b,
                        timestamps=t,
                        threshold=self.config["flatline_s"],
                    )
            if m["clipping_fraction"] is None:
                self.finding(
                    path,
                    stage,
                    "clipping",
                    "INFO",
                    "ADC range unavailable",
                    status="NOT_VERIFIED",
                    channel=name,
                )
            elif m["clipping_fraction"] > self.config["clipping_fraction_warn"]:
                self.finding(
                    path,
                    stage,
                    "clipping",
                    "WARN",
                    "samples at/outside configured ADC rails",
                    channel=name,
                    observed=m["clipping_fraction"],
                    threshold=self.config["clipping_fraction_warn"],
                )
            if m["spectral_status"] != "PASS":
                self.finding(
                    path,
                    stage,
                    "spectrum",
                    "INFO",
                    "short/invalid segments, zero power or unavailable frequency band",
                    status="NOT_VERIFIED",
                    channel=name,
                )
            rms = np.array([r["rms"] for r in rows])
            median = float(np.median(rms)) if len(rms) else 0.0
            mad = float(np.median(np.abs(rms - median))) if len(rms) else 0.0
            for row in rows:
                for rule, value, limit in [
                    ("line_noise", row["line_fraction"], self.config["line_fraction_warn"]),
                    (
                        "rms_outlier",
                        row["rms"] if mad > 0 else None,
                        median + self.config["rms_mad_multiplier"] * 1.4826 * mad,
                    ),
                ]:
                    if value is not None and value > limit:
                        self.finding(
                            path,
                            stage,
                            rule,
                            "WARN",
                            "review window; not an automatic rejection of muscle activity",
                            channel=name,
                            start=row["start"],
                            stop=row["stop"],
                            timestamps=t,
                            observed=value,
                            threshold=limit,
                        )
        return dict(t=t, signals={n: numeric[n] for n in names}, columns=columns, names=names)

    def compare(
        self,
        path: Path,
        before: dict[str, Any],
        after: dict[str, Any],
        stage: str,
        fs: float,
        feature: dict[str, Any],
    ) -> None:
        t, u = before["t"], after["t"]
        same_names = before["names"] == after["names"]
        same_size = len(t) == len(u)
        if stage == "retime":
            match = (
                same_names
                and same_size
                and all(
                    np.array_equal(before["signals"][n], after["signals"][n], equal_nan=True)
                    for n in before["names"]
                )
            )
            e1, e2 = before["columns"].get("episode_index"), after["columns"].get("episode_index")
            match = match and (
                (e1 is None and e2 is None)
                or (e1 is not None and e2 is not None and np.array_equal(e1, e2))
            )
            self.finding(
                path,
                stage,
                "sample_integrity",
                "INFO" if match else "ERROR",
                "backup/current sample and episode equality",
                status="PASS" if match else "FAIL",
            )
            if same_size and np.isfinite(t).all() and np.isfinite(u).all():
                self.finding(
                    path,
                    stage,
                    "time_shift",
                    "INFO",
                    "relative to existing backup, not external synchronization error",
                    status="PASS",
                    observed={
                        "quantiles_s": np.quantile(u - t, [0, 0.5, 0.95, 1]).tolist(),
                        "max_abs_s": float(np.max(np.abs(u - t))),
                    },
                )
            return
        method = feature.get("pipeline", {}).get("method")
        valid: Any = np.ones(len(u), dtype=bool)
        if method != "offline_filtfilt":
            if method != "causal_stream":
                self.finding(
                    path,
                    stage,
                    "processing_mode",
                    "INFO",
                    "processing method unavailable; comparison skipped",
                    status="NOT_VERIFIED",
                )
                return
            mask_path = Path(str(path) + ".mask.parquet")
            self.protected.append(mask_path)
            if not mask_path.exists():
                self.finding(
                    path, stage, "mask", "INFO", "causal mask unavailable", status="NOT_VERIFIED"
                )
                return
            try:
                mask = read_columns(mask_path)
                if (
                    not np.array_equal(mask["timestamp"], u)
                    or len(mask["valid"]) != len(u)
                    or np.asarray(mask["valid"]).dtype.kind != "b"
                ):
                    raise ValueError("mask is not timestamp-aligned boolean data")
                valid = np.asarray(mask["valid"], dtype=bool)
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.finding(path, stage, "mask", "ERROR", str(error))
                return
            if not (
                np.isfinite(t).all()
                and np.isfinite(u).all()
                and np.all(np.diff(t) > 0)
                and np.all(np.diff(u) > 0)
            ):
                self.finding(
                    path,
                    stage,
                    "alignment",
                    "ERROR",
                    "causal comparison requires strictly increasing finite timestamps",
                )
                return
            indices = np.searchsorted(t, u)
            aligned = (
                same_names
                and np.all(indices < len(t))
                and np.array_equal(t[np.minimum(indices, len(t) - 1)], u)
            )
            if aligned:
                self.finding(
                    path,
                    stage,
                    "causal_coverage",
                    "INFO",
                    "timestamp-matched source samples; omitted samples are not silently truncated",
                    status="PASS",
                    observed={
                        "input_samples": len(t),
                        "output_samples": len(u),
                        "omitted_samples": len(t) - len(u),
                        "valid_samples": int(valid.sum()),
                    },
                )
                before = dict(
                    before,
                    t=t[indices],
                    signals={n: x[indices] for n, x in before["signals"].items()},
                )
                t = before["t"]
        else:
            aligned = same_names and same_size and np.array_equal(t, u)
        self.finding(
            path,
            stage,
            "alignment",
            "INFO" if aligned else "ERROR",
            "source/output sample alignment; no truncation performed",
            status="PASS" if aligned else "FAIL",
        )
        if not aligned:
            return
        for name in before["names"]:
            x, y = before["signals"][name], after["signals"][name]
            spans = windows(
                t,
                x,
                fs,
                self.options.window_s,
                self.options.step_s,
                self.config["gap_periods"],
                valid & np.isfinite(y),
            )
            pre, post = [], []
            for a, b in spans:
                pre.append(spectrum(x[a:b], fs))
                post.append(spectrum(y[a:b], fs))
            row: dict[str, Any] = dict(
                file=str(path), stage="comparison", channel=name, windows=len(spans)
            )
            for key in ("line_power", "nonline_band_power"):
                pairs = [
                    (a[key], b[key])
                    for a, b in zip(pre, post, strict=True)
                    if a[key] is not None and b[key] is not None
                ]
                row[key + "_change_db"] = (
                    db_ratio(
                        float(np.mean([b for a, b in pairs])), float(np.mean([a for a, b in pairs]))
                    )
                    if pairs
                    else None
                )
            for label, a, b in [
                ("head", 0, min(len(t), round(fs))),
                ("tail", max(0, len(t) - round(fs)), len(t)),
            ]:
                good = valid[a:b] & np.isfinite(x[a:b]) & np.isfinite(y[a:b])
                row[label + "_rms_change_db"] = (
                    db_ratio(float(np.mean(y[a:b] ** 2)), float(np.mean(x[a:b] ** 2)))
                    if b - a >= round(fs)
                    and good.all()
                    and np.all(np.diff(t[a:b]) > 0)
                    and np.all(np.diff(t[a:b]) <= self.config["gap_periods"] / fs)
                    else None
                )
            self.channels.append(row)
            if not spans:
                self.finding(
                    path,
                    stage,
                    "spectral_comparison",
                    "INFO",
                    "no common valid full windows",
                    status="NOT_VERIFIED",
                    channel=name,
                )

    def file(self, path: Path) -> None:
        info, feature, root = metadata(path)
        if root:
            self.protected.extend((root / "meta").glob("*"))
        fs = self.options.rate if self.options.rate is not None else feature.get("fps", 2000)
        if not isinstance(fs, (int, float)) or not np.isfinite(fs) or fs <= 0:
            raise ValueError("invalid feature fps")
        current = self.inspect(path, "current", fs, feature)
        backup = Path(str(path) + ".bak")
        if backup.exists():
            raw = self.inspect(backup, "backup", fs, {})
            if raw is not None and current is not None:
                self.compare(path, raw, current, "retime", fs, feature)
        else:
            self.finding(
                path,
                "retime",
                "backup",
                "INFO",
                "backup absent; pre-rebuild gaps cannot be verified",
                status="NOT_RUN",
            )
        derived = path.with_name(path.stem + ".denoised" + path.suffix)
        if not derived.exists():
            self.finding(
                path,
                "denoised",
                "derived_missing",
                "ERROR" if self.options.require_denoised else "INFO",
                "denoised file absent",
                status="FAIL" if self.options.require_denoised else "NOT_RUN",
            )
            return
        derived_feature = info.get("features", {}).get(derived.stem, {})
        report_path = Path(str(derived) + ".report.json")
        self.protected.append(report_path)
        if not derived_feature.get("pipeline") and report_path.exists():
            derived_feature = dict(derived_feature, pipeline=json.loads(report_path.read_text()))
        output = self.inspect(derived, "denoised", fs, derived_feature)
        if root:
            if derived.stem not in info.get("features", {}):
                self.finding(
                    derived,
                    "denoised",
                    "registration",
                    "ERROR",
                    "derived feature missing from meta",
                )
            stats_path = root / "meta/episodes_stats.jsonl"
            episode_values = current["columns"].get("episode_index") if current else None
            episodes = (
                set(np.asarray(episode_values, dtype=int).tolist())
                if episode_values is not None
                else set()
            )
            if not episodes and path.parent.name.startswith("episode_"):
                episodes = {int(path.parent.name.removeprefix("episode_"))}
            stats = (
                [json.loads(line) for line in stats_path.read_text().splitlines() if line.strip()]
                if stats_path.exists()
                else []
            )
            covered = {r.get("episode_index") for r in stats if derived.stem in r.get("stats", {})}
            if not episodes or not episodes.issubset(covered):
                self.finding(
                    derived,
                    "denoised",
                    "stats",
                    "WARN",
                    "derived episode statistics missing or episode identity unavailable",
                    observed=sorted(covered),
                    threshold=sorted(episodes),
                )
        if current is not None and output is not None:
            self.compare(derived, current, output, "denoised", fs, derived_feature)


def run_qc(options: QcOptions) -> dict[str, Any]:
    """Inspect all inputs without mutation; return strict-JSON-compatible report and exit_code."""
    start = time.perf_counter()
    config = _config(options)
    paths = discover(options.inputs)
    scan = _Scan(options, config)
    files = []
    for path in paths:
        root = find_session_root(path)
        if root:
            scan.protected.extend((root / "meta").glob("*"))
        scan.protected.extend(path.parent.glob(path.stem + ".*"))
        offset = len(scan.findings)
        try:
            scan.file(path)
        except (OSError, ValueError, TypeError, KeyError) as error:
            scan.incomplete = True
            scan.finding(path, "file", "execution", "ERROR", str(error), status="NOT_VERIFIED")
        findings = scan.findings[offset:]
        files.append(
            {
                "file": str(path),
                "status": "FAIL"
                if any(f["severity"] == "ERROR" for f in findings)
                else "WARN"
                if any(f["severity"] == "WARN" for f in findings)
                else "NOT_VERIFIED"
                if any(f["status"] in ("NOT_RUN", "NOT_VERIFIED") for f in findings)
                else "PASS",
            }
        )
    errors = sum(f["severity"] == "ERROR" for f in scan.findings)
    warnings = sum(f["severity"] == "WARN" for f in scan.findings)
    code = 2 if scan.incomplete else 1 if errors or (options.strict and warnings) else 0
    payload = dict(
        schema_version=1,
        threshold_version=1,
        config=config,
        rate_override_hz=options.rate,
        window_s=options.window_s,
        step_s=options.step_s,
        strict=options.strict,
        require_denoised=options.require_denoised,
        files=files,
        channels=scan.channels,
        findings=scan.findings,
        errors=errors,
        warnings=warnings,
        status="INCOMPLETE"
        if scan.incomplete
        else "FAIL"
        if errors
        else "WARN"
        if warnings
        else "PASS",
        exit_code=code,
        elapsed_s=time.perf_counter() - start,
    )
    payload["unchecked"] = sum(f["status"] in ("NOT_RUN", "NOT_VERIFIED") for f in scan.findings)
    if payload["status"] == "PASS" and payload["unchecked"]:
        payload["status"] = "NOT_VERIFIED"
    result: dict[str, Any] = jsonable(payload)
    if options.output_dir:
        protected = [*paths, *scan.protected]
        if options.config:
            protected.append(options.config)
        validate_output_paths(
            protected,
            [options.output_dir / n for n in ("summary.json", "channels.csv", "findings.csv")],
        )
        write_reports(options.output_dir, result, protected)
    return result
