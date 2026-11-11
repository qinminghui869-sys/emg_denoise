"""Pure signal metrics; engineering heuristics, not physiological labels."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import welch

from .._typing import FloatArray


def runs(mask: Any) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(
        zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist(), strict=True)
    )


def windows(
    t: FloatArray,
    x: FloatArray,
    fs: float,
    window: float,
    step: float,
    gap: float,
    valid: Any = None,
) -> list[tuple[int, int]]:
    good = np.isfinite(t) & np.isfinite(x)
    if valid is not None:
        good &= valid
    length, stride = max(2, round(window * fs)), max(1, round(step * fs))
    breaks = np.flatnonzero((np.diff(t) <= 0) | (np.diff(t) > gap / fs)) + 1
    result: list[tuple[int, int]] = []
    for start, stop in runs(good):
        bounds = [start, *breaks[(breaks > start) & (breaks < stop)].tolist(), stop]
        for a, b in zip(bounds[:-1], bounds[1:], strict=True):
            result.extend((i, i + length) for i in range(a, b - length + 1, stride))
    return result


def spectrum(x: FloatArray, fs: float) -> dict[str, Any]:
    f, p = welch(x, fs=fs, nperseg=min(len(x), max(2, round(fs))), detrend="constant")
    df = f[1] - f[0]
    total = float(p.sum() * df)
    band = (f >= 20) & (f <= 450)
    harmonic = np.zeros(len(f), dtype=bool)
    for hz in range(50, 451, 50):
        harmonic |= np.abs(f - hz) <= 2

    def energy(mask: Any) -> float:
        return float(p[mask].sum() * df)

    low = energy(f < 20) if fs / 2 >= 20 else None
    emg = energy(band) if fs / 2 >= 450 else None
    high = energy(f > 450) if fs / 2 > 450 else None
    line = energy(band & harmonic) if emg is not None else None
    return {
        "low_fraction": low / total if low is not None and total > 0 else None,
        "band_fraction": emg / total if emg is not None and total > 0 else None,
        "high_fraction": high / total if high is not None and total > 0 else None,
        "line_fraction": line / emg if line is not None and emg else None,
        "line_power": line,
        "nonline_band_power": energy(band & ~harmonic) if emg is not None else None,
    }


def metrics(
    t: FloatArray, x: FloatArray, fs: float, config: dict[str, Any], window: float, step: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    finite = np.isfinite(x)
    values = x[finite]
    repeated = (
        finite[1:]
        & finite[:-1]
        & (x[1:] == x[:-1])
        & np.isfinite(np.diff(t))
        & (np.diff(t) > 0)
        & (np.diff(t) <= config["gap_periods"] / fs)
    )
    flat = [(a, b + 1) for a, b in runs(repeated)]
    longest = max(((b - a) / fs for a, b in flat), default=0.0)
    result: dict[str, Any] = {
        "samples": len(x),
        "finite_fraction": float(finite.mean()) if len(x) else 0.0,
        "mean": float(values.mean()) if values.size else None,
        "std": float(values.std()) if values.size else None,
        "rms": float(np.sqrt(np.mean(values**2))) if values.size else None,
        "peak_to_peak": float(np.ptp(values)) if values.size else None,
        "longest_constant_s": longest,
        "constant_intervals": [(a, b) for a, b in flat if (b - a) / fs >= config["flatline_s"]],
        "clipping_fraction": None,
    }
    if config["adc_min"] is not None:
        result["clipping_fraction"] = (
            float(np.mean((values <= config["adc_min"]) | (values >= config["adc_max"])))
            if values.size
            else None
        )
    rows = []
    for a, b in windows(t, x, fs, window, step, config["gap_periods"]):
        chunk = x[a:b]
        rows.append(
            {
                "start": a,
                "stop": b,
                "rms": float(np.sqrt(np.mean(chunk**2))),
                "peak_to_peak": float(np.ptp(chunk)),
                **spectrum(chunk, fs),
            }
        )
    result["windows"] = len(rows)
    for key in ("low_fraction", "band_fraction", "high_fraction", "line_fraction"):
        v = [r[key] for r in rows if r[key] is not None]
        result[key] = float(np.mean(v)) if v else None
    result["spectral_status"] = (
        "PASS"
        if rows and fs / 2 > 450 and any(r["line_fraction"] is not None for r in rows)
        else "NOT_VERIFIED"
    )
    return result, rows


def db_ratio(after: float | None, before: float | None) -> float | None:
    return (
        float(10 * np.log10(after / before))
        if after is not None and before is not None and after > 0 and before > 0
        else None
    )
