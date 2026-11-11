"""Offline zero-phase file denoise (not the causal stream).

Mirror-pad → sosfiltfilt(20 Hz highpass + 450 Hz lowpass) → unpad →
repair periodic edges → period-pad → comb filtfilt → unpad → edge fade.

Channel DC is not removed here. The viewer DC checkbox only affects display
and the spectrum.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy import signal

from .._typing import FloatArray
from .filters import (
    COMB_BANDWIDTH,
    Modality,
    comb_coefficients,
    emg_bandpass_sos,
    resolved_preset,
    sample_matrix,
)

COMB_F0_HZ = 50.0


def mirror_pad(samples: FloatArray) -> tuple[FloatArray, bool]:
    """Whole-signal even mirror: ``ABC → CBABCBA`` (length ``3N-2``)."""
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    if n < 2:
        return x.copy(), False
    left = x[1:][::-1]
    right = x[:-1][::-1]
    return np.concatenate([left, x, right], axis=0), True


def mirror_unpad(padded: FloatArray, n: int) -> FloatArray:
    if n < 2:
        return np.asarray(padded, dtype=np.float64).copy()
    return np.asarray(padded[n - 1 : 2 * n - 1], dtype=np.float64).copy()


def _period_pad(samples: FloatArray, fs: float, f0: float) -> tuple[FloatArray, int]:
    """Pad with period-continuous copies so a narrow notch/comb is settled at the seam."""
    x = np.asarray(samples, dtype=np.float64)
    n = x.shape[0]
    delay = max(1, int(round(fs / f0)))
    npre = max(delay * 4, int(round(2 * fs)))
    if n < delay * 4:
        return x.copy(), 0
    head = np.empty((npre, x.shape[1]), dtype=np.float64)
    tail = np.empty((npre, x.shape[1]), dtype=np.float64)
    for i in range(npre):
        head[npre - 1 - i] = x[(delay - 1 - (i % delay) + delay) % delay]
        tail[i] = x[n - delay + (i % delay)]
    return np.concatenate([head, x, tail], axis=0), npre


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _percentile_abs(x: FloatArray, start: int, stop: int, fraction: float) -> float:
    chunk = np.abs(x[start:stop])
    if chunk.size == 0:
        return 0.0
    flat = np.sort(chunk.reshape(-1))
    index = int(round(fraction * (flat.size - 1)))
    return float(flat[min(max(index, 0), flat.size - 1)])


def repair_periodic_edges(samples: FloatArray, fs: float, f0: float) -> FloatArray:
    """Overwrite non-periodic end cycles with the first clean interior period."""
    x = np.asarray(samples, dtype=np.float64).copy()
    if x.ndim == 1:
        x = x[:, None]
    n, channels = x.shape
    delay = max(1, int(round(fs / f0)))
    if n < 8 * delay:
        return x
    for channel in range(channels):
        z = np.asarray(x[:, channel], dtype=np.float64).copy()

        def mis(i: int, series: FloatArray = z) -> float:
            return float(np.max(np.abs(series[i : i + delay] - series[i + delay : i + 2 * delay])))

        mid = n // 2
        refs = [mis(i) for i in range(mid, min(mid + int(round(fs)), n - 2 * delay), delay)]
        ref = _median(refs) + 1e-9
        peak = _percentile_abs(z, mid, min(n, mid + int(round(fs))), 0.99) + 1e-9
        thr = max(8 * ref, 0.08 * peak)
        span = min(int(round(1.5 * fs)), n // 4 - 2 * delay)
        head = 0
        for i in range(0, max(span, 0), delay):
            if mis(i) > thr:
                head = i + delay
            else:
                break
        tail = 0
        for i in range(0, max(span, 0), delay):
            j = n - 2 * delay - i
            if j < 0:
                break
            if mis(j) > thr:
                tail = i + delay
            else:
                break
        if head >= delay:
            src = z[head : head + delay]
            for i in range(head):
                z[head - 1 - i] = src[(delay - 1 - (i % delay)) % delay]
        if tail >= delay:
            src = z[n - tail - delay : n - tail]
            for i in range(tail):
                z[n - tail + i] = src[i % delay]
        x[:, channel] = z
    return x


def fade_edge_ring(samples: FloatArray, fs: float) -> FloatArray:
    """Crossfade a decaying edge ring toward an even reflection of the interior."""
    x = np.asarray(samples, dtype=np.float64).copy()
    if x.ndim == 1:
        x = x[:, None]
    n, channels = x.shape
    hop = max(8, int(round(0.05 * fs)))
    max_l = min(n // 5, int(round(0.8 * fs)))
    if max_l < hop * 4:
        return x
    for channel in range(channels):
        z = np.asarray(x[:, channel], dtype=np.float64).copy()

        def rms(a: int, b: int, series: FloatArray = z) -> float:
            section = series[a:b]
            return float(np.sqrt(np.mean(section * section))) if section.size else 0.0

        for from_end in (False, True):
            interior_a = n - max_l - int(round(0.5 * fs)) if from_end else max_l
            interior_b = interior_a + int(round(0.5 * fs))
            if interior_a < 0 or interior_b > n:
                continue
            interior = rms(interior_a, interior_b)
            n_h = max_l // hop
            hops = [
                rms(
                    n - (h + 1) * hop if from_end else h * hop,
                    (n - h * hop) if from_end else (h + 1) * hop,
                )
                for h in range(n_h)
            ]
            n_hop = 0
            for h, value in enumerate(hops):
                if value <= 2.2 * interior:
                    break
                if h > 0 and value > hops[h - 1] * 1.25:
                    break
                n_hop = h + 1
            if n_hop < 2 or hops[n_hop - 1] > hops[0] * 0.85:
                continue
            length = n_hop * hop
            for i in range(length):
                weight = 0.5 * (1.0 - np.cos(np.pi * (length - 1 - i) / max(length - 1, 1)))
                if not from_end:
                    z[i] = (1.0 - weight) * z[i] + weight * z[2 * length - 1 - i]
                else:
                    idx = n - 1 - i
                    z[idx] = (1.0 - weight) * z[idx] + weight * z[n - length - 1 - i]
        x[:, channel] = z
    return x


def repair_edge_reflect(samples: FloatArray, fs: float) -> FloatArray:
    """Replace the first/last ~50 ms by an even reflection of the interior."""
    x = np.asarray(samples, dtype=np.float64).copy()
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    length = min(n // 4, max(8, int(round(0.05 * fs))))
    if length < 8 or n < 4 * length:
        return x
    x[:length] = x[2 * length - 1 : length - 1 : -1]
    x[-length:] = x[-length - 1 : -2 * length - 1 : -1]
    return x


def offline_denoise(
    samples: ArrayLike,
    *,
    modality: Modality | str = "emg",
    sample_rate: float | None = None,
) -> tuple[FloatArray, dict[str, Any]]:
    """Apply the viewer-style offline chain; returns ``(samples, diagnostics)``."""
    preset = resolved_preset(modality, sample_rate)
    fs = float(preset.sample_rate)
    x = sample_matrix(samples)
    n = x.shape[0]
    stages: list[str] = []
    if n == 0:
        return x.copy(), {"stages": stages, "method": "offline_filtfilt", "fs": fs}

    padded, used_mirror = mirror_pad(x)
    sos = emg_bandpass_sos(int(round(fs)))
    stages.extend(["高通2@20Hz", "低通4@450Hz"])
    y = signal.sosfiltfilt(sos, padded, axis=0)
    y = mirror_unpad(y, n) if used_mirror else np.asarray(y, dtype=np.float64)

    y = repair_periodic_edges(y, fs, COMB_F0_HZ)
    padded_comb, npre = _period_pad(y, fs, COMB_F0_HZ)
    b, a = comb_coefficients()
    if npre > 0:
        filtered = signal.filtfilt(b, a, padded_comb, axis=0)
        y = np.asarray(filtered[npre : npre + n], dtype=np.float64)
    else:
        y = np.asarray(signal.filtfilt(b, a, y, axis=0), dtype=np.float64)
    y = fade_edge_ring(y, fs)
    stages.append(f"梳状{COMB_F0_HZ:g}/{COMB_BANDWIDTH:g}Hz")
    note_bits = ["镜像padding", "filtfilt", "梳状周期延拓", "端点消振"]

    diagnostics = {
        "method": "offline_filtfilt",
        "fs": fs,
        "stages": stages,
        "note": " · ".join([*note_bits, f"fs={fs:g} Hz"]),
    }
    return y, diagnostics
