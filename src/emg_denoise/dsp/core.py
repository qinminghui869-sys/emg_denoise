"""Timestamp-preserving streaming interface with explicit quality and latency state."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .filters import (
    CausalFilterBank,
    Modality,
    ModalityPreset,
    resolved_preset,
    sample_matrix,
    validate_channels,
)


@dataclass(frozen=True, slots=True)
class DenoiserConfig:
    """Immutable session configuration for the EMG causal chain."""

    channels: int = 1
    modality: Modality = "emg"
    #: Override the 2000 Hz default. The comb delay is fixed, so the rate must stay near 2 kHz.
    sample_rate_hz: int | None = None

    def __post_init__(self) -> None:
        validate_channels(self.channels)
        resolved_preset(self.modality, self.sample_rate_hz)

    @property
    def preset(self) -> ModalityPreset:
        return resolved_preset(self.modality, self.sample_rate_hz)

    @property
    def sample_rate(self) -> int:
        return self.preset.sample_rate

    @property
    def block_size(self) -> int:
        return self.preset.block_size

    @property
    def warmup_samples(self) -> int:
        return self.preset.warmup_samples


@dataclass(frozen=True, slots=True)
class DenoisedBlock:
    """Aligned voltage samples, acquisition timestamps and software latency observations."""

    samples: NDArray[np.float64]
    timestamps: NDArray[np.float64]
    valid: NDArray[np.bool_]
    discontinuity: bool
    segment_id: int
    processing_ms: float
    buffering_delay_ms: float
    estimated_digital_latency_ms: float
    observed_end_to_end_ms: float | None


class RealtimeDenoiser:
    """Causal streaming denoiser for EMG at 2 kHz.

    process() accepts any positive sample count and internally consumes at most
    one hop (20 samples at 2000 Hz) at a time.
    A forward timestamp gap resets state and warmup. Backward/repeated time is
    rejected; call reset() before restarting a device's timestamp clock.
    """

    def __init__(self, config: DenoiserConfig | None = None) -> None:
        self._config = config or DenoiserConfig()
        self._bank = CausalFilterBank(
            self.config.channels,
            self.config.modality,
            sample_rate=self.config.sample_rate,
        )
        self.segment_id = -1
        self.reset()

    @property
    def config(self) -> DenoiserConfig:
        return self._config

    def reset(self) -> None:
        """Preserve session config; clear all signal and time state."""
        self._bank.reset()
        self._received = 0
        self._last_timestamp: float | None = None
        self._pending_discontinuity = True
        self.segment_id += 1

    def process(
        self, samples: ArrayLike, timestamps: ArrayLike, *, received_at: float | None = None
    ) -> list[DenoisedBlock]:
        """Return ready output blocks for the submitted samples.

        received_at is optional host receipt time in the SAME clock domain as
        acquisition timestamps. Without it, observed end-to-end latency is None;
        the reported digital estimate excludes transport and scheduling delays.
        """
        call_started = perf_counter()
        fs = self.config.sample_rate
        block = self.config.block_size
        warmup = self.config.warmup_samples
        x = sample_matrix(samples, self.config.channels)
        t = np.asarray(timestamps, dtype=np.float64)
        if t.ndim != 1 or len(t) != len(x) or not np.isfinite(t).all():
            raise ValueError("timestamps must be a finite vector, one timestamp per sample")
        if not len(x):
            return []
        if len(t) > 1 and not np.all(np.abs(np.diff(t) - 1 / fs) <= 1e-6):
            raise ValueError(
                f"timestamps inside a block must be contiguous at {fs} Hz; split at gaps"
            )
        if received_at is not None and (not np.isfinite(received_at) or received_at < t[-1]):
            raise ValueError(
                "received_at must be finite, in the acquisition clock, and >= the last sample time"
            )
        # Validate the whole input BEFORE modifying filter state.
        if self._last_timestamp is not None:
            delta = t[0] - self._last_timestamp
            if delta <= 0:
                raise ValueError(
                    "backward/duplicate timestamps; call reset() for a restarted device clock"
                )
            if abs(delta - 1 / fs) > 1e-6:
                self.reset()
        result = []
        for offset in range(0, len(x), block):
            hop_started = call_started if offset == 0 else perf_counter()
            xx, tt = x[offset : offset + block], t[offset : offset + block]
            filtered = self._bank.process(xx)
            valid = np.arange(self._received, self._received + len(xx)) >= warmup
            self._received += len(xx)
            self._last_timestamp = float(tt[-1])
            now = perf_counter()
            cost_ms = (now - hop_started) * 1000
            wait_ms = max(0.0, float(tt[-1] - tt[0]) * 1000)
            observed = (
                None
                if received_at is None
                else float((received_at - tt[0]) * 1000 + (now - call_started) * 1000)
            )
            result.append(
                DenoisedBlock(
                    samples=filtered,
                    timestamps=tt.copy(),
                    valid=valid,
                    discontinuity=self._pending_discontinuity,
                    segment_id=self.segment_id,
                    processing_ms=cost_ms,
                    buffering_delay_ms=wait_ms,
                    estimated_digital_latency_ms=wait_ms + cost_ms,
                    observed_end_to_end_ms=observed,
                )
            )
            self._pending_discontinuity = False
        return result
