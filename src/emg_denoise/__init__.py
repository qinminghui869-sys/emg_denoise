"""EMG timestamp rebuild, offline zero-phase denoise, and causal streaming."""

from .dsp import (
    CausalFilterBank,
    DenoisedBlock,
    DenoiserConfig,
    Modality,
    RealtimeDenoiser,
    modality_preset,
)
from .pipeline import TimebaseReport, rebuild_timestamps

__version__ = "0.3.0"

__all__ = [
    "__version__",
    "DenoisedBlock",
    "DenoiserConfig",
    "RealtimeDenoiser",
    "CausalFilterBank",
    "Modality",
    "modality_preset",
    "TimebaseReport",
    "rebuild_timestamps",
]
