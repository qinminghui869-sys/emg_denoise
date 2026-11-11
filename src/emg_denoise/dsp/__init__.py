"""EMG DSP: fixed filters, offline zero-phase denoise, and the causal stream."""

from .core import DenoisedBlock, DenoiserConfig, RealtimeDenoiser
from .filters import (
    CausalFilterBank,
    IIRComb,
    Modality,
    ModalityPreset,
    modality_preset,
)
from .offline import offline_denoise

__all__ = [
    "DenoisedBlock",
    "DenoiserConfig",
    "RealtimeDenoiser",
    "CausalFilterBank",
    "IIRComb",
    "Modality",
    "ModalityPreset",
    "modality_preset",
    "offline_denoise",
]
