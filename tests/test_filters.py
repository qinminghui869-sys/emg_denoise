import numpy as np
import pytest
from scipy import signal

from emg_denoise.dsp.filters import (
    CausalFilterBank,
    IIRComb,
    assert_emg_recording,
    bandpass_sos,
    comb_coefficients,
    modality_preset,
)


def test_comb_matches_independent_scipy_design_for_irregular_chunks():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(5003, 3))
    b, a = signal.iircomb(50, 25, fs=2000, ftype="notch", pass_zero=False)
    expected = signal.lfilter(b, a, x, axis=0)
    comb = IIRComb(3)
    sizes = [1, 17, 40, 83, 2, 20]
    got, start, i = [], 0, 0
    while start < len(x):
        n = sizes[i % len(sizes)]
        got.append(comb.process(x[start : start + n]))
        start += n
        i += 1
    np.testing.assert_allclose(np.concatenate(got), expected, atol=2e-13, rtol=2e-12)
    actual_b, actual_a = comb_coefficients()
    np.testing.assert_allclose(actual_b, b, atol=1e-15)
    np.testing.assert_allclose(actual_a, a, atol=1e-15)


def test_causal_emg_chain_matches_scipy_and_channels_are_independent():
    x = np.random.default_rng(4).normal(size=(8000, 4))
    b, a = signal.iircomb(50, 25, fs=2000)
    reference = signal.lfilter(b, a, signal.sosfilt(bandpass_sos(), x, axis=0), axis=0)
    bank = CausalFilterBank(4)
    got = np.concatenate([bank.process(x[i : i + 20]) for i in range(0, len(x), 20)])
    np.testing.assert_allclose(got, reference, atol=1e-12)
    for c in range(4):
        np.testing.assert_allclose(
            CausalFilterBank(1).process(x[:, c])[:, 0], got[:, c], atol=1e-12
        )


def test_zero_weak_signal_linearity_and_reset():
    x = np.random.default_rng(5).normal(size=(4000, 1))
    bank = CausalFilterBank(1)
    np.testing.assert_array_equal(bank.process(np.zeros((4000, 1))), 0)
    y = bank.process(x)
    np.testing.assert_allclose(CausalFilterBank(1).process(x * 1e-9), y * 1e-9, atol=1e-22)
    bank.reset()
    np.testing.assert_array_equal(bank.process(x), y)


@pytest.mark.parametrize("k", [1, 2, 3, 5, 9, 13, 19])
def test_nominal_harmonics_at_least_30db_after_one_second(k):
    t = np.arange(8000) / 2000
    x = np.sin(2 * np.pi * 50 * k * t)
    y = IIRComb(1).process(x)[:, 0]
    gain = np.linalg.norm(y[2000:]) / np.linalg.norm(x[2000:])
    assert gain < 10 ** (-30 / 20)


def test_comb_keeps_off_notch_band_and_reports_drift_limitation():
    b, a = comb_coefficients()
    f = np.linspace(20, 450, 17201)
    _, h = signal.freqz(b, a, worN=f, fs=2000)
    outside = np.abs((f + 25) % 50 - 25) >= 5 - 1e-9
    assert np.min(np.abs(h[outside])) >= 10 ** (-0.5 / 20)
    _, drift = signal.freqz(b, a, worN=[49.8, 50.2], fs=2000)
    np.testing.assert_allclose(20 * np.log10(np.abs(drift)), -14.1603, atol=0.001)


def test_only_emg_is_accepted():
    assert modality_preset("emg").sample_rate == 2000
    with pytest.raises(ValueError, match="emg"):
        modality_preset("eeg")
    assert_emg_recording("observation.left_wrist_emg.parquet")
    assert_emg_recording("/data/eeg_session/observation.left_wrist_emg.parquet")
    with pytest.raises(ValueError, match="EEG"):
        assert_emg_recording("observation.eeg.parquet")
