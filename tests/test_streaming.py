import numpy as np
import pytest

from emg_denoise import DenoiserConfig, RealtimeDenoiser
from emg_denoise.dsp.filters import CausalFilterBank, IIRComb, sample_matrix


def collect(engine, x, t, sizes=(20,)):
    result = []
    start = i = 0
    while start < len(x):
        count = sizes[i % len(sizes)]
        result.extend(engine.process(x[start : start + count], t[start : start + count]))
        start += count
        i += 1
    return result


def test_timestamps_units_validity_and_partial_tail():
    x = np.random.default_rng(7).normal(size=(5003, 2)) * 0.0002
    t = 1_700_000_000.0 + np.arange(len(x)) / 2000
    engine = RealtimeDenoiser(DenoiserConfig(channels=2))
    out = collect(engine, x, t, (17, 41, 3, 20))
    np.testing.assert_array_equal(np.concatenate([b.timestamps for b in out]), t)
    np.testing.assert_allclose(
        np.concatenate([b.samples for b in out]), CausalFilterBank(2).process(x), atol=1e-17
    )
    np.testing.assert_array_equal(np.concatenate([b.valid for b in out]), np.arange(len(x)) >= 2000)
    assert all(b.observed_end_to_end_ms is None for b in out)
    assert out[0].discontinuity and sum(b.discontinuity for b in out) == 1


def test_gap_resets_all_filter_state_and_warmup():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(3000, 1))
    engine = RealtimeDenoiser()
    collect(engine, x, np.arange(len(x)) / 2000)
    following = engine.process(x[:20], 5 + np.arange(20) / 2000)[0]
    expected = RealtimeDenoiser().process(x[:20], 5 + np.arange(20) / 2000)[0]
    np.testing.assert_array_equal(following.samples, expected.samples)
    assert following.discontinuity and following.segment_id == 1
    assert not following.valid.any()


def test_bad_input_does_not_poison_state_and_backward_time_requires_reset():
    engine = RealtimeDenoiser()
    t = np.arange(20) / 2000
    with pytest.raises(ValueError, match="finite"):
        engine.process(np.full(20, np.nan), t)
    with pytest.raises(ValueError, match="contiguous"):
        engine.process(np.zeros(20), np.arange(20) / 1000)
    out = engine.process(np.ones(20), t)[0]
    expected = RealtimeDenoiser().process(np.ones(20), t)[0]
    np.testing.assert_array_equal(out.samples, expected.samples)
    with pytest.raises(ValueError, match="backward"):
        engine.process(np.zeros(20), t)
    engine.reset()
    assert engine.process(np.zeros(20), t)[0].discontinuity


def test_actual_receipt_latency_is_separate_from_digital_estimate():
    t = np.arange(20) / 2000
    out = RealtimeDenoiser().process(np.zeros(20), t, received_at=0.05)[0]
    assert out.observed_end_to_end_ms >= 50
    assert out.buffering_delay_ms == pytest.approx(9.5)
    with pytest.raises(ValueError, match="received_at"):
        RealtimeDenoiser().process(np.zeros(20), t, received_at=0)


@pytest.mark.parametrize("constructor", [DenoiserConfig, CausalFilterBank, IIRComb])
def test_invalid_channel_counts_are_consistently_rejected(constructor):
    with pytest.raises(ValueError, match="positive integer"):
        constructor(channels=0)


def test_session_config_is_read_only_and_empty_inputs_have_defined_behavior():
    engine = RealtimeDenoiser()
    with pytest.raises(AttributeError):
        engine.config = DenoiserConfig(channels=2)
    with pytest.raises(ValueError, match="emg"):
        DenoiserConfig(modality="ecg")  # type: ignore[arg-type]
    assert engine.process(np.empty((0, 1)), []) == []
    with pytest.raises(ValueError, match="one timestamp"):
        engine.process(np.zeros(20), [0])
    with pytest.raises(ValueError, match="shape"):
        sample_matrix(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="channels"):
        sample_matrix(np.zeros((20, 2)), channels=1)
