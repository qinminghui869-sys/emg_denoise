import json
from pathlib import Path

import numpy as np
import pytest

from emg_denoise.cli import main
from emg_denoise.dsp.filters import FS
from emg_denoise.io import load_recording
from emg_denoise.pipeline.retime import RetimeOptions, load_acquisition, run_retime
from emg_denoise.pipeline.timebase import (
    OBSERVED_TICK_S,
    estimate_tick,
    find_resync_points,
    rebuild_timestamps,
)

PERIOD_S = 1.0 / FS


def quantized_stamps(samples: int, *, tick: float = OBSERVED_TICK_S, start: float = 0.0):
    """Reproduce acquisition behaviour: a true period reported on a coarser clock."""
    ideal = start + np.arange(samples, dtype=np.float64) * PERIOD_S
    return np.floor(ideal / tick).astype(np.float64) * tick


def test_quantized_fixture_reproduces_the_four_observed_gaps():
    gaps = np.unique(np.round(np.diff(quantized_stamps(4000)) * 1e6, 3))
    assert gaps.tolist() == [451.088, 902.176]


def test_estimate_tick_recovers_the_clock_resolution():
    assert estimate_tick(quantized_stamps(4000)) == pytest.approx(OBSERVED_TICK_S, rel=1e-9)


def test_resync_points_match_the_two_tick_jumps():
    stamps = quantized_stamps(4000)
    expected = np.flatnonzero(np.rint(np.diff(stamps) / OBSERVED_TICK_S) >= 2) + 1
    assert find_resync_points(stamps, OBSERVED_TICK_S).tolist() == expected.tolist()


def test_stitched_rebuild_is_exactly_uniform():
    samples = 4000
    rebuilt, report = rebuild_timestamps(
        method="uniform", timestamps=quantized_stamps(samples), period_s=PERIOD_S
    )
    assert np.abs(np.diff(rebuilt) - PERIOD_S).max() < 1e-12
    # Quantization leaves up to a tick at each endpoint, so the measured period
    # is only recoverable to within tick/(n-1).
    assert report.measured_period_s == pytest.approx(PERIOD_S, abs=OBSERVED_TICK_S / (samples - 1))
    assert report.warnings() == ()


def test_segment_length_matches_the_resync_prediction():
    _, report = rebuild_timestamps(
        method="uniform", timestamps=quantized_stamps(20000), period_s=PERIOD_S
    )
    mean_length = report.samples / report.segments
    assert mean_length == pytest.approx(report.expected_segment_samples, rel=0.02)


def test_unstitched_rebuild_stays_within_half_a_tick():
    """Centroid anchoring bounds the offset; shifting to zero would undo that."""
    stamps = quantized_stamps(4000, start=1.0)
    rebuilt, report = rebuild_timestamps(
        method="uniform", timestamps=stamps, period_s=PERIOD_S, stitch=False, non_negative=False
    )
    assert report.max_offset_s <= OBSERVED_TICK_S / 2
    assert np.abs(rebuilt - stamps).max() <= OBSERVED_TICK_S / 2


def test_stitching_trades_seams_for_accumulated_offset():
    stamps = quantized_stamps(4000, start=1.0)
    _, stitched = rebuild_timestamps(
        method="uniform", timestamps=stamps, period_s=PERIOD_S, non_negative=False
    )
    _, anchored = rebuild_timestamps(
        method="uniform", timestamps=stamps, period_s=PERIOD_S, stitch=False, non_negative=False
    )
    assert anchored.max_offset_s < stitched.max_offset_s


def test_non_negative_shifts_the_axis_to_zero():
    stamps = quantized_stamps(400)
    rebuilt, _ = rebuild_timestamps(
        method="uniform", timestamps=stamps, period_s=PERIOD_S, non_negative=True
    )
    assert rebuilt[0] >= 0.0


def test_allowing_negative_preserves_the_centroid_anchor():
    stamps = quantized_stamps(400)
    anchored, _ = rebuild_timestamps(
        method="uniform", timestamps=stamps, period_s=PERIOD_S, non_negative=False
    )
    shifted, _ = rebuild_timestamps(
        method="uniform", timestamps=stamps, period_s=PERIOD_S, non_negative=True
    )
    assert np.abs(np.diff(anchored) - PERIOD_S).max() < 1e-12
    assert shifted[0] == pytest.approx(0.0)


def test_report_flags_a_period_that_contradicts_the_stamps():
    _, report = rebuild_timestamps(quantized_stamps(4000), period_s=1.0 / 500)
    assert any("differs from the requested" in warning for warning in report.warnings())


@pytest.mark.parametrize(
    "stamps, message",
    [
        (np.array([0.0]), "at least two"),
        (np.array([[0.0, 1.0]]), "one-dimensional"),
        (np.array([0.0, np.nan]), "finite"),
        (np.array([0.0, 0.0]), "strictly increasing"),
        (np.array([1.0, 0.0]), "strictly increasing"),
    ],
)
def test_rebuild_rejects_unusable_timestamps(stamps, message):
    with pytest.raises(ValueError, match=message):
        rebuild_timestamps(stamps, period_s=PERIOD_S)


def test_rebuild_rejects_a_nonpositive_period():
    with pytest.raises(ValueError, match="period must be positive"):
        rebuild_timestamps(quantized_stamps(100), period_s=0.0)


def test_rebuild_rejects_a_nonpositive_tick():
    with pytest.raises(ValueError, match="tick must be positive"):
        rebuild_timestamps(quantized_stamps(100), period_s=PERIOD_S, tick_s=0.0)


def write_acquisition_csv(path, samples=4000, channels=2):
    """An acquisition-style CSV: quantized stamps and ``dim_N`` channel columns."""
    stamps = quantized_stamps(samples)
    rows = ["timestamp," + ",".join(f"dim_{i}" for i in range(channels))]
    for index, stamp in enumerate(stamps):
        values = ",".join(
            format(float(index * 10 + channel), ".17g") for channel in range(channels)
        )
        rows.append(f"{format(stamp, '.17g')},{values}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return stamps


def test_raw_acquisition_is_rejected_by_stage_two(tmp_path):
    """Stage one exists because the denoiser refuses tick-quantized input."""
    source = tmp_path / "raw.csv"
    write_acquisition_csv(source, samples=100, channels=1)
    with pytest.raises(ValueError, match="contiguous"):
        load_recording(source)


def test_retime_output_is_accepted_by_stage_two(tmp_path):
    source, output = tmp_path / "raw.csv", tmp_path / "retimed.csv"
    write_acquisition_csv(source, samples=4000, channels=2)
    payload = run_retime(RetimeOptions(source=source, output=output))

    recording = load_acquisition(output)
    assert recording.channel_names == ("dim_0", "dim_1")
    assert recording.samples.shape == (payload["samples"], 2)
    assert payload["warnings"] == []
    assert payload["resync_segments"] > 1


def test_retime_preserves_sample_values_exactly(tmp_path):
    source, output = tmp_path / "raw.csv", tmp_path / "retimed.csv"
    write_acquisition_csv(source, samples=600, channels=3)
    run_retime(RetimeOptions(source=source, output=output))
    assert np.array_equal(load_acquisition(source).samples, load_acquisition(output).samples)


def test_retime_writes_a_report_beside_the_output(tmp_path):
    source, output = tmp_path / "raw.csv", tmp_path / "retimed.csv"
    write_acquisition_csv(source, samples=600, channels=1)
    run_retime(RetimeOptions(source=source, output=output, sidecar=True))
    report = json.loads((tmp_path / "retimed.csv.timebase.json").read_text())
    assert report["tick_us"] == pytest.approx(OBSERVED_TICK_S * 1e6, rel=1e-6)
    assert report["measured_period_us"] == pytest.approx(PERIOD_S * 1e6, abs=1.0)


def test_retime_skips_sidecar_by_default(tmp_path):
    source, output = tmp_path / "raw.csv", tmp_path / "retimed.csv"
    write_acquisition_csv(source, samples=4000, channels=1)
    payload = run_retime(RetimeOptions(source=source, output=output))
    assert payload["report"] is None
    assert not (tmp_path / "retimed.csv.timebase.json").exists()


def test_retime_rejects_an_output_that_would_clobber_the_source(tmp_path):
    source = tmp_path / "raw.csv"
    write_acquisition_csv(source, samples=100, channels=1)
    with pytest.raises(ValueError, match="distinct"):
        run_retime(RetimeOptions(source=source, output=source))


def test_inplace_retime_backs_up_and_rewrites_parquet(tmp_path):
    source = tmp_path / "observation.left_wrist_emg.parquet"
    write_acquisition_parquet(source, samples=4000, channels=2)
    original = source.read_bytes()
    payload = run_retime(RetimeOptions(source=source, inplace=True))

    backup = Path(f"{source}.bak")
    assert backup.is_file()
    assert backup.read_bytes() == original
    assert source.is_file()
    assert payload["backup"] == str(backup)
    assert payload["warnings"] == []

    rebuilt = load_acquisition(source)
    assert np.all(np.diff(rebuilt.timestamps) > 0)
    assert rebuilt.timestamps[0] == load_acquisition(source).timestamps[0]
    assert np.array_equal(rebuilt.samples, load_acquisition(backup).samples)
    assert "episode_index" not in rebuilt.channel_names
    assert list(rebuilt.extras) == ["episode_index"]


def test_inplace_refuses_an_existing_backup_without_force(tmp_path):
    source = tmp_path / "observation.left_wrist_emg.parquet"
    write_acquisition_parquet(source, samples=200, channels=1)
    run_retime(RetimeOptions(source=source, inplace=True))
    with pytest.raises(ValueError, match="backup exists"):
        run_retime(RetimeOptions(source=source, inplace=True))


def test_inplace_force_keeps_the_original_backup(tmp_path):
    source = tmp_path / "observation.left_wrist_emg.parquet"
    write_acquisition_parquet(source, samples=800, channels=1)
    run_retime(RetimeOptions(source=source, inplace=True))
    backup_bytes = Path(f"{source}.bak").read_bytes()
    first = load_acquisition(source).timestamps.copy()
    run_retime(RetimeOptions(source=source, inplace=True, force=True))
    assert Path(f"{source}.bak").read_bytes() == backup_bytes
    assert np.allclose(load_acquisition(source).timestamps, first)


def test_parquet_output_path_writes_parquet(tmp_path):
    source = tmp_path / "raw.parquet"
    output = tmp_path / "retimed.parquet"
    write_acquisition_parquet(source, samples=600, channels=2)
    run_retime(RetimeOptions(source=source, output=output))
    rebuilt = load_acquisition(output)
    assert np.all(np.diff(rebuilt.timestamps) > 0)
    assert rebuilt.timestamps[0] == load_acquisition(source).timestamps[0]
    assert source.is_file()
    assert not Path(f"{source}.bak").exists()


def test_retime_rejects_a_recording_without_a_timestamp_column(tmp_path):
    source = tmp_path / "raw.csv"
    source.write_text("time,dim_0\n0,1\n0.0005,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="timestamp"):
        load_acquisition(source)


def test_retime_rejects_a_recording_without_channels(tmp_path):
    source = tmp_path / "raw.csv"
    source.write_text("timestamp,episode_index\n0,0\n0.0005,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no channel columns"):
        load_acquisition(source)


def test_retime_rejects_a_nonpositive_rate(tmp_path):
    source = tmp_path / "raw.csv"
    write_acquisition_csv(source, samples=100, channels=1)
    with pytest.raises(ValueError, match="rate must be positive"):
        run_retime(RetimeOptions(source=source, output=tmp_path / "out.csv", rate_hz=0.0))


def test_cli_retime_then_process_completes_both_stages(tmp_path, capsys):
    source = tmp_path / "raw.csv"
    retimed, denoised = tmp_path / "retimed.csv", tmp_path / "denoised.csv"
    write_acquisition_csv(source, samples=4000, channels=1)

    assert main(["retime", "--json", str(source), "--output", str(retimed)]) == 0
    stage_one = json.loads(capsys.readouterr().out)
    assert stage_one["rate_hz"] == float(FS)

    assert main(["process", "--json", str(retimed), "--output", str(denoised)]) == 0
    stage_two = json.loads(capsys.readouterr().out)
    # A rebuilt axis must not look discontinuous to the streaming engine.
    assert stage_two["segments"] == 1
    assert stage_two["input_samples"] == stage_one["samples"]


def test_cli_retime_reports_failure_when_the_rate_contradicts_the_data(tmp_path, capsys):
    source = tmp_path / "raw.csv"
    write_acquisition_csv(source, samples=2000, channels=1)
    exit_code = main(
        [
            "retime",
            "--json",
            str(source),
            "--output",
            str(tmp_path / "out.csv"),
            "--rate",
            "500",
        ]
    )
    assert exit_code == 1
    assert json.loads(capsys.readouterr().out)["warnings"]


def test_cli_retime_rejects_a_missing_input(tmp_path):
    assert main(["retime", str(tmp_path / "absent.csv"), "--output", str(tmp_path / "o.csv")]) == 2


def write_acquisition_parquet(path, samples=4000, channels=8, columns=None):
    """An acquisition-style parquet, matching how the recorder stores episodes."""
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    stamps = quantized_stamps(samples)
    data = {
        "timestamp": pa.array(stamps, type=pa.float64()),
        "episode_index": pa.array(np.zeros(samples, dtype=np.int64)),
    }
    for channel in range(channels):
        values = np.arange(samples, dtype=np.float32) * 10.0 + channel
        data[f"dim_{channel}"] = pa.array(values, type=pa.float32())
    table = pa.table(data if columns is None else {k: data[k] for k in columns})
    pq.write_table(table, path)
    return stamps


def test_parquet_channels_are_detected_in_numeric_order(tmp_path):
    source = tmp_path / "raw.parquet"
    write_acquisition_parquet(source, samples=200, channels=12)
    acquisition = load_acquisition(source)
    # dim_10 must follow dim_9, which lexicographic ordering would not do.
    assert acquisition.channel_names == tuple(f"dim_{i}" for i in range(12))
    assert acquisition.samples.shape == (200, 12)


def test_parquet_metadata_columns_are_not_treated_as_channels(tmp_path):
    source = tmp_path / "raw.parquet"
    write_acquisition_parquet(source, samples=200, channels=3)
    assert "episode_index" not in load_acquisition(source).channel_names


def test_legacy_valid_segment_columns_are_not_signal_channels(tmp_path):
    """Old denoised files may carry QC columns; upstream must not see them as EMG."""
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    source = tmp_path / "legacy.denoised.parquet"
    n = 100
    pq.write_table(
        pa.table(
            {
                "timestamp": pa.array(np.arange(n, dtype=np.float64) / 2000.0),
                "episode_index": pa.array(np.zeros(n, dtype=np.int64)),
                "dim_0": pa.array(np.ones(n, dtype=np.float32)),
                "dim_1": pa.array(np.ones(n, dtype=np.float32) * 2),
                "valid": pa.array(np.ones(n, dtype=bool)),
                "segment_id": pa.array(np.zeros(n, dtype=np.int64)),
            }
        ),
        source,
    )
    acquisition = load_acquisition(source)
    assert acquisition.channel_names == ("dim_0", "dim_1")
    assert acquisition.samples.shape == (n, 2)
    assert "valid" in acquisition.extras
    assert "segment_id" in acquisition.extras


def test_parquet_retime_feeds_stage_two(tmp_path):
    source, output = tmp_path / "raw.parquet", tmp_path / "retimed.csv"
    write_acquisition_parquet(source, samples=4000, channels=8)
    payload = run_retime(RetimeOptions(source=source, output=output))

    assert payload["channels"] == 8
    assert payload["warnings"] == []
    recording = load_acquisition(output)
    assert recording.samples.shape == (payload["samples"], 8)
    assert np.array_equal(load_acquisition(source).samples, recording.samples)


def test_parquet_without_a_timestamp_column_is_rejected(tmp_path):
    source = tmp_path / "raw.parquet"
    write_acquisition_parquet(source, samples=100, channels=2, columns=["dim_0", "dim_1"])
    with pytest.raises(ValueError, match="timestamp"):
        load_acquisition(source)


def test_parquet_without_channels_is_rejected(tmp_path):
    source = tmp_path / "raw.parquet"
    write_acquisition_parquet(source, samples=100, columns=["timestamp", "episode_index"])
    with pytest.raises(ValueError, match="no channel columns"):
        load_acquisition(source)


def test_single_sample_recording_is_rejected(tmp_path):
    source = tmp_path / "raw.parquet"
    write_acquisition_parquet(source, samples=1, channels=1)
    with pytest.raises(ValueError, match="at least two samples"):
        load_acquisition(source)


def test_retime_accepts_an_explicit_tick(tmp_path):
    source, output = tmp_path / "raw.parquet", tmp_path / "retimed.csv"
    write_acquisition_parquet(source, samples=2000, channels=1)
    payload = run_retime(RetimeOptions(source=source, output=output, tick_us=OBSERVED_TICK_S * 1e6))
    assert payload["tick_us"] == pytest.approx(OBSERVED_TICK_S * 1e6)


def test_retime_without_stitching_leaves_segment_seams(tmp_path):
    source = tmp_path / "raw.parquet"
    write_acquisition_parquet(source, samples=2000, channels=1)
    options = RetimeOptions(
        source=source, output=tmp_path / "seamed.csv", stitch=False, method="uniform"
    )
    # Unstitched output keeps each segment aligned but is no longer uniform,
    # so it must not silently reach stage two.
    with pytest.raises(ValueError, match="deviate from the period"):
        run_retime(options)


def test_anchor_interpolation_variable_spacing_and_untouched_edges():
    tick = OBSERVED_TICK_S
    gaps = np.ones(50) * tick
    gaps[[4, 13, 31, 42]] = 2 * tick
    stamps = -0.002 + np.r_[0, np.cumsum(gaps)]
    rebuilt, report = rebuild_timestamps(stamps, tick_s=tick)
    anchors = np.array([5, 14, 32, 43])
    assert report.anchor_indices == tuple(anchors)
    np.testing.assert_array_equal(rebuilt[:6], stamps[:6])
    np.testing.assert_array_equal(rebuilt[43:], stamps[43:])
    np.testing.assert_array_equal(rebuilt[anchors], stamps[anchors])
    for a, b in zip(anchors[:-1], anchors[1:], strict=True):
        np.testing.assert_allclose(
            np.diff(rebuilt[a : b + 1]), (stamps[b] - stamps[a]) / (b - a), atol=1e-15, rtol=0
        )
    assert report.interpolated_intervals == ((5, 14), (14, 32), (32, 43))
    assert rebuilt[0] < 0  # no global shift even with default non_negative=True


@pytest.mark.parametrize("jumps", [[], [4]])
def test_insufficient_anchors_leave_all_samples_unchanged(jumps):
    gaps = np.full(30, OBSERVED_TICK_S)
    gaps[jumps] *= 2
    stamps = np.r_[0, np.cumsum(gaps)]
    rebuilt, report = rebuild_timestamps(stamps)
    np.testing.assert_array_equal(rebuilt, stamps)
    assert report.interpolated_intervals == ()


def test_gap_is_not_anchor_and_not_smeared():
    stamps = quantized_stamps(1000)
    stamps[500:] += 0.1
    rebuilt, report = rebuild_timestamps(stamps)
    assert 500 not in report.anchor_indices
    assert report.skipped_intervals
    for a, b in report.skipped_intervals:
        np.testing.assert_array_equal(rebuilt[a : b + 1], stamps[a : b + 1])
    assert rebuilt[500] - rebuilt[499] == stamps[500] - stamps[499]


def test_retime_then_offline_preserves_times_and_rejects_causal(tmp_path):
    from emg_denoise.pipeline.replay import ReplayOptions, run_replay

    source, retimed = tmp_path / "raw.csv", tmp_path / "retimed.csv"
    original = write_acquisition_csv(source, samples=4000, channels=1)
    run_retime(RetimeOptions(source=source, output=retimed))
    recording = load_acquisition(retimed)
    assert recording.timestamps[0] == original[0]
    assert recording.timestamps[-1] == original[-1]
    out = tmp_path / "filtered.csv"
    result = run_replay(ReplayOptions(source=retimed, output=out))
    assert result["resampled_segments"] == 1
    np.testing.assert_array_equal(load_acquisition(out).timestamps, recording.timestamps)
    assert np.isfinite(load_acquisition(out).samples).all()
    with pytest.raises(ValueError, match="causal processing requires a uniform"):
        run_replay(ReplayOptions(source=retimed, output=tmp_path / "causal.csv", causal=True))


def test_offline_irregular_matches_explicit_grid_reference():
    from emg_denoise.dsp.offline import offline_denoise
    from emg_denoise.pipeline.replay import _denoise_arrays

    t, _ = rebuild_timestamps(quantized_stamps(4000))
    x = (np.sin(2 * np.pi * 80 * t) + 0.2 * np.sin(2 * np.pi * 50 * t))[:, None]
    stamps, actual, valid, _, report = _denoise_arrays(
        t, x, modality="emg", causal=False, sample_rate_hz=2000
    )
    grid = np.linspace(t[0], t[-1], round((t[-1] - t[0]) * 2000) + 1)
    regular = np.interp(grid, t, x[:, 0])[:, None]
    filtered, _ = offline_denoise(
        regular, modality="emg", sample_rate=(len(grid) - 1) / (grid[-1] - grid[0])
    )
    expected = np.interp(t, grid, filtered[:, 0])
    np.testing.assert_allclose(actual[:, 0], expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(stamps, t)
    assert valid.all() and report.resampled_segments == 1


def test_offline_short_segment_is_explicitly_invalid():
    from emg_denoise.pipeline.replay import _denoise_arrays

    t = np.arange(10) / 2000
    x = np.ones((10, 1))
    stamps, y, valid, _, report = _denoise_arrays(
        t, x, modality="emg", causal=False, sample_rate_hz=2000
    )
    np.testing.assert_array_equal(y, x)
    np.testing.assert_array_equal(stamps, t)
    assert not valid.any()
    assert report.unfiltered_samples == 10 and report.valid_samples == 0
