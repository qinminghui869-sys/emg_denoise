from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from emg_denoise.cli import main
from emg_denoise.qc import DEFAULTS, QcOptions, run_qc
from emg_denoise.qc.discovery import channel_names, discover
from emg_denoise.qc.signal import metrics, spectrum, windows


def write(path, t, **signals):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"timestamp": t, **signals}), path)
    return path


def setup_session(tmp_path, method="offline_filtfilt", samples=8000):
    root = tmp_path / "session"
    path = root / "data/chunk-001/episode_001000/observation.left_wrist_emg.parquet"
    rng = np.random.default_rng(4)
    t = np.arange(samples) / 2000
    x = rng.normal(size=samples) + 3 * np.sin(2 * np.pi * 50 * t)
    write(path, t, dim_0=x, episode_index=np.full(samples, 1000))
    derived = path.with_name(path.stem + ".denoised.parquet")
    write(
        derived, t, dim_0=x - 3 * np.sin(2 * np.pi * 50 * t), episode_index=np.full(samples, 1000)
    )
    (root / "meta").mkdir()
    info = {
        "features": {
            path.stem: {"fps": 2000, "shape": [1], "timebase": {"rebuilt": True}},
            derived.stem: {"shape": [1], "pipeline": {"method": method}},
        }
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/episodes_stats.jsonl").write_text(
        json.dumps({"episode_index": 1000, "stats": {derived.stem: {}}}) + "\n"
    )
    return root, path, derived, t, x


def rules(report):
    return {r["rule"] for r in report["findings"]}


def test_session_comparison_and_readonly(tmp_path):
    root, path, derived, t, x = setup_session(tmp_path)
    write(Path(str(path) + ".bak"), t + 0.0003, dim_0=x, episode_index=np.full(len(t), 1000))
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
    report = run_qc(QcOptions((str(root),), output_dir=tmp_path / "report"))
    assert report["exit_code"] == 0
    assert report["warnings"] > 0  # unfiltered 50 Hz energy
    assert {"alignment", "time_shift", "sample_integrity"} <= rules(report)
    compare = next(r for r in report["channels"] if r["stage"] == "comparison")
    assert compare["line_power_change_db"] < -10
    assert compare["windows"] == 3
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in hashes} == hashes
    assert json.loads((tmp_path / "report/summary.json").read_text()) == report
    assert (tmp_path / "report/channels.csv").exists()
    assert run_qc(QcOptions((str(root),), strict=True))["exit_code"] == 1


def test_bad_signal_time_and_schema(tmp_path):
    t = np.arange(8000) / 2000
    t[100] = t[99]
    t[4000:] += 0.02
    x = np.ones(8000)
    y = np.random.default_rng(1).normal(size=8000)
    y[200:202] = [np.nan, np.inf]
    path = write(tmp_path / "emg.parquet", t, dim_10=y, dim_2=x, valid=np.ones(8000, dtype=bool))
    report = run_qc(QcOptions((str(path),), require_denoised=True))
    assert report["exit_code"] == 1
    assert {
        "constant",
        "signal_finite",
        "time_order",
        "time_gap",
        "legacy_qc_column",
        "derived_missing",
    } <= rules(report)
    assert [r["channel"] for r in report["channels"]] == ["dim_2", "dim_10"]
    finding = next(f for f in report["findings"] if f["rule"] == "signal_finite")
    assert (finding["start_sample"], finding["stop_sample"]) == (200, 202)


def test_tick_jitter_is_not_gap(tmp_path):
    t = np.floor(np.arange(10000) * 0.0005 / 0.00045108795) * 0.00045108795
    path = write(tmp_path / "emg.parquet", t, dim_0=np.random.default_rng(1).normal(size=len(t)))
    report = run_qc(QcOptions((str(path),)))
    assert "time_gap" not in rules(report)
    assert "sample_rate" not in rules(report)


def test_windows_do_not_cross_bad_samples_or_time(tmp_path):
    t = np.arange(12000) / 2000
    x = np.ones(12000)
    x[4500] = np.nan
    t[9000:] += 0.02
    result = windows(t, x, 2000, 2, 1, 3)
    assert all(not (a <= 4500 < b) and not (a < 9000 < b) for a, b in result)
    m, _ = metrics(t, x, 2000, DEFAULTS, 2, 1)
    assert m["longest_constant_s"] < 2.3
    assert spectrum(np.zeros(4000), 2000)["line_fraction"] is None
    assert spectrum(np.ones(1000), 250)["band_fraction"] is None


def test_csv_nonnumeric_empty_and_continue(tmp_path):
    (tmp_path / "bad_emg.csv").write_text("timestamp,a\n0,hello\n0.0005,\n0.001,1\n")
    (tmp_path / "empty_emg.csv").write_text("timestamp,a\n")
    (tmp_path / "broken_emg.csv").write_text("timestamp,a\n0,1,2\n")
    (tmp_path / "missing_emg.csv").write_text("a,b\n1,2\n")
    report = run_qc(QcOptions((str(tmp_path),)))
    assert len(report["files"]) == 4
    assert report["exit_code"] == 2
    assert {"nonnumeric", "signal_finite", "empty", "read", "schema"} <= rules(report)


def test_invalid_config_and_inputs(tmp_path):
    with pytest.raises(ValueError, match="no inputs"):
        run_qc(QcOptions((str(tmp_path / "absent"),)))
    with pytest.raises(ValueError, match="no EMG"):
        discover((str(tmp_path),))
    for value in (
        {"bad": 2},
        {"adc_min": 0},
        {"gap_periods": 0},
        {"flatline_s": True},
        {"rate_warn_fraction": 0.1},
        {"line_fraction_warn": 2},
        {"adc_min": 2, "adc_max": 1},
        [],
    ):
        config = tmp_path / "config.json"
        config.write_text(json.dumps(value))
        with pytest.raises(ValueError):
            run_qc(QcOptions((str(tmp_path),), config=config))
    for rate in (0, -1, float("nan")):
        with pytest.raises(ValueError):
            run_qc(QcOptions((str(tmp_path),), rate=rate))


def test_shape_rate_uniform_stats_and_mismatch(tmp_path):
    root, path, derived, t, x = setup_session(tmp_path)
    info = json.loads((root / "meta/info.json").read_text())
    info["features"][path.stem]["shape"] = [2]
    info["features"][path.stem]["fps"] = 1000
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/episodes_stats.jsonl").unlink()
    write(derived, t + 0.1, dim_0=x)
    write(Path(str(path) + ".bak"), t, dim_0=x + 1)
    report = run_qc(QcOptions((str(root),)))
    assert {
        "shape",
        "sample_rate",
        "rebuilt_uniform",
        "stats",
        "sample_integrity",
        "alignment",
    } <= rules(report)
    assert not any(r["stage"] == "comparison" for r in report["channels"])
    assert report["exit_code"] == 1


def test_causal_mask_and_unknown_method(tmp_path):
    root, path, derived, t, x = setup_session(tmp_path, method="causal_stream")
    report = run_qc(QcOptions((str(path),)))
    assert "mask" in rules(report)
    mask = Path(str(derived) + ".mask.parquet")
    write(mask, t, valid=np.arange(len(t)) >= 2000, segment_id=np.zeros(len(t), dtype=int))
    report = run_qc(QcOptions((str(path),)))
    assert any(r["stage"] == "comparison" for r in report["channels"])
    write(mask, t + 0.1, valid=np.ones(len(t), dtype=bool))
    assert run_qc(QcOptions((str(path),)))["exit_code"] == 1
    info = json.loads((root / "meta/info.json").read_text())
    info["features"][derived.stem]["pipeline"] = {}
    (root / "meta/info.json").write_text(json.dumps(info))
    report = run_qc(QcOptions((str(path),)))
    assert not any(r["stage"] == "comparison" for r in report["channels"])


def test_flatline_adc_short_and_rms_outlier(tmp_path):
    rng = np.random.default_rng(2)
    t = np.arange(16000) / 2000
    x = rng.normal(0, 0.05, len(t))
    x[1000:3500] = 0.1
    x[12000:14000] *= 100
    path = write(tmp_path / "emg.parquet", t, dim_0=x)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"adc_min": -0.2, "adc_max": 0.2}))
    report = run_qc(QcOptions((str(path),), config=config))
    assert {"flatline", "clipping", "rms_outlier"} <= rules(report)
    short = write(tmp_path / "short.parquet", [0, 0.0005], dim_0=[1.0, 2.0])
    report = run_qc(QcOptions((str(short),)))
    assert report["channels"][0]["windows"] == 0
    assert report["status"] == "NOT_VERIFIED"


def test_cli_and_report_protection(tmp_path, capsys):
    path = tmp_path / "channels.csv"
    path.write_text("timestamp,a\n0,1\n0.0005,1\n")
    original = path.read_bytes()
    assert main(["qc", str(path), "--output-dir", str(tmp_path)]) == 2
    assert path.read_bytes() == original
    assert main(["qc", str(path), "--json"]) == 1
    data = json.loads(capsys.readouterr().out)
    assert data["errors"] == 1
    assert main(["qc", str(path)]) == 1
    assert "EMG QC" in capsys.readouterr().out
    assert channel_names({"timestamp": [], "a": [], "sample_index": []}) == ["a"]


def test_missing_registration_metadata_failure_and_sidecar(tmp_path):
    root, path, derived, t, x = setup_session(tmp_path)
    (root / "meta/info.json").write_text("{}")
    Path(str(derived) + ".report.json").write_text(json.dumps({"method": "offline_filtfilt"}))
    report = run_qc(QcOptions((str(path),)))
    assert "alignment" in rules(report)
    (root / "meta/info.json").write_text("invalid")
    assert run_qc(QcOptions((str(path),)))["exit_code"] == 2


def test_multiple_episodes_glob_and_alias_protection(tmp_path):
    root, path, derived, t, x = setup_session(tmp_path)
    second = root / "data/chunk-002/episode_002000/observation.left_wrist_emg.parquet"
    write(second, t, dim_0=x, episode_index=np.full(len(t), 2000))
    report = run_qc(QcOptions((str(root / "data/**/*.parquet"),), require_denoised=True))
    assert len(report["files"]) == 2
    assert report["exit_code"] == 1
    out = tmp_path / "report"
    out.mkdir()
    (out / "summary.json").symlink_to(root / "meta/info.json")
    before = (root / "meta/info.json").read_bytes()
    with pytest.raises(ValueError, match="distinct"):
        run_qc(QcOptions((str(root),), output_dir=out))
    assert (root / "meta/info.json").read_bytes() == before


def test_causal_tail_is_reported(tmp_path):
    root, path, derived, t, x = setup_session(tmp_path, method="causal_stream", samples=8011)
    write(derived, t[:-11], dim_0=x[:-11])
    write(Path(str(derived) + ".mask.parquet"), t[:-11], valid=np.arange(8000) >= 2000)
    report = run_qc(QcOptions((str(path),)))
    coverage = next(f for f in report["findings"] if f["rule"] == "causal_coverage")
    assert coverage["observed"]["omitted_samples"] == 11
    assert any(r["stage"] == "comparison" for r in report["channels"])
