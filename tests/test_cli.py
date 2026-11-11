import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from emg_denoise.cli import main


def write_csv(path, t, x, header="timestamp,emg1"):
    np.savetxt(
        path,
        np.column_stack((t, x)),
        delimiter=",",
        header=header,
        comments="",
        fmt="%.17g",
    )


def test_csv_stream_preserves_epoch_time_partial_tail_gap_and_reports(tmp_path):
    t = 1_700_000_000 + np.arange(5003) / 2000
    t[2501:] += 1
    x = np.sin(np.arange(len(t)))
    source, output = tmp_path / "input.csv", tmp_path / "out.csv"
    write_csv(source, t, x)
    assert main(["process", "--sidecar", "--json", str(source), "--output", str(output)]) == 0
    data = np.loadtxt(output, delimiter=",", skiprows=1)
    np.testing.assert_array_equal(data[:, 0], t)
    assert data.shape == (5003, 2)
    report = json.loads(Path(str(output) + ".report.json").read_text())
    assert report["input_samples"] == report["output_samples"] == 5003
    assert report["segments"] == 2
    assert report["modality"] == "emg"
    assert report["observed_end_to_end_ms"] is None


def test_eeg_filename_is_rejected(tmp_path):
    source, output = tmp_path / "observation.eeg.csv", tmp_path / "out.csv"
    write_csv(source, np.arange(100) / 2000, np.zeros(100))
    assert main(["process", str(source), "--output", str(output)]) == 2
    assert not output.exists()


def test_csv_does_not_overwrite_input(tmp_path):
    t = np.arange(6000) / 2000
    source = tmp_path / "rest.csv"
    write_csv(source, t, np.random.default_rng(20).normal(size=6000))
    original = source.read_bytes()
    assert main(["process", str(source), "--output", str(source)]) == 2
    assert source.read_bytes() == original


def test_process_parquet_defaults_to_sibling_denoised_output(tmp_path):
    pytest.importorskip("pyarrow")
    from tests.test_timebase import write_acquisition_parquet

    source = tmp_path / "observation.left_wrist_emg.parquet"
    # Uniform stamps so stage two accepts the recording without a prior retime.
    write_acquisition_parquet(source, samples=4000, channels=2)
    # Overwrite with a contiguous axis (acquisition helper writes quantized stamps).
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(source)
    stamps = pa.array(np.arange(4000, dtype=np.float64) / 2000.0)
    columns = {name: table[name] for name in table.column_names if name != "timestamp"}
    pq.write_table(pa.table({"timestamp": stamps, **columns}), source)

    assert main(["process", "--sidecar", "--json", str(source)]) == 0
    output = tmp_path / "observation.left_wrist_emg.denoised.parquet"
    assert output.is_file()
    report = json.loads(Path(f"{output}.report.json").read_text())
    assert report["output"] == str(output)
    assert report["input_samples"] == report["output_samples"] == 4000
    rebuilt = pq.read_table(output)
    assert "valid" not in rebuilt.column_names
    assert "segment_id" not in rebuilt.column_names
    assert "dim_0" in rebuilt.column_names
    assert "episode_index" in rebuilt.column_names
    mask = pq.read_table(Path(f"{output}.mask.parquet"))
    assert set(mask.column_names) >= {"timestamp", "sample_index", "valid", "segment_id"}


def test_python_module_entrypoint_uses_the_same_cli(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "emg-denoise",
            "process",
            str(tmp_path / "missing.csv"),
            "--output",
            str(tmp_path / "out.csv"),
        ],
    )
    with pytest.raises(SystemExit) as result:
        runpy.run_module("emg_denoise", run_name="__main__")
    assert result.value.code == 2
