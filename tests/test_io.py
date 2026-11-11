import json
from pathlib import Path

import numpy as np
import pytest

from emg_denoise.cli import main
from emg_denoise.io import (
    atomic_text_writer,
    dump_json,
    jsonable,
    load_json_object,
    load_recording,
    validate_output_paths,
)


@pytest.mark.parametrize(
    "content, message",
    [
        ("", "empty"),
        ("time,emg\n0,1\n", "header"),
        ("timestamp,emg,emg\n0,1,2\n", "unique"),
        ("timestamp,valid\n0,1\n", "reserved"),
        ("timestamp,emg\n", "no samples"),
        ("timestamp,emg\n0,1,2\n", "columns"),
        ("timestamp,emg\n0,text\n", "numeric"),
        ("timestamp,emg\n0,nan\n", "finite"),
        ("timestamp,emg\n0,1\n0.001,2\n", "contiguous"),
    ],
)
def test_recording_rejects_malformed_data(tmp_path, content, message):
    source = tmp_path / "input.csv"
    source.write_text(content)
    with pytest.raises(ValueError, match=message):
        load_recording(source)


def test_atomic_replay_failure_preserves_destination_and_removes_temporary_files(tmp_path):
    source, output = tmp_path / "input.csv", tmp_path / "output.csv"
    source.write_text("timestamp,emg\n0,1\n0.0005,not-a-number\n")
    output.write_text("previous successful recording\n")
    assert main(["process", str(source), "--output", str(output)]) == 2
    assert output.read_text() == "previous successful recording\n"
    assert not list(tmp_path.glob(".*.tmp"))
    assert not Path(f"{output}.report.json").exists()


def test_path_guards_cover_symlink_and_hardlink_aliases(tmp_path):
    source = tmp_path / "input.csv"
    source.write_text("timestamp,emg\n0,1\n")
    symlink, hardlink = tmp_path / "symlink.csv", tmp_path / "hardlink.csv"
    symlink.symlink_to(source)
    hardlink.hardlink_to(source)
    for alias in [source, symlink, hardlink]:
        with pytest.raises(ValueError, match="distinct"):
            validate_output_paths([source], [alias])
    output = tmp_path / "out.csv"
    report = Path(f"{output}.report.json")
    with pytest.raises(ValueError, match="distinct"):
        validate_output_paths([source], [output, source])
    with pytest.raises(ValueError, match="distinct"):
        validate_output_paths([source], [output, report, report])


def test_strict_json_serialization_and_atomic_rollback(tmp_path):
    path = tmp_path / "report.json"
    dump_json(
        path, {"matrix": np.array([np.nan, np.inf, 1]), "yes": np.bool_(True), "n": np.int64(2)}
    )
    assert json.loads(path.read_text()) == {"matrix": [None, None, 1.0], "yes": True, "n": 2}
    before = path.read_bytes()
    with pytest.raises(RuntimeError), atomic_text_writer(path) as writer:
        writer.write("incomplete")
        raise RuntimeError("simulated writer failure")
    assert path.read_bytes() == before
    with pytest.raises(TypeError, match="keys"):
        jsonable({1: "not a string key"})
    with pytest.raises(TypeError, match="unsupported"):
        jsonable(object())


def test_json_input_shape_is_validated(tmp_path):
    source = tmp_path / "report.json"
    source.write_text("[]")
    with pytest.raises(ValueError, match="object"):
        load_json_object(source)
