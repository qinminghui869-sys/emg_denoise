import json
from pathlib import Path

import pytest

from emg_denoise.cli import main
from emg_denoise.pipeline.batch import (
    BatchOptions,
    destination,
    mirror_root,
    resolve_inputs,
    run_batch,
)
from tests.test_timebase import write_acquisition_parquet


def build_session_tree(root, sessions=("L3-002", "L3-003"), sides=("left", "right"), samples=600):
    """Mimic the acquisition layout: one file name repeated across sessions."""
    created = []
    for session in sessions:
        episode = root / session / "data" / "chunk-000" / "episode_000000"
        episode.mkdir(parents=True, exist_ok=True)
        for side in sides:
            path = episode / f"observation.{side}_wrist_emg.parquet"
            write_acquisition_parquet(path, samples=samples, channels=2)
            created.append(path)
    return created


def test_outputs_do_not_collide_for_identically_named_inputs(tmp_path):
    """The reason batch mirrors paths: four inputs share two file names."""
    sources = build_session_tree(tmp_path / "raw")
    assert len({path.name for path in sources}) == 2

    payload = run_batch(
        BatchOptions(patterns=(str(tmp_path / "raw/**/*.parquet"),), output_dir=tmp_path / "out")
    )

    assert payload["succeeded"] == 4
    assert payload["failed"] == 0
    retimed = {entry["retimed"] for entry in payload["recordings"]}
    assert len(retimed) == 4
    assert len(list((tmp_path / "out").rglob("retimed.parquet"))) == 4


def test_mirrored_output_keeps_the_session_directory(tmp_path):
    build_session_tree(tmp_path / "raw", sessions=("L3-002", "L3-003"), sides=("left",))
    run_batch(
        BatchOptions(patterns=(str(tmp_path / "raw/**/*.parquet"),), output_dir=tmp_path / "out")
    )
    produced = sorted(
        str(p.relative_to(tmp_path / "out")) for p in (tmp_path / "out").rglob("retimed.parquet")
    )
    assert any("L3-002" in path for path in produced)
    assert any("L3-003" in path for path in produced)


def test_batch_runs_both_stages_when_denoising(tmp_path):
    build_session_tree(tmp_path / "raw", sessions=("L3-002",), sides=("left",), samples=4000)
    payload = run_batch(
        BatchOptions(
            patterns=(str(tmp_path / "raw/**/*.parquet"),),
            output_dir=tmp_path / "out",
            denoise=True,
        )
    )
    entry = payload["recordings"][0]
    # A rebuilt axis must not look discontinuous to the streaming engine.
    assert entry["segments"] == 1
    assert entry["warnings"] == []
    assert list((tmp_path / "out").rglob("denoised.parquet"))


def test_existing_denoised_output_is_refused_without_force(tmp_path):
    build_session_tree(tmp_path / "raw", sessions=("L3-002",), sides=("left",), samples=4000)
    options = BatchOptions(
        patterns=(str(tmp_path / "raw/**/*.parquet"),),
        output_dir=tmp_path / "out",
        denoise=True,
    )
    assert run_batch(options)["succeeded"] == 1
    # retimed.parquet is refused first, so force it through to reach stage two's guard.
    forced = BatchOptions(
        patterns=options.patterns, output_dir=options.output_dir, denoise=True, force=False
    )
    assert run_batch(forced)["failed"] == 1


def test_a_split_recording_is_flagged(tmp_path, monkeypatch):
    """Stage two reporting more than one segment means the axis was not uniform."""
    import emg_denoise.pipeline.batch as batch_module

    build_session_tree(tmp_path / "raw", sessions=("L3-002",), sides=("left",), samples=4000)

    def fake_replay(options):
        return {"segments": 7, "valid_samples": 0, "unemitted_samples": 0}

    monkeypatch.setattr(batch_module, "run_replay", fake_replay)
    payload = run_batch(
        BatchOptions(
            patterns=(str(tmp_path / "raw/**/*.parquet"),),
            output_dir=tmp_path / "out",
            denoise=True,
        )
    )
    assert payload["flagged"] == 1
    assert "7 segments" in payload["recordings"][0]["warnings"][0]


def test_batch_writes_a_report(tmp_path):
    build_session_tree(tmp_path / "raw", sessions=("L3-002",), sides=("left", "right"))
    payload = run_batch(
        BatchOptions(patterns=(str(tmp_path / "raw/**/*.parquet"),), output_dir=tmp_path / "out")
    )
    report = json.loads((tmp_path / "out" / "batch.json").read_text())
    assert report["matched"] == 2
    assert report["total_samples"] == payload["total_samples"]


def test_one_bad_recording_does_not_abort_the_batch(tmp_path):
    build_session_tree(tmp_path / "raw", sessions=("L3-002",), sides=("left",))
    broken = (
        tmp_path / "raw" / "L3-002" / "data" / "chunk-000" / "episode_000000" / "broken.parquet"
    )
    broken.write_bytes(b"not a parquet file")

    payload = run_batch(
        BatchOptions(patterns=(str(tmp_path / "raw/**/*.parquet"),), output_dir=tmp_path / "out")
    )
    assert payload["succeeded"] == 1
    assert payload["failed"] == 1
    assert payload["failures"][0]["input"].endswith("broken.parquet")


def test_existing_outputs_are_refused_without_force(tmp_path):
    build_session_tree(tmp_path / "raw", sessions=("L3-002",), sides=("left",))
    options = BatchOptions(
        patterns=(str(tmp_path / "raw/**/*.parquet"),), output_dir=tmp_path / "out"
    )
    assert run_batch(options)["succeeded"] == 1
    assert run_batch(options)["failed"] == 1


def test_force_overwrites_existing_outputs(tmp_path):
    build_session_tree(tmp_path / "raw", sessions=("L3-002",), sides=("left",))
    pattern = (str(tmp_path / "raw/**/*.parquet"),)
    run_batch(BatchOptions(patterns=pattern, output_dir=tmp_path / "out"))
    again = run_batch(BatchOptions(patterns=pattern, output_dir=tmp_path / "out", force=True))
    assert again["succeeded"] == 1


def test_reruns_do_not_consume_their_own_outputs(tmp_path):
    """Derived reports must never be mistaken for inputs."""
    build_session_tree(tmp_path / "raw", sessions=("L3-002",), sides=("left",))
    run_batch(
        BatchOptions(patterns=(str(tmp_path / "raw/**/*.parquet"),), output_dir=tmp_path / "raw")
    )
    inputs = resolve_inputs((str(tmp_path / "raw/**/*"),))
    assert not any(str(path).endswith(".timebase.json") for path in inputs)


def test_a_plain_path_works_without_globbing(tmp_path):
    sources = build_session_tree(tmp_path / "raw", sessions=("L3-002",), sides=("left",))
    assert resolve_inputs((str(sources[0]),)) == [sources[0]]


def test_unmatched_patterns_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="no input files matched"):
        resolve_inputs((str(tmp_path / "absent/*.parquet"),))


def test_mirror_root_of_a_single_directory_is_that_directory(tmp_path):
    sources = build_session_tree(tmp_path / "raw", sessions=("L3-002",), sides=("left", "right"))
    assert mirror_root(sources) == sources[0].resolve().parent


def test_mirror_root_climbs_to_the_common_parent_across_sessions(tmp_path):
    sources = build_session_tree(tmp_path / "raw", sessions=("L3-002", "L3-003"), sides=("left",))
    assert mirror_root(sources) == (tmp_path / "raw").resolve()


def test_mirror_root_handles_inputs_from_unrelated_trees(tmp_path):
    left = build_session_tree(tmp_path / "a", sessions=("L3-002",), sides=("left",))
    right = build_session_tree(tmp_path / "b", sessions=("L3-003",), sides=("left",))
    root = mirror_root([*left, *right])
    assert root == tmp_path.resolve()


def test_unrelated_trees_still_produce_distinct_outputs(tmp_path):
    build_session_tree(tmp_path / "a", sessions=("L3-002",), sides=("left",))
    build_session_tree(tmp_path / "b", sessions=("L3-002",), sides=("left",))
    payload = run_batch(
        BatchOptions(
            patterns=(str(tmp_path / "a/**/*.parquet"), str(tmp_path / "b/**/*.parquet")),
            output_dir=tmp_path / "out",
        )
    )
    assert payload["succeeded"] == 2
    assert len({entry["retimed"] for entry in payload["recordings"]}) == 2


def test_destination_keeps_the_input_stem(tmp_path):
    sources = build_session_tree(tmp_path / "raw", sessions=("L3-002",), sides=("left",))
    root = mirror_root(sources)
    target = destination(sources[0], root, tmp_path / "out")
    assert target.name == "observation.left_wrist_emg"


def test_cli_batch_reports_success(tmp_path, capsys):
    build_session_tree(tmp_path / "raw", sessions=("L3-002",), sides=("left", "right"))
    exit_code = main(
        [
            "batch",
            "--json",
            str(tmp_path / "raw/**/*.parquet"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["succeeded"] == 2


def test_cli_batch_prints_a_readable_summary_by_default(tmp_path, capsys):
    build_session_tree(tmp_path / "raw", sessions=("L3-002",), sides=("left",))
    exit_code = main(
        ["batch", str(tmp_path / "raw/**/*.parquet"), "--output-dir", str(tmp_path / "out")]
    )
    text = capsys.readouterr().out
    assert exit_code == 0
    assert "batch summary" in text
    assert "succeeded=1" in text
    assert "OK" in text
    assert "observation.left_wrist_emg.parquet" in text


def test_cli_batch_fails_acceptance_when_a_recording_is_flagged(tmp_path, capsys):
    build_session_tree(tmp_path / "raw", sessions=("L3-002",), sides=("left",))
    exit_code = main(
        [
            "batch",
            "--json",
            str(tmp_path / "raw/**/*.parquet"),
            "--output-dir",
            str(tmp_path / "out"),
            "--rate",
            "500",
        ]
    )
    assert exit_code == 1
    assert json.loads(capsys.readouterr().out)["flagged"] == 1


def test_cli_batch_rejects_unmatched_inputs(tmp_path):
    assert (
        main(["batch", str(tmp_path / "none/*.parquet"), "--output-dir", str(tmp_path / "o")]) == 2
    )


def test_inplace_batch_rewrites_matched_parquets(tmp_path):
    sources = build_session_tree(tmp_path / "raw", sessions=("L3-002",), sides=("left", "right"))
    originals = {path: path.read_bytes() for path in sources}
    payload = run_batch(BatchOptions(patterns=(str(tmp_path / "raw/**/*.parquet"),), inplace=True))
    assert payload["succeeded"] == 2
    assert payload["inplace"] is True
    for path in sources:
        backup = Path(f"{path}.bak")
        assert backup.read_bytes() == originals[path]
        assert path.is_file()
        assert path.stat().st_size > 0
    assert (sources[0].parent / "batch.json").is_file()


def test_inplace_batch_with_denoise_writes_sibling_parquet(tmp_path):
    build_session_tree(tmp_path / "raw", sessions=("L3-002",), sides=("left",), samples=4000)
    sources = sorted((tmp_path / "raw").rglob("*.parquet"))
    payload = run_batch(
        BatchOptions(
            patterns=(str(tmp_path / "raw/**/*.parquet"),),
            inplace=True,
            denoise=True,
            sidecar=True,
        )
    )
    assert payload["succeeded"] == 1
    denoised = sources[0].with_name(f"{sources[0].stem}.denoised.parquet")
    assert Path(payload["recordings"][0]["denoised"]) == denoised
    assert denoised.is_file()
    assert Path(f"{denoised}.report.json").is_file()


def test_cli_inplace_batch(tmp_path, capsys):
    build_session_tree(tmp_path / "raw", sessions=("L3-002",), sides=("left", "right"))
    exit_code = main(["batch", "--json", str(tmp_path / "raw/**/*.parquet"), "--inplace"])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["succeeded"] == 2
    assert payload["inplace"] is True
