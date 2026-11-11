#!/usr/bin/env python3
"""Export one episode's wrist EMG parquet into compact JSON for viewer.html.

Requires the parquet extra: ``python -m pip install -e '.[parquet]'``.
Pass the session directory with ``--src``. The JSON records the session name, not a local path.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np


def _enc_f32(arr: np.ndarray) -> str:
    raw = np.ascontiguousarray(arr, dtype=np.float32).tobytes()
    return base64.b64encode(raw).decode("ascii")


def _load_side(episode_dir: Path, name: str, decim: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("reading parquet requires emg-denoise[parquet]") from error

    columns = ["timestamp", *[f"dim_{i}" for i in range(8)]]
    table = pq.read_table(episode_dir / name, columns=columns)
    timestamps = np.asarray(table.column("timestamp").to_numpy(), dtype=np.float64)
    channels = np.column_stack(
        [np.asarray(table.column(f"dim_{i}").to_numpy(), dtype=np.float32) for i in range(8)]
    )
    return timestamps[::decim], channels[::decim]


def export(src: Path, out: Path, *, original_fs: float, target_fs: float) -> Path:
    episode_dir = src / "data" / "chunk-000" / "episode_000000"
    if not episode_dir.is_dir():
        raise FileNotFoundError(f"episode dir not found: {episode_dir}")

    decim = max(1, int(round(original_fs / target_fs)))
    effective_fs = original_fs / decim

    left_t, left_x = _load_side(episode_dir, "observation.left_wrist_emg.parquet", decim)
    right_t, right_x = _load_side(episode_dir, "observation.right_wrist_emg.parquet", decim)
    count = min(len(left_t), len(right_t))
    left_t, left_x = left_t[:count], left_x[:count]
    right_t, right_x = right_t[:count], right_x[:count]
    origin = float(min(left_t[0], right_t[0]))
    relative = (left_t - origin).astype(np.float32)

    def side(prefix: str, values: np.ndarray) -> dict[str, object]:
        return {
            "names": [f"{prefix}{i}" for i in range(8)],
            "channels": [_enc_f32(values[:, i]) for i in range(8)],
            "mean": values.mean(axis=0).astype(float).tolist(),
            "std": values.std(axis=0).astype(float).tolist(),
        }

    payload = {
        "source": f"{src.name} / episode_000000",
        "source_path": src.name,
        "original_fs_hz": float(original_fs),
        "fs_hz": float(effective_fs),
        "n_samples": int(count),
        "duration_s": float(relative[-1] - relative[0]),
        "unit": "ADC counts",
        "dtype": "float32",
        "encoding": "base64",
        "t0_s": origin,
        "time": _enc_f32(relative),
        "left": side("L", left_x),
        "right": side("R", right_x),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="LeRobot session directory")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "sample_emg.json",
    )
    parser.add_argument("--original-fs", type=float, default=2000.0)
    parser.add_argument(
        "--target-fs",
        type=float,
        default=1000.0,
        help="Decimated rate for the viewer (1000 Hz keeps 450 Hz lowpass usable)",
    )
    args = parser.parse_args()
    path = export(args.src, args.out, original_fs=args.original_fs, target_fs=args.target_fs)
    print(f"wrote {path} ({path.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
