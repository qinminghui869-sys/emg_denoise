"""Command-line boundary: parse options, invoke services, print a readable summary."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .dsp.filters import FS, Modality, modality_preset
from .io import jsonable
from .pipeline.batch import BatchOptions, run_batch
from .pipeline.meta import backfill_session_meta
from .pipeline.replay import ReplayOptions, default_process_output, run_replay
from .pipeline.retime import RetimeOptions, run_retime
from .qc import QcOptions, run_qc
from .qc.report import format_summary


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI without performing I/O or configuring application state."""
    parser = argparse.ArgumentParser(description="EMG 时基重建与去噪（文件默认离线 filtfilt）")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--json",
        action="store_true",
        help="向标准输出打印完整 JSON（默认打印可读摘要）",
    )
    common.add_argument(
        "--sidecar",
        action="store_true",
        help="额外写出 .timebase.json / .report.json；默认只更新 meta/info.json 与 batch.json",
    )
    common.add_argument(
        "--no-update-meta",
        action="store_true",
        help="不写入会话 meta/info.json",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    retime = sub.add_parser(
        "retime",
        parents=[common],
        help="阶段一：EMG 双锚点插值，未覆盖首尾保持原样",
    )
    retime.add_argument("input", type=Path, help="采集parquet或CSV；需含timestamp列")
    retime.add_argument("--output", type=Path, help="输出路径；--inplace 时省略")
    retime.add_argument(
        "--inplace",
        action="store_true",
        help="将源parquet改名为*.bak，重建结果写回原文件名",
    )
    retime.add_argument("--force", action="store_true", help="就地重建时允许已有*.bak存在")
    retime.add_argument(
        "--modality",
        choices=("emg",),
        default="emg",
        help="仅支持 EMG（2000 Hz；可用 --rate 覆盖）",
    )
    retime.add_argument("--rate", type=float, help="真实采样率Hz；默认 2000")
    retime.add_argument(
        "--method",
        choices=("anchor_interpolation", "uniform"),
        default=None,
        help="默认双锚点插值，首尾保留；uniform 为旧固定周期模式",
    )
    retime.add_argument("--tick-us", type=float, help="时钟分辨率us；默认由数据估计")
    retime.add_argument(
        "--allow-negative", action="store_true", help="仅 uniform 模式：不平移首点至零"
    )
    batch = sub.add_parser(
        "batch",
        parents=[common],
        help="批量两阶段处理；输出按输入相对路径镜像，避免同名覆盖",
    )
    batch.add_argument("inputs", nargs="+", help="文件路径或glob模式，如 '数据集/**/*_emg.parquet'")
    batch.add_argument("--output-dir", type=Path, help="镜像输出目录；--inplace 时可省略")
    batch.add_argument(
        "--inplace",
        action="store_true",
        help="就地重建：源parquet→*.bak，重建结果写回原文件名；可与 --denoise 同用",
    )
    batch.add_argument(
        "--modality",
        choices=("emg",),
        default="emg",
        help="仅支持 EMG",
    )
    batch.add_argument("--rate", type=float, help="真实采样率Hz；默认 2000")
    batch.add_argument("--tick-us", type=float, help="时钟分辨率us；默认逐文件估计")
    batch.add_argument(
        "--allow-negative", action="store_true", help="仅 uniform 模式：不平移首点至零"
    )
    batch.add_argument(
        "--denoise",
        action="store_true",
        help="重建后继续去噪；镜像模式写 denoised.parquet/csv，就地模式写 <stem>.denoised.parquet",
    )
    batch.add_argument("--force", action="store_true", help="覆盖已存在的输出或已有*.bak")
    batch.add_argument(
        "--causal",
        action="store_true",
        help="阶段二使用流式因果滤波（默认离线 filtfilt，减轻首尾振铃）",
    )
    meta = sub.add_parser(
        "meta-backfill",
        parents=[common],
        help="从已有 sidecar 回填 meta/info.json，并默认删除 sidecar",
    )
    meta.add_argument(
        "sessions",
        nargs="+",
        type=Path,
        help="会话根目录（含 meta/info.json），或直接传 episode 上一级会话路径",
    )
    meta.add_argument(
        "--keep-sidecars",
        action="store_true",
        help="回填后保留 .timebase.json / .report.json",
    )
    replay = sub.add_parser(
        "process",
        parents=[common],
        help="阶段二：离线 filtfilt 去噪（与 Viewer 同款）；离线 EMG 接受校正点均分后的时基，因果模式要求均匀时基",
    )
    replay.add_argument("input", type=Path, help="CSV 或 parquet（含 timestamp 与通道）")
    replay.add_argument(
        "--output",
        type=Path,
        help="输出路径；默认与输入同目录的 <stem>.denoised.parquet|.csv",
    )
    replay.add_argument(
        "--modality",
        choices=("emg",),
        default="emg",
        help="EMG：高通 20 Hz、低通 450 Hz、梳状 50 Hz",
    )
    replay.add_argument(
        "--rate",
        type=float,
        help="真实采样率Hz；默认读 meta/info.json fps，否则测时间戳，再否则 2000",
    )
    replay.add_argument(
        "--causal",
        action="store_true",
        help="改用流式因果滤波（默认可视化同款离线 filtfilt，减轻首尾振铃）",
    )
    qc = sub.add_parser("qc", help="只读 EMG 信号质量与去噪完整性检查")
    qc.add_argument("inputs", nargs="+", help="文件、episode/会话目录或 glob")
    qc.add_argument("--output-dir", type=Path)
    qc.add_argument("--rate", type=float)
    qc.add_argument("--window-s", type=float, default=2.0)
    qc.add_argument("--step-s", type=float, default=1.0)
    qc.add_argument("--config", type=Path)
    qc.add_argument("--require-denoised", action="store_true")
    qc.add_argument("--strict", action="store_true")
    qc.add_argument("--json", action="store_true")
    return parser


def _resolve_modality(flag: str) -> Modality:
    return modality_preset(flag).modality


def _default_rate() -> float:
    return float(FS)


def _short_path(path: str | Path, root: str | None = None) -> str:
    text = str(path)
    if root:
        prefix = str(root).rstrip("/") + "/"
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def _format_batch(payload: dict[str, Any]) -> str:
    root = payload.get("root")
    lines = [
        "batch summary",
        (
            f"  matched={payload['matched']}  succeeded={payload['succeeded']}  "
            f"failed={payload['failed']}  flagged={payload['flagged']}  "
            f"samples={payload['total_samples']}"
        ),
        (
            f"  modality={payload.get('modality', 'emg')}  "
            f"inplace={payload.get('inplace', False)}  "
            f"denoise={payload.get('denoised', False)}"
        ),
        f"  report={payload['report']}",
        "",
    ]
    for entry in payload.get("recordings", []):
        name = _short_path(entry["input"], root if isinstance(root, str) else None)
        rate = entry.get("measured_rate_hz")
        rate_text = f"{rate:.3f} Hz" if isinstance(rate, (int, float)) else "?"
        lines.append(f"OK   {name}")
        lines.append(
            f"     samples={entry.get('samples')}  channels={entry.get('channels')}  "
            f"rate≈{rate_text}  resync={entry.get('resync_segments')}"
        )
        if "denoised" in entry:
            lines.append(
                f"     valid={entry.get('valid_samples')}/{entry.get('samples')}  "
                f"segments={entry.get('segments')}  "
                f"denoised={_short_path(entry['denoised'], root if isinstance(root, str) else None)}"
            )
        elif entry.get("retimed"):
            lines.append(
                f"     retimed={_short_path(entry['retimed'], root if isinstance(root, str) else None)}"
            )
        warnings = entry.get("warnings") or []
        for warning in warnings:
            lines.append(f"     warning: {warning}")
        lines.append("")
    for failure in payload.get("failures", []):
        name = _short_path(failure["input"], root if isinstance(root, str) else None)
        lines.append(f"FAIL {name}")
        lines.append(f"     {failure['error']}")
        lines.append("")
    if payload.get("note"):
        lines.append(f"note: {payload['note']}")
    return "\n".join(lines).rstrip() + "\n"


def _format_retime(payload: dict[str, Any]) -> str:
    lines = [
        "retime summary",
        f"  output={payload.get('output')}",
        f"  report={payload.get('report')}",
        (
            f"  samples={payload.get('samples')}  channels={payload.get('channels')}  "
            f"modality={payload.get('modality', '?')}"
        ),
        (
            f"  rate={payload.get('rate_hz')} Hz  measured≈{payload.get('measured_rate_hz'):.3f} Hz  "
            f"tick={payload.get('tick_us'):.3f} us"
            if isinstance(payload.get("measured_rate_hz"), (int, float))
            and isinstance(payload.get("tick_us"), (int, float))
            else f"  rate={payload.get('rate_hz')} Hz"
        ),
        (
            f"  resync_segments={payload.get('resync_segments')}  "
            f"max_offset={payload.get('max_offset_us'):.3f} us"
            if isinstance(payload.get("max_offset_us"), (int, float))
            else f"  resync_segments={payload.get('resync_segments')}"
        ),
    ]
    if payload.get("backup"):
        lines.append(f"  backup={payload['backup']}")
    for warning in payload.get("warnings") or []:
        lines.append(f"  warning: {warning}")
    return "\n".join(lines) + "\n"


def _format_process(payload: dict[str, Any]) -> str:
    lines = [
        "process summary",
        f"  output={payload.get('output')}",
        f"  report={payload.get('report')}",
        (
            f"  modality={payload.get('modality')}  rate={payload.get('sample_rate_hz')} Hz  "
            f"samples={payload.get('input_samples')}  "
            f"valid={payload.get('valid_samples')}/{payload.get('output_samples')}  "
            f"segments={payload.get('segments')}"
        ),
    ]
    if payload.get("unemitted_samples"):
        lines.append(f"  unemitted={payload['unemitted_samples']}")
    return "\n".join(lines) + "\n"


def _format_meta_backfill(payload: dict[str, Any]) -> str:
    lines = [f"meta-backfill summary  sessions={payload.get('count', 0)}", ""]
    for entry in payload.get("sessions", []):
        lines.append(f"OK   {entry.get('session')}")
        lines.append(f"     info={entry.get('info')}")
        lines.append(f"     updated={len(entry.get('updated') or [])} features")
        removed = entry.get("removed_sidecars") or []
        lines.append(f"     removed_sidecars={len(removed)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def format_payload(command: str, payload: dict[str, Any]) -> str:
    """Render a human-readable summary for stdout."""
    if command == "batch":
        return _format_batch(payload)
    if command == "qc":
        return format_summary(payload)
    if command == "retime":
        return _format_retime(payload)
    if command == "meta-backfill":
        return _format_meta_backfill(payload)
    return _format_process(payload)


def _execute(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if args.command == "qc":
        payload = run_qc(
            QcOptions(
                inputs=tuple(args.inputs),
                output_dir=args.output_dir,
                rate=args.rate,
                window_s=args.window_s,
                step_s=args.step_s,
                config=args.config,
                require_denoised=args.require_denoised,
                strict=args.strict,
            )
        )
        return payload, int(payload["exit_code"])
    if args.command == "retime":
        if not args.inplace and args.output is None:
            raise ValueError("--output is required unless --inplace is set")
        modality = _resolve_modality(args.modality)
        rate = _default_rate() if args.rate is None else args.rate
        payload = run_retime(
            RetimeOptions(
                source=args.input,
                method=args.method or "anchor_interpolation",
                output=args.output,
                rate_hz=rate,
                tick_us=args.tick_us,
                non_negative=not args.allow_negative,
                inplace=args.inplace,
                force=args.force,
                sidecar=args.sidecar,
                update_meta=not args.no_update_meta,
            )
        )
        payload["modality"] = modality
        # Warnings mean the rebuilt axis is suspect; fail software acceptance.
        return payload, 1 if payload["warnings"] else 0
    if args.command == "batch":
        if not args.inplace and args.output_dir is None:
            raise ValueError("--output-dir is required unless --inplace is set")
        batch_modality = _resolve_modality(args.modality)
        payload = run_batch(
            BatchOptions(
                patterns=tuple(args.inputs),
                output_dir=args.output_dir,
                rate_hz=args.rate,
                tick_us=args.tick_us,
                non_negative=not args.allow_negative,
                denoise=args.denoise,
                modality=batch_modality,
                force=args.force,
                inplace=args.inplace,
                sidecar=args.sidecar,
                update_meta=not args.no_update_meta,
                causal=args.causal,
            )
        )
        # Any failed or flagged recording fails acceptance for the whole batch.
        return payload, 1 if payload["failed"] or payload["flagged"] else 0
    if args.command == "meta-backfill":
        results = [
            backfill_session_meta(session, remove_sidecars=not args.keep_sidecars)
            for session in args.sessions
        ]
        return {"sessions": results, "count": len(results)}, 0
    process_modality = _resolve_modality(args.modality)
    output = args.output if args.output is not None else default_process_output(args.input)
    return run_replay(
        ReplayOptions(
            source=args.input,
            output=output,
            modality=process_modality,
            sample_rate_hz=args.rate,
            sidecar=args.sidecar,
            update_meta=not args.no_update_meta,
            causal=args.causal,
        )
    ), 0


def main(argv: Sequence[str] | None = None) -> int:
    """Exit 0 on success, 1 on failed software acceptance, 2 on invalid input/I/O."""
    args = build_parser().parse_args(argv)
    try:
        payload, exit_code = _execute(args)
        if args.json:
            print(json.dumps(jsonable(payload), ensure_ascii=False, allow_nan=False))
        else:
            print(format_payload(args.command, payload), end="")
        return exit_code
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
