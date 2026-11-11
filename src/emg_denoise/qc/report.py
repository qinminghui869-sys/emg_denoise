"""Strict JSON and flat CSV output, with source-path protection."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ..io import atomic_text_writer, dump_json, jsonable, validate_output_paths


def write_reports(directory: Path, payload: dict[str, Any], inputs: list[Path]) -> None:
    outputs = [directory / n for n in ("summary.json", "channels.csv", "findings.csv")]
    validate_output_paths(inputs, outputs)
    dump_json(outputs[0], payload)
    for output, key in zip(outputs[1:], ("channels", "findings"), strict=True):
        rows = jsonable(payload[key])
        fields = list(dict.fromkeys(k for row in rows for k in row)) or [
            "file",
            "stage",
            "channel",
            "status",
        ]
        with atomic_text_writer(output) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                        for k, v in row.items()
                    }
                )


def format_summary(payload: dict[str, Any]) -> str:
    lines = [
        f"EMG QC: {payload['status']} | files={len(payload['files'])} errors={payload['errors']} warnings={payload['warnings']}"
    ]
    failures = [f for f in payload["findings"] if f["severity"] in ("ERROR", "WARN")]
    for finding in failures[:30]:
        lines.append(
            f"  {finding['severity']} {finding['file']} [{finding['stage']}/{finding.get('channel') or '-'}] {finding['rule']}: {finding['message']}"
        )
    if len(failures) > 30:
        lines.append(f"  ... {len(failures) - 30} more findings (use --json or --output-dir)")
    lines.append("工程门限用于人工复核；未执行/未验证项见完整报告。")
    return "\n".join(lines) + "\n"
