"""Read actual W4A4 installation receipts from archived server logs (CPU only)."""
from __future__ import annotations

import json
from pathlib import Path

from publication_guard import require


def validate_activation_log(path: Path, protocol: dict, *, quantized: bool) -> dict | None:
    """Validate one task's server, after the caller has checked its byte identity.

    A requested environment switch cannot establish which modules were wrapped.
    Each quantized task must contain its own complete installer report; another
    task's report cannot substitute for a missing one.
    """
    marker = "[fp4vla] activation-only "
    reports = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if marker not in line:
            continue
        try:
            report = json.loads(line.split(marker, 1)[1].strip())
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid activation installation JSON: {path}") from exc
        require(isinstance(report, dict), f"Invalid activation installation object: {path}")
        reports.append(report)
    if not quantized:
        require(not reports, f"BF16 server unexpectedly installed activation QDQ: {path}")
        return None
    require(len(reports) == 1, f"Expected one activation installation report per server: {path}")
    report = reports[0]
    scope = protocol.get("quantization_scope", {})
    ordinary = scope.get("ordinary_linear_operators")
    category = scope.get("category_linear_layers")
    total = scope.get("activation_linear_count")
    require(all(type(n) is int and n > 0 for n in (ordinary, category, total)) and
            ordinary + category == total and scope.get("category_w4a4_required") is True,
            "Protocol lacks the complete W4A4 activation scope")
    expected = {"ordinary_total": ordinary, "ordinary_w4a4": ordinary, "ordinary_bf16": 0,
                "category_total": category, "category_w4a4": category, "category_bf16": 0}
    for key, value in expected.items():
        require(type(report.get(key)) is int and report[key] == value,
                f"Actual activation coverage differs from protocol: {path}/{key}")
    require(report.get("mode") == "all" and report.get("ste") is False,
            f"Evaluation activation installation must use all/STE-off: {path}")
    records = report.get("records")
    require(isinstance(records, list) and len(records) == total and
            all(isinstance(row, dict) and isinstance(row.get("name"), str) and row["name"] and
                row.get("format") == "W4A4" for row in records),
            f"Activation module records do not substantiate W4A4 coverage: {path}")
    require(len({row["name"] for row in records}) == total,
            f"Duplicate activation module records: {path}")
    return {**expected, "mode": "all", "ste": False, "module_records": total}
