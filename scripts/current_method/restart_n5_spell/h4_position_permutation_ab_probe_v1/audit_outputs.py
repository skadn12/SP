"""Offline audit of raw Solver outputs for the position A/B probe."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from ..core import atomic_json, atomic_text


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(run_root: Path) -> list[dict[str, Any]]:
    return [_load(path) for path in sorted((run_root / "results").glob("*.json"))]


def _perms(run_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    for line in (run_root / "permutation_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[(str(row["candidate_id"]), str(row["permutation_id"]))] = row
    return result


def audit(run_root: Path) -> dict[str, Any]:
    rows = _rows(run_root)
    perms = _perms(run_root)
    issues: Counter[str] = Counter()
    examples: dict[str, dict[str, Any]] = {}
    parsed_status = Counter()
    output_modes = Counter()
    qtypes = Counter()
    b_reserve_mismatch = 0
    seed_mismatch = 0
    by_condition: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    def issue(name: str, row: dict[str, Any], detail: Any = None) -> None:
        issues[name] += 1
        if name not in examples:
            examples[name] = {"request_id": row.get("request_id"), "raw_output": row.get("raw_output"), "detail": detail}

    for row in rows:
        qtype = str(row.get("question_type")); mode = str(row.get("output_mode"))
        output_modes[mode] += 1; qtypes[qtype] += 1
        raw = str(row.get("raw_output") or "")
        parsed = row.get("parse") or {}
        parsed_status[str(parsed.get("parse_status"))] += 1
        if raw.count("<box>") != 1 or raw.count("</box>") != 1 or not re.fullmatch(r"\s*<box>.*</box>\s*", raw, re.DOTALL):
            issue("box_contract_failure", row)
        valid_status = {"valid_boxed_letter", "success"}
        if str(parsed.get("parse_status")) not in valid_status:
            issue("parser_status_failure", row, parsed)
        if row.get("technical_status") is not None:
            issue("technical_status_present", row, row.get("technical_status"))
        if not row.get("scored"):
            issue("unscored", row, parsed)
        permutation = perms.get((str(row.get("candidate_id")), str(row.get("permutation_id"))))
        if permutation is None:
            issue("permutation_missing", row)
            continue
        mapping = {str(k): str(v) for k, v in (permutation.get("display_to_canonical_mapping") or {}).items()}
        if mode == "label":
            predicted = [str(x) for x in row.get("parsed_answer") or []]
            if any(value not in mapping for value in predicted):
                issue("label_prediction_outside_mapping", row, predicted)
            inverse = {canonical: label for label, canonical in mapping.items()}
            expected_gold = [inverse[value] for value in row.get("canonical_gold") or [] if value in inverse]
            if list(row.get("displayed_gold") or []) != expected_gold:
                issue("ordering_or_display_gold_metadata_mismatch", row, {"recorded": row.get("displayed_gold"), "expected": expected_gold})
        else:
            pieces = list(parsed.get("parsed_display_labels") or [])
            if len(pieces) != len(set(pieces)):
                issue("content_duplicate_item", row, pieces)
            if parsed.get("parsed_canonical") is None:
                issue("content_canonical_mapping_missing", row, parsed)
            inner = raw[raw.find("<box>") + 6:raw.rfind("</box>")]
            if any(piece.strip().startswith(tuple(f"{key}." for key in mapping)) for piece in inner.split(" || ")):
                issue("content_label_prefix", row, inner)
        by_condition[(str(row.get("candidate_id")), str(row.get("permutation_id")), str(row.get("context_mode")), mode)] = row

    # B1 and B2 must use the same dynamic output reserve for every pair.
    for (cid, pid, context, mode), row in list(by_condition.items()):
        if mode != "content_visible":
            continue
        other = by_condition.get((cid, pid, context, "content_hidden"))
        if other is not None and row.get("output_reserve") != other.get("output_reserve"):
            b_reserve_mismatch += 1
    # Seeds must be invariant to context, permutation and output mode.
    seed_groups: dict[tuple[str, int], set[int]] = {}
    for row in rows:
        key = (str(row.get("candidate_id")), int(row.get("slot_id") or 0))
        seed_groups.setdefault(key, set()).add(int(row.get("seed")))
    seed_mismatch = sum(len(values) != 1 for values in seed_groups.values())
    if b_reserve_mismatch:
        issues["b1_b2_output_reserve_mismatch"] = b_reserve_mismatch
    if seed_mismatch:
        issues["paired_seed_mismatch"] = seed_mismatch

    model_output_issue_names = {
        "box_contract_failure", "parser_status_failure", "technical_status_present", "unscored",
        "label_prediction_outside_mapping", "content_duplicate_item", "content_canonical_mapping_missing",
        "content_label_prefix", "b1_b2_output_reserve_mismatch", "paired_seed_mismatch",
    }
    model_output_issues = {name: count for name, count in issues.items() if name in model_output_issue_names}
    metadata_issues = {name: count for name, count in issues.items() if name not in model_output_issue_names}
    if model_output_issues:
        conclusion = "Raw output issues require review before interpreting the affected condition."
    elif metadata_issues:
        conclusion = "Raw Solver outputs satisfy the strict output contract; metadata or scoring issues remain for review."
    else:
        conclusion = "All raw Solver outputs satisfy the strict output contract and no metadata or scoring issues were found."
    return {
        "audit_version": "h4_position_ab_raw_output_audit_v1",
        "run_root": str(run_root.resolve()),
        "row_count": len(rows),
        "permutation_count": len(perms),
        "output_modes": dict(output_modes),
        "question_types": dict(qtypes),
        "parser_status": dict(parsed_status),
        "model_output_issues": model_output_issues,
        "metadata_or_scoring_issues": metadata_issues,
        "model_output_clean": not model_output_issues,
        "examples": examples,
        "conclusion": conclusion,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args(argv)
    report = audit(args.run_root.resolve())
    atomic_json(args.run_root / "output_audit.json", report)
    atomic_text(args.run_root / "output_audit.md", "# H4 position A/B raw output audit\n\n" + json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["model_output_clean"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
