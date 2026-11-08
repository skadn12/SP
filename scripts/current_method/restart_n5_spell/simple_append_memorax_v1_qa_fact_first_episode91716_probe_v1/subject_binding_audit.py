"""Offline audit for the completed subject-binding live probe."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from ..core import atomic_json, atomic_text
from .inputs import DEFAULT_EXPANDED, SUBJECT_BINDING_TARGET_RUN_ROOT


RUN_ROOT = SUBJECT_BINDING_TARGET_RUN_ROOT
MODULE_ROOT = Path(__file__).resolve().parent


def run_audit() -> dict[str, Any]:
    expanded = json.loads(DEFAULT_EXPANDED.read_text(encoding="utf-8"))
    turns = {
        position: turn
        for position, turn in enumerate(expanded.get("turns") or [], 1)
    }
    window_reports = json.loads((RUN_ROOT / "qa/window_reports.json").read_text(encoding="utf-8"))
    reason_counts: Counter[str] = Counter()
    window_rows: list[dict[str, Any]] = []
    valid_facts: list[dict[str, Any]] = []
    for path in sorted((RUN_ROOT / "qa/windows").glob("*/validation.json")):
        validation = json.loads(path.read_text(encoding="utf-8"))
        reasons = Counter(
            error
            for item in validation.get("items") or []
            for error in item.get("mechanical_errors") or []
        )
        reason_counts.update(reasons)
        valid_facts.extend(validation.get("valid_facts") or [])
        window_rows.append({
            "window_id": validation.get("window_id"),
            "raw": validation.get("raw_fact_count", 0),
            "valid": validation.get("mechanically_valid_count", 0),
            "quarantine": validation.get("quarantine_count", 0),
            "reasons": dict(reasons),
        })
    binding_issues: list[dict[str, Any]] = []
    for fact in valid_facts:
        positions = fact.get("source_positions") or []
        mode = fact.get("binding_mode")
        subject = fact.get("subject")
        if mode == "speaker_at_position" and positions:
            speaker = str(turns[positions[0]].get("speaker") or "")
            if speaker != subject:
                binding_issues.append({
                    "fact_id": fact.get("fact_id"),
                    "issue": "resolved_speaker_subject_mismatch",
                    "speaker": speaker,
                    "subject": subject,
                })
        if mode == "known_entity" and not subject:
            binding_issues.append({
                "fact_id": fact.get("fact_id"),
                "issue": "known_entity_subject_empty",
            })
    report = {
        "run_root": str(RUN_ROOT),
        "api_calls_in_audit": 0,
        "window_count": len(window_rows),
        "raw_fact_count": sum(row["raw"] for row in window_rows),
        "mechanically_valid_fact_count": sum(row["valid"] for row in window_rows),
        "mechanical_quarantine_count": sum(row["quarantine"] for row in window_rows),
        "quarantine_reason_counts": dict(reason_counts),
        "valid_fact_binding_issue_count": len(binding_issues),
        "valid_fact_binding_issues": binding_issues,
        "valid_facts_subjects": Counter(fact.get("subject") for fact in valid_facts),
        "windows": window_rows,
        "capacity_is_dominant_failure": reason_counts.get("fact_capacity_exceeded", 0) > reason_counts.get("named_entity_requires_known_entity_binding", 0),
        "subject_binding_mechanical_gate": "pass" if not binding_issues else "fail",
        "semantic_audit_required": True,
        "reverse_writer_ready": False,
        "historical_v1_run_modified": False,
        "solver_calls": 0,
        "expansion_calls": 0,
        "memorax_calls": 0,
    }
    # Counter is not JSON serializable; retain deterministic plain counts.
    report["valid_facts_subjects"] = dict(Counter(fact.get("subject") for fact in valid_facts))
    atomic_json(RUN_ROOT / "reports/SUBJECT_BINDING_OFFLINE_AUDIT.json", report)
    atomic_text(RUN_ROOT / "reports/SUBJECT_BINDING_OFFLINE_AUDIT.md", "\n".join([
        "# Episode 91716 Subject-Binding Offline Audit", "",
        f"- Raw / valid / quarantine facts: `{report['raw_fact_count']} / {report['mechanically_valid_fact_count']} / {report['mechanical_quarantine_count']}`",
        f"- API calls in audit: `{report['api_calls_in_audit']}`",
        f"- Dominant quarantine reason: `{max(reason_counts, key=reason_counts.get) if reason_counts else 'none'}`",
        f"- Valid-fact binding issues: `{report['valid_fact_binding_issue_count']}`",
        f"- Subject-binding mechanical gate: `{report['subject_binding_mechanical_gate']}`",
        "- Semantic audit required: `true`",
        "- Reverse Writer ready: `false`",
        "",
        "The new contract stopped free subject assignment. The main remaining",
        "failure in this probe is output capacity overflow, which is separate",
        "from subject binding and is not repaired or retried here.",
        "",
    ]))
    return report


if __name__ == "__main__":
    print(json.dumps(run_audit(), ensure_ascii=False, indent=2))
