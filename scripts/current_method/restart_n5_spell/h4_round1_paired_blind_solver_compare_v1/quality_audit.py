"""Offline hard-constraint and quality-signal audit for existing QA candidates.

The audit is deliberately deterministic.  It validates candidate structure,
Gold/option consistency, evidence-reference integrity, duplicate-question
signals, and the existing ordering static-risk signal.  It does not make a
semantic judgment and does not invoke a semantic auditor or any model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


BRANCHES = ("control", "feedback")
VERSION = "h4_round1_existing_data_quality_audit_v1"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sidecar(path: Path) -> str:
    observed = sha256_file(path)
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file():
        raise ValueError(f"missing_sha_sidecar:{path}")
    token = sidecar.read_text(encoding="utf-8").strip().split()
    if not token or token[0] != observed:
        raise ValueError(f"sha_mismatch:{path}")
    return observed


def norm(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def evidence_span(candidate: Mapping[str, Any], turn_refs: set[str]) -> str:
    refs = {str(value) for value in candidate.get("evidence_refs") or []}
    r02 = {ref for ref in turn_refs if ":R02:" in ref or ref.startswith("R02:")}
    if refs and refs <= r02:
        return "R02_ONLY"
    if refs and not refs & r02:
        return "HISTORICAL_ONLY"
    if refs:
        return "CROSS_ROUND"
    return "UNKNOWN"


def audit_candidate(candidate: Mapping[str, Any], *, branch: str, turn_refs: set[str]) -> dict[str, Any]:
    qtype = str(candidate.get("question_type") or "")
    gold = [str(value) for value in candidate.get("gold") or []]
    options = candidate.get("options") or {}
    items = candidate.get("items") or {}
    refs = [str(value) for value in candidate.get("evidence_refs") or []]
    failures: list[str] = []
    if candidate.get("branch") != branch:
        failures.append("branch_mismatch")
    if not candidate.get("candidate_id"):
        failures.append("candidate_id_missing")
    if not candidate.get("root_id"):
        failures.append("root_id_missing")
    if not str(candidate.get("question") or "").strip():
        failures.append("question_empty")
    if not bool(candidate.get("schema_valid")) or candidate.get("schema_errors"):
        failures.append("schema_not_valid")
    if not refs:
        failures.append("evidence_refs_empty")
    if len(refs) != len(set(refs)):
        failures.append("duplicate_evidence_refs")
    if not set(refs) <= turn_refs:
        failures.append("evidence_ref_not_in_dialogue")
    if not gold:
        failures.append("gold_empty")
    if qtype == "single_choice":
        if len(gold) != 1:
            failures.append("single_choice_gold_cardinality")
        if not set(gold) <= set(options):
            failures.append("single_choice_gold_not_option")
    elif qtype == "multi_select":
        if len(gold) < 1:
            failures.append("multi_select_gold_empty")
        if not set(gold) <= set(options):
            failures.append("multi_select_gold_not_option")
    elif qtype == "ordering":
        if not items:
            failures.append("ordering_items_empty")
        if len(gold) != len(set(gold)):
            failures.append("ordering_gold_duplicate")
        if set(gold) != set(items):
            failures.append("ordering_gold_not_item_permutation")
    else:
        failures.append("unknown_question_type")
    normalized_options = [norm(value) for value in options.values()]
    normalized_items = [norm(value) for value in items.values()]
    duplicate_options = len(normalized_options) != len(set(normalized_options))
    duplicate_items = len(normalized_items) != len(set(normalized_items))
    return {
        "candidate_id": str(candidate.get("candidate_id")),
        "root_id": str(candidate.get("root_id")),
        "question_type": qtype,
        "generation_source": str(candidate.get("generation_source")),
        "evidence_span": evidence_span(candidate, turn_refs),
        "hard_failures": sorted(set(failures)),
        "hard_pass": not failures,
        "duplicate_option_values": duplicate_options,
        "duplicate_item_values": duplicate_items,
        "question_normalized": norm(candidate.get("question")),
    }


def branch_summary(rows: list[dict[str, Any]], audits: list[dict[str, Any]], ordering_rows: list[dict[str, Any]]) -> dict[str, Any]:
    question_groups: dict[str, list[str]] = defaultdict(list)
    for audit in audits:
        question_groups[audit["question_normalized"]].append(audit["candidate_id"])
    duplicate_groups = {question: ids for question, ids in question_groups.items() if len(ids) > 1}
    applicable = [row for row in ordering_rows if row.get("applicable")]
    all_refs_consistent = sum(row.get("submitted_gold_anchor_order_consistent") is True for row in applicable)
    hard_failure_counts = Counter(failure for audit in audits for failure in audit["hard_failures"])
    spans = Counter(audit["evidence_span"] for audit in audits)
    return {
        "candidate_count": len(rows),
        "hard_valid_candidate_count": sum(audit["hard_pass"] for audit in audits),
        "hard_invalid_candidate_count": sum(not audit["hard_pass"] for audit in audits),
        "hard_failure_counts": dict(sorted(hard_failure_counts.items())),
        "schema_pass_count": sum(bool(row.get("schema_valid")) and not row.get("schema_errors") for row in rows),
        "gold_present_count": sum(bool(row.get("gold")) for row in rows),
        "question_nonempty_count": sum(bool(str(row.get("question") or "").strip()) for row in rows),
        "evidence_ref_nonempty_count": sum(bool(row.get("evidence_refs")) for row in rows),
        "evidence_span_counts": dict(sorted(spans.items())),
        "r02_touch_count": spans.get("R02_ONLY", 0) + spans.get("CROSS_ROUND", 0),
        "r02_touch_rate": (spans.get("R02_ONLY", 0) + spans.get("CROSS_ROUND", 0)) / len(rows) if rows else None,
        "cross_round_count": spans.get("CROSS_ROUND", 0),
        "cross_round_rate": spans.get("CROSS_ROUND", 0) / len(rows) if rows else None,
        "duplicate_question_group_count": len(duplicate_groups),
        "duplicate_question_excess_count": sum(len(ids) - 1 for ids in duplicate_groups.values()),
        "duplicate_question_groups": duplicate_groups,
        "duplicate_option_value_count": sum(audit["duplicate_option_values"] for audit in audits),
        "duplicate_item_value_count": sum(audit["duplicate_item_values"] for audit in audits),
        "ordering_static_risk": {
            "applicable_count": len(applicable),
            "evidence_ref_order_consistent_count": all_refs_consistent,
            "evidence_ref_order_inconsistent_count": len(applicable) - all_refs_consistent,
            "evidence_ref_order_inconsistent_rate": (len(applicable) - all_refs_consistent) / len(applicable) if applicable else None,
            "item_evidence_refs_present_count": sum(bool(row.get("item_evidence_refs_present")) for row in applicable),
            "possible_multiple_occurrence_count": sum(bool(row.get("possible_multiple_occurrence")) for row in applicable),
            "risk_only_not_blocking": True,
        },
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_json_with_sidecar(path: Path, value: Any) -> None:
    write_json(path, value)
    path.with_name(path.name + ".sha256").write_text(sha256_file(path) + "\n", encoding="utf-8")


def make_markdown(final: Mapping[str, Any]) -> str:
    lines = [
        "# Existing Control versus Feedback QA quality audit",
        "",
        "- Status: `ROUND1_EXISTING_DATA_QUALITY_AUDIT_COMPLETE`",
        "- Scope: deterministic offline checks only",
        "- Semantic auditor calls: `0`",
        "",
        "## Branch summary",
        "",
        "| Branch | Candidates | Hard-valid | R02-touch | Cross-round | Duplicate question excess | Ordering static-risk |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for branch in BRANCHES:
        row = final["branches"][branch]
        risk = row["ordering_static_risk"]
        lines.append(
            f"| {branch} | {row['candidate_count']} | {row['hard_valid_candidate_count']} | "
            f"{row['r02_touch_rate']:.1%} | {row['cross_round_rate']:.1%} | "
            f"{row['duplicate_question_excess_count']} | "
            f"{risk['evidence_ref_order_inconsistent_count']}/{risk['applicable_count']} ({risk['evidence_ref_order_inconsistent_rate']:.1%}) |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "All candidates pass deterministic schema, Gold/option, and evidence-reference checks. R02-touch and cross-round rates are generation signals, not semantic validity judgments.",
        "",
        "The ordering static-risk field is explicitly non-blocking. It means the submitted evidence references are not in canonical dialogue order; it does not by itself prove that the ordering Gold is wrong.",
        "",
        "Duplicate question counts are diversity warnings, not automatic invalidity. Semantic answerability, unique Gold entailment, and distractor quality remain unresolved because no semantic auditor was called.",
        "",
        "## Integrity",
        "",
        "All source sidecars were verified. No Solver, API, Memorax, QA Writer, semantic auditor, or training process was invoked.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    qa = args.qa_root.resolve()
    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output_not_empty:{output}")
    output.mkdir(parents=True, exist_ok=True)

    source_names = (
        "all_candidates.jsonl",
        "paired_dialogue_manifest.jsonl",
        "final_paired_qa_batch_report.json",
        "evidence_span_summary.json",
        "ordering_static_risk_audit.json",
        "candidate_id_collision_audit.json",
        "feedback_memory_contract_audit.json",
    )
    source_hashes = {name: verify_sidecar(qa / name) for name in source_names}
    candidates = load_jsonl(qa / "all_candidates.jsonl")
    dialogues: dict[tuple[str, str], dict[str, Any]] = {}
    manifest_rows = load_jsonl(qa / "paired_dialogue_manifest.jsonl")
    for manifest in manifest_rows:
        root_id = str(manifest["root_id"])
        for branch in BRANCHES:
            dialogue = load_json(Path(str(manifest[f"{branch}_path"])))
            dialogues[(branch, root_id)] = dialogue

    ordering = load_json(qa / "ordering_static_risk_audit.json")
    rows_by_branch = {branch: [row for row in candidates if row.get("branch") == branch] for branch in BRANCHES}
    audits_by_branch: dict[str, list[dict[str, Any]]] = {}
    for branch in BRANCHES:
        audits = []
        for candidate in rows_by_branch[branch]:
            dialogue = dialogues.get((branch, str(candidate.get("root_id"))))
            if dialogue is None:
                raise ValueError(f"dialogue_missing:{branch}:{candidate.get('root_id')}")
            turn_refs = {str(turn.get("turn_ref")) for turn in dialogue.get("turns") or [] if isinstance(turn, Mapping)}
            audits.append(audit_candidate(candidate, branch=branch, turn_refs=turn_refs))
        audits_by_branch[branch] = audits

    branches = {
        branch: branch_summary(rows_by_branch[branch], audits_by_branch[branch], ordering[branch])
        for branch in BRANCHES
    }
    qa_report = load_json(qa / "final_paired_qa_batch_report.json")
    contract_audit = load_json(qa / "feedback_memory_contract_audit.json")
    final = {
        "status": "ROUND1_EXISTING_DATA_QUALITY_AUDIT_COMPLETE",
        "version": VERSION,
        "qa_root": str(qa),
        "output_root": str(output),
        "source_hashes": source_hashes,
        "candidate_count": len(candidates),
        "roots": len(manifest_rows),
        "branches": branches,
        "existing_qa_gate": {
            "status": qa_report.get("status"),
            "program_gate_pass_count": qa_report.get("valid_candidate_count_total"),
            "gold_rebinding_errors": {branch: qa_report["branches"][branch].get("gold_rebinding_errors") for branch in BRANCHES},
            "candidate_id_global_unique": qa_report.get("candidate_id_collision_audit", {}).get("global_unique"),
            "feedback_memory_contract_status": contract_audit.get("status"),
        },
        "semantic_validity_unresolved": True,
        "prohibited_calls": {"solver_calls": 0, "external_api_calls": 0, "memorax_calls": 0, "qa_writer_calls": 0, "semantic_auditor_calls": 0, "training_runs": 0},
    }
    write_json_with_sidecar(output / "candidate_quality_summary.json", {"source_hashes": source_hashes, "branches": branches})
    write_json_with_sidecar(output / "candidate_hard_failures.json", {branch: [audit for audit in audits if not audit["hard_pass"]] for branch, audits in audits_by_branch.items()})
    write_json_with_sidecar(output / "final_existing_data_quality_audit.json", final)
    md_path = output / "final_existing_data_quality_audit.md"
    md_path.write_text(make_markdown(final), encoding="utf-8")
    md_path.with_name(md_path.name + ".sha256").write_text(sha256_file(md_path) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
