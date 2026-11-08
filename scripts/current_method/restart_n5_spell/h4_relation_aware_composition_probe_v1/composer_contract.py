"""Deterministic envelope assembly for the isolated Composer probe.

The LLM supplies only the question/option layer. Candidate IDs, evidence
groups, source contributions, and the canonical gold text are assembled from
the frozen candidate so the model cannot corrupt the evidence contract.
"""

from __future__ import annotations

from typing import Any, Mapping


def assemble_canonical_response(
    raw: Mapping[str, Any], *, candidate: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_qids = [str(qid) for qid in candidate["source_qids"]]
    unique = candidate.get("unique_contribution_closure") or {}
    groups = [
        {"source_qa_id": qid, "turn_refs": [str(ref) for ref in (unique.get(qid) or [])]}
        for qid in source_qids
    ]
    required_refs = sorted({ref for group in groups for ref in group["turn_refs"]})
    options = raw.get("options") if isinstance(raw.get("options"), Mapping) else {}
    options = {letter: str(options.get(letter) or "") for letter in "ABCD"}
    correct = str(raw.get("correct_option") or "").upper()
    gold = options.get(correct, "") if correct in options else ""
    canonical = {
        "candidate_id": str(candidate["candidate_id"]),
        "required_qa_ids": source_qids,
        "target": str(candidate.get("target") or ""),
        "question": str(raw.get("question") or ""),
        "answer_dimension": str(raw.get("answer_dimension") or ""),
        "gold_text": gold,
        "options": options,
        "correct_option": correct,
        "source_contributions": candidate.get("source_contributions") or [],
        "required_evidence_groups": groups,
        "required_evidence_turn_refs": required_refs,
        "distractor_mutations": raw.get("distractor_mutations") or {},
        "is_mechanical_conjunction": bool(raw.get("is_mechanical_conjunction", False)),
    }
    errors: list[str] = []
    if not isinstance(raw, Mapping):
        errors.append("raw_response_not_object")
    for field in ("question", "answer_dimension"):
        if not str(raw.get(field) or "").strip():
            errors.append(f"missing_raw_{field}")
    if set(options) != set("ABCD") or any(not value.strip() for value in options.values()):
        errors.append("raw_options_incomplete")
    if correct not in "ABCD":
        errors.append("raw_correct_option_invalid")
    raw_gold = str(raw.get("gold_text") or "")
    if raw_gold and raw_gold.strip() != gold.strip():
        errors.append("raw_gold_disagrees_with_selected_option")
    return canonical, {
        "valid": not errors,
        "errors": errors,
        "assembled_fields": [
            "candidate_id", "required_qa_ids", "target", "gold_text",
            "source_contributions", "required_evidence_groups",
            "required_evidence_turn_refs",
        ],
        "raw_keys": sorted(str(key) for key in raw.keys()) if isinstance(raw, Mapping) else [],
    }
