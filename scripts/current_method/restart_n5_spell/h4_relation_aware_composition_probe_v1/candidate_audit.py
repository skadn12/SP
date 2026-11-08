"""Deterministic audit and contribution-level evidence closure compiler."""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from scripts.current_method.restart_n5_spell.qa_composition_probe_v1.contracts import lexical_tokens


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").casefold()).strip()


def _tokens(value: Any) -> set[str]:
    return lexical_tokens(value)


def _overlap(left: Iterable[str], right: Iterable[str]) -> set[str]:
    return set(left) & set(right)


def _source_text(qa: Any) -> str:
    return " ".join([qa.question, qa.gold_text, *qa.evidence_texts])


def _closure_refs(
    *, contribution: str, seed_refs: set[str], qa: Any,
    dialogue_rows: list[dict[str, Any]],
) -> set[str]:
    """Expand only close paraphrases of the contribution, never whole source QA."""
    anchors = _tokens(contribution)
    rows_by_ref = {str(row.get("turn_id")): row for row in dialogue_rows}
    seed_text = " ".join(str(rows_by_ref.get(ref, {}).get("text") or "") for ref in seed_refs)
    seed_tokens = _tokens(seed_text)
    result = set(seed_refs)
    if not anchors:
        return result
    for row in dialogue_rows:
        ref = str(row.get("turn_id") or "")
        if ref not in set(qa.evidence_turn_refs):
            continue
        row_tokens = _tokens(row.get("text"))
        # A semantic closure candidate needs at least two anchor/support words,
        # or a strong overlap with the declared contribution evidence text.
        anchor_hits = row_tokens & anchors
        seed_hits = row_tokens & seed_tokens
        if len(anchor_hits) >= 2 or (len(anchors) <= 3 and len(anchor_hits) >= 1 and len(seed_hits) >= 2):
            result.add(ref)
    return result


def audit_selector_output(
    raw: Mapping[str, Any], *, atomic_by_id: Mapping[str, Any],
    dialogue_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = raw.get("candidates")
    errors: list[str] = []
    if not isinstance(rows, list):
        return {"valid": False, "errors": ["candidates_not_list"], "candidates": []}
    if len(rows) > 5:
        errors.append("more_than_five_candidates")
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows[:5], 1):
        if not isinstance(row, Mapping):
            errors.append(f"candidate_{index}_not_object")
            continue
        candidate_id = str(row.get("candidate_id") or f"C{index:02d}")
        source_qids = [str(value) for value in (row.get("source_qids") or [])]
        candidate_errors: list[str] = []
        if candidate_id in seen_ids:
            candidate_errors.append("duplicate_candidate_id")
        seen_ids.add(candidate_id)
        if len(source_qids) < 2 or len(set(source_qids)) != len(source_qids):
            candidate_errors.append("source_qids_must_contain_two_distinct_ids")
        if any(qid not in atomic_by_id for qid in source_qids):
            candidate_errors.append("unknown_source_qid")
        contributions = row.get("source_contributions")
        if not isinstance(contributions, list):
            candidate_errors.append("source_contributions_not_list")
            contributions = []
        by_qid: dict[str, dict[str, Any]] = {}
        for contribution in contributions:
            if not isinstance(contribution, Mapping):
                candidate_errors.append("contribution_not_object")
                continue
            qid = str(contribution.get("source_qid") or "")
            info = str(contribution.get("unique_information") or "").strip()
            refs = [str(ref) for ref in (contribution.get("evidence_turn_refs") or [])]
            if qid not in source_qids:
                candidate_errors.append(f"contribution_qid_not_in_sources:{qid}")
                continue
            if not info:
                candidate_errors.append(f"empty_contribution:{qid}")
            if qid in by_qid:
                candidate_errors.append(f"duplicate_contribution:{qid}")
            by_qid[qid] = {"source_qid": qid, "unique_information": info, "declared_refs": refs}
        if set(by_qid) != set(source_qids):
            candidate_errors.append("every_source_needs_one_contribution")
        source_tokens = {qid: _tokens(_source_text(atomic_by_id[qid])) for qid in source_qids if qid in atomic_by_id}
        contribution_tokens = {qid: _tokens(by_qid[qid]["unique_information"]) for qid in by_qid}
        for qid, item in by_qid.items():
            qa = atomic_by_id[qid]
            refs = set(item["declared_refs"])
            allowed = set(qa.evidence_turn_refs)
            if not refs or not refs.issubset(allowed):
                candidate_errors.append(f"contribution_refs_invalid:{qid}")
            support = contribution_tokens[qid] & source_tokens.get(qid, set())
            other = set().union(*(source_tokens.get(other_qid, set()) for other_qid in source_qids if other_qid != qid))
            unique_support = support - other
            if len(support) < max(1, min(2, len(contribution_tokens[qid]))):
                candidate_errors.append(f"contribution_not_supported_by_source:{qid}")
            if not unique_support:
                candidate_errors.append(f"contribution_not_unique_against_other_sources:{qid}")
            item["support_tokens"] = sorted(support)
            item["unique_support_tokens"] = sorted(unique_support)
            item["closure_refs"] = sorted(_closure_refs(
                contribution=item["unique_information"], seed_refs=refs,
                qa=qa, dialogue_rows=dialogue_rows))
        all_contribution_tokens = set().union(*(contribution_tokens.values())) if contribution_tokens else set()
        target = str(row.get("target") or "").strip()
        target_tokens = _tokens(target)
        if not target:
            candidate_errors.append("missing_target")
        if all_contribution_tokens and len(target_tokens & all_contribution_tokens) / max(1, len(all_contribution_tokens)) >= 0.7:
            candidate_errors.append("target_leaks_supporting_contributions")
        closure_sets = {qid: set(item.get("closure_refs") or []) for qid, item in by_qid.items()}
        unique_closure = {
            qid: sorted(refs - set().union(*(other for other_qid, other in closure_sets.items() if other_qid != qid)))
            for qid, refs in closure_sets.items()
        }
        if any(not refs for refs in unique_closure.values()):
            candidate_errors.append("source_contribution_has_no_unique_evidence_closure")
        normalized.append({
            "candidate_id": candidate_id,
            "source_qids": source_qids,
            "target": target,
            "source_contributions": list(by_qid.values()),
            "why_composable": str(row.get("why_composable") or ""),
            "why_each_source_should_be_necessary": str(row.get("why_each_source_should_be_necessary") or ""),
            "unique_contribution_closure": unique_closure,
            "static_errors": candidate_errors,
            "static_valid": not candidate_errors,
        })
        if candidate_errors:
            errors.extend([f"{candidate_id}:{value}" for value in candidate_errors])
    valid = bool(normalized) and any(row["static_valid"] for row in normalized)
    return {"valid": valid, "errors": errors, "candidates": normalized}
