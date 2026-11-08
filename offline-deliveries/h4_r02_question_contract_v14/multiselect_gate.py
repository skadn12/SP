"""Multiselect scoring contract; citation metadata is diagnostic only.

Question and option text, raw relations and evidence are never repaired here.
The generator's supports relations determine Gold; semantics are not certified.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

MODEL = "Qwen/Qwen3.6-35B-A3B"
LABELS = "ABCDEF"
GATE = "multiselect_scoring_contract_v14"
CITATION_GATE = "DIAGNOSTIC_ONLY"


def valid_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def citation_diagnostics(evidence: dict, turns: dict) -> list[dict]:
    """Record malformed/nonmatching citation metadata without rejecting it."""
    findings = []
    for label in LABELS:
        entry = evidence.get(label)
        if not isinstance(entry, dict):
            findings.append({"label": label, "code": "citation_not_object"})
            continue
        ref = entry.get("turn_ref")
        turn = turns.get(ref) if isinstance(ref, str) else None
        if turn is None:
            findings.append({"label": label, "code": "turn_ref_unavailable"})
        else:
            if entry.get("speaker") != turn.get("speaker"):
                findings.append({"label": label, "code": "speaker_mismatch"})
            if not valid_text(entry.get("quote")) or entry["quote"] not in turn.get("text", ""):
                findings.append({"label": label, "code": "quote_not_verbatim"})
        for field in ("turn_ref", "speaker", "quote"):
            if not valid_text(entry.get(field)):
                findings.append({"label": label, "code": field + "_missing_or_not_text"})
        if not valid_text(entry.get("reason", entry.get("rationale"))):
            findings.append({"label": label, "code": "rationale_missing_or_not_text"})
    return findings


def validate_normalized(question: dict, source: dict) -> tuple[list[str], list[dict]]:
    """Shared generator/evaluator contract, excluding file I/O and semantics."""
    errors = []
    if not valid_text(question.get("question")):
        errors.append("question_missing")
    options, gold = question.get("options"), question.get("gold")
    if not isinstance(options, dict) or set(options) != set(LABELS):
        errors.append("six_options_required")
    elif any(not valid_text(text) for text in options.values()):
        errors.append("option_proposition_missing")
    elif len({" ".join(text.casefold().split()) for text in options.values()}) != 6:
        errors.append("options_not_unique")
    if not isinstance(gold, list) or not 2 <= len(gold) <= 6:
        errors.append("gold_cardinality_invalid")
    elif any(not isinstance(label, str) or label not in tuple(LABELS) for label in gold) or len(set(gold)) != len(gold):
        errors.append("gold_labels_invalid")
    evidence = question.get("option_evidence")
    if not isinstance(evidence, dict) or set(evidence) != set(LABELS):
        return errors + ["complete_relations_required"], []
    for label in LABELS:
        entry = evidence[label]
        if not isinstance(entry, dict) or entry.get("relation") not in ("supports", "contradicts"):
            errors.append(label + ":relation_invalid")
        elif isinstance(gold, list) and ((entry["relation"] == "supports") != (label in gold)):
            errors.append(label + ":relation_gold_mismatch")
    turns = {turn["turn_ref"]: turn for turn in source.get("turns", [])}
    return errors, citation_diagnostics(evidence, turns)


def validate_candidate(row: Any, expected_id: str, turns: dict,
                       root: str, source_path: Path, source_sha: str) -> tuple[list[str], dict | None]:
    """Drop-in replacement for the old bridge.validate_candidate API."""
    errors = []
    if not isinstance(row, dict):
        return ["question_schema_invalid"], None
    if row.get("candidate_id") != expected_id:
        errors.append("candidate_id_invalid")
    options = row.get("options")
    if not isinstance(options, list) or len(options) != 6:
        return errors + ["six_options_required"], None
    if any(not isinstance(option, dict) for option in options):
        return errors + ["option_schema_invalid"], None
    source_file = Path(source_path)
    try:
        data = source_file.read_bytes()
        source = json.loads(data)
        if hashlib.sha256(data).hexdigest() != source_sha:
            errors.append("source_sha_mismatch")
        if not isinstance(source, dict) or source.get("root_id") != root:
            errors.append("source_root_mismatch")
        raw_turns = source.get("turns") if isinstance(source, dict) else None
        if (not isinstance(raw_turns, list) or not raw_turns
                or any(not isinstance(turn, dict) or not all(valid_text(turn.get(field))
                       for field in ("turn_ref", "speaker", "text")) for turn in raw_turns)):
            errors.append("source_turns_invalid")
        elif len({turn["turn_ref"] for turn in raw_turns}) != len(raw_turns):
            errors.append("source_turn_refs_duplicate")
        elif {turn["turn_ref"]: turn for turn in raw_turns} != turns:
            errors.append("source_turns_binding_mismatch")
    except (OSError, ValueError, TypeError):
        errors.append("source_file_unreadable")
    if errors:
        return errors, None
    gold = [label for label, option in zip(LABELS, options) if option.get("relation") == "supports"]
    if "gold" in row and (not isinstance(row["gold"], list)
                          or any(not isinstance(label, str) for label in row["gold"])
                          or len(set(row["gold"])) != len(row["gold"])
                          or set(row["gold"]) != set(gold)):
        errors.append("raw_gold_relation_mismatch")
    evidence = {}
    for label, option in zip(LABELS, options):
        entry = {key: deepcopy(value) for key, value in option.items() if key != "proposition"}
        entry["reason"] = deepcopy(option.get("rationale", option.get("reason")))
        evidence[label] = entry
    candidate = {"candidate_id": expected_id, "root_id": root, "question_type": "multi_select",
                 "topic": deepcopy(row.get("topic")), "question": deepcopy(row.get("question")),
                 "options": {label: deepcopy(option.get("proposition")) for label, option in zip(LABELS, options)},
                 "gold": gold, "option_evidence": evidence,
                 "answer_reason": "Gold is the raw Qwen supports relations; no semantic review was performed.",
                 "source_path": str(source_file.resolve()), "source_sha256": source_sha,
                 "mechanical_gate_version": GATE, "citation_gate": CITATION_GATE,
                 "raw_model_candidate": deepcopy(row), "semantic_review_performed": False}
    structural_errors, diagnostics = validate_normalized(candidate, source)
    errors.extend(structural_errors)
    if errors:
        return errors, None
    candidate["citation_diagnostics"] = diagnostics
    if not valid_text(row.get("topic")):
        candidate["citation_diagnostics"].append({"code": "topic_missing_or_not_text"})
    return [], candidate


# Retain both names for callers that previously described this step as normalization.
normalize = validate_candidate
