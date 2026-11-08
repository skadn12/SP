"""Deterministic compiler for the shared feedback views.

The old Question Bank cards remain the immutable backend source.  This module
projects them once into role-specific views, so expansion and QA cannot drift
by maintaining separate feedback records.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import COMPILER_RUN, HIDDEN_BACKEND_KEYS, HISTORICAL_FEEDBACK_RUN, R02_INPUT_RUN, REROUTE_RUN, R03_ROOTS


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _candidate_value(row: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if key in row:
        return row[key]
    nested = row.get("candidate")
    return nested.get(key, default) if isinstance(nested, Mapping) else default


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def _metric(card: Mapping[str, Any], model: str, context: str) -> dict[str, Any]:
    profile = card.get("canonical_confusion_summary") or {}
    row = profile.get(model) or {}
    correct = _number(row.get(f"{context}_correct"))
    total = _number(row.get(f"{context}_total"))
    result: dict[str, Any] = {}
    if correct is not None and total is not None and total >= 0:
        result.update({"correct_count": correct, "total_count": total})
        result["accuracy"] = (correct / total) if total else None
    else:
        # Accuracy may be present in a legacy observation without denominators.
        profile_row = (card.get("solver_profile") or {}).get(model) or {}
        policy_name = model.removeprefix("M_").split("_", 1)[0].lower()
        legacy_key = f"{policy_name}_{'full' if context == 'full' else 'no'}"
        value = _number(profile_row.get(legacy_key))
        if value is None:
            value = _number(profile_row.get("full" if context == "full" else "no"))
        if value is not None:
            result["accuracy"] = value
            result["sample_size"] = "UNKNOWN"
    if not result:
        result["accuracy"] = "UNKNOWN"
        result["sample_size"] = "UNKNOWN"
    return result


def _performance(card: Mapping[str, Any]) -> dict[str, Any]:
    source_sha = card.get("source_sha256") or {}
    snapshot_id = str(source_sha.get("candidate") or "historical_frozen_solver_observation")
    return {
        model: {
            "full_context": _metric(card, model, "full"),
            "no_context": _metric(card, model, "no"),
            "observation_kind": "behavior_observation",
            "evaluation_snapshot": "historical_frozen_solver_observation",
            "observation_snapshot_id": snapshot_id,
        }
        for model in ("M_clean_policy", "M_inclusive_policy")
    }


def _dialogue_record(row: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(row["input_dialogue_path"]))
    dialogue = read_json(path)
    observed_file_sha = digest_file(path)
    if observed_file_sha != str(row.get("input_dialogue_file_sha256")):
        raise ValueError(f"authoritative_dialogue_file_sha_mismatch:{row.get('root_id')}")
    if str(dialogue.get("complete_dialogue_sha256")) != str(row.get("complete_dialogue_sha256")):
        raise ValueError(f"authoritative_dialogue_content_sha_mismatch:{row.get('root_id')}")
    turns = []
    for source in dialogue.get("turns") or []:
        turns.append({
            "turn_ref": str(source.get("turn_ref") or source.get("turn_id") or ""),
            "speaker": str(source.get("speaker") or ""),
            "text": str(source.get("text") or ""),
        })
    checks = dict(row.get("checks") or {})
    if str(row.get("source_prefix")) != "Parent+R01+R02-feedback":
        raise ValueError(f"wrong_new_r03_source_prefix:{row.get('root_id')}")
    if not all(checks.get(k) is True for k in (
        "latest_round_is_r02", "r03_turns_absent", "rounds_are_r01_r02_only",
        "root_id_match", "turn_refs_unique", "file_sha_match",
    )):
        raise ValueError(f"authoritative_dialogue_check_failed:{row.get('root_id')}")
    return {
        "root_id": str(row["root_id"]),
        "source_prefix": "Parent+R01+R02-feedback",
        "complete_dialogue_sha256": str(row["complete_dialogue_sha256"]),
        "input_dialogue_file_sha256": observed_file_sha,
        "turns": turns,
        "turn_count": len(turns),
        "checks": checks,
        "input_dialogue_path": str(path),
    }


def load_frozen_inputs() -> dict[str, Any]:
    """Load only local frozen artifacts required for projection and samples."""
    cards_path = COMPILER_RUN / "r03_expansion_feedback_cards_full.jsonl"
    candidates_path = REROUTE_RUN / "provisional_question_bank_entries.jsonl"
    dialogue_path = R02_INPUT_RUN / "r03_input_dialogue_identity_manifest.jsonl"
    common_path = HISTORICAL_FEEDBACK_RUN / "common_feedback_inputs.jsonl"
    cards = read_jsonl(cards_path)
    candidates = {str(row["candidate_id"]): row for row in read_jsonl(candidates_path)}
    dialogues = {str(row["root_id"]): _dialogue_record(row) for row in read_jsonl(dialogue_path)}
    common = {str(row.get("source_identity", {}).get("root_id")): row for row in read_jsonl(common_path)} if common_path.is_file() else {}
    if set(dialogues) != set(R03_ROOTS):
        raise ValueError(f"authoritative_root_set_mismatch:{sorted(dialogues)}")
    if len(cards) != len(R03_ROOTS):
        raise ValueError(f"expected_one_frozen_card_per_root:{len(cards)}")
    records = [build_backend_record(card, candidates, dialogues, common) for card in cards]
    return {
        "cards_path": cards_path, "candidates_path": candidates_path,
        "dialogue_path": dialogue_path, "common_path": common_path, "common": common,
        "cards": cards, "records": records,
        "dialogues": dialogues,
    }


def build_backend_record(
    card: Mapping[str, Any], candidates: Mapping[str, Mapping[str, Any]],
    dialogues: Mapping[str, Mapping[str, Any]],
    common_feedback: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    root = str(card["root_id"])
    candidate_id = str(card["source_candidate_id"])
    candidate = candidates.get(candidate_id)
    if candidate is None:
        raise ValueError(f"candidate_missing_for_card:{candidate_id}")
    option_value = _candidate_value(candidate, "options")
    item_value = _candidate_value(candidate, "items")
    gold_value = _candidate_value(candidate, "gold")
    evidence_value = _candidate_value(candidate, "evidence_refs")
    options_or_items = option_value if option_value else (item_value if item_value else {})
    qa = {
        "question_type": _candidate_value(candidate, "question_type"),
        "question": _candidate_value(candidate, "question"),
        "options_or_items": options_or_items,
        "canonical_gold": gold_value,
        "evidence_refs": list(evidence_value or []),
        "source_data_status": {
            "options_or_items": "present" if options_or_items else "missing",
            "canonical_gold": "present" if gold_value is not None else "missing",
            "evidence_refs": "present" if evidence_value else "empty_or_missing",
        },
    }
    return {
        "record_version": "shared_feedback_backend_record_v2",
        "root_id": root,
        "source_cluster_id": str(card["source_cluster_id"]),
        "source_candidate_id": candidate_id,
        "source_arm": card.get("source_arm"),
        "historical_qa": qa,
        "performance": _performance(card),
        # Only a basis and canonical counts are retained.  Counts are not
        # relabeled as wrong selections, omissions, or capability diagnoses.
        "canonical_observation": dict(card.get("canonical_confusion_summary") or {}),
        # Historical QA feedback is retained for audit/reproducibility only.
        # It is intentionally never copied into a public role projection.
        "historical_feedback_backend_only": dict((common_feedback or {}).get(root) or {}),
        "backend_source": {
            "card_sha256": digest_file(COMPILER_RUN / "r03_expansion_feedback_cards_full.jsonl"),
            "candidate_row_sha256": hashlib.sha256(canonical(candidate).encode()).hexdigest(),
            "dialogue": dict(dialogues[root]),
            "card_version": card.get("card_version"),
            "target_round": card.get("target_round"),
            "source_evidence_refs": list(card.get("source_evidence_refs") or []),
            "historical_labels": {
                "lifecycle": card.get("lifecycle"),
                "operation": card.get("operation"),
                "selection_weight": card.get("selection_weight"),
                "source_sha256": card.get("source_sha256"),
                "target_dialogue_sha256": card.get("target_dialogue_sha256"),
                "target_namespace": card.get("target_namespace"),
            },
        },
    }


def expansion_view(record: Mapping[str, Any]) -> dict[str, Any]:
    qa = record["historical_qa"]
    # Frozen v2 projection retained for historical prompt reproduction.
    return {
        "root_id": record["root_id"],
        "source_cluster_id": record["source_cluster_id"],
        "historical_qa": {
            "question_type": qa.get("question_type"),
            "question": qa.get("question"),
            "evidence_refs": list(qa.get("evidence_refs") or []),
        },
        "solver_observations": record["performance"],
    }


def expansion_view_v3(record: Mapping[str, Any]) -> dict[str, Any]:
    """Complete role view for the repaired expansion route."""
    qa = record["historical_qa"]
    status = qa.get("source_data_status") or {}
    if status.get("options_or_items") == "missing" or status.get("canonical_gold") == "missing":
        return {
            "root_id": record["root_id"],
            "source_cluster_id": record["source_cluster_id"],
            "source_candidate_id": record["source_candidate_id"],
            "historical_qa": None,
            "historical_qa_status": "SOURCE_DATA_INCOMPLETE_NOT_MODEL_VISIBLE",
            "solver_observations": record["performance"],
        }
    return {
        "root_id": record["root_id"],
        "source_cluster_id": record["source_cluster_id"],
        "source_candidate_id": record["source_candidate_id"],
        "historical_qa": {
            "question_type": qa.get("question_type"),
            "question": qa.get("question"),
            "options_or_items": qa.get("options_or_items") or {},
            "canonical_gold": qa.get("canonical_gold"),
            "evidence_refs": list(qa.get("evidence_refs") or []),
            "source_data_status": status,
        },
        "solver_observations": record["performance"],
    }


def qa_targeted_view(record: Mapping[str, Any]) -> dict[str, Any]:
    observation = {
        "basis": record["canonical_observation"].get("basis"),
        "M_clean_policy": {key: value for key, value in
                           (record["canonical_observation"].get("M_clean_policy") or {}).items()
                           if key != "position_mechanism"},
        "M_inclusive_policy": {key: value for key, value in
                               (record["canonical_observation"].get("M_inclusive_policy") or {}).items()
                               if key != "position_mechanism"},
    }
    return {
        "root_id": record["root_id"],
        "source_cluster_id": record["source_cluster_id"],
        "source_candidate_id": record["source_candidate_id"],
        "historical_qa": record["historical_qa"],
        "solver_observations": record["performance"],
        "canonical_observation": observation,
    }


def qa_composition_view(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    selected = list(records)
    return {
        "historical_qas": [{
            "root_id": r["root_id"], "source_cluster_id": r["source_cluster_id"],
            "source_candidate_id": r["source_candidate_id"],
            **r["historical_qa"],
        } for r in selected],
        "source_count": len(selected),
    }


def qa_fresh_view(record: Mapping[str, Any], negative_signatures: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    # Fresh deliberately has no source question, options, gold, or source ID.
    signatures = [{
        "normalized_stem_signature": row.get("normalized_stem_signature"),
        "prompt_cluster_count": row.get("prompt_cluster_count"),
        "candidate_count": row.get("candidate_count"),
    } for row in negative_signatures]
    return {
        "root_id": record["root_id"],
        "historical_coverage": signatures,
        "constraint_note": "signature_only_duplicate_avoidance; no source QA binding",
    }


def qa_fresh_view_v3(record: Mapping[str, Any], negative_signatures: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Fresh routing identity plus coverage, without a source-question bind."""
    view = qa_fresh_view(record, negative_signatures)
    view["source_cluster_id"] = record["source_cluster_id"]
    return view


def hidden_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) in HIDDEN_BACKEND_KEYS:
                found.add(str(key))
            found |= hidden_keys(child)
    elif isinstance(value, list):
        for child in value:
            found |= hidden_keys(child)
    return found
