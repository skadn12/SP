#!/usr/bin/env python3
"""Audit Stage3 final-QA candidates for offline, truth-strict salvage.

This tool is intentionally read-only with respect to the source run.  It
reconstructs the exact append-only dialogue from saved API payloads, replays
the local graph/dialogue/alignment gates, verifies the frozen dependency
hashes, and then separates:

* already accepted strict-clean rows;
* candidates whose QA representation has a unique local repair;
* candidates that need a semantic decision or rewrite; and
* candidates whose question/target truth is not established by the dialogue.

It never calls a model API and never writes a repaired clean sample.  The
repairable output is an audit queue for the next, separately tested compiler
step.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.current_method import native_api_graph_dialogue_expansion as native  # noqa: E402


SCHEMA_VERSION = "stage3_fastlane_salvage_audit_v1"
CLASS_STRICT_CLEAN = "strict_clean"
CLASS_REPAIRABLE = "representation_repairable"
CLASS_DIAGNOSTIC = "diagnostic_semantic_rewrite_required"
CLASS_INVALID_TRUTH = "invalid_truth_or_target"
CLASS_UNRECONSTRUCTABLE = "unreconstructable"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def required_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def event_id(event: dict[str, Any]) -> str:
    return native.normalize_space(event.get("event_instance_id") or event.get("event_id"))


def payload_turns(record: dict[str, Any]) -> list[dict[str, Any]]:
    payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
    return list(native._as_dialogue_list(native.appended_dialogue_from_proposal(payload)))


@dataclass
class ReconstructedDialogue:
    turns: list[dict[str, Any]]
    dialogue_attempt: int
    audit: dict[str, Any]


def reconstruct_saved_appended_dialogue(raw_record: dict[str, Any]) -> ReconstructedDialogue:
    """Rebuild the exact final marked append in generation order.

    The long-form protocol stores the initial fact chunk, each neutral
    continuation, the fact chunk appended immediately after that continuation,
    and any final visible-claim append repairs in separate audit arrays.  The
    saved continuation prefix count is treated as a fail-closed integrity gate.
    """

    attempts = [
        item
        for item in raw_record.get("api_dialogue_attempts") or []
        if isinstance(item, dict)
    ]
    if not attempts:
        raise ValueError("no saved api_dialogue_attempts")
    initial = attempts[-1]
    try:
        dialogue_attempt = int(initial.get("attempt"))
    except (TypeError, ValueError) as exc:
        raise ValueError("saved dialogue attempt has no integer attempt id") from exc
    turns = payload_turns(initial)
    if not turns:
        raise ValueError("saved final dialogue attempt has no turns")

    fact_chunks: dict[tuple[int, int], dict[str, Any]] = {}
    for item in raw_record.get("api_fact_only_evidence_chunks") or []:
        if not isinstance(item, dict):
            continue
        try:
            key = (int(item.get("dialogue_attempt")), int(item.get("fact_chunk_index")))
        except (TypeError, ValueError):
            raise ValueError("fact-only chunk has invalid attempt or chunk index")
        if key in fact_chunks:
            raise ValueError(f"duplicate fact-only chunk key: {key}")
        fact_chunks[key] = item

    prefix_checks: list[dict[str, Any]] = []
    consumed_fact_keys: set[tuple[int, int]] = set()
    continuation_count = 0
    appended_fact_turn_count = 0
    for continuation in raw_record.get("api_dialogue_continuations") or []:
        if not isinstance(continuation, dict):
            continue
        if int(continuation.get("dialogue_attempt") or -1) != dialogue_attempt:
            continue
        continuation_count += 1
        expected_prefix = int(continuation.get("prefix_turn_count") or -1)
        actual_prefix = len(turns)
        prefix_checks.append(
            {
                "continuation_index": continuation_count,
                "expected_prefix_turn_count": expected_prefix,
                "actual_prefix_turn_count": actual_prefix,
                "pass": expected_prefix == actual_prefix,
            }
        )
        if expected_prefix != actual_prefix:
            raise ValueError(
                "saved continuation prefix mismatch: "
                f"expected={expected_prefix}, actual={actual_prefix}"
            )
        continuation_turns = payload_turns(continuation)
        meta = continuation.get("meta") if isinstance(continuation.get("meta"), dict) else {}
        if meta.get("used") and not continuation_turns:
            raise ValueError("used continuation contains no turns")
        if meta.get("used"):
            turns.extend(copy.deepcopy(continuation_turns))

        chunk_index = meta.get("fact_chunk_index")
        if chunk_index is None:
            continue
        key = (dialogue_attempt, int(chunk_index))
        fact_chunk = fact_chunks.get(key)
        if fact_chunk is None:
            continue
        consumed_fact_keys.add(key)
        fact_meta = fact_chunk.get("meta") if isinstance(fact_chunk.get("meta"), dict) else {}
        fact_turns = payload_turns(fact_chunk)
        if fact_meta.get("used") and not fact_turns:
            raise ValueError(f"used fact-only chunk {key} contains no turns")
        if fact_meta.get("used"):
            turns.extend(copy.deepcopy(fact_turns))
            appended_fact_turn_count += len(fact_turns)

    selected_fact_keys = {
        key for key in fact_chunks if key[0] == dialogue_attempt
    }
    unconsumed_fact_keys = sorted(selected_fact_keys - consumed_fact_keys)
    if unconsumed_fact_keys:
        raise ValueError(f"unconsumed fact-only chunks: {unconsumed_fact_keys}")

    visible_repair_count = 0
    visible_repair_turn_count = 0
    for repair in raw_record.get("api_visible_claim_append_repairs") or []:
        if not isinstance(repair, dict):
            continue
        if int(repair.get("dialogue_attempt") or -1) != dialogue_attempt:
            continue
        visible_repair_count += 1
        repair_meta = repair.get("meta") if isinstance(repair.get("meta"), dict) else {}
        repair_turns = payload_turns(repair)
        if repair_meta.get("used") and not repair_turns:
            raise ValueError("used visible-claim repair contains no turns")
        if repair_meta.get("used"):
            turns.extend(copy.deepcopy(repair_turns))
            visible_repair_turn_count += len(repair_turns)

    return ReconstructedDialogue(
        turns=turns,
        dialogue_attempt=dialogue_attempt,
        audit={
            "initial_turn_count": len(payload_turns(initial)),
            "continuation_count": continuation_count,
            "fact_chunk_count": len(selected_fact_keys),
            "appended_fact_turn_count": appended_fact_turn_count,
            "visible_repair_count": visible_repair_count,
            "visible_repair_turn_count": visible_repair_turn_count,
            "final_appended_marked_turn_count": len(turns),
            "prefix_checks": prefix_checks,
            "all_prefix_checks_pass": all(item["pass"] for item in prefix_checks),
            "unconsumed_fact_chunk_keys": unconsumed_fact_keys,
        },
    )


def resolve_artifact_path(raw_record: dict[str, Any], run_dir: Path) -> Path:
    raw_path = native.normalize_space(raw_record.get("materialized_stage3_graph_artifact_path"))
    if not raw_path:
        meta = raw_record.get("api_graph_meta") if isinstance(raw_record.get("api_graph_meta"), dict) else {}
        raw_path = native.normalize_space(meta.get("artifact_path"))
    if not raw_path:
        raise ValueError("candidate has no graph artifact path")
    path = Path(raw_path)
    if not path.is_absolute():
        path = run_dir / path
    if not path.is_file():
        raise ValueError(f"graph artifact does not exist: {path}")
    return path.resolve()


def frozen_dependencies(raw_record: dict[str, Any]) -> dict[str, str]:
    attempts = [
        item
        for item in raw_record.get("api_stage64_final_qa_attempts") or []
        if isinstance(item, dict)
    ]
    if not attempts:
        raise ValueError("candidate did not reach frozen final QA")
    deps = required_object(attempts[-1].get("frozen_dependencies"), "frozen_dependencies")
    required = ("graph_snapshot_hash", "dialogue_hash", "alignment_hash")
    result = {key: native.normalize_space(deps.get(key)) for key in required}
    if any(not result[key] for key in required):
        raise ValueError(f"incomplete frozen dependencies: {result}")
    return result


def parent_index(parent_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in parent_rows:
        key = native.normalize_space(row.get("sample_id") or row.get("source_id"))
        if not key:
            continue
        if key in result:
            raise ValueError(f"duplicate parent id: {key}")
        result[key] = row
    return result


def replay_candidate(
    raw_record: dict[str, Any],
    run_dir: Path,
    parents_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    artifact_path = resolve_artifact_path(raw_record, run_dir)
    artifact = required_object(json.loads(artifact_path.read_text(encoding="utf-8")), "artifact")
    current_event_plan = copy.deepcopy(
        required_object(artifact.get("current_event_plan"), "artifact.current_event_plan")
    )
    anchor_to_event = required_object(
        current_event_plan.get("current_anchor_to_event_instance_id"),
        "current_event_plan.current_anchor_to_event_instance_id",
    )
    seed_sample_id = native.normalize_space(raw_record.get("seed_sample_id"))
    seed_row = parents_by_id.get(seed_sample_id)
    if seed_row is None:
        raise ValueError(f"no unique parent row for {seed_sample_id!r}")
    raw_parent_turns, parent_dialogue_source = native.parent_dialogue_turns_from_row(seed_row)
    known_anchors = list(anchor_to_event) + native.collect_legacy_anchor_strings(seed_row)
    parent_turns = native.solver_visible_parent_dialogue_turns(raw_parent_turns, known_anchors)
    reconstructed = reconstruct_saved_appended_dialogue(raw_record)
    sample_index = int(raw_record.get("generation_slot_id") or 0)
    candidate = native.validate_appended_dialogue_candidate(
        appended_dialogue_marked=reconstructed.turns,
        event_alignment_proposal=None,
        current_event_plan=current_event_plan,
        anchor_to_event=anchor_to_event,
        parent_dialogue_turns=parent_turns,
        known_anchors=known_anchors,
        sample_index=sample_index,
        # Length already passed before frozen final QA.  Replay proves exact
        # identity by hash and reruns semantic/alignment/dispersion gates; it
        # deliberately avoids loading a tokenizer.
        min_added_turns=0,
        min_added_tokens=0,
        tokenizer_path=None,
        parent_baseline_tokens=0,
        fallback_min_added_tokens=0,
        emergency_min_added_tokens=0,
        require_evidence_dispersion=True,
        deterministic_scaffold_used=False,
        formal_mining=True,
        local_evidence_compiler_used=False,
    )
    final_dialogue = required_object(candidate.get("final_dialogue"), "replayed final_dialogue")
    alignment_result = required_object(candidate.get("alignment_result"), "replayed alignment_result")
    current_alignment = required_object(
        alignment_result.get("current_alignment"), "replayed current_alignment"
    )
    actual_hashes = {
        "graph_snapshot_hash": native.normalize_space(current_event_plan.get("graph_snapshot_hash")),
        "dialogue_hash": native.normalize_space(final_dialogue.get("dialogue_hash")),
        "alignment_hash": native.normalize_space(alignment_result.get("alignment_hash")),
    }
    expected_hashes = frozen_dependencies(raw_record)
    hash_match = actual_hashes == expected_hashes
    replay_pass = bool(candidate.get("pass")) and hash_match
    graph_update = native.compile_graph_update_from_current_event_plan(
        current_event_plan, current_alignment, actual_hashes
    )
    return {
        "pass": replay_pass,
        "validator_pass": bool(candidate.get("pass")),
        "validator_reason": candidate.get("reason"),
        "hash_match": hash_match,
        "expected_hashes": expected_hashes,
        "actual_hashes": actual_hashes,
        "artifact_path": str(artifact_path),
        "artifact_sha256": file_sha256(artifact_path),
        "artifact_id": artifact.get("artifact_id") or raw_record.get("materialized_stage3_graph_artifact_id") or raw_record.get("fixed_stage3_graph_artifact_id"),
        "parent_dialogue_source": parent_dialogue_source,
        "parent_turn_count": len(parent_turns),
        "reconstruction": reconstructed.audit,
        "evidence_dispersion_pass": bool(
            (candidate.get("evidence_dispersion") or {}).get("pass")
        ),
        "required_event_count": int(
            (candidate.get("evidence_dispersion") or {}).get("required_event_count") or 0
        ),
        "aligned_event_count": len(current_alignment),
        "current_event_plan": current_event_plan,
        "current_alignment": current_alignment,
        "graph_update": graph_update,
        "final_dialogue": final_dialogue,
        "appended_dialogue_marked": reconstructed.turns,
        "known_anchors": known_anchors,
    }


def qa_candidates(raw_record: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source_field in ("api_stage64_final_qa_attempts", "api_stage64_qa_repair_attempts"):
        for source_index, item in enumerate(raw_record.get(source_field) or []):
            if not isinstance(item, dict):
                continue
            payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
            proposal = payload.get("qa_proposal")
            if not isinstance(proposal, dict):
                continue
            result.append(
                {
                    "candidate_index": len(result),
                    "source_field": source_field,
                    "source_index": source_index,
                    "failure_that_triggered_candidate": item.get("failure_reason"),
                    "proposal_sha256": json_sha256(proposal),
                    "proposal": copy.deepcopy(proposal),
                }
            )
    return result


def question_text(proposal: dict[str, Any]) -> str:
    return native.normalize_space(proposal.get("question") or proposal.get("question_text"))


def option_audit(proposal: dict[str, Any]) -> dict[str, Any]:
    options = proposal.get("options") if isinstance(proposal.get("options"), list) else []
    texts = [normalized_text(item.get("text")) for item in options if isinstance(item, dict)]
    duplicates = sorted(text for text, count in Counter(texts).items() if text and count > 1)
    truth_values = [item.get("truth_value") for item in options if isinstance(item, dict)]
    true_labels = [
        native.normalize_space(item.get("id")).upper()
        for item in options
        if isinstance(item, dict) and item.get("truth_value") is True
    ]
    gold = [native.normalize_space(item).upper() for item in proposal.get("gold") or []]
    return {
        "option_count": len(options),
        "unique_option_text_count": len(set(texts)),
        "duplicate_option_texts": duplicates,
        "all_truth_values_boolean": len(truth_values) == len(options)
        and all(isinstance(value, bool) for value in truth_values),
        "true_option_labels": true_labels,
        "gold_labels": gold,
        "truth_labels_match_gold": bool(gold) and true_labels == gold,
    }


def events_by_id(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        event_id(item): item
        for item in plan.get("events") or []
        if isinstance(item, dict) and event_id(item)
    }


def proposal_refs(proposal: dict[str, Any], field: str) -> list[str]:
    query_plan = proposal.get("query_plan") if isinstance(proposal.get("query_plan"), dict) else {}
    value = query_plan.get(field)
    if isinstance(value, list):
        return [native.normalize_space(item) for item in value if native.normalize_space(item)]
    return []


def proposal_anchor_group(proposal: dict[str, Any], field: str) -> list[str]:
    anchors = proposal.get("anchor_refs") if isinstance(proposal.get("anchor_refs"), dict) else {}
    value = anchors.get(field)
    if isinstance(value, list):
        return [native.normalize_space(item) for item in value if native.normalize_space(item)]
    return []


def best_comparison_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    def score(item: dict[str, Any]) -> tuple[int, int, int]:
        proposal = item["proposal"]
        text = normalized_text(question_text(proposal))
        banned = int("latest confirmed detail" in text)
        return (banned, -int(isinstance(proposal.get("query_plan"), dict)), -item["candidate_index"])

    return min(candidates, key=score)


def derive_comparison_contract(plan: dict[str, Any]) -> dict[str, Any] | None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in plan.get("events") or []:
        if not isinstance(item, dict):
            continue
        try:
            float(item.get("value"))
        except (TypeError, ValueError):
            continue
        grouped[native.normalize_space(item.get("slot"))].append(item)
    contracts: list[dict[str, Any]] = []
    for slot, items in grouped.items():
        entities = sorted({native.normalize_space(item.get("entity")) for item in items})
        if len(entities) != 2 or len(items) < 4:
            continue
        active = [item for item in items if str(item.get("status")) in native.ACTIVE_STATUSES]
        if len(active) != 2 or {native.normalize_space(item.get("entity")) for item in active} != set(entities):
            continue
        current_times = {
            int(item.get("global_time_index") if item.get("global_time_index") is not None else item.get("time_index") or 0)
            for item in active
        }
        if len(current_times) != 1:
            continue
        inactive = [item for item in items if item not in active]
        inactive_by_time: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for item in inactive:
            time_value = int(
                item.get("global_time_index")
                if item.get("global_time_index") is not None
                else item.get("time_index") or 0
            )
            inactive_by_time[time_value].append(item)
        matched_old_pairs = [
            pair
            for pair in inactive_by_time.values()
            if len(pair) == 2
            and {native.normalize_space(item.get("entity")) for item in pair} == set(entities)
        ]
        if len(matched_old_pairs) != 1:
            continue
        current_by_entity = {
            native.normalize_space(item.get("entity")): item for item in active
        }
        left, right = entities
        answer = abs(float(current_by_entity[left]["value"]) - float(current_by_entity[right]["value"]))
        answer_text = str(int(answer)) if answer.is_integer() else str(answer)
        current_ids = [event_id(current_by_entity[left]), event_id(current_by_entity[right])]
        old_by_entity = {
            native.normalize_space(item.get("entity")): item for item in matched_old_pairs[0]
        }
        old_ids = [event_id(old_by_entity[left]), event_id(old_by_entity[right])]
        contracts.append(
            {
                "slot": slot,
                "entities": entities,
                "current_event_ids": current_ids,
                "historical_event_ids": old_ids,
                "reasoning_event_ids": current_ids + old_ids,
                "answer": answer_text,
            }
        )
    return contracts[0] if len(contracts) == 1 else None


def classify_comparison(
    plan: dict[str, Any],
    target: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> tuple[str, list[str], dict[str, Any], list[str], int | None]:
    contract = derive_comparison_contract(plan)
    if contract is None:
        return (
            CLASS_DIAGNOSTIC,
            ["comparison_contract_not_unique"],
            {"unique_contract": False},
            [],
            None,
        )
    selected = best_comparison_candidate(candidates)
    proposal = selected["proposal"]
    options = proposal.get("options") if isinstance(proposal.get("options"), list) else []
    matching_options = [
        item
        for item in options
        if isinstance(item, dict) and native.normalize_space(item.get("text")) == contract["answer"]
    ]
    gold_labels = [native.normalize_space(item).upper() for item in proposal.get("gold") or []]
    answer_truth_pass = (
        len(matching_options) == 1
        and [native.normalize_space(matching_options[0].get("id")).upper()] == gold_labels
        and native.normalize_space(proposal.get("gold_answer_text")) == contract["answer"]
    )
    qtext = normalized_text(question_text(proposal))
    question_scope_pass = all(normalized_text(entity) in qtext for entity in contract["entities"])
    question_scope_pass = question_scope_pass and all(
        token in qtext for token in normalized_text(contract["slot"]).replace("_", " ").split()
    )
    truth = {
        "unique_contract": True,
        "contract": contract,
        "answer_truth_pass": answer_truth_pass,
        "question_scope_pass": question_scope_pass,
        "selected_candidate_index": selected["candidate_index"],
        "option_audit": option_audit(proposal),
        "dialogue_contract_fields_explicit": True,
    }
    if not answer_truth_pass or not question_scope_pass:
        return (
            CLASS_INVALID_TRUTH,
            ["comparison_question_or_gold_not_uniquely_grounded"],
            truth,
            [],
            selected["candidate_index"],
        )
    repairs = [
        "derive option.truth_value from the unique computed answer",
        "canonicalize reasoning/dependency refs to the unique current and historical time-matched pairs",
        "mark non-gold numeric options as synthetic computed distractors instead of false event provenance",
    ]
    if "latest confirmed detail" in normalized_text(question_text(candidates[0]["proposal"])):
        repairs.append("use the already saved non-meta question wording from the later QA candidate")
    return (
        CLASS_REPAIRABLE,
        ["unique_numeric_contract", "truth_preserving_representation_completion"],
        truth,
        repairs,
        selected["candidate_index"],
    )


def classify_count(
    plan: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> tuple[str, list[str], dict[str, Any], list[str], int | None]:
    members: list[dict[str, Any]] = []
    for item in plan.get("events") or []:
        if isinstance(item, dict) and isinstance(item.get("set_membership"), dict):
            members.append(item)
    set_ids = {native.normalize_space(item["set_membership"].get("set_id")) for item in members}
    conditions = {native.normalize_space(item["set_membership"].get("condition")) for item in members}
    included = [item for item in members if item["set_membership"].get("included") is True]
    excluded = [item for item in members if item["set_membership"].get("included") is False]
    contract_unique = bool(members) and len(set_ids) == 1 and len(conditions) == 1 and len(included) + len(excluded) == len(members)
    proposal = candidates[-1]["proposal"]
    expected_answer = str(len(included))
    answer_pass = (
        contract_unique
        and native.normalize_space(proposal.get("gold_answer_text")) == expected_answer
    )
    # The graph-only included flag is not enough.  A count is answerable from
    # the dialogue only when the shared predicate/decision is stated in the
    # visible support claim.  Generic emotional wording is deliberately not
    # treated as an exact include/exclude decision.
    explicit_membership_event_ids = []
    for item in members:
        claim = normalized_text(item.get("dialogue_support_claim"))
        membership = item["set_membership"]
        decision_words = (
            ("included", "qualifies", "counts toward")
            if membership.get("included") is True
            else ("excluded", "does not qualify", "does not count")
        )
        if any(word in claim for word in decision_words):
            explicit_membership_event_ids.append(event_id(item))
    dialogue_membership_explicit = len(explicit_membership_event_ids) == len(members)
    truth = {
        "unique_graph_membership_contract": contract_unique,
        "set_id": next(iter(set_ids), ""),
        "condition": next(iter(conditions), ""),
        "included_event_ids": [event_id(item) for item in included],
        "excluded_event_ids": [event_id(item) for item in excluded],
        "graph_computed_answer": expected_answer,
        "proposal_answer_matches_graph": answer_pass,
        "dialogue_membership_decision_explicit": dialogue_membership_explicit,
        "explicit_membership_event_ids": explicit_membership_event_ids,
        "option_audit": option_audit(proposal),
    }
    if not contract_unique or not answer_pass:
        return (
            CLASS_INVALID_TRUTH,
            ["count_graph_contract_or_answer_invalid"],
            truth,
            [],
            candidates[-1]["candidate_index"],
        )
    if not dialogue_membership_explicit:
        return (
            CLASS_INVALID_TRUTH,
            [
                "count_membership_exists_only_in_ir",
                "dialogue_does_not_explicitly_support_include_exclude_decisions",
            ],
            truth,
            [],
            candidates[-1]["candidate_index"],
        )
    return (
        CLASS_REPAIRABLE,
        ["unique_count_contract", "truth_preserving_representation_completion"],
        truth,
        [
            "derive option.truth_value from the unique count",
            "materialize the exact shared filter predicate and included refs",
        ],
        candidates[-1]["candidate_index"],
    )


def mentioned_value(question: str, event: dict[str, Any]) -> bool:
    q = normalized_text(question)
    value = normalized_text(event.get("value") or event.get("dialogue_value"))
    if value and value in q:
        return True
    tokens = [token for token in re.findall(r"[a-z0-9]+", value) if len(token) >= 5]
    return bool(tokens) and len(set(tokens) & set(re.findall(r"[a-z0-9]+", q))) >= min(3, len(set(tokens)))


def classify_bridge(
    plan: dict[str, Any],
    target: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> tuple[str, list[str], dict[str, Any], list[str], int | None]:
    proposal = candidates[-1]["proposal"]
    template = native.normalize_space(target.get("reasoning_template"))
    by_id = events_by_id(plan)
    truth: dict[str, Any] = {
        "reasoning_template": template,
        "option_audit": option_audit(proposal),
    }
    if template == "follow_intermediate_dependency":
        terminals = [
            item
            for item in by_id.values()
            if isinstance(item.get("depends_on"), list) and item.get("depends_on")
        ]
        unique_terminal = terminals[0] if len(terminals) == 1 else None
        expected = (
            [native.normalize_space(item) for item in unique_terminal.get("depends_on") or []]
            + [event_id(unique_terminal)]
            if unique_terminal is not None
            else []
        )
        resolved = proposal_refs(proposal, "reasoning_anchor_path")
        if not resolved:
            resolved = proposal_anchor_group(proposal, "support")
        qtext = question_text(proposal)
        mentioned_dependencies = [
            ref for ref in expected[:-1] if ref in by_id and mentioned_value(qtext, by_id[ref])
        ]
        terminal_claim_states_dependency = bool(
            unique_terminal
            and re.search(
                r"\b(?:because|therefore|after|depends? on|led to|caused by)\b",
                normalized_text(unique_terminal.get("dialogue_support_claim")),
            )
        )
        truth.update(
            {
                "unique_dependency_terminal": event_id(unique_terminal) if unique_terminal else None,
                "expected_reasoning_event_ids": expected,
                "proposal_reasoning_event_ids": resolved,
                "question_mentions_dependency_event_ids": mentioned_dependencies,
                "terminal_dialogue_claim_expresses_dependency": terminal_claim_states_dependency,
            }
        )
        return (
            CLASS_INVALID_TRUTH,
            [
                "bridge_reasoning_path_not_realized",
                "dependency_exists_in_ir_but_not_in_question_or_dialogue_claim",
            ],
            truth,
            [],
            candidates[-1]["candidate_index"],
        )

    support_refs = proposal_refs(proposal, "reasoning_anchor_path")
    if len(support_refs) < 2:
        support_refs = proposal_anchor_group(proposal, "support")
    support_events = [by_id[ref] for ref in support_refs if ref in by_id]
    entities = [native.normalize_space(item.get("entity")) for item in support_events]
    mentioned = [event_id(item) for item in support_events if mentioned_value(question_text(proposal), item)]
    truth.update(
        {
            "support_event_ids": [event_id(item) for item in support_events],
            "support_entities": entities,
            "question_mentions_support_event_ids": mentioned,
            "same_entity_conjunction": bool(entities) and len(set(entities)) == 1,
        }
    )
    reasons: list[str] = []
    if len(set(entities)) > 1:
        reasons.append("bridge_conjoins_facts_about_different_entities")
    if len(mentioned) < 2:
        reasons.append("bridge_question_realizes_fewer_than_two_support_facts")
    if truth["option_audit"]["duplicate_option_texts"]:
        reasons.append("single_choice_options_are_answer_ambiguous")
    if not reasons:
        reasons.append("bridge_target_not_uniquely_proven")
    return (
        CLASS_INVALID_TRUTH,
        reasons,
        truth,
        [],
        candidates[-1]["candidate_index"],
    )


def classify_relation_lookup(
    plan: dict[str, Any],
    target: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> tuple[str, list[str], dict[str, Any], list[str], int | None]:
    by_id = events_by_id(plan)
    preferred_type = native.normalize_space(target.get("preferred_answer_type"))
    candidate_audits = []
    for item in candidates:
        proposal = item["proposal"]
        refs = proposal_refs(proposal, "reasoning_anchor_path")
        candidate_audits.append(
            {
                "candidate_index": item["candidate_index"],
                "answer_type": native.normalize_space(proposal.get("answer_type")),
                "question": question_text(proposal),
                "reasoning_anchor_path": refs,
                "valid_reasoning_ref_count": len([ref for ref in refs if ref in by_id]),
                "option_audit": option_audit(proposal),
            }
        )
    satisfies_target = [
        item
        for item in candidate_audits
        if item["answer_type"] == preferred_type
        and item["valid_reasoning_ref_count"] >= 2
        and not item["option_audit"]["duplicate_option_texts"]
    ]
    truth = {
        "preferred_answer_type": preferred_type,
        "candidate_audits": candidate_audits,
        "saved_candidate_satisfying_answer_type_reasoning_and_unique_options": bool(satisfies_target),
    }
    if satisfies_target:
        return (
            CLASS_REPAIRABLE,
            ["relation_contract_saved_candidate_is_truth_complete"],
            truth,
            ["canonicalize redundant relation query-plan fields"],
            satisfies_target[0]["candidate_index"],
        )
    return (
        CLASS_DIAGNOSTIC,
        [
            "no_saved_candidate_jointly_realizes_relation_hop_and_preferred_answer_type",
            "repair_requires_question_and_option_semantic_rewrite",
        ],
        truth,
        [],
        None,
    )


def reject_error_text(raw_record: dict[str, Any]) -> str:
    value = raw_record.get("reject_error")
    if isinstance(value, dict):
        return native.normalize_space(value.get("error") or value.get("reason"))
    return native.normalize_space(value)


def classify_replayed_candidate(
    raw_record: dict[str, Any], replay: dict[str, Any]
) -> tuple[str, list[str], dict[str, Any], list[str], int | None]:
    if not replay.get("pass"):
        return (
            CLASS_UNRECONSTRUCTABLE,
            ["frozen_dialogue_or_alignment_replay_failed"],
            {
                "validator_pass": replay.get("validator_pass"),
                "hash_match": replay.get("hash_match"),
            },
            [],
            None,
        )
    if raw_record.get("accepted") is True:
        return (
            CLASS_STRICT_CLEAN,
            ["original_clean_and_frozen_replay_exact"],
            {"original_clean": True},
            [],
            None,
        )
    candidates = qa_candidates(raw_record)
    if not candidates:
        return (
            CLASS_UNRECONSTRUCTABLE,
            ["no_saved_qa_proposal"],
            {},
            [],
            None,
        )
    plan = replay["current_event_plan"]
    target = raw_record.get("stage64_target") if isinstance(raw_record.get("stage64_target"), dict) else {}
    operation = native.normalize_space(target.get("qa_operation") or plan.get("operation"))
    if operation == "comparison":
        return classify_comparison(plan, target, candidates)
    if operation == "count":
        return classify_count(plan, candidates)
    if operation == "bridge":
        return classify_bridge(plan, target, candidates)
    if operation == "relation_lookup":
        return classify_relation_lookup(plan, target, candidates)
    return (
        CLASS_DIAGNOSTIC,
        ["operation_has_no_fastlane_v1_truth_classifier"],
        {"operation": operation},
        [],
        None,
    )


def compact_qa_candidate(item: dict[str, Any]) -> dict[str, Any]:
    proposal = item["proposal"]
    return {
        "candidate_index": item["candidate_index"],
        "source_field": item["source_field"],
        "source_index": item["source_index"],
        "failure_that_triggered_candidate": item["failure_that_triggered_candidate"],
        "proposal_sha256": item["proposal_sha256"],
        "question": question_text(proposal),
        "answer_type": proposal.get("answer_type"),
        "gold": proposal.get("gold"),
        "gold_answer_text": proposal.get("gold_answer_text"),
        "option_audit": option_audit(proposal),
        "reasoning_template": (
            proposal.get("query_plan", {}).get("reasoning_template")
            if isinstance(proposal.get("query_plan"), dict)
            else None
        ),
        "reasoning_anchor_path": proposal_refs(proposal, "reasoning_anchor_path"),
        "dependency_closure_refs": proposal_refs(proposal, "dependency_closure_refs"),
    }


def source_paths(run_dir: Path) -> dict[str, Path]:
    generation = run_dir / "generation"
    return {
        "raw": generation / "graph_conditioned_api_dialogue_raw.jsonl",
        "rejects": generation / "graph_conditioned_api_dialogue_rejects.jsonl",
        "validated": generation / "graph_conditioned_api_dialogue_validated.jsonl",
        "clean": generation / "clean_samples.jsonl",
        "parents": run_dir / "reports" / "round_dataset_train_seed_rows.jsonl",
    }


def audit_run(run_dir: Path, *, expected_candidate_count: int | None = None) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    paths = source_paths(run_dir)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError(f"source run is incomplete; missing={missing}")
    source_hashes_before = {name: file_sha256(path) for name, path in paths.items()}
    raw_rows = read_jsonl(paths["raw"])
    parent_rows = read_jsonl(paths["parents"])
    validated_rows = read_jsonl(paths["validated"])
    clean_rows = read_jsonl(paths["clean"])
    reject_rows = read_jsonl(paths["rejects"])
    parents_by_id = parent_index(parent_rows)
    validated_by_slot = {
        int(row.get("generation_slot_id") or 0): row for row in validated_rows
    }
    clean_by_slot = {
        int(row.get("generation_slot_id") or 0): row for row in clean_rows
    }
    reject_line_by_attempt = {
        int(row.get("attempt")): index
        for index, row in enumerate(reject_rows, start=1)
        if str(row.get("attempt") if row.get("attempt") is not None else "").isdigit()
    }

    candidate_rows = [
        (line_number, row)
        for line_number, row in enumerate(raw_rows, start=1)
        if isinstance(row.get("api_stage64_final_qa_attempts"), list)
        and row.get("api_stage64_final_qa_attempts")
    ]
    if expected_candidate_count is not None and len(candidate_rows) != expected_candidate_count:
        raise ValueError(
            f"expected {expected_candidate_count} frozen final-QA candidates, found {len(candidate_rows)}"
        )

    matrix: list[dict[str, Any]] = []
    strict_clean_rows: list[dict[str, Any]] = []
    oracle_checks: list[dict[str, Any]] = []
    for raw_line_number, raw_record in candidate_rows:
        replay_error: str | None = None
        try:
            replay = replay_candidate(raw_record, run_dir, parents_by_id)
        except Exception as exc:  # fail-closed: retain a matrix row, never promote.
            replay_error = f"{type(exc).__name__}: {exc}"
            replay = {
                "pass": False,
                "validator_pass": False,
                "hash_match": False,
                "expected_hashes": {},
                "actual_hashes": {},
                "artifact_path": None,
                "artifact_sha256": None,
                "artifact_id": None,
                "parent_turn_count": None,
                "reconstruction": {},
                "evidence_dispersion_pass": False,
                "required_event_count": None,
                "aligned_event_count": None,
            }
        classification, reasons, truth_audit, repairs, selected_index = (
            classify_replayed_candidate(raw_record, replay)
        )
        target = raw_record.get("stage64_target") if isinstance(raw_record.get("stage64_target"), dict) else {}
        qas = qa_candidates(raw_record)
        row = {
            "schema_version": SCHEMA_VERSION,
            "source_run_dir": str(run_dir),
            "source_raw_line_number": raw_line_number,
            "source_reject_line_number": reject_line_by_attempt.get(int(raw_record.get("attempt") or 0)),
            "attempt": raw_record.get("attempt"),
            "generation_slot_id": raw_record.get("generation_slot_id"),
            "target_index": raw_record.get("target_index"),
            "seed_sample_id": raw_record.get("seed_sample_id"),
            "sample_id": raw_record.get("sample_id"),
            "originally_accepted": bool(raw_record.get("accepted")),
            "qa_operation": target.get("qa_operation"),
            "reasoning_template": target.get("reasoning_template"),
            "preferred_answer_type": target.get("preferred_answer_type"),
            "source_reject_error": reject_error_text(raw_record),
            "classification": classification,
            "classification_reason_codes": reasons,
            "truth_audit": truth_audit,
            "planned_representation_repairs": repairs,
            "selected_saved_qa_candidate_index": selected_index,
            "saved_qa_candidate_count": len(qas),
            "saved_qa_candidates": [compact_qa_candidate(item) for item in qas],
            "replay": {
                key: replay.get(key)
                for key in (
                    "pass",
                    "validator_pass",
                    "validator_reason",
                    "hash_match",
                    "expected_hashes",
                    "actual_hashes",
                    "artifact_path",
                    "artifact_sha256",
                    "artifact_id",
                    "parent_dialogue_source",
                    "parent_turn_count",
                    "reconstruction",
                    "evidence_dispersion_pass",
                    "required_event_count",
                    "aligned_event_count",
                )
            },
            "replay_error": replay_error,
            "api_calls_made": 0,
            "source_run_mutated": False,
        }

        if raw_record.get("accepted") is True and replay.get("pass"):
            slot = int(raw_record.get("generation_slot_id") or 0)
            validated = validated_by_slot.get(slot)
            clean = clean_by_slot.get(slot)
            if validated is None or clean is None:
                raise ValueError(f"accepted slot {slot} lacks validated/clean oracle")
            appended_exact = replay["appended_dialogue_marked"] == validated.get(
                "appended_dialogue_marked"
            )
            hashes_exact = replay["actual_hashes"] == validated.get("hashes")
            oracle_clean_exact = clean == validated.get("clean_sample")
            oracle_check = {
                "generation_slot_id": slot,
                "appended_dialogue_marked_exact": appended_exact,
                "frozen_hashes_exact": hashes_exact,
                "clean_row_exact": oracle_clean_exact,
                "pass": appended_exact and hashes_exact and oracle_clean_exact,
            }
            oracle_checks.append(oracle_check)
            if not oracle_check["pass"]:
                row["classification"] = CLASS_UNRECONSTRUCTABLE
                row["classification_reason_codes"] = ["accepted_oracle_cross_check_failed"]
            else:
                native.validate_final_clean_sample(
                    clean, replay["known_anchors"], replay["actual_hashes"]
                )
                strict_clean_rows.append(copy.deepcopy(clean))
        matrix.append(row)

    source_hashes_after = {name: file_sha256(path) for name, path in paths.items()}
    source_immutable = source_hashes_before == source_hashes_after
    if not source_immutable:
        raise RuntimeError("source run changed during offline audit")
    counts = Counter(row["classification"] for row in matrix)
    report = {
        "schema_version": SCHEMA_VERSION,
        "offline_only": True,
        "api_calls_made": 0,
        "source_run_dir": str(run_dir),
        "source_file_sha256_before": source_hashes_before,
        "source_file_sha256_after": source_hashes_after,
        "source_run_immutable": source_immutable,
        "raw_row_count": len(raw_rows),
        "frozen_final_qa_candidate_count": len(matrix),
        "original_accepted_count": sum(int(row["originally_accepted"]) for row in matrix),
        "classification_counts": dict(sorted(counts.items())),
        "replay_pass_count": sum(int(row["replay"]["pass"]) for row in matrix),
        "frozen_hash_triple_match_count": sum(int(row["replay"]["hash_match"]) for row in matrix),
        "dialogue_alignment_dispersion_pass_count": sum(
            int(row["replay"]["evidence_dispersion_pass"]) for row in matrix
        ),
        "accepted_oracle_checks": oracle_checks,
        "accepted_oracle_pass_count": sum(int(item["pass"]) for item in oracle_checks),
        "strict_clean_row_count": len(strict_clean_rows),
        "repairable_rows_are_not_clean": True,
        "truth_policy": (
            "Graph-only metadata is insufficient: every answer-defining relation, "
            "membership decision, status, or comparison operand must be uniquely "
            "supported by the frozen dialogue. Only truth-preserving, deterministic "
            "representation completion is marked repairable."
        ),
    }
    return {
        "matrix": matrix,
        "strict_clean_rows": strict_clean_rows,
        "report": report,
    }


def markdown_report(result: dict[str, Any], output_dir: Path) -> str:
    report = result["report"]
    counts = report["classification_counts"]
    lines = [
        "# Stage3 Fastlane v1：run175 离线 salvage 审计",
        "",
        "## 结论",
        "",
        f"- 冻结 final-QA 候选：{report['frozen_final_qa_candidate_count']}",
        f"- dialogue/graph/alignment 精确重放通过：{report['replay_pass_count']}",
        f"- 三项冻结哈希完全一致：{report['frozen_hash_triple_match_count']}",
        f"- 原 strict clean：{counts.get(CLASS_STRICT_CLEAN, 0)}",
        f"- 表示层可唯一修复：{counts.get(CLASS_REPAIRABLE, 0)}",
        f"- 需要语义决策/重写：{counts.get(CLASS_DIAGNOSTIC, 0)}",
        f"- 真值或目标合同失败：{counts.get(CLASS_INVALID_TRUTH, 0)}",
        f"- 不可重建：{counts.get(CLASS_UNRECONSTRUCTABLE, 0)}",
        "- API 调用：0",
        f"- run175 源文件保持不变：{report['source_run_immutable']}",
        "",
        "repairable 只是下一阶段的本地 compiler 输入，本阶段没有把任何拒绝样本写入 clean。",
        "",
        "## 逐条矩阵",
        "",
        "| raw 行 | attempt | operation | 分类 | 原因 |",
        "|---:|---:|---|---|---|",
    ]
    for row in result["matrix"]:
        reasons = ", ".join(row["classification_reason_codes"])
        lines.append(
            f"| {row['source_raw_line_number']} | {row['attempt']} | "
            f"{row['qa_operation']} | {row['classification']} | {reasons} |"
        )
    lines.extend(
        [
            "",
            "## 输出",
            "",
            f"- `{output_dir / 'salvage_matrix.jsonl'}`",
            f"- `{output_dir / 'strict_clean.jsonl'}`",
            f"- `{output_dir / 'repairable_candidates.jsonl'}`",
            f"- `{output_dir / 'diagnostic.jsonl'}`",
            f"- `{output_dir / 'invalid_truth.jsonl'}`",
            f"- `{output_dir / 'unreconstructable.jsonl'}`",
            f"- `{output_dir / 'report.json'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def write_result(result: dict[str, Any], output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise ValueError(f"output directory already exists: {output_dir}")
        if any(output_dir.iterdir()):
            raise ValueError(
                "--overwrite only permits an existing empty directory; "
                "refusing to replace prior audit artifacts"
            )
    else:
        output_dir.mkdir(parents=True)
    matrix = result["matrix"]
    write_jsonl(output_dir / "salvage_matrix.jsonl", matrix)
    write_jsonl(output_dir / "strict_clean.jsonl", result["strict_clean_rows"])
    class_outputs = {
        "repairable_candidates.jsonl": CLASS_REPAIRABLE,
        "diagnostic.jsonl": CLASS_DIAGNOSTIC,
        "invalid_truth.jsonl": CLASS_INVALID_TRUTH,
        "unreconstructable.jsonl": CLASS_UNRECONSTRUCTABLE,
    }
    for filename, classification in class_outputs.items():
        write_jsonl(
            output_dir / filename,
            [row for row in matrix if row["classification"] == classification],
        )
    write_json(output_dir / "report.json", result["report"])
    (output_dir / "report.md").write_text(
        markdown_report(result, output_dir), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-candidates", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit_run(
        args.run_dir,
        expected_candidate_count=args.expected_candidates,
    )
    write_result(result, args.output_dir.resolve(), overwrite=args.overwrite)
    print(json.dumps(result["report"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
