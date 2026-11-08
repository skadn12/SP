#!/usr/bin/env python3
"""Pure-local expansion of one accepted Stage3 dialogue into QA rows.

The dialogue, graph, alignment, and Claim IR are frozen before this module is
called.  This module only projects independently accepted sibling QAs into
ordinary one-QA training rows.  It deliberately does not run validators, call
an API, or choose between valid siblings.

Splitting is performed at the canonical parent-lineage level.  Consequently,
different dialogue families descended from the same parent lineage cannot be
split across train and eval after multi-QA expansion.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from typing import Any


SCHEMA_VERSION = "stage3_multiqa_rows_v1"
MARKED_ROW_SCHEMA_VERSION = "stage3_marked_qa_row_v1"
MIN_QAS_PER_DIALOGUE = 3
MAX_QAS_PER_DIALOGUE = 6
STAGE64_TARGET_IDENTITY_VERSION = "stage64_target_identity_v2"

QA_ALIAS_FIELDS = (
    "question_type",
    "question",
    "question_text",
    "options",
    "gold",
    "gold_answer_text",
    "normalized_gold_answer",
    "qa_operation",
    "qa_subtype",
    "reasoning_subtype",
    "reasoning_template",
    "hop_count",
    "answer_type",
    "answer_format",
    "hard_negative_strategy",
    "forbid_max_time_shortcut",
    "query_plan",
    "anchor_refs",
    "evidence_turn_ids",
)

TARGET_FIELDS = (
    "qa_operation",
    "qa_subtype",
    "reasoning_subtype",
    "reasoning_template",
    "hop_count",
    "hard_negative_strategy",
)


class MultiQARowExpansionError(ValueError):
    """Raised when the dialogue envelope itself is not safe to expand."""


class SiblingQARowError(ValueError):
    """Raised for a malformed sibling without rejecting its dialogue family."""


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _fold(value: Any) -> str:
    return _text(value).casefold()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _id(prefix: str, value: Any, *, length: int = 24) -> str:
    return f"{prefix}_{_sha256(value)[:length]}"


def assign_split(split_group: str, *, eval_per_mille: int = 100) -> str:
    """Assign a deterministic split from a parent-lineage split group."""

    if not 0 <= eval_per_mille <= 1000:
        raise ValueError("eval_per_mille must be in [0, 1000]")
    normalized = _text(split_group)
    if not normalized:
        raise ValueError("split_group is required")
    bucket = int(hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12], 16) % 1000
    return "eval" if bucket < eval_per_mille else "train"


def _clean_sample(envelope: Mapping[str, Any]) -> dict[str, Any]:
    nested = envelope.get("clean_sample")
    if isinstance(nested, Mapping):
        return copy.deepcopy(dict(nested))
    return copy.deepcopy(dict(envelope))


def _require_accepted_dialogue(sample: Mapping[str, Any]) -> None:
    native_contract = sample.get("native_contract")
    validation = sample.get("validation")
    if not isinstance(native_contract, Mapping) or native_contract.get("valid") is not True:
        raise MultiQARowExpansionError(
            "multi-QA rows require an accepted dialogue with native_contract.valid=true"
        )
    if not isinstance(validation, Mapping):
        raise MultiQARowExpansionError("accepted dialogue has no validation envelope")
    required_checks = (
        "graph_dialogue_consistent",
        "gold_derivable",
        "option_valid",
        "anti_shortcut_pass",
        "graph_validation_pass",
    )
    failed = [field for field in required_checks if validation.get(field) is not True]
    if failed:
        raise MultiQARowExpansionError(
            f"accepted dialogue failed required gates: {sorted(failed)}"
        )


def canonical_parent_lineage_id(sample: Mapping[str, Any]) -> str:
    """Return the earliest durable parent-lineage identity available."""

    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), Mapping) else {}
    lineage = (
        sample.get("parent_graph_lineage")
        if isinstance(sample.get("parent_graph_lineage"), Mapping)
        else {}
    )
    candidates = (
        sample.get("canonical_parent_lineage_id"),
        metadata.get("canonical_parent_lineage_id"),
        sample.get("lineage_group"),
        metadata.get("lineage_group"),
        lineage.get("canonical_parent_lineage_id"),
        lineage.get("lineage_group"),
        lineage.get("root_parent_graph_id"),
        lineage.get("parent_graph_id"),
        sample.get("parent_sample_id"),
        sample.get("source_id"),
    )
    result = next((_text(item) for item in candidates if _text(item)), "")
    if not result:
        raise MultiQARowExpansionError(
            "accepted dialogue has no canonical parent-lineage identity"
        )
    return result


def lineage_split_group(sample: Mapping[str, Any]) -> tuple[str, str]:
    canonical_id = canonical_parent_lineage_id(sample)
    return canonical_id, _id("parent_lineage", canonical_id)


def _coherent_hash(label: str, values: list[Any], *, required: bool = True) -> str:
    normalized = {_text(value) for value in values if _text(value)}
    if len(normalized) > 1:
        raise MultiQARowExpansionError(
            f"conflicting frozen {label} hashes in accepted envelope: {sorted(normalized)}"
        )
    if not normalized:
        if required:
            raise MultiQARowExpansionError(
                f"accepted envelope has no frozen {label} hash"
            )
        return ""
    return next(iter(normalized))


def frozen_hashes(
    envelope: Mapping[str, Any], sample: Mapping[str, Any], bundle: Mapping[str, Any]
) -> dict[str, str]:
    """Collect and cross-check hashes that no QA row may change."""

    marked_hashes = envelope.get("hashes") if isinstance(envelope.get("hashes"), Mapping) else {}
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), Mapping) else {}
    dialogue = sample.get("dialogue") if isinstance(sample.get("dialogue"), Mapping) else {}
    claim_ir = bundle.get("claim_ir") if isinstance(bundle.get("claim_ir"), Mapping) else {}
    return {
        "graph_snapshot_hash": _coherent_hash(
            "graph",
            [
                marked_hashes.get("graph_snapshot_hash"),
                sample.get("graph_snapshot_hash"),
                metadata.get("graph_snapshot_hash"),
                bundle.get("graph_snapshot_hash"),
            ],
        ),
        "dialogue_hash": _coherent_hash(
            "dialogue",
            [
                marked_hashes.get("dialogue_hash"),
                dialogue.get("dialogue_hash"),
                metadata.get("dialogue_hash"),
            ],
        ),
        "alignment_hash": _coherent_hash(
            "alignment",
            [marked_hashes.get("alignment_hash"), metadata.get("alignment_hash")],
        ),
        "claim_bundle_hash": _coherent_hash(
            "claim",
            [
                envelope.get("claim_bundle_hash"),
                sample.get("claim_bundle_hash"),
                bundle.get("claim_bundle_hash"),
                claim_ir.get("claim_bundle_hash"),
            ],
        ),
    }


def _qa_reports(
    qa: Mapping[str, Any], override: Mapping[str, Any] | None
) -> tuple[bool, dict[str, Any], list[str]]:
    """Resolve independent proof and semantic acceptance without running gates."""

    if override is not None:
        report = copy.deepcopy(dict(override))
        if report.get("accepted") is True:
            return True, report, []
        return False, report, [_text(report.get("reason") or "sibling_gate_rejected")]

    proof = qa.get("independent_validation_report")
    semantic = qa.get("independent_semantic_validation_report")
    if not isinstance(semantic, Mapping):
        semantic = qa.get("semantic_validation_report")
    reasons: list[str] = []
    if not isinstance(proof, Mapping):
        reasons.append("missing_independent_proof_report")
    elif proof.get("pass") is not True:
        reasons.append("independent_proof_gate_failed")
    if not isinstance(semantic, Mapping):
        reasons.append("missing_independent_semantic_report")
    elif semantic.get("pass") is not True:
        reasons.append("independent_semantic_gate_failed")
    report = {
        "accepted": not reasons,
        "proof_report": copy.deepcopy(dict(proof)) if isinstance(proof, Mapping) else None,
        "semantic_report": (
            copy.deepcopy(dict(semantic)) if isinstance(semantic, Mapping) else None
        ),
        "source": "embedded_independent_reports",
    }
    return not reasons, report, reasons


def _event_ids(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return list(dict.fromkeys(_text(item) for item in values if _text(item)))


def _evidence_turn_ids(qa: Mapping[str, Any]) -> list[str]:
    explicit = _event_ids(qa.get("evidence_turn_ids"))
    if explicit:
        return explicit
    result: list[str] = []
    for row in qa.get("evidence_ledger") or []:
        if not isinstance(row, Mapping):
            continue
        turn_id = _text(row.get("turn_id"))
        if turn_id and turn_id not in result:
            result.append(turn_id)
    return result


def _normalize_options(qa: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str], str]:
    raw_options = qa.get("options")
    raw_gold = qa.get("gold")
    gold_answer = _text(qa.get("gold_answer_text") or qa.get("gold_answer"))
    if not isinstance(raw_options, list) or len(raw_options) < 2:
        raise SiblingQARowError("QA must contain at least two options")
    if not isinstance(raw_gold, list) or len(raw_gold) != 1:
        raise SiblingQARowError("QA must contain exactly one gold option id")
    gold = [_text(raw_gold[0])]
    if not gold[0] or not gold_answer:
        raise SiblingQARowError("QA has no gold id or gold answer text")
    options: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw in raw_options:
        if not isinstance(raw, Mapping):
            raise SiblingQARowError("QA option is not an object")
        option = copy.deepcopy(dict(raw))
        option_id = _text(option.get("id") or option.get("option_id"))
        option_text = _text(option.get("text"))
        if not option_id or not option_text or option_id in seen_ids:
            raise SiblingQARowError("QA options require unique ids and non-empty text")
        seen_ids.add(option_id)
        option["id"] = option_id
        option["option_id"] = option_id
        is_gold = option_id == gold[0]
        option["is_gold"] = is_gold
        option["truth_value"] = is_gold
        options.append(option)
    if gold[0] not in seen_ids:
        raise SiblingQARowError("gold option id is absent from options")
    gold_text = next(option["text"] for option in options if option["id"] == gold[0])
    if _fold(gold_text) != _fold(gold_answer):
        raise SiblingQARowError("gold answer text conflicts with the gold option")
    return options, gold, gold_text


def _sync_query_plan(qa: dict[str, Any]) -> dict[str, Any]:
    raw = qa.get("query_plan")
    if not isinstance(raw, Mapping):
        raise SiblingQARowError("QA has no locally compiled query_plan")
    query_plan = copy.deepcopy(dict(raw))
    expected = {
        "qa_operation": qa["qa_operation"],
        "qa_subtype": qa["qa_subtype"],
        "reasoning_subtype": qa["reasoning_subtype"],
        "reasoning_template": qa["reasoning_template"],
        "hop_count": qa["hop_count"],
        "hard_negative_strategy": qa["hard_negative_strategy"],
        "answer_type": qa["answer_type"],
    }
    for field, value in expected.items():
        if field in query_plan and query_plan[field] != value:
            raise SiblingQARowError(
                f"query_plan.{field} conflicts with the frozen QA contract"
            )
        query_plan[field] = copy.deepcopy(value)
    gold_answer = qa["gold_answer_text"]
    for field, value in (
        ("gold_answer", gold_answer),
        ("normalized_gold_answer", _fold(gold_answer)),
        ("forbid_max_time_shortcut", qa["forbid_max_time_shortcut"]),
    ):
        if field in query_plan and query_plan[field] != value:
            raise SiblingQARowError(
                f"query_plan.{field} conflicts with the frozen QA contract"
            )
        query_plan[field] = copy.deepcopy(value)
    proof = qa.get("proof") if isinstance(qa.get("proof"), Mapping) else {}
    ref_sources = {
        "reasoning_anchor_path": "reasoning_path_event_instance_ids",
        "required_anchor_refs": "required_event_instance_ids",
        "supporting_anchor_refs": "required_event_instance_ids",
        "gold_anchor_refs": "answer_event_instance_ids",
        "dependency_closure_refs": "dependency_closure_event_instance_ids",
    }
    for query_field, proof_field in ref_sources.items():
        if query_field not in query_plan:
            values = _event_ids(proof.get(proof_field))
            if values:
                query_plan[query_field] = values
    return query_plan


def render_sibling_qa(raw_qa: Mapping[str, Any]) -> dict[str, Any]:
    """Render one accepted local QA contract into the legacy one-row shape."""

    qa = copy.deepcopy(dict(raw_qa))
    qa_id = _text(qa.get("qa_id"))
    question = _text(
        qa.get("question") or qa.get("question_text") or qa.get("question_text_draft")
    )
    required_text_fields = (
        "qa_operation",
        "qa_subtype",
        "reasoning_subtype",
        "reasoning_template",
        "answer_type",
        "hard_negative_strategy",
    )
    missing = [field for field in required_text_fields if not _text(qa.get(field))]
    if not qa_id or not question or missing:
        raise SiblingQARowError(
            f"QA lacks required frozen fields: {sorted(missing + (['qa_id'] if not qa_id else []) + (['question'] if not question else []))}"
        )
    hop_count = qa.get("hop_count")
    if not isinstance(hop_count, int) or isinstance(hop_count, bool) or hop_count < 1:
        raise SiblingQARowError("QA hop_count must be a positive integer")
    options, gold, gold_text = _normalize_options(qa)
    qa.update(
        {
            "qa_id": qa_id,
            "question_type": _text(qa.get("question_type") or "single_choice"),
            "question": question,
            "question_text": question,
            "question_surface_frozen": True,
            "answer_format": _text(qa.get("answer_format") or "single_choice_mcq"),
            "qa_operation": _text(qa.get("qa_operation")),
            "qa_subtype": _text(qa.get("qa_subtype")),
            "reasoning_subtype": _text(qa.get("reasoning_subtype")),
            "reasoning_template": _text(qa.get("reasoning_template")),
            "answer_type": _text(qa.get("answer_type")),
            "hard_negative_strategy": _text(qa.get("hard_negative_strategy")),
            "forbid_max_time_shortcut": qa.get("forbid_max_time_shortcut") is True,
            "options": options,
            "gold": gold,
            "gold_answer_text": gold_text,
            "normalized_gold_answer": _fold(gold_text),
            "evidence_turn_ids": _evidence_turn_ids(qa),
            "semantic_contract_frozen": True,
        }
    )
    qa["query_plan"] = _sync_query_plan(qa)
    if isinstance(qa.get("anchor_refs"), Mapping):
        anchor_refs = copy.deepcopy(dict(qa["anchor_refs"]))
    else:
        query = qa["query_plan"]
        anchor_refs = {
            "support": _event_ids(query.get("supporting_anchor_refs")),
            "gold": _event_ids(query.get("gold_anchor_refs")),
            "distractors": _event_ids(query.get("distractor_anchor_refs")),
        }
    qa["anchor_refs"] = anchor_refs
    return qa


def _target_for_qa(qa: Mapping[str, Any]) -> dict[str, Any]:
    target = {field: copy.deepcopy(qa[field]) for field in TARGET_FIELDS}
    target.update(
        {
            "preferred_answer_type": qa["answer_type"],
            "forbid_max_time_shortcut": qa["forbid_max_time_shortcut"],
            "target_basis": "truth_first_local_sibling_qa",
            "source_qa_id": qa["qa_id"],
        }
    )
    return target


def _stage64_target_identity(
    target_index: int,
    target: Mapping[str, Any],
    target_plan_sha256: str,
) -> str:
    return _sha256(
        {
            "schema_version": STAGE64_TARGET_IDENTITY_VERSION,
            "target_plan_sha256": target_plan_sha256,
            "target_index": int(target_index),
            "stage64_target": dict(target),
        }
    )


def _row_bundle(
    source_bundle: Mapping[str, Any],
    qa: Mapping[str, Any],
    *,
    accepted_qa_ids: list[str],
    rejected_qas: list[dict[str, Any]],
    target: Mapping[str, Any],
) -> dict[str, Any]:
    bundle = copy.deepcopy(dict(source_bundle))
    bundle.update(
        {
            "schema_version": _text(bundle.get("schema_version") or "stage3_claim_first_multiqa_v1"),
            "qas": [copy.deepcopy(dict(qa))],
            "qa_count": 1,
            "validator_pass_qa_count": 1,
            "semantic_validator_pass_qa_count": 1,
            "validator_reject_qa_count": 0,
            "semantic_validator_reject_qa_count": 0,
            "qa_reject_count": len(rejected_qas),
            "qa_rejects": copy.deepcopy(rejected_qas),
            "source_dialogue_accepted_qa_ids": list(accepted_qa_ids),
            "source_dialogue_accepted_qa_count": len(accepted_qa_ids),
            "active_qa_id": qa["qa_id"],
            "target_fulfillment_qa_id": qa["qa_id"],
            "target_fulfillment_status": "row_target_exact_match",
            "target_fulfillment_pass": True,
            "immutable_stage64_target": copy.deepcopy(dict(target)),
            "multiqa_rows_expanded": True,
        }
    )
    return bundle


def _build_clean_row(
    source_sample: Mapping[str, Any],
    source_bundle: Mapping[str, Any],
    qa: Mapping[str, Any],
    *,
    source_sample_id: str,
    family_id: str,
    canonical_lineage_id: str,
    split_group: str,
    split: str,
    accepted_qa_ids: list[str],
    rejected_qas: list[dict[str, Any]],
    hashes: Mapping[str, str],
    row_index: int,
) -> dict[str, Any]:
    row = copy.deepcopy(dict(source_sample))
    source_target = (
        source_sample.get("stage64_target")
        if isinstance(source_sample.get("stage64_target"), Mapping)
        else None
    )
    preserve_source_target = qa.get("qa_role") == "primary" and source_target is not None
    target = (
        copy.deepcopy(dict(source_target))
        if preserve_source_target
        else _target_for_qa(qa)
    )
    for field in TARGET_FIELDS:
        target_field = "preferred_answer_type" if field == "answer_type" else field
        if target_field in target and target[target_field] != qa[field]:
            raise SiblingQARowError(
                f"stage64_target.{target_field} conflicts with frozen QA semantics"
            )
    if target.get("preferred_answer_type") != qa["answer_type"]:
        raise SiblingQARowError(
            "stage64_target.preferred_answer_type conflicts with frozen QA semantics"
        )
    rendered_qa = copy.deepcopy(dict(qa))
    rendered_qa.update(
        {
            "immutable_stage64_target": copy.deepcopy(target),
            "operation_specific_target": copy.deepcopy(target),
            "fulfills_immutable_stage64_target": True,
            "target_fulfillment_match": True,
        }
    )
    if preserve_source_target:
        target_index = source_sample.get("target_index")
        target_plan_sha256 = _text(
            source_sample.get("stage64_target_plan_sha256")
        )
        target_id = _text(source_sample.get("stage64_target_identity"))
        if (
            not isinstance(target_index, int)
            or isinstance(target_index, bool)
            or target_index < 0
            or not target_plan_sha256
            or not target_id
            or target_id != _text(source_sample.get("stage64_target_id"))
            or target_id
            != _stage64_target_identity(target_index, target, target_plan_sha256)
        ):
            raise SiblingQARowError(
                "primary QA cannot preserve invalid source Stage64 target provenance"
            )
    else:
        target_index = row_index
        target_plan_sha256 = _sha256(
            {
                "schema_version": "stage3_truth_first_qa_target_plan_v1",
                "source_dialogue_sample_id": source_sample_id,
                "dialogue_family_id": family_id,
                "qa_id": qa["qa_id"],
                "stage64_target": target,
            }
        )
        target_id = _stage64_target_identity(
            target_index, target, target_plan_sha256
        )
    row_id = _id(
        "stage3_qa_row",
        {
            "source_sample_id": source_sample_id,
            "dialogue_family_id": family_id,
            "qa_id": qa["qa_id"],
            "dialogue_hash": hashes["dialogue_hash"],
        },
    )
    row_bundle = _row_bundle(
        source_bundle,
        rendered_qa,
        accepted_qa_ids=accepted_qa_ids,
        rejected_qas=rejected_qas,
        target=target,
    )
    row.update(
        {
            "sample_id": row_id,
            "source_dialogue_sample_id": source_sample_id,
            "source_dialogue_qa_count": len(accepted_qa_ids),
            "qa_row_index": row_index,
            "qa_row_schema_version": SCHEMA_VERSION,
            "qa_id": rendered_qa["qa_id"],
            "qa": copy.deepcopy(rendered_qa),
            "dialogue_family_id": family_id,
            "canonical_parent_lineage_id": canonical_lineage_id,
            "split_group": split_group,
            "split_group_basis": "canonical_parent_lineage",
            "split": split,
            "dialogue_family_split": split,
            "stage64_target": target,
            "stage64_target_id": target_id,
            "stage64_target_identity": target_id,
            "stage64_target_plan_sha256": target_plan_sha256,
            "target_index": target_index,
            "source_dialogue_stage64_target_identity": source_sample.get(
                "stage64_target_identity"
            ),
            "source_dialogue_target_preserved": preserve_source_target,
            "claim_first_multiqa_bundle": row_bundle,
            "claim_first_primary_qa_id": rendered_qa["qa_id"],
            "claim_first_target_fulfillment_qa_id": rendered_qa["qa_id"],
            "claim_first_target_fulfillment_pass": True,
            "claim_first_qa_count": 1,
            "frozen_source_hashes": copy.deepcopy(dict(hashes)),
        }
    )
    for field in QA_ALIAS_FIELDS:
        row[field] = copy.deepcopy(rendered_qa[field])
    row["gold_answer"] = rendered_qa["gold_answer_text"]
    row["gold_answers"] = [rendered_qa["gold_answer_text"]]
    row["gold_options"] = copy.deepcopy(rendered_qa["gold"])
    row["gold_option"] = rendered_qa["gold"][0]
    metadata = copy.deepcopy(dict(row.get("metadata") or {}))
    metadata.update(
        {
            "dialogue_family_id": family_id,
            "canonical_parent_lineage_id": canonical_lineage_id,
            "split_group": split_group,
            "dialogue_family_split": split,
            "multiqa_row_expanded": True,
            "multiqa_active_qa_id": rendered_qa["qa_id"],
            "stage64_target_index": target_index,
            "stage64_target_identity": target_id,
            "stage64_target_id": target_id,
            "stage64_target_plan_sha256": target_plan_sha256,
            "qa_operation": rendered_qa["qa_operation"],
            "attack_skill": rendered_qa["qa_operation"],
        }
    )
    row["metadata"] = metadata
    row["attack_skill"] = rendered_qa["qa_operation"]
    attack_plan = copy.deepcopy(dict(row.get("attack_plan") or {}))
    attack_plan["attack_skill"] = rendered_qa["qa_operation"]
    attack_plan["qa_operation"] = rendered_qa["qa_operation"]
    row["attack_plan"] = attack_plan
    row["difficulty_cell"] = (
        f"{rendered_qa['question_type']}|{rendered_qa['qa_operation']}"
    )
    return row


def _marked_row(
    envelope: Mapping[str, Any],
    clean_row: Mapping[str, Any],
    *,
    family_id: str,
    canonical_lineage_id: str,
    split_group: str,
    split: str,
    hashes: Mapping[str, str],
) -> dict[str, Any]:
    marked = copy.deepcopy(dict(envelope))
    qa = clean_row["qa"]
    marked.update(
        {
            "schema_version": MARKED_ROW_SCHEMA_VERSION,
            "clean_sample": copy.deepcopy(dict(clean_row)),
            "qa_row_id": clean_row["sample_id"],
            "qa_id": qa["qa_id"],
            "dialogue_family_id": family_id,
            "canonical_parent_lineage_id": canonical_lineage_id,
            "split_group": split_group,
            "split": split,
            "dialogue_family_split": split,
            "stage64_target": copy.deepcopy(clean_row["stage64_target"]),
            "stage64_target_id": clean_row["stage64_target_id"],
            "stage64_target_identity": clean_row["stage64_target_identity"],
            "stage64_target_plan_sha256": clean_row["stage64_target_plan_sha256"],
            "target_index": clean_row["target_index"],
            "claim_first_multiqa_bundle": copy.deepcopy(
                clean_row["claim_first_multiqa_bundle"]
            ),
            "claim_bundle_hash": hashes["claim_bundle_hash"],
            "qa_row_bundle": {
                "schema_version": MARKED_ROW_SCHEMA_VERSION,
                "qa_row_id": clean_row["sample_id"],
                "qa_id": qa["qa_id"],
                "source_dialogue_sample_id": clean_row["source_dialogue_sample_id"],
                "dialogue_family_id": family_id,
                "canonical_parent_lineage_id": canonical_lineage_id,
                "split_group": split_group,
                "split": split,
                "frozen_source_hashes": copy.deepcopy(dict(hashes)),
            },
        }
    )
    # Keep the original audit hash envelope byte-for-byte.  QA-row identity has
    # its own fields and must never masquerade as a new graph/dialogue snapshot.
    if isinstance(envelope.get("hashes"), Mapping):
        marked["hashes"] = copy.deepcopy(dict(envelope["hashes"]))
    return marked


def expand_accepted_dialogue(
    envelope: Mapping[str, Any],
    *,
    sibling_results: Mapping[str, Mapping[str, Any]] | None = None,
    min_qas: int = MIN_QAS_PER_DIALOGUE,
    max_qas: int = MAX_QAS_PER_DIALOGUE,
    eval_per_mille: int = 100,
) -> dict[str, Any]:
    """Expand independently accepted sibling QAs into marked training rows.

    A bad sibling is recorded in ``qa_rejects`` and does not remove the frozen
    dialogue or any other valid sibling.  Dataset rows are emitted only when
    the family still contains ``min_qas..max_qas`` accepted, renderable QAs.
    """

    if min_qas < 1 or max_qas < min_qas:
        raise ValueError("invalid min_qas/max_qas")
    if not isinstance(envelope, Mapping):
        raise TypeError("envelope must be a mapping")
    source_fingerprint = _sha256(envelope)
    sample = _clean_sample(envelope)
    _require_accepted_dialogue(sample)
    bundle = sample.get("claim_first_multiqa_bundle")
    if not isinstance(bundle, Mapping):
        raise MultiQARowExpansionError("accepted dialogue has no multi-QA bundle")
    qas = bundle.get("qas")
    if not isinstance(qas, list):
        raise MultiQARowExpansionError("multi-QA bundle has no sibling QA list")
    family_id = _text(bundle.get("dialogue_family_id") or sample.get("dialogue_family_id"))
    if not family_id:
        raise MultiQARowExpansionError("accepted dialogue has no dialogue_family_id")
    if sample.get("dialogue_family_id") and _text(sample.get("dialogue_family_id")) != family_id:
        raise MultiQARowExpansionError("conflicting dialogue_family_id values")
    canonical_lineage_id, split_group = lineage_split_group(sample)
    split = assign_split(split_group, eval_per_mille=eval_per_mille)
    hashes = frozen_hashes(envelope, sample, bundle)
    source_sample_id = _text(sample.get("sample_id"))
    if not source_sample_id:
        raise MultiQARowExpansionError("accepted dialogue has no sample_id")

    qa_id_counts = Counter(
        _text(item.get("qa_id"))
        for item in qas
        if isinstance(item, Mapping) and _text(item.get("qa_id"))
    )
    rejected: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    acceptance_by_id: dict[str, dict[str, Any]] = {}
    for raw_qa in qas:
        if not isinstance(raw_qa, Mapping):
            rejected.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "qa_id": None,
                    "reason_codes": ["non_object_sibling_qa"],
                    "dialogue_retained": True,
                    "sibling_qas_retained": True,
                }
            )
            continue
        qa_id = _text(raw_qa.get("qa_id"))
        if not qa_id:
            rejected.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "qa_id": None,
                    "reason_codes": ["missing_qa_id"],
                    "qa": copy.deepcopy(dict(raw_qa)),
                    "dialogue_retained": True,
                    "sibling_qas_retained": True,
                }
            )
            continue
        if qa_id_counts[qa_id] > 1:
            rejected.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "qa_id": qa_id,
                    "reason_codes": ["duplicate_qa_id"],
                    "qa": copy.deepcopy(dict(raw_qa)),
                    "dialogue_retained": True,
                    "sibling_qas_retained": True,
                }
            )
            continue
        override = sibling_results.get(qa_id) if sibling_results is not None else None
        gate_pass, gate_report, reason_codes = _qa_reports(raw_qa, override)
        acceptance_by_id[qa_id] = gate_report
        if not gate_pass:
            rejected.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "qa_id": qa_id,
                    "reason_codes": reason_codes,
                    "validation_report": gate_report,
                    "qa": copy.deepcopy(dict(raw_qa)),
                    "dialogue_retained": True,
                    "sibling_qas_retained": True,
                }
            )
            continue
        try:
            rendered = render_sibling_qa(raw_qa)
        except SiblingQARowError as exc:
            rejected.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "qa_id": qa_id,
                    "reason_codes": ["qa_row_render_failed"],
                    "detail": str(exc),
                    "validation_report": gate_report,
                    "qa": copy.deepcopy(dict(raw_qa)),
                    "dialogue_retained": True,
                    "sibling_qas_retained": True,
                }
            )
            continue
        accepted.append(rendered)

    accepted.sort(key=lambda item: item["qa_id"])
    accepted_ids = [item["qa_id"] for item in accepted]
    family_ready = min_qas <= len(accepted) <= max_qas
    clean_rows: list[dict[str, Any]] = []
    marked_rows: list[dict[str, Any]] = []
    if family_ready:
        for index, qa in enumerate(accepted):
            qa["dialogue_family_id"] = family_id
            qa["canonical_parent_lineage_id"] = canonical_lineage_id
            qa["split_group"] = split_group
            qa["dialogue_family_split"] = split
            row = _build_clean_row(
                sample,
                bundle,
                qa,
                source_sample_id=source_sample_id,
                family_id=family_id,
                canonical_lineage_id=canonical_lineage_id,
                split_group=split_group,
                split=split,
                accepted_qa_ids=accepted_ids,
                rejected_qas=rejected,
                hashes=hashes,
                row_index=index,
            )
            clean_rows.append(row)
            marked_rows.append(
                _marked_row(
                    envelope,
                    row,
                    family_id=family_id,
                    canonical_lineage_id=canonical_lineage_id,
                    split_group=split_group,
                    split=split,
                    hashes=hashes,
                )
            )

    if _sha256(envelope) != source_fingerprint:
        raise RuntimeError("multi-QA expansion mutated the source envelope")
    row_ids = [row["sample_id"] for row in clean_rows]
    if len(row_ids) != len(set(row_ids)):
        raise RuntimeError("multi-QA expansion produced duplicate row ids")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "source_dialogue_sample_id": source_sample_id,
        "dialogue_family_id": family_id,
        "canonical_parent_lineage_id": canonical_lineage_id,
        "split_group": split_group,
        "split_group_basis": "canonical_parent_lineage",
        "split": split,
        "input_sibling_qa_count": len(qas),
        "accepted_sibling_qa_count": len(accepted),
        "rejected_sibling_qa_count": len(rejected),
        "expanded_qa_row_count": len(clean_rows),
        "minimum_required_qa_count": min_qas,
        "maximum_allowed_qa_count": max_qas,
        "family_ready": family_ready,
        "dialogue_retained": True,
        "valid_siblings_retained": True,
        "frozen_hashes_preserved": True,
        "source_envelope_unchanged": True,
        "accepted_qa_ids": accepted_ids,
        "rejected_qa_ids": [row.get("qa_id") for row in rejected],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "family_ready": family_ready,
        "dialogue_family_id": family_id,
        "canonical_parent_lineage_id": canonical_lineage_id,
        "split_group": split_group,
        "split": split,
        "frozen_hashes": hashes,
        "retained_accepted_qas": copy.deepcopy(accepted),
        "qa_rejects": rejected,
        "qa_rows": clean_rows,
        "marked_qa_rows": marked_rows,
        "acceptance_reports_by_qa_id": acceptance_by_id,
        "summary": summary,
    }


__all__ = [
    "MARKED_ROW_SCHEMA_VERSION",
    "MAX_QAS_PER_DIALOGUE",
    "MIN_QAS_PER_DIALOGUE",
    "MultiQARowExpansionError",
    "SCHEMA_VERSION",
    "SiblingQARowError",
    "assign_split",
    "canonical_parent_lineage_id",
    "expand_accepted_dialogue",
    "frozen_hashes",
    "lineage_split_group",
    "render_sibling_qa",
]
