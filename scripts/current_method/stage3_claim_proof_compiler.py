#!/usr/bin/env python3
"""Deterministic Claim-first proof and multi-QA compiler for Stage3.

This module deliberately sits between the frozen graph/claim bundle and any
language model.  It compiles answer truth, evidence paths and operation-local
targets without changing the graph snapshot.  The drafts emitted here are
contracts, not final training questions; an API may later naturalize only the
surface fields while all ``frozen_*`` fields remain byte-for-byte stable.

The compiler is intentionally independent from the Stage64 QA normalizer.  A
single graph may therefore produce several operation-specific QA contracts
without pretending that the graph itself has several operations.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import re
from collections import defaultdict
from typing import Any, Iterable


SCHEMA_VERSION = "stage3_claim_first_multiqa_v1"
PROOF_SCHEMA_VERSION = "stage3_claim_proof_v1"
ACTIVE_STATUSES = {"active", "restored", "completed", "confirmed"}
STAGE64_CURRENT_STATUSES = {"active", "restored"}
IMMUTABLE_TARGET_FIELDS = (
    "reasoning_subtype",
    "reasoning_template",
    "hop_count",
    "hard_negative_strategy",
)
TEMPORAL_HARD_NEGATIVE_STATUSES = {
    "same_slot_canceled_value": {"canceled"},
    "same_slot_outdated_value": {"outdated"},
    "same_slot_report_or_belief_value": {"report", "belief"},
}

# Independent, fail-closed mirror of the immutable Stage64 reasoning signatures.
# The compiler does not import the native generator; a cross-module test guards
# against drift between these two independently maintained contracts.
IMMUTABLE_TARGET_SIGNATURES: dict[str, dict[str, tuple[str, str, int, bool]]] = {
    "temporal_state_update": {
        "same_slot_canceled_value": (
            "status_gated_canceled_update",
            "reject_canceled_value_then_resolve_update",
            4,
            True,
        ),
        "same_slot_outdated_value": (
            "status_gated_outdated_update",
            "reject_outdated_value_then_resolve_update",
            4,
            True,
        ),
        "same_slot_report_or_belief_value": (
            "epistemic_report_vs_current_update",
            "reject_report_or_belief_then_resolve_update",
            4,
            True,
        ),
    },
    "bridge": {
        "one_hop_preserving_alternative": (
            "conjunctive_two_support_bridge", "join_two_support_facts", 2, False
        ),
        "intermediate_entity_swap": (
            "dependency_chain_bridge", "follow_intermediate_dependency", 2, False
        ),
        "correct_hop_wrong_scope": (
            "scope_constrained_bridge", "join_support_under_scope_constraint", 2, False
        ),
    },
    "comparison": {
        "near_numeric_off_by_one": (
            "same_slot_absolute_difference", "bind_pair_then_compute_difference", 2, False
        ),
        "wrong_time_same_slot_pair": (
            "time_qualified_comparison", "select_time_matched_pair_then_compare", 2, False
        ),
        "swapped_entity_close_value": (
            "entity_bound_comparison", "bind_entities_then_compare_values", 2, False
        ),
    },
    "visibility": {
        "correct_fact_wrong_observer": (
            "observer_permission_scope", "resolve_observer_permission_and_scope", 2, False
        ),
        "active_but_wrong_scope": (
            "scope_intersection_visibility", "intersect_resource_and_group_scope", 2, False
        ),
        "right_observer_wrong_condition": (
            "conditional_access_visibility", "check_condition_then_permission", 2, False
        ),
    },
    "count": {
        "include_exclude_boundary_count": (
            "filtered_membership_count", "filter_members_then_count", 2, False
        ),
        "off_by_one_excluded_member": (
            "exception_aware_count", "apply_exclusions_then_count", 2, False
        ),
        "wrong_scope_near_count": (
            "scope_filtered_count", "select_scope_then_filter_and_count", 2, False
        ),
    },
    "cause_consequence": {
        "temporally_adjacent_non_cause": (
            "direct_causal_dependency", "trace_cause_to_effect", 2, False
        ),
        "earlier_but_noncausal_event": (
            "causal_vs_temporal_sequence", "disambiguate_dependency_from_sequence", 2, False
        ),
        "right_cause_wrong_effect": (
            "consequence_attribution", "trace_dependency_to_unique_effect", 2, False
        ),
    },
    "relation_lookup": {
        "same_relation_wrong_entity": (
            "entity_constrained_relation", "bind_entity_then_relation", 2, False
        ),
        "same_entity_wrong_slot": (
            "dual_constraint_relation", "intersect_entity_and_relation_constraints", 2, False
        ),
        "same_slot_wrong_value": (
            "same_slot_relation_disambiguation", "resolve_same_slot_competitors", 2, False
        ),
    },
}

ANSWER_PROJECTION_BY_OPERATION = {
    "temporal_state_update": "current_value",
    "bridge": "binder_joined_support_value",
    "comparison": "absolute_difference",
    "visibility": "visible_value_under_scope",
    "count": "included_member_count",
    "cause_consequence": "consequence_value",
    "relation_lookup": "bound_relation_value",
}


class ClaimProofCompileError(ValueError):
    """Raised when a frozen claim bundle cannot yield an unambiguous proof."""


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _fold(value: Any) -> str:
    return _text(value).casefold()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(prefix: str, value: Any, length: int = 24) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:length]}"


def _claim_id(claim: dict[str, Any]) -> str:
    return _text(claim.get("claim_id"))


def _claim_type(claim: dict[str, Any]) -> str:
    return _fold(claim.get("claim_type") or claim.get("node_type"))


def _is_node_claim(claim: dict[str, Any]) -> bool:
    kind = _claim_type(claim)
    return kind in {"node", "node_claim", "node_fact", "event", "event_claim"}


def _claims(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    raw = bundle.get("claims")
    if not isinstance(raw, list):
        raise ClaimProofCompileError("claim bundle has no claims list")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ClaimProofCompileError("claim bundle contains a non-object claim")
        claim_id = _claim_id(item)
        if not claim_id or claim_id in seen:
            raise ClaimProofCompileError(f"missing or duplicate claim_id: {claim_id!r}")
        seen.add(claim_id)
        result.append(item)
    return result


def claim_event_mapping(
    bundle: dict[str, Any], claim_ids: Iterable[str]
) -> dict[str, str]:
    """Return a lossless claim -> subject-event mapping for a proof path.

    A legacy query plan addresses graph events while Claim-first truth addresses
    claims.  Silently dropping an unmapped claim (or allowing two proof claims
    to collapse onto one event) changes the proof, so both cases are hard
    compiler errors.  Callers get insertion order matching ``claim_ids``.
    """

    ordered_claim_ids = list(dict.fromkeys(_text(item) for item in claim_ids))
    if any(not claim_id for claim_id in ordered_claim_ids):
        raise ClaimProofCompileError("proof path contains an empty claim_id")
    claims_by_id = {_claim_id(item): item for item in _claims(bundle)}
    result: dict[str, str] = {}
    event_to_claim: dict[str, str] = {}
    for claim_id in ordered_claim_ids:
        claim = claims_by_id.get(claim_id)
        if claim is None:
            raise ClaimProofCompileError(
                f"proof path references unknown claim_id: {claim_id}"
            )
        event_id = _text(claim.get("subject_event_id"))
        if not event_id:
            raise ClaimProofCompileError(
                f"claim lacks subject_event_id: {claim_id}"
            )
        previous_claim_id = event_to_claim.get(event_id)
        if previous_claim_id is not None and previous_claim_id != claim_id:
            raise ClaimProofCompileError(
                "claim-to-event mapping is not one-to-one: "
                f"{previous_claim_id} and {claim_id} both map to {event_id}"
            )
        result[claim_id] = event_id
        event_to_claim[event_id] = claim_id
    if len(result) != len(event_to_claim):  # defensive; the branch above owns this.
        raise ClaimProofCompileError("claim-to-event mapping is not bijective")
    return result


def _ledger_by_claim(bundle: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Accept both the canonical list ledger and the temporary mapping view."""

    raw = bundle.get("support_ledger") or bundle.get("claim_support_ledger") or []
    result: dict[str, list[dict[str, str]]] = defaultdict(list)
    if isinstance(raw, dict) and isinstance(raw.get("entries"), list):
        iterable = raw["entries"]
    elif isinstance(raw, dict):
        iterable: list[Any] = []
        for claim_id, value in raw.items():
            values = value if isinstance(value, list) else [value]
            for item in values:
                if isinstance(item, dict):
                    merged = copy.deepcopy(item)
                    merged.setdefault("claim_id", claim_id)
                    iterable.append(merged)
    elif isinstance(raw, list):
        iterable = raw
    else:
        raise ClaimProofCompileError("support ledger must be a list or object")
    for item in iterable:
        if not isinstance(item, dict):
            continue
        claim_id = _text(item.get("claim_id"))
        turn_id = _text(item.get("turn_id"))
        quote = _text(item.get("exact_support_quote") or item.get("support_quote"))
        if claim_id and turn_id and quote:
            entry = {
                "claim_id": claim_id,
                "turn_id": turn_id,
                "exact_support_quote": quote,
            }
            if entry not in result[claim_id]:
                result[claim_id].append(entry)
    return dict(result)


def _eligible_nodes(
    bundle: dict[str, Any], *, require_support: bool
) -> list[dict[str, Any]]:
    ledger = _ledger_by_claim(bundle)
    result = []
    for claim in _claims(bundle):
        if require_support:
            eligible = claim.get("qa_eligible") is True
        else:
            semantic_candidate = claim.get("qa_candidate")
            if semantic_candidate is None:
                semantic_candidate = claim.get("qa_eligible")
            eligible = bool(semantic_candidate) and _fold(
                claim.get("edge_role")
            ) == "semantic"
        if not _is_node_claim(claim) or not eligible:
            continue
        if require_support and _claim_id(claim) not in ledger:
            continue
        if not all(_text(claim.get(key)) for key in ("entity", "slot")):
            continue
        if claim.get("value") is None or not _text(claim.get("value")):
            continue
        result.append(claim)
    return sorted(
        result,
        key=lambda item: (
            int(item.get("global_time_index") or item.get("time_index") or 0),
            _claim_id(item),
        ),
    )


def _bundle_hash(bundle: dict[str, Any]) -> str:
    for key in ("claim_bundle_hash", "frozen_claims_hash", "frozen_hash"):
        value = _text(bundle.get(key))
        if value:
            return value
    # Compatibility only. Production bundles should always carry the frozen
    # hash generated by stage3_claim_ir.
    return _hash("cb", bundle)


def _source_graph_operation(bundle: dict[str, Any]) -> str:
    plan = bundle.get("current_event_plan")
    plan_operation = plan.get("operation") if isinstance(plan, dict) else None
    return _text(
        bundle.get("source_graph_operation")
        or bundle.get("operation")
        or plan_operation
    )


def dialogue_family_id(
    bundle: dict[str, Any],
    *,
    seed_sample_id: str = "",
    lineage_group_id: str = "",
) -> str:
    """Return a split-safe family id shared by every QA from this lineage graph."""

    graph_hash = _text(bundle.get("graph_snapshot_hash"))
    if not graph_hash:
        raise ClaimProofCompileError("claim bundle has no graph_snapshot_hash")
    return _hash(
        "family",
        {
            "graph_snapshot_hash": graph_hash,
            "seed_sample_id": _text(seed_sample_id),
            "lineage_group_id": _text(lineage_group_id),
        },
    )


def assign_family_split(family_id: str, *, eval_per_mille: int = 100) -> str:
    """Deterministically split at family level, never at individual-QA level."""

    if not 0 <= eval_per_mille <= 1000:
        raise ValueError("eval_per_mille must be in [0, 1000]")
    bucket = int(hashlib.sha256(family_id.encode("utf-8")).hexdigest()[:12], 16) % 1000
    return "eval" if bucket < eval_per_mille else "train"


def _slot_label(slot: Any) -> str:
    return _text(slot).replace("_", " ")


def _status(claim: dict[str, Any]) -> str:
    return _fold(claim.get("status") or "active")


def _support_subset(
    ledger: dict[str, list[dict[str, str]]], claim_ids: Iterable[str]
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for claim_id in claim_ids:
        rows = ledger.get(claim_id) or []
        if not rows:
            raise ClaimProofCompileError(f"required claim lacks exact support: {claim_id}")
        result.extend(copy.deepcopy(rows))
    return result


def _option_pool(nodes: list[dict[str, Any]], gold_text: str) -> list[dict[str, Any]]:
    """Choose graph-grounded unique distractors before synthetic fallbacks."""

    seen = {_fold(gold_text)}
    result: list[dict[str, Any]] = []
    for claim in nodes:
        value = _text(claim.get("value"))
        if not value or _fold(value) in seen:
            continue
        seen.add(_fold(value))
        result.append(
            {
                "text": value,
                "truth_value": False,
                "source_claim_id": _claim_id(claim),
                "distractor_origin": "frozen_graph_node",
            }
        )
    return result


def _ordered_options(
    *,
    qa_seed: Any,
    gold_text: str,
    gold_claim_id: str | None,
    distractors: list[dict[str, Any]],
    desired_count: int = 4,
) -> tuple[list[dict[str, Any]], list[str]]:
    unique: list[dict[str, Any]] = []
    seen = {_fold(gold_text)}
    for item in distractors:
        text = _text(item.get("text"))
        if text and _fold(text) not in seen:
            seen.add(_fold(text))
            unique.append(copy.deepcopy(item))
        if len(unique) >= desired_count - 1:
            break
    if len(unique) < desired_count - 1:
        raise ClaimProofCompileError("fewer than three unique distractors")
    options = [
        {
            "text": gold_text,
            "truth_value": True,
            "source_claim_id": gold_claim_id,
            "distractor_origin": "gold_claim" if gold_claim_id else "local_derivation",
        },
        *unique,
    ]
    # Stable per-QA rotation avoids making the gold position a global template.
    rotation = int(hashlib.sha256(_canonical_json(qa_seed).encode()).hexdigest()[:8], 16) % len(options)
    options = options[rotation:] + options[:rotation]
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for index, item in enumerate(options):
        item["id"] = labels[index]
    gold = [item["id"] for item in options if item["truth_value"] is True]
    return options, gold


def _proof_and_qa(
    *,
    bundle: dict[str, Any],
    family_id: str,
    source_graph_operation: str,
    operation: str,
    subtype: str,
    template: str,
    hop_count: int,
    answer_projection: str,
    question_draft: str,
    answer_type: str,
    gold_answer_text: str,
    answer_claim_ids: list[str],
    required_claim_ids: list[str],
    reasoning_claim_ids: list[str],
    dependency_closure_claim_ids: list[str],
    derivation: dict[str, Any],
    options: list[dict[str, Any]],
    gold: list[str],
    ledger: dict[str, list[dict[str, str]]],
    require_support: bool,
) -> dict[str, Any]:
    frozen_hash = _bundle_hash(bundle)
    graph_hash = _text(bundle.get("graph_snapshot_hash"))
    required_claim_ids = list(dict.fromkeys(required_claim_ids))
    reasoning_claim_ids = list(dict.fromkeys(reasoning_claim_ids))
    answer_claim_ids = list(dict.fromkeys(answer_claim_ids))
    dependency_closure_claim_ids = list(
        dict.fromkeys(dependency_closure_claim_ids)
    )
    operand_claim_ids = list(
        dict.fromkeys(derivation.get("operand_claim_ids") or [])
    )
    distractor_claim_ids = list(
        dict.fromkeys(
            _text(item.get("source_claim_id"))
            for item in options
            if item.get("truth_value") is False
            and _text(item.get("source_claim_id"))
        )
    )
    mapped_claim_ids = list(
        dict.fromkeys(
            required_claim_ids
            + reasoning_claim_ids
            + answer_claim_ids
            + dependency_closure_claim_ids
            + operand_claim_ids
            + distractor_claim_ids
        )
    )
    claim_event_map = claim_event_mapping(bundle, mapped_claim_ids)
    event_claim_map = {event_id: claim_id for claim_id, event_id in claim_event_map.items()}

    def event_ids(claim_ids: Iterable[str]) -> list[str]:
        result: list[str] = []
        for claim_id in dict.fromkeys(_text(item) for item in claim_ids):
            if not claim_id:
                raise ClaimProofCompileError("proof path contains an empty claim_id")
            event_id = claim_event_map.get(claim_id)
            if not event_id:
                raise ClaimProofCompileError(
                    f"proof claim is absent from lossless event mapping: {claim_id}"
                )
            result.append(event_id)
        return result

    required_event_ids = event_ids(required_claim_ids)
    reasoning_event_ids = event_ids(reasoning_claim_ids)
    answer_event_ids = event_ids(answer_claim_ids)
    distractor_event_ids = event_ids(distractor_claim_ids)
    dependency_event_ids = event_ids(dependency_closure_claim_ids)
    operand_event_ids = event_ids(operand_claim_ids)
    # A locally derived scalar (difference/count) has no node claim of its own,
    # but its gold still retains the complete graph evidence projection.
    gold_reference_event_ids = answer_event_ids or operand_event_ids
    proof_core = {
        "schema_version": PROOF_SCHEMA_VERSION,
        "operation": operation,
        "required_claim_ids": required_claim_ids,
        "reasoning_path_claim_ids": reasoning_claim_ids,
        "dependency_closure_claim_ids": dependency_closure_claim_ids,
        "gold_operand_claim_ids": operand_claim_ids,
        "required_event_instance_ids": required_event_ids,
        "reasoning_path_event_instance_ids": reasoning_event_ids,
        "answer_event_instance_ids": gold_reference_event_ids,
        "dependency_closure_event_instance_ids": dependency_event_ids,
        "gold_operand_event_instance_ids": operand_event_ids,
        "claim_id_to_subject_event_id": copy.deepcopy(claim_event_map),
        "gold": copy.deepcopy(gold),
        "gold_answer_text": gold_answer_text,
        "answer_type": answer_type,
        "derivation": {
            **copy.deepcopy(derivation),
            "answer_claim_ids": answer_claim_ids,
            "answer_event_instance_ids": gold_reference_event_ids,
        },
    }
    proof_id = _hash("proof", {"bundle": frozen_hash, "proof": proof_core})
    proof = {"proof_id": proof_id, **proof_core}
    for item in options:
        source_claim_id = _text(item.get("source_claim_id"))
        source_event_id = claim_event_map.get(source_claim_id)
        if source_event_id:
            item["source_event_instance_id"] = source_event_id
            item["anchor_ref"] = source_event_id
        if item.get("truth_value") is True:
            if answer_claim_ids:
                item.setdefault("anchor_kind", "claim")
            else:
                item["anchor_kind"] = "derived_claim"
                item["source_claim_id"] = proof_id
                item["support_event_instance_ids"] = (
                    operand_event_ids or required_event_ids
                )
                if gold_reference_event_ids:
                    item["source_event_instance_id"] = gold_reference_event_ids[0]
                    item["anchor_ref"] = gold_reference_event_ids[0]
        else:
            item.setdefault(
                "anchor_kind",
                "claim" if _text(item.get("source_claim_id")) else "synthetic",
            )
    operation_proof_view_core = {
        "schema_version": "stage64_operation_proof_view_v1",
        "source_graph_snapshot_hash": graph_hash,
        "claim_bundle_hash": frozen_hash,
        "qa_operation": operation,
        "qa_eligible_claim_ids": required_claim_ids,
        "reasoning_claim_ids": reasoning_claim_ids,
        "required_claim_ids": required_claim_ids,
        "gold_claim_ids": answer_claim_ids,
        "distractor_claim_ids": distractor_claim_ids,
        "dependency_closure_claim_ids": dependency_closure_claim_ids,
        "reasoning_event_instance_ids": reasoning_event_ids,
        "required_event_instance_ids": required_event_ids,
        "supporting_event_instance_ids": required_event_ids,
        "gold_event_instance_ids": gold_reference_event_ids,
        "distractor_event_instance_ids": distractor_event_ids,
        "dependency_closure_event_instance_ids": dependency_event_ids,
        "answer_projection": answer_projection,
        "answer_type": answer_type,
        "gold_answer": gold_answer_text,
        "derivation": copy.deepcopy(proof["derivation"]),
    }
    operation_proof_view = {
        **operation_proof_view_core,
        "proof_sha256": _hash("opv", operation_proof_view_core, length=64)[4:],
    }
    qa_id = _hash("qa", {"family": family_id, "proof_id": proof_id})
    target = {
        "target_id": _hash("target", {"qa_id": qa_id, "operation": operation}),
        "qa_operation": operation,
        "qa_subtype": subtype,
        "reasoning_template": template,
        "hop_count": hop_count,
        "source_graph_operation": source_graph_operation,
        "proof_id": proof_id,
        "graph_snapshot_hash": graph_hash,
        "claim_bundle_hash": frozen_hash,
    }
    query_plan = {
        "schema_version": "stage3_claim_query_plan_compat_v1",
        "qa_operation": operation,
        "reasoning_subtype": subtype,
        "reasoning_template": template,
        "hop_count": hop_count,
        "hard_negative_strategy": "claim_first_graph_grounded_options",
        "reasoning_anchor_path": reasoning_event_ids,
        "required_anchor_refs": required_event_ids,
        "supporting_anchor_refs": required_event_ids,
        "gold_anchor_refs": gold_reference_event_ids,
        "distractor_anchor_refs": distractor_event_ids,
        "dependency_closure_refs": dependency_event_ids,
        "resolved_reasoning_event_instance_ids": reasoning_event_ids,
        "resolved_required_event_instance_ids": required_event_ids,
        "resolved_supporting_event_instance_ids": required_event_ids,
        "resolved_gold_event_instance_ids": gold_reference_event_ids,
        "resolved_distractor_event_instance_ids": distractor_event_ids,
        "resolved_dependency_closure_event_instance_ids": dependency_event_ids,
        "claim_id_to_event_instance_id": copy.deepcopy(claim_event_map),
        "event_instance_id_to_claim_id": copy.deepcopy(event_claim_map),
        "answer_projection": answer_projection,
        "gold_answer": gold_answer_text,
        "normalized_gold_answer": _fold(gold_answer_text),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "qa_id": qa_id,
        "dialogue_family_id": family_id,
        "qa_operation": operation,
        "qa_subtype": subtype,
        "reasoning_subtype": subtype,
        "reasoning_template": template,
        "hop_count": hop_count,
        "hard_negative_strategy": "claim_first_graph_grounded_options",
        "forbid_max_time_shortcut": False,
        "answer_type": answer_type,
        "question_text_draft": question_draft,
        "question_surface_frozen": False,
        "options": options,
        "gold": gold,
        "gold_answer_text": gold_answer_text,
        "proof": proof,
        "operation_proof_view": operation_proof_view,
        "operation_specific_target": target,
        "claim_event_map": copy.deepcopy(claim_event_map),
        "event_claim_map": copy.deepcopy(event_claim_map),
        "query_plan": query_plan,
        "evidence_claim_ids": required_claim_ids,
        "evidence_ledger": (
            _support_subset(ledger, required_claim_ids) if require_support else []
        ),
        "support_binding_status": "bound" if require_support else "pending",
        "frozen_graph_snapshot_hash": graph_hash,
        "frozen_claim_bundle_hash": frozen_hash,
        "semantic_contract_frozen": True,
        "surface_naturalization_allowed_fields": ["question_text", "option_texts"],
    }


def _relation_lookup_candidates(
    bundle: dict[str, Any],
    family_id: str,
    nodes: list[dict[str, Any]],
    *,
    require_support: bool,
) -> list[dict[str, Any]]:
    ledger = _ledger_by_claim(bundle)
    source_op = _source_graph_operation(bundle)
    by_entity_slot: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for claim in nodes:
        by_entity_slot[(_text(claim.get("entity")), _text(claim.get("slot")))].append(claim)
    unique_facts = [values[0] for values in by_entity_slot.values() if len(values) == 1]
    by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for claim in unique_facts:
        by_entity[_text(claim.get("entity"))].append(claim)
    binder_occurrences: dict[tuple[str, str], set[str]] = defaultdict(set)
    for claim in unique_facts:
        binder_occurrences[(_fold(claim.get("slot")), _fold(claim.get("value")))].add(
            _text(claim.get("entity"))
        )
    pool = _option_pool(nodes, "")
    result: list[dict[str, Any]] = []
    for entity in sorted(by_entity):
        facts = sorted(by_entity[entity], key=lambda item: (_text(item.get("slot")), _claim_id(item)))
        for binder in facts:
            binder_key = (_fold(binder.get("slot")), _fold(binder.get("value")))
            if len(binder_occurrences[binder_key]) != 1:
                continue
            for target in facts:
                if target is binder or _fold(target.get("slot")) == _fold(binder.get("slot")):
                    continue
                binder_id, target_id = _claim_id(binder), _claim_id(target)
                gold_text = _text(target.get("value"))
                preferred = [
                    item
                    for item in nodes
                    if _fold(item.get("slot")) == _fold(target.get("slot"))
                    and _claim_id(item) != target_id
                ]
                distractors = _option_pool(preferred + nodes, gold_text)
                try:
                    options, gold = _ordered_options(
                        qa_seed={"binder": binder_id, "target": target_id},
                        gold_text=gold_text,
                        gold_claim_id=target_id,
                        distractors=distractors,
                    )
                except ClaimProofCompileError:
                    continue
                question = (
                    f"What is the {_slot_label(target.get('slot'))} of the person whose "
                    f"{_slot_label(binder.get('slot'))} is {_text(binder.get('value'))}?"
                )
                result.append(
                    _proof_and_qa(
                        bundle=bundle,
                        family_id=family_id,
                        source_graph_operation=source_op,
                        operation="relation_lookup",
                        subtype="entity_constrained_relation",
                        template="bind_entity_then_relation",
                        hop_count=2,
                        answer_projection="bound_attribute_value",
                        question_draft=question,
                        answer_type="AttributeValue",
                        gold_answer_text=gold_text,
                        answer_claim_ids=[target_id],
                        required_claim_ids=[binder_id, target_id],
                        reasoning_claim_ids=[binder_id, target_id],
                        dependency_closure_claim_ids=[binder_id, target_id],
                        derivation={
                            "rule": "bind_unique_entity_then_read_attribute",
                            "binder_claim_ids": [binder_id],
                            "target_claim_ids": [target_id],
                        },
                        options=options,
                        gold=gold,
                        ledger=ledger,
                        require_support=require_support,
                    )
                )
    return result


def _status_lookup_candidates(
    bundle: dict[str, Any],
    family_id: str,
    nodes: list[dict[str, Any]],
    *,
    require_support: bool,
) -> list[dict[str, Any]]:
    ledger = _ledger_by_claim(bundle)
    source_op = _source_graph_operation(bundle)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for claim in nodes:
        groups[(_text(claim.get("entity")), _text(claim.get("slot")))].append(claim)
    result: list[dict[str, Any]] = []
    for (entity, slot), group in sorted(groups.items()):
        if len(group) < 4:
            continue
        by_status: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for claim in group:
            by_status[_status(claim)].append(claim)
        for status, matches in sorted(by_status.items()):
            if len(matches) != 1:
                continue
            target = matches[0]
            target_id = _claim_id(target)
            gold_text = _text(target.get("value"))
            distractors = _option_pool([item for item in group if item is not target], gold_text)
            try:
                options, gold = _ordered_options(
                    qa_seed={"status": status, "target": target_id},
                    gold_text=gold_text,
                    gold_claim_id=target_id,
                    distractors=distractors,
                )
            except ClaimProofCompileError:
                continue
            group_ids = [_claim_id(item) for item in group]
            result.append(
                _proof_and_qa(
                    bundle=bundle,
                    family_id=family_id,
                    source_graph_operation=source_op,
                    operation="temporal_state_update",
                    subtype="status_conditioned_value",
                    template="select_entity_slot_by_status",
                    hop_count=len(group_ids),
                    answer_projection="status_conditioned_value",
                    question_draft=(
                        f"Which {_slot_label(slot)} is recorded as {status.replace('_', ' ')} "
                        f"for {entity}?"
                    ),
                    answer_type="AttributeValue",
                    gold_answer_text=gold_text,
                    answer_claim_ids=[target_id],
                    required_claim_ids=group_ids,
                    reasoning_claim_ids=group_ids,
                    dependency_closure_claim_ids=group_ids,
                    derivation={
                        "rule": "select_unique_status_in_entity_slot",
                        "status": status,
                        "candidate_claim_ids": group_ids,
                        "target_claim_ids": [target_id],
                    },
                    options=options,
                    gold=gold,
                    ledger=ledger,
                    require_support=require_support,
                )
            )
    return result


def _direct_attribute_candidates(
    bundle: dict[str, Any],
    family_id: str,
    nodes: list[dict[str, Any]],
    *,
    require_support: bool,
) -> list[dict[str, Any]]:
    """Compile strict one-hop fallbacks for sparse but well-supported graphs.

    These are intentionally appended after multi-claim candidates and are
    tagged as one-hop. They prevent one bad relation edge from discarding an
    otherwise valuable dialogue while keeping difficulty visible to training
    and evaluation code.
    """

    ledger = _ledger_by_claim(bundle)
    source_op = _source_graph_operation(bundle)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for claim in nodes:
        groups[(_text(claim.get("entity")), _text(claim.get("slot")))].append(claim)
    result: list[dict[str, Any]] = []
    for (entity, slot), matches in sorted(groups.items()):
        if len(matches) != 1:
            continue
        target = matches[0]
        target_id = _claim_id(target)
        gold_text = _text(target.get("value"))
        preferred = [
            item
            for item in nodes
            if _fold(item.get("slot")) == _fold(slot) and item is not target
        ]
        distractors = _option_pool(preferred + nodes, gold_text)
        try:
            options, gold = _ordered_options(
                qa_seed={"direct": target_id},
                gold_text=gold_text,
                gold_claim_id=target_id,
                distractors=distractors,
            )
        except ClaimProofCompileError:
            continue
        result.append(
            _proof_and_qa(
                bundle=bundle,
                family_id=family_id,
                source_graph_operation=source_op,
                operation="relation_lookup",
                subtype="direct_attribute_lookup",
                template="read_named_entity_attribute",
                hop_count=1,
                answer_projection="attribute_value",
                question_draft=f"What is {entity}'s {_slot_label(slot)}?",
                answer_type="AttributeValue",
                gold_answer_text=gold_text,
                answer_claim_ids=[target_id],
                required_claim_ids=[target_id],
                reasoning_claim_ids=[target_id],
                dependency_closure_claim_ids=[target_id],
                derivation={
                    "rule": "read_named_entity_attribute",
                    "target_claim_ids": [target_id],
                },
                options=options,
                gold=gold,
                ledger=ledger,
                require_support=require_support,
            )
        )
    return result


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _number_text(number: float) -> str:
    return str(int(number)) if float(number).is_integer() else str(number)


def _numeric_distractors(answer: float, operands: list[float]) -> list[dict[str, Any]]:
    candidates = [
        abs(operands[0] + operands[1]),
        min(operands),
        max(operands),
        abs(answer + 1),
        abs(answer - 1),
        abs(operands[0] - operands[1]) + 2,
    ]
    result = []
    seen = {_number_text(answer)}
    for value in candidates:
        text = _number_text(value)
        if text in seen:
            continue
        seen.add(text)
        result.append(
            {
                "text": text,
                "truth_value": False,
                "source_claim_id": None,
                "distractor_origin": "local_numeric_mutation",
            }
        )
    return result


def _comparison_candidates(
    bundle: dict[str, Any],
    family_id: str,
    nodes: list[dict[str, Any]],
    *,
    require_support: bool,
) -> list[dict[str, Any]]:
    ledger = _ledger_by_claim(bundle)
    source_op = _source_graph_operation(bundle)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for claim in nodes:
        if _numeric(claim.get("value")) is not None:
            groups[_text(claim.get("slot"))].append(claim)
    result: list[dict[str, Any]] = []
    for slot, group in sorted(groups.items()):
        active = [claim for claim in group if _status(claim) in ACTIVE_STATUSES]
        if len(active) != 2:
            continue
        entities = [_text(item.get("entity")) for item in active]
        if len(set(entities)) != 2:
            continue
        values = [_numeric(item.get("value")) for item in active]
        assert all(value is not None for value in values)
        left, right = float(values[0]), float(values[1])
        answer = abs(left - right)
        gold_text = _number_text(answer)
        operand_ids = [_claim_id(item) for item in active]
        dependency_ids = [_claim_id(item) for item in group]
        options, gold = _ordered_options(
            qa_seed={"slot": slot, "operands": operand_ids},
            gold_text=gold_text,
            gold_claim_id=None,
            distractors=_numeric_distractors(answer, [left, right]),
        )
        result.append(
            _proof_and_qa(
                bundle=bundle,
                family_id=family_id,
                source_graph_operation=source_op,
                operation="comparison",
                subtype="time_matched_absolute_difference",
                template="select_current_pair_then_compare",
                hop_count=len(dependency_ids),
                answer_projection="absolute_difference",
                question_draft=(
                    f"What is the absolute difference between the current {_slot_label(slot)} "
                    f"for {entities[0]} and {entities[1]}?"
                ),
                answer_type="Numeric",
                gold_answer_text=gold_text,
                answer_claim_ids=[],
                required_claim_ids=dependency_ids,
                reasoning_claim_ids=dependency_ids,
                dependency_closure_claim_ids=dependency_ids,
                derivation={
                    "rule": "absolute_difference",
                    "operand_claim_ids": operand_ids,
                    "operand_values": [left, right],
                    "computed_value": gold_text,
                },
                options=options,
                gold=gold,
                ledger=ledger,
                require_support=require_support,
            )
        )
    return result


def _immutable_target_contract(target: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(target, dict) or not target:
        raise ClaimProofCompileError("immutable Stage64 target must be a non-empty object")
    operation = _text(target.get("qa_operation"))
    answer_type = _text(target.get("preferred_answer_type"))
    missing = [
        field for field in IMMUTABLE_TARGET_FIELDS if target.get(field) in (None, "")
    ]
    if not operation:
        missing.append("qa_operation")
    if not answer_type:
        missing.append("preferred_answer_type")
    if missing:
        raise ClaimProofCompileError(
            "immutable Stage64 target lacks fields: " + ", ".join(sorted(set(missing)))
        )
    hop_count = target.get("hop_count")
    if isinstance(hop_count, bool) or not isinstance(hop_count, int) or hop_count < 1:
        raise ClaimProofCompileError("immutable Stage64 target hop_count must be positive int")
    contract = {
        "qa_operation": operation,
        "preferred_answer_type": answer_type,
        "reasoning_subtype": _text(target.get("reasoning_subtype")),
        "reasoning_template": _text(target.get("reasoning_template")),
        "hop_count": hop_count,
        "hard_negative_strategy": _text(target.get("hard_negative_strategy")),
        "forbid_max_time_shortcut": target.get("forbid_max_time_shortcut") is True,
    }
    strategy = contract["hard_negative_strategy"]
    expected = IMMUTABLE_TARGET_SIGNATURES.get(operation, {}).get(strategy)
    actual = (
        contract["reasoning_subtype"],
        contract["reasoning_template"],
        contract["hop_count"],
        contract["forbid_max_time_shortcut"],
    )
    if expected is None or actual != expected:
        raise ClaimProofCompileError(
            "immutable Stage64 target is not a canonical reasoning signature: "
            f"operation={operation!r}, signature={actual!r}, "
            f"hard_negative_strategy={strategy!r}"
        )
    if operation in {"comparison", "count"} and answer_type != "Number":
        raise ClaimProofCompileError(
            f"immutable {operation} target requires preferred_answer_type=Number"
        )
    if operation == "bridge" and answer_type == "YesNo":
        raise ClaimProofCompileError(
            "immutable bridge target cannot use preferred_answer_type=YesNo"
        )
    return contract


def _plan_events_by_id(bundle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    plan = bundle.get("current_event_plan")
    if not isinstance(plan, dict) or not isinstance(plan.get("events"), list):
        raise ClaimProofCompileError(
            "immutable target compilation requires current_event_plan.events"
        )
    result: dict[str, dict[str, Any]] = {}
    for event in plan["events"]:
        if not isinstance(event, dict):
            continue
        event_id = _text(event.get("event_instance_id") or event.get("event_id"))
        if not event_id:
            raise ClaimProofCompileError("current_event_plan contains event without id")
        if event_id in result:
            raise ClaimProofCompileError(
                f"current_event_plan has duplicate event id: {event_id}"
            )
        result[event_id] = event
    return result



def _plan_event_rows(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    by_id = _plan_events_by_id(bundle)
    rows = list(by_id.values())
    if not rows:
        raise ClaimProofCompileError("immutable target graph has no events")
    return rows


def _node_claims_by_event(
    nodes: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for claim in nodes:
        event_id = _text(claim.get("subject_event_id"))
        if not event_id or event_id in result:
            raise ClaimProofCompileError(
                f"immutable target node mapping is missing or ambiguous: {event_id!r}"
            )
        result[event_id] = claim
    return result


def _semantic_claims_by_event(
    bundle: dict[str, Any],
    claim_type: str,
    *,
    require_support: bool,
) -> dict[str, dict[str, Any]]:
    ledger = _ledger_by_claim(bundle)
    result: dict[str, dict[str, Any]] = {}
    for claim in _claims(bundle):
        if _fold(claim.get("claim_type")) != _fold(claim_type):
            continue
        eligible = (
            claim.get("qa_eligible") is True
            if require_support
            else bool(claim.get("qa_candidate"))
            and _fold(claim.get("edge_role")) == "semantic"
        )
        claim_id = _claim_id(claim)
        if not eligible or (require_support and claim_id not in ledger):
            continue
        event_id = _text(claim.get("subject_event_id"))
        if not event_id or event_id in result:
            raise ClaimProofCompileError(
                f"immutable target {claim_type} claim mapping is ambiguous: {event_id!r}"
            )
        result[event_id] = claim
    return result


def _event_id(event: dict[str, Any]) -> str:
    return _text(event.get("event_instance_id") or event.get("event_id"))


def _event_dependencies(event: dict[str, Any]) -> list[str]:
    raw = event.get("depends_on")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ClaimProofCompileError("immutable target depends_on must be a list")
    return [_text(item) for item in raw if _text(item)]


_PERSON_VALUE_TOKEN_RE = re.compile(
    r"[^\W\d_]+(?:[-'][^\W\d_]+)*\.?",
    re.UNICODE,
)
_PERSON_VALUE_PARTICLES = frozenset(
    {"al", "bin", "da", "de", "del", "der", "di", "du", "la", "le", "van", "von"}
)
_PERSON_PREDICATE_TOKENS = frozenset(
    {
        "advisor", "adviser", "artist", "assignee", "author", "captain",
        "child", "coach", "colleague", "companion", "contact", "coordinator",
        "curator", "delegate", "director", "discoverer", "employee", "founder",
        "friend", "guest", "host", "leader", "manager", "member", "mentor",
        "name", "observer", "organizer", "owner", "parent", "participant",
        "partner", "person", "recipient", "relative", "resident", "sibling",
        "speaker", "spouse", "teacher", "winner",
    }
)
_PERSON_PREDICATE_CONFLICT_TOKENS = frozenset(
    {
        "address", "age", "amount", "appointment", "approval", "condition",
        "confirmation", "count", "date", "day", "deadline", "destination",
        "duration", "hour", "level", "location", "minute", "month", "number",
        "permission", "quantity", "result", "room", "route", "schedule", "scope",
        "score", "site", "state", "status", "time", "venue", "verification",
        "week", "year",
    }
)


def _person_value_is_atomic(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    words = value.strip().split()
    if not 1 <= len(words) <= 5:
        return False
    for word in words:
        if _PERSON_VALUE_TOKEN_RE.fullmatch(word) is None:
            return False
        segments = re.split(r"[-']", word.rstrip("."))
        if any(
            not segment
            or (
                not segment[0].isupper()
                and segment.casefold() not in _PERSON_VALUE_PARTICLES
            )
            for segment in segments
        ):
            return False
    return True


def _person_answer_slot_compatible(slot: Any) -> bool:
    normalized = re.sub(r"[\W]+", "_", _text(slot).casefold()).strip("_")
    if not normalized:
        return False
    tokens = [token for token in normalized.split("_") if token]
    variants = set(tokens)
    variants.update(
        token[:-1] for token in tokens if len(token) > 3 and token.endswith("s")
    )
    if variants.intersection(_PERSON_PREDICATE_CONFLICT_TOKENS):
        return False
    return bool(
        variants.intersection(_PERSON_PREDICATE_TOKENS)
        or normalized.endswith("_by")
    )


def _answer_slot_compatible(slot: Any, answer_type: str) -> bool:
    if answer_type == "Person":
        return _person_answer_slot_compatible(slot)
    return bool(_text(slot))


def _atomic_answer_compatible(value: Any, answer_type: str) -> bool:
    if answer_type == "Number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if answer_type == "YesNo":
        return isinstance(value, bool) or _fold(value) in {"yes", "no", "true", "false"}
    if answer_type == "Person":
        return _person_value_is_atomic(value)
    named_types = {
        "Person",
        "GroupOrOrg",
        "LocationOrScope",
        "DateOrTime",
        "ProperNounOrNamedEntity",
    }
    if answer_type in named_types and not isinstance(value, str):
        return False
    if isinstance(value, (dict, list, tuple, set, bool)):

        return False
    text = _text(value)
    words = re.findall("[A-Za-z]+", text.casefold())
    if (
        answer_type in named_types
        and len(words) >= 3
        and set(words).intersection({"is", "was", "were", "has", "have", "will", "because"})
    ):
        return False
    return bool(
        text
        and len(text.split()) <= 24
        and "Q:" not in text
        and "A:" not in text
        and chr(10) not in text
    )


def _apply_immutable_target_contract(
    qa: dict[str, Any],
    immutable_target: dict[str, Any],
    contract: dict[str, Any],
    *,
    query_plan_updates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    qa["reasoning_subtype"] = contract["reasoning_subtype"]
    qa["reasoning_template"] = contract["reasoning_template"]
    qa["hop_count"] = contract["hop_count"]
    qa["hard_negative_strategy"] = contract["hard_negative_strategy"]
    qa["forbid_max_time_shortcut"] = contract["forbid_max_time_shortcut"]
    qa["answer_type"] = contract["preferred_answer_type"]
    qa["fulfills_immutable_stage64_target"] = True
    qa["immutable_stage64_target"] = copy.deepcopy(immutable_target)
    qa["target_fulfillment_match"] = copy.deepcopy(contract)
    qa["operation_specific_target"] = {
        **copy.deepcopy(qa["operation_specific_target"]),
        **copy.deepcopy(contract),
        "immutable": True,
    }
    qa["query_plan"].update(
        {
            "qa_operation": contract["qa_operation"],
            "reasoning_subtype": contract["reasoning_subtype"],
            "reasoning_template": contract["reasoning_template"],
            "hop_count": contract["hop_count"],
            "hard_negative_strategy": contract["hard_negative_strategy"],
            "forbid_max_time_shortcut": contract["forbid_max_time_shortcut"],
            **copy.deepcopy(query_plan_updates or {}),
        }
    )
    return qa


def _near_integer_options(
    answer: int,
    *,
    qa_seed: Any,
    anchor_claims: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    if answer < 1 or len(anchor_claims) < 3:
        raise ClaimProofCompileError(
            "derived numeric target needs a positive answer and three graph anchors"
        )
    values = [answer - 1, answer + 1, answer + 2]
    distractors = [
        {
            "text": str(value),
            "truth_value": False,
            "source_claim_id": _claim_id(anchor_claims[index]),
            "distractor_origin": "nearby_graph_anchored_numeric_mutation",
        }
        for index, value in enumerate(values)
    ]
    return _ordered_options(
        qa_seed=qa_seed,
        gold_text=str(answer),
        gold_claim_id=None,
        distractors=distractors,
    )


def _bridge_target_candidate(
    bundle: dict[str, Any],
    family_id: str,
    nodes: list[dict[str, Any]],
    immutable_target: dict[str, Any],
    *,
    require_support: bool,
) -> dict[str, Any]:
    contract = _immutable_target_contract(immutable_target)
    events = _plan_event_rows(bundle)
    if not 6 <= len(events) <= 8:
        raise ClaimProofCompileError("immutable bridge target requires 6-8 events")
    ids = [_event_id(event) for event in events]
    if _event_dependencies(events[2]) != ids[:2] or any(
        _event_dependencies(event) for index, event in enumerate(events) if index != 2
    ):
        raise ClaimProofCompileError(
            "immutable bridge target requires one exact two-support closure"
        )
    binder_event, target_event, terminal_event = events[:3]
    binder_value = binder_event.get("value")
    target_entity = target_event.get("entity")
    if (
        not isinstance(binder_value, str)
        or not binder_value.strip()
        or not isinstance(target_entity, str)
        or binder_value != target_entity
    ):
        raise ClaimProofCompileError(
            "immutable bridge supports lack an explicit binder-to-target join"
        )
    if not _answer_slot_compatible(
        target_event.get("slot"), contract["preferred_answer_type"]
    ):
        raise ClaimProofCompileError(
            "bridge answer slot does not match target answer type"
        )
    answer_surface = _fold(target_event.get("value"))
    terminal_surfaces = [
        _fold(terminal_event.get(field))
        for field in ("entity", "slot", "value")
    ]
    if (
        not answer_surface
        or any(
            re.search(
                rf"(?<!\w){re.escape(answer_surface)}(?!\w)",
                surface,
            )
            for surface in terminal_surfaces
            if surface
        )
    ):
        raise ClaimProofCompileError(
            "immutable bridge terminal connector copies the answer value"
        )
    if (
        contract["reasoning_template"] == "join_support_under_scope_constraint"
        and not any(
            isinstance(event.get("scope"), dict) and event.get("scope")
            for event in events[:3]
        )
    ):
        raise ClaimProofCompileError("scope-constrained bridge lacks scoped support")
    if (
        contract["hard_negative_strategy"]
        == "one_hop_preserving_alternative"
        and _fold(events[3].get("entity")) != _fold(terminal_event.get("entity"))
        and _fold(events[3].get("slot")) != _fold(terminal_event.get("slot"))
    ):
        raise ClaimProofCompileError(
            "bridge one-hop hard negative does not preserve the terminal entity"
        )
    by_event = _node_claims_by_event(nodes)
    claims = [by_event.get(event_id) for event_id in ids]
    if any(claim is None for claim in claims):
        raise ClaimProofCompileError("immutable bridge event lacks a node claim")
    typed_claims = [claim for claim in claims if isinstance(claim, dict)]
    binder_claim, answer_claim, terminal_claim = typed_claims[:3]
    distractor_claims = typed_claims[3:]
    if not any(
        _fold(claim.get("entity")) == _fold(terminal_claim.get("entity"))
        or _fold(claim.get("slot")) == _fold(terminal_claim.get("slot"))
        for claim in distractor_claims
    ):
        raise ClaimProofCompileError(
            "immutable bridge lacks a terminal-related hard negative"
        )
    answer = answer_claim.get("value")
    if not _atomic_answer_compatible(answer, contract["preferred_answer_type"]):
        raise ClaimProofCompileError("bridge derived value does not match target answer type")
    typed_distractor_claims = [
        claim
        for claim in distractor_claims
        if _atomic_answer_compatible(
            claim.get("value"), contract["preferred_answer_type"]
        )
    ]
    options, gold = _ordered_options(
        qa_seed={"target": contract, "events": ids[:3]},
        gold_text=_text(answer),
        gold_claim_id=_claim_id(answer_claim),
        distractors=_option_pool(typed_distractor_claims, _text(answer)),
    )
    closure_ids = [_claim_id(claim) for claim in typed_claims[:3]]
    binder_entity = _text(binder_claim.get("entity"))
    binder_slot = _slot_label(binder_claim.get("slot"))
    binder_value = _text(binder_claim.get("value"))
    answer_slot = _slot_label(answer_claim.get("slot"))
    qa = _proof_and_qa(
        bundle=bundle,
        family_id=family_id,
        source_graph_operation="bridge",
        operation="bridge",
        subtype=contract["reasoning_subtype"],
        template=contract["reasoning_template"],
        hop_count=contract["hop_count"],
        answer_projection=ANSWER_PROJECTION_BY_OPERATION["bridge"],
        question_draft=(
            f"For {binder_value}, identified by {binder_entity} through {binder_slot}, "
            f"what is the {answer_slot}?"
        ),
        answer_type=contract["preferred_answer_type"],
        gold_answer_text=_text(answer),
        answer_claim_ids=[_claim_id(answer_claim)],
        required_claim_ids=closure_ids,
        reasoning_claim_ids=closure_ids,
        dependency_closure_claim_ids=closure_ids,
        derivation={
            "rule": "bind_support_then_follow_dependency_closure",
            "binder_claim_ids": [_claim_id(binder_claim)],
            "target_claim_ids": [_claim_id(answer_claim)],
            "terminal_claim_ids": [_claim_id(terminal_claim)],
        },
        options=options,
        gold=gold,
        ledger=_ledger_by_claim(bundle),
        require_support=require_support,
    )
    proof_id = _text((qa.get("proof") or {}).get("proof_id"))
    for option in qa["options"]:
        if option.get("truth_value") is True:
            option.update(
                {
                    "source_claim_id": proof_id,
                    "source_event_instance_id": ids[2],
                    "anchor_ref": ids[2],
                    "anchor_kind": "derived_claim",
                    "support_event_instance_ids": ids[:2],
                }
            )
    qa["query_plan"].update(
        {
            "gold_anchor_refs": [ids[2]],
            "resolved_gold_event_instance_ids": [ids[2]],
            "dependency_closure_refs": ids[:3],
            "resolved_dependency_closure_event_instance_ids": ids[:3],
        }
    )
    return _apply_immutable_target_contract(qa, immutable_target, contract)


def _comparison_target_candidate(
    bundle: dict[str, Any],
    family_id: str,
    nodes: list[dict[str, Any]],
    immutable_target: dict[str, Any],
    *,
    require_support: bool,
) -> dict[str, Any]:
    contract = _immutable_target_contract(immutable_target)
    events = _plan_event_rows(bundle)
    if len(events) != 8:
        raise ClaimProofCompileError("immutable comparison target requires eight events")
    first, second, old_first, old_second = events[:4]
    pair_ok = (
        _text(first.get('entity')) != _text(second.get('entity'))
        and _text(first.get('slot')) == _text(second.get('slot'))
        and isinstance(first.get("time_index"), int)
        and first.get("time_index") == second.get("time_index")
        and _fold(first.get("status")) in STAGE64_CURRENT_STATUSES
        and _fold(second.get("status")) in STAGE64_CURRENT_STATUSES
        and isinstance(first.get('value'), int)
        and not isinstance(first.get('value'), bool)
        and isinstance(second.get('value'), int)
        and not isinstance(second.get('value'), bool)
        and first.get('value') != second.get('value')
    )
    contrast_ok = (
        {_text(old_first.get('entity')), _text(old_second.get('entity'))}
        == {_text(first.get('entity')), _text(second.get('entity'))}
        and _text(old_first.get('slot')) == _text(first.get('slot'))
        and _text(old_second.get('slot')) == _text(first.get('slot'))
        and old_first.get("time_index") == old_second.get("time_index")
        and old_first.get("time_index") != first.get("time_index")
        and _fold(old_first.get("status")) not in STAGE64_CURRENT_STATUSES
        and _fold(old_second.get("status")) not in STAGE64_CURRENT_STATUSES
    )
    if not pair_ok or not contrast_ok:
        raise ClaimProofCompileError("immutable comparison roles are not uniquely realized")
    ids = [_event_id(event) for event in events]
    active_numeric_groups: dict[tuple[str, int], list[str]] = defaultdict(list)
    for event in events:
        event_time = event.get("time_index")
        event_value = event.get("value")
        if (
            isinstance(event_time, int)
            and isinstance(event_value, (int, float))
            and not isinstance(event_value, bool)
            and _fold(event.get("status")) in STAGE64_CURRENT_STATUSES
        ):
            active_numeric_groups[
                (_fold(event.get("slot")), event_time)
            ].append(_event_id(event))
    comparison_groups = [
        group for group in active_numeric_groups.values() if len(group) >= 2
    ]
    if len(comparison_groups) != 1 or comparison_groups[0] != ids[:2]:
        raise ClaimProofCompileError(
            "immutable comparison has another active same-time numeric pair"
        )
    by_event = _node_claims_by_event(nodes)
    claims = [by_event.get(event_id) for event_id in ids]
    if any(claim is None for claim in claims[:5]):
        raise ClaimProofCompileError("immutable comparison event lacks a node claim")
    typed = [claim for claim in claims if isinstance(claim, dict)]
    pair_claims = typed[:2]
    answer = abs(int(first["value"]) - int(second["value"]))
    options, gold = _near_integer_options(
        answer,
        qa_seed={"target": contract, "pair": ids[:2]},
        anchor_claims=typed[2:5],
    )
    include_contrast = contract["reasoning_template"] == "select_time_matched_pair_then_compare"
    reasoning_claims = typed[:4] if include_contrast else pair_claims
    reasoning_ids = [_claim_id(claim) for claim in reasoning_claims]
    pair_claim_ids = [_claim_id(claim) for claim in pair_claims]
    pair_event_ids = ids[:2]
    closure_event_ids = ids[:4] if include_contrast else pair_event_ids
    qa = _proof_and_qa(
        bundle=bundle,
        family_id=family_id,
        source_graph_operation="comparison",
        operation="comparison",
        subtype=contract["reasoning_subtype"],
        template=contract["reasoning_template"],
        hop_count=contract["hop_count"],
        answer_projection=ANSWER_PROJECTION_BY_OPERATION["comparison"],
        question_draft=(
            f"What is the current absolute difference in {_slot_label(first.get('slot'))} "
            f"between {_text(first.get('entity'))} and {_text(second.get('entity'))}?"
        ),
        answer_type="Number",
        gold_answer_text=str(answer),
        answer_claim_ids=[],
        required_claim_ids=reasoning_ids,
        reasoning_claim_ids=reasoning_ids,
        dependency_closure_claim_ids=reasoning_ids,
        derivation={
            "rule": "absolute_difference",
            "operand_claim_ids": pair_claim_ids,
            "operand_values": [first["value"], second["value"]],
            "computed_value": str(answer),
        },
        options=options,
        gold=gold,
        ledger=_ledger_by_claim(bundle),
        require_support=require_support,
    )
    qa["query_plan"].update(
        {
            "required_anchor_refs": pair_event_ids,
            "supporting_anchor_refs": pair_event_ids,
            "gold_anchor_refs": pair_event_ids,
            "resolved_required_event_instance_ids": pair_event_ids,
            "resolved_supporting_event_instance_ids": pair_event_ids,
            "resolved_gold_event_instance_ids": pair_event_ids,
            "dependency_closure_refs": closure_event_ids,
            "resolved_dependency_closure_event_instance_ids": closure_event_ids,
        }
    )
    return _apply_immutable_target_contract(qa, immutable_target, contract)


def _count_target_candidate(
    bundle: dict[str, Any],
    family_id: str,
    nodes: list[dict[str, Any]],
    immutable_target: dict[str, Any],
    *,
    require_support: bool,
) -> dict[str, Any]:
    contract = _immutable_target_contract(immutable_target)
    events = _plan_event_rows(bundle)
    if len(events) != 8:
        raise ClaimProofCompileError("immutable count target requires eight events")
    raw_memberships = [event.get("set_membership") for event in events]
    if any(not isinstance(item, dict) for item in raw_memberships):
        raise ClaimProofCompileError("immutable count event lacks typed membership")
    memberships = [item for item in raw_memberships if isinstance(item, dict)]
    dimensions = {
        (
            _text(item.get("set_id")),
            _text(item.get("condition")),
            _text(event.get("entity")),
            _text(event.get("slot")),
        )
        for event, item in zip(events, memberships)
    }
    included = [item.get("included") for item in memberships]
    values = {_fold(event.get("value")) for event in events}
    true_count = sum(value is True for value in included)
    false_count = sum(value is False for value in included)
    if (
        len(dimensions) != 1
        or any(not value for value in next(iter(dimensions), ()))
        or len(values) != len(events)
        or any(not isinstance(value, bool) for value in included)
        or true_count < 2
        or false_count < 2
    ):
        raise ClaimProofCompileError("immutable count candidate set is not unique")

    ids = [_event_id(event) for event in events]
    membership_by_event = _semantic_claims_by_event(
        bundle, "membership", require_support=require_support
    )
    claims = [membership_by_event.get(event_id) for event_id in ids]
    if any(claim is None for claim in claims):
        raise ClaimProofCompileError(
            "immutable count event lacks one semantic membership claim"
        )
    typed = [claim for claim in claims if isinstance(claim, dict)]
    for claim, membership in zip(typed, memberships):
        expected_predicate = "member_of" if membership["included"] else "not_member_of"
        if (
            claim.get("included") is not membership["included"]
            or _fold(claim.get("predicate")) != expected_predicate
            or _text(claim.get("set_id")) != _text(membership.get("set_id"))
        ):
            raise ClaimProofCompileError(
                "immutable count membership claim does not preserve typed truth"
            )

    included_claims = [
        claim for claim, value in zip(typed, included) if value is True
    ]
    excluded_claims = [
        claim for claim, value in zip(typed, included) if value is False
    ]
    answer = len(included_claims)
    options, gold = _near_integer_options(
        answer,
        qa_seed={"target": contract, "set": next(iter(dimensions))},
        anchor_claims=[excluded_claims[0], excluded_claims[1], included_claims[0]],
    )
    claim_ids = [_claim_id(claim) for claim in typed]
    included_claim_ids = [_claim_id(claim) for claim in included_claims]
    set_id, condition, entity, slot = next(iter(dimensions))
    qa = _proof_and_qa(
        bundle=bundle,
        family_id=family_id,
        source_graph_operation="count",
        operation="count",
        subtype=contract["reasoning_subtype"],
        template=contract["reasoning_template"],
        hop_count=contract["hop_count"],
        answer_projection=ANSWER_PROJECTION_BY_OPERATION["count"],
        question_draft=(
            f"In the {set_id} set for {entity} {_slot_label(slot)}, how many entries "
            f"meet the condition {condition}?"
        ),
        answer_type="Number",
        gold_answer_text=str(answer),
        answer_claim_ids=[],
        required_claim_ids=claim_ids,
        reasoning_claim_ids=claim_ids,
        dependency_closure_claim_ids=claim_ids,
        derivation={
            "rule": "count_distinct",
            "operand_claim_ids": claim_ids,
            "included_claim_ids": included_claim_ids,
            "computed_value": str(answer),
            "filter_condition": condition,
            "set_id": set_id,
        },
        options=options,
        gold=gold,
        ledger=_ledger_by_claim(bundle),
        require_support=require_support,
    )
    return _apply_immutable_target_contract(
        qa,
        immutable_target,
        contract,
        query_plan_updates={
            "filter_predicate": {"condition": condition, "included": True},
            "candidate_event_instance_ids": ids,
            "included_event_instance_ids": [
                event_id
                for event_id, value in zip(ids, included)
                if value is True
            ],
        },
    )


def _cause_target_candidate(
    bundle: dict[str, Any],
    family_id: str,
    nodes: list[dict[str, Any]],
    immutable_target: dict[str, Any],
    *,
    require_support: bool,
) -> dict[str, Any]:
    contract = _immutable_target_contract(immutable_target)
    events = _plan_event_rows(bundle)
    if not 6 <= len(events) <= 8:
        raise ClaimProofCompileError("immutable cause target requires 6-8 events")
    ids = [_event_id(event) for event in events]
    edges = [
        (dependency, ids[index])
        for index, event in enumerate(events)
        for dependency in _event_dependencies(event)
    ]
    if (
        edges != [(ids[0], ids[1])]
        or not isinstance(events[0].get("time_index"), int)
        or not isinstance(events[1].get("time_index"), int)
        or events[0]["time_index"] >= events[1]["time_index"]
    ):
        raise ClaimProofCompileError("immutable cause graph lacks one directed edge")

    cause_by_event = _semantic_claims_by_event(
        bundle, "cause", require_support=require_support
    )
    cause_claim = cause_by_event.get(ids[0])
    node_by_event = _node_claims_by_event(nodes)
    effect_claim = node_by_event.get(ids[1])
    distractor_claims = [node_by_event.get(event_id) for event_id in ids[2:]]
    if (
        not isinstance(cause_claim, dict)
        or not isinstance(effect_claim, dict)
        or any(not isinstance(claim, dict) for claim in distractor_claims[:3])
        or [_text(item) for item in cause_claim.get("object_event_ids") or []]
        != [ids[1]]
        or [_text(item) for item in cause_claim.get("source_event_ids") or []]
        != ids[:2]
    ):
        raise ClaimProofCompileError(
            "immutable cause edge lacks one lossless semantic cause claim"
        )

    answer = effect_claim.get("value")
    if not _atomic_answer_compatible(answer, contract["preferred_answer_type"]):
        raise ClaimProofCompileError(
            "cause consequence value does not match target answer type"
        )
    typed_distractors = [
        claim for claim in distractor_claims if isinstance(claim, dict)
    ]
    if not any(
        _fold(claim.get("entity")) == _fold(effect_claim.get("entity"))
        or _fold(claim.get("slot")) == _fold(effect_claim.get("slot"))
        for claim in typed_distractors
    ):
        raise ClaimProofCompileError(
            "immutable cause target lacks an effect-related hard negative"
        )
    options, gold = _ordered_options(
        qa_seed={"target": contract, "edge": ids[:2]},
        gold_text=_text(answer),
        gold_claim_id=_claim_id(effect_claim),
        distractors=_option_pool(typed_distractors, _text(answer)),
    )
    cause_claim_id = _claim_id(cause_claim)
    effect_claim_id = _claim_id(effect_claim)
    path_ids = [cause_claim_id, effect_claim_id]
    cause_entity = _text(events[0].get("entity"))
    cause_value = _text(events[0].get("value"))
    qa = _proof_and_qa(
        bundle=bundle,
        family_id=family_id,
        source_graph_operation="cause_consequence",
        operation="cause_consequence",
        subtype=contract["reasoning_subtype"],
        template=contract["reasoning_template"],
        hop_count=contract["hop_count"],
        answer_projection=ANSWER_PROJECTION_BY_OPERATION["cause_consequence"],
        question_draft=(
            f"What effect resulted from {cause_entity} having {cause_value}?"
        ),
        answer_type=contract["preferred_answer_type"],
        gold_answer_text=_text(answer),
        answer_claim_ids=[effect_claim_id],
        required_claim_ids=path_ids,
        reasoning_claim_ids=path_ids,
        dependency_closure_claim_ids=path_ids,
        derivation={
            "rule": "follow_cause_to_effect",
            "cause_claim_ids": [cause_claim_id],
            "target_claim_ids": [effect_claim_id],
            "cause_event_instance_id": ids[0],
            "effect_event_instance_id": ids[1],
        },
        options=options,
        gold=gold,
        ledger=_ledger_by_claim(bundle),
        require_support=require_support,
    )
    return _apply_immutable_target_contract(
        qa,
        immutable_target,
        contract,
        query_plan_updates={
            "cause_direction": "cause_to_effect",
            "cause_source_ref": ids[0],
            "effect_target_ref": ids[1],
            "dependency_closure_refs": ids[:2],
            "resolved_dependency_closure_event_instance_ids": ids[:2],
        },
    )


def _relation_target_candidate(
    bundle: dict[str, Any],
    family_id: str,
    nodes: list[dict[str, Any]],
    immutable_target: dict[str, Any],
    *,
    require_support: bool,
) -> dict[str, Any]:
    contract = _immutable_target_contract(immutable_target)
    events = _plan_event_rows(bundle)
    if len(events) != 8:
        raise ClaimProofCompileError("immutable relation target requires eight events")
    ids = [_event_id(event) for event in events]
    binder, target, same_entity, wrong_entity = events[:4]
    if (
        _event_dependencies(target) != [ids[0]]
        or any(_event_dependencies(event) for index, event in enumerate(events) if index != 1)
        or _text(binder.get('value')) != _text(target.get('entity'))
        or _text(same_entity.get('entity')) != _text(target.get('entity'))
        or _text(same_entity.get('slot')) != _text(target.get('slot'))
        or _fold(same_entity.get('value')) == _fold(target.get('value'))
        or _text(wrong_entity.get('entity')) == _text(target.get('entity'))
        or _text(wrong_entity.get('slot')) != _text(target.get('slot'))
    ):
        raise ClaimProofCompileError("immutable relation roles are not uniquely realized")
    by_event = _node_claims_by_event(nodes)
    claims = [by_event.get(event_id) for event_id in ids]
    if any(claim is None for claim in claims[:5]):
        raise ClaimProofCompileError("immutable relation event lacks a node claim")
    typed = [claim for claim in claims if isinstance(claim, dict)]
    answer_claim = typed[1]
    answer = answer_claim.get('value')
    if not _atomic_answer_compatible(answer, contract["preferred_answer_type"]):
        raise ClaimProofCompileError("relation value does not match target answer type")
    distractor_claims = [typed[2], typed[3], *typed[4:]]
    options, gold = _ordered_options(
        qa_seed={"target": contract, "binder": ids[0]},
        gold_text=_text(answer),
        gold_claim_id=_claim_id(answer_claim),
        distractors=_option_pool(distractor_claims, _text(answer)),
    )
    reasoning_claims = typed[:4]
    reasoning_ids = [_claim_id(claim) for claim in reasoning_claims]
    closure_ids = reasoning_ids[:2]
    qa = _proof_and_qa(
        bundle=bundle,
        family_id=family_id,
        source_graph_operation="relation_lookup",
        operation="relation_lookup",
        subtype=contract["reasoning_subtype"],
        template=contract["reasoning_template"],
        hop_count=contract["hop_count"],
        answer_projection=ANSWER_PROJECTION_BY_OPERATION["relation_lookup"],
        question_draft=(
            f"For {_text(target.get('entity'))}, identified by {_text(binder.get('entity'))} "
            f"through {_slot_label(binder.get('slot'))}, what is the "
            f"{_slot_label(target.get('slot'))}?"
        ),
        answer_type=contract["preferred_answer_type"],
        gold_answer_text=_text(answer),
        answer_claim_ids=[_claim_id(answer_claim)],
        required_claim_ids=reasoning_ids,
        reasoning_claim_ids=reasoning_ids,
        dependency_closure_claim_ids=closure_ids,
        derivation={
            "rule": "bind_unique_entity_then_read_attribute",
            "binder_claim_ids": [reasoning_ids[0]],
            "target_claim_ids": [reasoning_ids[1]],
            "same_entity_competitor_claim_ids": [reasoning_ids[2]],
            "wrong_entity_claim_ids": [reasoning_ids[3]],
        },
        options=options,
        gold=gold,
        ledger=_ledger_by_claim(bundle),
        require_support=require_support,
    )
    return _apply_immutable_target_contract(qa, immutable_target, contract)


def _visibility_target_candidate(
    bundle: dict[str, Any],
    family_id: str,
    nodes: list[dict[str, Any]],
    immutable_target: dict[str, Any],
    *,
    require_support: bool,
) -> dict[str, Any]:
    contract = _immutable_target_contract(immutable_target)
    if contract["preferred_answer_type"] not in {"Person", "GroupOrOrg"}:
        raise ClaimProofCompileError(
            "visibility Claim-first currently requires an observer answer type"
        )
    events = _plan_event_rows(bundle)
    if len(events) != 8:
        raise ClaimProofCompileError("immutable visibility target requires eight events")
    ids = [_event_id(event) for event in events]
    if _event_dependencies(events[1]) != [ids[0]] or any(
        _event_dependencies(event) for index, event in enumerate(events) if index != 1
    ):
        raise ClaimProofCompileError("immutable visibility graph lacks one resource edge")
    scopes = [event.get("scope") if isinstance(event.get("scope"), dict) else {} for event in events]
    gold_scope = scopes[1]
    fields = ("observer", "permission", "condition")
    if any(not _text(gold_scope.get(field)) for field in fields):
        raise ClaimProofCompileError("visibility gold scope is incomplete")
    for index, field in ((2, "observer"), (3, "permission"), (4, "condition")):
        differences = {
            name
            for name in fields
            if _fold(scopes[index].get(name)) != _fold(gold_scope.get(name))
        }
        if differences != {field}:
            raise ClaimProofCompileError(
                f"visibility event {index + 1} is not the required {field} minimal pair"
            )
    resource_surface = _fold(events[0].get("value") or events[0].get("entity"))
    for event, scope in zip(events[1:], scopes[1:]):
        typed_pairs = (
            (event.get("observer"), scope.get("observer")),
            (
                event.get("action") or event.get("permission"),
                scope.get("permission") or scope.get("action"),
            ),
            (event.get("condition"), scope.get("condition")),
        )
        if any(
            left is not None and _fold(left) != _fold(right)
            for left, right in typed_pairs
        ):
            raise ClaimProofCompileError(
                "visibility top-level typed roles disagree with scope"
            )
        if _fold(event.get("resource") or event.get("entity")) != resource_surface:
            raise ClaimProofCompileError(
                "visibility permission points at a different resource"
            )
        access_class = event.get("access_class")
        if access_class is not None and _fold(access_class) != _fold(event.get("value")):
            raise ClaimProofCompileError(
                "visibility access class disagrees with event value"
            )
    node_by_event = _node_claims_by_event(nodes)
    relation_by_event = _semantic_claims_by_event(
        bundle, "visibility", require_support=require_support
    )
    resource_claim = node_by_event.get(ids[0])
    gold_claim = relation_by_event.get(ids[1])
    if resource_claim is None or gold_claim is None:
        raise ClaimProofCompileError("visibility resource or typed permission claim is missing")
    if _status(gold_claim) not in STAGE64_CURRENT_STATUSES:
        raise ClaimProofCompileError("visibility gold permission is not active")
    active_observers = {
        _fold(scope.get("observer"))
        for event, scope in zip(events[1:], scopes[1:])
        if _fold(event.get("status")) in STAGE64_CURRENT_STATUSES
        and _fold(event.get("slot")) == _fold(events[1].get("slot"))
        and _fold(event.get("value")) == _fold(events[1].get("value"))
    }
    if active_observers != {_fold(gold_claim.get("observer"))}:
        raise ClaimProofCompileError(
            "visibility scope does not yield one active observer projection"
        )
    distinct: list[dict[str, Any]] = [gold_claim]
    seen = {_fold(gold_claim.get("observer"))}
    for event_id in ids[2:]:
        claim = relation_by_event.get(event_id)
        observer = _fold((claim or {}).get("observer"))
        if claim is not None and observer and observer not in seen:
            seen.add(observer)
            distinct.append(claim)
        if len(distinct) == 4:
            break
    if len(distinct) != 4:
        raise ClaimProofCompileError(
            "visibility target lacks four distinct graph-grounded observer options"
        )
    answer = _text(gold_claim.get("observer"))
    options, gold = _ordered_options(
        qa_seed={"target": contract, "scope": gold_scope},
        gold_text=answer,
        gold_claim_id=_claim_id(gold_claim),
        distractors=[
            {
                "text": _text(claim.get("observer")),
                "truth_value": False,
                "source_claim_id": _claim_id(claim),
                "distractor_origin": "typed_visibility_scope_competitor",
            }
            for claim in distinct[1:]
        ],
    )
    required_ids = [_claim_id(resource_claim), _claim_id(gold_claim)]
    reasoning_claims = [resource_claim, *distinct]
    reasoning_ids = [_claim_id(claim) for claim in reasoning_claims]
    resource = _text(gold_claim.get("resource") or events[0].get('value') or events[0].get('entity'))
    qa = _proof_and_qa(
        bundle=bundle,
        family_id=family_id,
        source_graph_operation="visibility",
        operation="visibility",
        subtype=contract["reasoning_subtype"],
        template=contract["reasoning_template"],
        hop_count=contract["hop_count"],
        answer_projection=ANSWER_PROJECTION_BY_OPERATION["visibility"],
        question_draft=(
            f"Under the condition {_text(gold_scope.get('condition'))}, who has "
            f"{_text(gold_scope.get('permission'))} access to {resource}?"
        ),
        answer_type=contract["preferred_answer_type"],
        gold_answer_text=answer,
        answer_claim_ids=[_claim_id(gold_claim)],
        required_claim_ids=required_ids,
        reasoning_claim_ids=reasoning_ids,
        dependency_closure_claim_ids=required_ids,
        derivation={
            "rule": "resolve_observer_under_permission_scope",
            "answer_claim_ids": [_claim_id(gold_claim)],
            "resource_claim_ids": [_claim_id(resource_claim)],
            "scope": copy.deepcopy(gold_scope),
        },
        options=options,
        gold=gold,
        ledger=_ledger_by_claim(bundle),
        require_support=require_support,
    )
    return _apply_immutable_target_contract(
        qa,
        immutable_target,
        contract,
        query_plan_updates={
            "filter_predicate": {
                field: copy.deepcopy(gold_scope.get(field)) for field in fields
            },
        },
    )


def _immutable_target_candidate(
    bundle: dict[str, Any],
    family_id: str,
    nodes: list[dict[str, Any]],
    immutable_target: dict[str, Any],
    *,
    require_support: bool,
) -> dict[str, Any]:
    operation = _text(immutable_target.get("qa_operation"))
    if _source_graph_operation(bundle) != operation:
        raise ClaimProofCompileError(
            "immutable target operation does not match frozen graph operation"
        )
    compilers = {
        "temporal_state_update": _temporal_target_candidate,
        "bridge": _bridge_target_candidate,
        "comparison": _comparison_target_candidate,
        "visibility": _visibility_target_candidate,
        "count": _count_target_candidate,
        "cause_consequence": _cause_target_candidate,
        "relation_lookup": _relation_target_candidate,
    }
    compiler = compilers.get(operation)
    if compiler is None:
        raise ClaimProofCompileError(
            f"immutable target compiler has no rule for operation {operation!r}"
        )
    return compiler(
        bundle,
        family_id,
        nodes,
        immutable_target,
        require_support=require_support,
    )


def _temporal_target_candidate(
    bundle: dict[str, Any],
    family_id: str,
    nodes: list[dict[str, Any]],
    immutable_target: dict[str, Any],
    *,
    require_support: bool,
) -> dict[str, Any]:
    """Compile the current active value while retaining a later invalid update.

    This is deliberately status-generic: the target strategy supplies the
    invalid status family.  It never treats canceled/outdated/reported values
    as current truth and never infers truth from maximum time alone.
    """

    contract = _immutable_target_contract(immutable_target)
    if contract["qa_operation"] != "temporal_state_update":
        raise ClaimProofCompileError(
            "immutable target compiler currently has no truthful graph rule for "
            f"operation {contract['qa_operation']!r}"
        )
    if _source_graph_operation(bundle) != contract["qa_operation"]:
        raise ClaimProofCompileError(
            "immutable target operation does not match frozen graph operation"
        )
    required_invalid_statuses = TEMPORAL_HARD_NEGATIVE_STATUSES.get(
        contract["hard_negative_strategy"]
    )
    if not required_invalid_statuses or not contract["forbid_max_time_shortcut"]:
        raise ClaimProofCompileError(
            "temporal immutable target requires a known anti-max-time strategy"
        )
    plan_by_id = _plan_events_by_id(bundle)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for claim in nodes:
        groups[(_fold(claim.get("entity")), _fold(claim.get("slot")))].append(claim)

    valid_paths: list[list[dict[str, Any]]] = []
    for group in groups.values():
        ordered = sorted(
            group,
            key=lambda item: (
                int(item.get("global_time_index") or item.get("time_index") or 0),
                _claim_id(item),
            ),
        )
        if len(ordered) < contract["hop_count"]:
            continue
        for path_tuple in itertools.combinations(ordered, contract["hop_count"]):
            path = list(path_tuple)
            active = [
                item for item in path if _status(item) in STAGE64_CURRENT_STATUSES
            ]
            if len(active) != 1:
                continue
            active_index = path.index(active[0])
            later_invalid = [
                item
                for item in path[active_index + 1 :]
                if _status(item) in required_invalid_statuses
            ]
            if not later_invalid:
                continue
            if any(
                _status(item) in STAGE64_CURRENT_STATUSES
                for item in path
                if item is not active[0]
            ):
                continue
            if len({_fold(item.get("value")) for item in path}) != len(path):
                continue
            event_ids = claim_event_mapping(
                bundle, [_claim_id(item) for item in path]
            )
            path_event_ids = [event_ids[_claim_id(item)] for item in path]
            if any(event_id not in plan_by_id for event_id in path_event_ids):
                continue
            # Stage64's legacy quality gate requires the resolved reasoning path
            # to retain the actual dependency chain, not merely chronological IDs.
            if any(
                previous_id
                not in {
                    _text(value)
                    for value in (plan_by_id[current_id].get("depends_on") or [])
                }
                for previous_id, current_id in zip(
                    path_event_ids, path_event_ids[1:]
                )
            ):
                continue
            valid_paths.append(path)
    if len(valid_paths) != 1:
        raise ClaimProofCompileError(
            "immutable temporal target does not resolve to exactly one "
            f"status/dependency path (found {len(valid_paths)})"
        )

    path = valid_paths[0]
    active_claim = next(
        item for item in path if _status(item) in STAGE64_CURRENT_STATUSES
    )
    active_claim_id = _claim_id(active_claim)
    hard_negative_claim_ids = [
        _claim_id(item)
        for item in path[path.index(active_claim) + 1 :]
        if _status(item) in required_invalid_statuses
    ]
    path_claim_ids = [_claim_id(item) for item in path]
    distractors = [
        {
            "text": _text(item.get("value")),
            "truth_value": False,
            "source_claim_id": _claim_id(item),
            "distractor_origin": (
                "immutable_target_hard_negative"
                if _claim_id(item) in hard_negative_claim_ids
                else "same_entity_slot_temporal_update"
            ),
        }
        for item in path
        if item is not active_claim
    ]
    options, gold = _ordered_options(
        qa_seed={
            "immutable_target": contract,
            "path_claim_ids": path_claim_ids,
        },
        gold_text=_text(active_claim.get("value")),
        gold_claim_id=active_claim_id,
        distractors=distractors,
        desired_count=contract["hop_count"],
    )
    invalid_label = {
        "same_slot_canceled_value": "canceled proposal",
        "same_slot_outdated_value": "outdated update",
        "same_slot_report_or_belief_value": "unconfirmed report",
    }[contract["hard_negative_strategy"]]
    qa = _proof_and_qa(
        bundle=bundle,
        family_id=family_id,
        source_graph_operation=_source_graph_operation(bundle),
        operation=contract["qa_operation"],
        subtype=contract["reasoning_subtype"],
        template=contract["reasoning_template"],
        hop_count=contract["hop_count"],
        answer_projection="current_value",
        question_draft=(
            f"After accounting for the {invalid_label}, what is "
            f"{_text(active_claim.get('entity'))}'s current "
            f"{_slot_label(active_claim.get('slot'))}?"
        ),
        answer_type=contract["preferred_answer_type"],
        gold_answer_text=_text(active_claim.get("value")),
        answer_claim_ids=[active_claim_id],
        required_claim_ids=path_claim_ids,
        reasoning_claim_ids=path_claim_ids,
        dependency_closure_claim_ids=path_claim_ids,
        derivation={
            "rule": "reject_targeted_noncurrent_update_then_resolve_active_value",
            "candidate_claim_ids": path_claim_ids,
            "target_claim_ids": [active_claim_id],
            "hard_negative_claim_ids": hard_negative_claim_ids,
            "hard_negative_statuses": sorted(required_invalid_statuses),
            "immutable_stage64_target": copy.deepcopy(contract),
        },
        options=options,
        gold=gold,
        ledger=_ledger_by_claim(bundle),
        require_support=require_support,
    )
    qa["reasoning_subtype"] = contract["reasoning_subtype"]
    qa["reasoning_template"] = contract["reasoning_template"]
    qa["hard_negative_strategy"] = contract["hard_negative_strategy"]
    qa["forbid_max_time_shortcut"] = contract["forbid_max_time_shortcut"]
    qa["fulfills_immutable_stage64_target"] = True
    qa["immutable_stage64_target"] = copy.deepcopy(immutable_target)
    qa["target_fulfillment_match"] = copy.deepcopy(contract)
    qa["operation_specific_target"] = {
        **copy.deepcopy(qa["operation_specific_target"]),
        **copy.deepcopy(contract),
        "immutable": True,
    }
    qa["query_plan"].update(
        {
            "reasoning_subtype": contract["reasoning_subtype"],
            "reasoning_template": contract["reasoning_template"],
            "hop_count": contract["hop_count"],
            "hard_negative_strategy": contract["hard_negative_strategy"],
            "forbid_max_time_shortcut": contract["forbid_max_time_shortcut"],
        }
    )
    return qa


def compile_claim_first_qas(
    bundle: dict[str, Any],
    *,
    seed_sample_id: str = "",
    lineage_group_id: str = "",
    min_qas: int = 3,
    max_qas: int = 6,
    eval_per_mille: int = 100,
    require_support: bool = True,
    immutable_stage64_target: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile 3--6 truth-frozen QA skeletons from one supported claim bundle.

    The selection favors operation diversity, then proof diversity.  Failure to
    reach ``min_qas`` rejects the dialogue family before an API call.  When an
    immutable Stage64 target is supplied, one truthful target-fulfillment QA is
    compiled and reserved inside ``max_qas``; target fields are never inferred
    from or rewritten to match an unrelated proof.
    """

    if min_qas < 1 or max_qas < min_qas:
        raise ValueError("invalid min_qas/max_qas")
    nodes = _eligible_nodes(bundle, require_support=require_support)
    if not nodes:
        raise ClaimProofCompileError("no supported qa-eligible node claims")
    family_id = dialogue_family_id(
        bundle, seed_sample_id=seed_sample_id, lineage_group_id=lineage_group_id
    )
    target_candidate = (
        _immutable_target_candidate(
            bundle,
            family_id,
            nodes,
            immutable_stage64_target,
            require_support=require_support,
        )
        if immutable_stage64_target is not None
        else None
    )
    candidates = (
        ([target_candidate] if target_candidate is not None else [])
        + _comparison_candidates(
            bundle, family_id, nodes, require_support=require_support
        )
        + _status_lookup_candidates(
            bundle, family_id, nodes, require_support=require_support
        )
        + _relation_lookup_candidates(
            bundle, family_id, nodes, require_support=require_support
        )
        + _direct_attribute_candidates(
            bundle, family_id, nodes, require_support=require_support
        )
    )
    # Deduplicate identical proof paths and interleave operations so the first
    # N questions are not merely paraphrases of one template.
    unique: list[dict[str, Any]] = []
    seen_paths: set[tuple[str, tuple[str, ...], str]] = set()
    for item in candidates:
        proof = item["proof"]
        key = (
            item["qa_subtype"],
            tuple(proof["required_claim_ids"]),
            _fold(item["gold_answer_text"]),
        )
        if key not in seen_paths:
            seen_paths.add(key)
            unique.append(item)
    by_operation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in unique:
        by_operation[item["qa_operation"]].append(item)
    selected: list[dict[str, Any]] = []
    used_claim_counts: dict[str, int] = defaultdict(int)
    used_subtype_counts: dict[str, int] = defaultdict(int)
    used_gold_counts: dict[str, int] = defaultdict(int)
    if target_candidate is not None:
        target_matches = [
            item
            for item in by_operation[target_candidate["qa_operation"]]
            if item["qa_id"] == target_candidate["qa_id"]
        ]
        if len(target_matches) != 1:
            raise ClaimProofCompileError(
                "immutable target fulfillment QA was lost or duplicated during deduplication"
            )
        reserved = target_matches[0]
        by_operation[reserved["qa_operation"]] = [
            item
            for item in by_operation[reserved["qa_operation"]]
            if item["qa_id"] != reserved["qa_id"]
        ]
        selected.append(reserved)
        for claim_id in reserved["proof"]["required_claim_ids"]:
            used_claim_counts[claim_id] += 1
        used_subtype_counts[reserved["qa_subtype"]] += 1
        used_gold_counts[_fold(reserved["gold_answer_text"])] += 1
    operation_order = ["comparison", "temporal_state_update", "relation_lookup"]
    while len(selected) < max_qas and any(by_operation.values()):
        progressed = False
        for operation in operation_order:
            if by_operation[operation] and len(selected) < max_qas:
                candidates_for_operation = by_operation[operation]

                def diversity_key(
                    item: dict[str, Any]
                ) -> tuple[int, int, int, int, str]:
                    claim_ids = item["proof"]["required_claim_ids"]
                    repeated_claim_cost = sum(
                        used_claim_counts[claim_id] for claim_id in claim_ids
                    )
                    subtype = item["qa_subtype"]
                    # Prefer multi-claim proofs when diversity cost is tied;
                    # one-hop proofs remain an explicit sparse-graph fallback.
                    hop_preference = -min(int(item.get("hop_count") or 0), 2)
                    return (
                        repeated_claim_cost,
                        used_gold_counts[_fold(item["gold_answer_text"])],
                        used_subtype_counts[subtype],
                        hop_preference,
                        item["qa_id"],
                    )

                chosen_index = min(
                    range(len(candidates_for_operation)),
                    key=lambda index: diversity_key(candidates_for_operation[index]),
                )
                chosen = candidates_for_operation.pop(chosen_index)
                selected.append(chosen)
                for claim_id in chosen["proof"]["required_claim_ids"]:
                    used_claim_counts[claim_id] += 1
                used_subtype_counts[chosen["qa_subtype"]] += 1
                used_gold_counts[_fold(chosen["gold_answer_text"])] += 1
                progressed = True
        if not progressed:
            break
    if len(selected) < min_qas:
        raise ClaimProofCompileError(
            f"only {len(selected)} unambiguous QA proofs; need at least {min_qas}"
        )
    split = assign_family_split(family_id, eval_per_mille=eval_per_mille)
    for item in selected:
        item["dialogue_family_split"] = split
    result = {
        "schema_version": SCHEMA_VERSION,
        "dialogue_family_id": family_id,
        "dialogue_family_split": split,
        "graph_snapshot_hash": _text(bundle.get("graph_snapshot_hash")),
        "claim_bundle_hash": _bundle_hash(bundle),
        "source_graph_operation": _source_graph_operation(bundle),
        "qa_count": len(selected),
        "qa_operation_counts": dict(
            sorted(
                (operation, sum(item["qa_operation"] == operation for item in selected))
                for operation in {item["qa_operation"] for item in selected}
            )
        ),
        "qa_subtype_counts": dict(
            sorted(
                (subtype, sum(item["qa_subtype"] == subtype for item in selected))
                for subtype in {item["qa_subtype"] for item in selected}
            )
        ),
        "unique_question_draft_count": len(
            {_fold(item["question_text_draft"]) for item in selected}
        ),
        "unique_gold_answer_count": len(
            {_fold(item["gold_answer_text"]) for item in selected}
        ),
        "unique_reasoning_path_count": len(
            {
                tuple(item["proof"]["reasoning_path_claim_ids"])
                for item in selected
            }
        ),
        "one_hop_qa_count": sum(
            int(item.get("hop_count") == 1) for item in selected
        ),
        "qas": selected,
        "api_calls_made": 0,
        "semantic_contract_frozen": True,
        "support_binding_status": "bound" if require_support else "pending",
    }
    if target_candidate is not None:
        fulfillment_ids = [
            item["qa_id"]
            for item in selected
            if item.get("fulfills_immutable_stage64_target") is True
        ]
        if fulfillment_ids != [target_candidate["qa_id"]]:
            raise ClaimProofCompileError(
                "immutable Stage64 target must have exactly one fulfillment QA"
            )
        result.update(
            {
                "immutable_stage64_target": copy.deepcopy(
                    immutable_stage64_target
                ),
                "target_fulfillment_qa_id": fulfillment_ids[0],
                "target_fulfillment_status": "compiled",
            }
        )
    return result


def bind_compiled_qas_to_support(
    compiled: dict[str, Any], supported_bundle: dict[str, Any]
) -> dict[str, Any]:
    """Bind exact dialogue support without changing any frozen QA semantics.

    Each QA is bound independently. A missing/invalid support row rejects only
    that QA and is retained in ``qa_support_rejects``; it never deletes the
    dialogue or sibling QAs.
    """

    expected_hash = _text(compiled.get("claim_bundle_hash"))
    actual_hash = _bundle_hash(supported_bundle)
    if not expected_hash or expected_hash != actual_hash:
        raise ClaimProofCompileError(
            "supported bundle does not match the frozen pre-dialogue claim bundle"
        )
    if _text(compiled.get("graph_snapshot_hash")) != _text(
        supported_bundle.get("graph_snapshot_hash")
    ):
        raise ClaimProofCompileError("graph snapshot changed before support binding")
    ledger = _ledger_by_claim(supported_bundle)
    accepted: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    for raw_qa in compiled.get("qas") or []:
        qa = copy.deepcopy(raw_qa)
        qa_id = _text(qa.get("qa_id"))
        required = list((qa.get("proof") or {}).get("required_claim_ids") or [])
        try:
            evidence = _support_subset(ledger, required)
        except ClaimProofCompileError as exc:
            rejects.append(
                {
                    "qa_id": qa_id,
                    "reason": "required_claim_support_missing",
                    "error": str(exc),
                    "required_claim_ids": required,
                }
            )
            continue
        qa["evidence_ledger"] = evidence
        qa["support_binding_status"] = "bound"
        accepted.append(qa)
    result = copy.deepcopy(compiled)
    result["qas"] = accepted
    result["qa_count"] = len(accepted)
    result["pre_dialogue_qa_count"] = len(compiled.get("qas") or [])
    result["qa_support_reject_count"] = len(rejects)
    result["qa_support_rejects"] = rejects
    result["dialogue_retained"] = True
    result["support_binding_status"] = (
        "bound_all" if not rejects else "bound_partial"
    )
    target_qa_id = _text(result.get("target_fulfillment_qa_id"))
    if target_qa_id:
        accepted_ids = {_text(item.get("qa_id")) for item in accepted}
        if target_qa_id in accepted_ids:
            result["target_fulfillment_status"] = "support_bound"
        else:
            result["target_fulfillment_status"] = "support_rejected"
    result["api_calls_made"] = 0
    return result


__all__ = [
    "ClaimProofCompileError",
    "assign_family_split",
    "bind_compiled_qas_to_support",
    "claim_event_mapping",
    "compile_claim_first_qas",
    "dialogue_family_id",
]
