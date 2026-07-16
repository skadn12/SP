#!/usr/bin/env python3
"""Update Stage3 challenger operation shares from audited cyclic solver results."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts.current_method import build_stage3_parent_v2, semantic_graph_v2, stage64_quality
except ModuleNotFoundError:
    from current_method import build_stage3_parent_v2, semantic_graph_v2, stage64_quality


SCHEMA_VERSION = "stage3_operation_controller_v5"
GENERATION_PROTOCOL_SCHEMA_VERSION = "stage3_generation_protocol_v1"
REQUIRED_GRAPH_GATE_VERSION = "semantic_graph_v17"
REQUIRED_NATIVE_GENERATOR_VERSION = "graph_conditioned_api_dialogue_v1"
STAGE64_TARGET_IDENTITY_VERSION = "stage64_target_identity_v2"
REASONING_SIGNATURE_FIELDS = (
    "reasoning_subtype",
    "reasoning_template",
    "hop_count",
    "hard_negative_strategy",
)
AUDIT_STATUSES = (
    "robust_frontier",
    "robust_easy5",
    "robust_hard0",
    "position_sensitive",
)
OPERATIONS = (
    "temporal_state_update",
    "bridge",
    "comparison",
    "visibility",
    "count",
    "cause_consequence",
    "relation_lookup",
)
DEFAULT_OPERATION_WEIGHTS = {
    "temporal_state_update": 30.0,
    "bridge": 18.0,
    "comparison": 18.0,
    "visibility": 18.0,
    "count": 12.0,
    "cause_consequence": 12.0,
    "relation_lookup": 12.0,
}
REQUIRED_ANSWER_TYPES = {"comparison": ("Number",), "count": ("Number",)}
ANSWER_TYPE_CANDIDATES = {
    "temporal_state_update": (
        "LocationOrScope",
        "DateOrTime",
        "Event",
        "ObjectOrCommonNoun",
        "AttributeValue",
    ),
    "bridge": (
        "Person",
        "GroupOrOrg",
        "LocationOrScope",
        "Event",
        "ObjectOrCommonNoun",
        "ProperNounOrNamedEntity",
    ),
    "comparison": ("Number",),
    "visibility": ("Person", "GroupOrOrg", "LocationOrScope", "YesNo"),
    "count": ("Number",),
    "cause_consequence": (
        "Person",
        "GroupOrOrg",
        "YesNo",
        "Event",
        "ObjectOrCommonNoun",
        "AttributeValue",
    ),
    "relation_lookup": (
        "Person",
        "GroupOrOrg",
        "LocationOrScope",
        "DateOrTime",
        "Event",
        "ObjectOrCommonNoun",
        "AttributeValue",
        "ProperNounOrNamedEntity",
    ),
}
EASY5_MAJORITY_THRESHOLD = 0.60
LOW_FRONTIER_YIELD_THRESHOLD = 0.20
HIGH_HARD0_THRESHOLD = 0.40


def _source_fingerprint(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for source in paths:
        source = source.resolve()
        if source.suffix == ".pyc" and source.with_suffix(".py").exists():
            source = source.with_suffix(".py")
        digest.update(source.read_bytes())
    return digest.hexdigest()


def _quality_gate_source_fingerprint() -> str:
    return _source_fingerprint(Path(str(module.__file__)) for module in (semantic_graph_v2, stage64_quality))


def _native_generation_source_fingerprint() -> str:
    return _source_fingerprint(
        [Path(__file__).resolve().with_name("native_api_graph_dialogue_expansion.py")]
    )


def generation_protocol_payload(
    *,
    graph_gate_version: str = REQUIRED_GRAPH_GATE_VERSION,
    native_generator_version: str = REQUIRED_NATIVE_GENERATOR_VERSION,
) -> dict[str, Any]:
    """Return the semantic-generation contract accepted by this controller."""

    return {
        "schema_version": GENERATION_PROTOCOL_SCHEMA_VERSION,
        "graph_gate_version": str(graph_gate_version),
        "semantic_graph_schema_version": semantic_graph_v2.SCHEMA_VERSION,
        "semantic_quality_gate": "stage64_quality.semantic_quality_issues",
        "semantic_quality_gate_fingerprint": _quality_gate_source_fingerprint(),
        "native_generator_version": str(native_generator_version),
        "native_generation_source_fingerprint": _native_generation_source_fingerprint(),
        "parent_dialogue_contract": {
            "historical_deterministic_padding_allowed": False,
            "cleanup_version": build_stage3_parent_v2.DETERMINISTIC_PADDING_CLEANUP_VERSION,
            "known_padding_signature_sha256": (
                build_stage3_parent_v2.deterministic_padding_signature_sha256()
            ),
        },
        "solver_audit": {
            "permutation_mode": "cyclic",
            "k_eval_per_permutation": 5,
            "required_frontier_permutation_share": 1.0,
        },
    }


def protocol_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def generation_protocol(
    *,
    graph_gate_version: str = REQUIRED_GRAPH_GATE_VERSION,
    native_generator_version: str = REQUIRED_NATIVE_GENERATOR_VERSION,
) -> dict[str, Any]:
    payload = generation_protocol_payload(
        graph_gate_version=graph_gate_version,
        native_generator_version=native_generator_version,
    )
    return {**payload, "fingerprint": protocol_fingerprint(payload)}


def attach_generation_protocol(
    sample: dict[str, Any],
    *,
    graph_gate_version: str,
) -> dict[str, Any]:
    """Attach audited generation provenance before shuffle/recalibration."""

    native_version = str(sample.get("native_generator_version") or "")
    sample["stage3_generation_protocol"] = generation_protocol(
        graph_gate_version=graph_gate_version,
        native_generator_version=native_version,
    )
    return sample


def validate_generation_protocol(sample: dict[str, Any]) -> dict[str, Any]:
    protocol = sample.get("stage3_generation_protocol")
    if not isinstance(protocol, dict):
        raise ValueError("Missing stage3_generation_protocol fingerprint")
    fingerprint = str(protocol.get("fingerprint") or "")
    payload = {key: value for key, value in protocol.items() if key != "fingerprint"}
    if not fingerprint or fingerprint != protocol_fingerprint(payload):
        raise ValueError("Invalid stage3_generation_protocol fingerprint")
    expected = generation_protocol()
    if protocol != expected:
        raise ValueError(
            "Unsupported Stage3 generation protocol: "
            f"graph_gate_version={protocol.get('graph_gate_version')!r}, "
            f"native_generator_version={protocol.get('native_generator_version')!r}; "
            f"expected_graph_gate_version={REQUIRED_GRAPH_GATE_VERSION!r}"
        )
    return protocol


def validate_controller_sample(sample: dict[str, Any]) -> dict[str, Any]:
    protocol = validate_generation_protocol(sample)
    issues = stage64_quality.semantic_quality_issues(sample)
    if issues:
        reasons = sorted({str(issue.get("reason") or "unknown") for issue in issues})
        raise ValueError(
            f"Stage3 controller sample failed current semantic quality: {reasons}; "
            f"sample_id={sample.get('sample_id')!r}"
        )
    return protocol


def validate_state_protocol(state: dict[str, Any]) -> None:
    if state.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            "Stage3 operation-controller state schema is missing or stale; "
            f"expected {SCHEMA_VERSION}. Start a new state path."
        )
    expected = generation_protocol()
    if state.get("generation_protocol") != expected:
        raise RuntimeError(
            "Stage3 operation-controller state generation protocol is missing or stale; "
            "start a new state path for semantic_graph_v17 results."
        )


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def normalized_shares(weights: dict[str, float], keys: Iterable[str]) -> dict[str, float]:
    ordered = list(keys)
    nonnegative = {key: max(0.0, float(weights.get(key) or 0.0)) for key in ordered}
    total = sum(nonnegative.values())
    if total <= 0.0:
        return {key: round(1.0 / len(ordered), 9) for key in ordered}
    raw = {key: value / total for key, value in nonnegative.items()}
    # Leave the final key as the residual so serialized shares sum exactly to one.
    output: dict[str, float] = {}
    running = 0.0
    for key in ordered[:-1]:
        output[key] = round(raw[key], 9)
        running += output[key]
    output[ordered[-1]] = round(1.0 - running, 9)
    return output


def shares_with_floor(
    scores: dict[str, float],
    keys: Iterable[str],
    *,
    minimum_share: float,
) -> dict[str, float]:
    ordered = list(keys)
    if not ordered:
        return {}
    floor = float(minimum_share)
    if floor < 0.0 or floor * len(ordered) >= 1.0:
        raise ValueError("minimum_share must be non-negative and leave positive allocation mass")
    residual = 1.0 - floor * len(ordered)
    weighted = normalized_shares(scores, ordered)
    raw = {key: floor + residual * weighted[key] for key in ordered}
    return normalized_shares(raw, ordered)


def _formal_v2_reasoning_target(sample: dict[str, Any], target: dict[str, Any]) -> bool:
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    declared_schema = str(
        sample.get("stage64_target_identity_schema_version")
        or metadata.get("stage64_target_identity_schema_version")
        or ""
    )
    has_identity = bool(
        sample.get("stage64_target_identity")
        or sample.get("stage64_target_id")
        or metadata.get("stage64_target_identity")
        or metadata.get("stage64_target_id")
    )
    return declared_schema == STAGE64_TARGET_IDENTITY_VERSION or (
        has_identity and any(field in target for field in REASONING_SIGNATURE_FIELDS)
    )


def sample_reasoning_signature(sample: dict[str, Any]) -> dict[str, Any] | None:
    """Return the immutable Stage64 reasoning signature, or ``None`` for legacy rows.

    ``stage64_target`` is the only authority. Rows produced under a Stage3
    generation protocol, or rows that contain any reasoning-target field, are
    fail-closed. Legacy non-formal rows with neither signal remain reportable as
    untyped data rather than receiving a fabricated post-hoc signature.
    """

    target = sample.get("stage64_target")
    target_object = target if isinstance(target, dict) else {}
    target_has_reasoning = any(
        field in target_object for field in REASONING_SIGNATURE_FIELDS
    )
    strict = isinstance(sample.get("stage3_generation_protocol"), dict) or target_has_reasoning
    formal_v2 = _formal_v2_reasoning_target(sample, target_object)
    if not strict and not formal_v2:
        return None
    if not isinstance(target, dict):
        raise ValueError(
            "Reasoning-signature audit requires authoritative sample.stage64_target"
        )

    operation = target.get("qa_operation")
    if not isinstance(operation, str) or not operation.strip():
        raise ValueError("stage64_target has missing or invalid qa_operation")
    if operation not in OPERATIONS:
        raise ValueError(f"Unsupported stage64_target qa_operation: {operation!r}")
    signature: dict[str, Any] = {"operation": operation}
    for field in REASONING_SIGNATURE_FIELDS:
        value = target.get(field)
        if field == "hop_count":
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"stage64_target has missing or invalid hop_count: {value!r}"
                )
        elif not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"stage64_target has missing or invalid {field}: {value!r}"
            )
        signature[field] = value

    qa = sample.get("qa") if isinstance(sample.get("qa"), dict) else {}
    qa_query_plan = qa.get("query_plan") if isinstance(qa.get("query_plan"), dict) else {}
    top_query_plan = (
        sample.get("query_plan") if isinstance(sample.get("query_plan"), dict) else {}
    )
    locations = (
        ("top-level", sample),
        ("qa", qa),
        ("qa.query_plan", qa_query_plan),
        ("query_plan", top_query_plan),
    )
    target_values = {"qa_operation": operation, **{field: signature[field] for field in REASONING_SIGNATURE_FIELDS}}
    for location_name, payload in locations:
        for field, expected in target_values.items():
            if field in payload and payload.get(field) != expected:
                raise ValueError(
                    "Reasoning-signature drift from stage64_target: "
                    f"{location_name}.{field}={payload.get(field)!r}, expected={expected!r}"
                )
    return signature


def reasoning_signature_key(signature: dict[str, Any]) -> str:
    """Serialize a reasoning signature as a stable, human-readable metric key."""

    return "|".join(
        (
            str(signature["operation"]),
            str(signature["reasoning_subtype"]),
            str(signature["reasoning_template"]),
            f"hop={signature['hop_count']}",
            str(signature["hard_negative_strategy"]),
        )
    )


def sample_cell(sample: dict[str, Any]) -> tuple[str, str]:
    qa = sample.get("qa") if isinstance(sample.get("qa"), dict) else {}
    signature = sample_reasoning_signature(sample)
    operation = (
        str(signature["operation"])
        if signature is not None
        else str(sample.get("qa_operation") or qa.get("qa_operation") or "").strip()
    )
    answer_type = str(sample.get("answer_type") or qa.get("answer_type") or "").strip()
    if operation not in OPERATIONS:
        raise ValueError(f"Unsupported or missing qa_operation: {operation!r}")
    if not answer_type:
        raise ValueError(f"Missing answer_type for operation {operation!r}")
    return operation, answer_type


def audited_outcome(sample: dict[str, Any]) -> tuple[str, int, int, int]:
    audit = sample.get("stage64_permutation_eval")
    if not isinstance(audit, dict):
        raise ValueError("Missing stage64_permutation_eval")
    status = str(audit.get("status") or "")
    if status not in AUDIT_STATUSES:
        raise ValueError(f"Unsupported cyclic aggregate status: {status!r}")
    permutations = audit.get("permutations")
    if not isinstance(permutations, list) or not permutations:
        raise ValueError("Missing evaluated cyclic permutations")
    k_eval = int(audit.get("k_eval_per_permutation") or 0)
    if k_eval != 5:
        raise ValueError(f"Stage3 controller requires k=5 cyclic audits, got {k_eval}")
    if float(audit.get("required_frontier_permutation_share") or 0.0) != 1.0:
        raise ValueError("Stage3 controller requires strict all-position frontier share 1.0")
    counts: list[int] = []
    for permutation in permutations:
        if not isinstance(permutation, dict):
            raise ValueError("Invalid cyclic permutation audit")
        value = permutation.get("exact_correct_count")
        if not isinstance(value, int) or not 0 <= value <= k_eval:
            raise ValueError(f"Invalid cyclic exact_correct_count: {value!r}")
        counts.append(value)
    return status, sum(counts), len(counts) * k_eval, len(counts)


def _materialize_metrics(
    counts: dict[str, Counter[str]],
    exact_rollout_sums: Counter[str],
    rollout_totals: Counter[str],
    *,
    signatures: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for key, values in sorted(counts.items()):
        total = int(values.get("total") or 0)
        exact_rollouts = int(exact_rollout_sums.get(key) or 0)
        evaluated_rollouts = int(rollout_totals.get(key) or 0)
        weighted_accuracy = (
            round(exact_rollouts / evaluated_rollouts, 6)
            if evaluated_rollouts
            else None
        )
        status_counts = {
            status: int(values.get(status) or 0) for status in AUDIT_STATUSES
        }
        record: dict[str, Any] = {
            "total": total,
            **status_counts,
            "status_counts": status_counts,
            "exact_correct_rollout_count": exact_rollouts,
            "evaluated_rollout_count": evaluated_rollouts,
            "weighted_rollout_exact_accuracy": weighted_accuracy,
            # Kept as a compatibility alias. K is fixed, so this is the same
            # rollout-weighted statistic rather than an average of batch means.
            "mean_permutation_soft_accuracy": weighted_accuracy,
            "frontier_yield": (
                round(status_counts["robust_frontier"] / total, 6)
                if total
                else None
            ),
        }
        if signatures is not None:
            record["signature"] = dict(signatures[key])
        output[key] = record
    return output


def summarize_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    cell_counts: dict[str, Counter[str]] = defaultdict(Counter)
    operation_counts: dict[str, Counter[str]] = defaultdict(Counter)
    signature_counts: dict[str, Counter[str]] = defaultdict(Counter)
    cell_exact_rollouts: Counter[str] = Counter()
    operation_exact_rollouts: Counter[str] = Counter()
    signature_exact_rollouts: Counter[str] = Counter()
    cell_rollout_totals: Counter[str] = Counter()
    operation_rollout_totals: Counter[str] = Counter()
    signature_rollout_totals: Counter[str] = Counter()
    signatures: dict[str, dict[str, Any]] = {}
    answer_types: dict[str, set[str]] = defaultdict(set)
    permutation_count = 0
    for sample in samples:
        validate_controller_sample(sample)
        operation, answer_type = sample_cell(sample)
        signature = sample_reasoning_signature(sample)
        if signature is None:
            raise ValueError(
                "Stage3 controller refuses legacy rows without a reasoning signature"
            )
        signature_key = reasoning_signature_key(signature)
        status, exact_rollouts, evaluated_rollouts, permutations = audited_outcome(sample)
        cell = f"{operation}|{answer_type}"
        cell_counts[cell][status] += 1
        operation_counts[operation][status] += 1
        signature_counts[signature_key][status] += 1
        cell_counts[cell]["total"] += 1
        operation_counts[operation]["total"] += 1
        signature_counts[signature_key]["total"] += 1
        cell_exact_rollouts[cell] += exact_rollouts
        operation_exact_rollouts[operation] += exact_rollouts
        signature_exact_rollouts[signature_key] += exact_rollouts
        cell_rollout_totals[cell] += evaluated_rollouts
        operation_rollout_totals[operation] += evaluated_rollouts
        signature_rollout_totals[signature_key] += evaluated_rollouts
        signatures[signature_key] = signature
        answer_types[operation].add(answer_type)
        permutation_count += permutations

    return {
        "sample_count": len(samples),
        "generation_protocol_fingerprint": generation_protocol()["fingerprint"],
        "evaluated_permutation_count": permutation_count,
        "operations": _materialize_metrics(
            operation_counts,
            operation_exact_rollouts,
            operation_rollout_totals,
        ),
        "cells": _materialize_metrics(
            cell_counts,
            cell_exact_rollouts,
            cell_rollout_totals,
        ),
        "reasoning_signatures": _materialize_metrics(
            signature_counts,
            signature_exact_rollouts,
            signature_rollout_totals,
            signatures=signatures,
        ),
        "observed_answer_types": {
            operation: sorted(values) for operation, values in sorted(answer_types.items())
        },
    }


def merge_metrics(existing: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key in sorted(set(existing) | set(current)):
        old = existing.get(key) if isinstance(existing.get(key), dict) else {}
        new = current.get(key) if isinstance(current.get(key), dict) else {}
        old_total = int(old.get("total") or 0)
        new_total = int(new.get("total") or 0)
        total = old_total + new_total
        old_exact = int(old.get("exact_correct_rollout_count") or 0)
        new_exact = int(new.get("exact_correct_rollout_count") or 0)
        exact_rollouts = old_exact + new_exact
        old_rollouts = int(old.get("evaluated_rollout_count") or 0)
        new_rollouts = int(new.get("evaluated_rollout_count") or 0)
        evaluated_rollouts = old_rollouts + new_rollouts
        status_counts = {
            status: int(old.get(status) or 0) + int(new.get(status) or 0)
            for status in AUDIT_STATUSES
        }
        merged: dict[str, Any] = {
            "total": total,
            **status_counts,
            "status_counts": status_counts,
            "exact_correct_rollout_count": exact_rollouts,
            "evaluated_rollout_count": evaluated_rollouts,
            "weighted_rollout_exact_accuracy": (
                round(exact_rollouts / evaluated_rollouts, 6)
                if evaluated_rollouts
                else None
            ),
            "mean_permutation_soft_accuracy": (
                round(exact_rollouts / evaluated_rollouts, 6)
                if evaluated_rollouts
                else None
            ),
            "frontier_yield": (
                round(status_counts["robust_frontier"] / total, 6)
                if total
                else None
            ),
        }
        signature = new.get("signature") or old.get("signature")
        if isinstance(signature, dict):
            merged["signature"] = dict(signature)
        output[key] = merged
    return output


def reliable_yield_scores(metrics: dict[str, dict[str, Any]], keys: Iterable[str]) -> dict[str, float]:
    # Beta(1,1) smoothing preserves exploration for unseen or sparse cells.
    return {
        key: (int((metrics.get(key) or {}).get("robust_frontier") or 0) + 1.0)
        / (int((metrics.get(key) or {}).get("total") or 0) + 2.0)
        for key in keys
    }


def outcome_profile(summary: dict[str, Any]) -> dict[str, Any]:
    operations = summary.get("operations") if isinstance(summary.get("operations"), dict) else {}
    counts = {
        status: sum(int((operations.get(operation) or {}).get(status) or 0) for operation in OPERATIONS)
        for status in ("robust_frontier", "robust_easy5", "robust_hard0", "position_sensitive")
    }
    total = sum(int((operations.get(operation) or {}).get("total") or 0) for operation in OPERATIONS)
    return {
        "sample_count": total,
        "counts": counts,
        "rates": {
            status: round(count / total, 6) if total else 0.0
            for status, count in counts.items()
        },
    }


def difficulty_plan(
    aggregate: dict[str, Any],
    *,
    recent_batch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    aggregate_profile = outcome_profile(aggregate)
    recent_profile = outcome_profile(recent_batch or {})
    if recent_profile["sample_count"]:
        signal_scope = "recent_batch"
        signal = recent_profile
    else:
        signal_scope = "aggregate"
        signal = aggregate_profile
    rates = signal["rates"]
    if not signal["sample_count"]:
        target = "boundary"
        reason = "no_cyclic_k5_outcomes_keep_boundary"
    elif rates["robust_hard0"] >= HIGH_HARD0_THRESHOLD:
        target = "easy"
        reason = "high_robust_hard0_backoff"
    elif (
        rates["robust_easy5"] >= EASY5_MAJORITY_THRESHOLD
        and rates["robust_frontier"] < LOW_FRONTIER_YIELD_THRESHOLD
    ):
        target = "harder_boundary"
        reason = "mostly_robust_easy5_with_low_frontier_yield"
    else:
        target = "boundary"
        reason = "mixed_outcomes_keep_boundary"
    return {
        "item_difficulty_target": target,
        "reason": reason,
        "signal_scope": signal_scope,
        "signal": signal,
        "recent_batch": recent_profile,
        "aggregate": aggregate_profile,
        "thresholds": {
            "easy5_majority": EASY5_MAJORITY_THRESHOLD,
            "low_frontier_yield": LOW_FRONTIER_YIELD_THRESHOLD,
            "high_hard0": HIGH_HARD0_THRESHOLD,
        },
        "semantic_rejects_used_as_solver_signal": False,
    }


def build_plan(
    aggregate: dict[str, Any],
    *,
    minimum_operation_share: float,
    minimum_answer_type_share: float = 0.05,
    recent_batch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    operation_metrics = aggregate.get("operations") if isinstance(aggregate.get("operations"), dict) else {}
    if sum(int((operation_metrics.get(op) or {}).get("total") or 0) for op in OPERATIONS) == 0:
        operation_shares = normalized_shares(DEFAULT_OPERATION_WEIGHTS, OPERATIONS)
    else:
        operation_shares = shares_with_floor(
            reliable_yield_scores(operation_metrics, OPERATIONS),
            OPERATIONS,
            minimum_share=minimum_operation_share,
        )

    cells = aggregate.get("cells") if isinstance(aggregate.get("cells"), dict) else {}
    observed = aggregate.get("observed_answer_types")
    observed = observed if isinstance(observed, dict) else {}
    per_operation: dict[str, dict[str, float]] = {}
    for operation in OPERATIONS:
        required = REQUIRED_ANSWER_TYPES.get(operation)
        if required:
            per_operation[operation] = normalized_shares({value: 1.0 for value in required}, required)
            continue
        allowed_answer_types = set(ANSWER_TYPE_CANDIDATES[operation])
        observed_allowed = {
            str(value)
            for value in (observed.get(operation) or [])
            if str(value) in allowed_answer_types
        }
        answer_types = sorted(allowed_answer_types | observed_allowed)
        scores = reliable_yield_scores(cells, (f"{operation}|{value}" for value in answer_types))
        by_answer = {value: scores[f"{operation}|{value}"] for value in answer_types}
        floor = min(minimum_answer_type_share, (1.0 - 1e-6) / len(answer_types))
        per_operation[operation] = shares_with_floor(
            by_answer,
            answer_types,
            minimum_share=floor,
        )
    next_difficulty = difficulty_plan(aggregate, recent_batch=recent_batch)
    return {
        "schema_version": SCHEMA_VERSION,
        "target_metric": "robust_frontier_yield_from_cyclic_k5",
        "operation_shares": operation_shares,
        "answer_type_shares_by_operation": per_operation,
        "answer_type_candidate_source": "predefined_per_operation_with_observed_values_filtered",
        "item_difficulty_target": next_difficulty["item_difficulty_target"],
        "difficulty_control": next_difficulty,
        "minimum_operation_share": minimum_operation_share,
        "minimum_answer_type_share": minimum_answer_type_share,
        "semantic_rejects_used_as_solver_signal": False,
        "extra_solver_evaluation_run": False,
    }


def update_state(
    *,
    samples_path: Path,
    state_path: Path,
    accepted_solver_model: str,
    minimum_operation_share: float = 0.08,
) -> dict[str, Any]:
    samples = read_jsonl(samples_path)
    batch_bytes = samples_path.read_bytes()
    batch_id = hashlib.sha256(batch_bytes).hexdigest()
    batch_summary = summarize_batch(samples)
    existing: dict[str, Any] = {}
    duplicate_batch = False
    if state_path.exists():
        existing = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            raise ValueError(f"Expected controller state object: {state_path}")
        validate_state_protocol(existing)
        old_solver = str(existing.get("accepted_solver_model") or "")
        if old_solver and Path(old_solver).resolve() != Path(accepted_solver_model).resolve():
            raise RuntimeError(
                f"Stage3 operation-controller solver mismatch: {old_solver} != {accepted_solver_model}"
            )
        for batch in existing.get("batches") or []:
            if isinstance(batch, dict) and batch.get("batch_id") == batch_id:
                duplicate_batch = True
                break

    old_aggregate = existing.get("aggregate") if isinstance(existing.get("aggregate"), dict) else {}
    if duplicate_batch and old_aggregate:
        aggregate = {
            "sample_count": int(old_aggregate.get("sample_count") or 0),
            "evaluated_permutation_count": int(old_aggregate.get("evaluated_permutation_count") or 0),
            "operations": dict(old_aggregate.get("operations") or {}),
            "cells": dict(old_aggregate.get("cells") or {}),
            "reasoning_signatures": dict(
                old_aggregate.get("reasoning_signatures") or {}
            ),
        }
    else:
        aggregate = {
            "sample_count": int(old_aggregate.get("sample_count") or 0) + batch_summary["sample_count"],
            "evaluated_permutation_count": int(old_aggregate.get("evaluated_permutation_count") or 0)
            + batch_summary["evaluated_permutation_count"],
            "operations": merge_metrics(old_aggregate.get("operations") or {}, batch_summary["operations"]),
            "cells": merge_metrics(old_aggregate.get("cells") or {}, batch_summary["cells"]),
            "reasoning_signatures": merge_metrics(
                old_aggregate.get("reasoning_signatures") or {},
                batch_summary["reasoning_signatures"],
            ),
        }
    answer_types: dict[str, set[str]] = defaultdict(set)
    for source in (old_aggregate.get("observed_answer_types") or {}, batch_summary["observed_answer_types"]):
        if isinstance(source, dict):
            for operation, values in source.items():
                answer_types[str(operation)].update(str(value) for value in (values or []) if str(value))
    aggregate["observed_answer_types"] = {
        operation: sorted(values) for operation, values in sorted(answer_types.items())
    }
    plan = build_plan(
        aggregate,
        minimum_operation_share=minimum_operation_share,
        recent_batch=batch_summary,
    )
    batches = list(existing.get("batches") or [])
    if not duplicate_batch:
        batches.append(
            {
                "batch_id": batch_id,
                "samples_path": str(samples_path.resolve()),
                "sample_count": batch_summary["sample_count"],
                "evaluated_permutation_count": batch_summary["evaluated_permutation_count"],
                "generation_protocol_fingerprint": batch_summary[
                    "generation_protocol_fingerprint"
                ],
            }
        )
    state = {
        "schema_version": SCHEMA_VERSION,
        "generation_protocol": generation_protocol(),
        "accepted_solver_model": str(Path(accepted_solver_model).resolve()),
        "batches": batches,
        "aggregate": aggregate,
        "stage3_operation_plan": plan,
    }
    write_json(state_path, state)
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update Stage3 operation shares from existing cyclic K5 recalibration rows."
    )
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--state-path", type=Path, required=True)
    parser.add_argument("--accepted-solver-model", required=True)
    parser.add_argument("--minimum-operation-share", type=float, default=0.08)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state = update_state(
        samples_path=args.samples,
        state_path=args.state_path,
        accepted_solver_model=args.accepted_solver_model,
        minimum_operation_share=args.minimum_operation_share,
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
