#!/usr/bin/env python3
"""Prepare and validate QA-only Stage3 solver-feedback hardening work.

``export`` turns trusted cyclic K5 outcomes into a self-contained challenger
queue. ``generate`` may call the configured challenger for QA-only proposals
with resumable checkpoints. ``validate`` checks the frozen generation bundle
and existing Stage64 QA/semantic gates before publishing unevaluated samples
for the normal shuffle + cyclic K5 path. The script never loads a solver,
launches training, or permits graph/dialogue mutation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shlex
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.current_method import accumulate_stage64_reliable_frontier as accumulator  # noqa: E402
from scripts.current_method import native_api_graph_dialogue_expansion as native  # noqa: E402
from scripts.current_method import recalibrate_stage64_shuffled_frontier as recal  # noqa: E402
from scripts.current_method import stage3_operation_controller as stage3_operations  # noqa: E402
from scripts.current_method import stage3_multiqa_rows  # noqa: E402
from scripts.current_method import stage64_quality  # noqa: E402


SCHEMA_VERSION = "stage3_solver_feedback_mining_v3"
GRAPH_CONTRACT_LABEL = stage3_operations.REQUIRED_GRAPH_GATE_VERSION
AUDIT_STATUSES = (
    "robust_easy5",
    "robust_hard0",
    "position_sensitive",
    "robust_frontier",
)
REWRITE_ACTION_BY_STATUS = {
    "robust_easy5": "harden",
    "robust_hard0": "soften_or_diagnostic",
    "position_sensitive": "repair_position_sensitivity",
}
# Kept as a public compatibility name for callers that mean "send to the
# QA-only challenger", rather than every valid cyclic-audit status.
ELIGIBLE_STATUSES = tuple(REWRITE_ACTION_BY_STATUS)
TARGET_BAND = {
    "per_position_exact_count_min": 1,
    "per_position_exact_count_max": 4,
    "k_eval_per_position": 5,
    "aggregate_accuracy_min": 0.2,
    "aggregate_accuracy_max": 0.8,
    "preferred_accuracy_center": 0.5,
    "required_position_share": 1.0,
}
DUPLICATE_PENALTY_WEIGHT = 0.15
POSITION_PENALTY_WEIGHT = 0.25
RESPONSE_KEYS = {"queue_id", "frozen_bundle_sha256", "qa_proposal"}
FORBIDDEN_RESPONSE_KEYS = {
    "clean_dialogue",
    "current_alignment",
    "current_event_plan",
    "dialogue",
    "dialogue_update",
    "graph",
    "graph_update",
    "solver_eval",
    "stage64_permutation_eval",
}
STALE_SAMPLE_FIELDS = {
    "solver_eval",
    "solver_eval_k1",
    "frontier_label",
    "frontier_pool_priority",
    "frontier_type",
    "stage64_recalibration",
    "stage64_permutation_eval",
    "option_shuffle",
}
STALE_METADATA_FIELDS = {
    "option_shuffle_applied",
    "option_shuffle_seed",
    "option_shuffle_original_gold",
    "option_shuffle_target_gold",
    "option_shuffle_note",
}
STAGE64_REASONING_TARGET_FIELDS = tuple(
    getattr(
        native,
        "STAGE64_REASONING_TARGET_FIELDS",
        ("reasoning_subtype", "reasoning_template", "hop_count", "hard_negative_strategy"),
    )
)
STAGE64_TARGET_PROVENANCE_FIELDS = (
    "generation_slot_id",
    "target_index",
    "stage64_target_identity",
    "stage64_target_id",
    "stage64_target_plan_sha256",
)
STAGE64_REQUIRED_REASONING_PROVENANCE_FIELDS = (
    "target_index",
    "stage64_target_identity",
    "stage64_target_id",
    "stage64_target_plan_sha256",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_qa_responses(
    queue_dir: Path,
    config_path: Path,
    responses_path: Path,
    *,
    timeout: int = 180,
    max_retries: int = 1,
    max_tokens: int = 4096,
    temperature: float = 0.35,
    max_items: int | None = None,
) -> dict[str, Any]:
    manifest = read_json(queue_dir / "queue_ready_manifest.json")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("training_authorized") is not False:
        raise RuntimeError("Queue manifest is not a trusted QA-only manifest")
    queue = read_jsonl(queue_dir / "hardening_queue.jsonl")
    queue_by_id = {str(item.get("queue_id") or ""): item for item in queue}
    if not queue_by_id or len(queue_by_id) != len(queue):
        raise RuntimeError("Hardening queue IDs are empty or duplicated")

    config = read_json(config_path)
    api = config.get("api") if isinstance(config.get("api"), dict) else {}
    api_key = str(api.get("api_key") or "")
    base_url = str(api.get("base_url") or "")
    model = str(api.get("model") or "")
    if not api_key or not base_url or not model:
        raise RuntimeError("Challenger API config requires api_key, base_url, and model")
    runtime = SimpleNamespace(
        challenger_api_key=api_key,
        challenger_base_url=base_url,
        challenger_model=model,
        challenger_temperature=float(temperature),
        challenger_max_tokens=int(max_tokens),
        llm_timeout=int(timeout),
        max_llm_retries=int(max_retries),
        api_yield_normalizers=False,
    )

    existing = read_jsonl(responses_path) if responses_path.exists() else []
    response_by_id: dict[str, dict[str, Any]] = {}
    for response in existing:
        queue_id = str(response.get("queue_id") or "")
        if queue_id not in queue_by_id or queue_id in response_by_id:
            raise RuntimeError(f"Invalid existing challenger response queue_id: {queue_id!r}")
        response_by_id[queue_id] = response
    audit_path = responses_path.with_name("challenger_qa_response_audit.jsonl")
    audits = read_jsonl(audit_path) if audit_path.exists() else []
    failures = 0
    attempted = 0
    pending = [queue_by_id[key] for key in sorted(queue_by_id) if key not in response_by_id]
    if max_items is not None:
        pending = pending[: max(0, int(max_items))]
    for item in pending:
        queue_id = str(item["queue_id"])
        candidate, meta = native.call_raw_json_api(
            runtime,
            item["challenger_request"],
            system_prompt=(
                "You are a strict JSON-only QA hardening challenger. Return exactly queue_id, "
                "frozen_bundle_sha256, and qa_proposal. Never emit or modify graph/dialogue."
            ),
            required_keys=("queue_id", "frozen_bundle_sha256", "qa_proposal"),
            call_kind="api_qa_hardening",
            max_tokens_override=int(max_tokens),
        )
        attempted += 1
        audits.append(
            {
                "queue_id": queue_id,
                "success": isinstance(candidate, dict),
                "meta": meta,
                "response": copy.deepcopy(candidate),
            }
        )
        if isinstance(candidate, dict):
            response_by_id[queue_id] = candidate
            write_jsonl(responses_path, [response_by_id[key] for key in sorted(response_by_id)])
        else:
            failures += 1
        write_jsonl(audit_path, audits)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "mode": "qa_only_challenger_response_generation",
        "queue_count": len(queue_by_id),
        "preexisting_response_count": len(existing),
        "attempted_count": attempted,
        "successful_response_count": len(response_by_id),
        "failed_attempt_count": failures,
        "remaining_count": len(queue_by_id) - len(response_by_id),
        "responses": str(responses_path.resolve()),
        "audit": str(audit_path.resolve()),
        "graph_dialogue_mutation_allowed": False,
        "automatic_solver_calls": False,
        "automatic_training": False,
        "training_authorized": False,
    }
    write_json(queue_dir / "qa_response_generation_summary.json", summary)
    return summary


def normalized_text(value: Any) -> str:
    return stage64_quality.normalized_option_text(value)


def stage64_target_freeze_payload(sample: dict[str, Any]) -> dict[str, Any] | None:
    """Freeze a formal Stage64 target and its identity without breaking legacy rows."""

    raw_target = sample.get("stage64_target")
    if raw_target is None:
        return None
    if not isinstance(raw_target, dict):
        raise RuntimeError(
            f"Sample {sample.get('sample_id')!r} has a non-object stage64_target"
        )
    target = copy.deepcopy(raw_target)
    present_reasoning_fields = [
        field for field in STAGE64_REASONING_TARGET_FIELDS if field in target
    ]
    reasoning_contract: dict[str, Any] | None = None
    if present_reasoning_fields:
        missing_reasoning_fields = [
            field for field in STAGE64_REASONING_TARGET_FIELDS if field not in target
        ]
        if missing_reasoning_fields:
            raise RuntimeError(
                f"Sample {sample.get('sample_id')!r} has a partial Stage64 reasoning "
                f"target: missing={missing_reasoning_fields}"
            )
        qa = sample.get("qa") if isinstance(sample.get("qa"), dict) else {}
        local_truth_sibling = (
            target.get("target_basis") == "truth_first_local_sibling_qa"
            and sample.get("qa_row_schema_version")
            == stage3_multiqa_rows.SCHEMA_VERSION
            and qa.get("qa_role") == "sibling"
            and qa.get("truth_first_generate_by_construction") is True
            and qa.get("semantic_contract_frozen") is True
        )
        if local_truth_sibling:
            semantic_issues = stage64_quality.semantic_quality_issues(sample)
            if semantic_issues:
                raise RuntimeError(
                    f"Sample {sample.get('sample_id')!r} has an invalid local "
                    f"Truth-first sibling target: {semantic_issues[:4]}"
                )
        else:
            try:
                canonical_reasoning = native._stage64_reasoning_contract_from_target(  # noqa: SLF001
                    target
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Sample {sample.get('sample_id')!r} has an invalid Stage64 reasoning target"
                ) from exc
            if not isinstance(canonical_reasoning, dict):
                raise RuntimeError(
                    f"Sample {sample.get('sample_id')!r} lacks a canonical Stage64 reasoning target"
                )
        reasoning_contract = {
            field: copy.deepcopy(target[field]) for field in STAGE64_REASONING_TARGET_FIELDS
        }
        if "forbid_max_time_shortcut" in target:
            reasoning_contract["forbid_max_time_shortcut"] = copy.deepcopy(
                target["forbid_max_time_shortcut"]
            )

    provenance = {
        field: copy.deepcopy(sample[field])
        for field in STAGE64_TARGET_PROVENANCE_FIELDS
        if field in sample
    }
    if reasoning_contract is not None:
        missing_provenance = [
            field
            for field in STAGE64_REQUIRED_REASONING_PROVENANCE_FIELDS
            if field not in provenance
            or provenance[field] is None
            or (isinstance(provenance[field], str) and not provenance[field].strip())
        ]
        if missing_provenance:
            raise RuntimeError(
                f"Sample {sample.get('sample_id')!r} has a reasoning target without "
                f"complete immutable provenance: missing={missing_provenance}"
            )
        target_index = provenance["target_index"]
        if isinstance(target_index, bool) or not isinstance(target_index, int) or target_index < 0:
            raise RuntimeError(
                f"Sample {sample.get('sample_id')!r} has invalid Stage64 target_index"
            )
        target_identity = str(provenance["stage64_target_identity"]).strip()
        target_id = str(provenance["stage64_target_id"]).strip()
        target_plan_sha256 = str(provenance["stage64_target_plan_sha256"]).strip()
        if target_identity != target_id:
            raise RuntimeError(
                f"Sample {sample.get('sample_id')!r} has mismatched Stage64 target identity/id"
            )
        expected_identity = native._stage64_target_identity(  # noqa: SLF001
            target_index,
            target,
            target_plan_sha256,
        )
        if target_identity != expected_identity:
            raise RuntimeError(
                f"Sample {sample.get('sample_id')!r} has invalid Stage64 target identity provenance"
            )

    payload = {
        "stage64_target": target,
        "stage64_target_sha256": sha256_json(target),
        "provenance": provenance,
        "reasoning_contract": reasoning_contract,
    }
    payload["sha256"] = sha256_json(payload)
    return payload


def validate_qa_reasoning_contract(
    qa: dict[str, Any],
    target_contract: dict[str, Any] | None,
    *,
    owner: str,
) -> None:
    """Require both QA and query_plan to echo the frozen reasoning identity."""

    if not isinstance(target_contract, dict):
        return
    reasoning_contract = target_contract.get("reasoning_contract")
    if not isinstance(reasoning_contract, dict):
        return
    query_plan = qa.get("query_plan")
    if not isinstance(query_plan, dict):
        raise RuntimeError(f"{owner} lacks query_plan for frozen Stage64 reasoning target")
    fields = list(STAGE64_REASONING_TARGET_FIELDS)
    if "forbid_max_time_shortcut" in reasoning_contract:
        fields.append("forbid_max_time_shortcut")
    for field in fields:
        expected = reasoning_contract[field]
        if qa.get(field) != expected:
            raise RuntimeError(
                f"{owner} changed frozen Stage64 reasoning field {field}: "
                f"actual={qa.get(field)!r}, expected={expected!r}"
            )
        if query_plan.get(field) != expected:
            raise RuntimeError(
                f"{owner}.query_plan changed frozen Stage64 reasoning field {field}: "
                f"actual={query_plan.get(field)!r}, expected={expected!r}"
            )


def qa_gold_contract(sample: dict[str, Any]) -> dict[str, Any]:
    qa = sample.get("qa") if isinstance(sample.get("qa"), dict) else {}
    options = qa.get("options") if isinstance(qa.get("options"), list) else []
    gold = qa.get("gold") if isinstance(qa.get("gold"), list) else []
    if len(gold) != 1:
        raise RuntimeError(f"Sample {sample.get('sample_id')!r} must have one gold label")
    gold_label = str(gold[0])
    matches = [item for item in options if isinstance(item, dict) and str(item.get("id")) == gold_label]
    if len(matches) != 1:
        raise RuntimeError(f"Sample {sample.get('sample_id')!r} gold label does not resolve uniquely")
    option = matches[0]
    source_event_id = str(option.get("source_event_instance_id") or option.get("event_instance_id") or "")
    gold_text = str(option.get("text") or qa.get("gold_answer_text") or "").strip()
    if not source_event_id or not gold_text:
        raise RuntimeError(f"Sample {sample.get('sample_id')!r} lacks frozen gold evidence")
    query_plan = qa.get("query_plan") if isinstance(qa.get("query_plan"), dict) else {}
    resolved_gold_ids = [
        str(item)
        for item in (query_plan.get("resolved_gold_event_instance_ids") or [])
        if str(item)
    ]
    if source_event_id not in resolved_gold_ids:
        resolved_gold_ids.append(source_event_id)
    contract = {
        "original_label": gold_label,
        "answer_text": gold_text,
        "normalized_answer_text": normalized_text(gold_text),
        "source_event_instance_id": source_event_id,
        "resolved_gold_event_instance_ids": sorted(set(resolved_gold_ids)),
        "qa_operation": str(qa.get("qa_operation") or sample.get("qa_operation") or ""),
        "answer_type": str(qa.get("answer_type") or sample.get("answer_type") or ""),
    }
    if not contract["qa_operation"] or not contract["answer_type"]:
        raise RuntimeError(f"Sample {sample.get('sample_id')!r} lacks operation/answer type")
    contract["sha256"] = sha256_json(contract)
    return contract


def marked_clean_sample(bundle: dict[str, Any]) -> dict[str, Any]:
    sample = bundle.get("clean_sample")
    if not isinstance(sample, dict):
        raise RuntimeError("Marked bundle lacks clean_sample")
    return sample


def marked_bundle_id(bundle: dict[str, Any]) -> str:
    return str(marked_clean_sample(bundle).get("sample_id") or "")


def locate_marked_samples(recalibration_dir: Path) -> Path:
    candidates = [
        recalibration_dir.parent / "claim_first_marked_qa_rows.jsonl",
        recalibration_dir.parent.parent
        / "generation"
        / "claim_first_marked_qa_rows.jsonl",
        recalibration_dir.parent
        / "generation"
        / "claim_first_marked_qa_rows.jsonl",
        recalibration_dir.parent.parent / "generation" / "marked_samples.jsonl",
        recalibration_dir.parent / "generation" / "marked_samples.jsonl",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        "Cannot safely export QA-only work without generation/"
        "claim_first_marked_qa_rows.jsonl or marked_samples.jsonl: "
        f"{recalibration_dir}"
    )


def validate_marked_bundle(sample: dict[str, Any], bundle: dict[str, Any]) -> None:
    sample_id = str(sample.get("sample_id") or "")
    if marked_bundle_id(bundle) != sample_id:
        raise RuntimeError(f"Marked bundle sample_id mismatch for {sample_id!r}")
    required_dicts = ("current_event_plan", "current_alignment", "clean_dialogue", "hashes")
    missing = [key for key in required_dicts if not isinstance(bundle.get(key), dict)]
    if missing:
        raise RuntimeError(f"Marked bundle {sample_id!r} lacks required objects: {missing}")
    marked_sample = marked_clean_sample(bundle)
    comparisons = {
        "graph_update": marked_sample.get("graph_update") == sample.get("graph_update"),
        "dialogue": marked_sample.get("dialogue") == sample.get("dialogue"),
        "graph_snapshot_hash": (
            bundle["hashes"].get("graph_snapshot_hash")
            == (sample.get("graph_update") or {}).get("graph_snapshot_hash")
        ),
        "dialogue_hash": bundle["hashes"].get("dialogue_hash") == (sample.get("dialogue") or {}).get("dialogue_hash"),
    }
    if not all(comparisons.values()):
        raise RuntimeError(f"Frozen marked bundle mismatch for {sample_id!r}: {comparisons}")
    marked_gold = qa_gold_contract(marked_sample)
    recalibrated_gold = qa_gold_contract(sample)
    semantic_gold_equal = all(
        marked_gold[key] == recalibrated_gold[key]
        for key in ("normalized_answer_text", "source_event_instance_id", "qa_operation", "answer_type")
    )
    if not semantic_gold_equal:
        raise RuntimeError(f"Option shuffle changed gold semantics for {sample_id!r}")
    marked_target = stage64_target_freeze_payload(marked_sample)
    recalibrated_target = stage64_target_freeze_payload(sample)
    if marked_target != recalibrated_target:
        raise RuntimeError(f"Option shuffle changed frozen Stage64 target for {sample_id!r}")


def frozen_bundle_payload(sample: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": str(sample.get("sample_id") or ""),
        "stage64_target_contract": stage64_target_freeze_payload(sample),
        "graph_update": copy.deepcopy(sample.get("graph_update")),
        "dialogue": copy.deepcopy(sample.get("dialogue")),
        "expanded_dialogue_text": sample.get("expanded_dialogue_text"),
        "current_event_plan": copy.deepcopy(bundle.get("current_event_plan")),
        "current_alignment": copy.deepcopy(bundle.get("current_alignment")),
        "clean_dialogue": copy.deepcopy(bundle.get("clean_dialogue")),
        "hashes": copy.deepcopy(bundle.get("hashes")),
        "current_anchor_to_event_instance_id": copy.deepcopy(
            bundle.get("current_anchor_to_event_instance_id") or {}
        ),
        "current_event_instance_to_anchor": copy.deepcopy(
            bundle.get("current_event_instance_to_anchor") or {}
        ),
    }


def frozen_contract(sample: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    payload = frozen_bundle_payload(sample, bundle)
    gold = qa_gold_contract(sample)
    contract = {
        "graph_sha256": sha256_json(payload["graph_update"]),
        "dialogue_sha256": sha256_json(payload["dialogue"]),
        "generation_bundle_sha256": sha256_json(
            {
                "current_event_plan": payload["current_event_plan"],
                "current_alignment": payload["current_alignment"],
                "clean_dialogue": payload["clean_dialogue"],
                "hashes": payload["hashes"],
            }
        ),
        "frozen_bundle_sha256": sha256_json(payload),
        "gold": gold,
        "stage64_target_contract": copy.deepcopy(payload["stage64_target_contract"]),
        "immutable_fields": [
            "graph_update",
            "dialogue",
            "expanded_dialogue_text",
            "current_event_plan",
            "current_alignment",
            "clean_dialogue",
            "hashes",
            "stage64_target",
            "generation_slot_id",
            "target_index",
            "stage64_target_identity",
            "stage64_target_id",
            "stage64_target_plan_sha256",
            "gold.answer_text",
            "gold.source_event_instance_id",
        ],
    }
    contract["sha256"] = sha256_json(contract)
    return contract


def qa_duplicate_fingerprint(sample: dict[str, Any]) -> str:
    """Fingerprint user-visible QA semantics independently of option order."""
    qa = sample.get("qa") if isinstance(sample.get("qa"), dict) else {}
    options = qa.get("options") if isinstance(qa.get("options"), list) else []
    visible = {
        "question": normalized_text(qa.get("question") or qa.get("question_text")),
        "qa_operation": str(qa.get("qa_operation") or sample.get("qa_operation") or ""),
        "qa_subtype": str(qa.get("qa_subtype") or sample.get("qa_subtype") or ""),
        "answer_type": str(qa.get("answer_type") or sample.get("answer_type") or ""),
        "options": sorted(
            normalized_text(option.get("text"))
            for option in options
            if isinstance(option, dict)
        ),
    }
    return sha256_json(visible)


def cyclic_status_from_counts(counts: list[int], *, k_eval: int) -> str:
    if counts and all(count == 0 for count in counts):
        return "robust_hard0"
    if counts and all(count == k_eval for count in counts):
        return "robust_easy5"
    if counts and all(0 < count < k_eval for count in counts):
        return "robust_frontier"
    return "position_sensitive"


def feedback_decision(status: str) -> dict[str, Any]:
    if status == "robust_frontier":
        return {
            "action": "retain_frontier",
            "disposition": "retain_without_rewrite",
            "qa_rewrite_requested": False,
            "diagnostic_only_allowed": False,
            "reason": "Every cyclic position is inside the reliable 1/5-4/5 frontier band.",
        }
    action = REWRITE_ACTION_BY_STATUS[status]
    return {
        "action": action,
        "disposition": "qa_only_rewrite_queue",
        "qa_rewrite_requested": True,
        "diagnostic_only_allowed": status == "robust_hard0",
        "reason": {
            "robust_easy5": "The fixed Stage2 solver mastered every cyclic position.",
            "robust_hard0": (
                "The fixed Stage2 solver failed every cyclic position; retain a diagnostic copy "
                "and request a controlled softening toward the boundary."
            ),
            "position_sensitive": (
                "Success changes with answer position, so surface/position cues must be repaired "
                "before the item can be considered reliable."
            ),
        }[status],
    }


def _semantic_option_for_prediction(
    options: list[dict[str, Any]],
    *,
    shift: int,
    parsed_label: str,
) -> dict[str, Any] | None:
    labels = recal.LETTERS[: len(options)]
    if parsed_label not in labels:
        return None
    displayed_index = labels.index(parsed_label)
    return options[(int(shift) + displayed_index) % len(options)]


def _score_feedback(feedback: dict[str, Any], *, duplicate_count: int) -> None:
    duplicate_count = max(1, int(duplicate_count))
    p_exact = float(feedback["aggregate_exact_accuracy"])
    position_accuracies = [float(item["exact_accuracy"]) for item in feedback["position_results"]]
    position_range = max(position_accuracies) - min(position_accuracies)
    boundary_reward = max(0.0, 1.0 - 2.0 * abs(p_exact - TARGET_BAND["preferred_accuracy_center"]))
    duplicate_penalty = DUPLICATE_PENALTY_WEIGHT * ((duplicate_count - 1) / duplicate_count)
    position_penalty = POSITION_PENALTY_WEIGHT * position_range
    feedback["duplicate_audit"] = {
        "visible_qa_fingerprint": feedback["visible_qa_fingerprint"],
        "audit_pool_occurrence_count": duplicate_count,
        "is_duplicate": duplicate_count > 1,
        "penalty": round(duplicate_penalty, 6),
    }
    feedback["position_audit"] = {
        "exact_accuracy_min": round(min(position_accuracies), 6),
        "exact_accuracy_max": round(max(position_accuracies), 6),
        "exact_accuracy_range": round(position_range, 6),
        "penalty": round(position_penalty, 6),
    }
    feedback["difficulty_score"] = {
        "formula": "max(0, boundary_reward - duplicate_penalty - position_penalty)",
        "boundary_reward": round(boundary_reward, 6),
        "duplicate_penalty": round(duplicate_penalty, 6),
        "position_penalty": round(position_penalty, 6),
        "score": round(max(0.0, boundary_reward - duplicate_penalty - position_penalty), 6),
    }


def validate_cyclic_audit(
    sample: dict[str, Any],
    *,
    k_eval: int = 5,
    duplicate_count: int = 1,
) -> dict[str, Any]:
    sample_id = str(sample.get("sample_id") or "")
    qa = sample.get("qa") if isinstance(sample.get("qa"), dict) else {}
    arity = len(qa.get("options") or []) if isinstance(qa.get("options"), list) else 0
    audit = sample.get("stage64_permutation_eval")
    if not isinstance(audit, dict):
        raise RuntimeError(f"Sample {sample_id!r} lacks cyclic permutation audit")
    status = str(audit.get("status") or "")
    if status not in AUDIT_STATUSES:
        raise RuntimeError(f"Sample {sample_id!r} has unknown cyclic status {status!r}")
    if int(audit.get("k_eval_per_permutation") or 0) != k_eval:
        raise RuntimeError(f"Sample {sample_id!r} is not K={k_eval}")
    permutations = audit.get("permutations") if isinstance(audit.get("permutations"), list) else []
    if arity not in (2, 4) or len(permutations) != arity or int(audit.get("permutation_count") or 0) != arity:
        raise RuntimeError(f"Sample {sample_id!r} has incomplete cyclic positions")
    expected_labels = set(recal.LETTERS[:arity])
    seen_labels: set[str] = set()
    counts: list[int] = []
    feedback_permutations: list[dict[str, Any]] = []
    wrong_predictions: list[dict[str, Any]] = []
    predicted_labels: Counter[str] = Counter()
    semantic_option_selections: Counter[str] = Counter()
    invalid_prediction_count = 0
    option_by_key: dict[str, dict[str, Any]] = {}
    source_options = [copy.deepcopy(item) for item in qa.get("options") or [] if isinstance(item, dict)]
    for index, option in enumerate(source_options):
        option_key = str(
            option.get("source_event_instance_id")
            or option.get("event_instance_id")
            or option.get("anchor_ref")
            or f"option_index:{index}"
        )
        option_by_key[option_key] = {
            "option_key": option_key,
            "text": str(option.get("text") or ""),
            "normalized_text": normalized_text(option.get("text")),
            "source_event_instance_id": str(option.get("source_event_instance_id") or ""),
            "is_gold": option.get("truth_value") is True or option.get("is_gold") is True,
        }
    for permutation in permutations:
        if not isinstance(permutation, dict):
            raise RuntimeError(f"Sample {sample_id!r} has invalid permutation row")
        gold = permutation.get("gold")
        if isinstance(gold, list):
            gold = gold[0] if len(gold) == 1 else ""
        gold = str(gold or "")
        seen_labels.add(gold)
        count = permutation.get("exact_correct_count")
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= k_eval:
            raise RuntimeError(f"Sample {sample_id!r} has invalid cyclic exact count")
        predictions = permutation.get("predictions") if isinstance(permutation.get("predictions"), list) else []
        if len(predictions) != k_eval or not all(isinstance(item, dict) for item in predictions):
            raise RuntimeError(f"Sample {sample_id!r} lacks {k_eval} predictions for a cyclic position")
        exact_flags = [item.get("exact_correct") is True for item in predictions]
        if sum(exact_flags) != count:
            raise RuntimeError(f"Sample {sample_id!r} cyclic exact flags disagree with count")
        counts.append(count)
        copied_predictions: list[dict[str, Any]] = []
        for prediction in predictions:
            copied = copy.deepcopy(prediction)
            copied_predictions.append(copied)
            parsed = str(prediction.get("parsed") or prediction.get("parsed_prediction") or "")
            if parsed:
                predicted_labels[parsed] += 1
            selected_option = _semantic_option_for_prediction(
                source_options,
                shift=int(permutation.get("shift") or 0),
                parsed_label=parsed,
            )
            selected_option_key: str | None = None
            if selected_option is None:
                invalid_prediction_count += 1
            else:
                selected_option_key = str(
                    selected_option.get("source_event_instance_id")
                    or selected_option.get("event_instance_id")
                    or selected_option.get("anchor_ref")
                    or ""
                )
                if selected_option_key:
                    semantic_option_selections[selected_option_key] += 1
            if prediction.get("exact_correct") is not True:
                wrong_predictions.append(
                    {
                        "shift": permutation.get("shift"),
                        "gold": gold,
                        "prediction": prediction.get("prediction"),
                        "parsed": parsed or None,
                        "selected_option_key": selected_option_key,
                    }
                )
        feedback_permutations.append(
            {
                "shift": permutation.get("shift"),
                "gold": gold,
                "exact_correct_count": count,
                "bucket": permutation.get("bucket"),
                "soft_accuracy": permutation.get("soft_accuracy"),
                "predictions": copied_predictions,
            }
        )
    if seen_labels != expected_labels:
        raise RuntimeError(f"Sample {sample_id!r} cyclic audit does not cover every answer label")
    derived_status = cyclic_status_from_counts(counts, k_eval=k_eval)
    if status != derived_status:
        raise RuntimeError(
            f"Sample {sample_id!r} cyclic status {status!r} disagrees with counts; expected {derived_status!r}"
        )
    position_results = [
        {
            "shift": row["shift"],
            "gold": row["gold"],
            "exact_correct_count": count,
            "exact_accuracy": round(count / k_eval, 6),
            "inside_target_band": TARGET_BAND["per_position_exact_count_min"]
            <= count
            <= TARGET_BAND["per_position_exact_count_max"],
        }
        for row, count in zip(feedback_permutations, counts)
    ]
    aggregate_exact_count = sum(counts)
    aggregate_total = arity * k_eval
    aggregate_accuracy = aggregate_exact_count / aggregate_total
    option_confusion = []
    wrong_confusion: Counter[str] = Counter(
        str(row["selected_option_key"])
        for row in wrong_predictions
        if row.get("selected_option_key")
    )
    for option_key, option in sorted(option_by_key.items()):
        selected_count = semantic_option_selections[option_key]
        option_confusion.append(
            {
                **option,
                "selected_count": selected_count,
                "selection_share": round(selected_count / aggregate_total, 6),
                "wrong_selection_count": wrong_confusion[option_key],
            }
        )
    error_signal_by_status = {
        "robust_easy5": [
            {
                "code": "too_easy_all_cyclic_positions",
                "detail": "Fixed Stage2 solved every sample at every answer position; harden reasoning and distractors.",
            }
        ],
        "robust_hard0": [
            {
                "code": "too_hard_all_cyclic_positions",
                "detail": "Fixed Stage2 failed every sample at every answer position; soften toward the target band or keep diagnostic-only.",
            }
        ],
        "position_sensitive": [
            {
                "code": "position_sensitive_not_reliable_frontier",
                "detail": "Difficulty changed with answer position; remove positional/surface cues while preserving reasoning.",
            },
            {
                "code": "cyclic_exact_count_variation",
                "counts": counts,
                "range": max(counts) - min(counts),
            },
        ],
        "robust_frontier": [
            {
                "code": "reliable_frontier_all_cyclic_positions",
                "detail": "Every answer position is inside the 1/5-4/5 frontier band; retain without QA rewrite.",
            }
        ],
    }
    feedback = {
        "status": status,
        "decision": feedback_decision(status),
        "k_eval_per_permutation": k_eval,
        "permutation_count": arity,
        "frontier_permutation_share": audit.get("frontier_permutation_share"),
        "target_band": copy.deepcopy(TARGET_BAND),
        "aggregate_exact_correct_count": aggregate_exact_count,
        "aggregate_rollout_count": aggregate_total,
        "aggregate_exact_accuracy": round(aggregate_accuracy, 6),
        "all_positions_inside_target_band": all(row["inside_target_band"] for row in position_results),
        "position_results": position_results,
        "permutations": feedback_permutations,
        "wrong_predictions": wrong_predictions,
        "predicted_label_histogram": dict(sorted(predicted_labels.items())),
        "semantic_option_confusion": option_confusion,
        "invalid_prediction_count": invalid_prediction_count,
        "visible_qa_fingerprint": qa_duplicate_fingerprint(sample),
        "error_signals": error_signal_by_status[status],
    }
    _score_feedback(feedback, duplicate_count=duplicate_count)
    return feedback


def validate_batch_summary(
    directory: Path,
    summary: dict[str, Any],
    accepted_stage2_solver_model: Path,
) -> dict[str, Any]:
    config, config_sources = accumulator.fixed_solver_eval_config(summary)
    accepted_resolved = str(accepted_stage2_solver_model.resolve())
    gates = {
        "accepted_solver_path_exists": accepted_stage2_solver_model.exists(),
        "k_eval_is_5": summary.get("k_eval") == 5,
        "cyclic_permutations": summary.get("permutation_mode") == "cyclic",
        "all_positions_required": float(summary.get("min_frontier_permutation_share") or 0.0) == 1.0,
        "semantic_gate_enabled": (summary.get("quality") or {}).get("semantic_quality_gate_enabled") is True,
        "decoding_healthy": (summary.get("decoding_health") or {}).get("healthy") is True,
        "accepted_solver_matches": str(summary.get("accepted_stage2_solver_resolved") or "") == accepted_resolved,
        "loaded_solver_matches": str(summary.get("loaded_solver_model_resolved") or "") == accepted_resolved,
        "adapter_absent": not summary.get("loaded_solver_adapter"),
        "base_model_recorded": bool(str(summary.get("base_model") or "")),
        "solver_eval_config_complete": all(config.get(key) is not None for key in accumulator.SOLVER_EVAL_CONFIG_FIELDS),
    }
    if not all(gates.values()):
        raise RuntimeError(f"Untrusted Stage3 recalibration batch {directory}: {gates}")
    return {"gates": gates, "solver_eval_config": config, "config_sources": config_sources}


def challenger_feedback_payload(
    solver_identity: dict[str, Any],
    solver_feedback: dict[str, Any],
) -> dict[str, Any]:
    """Expose semantic aggregate feedback without shifted labels/predictions."""

    position_counts = [
        int(item.get("exact_correct_count") or 0)
        for item in (solver_feedback.get("position_results") or [])
        if isinstance(item, dict)
    ]
    semantic_confusion = []
    for item in solver_feedback.get("semantic_option_confusion") or []:
        if not isinstance(item, dict):
            continue
        semantic_confusion.append(
            {
                key: copy.deepcopy(item.get(key))
                for key in (
                    "option_key",
                    "text",
                    "normalized_text",
                    "source_event_instance_id",
                    "selected_count",
                    "selection_share",
                    "wrong_selection_count",
                )
            }
        )
    return {
        "solver": copy.deepcopy(solver_identity),
        "status": solver_feedback.get("status"),
        "decision": copy.deepcopy(solver_feedback.get("decision")),
        "k_eval_per_permutation": solver_feedback.get("k_eval_per_permutation"),
        "permutation_count": solver_feedback.get("permutation_count"),
        "target_band": copy.deepcopy(solver_feedback.get("target_band")),
        "aggregate_exact_correct_count": solver_feedback.get(
            "aggregate_exact_correct_count"
        ),
        "aggregate_rollout_count": solver_feedback.get("aggregate_rollout_count"),
        "aggregate_exact_accuracy": solver_feedback.get("aggregate_exact_accuracy"),
        "cyclic_exact_count_summary": {
            "counts_sorted": sorted(position_counts),
            "minimum": min(position_counts) if position_counts else None,
            "maximum": max(position_counts) if position_counts else None,
        },
        "semantic_option_confusion": semantic_confusion,
        "invalid_prediction_count": solver_feedback.get("invalid_prediction_count"),
        "error_signals": copy.deepcopy(solver_feedback.get("error_signals")),
        "duplicate_audit": copy.deepcopy(solver_feedback.get("duplicate_audit")),
        "position_audit": copy.deepcopy(solver_feedback.get("position_audit")),
        "difficulty_score": copy.deepcopy(solver_feedback.get("difficulty_score")),
        "raw_shifted_labels_omitted": True,
        "raw_predictions_omitted": True,
    }


def compact_evidence_dialogue(
    dialogue: dict[str, Any],
    event_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    evidence_turn_ids = {
        str(turn_id)
        for candidate in event_candidates
        if isinstance(candidate, dict)
        for field in ("turn_ids", "evidence_turn_ids")
        for turn_id in (
            candidate.get(field)
            if isinstance(candidate.get(field), list)
            else []
        )
        if str(turn_id)
    }
    all_turns = [
        turn
        for turn in (dialogue.get("turns") or [])
        if isinstance(turn, dict)
    ]
    evidence_turns = [
        copy.deepcopy(turn)
        for turn in all_turns
        if str(turn.get("turn_id") or "") in evidence_turn_ids
    ]
    return {
        "dialogue_hash": dialogue.get("dialogue_hash"),
        "turns": evidence_turns,
        "evidence_turn_ids": sorted(evidence_turn_ids),
        "full_turn_count": len(all_turns),
        "omitted_non_evidence_turn_count": len(all_turns) - len(evidence_turns),
        "evidence_only": True,
    }


def queue_item(
    sample: dict[str, Any],
    bundle: dict[str, Any],
    *,
    batch_dir: Path,
    generation_seed: int,
    solver_identity: dict[str, Any],
    solver_feedback: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = frozen_contract(sample, bundle)
    qa = sample.get("qa") if isinstance(sample.get("qa"), dict) else {}
    operation = str(qa.get("qa_operation") or sample.get("qa_operation") or "")
    answer_type = str(qa.get("answer_type") or sample.get("answer_type") or "")
    target_contract = contract.get("stage64_target_contract")
    reasoning_contract = (
        target_contract.get("reasoning_contract")
        if isinstance(target_contract, dict)
        and isinstance(target_contract.get("reasoning_contract"), dict)
        else None
    )
    queue_id = f"stage3_qah_{sha256_json([sample.get('sample_id'), contract['frozen_bundle_sha256'], qa])[:20]}"
    context = frozen_bundle_payload(sample, bundle)
    event_candidates = native._stage64_event_candidates_for_prompt(  # noqa: SLF001
        context["current_event_plan"],
        context["current_alignment"],
    )
    action = str((solver_feedback.get("decision") or {}).get("action") or "")
    action_target = {
        "harden": {
            "mode": "qa_only_solver_feedback_hardening",
            "difficulty_target": "harder_boundary",
            "goal": "Move a mastered item into the robust 1/5-4/5 band at every cyclic position.",
            "instruction": (
                "Increase necessary reasoning depth and replace weak distractors with graph-supported "
                "near misses while preserving the frozen gold."
            ),
        },
        "soften_or_diagnostic": {
            "mode": "qa_only_solver_feedback_softening",
            "difficulty_target": "softer_boundary",
            "goal": "Move a robust hard0 item into the robust 1/5-4/5 band, or retain it diagnostic-only.",
            "instruction": (
                "Reduce unnecessary ambiguity or reasoning depth without adding lexical answer cues; "
                "keep the original diagnostic row separate."
            ),
        },
        "repair_position_sensitivity": {
            "mode": "qa_only_position_sensitivity_repair",
            "difficulty_target": "position_invariant_boundary",
            "goal": "Remove answer-position/surface effects and reach 1/5-4/5 at every cyclic position.",
            "instruction": (
                "Balance option wording, length, specificity, and answer type so the semantic reasoning, "
                "not the displayed label or surface form, determines the result."
            ),
        },
    }.get(action)
    if action_target is None:
        raise RuntimeError(f"Unsupported QA rewrite action {action!r}")
    rewrite_instruction = (
        "Keep all four same-slot chain option values and source_event_instance_id references fixed. "
        "Harden by substantively rewriting the question so validity/status reasoning is necessary and direct answer hints are reduced."
        if action == "harden" and operation == "temporal_state_update"
        else action_target["instruction"]
    )
    reasoning_schema = (
        {
            field: copy.deepcopy(reasoning_contract[field])
            for field in STAGE64_REASONING_TARGET_FIELDS
        }
        if reasoning_contract is not None
        else {}
    )
    if reasoning_contract is not None and "forbid_max_time_shortcut" in reasoning_contract:
        reasoning_schema["forbid_max_time_shortcut"] = copy.deepcopy(
            reasoning_contract["forbid_max_time_shortcut"]
        )
    qa_output_schema = {
        "question_type": "single_choice",
        "question": "substantively rewritten solver-facing question ending in ?",
        "question_text": "exactly the same string as question",
        "qa_operation": operation,
        "qa_subtype": qa.get("qa_subtype"),
        "answer_type": answer_type,
        "answer_format": "single_choice_mcq",
        **copy.deepcopy(reasoning_schema),
        "options": [
            {
                "id": "A-D in display order",
                "text": "answer value",
                "answer_type": answer_type,
                "source_event_instance_id": "durable event id from event_candidates",
                "anchor_ref": "same durable event id",
                "anchor_kind": "event",
                "truth_value": "JSON boolean",
            }
        ],
        "gold": ["exactly one option id"],
        "gold_answer_text": contract["gold"]["answer_text"],
        "normalized_gold_answer": "normalized gold answer",
        "anchor_refs": {
            "support": ["durable event ids"],
            "gold": [contract["gold"]["source_event_instance_id"]],
            "distractors": ["durable event ids for non-gold options"],
        },
        "query_plan": {
            "answer_source": "graph_anchor",
            "qa_operation": operation,
            "qa_subtype": qa.get("qa_subtype"),
            "answer_type": answer_type,
            **copy.deepcopy(reasoning_schema),
            "reasoning_anchor_path": ["durable event ids"],
            "required_anchor_refs": ["durable event ids"],
            "supporting_anchor_refs": ["durable event ids"],
            "gold_anchor_refs": [contract["gold"]["source_event_instance_id"]],
            "distractor_anchor_refs": ["durable event ids"],
            "gold_answer": contract["gold"]["answer_text"],
            "normalized_gold_answer": "normalized gold answer",
        },
    }
    item = {
        "schema_version": SCHEMA_VERSION,
        "queue_id": queue_id,
        "source": {
            "recalibration_dir": str(batch_dir.resolve()),
            "sample_id": str(sample.get("sample_id") or ""),
            "lineage_group": str(sample.get("lineage_group") or ""),
            "status": solver_feedback["status"],
            "generation_seed": int(generation_seed),
            "stage64_target_provenance": copy.deepcopy(
                target_contract.get("provenance")
                if isinstance(target_contract, dict)
                else None
            ),
        },
        "target": {
            "mode": action_target["mode"],
            "action": action,
            "qa_operation": operation,
            "answer_type": answer_type,
            "qa_subtype": qa.get("qa_subtype"),
            "stage64_target": copy.deepcopy(
                target_contract.get("stage64_target")
                if isinstance(target_contract, dict)
                else None
            ),
            "stage64_reasoning_contract": copy.deepcopy(reasoning_contract),
            "stage64_target_provenance": copy.deepcopy(
                target_contract.get("provenance")
                if isinstance(target_contract, dict)
                else None
            ),
            "option_arity": len(qa.get("options") or []),
            "difficulty_target": action_target["difficulty_target"],
            "target_band": copy.deepcopy(TARGET_BAND),
            "goal": action_target["goal"],
        },
        "frozen_contract": contract,
        "challenger_request": {
            "task": "rewrite qa_proposal only using the frozen graph/dialogue and fixed Stage2 feedback",
            "requirements": [
                "Return only queue_id, frozen_bundle_sha256, and qa_proposal.",
                "Return one complete, self-contained qa_proposal object, not a patch or list of changed fields.",
                "Do not change, summarize, regenerate, or emit graph/dialogue fields.",
                "The graph/dialogue containers themselves are forbidden in the response, but QA-internal source_event_instance_id, anchor_refs, and query_plan references are mandatory.",
                "Preserve qa_operation, answer_type, gold answer text, and gold source event exactly.",
                "qa_proposal.gold must be a JSON array containing exactly the id of the sole truth_value=true option. That option's text, gold_answer_text, source_event_instance_id, anchor_ref, and query_plan/anchor_refs gold references must all identify the same frozen gold.",
                *(
                    [
                        "Preserve the immutable Stage64 reasoning target exactly: qa_proposal and "
                        "qa_proposal.query_plan must each echo reasoning_subtype, reasoning_template, "
                        "hop_count, hard_negative_strategy, and forbid_max_time_shortcut when supplied."
                    ]
                    if reasoning_contract is not None
                    else []
                ),
                "Use only event refs and facts supplied in event_candidates/frozen context; invent no facts.",
                "Make distractors plausible and reasoning-dependent, not lexically obvious.",
                "Do not mention solver predictions, correctness, answer labels, graph/schema terms, or this hardening task in user-visible text.",
                "Do not target one correct label: a deterministic shuffle and full cyclic K5 audit run afterward.",
                "Changing only metadata, option labels/order, casing, whitespace, or punctuation is a no-op and will be rejected. Substantively rewrite the question and/or at least one option's semantic content.",
                *(
                    [
                        "For hardening, the rewritten question must not repeat the normalized previous question. Make the reasoning demand materially different while preserving the frozen gold.",
                    ]
                    if action == "harden"
                    else []
                ),
                *(
                    [
                        "For temporal_state_update, require natural resolution of update validity/status and the full same-slot chain. Do not use stock answer-leading wording such as 'disregarding the canceled proposal' or 'rejecting the canceled update'.",
                    ]
                    if operation == "temporal_state_update"
                    else []
                ),
                rewrite_instruction,
            ],
            "previous_qa": copy.deepcopy(qa),
            "immutable_stage64_target_contract": copy.deepcopy(target_contract),
            "event_candidates": event_candidates,
            # Keep random generation anchors and alignment internals in the
            # local sidecar.  The challenger only needs the clean compiled
            # graph/dialogue plus durable event candidates.
            "frozen_graph": copy.deepcopy(context["graph_update"]),
            "frozen_dialogue": compact_evidence_dialogue(
                context["dialogue"], event_candidates
            ),
            "fixed_stage2_feedback": challenger_feedback_payload(
                solver_identity, solver_feedback
            ),
            "required_output_schema": {
                "queue_id": queue_id,
                "frozen_bundle_sha256": contract["frozen_bundle_sha256"],
                "qa_proposal": qa_output_schema,
            },
        },
    }
    frozen = {
        "schema_version": SCHEMA_VERSION,
        "queue_id": queue_id,
        "frozen_contract": contract,
        "frozen_bundle": context,
        "original_recalibrated_sample": copy.deepcopy(sample),
    }
    return item, frozen


def closed_loop_commands(
    *,
    queue_dir: Path,
    accepted_solver: str,
    base_model: str,
    config: dict[str, Any],
    generation_seed: int,
    final_eval_seed: int,
) -> list[str]:
    script = ROOT / "scripts/current_method/stage3_solver_feedback_mining.py"
    shuffle = ROOT / "scripts/current_method/shuffle_stage64_options.py"
    recalibrate = ROOT / "scripts/current_method/recalibrate_stage64_shuffled_frontier.py"
    validated = queue_dir / "validated"
    shuffled = queue_dir / "shuffled"
    recalibrated = queue_dir / "recalibrated"
    responses = queue_dir / "challenger_qa_responses.jsonl"
    quote = lambda value: shlex.quote(str(value))
    return [
        " ".join(
            [
                "python",
                quote(script),
                "validate",
                "--queue-dir",
                quote(queue_dir),
                "--responses",
                quote(responses),
                "--output-dir",
                quote(validated),
            ]
        ),
        " ".join(
            [
                "python",
                quote(shuffle),
                "--input",
                quote(validated / "revalidated_samples.jsonl"),
                "--output-dir",
                quote(shuffled),
                "--seed",
                str(final_eval_seed),
            ]
        ),
        " ".join(
            [
                "python",
                quote(recalibrate),
                "--input",
                quote(shuffled / "clean_samples.jsonl"),
                "--output-dir",
                quote(recalibrated),
                "--report-dir",
                quote(recalibrated / "reports"),
                "--base-model",
                quote(base_model),
                "--solver-model",
                quote(accepted_solver),
                "--accepted-stage2-solver-model",
                quote(accepted_solver),
                "--solver-tokenizer",
                quote(base_model),
                "--k-eval",
                str(config["k_eval"]),
                "--max-model-len",
                str(config["max_model_len"]),
                "--max-new-tokens",
                str(config["max_new_tokens"]),
                "--temperature",
                str(config["temperature"]),
                "--top-p",
                str(config["top_p"]),
                "--seed",
                str(final_eval_seed),
                "--generation-seed",
                str(generation_seed),
                "--permutation-mode",
                "cyclic",
                "--min-frontier-permutation-share",
                "1.0",
                "--heldout-ratio",
                "0.0",
            ]
        ),
    ]


def export_queue(
    recalibration_dirs: list[Path],
    output_dir: Path,
    *,
    accepted_stage2_solver_model: Path,
    final_eval_seed: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in (
        "queue_ready_manifest.json",
        "hardening_queue.jsonl",
        "frozen_samples.jsonl",
        "feedback_manifest.jsonl",
        "robust_frontier_candidates.jsonl",
        "diagnostic_hard0_samples.jsonl",
        "closed_loop_commands.sh",
    ):
        (output_dir / stale).unlink(missing_ok=True)
    all_items: list[dict[str, Any]] = []
    all_frozen: list[dict[str, Any]] = []
    feedback_records: list[dict[str, Any]] = []
    retained_frontier: list[dict[str, Any]] = []
    diagnostic_hard0: list[dict[str, Any]] = []
    batch_records: list[dict[str, Any]] = []
    common_config: dict[str, Any] | None = None
    common_solver: str | None = None
    common_base_model: str | None = None
    common_generation_seed: int | None = None
    dedupe_keys: set[tuple[str, str]] = set()
    status_counts: Counter[str] = Counter()
    rewrite_status_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    skipped_counts: Counter[str] = Counter()
    for directory in recalibration_dirs:
        directory = directory.resolve()
        summary_path = directory / "recalibration_summary.json"
        samples_path = directory / "clean_samples.jsonl"
        if not summary_path.is_file() or not samples_path.is_file():
            raise RuntimeError(f"Missing completed recalibration artifacts in {directory}")
        summary = read_json(summary_path)
        trust = validate_batch_summary(directory, summary, accepted_stage2_solver_model)
        generation_seed = summary.get("generation_seed")
        if isinstance(generation_seed, bool) or not isinstance(generation_seed, int):
            raise RuntimeError(
                f"Recalibration batch lacks an integer challenger generation_seed: {directory}"
            )
        if common_generation_seed is None:
            common_generation_seed = generation_seed
        elif generation_seed != common_generation_seed:
            raise RuntimeError(
                "QA-only export requires one challenger generation seed per queue; "
                "export different generation batches to separate queue directories"
            )
        if common_config is None:
            common_config = trust["solver_eval_config"]
        elif common_config != trust["solver_eval_config"]:
            raise RuntimeError("Fixed solver K5 configuration differs across recalibration batches")
        solver = str(summary.get("accepted_stage2_solver_resolved") or "")
        base_model = str(summary.get("base_model") or "")
        if common_solver is None:
            common_solver = solver
            common_base_model = base_model
        elif solver != common_solver or base_model != common_base_model:
            raise RuntimeError("Solver/base-model provenance differs across recalibration batches")
        marked_path = locate_marked_samples(directory)
        marked_rows = read_jsonl(marked_path)
        marked_by_id: dict[str, dict[str, Any]] = {}
        for bundle in marked_rows:
            sample_id = marked_bundle_id(bundle)
            if not sample_id or sample_id in marked_by_id:
                raise RuntimeError(f"Invalid or duplicate marked bundle sample_id in {marked_path}")
            marked_by_id[sample_id] = bundle
        samples = read_jsonl(samples_path)
        if len(samples) != int(summary.get("clean_count") or 0):
            raise RuntimeError(f"clean_samples count differs from recalibration summary in {directory}")
        selected_count = 0
        for sample in samples:
            try:
                stage3_operations.validate_generation_protocol(sample)
            except ValueError as exc:
                raise RuntimeError(
                    f"Audited sample {sample.get('sample_id')!r} has missing or stale Stage3 generation provenance: {exc}"
                ) from exc
            status = str((sample.get("stage64_permutation_eval") or {}).get("status") or "")
            if status not in AUDIT_STATUSES:
                raise RuntimeError(
                    f"Sample {sample.get('sample_id')!r} has missing/unknown cyclic status {status!r}"
                )
            sample_id = str(sample.get("sample_id") or "")
            bundle = marked_by_id.get(sample_id)
            if bundle is None:
                raise RuntimeError(
                    f"Audited sample {sample_id!r} has no original marked generation bundle; refusing reconstruction"
                )
            validate_marked_bundle(sample, bundle)
            operation = str((sample.get("qa") or {}).get("qa_operation") or sample.get("qa_operation") or "")
            frozen_target = stage64_target_freeze_payload(sample)
            graph_issues = native.semantic_graph_plan_quality_issues(
                bundle["current_event_plan"],
                operation,
                copy.deepcopy(
                    frozen_target.get("stage64_target")
                    if isinstance(frozen_target, dict)
                    else None
                ),
            )
            if graph_issues:
                raise RuntimeError(
                    f"Audited sample {sample_id!r} fails the current {GRAPH_CONTRACT_LABEL} graph gate: "
                    f"{graph_issues[:4]}"
                )
            semantic_issues = stage64_quality.semantic_quality_issues(sample)
            if semantic_issues:
                raise RuntimeError(
                    f"Audited sample {sample_id!r} fails the current {GRAPH_CONTRACT_LABEL} gate: {semantic_issues[:4]}"
                )
            feedback = validate_cyclic_audit(sample)
            contract = frozen_contract(sample, bundle)
            dedupe_key = (str(sample.get("lineage_group") or ""), contract["graph_sha256"])
            if dedupe_key in dedupe_keys:
                skipped_counts["duplicate_lineage_graph_root"] += 1
                continue
            dedupe_keys.add(dedupe_key)
            solver_identity = {
                "accepted_stage2_solver_resolved": solver,
                "base_model": base_model,
                "adapter": None,
                "sampling": copy.deepcopy(common_config),
            }
            queue_id: str | None = None
            if status in ELIGIBLE_STATUSES:
                item, frozen = queue_item(
                    sample,
                    bundle,
                    batch_dir=directory,
                    generation_seed=generation_seed,
                    solver_identity=solver_identity,
                    solver_feedback=feedback,
                )
                queue_id = item["queue_id"]
                all_items.append(item)
                all_frozen.append(frozen)
                rewrite_status_counts[status] += 1
            elif status == "robust_frontier":
                retained_frontier.append(copy.deepcopy(sample))
            if status == "robust_hard0":
                diagnostic_hard0.append(copy.deepcopy(sample))
            action = str(feedback["decision"]["action"])
            action_counts[action] += 1
            feedback_records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "sample_id": sample_id,
                    "lineage_group": str(sample.get("lineage_group") or ""),
                    "recalibration_dir": str(directory),
                    "queue_id": queue_id,
                    "frozen_contract": contract,
                    "solver": copy.deepcopy(solver_identity),
                    "solver_feedback": copy.deepcopy(feedback),
                    "graph_dialogue_mutation_allowed": False,
                    "gold_semantics_mutation_allowed": False,
                }
            )
            status_counts[status] += 1
            selected_count += 1
        batch_records.append(
            {
                "recalibration_dir": str(directory),
                "summary_sha256": sha256_file(summary_path),
                "samples_sha256": sha256_file(samples_path),
                "marked_samples": str(marked_path.resolve()),
                "marked_samples_sha256": sha256_file(marked_path),
                "clean_count": len(samples),
                "marked_count": len(marked_rows),
                "selected_count": selected_count,
                "trust": trust,
            }
        )
    if (
        not feedback_records
        or common_config is None
        or common_solver is None
        or common_base_model is None
        or common_generation_seed is None
    ):
        raise RuntimeError("No valid cyclic-audit samples were exported")
    search_eval_seed = int(common_config["seed"])
    if int(final_eval_seed) == search_eval_seed:
        raise RuntimeError(
            "final_eval_seed must differ from the search/evolution solver seed"
        )
    all_items.sort(key=lambda row: row["queue_id"])
    all_frozen.sort(key=lambda row: row["queue_id"])
    retained_frontier.sort(key=lambda row: str(row.get("sample_id") or ""))
    diagnostic_hard0.sort(key=lambda row: str(row.get("sample_id") or ""))
    feedback_records.sort(key=lambda row: (str(row["sample_id"]), str(row.get("queue_id") or "")))
    duplicate_counts = Counter(
        str(record["solver_feedback"]["visible_qa_fingerprint"])
        for record in feedback_records
    )
    item_by_id = {str(item["queue_id"]): item for item in all_items}
    for record in feedback_records:
        feedback = record["solver_feedback"]
        duplicate_count = duplicate_counts[str(feedback["visible_qa_fingerprint"])]
        _score_feedback(feedback, duplicate_count=duplicate_count)
        queue_id = str(record.get("queue_id") or "")
        if queue_id:
            fixed_feedback = item_by_id[queue_id]["challenger_request"]["fixed_stage2_feedback"]
            solver_identity = copy.deepcopy(fixed_feedback["solver"])
            fixed_feedback.clear()
            fixed_feedback.update(
                challenger_feedback_payload(solver_identity, feedback)
            )
    queue_ids = [row["queue_id"] for row in all_items]
    if len(queue_ids) != len(set(queue_ids)):
        raise RuntimeError("Duplicate hardening queue_id")
    write_jsonl(output_dir / "hardening_queue.jsonl", all_items)
    write_jsonl(output_dir / "frozen_samples.jsonl", all_frozen)
    write_jsonl(output_dir / "feedback_manifest.jsonl", feedback_records)
    write_jsonl(output_dir / "robust_frontier_candidates.jsonl", retained_frontier)
    write_jsonl(output_dir / "diagnostic_hard0_samples.jsonl", diagnostic_hard0)
    commands = (
        closed_loop_commands(
            queue_dir=output_dir.resolve(),
            accepted_solver=common_solver,
            base_model=common_base_model,
            config=common_config,
            generation_seed=common_generation_seed,
            final_eval_seed=int(final_eval_seed),
        )
        if all_items
        else []
    )
    (output_dir / "closed_loop_commands.sh").write_text("\n\n".join(commands) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "mode": "offline_qa_only_solver_feedback_export",
        "graph_contract_gate": GRAPH_CONTRACT_LABEL,
        "graph_contract_gate_code_sha256": sha256_file(ROOT / "scripts/current_method/stage64_quality.py"),
        "queue_count": len(all_items),
        "status_counts": dict(sorted(status_counts.items())),
        "rewrite_status_counts": dict(sorted(rewrite_status_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "feedback_manifest_count": len(feedback_records),
        "retained_frontier_count": len(retained_frontier),
        "diagnostic_hard0_count": len(diagnostic_hard0),
        "skipped_counts": dict(sorted(skipped_counts.items())),
        "eligible_statuses": list(ELIGIBLE_STATUSES),
        "audited_statuses": list(AUDIT_STATUSES),
        "target_band": copy.deepcopy(TARGET_BAND),
        "difficulty_score": {
            "formula": "max(0, boundary_reward - duplicate_penalty - position_penalty)",
            "duplicate_penalty_weight": DUPLICATE_PENALTY_WEIGHT,
            "position_penalty_weight": POSITION_PENALTY_WEIGHT,
        },
        "batches": batch_records,
        "fixed_solver": {
            "accepted_stage2_solver_resolved": common_solver,
            "base_model": common_base_model,
            "adapter": None,
            "sampling": common_config,
        },
        "search_eval_seed": search_eval_seed,
        "generation_seed": common_generation_seed,
        "final_eval_seed": int(final_eval_seed),
        "independent_final_eval_seed": True,
        "invariants": {
            "marked_generation_bundle_required": True,
            "graph_dialogue_frozen": True,
            "gold_semantics_frozen": True,
            "challenger_may_edit_only_qa_proposal": True,
            "cyclic_k5_must_be_rerun_after_rewrite": True,
            "one_candidate_per_lineage_graph_root": True,
            "robust_frontier_retained_without_rewrite": True,
            "hard0_preserved_as_diagnostic_before_softening": True,
        },
        "automatic_api_calls": False,
        "automatic_solver_calls": False,
        "automatic_training": False,
        "training_authorized": False,
        "response_file": str((output_dir / "challenger_qa_responses.jsonl").resolve()),
        "feedback_manifest": str((output_dir / "feedback_manifest.jsonl").resolve()),
        "robust_frontier_candidates": str((output_dir / "robust_frontier_candidates.jsonl").resolve()),
        "diagnostic_hard0_samples": str((output_dir / "diagnostic_hard0_samples.jsonl").resolve()),
        "response_schema": {
            "queue_id": "exact queue id",
            "frozen_bundle_sha256": "exact frozen contract value",
            "qa_proposal": "QA-only object",
        },
        "closed_loop_commands": commands,
    }
    write_json(output_dir / "queue_ready_manifest.json", manifest)
    return manifest


def recursive_forbidden_keys(value: Any, *, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if str(key).lower() in FORBIDDEN_RESPONSE_KEYS:
                found.append(child_path)
            found.extend(recursive_forbidden_keys(child, path=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(recursive_forbidden_keys(child, path=f"{path}[{index}]"))
    return found


def verify_frozen_wrapper(wrapper: dict[str, Any]) -> None:
    contract = wrapper.get("frozen_contract") if isinstance(wrapper.get("frozen_contract"), dict) else {}
    bundle = wrapper.get("frozen_bundle") if isinstance(wrapper.get("frozen_bundle"), dict) else {}
    sample = wrapper.get("original_recalibrated_sample")
    if not isinstance(sample, dict) or not bundle:
        raise RuntimeError("Frozen wrapper is incomplete")
    unsigned_contract = copy.deepcopy(contract)
    stored_contract_sha256 = unsigned_contract.pop("sha256", None)
    if sha256_json(unsigned_contract) != stored_contract_sha256:
        raise RuntimeError("Frozen contract fingerprint mismatch")
    if sha256_json(bundle) != contract.get("frozen_bundle_sha256"):
        raise RuntimeError("Frozen bundle fingerprint mismatch")
    expected_graph = bundle.get("graph_update")
    expected_dialogue = bundle.get("dialogue")
    if sample.get("graph_update") != expected_graph or sample.get("dialogue") != expected_dialogue:
        raise RuntimeError("Frozen sample graph/dialogue differs from frozen bundle")
    if sha256_json(expected_graph) != contract.get("graph_sha256"):
        raise RuntimeError("Frozen graph fingerprint mismatch")
    if sha256_json(expected_dialogue) != contract.get("dialogue_sha256"):
        raise RuntimeError("Frozen dialogue fingerprint mismatch")
    if qa_gold_contract(sample) != contract.get("gold"):
        raise RuntimeError("Frozen gold contract mismatch")
    sample_target = stage64_target_freeze_payload(sample)
    if sample_target != contract.get("stage64_target_contract"):
        raise RuntimeError("Frozen Stage64 target contract mismatch")
    if sample_target != bundle.get("stage64_target_contract"):
        raise RuntimeError("Frozen Stage64 target bundle mismatch")


def repaired_gold_contract(qa: dict[str, Any]) -> dict[str, Any]:
    return qa_gold_contract({"sample_id": "repaired", "qa": qa})


def complete_qa_hardening_metadata(
    wrapper: dict[str, Any],
    qa_proposal: dict[str, Any],
) -> dict[str, Any]:
    """Fill only frozen or independently corroborated QA metadata."""

    if not isinstance(qa_proposal, dict):
        raise RuntimeError("QA hardening proposal must be an object")
    original = wrapper["original_recalibrated_sample"]
    original_qa = original.get("qa") if isinstance(original.get("qa"), dict) else {}
    contract = wrapper["frozen_contract"]
    old_gold = contract["gold"]
    target_contract = contract.get("stage64_target_contract")
    reasoning_contract = (
        target_contract.get("reasoning_contract")
        if isinstance(target_contract, dict)
        and isinstance(target_contract.get("reasoning_contract"), dict)
        else None
    )
    expected_operation = str(old_gold["qa_operation"])
    expected_subtype = str(original_qa.get("qa_subtype") or "")
    expected_answer_type = str(old_gold["answer_type"])
    expected_gold_answer = str(old_gold["answer_text"])
    if not expected_subtype:
        raise RuntimeError("Frozen QA lacks qa_subtype for deterministic completion")

    normalized = copy.deepcopy(qa_proposal)

    # API responses sometimes preserve every option label/content but serialize the
    # list out of A-D order. Canonicalize only the unambiguous structural case; all
    # missing, duplicate, or conflicting labels remain invalid for the native gate.
    raw_options = normalized.get("options")
    canonical_option_labels = native.LETTERS[:4]
    if (
        isinstance(raw_options, list)
        and len(raw_options) == len(canonical_option_labels)
        and all(isinstance(option, dict) for option in raw_options)
    ):
        option_by_label: dict[str, dict[str, Any]] = {}
        labels_are_unambiguous = True
        for option in raw_options:
            label = native.normalize_space(option.get("id")).upper()
            option_id = native.normalize_space(option.get("option_id")).upper()
            if option_id and option_id != label:
                raise RuntimeError(
                    "QA hardening explicit conflict between option.id and option.option_id"
                )
            if label not in canonical_option_labels or label in option_by_label:
                labels_are_unambiguous = False
                break
            option_by_label[label] = option
        if labels_are_unambiguous and set(option_by_label) == set(
            canonical_option_labels
        ):
            normalized["options"] = [
                copy.deepcopy(option_by_label[label])
                for label in canonical_option_labels
            ]

    def fill_exact(
        container: dict[str, Any],
        field: str,
        expected: Any,
        *,
        owner: str,
        normalized_equivalence: bool = False,
    ) -> None:
        if field not in container:
            container[field] = copy.deepcopy(expected)
            return
        actual = container.get(field)
        matches = (
            isinstance(actual, str)
            and isinstance(expected, str)
            and normalized_text(actual) == normalized_text(expected)
            if normalized_equivalence
            else type(actual) is type(expected) and actual == expected
        )
        if not matches:
            raise RuntimeError(
                f"QA hardening explicit conflict in {owner}.{field}: "
                f"actual={actual!r}, expected={expected!r}"
            )

    for field, expected in (
        ("qa_operation", expected_operation),
        ("qa_subtype", expected_subtype),
        ("answer_type", expected_answer_type),
        ("question_type", "single_choice"),
        ("answer_format", "single_choice_mcq"),
        ("gold_answer_text", expected_gold_answer),
    ):
        fill_exact(normalized, field, expected, owner="qa_proposal")
    fill_exact(
        normalized,
        "normalized_gold_answer",
        normalized_text(expected_gold_answer),
        owner="qa_proposal",
        normalized_equivalence=True,
    )

    if reasoning_contract is not None:
        for field in STAGE64_REASONING_TARGET_FIELDS:
            fill_exact(
                normalized,
                field,
                reasoning_contract[field],
                owner="qa_proposal",
            )
        if "forbid_max_time_shortcut" in reasoning_contract:
            fill_exact(
                normalized,
                "forbid_max_time_shortcut",
                reasoning_contract["forbid_max_time_shortcut"],
                owner="qa_proposal",
            )

    anchor_refs = normalized.get("anchor_refs")
    if not isinstance(anchor_refs, dict):
        raise RuntimeError(
            "QA hardening metadata completion requires explicit anchor_refs"
        )
    bundle = wrapper["frozen_bundle"]
    alias_to_event_id, _turn_map = native._stage64_ref_maps(  # noqa: SLF001
        bundle["current_event_plan"],
        bundle["graph_update"],
        bundle["current_alignment"],
    )

    def resolved_group(field: str) -> list[str]:
        raw = anchor_refs.get(field)
        if not isinstance(raw, list) or not raw:
            raise RuntimeError(
                f"QA hardening metadata completion requires anchor_refs.{field}"
            )
        resolved: list[str] = []
        for ref in raw:
            event_id = native._resolve_stage64_ref(  # noqa: SLF001
                str(ref), alias_to_event_id
            )
            if not event_id or event_id in resolved:
                raise RuntimeError(
                    f"QA hardening anchor_refs.{field} is unresolved or duplicated"
                )
            resolved.append(event_id)
        return resolved

    support_ids = resolved_group("support")
    gold_ids = resolved_group("gold")
    distractor_ids = resolved_group("distractors")
    if old_gold["source_event_instance_id"] not in gold_ids:
        raise RuntimeError("QA hardening anchor_refs.gold conflicts with frozen gold source")

    query_plan = normalized.get("query_plan")
    if query_plan is None and "query_plan" in normalized:
        raise RuntimeError("QA hardening query_plan must be an object when explicit")
    if query_plan is None:
        normalized["query_plan"] = {}
        query_plan = normalized["query_plan"]
    if not isinstance(query_plan, dict):
        raise RuntimeError("QA hardening query_plan must be an object")

    query_scalar_fields = {
        "answer_source": "graph_anchor",
        "qa_operation": expected_operation,
        "qa_subtype": expected_subtype,
        "answer_type": expected_answer_type,
        "gold_answer": expected_gold_answer,
        "normalized_gold_answer": normalized_text(expected_gold_answer),
    }
    if reasoning_contract is not None:
        query_scalar_fields.update(
            {
                field: copy.deepcopy(reasoning_contract[field])
                for field in STAGE64_REASONING_TARGET_FIELDS
            }
        )
        if "forbid_max_time_shortcut" in reasoning_contract:
            query_scalar_fields["forbid_max_time_shortcut"] = copy.deepcopy(
                reasoning_contract["forbid_max_time_shortcut"]
            )
    for field, expected in query_scalar_fields.items():
        fill_exact(
            query_plan,
            field,
            expected,
            owner="qa_proposal.query_plan",
            normalized_equivalence=field == "normalized_gold_answer",
        )

    expected_query_refs = {
        "reasoning_anchor_path": support_ids,
        "required_anchor_refs": support_ids,
        "supporting_anchor_refs": support_ids,
        "gold_anchor_refs": gold_ids,
        "distractor_anchor_refs": distractor_ids,
    }
    for field, expected_ids in expected_query_refs.items():
        if field not in query_plan:
            query_plan[field] = copy.deepcopy(expected_ids)
            continue
        if not native._stage64_query_plan_completion_exact_refs(  # noqa: SLF001
            query_plan.get(field),
            expected_ids,
            alias_to_event_id,
            ordered=field == "reasoning_anchor_path",
        ):
            raise RuntimeError(
                f"QA hardening explicit conflict in qa_proposal.query_plan.{field}"
            )

    resolved_expectations = {
        "resolved_reasoning_event_instance_ids": support_ids,
        "resolved_required_event_instance_ids": support_ids,
        "resolved_supporting_event_instance_ids": support_ids,
        "resolved_gold_event_instance_ids": gold_ids,
        "resolved_distractor_event_instance_ids": distractor_ids,
    }
    for field, expected_ids in resolved_expectations.items():
        if field not in query_plan:
            continue
        if not native._stage64_query_plan_completion_exact_refs(  # noqa: SLF001
            query_plan.get(field),
            expected_ids,
            alias_to_event_id,
            ordered=field == "resolved_reasoning_event_instance_ids",
        ):
            raise RuntimeError(
                f"QA hardening explicit conflict in qa_proposal.query_plan.{field}"
            )
    return normalized


def qa_visible_fingerprint(qa: dict[str, Any]) -> str:
    options = qa.get("options") if isinstance(qa.get("options"), list) else []
    return sha256_json(
        {
            "question": normalized_text(qa.get("question") or qa.get("question_text")),
            "options": sorted(
                normalized_text(option.get("text"))
                for option in options
                if isinstance(option, dict)
            ),
        }
    )


def synchronize_repaired_sample(
    original: dict[str, Any],
    qa: dict[str, Any],
    *,
    queue_id: str,
    frozen_bundle_sha256: str,
) -> dict[str, Any]:
    row = copy.deepcopy(original)
    target_contract = stage64_target_freeze_payload(original)
    for key in STALE_SAMPLE_FIELDS:
        row.pop(key, None)
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for key in STALE_METADATA_FIELDS:
        metadata.pop(key, None)
    row["metadata"] = metadata
    row["qa"] = copy.deepcopy(qa)
    row.update(
        {
            "qa_operation": qa.get("qa_operation"),
            "qa_subtype": qa.get("qa_subtype"),
            **{
                field: copy.deepcopy(qa.get(field))
                for field in STAGE64_REASONING_TARGET_FIELDS
                if field in qa
            },
            **(
                {
                    "forbid_max_time_shortcut": copy.deepcopy(
                        qa.get("forbid_max_time_shortcut")
                    )
                }
                if "forbid_max_time_shortcut" in qa
                else {}
            ),
            "answer_type": qa.get("answer_type"),
            "answer_format": qa.get("answer_format"),
            "question": qa.get("question"),
            "question_text": qa.get("question_text") or qa.get("question"),
            "question_type": qa.get("question_type"),
            "query_plan": copy.deepcopy(qa.get("query_plan")),
            "anchor_refs": copy.deepcopy(qa.get("anchor_refs")),
            "gold_option": (qa.get("gold") or [None])[0],
            "gold_answer_text": qa.get("gold_answer_text"),
            "normalized_gold_answer": qa.get("normalized_gold_answer"),
            "evidence_turn_ids": copy.deepcopy(qa.get("evidence_turn_ids") or []),
            "difficulty_subtype": qa.get("difficulty_subtype"),
            "hard_single_choice_template": qa.get("hard_single_choice_template"),
            "distractor_error_types": copy.deepcopy(qa.get("distractor_error_types") or []),
            "graph_validation_pass": True,
            "attack_skill": qa.get("qa_operation"),
        }
    )
    native.apply_top_level_qa_compat_fields(row)
    attack_plan = row.get("attack_plan") if isinstance(row.get("attack_plan"), dict) else {}
    attack_plan.update(
        {
            "attack_skill": qa.get("qa_operation"),
            "difficulty_subtype": qa.get("qa_subtype") or qa.get("difficulty_subtype"),
            "hard_single_choice_template": qa.get("hard_single_choice_template"),
            "reasoning_mechanism": "qa_only_solver_feedback_hardening_on_frozen_graph_dialogue",
            "target_solver_weakness": "fixed_stage2_cyclic_k5_feedback",
        }
    )
    row["attack_plan"] = attack_plan
    metadata.update(
        {
            "qa_operation": qa.get("qa_operation"),
            "qa_subtype": qa.get("qa_subtype"),
            **{
                field: copy.deepcopy(qa.get(field))
                for field in STAGE64_REASONING_TARGET_FIELDS
                if field in qa
            },
            **(
                {
                    "forbid_max_time_shortcut": copy.deepcopy(
                        qa.get("forbid_max_time_shortcut")
                    )
                }
                if "forbid_max_time_shortcut" in qa
                else {}
            ),
            "answer_type": qa.get("answer_type"),
            "answer_format": qa.get("answer_format"),
            "difficulty_subtype": qa.get("difficulty_subtype"),
            "hard_single_choice_template": qa.get("hard_single_choice_template"),
            "distractor_error_types": copy.deepcopy(qa.get("distractor_error_types") or []),
            "graph_validation_pass": True,
            "attack_skill": qa.get("qa_operation"),
            "qa_only_solver_feedback_hardening": True,
            "qa_only_hardening_queue_id": queue_id,
            "qa_only_hardening_frozen_bundle_sha256": frozen_bundle_sha256,
            "qa_only_hardening_requires_new_cyclic_k5": True,
        }
    )
    source_sample_id = str(original.get("sample_id") or "")
    row["sample_id"] = f"{source_sample_id}::qa_hardened_{sha256_json(qa)[:12]}"
    row["qa_hardening_provenance"] = {
        "schema_version": SCHEMA_VERSION,
        "queue_id": queue_id,
        "source_sample_id": source_sample_id,
        "frozen_bundle_sha256": frozen_bundle_sha256,
        "stage64_target_contract": copy.deepcopy(target_contract),
        "graph_dialogue_unchanged": True,
        "gold_semantics_unchanged": True,
        "stale_solver_eval_removed": True,
        "requires_option_shuffle_and_fixed_stage2_cyclic_k5": True,
    }
    row["validation"] = native.base.validate_sample(row)
    return row


def build_repaired_sample(
    wrapper: dict[str, Any],
    qa_proposal: dict[str, Any],
) -> dict[str, Any]:
    verify_frozen_wrapper(wrapper)
    bundle = wrapper["frozen_bundle"]
    original = wrapper["original_recalibrated_sample"]
    contract = wrapper["frozen_contract"]
    old_gold = contract["gold"]
    target_contract = contract.get("stage64_target_contract")
    original_qa = original.get("qa") if isinstance(original.get("qa"), dict) else {}
    qa_proposal = complete_qa_hardening_metadata(wrapper, qa_proposal)
    if qa_visible_fingerprint(qa_proposal) == qa_visible_fingerprint(original_qa):
        raise RuntimeError(
            "QA hardening returned a metadata-only or reordered no-op; "
            "question/options are semantically unchanged"
        )
    if str(qa_proposal.get("qa_operation") or "") != old_gold["qa_operation"]:
        raise RuntimeError("QA hardening changed qa_operation")
    if str(qa_proposal.get("answer_type") or "") != old_gold["answer_type"]:
        raise RuntimeError("QA hardening changed answer_type")
    validate_qa_reasoning_contract(
        qa_proposal,
        target_contract,
        owner="QA hardening proposal",
    )
    graph_issues = native.semantic_graph_plan_quality_issues(
        bundle["current_event_plan"],
        old_gold["qa_operation"],
        copy.deepcopy(
            target_contract.get("stage64_target")
            if isinstance(target_contract, dict)
            else None
        ),
    )
    if graph_issues:
        raise RuntimeError(f"Frozen graph failed the current {GRAPH_CONTRACT_LABEL} graph gate: {graph_issues[:4]}")
    qa = native._build_stage64_api_qa_from_proposal(  # noqa: SLF001
        current_event_plan=bundle["current_event_plan"],
        graph_update=bundle["graph_update"],
        current_alignment=bundle["current_alignment"],
        qa_proposal=qa_proposal,
        hashes=bundle["hashes"],
        stage64_target=copy.deepcopy(
            target_contract.get("stage64_target")
            if isinstance(target_contract, dict)
            else {"qa_operation": old_gold["qa_operation"]}
        ),
    )
    validate_qa_reasoning_contract(
        qa,
        target_contract,
        owner="Materialized QA",
    )
    new_gold = repaired_gold_contract(qa)
    for key in ("normalized_answer_text", "source_event_instance_id", "qa_operation", "answer_type"):
        if new_gold[key] != old_gold[key]:
            raise RuntimeError(f"QA hardening changed frozen gold field {key}")
    if qa_visible_fingerprint(qa) == qa_visible_fingerprint(original_qa):
        raise RuntimeError(
            "QA hardening returned the original user-visible question/options unchanged"
        )
    row = synchronize_repaired_sample(
        original,
        qa,
        queue_id=wrapper["queue_id"],
        frozen_bundle_sha256=contract["frozen_bundle_sha256"],
    )
    if row.get("graph_update") != bundle["graph_update"] or row.get("dialogue") != bundle["dialogue"]:
        raise RuntimeError("QA materialization changed frozen graph/dialogue")
    if stage64_target_freeze_payload(row) != target_contract:
        raise RuntimeError("QA materialization changed frozen Stage64 target/provenance")
    issues = stage64_quality.semantic_quality_issues(row)
    if issues:
        raise RuntimeError(f"Repaired QA failed {GRAPH_CONTRACT_LABEL} semantic gate: {issues[:4]}")
    known_anchors = list((bundle.get("current_anchor_to_event_instance_id") or {}).keys())
    native.validate_final_clean_sample(row, known_anchors, bundle["hashes"])
    return row


def validate_responses(
    queue_dir: Path,
    responses_path: Path,
    output_dir: Path,
    *,
    allow_partial: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ready_path = output_dir / "revalidated_samples.jsonl"
    ready_manifest_path = output_dir / "ready_for_shuffle_manifest.json"
    ready_path.unlink(missing_ok=True)
    ready_manifest_path.unlink(missing_ok=True)
    manifest = read_json(queue_dir / "queue_ready_manifest.json")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("training_authorized") is not False:
        raise RuntimeError("Queue manifest is not a trusted offline QA-only manifest")
    queue_rows = read_jsonl(queue_dir / "hardening_queue.jsonl")
    frozen_rows = read_jsonl(queue_dir / "frozen_samples.jsonl")
    queue_by_id = {str(row.get("queue_id") or ""): row for row in queue_rows}
    frozen_by_id = {str(row.get("queue_id") or ""): row for row in frozen_rows}
    if not queue_by_id or len(queue_by_id) != len(queue_rows) or set(queue_by_id) != set(frozen_by_id):
        raise RuntimeError("Queue/frozen bundle IDs are empty, duplicate, or inconsistent")
    responses = read_jsonl(responses_path)
    response_by_id: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    for response in responses:
        queue_id = str(response.get("queue_id") or "")
        if queue_id in response_by_id:
            raise RuntimeError(f"Duplicate challenger response for {queue_id!r}")
        response_by_id[queue_id] = response
    unknown_ids = sorted(set(response_by_id) - set(queue_by_id))
    if unknown_ids:
        raise RuntimeError(f"Responses contain unknown queue IDs: {unknown_ids[:5]}")
    missing_ids = sorted(set(queue_by_id) - set(response_by_id))
    accepted: list[dict[str, Any]] = []
    for queue_id, response in sorted(response_by_id.items()):
        try:
            extra_keys = sorted(set(response) - RESPONSE_KEYS)
            if extra_keys:
                raise RuntimeError(f"Response emitted forbidden top-level keys: {extra_keys}")
            forbidden = recursive_forbidden_keys(response.get("qa_proposal"))
            if forbidden:
                raise RuntimeError(f"qa_proposal emitted graph/dialogue/eval keys: {forbidden[:5]}")
            wrapper = frozen_by_id[queue_id]
            expected_hash = wrapper["frozen_contract"]["frozen_bundle_sha256"]
            if response.get("frozen_bundle_sha256") != expected_hash:
                raise RuntimeError("Response frozen_bundle_sha256 mismatch")
            proposal = response.get("qa_proposal")
            if not isinstance(proposal, dict):
                raise RuntimeError("Response lacks qa_proposal object")
            accepted.append(build_repaired_sample(wrapper, proposal))
        except Exception as exc:  # keep all rejected API work auditable
            rejected.append(
                {
                    "queue_id": queue_id,
                    "source_sample_id": queue_by_id[queue_id].get("source", {}).get("sample_id"),
                    "reason": str(exc),
                    "response": copy.deepcopy(response),
                }
            )
    write_jsonl(output_dir / "accepted_revalidated_samples.jsonl", accepted)
    write_jsonl(output_dir / "rejected_repairs.jsonl", rejected)
    complete = not missing_ids
    ready = bool(accepted) and not rejected and (complete or allow_partial)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "mode": "offline_revalidate_external_qa_proposals",
        "queue_count": len(queue_by_id),
        "response_count": len(responses),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "missing_response_count": len(missing_ids),
        "missing_queue_ids": missing_ids,
        "allow_partial": allow_partial,
        "ready_for_shuffle": ready,
        "graph_dialogue_frozen": True,
        "gold_semantics_frozen": True,
        "stale_solver_eval_removed": True,
        "automatic_solver_calls": False,
        "automatic_training": False,
        "training_authorized": False,
    }
    write_json(output_dir / "validation_summary.json", summary)
    if ready:
        write_jsonl(ready_path, accepted)
        write_json(
            ready_manifest_path,
            {
                **summary,
                "next_required_step": "deterministic option shuffle, then accepted Stage2 cyclic K5 recalibration",
            },
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline Stage3 QA-only solver-feedback mining orchestrator.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser(
        "export",
        help="Classify trusted cyclic K5 outcomes and export QA-only rewrite/retention manifests.",
    )
    export.add_argument("--recalibration-dir", type=Path, action="append", required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--accepted-stage2-solver-model", type=Path, required=True)
    export.add_argument("--final-eval-seed", type=int, required=True)
    generate = subparsers.add_parser("generate", help="Call the challenger for resumable QA-only responses.")
    generate.add_argument("--queue-dir", type=Path, required=True)
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--responses", type=Path, default=None)
    generate.add_argument("--timeout", type=int, default=180)
    generate.add_argument("--max-retries", type=int, default=1)
    generate.add_argument("--max-tokens", type=int, default=4096)
    generate.add_argument("--temperature", type=float, default=0.35)
    generate.add_argument("--max-items", type=int, default=None)
    validate = subparsers.add_parser("validate", help="Validate externally generated QA-only responses.")
    validate.add_argument("--queue-dir", type=Path, required=True)
    validate.add_argument("--responses", type=Path, required=True)
    validate.add_argument("--output-dir", type=Path, required=True)
    validate.add_argument(
        "--allow-partial",
        action="store_true",
        help="Publish valid returned rows even when some queue items have no response.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "export":
        summary = export_queue(
            args.recalibration_dir,
            args.output_dir,
            accepted_stage2_solver_model=args.accepted_stage2_solver_model,
            final_eval_seed=args.final_eval_seed,
        )
    elif args.command == "generate":
        summary = generate_qa_responses(
            args.queue_dir,
            args.config,
            args.responses or (args.queue_dir / "challenger_qa_responses.jsonl"),
            timeout=args.timeout,
            max_retries=args.max_retries,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            max_items=args.max_items,
        )
    else:
        summary = validate_responses(
            args.queue_dir,
            args.responses,
            args.output_dir,
            allow_partial=args.allow_partial,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("ready_for_shuffle", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
