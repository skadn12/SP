"""Deterministic, offline compilation of Round-1 feedback artifacts."""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..core import canonical_json, sha256_bytes, sha256_file
from ..h4_round0_b0_k3_difficulty_v1.io_utils import (
    load_json,
    load_jsonl,
    verify_sha_sidecar,
    write_once_json,
    write_once_jsonl,
    write_once_text,
)
from ..h4_round0_b0_k3_difficulty_v1.difficulty import summarize_candidate_result
from . import (
    ALGORITHM_VERSION,
    API_CONFIG_PATH,
    DEFAULT_RUN_ROOT,
    DEV_RUN,
    ENDPOINT_CONTEXT_LIMIT,
    EVIDENCE_RUN,
    EXPECTED_CANDIDATES,
    EXPECTED_ROOTS,
    EXPECTED_TECHNICAL,
    EXPANSION_OUTPUT_RESERVE,
    FORMAL_ROOT_COUNT,
    FUTURE_R02_DIALOGUE_RESERVE,
    FUTURE_SFT_LIMIT,
    FUTURE_SOLVER_FIXED_OVERHEAD,
    FUTURE_SOLVER_LIMIT,
    FUTURE_SOLVER_OUTPUT_RESERVE,
    M0_DIFFICULTY_RUN,
    RESERVE_ROOT_COUNT,
    ROUND1_PLAN_RUN,
    TOKENIZER_PATH,
    TRAINING_RUN,
    TRAIN_REEVAL_RUN,
    VERSION,
)


MODEL_NAMES = ("M0", "M_clean_policy", "M_inclusive_policy")
POST_MODELS = ("M_clean_policy", "M_inclusive_policy")
BAD_BUCKETS = {
    "hard_candidate", "full_zero_review", "full_context_worse",
    "technical_excluded",
}
FRONTIER_TRANSITIONS = {
    "EMERGING_FRONTIER_BOTH",
    "CONSENSUS_RESIDUAL",
    "EMERGING_FRONTIER_CLEAN_ONLY",
    "EMERGING_FRONTIER_INCLUSIVE_ONLY",
    "MODEL_DISAGREEMENT",
}
EMERGING_TRANSITIONS = {
    "EMERGING_FRONTIER_BOTH",
    "EMERGING_FRONTIER_CLEAN_ONLY",
    "EMERGING_FRONTIER_INCLUSIVE_ONLY",
}
PRIMARY_ORDER = (
    "PERSISTENT_FULL_ZERO",
    "M0_ZERO_TO_RESOLVED",
    "EMERGING_FRONTIER_BOTH",
    "EMERGING_FRONTIER_CLEAN_ONLY",
    "EMERGING_FRONTIER_INCLUSIVE_ONLY",
    "GRADUATED_BOTH",
    "CONSENSUS_RESIDUAL",
    "REGRESSED",
    "MODEL_DISAGREEMENT",
    "OTHER_STABLE",
)
ROUTES = (
    "EXPANSION_FEEDBACK_ELIGIBLE",
    "UNTOUCHED_PROBE_ELIGIBLE",
    "TRAINING_FEEDBACK_ONLY",
    "REPLAY",
    "RETIRE",
    "QUARANTINE",
)
GAP_TYPES = (
    "plan_vs_execution",
    "before_vs_after_state",
    "cause_vs_consequence",
    "temporal_precedence",
    "distributed_evidence_omission",
    "scope_boundary_confusion",
)

TRANSITION_PRIORITY = {
    "EMERGING_FRONTIER_BOTH": 0,
    "CONSENSUS_RESIDUAL": 1,
    "EMERGING_FRONTIER_CLEAN_ONLY": 2,
    "EMERGING_FRONTIER_INCLUSIVE_ONLY": 2,
    "MODEL_DISAGREEMENT": 3,
}
QTYPE_PRIORITY = {"ordering": 0, "multi_select": 1, "single_choice": 2}

EVENT_OPERATORS = {
    "plan_vs_execution": [
        "recall an existing plan",
        "introduce a new constraint",
        "state a revised decision",
        "show the revised action being executed",
        "show a separate observable consequence",
    ],
    "before_vs_after_state": [
        "establish the prior state",
        "introduce one explicit state-changing event",
        "state the new state",
        "demonstrate the new state in a separate action",
    ],
    "cause_vs_consequence": [
        "state a concrete cause",
        "show the immediate response",
        "show a distinct observable consequence",
        "preserve the causal direction",
    ],
    "temporal_precedence": [
        "recall an earlier event",
        "add a later trigger with a unique temporal anchor",
        "show the resulting action",
        "show a final consequence in a separate turn",
    ],
    "distributed_evidence_omission": [
        "establish one relevant fact",
        "add a second fact in a separate scene",
        "introduce a constraint linking both facts",
        "show an outcome that requires integrating both facts",
    ],
    "scope_boundary_confusion": [
        "state the original scope",
        "introduce a nearby but distinct alternative",
        "make the applicable scope explicit",
        "show one action whose ownership and scope are unambiguous",
    ],
}

EXPANSION_SYSTEM_CONTRACT = """You are planning one append-only R02 continuation.
The complete Parent+R01 transcript is authoritative. Memorax retrieval is a
non-authoritative recall aid. Use the feedback card only as an abstract ability
target: never repeat an old question, answer, option set, or label. Produce a
bounded, directly observable state transition with unique temporal anchors.
Preserve all established facts and speakers. The continuation must end with a
stable consequence that is already true, not merely promised. Output only the
frozen planner JSON contract; do not expose the feedback card in dialogue."""


def _stable_hash(*parts: Any) -> str:
    return sha256_bytes(canonical_json([str(x) for x in parts]).encode("utf-8"))


def _sha_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _verify_source(path: Path) -> str:
    return verify_sha_sidecar(path)


def _index(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        value = str(row.get(key) or "")
        if not value or value in result:
            raise RuntimeError(f"invalid_or_duplicate_{key}:{value}")
        result[value] = row
    return result


def _state(value: Mapping[str, Any]) -> str:
    accuracy = float(value["full_context_accuracy"])
    if accuracy == 0.0:
        return "zero"
    if accuracy == 1.0:
        return "full"
    if 0.0 < accuracy < 1.0:
        return "partial"
    raise RuntimeError(f"invalid_full_accuracy:{accuracy}")


def recompute_primary(row: Mapping[str, Any]) -> str:
    models = row["models"]
    if any(
        int(models[name].get(field) or 0) > 0
        for name in MODEL_NAMES
        for field in ("technical_failure_count", "parse_failure_count", "truncated_request_count")
    ):
        return "TECHNICAL"
    m0, clean, inclusive = (models[name] for name in MODEL_NAMES)
    m0_state, clean_state, inclusive_state = map(_state, (m0, clean, inclusive))
    if m0_state == "zero":
        if clean_state == inclusive_state == "zero":
            return "PERSISTENT_FULL_ZERO"
        if clean_state == "full" or inclusive_state == "full":
            return "M0_ZERO_TO_RESOLVED"
        if clean_state == inclusive_state == "partial":
            return "EMERGING_FRONTIER_BOTH"
        if clean_state == "partial":
            return "EMERGING_FRONTIER_CLEAN_ONLY"
        if inclusive_state == "partial":
            return "EMERGING_FRONTIER_INCLUSIVE_ONLY"
        raise RuntimeError("m0_zero_transition_not_classified")
    if clean_state == inclusive_state == "full":
        return "GRADUATED_BOTH"
    if (
        str(m0["difficulty_bucket"]) in {"easy", "full_context_resolved"}
        and (
            str(clean["difficulty_bucket"]) in BAD_BUCKETS
            or str(inclusive["difficulty_bucket"]) in BAD_BUCKETS
        )
    ):
        return "REGRESSED"
    if clean["difficulty_bucket"] == inclusive["difficulty_bucket"] == "hard_candidate":
        return "CONSENSUS_RESIDUAL"
    clean_signature = (
        clean["difficulty_bucket"], clean["full_context_accuracy"],
        clean["no_context_accuracy"],
    )
    inclusive_signature = (
        inclusive["difficulty_bucket"], inclusive["full_context_accuracy"],
        inclusive["no_context_accuracy"],
    )
    if clean_signature != inclusive_signature:
        return "MODEL_DISAGREEMENT"
    return "OTHER_STABLE"


def transition_flags(row: Mapping[str, Any]) -> dict[str, bool]:
    models = row["models"]
    clean, inclusive = (models[name] for name in POST_MODELS)
    mechanisms = {
        str(row.get("position_mechanism") or ""),
        str(clean.get("position_mechanism") or ""),
        str(inclusive.get("position_mechanism") or ""),
    }
    exposure = str(row["training_exposure"])
    return {
        "clean_partial": _state(clean) == "partial",
        "inclusive_partial": _state(inclusive) == "partial",
        "both_full_greater_than_no": (
            float(clean["full_context_accuracy"]) > float(clean["no_context_accuracy"])
            and float(inclusive["full_context_accuracy"]) > float(inclusive["no_context_accuracy"])
        ),
        "current_context_worse": (
            float(clean["full_context_accuracy"]) < float(clean["no_context_accuracy"])
            or float(inclusive["full_context_accuracy"]) < float(inclusive["no_context_accuracy"])
        ),
        "pure_position_only": str(row.get("position_mechanism") or "") == "position_only",
        "mixed": "mixed" in mechanisms,
        "unexposed_to_both": exposure == "UNEXPOSED_TO_BOTH",
        "directly_trained_by_clean": exposure in {"SHARED_TRAINED", "CLEAN_ONLY_TRAINED"},
        "directly_trained_by_inclusive": exposure in {"SHARED_TRAINED", "INCLUSIVE_ONLY_TRAINED"},
    }


def _flatten_full_predictions(result: Mapping[str, Any]) -> list[list[str]]:
    predictions: list[list[str]] = []
    for request in result.get("requests") or []:
        if request.get("context_mode") != "full_context" or request.get("scored") is not True:
            continue
        prediction = request.get("canonical_prediction")
        if isinstance(prediction, list):
            predictions.append([str(x) for x in prediction])
    return predictions


def _ordering_features(gold: Sequence[str], predictions: Sequence[Sequence[str]]) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    total_inversions = 0
    total_comparisons = 0
    for left_index in range(len(gold)):
        for right_index in range(left_index + 1, len(gold)):
            left, right = str(gold[left_index]), str(gold[right_index])
            comparable = inverted = 0
            for prediction in predictions:
                if left in prediction and right in prediction:
                    comparable += 1
                    inverted += int(prediction.index(left) > prediction.index(right))
            total_inversions += inverted
            total_comparisons += comparable
            pairs.append({
                "earlier_item": left,
                "later_item": right,
                "comparison_count": comparable,
                "inversion_count": inverted,
                "inversion_rate": inverted / comparable if comparable else None,
            })
    return {
        "pairwise_inversions": pairs,
        "total_comparison_count": total_comparisons,
        "total_inversion_count": total_inversions,
        "pairwise_inversion_rate": total_inversions / total_comparisons if total_comparisons else None,
    }


def _multi_features(
    gold: Sequence[str], option_ids: Sequence[str], predictions: Sequence[Sequence[str]],
) -> dict[str, Any]:
    gold_set = set(map(str, gold))
    denominator = len(predictions)
    rows = []
    for option_id in option_ids:
        false_negative = sum(option_id in gold_set and option_id not in p for p in predictions)
        false_positive = sum(option_id not in gold_set and option_id in p for p in predictions)
        rows.append({
            "canonical_option_id": option_id,
            "is_gold_member": option_id in gold_set,
            "false_negative_count": false_negative,
            "false_negative_rate": false_negative / denominator if denominator else None,
            "false_positive_count": false_positive,
            "false_positive_rate": false_positive / denominator if denominator else None,
        })
    return {"rollout_count": denominator, "canonical_option_error_rates": rows}


def _single_features(
    gold: Sequence[str], clean_predictions: Sequence[Sequence[str]],
    inclusive_predictions: Sequence[Sequence[str]],
) -> dict[str, Any]:
    gold_id = str(gold[0])
    counters = []
    for predictions in (clean_predictions, inclusive_predictions):
        counter = Counter(
            p[0] for p in predictions if len(p) == 1 and str(p[0]) != gold_id
        )
        counters.append(counter)
    dominant = []
    for counter in counters:
        if not counter:
            dominant.append(None)
        else:
            dominant.append(sorted(counter.items(), key=lambda x: (-x[1], x[0]))[0][0])
    common = dominant[0] if dominant[0] is not None and dominant[0] == dominant[1] else None
    return {
        "clean_dominant_canonical_distractor": dominant[0],
        "inclusive_dominant_canonical_distractor": dominant[1],
        "common_dominant_canonical_distractor": common,
        "clean_wrong_prediction_counts": dict(sorted(counters[0].items())),
        "inclusive_wrong_prediction_counts": dict(sorted(counters[1].items())),
    }


def infer_gap(candidate: Mapping[str, Any], feature: Mapping[str, Any]) -> tuple[str, list[str]]:
    qtype = str(candidate["question_type"])
    metadata = " ".join(
        str(candidate.get(key) or "")
        for key in (
            "required_reasoning", "hypothesized_reasoning_type",
            "discriminative_variable", "distractor_strategy",
        )
    ).casefold()
    if qtype == "ordering":
        if any(word in metadata for word in ("plan", "execution", "decision", "action")):
            return "plan_vs_execution", ["ordering", "plan_or_execution_metadata"]
        if any(word in metadata for word in ("state", "before", "after", "transition")):
            return "before_vs_after_state", ["ordering", "state_transition_metadata"]
        return "temporal_precedence", ["ordering", "temporal_sequence"]
    if any(word in metadata for word in ("cause", "causal", "consequence", "result")):
        return "cause_vs_consequence", ["causal_metadata"]
    if any(word in metadata for word in ("before", "after", "state", "transition", "change")):
        return "before_vs_after_state", ["state_transition_metadata"]
    if qtype == "multi_select" or len(candidate.get("evidence_refs") or []) > 1:
        return "distributed_evidence_omission", ["multi_or_distributed_evidence"]
    return "scope_boundary_confusion", ["single_choice_or_scope_distractor"]


def _candidate_feedback_feature(
    source: Mapping[str, Any], clean_result: Mapping[str, Any],
    inclusive_result: Mapping[str, Any], primary: str, flags: Mapping[str, bool],
) -> dict[str, Any]:
    candidate = source["candidate"]
    clean_predictions = _flatten_full_predictions(clean_result)
    inclusive_predictions = _flatten_full_predictions(inclusive_result)
    all_predictions = clean_predictions + inclusive_predictions
    qtype = str(source["question_type"])
    gold = [str(x) for x in candidate["gold"]]
    if qtype == "ordering":
        error_feature = _ordering_features(gold, all_predictions)
    elif qtype == "multi_select":
        error_feature = _multi_features(gold, sorted(candidate["options"]), all_predictions)
    else:
        error_feature = _single_features(gold, clean_predictions, inclusive_predictions)
    gap, reasons = infer_gap(candidate, error_feature)
    return {
        "feedback_feature_version": f"{VERSION}_candidate_features_v1",
        "candidate_id": source["candidate_id"],
        "candidate_sha256": _sha_json(candidate),
        "candidate_pool_sha256": source.get("candidate_pool_sha256"),
        "dialogue_sha256": source.get("dialogue_sha256"),
        "root_id": source["root_id"],
        "question_type": qtype,
        "generation_source": source["generation_source"],
        "primary_transition": primary,
        "evidence_status": source.get("evidence_status"),
        "training_exposure": source["training_exposure"],
        "required_reasoning": candidate.get("required_reasoning") or [],
        "hypothesized_reasoning_type": candidate.get("hypothesized_reasoning_type"),
        "discriminative_variable": candidate.get("discriminative_variable"),
        "distractor_strategy": candidate.get("distractor_strategy"),
        "evidence_refs": [str(x) for x in candidate.get("evidence_refs") or []],
        "gap_type": gap,
        "gap_mapping_reasons": reasons,
        "full_context_error_feature": error_feature,
        "full_context_rollout_count": len(all_predictions),
        "flags": dict(flags),
    }


def _reevaluated_model_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    """Derive full/no/bucket from raw offline requests, never old labels."""
    summary = summarize_candidate_result(result)
    requests = list(result.get("requests") or [])
    technical = sum(not bool(item.get("scored")) for item in requests)
    truncated = sum(item.get("parse_status") == "output_truncated" for item in requests)
    parsed_failure = sum(
        not bool(item.get("scored"))
        and item.get("failure_type") not in {"solver_error", "solver_output_truncated"}
        for item in requests
    )
    return {
        "difficulty_bucket": summary["difficulty_bucket"],
        "full_context_accuracy": summary["full_context_accuracy"],
        "full_context_correct_count": summary["full_context_correct_count"],
        "full_context_total_count": summary["full_context_total_count"],
        "no_context_accuracy": summary["no_context_accuracy"],
        "no_context_correct_count": summary["no_context_correct_count"],
        "no_context_total_count": summary["no_context_total_count"],
        "position_mechanism": summary["position_mechanism"],
        "context_gain": (
            float(summary["full_context_accuracy"])
            - float(summary["no_context_accuracy"])
        ),
        "canonical_flip_rate": summary["canonical_flip_rate"],
        "technical_failure_count": technical,
        "parse_failure_count": parsed_failure,
        "truncated_request_count": truncated,
    }


def _frontier_rank(row: Mapping[str, Any]) -> tuple[Any, ...]:
    flags = row["flags"]
    models = row["models"]
    context_gain = (
        float(models["M_clean_policy"]["context_gain"])
        + float(models["M_inclusive_policy"]["context_gain"])
    ) / 2.0
    return (
        TRANSITION_PRIORITY.get(str(row["primary_transition"]), 99),
        0 if flags["unexposed_to_both"] else 1,
        QTYPE_PRIORITY[str(row["question_type"])],
        0 if flags["mixed"] else 1,
        0 if flags["both_full_greater_than_no"] else 1,
        0 if row.get("evidence_status") == "EVIDENCE_CLEAN" else 1,
        -context_gain,
        _stable_hash(ALGORITHM_VERSION, row["candidate_id"]),
    )


def _safe_frontier(row: Mapping[str, Any], confirmed_invalid: set[str]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if row["primary_transition"] not in FRONTIER_TRANSITIONS:
        reasons.append("not_frontier_transition")
    if row["primary_transition"] == "TECHNICAL":
        reasons.append("technical")
    if row["primary_transition"] == "PERSISTENT_FULL_ZERO":
        reasons.append("persistent_full_zero")
    if row["flags"]["current_context_worse"]:
        reasons.append("current_context_worse")
    if row["flags"]["pure_position_only"]:
        reasons.append("pure_position_only")
    if row["candidate_id"] in confirmed_invalid:
        reasons.append("confirmed_defect")
    if row.get("evidence_status") in {"EVIDENCE_INVALID", "EVIDENCE_UNCLASSIFIABLE"}:
        reasons.append("programmatic_evidence_quality_blocker")
    return not reasons, reasons


def assign_routes(
    transitions: list[dict[str, Any]], confirmed_invalid: set[str],
) -> list[dict[str, Any]]:
    safe_by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    safe_ids: set[str] = set()
    exclusion_by_id: dict[str, list[str]] = {}
    for row in transitions:
        safe, reasons = _safe_frontier(row, confirmed_invalid)
        exclusion_by_id[row["candidate_id"]] = reasons
        if safe:
            safe_by_root[row["root_id"]].append(row)
            safe_ids.add(row["candidate_id"])
    expansion_ids: set[str] = set()
    for root_rows in safe_by_root.values():
        root_rows.sort(key=_frontier_rank)
        expansion_ids.add(root_rows[0]["candidate_id"])

    output = []
    for row in transitions:
        cid = row["candidate_id"]
        primary = row["primary_transition"]
        flags = row["flags"]
        if cid in expansion_ids:
            route, reason = "EXPANSION_FEEDBACK_ELIGIBLE", "highest_ranked_safe_frontier_in_root"
        elif cid in safe_ids:
            route, reason = "UNTOUCHED_PROBE_ELIGIBLE", "safe_frontier_held_out_from_feedback"
        elif primary == "PERSISTENT_FULL_ZERO" or cid in confirmed_invalid or row.get("evidence_status") in {"EVIDENCE_INVALID", "EVIDENCE_UNCLASSIFIABLE"}:
            route, reason = "QUARANTINE", "persistent_zero_or_quality_blocker"
        elif flags["pure_position_only"] or flags["current_context_worse"]:
            route, reason = "TRAINING_FEEDBACK_ONLY", "position_or_context_behavior_not_expansion_seed"
        elif primary == "REGRESSED":
            route, reason = "REPLAY", "post_training_regression"
        elif primary in {"GRADUATED_BOTH", "M0_ZERO_TO_RESOLVED"}:
            replay = int(_stable_hash("stable_replay", cid)[:8], 16) % 20 == 0
            route = "REPLAY" if replay else "RETIRE"
            reason = "stable_solution_replay_sample" if replay else "stable_solution_retired"
        else:
            route, reason = "RETIRE", "not_selected_for_expansion_feedback"
        output.append({
            "routing_version": f"{VERSION}_routing_v1",
            "candidate_id": cid,
            "root_id": row["root_id"],
            "question_type": row["question_type"],
            "generation_source": row["generation_source"],
            "primary_transition": primary,
            "gap_type": row["gap_type"],
            "evidence_status": row.get("evidence_status"),
            "evidence_confidence": "reduced" if row.get("evidence_status") == "EVIDENCE_RISK" else "normal",
            "training_exposure": row["training_exposure"],
            "position_mechanism": row["position_mechanism"],
            "lifecycle_route": route,
            "route_reason": reason,
            "expansion_exclusion_reasons": exclusion_by_id[cid],
            "stable_order_key": _stable_hash(ALGORITHM_VERSION, "route", cid),
        })
    if len(output) != EXPECTED_CANDIDATES or set(Counter(x["lifecycle_route"] for x in output)) - set(ROUTES):
        raise RuntimeError("routing_not_closed")
    return output


def _render_dialogue(dialogue: Mapping[str, Any]) -> str:
    return "\n".join(
        f"[{row['turn_ref']}] {row['speaker']}: {row['text']}"
        for row in dialogue.get("turns") or []
    )


def _load_tokenizer() -> tuple[Any, dict[str, Any]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
    files = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        path = TOKENIZER_PATH / name
        if not path.is_file():
            raise RuntimeError(f"tokenizer_file_missing:{path}")
        files[name] = {"sha256": sha256_file(path), "size": path.stat().st_size}
    return tokenizer, {
        "target_api_model": "Qwen/Qwen3-14B",
        "counter_path": str(TOKENIZER_PATH.resolve()),
        "counter_role": "frozen_Qwen3_family_tokenizer_used_by_existing_expansion_runtime",
        "local_files_only": True,
        "files": files,
        "tokenizer_class": tokenizer.__class__.__name__,
    }


def _count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _length_bucket(values: Mapping[str, int]) -> tuple[dict[str, str], dict[str, int]]:
    ordered = sorted(values.values())
    low = ordered[(len(ordered) - 1) // 3]
    high = ordered[(2 * (len(ordered) - 1)) // 3]
    buckets = {
        key: "short" if value <= low else "medium" if value <= high else "long"
        for key, value in values.items()
    }
    return buckets, {"short_max": low, "medium_max": high}


def _build_root_pairs(
    routing: Sequence[Mapping[str, Any]], transitions: Mapping[str, Mapping[str, Any]],
    features: Mapping[str, Mapping[str, Any]], root_plan: Mapping[str, Mapping[str, Any]],
    length_buckets: Mapping[str, str], dialogue_tokens: Mapping[str, int],
) -> list[dict[str, Any]]:
    by_root: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in routing:
        by_root[str(row["root_id"])].append(row)
    pairs = []
    for root_id, rows in sorted(by_root.items()):
        seed_rows = [x for x in rows if x["lifecycle_route"] == "EXPANSION_FEEDBACK_ELIGIBLE"]
        probe_rows = [x for x in rows if x["lifecycle_route"] == "UNTOUCHED_PROBE_ELIGIBLE"]
        if len(seed_rows) != 1 or not probe_rows:
            continue
        seed = seed_rows[0]
        seed_gap = str(seed["gap_type"])
        probe = sorted(
            probe_rows,
            key=lambda x: (
                0 if x["gap_type"] == seed_gap else 1,
                _frontier_rank(transitions[str(x["candidate_id"])]),
            ),
        )[0]
        seed_transition = str(seed["primary_transition"])
        temporal = seed["question_type"] == "ordering" or seed_gap in {
            "temporal_precedence", "plan_vs_execution", "before_vs_after_state",
        }
        multi = seed["question_type"] == "multi_select" or probe["question_type"] == "multi_select"
        plan = root_plan[root_id]
        pairs.append({
            "root_id": root_id,
            "logical_shard": plan["logical_shard"],
            "logical_shard_index": plan["logical_shard_index"],
            "seed_candidate_id": seed["candidate_id"],
            "probe_candidate_id": probe["candidate_id"],
            "seed_gap_type": seed_gap,
            "probe_gap_type": probe["gap_type"],
            "same_gap_family": seed_gap == probe["gap_type"],
            "seed_question_type": seed["question_type"],
            "probe_question_type": probe["question_type"],
            "seed_primary_transition": seed_transition,
            "has_ordering_or_temporal": temporal,
            "has_multi_select": multi,
            "has_emerging_frontier": seed_transition in EMERGING_TRANSITIONS,
            "dialogue_token_count": dialogue_tokens[root_id],
            "length_bucket": length_buckets[root_id],
            "selection_rank": _frontier_rank(transitions[str(seed["candidate_id"])]),
            "stable_order_key": _stable_hash(ALGORITHM_VERSION, "root", root_id),
        })
    return pairs


def _select_formal(pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_shard: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_shard[str(pair["logical_shard"])].append(pair)
    shards = sorted(by_shard)
    choices = []
    for shard in shards:
        rows = sorted(by_shard[shard], key=lambda x: (x["selection_rank"], x["stable_order_key"]))
        # All seven logical shards have eligible roots. Taking one or two from
        # every shard exactly expresses full coverage and the per-shard cap.
        choices.append([(x,) for x in rows] + list(itertools.combinations(rows, 2)))

    # State keeps only the best stable-ranked selection for each coverage shape.
    states: dict[tuple[int, int, int, int, int], tuple[tuple[Any, ...], tuple[Mapping[str, Any], ...]]] = {
        (0, 0, 0, 0, 0): (tuple(), tuple())
    }
    length_bits = {"short": 1, "medium": 2, "long": 4}
    for shard_choices in choices:
        updated: dict[tuple[int, int, int, int, int], tuple[tuple[Any, ...], tuple[Mapping[str, Any], ...]]] = {}
        for state, (score, selected) in states.items():
            count, temporal, multi, emerging, length_mask = state
            for choice in shard_choices:
                new_count = count + len(choice)
                if new_count > FORMAL_ROOT_COUNT:
                    continue
                choice_length_mask = 0
                for item in choice:
                    choice_length_mask |= length_bits[str(item["length_bucket"])]
                new_state = (
                    new_count,
                    min(4, temporal + sum(bool(x["has_ordering_or_temporal"]) for x in choice)),
                    min(3, multi + sum(bool(x["has_multi_select"]) for x in choice)),
                    min(2, emerging + sum(bool(x["has_emerging_frontier"]) for x in choice)),
                    length_mask | choice_length_mask,
                )
                choice_score = tuple(
                    (x["selection_rank"], x["stable_order_key"]) for x in choice
                )
                new_score = score + choice_score
                old = updated.get(new_state)
                if old is None or new_score < old[0]:
                    updated[new_state] = (new_score, selected + tuple(choice))
        states = updated
    valid = [
        (state, value) for state, value in states.items()
        if state[0] == FORMAL_ROOT_COUNT and state[1] >= 4 and state[2] >= 3
        and state[3] >= 2 and state[4] == 7
    ]
    if not valid:
        raise RuntimeError("formal_root_constraints_infeasible")
    # Every represented shard contributes to score through a large leading bonus.
    ranked = []
    for state, (score, selected) in valid:
        shard_count = len({str(x["logical_shard"]) for x in selected})
        ranked.append((-shard_count, score, selected))
    selected = min(ranked, key=lambda x: (x[0], x[1]))[2]
    return [dict(x) for x in sorted(selected, key=lambda x: (x["logical_shard"], x["stable_order_key"]))]


def _select_reserve(
    pairs: Sequence[Mapping[str, Any]], formal: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    excluded = {str(x["root_id"]) for x in formal}
    candidates = sorted(
        (x for x in pairs if str(x["root_id"]) not in excluded),
        key=lambda x: (x["selection_rank"], x["stable_order_key"]),
    )
    selected: list[Mapping[str, Any]] = []
    shard_counts: Counter[str] = Counter()
    while len(selected) < RESERVE_ROOT_COUNT:
        available = [x for x in candidates if x not in selected and shard_counts[str(x["logical_shard"])] < 2]
        if not available:
            raise RuntimeError("reserve_root_constraints_infeasible")
        uncovered = {str(x["logical_shard"]) for x in available} - set(shard_counts)
        pool = [x for x in available if str(x["logical_shard"]) in uncovered] or available
        chosen = pool[0]
        selected.append(chosen)
        shard_counts[str(chosen["logical_shard"])] += 1
    return [dict(x) for x in selected]


def _feedback_card(root_pair: Mapping[str, Any], feature: Mapping[str, Any]) -> dict[str, Any]:
    anchors = [str(x) for x in feature.get("evidence_refs") or []]
    anchors = sorted(dict.fromkeys(anchors), key=lambda x: (0 if x.startswith("R01:") else 1, x))[:3]
    if not anchors:
        raise RuntimeError(f"feedback_card_has_no_anchor:{root_pair['root_id']}")
    gap = str(feature["gap_type"])
    global_priority = [
        "temporal_ordering" if gap in {"temporal_precedence", "plan_vs_execution"} else "explicit_state_transition",
        "cross_round_evidence_integration",
    ]
    return {
        "root_id": root_pair["root_id"],
        "feedback_version": ALGORITHM_VERSION,
        "global_priority": global_priority,
        "target": {
            "gap_type": gap,
            "frontier_status": root_pair["seed_primary_transition"].casefold(),
            "source_candidate_sha256": feature.get("candidate_pool_sha256") or feature["candidate_sha256"],
            "anchor_refs": anchors,
            "desired_event_operator": EVENT_OPERATORS[gap],
            "constraints": [
                "unique temporal anchor for every stage",
                "append-only continuation",
                "do not repeat the old question",
                "do not reveal Gold or answer labels",
                "show the final consequence in a separate observable turn",
            ],
        },
    }


def _audit_card(card: Mapping[str, Any]) -> list[str]:
    text = json.dumps(card, ensure_ascii=False)
    errors = []
    for key in ("question", "options", "items", "gold", "raw_output", "canonical_prediction", "rollout"):
        if re.search(rf'"{re.escape(key)}"\s*:', text, flags=re.IGNORECASE):
            errors.append(f"forbidden_key:{key}")
    if re.search(r"<box>|\bI[1-4](?:\s*,\s*I[1-4])+\b|\b[A-E](?:\s*,\s*[A-E])+\b", text):
        errors.append("answer_label_sequence_present")
    return errors


def _render_feedback_prompt(
    dialogue: Mapping[str, Any], memory_overview: Mapping[str, Any],
    card: Mapping[str, Any],
) -> str:
    return "\n\n".join((
        "SYSTEM / EXPANSION CONTRACT\n" + EXPANSION_SYSTEM_CONTRACT,
        "AUTHORITATIVE COMPLETE PARENT + R01\n" + _render_dialogue(dialogue),
        "R01 MEMORY RETRIEVAL (NON-AUTHORITATIVE TEMPLATE)\n" + json.dumps(
            memory_overview, ensure_ascii=False, sort_keys=True),
        "ABSTRACT FEEDBACK CARD\n" + json.dumps(card, ensure_ascii=False, sort_keys=True),
        "OUTPUT CONTRACT\nReturn only the existing bounded planner JSON schema for one R02 continuation.",
    ))


def _write_report_markdown(report: Mapping[str, Any]) -> str:
    counts = report["primary_transition_counts"]
    routes = report["routing_counts"]
    lines = [
        "# Round-1 Offline Feedback Compiler v1",
        "",
        f"- Status: `{report['status']}`",
        f"- Next status: `{report['next_status']}`",
        f"- Train candidates / roots: `{report['candidate_count']}` / `{report['root_count']}`",
        f"- Technical: `{counts.get('TECHNICAL', 0)}`",
        f"- Primary transitions closed: `{report['transition_closed']}`",
        "",
        "## Primary Transitions",
        "",
    ]
    lines.extend(f"- {name}: `{count}`" for name, count in counts.items())
    lines.extend(("", "## Lifecycle Routing", ""))
    lines.extend(f"- {name}: `{count}`" for name, count in routes.items())
    lines.extend((
        "",
        "## Pilot Selection",
        "",
        f"- Formal roots: `{report['formal_root_count']}`",
        f"- Reserve roots: `{report['reserve_root_count']}`",
        f"- Feedback seeds / untouched probes: `{report['feedback_seed_count']}` / `{report['untouched_probe_count']}`",
        f"- Formal shard coverage: `{report['formal_selection_coverage']['logical_shard_count']}`",
        f"- Card max tokens: `{report['token_budget_summary']['max_feedback_card_tokens']}`",
        f"- Expansion max input + reserve: `{report['token_budget_summary']['max_expansion_total_tokens']}` / `{ENDPOINT_CONTEXT_LIMIT}`",
        f"- API calls: `0`; Solver calls: `0`; Memorax writes: `0`",
        "",
    ))
    return "\n".join(lines)


def compile_feedback(run_root: Path = DEFAULT_RUN_ROOT) -> dict[str, Any]:
    run_root = run_root.resolve()
    if (
        run_root.exists() and any(run_root.iterdir())
        and not (run_root / "run_identity.json").is_file()
    ):
        raise RuntimeError(f"run_root_exists_without_identity:{run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    write_once_json(run_root / "run_identity.json", {
        "run_identity_version": f"{VERSION}_run_identity_v1",
        "run_root": str(run_root),
        "algorithm_version": ALGORITHM_VERSION,
        "offline_only": True,
        "historical_inputs_read_only": True,
    })

    source_paths = {
        "candidate_transition_matrix": TRAIN_REEVAL_RUN / "candidate_transition_matrix.jsonl",
        "candidate_training_exposure": TRAIN_REEVAL_RUN / "candidate_training_exposure_manifest.jsonl",
        "clean_candidate_results": TRAIN_REEVAL_RUN / "clean_train/candidate_results.jsonl",
        "inclusive_candidate_results": TRAIN_REEVAL_RUN / "inclusive_train/candidate_results.jsonl",
        "train_reevaluation_report": TRAIN_REEVAL_RUN / "final_train_reevaluation_report.json",
        "difficulty_manifest": M0_DIFFICULTY_RUN / "difficulty_manifest.jsonl",
        "root_split_manifest": M0_DIFFICULTY_RUN / "root_split_manifest.jsonl",
        "solver_contract": M0_DIFFICULTY_RUN / "solver_contract.json",
        "clean_training_manifest": TRAINING_RUN / "m_clean_policy_training_manifest.jsonl",
        "inclusive_training_manifest": TRAINING_RUN / "m_inclusive_policy_training_manifest.jsonl",
        "confirmed_invalid_blacklist": TRAINING_RUN / "confirmed_invalid_blacklist.json",
        "hard_evidence_status": EVIDENCE_RUN / "hard_evidence_status_manifest.jsonl",
        "hard_evidence_summary": EVIDENCE_RUN / "hard_evidence_summary.json",
        "dev_model_summary": DEV_RUN / "dev_model_summary.json",
        "dev_by_question_type": DEV_RUN / "dev_by_question_type.json",
        "round1_root_manifest": ROUND1_PLAN_RUN / "round1_root_manifest.jsonl",
        "round1_logical_shards": ROUND1_PLAN_RUN / "logical_shards.json",
        "round1_contract": ROUND1_PLAN_RUN / "round1_contract.json",
    }
    source_sha = {name: _verify_source(path) for name, path in source_paths.items()}

    matrix = load_jsonl(source_paths["candidate_transition_matrix"])
    matrix_by_id = _index(matrix, "candidate_id")
    train_difficulty = [
        row for row in load_jsonl(source_paths["difficulty_manifest"])
        if row.get("split") == "train"
    ]
    difficulty_by_id = _index(train_difficulty, "candidate_id")
    exposure_by_id = _index(load_jsonl(source_paths["candidate_training_exposure"]), "candidate_id")
    clean_results = _index(load_jsonl(source_paths["clean_candidate_results"]), "candidate_id")
    inclusive_results = _index(load_jsonl(source_paths["inclusive_candidate_results"]), "candidate_id")
    candidate_sets = [set(x) for x in (matrix_by_id, difficulty_by_id, exposure_by_id, clean_results, inclusive_results)]
    if any(candidate_set != candidate_sets[0] for candidate_set in candidate_sets[1:]):
        raise RuntimeError("train_candidate_identity_not_aligned")
    if len(matrix_by_id) != EXPECTED_CANDIDATES:
        raise RuntimeError(f"train_candidate_count_mismatch:{len(matrix_by_id)}")
    roots = {str(row["root_id"]) for row in matrix}
    if len(roots) != EXPECTED_ROOTS or any(row.get("split") != "train" for row in matrix):
        raise RuntimeError("train_root_or_split_mismatch")

    evidence_rows = load_jsonl(source_paths["hard_evidence_status"])
    evidence_by_id = {
        str(row["candidate_id"]): row for row in evidence_rows if row.get("split") == "train"
    }
    expected_hard = {
        cid for cid, row in difficulty_by_id.items()
        if row["difficulty_bucket"] == "hard_candidate"
    }
    if set(evidence_by_id) != expected_hard or len(evidence_by_id) != 544:
        raise RuntimeError("train_hard_evidence_identity_mismatch")

    blacklist = load_json(source_paths["confirmed_invalid_blacklist"])
    confirmed_invalid = {
        str(row["candidate_id"]) for row in blacklist.get("entries") or []
    }
    if not confirmed_invalid <= set(matrix_by_id):
        raise RuntimeError("blacklist_contains_non_train_candidate")

    root_plan_rows = load_jsonl(source_paths["round1_root_manifest"])
    root_plan = _index(root_plan_rows, "root_id")
    if set(root_plan) != roots or len(root_plan) != EXPECTED_ROOTS:
        raise RuntimeError("round1_plan_root_identity_mismatch")
    for root_id, row in root_plan.items():
        dialogue_path = Path(str(row["canonical_dialogue_path"]))
        if not dialogue_path.is_file() or sha256_file(dialogue_path) != row["canonical_dialogue_sha256"]:
            raise RuntimeError(f"r01_dialogue_sha_drift:{root_id}")
        if "r02" in dialogue_path.as_posix().casefold():
            raise RuntimeError(f"r02_path_forbidden:{root_id}")
        for field in ("memory_intent", "memory_completion"):
            path = Path(str(row[f"{field}_path"]))
            if not path.is_file() or sha256_file(path) != row[f"{field}_sha256"]:
                raise RuntimeError(f"r01_{field}_sha_drift:{root_id}")

    api_config = load_json(API_CONFIG_PATH)
    api_model = str(api_config.get("api", {}).get("model") or "")
    if api_model != "Qwen/Qwen3-14B":
        raise RuntimeError(f"api_model_contract_mismatch:{api_model}")
    tokenizer, tokenizer_identity = _load_tokenizer()

    dialogue_cache: dict[str, dict[str, Any]] = {}
    dialogue_tokens: dict[str, int] = {}
    for root_id, plan in root_plan.items():
        dialogue = load_json(Path(plan["canonical_dialogue_path"]))
        refs = [str(x.get("turn_ref") or "") for x in dialogue.get("turns") or []]
        if dialogue.get("root_id") != root_id or any(x.startswith("R02:") for x in refs):
            raise RuntimeError(f"dialogue_scope_not_parent_plus_r01:{root_id}")
        if not any(x.startswith("R01:") for x in refs):
            raise RuntimeError(f"r01_missing:{root_id}")
        dialogue_cache[root_id] = dialogue
        dialogue_tokens[root_id] = _count(tokenizer, _render_dialogue(dialogue))
    length_buckets, length_thresholds = _length_bucket(dialogue_tokens)

    # Validate that Clean/Inclusive used precisely paired requests before deriving features.
    for cid in sorted(matrix_by_id):
        left = clean_results[cid]
        right = inclusive_results[cid]
        left_keys = {
            (x["request_id"], x["context_mode"], x["variant_id"], x["slot_id"], x["seed"])
            for x in left.get("requests") or []
        }
        right_keys = {
            (x["request_id"], x["context_mode"], x["variant_id"], x["slot_id"], x["seed"])
            for x in right.get("requests") or []
        }
        if left_keys != right_keys or len(left_keys) != int(left["request_count"]):
            raise RuntimeError(f"trained_model_request_alignment_mismatch:{cid}")

    transition_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for cid in sorted(matrix_by_id):
        old = matrix_by_id[cid]
        source = difficulty_by_id[cid]
        if source["root_id"] != old["root_id"] or source["question_type"] != old["question_type"]:
            raise RuntimeError(f"candidate_metadata_alignment_mismatch:{cid}")
        if exposure_by_id[cid]["training_exposure"] != old["training_exposure"]:
            raise RuntimeError(f"exposure_alignment_mismatch:{cid}")
        # Recompute model behavior from the raw offline reevaluation rows. The
        # previous matrix is used only for lineage/identity, never as the
        # authority for a primary transition label.
        m0_source = {
            key: source.get(key)
            for key in (
                "difficulty_bucket", "full_context_accuracy",
                "full_context_correct_count", "full_context_total_count",
                "no_context_accuracy", "no_context_correct_count",
                "no_context_total_count", "position_mechanism",
                "canonical_flip_rate",
            )
        }
        m0_source["context_gain"] = float(source["full_context_accuracy"]) - float(source["no_context_accuracy"])
        m0_source.update({
            "technical_failure_count": 0,
            "parse_failure_count": 0,
            "truncated_request_count": 0,
        })
        clean_model = _reevaluated_model_summary(clean_results[cid])
        inclusive_model = _reevaluated_model_summary(inclusive_results[cid])
        old_m0 = old["models"]["M0"]
        for key in ("difficulty_bucket", "full_context_accuracy", "no_context_accuracy", "position_mechanism"):
            if old_m0.get(key) != m0_source.get(key):
                raise RuntimeError(f"m0_source_metric_drift:{cid}:{key}")
        for model_name, derived in (("M_clean_policy", clean_model), ("M_inclusive_policy", inclusive_model)):
            old_model = old["models"][model_name]
            for key in ("difficulty_bucket", "full_context_accuracy", "no_context_accuracy", "position_mechanism"):
                if old_model.get(key) != derived.get(key):
                    raise RuntimeError(f"reevaluation_metric_drift:{cid}:{model_name}:{key}")
        model_payload = {
            "M0": {**old_m0, **m0_source},
            "M_clean_policy": {**old["models"]["M_clean_policy"], **clean_model},
            "M_inclusive_policy": {**old["models"]["M_inclusive_policy"], **inclusive_model},
        }
        recomputed_row = {**old, "models": model_payload}
        primary = recompute_primary(recomputed_row)
        flags = transition_flags({**recomputed_row, "training_exposure": old["training_exposure"]})
        evidence_status = evidence_by_id.get(cid, {}).get("evidence_status")
        if source["difficulty_bucket"] != "hard_candidate" and evidence_status is not None:
            raise RuntimeError(f"non_hard_evidence_status_present:{cid}")
        transition = {
            "primary_transition_version": f"{VERSION}_primary_transition_v1",
            "candidate_id": cid,
            "root_id": old["root_id"],
            "split": "train",
            "question_type": old["question_type"],
            "generation_source": old["generation_source"],
            "position_mechanism": old["position_mechanism"],
            "evidence_status": evidence_status,
            "training_exposure": old["training_exposure"],
            "primary_transition": primary,
            "previous_primary_transition": old.get("primary_transition"),
            "models": model_payload,
            "flags": flags,
        }
        feature = _candidate_feedback_feature(
            {**source, "training_exposure": old["training_exposure"], "evidence_status": evidence_status},
            clean_results[cid], inclusive_results[cid], primary, flags,
        )
        transition["gap_type"] = feature["gap_type"]
        transition_rows.append(transition)
        feature_rows.append(feature)

    primary_counts = Counter(row["primary_transition"] for row in transition_rows)
    technical_count = primary_counts.get("TECHNICAL", 0)
    if sum(primary_counts.values()) != EXPECTED_CANDIDATES or technical_count != EXPECTED_TECHNICAL:
        raise RuntimeError(f"primary_transition_not_closed:{dict(primary_counts)}")
    unexpected_primary = set(primary_counts) - set(PRIMARY_ORDER) - {"TECHNICAL"}
    if unexpected_primary:
        raise RuntimeError(f"unexpected_primary_transition:{sorted(unexpected_primary)}")

    transition_by_id = _index(transition_rows, "candidate_id")
    feature_by_id = _index(feature_rows, "candidate_id")
    routing = assign_routes(transition_rows, confirmed_invalid)
    routing_counts = Counter(row["lifecycle_route"] for row in routing)

    root_pairs = _build_root_pairs(
        routing, transition_by_id, feature_by_id, root_plan,
        length_buckets, dialogue_tokens,
    )
    formal = _select_formal(root_pairs)
    reserve = _select_reserve(root_pairs, formal)
    if len(formal) != FORMAL_ROOT_COUNT or len(reserve) != RESERVE_ROOT_COUNT:
        raise RuntimeError("formal_or_reserve_count_mismatch")
    if {x["root_id"] for x in formal} & {x["root_id"] for x in reserve}:
        raise RuntimeError("formal_reserve_overlap")

    seed_rows = []
    probe_rows = []
    cards = []
    for pair in formal:
        seed_id, probe_id = pair["seed_candidate_id"], pair["probe_candidate_id"]
        seed_feature = feature_by_id[seed_id]
        seed_route = next(x for x in routing if x["candidate_id"] == seed_id)
        probe_route = next(x for x in routing if x["candidate_id"] == probe_id)
        if seed_route["lifecycle_route"] != "EXPANSION_FEEDBACK_ELIGIBLE" or probe_route["lifecycle_route"] != "UNTOUCHED_PROBE_ELIGIBLE":
            raise RuntimeError(f"seed_probe_route_mismatch:{pair['root_id']}")
        if seed_id == probe_id:
            raise RuntimeError(f"seed_probe_identity_collision:{pair['root_id']}")
        dialogue_refs = {
            str(item.get("turn_ref") or "")
            for item in dialogue_cache[str(pair["root_id"])].get("turns") or []
        }
        missing_anchors = sorted(set(seed_feature.get("evidence_refs") or []) - dialogue_refs)
        if missing_anchors:
            raise RuntimeError(f"feedback_anchor_not_in_parent_plus_r01:{pair['root_id']}:{missing_anchors}")
        seed_rows.append({
            "feedback_seed_version": f"{VERSION}_seed_v1",
            **pair,
            "candidate_id": seed_id,
            "candidate_sha256": seed_feature["candidate_sha256"],
            "evidence_status": transition_by_id[seed_id].get("evidence_status"),
            "evidence_confidence": "reduced" if transition_by_id[seed_id].get("evidence_status") == "EVIDENCE_RISK" else "normal",
        })
        probe_rows.append({
            "untouched_probe_version": f"{VERSION}_probe_v1",
            "root_id": pair["root_id"],
            "candidate_id": probe_id,
            "candidate_sha256": feature_by_id[probe_id]["candidate_sha256"],
            "question_type": pair["probe_question_type"],
            "gap_type": pair["probe_gap_type"],
            "primary_transition": transition_by_id[probe_id]["primary_transition"],
            "same_gap_family_as_seed": pair["same_gap_family"],
            "feedback_exposure_forbidden": True,
        })
        card = _feedback_card(pair, seed_feature)
        errors = _audit_card(card)
        if errors:
            raise RuntimeError(f"feedback_card_forbidden_content:{pair['root_id']}:{errors}")
        cards.append(card)

    budget_rows = []
    memory_forks = []
    for pair, card in zip(formal, cards):
        root_id = str(pair["root_id"])
        plan = root_plan[root_id]
        memory_root = Path(plan["memory_physical_root"])
        memory_path = memory_root / "memory/post_generation_overview.json"
        parent_intent = memory_root / "memory/parent_add_intent.json"
        if not memory_path.is_file() or not parent_intent.is_file():
            raise RuntimeError(f"r01_memory_fork_source_missing:{root_id}")
        memory_overview = load_json(memory_path)
        if memory_overview.get("root_id") != root_id:
            raise RuntimeError(f"r01_memory_overview_root_mismatch:{root_id}")
        prompt = _render_feedback_prompt(dialogue_cache[root_id], memory_overview, card)
        prompt_tokens = _count(tokenizer, prompt)
        card_tokens = _count(tokenizer, json.dumps(card, ensure_ascii=False, sort_keys=True))
        dialogue_count = dialogue_tokens[root_id]
        future_solver_total = (
            dialogue_count + FUTURE_R02_DIALOGUE_RESERVE
            + FUTURE_SOLVER_FIXED_OVERHEAD + FUTURE_SOLVER_OUTPUT_RESERVE
        )
        future_sft_total = dialogue_count + FUTURE_R02_DIALOGUE_RESERVE + FUTURE_SOLVER_FIXED_OVERHEAD
        row = {
            "root_id": root_id,
            "tokenizer": tokenizer_identity,
            "parent_plus_r01_dialogue_tokens": dialogue_count,
            "feedback_card_tokens": card_tokens,
            "expansion_prompt_tokens": prompt_tokens,
            "expansion_output_reserve_tokens": EXPANSION_OUTPUT_RESERVE,
            "expansion_total_tokens": prompt_tokens + EXPANSION_OUTPUT_RESERVE,
            "endpoint_context_limit": ENDPOINT_CONTEXT_LIMIT,
            "parent_plus_r01_truncated": False,
            "card_within_1200": card_tokens <= 1_200,
            "expansion_within_endpoint": prompt_tokens + EXPANSION_OUTPUT_RESERVE <= ENDPOINT_CONTEXT_LIMIT,
            "future_r02_dialogue_reserve_tokens": FUTURE_R02_DIALOGUE_RESERVE,
            "future_solver_total_tokens": future_solver_total,
            "future_solver_within_16384": future_solver_total <= FUTURE_SOLVER_LIMIT,
            "future_sft_total_tokens": future_sft_total,
            "future_sft_over_8192_warning": future_sft_total > FUTURE_SFT_LIMIT,
            "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        }
        if not row["card_within_1200"] or not row["expansion_within_endpoint"] or not row["future_solver_within_16384"]:
            raise RuntimeError(f"token_budget_blocker:{root_id}:{row}")
        budget_rows.append(row)

        fork_suffix = _stable_hash(ALGORITHM_VERSION, "memory_fork", root_id)[:12]
        memory_forks.append({
            "root_id": root_id,
            "fork_plan_version": f"{VERSION}_r01_memory_fork_v1",
            "source_r01_user_id": plan["prior_user_id"],
            "source_r01_session_id": plan["prior_session_id"],
            "source_namespace_reuse_forbidden": True,
            "reason": "source_namespace_already_contains_uncontrolled_R02",
            "r01_only_reconstruction": {
                "parent_add_intent_path": str(parent_intent.resolve()),
                "parent_add_intent_sha256": sha256_file(parent_intent),
                "r01_add_intent_path": str(Path(plan["memory_intent_path"]).resolve()),
                "r01_add_intent_sha256": plan["memory_intent_sha256"],
                "r01_add_completion_path": str(Path(plan["memory_completion_path"]).resolve()),
                "r01_add_completion_sha256": plan["memory_completion_sha256"],
                "r01_post_generation_overview_path": str(memory_path.resolve()),
                "r01_post_generation_overview_sha256": sha256_file(memory_path),
            },
            "branches": {
                "control": {
                    "namespace": f"gxy-selfplay:round1-paired-control:{fork_suffix}",
                    "r02_session_id": f"{root_id}:memorax:R02-control",
                    "feedback_card_exposed": False,
                },
                "feedback": {
                    "namespace": f"gxy-selfplay:round1-paired-feedback:{fork_suffix}",
                    "r02_session_id": f"{root_id}:memorax:R02-feedback",
                    "feedback_card_exposed": True,
                },
            },
            "post_bootstrap_write_scope": "R02_delta_only",
            "memorax_writes_performed": 0,
        })

    dev_summary = load_json(source_paths["dev_model_summary"])
    dev_qtype = load_json(source_paths["dev_by_question_type"])
    dev_global = {
        "dev_global_feedback_version": f"{VERSION}_dev_global_v1",
        "source_scope": "aggregate_only",
        "source_files": {
            "dev_model_summary.json": source_sha["dev_model_summary"],
            "dev_by_question_type.json": source_sha["dev_by_question_type"],
        },
        "dev_candidate_count_verified": all(
            int(x["candidate_count"]) == 485 for x in dev_summary["models"].values()
        ),
        "root_specific_dev_content_used": False,
        "weights": {"increase": 1.25, "maintain": 1.0, "downweight": 0.5},
        "increase": [
            "temporal_ordering", "plan_vs_execution",
            "explicit_state_transition", "cross_round_evidence_integration",
        ],
        "maintain": ["multi_select_aggregation"],
        "downweight": ["simple_local_single_choice", "pure_position_only_difficulty"],
        "aggregate_question_type_metrics": {
            model: {
                qtype: value["candidate_weighted"]
                for qtype, value in dev_qtype[model].items()
            }
            for model in MODEL_NAMES
        },
    }

    formal_coverage = {
        "ordering_or_temporal_root_count": sum(x["has_ordering_or_temporal"] for x in formal),
        "multi_select_root_count": sum(x["has_multi_select"] for x in formal),
        "emerging_frontier_root_count": sum(x["has_emerging_frontier"] for x in formal),
        "logical_shard_count": len({x["logical_shard"] for x in formal}),
        "logical_shard_counts": dict(sorted(Counter(x["logical_shard"] for x in formal).items())),
        "length_bucket_counts": dict(sorted(Counter(x["length_bucket"] for x in formal).items())),
        "question_type_counts": dict(sorted(Counter(x["seed_question_type"] for x in formal).items())),
        "gap_type_counts": dict(sorted(Counter(x["seed_gap_type"] for x in formal).items())),
        "same_gap_seed_probe_count": sum(x["same_gap_family"] for x in formal),
    }
    selection_checks = {
        "formal_root_count_10": len(formal) == 10,
        "reserve_root_count_5": len(reserve) == 5,
        "ordering_or_temporal_at_least_4": formal_coverage["ordering_or_temporal_root_count"] >= 4,
        "multi_select_at_least_3": formal_coverage["multi_select_root_count"] >= 3,
        "emerging_at_least_2": formal_coverage["emerging_frontier_root_count"] >= 2,
        "each_shard_at_most_2": max(formal_coverage["logical_shard_counts"].values()) <= 2,
        "all_7_shards_covered": formal_coverage["logical_shard_count"] == 7,
        "short_medium_long_covered": set(formal_coverage["length_bucket_counts"]) == {"short", "medium", "long"},
        "distinct_seed_probe_per_root": all(x["seed_candidate_id"] != x["probe_candidate_id"] for x in formal),
    }
    if not all(selection_checks.values()):
        raise RuntimeError(f"root_selection_constraints_failed:{selection_checks}")

    token_summary = {
        "target_api_model": api_model,
        "tokenizer_identity": tokenizer_identity,
        "endpoint_context_limit": ENDPOINT_CONTEXT_LIMIT,
        "expansion_output_reserve": EXPANSION_OUTPUT_RESERVE,
        "max_feedback_card_tokens": max(x["feedback_card_tokens"] for x in budget_rows),
        "max_expansion_prompt_tokens": max(x["expansion_prompt_tokens"] for x in budget_rows),
        "max_expansion_total_tokens": max(x["expansion_total_tokens"] for x in budget_rows),
        "max_future_solver_total_tokens": max(x["future_solver_total_tokens"] for x in budget_rows),
        "future_sft_warning_count": sum(x["future_sft_over_8192_warning"] for x in budget_rows),
        "all_cards_within_1200": all(x["card_within_1200"] for x in budget_rows),
        "all_expansion_requests_fit": all(x["expansion_within_endpoint"] for x in budget_rows),
        "all_future_solver_requests_fit": all(x["future_solver_within_16384"] for x in budget_rows),
        "parent_plus_r01_truncated_count": sum(x["parent_plus_r01_truncated"] for x in budget_rows),
        "per_root": budget_rows,
    }

    preflight_checks = {
        "train_candidate_count_2294": len(matrix_by_id) == EXPECTED_CANDIDATES,
        "candidate_ids_unique": len(matrix_by_id) == len(matrix),
        "train_root_count_70": len(roots) == EXPECTED_ROOTS,
        "candidate_sets_aligned": all(x == candidate_sets[0] for x in candidate_sets),
        "technical_zero": technical_count == 0,
        "primary_transitions_closed": sum(primary_counts.values()) == EXPECTED_CANDIDATES,
        "hard_evidence_scope_exact": set(evidence_by_id) == expected_hard,
        "round1_plan_roots_exact": set(root_plan) == roots,
        "parent_plus_r01_only": True,
        "dev_aggregate_only": True,
        "api_model_qwen3_14b": api_model == "Qwen/Qwen3-14B",
        "local_tokenizer_loaded": True,
        "source_sha_verified": True,
        "test_access_zero": True,
        "r02_content_access_zero": True,
    }
    preflight = {
        "preflight_version": f"{VERSION}_preflight_v1",
        "status": "PASS" if all(preflight_checks.values()) else "BLOCKED",
        "run_root": str(run_root),
        "source_runs_read_only": True,
        "source_paths": {name: str(path.resolve()) for name, path in source_paths.items()},
        "source_sha256": source_sha,
        "checks": preflight_checks,
        "failed_checks": [name for name, passed in preflight_checks.items() if not passed],
        "candidate_count": len(matrix_by_id),
        "root_count": len(roots),
        "api_calls": 0,
        "solver_calls": 0,
        "qa_writer_calls": 0,
        "memorax_calls": 0,
        "semantic_auditor_calls": 0,
    }

    eligible_root_rows = []
    selected_ids = {x["root_id"] for x in formal}
    reserve_ids = {x["root_id"] for x in reserve}
    for pair in root_pairs:
        selected_role = "formal" if pair["root_id"] in selected_ids else "reserve" if pair["root_id"] in reserve_ids else None
        eligible_root_rows.append({
            "eligible_root_version": f"{VERSION}_eligible_root_v1",
            **pair,
            "selected_role": selected_role,
        })

    root_selection_report = {
        "root_selection_version": f"{VERSION}_selection_v1",
        "algorithm": "eligibility_gate_stratified_coverage_stable_hash_dp_v1",
        "eligible_root_count": len(root_pairs),
        "formal_root_count": len(formal),
        "reserve_root_count": len(reserve),
        "constraints": {
            "ordering_or_temporal_min": 4,
            "multi_select_min": 3,
            "emerging_frontier_min": 2,
            "logical_shard_max_per_shard": 2,
            "logical_shards_target": 7,
            "length_buckets_required": ["short", "medium", "long"],
        },
        "checks": selection_checks,
        "formal_coverage": formal_coverage,
        "length_bucket_thresholds": length_thresholds,
        "selection_did_not_read_r02": True,
    }

    routing_summary = {
        "routing_summary_version": f"{VERSION}_routing_summary_v1",
        "candidate_count": len(routing),
        "route_counts": dict(sorted(routing_counts.items())),
        "route_by_question_type": {
            qtype: dict(sorted(Counter(x["lifecycle_route"] for x in routing if x["question_type"] == qtype).items()))
            for qtype in ("single_choice", "multi_select", "ordering")
        },
        "route_by_primary_transition": {
            primary: dict(sorted(Counter(x["lifecycle_route"] for x in routing if x["primary_transition"] == primary).items()))
            for primary in sorted(primary_counts)
        },
    }

    final = {
        "final_report_version": f"{VERSION}_final_v1",
        "status": "ROUND1_FEEDBACK_COMPILER_COMPLETE",
        "next_status": "READY_FOR_PAIRED_EXPANSION_SMOKE",
        "candidate_count": len(transition_rows),
        "root_count": len(roots),
        "transition_closed": sum(primary_counts.values()) == EXPECTED_CANDIDATES,
        "primary_transition_counts": {
            name: primary_counts.get(name, 0) for name in PRIMARY_ORDER
        },
        "technical_count": technical_count,
        "m0_full_zero_transition_counts": {
            name: sum(
                row["primary_transition"] == name
                and float(row["models"]["M0"]["full_context_accuracy"]) == 0.0
                for row in transition_rows
            )
            for name in PRIMARY_ORDER
        },
        "m0_full_zero_closed_count": sum(
            float(row["models"]["M0"]["full_context_accuracy"]) == 0.0
            for row in transition_rows
        ),
        "routing_counts": dict(sorted(routing_counts.items())),
        "formal_root_count": len(formal),
        "reserve_root_count": len(reserve),
        "formal_root_ids": [x["root_id"] for x in formal],
        "reserve_root_ids": [x["root_id"] for x in reserve],
        "feedback_seed_count": len(seed_rows),
        "untouched_probe_count": len(probe_rows),
        "feedback_seed_distribution": {
            "question_type": dict(sorted(Counter(x["seed_question_type"] for x in seed_rows).items())),
            "gap_type": dict(sorted(Counter(x["seed_gap_type"] for x in seed_rows).items())),
            "primary_transition": dict(sorted(Counter(x["seed_primary_transition"] for x in seed_rows).items())),
            "evidence_status": dict(sorted(Counter(str(x["evidence_status"] or "NOT_CLASSIFIED_NON_HARD") for x in seed_rows).items())),
        },
        "untouched_probe_distribution": {
            "question_type": dict(sorted(Counter(x["question_type"] for x in probe_rows).items())),
            "gap_type": dict(sorted(Counter(x["gap_type"] for x in probe_rows).items())),
            "same_gap_family_count": sum(x["same_gap_family_as_seed"] for x in probe_rows),
        },
        "formal_selection_coverage": formal_coverage,
        "token_budget_summary": {key: value for key, value in token_summary.items() if key != "per_root"},
        "memory_fork_plan_count": len(memory_forks),
        "memorax_writes": 0,
        "api_calls": 0,
        "solver_calls": 0,
        "qa_writer_calls": 0,
        "semantic_auditor_calls": 0,
        "test_access": 0,
        "r02_content_access": 0,
    }

    write_once_json(run_root / "preflight_report.json", preflight)
    write_once_json(run_root / "dev_global_feedback.json", dev_global)
    write_once_jsonl(run_root / "candidate_feedback_features.jsonl", feature_rows)
    write_once_jsonl(run_root / "candidate_primary_transition.jsonl", transition_rows)
    write_once_jsonl(run_root / "feedback_routing_manifest.jsonl", routing)
    write_once_json(run_root / "feedback_routing_summary.json", routing_summary)
    write_once_jsonl(run_root / "feedback_seed_manifest.jsonl", seed_rows)
    write_once_jsonl(run_root / "untouched_probe_manifest.jsonl", probe_rows)
    write_once_jsonl(run_root / "eligible_root_pool.jsonl", eligible_root_rows)
    write_once_jsonl(run_root / "selected_10_roots.jsonl", formal)
    write_once_jsonl(run_root / "reserve_5_roots.jsonl", reserve)
    write_once_json(run_root / "root_selection_report.json", root_selection_report)
    write_once_jsonl(run_root / "root_feedback_cards.jsonl", cards)
    write_once_json(run_root / "feedback_prompt_token_budget.json", token_summary)
    write_once_json(run_root / "r01_only_memory_fork_plan.json", {
        "fork_plan_version": f"{VERSION}_memory_forks_v1",
        "source_state": "frozen_R01_only",
        "existing_uncontrolled_r02_read": False,
        "memorax_writes_performed": 0,
        "roots": memory_forks,
    })
    write_once_json(run_root / "final_feedback_compiler_report.json", final)
    write_once_text(
        run_root / "final_feedback_compiler_report.md",
        _write_report_markdown(final),
    )
    write_once_text(
        run_root / "final_feedback_compiler_report.md.sha256",
        sha256_file(run_root / "final_feedback_compiler_report.md") + "\n",
    )
    return final
