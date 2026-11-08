from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any


CANDIDATE_TYPE_ORDER = (
    "cross_session_chain",
    "cross_session_bridge",
    "conflict_chain",
    "relation_chain",
    "update_chain",
    "state_update",
    "temporal_pair",
)

EVOLUTION_TYPE_ORDER = (
    "evidence_bridge",
    "multi_hop_attribute",
    "conflict_resolution",
    "final_state",
    "person_relation_temporal",
    "temporal_ordering",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


def selected_quality(stats: dict[str, Any]) -> float:
    selected_rate = as_float(stats.get("selected_rate"))
    critic_rate = as_float(stats.get("critic_safe_rate"))
    option_rate = as_float(stats.get("option_safe_rate"))
    rule_rate = as_float(stats.get("rule_safe_rate"), 1.0)
    return clamp(0.20 * rule_rate + 0.30 * critic_rate + 0.30 * option_rate + 0.20 * selected_rate, 0.0, 1.0)


def frontier_score(acc: float, *, target_low: float, target_high: float) -> float:
    if acc < target_low:
        # Very low solver accuracy is useful only if quality is strong; keep it below true frontier.
        return clamp(0.50 + 0.50 * (acc / max(target_low, 1e-6)), 0.0, 1.0)
    if acc <= target_high:
        return 1.0
    if acc >= 1.0:
        return 0.0
    return clamp((1.0 - acc) / max(1.0 - target_high, 1e-6), 0.0, 1.0)


def count_bonus(count: int, *, target: int) -> float:
    if count <= 0:
        return 0.35
    if count < target:
        return 1.0 + 0.18 * (1.0 - count / max(target, 1))
    return 1.0


def failure_counts(failures: list[Any], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in failures:
        if not isinstance(item, dict):
            continue
        value = item.get(key)
        if not value:
            continue
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def normalize_weights(raw: dict[str, float], *, min_weight: float = 0.0, max_weight: float = 0.45) -> dict[str, float]:
    # Keep reward magnitudes intact before normalization; max_weight caps only final policy mass.
    clipped = {key: max(0.0, value) for key, value in raw.items()}
    total = sum(clipped.values())
    if total <= 0:
        return {key: round(1.0 / len(clipped), 4) for key in clipped}
    normalized = {key: value / total for key, value in clipped.items()}

    # Re-clip after normalization so one mode cannot dominate the next challenger round.
    over_cap = {key: value for key, value in normalized.items() if value > max_weight}
    if over_cap and len(normalized) > len(over_cap):
        capped_total = sum(max_weight for _ in over_cap)
        rest_total = sum(value for key, value in normalized.items() if key not in over_cap)
        remaining = max(0.0, 1.0 - capped_total)
        normalized = {
            key: (max_weight if key in over_cap else (value / rest_total * remaining if rest_total > 0 else 0.0))
            for key, value in normalized.items()
        }
    return {key: round(value, 4) for key, value in normalized.items()}


def build_weight_arg(weights: dict[str, float]) -> str:
    return ",".join(f"{key}={weights[key]:.4f}".rstrip("0").rstrip(".") for key in CANDIDATE_TYPE_ORDER if key in weights)


def score_candidate_types(
    multiqa_report: dict[str, Any],
    solver_comparison: dict[str, Any],
    *,
    target_low: float,
    target_high: float,
    min_count_target: int,
    temporal_floor: float,
) -> dict[str, dict[str, Any]]:
    operator_report = multiqa_report.get("operator_report_by_candidate_type", {})
    if not isinstance(operator_report, dict):
        operator_report = {}
    comparison = solver_comparison.get("by_candidate_type", {})
    if not isinstance(comparison, dict):
        comparison = {}
    failures = solver_comparison.get("lora_failures", [])
    if not isinstance(failures, list):
        failures = []
    failure_by_type = failure_counts(failures, "candidate_type")

    scores: dict[str, dict[str, Any]] = {}
    for candidate_type in CANDIDATE_TYPE_ORDER:
        stats = operator_report.get(candidate_type, {})
        if not isinstance(stats, dict):
            stats = {}
        comp = comparison.get(candidate_type, {})
        if not isinstance(comp, dict):
            comp = {}

        generated_count = as_int(stats.get("generated_count"))
        selected_count = as_int(stats.get("selected_count"), as_int(comp.get("count")))
        lora_acc = as_float(comp.get("lora_accuracy"), as_float(stats.get("average_full_context_accuracy"), 1.0))
        base_acc = as_float(comp.get("base_accuracy"), as_float(stats.get("average_full_context_accuracy")))
        quality = selected_quality(stats) if stats else 0.0
        frontier = frontier_score(lora_acc, target_low=target_low, target_high=target_high)
        hard_pressure = clamp(1.0 - lora_acc, 0.0, 1.0)
        fail_bonus = min(0.30, 0.10 * failure_by_type.get(candidate_type, 0))
        exploration = count_bonus(selected_count, target=min_count_target)
        production = clamp(math.sqrt(max(selected_count, 0) + 1.0) / math.sqrt(max(min_count_target, 1) + 1.0), 0.35, 1.0)

        reward = quality * (0.45 * frontier + 0.35 * hard_pressure + 0.20 * production) * exploration + fail_bonus
        if generated_count <= 0:
            reward *= 0.25
        if candidate_type == "temporal_pair":
            reward = min(reward, temporal_floor)
        if lora_acc >= 0.90 and failure_by_type.get(candidate_type, 0) == 0:
            reward *= 0.35
        if quality < 0.45 and failure_by_type.get(candidate_type, 0) == 0:
            reward *= 0.60

        scores[candidate_type] = {
            "reward": round(reward, 6),
            "quality": round(quality, 6),
            "frontier_score": round(frontier, 6),
            "hard_pressure": round(hard_pressure, 6),
            "failure_bonus": round(fail_bonus, 6),
            "generated_count": generated_count,
            "selected_count": selected_count,
            "base_accuracy": round(base_acc, 6),
            "lora_accuracy": round(lora_acc, 6),
            "full_context_accuracy": round(as_float(stats.get("average_full_context_accuracy")), 6),
            "evidence_window_accuracy": round(as_float(stats.get("average_evidence_window_accuracy")), 6),
            "critic_safe_rate": round(as_float(stats.get("critic_safe_rate")), 6),
            "option_safe_rate": round(as_float(stats.get("option_safe_rate")), 6),
            "lora_failure_count": failure_by_type.get(candidate_type, 0),
        }
    return scores


def score_evolution_types(
    multiqa_report: dict[str, Any],
    solver_comparison: dict[str, Any],
    *,
    target_low: float,
    target_high: float,
    min_count_target: int,
) -> dict[str, dict[str, Any]]:
    operator_report = multiqa_report.get("operator_report_by_evolution_type", {})
    if not isinstance(operator_report, dict):
        operator_report = {}
    comparison = solver_comparison.get("by_evolution_type", {})
    if not isinstance(comparison, dict):
        comparison = {}
    failures = solver_comparison.get("lora_failures", [])
    if not isinstance(failures, list):
        failures = []
    failure_by_type = failure_counts(failures, "evolution_type")

    scores: dict[str, dict[str, Any]] = {}
    for evolution_type in EVOLUTION_TYPE_ORDER:
        stats = operator_report.get(evolution_type, {})
        if not isinstance(stats, dict):
            stats = {}
        comp = comparison.get(evolution_type, {})
        if not isinstance(comp, dict):
            comp = {}

        selected_count = as_int(stats.get("selected_count"), as_int(comp.get("count")))
        generated_count = as_int(stats.get("generated_count"))
        lora_acc = as_float(comp.get("lora_accuracy"), as_float(stats.get("average_full_context_accuracy"), 1.0))
        quality = selected_quality(stats) if stats else 0.0
        frontier = frontier_score(lora_acc, target_low=target_low, target_high=target_high)
        hard_pressure = clamp(1.0 - lora_acc, 0.0, 1.0)
        fail_bonus = min(0.30, 0.10 * failure_by_type.get(evolution_type, 0))
        exploration = count_bonus(selected_count, target=min_count_target)
        reward = quality * (0.50 * frontier + 0.35 * hard_pressure + 0.15 * exploration) + fail_bonus
        if generated_count <= 0:
            reward *= 0.25
        if evolution_type == "temporal_ordering":
            reward *= 0.20
        if lora_acc >= 0.90 and failure_by_type.get(evolution_type, 0) == 0:
            reward *= 0.35

        scores[evolution_type] = {
            "reward": round(reward, 6),
            "quality": round(quality, 6),
            "frontier_score": round(frontier, 6),
            "hard_pressure": round(hard_pressure, 6),
            "generated_count": generated_count,
            "selected_count": selected_count,
            "lora_accuracy": round(lora_acc, 6),
            "critic_safe_rate": round(as_float(stats.get("critic_safe_rate")), 6),
            "option_safe_rate": round(as_float(stats.get("option_safe_rate")), 6),
            "lora_failure_count": failure_by_type.get(evolution_type, 0),
        }
    return scores


def choose_allowed_evolution_types(scores: dict[str, dict[str, Any]], *, min_count: int) -> list[str]:
    ranked = sorted(
        ((key, as_float(value.get("reward"))) for key, value in scores.items() if key != "temporal_ordering"),
        key=lambda item: item[1],
        reverse=True,
    )
    allowed: list[str] = []
    for key, reward in ranked:
        stats = scores[key]
        if reward <= 0:
            continue
        if as_int(stats.get("selected_count")) <= 0 and as_int(stats.get("generated_count")) <= 0:
            continue
        allowed.append(key)

    # Keep enough operator diversity even if one family dominates this round.
    if len(allowed) < min_count:
        for key, _reward in ranked:
            if key not in allowed:
                allowed.append(key)
            if len(allowed) >= min_count:
                break
    return allowed


def build_notes(candidate_scores: dict[str, dict[str, Any]], evolution_scores: dict[str, dict[str, Any]]) -> list[str]:
    ranked_candidates = sorted(candidate_scores.items(), key=lambda item: as_float(item[1].get("reward")), reverse=True)
    ranked_evolutions = sorted(evolution_scores.items(), key=lambda item: as_float(item[1].get("reward")), reverse=True)
    notes = []
    if ranked_candidates:
        top = ", ".join(f"{name}:{stats['reward']}" for name, stats in ranked_candidates[:3])
        notes.append(f"Top candidate types by automatic reward: {top}.")
    if ranked_evolutions:
        top = ", ".join(f"{name}:{stats['reward']}" for name, stats in ranked_evolutions[:3])
        notes.append(f"Top evolution types by automatic reward: {top}.")
    solved = [
        name
        for name, stats in candidate_scores.items()
        if as_float(stats.get("lora_accuracy")) >= 0.90 and as_int(stats.get("lora_failure_count")) == 0
    ]
    if solved:
        notes.append(f"Down-weighted solved candidate types: {', '.join(solved)}.")
    risky = [
        name
        for name, stats in candidate_scores.items()
        if as_float(stats.get("quality")) < 0.55 and as_float(stats.get("reward")) > 0.10
    ]
    if risky:
        notes.append(f"High-pressure but lower-quality types should keep critic/option guards strict: {', '.join(risky)}.")
    return notes


def main() -> None:
    parser = argparse.ArgumentParser(description="Update challenger QA generation policy from solver feedback.")
    parser.add_argument("--report", required=True, type=Path, help="Graph multi-QA report JSON.")
    parser.add_argument("--solver-comparison", required=True, type=Path, help="Base-vs-solver comparison report JSON.")
    parser.add_argument("--output", required=True, type=Path, help="Output policy JSON.")
    parser.add_argument("--target-low", type=float, default=0.30, help="Lower edge of desired solver frontier accuracy.")
    parser.add_argument("--target-high", type=float, default=0.70, help="Upper edge of desired solver frontier accuracy.")
    parser.add_argument("--min-count-target", type=int, default=6, help="Exploration target count per operator.")
    parser.add_argument("--max-weight", type=float, default=0.45, help="Max normalized candidate-type weight.")
    parser.add_argument("--temporal-floor", type=float, default=0.0, help="Reward floor/cap for temporal_pair.")
    parser.add_argument("--min-evolution-types", type=int, default=4, help="Minimum allowed evolution type diversity.")
    args = parser.parse_args()

    multiqa_report = load_json(args.report)
    solver_comparison = load_json(args.solver_comparison)
    candidate_scores = score_candidate_types(
        multiqa_report,
        solver_comparison,
        target_low=args.target_low,
        target_high=args.target_high,
        min_count_target=args.min_count_target,
        temporal_floor=args.temporal_floor,
    )
    evolution_scores = score_evolution_types(
        multiqa_report,
        solver_comparison,
        target_low=args.target_low,
        target_high=args.target_high,
        min_count_target=args.min_count_target,
    )

    raw_weights = {key: as_float(value.get("reward")) for key, value in candidate_scores.items()}
    weights = normalize_weights(raw_weights, max_weight=args.max_weight)
    allowed = choose_allowed_evolution_types(evolution_scores, min_count=args.min_evolution_types)
    output = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_report": str(args.report),
        "source_solver_comparison": str(args.solver_comparison),
        "target_frontier_accuracy": {
            "low": args.target_low,
            "high": args.target_high,
        },
        "candidate_type_weights": weights,
        "candidate_type_weights_arg": build_weight_arg(weights),
        "allowed_evolution_types": allowed,
        "allowed_evolution_types_arg": ",".join(allowed),
        "reward_by_candidate_type": candidate_scores,
        "reward_by_evolution_type": evolution_scores,
        "notes": build_notes(candidate_scores, evolution_scores),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({
        "output": str(args.output),
        "candidate_type_weights_arg": output["candidate_type_weights_arg"],
        "allowed_evolution_types_arg": output["allowed_evolution_types_arg"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
