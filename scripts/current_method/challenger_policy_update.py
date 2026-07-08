from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_CHALLENGER_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_CHALLENGER_MODEL = "Qwen/Qwen3-14B"
REQUIRED_POLICY_KEYS = (
    "policy_id",
    "created_by",
    "method",
    "parent_run",
    "target_round",
    "fixed_solver",
    "attack_hypotheses",
    "attack_skill_weights",
    "question_type_weights",
    "difficulty_targets",
    "anti_shortcut_rules",
    "generation_instructions",
    "expected_frontier_profile",
)
REQUIRED_API_PAYLOAD_KEYS = ("attack_plan", "graph_update", "dialogue", "qa")

DEFAULT_ATTACK_SKILL_TAXONOMY = {
    "belief_vs_reality": 0.13,
    "who_knows_what": 0.13,
    "permission_visibility_scope": 0.10,
    "delayed_activation": 0.10,
    "expired_memory": 0.10,
    "cancel_restore": 0.10,
    "conflicting_reports": 0.10,
    "conditional_plan": 0.10,
    "multi_object_dependency": 0.10,
    "timeline_fork_merge": 0.04,
}
DEFAULT_QUESTION_TYPE_TAXONOMY = {
    "multi_select": 0.225,
    "ordering": 0.225,
    "evidence_selection": 0.225,
    "invalidated_statement_detection": 0.225,
    "state_reconstruction": 0.10,
}


class PolicyUpdateError(RuntimeError):
    pass


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_optional_json(path: str | Path) -> Any | None:
    target = Path(path)
    if not target.exists():
        return None
    try:
        return read_json(target)
    except json.JSONDecodeError:
        return None


def read_jsonl_preview(path: str | Path, limit: int = 50) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
            if len(rows) >= limit:
                break
    return rows


def load_challenger_config(config_path: str | Path) -> dict[str, str | None]:
    payload = read_optional_json(config_path) or {}
    api = payload.get("api") if isinstance(payload, dict) and isinstance(payload.get("api"), dict) else {}
    api_key = api.get("api_key") or os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("OPENAI_API_KEY")
    return {
        "base_url": str(api.get("base_url") or DEFAULT_CHALLENGER_BASE_URL),
        "api_key": str(api_key) if api_key else None,
        "model": str(api.get("model") or DEFAULT_CHALLENGER_MODEL),
    }


def _counter_from_mapping(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    return {}


def _compact_eval_report(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    compact: dict[str, Any] = {}
    for key in (
        "model",
        "adapter_dir",
        "input_root",
        "backend",
        "max_length",
        "max_model_len",
        "max_new_tokens",
        "sampling_k",
        "sample_temperature",
        "summary",
        "prompt_length_stats",
    ):
        if key in payload:
            compact[key] = payload.get(key)
    eval_sets = payload.get("eval_sets")
    if isinstance(eval_sets, list):
        compact["eval_sets"] = []
        for item in eval_sets:
            if isinstance(item, dict):
                compact["eval_sets"].append({key: value for key, value in item.items() if key != "results"})
            else:
                compact["eval_sets"].append(item)
    return compact


def _compact_historical_eval(historical_eval: Any) -> Any:
    if not isinstance(historical_eval, dict):
        return historical_eval
    compact = dict(historical_eval)
    compact["overall_summary"] = _compact_eval_report(historical_eval.get("overall_summary"))
    compact["k5_soft_summary"] = _compact_eval_report(historical_eval.get("k5_soft_summary"))
    return compact


def _question_type_breakdown_from_eval(historical_eval: dict[str, Any]) -> dict[str, Any] | None:
    direct_breakdown = historical_eval.get("question_type_breakdown")
    if isinstance(direct_breakdown, dict) and direct_breakdown:
        return direct_breakdown
    overall_summary = historical_eval.get("overall_summary")
    summary = overall_summary.get("summary") if isinstance(overall_summary, dict) else None
    by_question_type = summary.get("by_question_type") if isinstance(summary, dict) else None
    return by_question_type if isinstance(by_question_type, dict) else None


def _build_active_solver_error_profile(
    *,
    eval_dir: Path | None,
    fixed_solver: str,
    historical_eval: dict[str, Any],
) -> dict[str, Any] | None:
    if eval_dir is None:
        return None
    return {
        "source_eval_dir": str(eval_dir),
        "fixed_solver": fixed_solver,
        "overall_summary": historical_eval.get("overall_summary"),
        "question_type_breakdown": _question_type_breakdown_from_eval(historical_eval),
        "interpretation": (
            "When policy_source_eval_dir is provided, this eval directory represents the active fixed solver "
            "error distribution for the next challenger policy rather than stale historical r2 defaults."
        ),
        "r4_notes": [
            "current_method_r4 should attack the newly trained current_method_r3 fixed solver.",
            "r4 challenger policy should use the active fixed solver full generation pool error profile.",
            "ordering is the main reliable residual weakness.",
            "state_reconstruction is near-saturated and should not dominate generation.",
            "invalidated_statement_detection is excluded from main training because manual audit found semantic instability.",
        ],
        "preferred_question_type_bias": {
            "ordering": "primary reliable residual weakness",
            "multi_select": "secondary coverage",
            "evidence_selection": "secondary coverage",
            "state_reconstruction": "low coverage only",
        },
        "disabled_question_types": ["invalidated_statement_detection"],
        "audit_only_question_types": ["invalidated_statement_detection"],
    }


def _renormalize_compiled_weights(weights: dict[str, float]) -> dict[str, float]:
    positive_weights = {key: float(value) for key, value in weights.items() if float(value) > 0}
    total = sum(positive_weights.values())
    if total <= 0:
        return {}
    return {key: round(value / total, 6) for key, value in sorted(positive_weights.items())}


def build_policy_seed(
    *,
    run_name: str,
    run_root: str | Path,
    policy_dir: str | Path,
    reports_dir: str | Path,
    round_id: str,
    parent_round_id: str | None,
    fixed_solver: str,
    seed_source: str | Path,
    seed_scope: str,
    policy_source_eval_dir: str | Path | None = None,
    previous_run_root: str | Path | None = None,
    solver_eval_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_root = Path(run_root)
    reports_dir = Path(reports_dir)
    policy_dir = Path(policy_dir)
    previous_root = Path(previous_run_root) if previous_run_root else run_root.parent / str(parent_round_id or "previous")
    previous_reports = previous_root / "reports"
    previous_generation = previous_root / "generation"
    eval_dir = Path(policy_source_eval_dir) if policy_source_eval_dir else None

    previous_failure_matrix = read_optional_json(previous_reports / "failure_matrix.json")
    previous_report = read_optional_json(previous_reports / "report.json")
    previous_frontier_preview = read_jsonl_preview(previous_generation / "frontier.jsonl", limit=100)

    historical_eval = {
        "overall_summary": read_optional_json(eval_dir / "overall_summary.json") if eval_dir else None,
        "question_type_breakdown": read_optional_json(eval_dir / "question_type_breakdown.json") if eval_dir else None,
        "k5_soft_summary": read_optional_json(eval_dir / "k5_dev_heldout_soft_summary.json") if eval_dir else None,
    }
    historical_eval = _compact_historical_eval(historical_eval)
    active_solver_error_profile = _build_active_solver_error_profile(
        eval_dir=eval_dir,
        fixed_solver=fixed_solver,
        historical_eval=historical_eval,
    )
    active_solver_eval = solver_eval_summary or read_optional_json(reports_dir / "solver_eval_summary.json")

    soft_buckets: Counter[str] = Counter()
    qtype_breakdown: dict[str, Any] = {}
    attack_skill_breakdown: dict[str, Any] = {}
    bad_question_reasons: Counter[str] = Counter()
    shortcut_warnings: Counter[str] = Counter()
    if isinstance(previous_failure_matrix, dict):
        qtype_breakdown = _counter_from_mapping(previous_failure_matrix.get("by_question_type"))
        attack_skill_breakdown = _counter_from_mapping(previous_failure_matrix.get("by_attack_skill"))
        for key, value in _counter_from_mapping(previous_failure_matrix.get("by_frontier_category")).items():
            if str(key) in {"bad_question", "format_issue", "gold_issue", "ambiguous_issue"}:
                bad_question_reasons[str(key)] += int(value or 0)
        for key, value in _counter_from_mapping(previous_failure_matrix.get("shortcut_failures")).items():
            shortcut_warnings[str(key)] += int(value or 0)
        for group in (
            previous_failure_matrix.get("soft_buckets_by_question_type"),
            previous_failure_matrix.get("soft_buckets_by_attack_skill"),
        ):
            if isinstance(group, dict):
                for buckets in group.values():
                    if isinstance(buckets, dict):
                        for bucket, count in buckets.items():
                            soft_buckets[str(bucket)] += int(count or 0)

    policy_seed = {
        "method": "current_method",
        "run_name": run_name,
        "run_root": str(run_root),
        "round": round_id,
        "parent_round": parent_round_id,
        "fixed_solver": fixed_solver,
        "seed_source": str(seed_source),
        "seed_scope": seed_scope,
        "previous_feedback": {
            "previous_run_root": str(previous_root),
            "failure_matrix": previous_failure_matrix,
            "frontier_preview": previous_frontier_preview,
            "report": previous_report,
            "solver_eval_summary": active_solver_eval,
            "historical_r2_eval_report": historical_eval,
            "k5_soft_accuracy_buckets": dict(sorted(soft_buckets.items())),
            "question_type_breakdown": qtype_breakdown,
            "attack_skill_breakdown": attack_skill_breakdown,
            "bad_question_reasons": dict(sorted(bad_question_reasons.items())),
            "shortcut_warnings": dict(sorted(shortcut_warnings.items())),
        },
        "active_solver_error_profile": active_solver_error_profile,
        "supported_attack_skill_taxonomy": DEFAULT_ATTACK_SKILL_TAXONOMY,
        "supported_question_type_taxonomy": DEFAULT_QUESTION_TYPE_TAXONOMY,
        "required_api_payload_keys": list(REQUIRED_API_PAYLOAD_KEYS),
        "frontier_target_profile": {
            "primary_label": "clean_near_miss",
            "target_soft_accuracy_range": [0.2, 0.6],
            "minimum_partial_credit": 0.4,
            "allow_hard_correct": True,
            "dev_or_heldout_frontier_count": 0,
            "max_largest_question_type_share": 0.35,
            "max_largest_attack_skill_share": 0.25,
        },
        "hard_constraints": [
            "train-only sample0-sample5 seeds",
            "no dev or heldout seeds",
            "no unconstrained LLM gold",
            "gold must be graph/state-chain derivable",
            "anti-shortcut validation must pass",
            "no local template fallback",
            "challenger API payload must contain attack_plan, graph_update, dialogue, qa",
        ],
    }
    if active_solver_error_profile is not None:
        policy_seed["hard_constraints"].extend(
            [
                "disabled question types must not enter main training frontier",
                "invalidated_statement_detection is audit-only unless validation/gold logic is repaired",
            ]
        )
    policy_seed["policy_seed_compaction"] = {
        "compact_eval_reports": True,
        "dropped_large_result_lists": True,
    }
    policy_seed["policy_seed_compaction"]["approximate_json_bytes"] = len(
        json.dumps(policy_seed, ensure_ascii=False).encode("utf-8")
    )
    write_json(policy_dir / "policy_seed.json", policy_seed)
    return policy_seed


def json_candidate_from_text(text: str) -> tuple[dict[str, Any] | None, bool]:
    try:
        payload = json.loads(text)
        return (payload, True) if isinstance(payload, dict) else (None, False)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                payload = json.loads(text[start : end + 1])
                return (payload, True) if isinstance(payload, dict) else (None, False)
            except json.JSONDecodeError:
                return None, False
    return None, False


def validate_raw_policy(policy: Any) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not isinstance(policy, dict):
        return None, {"error_type": "invalid_policy_json", "missing_keys": list(REQUIRED_POLICY_KEYS)}
    missing = [key for key in REQUIRED_POLICY_KEYS if key not in policy]
    if missing:
        return None, {"error_type": "missing_policy_keys", "missing_keys": missing}
    if policy.get("created_by") != "challenger_api":
        return None, {"error_type": "invalid_created_by", "missing_keys": []}
    if policy.get("method") != "current_method":
        return None, {"error_type": "invalid_method", "missing_keys": []}
    return policy, None


def _policy_prompt(policy_seed: dict[str, Any], previous_error: dict[str, Any] | None = None) -> str:
    task = {
        "role": "challenger_policy_optimizer",
        "instructions": [
            "You are the challenger policy optimizer.",
            "Your job is not to write samples now.",
            "Your job is to analyze the fixed solver's failure feedback and propose the next attack policy.",
            "If policy_seed.active_solver_error_profile is present, use it as the active fixed solver feedback for the next policy.",
            "For r4, do not rely on stale historical defaults when active_solver_error_profile is available.",
            "If disabled_question_types is non-empty, assign those question types zero weight or audit-only status.",
            "Do not treat semantically unstable invalidation samples as solver weaknesses.",
            "Prefer ordering-heavy but diversity-constrained attacks when ordering is identified as the main reliable residual weakness.",
            "Avoid overproducing saturated state_reconstruction tasks.",
            "Prioritize clean near-miss frontier yield.",
            "Target soft accuracy range is 0.2 to 0.6.",
            "Do not create ambiguous questions, gold-not-derivable questions, shortcut questions, or impossible 0/5 garbage questions.",
            "Keep question_type and attack_skill diverse.",
            "Return exactly one JSON object.",
        ],
        "required_output_schema": {
            "policy_id": "string",
            "created_by": "challenger_api",
            "method": "current_method",
            "parent_run": "string",
            "target_round": policy_seed.get("round"),
            "fixed_solver": policy_seed.get("fixed_solver"),
            "attack_hypotheses": [],
            "attack_skill_weights": {},
            "question_type_weights": {},
            "difficulty_targets": {
                "target_soft_accuracy_range": [0.2, 0.6],
                "avoid_easy_threshold": 0.8,
                "avoid_too_hard_threshold": 0.0,
            },
            "anti_shortcut_rules": [],
            "generation_instructions": [],
            "expected_frontier_profile": {},
        },
        "policy_seed": policy_seed,
    }
    if previous_error:
        task["previous_response_error"] = previous_error
    return json.dumps(task, ensure_ascii=False)


def _post_chat_completion(config: dict[str, str | None], prompt: str, *, timeout: int, max_tokens: int) -> str:
    api_key = config.get("api_key")
    if not api_key:
        raise PolicyUpdateError("missing_api_config")
    url = str(config.get("base_url") or DEFAULT_CHALLENGER_BASE_URL).rstrip("/") + "/chat/completions"
    payload = {
        "model": config.get("model") or DEFAULT_CHALLENGER_MODEL,
        "messages": [
            {"role": "system", "content": "You are a strict JSON-only challenger policy optimizer."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return str(data["choices"][0]["message"]["content"])


def fallback_policy(policy_seed: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy_id": f"{policy_seed.get('run_name')}_fallback_policy",
        "created_by": "script_fallback",
        "method": "current_method",
        "parent_run": str(policy_seed.get("parent_round") or "r2"),
        "target_round": str(policy_seed.get("round") or "r3"),
        "fixed_solver": str(policy_seed.get("fixed_solver") or ""),
        "attack_hypotheses": [],
        "attack_skill_weights": dict(DEFAULT_ATTACK_SKILL_TAXONOMY),
        "question_type_weights": dict(DEFAULT_QUESTION_TYPE_TAXONOMY),
        "difficulty_targets": {
            "target_soft_accuracy_range": [0.2, 0.6],
            "avoid_easy_threshold": 0.8,
            "avoid_too_hard_threshold": 0.0,
        },
        "anti_shortcut_rules": [],
        "generation_instructions": [],
        "expected_frontier_profile": {},
    }


def challenger_update_policy(
    *,
    policy_seed: dict[str, Any],
    config_path: str | Path,
    raw_policy_path: str | Path,
    max_retries: int = 2,
    timeout: int = 180,
    max_tokens: int = 4096,
    allow_policy_fallback: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_challenger_config(config_path)
    errors: list[dict[str, Any]] = []
    previous_error: dict[str, Any] | None = None
    for attempt in range(1, max_retries + 2):
        try:
            content = _post_chat_completion(config, _policy_prompt(policy_seed, previous_error), timeout=timeout, max_tokens=max_tokens)
            candidate, parsed = json_candidate_from_text(content)
            if not parsed:
                previous_error = {
                    "attempt": attempt,
                    "error_type": "invalid_json",
                    "raw_response_preview": content[:800].replace("\n", "\\n"),
                    "missing_keys": [],
                }
                errors.append(previous_error)
            else:
                policy, validation_error = validate_raw_policy(candidate)
                if policy is not None:
                    write_json(raw_policy_path, policy)
                    return policy, {
                        "policy_created_by": "challenger_api",
                        "policy_source": "challenger_update_policy",
                        "policy_retry_count": attempt - 1,
                        "policy_fallback_used": False,
                        "policy_update_errors": errors,
                        "challenger_model": config.get("model"),
                        "challenger_base_url": config.get("base_url"),
                    }
                previous_error = {
                    "attempt": attempt,
                    **(validation_error or {"error_type": "invalid_policy"}),
                    "raw_response_preview": content[:800].replace("\n", "\\n"),
                }
                errors.append(previous_error)
        except (KeyError, json.JSONDecodeError, urllib.error.URLError, TimeoutError, PolicyUpdateError) as exc:
            previous_error = {
                "attempt": attempt,
                "error_type": "policy_api_error",
                "error": str(exc),
                "missing_keys": [],
                "raw_response_preview": "",
            }
            errors.append(previous_error)
        if attempt <= max_retries:
            time.sleep(min(2**attempt, 20))

    if allow_policy_fallback:
        policy = fallback_policy(policy_seed)
        write_json(raw_policy_path, policy)
        return policy, {
            "policy_created_by": "script_fallback",
            "policy_source": "challenger_update_policy",
            "policy_retry_count": len(errors),
            "policy_fallback_used": True,
            "policy_update_errors": errors,
            "challenger_model": config.get("model"),
            "challenger_base_url": config.get("base_url"),
        }
    raise PolicyUpdateError(f"challenger policy update failed after {len(errors)} attempts: {errors[-1] if errors else {}}")


def normalize_weights(raw: Any, defaults: dict[str, float]) -> tuple[dict[str, float], list[str]]:
    warnings: list[str] = []
    supported = set(defaults)
    if not isinstance(raw, dict):
        raw = {}
        warnings.append("weights_not_object")
    weights: dict[str, float] = {}
    for key, value in raw.items():
        if key not in supported:
            warnings.append(f"unsupported_weight_key:{key}")
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            warnings.append(f"non_numeric_weight:{key}")
            continue
        if numeric > 0:
            weights[key] = numeric
    if not weights:
        weights = {key: float(value) for key, value in defaults.items()}
        warnings.append("used_seed_default_distribution")
    total = sum(weights.values())
    if total <= 0:
        uniform = 1.0 / len(defaults)
        return {key: round(uniform, 6) for key in defaults}, warnings
    normalized = {key: value / total for key, value in weights.items()}
    missing = [key for key in defaults if key not in normalized]
    if missing:
        floor = min(0.01, 0.5 / max(1, len(defaults)))
        scale = max(0.0, 1.0 - floor * len(missing))
        normalized = {key: value * scale for key, value in normalized.items()}
        for key in missing:
            normalized[key] = floor
        denom = sum(normalized.values())
        normalized = {key: value / denom for key, value in normalized.items()}
    return {key: round(value, 6) for key, value in sorted(normalized.items())}, warnings


def clamp_largest_share(weights: dict[str, float], max_share: float) -> dict[str, float]:
    if not weights:
        return {}
    if max_share <= 0:
        uniform = 1.0 / len(weights)
        return {key: round(uniform, 6) for key in sorted(weights)}
    if max_share * len(weights) < 1.0:
        uniform = 1.0 / len(weights)
        return {key: round(uniform, 6) for key in sorted(weights)}
    values = {key: max(0.0, float(value)) for key, value in weights.items()}
    total = sum(values.values()) or 1.0
    values = {key: value / total for key, value in values.items()}
    for _ in range(100):
        offenders = [key for key, value in values.items() if value > max_share + 1e-9]
        if not offenders:
            break
        excess = 0.0
        for key in offenders:
            excess += values[key] - max_share
            values[key] = max_share
        receivers = [key for key, value in values.items() if key not in offenders and value < max_share - 1e-9]
        if not receivers:
            break
        receiver_total = sum(values[key] for key in receivers)
        if receiver_total <= 0:
            add = excess / len(receivers)
            for key in receivers:
                values[key] += add
        else:
            for key in receivers:
                values[key] += excess * values[key] / receiver_total
    denom = sum(values.values()) or 1.0
    values = {key: value / denom for key, value in values.items()}
    return {key: round(value, 6) for key, value in sorted(values.items())}


def validate_and_compile_policy(
    *,
    raw_policy: dict[str, Any],
    policy_seed: dict[str, Any],
    compiled_policy_path: str | Path,
    max_largest_question_type_share: float = 0.35,
    max_largest_attack_skill_share: float = 0.25,
    policy_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    valid_policy, validation_error = validate_raw_policy(raw_policy)
    if valid_policy is None:
        raise PolicyUpdateError(f"raw challenger policy failed validation: {validation_error}")
    if str(valid_policy.get("fixed_solver")) != str(policy_seed.get("fixed_solver")):
        raise PolicyUpdateError(
            f"raw challenger policy fixed_solver mismatch: {valid_policy.get('fixed_solver')} != {policy_seed.get('fixed_solver')}"
        )
    if str(valid_policy.get("target_round")) != str(policy_seed.get("round")):
        raise PolicyUpdateError(
            f"raw challenger policy target_round mismatch: {valid_policy.get('target_round')} != {policy_seed.get('round')}"
        )
    attack_weights, attack_warnings = normalize_weights(
        valid_policy.get("attack_skill_weights"),
        policy_seed.get("supported_attack_skill_taxonomy") or DEFAULT_ATTACK_SKILL_TAXONOMY,
    )
    question_weights, question_warnings = normalize_weights(
        valid_policy.get("question_type_weights"),
        policy_seed.get("supported_question_type_taxonomy") or DEFAULT_QUESTION_TYPE_TAXONOMY,
    )
    active_solver_error_profile = (
        policy_seed.get("active_solver_error_profile") if isinstance(policy_seed.get("active_solver_error_profile"), dict) else None
    )
    disabled_question_types = set((active_solver_error_profile or {}).get("disabled_question_types") or [])
    audit_only_question_types = list((active_solver_error_profile or {}).get("audit_only_question_types") or [])
    if disabled_question_types:
        removed_question_types = sorted(key for key in question_weights if key in disabled_question_types)
        question_weights = {key: value for key, value in question_weights.items() if key not in disabled_question_types}
        if removed_question_types:
            question_warnings.append(f"removed_disabled_question_types:{','.join(removed_question_types)}")
        if not question_weights:
            raise PolicyUpdateError("all compiled question types were disabled by active_solver_error_profile")
        question_weights = _renormalize_compiled_weights(question_weights)
    attack_weights = clamp_largest_share(attack_weights, max_largest_attack_skill_share)
    question_weights = clamp_largest_share(question_weights, max_largest_question_type_share)

    compiled = {
        "policy_id": valid_policy["policy_id"],
        "created_by": valid_policy["created_by"],
        "method": "current_method",
        "parent_run": valid_policy["parent_run"],
        "target_round": valid_policy["target_round"],
        "fixed_solver": valid_policy["fixed_solver"],
        "policy_source": "challenger_update_policy",
        "policy_validation_pass": True,
        "policy_fallback_used": bool((policy_meta or {}).get("policy_fallback_used", False)),
        "policy_retry_count": int((policy_meta or {}).get("policy_retry_count", 0)),
        "attack_hypotheses": list(valid_policy.get("attack_hypotheses") or []),
        "attack_skill_weights": attack_weights,
        "question_type_weights": question_weights,
        "difficulty_targets": {
            **{
                "target_soft_accuracy_range": [0.2, 0.6],
                "avoid_easy_threshold": 0.8,
                "avoid_too_hard_threshold": 0.0,
            },
            **(valid_policy.get("difficulty_targets") if isinstance(valid_policy.get("difficulty_targets"), dict) else {}),
        },
        "anti_shortcut_rules": list(valid_policy.get("anti_shortcut_rules") or []),
        "generation_instructions": list(valid_policy.get("generation_instructions") or []),
        "expected_frontier_profile": valid_policy.get("expected_frontier_profile") or {},
        "required_api_payload_keys": list(REQUIRED_API_PAYLOAD_KEYS),
        "hard_validation_gates": [
            "forbid_dev_heldout_seed",
            "forbid_unconstrained_llm_gold",
            "require_gold_graph_or_state_chain_derivable",
            "require_anti_shortcut_pass",
            "require_no_local_template_fallback",
            "require_challenger_api_payload_keys",
        ],
        "seed_constraints": {
            "seed_scope": policy_seed.get("seed_scope"),
            "allowed_sample_names": ["sample0", "sample1", "sample2", "sample3", "sample4", "sample5"],
            "forbidden_splits": ["dev", "heldout"],
            "forbid_frontier_from_dev_heldout": True,
        },
        "disabled_question_types": sorted(disabled_question_types),
        "audit_only_question_types": audit_only_question_types,
        "active_solver_error_profile_source": (active_solver_error_profile or {}).get("source_eval_dir"),
        "active_solver_error_profile_used": active_solver_error_profile is not None,
        "normalization_warnings": attack_warnings + question_warnings,
        "source_policy_seed": str(Path(compiled_policy_path).with_name("policy_seed.json")),
    }
    write_json(compiled_policy_path, compiled)
    return compiled
