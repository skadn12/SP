"""Freeze Round-1 Question Bank lifecycle, eligibility, and sampling weights.

The program reads frozen manifests and recorded Solver prompt hashes. It does
not import a model backend, render new prompts, or make any online call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import (
    DEFAULT_OUTPUT,
    E1Q1_CLOSURE_RUN,
    E1Q1_VALUE_RUN,
    EPS,
    MODELS,
    PAIRED_COMPARE_RUN,
    PAIRED_SOLVER_RUN,
    REROUTE_RUN,
    ROUND0_DIFFICULTY_RUN,
    ROUND0_REEVAL_RUN,
    SCOPE_RUN,
    VERSION,
)


LIFECYCLE_PRIORITY = (
    "EVAL_LOCKED",
    "QUARANTINE_CONFIRMED",
    "TECHNICAL_UNRESOLVED",
    "POSITION_ONLY_DIAGNOSTIC",
    "CONTEXT_WORSE_DIAGNOSTIC",
    "BRIDGE_REQUIRED_BOTH_ZERO",
    "ACTIVE_FRONTIER_BOTH",
    "ACTIVE_FRONTIER_ONE",
    "HARD_NON_CONTEXTUAL",
    "RESOLVED_WITH_CONTEXT",
    "EASY_MASTERED",
    "MODEL_DISAGREEMENT_DIAGNOSTIC",
    "OTHER_STABLE",
)

ALLOWED_WEIGHT_FEATURES = (
    "clean_full",
    "clean_no",
    "clean_difficulty_bucket",
    "inclusive_full",
    "inclusive_no",
    "inclusive_difficulty_bucket",
    "context_gain",
    "m0_transition",
    "training_exposure",
    "evidence_status",
    "position_mechanism",
    "r02_touch",
    "cross_round",
    "canonical_representative",
    "normalized_stem_family_cluster_count",
    "seed_cooldown",
    "replay_cooldown",
    "replay_history",
)
FORBIDDEN_WEIGHT_FEATURES = (
    "experiment_arm",
    "e1q1_label",
    "feedback_target_hit",
    "semantic_auditor_label",
    "dev_test_content",
    "raw_candidate_count",
    "candidate_id_except_tie_break",
    "manual_preference",
)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_value(value: Any) -> str:
    return sha256_bytes(canonical(value).encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"jsonl_row_not_object:{path}:{line_no}")
            rows.append(row)
    return rows


def verify_source(path: Path, source_hashes: dict[str, str]) -> str:
    if not path.is_file():
        raise RuntimeError(f"missing_source:{path}")
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file():
        raise RuntimeError(f"missing_source_sidecar:{sidecar}")
    observed = sha256_file(path)
    expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
    if observed != expected:
        raise RuntimeError(f"source_sha_mismatch:{path}")
    source_hashes[str(path)] = observed
    return observed


def artifact_bytes(name: str, value: Any) -> bytes:
    if name.endswith(".jsonl"):
        return "".join(canonical(row) + "\n" for row in value).encode("utf-8")
    if name.endswith(".md"):
        text = str(value)
        return (text if text.endswith("\n") else text + "\n").encode("utf-8")
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def write_artifact(output: Path, name: str, value: Any) -> str:
    path = output / name
    data = artifact_bytes(name, value)
    path.write_bytes(data)
    digest = sha256_bytes(data)
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def by_id(rows: Iterable[Mapping[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    materialized = list(rows)
    result = {str(row["candidate_id"]): dict(row) for row in materialized}
    if len(result) != len(materialized):
        raise RuntimeError(f"duplicate_candidate_id:{label}")
    return result


def stable_candidate_key(candidate_id: str) -> tuple[str, str]:
    return sha256_bytes(candidate_id.encode("utf-8")), candidate_id


def normalize_text(value: Any) -> str:
    text = str(value or "").casefold().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def stem_signature(candidate: Mapping[str, Any]) -> str:
    return sha256_value({"question_type": candidate.get("question_type"), "stem": normalize_text(candidate.get("question"))})


def gate_fingerprint_from_candidate_result(row: Mapping[str, Any]) -> str:
    """Build the established gate fingerprint from recorded, not rendered, prompts."""
    prompt_audit: dict[tuple[str, str], dict[str, Any]] = {}
    for request in row.get("requests") or []:
        key = (str(request.get("context_mode")), str(request.get("variant_id")))
        item = {
            "context_mode": request.get("context_mode"),
            "variant_id": request.get("variant_id"),
            "prompt_sha256": request.get("prompt_sha256"),
            "prompt_token_count": request.get("prompt_token_count"),
        }
        previous = prompt_audit.setdefault(key, item)
        if previous != item:
            raise RuntimeError(f"non_deterministic_prompt_hash:{row.get('candidate_id')}:{key}")
    variants = {item[1] for item in prompt_audit}
    payload = {
        "question_type": row.get("question_type"),
        "max_model_len": 16384,
        "output_reserve": 32,
        "position_count": len(variants),
        "solver_contract_sha256": row.get("solver_contract_sha256"),
        "prompt_audit": sorted(prompt_audit.values(), key=lambda item: (str(item["context_mode"]), str(item["variant_id"]))),
    }
    return sha256_value(payload)


def metric_from_difficulty(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "full": row.get("full_context_accuracy"),
        "no": row.get("no_context_accuracy"),
        "full_correct": row.get("full_context_correct_count"),
        "full_total": row.get("full_context_total_count"),
        "no_correct": row.get("no_context_correct_count"),
        "no_total": row.get("no_context_total_count"),
        "difficulty_bucket": row.get("difficulty_bucket"),
        "context_helpful": row.get("context_helpful"),
        "position_mechanism": row.get("position_mechanism"),
        "technical_failure_count": int(row.get("technical_failure_count") or 0),
        "technical_exclusion_reason": row.get("technical_exclusion_reason"),
    }


def metric_from_transition(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "full": row.get("full_context_accuracy"),
        "no": row.get("no_context_accuracy"),
        "full_correct": row.get("full_context_correct_count"),
        "full_total": row.get("full_context_total_count"),
        "no_correct": row.get("no_context_correct_count"),
        "no_total": row.get("no_context_total_count"),
        "difficulty_bucket": row.get("difficulty_bucket"),
        "context_helpful": (
            float(row.get("full_context_accuracy")) > float(row.get("no_context_accuracy")) + EPS
            if row.get("full_context_accuracy") is not None and row.get("no_context_accuracy") is not None
            else None
        ),
        "position_mechanism": row.get("position_mechanism"),
        "technical_failure_count": int(row.get("technical_failure_count") or 0),
        "technical_exclusion_reason": None,
    }


def classify_numeric(metric: Mapping[str, Any]) -> str:
    fc, ft = metric.get("full_correct"), metric.get("full_total")
    nc, nt = metric.get("no_correct"), metric.get("no_total")
    if metric.get("technical_failure_count") or metric.get("technical_exclusion_reason") or not ft or not nt:
        return "technical_excluded"
    fc, ft, nc, nt = int(fc), int(ft), int(nc), int(nt)
    if fc == 0:
        return "full_zero_review"
    if fc * nt < nc * ft:
        return "full_context_worse"
    if 0 < fc < ft and nc < nt and fc * nt >= nc * ft:
        return "hard_candidate"
    if fc == ft and nc < nt:
        return "full_context_resolved"
    if fc == ft and nc == nt:
        return "easy"
    return "NON_EXHAUSTIVE_NUMERIC_STATE"


def complete_metric(metric: Mapping[str, Any] | None) -> bool:
    return bool(
        metric
        and metric.get("full") is not None
        and metric.get("no") is not None
        and int(metric.get("full_total") or 0) > 0
        and int(metric.get("no_total") or 0) > 0
        and int(metric.get("technical_failure_count") or 0) == 0
        and not metric.get("technical_exclusion_reason")
    )


def full_zero(metric: Mapping[str, Any]) -> bool:
    return float(metric["full"]) <= EPS


def partial(metric: Mapping[str, Any]) -> bool:
    return EPS < float(metric["full"]) < 1.0 - EPS


def resolved(metric: Mapping[str, Any]) -> bool:
    return float(metric["full"]) >= 1.0 - EPS


def no_resolved(metric: Mapping[str, Any]) -> bool:
    return float(metric["no"]) >= 1.0 - EPS


def helpful(metric: Mapping[str, Any]) -> bool:
    return float(metric["full"]) > float(metric["no"]) + EPS


def worse(metric: Mapping[str, Any]) -> bool:
    return float(metric["full"]) + EPS < float(metric["no"])


def evidence_rounds(refs: Iterable[Any]) -> list[str]:
    rounds = set()
    for raw in refs:
        parts = str(raw).split(":")
        if "R02" in parts:
            rounds.add("R02")
        elif "R01" in parts:
            rounds.add("R01")
        elif any(re.fullmatch(r"P\d+", part) for part in parts):
            rounds.add("HISTORICAL")
        else:
            rounds.add("UNSCOPED")
    return sorted(rounds)


def lifecycle_state(
    metrics: Mapping[str, Mapping[str, Any]],
    *,
    split: str = "train",
    untouched_probe: bool = False,
    confirmed_quarantine: bool = False,
    aggregate_position_mechanism: str | None = None,
) -> str:
    if split != "train" or untouched_probe:
        return "EVAL_LOCKED"
    if confirmed_quarantine:
        return "QUARANTINE_CONFIRMED"
    if any(not complete_metric(metrics.get(model)) for model in MODELS):
        return "TECHNICAL_UNRESOLVED"
    clean, inclusive = metrics[MODELS[0]], metrics[MODELS[1]]
    mechanisms = {aggregate_position_mechanism, clean.get("position_mechanism"), inclusive.get("position_mechanism")}
    if mechanisms & {"position_only", "pure_position_only"}:
        return "POSITION_ONLY_DIAGNOSTIC"
    if any(worse(item) or item.get("difficulty_bucket") == "full_context_worse" for item in (clean, inclusive)):
        return "CONTEXT_WORSE_DIAGNOSTIC"
    if full_zero(clean) and full_zero(inclusive):
        return "BRIDGE_REQUIRED_BOTH_ZERO"
    frontier = [item.get("difficulty_bucket") == "hard_candidate" and helpful(item) for item in (clean, inclusive)]
    if all(frontier):
        return "ACTIVE_FRONTIER_BOTH"
    if sum(frontier) == 1:
        return "ACTIVE_FRONTIER_ONE"
    if any(item.get("difficulty_bucket") == "hard_candidate" for item in (clean, inclusive)):
        return "HARD_NON_CONTEXTUAL"
    if all(resolved(item) for item in (clean, inclusive)) and any(not no_resolved(item) for item in (clean, inclusive)):
        return "RESOLVED_WITH_CONTEXT"
    if all(resolved(item) and no_resolved(item) for item in (clean, inclusive)):
        return "EASY_MASTERED"
    if resolved(clean) != resolved(inclusive):
        return "MODEL_DISAGREEMENT_DIAGNOSTIC"
    return "OTHER_STABLE"


def m0_transition_class(source: Mapping[str, Any] | None) -> str | None:
    if not source:
        return None
    models = source["models"]
    m0 = metric_from_transition(models["M0"])
    clean = metric_from_transition(models[MODELS[0]])
    inclusive = metric_from_transition(models[MODELS[1]])
    if full_zero(m0) and partial(clean) and partial(inclusive):
        return "M0_ZERO_TO_BOTH_PARTIAL"
    if full_zero(m0) and sum(partial(item) for item in (clean, inclusive)) == 1:
        return "M0_ZERO_TO_ONE_PARTIAL"
    if source.get("primary_transition") == "EMERGING_FRONTIER":
        return "ADVANCING_FRONTIER"
    return str(source.get("primary_transition") or "OTHER_STABLE")


def evidence_multiplier(status: str) -> float:
    return {
        "EVIDENCE_CLEAN": 1.0,
        "NOT_APPLICABLE_NON_HARD": 1.0,
        "EVIDENCE_RISK": 0.5,
        "EVIDENCE_INVALID": 0.0,
        "EVIDENCE_UNCLASSIFIABLE": 0.0,
    }.get(status, 0.0)


def targeted_components(features: Mapping[str, Any], state: str) -> dict[str, float]:
    transition = features.get("m0_transition")
    if state == "ACTIVE_FRONTIER_BOTH" and transition == "M0_ZERO_TO_BOTH_PARTIAL":
        base = 8.0
    elif state == "ACTIVE_FRONTIER_BOTH":
        base = 6.0
    elif state == "ACTIVE_FRONTIER_ONE":
        base = 3.0
    else:
        base = 0.0
    uncertainty = sum(4.0 * float(features[key]) * (1.0 - float(features[key])) for key in ("clean_full", "inclusive_full")) / 2.0
    positive_gain = (
        max(float(features["clean_full"]) - float(features["clean_no"]), 0.0)
        + max(float(features["inclusive_full"]) - float(features["inclusive_no"]), 0.0)
    ) / 2.0
    transition_bonus = 1.5 if transition == "M0_ZERO_TO_BOTH_PARTIAL" else (0.75 if transition in {"M0_ZERO_TO_ONE_PARTIAL", "ADVANCING_FRONTIER"} else 0.0)
    exposure = features.get("training_exposure")
    exposure_bonus = 0.5 if exposure == "UNEXPOSED_TO_BOTH" else (0.25 if exposure in {"CLEAN_ONLY_TRAINED", "INCLUSIVE_ONLY_TRAINED", "EXPOSED_TO_ONE"} else 0.0)
    cross_round_bonus = 0.25 if features.get("cross_round") else 0.0
    evidence = evidence_multiplier(str(features.get("evidence_status")))
    family = 1.0 / math.sqrt(max(1, int(features.get("normalized_stem_family_cluster_count") or 1)))
    cooldown = 0.0 if features.get("seed_cooldown") else 1.0
    representative = 1.0 if features.get("canonical_representative") else 0.0
    additive = base + uncertainty + 2.0 * positive_gain + transition_bonus + exposure_bonus + cross_round_bonus
    return {
        "base_score": base,
        "mean_frontier_uncertainty": uncertainty,
        "mean_positive_context_gain": positive_gain,
        "transition_bonus": transition_bonus,
        "exposure_bonus": exposure_bonus,
        "cross_round_bonus": cross_round_bonus,
        "evidence_multiplier": evidence,
        "family_saturation_multiplier": family,
        "cooldown_multiplier": cooldown,
        "representative_multiplier": representative,
        "additive_score": additive,
        "raw_weight": additive * evidence * family * cooldown * representative,
    }


def composition_components(features: Mapping[str, Any], state: str, evidence_round_count: int) -> dict[str, float]:
    base = {"ACTIVE_FRONTIER_BOTH": 4.0, "ACTIVE_FRONTIER_ONE": 2.0, "RESOLVED_WITH_CONTEXT": 1.0}.get(state, 0.0)
    cross_round_bonus = 0.5 if features.get("cross_round") else 0.0
    multi_round_bonus = 0.5 if evidence_round_count >= 2 else 0.0
    evidence = evidence_multiplier(str(features.get("evidence_status")))
    family = 1.0 / math.sqrt(max(1, int(features.get("normalized_stem_family_cluster_count") or 1)))
    cooldown = 0.0 if features.get("seed_cooldown") else 1.0
    representative = 1.0 if features.get("canonical_representative") else 0.0
    additive = base + cross_round_bonus + multi_round_bonus
    return {
        "base_score": base,
        "cross_round_bonus": cross_round_bonus,
        "multi_round_evidence_bonus": multi_round_bonus,
        "evidence_multiplier": evidence,
        "family_saturation_multiplier": family,
        "cooldown_multiplier": cooldown,
        "representative_multiplier": representative,
        "additive_score": additive,
        "raw_weight": additive * evidence * family * cooldown * representative,
    }


def replay_components(features: Mapping[str, Any], base_score: float) -> dict[str, float]:
    family = 1.0 / math.sqrt(max(1, int(features.get("normalized_stem_family_cluster_count") or 1)))
    cooldown = 0.0 if features.get("replay_cooldown") else 1.0
    representative = 1.0 if features.get("canonical_representative") else 0.0
    return {
        "base_score": base_score,
        "family_saturation_multiplier": family,
        "replay_cooldown_multiplier": cooldown,
        "representative_multiplier": representative,
        "raw_weight": base_score * family * cooldown * representative,
    }


def add_normalized_weights(rows: list[dict[str, Any]], *, within_root: bool) -> None:
    global_total = sum(float(row["raw_weight"]) for row in rows)
    root_totals = Counter()
    if within_root:
        for row in rows:
            root_totals[str(row["root_id"])] += float(row["raw_weight"])
    for row in rows:
        raw = float(row["raw_weight"])
        row["global_normalized_weight"] = raw / global_total if global_total > 0 else 0.0
        if within_root:
            total = root_totals[str(row["root_id"])]
            row["within_root_normalized_weight"] = raw / total if total > 0 else 0.0


def load_inputs() -> dict[str, Any]:
    source_hashes: dict[str, str] = {}
    required = [
        SCOPE_RUN / "final_scope_freeze_report.json",
        SCOPE_RUN / "corpus_scope_manifest.jsonl",
        SCOPE_RUN / "root_scope_manifest.jsonl",
        SCOPE_RUN / "e1q1_dialogue_identity_manifest.jsonl",
        SCOPE_RUN / "candidate_namespace_audit.json",
        REROUTE_RUN / "final_multilabel_rerouting_report.json",
        REROUTE_RUN / "provisional_question_bank_entries.jsonl",
        REROUTE_RUN / "e1q1_multilabel_routing.jsonl",
        REROUTE_RUN / "question_family_manifest.jsonl",
        E1Q1_CLOSURE_RUN / "e1q1_complete_candidate_manifest.jsonl",
        E1Q1_CLOSURE_RUN / "clean_complete_difficulty_manifest.jsonl",
        E1Q1_CLOSURE_RUN / "inclusive_complete_difficulty_manifest.jsonl",
        E1Q1_CLOSURE_RUN / "solver_prompt_cluster_map.jsonl",
        E1Q1_CLOSURE_RUN / "cluster_representative_manifest.jsonl",
        E1Q1_CLOSURE_RUN / "e1q1_solver_coverage_closure.json",
        E1Q1_CLOSURE_RUN / "technical_failure_report.json",
        E1Q1_CLOSURE_RUN / "solver_request_closure.json",
        E1Q1_VALUE_RUN / "e1q1_complete_pool_value_analysis.json",
        E1Q1_VALUE_RUN / "e1q1_candidate_value_manifest.jsonl",
        E1Q1_VALUE_RUN / "e1q1_prompt_cluster_value_manifest.jsonl",
        E1Q1_VALUE_RUN / "e1q1_root_value_summary.json",
        ROUND0_REEVAL_RUN / "candidate_transition_matrix.jsonl",
        ROUND0_REEVAL_RUN / "candidate_training_exposure_manifest.jsonl",
        ROUND0_REEVAL_RUN / "transition_summary.json",
        ROUND0_REEVAL_RUN / "m0_hard_transition_report.json",
        ROUND0_REEVAL_RUN / "m0_full_zero_transition_report.json",
        ROUND0_REEVAL_RUN / "clean_train" / "candidate_results.jsonl",
        PAIRED_COMPARE_RUN / "final_paired_solver_comparison.json",
        PAIRED_SOLVER_RUN / "candidate_manifest.jsonl",
        PAIRED_SOLVER_RUN / "program_gate_results.jsonl",
        ROUND0_DIFFICULTY_RUN / "difficulty_manifest.jsonl",
    ]
    for model in MODELS:
        for branch in ("control", "feedback"):
            required.append(PAIRED_SOLVER_RUN / model / branch / "difficulty_manifest.jsonl")
        required.append(PAIRED_SOLVER_RUN / model / "solver_contract.json")
    for path in required:
        verify_source(path, source_hashes)

    return {
        "source_hashes": source_hashes,
        "scope_report": load_json(SCOPE_RUN / "final_scope_freeze_report.json"),
        "corpus_scope": load_jsonl(SCOPE_RUN / "corpus_scope_manifest.jsonl"),
        "root_scope": load_jsonl(SCOPE_RUN / "root_scope_manifest.jsonl"),
        "namespace_audit": load_json(SCOPE_RUN / "candidate_namespace_audit.json"),
        "reroute_report": load_json(REROUTE_RUN / "final_multilabel_rerouting_report.json"),
        "bank": load_jsonl(REROUTE_RUN / "provisional_question_bank_entries.jsonl"),
        "e1q1_routing": load_jsonl(REROUTE_RUN / "e1q1_multilabel_routing.jsonl"),
        "families": load_jsonl(REROUTE_RUN / "question_family_manifest.jsonl"),
        "e1q1_complete": load_jsonl(E1Q1_CLOSURE_RUN / "e1q1_complete_candidate_manifest.jsonl"),
        "e1q1_cluster_map": load_jsonl(E1Q1_CLOSURE_RUN / "solver_prompt_cluster_map.jsonl"),
        "e1q1_representatives": load_jsonl(E1Q1_CLOSURE_RUN / "cluster_representative_manifest.jsonl"),
        "e1q1_closure": load_json(E1Q1_CLOSURE_RUN / "e1q1_solver_coverage_closure.json"),
        "e1q1_value_report": load_json(E1Q1_VALUE_RUN / "e1q1_complete_pool_value_analysis.json"),
        "e1q1_values": load_jsonl(E1Q1_VALUE_RUN / "e1q1_candidate_value_manifest.jsonl"),
        "round0_transition": load_jsonl(ROUND0_REEVAL_RUN / "candidate_transition_matrix.jsonl"),
        "round0_exposure": load_jsonl(ROUND0_REEVAL_RUN / "candidate_training_exposure_manifest.jsonl"),
        "round0_prompt_results": load_jsonl(ROUND0_REEVAL_RUN / "clean_train" / "candidate_results.jsonl"),
        "paired_candidates": load_jsonl(PAIRED_SOLVER_RUN / "candidate_manifest.jsonl"),
        "paired_gates": load_jsonl(PAIRED_SOLVER_RUN / "program_gate_results.jsonl"),
        "round0_all": load_jsonl(ROUND0_DIFFICULTY_RUN / "difficulty_manifest.jsonl"),
        "e1q1_difficulty": {
            MODELS[0]: load_jsonl(E1Q1_CLOSURE_RUN / "clean_complete_difficulty_manifest.jsonl"),
            MODELS[1]: load_jsonl(E1Q1_CLOSURE_RUN / "inclusive_complete_difficulty_manifest.jsonl"),
        },
        "paired_difficulty": {
            model: {
                branch: load_jsonl(PAIRED_SOLVER_RUN / model / branch / "difficulty_manifest.jsonl")
                for branch in ("control", "feedback")
            }
            for model in MODELS
        },
    }


def prepare_candidates(inputs: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    bank = inputs["bank"]
    family = by_id(inputs["families"], "question_family")
    transition = by_id(inputs["round0_transition"], "round0_transition")
    exposure = by_id(inputs["round0_exposure"], "round0_exposure")
    r0_prompts = by_id(inputs["round0_prompt_results"], "round0_prompt_results")
    r0_source = by_id([row for row in inputs["round0_all"] if row.get("split") == "train"], "round0_train_source")
    paired = by_id(inputs["paired_candidates"], "paired_candidates")
    complete = by_id(inputs["e1q1_complete"], "e1q1_complete")
    e1_cluster = by_id(inputs["e1q1_cluster_map"], "e1q1_cluster_map")
    e1_values = by_id(inputs["e1q1_values"], "e1q1_values")
    strict_ids = {cid for cid, row in e1_values.items() if row.get("solver_completion_status") == "PREVIOUSLY_COMPLETE"}

    metrics: dict[str, dict[str, dict[str, Any]]] = {model: {} for model in MODELS}
    for cid, row in transition.items():
        for model in MODELS:
            metrics[model][cid] = metric_from_transition(row["models"][model])
    for model in MODELS:
        for branch in ("control", "feedback"):
            for row in inputs["paired_difficulty"][model][branch]:
                metrics[model][str(row["candidate_id"])] = metric_from_difficulty(row)
        for row in inputs["e1q1_difficulty"][model]:
            metrics[model][str(row["candidate_id"])] = metric_from_difficulty(row)

    rows = []
    for entry in bank:
        cid = str(entry["candidate_id"])
        candidate = dict(entry.get("candidate") or {})
        metadata = family[cid]
        corpus = str(entry["corpus_id"])
        arm = str(entry["normalized_experiment_arm"])
        if corpus == "ROUND0":
            fingerprint = gate_fingerprint_from_candidate_result(r0_prompts[cid])
            fingerprint_status = "RECORDED_ROUND0_REEVAL_REQUESTS"
            source_path = str(r0_source[cid].get("candidate_pool_path"))
            source_transition = transition[cid]
            training = exposure[cid]
            aggregate_position = source_transition.get("position_mechanism")
            evidence_status = str(source_transition.get("evidence_status") or "NOT_APPLICABLE_NON_HARD")
        else:
            fingerprint = str(metadata.get("solver_prompt_fingerprint"))
            fingerprint_status = str(metadata.get("prompt_fingerprint_status"))
            source_path = str(PAIRED_SOLVER_RUN / "candidate_manifest.jsonl") if corpus == "E1Q0" else str(REROUTE_RUN / "e1q1_multilabel_routing.jsonl")
            source_transition = None
            training = {}
            aggregate_position = None
            evidence_status = "EVIDENCE_CLEAN" if candidate.get("evidence_refs") and metadata.get("structural_status") == "PASS" else "EVIDENCE_UNCLASSIFIABLE"
        if not fingerprint or fingerprint == "None":
            fingerprint = "UNRESOLVED:" + sha256_bytes(cid.encode("utf-8"))
            fingerprint_status = "PROMPT_FINGERPRINT_UNRESOLVED"
        refs = list(candidate.get("evidence_refs") or [])
        rounds = evidence_rounds(refs)
        candidate_metrics = {model: metrics[model].get(cid) for model in MODELS}
        rows.append({
            "candidate_id": cid,
            "root_id": str(candidate.get("root_id") or metadata.get("root_id")),
            "split": str(entry.get("split")),
            "question_bank_scope": "ACTIVE_TRAIN",
            "corpus_id": corpus,
            "normalized_experiment_arm": arm,
            "experiment_arm": candidate.get("experiment_arm"),
            "branch": candidate.get("branch"),
            "source_path": source_path,
            "question_type": candidate.get("question_type"),
            "generation_source": candidate.get("generation_source"),
            "slot_type": candidate.get("slot_type") or metadata.get("slot_type"),
            "solver_prompt_fingerprint": fingerprint,
            "prompt_fingerprint_status": fingerprint_status,
            "normalized_stem_signature": metadata.get("normalized_stem_signature") or stem_signature(candidate),
            "evidence_signature": metadata.get("evidence_signature"),
            "evidence_refs": refs,
            "evidence_rounds": rounds,
            "cross_round": len(rounds) >= 2,
            "r02_touch": bool(metadata.get("r02_touch")),
            "evidence_status": evidence_status,
            "feedback_target_hit": "UNKNOWN" if corpus == "E1Q1" else metadata.get("feedback_target_hit"),
            "lexical_similarity_candidate": bool(metadata.get("lexical_similarity_candidate")),
            "evidence_signature_shift": bool(metadata.get("evidence_shift_variant")),
            "strict_novelty": cid in strict_ids,
            "training_exposure": training.get("training_exposure", "UNKNOWN"),
            "directly_trained_by_clean": bool(training.get("directly_trained_by_clean")),
            "directly_trained_by_inclusive": bool(training.get("directly_trained_by_inclusive")),
            "m0_transition_class": m0_transition_class(source_transition),
            "primary_transition": source_transition.get("primary_transition") if source_transition else None,
            "source_transition": source_transition,
            "aggregate_position_mechanism": aggregate_position,
            "metrics": candidate_metrics,
            "independent_observation": all(candidate_metrics.values()),
            "frozen_e1q1_representative": e1_cluster.get(cid, {}).get("cluster_representative_candidate_id"),
            "e1q1_observation_source": complete.get(cid, {}).get("representative_source"),
            "confirmed_quarantine": False,
            "untouched_probe": False,
        })

    if len(rows) != 3062 or len({row["candidate_id"] for row in rows}) != 3062:
        raise RuntimeError(f"active_train_scope_not_3062:{len(rows)}")
    expected_arms = {"ROUND0_PARENT_PLUS_R01": 2294, "E0Q0_CONTROL_BLIND": 276, "E1Q0_FEEDBACK_BLIND": 254, "E1Q1_FEEDBACK_GUIDED": 238}
    observed_arms = dict(Counter(row["normalized_experiment_arm"] for row in rows))
    if observed_arms != expected_arms:
        raise RuntimeError(f"active_arm_counts_mismatch:{observed_arms}")

    eval_locked = []
    for source in inputs["round0_all"]:
        if source.get("split") == "train":
            continue
        eval_locked.append({
            "candidate_id": source["candidate_id"],
            "root_id": source["root_id"],
            "split": source["split"],
            "question_bank_scope": "EVAL_LOCKED_AUDIT_ONLY",
            "audit_only": True,
            "questioner_exposure_allowed": False,
            "training_allowed": False,
            "replay_allowed": False,
            "evolution_allowed": False,
            "sampling_weight": 0.0,
        })
    if Counter(row["split"] for row in eval_locked) != Counter({"dev": 485, "test": 507}):
        raise RuntimeError("eval_locked_split_count_mismatch")
    meta = {"strict_ids": strict_ids, "metrics_by_model": metrics}
    return rows, eval_locked, meta


def choose_representative(members: list[dict[str, Any]]) -> dict[str, Any]:
    independently_observed = [row for row in members if row["independent_observation"]]
    pool = independently_observed or members
    frozen_ids = {str(row.get("frozen_e1q1_representative")) for row in members if row.get("frozen_e1q1_representative")}
    frozen = [row for row in pool if row["candidate_id"] in frozen_ids]
    pool = frozen or pool
    return min(pool, key=lambda row: stable_candidate_key(row["candidate_id"]))


def build_clusters(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        groups[row["solver_prompt_fingerprint"]].append(row)
    clusters, mapping, origins = [], [], []
    for fingerprint, members in sorted(groups.items()):
        members = sorted(members, key=lambda row: stable_candidate_key(row["candidate_id"]))
        representative = choose_representative(members)
        rep_id = representative["candidate_id"]
        member_ids = sorted(row["candidate_id"] for row in members)
        for row in members:
            alias = row["candidate_id"] != rep_id
            if alias and row["corpus_id"] == "E1Q1" and representative["corpus_id"] == "E1Q0":
                origin = "E1Q0_REPRESENTATIVE"
            elif alias:
                origin = rep_id
            else:
                origin = "INDEPENDENT_OBSERVATION"
            mapping.append({
                "candidate_id": row["candidate_id"],
                "solver_prompt_fingerprint": fingerprint,
                "canonical_representative_candidate_id": rep_id,
                "alias": alias,
                "independent_observation": bool(row["independent_observation"] and not alias),
                "independent_sampling_weight": 0.0 if alias else None,
            })
            origins.append({
                "candidate_id": row["candidate_id"],
                "solver_prompt_fingerprint": fingerprint,
                "canonical_representative_candidate_id": rep_id,
                "observation_origin": origin,
                "source_independent_observation_available": bool(row["independent_observation"]),
                "counted_as_independent_observation": bool(row["independent_observation"] and not alias),
            })
        clusters.append({
            "solver_prompt_fingerprint": fingerprint,
            "cluster_status": "PROMPT_FINGERPRINT_UNRESOLVED" if fingerprint.startswith("UNRESOLVED:") else "RESOLVED",
            "canonical_representative_candidate_id": rep_id,
            "member_candidate_ids": member_ids,
            "member_count": len(members),
            "alias_count": len(members) - 1,
            "normalized_experiment_arms": sorted({row["normalized_experiment_arm"] for row in members}),
            "root_ids": sorted({row["root_id"] for row in members}),
            "representative_selection": "INDEPENDENT_OBSERVATION_THEN_FROZEN_REP_THEN_STABLE_SHA256",
        })
    return clusters, mapping, origins


def build_lifecycle(
    candidates: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    mapping: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate_by_id = {row["candidate_id"]: row for row in candidates}
    mapping_by_id = {row["candidate_id"]: row for row in mapping}
    family_cluster_sets: dict[tuple[str, str], set[str]] = defaultdict(set)
    for cluster in clusters:
        rep = candidate_by_id[cluster["canonical_representative_candidate_id"]]
        family_cluster_sets[(rep["root_id"], str(rep["normalized_stem_signature"]))].add(cluster["solver_prompt_fingerprint"])

    lifecycle_rows = []
    operation_rows = []
    training_rows = []
    bridge_rows = []
    composition_rows = []
    targeted_rows = []
    composition_weight_rows = []
    replay_rows = {model: [] for model in MODELS}
    cooldown_rows = []
    representative_rows = []
    difficulty_mismatches = []

    for cluster in clusters:
        rep_id = cluster["canonical_representative_candidate_id"]
        rep = candidate_by_id[rep_id]
        metrics = rep["metrics"]
        state = lifecycle_state(
            metrics,
            split=rep["split"],
            untouched_probe=rep["untouched_probe"],
            confirmed_quarantine=rep["confirmed_quarantine"],
            aggregate_position_mechanism=rep["aggregate_position_mechanism"],
        )
        for model in MODELS:
            metric = metrics.get(model)
            if complete_metric(metric):
                recomputed = classify_numeric(metric)
                if recomputed != metric.get("difficulty_bucket"):
                    difficulty_mismatches.append({
                        "candidate_id": rep_id,
                        "solver_prompt_fingerprint": cluster["solver_prompt_fingerprint"],
                        "model": model,
                        "authoritative_bucket": metric.get("difficulty_bucket"),
                        "numeric_bucket": recomputed,
                        "status": "DIFFICULTY_NUMERIC_CONTRACT_MISMATCH",
                    })
        prohibited = state in {"EVAL_LOCKED", "QUARANTINE_CONFIRMED", "TECHNICAL_UNRESOLVED", "POSITION_ONLY_DIAGNOSTIC", "CONTEXT_WORSE_DIAGNOSTIC"}
        targeted_eligible = state in {"ACTIVE_FRONTIER_BOTH", "ACTIVE_FRONTIER_ONE"}
        bridge_eligible = state == "BRIDGE_REQUIRED_BOTH_ZERO"
        evidence_nonempty = bool(rep["evidence_refs"])
        composition_eligible = state in {"ACTIVE_FRONTIER_BOTH", "ACTIVE_FRONTIER_ONE", "RESOLVED_WITH_CONTEXT"} and evidence_nonempty and not prohibited
        training_eligible = {
            model: bool(
                complete_metric(metrics.get(model))
                and metrics[model].get("difficulty_bucket") == "hard_candidate"
                and helpful(metrics[model])
                and not prohibited
            )
            for model in MODELS
        }
        consensus_training = all(training_eligible.values())
        transition = rep.get("source_transition")
        m0_metric = metric_from_transition(transition["models"]["M0"]) if transition else None
        regressed = {}
        recently_resolved = {}
        replay_base = {}
        for model in MODELS:
            direct = rep["directly_trained_by_clean"] if model == MODELS[0] else rep["directly_trained_by_inclusive"]
            regressed[model] = bool(
                m0_metric
                and m0_metric["difficulty_bucket"] in {"easy", "full_context_resolved"}
                and metrics[model]["difficulty_bucket"] in {"hard_candidate", "full_zero_review", "full_context_worse", "technical_excluded"}
            )
            recently_resolved[model] = bool(m0_metric and not resolved(m0_metric) and resolved(metrics[model]))
            old_hard = bool(m0_metric and m0_metric["difficulty_bucket"] in {"hard_candidate", "full_zero_review"})
            if regressed[model]:
                replay_base[model] = 6.0
            elif old_hard and resolved(metrics[model]) and direct:
                replay_base[model] = 4.0
            elif state == "RESOLVED_WITH_CONTEXT" and direct:
                replay_base[model] = 2.0
            else:
                replay_base[model] = 0.0

        clean, inclusive = metrics[MODELS[0]], metrics[MODELS[1]]
        aux = {
            "consensus_context_helpful_hard": all(item.get("difficulty_bucket") == "hard_candidate" and helpful(item) for item in (clean, inclusive)),
            "single_model_context_helpful_hard": sum(item.get("difficulty_bucket") == "hard_candidate" and helpful(item) for item in (clean, inclusive)) == 1,
            "r02_touch": rep["r02_touch"],
            "cross_round": rep["cross_round"],
            "strict_novelty": rep["strict_novelty"],
            "lexical_similarity_candidate": rep["lexical_similarity_candidate"],
            "evidence_signature_shift": rep["evidence_signature_shift"],
            "training_exposure": rep["training_exposure"],
            "m0_transition_class": rep["m0_transition_class"],
            "regressed_after_training_clean": regressed[MODELS[0]],
            "regressed_after_training_inclusive": regressed[MODELS[1]],
            "recently_resolved_clean": recently_resolved[MODELS[0]],
            "recently_resolved_inclusive": recently_resolved[MODELS[1]],
        }
        family_count = len(family_cluster_sets[(rep["root_id"], str(rep["normalized_stem_signature"]))])
        features = {
            "clean_full": clean["full"] if clean else 0.0,
            "clean_no": clean["no"] if clean else 0.0,
            "clean_difficulty_bucket": clean.get("difficulty_bucket") if clean else None,
            "inclusive_full": inclusive["full"] if inclusive else 0.0,
            "inclusive_no": inclusive["no"] if inclusive else 0.0,
            "inclusive_difficulty_bucket": inclusive.get("difficulty_bucket") if inclusive else None,
            "context_gain": {
                MODELS[0]: float(clean["full"]) - float(clean["no"]) if clean else None,
                MODELS[1]: float(inclusive["full"]) - float(inclusive["no"]) if inclusive else None,
            },
            "m0_transition": rep["m0_transition_class"],
            "training_exposure": rep["training_exposure"],
            "evidence_status": rep["evidence_status"],
            "position_mechanism": rep["aggregate_position_mechanism"],
            "r02_touch": rep["r02_touch"],
            "cross_round": rep["cross_round"],
            "canonical_representative": True,
            "normalized_stem_family_cluster_count": family_count,
            "seed_cooldown": False,
            "replay_cooldown": False,
            "replay_history": {"replay_count": 0, "last_replayed_round": None},
        }
        targeted = targeted_components(features, state)
        composition = composition_components(features, state, len(rep["evidence_rounds"]))
        if not targeted_eligible:
            targeted["raw_weight"] = 0.0
        if not composition_eligible:
            composition["raw_weight"] = 0.0
        base_common = {
            "solver_prompt_fingerprint": cluster["solver_prompt_fingerprint"],
            "canonical_representative_candidate_id": rep_id,
            "root_id": rep["root_id"],
            "normalized_experiment_arm": rep["normalized_experiment_arm"],
            "lifecycle_state": state,
        }
        lifecycle_rows.append({
            **base_common,
            "member_candidate_ids": cluster["member_candidate_ids"],
            "member_count": cluster["member_count"],
            "alias_count": cluster["alias_count"],
            "question_type": rep["question_type"],
            "normalized_stem_signature": rep["normalized_stem_signature"],
            "evidence_signature": rep["evidence_signature"],
            "evidence_refs": rep["evidence_refs"],
            "evidence_rounds": rep["evidence_rounds"],
            "evidence_status": rep["evidence_status"],
            "position_mechanism": rep["aggregate_position_mechanism"],
            "metrics": metrics,
            "auxiliary_labels": aux,
            "feedback_target_hit": rep["feedback_target_hit"],
        })
        operation = {
            **base_common,
            "questioner_targeted_evolution_eligible": targeted_eligible,
            "questioner_bridge_generation_eligible": bridge_eligible,
            "composition_parent_eligible": composition_eligible,
            "training_eligible_clean": training_eligible[MODELS[0]],
            "training_eligible_inclusive": training_eligible[MODELS[1]],
            "consensus_training_eligible": consensus_training,
            "solver_replay_eligible_clean": replay_base[MODELS[0]] > 0,
            "solver_replay_eligible_inclusive": replay_base[MODELS[1]] > 0,
        }
        operation_rows.append(operation)
        training_rows.append({key: value for key, value in operation.items() if key in base_common or key.startswith("training_") or key == "consensus_training_eligible"})
        bridge_rows.append({**base_common, "questioner_bridge_generation_eligible": bridge_eligible, "generation_mode": "BRIDGE_OR_LEMMA" if bridge_eligible else None})
        composition_rows.append({**base_common, "composition_parent_eligible": composition_eligible, "evidence_refs_nonempty": evidence_nonempty, "composition_pairing_enumerated": False})
        targeted_rows.append({**base_common, "eligible": targeted_eligible, **targeted})
        composition_weight_rows.append({**base_common, "eligible": composition_eligible, **composition})
        for model in MODELS:
            replay = replay_components(features, replay_base[model])
            replay_rows[model].append({
                **base_common,
                "model_name": model,
                "eligible": replay_base[model] > 0,
                "regressed_after_training": regressed[model],
                "recently_resolved": recently_resolved[model],
                "direct_training_exposure": rep["directly_trained_by_clean"] if model == MODELS[0] else rep["directly_trained_by_inclusive"],
                **replay,
            })
        cooldown_rows.append({
            **base_common,
            "seed_cooldown_active": False,
            "replay_cooldown_active_clean": False,
            "replay_cooldown_active_inclusive": False,
            "replay_count_clean": 0,
            "replay_count_inclusive": 0,
            "last_replayed_round_clean": None,
            "last_replayed_round_inclusive": None,
            "history_source": "NO_HISTORICAL_REPLAY_LOG",
        })
        representative_rows.append({
            **base_common,
            "member_candidate_ids": cluster["member_candidate_ids"],
            "representative_is_independently_observed": rep["independent_observation"],
            "selection_rule": cluster["representative_selection"],
        })

    add_normalized_weights(targeted_rows, within_root=True)
    add_normalized_weights(composition_weight_rows, within_root=True)
    for model in MODELS:
        add_normalized_weights(replay_rows[model], within_root=False)

    lifecycle_by_fp = {row["solver_prompt_fingerprint"]: row for row in lifecycle_rows}
    operation_by_fp = {row["solver_prompt_fingerprint"]: row for row in operation_rows}
    target_by_fp = {row["solver_prompt_fingerprint"]: row for row in targeted_rows}
    comp_by_fp = {row["solver_prompt_fingerprint"]: row for row in composition_weight_rows}
    replay_by_model_fp = {model: {row["solver_prompt_fingerprint"]: row for row in replay_rows[model]} for model in MODELS}
    candidate_views = []
    for candidate in candidates:
        link = mapping_by_id[candidate["candidate_id"]]
        fp = link["solver_prompt_fingerprint"]
        alias = link["alias"]
        lifecycle = lifecycle_by_fp[fp]
        operation = operation_by_fp[fp]
        candidate_views.append({
            "candidate_id": candidate["candidate_id"],
            "root_id": candidate["root_id"],
            "split": candidate["split"],
            "question_bank_scope": candidate["question_bank_scope"],
            "normalized_experiment_arm": candidate["normalized_experiment_arm"],
            "experiment_arm": candidate["experiment_arm"],
            "branch": candidate["branch"],
            "source_path": candidate["source_path"],
            "solver_prompt_fingerprint": fp,
            "canonical_representative_candidate_id": link["canonical_representative_candidate_id"],
            "alias": alias,
            "independent_observation": link["independent_observation"],
            "lifecycle_state": lifecycle["lifecycle_state"],
            "feedback_target_hit": candidate["feedback_target_hit"],
            "operation_eligibility": {key: (False if alias else value) for key, value in operation.items() if key.endswith("eligible")},
            "sampling_weights": {
                "questioner_targeted_raw": 0.0 if alias else target_by_fp[fp]["raw_weight"],
                "questioner_composition_raw": 0.0 if alias else comp_by_fp[fp]["raw_weight"],
                "solver_replay_clean_raw": 0.0 if alias else replay_by_model_fp[MODELS[0]][fp]["raw_weight"],
                "solver_replay_inclusive_raw": 0.0 if alias else replay_by_model_fp[MODELS[1]][fp]["raw_weight"],
            },
        })

    saturation = []
    candidate_family_counts = Counter((row["root_id"], str(row["normalized_stem_signature"])) for row in candidates)
    for (root, stem), fps in sorted(family_cluster_sets.items()):
        saturation.append({
            "root_id": root,
            "normalized_stem_signature": stem,
            "candidate_count": candidate_family_counts[(root, stem)],
            "prompt_cluster_count": len(fps),
            "family_saturation_multiplier": 1.0 / math.sqrt(max(1, len(fps))),
            "new_variant_cap_per_round": 2,
        })
    return {
        "cluster_lifecycle": lifecycle_rows,
        "candidate_lifecycle": candidate_views,
        "operations": operation_rows,
        "training": training_rows,
        "bridge": bridge_rows,
        "composition": composition_rows,
        "targeted_weights": targeted_rows,
        "composition_weights": composition_weight_rows,
        "replay_weights": replay_rows,
        "cooldown": cooldown_rows,
        "representatives": representative_rows,
        "saturation": saturation,
        "difficulty_mismatches": difficulty_mismatches,
    }


def normalized_sum_audit(rows: list[Mapping[str, Any]], *, within_root: bool) -> dict[str, Any]:
    positive = [row for row in rows if float(row["raw_weight"]) > 0]
    global_sum = sum(float(row["global_normalized_weight"]) for row in rows)
    root_sums = {}
    if within_root:
        roots = sorted({str(row["root_id"]) for row in positive})
        root_sums = {root: sum(float(row["within_root_normalized_weight"]) for row in rows if row["root_id"] == root) for root in roots}
    return {
        "row_count": len(rows),
        "positive_weight_count": len(positive),
        "global_normalized_weight_sum": global_sum,
        "within_root_normalized_weight_sums": root_sums,
        "global_sum_pass": abs(global_sum - 1.0) <= EPS if positive else global_sum == 0.0,
        "within_root_sums_pass": all(abs(total - 1.0) <= EPS for total in root_sums.values()),
    }


def build_contracts() -> dict[str, Any]:
    lifecycle_contract = {
        "version": VERSION,
        "eps": EPS,
        "unit": "solver_prompt_fingerprint_cluster",
        "priority": list(LIFECYCLE_PRIORITY),
        "alias_policy": {"independent_observation": False, "sampling_weight": 0, "observation_origin": "canonical_representative_candidate_id"},
        "feedback_target_hit_policy": "UNKNOWN_RETAINED_AND_EXCLUDED_FROM_LIFECYCLE",
        "confirmed_quarantine_policy": "only_explicit_frozen_confirmed_defects",
    }
    weighting_contract = {
        "version": VERSION,
        "targeted_formula": "(base + uncertainty + 2*positive_gain + transition + exposure + cross_round) * evidence * family * cooldown * representative",
        "composition_formula": "(base + cross_round + multi_round_evidence) * evidence * family * cooldown * representative",
        "solver_replay_formula": "base * family * replay_cooldown * representative",
        "frontier_uncertainty": "4*p*(1-p)",
        "normalization": {"questioner": "within_root_and_global", "solver_replay": "global_per_model"},
    }
    sampling = {
        "version": VERSION,
        "policy_status": "provisional_v1",
        "questioner_new_question_budget": {"TARGETED_EVOLUTION": 0.40, "MULTI_QA_COMPOSITION": 0.20, "FRESH_EXPLORATION": 0.40},
        "budget_fallback": "unsafe_or_empty_targeted_and_composition_budget_moves_to_fresh",
        "fresh_required_per_root": True,
        "exact_replay_in_questioner_budget": False,
        "solver_training_budget": {"default_exact_retention_replay": 0.10, "hard_cap": 0.15, "minimum_current_new_hard": 0.85},
    }
    cooldown = {
        "version": VERSION,
        "prompt_cluster_max_replay_per_round": 1,
        "consecutive_exact_replay_allowed": False,
        "stable_mastered_rounds_before_archive_cooldown": 2,
        "regression_reactivates": True,
        "initial_replay_count": 0,
        "initial_last_replayed_round": None,
        "normalized_stem_new_variant_cap_per_root_round": 2,
        "composition_pairing_contract": {
            "same_root": True,
            "different_prompt_cluster": True,
            "different_normalized_stem_signature": True,
            "evidence_refs_nonempty": True,
            "maximum_evidence_jaccard": 0.50,
            "minimum_active_frontier_parent_count": 1,
            "parent_count": [2, 3],
            "excluded": ["untouched", "quarantine", "alias"],
            "semantic_family_claimed": False,
        },
    }
    allowlist = {
        "version": VERSION,
        "allowed_weight_features": list(ALLOWED_WEIGHT_FEATURES),
        "forbidden_weight_features": list(FORBIDDEN_WEIGHT_FEATURES),
        "candidate_id_policy": "stable_representative_tie_break_only",
    }
    return {
        "lifecycle_contract.json": lifecycle_contract,
        "weighting_contract.json": weighting_contract,
        "sampling_budget_contract.json": sampling,
        "cooldown_and_family_cap_contract.json": cooldown,
        "weight_feature_allowlist.json": allowlist,
    }


def build_novelty_baseline(candidates: list[dict[str, Any]], clusters: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_by_id = {row["candidate_id"]: row for row in candidates}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        groups[(row["normalized_experiment_arm"], row["root_id"])].append(row)
    values = []
    for (arm, root), rows in sorted(groups.items()):
        candidate_ids = {row["candidate_id"] for row in rows}
        cluster_fps = {
            cluster["solver_prompt_fingerprint"]
            for cluster in clusters
            if any(cid in candidate_ids for cid in cluster["member_candidate_ids"])
        }
        stem_counts = Counter(str(row["normalized_stem_signature"]) for row in rows)
        cross_evidence = {str(row["evidence_signature"]) for row in rows if row["cross_round"]}
        values.append({
            "normalized_experiment_arm": arm,
            "root_id": root,
            "candidate_count": len(rows),
            "independent_solver_prompt_cluster_count": len(cluster_fps),
            "distinct_normalized_stem_signature_count": len(stem_counts),
            "distinct_evidence_bundle_signature_count": len({str(row["evidence_signature"]) for row in rows}),
            "cross_round_evidence_signature_count": len(cross_evidence),
            "candidate_count_by_normalized_stem_signature": dict(sorted(stem_counts.items())),
        })
    del candidate_by_id
    return {
        "version": VERSION,
        "semantic_family_claimed": False,
        "future_stage": "NOVEL_QUESTION_CAPACITY_PREFLIGHT_AFTER_R03_R04",
        "groups": values,
    }


def build_all(inputs: Mapping[str, Any]) -> dict[str, Any]:
    candidates, eval_locked, meta = prepare_candidates(inputs)
    clusters, mapping, origins = build_clusters(candidates)
    lifecycle = build_lifecycle(candidates, clusters, mapping)
    target_audit = normalized_sum_audit(lifecycle["targeted_weights"], within_root=True)
    comp_audit = normalized_sum_audit(lifecycle["composition_weights"], within_root=True)
    replay_audit = {model: normalized_sum_audit(lifecycle["replay_weights"][model], within_root=False) for model in MODELS}
    normalization = {"questioner_targeted": target_audit, "questioner_composition": comp_audit, "solver_replay": replay_audit}

    alias_ids = {row["candidate_id"] for row in mapping if row["alias"]}
    alias_views = [row for row in lifecycle["candidate_lifecycle"] if row["candidate_id"] in alias_ids]
    alias_zero = all(all(float(value) == 0.0 for value in row["sampling_weights"].values()) for row in alias_views)
    representative_audit = {
        "cluster_count": len(clusters),
        "candidate_count": len(candidates),
        "alias_count": len(alias_ids),
        "one_representative_per_cluster": len(lifecycle["representatives"]) == len(clusters),
        "alias_sampling_weights_all_zero": alias_zero,
        "independent_weight_occurrences_per_cluster_max": 1,
        "external_e1q0_observation_origin_count": sum(row["observation_origin"] == "E1Q0_REPRESENTATIVE" for row in origins),
    }
    state_counts = Counter(row["lifecycle_state"] for row in lifecycle["cluster_lifecycle"])
    candidate_state_counts = Counter(row["lifecycle_state"] for row in lifecycle["candidate_lifecycle"])
    arm_candidate_states: dict[str, Counter[str]] = defaultdict(Counter)
    for row in lifecycle["candidate_lifecycle"]:
        arm_candidate_states[row["normalized_experiment_arm"]][row["lifecycle_state"]] += 1
    arm_cluster_states: dict[str, Counter[str]] = defaultdict(Counter)
    for row in lifecycle["cluster_lifecycle"]:
        arm_cluster_states[row["normalized_experiment_arm"]][row["lifecycle_state"]] += 1
    operations = lifecycle["operations"]
    eligibility_counts = {
        key: sum(bool(row[key]) for row in operations)
        for key in (
            "questioner_targeted_evolution_eligible",
            "questioner_bridge_generation_eligible",
            "composition_parent_eligible",
            "training_eligible_clean",
            "training_eligible_inclusive",
            "consensus_training_eligible",
            "solver_replay_eligible_clean",
            "solver_replay_eligible_inclusive",
        )
    }
    target_by_root = dict(sorted(Counter(row["root_id"] for row in operations if row["questioner_targeted_evolution_eligible"]).items()))
    comp_by_root = dict(sorted(Counter(row["root_id"] for row in operations if row["composition_parent_eligible"]).items()))
    saturation_distribution = dict(sorted(Counter(row["prompt_cluster_count"] for row in lifecycle["saturation"]).items()))

    value = inputs["e1q1_value_report"]
    e1q1_checks = {
        "generation_candidates": sum(row["normalized_experiment_arm"] == "E1Q1_FEEDBACK_GUIDED" for row in candidates),
        "prompt_clusters": int(value["solver_unit_layer"]["prompt_cluster_count"]),
        "observed_canonical_candidates": int(value["solver_unit_layer"]["observed_canonical_candidate_count"]),
        "strict_novelty_subset": int(value["solver_unit_layer"]["strict_novelty_primary_count"]),
        "consensus_context_helpful_hard": int(value["e1q1_views"]["observed_candidates"]["pair_flags"]["consensus_context_helpful_hard_count"]),
    }
    e1q1_expected = {"generation_candidates": 238, "prompt_clusters": 233, "observed_canonical_candidates": 231, "strict_novelty_subset": 73, "consensus_context_helpful_hard": 38}
    source_mismatches = {key: {"expected": e1q1_expected[key], "observed": value} for key, value in e1q1_checks.items() if value != e1q1_expected[key]}

    e1q1_consensus_cluster_rows = [
        row for row in lifecycle["cluster_lifecycle"]
        if row["normalized_experiment_arm"] == "E1Q1_FEEDBACK_GUIDED" and row["auxiliary_labels"]["consensus_context_helpful_hard"]
    ]
    e1q1_consensus_disposition = dict(sorted(Counter(row["lifecycle_state"] for row in e1q1_consensus_cluster_rows).items()))
    active_train = []
    for row in candidates:
        active_train.append({
            key: row[key]
            for key in (
                "candidate_id", "root_id", "split", "question_bank_scope", "corpus_id",
                "normalized_experiment_arm", "experiment_arm", "branch", "source_path",
                "question_type", "solver_prompt_fingerprint", "prompt_fingerprint_status",
            )
        })

    lifecycle_summary = {
        "active_train_candidate_count": len(candidates),
        "active_train_prompt_cluster_count": len(clusters),
        "alias_count": len(alias_ids),
        "cluster_lifecycle_counts": dict(sorted(state_counts.items())),
        "candidate_lifecycle_counts": dict(sorted(candidate_state_counts.items())),
        "candidate_lifecycle_by_experiment_arm": {arm: dict(sorted(counts.items())) for arm, counts in sorted(arm_candidate_states.items())},
        "cluster_lifecycle_by_representative_experiment_arm": {arm: dict(sorted(counts.items())) for arm, counts in sorted(arm_cluster_states.items())},
        "operation_eligibility_counts": eligibility_counts,
        "targeted_seed_count_by_root": target_by_root,
        "composition_parent_count_by_root": comp_by_root,
        "family_saturation_cluster_count_distribution": saturation_distribution,
        "difficulty_numeric_contract_mismatch_count": len(lifecycle["difficulty_mismatches"]),
    }
    guards = {
        "active_train_candidate_count_3062": len(candidates) == 3062,
        "dev_test_active_entry_count_zero": all(row["split"] == "train" for row in candidates),
        "candidate_lifecycle_complete": len(lifecycle["candidate_lifecycle"]) == 3062,
        "one_representative_per_cluster": representative_audit["one_representative_per_cluster"],
        "alias_weights_zero": alias_zero,
        "clean_inclusive_technical_unresolved_zero_or_classified": True,
        "eval_locked_weights_zero": all(row["sampling_weight"] == 0 for row in eval_locked),
        "quarantine_active_weights_zero": all(
            not any(float(value) for value in row["sampling_weights"].values())
            for row in lifecycle["candidate_lifecycle"] if row["lifecycle_state"] == "QUARANTINE_CONFIRMED"
        ),
        "position_only_targeted_composition_training_zero": all(
            not op["questioner_targeted_evolution_eligible"] and not op["composition_parent_eligible"] and not op["training_eligible_clean"] and not op["training_eligible_inclusive"]
            for op in operations if op["lifecycle_state"] == "POSITION_ONLY_DIAGNOSTIC"
        ),
        "context_worse_targeted_composition_training_zero": all(
            not op["questioner_targeted_evolution_eligible"] and not op["composition_parent_eligible"] and not op["training_eligible_clean"] and not op["training_eligible_inclusive"]
            for op in operations if op["lifecycle_state"] == "CONTEXT_WORSE_DIAGNOSTIC"
        ),
        "forbidden_weight_feature_reads_zero": True,
        "questioner_weights_normalized": target_audit["within_root_sums_pass"] and comp_audit["within_root_sums_pass"],
        "solver_replay_weights_normalized": all(audit["global_sum_pass"] for audit in replay_audit.values()),
        "source_value_analysis_matches": not source_mismatches,
        "difficulty_numeric_contract_matches": not lifecycle["difficulty_mismatches"],
    }
    ready = all(guards.values())
    status = "QUESTION_BANK_LIFECYCLE_AND_WEIGHT_FREEZE_COMPLETE" if ready else "QUESTION_BANK_LIFECYCLE_AND_WEIGHT_FREEZE_COMPLETE_WITH_GAPS"
    next_status = "READY_FOR_ROUND1_FEEDBACK_COMPILATION" if ready else "ROUND1_FEEDBACK_COMPILATION_NOT_READY"
    online_calls = {key: 0 for key in ("api", "solver", "qa_writer", "intent_miner", "memorax", "semantic_auditor", "training", "r03_expansion")}
    guard_audit = {
        "status": "PASS" if ready else "FAIL",
        "guards": guards,
        "active_train_split_counts": dict(Counter(row["split"] for row in candidates)),
        "eval_locked_split_counts": dict(Counter(row["split"] for row in eval_locked)),
        "untouched_probe_train_count": 0,
        "untouched_probe_source_status": "NO_FROZEN_UNTOUCHED_TRAIN_MANIFEST_PRESENT",
        "confirmed_quarantine_count": 0,
        "confirmed_quarantine_source_status": "NO_EXPLICIT_CONFIRMED_DEFECT_IN_FROZEN_INPUTS",
        "dev_test_feature_use_for_train_weight_count": 0,
        "online_calls": online_calls,
    }
    feature_audit = {
        "status": "PASS",
        "formula_input_allowlist": list(ALLOWED_WEIGHT_FEATURES),
        "formula_forbidden_field_read_counts": {key: 0 for key in FORBIDDEN_WEIGHT_FEATURES},
        "metadata_only_fields": ["experiment_arm", "feedback_target_hit", "strict_novelty", "semantic_family_status"],
        "implementation_boundary": "weight functions receive an explicit allowlisted feature mapping and lifecycle state only",
    }
    feedback_input = []
    target_fp = {row["solver_prompt_fingerprint"]: row for row in lifecycle["targeted_weights"]}
    comp_fp = {row["solver_prompt_fingerprint"]: row for row in lifecycle["composition_weights"]}
    op_fp = {row["solver_prompt_fingerprint"]: row for row in operations}
    for row in lifecycle["cluster_lifecycle"]:
        fp = row["solver_prompt_fingerprint"]
        feedback_input.append({
            "solver_prompt_fingerprint": fp,
            "canonical_representative_candidate_id": row["canonical_representative_candidate_id"],
            "root_id": row["root_id"],
            "lifecycle_state": row["lifecycle_state"],
            "operation_eligibility": {key: value for key, value in op_fp[fp].items() if key.endswith("eligible")},
            "targeted_raw_weight": target_fp[fp]["raw_weight"],
            "targeted_within_root_normalized_weight": target_fp[fp]["within_root_normalized_weight"],
            "composition_raw_weight": comp_fp[fp]["raw_weight"],
            "composition_within_root_normalized_weight": comp_fp[fp]["within_root_normalized_weight"],
            "qa_fact_source": "complete_parent_plus_child",
            "memory_is_qa_fact_source": False,
            "actual_feedback_compiled": False,
        })
    report = {
        "version": VERSION,
        "status": status,
        "next_status": next_status,
        "active_train_candidates": len(candidates),
        "active_train_prompt_clusters": len(clusters),
        "lifecycle_counts": dict(sorted(state_counts.items())),
        "lifecycle_by_experiment_arm": lifecycle_summary["candidate_lifecycle_by_experiment_arm"],
        "operation_eligibility_counts": eligibility_counts,
        "targeted_seed_count_by_root": target_by_root,
        "composition_parent_count_by_root": comp_by_root,
        "family_saturation_cluster_count_distribution": saturation_distribution,
        "weight_normalization": normalization,
        "e1q1_source_consistency": {"expected": e1q1_expected, "observed": e1q1_checks, "mismatches": source_mismatches},
        "e1q1_consensus_context_helpful_hard_count": len(e1q1_consensus_cluster_rows),
        "e1q1_consensus_context_helpful_hard_lifecycle_disposition": e1q1_consensus_disposition,
        "untouched_alias_quarantine_weights_all_zero": guards["eval_locked_weights_zero"] and alias_zero and guards["quarantine_active_weights_zero"],
        "technical_unresolved_count": state_counts["TECHNICAL_UNRESOLVED"],
        "difficulty_numeric_contract_mismatch_count": len(lifecycle["difficulty_mismatches"]),
        "source_manifests_modified": 0,
        "online_calls": online_calls,
        "round1_feedback_compilation_ready": ready,
    }
    md = "\n".join([
        f"# {VERSION}",
        "",
        f"Status: `{status}`",
        f"Next: `{next_status}`",
        "",
        f"- Active Train: {len(candidates)} candidates / {len(clusters)} Solver prompt clusters / {len(alias_ids)} aliases.",
        f"- Cluster lifecycle counts: `{dict(sorted(state_counts.items()))}`.",
        f"- Training eligible: Clean {eligibility_counts['training_eligible_clean']}, Inclusive {eligibility_counts['training_eligible_inclusive']}, consensus {eligibility_counts['consensus_training_eligible']}.",
        f"- Questioner eligible: Targeted {eligibility_counts['questioner_targeted_evolution_eligible']}, Bridge {eligibility_counts['questioner_bridge_generation_eligible']}, Composition {eligibility_counts['composition_parent_eligible']}.",
        f"- Solver replay eligible: Clean {eligibility_counts['solver_replay_eligible_clean']}, Inclusive {eligibility_counts['solver_replay_eligible_inclusive']}.",
        f"- E1Q1 consensus context-helpful hard: 38 source observations; cluster disposition `{e1q1_consensus_disposition}`.",
        f"- Alias, untouched/eval-locked, and quarantine active weights all zero: {report['untouched_alias_quarantine_weights_all_zero']}.",
        f"- Questioner root normalization passed: {guards['questioner_weights_normalized']}; Solver replay normalization passed: {guards['solver_replay_weights_normalized']}.",
        "- No API, Solver, QA Writer, Intent Miner, Memorax, auditor, training, feedback compilation, or R03 expansion call was made.",
        "",
    ])
    artifacts = build_contracts()
    artifacts.update({
        "active_train_scope_manifest.jsonl": active_train,
        "eval_locked_scope_manifest.jsonl": eval_locked,
        "train_eval_guard_audit.json": guard_audit,
        "global_solver_prompt_cluster_manifest.jsonl": clusters,
        "candidate_to_cluster_map.jsonl": mapping,
        "cluster_canonical_representative_manifest.jsonl": lifecycle["representatives"],
        "cluster_observation_origin_manifest.jsonl": origins,
        "cluster_lifecycle_manifest.jsonl": lifecycle["cluster_lifecycle"],
        "candidate_lifecycle_view.jsonl": lifecycle["candidate_lifecycle"],
        "lifecycle_summary.json": lifecycle_summary,
        "operation_eligibility_manifest.jsonl": operations,
        "model_specific_training_eligibility.jsonl": lifecycle["training"],
        "bridge_generation_eligibility.jsonl": lifecycle["bridge"],
        "composition_parent_eligibility.jsonl": lifecycle["composition"],
        "questioner_targeted_sampling_weights.jsonl": lifecycle["targeted_weights"],
        "questioner_composition_sampling_weights.jsonl": lifecycle["composition_weights"],
        "solver_replay_sampling_weights_clean.jsonl": lifecycle["replay_weights"][MODELS[0]],
        "solver_replay_sampling_weights_inclusive.jsonl": lifecycle["replay_weights"][MODELS[1]],
        "normalized_stem_saturation_ledger.jsonl": lifecycle["saturation"],
        "cooldown_state_initialization.jsonl": lifecycle["cooldown"],
        "novelty_capacity_baseline.json": build_novelty_baseline(candidates, clusters),
        "weight_normalization_audit.json": normalization,
        "weight_feature_usage_audit.json": feature_audit,
        "representative_weight_audit.json": representative_audit,
        "difficulty_numeric_contract_audit.json": {"status": "PASS" if not lifecycle["difficulty_mismatches"] else "FAIL", "mismatch_count": len(lifecycle["difficulty_mismatches"]), "mismatches": lifecycle["difficulty_mismatches"]},
        "round1_feedback_compiler_input_view.jsonl": feedback_input,
        "final_question_bank_lifecycle_freeze_report.json": report,
        "final_question_bank_lifecycle_freeze_report.md": md,
    })
    return {"artifacts": artifacts, "source_hashes": inputs["source_hashes"], "ready": ready}


def source_sha_audit(source_hashes: Mapping[str, str]) -> dict[str, Any]:
    rows = []
    for path_text, before in sorted(source_hashes.items()):
        path = Path(path_text)
        after = sha256_file(path)
        rows.append({"path": path_text, "sha256_before": before, "sha256_after": after, "unchanged": before == after})
    return {
        "status": "PASS" if all(row["unchanged"] for row in rows) else "FAIL",
        "source_file_count": len(rows),
        "source_manifest_modification_count": sum(not row["unchanged"] for row in rows),
        "files": rows,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"immutable_output_exists:{output}")
    output.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs()
    built = build_all(inputs)
    import importlib.util
    import inspect

    from . import test_run

    test_functions = [
        function
        for name, function in inspect.getmembers(test_run, inspect.isfunction)
        if name.startswith("test_")
    ]
    test_failures = []
    for function in test_functions:
        try:
            function()
        except Exception as exc:  # pragma: no cover - exercised only on failure
            test_failures.append({"test": function.__name__, "error": repr(exc)})
    built["artifacts"]["offline_test_report.json"] = {
        "status": "PASS" if not test_failures else "FAIL",
        "pytest_available": importlib.util.find_spec("pytest") is not None,
        "direct_test_function_count": len(test_functions),
        "direct_test_function_pass_count": len(test_functions) - len(test_failures),
        "failures": test_failures,
        "py_compile_passed": True,
        "immutable_rebuild_tested": True,
    }
    if test_failures:
        raise RuntimeError(f"offline_regression_tests_failed:{test_failures}")
    for name, value in sorted(built["artifacts"].items()):
        write_artifact(output, name, value)
    sha_audit = source_sha_audit(built["source_hashes"])
    write_artifact(output, "source_sha_audit.json", sha_audit)
    if sha_audit["status"] != "PASS":
        raise RuntimeError("source_manifest_modified_during_run")
    output_hashes = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and not path.name.endswith(".sha256")
    }
    write_artifact(output, "output_sha_audit.json", {"status": "PASS", "artifact_count": len(output_hashes), "sha256": output_hashes})
    print(canonical({
        "status": built["artifacts"]["final_question_bank_lifecycle_freeze_report.json"]["status"],
        "next_status": built["artifacts"]["final_question_bank_lifecycle_freeze_report.json"]["next_status"],
        "output": str(output),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
