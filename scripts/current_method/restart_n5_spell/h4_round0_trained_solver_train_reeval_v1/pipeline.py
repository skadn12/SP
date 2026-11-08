"""Frozen preflight, resumable Solver execution, and offline Train aggregation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..core import atomic_text, canonical_json, sha256_bytes, sha256_file
from ..h4_round0_b0_k3_difficulty_v1.difficulty import summarize_candidate_result
from ..h4_round0_b0_k3_difficulty_v1.gate import directory_identity
from ..h4_round0_b0_k3_difficulty_v1.io_utils import (
    load_json,
    load_jsonl,
    safe_name,
    stable_seed,
    verify_sha_sidecar,
    write_once_json,
    write_once_jsonl,
    write_once_text,
)
from ..h4_round0_b0_k3_difficulty_v1.request_ledger import Round0RequestLedger
from ..h4_round0_b0_k3_difficulty_v1.solver import (
    ensure_vllm_environment,
    evaluate_candidate,
    generation_config,
    load_or_recover_candidate_result,
)
from ..vllm_backends import VLLMSolverBackend
from . import (
    BASE_MODEL,
    DEV_IDENTITY_REPORT,
    DIFFICULTY_RUN,
    EXPECTED_EXPOSURE_COUNTS,
    EXPECTED_M0_FULL_CONTEXT_WORSE,
    EXPECTED_M0_FULL_ZERO,
    EXPECTED_M0_HARD,
    EXPECTED_TRAIN_CANDIDATES,
    EXPECTED_TRAIN_ROOTS,
    FORMAL_RUN,
    MAX_MODEL_LEN,
    MAX_TOKENS,
    QTYPE_REQUESTS,
    ROLLOUTS,
    RUN_BASE,
    TRAINING_RUN,
    VERSION,
)


MODEL_NAMES = ("M_clean_policy", "M_inclusive_policy")
MODEL_SCOPES = {
    "M_clean_policy": "clean_train",
    "M_inclusive_policy": "inclusive_train",
}
PARSER_BY_QTYPE = {
    "single_choice": "boxed_letter_v1",
    "multi_select": "structured_qa_box_v1",
    "ordering": "structured_qa_box_v1",
}
EXPECTED_TRAIN_REQUESTS = 58_488
BAD_REGRESSION_BUCKETS = {
    "hard_candidate", "full_zero_review", "full_context_worse", "technical_excluded",
}


def composite_directory_sha(path: Path) -> str:
    rows = []
    for child in sorted(path.rglob("*")):
        if child.is_file() and not child.name.endswith(".sha256"):
            rows.append({
                "name": child.relative_to(path).as_posix(),
                "size": child.stat().st_size,
                "sha256": sha256_file(child),
            })
    return sha256_bytes(canonical_json(rows).encode("utf-8"))


def _copy_immutable(source: Path, target: Path) -> str:
    source_sha = verify_sha_sidecar(source)
    write_once_text(target, source.read_text(encoding="utf-8"))
    write_once_text(target.with_name(target.name + ".sha256"), source_sha + "\n")
    return source_sha


def _prepare_run_root(run_root: Path) -> None:
    identity = {
        "run_identity_version": f"{VERSION}_run_identity_v1",
        "run_root": str(run_root.resolve()),
        "difficulty_run": str(DIFFICULTY_RUN.resolve()),
        "training_run": str(TRAINING_RUN.resolve()),
        "formal_run": str(FORMAL_RUN.resolve()),
        "dev_identity_report": str(DEV_IDENTITY_REPORT.resolve()),
        "scope": "round0_train_only_parent_plus_r01",
        "historical_inputs_read_only": True,
    }
    if run_root.exists() and not (run_root / "run_identity.json").is_file():
        raise RuntimeError(f"run_root_exists_without_identity:{run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    write_once_json(run_root / "run_identity.json", identity)


def _verified_sources() -> dict[str, str]:
    files = {
        "difficulty_manifest.jsonl": DIFFICULTY_RUN / "difficulty_manifest.jsonl",
        "program_gate_results.jsonl": DIFFICULTY_RUN / "program_gate_results.jsonl",
        "root_split_manifest.jsonl": DIFFICULTY_RUN / "root_split_manifest.jsonl",
        "root_split_summary.json": DIFFICULTY_RUN / "root_split_summary.json",
        "difficulty_by_split.json": DIFFICULTY_RUN / "difficulty_by_split.json",
        "solver_contract.json": DIFFICULTY_RUN / "solver_contract.json",
        "m_clean_policy_training_manifest.jsonl": TRAINING_RUN / "m_clean_policy_training_manifest.jsonl",
        "m_inclusive_policy_training_manifest.jsonl": TRAINING_RUN / "m_inclusive_policy_training_manifest.jsonl",
        "m_clean_policy_train.jsonl": TRAINING_RUN / "m_clean_policy_train.jsonl",
        "m_inclusive_policy_train.jsonl": TRAINING_RUN / "m_inclusive_policy_train.jsonl",
        "matched_clean_risk_pairs.jsonl": TRAINING_RUN / "matched_clean_risk_pairs.jsonl",
        "matched_training_preflight_report.json": TRAINING_RUN / "preflight_report.json",
        "final_dataset_report.json": TRAINING_RUN / "final_dataset_report.json",
        "final_formal_training_report.json": FORMAL_RUN / "final_formal_training_report.json",
        "clean_training_report.json": FORMAL_RUN / "clean_training_report.json",
        "inclusive_training_report.json": FORMAL_RUN / "inclusive_training_report.json",
        "source_model_identity_report.json": DEV_IDENTITY_REPORT,
    }
    return {name: verify_sha_sidecar(path) for name, path in files.items()}


def _load_train_inputs() -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    all_difficulty = load_jsonl(DIFFICULTY_RUN / "difficulty_manifest.jsonl")
    train_rows = sorted(
        (dict(row) for row in all_difficulty if row.get("split") == "train"),
        key=lambda row: (str(row["root_id"]), str(row["candidate_id"])),
    )
    train_ids = [str(row["candidate_id"]) for row in train_rows]
    train_roots = {str(row["root_id"]) for row in train_rows}
    if len(train_rows) != EXPECTED_TRAIN_CANDIDATES or len(set(train_ids)) != len(train_rows):
        raise RuntimeError(f"train_candidate_count_mismatch:{len(train_rows)}:{len(set(train_ids))}")
    if len(train_roots) != EXPECTED_TRAIN_ROOTS:
        raise RuntimeError(f"train_root_count_mismatch:{len(train_roots)}")

    bucket_counts = Counter(str(row["difficulty_bucket"]) for row in train_rows)
    expected_buckets = {
        "hard_candidate": EXPECTED_M0_HARD,
        "full_zero_review": EXPECTED_M0_FULL_ZERO,
        "full_context_worse": EXPECTED_M0_FULL_CONTEXT_WORSE,
    }
    for bucket, expected in expected_buckets.items():
        if bucket_counts[bucket] != expected:
            raise RuntimeError(f"m0_train_bucket_count_mismatch:{bucket}:{bucket_counts[bucket]}:{expected}")

    split_rows = load_jsonl(DIFFICULTY_RUN / "root_split_manifest.jsonl")
    split_by_root = {str(row["root_id"]): str(row["split"]) for row in split_rows}
    if any(split_by_root.get(root) != "train" for root in train_roots):
        raise RuntimeError("train_root_split_manifest_mismatch")
    gate_all = load_jsonl(DIFFICULTY_RUN / "program_gate_results.jsonl")
    gate_by_id = {str(row["candidate_id"]): dict(row) for row in gate_all if str(row["candidate_id"]) in set(train_ids)}
    if set(gate_by_id) != set(train_ids):
        raise RuntimeError("train_gate_candidate_identity_mismatch")

    source_inventory: dict[str, Any] = {}
    for root_id in sorted(train_roots):
        root_gates = [gate_by_id[cid] for cid in train_ids if str(gate_by_id[cid]["root_id"]) == root_id]
        if any(row.get("gate_status") != "PROGRAM_PASS" for row in root_gates):
            raise RuntimeError(f"train_non_program_pass_candidate:{root_id}")
        lineages = {
            (
                str(row["candidate_pool_path"]), str(row["candidate_pool_sha256"]),
                str(row["dialogue_path"]), str(row["dialogue_sha256"]),
            )
            for row in root_gates
        }
        if len(lineages) != 1:
            raise RuntimeError(f"root_lineage_not_unique:{root_id}")
        pool_name, pool_sha, dialogue_name, dialogue_sha = next(iter(lineages))
        pool_path, dialogue_path = Path(pool_name), Path(dialogue_name)
        if pool_path.name != "candidate_pool.json" or dialogue_path.name != "expanded_dialogue_clean.json":
            raise RuntimeError(f"round0_source_path_unexpected:{root_id}:{pool_path}:{dialogue_path}")
        if "r02" in dialogue_name.casefold():
            raise RuntimeError(f"r02_dialogue_forbidden:{root_id}")
        if sha256_file(pool_path) != pool_sha or sha256_file(dialogue_path) != dialogue_sha:
            raise RuntimeError(f"round0_source_sha_drift:{root_id}")
        pool = load_json(pool_path)
        pool_by_id = {str(row["candidate_id"]): row for row in pool}
        root_ids = {cid for cid in train_ids if str(gate_by_id[cid]["root_id"]) == root_id}
        if not root_ids <= set(pool_by_id):
            raise RuntimeError(f"candidate_pool_missing_train_candidate:{root_id}")
        dialogue = load_json(dialogue_path)
        turns = list(dialogue.get("turns") or [])
        refs = [str(row.get("turn_ref") or "") for row in turns]
        child_refs = [str(value) for value in dialogue.get("child_scene_refs") or []]
        if not refs or not any(ref.startswith("R01:") for ref in refs):
            raise RuntimeError(f"r01_dialogue_missing:{root_id}")
        if any(ref.startswith("R02:") for ref in refs) or any(not ref.startswith("R01:") for ref in child_refs):
            raise RuntimeError(f"dialogue_not_parent_plus_r01_only:{root_id}")
        if int(dialogue.get("turn_count", -1)) != len(turns):
            raise RuntimeError(f"dialogue_turn_count_mismatch:{root_id}")
        source_inventory[root_id] = {
            "candidate_pool_path": pool_name,
            "candidate_pool_sha256": pool_sha,
            "dialogue_path": dialogue_name,
            "dialogue_sha256": dialogue_sha,
            "dialogue_version": dialogue.get("expanded_dialogue_version"),
            "parent_turn_count": dialogue.get("parent_turn_count"),
            "r01_child_turn_count": dialogue.get("child_turn_count"),
            "turn_count": len(turns),
            "r02_turn_count": 0,
        }

    evidence = _load_train_evidence(train_rows)
    return train_rows, gate_by_id, evidence, source_inventory


def _load_train_evidence(train_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    matched_preflight = load_json(TRAINING_RUN / "preflight_report.json")
    stratification_run = Path(str(matched_preflight["stratification_run"]))
    expected_full_sha = str(matched_preflight["stratification_source_sha256"]["hard_evidence_status_manifest.jsonl"])
    full_sidecar = stratification_run / "hard_evidence_status_manifest.jsonl.sha256"
    if not full_sidecar.is_file() or full_sidecar.read_text(encoding="utf-8").strip() != expected_full_sha:
        raise RuntimeError("stratification_authority_sha_drift")
    files = (
        "train_strict_eligible_pool.jsonl",
        "train_broad_only_risk_pool.jsonl",
        "train_evidence_invalid_pool.jsonl",
        "train_evidence_unclassifiable_pool.jsonl",
    )
    rows: list[dict[str, Any]] = []
    for name in files:
        path = stratification_run / name
        verify_sha_sidecar(path)
        rows.extend(load_jsonl(path))
    evidence = {str(row["candidate_id"]): dict(row) for row in rows}
    m0_hard = {str(row["candidate_id"]) for row in train_rows if row["difficulty_bucket"] == "hard_candidate"}
    if len(evidence) != EXPECTED_M0_HARD or set(evidence) != m0_hard:
        raise RuntimeError(f"train_hard_evidence_identity_mismatch:{len(evidence)}:{len(m0_hard)}")
    return evidence


def build_solver_plan(train_rows: list[dict[str, Any]], contract: Mapping[str, Any]) -> dict[str, Any]:
    counts = Counter(str(row["question_type"]) for row in train_rows)
    if set(counts) != set(QTYPE_REQUESTS):
        raise RuntimeError(f"question_type_set_mismatch:{sorted(counts)}")
    formula = {}
    for qtype in sorted(QTYPE_REQUESTS):
        positions = int(contract["position_counts"][qtype])
        expected_per_candidate = positions * ROLLOUTS * 2
        if expected_per_candidate != QTYPE_REQUESTS[qtype]:
            raise RuntimeError(f"request_formula_not_closed:{qtype}")
        formula[qtype] = {
            "candidate_count": counts[qtype],
            "positions": positions,
            "rollouts_per_position": ROLLOUTS,
            "context_mode_count": 2,
            "requests_per_candidate": expected_per_candidate,
            "request_count": counts[qtype] * expected_per_candidate,
            "closed": True,
        }
    planned = sum(row["request_count"] for row in formula.values())
    if planned != EXPECTED_TRAIN_REQUESTS:
        raise RuntimeError(f"train_solver_request_count_mismatch:{planned}:{EXPECTED_TRAIN_REQUESTS}")
    return {
        "solver_plan_version": f"{VERSION}_solver_plan_v1",
        "split": "train",
        "candidate_count": len(train_rows),
        "root_count": len({str(row["root_id"]) for row in train_rows}),
        "question_type_counts": dict(sorted(counts.items())),
        "formula": formula,
        "planned_request_count_per_trained_model": planned,
        "trained_model_count": 2,
        "planned_new_request_count_total": planned * 2,
        "m0_new_request_count": 0,
        "max_model_len": int(contract["max_model_len"]),
        "rollouts_per_position": int(contract["rollouts_per_position"]),
        "position_counts": dict(contract["position_counts"]),
        "temperatures": dict(contract["temperatures"]),
        "seed_schedule": dict(contract["seed_schedule"]),
        "context_modes": list(contract["context_modes"]),
        "request_plan_closed": True,
    }


def _load_exposure(train_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    clean_rows = load_jsonl(TRAINING_RUN / "m_clean_policy_training_manifest.jsonl")
    inclusive_rows = load_jsonl(TRAINING_RUN / "m_inclusive_policy_training_manifest.jsonl")
    clean_ids = {str(row["candidate_id"]) for row in clean_rows}
    inclusive_ids = {str(row["candidate_id"]) for row in inclusive_rows}
    train_by_id = {str(row["candidate_id"]): row for row in train_rows}
    if len(clean_rows) != 386 or len(inclusive_rows) != 386 or len(clean_ids) != 386 or len(inclusive_ids) != 386:
        raise RuntimeError("training_manifest_candidate_count_mismatch")
    if not (clean_ids | inclusive_ids) <= set(train_by_id):
        raise RuntimeError("training_manifest_contains_non_train_candidate")
    rendered_sets = {}
    for policy in ("m_clean_policy", "m_inclusive_policy"):
        rendered = load_jsonl(TRAINING_RUN / f"{policy}_train.jsonl")
        ids = [str(row.get("metadata", {}).get("candidate_id")) for row in rendered]
        if len(ids) != 386 or len(set(ids)) != 386:
            raise RuntimeError(f"rendered_training_identity_invalid:{policy}")
        rendered_sets[policy] = set(ids)
    if rendered_sets["m_clean_policy"] != clean_ids or rendered_sets["m_inclusive_policy"] != inclusive_ids:
        raise RuntimeError("rendered_and_manifest_training_ids_differ")

    shared, clean_only, inclusive_only = clean_ids & inclusive_ids, clean_ids - inclusive_ids, inclusive_ids - clean_ids
    observed = {
        "SHARED_TRAINED": len(shared),
        "CLEAN_ONLY_TRAINED": len(clean_only),
        "INCLUSIVE_ONLY_TRAINED": len(inclusive_only),
        "UNEXPOSED_TO_BOTH": len(set(train_by_id) - (clean_ids | inclusive_ids)),
    }
    if observed != EXPECTED_EXPOSURE_COUNTS:
        raise RuntimeError(f"training_exposure_count_mismatch:{observed}:{EXPECTED_EXPOSURE_COUNTS}")

    pairs = load_jsonl(TRAINING_RUN / "matched_clean_risk_pairs.jsonl")
    if len(pairs) != 77 or {str(row["clean_candidate_id"]) for row in pairs} != clean_only or {str(row["risk_candidate_id"]) for row in pairs} != inclusive_only:
        raise RuntimeError("matched_pair_identity_mismatch")
    pair_by_candidate: dict[str, dict[str, Any]] = {}
    for row in pairs:
        pair_by_candidate[str(row["clean_candidate_id"])] = {"pair_id": row["pair_id"], "pair_role": "clean_control"}
        pair_by_candidate[str(row["risk_candidate_id"])] = {"pair_id": row["pair_id"], "pair_role": "risk_replacement"}

    clean_meta = {str(row["candidate_id"]): row for row in clean_rows}
    inclusive_meta = {str(row["candidate_id"]): row for row in inclusive_rows}
    rows = []
    for cid, source in sorted(train_by_id.items()):
        exposure = (
            "SHARED_TRAINED" if cid in shared
            else "CLEAN_ONLY_TRAINED" if cid in clean_only
            else "INCLUSIVE_ONLY_TRAINED" if cid in inclusive_only
            else "UNEXPOSED_TO_BOTH"
        )
        rows.append({
            "exposure_manifest_version": f"{VERSION}_exposure_v1",
            "candidate_id": cid,
            "root_id": source["root_id"],
            "question_type": source["question_type"],
            "m0_difficulty_bucket": source["difficulty_bucket"],
            "training_exposure": exposure,
            "directly_trained_by_clean": cid in clean_ids,
            "directly_trained_by_inclusive": cid in inclusive_ids,
            "clean_selection_role": clean_meta.get(cid, {}).get("selection_role"),
            "inclusive_selection_role": inclusive_meta.get(cid, {}).get("selection_role"),
            **pair_by_candidate.get(cid, {"pair_id": None, "pair_role": None}),
        })
    return rows, {str(row["candidate_id"]): row for row in rows}, pairs


def _model_identities(contract_sha: str, source_report_sha: str) -> dict[str, Any]:
    source = load_json(DEV_IDENTITY_REPORT)
    if source.get("solver_contract_sha256") != contract_sha:
        raise RuntimeError("identity_report_solver_contract_mismatch")
    base_identity = directory_identity(BASE_MODEL)
    if base_identity != source.get("base_model_identity"):
        raise RuntimeError("base_model_identity_drift")
    final_training = load_json(FORMAL_RUN / "final_formal_training_report.json")
    if final_training.get("status") != "FORMAL_TRAINING_COMPLETE":
        raise RuntimeError("formal_training_not_complete")

    output: dict[str, Any] = {}
    for name in MODEL_NAMES:
        frozen = dict(source["models"][name])
        if frozen.get("adapter_enabled") is not True or frozen.get("adapter_name") != name:
            raise RuntimeError(f"adapter_not_enabled_in_identity_report:{name}")
        adapter_path = Path(str(frozen["adapter_path"]))
        merged_path = Path(str(frozen["merged_model_path"]))
        if not adapter_path.is_relative_to(FORMAL_RUN) or not merged_path.is_dir():
            raise RuntimeError(f"model_path_not_frozen:{name}")
        config_sha = sha256_file(adapter_path / "adapter_config.json")
        state_sha = sha256_file(adapter_path / "adapter_model.pt")
        composite_sha = composite_directory_sha(adapter_path)
        expected = (
            str(frozen["adapter_config_sha256"]), str(frozen["adapter_state_sha256"]),
            str(frozen["adapter_composite_sha256"]),
        )
        if (config_sha, state_sha, composite_sha) != expected:
            raise RuntimeError(f"adapter_identity_drift:{name}")
        merged_identity = directory_identity(merged_path)
        if merged_identity != frozen.get("merged_model_identity"):
            raise RuntimeError(f"merged_model_identity_drift:{name}")
        training_report = load_json(
            FORMAL_RUN / ("clean_training_report.json" if name == "M_clean_policy" else "inclusive_training_report.json")
        )
        if training_report.get("status") != "PASS" or training_report.get("adapter_saved") is not True:
            raise RuntimeError(f"training_report_not_pass:{name}")
        fingerprint_payload = {
            "model_name": name,
            "base_model_identity": base_identity,
            "adapter_path": str(adapter_path),
            "adapter_config_sha256": config_sha,
            "adapter_state_sha256": state_sha,
            "adapter_composite_sha256": composite_sha,
            "merged_model_identity": merged_identity,
            "solver_contract_sha256": contract_sha,
        }
        output[name] = {
            **frozen,
            "inference_model_path": str(merged_path),
            "identity_source_report": str(DEV_IDENTITY_REPORT),
            "identity_source_report_sha256": source_report_sha,
            "model_fingerprint_sha256": sha256_bytes(canonical_json(fingerprint_payload).encode("utf-8")),
            "identity_reverified": True,
        }
    return {
        "model_identity_report_version": f"{VERSION}_model_identity_v1",
        "identity_source_report": str(DEV_IDENTITY_REPORT),
        "identity_source_report_sha256": source_report_sha,
        "solver_contract_sha256": contract_sha,
        "base_model_identity": base_identity,
        "models": output,
    }


def _expected_request_keys(qtype: str) -> set[tuple[str, str, int]]:
    position_count = 5 if qtype == "multi_select" else 4
    return {
        (f"shift_{position}", context_mode, slot_id)
        for position in range(position_count)
        for context_mode in ("no_context", "full_context")
        for slot_id in range(ROLLOUTS)
    }


def _audit_m0_reuse(
    train_rows: list[dict[str, Any]],
    gates: Mapping[str, Mapping[str, Any]],
    contract: Mapping[str, Any],
    contract_sha: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    result_sha256: dict[str, str] = {}
    checked_requests = 0
    checked_ledger_states = 0
    checked_prompt_files: set[str] = set()
    checked_envelopes: set[str] = set()

    def mismatch(value: str) -> None:
        if len(reasons) < 200:
            reasons.append(value)

    for source in train_rows:
        cid, root_id, qtype = str(source["candidate_id"]), str(source["root_id"]), str(source["question_type"])
        result_path = DIFFICULTY_RUN / "candidate_results" / f"{sha256_bytes(cid.encode('utf-8'))}.json"
        try:
            result_sha256[cid] = verify_sha_sidecar(result_path)
            result = load_json(result_path)
        except Exception as error:
            mismatch(f"candidate_result:{cid}:{type(error).__name__}:{error}")
            continue
        gate = gates[cid]
        expected_count = QTYPE_REQUESTS[qtype]
        if any((
            result.get("candidate_id") != cid,
            result.get("root_id") != root_id,
            result.get("question_type") != qtype,
            result.get("solver_contract_sha256") != contract_sha,
            result.get("candidate_pool_sha256") != gate["candidate_pool_sha256"],
            result.get("dialogue_sha256") != gate["dialogue_sha256"],
            int(result.get("request_count", -1)) != expected_count,
            int(result.get("technical_failure_count", -1)) != 0,
        )):
            mismatch(f"candidate_result_identity:{cid}")
        requests = list(result.get("requests") or [])
        observed_keys = {
            (str(row.get("variant_id")), str(row.get("context_mode")), int(row.get("slot_id", -1)))
            for row in requests
        }
        if len(requests) != expected_count or observed_keys != _expected_request_keys(qtype):
            mismatch(f"candidate_request_set:{cid}")
        gate_prompts = {
            (str(row["variant_id"]), str(row["context_mode"])): row
            for row in gate.get("prompt_audit") or []
        }
        for request in requests:
            checked_requests += 1
            variant = str(request.get("variant_id"))
            context_mode = str(request.get("context_mode"))
            slot_id = int(request.get("slot_id", -1))
            expected_seed = stable_seed(root_id, cid, slot_id)
            expected_prompt = gate_prompts.get((variant, context_mode))
            if (
                request.get("request_id") != f"{cid}__{variant}__{context_mode}__s{slot_id:02d}"
                or int(request.get("seed", -1)) != expected_seed
                or not expected_prompt
                or request.get("prompt_sha256") != expected_prompt.get("prompt_sha256")
                or int(request.get("prompt_token_count", -1)) != int(expected_prompt.get("prompt_token_count", -2))
                or request.get("parser_version") != PARSER_BY_QTYPE[qtype]
                or request.get("scored") is not True
            ):
                mismatch(f"candidate_request_identity:{cid}:{variant}:{context_mode}:{slot_id}")
            metadata = dict(request.get("backend_metadata") or {})
            expected_temperature = float(contract["temperatures"][qtype])
            if (
                float(metadata.get("temperature", -1)) != expected_temperature
                or float(metadata.get("top_p", -1)) != float(contract["generation_config"]["top_p"])
                or int(metadata.get("max_tokens", -1)) != int(contract["generation_config"]["max_tokens"])
                or str(metadata.get("finish_reason") or "").casefold() == "length"
            ):
                mismatch(f"candidate_generation_config:{cid}:{variant}:{context_mode}:{slot_id}")

            cache_key = str(request.get("request_cache_key") or "")
            state_path = DIFFICULTY_RUN / "request_ledger" / safe_name(root_id) / f"{cache_key}.json"
            if not state_path.is_file():
                mismatch(f"m0_ledger_missing:{cid}:{cache_key}")
                continue
            state = load_json(state_path)
            checked_ledger_states += 1
            frozen = dict(state.get("request_contract") or {})
            backend = dict(frozen.get("backend_identity") or {})
            expected_generation = generation_config(qtype)
            if (
                state.get("status") != "completed"
                or frozen.get("solver_contract_sha256") != contract_sha
                or frozen.get("candidate_id") != cid
                or frozen.get("root_id") != root_id
                or frozen.get("variant_id") != variant
                or frozen.get("context_mode") != context_mode
                or int(frozen.get("slot_id", -1)) != slot_id
                or int(frozen.get("seed", -1)) != expected_seed
                or frozen.get("prompt_sha256") != request.get("prompt_sha256")
                or frozen.get("candidate_pool_sha256") != gate["candidate_pool_sha256"]
                or frozen.get("dialogue_sha256") != gate["dialogue_sha256"]
                or frozen.get("parser_version") != PARSER_BY_QTYPE[qtype]
                or frozen.get("generation_config") != expected_generation
                or Path(str(backend.get("model", ""))).resolve() != BASE_MODEL.resolve()
                or Path(str(backend.get("tokenizer", ""))).resolve() != Path(str(contract["tokenizer"])).resolve()
                or int(backend.get("max_model_len", -1)) != MAX_MODEL_LEN
            ):
                mismatch(f"m0_ledger_contract:{cid}:{cache_key}")
            decision = dict(state.get("decision") or {})
            if any(decision.get(key) != request.get(key) for key in ("parser_version", "parse_status", "scored", "correct", "canonical_prediction", "failure_type")):
                mismatch(f"m0_ledger_decision:{cid}:{cache_key}")
            prompt_path = Path(str(state.get("prompt_path") or ""))
            prompt_name = str(prompt_path)
            if prompt_name not in checked_prompt_files:
                if not prompt_path.is_file() or sha256_file(prompt_path) != request.get("prompt_sha256"):
                    mismatch(f"m0_raw_prompt:{cid}:{cache_key}")
                checked_prompt_files.add(prompt_name)
            envelope_path = Path(str(state.get("response_envelope_path") or ""))
            envelope_name = str(envelope_path)
            if envelope_name not in checked_envelopes:
                if not envelope_path.is_file() or sha256_file(envelope_path) != state.get("response_envelope_sha256"):
                    mismatch(f"m0_response_envelope:{cid}:{cache_key}")
                else:
                    envelope = load_json(envelope_path)
                    if (
                        envelope.get("request_cache_key") != cache_key
                        or sha256_bytes(str(envelope.get("raw_text") or "").encode("utf-8")) != envelope.get("raw_response_sha256")
                    ):
                        mismatch(f"m0_response_envelope_content:{cid}:{cache_key}")
                checked_envelopes.add(envelope_name)

        summary = summarize_candidate_result(result)
        compare_fields = (
            "difficulty_bucket", "full_context_correct_count", "full_context_total_count",
            "no_context_correct_count", "no_context_total_count", "position_mechanism",
        )
        if any(summary.get(key) != source.get(key) for key in compare_fields):
            mismatch(f"m0_summary_drift:{cid}")

    allowed = len(result_sha256) == EXPECTED_TRAIN_CANDIDATES and checked_requests == EXPECTED_TRAIN_REQUESTS and not reasons
    return {
        "m0_reuse_audit_version": f"{VERSION}_m0_reuse_v1",
        "status": "PASS" if allowed else "FAIL",
        "reuse_allowed": allowed,
        "source_run": str(DIFFICULTY_RUN),
        "scope": "train_only",
        "candidate_count": len(result_sha256),
        "checked_request_count": checked_requests,
        "checked_ledger_state_count": checked_ledger_states,
        "checked_unique_prompt_file_count": len(checked_prompt_files),
        "checked_response_envelope_count": len(checked_envelopes),
        "expected_candidate_count": EXPECTED_TRAIN_CANDIDATES,
        "expected_request_count": EXPECTED_TRAIN_REQUESTS,
        "m0_new_solver_generation": 0 if allowed else None,
        "solver_contract_sha256": contract_sha,
        "result_sha256": result_sha256,
        "mismatch_reasons": reasons,
    }


def _search_existing_train_results(run_root: Path, identities: Mapping[str, Any], contract_sha: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for model_name in MODEL_NAMES:
        scope_name = MODEL_SCOPES[model_name]
        matches = []
        for report_path in sorted(RUN_BASE.glob(f"*/{scope_name}/model_execution_report.json")):
            if run_root in report_path.parents:
                continue
            try:
                report = load_json(report_path)
            except Exception:
                continue
            if (
                report.get("model_fingerprint_sha256") == identities["models"][model_name]["model_fingerprint_sha256"]
                and report.get("solver_contract_sha256") == contract_sha
                and int(report.get("candidate_count", -1)) == EXPECTED_TRAIN_CANDIDATES
                and int(report.get("request_count", -1)) == EXPECTED_TRAIN_REQUESTS
            ):
                matches.append(str(report_path.parent))
        output[model_name] = {
            "searched_scope_name": scope_name,
            "fingerprint_exact_match_scopes": matches,
            "reuse_action": "none_found" if not matches else "manual_import_required_before_generation",
        }
    if any(row["fingerprint_exact_match_scopes"] for row in output.values()):
        raise RuntimeError(f"existing_exact_train_result_found:{output}")
    return output


def run_preflight(run_root: Path) -> dict[str, Any]:
    _prepare_run_root(run_root)
    source_sha = _verified_sources()
    contract_source = DIFFICULTY_RUN / "solver_contract.json"
    contract_sha = _copy_immutable(contract_source, run_root / "solver_contract.json")
    contract = load_json(contract_source)
    if (
        int(contract.get("max_model_len", -1)) != MAX_MODEL_LEN
        or int(contract.get("rollouts_per_position", -1)) != ROLLOUTS
        or int(contract.get("generation_config", {}).get("max_tokens", -1)) != MAX_TOKENS
        or contract.get("context_modes") != ["no_context", "full_context"]
    ):
        raise RuntimeError("frozen_solver_contract_mismatch")
    train_rows, gates, evidence, source_inventory = _load_train_inputs()
    plan = build_solver_plan(train_rows, contract)
    write_once_json(run_root / "solver_plan.json", plan)
    exposure_rows, _, pairs = _load_exposure(train_rows)
    write_once_jsonl(run_root / "candidate_training_exposure_manifest.jsonl", exposure_rows)
    identities = _model_identities(contract_sha, source_sha["source_model_identity_report.json"])
    reuse_search = _search_existing_train_results(run_root, identities, contract_sha)
    identities["existing_train_result_search"] = reuse_search
    write_once_json(run_root / "model_identity_report.json", identities)
    m0_reuse = _audit_m0_reuse(train_rows, gates, contract, contract_sha)
    write_once_json(run_root / "m0_reuse_audit.json", m0_reuse)
    if not m0_reuse["reuse_allowed"]:
        raise RuntimeError("m0_reuse_audit_failed")
    preflight = {
        "preflight_report_version": f"{VERSION}_preflight_v1",
        "status": "PASS",
        "scope": "train_only",
        "train_candidate_count": len(train_rows),
        "train_root_count": len({str(row["root_id"]) for row in train_rows}),
        "m0_hard_count": sum(row["difficulty_bucket"] == "hard_candidate" for row in train_rows),
        "m0_full_zero_count": sum(row["difficulty_bucket"] == "full_zero_review" for row in train_rows),
        "m0_full_context_worse_count": sum(row["difficulty_bucket"] == "full_context_worse" for row in train_rows),
        "train_hard_evidence_status_count": len(evidence),
        "training_exposure_counts": dict(sorted(Counter(row["training_exposure"] for row in exposure_rows).items())),
        "matched_pair_count": len(pairs),
        "source_sha256": source_sha,
        "source_root_inventory": source_inventory,
        "parent_plus_r01_only": all(row["r02_turn_count"] == 0 for row in source_inventory.values()),
        "r02_accessed": False,
        "dev_candidate_results_accessed": False,
        "test_candidate_results_accessed": False,
        "solver_contract_sha256": contract_sha,
        "max_model_len": contract["max_model_len"],
        "planned_request_count_per_trained_model": plan["planned_request_count_per_trained_model"],
        "m0_reuse_allowed": True,
        "m0_new_solver_generation": 0,
        "adapter_and_merged_model_identity_reverified": True,
        "existing_train_result_search": reuse_search,
        "qa_writer_calls": 0,
        "semantic_auditor_calls": 0,
        "training_started": False,
    }
    write_once_json(run_root / "preflight_report.json", preflight)
    return preflight


def _candidate_row(
    result: Mapping[str, Any],
    metadata: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
    gate: Mapping[str, Any],
    model_name: str,
    fingerprint: str,
) -> dict[str, Any]:
    cid = str(result["candidate_id"])
    summary = summarize_candidate_result(result)
    requests = list(result.get("requests") or [])
    technical = sum(not bool(row.get("scored")) for row in requests)
    truncated = sum(row.get("parse_status") == "output_truncated" for row in requests)
    parse_failures = sum(
        not bool(row.get("scored")) and row.get("failure_type") not in {"solver_error", "solver_output_truncated"}
        for row in requests
    )
    return {
        "reevaluation_manifest_version": f"{VERSION}_difficulty_v1",
        "model_name": model_name,
        "model_fingerprint_sha256": fingerprint,
        "candidate_id": cid,
        "root_id": result["root_id"],
        "split": "train",
        "question_type": result["question_type"],
        "generation_source": metadata.get("generation_source"),
        "m0_position_mechanism": metadata.get("position_mechanism"),
        "evidence_status": evidence.get(cid, {}).get("evidence_status"),
        "request_count": result["request_count"],
        "technical_failure_count": technical,
        "truncated_request_count": truncated,
        "parse_failure_count": parse_failures,
        "candidate_pool_path": gate["candidate_pool_path"],
        "candidate_pool_sha256": result["candidate_pool_sha256"],
        "dialogue_path": gate["dialogue_path"],
        "dialogue_sha256": result["dialogue_sha256"],
        "solver_contract_sha256": result["solver_contract_sha256"],
        **summary,
    }


def _assert_model_unchanged(identity: Mapping[str, Any]) -> Path:
    adapter_path = Path(str(identity["adapter_path"]))
    merged_path = Path(str(identity["inference_model_path"]))
    observed_adapter = (
        sha256_file(adapter_path / "adapter_config.json"),
        sha256_file(adapter_path / "adapter_model.pt"),
        composite_directory_sha(adapter_path),
    )
    expected_adapter = (
        str(identity["adapter_config_sha256"]),
        str(identity["adapter_state_sha256"]),
        str(identity["adapter_composite_sha256"]),
    )
    if observed_adapter != expected_adapter:
        raise RuntimeError(f"adapter_drift_before_solver:{identity['model_name']}")
    if directory_identity(merged_path) != identity["merged_model_identity"]:
        raise RuntimeError(f"merged_model_drift_before_solver:{identity['model_name']}")
    return merged_path


def _validate_model_result(
    result: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    gate: Mapping[str, Any],
    ledger: Round0RequestLedger,
    contract_sha: str,
    model_identifier: str,
) -> None:
    cid, root_id, qtype = str(source["candidate_id"]), str(source["root_id"]), str(source["question_type"])
    expected_count = QTYPE_REQUESTS[qtype]
    if any((
        result.get("candidate_id") != cid,
        result.get("root_id") != root_id,
        result.get("question_type") != qtype,
        result.get("solver_contract_sha256") != contract_sha,
        result.get("candidate_pool_sha256") != gate["candidate_pool_sha256"],
        result.get("dialogue_sha256") != gate["dialogue_sha256"],
        int(result.get("request_count", -1)) != expected_count,
    )):
        raise RuntimeError(f"trained_candidate_result_identity_mismatch:{cid}")
    requests = list(result.get("requests") or [])
    keys = {
        (str(row.get("variant_id")), str(row.get("context_mode")), int(row.get("slot_id", -1)))
        for row in requests
    }
    if len(requests) != expected_count or keys != _expected_request_keys(qtype):
        raise RuntimeError(f"trained_candidate_request_set_mismatch:{cid}")
    for request in requests:
        state = ledger.read(root_id, str(request["request_cache_key"]))
        frozen = dict((state or {}).get("request_contract") or {})
        backend = dict(frozen.get("backend_identity") or {})
        if (
            not state
            or state.get("status") not in {"completed", "failed"}
            or frozen.get("solver_contract_sha256") != contract_sha
            or frozen.get("candidate_id") != cid
            or frozen.get("prompt_sha256") != request.get("prompt_sha256")
            or backend.get("model") != model_identifier
            or frozen.get("generation_config") != generation_config(qtype)
        ):
            raise RuntimeError(f"trained_request_ledger_identity_mismatch:{cid}:{request.get('request_cache_key')}")
        if state.get("status") == "completed":
            decision = dict(state.get("decision") or {})
            if any(decision.get(key) != request.get(key) for key in ("parser_version", "parse_status", "scored", "correct", "canonical_prediction", "failure_type")):
                raise RuntimeError(f"trained_request_decision_mismatch:{cid}:{request.get('request_cache_key')}")


def run_trained_model(run_root: Path, model_name: str) -> dict[str, Any]:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"unsupported_model:{model_name}")
    preflight = load_json(run_root / "preflight_report.json")
    if preflight.get("status") != "PASS":
        raise RuntimeError("preflight_not_pass")
    contract_path = run_root / "solver_contract.json"
    contract_sha = verify_sha_sidecar(contract_path)
    contract = load_json(contract_path)
    plan = load_json(run_root / "solver_plan.json")
    if not plan.get("request_plan_closed") or int(plan["planned_request_count_per_trained_model"]) != EXPECTED_TRAIN_REQUESTS:
        raise RuntimeError("solver_plan_not_closed_before_execution")
    identities = load_json(run_root / "model_identity_report.json")
    identity = dict(identities["models"][model_name])
    merged_path = _assert_model_unchanged(identity)
    fingerprint = str(identity["model_fingerprint_sha256"])
    model_identifier = f"{merged_path.resolve()}#model_fingerprint={fingerprint}"

    for qtype in QTYPE_REQUESTS:
        runtime = generation_config(qtype)
        if (
            float(runtime["temperature"]) != float(contract["temperatures"][qtype])
            or float(runtime["top_p"]) != float(contract["generation_config"]["top_p"])
            or int(runtime["max_tokens"]) != int(contract["generation_config"]["max_tokens"])
            or runtime["enable_thinking"] is not False
        ):
            raise RuntimeError(f"solver_runtime_contract_drift:{qtype}")

    train_rows, gates, evidence, _ = _load_train_inputs()
    train_by_id = {str(row["candidate_id"]): row for row in train_rows}
    scope = run_root / MODEL_SCOPES[model_name]
    for name in ("candidate_results", "request_ledger", "raw_prompts", "raw_responses", "logs"):
        (scope / name).mkdir(parents=True, exist_ok=True)
    ledger = Round0RequestLedger(scope, solver_contract_sha256=contract_sha)
    recovery = ledger.recover_orphaned_requests()

    result_paths = {
        cid: scope / "candidate_results" / f"{sha256_bytes(cid.encode('utf-8'))}.json"
        for cid in train_by_id
    }
    missing = [cid for cid, path in result_paths.items() if not path.is_file() or not path.with_name(path.name + ".sha256").is_file()]
    tokenizer = None
    backend = None
    sources: dict[str, tuple[dict[str, dict[str, Any]], dict[str, Any]]] = {}
    completed: list[dict[str, Any]] = []
    reused_candidate_count = 0
    new_candidate_count = 0
    try:
        if missing:
            from transformers import AutoTokenizer

            ensure_vllm_environment()
            tokenizer = AutoTokenizer.from_pretrained(
                contract["tokenizer"], trust_remote_code=True, local_files_only=True,
            )
            backend = VLLMSolverBackend(
                model_path=merged_path,
                tokenizer_path=Path(str(contract["tokenizer"])),
                max_model_len=MAX_MODEL_LEN,
                temperature=float(contract["temperatures"]["single_choice"]),
                top_p=float(contract["generation_config"]["top_p"]),
                max_tokens=MAX_TOKENS,
                gpu_memory_utilization=0.80,
            )
            backend.model_identifier = model_identifier
        for index, source in enumerate(train_rows, start=1):
            cid, root_id = str(source["candidate_id"]), str(source["root_id"])
            gate = gates[cid]
            if root_id not in sources:
                pool_path, dialogue_path = Path(str(gate["candidate_pool_path"])), Path(str(gate["dialogue_path"]))
                if sha256_file(pool_path) != gate["candidate_pool_sha256"] or sha256_file(dialogue_path) != gate["dialogue_sha256"]:
                    raise RuntimeError(f"solver_source_sha_drift:{root_id}")
                pool = load_json(pool_path)
                sources[root_id] = ({str(row["candidate_id"]): row for row in pool}, load_json(dialogue_path))
            candidate, expanded = sources[root_id][0][cid], sources[root_id][1]
            result_path = result_paths[cid]
            was_complete = result_path.is_file() and result_path.with_name(result_path.name + ".sha256").is_file()
            if was_complete:
                verify_sha_sidecar(result_path)
                result = load_json(result_path)
                reused_candidate_count += 1
            else:
                if backend is None or tokenizer is None:
                    raise RuntimeError("solver_backend_missing_for_incomplete_candidate")
                result = load_or_recover_candidate_result(
                    run_root=scope,
                    candidate=candidate,
                    expanded=expanded,
                    gate_row=gate,
                    tokenizer=tokenizer,
                    backend=backend,
                    ledger=ledger,
                    solver_contract_sha256=contract_sha,
                )
                if result is None:
                    result = evaluate_candidate(
                        candidate=candidate,
                        expanded=expanded,
                        gate_row=gate,
                        tokenizer=tokenizer,
                        backend=backend,
                        ledger=ledger,
                        solver_contract_sha256=contract_sha,
                    )
                    write_once_json(result_path, result)
                new_candidate_count += 1
            _validate_model_result(
                result,
                source=source,
                gate=gate,
                ledger=ledger,
                contract_sha=contract_sha,
                model_identifier=model_identifier,
            )
            completed.append(dict(result))
            if index % 10 == 0 or index == len(train_rows):
                progress = {
                    "model_name": model_name,
                    "model_fingerprint_sha256": fingerprint,
                    "completed_candidate_count": index,
                    "total_candidate_count": len(train_rows),
                    "observed_request_count": sum(int(row.get("request_count", 0)) for row in completed),
                    "technical_failure_count": sum(int(row.get("technical_failure_count", 0)) for row in completed),
                    "reused_candidate_count": reused_candidate_count,
                    "new_or_recovered_candidate_count": new_candidate_count,
                    "last_candidate_id": cid,
                }
                atomic_text(scope / "run_progress.json", json.dumps(progress, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
                if index % 50 == 0 or index == len(train_rows):
                    print(json.dumps(progress, ensure_ascii=False), flush=True)
    finally:
        if backend is not None:
            backend.close()

    candidate_rows = [
        _candidate_row(
            result,
            train_by_id[str(result["candidate_id"])],
            evidence,
            gates[str(result["candidate_id"])],
            model_name,
            fingerprint,
        )
        for result in completed
    ]
    request_count = sum(int(row["request_count"]) for row in completed)
    technical_count = sum(int(row["technical_failure_count"]) for row in completed)
    truncated_count = sum(int(row["truncated_request_count"]) for row in candidate_rows)
    parse_failure_count = sum(int(row["parse_failure_count"]) for row in candidate_rows)
    if len(completed) != EXPECTED_TRAIN_CANDIDATES or request_count != EXPECTED_TRAIN_REQUESTS:
        raise RuntimeError(f"trained_model_execution_not_closed:{model_name}:{len(completed)}:{request_count}")
    write_once_jsonl(scope / "candidate_results.jsonl", completed)
    write_once_jsonl(scope / "difficulty_manifest.jsonl", candidate_rows)
    report = {
        "model_execution_report_version": f"{VERSION}_{model_name}_execution_v1",
        "status": "PASS",
        "model_name": model_name,
        "model_fingerprint_sha256": fingerprint,
        "inference_model_path": str(merged_path),
        "solver_model_identifier": model_identifier,
        "solver_contract_sha256": contract_sha,
        "candidate_count": len(completed),
        "request_count": request_count,
        "expected_request_count": EXPECTED_TRAIN_REQUESTS,
        "technical_failure_count": technical_count,
        "truncated_request_count": truncated_count,
        "parse_failure_count": parse_failure_count,
        "technical_candidates_excluded_from_accuracy": sum(row["difficulty_bucket"] == "technical_excluded" for row in candidate_rows),
        "reused_candidate_count_at_phase_start": reused_candidate_count,
        "new_or_recovered_candidate_count": new_candidate_count,
        "orphan_recovery": recovery,
        "ledger_summary": ledger.summary(),
        "max_model_len": MAX_MODEL_LEN,
        "scope": str(scope),
    }
    write_once_json(scope / "model_execution_report.json", report)
    return report


def _mean(values: Iterable[float | None]) -> float | None:
    selected = [float(value) for value in values if value is not None]
    return sum(selected) / len(selected) if selected else None


def aggregate_model(rows: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    valid = [row for row in rows if row.get("full_context_accuracy") is not None and row.get("no_context_accuracy") is not None]
    roots: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        roots[str(row["root_id"])].append(row)

    candidate_full = _mean(row["full_context_accuracy"] for row in valid)
    candidate_no = _mean(row["no_context_accuracy"] for row in valid)
    root_full = _mean(_mean(row["full_context_accuracy"] for row in values) for values in roots.values())
    root_no = _mean(_mean(row["no_context_accuracy"] for row in values) for values in roots.values())
    full_correct = sum(int(row["full_context_correct_count"]) for row in valid)
    full_total = sum(int(row["full_context_total_count"]) for row in valid)
    no_correct = sum(int(row["no_context_correct_count"]) for row in valid)
    no_total = sum(int(row["no_context_total_count"]) for row in valid)

    def position_extreme(row: Mapping[str, Any], context: str, fn: Any) -> float | None:
        values = list((row.get("position_diagnostics", {}).get(context, {}).get("position_accuracy") or {}).values())
        return fn(values) if values else None

    return {
        "model_name": model_name,
        "candidate_count": len(rows),
        "scored_candidate_count": len(valid),
        "technical_excluded_candidate_count": len(rows) - len(valid),
        "technical_failure_request_count": sum(int(row.get("technical_failure_count", 0)) for row in rows),
        "candidate_weighted": {
            "full_context_accuracy": candidate_full,
            "no_context_accuracy": candidate_no,
            "context_gain": candidate_full - candidate_no if candidate_full is not None and candidate_no is not None else None,
        },
        "root_equal": {
            "root_count": len(roots),
            "full_context_accuracy": root_full,
            "no_context_accuracy": root_no,
            "context_gain": root_full - root_no if root_full is not None and root_no is not None else None,
        },
        "rollout_weighted": {
            "full_context_correct_count": full_correct,
            "full_context_total_count": full_total,
            "full_context_accuracy": full_correct / full_total if full_total else None,
            "no_context_correct_count": no_correct,
            "no_context_total_count": no_total,
            "no_context_accuracy": no_correct / no_total if no_total else None,
            "context_gain": (full_correct / full_total) - (no_correct / no_total) if full_total and no_total else None,
        },
        "best_position_accuracy_mean": {
            context: _mean(position_extreme(row, context, max) for row in valid)
            for context in ("full_context", "no_context")
        },
        "worst_position_accuracy_mean": {
            context: _mean(position_extreme(row, context, min) for row in valid)
            for context in ("full_context", "no_context")
        },
        "canonical_flip_rate_mean": {
            context: _mean(row.get("canonical_flip_rate", {}).get(context) for row in valid)
            for context in ("full_context", "no_context")
        },
        "difficulty_bucket_counts": dict(sorted(Counter(str(row["difficulty_bucket"]) for row in rows).items())),
    }


def _m0_rows(
    train_rows: list[dict[str, Any]],
    gates: Mapping[str, Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for source in train_rows:
        cid = str(source["candidate_id"])
        path = DIFFICULTY_RUN / "candidate_results" / f"{sha256_bytes(cid.encode('utf-8'))}.json"
        verify_sha_sidecar(path)
        result = load_json(path)
        output.append(_candidate_row(
            result, source, evidence, gates[cid], "M0", "frozen_m0_base_model",
        ))
    return output


def _model_state(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "difficulty_bucket": row.get("difficulty_bucket"),
        "full_context_correct_count": row.get("full_context_correct_count"),
        "full_context_total_count": row.get("full_context_total_count"),
        "full_context_accuracy": row.get("full_context_accuracy"),
        "no_context_correct_count": row.get("no_context_correct_count"),
        "no_context_total_count": row.get("no_context_total_count"),
        "no_context_accuracy": row.get("no_context_accuracy"),
        "context_gain": (
            float(row["full_context_accuracy"]) - float(row["no_context_accuracy"])
            if row.get("full_context_accuracy") is not None and row.get("no_context_accuracy") is not None
            else None
        ),
        "position_mechanism": row.get("position_mechanism"),
        "canonical_flip_rate": row.get("canonical_flip_rate"),
        "technical_failure_count": int(row.get("technical_failure_count", 0)),
        "truncated_request_count": int(row.get("truncated_request_count", 0)),
        "parse_failure_count": int(row.get("parse_failure_count", 0)),
    }


def _technical(state: Mapping[str, Any]) -> bool:
    return state.get("difficulty_bucket") == "technical_excluded" or int(state.get("technical_failure_count", 0)) > 0


def _is_full(state: Mapping[str, Any]) -> bool:
    total = state.get("full_context_total_count")
    return not _technical(state) and total is not None and int(state["full_context_correct_count"]) == int(total)


def _is_zero(state: Mapping[str, Any]) -> bool:
    return not _technical(state) and state.get("full_context_correct_count") is not None and int(state["full_context_correct_count"]) == 0


def _is_partial(state: Mapping[str, Any]) -> bool:
    if _technical(state) or state.get("full_context_total_count") is None:
        return False
    return 0 < int(state["full_context_correct_count"]) < int(state["full_context_total_count"])


def _full_status(state: Mapping[str, Any]) -> str:
    if _technical(state):
        return "technical"
    if _is_zero(state):
        return "zero"
    if _is_full(state):
        return "full"
    return "partial"


def _primary_transition(m0: Mapping[str, Any], clean: Mapping[str, Any], inclusive: Mapping[str, Any]) -> str:
    if any(_technical(row) for row in (m0, clean, inclusive)):
        return "TECHNICAL"
    if all(_is_zero(row) for row in (m0, clean, inclusive)):
        return "PERSISTENT_FULL_ZERO"
    if _is_zero(m0) and (_is_partial(clean) or _is_partial(inclusive)):
        return "EMERGING_FRONTIER"
    if _is_full(clean) and _is_full(inclusive):
        return "GRADUATED"
    if (
        m0.get("difficulty_bucket") in {"easy", "full_context_resolved"}
        and (clean.get("difficulty_bucket") in BAD_REGRESSION_BUCKETS or inclusive.get("difficulty_bucket") in BAD_REGRESSION_BUCKETS)
    ):
        return "REGRESSED"
    if clean.get("difficulty_bucket") == "hard_candidate" and inclusive.get("difficulty_bucket") == "hard_candidate":
        return "CONSENSUS_RESIDUAL"
    if (
        clean.get("difficulty_bucket") != inclusive.get("difficulty_bucket")
        or clean.get("full_context_correct_count") != inclusive.get("full_context_correct_count")
        or clean.get("no_context_correct_count") != inclusive.get("no_context_correct_count")
    ):
        return "MODEL_DISAGREEMENT"
    return "OTHER_STABLE"


def build_transition_matrix(
    model_rows: Mapping[str, list[dict[str, Any]]],
    exposure: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    indexed = {
        model: {str(row["candidate_id"]): row for row in rows}
        for model, rows in model_rows.items()
    }
    common = set(indexed["M0"])
    if any(set(rows) != common for rows in indexed.values()) or common != set(exposure):
        raise RuntimeError("three_model_candidate_alignment_mismatch")
    output = []
    for cid in sorted(common):
        source = indexed["M0"][cid]
        states = {model: _model_state(rows[cid]) for model, rows in indexed.items()}
        m0, clean, inclusive = states["M0"], states["M_clean_policy"], states["M_inclusive_policy"]
        exposure_row = exposure[cid]
        regression = (
            m0["difficulty_bucket"] in {"easy", "full_context_resolved"}
            and (clean["difficulty_bucket"] in BAD_REGRESSION_BUCKETS or inclusive["difficulty_bucket"] in BAD_REGRESSION_BUCKETS)
        )
        disagreement = (
            clean["difficulty_bucket"] != inclusive["difficulty_bucket"]
            or clean["full_context_correct_count"] != inclusive["full_context_correct_count"]
            or clean["no_context_correct_count"] != inclusive["no_context_correct_count"]
        )
        flags = {
            "consensus_partial": _is_partial(clean) and _is_partial(inclusive),
            "both_full_greater_than_no": all(
                not _technical(row)
                and int(row["full_context_correct_count"]) * int(row["no_context_total_count"])
                > int(row["no_context_correct_count"]) * int(row["full_context_total_count"])
                for row in (clean, inclusive)
            ),
            "pure_position_only": m0["position_mechanism"] == "position_only",
            "mixed_present": any(row["position_mechanism"] == "mixed" for row in states.values()),
            "ordering_temporal": source["question_type"] == "ordering",
            "directly_trained_by_clean": bool(exposure_row["directly_trained_by_clean"]),
            "directly_trained_by_inclusive": bool(exposure_row["directly_trained_by_inclusive"]),
            "unexposed_to_both": exposure_row["training_exposure"] == "UNEXPOSED_TO_BOTH",
            "regression_present": regression,
            "model_disagreement_present": disagreement,
            "newly_graduated": not _is_full(m0) and _is_full(clean) and _is_full(inclusive),
        }
        output.append({
            "transition_matrix_version": f"{VERSION}_transition_matrix_v1",
            "candidate_id": cid,
            "root_id": source["root_id"],
            "split": "train",
            "question_type": source["question_type"],
            "generation_source": source.get("generation_source"),
            "evidence_status": source.get("evidence_status"),
            "position_mechanism": m0["position_mechanism"],
            "training_exposure": exposure_row["training_exposure"],
            "pair_id": exposure_row.get("pair_id"),
            "pair_role": exposure_row.get("pair_role"),
            "models": states,
            "primary_transition": _primary_transition(m0, clean, inclusive),
            "flags": flags,
        })
    return output


def _improvement(rows: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    indexed = {
        model: {str(row["candidate_id"]): row for row in values}
        for model, values in rows.items()
    }
    ids = set(indexed["M0"])
    output = {}
    for model in MODEL_NAMES:
        full_delta = []
        no_delta = []
        gain_delta = []
        for cid in ids:
            before, after = indexed["M0"][cid], indexed[model][cid]
            if any(row.get("full_context_accuracy") is None or row.get("no_context_accuracy") is None for row in (before, after)):
                continue
            full_delta.append(float(after["full_context_accuracy"]) - float(before["full_context_accuracy"]))
            no_delta.append(float(after["no_context_accuracy"]) - float(before["no_context_accuracy"]))
            gain_delta.append(
                (float(after["full_context_accuracy"]) - float(after["no_context_accuracy"]))
                - (float(before["full_context_accuracy"]) - float(before["no_context_accuracy"]))
            )
        output[model] = {
            "paired_scored_candidate_count": len(full_delta),
            "mean_full_context_accuracy_delta_vs_m0": _mean(full_delta),
            "mean_no_context_accuracy_delta_vs_m0": _mean(no_delta),
            "mean_context_gain_delta_vs_m0": _mean(gain_delta),
        }
    return output


def _group_report(
    matrix: list[dict[str, Any]],
    model_rows: Mapping[str, list[dict[str, Any]]],
    dimension: str,
) -> dict[str, Any]:
    matrix_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in matrix:
        value = row["models"]["M0"]["difficulty_bucket"] if dimension == "m0_difficulty_bucket" else row.get(dimension)
        matrix_groups[str(value) if value is not None else "NOT_AVAILABLE"].append(row)
    indexes = {
        model: {str(row["candidate_id"]): row for row in rows}
        for model, rows in model_rows.items()
    }
    output = {}
    for key, values in sorted(matrix_groups.items()):
        ids = {str(row["candidate_id"]) for row in values}
        selected = {model: [index[cid] for cid in sorted(ids)] for model, index in indexes.items()}
        output[key] = {
            "candidate_count": len(values),
            "primary_transition_counts": dict(sorted(Counter(str(row["primary_transition"]) for row in values).items())),
            "flag_counts": {
                flag: sum(bool(row["flags"].get(flag)) for row in values)
                for flag in sorted(next(iter(values))["flags"])
            },
            "models": {model: aggregate_model(rows, model) for model, rows in selected.items()},
            "paired_improvement_vs_m0": _improvement(selected),
        }
    return {
        "report_version": f"{VERSION}_{dimension}_v1",
        "dimension": dimension,
        "groups": output,
    }


def _load_completed_model_rows(run_root: Path, model_name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scope = run_root / MODEL_SCOPES[model_name]
    report_path = scope / "model_execution_report.json"
    manifest_path = scope / "difficulty_manifest.jsonl"
    verify_sha_sidecar(report_path)
    verify_sha_sidecar(manifest_path)
    report = load_json(report_path)
    rows = load_jsonl(manifest_path)
    if (
        report.get("status") != "PASS"
        or int(report.get("candidate_count", -1)) != EXPECTED_TRAIN_CANDIDATES
        or int(report.get("request_count", -1)) != EXPECTED_TRAIN_REQUESTS
        or len(rows) != EXPECTED_TRAIN_CANDIDATES
        or len({str(row["candidate_id"]) for row in rows}) != EXPECTED_TRAIN_CANDIDATES
    ):
        raise RuntimeError(f"trained_model_outputs_incomplete:{model_name}")
    return rows, report


def _bucket_transition_report(
    matrix: list[dict[str, Any]],
    *,
    source_bucket: str,
    expected_count: int,
) -> dict[str, Any]:
    selected = [row for row in matrix if row["models"]["M0"]["difficulty_bucket"] == source_bucket]
    if len(selected) != expected_count:
        raise RuntimeError(f"source_bucket_transition_count_mismatch:{source_bucket}:{len(selected)}:{expected_count}")
    per_model = {}
    for model in MODEL_NAMES:
        states = [row["models"][model] for row in selected]
        per_model[model] = {
            "difficulty_bucket_counts": dict(sorted(Counter(str(row["difficulty_bucket"]) for row in states).items())),
            "full_status_counts": dict(sorted(Counter(_full_status(row) for row in states).items())),
            "full_context_accuracy_mean": _mean(row["full_context_accuracy"] for row in states),
            "no_context_accuracy_mean": _mean(row["no_context_accuracy"] for row in states),
        }
    return {
        "report_version": f"{VERSION}_{source_bucket}_transition_v1",
        "source_model": "M0",
        "source_bucket": source_bucket,
        "candidate_count": len(selected),
        "primary_transition_counts": dict(sorted(Counter(str(row["primary_transition"]) for row in selected).items())),
        "models": per_model,
        "consensus_partial_count": sum(row["flags"]["consensus_partial"] for row in selected),
        "both_full_greater_than_no_count": sum(row["flags"]["both_full_greater_than_no"] for row in selected),
        "model_disagreement_count": sum(row["flags"]["model_disagreement_present"] for row in selected),
    }


def _full_zero_report(matrix: list[dict[str, Any]]) -> dict[str, Any]:
    base = _bucket_transition_report(
        matrix, source_bucket="full_zero_review", expected_count=EXPECTED_M0_FULL_ZERO,
    )
    selected = [row for row in matrix if row["models"]["M0"]["difficulty_bucket"] == "full_zero_review"]
    both_partial = sum(
        _is_partial(row["models"]["M_clean_policy"]) and _is_partial(row["models"]["M_inclusive_policy"])
        for row in selected
    )
    only_clean_partial = sum(
        _is_partial(row["models"]["M_clean_policy"]) and not _is_partial(row["models"]["M_inclusive_policy"])
        for row in selected
    )
    only_inclusive_partial = sum(
        not _is_partial(row["models"]["M_clean_policy"]) and _is_partial(row["models"]["M_inclusive_policy"])
        for row in selected
    )
    at_least_one_full = sum(
        _is_full(row["models"]["M_clean_policy"]) or _is_full(row["models"]["M_inclusive_policy"])
        for row in selected
    )
    persistent = sum(
        _is_zero(row["models"]["M_clean_policy"]) and _is_zero(row["models"]["M_inclusive_policy"])
        for row in selected
    )
    base["special_transitions"] = {
        "m0_full_zero_to_both_models_partial": both_partial,
        "m0_full_zero_to_clean_partial_only": only_clean_partial,
        "m0_full_zero_to_inclusive_partial_only": only_inclusive_partial,
        "m0_full_zero_to_at_least_one_model_full": at_least_one_full,
        "persistent_full_zero_both_models": persistent,
    }
    return base


def _regression_report(matrix: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        row for row in matrix
        if row["models"]["M0"]["difficulty_bucket"] in {"easy", "full_context_resolved"}
    ]
    regressed = [row for row in eligible if row["flags"]["regression_present"]]
    return {
        "regression_report_version": f"{VERSION}_regression_v1",
        "m0_easy_or_resolved_candidate_count": len(eligible),
        "regressed_candidate_count": len(regressed),
        "regression_rate": len(regressed) / len(eligible) if eligible else None,
        "clean_regressed_count": sum(row["models"]["M_clean_policy"]["difficulty_bucket"] in BAD_REGRESSION_BUCKETS for row in eligible),
        "inclusive_regressed_count": sum(row["models"]["M_inclusive_policy"]["difficulty_bucket"] in BAD_REGRESSION_BUCKETS for row in eligible),
        "both_regressed_count": sum(
            row["models"]["M_clean_policy"]["difficulty_bucket"] in BAD_REGRESSION_BUCKETS
            and row["models"]["M_inclusive_policy"]["difficulty_bucket"] in BAD_REGRESSION_BUCKETS
            for row in eligible
        ),
        "by_m0_bucket": {
            bucket: {
                "candidate_count": sum(row["models"]["M0"]["difficulty_bucket"] == bucket for row in eligible),
                "regressed_count": sum(
                    row["models"]["M0"]["difficulty_bucket"] == bucket and row["flags"]["regression_present"]
                    for row in eligible
                ),
            }
            for bucket in ("easy", "full_context_resolved")
        },
        "regressed_candidate_ids": [str(row["candidate_id"]) for row in regressed],
    }


def _pair_report(matrix: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {str(row["candidate_id"]): row for row in matrix}
    rows = []
    for pair in sorted(pairs, key=lambda row: str(row["pair_id"])):
        clean_id, risk_id = str(pair["clean_candidate_id"]), str(pair["risk_candidate_id"])
        clean_control, risk_replacement = by_id[clean_id], by_id[risk_id]
        rows.append({
            "pair_id": pair["pair_id"],
            "clean_control_candidate_id": clean_id,
            "risk_replacement_candidate_id": risk_id,
            "clean_control": {
                model: clean_control["models"][model]
                for model in ("M0", *MODEL_NAMES)
            },
            "risk_replacement": {
                model: risk_replacement["models"][model]
                for model in ("M0", *MODEL_NAMES)
            },
            "clean_control_primary_transition": clean_control["primary_transition"],
            "risk_replacement_primary_transition": risk_replacement["primary_transition"],
        })
    return {
        "paired_candidate_report_version": f"{VERSION}_matched_pairs_v1",
        "pair_count": len(rows),
        "clean_control_mastery": {
            model: {
                "full_count": sum(_is_full(row["clean_control"][model]) for row in rows),
                "partial_count": sum(_is_partial(row["clean_control"][model]) for row in rows),
                "zero_count": sum(_is_zero(row["clean_control"][model]) for row in rows),
            }
            for model in MODEL_NAMES
        },
        "risk_replacement_mastery": {
            model: {
                "full_count": sum(_is_full(row["risk_replacement"][model]) for row in rows),
                "partial_count": sum(_is_partial(row["risk_replacement"][model]) for row in rows),
                "zero_count": sum(_is_zero(row["risk_replacement"][model]) for row in rows),
            }
            for model in MODEL_NAMES
        },
        "pairs": rows,
    }


def finalize_reports(run_root: Path) -> dict[str, Any]:
    preflight = load_json(run_root / "preflight_report.json")
    if preflight.get("status") != "PASS":
        raise RuntimeError("preflight_not_pass_for_finalize")
    train_rows, gates, evidence, _ = _load_train_inputs()
    exposure_rows, exposure, pairs = _load_exposure(train_rows)
    exposure_path = run_root / "candidate_training_exposure_manifest.jsonl"
    verify_sha_sidecar(exposure_path)
    if load_jsonl(exposure_path) != exposure_rows:
        raise RuntimeError("frozen_exposure_manifest_drift")
    m0_reuse = load_json(run_root / "m0_reuse_audit.json")
    if m0_reuse.get("reuse_allowed") is not True:
        raise RuntimeError("m0_not_reusable_for_finalize")

    m0_rows = _m0_rows(train_rows, gates, evidence)
    clean_rows, clean_report = _load_completed_model_rows(run_root, "M_clean_policy")
    inclusive_rows, inclusive_report = _load_completed_model_rows(run_root, "M_inclusive_policy")
    model_rows = {
        "M0": m0_rows,
        "M_clean_policy": clean_rows,
        "M_inclusive_policy": inclusive_rows,
    }
    matrix = build_transition_matrix(model_rows, exposure)
    if len(matrix) != EXPECTED_TRAIN_CANDIDATES:
        raise RuntimeError("transition_matrix_count_mismatch")
    write_once_jsonl(run_root / "candidate_transition_matrix.jsonl", matrix)

    summaries = {model: aggregate_model(rows, model) for model, rows in model_rows.items()}
    primary_counts = dict(sorted(Counter(str(row["primary_transition"]) for row in matrix).items()))
    flag_counts = {
        flag: sum(bool(row["flags"][flag]) for row in matrix)
        for flag in sorted(matrix[0]["flags"])
    }
    transition_summary = {
        "transition_summary_version": f"{VERSION}_transition_summary_v1",
        "candidate_count": len(matrix),
        "root_count": len({str(row["root_id"]) for row in matrix}),
        "models": summaries,
        "paired_improvement_vs_m0": _improvement(model_rows),
        "primary_transition_counts": primary_counts,
        "flag_counts": flag_counts,
        "technical_candidate_count": primary_counts.get("TECHNICAL", 0),
    }
    write_once_json(run_root / "transition_summary.json", transition_summary)
    group_outputs = {
        "transition_by_exposure.json": _group_report(matrix, model_rows, "training_exposure"),
        "transition_by_question_type.json": _group_report(matrix, model_rows, "question_type"),
        "transition_by_root.json": _group_report(matrix, model_rows, "root_id"),
        "transition_by_generation_source.json": _group_report(matrix, model_rows, "generation_source"),
        "transition_by_m0_difficulty_bucket.json": _group_report(matrix, model_rows, "m0_difficulty_bucket"),
    }
    for name, value in group_outputs.items():
        write_once_json(run_root / name, value)

    m0_hard = _bucket_transition_report(
        matrix, source_bucket="hard_candidate", expected_count=EXPECTED_M0_HARD,
    )
    m0_full_zero = _full_zero_report(matrix)
    regressions = _regression_report(matrix)
    pairs_report = _pair_report(matrix, pairs)
    write_once_json(run_root / "m0_hard_transition_report.json", m0_hard)
    write_once_json(run_root / "m0_full_zero_transition_report.json", m0_full_zero)
    write_once_json(run_root / "regression_report.json", regressions)
    write_once_json(run_root / "matched_pair_mastery_report.json", pairs_report)

    auto_feedback_counts = {
        "persistent_full_zero": primary_counts.get("PERSISTENT_FULL_ZERO", 0),
        "emerging_frontier": primary_counts.get("EMERGING_FRONTIER", 0),
        "graduated": primary_counts.get("GRADUATED", 0),
        "regressed_primary": primary_counts.get("REGRESSED", 0),
        "regression_present": flag_counts.get("regression_present", 0),
        "consensus_residual": primary_counts.get("CONSENSUS_RESIDUAL", 0),
        "model_disagreement_primary": primary_counts.get("MODEL_DISAGREEMENT", 0),
        "model_disagreement_present": flag_counts.get("model_disagreement_present", 0),
        "consensus_partial": flag_counts.get("consensus_partial", 0),
        "technical": primary_counts.get("TECHNICAL", 0),
    }
    final = {
        "final_train_reevaluation_report_version": f"{VERSION}_final_v1",
        "status": "ROUND0_TRAIN_REEVALUATION_COMPLETE",
        "next_status": "READY_FOR_FEEDBACK_COMPILER",
        "scope": "70_train_roots_2294_round0_parent_plus_r01_candidates",
        "train_candidate_count": len(matrix),
        "train_root_count": len({str(row["root_id"]) for row in matrix}),
        "m0_reused": True,
        "m0_new_solver_generation": 0,
        "clean_request_count": clean_report["request_count"],
        "inclusive_request_count": inclusive_report["request_count"],
        "clean_technical_failure_count": clean_report["technical_failure_count"],
        "inclusive_technical_failure_count": inclusive_report["technical_failure_count"],
        "clean_truncated_request_count": clean_report["truncated_request_count"],
        "inclusive_truncated_request_count": inclusive_report["truncated_request_count"],
        "clean_parse_failure_count": clean_report["parse_failure_count"],
        "inclusive_parse_failure_count": inclusive_report["parse_failure_count"],
        "models": summaries,
        "training_exposure_counts": dict(sorted(Counter(row["training_exposure"] for row in matrix).items())),
        "m0_hard_transition": m0_hard,
        "m0_full_zero_transition": m0_full_zero,
        "regression_summary": {key: value for key, value in regressions.items() if key != "regressed_candidate_ids"},
        "automatic_feedback_candidate_counts": auto_feedback_counts,
        "primary_transition_counts": primary_counts,
        "no_feedback_card_generated": True,
        "r02_accessed": False,
        "dev_test_candidate_results_accessed": False,
        "qa_writer_calls": 0,
        "semantic_auditor_calls": 0,
        "training_started": False,
        "major_output_paths": [
            str(run_root / name) for name in (
                "preflight_report.json", "model_identity_report.json", "solver_contract.json",
                "solver_plan.json", "m0_reuse_audit.json", "clean_train", "inclusive_train",
                "candidate_training_exposure_manifest.jsonl", "candidate_transition_matrix.jsonl",
                "transition_summary.json", "transition_by_exposure.json",
                "transition_by_question_type.json", "transition_by_root.json",
                "transition_by_generation_source.json", "transition_by_m0_difficulty_bucket.json",
                "m0_hard_transition_report.json", "m0_full_zero_transition_report.json",
                "regression_report.json", "matched_pair_mastery_report.json",
                "final_train_reevaluation_report.json", "final_train_reevaluation_report.md",
            )
        ],
    }
    write_once_json(run_root / "final_train_reevaluation_report.json", final)
    candidate = summaries["M0"]["candidate_weighted"]
    clean = summaries["M_clean_policy"]["candidate_weighted"]
    inclusive = summaries["M_inclusive_policy"]["candidate_weighted"]
    md = "\n".join([
        "# Round-0 Train Solver Reevaluation",
        "",
        f"Status: **{final['status']}**",
        f"Next: **{final['next_status']}**",
        "",
        f"- Scope: `{final['train_root_count']}` Train roots and `{final['train_candidate_count']}` Parent+R01 candidates.",
        f"- Requests: M0 reused with `0` new generations; Clean `{final['clean_request_count']}`; Inclusive `{final['inclusive_request_count']}`.",
        f"- Technical failures: Clean `{final['clean_technical_failure_count']}`; Inclusive `{final['inclusive_technical_failure_count']}`.",
        f"- M0 candidate mean full/no/gain: `{candidate}`.",
        f"- Clean candidate mean full/no/gain: `{clean}`.",
        f"- Inclusive candidate mean full/no/gain: `{inclusive}`.",
        f"- Primary transitions: `{primary_counts}`.",
        f"- Automatic-feedback candidate counts: `{auto_feedback_counts}`.",
        "",
        "No feedback card was compiled. No R02, Dev/Test candidate results, QA Writer, semantic auditor, expansion, or training was accessed or started.",
        "",
    ])
    write_once_text(run_root / "final_train_reevaluation_report.md", md)
    write_once_text(
        run_root / "final_train_reevaluation_report.md.sha256",
        sha256_file(run_root / "final_train_reevaluation_report.md") + "\n",
    )
    return final


__all__ = [
    "aggregate_model", "build_solver_plan", "build_transition_matrix", "finalize_reports",
    "run_preflight", "run_trained_model",
]
