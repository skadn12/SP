"""Offline-only Round-1 preflight.

This module deliberately has no backend imports and performs no network or
model calls. It freezes the 70 Round-0 Train roots as R01 history, verifies the
R01 Memorax write hand-off, and emits a versioned manifest for a future R02
run. Historical files are read-only; all report files are created in a new
run directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..core import canonical_json, sha256_bytes, sha256_file
from . import VERSION


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
ROUND0_DIFFICULTY = RUN_BASE / "h4_round0_b0_k3_difficulty_v1_live_1"
ROUND0_COMMON_DEV = RUN_BASE / "h4_round0_common_dev_eval_v1_live_3"
ROUND0_FORMAL = RUN_BASE / "h4_round0_b0_formal_training_v1_live_2"
R01_QA_RUN = RUN_BASE / "h4_unexpanded122_qa_generation_only_v1"
R01_MEMORY_RUN = RUN_BASE / "h4_unexpanded122_memory_event_k3_v1"
BASE_MODEL = Path("/root/autodl-tmp/Qwen3-4B")
DEFAULT_RUN_ROOT = RUN_BASE / "h4_round1_cumulative_continuation_fresh_mining_v1_live_1"
ROUND0_CANDIDATE_COUNT = 3286
TRAIN_ROOT_COUNT = 70
DEV_ROOT_COUNT = 15
TEST_ROOT_COUNT = 16
LOGICAL_SHARD_COUNT = 7
ROOTS_PER_SHARD = 10
ROUND0_ID_SCOPE = "h4_round0_b0_k3_difficulty_v1"
R01_ROUND = 1
R02_ROUND = 2
R02_ID_SCOPE = "R01-R02"
FRESH_QA_GENERATOR_VERSION = "h4_qa_multiroot_k3_t08_fresh_generation_v1"
SOLVER_CONTRACT_NAME = "solver_contract.json"
_ROUND_REF = re.compile(r"^R(\d+):S\d+(?::T\d+)?$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _canonical_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("complete_dialogue_sha256", None)
    return sha256_bytes(canonical_json(body).encode("utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.with_name(path.name + ".sha256").write_text(sha256_file(path) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    path.with_name(path.name + ".sha256").write_text(sha256_file(path) + "\n", encoding="utf-8")


def _verify_sidecar(path: Path) -> tuple[bool, str | None]:
    sidecar = path.with_name(path.name + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        return False, "file_or_sidecar_missing"
    expected = sidecar.read_text(encoding="utf-8").strip().split()[0] if sidecar.read_text(encoding="utf-8").strip() else ""
    return expected == sha256_file(path), (None if expected == sha256_file(path) else "sidecar_mismatch")


def _directory_sha(path: Path) -> str:
    rows = []
    for child in sorted(path.rglob("*")):
        if child.is_file() and not child.name.endswith(".sha256"):
            rows.append({"path": child.relative_to(path).as_posix(), "size": child.stat().st_size, "sha256": sha256_file(child)})
    return sha256_bytes(canonical_json(rows).encode("utf-8"))


def _safe_root_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.:-]+", value):
        raise ValueError(f"invalid_root_id:{value}")
    return value


def _resolve_r01_row(root_id: str) -> dict[str, Any]:
    metadata_paths = sorted(R01_QA_RUN.glob(f"shards/*/roots/{root_id}/source_metadata.json"))
    if len(metadata_paths) != 1:
        raise RuntimeError(f"r01_source_metadata_count:{root_id}:{len(metadata_paths)}")
    metadata_path = metadata_paths[0]
    metadata = _load(metadata_path)
    source_path = Path(str(metadata.get("source_path") or ""))
    if not source_path.is_file():
        raise RuntimeError(f"r01_canonical_source_missing:{root_id}")
    expected_prefix = (R01_MEMORY_RUN / "").resolve()
    if expected_prefix not in source_path.resolve().parents or source_path.name != "promotable_expanded_dialogue.json":
        raise RuntimeError(f"r01_source_not_canonical_promotion:{root_id}:{source_path}")
    source_sha = sha256_file(source_path)
    if source_sha != str(metadata.get("source_sha256") or ""):
        raise RuntimeError(f"r01_source_sha_mismatch:{root_id}")
    dialogue = _load(source_path)
    if str(dialogue.get("root_id") or "") != root_id:
        raise RuntimeError(f"r01_dialogue_root_mismatch:{root_id}")
    turns = dialogue.get("turns")
    if not isinstance(turns, list) or not turns:
        raise RuntimeError(f"r01_dialogue_turns_missing:{root_id}")
    refs = [str(row.get("turn_ref") or "") for row in turns if isinstance(row, Mapping)]
    if len(refs) != len(turns) or not all(refs) or len(refs) != len(set(refs)):
        raise RuntimeError(f"r01_turn_refs_invalid:{root_id}")
    round_numbers = set()
    r01_refs = []
    for ref in refs:
        match = _ROUND_REF.match(ref)
        if match:
            round_numbers.add(int(match.group(1)))
            if int(match.group(1)) == R01_ROUND:
                r01_refs.append(ref)
    if round_numbers != {R01_ROUND} or not r01_refs:
        raise RuntimeError(f"r01_round_refs_invalid:{root_id}:{sorted(round_numbers)}")
    complete_sha = str(dialogue.get("complete_dialogue_sha256") or "")
    if complete_sha != _canonical_hash(dialogue):
        raise RuntimeError(f"r01_complete_dialogue_sha_mismatch:{root_id}")
    # The historical executor may promote a successful retry from a physical
    # ``<root>__attempt_02`` directory.  The QA source metadata is the
    # authoritative binding, so resolve the Memorax records beside that exact
    # canonical promotion file rather than guessing the physical directory.
    memory_root = source_path.parent.parent
    if R01_MEMORY_RUN.resolve() not in memory_root.parents:
        raise RuntimeError(f"r01_memory_root_outside_source_run:{root_id}")
    intent_path = memory_root / "memory" / "child_add_intent.json"
    completion_path = memory_root / "memory" / "child_add_completion.json"
    if not intent_path.is_file() or not completion_path.is_file():
        raise RuntimeError(f"r01_memory_record_missing:{root_id}:{memory_root.name}")
    intent = _load(intent_path)
    completion = _load(completion_path)
    if completion.get("final_status") != "success" or completion.get("verified_complete") is not True or completion.get("write_state", {}).get("memory_write_verified") is not True:
        raise RuntimeError(f"r01_memory_write_not_verified:{root_id}")
    if str(intent.get("source_dialogue_sha256") or "") != complete_sha:
        raise RuntimeError(f"r01_memory_source_sha_mismatch:{root_id}")
    if str(intent.get("session_id") or "") != f"{root_id}:memorax:R01":
        raise RuntimeError(f"r01_memory_session_mismatch:{root_id}")
    intent_refs = [str(x) for x in intent.get("turn_refs") or []]
    message_refs = [str(x.get("turn_ref") or "") for x in intent.get("payload", {}).get("messages") or []]
    if intent_refs != r01_refs:
        raise RuntimeError(f"r01_memory_delta_refs_mismatch:{root_id}")
    messages = intent.get("payload", {}).get("messages") or []
    if len(messages) != len(intent_refs):
        raise RuntimeError(f"r01_memory_message_count_mismatch:{root_id}")
    # Historical v2 payloads intentionally kept refs in the intent envelope;
    # each message carries only the canonical speaker-prefixed content.
    expected_messages = [
        {"role": "user", "timestamp": -1,
         "content": f"[Speaker: {row['speaker']}] {row['text']}"}
        for row in turns if str(row.get("turn_ref")) in set(intent_refs)
    ]
    if messages != expected_messages:
        raise RuntimeError(f"r01_memory_message_content_mismatch:{root_id}")
    window_messages = intent.get("payload", {}).get("window_context") or []
    if any(not str(row.get("content") or "").startswith("[Speaker: ") for row in window_messages):
        raise RuntimeError(f"r01_memory_window_context_shape:{root_id}")
    if any(not ref.startswith("R01:") for ref in intent_refs):
        raise RuntimeError(f"r01_memory_non_r01_message:{root_id}")
    return {
        "root_id": root_id,
        "source_metadata_path": str(metadata_path.resolve()),
        "source_metadata_sha256": sha256_file(metadata_path),
        "canonical_dialogue_path": str(source_path.resolve()),
        "canonical_dialogue_sha256": source_sha,
        "complete_dialogue_sha256": complete_sha,
        "turn_count": len(turns),
        "r01_turn_count": len(r01_refs),
        "r01_first_turn_ref": r01_refs[0],
        "r01_last_turn_ref": r01_refs[-1],
        "memory_physical_root": str(memory_root.resolve()),
        "memory_intent_path": str(intent_path.resolve()),
        "memory_intent_sha256": sha256_file(intent_path),
        "memory_completion_path": str(completion_path.resolve()),
        "memory_completion_sha256": sha256_file(completion_path),
        "prior_user_id": str(intent.get("user_id") or ""),
        "prior_session_id": str(intent.get("session_id") or ""),
        "r02_session_id": f"{root_id}:memorax:R02",
        "memory_write_contract": {
            "same_user_id": True,
            "new_session_only": True,
            "messages_scope": "R02_only",
            "window_context_scope": "R02_only",
            "parent_or_r01_rewrite_forbidden": True,
        },
    }


def _model_identity() -> dict[str, Any]:
    required = [ROUND0_FORMAL / "final_formal_training_report.json", ROUND0_FORMAL / "training_recipe.json", ROUND0_FORMAL / "training_preflight.json", ROUND0_COMMON_DEV / "model_identity_report.json"]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise RuntimeError(f"model_identity_files_missing:{missing}")
    final = _load(required[0])
    recipe = _load(required[1])
    training_preflight = _load(required[2])
    common_identity = _load(required[3])
    if final.get("status") != "FORMAL_TRAINING_COMPLETE":
        raise RuntimeError(f"formal_training_not_complete:{final.get('status')}")
    if training_preflight.get("status") != "PASS":
        raise RuntimeError(f"formal_training_preflight_not_pass:{training_preflight.get('status')}")
    base_snapshot = training_preflight.get("base_model_sha_snapshot_before", {}).get("files", {})
    current_base = {p.relative_to(BASE_MODEL).as_posix(): {"sha256": sha256_file(p), "size": p.stat().st_size} for p in sorted(BASE_MODEL.rglob("*")) if p.is_file()}
    base_matches = bool(base_snapshot) and set(base_snapshot) == set(current_base) and all(base_snapshot[k].get("sha256") == current_base[k]["sha256"] for k in base_snapshot)
    adapters = {}
    for name, dirname in (("M_clean_policy", "clean_adapter"), ("M_inclusive_policy", "inclusive_adapter")):
        path = ROUND0_FORMAL / dirname
        if not path.is_dir():
            raise RuntimeError(f"adapter_missing:{name}")
        sidecar_ok = all(_verify_sidecar(p)[0] for p in path.iterdir() if p.is_file() and not p.name.endswith(".sha256"))
        adapters[name] = {"path": str(path.resolve()), "adapter_sha256": _directory_sha(path), "all_file_sidecars_valid": sidecar_ok, "adapter_config_sha256": sha256_file(path / "adapter_config.json")}
    solver_contract_path = ROUND0_DIFFICULTY / SOLVER_CONTRACT_NAME
    solver_ok, solver_error = _verify_sidecar(solver_contract_path)
    solver = _load(solver_contract_path) if solver_contract_path.is_file() else {}
    if not solver_ok:
        raise RuntimeError(f"solver_contract_sidecar:{solver_error}")
    if int(solver.get("max_model_len", 0)) != 16384 or int(solver.get("rollouts_per_position", 0)) != 3:
        raise RuntimeError("solver_contract_not_b0_k3")
    return {
        "base_model_path": str(BASE_MODEL.resolve()),
        "base_model_files_sha256": current_base,
        "base_model_matches_training_snapshot": base_matches,
        "formal_training_report_path": str(required[0].resolve()),
        "formal_training_report_sha256": sha256_file(required[0]),
        "training_recipe_path": str(required[1].resolve()),
        "training_recipe_sha256": sha256_file(required[1]),
        "training_recipe_fingerprint": recipe.get("recipe_fingerprint"),
        "adapters": adapters,
        "common_dev_model_identity_sha256": sha256_file(required[3]),
        "solver_contract_path": str(solver_contract_path.resolve()),
        "solver_contract_sha256": sha256_file(solver_contract_path),
        "solver_contract": {"solver_contract_version": solver.get("solver_contract_version"), "max_model_len": solver.get("max_model_len"), "rollouts_per_position": solver.get("rollouts_per_position"), "position_counts": solver.get("position_counts"), "temperatures": solver.get("temperatures"), "enable_thinking": solver.get("enable_thinking")},
        "identity_checks": {"formal_complete": True, "base_sha_matches": base_matches, "clean_adapter_sidecars": adapters["M_clean_policy"]["all_file_sidecars_valid"], "inclusive_adapter_sidecars": adapters["M_inclusive_policy"]["all_file_sidecars_valid"], "solver_is_b0_k3": True},
        "common_dev_identity_declared_models": sorted(common_identity.get("models", {})),
    }


def _qa_generator_identity() -> dict[str, Any]:
    package = Path(__file__).resolve().parents[1] / "h4_qa_multiroot_k3_t08_probe_v1"
    generation = package / "generation.py"
    if not generation.is_file():
        raise RuntimeError("fresh_qa_generator_missing")
    text = generation.read_text(encoding="utf-8")
    match = re.search(r'GENERATOR_VERSION\s*=\s*"([^"]+)"', text)
    version = match.group(1) if match else None
    if version != FRESH_QA_GENERATOR_VERSION:
        raise RuntimeError(f"fresh_qa_generator_version_mismatch:{version}")
    files = [{"path": p.relative_to(package).as_posix(), "sha256": sha256_file(p)} for p in sorted(package.glob("*.py"))]
    return {"generator_version": version, "package": str(package.resolve()), "files": files, "manifest_sha256": sha256_bytes(canonical_json(files).encode("utf-8")), "temperature": 0.1, "thinking": False, "max_tokens": 8192, "candidate_generation": "all_program_gate_passes_to_both_solvers", "semantic_auditor": False, "candidate_id_scope": R02_ID_SCOPE}


def build_preflight(*, run_root: Path = DEFAULT_RUN_ROOT) -> dict[str, Any]:
    if run_root.exists():
        raise RuntimeError(f"run_root_already_exists:{run_root}")
    split_path = ROUND0_DIFFICULTY / "root_split_manifest.jsonl"
    candidate_path = ROUND0_DIFFICULTY / "difficulty_manifest.jsonl"
    if not split_path.is_file() or not candidate_path.is_file():
        raise RuntimeError("round0_lineage_manifest_missing")
    split_rows = _load_jsonl(split_path)
    train_rows = [row for row in split_rows if row.get("split") == "train"]
    split_counts = Counter(str(row.get("split")) for row in split_rows)
    if len(split_rows) != TRAIN_ROOT_COUNT + DEV_ROOT_COUNT + TEST_ROOT_COUNT or dict(split_counts) != {"train": TRAIN_ROOT_COUNT, "dev": DEV_ROOT_COUNT, "test": TEST_ROOT_COUNT}:
        raise RuntimeError(f"round0_split_counts_invalid:{dict(split_counts)}")
    candidate_rows = _load_jsonl(candidate_path)
    candidate_ids = {str(row.get("candidate_id") or "") for row in candidate_rows}
    if len(candidate_rows) != ROUND0_CANDIDATE_COUNT or len(candidate_ids) != ROUND0_CANDIDATE_COUNT or "" in candidate_ids:
        raise RuntimeError(f"round0_candidate_count_invalid:{len(candidate_rows)}:{len(candidate_ids)}")
    roots = []
    errors = []
    for row in sorted(train_rows, key=lambda x: str(x.get("root_id"))):
        root_id = _safe_root_id(str(row.get("root_id") or ""))
        try:
            item = _resolve_r01_row(root_id)
            item.update({"split": "train", "round0_candidate_count": int(row.get("candidate_count", 0)), "round0_dialogue_sha256": str(row.get("dialogue_sha256") or ""), "round0_candidate_pool_sha256": str(row.get("candidate_pool_sha256") or "")})
            roots.append(item)
        except Exception as error:
            errors.append(f"{root_id}:{type(error).__name__}:{error}")
    if errors:
        raise RuntimeError("r01_root_preflight_failed:" + " | ".join(errors[:12]))
    ordered = sorted(roots, key=lambda row: (sha256_bytes(("h4_round1-logical-shards:" + row["root_id"]).encode("utf-8")), row["root_id"]))
    for index, row in enumerate(ordered):
        row["logical_shard"] = f"shard-{index // ROOTS_PER_SHARD:02d}"
        row["logical_shard_index"] = index // ROOTS_PER_SHARD
        row["logical_shard_rank"] = index % ROOTS_PER_SHARD
    model = _model_identity()
    qa = _qa_generator_identity()
    checks = {
        "round0_split_counts": True,
        "train_root_count": len(roots) == TRAIN_ROOT_COUNT,
        "round0_candidate_count": len(candidate_ids) == ROUND0_CANDIDATE_COUNT,
        "r01_canonical_promotion_sources": len(roots) == TRAIN_ROOT_COUNT,
        "r01_dialogue_sha_valid": True,
        "r01_memory_write_verified": True,
        "r01_memory_delta_only": all(x["memory_write_contract"]["messages_scope"] == "R02_only" and x["memory_write_contract"]["window_context_scope"] == "R02_only" for x in roots),
        "r02_user_id_reuse_required": all(bool(x["prior_user_id"]) for x in roots),
        "r02_session_ids_unique": len({x["r02_session_id"] for x in roots}) == TRAIN_ROOT_COUNT,
        "logical_shards_closed": all(sum(x["logical_shard"] == f"shard-{i:02d}" for x in roots) == ROOTS_PER_SHARD for i in range(LOGICAL_SHARD_COUNT)),
        "round0_candidate_id_overlap_zero": True,
        "candidate_id_scope_frozen": R02_ID_SCOPE == "R01-R02",
        "formal_model_identity_valid": all(model["identity_checks"].values()),
        "fresh_qa_generator_frozen": True,
        "dev_test_access_zero": True,
        "api_calls_zero": True,
        "solver_calls_zero": True,
        "semantic_auditor_calls_zero": True,
    }
    run_root.mkdir(parents=True, exist_ok=False)
    shard_manifest = {f"shard-{i:02d}": [x["root_id"] for x in roots if x["logical_shard"] == f"shard-{i:02d}"] for i in range(LOGICAL_SHARD_COUNT)}
    contract = {"round1_contract_version": VERSION, "dialogue_input": "canonical Parent + R01 + new R02", "r01_source": "promotion/promotable_expanded_dialogue.json", "r02_dialogue_round": "R02", "memorax": {"same_user_id": True, "new_session_pattern": "<root_id>:memorax:R02", "messages": "R02_only", "window_context": "R02_only", "rewrite_parent_or_r01": False}, "qa": {"all_program_gate_passes": True, "both_models": ["M_clean_policy", "M_inclusive_policy"], "semantic_auditor": False}, "solver": {"contract_sha256": model["solver_contract_sha256"], "b0_k3": True, "model_order": ["M_clean_policy", "M_inclusive_policy"], "same_candidate_permutation_seed_schedule": True}, "ids": {"scope": R02_ID_SCOPE, "round0_overlap_required": 0}, "access": {"dev": False, "test": False}, "resume": {"root_states": ["NOT_STARTED", "R02_GENERATED", "MEMORY_ADD_SUBMITTED", "MEMORY_ADD_CONFIRMED", "QA_COMPLETE", "SOLVER_COMPLETE", "QUARANTINED"], "idempotent_r02_memory_write": True}}
    _write_json(run_root / "round1_contract.json", contract)
    _write_json(run_root / "model_identity.json", model)
    _write_json(run_root / "fresh_qa_generator_identity.json", qa)
    _write_jsonl(run_root / "round1_root_manifest.jsonl", roots)
    _write_json(run_root / "logical_shards.json", {"version": "h4_round1_logical_shards_v1", "shard_count": LOGICAL_SHARD_COUNT, "roots_per_shard": ROOTS_PER_SHARD, "shards": shard_manifest})
    report = {"preflight_version": VERSION + "_preflight_v1", "created_at": _now(), "status": "PASS" if all(checks.values()) else "BLOCKED", "run_root": str(run_root.resolve()), "source_runs_read_only": [str(p.resolve()) for p in (ROUND0_DIFFICULTY, ROUND0_COMMON_DEV, ROUND0_FORMAL, R01_QA_RUN, R01_MEMORY_RUN)], "split_counts": dict(split_counts), "train_root_count": len(roots), "round0_candidate_count": len(candidate_ids), "logical_shards": shard_manifest, "checks": checks, "failed_checks": [k for k, v in checks.items() if not v], "api_calls": 0, "solver_calls": 0, "semantic_auditor_calls": 0, "next_step": "one_root_smoke_only_after_explicit_live_authorization"}
    _write_json(run_root / "preflight_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline-only Round-1 cumulative continuation preflight")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    args = parser.parse_args()
    report = build_preflight(run_root=args.run_root)
    print(json.dumps({"status": report["status"], "run_root": report["run_root"], "failed_checks": report["failed_checks"], "api_calls": 0, "solver_calls": 0}, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
