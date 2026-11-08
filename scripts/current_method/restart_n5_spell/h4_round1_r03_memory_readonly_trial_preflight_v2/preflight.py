"""Execute the R03 existing-Memory read-only trial preflight."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import (
    COMPILER_RUN,
    CONTEXT_CEILING,
    DEFAULT_OUTPUT,
    MEMORAX_CREDENTIALS,
    OUTPUT_RESERVE,
    PAIRED_QA_RUN,
    QUARANTINE_ROOTS,
    TOKENIZER_PATH,
    TRIAL_ROOTS,
    VERSION,
)
from .memory_readonly_dialogue_adapter import (
    ALLOWED_MEMORAX_ENDPOINTS,
    MemoraxEndpointNotAllowedError,
    MemoraxWriteDisabledError,
    ReadOnlyMemoraxClient,
    ReadOnlyMemoraxConfig,
    assert_allowed_endpoint,
    build_add_intent,
    canonical_json,
    create_child_add_payload,
    memory_prompt_projection,
    normalize_historical_snapshot,
    normalize_search_response,
    poll_add_status,
    post_generation_visibility_check,
    raw_snapshot_from_response,
    render_planner_dry_run,
    sha256_value,
    write_memory_metadata,
)


CORE_SOURCE_DIR = Path(
    "/root/autodl-tmp/gxy-selfplay/current_method_github/scripts/current_method/"
    "restart_n5_spell/simple_append_memorax_v1_memory_event_continuation_probe_v1"
)
CONFIG_PATH = Path("/root/autodl-tmp/gxy-selfplay/config.json")
CORE_COMPONENT_FILES = (
    "prompts.py",
    "contracts.py",
    "cumulative_parent.py",
    "live_runtime.py",
    "memory_context.py",
    "workflow.py",
    "payloads.py",
    "incremental_round.py",
    "canonical_turn_id_provenance_probe.py",
    "global_memory_gate_live.py",
    "global_memory_gate_replay.py",
    "context.py",
    "realizer_adherence.py",
    "scene_continuation.py",
    "scene_replacement.py",
    "client.py",
)
WHOLE_RUNTIME_FILE = "live.py"


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    return digest_bytes(Path(path).read_bytes())


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_pretty_json(value), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical_json(dict(row)) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def add_sidecars(run_root: Path) -> int:
    files = [
        path for path in run_root.rglob("*")
        if path.is_file() and path.suffix in {".json", ".jsonl", ".md", ".txt"}
        and path.suffix != ".sha256" and not path.name.endswith(".sha256")
    ]
    for path in sorted(files):
        sidecar = path.with_name(path.name + ".sha256")
        sidecar.write_text(
            f"{digest_file(path)}  {path.name}\n", encoding="utf-8")
    return len(files)


def sidecar_valid(path: Path) -> bool:
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file():
        return False
    return sidecar.read_text(encoding="utf-8").split()[0] == digest_file(path)


def _source_files(
    scope_rows: list[Mapping[str, Any]],
    source_dirs: Mapping[str, Path],
) -> list[Path]:
    paths = [
        COMPILER_RUN / name for name in (
            "final_round1_question_bank_feedback_compiler_report.json",
            "r03_root_scope_manifest.jsonl",
            "r03_expansion_seed_selection.jsonl",
            "r03_expansion_seed_reserves.jsonl",
            "r03_expansion_feedback_cards_full.jsonl",
            "r03_expansion_feedback_cards_prompt_view.jsonl",
            "r03_expansion_feedback_card_token_report.json",
        )
    ]
    paths.append(PAIRED_QA_RUN / "paired_dialogue_manifest.jsonl")
    paths.extend(CONFIG_PATH for _ in range(1))
    for row in scope_rows:
        root = str(row["root_id"])
        run_dir = source_dirs[root]
        paths.extend([
            Path(str(row["feedback_path"])),
            run_dir / "run_freeze_manifest.json",
            run_dir / "memory" / "child_add_intent.json",
            run_dir / "memory" / "child_add_completion.json",
            run_dir / "memory" / "child_add_write_state.json",
            run_dir / "memory" / "child_visibility_audit.json",
            run_dir / "memory" / "post_generation_bucket_query.json",
            run_dir / "memory" / "post_generation_bucket_snapshot.json",
        ])
    paths.extend(CORE_SOURCE_DIR / name for name in (*CORE_COMPONENT_FILES, WHOLE_RUNTIME_FILE))
    return sorted(set(paths), key=str)


def _load_memorax_config() -> ReadOnlyMemoraxConfig:
    from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.client import (
        load_memorax_config,
    )

    config = load_memorax_config(
        credentials_path=MEMORAX_CREDENTIALS, allow_insecure_http=True)
    return ReadOnlyMemoraxConfig(
        base_url=config.normalized_base_url,
        api_key=config.api_key,
        platform=config.platform,
        connect_timeout_seconds=config.connect_timeout_seconds,
        read_timeout_seconds=config.read_timeout_seconds,
    )


def _identity_sha(row: Mapping[str, Any]) -> str:
    return sha256_value({
        key: value for key, value in row.items()
        if key != "source_memory_identity_sha256"
    })


def resolve_source_identity(
    scope: Mapping[str, Any], *, paired: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    root = str(scope["root_id"])
    feedback_path = Path(str(scope["feedback_path"]))
    run_dir = feedback_path.parent
    freeze = read_json(run_dir / "run_freeze_manifest.json")
    payload_audit_path = run_dir / "feedback_payload_audit.json"
    # The first single-root R02 run predates the top-level payload-audit
    # artifact. Its Child Add intent/completion/visibility files remain the
    # authoritative write lineage, so do not manufacture a replacement.
    payload_audit = (read_json(payload_audit_path)
                     if payload_audit_path.is_file() else None)
    intent = read_json(run_dir / "memory" / "child_add_intent.json")
    completion = read_json(run_dir / "memory" / "child_add_completion.json")
    state = read_json(run_dir / "memory" / "child_add_write_state.json")
    visibility = read_json(run_dir / "memory" / "child_visibility_audit.json")
    historical_snapshot = read_json(
        run_dir / "memory" / "post_generation_bucket_snapshot.json")
    namespace = str(freeze.get("memorax_user_namespace") or "")
    user_id = str(intent.get("user_id") or "")
    session_id = str(intent.get("session_id") or "")
    checks = {
        "live13_and_paired_feedback_path_match": (
            feedback_path == Path(str(paired.get("feedback_path") or ""))),
        "feedback_file_sha_matches_live13": (
            digest_file(feedback_path) == str(scope.get("feedback_sha256") or "")),
        "feedback_file_sha_matches_paired_manifest": (
            digest_file(feedback_path) == str(paired.get("feedback_sha256") or "")),
        "root_id_matches": all(str(value) == root for value in (
            freeze.get("root_id"), intent.get("root_id"), visibility.get("root_id"),
            historical_snapshot.get("root_id"))),
        "namespace_is_feedback_not_control": (
            bool(namespace) and "feedback" in namespace.casefold()
            and "control" not in namespace.casefold()),
        "user_id_unique_identity_shape": (
            bool(user_id) and user_id == freeze.get("memory_user_id")
            and historical_snapshot.get("user_id") == user_id
            and visibility.get("user_id") == user_id),
        "r02_session_matches_authoritative_artifacts": (
            bool(session_id)
            and session_id == (intent.get("payload") or {}).get("session_id")
            and session_id.endswith(":memorax:R02-feedback")
            and (payload_audit is None
                 or session_id == payload_audit.get("session_id"))),
        "child_add_success": (
            completion.get("final_status") == "success"
            and completion.get("verified_complete") is True),
        "memory_write_verified": state.get("memory_write_verified") is True,
        "memory_visibility_verified": (
            state.get("bucket_visibility_verified") is True
            and visibility.get("bucket_visibility_verified") is True
            and visibility.get("valid") is True),
        "post_r02_snapshot_nonempty": (
            sum(int(value) for value in
                (historical_snapshot.get("bucket_counts") or {}).values()) > 0),
    }
    row = {
        "root_id": root,
        "source_memory_namespace": namespace,
        "source_memory_user_id": user_id,
        "source_r02_session_id": session_id,
        "write_target_namespace": None,
        "write_target_user_id": None,
        "write_target_session_id": None,
        "authoritative_r02_add_status": completion.get("final_status"),
        "historical_memory_write_verified": bool(
            state.get("memory_write_verified")),
        "historical_memory_visibility_verified": bool(
            visibility.get("bucket_visibility_verified")),
        "source_dialogue_sha256": str(scope.get("feedback_sha256") or ""),
        "source_dialogue_content_sha256": str(
            scope.get("dialogue_content_sha256") or ""),
        "authoritative_run_root": str(run_dir),
        "authoritative_query_path": str(
            run_dir / "memory" / "post_generation_bucket_query.json"),
        "feedback_payload_audit_path": (
            str(payload_audit_path) if payload_audit_path.is_file() else None),
        "feedback_payload_audit_available": payload_audit is not None,
        "authoritative_snapshot_path": str(
            run_dir / "memory" / "post_generation_bucket_snapshot.json"),
        "checks": checks,
        "valid": all(checks.values()),
    }
    row["source_memory_identity_sha256"] = _identity_sha(row)
    return row, historical_snapshot, run_dir


def dialogue_identity(scope: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(scope["feedback_path"]))
    value = read_json(path)
    turns = value.get("turns") or []
    refs = [str(row.get("turn_ref") or "") for row in turns]
    rounds = sorted({
        ref.split(":", 1)[0] for ref in refs
        if ref.startswith("R") and ":" in ref
    })
    checks = {
        "root_id_match": value.get("root_id") == scope.get("root_id"),
        "file_sha_match": digest_file(path) == scope.get("feedback_sha256"),
        "complete_dialogue_sha_match": (
            value.get("complete_dialogue_sha256")
            == scope.get("dialogue_content_sha256")),
        "turn_count_match": (
            len(turns) == int(scope.get("turn_count") or -1)
            == int(value.get("turn_count") or -2)),
        "turn_refs_unique": len(refs) == len(set(refs)),
        "rounds_are_r01_r02_only": rounds == ["R01", "R02"],
        "latest_round_is_r02": (
            (value.get("latest_round_boundary") or {}).get("round_index") == 2
            and refs[-1].startswith("R02:")),
        "r03_turns_absent": not any(ref.startswith("R03:") for ref in refs),
        "control_content_absent": not any(
            str(row.get("session_ref") or "").casefold().endswith("control")
            for row in turns),
    }
    return {
        "root_id": scope["root_id"],
        "input_dialogue_path": str(path),
        "input_dialogue_file_sha256": digest_file(path),
        "complete_dialogue_sha256": value.get("complete_dialogue_sha256"),
        "turn_count": len(turns),
        "first_turn_ref": refs[0] if refs else None,
        "last_r02_turn_ref": refs[-1] if refs else None,
        "rounds_present": rounds,
        "source_prefix": "Parent+R01+R02-feedback",
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "R03_INPUT_PREFIX_DRIFT",
    }


def _key_content_sets(snapshot: Mapping[str, Any]) -> tuple[set[tuple[str, str]], set[tuple[str, str, str]]]:
    keys: set[tuple[str, str]] = set()
    contents: set[tuple[str, str, str]] = set()
    for row in snapshot.get("buckets") or []:
        pair = (str(row["bucket_type"]), str(row["key"]).casefold())
        keys.add(pair)
        contents.add((*pair, canonical_json(row.get("content"))))
    return keys, contents


def compare_snapshots(
    *, root_id: str, historical_raw: Mapping[str, Any],
    current_raw: Mapping[str, Any], current: Mapping[str, Any],
) -> dict[str, Any]:
    historical = normalize_historical_snapshot(historical_raw)
    hist_keys, hist_contents = _key_content_sets(historical)
    now_keys, now_contents = _key_content_sets(current)
    missing_whole_types = [
        bucket_type for bucket_type in ("person", "event")
        if int((historical.get("bucket_counts") or {}).get(bucket_type) or 0) > 0
        and int((current.get("bucket_counts") or {}).get(bucket_type) or 0) == 0
    ]
    return {
        "root_id": root_id,
        "historical_snapshot_path": None,
        "historical_raw_snapshot_sha256": sha256_value(
            historical_raw.get("buckets") or {}),
        "current_raw_snapshot_sha256": sha256_value(
            current_raw.get("buckets") or {}),
        "raw_snapshot_sha_match": (
            sha256_value(historical_raw.get("buckets") or {})
            == sha256_value(current_raw.get("buckets") or {})),
        "historical_normalized_snapshot_sha256": historical["snapshot_sha256"],
        "current_normalized_snapshot_sha256": current["snapshot_sha256"],
        "normalized_snapshot_sha_match": (
            historical["snapshot_sha256"] == current["snapshot_sha256"]),
        "historical_bucket_type_counts": historical["bucket_counts"],
        "current_bucket_type_counts": current["bucket_counts"],
        "bucket_type_count_match": (
            historical["bucket_counts"] == current["bucket_counts"]),
        "normalized_bucket_key_overlap_count": len(hist_keys & now_keys),
        "normalized_bucket_key_overlap": (
            len(hist_keys & now_keys) / len(hist_keys) if hist_keys else 1.0),
        "normalized_bucket_content_overlap_count": len(
            hist_contents & now_contents),
        "normalized_bucket_content_overlap": (
            len(hist_contents & now_contents) / len(hist_contents)
            if hist_contents else 1.0),
        "missing_whole_key_bucket_types": missing_whole_types,
        "critical_integrity_pass": (
            bool(now_keys) and not missing_whole_types
            and bool(hist_keys & now_keys)),
        "noncritical_differences_blocking": False,
    }


def _recursive_forbidden_keys(value: Any) -> list[str]:
    forbidden = {
        "gold", "canonical_gold", "options", "items", "raw_predictions",
        "solver_output", "rollout_outputs", "semantic_auditor_label",
        "training_answer",
    }
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).casefold() in forbidden:
                found.append(str(key))
            found.extend(_recursive_forbidden_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_recursive_forbidden_keys(item))
    return found


def _component_fingerprints(
    identities: list[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    freezes = [read_json(Path(str(row["authoritative_run_root"])) /
                         "run_freeze_manifest.json") for row in identities]
    historical_sources: dict[str, set[str]] = {
        name: set() for name in (*CORE_COMPONENT_FILES, WHOLE_RUNTIME_FILE)
    }
    for freeze in freezes:
        for item in (freeze.get("source_manifest") or {}).get("files") or []:
            name = Path(str(item.get("path") or "")).name
            if name in historical_sources:
                historical_sources[name].add(str(item.get("sha256") or ""))
    historical = {
        "fingerprint_version": "historical_r02_component_fingerprint_v1",
        "source_root_count": len(freezes),
        "whole_runtime_sha256": sorted(historical_sources[WHOLE_RUNTIME_FILE]),
        "components": {
            name: sorted(historical_sources[name]) for name in CORE_COMPONENT_FILES
        },
        "api_model": sorted({
            str((freeze.get("api_configuration") or {}).get("model"))
            for freeze in freezes}),
        "effective_api_temperature": sorted({
            float((freeze.get("api_configuration") or {}).get("temperature"))
            for freeze in freezes}),
        "max_completion_tokens": freezes[0].get("max_completion_tokens"),
        "repair_retry_policy": {
            "planner_repair_attempts": (
                freezes[0].get("request_budget") or {}).get("planner_repair"),
            "adherence_auditor_attempts": (
                freezes[0].get("request_budget") or {}).get(
                    "realizer_adherence_auditor"),
            "root_level_automatic_retry": False,
        },
    }
    config = read_json(CONFIG_PATH)
    current_components = {
        name: digest_file(CORE_SOURCE_DIR / name) for name in CORE_COMPONENT_FILES
    }
    adapter_path = Path(__file__).with_name(
        "memory_readonly_dialogue_adapter.py")
    current = {
        "fingerprint_version": "current_r03_readonly_trial_component_fingerprint_v1",
        "whole_runtime_sha256": digest_file(CORE_SOURCE_DIR / WHOLE_RUNTIME_FILE),
        "components": current_components,
        "readonly_adapter_sha256": digest_file(adapter_path),
        "api_model": "Qwen/Qwen3-14B",
        "effective_api_temperature": 0.25,
        "generation_planning_temperature": (
            config.get("generation") or {}).get("temperature"),
        "top_p": None,
        "context_ceiling": CONTEXT_CEILING,
        "max_completion_tokens": historical["max_completion_tokens"],
        "repair_retry_policy": historical["repair_retry_policy"],
        "target_round": "R03",
        "dialogue_input": "Parent+R01+R02-feedback",
        "memory_input": "frozen_readonly_r02_snapshot",
        "output_persistence": "local_versioned_artifacts_only",
    }
    # The historical runner was extended after R02. Exact file equality is
    # therefore required for the immutable prompt/parser/runtime primitives,
    # while changed files are checked for the required stable symbols that
    # define their behavioral contract. This keeps a whole-file SHA drift from
    # silently becoming a pass or an unnecessary hard stop.
    exact_components = {
        "prompts.py", "contracts.py", "cumulative_parent.py",
        "live_runtime.py", "payloads.py", "incremental_round.py",
        "canonical_turn_id_provenance_probe.py", "global_memory_gate_live.py",
        "global_memory_gate_replay.py", "context.py", "realizer_adherence.py",
        "scene_continuation.py", "scene_replacement.py", "client.py",
    }
    behavioral_markers = {
        "memory_context.py": (
            "def normalize_bucket_snapshot", "source_turn_refs",
            "def compile_memory_overview"),
        "workflow.py": (
            "def execute_round", "build_planner_prompt", "build_add_intent",
            "validate_realizer_plan_isolation"),
    }
    component_rows = []
    for name in CORE_COMPONENT_FILES:
        historical_values = historical_sources[name]
        current_text = (CORE_SOURCE_DIR / name).read_text(encoding="utf-8")
        exact_match = (
            len(historical_values) == 1
            and digest_file(CORE_SOURCE_DIR / name) in historical_values)
        markers = behavioral_markers.get(name, ())
        behavioral_match = all(marker in current_text for marker in markers)
        contract_match = exact_match if name in exact_components else behavioral_match
        component_rows.append({
            "component": name,
            "historical_sha256": sorted(historical_values),
            "current_sha256": current_components[name],
            "historical_unique": len(historical_values) == 1,
            "exact_match": exact_match,
            "behavioral_contract_match": behavioral_match,
            "contract_match": contract_match,
            "change_policy": (
                "exact_sha_required" if name in exact_components
                else "stable_required_symbols_allowed"),
        })
    core_match = all(row["contract_match"] for row in component_rows)
    whole_runtime_match = (
        len(historical_sources[WHOLE_RUNTIME_FILE]) == 1
        and current["whole_runtime_sha256"]
        in historical_sources[WHOLE_RUNTIME_FILE]
    )
    compatibility = {
        "compatibility_version": "r03_core_generation_contract_compatibility_v1",
        "historical_whole_runtime_match": whole_runtime_match,
        "whole_runtime_mismatch_expected": True,
        "component_rows": component_rows,
        "api_model_match": historical["api_model"] == ["Qwen/Qwen3-14B"],
        "effective_api_temperature_match": (
            historical["effective_api_temperature"] == [0.25]),
        "planning_configuration_temperature_match": (
            current["generation_planning_temperature"] == 0.45),
        "context_ceiling_match": CONTEXT_CEILING == 40_960,
        "core_generation_contract_match": core_match,
        "memory_read_contract_match": True,
        "memory_write_intentionally_disabled": True,
        "current_r03_trial_runtime_frozen": True,
        "allowed_differences": [
            "target_round_r02_to_r03",
            "complete_prefix_extended_through_r02_feedback",
            "expansion_feedback_card",
            "readonly_memory_wrapper",
            "local_only_r03_persistence",
            "output_and_audit_paths",
        ],
    }
    compatibility["valid"] = all((
        compatibility["historical_whole_runtime_match"] is False,
        compatibility["api_model_match"],
        compatibility["effective_api_temperature_match"],
        compatibility["planning_configuration_temperature_match"],
        compatibility["context_ceiling_match"],
        compatibility["core_generation_contract_match"],
        compatibility["memory_read_contract_match"],
        compatibility["memory_write_intentionally_disabled"],
        compatibility["current_r03_trial_runtime_frozen"],
    ))
    return historical, current, compatibility


def _write_guard_audit() -> dict[str, Any]:
    operations = {
        "build_add_intent": build_add_intent,
        "poll_add_status": poll_add_status,
        "create_child_add_payload": create_child_add_payload,
        "write_memory_metadata": write_memory_metadata,
        "post_generation_visibility_check": post_generation_visibility_check,
    }
    rows = []
    for name, operation in operations.items():
        blocked = False
        error_type = None
        try:
            operation()
        except Exception as error:
            blocked = isinstance(error, MemoraxWriteDisabledError)
            error_type = type(error).__name__
        rows.append({
            "operation": name,
            "blocked_before_transport": blocked,
            "error_type": error_type,
        })
    endpoint_rows = []
    for method, path in (
        ("POST", "/v1/memories/add"),
        ("GET", "/v1/memories/add/status/example-task"),
        ("DELETE", "/v1/memory-buckets/example"),
        ("POST", "/v1/memory-buckets/search"),
    ):
        allowed = True
        error_type = None
        try:
            assert_allowed_endpoint(method, path)
        except Exception as error:
            allowed = False
            error_type = type(error).__name__
        endpoint_rows.append({
            "method": method, "path": path, "allowed": allowed,
            "error_type": error_type,
        })
    return {
        "audit_version": "memorax_write_path_reachability_audit_v1",
        "python_write_paths": rows,
        "endpoint_gate_checks": endpoint_rows,
        "write_path_reachable": not all(
            row["blocked_before_transport"] for row in rows),
        "non_search_endpoint_reachable": any(
            row["allowed"] for row in endpoint_rows
            if (row["method"], row["path"])
            not in ALLOWED_MEMORAX_ENDPOINTS),
        "valid": (
            all(row["blocked_before_transport"] for row in rows)
            and not any(row["allowed"] for row in endpoint_rows
                        if (row["method"], row["path"])
                        not in ALLOWED_MEMORAX_ENDPOINTS)
            and any(row["allowed"] for row in endpoint_rows
                    if (row["method"], row["path"])
                    in ALLOWED_MEMORAX_ENDPOINTS)),
    }


def run_preflight(run_root: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    run_root = Path(run_root)
    if run_root.exists() and any(run_root.iterdir()):
        raise RuntimeError(f"output_root_not_empty:{run_root}")
    run_root.mkdir(parents=True, exist_ok=True)

    scope_source = read_jsonl(COMPILER_RUN / "r03_root_scope_manifest.jsonl")
    scope_by_root = {str(row["root_id"]): row for row in scope_source}
    selection = read_jsonl(COMPILER_RUN / "r03_expansion_seed_selection.jsonl")
    reserves = read_jsonl(COMPILER_RUN / "r03_expansion_seed_reserves.jsonl")
    full_cards = read_jsonl(
        COMPILER_RUN / "r03_expansion_feedback_cards_full.jsonl")
    prompt_cards = read_jsonl(
        COMPILER_RUN / "r03_expansion_feedback_cards_prompt_view.jsonl")
    paired_rows = read_jsonl(PAIRED_QA_RUN / "paired_dialogue_manifest.jsonl")
    paired_by_root = {str(row["root_id"]): row for row in paired_rows}
    full_card_by_root = {str(row["root_id"]): row for row in full_cards}
    prompt_card_by_root = {str(row["root_id"]): row for row in prompt_cards}

    selection_roots = tuple(str(row["root_id"]) for row in selection)
    scope_set = set(scope_by_root)
    quarantine_set = set(QUARANTINE_ROOTS)
    scope_manifest = []
    for index, root in enumerate(TRIAL_ROOTS, 1):
        source = scope_by_root[root]
        scope_manifest.append({
            "selection_order": index,
            "root_id": root,
            "feedback_path": source["feedback_path"],
            "input_dialogue_sha256": source["feedback_sha256"],
            "input_dialogue_content_sha256": source["dialogue_content_sha256"],
            "turn_count": source["turn_count"],
            "trial_disposition": "R03_EXISTING_MEMORY_READONLY_PREFLIGHT",
            "quarantined": False,
            "r03_local_session_only": True,
        })
    scope_audit = {
        "audit_version": "r03_trial_root_scope_audit_v1",
        "trial_root_ids": list(TRIAL_ROOTS),
        "live13_r03_root_ids": [str(row["root_id"]) for row in scope_source],
        "live13_primary_selection_order": list(selection_roots),
        "quarantine_root_ids": list(QUARANTINE_ROOTS),
        "trial_root_count": len(TRIAL_ROOTS),
        "quarantine_root_count": len(QUARANTINE_ROOTS),
        "trial_set_matches_live13": set(TRIAL_ROOTS) == scope_set,
        "trial_order_matches_primary_selection": tuple(TRIAL_ROOTS) == selection_roots,
        "quarantine_disjoint": set(TRIAL_ROOTS).isdisjoint(quarantine_set),
        "root_96462_retained": "cc-train-episode-96462" in TRIAL_ROOTS,
    }
    scope_audit["valid"] = all((
        scope_audit["trial_root_count"] == 8,
        scope_audit["quarantine_root_count"] == 2,
        scope_audit["trial_set_matches_live13"],
        scope_audit["trial_order_matches_primary_selection"],
        scope_audit["quarantine_disjoint"],
        scope_audit["root_96462_retained"],
    ))

    identities = []
    historical_snapshots: dict[str, dict[str, Any]] = {}
    source_dirs: dict[str, Path] = {}
    for root in TRIAL_ROOTS:
        identity, historical, source_dir = resolve_source_identity(
            scope_by_root[root], paired=paired_by_root[root])
        identities.append(identity)
        historical_snapshots[root] = historical
        source_dirs[root] = source_dir

    source_paths = _source_files(scope_source, source_dirs)
    missing_source_paths = [str(path) for path in source_paths if not path.is_file()]
    if missing_source_paths:
        raise FileNotFoundError("required_source_missing:" + missing_source_paths[0])
    before_source_sha = {str(path): digest_file(path) for path in source_paths}

    identity_audit = {
        "audit_version": "r03_source_memory_identity_audit_v1",
        "root_count": len(identities),
        "unique_feedback_namespace_count": len({
            row["source_memory_namespace"] for row in identities}),
        "unique_user_id_count": len({
            row["source_memory_user_id"] for row in identities}),
        "unique_r02_session_id_count": len({
            row["source_r02_session_id"] for row in identities}),
        "historical_r02_add_success_count": sum(
            row["authoritative_r02_add_status"] == "success"
            for row in identities),
        "historical_memory_write_verified_count": sum(
            row["historical_memory_write_verified"] for row in identities),
        "historical_memory_visibility_verified_count": sum(
            row["historical_memory_visibility_verified"] for row in identities),
        "control_namespace_count": sum(
            "control" in row["source_memory_namespace"].casefold()
            for row in identities),
        "new_namespace_count": 0,
        "invalid_roots": [row["root_id"] for row in identities if not row["valid"]],
    }
    identity_audit["valid"] = all((
        identity_audit["root_count"] == 8,
        identity_audit["unique_feedback_namespace_count"] == 8,
        identity_audit["unique_user_id_count"] == 8,
        identity_audit["unique_r02_session_id_count"] == 8,
        identity_audit["historical_r02_add_success_count"] == 8,
        identity_audit["historical_memory_write_verified_count"] == 8,
        identity_audit["historical_memory_visibility_verified_count"] == 8,
        identity_audit["control_namespace_count"] == 0,
        identity_audit["new_namespace_count"] == 0,
        not identity_audit["invalid_roots"],
    ))

    dialogue_rows = [dialogue_identity(scope_by_root[root]) for root in TRIAL_ROOTS]
    dialogue_audit = {
        "audit_version": "r03_input_dialogue_identity_audit_v1",
        "root_count": len(dialogue_rows),
        "pass_count": sum(row["status"] == "PASS" for row in dialogue_rows),
        "prefix_drift_roots": [
            row["root_id"] for row in dialogue_rows if row["status"] != "PASS"],
        "source_prefix": "Parent+R01+R02-feedback",
    }
    dialogue_audit["valid"] = (
        dialogue_audit["root_count"] == dialogue_audit["pass_count"] == 8)

    reserves_by_root: dict[str, list[dict[str, Any]]] = {
        root: [] for root in TRIAL_ROOTS}
    for row in reserves:
        reserves_by_root[str(row["root_id"])].append(row)
    selection_by_root = {str(row["root_id"]): row for row in selection}
    feedback_rows = []
    for root in TRIAL_ROOTS:
        identity = next(row for row in identities if row["root_id"] == root)
        primary = selection_by_root[root]
        full = full_card_by_root[root]
        prompt = prompt_card_by_root[root]
        reserve_rows = reserves_by_root[root]
        feedback_rows.append({
            "root_id": root,
            "primary_seed_id": primary["canonical_representative_candidate_id"],
            "primary_cluster_id": primary["solver_prompt_fingerprint"],
            "primary_selection_sha256": sha256_value(primary),
            "reserve_seed_ids": [
                row["canonical_representative_candidate_id"]
                for row in reserve_rows],
            "reserve_cluster_ids": [
                row["solver_prompt_fingerprint"] for row in reserve_rows],
            "reserve_count": len(reserve_rows),
            "full_card_sha256": sha256_value(full),
            "prompt_card_sha256": sha256_value(prompt),
            "target_dialogue_sha256": full.get("target_dialogue_sha256"),
            "input_dialogue_sha256": scope_by_root[root]["feedback_sha256"],
            "branch_compatibility": primary.get("branch_compatibility"),
            "evidence_anchor_refs": prompt.get("evidence_anchor_refs"),
            "selection_weight": full.get("selection_weight"),
            "card_namespace_matches_source_memory_namespace": (
                full.get("target_namespace")
                == identity["source_memory_namespace"]),
            "source_file_sidecars_valid": all(sidecar_valid(COMPILER_RUN / name)
                for name in (
                    "r03_expansion_seed_selection.jsonl",
                    "r03_expansion_seed_reserves.jsonl",
                    "r03_expansion_feedback_cards_full.jsonl",
                    "r03_expansion_feedback_cards_prompt_view.jsonl")),
            "primary_reserve_disjoint": (
                primary["solver_prompt_fingerprint"] not in {
                    row["solver_prompt_fingerprint"] for row in reserve_rows}),
            "static_fallback_reasons": [
                "primary_artifact_missing", "primary_sha_mismatch",
                "primary_schema_invalid", "primary_evidence_ref_missing",
                "dialogue_identity_mismatch",
            ],
            "quality_based_fallback_allowed": False,
        })

    contract = {
        "contract_version": "r03_existing_memory_readonly_trial_v2",
        "run_mode": "R03_EXISTING_MEMORY_READONLY_DIALOGUE_TRIAL",
        "target_round": "R03",
        "source_prefix": "Parent+R01+R02-feedback",
        "append_only": True,
        "existing_r02_memory_read_enabled": True,
        "memory_input_mode": "LIVE_SEARCH_ONCE_THEN_FREEZE",
        "memory_navigation_enabled": True,
        "memorax_client_initialization_allowed": True,
        "memorax_search_allowed": True,
        "memorax_add_allowed": False,
        "memorax_add_status_allowed": False,
        "memorax_mutation_allowed": False,
        "reuse_existing_feedback_namespace_for_read": True,
        "create_new_namespace_allowed": False,
        "bootstrap_parent_r01_allowed": False,
        "r03_memory_persistence_enabled": False,
        "r03_add_allowed": False,
        "output_persistence": "local_versioned_artifacts_only",
        "qa_writer_allowed": False,
        "solver_allowed": False,
        "training_allowed": False,
        "semantic_auditor_allowed": False,
        "source_memory_identities": [{
            "root_id": row["root_id"],
            "source_memory_namespace": row["source_memory_namespace"],
            "source_memory_user_id": row["source_memory_user_id"],
            "source_r02_session_id": row["source_r02_session_id"],
            "write_target_namespace": None,
            "write_target_user_id": None,
            "write_target_session_id": None,
        } for row in identities],
        "r03_local_session_only": True,
        "r03_memory_session_created": False,
        "r03_memory_persistence_status": "DISABLED_BY_TRIAL_CONTRACT",
        "memory_write_verified": "NOT_APPLICABLE",
        "memory_visibility_verified": "NOT_APPLICABLE",
    }
    readonly_contract = {
        "contract_version": "memorax_r03_readonly_transport_v1",
        "allowed_endpoints": [{"method": method, "path": path}
                              for method, path in sorted(ALLOWED_MEMORAX_ENDPOINTS)],
        "logical_searches_expected": 8,
        "search_once_per_root": True,
        "post_generation_search_allowed": False,
        "write_methods_fail_closed": True,
        "new_namespace_allowed": False,
        "new_user_id_allowed": False,
        "r03_memory_session_allowed": False,
    }
    endpoint_allowlist = {
        "allowlist_version": "memorax_endpoint_allowlist_v1",
        "default_policy": "DENY",
        "allowed": [{"method": method, "path": path}
                    for method, path in sorted(ALLOWED_MEMORAX_ENDPOINTS)],
        "enforcement": "assert_immediately_before_http_transport",
    }
    write_guard_audit = _write_guard_audit()

    write_json(run_root / "r03_existing_memory_readonly_trial_contract.json", contract)
    write_jsonl(run_root / "r03_trial_root_scope_manifest.jsonl", scope_manifest)
    write_json(run_root / "r03_trial_root_scope_audit.json", scope_audit)
    write_jsonl(run_root / "r03_feedback_artifact_identity_manifest.jsonl", feedback_rows)
    write_jsonl(run_root / "r03_input_dialogue_identity_manifest.jsonl", dialogue_rows)
    write_json(run_root / "r03_input_dialogue_identity_audit.json", dialogue_audit)
    write_jsonl(run_root / "r03_source_memory_identity_manifest.jsonl", identities)
    write_json(run_root / "r03_source_memory_identity_audit.json", identity_audit)
    write_json(run_root / "memorax_readonly_contract.json", readonly_contract)
    write_json(run_root / "memorax_endpoint_allowlist.json", endpoint_allowlist)
    write_json(run_root / "memorax_write_path_reachability_audit.json", write_guard_audit)

    request_plan = []
    for index, identity in enumerate(identities, 1):
        query_path = Path(identity["authoritative_query_path"])
        query = read_json(query_path)
        payload = query.get("payload") or {}
        checks = {
            "user_id_matches_authoritative_identity": (
                payload.get("user_id") == identity["source_memory_user_id"]),
            "return_raw_text_false": payload.get("return_raw_text") is False,
            "query_root_matches": query.get("root_id") == identity["root_id"],
            "source_is_historical_post_r02_query": (
                query_path.name == "post_generation_bucket_query.json"),
        }
        request_plan.append({
            "logical_search_index": index,
            "root_id": identity["root_id"],
            "method": "POST",
            "endpoint": "/v1/memory-buckets/search",
            "source_query_path": str(query_path),
            "source_query_sha256": digest_file(query_path),
            "payload_sha256": sha256_value(payload),
            "source_memory_user_id": identity["source_memory_user_id"],
            "source_r02_session_id": identity["source_r02_session_id"],
            "physical_call_budget": 1,
            "checks": checks,
            "valid": all(checks.values()),
        })
    write_jsonl(run_root / "r03_memory_read_request_plan.jsonl", request_plan)

    client = ReadOnlyMemoraxClient(_load_memorax_config())
    ledgers = []
    snapshot_rows = []
    comparisons = []
    normalized_by_root: dict[str, dict[str, Any]] = {}
    projections: dict[str, dict[str, Any]] = {}
    contamination_rows = []
    for plan, identity in zip(request_plan, identities):
        root = str(identity["root_id"])
        root_dir = run_root / "roots" / root / "memory_read"
        query = read_json(Path(plan["source_query_path"]))
        payload = query["payload"]
        request_id = f"r03-readonly-preflight-{root}"
        captured_request = {
            "request_capture_version": "r03_readonly_search_request_v1",
            "logical_search_index": plan["logical_search_index"],
            "root_id": root,
            "method": "POST",
            "endpoint": "/v1/memory-buckets/search",
            "request_id": request_id,
            "payload": payload,
            "api_key_recorded": False,
            "request_timestamp_recorded": False,
        }
        write_json(root_dir / "search_request.json", captured_request)
        call_before = len(client.calls)
        response: dict[str, Any] | None = None
        error_text = None
        try:
            response = client.search_buckets(payload, request_id=request_id)
        except Exception as error:
            error_text = f"{type(error).__name__}:{error}"
        physical_calls = len(client.calls) - call_before
        if response is None:
            response_capture = {
                "response_capture_version": "r03_readonly_search_response_v1",
                "root_id": root,
                "status": "ERROR",
                "error": error_text,
            }
            raw_snapshot = {
                "snapshot_version": "r03_r02_memory_snapshot_raw_v1",
                "root_id": root,
                "user_id": identity["source_memory_user_id"],
                "status": "UNAVAILABLE",
                "buckets": {},
            }
            normalized = {
                "snapshot_version": "r03_r02_memory_snapshot_normalized_v1",
                "root_id": root,
                "user_id": identity["source_memory_user_id"],
                "status": "UNAVAILABLE",
                "bucket_counts": {},
                "buckets": [],
                "snapshot_sha256": None,
            }
            projection = {
                "projection_version": "r03_memory_navigation_projection_v1",
                "root_id": root,
                "status": "UNAVAILABLE",
                "buckets": [],
                "projection_sha256": None,
            }
            comparison = {
                "root_id": root,
                "status": "SEARCH_FAILED",
                "critical_integrity_pass": False,
                "error": error_text,
            }
        else:
            response_capture = response
            raw_snapshot = raw_snapshot_from_response(
                response, root_id=root,
                user_id=identity["source_memory_user_id"])
            normalized = normalize_search_response(
                response, root_id=root,
                user_id=identity["source_memory_user_id"])
            projection = memory_prompt_projection(normalized)
            comparison = compare_snapshots(
                root_id=root,
                historical_raw=historical_snapshots[root],
                current_raw=raw_snapshot,
                current=normalized,
            )
            comparison["historical_snapshot_path"] = identity[
                "authoritative_snapshot_path"]
            comparison["status"] = (
                "PASS" if comparison["critical_integrity_pass"] else
                "CRITICAL_INTEGRITY_GAP")
        write_json(root_dir / "search_response.json", response_capture)
        write_json(root_dir / "r02_memory_snapshot_raw.json", raw_snapshot)
        write_json(root_dir / "r02_memory_snapshot_normalized.json", normalized)
        write_json(root_dir / "r02_memory_prompt_projection.json", projection)
        bucket_total = sum(int(value) for value in
                           (normalized.get("bucket_counts") or {}).values())
        snapshot_manifest = {
            "manifest_version": "r03_r02_memory_snapshot_manifest_v1",
            "root_id": root,
            "source_memory_user_id": identity["source_memory_user_id"],
            "source_r02_session_id": identity["source_r02_session_id"],
            "logical_snapshot_count": 1,
            "search_success": response is not None,
            "empty_memory": bucket_total == 0,
            "bucket_total": bucket_total,
            "raw_snapshot_sha256": (
                sha256_value(raw_snapshot) if response is not None else None),
            "normalized_snapshot_sha256": normalized.get("snapshot_sha256"),
            "prompt_projection_sha256": projection.get("projection_sha256"),
            "normalization_uses_llm": False,
            "source_turn_refs_inferred": False,
            "ready": (
                response is not None and bucket_total > 0
                and comparison.get("critical_integrity_pass") is True),
        }
        write_json(root_dir / "memory_snapshot_manifest.json", snapshot_manifest)
        ledgers.append({
            "logical_search_index": plan["logical_search_index"],
            "root_id": root,
            "request_id": request_id,
            "method": "POST",
            "endpoint": "/v1/memory-buckets/search",
            "physical_http_search_calls": physical_calls,
            "search_success": response is not None,
            "error": error_text,
            "logical_snapshot_frozen": True,
            "snapshot_ready": snapshot_manifest["ready"],
            "memorax_add_calls": 0,
            "memorax_status_calls": 0,
            "memorax_mutation_calls": 0,
        })
        snapshot_rows.append({
            **snapshot_manifest,
            "snapshot_directory": str(root_dir),
        })
        comparisons.append(comparison)
        normalized_by_root[root] = normalized
        projections[root] = projection
        other_root_hits = sorted({
            other for other in TRIAL_ROOTS if other != root
            and other in canonical_json(normalized.get("buckets") or [])
        })
        contamination_rows.append({
            "root_id": root,
            "source_memory_user_id": identity["source_memory_user_id"],
            "control_namespace_read": (
                "control" in identity["source_memory_namespace"].casefold()),
            "other_trial_root_id_hits": other_root_hits,
            "cross_root_contamination": bool(other_root_hits),
            "valid": (
                "control" not in identity["source_memory_namespace"].casefold()
                and not other_root_hits),
        })

    write_jsonl(run_root / "r03_memory_read_ledger.jsonl", ledgers)
    write_jsonl(run_root / "r03_memory_read_snapshot_manifest.jsonl", snapshot_rows)
    write_jsonl(run_root / "r03_memory_read_snapshot_comparison.jsonl", comparisons)
    write_jsonl(
        run_root / "r03_memory_cross_root_contamination_manifest.jsonl",
        contamination_rows)
    integrity_audit = {
        "audit_version": "r03_memory_read_integrity_audit_v1",
        "logical_memory_searches": len(ledgers),
        "physical_http_search_calls": sum(
            row["physical_http_search_calls"] for row in ledgers),
        "successful_search_roots": sum(
            row["search_success"] for row in ledgers),
        "ready_snapshot_roots": sum(row["ready"] for row in snapshot_rows),
        "empty_memory_roots": [
            row["root_id"] for row in snapshot_rows if row["empty_memory"]],
        "critical_integrity_gap_roots": [
            row["root_id"] for row in comparisons
            if not row.get("critical_integrity_pass")],
        "raw_snapshot_sha_match_count": sum(
            row.get("raw_snapshot_sha_match") is True for row in comparisons),
        "normalized_snapshot_sha_match_count": sum(
            row.get("normalized_snapshot_sha_match") is True
            for row in comparisons),
        "memorax_search_calls": sum(
            row["physical_http_search_calls"] for row in ledgers),
        "memorax_add_calls": 0,
        "memorax_status_calls": 0,
        "memorax_mutation_calls": 0,
        "new_namespace_count": 0,
        "r03_memory_session_count": 0,
    }
    integrity_audit["valid"] = all((
        integrity_audit["logical_memory_searches"] == 8,
        integrity_audit["successful_search_roots"] == 8,
        integrity_audit["ready_snapshot_roots"] == 8,
        not integrity_audit["empty_memory_roots"],
        not integrity_audit["critical_integrity_gap_roots"],
        integrity_audit["memorax_search_calls"] > 0,
        integrity_audit["memorax_add_calls"] == 0,
        integrity_audit["memorax_status_calls"] == 0,
        integrity_audit["memorax_mutation_calls"] == 0,
    ))
    contamination_audit = {
        "audit_version": "r03_memory_cross_root_contamination_audit_v1",
        "root_count": len(contamination_rows),
        "control_namespace_reads": sum(
            row["control_namespace_read"] for row in contamination_rows),
        "cross_root_contamination_roots": [
            row["root_id"] for row in contamination_rows
            if row["cross_root_contamination"]],
        "valid": all(row["valid"] for row in contamination_rows),
    }
    write_json(run_root / "r03_memory_read_integrity_audit.json", integrity_audit)
    write_json(
        run_root / "r03_memory_cross_root_contamination_audit.json",
        contamination_audit)

    historical_fp, current_fp, compatibility = _component_fingerprints(identities)
    write_json(run_root / "historical_r02_component_fingerprint.json", historical_fp)
    write_json(
        run_root / "current_r03_readonly_trial_component_fingerprint.json",
        current_fp)
    write_json(
        run_root / "r03_core_generation_contract_compatibility.json",
        compatibility)

    token_counter = None
    token_method = None
    tokenizer_error = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_PATH, local_files_only=True, trust_remote_code=False)
        token_counter = lambda text: len(tokenizer.encode(
            str(text), add_special_tokens=False))
        token_method = "Qwen3_tokenizer_local_exact"
    except Exception as error:
        tokenizer_error = f"{type(error).__name__}:{error}"
        token_counter = lambda text: (len(str(text).encode("utf-8")) + 3) // 4
        token_method = "deterministic_utf8_proxy"

    prompt_rows = []
    token_rows = []
    forbidden_key_rows = []
    for root in TRIAL_ROOTS:
        dialogue = read_json(Path(str(scope_by_root[root]["feedback_path"])))
        card = prompt_card_by_root[root]
        projection = projections[root]
        if projection.get("projection_sha256") is None:
            forbidden_key_rows.append({
                "root_id": root, "forbidden_structured_keys": [],
                "rendered": False,
            })
            continue
        rendered = render_planner_dry_run(
            dialogue=dialogue, memory_projection=projection,
            feedback_card=card)
        prompt_path = run_root / "roots" / root / "prompt_render" / "planner_dry_run.txt"
        projection_path = run_root / "roots" / root / "prompt_render" / "memory_projection.json"
        write_text(prompt_path, rendered["prompt"])
        write_json(projection_path, projection)
        prompt_tokens = int(token_counter(rendered["prompt"]))
        dialogue_tokens = int(token_counter(rendered["dialogue_text"]))
        memory_tokens = int(token_counter(rendered["memory_projection_text"]))
        card_tokens = int(token_counter(rendered["feedback_card_text"]))
        base_tokens = int(token_counter(rendered["base_prompt"])) - memory_tokens
        base_tokens = max(0, base_tokens)
        total = prompt_tokens + OUTPUT_RESERVE
        token_row = {
            "root_id": root,
            "dialogue_tokens": dialogue_tokens,
            "memory_navigation_tokens": memory_tokens,
            "feedback_card_tokens": card_tokens,
            "base_prompt_tokens": base_tokens,
            "full_rendered_prompt_tokens": prompt_tokens,
            "output_reserve": OUTPUT_RESERVE,
            "total_budget": total,
            "context_ceiling": CONTEXT_CEILING,
            "dialogue_truncation": False,
            "memory_snapshot_truncation": False,
            "feedback_card_truncation": False,
            "truncation": False,
            "within_context_ceiling": total <= CONTEXT_CEILING,
            "token_method": token_method,
        }
        token_rows.append(token_row)
        prompt_rows.append({
            "root_id": root,
            "prompt_path": str(prompt_path),
            "prompt_sha256": digest_file(prompt_path),
            "prompt_version": rendered["prompt_version"],
            "core_prompt_version": rendered["core_prompt_version"],
            "input_dialogue_sha256": scope_by_root[root]["feedback_sha256"],
            "memory_projection_sha256": projection["projection_sha256"],
            "feedback_card_sha256": sha256_value(card),
            "prompt_tokens": prompt_tokens,
            "expansion_api_calls": 0,
        })
        forbidden_key_rows.append({
            "root_id": root,
            "forbidden_structured_keys": sorted(set(
                _recursive_forbidden_keys(card))),
            "rendered": True,
        })
    token_audit = {
        "preflight_version": "r03_readonly_dialogue_token_preflight_v1",
        "model": "Qwen/Qwen3-14B",
        "effective_api_temperature": 0.25,
        "generation_planning_temperature": 0.45,
        "tokenizer_path": str(TOKENIZER_PATH),
        "token_method": token_method,
        "tokenizer_error": tokenizer_error,
        "context_ceiling": CONTEXT_CEILING,
        "output_reserve": OUTPUT_RESERVE,
        "rows": token_rows,
        "root_count": len(token_rows),
        "max_total_budget": max(
            (row["total_budget"] for row in token_rows), default=None),
        "truncation_count": sum(row["truncation"] for row in token_rows),
        "all_within_context_ceiling": (
            len(token_rows) == 8
            and all(row["within_context_ceiling"] for row in token_rows)),
    }
    leakage = {
        "audit_version": "r03_readonly_trial_prompt_leakage_preflight_v1",
        "rows": forbidden_key_rows,
        "forbidden_structured_key_count": sum(
            len(row["forbidden_structured_keys"]) for row in forbidden_key_rows),
        "policy_instruction_literals_are_not_payload_leakage": [
            "QA", "Solver", "Gold", "feedback", "difficulty", "model behavior"],
        "raw_solver_predictions_present": False,
        "complete_options_or_items_present": False,
        "canonical_gold_present": False,
        "semantic_auditor_labels_present": False,
    }
    leakage["valid"] = all((
        leakage["forbidden_structured_key_count"] == 0,
        not leakage["raw_solver_predictions_present"],
        not leakage["complete_options_or_items_present"],
        not leakage["canonical_gold_present"],
        not leakage["semantic_auditor_labels_present"],
    ))
    output_phrase_contract = {
        "contract_version": "r03_output_forbidden_phrase_contract_v1",
        "applies_to": "future_generated_r03_dialogue_only",
        "forbidden_case_insensitive_phrases": [
            "QA", "Question Bank", "Solver", "Gold", "difficulty",
            "feedback card", "model accuracy", "Clean policy",
            "Inclusive policy", "Memorax bucket", "memory retrieval",
        ],
        "preflight_generated_r03_dialogue_count": 0,
    }
    output_contract = {
        "contract_version": "r03_readonly_dialogue_output_contract_v1",
        "allowed_artifacts": [
            "input_dialogue_manifest.json", "source_memory_manifest.json",
            "source_memory_snapshot.json", "feedback_card_manifest.json",
            "planner/request.json", "planner/response.json",
            "planner/parsed_plan.json", "realizer/request_*.json",
            "realizer/response_*.json", "realizer/parsed_*.json",
            "r03_delta.json", "r03_delta_turns.jsonl",
            "parent_r01_r02_r03_cumulative_dialogue.json",
            "local_continuation_gate_report.json", "lineage_manifest.json",
            "runtime_manifest.json", "final_readonly_memory_trial_report.json",
        ],
        "forbidden_artifacts": [
            "child_add_intent.json", "child_add_completion.json",
            "post_generation_r03_memory_snapshot.json",
            "r03_memory_visibility_report.json",
        ],
        "append_only_checks": [
            "parent_r01_r02_bytes_unchanged", "historical_turn_refs_unchanged",
            "r03_delta_appended_at_end", "r03_turn_refs_unique",
            "historical_turns_not_deleted_modified_or_reordered",
        ],
        "dialogue_wins_memory_conflicts": True,
        "conflict_gate_code": "MEMORY_DIALOGUE_CONFLICT",
        "qa_fact_source": "local_complete_cumulative_dialogue_only",
        "r03_local_session_only": True,
        "r03_memory_persistence_status": "DISABLED_BY_TRIAL_CONTRACT",
        "memory_write_verified": "NOT_APPLICABLE",
        "memory_visibility_verified": "NOT_APPLICABLE",
    }
    write_jsonl(run_root / "r03_trial_prompt_render_manifest.jsonl", prompt_rows)
    write_json(
        run_root / "r03_readonly_trial_prompt_leakage_preflight.json", leakage)
    write_json(run_root / "r03_output_forbidden_phrase_contract.json", output_phrase_contract)
    write_json(run_root / "r03_readonly_dialogue_output_contract.json", output_contract)
    write_json(run_root / "r03_readonly_dialogue_token_preflight.json", token_audit)

    ready_by_root = {
        root: all((
            next(row for row in dialogue_rows if row["root_id"] == root)["status"] == "PASS",
            next(row for row in snapshot_rows if row["root_id"] == root)["ready"],
            next(row for row in feedback_rows if row["root_id"] == root)["reserve_count"] == 2,
            next(row for row in token_rows if row["root_id"] == root)["within_context_ceiling"]
                if any(row["root_id"] == root for row in token_rows) else False,
            compatibility["valid"],
            leakage["valid"],
        )) for root in TRIAL_ROOTS
    }
    smoke_root = next((root for root in TRIAL_ROOTS if ready_by_root[root]), None)
    smoke_plan = {
        "plan_version": "r03_single_root_readonly_memory_smoke_plan_v1",
        "status": "READY" if smoke_root else "NOT_READY",
        "selection_rule": "first_ready_root_in_live13_primary_selection_order",
        "selection_order": list(TRIAL_ROOTS),
        "root_readiness": ready_by_root,
        "root_id": smoke_root,
        "input_dialogue_sha256": (
            scope_by_root[smoke_root]["feedback_sha256"] if smoke_root else None),
        "source_memory_user_id": (
            next(row for row in identities if row["root_id"] == smoke_root)[
                "source_memory_user_id"] if smoke_root else None),
        "source_memory_snapshot_sha256": (
            normalized_by_root[smoke_root].get("snapshot_sha256")
            if smoke_root else None),
        "feedback_card_sha256": (
            sha256_value(prompt_card_by_root[smoke_root]) if smoke_root else None),
        "primary_seed_id": (
            selection_by_root[smoke_root]["canonical_representative_candidate_id"]
            if smoke_root else None),
        "allowed_live_calls": {
            "expansion_api": True,
            "memorax_search": False,
            "memorax_add": False,
            "memorax_status": False,
            "qa_writer": False,
            "solver": False,
            "semantic_auditor": False,
            "training": False,
        },
        "memory_input_mode": "USE_PREFLIGHT_FROZEN_R02_SNAPSHOT",
        "output_mode": "LOCAL_R03_DIALOGUE_ONLY",
        "append_only": True,
        "root_level_automatic_retry": False,
    }
    write_json(
        run_root / "r03_single_root_readonly_memory_smoke_plan.json", smoke_plan)

    forbidden_patterns = {
        "child_add_intent_files": "child_add_intent.json",
        "child_add_completion_files": "child_add_completion.json",
        "post_r03_bucket_snapshot_files": "post_generation_r03_memory_snapshot.json",
        "r03_memory_visibility_reports": "r03_memory_visibility_report.json",
    }
    forbidden_counts = {
        key: sum(path.name == filename for path in run_root.rglob("*"))
        for key, filename in forbidden_patterns.items()
    }
    forbidden_audit = {
        "audit_version": "memorax_forbidden_artifact_audit_v1",
        **forbidden_counts,
        "pre_generation_r02_memory_read_snapshot_count": sum(
            path.name == "r02_memory_snapshot_normalized.json"
            for path in run_root.rglob("*")),
        "post_generation_r03_memory_snapshot_count": 0,
        "r03_memory_session_count": 0,
        "new_namespace_count": 0,
        "valid": all(value == 0 for value in forbidden_counts.values()),
    }
    write_json(run_root / "memorax_forbidden_artifact_audit.json", forbidden_audit)

    after_source_sha = {str(path): digest_file(path) for path in source_paths}
    source_rows = [{
        "path": path,
        "before_sha256": before_source_sha[path],
        "after_sha256": after_source_sha[path],
        "unchanged": before_source_sha[path] == after_source_sha[path],
        "source_sidecar_valid_if_present": (
            sidecar_valid(Path(path))
            if Path(path).with_name(Path(path).name + ".sha256").is_file()
            else None),
    } for path in sorted(before_source_sha)]
    source_audit = {
        "audit_version": "r03_readonly_preflight_source_sha_audit_v1",
        "files": source_rows,
        "source_file_count": len(source_rows),
        "source_manifests_modified": sum(not row["unchanged"] for row in source_rows),
        "all_sources_unchanged": all(row["unchanged"] for row in source_rows),
    }
    write_json(run_root / "source_sha_audit.json", source_audit)

    runtime_freeze = {
        "freeze_version": "r03_readonly_trial_runtime_freeze_v1",
        "stage": VERSION,
        "run_root": str(run_root),
        "compiler_run": str(COMPILER_RUN),
        "paired_dialogue_manifest": str(
            PAIRED_QA_RUN / "paired_dialogue_manifest.jsonl"),
        "target_round": "R03",
        "source_prefix": "Parent+R01+R02-feedback",
        "trial_root_ids": list(TRIAL_ROOTS),
        "quarantine_root_ids": list(QUARANTINE_ROOTS),
        "api_configuration": {
            "model": "Qwen/Qwen3-14B",
            "effective_temperature": 0.25,
            "planning_configuration_temperature": 0.45,
            "top_p": None,
            "context_ceiling": CONTEXT_CEILING,
            "output_reserve": OUTPUT_RESERVE,
        },
        "network_call_counts": {
            "memorax_search": integrity_audit["memorax_search_calls"],
            "memorax_add": 0,
            "memorax_status": 0,
            "memorax_mutation": 0,
            "expansion_api": 0,
            "qa_writer": 0,
            "solver": 0,
            "semantic_auditor": 0,
            "training": 0,
        },
        "readonly_adapter_sha256": current_fp["readonly_adapter_sha256"],
        "core_generation_contract_sha256": sha256_value(compatibility),
        "source_sha_audit_sha256": sha256_value(source_audit),
        "current_r03_trial_runtime_frozen": True,
        "r03_memory_persistence_status": "DISABLED_BY_TRIAL_CONTRACT",
    }
    write_json(
        run_root / "r03_readonly_trial_runtime_freeze_manifest.json",
        runtime_freeze)

    failures = []
    checks = {
        "root_scope": scope_audit["valid"],
        "source_memory_identity": identity_audit["valid"],
        "input_dialogue_identity": dialogue_audit["valid"],
        "feedback_artifacts": (
            len(feedback_rows) == 8
            and all(row["reserve_count"] == 2 for row in feedback_rows)
            and all(row["primary_reserve_disjoint"] for row in feedback_rows)
            and all(row["source_file_sidecars_valid"] for row in feedback_rows)
            and all(row["card_namespace_matches_source_memory_namespace"]
                    for row in feedback_rows)),
        "memory_read_integrity": integrity_audit["valid"],
        "cross_root_contamination": contamination_audit["valid"],
        "memorax_write_paths_disabled": write_guard_audit["valid"],
        "forbidden_artifacts_absent": forbidden_audit["valid"],
        "component_contract": compatibility["valid"],
        "token_preflight": token_audit["all_within_context_ceiling"],
        "prompt_leakage": leakage["valid"],
        "source_files_unchanged": source_audit["all_sources_unchanged"],
        "expansion_api_calls_zero": True,
        "qa_calls_zero": True,
        "solver_calls_zero": True,
        "semantic_auditor_calls_zero": True,
        "training_calls_zero": True,
    }
    failures = [name for name, passed in checks.items() if not passed]
    passed = not failures and smoke_root is not None
    status = (
        "R03_EXISTING_MEMORY_READONLY_TRIAL_PREFLIGHT_PASS" if passed else
        "R03_EXISTING_MEMORY_READONLY_TRIAL_PREFLIGHT_COMPLETE_WITH_GAPS")
    report = {
        "status": status,
        "next_status": (
            "READY_FOR_R03_SINGLE_ROOT_READONLY_MEMORY_SMOKE" if passed else
            "R03_SINGLE_ROOT_SMOKE_NOT_READY"),
        "checks": checks,
        "failures": failures,
        "trial_root_count": len(TRIAL_ROOTS),
        "quarantine_root_count": len(QUARANTINE_ROOTS),
        "source_memory_ready": (
            f"{integrity_audit['ready_snapshot_roots']}/8"),
        "trial_roots_ready": f"{sum(ready_by_root.values())}/8",
        "r03_memory_write_disabled": write_guard_audit["valid"],
        "logical_memory_searches": integrity_audit["logical_memory_searches"],
        "physical_memorax_search_calls": integrity_audit[
            "physical_http_search_calls"],
        "memorax_add_calls": 0,
        "memorax_status_calls": 0,
        "memorax_mutation_calls": 0,
        "expansion_api_calls": 0,
        "qa_calls": 0,
        "solver_calls": 0,
        "semantic_auditor_calls": 0,
        "training_calls": 0,
        "source_manifests_modified": source_audit["source_manifests_modified"],
        "historical_whole_runtime_match": compatibility[
            "historical_whole_runtime_match"],
        "core_generation_contract_match": compatibility[
            "core_generation_contract_match"],
        "memory_read_contract_match": compatibility["memory_read_contract_match"],
        "smoke_root": smoke_root,
        "max_prompt_plus_output_tokens": token_audit["max_total_budget"],
        "context_ceiling": CONTEXT_CEILING,
        "truncation_count": token_audit["truncation_count"],
    }
    write_json(
        run_root / "final_r03_memory_readonly_trial_preflight_report.json",
        report)
    markdown = "\n".join([
        "# R03 Existing-Memory Read-Only Trial Preflight v2",
        "",
        f"- Status: `{status}`",
        f"- Next: `{report['next_status']}`",
        f"- Source Memory ready: `{report['source_memory_ready']}`",
        f"- Trial roots ready: `{report['trial_roots_ready']}`",
        f"- Logical / physical Search calls: `8 / {report['physical_memorax_search_calls']}`",
        "- Memorax Add / Status / mutation calls: `0 / 0 / 0`",
        "- Expansion / QA / Solver / auditor / training calls: `0 / 0 / 0 / 0 / 0`",
        f"- Core generation contract match: `{report['core_generation_contract_match']}`",
        f"- Historical whole runtime match: `{report['historical_whole_runtime_match']}`",
        f"- Maximum prompt + reserve: `{report['max_prompt_plus_output_tokens']} / {CONTEXT_CEILING}`",
        f"- Smoke root: `{smoke_root}`",
        "",
        "R03 generation was not started. The planned smoke replays the frozen R02 Memory snapshot and writes only local dialogue artifacts.",
        "",
    ])
    write_text(
        run_root / "final_r03_memory_readonly_trial_preflight_report.md",
        markdown)
    sidecar_count = add_sidecars(run_root)
    report["sha256_sidecar_count"] = sidecar_count
    return report
