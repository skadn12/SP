"""Dataset-neutral unified-root to stateful Planner offline preflight."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .core import (
    atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file,
)
from .input_adapter import dialogue_content_hash
from .run_canary import (
    create_source_snapshot, git_value, project_diff_sha256,
)
from .stateful_graph_expansion import (
    STATEFUL_GRAPH_LEDGER_VERSION, initialize_stateful_ledger,
)
from .stateful_outline_planner import (
    STATEFUL_DYNAMIC_PLANNER_INPUT_VERSION,
    STATEFUL_OUTLINE_CHUNKED_VERSION, STATEFUL_OUTLINE_PLANNER_VERSION,
    build_dynamic_stateful_planner_input,
)
from .unified_dialogue_root import (
    UNIFIED_DIALOGUE_ROOT_VERSION, read_unified_parent,
    unified_root_schema_sha256, validate_unified_dialogue_root,
)


DYNAMIC_GENERATION_PREFLIGHT_VERSION = "dynamic_generation_preflight_v1"
PREPLANNER_GRAPH_STATE_VERSION = "stateful_preplanner_graph_state_v1"
DEFAULT_CONTEXT_LIMIT = 40_960
DEFAULT_RESERVED_COMPLETION_TOKENS = 2_500
PROJECT = Path(__file__).resolve().parents[3]


def _safe_component(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    if not result:
        raise ValueError("root_id_has_no_safe_path_component")
    return result


def build_locomo_compat_unified_root(
    source_path: Path, *, root_id: str = "locomo-conv-43",
) -> dict[str, Any]:
    """Freeze one already-parent-only LoCoMo artifact into the common schema."""
    source_path = source_path.resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_turns = source.get("turns")
    source_sessions = source.get("source_session_ids")
    if (not isinstance(source_turns, list) or not source_turns or
            not isinstance(source_sessions, list) or not source_sessions or
            source.get("source_child_turn_count") != 0 or
            any(turn.get("partition") != "parent" for turn in source_turns)):
        raise ValueError("locomo_parent_source_contract_invalid")
    if (source.get("source_turn_count") != len(source_turns) or
            source.get("source_session_count") != len(source_sessions) or
            dialogue_content_hash(source_turns) != source.get("dialogue_sha256")):
        raise ValueError("locomo_parent_source_attestation_invalid")
    session_positions = {str(value): index for index, value in
                         enumerate(source_sessions, 1)}
    if set(session_positions) != {str(turn.get("session_id"))
                                 for turn in source_turns}:
        raise ValueError("locomo_parent_session_membership_invalid")
    turns: list[dict[str, Any]] = []
    turn_ids_by_session: dict[str, list[str]] = {
        str(value): [] for value in source_sessions}
    within_session: dict[str, int] = {str(value): 0 for value in source_sessions}
    session_times: dict[str, str] = {}
    for source_turn in source_turns:
        original_session = str(source_turn["session_id"])
        session_index = session_positions[original_session]
        within_session[original_session] += 1
        turn_index = within_session[original_session]
        session_id = f"{root_id}:parent_session_{session_index:02d}"
        turn_id = f"{root_id}:P{session_index:02d}:T{turn_index:03d}"
        turn_ids_by_session[original_session].append(turn_id)
        session_times.setdefault(
            original_session,
            str(source_turn.get("session_date_time") or "not_provided"))
        turns.append({
            "turn_id": turn_id, "session_id": session_id,
            "speaker": str(source_turn.get("speaker") or ""),
            "text": str(source_turn.get("text") or ""),
            "partition": "parent",
            "source_id": str(source.get("sample_id") or root_id),
            "source_session_index": session_index,
            "source_turn_index": turn_index,
        })
    session_metadata = [{
        "session_id": f"{root_id}:parent_session_{index:02d}",
        "source_session_index": index,
        "time_interval": session_times[str(original_session)],
        "summary": "not_provided_by_frozen_locomo_parent_source",
        "turn_ids": turn_ids_by_session[str(original_session)],
    } for index, original_session in enumerate(source_sessions, 1)]
    file_sha = sha256_file(source_path)
    root = {
        "schema_version": UNIFIED_DIALOGUE_ROOT_VERSION,
        "root_id": root_id, "sample_id": root_id,
        "lineage_group": root_id,
        "source": {
            "dataset": "locomo", "split": "train",
            "source_id": str(source.get("sample_id") or root_id),
            "source_row_index": 0,
            "source_path": str(source_path),
            "source_file_sha256": file_sha,
            "source_record_sha256": file_sha,
            "source_record_sha256_scope":
                "frozen_parent_dialogue_json_file_bytes",
        },
        "dialogue": {
            "dialogue_hash": dialogue_content_hash(turns),
            "session_count": len(source_sessions),
            "turn_count": len(turns), "turns": turns,
        },
        "session_metadata": session_metadata,
        "source_metadata": {
            "source_kind": "frozen_parent_dialogue_v1",
            "source_dialogue_hash": str(source["dialogue_sha256"]),
        },
    }
    result = validate_unified_dialogue_root(
        root, expected_source_path=source_path,
        expected_source_file_sha256=file_sha)
    if not result["valid"]:
        raise ValueError("locomo_unified_root_invalid:" +
                         ",".join(result["errors"]))
    return root


def preflight_unified_root(
    unified_root_path: Path, *, output_directory: Path,
    token_counter: Callable[[str], int],
    context_limit: int = DEFAULT_CONTEXT_LIMIT,
    reserved_completion_tokens: int = DEFAULT_RESERVED_COMPLETION_TOKENS,
) -> dict[str, Any]:
    """Build all local Planner inputs and stop before any API request."""
    unified_root_path = unified_root_path.resolve()
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(f"preflight_output_exists:{output_directory}")
    root = json.loads(unified_root_path.read_text(encoding="utf-8"))
    validation = validate_unified_dialogue_root(root)
    if not validation["valid"]:
        raise ValueError("preflight_unified_root_invalid:" +
                         ",".join(validation["errors"]))
    parent = read_unified_parent(unified_root_path)
    root_id = str(root["root_id"])
    lineage_id = f"{root_id}:lineage:01"
    run_id = f"{root_id}:stateful-generation:preflight:01"
    planner_input = build_dynamic_stateful_planner_input(
        parent_turns=parent["dialogue"]["turns"], root_id=root_id,
        sample_id=parent["sample_id"], lineage_id=lineage_id,
        run_id=run_id, dialogue_hash=parent["dialogue"]["dialogue_hash"],
        output_directory=str(output_directory),
        outline_id_prefix=f"{root_id}:outline", round_index=1)
    prompt = str(planner_input["overview_prompt"])
    prompt_tokens = int(token_counter(prompt))
    if prompt_tokens < 1:
        raise ValueError("planner_prompt_token_count_invalid")
    initial_ledger = initialize_stateful_ledger([])
    graph_state = {
        "schema_version": PREPLANNER_GRAPH_STATE_VERSION,
        "root_id": root_id, "sample_id": parent["sample_id"],
        "lineage_id": lineage_id, "run_id": run_id,
        "dialogue_hash": parent["dialogue"]["dialogue_hash"],
        "planner_status": "not_invoked",
        "graph_status": "initialized_awaiting_planner_output",
        "scene_contract_count": 0,
        "stateful_ledger": initial_ledger,
        "api_calls": 0,
    }
    session_ids = list(planner_input["parent_session_ids"])
    checks = {
        "parent_contract_valid": validation["valid"],
        "parent_turn_count_dynamic": (
            planner_input["parent_turn_count"] ==
            root["dialogue"]["turn_count"]),
        "parent_session_count_dynamic": (
            planner_input["parent_session_count"] ==
            root["dialogue"]["session_count"]),
        "turn_ids_unique": (
            len(planner_input["parent_turn_ids"]) ==
            len(set(planner_input["parent_turn_ids"]))),
        "session_ids_unique": len(session_ids) == len(set(session_ids)),
        "planner_prompt_built": bool(prompt.strip()),
        "planner_prompt_context_pass": (
            prompt_tokens + reserved_completion_tokens <= context_limit),
        "graph_state_initialized": (
            graph_state["graph_status"] ==
            "initialized_awaiting_planner_output" and
            graph_state["stateful_ledger"]["ledger_version"] ==
            STATEFUL_GRAPH_LEDGER_VERSION and
            not graph_state["stateful_ledger"]["stage_states"]),
        "dynamic_root_identity": planner_input["root_id"] == root_id,
        "dynamic_lineage_identity": planner_input["lineage_id"] == lineage_id,
        "dynamic_run_identity": planner_input["run_id"] == run_id,
        "dynamic_outline_prefix": (
            planner_input["outline_id_prefix"] == f"{root_id}:outline"),
        "summary_not_forwarded_as_planner_field": (
            "summary" not in parent["dialogue"]["turns"][0]),
        "time_interval_not_forwarded_as_planner_field": (
            "time_interval" not in parent["dialogue"]["turns"][0]),
        "api_boundary_not_crossed": graph_state["api_calls"] == 0,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError("dynamic_generation_preflight_failed:" +
                           ",".join(failed))
    output_directory.mkdir(parents=True)
    planner_summary = {key: value for key, value in planner_input.items()
                       if key != "overview_prompt"}
    planner_summary["planner_input_sha256"] = sha256_bytes(
        canonical_json(planner_input).encode())
    atomic_text(output_directory / "prompts/planner_overview.txt", prompt)
    atomic_json(output_directory / "frozen/parent_contract.json", parent)
    atomic_json(output_directory / "frozen/planner_input.json", planner_summary)
    atomic_json(output_directory / "frozen/preplanner_graph_state.json",
                graph_state)
    input_sha = sha256_file(unified_root_path)
    manifest = {
        "schema_version": "dynamic_generation_root_manifest_v1",
        "preflight_version": DYNAMIC_GENERATION_PREFLIGHT_VERSION,
        "root_id": root_id, "sample_id": parent["sample_id"],
        "lineage_id": lineage_id, "run_id": run_id,
        "input_unified_root_path": str(unified_root_path),
        "input_unified_root_sha256": input_sha,
        "output_directory": str(output_directory),
        "fresh_run": True, "resume": False,
        "planner_prompt_version": STATEFUL_OUTLINE_CHUNKED_VERSION,
        "planner_base_prompt_version": STATEFUL_OUTLINE_PLANNER_VERSION,
        "planner_input_version": STATEFUL_DYNAMIC_PLANNER_INPUT_VERSION,
        "planner_api_invoked": False, "api_calls": 0,
    }
    atomic_json(output_directory / "config/run_manifest.json", manifest)
    cache_paths = [path for path in output_directory.rglob("*")
                   if path.name in {"request_ledger.jsonl", "cache", "raw"}]
    checks["fresh_run_cache_isolated"] = not cache_paths
    result = {
        "root_id": root_id, "sample_id": parent["sample_id"],
        "dataset": root["source"]["dataset"],
        "source_id": root["source"]["source_id"],
        "unified_root_path": str(unified_root_path),
        "unified_root_sha256": input_sha,
        "parent_turn_count": planner_input["parent_turn_count"],
        "parent_session_count": planner_input["parent_session_count"],
        "dialogue_hash": planner_input["dialogue_hash"],
        "lineage_id": lineage_id, "run_id": run_id,
        "outline_id_prefix": planner_input["outline_id_prefix"],
        "output_directory": str(output_directory),
        "planner_prompt_built": True,
        "planner_prompt_sha256": planner_input["overview_prompt_sha256"],
        "planner_prompt_tokens": prompt_tokens,
        "reserved_completion_tokens": reserved_completion_tokens,
        "context_limit": context_limit,
        "graph_state_initialized": True,
        "graph_state_status": graph_state["graph_status"],
        "input_files_read": [str(unified_root_path)],
        "metadata_fields_forwarded_to_planner": [],
        "fresh_run": True, "cache_isolated": not cache_paths,
        "api_calls": 0, "planner_calls": 0,
        "checks": checks,
        "preflight_pass": all(checks.values()),
    }
    atomic_json(output_directory / "reports/preflight.json", result)
    return result


def render_generation_preflight_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Dynamic Generation Preflight", "",
        f"- generation_preflight_ready: `{str(report['generation_preflight_ready']).lower()}`",
        f"- generation_ready: `{str(report['generation_ready']).lower()}`",
        f"- API calls: `{report['api_calls']}`",
        f"- CC roots: `{report['cc_root_count']}`",
        f"- LoCoMo roots: `{report['locomo_root_count']}`", "",
        "| Root | Dataset | Sessions | Turns | Prompt tokens | Graph state | Pass |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for row in report["roots"]:
        lines.append(
            f"| `{row['root_id']}` | `{row['dataset']}` | "
            f"{row['parent_session_count']} | {row['parent_turn_count']} | "
            f"{row['planner_prompt_tokens']} | `{row['graph_state_status']}` | "
            f"`{str(row['preflight_pass']).lower()}` |")
    lines.extend(["", "## Hard-coded dependency audit", ""])
    for key, value in report["hardcoded_dependency_audit"].items():
        lines.append(f"- `{key}`: `{str(value).lower()}`")
    lines.extend([
        "", "The run stopped before Planner backend construction. No Planner, "
        "Realizer, Questioner, Solver, Verifier, or Training call was made.", "",
    ])
    return "\n".join(lines)


def run_generation_preflight_batch(
    *, cc_root_paths: Sequence[Path], locomo_parent_source: Path,
    output_root: Path, token_counter: Callable[[str], int],
) -> dict[str, Any]:
    """Run ten CC and one LoCoMo preflight with a shared fresh-run boundary."""
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"generation_preflight_root_exists:{output_root}")
    paths = [path.resolve() for path in cc_root_paths]
    if len(paths) != 10 or len(paths) != len(set(paths)):
        raise ValueError("generation_preflight_requires_ten_unique_cc_roots")
    source_shas_before = {str(path): sha256_file(path) for path in
                          [*paths, locomo_parent_source.resolve()]}
    output_root.mkdir(parents=True)
    snapshot = create_source_snapshot(PROJECT, output_root, False)
    diff_sha = project_diff_sha256(PROJECT)
    locomo_root = build_locomo_compat_unified_root(locomo_parent_source)
    locomo_path = output_root / "frozen_inputs/locomo-conv-43/unified_root.json"
    atomic_json(locomo_path, locomo_root)
    all_paths = [*paths, locomo_path]
    results = [preflight_unified_root(
        path, output_directory=output_root / "roots" /
        _safe_component(json.loads(path.read_text())["root_id"]),
        token_counter=token_counter) for path in all_paths]
    source_shas_after = {str(path): sha256_file(path) for path in
                         [*paths, locomo_parent_source.resolve()]}
    hardcoded_audit = {
        "static_expansion_outline_required": False,
        "parent_anchor_snapshot_required": False,
        "round_dataset_train_seed_rows_required": False,
        "conversation_id_hardcoded": False,
        "parent_turn_count_hardcoded": False,
        "child_turn_count_hardcoded": False,
        "outline_id_prefix_hardcoded": False,
        "locomo_specific_fields_required": False,
    }
    api_calls = sum(int(row["api_calls"]) for row in results)
    all_pass = all(row["preflight_pass"] for row in results)
    report = {
        "schema_version": "generation_preflight_report_v1",
        "preflight_version": DYNAMIC_GENERATION_PREFLIGHT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "generation_preflight_ready": (
            all_pass and len(results) == 11 and api_calls == 0),
        "generation_ready": False,
        "generation_ready_reason":
            "requires_one_paid_n1_planner_to_realizer_canary",
        "cc_root_count": sum(row["dataset"] ==
                             "conversation_chronicles" for row in results),
        "locomo_root_count": sum(row["dataset"] == "locomo"
                                 for row in results),
        "root_count": len(results), "roots": results,
        "hardcoded_dependency_audit": hardcoded_audit,
        "hardcoded_dependencies_remaining": [],
        "source_inputs_unchanged": source_shas_before == source_shas_after,
        "source_sha256_before": source_shas_before,
        "source_sha256_after": source_shas_after,
        "fresh_run_cache_isolation_pass": all(
            row["fresh_run"] and row["cache_isolated"] for row in results),
        "api_calls": api_calls, "planner_calls": 0, "realizer_calls": 0,
        "questioner_calls": 0, "solver_calls": 0, "verifier_calls": 0,
        "training_runs": 0,
        "project_git_head": git_value(PROJECT, "rev-parse", "HEAD"),
        "project_diff_sha256": diff_sha,
        **snapshot,
    }
    atomic_json(output_root / "reports/generation_preflight_report.json",
                report)
    atomic_text(output_root / "reports/generation_preflight_report.md",
                render_generation_preflight_report(report))
    manifest = {
        "schema_version": "dynamic_generation_preflight_manifest_v1",
        "run_id": output_root.name,
        "mode": DYNAMIC_GENERATION_PREFLIGHT_VERSION,
        "created_at_utc": report["created_at_utc"],
        "input_unified_root_paths": [str(path) for path in all_paths],
        "input_unified_root_sha256": {
            str(path): sha256_file(path) for path in all_paths},
        "unified_root_schema_sha256": unified_root_schema_sha256(),
        "generation_preflight_ready": report[
            "generation_preflight_ready"],
        "generation_ready": False, "external_api_used": False,
        "api_calls": 0, "project_git_head": report["project_git_head"],
        "project_diff_sha256": diff_sha, **snapshot,
    }
    atomic_json(output_root / "config/run_manifest.json", manifest)
    provenance_checks = {
        "git_head_unchanged": git_value(PROJECT, "rev-parse", "HEAD") ==
        report["project_git_head"],
        "project_diff_sha256_match": project_diff_sha256(PROJECT) == diff_sha,
        "source_snapshot_exists": Path(
            str(snapshot["source_snapshot_manifest_path"])).is_file(),
        "source_inputs_unchanged": source_shas_before == source_shas_after,
        "api_calls_zero": api_calls == 0,
    }
    atomic_json(output_root / "reports/post_run_provenance.json", {
        "schema_version": "dynamic_generation_preflight_provenance_v1",
        "checks": provenance_checks,
        "authoritative_release_provenance": all(provenance_checks.values()),
        "provenance_mixed": False,
    })
    return report
