"""Zero-API preflight for a bounded Simple Append + Memorax batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..core import atomic_json, atomic_text, sha256_file
from ..simple_append_v1.live_canary import (
    CONTEXT_LIMIT,
    DEFAULT_API_CONFIG,
    DEFAULT_TOKENIZER,
    PLANNER_MAX_TOKENS,
    ROUND_REALIZER_MAX_TOKENS,
)
from ..simple_append_v1.preflight import run as run_simple_preflight
from ..simple_append_v1.qa_live import (
    DEFAULT_SOLVER_MODEL,
    WRITER_MAX_TOKENS,
)
from .memory_context import compile_memory_overview, normalize_bucket_snapshot
from .payloads import memorax_user_id
from .prompts import (
    build_memory_augmented_mcq_prompt,
    build_memory_augmented_planner_prompt,
    build_memory_augmented_realizer_prompt,
)
from .qa_admission import classify_qa_admission_tier
from .single_root_live import (
    DEFAULT_MEMORAX_CREDENTIALS,
    run_preflight as run_single_root_preflight,
)


BATCH_PREFLIGHT_VERSION = "simple_append_memorax_batch_preflight_v2"
INPUT_MANIFEST_VERSION = "simple_append_memorax_input_manifest_v1"
MAX_BATCH_ROOTS = 11
DEFAULT_BATCH_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "simple_append_memorax_v1_batch_pilot_1")
DEFAULT_REPORT_ROOT = Path(__file__).with_name("reports")
ROOTS_DIR = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-inputs/"
    "conversation_chronicles_v1/roots")
DEFAULT_ROOT_IDS = (
    "cc-train-episode-12596",
    "cc-train-episode-13836",
)


def load_input_manifest(path: Path) -> dict[str, Path]:
    """Load a frozen root-to-input map without copying or rewriting inputs."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(value, Mapping) or
            value.get("manifest_version") != INPUT_MANIFEST_VERSION):
        raise ValueError("memorax_input_manifest_version_invalid")
    roots = value.get("roots")
    if not isinstance(roots, Mapping) or not roots:
        raise ValueError("memorax_input_manifest_roots_required")
    result: dict[str, Path] = {}
    for raw_root_id, raw_path in roots.items():
        root_id = str(raw_root_id)
        candidate = Path(str(raw_path)).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(
                f"memorax_input_manifest_path_missing:{root_id}")
        result[root_id] = candidate
    return result


def _empty_response() -> dict[str, Any]:
    return {"success": True, "data": {"buckets": {
        "person": {}, "location": {}, "event": {},
        "item": {}, "dreaming": {},
    }}}


def _one(
    *, root_id: str, batch_root: Path, api_config: Path,
    tokenizer_path: Path, solver_model: Path, memorax_credentials: Path,
    allow_insecure_http: bool, input_path: Path | None = None,
) -> dict[str, Any]:
    input_path = (Path(input_path).resolve() if input_path is not None else
                  ROOTS_DIR / root_id / "unified_root.json")
    expansion_root = batch_root / "roots" / root_id / "expansion"
    qa_root = batch_root / "roots" / root_id / "qa"
    input_sha_before = sha256_file(input_path) if input_path.is_file() else None
    live = run_single_root_preflight(
        input_path=input_path,
        run_root=expansion_root,
        api_config_path=api_config,
        tokenizer_path=tokenizer_path,
        memorax_credentials_path=memorax_credentials,
        allow_insecure_memorax_http=allow_insecure_http,
        reuse_existing_parent_memory=False,
        wait_for_memory_completion=False,
    )
    simple = run_simple_preflight(parent_path=input_path)
    tail = simple["tail_context"]
    snapshot = normalize_bucket_snapshot(
        _empty_response(), root_id=root_id, user_id=memorax_user_id(root_id))
    overview = compile_memory_overview(
        snapshot,
        recent_dialogue_text="\n".join(
            str(row["text"]) for row in tail["selected_turns"]),
    )
    parent_value = json.loads(input_path.read_text(encoding="utf-8"))
    from .durable_contracts import build_parent_anchor_registry
    planner = build_memory_augmented_planner_prompt(
        tail, overview,
        build_parent_anchor_registry(parent_value["dialogue"]["turns"]))
    realizer = build_memory_augmented_realizer_prompt(
        compiled_plan=simple["compiled_plan"], tail_context=tail,
        memory_overview=overview)
    mcq = build_memory_augmented_mcq_prompt(
        simple["expanded_dialogue"], overview, target_questions=8,
        focus_round_index=1)
    observation_tier = classify_qa_admission_tier({
        "admitted": True, "semantic_evidence_audit_executed": False})
    checks = {
        "input_exists": input_path.is_file(),
        "root_identity": simple["root_id"] == root_id,
        "single_root_live_preflight": live["decision"] == "pass",
        "base_structural_preflight": simple["status"] == "passed",
        "expansion_root_absent": not expansion_root.exists(),
        "qa_root_absent": not qa_root.exists(),
        "last_two_parent_sessions": tail["selected_session_count"] == 2,
        "true_parent_tail": (
            tail["parent_tail_turn_id"] ==
            simple["expanded_dialogue"]["child_boundary"]["last_parent_turn_ref"]),
        "root_isolated_memory_user": (
            snapshot["user_id"] == memorax_user_id(root_id)),
        "planner_prompt_fits": (
            planner["token_estimate"] + PLANNER_MAX_TOKENS <= CONTEXT_LIMIT),
        "realizer_prompt_fits": (
            realizer["token_estimate"] + ROUND_REALIZER_MAX_TOKENS <=
            CONTEXT_LIMIT),
        "qa_prompt_fits": (
            mcq["token_estimate"] + WRITER_MAX_TOKENS <= CONTEXT_LIMIT),
        "complete_dialogue_is_qa_source": (
            simple["expanded_dialogue"]["qa_fact_source"] ==
            "complete_parent_plus_child"),
        "memory_not_qa_fact_source": (
            mcq["memory_is_qa_fact_source"] is False),
        "solver_model_exists": solver_model.is_dir(),
        "solver_frontier_admission_is_training_candidate": (
            observation_tier["admission_tier"] ==
            "training_candidate_unaudited_evidence" and
            observation_tier["training_ready"] is True),
        "input_unchanged": (
            input_sha_before is not None and
            sha256_file(input_path) == input_sha_before),
        "api_calls_zero": True,
    }
    return {
        "root_id": root_id,
        "input_path": str(input_path),
        "input_sha256": input_sha_before,
        "planned_expansion_root": str(expansion_root),
        "planned_qa_root": str(qa_root),
        "checks": checks,
        "ready": all(checks.values()),
        "parent_turn_count": simple["expanded_dialogue"]["parent_turn_count"],
        "mock_child_turn_count": simple["expanded_dialogue"]["child_turn_count"],
        "mock_complete_turn_count": simple["expanded_dialogue"]["turn_count"],
        "prompt_token_estimates": {
            "planner": planner["token_estimate"],
            "round_realizer": realizer["token_estimate"],
            "mcq_writer": mcq["token_estimate"],
        },
        "memorax_user_id": memorax_user_id(root_id),
        "parent_memory_mode": "initialize_async_pending",
        "planned_request_budget": {
            "planner": 1,
            "round_realizer": 1,
            "mcq_writer": 1,
            "local_solver_physical_max": 264,
            "memorax_add": 2,
            "memorax_search": 2,
            "memorax_status_max": 0,
        },
    }


def run(
    *, batch_root: Path = DEFAULT_BATCH_ROOT,
    root_ids: Sequence[str] = DEFAULT_ROOT_IDS,
    api_config: Path = DEFAULT_API_CONFIG,
    tokenizer_path: Path = DEFAULT_TOKENIZER,
    solver_model: Path = DEFAULT_SOLVER_MODEL,
    memorax_credentials: Path = DEFAULT_MEMORAX_CREDENTIALS,
    allow_insecure_http: bool = True,
    input_paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    rows = [_one(
        root_id=str(root_id), batch_root=batch_root,
        api_config=api_config, tokenizer_path=tokenizer_path,
        solver_model=solver_model, memorax_credentials=memorax_credentials,
        allow_insecure_http=allow_insecure_http,
        input_path=(input_paths or {}).get(str(root_id)),
    ) for root_id in root_ids]
    users = [row["memorax_user_id"] for row in rows]
    checks = {
        "root_count_within_supported_range": 1 <= len(rows) <= MAX_BATCH_ROOTS,
        "root_ids_unique": len(set(root_ids)) == len(root_ids),
        "memorax_user_ids_unique": len(set(users)) == len(users),
        "batch_root_absent": not batch_root.exists(),
        "all_roots_ready": all(row["ready"] for row in rows),
        "cross_root_ref_count_zero": all(
            row["root_id"] in row["memorax_user_id"] for row in rows),
        "api_calls_zero": True,
    }
    return {
        "preflight_version": BATCH_PREFLIGHT_VERSION,
        "status": "passed" if all(checks.values()) else "failed",
        "batch_root": str(batch_root),
        "input_manifest_version": (
            INPUT_MANIFEST_VERSION if input_paths is not None else None),
        "checks": checks,
        "roots": rows,
        "totals": {
            "root_count": len(rows),
            "parent_turns": sum(row["parent_turn_count"] for row in rows),
            "mock_child_turns": sum(row["mock_child_turn_count"] for row in rows),
            "mock_complete_turns": sum(
                row["mock_complete_turn_count"] for row in rows),
            "paid_llm_request_budget": 3 * len(rows),
            "local_solver_physical_max": 264 * len(rows),
            "memorax_add": 2 * len(rows),
            "memorax_search": 2 * len(rows),
            "memorax_status_max": 0,
        },
        "api_calls": 0,
        "run_root_created": batch_root.exists(),
    }


def _markdown(report: dict[str, Any]) -> str:
    rows = [
        "# Memorax Batch Preflight",
        "",
        f"- Status: `{report['status']}`",
        f"- API calls: `{report['api_calls']}`",
        f"- Planned batch root exists: `{report['run_root_created']}`",
        "",
        "| Root | Parent | Mock Child | Planner | Realizer | QA Writer | Ready |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["roots"]:
        prompts = row["prompt_token_estimates"]
        rows.append(
            f"| {row['root_id']} | {row['parent_turn_count']} | "
            f"{row['mock_child_turn_count']} | {prompts['planner']} | "
            f"{prompts['round_realizer']} | {prompts['mcq_writer']} | "
            f"{str(row['ready']).lower()} |")
    rows.extend([
        "",
        "The report uses deterministic 48-turn Child fixtures. It sends no API "
        "request and does not create the planned batch root.",
        "",
    ])
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument(
        "--root-id", action="append", dest="root_ids",
        help=("Authoritative CC root ID; repeat for each root "
              f"(1-{MAX_BATCH_ROOTS} roots)."))
    args = parser.parse_args()
    root_ids = tuple(args.root_ids) if args.root_ids else DEFAULT_ROOT_IDS
    report = run(batch_root=args.batch_root.resolve(), root_ids=root_ids)
    atomic_json(
        args.report_root / "MEMORAX_BATCH_PREFLIGHT.json", report)
    atomic_text(
        args.report_root / "MEMORAX_BATCH_PREFLIGHT.md",
        _markdown(report))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
