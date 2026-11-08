"""Promote an already audited Child, then write it to Memorax exactly once."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..core import atomic_json, atomic_text, sha256_file
from ..simple_append_v1.assembly import (
    assemble_expanded_dialogue,
    validate_expanded_dialogue,
)
from .async_write import submit_add_and_wait
from .client import MemoraxClient, load_memorax_config
from .memory_context import (
    audit_bucket_delta_visibility,
    compile_memory_overview,
    normalize_bucket_snapshot,
)
from .payloads import build_add_intent
from .promotion import build_promotable_child, validate_promotable_child
from .write_state import apply_visibility_result


RECOVERY_VERSION = "audited_child_promotion_recovery_v1"
DEFAULT_SOURCE_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_41625_memory_only_"
    "fine_grained_candidate_fix_live_1")
DEFAULT_AUDIT_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_41625_realizer_adherence_v6_probe_1")
DEFAULT_OUTPUT_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_41625_child_promotion_recovery_v1")
DEFAULT_CREDENTIALS = Path("/root/.config/memorax/credentials.env")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"recovery_json_object_required:{path}")
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _paths(source_run: Path, audit_run: Path) -> dict[str, Path]:
    return {
        "source_freeze": source_run / "run_freeze_manifest.json",
        "parent_add_intent": source_run / "memory/parent_add_intent.json",
        "parent_add_completion": source_run / "memory/parent_add_completion.json",
        "pre_query": source_run / "memory/pre_generation_bucket_query.json",
        "pre_snapshot": source_run / "memory/pre_generation_bucket_snapshot.json",
        "plan": source_run / "planner/materialized_plan.json",
        "child": source_run / "child_dialogue.json",
        "auditor_validation": audit_run / "realizer_adherence/validation.json",
        "audit": audit_run / "realizer_adherence/audit.json",
    }


def _materialize(
    *, source_run: Path, audit_run: Path,
) -> dict[str, Any]:
    paths = _paths(source_run, audit_run)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError("recovery_source_artifacts_missing:" + ",".join(missing))
    source_freeze = _load(paths["source_freeze"])
    parent_path = Path(str(source_freeze.get("input_path") or ""))
    if not parent_path.is_file():
        raise ValueError("recovery_parent_missing")
    parent = _load(parent_path)
    plan = _load(paths["plan"])
    child = _load(paths["child"])
    audit = _load(paths["audit"])
    audit_validation = _load(paths["auditor_validation"])
    parent_intent = _load(paths["parent_add_intent"])
    parent_completion = _load(paths["parent_add_completion"])
    pre_query = _load(paths["pre_query"])
    pre_snapshot = _load(paths["pre_snapshot"])

    promotion = build_promotable_child(
        parent=parent, plan=plan, child=child, adherence_audit=audit)
    promotable = promotion.get("promotable_child") or {}
    promoted_validation = validate_promotable_child(
        promotable, source_child=child) if promotion.get("valid") else {
            "valid": False, "errors": ["promotion_not_valid"]}
    expanded = assemble_expanded_dialogue(
        parent=parent, child=promotable) if promoted_validation.get("valid") else {}
    expanded_validation = validate_expanded_dialogue(
        expanded, parent=parent) if expanded else {
            "valid": False, "errors": ["promotable_expanded_missing"]}

    root_id = str(parent.get("root_id") or "")
    user_id = str(parent_intent.get("user_id") or "")
    suffix = f":{root_id}"
    namespace = user_id[:-len(suffix)] if user_id.endswith(suffix) else ""
    child_turns = [
        {**turn, "session_ref": scene["session_ref"]}
        for scene in promotable.get("scenes") or []
        for turn in scene.get("dialogue_turns") or []
    ]
    parent_turns = list((parent.get("dialogue") or {}).get("turns") or [])
    child_intent = build_add_intent(
        root_id=root_id,
        round_index=1,
        source_dialogue_sha256=str(expanded.get("complete_dialogue_sha256") or ""),
        turns=child_turns,
        recent_context_turns=parent_turns[-20:],
        async_mode=True,
        user_namespace=namespace,
    ) if expanded_validation.get("valid") and namespace else {}

    return {
        "paths": paths,
        "parent_path": parent_path,
        "parent": parent,
        "plan": plan,
        "child": child,
        "audit": audit,
        "audit_validation": audit_validation,
        "parent_intent": parent_intent,
        "parent_completion": parent_completion,
        "pre_query": pre_query,
        "pre_snapshot": pre_snapshot,
        "promotion": promotion,
        "promotable_child": promotable,
        "promotable_child_validation": promoted_validation,
        "expanded": expanded,
        "expanded_validation": expanded_validation,
        "child_intent": child_intent,
        "namespace": namespace,
    }


def preflight(
    *, source_run: Path, audit_run: Path, output_root: Path,
    credentials_path: Path, allow_insecure_http: bool,
) -> dict[str, Any]:
    errors: list[str] = []
    checks: dict[str, bool] = {
        "output_root_absent": not output_root.exists(),
        "source_run_exists": source_run.is_dir(),
        "audit_run_exists": audit_run.is_dir(),
        "source_child_add_intent_absent": not (
            source_run / "memory/child_add_intent.json").exists(),
        "source_child_add_completion_absent": not (
            source_run / "memory/child_add_completion.json").exists(),
        "credentials_exist": credentials_path.is_file(),
        "api_calls_zero": True,
        "planner_calls_zero": True,
        "gate_calls_zero": True,
        "realizer_calls_zero": True,
        "auditor_calls_zero": True,
    }
    materialized: dict[str, Any] = {}
    try:
        materialized = _materialize(source_run=source_run, audit_run=audit_run)
        audit_validation = materialized["audit_validation"]
        audit_summary = materialized["audit"].get("summary") or {}
        promotion_audit = materialized["promotion"].get("audit") or {}
        parent_completion = materialized["parent_completion"]
        query_user = str((materialized["pre_query"].get("payload") or {}).get(
            "user_id") or "")
        checks.update({
            "auditor_validation_valid": audit_validation.get("valid") is True,
            "auditor_source_unchanged": audit_validation.get(
                "source_run_unchanged") is True,
            "auditor_exactly_one_call": audit_validation.get("api_calls") == 1,
            "auditor_three_of_three_pass": (
                audit_summary.get("scene_count") == 3
                and audit_summary.get("scene_pass_count") == 3),
            "promotion_contract_valid": materialized["promotion"].get("valid") is True,
            "full_round_ready": promotion_audit.get("full_round_ready") is True,
            "three_scenes_promoted": promotion_audit.get(
                "promoted_scene_count") == 3,
            "promotable_child_valid": materialized[
                "promotable_child_validation"].get("valid") is True,
            "promotable_expanded_valid": materialized[
                "expanded_validation"].get("valid") is True,
            "parent_add_verified": parent_completion.get(
                "verified_complete") is True,
            "namespace_matches_parent": bool(materialized["namespace"]),
            "query_user_matches_parent": query_user == materialized[
                "parent_intent"].get("user_id"),
            "child_intent_user_matches_parent": materialized[
                "child_intent"].get("user_id") == materialized[
                    "parent_intent"].get("user_id"),
        })
    except Exception as error:  # surfaced as a deterministic preflight failure
        errors.append(f"{type(error).__name__}:{error}")
        checks["materialization_succeeded"] = False
    try:
        memory_config = load_memorax_config(
            credentials_path=credentials_path,
            allow_insecure_http=allow_insecure_http)
        checks["memorax_config_valid"] = True
        memory_config_view = memory_config.redacted()
    except Exception as error:
        checks["memorax_config_valid"] = False
        errors.append(f"{type(error).__name__}:{error}")
        memory_config_view = None

    source_hashes = {
        name: sha256_file(path) for name, path in
        (materialized.get("paths") or {}).items() if path.is_file()
    }
    parent_path = materialized.get("parent_path")
    if isinstance(parent_path, Path) and parent_path.is_file():
        source_hashes["parent"] = sha256_file(parent_path)
    return {
        "preflight_version": RECOVERY_VERSION,
        "decision": "pass" if all(checks.values()) and not errors else "fail",
        "checks": checks,
        "failed_checks": [key for key, passed in checks.items() if not passed],
        "errors": errors,
        "source_run": str(source_run.resolve()),
        "audit_run": str(audit_run.resolve()),
        "output_root": str(output_root.resolve()),
        "source_hashes": source_hashes,
        "root_id": str((materialized.get("parent") or {}).get("root_id") or ""),
        "memory_user_id": str((materialized.get("parent_intent") or {}).get(
            "user_id") or ""),
        "promoted_scene_refs": list(((materialized.get("promotion") or {}).get(
            "audit") or {}).get("promoted_scene_refs") or []),
        "child_add_request_id": str((materialized.get("child_intent") or {}).get(
            "request_id") or ""),
        "memorax_configuration": memory_config_view,
        "request_budget": {
            "child_add": 1,
            "post_search": 1,
            "planner": 0,
            "gate": 0,
            "realizer": 0,
            "auditor": 0,
            "retry": 0,
        },
    }


def _write_report(root: Path, summary: Mapping[str, Any]) -> None:
    visibility = summary.get("visibility_audit") or {}
    lines = [
        "# Audited Child Promotion Recovery",
        "",
        f"- status: `{summary.get('status')}`",
        f"- failure_stage: `{summary.get('failure_stage')}`",
        f"- promoted scenes: `{summary.get('promoted_scene_count')}/3`",
        f"- Child Add calls: `{summary.get('child_add_calls')}`",
        f"- Post Search calls: `{summary.get('post_search_calls')}`",
        "- Planner / Gate / Realizer / Auditor calls: `0 / 0 / 0 / 0`",
        f"- write verified: `{summary.get('memory_write_verified')}`",
        f"- bucket visibility verified: `{summary.get('memory_visibility_verified')}`",
        f"- changed buckets: `{len(visibility.get('changed_bucket_refs') or [])}`",
        f"- source artifacts unchanged: `{summary.get('source_artifacts_unchanged')}`",
    ]
    atomic_text(root / "reports/CHILD_PROMOTION_RECOVERY_REPORT.md",
                "\n".join(lines) + "\n")


def execute_live(
    *, report: Mapping[str, Any], source_run: Path, audit_run: Path,
    output_root: Path, credentials_path: Path, allow_insecure_http: bool,
) -> dict[str, Any]:
    if report.get("decision") != "pass":
        raise RuntimeError("promotion_recovery_preflight_failed")
    if output_root.exists():
        raise FileExistsError(f"promotion_recovery_output_exists:{output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    for part in ("promotion", "memory", "reports"):
        (output_root / part).mkdir(parents=True, exist_ok=True)
    freeze = {**dict(report), "frozen_at": _now(), "no_resume": True}
    atomic_json(output_root / "run_freeze_manifest.json", freeze)
    atomic_json(output_root / "run_status.json", {
        "status": "initialized", "child_add_calls": 0,
        "post_search_calls": 0})
    materialized = _materialize(source_run=source_run, audit_run=audit_run)
    atomic_json(output_root / "promotion/audit.json",
                materialized["promotion"].get("audit") or {})
    atomic_json(output_root / "promotion/validation.json",
                materialized["promotion"])
    atomic_json(output_root / "promotion/promotable_child.json",
                materialized["promotable_child"])
    atomic_json(output_root / "promotion/promotable_child_validation.json",
                materialized["promotable_child_validation"])
    atomic_json(output_root / "promotion/promotable_expanded_dialogue.json",
                materialized["expanded"])
    atomic_json(output_root / "promotion/promotable_expanded_validation.json",
                materialized["expanded_validation"])
    atomic_json(output_root / "memory/child_add_intent.json",
                materialized["child_intent"])

    def assert_frozen() -> None:
        current = {
            name: sha256_file(path) for name, path in
            materialized["paths"].items()}
        current["parent"] = sha256_file(materialized["parent_path"])
        if current != report.get("source_hashes"):
            raise RuntimeError("promotion_recovery_source_drift")

    memorax = MemoraxClient(load_memorax_config(
        credentials_path=credentials_path,
        allow_insecure_http=allow_insecure_http))
    child_state: dict[str, Any] = {}
    child_completion: dict[str, Any] | None = None
    post_snapshot: dict[str, Any] | None = None
    visibility: dict[str, Any] | None = None
    failure_stage = None
    error = None
    try:
        assert_frozen()
        atomic_json(output_root / "run_status.json", {
            "status": "child_add_started", "child_add_calls": 1,
            "post_search_calls": 0})

        def on_state(state: Mapping[str, Any]) -> None:
            child_state.update(state)
            atomic_json(output_root / "memory/child_add_write_state.json", state)

        child_completion = submit_add_and_wait(
            gateway=memorax,
            intent=materialized["child_intent"],
            stage="audited_child_memory_add",
            on_state_change=on_state)
        atomic_json(output_root / "memory/child_add_completion.json",
                    child_completion)

        assert_frozen()
        query = materialized["pre_query"]
        raw_post = memorax.search_buckets(
            query["payload"],
            request_id=f"promotion-recovery-post-{materialized['parent']['root_id']}")
        atomic_json(output_root / "memory/post_generation_bucket_query.json", query)
        post_snapshot = normalize_bucket_snapshot(
            raw_post,
            root_id=str(materialized["parent"]["root_id"]),
            user_id=str(query["payload"]["user_id"]))
        atomic_json(output_root / "memory/post_generation_bucket_snapshot.json",
                    post_snapshot)
        visibility = audit_bucket_delta_visibility(
            materialized["pre_snapshot"], post_snapshot)
        child_state = apply_visibility_result(
            child_state,
            visible=bool(visibility.get("bucket_visibility_verified")))
        atomic_json(output_root / "memory/child_visibility_audit.json", visibility)
        atomic_json(output_root / "memory/child_add_write_state.json", child_state)
        recent = "\n".join(
            str(turn.get("text") or "")
            for scene in materialized["promotable_child"].get("scenes") or []
            for turn in scene.get("dialogue_turns") or [])
        atomic_json(output_root / "memory/post_generation_overview.json",
                    compile_memory_overview(
                        post_snapshot, recent_dialogue_text=recent))
    except Exception as caught:
        failure_stage = "child_add" if child_completion is None else "post_search"
        error = f"{type(caught).__name__}:{caught}"

    assert_frozen()
    add_calls = sum(row.get("path") == "/v1/memories/add"
                    for row in memorax.calls)
    search_calls = sum(row.get("path") == "/v1/memory-buckets/search"
                       for row in memorax.calls)
    summary = {
        "summary_version": RECOVERY_VERSION,
        "status": "completed" if child_completion and not error else "failed",
        "failure_stage": failure_stage,
        "error": error,
        "source_run": str(source_run.resolve()),
        "audit_run": str(audit_run.resolve()),
        "output_root": str(output_root.resolve()),
        "promoted_scene_count": len(materialized["promotable_child"].get(
            "scenes") or []),
        "child_add_calls": add_calls,
        "post_search_calls": search_calls,
        "memorax_calls": memorax.calls,
        "memory_write_verified": bool(child_state.get("memory_write_verified")),
        "memory_visibility_verified": bool(
            child_state.get("bucket_visibility_verified")),
        "child_add_state": child_state,
        "child_add_completion": child_completion,
        "visibility_audit": visibility,
        "source_artifacts_unchanged": True,
        "planner_calls": 0,
        "gate_calls": 0,
        "realizer_calls": 0,
        "auditor_calls": 0,
        "retries": 0,
    }
    atomic_json(output_root / "reports/CHILD_PROMOTION_RECOVERY_REPORT.json",
                summary)
    _write_report(output_root, summary)
    atomic_json(output_root / "run_status.json", {
        "status": summary["status"],
        "failure_stage": failure_stage,
        "child_add_calls": add_calls,
        "post_search_calls": search_calls})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--audit-run", type=Path, default=DEFAULT_AUDIT_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--credentials", type=Path, default=DEFAULT_CREDENTIALS)
    parser.add_argument("--allow-insecure-memorax-http", action="store_true")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    report = preflight(
        source_run=args.source_run,
        audit_run=args.audit_run,
        output_root=args.output_root,
        credentials_path=args.credentials,
        allow_insecure_http=args.allow_insecure_memorax_http)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["decision"] != "pass":
        return 2
    if args.live:
        summary = execute_live(
            report=report,
            source_run=args.source_run,
            audit_run=args.audit_run,
            output_root=args.output_root,
            credentials_path=args.credentials,
            allow_insecure_http=args.allow_insecure_memorax_http)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["status"] == "completed" else 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
