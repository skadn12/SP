"""One bounded live recovery that appends completion turns to failed scenes."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..core import atomic_json, atomic_text, sha256_bytes, sha256_file
from ..ledger import RequestLedger
from ..live_backends import (
    OpenAICompatibleChallengerClient,
    load_challenger_config,
)
from ..simple_append_v1.assembly import (
    append_expansion_round,
    validate_appended_round,
)
from .async_write import submit_add_and_wait
from .client import MemoraxClient, load_memorax_config
from .incremental_round import load_incremental_round_input
from .live_runtime import SimpleAppendLiveCaller, aggregate_usage
from .memory_context import (
    audit_bucket_delta_visibility,
    compile_memory_overview,
    normalize_bucket_snapshot,
)
from .payloads import build_add_intent, build_bucket_search_payload
from .promotion import build_promotable_child, validate_promotable_child
from .realizer_adherence import (
    build_adherence_auditor_input,
    build_adherence_auditor_prompt,
    validate_and_materialize_adherence_audit,
)
from .scene_continuation import (
    build_scene_continuation_prompt,
    build_scene_continuation_request,
    materialize_scene_continuation_output,
    merge_scene_continuations,
)


RUNNER_VERSION = "failed_scene_continuation_recovery_v1"
CONTEXT_LIMIT = 40_960
CONTINUATION_MAX_TOKENS = 4_096
AUDITOR_MAX_TOKENS = 4_096
TIMEOUT_SECONDS = 600
DEFAULT_SOURCE_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_41625_memory_only_incremental_r2_live_1")
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_41625_memory_only_incremental_"
    "r2_scene_continuation_live_1")
DEFAULT_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
DEFAULT_TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")
DEFAULT_CREDENTIALS = Path("/root/.config/memorax/credentials.env")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"scene_continuation_json_object_required:{path}")
    return value


def _manifest() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    rows = [{"path": str(path), "sha256": sha256_file(path)}
            for path in sorted(root.glob("*.py"))]
    body = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {"files": rows,
            "manifest_sha256": hashlib.sha256(body).hexdigest()}


def _source_paths(source_run: Path) -> dict[str, Path]:
    return {
        "freeze": source_run / "run_freeze_manifest.json",
        "status": source_run / "run_status.json",
        "plan": source_run / "planner/materialized_plan.json",
        "child": source_run / "child_dialogue.json",
        "audit": source_run / "realizer_adherence/audit.json",
        "audit_validation": source_run / "realizer_adherence/validation.json",
        "pre_snapshot": source_run / "memory/recovery_pre_generation_bucket_snapshot.json",
    }


def _materialize_source(source_run: Path) -> dict[str, Any]:
    paths = _source_paths(source_run)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError("scene_continuation_source_missing:" + ",".join(missing))
    freeze = _load(paths["freeze"])
    prior_path = Path(str(freeze.get("input_path") or ""))
    if not prior_path.is_file():
        raise ValueError("scene_continuation_prior_dialogue_missing")
    parent, prior_expanded, prior_validation = load_incremental_round_input(
        prior_path, round_index=2)
    plan = _load(paths["plan"])
    child = _load(paths["child"])
    audit = _load(paths["audit"])
    request = build_scene_continuation_request(
        plan=plan, child=child, adherence_audit=audit)
    prompt = build_scene_continuation_prompt(request)
    return {
        "paths": paths,
        "freeze": freeze,
        "status": _load(paths["status"]),
        "prior_path": prior_path,
        "parent": parent,
        "prior_expanded": prior_expanded,
        "prior_validation": prior_validation,
        "plan": plan,
        "child": child,
        "audit": audit,
        "audit_validation": _load(paths["audit_validation"]),
        "pre_snapshot": _load(paths["pre_snapshot"]),
        "request": request,
        "prompt": prompt,
    }


def preflight(
    *, source_run: Path, run_root: Path, config_path: Path,
    tokenizer_path: Path, credentials_path: Path,
    allow_insecure_http: bool,
) -> dict[str, Any]:
    checks = {
        "source_run_exists": source_run.is_dir(),
        "run_root_absent": not run_root.exists(),
        "config_exists": config_path.is_file(),
        "tokenizer_exists": tokenizer_path.exists(),
        "credentials_exists": credentials_path.is_file(),
        "api_calls_zero": True,
        "parent_add_zero": True,
        "planner_zero": True,
        "gate_zero": True,
    }
    details: dict[str, Any] = {}
    source: dict[str, Any] = {}
    prompt_tokens = 0
    try:
        source = _materialize_source(source_run)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True)
        prompt_tokens = len(tokenizer.encode(
            source["prompt"]["prompt"], add_special_tokens=False))
        freeze = source["freeze"]
        namespace = str(freeze.get("memorax_user_namespace") or "")
        user_id = str(freeze.get("memory_user_id") or "")
        root_id = str(source["parent"].get("root_id") or "")
        checks.update({
            "source_failed_before_child_add": not (
                source_run / "memory/child_add_intent.json").exists(),
            "source_failure_is_promotion_contract": (
                source["status"].get("failure_stage") == "promotion_contract"),
            "prior_dialogue_valid": source["prior_validation"].get("valid") is True,
            "source_round_is_two": source["child"].get("round_index") == 2,
            "eligible_scenes_present": source["request"].get(
                "eligible_scene_count") == 2,
            "eligible_scenes_are_s2_s3": [
                row.get("scene_ref") for row in
                source["request"].get("eligible_scenes") or []
            ] == ["R02:S02", "R02:S03"],
            "namespace_present": bool(namespace),
            "memory_user_matches_namespace": user_id == f"{namespace}:{root_id}",
            "continuation_context_fits": (
                prompt_tokens + CONTINUATION_MAX_TOKENS <= CONTEXT_LIMIT),
        })
        details.update({
            "root_id": root_id,
            "namespace": namespace,
            "memory_user_id": user_id,
            "prior_path": str(source["prior_path"]),
            "eligible_scene_refs": [
                row.get("scene_ref") for row in
                source["request"].get("eligible_scenes") or []],
        })
    except Exception as error:
        checks["source_materialization"] = False
        details["source_error"] = f"{type(error).__name__}:{error}"
    try:
        api_config = replace(load_challenger_config(config_path),
                             timeout_seconds=TIMEOUT_SECONDS)
        checks["model_configured"] = bool(api_config.model)
        details["api_configuration"] = {
            "model": api_config.model,
            "base_url": api_config.base_url,
            "temperature": api_config.temperature,
            "timeout_seconds": api_config.timeout_seconds,
        }
    except Exception as error:
        checks["model_configured"] = False
        details["api_error"] = f"{type(error).__name__}:{error}"
    try:
        memory_config = load_memorax_config(
            credentials_path=credentials_path,
            allow_insecure_http=allow_insecure_http)
        checks["memorax_configured"] = True
        details["memorax_configuration"] = memory_config.redacted()
    except Exception as error:
        checks["memorax_configured"] = False
        details["memorax_error"] = f"{type(error).__name__}:{error}"
    source_hashes = {
        name: sha256_file(path) for name, path in
        (source.get("paths") or {}).items() if path.is_file()
    }
    prior_path = source.get("prior_path")
    if isinstance(prior_path, Path) and prior_path.is_file():
        source_hashes["prior_dialogue"] = sha256_file(prior_path)
    return {
        "preflight_version": RUNNER_VERSION,
        "decision": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "details": details,
        "source_run": str(source_run.resolve()),
        "run_root": str(run_root.resolve()),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path) if config_path.is_file() else None,
        "tokenizer_path": str(tokenizer_path.resolve()),
        "credentials_path": str(credentials_path.resolve()),
        "source_hashes": source_hashes,
        "source_manifest": _manifest(),
        "continuation_prompt_tokens": prompt_tokens,
        "request_budget": {
            "continuation_realizer": 1,
            "whole_round_adherence_auditor": 1,
            "child_add": 1,
            "post_search": 1,
            "parent_add": 0,
            "planner": 0,
            "gate": 0,
            "retry": 0,
        },
    }


class _FreezeGuard:
    def __init__(self, freeze: Mapping[str, Any]):
        self.freeze = dict(freeze)

    def before_request(self, logical_role: str, role_version: str) -> None:
        if _manifest()["manifest_sha256"] != self.freeze[
                "source_manifest"]["manifest_sha256"]:
            raise RuntimeError("scene_continuation_source_drift")
        for name, expected in self.freeze["source_hashes"].items():
            path = (_source_paths(Path(self.freeze["source_run"])).get(name)
                    if name != "prior_dialogue" else
                    Path(self.freeze["details"]["prior_path"]))
            if path is None or sha256_file(path) != expected:
                raise RuntimeError(f"scene_continuation_input_drift:{name}")


def execute_live(
    *, report: Mapping[str, Any], source_run: Path, run_root: Path,
    config_path: Path, tokenizer_path: Path, credentials_path: Path,
    allow_insecure_http: bool,
) -> dict[str, Any]:
    if report.get("decision") != "pass":
        raise RuntimeError("scene_continuation_preflight_failed")
    if run_root.exists():
        raise FileExistsError(f"scene_continuation_run_exists:{run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    for part in ("continuation", "realizer_adherence", "promotion", "memory",
                 "reports", "raw/prompts", "attempts"):
        (run_root / part).mkdir(parents=True, exist_ok=True)
    freeze = {**dict(report), "frozen_at": _now(), "no_resume": True}
    atomic_json(run_root / "run_freeze_manifest.json", freeze)
    atomic_json(run_root / "run_status.json", {"status": "initialized"})
    source = _materialize_source(source_run)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                               local_files_only=True)
    counter = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    ledger = RequestLedger(
        run_root,
        manifest_sha256=freeze["source_manifest"]["manifest_sha256"],
        backend_mode="live")
    guard = _FreezeGuard(freeze)
    api_config = replace(load_challenger_config(config_path),
                         timeout_seconds=TIMEOUT_SECONDS)
    caller = SimpleAppendLiveCaller(
        run_root=run_root, ledger=ledger,
        client=OpenAICompatibleChallengerClient(api_config),
        token_counter=counter, root_id=str(source["parent"]["root_id"]),
        context_limit=CONTEXT_LIMIT, before_request=guard.before_request)
    memorax = MemoraxClient(load_memorax_config(
        credentials_path=credentials_path,
        allow_insecure_http=allow_insecure_http))
    failure_stage = None
    error_text = None
    summary: dict[str, Any] = {}
    try:
        atomic_json(run_root / "continuation/request.json", source["request"])
        atomic_text(run_root / "continuation/prompt.txt",
                    source["prompt"]["prompt"])
        raw, _, errors = caller.call_json(
            request_id="failed_scene_continuation_realizer_attempt_01",
            role="simple_append_round_realizer_v1",
            prompt=source["prompt"]["prompt"],
            prompt_version=source["prompt"]["prompt_version"],
            max_tokens=CONTINUATION_MAX_TOKENS)
        if errors:
            raise RuntimeError("continuation_transport_or_parse:" +
                               ",".join(errors))
        atomic_json(run_root / "continuation/output.json", raw)
        materialized = materialize_scene_continuation_output(
            raw, request=source["request"], child=source["child"])
        atomic_json(run_root / "continuation/validation.json", materialized)
        if not materialized["valid"]:
            raise RuntimeError("continuation_contract:" +
                               ",".join(materialized["errors"]))
        merged_child = merge_scene_continuations(
            source["child"], materialized)
        atomic_json(run_root / "continuation/merged_child.json", merged_child)

        auditor_input = build_adherence_auditor_input(
            plan=source["plan"], child=merged_child)
        auditor_prompt = build_adherence_auditor_prompt(
            auditor_input=auditor_input)
        atomic_json(run_root / "realizer_adherence/input.json", auditor_input)
        atomic_text(run_root / "realizer_adherence/prompt.txt",
                    auditor_prompt["prompt"])
        audit_raw, _, errors = caller.call_json(
            request_id="scene_continuation_adherence_auditor_attempt_01",
            role="memory_event_realizer_adherence_auditor_v1",
            prompt=auditor_prompt["prompt"],
            prompt_version=auditor_prompt["prompt_version"],
            max_tokens=AUDITOR_MAX_TOKENS)
        if errors:
            raise RuntimeError("auditor_transport_or_parse:" +
                               ",".join(errors))
        atomic_json(run_root / "realizer_adherence/output.json", audit_raw)
        audit_result = validate_and_materialize_adherence_audit(
            audit_raw, plan=source["plan"], child=merged_child)
        atomic_json(run_root / "realizer_adherence/validation.json", audit_result)
        atomic_json(run_root / "realizer_adherence/audit.json",
                    audit_result.get("audit") or {})
        if not audit_result["valid"]:
            raise RuntimeError("auditor_contract:" +
                               ",".join(audit_result["errors"]))

        promotion = build_promotable_child(
            parent=source["parent"], plan=source["plan"], child=merged_child,
            adherence_audit=audit_result["audit"])
        atomic_json(run_root / "promotion/validation.json", promotion)
        atomic_json(run_root / "promotion/audit.json",
                    promotion.get("audit") or {})
        if not promotion["valid"]:
            raise RuntimeError("promotion_contract:" +
                               ",".join(promotion["errors"]))
        promotable = promotion["promotable_child"]
        promoted_validation = validate_promotable_child(
            promotable, source_child=merged_child)
        atomic_json(run_root / "promotion/promotable_child.json", promotable)
        atomic_json(run_root / "promotion/promotable_child_validation.json",
                    promoted_validation)
        if not promoted_validation["valid"]:
            raise RuntimeError("promotion_materialization:" +
                               ",".join(promoted_validation["errors"]))
        expanded = append_expansion_round(
            prior_expanded=source["prior_expanded"], child=promotable)
        expanded_validation = validate_appended_round(
            expanded, prior_expanded=source["prior_expanded"])
        atomic_json(run_root / "promotion/promotable_expanded_dialogue.json",
                    expanded)
        atomic_json(run_root / "promotion/expanded_validation.json",
                    expanded_validation)
        if not expanded_validation["valid"]:
            raise RuntimeError("promotion_assembly:" +
                               ",".join(expanded_validation["errors"]))

        namespace = str(source["freeze"]["memorax_user_namespace"])
        child_turns = [
            {**turn, "session_ref": scene["session_ref"]}
            for scene in promotable["scenes"]
            for turn in scene["dialogue_turns"]
        ]
        parent_turns = list(source["parent"]["dialogue"]["turns"])
        child_intent = build_add_intent(
            root_id=str(source["parent"]["root_id"]), round_index=2,
            source_dialogue_sha256=expanded["complete_dialogue_sha256"],
            turns=child_turns, recent_context_turns=parent_turns[-20:],
            async_mode=True, user_namespace=namespace)
        atomic_json(run_root / "memory/child_add_intent.json", child_intent)
        child_completion = submit_add_and_wait(
            gateway=memorax, intent=child_intent,
            poll_delays_seconds=(0.5, 1, 2, 3, 5, 8, 13, 21),
            stage="scene_continuation_child_add")
        atomic_json(run_root / "memory/child_add_completion.json",
                    child_completion)
        query = build_bucket_search_payload(
            root_id=str(source["parent"]["root_id"]),
            user_namespace=namespace)
        post_raw = memorax.search_buckets(
            query["payload"], request_id="scene-continuation-r02-post-search")
        post_snapshot = normalize_bucket_snapshot(
            post_raw, root_id=str(source["parent"]["root_id"]),
            user_id=query["payload"]["user_id"])
        atomic_json(run_root / "memory/post_generation_bucket_snapshot.json",
                    post_snapshot)
        visibility = audit_bucket_delta_visibility(
            source["pre_snapshot"], post_snapshot)
        atomic_json(run_root / "memory/child_visibility_audit.json", visibility)
        post_overview = compile_memory_overview(
            post_snapshot,
            recent_dialogue_text="\n".join(turn["text"] for turn in child_turns))
        atomic_json(run_root / "memory/post_generation_overview.json",
                    post_overview)
        summary = {
            "status": "completed",
            "eligible_scene_refs": [
                row["scene_ref"] for row in source["request"]["eligible_scenes"]],
            "promoted_scene_refs": promotion["audit"]["promoted_scene_refs"],
            "rejected_scene_refs": promotion["audit"]["rejected_scene_refs"],
            "full_round_ready": promotion["audit"]["full_round_ready"],
            "partial_promotion": promotion["audit"]["partial_promotion"],
            "child_add_verified": child_completion.get("verified_complete") is True,
            "memory_visibility_verified": visibility.get(
                "bucket_visibility_verified") is True,
            "final_turn_count": len(expanded["turns"]),
        }
    except Exception as error:
        failure_stage = str(error).split(":", 1)[0]
        error_text = f"{type(error).__name__}:{error}"
        summary = {"status": "failed", "failure_stage": failure_stage,
                   "runtime_error": error_text}
    usage = aggregate_usage(caller.records)
    summary.update({
        "source_run": str(source_run.resolve()),
        "run_root": str(run_root.resolve()),
        "llm_calls": usage["total_requests"],
        "llm_tokens": usage["total_tokens"],
        "memorax_calls": len(memorax.calls),
        "parent_add_calls": 0,
        "planner_calls": 0,
        "gate_calls": 0,
        "retry_count": 0,
        "completed_at": _now(),
    })
    atomic_json(run_root / "reports/SCENE_CONTINUATION_LIVE_REPORT.json",
                summary)
    atomic_text(run_root / "reports/SCENE_CONTINUATION_LIVE_REPORT.md",
                "# Scene Continuation Live Report\n\n" +
                "\n".join(f"- {key}: `{value}`" for key, value in
                          summary.items() if not isinstance(value, (dict, list))) +
                "\n")
    atomic_json(run_root / "run_status.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--memorax-credentials", type=Path,
                        default=DEFAULT_CREDENTIALS)
    parser.add_argument("--allow-insecure-memorax-http", action="store_true")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    report = preflight(
        source_run=args.source_run, run_root=args.run_root,
        config_path=args.config, tokenizer_path=args.tokenizer,
        credentials_path=args.memorax_credentials,
        allow_insecure_http=args.allow_insecure_memorax_http)
    if not args.live:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["decision"] == "pass" else 2
    summary = execute_live(
        report=report, source_run=args.source_run, run_root=args.run_root,
        config_path=args.config, tokenizer_path=args.tokenizer,
        credentials_path=args.memorax_credentials,
        allow_insecure_http=args.allow_insecure_memorax_http)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
