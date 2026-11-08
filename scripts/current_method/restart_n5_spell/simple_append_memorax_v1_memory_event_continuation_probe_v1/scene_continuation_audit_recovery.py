"""Audit a mechanically normalized continuation response, then promote/write."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..core import atomic_json, atomic_text, sha256_file
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
    materialize_scene_continuation_output,
    merge_scene_continuations,
)
from .scene_continuation_live import (
    _manifest,
    _materialize_source,
    _source_paths,
    AUDITOR_MAX_TOKENS,
    CONTEXT_LIMIT,
    DEFAULT_CONFIG,
    DEFAULT_CREDENTIALS,
    DEFAULT_SOURCE_RUN,
    DEFAULT_TOKENIZER,
    TIMEOUT_SECONDS,
)


RUNNER_VERSION = "failed_scene_continuation_audit_recovery_v1"
DEFAULT_CONTINUATION_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_41625_memory_only_incremental_"
    "r2_scene_continuation_live_1")
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_41625_memory_only_incremental_"
    "r2_scene_continuation_audit_recovery_1")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"continuation_audit_json_object_required:{path}")
    return value


def _materialize(
    *, source_run: Path, continuation_run: Path,
) -> dict[str, Any]:
    source = _materialize_source(source_run)
    raw_path = continuation_run / "continuation/output.json"
    status_path = continuation_run / "run_status.json"
    if not raw_path.is_file() or not status_path.is_file():
        raise ValueError("continuation_audit_source_output_missing")
    raw = _load(raw_path)
    continuation_status = _load(status_path)
    materialized = materialize_scene_continuation_output(
        raw, request=source["request"], child=source["child"])
    if not materialized["valid"]:
        raise ValueError("continuation_audit_materialization_invalid:" +
                         ",".join(materialized["errors"]))
    if materialized.get("normalizations") != [
            "collapsed_exact_repeated_continuation_block"]:
        raise ValueError("continuation_audit_expected_exact_block_normalization")
    merged_child = merge_scene_continuations(
        source["child"], materialized)
    auditor_input = build_adherence_auditor_input(
        plan=source["plan"], child=merged_child)
    auditor_prompt = build_adherence_auditor_prompt(
        auditor_input=auditor_input)
    return {
        **source,
        "continuation_run": continuation_run,
        "continuation_raw_path": raw_path,
        "continuation_status_path": status_path,
        "continuation_status": continuation_status,
        "continuation_raw": raw,
        "continuation_materialized": materialized,
        "merged_child": merged_child,
        "auditor_input": auditor_input,
        "auditor_prompt": auditor_prompt,
    }


def preflight(
    *, source_run: Path, continuation_run: Path, run_root: Path,
    config_path: Path, tokenizer_path: Path, credentials_path: Path,
    allow_insecure_http: bool,
) -> dict[str, Any]:
    checks = {
        "source_run_exists": source_run.is_dir(),
        "continuation_run_exists": continuation_run.is_dir(),
        "run_root_absent": not run_root.exists(),
        "config_exists": config_path.is_file(),
        "tokenizer_exists": tokenizer_path.exists(),
        "credentials_exists": credentials_path.is_file(),
        "api_calls_zero": True,
        "realizer_zero": True,
        "parent_add_zero": True,
        "planner_zero": True,
        "gate_zero": True,
    }
    details: dict[str, Any] = {}
    source: dict[str, Any] = {}
    prompt_tokens = 0
    try:
        source = _materialize(
            source_run=source_run, continuation_run=continuation_run)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True)
        prompt_tokens = len(tokenizer.encode(
            source["auditor_prompt"]["prompt"], add_special_tokens=False))
        namespace = str(source["freeze"].get("memorax_user_namespace") or "")
        user_id = str(source["freeze"].get("memory_user_id") or "")
        root_id = str(source["parent"].get("root_id") or "")
        checks.update({
            "continuation_failed_only_on_count_contract": (
                source["continuation_status"].get("failure_stage") ==
                "continuation_contract"),
            "continuation_had_one_llm_call": (
                source["continuation_status"].get("llm_calls") == 1),
            "continuation_had_no_memory_calls": (
                source["continuation_status"].get("memorax_calls") == 0),
            "exact_duplicate_block_collapsed": source[
                "continuation_materialized"].get("normalizations") == [
                    "collapsed_exact_repeated_continuation_block"],
            "two_continuations_materialized": len(source[
                "continuation_materialized"]["continuations"]) == 2,
            "merged_turn_counts_14_18_18": [
                len(scene.get("dialogue_turns") or [])
                for scene in source["merged_child"]["scenes"]
            ] == [14, 18, 18],
            "auditor_context_fits": (
                prompt_tokens + AUDITOR_MAX_TOKENS <= CONTEXT_LIMIT),
            "namespace_present": bool(namespace),
            "memory_user_matches_namespace": user_id == f"{namespace}:{root_id}",
            "source_child_add_absent": not (
                source_run / "memory/child_add_intent.json").exists(),
            "continuation_child_add_absent": not (
                continuation_run / "memory/child_add_intent.json").exists(),
        })
        details.update({
            "root_id": root_id,
            "namespace": namespace,
            "memory_user_id": user_id,
            "prior_path": str(source["prior_path"]),
            "merged_turn_counts": [
                len(scene.get("dialogue_turns") or [])
                for scene in source["merged_child"]["scenes"]],
        })
    except Exception as error:
        checks["source_materialization"] = False
        details["source_error"] = f"{type(error).__name__}:{error}"
    try:
        config = replace(load_challenger_config(config_path),
                         timeout_seconds=TIMEOUT_SECONDS)
        checks["model_configured"] = bool(config.model)
        details["api_configuration"] = {
            "model": config.model, "base_url": config.base_url,
            "temperature": config.temperature,
            "timeout_seconds": config.timeout_seconds,
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
        "source:" + name: sha256_file(path)
        for name, path in (source.get("paths") or {}).items() if path.is_file()
    }
    for name in ("continuation_raw_path", "continuation_status_path"):
        path = source.get(name)
        if isinstance(path, Path) and path.is_file():
            source_hashes[name] = sha256_file(path)
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
        "continuation_run": str(continuation_run.resolve()),
        "run_root": str(run_root.resolve()),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path) if config_path.is_file() else None,
        "tokenizer_path": str(tokenizer_path.resolve()),
        "credentials_path": str(credentials_path.resolve()),
        "source_hashes": source_hashes,
        "source_manifest": _manifest(),
        "auditor_prompt_tokens": prompt_tokens,
        "request_budget": {
            "whole_round_adherence_auditor": 1,
            "child_add": 1,
            "post_search": 1,
            "continuation_realizer": 0,
            "parent_add": 0,
            "planner": 0,
            "gate": 0,
            "retry": 0,
        },
        "imported_continuation_response_reuse": 1,
    }


class _FreezeGuard:
    def __init__(self, freeze: Mapping[str, Any]):
        self.freeze = dict(freeze)

    def before_request(self, logical_role: str, role_version: str) -> None:
        if _manifest()["manifest_sha256"] != self.freeze[
                "source_manifest"]["manifest_sha256"]:
            raise RuntimeError("continuation_audit_code_drift")
        source_run = Path(self.freeze["source_run"])
        continuation_run = Path(self.freeze["continuation_run"])
        for name, expected in self.freeze["source_hashes"].items():
            if name.startswith("source:"):
                path = _source_paths(source_run).get(name.split(":", 1)[1])
            elif name == "continuation_raw_path":
                path = continuation_run / "continuation/output.json"
            elif name == "continuation_status_path":
                path = continuation_run / "run_status.json"
            else:
                path = Path(self.freeze["details"]["prior_path"])
            if path is None or sha256_file(path) != expected:
                raise RuntimeError(f"continuation_audit_input_drift:{name}")


def execute_live(
    *, report: Mapping[str, Any], source_run: Path, continuation_run: Path,
    run_root: Path, config_path: Path, tokenizer_path: Path,
    credentials_path: Path, allow_insecure_http: bool,
) -> dict[str, Any]:
    if report.get("decision") != "pass":
        raise RuntimeError("continuation_audit_preflight_failed")
    if run_root.exists():
        raise FileExistsError(f"continuation_audit_run_exists:{run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    for part in ("continuation", "realizer_adherence", "promotion", "memory",
                 "reports", "raw/prompts", "attempts"):
        (run_root / part).mkdir(parents=True, exist_ok=True)
    freeze = {**dict(report), "frozen_at": _now(), "no_resume": True}
    atomic_json(run_root / "run_freeze_manifest.json", freeze)
    atomic_json(run_root / "run_status.json", {"status": "initialized"})
    source = _materialize(
        source_run=source_run, continuation_run=continuation_run)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                               local_files_only=True)
    counter = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    ledger = RequestLedger(
        run_root,
        manifest_sha256=freeze["source_manifest"]["manifest_sha256"],
        backend_mode="live")
    guard = _FreezeGuard(freeze)
    caller = SimpleAppendLiveCaller(
        run_root=run_root, ledger=ledger,
        client=OpenAICompatibleChallengerClient(replace(
            load_challenger_config(config_path),
            timeout_seconds=TIMEOUT_SECONDS)),
        token_counter=counter, root_id=str(source["parent"]["root_id"]),
        context_limit=CONTEXT_LIMIT, before_request=guard.before_request)
    memorax = MemoraxClient(load_memorax_config(
        credentials_path=credentials_path,
        allow_insecure_http=allow_insecure_http))
    summary: dict[str, Any]
    try:
        atomic_json(run_root / "continuation/imported_output.json",
                    source["continuation_raw"])
        atomic_json(run_root / "continuation/materialized.json",
                    source["continuation_materialized"])
        atomic_json(run_root / "continuation/merged_child.json",
                    source["merged_child"])
        atomic_json(run_root / "realizer_adherence/input.json",
                    source["auditor_input"])
        atomic_text(run_root / "realizer_adherence/prompt.txt",
                    source["auditor_prompt"]["prompt"])
        raw, _, errors = caller.call_json(
            request_id="scene_continuation_adherence_auditor_attempt_01",
            role="memory_event_realizer_adherence_auditor_v1",
            prompt=source["auditor_prompt"]["prompt"],
            prompt_version=source["auditor_prompt"]["prompt_version"],
            max_tokens=AUDITOR_MAX_TOKENS)
        if errors:
            raise RuntimeError("auditor_transport_or_parse:" +
                               ",".join(errors))
        atomic_json(run_root / "realizer_adherence/output.json", raw)
        audit_result = validate_and_materialize_adherence_audit(
            raw, plan=source["plan"], child=source["merged_child"])
        atomic_json(run_root / "realizer_adherence/validation.json", audit_result)
        atomic_json(run_root / "realizer_adherence/audit.json",
                    audit_result.get("audit") or {})
        if not audit_result["valid"]:
            raise RuntimeError("auditor_contract:" +
                               ",".join(audit_result["errors"]))

        promotion = build_promotable_child(
            parent=source["parent"], plan=source["plan"],
            child=source["merged_child"],
            adherence_audit=audit_result["audit"])
        atomic_json(run_root / "promotion/validation.json", promotion)
        atomic_json(run_root / "promotion/audit.json",
                    promotion.get("audit") or {})
        if not promotion["valid"]:
            raise RuntimeError("promotion_contract:" +
                               ",".join(promotion["errors"]))
        promotable = promotion["promotable_child"]
        promoted_validation = validate_promotable_child(
            promotable, source_child=source["merged_child"])
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
        intent = build_add_intent(
            root_id=str(source["parent"]["root_id"]), round_index=2,
            source_dialogue_sha256=expanded["complete_dialogue_sha256"],
            turns=child_turns, recent_context_turns=parent_turns[-20:],
            async_mode=True, user_namespace=namespace)
        atomic_json(run_root / "memory/child_add_intent.json", intent)
        completion = submit_add_and_wait(
            gateway=memorax, intent=intent,
            poll_delays_seconds=(0.5, 1, 2, 3, 5, 8, 13, 21),
            stage="scene_continuation_audit_child_add")
        atomic_json(run_root / "memory/child_add_completion.json", completion)
        query = build_bucket_search_payload(
            root_id=str(source["parent"]["root_id"]),
            user_namespace=namespace)
        post_raw = memorax.search_buckets(
            query["payload"], request_id="scene-continuation-audit-r02-post")
        post_snapshot = normalize_bucket_snapshot(
            post_raw, root_id=str(source["parent"]["root_id"]),
            user_id=query["payload"]["user_id"])
        atomic_json(run_root / "memory/post_generation_bucket_snapshot.json",
                    post_snapshot)
        visibility = audit_bucket_delta_visibility(
            source["pre_snapshot"], post_snapshot)
        atomic_json(run_root / "memory/child_visibility_audit.json", visibility)
        atomic_json(run_root / "memory/post_generation_overview.json",
                    compile_memory_overview(
                        post_snapshot,
                        recent_dialogue_text="\n".join(
                            turn["text"] for turn in child_turns)))
        summary = {
            "status": "completed",
            "scene_verdicts": [
                {"scene_ref": row["scene_ref"],
                 "scene_verdict": row["scene_verdict"],
                 "development_status": row["development_status"],
                 "durable_outcome_status": row["durable_outcome_status"]}
                for row in audit_result["audit"]["scenes"]],
            "promoted_scene_refs": promotion["audit"]["promoted_scene_refs"],
            "rejected_scene_refs": promotion["audit"]["rejected_scene_refs"],
            "full_round_ready": promotion["audit"]["full_round_ready"],
            "partial_promotion": promotion["audit"]["partial_promotion"],
            "child_add_verified": completion.get("verified_complete") is True,
            "memory_visibility_verified": visibility.get(
                "bucket_visibility_verified") is True,
            "final_turn_count": len(expanded["turns"]),
        }
    except Exception as error:
        summary = {
            "status": "failed",
            "failure_stage": str(error).split(":", 1)[0],
            "runtime_error": f"{type(error).__name__}:{error}",
        }
    usage = aggregate_usage(caller.records)
    summary.update({
        "source_run": str(source_run.resolve()),
        "continuation_run": str(continuation_run.resolve()),
        "run_root": str(run_root.resolve()),
        "llm_calls": usage["total_requests"],
        "llm_tokens": usage["total_tokens"],
        "realizer_calls": 0,
        "auditor_calls": usage["roles"][
            "realizer_adherence_auditor"]["requests"],
        "memorax_calls": len(memorax.calls),
        "parent_add_calls": 0,
        "planner_calls": 0,
        "gate_calls": 0,
        "retry_count": 0,
        "imported_continuation_response_reuse": 1,
        "completed_at": _now(),
    })
    atomic_json(run_root / "reports/SCENE_CONTINUATION_AUDIT_RECOVERY.json",
                summary)
    atomic_text(run_root / "reports/SCENE_CONTINUATION_AUDIT_RECOVERY.md",
                "# Scene Continuation Audit Recovery\n\n" +
                "\n".join(f"- {key}: `{value}`" for key, value in
                          summary.items() if not isinstance(value, (dict, list))) +
                "\n")
    atomic_json(run_root / "run_status.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--continuation-run", type=Path,
                        default=DEFAULT_CONTINUATION_RUN)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--memorax-credentials", type=Path,
                        default=DEFAULT_CREDENTIALS)
    parser.add_argument("--allow-insecure-memorax-http", action="store_true")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    report = preflight(
        source_run=args.source_run, continuation_run=args.continuation_run,
        run_root=args.run_root, config_path=args.config,
        tokenizer_path=args.tokenizer,
        credentials_path=args.memorax_credentials,
        allow_insecure_http=args.allow_insecure_memorax_http)
    if not args.live:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["decision"] == "pass" else 2
    summary = execute_live(
        report=report, source_run=args.source_run,
        continuation_run=args.continuation_run, run_root=args.run_root,
        config_path=args.config, tokenizer_path=args.tokenizer,
        credentials_path=args.memorax_credentials,
        allow_insecure_http=args.allow_insecure_memorax_http)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
