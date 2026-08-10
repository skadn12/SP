"""Single-use Fresh R1 latest-state-aware Expansion canary."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..core import atomic_json, atomic_text, sha256_file
from ..ledger import RequestLedger
from ..live_backends import load_challenger_config
from ..simple_append_memorax_v1_fullr1_durable_devdiv_probe_v1.context import (
    build_fresh_r1_context,
)
from ..simple_append_v1.assembly import (
    assemble_expanded_dialogue, expanded_dialogue_markdown,
    validate_expanded_dialogue,
)
from .contracts import (
    compile_round_plan, materialize_resolved_developments,
    normalize_round_realizer_output, validate_planner_continuation,
    validate_planner_output, validate_resolver_output,
)
from .live_runtime import (
    NoRetryOpenAIClient, StateResolvedLiveCaller, aggregate_usage,
)
from .preflight import _markdown as preflight_markdown
from .preflight import run_preflight
from .prompts import (
    build_planner_prompt, build_round_realizer_prompt,
    build_state_resolver_prompt,
)
from .settings import (
    CONTEXT_LIMIT, DEFAULT_API_CONFIG, DEFAULT_INPUT, DEFAULT_OVERVIEW,
    DEFAULT_REPORT_ROOT, DEFAULT_RUN_ROOT, DEFAULT_TOKENIZER,
    LIVE_TIMEOUT_SECONDS, PLANNER_MAX_TOKENS, RESOLVER_MAX_TOKENS,
    ROUND_REALIZER_MAX_TOKENS, protected_integrity_snapshot, source_manifest,
)


RUNNER_VERSION = "simple_append_state_resolved_expansion_live_v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected_object:{path}")
    return value


def _prompt_artifact(run_root: Path, role: str, prompt: Mapping[str, Any]) -> None:
    atomic_text(run_root / role / "prompt.txt", str(prompt["prompt"]))
    atomic_json(run_root / role / "prompt_manifest.json", {
        key: value for key, value in prompt.items() if key != "prompt"
    })


class FreezeGuard:
    def __init__(self, *, run_root: Path, freeze: Mapping[str, Any]) -> None:
        self.run_root = run_root
        self.freeze = dict(freeze)
        self.first_api_at: str | None = None

    def before_request(self, logical_role: str, role_version: str) -> None:
        if logical_role not in {"state_resolver", "planner", "round_realizer"}:
            raise RuntimeError(
                f"state_resolved_forbidden_role:{logical_role}:{role_version}")
        if source_manifest()["manifest_sha256"] != (
                self.freeze["source_manifest"]["manifest_sha256"]):
            raise RuntimeError("state_resolved_source_drift_after_freeze")
        for path_key, sha_key in (
            ("input_path", "input_sha256"),
            ("overview_path", "overview_sha256"),
            ("api_config_path", "api_config_sha256"),
        ):
            if sha256_file(Path(self.freeze[path_key])) != self.freeze[sha_key]:
                raise RuntimeError(f"state_resolved_{path_key}_drift_after_freeze")
        if self.first_api_at is None:
            self.first_api_at = _now()
            atomic_json(self.run_root / "run_status.json", {
                "status": "live_requests_started",
                "first_api_request_at": self.first_api_at,
                "code_prompt_input_configuration_frozen": True,
            })


def _request_rows(records: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        metadata = record.get("response_metadata") or {}
        rows.append({
            "request_id": record.get("request_id"),
            "logical_role": record.get("logical_role"),
            "finish_reason": metadata.get("finish_reason"),
            "transport_attempt_count": metadata.get("transport_attempt_count"),
            "transport_retry_count": metadata.get("transport_retry_count"),
            "prompt_tokens": (
                record.get("provider_prompt_tokens")
                if record.get("provider_prompt_tokens") is not None
                else record.get("prompt_tokens_local")),
            "completion_tokens": (
                record.get("provider_completion_tokens")
                if record.get("provider_completion_tokens") is not None
                else record.get("completion_tokens_local")),
            "total_tokens": (
                record.get("provider_total_tokens")
                if record.get("provider_total_tokens") is not None
                else record.get("total_tokens_local")),
            "parse_errors": list(record.get("parse_errors") or []),
        })
    return rows


def execute_live(
    *, preflight: Mapping[str, Any], input_path: Path, overview_path: Path,
    run_root: Path, api_config_path: Path, tokenizer_path: Path,
) -> dict[str, Any]:
    if preflight.get("decision") != "pass":
        raise RuntimeError("STATE_RESOLVED_EXPANSION_PRE_API_PREFLIGHT_FAILED")
    if run_root.exists():
        raise FileExistsError(f"state_resolved_run_root_exists:{run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    for relative in (
        "resolver", "planner", "realizer", "reports", "attempts",
        "raw/prompts", "raw/backend_responses", "ledger",
    ):
        (run_root / relative).mkdir(parents=True, exist_ok=True)
    freeze = {
        **dict(preflight),
        "freeze_version": RUNNER_VERSION,
        "frozen_at": _now(),
        "fresh": True,
        "no_resume": True,
        "only_live_roles": ["state_resolver", "planner", "round_realizer"],
    }
    atomic_json(run_root / "run_freeze_manifest.json", freeze)
    atomic_json(run_root / "run_status.json", {
        "status": "initialized", "created_at": _now(), "api_calls": 0,
    })

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    token_counter = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    config = replace(
        load_challenger_config(api_config_path),
        timeout_seconds=LIVE_TIMEOUT_SECONDS)
    ledger = RequestLedger(
        run_root, manifest_sha256=freeze["source_manifest"]["manifest_sha256"],
        backend_mode="live")
    guard = FreezeGuard(run_root=run_root, freeze=freeze)
    caller = StateResolvedLiveCaller(
        run_root=run_root, ledger=ledger,
        client=NoRetryOpenAIClient(config), token_counter=token_counter,
        root_id=str(freeze["root_id"]), context_limit=CONTEXT_LIMIT,
        before_request=guard.before_request)

    failure_stage: str | None = None
    failure_source: str | None = None
    runtime_error: str | None = None
    result: dict[str, Any] = {}
    try:
        parent = _load(input_path)
        overview = _load(overview_path)
        tail = build_fresh_r1_context(parent, token_counter=token_counter)
        atomic_json(run_root / "selected_parent_context.json", tail)
        atomic_json(run_root / "memory_pre_generation_overview.json", overview)

        resolver_prompt = build_state_resolver_prompt(
            tail_context=tail, memory_overview=overview)
        _prompt_artifact(run_root, "resolver", resolver_prompt)
        resolver_output, _, resolver_parse_errors = caller.call_json(
            request_id="state_resolver_attempt_01", role="state_resolver_v1",
            prompt=str(resolver_prompt["prompt"]),
            prompt_version=str(resolver_prompt["prompt_version"]),
            max_tokens=RESOLVER_MAX_TOKENS)
        resolver_validation = validate_resolver_output(
            resolver_output, parent_turns=parent["dialogue"]["turns"])
        resolver_validation["parse_errors"] = resolver_parse_errors
        atomic_json(run_root / "resolver/output.json", resolver_output or {})
        atomic_json(run_root / "resolver/validation.json", resolver_validation)
        if resolver_parse_errors or not resolver_validation["valid"]:
            failure_stage = "state_resolver_contract"
            failure_source = "state_resolution"
            raise RuntimeError("state_resolver_contract_failed")
        resolved = materialize_resolved_developments(
            resolver_output or {}, parent_turns=parent["dialogue"]["turns"],
            root_id=parent["root_id"],
            parent_dialogue_sha256=parent["dialogue"]["dialogue_hash"])
        atomic_json(run_root / "resolver/resolved_developments.json", resolved)
        result["resolver_validation"] = resolver_validation
        result["resolved_developments"] = resolved

        planner_prompt = build_planner_prompt(
            resolved_set=resolved, parent_speakers=tail["parent_speakers"])
        _prompt_artifact(run_root, "planner", planner_prompt)
        planner_output, _, planner_parse_errors = caller.call_json(
            request_id="planner_attempt_01", role="state_resolved_planner_v1",
            prompt=str(planner_prompt["prompt"]),
            prompt_version=str(planner_prompt["prompt_version"]),
            max_tokens=PLANNER_MAX_TOKENS)
        planner_validation = validate_planner_output(
            planner_output, resolved_set=resolved,
            parent_speakers=tail["parent_speakers"])
        planner_validation["parse_errors"] = planner_parse_errors
        atomic_json(run_root / "planner/output.json", planner_output or {})
        atomic_json(run_root / "planner/validation.json", planner_validation)
        if planner_parse_errors or not planner_validation["valid"]:
            failure_stage = "planner_contract"
            failure_source = "planner_continuation"
            raise RuntimeError("state_resolved_planner_contract_failed")
        continuation = validate_planner_continuation(
            planner_output or {}, resolved_set=resolved)
        atomic_json(run_root / "planner/continuation_gate.json", continuation)
        if not continuation["valid"]:
            failure_stage = "planner_semantic_continuation"
            failure_source = "planner_continuation"
            raise RuntimeError("state_resolved_planner_repeated_latest_state")
        compiled = compile_round_plan(
            planner_output or {}, resolved_set=resolved, tail_context=tail)
        atomic_json(run_root / "planner/compiled_round_plan.json", compiled)
        result["planner_validation"] = planner_validation
        result["planner_continuation"] = continuation
        result["compiled_plan"] = compiled

        realizer_prompt = build_round_realizer_prompt(
            compiled_plan=compiled, tail_context=tail)
        _prompt_artifact(run_root, "realizer", realizer_prompt)
        realizer_output, _, realizer_parse_errors = caller.call_json(
            request_id="round_realizer_attempt_01",
            role="state_resolved_round_realizer_v1",
            prompt=str(realizer_prompt["prompt"]),
            prompt_version=str(realizer_prompt["prompt_version"]),
            max_tokens=ROUND_REALIZER_MAX_TOKENS)
        normalized = normalize_round_realizer_output(
            realizer_output, compiled_plan=compiled,
            parent_utterances=[row["text"] for row in parent["dialogue"]["turns"]],
            parent_turn_refs=[row["turn_id"] for row in parent["dialogue"]["turns"]])
        normalized["parse_errors"] = realizer_parse_errors
        atomic_json(run_root / "realizer/output.json", realizer_output or {})
        atomic_json(run_root / "realizer/validation.json", normalized)
        if realizer_parse_errors or not normalized["valid"]:
            failure_stage = "realizer_contract"
            failure_source = "realizer_adherence"
            raise RuntimeError("state_resolved_realizer_contract_failed")
        child = normalized["child"]
        expanded = assemble_expanded_dialogue(parent=parent, child=child)
        expanded_validation = validate_expanded_dialogue(expanded, parent=parent)
        atomic_json(run_root / "child_dialogue.json", child)
        atomic_json(run_root / "expanded_dialogue.json", expanded)
        atomic_json(run_root / "expanded_dialogue_validation.json", expanded_validation)
        if not expanded_validation["valid"]:
            failure_stage = "parent_child_assembly"
            failure_source = "realizer_adherence"
            raise RuntimeError("state_resolved_assembly_failed")
        atomic_text(run_root / "expanded_dialogue.md", expanded_dialogue_markdown(expanded))
        result["realizer_validation"] = normalized
        result["expanded_dialogue_validation"] = expanded_validation
        result["child_dialogue"] = child
        result["expanded_dialogue"] = expanded
    except Exception as error:
        runtime_error = f"{type(error).__name__}:{error}"
        if failure_stage is None:
            failure_stage = "runtime"
            failure_source = "infrastructure"

    usage = aggregate_usage(caller.records)
    success = (
        failure_stage is None and
        (result.get("expanded_dialogue_validation") or {}).get("valid") is True
    )
    compiled_scenes = list((result.get("compiled_plan") or {}).get("scenes") or [])
    child_scenes = list((result.get("child_dialogue") or {}).get("scenes") or [])
    child_by_ref = {str(row.get("scene_ref")): row for row in child_scenes}
    scene_rows = []
    for scene in compiled_scenes:
        development = scene.get("resolved_development") or {}
        scene_rows.append({
            "scene_ref": scene.get("scene_ref"),
            "development_id": scene.get("development_id"),
            "event_summary": development.get("event_summary"),
            "resolved_latest_state": development.get("latest_state"),
            "latest_state_evidence_turn_refs": development.get("evidence_turn_refs"),
            "planned_next_development": scene.get("new_development"),
            "planned_durable_outcome": scene.get("durable_outcome"),
            "already_existed_in_parent_mechanical_gate": False,
            "actual_turn_count": len(
                (child_by_ref.get(str(scene.get("scene_ref"))) or {}).get(
                    "dialogue_turns") or []),
            "actual_child_development": None,
            "durable_outcome_realized": None,
            "valid_continuation": None,
            "semantic_audit_status": "pending_human_offline_audit",
        })
    integrity_after = protected_integrity_snapshot()
    integrity = {
        "input_unchanged": sha256_file(input_path) == freeze["input_sha256"],
        "overview_unchanged": sha256_file(overview_path) == freeze["overview_sha256"],
        "api_config_unchanged": sha256_file(api_config_path) == freeze["api_config_sha256"],
        "source_manifest_unchanged": source_manifest()["manifest_sha256"] == (
            freeze["source_manifest"]["manifest_sha256"]),
        "protected_integrity_unchanged": integrity_after == freeze["protected_integrity"],
        "after": integrity_after,
    }
    summary = {
        "summary_version": RUNNER_VERSION,
        "status": "completed" if success else "failed",
        "failure_stage": failure_stage,
        "failure_source": failure_source,
        "runtime_error": runtime_error,
        "run_root": str(run_root),
        "root_id": freeze["root_id"],
        "api_usage": usage,
        "requests": _request_rows(caller.records),
        "state_resolver_calls": usage["roles"]["state_resolver"]["requests"],
        "planner_calls": usage["roles"]["planner"]["requests"],
        "realizer_calls": usage["roles"]["round_realizer"]["requests"],
        "retries": 0,
        "memorax_calls": 0,
        "qa_calls": 0,
        "solver_calls": 0,
        "resolver_development_count": len(
            (result.get("resolved_developments") or {}).get("developments") or []),
        "planner_scene_count": len(compiled_scenes),
        "child_turn_count": int(
            (result.get("expanded_dialogue_validation") or {}).get("child_turn_count") or 0),
        "scenes": scene_rows,
        "generation_ready": success,
        "qa_pipeline_ready": False,
        "state_resolved_expansion_supported": None,
        "latest_state_repetition_reduced": None,
        "planner_track_separation_passed": (
            len({row.get("development_id") for row in compiled_scenes}) == 3
            if compiled_scenes else False),
        "semantic_audit_required": success,
        "integrity": integrity,
        "cache_reuse": 0,
        "imported_response_reuse": 0,
        "completed_at": _now(),
    }
    atomic_json(run_root / "reports/STATE_RESOLVED_EXPANSION_LIVE_REPORT.json", summary)
    atomic_text(run_root / "reports/STATE_RESOLVED_EXPANSION_LIVE_REPORT.md", "\n".join([
        "# State-resolved Expansion Live Report", "",
        f"- Status: `{summary['status']}`",
        f"- Failure stage: `{summary['failure_stage']}`",
        f"- Failure source: `{summary['failure_source']}`",
        f"- Resolver / Planner / Realizer calls: `"
        f"{summary['state_resolver_calls']} / {summary['planner_calls']} / "
        f"{summary['realizer_calls']}`",
        f"- Total tokens: `{usage['total_tokens']}`",
        f"- Resolved developments: `{summary['resolver_development_count']}`",
        f"- Planner scenes: `{summary['planner_scene_count']}`",
        f"- Child turns: `{summary['child_turn_count']}`",
        f"- Generation ready: `{str(summary['generation_ready']).lower()}`",
        "- QA, Solver, Memorax API, Repair, R2/R3, and training calls: `0`",
        "- Semantic continuation fields remain pending until offline human audit.", "",
    ]))
    atomic_json(run_root / "reports/MECHANICAL_CONTINUATION_TABLE.json", {
        "scenes": scene_rows,
        "semantic_audit_required": success,
    })
    atomic_json(run_root / "run_status.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--overview", type=Path, default=DEFAULT_OVERVIEW)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--preflight-report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    preflight = run_preflight(
        input_path=args.input.resolve(), overview_path=args.overview.resolve(),
        run_root=args.run_root.resolve(), api_config_path=args.api_config.resolve(),
        tokenizer_path=args.tokenizer.resolve())
    if args.preflight_only:
        atomic_json(
            args.preflight_report_root / "STATE_RESOLVED_EXPANSION_PREFLIGHT.json",
            preflight)
        atomic_text(
            args.preflight_report_root / "STATE_RESOLVED_EXPANSION_PREFLIGHT.md",
            preflight_markdown(preflight))
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0 if preflight["decision"] == "pass" else 2
    if preflight["decision"] != "pass":
        print(json.dumps(
            {"status": "PRE_API_PREFLIGHT_FAILED", **preflight},
            ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    summary = execute_live(
        preflight=preflight, input_path=args.input.resolve(),
        overview_path=args.overview.resolve(), run_root=args.run_root.resolve(),
        api_config_path=args.api_config.resolve(), tokenizer_path=args.tokenizer.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
