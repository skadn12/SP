"""Single fresh R1 with a strict Planner-to-Realizer continuation gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from dataclasses import replace

from ..core import atomic_json, atomic_text, sha256_bytes, sha256_file
from ..ledger import RequestLedger
from ..live_backends import OpenAICompatibleChallengerClient, load_challenger_config
from ..unified_dialogue_root import validate_unified_dialogue_root
from .client import MemoraxClient, load_memorax_config
from .context import build_fresh_r1_context
from .live_runtime import SimpleAppendLiveCaller, aggregate_usage
from .payloads import MEMORAX_USER_NAMESPACE, build_bucket_search_payload, memorax_user_id
from .workflow import WorkflowError, execute_round


RUNNER_VERSION = "memory_event_continuation_semantic_gate_probe_v1"
DEFAULT_INPUT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-inputs/conversation_chronicles_v1/"
    "roots/cc-train-episode-13836/unified_root.json")
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_13836_simple_append_memorax_"
    "memory_event_continuation_semantic_gate_probe_v1_live_3_recovery_1")
DEFAULT_SOURCE_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_13836_simple_append_memorax_"
    "memory_event_continuation_semantic_gate_probe_v1_live_3")
DEFAULT_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
DEFAULT_TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")
DEFAULT_CREDENTIALS = Path("/root/.config/memorax/credentials.env")
CONTEXT_LIMIT = 40_960
PLANNER_MAX_TOKENS = 4_096
SEMANTIC_GATE_MAX_TOKENS = 2_048
REALIZER_MAX_TOKENS = 8_192
LIVE_TIMEOUT_SECONDS = 600


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _manifest() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    rows = [{"path": str(p), "sha256": sha256_file(p)}
            for p in sorted(root.glob("*.py")) if p.is_file()]
    body = json.dumps(rows, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode()
    return {"files": rows, "manifest_sha256": hashlib.sha256(body).hexdigest()}


def _config(path: Path):
    return replace(load_challenger_config(path), timeout_seconds=LIVE_TIMEOUT_SECONDS)


def preflight(*, input_path: Path, run_root: Path, config_path: Path,
              tokenizer_path: Path, credentials_path: Path,
              allow_insecure_http: bool, reuse_existing_namespace: bool = False,
              source_run_root: Path = DEFAULT_SOURCE_RUN_ROOT,
              memorax_user_namespace: str = MEMORAX_USER_NAMESPACE,
              assume_new_namespace_empty: bool = False) -> dict[str, Any]:
    checks = {
        "run_root_absent": not run_root.exists(),
        "input_exists": input_path.is_file(),
        "config_exists": config_path.is_file(),
        "tokenizer_exists": tokenizer_path.exists(),
        "credentials_exists": credentials_path.is_file(),
        "api_calls_zero": True,
        "qa_calls_zero": True,
        "expansion_calls_zero": True,
        "semantic_gate_calls_zero": True,
        "memorax_platform_memory": True,
        "recovery_source_exists": (not reuse_existing_namespace) or source_run_root.is_dir(),
        "recovery_parent_add_record": (not reuse_existing_namespace) or (
            source_run_root / "memory" / "parent_add_completion.json").is_file(),
        "recovery_child_add_absent": (not reuse_existing_namespace) or not (
            source_run_root / "memory" / "child_add_completion.json").is_file(),
        "recovery_namespace_matches": True,
        "new_namespace_assumption_recorded": (
            not assume_new_namespace_empty or bool(memorax_user_namespace)),
    }
    details: dict[str, Any] = {}
    parent: dict[str, Any] = {}
    api_config = None
    memory_config = None
    context = None
    try:
        parent = json.loads(input_path.read_text(encoding="utf-8"))
        expected_user_id = memorax_user_id(
            str(parent.get("root_id") or ""), namespace=memorax_user_namespace)
        validation = validate_unified_dialogue_root(parent)
        checks["authoritative_parent_valid"] = bool(validation["valid"])
        details["parent_validation"] = validation
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        counter = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
        context = build_fresh_r1_context(parent, token_counter=counter, budget_tokens=10_000)
        checks["full_parent_included"] = bool(context["full_parent_included"])
        checks["complete_sessions_only"] = bool(context["complete_sessions_only"])
        checks["parent_tail_preserved"] = context["parent_tail_turn_id"] == parent["dialogue"]["turns"][-1]["turn_id"]
        if reuse_existing_namespace:
            parent_intent_path = source_run_root / "memory" / "parent_add_intent.json"
            if parent_intent_path.is_file():
                source_intent = json.loads(parent_intent_path.read_text(encoding="utf-8"))
                checks["recovery_namespace_matches"] = (
                    source_intent.get("user_id") == expected_user_id)
            else:
                checks["recovery_namespace_matches"] = False
    except Exception as error:
        details["parent_or_tokenizer_error"] = f"{type(error).__name__}:{error}"
        checks.update({"authoritative_parent_valid": False, "full_parent_included": False,
                       "complete_sessions_only": False, "parent_tail_preserved": False})
    try:
        api_config = _config(config_path)
        checks["model_configured"] = bool(api_config.model)
        checks["timeout_frozen"] = api_config.timeout_seconds == LIVE_TIMEOUT_SECONDS
    except Exception as error:
        details["api_config_error"] = f"{type(error).__name__}:{error}"
        checks.update({"model_configured": False, "timeout_frozen": False})
    try:
        memory_config = load_memorax_config(credentials_path=credentials_path,
                                            allow_insecure_http=allow_insecure_http)
        checks["memorax_platform_memory"] = memory_config.platform == "memory"
        checks["memorax_transport_authorized"] = (
            memory_config.normalized_base_url.startswith("https://") or allow_insecure_http)
    except Exception as error:
        details["memorax_config_error"] = f"{type(error).__name__}:{error}"
        checks["memorax_transport_authorized"] = False
    return {
        "preflight_version": RUNNER_VERSION,
        "decision": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "failed_checks": [k for k, v in checks.items() if not v],
        "details": details,
        "input_path": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path) if input_path.is_file() else None,
        "root_id": parent.get("root_id"),
        "parent_dialogue_sha256": (parent.get("dialogue") or {}).get("dialogue_hash"),
        "run_root": str(run_root.resolve()),
        "reuse_existing_namespace": bool(reuse_existing_namespace),
        "assume_new_namespace_empty": bool(assume_new_namespace_empty),
        "source_run_root": str(source_run_root.resolve()) if reuse_existing_namespace else None,
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path) if config_path.is_file() else None,
        "tokenizer_path": str(tokenizer_path.resolve()),
        "credentials_path": str(credentials_path.resolve()),
        "memory_user_id": (memorax_user_id(
            parent.get("root_id"), namespace=memorax_user_namespace)
            if parent.get("root_id") else None),
        "memorax_user_namespace": memorax_user_namespace,
        "api_configuration": ({"model": api_config.model, "base_url": api_config.base_url,
            "temperature": api_config.temperature, "timeout_seconds": api_config.timeout_seconds,
            "api_key_recorded": False} if api_config else None),
        "memorax_configuration": memory_config.redacted() if memory_config else None,
        "request_budget": {"isolation_search": (
                0 if reuse_existing_namespace or assume_new_namespace_empty else 1),
            "parent_add": 0 if reuse_existing_namespace else 1,
            "pre_search": 1, "planner": 1,
            "semantic_continuation_gate": 1, "realizer": 1,
            "child_add": 1, "post_search": 1,
            "qa": 0, "solver": 0, "round_2": 0},
        "max_completion_tokens": {"planner": PLANNER_MAX_TOKENS,
                                   "semantic_continuation_gate": SEMANTIC_GATE_MAX_TOKENS,
                                   "realizer": REALIZER_MAX_TOKENS},
        "context": ({"turn_count": context["selected_turn_count"],
                     "session_count": context["selected_session_count"],
                     "raw_tokens": context["selected_raw_dialogue_tokens"]} if context else None),
        "source_manifest": _manifest(),
        "api_calls": 0,
    }


class FreezeGuard:
    def __init__(self, run_root: Path, freeze: Mapping[str, Any],
                 input_path: Path, config_path: Path):
        self.run_root, self.freeze = run_root, dict(freeze)
        self.input_path, self.config_path = input_path, config_path
        self.first_api_at = None

    def before_request(self, logical_role: str, role_version: str) -> None:
        allowed = {"planner", "semantic_continuation_gate", "round_realizer",
                   "isolation_search", "parent_add",
                   "pre_generation_search", "child_add", "post_generation_search", "add_status"}
        if logical_role not in allowed:
            raise RuntimeError(f"memory_event_continuation_forbidden_role:{logical_role}")
        if _manifest()["manifest_sha256"] != self.freeze["source_manifest"]["manifest_sha256"]:
            raise RuntimeError("memory_event_continuation_source_drift")
        if sha256_file(self.input_path) != self.freeze["input_sha256"]:
            raise RuntimeError("memory_event_continuation_input_drift")
        if sha256_file(self.config_path) != self.freeze["config_sha256"]:
            raise RuntimeError("memory_event_continuation_config_drift")
        if self.first_api_at is None:
            self.first_api_at = _now()
            atomic_json(self.run_root / "run_status.json", {
                "status":"live_requests_started", "first_api_request_at":self.first_api_at,
                "code_prompt_input_configuration_frozen":True})


def _sink(run_root: Path):
    def write(name: str, value: Any):
        path = run_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, Mapping) and isinstance(value.get("prompt"), str):
            atomic_text(path.with_suffix(".txt"), str(value["prompt"]))
            atomic_json(path.with_name(path.name + "_manifest.json"),
                        {k:v for k,v in value.items() if k != "prompt"})
        else:
            atomic_json(path.with_suffix(".json"), value)
    return write


def execute_live(*, report: Mapping[str, Any], input_path: Path, run_root: Path,
                 config_path: Path, tokenizer_path: Path, credentials_path: Path,
                 allow_insecure_http: bool,
                 reuse_existing_namespace: bool = False,
                 memorax_user_namespace: str = MEMORAX_USER_NAMESPACE,
                 assume_new_namespace_empty: bool = False) -> dict[str, Any]:
    if report.get("decision") != "pass":
        raise RuntimeError("MEMORY_EVENT_CONTINUATION_PREFLIGHT_FAILED")
    if run_root.exists():
        raise FileExistsError(f"memory_event_continuation_run_exists:{run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    for part in ("planner", "semantic_gate", "realizer", "memory", "reports",
                 "raw/prompts", "attempts"):
        (run_root / part).mkdir(parents=True, exist_ok=True)
    freeze = {**dict(report), "freeze_version":RUNNER_VERSION,
              "frozen_at":_now(), "fresh":True, "no_resume":True,
              "first_api_request":None}
    atomic_json(run_root / "run_freeze_manifest.json", freeze)
    atomic_json(run_root / "run_status.json", {"status":"initialized", "api_calls":0})
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
    counter=lambda text: len(tokenizer.encode(text,add_special_tokens=False))
    ledger=RequestLedger(run_root,manifest_sha256=freeze["source_manifest"]["manifest_sha256"],backend_mode="live")
    guard=FreezeGuard(run_root,freeze,input_path,config_path)
    caller=SimpleAppendLiveCaller(run_root=run_root,ledger=ledger,
        client=OpenAICompatibleChallengerClient(_config(config_path)),token_counter=counter,
        root_id=str(freeze["root_id"]),context_limit=CONTEXT_LIMIT,before_request=guard.before_request)
    memorax=MemoraxClient(load_memorax_config(credentials_path=credentials_path,
                                               allow_insecure_http=allow_insecure_http))
    failure_stage=None; runtime_error=None; result={}
    def planner_call(prompt):
        parsed,_,errors=caller.call_json(request_id="planner_attempt_01",role="simple_append_planner_v1",
            prompt=str(prompt["prompt"]),prompt_version=str(prompt["prompt_version"]),max_tokens=PLANNER_MAX_TOKENS)
        if errors: raise WorkflowError("planner_transport_or_parse",errors)
        return parsed
    def semantic_gate_call(prompt):
        parsed,_,errors=caller.call_json(
            request_id="semantic_continuation_gate_attempt_01",
            role="memory_event_semantic_continuation_gate_v1",
            prompt=str(prompt["prompt"]),
            prompt_version=str(prompt["prompt_version"]),
            max_tokens=SEMANTIC_GATE_MAX_TOKENS)
        if errors:
            raise WorkflowError("semantic_continuation_gate_transport_or_parse",errors)
        return parsed
    def realizer_call(prompt):
        parsed,_,errors=caller.call_json(request_id="round_realizer_attempt_01",role="simple_append_round_realizer_v1",
            prompt=str(prompt["prompt"]),prompt_version=str(prompt["prompt_version"]),max_tokens=REALIZER_MAX_TOKENS)
        if errors: raise WorkflowError("realizer_transport_or_parse",errors)
        return parsed
    try:
        parent=json.loads(input_path.read_text(encoding="utf-8"))
        result=execute_round(parent=parent,memorax=memorax,planner_call=planner_call,
            semantic_gate_call=semantic_gate_call,realizer_call=realizer_call,
            artifact_sink=_sink(run_root),
            before_memory_request=lambda stage, request_id: guard.before_request(stage,request_id),
            token_counter=counter,wait_for_memory_completion=True,
            reuse_existing_namespace=reuse_existing_namespace,
            memorax_user_namespace=memorax_user_namespace,
            assume_new_namespace_empty=assume_new_namespace_empty)
        atomic_text(run_root/"expanded_dialogue.md", _expanded_markdown(result["expanded_dialogue"]))
    except WorkflowError as error:
        failure_stage=error.stage; runtime_error=str(error)
    except Exception as error:
        failure_stage="runtime"; runtime_error=f"{type(error).__name__}:{error}"
    usage=aggregate_usage(caller.records)
    memory_records=memorax.calls
    add_records = [row for row in memory_records
                   if str(row.get("path")) == "/v1/memories/add"]
    parent_add_calls = 0 if reuse_existing_namespace else sum(
        "-r00-" in str(row.get("request_id")) for row in add_records)
    child_add_calls = sum(
        "-r01-" in str(row.get("request_id")) for row in add_records)
    def artifact(name):
        path=run_root/name
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
    planner_raw_output=result.get("planner_raw_output") or artifact(Path("planner/output.json"))
    planner_output=result.get("planner_output") or artifact(Path("planner/materialized_plan.json"))
    planner_output=planner_output or planner_raw_output
    planner_validation=result.get("planner_validation") or artifact(Path("planner/validation.json"))
    gate_output=result.get("semantic_gate_output") or artifact(Path("semantic_gate/output.json"))
    gate_validation=result.get("semantic_gate_validation") or artifact(Path("semantic_gate/validation.json"))
    candidate_selection=result.get("candidate_selection_audit") or artifact(
        Path("semantic_gate/candidate_selection.json"))
    summary={"summary_version":RUNNER_VERSION,"status":"completed" if result.get("generation_ready") else "failed",
        "failure_stage":failure_stage,"runtime_error":runtime_error,"run_root":str(run_root),
        "root_id":freeze.get("root_id"),"user_id":freeze.get("memory_user_id"),
        "planner_calls":usage["roles"]["planner"]["requests"],
        "semantic_gate_calls":usage["roles"]["semantic_continuation_gate"]["requests"],
        "realizer_calls":usage["roles"]["round_realizer"]["requests"],
        "planner_retries":0,"semantic_gate_retries":0,"realizer_retries":0,
        "llm_usage":usage,
        "memorax_calls":len(memory_records),"memorax_call_records":memory_records,
        "parent_add_calls":int(parent_add_calls),
        "child_add_calls": int(child_add_calls),
        "parent_add_skipped":bool(result.get("parent_add_skipped", reuse_existing_namespace)),
        "existing_namespace_reused":bool(reuse_existing_namespace),
        "memory_event_candidates":result.get("candidates",[]),
        "planner_validation":planner_validation,
        "planner_raw_output":planner_raw_output,
        "planner_materialized_plan":planner_output if planner_output is not planner_raw_output else None,
        "planner_scenes":planner_output.get("scenes",[]) if isinstance(planner_output,Mapping) else [],
        "semantic_gate_output":gate_output,
        "semantic_gate_validation":gate_validation,
        "candidate_selection_audit":candidate_selection,
        "realizer_validation":result.get("realizer_validation"),
        "realizer_input_isolation":result.get("realizer_input_isolation") or artifact(
            Path("realizer/input_validation.json")),
        "parent_add_verified":bool(result.get("parent_add_completion")),"child_add_verified":bool(result.get("child_add_completion")),
        "memory_write_verified":result.get("memory_write_verified",False),"memory_visibility_verified":result.get("memory_visibility_verified",False),
        "generation_ready":bool(result.get("generation_ready")),"qa_pipeline_ready":False,
        "expansion_calls":usage["total_requests"],
        "planner_gate_realizer_only_llm":True,"qa_calls":0,"solver_calls":0,
        "round_2":False,"r2":False,"r3":False,"training":False,"cache_reuse":0,"imported_response_reuse":0,
        "input_sha256":freeze.get("input_sha256"),"config_sha256":freeze.get("config_sha256"),
        "source_manifest":freeze.get("source_manifest"),"completed_at":_now()}
    atomic_json(run_root/"reports/MEMORY_EVENT_CONTINUATION_LIVE_REPORT.json",summary)
    atomic_text(run_root/"reports/MEMORY_EVENT_CONTINUATION_LIVE_REPORT.md",_summary_md(summary))
    atomic_json(run_root/"run_status.json",{"status":summary["status"],"failure_stage":failure_stage,"api_calls":len(memorax.calls)+len(caller.records)})
    return summary


def _expanded_markdown(expanded):
    lines=["# Expanded Dialogue","",f"Root: `{expanded.get('root_id')}`",""]
    parent_count=int(expanded.get("parent_turn_count") or 0)
    for i,row in enumerate(expanded.get("turns") or []):
        if i==parent_count:
            lines.extend(["", "---", "", "# CHILD START - ROUND 1", ""])
        label=row.get("scene_ref") or row.get("session_ref")
        lines.append(f"- `{row.get('turn_ref')}` | `{label}` | **{row.get('speaker')}**: {row.get('text')}")
    return "\n".join(lines)+"\n"


def _summary_md(s):
    return "\n".join(["# Memory Event Continuation Live Report","",
      f"- Status: `{s['status']}`",f"- Failure stage: `{s.get('failure_stage')}`",
      f"- Root: `{s.get('root_id')}`",f"- Planner calls: `{s['planner_calls']}`",
      f"- Semantic gate calls: `{s['semantic_gate_calls']}`",
      f"- Semantic gate decision: `{(s.get('semantic_gate_validation') or {}).get('overall_decision')}`",
      f"- Realizer calls: `{s['realizer_calls']}`",f"- Memorax calls: `{s['memorax_calls']}`",
      f"- Memory write verified: `{s['memory_write_verified']}`",
      f"- Memory visibility verified: `{s['memory_visibility_verified']}`",
      f"- Generation ready: `{s['generation_ready']}`", "- QA calls: `0`",
      "- Solver calls: `0`", "- Round 2/3: `false/false`", "- Training: `false`", "",
      "See the JSON report and stage artifacts for complete bucket, plan, dialogue, and write records.",""])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=DEFAULT_INPUT)
    parser.add_argument("--run-root",type=Path,default=DEFAULT_RUN_ROOT)
    parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    parser.add_argument("--tokenizer",type=Path,default=DEFAULT_TOKENIZER)
    parser.add_argument("--memorax-credentials",type=Path,default=DEFAULT_CREDENTIALS)
    parser.add_argument("--allow-insecure-memorax-http",action="store_true")
    parser.add_argument("--reuse-existing-namespace",action="store_true")
    parser.add_argument("--source-run-root",type=Path,default=DEFAULT_SOURCE_RUN_ROOT)
    parser.add_argument("--memorax-user-namespace",default=MEMORAX_USER_NAMESPACE)
    parser.add_argument("--assume-new-namespace-empty",action="store_true")
    parser.add_argument("--preflight-only",action="store_true")
    parser.add_argument("--live",action="store_true")
    args=parser.parse_args(argv)
    report=preflight(input_path=args.input,run_root=args.run_root,config_path=args.config,
        tokenizer_path=args.tokenizer,credentials_path=args.memorax_credentials,
        allow_insecure_http=args.allow_insecure_memorax_http,
        reuse_existing_namespace=args.reuse_existing_namespace,
        source_run_root=args.source_run_root,
        memorax_user_namespace=args.memorax_user_namespace,
        assume_new_namespace_empty=args.assume_new_namespace_empty)
    print(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True))
    if args.preflight_only or not args.live:
        return 0 if report["decision"]=="pass" else 2
    summary=execute_live(report=report,input_path=args.input,run_root=args.run_root,
        config_path=args.config,tokenizer_path=args.tokenizer,
        credentials_path=args.memorax_credentials,
        allow_insecure_http=args.allow_insecure_memorax_http,
        reuse_existing_namespace=args.reuse_existing_namespace,
        memorax_user_namespace=args.memorax_user_namespace,
        assume_new_namespace_empty=args.assume_new_namespace_empty)
    print(json.dumps(summary,ensure_ascii=False,indent=2,sort_keys=True))
    return 0 if summary["status"]=="completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
