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
from .incremental_round import (
    build_incremental_context,
    load_incremental_round_input,
)
from .live_runtime import (SimpleAppendLiveCaller, aggregate_usage,
                           TOTAL_REQUEST_BUDGET, ROLE_REQUEST_BUDGETS,
                           validate_outbound_role_coverage)
from .payloads import MEMORAX_USER_NAMESPACE, build_bucket_search_payload, memorax_user_id
from .workflow import WorkflowError, execute_round
from .profiles import (
    ExpansionExecutionProfile, LEGACY_PROFILE, R1_ALIGNED_BASELINE_PROFILE,
    R1_ALIGNED_FEEDBACK_PROFILE,
)


def _load_feedback_context(parent: Mapping[str, Any], *, feedback_input: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the frozen local feedback record for one authoritative root.

    This is deliberately local and read-only.  The expanded profile opts into
    the versioned renderer; the legacy profile never calls this helper.
    """
    root_id = str(parent.get("root_id") or "")
    if feedback_input is not None:
        payload = json.loads(feedback_input.read_text(encoding="utf-8"))
        record = payload.get("record") if isinstance(payload, Mapping) and "record" in payload else payload
        if not isinstance(record, Mapping) or str(record.get("root_id") or "") != root_id:
            raise ValueError(f"feedback_record_root_mismatch:{root_id}")
        record = dict(record)
    else:
        from ..h4_round1_r03_feedback_view_v2.compiler import load_frozen_inputs
        frozen = load_frozen_inputs()
        record = next((row for row in frozen["records"]
                       if str(row.get("root_id") or "") == root_id), None)
        if record is None:
            raise ValueError(f"feedback_record_missing_for_root:{root_id}")
    turns = []
    for row in (parent.get("dialogue") or {}).get("turns") or []:
        turns.append({
            "turn_ref": str(row.get("turn_id") or row.get("turn_ref") or ""),
            "speaker": str(row.get("speaker") or ""),
            "text": str(row.get("text") or ""),
        })
    return record, {"root_id": root_id, "turns": turns}


RUNNER_VERSION = "memory_only_fast_expansion_v1"
DEFAULT_INPUT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-inputs/conversation_chronicles_v1/"
    "roots/cc-train-episode-91716/unified_root.json")
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_91716_simple_append_memorax_"
    "memory_only_fast_expansion_v1_live_1")
DEFAULT_SOURCE_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_13836_simple_append_memorax_"
    "memory_event_continuation_semantic_gate_probe_v1_live_3")
DEFAULT_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
DEFAULT_TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")
DEFAULT_CREDENTIALS = Path("/root/.config/memorax/credentials.env")
CONTEXT_LIMIT = 40_960
PLANNER_MAX_TOKENS = 4_096
# The outline auditor returns one compact review per scene pair. 1,024 was
# insufficient for verbose-but-valid provider JSON (it can terminate at the
# token ceiling before the closing braces). Keep the response bounded while
# leaving enough room for the contract-required pair reviews.
PLANNER_OUTLINE_SEMANTIC_MAX_TOKENS = 4_096
EVENT_VERIFIER_MAX_TOKENS = 2_048
SEMANTIC_GATE_MAX_TOKENS = 2_048
REALIZER_MAX_TOKENS = 8_192
REALIZER_ADHERENCE_AUDITOR_MAX_TOKENS = 4_096
SCENE_CONTINUATION_MAX_TOKENS = 4_096
REALIZER_ADHERENCE_REAUDIT_MAX_TOKENS = 4_096
SCENE_REPLACEMENT_MAX_TOKENS = 4_096
LIVE_TIMEOUT_SECONDS = 600
REQUIRED_OUTBOUND_ROLES = (
    "simple_append_planner_v1",
    "simple_append_planner_candidate_repair_v1",
    "planner_outline_semantic_auditor_v1",
    "memory_event_realization_aware_semantic_gate_v1",
    "simple_append_round_realizer_v1",
    "memory_event_realizer_adherence_auditor_v1",
    "simple_append_scene_continuation_realizer_v1",
    "simple_append_scene_replacement_realizer_v1",
    "memory_event_realizer_adherence_reaudit_v1",
)


def _resolve_profile_route(*, profile: ExpansionExecutionProfile,
                           explicit_profile: bool,
                           memory_only_fast: bool | None,
                           event_bucket_only: bool | None) -> tuple[bool, bool]:
    """Resolve route flags without letting CLI defaults change a profile.

    An explicitly selected profile owns its route.  Calls without a profile
    retain the legacy profile defaults, while direct callers may still provide
    explicit legacy-compatible overrides.
    """
    if explicit_profile:
        return bool(profile.memory_only_fast), bool(profile.event_bucket_only)
    return (
        bool(profile.memory_only_fast)
        if memory_only_fast is None else bool(memory_only_fast),
        bool(profile.event_bucket_only)
        if event_bucket_only is None else bool(event_bucket_only),
    )


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
              assume_new_namespace_empty: bool = False,
              memory_only_fast: bool | None = None,
              event_bucket_only: bool | None = None,
              skip_planner_semantic_gate: bool = False,
              single_event_line_mode: bool = False,
              single_event_target_rank: int = 1,
              no_memorax_writes: bool = False,
              round_index: int = 1,
              execution_profile: ExpansionExecutionProfile | None = None,
              feedback_record: Mapping[str, Any] | None = None,
              feedback_dialogue: Mapping[str, Any] | None = None,
              feedback_input: Path | None = None) -> dict[str, Any]:
    profile = execution_profile or LEGACY_PROFILE
    memory_only_fast, event_bucket_only = _resolve_profile_route(
        profile=profile, explicit_profile=execution_profile is not None,
        memory_only_fast=memory_only_fast, event_bucket_only=event_bucket_only)
    feedback_required = bool(profile.feedback_view_version)
    incremental = int(round_index) > 1
    checks = {
        "development_policy_route_compatible": not profile.development_policy or (
            not single_event_line_mode and not skip_planner_semantic_gate),
        "run_root_absent": not run_root.exists(),
        "input_exists": input_path.is_file(),
        "config_exists": config_path.is_file(),
        "tokenizer_exists": tokenizer_path.exists(),
        "credentials_exists": credentials_path.is_file(),
        "api_calls_zero": True,
        "qa_calls_zero": True,
        "expansion_calls_zero": True,
        "semantic_gate_calls_zero": True,
        "planner_outline_semantic_auditor_calls_zero": True,
        "planner_pre_gate_mode_recorded": True,
        "realizer_adherence_auditor_calls_zero": True,
        "scene_continuation_realizer_calls_zero": True,
        "scene_replacement_realizer_calls_zero": True,
        "realizer_adherence_reaudit_calls_zero": True,
        "transcript_event_verifier_disabled": bool(memory_only_fast),
        "event_bucket_only_requires_memory_only_fast": (
            not event_bucket_only or memory_only_fast),
        "single_event_line_requires_memory_only_fast": (
            not single_event_line_mode or memory_only_fast),
        "single_event_target_rank_valid": (
            not single_event_line_mode or
            1 <= int(single_event_target_rank) <= 8),
        "memorax_platform_memory": True,
        "memorax_write_mode_recorded": True,
        "feedback_view_contract_available": (
            not feedback_required or
            (isinstance(feedback_record, Mapping) and
             isinstance(feedback_dialogue, Mapping))),
        "r1_feedback_input_exists": (
            profile.name not in {"r1_aligned_feedback_v1", "development_gain_v2"} or
            (feedback_input is not None and feedback_input.is_file())),
        "recovery_source_exists": (not reuse_existing_namespace) or source_run_root.is_dir(),
        "recovery_parent_add_record": ((not reuse_existing_namespace) or
            incremental or (source_run_root / "memory" /
                            "parent_add_completion.json").is_file()),
        "recovery_child_add_record": ((not reuse_existing_namespace) or
            (not incremental) or (source_run_root / "memory" /
                                  "child_add_completion.json").is_file()),
        "r2_requires_namespace_reuse": (not incremental or
                                         reuse_existing_namespace),
        "r2_requires_nonempty_prior_child": (not incremental or
            (source_run_root / "memory" / "child_add_completion.json").is_file()),
        "recovery_namespace_matches": True,
        "new_namespace_assumption_recorded": (
            not assume_new_namespace_empty or bool(memorax_user_namespace)),
    }
    details: dict[str, Any] = {}
    outbound_role_coverage = validate_outbound_role_coverage(
        REQUIRED_OUTBOUND_ROLES)
    checks["outbound_role_coverage"] = bool(outbound_role_coverage["valid"])
    details["outbound_role_coverage"] = outbound_role_coverage
    parent: dict[str, Any] = {}
    api_config = None
    memory_config = None
    context = None
    try:
        if incremental:
            parent, prior_expanded, prior_validation = load_incremental_round_input(
                input_path, round_index=round_index)
            checks["authoritative_parent_valid"] = bool(prior_validation["valid"])
            details["parent_validation"] = prior_validation
            details["prior_expanded_turn_count"] = len(
                prior_expanded.get("turns") or [])
        else:
            parent = json.loads(input_path.read_text(encoding="utf-8"))
            validation = validate_unified_dialogue_root(parent)
            checks["authoritative_parent_valid"] = bool(validation["valid"])
            details["parent_validation"] = validation
        expected_user_id = memorax_user_id(
            str(parent.get("root_id") or ""), namespace=memorax_user_namespace)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        counter = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
        context = (build_incremental_context(
            parent, token_counter=counter, budget_tokens=10_000)
            if incremental else build_fresh_r1_context(
                parent, token_counter=counter, budget_tokens=10_000))
        checks["full_parent_included"] = bool(context["full_parent_included"])
        checks["complete_sessions_only"] = bool(context["complete_sessions_only"])
        checks["parent_tail_preserved"] = context["parent_tail_turn_id"] == parent["dialogue"]["turns"][-1]["turn_id"]
        if reuse_existing_namespace:
            parent_intent_path = (source_run_root / "memory" /
                                  ("child_add_intent.json" if incremental else
                                   "parent_add_intent.json"))
            if parent_intent_path.is_file():
                source_intent = json.loads(parent_intent_path.read_text(encoding="utf-8"))
                checks["recovery_namespace_matches"] = (
                    source_intent.get("user_id") == expected_user_id)
            else:
                checks["recovery_namespace_matches"] = False
        if (feedback_required and feedback_record is None and
                profile.name in {"r1_aligned_feedback_v1", "development_gain_v2"} and
                feedback_input is None):
            raise ValueError("r1_feedback_input_required; refusing_historical_r03_fallback")
        if feedback_required and feedback_record is None:
            feedback_record, feedback_dialogue = _load_feedback_context(
                parent, feedback_input=feedback_input)
        if feedback_required:
            checks["feedback_view_contract_available"] = bool(
                isinstance(feedback_record, Mapping) and
                isinstance(feedback_dialogue, Mapping))
    except Exception as error:
        details["parent_or_tokenizer_error"] = f"{type(error).__name__}:{error}"
        checks.update({"authoritative_parent_valid": False, "full_parent_included": False,
                       "complete_sessions_only": False, "parent_tail_preserved": False,
                       "r2_prior_dialogue_valid": False})
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
    planner_repair_budget = 0 if (
        incremental or skip_planner_semantic_gate or single_event_line_mode or
        profile.development_policy) else 1
    semantic_gate_budget = (0 if skip_planner_semantic_gate else
                            (1 if (incremental or single_event_line_mode or profile.development_policy)
                             else 2))
    profile_budget = profile.worst_case_requests()
    profile_batches = int(profile_budget["realizer_normal_batches"])
    request_budget = {
        "isolation_search": (
            0 if reuse_existing_namespace or assume_new_namespace_empty else 1),
        "parent_add": 0 if reuse_existing_namespace else 1,
        "pre_search": 1,
        "event_verifier": 0 if memory_only_fast else 1,
        "planner": 1,
        "planner_repair": planner_repair_budget,
        "planner_outline_semantic_auditor": 1,
        "semantic_continuation_gate": semantic_gate_budget,
        "realizer": profile_batches,
        "realizer_adherence_auditor": (
            int(profile_budget["batch_adherence_audits"]) + 2
            if profile.sequential_batch_promotion else 3),
        "scene_continuation_realizer": 1,
        "scene_replacement_realizer": 1,
        "realizer_adherence_reaudit": 1,
        "child_add": 0 if no_memorax_writes else 1,
        "post_search": 0 if no_memorax_writes else 1,
        "qa": 0,
        "solver": 0,
        "round_2": 1 if incremental else 0,
    }
    if single_event_line_mode:
        request_budget["planner_outline_semantic_auditor"] = 0
        request_budget["scene_replacement_realizer"] = 0
    if profile.development_policy:
        request_budget["realizer_adherence_auditor"] = 1
        request_budget["scene_replacement_realizer"] = 0
    return {
        "preflight_version": RUNNER_VERSION,
        "decision": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "failed_checks": [k for k, v in checks.items() if not v],
        "details": details,
        "input_path": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path) if input_path.is_file() else None,
        "root_id": parent.get("root_id"),
        "round_index": int(round_index),
        "single_event_line_mode": bool(single_event_line_mode),
        "single_event_target_rank": int(single_event_target_rank),
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
        "total_llm_request_budget": sum(
            int(value) for key, value in request_budget.items()
            if key not in {"isolation_search", "parent_add", "pre_search",
                           "child_add", "post_search", "qa", "solver",
                           "round_2"}),
        "runtime_hard_llm_request_limit": int(profile_budget["total_model_requests"]),
        "max_completion_tokens": {"event_verifier": EVENT_VERIFIER_MAX_TOKENS,
                                   "planner": int(profile.planner_max_tokens),
                                   "planner_repair": int(profile.planner_max_tokens),
                                   "planner_outline_semantic_auditor": (
                                       profile.outline_max_tokens),
                                   "semantic_continuation_gate": profile.gate_max_tokens,
                                   "realizer": REALIZER_MAX_TOKENS,
                                   "realizer_adherence_auditor": (
                                       REALIZER_ADHERENCE_AUDITOR_MAX_TOKENS),
                                   "scene_continuation_realizer": (
                                       SCENE_CONTINUATION_MAX_TOKENS),
                                   "scene_replacement_realizer": (
                                       SCENE_REPLACEMENT_MAX_TOKENS),
                                   "realizer_adherence_reaudit": (
                                       REALIZER_ADHERENCE_REAUDIT_MAX_TOKENS)},
        "required_outbound_roles": list(REQUIRED_OUTBOUND_ROLES),
        "request_budget": request_budget,
        "outbound_role_coverage": outbound_role_coverage,
        "context": ({"turn_count": context["selected_turn_count"],
                     "session_count": context["selected_session_count"],
                     "raw_tokens": context["selected_raw_dialogue_tokens"]} if context else None),
        "source_manifest": _manifest(),
        "api_calls": 0,
        "memory_only_fast": bool(memory_only_fast),
        "event_bucket_only": bool(event_bucket_only),
        "no_memorax_writes": bool(no_memorax_writes),
        "skip_planner_semantic_gate": bool(skip_planner_semantic_gate),
        "incremental_round": bool(incremental),
        "feedback_view_version": profile.feedback_view_version,
        "feedback_view_required": feedback_required,
        "execution_profile": profile.to_dict(),
        "development_policy": profile.development_policy,
        "live_execution_ready": not bool(profile.development_policy),
        "live_execution_blocker": ("frozen_memory_and_bounded_http_adapter_not_authorized_or_installed"
                                   if profile.development_policy else None),
    }


class FreezeGuard:
    def __init__(self, run_root: Path, freeze: Mapping[str, Any],
                 input_path: Path, config_path: Path):
        self.run_root, self.freeze = run_root, dict(freeze)
        self.input_path, self.config_path = input_path, config_path
        self.first_api_at = None

    def before_request(self, logical_role: str, role_version: str) -> None:
        allowed = {"event_verifier", "planner", "planner_repair",
                   "planner_outline_semantic_auditor",
                   "semantic_continuation_gate",
                   "round_realizer", "realizer_adherence_auditor",
                   "scene_continuation_realizer",
                   "scene_replacement_realizer",
                   "realizer_adherence_reaudit",
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
                 assume_new_namespace_empty: bool = False,
                 memory_only_fast: bool | None = None,
                 event_bucket_only: bool | None = None,
                 skip_planner_semantic_gate: bool = False,
                 single_event_line_mode: bool = False,
                 single_event_target_rank: int = 1,
                 no_memorax_writes: bool = False,
                 round_index: int = 1,
                 execution_profile: ExpansionExecutionProfile | None = None,
                 feedback_record: Mapping[str, Any] | None = None,
                 feedback_dialogue: Mapping[str, Any] | None = None,
                 feedback_input: Path | None = None) -> dict[str, Any]:
    if report.get("decision") != "pass":
        raise RuntimeError("MEMORY_EVENT_CONTINUATION_PREFLIGHT_FAILED")
    if run_root.exists():
        raise FileExistsError(f"memory_event_continuation_run_exists:{run_root}")
    profile = execution_profile or LEGACY_PROFILE
    if profile.development_policy:
        # This change authorizes offline preparation only. The historical
        # live entry reads remote Memory and has its historical transport
        # policy; it must not masquerade as the prepared frozen comparison.
        raise RuntimeError(
            "development_gain_live_disabled:requires_frozen_memory_and_bounded_http_adapter;"
            "use_development_prepare_for_zero_api_preparation")
    if (profile.name in {"r1_aligned_feedback_v1", "development_gain_v2"} and
            feedback_record is None and feedback_input is None):
        raise ValueError("r1_feedback_input_required; refusing_historical_r03_fallback")
    if profile.feedback_view_version and (
            feedback_record is None or feedback_dialogue is None):
        feedback_parent = (
            load_incremental_round_input(input_path, round_index=round_index)[0]
            if int(round_index) > 1 else
            json.loads(input_path.read_text(encoding="utf-8")))
        feedback_record, feedback_dialogue = _load_feedback_context(
            feedback_parent, feedback_input=feedback_input)
    memory_only_fast, event_bucket_only = _resolve_profile_route(
        profile=profile, explicit_profile=execution_profile is not None,
        memory_only_fast=memory_only_fast, event_bucket_only=event_bucket_only)
    run_root.mkdir(parents=True, exist_ok=False)
    for part in ("event_verifier", "planner", "semantic_gate", "realizer",
                 "realizer_adherence", "promotion", "memory", "reports",
                 "raw/prompts", "attempts"):
        (run_root / part).mkdir(parents=True, exist_ok=True)
    freeze = {**dict(report), "freeze_version":RUNNER_VERSION,
              "frozen_at":_now(), "fresh":True, "no_resume":True,
              "first_api_request":None,
              "feedback_input": ({
                  "path": str(feedback_input.resolve()),
                  "sha256": sha256_file(feedback_input),
              } if feedback_input is not None else None)}
    atomic_json(run_root / "run_freeze_manifest.json", freeze)
    atomic_json(run_root / "run_status.json", {"status":"initialized", "api_calls":0})
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
    counter=lambda text: len(tokenizer.encode(text,add_special_tokens=False))
    ledger=RequestLedger(run_root,manifest_sha256=freeze["source_manifest"]["manifest_sha256"],backend_mode="live")
    guard=FreezeGuard(run_root,freeze,input_path,config_path)
    caller=SimpleAppendLiveCaller(run_root=run_root,ledger=ledger,
        client=OpenAICompatibleChallengerClient(_config(config_path)),token_counter=counter,
        root_id=str(freeze["root_id"]),context_limit=CONTEXT_LIMIT,before_request=guard.before_request,
        total_request_budget=profile.worst_case_requests()["total_model_requests"],
        role_request_budgets={
            **ROLE_REQUEST_BUDGETS,
            # Expanded profiles may dispatch several normal Realizer batches;
            # the role budget must cover that explicit schedule.
            "round_realizer": int(profile.worst_case_requests()[
                "realizer_normal_batches"]),
            "realizer_adherence_auditor": (
                # Batch audits + one whole-round audit + one replacement audit.
                int(profile.worst_case_requests()["batch_adherence_audits"]) + 2
                if profile.sequential_batch_promotion else
                ROLE_REQUEST_BUDGETS["realizer_adherence_auditor"]),
        })
    memorax_client = MemoraxClient(load_memorax_config(
        credentials_path=credentials_path,
        allow_insecure_http=allow_insecure_http))
    if no_memorax_writes:
        class _NoMemoraxWritesGateway:
            """Delegate reads while making any accidental Add fail closed."""

            def __init__(self, backend):
                self.backend = backend

            @property
            def calls(self):
                return self.backend.calls

            def search_buckets(self, payload, *, request_id):
                return self.backend.search_buckets(payload, request_id=request_id)

            def add(self, payload, *, request_id):
                raise RuntimeError("memorax_add_blocked:no_memorax_writes")

            def add_status(self, task_id, *, request_id):
                raise RuntimeError("memorax_add_status_blocked:no_memorax_writes")

        memorax = _NoMemoraxWritesGateway(memorax_client)
    else:
        memorax = memorax_client
    failure_stage=None; runtime_error=None; result={}
    semantic_gate_attempt = 0
    outline_semantic_attempt = 0
    def verifier_call(prompt):
        parsed,_,errors=caller.call_json(
            request_id="event_verifier_attempt_01",
            role="transcript_event_verifier_v1",
            prompt=str(prompt["prompt"]),
            prompt_version=str(prompt["prompt_version"]),
            max_tokens=EVENT_VERIFIER_MAX_TOKENS)
        if errors: raise WorkflowError("event_verifier_transport_or_parse",errors)
        return parsed
    def planner_call(prompt):
        parsed,_,errors=caller.call_json(request_id="planner_attempt_01",role="simple_append_planner_v1",
            prompt=str(prompt["prompt"]),prompt_version=str(prompt["prompt_version"]),
            max_tokens=int(profile.planner_max_tokens))
        if errors: raise WorkflowError("planner_transport_or_parse",errors)
        return parsed
    def planner_repair_call(prompt):
        parsed, _, errors = caller.call_json(
            request_id="planner_candidate_repair_attempt_01",
            role="simple_append_planner_candidate_repair_v1",
            prompt=str(prompt["prompt"]),
            prompt_version=str(prompt["prompt_version"]),
            max_tokens=int(profile.planner_max_tokens))
        if errors:
            raise WorkflowError("planner_candidate_repair_transport_or_parse",
                                errors)
        return parsed
    def semantic_gate_call(prompt):
        nonlocal semantic_gate_attempt
        semantic_gate_attempt += 1
        parsed,_,errors=caller.call_json(
            request_id=f"semantic_continuation_gate_attempt_{semantic_gate_attempt:02d}",
            role="memory_event_realization_aware_semantic_gate_v1",
            prompt=str(prompt["prompt"]),
            prompt_version=str(prompt["prompt_version"]),
            max_tokens=profile.gate_max_tokens)
        if errors:
            raise WorkflowError("semantic_continuation_gate_transport_or_parse",errors)
        return parsed
    def outline_semantic_call(prompt):
        nonlocal outline_semantic_attempt
        outline_semantic_attempt += 1
        prompt_text = str(prompt["prompt"])
        # Long cumulative dialogues can leave less than the nominal auditor
        # budget while still leaving enough room for its compact pair review.
        # Preserve the complete prompt and cap only this response reservation;
        # fail closed when even a 1K JSON review cannot fit.
        prompt_tokens = caller.token_counter(prompt_text)
        available_tokens = CONTEXT_LIMIT - prompt_tokens
        effective_max_tokens = min(
            profile.outline_max_tokens, available_tokens)
        if profile.development_policy and available_tokens < profile.outline_max_tokens:
            raise WorkflowError("planner_outline_semantic_context_overflow",
                                ["development_gain_refuses_silent_output_budget_reduction"])
        if effective_max_tokens < 1_024:
            raise WorkflowError(
                "planner_outline_semantic_context_overflow",
                [f"{prompt_tokens}+{PLANNER_OUTLINE_SEMANTIC_MAX_TOKENS}>"
                 f"{CONTEXT_LIMIT}",
                 f"available_completion_tokens:{available_tokens}"])
        parsed, _, errors = caller.call_json(
            request_id=("planner_outline_semantic_auditor_attempt_"
                        f"{outline_semantic_attempt:02d}"),
            role="planner_outline_semantic_auditor_v1",
            prompt=prompt_text,
            prompt_version=str(prompt["prompt_version"]),
            max_tokens=effective_max_tokens)
        if errors:
            raise WorkflowError("planner_outline_semantic_transport_or_parse",
                                errors)
        return parsed
    def realizer_call(prompt):
        batch_index = int(prompt.get("batch_index") or 1) if isinstance(prompt, Mapping) else 1
        parsed,_,errors=caller.call_json(request_id=f"round_realizer_batch_{batch_index:02d}_attempt_01",role="simple_append_round_realizer_v1",
            prompt=str(prompt["prompt"]),prompt_version=str(prompt["prompt_version"]),max_tokens=REALIZER_MAX_TOKENS)
        if errors: raise WorkflowError("realizer_transport_or_parse",errors)
        return parsed
    def adherence_call(prompt):
        parsed,_,errors=caller.call_json(
            request_id="realizer_adherence_auditor_attempt_01",
            role="memory_event_realizer_adherence_auditor_v1",
            prompt=str(prompt["prompt"]),
            prompt_version=str(prompt["prompt_version"]),
            max_tokens=REALIZER_ADHERENCE_AUDITOR_MAX_TOKENS)
        if errors:
            raise WorkflowError("realizer_adherence_auditor_transport_or_parse",
                                errors)
        return parsed
    def scene_continuation_call(prompt):
        parsed, _, errors = caller.call_json(
            request_id="failed_scene_continuation_realizer_attempt_01",
            role="simple_append_scene_continuation_realizer_v1",
            prompt=str(prompt["prompt"]),
            prompt_version=str(prompt["prompt_version"]),
            max_tokens=SCENE_CONTINUATION_MAX_TOKENS)
        if errors:
            raise WorkflowError(
                "scene_continuation_realizer_transport_or_parse", errors)
        return parsed
    def scene_replacement_call(prompt):
        parsed, _, errors = caller.call_json(
            request_id="failed_scene_replacement_realizer_attempt_01",
            role="simple_append_scene_replacement_realizer_v1",
            prompt=str(prompt["prompt"]),
            prompt_version=str(prompt["prompt_version"]),
            max_tokens=SCENE_REPLACEMENT_MAX_TOKENS)
        if errors:
            raise WorkflowError(
                "scene_replacement_realizer_transport_or_parse", errors)
        return parsed
    def adherence_reaudit_call(prompt):
        parsed, _, errors = caller.call_json(
            request_id="realizer_adherence_reaudit_attempt_01",
            role="memory_event_realizer_adherence_reaudit_v1",
            prompt=str(prompt["prompt"]),
            prompt_version=str(prompt["prompt_version"]),
            max_tokens=REALIZER_ADHERENCE_REAUDIT_MAX_TOKENS)
        if errors:
            raise WorkflowError(
                "realizer_adherence_reaudit_transport_or_parse", errors)
        return parsed
    def replacement_adherence_call(prompt):
        parsed, _, errors = caller.call_json(
            request_id="scene_replacement_adherence_auditor_attempt_01",
            role="memory_event_realizer_adherence_auditor_v1",
            prompt=str(prompt["prompt"]),
            prompt_version=str(prompt["prompt_version"]),
            max_tokens=REALIZER_ADHERENCE_AUDITOR_MAX_TOKENS)
        if errors:
            raise WorkflowError(
                "scene_replacement_adherence_transport_or_parse", errors)
        return parsed
    try:
        prior_expanded = None
        if int(round_index) > 1:
            parent, prior_expanded, _ = load_incremental_round_input(
                input_path, round_index=round_index)
        else:
            parent=json.loads(input_path.read_text(encoding="utf-8"))
        result=execute_round(parent=parent,memorax=memorax,verifier_call=verifier_call,
            planner_call=planner_call,
            planner_repair_call=(planner_repair_call if round_index == 1 else None),
            semantic_gate_call=semantic_gate_call,realizer_call=realizer_call,
            adherence_call=adherence_call,
            scene_continuation_call=scene_continuation_call,
            scene_replacement_call=scene_replacement_call,
            adherence_reaudit_call=adherence_reaudit_call,
            replacement_adherence_call=replacement_adherence_call,
            outline_semantic_call=outline_semantic_call,
            artifact_sink=_sink(run_root),
            before_memory_request=lambda stage, request_id: guard.before_request(stage,request_id),
            token_counter=counter,wait_for_memory_completion=True,
            reuse_existing_namespace=reuse_existing_namespace,
            memorax_user_namespace=memorax_user_namespace,
            assume_new_namespace_empty=assume_new_namespace_empty,
            memory_only_fast=memory_only_fast,
            event_bucket_only=event_bucket_only,
            round_index=round_index,
            prior_expanded=prior_expanded,
            expose_complete_memory_to_planner=int(round_index) > 1,
            skip_planner_semantic_gate=skip_planner_semantic_gate,
            realizer_style_tail_only=int(round_index) > 1,
            single_event_line_mode=single_event_line_mode,
            single_event_target_rank=single_event_target_rank,
            no_memorax_writes=no_memorax_writes,
            execution_profile=execution_profile,
            feedback_record=feedback_record,
            feedback_dialogue=feedback_dialogue,
            feedback_view_version=profile.feedback_view_version)
        atomic_text(run_root/"expanded_dialogue.md", _expanded_markdown(result["expanded_dialogue"]))
        if result.get("promotable_expanded_dialogue"):
            atomic_text(run_root/"promotable_expanded_dialogue.md",
                        _expanded_markdown(result["promotable_expanded_dialogue"]))
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
        f"-r{int(round_index):02d}-" in str(row.get("request_id"))
        for row in add_records)
    def artifact(name):
        path=run_root/name
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
    planner_raw_output=result.get("planner_raw_output") or artifact(Path("planner/output.json"))
    planner_repair_raw_output=result.get("planner_repair_raw_output") or artifact(
        Path("planner/repair_output.json"))
    planner_repair_validation=result.get("planner_repair_validation") or artifact(
        Path("planner/repair_validation.json"))
    planner_repair_merged_validation=result.get(
        "planner_repair_merged_validation") or artifact(
            Path("planner/repair_merged_validation.json"))
    verifier_output = (result.get("event_verifier_raw_output") or artifact(
        Path("event_verifier/output.json")))
    verifier_validation = (result.get("event_verifier_validation") or artifact(
        Path("event_verifier/validation.json")))
    verified_tracks_artifact = artifact(
        Path("event_verifier/materialized_tracks.json")) or {}
    verified_tracks = (result.get("verified_tracks") or
                       verified_tracks_artifact.get("verified_tracks") or [])
    planner_output=result.get("planner_output") or artifact(Path("planner/materialized_plan.json"))
    planner_output=planner_output or planner_raw_output
    planner_validation=result.get("planner_validation") or artifact(Path("planner/validation.json"))
    scene_quota_audit = (result.get("scene_quota_audit") or artifact(
        Path("planner/scene_quota_audit.json")) or {})
    candidate_artifact = artifact(Path("memory/event_candidates.json")) or {}
    candidate_rows = (result.get("candidates") or
                      candidate_artifact.get("candidates") or [])
    planner_outline_audit = result.get("planner_outline_audit") or artifact(
        Path("planner/outline_duplicate_audit.json"))
    gate_output=result.get("semantic_gate_output") or artifact(Path("semantic_gate/output.json"))
    gate_context=result.get("semantic_gate_context") or artifact(Path("semantic_gate/context.json"))
    gate_validation=result.get("semantic_gate_validation") or artifact(Path("semantic_gate/validation.json"))
    candidate_selection=result.get("candidate_selection_audit") or artifact(
        Path("semantic_gate/candidate_selection.json"))
    adherence_audit = result.get("realizer_adherence_audit") or artifact(
        Path("realizer_adherence/audit.json"))
    adherence_validation = result.get("realizer_adherence_validation") or artifact(
        Path("realizer_adherence/validation.json"))
    initial_adherence_audit = result.get("initial_adherence_audit") or artifact(
        Path("realizer_adherence/initial_audit.json"))
    continuation_request = result.get("scene_continuation_request") or artifact(
        Path("scene_continuation/request.json"))
    continuation_materialization = result.get(
        "scene_continuation_materialization") or artifact(
            Path("scene_continuation/validation.json"))
    continuation_raw_output = result.get(
        "scene_continuation_raw_output") or artifact(
            Path("scene_continuation/output.json"))
    adherence_reaudit_result = result.get(
        "adherence_reaudit_result") or artifact(
            Path("realizer_adherence/reaudit_validation.json"))
    replacement_request = result.get("scene_replacement_request") or artifact(
        Path("scene_replacement/request.json"))
    replacement_materialization = result.get(
        "scene_replacement_materialization") or artifact(
            Path("scene_replacement/validation.json"))
    replacement_adherence_result = result.get(
        "scene_replacement_adherence_result") or artifact(
            Path("scene_replacement/adherence_validation.json"))
    promotion_audit = result.get("promotion_audit") or artifact(
        Path("promotion/audit.json"))
    promotion_validation = result.get("promotion_validation") or artifact(
        Path("promotion/validation.json"))
    promotable_child = result.get("promotable_child_dialogue") or artifact(
        Path("promotion/promotable_child.json"))
    promotable_expanded = result.get("promotable_expanded_dialogue") or artifact(
        Path("promotion/promotable_expanded_dialogue.json"))
    parent_visibility_audit = result.get("parent_visibility_audit") or artifact(
        Path("memory/parent_visibility_audit.json"))
    child_visibility_audit = result.get("child_visibility_audit") or artifact(
        Path("memory/child_visibility_audit.json"))
    gate_expected_refs = list((gate_context or {}).get(
        "expected_candidate_refs") or [])
    gate_returned_refs = [str(row.get("candidate_ref") or "") for row in
                          (gate_output or {}).get("candidate_reviews") or []
                          if isinstance(row, Mapping)]
    development_mode_by_ref = {
        str(candidate.get("candidate_ref") or ""): str(
            scene.get("development_mode") or "")
        for scene in (gate_context or {}).get("scenes") or []
        if isinstance(scene, Mapping)
        for candidate in scene.get("candidates") or []
        if isinstance(candidate, Mapping)
    }
    normalized_gate_reviews = list((gate_validation or {}).get(
        "normalized_candidate_reviews") or [])
    gate_semantic_audit = [{
        "candidate_ref": str(row.get("candidate_ref") or ""),
        "development_mode": development_mode_by_ref.get(
            str(row.get("candidate_ref") or "")),
        "parent_relation": str(row.get("parent_relation") or ""),
        "decision": str(row.get("decision") or ""),
        "parent_evidence": list(row.get("evidence_turn_refs") or []),
        "reason": str(row.get("reason") or ""),
    } for row in normalized_gate_reviews if isinstance(row, Mapping)]
    gate_relation_counts = {
        relation: sum(row["parent_relation"] == relation
                      for row in gate_semantic_audit)
        for relation in ("REALIZED", "PLANNED_OR_INTENDED",
                         "RELATED_ONLY", "NONE")
    }
    planner_summary_output = ((result.get("planner_variant_selection") or {}).get(
        "value") or planner_raw_output or {})
    raw_continuations = ((planner_summary_output or {}).get(
        "selected_continuations") or [])
    raw_new_candidates = ((planner_summary_output or {}).get(
        "new_development_candidates") or [])
    selected_new_indices = [
        int(index) for index in
        scene_quota_audit.get("selected_new_candidate_indices") or []
        if isinstance(index, int) or str(index).isdigit()
    ]
    selected_new_rows = [
        raw_new_candidates[index - 1] for index in selected_new_indices
        if 1 <= index <= len(raw_new_candidates)
    ]
    summary={"summary_version":RUNNER_VERSION,"status":"completed" if result.get("generation_ready") else "failed",
        "failure_stage":failure_stage,"runtime_error":runtime_error,"run_root":str(run_root),
        "root_id":freeze.get("root_id"),"user_id":freeze.get("memory_user_id"),
        "event_verifier_calls":usage["roles"]["event_verifier"]["requests"],
        "planner_calls":usage["roles"]["planner"]["requests"],
        "planner_repair_calls":usage["roles"]["planner_repair"]["requests"],
        "planner_max_plan_variants": int(profile.max_plan_variants),
        "planner_variant_selection": result.get("planner_variant_selection"),
        "planner_outline_semantic_auditor_calls": usage["roles"][
            "planner_outline_semantic_auditor"]["requests"],
        "semantic_gate_calls":usage["roles"]["semantic_continuation_gate"]["requests"],
        "realizer_calls":usage["roles"]["round_realizer"]["requests"],
        "scene_continuation_realizer_calls":usage["roles"][
            "scene_continuation_realizer"]["requests"],
        "scene_replacement_realizer_calls":usage["roles"][
            "scene_replacement_realizer"]["requests"],
        "realizer_adherence_auditor_calls":usage["roles"][
            "realizer_adherence_auditor"]["requests"],
        "realizer_adherence_reaudit_calls":usage["roles"][
            "realizer_adherence_reaudit"]["requests"],
        "event_verifier_retries":0,"planner_retries":0,
        "planner_repair_retries":0,
        "planner_outline_semantic_auditor_retries":0,
        "semantic_gate_retries":0,"realizer_retries":0,
        "realizer_adherence_auditor_retries":0,
        "scene_continuation_realizer_retries":0,
        "scene_replacement_realizer_retries":0,
        "realizer_adherence_reaudit_retries":0,
        "llm_usage":usage,
        "memorax_calls":len(memory_records),"memorax_call_records":memory_records,
        "parent_add_calls":int(parent_add_calls),
        "child_add_calls": int(child_add_calls),
        "parent_add_skipped":bool(result.get("parent_add_skipped", reuse_existing_namespace)),
        "existing_namespace_reused":bool(reuse_existing_namespace),
        "memory_event_candidates":candidate_rows,
        "event_verifier_output":verifier_output,
        "event_verifier_validation":verifier_validation,
        "verified_tracks":verified_tracks,
        "planner_validation":planner_validation,
        "planner_outline_audit": planner_outline_audit,
        "planner_outline_semantic_audit": (
            result.get("planner_outline_semantic_audit") or artifact(
                Path("planner/outline_semantic_validation.json"))),
        "planner_raw_output":planner_raw_output,
        "planner_repair_raw_output":planner_repair_raw_output,
        "planner_repair_validation":planner_repair_validation,
        "planner_repair_merged_validation":planner_repair_merged_validation,
        "planner_materialized_plan":planner_output if planner_output is not planner_raw_output else None,
        "planner_scenes":planner_output.get("scenes",[]) if isinstance(planner_output,Mapping) else [],
        "semantic_gate_output":gate_output,
        "semantic_gate_validation":gate_validation,
        "planner_candidate_refs": gate_expected_refs,
        "gate_expected_refs": gate_expected_refs,
        "gate_returned_refs": gate_returned_refs,
        "gate_coverage_count": f"{len(set(gate_returned_refs) & set(gate_expected_refs))}/{len(gate_expected_refs)}",
        "gate_coverage_complete": (
            len(gate_returned_refs) == len(gate_expected_refs) and
            set(gate_returned_refs) == set(gate_expected_refs)),
        "gate_semantic_audit": gate_semantic_audit,
        "gate_parent_relation_counts": gate_relation_counts,
        "gate_pass_count": sum(row["decision"] == "pass"
                               for row in gate_semantic_audit),
        "gate_reject_count": sum(row["decision"] == "reject"
                                 for row in gate_semantic_audit),
        "existing_event_candidate_count": len(candidate_rows),
        "selected_continuation_count": len(raw_continuations),
        "selected_continuation_keys": [str(row.get("memory_event_ref") or "")
                                       for row in raw_continuations],
        "raw_new_candidate_count": len(raw_new_candidates),
        "scene_quota": scene_quota_audit.get("scene_quota", 3),
        "remaining_new_quota": scene_quota_audit.get(
            "remaining_new_quota"),
        "materialized_new_development_count": scene_quota_audit.get(
            "materialized_new_development_count"),
        "unused_new_candidate_count": scene_quota_audit.get(
            "unused_new_candidate_count"),
        "selected_new_candidate_indices": selected_new_indices,
        "skipped_new_candidate_indices": scene_quota_audit.get(
            "skipped_new_candidate_indices") or [],
        "final_materialized_scene_count": scene_quota_audit.get(
            "final_materialized_scene_count"),
        "scene_quota_audit": scene_quota_audit,
        "new_people_count": sum(len(row.get("new_people") or [])
                                for row in selected_new_rows),
        "new_people_names": [
            str(person.get("name") or "")
            for row in selected_new_rows
            for person in row.get("new_people") or []
            if isinstance(person, Mapping)],
        "new_development_people_audit": [{
            "new_people_count": len(row.get("new_people") or []),
            "new_people": list(row.get("new_people") or []),
            "participants": list(row.get("participants") or []),
            "undeclared_participant_count": len([
                participant for participant in row.get("participants") or []
                if str(participant) not in {
                    *(str(turn.get("speaker") or "")
                      for turn in parent.get("dialogue", {}).get("turns", [])),
                    *(str(person.get("name") or "")
                      for person in row.get("new_people") or []
                      if isinstance(person, Mapping)),
                }
            ]),
        } for row in selected_new_rows],
        "candidate_selection_audit":candidate_selection,
        "realizer_validation":result.get("realizer_validation"),
        "realizer_adherence_audit": adherence_audit,
        "realizer_adherence_validation": adherence_validation,
        "initial_adherence_audit": initial_adherence_audit,
        "scene_continuation_request": continuation_request,
        "scene_continuation_raw_output": continuation_raw_output,
        "scene_continuation_materialization": continuation_materialization,
        "scene_continuation_scene_count": int(
            (continuation_request or {}).get("eligible_scene_count") or 0),
        "adherence_reaudit_result": adherence_reaudit_result,
        "scene_replacement_request": replacement_request,
        "scene_replacement_materialization": replacement_materialization,
        "scene_replacement_adherence_result": replacement_adherence_result,
        "scene_replacement_selected_candidate_ref": (
            ((replacement_request or {}).get("selected_replacement") or {})
            .get("candidate_ref")),
        "scene_replacement_used": bool(
            (replacement_materialization or {}).get("replacements")),
        "promotion_audit": promotion_audit,
        "promotion_validation": promotion_validation,
        "promoted_scene_refs": (promotion_audit or {}).get(
            "promoted_scene_refs", []),
        "rejected_scene_refs": (promotion_audit or {}).get(
            "rejected_scene_refs", []),
        "promoted_scene_count": (promotion_audit or {}).get(
            "promoted_scene_count", 0),
        "full_round_ready": bool((promotion_audit or {}).get(
            "full_round_ready", result.get("full_round_ready", False))),
        "partial_promotion": bool((promotion_audit or {}).get(
            "partial_promotion", result.get("partial_promotion", False))),
        "promotable_child_dialogue": promotable_child,
        "promotable_expanded_dialogue": promotable_expanded,
        "realizer_input_isolation":result.get("realizer_input_isolation") or artifact(
            Path("realizer/input_validation.json")),
        "parent_add_verified":bool(result.get("parent_add_completion")),
        "child_add_verified":bool(
            result.get("child_add_completion") and not no_memorax_writes),
        "parent_visibility_audit": parent_visibility_audit,
        "child_visibility_audit": child_visibility_audit,
        "memory_write_verified":result.get("memory_write_verified",False),"memory_visibility_verified":result.get("memory_visibility_verified",False),
        "generation_ready":bool(result.get("generation_ready")),"qa_pipeline_ready":False,
        "expansion_calls":usage["total_requests"],
        "verifier_planner_gate_realizer_adherence_only_llm": False,
        "bounded_failed_scene_recovery_enabled": True,
        "qa_calls":0,"solver_calls":0,
        "round_2":bool(int(round_index) == 2),"r2":bool(int(round_index) == 2),
        "r3":bool(int(round_index) >= 3),"training":False,"cache_reuse":0,"imported_response_reuse":0,
        "event_bucket_only":bool(event_bucket_only),
        "no_memorax_writes": bool(no_memorax_writes),
        "skip_planner_semantic_gate": bool(skip_planner_semantic_gate),
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
      f"- Root: `{s.get('root_id')}`",
      f"- Event verifier calls: `{s['event_verifier_calls']}`",
      f"- Verified tracks: `{len(s.get('verified_tracks') or [])}`",
      f"- Planner calls: `{s['planner_calls']}`",
      f"- Planner repair calls: `{s.get('planner_repair_calls', 0)}`",
      f"- Planner outline semantic auditor calls: `{s.get('planner_outline_semantic_auditor_calls', 0)}`",
      f"- Semantic gate calls: `{s['semantic_gate_calls']}`",
      f"- Planner pre-Gate skipped: `{s.get('skip_planner_semantic_gate', False)}`",
      f"- Existing event candidates: `{s.get('existing_event_candidate_count')}`",
      f"- Selected continuations: `{s.get('selected_continuation_count')}`",
      f"- Raw new candidates: `{s.get('raw_new_candidate_count')}`",
      f"- Scene quota / remaining new quota: `{s.get('scene_quota')}/{s.get('remaining_new_quota')}`",
      f"- Materialized / unused new candidates: `{s.get('materialized_new_development_count')}/{s.get('unused_new_candidate_count')}`",
      f"- Final materialized scenes: `{s.get('final_materialized_scene_count')}`",
      f"- New people: `{s.get('new_people_count')}` {s.get('new_people_names')}",
      f"- Semantic gate decision: `{(s.get('semantic_gate_validation') or {}).get('overall_decision')}`",
      f"- Gate coverage: `{s.get('gate_coverage_count')}` (complete: `{s.get('gate_coverage_complete')}`)",
      f"- Gate parent relations: `{s.get('gate_parent_relation_counts')}`",
      f"- Gate pass/reject: `{s.get('gate_pass_count')}/{s.get('gate_reject_count')}`",
      f"- Realizer calls: `{s['realizer_calls']}`",
      f"- Failed-scene continuation calls: `{s.get('scene_continuation_realizer_calls', 0)}`",
      f"- Replacement realizer calls: `{s.get('scene_replacement_realizer_calls', 0)}`",
      f"- Adherence re-audit calls: `{s.get('realizer_adherence_reaudit_calls', 0)}`",
      f"- Replacement used: `{s.get('scene_replacement_used', False)}` ({s.get('scene_replacement_selected_candidate_ref')})",
      f"- Continuation-eligible scenes: `{s.get('scene_continuation_scene_count', 0)}`",
      f"- Memorax calls: `{s['memorax_calls']}`",
      f"- Promoted scenes: `{s.get('promoted_scene_count', 0)}` {s.get('promoted_scene_refs', [])}",
      f"- Rejected scenes: `{s.get('rejected_scene_refs', [])}`",
      f"- Full round ready: `{s.get('full_round_ready', False)}`",
      f"- Partial promotion: `{s.get('partial_promotion', False)}`",
      f"- Memory write verified: `{s['memory_write_verified']}`",
      f"- Memory visibility verified: `{s['memory_visibility_verified']}`",
      f"- Parent bucket delta visible: `{'yes' if (s.get('parent_visibility_audit') or {}).get('bucket_visibility_verified') else 'no/unknown'}`",
      f"- Child bucket delta visible: `{'yes' if (s.get('child_visibility_audit') or {}).get('bucket_visibility_verified') else 'no/unknown'}`",
      f"- Child changed buckets: `{len((s.get('child_visibility_audit') or {}).get('changed_bucket_refs') or [])}`",
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
    parser.add_argument("--memory-only-fast", action=argparse.BooleanOptionalAction,
                        default=None)
    parser.add_argument("--event-bucket-only", action=argparse.BooleanOptionalAction,
                        default=None)
    parser.add_argument("--skip-planner-semantic-gate", action="store_true",
                        help="Skip only the Planner pre-Gate LLM call; keep the Gate code and post-Realizer audit.")
    parser.add_argument("--single-event-line", action="store_true",
                        help="Expand one ranked Memory event through three ordered dependent scenes.")
    parser.add_argument("--single-event-target-rank", type=int, default=1,
                        help="1-based deterministic Memory track rank used by --single-event-line.")
    parser.add_argument("--no-memorax-writes", action="store_true",
                        help="Generate and audit locally without submitting the child Add.")
    parser.add_argument("--round-index", type=int, default=1)
    parser.add_argument(
        "--feedback-input", type=Path, default=None,
        help="Root-specific frozen feedback record for r1_aligned_feedback_v1 or development_gain_v2.",
    )
    parser.add_argument(
        "--profile", choices=(
            "legacy_v1", "expanded_multi_plan_v1", "expanded_single_plan_v1",
            "r1_aligned_baseline_v1", "r1_aligned_feedback_v1", "development_gain_v2"),
        default=None,
        help="Select a versioned expansion profile explicitly.",
    )
    parser.add_argument("--preflight-only",action="store_true")
    parser.add_argument("--live",action="store_true")
    args=parser.parse_args(argv)
    selected_profile = None
    if args.profile == "legacy_v1":
        selected_profile = LEGACY_PROFILE
    elif args.profile == "expanded_multi_plan_v1":
        from .profiles import EXPANDED_PROFILE
        selected_profile = EXPANDED_PROFILE
    elif args.profile == "expanded_single_plan_v1":
        from .profiles import EXPANDED_SINGLE_PLAN_PROFILE
        selected_profile = EXPANDED_SINGLE_PLAN_PROFILE
    elif args.profile == "r1_aligned_baseline_v1":
        selected_profile = R1_ALIGNED_BASELINE_PROFILE
    elif args.profile == "r1_aligned_feedback_v1":
        selected_profile = R1_ALIGNED_FEEDBACK_PROFILE
    elif args.profile == "development_gain_v2":
        from .profiles import DEVELOPMENT_GAIN_PROFILE
        selected_profile = DEVELOPMENT_GAIN_PROFILE
    report=preflight(input_path=args.input,run_root=args.run_root,config_path=args.config,
        tokenizer_path=args.tokenizer,credentials_path=args.memorax_credentials,
        allow_insecure_http=args.allow_insecure_memorax_http,
        reuse_existing_namespace=args.reuse_existing_namespace,
        source_run_root=args.source_run_root,
        memorax_user_namespace=args.memorax_user_namespace,
        assume_new_namespace_empty=args.assume_new_namespace_empty,
        memory_only_fast=args.memory_only_fast,
        event_bucket_only=args.event_bucket_only,
        skip_planner_semantic_gate=args.skip_planner_semantic_gate,
        single_event_line_mode=args.single_event_line,
        single_event_target_rank=args.single_event_target_rank,
        no_memorax_writes=args.no_memorax_writes,
        round_index=args.round_index,
        execution_profile=selected_profile,
        feedback_input=args.feedback_input)
    print(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True))
    if args.preflight_only or not args.live:
        return 0 if report["decision"]=="pass" else 2
    summary=execute_live(report=report,input_path=args.input,run_root=args.run_root,
        config_path=args.config,tokenizer_path=args.tokenizer,
        credentials_path=args.memorax_credentials,
        allow_insecure_http=args.allow_insecure_memorax_http,
        reuse_existing_namespace=args.reuse_existing_namespace,
        memorax_user_namespace=args.memorax_user_namespace,
        assume_new_namespace_empty=args.assume_new_namespace_empty,
        memory_only_fast=args.memory_only_fast,
        event_bucket_only=args.event_bucket_only,
        skip_planner_semantic_gate=args.skip_planner_semantic_gate,
        single_event_line_mode=args.single_event_line,
        single_event_target_rank=args.single_event_target_rank,
        no_memorax_writes=args.no_memorax_writes,
        round_index=args.round_index,
        execution_profile=selected_profile,
        feedback_input=args.feedback_input)
    print(json.dumps(summary,ensure_ascii=False,indent=2,sort_keys=True))
    return 0 if summary["status"]=="completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
