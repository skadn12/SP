"""Run one stateful outline -> dialogue -> Questioner -> Solver canary."""

from __future__ import annotations

import argparse
import collections
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from transformers import AutoTokenizer

from .core import atomic_json, sha256_bytes, sha256_file
from .ledger import RequestLedger
from .lineage_three_stage_canary import ApiBudget, run_single_object_task
from .live_backends import (
    LiveQuestionerBackend, LiveRealizerBackend, LiveSemanticRepairBackend,
    OpenAICompatibleChallengerClient, load_challenger_config,
)
from .minimal_admission import SPAN_IDS_ONLY_SCHEMA_PROFILE, SPELL_STYLE_MINIMAL_MODE
from .prompts import (
    QUESTIONER_FORMAT_RETRY_VERSION, QUESTIONER_SPAN_IDS_ONLY_PROMPT_VERSION,
    QUESTIONER_SPAN_IDS_ONLY_SCHEMA_VERSION,
    build_questioner_span_ids_only_feedback_prompt,
    questioner_format_retry_suffix_hash, span_ids_only_questioner_template_hash,
)
from .questioner_contract import (
    build_span_catalog, default_reasoning_contracts,
    reasoning_contract_sha256, span_catalog_sha256,
)
from .run_canary import (
    create_source_snapshot, git_value, project_diff_sha256,
    write_post_run_provenance,
)
from .run_cross_dialogue_calibration import (
    _base_manifest, _questioner_config, _run_questioner, _run_solver,
    _solver_config,
)
from .run_rolling_outline_30k_canary import _source_attestation
from .stateful_graph_expansion import (
    STATEFUL_COMPLETION_CONTRACT_VERSION,
    STATEFUL_GRAPH_EXPANSION_VERSION, STATEFUL_GRAPH_LEDGER_VERSION,
    STATEFUL_GRAPH_REALIZER_PROMPT_VERSION,
    STATEFUL_GRAPH_SCENE_CONTRACT_VERSION,
    STANDARD_STAGE_SCHEDULE,
    build_stateful_graph_realizer_prompt, canonical_sha256,
    compile_stateful_scene_contracts,
)
from .stateful_graph_orchestrator import StatefulGraphRoundOrchestrator
from .stateful_outline_planner import (
    RAW_STATEFUL_OUTLINE_SCHEMA, STATEFUL_OUTLINE_CHUNKED_VERSION,
    STATEFUL_OUTLINE_SCHEMA_VERSION, build_stateful_outline_arc_prompt,
    build_stateful_outline_overview_prompt, normalize_stateful_outline,
    validate_raw_stateful_outline, validate_stateful_outline_arc,
    validate_stateful_outline_overview,
)
from .vllm_backends import VLLMSolverBackend


PROJECT = Path(__file__).resolve().parents[3]
MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conv-41_stateful_expansion_qa_solver_canary_v1"
)
SOURCE_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conv-41_spell_full_graph_expansion_v7"
)
FEEDBACK_PATH = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "spell_cross_dialogue_calibration_5x3_v1/frozen/"
    "calibration_feedback_memory.json"
)
TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")
SOLVER = Path(
    "/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/"
    "accepted_models/solver_base_restart_stage2_merged"
)
CHALLENGER_MODEL = "Qwen/Qwen3-32B"
CHALLENGER_CONTEXT_LIMIT = 40_960
SOLVER_MAX_MODEL_LEN = 16_384
OUTLINE_OVERVIEW_MAX_TOKENS = 2_500
OUTLINE_ARC_MAX_TOKENS = 3_200


def _arc_scene_indices(arc_index: int) -> tuple[int, int, int, int]:
    assigned: list[int | None] = [None, None, None, None]
    for scene_index, schedule in enumerate(STANDARD_STAGE_SCHEDULE, 1):
        for scheduled_arc, stage_index in schedule:
            if scheduled_arc == arc_index - 1:
                assigned[stage_index] = scene_index
    if any(value is None for value in assigned):
        raise RuntimeError("stateful_arc_schedule_incomplete")
    return tuple(int(value) for value in assigned)


def _tokens(tokenizer: Any, turns: Sequence[Mapping[str, Any]]) -> int:
    text = "\n".join(
        f"[{turn.get('turn_id', '')}] {turn['speaker']}: {turn['text']}"
        for turn in turns)
    return len(tokenizer.encode(text, add_special_tokens=False))


def _with_session_ids(turns: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for position, source in enumerate(turns, 1):
        turn = dict(source)
        turn_id = str(turn.get("turn_id") or f"P{position:04d}")
        turn["turn_id"] = turn_id
        turn["session_id"] = str(turn.get("session_id") or
                                 (turn_id.split(":", 1)[0]
                                  if ":" in turn_id else "parent"))
        result.append(turn)
    return result


def _view(turns: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    dialogue = _with_session_ids(turns)
    catalog = build_span_catalog(dialogue)
    return {"combined_dialogue": dialogue, "span_catalog": catalog,
            "view_sha256": canonical_sha256(
                {"dialogue": dialogue, "span_catalog": catalog})}


def _usage(root: Path) -> dict[str, Any]:
    paid = collections.Counter()
    local = collections.Counter()
    tokens = collections.Counter()
    for path in root.glob("**/ledger/requests/*.json"):
        row = json.loads(path.read_text())
        if row.get("status") != "completed":
            continue
        intent = str(row.get("intent") or "unknown")
        if row.get("backend_mode") == "live":
            paid[intent] += 1
            usage = (row.get("response_metadata") or {}).get("usage") or {}
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                tokens[key] += int(usage.get(key) or 0)
        elif row.get("backend_mode") == "local":
            local[intent] += 1
    return {"paid_api_by_intent": dict(sorted(paid.items())),
            "total_paid_api_requests": sum(paid.values()),
            "prompt_tokens": tokens["prompt_tokens"],
            "completion_tokens": tokens["completion_tokens"],
            "total_tokens": tokens["total_tokens"],
            "local_vllm_by_intent": dict(sorted(local.items())),
            "local_vllm_generations": sum(local.values())}


def _questioner_artifacts(
    *, root: Path, compact_turns: Sequence[Mapping[str, Any]], tokenizer: Any,
    api: Any, feedback: Mapping[str, Any],
) -> dict[str, Any]:
    context = {"schema_version": "stateful_round_compact_questioner_view_v1",
               "lineage": "conv-41", "combined_dialogue": [
                   dict(turn) for turn in compact_turns]}
    catalog = build_span_catalog(context["combined_dialogue"])
    letters = ("A", "B", "C")
    contracts = default_reasoning_contracts(letters)
    prompt = build_questioner_span_ids_only_feedback_prompt(
        frozen_dialogue=context["combined_dialogue"],
        valid_turn_ids=[turn["turn_id"] for turn in
                        context["combined_dialogue"]],
        valid_session_ids=sorted({turn["session_id"] for turn in
                                  context["combined_dialogue"]}),
        assigned_gold_letters=letters, solver_feedback=feedback["feedback"],
        batch_index=0, span_catalog=catalog)
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    preflight = {"rendered_prompt_tokens": prompt_tokens,
                 "max_generation_tokens": api.questioner_max_tokens,
                 "total_tokens": prompt_tokens + api.questioner_max_tokens,
                 "backend_context_limit": CHALLENGER_CONTEXT_LIMIT,
                 "decision": "pass" if prompt_tokens +
                 api.questioner_max_tokens <= CHALLENGER_CONTEXT_LIMIT else
                 "reject_context_overflow", "truncated": False}
    atomic_json(root / "frozen/questioner_smoke_context.json", context)
    atomic_json(root / "frozen/questioner_span_catalog.json", catalog)
    atomic_json(root / "frozen/questioner_reasoning_contracts.json", contracts)
    atomic_json(root / "frozen/questioner_prompt_preflight.json", preflight)
    return {"context": context, "catalog": catalog, "contracts": contracts,
            "letters": letters, "prompt": prompt, "preflight": preflight}


def _manifest_base(*, root: Path, snapshot: Mapping[str, Any],
                   project_diff: str, mode: str) -> dict[str, Any]:
    return {**_base_manifest(run_id=root.name, mode=mode,
            snapshot=dict(snapshot), project_diff=project_diff),
            "stateful_graph_source_sha256": sha256_file(
                MODULE_ROOT / "stateful_graph_expansion.py"),
            "stateful_orchestrator_source_sha256": sha256_file(
                MODULE_ROOT / "stateful_graph_orchestrator.py"),
            "stateful_outline_planner_source_sha256": sha256_file(
                MODULE_ROOT / "stateful_outline_planner.py"),
            "stateful_runner_source_sha256": sha256_file(Path(__file__))}


def run(*, run_root: Path, api_config: Path) -> dict[str, Any]:
    if run_root.exists():
        raise FileExistsError(f"run_root_exists:{run_root}")
    source = _source_attestation(SOURCE_ROOT)
    parent = _with_session_ids(source["dialogue"]["parent_turns"])
    excluded_child_count = len(source["dialogue"]["child_turns"])
    if len(parent) != 237 or excluded_child_count != 134:
        raise RuntimeError("authoritative_parent_partition_mismatch")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    api = replace(load_challenger_config(api_config), model=CHALLENGER_MODEL,
                  temperature=0.25, enable_thinking=False,
                  timeout_seconds=600, questioner_max_tokens=4096,
                  realizer_max_tokens=4096)
    if not SOLVER.is_dir() or not TOKENIZER.is_dir():
        raise RuntimeError("solver_or_tokenizer_missing")
    solver_native = json.loads((SOLVER / "config.json").read_text())
    if int(solver_native.get("max_position_embeddings") or 0) < 40_960:
        raise RuntimeError("solver_native_context_too_short")
    feedback = json.loads(FEEDBACK_PATH.read_text())
    outline_overview_prompt = build_stateful_outline_overview_prompt(
        parent_turns=parent)
    outline_prompt_tokens = len(tokenizer.encode(
        outline_overview_prompt, add_special_tokens=False))
    if (outline_prompt_tokens + OUTLINE_OVERVIEW_MAX_TOKENS >
            CHALLENGER_CONTEXT_LIMIT):
        raise RuntimeError("stateful_outline_prompt_context_overflow")

    run_root.mkdir(parents=True)
    snapshot = create_source_snapshot(PROJECT, run_root, False)
    project_diff = project_diff_sha256(PROJECT)
    top_manifest = {
        **_manifest_base(root=run_root, snapshot=snapshot,
                         project_diff=project_diff,
                         mode="stateful_expansion_qa_solver_canary_v1"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_attestation": {key: value for key, value in source.items()
                               if key != "dialogue"},
        "generation_parent_turn_count": len(parent),
        "excluded_historical_child_turn_count": excluded_child_count,
        "generation_parent_sha256": canonical_sha256(parent),
        "feedback_path": str(FEEDBACK_PATH),
        "feedback_sha256": sha256_file(FEEDBACK_PATH),
        "challenger_model_identifier": CHALLENGER_MODEL,
        "solver_model_identifier": str(SOLVER),
        "tokenizer_path": str(TOKENIZER),
        "solver_max_model_len": SOLVER_MAX_MODEL_LEN,
        "budgets": {"outline_overview_normal": 1,
                    "outline_arc_normal": 4,
                    "outline_format_retry_global": 5,
                    "realizer_physical_max": 30,
                    "questioner_normal": 1, "questioner_format_retry": 1,
                    "solver_local_max": 192,
                    "verifier": 0, "training": 0},
    }
    atomic_json(run_root / "config/run_manifest.json", top_manifest)
    atomic_json(run_root / "frozen/authoritative_parent_dialogue.json", {
        "schema_version": "authoritative_parent_only_v1", "turns": parent,
        "parent_turn_count": len(parent), "historical_child_included": False})
    atomic_json(run_root / "frozen/feedback_memory.json", feedback)
    atomic_json(run_root / "reports/preflight.json", {
        "parent_turn_count": len(parent),
        "excluded_historical_child_turn_count": excluded_child_count,
        "parent_tokens": _tokens(tokenizer, parent),
        "outline_overview_prompt_tokens": outline_prompt_tokens,
        "outline_overview_reserved_completion_tokens":
            OUTLINE_OVERVIEW_MAX_TOKENS,
        "outline_overview_total_tokens": outline_prompt_tokens +
            OUTLINE_OVERVIEW_MAX_TOKENS,
        "challenger_context_limit": CHALLENGER_CONTEXT_LIMIT,
        "challenger_model_identifier": CHALLENGER_MODEL,
        "solver_path_exists": True, "tokenizer_path_exists": True,
        "decision": "pass"})

    client = OpenAICompatibleChallengerClient(api)
    planner_root = run_root / "planner"
    planner_root.mkdir()
    planner_manifest = {
        **_manifest_base(root=planner_root, snapshot=snapshot,
                         project_diff=project_diff,
                         mode="stateful_outline_planner_chunked"),
        "source_parent_sha256": canonical_sha256(parent),
        "source_parent_turn_count": len(parent),
        "excluded_historical_child_turn_count": excluded_child_count,
        "prompt_version": STATEFUL_OUTLINE_CHUNKED_VERSION,
        "schema_version": STATEFUL_OUTLINE_SCHEMA_VERSION,
        "completion_contract_version":
            STATEFUL_COMPLETION_CONTRACT_VERSION,
        "overview_semantic_prompt_sha256": sha256_bytes(
            outline_overview_prompt.encode()),
        "schema_sha256": canonical_sha256(RAW_STATEFUL_OUTLINE_SCHEMA),
        "challenger_model_identifier": api.model,
        "sampling_parameters": {"temperature": api.temperature,
                                "overview_max_tokens":
                                    OUTLINE_OVERVIEW_MAX_TOKENS,
                                "arc_max_tokens": OUTLINE_ARC_MAX_TOKENS,
                                "enable_thinking": api.enable_thinking}}
    atomic_json(planner_root / "config/run_manifest.json", planner_manifest)
    planner_ledger = RequestLedger(
        planner_root, sha256_file(planner_root / "config/run_manifest.json"),
        "live")
    planner_budget = ApiBudget(
        normal_caps={"stateful_outline_overview": 1,
                     "stateful_outline_arc": 4},
        format_retry_cap=5, physical_cap=10)
    lineage_view = _view(parent)
    overview_task = run_single_object_task(
        run_root=planner_root, ledger=planner_ledger,
        backend=LiveSemanticRepairBackend(
            client, OUTLINE_OVERVIEW_MAX_TOKENS), budget=planner_budget,
        stage="stateful_outline_overview",
        logical_request_id="stateful-outline-overview",
        semantic_prompt=outline_overview_prompt, lineage_view=lineage_view,
        validator=lambda raw: validate_stateful_outline_overview(
            raw, parent_turns=parent),
        schema_retry_decisions=("stateful_outline_overview_schema_invalid",))
    atomic_json(planner_root / "reports/overview_task.json", {
        key: value for key, value in overview_task.items()
        if key not in {"accepted_raw", "accepted_validation"}})
    if not overview_task["accepted"]:
        result = {"decision": "stateful_outline_planner_failed",
                  "failed_stage": "overview",
                  "planner_tasks": [{key: value for key, value in
                                     overview_task.items() if key not in {
                                         "accepted_raw",
                                         "accepted_validation"}}],
                  "api_usage": _usage(run_root),
                  "realizer_scenes": 0, "questioner_candidates": 0,
                  "solver_evaluated": 0}
        atomic_json(run_root / "reports/stateful_expansion_qa_solver_canary.json",
                    result)
        return result
    # scene_role is a program-owned deterministic 5/5 schedule and is absent
    # from the Challenger's overview object.
    overview = overview_task["accepted_validation"]["overview"]
    atomic_json(run_root / "frozen/raw_stateful_outline_overview.json",
                overview_task["accepted_raw"])
    atomic_json(run_root / "frozen/stateful_outline_overview.json", overview)
    planner_tasks = [overview_task]
    arcs: list[dict[str, Any]] = []
    arc_preflights: list[dict[str, Any]] = []
    for arc_index in range(1, 5):
        assigned_scene_indices = _arc_scene_indices(arc_index)
        arc_prompt = build_stateful_outline_arc_prompt(
            parent_turns=parent, overview=overview, arc_index=arc_index,
            assigned_scene_indices=assigned_scene_indices)
        arc_prompt_tokens = len(tokenizer.encode(
            arc_prompt, add_special_tokens=False))
        arc_preflight = {
            "arc_id": f"E{arc_index}",
            "assigned_scene_indices": list(assigned_scene_indices),
            "rendered_prompt_tokens": arc_prompt_tokens,
            "reserved_completion_tokens": OUTLINE_ARC_MAX_TOKENS,
            "total_tokens": arc_prompt_tokens + OUTLINE_ARC_MAX_TOKENS,
            "context_limit": CHALLENGER_CONTEXT_LIMIT,
            "decision": "pass" if arc_prompt_tokens +
                OUTLINE_ARC_MAX_TOKENS <= CHALLENGER_CONTEXT_LIMIT else
                "reject_context_overflow",
        }
        arc_preflights.append(arc_preflight)
        if arc_preflight["decision"] != "pass":
            result = {"decision": "stateful_outline_planner_failed",
                      "failed_stage": f"arc_E{arc_index}_preflight",
                      "planner_tasks": [], "arc_preflights": arc_preflights,
                      "api_usage": _usage(run_root), "realizer_scenes": 0,
                      "questioner_candidates": 0, "solver_evaluated": 0}
            atomic_json(
                run_root / "reports/stateful_expansion_qa_solver_canary.json",
                result)
            return result
        arc_task = run_single_object_task(
            run_root=planner_root, ledger=planner_ledger,
            backend=LiveSemanticRepairBackend(client, OUTLINE_ARC_MAX_TOKENS),
            budget=planner_budget, stage="stateful_outline_arc",
            logical_request_id=f"stateful-outline-E{arc_index}",
            semantic_prompt=arc_prompt, lineage_view=lineage_view,
            validator=lambda raw, assigned=assigned_scene_indices:
                validate_stateful_outline_arc(
                    raw, parent_turns=parent, overview=overview,
                    assigned_scene_indices=assigned),
            schema_retry_decisions=("stateful_outline_arc_schema_invalid",))
        planner_tasks.append(arc_task)
        atomic_json(planner_root / f"reports/arc_E{arc_index}_task.json", {
            key: value for key, value in arc_task.items()
            if key not in {"accepted_raw", "accepted_validation"}})
        if not arc_task["accepted"]:
            result = {"decision": "stateful_outline_planner_failed",
                      "failed_stage": f"arc_E{arc_index}",
                      "planner_tasks": [{key: value for key, value in
                                         task.items() if key not in {
                                             "accepted_raw",
                                             "accepted_validation"}}
                                        for task in planner_tasks],
                      "arc_preflights": arc_preflights,
                      "api_usage": _usage(run_root), "realizer_scenes": 0,
                      "questioner_candidates": 0, "solver_evaluated": 0}
            atomic_json(
                run_root / "reports/stateful_expansion_qa_solver_canary.json",
                result)
            return result
        arc = dict(arc_task["accepted_validation"]["arc"])
        arcs.append(arc)
        atomic_json(run_root / f"frozen/raw_stateful_outline_arc_E{arc_index}.json",
                    arc_task["accepted_raw"])
        atomic_json(run_root / f"frozen/stateful_outline_arc_E{arc_index}.json",
                    arc)
    atomic_json(run_root / "reports/outline_chunk_preflights.json",
                {"overview_prompt_tokens": outline_prompt_tokens,
                 "arcs": arc_preflights})
    raw_outline = {**dict(overview), "event_arcs": arcs}
    assembled_validation = validate_raw_stateful_outline(
        raw_outline, parent_turns=parent)
    if assembled_validation["decision"] != "stateful_outline_accept":
        result = {"decision": "stateful_outline_planner_failed",
                  "failed_stage": "deterministic_assembly_validation",
                  "assembly_validation": assembled_validation,
                  "api_usage": _usage(run_root), "realizer_scenes": 0,
                  "questioner_candidates": 0, "solver_evaluated": 0}
        atomic_json(run_root / "reports/stateful_expansion_qa_solver_canary.json",
                    result)
        return result
    outline = normalize_stateful_outline(raw_outline, parent_turns=parent)
    contracts = compile_stateful_scene_contracts(outline)
    atomic_json(run_root / "frozen/raw_stateful_outline.json", raw_outline)
    atomic_json(run_root / "frozen/stateful_outline.json", outline)
    atomic_json(run_root / "frozen/stateful_scene_contracts.json", {
        "contract_version": STATEFUL_GRAPH_SCENE_CONTRACT_VERSION,
        "contracts": contracts, "contracts_sha256": canonical_sha256(contracts)})

    expansion_root = run_root / "expansion"
    expansion_root.mkdir()
    expansion_manifest = {
        **_manifest_base(root=expansion_root, snapshot=snapshot,
                         project_diff=project_diff,
                         mode="live_stateful_graph_expansion"),
        "source_parent_sha256": canonical_sha256(parent),
        "outline_sha256": canonical_sha256(outline),
        "scene_contracts_sha256": canonical_sha256(contracts),
        "graph_expansion_version": STATEFUL_GRAPH_EXPANSION_VERSION,
        "graph_scene_contract_version": STATEFUL_GRAPH_SCENE_CONTRACT_VERSION,
        "state_ledger_version": STATEFUL_GRAPH_LEDGER_VERSION,
        "realizer_prompt_version": STATEFUL_GRAPH_REALIZER_PROMPT_VERSION,
        "realizer_prompt_template_sha256": sha256_bytes(
            build_stateful_graph_realizer_prompt(
                contract=contracts[0], recent_sessions=[],
                active_handoffs=[]).encode()),
        "challenger_model_identifier": api.model,
        "sampling_parameters": {"temperature": api.temperature,
                                "max_tokens": api.realizer_max_tokens,
                                "enable_thinking": api.enable_thinking},
        "questioner_generations": 0, "solver_generations": 0,
        "verifier_generations": 0, "training_runs": 0}
    atomic_json(expansion_root / "config/run_manifest.json", expansion_manifest)
    stateful = StatefulGraphRoundOrchestrator(
        run_root=expansion_root, manifest=expansion_manifest,
        contracts=contracts, tokenizer=tokenizer,
        realizer=LiveRealizerBackend(client), backend_mode="live",
        max_attempts_per_scene=3, max_realizer_generations=30)
    expansion = stateful.run()
    sessions = expansion["sessions"]
    child = [turn for session in sessions
             for turn in session.get("dialogue_turns") or []]
    if len(sessions) != 10:
        result = {"decision": "stateful_realizer_incomplete",
                  "realizer_scenes": len(sessions),
                  "completion": expansion["completion"],
                  "api_usage": _usage(run_root),
                  "questioner_candidates": 0, "solver_evaluated": 0}
        atomic_json(run_root / "reports/stateful_expansion_qa_solver_canary.json",
                    result)
        return result
    full_dialogue = parent + child
    anchor_turn_ids = {row["turn_id"] for row in outline["parent_anchors"]}
    compact_parent = [turn for turn in parent
                      if turn["turn_id"] in anchor_turn_ids]
    compact_dialogue = compact_parent + child
    if len({turn["turn_id"] for turn in full_dialogue}) != len(full_dialogue):
        raise RuntimeError("stateful_full_dialogue_turn_id_collision")
    atomic_json(run_root / "frozen/new_child_dialogue.json", {
        "turns": child, "turn_count": len(child),
        "dialogue_tokens": _tokens(tokenizer, child)})
    atomic_json(run_root / "frozen/full_parent_plus_new_child.json", {
        "parent_turns": parent, "child_turns": child,
        "combined_dialogue": full_dialogue})
    atomic_json(run_root / "frozen/compact_questioner_view.json", {
        "parent_anchor_turns": compact_parent, "child_turns": child,
        "combined_dialogue": compact_dialogue})

    questioner_root = run_root / "questioner"
    questioner_root.mkdir()
    artifacts = _questioner_artifacts(
        root=questioner_root, compact_turns=compact_dialogue,
        tokenizer=tokenizer, api=api, feedback=feedback)
    if artifacts["preflight"]["decision"] != "pass":
        result = {"decision": "questioner_context_overflow",
                  "realizer_scenes": len(sessions),
                  "completion": expansion["completion"],
                  "questioner_preflight": artifacts["preflight"],
                  "api_usage": _usage(run_root), "solver_evaluated": 0}
        atomic_json(run_root / "reports/stateful_expansion_qa_solver_canary.json",
                    result)
        return result
    qconfig = _questioner_config(run_root=questioner_root, api=api)
    qmanifest = {
        **_manifest_base(root=questioner_root, snapshot=snapshot,
                         project_diff=project_diff,
                         mode="stateful_compact_questioner_v8"),
        "lineage": "conv-41", "compact_view": True,
        "compact_parent_anchor_turn_count": len(compact_parent),
        "new_child_turn_count": len(child),
        "frozen_dialogue_sha256": canonical_sha256(compact_dialogue),
        "span_catalog_sha256": span_catalog_sha256(artifacts["catalog"]),
        "reasoning_contract_sha256": reasoning_contract_sha256(
            artifacts["contracts"]),
        "target_gold_letters": list(artifacts["letters"]),
        "questioner_prompt_version": QUESTIONER_SPAN_IDS_ONLY_PROMPT_VERSION,
        "questioner_prompt_sha256": span_ids_only_questioner_template_hash(),
        "rendered_prompt_sha256": sha256_bytes(artifacts["prompt"].encode()),
        "questioner_schema_version": QUESTIONER_SPAN_IDS_ONLY_SCHEMA_VERSION,
        "questioner_output_profile": SPAN_IDS_ONLY_SCHEMA_PROFILE,
        "format_retry_version": QUESTIONER_FORMAT_RETRY_VERSION,
        "format_retry_suffix_sha256": questioner_format_retry_suffix_hash(),
        "feedback_memory_sha256": sha256_file(FEEDBACK_PATH),
        "questioner_prompt_preflight": artifacts["preflight"],
        "challenger_model_identifier": api.model,
        "sampling_parameters": {"temperature": api.temperature,
                                "max_tokens": api.questioner_max_tokens},
        "budget": {"normal_batches": 1, "maximum_format_retries": 1,
                   "maximum_paid_requests": 2, "realizer": 0,
                   "verifier": 0, "training": 0},
        "live_backend_source_sha256": sha256_file(
            MODULE_ROOT / "live_backends.py")}
    atomic_json(questioner_root / "config/resolved_canary_config.json", qconfig)
    atomic_json(questioner_root / "config/run_manifest.json", qmanifest)
    qresult = _run_questioner(
        root=questioner_root, manifest=qmanifest, config=qconfig,
        artifacts=artifacts, backend=LiveQuestionerBackend(client),
        tokenizer=tokenizer, feedback_sha=sha256_file(FEEDBACK_PATH))
    write_post_run_provenance(questioner_root, PROJECT, qmanifest)

    solver_reports = []
    candidates = [admission for admission in qresult["dual"].get("items", [])
                  if admission.get("solver_eligible") is True and
                  admission["item_index"] in qresult["materialized"]]
    solver_backend = None
    if candidates:
        solver_backend = VLLMSolverBackend(
            model_path=SOLVER, tokenizer_path=TOKENIZER,
            max_model_len=SOLVER_MAX_MODEL_LEN, temperature=0.7, top_p=0.9,
            max_tokens=32, gpu_memory_utilization=0.80)
    solver_artifacts = {**artifacts,
        "context": {"schema_version": "stateful_full_solver_context_v1",
                    "lineage": "conv-41", "combined_dialogue": full_dialogue}}
    for admission in candidates:
        qa = qresult["materialized"][admission["item_index"]]
        solver_root = run_root / f"solvers/item_{admission['item_index']:02d}"
        solver_root.mkdir(parents=True)
        sconfig = _solver_config(run_root=solver_root)
        smanifest = {
            **_manifest_base(root=solver_root, snapshot=snapshot,
                             project_diff=project_diff,
                             mode="local_solver_stateful_round"),
            "lineage": "conv-41", "qa_id": qa["qa_id"],
            "source_questioner_root": str(questioner_root),
            "source_questioner_manifest_sha256": sha256_file(
                questioner_root / "config/run_manifest.json"),
            "materialized_qa_sha256": canonical_sha256(qa),
            "questioner_view_sha256": canonical_sha256(compact_dialogue),
            "full_solver_context_sha256": canonical_sha256(full_dialogue),
            "full_solver_parent_turn_count": len(parent),
            "full_solver_child_turn_count": len(child),
            "span_catalog_sha256": span_catalog_sha256(artifacts["catalog"]),
            "solver_max_model_len": SOLVER_MAX_MODEL_LEN,
            "vllm_backend_source_sha256": sha256_file(
                MODULE_ROOT / "vllm_backends.py"),
            "solver_protocol": {"enable_thinking": False,
                                "temperature": 0.7, "top_p": 0.9,
                                "max_tokens": 32,
                                "parser": "boxed_letter_v1",
                                "max_parse_retries_per_target": 1}}
        atomic_json(solver_root / "config/resolved_canary_config.json", sconfig)
        atomic_json(solver_root / "config/run_manifest.json", smanifest)
        report = _run_solver(
            root=solver_root, manifest=smanifest, config=sconfig,
            raw_batch=qresult["raw_batch"], admission=admission,
            materialized=qa, artifacts=solver_artifacts,
            tokenizer=tokenizer, backend=solver_backend)
        write_post_run_provenance(solver_root, PROJECT, smanifest)
        solver_reports.append({"item_index": admission["item_index"],
                               "qa_id": qa["qa_id"],
                               "solver_root": str(solver_root),
                               "final_decision": report["final_decision"],
                               "report": report})

    all_speakers = sorted({str(turn["speaker"]) for turn in full_dialogue})
    new_speakers = sorted({str(turn["speaker"]) for turn in child} -
                          {str(turn["speaker"]) for turn in parent})
    result = {
        "decision": "stateful_expansion_qa_solver_complete",
        "stateful_completion": expansion["completion"],
        "parent_turn_count": len(parent), "parent_tokens": _tokens(tokenizer, parent),
        "new_child_turn_count": len(child),
        "new_child_tokens": _tokens(tokenizer, child),
        "full_dialogue_turn_count": len(full_dialogue),
        "full_dialogue_tokens": _tokens(tokenizer, full_dialogue),
        "compact_questioner_turn_count": len(compact_dialogue),
        "compact_questioner_tokens": _tokens(tokenizer, compact_dialogue),
        "character_count": len(all_speakers), "characters": all_speakers,
        "new_character_count": len(new_speakers),
        "new_characters": new_speakers,
        "questioner_raw_candidates": qresult["report"]["raw_candidate_count"],
        "questioner_minimal_accepted": qresult["report"][
            "minimal_accepted_count"],
        "solver_evaluated": len(solver_reports),
        "solver_results": solver_reports,
        "api_usage": _usage(run_root),
        "ordinary_questioner_generations": qresult["report"][
            "paid_api_generations"],
        "verifier_generations": 0, "training_runs": 0,
    }
    atomic_json(run_root / "reports/stateful_expansion_qa_solver_canary.json",
                result)
    checks = {
        "git_head_match": git_value(PROJECT, "rev-parse", "HEAD") ==
            top_manifest["project_git_commit"],
        "project_diff_match": project_diff_sha256(PROJECT) == project_diff,
        "source_snapshot_match": sha256_file(Path(
            snapshot["source_snapshot_manifest_path"])) == snapshot[
                "source_snapshot_manifest_sha256"],
        "historical_child_excluded": excluded_child_count == 134,
        "realizer_only_stateful_v2": all(
            "completed_event_stage_ids" not in session for session in sessions),
        "verifier_zero": True, "training_zero": True,
    }
    atomic_json(run_root / "reports/post_run_provenance.json", {
        "authoritative_release_provenance": all(checks.values()),
        "provenance_mixed": not all(checks.values()), "checks": checks,
        "run_manifest_sha256": sha256_file(
            run_root / "config/run_manifest.json"),
        "report_sha256": sha256_file(
            run_root / "reports/stateful_expansion_qa_solver_canary.json")})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--api-config", type=Path, default=Path(
        "/root/autodl-tmp/gxy-selfplay/config.json"))
    args = parser.parse_args()
    result = run(run_root=args.run_root, api_config=args.api_config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
