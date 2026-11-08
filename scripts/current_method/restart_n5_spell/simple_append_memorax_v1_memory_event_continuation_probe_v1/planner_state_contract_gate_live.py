"""One bounded Planner-state -> Gate probe over frozen R4 artifacts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from transformers import AutoTokenizer

from ..core import atomic_json, atomic_text, sha256_file
from ..ledger import RequestLedger
from ..live_backends import OpenAICompatibleChallengerClient, load_challenger_config
from .contracts import build_semantic_gate_context, validate_semantic_gate_output
from .incremental_round import load_incremental_round_input
from .live import (CONTEXT_LIMIT, LIVE_TIMEOUT_SECONDS, PLANNER_MAX_TOKENS,
                   SEMANTIC_GATE_MAX_TOKENS, _manifest)
from .live_runtime import SimpleAppendLiveCaller, aggregate_usage
from .memory_context import compile_global_memory_state_view
from .prompts import build_semantic_continuation_gate_prompt


RUNS = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
R3 = RUNS / "conversation_chronicles_episode_41625_memory_only_incremental_r3_live_2"
R4 = RUNS / "conversation_chronicles_episode_41625_memory_only_incremental_r4_live_2"
RUN_ROOT = RUNS / "conversation_chronicles_episode_41625_planner_state_contract_gate_probe_1"
CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")

STATE_STAGES = {
    "PROPOSED", "IN_PROGRESS", "AWAITING_CONFIRMATION",
    "READY_TO_FINALIZE", "COMPLETED", "AMBIGUOUS",
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _memory_ref_map(view: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for rows in (view.get("buckets") or {}).values():
        for row in rows or []:
            for fact in row.get("memory_facts") or []:
                ref = str(fact.get("memory_ref") or "")
                if ref:
                    result[ref] = str(fact.get("text") or "")
    return result


def _candidate_rows(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for scene in plan.get("candidate_pool") or plan.get("scenes") or []:
        event_id = str(scene.get("memory_event_id") or "")
        for index, candidate in enumerate(
                scene.get("next_step_candidates") or [], 1):
            result.append({
                "candidate_ref": f"{event_id}:C{index:02d}",
                "development_mode": scene.get("development_mode"),
                "memory_event_display": scene.get("memory_event_display"),
                "source_basis": scene.get("source_basis"),
                "before_state": candidate.get("before_state"),
                "required_transition": candidate.get("required_transition"),
                "after_state": candidate.get("after_state"),
            })
    return result


def build_planner_state_prompt(
        *, view: Mapping[str, Any], candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    refs = [row["candidate_ref"] for row in candidates]
    prompt = f"""You are the state-contract stage of a Memory-only expansion
Planner. The proposed transitions are frozen; do not rewrite or improve them.
For each candidate, determine only how far its underlying event has already
progressed in the supplied Memorax facts.

Allowed current_state_stage values:
- PROPOSED: suggested or offered, but not started.
- IN_PROGRESS: action has started but is not complete.
- AWAITING_CONFIRMATION: outreach/action occurred and an external response is pending.
- READY_TO_FINALIZE: prerequisites exist, but final confirmation/result has not occurred.
- COMPLETED: the exact required_transition or after_state is explicitly already true.
- AMBIGUOUS: the supplied facts cannot support one of the other stages.

Do not turn pending into completed. "being finalized", "tentative", "will",
"plans to", "follow-up", and "responses pending" do not prove completion.
Same topic, actor, or object does not prove the exact transition is completed.
Use one to three state_basis_refs copied from supplied memory_ref values for
every non-AMBIGUOUS review. AMBIGUOUS may use an empty list. Copy each frozen
required_transition exactly into next_transition. completion_condition is one
short sentence describing the observable fact that would prove completion.

REQUIRED CANDIDATES
{json.dumps(refs, ensure_ascii=False)}

GLOBAL MEMORY FACTS
{json.dumps(view, ensure_ascii=False, sort_keys=True, indent=2)}

FROZEN CANDIDATES
{json.dumps(candidates, ensure_ascii=False, sort_keys=True, indent=2)}

Output only this JSON object with no extra keys:
{{
  "candidate_state_reviews": [
    {{
      "candidate_ref": "ME01:C01",
      "current_state_stage": "AWAITING_CONFIRMATION",
      "current_state_summary": "A concrete current state supported by Memory.",
      "state_basis_refs": ["GM:E001:C001"],
      "next_transition": "Copy the candidate required_transition exactly.",
      "completion_condition": "A concrete observable completion condition."
    }}
  ]
}}"""
    return {
        "prompt_version": "planner_memory_state_contract_prompt_v1",
        "prompt": prompt,
        "candidate_refs": refs,
    }


def validate_planner_state_output(
        value: Any, *, candidates: list[dict[str, Any]],
        memory_ref_map: Mapping[str, str],
) -> dict[str, Any]:
    errors: list[str] = []
    expected = {row["candidate_ref"]: row for row in candidates}
    if not isinstance(value, Mapping) or set(value) != {"candidate_state_reviews"}:
        return {"valid": False, "errors": ["planner_state_top_level_shape"]}
    rows = value.get("candidate_state_reviews")
    if not isinstance(rows, list) or len(rows) != len(expected):
        errors.append("planner_state_review_count")
        rows = rows if isinstance(rows, list) else []
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    keys = {"candidate_ref", "current_state_stage", "current_state_summary",
            "state_basis_refs", "next_transition", "completion_condition"}
    for index, row in enumerate(rows, 1):
        prefix = f"planner_state_review_{index}"
        if not isinstance(row, Mapping):
            errors.append(f"{prefix}_not_object")
            continue
        if set(row) != keys:
            errors.append(f"{prefix}_shape")
        ref = str(row.get("candidate_ref") or "")
        if ref not in expected:
            errors.append(f"{prefix}_unknown_candidate")
        if ref in seen:
            errors.append(f"{prefix}_duplicate_candidate")
        seen.add(ref)
        stage = str(row.get("current_state_stage") or "")
        if stage not in STATE_STAGES:
            errors.append(f"{prefix}_stage")
        summary = str(row.get("current_state_summary") or "").strip()
        condition = str(row.get("completion_condition") or "").strip()
        if not summary:
            errors.append(f"{prefix}_summary")
        if not condition:
            errors.append(f"{prefix}_completion_condition")
        basis = row.get("state_basis_refs")
        if (not isinstance(basis, list) or len(basis) > 3 or
                len(basis) != len(set(str(item) for item in basis)) or
                any(str(item) not in memory_ref_map for item in basis)):
            errors.append(f"{prefix}_basis_refs")
            basis = basis if isinstance(basis, list) else []
        if stage != "AMBIGUOUS" and not basis:
            errors.append(f"{prefix}_basis_refs_required")
        next_transition = str(row.get("next_transition") or "").strip()
        expected_transition = str((expected.get(ref) or {}).get(
            "required_transition") or "").strip()
        if next_transition != expected_transition:
            errors.append(f"{prefix}_next_transition_drift")
        normalized.append({
            "candidate_ref": ref,
            "current_state_stage": stage,
            "current_state_summary": summary,
            "state_basis_refs": [str(item) for item in basis],
            "next_transition": next_transition,
            "completion_condition": condition,
        })
    if seen != set(expected):
        errors.append("planner_state_candidate_coverage")
    eligible = [row["candidate_ref"] for row in normalized
                if row["current_state_stage"] not in {"COMPLETED", "AMBIGUOUS"}]
    return {
        "valid": not errors,
        "errors": errors,
        "normalized_reviews": normalized,
        "eligible_candidate_refs": eligible,
        "eligible_candidate_count": len(eligible),
    }


def _annotate_plan(
        plan: Mapping[str, Any], reviews: list[Mapping[str, Any]],
) -> dict[str, Any]:
    result = deepcopy(plan)
    by_ref = {str(row["candidate_ref"]): row for row in reviews}
    for scene in result.get("candidate_pool") or result.get("scenes") or []:
        event_id = str(scene.get("memory_event_id") or "")
        for index, candidate in enumerate(
                scene.get("next_step_candidates") or [], 1):
            review = by_ref[f"{event_id}:C{index:02d}"]
            candidate["planner_current_state_stage"] = review[
                "current_state_stage"]
            candidate["planner_current_state_summary"] = review[
                "current_state_summary"]
            candidate["planner_state_basis_refs"] = deepcopy(
                review["state_basis_refs"])
            candidate["planner_completion_condition"] = review[
                "completion_condition"]
    return result


def _filter_plan_to_eligible(
        plan: Mapping[str, Any], eligible_refs: set[str]) -> dict[str, Any]:
    """Keep only state-contract candidates eligible for Gate consideration."""
    result = deepcopy(plan)
    for group in ("candidate_pool", "scenes"):
        filtered_scenes: list[dict[str, Any]] = []
        for scene in result.get(group) or []:
            event_id = str(scene.get("memory_event_id") or "")
            candidates = []
            for index, candidate in enumerate(
                    scene.get("next_step_candidates") or [], 1):
                if f"{event_id}:C{index:02d}" in eligible_refs:
                    candidates.append(candidate)
            if candidates:
                scene["next_step_candidates"] = candidates
                filtered_scenes.append(scene)
        result[group] = filtered_scenes
    return result


def main() -> int:
    if RUN_ROOT.exists():
        raise FileExistsError(f"planner_state_gate_run_exists:{RUN_ROOT}")
    sources = {
        "prior_expanded": R3 / "promotion/promotable_expanded_dialogue.json",
        "memory_snapshot": R4 / "memory/recovery_pre_generation_bucket_snapshot.json",
        "event_candidates": R4 / "memory/event_candidates.json",
        "compiled_plan": R4 / "planner/compiled_plan.json",
        "config": CONFIG,
    }
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("planner_state_gate_sources_missing:" +
                                ",".join(missing))
    parent, _, input_validation = load_incremental_round_input(
        sources["prior_expanded"], round_index=4)
    snapshot = _read(sources["memory_snapshot"])
    event_candidates = _read(sources["event_candidates"])["candidates"]
    plan = _read(sources["compiled_plan"])
    view = compile_global_memory_state_view(snapshot)
    candidates = _candidate_rows(plan)
    ref_map = _memory_ref_map(view)
    state_prompt = build_planner_state_prompt(view=view, candidates=candidates)

    RUN_ROOT.mkdir(parents=True, exist_ok=False)
    for part in ("input", "planner_state", "semantic_gate", "reports",
                 "raw/prompts", "attempts"):
        (RUN_ROOT / part).mkdir(parents=True, exist_ok=True)
    source_hashes = {name: sha256_file(path) for name, path in sources.items()}
    freeze = {
        "freeze_version": "planner_state_contract_gate_probe_v1",
        "frozen_at": _now(), "run_root": str(RUN_ROOT),
        "root_id": parent["root_id"],
        "source_paths": {name: str(path) for name, path in sources.items()},
        "source_sha256": source_hashes,
        "source_manifest": _manifest(),
        "input_validation": input_validation,
        "planner_calls_max": 1, "semantic_gate_calls_max": 1,
        "retry": 0, "realizer_calls": 0, "memorax_calls": 0,
    }
    atomic_json(RUN_ROOT / "run_freeze_manifest.json", freeze)
    atomic_json(RUN_ROOT / "input/global_memory_state_view.json", view)
    atomic_json(RUN_ROOT / "input/frozen_candidates.json", {"candidates": candidates})
    atomic_json(RUN_ROOT / "input/compiled_plan.json", plan)
    atomic_text(RUN_ROOT / "planner_state/prompt.txt", state_prompt["prompt"])

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    counter = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    config = replace(load_challenger_config(CONFIG),
                     timeout_seconds=LIVE_TIMEOUT_SECONDS)
    ledger = RequestLedger(
        RUN_ROOT, manifest_sha256=freeze["source_manifest"]["manifest_sha256"],
        backend_mode="live")
    allowed_calls = {"planner": 1, "semantic_continuation_gate": 1}
    actual_calls = {key: 0 for key in allowed_calls}

    def before_request(logical_role: str, role_version: str) -> None:
        if logical_role not in allowed_calls:
            raise RuntimeError(f"planner_state_gate_forbidden_role:{logical_role}")
        actual_calls[logical_role] += 1
        if actual_calls[logical_role] > allowed_calls[logical_role]:
            raise RuntimeError(f"planner_state_gate_budget:{logical_role}")
        if _manifest()["manifest_sha256"] != freeze["source_manifest"][
                "manifest_sha256"]:
            raise RuntimeError("planner_state_gate_source_drift")
        for name, path in sources.items():
            if sha256_file(path) != source_hashes[name]:
                raise RuntimeError(f"planner_state_gate_input_drift:{name}")
        atomic_json(RUN_ROOT / "run_status.json", {
            "status": "live_request_started", "started_at": _now(),
            "logical_role": logical_role, "role_version": role_version,
            "actual_calls": actual_calls,
        })

    caller = SimpleAppendLiveCaller(
        run_root=RUN_ROOT, ledger=ledger,
        client=OpenAICompatibleChallengerClient(config), token_counter=counter,
        root_id=parent["root_id"], context_limit=CONTEXT_LIMIT,
        before_request=before_request)

    state_output, state_raw, state_parse_errors = caller.call_json(
        request_id="planner_state_contract_attempt_01",
        role="simple_append_planner_v1", prompt=state_prompt["prompt"],
        prompt_version=state_prompt["prompt_version"],
        max_tokens=PLANNER_MAX_TOKENS)
    atomic_text(RUN_ROOT / "planner_state/raw_output.txt", state_raw)
    if state_output is not None:
        atomic_json(RUN_ROOT / "planner_state/output.json", state_output)
    state_validation = (validate_planner_state_output(
        state_output, candidates=candidates, memory_ref_map=ref_map)
        if state_output is not None else
        {"valid": False, "errors": state_parse_errors or ["no_response"]})
    atomic_json(RUN_ROOT / "planner_state/validation.json", state_validation)

    gate_validation: dict[str, Any] = {
        "valid": False, "errors": ["planner_state_contract_failed"],
        "realizer_allowed": False,
    }
    if state_validation.get("valid"):
        annotated = _annotate_plan(
            plan, state_validation["normalized_reviews"])
        atomic_json(RUN_ROOT / "planner_state/annotated_plan.json", annotated)
        eligible_refs = set(state_validation.get(
            "eligible_candidate_refs") or [])
        if len(eligible_refs) < 3:
            gate_validation = {
                "valid": True,
                "errors": ["planner_state_insufficient_eligible_candidates"],
                "overall_decision": "reject",
                "passed_candidate_refs": [],
                "rejected_candidate_refs": [],
                "realizer_allowed": False,
                "gate_call_skipped": True,
            }
        else:
            gate_plan = _filter_plan_to_eligible(annotated, eligible_refs)
            atomic_json(RUN_ROOT / "planner_state/gate_plan.json", gate_plan)
            gate_context = build_semantic_gate_context(
                parent=parent, candidates=event_candidates, plan=gate_plan,
                memory_only_fast=True, global_memory_state_view=view)
            gate_prompt = build_semantic_continuation_gate_prompt(
                gate_context=gate_context)
            atomic_json(RUN_ROOT / "semantic_gate/context.json", gate_context)
            atomic_text(RUN_ROOT / "semantic_gate/prompt.txt", gate_prompt["prompt"])
            gate_output, gate_raw, gate_parse_errors = caller.call_json(
                request_id="semantic_continuation_gate_attempt_01",
                role="memory_event_realization_aware_semantic_gate_v1",
                prompt=gate_prompt["prompt"],
                prompt_version=gate_prompt["prompt_version"],
                max_tokens=SEMANTIC_GATE_MAX_TOKENS)
            atomic_text(RUN_ROOT / "semantic_gate/raw_output.txt", gate_raw)
            if gate_output is not None:
                atomic_json(RUN_ROOT / "semantic_gate/output.json", gate_output)
            gate_validation = (validate_semantic_gate_output(
                gate_output, plan=gate_plan, parent=parent,
                memory_only_fast=True, global_memory_state_view=view)
                if gate_output is not None else
                {"valid": False, "errors": gate_parse_errors or ["no_response"],
                 "realizer_allowed": False})
            atomic_json(RUN_ROOT / "semantic_gate/validation.json", gate_validation)

    usage = aggregate_usage(caller.records)
    report = {
        "probe_version": "planner_state_contract_gate_probe_v1",
        "planner_state_validation": state_validation,
        "semantic_gate_validation": gate_validation,
        "realizer_started": False,
        "actual_calls": actual_calls,
        "retry": 0,
        "usage": usage,
    }
    atomic_json(RUN_ROOT / "reports/PLANNER_STATE_CONTRACT_GATE_REPORT.json", report)
    lines = [
        "# Planner State Contract + Gate Probe", "",
        f"- Planner state contract valid: `{state_validation.get('valid')}`",
        f"- Gate contract valid: `{gate_validation.get('valid')}`",
        f"- Gate decision: `{gate_validation.get('overall_decision')}`",
        f"- Realizer started: `false`", "", "## Planner states", "",
    ]
    for row in state_validation.get("normalized_reviews") or []:
        lines.append(
            f"- `{row['candidate_ref']}`: `{row['current_state_stage']}` "
            f"via `{row['state_basis_refs']}`")
    atomic_text(RUN_ROOT / "reports/PLANNER_STATE_CONTRACT_GATE_REPORT.md",
                "\n".join(lines) + "\n")
    status = "completed" if (state_validation.get("valid") and
                              gate_validation.get("valid")) else "failed"
    atomic_json(RUN_ROOT / "run_status.json", {
        "status": status, "finished_at": _now(),
        "actual_calls": actual_calls, "retry": 0,
        "realizer_started": False,
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
