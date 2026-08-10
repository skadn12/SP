"""Memory-event continuation orchestration; no QA or state ledger."""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping, Protocol, Sequence

from ..simple_append_v1.assembly import assemble_expanded_dialogue, validate_expanded_dialogue
from .async_write import DEFAULT_POLL_DELAYS_SECONDS, MemoraxAsyncWriteError, submit_add_and_wait
from .client import MemoraxContractError, MemoraxTransportError
from .context import build_fresh_r1_context
from .contracts import (
    build_event_candidates,
    build_semantic_gate_context,
    materialize_planner_output,
    normalize_realizer_output,
    validate_realizer_plan_isolation,
    validate_planner_output,
    validate_semantic_gate_output,
)
from .memory_context import compile_memory_overview, normalize_bucket_snapshot
from .payloads import build_add_intent, build_bucket_search_payload
from .prompts import (
    build_planner_prompt,
    build_realizer_prompt,
    build_semantic_continuation_gate_prompt,
)

class Gateway(Protocol):
    calls: list[dict[str, Any]]
    def add(self, payload: Mapping[str, Any], *, request_id: str) -> dict[str, Any]: ...
    def search_buckets(self, payload: Mapping[str, Any], *, request_id: str) -> dict[str, Any]: ...
    def add_status(self, task_id: str, *, request_id: str) -> dict[str, Any]: ...

class WorkflowError(RuntimeError):
    def __init__(self, stage: str, errors: Sequence[str] = ()):
        self.stage, self.errors = str(stage), list(errors)
        super().__init__(f"memory_event_continuation_failed:{self.stage}:{','.join(self.errors)}")

def _emit(sink, name, value):
    if sink: sink(name, value)

def _recent(rows):
    return "\n".join(str(row.get("text") or "") for row in rows)

def execute_round(*, parent: Mapping[str, Any], memorax: Gateway,
                  planner_call: Callable, semantic_gate_call: Callable,
                  realizer_call: Callable,
                  artifact_sink=None, before_memory_request=None,
                  token_counter=None, wait_for_memory_completion=True,
                  reuse_existing_namespace: bool = False,
                  memorax_user_namespace: str | None = None,
                  assume_new_namespace_empty: bool = False,
                  poll_delays_seconds=DEFAULT_POLL_DELAYS_SECONDS,
                  sleep_fn=time.sleep) -> dict[str, Any]:
    token_counter = token_counter or (lambda text: max(0, (len(text.encode())+3)//4))
    root_id = str(parent["root_id"])
    tail = build_fresh_r1_context(parent, token_counter=token_counter, budget_tokens=10000)
    _emit(artifact_sink, "parent_context", tail)
    query = build_bucket_search_payload(
        root_id=root_id, user_namespace=memorax_user_namespace)

    def search(*, artifact_name, logical_role, request_id):
        # Artifact labels are filenames; logical roles are the frozen request
        # contract names checked immediately before an outbound call.
        if before_memory_request:
            before_memory_request(logical_role, request_id)
        value = memorax.search_buckets(query["payload"], request_id=request_id)
        snapshot = normalize_bucket_snapshot(value, root_id=root_id, user_id=query["payload"]["user_id"])
        _emit(artifact_sink, f"memory/{artifact_name}_bucket_query", query)
        _emit(artifact_sink, f"memory/{artifact_name}_bucket_snapshot", snapshot)
        return snapshot

    parent_turns = list(parent["dialogue"]["turns"])
    parent_completion = None
    parent_state: dict[str, Any] = {}
    if reuse_existing_namespace:
        # Recovery is explicitly read-first: the already written Parent is
        # reused and must never be submitted a second time.
        pre_snapshot = search(
            artifact_name="recovery_pre_generation",
            logical_role="pre_generation_search",
            request_id=f"memory-event-recovery-pre-{root_id}")
        _emit(artifact_sink, "memory/parent_add_skipped", {
            "reason": "existing_namespace_reused",
            "parent_add_calls": 0,
        })
    else:
        if not assume_new_namespace_empty:
            isolation = search(
                artifact_name="isolation", logical_role="isolation_search",
                request_id=f"memory-event-isolation-{root_id}")
            if sum(isolation["bucket_counts"].values()):
                raise WorkflowError("memory_isolation", ["namespace_not_empty"])

        parent_sha = str(parent["dialogue"]["dialogue_hash"])
        parent_intent = build_add_intent(root_id=root_id, round_index=0,
                                         source_dialogue_sha256=parent_sha,
                                         turns=parent_turns, async_mode=True,
                                         user_namespace=memorax_user_namespace)
        _emit(artifact_sink, "memory/parent_add_intent", parent_intent)
        if before_memory_request:
            before_memory_request("parent_add", parent_intent["request_id"])
        def on_parent(state):
            parent_state.update(state)
            _emit(artifact_sink, "memory/parent_add_write_state", state)
        try:
            parent_completion = submit_add_and_wait(
                gateway=memorax, intent=parent_intent,
                before_status_request=before_memory_request,
                poll_delays_seconds=poll_delays_seconds,
                sleep_fn=sleep_fn, stage="parent_memory_add",
                on_state_change=on_parent)
        except (MemoraxAsyncWriteError, MemoraxContractError,
                MemoraxTransportError) as e:
            raise WorkflowError("parent_memory_add", [str(e)])
        _emit(artifact_sink, "memory/parent_add_completion", parent_completion)
        pre_snapshot = search(
            artifact_name="pre_generation", logical_role="pre_generation_search",
            request_id=f"memory-event-pre-{root_id}")
    pre_overview=compile_memory_overview(pre_snapshot, recent_dialogue_text=_recent(parent_turns))
    _emit(artifact_sink,"memory/pre_generation_overview",pre_overview)
    candidates=build_event_candidates(pre_overview)
    _emit(artifact_sink,"memory/event_candidates",{"candidates":candidates})
    if len(candidates)<3:
        raise WorkflowError("memory_event_discovery",["fewer_than_three_concrete_events"])

    planner_prompt=build_planner_prompt(parent=parent,memory_overview=pre_overview,candidates=candidates)
    _emit(artifact_sink,"planner/prompt",planner_prompt)
    planner_raw=planner_call(planner_prompt)
    _emit(artifact_sink,"planner/output",planner_raw)
    validation=validate_planner_output(planner_raw,candidates=candidates,parent=parent)
    _emit(artifact_sink,"planner/validation",validation)
    if not validation["valid"]: raise WorkflowError("planner_continuation",validation["errors"])

    gate_context = build_semantic_gate_context(
        parent=parent, candidates=candidates, plan=planner_raw)
    gate_prompt = build_semantic_continuation_gate_prompt(
        gate_context=gate_context)
    _emit(artifact_sink, "semantic_gate/context", gate_context)
    _emit(artifact_sink, "semantic_gate/prompt", gate_prompt)
    gate_output = semantic_gate_call(gate_prompt)
    _emit(artifact_sink, "semantic_gate/output", gate_output)
    gate_validation = validate_semantic_gate_output(
        gate_output, plan=planner_raw, parent=parent)
    _emit(artifact_sink, "semantic_gate/validation", gate_validation)
    if not gate_validation["valid"]:
        raise WorkflowError("semantic_continuation_gate_contract",
                            gate_validation["errors"])
    reviews_by_ref = {
        str(review.get("candidate_ref")): dict(review)
        for review in gate_output.get("candidate_reviews") or []
        if isinstance(review, Mapping)}
    passed_refs = set(gate_validation["passed_candidate_refs"])
    selection_audit = {"events": []}
    for scene in planner_raw.get("scenes") or []:
        event_id = str(scene.get("memory_event_id") or "")
        candidate_rows = []
        selected_ref = None
        selected_candidate = None
        for index, candidate in enumerate(
                scene.get("next_step_candidates") or [], 1):
            candidate_ref = f"{event_id}:C{index:02d}"
            if selected_ref is None and candidate_ref in passed_refs:
                selected_ref = candidate_ref
                selected_candidate = candidate
            candidate_rows.append({
                "candidate_ref": candidate_ref,
                **dict(candidate),
                "gate_review": reviews_by_ref.get(candidate_ref),
            })
        selection_audit["events"].append({
            "memory_event_id": event_id,
            "candidates": candidate_rows,
            "selected_candidate_ref": selected_ref,
            "new_development": (selected_candidate or {}).get("new_development"),
            "durable_outcome": (selected_candidate or {}).get("durable_outcome"),
        })
    _emit(artifact_sink, "semantic_gate/candidate_selection", selection_audit)
    if gate_validation["overall_decision"] == "reject":
        raise WorkflowError(
            "semantic_continuation_gate",
            ["all_candidates_rejected:" + ",".join(
                gate_validation["events_without_pass"])]
        )
    try:
        planned = materialize_planner_output(
            planner_raw,
            passed_candidate_refs=gate_validation["passed_candidate_refs"])
    except (TypeError, ValueError) as error:
        raise WorkflowError("planner_materialization", [str(error)])
    _emit(artifact_sink, "planner/materialized_plan", planned)
    isolation = validate_realizer_plan_isolation(
        raw_plan=planner_raw, materialized_plan=planned,
        passed_candidate_refs=gate_validation["passed_candidate_refs"])
    _emit(artifact_sink, "realizer/input_validation", isolation)
    _emit(artifact_sink, "realizer/input", planned)
    if not isolation["valid"]:
        raise WorkflowError("realizer_input_isolation", isolation["errors"])

    realizer_prompt=build_realizer_prompt(parent=parent,plan=planned)
    _emit(artifact_sink,"realizer/prompt",realizer_prompt)
    realized=realizer_call(realizer_prompt)
    _emit(artifact_sink,"realizer/output",realized)
    normalized=normalize_realizer_output(realized,plan=planned,parent=parent)
    _emit(artifact_sink,"realizer/validation",normalized)
    if not normalized["valid"]: raise WorkflowError("realizer_adherence",normalized["errors"])
    child=normalized["child"]
    expanded=assemble_expanded_dialogue(parent=parent,child=child)
    expanded_validation=validate_expanded_dialogue(expanded,parent=parent)
    _emit(artifact_sink,"child_dialogue",child); _emit(artifact_sink,"expanded_dialogue",expanded)
    _emit(artifact_sink,"expanded_dialogue_validation",expanded_validation)
    if not expanded_validation["valid"]: raise WorkflowError("child_assembly",expanded_validation["errors"])
    child_turns=[{**row, "session_ref": scene["session_ref"]}
                 for scene in child["scenes"] for row in scene["dialogue_turns"]]
    child_intent=build_add_intent(root_id=root_id,round_index=1,
        source_dialogue_sha256=expanded["complete_dialogue_sha256"],turns=child_turns,
        recent_context_turns=parent_turns[-20:],async_mode=True,
        user_namespace=memorax_user_namespace)
    _emit(artifact_sink,"memory/child_add_intent",child_intent)
    if before_memory_request: before_memory_request("child_add",child_intent["request_id"])
    child_state={}
    def on_child(state): child_state.update(state); _emit(artifact_sink,"memory/child_add_write_state",state)
    try:
        child_completion=submit_add_and_wait(gateway=memorax,intent=child_intent,
            before_status_request=before_memory_request,poll_delays_seconds=poll_delays_seconds,
            sleep_fn=sleep_fn,stage="child_memory_add",on_state_change=on_child)
    except (MemoraxAsyncWriteError, MemoraxContractError, MemoraxTransportError) as e:
        raise WorkflowError("child_memory_add",[str(e)])
    _emit(artifact_sink,"memory/child_add_completion",child_completion)
    post_snapshot=search(
        artifact_name="post_generation", logical_role="post_generation_search",
        request_id=f"memory-event-post-{root_id}")
    post_overview=compile_memory_overview(post_snapshot,recent_dialogue_text=_recent(child_turns))
    _emit(artifact_sink,"memory/post_generation_overview",post_overview)
    return {"workflow_version":"memory_event_continuation_workflow_v1","root_id":root_id,
      "parent_context":tail,"candidates":candidates,"pre_overview":pre_overview,
      "planner_raw_output":planner_raw,"planner_output":planned,
      "planner_validation":validation,
      "semantic_gate_context":gate_context,"semantic_gate_output":gate_output,
      "semantic_gate_validation":gate_validation,
      "candidate_selection_audit":selection_audit,
      "realizer_input_isolation":isolation,"realizer_output":realized,
      "realizer_validation":normalized,"child_dialogue":child,"expanded_dialogue":expanded,
      "expanded_validation":expanded_validation,"post_overview":post_overview,
      "parent_add_completion":parent_completion,"parent_add_skipped":reuse_existing_namespace,
      "child_add_completion":child_completion,
      "memory_call_count":len(memorax.calls),"generation_ready":True,
      "memory_write_verified":bool((reuse_existing_namespace or parent_state.get("memory_write_verified")) and child_state.get("memory_write_verified")),
      "memory_visibility_verified":bool((reuse_existing_namespace or parent_state.get("bucket_visibility_verified")) and child_state.get("bucket_visibility_verified"))}
