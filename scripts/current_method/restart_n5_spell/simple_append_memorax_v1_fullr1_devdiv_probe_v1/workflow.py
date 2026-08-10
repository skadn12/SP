"""Single-round orchestration with optional Memorax context.

The stable ``simple_append_v1`` contracts remain authoritative.  This module
only sequences Memorax Add/Search around those contracts and is deliberately
transport-injectable so the complete path can be tested without a network.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping, Protocol, Sequence

from ..simple_append_v1.assembly import (
    assemble_expanded_dialogue,
    validate_expanded_dialogue,
)
from ..simple_append_v1.contracts import (
    compile_round_plan,
    normalize_round_realizer_output,
    validate_planner_output,
)
from .async_write import (
    DEFAULT_POLL_DELAYS_SECONDS,
    MemoraxAsyncWriteError,
    submit_add_without_wait,
    submit_add_and_wait,
)
from .client import MemoraxContractError, MemoraxTransportError
from .context import build_fresh_r1_context, select_fresh_r1_qa_context
from .memory_context import compile_memory_overview, normalize_bucket_snapshot
from .payloads import build_add_intent, build_bucket_search_payload
from .prompts import (
    build_memory_augmented_mcq_prompt,
    build_memory_augmented_planner_prompt,
    build_memory_augmented_realizer_prompt,
)


WORKFLOW_VERSION = "simple_append_memorax_fullr1_devdiv_probe_single_round_v1"


class MemoraxGateway(Protocol):
    calls: list[dict[str, Any]]

    def add(self, payload: Mapping[str, Any], *, request_id: str) -> dict[str, Any]:
        ...

    def search_buckets(
        self, payload: Mapping[str, Any], *, request_id: str,
    ) -> dict[str, Any]:
        ...

    def add_status(self, task_id: str, *, request_id: str) -> dict[str, Any]:
        ...


JsonRoleCall = Callable[[Mapping[str, Any]], Mapping[str, Any]]
ArtifactSink = Callable[[str, Mapping[str, Any]], None]
BeforeMemoryRequest = Callable[[str, str], None]


class MemoryWorkflowError(RuntimeError):
    """A stage-specific failure that never contains credentials."""

    def __init__(self, stage: str, errors: list[str] | None = None) -> None:
        self.stage = str(stage)
        self.errors = list(errors or [])
        suffix = ":" + ",".join(self.errors) if self.errors else ""
        super().__init__(f"memorax_workflow_failed:{self.stage}{suffix}")


def _emit(
    sink: ArtifactSink | None, name: str, value: Mapping[str, Any],
) -> None:
    if sink is not None:
        sink(str(name), value)


def _recent_text(turns: list[Mapping[str, Any]]) -> str:
    return "\n".join(str(row.get("text") or "") for row in turns)


def execute_single_round(
    *,
    parent: Mapping[str, Any],
    memorax: MemoraxGateway,
    planner_call: JsonRoleCall,
    realizer_call: JsonRoleCall,
    initialize_parent_memory: bool = True,
    session_count: int = 2,
    target_questions: int = 8,
    artifact_sink: ArtifactSink | None = None,
    before_memory_request: BeforeMemoryRequest | None = None,
    async_add: bool = True,
    wait_for_memory_completion: bool = True,
    poll_delays_seconds: Sequence[float] = DEFAULT_POLL_DELAYS_SECONDS,
    sleep_fn: Callable[[float], None] = time.sleep,
    raw_dialogue_token_counter: Callable[[str], int] | None = None,
) -> dict[str, Any]:
    """Execute one accepted expansion round and materialize its QA prompt.

    The Parent is added only when ``initialize_parent_memory`` is true.  The
    generated Child is added only after Planner, Realizer, and complete-dialogue
    assembly validation all pass.  No QA, Solver, or training call is made.
    """
    del session_count  # Fresh R1 selection is token-budgeted, not N-session based.
    token_counter = raw_dialogue_token_counter or (
        lambda text: max(0, (len(text.encode("utf-8")) + 3) // 4))
    tail = build_fresh_r1_context(parent, token_counter=token_counter)
    root_id = str(parent["root_id"])
    parent_turns = list(parent["dialogue"]["turns"])
    parent_sha = str(parent["dialogue"]["dialogue_hash"])
    _emit(artifact_sink, "tail_context", tail)

    parent_add_intent: dict[str, Any] | None = None
    parent_add_completion: dict[str, Any] | None = None
    child_add_completion: dict[str, Any] | None = None
    pending_memory_writes: list[dict[str, Any]] = []
    memory_write_states: list[dict[str, Any]] = []
    query = build_bucket_search_payload(root_id=root_id)

    if initialize_parent_memory:
        isolation_search_id = f"memorax-search-{root_id}-isolation-check"
        _emit(artifact_sink, "memory/isolation_bucket_query", query)
        if before_memory_request is not None:
            before_memory_request("isolation_search", isolation_search_id)
        isolation_response = memorax.search_buckets(
            query["payload"], request_id=isolation_search_id)
        isolation_snapshot = normalize_bucket_snapshot(
            isolation_response, root_id=root_id,
            user_id=query["payload"]["user_id"])
        _emit(
            artifact_sink, "memory/isolation_bucket_snapshot",
            isolation_snapshot)
        if sum(int(value) for value in isolation_snapshot["bucket_counts"].values()):
            raise MemoryWorkflowError(
                "memory_isolation", ["experiment_namespace_not_empty"])

    def submit_memory_intent(
        *, stage: str, artifact_prefix: str,
        intent: Mapping[str, Any],
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        latest_state: dict[str, Any] = {}

        def persist_state(value: Mapping[str, Any]) -> None:
            latest_state.clear()
            latest_state.update(dict(value))
            _emit(
                artifact_sink, f"memory/{artifact_prefix}_write_state",
                latest_state)

        try:
            if wait_for_memory_completion:
                completion = submit_add_and_wait(
                    gateway=memorax,
                    intent=intent,
                    before_status_request=before_memory_request,
                    poll_delays_seconds=poll_delays_seconds,
                    sleep_fn=sleep_fn,
                    stage=stage,
                    on_state_change=persist_state,
                )
            else:
                completion = submit_add_without_wait(
                    gateway=memorax,
                    intent=intent,
                    stage=stage,
                    on_state_change=persist_state,
                )
            if completion.get("pending"):
                pending = {
                    "stage": stage,
                    "write_intent_id": latest_state.get("write_intent_id"),
                    "attempt_id": latest_state.get("attempt_id"),
                    "request_id": latest_state.get("request_id"),
                    "task_id": latest_state.get("task_id"),
                    "intent_sha256": latest_state.get("payload_sha256"),
                    "status": latest_state.get("status"),
                    "error": None,
                    "authoritative": False,
                    "recovery_required": True,
                    "automatic_retry_allowed": False,
                }
                pending_memory_writes.append(pending)
                _emit(
                    artifact_sink, f"memory/{artifact_prefix}_pending", pending)
                return None, latest_state
            return dict(completion), latest_state
        except (MemoraxAsyncWriteError, MemoraxContractError,
                MemoraxTransportError) as error:
            pending = {
                "stage": stage,
                "write_intent_id": latest_state.get("write_intent_id"),
                "attempt_id": latest_state.get("attempt_id"),
                "request_id": (
                    latest_state.get("request_id") or intent.get("request_id")),
                "task_id": latest_state.get("task_id"),
                "intent_sha256": (
                    latest_state.get("payload_sha256")
                    or intent.get("payload_sha256")),
                "status": latest_state.get("status") or "ambiguous",
                "error": str(error),
                "authoritative": False,
                "recovery_required": True,
                "automatic_retry_allowed": False,
            }
            pending_memory_writes.append(pending)
            _emit(artifact_sink, f"memory/{artifact_prefix}_pending", pending)
            return None, latest_state
    if initialize_parent_memory:
        parent_add_intent = build_add_intent(
            root_id=root_id,
            round_index=0,
            source_dialogue_sha256=parent_sha,
            turns=parent_turns,
            async_mode=async_add,
        )
        _emit(artifact_sink, "memory/parent_add_intent", parent_add_intent)
        if before_memory_request is not None:
            before_memory_request("parent_add", parent_add_intent["request_id"])
        parent_add_completion, parent_write_state = submit_memory_intent(
            stage="parent_memory_add", artifact_prefix="parent_add",
            intent=parent_add_intent)
        if parent_write_state:
            memory_write_states.append(parent_write_state)
        _emit(
            artifact_sink,
            "memory/parent_add_completion",
            parent_add_completion,
        )

    _emit(artifact_sink, "memory/pre_generation_bucket_query", query)
    pre_search_id = f"memorax-search-{root_id}-pre-generation"
    if before_memory_request is not None:
        before_memory_request("pre_generation_search", pre_search_id)
    pre_response = memorax.search_buckets(
        query["payload"], request_id=pre_search_id)
    pre_snapshot = normalize_bucket_snapshot(
        pre_response, root_id=root_id, user_id=query["payload"]["user_id"])
    pre_overview = compile_memory_overview(
        pre_snapshot,
        recent_dialogue_text=_recent_text(list(tail["selected_turns"])),
    )
    _emit(artifact_sink, "memory/pre_generation_bucket_snapshot", pre_snapshot)
    _emit(artifact_sink, "memory/pre_generation_overview", pre_overview)

    planner_prompt = build_memory_augmented_planner_prompt(tail, pre_overview)
    _emit(artifact_sink, "planner/prompt", planner_prompt)
    planner_output = planner_call(planner_prompt)
    if not isinstance(planner_output, Mapping):
        raise MemoryWorkflowError("planner_parse", ["response_not_object"])
    planner_output = dict(planner_output)
    _emit(artifact_sink, "planner/output", planner_output)
    planner_validation = validate_planner_output(
        planner_output, parent_speakers=tail["parent_speakers"])
    _emit(artifact_sink, "planner/validation", planner_validation)
    if not planner_validation["valid"]:
        raise MemoryWorkflowError("planner_contract", planner_validation["errors"])
    compiled_plan = compile_round_plan(planner_output, tail_context=tail)
    _emit(artifact_sink, "planner/compiled_round_plan", compiled_plan)

    realizer_prompt = build_memory_augmented_realizer_prompt(
        compiled_plan=compiled_plan,
        tail_context=tail,
        memory_overview=pre_overview,
    )
    _emit(artifact_sink, "realizer/prompt", realizer_prompt)
    realizer_output = realizer_call(realizer_prompt)
    if not isinstance(realizer_output, Mapping):
        raise MemoryWorkflowError("realizer_parse", ["response_not_object"])
    realizer_output = dict(realizer_output)
    _emit(artifact_sink, "realizer/output", realizer_output)
    normalized = normalize_round_realizer_output(
        realizer_output,
        compiled_plan=compiled_plan,
        parent_utterances=[str(row["text"]) for row in parent_turns],
        parent_turn_refs=[str(row["turn_id"]) for row in parent_turns],
    )
    _emit(artifact_sink, "realizer/validation", normalized)
    if not normalized["valid"]:
        raise MemoryWorkflowError("realizer_contract", normalized["errors"])

    child = normalized["child"]
    expanded = assemble_expanded_dialogue(parent=parent, child=child)
    expanded_validation = validate_expanded_dialogue(expanded, parent=parent)
    _emit(artifact_sink, "child_dialogue", child)
    _emit(artifact_sink, "expanded_dialogue", expanded)
    _emit(artifact_sink, "expanded_dialogue_validation", expanded_validation)
    if not expanded_validation["valid"]:
        raise MemoryWorkflowError("parent_child_assembly", expanded_validation["errors"])

    child_turns = [
        row
        for scene in child["scenes"]
        for row in scene["dialogue_turns"]
    ]
    child_add_intent = build_add_intent(
        root_id=root_id,
        round_index=int(child["round_index"]),
        source_dialogue_sha256=str(expanded["complete_dialogue_sha256"]),
        turns=child_turns,
        recent_context_turns=list(tail["selected_turns"]),
        async_mode=async_add,
    )
    _emit(artifact_sink, "memory/child_add_intent", child_add_intent)
    if before_memory_request is not None:
        before_memory_request("child_add", child_add_intent["request_id"])
    child_add_completion, child_write_state = submit_memory_intent(
        stage="child_memory_add", artifact_prefix="child_add",
        intent=child_add_intent)
    if child_write_state:
        memory_write_states.append(child_write_state)
    _emit(
        artifact_sink,
        "memory/child_add_completion",
        child_add_completion,
    )

    _emit(artifact_sink, "memory/post_generation_bucket_query", query)
    post_search_id = f"memorax-search-{root_id}-post-generation"
    if before_memory_request is not None:
        before_memory_request("post_generation_search", post_search_id)
    post_response = memorax.search_buckets(
        query["payload"], request_id=post_search_id)
    post_snapshot = normalize_bucket_snapshot(
        post_response, root_id=root_id, user_id=query["payload"]["user_id"])
    post_overview = compile_memory_overview(
        post_snapshot, recent_dialogue_text=_recent_text(child_turns))
    _emit(artifact_sink, "memory/post_generation_bucket_snapshot", post_snapshot)
    _emit(artifact_sink, "memory/post_generation_overview", post_overview)

    qa_context = select_fresh_r1_qa_context(
        expanded, token_counter=token_counter)
    _emit(artifact_sink, "qa/context_view", qa_context)
    mcq_prompt = build_memory_augmented_mcq_prompt(
        qa_context,
        post_overview,
        target_questions=target_questions,
        focus_round_index=int(child["round_index"]),
    )
    _emit(artifact_sink, "qa/mcq_prompt", mcq_prompt)
    result = {
        "workflow_version": WORKFLOW_VERSION,
        "root_id": root_id,
        "round_index": int(child["round_index"]),
        "parent_memory_initialized": bool(initialize_parent_memory),
        "parent_add_intent": parent_add_intent,
        "selected_parent_context": tail,
        "pre_generation_memory_overview": pre_overview,
        "planner_validation": planner_validation,
        "compiled_plan": compiled_plan,
        "realizer_validation": normalized,
        "child_dialogue": child,
        "expanded_dialogue": expanded,
        "expanded_dialogue_validation": expanded_validation,
        "qa_context_view": qa_context,
        "child_add_intent": child_add_intent,
        "pending_memory_writes": pending_memory_writes,
        "memory_write_states": memory_write_states,
        "memory_write_verified": bool(memory_write_states) and all(
            row.get("memory_write_verified") is True
            for row in memory_write_states),
        "memory_visibility_verified": bool(memory_write_states) and all(
            row.get("bucket_visibility_verified") is True
            for row in memory_write_states),
        "memory_context_degraded": bool(pending_memory_writes),
        "post_generation_memory_overview": post_overview,
        "mcq_prompt": mcq_prompt,
        "memory_call_count": len(memorax.calls),
        "verified_parent_add": (
            parent_add_completion if initialize_parent_memory else None),
        "verified_child_add": child_add_completion,
        "generation_ready": True,
        "qa_prompt_ready": True,
        "qa_pipeline_ready": False,
        "memory_is_qa_fact_source": False,
    }
    _emit(artifact_sink, "workflow_result", result)
    return result
