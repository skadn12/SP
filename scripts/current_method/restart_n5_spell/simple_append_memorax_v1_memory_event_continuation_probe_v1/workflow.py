"""Memory-event continuation orchestration; no QA or state ledger."""

from __future__ import annotations

import time
from copy import deepcopy
import re
from typing import Any, Callable, Mapping, Protocol, Sequence
from . import development_policy as gain_policy

from ..simple_append_v1.assembly import (
    append_expansion_round,
    assemble_expanded_dialogue,
    validate_appended_round,
    validate_expanded_dialogue,
)
from .async_write import DEFAULT_POLL_DELAYS_SECONDS, MemoraxAsyncWriteError, submit_add_and_wait
from .client import MemoraxContractError, MemoraxTransportError
from .context import build_fresh_r1_context
from .incremental_round import build_incremental_context
from .contracts import (
    build_event_candidates,
    build_person_event_candidates,
    build_speaker_registry,
    build_semantic_gate_context,
    compile_planner_output,
    materialize_planner_output,
    materialize_planner_speaker_refs,
    materialize_verified_tracks,
    materialize_memory_only_tracks,
    normalize_realizer_output,
    validate_planner_outline_distinctness,
    validate_planner_outline_semantic_output,
    select_ranked_legal_candidate_set,
    select_planner_variant,
    validate_realizer_plan_isolation,
    validate_event_verifier_output,
    validate_event_binding_alignment,
    validate_planner_output,
    validate_planner_repair_output,
    validate_semantic_gate_output,
)
from .realizer_adherence import (
    build_adherence_auditor_input,
    build_adherence_auditor_prompt,
    validate_and_materialize_adherence_audit,
)
from .realizer_style import (
    _REALIZER_CONTINUITY_TAIL_MAX_TURNS,
    _REALIZER_CONTINUITY_TAIL_MAX_CHARS,
    _median_int,
    _build_realizer_style_context,
)
from .scene_continuation import (
    build_scene_continuation_prompt,
    build_scene_continuation_request,
    materialize_scene_continuation_output,
    merge_scene_continuations,
)
from .scene_replacement import (
    build_scene_replacement_prompt,
    build_scene_replacement_request,
    materialize_replacement_plan,
    materialize_scene_replacement_output,
    merge_scene_replacement,
)
from .promotion import build_promotable_child, validate_promotable_child
from .memory_context import (
    audit_bucket_delta_visibility,
    compile_memory_overview,
    compile_global_memory_state_view,
    normalize_bucket_snapshot,
)
from .payloads import build_add_intent, build_bucket_search_payload
from .prompts import (
    build_planner_prompt,
    build_planner_candidate_repair_prompt,
    build_event_verifier_prompt,
    build_realizer_prompt,
    build_planner_outline_semantic_prompt,
    build_semantic_continuation_gate_prompt,
)
from .write_state import apply_visibility_result
from .single_event_line import (
    build_gate_compatible_plan,
    build_single_event_line_prompt,
    materialize_single_event_line_plan,
    validate_materialized_single_event_line,
    validate_single_event_gate_plan,
    validate_single_event_line_plan,
)
from .profiles import ExpansionExecutionProfile

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


# These are cases where retrying the same candidate is not useful: the event
# identity/current state is not trustworthy, or the candidate has no usable
# continuation. The repair call receives a fresh slot instead of retrying it.
_NON_RETRYABLE_REPAIR_MARKERS = (
    "ambiguous", "unclear", "mixed", "closed", "no natural",
    "cannot identify", "cannot verify", "identity", "unknown event",
    "unbound", "wrong actor", "unsupported", "insufficient evidence",
)
_RETRYABLE_REPAIR_MARKERS = (
    "duplicate", "already realized", "already happened", "repeat",
    "restat", "weak transition", "future-only", "over-scoped",
    "too broad", "generic", "not concrete", "not scene-completable",
)


def _candidate_repair_targets(
        *, plan: Mapping[str, Any], gate_validation: Mapping[str, Any],
        required_count: int) -> list[dict[str, Any]]:
    """Build a bounded ordered repair plan without retrying dead event lines."""
    reviews = {
        str(row.get("candidate_ref") or ""): row
        for row in gate_validation.get("normalized_candidate_reviews") or []
        if isinstance(row, Mapping)
    }
    scenes_by_ref: dict[str, Mapping[str, Any]] = {}
    for scene in plan.get("candidate_pool") or plan.get("scenes") or []:
        if not isinstance(scene, Mapping):
            continue
        event_id = str(scene.get("memory_event_id") or "")
        for index, _candidate in enumerate(
                scene.get("next_step_candidates") or [], 1):
            scenes_by_ref[f"{event_id}:C{index:02d}"] = scene

    targets: list[dict[str, Any]] = []
    rejected_refs = [str(ref) for ref in
                     gate_validation.get("rejected_candidate_refs") or []]
    for ref in rejected_refs:
        if len(targets) >= int(required_count):
            break
        review = reviews.get(ref) or {}
        reason = str(review.get("reason") or "").strip()
        reason_lc = reason.casefold()
        scene = scenes_by_ref.get(ref) or {}
        development_mode = str(scene.get("development_mode") or "")
        non_retryable = any(marker in reason_lc
                            for marker in _NON_RETRYABLE_REPAIR_MARKERS)
        retryable = (
            str(review.get("parent_relation") or "") == "REALIZED"
            or any(marker in reason_lc
                   for marker in _RETRYABLE_REPAIR_MARKERS)
        )
        if non_retryable or not retryable:
            continue
        targets.append({
            "target_ref": ref,
            "target_kind": (
                "retry_same_memory_event"
                if development_mode == "continuation"
                else "replace_new_development"),
            "reason": reason or "candidate rejected by Semantic Gate",
        })

    # A same-event retry gets a same-call fallback. If the retry is rejected,
    # the fallback can fill the slot without a second Planner request. A
    # replacement/new-development target is already a fallback and does not
    # get another retry target.
    expanded_targets: list[dict[str, Any]] = []
    for target in targets:
        expanded_targets.append(target)
        if target.get("target_kind") in {
                "retry_same_memory_event", "replace_new_development"}:
            expanded_targets.append({
                "target_ref": f"{target.get('target_ref')}:fallback",
                "target_kind": "new_development_slot",
                "reason": "fallback_if_targeted_retry_is_rejected",
            })
    targets = expanded_targets

    # If there are not enough retryable event lines, explicitly request fresh
    # developments for the remaining scene slots. These are not retries of a
    # rejected candidate and are still ordered after retryable targets.
    while len(targets) < int(required_count):
        targets.append({
            "target_ref": f"NEW_SLOT_{len(targets) + 1:02d}",
            "target_kind": "new_development_slot",
            "reason": "fill_missing_scene_slot_after_non_retryable_rejection",
        })
    return targets

def _recent(rows):
    return "\n".join(str(row.get("text") or "") for row in rows)


def build_realizer_plan_batches(
        plan: Mapping[str, Any], *, batch_size: int,
        round_index: int = 1,
        batch_policy: str = "fixed_size") -> list[dict[str, Any]]:
    """Split a materialized plan into deterministic Realizer request batches.

    This is a pure scheduler helper.  Each batch retains the original scene
    IDs and ordering; callers must promote a batch before using it as context
    for a dependent later batch.
    """
    size = int(batch_size)
    if size < 1:
        raise ValueError("realizer_batch_size_invalid")
    if batch_policy not in {"fixed_size", "continuation_first"}:
        raise ValueError("realizer_batch_policy_invalid")
    scenes = list(plan.get("scenes") or [])
    if not scenes:
        return []
    batches: list[dict[str, Any]] = []
    scheduled_predecessor_scene_refs: list[str] = []
    # The expanded R03 route keeps the planner's continuation frontier
    # together.  Independent/new-development scenes retain the configured
    # batch size and are emitted in their original order.  We deliberately
    # operate on contiguous runs: no scene is moved across another scene, and
    # dependency order remains the planner's order.
    ranges: list[tuple[int, int]] = []
    if batch_policy == "continuation_first":
        start = 0
        while start < len(scenes):
            is_cont = (str(scenes[start].get("development_mode", "")) == "continuation"
                       or bool(scenes[start].get("dependencies")))
            end = start + 1
            if is_cont:
                while end < len(scenes):
                    nxt = scenes[end]
                    nxt_cont = (str(nxt.get("development_mode", "")) == "continuation"
                                or bool(nxt.get("dependencies")))
                    if not nxt_cont:
                        break
                    end += 1
                ranges.append((start, end))
            else:
                end = min(len(scenes), start + size)
                ranges.append((start, end))
            start = end
    else:
        ranges = [(start, min(len(scenes), start + size))
                  for start in range(0, len(scenes), size)]
    for start, end in ranges:
        batch = deepcopy(dict(plan))
        batch_scenes = deepcopy(scenes[start:end])
        for local_index, scene in enumerate(batch_scenes, start + 1):
            scene.setdefault(
                "scene_ref", f"R{int(round_index):02d}:S{local_index:02d}")
        batch["scenes"] = batch_scenes
        batch["batch_index"] = len(batches) + 1
        batch["batch_scene_start"] = start + 1
        batch["batch_scene_end"] = end
        batch["batch_policy"] = batch_policy
        # This is a static schedule only. Actual promoted refs are filled by
        # execute_round after each batch's local audit; a scene is never
        # considered promoted merely because it is earlier in the plan.
        batch["scheduled_predecessor_scene_refs"] = list(
            scheduled_predecessor_scene_refs)
        batch["prior_promoted_scene_refs"] = []
        batches.append(batch)
        scheduled_predecessor_scene_refs.extend(
            str(scene.get("scene_ref")) for scene in batch_scenes
            if scene.get("scene_ref"))
    return batches

def execute_round(*, parent: Mapping[str, Any], memorax: Gateway,
                  verifier_call: Callable, planner_call: Callable,
                  planner_repair_call: Callable | None = None,
                  semantic_gate_call: Callable,
                  realizer_call: Callable, adherence_call: Callable,
                  scene_continuation_call: Callable | None = None,
                  scene_replacement_call: Callable | None = None,
                  adherence_reaudit_call: Callable | None = None,
                  replacement_adherence_call: Callable | None = None,
                  artifact_sink=None, before_memory_request=None,
                  token_counter=None, wait_for_memory_completion=True,
                  reuse_existing_namespace: bool = False,
                  memorax_user_namespace: str | None = None,
                  assume_new_namespace_empty: bool = False,
                  memory_only_fast: bool = False,
                  event_bucket_only: bool = False,
                  round_index: int = 1,
                  prior_expanded: Mapping[str, Any] | None = None,
                  expose_complete_memory_to_planner: bool = False,
                  skip_planner_semantic_gate: bool = False,
                  realizer_style_tail_only: bool = False,
                  single_event_line_mode: bool = False,
                  single_event_target_rank: int = 1,
                  no_memorax_writes: bool = False,
                  outline_semantic_call: Callable | None = None,
                  feedback_record: Mapping[str, Any] | None = None,
                  feedback_dialogue: Mapping[str, Any] | None = None,
                  feedback_view_version: str | None = None,
                  execution_profile: ExpansionExecutionProfile | None = None,
                  max_continuations: int = 3,
                  max_new_development_candidates: int = 4,
                  max_plan_variants: int = 1,
                  scene_quota: int = 3,
                  new_development_slots: int | None = None,
                  realizer_batch_size: int = 3,
                  require_full_scene_quota: bool = True,
                  poll_delays_seconds=DEFAULT_POLL_DELAYS_SECONDS,
                  sleep_fn=time.sleep) -> dict[str, Any]:
    sequential_batch_promotion = False
    realizer_batch_policy = "fixed_size"
    development_policy = getattr(execution_profile, "development_policy", None)
    if execution_profile is not None:
        memory_only_fast = bool(execution_profile.memory_only_fast)
        event_bucket_only = bool(execution_profile.event_bucket_only)
        # The explicit profile owns the Realizer input boundary. This avoids
        # widening an expanded text-free route through the legacy boolean
        # default on the first round.
        realizer_style_tail_only = (
            execution_profile.realizer_input_mode == "text_free")
        max_continuations = int(execution_profile.max_continuations)
        max_new_development_candidates = int(
            execution_profile.max_new_development_candidates)
        max_plan_variants = int(execution_profile.max_plan_variants)
        scene_quota = int(execution_profile.total_scene_limit)
        new_development_slots = execution_profile.new_development_slots
        realizer_batch_size = int(execution_profile.realizer_batch_size)
        realizer_batch_policy = execution_profile.realizer_batch_policy
        sequential_batch_promotion = bool(
            execution_profile.sequential_batch_promotion)
        require_full_scene_quota = bool(execution_profile.require_full_scene_quota)
        feedback_view_version = execution_profile.feedback_view_version
    if development_policy and (skip_planner_semantic_gate or single_event_line_mode or
            outline_semantic_call is None):
        raise ValueError("development_gain_requires_existing_gate_and_outline_roles")
    if feedback_view_version and feedback_view_version != "h4_round1_r03_feedback_view_v2":
        raise ValueError("unsupported_feedback_view_version")
    if feedback_view_version and (feedback_record is None or feedback_dialogue is None):
        raise ValueError("feedback_view_requires_record_and_dialogue")
    # Legacy requires its fixed scene count.  Expanded profiles use the same
    # configured count as an upper bound and must not repair/generate merely
    # to fill unused slots.
    minimum_scene_count = int(scene_quota) if require_full_scene_quota else 1
    token_counter = token_counter or (lambda text: max(0, (len(text.encode())+3)//4))
    root_id = str(parent["root_id"])
    if int(round_index) < 1:
        raise ValueError("memory_event_round_index_invalid")
    if int(round_index) > 1 and not isinstance(prior_expanded, Mapping):
        raise ValueError("memory_event_prior_expanded_required")
    if single_event_line_mode and not memory_only_fast:
        raise ValueError("single_event_line_requires_memory_only_fast")
    tail = (build_fresh_r1_context(
        parent, token_counter=token_counter, budget_tokens=10000)
        if int(round_index) == 1 else build_incremental_context(
            parent, token_counter=token_counter, budget_tokens=10000))
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
    parent_visibility_audit: dict[str, Any] | None = None
    isolation_snapshot: dict[str, Any] | None = None
    if reuse_existing_namespace:
        # Recovery is explicitly read-first: the already written Parent is
        # reused and must never be submitted a second time.
        pre_snapshot = search(
            artifact_name="recovery_pre_generation",
            logical_role="pre_generation_search",
            request_id=f"memory-event-r{int(round_index):02d}-pre-{root_id}")
        _emit(artifact_sink, "memory/parent_add_skipped", {
            "reason": "existing_namespace_reused",
            "parent_add_calls": 0,
        })
    else:
        if not assume_new_namespace_empty:
            isolation_snapshot = search(
                artifact_name="isolation", logical_role="isolation_search",
                request_id=f"memory-event-isolation-{root_id}")
            if sum(isolation_snapshot["bucket_counts"].values()):
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
        if isolation_snapshot is not None:
            parent_visibility_audit = audit_bucket_delta_visibility(
                isolation_snapshot, pre_snapshot)
            parent_state = apply_visibility_result(
                parent_state,
                visible=bool(parent_visibility_audit["bucket_visibility_verified"]),
            )
            _emit(artifact_sink, "memory/parent_visibility_audit",
                  parent_visibility_audit)
            _emit(artifact_sink, "memory/parent_add_write_state", parent_state)
    pre_overview=compile_memory_overview(pre_snapshot, recent_dialogue_text=_recent(parent_turns))
    _emit(artifact_sink,"memory/pre_generation_overview",pre_overview)
    global_memory_state_view = compile_global_memory_state_view(pre_snapshot)
    _emit(artifact_sink, "memory/global_state_view", global_memory_state_view)
    memory_source_mode = "verified_tracks" if not memory_only_fast else "person_bucket"
    if memory_only_fast and event_bucket_only:
        candidates = build_event_candidates(
            pre_overview, person_events_only=True)
        memory_source_mode = "event_bucket_only"
    elif memory_only_fast:
        candidates = build_person_event_candidates(pre_overview)
        if not candidates:
            # A broad person [event @recent] roll-up is not a selectable event.
            # Use concrete event-bucket keys as a deterministic fallback rather
            # than inventing a person-to-event binding from that roll-up.
            candidates = build_event_candidates(
                pre_overview, person_events_only=True)
            memory_source_mode = "event_bucket_fallback"
    else:
        candidates = build_event_candidates(pre_overview)
    _emit(artifact_sink, "memory/event_candidates", {
        "candidates": candidates, "source_mode": memory_source_mode})

    verifier_raw = None
    if memory_only_fast:
        verifier_validation = {
            "valid": True, "mode": "memory_only_fast",
            "verified_track_count": 0, "errors": [],
        }
        verified_tracks = materialize_memory_only_tracks(
            candidates, event_bucket_only=event_bucket_only)
        _emit(artifact_sink, "event_verifier/skipped", {
            "reason": "memory_only_fast", "calls": 0,
        })
    else:
        verifier_prompt = build_event_verifier_prompt(parent=parent, candidates=candidates)
        _emit(artifact_sink, "event_verifier/prompt", verifier_prompt)
        verifier_raw = verifier_call(verifier_prompt)
        _emit(artifact_sink, "event_verifier/output", verifier_raw)
        verifier_validation = validate_event_verifier_output(
            verifier_raw, candidates=candidates, parent=parent)
        _emit(artifact_sink, "event_verifier/validation", verifier_validation)
        if not verifier_validation["valid"]:
            raise WorkflowError("event_verifier_contract", verifier_validation["errors"])
        verified_tracks = materialize_verified_tracks(
            verifier_raw, candidates=candidates, parent=parent)
    _emit(artifact_sink, "event_verifier/materialized_tracks", {
        "verified_tracks": verified_tracks,
        "event_bucket_only": bool(event_bucket_only)})
    if not verified_tracks:
        _emit(artifact_sink, "memory/event_discovery_validation", {
            "valid": False,
            "errors": ["no_concrete_memory_candidates"],
            "source_mode": memory_source_mode,
            "candidate_count": len(candidates),
        })
        raise WorkflowError("memory_event_discovery",
                            ["no_concrete_memory_candidates"])
    _emit(artifact_sink, "memory/event_discovery_validation", {
        "valid": True,
        "errors": [],
        "source_mode": memory_source_mode,
        "candidate_count": len(candidates),
        "verified_track_count": len(verified_tracks),
    })

    speaker_registry = build_speaker_registry(parent)
    _emit(artifact_sink, "planner/speaker_registry", speaker_registry)
    single_event_target = None
    if single_event_line_mode:
        target_index = int(single_event_target_rank) - 1
        if target_index < 0 or target_index >= len(verified_tracks):
            raise WorkflowError("single_event_target_selection", [
                f"target_rank_out_of_range:{single_event_target_rank}/"
                f"{len(verified_tracks)}"])
        single_event_target = deepcopy(verified_tracks[target_index])
        _emit(artifact_sink, "planner/single_event_target", {
            "selection_policy": "deterministic_verified_track_rank",
            "target_rank": int(single_event_target_rank),
            "target": single_event_target,
        })
        planner_prompt = build_single_event_line_prompt(
            target_track=single_event_target,
            speaker_registry=speaker_registry)
    else:
        if feedback_view_version:
            # The versioned role renderer owns the shared feedback view and
            # appends it to the existing Planner builder output.  It does not
            # change the Planner JSON contract or candidate selection rules.
            from ..h4_round1_r03_feedback_view_v2.renderers import (
                render_expansion_planner,
            )
            planner_prompt = render_expansion_planner(
                record=feedback_record, dialogue=feedback_dialogue,
                memory_overview=pre_overview, memory_candidates=candidates,
                verified_tracks=verified_tracks,
                memory_only_fast=memory_only_fast,
                speaker_registry=speaker_registry,
                strict_speaker_refs=memory_only_fast,
                expose_complete_memory=expose_complete_memory_to_planner,
                planner_max_continuations=max_continuations,
                planner_max_new_candidates=max_new_development_candidates,
                planner_scene_quota=scene_quota,
                planner_max_plan_variants=max_plan_variants,
                planner_max_tokens=getattr(
                    execution_profile, "planner_max_tokens", 4096),
            )
        else:
            planner_prompt = build_planner_prompt(
                parent=parent, memory_overview=pre_overview, candidates=candidates,
                verified_tracks=verified_tracks, memory_only_fast=memory_only_fast,
                event_bucket_only=event_bucket_only,
                memory_source_mode=memory_source_mode,
                speaker_registry=speaker_registry, strict_speaker_refs=memory_only_fast,
                expose_complete_memory=expose_complete_memory_to_planner,
                include_authoritative_transcript=True,
                max_continuations=max_continuations,
                max_new_development_candidates=max_new_development_candidates,
                max_plan_variants=max_plan_variants,
                scene_quota=scene_quota)
    if development_policy:
        planner_prompt = gain_policy.planner_prompt(planner_prompt)
    _emit(artifact_sink,"planner/prompt",planner_prompt)
    planner_raw=planner_call(planner_prompt)
    _emit(artifact_sink,"planner/output",planner_raw)
    # Every Planner route uses the shared canonical speaker materializer.  The
    # ordered single-event route is only a different scene compiler; it is not
    # allowed to bypass speaker_ref/new_people validation.
    speaker_materialization = (
        materialize_planner_speaker_refs(
            planner_raw, parent=parent, speaker_registry=speaker_registry)
        if (memory_only_fast or single_event_line_mode) else
        {"valid": True, "errors": [], "value": planner_raw,
         "speaker_registry": speaker_registry})
    # New-development candidates are a ranked pool. Quarantine a malformed
    # candidate so the local quota compiler can consider the next one; errors
    # in selected continuations remain fatal.
    if (not single_event_line_mode and
            not speaker_materialization["valid"] and memory_only_fast):
        candidate_errors = [str(error) for error in
                            speaker_materialization.get("errors") or []
                            if str(error).startswith(
                                "new_development_candidates_")]
        fatal_errors = [str(error) for error in
                        speaker_materialization.get("errors") or []
                        if not str(error).startswith(
                            "new_development_candidates_")]
        bad_indices = sorted({int(match.group(1)) - 1 for error in
                              candidate_errors
                              if (match := re.match(
                                  r"new_development_candidates_(\d+)_",
                                  error))})
        raw_candidates = list((planner_raw.get(
            "new_development_candidates") or [])
                              if isinstance(planner_raw, Mapping) else [])
        if candidate_errors and not fatal_errors and bad_indices:
            filtered_raw = deepcopy(dict(planner_raw))
            filtered_raw["new_development_candidates"] = [
                item for index, item in enumerate(raw_candidates)
                if index not in bad_indices]
            rerun = materialize_planner_speaker_refs(
                filtered_raw, parent=parent, speaker_registry=speaker_registry)
            if rerun["valid"]:
                rerun["quarantined_new_candidate_indices"] = [
                    index + 1 for index in bad_indices]
                rerun["quarantined_new_candidate_errors"] = candidate_errors
                speaker_materialization = rerun

    _emit(artifact_sink, "planner/speaker_materialization", speaker_materialization)
    if not speaker_materialization["valid"]:
        raise WorkflowError("planner_speaker_contract", speaker_materialization["errors"])
    planner_variant_selection = select_planner_variant(
        speaker_materialization["value"], max_plan_variants=max_plan_variants)
    _emit(artifact_sink, "planner/variant_selection", planner_variant_selection)
    if not planner_variant_selection["valid"]:
        raise WorkflowError("planner_variant_selection",
                            planner_variant_selection["errors"])
    planner_contract = planner_variant_selection["value"]
    if single_event_line_mode:
        for step in planner_contract.get("steps") or []:
            if not isinstance(step, Mapping):
                continue
            # The shared materializer historically emits `participants`.
            # For the ordered-chain route the semantic contract is explicitly
            # required coverage; rename it before the route-specific validator.
            if "participants" in step and "required_participants" not in step:
                step["required_participants"] = step.pop("participants")
    validation = (validate_single_event_line_plan(
        planner_contract, target_track=single_event_target or {},
        parent_speakers=[str(row.get("speaker") or "") for row in parent_turns])
        if single_event_line_mode else validate_planner_output(
            planner_contract, candidates=candidates, parent=parent,
            verified_tracks=verified_tracks,
            max_new_candidates=max_new_development_candidates,
            max_continuations=max_continuations,
            scene_quota=scene_quota,
            new_development_slots=new_development_slots,
            require_full_scene_quota=require_full_scene_quota))
    _emit(artifact_sink,"planner/validation",validation)
    if not validation["valid"]: raise WorkflowError("planner_continuation",validation["errors"])
    if single_event_line_mode:
        ordered_plan = materialize_single_event_line_plan(
            planner_contract, target_track=single_event_target or {},
            round_index=round_index,
            parent_speakers=[str(row.get("speaker") or "")
                             for row in parent_turns])
        ordered_validation = validate_materialized_single_event_line(
            ordered_plan, round_index=round_index)
        _emit(artifact_sink, "planner/single_event_ordered_plan", ordered_plan)
        _emit(artifact_sink, "planner/single_event_ordered_validation",
              ordered_validation)
        if not ordered_validation["valid"]:
            raise WorkflowError("single_event_ordered_compile",
                                ordered_validation["errors"])
        planner_compiled = build_gate_compatible_plan(ordered_plan)
        gate_plan_validation = validate_single_event_gate_plan(
            planner_compiled, round_index=round_index)
        _emit(artifact_sink, "planner/single_event_gate_plan_validation",
              gate_plan_validation)
        if not gate_plan_validation["valid"]:
            raise WorkflowError("single_event_gate_compile",
                                gate_plan_validation["errors"])
    else:
        planner_compiled = compile_planner_output(
            planner_contract, candidates=candidates,
            verified_tracks=verified_tracks,
            selected_new_candidate_indices=validation.get(
                "selected_new_candidate_indices"),
            valid_new_candidate_indices=validation.get(
                "valid_new_candidate_indices"),
            scene_quota=scene_quota,
            new_development_slots=new_development_slots)
    _emit(artifact_sink, "planner/compiled_plan", planner_compiled)
    event_binding_alignment = validate_event_binding_alignment(
        planner_compiled, candidates=candidates)
    _emit(artifact_sink, "planner/event_binding_alignment",
          event_binding_alignment)
    quota_audit = dict(planner_compiled.get("scene_quota_audit") or {})
    _emit(artifact_sink, "planner/scene_quota_audit", quota_audit)
    outline_audit = validate_planner_outline_distinctness(planner_compiled)
    _emit(artifact_sink, "planner/outline_duplicate_audit", outline_audit)
    if not outline_audit["valid"] and not development_policy:
        raise WorkflowError("planner_outline_duplicate", outline_audit["errors"])

    # Exact normalized duplicates are handled locally.  Lexical similarity is
    # retained only as diagnostic data; the semantic decision is delegated to
    # one compact model call after any bounded Planner repair is complete.
    outline_semantic_prompt = None
    outline_semantic_raw = None
    outline_semantic_audit = None

    gate_context = build_semantic_gate_context(
        parent=parent, candidates=candidates, plan=planner_compiled,
        memory_only_fast=memory_only_fast,
        global_memory_state_view=global_memory_state_view,
        minimum_pass_candidates=minimum_scene_count)
    gate_prompt = build_semantic_continuation_gate_prompt(
        gate_context=gate_context)
    if development_policy:
        gate_prompt = gain_policy.gate_prompt(gate_context)
    _emit(artifact_sink, "semantic_gate/context", gate_context)
    _emit(artifact_sink, "semantic_gate/prompt", gate_prompt)
    if skip_planner_semantic_gate:
        # Experimental fast path: preserve the Gate implementation and its
        # artifacts, but do not make an LLM request before Realizer. The
        # deterministic materializer still chooses the first candidate per
        # scene, and the post-Realizer adherence/promotion audit remains the
        # write protection layer.
        all_candidate_refs = [
            f"{scene.get('memory_event_id')}:C{candidate_index:02d}"
            for scene in (planner_compiled.get("candidate_pool") or
                          planner_compiled.get("scenes") or [])
            if isinstance(scene, Mapping)
            for candidate_index, candidate in enumerate(
                scene.get("next_step_candidates") or [], 1)
            if isinstance(candidate, Mapping)]
        synthetic_reviews = [{
            "candidate_ref": ref,
            "decision": "pass",
            "parent_relation": "SKIPPED",
            "reason": "planner_pre_gate_skipped",
            "evidence_turn_refs": [],
        } for ref in all_candidate_refs]
        gate_output = {
            "gate_skipped": True,
            "skip_reason": "planner_pre_gate_experiment",
            "candidate_reviews": synthetic_reviews,
            "overall_decision": "accept",
        }
        gate_validation = {
            "valid": True,
            "errors": [],
            "overall_decision": "accept",
            "passed_candidate_refs": all_candidate_refs,
            "passed_candidate_count": len(all_candidate_refs),
            "minimum_pass_candidates": minimum_scene_count,
            "normalized_candidate_reviews": synthetic_reviews,
            "gate_skipped": True,
        }
        _emit(artifact_sink, "semantic_gate/skipped", {
            "skipped": True,
            "reason": "planner_pre_gate_experiment",
            "candidate_ref_count": len(all_candidate_refs),
        })
    else:
        gate_output = semantic_gate_call(gate_prompt)
        _emit(artifact_sink, "semantic_gate/output", gate_output)
        gate_validation = (gain_policy.validate_gate(
            gate_output, plan=planner_compiled,
            memory_view=global_memory_state_view, minimum=minimum_scene_count)
            if development_policy else validate_semantic_gate_output(
            gate_output, plan=planner_compiled, parent=parent,
            memory_only_fast=memory_only_fast,
            global_memory_state_view=global_memory_state_view,
            minimum_pass_candidates=minimum_scene_count))
        _emit(artifact_sink, "semantic_gate/validation", gate_validation)
        if not gate_validation["valid"]:
            raise WorkflowError("semantic_continuation_gate_contract",
                                gate_validation["errors"])

    # A rejected candidate is never rewritten in place.  When the memory-only
    # route has too few passing candidates, one bounded Planner repair creates
    # ordered replacements. The unchanged Gate then evaluates the replacement
    # pool. Older callers without a repair callback retain the original
    # fail-fast behavior.
    # Gate-pass candidates are frozen before any repair. A replacement may
    # only fill missing scene slots; it can never displace an already passed
    # scene later in materialization.
    frozen_gate_passed_candidate_refs = list(
        gate_validation.get("passed_candidate_refs") or [])
    frozen_gate_reviews_by_ref = {
        str(row.get("candidate_ref") or ""): dict(row)
        for row in gate_validation.get("normalized_candidate_reviews") or []
        if isinstance(row, Mapping)
    }
    planner_repair_raw = None
    planner_repair_validation = None
    planner_repair_merged_validation = None
    planner_repair_prompt = None
    if (not single_event_line_mode and not skip_planner_semantic_gate and
            int(round_index) == 1 and memory_only_fast
            and planner_repair_call is not None
            and gate_validation.get("overall_decision") == "reject"
            and require_full_scene_quota
            and int(gate_validation.get("passed_candidate_count") or 0)
            < int(gate_validation.get("minimum_pass_candidates") or 3)):
        repair_needed = max(
            0,
            int(gate_validation.get("minimum_pass_candidates") or 3)
            - len(frozen_gate_passed_candidate_refs),
        )
        repair_targets = _candidate_repair_targets(
            plan=planner_compiled,
            gate_validation=gate_validation,
            required_count=repair_needed,
        )
        planner_repair_prompt = build_planner_candidate_repair_prompt(
            candidates=candidates, verified_tracks=verified_tracks,
            plan=planner_compiled, gate_validation=gate_validation,
            repair_targets=repair_targets,
            speaker_registry=speaker_registry)
        _emit(artifact_sink, "planner/repair_prompt", planner_repair_prompt)
        planner_repair_raw = planner_repair_call(planner_repair_prompt)
        _emit(artifact_sink, "planner/repair_output", planner_repair_raw)
        repair_materialization = materialize_planner_speaker_refs(
            planner_repair_raw, parent=parent,
            speaker_registry=speaker_registry)
        _emit(artifact_sink, "planner/repair_speaker_materialization",
              repair_materialization)
        if not repair_materialization["valid"]:
            raise WorkflowError("planner_candidate_repair",
                                repair_materialization["errors"])
        repair_contract = repair_materialization["value"]
        planner_repair_validation = validate_planner_repair_output(
            repair_contract, parent=parent,
            required_count=int(planner_repair_prompt.get(
                "needed_candidate_count") or 0))
        _emit(artifact_sink, "planner/repair_validation",
              planner_repair_validation)
        if not planner_repair_validation["valid"]:
            raise WorkflowError("planner_candidate_repair",
                                planner_repair_validation["errors"])
        merged_contract = deepcopy(dict(planner_contract))
        merged_candidates = list(
            merged_contract.get("new_development_candidates") or [])
        merged_candidates.extend(list(
            repair_contract.get("new_development_candidates") or []))
        merged_contract["new_development_candidates"] = merged_candidates
        merged_speaker_materialization = {
            "valid": True, "errors": [], "value": merged_contract,
            "speaker_registry": speaker_registry,
        }
        _emit(artifact_sink, "planner/repair_merged_speaker_materialization",
              merged_speaker_materialization)
        planner_repair_merged_validation = validate_planner_output(
            merged_contract, candidates=candidates, parent=parent,
            # The initial pool allows four new candidates. One bounded repair
            # may add a same-call fallback for each retryable continuation.
            verified_tracks=verified_tracks,
            max_new_candidates=max(10, max_new_development_candidates),
            max_continuations=max_continuations,
            scene_quota=scene_quota,
            new_development_slots=new_development_slots,
            require_full_scene_quota=require_full_scene_quota)
        _emit(artifact_sink, "planner/repair_merged_validation",
              planner_repair_merged_validation)
        if not planner_repair_merged_validation["valid"]:
            raise WorkflowError("planner_candidate_repair",
                                planner_repair_merged_validation["errors"])
        planner_contract = merged_contract
        validation = planner_repair_merged_validation
        planner_compiled = compile_planner_output(
            planner_contract, candidates=candidates,
            verified_tracks=verified_tracks,
            selected_new_candidate_indices=validation.get(
                "selected_new_candidate_indices"),
            valid_new_candidate_indices=validation.get(
                "valid_new_candidate_indices"),
            scene_quota=scene_quota,
            new_development_slots=new_development_slots)
        _emit(artifact_sink, "planner/compiled_plan_after_repair",
              planner_compiled)
        gate_context = build_semantic_gate_context(
            parent=parent, candidates=candidates, plan=planner_compiled,
            memory_only_fast=memory_only_fast,
            global_memory_state_view=global_memory_state_view,
            minimum_pass_candidates=minimum_scene_count)
        gate_prompt = build_semantic_continuation_gate_prompt(
            gate_context=gate_context)
        _emit(artifact_sink, "semantic_gate/repair_context", gate_context)
        _emit(artifact_sink, "semantic_gate/repair_prompt", gate_prompt)
        gate_output = semantic_gate_call(gate_prompt)
        _emit(artifact_sink, "semantic_gate/repair_output", gate_output)
        repaired_gate_validation = validate_semantic_gate_output(
            gate_output, plan=planner_compiled, parent=parent,
            memory_only_fast=memory_only_fast,
            global_memory_state_view=global_memory_state_view,
            minimum_pass_candidates=minimum_scene_count)
        _emit(artifact_sink, "semantic_gate/repair_validation",
              repaired_gate_validation)
        if not repaired_gate_validation["valid"]:
            raise WorkflowError("semantic_continuation_gate_contract",
                                repaired_gate_validation["errors"])

        # Preserve the initial Gate decisions. The second Gate call can audit
        # the merged pool, but it cannot revoke a previously passed scene or
        # make it lose its place to a replacement.
        repaired_refs = set(
            str(ref) for ref in
            repaired_gate_validation.get("passed_candidate_refs") or []) - set(
                frozen_gate_passed_candidate_refs)
        merged_passed = list(frozen_gate_passed_candidate_refs)
        merged_passed.extend(
            ref for ref in repaired_gate_validation.get(
                "passed_candidate_refs") or []
            if str(ref) in repaired_refs)
        gate_validation = dict(repaired_gate_validation)
        gate_validation["frozen_passed_candidate_refs"] = list(
            frozen_gate_passed_candidate_refs)
        gate_validation["passed_candidate_refs"] = merged_passed
        gate_validation["passed_candidate_count"] = len(merged_passed)
        gate_validation["overall_decision"] = (
            "pass" if len(merged_passed) >= int(
            gate_validation.get("minimum_pass_candidates") or minimum_scene_count)
            else "reject")
        gate_validation["realizer_allowed"] = (
            gate_validation["valid"] and
            gate_validation["overall_decision"] == "pass")
        merged_reviews = []
        for row in repaired_gate_validation.get(
                "normalized_candidate_reviews") or []:
            ref = str(row.get("candidate_ref") or "")
            merged_reviews.append(deepcopy(
                frozen_gate_reviews_by_ref.get(ref, row)))
        gate_validation["normalized_candidate_reviews"] = merged_reviews
        _emit(artifact_sink, "planner/targeted_repair_selection", {
            "attempt": 1,
            "repair_targets": repair_targets,
            "frozen_passed_candidate_refs": frozen_gate_passed_candidate_refs,
            "replacement_passed_candidate_refs": [
                str(ref) for ref in merged_passed
                if str(ref) not in set(frozen_gate_passed_candidate_refs)],
            "final_passed_candidate_refs": merged_passed,
            "retry_limit": 1,
        })

    # Review the final compiled scene set once, after any Planner repair has
    # been merged and re-gated. The prompt contains only these scene outlines.
    outline_filtered_candidate_refs: list[str] = []
    effective_passed_candidate_refs = list(
        gate_validation.get("passed_candidate_refs") or [])
    if outline_semantic_call is not None and not single_event_line_mode:
        outline_semantic_prompt = build_planner_outline_semantic_prompt(
            plan=planner_compiled)
        if development_policy:
            outline_semantic_prompt = gain_policy.outline_prompt(
                plan=planner_compiled, memory_view=global_memory_state_view,
                gate_validation=gate_validation)
        _emit(artifact_sink, "planner/outline_semantic_prompt",
              outline_semantic_prompt)
        if (outline_semantic_prompt.get("pair_count", 0) or
                development_policy and outline_semantic_prompt.get("candidate_count", 0)):
            outline_semantic_raw = outline_semantic_call(
                outline_semantic_prompt)
            _emit(artifact_sink, "planner/outline_semantic_output",
                  outline_semantic_raw)
            outline_semantic_audit = (gain_policy.validate_outline(
                outline_semantic_raw, plan=planner_compiled,
                memory_view=global_memory_state_view) if development_policy else
                validate_planner_outline_semantic_output(
                    outline_semantic_raw, plan=planner_compiled))
            _emit(artifact_sink, "planner/outline_semantic_validation",
                  outline_semantic_audit)
            if not outline_semantic_audit["valid"]:
                raise WorkflowError(
                    "planner_outline_semantic_contract",
                    outline_semantic_audit["errors"])
            # Duplicate/uncertain (including missing) pair reviews are
            # conflict edges, not instructions to rewrite the Planner.  Pick
            # the highest-ranked legal three-candidate set; an unreviewed
            # pair can therefore never silently pass as distinct.
            legal_selection = (gain_policy.select_candidates(
                effective_passed_candidate_refs, audit=outline_semantic_audit,
                limit=scene_quota) if development_policy else select_ranked_legal_candidate_set(
                effective_passed_candidate_refs,
                outline_audit=outline_semantic_audit,
                required_count=scene_quota,
                allow_partial=not require_full_scene_quota))
            if planner_repair_raw is not None:
                missing_frozen = [
                    ref for ref in frozen_gate_passed_candidate_refs
                    if ref not in set(
                        legal_selection.get("selected_candidate_refs") or [])
                ]
                if missing_frozen:
                    raise WorkflowError(
                        "planner_outline_semantic",
                        ["frozen_pass_candidate_conflict:" + ref
                         for ref in missing_frozen])
            _emit(artifact_sink, "planner/outline_semantic_selection_audit",
                  legal_selection)
            outline_filtered_candidate_refs = list(
                legal_selection.get("blocked_candidate_refs") or [])
            effective_passed_candidate_refs = list(
                legal_selection.get("selected_candidate_refs") or [])
            if not legal_selection["valid"]:
                raise WorkflowError(
                    "planner_outline_semantic",
                    ["insufficient_legal_scene_set:" + str(
                        len(effective_passed_candidate_refs)) + "/" + str(
                        scene_quota)] +
                    list(outline_semantic_audit.get("errors") or []))
            _emit(artifact_sink, "planner/outline_semantic_selection", {
                "blocked_candidate_refs": outline_filtered_candidate_refs,
                "gate_passed_candidate_refs": list(
                    gate_validation.get("passed_candidate_refs") or []),
                "effective_passed_candidate_refs":
                    effective_passed_candidate_refs,
            })
        else:
            outline_semantic_audit = {
                "valid": True, "overall_decision": "pass",
                "expected_pair_refs": [], "returned_pair_refs": [],
                "pair_reviews": [], "errors": [],
            }
            _emit(artifact_sink, "planner/outline_semantic_validation",
                  outline_semantic_audit)
    else:
        outline_semantic_audit = {
            "valid": True, "overall_decision": "skipped",
            "reason": ("ordered_single_event_chain"
                       if single_event_line_mode else
                       "outline_semantic_call_not_configured"),
            "expected_pair_refs": [], "returned_pair_refs": [],
            "pair_reviews": [], "errors": [],
        }
        _emit(artifact_sink, "planner/outline_semantic_validation",
              outline_semantic_audit)
    reviews_by_ref = {
        str(review.get("candidate_ref")): dict(review)
        for review in gate_validation.get("normalized_candidate_reviews") or []
        if isinstance(review, Mapping)}
    passed_refs = set(effective_passed_candidate_refs)
    selection_audit = {"events": []}
    for scene in (planner_compiled.get("candidate_pool") or
                  planner_compiled.get("scenes") or []):
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
            "memory_event_ref": scene.get("memory_event_ref"),
            "memory_event_key": scene.get("memory_event_key"),
            "development_mode": scene.get("development_mode"),
            "source_basis": scene.get("source_basis"),
            "new_people": scene.get("new_people"),
            "candidates": candidate_rows,
            "selected_candidate_ref": selected_ref,
            "before_state": (selected_candidate or {}).get("before_state"),
            "required_transition": (selected_candidate or {}).get(
                "required_transition"),
            "after_state": (selected_candidate or {}).get("after_state"),
            "new_development": (selected_candidate or {}).get(
                "required_transition"),
            "durable_outcome": (selected_candidate or {}).get("after_state"),
            "actor_bindings": (selected_candidate or {}).get("actor_bindings"),
            "interaction_beats": (selected_candidate or {}).get(
                "interaction_beats"),
        })
    _emit(artifact_sink, "semantic_gate/candidate_selection", selection_audit)
    if gate_validation["overall_decision"] == "reject":
        raise WorkflowError(
            "semantic_continuation_gate",
            ["insufficient_gate_pass_candidates:" + str(
             gate_validation.get("passed_candidate_count", 0)) + "/" +
             str(gate_validation.get("minimum_pass_candidates",
                                     minimum_scene_count))]
        )
    effective_scene_count = int(scene_quota)
    if not require_full_scene_quota:
        effective_scene_count = min(
            int(scene_quota),
            len(effective_passed_candidate_refs) or
            int(gate_validation.get("passed_candidate_count") or 0),
        )
        if effective_scene_count < 1:
            raise WorkflowError("scene_quota", ["no_legal_scene_candidates"])
        # The semantic outline selector may conservatively reduce the set.
        effective_scene_count = min(
            effective_scene_count, len(effective_passed_candidate_refs))
        if effective_scene_count < 1:
            raise WorkflowError("planner_outline_semantic",
                                ["no_legal_scene_candidates"])
    try:
        materialization_source = (gain_policy.materialization_input(
            planner_compiled, effective_passed_candidate_refs)
            if development_policy else planner_compiled)
        planned = materialize_planner_output(
            materialization_source,
            passed_candidate_refs=effective_passed_candidate_refs,
            scene_count=effective_scene_count)
    except (TypeError, ValueError) as error:
        raise WorkflowError("planner_materialization", [str(error)])
    if development_policy:
        planned = gain_policy.annotate_selected(
            planned, outline_semantic_audit, round_index=round_index)
    _emit(artifact_sink, "planner/materialized_plan", planned)
    isolation = validate_realizer_plan_isolation(
        raw_plan=materialization_source, materialized_plan=planned,
        passed_candidate_refs=effective_passed_candidate_refs,
        scene_count=effective_scene_count)
    if single_event_line_mode:
        chain_validation = validate_materialized_single_event_line(
            planned, round_index=round_index)
        isolation = {
            **isolation,
            "valid": bool(isolation["valid"] and chain_validation["valid"]),
            "errors": list(isolation.get("errors") or []) + list(
                chain_validation.get("errors") or []),
            "single_event_chain_validation": chain_validation,
        }
    _emit(artifact_sink, "realizer/input_validation", isolation)
    _emit(artifact_sink, "realizer/input", planned)
    if not isolation["valid"]:
        raise WorkflowError("realizer_input_isolation", isolation["errors"])

    style_excerpt: list[Mapping[str, Any]] = []
    style_context: Mapping[str, Any] | None = None
    style_profile: Mapping[str, Any] | None = None
    if realizer_style_tail_only:
        style_context = _build_realizer_style_context(parent_turns)
        style_profile = dict(style_context["style_profile"])
        _emit(artifact_sink, "realizer/style_context", style_context)
    realizer_batch_plans = build_realizer_plan_batches(
        planned, batch_size=realizer_batch_size, round_index=round_index,
        batch_policy=realizer_batch_policy)
    _emit(artifact_sink, "realizer/batch_schedule", {
        "batch_size": int(realizer_batch_size),
        "batch_policy": realizer_batch_policy,
        "batch_count": len(realizer_batch_plans),
        "purpose": "normal_generation",
        "dispatch_mode": "legacy_single_call" if len(realizer_batch_plans) == 1
        else "explicit_batch_schedule_requires_sequential_promotion",
    })
    realizer_prompts: list[dict[str, Any]] = []
    realizer_outputs: list[Any] = []
    batch_audits: list[dict[str, Any]] = []
    batch_promoted_scene_refs: list[str] = []
    batch_promoted_scenes: list[dict[str, Any]] = []
    batch_promoted_plan_scenes: list[dict[str, Any]] = []
    promoted_prefix_turns: list[dict[str, Any]] = []
    batch_generation_errors: list[str] = []
    for batch_index, batch_plan in enumerate(realizer_batch_plans, 1):
        promoted_prefix = (promoted_prefix_turns
                           if sequential_batch_promotion else None)
        if feedback_view_version:
            from ..h4_round1_r03_feedback_view_v2.renderers import (
                render_expansion_realizer,
            )
            prompt = render_expansion_realizer(
                record=feedback_record, dialogue=feedback_dialogue,
                plan=batch_plan, style_profile=style_profile,
                continuity_shape=(style_context or {}).get("continuity_tail")
                if realizer_style_tail_only else None,
                promoted_prefix=promoted_prefix,
                input_mode=("text_free" if realizer_style_tail_only
                            else "full_parent"),
                request_purpose="normal_generation",
            )
        else:
            prompt = build_realizer_prompt(
                parent=parent, plan=batch_plan, style_excerpt=style_excerpt,
                style_profile=style_profile,
                continuity_shape=(style_context or {}).get("continuity_tail")
                if realizer_style_tail_only else None,
                promoted_prefix=promoted_prefix,
                include_full_parent=not realizer_style_tail_only,
                request_purpose="normal_generation")
        prompt["batch_index"] = batch_index
        prompt["batch_scene_refs"] = [
            str(scene.get("scene_ref") or scene.get("memory_event_id") or "")
            for scene in batch_plan.get("scenes") or []]
        prompt["prior_promoted_scene_refs"] = list(batch_promoted_scene_refs)
        _emit(artifact_sink, f"realizer/prompt_batch_{batch_index:02d}", prompt)
        realizer_prompts.append(prompt)
        try:
            output = realizer_call(prompt)
        except Exception as error:
            if sequential_batch_promotion:
                batch_generation_errors.append(
                    f"batch_{batch_index}_realizer:{type(error).__name__}:{error}")
                # Independent later batches remain eligible; they receive only
                # the prefix that actually passed promotion so far.
                continue
            raise
        realizer_outputs.append(output)
        _emit(artifact_sink, f"realizer/output_batch_{batch_index:02d}", output)
        if not sequential_batch_promotion:
            continue

        batch_parent = deepcopy(dict(parent))
        batch_dialogue = deepcopy(dict(batch_parent.get("dialogue") or {}))
        batch_dialogue["turns"] = (
            list(batch_dialogue.get("turns") or []) +
            deepcopy(promoted_prefix_turns))
        batch_parent["dialogue"] = batch_dialogue
        batch_normalized = normalize_realizer_output(
            output, plan=batch_plan, parent=batch_parent,
            round_index=round_index)
        _emit(artifact_sink, f"realizer/validation_batch_{batch_index:02d}",
              batch_normalized)
        if not batch_normalized["valid"]:
            batch_generation_errors.extend(
                f"batch_{batch_index}_normalize:{error}"
                for error in batch_normalized.get("errors") or [])
            continue
        batch_child = batch_normalized["child"]
        batch_adherence_input = build_adherence_auditor_input(
            plan=batch_plan, child=batch_child)
        batch_adherence_prompt = build_adherence_auditor_prompt(
            auditor_input=batch_adherence_input)
        _emit(artifact_sink,
              f"realizer_adherence/batch_{batch_index:02d}_input",
              batch_adherence_input)
        _emit(artifact_sink,
              f"realizer_adherence/batch_{batch_index:02d}_prompt",
              batch_adherence_prompt)
        batch_adherence_raw = adherence_call(batch_adherence_prompt)
        _emit(artifact_sink,
              f"realizer_adherence/batch_{batch_index:02d}_output",
              batch_adherence_raw)
        batch_adherence_result = validate_and_materialize_adherence_audit(
            batch_adherence_raw, plan=batch_plan, child=batch_child)
        _emit(artifact_sink,
              f"realizer_adherence/batch_{batch_index:02d}_validation",
              batch_adherence_result)
        if not batch_adherence_result["valid"]:
            batch_generation_errors.extend(
                f"batch_{batch_index}_adherence:{error}"
                for error in batch_adherence_result.get("errors") or [])
            continue
        batch_promotion = build_promotable_child(
            parent=batch_parent, plan=batch_plan, child=batch_child,
            adherence_audit=batch_adherence_result["audit"],
            satisfied_dependency_refs=batch_promoted_scene_refs)
        _emit(artifact_sink,
              f"promotion/batch_{batch_index:02d}_audit",
              batch_promotion.get("audit") or {})
        if not batch_promotion["valid"]:
            batch_generation_errors.extend(
                f"batch_{batch_index}_promotion:{error}"
                for error in batch_promotion.get("errors") or [])
            continue
        promoted_batch = batch_promotion.get("promotable_child") or {}
        promoted_rows = list(promoted_batch.get("scenes") or [])
        if not promoted_rows:
            batch_generation_errors.append(
                f"batch_{batch_index}_zero_promoted_scenes")
            continue
        promoted_refs = [str(row.get("scene_ref") or "")
                         for row in promoted_rows]
        batch_promoted_scene_refs.extend(promoted_refs)
        batch_promoted_scenes.extend(deepcopy(promoted_rows))
        plan_by_ref = {
            str(scene.get("scene_ref") or ""): scene
            for scene in batch_plan.get("scenes") or []
            if isinstance(scene, Mapping)}
        batch_promoted_plan_scenes.extend(
            deepcopy(plan_by_ref[ref]) for ref in promoted_refs
            if ref in plan_by_ref)
        promoted_prefix_turns.extend(
            {"turn_id": str(turn.get("turn_ref") or ""),
             "turn_ref": str(turn.get("turn_ref") or ""),
             "speaker": str(turn.get("speaker") or ""),
             "text": str(turn.get("text") or "")}
            for scene in promoted_rows
            for turn in scene.get("dialogue_turns") or [])
        batch_audits.append({
            "batch_index": batch_index,
            "promoted_scene_refs": promoted_refs,
            "promoted_scene_count": len(promoted_refs),
        })

    if sequential_batch_promotion:
        if not batch_promoted_scenes:
            raise WorkflowError(
                "realizer_batch_promotion",
                ["zero_promoted_scenes"] + batch_generation_errors)
        planned = deepcopy(dict(planned))
        planned["scenes"] = batch_promoted_plan_scenes
        child = {
            "schema_version": "memory_event_continuation_child_v1",
            "root_id": parent["root_id"], "round_index": int(round_index),
            "scenes": batch_promoted_scenes,
        }
        realized = {"scenes": [
            scene for output in realizer_outputs
            for scene in (output.get("scenes") or [])
            if isinstance(output, Mapping)]}
        normalized = {"valid": True, "errors": [], "child": child,
                      "sequential_batch_promotion": True,
                      "batch_audits": batch_audits,
                      "batch_generation_errors": batch_generation_errors}
    elif len(realizer_outputs) == 1:
        realized = realizer_outputs[0]
    else:
        realized = {"scenes": [
            scene for output in realizer_outputs
            for scene in (output.get("scenes") or [])
            if isinstance(output, Mapping)]}
    realizer_prompt = (realizer_prompts[0] if len(realizer_prompts) == 1
                       else {"prompt_version": "realizer_batch_schedule_v1",
                             "batch_count": len(realizer_prompts),
                             "batches": realizer_prompts})
    _emit(artifact_sink,"realizer/prompt",realizer_prompt)
    _emit(artifact_sink,"realizer/output",realized)
    if not sequential_batch_promotion:
        normalized=normalize_realizer_output(
            realized, plan=planned, parent=parent, round_index=round_index)
    _emit(artifact_sink,"realizer/validation",normalized)
    if not normalized["valid"]: raise WorkflowError("realizer_adherence",normalized["errors"])
    child=normalized["child"]
    expanded = (assemble_expanded_dialogue(parent=parent, child=child)
                if int(round_index) == 1 else append_expansion_round(
                    prior_expanded=prior_expanded, child=child))
    expanded_validation = (validate_expanded_dialogue(
        expanded, parent=parent) if int(round_index) == 1 else
        validate_appended_round(expanded, prior_expanded=prior_expanded))
    _emit(artifact_sink,"child_dialogue",child); _emit(artifact_sink,"expanded_dialogue",expanded)
    _emit(artifact_sink,"expanded_dialogue_validation",expanded_validation)
    if not expanded_validation["valid"]: raise WorkflowError("child_assembly",expanded_validation["errors"])

    # The generated Child is provisional until one whole-round semantic audit
    # confirms every selected transition and durable outcome. No Memory write
    # is allowed before this gate passes.
    adherence_input = build_adherence_auditor_input(plan=planned, child=child)
    adherence_prompt = build_adherence_auditor_prompt(
        auditor_input=adherence_input)
    _emit(artifact_sink, "realizer_adherence/input", adherence_input)
    _emit(artifact_sink, "realizer_adherence/prompt", adherence_prompt)
    adherence_raw = adherence_call(adherence_prompt)
    _emit(artifact_sink, "realizer_adherence/output", adherence_raw)
    adherence_result = validate_and_materialize_adherence_audit(
        adherence_raw, plan=planned, child=child)
    _emit(artifact_sink, "realizer_adherence/audit",
          adherence_result.get("audit") or {})
    _emit(artifact_sink, "realizer_adherence/validation", adherence_result)
    if not adherence_result["valid"]:
        raise WorkflowError("realizer_adherence_audit_contract",
                            adherence_result["errors"])
    adherence_audit = adherence_result["audit"]
    initial_adherence_audit = deepcopy(adherence_audit)
    _emit(artifact_sink, "realizer_adherence/initial_audit",
          initial_adherence_audit)

    # A failed scene with correct participant bindings can be completed once
    # from its actual final turn.  Passing scenes are frozen; participant
    # drift remains quarantined because it cannot be repaired by appending
    # dialogue without changing the original actor contract.
    continuation_request = build_scene_continuation_request(
        plan=planned, child=child, adherence_audit=adherence_audit)
    _emit(artifact_sink, "scene_continuation/request", continuation_request)
    continuation_prompt = None
    continuation_raw = None
    continuation_materialization = None
    continuation_merged_child = None
    adherence_reaudit_prompt = None
    adherence_reaudit_raw = None
    adherence_reaudit_result = None
    replacement_request = None
    replacement_prompt = None
    replacement_raw = None
    replacement_materialization = None
    replacement_adherence_prompt = None
    replacement_adherence_raw = None
    replacement_adherence_result = None
    if (continuation_request["eligible_scene_count"] and
            scene_continuation_call is not None and
            adherence_reaudit_call is not None):
        _emit(artifact_sink, "scene_continuation/initial_child", child)
        _emit(artifact_sink, "scene_continuation/initial_expanded_dialogue",
              expanded)
        continuation_prompt = build_scene_continuation_prompt(
            continuation_request)
        _emit(artifact_sink, "scene_continuation/prompt", continuation_prompt)
        continuation_raw = scene_continuation_call(continuation_prompt)
        _emit(artifact_sink, "scene_continuation/output", continuation_raw)
        continuation_materialization = materialize_scene_continuation_output(
            continuation_raw, request=continuation_request, child=child)
        _emit(artifact_sink, "scene_continuation/validation",
              continuation_materialization)
        if not continuation_materialization["valid"]:
            raise WorkflowError(
                "scene_continuation_contract",
                continuation_materialization["errors"])
        continuation_merged_child = merge_scene_continuations(
            child, continuation_materialization)
        _emit(artifact_sink, "scene_continuation/merged_child",
              continuation_merged_child)
        child = continuation_merged_child
        expanded = (assemble_expanded_dialogue(
            parent=parent, child=child) if int(round_index) == 1 else
            append_expansion_round(
                prior_expanded=prior_expanded, child=child))
        expanded_validation = (validate_expanded_dialogue(
            expanded, parent=parent) if int(round_index) == 1 else
            validate_appended_round(
                expanded, prior_expanded=prior_expanded))
        _emit(artifact_sink, "scene_continuation/expanded_dialogue",
              expanded)
        _emit(artifact_sink, "scene_continuation/expanded_validation",
              expanded_validation)
        if not expanded_validation["valid"]:
            raise WorkflowError(
                "scene_continuation_assembly",
                expanded_validation["errors"])
        _emit(artifact_sink, "child_dialogue", child)
        _emit(artifact_sink, "expanded_dialogue", expanded)

        # Re-audit the merged Child once.  The second audit is the final audit
        # consumed by the unchanged promotion layer.
        adherence_reaudit_input = build_adherence_auditor_input(
            plan=planned, child=child)
        adherence_reaudit_prompt = build_adherence_auditor_prompt(
            auditor_input=adherence_reaudit_input)
        _emit(artifact_sink, "realizer_adherence/reaudit_input",
              adherence_reaudit_input)
        _emit(artifact_sink, "realizer_adherence/reaudit_prompt",
              adherence_reaudit_prompt)
        adherence_reaudit_raw = adherence_reaudit_call(
            adherence_reaudit_prompt)
        _emit(artifact_sink, "realizer_adherence/reaudit_output",
              adherence_reaudit_raw)
        adherence_reaudit_result = validate_and_materialize_adherence_audit(
            adherence_reaudit_raw, plan=planned, child=child)
        _emit(artifact_sink, "realizer_adherence/reaudit_validation",
              adherence_reaudit_result)
        if not adherence_reaudit_result["valid"]:
            raise WorkflowError(
                "realizer_adherence_reaudit_contract",
                adherence_reaudit_result["errors"])
        adherence_audit = adherence_reaudit_result["audit"]
        adherence_result = adherence_reaudit_result
        _emit(artifact_sink, "realizer_adherence/validation",
              adherence_reaudit_result)
        _emit(artifact_sink, "realizer_adherence/audit", adherence_audit)
        _emit(artifact_sink, "realizer_adherence/reaudit_audit",
              adherence_audit)
    elif continuation_request["eligible_scene_count"]:
        # Library callers that do not expose live recovery retain the previous
        # partial-promotion behavior. The live runner always supplies both
        # callbacks, so its bounded recovery path is active.
        _emit(artifact_sink, "scene_continuation/skipped", {
            "reason": "callbacks_not_configured",
            "eligible_scene_count": continuation_request["eligible_scene_count"],
        })

    # If the single continuation attempt still leaves a scene rejected, use
    # at most one unused Gate-approved candidate as a replacement.  This is a
    # different event/development, never a second retry of the failed event.
    replacement_request = ({
        "request_version": "scene_replacement_request_v1",
        "selected_replacement": None,
        "eligible_scene_count": 0,
        "reason": ("development_gain_does_not_auto_replace_selected_set" if
                   development_policy else "single_event_line_forbids_cross_event_replacement"),
    } if single_event_line_mode or development_policy else build_scene_replacement_request(
        raw_plan=planner_compiled,
        materialized_plan=planned,
        child=child,
        adherence_audit=adherence_audit,
        passed_candidate_refs=gate_validation.get("passed_candidate_refs") or [],
        outline_conflict_edges=list(
            (outline_semantic_audit or {}).get("conflict_edges") or []),
    ))
    _emit(artifact_sink, "scene_replacement/request", replacement_request)
    if (replacement_request.get("selected_replacement") is not None and
            scene_replacement_call is not None and
            replacement_adherence_call is not None):
        replacement_prompt = build_scene_replacement_prompt(replacement_request)
        _emit(artifact_sink, "scene_replacement/prompt", replacement_prompt)
        replacement_raw = scene_replacement_call(replacement_prompt)
        _emit(artifact_sink, "scene_replacement/output", replacement_raw)
        replacement_materialization = materialize_scene_replacement_output(
            replacement_raw, request=replacement_request)
        _emit(artifact_sink, "scene_replacement/validation",
              replacement_materialization)
        if not replacement_materialization["valid"]:
            raise WorkflowError("scene_replacement_contract",
                                replacement_materialization["errors"])
        planned = materialize_replacement_plan(planned, replacement_request)
        _emit(artifact_sink, "planner/materialized_plan_after_replacement", planned)
        child = merge_scene_replacement(child, replacement_materialization)
        expanded = (assemble_expanded_dialogue(
            parent=parent, child=child) if int(round_index) == 1 else
            append_expansion_round(
                prior_expanded=prior_expanded, child=child))
        expanded_validation = (validate_expanded_dialogue(
            expanded, parent=parent) if int(round_index) == 1 else
            validate_appended_round(
                expanded, prior_expanded=prior_expanded))
        _emit(artifact_sink, "scene_replacement/child_dialogue", child)
        _emit(artifact_sink, "scene_replacement/expanded_dialogue", expanded)
        _emit(artifact_sink, "scene_replacement/expanded_validation",
              expanded_validation)
        if not expanded_validation["valid"]:
            raise WorkflowError("scene_replacement_assembly",
                                expanded_validation["errors"])

        # The replacement must receive its own final audit.  It is the only
        # audit after replacement; a failed replacement is quarantined with no
        # further retry.
        replacement_adherence_input = build_adherence_auditor_input(
            plan=planned, child=child)
        replacement_adherence_prompt = build_adherence_auditor_prompt(
            auditor_input=replacement_adherence_input)
        _emit(artifact_sink, "scene_replacement/adherence_input",
              replacement_adherence_input)
        _emit(artifact_sink, "scene_replacement/adherence_prompt",
              replacement_adherence_prompt)
        replacement_adherence_raw = replacement_adherence_call(
            replacement_adherence_prompt)
        _emit(artifact_sink, "scene_replacement/adherence_output",
              replacement_adherence_raw)
        replacement_adherence_result = validate_and_materialize_adherence_audit(
            replacement_adherence_raw, plan=planned, child=child)
        _emit(artifact_sink, "scene_replacement/adherence_validation",
              replacement_adherence_result)
        if not replacement_adherence_result["valid"]:
            raise WorkflowError(
                "scene_replacement_adherence_contract",
                replacement_adherence_result["errors"])
        adherence_audit = replacement_adherence_result["audit"]
        adherence_result = replacement_adherence_result
        _emit(artifact_sink, "realizer_adherence/audit", adherence_audit)
    elif replacement_request.get("selected_replacement") is not None:
        _emit(artifact_sink, "scene_replacement/skipped", {
            "reason": "replacement_callbacks_not_configured",
        })

    # A whole-round Auditor still judges every scene.  Promotion is a second,
    # deterministic layer: it keeps only passing scenes that can be appended
    # without explicit dependency or exact/normalised duplicate violations.
    promotion_result = build_promotable_child(
        parent=parent, plan=planned, child=child,
        adherence_audit=adherence_audit)
    _emit(artifact_sink, "promotion/audit",
          promotion_result.get("audit") or {})
    _emit(artifact_sink, "promotion/validation", promotion_result)
    if not promotion_result["valid"]:
        raise WorkflowError("promotion_contract", promotion_result["errors"])
    promotable_child = promotion_result["promotable_child"]
    promotable_child_validation = validate_promotable_child(
        promotable_child, source_child=child)
    _emit(artifact_sink, "promotion/promotable_child", promotable_child)
    _emit(artifact_sink, "promotion/promotable_child_validation",
          promotable_child_validation)
    if not promotable_child_validation["valid"]:
        raise WorkflowError("promotion_materialization",
                            promotable_child_validation["errors"])
    if not (promotion_result["audit"] or {}).get("promotable_child_ready"):
        raise WorkflowError("realizer_adherence_audit",
                            ["no_promotable_scene"])
    promotable_expanded = (assemble_expanded_dialogue(
        parent=parent, child=promotable_child) if int(round_index) == 1 else
        append_expansion_round(
            prior_expanded=prior_expanded, child=promotable_child))
    promotable_expanded_validation = (validate_expanded_dialogue(
        promotable_expanded, parent=parent) if int(round_index) == 1 else
        validate_appended_round(
            promotable_expanded, prior_expanded=prior_expanded))
    _emit(artifact_sink, "promotion/promotable_expanded_dialogue",
          promotable_expanded)
    _emit(artifact_sink, "promotion/promotable_expanded_dialogue_validation",
          promotable_expanded_validation)
    if not promotable_expanded_validation["valid"]:
        raise WorkflowError("promotion_assembly",
                            promotable_expanded_validation["errors"])
    child_turns=[{**row, "session_ref": scene["session_ref"]}
                 for scene in promotable_child["scenes"]
                 for row in scene["dialogue_turns"]]
    # Round 2+ is an incremental Memorax write.  The canonical cumulative
    # dialogue remains the factual source for prompts and promotion, but the
    # persistence window must not re-submit Parent/R01 text.  Keeping the
    # window on the new child turns also makes the contract explicit for
    # backends that may use window_context during memory extraction.
    recent_context_turns = (
        parent_turns[-20:] if int(round_index) == 1 else child_turns)
    child_intent=build_add_intent(root_id=root_id,round_index=round_index,
        source_dialogue_sha256=promotable_expanded["complete_dialogue_sha256"],turns=child_turns,
        recent_context_turns=recent_context_turns,async_mode=True,
        user_namespace=memorax_user_namespace)
    _emit(artifact_sink,"memory/child_add_intent",child_intent)
    child_state={}
    if no_memorax_writes:
        # Generation is useful for an isolated retry, but this mode must never
        # submit the generated R02 dialogue to Memorax. Keep the intent as a
        # local audit artifact and make the absence of write verification
        # explicit instead of pretending that a write occurred.
        child_completion = {
            "status": "skipped",
            "reason": "no_memorax_writes",
            "verified_complete": False,
            "memorax_add_calls": 0,
        }
        child_state = {
            "status": "skipped",
            "stage": "child_memory_add",
            "add_submitted": False,
            "memory_write_verified": False,
            "bucket_visibility_verified": False,
            "automatic_retry_allowed": False,
        }
        _emit(artifact_sink, "memory/child_add_completion", child_completion)
        _emit(artifact_sink, "memory/child_add_write_state", child_state)
        post_snapshot = pre_snapshot
        child_visibility_audit = {
            "audit_version": "simple_append_memorax_bucket_delta_audit_v1",
            "valid": True,
            "errors": [],
            "root_id": root_id,
            "user_id": str(query["payload"]["user_id"]),
            "before_snapshot_sha256": str(pre_snapshot.get("snapshot_sha256") or ""),
            "after_snapshot_sha256": str(pre_snapshot.get("snapshot_sha256") or ""),
            "added_bucket_refs": [],
            "updated_bucket_refs": [],
            "removed_bucket_refs": [],
            "changed_bucket_refs": [],
            "added_bucket_count": 0,
            "updated_bucket_count": 0,
            "removed_bucket_count": 0,
            "bucket_visibility_verified": False,
            "semantic_visibility_verified": False,
            "reason": "no_memorax_writes",
        }
        _emit(artifact_sink, "memory/child_visibility_audit", child_visibility_audit)
        post_overview = compile_memory_overview(
            post_snapshot, recent_dialogue_text=_recent(child_turns))
    else:
        if before_memory_request:
            before_memory_request("child_add",child_intent["request_id"])
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
            request_id=f"memory-event-r{int(round_index):02d}-post-{root_id}")
        child_visibility_audit = audit_bucket_delta_visibility(
            pre_snapshot, post_snapshot)
        child_state = apply_visibility_result(
            child_state,
            visible=bool(child_visibility_audit["bucket_visibility_verified"]),
        )
        _emit(artifact_sink, "memory/child_visibility_audit", child_visibility_audit)
        _emit(artifact_sink, "memory/child_add_write_state", child_state)
        post_overview=compile_memory_overview(post_snapshot,recent_dialogue_text=_recent(child_turns))
    _emit(artifact_sink,"memory/post_generation_overview",post_overview)
    return {"workflow_version":"controlled_world_expansion_workflow_v1","root_id":root_id,
      "round_index":int(round_index),
      "parent_context":tail,"candidates":candidates,"pre_overview":pre_overview,
      "memory_source_mode": memory_source_mode,
      "global_memory_state_view": global_memory_state_view,
      "event_verifier_raw_output":verifier_raw,
      "event_verifier_validation":verifier_validation,
      "verified_tracks":verified_tracks,
      "planner_raw_output":planner_raw,"planner_compiled_output":planner_compiled,
      "planner_variant_selection": planner_variant_selection,
      "planner_event_binding_alignment": event_binding_alignment,
      "planner_repair_raw_output":planner_repair_raw,
      "planner_repair_validation":planner_repair_validation,
      "planner_repair_merged_validation":planner_repair_merged_validation,
      "planner_repair_prompt":planner_repair_prompt,
      "scene_quota_audit":quota_audit,
      "planner_outline_audit": outline_audit,
      "planner_outline_semantic_prompt": outline_semantic_prompt,
      "planner_outline_semantic_raw_output": outline_semantic_raw,
      "planner_outline_semantic_audit": outline_semantic_audit,
      "planner_outline_filtered_candidate_refs":
          outline_filtered_candidate_refs,
      "effective_passed_candidate_refs": effective_passed_candidate_refs,
      "planner_output":planned,
      "planner_validation":validation,
      "semantic_gate_context":gate_context,"semantic_gate_output":gate_output,
      "semantic_gate_validation":gate_validation,
      "skip_planner_semantic_gate": bool(skip_planner_semantic_gate),
      "candidate_selection_audit":selection_audit,
      "realizer_input_isolation":isolation,"realizer_output":realized,
      "realizer_batch_schedule": {
          "batch_size": int(realizer_batch_size),
          "batch_count": len(realizer_batch_plans),
          "sequential_batch_promotion": bool(sequential_batch_promotion),
          "batch_audits": batch_audits,
          "batch_promoted_scene_refs": list(batch_promoted_scene_refs),
          "batch_generation_errors": list(batch_generation_errors),
          "dispatch_mode": ("legacy_single_call" if len(realizer_batch_plans) == 1
                             else ("sequential_local_promotion"
                                   if sequential_batch_promotion
                                   else "explicit_batch_schedule_requires_sequential_promotion"))},
      "initial_adherence_audit": initial_adherence_audit,
      "scene_continuation_request": continuation_request,
      "scene_continuation_prompt": continuation_prompt,
      "scene_continuation_raw_output": continuation_raw,
      "scene_continuation_materialization": continuation_materialization,
      "scene_continuation_merged_child": continuation_merged_child,
      "scene_replacement_request": replacement_request,
      "scene_replacement_prompt": replacement_prompt,
      "scene_replacement_raw_output": replacement_raw,
      "scene_replacement_materialization": replacement_materialization,
      "scene_replacement_adherence_prompt": replacement_adherence_prompt,
      "scene_replacement_adherence_raw_output": replacement_adherence_raw,
      "scene_replacement_adherence_result": replacement_adherence_result,
      "adherence_reaudit_prompt": adherence_reaudit_prompt,
      "adherence_reaudit_raw_output": adherence_reaudit_raw,
      "adherence_reaudit_result": adherence_reaudit_result,
      "realizer_validation":normalized,"child_dialogue":child,"expanded_dialogue":expanded,
      "expanded_validation":expanded_validation,
      "promotable_child_dialogue":promotable_child,
      "promotable_expanded_dialogue":promotable_expanded,
      "promotable_expanded_validation":promotable_expanded_validation,
      "promotion_audit":promotion_result["audit"],
      "promotion_validation":promotion_result,
      "realizer_adherence_audit":adherence_audit,
      "realizer_adherence_validation":adherence_result,
      "post_overview":post_overview,
      "parent_add_completion":parent_completion,"parent_add_skipped":reuse_existing_namespace,
      "child_add_completion":child_completion,
      "parent_add_write_state":parent_state,
      "parent_visibility_audit":parent_visibility_audit,
      "child_add_write_state":child_state,
      "child_visibility_audit":child_visibility_audit,
      "memory_call_count":len(memorax.calls),
      "generation_ready":bool((promotion_result["audit"] or {}).get(
          "promotable_child_ready")),
      "full_round_ready":bool((promotion_result["audit"] or {}).get(
          "full_round_ready")),
      "partial_promotion":bool((promotion_result["audit"] or {}).get(
          "partial_promotion")),
      "memory_write_verified":bool((reuse_existing_namespace or parent_state.get("memory_write_verified")) and child_state.get("memory_write_verified")),
      "memory_visibility_verified":bool((reuse_existing_namespace or parent_state.get("bucket_visibility_verified")) and child_state.get("bucket_visibility_verified")),
      "event_bucket_only":bool(event_bucket_only),
      "no_memorax_writes": bool(no_memorax_writes)}
