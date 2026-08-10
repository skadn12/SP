"""Small deterministic contracts for Memory-event continuation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence


def build_event_candidates(overview: Mapping[str, Any], *, max_candidates: int = 8) -> list[dict[str, Any]]:
    """Build concrete event hints from event buckets and explicit person events.

    Memorax may place recent event summaries under a person's bucket rather than
    the event bucket.  Those summaries are accepted only when explicitly marked
    as events; their text is never merged by broad thematic similarity.
    """
    candidates: list[dict[str, Any]] = []
    seen_content: set[str] = set()

    def event_text(value: Any) -> str:
        text = " ".join(str(value or "").split()).strip()
        if text.startswith("[事件") and "]" in text:
            text = text.split("]", 1)[1].strip()
        return text

    def add_candidate(*, key: str, display: str, content: str) -> bool:
        normalized = event_text(content)
        content_key = normalized.casefold()
        if not key or not normalized or content_key in seen_content:
            return False
        seen_content.add(content_key)
        candidates.append({
            "memory_event_id": f"ME{len(candidates) + 1:02d}",
            "memory_event_key": key,
            "memory_event_display": display or key,
            "memory_event_content": [normalized],
        })
        return len(candidates) >= int(max_candidates)

    events = overview.get("events") or []
    if isinstance(events, list):
        for row in events:
            if len(candidates) >= int(max_candidates):
                break
            if not isinstance(row, Mapping):
                continue
            key = str(row.get("canonical_key") or row.get("display_key") or "").strip()
            display = str(row.get("display_key") or key).strip()
            content_values = row.get("content") or []
            if not isinstance(content_values, list):
                continue
            for content in content_values:
                if add_candidate(key=key, display=display, content=str(content)):
                    break

    people = overview.get("people") or []
    if isinstance(people, list):
        for row in people:
            if len(candidates) >= int(max_candidates):
                break
            if not isinstance(row, Mapping):
                continue
            person_key = str(row.get("canonical_key") or row.get("display_key") or "").strip()
            person_display = str(row.get("display_key") or person_key).strip()
            content_values = row.get("content") or []
            if not isinstance(content_values, list):
                continue
            event_index = 0
            for raw_content in content_values:
                content = str(raw_content).strip()
                # Only explicit event-labelled person facts are candidates.
                if not content.startswith("[") or not content.startswith("[事件"):
                    continue
                event_index += 1
                key = f"{person_key}:recent_event:{event_index:02d}"
                display = f"{person_display} recent event {event_index}"
                add_candidate(key=key, display=display, content=content)

    return candidates[:int(max_candidates)]


def _parent_turn_map(parent: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("turn_id")): row for row in parent["dialogue"]["turns"]}


def validate_planner_output(value: Any, *, candidates: Sequence[Mapping[str, Any]],
                            parent: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(value, Mapping):
        return {"valid": False, "errors": ["planner_not_object"]}
    if set(value) != {"round_goal", "scenes"}:
        errors.append("planner_top_level_shape")
    scenes = value.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != 3:
        errors.append("planner_requires_three_scenes")
        scenes = scenes if isinstance(scenes, list) else []
    allowed = {str(row.get("memory_event_id")) for row in candidates}
    turns = _parent_turn_map(parent)
    parent_speakers = {str(row.get("speaker")) for row in parent["dialogue"]["turns"]}
    used: set[str] = set()
    for index, scene in enumerate(scenes, 1):
        prefix = f"scene_{index}"
        if not isinstance(scene, Mapping):
            errors.append(f"{prefix}_not_object")
            continue
        event_id = str(scene.get("memory_event_id") or "")
        if event_id not in allowed: errors.append(f"{prefix}_unknown_memory_event")
        if event_id in used: errors.append(f"{prefix}_duplicate_memory_event")
        used.add(event_id)
        state = str(scene.get("verified_current_state") or "").strip()
        if not state: errors.append(f"{prefix}_empty_verified_current_state")
        refs = scene.get("evidence_turn_refs")
        if not isinstance(refs, list) or not refs: errors.append(f"{prefix}_evidence_required")
        else:
            for ref in refs:
                if str(ref) not in turns: errors.append(f"{prefix}_evidence_not_in_parent:{ref}")
        participants = scene.get("participants")
        if not isinstance(participants, list) or len(participants) < 2:
            errors.append(f"{prefix}_participants")
        elif any(str(v) not in parent_speakers for v in participants):
            errors.append(f"{prefix}_participant_not_in_parent")
        if {"selected_candidate_index", "new_development", "durable_outcome",
                "interaction_beats"} & set(scene):
            errors.append(f"{prefix}_forbidden_final_selection_fields")
        next_steps = scene.get("next_step_candidates")
        if (not isinstance(next_steps, list) or not 2 <= len(next_steps) <= 3
                or any(not isinstance(value, Mapping) or
                       set(value) != {"new_development", "durable_outcome",
                                      "interaction_beats"} or
                       not str(value.get("new_development") or "").strip() or
                       not str(value.get("durable_outcome") or "").strip() or
                       not isinstance(value.get("interaction_beats"), list) or
                       not 3 <= len(value.get("interaction_beats") or []) <= 7 or
                       any(not str(beat or "").strip()
                           for beat in value.get("interaction_beats") or [])
                       for value in next_steps)):
            errors.append(f"{prefix}_next_step_candidates")
            next_steps = next_steps if isinstance(next_steps, list) else []
        else:
            normalized_steps = [" ".join(
                str(value["new_development"]).split()).casefold()
                for value in next_steps]
            if len(set(normalized_steps)) != len(normalized_steps):
                errors.append(f"{prefix}_duplicate_next_step_candidates")
        for field in ("goal",):
            if not str(scene.get(field) or "").strip(): errors.append(f"{prefix}_empty_{field}")
    return {"valid": not errors, "errors": errors, "scene_count": len(scenes),
            "selected_memory_event_ids": sorted(used)}


def _candidate_ref(event_id: str, index: int) -> str:
    return f"{event_id}:C{index:02d}"


def materialize_planner_output(
    value: Mapping[str, Any], *, passed_candidate_refs: Sequence[str],
) -> dict[str, Any]:
    """Select the first Gate-approved candidate in Planner recommendation order."""
    materialized = deepcopy(dict(value))
    passed = {str(ref) for ref in passed_candidate_refs}
    scenes = materialized.get("scenes")
    if not isinstance(scenes, list):
        raise ValueError("planner_materialization_scenes_required")
    for index, scene in enumerate(scenes, 1):
        if not isinstance(scene, dict):
            raise ValueError(f"planner_materialization_scene_{index}_object")
        event_id = str(scene.get("memory_event_id") or "")
        candidates = scene.get("next_step_candidates")
        if not isinstance(candidates, list):
            raise ValueError(f"planner_materialization_scene_{index}_candidates")
        selected_ref = next((
            _candidate_ref(event_id, candidate_index)
            for candidate_index in range(1, len(candidates) + 1)
            if _candidate_ref(event_id, candidate_index) in passed), None)
        if selected_ref is None:
            raise ValueError(f"planner_materialization_scene_{index}_no_pass_candidate")
        selected_index = int(selected_ref.rsplit("C", 1)[1]) - 1
        candidate = candidates[selected_index]
        if not isinstance(candidate, Mapping):
            raise ValueError(f"planner_materialization_scene_{index}_candidate")
        scene.pop("next_step_candidates", None)
        scene.pop("interaction_beats", None)
        scene["selected_candidate_ref"] = selected_ref
        scene["new_development"] = str(candidate["new_development"]).strip()
        scene["durable_outcome"] = str(candidate["durable_outcome"]).strip()
        scene["interaction_beats"] = deepcopy(candidate["interaction_beats"])
    return materialized


def validate_realizer_plan_isolation(
    *, raw_plan: Mapping[str, Any], materialized_plan: Mapping[str, Any],
    passed_candidate_refs: Sequence[str],
) -> dict[str, Any]:
    """Prove that only the selected PASS candidate reaches the Realizer plan."""
    errors: list[str] = []
    raw_scenes = raw_plan.get("scenes")
    final_scenes = materialized_plan.get("scenes")
    if not isinstance(raw_scenes, list) or not isinstance(final_scenes, list):
        return {"valid": False, "errors": ["realizer_isolation_scene_shape"],
                "rejected_content_leak_count": 0}
    if len(raw_scenes) != len(final_scenes):
        errors.append("realizer_isolation_scene_count")
    passed = {str(value) for value in passed_candidate_refs}
    selected_refs: list[str] = []
    leak_count = 0
    forbidden_container_keys = {
        "next_step_candidates", "candidates", "all_candidates",
        "rejected_candidates", "selected_candidate_index",
    }
    for scene_index, (raw_scene, final_scene) in enumerate(
            zip(raw_scenes, final_scenes), 1):
        prefix = f"scene_{scene_index}"
        if not isinstance(raw_scene, Mapping) or not isinstance(final_scene, Mapping):
            errors.append(f"{prefix}_realizer_isolation_object")
            continue
        event_id = str(raw_scene.get("memory_event_id") or "")
        candidates = raw_scene.get("next_step_candidates")
        if not isinstance(candidates, list):
            errors.append(f"{prefix}_realizer_isolation_candidates")
            continue
        selected_index = next((
            index for index in range(1, len(candidates) + 1)
            if _candidate_ref(event_id, index) in passed), None)
        if selected_index is None:
            errors.append(f"{prefix}_realizer_isolation_no_pass")
            continue
        selected_ref = _candidate_ref(event_id, selected_index)
        selected_refs.append(selected_ref)
        selected = candidates[selected_index - 1]
        if not isinstance(selected, Mapping):
            errors.append(f"{prefix}_realizer_isolation_selected_object")
            continue
        if str(final_scene.get("selected_candidate_ref") or "") != selected_ref:
            errors.append(f"{prefix}_realizer_isolation_selected_ref")
        for field in ("new_development", "durable_outcome", "interaction_beats"):
            if final_scene.get(field) != selected.get(field):
                errors.append(f"{prefix}_realizer_isolation_{field}")
        if forbidden_container_keys & set(final_scene):
            errors.append(f"{prefix}_realizer_isolation_candidate_container")

        final_strings: set[str] = set()
        def collect_strings(value: Any) -> None:
            if isinstance(value, str):
                final_strings.add(value)
            elif isinstance(value, Mapping):
                for nested in value.values(): collect_strings(nested)
            elif isinstance(value, list):
                for nested in value: collect_strings(nested)
        collect_strings(final_scene)
        for candidate_index, candidate in enumerate(candidates, 1):
            if candidate_index == selected_index or not isinstance(candidate, Mapping):
                continue
            for field in ("new_development", "durable_outcome"):
                value = candidate.get(field)
                if isinstance(value, str) and value in final_strings:
                    leak_count += 1
                    errors.append(f"{prefix}_unselected_{field}_leak")
            for beat in candidate.get("interaction_beats") or []:
                if isinstance(beat, str) and beat in final_strings:
                    leak_count += 1
                    errors.append(f"{prefix}_unselected_interaction_beat_leak")
    return {"valid": not errors, "errors": errors,
            "selected_candidate_refs": selected_refs,
            "rejected_content_leak_count": leak_count}


def build_semantic_gate_context(*, parent: Mapping[str, Any],
                                candidates: Sequence[Mapping[str, Any]],
                                plan: Mapping[str, Any]) -> dict[str, Any]:
    """Expose complete Parent sessions containing Planner evidence refs."""
    candidate_by_id = {
        str(row.get("memory_event_id")): dict(row) for row in candidates}
    turn_by_ref = _parent_turn_map(parent)
    parent_turns = list(parent["dialogue"]["turns"])
    scenes: list[dict[str, Any]] = []
    for scene in plan.get("scenes") or []:
        event_id = str(scene.get("memory_event_id") or "")
        refs = [str(value) for value in scene.get("evidence_turn_refs") or []]
        session_ids = list(dict.fromkeys(
            str(turn_by_ref[ref].get("session_id"))
            for ref in refs if ref in turn_by_ref))
        evidence_window = [{
            "turn_ref": str(row.get("turn_id")),
            "speaker": str(row.get("speaker")),
            "text": str(row.get("text")),
        } for row in parent_turns
            if str(row.get("session_id")) in session_ids]
        candidate_rows = [{
            "candidate_ref": _candidate_ref(event_id, candidate_index),
            "new_development": candidate.get("new_development"),
            "durable_outcome": candidate.get("durable_outcome"),
            "interaction_beats": candidate.get("interaction_beats"),
        } for candidate_index, candidate in enumerate(
            scene.get("next_step_candidates") or [], 1)
            if isinstance(candidate, Mapping)]
        scenes.append({
            "memory_event_id": event_id,
            "memory_event": candidate_by_id.get(event_id, {}),
            "planner_verified_current_state": scene.get(
                "verified_current_state"),
            "planner_evidence_turn_refs": refs,
            "authoritative_parent_evidence": evidence_window,
            "candidates": candidate_rows,
        })
    return {"gate_context_version": "semantic_continuation_gate_context_v3",
            "scenes": scenes}


def validate_semantic_gate_output(value: Any, *, plan: Mapping[str, Any],
                                  parent: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(value, Mapping) or set(value) != {"candidate_reviews"}:
        return {"valid": False, "errors": ["gate_top_level_shape"],
                "overall_decision": "invalid"}
    reviews = value.get("candidate_reviews")
    expected_by_ref: dict[str, str] = {}
    refs_by_event: dict[str, list[str]] = {}
    for scene in plan.get("scenes") or []:
        event_id = str(scene.get("memory_event_id") or "")
        refs_by_event[event_id] = []
        for index, _ in enumerate(scene.get("next_step_candidates") or [], 1):
            ref = _candidate_ref(event_id, index)
            expected_by_ref[ref] = event_id
            refs_by_event[event_id].append(ref)
    parent_refs = set(_parent_turn_map(parent))
    if not isinstance(reviews, list) or len(reviews) != len(expected_by_ref):
        errors.append("gate_review_count")
        reviews = reviews if isinstance(reviews, list) else []
    seen: set[str] = set()
    passed: list[str] = []
    rejected: list[str] = []
    for index, review in enumerate(reviews, 1):
        prefix = f"gate_review_{index}"
        if not isinstance(review, Mapping):
            errors.append(f"{prefix}_not_object")
            continue
        if set(review) != {"candidate_ref", "decision",
                          "duplicate_with_parent", "evidence_turn_refs",
                          "reason"}:
            errors.append(f"{prefix}_shape")
        candidate_ref = str(review.get("candidate_ref") or "")
        if candidate_ref not in expected_by_ref:
            errors.append(f"{prefix}_unknown_candidate")
        if candidate_ref in seen:
            errors.append(f"{prefix}_duplicate_candidate")
        seen.add(candidate_ref)
        decision = str(review.get("decision") or "")
        duplicate = review.get("duplicate_with_parent")
        refs = review.get("evidence_turn_refs")
        reason = str(review.get("reason") or "").strip()
        if decision not in {"pass", "reject"}:
            errors.append(f"{prefix}_decision")
        if not isinstance(duplicate, bool):
            errors.append(f"{prefix}_duplicate_boolean")
        if decision == "pass" and duplicate is not False:
            errors.append(f"{prefix}_pass_duplicate_conflict")
        if decision == "reject" and duplicate is not True:
            errors.append(f"{prefix}_reject_duplicate_conflict")
        if not isinstance(refs, list):
            errors.append(f"{prefix}_evidence_shape")
            refs = []
        if len(refs) != len(set(str(ref) for ref in refs)):
            errors.append(f"{prefix}_evidence_duplicate")
        if any(str(ref) not in parent_refs for ref in refs):
            errors.append(f"{prefix}_evidence_not_parent")
        if decision == "reject" and not refs:
            errors.append(f"{prefix}_reject_evidence_required")
        if not reason:
            errors.append(f"{prefix}_reason")
        if decision == "reject":
            rejected.append(candidate_ref)
        elif decision == "pass":
            passed.append(candidate_ref)
    if sorted(seen) != sorted(expected_by_ref):
        errors.append("gate_candidate_coverage")
    events_without_pass = [
        event_id for event_id, refs in refs_by_event.items()
        if not any(ref in passed for ref in refs)]
    overall = "invalid" if errors else (
        "reject" if events_without_pass else "pass")
    return {"valid": not errors, "errors": errors,
            "overall_decision": overall,
            "passed_candidate_refs": passed,
            "rejected_candidate_refs": rejected,
            "events_without_pass": events_without_pass,
            "realizer_allowed": not errors and not events_without_pass}


def normalize_realizer_output(value: Any, *, plan: Mapping[str, Any],
                              parent: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(value, Mapping) or set(value) != {"scenes"}:
        return {"valid": False, "errors": ["realizer_top_level_shape"]}
    raw_scenes = value.get("scenes")
    planned = list(plan.get("scenes") or [])
    if not isinstance(raw_scenes, list) or len(raw_scenes) != len(planned):
        return {"valid": False, "errors": ["realizer_scene_count"]}
    parent_texts = {str(row.get("text")) for row in parent["dialogue"]["turns"]}
    parent_speakers = {str(row.get("speaker")) for row in parent["dialogue"]["turns"]}
    child_scenes: list[dict[str, Any]] = []
    for idx, (raw, plan_scene) in enumerate(zip(raw_scenes, planned), 1):
        ref = f"R01:S{idx:02d}"
        if not isinstance(raw, Mapping): errors.append(f"{ref}:scene_not_object"); continue
        turns = raw.get("dialogue_turns")
        if not isinstance(turns, list) or not 9 <= len(turns) <= 48:
            errors.append(f"{ref}:turn_budget")
            turns = turns if isinstance(turns, list) else []
        allowed = {str(v) for v in plan_scene.get("participants") or []}
        normalized=[]
        seen_text=set()
        for j, row in enumerate(turns,1):
            if not isinstance(row, Mapping): errors.append(f"{ref}:turn_{j}_object"); continue
            speaker=str(row.get("speaker") or "").strip(); text=str(row.get("text") or "").strip()
            if speaker not in allowed or speaker not in parent_speakers: errors.append(f"{ref}:turn_{j}_speaker")
            if not text: errors.append(f"{ref}:turn_{j}_empty")
            if text in parent_texts: errors.append(f"{ref}:turn_{j}_parent_exact_copy")
            if text in seen_text: errors.append(f"{ref}:turn_{j}_duplicate_text")
            seen_text.add(text)
            normalized.append({"turn_ref":f"{ref}:T{j:03d}","speaker":speaker,"text":text})
        child_scenes.append({"scene_ref":ref,"session_ref":f"{parent['root_id']}:r01:s{idx:02d}",
                             "memory_event_id":plan_scene.get("memory_event_id"),
                             "dialogue_turns":normalized})
    return {"valid": not errors, "errors": errors,
            "child":{"schema_version":"memory_event_continuation_child_v1",
                      "root_id":parent["root_id"],"round_index":1,"scenes":child_scenes}}
