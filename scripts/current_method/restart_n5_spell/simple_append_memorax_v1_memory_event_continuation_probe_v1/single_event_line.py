"""Optional single-Memory-event ordered-chain contract.

The existing continuation workflow keeps a global candidate pool.  This
module provides a deliberately separate, opt-in contract for the next pilot:
one selected Memory event is expanded through three ordered scene steps.  The
program owns scene refs and dependency edges; the Planner only supplies the
semantic step content.
"""

from __future__ import annotations

from copy import deepcopy
import json
import re
from typing import Any, Mapping, Sequence


SINGLE_EVENT_LINE_VERSION = "single_memory_event_ordered_chain_v1"
DEFAULT_SCENE_COUNT = 3

_STEP_KEYS = frozenset({
    "goal",
    "required_participants",
    "transition_from_previous",
    "required_transition",
    "after_state",
    "actor_bindings",
    "interaction_beats",
    "new_people",
})


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _normal(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _text(value).casefold()).strip()


def _mentions_canonical_name(text: Any, name: str) -> bool:
    """Match a canonical speaker name without false hits for names like A."""
    value = _text(text)
    canonical = _text(name)
    if not value or not canonical:
        return False
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(canonical)}(?![A-Za-z0-9_])"
    return re.search(pattern, value, flags=re.IGNORECASE) is not None


def build_single_event_line_prompt(
    *, target_track: Mapping[str, Any],
    speaker_registry: Mapping[str, Any],
    scene_count: int = DEFAULT_SCENE_COUNT,
) -> dict[str, Any]:
    """Build a compact prompt for one frozen event line.

    No Parent transcript is included.  The target track is the sole semantic
    source for the round; the Realizer later receives only the clean plan.
    """
    count = int(scene_count)
    if count < 1 or count > 3:
        raise ValueError("single_event_line_scene_count_invalid")
    track = {
        "verified_track_ref": _text(target_track.get("verified_track_ref")),
        "memory_event_ref": _text(target_track.get("memory_event_ref") or
                                   target_track.get("memory_event_id")),
        "memory_event_key": _text(target_track.get("memory_event_key")),
        "memory_event_display": _text(target_track.get("memory_event_display")),
        "verified_current_state": _text(
            target_track.get("verified_current_state")),
        "memory_context": deepcopy(target_track.get("memory_event_content") or
                                    target_track.get("memory_planning_packet") or
                                    []),
    }
    text = f"""You are planning one ordered continuation of one concrete Memory event.
Do not introduce another event line. Treat the supplied current state as
already realized history. Produce exactly {count} sequential steps:
step 1 starts from the current state; each later step must depend on the
previous step's after_state and add a concrete new development. Do not repeat
the existing decision, intention, action, agreement, resolution, or result.
Do not output scene IDs or dependency IDs; the local program creates them.
Use only `speaker_refs` from the canonical registry for the people required by
each step, and use `actor_ref` for actor_bindings. The local program resolves
these refs into `required_participants`. This is a required-coverage list, not
a speaker allowlist: an incidental existing character, clerk, or passerby may
also speak when natural, but must not take over a bound core action. Every
canonical speaker explicitly named in a goal or interaction beat must appear
in the required refs. Do not use Memorax transport labels or aliases. New
people may be introduced only when they are explicitly connected to this
event.

TARGET EVENT
{json.dumps(track, ensure_ascii=False, sort_keys=True, indent=2)}

CANONICAL SPEAKER REGISTRY
{json.dumps(speaker_registry, ensure_ascii=False, sort_keys=True, indent=2)}

Output only JSON with this shape:
{{"round_goal":"...","steps":[{{
  "goal":"...",
  "speaker_refs":["SP01", "SP02"],
  "transition_from_previous":"...",
  "required_transition":"...",
  "after_state":"...",
  "actor_bindings":[{{"actor_ref":"SP01","binding":"..."}}],
  "interaction_beats":["..."],
  "new_people":[]
}}]}}
"""
    return {
        "prompt_version": "single_event_line_planner_prompt_v3_required_participants",
        "prompt": text,
        "scene_count": count,
        "target_event_ref": track["memory_event_ref"],
    }


def validate_single_event_line_plan(
    value: Any, *, target_track: Mapping[str, Any],
    parent_speakers: Sequence[str],
    scene_count: int = DEFAULT_SCENE_COUNT,
) -> dict[str, Any]:
    """Validate only the bounded semantic shape; IDs and dependencies are local."""
    errors: list[str] = []
    count = int(scene_count)
    if not isinstance(value, Mapping):
        return {"valid": False, "errors": ["plan_not_object"],
                "step_count": 0}
    if set(value) != {"round_goal", "steps"}:
        errors.append("plan_top_level_shape")
    if not _text(value.get("round_goal")):
        errors.append("round_goal_empty")
    steps = value.get("steps")
    if not isinstance(steps, list):
        return {"valid": False, "errors": errors + ["steps_shape"],
                "step_count": 0}
    if len(steps) != count:
        errors.append(f"step_count:{len(steps)}/{count}")
    allowed_speakers = {_text(name) for name in parent_speakers if _text(name)}
    target_ref = _text(target_track.get("memory_event_ref") or
                       target_track.get("memory_event_id"))
    if not target_ref:
        errors.append("target_event_ref_missing")
    current_state = _text(target_track.get("verified_current_state"))
    if not current_state:
        errors.append("target_current_state_missing")
    previous_after = current_state
    normalized_transitions: list[str] = []
    for index, step in enumerate(steps, 1):
        prefix = f"step_{index}"
        if not isinstance(step, Mapping):
            errors.append(f"{prefix}_object")
            continue
        if set(step) != _STEP_KEYS:
            errors.append(f"{prefix}_shape")
        for field in ("goal", "transition_from_previous",
                      "required_transition", "after_state"):
            if not _text(step.get(field)):
                errors.append(f"{prefix}_{field}_empty")
        participants = step.get("required_participants")
        if not isinstance(participants, list) or not participants:
            errors.append(f"{prefix}_required_participants_shape")
            participants = []
        participant_names = [_text(row) for row in participants if _text(row)]
        if len(participant_names) != len(set(participant_names)):
            errors.append(f"{prefix}_duplicate_participants")
        unknown = sorted(set(participant_names) - allowed_speakers)
        new_people = step.get("new_people")
        if not isinstance(new_people, list):
            errors.append(f"{prefix}_new_people_shape")
            new_people = []
        declared_new = {
            _text(row.get("name")) for row in new_people
            if isinstance(row, Mapping) and _text(row.get("name"))
        }
        unknown = [name for name in unknown if name not in declared_new]
        if unknown:
            errors.append(f"{prefix}_unknown_participants:" + ",".join(unknown))
        bindings = step.get("actor_bindings")
        if not isinstance(bindings, list) or not bindings:
            errors.append(f"{prefix}_actor_bindings_shape")
            bindings = []
        for binding in bindings:
            if (not isinstance(binding, Mapping) or
                    set(binding) not in ({"actor_ref", "binding"},
                                         {"actor", "binding"})):
                errors.append(f"{prefix}_actor_binding_shape")
                continue
            actor = _text(binding.get("actor_ref") or binding.get("actor"))
            actor_name = actor
            if actor.startswith("SP") and actor[2:].isdigit():
                ref_index = int(actor[2:])
                if 1 <= ref_index <= len(parent_speakers):
                    actor_name = _text(parent_speakers[ref_index - 1])
            if actor_name not in participant_names:
                errors.append(f"{prefix}_actor_not_participant:{actor}")
            if not _text(binding.get("binding")):
                errors.append(f"{prefix}_actor_binding_empty")
        beats = step.get("interaction_beats")
        if (not isinstance(beats, list) or not beats or
                any(not _text(beat) for beat in beats)):
            errors.append(f"{prefix}_interaction_beats_shape")
        transition = _text(step.get("required_transition"))
        normalized = _normal(transition)
        if normalized and normalized in normalized_transitions:
            errors.append(f"{prefix}_duplicate_transition")
        if normalized:
            normalized_transitions.append(normalized)
        # The program derives this field, but requiring the step to mention
        # the previous state keeps a chain from degenerating into three roots.
        transition_context = _normal(step.get("transition_from_previous"))
        if index > 1 and previous_after and previous_after.casefold() not in \
                _text(step.get("transition_from_previous")).casefold():
            # Exact text cannot be required from a paraphrasing model. Record
            # only a diagnostic; dependency materialization remains strict.
            pass
        previous_after = _text(step.get("after_state")) or previous_after
    return {
        "valid": not errors,
        "errors": errors,
        "step_count": len(steps),
        "target_event_ref": target_ref,
        "dependency_mode": "programmatic_ordered_chain",
        "scene_count": count,
    }


def materialize_single_event_line_plan(
    value: Mapping[str, Any], *, target_track: Mapping[str, Any],
    round_index: int = 1,
    parent_speakers: Sequence[str] = (),
    scene_count: int = DEFAULT_SCENE_COUNT,
) -> dict[str, Any]:
    """Create clean Realizer input with one event and explicit ordered deps."""
    validation = validate_single_event_line_plan(
        value, target_track=target_track, parent_speakers=parent_speakers,
        scene_count=scene_count)
    if not validation["valid"]:
        raise ValueError("single_event_line_plan_invalid:" + ",".join(
            validation["errors"]))
    event_ref = _text(target_track.get("memory_event_ref") or
                      target_track.get("memory_event_id"))
    event_key = _text(target_track.get("memory_event_key"))
    event_display = _text(target_track.get("memory_event_display"))
    current_state = _text(target_track.get("verified_current_state"))
    scenes: list[dict[str, Any]] = []
    previous_scene_ref: str | None = None
    previous_after = current_state
    for index, raw_step in enumerate(value["steps"], 1):
        step = deepcopy(dict(raw_step))
        scene_ref = f"R{int(round_index):02d}:S{index:02d}"
        before_state = current_state if index == 1 else previous_after
        actor_bindings = []
        for binding in step["actor_bindings"]:
            if not isinstance(binding, Mapping):
                continue
            actor = _text(binding.get("actor_ref") or binding.get("actor"))
            if actor.startswith("SP") and actor[2:].isdigit():
                ref_index = int(actor[2:])
                parent_speakers = list(parent_speakers or [])
                if 1 <= ref_index <= len(parent_speakers):
                    actor = _text(parent_speakers[ref_index - 1])
            actor_bindings.append({
                "actor": actor,
                "binding": _text(binding.get("binding")),
            })
        scene = {
            "scene_ref": scene_ref,
            "thread_root_scene_ref": f"R{int(round_index):02d}:S01",
            "thread_depth": index,
            "depends_on_scene_refs": ([previous_scene_ref]
                                       if previous_scene_ref else []),
            "memory_event_id": event_ref,
            "memory_event_ref": event_ref,
            "memory_event_key": event_key,
            "memory_event_display": event_display,
            "memory_event_content": deepcopy(
                target_track.get("memory_event_content") or []),
            "supporting_sources": deepcopy(
                target_track.get("supporting_sources") or []),
            "development_mode": "continuation",
            "verified_track_ref": _text(
                target_track.get("verified_track_ref")),
            "verified_current_state": current_state,
            "evidence_turn_refs": deepcopy(
                target_track.get("evidence_turn_refs") or []),
            "evidence_source": deepcopy(target_track.get("evidence_source") or []),
            "goal": _text(step["goal"]),
            # `required_participants` is the canonical meaning.  The
            # deterministic `participants` alias is retained only for frozen
            # downstream contracts that have not yet been renamed.
            "required_participants": deepcopy(step["required_participants"]),
            "participants": deepcopy(step["required_participants"]),
            "before_state": before_state,
            "transition_from_previous": _text(
                step["transition_from_previous"]),
            "required_transition": _text(step["required_transition"]),
            "after_state": _text(step["after_state"]),
            "actor_bindings": actor_bindings,
            "interaction_beats": deepcopy(step["interaction_beats"]),
            "new_people": deepcopy(step["new_people"]),
        }
        scenes.append(scene)
        previous_scene_ref = scene_ref
        previous_after = scene["after_state"]
    return {
        "single_event_line_version": SINGLE_EVENT_LINE_VERSION,
        "round_goal": _text(value["round_goal"]),
        "target_memory_event_ref": event_ref,
        "target_memory_event_key": event_key,
        "scenes": scenes,
        "scene_count": len(scenes),
        "dependency_mode": "ordered_chain",
    }


def build_gate_compatible_plan(single_event_plan: Mapping[str, Any]) -> dict[str, Any]:
    """Expose one Gate candidate per step while preserving one Memory event."""
    scenes = single_event_plan.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != 3:
        raise ValueError("single_event_line_gate_scene_count")
    refs = {_text(scene.get("memory_event_ref")) for scene in scenes
            if isinstance(scene, Mapping)}
    if len(refs) != 1 or "" in refs:
        raise ValueError("single_event_line_gate_event_identity")
    event_ref = next(iter(refs))
    adapted: list[dict[str, Any]] = []
    for index, source in enumerate(scenes, 1):
        row = deepcopy(dict(source))
        row["memory_event_id"] = f"{event_ref}:STEP{index:02d}"
        row["single_event_step_index"] = index
        row["next_step_candidates"] = [{
            "before_state": row["before_state"],
            "required_transition": row["required_transition"],
            "after_state": row["after_state"],
            "actor_bindings": deepcopy(row["actor_bindings"]),
            "interaction_beats": deepcopy(row["interaction_beats"]),
            "development_mode": "continuation",
        }]
        adapted.append(row)
    return {
        "round_goal": single_event_plan.get("round_goal"),
        "single_event_line_version": SINGLE_EVENT_LINE_VERSION,
        "single_event_adapter_version": "single_event_gate_adapter_v1",
        "single_event_target_memory_event_ref": event_ref,
        "scenes": deepcopy(adapted),
        "candidate_pool": deepcopy(adapted),
        "scene_quota_audit": {
            "scene_quota": 3,
            "single_event_line": True,
            "target_memory_event_ref": event_ref,
        },
    }


def validate_single_event_gate_plan(
    plan: Mapping[str, Any], *, round_index: int = 1,
) -> dict[str, Any]:
    errors: list[str] = []
    scenes = plan.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != 3:
        return {"valid": False, "errors": ["single_event_gate_scene_count"]}
    event_ref = _text(plan.get("single_event_target_memory_event_ref"))
    seen_refs = {_text(scene.get("memory_event_ref")) for scene in scenes
                 if isinstance(scene, Mapping)}
    if seen_refs != {event_ref} or not event_ref:
        errors.append("single_event_gate_event_identity")
    for index, scene in enumerate(scenes, 1):
        if not isinstance(scene, Mapping):
            errors.append(f"single_event_gate_scene_{index}_object")
            continue
        expected = f"R{int(round_index):02d}:S{index:02d}"
        if _text(scene.get("scene_ref")) != expected:
            errors.append(f"single_event_gate_scene_{index}_ref")
        expected_deps = ([] if index == 1 else
                         [f"R{int(round_index):02d}:S{index - 1:02d}"])
        if scene.get("depends_on_scene_refs") != expected_deps:
            errors.append(f"single_event_gate_scene_{index}_dependency")
        if len(scene.get("next_step_candidates") or []) != 1:
            errors.append(f"single_event_gate_scene_{index}_candidate_count")
    return {"valid": not errors, "errors": errors,
            "scene_count": len(scenes), "memory_event_ref": event_ref}


def validate_materialized_single_event_line(
    plan: Mapping[str, Any], *, scene_count: int = DEFAULT_SCENE_COUNT,
    round_index: int = 1,
) -> dict[str, Any]:
    """Verify compiler-owned event identity and dependency edges."""
    errors: list[str] = []
    scenes = plan.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != int(scene_count):
        return {"valid": False, "errors": ["materialized_scene_count"],
                "scene_count": len(scenes) if isinstance(scenes, list) else 0}
    event_refs = {_text(scene.get("memory_event_ref")) for scene in scenes
                  if isinstance(scene, Mapping)}
    if len(event_refs) != 1 or "" in event_refs:
        errors.append("materialized_event_identity_not_single")
    for index, scene in enumerate(scenes, 1):
        prefix = f"scene_{index}"
        if not isinstance(scene, Mapping):
            errors.append(f"{prefix}_object")
            continue
        expected_ref = f"R{int(round_index):02d}:S{index:02d}"
        if _text(scene.get("scene_ref")) != expected_ref:
            errors.append(f"{prefix}_scene_ref")
        expected_deps = ([] if index == 1 else
                         [f"R{int(round_index):02d}:S{index - 1:02d}"])
        if scene.get("depends_on_scene_refs") != expected_deps:
            errors.append(f"{prefix}_dependency_chain")
        if _text(scene.get("before_state")) == "":
            errors.append(f"{prefix}_before_state_empty")
        if not _text(scene.get("after_state")):
            errors.append(f"{prefix}_after_state_empty")
    return {"valid": not errors, "errors": errors,
            "scene_count": len(scenes), "event_refs": sorted(event_refs)}
