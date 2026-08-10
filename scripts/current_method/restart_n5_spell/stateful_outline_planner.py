"""Typed outline planner contract for one stateful Graph expansion round."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any, Mapping, Sequence

from .input_adapter import dialogue_content_hash
from .stateful_graph_expansion import (
    COMPLETION_CONDITIONALITIES, COMPLETION_STATUSES,
    COMPLETION_TEMPORALITIES, STANDARD_STAGE_SCHEDULE,
    STAGE_ROLES, STATEFUL_CROSS_ARC_CONFLICT_GATE_VERSION,
    STATEFUL_COMPLETION_CONTRACT_VERSION,
    STATEFUL_GRAPH_OUTLINE_SCHEMA_VERSION,
    cross_arc_stage_conflict_gate,
    terminal_evidence_is_future_or_conditional,
    validate_stateful_outline,
)


STATEFUL_OUTLINE_PLANNER_VERSION = "stateful_outline_graph_planner_v2"
STATEFUL_OUTLINE_SCHEMA_VERSION = STATEFUL_GRAPH_OUTLINE_SCHEMA_VERSION
STATEFUL_OUTLINE_CHUNKED_VERSION = (
    "stateful_outline_graph_planner_chunked_program_roles_handoffs_v6")
STATEFUL_OUTLINE_OVERVIEW_ANCHOR_SELECTION_VERSION = (
    "stateful_outline_overview_anchor_selection_v1")
STATEFUL_OUTLINE_OVERVIEW_PARTICIPANT_INTERFACE_VERSION = (
    "stateful_outline_overview_participant_interface_v1")
STATEFUL_OUTLINE_E1_TERMINAL_INTERFACE_VERSION = (
    "stateful_outline_e1_terminal_interface_v4")
STATEFUL_OUTLINE_E1_RETRY_FREEZE_VERSION = (
    "stateful_outline_e1_retry_program_freeze_v1")
STATEFUL_OUTLINE_EVENT_IDENTITY_INTERFACE_VERSION = (
    "stateful_outline_event_identity_interface_v1")
STATEFUL_OUTLINE_CROSS_ARC_REPAIR_VERSION = (
    "stateful_outline_cross_arc_conflict_repair_v1")
STATEFUL_DYNAMIC_PLANNER_INPUT_VERSION = "stateful_dynamic_planner_input_v1"

_KEY = re.compile(r"^[a-z][a-z0-9_]*$")
_INTERFACE_NONTERMINAL_EVIDENCE_PREFIX = re.compile(
    r"^\s*(?:i|we)\s+(?:plan(?:ned|ning)?|agree(?:d)?(?:\s+to)?|"
    r"(?:was|were)\s+assign(?:ed)|start(?:ed|ing)?|"
    r"(?:am|are)\s+working\s+on|(?:am|are)\s+prepar(?:ing|ed))\b",
    re.IGNORECASE,
)
_AMBIGUOUS_MARKER_PREFIX = re.compile(
    r"^\s*(?:it(?:['\u2019]s|\s+is)|that(?:['\u2019]s|\s+is)|"
    r"this\s+is|they\b)", re.IGNORECASE)
MAX_CANONICAL_MARKER_WORDS = 18
MAX_CANONICAL_MARKER_CHARS = 160

STAGE_FIELDS = {
    "description", "object_id", "state_key", "pre_state",
    "event_identity_key", "stage_role", "predecessor_stage_ref",
    "target_post_state", "allowed_post_states", "terminal_post_states",
    "completion_event_type", "mandatory_resolution",
    "authorized_result_speakers", "completion_evidence_phrases",
    "completion_status", "completion_temporality",
    "completion_conditionality", "outcome_handoffs",
}
MODEL_STAGE_FIELDS = STAGE_FIELDS - {"outcome_handoffs"}
HANDOFF_FIELDS = {
    "handoff_key", "description", "owner", "created_on_post_state",
    "resolution_evidence_phrases", "authorized_result_speakers",
}
SCENE_FIELDS = {
    "purpose", "participants", "parent_anchor_aliases", "location",
    "scene_role",
}
MODEL_SCENE_FIELDS = SCENE_FIELDS - {"scene_role"}

_STRING_ARRAY_SCHEMA = {
    "type": "array", "minItems": 1, "uniqueItems": True,
    "items": {"type": "string", "minLength": 1},
}
RAW_STAGE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "required": sorted(STAGE_FIELDS),
    "properties": {
        "description": {"type": "string", "minLength": 1},
        "object_id": {"type": "string", "minLength": 1},
        "state_key": {"type": "string", "minLength": 1},
        "event_identity_key": {"type": "string", "minLength": 1},
        "stage_role": {"enum": sorted(STAGE_ROLES)},
        "predecessor_stage_ref": {"type": ["string", "null"]},
        "pre_state": {"type": "string", "minLength": 1},
        "target_post_state": {"type": "string", "minLength": 1},
        "allowed_post_states": _STRING_ARRAY_SCHEMA,
        "terminal_post_states": _STRING_ARRAY_SCHEMA,
        "completion_event_type": {"type": "string", "minLength": 1},
        "mandatory_resolution": {"const": True},
        "authorized_result_speakers": _STRING_ARRAY_SCHEMA,
        "completion_evidence_phrases": {
            "type": "array", "minItems": 1, "maxItems": 1,
            "items": {"type": "string", "minLength": 1,
                      "maxLength": MAX_CANONICAL_MARKER_CHARS},
        },
        "completion_status": {"enum": sorted(COMPLETION_STATUSES)},
        "completion_temporality": {
            "enum": sorted(COMPLETION_TEMPORALITIES)},
        "completion_conditionality": {
            "enum": sorted(COMPLETION_CONDITIONALITIES)},
        "outcome_handoffs": {"type": "array", "maxItems": 1},
    },
}


RAW_STATEFUL_OUTLINE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "title", "expansion_objective", "allowed_new_characters",
        "allowed_locations", "time_window", "event_arcs", "scenes",
    ],
    "properties": {
        "title": {"type": "string", "minLength": 1},
        "expansion_objective": {"type": "string", "minLength": 1},
        "allowed_new_characters": {
            "type": "array",
            "maxItems": 2,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "role"],
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string"},
                },
            },
        },
        "allowed_locations": {
            "type": "array", "minItems": 3, "maxItems": 3,
            "uniqueItems": True, "items": {"type": "string"},
        },
        "time_window": {"type": "string", "minLength": 1},
        "event_arcs": {
            "type": "array", "minItems": 4, "maxItems": 4,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["summary", "parent_anchor_aliases", "stages"],
                "properties": {
                    "summary": {"type": "string"},
                    "parent_anchor_aliases": {
                        "type": "array", "minItems": 1, "maxItems": 4,
                        "uniqueItems": True,
                        "items": {"type": "string"},
                    },
                    "stages": {
                        "type": "array", "minItems": 4, "maxItems": 4,
                        "items": RAW_STAGE_JSON_SCHEMA,
                    },
                },
            },
        },
        "scenes": {
            "type": "array", "minItems": 10, "maxItems": 10,
            "items": {"type": "object"},
        },
    },
}

OVERVIEW_FIELDS = {
    "title", "expansion_objective", "allowed_new_characters",
    "allowed_locations", "time_window", "scenes",
}
ARC_FIELDS = {"summary", "parent_anchor_aliases", "stages"}


def validate_stateful_outline_overview(
    overview: Any, *, parent_turns: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(overview, dict) or set(overview) != OVERVIEW_FIELDS:
        return {"decision": "stateful_outline_overview_schema_invalid",
                "reasons": ["overview_fields"]}
    errors: list[str] = []
    aliases = {f"T{position:04d}" for position, _ in
               enumerate(parent_turns, 1)}
    known_people = {str(turn.get("speaker")) for turn in parent_turns}
    for key in ("title", "expansion_objective", "time_window"):
        if not _text(overview.get(key)):
            errors.append(key)
    new_people = overview.get("allowed_new_characters")
    new_names: list[str] = []
    if not isinstance(new_people, list) or len(new_people) > 2:
        errors.append("allowed_new_characters")
        new_people = []
    for row in new_people:
        if (not isinstance(row, dict) or set(row) != {"name", "role"} or
                not _text(row.get("name")) or not _text(row.get("role"))):
            errors.append("new_character_schema")
        else:
            new_names.append(str(row["name"]).strip())
    if len(new_names) != len(set(new_names)):
        errors.append("duplicate_new_character")
    locations = overview.get("allowed_locations")
    if not _strings(locations, minimum=3, maximum=3):
        errors.append("allowed_locations")
        locations = []
    scenes = overview.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != 10:
        errors.append("scenes")
        scenes = []
    allowed_people = known_people | set(new_names)
    materialized_scenes: list[dict[str, Any]] = []
    for scene_index, scene in enumerate(scenes, 1):
        if not isinstance(scene, dict) or set(scene) != MODEL_SCENE_FIELDS:
            errors.append("scene_fields")
            continue
        support = scene.get("parent_anchor_aliases")
        if (not _text(scene.get("purpose")) or
                scene.get("location") not in locations or
                not _strings(scene.get("participants"), minimum=2,
                             maximum=4) or
                any(name not in allowed_people for name in
                    scene.get("participants") or []) or
                not isinstance(support, list) or len(support) > 3 or
                len(support) != len(set(support)) or
                any(alias not in aliases for alias in support or [])):
            errors.append("scene_schema")
        materialized_scenes.append({
            **dict(scene), "scene_role": "advance_parent_thread"
            if scene_index <= 5 else "create_new_state_chain"})
    if scenes and not scenes[0].get("parent_anchor_aliases"):
        errors.append("opening_scene_missing_parent_anchor")
    materialized = {**dict(overview), "scenes": materialized_scenes}
    return {"decision": "stateful_outline_overview_accept" if not errors else
            "stateful_outline_overview_schema_invalid",
            "reasons": sorted(set(errors)),
            **({"overview": materialized} if not errors else {})}


def validate_stateful_outline_arc(
    arc: Any, *, parent_turns: Sequence[Mapping[str, Any]],
    overview: Mapping[str, Any], assigned_scene_indices: Sequence[int],
) -> dict[str, Any]:
    if not isinstance(arc, dict) or set(arc) != ARC_FIELDS:
        return {"decision": "stateful_outline_arc_schema_invalid",
                "reasons": ["arc_fields"]}
    errors: list[str] = []
    aliases = {f"T{position:04d}" for position, _ in
               enumerate(parent_turns, 1)}
    if not _text(arc.get("summary")):
        errors.append("event_arc_summary")
    support = arc.get("parent_anchor_aliases")
    if (not _strings(support, minimum=1, maximum=4) or
            any(alias not in aliases for alias in support or [])):
        errors.append("event_arc_parent_anchors")
    stages = arc.get("stages")
    if not isinstance(stages, list) or len(stages) != 4:
        errors.append("event_arc_stages")
        stages = []
    known_people = {str(turn.get("speaker")) for turn in parent_turns}
    known_people.update(str(row.get("name")) for row in
                        overview.get("allowed_new_characters") or [])
    identities = set()
    materialized_stages: list[dict[str, Any]] = []
    for stage in stages:
        if not isinstance(stage, dict) or set(stage) != MODEL_STAGE_FIELDS:
            errors.append("stateful_stage_fields")
            continue
        materialized_stages.append({**dict(stage), "outcome_handoffs": []})
        if any(not _text(stage.get(key)) for key in (
                "description", "object_id", "state_key", "pre_state",
                "event_identity_key", "target_post_state",
                "completion_event_type")):
            errors.append("stateful_stage_text")
        if (not _KEY.fullmatch(str(stage.get("object_id") or "")) or
                not _KEY.fullmatch(str(stage.get("state_key") or "")) or
                not _KEY.fullmatch(str(stage.get("event_identity_key") or ""))):
            errors.append("stateful_stage_key")
        if stage.get("stage_role") not in STAGE_ROLES:
            errors.append("stage_role_invalid")
        if (stage.get("stage_role") == "base_fact_completion" and
                stage.get("predecessor_stage_ref") is not None):
            errors.append("base_fact_has_predecessor")
        if (stage.get("stage_role") == "derived_state" and
                not _text(stage.get("predecessor_stage_ref"))):
            errors.append("derived_state_predecessor_invalid")
        identity = (stage.get("object_id"), stage.get("state_key"))
        if identity in identities:
            errors.append("duplicate_state_instance")
        identities.add(identity)
        allowed = stage.get("allowed_post_states")
        terminal = stage.get("terminal_post_states")
        if not _strings(allowed, minimum=1, maximum=4):
            errors.append("allowed_post_states"); allowed = []
        if (not _strings(terminal, minimum=1, maximum=4) or
                not set(terminal or []) <= set(allowed)):
            errors.append("terminal_post_states")
        if stage.get("target_post_state") not in allowed:
            errors.append("target_post_state")
        if stage.get("target_post_state") not in terminal:
            errors.append("target_post_state_not_terminal")
        if stage.get("pre_state") == stage.get("target_post_state"):
            errors.append("no_state_change")
        if stage.get("mandatory_resolution") is not True:
            errors.append("mandatory_resolution")
        if (not _strings(stage.get("authorized_result_speakers"), minimum=1,
                         maximum=3) or
                any(name not in known_people for name in
                    stage.get("authorized_result_speakers") or [])):
            errors.append("authorized_result_speakers")
        errors.extend(structured_completion_contract_errors(stage))
    scenes = overview.get("scenes") or []
    scheduled_people = [set(scenes[index - 1].get("participants") or [])
                        for index in assigned_scene_indices
                        if 1 <= index <= len(scenes)]
    for stage, participants in zip(stages, scheduled_people):
        if not participants.intersection(
                stage.get("authorized_result_speakers") or []):
            errors.append("scene_missing_stage_result_speaker")
    return {"decision": "stateful_outline_arc_accept" if not errors else
            "stateful_outline_arc_schema_invalid",
            "reasons": sorted(set(errors)),
            **({"arc": {**dict(arc), "stages": materialized_stages}}
               if not errors else {})}


def build_stateful_outline_overview_prompt(
    *, parent_turns: Sequence[Mapping[str, Any]],
) -> str:
    dialogue = "\n".join(
        f"[T{position:04d}] {turn['speaker']}: {turn['text']}"
        for position, turn in enumerate(parent_turns, 1))
    rules = [
        "Plan the shared frame for one ten-scene stateful continuation.",
        "Return overview JSON only; do not generate event_arcs, dialogue, QA, options, or answers.",
        "Create exactly ten scenes in array order. Scenes 1–5 advance parent threads; scenes 6–10 create new state chains. The program assigns these roles, so DO NOT output scene_role.",
        "Use exactly three locations and at most two useful new named characters.",
        "allowed_participants contains the exact frozen existing speaker names. Copy these names character-for-character in every scene participants array. Do not change singular/plural form, abbreviate, normalize, rename, or infer an alias.",
        "A participant must be either an exact name from allowed_participants or an exact name declared in this same output's allowed_new_characters. Copy declared new-character names exactly as well.",
        "Every scene participants array must contain 2-4 distinct exact names.",
        "The first scene must cite at least one T#### parent alias; every alias must be supplied below.",
        "Each scene parent_anchor_aliases array must contain 0-3 aliases. Select only the most critical historical anchors for that scene; do not list every related session, turn, or topic.",
        "Avoid ten scenes of proposal preparation: include spaces for decisions, execution, discovery, failure/rejection, revision, and outcome.",
        "Return strict JSON with exactly the required keys.",
    ]
    schema = {"type": "object", "additionalProperties": False,
              "required": sorted(OVERVIEW_FIELDS)}
    output_shape = {
        "title": "string", "expansion_objective": "string",
        "allowed_new_characters": [{"name": "string", "role": "string"}],
        "allowed_locations": ["string", "string", "string"],
        "time_window": "string",
        "scenes": [{"purpose": "string",
                    "participants": ["existing or declared speaker"],
                    "parent_anchor_aliases": [
                        "0-3 most critical T#### aliases only"],
                    "location": "one of allowed_locations"}],
    }
    return "\n\n".join([
        f"PROMPT_VERSION: {STATEFUL_OUTLINE_CHUNKED_VERSION}:overview:"
        f"{STATEFUL_OUTLINE_OVERVIEW_ANCHOR_SELECTION_VERSION}:"
        f"{STATEFUL_OUTLINE_OVERVIEW_PARTICIPANT_INTERFACE_VERSION}",
        "RULES\n- " + "\n- ".join(rules),
        "ALLOWED_PARTICIPANTS (exact frozen existing speaker names)\n" +
        json.dumps(sorted({str(turn["speaker"]) for turn in parent_turns}),
                   ensure_ascii=False),
        "FROZEN PARENT DIALOGUE\n" + dialogue,
        "OUTPUT TOP-LEVEL SCHEMA\n" + json.dumps(schema, sort_keys=True),
        "OUTPUT SHAPE (the scenes array must contain exactly ten objects)\n" +
        json.dumps(output_shape, ensure_ascii=False, indent=2),
        "REQUIRED SCENE KEYS\n" + json.dumps(sorted(MODEL_SCENE_FIELDS)),
    ])


def overview_anchor_overflow_feedback(
    overview: Any, *, maximum: int = 3,
) -> list[dict[str, int]]:
    """Expose anchor-count violations for a Planner-directed retry only."""
    if not isinstance(overview, Mapping):
        return []
    scenes = overview.get("scenes")
    if not isinstance(scenes, list):
        return []
    return [{"scene_index": index, "anchor_count": len(support)}
            for index, scene in enumerate(scenes, 1)
            if isinstance(scene, Mapping) and
            isinstance((support := scene.get("parent_anchor_aliases")), list) and
            len(support) > maximum]


def build_stateful_outline_overview_anchor_repair_suffix(
    *, previous_overview: Mapping[str, Any], maximum: int = 3,
) -> str | None:
    """Ask the Planner to repair only cardinality, never rewrite it locally."""
    violations = overview_anchor_overflow_feedback(
        previous_overview, maximum=maximum)
    if not violations:
        return None
    feedback = "\n".join(
        f"scene {row['scene_index']} currently has {row['anchor_count']} anchors; "
        f"the maximum allowed is {maximum}."
        for row in violations)
    return "\n\n".join([
        "ANCHOR-SELECTION REPAIR REQUIRED",
        "Your previous overview parsed as JSON, but only the listed scenes "
        "violate the parent_anchor_aliases cardinality limit.",
        feedback,
        "For every scene, parent_anchor_aliases must contain 0-3 aliases. "
        "Select only the most critical historical anchors; do not list all "
        "related sessions, turns, or topics.",
        "Regenerate the complete overview JSON. Preserve every scene's purpose, "
        "array order, participants, location, title, objective, characters, "
        "locations, and time window. Change only parent_anchor_aliases by "
        "compressing each over-limit list to the most critical 0-3 aliases. "
        "Do not add or remove scenes, people, or anchors outside this repair.",
        "Return JSON only, with the same required schema and no Markdown.",
        "PREVIOUS OVERVIEW JSON\n" + json.dumps(
            previous_overview, ensure_ascii=False, sort_keys=True, indent=2),
    ])


def overview_participant_error_feedback(
    overview: Any, *, parent_turns: Sequence[Mapping[str, Any]],
    minimum: int = 2, maximum: int = 4,
) -> list[dict[str, Any]]:
    """Expose exact participant violations without fuzzy identity matching."""
    if not isinstance(overview, Mapping):
        return []
    allowed = {str(turn.get("speaker")) for turn in parent_turns}
    declared = overview.get("allowed_new_characters")
    if isinstance(declared, list):
        allowed.update(
            str(row.get("name")) for row in declared
            if isinstance(row, Mapping) and isinstance(row.get("name"), str))
    scenes = overview.get("scenes")
    if not isinstance(scenes, list):
        return []
    feedback: list[dict[str, Any]] = []
    for scene_index, scene in enumerate(scenes, 1):
        if not isinstance(scene, Mapping):
            continue
        participants = scene.get("participants")
        if not isinstance(participants, list):
            feedback.append({
                "code": "participant_count_invalid",
                "scene_index": scene_index,
                "field": "participants",
                "actual_count": None,
                "minimum_count": minimum,
                "maximum_count": maximum,
                "original_value": participants,
            })
            continue
        if not minimum <= len(participants) <= maximum:
            feedback.append({
                "code": "participant_count_invalid",
                "scene_index": scene_index,
                "field": "participants",
                "actual_count": len(participants),
                "minimum_count": minimum,
                "maximum_count": maximum,
                "original_value": list(participants),
            })
        invalid = [name for name in participants if name not in allowed]
        if invalid:
            feedback.append({
                "code": "participant_name_not_allowed",
                "scene_index": scene_index,
                "field": "participants",
                "invalid_names": invalid,
                "allowed_names": sorted(allowed),
                "original_value": list(participants),
            })
    return feedback


def build_stateful_outline_overview_interface_repair_suffix(
    *, previous_overview: Mapping[str, Any],
    validation: Mapping[str, Any],
    parent_turns: Sequence[Mapping[str, Any]], maximum_anchors: int = 3,
) -> str | None:
    """Repair all detected Overview participant/anchor interface errors once."""
    anchor_feedback = overview_anchor_overflow_feedback(
        previous_overview, maximum=maximum_anchors)
    participant_feedback = overview_participant_error_feedback(
        previous_overview, parent_turns=parent_turns)
    if not anchor_feedback and not participant_feedback:
        return None
    allowed_existing = sorted({str(turn.get("speaker"))
                               for turn in parent_turns})
    complete_feedback: list[dict[str, Any]] = [
        {"code": "anchor_count_exceeded", **row,
         "maximum_count": maximum_anchors}
        for row in anchor_feedback]
    complete_feedback.extend(participant_feedback)
    fields_to_change = sorted({
        "parent_anchor_aliases" if row["code"] == "anchor_count_exceeded"
        else "participants" for row in complete_feedback})
    return "\n\n".join([
        "OVERVIEW PARTICIPANT-INTERFACE REPAIR REQUIRED",
        "The previous Overview parsed as JSON but failed the frozen interface. "
        "This is the single permitted retry.",
        "Actual validator errors: " + json.dumps(
            list(validation.get("reasons") or [])),
        "Complete field-level error set: " + json.dumps(
            complete_feedback, ensure_ascii=False, sort_keys=True),
        "allowed_participants exact existing names: " + json.dumps(
            allowed_existing, ensure_ascii=False) + ". Copy these names "
            "character-for-character. Do not change singular/plural form, "
            "abbreviate, normalize, rename, or infer aliases. New characters "
            "may only use the exact names already declared in the previous "
            "allowed_new_characters array.",
        "Every scene participants array must contain 2-4 distinct allowed "
        "names. Replace each listed illegal participant name with the correct "
        "exact frozen name and fix every listed participant count error.",
        f"Fields permitted to change in this retry: {json.dumps(fields_to_change)}. "
        "Preserve the title, objective, characters, locations, time window, all "
        "ten scene purposes, scene order, locations, and every already-legal "
        "parent_anchor_aliases list. Do not rewrite scene content or legal "
        "anchors. If an anchor list is explicitly listed as over-limit, only "
        f"compress that list to 0-{maximum_anchors}; never truncate locally.",
        "Regenerate the complete Overview JSON object. Return JSON only with "
        "the same frozen schema and no Markdown.",
        "PREVIOUS OVERVIEW JSON\n" + json.dumps(
            previous_overview, ensure_ascii=False, sort_keys=True, indent=2),
    ])


def build_stateful_outline_arc_prompt(
    *, parent_turns: Sequence[Mapping[str, Any]], overview: Mapping[str, Any],
    arc_index: int, assigned_scene_indices: Sequence[int],
    prior_arcs: Sequence[Mapping[str, Any]] = (),
) -> str:
    dialogue = "\n".join(
        f"[T{position:04d}] {turn['speaker']}: {turn['text']}"
        for position, turn in enumerate(parent_turns, 1))
    assigned = [overview["scenes"][index - 1]
                for index in assigned_scene_indices]
    rules = [
        f"Generate only event arc E{arc_index}, with exactly four typed stages.",
        f"Use unique lowercase snake_case object_id values prefixed e{arc_index}_ and unique state_key values.",
        "Give every underlying real-world fact a stable lowercase snake_case event_identity_key. Different facts must use different keys; reuse a key only for a genuine later state derived from that exact fact.",
        "Set stage_role=base_fact_completion for the one stage that completes a base fact and predecessor_stage_ref=null. A base fact may be completed only once across E1-E4.",
        "Set stage_role=derived_state only for a new later state caused by an earlier stage with the same event_identity_key. Then predecessor_stage_ref must exactly name that earlier stage (for example E1:stage_04), and pre_state must equal the predecessor target_post_state.",
        "A derived marker must state only its new post-state. It must not repeat the predecessor's completed base fact.",
        "Each stage changes one specific state from pre_state to target_post_state; plans and mentions are not completion.",
        "Every target result must be terminal and observable, including realistic failure, rejection, cancellation, or revision where natural.",
        "Every mandatory terminal stage must set completion_status=achieved, completion_temporality=current_or_past, and completion_conditionality=unconditional. These fixed fields describe the frozen target contract; they are not free-form claims.",
        "completion_evidence_phrases must contain exactly one short canonical marker (at most 18 words and 160 characters) that begins with I or We, names the result directly, and can be spoken verbatim by an authorized result speaker. Keep reasons, conditions, and psychological detail out of the marker and leave them for Realizer dialogue.",
        "Canonical markers must not begin with a future plan or condition such as 'I might...', 'I will...', 'If...', 'When...', or 'Once...'. The validator does not scan arbitrary internal clauses for these words; the structured completion fields carry that meaning.",
        "Every authorized result speaker must be present in that stage's assigned scene shown below.",
        "Do not output outcome_handoffs. This single-round canary assigns an empty handoff list programmatically.",
        "Use 1–4 real T#### parent aliases but do not copy parent quotes.",
        "Return only summary, parent_anchor_aliases, and stages; no dialogue, QA, options, answers, or Markdown.",
    ]
    if arc_index == 1:
        rules[3:3] = [
            "E1 STRUCTURED COMPLETION INTERFACE: Output the three fixed completion fields for every stage: achieved, current_or_past, and unconditional. The Realizer later reports mentioned, in_progress, completed, failed, or carry_forward against this frozen target contract.",
            "E1 TERMINAL INTERFACE: Planning, agreement, assignment, starting, working on, and preparation are not completion. Put such conditions in pre_state when relevant; never use them as target_post_state or completion evidence.",
            "E1 TERMINAL INTERFACE: A completion phrase must state an already observable final result, such as submitted, delivered, approved, booked, received, finished, rejected, cancelled, or explicitly confirmed completed.",
            "E1 CANONICAL-EVIDENCE INTERFACE: Every evidence phrase must be independently understandable, directly speakable, at most 18 words and 160 characters, and begin with I or We. State only the terminal marker; move explanations and subordinate conditions into later Realizer dialogue.",
            "E1 CANONICAL-EVIDENCE INTERFACE: Valid examples are: 'I confirmed the venue reservation.', 'I submitted the completed report this morning.', 'I have reflected on what happened with my dad.', and 'I still stand by my spending values.' Invalid examples are: 'I will decide tomorrow.', 'I might finish it.', 'If I decide, I will act.', 'When I finish, I will report back.', 'Once I understand, I will respond.', and 'It's finally done.'.",
            "E1 TERMINAL INTERFACE: If a later Realizer scene has not reached the frozen result, it must use mentioned, in_progress, failed, or carry_forward. Do not invent weak completion evidence to fill the schema.",
            "E1 TERMINAL INTERFACE: Use plan_creation or plan_finalization only when a concrete plan artifact has actually been finished and the evidence explicitly states that completed deliverable, not merely a decision, intention, or preparation.",
            "E1 TERMINAL INTERFACE: Select only 1–4 critical historical anchors and never add broad topic matches.",
        ]
    output_shape = {
        "summary": "string", "parent_anchor_aliases": ["T####"],
        "stages": [{
            "description": "explicit state transition",
            "object_id": f"e{arc_index}_lowercase_snake_case",
            "state_key": "lowercase_snake_case", "pre_state": "string",
            "event_identity_key": "stable_lowercase_snake_case_fact_key",
            "stage_role": "base_fact_completion or derived_state",
            "predecessor_stage_ref": None,
            "target_post_state": "string",
            "allowed_post_states": ["string"],
            "terminal_post_states": ["string"],
            "completion_event_type": "string", "mandatory_resolution": True,
            "authorized_result_speakers": ["scene participant"],
            "completion_status": "achieved",
            "completion_temporality": "current_or_past",
            "completion_conditionality": "unconditional",
            "completion_evidence_phrases": ["short canonical result marker"],
        }],
    }
    return "\n\n".join([
        f"PROMPT_VERSION: {STATEFUL_OUTLINE_CHUNKED_VERSION}:arc_{arc_index}:"
        f"{STATEFUL_OUTLINE_E1_TERMINAL_INTERFACE_VERSION if arc_index == 1 else 'base'}:"
        f"{STATEFUL_OUTLINE_EVENT_IDENTITY_INTERFACE_VERSION}",
        "RULES\n- " + "\n- ".join(rules),
        "FROZEN OVERVIEW\n" + json.dumps(overview, ensure_ascii=False,
                                          sort_keys=True, indent=2),
        "ASSIGNED SCENES FOR THE FOUR STAGES\n" + json.dumps(
            assigned, ensure_ascii=False, sort_keys=True, indent=2),
        "ALREADY ACCEPTED EARLIER ARCS (read-only; use their canonical stage "
        "references only for genuine derived states)\n" + json.dumps([
            {"arc_id": f"E{position}", "arc": dict(arc)}
            for position, arc in enumerate(prior_arcs, 1)
        ], ensure_ascii=False, sort_keys=True, indent=2),
        "FROZEN PARENT DIALOGUE\n" + dialogue,
        "OUTPUT SHAPE (the stages array must contain exactly four objects)\n" +
        json.dumps(output_shape, ensure_ascii=False, indent=2),
        "REQUIRED ARC KEYS\n" + json.dumps(sorted(ARC_FIELDS)),
        "REQUIRED STAGE KEYS\n" + json.dumps(sorted(MODEL_STAGE_FIELDS)),
    ])


def completion_evidence_interface_nonterminal(value: Any) -> bool:
    """Classify phrases for retry feedback without changing the hard validator."""
    if not isinstance(value, str):
        return False
    return (terminal_evidence_is_future_or_conditional(value) or
            _INTERFACE_NONTERMINAL_EVIDENCE_PREFIX.search(value) is not None)


def arc_completion_evidence_feedback(arc: Any) -> list[dict[str, Any]]:
    """Return exact E1 completion phrases that need directed regeneration."""
    if not isinstance(arc, Mapping):
        return []
    stages = arc.get("stages")
    if not isinstance(stages, list):
        return []
    feedback: list[dict[str, Any]] = []
    for stage_index, stage in enumerate(stages, 1):
        if not isinstance(stage, Mapping):
            continue
        phrases = stage.get("completion_evidence_phrases")
        if not isinstance(phrases, list):
            continue
        for phrase in phrases:
            if completion_evidence_interface_nonterminal(phrase):
                feedback.append({
                    "stage_index": stage_index,
                    "target_post_state": stage.get("target_post_state"),
                    "completion_evidence_phrase": phrase,
                })
    return feedback


def arc_e1_retry_error_feedback(
    arc: Any, validation: Mapping[str, Any], *, maximum_anchors: int = 4,
) -> list[dict[str, Any]]:
    """Expand every retryable E1 validator reason into field-level feedback."""
    if not isinstance(arc, Mapping):
        return []
    reasons = {str(reason) for reason in validation.get("reasons") or []}
    feedback: list[dict[str, Any]] = []
    anchors = arc.get("parent_anchor_aliases")
    if ("event_arc_parent_anchors" in reasons and isinstance(anchors, list)
            and len(anchors) > maximum_anchors):
        feedback.append({
            "code": "anchor_count_exceeded",
            "field": "parent_anchor_aliases",
            "actual_count": len(anchors),
            "maximum_count": maximum_anchors,
            "original_value": list(anchors),
        })
    stages = arc.get("stages")
    if not isinstance(stages, list):
        return feedback
    for stage_index, stage in enumerate(stages, 1):
        if not isinstance(stage, Mapping):
            continue
        if "stateful_stage_fields" in reasons:
            missing = sorted(MODEL_STAGE_FIELDS - set(stage))
            extra = sorted(set(stage) - MODEL_STAGE_FIELDS)
            for field in missing:
                feedback.append({
                    "code": "missing_stage_field", "field": field,
                    "stage_index": stage_index, "expected": (
                        "achieved" if field == "completion_status" else
                        "current_or_past" if field ==
                        "completion_temporality" else
                        "unconditional" if field ==
                        "completion_conditionality" else "required"),
                })
            for field in extra:
                feedback.append({
                    "code": "unexpected_stage_field", "field": field,
                    "stage_index": stage_index,
                    "original_value": stage.get(field),
                })
        structured_fields = {
            "completion_status": ("achieved", {
                "completion_status_invalid",
                "mandatory_terminal_completion_not_achieved",
                "completion_contract_inconsistent",
            }),
            "completion_temporality": ("current_or_past", {
                "completion_temporality_invalid",
                "mandatory_terminal_completion_not_current_or_past",
                "completion_contract_inconsistent",
            }),
            "completion_conditionality": ("unconditional", {
                "completion_conditionality_invalid",
                "mandatory_terminal_completion_not_unconditional",
                "completion_contract_inconsistent",
                "completion_marker_contract_inconsistent",
            }),
        }
        for field, (expected, related_reasons) in structured_fields.items():
            if (reasons & related_reasons and stage.get(field) != expected and
                    not any(row.get("stage_index") == stage_index and
                            row.get("field") == field for row in feedback)):
                feedback.append({
                    "code": f"{field}_invalid", "field": field,
                    "stage_index": stage_index,
                    "expected": expected, "original_value": stage.get(field),
                })
        if ("target_post_state_not_terminal" in reasons and
                stage.get("target_post_state") not in
                (stage.get("terminal_post_states") or [])):
            feedback.append({
                "code": "target_post_state_not_terminal",
                "field": "target_post_state",
                "stage_index": stage_index,
                "original_value": stage.get("target_post_state"),
                "terminal_post_states": stage.get("terminal_post_states"),
            })
        phrases = stage.get("completion_evidence_phrases")
        if not isinstance(phrases, list):
            continue
        for phrase in phrases:
            if ("completion_evidence_phrase_not_speakable" in reasons and
                    isinstance(phrase, str) and
                    not _first_person_marker(phrase)):
                feedback.append({
                    "code": "completion_evidence_phrase_not_speakable",
                    "field": "completion_evidence_phrases",
                    "stage_index": stage_index,
                    "target_post_state": stage.get("target_post_state"),
                    "original_value": phrase,
                })
            if ("completion_evidence_phrase_not_canonical_short" in reasons and
                    "completion_evidence_phrase_not_canonical_short" in
                    canonical_completion_marker_errors(phrase)):
                feedback.append({
                    "code": "completion_evidence_phrase_not_canonical_short",
                    "field": "completion_evidence_phrases",
                    "stage_index": stage_index,
                    "target_post_state": stage.get("target_post_state"),
                    "original_value": phrase,
                    "maximum_words": MAX_CANONICAL_MARKER_WORDS,
                    "maximum_characters": MAX_CANONICAL_MARKER_CHARS,
                })
            if ("completion_evidence_phrase_nonterminal" in reasons and
                    completion_evidence_interface_nonterminal(phrase)):
                feedback.append({
                    "code": "completion_evidence_phrase_nonterminal",
                    "field": "completion_evidence_phrases",
                    "stage_index": stage_index,
                    "target_post_state": stage.get("target_post_state"),
                    "original_value": phrase,
                })
    return feedback


def build_stateful_outline_e1_retry_freeze_contract(
    previous_arc: Mapping[str, Any], validation: Mapping[str, Any], *,
    maximum_anchors: int = 4, arc_id: str = "E1",
) -> dict[str, Any]:
    """Freeze all valid arc fields and expose only explicitly invalid paths."""
    feedback = arc_e1_retry_error_feedback(
        previous_arc, validation, maximum_anchors=maximum_anchors)
    mutable_stage_fields: dict[int, set[str]] = {}
    mutable_arc_fields: set[str] = set()
    for row in feedback:
        if row["code"] == "anchor_count_exceeded":
            mutable_arc_fields.add("parent_anchor_aliases")
        elif isinstance(row.get("stage_index"), int):
            mutable_stage_fields.setdefault(int(row["stage_index"]), set()).add(
                str(row["field"]))
    stages = list(previous_arc.get("stages") or [])
    frozen_stage_payloads = {}
    frozen_stage_sha256 = {}
    mutable_stage_ids = []
    for index, stage in enumerate(stages, 1):
        stage_id = f"{arc_id}.stage_{index:02d}"
        if index in mutable_stage_fields:
            mutable_stage_ids.append(stage_id)
            payload = {
                key: value for key, value in dict(stage).items()
                if key not in mutable_stage_fields[index]}
        else:
            payload = dict(stage)
        frozen_stage_payloads[stage_id] = payload
        frozen_stage_sha256[stage_id] = canonical_sha256(payload)
    return {
        "freeze_version": STATEFUL_OUTLINE_E1_RETRY_FREEZE_VERSION,
        "arc_id": arc_id,
        "valid_stage_ids": [
            f"{arc_id}.stage_{index:02d}"
            for index in range(1, len(stages) + 1)
            if index not in mutable_stage_fields],
        "mutable_stage_ids": mutable_stage_ids,
        "mutable_fields_by_stage": {
            f"{arc_id}.stage_{index:02d}": sorted(fields)
            for index, fields in sorted(mutable_stage_fields.items())},
        "mutable_arc_fields": sorted(mutable_arc_fields),
        "frozen_arc_fields": {
            key: value for key, value in dict(previous_arc).items()
            if key != "stages" and key not in mutable_arc_fields},
        "frozen_stage_payloads": frozen_stage_payloads,
        "frozen_stage_sha256": frozen_stage_sha256,
        "source_arc_sha256": canonical_sha256(previous_arc),
    }


def validate_stateful_outline_e1_retry_freeze(
    retry_arc: Any, freeze_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject any retry mutation outside the program-authorized field set."""
    errors: list[str] = []
    diagnostics: list[dict[str, Any]] = []
    if not isinstance(retry_arc, Mapping):
        return {"valid": False, "errors": ["retry_arc_not_object"],
                "diagnostics": []}
    for key, frozen_value in (freeze_contract.get("frozen_arc_fields") or
                              {}).items():
        if retry_arc.get(key) != frozen_value:
            errors.append("retry_modified_frozen_arc_field")
            diagnostics.append({"field": key, "scope": "arc"})
    retry_stages = retry_arc.get("stages")
    frozen_payloads = freeze_contract.get("frozen_stage_payloads") or {}
    mutable_fields = freeze_contract.get("mutable_fields_by_stage") or {}
    if (not isinstance(retry_stages, list) or
            len(retry_stages) != len(frozen_payloads)):
        errors.append("retry_stage_count_changed")
    else:
        arc_id = str(freeze_contract.get("arc_id") or "E1")
        for index, stage in enumerate(retry_stages, 1):
            stage_id = f"{arc_id}.stage_{index:02d}"
            if not isinstance(stage, Mapping):
                errors.append("retry_modified_frozen_stage")
                diagnostics.append({"stage_id": stage_id,
                                    "error": "stage_not_object"})
                continue
            allowed = set(mutable_fields.get(stage_id) or [])
            comparable = {key: value for key, value in dict(stage).items()
                          if key not in allowed}
            if canonical_sha256(comparable) != (
                    freeze_contract.get("frozen_stage_sha256") or {}).get(
                        stage_id):
                errors.append("retry_modified_frozen_stage")
                changed_fields = sorted({
                    key for key in set(comparable) |
                    set(frozen_payloads.get(stage_id) or {})
                    if comparable.get(key) !=
                    (frozen_payloads.get(stage_id) or {}).get(key)})
                diagnostics.append({
                    "stage_id": stage_id,
                    "allowed_mutable_fields": sorted(allowed),
                    "changed_frozen_fields": changed_fields,
                })
    return {"valid": not errors, "errors": sorted(set(errors)),
            "diagnostics": diagnostics,
            "freeze_version": freeze_contract.get("freeze_version")}


class StatefulOutlineArcRetryValidator:
    """Stateful two-attempt validator that enforces an arc freeze contract."""

    def __init__(
        self, *, parent_turns: Sequence[Mapping[str, Any]],
        overview: Mapping[str, Any], assigned_scene_indices: Sequence[int],
        arc_index: int,
    ) -> None:
        self.parent_turns = parent_turns
        self.overview = overview
        self.assigned_scene_indices = assigned_scene_indices
        self.arc_index = arc_index
        self.freeze_contract: dict[str, Any] | None = None

    def __call__(self, arc: Any) -> dict[str, Any]:
        validation = validate_stateful_outline_arc(
            arc, parent_turns=self.parent_turns, overview=self.overview,
            assigned_scene_indices=self.assigned_scene_indices)
        if self.freeze_contract is None:
            if (isinstance(arc, Mapping) and validation["decision"] ==
                    "stateful_outline_arc_schema_invalid"):
                self.freeze_contract = (
                    build_stateful_outline_e1_retry_freeze_contract(
                        arc, validation, arc_id=f"E{self.arc_index}"))
                validation["retry_freeze_contract"] = self.freeze_contract
            return validation
        freeze = validate_stateful_outline_e1_retry_freeze(
            arc, self.freeze_contract)
        validation["retry_freeze_validation"] = freeze
        if not freeze["valid"]:
            validation["decision"] = "stateful_outline_arc_schema_invalid"
            validation["reasons"] = sorted(set(
                list(validation.get("reasons") or []) + freeze["errors"]))
            validation.pop("arc", None)
        return validation


class StatefulOutlineE1RetryValidator(StatefulOutlineArcRetryValidator):
    """Backward-compatible E1 binding for focused tests and older callers."""

    def __init__(
        self, *, parent_turns: Sequence[Mapping[str, Any]],
        overview: Mapping[str, Any], assigned_scene_indices: Sequence[int],
    ) -> None:
        super().__init__(parent_turns=parent_turns, overview=overview,
                         assigned_scene_indices=assigned_scene_indices,
                         arc_index=1)


def build_cross_arc_stage_conflict_freeze_contract(
    *, arcs: Sequence[Mapping[str, Any]], conflict_gate: Mapping[str, Any],
    arc_id: str,
) -> dict[str, Any]:
    """Freeze every non-conflicting stage before one Planner rewrite."""
    arc_index = int(arc_id[1:]) - 1
    previous_arc = dict(arcs[arc_index])
    mutable_refs = {
        str(value) for value in conflict_gate.get("conflicting_stage_refs") or []
        if str(value).startswith(f"{arc_id}:")}
    stages = list(previous_arc.get("stages") or [])
    frozen_payloads: dict[str, Any] = {}
    frozen_sha: dict[str, str] = {}
    for position, stage in enumerate(stages, 1):
        ref = f"{arc_id}:stage_{position:02d}"
        if ref not in mutable_refs:
            frozen_payloads[ref] = dict(stage)
            frozen_sha[ref] = canonical_sha256(stage)
    other_arcs = [dict(arc) for index, arc in enumerate(arcs)
                  if index != arc_index]
    return {
        "freeze_version": STATEFUL_OUTLINE_CROSS_ARC_REPAIR_VERSION,
        "gate_version": STATEFUL_CROSS_ARC_CONFLICT_GATE_VERSION,
        "arc_id": arc_id,
        "mutable_stage_refs": sorted(mutable_refs),
        "frozen_stage_payloads": frozen_payloads,
        "frozen_stage_sha256": frozen_sha,
        "frozen_arc_fields": {
            key: value for key, value in previous_arc.items()
            if key != "stages"},
        "other_arcs_sha256": canonical_sha256(other_arcs),
        "source_arcs_sha256": canonical_sha256(list(arcs)),
    }


def validate_cross_arc_stage_conflict_freeze(
    candidate_arc: Any, freeze_contract: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    diagnostics: list[dict[str, Any]] = []
    if not isinstance(candidate_arc, Mapping):
        return {"valid": False, "errors": ["repair_arc_not_object"],
                "diagnostics": []}
    for key, value in (freeze_contract.get("frozen_arc_fields") or {}).items():
        if candidate_arc.get(key) != value:
            errors.append("repair_modified_frozen_arc_field")
            diagnostics.append({"scope": "arc", "field": key})
    stages = candidate_arc.get("stages")
    if not isinstance(stages, list) or len(stages) != 4:
        errors.append("repair_stage_count_changed")
    else:
        arc_id = str(freeze_contract["arc_id"])
        for position, stage in enumerate(stages, 1):
            ref = f"{arc_id}:stage_{position:02d}"
            expected = (freeze_contract.get("frozen_stage_payloads") or {}).get(ref)
            if expected is not None and canonical_sha256(stage) != canonical_sha256(expected):
                errors.append("retry_modified_frozen_stage")
                diagnostics.append({"stage_ref": ref,
                                    "required_sha256": canonical_sha256(expected),
                                    "reported_sha256": canonical_sha256(stage)})
    return {"valid": not errors, "errors": sorted(set(errors)),
            "diagnostics": diagnostics}


def build_cross_arc_stage_conflict_repair_suffix(
    *, arcs: Sequence[Mapping[str, Any]], conflict_gate: Mapping[str, Any],
    arc_id: str,
) -> str:
    contract = build_cross_arc_stage_conflict_freeze_contract(
        arcs=arcs, conflict_gate=conflict_gate, arc_id=arc_id)
    relevant = [row for row in conflict_gate.get("diagnostics") or []
                if str(row.get("conflicting_stage") or "").startswith(
                    f"{arc_id}:")]
    return "\n\n".join([
        "CROSS-ARC STAGE CONFLICT REPAIR",
        "The four arcs are individually valid, but the deterministic global "
        "gate found duplicate or unchained event obligations. This is one "
        "Planner rewrite of only the named conflicting stage(s).",
        "CONFLICT DIAGNOSTICS\n" + json.dumps(
            relevant, ensure_ascii=False, sort_keys=True, indent=2),
        "PROGRAM FREEZE CONTRACT\n" + json.dumps(
            contract, ensure_ascii=False, sort_keys=True, indent=2),
        "Preserve the arc summary, anchors, stage count, stage order, and every "
        "stage not listed in mutable_stage_refs byte-for-byte. Do not alter "
        "another arc. Rewrite only the conflicting stage(s).",
        "One base fact may be completed only once. If a conflicting stage "
        "really describes a later consequence, set stage_role=derived_state, "
        "reuse the exact event_identity_key, set predecessor_stage_ref to the "
        "exact earlier stage, make pre_state equal that predecessor's "
        "target_post_state, and give it a genuinely new target_post_state. Its "
        "marker must state only the new derived state and must not repeat the "
        "base completion. Otherwise choose a genuinely different event with a "
        "new identity key. Do not delete a stage or weaken any completion gate.",
        "Return the complete repaired arc JSON only; no Markdown.",
    ])


class CrossArcStageConflictRepairValidator:
    """Validate one conflict-arc rewrite against local and global contracts."""

    def __init__(
        self, *, parent_turns: Sequence[Mapping[str, Any]],
        overview: Mapping[str, Any], assigned_scene_indices: Sequence[int],
        arcs: Sequence[Mapping[str, Any]], conflict_gate: Mapping[str, Any],
        arc_id: str,
    ) -> None:
        self.parent_turns = parent_turns
        self.overview = overview
        self.assigned_scene_indices = assigned_scene_indices
        self.arcs = [deepcopy(dict(arc)) for arc in arcs]
        self.arc_id = arc_id
        self.arc_index = int(arc_id[1:]) - 1
        self.freeze_contract = build_cross_arc_stage_conflict_freeze_contract(
            arcs=arcs, conflict_gate=conflict_gate, arc_id=arc_id)

    def __call__(self, candidate_arc: Any) -> dict[str, Any]:
        local = validate_stateful_outline_arc(
            candidate_arc, parent_turns=self.parent_turns,
            overview=self.overview,
            assigned_scene_indices=self.assigned_scene_indices)
        if local["decision"] != "stateful_outline_arc_accept":
            return local
        freeze = validate_cross_arc_stage_conflict_freeze(
            candidate_arc, self.freeze_contract)
        if not freeze["valid"]:
            return {
                "decision": "stateful_outline_cross_arc_repair_invalid",
                "reasons": freeze["errors"],
                "freeze_validation": freeze,
            }
        candidate_arcs = deepcopy(self.arcs)
        candidate_arcs[self.arc_index] = dict(local["arc"])
        global_gate = cross_arc_stage_conflict_gate(candidate_arcs)
        if not global_gate["valid"]:
            return {
                "decision": "stateful_outline_cross_arc_repair_invalid",
                "reasons": global_gate["errors"],
                "freeze_validation": freeze,
                "cross_arc_conflict_gate": global_gate,
            }
        return {
            "decision": "stateful_outline_cross_arc_repair_accept",
            "reasons": [], "arc": dict(local["arc"]),
            "freeze_validation": freeze,
            "cross_arc_conflict_gate": global_gate,
        }


def build_stateful_outline_e1_repair_suffix(
    *, previous_arc: Mapping[str, Any], validation: Mapping[str, Any],
    maximum_anchors: int = 4, arc_id: str = "E1",
) -> str:
    """Build a one-shot E1 repair that freezes valid semantic fields."""
    reasons = [str(reason) for reason in validation.get("reasons") or []]
    error_feedback = arc_e1_retry_error_feedback(
        previous_arc, validation, maximum_anchors=maximum_anchors)
    freeze_contract = build_stateful_outline_e1_retry_freeze_contract(
        previous_arc, validation, maximum_anchors=maximum_anchors,
        arc_id=arc_id)
    anchors = previous_arc.get("parent_anchor_aliases")
    legal_anchors = (isinstance(anchors, list) and 1 <= len(anchors) <=
                     maximum_anchors and len(anchors) == len(set(anchors)))
    details: list[str] = [
        "Actual validation errors: " + json.dumps(reasons),
        "Complete field-level error set: " + json.dumps(
            error_feedback, ensure_ascii=False, sort_keys=True),
        "Program-enforced retry freeze contract: " + json.dumps(
            freeze_contract, ensure_ascii=False, sort_keys=True),
    ]
    if ("completion_evidence_phrase_nonterminal" in reasons and not any(
            row["code"] == "completion_evidence_phrase_nonterminal"
            for row in error_feedback)):
        details.append(
            "At least one completion_evidence_phrase is non-terminal; inspect "
            "all four stages and replace only invalid completion evidence.")
    if legal_anchors:
        details.append(
            "The existing parent_anchor_aliases are valid and frozen: " +
            json.dumps(anchors) + ". Preserve this exact list and order.")
    else:
        details.append(
            f"parent_anchor_aliases currently has "
            f"{len(anchors) if isinstance(anchors, list) else 'an invalid'} "
            f"value; repair it to 1-{maximum_anchors} existing aliases.")
    return "\n\n".join([
        f"{arc_id} DIRECTED COMPLETION-CONTRACT REPAIR REQUIRED",
        f"The previous {arc_id} object parsed as JSON but failed the frozen "
        f"{arc_id} "
        "interface. This is the single permitted retry.",
        "\n".join(details),
        "For every mandatory terminal stage, completion_status must be "
        "achieved, completion_temporality must be current_or_past, and "
        "completion_conditionality must be unconditional. Repair only a typed "
        "field explicitly named in the field-level error set.",
        "A completion evidence phrase is a short canonical marker, not an "
        "explanation. It must begin with I or We, name the concrete result, "
        f"contain at most {MAX_CANONICAL_MARKER_WORDS} words and "
        f"{MAX_CANONICAL_MARKER_CHARS} characters, and be directly speakable. "
        "Move reasons, subordinate conditions, and psychological detail to "
        "the later Realizer dialogue.",
        "Do not begin a canonical marker with a future plan or condition: "
        "'I might...', 'I will...', 'If...', 'When...', and 'Once...' are "
        "invalid. Valid compact examples include 'I have reflected on what "
        "happened with my dad.' and 'I still stand by my spending values.'.",
        "The structured completion fields freeze the target contract. The "
        "downstream Realizer still independently reports mentioned, "
        "in_progress, completed, failed, or carry_forward based on dialogue "
        "evidence; the Planner fields do not bypass that gate.",
        f"Use at most {maximum_anchors} arc anchors. Do not add any new anchor. "
        "When the existing anchor list is valid, preserve it exactly; never "
        "truncate or rewrite it locally.",
        f"Regenerate the complete {arc_id} JSON object. Preserve the four-stage "
        "count, people, event goals, order, summary, descriptions, object_id, "
        "state_key, pre_state, target_post_state, allowed_post_states, "
        "terminal_post_states, mandatory_resolution, authorized speakers, "
        "every structured completion field not explicitly listed as invalid, "
        "and "
        "all already-valid fields and every already-valid evidence phrase. "
        "Change only fields explicitly named in the complete field-level error "
        "set: compress an over-limit anchor list and replace only each listed "
        "invalid evidence phrase with a compact canonical marker naming its "
        "result. Do not rewrite any other stage or evidence.",
        "Return the complete legal JSON only, with no Markdown.",
        f"PREVIOUS {arc_id} ARC JSON\n" + json.dumps(
            previous_arc, ensure_ascii=False, sort_keys=True, indent=2),
    ])


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _strings(value: Any, *, minimum: int = 1, maximum: int | None = None) -> bool:
    return (isinstance(value, list) and len(value) >= minimum and
            (maximum is None or len(value) <= maximum) and
            all(_text(item) for item in value) and len(value) == len(set(value)))


def _first_person_marker(value: str) -> bool:
    folded = str(value).strip().casefold()
    return (folded.startswith(("i ", "i'", "i\u2019", "we ", "we'", "we\u2019"))
            and _AMBIGUOUS_MARKER_PREFIX.search(str(value)) is None)


def canonical_completion_marker_errors(value: Any) -> list[str]:
    """Validate only the intentionally narrow canonical marker interface."""
    if not _text(value):
        return ["completion_evidence_phrases"]
    marker = str(value).strip()
    errors: list[str] = []
    words = re.findall(r"\b[\w'\u2019-]+\b", marker)
    if (len(marker) > MAX_CANONICAL_MARKER_CHARS or
            len(words) > MAX_CANONICAL_MARKER_WORDS or "\n" in marker or
            len(re.findall(r"[.!?]", marker)) > 1):
        errors.append("completion_evidence_phrase_not_canonical_short")
    if not _first_person_marker(marker):
        errors.append("completion_evidence_phrase_not_speakable")
    if terminal_evidence_is_future_or_conditional(marker):
        errors.append("completion_evidence_phrase_nonterminal")
    if _INTERFACE_NONTERMINAL_EVIDENCE_PREFIX.search(marker):
        errors.append("completion_evidence_phrase_nonterminal")
    return sorted(set(errors))


def structured_completion_contract_errors(stage: Mapping[str, Any]) -> list[str]:
    """Cross-check typed completion fields against one canonical marker."""
    errors: list[str] = []
    status = stage.get("completion_status")
    temporality = stage.get("completion_temporality")
    conditionality = stage.get("completion_conditionality")
    if status not in COMPLETION_STATUSES:
        errors.append("completion_status_invalid")
    if temporality not in COMPLETION_TEMPORALITIES:
        errors.append("completion_temporality_invalid")
    if conditionality not in COMPLETION_CONDITIONALITIES:
        errors.append("completion_conditionality_invalid")
    if stage.get("mandatory_resolution") is True:
        if status != "achieved":
            errors.append("mandatory_terminal_completion_not_achieved")
        if temporality != "current_or_past":
            errors.append("mandatory_terminal_completion_not_current_or_past")
        if conditionality != "unconditional":
            errors.append("mandatory_terminal_completion_not_unconditional")
    if (status == "achieved" and
            (temporality == "future" or conditionality == "conditional")):
        errors.append("completion_contract_inconsistent")
    phrases = stage.get("completion_evidence_phrases")
    if not _strings(phrases, minimum=1, maximum=1):
        errors.append("completion_evidence_phrases")
    else:
        errors.extend(canonical_completion_marker_errors(phrases[0]))
        if (conditionality == "unconditional" and
                terminal_evidence_is_future_or_conditional(phrases[0])):
            errors.append("completion_marker_contract_inconsistent")
    return sorted(set(errors))


def validate_raw_stateful_outline(
    raw: Any, *, parent_turns: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != set(
            RAW_STATEFUL_OUTLINE_SCHEMA["required"]):
        return {"decision": "stateful_outline_schema_invalid",
                "reasons": ["top_level_fields"]}
    errors: list[str] = []
    aliases = {f"T{position:04d}": turn for position, turn in
               enumerate(parent_turns, 1)}
    known_people = {str(turn.get("speaker")) for turn in parent_turns}
    for key in ("title", "expansion_objective", "time_window"):
        if not _text(raw.get(key)):
            errors.append(key)
    new_people = raw.get("allowed_new_characters")
    new_names: list[str] = []
    if not isinstance(new_people, list) or len(new_people) > 2:
        errors.append("allowed_new_characters")
        new_people = []
    for row in new_people:
        if (not isinstance(row, dict) or set(row) != {"name", "role"} or
                not _text(row.get("name")) or not _text(row.get("role"))):
            errors.append("new_character_schema")
        else:
            new_names.append(str(row["name"]).strip())
    locations = raw.get("allowed_locations")
    if not _strings(locations, minimum=3, maximum=3):
        errors.append("allowed_locations")
        locations = []
    arcs = raw.get("event_arcs")
    if not isinstance(arcs, list) or len(arcs) != 4:
        errors.append("event_arcs")
        arcs = []
    seen_instances: set[tuple[str, str]] = set()
    state_keys: set[str] = set()
    terminal_count = 0
    for arc in arcs:
        if (not isinstance(arc, dict) or
                set(arc) != {"summary", "parent_anchor_aliases", "stages"} or
                not _text(arc.get("summary"))):
            errors.append("event_arc_schema")
            continue
        support = arc.get("parent_anchor_aliases")
        if (not _strings(support, minimum=1, maximum=4) or
                any(alias not in aliases for alias in support)):
            errors.append("event_arc_parent_anchors")
        stages = arc.get("stages")
        if not isinstance(stages, list) or len(stages) != 4:
            errors.append("event_arc_stages")
            continue
        for stage in stages:
            if not isinstance(stage, dict) or set(stage) != STAGE_FIELDS:
                errors.append("stateful_stage_fields")
                continue
            for key in ("description", "object_id", "state_key", "pre_state",
                        "event_identity_key", "target_post_state",
                        "completion_event_type"):
                if not _text(stage.get(key)):
                    errors.append("stateful_stage_text")
            if (not _KEY.fullmatch(str(stage.get("object_id") or "")) or
                    not _KEY.fullmatch(str(stage.get("state_key") or "")) or
                    not _KEY.fullmatch(str(stage.get("event_identity_key") or ""))):
                errors.append("stateful_stage_key")
            if stage.get("stage_role") not in STAGE_ROLES:
                errors.append("stage_role_invalid")
            if (stage.get("stage_role") == "base_fact_completion" and
                    stage.get("predecessor_stage_ref") is not None):
                errors.append("base_fact_has_predecessor")
            if (stage.get("stage_role") == "derived_state" and
                    not _text(stage.get("predecessor_stage_ref"))):
                errors.append("derived_state_predecessor_invalid")
            identity = (str(stage.get("object_id")), str(stage.get("state_key")))
            if identity in seen_instances:
                errors.append("duplicate_state_instance")
            seen_instances.add(identity)
            state_keys.add(str(stage.get("state_key")))
            allowed = stage.get("allowed_post_states")
            terminal = stage.get("terminal_post_states")
            if not _strings(allowed, minimum=1, maximum=4):
                errors.append("allowed_post_states")
                allowed = []
            if (not _strings(terminal, minimum=1, maximum=4) or
                    not set(terminal) <= set(allowed)):
                errors.append("terminal_post_states")
                terminal = []
            if stage.get("target_post_state") not in allowed:
                errors.append("target_post_state")
            if stage.get("target_post_state") not in terminal:
                errors.append("target_post_state_not_terminal")
            if stage.get("pre_state") == stage.get("target_post_state"):
                errors.append("no_state_change")
            if stage.get("target_post_state") in terminal:
                terminal_count += 1
            if stage.get("mandatory_resolution") is not True:
                errors.append("mandatory_resolution")
            allowed_people = known_people | set(new_names)
            if not _strings(stage.get("authorized_result_speakers"), minimum=1,
                            maximum=3):
                errors.append("authorized_result_speakers")
            errors.extend(structured_completion_contract_errors(stage))
            if any(name not in allowed_people for name in
                   stage.get("authorized_result_speakers") or []):
                errors.append("unauthorized_result_speaker")
            handoffs = stage.get("outcome_handoffs")
            if not isinstance(handoffs, list) or len(handoffs) > 1:
                errors.append("outcome_handoffs")
                handoffs = []
            for handoff in handoffs:
                if (not isinstance(handoff, dict) or
                        set(handoff) != HANDOFF_FIELDS or
                        any(not _text(handoff.get(key)) for key in (
                            "handoff_key", "description", "owner",
                            "created_on_post_state")) or
                        not _KEY.fullmatch(str(handoff.get("handoff_key") or "")) or
                        handoff.get("created_on_post_state") not in allowed or
                        not _strings(handoff.get("resolution_evidence_phrases"),
                                     minimum=1, maximum=3) or
                        not _strings(handoff.get("authorized_result_speakers"),
                                     minimum=1, maximum=3) or
                        handoff.get("owner") not in allowed_people or
                        any(name not in allowed_people for name in
                            handoff.get("authorized_result_speakers") or [])):
                    errors.append("outcome_handoff_schema")
    conflict_gate = cross_arc_stage_conflict_gate(arcs)
    errors.extend(f"cross_arc:{value}" for value in conflict_gate["errors"])
    if len(state_keys) < 6:
        errors.append("insufficient_distinct_state_keys")
    if terminal_count < 8:
        errors.append("insufficient_terminal_outcomes")

    scenes = raw.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != 10:
        errors.append("scenes")
        scenes = []
    allowed_people = known_people | set(new_names)
    roles = []
    for scene in scenes:
        if not isinstance(scene, dict) or set(scene) != SCENE_FIELDS:
            errors.append("scene_fields")
            continue
        if (not _text(scene.get("purpose")) or
                scene.get("scene_role") not in {
                    "advance_parent_thread", "create_new_state_chain"} or
                scene.get("location") not in locations or
                not _strings(scene.get("participants"), minimum=2, maximum=4) or
                any(name not in allowed_people for name in
                    scene.get("participants") or []) or
                not isinstance(scene.get("parent_anchor_aliases"), list) or
                len(scene.get("parent_anchor_aliases")) > 3 or
                any(alias not in aliases for alias in
                    scene.get("parent_anchor_aliases") or [])):
            errors.append("scene_schema")
        roles.append(scene.get("scene_role"))
    parent_role_count = roles.count("advance_parent_thread")
    if not 4 <= parent_role_count <= 6:
        errors.append("scene_role_balance")
    if scenes and not scenes[0].get("parent_anchor_aliases"):
        errors.append("opening_scene_missing_parent_anchor")
    if len(arcs) == 4 and len(scenes) == 10:
        for scene, schedule in zip(scenes, STANDARD_STAGE_SCHEDULE):
            if not isinstance(scene, dict):
                continue
            participants = set(scene.get("participants") or [])
            for arc_index, stage_index in schedule:
                arc_stages = (arcs[arc_index].get("stages") or []) \
                    if isinstance(arcs[arc_index], dict) else []
                if len(arc_stages) <= stage_index:
                    continue
                stage = arc_stages[stage_index]
                if (isinstance(stage, dict) and not participants.intersection(
                        stage.get("authorized_result_speakers") or [])):
                    errors.append("scene_missing_stage_result_speaker")
    return {"decision": "stateful_outline_accept" if not errors else
            "stateful_outline_schema_invalid", "reasons": sorted(set(errors)),
            **({"raw_plan": raw} if not errors else {})}


def normalize_stateful_outline(
    raw: Mapping[str, Any], *, parent_turns: Sequence[Mapping[str, Any]],
    root_id: str = "unified-parent", sample_id: str | None = None,
    dialogue_hash: str | None = None, parent_session_count: int | None = None,
    outline_id_prefix: str | None = None, round_index: int = 1,
) -> dict[str, Any]:
    check = validate_raw_stateful_outline(raw, parent_turns=parent_turns)
    if check["decision"] != "stateful_outline_accept":
        raise ValueError("cannot_normalize_invalid_stateful_outline")
    alias_index = {f"T{position:04d}": dict(turn) for position, turn in
                   enumerate(parent_turns, 1)}
    referenced = {alias for arc in raw["event_arcs"]
                  for alias in arc["parent_anchor_aliases"]}
    referenced.update(alias for scene in raw["scenes"]
                      for alias in scene["parent_anchor_aliases"])
    ordered = [alias for alias in alias_index if alias in referenced]
    anchor_ids = {alias: f"P{position}" for position, alias in
                  enumerate(ordered, 1)}
    anchors = [{"anchor_id": anchor_ids[alias],
                "turn_id": alias_index[alias]["turn_id"],
                "session_id": alias_index[alias].get("session_id"),
                "speaker": alias_index[alias]["speaker"],
                "text": alias_index[alias]["text"]} for alias in ordered]
    arcs = []
    for arc_index, arc in enumerate(raw["event_arcs"], 1):
        stages = []
        for stage in arc["stages"]:
            stages.append({
                **dict(stage),
                "state_instance_id": (
                    f"{stage['object_id']}:{stage['state_key']}:v1"),
                "version": 1, "predecessor_state_instance_id": None,
            })
        arcs.append({"event_id": f"E{arc_index}",
                     "summary": arc["summary"].strip(),
                     "parent_anchor_ids": [anchor_ids[value] for value in
                                           arc["parent_anchor_aliases"]],
                     "stages": stages})
    scenes = [{"scene_id": f"S{position}",
               "purpose": scene["purpose"].strip(),
               "participants": list(scene["participants"]),
               "required_parent_anchor_ids": [anchor_ids[value] for value in
                                               scene["parent_anchor_aliases"]],
               "session_location": scene["location"],
               "scene_role": scene["scene_role"],
               "target_dialogue_token_range": [320, 480],
               "target_turn_range": [7, 13]}
              for position, scene in enumerate(raw["scenes"], 1)]
    actual_dialogue_hash = dialogue_content_hash(
        [dict(turn) for turn in parent_turns])
    if dialogue_hash is not None and dialogue_hash != actual_dialogue_hash:
        raise ValueError("stateful_outline_parent_dialogue_hash_mismatch")
    actual_session_count = len({str(turn.get("session_id") or "")
                                for turn in parent_turns})
    if (parent_session_count is not None and
            parent_session_count != actual_session_count):
        raise ValueError("stateful_outline_parent_session_count_mismatch")
    prefix = str(outline_id_prefix or root_id).strip()
    if not prefix or not str(root_id).strip() or round_index < 1:
        raise ValueError("stateful_outline_dynamic_identity_invalid")
    outline = {
        "schema_version": STATEFUL_GRAPH_OUTLINE_SCHEMA_VERSION,
        "completion_contract_version": STATEFUL_COMPLETION_CONTRACT_VERSION,
        "outline_id": f"{prefix}:stateful-round:{round_index:02d}",
        "source_parent": {
            "root_id": str(root_id),
            "sample_id": str(sample_id or root_id),
            "dialogue_hash": actual_dialogue_hash,
            "parent_turn_count": len(parent_turns),
            "parent_session_count": actual_session_count,
            "core_speakers": sorted({str(turn["speaker"]) for turn in
                                     parent_turns})},
        "title": raw["title"].strip(),
        "expansion_objective": raw["expansion_objective"].strip(),
        "allowed_new_characters": list(raw["allowed_new_characters"]),
        "allowed_locations": list(raw["allowed_locations"]),
        "time_window": raw["time_window"].strip(),
        "parent_anchors": anchors,
        "planned_event_arcs": arcs, "scene_plan": scenes,
        "target": {"scene_count": 10, "minimum_accepted_child_tokens": 3000},
        "raw_plan_sha256": canonical_sha256(raw),
    }
    errors = validate_stateful_outline(outline)
    if errors:
        raise ValueError("normalized_stateful_outline_invalid:" + ",".join(errors))
    return outline


def build_dynamic_stateful_planner_input(
    *, parent_turns: Sequence[Mapping[str, Any]], root_id: str, sample_id: str,
    lineage_id: str, run_id: str, dialogue_hash: str,
    output_directory: str, outline_id_prefix: str | None = None,
    round_index: int = 1,
) -> dict[str, Any]:
    """Bind dataset-neutral run identity to the unchanged Planner prompt."""
    turns = [dict(turn) for turn in parent_turns]
    if not turns or any(turn.get("partition") != "parent" for turn in turns):
        raise ValueError("dynamic_planner_parent_contract_invalid")
    turn_ids = [str(turn.get("turn_id") or "") for turn in turns]
    session_ids = [str(turn.get("session_id") or "") for turn in turns]
    if (any(not value for value in turn_ids + session_ids) or
            len(turn_ids) != len(set(turn_ids))):
        raise ValueError("dynamic_planner_parent_identity_invalid")
    actual_hash = dialogue_content_hash(turns)
    if actual_hash != dialogue_hash:
        raise ValueError("dynamic_planner_dialogue_hash_mismatch")
    identities = [root_id, sample_id, lineage_id, run_id, output_directory]
    if any(not isinstance(value, str) or not value.strip() for value in identities):
        raise ValueError("dynamic_planner_run_identity_invalid")
    if round_index < 1:
        raise ValueError("dynamic_planner_round_index_invalid")
    prompt = build_stateful_outline_overview_prompt(parent_turns=turns)
    return {
        "input_version": STATEFUL_DYNAMIC_PLANNER_INPUT_VERSION,
        "root_id": root_id, "sample_id": sample_id,
        "lineage_id": lineage_id, "run_id": run_id,
        "round_index": round_index,
        "outline_id_prefix": str(outline_id_prefix or root_id),
        "output_directory": output_directory,
        "dialogue_hash": actual_hash,
        "parent_turn_count": len(turns),
        "parent_session_count": len(set(session_ids)),
        "parent_turn_ids": turn_ids,
        "parent_session_ids": list(dict.fromkeys(session_ids)),
        "overview_prompt": prompt,
        "overview_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "overview_schema_sha256": canonical_sha256(
            RAW_STATEFUL_OUTLINE_SCHEMA),
    }


def build_stateful_outline_prompt(
    *, parent_turns: Sequence[Mapping[str, Any]],
) -> str:
    dialogue = "\n".join(
        f"[T{position:04d}] {turn['speaker']}: {turn['text']}"
        for position, turn in enumerate(parent_turns, 1))
    speakers = ", ".join(sorted({str(turn["speaker"]) for turn in parent_turns}))
    rules = [
        "Create one ten-scene continuation outline from the frozen parent dialogue.",
        "Return planning JSON only. Do not write dialogue, questions, options, answers, or solver content.",
        "Create exactly four event arcs with exactly four typed stages each.",
        "Every stage must change one specific versioned state object from pre_state to target_post_state.",
        "A stage is not complete merely because it is mentioned, planned, prepared, or waiting for approval.",
        "Make target outcomes explicit and observable; rejection, failure, cancellation, and revision are valid outcomes.",
        "Use at least six distinct state_key values and at least eight terminal target outcomes.",
        "Do not reuse the same object_id/state_key pair in two stages.",
        "Every mandatory terminal stage must set completion_status=achieved, completion_temporality=current_or_past, and completion_conditionality=unconditional.",
        "completion_evidence_phrases contain one short canonical marker (at most 18 words and 160 characters) beginning with I or We; keep explanations and conditions in the later dialogue.",
        "A canonical marker must not begin with future or conditional wording such as I will, I might, If, When, or Once.",
        "Each stage may create at most one concrete handoff; use [] if none.",
        "Use only T#### aliases from the parent dialogue. Do not copy parent quotes into the output.",
        "Exactly 4 to 6 scenes must have scene_role advance_parent_thread; the rest create_new_state_chain.",
        "Across the ten scenes, include genuine decisions/results and new event chains, not repeated proposal preparation.",
        f"Existing speakers: {speakers}. Declare at most two new named characters.",
        "All identifiers object_id, state_key, and handoff_key must be lowercase snake_case.",
        "Return one JSON object matching the schema exactly, with no Markdown.",
    ]
    return "\n\n".join([
        f"PROMPT_VERSION: {STATEFUL_OUTLINE_PLANNER_VERSION}",
        "RULES\n- " + "\n- ".join(rules),
        "FROZEN PARENT DIALOGUE\n" + dialogue,
        "OUTPUT JSON SCHEMA\n" + json.dumps(
            RAW_STATEFUL_OUTLINE_SCHEMA, ensure_ascii=False,
            sort_keys=True, indent=2),
        "REQUIRED STAGE KEYS\n" + json.dumps(sorted(STAGE_FIELDS)),
        "REQUIRED OUTCOME HANDOFF KEYS\n" + json.dumps(sorted(HANDOFF_FIELDS)),
        "REQUIRED SCENE KEYS\n" + json.dumps(sorted(SCENE_FIELDS)),
    ])
