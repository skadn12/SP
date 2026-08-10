"""Deterministic contracts for the rolling outline-Graph 30k canary.

The Challenger chooses continuation facts and scene structure.  This module only
validates identifiers and materializes that API output into the already-audited
``stage3_expansion_outline_v1`` representation consumed by the Graph Realizer.
It deliberately contains no QA generation or semantic difficulty heuristics.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from .input_adapter import dialogue_content_hash


CANARY_VERSION = "rolling_outline_graph_30k_canary_v1"
OUTLINE_PLANNER_PROMPT_VERSION = "rolling_outline_graph_planner_derived_anchors_v1"
OUTLINE_PLANNER_SCHEMA_VERSION = "rolling_outline_graph_plan_derived_anchors_v1"
TARGET_DIALOGUE_TOKENS = 30_000
MAX_EXPANSION_ROUNDS = 5
SCENES_PER_ROUND = 10
EVENT_ARCS_PER_ROUND = 4
STAGES_PER_ARC = 4
TARGET_TOKENS_PER_SCENE = 420


RAW_OUTLINE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "title", "expansion_objective",
        "allowed_new_characters", "allowed_locations", "time_window",
        "event_arcs", "scenes",
    ],
    "properties": {
        "title": {"type": "string"},
        "expansion_objective": {"type": "string"},
        "allowed_new_characters": {
            "type": "array", "minItems": 0, "maxItems": 2,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["name", "role"],
                "properties": {
                    "name": {"type": "string"}, "role": {"type": "string"},
                },
            },
        },
        "allowed_locations": {
            "type": "array", "minItems": 3, "maxItems": 3,
            "uniqueItems": True, "items": {"type": "string"},
        },
        "time_window": {"type": "string"},
        "event_arcs": {
            "type": "array", "minItems": 4, "maxItems": 4,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["summary", "stages", "anchor_aliases"],
                "properties": {
                    "summary": {"type": "string"},
                    "stages": {
                        "type": "array", "minItems": 4, "maxItems": 4,
                        "items": {"type": "string"},
                    },
                    "anchor_aliases": {
                        "type": "array", "minItems": 1, "maxItems": 4,
                        "uniqueItems": True, "items": {"type": "string"},
                    },
                },
            },
        },
        "scenes": {
            "type": "array", "minItems": 10, "maxItems": 10,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["purpose", "participants", "anchor_aliases",
                             "handoff_requirement"],
                "properties": {
                    "purpose": {"type": "string"},
                    "participants": {
                        "type": "array", "minItems": 2, "maxItems": 4,
                        "uniqueItems": True, "items": {"type": "string"},
                    },
                    "anchor_aliases": {
                        "type": "array", "maxItems": 3,
                        "uniqueItems": True, "items": {"type": "string"},
                    },
                    "handoff_requirement": {"type": "string"},
                },
            },
        },
    },
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_raw_outline_plan(
    raw: Any, *, cumulative_turns: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Strict structural and identifier validation; no story-quality scoring."""
    if not isinstance(raw, dict) or set(raw) != set(RAW_OUTLINE_SCHEMA["required"]):
        return {"decision": "rolling_outline_schema_invalid", "reasons": ["fields"]}
    errors: list[str] = []
    alias_index = {f"T{position:04d}": turn for position, turn in
                   enumerate(cumulative_turns, 1)}
    known_people = {str(turn.get("speaker")) for turn in cumulative_turns}
    for key in ("title", "expansion_objective", "time_window"):
        if not _text(raw.get(key)):
            errors.append(key)
    new_people = raw.get("allowed_new_characters")
    new_names: list[str] = []
    if not isinstance(new_people, list) or not 0 <= len(new_people) <= 2:
        errors.append("allowed_new_characters")
        new_people = []
    for row in new_people:
        if (not isinstance(row, dict) or set(row) != {"name", "role"} or
                not _text(row.get("name")) or not _text(row.get("role"))):
            errors.append("allowed_new_character_schema")
            continue
        new_names.append(row["name"].strip())
    locations = raw.get("allowed_locations")
    if (not isinstance(locations, list) or len(locations) != 3 or
            len(set(locations)) != 3 or any(not _text(value) for value in locations)):
        errors.append("allowed_locations")
    arcs = raw.get("event_arcs")
    if not isinstance(arcs, list) or len(arcs) != EVENT_ARCS_PER_ROUND:
        errors.append("event_arcs")
        arcs = []
    for arc in arcs:
        if (not isinstance(arc, dict) or set(arc) != {
                "summary", "stages", "anchor_aliases"} or
                not _text(arc.get("summary"))):
            errors.append("event_arc_schema")
            continue
        stages = arc.get("stages")
        support = arc.get("anchor_aliases")
        if (not isinstance(stages, list) or len(stages) != STAGES_PER_ARC or
                any(not _text(value) for value in stages)):
            errors.append("event_arc_stages")
        if (not isinstance(support, list) or not 1 <= len(support) <= 4 or
                len(set(support)) != len(support) or
                any(value not in alias_index for value in support)):
            errors.append("event_arc_anchors")
    scenes = raw.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != SCENES_PER_ROUND:
        errors.append("scenes")
        scenes = []
    allowed_people = known_people | set(new_names)
    for index, scene in enumerate(scenes, 1):
        if (not isinstance(scene, dict) or set(scene) != {
                "purpose", "participants", "anchor_aliases",
                "handoff_requirement"} or not _text(scene.get("purpose")) or
                not _text(scene.get("handoff_requirement"))):
            errors.append("scene_schema")
            continue
        participants = scene.get("participants")
        support = scene.get("anchor_aliases")
        if (not isinstance(participants, list) or not 2 <= len(participants) <= 4 or
                len(set(participants)) != len(participants) or
                any(not _text(value) or value not in allowed_people
                    for value in participants)):
            errors.append("scene_participants")
        if (not isinstance(support, list) or len(support) > 3 or
                len(set(support)) != len(support) or
                any(value not in alias_index for value in support)):
            errors.append("scene_anchors")
    if scenes and not scenes[0].get("anchor_aliases"):
        errors.append("opening_scene_missing_anchor")
    return {"decision": "rolling_outline_accept" if not errors else
            "rolling_outline_schema_invalid", "reasons": sorted(set(errors)),
            **({"raw_plan": raw} if not errors else {})}


def normalize_outline_plan(
    raw: Mapping[str, Any], *, cumulative_turns: Sequence[Mapping[str, Any]],
    round_index: int,
) -> dict[str, Any]:
    check = validate_raw_outline_plan(raw, cumulative_turns=cumulative_turns)
    if check["decision"] != "rolling_outline_accept":
        raise ValueError("cannot_normalize_invalid_rolling_outline")
    alias_index = {f"T{position:04d}": dict(turn) for position, turn in
                   enumerate(cumulative_turns, 1)}
    referenced_aliases = {value for arc in raw["event_arcs"]
                          for value in arc["anchor_aliases"]}
    referenced_aliases.update(value for scene in raw["scenes"]
                              for value in scene["anchor_aliases"])
    ordered_aliases = [alias for alias in alias_index if alias in referenced_aliases]
    anchor_ids = {alias: f"P{position}" for position, alias in
                  enumerate(ordered_aliases, 1)}
    parent_anchors = [{"anchor_id": anchor_ids[alias],
        "turn_id": alias_index[alias]["turn_id"],
        "speaker": alias_index[alias]["speaker"],
        "text": alias_index[alias]["text"],
        "continuity_role": f"API-selected continuity anchor for round {round_index}."}
        for alias in ordered_aliases]
    known_people = sorted({str(turn["speaker"]) for turn in cumulative_turns})
    new_people = []
    used_participants = {name for scene in raw["scenes"]
                         for name in scene["participants"]}
    materialized_names: set[str] = set()
    for row in raw["allowed_new_characters"]:
        if (row["name"] not in used_participants or row["name"] in known_people or
                row["name"] in materialized_names):
            continue
        first = next(position for position, scene in enumerate(raw["scenes"], 1)
                     if row["name"] in scene["participants"])
        new_people.append({"name": row["name"].strip(), "role": row["role"].strip(),
                           "introduction_scene": f"S{first}"})
        materialized_names.add(row["name"])
    arcs = [{"event_id": f"E{position}", "summary": row["summary"].strip(),
             "stages": [str(value).strip() for value in row["stages"]],
             "parent_anchor_ids": [anchor_ids[value]
                                   for value in row["anchor_aliases"]]}
            for position, row in enumerate(raw["event_arcs"], 1)]
    scenes = []
    for position, row in enumerate(raw["scenes"], 1):
        scenes.append({"scene_id": f"S{position}",
            "target_child_tokens": TARGET_TOKENS_PER_SCENE,
            "acceptable_token_range": [360, 480], "target_turns": [12, 16],
            "participants": list(row["participants"]),
            "purpose": row["purpose"].strip(),
            "required_parent_anchor_ids": [anchor_ids[value]
                                           for value in row["anchor_aliases"]],
            "allowed_output": ["natural dialogue", "bounded local logistics",
                "reactions, clarification, disagreement, uncertainty, and reflection",
                "one or more planned event-arc stages"],
            "forbidden_output": ["QA questions, options, answer labels, or gold answers",
                "unplanned named characters", "major unrelated life events",
                "database-style status summaries",
                "claims of success before the corresponding event occurs"],
            "handoff_requirement": row["handoff_requirement"].strip()})
    return {
        "schema_version": "stage3_expansion_outline_v1",
        "outline_id": f"conv-41:outline:rolling:{round_index:02d}",
        "outline_role": "dialogue_expansion_plan_only",
        "source_parent": {"sample_id": "conv-41-rolling-cumulative",
            "lineage_group": "conv-41", "dialogue_hash": dialogue_content_hash(
                [dict(turn) for turn in cumulative_turns]),
            "parent_graph_hash": "not_used_for_qa_or_rolling_planning",
            "parent_turn_count": len(cumulative_turns),
            "core_speakers": known_people},
        "qa_grounding_policy": {"graph_used_to_verify_qa_gold": False,
            "graph_used_to_generate_qa_gold": False,
            "frozen_dialogue_is_future_qa_source_of_truth": True,
            "qa_generation_allowed_during_realization": False,
            "qa_fields_forbidden_in_outline": ["question", "options", "gold",
                "gold_answer", "qa_operation", "qa_evidence_turn_ids"]},
        "title": raw["title"].strip(),
        "target": {"child_token_growth": TARGET_TOKENS_PER_SCENE * SCENES_PER_ROUND,
            "scene_count": SCENES_PER_ROUND, "tokens_per_scene": TARGET_TOKENS_PER_SCENE,
            "minimum_accepted_child_tokens": 4000,
            "maximum_new_named_characters": 2},
        "parent_anchors": parent_anchors,
        "expansion_objective": raw["expansion_objective"].strip(),
        "allowed_new_characters": new_people,
        "allowed_locations": [str(value).strip() for value in raw["allowed_locations"]],
        "time_window": raw["time_window"].strip(),
        "planned_event_arcs": arcs, "scene_plan": scenes,
        "structural_complexity_targets": {"cross_scene_dependencies": 5,
            "delayed_consequences": 2, "superseded_or_revised_plans": 1,
            "role_or_visibility_boundaries": 1, "mixed_outcome_or_pending_result": 1,
            "single_scene_final_answer_summaries": 0,
            "note": "Narrative targets only; never QA gold."},
        "fact_layers": {"durable_parent_constraints": "Selected anchors remain binding.",
            "planned_child_arc": "The API-selected arcs and scenes constrain realization.",
            "local_scene_facts": "Bounded logistics may be created and ledgered.",
            "natural_language_only": "Natural reactions need not become graph facts."},
        "realizer_contract": {"generate_one_scene_per_api_call": True,
            "outline_is_read_only": True,
            "previous_context": {"include_parent_anchor_excerpts": "selected only",
                "include_previous_scene_summaries": 2,
                "include_full_previous_child_dialogue": False},
            "retry_policy": "Reject and regenerate the individual scene; never rewrite locally."},
        "naturalness_contract": {"do_not_read_graph_fields_aloud": True,
            "avoid_interview_style_question_answer_repetition": True,
            "avoid_repeating_the_same_core_fact": True,
            "allow_uncertainty_and_partial_information": True,
            "allow_scenes_with_reflection_or_preparation": True,
            "require_character_specific_reactions": True,
            "do_not_use_length_filler": True},
        "forbidden_conflicts": ["Do not claim John already holds public office.",
            "Do not overwrite established education or degree history.",
            "Do not invent guaranteed funding or an implausible immediate policy victory.",
            "Do not change Maria and John's established supportive friendship.",
            "Do not introduce unrelated travel, romance, illness, or family crises."],
        "post_realization_pipeline": ["freeze cumulative dialogue",
            "run Questioner from frozen dialogue only", "run unchanged Solver filters"],
        "rolling_round_index": round_index,
        "raw_plan_sha256": canonical_sha256(raw),
    }


def build_outline_planner_prompt(
    *, cumulative_turns: Sequence[Mapping[str, Any]], round_index: int,
) -> str:
    dialogue = "\n".join(
        f"[T{position:04d}] {turn['speaker']}: {turn['text']}"
        for position, turn in enumerate(cumulative_turns, 1))
    existing = ", ".join(sorted({str(turn["speaker"])
                                  for turn in cumulative_turns}))
    rules = [
        "Create the next bounded continuation outline for the frozen dialogue.",
        "Return planning data only; never generate QA, options, gold answers, or dialogue.",
        "Use only supplied T#### aliases in every anchor_aliases array. The program resolves the original turn IDs.",
        "Create four distinct four-stage event arcs distributed across ten scenes.",
        "The continuation must advance existing situations without repeating an already completed arc.",
        "Use cross-scene consequences, a discovered error or revised state, delayed outcomes, and realistic uncertainty.",
        "Do not design a question or optimize a particular answer. The future frozen dialogue is the sole QA source.",
        "You may introduce zero, one, or two genuinely useful named characters and exactly three plausible locations.",
        f"Existing speaker roster (never declare these as new): {existing}.",
        "All participants must be existing speakers or declared new characters.",
        "Keep established facts and relationships consistent; avoid unrelated dramatic events and filler.",
        "Return one JSON object matching the schema exactly, with no Markdown.",
    ]
    return "\n\n".join([
        f"PROMPT_VERSION: {OUTLINE_PLANNER_PROMPT_VERSION}",
        f"ROLLING ROUND: {round_index}",
        "PLANNING RULES\n- " + "\n- ".join(rules),
        "FROZEN CUMULATIVE DIALOGUE\n" + dialogue,
        "OUTPUT JSON SCHEMA\n" + canonical_json(RAW_OUTLINE_SCHEMA),
    ])


def remap_round_child_turns(
    turns: Sequence[Mapping[str, Any]], *, round_index: int,
) -> list[dict[str, Any]]:
    result = []
    for position, turn in enumerate(turns, 1):
        local_session = str(turn.get("session_id") or "session_00")
        result.append({**dict(turn), "turn_id": f"R{round_index:02d}C{position:04d}",
            "session_id": f"round_{round_index:02d}_{local_session}"})
    return result


def remap_baseline_turns(turns: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    child_position = 0
    for turn in turns:
        row = dict(turn)
        turn_id = str(row.get("turn_id") or "")
        if turn_id.startswith("C"):
            child_position += 1
            row["turn_id"] = f"B00C{child_position:04d}"
            row["session_id"] = f"baseline_{row.get('session_id') or 'child'}"
        else:
            row["session_id"] = str(row.get("session_id") or
                                    (turn_id.split(":", 1)[0] if ":" in turn_id
                                     else "parent"))
        result.append(row)
    if len({row["turn_id"] for row in result}) != len(result):
        raise ValueError("baseline_turn_id_collision")
    return result
