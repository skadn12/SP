"""Contracts for expanding a frozen dialogue at explicit temporal gaps.

The API chooses the narrative arcs, each scene's arc, and insertion gaps. Local
code validates identifiers/order and maps each arc's ordered stages onto the
API-selected scenes for that arc. Original dialogue turns remain immutable and
are preserved in their exact order.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from .graph_expansion import (
    GRAPH_STAGE_ACCOUNTING_COVERAGE_ONLY,
    graph_contract_sha256,
)
from .input_adapter import dialogue_content_hash
from .prompts import build_graph_realizer_prompt


TEMPORAL_INSERTION_CANARY_VERSION = (
    "temporal_insertion_outline_expansion_canary_v3"
)
TEMPORAL_OUTLINE_PROMPT_VERSION = (
    "temporal_insertion_outline_planner_v3_named_participants_v1"
)
TEMPORAL_OUTLINE_SCHEMA_VERSION = "temporal_insertion_outline_plan_v3"
TEMPORAL_GRAPH_CONTRACT_VERSION = "temporal_insertion_graph_scene_contract_v3"
TEMPORAL_REALIZER_PROMPT_VERSION = "temporal_insertion_graph_realizer_v3"

SCENE_COUNT = 10
ARC_COUNT = 4
STAGES_PER_ARC = 4


TEMPORAL_OUTLINE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "title",
        "expansion_objective",
        "allowed_new_characters",
        "event_arcs",
        "scenes",
    ],
    "properties": {
        "title": {"type": "string"},
        "expansion_objective": {"type": "string"},
        "allowed_new_characters": {
            "type": "array",
            "minItems": 0,
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
        "event_arcs": {
            "type": "array",
            "minItems": ARC_COUNT,
            "maxItems": ARC_COUNT,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["summary", "stages"],
                "properties": {
                    "summary": {"type": "string"},
                    "stages": {
                        "type": "array",
                        "minItems": STAGES_PER_ARC,
                        "maxItems": STAGES_PER_ARC,
                        "items": {"type": "string"},
                    },
                },
            },
        },
        "scenes": {
            "type": "array",
            "minItems": SCENE_COUNT,
            "maxItems": SCENE_COUNT,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "purpose",
                    "participants",
                    "location",
                    "arc_index",
                    "anchor_aliases",
                    "handoff_requirement",
                ],
                "properties": {
                    "purpose": {"type": "string"},
                    "participants": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 4,
                        "uniqueItems": True,
                        "items": {"type": "string"},
                    },
                    "location": {"type": "string"},
                    "arc_index": {"type": "integer", "minimum": 1,
                                  "maximum": ARC_COUNT},
                    "anchor_aliases": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "uniqueItems": True,
                        "items": {"type": "string",
                                  "pattern": "^T[0-9]{4}$"},
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


def ordered_session_ids(
        turns: Sequence[Mapping[str, Any]],
) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for turn in turns:
        session_id = str(turn.get("session_id") or "")
        if not session_id or session_id in seen:
            continue
        result.append(session_id)
        seen.add(session_id)
    return result


def insertion_gap_index(
        turns: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], int]:
    sessions = ordered_session_ids(turns)
    if not sessions:
        return {}
    pairs = {(sessions[index], sessions[index + 1]): index + 1
             for index in range(len(sessions) - 1)}
    pairs[(sessions[-1], "END")] = len(sessions)
    return pairs


def validate_temporal_outline_plan(
        raw: Any, *, original_turns: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate structure, identifiers, temporal placement, and stage coverage."""
    required = set(TEMPORAL_OUTLINE_SCHEMA["required"])
    if not isinstance(raw, dict) or set(raw) != required:
        return {"decision": "temporal_outline_schema_invalid",
                "reasons": ["fields"]}

    errors: list[str] = []
    alias_index = {f"T{position:04d}": dict(turn)
                   for position, turn in enumerate(original_turns, 1)}
    turn_position = {alias: position for position, alias in
                     enumerate(alias_index, 1)}
    sessions = ordered_session_ids(original_turns)
    known_people = {str(turn.get("speaker") or "") for turn in original_turns}

    for key in ("title", "expansion_objective"):
        if not _text(raw.get(key)):
            errors.append(key)

    new_characters = raw.get("allowed_new_characters")
    new_names: list[str] = []
    if (not isinstance(new_characters, list) or
            not 0 <= len(new_characters) <= 2):
        errors.append("allowed_new_characters")
        new_characters = []
    for row in new_characters:
        if (not isinstance(row, dict) or set(row) != {"name", "role"} or
                not _text(row.get("name")) or not _text(row.get("role"))):
            errors.append("allowed_new_character_schema")
            continue
        name = str(row["name"]).strip()
        if name in known_people or name in new_names:
            errors.append("allowed_new_character_duplicate_or_existing")
        new_names.append(name)

    arcs = raw.get("event_arcs")
    if not isinstance(arcs, list) or len(arcs) != ARC_COUNT:
        errors.append("event_arcs")
        arcs = []
    for arc in arcs:
        if (not isinstance(arc, dict) or set(arc) != {"summary", "stages"} or
                not _text(arc.get("summary"))):
            errors.append("event_arc_schema")
            continue
        stages = arc.get("stages")
        if (not isinstance(stages, list) or len(stages) != STAGES_PER_ARC or
                any(not _text(stage) for stage in stages) or
                len({str(stage).strip().casefold() for stage in stages}) !=
                STAGES_PER_ARC):
            errors.append("event_arc_stages")

    scenes = raw.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != SCENE_COUNT:
        errors.append("scenes")
        scenes = []
    allowed_people = known_people | set(new_names)
    expected_scene_fields = set(
        TEMPORAL_OUTLINE_SCHEMA["properties"]["scenes"]["items"]["required"])
    arc_scene_gaps: dict[int, list[int]] = defaultdict(list)
    for scene_index, scene in enumerate(scenes, 1):
        if not isinstance(scene, dict) or set(scene) != expected_scene_fields:
            errors.append("scene_schema")
            continue
        if (not _text(scene.get("purpose")) or
                not _text(scene.get("handoff_requirement"))):
            errors.append("scene_text")
        participants = scene.get("participants")
        if (not isinstance(participants, list) or
                not 2 <= len(participants) <= 4 or
                len(set(participants)) != len(participants) or
                any(not _text(value) or value not in allowed_people
                    for value in participants)):
            errors.append("scene_participants")
        if not _text(scene.get("location")):
            errors.append("scene_location")
        arc_index = scene.get("arc_index")
        if (not isinstance(arc_index, int) or not 1 <= arc_index <= ARC_COUNT):
            errors.append("scene_arc_index")
            arc_index = 0
        aliases = scene.get("anchor_aliases")
        if (not isinstance(aliases, list) or not 1 <= len(aliases) <= 3 or
                len(set(aliases)) != len(aliases) or
                any(alias not in alias_index for alias in aliases)):
            errors.append("scene_anchors")
            aliases = []
        if aliases and arc_index:
            latest_alias = max(aliases, key=turn_position.__getitem__)
            latest_session = str(alias_index[latest_alias]["session_id"])
            arc_scene_gaps[arc_index].append(sessions.index(latest_session) + 1)
    for arc_index in range(1, ARC_COUNT + 1):
        gaps_for_arc = arc_scene_gaps.get(arc_index) or []
        if not 1 <= len(gaps_for_arc) <= STAGES_PER_ARC:
            errors.append("arc_scene_count_invalid")

    return {
        "decision": ("temporal_outline_accept" if not errors else
                     "temporal_outline_schema_invalid"),
        "reasons": sorted(set(errors)),
        "source_session_ids": sessions,
        **({"raw_plan": raw} if not errors else {}),
    }


def normalize_temporal_outline_plan(
        raw: Mapping[str, Any], *,
        original_turns: Sequence[Mapping[str, Any]],
        round_index: int,
) -> dict[str, Any]:
    check = validate_temporal_outline_plan(raw, original_turns=original_turns)
    if check["decision"] != "temporal_outline_accept":
        raise ValueError("cannot_normalize_invalid_temporal_outline")
    alias_index = {f"T{position:04d}": dict(turn)
                   for position, turn in enumerate(original_turns, 1)}
    referenced = {alias for scene in raw["scenes"]
                  for alias in scene["anchor_aliases"]}
    ordered_aliases = [alias for alias in alias_index if alias in referenced]
    anchor_ids = {alias: f"P{position}" for position, alias in
                  enumerate(ordered_aliases, 1)}
    parent_anchors = [{
        "anchor_id": anchor_ids[alias],
        "source_alias": alias,
        "turn_id": alias_index[alias]["turn_id"],
        "session_id": alias_index[alias]["session_id"],
        "speaker": alias_index[alias]["speaker"],
        "text": alias_index[alias]["text"],
        "continuity_role": (
            f"API-selected evidence before a temporal insertion in round "
            f"{round_index}."),
    } for alias in ordered_aliases]
    known_people = sorted({str(turn["speaker"]) for turn in original_turns})
    used_people = {name for scene in raw["scenes"]
                   for name in scene["participants"]}
    source_sessions = ordered_session_ids(original_turns)
    source_session_order = {session_id: index for index, session_id in
                            enumerate(source_sessions)}
    api_scene_order = sorted(
        range(len(raw["scenes"])),
        key=lambda api_index: (
            source_session_order[str(alias_index[max(
                raw["scenes"][api_index]["anchor_aliases"],
                key=lambda alias: list(alias_index).index(alias))][
                    "session_id"])],
            api_index,
        ),
    )
    normalized_scene_position = {
        api_index: position for position, api_index in
        enumerate(api_scene_order, 1)
    }
    new_people = []
    for row in raw["allowed_new_characters"]:
        name = str(row["name"]).strip()
        if name not in used_people:
            continue
        first_scene = min(
            normalized_scene_position[api_index]
            for api_index, scene in enumerate(raw["scenes"])
            if name in scene["participants"]
        )
        new_people.append({"name": name, "role": str(row["role"]).strip(),
                           "introduction_scene": f"S{first_scene}"})

    arcs = []
    for arc_index, arc in enumerate(raw["event_arcs"], 1):
        scene_aliases = [alias for scene in raw["scenes"]
                         if scene["arc_index"] == arc_index
                         for alias in scene["anchor_aliases"]]
        arcs.append({
            "event_id": f"E{arc_index}",
            "summary": str(arc["summary"]).strip(),
            "stages": [str(value).strip() for value in arc["stages"]],
            "parent_anchor_ids": list(dict.fromkeys(
                anchor_ids[alias] for alias in scene_aliases)),
        })

    gaps = insertion_gap_index(original_turns)
    sessions = ordered_session_ids(original_turns)
    source_session_index = {session_id: index
                            for index, session_id in enumerate(sessions)}
    positioned_scenes = []
    for api_position, scene in enumerate(raw["scenes"]):
        latest_alias = max(scene["anchor_aliases"],
                           key=lambda alias: list(alias_index).index(alias))
        after = str(alias_index[latest_alias]["session_id"])
        after_index = source_session_index[after]
        before = (sessions[after_index + 1]
                  if after_index + 1 < len(sessions) else "END")
        positioned_scenes.append((gaps[(after, before)], api_position,
                                  scene, after, before))
    positioned_scenes.sort(key=lambda row: (row[0], row[1]))
    arc_scene_counts = Counter(
        int(scene["arc_index"]) for scene in raw["scenes"])
    arc_scene_ordinals: dict[int, int] = defaultdict(int)
    scenes = []
    for scene_index, (gap_index, api_position, scene, after, before) in \
            enumerate(positioned_scenes, 1):
        arc_index = int(scene["arc_index"])
        event_id = f"E{arc_index}"
        arc = arcs[arc_index - 1]
        arc_scene_ordinals[arc_index] += 1
        ordinal = arc_scene_ordinals[arc_index]
        partitions_by_count = {
            1: [[1, 2, 3, 4]],
            2: [[1, 2], [3, 4]],
            3: [[1], [2], [3, 4]],
            4: [[1], [2], [3], [4]],
        }
        partitions = partitions_by_count[arc_scene_counts[arc_index]]
        stage_indices = partitions[ordinal - 1]
        refs = [{
            "event_id": event_id,
            "stage_index": stage_index - 1,
            "stage_id": f"{event_id}:stage_{stage_index:02d}",
            "stage": arc["stages"][stage_index - 1],
        } for stage_index in stage_indices]
        scenes.append({
            "scene_id": f"S{scene_index}",
            "api_scene_index": api_position + 1,
            "target_child_tokens": 420,
            "acceptable_token_range": [360, 480],
            "target_turns": [10, 14],
            "participants": list(scene["participants"]),
            "location": str(scene["location"]).strip(),
            "purpose": str(scene["purpose"]).strip(),
            "required_parent_anchor_ids": [anchor_ids[alias]
                                           for alias in
                                           scene["anchor_aliases"]],
            "assigned_event_stage_refs": refs,
            "insertion_after_session_id": after,
            "insertion_before_session_id": before,
            "insertion_gap_index": gap_index,
            "handoff_requirement": str(
                scene["handoff_requirement"]).strip(),
            "allowed_output": [
                "natural dialogue that adds a new episode",
                "actions, discoveries, decisions, revisions, or outcomes",
                "local reactions and bounded logistics",
            ],
            "forbidden_output": [
                "QA questions, options, labels, or gold answers",
                "restating an existing dialogue episode as if it were new",
                "using later boundary facts as already known",
                "unplanned major events or named characters",
            ],
        })

    return {
        "schema_version": TEMPORAL_OUTLINE_SCHEMA_VERSION,
        "outline_id": f"conv-41:outline:temporal-insertion:{round_index:02d}",
        "outline_role": "dialogue_temporal_insertion_plan_only",
        "source_parent": {
            "sample_id": "conv-41-parent-dialogue",
            "lineage_group": "conv-41",
            "dialogue_hash": dialogue_content_hash(
                [dict(turn) for turn in original_turns]),
            "parent_graph_hash": "not_used_for_qa_gold",
            "parent_turn_count": len(original_turns),
            "core_speakers": known_people,
        },
        "qa_grounding_policy": {
            "outline_used_to_generate_qa_gold": False,
            "frozen_interleaved_dialogue_is_future_qa_source_of_truth": True,
            "qa_generation_allowed_during_realization": False,
        },
        "title": str(raw["title"]).strip(),
        "target": {
            "child_token_growth": 4200,
            "scene_count": SCENE_COUNT,
            "tokens_per_scene": 420,
            "minimum_accepted_child_tokens": 3600,
            "maximum_new_named_characters": 2,
        },
        "parent_anchors": parent_anchors,
        "expansion_objective": str(raw["expansion_objective"]).strip(),
        "allowed_new_characters": new_people,
        "allowed_locations": list(dict.fromkeys(
            str(scene["location"]).strip() for scene in raw["scenes"])),
        "planned_event_arcs": arcs,
        "scene_plan": scenes,
        "temporal_contract": {
            "original_turns_immutable": True,
            "original_order_immutable": True,
            "scene_order_is_chronological": True,
            "later_boundary_context_is_constraint_only": True,
            "fixed_positional_stage_schedule_used": False,
            "insertion_gap_derived_from_latest_anchor": True,
        },
        "naturalness_contract": {
            "do_not_repeat_anchor_wording": True,
            "do_not_use_length_filler": True,
            "require_new_episode_content": True,
            "allow_uncertainty_and_non_positive_outcomes": True,
        },
        "forbidden_conflicts": [
            "Do not change any immutable original dialogue turn.",
            "Do not reveal a fact before the original dialogue establishes it.",
            "Do not contradict the first original session after the insertion.",
            "Do not replay an event that already occurred before the insertion.",
            "Do not generate QA or solver content.",
        ],
        "rolling_round_index": round_index,
        "raw_plan_sha256": canonical_sha256(raw),
    }


def _session_turns(
        turns: Sequence[Mapping[str, Any]], session_id: str,
) -> list[dict[str, Any]]:
    return [dict(turn) for turn in turns
            if str(turn.get("session_id") or "") == session_id]


def compile_temporal_scene_contracts(
        outline: Mapping[str, Any], *,
        original_turns: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    anchors = {str(row["anchor_id"]): dict(row)
               for row in outline.get("parent_anchors") or []}
    arcs = {str(row["event_id"]): dict(row)
            for row in outline.get("planned_event_arcs") or []}
    contracts = []
    for position, scene in enumerate(outline.get("scene_plan") or [], 1):
        refs = list(scene["assigned_event_stage_refs"])
        stage_records = [{
            "stage_id": ref["stage_id"],
            "event_id": ref["event_id"],
            "event_summary": arcs[ref["event_id"]]["summary"],
            "stage_index": ref["stage_index"],
            "stage": ref["stage"],
        } for ref in refs]
        required_ids = list(scene["required_parent_anchor_ids"])
        after = str(scene["insertion_after_session_id"])
        before = str(scene["insertion_before_session_id"])
        preceding = _session_turns(original_turns, after)[-4:]
        following = ([] if before == "END" else
                     _session_turns(original_turns, before)[:4])
        contract = {
            "contract_version": TEMPORAL_GRAPH_CONTRACT_VERSION,
            "scene_index": position,
            "scene_id": str(scene["scene_id"]),
            "purpose": str(scene["purpose"]),
            "required_participants": list(scene["participants"]),
            "required_parent_anchor_ids": required_ids,
            "parent_anchor_evidence": [anchors[anchor_id]
                                       for anchor_id in required_ids],
            "session_location": str(scene["location"]),
            "relative_time": f"after {after} and before {before}",
            "insertion_after_session_id": after,
            "insertion_before_session_id": before,
            "insertion_gap_index": int(scene["insertion_gap_index"]),
            "preceding_original_context": preceding,
            "following_original_context_constraint_only": following,
            "target_dialogue_tokens": int(scene["target_child_tokens"]),
            "target_approximate_english_word_range": [250, 320],
            "target_dialogue_token_range": list(
                scene["acceptable_token_range"]),
            "hard_dialogue_token_range": [290, 600],
            "target_turn_range": list(scene["target_turns"]),
            "hard_turn_range": [8, 18],
            "handoff_requirement": str(scene["handoff_requirement"]),
            "new_unresolved_hook_policy": "forbidden",
            "stage_accounting_mode": GRAPH_STAGE_ACCOUNTING_COVERAGE_ONLY,
            "assigned_event_stages": stage_records,
            "assigned_event_arc_ids": list(dict.fromkeys(
                row["event_id"] for row in stage_records)),
            "assigned_event_stage_ids": [row["stage_id"]
                                         for row in stage_records],
            "is_final_scene": position == len(outline["scene_plan"]),
        }
        contract["contract_sha256"] = graph_contract_sha256(contract)
        contracts.append(contract)
    expected = {f"E{arc}:stage_{stage:02d}"
                for arc in range(1, ARC_COUNT + 1)
                for stage in range(1, STAGES_PER_ARC + 1)}
    actual = {stage_id for contract in contracts
              for stage_id in contract["assigned_event_stage_ids"]}
    if actual != expected:
        raise ValueError("temporal_contract_stage_coverage_invalid")
    return contracts


def build_temporal_outline_prompt(
        *, original_turns: Sequence[Mapping[str, Any]], round_index: int,
) -> str:
    sessions = ordered_session_ids(original_turns)
    dialogue_lines: list[str] = []
    previous = None
    for position, turn in enumerate(original_turns, 1):
        session_id = str(turn["session_id"])
        if session_id != previous:
            dialogue_lines.append(f"--- SESSION {session_id} ---")
            previous = session_id
        dialogue_lines.append(
            f"[T{position:04d}] {turn['speaker']}: {turn['text']}")
    gaps = [f"after {sessions[index]} / before {sessions[index + 1]}"
            for index in range(len(sessions) - 1)]
    gaps.append(f"after {sessions[-1]} / before END")
    rules = [
        "Expand the supplied frozen dialogue by designing ten new dialogue scenes.",
        "New scenes may be inserted between original sessions or after the final session.",
        "Every original turn and the original session order are immutable.",
        "Return planning data only. Never generate dialogue, QA, options, answers, or solver content.",
        "Create exactly four distinct event arcs with exactly four ordered stages each.",
        "Each scene explicitly chooses exactly one arc_index.",
        "Each of the four arcs must be assigned to at least one and at most four scenes; the total remains exactly ten scenes.",
        "The program sorts each arc's scenes by their anchor-derived temporal positions and deterministically maps that arc's four ordered stages onto those scenes.",
        "The program inserts each new scene immediately after the original session containing its latest anchor alias, then sorts scenes chronologically.",
        "Every anchor_aliases value must be copied exactly from a [T####] label in the frozen dialogue, for example [\"T0001\"]. Never put a name, topic, quote, or session ID in anchor_aliases.",
        "Facts in the following original session are future consistency constraints, not facts the inserted speakers already know.",
        "Each inserted scene must add a new action, discovery, decision, revision, attempt, consequence, or outcome.",
        "Do not simply repeat, summarize, or rephrase an event already stated before the insertion gap.",
        "A later original summary may be supported by showing the missing earlier episode, but do not copy its wording.",
        "Keep time, people, relationships, and established facts consistent on both sides of every insertion.",
        "Use concrete stages. Do not make all stages planning, reflection, preparation, or general encouragement.",
        "Do not optimize a future question. The interleaved dialogue will be the only QA source of truth.",
        "Declare zero to two genuinely new named characters; do not redeclare a person already mentioned in the dialogue.",
        "Every participants entry must exactly match either an existing speaker name in the frozen dialogue or a name declared in allowed_new_characters. Never use undeclared groups, roles, or collective labels such as Volunteers, Community Members, Council Members, Attendees, Staff, or Leaders.",
        "Return one JSON object matching the schema exactly, with no Markdown.",
    ]
    return "\n\n".join([
        f"PROMPT_VERSION: {TEMPORAL_OUTLINE_PROMPT_VERSION}",
        f"ROUND: {round_index}",
        "PLANNING RULES\n- " + "\n- ".join(rules),
        "ALLOWED INSERTION GAPS\n- " + "\n- ".join(gaps),
        "FROZEN ORIGINAL DIALOGUE\n" + "\n".join(dialogue_lines),
        "OUTPUT JSON SCHEMA\n" + canonical_json(TEMPORAL_OUTLINE_SCHEMA),
    ])


def build_temporal_realizer_prompt(
        *, graph_scene_contract: Mapping[str, Any], **kwargs: Any,
) -> str:
    base = build_graph_realizer_prompt(
        graph_scene_contract=graph_scene_contract, **kwargs)
    rules = [
        "This scene will be inserted at the frozen temporal gap in the contract, not appended blindly at the end.",
        "Treat preceding_original_context as already known and immutable.",
        "Treat following_original_context_constraint_only as future text that must remain consistent; speakers cannot cite it as already known.",
        "Generate the missing new episode between those boundaries.",
        "Do not repeat or paraphrase the parent anchors as the main content of the scene.",
        "Realize every explicitly assigned stage. These stages were mapped in order within the API-selected event arc.",
        "Do not output insertion metadata or alter the required JSON schema.",
    ]
    return "\n\n".join([
        base,
        f"TEMPORAL_PROMPT_VERSION: {TEMPORAL_REALIZER_PROMPT_VERSION}",
        "TEMPORAL INSERTION RULES\n- " + "\n- ".join(rules),
    ])


def temporal_realizer_template_hash() -> str:
    material = (TEMPORAL_REALIZER_PROMPT_VERSION +
                inspect.getsource(build_temporal_realizer_prompt))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def merge_temporal_insertions(
        *, original_turns: Sequence[Mapping[str, Any]],
        sessions: Sequence[Mapping[str, Any]],
        contracts: Sequence[Mapping[str, Any]],
        round_index: int,
) -> dict[str, Any]:
    if len(sessions) != len(contracts):
        raise ValueError("temporal_session_contract_count_mismatch")
    by_scene = {str(session["graph_scene_id"]): session
                for session in sessions}
    child_position = 0
    insertions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_child: list[dict[str, Any]] = []
    manifest = []
    for contract in contracts:
        scene_id = str(contract["scene_id"])
        session = by_scene.get(scene_id)
        if session is None:
            raise ValueError("temporal_scene_missing")
        remapped = []
        for turn_position, turn in enumerate(session["dialogue_turns"], 1):
            child_position += 1
            row = {
                **dict(turn),
                "turn_id": f"I{round_index:02d}C{child_position:04d}",
                "session_id": f"insert_round_{round_index:02d}_{scene_id.lower()}",
            }
            remapped.append(row)
            all_child.append(row)
        after = str(contract["insertion_after_session_id"])
        insertions[after].extend(remapped)
        manifest.append({
            "scene_id": scene_id,
            "after_session_id": after,
            "before_session_id": str(contract[
                "insertion_before_session_id"]),
            "insertion_gap_index": int(contract["insertion_gap_index"]),
            "assigned_event_stage_ids": list(
                contract["assigned_event_stage_ids"]),
            "generated_session_id": remapped[0]["session_id"],
            "generated_turn_ids": [row["turn_id"] for row in remapped],
        })

    combined: list[dict[str, Any]] = []
    original_sessions = ordered_session_ids(original_turns)
    for session_id in original_sessions:
        combined.extend(_session_turns(original_turns, session_id))
        combined.extend(insertions.pop(session_id, []))
    if insertions:
        raise ValueError("temporal_insertion_after_unknown_session")
    original_ids = {str(turn["turn_id"]) for turn in original_turns}
    preserved = [turn for turn in combined
                 if str(turn["turn_id"]) in original_ids]
    if preserved != [dict(turn) for turn in original_turns]:
        raise ValueError("original_dialogue_not_preserved")
    if len({str(turn["turn_id"]) for turn in combined}) != len(combined):
        raise ValueError("temporal_merged_turn_id_duplicate")
    return {
        "turns": combined,
        "inserted_turns": all_child,
        "insertion_manifest": manifest,
        "original_turn_count": len(original_turns),
        "inserted_turn_count": len(all_child),
        "combined_turn_count": len(combined),
        "original_order_preserved": True,
    }
