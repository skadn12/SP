"""Eleven-root zero-API preflight for dialogue-first participant identities."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from ..core import atomic_json, atomic_text
from .action_availability_v11 import compile_action_availability_view
from .interaction_links import compile_source_interaction_links
from .live_contracts_v11 import build_planner_prompt_v11
from .offline_preflight_v11 import (
    CC_IDS, CC_ROOTS, LOCOMO_ATTESTATION, LOCOMO_JSONL,
    _sessions, build_round_blueprint_fixture,
)
from .inputs import load_locomo_parent_read_only, load_unified_root_read_only
from .parent_anchor_compiler import compile_anchor_registry
from .placement_choice_registry import (
    compile_placement_choice_registry, compile_source_anchor_registry,
)
from .planner_outline_v11 import (
    compile_planner_outline_v11, project_internal_v4_fixture_to_outline_v11,
    validate_planner_outline_v11,
)
from .planning_state_view import compile_planning_state_view
from .source_memory_initializer_v2 import initialize_source_world_memory_v2
from .structural_expansion_gate import validate_round_blueprint_structure


REPORT_DIR = Path(__file__).resolve().parent / "reports"
PREFLIGHT_VERSION = "dialogue_first_participant_11_root_preflight_v1"


def _new(alias: str, display: str) -> dict[str, Any]:
    return {
        "participant_alias": alias, "display_name": display,
        "scene_role": "supporting",
        "entry_context": "The participant naturally joins the anchored current scene.",
        "anchor_connection": "The participant is connected to the current place or task.",
        "intents": ["contribute one concrete scene-specific detail"],
    }


def _alias(alias: str) -> dict[str, Any]:
    return {"participant_alias": alias, "scene_role": "supporting",
            "intents": ["contribute one concrete scene-specific detail"]}


def _one(root_id: str, source_kind: str, loaded: Mapping[str, Any]) -> dict[str, Any]:
    initialized = initialize_source_world_memory_v2(loaded["root"], loaded["parent"])
    turns = loaded["parent"]["dialogue"]["turns"]
    links = compile_source_interaction_links(
        root_id=str(loaded["root"]["root_id"]), sessions=_sessions(turns),
        speaker_to_entity=initialized["speaker_to_entity_ref"])
    planning = compile_planning_state_view(
        memory=initialized["memory"], interaction_links=links)
    complete = initialized["complete_dialogue"]
    anchors = compile_anchor_registry(
        complete_dialogue=complete, planning_state=planning)
    sources = compile_source_anchor_registry(anchors)
    placements = compile_placement_choice_registry(
        complete_dialogue=complete, anchor_registry=anchors)
    availability = compile_action_availability_view(planning_state=planning)
    entity_refs = [str(row["entity_ref"]) for row in planning["entities"]]
    first_anchor = sources["source_anchors"][0]["source_anchor_id"]
    fixture = build_round_blueprint_fixture(
        source_anchor_id=first_anchor, placement_choice_id="PC_TAIL_AFTER",
        entity_refs=entity_refs[:2], single_action_selection={}, scene_count=5)
    outline = project_internal_v4_fixture_to_outline_v11(fixture)
    outline["scenes"][0]["speaker_participants"][1] = _new(
        "passerby", "Passerby")
    outline["scenes"][1]["speaker_participants"][1] = _new(
        "candidate_c", "Connected participant C")
    outline["scenes"][2]["speaker_participants"][1] = _alias("candidate_c")
    validation = validate_planner_outline_v11(
        outline, source_anchor_registry=sources,
        placement_choice_registry=placements,
        authoritative_entity_refs=set(entity_refs),
        authoritative_entities=planning["entities"],
        action_availability_view=availability)
    compiled = None
    gate = {"valid": False, "errors": ["not_compiled"]}
    if validation["valid"]:
        compiled = compile_planner_outline_v11(
            outline, source_anchor_registry=sources,
            placement_choice_registry=placements, planning_state=planning,
            action_availability_view=availability)
        gate = validate_round_blueprint_structure(
            compiled_plan=compiled, source_anchor_registry=sources,
            placement_choice_registry=placements, planning_state=planning,
            complete_dialogue=complete)
    prompt, prompt_sha = build_planner_prompt_v11(
        root_id=str(planning["root_id"]), planning_state=planning,
        source_anchor_registry=sources, placement_choice_registry=placements,
        action_availability_view=availability,
        dialogue_first_participants=True)
    checks = {
        "memory_has_no_parent_utterance_records": not any(
            row.get("record_type") == "parent_utterance"
            for row in initialized["memory"]["authoritative_records"]),
        "source_turn_registry_resolves": initialized[
            "source_turn_registry_validation"]["valid"],
        "source_turn_registry_has_no_text": not any(
            "text" in row for row in initialized["source_turn_registry"]["turns"]),
        "planner_outline_valid": validation["valid"],
        "planner_outline_compiled": compiled is not None,
        "round_local_refs_program_generated": bool(compiled) and {
            row["participant_ref"]
            for row in compiled["round_participant_registry"]["participants"]
        } == {"R01:P01", "R01:P02"},
        "persistent_ref_stable": bool(compiled) and [
            row["participant_ref"]
            for alias in ("s2", "s3")
            for row in compiled["round_participant_registry"]["scene_bindings"][alias]
            if row.get("participant_alias") == "candidate_c"
        ] == ["R01:P02", "R01:P02"],
        "structural_gate": gate["valid"],
        "dialogue_first_prompt_contract": (
            "New speakers may speak immediately" in prompt and
            "round_participant_registry_v3_action_or_speaker_declare_then_reference"
            in prompt and
            "participant_scope" not in prompt),
    }
    return {
        "root_id": root_id, "source_kind": source_kind,
        "turn_count": len(turns), "entity_count": len(planning["entities"]),
        "authoritative_record_count": len(
            initialized["memory"]["authoritative_records"]),
        "source_turn_registry_count": initialized[
            "source_turn_registry"]["turn_count"],
        "planner_prompt_sha256": prompt_sha,
        "participant_registry": (deepcopy(compiled[
            "round_participant_registry"]) if compiled else None),
        "validation_errors": validation["errors"],
        "structural_errors": gate["errors"],
        "checks": checks, "passed": all(checks.values()),
    }


def run_eleven_root_preflight() -> dict[str, Any]:
    rows = []
    for root_id in CC_IDS:
        loaded = load_unified_root_read_only(
            CC_ROOTS / root_id / "unified_root.json")
        rows.append(_one(root_id, "conversation_chronicles", loaded))
    locomo = load_locomo_parent_read_only(LOCOMO_JSONL, LOCOMO_ATTESTATION)
    rows.append(_one("locomo-conv-43", "locomo_attested_record_77", locomo))
    return {
        "preflight_version": PREFLIGHT_VERSION,
        "root_count": len(rows),
        "passed_count": sum(row["passed"] for row in rows),
        "all_passed": len(rows) == 11 and all(row["passed"] for row in rows),
        "api_calls": 0, "network_calls": 0, "roots": rows,
    }


def write_preflight_reports(result: Mapping[str, Any]) -> None:
    atomic_json(REPORT_DIR / "V11_DIALOGUE_FIRST_PARTICIPANT_PREFLIGHT.json",
                dict(result))
    lines = [
        "# V11 Dialogue-First Participant 11-Root Preflight", "",
        f"- Result: {result['passed_count']}/{result['root_count']}",
        f"- API calls: {result['api_calls']}", "",
        "| Root | Turns | Entities | Memory records | Turn registry | Passed |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in result["roots"]:
        lines.append(
            f"| {row['root_id']} | {row['turn_count']} | {row['entity_count']} | "
            f"{row['authoritative_record_count']} | {row['source_turn_registry_count']} | "
            f"{str(row['passed']).lower()} |")
    atomic_text(REPORT_DIR / "V11_DIALOGUE_FIRST_PARTICIPANT_PREFLIGHT.md",
                "\n".join(lines) + "\n")


if __name__ == "__main__":
    result = run_eleven_root_preflight()
    write_preflight_reports(result)
    print(json.dumps({key: result[key] for key in (
        "root_count", "passed_count", "all_passed", "api_calls")}, indent=2))
    raise SystemExit(0 if result["all_passed"] else 1)
