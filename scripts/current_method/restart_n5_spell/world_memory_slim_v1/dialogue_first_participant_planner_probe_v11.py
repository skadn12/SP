"""Unique Planner-only probe for dialogue-first participant identities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..core import sha256_bytes
from . import planner_probe_11 as previous
from .action_availability_v11 import compile_action_availability_view
from .controlled_expansion import compile_action_menu_v3, validate_action_menu_v3
from .interaction_links import compile_source_interaction_links
from .live_contracts_v11 import build_planner_prompt_v11
from .narrative_action_contracts import action_contract_registry
from .parent_anchor_compiler import compile_anchor_registry
from .placement_choice_registry import (
    compile_placement_choice_registry,
    compile_source_anchor_registry,
)
from .planner_only_probe_v11 import _sessions
from .planning_state_view import compile_planning_state_view
from .source_memory_initializer_v1 import authoritative_affordance_view
from .source_memory_initializer_v2 import initialize_source_world_memory_v2


runtime = previous.runtime
_base_build_planner_probe_input = runtime.build_planner_probe_input
_base_validate_planner_outline = runtime.validate_planner_outline_v11

PARTICIPANT_PROBE_CONTRACT = {
    "contract_version": "dialogue_first_participant_planner_probe_v2_alias_reuse",
    "participant_alias_declaration": "once",
    "later_alias_reference": "allowed_in_any_later_scene",
    "planner_lifecycle_label": "forbidden",
    "memory_transaction_required_for_speech": False,
}


def _build_dialogue_first_probe_input(input_path: Path) -> dict[str, Any]:
    values = _base_build_planner_probe_input(input_path)
    source = initialize_source_world_memory_v2(values["root"], values["parent"])
    interactions = compile_source_interaction_links(
        root_id=runtime.ROOT_ID,
        sessions=_sessions(values["parent"]),
        speaker_to_entity=source["speaker_to_entity_ref"],
    )
    complete = source["complete_dialogue"]
    planning = compile_planning_state_view(
        memory=source["memory"], interaction_links=interactions,
        scene_lineage=(), recent_round_end={})
    anchors = compile_anchor_registry(
        complete_dialogue=complete, planning_state=planning, tail_turn_count=7)
    source_anchors = compile_source_anchor_registry(anchors)
    placement_choices = compile_placement_choice_registry(
        complete_dialogue=complete, anchor_registry=anchors)
    action_menu = compile_action_menu_v3(
        authoritative_affordance_view(source["memory"]), root_id=runtime.ROOT_ID,
        interaction_links=interactions, maximum_actions=16)
    action_validation = validate_action_menu_v3(action_menu)
    if not action_validation["valid"]:
        raise ValueError("dialogue_first_probe_action_menu_invalid")
    role_table = [{
        "entity_ref": row["entity_ref"],
        "display_name": row["display_name"],
        "entity_type": row.get("entity_type") or "person",
    } for row in planning.get("entities") or []]
    contracts = action_contract_registry()
    availability = compile_action_availability_view(planning_state=planning)
    prompt, frozen_input_sha = build_planner_prompt_v11(
        root_id=runtime.ROOT_ID,
        planning_state=planning,
        source_anchor_registry=source_anchors,
        placement_choice_registry=placement_choices,
        action_contracts=contracts,
        action_availability_view=availability,
        dialogue_first_participants=True,
    )
    coverage_instruction = (
        "DIALOGUE-FIRST PARTICIPANT IDENTITY\n"
        "New speakers are optional and may speak immediately. Historical authoritative people "
        "always use participant_ref in every scene; never copy an authoritative entity ref into "
        "participant_alias. Declare a genuinely new participant_alias once with identity fields "
        "or through introduce_connected_person; later appearances of that new person use only "
        "the alias plus scene role and intents. Participant-valued action arguments use exact "
        "authoritative refs or declared lowercase aliases, never display names. Do not output "
        "entity_role, connection_type, connection_target_ref, or connection_description in a "
        "speaker object. connection_target_ref belongs only inside "
        "introduce_connected_person.arguments and must not be duplicated into the speaker. "
        "Do not output "
        "scene_presence, lifecycle, or persistence labels and do not invent round refs, "
        "authoritative entity IDs, or a person-creation action merely to authorize speech. "
        "Keep narrative-action typed references within their frozen authoritative "
        "namespaces. Scope coverage is audited separately and is not a Planner validity "
        "quota."
    )
    marker = "FROZEN INPUT SHA256\n"
    if prompt.count(marker) != 1:
        raise RuntimeError("dialogue_first_probe_prompt_marker_drift")
    prompt = prompt.replace(marker, coverage_instruction + "\n\n" + marker, 1)
    values.update({
        "source": source,
        "interaction_links": interactions,
        "complete_dialogue": complete,
        "source_turn_registry": source["source_turn_registry"],
        "planning_state": planning,
        "anchor_registry": anchors,
        "source_anchor_registry": source_anchors,
        "placement_choice_registry": placement_choices,
        "action_menu": action_menu,
        "role_table": role_table,
        "action_availability_view": availability,
        "shared_narrative_action_contracts": contracts,
        "dialogue_first_participant_probe_contract": PARTICIPANT_PROBE_CONTRACT,
        "prompt": prompt,
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "frozen_planner_input_sha256": frozen_input_sha,
    })
    return values


def _participant_coverage(
    outline: Mapping[str, Any], *, authoritative_entity_refs: set[str],
) -> dict[str, Any]:
    appearances: dict[str, list[str]] = {}
    declarations: list[str] = []
    scope_fields: list[str] = []
    authoritative_refs_used_as_aliases: list[str] = []
    for scene in outline.get("scenes") or []:
        if not isinstance(scene, Mapping):
            continue
        scene_alias = str(scene.get("scene_alias") or "")
        for speaker in scene.get("speaker_participants") or []:
            if not isinstance(speaker, Mapping) or "participant_alias" not in speaker:
                continue
            alias = str(speaker.get("participant_alias") or "")
            appearances.setdefault(alias, []).append(scene_alias)
            if alias in authoritative_entity_refs:
                authoritative_refs_used_as_aliases.append(
                    f"{scene_alias}:{alias}")
            if "display_name" in speaker:
                declarations.append(alias)
            if "participant_scope" in speaker:
                scope_fields.append(f"{scene_alias}:{alias}")
    reused = sorted(alias for alias, scenes in appearances.items() if len(scenes) >= 2)
    checks = {
        "planner_scope_fields_absent": not scope_fields,
        "each_alias_declared_once": len(declarations) == len(set(declarations)),
        "authoritative_refs_never_used_as_aliases": not (
            authoritative_refs_used_as_aliases),
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "declared_aliases": sorted(set(declarations)),
        "reused_aliases": reused,
        "scope_field_paths": scope_fields,
        "authoritative_ref_alias_paths": authoritative_refs_used_as_aliases,
        "appearance_scene_aliases": appearances,
    }


def _validate_dialogue_first_outline(
    outline: Mapping[str, Any], **kwargs: Any,
) -> dict[str, Any]:
    structural = _base_validate_planner_outline(outline, **kwargs)
    coverage = _participant_coverage(
        outline,
        authoritative_entity_refs=set(kwargs.get("authoritative_entity_refs") or []),
    ) if isinstance(outline, Mapping) else {
        "valid": False, "checks": {}, "declared_aliases": [],
        "reused_aliases": [], "scope_field_paths": [],
        "authoritative_ref_alias_paths": [], "appearance_scene_aliases": {},
    }
    result = dict(structural)
    result["dialogue_first_participant_coverage"] = coverage
    result["errors"] = list(structural.get("errors") or [])
    result["valid"] = structural.get("valid") is True
    return result


runtime.PROBE_VERSION = "v11_dialogue_first_participant_planner_probe_1_v1"
runtime.PROBE_DISPLAY_NAME = "V11 Dialogue-First Participant Planner-only Probe 1"
runtime.PROBE_RUN_SUFFIX = "v11_dialogue_first_participant_planner_probe_1"
runtime.PREFLIGHT_VERSION = "v11_dialogue_first_participant_probe_preflight_v1"
runtime.REQUEST_ID = "planner_v11_dialogue_first_participant_probe_01"
runtime.REPORT_BASENAME = "V11_DIALOGUE_FIRST_PARTICIPANT_PLANNER_PROBE_REPORT"
runtime.LAUNCHER_PATH = Path(__file__).resolve()
runtime.RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_v11_dialogue_first_participant_planner_probe_1"
)
runtime.HISTORY_PATHS = {
    **runtime.HISTORY_PATHS,
    "live_canary_14": Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
        "conversation_chronicles_episode_61555_slim_continuous_expansion_"
        "v11_live_canary_14"
    ),
    "person_event_probe_2": Path(
        "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
        "conversation_chronicles_episode_61555_v11_person_event_planner_probe_2"
    ),
}
runtime.EXPECTED_HISTORY = {
    **runtime.EXPECTED_HISTORY,
    "live_canary_14": "f950428152ab5937d5ba41c0774d35e60e3be63a81be450767ebc1ae6a2596b7",
    "person_event_probe_2": "0efa67ba94452ca0faed4b0b2f3f7f852dce2f4f03e5f964d2f0dc129b7e3da6",
}
runtime.build_planner_probe_input = _build_dialogue_first_probe_input
runtime.validate_planner_outline_v11 = _validate_dialogue_first_outline


if __name__ == "__main__":
    raise SystemExit(runtime.main())
