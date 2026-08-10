from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from tests.test_v11_slim_planner_outline import fixture as outline_fixture
from tests.test_world_memory_slim_live_fresh_e2e_v7 import source_fixture
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.append_only_memory import (
    apply_scene_commit,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.complete_dialogue_view import (
    compose_complete_dialogue_view,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.continuous_round_plan_v1 import (
    build_scene_dependency_dag,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.durable_update_compiler import (
    compile_participant_persistence_updates,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_contracts_v11 import (
    build_planner_prompt_v11, normalize_realizer_dialogue_v11,
    planner_retry_guidance_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.narrative_action_contracts import (
    migrate_legacy_connected_person_contract_v2,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.participant_persistence import (
    build_dialogue_first_round_validator_prompt,
    build_participant_persistence_prompt_extension,
    participant_persistence_review_schema,
    validate_participant_persistence_reviews,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.planner_outline_v11 import (
    compile_planner_outline_v11, validate_planner_outline_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import (
    hash_tree,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_participant_registry import (
    compile_round_participant_registry,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.source_memory_initializer_v1 import (
    initialize_source_world_memory,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.source_memory_initializer_v2 import (
    initialize_source_world_memory_v2,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.source_turn_registry import (
    compile_source_turn_registry, validate_source_turn_registry,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.transactional_binder import (
    build_allocation_plan, commit_scene_transaction,
)


def _new(alias: str, display: str = "Co-worker C") -> dict:
    return {
        "participant_alias": alias,
        "display_name": display,
        "scene_role": "supporting",
        "entry_context": "C joins the current project discussion in the same office.",
        "anchor_connection": "C is connected to the project discussed by A and B.",
        "intents": ["clarify one practical project detail"],
    }


def _alias(alias: str) -> dict:
    return {"participant_alias": alias, "scene_role": "supporting",
            "intents": ["clarify one practical project detail"]}


PROBE_2 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_v11_dialogue_first_participant_"
    "planner_probe_2"
)
PROBE_2_SHA = "9ec1139411667f5e37649280fb38de4a972a96e283996ee629b42a3101e04563"
PROBE_3 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_v11_dialogue_first_participant_"
    "planner_probe_3"
)
PROBE_3_SHA = "cb074e719f70fff891c0fa31176bcf65246476a372b0b6bbb0f31ddcd0ea375d"
CANARY_16 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_continuous_expansion_"
    "v11_live_canary_16"
)
CANARY_16_SHA = "1b6fc0e04658afc08104240a79184704b4893262d477e1b5adc4cd4dd242b5e5"
PROBE_5 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_v11_dialogue_first_participant_"
    "planner_probe_5"
)
PROBE_5_SHA = "f87476acb598eac387d7451986fb50b5b08b8a6c01edc727ff02840058e3d6e1"
PROBE_6 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_v11_dialogue_first_participant_"
    "planner_probe_6"
)
PROBE_6_SHA = "7c8ed556f5ec6a19fb869c9ba6f1b0385d5300d63009c194191760c1ae06d8a3"


def _round_fixture(*, persistent: bool = True, same_name: bool = False):
    root, parent = source_fixture()
    initialized = initialize_source_world_memory_v2(root, parent)
    entity = initialized["memory"]["entities"][0]
    display = entity["display_name"] if same_name else "Co-worker C"
    model_scenes = [{
        "scene_alias": "s1", "dependencies": [],
        "speaker_participants": [
            {"participant_ref": entity["entity_ref"], "scene_role": "primary",
             "intents": ["ask for a concrete project detail"]},
            _new("c1", display),
        ],
    }]
    registry = compile_round_participant_registry(
        scenes=model_scenes, authoritative_entities=initialized["memory"]["entities"])
    participant_ref = registry["participants"][0]["participant_ref"]
    child = {
        "root_id": root["root_id"], "child_id": "child-r1",
        "scenes": [{
            "scene_ref": "R01:S01", "session_ref": "R01:S01:SESSION",
            "dialogue_turns": [
                {"turn_ref": "R01:S01:T001", "speaker_ref": entity["entity_ref"],
                 "speaker_label": entity["display_name"], "text": "What changed in the plan?"},
                {"turn_ref": "R01:S01:T002", "speaker_ref": participant_ref,
                 "speaker_label": display,
                 "text": "I joined this project to verify the revised schedule."},
            ],
        }],
    }
    complete = compose_complete_dialogue_view(
        original_parent=parent,
        promoted_rounds=[{"round_index": 1, "child": child}],
        speaker_to_entity_ref=initialized["speaker_to_entity_ref"],
    )
    source_registry = compile_source_turn_registry(complete)
    realized = [{
        "scene_alias": "s1", "scene_ref": "R01:S01",
        "predecessor_scene_refs": [],
        "dialogue_turns": deepcopy(child["scenes"][0]["dialogue_turns"]),
    }]
    review = {
        "participant_ref": participant_ref,
        "persistence_decision": "persist" if persistent else "do_not_persist",
        "identity_status": "confirmed",
        "connection_status": "confirmed" if persistent else "uncertain",
        "equivalence_status": "distinct",
        "equivalent_entity_ref": None,
        "verified_connection": ({
            "connection_kind": "interaction_link",
            "target_ref": entity["entity_ref"], "relation_type": None,
            "description": "C participates in the existing project discussion.",
            "evidence_turn_refs": ["R01:S01:T002"],
        } if persistent else None),
        "current_scene_evidence_turn_refs": ["R01:S01:T002"],
        "dependency_context_turn_refs": [],
        "reason": "The dialogue establishes the identity and project connection.",
    }
    return initialized, registry, complete, source_registry, realized, review


class DialogueFirstParticipantTests(unittest.TestCase):
    def test_source_memory_v2_keeps_transcript_out_of_memory(self):
        root, parent = source_fixture()
        result = initialize_source_world_memory_v2(root, parent)
        self.assertEqual(result["source_statistics"]["parent_utterance_record_count"], 0)
        self.assertEqual(result["memory"]["authoritative_records"], [])
        self.assertEqual(len(result["memory"]["entities"]), 2)
        self.assertTrue(result["source_turn_registry_validation"]["valid"])
        self.assertFalse(any("text" in row for row in
                             result["source_turn_registry"]["turns"]))
        legacy = initialize_source_world_memory(root, parent)
        self.assertEqual(len(legacy["memory"]["authoritative_records"]), 4)

    def test_source_turn_registry_detects_text_drift(self):
        initialized, _, complete, registry, _, _ = _round_fixture()
        del initialized
        self.assertTrue(validate_source_turn_registry(registry, complete)["valid"])
        changed = deepcopy(complete)
        changed["turns"][-1]["text"] = "silently changed"
        invalid = validate_source_turn_registry(registry, changed)
        self.assertTrue(any("text_sha_mismatch" in error for error in invalid["errors"]))

    def test_round_speaker_needs_no_memory_entity(self):
        initialized, registry, _, _, _, review = _round_fixture(persistent=False)
        self.assertTrue(registry["valid"], registry["errors"])
        ref = registry["participants"][0]["participant_ref"]
        self.assertTrue(ref.startswith("R01:P"))
        self.assertNotIn(ref, {row["entity_ref"] for row in initialized["memory"]["entities"]})
        self.assertEqual(review["persistence_decision"], "do_not_persist")

    def test_declared_alias_ref_is_stable_across_later_scenes(self):
        initialized = initialize_source_world_memory_v2(*source_fixture())
        existing = initialized["memory"]["entities"][0]["entity_ref"]
        existing_row = {"participant_ref": existing, "scene_role": "primary",
                        "intents": ["continue the discussion"]}
        first = _new("c1")
        second = _alias("c1")
        scenes = [
            {"scene_alias": "s1", "dependencies": [],
             "speaker_participants": [existing_row, first]},
            {"scene_alias": "s2", "dependencies": [{
                "predecessor_scene_alias": "s1", "dependency_type": "continues"}],
             "speaker_participants": [existing_row, second]},
        ]
        registry = compile_round_participant_registry(
            scenes=scenes, authoritative_entities=initialized["memory"]["entities"])
        self.assertTrue(registry["valid"], registry["errors"])
        refs = [row["participant_ref"] for rows in registry["scene_bindings"].values()
                for row in rows if row["participant_alias"] == "c1"]
        self.assertEqual(refs, ["R01:P01", "R01:P01"])
        scenes[1]["dependencies"] = []
        independent = compile_round_participant_registry(
            scenes=scenes, authoritative_entities=initialized["memory"]["entities"])
        self.assertTrue(independent["valid"], independent["errors"])

    def test_alias_cross_scene_reuse_is_allowed_after_declaration(self):
        initialized = initialize_source_world_memory_v2(*source_fixture())
        existing = initialized["memory"]["entities"][0]["entity_ref"]
        row = {"participant_ref": existing, "scene_role": "primary",
               "intents": ["continue"]}
        local = _new("passerby", "Passerby")
        scenes = [
            {"scene_alias": "s1", "dependencies": [],
             "speaker_participants": [row, local]},
            {"scene_alias": "s2", "dependencies": [{
                "predecessor_scene_alias": "s1", "dependency_type": "continues"}],
             "speaker_participants": [row, _alias("passerby")]},
        ]
        result = compile_round_participant_registry(
            scenes=scenes, authoritative_entities=initialized["memory"]["entities"])
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["alias_to_round_ref"]["passerby"], "R01:P01")

    def test_alias_reference_must_follow_one_declaration(self):
        initialized = initialize_source_world_memory_v2(*source_fixture())
        existing = initialized["memory"]["entities"][0]["entity_ref"]
        row = {"participant_ref": existing, "scene_role": "primary",
               "intents": ["continue"]}
        undeclared = compile_round_participant_registry(
            scenes=[{"scene_alias": "s1", "dependencies": [],
                     "speaker_participants": [row, _alias("passerby")]}],
            authoritative_entities=initialized["memory"]["entities"])
        self.assertTrue(any("participant_alias_not_declared" in value
                            for value in undeclared["errors"]))
        redeclared = compile_round_participant_registry(
            scenes=[
                {"scene_alias": "s1", "dependencies": [],
                 "speaker_participants": [row, _new("passerby", "Passerby")]},
                {"scene_alias": "s2", "dependencies": [],
                 "speaker_participants": [row, _new("passerby", "Passerby")]},
            ], authoritative_entities=initialized["memory"]["entities"])
        self.assertTrue(any("participant_alias_redeclared" in value
                            for value in redeclared["errors"]))

    def test_connected_person_action_can_declare_later_speaker_alias(self):
        initialized = initialize_source_world_memory_v2(*source_fixture())
        existing = initialized["memory"]["entities"][0]["entity_ref"]
        existing_speaker = {"participant_ref": existing, "scene_role": "primary",
                            "intents": ["continue"]}
        person_action = {
            "action_mode": "new", "action_type": "introduce_connected_person",
            "arguments": {
                "participant_alias": "lena", "display_name": "Lena",
                "entity_role": "friend", "connection_type": "existing_person",
                "connection_target_ref": existing,
                "connection_description": "Lena is connected to A.",
            },
        }
        scenes = [
            {"scene_alias": "s1", "new_development": "A mentions Lena.",
             "primary_action": person_action, "secondary_action": None,
             "speaker_participants": [existing_speaker]},
            {"scene_alias": "s2", "new_development": "Lena joins later.",
             "primary_action": None, "secondary_action": None,
             "speaker_participants": [existing_speaker, _alias("lena")]},
        ]
        registry = compile_round_participant_registry(
            scenes=scenes, authoritative_entities=initialized["memory"]["entities"])
        self.assertTrue(registry["valid"], registry["errors"])
        self.assertEqual(registry["alias_to_round_ref"]["lena"], "R01:P01")
        participant = registry["participants"][0]
        self.assertEqual(participant["declaration_source"],
                         "introduce_connected_person_action")
        self.assertEqual(participant["declaration_action_ref"], "S01.primary_action")
        self.assertEqual(participant["appearance_scene_aliases"], ["s2"])

    def test_action_and_speaker_alias_identity_conflict_is_rejected(self):
        initialized = initialize_source_world_memory_v2(*source_fixture())
        existing = initialized["memory"]["entities"][0]["entity_ref"]
        action = {
            "action_mode": "new", "action_type": "introduce_connected_person",
            "arguments": {
                "participant_alias": "lena", "display_name": "Lena",
                "entity_role": "friend", "connection_type": "existing_person",
                "connection_target_ref": existing,
                "connection_description": "Lena is connected to A.",
            },
        }
        conflicting = _new("lena", "Someone Else")
        registry = compile_round_participant_registry(
            scenes=[{"scene_alias": "s1", "new_development": "Lena arrives.",
                     "primary_action": action, "secondary_action": None,
                     "speaker_participants": [conflicting]}],
            authoritative_entities=initialized["memory"]["entities"])
        self.assertFalse(registry["valid"])
        self.assertTrue(any("participant_alias_identity_conflict:lena" in value
                            for value in registry["errors"]))

    def test_outline_compiler_accepts_round_local_speaker(self):
        inputs, source, placements, outline, planning, availability, refs = outline_fixture(5)
        outline["scenes"][0]["speaker_participants"][1] = _new(
            "passerby", "Passerby")
        validation = validate_planner_outline_v11(
            outline, source_anchor_registry=source,
            placement_choice_registry=placements,
            authoritative_entity_refs=refs, authoritative_entities=planning["entities"],
            action_availability_view=availability)
        self.assertTrue(validation["valid"], validation["errors"])
        compiled = compile_planner_outline_v11(
            outline, source_anchor_registry=source,
            placement_choice_registry=placements, planning_state=planning,
            action_availability_view=availability)
        speaker = compiled["scenes"][0]["speaker_participants"][1]
        self.assertEqual(speaker["participant_ref"], "R01:P01")
        self.assertEqual(speaker["display_name"], "Passerby")
        self.assertFalse(speaker.get("is_authoritative_memory_entity", False))
        self.assertIn("round_participant_registry", compiled)
        self.assertNotEqual(compiled["scenes"][0]["primary_action"].get(
            "resolved_action_type"), "introduce_connected_person")
        scene = compiled["scenes"][0]
        allowed = {row["participant_ref"] for row in scene["speaker_participants"]}
        normalized = normalize_realizer_dialogue_v11({"dialogue_turns": [
            {"speaker_ref": next(iter(allowed - {"R01:P01"})),
             "text": "Can you point me toward the station?"},
            {"speaker_ref": "R01:P01",
             "text": "Yes, continue two blocks and turn left."},
        ]}, scene_ref=scene["scene_ref"], allowed_speaker_refs=allowed)
        self.assertTrue(normalized["valid"], normalized["errors"])

    def test_dialogue_first_prompt_allows_new_speakers_without_memory(self):
        _, source, placements, _, planning, availability, _ = outline_fixture(5)
        prompt, _ = build_planner_prompt_v11(
            root_id=planning["root_id"], planning_state=planning,
            source_anchor_registry=source, placement_choice_registry=placements,
            action_availability_view=availability,
            dialogue_first_participants=True)
        self.assertIn("New speakers may speak immediately", prompt)
        self.assertNotIn("participant_scope", prompt)
        self.assertIn(
            "round_participant_registry_v3_action_or_speaker_declare_then_reference",
            prompt)
        self.assertIn(
            "After either form, every appearance of that ROUND-LOCAL person uses exact keys",
            prompt)
        self.assertIn("authoritative_refs_never_use_participant_alias", prompt)
        self.assertIn("scene_presence_is_compiler_derived", prompt)

    def test_connected_person_scene_presence_is_not_planner_owned(self):
        _, _, _, _, _, availability, refs = outline_fixture(5)
        action = {
            "action_mode": "new", "action_type": "introduce_connected_person",
            "arguments": {
                "participant_alias": "local_1", "display_name": "Lena",
                "entity_role": "visitor", "connection_type": "existing_person",
                "connection_target_ref": next(iter(refs)),
                "connection_description": "Lena is connected to an existing person.",
                "scene_presence": "speaking",
            },
        }
        from scripts.current_method.restart_n5_spell.world_memory_slim_v1.narrative_action_contracts import (
            validate_action_instance,
        )
        result = validate_action_instance(
            action, authoritative_entity_refs=refs,
            action_availability_view=availability,
            known_world_refs=set(availability[
                "typed_reference_registry"]["all_typed_refs"]))
        self.assertFalse(result["valid"])
        self.assertIn("action.arguments:unexpected:scene_presence", result["errors"])

    def test_retry_guidance_separates_refs_aliases_and_display_names(self):
        _, _, _, _, _, availability, _ = outline_fixture(5)
        guidance = "\n".join(planner_retry_guidance_v11([
            "scene:2:speaker:1:participant_alias_not_declared:"
            "cc-train-episode-61555:E:333e9451b96f",
            "action:S01.primary_action.arguments.target:unknown_typed_ref:"
            "state_target_refs:Co-workers A",
            "scene:7:link_type:resolution",
        ], action_availability_view=availability))
        self.assertIn("always a speaker object with participant_ref", guidance)
        self.assertIn("Display names such as Co-workers A or Manager C", guidance)
        self.assertIn("connection_target_ref only in introduce_connected_person.arguments",
                      guidance)
        self.assertIn("resolution are not scene link_type values", guidance)

    def test_planner_prompt_keeps_connected_person_target_out_of_speaker(self):
        _, source, placements, _, planning, availability, _ = outline_fixture(5)
        prompt, _ = build_planner_prompt_v11(
            root_id=planning["root_id"], planning_state=planning,
            source_anchor_registry=source, placement_choice_registry=placements,
            action_availability_view=availability,
            dialogue_first_participants=True)
        self.assertIn(
            "connection_target_ref belongs only inside "
            "introduce_connected_person.arguments", prompt)
        self.assertIn('"speaker_forbidden_action_argument_keys"', prompt)
        speaker_template = prompt.split(
            "TOP-LEVEL AND SCENE KEY SKELETON ONLY", 1)[1]
        speaker_template = speaker_template.split("FINAL ACTION CHECK", 1)[0]
        self.assertNotIn('"connection_target_ref"', speaker_template)

    def test_unavailable_actions_are_not_exposed_by_main_planner_prompt(self):
        _, source, placements, _, planning, availability, _ = outline_fixture(5)
        self.assertFalse(availability["actions"][
            "resolve_or_deescalate_conflict"]["available"])
        prompt, _ = build_planner_prompt_v11(
            root_id=planning["root_id"], planning_state=planning,
            source_anchor_registry=source, placement_choice_registry=placements,
            action_availability_view=availability,
            dialogue_first_participants=True)
        self.assertNotIn("resolve_or_deescalate_conflict", prompt)

    def test_participant_alias_coverage_is_audit_only(self):
        from scripts.current_method.restart_n5_spell.world_memory_slim_v1 import (
            dialogue_first_participant_planner_probe_v11 as probe,
        )

        _, source, placements, outline, planning, availability, refs = outline_fixture(5)
        outline["scenes"][0]["speaker_participants"][1] = _new(
            "passerby", "Passerby")
        result = probe._validate_dialogue_first_outline(
            outline, source_anchor_registry=source,
            placement_choice_registry=placements,
            authoritative_entity_refs=refs,
            authoritative_entities=planning["entities"],
            action_availability_view=availability)
        self.assertTrue(result["valid"], result["errors"])
        coverage = result["dialogue_first_participant_coverage"]
        self.assertTrue(coverage["valid"])
        self.assertEqual(coverage["declared_aliases"], ["passerby"])
        self.assertEqual(coverage["reused_aliases"], [])

    def test_probe2_tmp_as_alias_is_rejected_without_history_rewrite(self):
        self.assertEqual(hash_tree(PROBE_2)["tree_sha256"], PROBE_2_SHA)
        parsed = json.loads((PROBE_2 / "planner_response_parsed.json").read_text())
        initialized = initialize_source_world_memory_v2(*source_fixture())
        registry = compile_round_participant_registry(
            scenes=parsed["scenes"],
            authoritative_entities=initialized["memory"]["entities"])
        self.assertFalse(registry["valid"])
        self.assertTrue(any("speaker_schema" in error
                            for error in registry["errors"]))
        self.assertIn(":TMP:", json.dumps(parsed))

    def test_probe3_alias_shadow_preserves_duplicate_action_gate(self):
        from scripts.current_method.restart_n5_spell.world_memory_slim_v1 import (
            dialogue_first_participant_planner_probe_v11 as probe,
        )
        from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_preflight_b2_v11 import (
            DEFAULT_INPUT,
        )

        self.assertEqual(hash_tree(PROBE_3)["tree_sha256"], PROBE_3_SHA)
        original = json.loads((PROBE_3 / "planner_response_parsed.json").read_text())
        values = probe._build_dialogue_first_probe_input(DEFAULT_INPUT)
        shadow = migrate_legacy_connected_person_contract_v2(original)
        declared = False
        for scene in shadow["scenes"]:
            speakers = []
            for speaker in scene["speaker_participants"]:
                if speaker.get("participant_alias") != "local_2":
                    speakers.append(speaker)
                    continue
                if not declared:
                    declaration = deepcopy(speaker)
                    declaration.pop("participant_scope", None)
                    speakers.append(declaration)
                    declared = True
                else:
                    speakers.append({
                        "participant_alias": "local_2",
                        "scene_role": speaker["scene_role"],
                        "intents": deepcopy(speaker["intents"]),
                    })
            scene["speaker_participants"] = speakers
        duplicate = validate_planner_outline_v11(
            shadow, source_anchor_registry=values["source_anchor_registry"],
            placement_choice_registry=values["placement_choice_registry"],
            authoritative_entity_refs={row["entity_ref"] for row in values["role_table"]},
            authoritative_entities=values["planning_state"]["entities"],
            action_availability_view=values["action_availability_view"])
        self.assertFalse(duplicate["valid"])
        self.assertFalse(any("unknown_entity:local_2" in error or
                             "unknown_typed_ref:participant_refs:local_2" in error or
                             "speaker_schema" in error for error in duplicate["errors"]))
        self.assertTrue(any("redundant_independent_action" in error
                            for error in duplicate["errors"]))

        repaired = deepcopy(shadow)
        repaired["scenes"][6]["primary_action"] = {
            "action_mode": "continuation",
            "continuation_of": "S05.primary_action",
            "progression": {
                "stage": "consolidation",
                "new_step": "A turns the earlier emotional resolution into a concrete plan for the next family conversation.",
            },
        }
        repaired["scenes"][6]["secondary_action"] = {
            "action_mode": "continuation",
            "continuation_of": "S05.secondary_action",
            "progression": {
                "stage": "development",
                "new_step": "Friend C turns the earlier perspective into a specific follow-up arrangement with A and B.",
            },
        }
        validation = validate_planner_outline_v11(
            repaired, source_anchor_registry=values["source_anchor_registry"],
            placement_choice_registry=values["placement_choice_registry"],
            authoritative_entity_refs={row["entity_ref"] for row in values["role_table"]},
            authoritative_entities=values["planning_state"]["entities"],
            action_availability_view=values["action_availability_view"])
        self.assertTrue(validation["valid"], validation["errors"])
        compiled = compile_planner_outline_v11(
            repaired, source_anchor_registry=values["source_anchor_registry"],
            placement_choice_registry=values["placement_choice_registry"],
            planning_state=values["planning_state"],
            action_availability_view=values["action_availability_view"])
        self.assertEqual(compiled["round_participant_registry"][
            "alias_to_round_ref"]["local_2"], "R01:P01")
        self.assertEqual(compiled["scenes"][3]["secondary_action"][
            "resolved_arguments"]["sender"], "R01:P01")
        self.assertIn("R01:P01", compiled["scenes"][5]["secondary_action"][
            "resolved_arguments"]["participant_refs"])

    def test_canary16_attempt1_action_alias_shadow_passes_all_planner_gates(self):
        from scripts.current_method.restart_n5_spell.world_memory_slim_v1 import (
            dialogue_first_participant_planner_probe_v11 as probe,
        )
        from scripts.current_method.restart_n5_spell.world_memory_slim_v1.growth_profile_v11 import (
            frozen_person_event_growth_profile, validate_planner_growth_profile_v11,
        )
        from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_preflight_b2_v11 import (
            DEFAULT_INPUT,
        )
        from scripts.current_method.restart_n5_spell.world_memory_slim_v1.structural_expansion_gate import (
            validate_round_blueprint_structure,
        )

        self.assertEqual(hash_tree(CANARY_16)["tree_sha256"], CANARY_16_SHA)
        historical_outline = json.loads((CANARY_16 /
            "attempts/planner_v11_attempt_01.json").read_text())["parsed_response"]
        outline = migrate_legacy_connected_person_contract_v2(historical_outline)
        values = probe._build_dialogue_first_probe_input(DEFAULT_INPUT)
        validation = validate_planner_outline_v11(
            outline, source_anchor_registry=values["source_anchor_registry"],
            placement_choice_registry=values["placement_choice_registry"],
            authoritative_entity_refs={row["entity_ref"] for row in values["role_table"]},
            authoritative_entities=values["planning_state"]["entities"],
            action_availability_view=values["action_availability_view"])
        self.assertTrue(validation["valid"], validation["errors"])
        growth = validate_planner_growth_profile_v11(
            outline, profile=frozen_person_event_growth_profile())
        self.assertTrue(growth["valid"], growth["errors"])
        compiled = compile_planner_outline_v11(
            outline, source_anchor_registry=values["source_anchor_registry"],
            placement_choice_registry=values["placement_choice_registry"],
            planning_state=values["planning_state"],
            action_availability_view=values["action_availability_view"])
        structural = validate_round_blueprint_structure(
            compiled_plan=compiled,
            source_anchor_registry=values["source_anchor_registry"],
            placement_choice_registry=values["placement_choice_registry"],
            planning_state=values["planning_state"],
            complete_dialogue=values["complete_dialogue"])
        self.assertTrue(structural["valid"], structural["errors"])
        participant = compiled["round_participant_registry"]["participants"][0]
        self.assertEqual(participant["participant_ref"], "R01:P01")
        self.assertEqual(participant["declaration_action_ref"], "S01.primary_action")
        self.assertEqual(participant["appearance_scene_aliases"],
                         ["s2", "s3", "s4", "s5", "s6", "s7"])
        self.assertEqual(hash_tree(CANARY_16)["tree_sha256"], CANARY_16_SHA)

    def test_probe5_namespace_shadow_repair_passes_all_planner_gates(self):
        from scripts.current_method.restart_n5_spell.world_memory_slim_v1 import (
            dialogue_first_participant_planner_probe_v11 as probe,
        )
        from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_preflight_b2_v11 import (
            DEFAULT_INPUT,
        )
        from scripts.current_method.restart_n5_spell.world_memory_slim_v1.structural_expansion_gate import (
            validate_round_blueprint_structure,
        )

        self.assertEqual(hash_tree(PROBE_5)["tree_sha256"], PROBE_5_SHA)
        original = json.loads((PROBE_5 / "planner_response_parsed.json").read_text())
        values = probe._build_dialogue_first_probe_input(DEFAULT_INPUT)
        authoritative_refs = {row["entity_ref"] for row in values["role_table"]}
        original_validation = probe._validate_dialogue_first_outline(
            original, source_anchor_registry=values["source_anchor_registry"],
            placement_choice_registry=values["placement_choice_registry"],
            authoritative_entity_refs=authoritative_refs,
            authoritative_entities=values["planning_state"]["entities"],
            action_availability_view=values["action_availability_view"])
        self.assertFalse(original_validation["valid"])
        self.assertFalse(original_validation[
            "dialogue_first_participant_coverage"]["valid"])
        self.assertTrue(any("participant_alias_not_declared" in error
                            for error in original_validation["errors"]))
        self.assertFalse(any("unknown_typed_ref" in error
                             for error in original_validation["errors"]))

        repaired = migrate_legacy_connected_person_contract_v2(original)
        name_to_ref = {row["display_name"]: row["entity_ref"]
                       for row in values["planning_state"]["entities"]}
        replacements = {**name_to_ref, "Manager C": "local_1"}
        for scene in repaired["scenes"]:
            for speaker in scene["speaker_participants"]:
                alias = speaker.get("participant_alias")
                if alias in authoritative_refs:
                    speaker["participant_ref"] = speaker.pop("participant_alias")
            for slot in ("primary_action", "secondary_action"):
                action = scene.get(slot)
                if not isinstance(action, dict) or action.get("action_mode") != "new":
                    continue
                arguments = action.get("arguments") or {}
                for field, value in list(arguments.items()):
                    if field == "display_name":
                        continue
                    if isinstance(value, str):
                        arguments[field] = replacements.get(value, value)
                    elif isinstance(value, list):
                        arguments[field] = [replacements.get(item, item)
                                            for item in value]
                    elif isinstance(value, dict):
                        for child_field, child in list(value.items()):
                            if isinstance(child, str):
                                value[child_field] = replacements.get(child, child)
        repaired["scenes"][6]["link_type"] = "direct_continuation"
        validation = probe._validate_dialogue_first_outline(
            repaired, source_anchor_registry=values["source_anchor_registry"],
            placement_choice_registry=values["placement_choice_registry"],
            authoritative_entity_refs=authoritative_refs,
            authoritative_entities=values["planning_state"]["entities"],
            action_availability_view=values["action_availability_view"])
        self.assertTrue(validation["valid"], validation["errors"])
        self.assertTrue(validation["dialogue_first_participant_coverage"]["valid"])
        compiled = compile_planner_outline_v11(
            repaired, source_anchor_registry=values["source_anchor_registry"],
            placement_choice_registry=values["placement_choice_registry"],
            planning_state=values["planning_state"],
            action_availability_view=values["action_availability_view"])
        structural = validate_round_blueprint_structure(
            compiled_plan=compiled,
            source_anchor_registry=values["source_anchor_registry"],
            placement_choice_registry=values["placement_choice_registry"],
            planning_state=values["planning_state"],
            complete_dialogue=values["complete_dialogue"])
        self.assertTrue(structural["valid"], structural["errors"])
        self.assertEqual(compiled["round_participant_registry"][
            "alias_to_round_ref"], {"local_1": "R01:P01"})
        self.assertNotIn("scene_presence", json.dumps(repaired))
        self.assertEqual(hash_tree(PROBE_5)["tree_sha256"], PROBE_5_SHA)

    def test_probe6_action_target_duplication_shadow_passes_all_planner_gates(self):
        from scripts.current_method.restart_n5_spell.world_memory_slim_v1 import (
            dialogue_first_participant_planner_probe_v11 as probe,
        )
        from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_preflight_b2_v11 import (
            DEFAULT_INPUT,
        )
        from scripts.current_method.restart_n5_spell.world_memory_slim_v1.structural_expansion_gate import (
            validate_round_blueprint_structure,
        )

        self.assertEqual(hash_tree(PROBE_6)["tree_sha256"], PROBE_6_SHA)
        original = json.loads((PROBE_6 / "planner_response_parsed.json").read_text())
        values = probe._build_dialogue_first_probe_input(DEFAULT_INPUT)
        authoritative_refs = {row["entity_ref"] for row in values["role_table"]}
        original_validation = probe._validate_dialogue_first_outline(
            original, source_anchor_registry=values["source_anchor_registry"],
            placement_choice_registry=values["placement_choice_registry"],
            authoritative_entity_refs=authoritative_refs,
            authoritative_entities=values["planning_state"]["entities"],
            action_availability_view=values["action_availability_view"])
        self.assertFalse(original_validation["valid"])
        self.assertEqual(original_validation["errors"], [
            "scene:2:speaker:3:schema",
            "scene:2:speaker:3:speaker_schema",
        ])

        repaired = deepcopy(original)
        declaration = repaired["scenes"][1]["speaker_participants"][2]
        self.assertEqual(
            declaration.pop("connection_target_ref"),
            repaired["scenes"][1]["primary_action"]["arguments"][
                "connection_target_ref"])
        validation = probe._validate_dialogue_first_outline(
            repaired, source_anchor_registry=values["source_anchor_registry"],
            placement_choice_registry=values["placement_choice_registry"],
            authoritative_entity_refs=authoritative_refs,
            authoritative_entities=values["planning_state"]["entities"],
            action_availability_view=values["action_availability_view"])
        self.assertTrue(validation["valid"], validation["errors"])
        compiled = compile_planner_outline_v11(
            repaired, source_anchor_registry=values["source_anchor_registry"],
            placement_choice_registry=values["placement_choice_registry"],
            planning_state=values["planning_state"],
            action_availability_view=values["action_availability_view"])
        dag = build_scene_dependency_dag(compiled)
        self.assertEqual(len(dag["nodes"]), 7)
        self.assertTrue(dag["scene_dependency_dag_sha256"])
        structural = validate_round_blueprint_structure(
            compiled_plan=compiled,
            source_anchor_registry=values["source_anchor_registry"],
            placement_choice_registry=values["placement_choice_registry"],
            planning_state=values["planning_state"],
            complete_dialogue=values["complete_dialogue"])
        self.assertTrue(structural["valid"], structural["errors"])
        self.assertNotIn("connection_target_ref", declaration)
        self.assertEqual(hash_tree(PROBE_6)["tree_sha256"], PROBE_6_SHA)

    def test_persistent_candidate_compiles_atomic_person_and_connection(self):
        initialized, registry, _, source_registry, realized, review = _round_fixture()
        validation = validate_participant_persistence_reviews(
            [review], round_participant_registry=registry,
            realized_scenes=realized, source_turn_registry=source_registry,
            authoritative_memory=initialized["memory"],
            accepted_scene_refs=["R01:S01"])
        self.assertTrue(validation["valid"], validation["errors"])
        result = compile_participant_persistence_updates(
            reviews=[review], round_participant_registry=registry,
            realized_scenes=realized, source_turn_registry=source_registry,
            authoritative_memory=initialized["memory"],
            accepted_scene_refs=["R01:S01"])
        self.assertTrue(result["valid"], result["errors"])
        proposals = result["compiled_proposals"]
        self.assertEqual([row["operation"] for row in proposals],
                         ["add_person", "add_interaction_link"])
        self.assertEqual(proposals[0]["atomic_group_ref"],
                         proposals[1]["atomic_group_ref"])
        self.assertEqual(proposals[1]["proposal_dependencies"],
                         [proposals[0]["proposal_ref"]])
        self.assertNotIn("add_event", [row["operation"] for row in proposals])
        self.assertEqual(result["orphan_person_count"], 0)
        self.assertEqual(result["partial_atomic_commit_count"], 0)

    def test_round_validator_prompt_extension_reviews_scope_separately(self):
        _, registry, _, _, _, _ = _round_fixture()
        prompt = build_participant_persistence_prompt_extension(registry)
        schema = participant_persistence_review_schema()
        self.assertNotIn("planned_scope", prompt)
        self.assertIn("Planner does not decide persistence", prompt)
        self.assertIn("persistence_decision", prompt)
        self.assertIn("Speaking alone never creates Memory", prompt)
        self.assertEqual(schema["persistence_decision_enum"], [
            "do_not_persist", "invalid", "persist", "uncertain"])
        self.assertIn("PARTICIPANT_REVIEWS TEMPLATE", prompt)
        combined = build_dialogue_first_round_validator_prompt(
            {"scenes": []}, registry)
        self.assertIn("participant_reviews", combined)
        self.assertTrue(combined.endswith(
            "NEVER OUTPUT type, properties, required, or additionalProperties."))

    def test_binder_allocates_official_entity_only_after_persist_review(self):
        initialized, registry, _, source_registry, realized, review = _round_fixture()
        result = compile_participant_persistence_updates(
            reviews=[review], round_participant_registry=registry,
            realized_scenes=realized, source_turn_registry=source_registry,
            authoritative_memory=initialized["memory"], accepted_scene_refs=["R01:S01"])
        proposals = result["compiled_proposals"]
        plan = build_allocation_plan(
            root_id=initialized["memory"]["root_id"], round_index=1, scene_index=1,
            compiled_proposals=proposals,
            high_water_marks=initialized["memory"]["high_water_marks"])
        round_ref = registry["participants"][0]["participant_ref"]
        binder_ref = next(value for key, value in plan["temporary_ref_allocations"].items()
                          if ":TMP:E:DU" in key)
        official_ref = binder_ref
        self.assertNotEqual(round_ref, official_ref)
        claims = [{"proposal_ref": row["proposal_ref"],
                   "realization_status": "realized"} for row in proposals]
        reviews = [{"proposal_ref": row["proposal_ref"],
                    "review_decision": "confirmed",
                    "verification_status": "semantically_verified"} for row in proposals]
        evidence = {row["proposal_ref"]: ["R01:S01:T002"] for row in proposals}
        sha = {key: [hashlib.sha256(
            b"I joined this project to verify the revised schedule.").hexdigest()]
               for key in evidence}
        memory = apply_scene_commit(
            initialized["memory"], scene_ref="R01:S01", round_index=1,
            scene_index=1, compiled_proposals=proposals,
            proposal_claims=claims,
            committer_output={"scope_status": "clean",
                              "world_consistency_status": "clean",
                              "proposal_reviews": reviews},
            allocation_plan=plan, evidence_final_turn_refs=evidence,
            evidence_text_sha256=sha)
        self.assertIn(official_ref, {row["entity_ref"] for row in memory["entities"]})
        self.assertNotIn(round_ref, {row["entity_ref"] for row in memory["entities"]})
        added = next(row for row in memory["entities"]
                     if row["entity_ref"] == official_ref)
        self.assertEqual(added["scene_presence"], "speaking")
        with tempfile.TemporaryDirectory() as temp:
            tx = commit_scene_transaction(
                run_root=Path(temp), run_id="dialogue-first-test",
                scene_plan=realized[0], proposal_fingerprints=[
                    row["proposal_fingerprint"] for row in proposals],
                allocation_plan=plan, artifacts={"world_memory_final.json": memory})
            self.assertEqual(tx["status"], "COMMITTED")

    def test_insufficient_or_rejected_candidate_never_allocates(self):
        initialized, registry, _, source_registry, realized, review = _round_fixture()
        review["persistence_decision"] = "do_not_persist"
        review["connection_status"] = "uncertain"
        review["verified_connection"] = None
        result = compile_participant_persistence_updates(
            reviews=[review], round_participant_registry=registry,
            realized_scenes=realized, source_turn_registry=source_registry,
            authoritative_memory=initialized["memory"], accepted_scene_refs=["R01:S01"])
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["compiled_proposals"], [])
        self.assertEqual(len(result["audit_only_reviews"]), 1)
        review["persistence_decision"] = "persist"
        rejected_source = compile_participant_persistence_updates(
            reviews=[review], round_participant_registry=registry,
            realized_scenes=realized, source_turn_registry=source_registry,
            authoritative_memory=initialized["memory"], accepted_scene_refs=[])
        self.assertFalse(rejected_source["valid"])
        self.assertEqual(rejected_source["compiled_proposals"], [])

    def test_invalid_connection_blocks_entire_atomic_group(self):
        initialized, registry, _, source_registry, realized, review = _round_fixture()
        review["verified_connection"]["target_ref"] = "other-root:E000001"
        result = compile_participant_persistence_updates(
            reviews=[review], round_participant_registry=registry,
            realized_scenes=realized, source_turn_registry=source_registry,
            authoritative_memory=initialized["memory"], accepted_scene_refs=["R01:S01"])
        self.assertFalse(result["valid"])
        self.assertEqual(result["compiled_proposals"], [])
        self.assertEqual(result["accepted_updates"], [])
        self.assertEqual(result["orphan_person_count"], 0)
        self.assertEqual(result["partial_atomic_commit_count"], 0)

    def test_same_name_is_not_fuzzy_merged(self):
        initialized, registry, _, source_registry, realized, review = _round_fixture(
            same_name=True)
        self.assertTrue(any("display_name_collision_not_merged" in value
                            for value in registry["warnings"]))
        result = compile_participant_persistence_updates(
            reviews=[review], round_participant_registry=registry,
            realized_scenes=realized, source_turn_registry=source_registry,
            authoritative_memory=initialized["memory"], accepted_scene_refs=["R01:S01"])
        self.assertTrue(result["valid"], result["errors"])
        allocation = build_allocation_plan(
            root_id=initialized["memory"]["root_id"], round_index=1, scene_index=1,
            compiled_proposals=result["compiled_proposals"],
            high_water_marks=initialized["memory"]["high_water_marks"])
        new_ref = next(iter(allocation["temporary_ref_allocations"].values()))
        self.assertNotIn(new_ref, {row["entity_ref"] for row in
                                   initialized["memory"]["entities"]})


if __name__ == "__main__":
    unittest.main()
