from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from tests.test_v11_shared_narrative_action_menu import migrated_fixture
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.action_availability_v11 import (
    compile_action_availability_view,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.durable_update_compiler import (
    compile_durable_updates,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.growth_admission import (
    audit_authoritative_growth,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.growth_profile_v11 import (
    evaluate_growth_profile_admission,
    frozen_person_event_growth_profile,
    validate_growth_profile_definition,
    validate_planner_growth_profile_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_contracts_v11 import (
    build_planner_prompt_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_preflight_b2_v11 import (
    DEFAULT_INPUT,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_fresh_e2e_v11 import (
    run_mock_hybrid_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.narrative_action_contracts import (
    validate_action_instance,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.planning_state_view import (
    compile_planning_state_view,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.planner_only_probe_v11 import (
    build_planner_probe_input,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_expansion_validator_v1 import (
    validate_round_validator_output,
)


CANARY14 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_continuous_expansion_v11_live_canary_14"
)
CANARY14_SHA = "f950428152ab5937d5ba41c0774d35e60e3be63a81be450767ebc1ae6a2596b7"
PERSON_EVENT_PROBE = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_v11_person_event_planner_probe_1"
)
PERSON_EVENT_PROBE_SHA = (
    "2bfb3faf8dbce246bcf7f58b359d5b6367403979836bc873722b9618b0cac63c"
)


def person_event_inputs():
    inputs, _, _, plan = migrated_fixture(6)
    entities = [row["entity_ref"] for row in inputs["planning_state"]["entities"]]
    plan["scenes"][0]["primary_action"] = {
        "action_mode": "new", "action_type": "introduce_connected_person",
        "arguments": {
            "participant_alias": "co_worker_c",
            "display_name": "Co-worker C", "entity_role": "project designer",
            "connection_type": "existing_person",
            "connection_target_ref": entities[0],
            "connection_description": (
                "C was introduced through A as a participant in the same project."),
        },
    }
    plan["scenes"][1]["primary_action"] = {
        "action_mode": "new", "action_type": "introduce_connected_event",
        "arguments": {
            "event_alias": "laptop_purchase",
            "description": "A completed the laptop purchase yesterday.",
            "participant_refs": [entities[0]],
            "connected_existing": {"ref_type": "person", "ref": entities[0]},
            "narrative_time_expression": "yesterday",
            "narrative_time_relation": "past",
            "occurrence_status": "occurred",
            "event_consequence": "A has less money available for another purchase.",
        },
    }
    inputs["planner_output"] = plan
    return inputs


class PersonEventGrowthTests(unittest.TestCase):
    def test_growth_profile_definition_and_planner_coverage(self):
        profile = frozen_person_event_growth_profile()
        self.assertTrue(validate_growth_profile_definition(profile)["valid"])
        plan = person_event_inputs()["planner_output"]
        validation = validate_planner_growth_profile_v11(plan, profile=profile)
        self.assertTrue(validation["valid"], validation["errors"])
        self.assertEqual(validation["distinct_growth_scene_count"], 2)
        self.assertTrue(validation["distinct_scene_distribution_preferred"])
        missing = deepcopy(plan)
        missing["scenes"][1]["primary_action"] = {
            "action_mode": "new", "action_type": "develop_character_or_task_state",
            "arguments": {"target": missing["scenes"][1]["speaker_participants"][0][
                "participant_ref"], "dimension": "confidence", "direction": "increase",
                "trigger": "a concrete result"},
        }
        invalid = validate_planner_growth_profile_v11(missing, profile=profile)
        self.assertIn("growth_profile_missing_new_action:introduce_connected_event",
                      invalid["errors"])

    def test_person_and_event_growth_may_share_one_scene(self):
        plan = person_event_inputs()["planner_output"]
        event_action = deepcopy(plan["scenes"][1]["primary_action"])
        plan["scenes"][0]["secondary_action"] = event_action
        plan["scenes"][1]["primary_action"] = {
            "action_mode": "new",
            "action_type": "develop_character_or_task_state",
            "arguments": {
                "target": plan["scenes"][1]["speaker_participants"][0][
                    "participant_ref"],
                "dimension": "confidence", "direction": "increase",
                "trigger": "a concrete result",
            },
        }
        validation = validate_planner_growth_profile_v11(
            plan, profile=frozen_person_event_growth_profile())
        self.assertTrue(validation["valid"], validation["errors"])
        self.assertEqual(validation["distinct_growth_scene_count"], 1)
        self.assertFalse(validation["distinct_scene_distribution_preferred"])
        self.assertTrue(validation["distinct_scene_distribution_is_advisory"])

    def test_connected_event_temporal_relation_is_structural_not_keyword_based(self):
        action = deepcopy(person_event_inputs()["planner_output"]["scenes"][1][
            "primary_action"])
        self.assertTrue(validate_action_instance(action)["valid"])
        action["arguments"]["narrative_time_expression"] = "next week"
        action["arguments"]["narrative_time_relation"] = "future"
        invalid = validate_action_instance(action)
        self.assertFalse(invalid["valid"])
        self.assertTrue(any("event_temporal_status_mismatch:occurred:future" in error
                            for error in invalid["errors"]))
        action["arguments"]["narrative_time_expression"] = "a deliberately opaque label"
        action["arguments"]["narrative_time_relation"] = "past"
        self.assertTrue(validate_action_instance(action)["valid"])

    def test_person_event_probe_shadow_relaxes_distribution_but_not_time_contract(self):
        self.assertEqual(hash_tree(PERSON_EVENT_PROBE)["tree_sha256"],
                         PERSON_EVENT_PROBE_SHA)
        parsed = json.loads((PERSON_EVENT_PROBE /
            "planner_response_parsed.json").read_text(encoding="utf-8"))
        growth = validate_planner_growth_profile_v11(
            parsed, profile=frozen_person_event_growth_profile())
        self.assertTrue(growth["valid"], growth["errors"])
        self.assertEqual(growth["distinct_growth_scene_count"], 1)
        event_action = parsed["scenes"][0]["secondary_action"]
        event_validation = validate_action_instance(event_action)
        self.assertFalse(event_validation["valid"])
        self.assertTrue(any("narrative_time_relation" in error
                            for error in event_validation["errors"]))

    def test_growth_profile_is_frozen_in_planner_prompt_without_confirming(self):
        values = build_planner_probe_input(DEFAULT_INPUT)
        profile = frozen_person_event_growth_profile()
        prompt, prompt_sha = build_planner_prompt_v11(
            root_id=values["planning_state"]["root_id"], planning_state=values["planning_state"],
            source_anchor_registry=values["source_anchor_registry"],
            placement_choice_registry=values["placement_choice_registry"],
            action_contracts=values["shared_narrative_action_contracts"],
            action_availability_view=values["action_availability_view"],
            growth_profile=profile)
        self.assertIn('"profile_id":"person_event_growth_canary_v1"', prompt)
        self.assertIn("not a Memory confirmation instruction", prompt)
        self.assertIn("same scene or different connected scenes", prompt)
        self.assertNotIn("Across at least two connected scenes", prompt)
        self.assertIn('"participant_alias"', prompt)
        self.assertIn('"event_alias"', prompt)
        self.assertNotIn(":TMP:", prompt)
        self.assertNotIn('"available":false', prompt)
        self.assertEqual(prompt_sha, build_planner_prompt_v11(
            root_id=values["planning_state"]["root_id"], planning_state=values["planning_state"],
            source_anchor_registry=values["source_anchor_registry"],
            placement_choice_registry=values["placement_choice_registry"],
            action_contracts=values["shared_narrative_action_contracts"],
            action_availability_view=values["action_availability_view"],
            growth_profile=profile)[1])

    def test_narrative_alias_format_is_required(self):
        inputs = person_event_inputs()
        availability = compile_action_availability_view(
            planning_state=inputs["planning_state"])
        action = deepcopy(inputs["planner_output"]["scenes"][0]["primary_action"])
        action["arguments"]["participant_alias"] = "TMP:E:P01"
        validation = validate_action_instance(
            action,
            known_world_refs=set(availability["typed_reference_registry"][
                "all_typed_refs"]),
            authoritative_entity_refs=set(availability["typed_reference_registry"][
                "participant_refs"]), action_availability_view=availability)
        self.assertFalse(validation["valid"])
        self.assertTrue(any("narrative_alias_invalid" in error
                            for error in validation["errors"]))

    def test_legacy_planner_creation_keys_are_rejected(self):
        plan = person_event_inputs()["planner_output"]
        person = deepcopy(plan["scenes"][0]["primary_action"])
        person["arguments"]["temporary_entity_key"] = "root:TMP:E:P01"
        person["arguments"].pop("participant_alias")
        self.assertFalse(validate_action_instance(person)["valid"])
        event = deepcopy(plan["scenes"][1]["primary_action"])
        event["arguments"]["temporary_event_key"] = "root:TMP:EV:EV01"
        event["arguments"].pop("event_alias")
        self.assertFalse(validate_action_instance(event)["valid"])

    def test_connected_event_is_shared_action_with_typed_refs(self):
        inputs = person_event_inputs()
        action = inputs["planner_output"]["scenes"][1]["primary_action"]
        availability = compile_action_availability_view(
            planning_state=inputs["planning_state"])
        self.assertIn("introduce_connected_event", availability["actions"])
        self.assertTrue(validate_action_instance(
            action,
            known_world_refs=set(availability["typed_reference_registry"]["all_typed_refs"]),
            authoritative_entity_refs=set(
                availability["typed_reference_registry"]["participant_refs"]),
            action_availability_view=availability)["valid"])

    def test_person_and_occurred_event_compile_and_commit(self):
        inputs = person_event_inputs()
        with tempfile.TemporaryDirectory() as temp:
            result = run_mock_hybrid_v11(
                **inputs, run_root=Path(temp), case="all_accept")
        self.assertTrue(result["completed"], result.get("detail"))
        operations = [row["compiled_proposal"]["operation"]
                      for row in result["durable_update_compile_result"]["accepted_updates"]]
        self.assertEqual(operations[:3], [
            "add_person", "add_interaction_link", "add_event"])
        person, connection = [row["compiled_proposal"] for row in
                              result["durable_update_compile_result"]["accepted_updates"][:2]]
        self.assertEqual(person["compiled_payload"]["scene_presence"],
                         "mentioned_only")
        self.assertNotIn(":TMP:", json.dumps(inputs["planner_output"]))
        self.assertEqual(person["provenance"]["source_participant_alias"],
                         "co_worker_c")
        self.assertTrue(person["temporary_refs"])
        self.assertTrue(all(":TMP:E:DU" in value
                            for value in person["temporary_refs"]))
        event = result["durable_update_compile_result"]["accepted_updates"][2][
            "compiled_proposal"]
        self.assertEqual(event["provenance"]["source_event_alias"],
                         "laptop_purchase")
        self.assertEqual(person["atomic_group_ref"], connection["atomic_group_ref"])
        self.assertEqual(connection["proposal_dependencies"], [person["proposal_ref"]])
        self.assertEqual(len(result["transactions"]), 2)
        self.assertTrue(all(row["status"] == "COMMITTED" for row in result["transactions"]))
        audit = audit_authoritative_growth(
            memory_initial=inputs["authoritative_memory"],
            memory_final=result["memory_final"],
            accepted_updates=result["durable_update_compile_result"]["accepted_updates"])
        self.assertTrue(audit["person_growth_ready"], audit)
        self.assertTrue(audit["event_growth_ready"], audit)
        self.assertEqual(audit["orphan_entity_refs"], [])
        self.assertEqual(audit["partial_atomic_groups"], [])
        self.assertEqual(audit["cross_root_refs"], [])
        admission = evaluate_growth_profile_admission(
            audit, profile=frozen_person_event_growth_profile())
        self.assertTrue(admission["profile_ready"], admission)

    def test_connected_person_scene_presence_is_derived_from_actual_speech(self):
        inputs = person_event_inputs()
        with tempfile.TemporaryDirectory() as temp:
            base = run_mock_hybrid_v11(
                **inputs, run_root=Path(temp), case="accept_no_updates")
        plan = deepcopy(base["compiled_round_plan"])
        realized = deepcopy(base["realized_scenes"])
        scene = plan["scenes"][0]
        alias = scene["primary_action"]["resolved_arguments"]["participant_alias"]
        round_ref = "R01:P01"
        plan["round_participant_registry"] = {
            "alias_to_round_ref": {alias: round_ref}}
        evidence = realized[0]["dialogue_turns"][0]["turn_ref"]
        realized[0]["dialogue_turns"][0]["speaker_ref"] = round_ref
        compiled = compile_durable_updates(
            candidates=[{
                "candidate_update_key": "DU_SPEAKING",
                "source_scene_ref": scene["scene_ref"],
                "source_action_role": "primary",
                "evidence_turn_refs": [evidence],
                "verification_status": "confirmed", "atomic_group_key": None,
                "depends_on_candidate_keys": [],
                "verified_connection": {
                    "verification_status": "confirmed",
                    "connection_kind": "interaction_link", "relation_type": None,
                    "evidence_turn_refs": [evidence],
                    "reason": "the speaking participant states the connection",
                },
            }],
            compiled_round_plan=plan,
            scene_decision_closure=base["scene_decision_closure"],
            realized_scenes=realized,
            authoritative_memory=inputs["authoritative_memory"],
            planning_state=inputs["planning_state"],
            action_menu=inputs["action_menu"])
        person = next(
            row["compiled_proposal"]
            for row in compiled["accepted_updates"]
            if row["compiled_proposal"]["operation"] == "add_person")
        self.assertEqual(person["compiled_payload"]["scene_presence"], "speaking")

    def test_round2_projection_discovers_generated_non_speaker_entity(self):
        inputs = person_event_inputs()
        with tempfile.TemporaryDirectory() as temp:
            result = run_mock_hybrid_v11(
                **inputs, run_root=Path(temp), case="all_accept")
        planning = compile_planning_state_view(
            memory=result["memory_final"], interaction_links=())
        initial_refs = {row.get("entity_ref") or row.get("entity_id")
                        for row in inputs["authoritative_memory"]["entities"]}
        generated = [row for row in planning["entities"]
                     if row["origin"] == "generated_extension" and
                     row["entity_ref"] not in initial_refs]
        self.assertEqual([row["display_name"] for row in generated], ["Co-worker C"])
        self.assertTrue(generated[0]["source_turn_refs"])
        self.assertTrue(any(row["record_type"] == "interaction_link"
                            for row in planning["entity_connections"]))

    def test_unconfirmed_person_connection_never_allocates(self):
        inputs = person_event_inputs()
        with tempfile.TemporaryDirectory() as temp:
            base = run_mock_hybrid_v11(
                **inputs, run_root=Path(temp), case="accept_no_updates")
        plan = deepcopy(base["compiled_round_plan"])
        scene = plan["scenes"][0]
        evidence = base["realized_scenes"][0]["dialogue_turns"][0]["turn_ref"]
        compiled = compile_durable_updates(
            candidates=[{
                "candidate_update_key": "DU_PERSON", "source_scene_ref": scene["scene_ref"],
                "source_action_role": "primary", "evidence_turn_refs": [evidence],
                "verification_status": "confirmed", "atomic_group_key": None,
                "depends_on_candidate_keys": [],
                "verified_connection": {
                    "verification_status": "uncertain",
                    "connection_kind": "interaction_link", "relation_type": None,
                    "evidence_turn_refs": [evidence], "reason": "support is ambiguous",
                },
            }],
            compiled_round_plan=plan,
            scene_decision_closure=base["scene_decision_closure"],
            realized_scenes=base["realized_scenes"],
            authoritative_memory=inputs["authoritative_memory"],
            planning_state=inputs["planning_state"], action_menu=inputs["action_menu"])
        self.assertEqual(compiled["accepted_updates"], [])
        self.assertTrue(any("connected_person_connection_not_confirmed" in row["errors"]
                            for row in compiled["rejected_updates"]))

    def test_rejected_person_candidate_does_not_poison_later_valid_identity(self):
        inputs = person_event_inputs()
        with tempfile.TemporaryDirectory() as temp:
            base = run_mock_hybrid_v11(
                **inputs, run_root=Path(temp), case="accept_no_updates")
        scene = base["compiled_round_plan"]["scenes"][0]
        evidence = base["realized_scenes"][0]["dialogue_turns"][0]["turn_ref"]

        def candidate(key, status):
            return {
                "candidate_update_key": key,
                "source_scene_ref": scene["scene_ref"],
                "source_action_role": "primary",
                "evidence_turn_refs": [evidence],
                "verification_status": status,
                "atomic_group_key": None,
                "depends_on_candidate_keys": [],
                "verified_connection": {
                    "verification_status": "confirmed",
                    "connection_kind": "interaction_link",
                    "relation_type": None,
                    "evidence_turn_refs": [evidence],
                    "reason": "the dialogue explicitly connects C to A",
                },
            }

        compiled = compile_durable_updates(
            candidates=[candidate("DU_REJECTED", "rejected"),
                        candidate("DU_ACCEPTED", "confirmed")],
            compiled_round_plan=base["compiled_round_plan"],
            scene_decision_closure=base["scene_decision_closure"],
            realized_scenes=base["realized_scenes"],
            authoritative_memory=inputs["authoritative_memory"],
            planning_state=inputs["planning_state"], action_menu=inputs["action_menu"])
        self.assertEqual({row["candidate_update_key"]
                          for row in compiled["accepted_updates"]}, {"DU_ACCEPTED"})
        self.assertEqual([row["compiled_proposal"]["operation"]
                          for row in compiled["accepted_updates"]],
                         ["add_person", "add_interaction_link"])

    def test_non_occurred_event_remains_audit_only(self):
        inputs = person_event_inputs()
        inputs["planner_output"]["scenes"][1]["primary_action"]["arguments"][
            "occurrence_status"] = "scheduled"
        inputs["planner_output"]["scenes"][1]["primary_action"]["arguments"][
            "narrative_time_relation"] = "future"
        with tempfile.TemporaryDirectory() as temp:
            result = run_mock_hybrid_v11(
                **inputs, run_root=Path(temp), case="all_accept")
        operations = [row["compiled_proposal"]["operation"]
                      for row in result["durable_update_compile_result"]["accepted_updates"]]
        self.assertNotIn("add_event", operations)
        self.assertTrue(any(row["reason"] == "connected_event_not_occurred"
                            for row in result["durable_update_compile_result"][
                                "audit_only_observations"]))

    def test_validator_requires_typed_connection_claim_for_person_candidate(self):
        inputs = person_event_inputs()
        with tempfile.TemporaryDirectory() as temp:
            result = run_mock_hybrid_v11(
                **inputs, run_root=Path(temp), case="all_accept")
        person_candidate = result["round_validator_output"]["durable_updates"][0]
        self.assertEqual(person_candidate["verified_connection"]["connection_kind"],
                         "interaction_link")
        broken = deepcopy(result["round_validator_output"])
        broken["durable_updates"][0].pop("verified_connection")
        validation = validate_round_validator_output(
            broken, realized_scenes=result["realized_scenes"],
            review_world_view=result["review_world_view"],
            local_gate_results=result["local_scene_gate_results"])
        self.assertFalse(validation["valid"])
        self.assertTrue(any("durable_update:0:verified_connection_schema" in error
                            for error in validation["errors"]))

    def test_explicit_relation_requires_validator_relation_type(self):
        inputs = person_event_inputs()
        with tempfile.TemporaryDirectory() as temp:
            base = run_mock_hybrid_v11(
                **inputs, run_root=Path(temp), case="accept_no_updates")
        scene = base["compiled_round_plan"]["scenes"][0]
        evidence = base["realized_scenes"][0]["dialogue_turns"][0]["turn_ref"]
        candidate = {
            "candidate_update_key": "DU_REL", "source_scene_ref": scene["scene_ref"],
            "source_action_role": "primary", "evidence_turn_refs": [evidence],
            "verification_status": "confirmed", "atomic_group_key": None,
            "depends_on_candidate_keys": [],
            "verified_connection": {
                "verification_status": "confirmed", "connection_kind": "explicit_relation",
                "relation_type": "coworker", "evidence_turn_refs": [evidence],
                "reason": "the dialogue explicitly states that C is A's coworker",
            },
        }
        compiled = compile_durable_updates(
            candidates=[candidate], compiled_round_plan=base["compiled_round_plan"],
            scene_decision_closure=base["scene_decision_closure"],
            realized_scenes=base["realized_scenes"],
            authoritative_memory=inputs["authoritative_memory"],
            planning_state=inputs["planning_state"], action_menu=inputs["action_menu"])
        self.assertEqual([row["compiled_proposal"]["operation"]
                          for row in compiled["accepted_updates"]],
                         ["add_person", "add_relation"])

    def test_event_duplicate_is_rejected_against_authoritative_memory(self):
        inputs = person_event_inputs()
        with tempfile.TemporaryDirectory() as temp:
            result = run_mock_hybrid_v11(
                **inputs, run_root=Path(temp), case="all_accept")
        event_candidate = deepcopy(result["round_validator_output"]["durable_updates"][1])
        event_candidate["depends_on_candidate_keys"] = []
        planning_final = compile_planning_state_view(
            memory=result["memory_final"], interaction_links=())
        compiled = compile_durable_updates(
            candidates=[event_candidate],
            compiled_round_plan=result["compiled_round_plan"],
            scene_decision_closure=result["scene_decision_closure"],
            realized_scenes=result["realized_scenes"],
            authoritative_memory=result["memory_final"],
            planning_state=planning_final, action_menu=inputs["action_menu"])
        self.assertEqual(compiled["accepted_updates"], [])
        self.assertTrue(any("candidate_authoritative_operation_duplicate" in row["errors"]
                            for row in compiled["rejected_updates"]))

    def test_canary14_migrated_person_shadow_compiles_two_atomic_proposals(self):
        plan = json.loads((CANARY14 / "planner/compiled_round_plan.json").read_text())
        realized = json.loads((CANARY14 / "child_dialogue_full.json").read_text())["scenes"]
        closure = json.loads((CANARY14 / "scene_decision_closure.json").read_text())
        memory = json.loads((CANARY14 / "world_memory_initial.json").read_text())
        planning = json.loads((CANARY14 / "planning_state_view.json").read_text())
        menu = json.loads((CANARY14 / "internal_durable_affordance_menu.json").read_text())
        action = plan["scenes"][3]["primary_action"]
        old = action["resolved_arguments"]
        action["resolved_arguments"] = {
            "participant_alias": "co_worker_c",
            "display_name": "Co-worker C", "entity_role": "project designer",
            "connection_type": "existing_person",
            "connection_target_ref": old["connection_to_existing_world"]["existing_ref"],
            "connection_description": old["connection_to_existing_world"]["description"],
        }
        candidate = next(row for row in json.loads(
            (CANARY14 / "round_validator_output.json").read_text())["durable_updates"]
                         if row["candidate_update_key"] == "DU004")
        candidate["depends_on_candidate_keys"] = []
        candidate["verified_connection"] = {
            "verification_status": "confirmed", "connection_kind": "interaction_link",
            "relation_type": None, "evidence_turn_refs": list(candidate["evidence_turn_refs"]),
            "reason": "shadow fixture verifies only the explicit project connection",
        }
        compiled = compile_durable_updates(
            candidates=[candidate], compiled_round_plan=plan,
            scene_decision_closure=closure, realized_scenes=realized,
            authoritative_memory=memory, planning_state=planning, action_menu=menu)
        self.assertEqual([row["compiled_proposal"]["operation"]
                          for row in compiled["accepted_updates"]],
                         ["add_person", "add_interaction_link"])
        self.assertEqual(hash_tree(CANARY14)["tree_sha256"], CANARY14_SHA)

    def test_canary14_original_contract_stays_failed_and_read_only(self):
        plan = json.loads((CANARY14 / "planner/compiled_round_plan.json").read_text())
        action = plan["scenes"][3]["primary_action"]
        raw_arguments = action["resolved_arguments"]
        live_shape = {"action_mode": "new", "action_type": "introduce_connected_person",
                      "arguments": raw_arguments}
        validation = validate_action_instance(live_shape)
        self.assertFalse(validation["valid"])
        self.assertTrue(any("missing:display_name" in error
                            for error in validation["errors"]))
        self.assertEqual(hash_tree(CANARY14)["tree_sha256"], CANARY14_SHA)


if __name__ == "__main__":
    unittest.main()
