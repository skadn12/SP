from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.append_only_memory import (
    apply_scene_commit, evaluate_scene_commit, initialize_append_only_memory,
    normalize_state_dimension,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.claim_validator_v2 import (
    realizer_retry_route, validate_proposal_claims,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.controlled_expansion import (
    compile_action_menu_v3, validate_action_menu_v3,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.cross_round_consumers import (
    load_lineage_surface, load_report_surface, load_resume_surface, load_round_parent,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.dialogue_quality import (
    validate_dialogue_quality,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.evidence_scope import (
    committer_retry_disposition, validate_committer_scope_output,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.interaction_links import (
    compile_generated_interaction_links, compile_source_interaction_links,
    validate_scene_connectivity,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.operation_registry import (
    OPERATION_REGISTRY, validate_operation_registry,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.planner_v3 import (
    PLANNER_SCHEMA_VERSION_V3, compile_scene_plan_v3, proposal_fingerprint,
    validate_planner_v3,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.promotion import (
    assemble_child_surfaces, load_promotable_child,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance_ledger import (
    initialize_provenance_ledger, ledger_entry,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.realizer_v3 import (
    validate_claim_only_realizer_shape, validate_full_realizer_shape,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.relation_registry import (
    derive_relation_edges, validate_relation_registry,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.transactional_binder import (
    validate_scene_local_evidence_refs, build_allocation_plan, commit_scene_transaction,
    recover_prepared_transactions,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.turn_budget import (
    compile_turn_budget,
)


ROOT = "fixture-root"
E1 = f"{ROOT}:E000001"
E2 = f"{ROOT}:E000002"


def fixture_memory() -> dict:
    entities = [{"entity_id": E1, "display_name": "A", "semantic_status": "source_grounded"},
                {"entity_id": E2, "display_name": "B", "semantic_status": "source_grounded"}]
    return {"root_id": ROOT, "entities": entities, "event_history": [], "current_states": [],
            "open_threads": [], "relations": []}


def fixture_plan(action: dict) -> dict:
    operation = action["operation"]
    content = {
        "add_event": {"description": "A concrete new development."},
        "add_person": {"display_role": "a person connected to the situation", "scene_presence": "mentioned_only"},
        "add_relation": {"source_ref": E1, "target_ref": E2, "relation_type": "knows"},
        "create_state": {"state_category": "attitude", "state_dimension": "Acceptance of Support",
                         "initial_value": "hesitant"},
        "add_shared_task": {"task_description": "Plan the next step."},
        "open_thread": {"description": "Revisit the practical question."},
        "create_conflict": {"conflict_description": "They disagree on timing."},
        "create_misunderstanding": {"misunderstanding": "They read the message differently."},
        "introduce_external_constraint": {"constraint": "The deadline moves earlier."},
        "add_information_asymmetry": {"information": "A knows the schedule changed."},
        "add_third_party_report": {"reported_content": "A third party reports a change."},
    }[operation]
    content = {key: value for key, value in content.items()
               if key not in (action.get("frozen_payload") or {})}
    speakers = [{"participant_ref": E1, "scene_role": "primary", "intents": ["raise issue", "seek step"]},
                {"participant_ref": E2, "scene_role": "supporting", "intents": ["offer option"]}]
    return {"planner_schema_version": PLANNER_SCHEMA_VERSION_V3,
            "scene_budget": {"minimum": 2, "target": 3, "maximum": 4},
            "round_arc": {"starting_situation": "start", "central_tension": "tension",
                          "ending_direction": "direction"},
            "scene_goals": [
                {"goal": "advance", "speaker_participants": speakers,
                 "interaction_beats": ["one", "two", "three"],
                 "continuity_requirements": ["respect memory"],
                 "proposal_selection": {
                     "selection_type": "single_action",
                     "action_id": action["action_id"],
                     "proposed_content": content,
                 },
                 "desired_outcome": "one concrete extension"},
                {"goal": "close", "speaker_participants": speakers,
                 "interaction_beats": ["one", "two", "three"],
                 "continuity_requirements": ["respect memory"],
                 "proposal_selection": {"selection_type": "none"},
                 "desired_outcome": "natural close"}]}


def compiled_fixture():
    menu = compile_action_menu_v3(fixture_memory(), root_id=ROOT)
    action = next(row for row in menu["actions"] if row["operation"] == "add_event")
    plan = fixture_plan(action)
    compiled = compile_scene_plan_v3(plan, action_menu=menu, round_index=1)
    return menu, plan, compiled["scenes"][0]


def dialogue_for(scene: dict) -> list[dict]:
    budget = compile_turn_budget(scene)
    speakers = [row["participant_ref"] for row in scene["speaker_participants"]]
    return [{"turn_ref": f"{scene['scene_ref']}:T{i:03d}",
             "speaker_ref": speakers[(i - 1) % 2], "text": f"New fixture utterance {i}."}
            for i in range(1, budget["target"] + 1)]


class RegistriesAndExpansionTests(unittest.TestCase):
    def test_operation_and_relation_registries(self):
        self.assertTrue(validate_operation_registry()["valid"])
        self.assertEqual(23, len(OPERATION_REGISTRY))
        self.assertTrue(validate_relation_registry()["valid"])

    def test_inverse_and_symmetric_relation_is_local(self):
        edges = derive_relation_edges("R01:S01:P01", {"source_ref": E1, "target_ref": E2,
                                                       "relation_type": "manager_of"})
        self.assertEqual("reports_to", edges[1]["relation_type"])
        symmetric = derive_relation_edges("R01:S01:P01", {"source_ref": E1, "target_ref": E2,
                                                           "relation_type": "sibling"})
        self.assertEqual("sibling", symmetric[1]["relation_type"])

    def test_action_budget_and_no_fake_create_fill(self):
        menu = compile_action_menu_v3(fixture_memory(), root_id=ROOT)
        self.assertTrue(validate_action_menu_v3(menu)["valid"])
        self.assertLessEqual(len(menu["actions"]), 16)
        self.assertLessEqual(max(__import__("collections").Counter(
            row["operation"] for row in menu["actions"]).values()), 2)

    def test_no_trust_or_authority_action_without_state(self):
        menu = compile_action_menu_v3(fixture_memory(), root_id=ROOT)
        operations = {row["operation"] for row in menu["actions"]}
        self.assertNotIn("change_trust", operations)
        self.assertNotIn("change_authority", operations)


class ProvenanceTests(unittest.TestCase):
    def test_cross_round_generated_origin_preserved(self):
        prior = ledger_entry(record_ref="M1", root_source_id=ROOT,
                             origin="generated_extension", verification_status="semantically_verified",
                             introduced_round=1, introduced_scene_ref="R01:S01", proposal_ref="R01:S01:P01",
                             support_turn_refs=["T1"], authoritative_memory_ref="M1", parent_record_refs=[])
        result = initialize_provenance_ledger(
            root_source_id=ROOT, original_source_records=[],
            previous_authoritative_memory=[{"memory_ref": "M1"}],
            previous_provenance_ledger=[prior],
            promoted_child_dialogue={"scenes": [{"text": "fact"}]},
            promotion_manifest={"committed_memory_refs": ["M1"]})
        self.assertEqual("generated_extension", result["authoritative_memory"][0]["origin"])
        self.assertEqual(["M1"], result["promoted_linked_memory_refs"])

    def test_rejected_and_uncertain_not_whitened(self):
        for status in ("rejected", "uncertain", "dependency_blocked", "atomic_blocked", "scene_blocked"):
            row = ledger_entry(record_ref=status, root_source_id=ROOT,
                               origin="generated_extension", verification_status=status,
                               introduced_round=1, introduced_scene_ref="R01:S01", proposal_ref="P1",
                               support_turn_refs=[], authoritative_memory_ref=None, parent_record_refs=[])
            self.assertNotEqual("source_grounded", row["verification_status"])
        with self.assertRaises(ValueError):
            ledger_entry(record_ref="bad", root_source_id=ROOT, origin="generated_extension",
                         verification_status="source_grounded", introduced_round=1,
                         introduced_scene_ref=None, proposal_ref=None, support_turn_refs=[],
                         authoritative_memory_ref=None, parent_record_refs=[])

    def test_promoted_text_does_not_create_fact(self):
        result = initialize_provenance_ledger(
            root_source_id=ROOT, original_source_records=[],
            promoted_child_dialogue={"text": "brand new unsupported fact"},
            promotion_manifest={"committed_memory_refs": []})
        self.assertEqual([], result["provenance_ledger"])


class InteractionTests(unittest.TestCase):
    def test_source_links_are_adjacent_not_clique_and_stable(self):
        sessions = [{"session_id": "S1", "turns": [
            {"turn_id": "T1", "speaker": "A"}, {"turn_id": "T2", "speaker": "B"},
            {"turn_id": "T3", "speaker": "C"}]}]
        mapping = {"A": f"{ROOT}:E1", "B": f"{ROOT}:E2", "C": f"{ROOT}:E3"}
        links = compile_source_interaction_links(root_id=ROOT, sessions=sessions,
                                                 speaker_to_entity=mapping)
        self.assertEqual(2, len(links))
        self.assertNotIn([f"{ROOT}:E1", f"{ROOT}:E3"], [row["entity_refs"] for row in links])
        reverse = compile_source_interaction_links(root_id=ROOT, sessions=sessions,
                                                   speaker_to_entity=mapping)
        self.assertEqual(links, reverse)

    def test_generated_link_requires_commit_and_promotion(self):
        self.assertEqual([], compile_generated_interaction_links(
            root_id=ROOT, scene_ref="R01:S01", speaker_refs=[E1, E2], committed=True,
            promoted=False, authoritative_entity_refs={E1, E2}))
        links = compile_generated_interaction_links(
            root_id=ROOT, scene_ref="R01:S01", speaker_refs=[E1, E2], committed=True,
            promoted=True, authoritative_entity_refs={E1, E2})
        self.assertEqual("generated_extension", links[0]["origin"])

    def test_connectivity_and_bridge(self):
        failed = validate_scene_connectivity(primary_refs=[E1, E2], supporting_refs=[])
        self.assertFalse(failed["valid"])
        bridged = validate_scene_connectivity(primary_refs=[E1, E2], supporting_refs=[],
                                               allows_connectivity_bridge=True)
        self.assertTrue(bridged["valid"])


class PlannerAndProposalTests(unittest.TestCase):
    def test_planner_v3_and_local_refs(self):
        menu, plan, scene = compiled_fixture()
        result = validate_planner_v3(plan, action_menu=menu,
                                     authoritative_entity_refs={E1, E2})
        self.assertTrue(result["valid"], result)
        self.assertEqual("R01:S01", scene["scene_ref"])
        self.assertEqual("R01:S01:P01", scene["compiled_proposals"][0]["proposal_ref"])
        self.assertEqual("add_event", scene["compiled_proposals"][0]["operation"])

    def test_planner_program_owned_field_rejected(self):
        menu, plan, _ = compiled_fixture()
        plan = deepcopy(plan)
        plan["scene_goals"][0]["scene_ref"] = "model-created"
        result = validate_planner_v3(plan, action_menu=menu,
                                     authoritative_entity_refs={E1, E2})
        self.assertFalse(result["valid"])
        self.assertTrue(any("program_owned" in error for error in result["errors"]))

    def test_temporary_entity_cannot_be_speaker(self):
        menu, plan, _ = compiled_fixture()
        plan = deepcopy(plan)
        plan["scene_goals"][0]["speaker_participants"][0]["participant_ref"] = "temp:person"
        self.assertFalse(validate_planner_v3(
            plan, action_menu=menu, authoritative_entity_refs={E1, E2})["valid"])

    def test_three_proposals_rejected(self):
        menu, plan, _ = compiled_fixture()
        plan = deepcopy(plan)
        plan["scene_goals"][0]["proposal_selection"]["proposal_contents"] = {
            "A0001": {}, "A0002": {}, "A0003": {}}
        self.assertFalse(validate_planner_v3(
            plan, action_menu=menu, authoritative_entity_refs={E1, E2})["valid"])

    def test_fingerprint_nfkc_and_duplicate_freeze(self):
        left = proposal_fingerprint("add_event", [E1], {"description": "Ａ  test"})
        right = proposal_fingerprint("add_event", [E1], {"description": "A  test"})
        self.assertEqual(left, right)
        menu, plan, _ = compiled_fixture()
        plan = deepcopy(plan)
        plan["scene_goals"][1]["proposal_selection"] = deepcopy(
            plan["scene_goals"][0]["proposal_selection"])
        with self.assertRaises(ValueError):
            compile_scene_plan_v3(plan, action_menu=menu, round_index=1)

    def test_turn_budget_always_ordered(self):
        _, _, scene = compiled_fixture()
        budget = compile_turn_budget(scene)
        self.assertLessEqual(budget["minimum"], budget["target"])
        self.assertLessEqual(budget["target"], budget["maximum"])

    def test_person_identity_relation_is_atomic_dependency_bundle(self):
        menu = compile_action_menu_v3(fixture_memory(), root_id=ROOT)
        person = next(row for row in menu["actions"] if row["operation"] == "add_person")
        relation = next(row for row in menu["actions"] if row["operation"] == "add_relation")
        plan = fixture_plan(person)
        plan["scene_goals"][0]["proposal_selection"] = {
            "selection_type": "bundle",
            "bundle_id": "B0001",
            "proposal_contents": {
                person["action_id"]: {
                    "display_role": "a person involved in the situation"},
                relation["action_id"]: {"relation_type": "sibling"},
            },
        }
        validation = validate_planner_v3(plan, action_menu=menu,
                                         authoritative_entity_refs={E1, E2})
        self.assertTrue(validation["valid"], validation)
        scene = compile_scene_plan_v3(plan, action_menu=menu, round_index=1)["scenes"][0]
        self.assertEqual(["R01:S01:P01"], scene["compiled_proposals"][1]["proposal_dependencies"])
        self.assertEqual(scene["compiled_proposals"][0]["atomic_group_ref"],
                         scene["compiled_proposals"][1]["atomic_group_ref"])
        allocation = build_allocation_plan(
            root_id=ROOT, round_index=1, scene_index=1,
            compiled_proposals=scene["compiled_proposals"], high_water_marks={})
        local_temp = scene["compiled_proposals"][0]["temporary_refs"][0]
        self.assertEqual(allocation["proposal_allocations"]["R01:S01:P01"]["entity_ref"],
                         allocation["temporary_ref_allocations"][local_temp])
        self.assertEqual(local_temp,
                         scene["compiled_proposals"][1]["compiled_payload"]["target_ref"])

    def test_add_person_role_does_not_implicitly_create_relation(self):
        menu = compile_action_menu_v3(fixture_memory(), root_id=ROOT)
        person = next(row for row in menu["actions"] if row["operation"] == "add_person")
        relation = next(row for row in menu["actions"] if row["operation"] == "add_relation")
        plan = fixture_plan(person)
        plan["scene_goals"][0]["proposal_selection"] = {
            "selection_type": "bundle",
            "bundle_id": "B0001",
            "proposal_contents": {
                person["action_id"]: {"display_role": "A's sister"},
                relation["action_id"]: {"relation_type": "sibling"},
            },
        }
        with self.assertRaises(ValueError):
            compile_scene_plan_v3(plan, action_menu=menu, round_index=1)


class RealizerQualityClaimCommitterTests(unittest.TestCase):
    def setUp(self):
        _, _, self.scene = compiled_fixture()
        self.dialogue = dialogue_for(self.scene)
        self.proposal_ref = self.scene["compiled_proposals"][0]["proposal_ref"]

    def test_full_and_claim_only_schema_are_disjoint(self):
        full = {"dialogue_turns": self.dialogue, "proposal_claims": []}
        self.assertTrue(validate_full_realizer_shape(full)["valid"])
        self.assertFalse(validate_claim_only_realizer_shape(full)["valid"])
        self.assertTrue(validate_claim_only_realizer_shape({"proposal_claims": []})["valid"])
        self.assertFalse(validate_full_realizer_shape({"dialogue_turns": self.dialogue,
                                                       "proposal_claims": [],
                                                       "new_open_thread_claims": []})["valid"])

    def test_quality_gate_and_retry_routing(self):
        quality = validate_dialogue_quality(
            scene_ref=self.scene["scene_ref"], dialogue_turns=self.dialogue,
            speaker_participants=self.scene["speaker_participants"],
            turn_budget=compile_turn_budget(self.scene))
        self.assertTrue(quality["valid"], quality)
        bad = deepcopy(self.dialogue)
        bad[1]["text"] = bad[0]["text"]
        failed = validate_dialogue_quality(
            scene_ref=self.scene["scene_ref"], dialogue_turns=bad,
            speaker_participants=self.scene["speaker_participants"],
            turn_budget=compile_turn_budget(self.scene))
        self.assertEqual("full_scene_retry", realizer_retry_route(
            json_parsed=True, dialogue_quality=failed, claim_validation=None,
            requests_used=1))

    def test_one_proposal_exactly_one_claim_and_subset(self):
        claims = [{"proposal_ref": self.proposal_ref, "realization_status": "realized",
                   "candidate_evidence_turn_refs": [self.dialogue[0]["turn_ref"]]}]
        validation = validate_proposal_claims(
            scene_ref=self.scene["scene_ref"], proposal_claims=claims,
            compiled_proposals=self.scene["compiled_proposals"], dialogue_turns=self.dialogue)
        self.assertTrue(validation["valid"], validation)
        duplicate = claims + claims
        failed = validate_proposal_claims(
            scene_ref=self.scene["scene_ref"], proposal_claims=duplicate,
            compiled_proposals=self.scene["compiled_proposals"], dialogue_turns=self.dialogue)
        self.assertFalse(failed["valid"])
        output = {"proposal_reviews": [{"proposal_ref": self.proposal_ref,
                  "review_decision": "confirmed", "verification_status": "semantically_verified",
                  "accepted_evidence_turn_refs": [self.dialogue[1]["turn_ref"]],
                  "reason_code": "supported"}], "scope_status": "clean",
                  "world_consistency_status": "clean"}
        result = validate_committer_scope_output(output,
            compiled_proposals=self.scene["compiled_proposals"], proposal_claims=claims)
        self.assertFalse(result["valid"])
        self.assertTrue(any("not_candidate_subset" in value for value in result["errors"]))

    def test_claim_failure_claim_only_and_uncertain_full_retry(self):
        quality = {"valid": True}
        bad_claim = {"valid": False}
        self.assertEqual("claim_only_retry", realizer_retry_route(
            json_parsed=True, dialogue_quality=quality, claim_validation=bad_claim,
            requests_used=1))
        uncertain = [{"proposal_ref": self.proposal_ref,
                      "realization_status": "realization_uncertain",
                      "candidate_evidence_turn_refs": []}]
        self.assertEqual("full_scene_retry", realizer_retry_route(
            json_parsed=True, dialogue_quality=quality, claim_validation={"valid": True},
            claims=uncertain, requests_used=2))
        self.assertEqual("scene_failed", realizer_retry_route(
            json_parsed=True, dialogue_quality=quality, claim_validation={"valid": True},
            claims=uncertain, requests_used=3))

    def test_valid_rejected_is_final_not_retry(self):
        claims = [{"proposal_ref": self.proposal_ref, "realization_status": "realized",
                   "candidate_evidence_turn_refs": [self.dialogue[0]["turn_ref"]]}]
        output = {"proposal_reviews": [{"proposal_ref": self.proposal_ref,
                  "review_decision": "rejected", "verification_status": "rejected",
                  "accepted_evidence_turn_refs": [], "reason_code": "unsupported"}],
                  "scope_status": "clean", "world_consistency_status": "clean"}
        validation = validate_committer_scope_output(
            output, compiled_proposals=self.scene["compiled_proposals"], proposal_claims=claims)
        self.assertTrue(validation["valid"])
        self.assertEqual("final_nonconfirmed", committer_retry_disposition(validation, output))

    def test_proposal_free_scene_requires_empty_claims_and_scope_only(self):
        no_proposals = []
        valid = validate_proposal_claims(
            scene_ref=self.scene["scene_ref"], proposal_claims=[],
            compiled_proposals=no_proposals, dialogue_turns=self.dialogue)
        self.assertTrue(valid["valid"])
        invalid = validate_proposal_claims(
            scene_ref=self.scene["scene_ref"], proposal_claims=[{
                "proposal_ref": "unknown", "realization_status": "realized",
                "candidate_evidence_turn_refs": [self.dialogue[0]["turn_ref"]]}],
            compiled_proposals=no_proposals, dialogue_turns=self.dialogue)
        self.assertFalse(invalid["valid"])
        scope_only = {"proposal_reviews": [], "scope_status": "clean",
                      "world_consistency_status": "clean"}
        self.assertTrue(validate_committer_scope_output(
            scope_only, compiled_proposals=[], proposal_claims=[])["valid"])

    def test_quality_exact_copy_is_deterministic(self):
        copied = deepcopy(self.dialogue)
        copied[0]["text"] = '“Copied source”'
        result = validate_dialogue_quality(
            scene_ref=self.scene["scene_ref"], dialogue_turns=copied,
            speaker_participants=self.scene["speaker_participants"],
            turn_budget=compile_turn_budget(self.scene),
            parent_utterances=['"Copied source"'])
        self.assertFalse(result["valid"])
        self.assertIn("parent_exact_copy", result["errors"])


class AppendOnlyTransactionPromotionTests(unittest.TestCase):
    def _commit_fixture(self):
        _, _, scene = compiled_fixture()
        dialogue = dialogue_for(scene)
        pref = scene["compiled_proposals"][0]["proposal_ref"]
        claims = [{"proposal_ref": pref, "realization_status": "realized",
                   "candidate_evidence_turn_refs": [dialogue[0]["turn_ref"]]}]
        committer = {"proposal_reviews": [{"proposal_ref": pref,
                    "review_decision": "confirmed", "verification_status": "semantically_verified",
                    "accepted_evidence_turn_refs": [dialogue[0]["turn_ref"]],
                    "reason_code": "supported"}], "scope_status": "clean",
                    "world_consistency_status": "clean"}
        memory = initialize_append_only_memory(root_id=ROOT,
            entities=fixture_memory()["entities"])
        allocation = build_allocation_plan(root_id=ROOT, round_index=1, scene_index=1,
            compiled_proposals=scene["compiled_proposals"], high_water_marks=memory["high_water_marks"])
        binding = validate_scene_local_evidence_refs(scene_ref=scene["scene_ref"],
            dialogue_turns=dialogue, accepted_evidence_by_proposal={pref: [dialogue[0]["turn_ref"]]})
        return scene, dialogue, claims, committer, memory, allocation, binding

    def test_scene_all_or_nothing(self):
        scene, _, claims, committer, _, _, _ = self._commit_fixture()
        blocked = deepcopy(committer)
        blocked["scope_status"] = "extraneous_memory_claim_detected"
        gate = evaluate_scene_commit(compiled_proposals=scene["compiled_proposals"],
                                     proposal_claims=claims, committer_output=blocked)
        self.assertFalse(gate["commit_ready"])
        self.assertEqual("scene_blocked", next(iter(gate["proposal_commit_statuses"].values())))

    def test_append_only_commit_and_state_normalization(self):
        scene, _, claims, committer, memory, allocation, binding = self._commit_fixture()
        before = deepcopy(memory)
        after = apply_scene_commit(memory, scene_ref=scene["scene_ref"], round_index=1,
            scene_index=1, compiled_proposals=scene["compiled_proposals"],
            proposal_claims=claims, committer_output=committer, allocation_plan=allocation,
            evidence_final_turn_refs=binding["evidence_final_turn_refs"],
            evidence_text_sha256=binding["evidence_text_sha256"])
        self.assertEqual(before, memory)
        self.assertEqual(1, len(after["authoritative_records"]))
        self.assertEqual("generated_extension", after["authoritative_records"][0]["origin"])
        self.assertEqual("acceptance_of_support", normalize_state_dimension(" Acceptance-of-Support "))

    def test_rejected_does_not_write_authoritative(self):
        scene, _, claims, committer, memory, allocation, binding = self._commit_fixture()
        committer["proposal_reviews"][0].update(
            review_decision="rejected", verification_status="rejected",
            accepted_evidence_turn_refs=[])
        after = apply_scene_commit(memory, scene_ref=scene["scene_ref"], round_index=1,
            scene_index=1, compiled_proposals=scene["compiled_proposals"],
            proposal_claims=claims, committer_output=committer, allocation_plan=allocation,
            evidence_final_turn_refs=binding["evidence_final_turn_refs"],
            evidence_text_sha256=binding["evidence_text_sha256"])
        self.assertEqual([], after["authoritative_records"])
        self.assertEqual(1, len(after["pending_updates"]))

    def test_transaction_idempotent_and_recoverable(self):
        scene, _, _, _, _, allocation, _ = self._commit_fixture()
        fingerprints = [row["proposal_fingerprint"] for row in scene["compiled_proposals"]]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = commit_scene_transaction(run_root=root, run_id="run", scene_plan=scene,
                proposal_fingerprints=fingerprints, allocation_plan=allocation,
                artifacts={"memory.json": {"records": [1]}})
            second = commit_scene_transaction(run_root=root, run_id="run", scene_plan=scene,
                proposal_fingerprints=fingerprints, allocation_plan=allocation,
                artifacts={"memory.json": {"records": [2]}})
            self.assertEqual("COMMITTED", first["status"])
            self.assertTrue(second["idempotent_resume"])
            self.assertEqual({"records": [1]}, json.loads((root / "memory.json").read_text()))
            self.assertEqual([], recover_prepared_transactions(root))

    def test_full_and_promotable_split(self):
        scene, dialogue, claims, committer, _, allocation, _ = self._commit_fixture()
        pref = scene["compiled_proposals"][0]["proposal_ref"]
        committed_ref = allocation["proposal_allocations"][pref]["memory_ref"]
        accepted = {"scene_ref": scene["scene_ref"], "quality_valid": True,
                    "claims_valid": True, "committer_contract_valid": True,
                    "compiled_proposals": scene["compiled_proposals"],
                    "committer_output": committer,
                    "scene_commit_result": {"commit_ready": True},
                    "bound_scene": {"scene_ref": scene["scene_ref"], "dialogue_turns": dialogue},
                    "committed_memory_refs": [committed_ref]}
        rejected = deepcopy(accepted)
        rejected["scene_ref"] = "R01:S02"
        rejected["bound_scene"]["scene_ref"] = "R01:S02"
        rejected["scene_commit_result"] = {"commit_ready": False}
        surfaces = assemble_child_surfaces(child_id="child", root_id=ROOT,
                                           scene_results=[accepted, rejected])
        self.assertEqual(2, len(surfaces["child_dialogue_full.json"]["scenes"]))
        self.assertEqual(1, len(surfaces["child_dialogue_promotable.json"]["scenes"]))
        self.assertEqual(surfaces["child_dialogue.json"],
                         surfaces["child_dialogue_promotable.json"])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "child_dialogue_promotable.json"
            path.write_text(json.dumps(surfaces["child_dialogue_promotable.json"]), encoding="utf-8")
            self.assertEqual("promotable", load_promotable_child(Path(temp))["surface"])
            for consumer in (load_round_parent, load_lineage_surface,
                             load_resume_surface, load_report_surface):
                self.assertEqual("promotable", consumer(Path(temp))["surface"])

    def test_update_appends_transition_and_preserves_old_record(self):
        base = {"memory_ref": f"{ROOT}:M000001", "logical_key": f"state:{E1}:attitude:confidence",
                "record_type": "state", "value": "low", "subject_refs": [E1],
                "authoritative_version": 1, "origin": "source_grounded",
                "verification_status": "source_grounded"}
        memory = initialize_append_only_memory(
            root_id=ROOT, entities=fixture_memory()["entities"],
            authoritative_records=[base])
        proposal = {"proposal_ref": "R02:S01:P01", "operation": "update_state",
                    "compiled_payload": {"target_memory_ref": base["memory_ref"],
                                         "new_value": "steady"},
                    "connected_memory_refs": [base["memory_ref"]],
                    "target_memory_ref": base["memory_ref"], "pre_state": "low",
                    "authoritative_version": 1, "proposal_fingerprint": "f" * 64,
                    "proposal_dependencies": [], "atomic_group_ref": None}
        claim = {"proposal_ref": proposal["proposal_ref"], "realization_status": "realized",
                 "candidate_evidence_turn_refs": ["R02:S01:T001"]}
        committer = {"proposal_reviews": [{"proposal_ref": proposal["proposal_ref"],
                    "review_decision": "confirmed", "verification_status": "semantically_verified",
                    "accepted_evidence_turn_refs": ["R02:S01:T001"], "reason_code": "supported"}],
                    "scope_status": "clean", "world_consistency_status": "authorized_change"}
        allocation = build_allocation_plan(root_id=ROOT, round_index=2, scene_index=1,
            compiled_proposals=[proposal], high_water_marks=memory["high_water_marks"])
        after = apply_scene_commit(memory, scene_ref="R02:S01", round_index=2, scene_index=1,
            compiled_proposals=[proposal], proposal_claims=[claim], committer_output=committer,
            allocation_plan=allocation,
            evidence_final_turn_refs={proposal["proposal_ref"]: ["child:turn:1"]},
            evidence_text_sha256={proposal["proposal_ref"]: ["a" * 64]})
        self.assertEqual(base["memory_ref"], after["authoritative_records"][0]["memory_ref"])
        self.assertEqual(2, len(after["authoritative_records"]))
        self.assertEqual(base["memory_ref"], after["memory_transitions"][0]["from_memory_ref"])
        self.assertEqual(2, after["authoritative_records"][1]["authoritative_version"])

    def test_stale_update_rejected(self):
        base = {"memory_ref": f"{ROOT}:M000001", "logical_key": "state:key",
                "record_type": "state", "value": "low", "authoritative_version": 1,
                "origin": "source_grounded", "verification_status": "source_grounded"}
        memory = initialize_append_only_memory(root_id=ROOT, entities=[], authoritative_records=[base])
        proposal = {"proposal_ref": "R02:S01:P01", "operation": "update_state",
                    "compiled_payload": {"new_value": "high"}, "connected_memory_refs": [],
                    "target_memory_ref": base["memory_ref"], "pre_state": "wrong",
                    "authoritative_version": 1, "proposal_fingerprint": "b" * 64,
                    "proposal_dependencies": [], "atomic_group_ref": None}
        claim = {"proposal_ref": proposal["proposal_ref"], "realization_status": "realized",
                 "candidate_evidence_turn_refs": ["R02:S01:T001"]}
        committer = {"proposal_reviews": [{"proposal_ref": proposal["proposal_ref"],
                    "review_decision": "confirmed", "verification_status": "semantically_verified",
                    "accepted_evidence_turn_refs": ["R02:S01:T001"], "reason_code": "supported"}],
                    "scope_status": "clean", "world_consistency_status": "authorized_change"}
        allocation = build_allocation_plan(root_id=ROOT, round_index=2, scene_index=1,
            compiled_proposals=[proposal], high_water_marks=memory["high_water_marks"])
        with self.assertRaises(ValueError):
            apply_scene_commit(memory, scene_ref="R02:S01", round_index=2, scene_index=1,
                compiled_proposals=[proposal], proposal_claims=[claim], committer_output=committer,
                allocation_plan=allocation,
                evidence_final_turn_refs={proposal["proposal_ref"]: ["turn"]},
                evidence_text_sha256={proposal["proposal_ref"]: ["a" * 64]})


if __name__ == "__main__":
    unittest.main()
