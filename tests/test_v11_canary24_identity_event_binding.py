import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.durable_update_compiler import (
    compile_durable_updates,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_canary_v11 import (
    _apply_durable_updates,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_expansion_validator_v1 import (
    compile_round_validator_model_output,
    validate_round_validator_output,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_validator_contract_repair_v11 import (
    apply_round_validator_contract_repair,
    build_round_validator_contract_repair_plan,
    validate_round_validator_contract_repair_output,
)


CANARY = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_continuous_expansion_"
    "v11_live_canary_24")
CANARY_SHA = "da856fba136a1dddcb6d5887ad82d5e69e35944775ab077ac3d1652288e12578"


def load(relative):
    return json.loads((CANARY / relative).read_text(encoding="utf-8"))


class Canary24IdentityEventBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = load("attempts/round_validator_attempt_01.json")[
            "parsed_response"]
        cls.run_audit = load("round_validator_run.json")
        cls.validator_input = load("round_validator_input.json")
        cls.review = load("review_world_view.json")
        cls.gates = load("local_scene_gate_results.json")
        cls.realized = [json.loads(path.read_text(encoding="utf-8"))
                        for path in sorted((CANARY / "scenes").glob(
                            "*/realized_scene.json"))]
        cls.compiled_plan = load("planner/compiled_round_plan.json")
        cls.closure = load("scene_decision_closure.json")
        cls.memory = load("world_memory_initial.json")
        cls.planning_state = load("planning_state_view.json")
        cls.action_menu = load("internal_durable_affordance_menu.json")

    def repair_plan(self):
        historical = {
            row["source_anchor_id"]: row["source_refs"]
            for row in self.validator_input["historical_anchor_index"]}
        return build_round_validator_contract_repair_plan(
            model_output=self.base, realized_scenes=self.realized,
            compilation_errors=self.run_audit["attempts"][0]["validation"]["errors"],
            historical_anchor_refs=historical)

    def safe_base_and_plan(self):
        base = deepcopy(self.base)
        base["durable_updates"][0]["verified_connection"] = {
            "verification_status": "confirmed",
            "connection_kind": "interaction_link",
            "relation_type": None,
            "evidence_turn_refs": ["R01:S01:T001"],
            "reason": (
                "The dialogue explicitly connects the introduced colleague "
                "to existing participants."
            ),
        }
        compilation = compile_round_validator_model_output(
            base, realized_scenes=self.realized
        )
        historical = {
            row["source_anchor_id"]: row["source_refs"]
            for row in self.validator_input["historical_anchor_index"]
        }
        plan = build_round_validator_contract_repair_plan(
            model_output=base,
            realized_scenes=self.realized,
            compilation_errors=compilation["errors"],
            historical_anchor_refs=historical,
        )
        return base, plan

    def repair_fixture(self, plan=None):
        plan = plan or self.repair_plan()
        rows = []
        for target in plan["targets"]:
            rows.append({
                "repair_kind": "action_realization",
                "action_ref": target["action_ref"],
                "evidence_turn_refs": target["original_evidence_turn_refs"],
                "progression_evidence_turn_refs":
                    target["candidate_evidence_turn_refs"][:1],
            })
        for target in plan["durable_update_targets"]:
            rows.append({
                "repair_kind": "durable_update_evidence",
                "candidate_update_key": target["candidate_update_key"],
                "evidence_turn_refs": target["allowed_evidence_turn_refs"][:1],
            })
        for target in plan["durable_removal_targets"]:
            rows.append({
                "repair_kind": "durable_update_nullification",
                "candidate_update_key": target["candidate_update_key"],
                "replacement": None,
            })
        for target in plan["verified_connection_targets"]:
            rows.append({
                "repair_kind": "verified_connection",
                "candidate_update_key": target["candidate_update_key"],
                "verified_connection": {
                    "verification_status": "confirmed",
                    "connection_kind": "interaction_link",
                    "relation_type": None,
                    "evidence_turn_refs": target[
                        "allowed_evidence_turn_refs"][:1],
                    "reason": ("The dialogue explicitly connects the new person "
                               "to an existing participant."),
                },
            })
        for target in plan["observation_targets"]:
            scene_ref = target["allowed_scene_refs"][0]
            rows.append({
                "repair_kind": "observation_reference",
                "observation_key": target["observation_key"],
                "source_scene_refs": [scene_ref],
                "evidence_turn_refs": target[
                    "allowed_evidence_turn_refs_by_scene"][scene_ref][:1],
            })
        return {"repairs": rows}

    def reconstructed_validator_output(self):
        base, plan = self.safe_base_and_plan()
        fixture = self.repair_fixture(plan)
        validation = validate_round_validator_contract_repair_output(
            fixture, repair_plan=plan)
        self.assertTrue(validation["valid"], validation["errors"])
        applied = apply_round_validator_contract_repair(
            base_model_output=base, repair_output=fixture,
            repair_plan=plan)
        self.assertTrue(applied["valid"], applied["errors"])
        compiled = compile_round_validator_model_output(
            applied["reconstructed_output"], realized_scenes=self.realized)
        self.assertTrue(compiled["valid"], compiled["errors"])
        strict = validate_round_validator_output(
            compiled["compiled_output"], realized_scenes=self.realized,
            review_world_view=self.review, local_gate_results=self.gates)
        self.assertTrue(strict["valid"], strict["errors"])
        return compiled["compiled_output"]

    def compile_shadow(self, candidates=None):
        output = self.reconstructed_validator_output()
        return compile_durable_updates(
            candidates=(output["durable_updates"] if candidates is None
                        else candidates),
            compiled_round_plan=self.compiled_plan,
            scene_decision_closure=self.closure,
            realized_scenes=self.realized,
            authoritative_memory=self.memory,
            planning_state=self.planning_state,
            action_menu=self.action_menu)

    def test_historical_missing_connection_removes_dependent_candidate_chain(self):
        plan = self.repair_plan()
        self.assertTrue(plan["valid"], plan["errors"])
        self.assertEqual(plan["target_candidate_update_keys"], [])
        self.assertEqual(plan["target_removed_update_keys"],
                         ["DU001", "DU002", "DU003", "DU004", "DU005", "DU006"])
        reasons = {
            row["candidate_update_key"]: row["reason"]
            for row in plan["durable_removal_targets"]
        }
        self.assertEqual(
            reasons["DU001"], "verification_status_invalid_or_missing"
        )
        self.assertEqual(reasons["DU002"], "dependency_candidate_removed")
        self.assertTrue(all(
            reasons[key] == "identity_introduction_continuation_not_new_identity"
            for key in ("DU003", "DU004", "DU005", "DU006")
        ))
        self.assertEqual(plan["verified_connection_targets"], [])

    def test_explicit_verified_connection_fixture_preserves_new_roots(self):
        _, plan = self.safe_base_and_plan()
        self.assertTrue(plan["valid"], plan["errors"])
        self.assertEqual(plan["target_candidate_update_keys"], ["DU001", "DU002"])
        self.assertEqual(
            plan["target_removed_update_keys"],
            ["DU003", "DU004", "DU005", "DU006"],
        )

    def test_person_connection_and_dependent_event_compile_once(self):
        compiled = self.compile_shadow()
        self.assertEqual(compiled["rejected_updates"], [])
        self.assertEqual([
            row["compiled_proposal"]["operation"]
            for row in compiled["accepted_updates"]], [
                "add_person", "add_interaction_link", "add_event"])
        person = compiled["accepted_updates"][0]["compiled_proposal"]
        event = compiled["accepted_updates"][2]["compiled_proposal"]
        self.assertIn("R01:P01", person["temporary_refs"])
        internal_ref = person["provenance"]["internal_temporary_ref"]
        self.assertEqual(event["compiled_payload"]["participant_refs"],
                         [internal_ref])
        self.assertIn(internal_ref, event["temporary_refs"])
        self.assertEqual(event["proposal_dependencies"], [
            "R01:S01:PDU01", "R01:S01:PDU02"])

    def test_round_participant_requires_explicit_person_candidate_dependency(self):
        output = self.reconstructed_validator_output()
        candidates = deepcopy(output["durable_updates"])
        candidates[1]["depends_on_candidate_keys"] = []
        compiled = self.compile_shadow(candidates)
        event_rejections = [row for row in compiled["rejected_updates"]
                            if row["candidate"].get(
                                "candidate_update_key") == "DU002"]
        self.assertEqual(len(event_rejections), 1)
        self.assertIn("connected_event_round_participant_dependency_missing",
                      event_rejections[0]["errors"])

    def test_transactions_bind_later_event_to_allocated_entity(self):
        compiled = self.compile_shadow()
        generated = {str(row["scene_ref"]): row for row in self.realized}
        with tempfile.TemporaryDirectory() as temp:
            final, transactions = _apply_durable_updates(
                run_root=Path(temp), memory=self.memory,
                compiled_plan=self.compiled_plan, generated=generated,
                accepted_updates=compiled["accepted_updates"])
            self.assertEqual([row["status"] for row in transactions],
                             ["COMMITTED", "COMMITTED"])
            person_record = next(row for row in final["authoritative_records"]
                                 if row.get("operation") == "add_person")
            event_record = next(row for row in final["authoritative_records"]
                                if row.get("operation") == "add_event")
            allocated = person_record["allocated_entity_ref"]
            self.assertTrue(allocated.startswith(self.memory["root_id"] + ":E"))
            self.assertEqual(event_record["payload"]["participant_refs"],
                             [allocated])
            self.assertNotIn("TMP", json.dumps(event_record))

    def test_history_unchanged(self):
        self.assertEqual(hash_tree(CANARY)["tree_sha256"], CANARY_SHA)


if __name__ == "__main__":
    unittest.main()
