"""Zero-API tests for the Realizer Adherence Auditor contract."""

from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path

from .realizer_adherence import (
    build_adherence_auditor_input,
    build_adherence_auditor_prompt,
    validate_and_materialize_adherence_audit,
)


MODULE_ROOT = Path(__file__).resolve().parent
SOURCE_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_34406_memory_event_continuation_"
    "contract_v2_live_1")
JUDGMENT = MODULE_ROOT / "fixtures" / (
    "episode_34406_realizer_adherence_semantic_judgment.json")


def _fixture():
    plan = {"scenes": [{
        "memory_event_id": "ME01",
        "participants": ["A", "B"],
        "verified_current_state": "B declined.",
        "new_development": "B changes the decision.",
        "durable_outcome": "B commits.",
        "interaction_beats": ["B changes the decision."],
    }]}
    child = {"scenes": [{
        "scene_ref": "R01:S01",
        "memory_event_id": "ME01",
        "dialogue_turns": [
            {"turn_ref": "R01:S01:T001", "speaker": "A", "text": "I might go."},
            {"turn_ref": "R01:S01:T002", "speaker": "B", "text": "Okay."},
        ],
    }]}
    judgment = {"scene_audits": [{
        "participant_adherence": "fail",
        "participant_evidence_positions": [1],
        "development_status": "not_realized",
        "development_evidence_positions": [1],
        "durable_outcome_status": "not_realized",
        "durable_outcome_evidence_positions": [1, 2],
        "reason": "A performs the action assigned to B.",
    }]}
    return plan, child, judgment


class RealizerAdherenceContractTest(unittest.TestCase):
    def test_input_and_prompt_expose_positions_not_turn_ids(self):
        plan, child, _ = _fixture()
        auditor_input = build_adherence_auditor_input(plan=plan, child=child)
        dialogue = auditor_input["scenes"][0]["child_dialogue"]
        self.assertEqual([row["position"] for row in dialogue], [1, 2])
        self.assertNotIn("turn_ref", dialogue[0])
        self.assertEqual(
            auditor_input["adherence_auditor_input_version"],
            "realizer_adherence_input_v3_required_participants")
        self.assertEqual(
            auditor_input["scenes"][0]["selected_actor_bindings"], [])
        prompt = build_adherence_auditor_prompt(
            auditor_input=auditor_input)["prompt"]
        self.assertIn("Do not output scene IDs, turn IDs", prompt)
        self.assertIn("selected_actor_bindings", prompt)

    def test_prompt_requires_exact_audit_cardinality(self):
        auditor_input = {"scenes": [{"scene_index": index}
                                     for index in range(1, 4)]}
        result = build_adherence_auditor_prompt(auditor_input=auditor_input)
        self.assertEqual(result["scene_count"], 3)
        self.assertEqual(result["prompt_version"],
                         "realizer_adherence_auditor_prompt_v7_concise_description")
        self.assertIn("There are exactly 3 input scenes", result["prompt"])
        self.assertIn("Return exactly 3", result["prompt"])
        self.assertIn("explicit resultative statement", result["prompt"])
        self.assertIn("Judge the exact selected state", result["prompt"])
        self.assertIn("Scan the entire scene", result["prompt"])
        self.assertIn("No semantic verdict or evidence position is prefilled",
                      result["prompt"])

    def test_prompt_does_not_seed_fixed_verdicts_or_evidence_positions(self):
        auditor_input = {"scenes": [{"scene_index": index}
                                     for index in range(1, 4)]}
        prompt = build_adherence_auditor_prompt(
            auditor_input=auditor_input)["prompt"]
        self.assertNotIn('"durable_outcome_status": "partial"', prompt)
        self.assertNotIn('"participant_evidence_positions": [1]', prompt)
        self.assertNotIn('"development_evidence_positions": [1, 2]', prompt)
        self.assertNotIn('"durable_outcome_evidence_positions": [2, 3]', prompt)
        self.assertIn("agreement, decision, awareness", prompt)
        self.assertIn("the meeting is scheduled", prompt)
        self.assertIn("demonstrably brokers the contact", prompt)
        self.assertIn("Mere\nco-presence", prompt)
        self.assertIn("concrete details", prompt)

    def test_input_preserves_frozen_actor_bindings(self):
        plan, child, _ = _fixture()
        plan["scenes"][0]["actor_bindings"] = [
            {"actor": "B", "binding": "B owns the selected action."}]
        auditor_input = build_adherence_auditor_input(plan=plan, child=child)
        self.assertEqual(
            auditor_input["scenes"][0]["selected_actor_bindings"],
            [{"actor": "B", "binding": "B owns the selected action."}])

    def test_program_maps_positions_and_computes_fail(self):
        plan, child, judgment = _fixture()
        result = validate_and_materialize_adherence_audit(
            judgment, plan=plan, child=child)
        self.assertTrue(result["valid"])
        scene = result["audit"]["scenes"][0]
        self.assertEqual(scene["scene_ref"], "R01:S01")
        self.assertEqual(scene["participant_evidence_turn_refs"],
                         ["R01:S01:T001"])
        self.assertEqual(scene["scene_verdict"], "FAIL")
        self.assertEqual(result["audit"]["summary"]["participant_drift_count"], 1)

    def test_pass_requires_all_three_realized_conditions(self):
        plan, child, judgment = _fixture()
        row = judgment["scene_audits"][0]
        row["participant_adherence"] = "pass"
        row["development_status"] = "realized"
        row["durable_outcome_status"] = "realized"
        result = validate_and_materialize_adherence_audit(
            judgment, plan=plan, child=child)
        self.assertTrue(result["valid"])
        self.assertEqual(result["audit"]["scenes"][0]["scene_verdict"], "PASS")

    def test_rejects_free_ids_unknown_status_and_bad_positions(self):
        plan, child, judgment = _fixture()
        bad = deepcopy(judgment)
        bad["scene_audits"][0]["scene_ref"] = "invented"
        result = validate_and_materialize_adherence_audit(
            bad, plan=plan, child=child)
        self.assertFalse(result["valid"])
        self.assertIn("adherence_scene_1_judgment_shape", result["errors"])

        bad = deepcopy(judgment)
        bad["scene_audits"][0]["development_status"] = "mostly"
        bad["scene_audits"][0]["participant_evidence_positions"] = [3]
        result = validate_and_materialize_adherence_audit(
            bad, plan=plan, child=child)
        self.assertFalse(result["valid"])
        self.assertIn("adherence_scene_1_development_status", result["errors"])
        self.assertIn(
            "adherence_scene_1_participant_evidence_positions_out_of_range:3",
            result["errors"])

    def test_rejects_duplicate_positions_and_scene_count(self):
        plan, child, judgment = _fixture()
        bad = deepcopy(judgment)
        bad["scene_audits"][0]["development_evidence_positions"] = [1, 1]
        result = validate_and_materialize_adherence_audit(
            bad, plan=plan, child=child)
        self.assertFalse(result["valid"])
        self.assertIn(
            "adherence_scene_1_development_evidence_positions_duplicate",
            result["errors"])
        result = validate_and_materialize_adherence_audit(
            {"scene_audits": []}, plan=plan, child=child)
        self.assertFalse(result["valid"])
        self.assertIn("adherence_audit_scene_count", result["errors"])

    def test_overflow_is_warning_and_materialized_evidence_is_bounded(self):
        plan, child, judgment = _fixture()
        child["scenes"][0]["dialogue_turns"] = [
            {"turn_ref": f"R01:S01:T{index:03d}", "speaker": "B",
             "text": f"Turn {index}."}
            for index in range(1, 6)]
        judgment["scene_audits"][0]["participant_evidence_positions"] = [
            1, 2, 3, 4, 5]
        result = validate_and_materialize_adherence_audit(
            judgment, plan=plan, child=child)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(
            result["audit"]["scenes"][0]["participant_evidence_turn_refs"],
            [f"R01:S01:T{index:03d}" for index in range(1, 5)])
        self.assertIn(
            "adherence_scene_1_participant_evidence_positions_too_many:5>4",
            result["warnings"])
        self.assertEqual(result["audit"]["summary"]["evidence_overflow_count"], 1)

    def test_not_realized_status_allows_empty_semantic_evidence(self):
        plan, child, judgment = _fixture()
        row = judgment["scene_audits"][0]
        row["development_evidence_positions"] = []
        row["durable_outcome_evidence_positions"] = []
        result = validate_and_materialize_adherence_audit(
            judgment, plan=plan, child=child)
        self.assertTrue(result["valid"], result["errors"])
        scene = result["audit"]["scenes"][0]
        self.assertEqual(scene["development_evidence_turn_refs"], [])
        self.assertEqual(scene["durable_outcome_evidence_turn_refs"], [])

    def test_source_event_alignment_is_required(self):
        plan, child, _ = _fixture()
        child["scenes"][0]["memory_event_id"] = "ME02"
        with self.assertRaisesRegex(ValueError, "event_mismatch"):
            build_adherence_auditor_input(plan=plan, child=child)

    @unittest.skipUnless(SOURCE_RUN.is_dir(), "frozen 34406 run unavailable")
    def test_34406_known_human_conclusions(self):
        plan = json.loads((SOURCE_RUN / "planner" / "materialized_plan.json").read_text())
        child = json.loads((SOURCE_RUN / "child_dialogue.json").read_text())
        judgment = json.loads(JUDGMENT.read_text())
        result = validate_and_materialize_adherence_audit(
            judgment, plan=plan, child=child)
        self.assertTrue(result["valid"])
        scenes = result["audit"]["scenes"]
        self.assertEqual([row["scene_verdict"] for row in scenes],
                         ["FAIL", "FAIL", "FAIL"])
        self.assertEqual(scenes[0]["participant_adherence"], "fail")
        self.assertEqual(scenes[1]["development_status"], "realized")
        self.assertEqual(scenes[1]["durable_outcome_status"], "partial")
        self.assertEqual(scenes[2]["development_status"], "realized")
        self.assertEqual(scenes[2]["durable_outcome_status"], "partial")


if __name__ == "__main__":
    unittest.main()
