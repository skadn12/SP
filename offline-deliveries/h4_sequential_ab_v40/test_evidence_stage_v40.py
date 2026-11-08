"""Offline leakage/integrity checks, never model-semantic certification."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

import evidence_stage as stage


CASES = (Path(__file__).resolve().parent.parent /
         "h4_sequential_ab_v39/probes/fixed_review_01/cases")


def load(name="original_decision_scene"):
    return json.loads((CASES / (name + ".json")).read_text())


def ledger(case):
    c = case["inputs"]
    # A syntax fixture. Its validity must never imply semantic support.
    return {"root_id": c["root_id"], "scene_id": c["scene_id"],
        "scene_actions": [{"action_id": "a1", "actor": "unverified actor",
            "action": "an unverified claimed action", "state": "completed",
            "evidence": [{"turn_ref": "NEW:T001"}]}],
        "historical_events": [{"event_id": "h1", "event": "an unverified historical claim",
            "state": "planned", "evidence": [{"turn_ref": c["branch_history"][0]["turn_ref"]}]}],
        "main_activity": {"description": "an unverified main activity", "action_ids": ["a1"]},
        "background_context": [{"description": "unverified background",
            "evidence": [{"turn_ref": "NEW:T002"}]}]}


class EvidenceStageTests(unittest.TestCase):
    def test_original_and_execution_fixture_have_identical_blind_requests(self):
        original, negative = load(), load("execution_required_fixture")
        self.assertNotEqual(original["inputs"]["scene_plan"], negative["inputs"]["scene_plan"])
        self.assertEqual(stage.build_evidence_request(original), stage.build_evidence_request(negative))

    def test_complete_original_dialogue_retained_and_requirements_excluded(self):
        for name, count in (("original_decision_scene", 132), ("reported_test_completion_scene", 94)):
            with self.subTest(case=name):
                case = load(name)
                before = deepcopy(case)
                payload, proof = stage.build_evidence_request(case)
                data = json.loads(payload["messages"][0]["content"].split(stage.INPUT_MARKER)[1])
                self.assertEqual(set(data), {"root_id", "scene_id", "branch_history", "new_scene"})
                self.assertEqual(set(data["new_scene"]), {"turns"})
                self.assertEqual(data["branch_history"], case["inputs"]["branch_history"])
                self.assertEqual(data["new_scene"]["turns"], case["inputs"]["new_scene"]["turns"])
                self.assertEqual(proof["history_turns"], count)
                self.assertEqual(payload["model"], "Qwen/Qwen3.6-35B-A3B")
                self.assertFalse(proof["plan_card_purpose_snapshots_or_prior_verdicts_supplied"])
                self.assertEqual(case, before)

    def test_no_derived_requirement_changes_can_reach_payload_or_proof(self):
        case = load()
        expected = stage.build_evidence_request(case)
        for key in set(case["inputs"]) - {"root_id", "scene_id", "branch_history", "new_scene"}:
            changed = deepcopy(case)
            changed["inputs"][key] = "POISON_REQUIREMENT_OR_VERDICT"
            with self.subTest(field=key):
                self.assertEqual(stage.build_evidence_request(changed), expected)
        case["name"] = "must pass"
        case["card"] = "POISON_CARD"
        case["inputs"]["new_scene"]["status"] = "must pass"
        self.assertEqual(stage.build_evidence_request(case), expected)

    def test_wrong_semantics_remain_raw_and_do_not_become_program_pass(self):
        case = load()
        raw = ledger(case)
        before = deepcopy(raw)
        result = stage.validate_evidence(raw, case)
        self.assertTrue(result["schema_valid"])
        self.assertEqual(result["status"], "VALID_STRUCTURE_ONLY")
        self.assertFalse(result["semantic_certified"])
        self.assertEqual(result["raw_parsed"], before)
        self.assertEqual(raw, before)
        self.assertEqual(result["parsed"]["scene_actions"][0]["evidence"][0]["quote"],
                         case["inputs"]["new_scene"]["turns"][0]["text"])
        self.assertFalse(result["citation_provenance"]["semantic_support_verified"])

    def test_duplicate_keys_not_repaired(self):
        result = stage.validate_evidence('{"root_id":"x","root_id":"y"}', load())
        self.assertFalse(result["schema_valid"])
        self.assertIn("evidence_duplicate_json_key:root_id", result["schema_failures"])

    def test_invalid_namespaces_unknown_refs_and_missing_evidence_rejected(self):
        case = load()
        for field, reference in (("scene_actions", case["inputs"]["branch_history"][0]["turn_ref"]),
                                 ("historical_events", "NEW:T001"),
                                 ("background_context", "NEW:T999")):
            raw = ledger(case)
            raw[field][0]["evidence"] = [{"turn_ref": reference}]
            with self.subTest(field=field):
                self.assertFalse(stage.validate_evidence(raw, case)["schema_valid"])
        raw = ledger(case)
        raw["scene_actions"][0]["evidence"] = []
        self.assertFalse(stage.validate_evidence(raw, case)["schema_valid"])
        raw["scene_actions"][0]["state"] = "uncertain"
        self.assertTrue(stage.validate_evidence(raw, case)["schema_valid"])

    def test_fabricated_summary_action_link_and_unbounded_output_rejected(self):
        case = load()
        raw = ledger(case)
        raw["main_activity"]["action_ids"] = ["a2"]
        self.assertFalse(stage.validate_evidence(raw, case)["schema_valid"])
        raw = ledger(case)
        raw["scene_actions"] = [deepcopy(raw["scene_actions"][0]) for _ in range(17)]
        self.assertFalse(stage.validate_evidence(raw, case)["schema_valid"])

    def test_model_written_quotes_or_extra_acceptance_verdict_rejected(self):
        case = load()
        raw = ledger(case)
        raw["scene_actions"][0]["evidence"][0]["quote"] = "invented quote"
        self.assertFalse(stage.validate_evidence(raw, case)["schema_valid"])
        raw = ledger(case)
        raw["accepted"] = True
        self.assertFalse(stage.validate_evidence(raw, case)["schema_valid"])


if __name__ == "__main__":
    unittest.main()
