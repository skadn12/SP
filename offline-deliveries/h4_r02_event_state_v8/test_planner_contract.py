"""Reference-integrity regressions; no provider API or generated dialogue."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import unittest

import planner_contract as contract


class PlannerContractTests(unittest.TestCase):
    def setUp(self):
        self.card = {
            "card_id": "six_step_card", "guidance_level": "scaffold",
            "source_id": "must-not-leak", "source_text": "RAW PRIVATE SOURCE",
            "actors": [{"actor_id": "a1", "role": "requester", "source_name": "PRIVATE NAME"}],
            "steps": [{"step_id": f"s{i}", "actor_id": "a1", "action": f"abstract action {i}",
                       "state": "proposed", "evidence": [{"quote": "RAW PRIVATE QUOTE"}]}
                      for i in range(1, 7)],
            "dependencies": [
                {"before": "s1", "after": "s2", "relation": "order", "evidence": [{"quote": "RAW PRIVATE QUOTE"}]},
                {"before": "s2", "after": "s3", "relation": "condition_change"},
                {"before": "s4", "after": "s5", "relation": "correction"},
            ],
        }
        self.scene = {
            "dataset_card_id": self.card["card_id"],
            "developments": [f"abstract development {i}" for i in range(1, 6)],
            "card_development_links": [
                {"card_step_ids": ["s1", "s2"], "dependency_indices": [1], "development_indices": [1]},
                {"card_step_ids": ["s3"], "dependency_indices": [2], "development_indices": [2]},
                {"card_step_ids": ["s4"], "dependency_indices": [], "development_indices": [3]},
                {"card_step_ids": ["s5"], "dependency_indices": [3], "development_indices": [4]},
                {"card_step_ids": ["s6"], "dependency_indices": [], "development_indices": [5]},
            ],
        }

    def test_six_steps_three_dependencies_five_developments_is_valid_without_mutation(self):
        originals = deepcopy((self.card, self.scene))
        self.assertTrue(contract.validate_mapping(self.scene, self.card))
        self.assertEqual(contract.collect_mapping_errors(self.scene, self.card), [])
        self.assertEqual((self.card, self.scene), originals)

    def test_numbered_context_strips_provenance_and_separates_reference_spaces(self):
        original = deepcopy(self.card)
        context = contract.mapping_context(self.card, self.scene)
        self.assertEqual(context["allowed_reference_values"], {
            "card_step_ids": [f"s{i}" for i in range(1, 7)],
            "dependency_indices": [1, 2, 3], "development_indices": [1, 2, 3, 4, 5],
        })
        self.assertEqual(context["numbered_dependencies"][2], {
            "dependency_index": 3, "before": "s4", "after": "s5", "relation": "correction",
        })
        serialized = json.dumps(context)
        for forbidden in ("source_id", "source_text", "source_name", "evidence", "RAW PRIVATE", "PRIVATE NAME"):
            self.assertNotIn(forbidden, serialized.replace("source_is_root_evidence", ""))
        self.assertEqual(self.card, original)
        context["card_guidance"]["steps"][0]["action"] = "caller mutation"
        self.assertEqual(self.card, original)

    def test_unplanned_context_does_not_claim_five_actual_developments(self):
        context = contract.mapping_context(self.card)
        self.assertIn("potential_positions_only", context["development_reference_scope"])
        self.scene["developments"] = self.scene["developments"][:2]
        context = contract.mapping_context(self.card, self.scene)
        self.assertEqual(context["allowed_reference_values"]["development_indices"], [1, 2])
        self.assertEqual(context["development_reference_scope"], "actual_scene_developments")

    def test_all_errors_include_later_out_of_range_and_missing_dependency(self):
        links = self.scene["card_development_links"]
        links[0]["dependency_indices"] = [4]
        links[1]["development_indices"] = [6]
        links[3]["dependency_indices"] = []
        original = deepcopy(self.scene)
        errors = contract.collect_mapping_errors(self.scene, self.card)
        codes = {error["code"] for error in errors}
        self.assertIn("card_development_link_dependencies_invalid", codes)
        self.assertIn("card_development_link_indices_invalid", codes)
        self.assertIn("card_dependencies_not_fully_mapped", codes)
        missing = next(e for e in errors if e["code"] == "card_dependencies_not_fully_mapped")
        self.assertEqual(missing["missing"], [1, 3])
        invalid = next(e for e in errors if e["code"] == "card_development_link_dependencies_invalid")
        self.assertEqual(invalid["path"], "scene.card_development_links[0].dependency_indices")
        self.assertEqual(invalid["actual"], [4])
        self.assertEqual(invalid["allowed"], [1, 2, 3])
        with self.assertRaises(contract.MappingValidationError) as caught:
            contract.validate_mapping(self.scene, self.card)
        self.assertEqual(str(caught.exception), errors[0]["code"])
        self.assertEqual(caught.exception.errors, errors)
        self.assertEqual(self.scene, original)

    def test_seed_with_explicit_dependencies_cannot_drop_them(self):
        self.card["guidance_level"] = "seed"
        for link in self.scene["card_development_links"]:
            link["dependency_indices"] = []
        with self.assertRaisesRegex(contract.MappingValidationError, "card_dependencies_not_fully_mapped"):
            contract.validate_mapping(self.scene, self.card)
        self.card["dependencies"] = []
        self.assertTrue(contract.validate_mapping(self.scene, self.card))

    def test_bool_and_string_indices_do_not_count_as_coverage(self):
        self.scene["card_development_links"][0]["dependency_indices"] = [True, "1"]
        self.scene["card_development_links"][0]["development_indices"] = [True, "1"]
        errors = contract.collect_mapping_errors(self.scene, self.card)
        self.assertEqual(next(e for e in errors if e["code"] == "card_dependencies_not_fully_mapped")["missing"], [1])
        self.assertIn("card_development_link_indices_invalid", {e["code"] for e in errors})

    def test_duplicate_references_are_rejected_within_and_across_links(self):
        self.scene["card_development_links"][0]["card_step_ids"] = ["s1", "s2", "s1"]
        self.scene["card_development_links"][1]["dependency_indices"] = [1, 2]
        errors = contract.collect_mapping_errors(self.scene, self.card)
        self.assertIn("card_step_mapped_more_than_once", {e["code"] for e in errors})
        self.assertIn("card_dependency_mapped_more_than_once", {e["code"] for e in errors})

    def test_malformed_fields_still_collect_other_independent_faults(self):
        self.scene["card_development_links"][0] = {"card_step_ids": {"s1": True}, "extra": []}
        self.scene["card_development_links"].append(None)
        errors = contract.collect_mapping_errors(self.scene, self.card)
        codes = {e["code"] for e in errors}
        self.assertTrue({"card_development_link_fields_invalid", "card_development_link_steps_invalid",
                         "card_development_link_dependencies_invalid", "card_development_link_indices_invalid",
                         "card_steps_not_fully_mapped", "card_dependencies_not_fully_mapped"}.issubset(codes))
        self.assertTrue(contract.collect_mapping_errors([], None))

    def test_six_developments_are_not_an_escape_from_six_step_mapping(self):
        self.scene["developments"].append("sixth development")
        self.assertIn("scene_developments_count_invalid", {
            error["code"] for error in contract.collect_mapping_errors(self.scene, self.card)})

    def test_repair_preserves_raw_and_lists_every_fault_and_legal_value(self):
        self.scene["card_development_links"][0]["dependency_indices"] = [6]
        raw = json.dumps({"scenes": [self.scene]})
        original = deepcopy(self.card)
        try:
            contract.validate_mapping(self.scene, self.card)
        except contract.MappingValidationError as error:
            feedback = contract.build_repair_feedback(error, raw, self.card)
        self.assertEqual(feedback["previous_response"], raw)
        self.assertEqual(feedback["errors"], contract.collect_mapping_errors(self.scene, self.card))
        self.assertEqual(feedback["allowed_reference_values"]["dependency_indices"], [1, 2, 3])
        self.assertEqual(self.card, original)
        fallback = contract.build_repair_feedback(ValueError("malformed_json"), "{broken", self.card)
        self.assertEqual(fallback["failure"], "malformed_json")
        self.assertEqual(fallback["previous_response"], "{broken")
        self.assertEqual(fallback["errors"][0]["code"], "malformed_json")

    def test_complete_historical_planner_replay_remains_rejected(self):
        run = Path(__file__).resolve().parent.parent / "h4_r02_stage_repair_v5/runs/r02_stage_repair_v5_01"
        if not run.exists():
            self.skipTest("immutable v5 live audit not available")
        requests = sorted((run / "requests").glob("*-B-*-planner-*"))
        self.assertEqual(len(requests), 16)
        by_response = Counter()
        for directory in requests:
            request = json.loads((directory / "REQUEST.json").read_text())
            prompt = request["messages"][0]["content"]
            context, _ = json.JSONDecoder().raw_decode(prompt.split("\nINPUT_JSON\n", 1)[1])
            raw = json.loads((directory / "RESPONSE.json").read_text())["text"]
            response = json.loads(raw)
            errors = contract.collect_mapping_errors(response["scenes"][0], context["dataset_cards"][0])
            self.assertTrue(errors, directory.name)
            with self.assertRaises(contract.MappingValidationError):
                contract.validate_mapping(response["scenes"][0], context["dataset_cards"][0])
            by_response.update({e["code"] for e in errors})
        self.assertEqual(by_response["card_development_link_dependencies_invalid"], 13)
        self.assertEqual(by_response["card_development_link_indices_invalid"], 3)
        self.assertEqual(by_response["card_dependencies_not_fully_mapped"], 5)


if __name__ == "__main__":
    unittest.main()
