"""Offline contract regressions; shape acceptance never certifies B semantics."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from jsonschema import Draft202012Validator

import planner_contract as contract
import runner


V23_RUN = Path(__file__).resolve().parent.parent / "h4_sequential_ab_v23/runs/sequential_pilot_3roots_01"


class PlannerV24Tests(unittest.TestCase):
    def setUp(self):
        self.route = "dataset_new_event"

    def example(self, route=None):
        return contract.planner_shape_example(route or self.route)

    def historical_plans(self):
        paths = sorted((V23_RUN / "requests").glob("*-B-*-planner-*/RESPONSE.json"))
        self.assertEqual(len(paths), 6, "The actual six v23 failed B responses must remain available.")
        for path in paths:
            response = json.loads(path.read_text())
            yield path, json.loads(response["text"])

    def test_prompt_schema_and_example_share_one_authority_for_both_routes(self):
        for route in contract.ROUTES:
            with self.subTest(route=route):
                schema = contract.planner_output_schema(route)
                prompt = contract.planner_contract_prompt(route)
                schema_text, example_text = prompt.split("PLANNER_OUTPUT_SCHEMA_JSON\n", 1)[1].split(
                    "\nPLANNER_SHAPE_EXAMPLE_JSON\n", 1)
                self.assertEqual(json.loads(schema_text), schema)
                example = json.loads(example_text)
                self.assertEqual(example, self.example(route))
                Draft202012Validator.check_schema(schema)
                Draft202012Validator(schema).validate(example)
                self.assertEqual(contract.collect_planner_schema_errors(example, route), [])
                self.assertEqual(next(iter(example["scenes"][0])), "event_definition")
                self.assertNotIn("history_context_reason", schema["properties"]["scenes"]["items"]["properties"])

    def test_six_real_v23_failures_reveal_actual_faults_without_obsolete_reason_or_rewriting(self):
        for path, plan in self.historical_plans():
            with self.subTest(request=path.parent.name):
                before = deepcopy(plan)
                legacy = json.loads((path.parent / "VALIDATION.json").read_text())
                self.assertEqual(legacy["reason"], "scene_fields_invalid")
                self.assertNotIn("history_context_reason", plan["scenes"][0])
                self.assertTrue(plan["scenes"][0]["state_transition"]["basis_reason"].strip())
                errors = contract.collect_planner_schema_errors(plan, self.route)
                duplicate_goal = [error for error in errors if error["path"] == "response.scenes[0].event_definition.main_goal"]
                self.assertEqual(len(duplicate_goal), 1)
                errors = [error for error in errors if error not in duplicate_goal]
                if path.parent.name.startswith("012-"):
                    self.assertEqual({error["path"] for error in errors}, {
                        "response.scenes[0].developments",
                        "response.scenes[0].card_development_links[3].development_indices[1]"})
                    self.assertTrue(all(error["actual"] == 6 for error in errors))
                    self.assertFalse(Draft202012Validator(contract.planner_output_schema(self.route)).is_valid(plan))
                else:
                    self.assertEqual(errors, [])
                    self.assertFalse(Draft202012Validator(contract.planner_output_schema(self.route)).is_valid(plan))
                self.assertEqual(plan, before)
                # There was no v23 B realization/review; this checks the obsolete
                # field failure only, and does not write or promote any candidate.
                self.assertIn("Shape validity does not prove", contract.planner_contract_prompt(self.route))

    def test_missing_current_field_on_each_actual_response_has_precise_repair(self):
        for path, plan in self.historical_plans():
            with self.subTest(request=path.parent.name):
                baseline_errors = contract.collect_planner_schema_errors(plan, self.route)
                del plan["scenes"][0]["state_transition"]["basis_reason"]
                before = deepcopy(plan)
                with self.assertRaises(contract.PlannerSchemaValidationError) as caught:
                    contract.validate_planner_schema(plan, self.route)
                self.assertEqual(len(caught.exception.errors), len(baseline_errors) + 1)
                error = next(error for error in caught.exception.errors
                             if error["code"] == "planner_schema_missing_field")
                self.assertEqual(error["path"], "response.scenes[0].state_transition.basis_reason")
                self.assertEqual(error["missing"], ["basis_reason"])
                self.assertEqual(error["expected_type"], "string")
                feedback = contract.build_repair_feedback(caught.exception, plan, route=self.route)
                self.assertEqual(feedback["errors"], caught.exception.errors)
                self.assertEqual(feedback["previous_response"], before)
                self.assertEqual(feedback["planner_output_schema"], contract.planner_output_schema(self.route))
                self.assertFalse(feedback["semantic_certified"])
                self.assertTrue(feedback["shape_example_is_not_a_candidate"])
                self.assertEqual(plan, before)

    def test_all_nested_missing_extra_and_type_errors_are_reported_together(self):
        plan = self.example()
        scene = plan["scenes"][0]
        del scene["event_definition"]["trigger"]
        scene["event_definition"]["main_goal"] = 8
        scene["history_context_reason"] = "obsolete duplicated explanation"
        scene["state_transition"]["basis_fact_ids"] = [None, "f1"]
        scene["state_transition"]["basis_reason"] = []
        scene["card_development_links"][0]["development_indices"] = [True, "1"]
        scene["source_anchors"] = [{"turn_ref": 7, "unrecognized": "bad"}]
        before = deepcopy(plan)
        errors = contract.collect_planner_schema_errors(plan, self.route)
        paths = {error["path"] for error in errors}
        expected = {
            "response.scenes[0].event_definition.trigger",
            "response.scenes[0].event_definition.main_goal",
            "response.scenes[0].history_context_reason",
            "response.scenes[0].state_transition.basis_fact_ids[0]",
            "response.scenes[0].state_transition.basis_reason",
            "response.scenes[0].card_development_links[0].development_indices[0]",
            "response.scenes[0].card_development_links[0].development_indices[1]",
            "response.scenes[0].source_anchors[0].quote",
            "response.scenes[0].source_anchors[0].turn_ref",
            "response.scenes[0].source_anchors[0].unrecognized",
        }
        self.assertEqual(paths, expected)
        self.assertEqual(plan, before)

    def test_runner_reports_complete_schema_errors_before_generic_checks(self):
        plan = self.example()
        del plan["scenes"][0]["state_transition"]["basis_reason"]
        plan["scenes"][0]["actors"] = None
        with patch.object(runner.base, "validate_topic_ids", side_effect=AssertionError("generic validator reached")):
            with self.assertRaises(contract.PlannerSchemaValidationError) as caught:
                runner.validate_plan(plan, plan["root_id"], self.route, [], [], "S01")
        self.assertEqual({error["path"] for error in caught.exception.errors}, {
            "response.scenes[0].actors", "response.scenes[0].state_transition.basis_reason"})

    def test_parent_shape_errors_do_not_hide_independent_sibling_errors(self):
        plan = self.example()
        plan["scenes"] = [None, {"unexpected": 2}]
        plan["shortfall_reason"] = None
        errors = contract.collect_planner_schema_errors(plan, self.route)
        self.assertIn("response.scenes[0]", {error["path"] for error in errors})
        self.assertIn("response.scenes[1].event_definition", {error["path"] for error in errors})
        self.assertIn("response.shortfall_reason", {error["path"] for error in errors})
        self.assertIn("response.scenes", {error["path"] for error in errors})

    def test_empty_history_references_require_one_nonblank_explanation_even_without_snapshot(self):
        plan = self.example()
        for value in ("", "  ", None):
            with self.subTest(value=value):
                plan["scenes"][0]["state_transition"]["basis_reason"] = value
                with self.assertRaises(contract.PlannerSchemaValidationError):
                    runner.validate_plan(plan, plan["root_id"], self.route, [], [], "S01")

    def test_declaring_distinct_event_is_only_shape_not_semantic_approval(self):
        plan = self.example()
        scene = plan["scenes"][0]
        scene["event_definition"]["trigger"] = "Book transport for the existing interview."
        scene["goal"] = "Reach the existing interview."
        self.assertTrue(contract.validate_planner_schema(plan, self.route))
        self.assertNotIn("semantic_certified", plan)
        self.assertIn("not prove a distinct B event", contract.planner_contract_prompt(self.route))

    def mapping_fixture(self, level):
        card = {"card_id": "source-card", "guidance_level": level, "steps": [
            {"step_id": "s1", "action": "Request a celebratory dinner", "state": "proposed"},
            {"step_id": "s2", "action": "Choose a time", "state": "agreed"},
            {"step_id": "s3", "action": "Confirm booking", "state": "reported"},
        ], "dependencies": [{"before": "s1", "after": "s2", "relation": "choose after request"}],
                "source_quotes": ["NEVER ROOT EVIDENCE"]}
        scene = {"dataset_card_id": "source-card",
                 "developments": ["Propose the new dinner.", "Agree on who will organize it."],
                 "card_development_links": [{"card_step_ids": ["s1"], "dependency_indices": [],
                                            "development_indices": [1, 2]}]}
        return scene, card

    def test_seed_selects_premise_steps_without_full_workflow_coverage(self):
        scene, card = self.mapping_fixture("seed")
        before = deepcopy((scene, card))
        self.assertTrue(contract.validate_mapping(scene, card))
        context = contract.mapping_context(card, scene)
        self.assertIn("optional", context["coverage_requirement"])
        self.assertFalse(context["card_guidance"]["source_is_root_evidence"])
        self.assertNotIn("NEVER ROOT EVIDENCE", json.dumps(context))
        self.assertEqual((scene, card), before)

    def test_scaffold_still_requires_all_source_steps_and_dependencies(self):
        scene, card = self.mapping_fixture("scaffold")
        errors = contract.collect_mapping_errors(scene, card)
        by_code = {error["code"]: error for error in errors}
        self.assertEqual(by_code["card_steps_not_fully_mapped"]["missing"], ["s2", "s3"])
        self.assertEqual(by_code["card_dependencies_not_fully_mapped"]["missing"], [1])
        scene["card_development_links"][0]["card_step_ids"] = ["s1", "s2", "s3"]
        scene["card_development_links"][0]["dependency_indices"] = [1]
        self.assertTrue(contract.validate_mapping(scene, card))

    def test_seed_still_requires_actual_selected_step_and_valid_references(self):
        scene, card = self.mapping_fixture("seed")
        link = scene["card_development_links"][0]
        link.update(card_step_ids=["invented"], dependency_indices=[True, 2], development_indices=[9])
        errors = contract.collect_mapping_errors(scene, card)
        codes = {error["code"] for error in errors}
        self.assertTrue({"card_seed_premise_source_step_missing", "card_development_link_steps_invalid",
                         "card_development_link_dependencies_invalid", "card_development_link_indices_invalid"}
                        <= codes)
        self.assertNotIn("card_steps_not_fully_mapped", codes)
        self.assertNotIn("card_dependencies_not_fully_mapped", codes)

    def test_duplicate_selected_seed_step_is_not_silently_removed(self):
        scene, card = self.mapping_fixture("seed")
        scene["card_development_links"].append(deepcopy(scene["card_development_links"][0]))
        before = deepcopy(scene)
        with self.assertRaises(contract.MappingValidationError) as caught:
            contract.validate_mapping(scene, card)
        self.assertIn("card_step_mapped_more_than_once", {error["code"] for error in caught.exception.errors})
        self.assertEqual(scene, before)


if __name__ == "__main__":
    unittest.main()
