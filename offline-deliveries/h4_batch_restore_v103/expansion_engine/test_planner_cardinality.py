"""Offline regressions for the actual v90 cardinality failures.

The merged plans below are test fixtures only. The production repair never
merges, truncates or rewrites model answers itself.
"""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import unittest

import planner_contract as pc


HERE = Path(__file__).resolve().parent
V90_REQUESTS = (
    HERE.parents[1] / "h4_end_to_end_expansion_mixed_v90/expansion_engine"
    / "runs/four_roots_B_scene01/requests"
)
REAL_FAILURE_DIRECTORIES = (
    "011-cc-train-episode-104814-B-S01-planner-1",
    "012-cc-train-episode-104814-B-S01-planner-2",
    "020-cc-train-episode-194259-B-S01-planner-1",
    "021-cc-train-episode-194259-B-S01-planner-2",
)


def load_real_failure(name):
    folder = V90_REQUESTS / name
    request = json.loads((folder / "REQUEST.json").read_text())
    prompt = request["messages"][0]["content"]
    context, _ = json.JSONDecoder().raw_decode(prompt.split("\nINPUT_JSON\n", 1)[1])
    response = json.loads((folder / "RESPONSE.json").read_text())
    return {
        "name": name,
        "raw": response["text"],
        "plan": json.loads(response["text"]),
        "card": context["dataset_cards"][0],
        "validation": json.loads((folder / "VALIDATION.json").read_text()),
        "file_hashes": {
            str(folder / filename): sha256((folder / filename).read_bytes()).hexdigest()
            for filename in ("REQUEST.json", "RESPONSE.json", "VALIDATION.json")
        },
    }


def merged_test_scene(failure):
    """Keep all card actions in five developments, solely as an offline fixture."""
    scene = deepcopy(failure["plan"]["scenes"][0])
    # The retry mappings correctly reference original position six. Start from
    # them so this fixture also exercises renumbering rather than silent drops.
    merge_position = 1 if "104814" in failure["name"] else 5  # one-based
    first = scene["developments"][merge_position - 1]
    second = scene["developments"][merge_position]
    relation = " As part of that same request, " if merge_position == 1 else " In response, "
    scene["developments"][merge_position - 1:merge_position + 1] = [first + relation + second]
    for link in scene["card_development_links"]:
        indices = [index - 1 if index > merge_position else index
                   for index in link["development_indices"]]
        link["development_indices"] = list(dict.fromkeys(indices))
    return scene


class PlannerCardinalityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.failures = [load_real_failure(name) for name in REAL_FAILURE_DIRECTORIES]
        cls.retries = [failure for failure in cls.failures if failure["name"].endswith("-2")]

    @classmethod
    def tearDownClass(cls):
        for failure in cls.failures:
            for filename, expected in failure["file_hashes"].items():
                if sha256(Path(filename).read_bytes()).hexdigest() != expected:
                    raise AssertionError("Historical regression evidence changed: " + filename)

    def test_actual_failures_repeated_six_developments(self):
        for initial, retry in zip(self.failures[::2], self.failures[1::2]):
            self.assertEqual(initial["plan"]["scenes"][0]["developments"],
                             retry["plan"]["scenes"][0]["developments"])
            self.assertEqual(len(initial["plan"]["scenes"][0]["developments"]), 6)
            for failure in (initial, retry):
                self.assertEqual(failure["validation"]["reason"], "scene_developments_count_invalid")

    def test_real_failure_feedback_never_authorizes_index_six(self):
        for failure in self.failures:
            scene, card = failure["plan"]["scenes"][0], failure["card"]
            error = pc.MappingValidationError(pc.collect_mapping_errors(scene, card))
            feedback = pc.build_repair_feedback(error, failure["raw"], card)
            self.assertEqual(feedback["allowed_reference_values"]["development_indices"], [1, 2, 3, 4, 5])
            context = feedback["card_mapping_context"]
            self.assertEqual(context["observed_scene_development_count"], 6)
            self.assertFalse(context["observed_scene_development_count_valid"])
            self.assertIn("corrected_response", context["development_reference_scope"])
            self.assertTrue(feedback["development_count_repair"]["must_revise_developments_array"])
            self.assertTrue(feedback["development_count_repair"]["must_rebuild_all_card_development_links"])

    def test_feedback_preserves_raw_card_steps_dependencies_and_errors(self):
        for failure in self.failures:
            card, plan = deepcopy(failure["card"]), deepcopy(failure["plan"])
            before = deepcopy((card, plan))
            errors = pc.collect_mapping_errors(plan["scenes"][0], card)
            error = pc.MappingValidationError(errors)
            error_snapshot = deepcopy(error.errors)
            feedback = pc.build_repair_feedback(error, plan, card)
            self.assertEqual((card, plan), before)
            self.assertEqual(feedback["previous_response"], plan)
            self.assertEqual(error.errors, error_snapshot)
            self.assertEqual(feedback["errors"], errors)
            guidance = feedback["card_mapping_context"]["card_guidance"]
            for field in ("steps", "dependencies"):
                self.assertEqual(guidance[field], card[field])
            # Mutating diagnostics must not mutate the original inputs.
            feedback["previous_response"]["scenes"][0]["developments"].clear()
            guidance["steps"].clear()
            self.assertEqual((card, plan), before)

    def test_valid_scene_context_keeps_actual_reference_domain(self):
        card = self.failures[0]["card"]
        for count in (2, 3, 4, 5):
            context = pc.mapping_context(card, {"developments": ["new action"] * count})
            self.assertEqual(context["allowed_reference_values"]["development_indices"], list(range(1, count + 1)))
            self.assertEqual(context["development_reference_scope"], "actual_scene_developments")
            self.assertTrue(context["observed_scene_development_count_valid"])

    def test_invalid_or_unplanned_context_uses_only_corrected_potential_domain(self):
        card = self.failures[0]["card"]
        scenes = (None, {}, {"developments": "not an array"})
        scenes += tuple({"developments": ["new action"] * count} for count in (0, 1, 6, 9))
        for scene in scenes:
            with self.subTest(scene=scene):
                context = pc.mapping_context(card, scene)
                self.assertEqual(context["allowed_reference_values"]["development_indices"], [1, 2, 3, 4, 5])
                self.assertNotEqual(context["development_reference_scope"], "actual_scene_developments")

    def test_oversized_actual_responses_still_rejected(self):
        for failure in self.failures:
            scene = failure["plan"]["scenes"][0]
            with self.assertRaises(pc.MappingValidationError) as caught:
                pc.validate_mapping(scene, failure["card"])
            self.assertIn("scene_developments_count_invalid", [error["code"] for error in caught.exception.errors])

    def test_complete_five_development_fixtures_pass_without_card_loss(self):
        for failure in self.retries:
            scene = merged_test_scene(failure)
            self.assertEqual(len(scene["developments"]), 5)
            self.assertTrue(pc.validate_mapping(scene, failure["card"]))
            mapped_steps = [step for link in scene["card_development_links"] for step in link["card_step_ids"]]
            mapped_dependencies = [dep for link in scene["card_development_links"] for dep in link["dependency_indices"]]
            self.assertCountEqual(mapped_steps, [step["step_id"] for step in failure["card"]["steps"]])
            self.assertCountEqual(mapped_dependencies, [1, 2, 3])
            self.assertEqual(len(mapped_steps), 6)

    def test_corrected_scene_with_stale_index_six_rejected(self):
        for failure in self.retries:
            scene = merged_test_scene(failure)
            scene["card_development_links"][-1]["development_indices"] = [6]
            with self.assertRaises(pc.MappingValidationError) as caught:
                pc.validate_mapping(scene, failure["card"])
            self.assertIn("card_development_link_indices_invalid", [error["code"] for error in caught.exception.errors])

    def test_missing_step_or_dependency_cannot_satisfy_contract(self):
        for failure in self.retries:
            for field, code in (("card_step_ids", "card_steps_not_fully_mapped"),
                                ("dependency_indices", "card_dependencies_not_fully_mapped")):
                scene = merged_test_scene(failure)
                link = next(link for link in scene["card_development_links"] if link[field])
                link[field].pop()
                with self.assertRaises(pc.MappingValidationError) as caught:
                    pc.validate_mapping(scene, failure["card"])
                self.assertIn(code, [error["code"] for error in caught.exception.errors])

    def test_duplicate_step_or_dependency_still_rejected(self):
        for failure in self.retries:
            for field, code in (("card_step_ids", "card_step_mapped_more_than_once"),
                                ("dependency_indices", "card_dependency_mapped_more_than_once")):
                scene = merged_test_scene(failure)
                link = next(link for link in scene["card_development_links"] if link[field])
                link[field].append(link[field][0])
                with self.assertRaises(pc.MappingValidationError) as caught:
                    pc.validate_mapping(scene, failure["card"])
                self.assertIn(code, [error["code"] for error in caught.exception.errors])

    def test_boolean_reference_is_not_an_integer_index(self):
        failure = self.retries[0]
        scene = merged_test_scene(failure)
        scene["card_development_links"][0]["development_indices"] = [True]
        with self.assertRaises(pc.MappingValidationError) as caught:
            pc.validate_mapping(scene, failure["card"])
        self.assertIn("card_development_link_indices_invalid", [error["code"] for error in caught.exception.errors])

    def test_final_reminder_is_pure_and_only_uses_known_error_code(self):
        repair = {"errors": [{"code": "scene_developments_count_invalid", "actual": 6}],
                  "previous_response": "MALICIOUS_OR_HISTORICAL_TEXT"}
        before = deepcopy(repair)
        instruction = pc.final_planner_instruction(repair)
        self.assertIn("CARDINALITY REPAIR REQUIRED", instruction)
        self.assertIn("Changing only indices", instruction)
        self.assertNotIn(repair["previous_response"], instruction)
        self.assertEqual(repair, before)
        for other in (None, {}, {"errors": {}}, {"errors": ["bad error"]},
                      {"errors": [{"code": "unrelated_failure"}]}):
            reminder = pc.final_planner_instruction(other)
            self.assertIn("2-5 entries", reminder)
            self.assertNotIn("CARDINALITY REPAIR REQUIRED", reminder)


if __name__ == "__main__":
    unittest.main()
