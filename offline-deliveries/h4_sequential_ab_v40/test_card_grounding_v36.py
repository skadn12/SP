from copy import deepcopy
import json
from pathlib import Path
import unittest

import card_grounding as cg
import planner_contract as pc


def source_and_scene():
    card = {"card_id": "source-card", "guidance_level": "scaffold",
        "actors": [{"actor_id": "a1", "role": "requester"}, {"actor_id": "a2", "role": "provider"}],
        "steps": [{"step_id": "s1", "actor_id": "a1", "action": "request details", "state": "requested"},
                  {"step_id": "s2", "actor_id": "a2", "action": "provide details", "state": "reported"}],
        "dependencies": [{"before": "s1", "after": "s2", "relation": "prerequisite"}]}
    scene = {"dataset_card_id": "source-card", "actors": ["Emily", "James"],
        "developments": ["Emily asks for details.", "James gives details.", "Emily thanks James."],
        "card_actor_bindings": [{"actor_id": "a1", "participant": "Emily"}, {"actor_id": "a2", "participant": "James"}],
        "card_development_links": [
            {"card_step_ids": ["s1"], "dependency_indices": [], "development_indices": [1]},
            {"card_step_ids": ["s2"], "dependency_indices": [1], "development_indices": [2]}]}
    return card, scene


def checks():
    return [{"kind": "step", "id": "s1", "verdict": "compatible", "evidence_development_indices": [1]},
            {"kind": "step", "id": "s2", "verdict": "compatible", "evidence_development_indices": [2]},
            {"kind": "dependency", "id": 1, "verdict": "compatible", "evidence_development_indices": [1, 2]}]


class GroundingTests(unittest.TestCase):
    def test_strict_schema_optional_legacy_and_new_required(self):
        legacy = pc.planner_shape_example(pc.ROUTES[1])
        self.assertTrue(pc.validate_planner_schema(legacy, pc.ROUTES[1]))
        self.assertTrue(pc.collect_planner_schema_errors(legacy, pc.ROUTES[1], require_grounding=True))
        new = pc.planner_shape_example(pc.ROUTES[1], require_grounding=True)
        self.assertTrue(pc.validate_planner_schema(new, pc.ROUTES[1], require_grounding=True))
        self.assertIn("card_actor_bindings", new["scenes"][0])
        self.assertNotIn("card_actor_bindings", pc.planner_shape_example(pc.ROUTES[0], require_grounding=True)["scenes"][0])

    def test_fixed_source_roles_and_current_endpoint_projection(self):
        card, scene = source_and_scene()
        self.assertTrue(pc.validate_mapping(scene, card, require_grounding=True))
        view = cg.expected_card_grounding(scene, [card])
        self.assertEqual(view["steps"][0]["participant"], "Emily")
        self.assertEqual(view["dependencies"][0]["before_development_indices"], [1])
        self.assertEqual(view["dependencies"][0]["after_development_indices"], [2])
        self.assertFalse(cg.validate_card_grounding_checks(checks(), scene, [card]))
        self.assertEqual(pc.mapping_context(card, scene, require_grounding=True)["current_card_grounding"], view)

    def test_source_role_binding_missing_duplicate_and_unselected(self):
        for mutated in (None, [{"actor_id": "a1", "participant": "Emily"}],
                        [{"actor_id": "a1", "participant": "Emily"}] * 2,
                        [{"actor_id": "a3", "participant": "Emily"}]):
            card, scene = source_and_scene()
            scene["card_actor_bindings"] = mutated
            self.assertTrue(cg.collect_mapping_grounding_errors(scene, card))

    def test_reported_actor_and_same_participant_are_not_structural_failures(self):
        card, scene = source_and_scene()
        scene["card_actor_bindings"][0]["participant"] = "An explicitly reported third participant"
        scene["card_actor_bindings"][1]["participant"] = "An explicitly reported third participant"
        self.assertFalse(cg.collect_mapping_grounding_errors(scene, card))

    def test_typed_malformed_actor_values_fail_without_hash_crash(self):
        for value in ([], {}, None, 3):
            card, scene = source_and_scene()
            scene["card_actor_bindings"][0]["actor_id"] = value
            self.assertTrue(cg.collect_mapping_grounding_errors(scene, card))
            card, scene = source_and_scene()
            card["steps"][0]["actor_id"] = value
            self.assertTrue(cg.collect_mapping_grounding_errors(scene, card))

    def test_dependency_requires_current_endpoint_mapping_even_for_seed(self):
        card, scene = source_and_scene()
        card["guidance_level"] = "seed"
        scene["card_development_links"] = [scene["card_development_links"][1]]
        self.assertFalse(pc.collect_mapping_errors(scene, card))
        errors = pc.collect_mapping_errors(scene, card, require_grounding=True)
        self.assertIn("card_dependency_endpoint_mapping_missing", [e["code"] for e in errors])

    def test_seed_can_omit_unused_steps_dependencies_and_roles(self):
        card, scene = source_and_scene()
        card["guidance_level"] = "seed"
        scene["card_development_links"] = [scene["card_development_links"][0]]
        scene["card_actor_bindings"] = [scene["card_actor_bindings"][0]]
        self.assertTrue(pc.validate_mapping(scene, card, require_grounding=True))

    def test_step_cannot_borrow_current_action_from_unlinked_development(self):
        card, scene = source_and_scene()
        rows = checks(); rows[0]["evidence_development_indices"] = [2]
        errors = cg.validate_card_grounding_checks(rows, scene, [card])
        self.assertIn("card_plan_check_grounding_evidence_outside_mapping", [e["code"] for e in errors])
        self.assertEqual(rows[0]["verdict"], "compatible")

    def test_positive_subset_is_allowed_negative_empty_is_allowed(self):
        card, scene = source_and_scene()
        scene["card_development_links"][0]["development_indices"] = [1, 3]
        rows = checks()
        self.assertFalse(cg.validate_card_grounding_checks(rows, scene, [card]))
        rows[2]["verdict"] = "revise"; rows[2]["evidence_development_indices"] = []
        rows[0]["verdict"] = "revise"; rows[0]["evidence_development_indices"] = []
        self.assertFalse(cg.validate_card_grounding_checks(rows, scene, [card]))

    def test_compatible_requires_explicit_evidence_and_no_boolean_indices(self):
        card, scene = source_and_scene()
        for evidence in ([], [True], [0], [1, 1], [4]):
            rows = checks(); rows[0]["evidence_development_indices"] = evidence
            self.assertTrue(cg.validate_card_grounding_checks(rows, scene, [card]))

    def test_dependency_cannot_be_true_only_after_endpoint_revision(self):
        card, scene = source_and_scene()
        rows = checks(); rows[0]["verdict"] = "revise"
        frozen = deepcopy(rows)
        errors = cg.validate_card_grounding_checks(rows, scene, [card])
        self.assertIn("card_dependency_positive_with_noncompatible_endpoint", [e["code"] for e in errors])
        self.assertEqual(rows, frozen)

    def test_dependency_current_evidence_must_cover_both_sides(self):
        card, scene = source_and_scene()
        rows = checks(); rows[2]["evidence_development_indices"] = [2]
        errors = cg.validate_card_grounding_checks(rows, scene, [card])
        self.assertIn("card_dependency_current_endpoint_evidence_required", [e["code"] for e in errors])

    def test_actual_s02_cannot_borrow_development_two_for_mapping_one(self):
        root = Path(__file__).resolve().parent.parent / "h4_sequential_ab_v34/runs/inherited_a3_b3_01/requests"
        card = json.loads(json.loads((next(root.glob("002-*")) / "RESPONSE.json").read_text())["text"])["cards"][0]
        scene = json.loads(json.loads((next(root.glob("012-*")) / "RESPONSE.json").read_text())["text"])["scenes"][0]
        # Synthetic binding is structural test data, never a production plan.
        scene["card_actor_bindings"] = [{"actor_id": "actor_1", "participant": "James"}]
        rows = [{"kind": "step", "id": "s1", "verdict": "compatible", "evidence_development_indices": [2]}]
        self.assertIn("card_plan_check_grounding_evidence_outside_mapping", [e["code"] for e in cg.validate_card_grounding_checks(rows, scene, [card])])
        self.assertEqual(cg.expected_card_grounding(scene, [card])["steps"][0]["development_indices"], [1])

    def test_repair_contract_preserves_required_actor_schema(self):
        card, scene = source_and_scene()
        error = pc.MappingValidationError([{"code": "card_actor_bindings_required", "path": "scene.card_actor_bindings", "actual": None}])
        result = pc.build_repair_feedback(error, {"route": pc.ROUTES[1], "scenes": [scene]}, card, require_grounding=True)
        self.assertIn("card_actor_bindings", result["planner_output_schema"]["properties"]["scenes"]["items"]["required"])


if __name__ == "__main__":
    unittest.main()
