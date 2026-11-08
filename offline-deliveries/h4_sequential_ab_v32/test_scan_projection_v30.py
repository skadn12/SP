"""The history scanner sees proposed work and timing, not planner verdict labels."""
from copy import deepcopy
import json
import unittest

import event_state as scan
import test_event_v19 as fixtures


class ScanProjectionV30Tests(unittest.TestCase):
    def setUp(self):
        self.scene = {
            "scene_id": "S02", "actors": ["Emily", "James"],
            "goal": "Resolve a delayed beverage delivery for the dinner.",
            "developments": ["Emily describes the late delivery.", "They select a substitute."],
            "source_anchors": [{"turn_ref": "R02:B:S01:T004", "quote": "I am planning a family dinner."}],
            "dataset_card_id": "card-2", "card_development_links": [
                {"card_step_ids": ["s1"], "dependency_indices": [], "development_indices": [1, 2]}],
            "feedback_topic_ids": ["feedback-not-scanner-history"],
            "event_definition": {"trigger": "A delivery will miss the dinner.",
                                 "history_relation": "distinct_event"},
            "state_transition": {"basis_fact_ids": [],
                "basis_reason": "The old dinner does not exist in history.",
                "new_event": "This is a separate new dinner event.",
                "temporal_placement": "During preparation of that same family dinner."},
        }

    def test_projection_preserves_actual_work_trigger_and_placement(self):
        before = deepcopy(self.scene)
        projected = scan.scan_plan_context(self.scene)
        self.assertEqual(projected["event_definition"], {"trigger": self.scene["event_definition"]["trigger"]})
        self.assertEqual(projected["state_transition"], {"basis_fact_ids": [],
            "temporal_placement": self.scene["state_transition"]["temporal_placement"]})
        for key in ("scene_id", "actors", "goal", "developments", "source_anchors",
                    "dataset_card_id", "card_development_links"):
            self.assertEqual(projected[key], self.scene[key])
        self.assertNotIn("feedback_topic_ids", projected)
        self.assertEqual(self.scene, before)
        projected["actors"].append("Someone else")
        self.assertEqual(self.scene, before)

    def test_projection_is_idempotent_for_full_and_partial_inputs(self):
        for scene in (self.scene, {"scene_id": "S01", "developments": ["Do work"]},
                      {"event_definition": {"history_relation": "distinct_event"},
                       "state_transition": {"new_event": "Positive claim"}}):
            with self.subTest(scene=scene):
                once = scan.scan_plan_context(scene)
                self.assertEqual(scan.scan_plan_context(once), once)

    def test_planner_positive_claim_changes_cannot_change_projection(self):
        changed = deepcopy(self.scene)
        changed["event_definition"]["history_relation"] = "continuation"
        changed["state_transition"]["basis_reason"] = "An entirely different route assertion."
        changed["state_transition"]["new_event"] = "Different identity claim."
        self.assertEqual(scan.scan_plan_context(self.scene), scan.scan_plan_context(changed))
        changed["state_transition"]["temporal_placement"] = "A different later occasion."
        self.assertNotEqual(scan.scan_plan_context(self.scene), scan.scan_plan_context(changed))

    def test_build_request_uses_projection_without_mutating_scene(self):
        f = fixtures.EventV19Tests()
        f.setUp()
        scene = deepcopy(f.scene)
        scene.update({key: deepcopy(value) for key, value in self.scene.items()
                      if key in ("actors", "state_transition", "source_anchors")})
        before = deepcopy(scene)
        request = scan.build_request(f.root, f.route, scene, f.history, [], {})
        data = json.loads(request["messages"][0]["content"].split("EVENT_STATE_INPUT_JSON\n", 1)[1])
        self.assertEqual(data["scene_plan"], scan.scan_plan_context(scene))
        self.assertEqual(scene, before)
        self.assertEqual(data["expected_development_indices"], [1, 2])

    def test_non_object_input_rejected(self):
        with self.assertRaisesRegex(ValueError, "event_state_scan_plan_not_object"):
            scan.scan_plan_context([])


if __name__ == "__main__":
    unittest.main()
