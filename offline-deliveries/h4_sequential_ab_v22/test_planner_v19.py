"""Offline route-specific planner validation; no provider or persistent run."""
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import runner
import source_inventory


class PlannerV19Tests(unittest.TestCase):
    def setUp(self):
        self.root = "fixture-root"
        self.history = [
            {"turn_ref": "P:T001", "speaker": "Emily", "text": "I will attend an interview."},
            {"turn_ref": "R02:A:S01:T001", "speaker": "James", "text": "Your interview is finished."},
        ]
        self.card = {"card_id": "new-card", "steps": [
            {"step_id": "s1", "actor_id": "p1", "action": "request a table", "state": "proposed"},
            {"step_id": "s2", "actor_id": "p2", "action": "confirm booking", "state": "reported_action"},
        ], "dependencies": []}

    def fixture(self, route):
        new = route == runner.ROUTES[1]
        goal = "Arrange a new dinner." if new else "Discuss the interview outcome."
        scene = {"scene_id": "S01", "actors": ["Emily", "James"],
            "source_anchors": [] if new else [{"turn_ref": "P:T001", "quote": "I will attend an interview."}],
            "goal": goal, "developments": ["Choose a time.", "Confirm the next action."],
            "dataset_card_id": "new-card" if new else None,
            "card_development_links": [{"card_step_ids": ["s1", "s2"], "dependency_indices": [],
                                        "development_indices": [1, 2]}] if new else [],
            "feedback_topic_ids": [],
            "event_definition": {"trigger": "A new dinner invitation." if new else "A reported interview outcome.",
                                 "main_goal": goal, "history_relation": "distinct_event" if new else "continuation"},
            "state_transition": {"basis_fact_ids": [] if new else ["f1"],
                                 "new_event": goal, "temporal_placement": "After the existing conversation."}}
        if new:
            scene["history_context_reason"] = "No historical event is needed as the dinner's premise."
            scene["state_transition"]["basis_reason"] = "The new card defines the dinner; no specific fact is a premise."
        state = {"route": route, "facts": [{"fact_id": "f1"}]}
        plan = {"root_id": self.root, "route": route, "scenes": [scene], "shortfall_reason": ""}
        return plan, state

    def validate(self, plan, state):
        return runner.validate_plan(plan, self.root, plan["route"], self.history,
            [self.card] if plan["route"] == runner.ROUTES[1] else [], "S01", latest_state=state)

    def test_b_without_old_event_anchors_or_basis_passes_without_mutation(self):
        plan, state = self.fixture(runner.ROUTES[1])
        original = deepcopy(plan)
        with patch.object(runner.base, "validate_plan", side_effect=AssertionError("legacy anchor validator called")):
            self.assertEqual(self.validate(plan, state), original)
        self.assertEqual(plan, original)

    def test_a_preserves_required_anchor_and_basis(self):
        plan, state = self.fixture(runner.ROUTES[0])
        self.validate(plan, state)
        plan["scenes"][0]["source_anchors"] = []
        with self.assertRaisesRegex(ValueError, "scene_anchors_missing"):
            self.validate(plan, state)
        plan, state = self.fixture(runner.ROUTES[0])
        plan["scenes"][0]["state_transition"]["basis_fact_ids"] = []
        with self.assertRaisesRegex(ValueError, "state_transition_fact_ids_invalid"):
            self.validate(plan, state)

    def test_b_empty_references_require_explicit_reasons(self):
        for key in ("history_context_reason", "basis_reason"):
            plan, state = self.fixture(runner.ROUTES[1])
            target = plan["scenes"][0] if key == "history_context_reason" else plan["scenes"][0]["state_transition"]
            target[key] = ""
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(plan, state)

    def test_b_context_only_accepted_a_anchor_allowed(self):
        plan, state = self.fixture(runner.ROUTES[1])
        plan["scenes"][0]["source_anchors"] = [{"turn_ref": "R02:A:S01:T001", "quote": "Your interview is finished."}]
        plan["scenes"][0]["state_transition"]["basis_fact_ids"] = ["f1"]
        self.validate(plan, state)

    def test_nonempty_b_references_must_be_real(self):
        plan, state = self.fixture(runner.ROUTES[1])
        plan["scenes"][0]["state_transition"]["basis_fact_ids"] = ["invented"]
        with self.assertRaisesRegex(ValueError, "state_transition_fact_ids_invalid"):
            self.validate(plan, state)
        plan, state = self.fixture(runner.ROUTES[1])
        plan["scenes"][0]["source_anchors"] = [{"turn_ref": "P:T001", "quote": "Invented evidence"}]
        with self.assertRaisesRegex(ValueError, "anchor_quote_not_exact"):
            self.validate(plan, state)

    def test_card_mapping_is_not_relaxed_by_empty_history_references(self):
        plan, state = self.fixture(runner.ROUTES[1])
        plan["scenes"][0]["card_development_links"][0]["card_step_ids"] = ["s1"]
        with self.assertRaises(ValueError):
            self.validate(plan, state)

    def test_event_definition_and_goal_have_one_authority(self):
        plan, state = self.fixture(runner.ROUTES[1])
        plan["scenes"][0]["event_definition"]["main_goal"] = "Continue the old interview."
        with self.assertRaisesRegex(ValueError, "event_definition_goal_or_trigger_invalid"):
            self.validate(plan, state)

    def test_no_scene_requires_explicit_shortfall(self):
        plan, state = self.fixture(runner.ROUTES[1])
        plan["scenes"] = []
        with self.assertRaisesRegex(ValueError, "planner_shortfall_reason_inconsistent"):
            self.validate(plan, state)
        plan["shortfall_reason"] = "No coherent new event fits the card."
        self.validate(plan, state)

    def test_v18_through_v22_reservations_are_consumed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = {name: root / name for name in (
                "v1_runs", "v2_runs", "v4_runs", "v5_runs", "v6_runs", "v7_runs", "v8_runs",
                "v9_runs", "v11_runs", "v12_runs", "v15_runs", "v18_runs", "v19_runs", "v20_runs", "v21_runs")}
            current = root / "v22"
            item = {"dataset": "fixture", "split": "train", "item_id": "old", "source_id": "fixture:old",
                    "source_item_sha256": "a" * 64}
            source_inventory.reserve(args["v18_runs"] / "prior", {"r": [item]},
                                     source_inventory.collect_prior(current, **args))
            middle = dict(item, item_id="middle", source_id="fixture:middle")
            source_inventory.reserve(args["v19_runs"] / "prior", {"r": [middle]},
                                     source_inventory.collect_prior(current, **args))
            recent = dict(item, item_id="recent", source_id="fixture:recent")
            source_inventory.reserve(args["v20_runs"] / "prior", {"r": [recent]},
                                     source_inventory.collect_prior(current, **args))
            latest = dict(item, item_id="latest", source_id="fixture:latest")
            source_inventory.reserve(args["v21_runs"] / "prior", {"r": [latest]},
                                     source_inventory.collect_prior(current, **args))
            prior = source_inventory.collect_prior(current, **args)
            with self.assertRaisesRegex(ValueError, "source_event_already_consumed"):
                source_inventory.validate_available({"r": [item]}, prior)
            new = dict(item, item_id="new", source_id="fixture:new")
            source_inventory.reserve(current / "next", {"r": [new]}, prior)
            all_used = source_inventory.collect_prior(current, **args)
            self.assertEqual({row["origin"] for row in all_used["accepted_uses"]},
                             {"v18_reservation", "v19_reservation", "v20_reservation", "v21_reservation", "v22_reservation"})


if __name__ == "__main__":
    unittest.main()
