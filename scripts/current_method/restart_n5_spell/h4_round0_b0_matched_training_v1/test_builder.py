from __future__ import annotations

import unittest

from .builder import assign_slot_targets, build_display, match_at_rate, pair_is_eligible, target_signatures


def _candidate(question_type: str, gold: list[str]) -> dict:
    keys = {
        "single_choice": ("A", "B", "C", "D"),
        "multi_select": ("A", "B", "C", "D", "E"),
        "ordering": ("I1", "I2", "I3", "I4"),
    }[question_type]
    field = "items" if question_type == "ordering" else "options"
    return {
        "candidate_id": f"r:{question_type}", "question_type": question_type,
        "correct_option": gold[0] if question_type == "single_choice" else None,
        "gold": gold,
        field: {key: f"text-{key}" for key in keys},
        "options" if field == "items" else "items": {},
    }


class BuilderTests(unittest.TestCase):
    def test_display_rebinding_preserves_each_question_type(self) -> None:
        cases = (
            (_candidate("single_choice", ["A"]), ("D",)),
            (_candidate("multi_select", ["A", "C", "E"]), ("A", "B", "D")),
            (_candidate("ordering", ["I2", "I1", "I4", "I3"]), ("I4", "I3", "I2", "I1")),
        )
        for candidate, target in cases:
            display = build_display(candidate, desired_target=target, namespace="test")
            rebound = [display["display_to_canonical"][key] for key in target]
            if candidate["question_type"] == "multi_select":
                self.assertEqual(set(rebound), set(candidate["gold"]))
            else:
                self.assertEqual(rebound, candidate["gold"])
            self.assertTrue(display["semantic_gold_preserved"])

    def test_slot_targets_are_stable_and_pair_shared(self) -> None:
        slots = [
            {"slot_id": "shared:a", "question_type": "single_choice", "gold_cardinality": 1},
            {"slot_id": "replacement:p1", "question_type": "single_choice", "gold_cardinality": 1},
            {"slot_id": "replacement:p2", "question_type": "multi_select", "gold_cardinality": 3},
        ]
        first = assign_slot_targets(slots)
        second = assign_slot_targets(list(reversed(slots)))
        self.assertEqual(first, second)
        self.assertIn(first["replacement:p1"], target_signatures("single_choice", 1))
        self.assertIn(first["replacement:p2"], target_signatures("multi_select", 3))

    def test_pair_hard_constraints_and_thresholds(self) -> None:
        base = {
            "candidate_id": "risk", "root_id": "root-a",
            "question_type": "ordering", "gold_cardinality": 4,
            "position_mechanism": "mixed", "full_context_accuracy": 0.5,
            "no_context_accuracy": 0.1, "context_token_count": 4000,
            "training_prompt_token_count": 4200,
        }
        self.assertTrue(pair_is_eligible(base, {**base, "full_context_accuracy": 0.6}))
        self.assertFalse(pair_is_eligible(base, {**base, "full_context_accuracy": 0.8}))
        self.assertFalse(pair_is_eligible(base, {**base, "position_mechanism": "position_only"}))

    def test_infeasible_twenty_percent_can_fall_back_to_ten(self) -> None:
        base = {
            "root_id": "root-a", "question_type": "single_choice",
            "gold_cardinality": 1, "position_mechanism": "position_only",
            "full_context_accuracy": 0.5, "no_context_accuracy": 0.1,
            "context_token_count": 4000, "training_prompt_token_count": 4200,
            "generation_source": "free_exploration",
        }
        cleans = [{**base, "candidate_id": f"clean-{index}"} for index in range(10)]
        risks = [{**base, "candidate_id": "risk-0"}]
        ten = match_at_rate(cleans, risks, rate=0.10)
        twenty = match_at_rate(cleans, risks, rate=0.20)
        self.assertTrue(ten["feasible"])
        self.assertEqual(ten["pair_count"], 1)
        self.assertFalse(twenty["feasible"])
        self.assertEqual(twenty["eligible_pair_capacity"], 1)


if __name__ == "__main__":
    unittest.main()
