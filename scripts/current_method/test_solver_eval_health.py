from __future__ import annotations

import unittest

from scripts.current_method import diagnostics_context_ablation as diag


def evaluated(prediction: str, parsed: str) -> dict:
    return {
        "sample_id": "health-1",
        "qa": {
            "question_type": "single_choice",
            "gold": ["B"],
            "options": [{"id": value, "text": value} for value in "ABCD"],
        },
        "solver_eval": {
            "predictions": [
                {"prediction": prediction, "parsed": parsed, "exact_correct": parsed == "B", "partial_credit": float(parsed == "B")}
            ],
            "diagnostics": {"eval_context_overflow": False},
        },
    }


class SolverHealthTest(unittest.TestCase):
    def test_strict_parser_does_not_mine_label_from_prose(self) -> None:
        self.assertEqual(diag.base.parse_letters("each();", set("ABCD")), [])
        self.assertEqual(diag.base.parse_letters("in and premise", set("ABCD")), [])
        self.assertEqual(diag.base.parse_letters("B. answer text", set("ABCD")), ["B"])
        self.assertEqual(diag.base.parse_letters("C>D>A>B", set("ABCD")), ["C", "D", "A", "B"])

    def test_health_gate_accepts_leading_label(self) -> None:
        health = diag.solver_eval_health([evaluated("B. answer text", "B")], k_eval=1)
        self.assertTrue(health["healthy"])

    def test_health_gate_rejects_gibberish_even_when_parsed(self) -> None:
        health = diag.solver_eval_health([evaluated("in and premise", "A")], k_eval=1)
        self.assertFalse(health["healthy"])
        self.assertEqual(health["single_choice_bad_prefix_count"], 1)


if __name__ == "__main__":
    unittest.main()
