from __future__ import annotations

import unittest

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.continuous_expansion_mcq import (
    build_full_context_mcq_solver_prompt,
    validate_mcq_solver_output,
)


class ContinuousExpansionMCQTest(unittest.TestCase):
    def test_solver_prompt_has_no_reference_answer(self) -> None:
        prompt = build_full_context_mcq_solver_prompt(
            frozen_dialogue=[{"turn_id": "T1", "session_id": "S1",
                              "speaker": "A", "text": "fact"}],
            materialized_qa={"question": "Q", "options": {
                "A": "a", "B": "b", "C": "c", "D": "d"},
                "reference_answer_letter": "C"})
        self.assertIn('"options"', prompt)
        self.assertNotIn("reference_answer_letter", prompt)

    def test_solver_schema(self) -> None:
        self.assertEqual("mcq_solver_contract_accept",
            validate_mcq_solver_output({"answer_letter": "B"})["decision"])
        self.assertEqual("mcq_solver_schema_invalid",
            validate_mcq_solver_output({"answer_letter": "E"})["decision"])
        self.assertEqual("mcq_solver_schema_invalid",
            validate_mcq_solver_output({"answer_letter": "B", "reason": "x"})[
                "decision"])


if __name__ == "__main__":
    unittest.main()
