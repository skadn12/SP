import unittest

from scripts.current_method.restart_n5_spell.semantic_neighbor_distractor_canary import (
    SEMANTIC_NEIGHBOR_PROMPT_VERSION,
    build_semantic_neighbor_prompt,
    deterministic_semantic_neighbor_placement,
    semantic_neighbor_prompt_sha256,
    semantic_neighbor_schema_sha256,
)


class SemanticNeighborDistractorCanaryTests(unittest.TestCase):
    def setUp(self):
        self.qa = {
            "qa_id": "johns_motivation_for_politics",
            "question": "What reinforced John's commitment?",
            "correct_answer_text": "His participation in community meetings.",
            "answer_support_span_ids": ["S1"],
        }

    def test_prompt_freezes_question_and_uses_fixed_diagnosis(self):
        prompt = build_semantic_neighbor_prompt(
            frozen_dialogue=[{"turn_id": "T1", "text": "Text"}],
            base_qa=self.qa,
            previous_distractors=["old one", "old two", "old three"],
        )
        self.assertIn(SEMANTIC_NEIGHBOR_PROMPT_VERSION, prompt)
        self.assertIn("bad_ground_semantic_leakage", prompt)
        self.assertIn(self.qa["question"], prompt)
        self.assertIn("changing exactly one decisive slot", prompt)
        self.assertIn("do not need to be facts stated elsewhere", prompt)

    def test_prompt_requires_three_previous_distractors(self):
        with self.assertRaisesRegex(ValueError, "exactly_three"):
            build_semantic_neighbor_prompt(
                frozen_dialogue=[], base_qa=self.qa,
                previous_distractors=["one", "two"],
            )

    def test_placement_is_deterministic_and_content_independent(self):
        first = deterministic_semantic_neighbor_placement(self.qa["qa_id"])
        second = deterministic_semantic_neighbor_placement(self.qa["qa_id"])
        self.assertEqual(first, second)
        self.assertEqual(first["target_letter"], "A")
        self.assertFalse(first["content_dependent"])

    def test_hashes_are_stable_shape(self):
        self.assertRegex(semantic_neighbor_prompt_sha256(), r"^[0-9a-f]{64}$")
        self.assertRegex(semantic_neighbor_schema_sha256(), r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
