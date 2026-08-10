from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.production_batch_v11 import (
    QA_CANDIDATE_LIMIT,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import (
    hash_tree,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.single_root_mcq_materializer import (
    build_shape_balanced_distractor_prompt,
    compile_factorial_distractors,
)


PILOT33 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "slim_continuous_expansion_v11_end_to_end_batch_pilot_33"
)
PILOT33_SHA = "5b6c4573254b56be10c90951ce7522c02b441aba1e7a3406eb0bc10056260813"
ROOT = PILOT33 / "roots/cc-train-episode-216742/qa_candidates/candidate_01"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class BatchPilot33RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.gold = load(ROOT / "03_semantic_audit/qa/admitted_gold.json")
        cls.view = load(ROOT / "02_questioner/frozen/qa_source_view.json")
        cls.catalog = load(ROOT / "02_questioner/frozen/span_catalog.json")
        cls.raw = json.loads(next(
            (ROOT / "04_mcq_materializer/raw/backend_responses").glob("*.txt")
        ).read_text(encoding="utf-8"))

    def test_historical_length_and_extreme_cues_are_rejected(self):
        result = compile_factorial_distractors(
            self.raw, correct_answer_text=self.gold["correct_answer_text"])
        self.assertEqual("distractor_shape_invalid", result["decision"])
        self.assertTrue(any(
            reason.endswith("_length_mismatch") or
            reason.endswith("_extreme_cue")
            for reason in result["reasons"]))

    def test_prompt_requires_component_parity_and_moderate_wording(self):
        prompt = build_shape_balanced_distractor_prompt(
            frozen_dialogue=self.view["combined_dialogue"],
            span_catalog=self.catalog, base_qa=self.gold)
        self.assertIn("approximate length", prompt)
        self.assertIn("equally moderate and plausible", prompt)

    def test_batch_candidate_pool_is_cost_bounded(self):
        self.assertEqual(3, QA_CANDIDATE_LIMIT)

    def test_pilot33_remains_read_only(self):
        self.assertEqual(PILOT33_SHA, hash_tree(PILOT33)["tree_sha256"])


if __name__ == "__main__":
    unittest.main()
