from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import (
    hash_tree,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.single_root_mcq_materializer import (
    build_shape_balanced_distractor_prompt,
    validate_factorial_components_with_shape_gate,
)
from scripts.current_method.restart_n5_spell.single_item_qa_canary import (
    deterministic_answer_placement,
)


PILOT32 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "slim_continuous_expansion_v11_end_to_end_batch_pilot_32"
)
PILOT32_SHA = "36ab9d491e8276ab6cad9bd9c5fe4fb36048370b286ee426ad5e1fb8502480ef"
ROOT = PILOT32 / "roots/cc-train-episode-41625/qa_candidates/candidate_01"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class BatchPilot32RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.gold = load(ROOT / "03_semantic_audit/qa/admitted_gold.json")
        cls.catalog = load(ROOT / "02_questioner/frozen/span_catalog.json")
        cls.view = load(ROOT / "02_questioner/frozen/qa_source_view.json")
        cls.raw = json.loads(next(
            (ROOT / "04_mcq_materializer/raw/backend_responses").glob("*.txt")
        ).read_text(encoding="utf-8"))

    def validate(self, raw):
        return validate_factorial_components_with_shape_gate(
            raw, core_qa=self.gold, span_catalog=self.catalog,
            placement=deterministic_answer_placement(
                "cc-train-episode-41625-round-1", 0))

    def test_near_synonym_counterfactual_is_rejected_as_local_mutation(self):
        result = self.validate(self.raw)
        self.assertEqual("distractor_shape_invalid", result["decision"])
        self.assertIn("factorial_component_small_local_mutation",
                      result["reasons"])

    def test_prompt_requires_mutually_incompatible_hard_negatives(self):
        prompt = build_shape_balanced_distractor_prompt(
            frozen_dialogue=self.view["combined_dialogue"],
            span_catalog=self.catalog, base_qa=self.gold)
        self.assertIn("mutually incompatible", prompt)
        self.assertIn("different speaker, episode, time, or outcome", prompt)
        self.assertIn("one-slot substitution", prompt)

    def test_retry_does_not_weaken_to_synonym_or_short_slot_swap(self):
        result = self.validate(self.raw)
        self.assertTrue(result["factorial_component_local_mutations"])
        self.assertTrue(all(
            item["small_local_mutation"]
            for item in result["factorial_component_local_mutations"]))

    def test_pilot32_remains_read_only(self):
        self.assertEqual(PILOT32_SHA, hash_tree(PILOT32)["tree_sha256"])


if __name__ == "__main__":
    unittest.main()
