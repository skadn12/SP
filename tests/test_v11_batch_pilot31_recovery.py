from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import (
    hash_tree,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.single_root_mcq_materializer import (
    build_shape_balanced_distractor_prompt,
    decompose_cross_boundary_answer,
    preflight,
)


PILOT31 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "slim_continuous_expansion_v11_end_to_end_batch_pilot_31"
)
PILOT31_SHA = "4c4e36478050f4508577aa89c905d3bd969531ad40e3ab8cb4dcc1ae597aa798"
API_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class BatchPilot31RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mcq_source = PILOT31 / (
            "roots/cc-train-episode-41625/qa_candidates/candidate_01")
        cls.gold = load(cls.mcq_source / "03_semantic_audit/qa/admitted_gold.json")
        cls.view = load(cls.mcq_source / "02_questioner/frozen/qa_source_view.json")
        cls.catalog = load(cls.mcq_source / "02_questioner/frozen/span_catalog.json")

    def test_later_comma_gold_is_decomposable(self) -> None:
        parts = decompose_cross_boundary_answer(self.gold["correct_answer_text"])
        self.assertIn("Later", parts["connector"])
        self.assertTrue(parts["parent_component"])
        self.assertTrue(parts["child_component"])

    def test_later_comma_prompt_preflight_is_zero_api_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = preflight(
                semantic_audit_run=self.mcq_source / "03_semantic_audit",
                qa_source_run=self.mcq_source / "02_questioner",
                run_root=Path(tmp) / "mcq",
                api_config_path=API_CONFIG, tokenizer_path=TOKENIZER,
                root_id="cc-train-episode-41625")
        self.assertTrue(report["valid"], report["failed_checks"])
        self.assertEqual(0, report["api_calls"])
        self.assertTrue(report["checks"]["factorial_answer_decomposable"])

    def test_prompt_uses_the_actual_later_comma_gold_components(self) -> None:
        prompt = build_shape_balanced_distractor_prompt(
            frozen_dialogue=self.view["combined_dialogue"],
            span_catalog=self.catalog, base_qa=self.gold)
        self.assertIn("FACTORIAL COUNTERFACTUAL COMPONENT CONTRACT", prompt)
        self.assertIn("counterfactual_parent_component", prompt)
        self.assertIn("counterfactual_child_component", prompt)

    def test_pilot31_is_unchanged(self) -> None:
        self.assertEqual(PILOT31_SHA, hash_tree(PILOT31)["tree_sha256"])


if __name__ == "__main__":
    unittest.main()
