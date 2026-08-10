from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.single_item_qa_canary import (
    deterministic_answer_placement,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.continuous_expansion_qa import (
    validate_cross_boundary_core_object,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import (
    hash_tree,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.single_root_mcq_materializer import (
    build_shape_balanced_distractor_prompt,
    compile_factorial_distractors,
    decompose_cross_boundary_answer,
    validate_factorial_components_with_shape_gate,
)


PILOT29 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "slim_continuous_expansion_v11_end_to_end_batch_pilot_29"
)
PILOT29_SHA = "fb3269edf020beee6ef1258796d6c98f7fa7f7ad55f6617355a6a38663b48fc7"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class BatchPilot29RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root1 = PILOT29 / "roots/cc-train-episode-216742/qa_candidates/candidate_01"
        cls.root2 = PILOT29 / "roots/cc-train-episode-41625/qa_candidates/candidate_01"

    def test_historical_unsupported_causal_question_is_now_rejected(self) -> None:
        qa_run = self.root1 / "02_questioner"
        summary = load(qa_run / "reports/QA_CANARY_SUMMARY.json")
        raw = summary["questioner_result"]["accepted_raw"]
        view = load(qa_run / "frozen/qa_source_view.json")
        catalog = load(qa_run / "frozen/span_catalog.json")
        frozen = load(qa_run / "run_freeze_manifest.json")[
            "frozen_support_pair"]
        result = validate_cross_boundary_core_object(
            raw, span_catalog=catalog,
            partition_by_turn=view["partition_by_turn"],
            lineage_pair_registry=view["lineage_pair_registry"],
            frozen_support_pair=frozen, evidence_mismatch_policy="mark_only")
        self.assertEqual("cross_boundary_qa_contract_reject", result["decision"])
        self.assertIn("cross_boundary_qa_unsupported_causal_question_form",
                      result["reasons"])

    def test_a_descriptive_comparison_question_keeps_the_same_frozen_pair(self) -> None:
        qa_run = self.root1 / "02_questioner"
        summary = load(qa_run / "reports/QA_CANARY_SUMMARY.json")
        raw = dict(summary["questioner_result"]["accepted_raw"])
        raw["question"] = (
            "How did the speaker's handling of personal responsibilities differ "
            "between the earlier and later conversations?")
        view = load(qa_run / "frozen/qa_source_view.json")
        catalog = load(qa_run / "frozen/span_catalog.json")
        frozen = load(qa_run / "run_freeze_manifest.json")[
            "frozen_support_pair"]
        result = validate_cross_boundary_core_object(
            raw, span_catalog=catalog,
            partition_by_turn=view["partition_by_turn"],
            lineage_pair_registry=view["lineage_pair_registry"],
            frozen_support_pair=frozen,
            evidence_mismatch_policy="mark_only")
        self.assertEqual("cross_boundary_qa_accept", result["decision"],
                         result.get("reasons"))

    def test_both_real_gold_forms_are_decomposable(self) -> None:
        for root in (self.root1, self.root2):
            gold = load(root / "03_semantic_audit/qa/admitted_gold.json")
            parts = decompose_cross_boundary_answer(gold["correct_answer_text"])
            rebuilt = (parts["parent_component"] + parts["connector"] +
                       parts["child_component"])
            self.assertEqual(gold["correct_answer_text"], rebuilt)

    def test_old_three_distractor_response_is_rejected_by_new_schema(self) -> None:
        old = {"distractors": ["one", "two", "three"]}
        gold = load(self.root2 / "03_semantic_audit/qa/admitted_gold.json")
        result = compile_factorial_distractors(
            old, correct_answer_text=gold["correct_answer_text"])
        self.assertEqual("distractor_schema_invalid", result["decision"])

    def test_factorial_components_compile_and_pass_the_existing_shape_gate(self) -> None:
        gold = load(self.root2 / "03_semantic_audit/qa/admitted_gold.json")
        catalog = load(self.root2 / "02_questioner/frozen/span_catalog.json")
        raw = {
            "counterfactual_parent_component": (
                "Classmates B initially doubted that coordinated community action "
                "could make a meaningful difference"),
            "counterfactual_child_component": (
                "rejected the offered support and remained doubtful about "
                "collaboration."),
        }
        result = validate_factorial_components_with_shape_gate(
            raw, core_qa=gold, span_catalog=catalog,
            placement=deterministic_answer_placement(
                "cc-train-episode-41625-round-1", 0))
        self.assertEqual("distractor_accept", result["decision"],
                         result.get("reasons"))
        self.assertFalse(result["option_shape"]["answer_shape_leakage_risk"])

    def test_prompt_requests_components_and_places_schema_last(self) -> None:
        gold = load(self.root2 / "03_semantic_audit/qa/admitted_gold.json")
        view = load(self.root2 / "02_questioner/frozen/qa_source_view.json")
        catalog = load(self.root2 / "02_questioner/frozen/span_catalog.json")
        prompt = build_shape_balanced_distractor_prompt(
            frozen_dialogue=view["combined_dialogue"], span_catalog=catalog,
            base_qa=gold)
        self.assertNotIn('OUTPUT ONLY {"distractors"', prompt)
        self.assertTrue(prompt.endswith(
            'OUTPUT ONLY {"counterfactual_parent_component":"...",'
            '"counterfactual_child_component":"..."}'))

    def test_pilot29_is_unchanged(self) -> None:
        self.assertEqual(PILOT29_SHA, hash_tree(PILOT29)["tree_sha256"])


if __name__ == "__main__":
    unittest.main()
