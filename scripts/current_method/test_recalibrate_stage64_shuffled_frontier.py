from __future__ import annotations

import copy
import unittest

from scripts.current_method import recalibrate_stage64_shuffled_frontier as recal
from scripts.current_method import stage3_operation_controller as controller


def sample(sample_id: str = "s1", *, parent: str = "conv-1") -> dict:
    return {
        "sample_id": sample_id,
        "source_id": f"source-{sample_id}",
        "frontier_from_split": "train",
        "seed_source": {
            "allowed_train_seed": True,
            "split": "train",
            "parent_sample_id": parent,
            "parent_chunk_id": f"{parent}-chunk",
        },
        "attack_plan": {"attack_skill": "comparison"},
        "qa": {
            "question_type": "single_choice",
            "question": "Which option is correct?",
            "gold": ["B"],
            "options": [
                {"id": "A", "text": "alpha", "is_gold": False},
                {"id": "B", "text": "bravo", "is_gold": True},
                {"id": "C", "text": "charlie", "is_gold": False},
                {"id": "D", "text": "delta", "is_gold": False},
            ],
        },
        "validation": {
            "option_valid": True,
            "gold_derivable": True,
            "graph_dialogue_consistent": True,
            "anti_shortcut_pass": True,
            "warnings": [],
        },
    }


def attach_eval(row: dict, correct: int, *, k: int = 5) -> None:
    row["solver_eval"] = {
        "k": k,
        "predictions": [
            {"prediction": "A", "parsed": "A", "exact_correct": index < correct, "partial_credit": float(index < correct)}
            for index in range(k)
        ],
        "exact_correct_count": correct,
        "soft_accuracy": correct / k,
        "partial_credit": correct / k,
        "bucket": f"{correct}/{k}",
        "diagnostics": {},
    }


def attach_reasoning_target(row: dict, *, operation: str = "bridge") -> None:
    reasoning = {
        "reasoning_subtype": "conjunctive_two_support_bridge",
        "reasoning_template": "join_two_support_facts",
        "hop_count": 2,
        "hard_negative_strategy": "one_hop_preserving_alternative",
    }
    row.update({"qa_operation": operation, **reasoning})
    row["qa"].update({"qa_operation": operation, **reasoning})
    row["qa"]["query_plan"] = {"qa_operation": operation, **reasoning}
    row["query_plan"] = {"qa_operation": operation, **reasoning}
    row["stage64_target"] = {"qa_operation": operation, **reasoning}
    row["stage64_target_identity"] = "formal-v2-signature-test"
    row["stage3_generation_protocol"] = controller.generation_protocol()


class Stage64RecalibrationTest(unittest.TestCase):
    def test_exact_duplicate_and_gold_ambiguity_are_rejected(self) -> None:
        row = sample()
        row["qa"]["options"][3]["text"] = "bravo"
        reasons = {item["reason"] for item in recal.option_quality_issues(row)}
        self.assertIn("exact_duplicate_option_text", reasons)
        self.assertIn("gold_text_maps_to_multiple_labels", reasons)

    def test_cyclic_permutations_put_gold_in_every_position(self) -> None:
        row = sample()
        variants, grouped = recal.build_permutation_rows([row], mode="cyclic")
        self.assertEqual(len(variants), 4)
        self.assertEqual({variant["qa"]["gold"][0] for variant in variants}, {"A", "B", "C", "D"})
        for variant in variants:
            gold = variant["qa"]["gold"][0]
            gold_text = next(item["text"] for item in variant["qa"]["options"] if item["id"] == gold)
            self.assertEqual(gold_text, "bravo")
        self.assertEqual(len(grouped["s1"]), 4)

    def test_only_all_frontier_permutations_become_robust_frontier(self) -> None:
        rows = [sample("frontier"), sample("sensitive"), sample("easy")]
        _, grouped = recal.build_permutation_rows(rows, mode="cyclic")
        for value in grouped["frontier"]:
            attach_eval(value, 2)
        for index, value in enumerate(grouped["sensitive"]):
            attach_eval(value, 5 if index == 0 else 2)
        for value in grouped["easy"]:
            attach_eval(value, 5)
        summary = recal.aggregate_permutation_evaluations(
            rows,
            grouped,
            k_eval=5,
            min_frontier_permutation_share=1.0,
        )
        self.assertEqual(rows[0]["stage64_permutation_eval"]["status"], "robust_frontier")
        self.assertEqual(rows[1]["solver_eval"]["bucket"], "position_sensitive")
        self.assertEqual(rows[2]["stage64_permutation_eval"]["status"], "robust_easy5")
        self.assertEqual(summary["status_counts"], {"robust_frontier": 1, "position_sensitive": 1, "robust_easy5": 1})
        self.assertEqual(summary["reasoning_signature_typed_count"], 0)
        self.assertEqual(summary["reasoning_signature_legacy_untyped_count"], 3)
        self.assertEqual(summary["reasoning_signature_distribution"], {"legacy_untyped": 3})

    def test_reasoning_signature_metrics_are_target_defined_and_rollout_weighted(self) -> None:
        rows = [sample("frontier-1"), sample("frontier-2")]
        for row in rows:
            attach_reasoning_target(row)
        _, grouped = recal.build_permutation_rows(rows, mode="cyclic")
        for value in grouped["frontier-1"]:
            attach_eval(value, 1)
        for value in grouped["frontier-2"]:
            attach_eval(value, 3)
        summary = recal.aggregate_permutation_evaluations(
            rows,
            grouped,
            k_eval=5,
            min_frontier_permutation_share=1.0,
        )
        self.assertEqual(summary["reasoning_signature_typed_count"], 2)
        self.assertEqual(summary["reasoning_signature_legacy_untyped_count"], 0)
        self.assertEqual(len(summary["reasoning_signature_metrics"]), 1)
        metric = next(iter(summary["reasoning_signature_metrics"].values()))
        self.assertEqual(metric["total"], 2)
        self.assertEqual(metric["status_counts"]["robust_frontier"], 2)
        self.assertEqual(metric["exact_correct_rollout_count"], 16)
        self.assertEqual(metric["evaluated_rollout_count"], 40)
        self.assertEqual(metric["weighted_rollout_exact_accuracy"], 0.4)
        self.assertEqual(metric["frontier_yield"], 1.0)

        counts = recal.count_summary(rows, rows, [], [])
        self.assertEqual(counts["reasoning_signature_typed_count"], 2)
        self.assertEqual(
            counts["reasoning_signature_distribution"],
            summary["reasoning_signature_distribution"],
        )
        self.assertEqual(
            counts["reasoning_signature_metrics"],
            summary["reasoning_signature_metrics"],
        )

    def test_parent_sample_grouping_prevents_chunk_level_leakage(self) -> None:
        rows = [sample(f"s{index}", parent="conv-shared" if index < 2 else f"conv-{index}") for index in range(5)]
        for row in rows:
            attach_eval(row, 2)
        train, heldout, _ = recal.split_frontier_by_group(
            copy.deepcopy(rows),
            seed=7,
            heldout_ratio=0.4,
            group_level="parent_sample_id",
        )
        train_groups = {recal.group_key(row, level="parent_sample_id") for row in train}
        heldout_groups = {recal.group_key(row, level="parent_sample_id") for row in heldout}
        self.assertFalse(train_groups & heldout_groups)

    def test_zero_heldout_ratio_keeps_batch_frontier_as_pool(self) -> None:
        rows = [sample(f"s{index}", parent=f"conv-{index}") for index in range(3)]
        for row in rows:
            attach_eval(row, 2)
        train, heldout, summary = recal.split_frontier_by_group(
            copy.deepcopy(rows),
            seed=7,
            heldout_ratio=0.0,
            group_level="parent_sample_id",
        )
        self.assertEqual(len(train), 3)
        self.assertEqual(heldout, [])
        self.assertEqual(summary["heldout_groups"], 0)


if __name__ == "__main__":
    unittest.main()
