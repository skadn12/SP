from __future__ import annotations

import copy
import json
import io
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from scripts.current_method import accumulate_stage64_reliable_frontier as subject
from scripts.current_method import recalibrate_stage64_shuffled_frontier as recal


def robust_row(sample_id: str = "sample-1", *, parent: str = "conversation-1", arity: int = 4) -> dict:
    values = ("Red notebook", "Blue binder", "Ceramic mug", "Brass key")
    events = [
        {
            "event_instance_id": f"event-{index}",
            "entity": "Noah" if index == 3 else "Maya",
            "slot": "item_beside_lamp",
            "value": values[index - 1],
            "status": "active" if index == 1 else "outdated",
        }
        for index in range(1, arity + 1)
    ]
    row = {
        "sample_id": sample_id,
        "lineage_group": parent,
        "seed_source": {"parent_sample_id": parent},
        "dialogue": {
            "turns": [
                {"speaker": "Maya", "text": "I left the red notebook beside the lamp."},
                {"speaker": "Noah", "text": "I remember where you put it."},
            ]
        },
        "graph_update": {"events": events},
        "qa": {
            "question_type": "single_choice",
            "question": "What item did Maya leave beside the lamp?",
            "qa_operation": "relation_lookup",
            "qa_subtype": "object_location_lookup",
            "answer_type": "ObjectOrCommonNoun",
            "gold": ["A"],
            "options": [
                {"id": label, "text": text, "source_event_instance_id": f"event-{index}"}
                for index, (label, text) in enumerate(
                    zip("ABCD"[:arity], values[:arity]),
                    start=1,
                )
            ],
            "query_plan": {
                "reasoning_anchor_path": ["event-1", "event-2", "event-3"]
            },
        },
        "stage64_permutation_eval": {
            "status": "robust_frontier",
            "k_eval_per_permutation": 5,
            "permutation_count": arity,
            "permutations": [],
        },
    }
    for shift in range(arity):
        count = (shift % 4) + 1
        variant = recal.cyclic_option_permutation(row, shift=shift)
        row["stage64_permutation_eval"]["permutations"].append(
            {
                "shift": shift,
                "gold": recal.qa_gold(variant),
                "exact_correct_count": count,
                "bucket": f"{count}/5",
                "soft_accuracy": count / 5,
                "partial_credit": count / 5,
                "predictions": [
                    {"prediction": recal.qa_gold(variant), "exact_correct": index < count}
                    for index in range(5)
                ],
            }
        )
    return row


def batch_summary(*, frontier_count: int = 1, generation_seed: int | None = None, eval_seed: int = 7) -> dict:
    return {
        "k_eval": 5,
        "max_model_len": 32768,
        "max_new_tokens": 80,
        "temperature": 0.7,
        "top_p": 0.9,
        "seed": eval_seed,
        "generation_seed": generation_seed,
        "permutation_mode": "cyclic",
        "min_frontier_permutation_share": 1.0,
        "quality": {"semantic_quality_gate_enabled": True},
        "decoding_health": {"healthy": True},
        "loaded_solver_model_resolved": "/model/stage2",
        "accepted_stage2_solver_resolved": "/model/stage2",
        "frontier_count": frontier_count,
    }


class AccumulateStage64FrontierTest(unittest.TestCase):
    def test_lineage_group_takes_precedence_over_stage2_parent_sample(self) -> None:
        rows = []
        for index, lineage in enumerate(("root-a", "root-a", "root-b", "root-c")):
            row = robust_row(f"sample-{index}", parent=f"stage2-parent-{index}")
            row["lineage_group"] = lineage
            row["metadata"] = {"lineage_group": lineage}
            rows.append(row)
        train, heldout, summary = recal.split_frontier_by_group(
            rows,
            seed=19,
            heldout_ratio=0.25,
            group_level="lineage_group",
        )
        self.assertEqual(summary["group_level"], "lineage_group")
        self.assertFalse(
            {subject.parent_sample_id(row) for row in train}
            & {subject.parent_sample_id(row) for row in heldout}
        )
        root_a_partitions = {
            "train" if row in train else "heldout"
            for row in rows
            if row["lineage_group"] == "root-a"
        }
        self.assertEqual(len(root_a_partitions), 1)

    def _write_batch(
        self,
        directory: Path,
        *,
        rows: list[dict] | None = None,
        generation_seed: int | None = None,
        eval_seed: int = 7,
    ) -> None:
        rows = rows or [robust_row()]
        recal.write_json(
            directory / "recalibration_summary.json",
            batch_summary(frontier_count=len(rows), generation_seed=generation_seed, eval_seed=eval_seed),
        )
        recal.write_jsonl(directory / "frontier.jsonl", rows)

    def _run(self, batch_dirs: list[Path], output: Path, *, min_frontier: int, seed: int = 17) -> dict:
        argv = ["accumulate"]
        for batch in batch_dirs:
            argv.extend(("--recalibration-dir", str(batch)))
        argv.extend(
            (
                "--output-dir",
                str(output),
                "--min-frontier",
                str(min_frontier),
                "--heldout-ratio",
                "0.25",
                "--seed",
                str(seed),
            )
        )
        with mock.patch("sys.argv", argv):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(subject.main(), 0)
        return json.loads((output / "accumulation_summary.json").read_text(encoding="utf-8"))

    def test_validate_batch_accepts_robust_semantic_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._write_batch(directory)
            summary, rows = subject.validate_batch(directory)
        self.assertEqual(summary["frontier_count"], 1)
        self.assertEqual(len(rows), 1)

    def test_below_threshold_writes_pool_without_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch = root / "batch"
            output = root / "accumulated"
            self._write_batch(batch)
            (output / subject.TRAINING_READY_MANIFEST).parent.mkdir(parents=True)
            (output / subject.TRAINING_READY_MANIFEST).write_text("{}\n", encoding="utf-8")
            summary = self._run([batch], output, min_frontier=2)
            self.assertFalse(summary["threshold_reached"])
            self.assertFalse(summary["split_created"])
            self.assertFalse((output / "splits").exists())
            self.assertFalse((output / subject.TRAINING_READY_MANIFEST).exists())

    def test_materialization_rejects_missing_target_audit(self) -> None:
        row = robust_row()
        row["stage64_permutation_eval"]["permutations"] = [
            item for item in row["stage64_permutation_eval"]["permutations"] if item["gold"] != "A"
        ]
        materialized, rejection = subject.materialize_audited_permutation(
            row,
            target_gold="A",
            partition="train",
            balance_seed=11,
        )
        self.assertIsNone(materialized)
        self.assertEqual(rejection["reason"], "missing_target_permutation")

    def test_generation_seed_may_differ_but_eval_seed_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch1 = root / "batch1"
            batch2 = root / "batch2"
            self._write_batch(
                batch1,
                rows=[robust_row("sample-1", parent="conversation-1")],
                generation_seed=101,
            )
            self._write_batch(
                batch2,
                rows=[robust_row("sample-2", parent="conversation-2")],
                generation_seed=202,
            )
            summary = self._run([batch1, batch2], root / "accepted", min_frontier=3)
            self.assertEqual(summary["batch_generation_seeds"], [101, 202])
            self.assertFalse(summary["generation_seed_compatibility_required"])

            mismatch = root / "mismatch"
            self._write_batch(
                mismatch,
                rows=[robust_row("sample-3", parent="conversation-3")],
                generation_seed=303,
                eval_seed=8,
            )
            with self.assertRaisesRegex(RuntimeError, "fixed solver-eval configuration mismatch"):
                self._run([batch1, mismatch], root / "rejected", min_frontier=3)
            self.assertFalse((root / "rejected" / subject.TRAINING_READY_MANIFEST).exists())

    def test_explicit_solver_eval_seed_wins_over_legacy_top_level_seed(self) -> None:
        first = batch_summary(generation_seed=101, eval_seed=101)
        second = batch_summary(generation_seed=202, eval_seed=202)
        first["solver_eval_seed"] = 7
        second["solver_eval_seed"] = 7
        first_config, first_sources = subject.fixed_solver_eval_config(first)
        second_config, second_sources = subject.fixed_solver_eval_config(second)
        self.assertEqual(first_config, second_config)
        self.assertEqual(first_config["seed"], 7)
        self.assertEqual(first_sources["seed"], "top_level.solver_eval_seed")
        self.assertEqual(second_sources["seed"], "top_level.solver_eval_seed")

    def test_balance_rejection_prevents_training_ready_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch = root / "batch"
            output = root / "accumulated"
            rows = [robust_row(f"sample-{index}", parent=f"conversation-{index}") for index in range(12)]
            rows[0]["stage64_permutation_eval"]["permutations"].pop()
            self._write_batch(batch, rows=rows)
            output.mkdir(parents=True)
            (output / subject.TRAINING_READY_MANIFEST).write_text("{}\n", encoding="utf-8")
            summary = self._run([batch], output, min_frontier=12, seed=31)
            self.assertFalse(summary["training_ready"])
            self.assertEqual(
                summary["train_balance_rejected_count"] + summary["heldout_balance_rejected_count"],
                1,
            )
            self.assertFalse((output / subject.TRAINING_READY_MANIFEST).exists())

    def test_balances_each_split_from_audited_positions_and_writes_ready_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch = root / "batch"
            output = root / "accumulated"
            rows = [
                robust_row(
                    f"sample-{index}",
                    parent=f"conversation-{index}",
                    arity=3 if index % 2 else 4,
                )
                for index in range(24)
            ]
            canonical = copy.deepcopy(rows)
            self._write_batch(batch, rows=rows, generation_seed=101)
            summary = self._run([batch], output, min_frontier=24, seed=29)

            self.assertTrue(summary["training_ready"])
            self.assertTrue(all(summary["readiness_gates"].values()))
            self.assertTrue((output / subject.TRAINING_READY_MANIFEST).exists())
            self.assertEqual(recal.read_jsonl(output / "frontier_pool.jsonl"), canonical)
            self.assertEqual({recal.qa_gold(row) for row in canonical}, {"A"})

            split = output / "splits/frontier_lineage_group_holdout_seed29"
            train = recal.read_jsonl(split / "frontier_train.jsonl")
            heldout = recal.read_jsonl(split / "frontier_heldout.jsonl")
            for partition_rows in (train, heldout):
                arities = {subject.option_arity(row) for row in partition_rows}
                for arity in arities:
                    same_arity = [row for row in partition_rows if subject.option_arity(row) == arity]
                    counts = Counter(recal.qa_gold(row) for row in same_arity)
                    labels = recal.LETTERS[:arity]
                    self.assertLessEqual(
                        max(counts.get(label, 0) for label in labels)
                        - min(counts.get(label, 0) for label in labels),
                        1,
                    )
                for row in partition_rows:
                    self.assertEqual(
                        len(row["stage64_permutation_eval"]["permutations"]),
                        subject.option_arity(row),
                    )
                    selected = row["stage3_option_balance"]["selected_permutation_eval"]
                    self.assertEqual(selected["gold"], recal.qa_gold(row))
                    self.assertEqual(selected["shift"], row["stage64_permutation"]["shift"])
                    self.assertEqual(selected["exact_correct_count"], row["solver_eval"]["exact_correct_count"])

            train_groups = {subject.parent_sample_id(row) for row in train}
            heldout_groups = {subject.parent_sample_id(row) for row in heldout}
            self.assertFalse(train_groups & heldout_groups)
            self.assertEqual(recal.read_jsonl(split / "frontier_train_balance_rejected.jsonl"), [])
            self.assertEqual(recal.read_jsonl(split / "frontier_heldout_balance_rejected.jsonl"), [])


if __name__ == "__main__":
    unittest.main()
