from __future__ import annotations

import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.current_method import build_stage3_train_mix as builder


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sample(
    sample_id: str,
    group: str,
    *,
    question_id: str | None = None,
    dialogue_hash: str | None = None,
    text: str | None = None,
    question: str | None = None,
) -> dict:
    question_id = question_id or f"q-{sample_id}"
    dialogue_hash = dialogue_hash or f"d-{sample_id}"
    text = text or f"{sample_id} says the meeting is in Paris."
    question = question or f"Where is the {sample_id} meeting?"
    return {
        "sample_id": sample_id,
        "lineage_group": group,
        "question_id": question_id,
        "dialogue_hash": dialogue_hash,
        "seed_source": {"parent_sample_id": group},
        "dialogue": {
            "dialogue_hash": dialogue_hash,
            "turns": [
                {
                    "turn_id": "S1:1",
                    "session_id": "S1",
                    "speaker": "Ada",
                    "text": text,
                }
            ],
        },
        "qa": {
            "question_id": question_id,
            "question_type": "single_choice",
            "question": question,
            "options": [
                {"id": "A", "text": "Paris"},
                {"id": "B", "text": "Rome"},
                {"id": "C", "text": "Oslo"},
                {"id": "D", "text": "Lima"},
            ],
            "gold": ["A"],
        },
    }


def ready_manifest(root: Path, train: list[dict], heldout: list[dict]) -> Path:
    train_path = root / "accumulation" / "splits" / "frontier_train.jsonl"
    heldout_path = root / "accumulation" / "splits" / "frontier_heldout.jsonl"
    manifest_path = root / "accumulation" / "training_ready_manifest.json"
    write_jsonl(train_path, train)
    write_jsonl(heldout_path, heldout)
    write_json(
        manifest_path,
        {
            "schema_version": builder.MANIFEST_SCHEMA,
            "status": "training_ready",
            "group_level": "lineage_group",
            "readiness_gates": {
                "minimum_reliable_frontier_reached": True,
                "all_train_rows_materialized_from_audit": True,
                "all_heldout_rows_materialized_from_audit": True,
                "train_gold_balanced_within_each_arity": True,
                "heldout_gold_balanced_within_each_arity": True,
                "train_heldout_parent_group_overlap_zero": True,
                "train_partition_nonempty": True,
                "heldout_partition_nonempty": True,
            },
            "artifacts": {
                "balanced_frontier_train": str(train_path),
                "balanced_frontier_heldout": str(heldout_path),
            },
        },
    )
    return manifest_path


def args_for(
    manifest: Path,
    output: Path,
    replay: list[Path],
    *,
    replay_ratio: float = 0.25,
    validation_ratio: float = 0.0,
    seed: int = 17,
) -> argparse.Namespace:
    return argparse.Namespace(
        training_ready_manifest=manifest,
        replay_jsonl=replay,
        output_dir=output,
        replay_ratio=replay_ratio,
        validation_ratio=validation_ratio,
        seed=seed,
        overwrite=False,
    )


class BuildStage3TrainMixTest(unittest.TestCase):
    def test_replay_may_share_a_training_lineage_but_not_training_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = [
                sample("train-a", "shared-train-group"),
                sample("train-b", "other-train-group"),
            ]
            manifest = ready_manifest(root, train, [sample("heldout", "heldout-group")])
            replay_path = root / "replay.jsonl"
            write_jsonl(
                replay_path,
                [sample("replay-same-lineage", "shared-train-group")],
            )

            summary = builder.build(
                args_for(manifest, root / "output", [replay_path], replay_ratio=1 / 3)
            )

            self.assertEqual(summary["status"], "audit_passed")
            self.assertEqual(summary["replay_selected_count"], 1)
            selected = read_jsonl(root / "output" / "replay_selected.jsonl")
            self.assertEqual(selected[0]["sample_id"], "replay-same-lineage")

    def test_manifest_gates_must_all_be_true(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = ready_manifest(
                root,
                [sample("train-1", "train-group")],
                [sample("heldout-1", "heldout-group")],
            )
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["readiness_gates"]["minimum_reliable_frontier_reached"] = False
            write_json(manifest, payload)
            with self.assertRaisesRegex(RuntimeError, "readiness gates"):
                builder.build(
                    args_for(
                        manifest,
                        root / "output",
                        [],
                        replay_ratio=0.0,
                    )
                )

    def test_official_heldout_is_exclusion_only_across_all_identities(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = [sample(f"train-{index}", f"train-group-{index}") for index in range(4)]
            heldout = sample(
                "heldout-sample",
                "heldout-group",
                question_id="heldout-question",
                dialogue_hash="heldout-dialogue",
                text="A shared heldout content sentence.",
                question="What is the shared heldout fact?",
            )
            manifest = ready_manifest(root, train, [heldout])
            replay_path = root / "replay.jsonl"
            collision_rows = [
                sample("heldout-sample", "unique-sample-group"),
                sample("question-collision", "unique-question-group", question_id="heldout-question"),
                sample("dialogue-collision", "unique-dialogue-group", dialogue_hash="heldout-dialogue"),
                sample("group-collision", "heldout-group"),
                sample(
                    "content-collision",
                    "unique-content-group",
                    text="  A SHARED heldout content sentence. ",
                    question=" What is the shared heldout fact? ",
                ),
            ]
            unique_replay = sample("replay-unique", "replay-unique-group")
            write_jsonl(replay_path, collision_rows + [unique_replay])

            summary = builder.build(
                args_for(manifest, root / "output", [replay_path], replay_ratio=0.20)
            )

            self.assertEqual(summary["status"], "audit_passed")
            self.assertEqual(summary["replay_selected_count"], 1)
            self.assertTrue(summary["leakage_audit"]["train_vs_official_heldout"]["zero"])
            selected = read_jsonl(root / "output" / "replay_selected.jsonl")
            self.assertEqual([row["sample_id"] for row in selected], ["replay-unique"])
            exclusions = read_jsonl(root / "output" / "replay_excluded.jsonl")
            protected = [row for row in exclusions if row["reason"] == "protected_partition_overlap"]
            matched_keys = {
                key
                for row in protected
                for key in row.get("identity_matches", {})
            }
            self.assertEqual(matched_keys, set(builder.IDENTITY_KEYS))
            valid_manifest = json.loads(
                (root / "output" / "validation_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(valid_manifest["status"], "no_valid")
            self.assertFalse(valid_manifest["official_stage3_heldout_used"])

    def test_replay_ratio_and_selection_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = [sample(f"train-{index}", f"train-group-{index}") for index in range(6)]
            manifest = ready_manifest(root, train, [sample("heldout", "heldout-group")])
            replay_path = root / "replay.jsonl"
            write_jsonl(
                replay_path,
                [sample(f"replay-{index}", f"replay-group-{index}") for index in range(8)],
            )

            first = builder.build(
                args_for(manifest, root / "output-1", [replay_path], replay_ratio=0.25, seed=31)
            )
            second = builder.build(
                args_for(manifest, root / "output-2", [replay_path], replay_ratio=0.25, seed=31)
            )

            self.assertEqual(first["replay_requested_count"], 2)
            self.assertEqual(first["replay_actual_ratio"], 0.25)
            first_replay = [row["sample_id"] for row in read_jsonl(root / "output-1" / "replay_selected.jsonl")]
            second_replay = [row["sample_id"] for row in read_jsonl(root / "output-2" / "replay_selected.jsonl")]
            first_mix = [row["sample_id"] for row in read_jsonl(root / "output-1" / "train_mix.jsonl")]
            second_mix = [row["sample_id"] for row in read_jsonl(root / "output-2" / "train_mix.jsonl")]
            self.assertEqual(first_replay, second_replay)
            self.assertEqual(first_mix, second_mix)
            provenance = json.loads((root / "output-1" / "provenance.json").read_text(encoding="utf-8"))
            self.assertTrue(provenance["inputs"]["training_ready_manifest"]["sha256"])
            self.assertTrue(provenance["outputs"]["sft_train"]["sha256"])
            self.assertTrue((root / "output-1" / "provenance.sha256").read_text(encoding="ascii"))

    def test_frontier_and_replay_duplicates_are_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = sample("train-original", "train-group")
            duplicate = copy.deepcopy(original)
            duplicate["sample_id"] = "train-duplicate-id"
            manifest = ready_manifest(
                root,
                [original, duplicate, sample("train-2", "train-group-2")],
                [sample("heldout", "heldout-group")],
            )
            replay_path = root / "replay.jsonl"
            replay_original = sample("replay-original", "replay-group")
            replay_duplicate = copy.deepcopy(replay_original)
            replay_duplicate["sample_id"] = "replay-duplicate-id"
            write_jsonl(replay_path, [replay_original, replay_duplicate, sample("replay-2", "replay-group-2")])

            summary = builder.build(
                args_for(manifest, root / "output", [replay_path], replay_ratio=0.50)
            )

            self.assertEqual(summary["frontier_duplicate_excluded_count"], 1)
            self.assertEqual(summary["frontier_train_count"], 2)
            self.assertEqual(summary["replay_requested_count"], 2)
            self.assertEqual(summary["replay_selected_count"], 2)
            exclusions = read_jsonl(root / "output" / "replay_excluded.jsonl")
            self.assertEqual(sum(row["reason"] == "duplicate_replay" for row in exclusions), 1)
            mix_ids = [row["sample_id"] for row in read_jsonl(root / "output" / "train_mix.jsonl")]
            self.assertEqual(len(mix_ids), len(set(mix_ids)))

    def test_validation_is_reserved_from_train_groups_not_official_heldout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = [sample(f"train-{index}", f"group-{index}") for index in range(8)]
            manifest = ready_manifest(root, train, [sample("official-heldout", "official-group")])
            summary = builder.build(
                args_for(
                    manifest,
                    root / "output",
                    [],
                    replay_ratio=0.0,
                    validation_ratio=0.25,
                    seed=9,
                )
            )
            self.assertEqual(summary["validation"]["status"], "ready")
            self.assertEqual(summary["train_derived_validation_count"], 2)
            valid = read_jsonl(root / "output" / "validation_train_derived.jsonl")
            train_mix = read_jsonl(root / "output" / "train_mix.jsonl")
            self.assertNotIn("official-heldout", {row["sample_id"] for row in valid})
            self.assertTrue(builder.overlap_summary(train_mix, valid)["zero"])

    def test_manifest_group_level_controls_the_canonical_group_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = sample("train", "raw-lineage-shared")
            heldout = sample("heldout", "raw-lineage-shared")
            train["seed_source"]["parent_sample_id"] = "parent-train"
            heldout["seed_source"]["parent_sample_id"] = "parent-heldout"
            manifest = ready_manifest(root, [train], [heldout])
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["group_level"] = "parent_sample_id"
            write_json(manifest, payload)

            summary = builder.build(
                args_for(manifest, root / "output", [], replay_ratio=0.0)
            )

            self.assertEqual(summary["status"], "audit_passed")
            self.assertEqual(summary["group_level"], "parent_sample_id")
            self.assertTrue(summary["leakage_audit"]["train_vs_official_heldout"]["zero"])


if __name__ == "__main__":
    unittest.main()
