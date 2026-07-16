from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.current_method import build_stage3_parent_v2 as builder


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def train_row(sample_id: str, parent_id: str, text: str) -> dict:
    return {
        "sample_id": sample_id,
        "source_id": f"source-{sample_id}",
        "split": "train",
        "training_use": "train_candidate",
        "speaker_id": "speaker-registry-1",
        "session_id": "source-session-1",
        "turn_id": "source-turn-collection-1",
        "dialogue": {
            "dialogue_hash": f"dialogue-{sample_id}",
            "turns": [
                {"turn_id": "S1:1", "session_id": "S1", "speaker": "Ada", "text": text},
                {"turn_id": "S1:2", "session_id": "S1", "speaker": "Ben", "text": "Understood."},
            ],
        },
        "seed_source": {
            "split": "train",
            "source_split": "train",
            "training_use": "train_candidate",
            "allowed_train_seed": True,
            "parent_sample_id": parent_id,
        },
        "provenance": {"round": "stage2", "note": "keep me", "qa": {"gold": "A"}},
        "graph_update": {
            "graph_snapshot_hash": f"graph-{sample_id}",
            "events": [
                {
                    "event_id": "clean",
                    "entity": "Ada",
                    "slot": "meeting_city",
                    "value": "Paris",
                    "status": "active",
                    "time_index": 1,
                    "evidence_turn_ids": ["S1:1"],
                },
                {
                    "event_id": "raw",
                    "entity": "Ada",
                    "slot": "raw_qa_temporal",
                    "value": "Q: Where is the meeting? A: Paris",
                },
            ],
        },
        "qa": {"question": "Where?", "options": ["Paris"], "gold": "A"},
        "solver_eval": {"bucket": "2/5"},
        "frontier_label": {"is_frontier": True},
        "question": "Where?",
        "options": ["Paris"],
        "gold_answer": "A",
    }


def add_historical_padding(row: dict) -> list[str]:
    padding_ids: list[str] = []
    for index, text in enumerate(builder.KNOWN_DETERMINISTIC_PADDING_TEXTS, start=1):
        turn_id = f"P:{index}"
        padding_ids.append(turn_id)
        row["dialogue"]["turns"].append(
            {
                "turn_id": turn_id,
                "session_id": "padding",
                "speaker": "Historical filler",
                "text": text,
            }
        )
    return padding_ids


class BuildStage3ParentV2Test(unittest.TestCase):
    def test_migration_is_immutable_and_strips_old_challenger_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "stage2.jsonl"
            row = train_row("s1", "parent-train-1", "The meeting is in Paris.")
            write_jsonl(source, [row])
            before = source.read_bytes()

            accepted, rejected, summary = builder.migrate([source], [])

            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(len(accepted), 1)
            self.assertFalse(rejected)
            parent = accepted[0]
            self.assertEqual(parent["dialogue"], row["dialogue"])
            self.assertEqual(parent["turns"], row["dialogue"]["turns"])
            self.assertEqual(parent["speaker_id"], row["speaker_id"])
            self.assertEqual(parent["session_id"], row["session_id"])
            self.assertEqual(parent["turn_id"], row["turn_id"])
            self.assertEqual(parent["provenance"]["provenance"]["note"], "keep me")
            self.assertNotIn("qa", parent["provenance"]["provenance"])
            self.assertEqual([event["event_id"] for event in parent["semantic_parent_graph"]["events"]], ["clean"])
            self.assertFalse(semantic_issues := builder.semantic_graph_v2.validate_graph_delta(parent["semantic_parent_graph"]))
            self.assertNotIn("qa", parent)
            self.assertNotIn("solver_eval", parent)
            self.assertNotIn("frontier_label", parent)
            self.assertNotIn("graph_update", parent)
            expected_hash = hashlib.sha256(builder.canonical_json(row).encode("utf-8")).hexdigest()
            self.assertEqual(parent["source_snapshot"]["row_sha256"], expected_hash)
            self.assertEqual(summary["retained_semantic_event_count"], 1)
            self.assertEqual(summary["removed_legacy_or_qa_graph_event_count"], 1)

    def test_padding_cleanup_is_opt_in_exact_synchronized_and_versioned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.jsonl"
            row = train_row("padding", "padding-parent", "Keep this semantic turn.")
            padding_ids = add_historical_padding(row)
            near_match_id = "P:near"
            row["dialogue"]["turns"].append(
                {
                    "turn_id": near_match_id,
                    "speaker": "Reviewer",
                    "text": builder.KNOWN_DETERMINISTIC_PADDING_TEXTS[0] + " ",
                }
            )
            # The migration must source one canonical turn list and then keep
            # both output representations synchronized.
            row["turns"] = [{"turn_id": "stale", "speaker": "X", "text": "stale"}]
            write_jsonl(source, [row])
            before = source.read_bytes()

            default_parents, _, default_summary = builder.migrate([source], [])
            cleaned_parents, _, cleaned_summary = builder.migrate(
                [source], [], strip_deterministic_padding=True
            )

            self.assertEqual(source.read_bytes(), before)
            default_parent = default_parents[0]
            cleaned_parent = cleaned_parents[0]
            expected_legacy_id = f"stage3_parent_v2_{builder.sha256_payload(row)[:20]}"
            self.assertEqual(default_parent["sample_id"], expected_legacy_id)
            self.assertNotEqual(cleaned_parent["sample_id"], default_parent["sample_id"])
            self.assertNotIn("deterministic_padding_cleanup", default_parent["metadata"])
            self.assertEqual(default_parent["turns"], row["dialogue"]["turns"])
            self.assertFalse(default_summary["deterministic_padding_cleanup"]["enabled"])

            cleaned_turns = cleaned_parent["turns"]
            cleaned_ids = [turn["turn_id"] for turn in cleaned_turns]
            self.assertEqual(cleaned_parent["dialogue"]["turns"], cleaned_turns)
            self.assertEqual(cleaned_ids, ["S1:1", "S1:2", near_match_id])
            self.assertFalse(set(padding_ids) & set(cleaned_ids))
            self.assertTrue(builder.known_deterministic_padding_text(row["dialogue"]["turns"][2]["text"]))
            self.assertFalse(builder.known_deterministic_padding_text(cleaned_turns[-1]["text"]))
            self.assertEqual(
                len(builder.detect_historical_deterministic_padding(row["dialogue"]["turns"])),
                len(builder.KNOWN_DETERMINISTIC_PADDING_TEXTS),
            )

            audit = cleaned_parent["metadata"]["deterministic_padding_cleanup"]
            self.assertTrue(audit["enabled"])
            self.assertEqual(audit["pre_turn_count"], len(row["dialogue"]["turns"]))
            self.assertEqual(audit["post_turn_count"], 3)
            self.assertEqual(audit["removed_turn_count"], len(padding_ids))
            self.assertEqual(audit["removed_turn_ids"], sorted(padding_ids))
            self.assertNotEqual(audit["pre_turns_sha256"], audit["post_turns_sha256"])
            self.assertEqual(
                audit["post_dialogue_content_sha256"],
                builder.dialogue_content_hash(cleaned_turns),
            )
            self.assertEqual(
                cleaned_parent["dialogue"]["dialogue_hash"],
                builder.dialogue_content_hash(cleaned_turns),
            )
            cleanup_summary = cleaned_summary["deterministic_padding_cleanup"]
            self.assertTrue(cleanup_summary["enabled"])
            self.assertEqual(cleanup_summary["processed_row_count"], 1)
            self.assertEqual(cleanup_summary["affected_row_count"], 1)
            self.assertEqual(cleanup_summary["removed_turn_count"], len(padding_ids))
            self.assertEqual(cleanup_summary["pre_turn_count"], len(row["dialogue"]["turns"]))
            self.assertEqual(cleanup_summary["post_turn_count"], 3)

    def test_padding_cleanup_fails_closed_on_retained_graph_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.jsonl"
            row = train_row("padding-evidence", "padding-parent", "Keep the first turn.")
            padding_ids = add_historical_padding(row)
            row["graph_update"]["events"][0]["evidence_turn_ids"] = [padding_ids[0]]
            write_jsonl(source, [row])

            with self.assertRaisesRegex(
                builder.DeterministicPaddingEvidenceReferenceError,
                padding_ids[0],
            ):
                builder.migrate([source], [], strip_deterministic_padding=True)

    def test_legacy_qa_dialogue_cleanup_is_deterministic_and_preserves_turn_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.jsonl"
            row = train_row("qa-dialogue", "qa-parent", "Keep this evidence turn.")
            row["dialogue"]["turns"].extend(
                [
                    {
                        "turn_id": "S1:3",
                        "speaker": "Reviewer",
                        "text": "Ada raw_qa_temporal: only Jordan can see this report.",
                    },
                    {
                        "turn_id": "S1:4",
                        "speaker": "Analyst",
                        "text": "Latest detail is Q: Where is the meeting? A: Paris",
                    },
                    {
                        "turn_id": "S1:5",
                        "speaker": "Ada",
                        "text": "Q: Should we review this again tomorrow?",
                    },
                ]
            )
            write_jsonl(source, [row])

            parents, rejected, summary = builder.migrate([source], [])

            self.assertFalse(rejected)
            self.assertEqual(len(parents), 1)
            parent = parents[0]
            retained_ids = [turn["turn_id"] for turn in parent["turns"]]
            self.assertEqual(retained_ids, ["S1:1", "S1:2", "S1:5"])
            self.assertEqual(parent["dialogue"]["turns"], parent["turns"])
            self.assertEqual(
                parent["dialogue"]["dialogue_hash"],
                builder.dialogue_content_hash(parent["turns"]),
            )
            self.assertEqual(
                parent["semantic_parent_graph"]["events"][0]["evidence_turn_ids"],
                ["S1:1"],
            )
            audit = parent["metadata"]["legacy_qa_dialogue_cleanup"]
            self.assertEqual(audit["removed_turn_ids"], ["S1:3", "S1:4"])
            self.assertEqual(audit["removed_turn_count"], 2)
            cleanup_summary = summary["legacy_qa_dialogue_cleanup"]
            self.assertTrue(cleanup_summary["enabled"])
            self.assertEqual(cleanup_summary["affected_row_count"], 1)
            self.assertEqual(cleanup_summary["removed_turn_count"], 2)

    def test_legacy_qa_dialogue_cleanup_rejects_retained_evidence_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.jsonl"
            row = train_row("qa-evidence", "qa-parent", "Keep the ordinary turn.")
            row["dialogue"]["turns"].append(
                {
                    "turn_id": "S1:qa-evidence",
                    "speaker": "Analyst",
                    "text": "Q: Where is the meeting? A: Paris",
                }
            )
            row["graph_update"]["events"][0]["evidence_turn_ids"] = [
                "S1:qa-evidence"
            ]
            write_jsonl(source, [row])

            parents, rejected, summary = builder.migrate([source], [])

            self.assertFalse(parents)
            self.assertEqual(len(rejected), 1)
            self.assertIn(
                "legacy_qa_dialogue_cleanup_would_remove_retained_evidence",
                rejected[0]["reasons"],
            )
            self.assertEqual(
                rejected[0]["legacy_qa_dialogue_cleanup"][
                    "blocked_retained_evidence_turn_ids"
                ],
                ["S1:qa-evidence"],
            )
            self.assertEqual(
                summary["legacy_qa_dialogue_cleanup"][
                    "blocked_retained_evidence_row_count"
                ],
                1,
            )

    def test_multiple_inputs_enforce_train_forbidden_and_deduplication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            forbidden = root / "heldout.jsonl"
            accepted = train_row("accepted", "parent-a", "Unique accepted dialogue.")
            duplicate_content = train_row("dupe-content", "parent-b", "Unique accepted dialogue.")
            duplicate_lineage = train_row("dupe-lineage", "parent-a", "Different descendant on same lineage.")
            duplicate_lineage["metadata"] = {"root_seed_parent_id": "parent-a"}
            del duplicate_lineage["seed_source"]["parent_sample_id"]
            heldout = train_row("heldout-row", "parent-c", "Heldout dialogue.")
            heldout["split"] = "heldout"
            wrong_use = train_row("wrong-use", "parent-d", "Wrong training use.")
            wrong_use["training_use"] = "eval_only"
            leaked = train_row("leaked", "heldout-parent", "Forbidden lineage dialogue.")
            write_jsonl(first, [accepted, heldout, wrong_use])
            write_jsonl(second, [duplicate_content, duplicate_lineage, leaked])
            write_jsonl(forbidden, [{"sample_id": "heldout-sample", "parent_sample_id": "heldout-parent"}])

            parents, rejected, summary = builder.migrate([first, second], [forbidden])

            self.assertEqual(
                [row["source_snapshot"]["source_sample_id"] for row in parents],
                ["accepted", "dupe-lineage"],
            )
            self.assertEqual(parents[0]["lineage_group"], "parent-a")
            self.assertEqual(parents[1]["lineage_group"], "parent-a")
            reasons = {row["source_sample_id"]: row["reasons"] for row in rejected}
            self.assertIn("forbidden_split:split=heldout", reasons["heldout-row"])
            self.assertIn("training_use_not_train_candidate:eval_only", reasons["wrong-use"])
            self.assertIn("duplicate_dialogue_content", reasons["dupe-content"])
            self.assertIn("forbidden_lineage_identifier", reasons["leaked"])
            self.assertEqual(summary["input_count"], 6)
            self.assertEqual(summary["accepted_count"], 2)
            self.assertEqual(summary["rejected_count"], 4)
            self.assertEqual(summary["lineage_groups_with_multiple_dialogues"], 1)

    def test_exact_mode_matches_same_class_and_only_audits_lineage_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jsonl"
            forbidden = root / "forbidden.jsonl"
            cross_class = train_row("heldout-source", "shared-root", "Cross-class identity is allowed.")
            cross_class["source_id"] = "train-source"
            cross_class["question_id"] = "train-question"
            same_sample = train_row("heldout-sample", "train-root-2", "Same sample ID is rejected.")
            same_source = train_row("train-s3", "train-root-3", "Reusable source chunk is audited.")
            same_source["source_id"] = "heldout-source"
            same_question = train_row("train-s4", "train-root-4", "Same question ID is rejected.")
            same_question["question_id"] = "heldout-question"
            same_dialogue_hash = train_row("train-s5", "train-root-5", "Same dialogue hash is rejected.")
            same_dialogue_hash["dialogue"]["dialogue_hash"] = "heldout-dialogue"
            write_jsonl(
                source,
                [cross_class, same_sample, same_source, same_question, same_dialogue_hash],
            )
            write_jsonl(
                forbidden,
                [
                    {
                        "sample_id": "heldout-sample",
                        "source_id": "heldout-source",
                        "question_id": "heldout-question",
                        "dialogue": {"dialogue_hash": "heldout-dialogue"},
                        "seed_source": {"parent_sample_id": "shared-root"},
                    }
                ],
            )

            parents, rejected, summary = builder.migrate(
                [source], [forbidden], forbidden_match_mode="exact"
            )

            self.assertEqual(
                [row["source_snapshot"]["source_sample_id"] for row in parents],
                ["heldout-source", "train-s3"],
            )
            parent = parents[0]
            self.assertEqual(parent["lineage_group"], "shared-root")
            audit = parent["metadata"]["forbidden_lineage_overlap_audit"]
            self.assertTrue(audit["has_overlap"])
            self.assertFalse(audit["hard_rejected"])
            self.assertIn("shared-root", audit["identifiers"])
            matches = {
                row["source_sample_id"]: row["forbidden_exact_identity_matches"]
                for row in rejected
            }
            self.assertEqual(matches["heldout-sample"], {"sample_id": ["heldout-sample"]})
            self.assertEqual(matches["train-s4"], {"question_id": ["heldout-question"]})
            self.assertEqual(matches["train-s5"], {"dialogue_hash": ["heldout-dialogue"]})
            self.assertEqual(summary["forbidden_match_mode"], "exact")
            self.assertEqual(summary["accepted_lineage_overlap_row_count"], 2)
            self.assertGreaterEqual(summary["lineage_overlap_row_count"], 1)

    def test_exact_sample_identity_is_deduplicated_without_lineage_dedup(self) -> None:
        first = train_row("same-id", "shared-root", "First distinct dialogue.")
        second = train_row("same-id", "shared-root", "Second distinct dialogue.")
        second["source_id"] = "another-source"
        second["dialogue"]["dialogue_hash"] = "another-dialogue"
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.jsonl"
            write_jsonl(source, [first, second])
            parents, rejected, _ = builder.migrate(
                [source], [], forbidden_match_mode="exact"
            )
        self.assertEqual(len(parents), 1)
        self.assertEqual(rejected[0]["reasons"], ["duplicate_exact_identity"])
        self.assertEqual(
            rejected[0]["duplicate_exact_identity_matches"],
            {"sample_id": ["same-id"]},
        )

    def test_semantic_graph_may_be_empty_and_source_is_not_mutated(self) -> None:
        row = train_row("raw-only", "parent-raw", "A clean dialogue with no usable legacy graph.")
        row["graph_update"]["events"] = [copy.deepcopy(row["graph_update"]["events"][1])]
        original = copy.deepcopy(row)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.jsonl"
            write_jsonl(source, [row])
            parents, _, _ = builder.migrate([source], [])
        self.assertEqual(row, original)
        self.assertEqual(parents[0]["semantic_parent_graph"]["events"], [])
        self.assertEqual(parents[0]["semantic_parent_graph"]["event_count"], 0)

    def test_require_nonempty_semantic_parent_graph_rejects_empty_parent(self) -> None:
        row = train_row("raw-only-formal", "parent-raw", "No usable legacy graph.")
        row["graph_update"]["events"] = [copy.deepcopy(row["graph_update"]["events"][1])]
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.jsonl"
            write_jsonl(source, [row])
            parents, rejected, summary = builder.migrate(
                [source], [], require_nonempty_semantic_parent_graph=True
            )

        self.assertFalse(parents)
        self.assertEqual(rejected[0]["reasons"], ["empty_semantic_parent_graph"])
        self.assertTrue(summary["require_nonempty_semantic_parent_graph"])
        admission = summary["semantic_parent_graph_admission"]
        self.assertTrue(admission["require_nonempty"])
        self.assertEqual(admission["empty_input_row_count"], 1)
        self.assertEqual(admission["rejected_empty_row_count"], 1)
        self.assertEqual(admission["accepted_empty_row_count"], 0)
        self.assertTrue(admission["fail_closed_requirement_satisfied"])

    def test_cli_writes_the_required_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jsonl"
            output = root / "output"
            row = train_row("cli", "cli-parent", "CLI migration dialogue.")
            add_historical_padding(row)
            write_jsonl(source, [row])
            argv = [
                "build_stage3_parent_v2.py",
                "--input",
                str(source),
                "--forbidden-match-mode",
                "exact",
                "--strip-historical-deterministic-padding",
                "--require-nonempty-semantic-parent-graph",
                "--output-dir",
                str(output),
            ]
            with mock.patch("sys.argv", argv), mock.patch("builtins.print"):
                self.assertEqual(builder.main(), 0)
            self.assertTrue((output / "parent_samples.jsonl").is_file())
            self.assertTrue((output / "rejected.jsonl").is_file())
            summary = json.loads((output / "migration_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["accepted_count"], 1)
            self.assertEqual(summary["rejected_count"], 0)
            self.assertEqual(summary["forbidden_match_mode"], "exact")
            self.assertTrue(summary["require_nonempty_semantic_parent_graph"])
            self.assertEqual(
                summary["semantic_parent_graph_admission"]["accepted_empty_row_count"],
                0,
            )
            self.assertTrue(summary["deterministic_padding_cleanup"]["enabled"])
            self.assertEqual(
                summary["deterministic_padding_cleanup"]["removed_turn_count"],
                len(builder.KNOWN_DETERMINISTIC_PADDING_TEXTS),
            )


if __name__ == "__main__":
    unittest.main()
