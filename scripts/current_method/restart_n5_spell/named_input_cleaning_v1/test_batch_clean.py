from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from .batch_clean import (
    _unresolved_short_tokens,
    clean_expanded_dialogue,
    clean_text,
    build_replacement_plan,
    run_batch,
    validate_cleaned_record,
)


class BatchCleaningTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mapping = {"Classmates A": "Maya", "Classmates B": "Jordan"}

    def test_short_alias_is_context_limited(self) -> None:
        plan = build_replacement_plan(self.mapping)
        counts: dict[str, int] = {}
        text = (
            "Hey, B. A bakery could help. A bit later, thank you, A. "
            "A check might still be useful."
        )
        cleaned = clean_text(text, plan, counts)
        self.assertEqual(
            cleaned,
            "Hey, Jordan. A bakery could help. A bit later, thank you, Maya. "
            "A check might still be useful.",
        )
        self.assertEqual(counts["direct_address"], 2)

    def test_short_pair_and_full_alias(self) -> None:
        plan = build_replacement_plan(self.mapping)
        counts: dict[str, int] = {}
        cleaned = clean_text(
            "Classmates A and B talked; Classmate B thanked Classmate A.",
            plan, counts,
        )
        self.assertEqual(cleaned, "Maya and Jordan talked; Jordan thanked Maya.")

    def test_probable_bare_subject_is_reported_instead_of_silently_kept(self) -> None:
        self.assertEqual(
            _unresolved_short_tokens("A said hello.", enabled=True), ["A"])
        self.assertEqual(
            _unresolved_short_tokens("A bakery opened.", enabled=True), [])

    def test_bare_pair_is_replaced_without_touching_articles(self) -> None:
        plan = build_replacement_plan(self.mapping)
        counts: dict[str, int] = {}
        cleaned = clean_text(
            "A and B compared notes. A bakery opened, and B agreed.",
            plan, counts,
        )
        self.assertEqual(
            cleaned,
            "Maya and Jordan compared notes. A bakery opened, and B agreed.",
        )
        self.assertEqual(counts["short_pair"], 1)

    def test_raw_jsonl_batch_emits_cleaned_source_and_bound_provenance(self) -> None:
        record = {
            "dataID": "episode-raw-fixture",
            "relationship": "Classmates",
            "time_interval": ["day 1", "day 2", "day 3", "day 4", "day 5"],
            "summary": ["Classmates A and B talk."] * 5,
        }
        for index, name in enumerate(("first", "second", "third", "fourth", "fifth"), 1):
            record[f"{name}_session_speakers"] = ["Classmates A", "Classmates B"]
            record[f"{name}_session_dialogue"] = [
                "Hey, B. A bakery opened.",
                "Thanks, A.",
            ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            source.write_text(json.dumps(record) + "\n", encoding="utf-8")
            mapping = root / "mapping.json"
            mapping.write_text(json.dumps({
                "episode-raw-fixture": {
                    "speaker_map": self.mapping,
                }
            }), encoding="utf-8")
            output = root / "out"
            report = run_batch(
                input_path=source, output_root=output,
                mapping_path=mapping, expected_count=1)
            self.assertEqual(report["status"], "passed")
            cleaned_jsonl = output / "cleaned_records.jsonl"
            self.assertTrue(cleaned_jsonl.is_file())
            cleaned_line = cleaned_jsonl.read_bytes()
            normalized = json.loads(
                (output / "roots/cc-train-episode-raw-fixture/unified_root.json").read_text(
                    encoding="utf-8"))
            source_meta = normalized["source"]
            self.assertEqual(source_meta["source_path"], str(cleaned_jsonl.resolve()))
            self.assertEqual(source_meta["source_file_sha256"], report["cleaned_jsonl_sha256"])
            self.assertEqual(
                source_meta["source_record_sha256"],
                __import__("hashlib").sha256(cleaned_line).hexdigest(),
            )

    def test_expanded_structure_and_hash_are_stable(self) -> None:
        original = {
            "expanded_dialogue_version": "fixture_v1",
            "root_id": "cc-train-episode-fixture",
            "turn_count": 2,
            "turns": [
                {"turn_ref": "P01:T001", "session_ref": "P01",
                 "scene_ref": None, "speaker": "Classmates A",
                 "text": "Hey, B. A bakery opened.",
                 "origin": "source_grounded"},
                {"turn_ref": "P01:T002", "session_ref": "P01",
                 "scene_ref": None, "speaker": "Classmates B",
                 "text": "Thanks, A.", "origin": "source_grounded"},
            ],
            "complete_dialogue_sha256": "0" * 64,
        }
        first, first_counts, _ = clean_expanded_dialogue(
            copy.deepcopy(original), self.mapping)
        second, second_counts, _ = clean_expanded_dialogue(
            copy.deepcopy(original), self.mapping)
        validation = validate_cleaned_record(
            original, first, self.mapping, fmt="expanded_dialogue")
        self.assertTrue(validation["valid"])
        self.assertEqual(first, second)
        self.assertEqual(first_counts, second_counts)
        self.assertEqual(first["turns"][0]["text"],
                         "Hey, Jordan. A bakery opened.")
        self.assertNotEqual(first["complete_dialogue_sha256"], "0" * 64)

    def test_batch_requires_explicit_mapping_and_is_repeatable(self) -> None:
        source = Path(
            "/root/autodl-tmp/gxy-selfplay/restart-n5-inputs/"
            "conversation_chronicles_v1/roots/cc-train-episode-178392/"
            "unified_root.json")
        if not source.is_file():
            self.skipTest("workspace fixture missing")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapping = root / "mapping.json"
            mapping.write_text(json.dumps({
                "cc-train-episode-178392": {
                    "speaker_map": self.mapping,
                }
            }), encoding="utf-8")
            output = root / "out"
            report = run_batch(
                input_path=source, output_root=output,
                mapping_path=mapping, expected_count=1)
            self.assertEqual(report["status"], "passed")
            self.assertTrue(report["global_checks"]["source_unchanged"])
            self.assertTrue(report["records"][0]["repeatable"])
            cleaned = json.loads((
                output / "roots/cc-train-episode-178392/unified_root.json"
            ).read_text(encoding="utf-8"))
            self.assertEqual(
                {turn["speaker"] for turn in cleaned["dialogue"]["turns"]},
                {"Maya", "Jordan"},
            )


if __name__ == "__main__":
    unittest.main()
