from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.conversation_chronicles_adapter import (
    build_unified_root, prepare_cc_roots, select_cc_records,
    validate_cc_source_record,
)
from scripts.current_method.restart_n5_spell.prepare_conversation_chronicles_roots import (
    run,
)
from scripts.current_method.restart_n5_spell.unified_dialogue_root import (
    read_unified_parent, validate_unified_dialogue_root,
    validate_unified_root_collection,
)


SESSION_NAMES = ("first", "second", "third", "fourth", "fifth")


def _record(index: int) -> dict:
    row = {
        "dataID": f"episode-{index}",
        "relationship": "Colleagues",
        "time_interval": [
            "Start", "A day after", "A week after", "A month after",
            "A year after",
        ],
        "summary": [f"summary {index} session {session}"
                    for session in range(1, 6)],
        # This field proves that the adapter does not propagate source QA.
        "qa": [{"question": f"secret question {index}", "answer": "secret"}],
    }
    for session, name in enumerate(SESSION_NAMES, 1):
        row[f"{name}_session_dialogue"] = [
            f"episode {index} session {session} turn {turn}"
            for turn in range(1, 4)]
        row[f"{name}_session_speakers"] = [
            "Alex" if turn % 2 else "Blair" for turn in range(1, 4)]
    return row


def _write_source(path: Path, count: int = 15) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            handle.write(json.dumps(_record(index), ensure_ascii=False) + "\n")


class ConversationChroniclesAdapterTests(unittest.TestCase):
    def test_source_record_requires_five_aligned_sessions(self) -> None:
        valid = _record(1)
        self.assertTrue(validate_cc_source_record(valid)["valid"])
        missing = copy.deepcopy(valid)
        del missing["fifth_session_dialogue"]
        self.assertFalse(validate_cc_source_record(missing)["valid"])
        mismatched = copy.deepcopy(valid)
        mismatched["third_session_speakers"].pop()
        result = validate_cc_source_record(mismatched)
        self.assertFalse(result["valid"])
        self.assertIn("dialogue_speaker_arrays_aligned", result["errors"])

    def test_fixed_selection_is_reproducible_and_unique(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "train.jsonl"
            _write_source(source)
            first = select_cc_records(source, count=10, seed=20260721)
            second = select_cc_records(source, count=10, seed=20260721)
            first_ids = [row["source_id"] for row in first["records"]]
            second_ids = [row["source_id"] for row in second["records"]]
            self.assertEqual(first_ids, second_ids)
            self.assertEqual(len(first_ids), 10)
            self.assertEqual(len(set(first_ids)), 10)
            self.assertEqual(first["source_file_sha256"],
                             second["source_file_sha256"])

    def test_adapter_preserves_order_and_excludes_qa(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "train.jsonl"
            _write_source(source)
            prepared = prepare_cc_roots(source)
            root = prepared["roots"][0]
            self.assertEqual(root["dialogue"]["session_count"], 5)
            self.assertEqual(root["dialogue"]["turn_count"], 15)
            self.assertTrue(all(turn["partition"] == "parent"
                                for turn in root["dialogue"]["turns"]))
            self.assertNotIn("secret question", json.dumps(root))
            indices = [(turn["source_session_index"],
                        turn["source_turn_index"])
                       for turn in root["dialogue"]["turns"]]
            self.assertEqual(indices, sorted(indices))
            self.assertTrue(prepared["collection_validation"]["valid"])
            self.assertTrue(all(all(checks.values()) for checks in
                                prepared["source_mapping_checks"]))

    def test_unified_root_validator_detects_contract_violations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "train.jsonl"
            _write_source(source)
            root = prepare_cc_roots(source)["roots"][0]
            self.assertTrue(validate_unified_dialogue_root(root)["valid"])

            bad_parent = copy.deepcopy(root)
            bad_parent["dialogue"]["turns"][0]["partition"] = "child"
            result = validate_unified_dialogue_root(bad_parent)
            self.assertIn("all_parent", result["errors"])

            bad_session = copy.deepcopy(root)
            bad_session["session_metadata"][0]["turn_ids"].pop()
            result = validate_unified_dialogue_root(bad_session)
            self.assertIn("session_turn_ids_exact", result["errors"])

            bad_hash = copy.deepcopy(root)
            bad_hash["dialogue"]["dialogue_hash"] = "0" * 64
            result = validate_unified_dialogue_root(bad_hash)
            self.assertIn("dialogue_hash_recomputed", result["errors"])

            bad_order = copy.deepcopy(root)
            bad_order["dialogue"]["turns"][0:2] = reversed(
                bad_order["dialogue"]["turns"][0:2])
            result = validate_unified_dialogue_root(bad_order)
            self.assertIn("turn_order_preserved", result["errors"])

            mixed = copy.deepcopy(root)
            mixed["dialogue"]["turns"][0]["source_id"] = "episode-other"
            result = validate_unified_dialogue_root(mixed)
            self.assertIn("single_source_only", result["errors"])

            empty_speaker = copy.deepcopy(root)
            empty_speaker["dialogue"]["turns"][0]["speaker"] = ""
            result = validate_unified_dialogue_root(empty_speaker)
            self.assertIn("speaker_text_nonempty", result["errors"])

    def test_collection_validator_detects_global_identity_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "train.jsonl"
            _write_source(source)
            prepared = prepare_cc_roots(source)
            roots = copy.deepcopy(prepared["roots"])
            roots[1]["root_id"] = roots[0]["root_id"]
            result = validate_unified_root_collection(roots)
            self.assertFalse(result["valid"])
            self.assertFalse(result["global_checks"][
                "root_ids_globally_unique"])

            roots = copy.deepcopy(prepared["roots"])
            roots[1]["dialogue"]["turns"][0]["turn_id"] = (
                roots[0]["dialogue"]["turns"][0]["turn_id"])
            result = validate_unified_root_collection(roots)
            self.assertFalse(result["global_checks"][
                "turn_ids_globally_unique"])

            roots = copy.deepcopy(prepared["roots"])
            roots[1]["source"]["source_id"] = roots[0]["source"]["source_id"]
            result = validate_unified_root_collection(roots)
            self.assertFalse(result["global_checks"]["source_ids_unique"])

    def test_root_reader_emits_only_minimum_parent_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "train.jsonl"
            _write_source(source)
            root = prepare_cc_roots(source)["roots"][0]
            path = Path(directory) / "unified_root.json"
            path.write_text(json.dumps(root), encoding="utf-8")
            parent = read_unified_parent(path)
            self.assertEqual(set(parent), {
                "sample_id", "lineage_group", "dialogue"})
            self.assertEqual(set(parent["dialogue"]), {
                "dialogue_hash", "turns"})
            self.assertEqual(set(parent["dialogue"]["turns"][0]), {
                "turn_id", "session_id", "speaker", "text", "partition"})
            self.assertEqual(parent["dialogue"]["dialogue_hash"],
                             root["dialogue"]["dialogue_hash"])

    def test_ten_root_dry_run_serializes_and_reloads_losslessly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "train.jsonl"
            output = base / "output"
            _write_source(source)
            report = run(source_path=source, output_root=output)
            self.assertTrue(report["adapter_ready"])
            self.assertTrue(report["schema_ready"])
            self.assertTrue(report["root_reader_ready"])
            self.assertFalse(report["generation_ready"])
            self.assertTrue(all(report["global_checks"].values()))
            self.assertEqual(len(report["roots"]), 10)
            self.assertEqual(report["api_calls"], 0)
            for row in report["roots"]:
                self.assertEqual(row["session_count"], 5)
                self.assertEqual(row["turn_count"], 15)
                self.assertTrue(row["serialization_round_trip_valid"])
                self.assertTrue(row["root_reader_valid"])
                self.assertTrue(Path(row["path"]).is_file())
            manifest = json.loads(
                (output / "selection_manifest.json").read_text())
            self.assertEqual(manifest["selected_root_count"], 10)
            self.assertFalse(manifest["external_api_used"])
            with self.assertRaises(FileExistsError):
                run(source_path=source, output_root=output)


if __name__ == "__main__":
    unittest.main()
