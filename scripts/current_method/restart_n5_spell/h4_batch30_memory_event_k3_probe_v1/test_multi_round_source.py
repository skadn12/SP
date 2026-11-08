from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ..simple_append_v1.assembly import append_expansion_round, validate_appended_round
from .plan import _expanded_dialogue_hash, select_successful_expansion_source
from .test_plan import _write_expansion_fixture


class MultiRoundSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.root = _write_expansion_fixture(base, root_id="root-1", input_sha256="a" * 64)
        self.prior = json.loads((self.root / "expanded_dialogue.json").read_text())
        self.prior_path = base / "prior.json"
        self.prior_path.write_text(json.dumps(self.prior))
        self.input_hash = hashlib.sha256(self.prior_path.read_bytes()).hexdigest()
        self.expanded = append_expansion_round(prior_expanded=self.prior, child={
            "root_id": "root-1", "round_index": 2,
            "scenes": [{"scene_ref": "R02:S01", "session_ref": "r02s01",
                        "dialogue_turns": [{"turn_ref": "R02:S01:T001",
                                            "speaker": "Alex", "text": "It is booked."}]}],
        })
        self.validation = validate_appended_round(self.expanded, prior_expanded=self.prior)
        self.freeze = {
            "decision": "pass", "root_id": "root-1", "input_path": str(self.prior_path),
            "input_sha256": self.input_hash,
            "parent_dialogue_sha256": self.prior["complete_dialogue_sha256"],
        }
        self.report = {
            "status": "completed", "generation_ready": True,
            "root_id": "root-1", "run_root": str(self.root.resolve()),
            "input_sha256": self.input_hash,
        }

    def _select(self) -> dict:
        self.expanded["complete_dialogue_sha256"] = _expanded_dialogue_hash(self.expanded)
        self.report.update({"promotable_expanded_dialogue": self.expanded,
                            "promotion_validation": self.validation})
        files = {
            "run_freeze_manifest.json": self.freeze,
            "reports/MEMORY_EVENT_CONTINUATION_LIVE_REPORT.json": self.report,
            "promotion/promotable_expanded_dialogue.json": self.expanded,
            "promotion/promotable_expanded_dialogue_validation.json": self.validation,
        }
        for relative, value in files.items():
            (self.root / relative).write_text(json.dumps(value))
        return select_successful_expansion_source(
            self.root, expected_root_id="root-1", expected_input_sha256=self.input_hash,
            expected_parent_dialogue_sha256=self.prior["complete_dialogue_sha256"],
        )

    def test_accepts_actual_incremental_validation_schema(self) -> None:
        result = self._select()
        self.assertTrue(result["eligible"], result)
        self.assertTrue(result["checks"]["selected_appended_round_valid"])
        self.assertEqual(result["source_kind"], "promotion")

    def test_prior_file_drift_is_rejected(self) -> None:
        self.prior_path.write_text("{}")
        result = self._select()
        self.assertFalse(result["eligible"])
        self.assertIn("selected_prior_file_hash_matches", result["reason_codes"])

    def test_rehashed_history_rewrite_is_rejected(self) -> None:
        self.expanded["turns"][0]["text"] = "Rewritten history"
        result = self._select()
        self.assertFalse(result["eligible"])
        self.assertIn("selected_appended_round_valid", result["reason_codes"])
        self.assertTrue(result["checks"]["selected_complete_hash_matches"])

    def test_hash_chains_cannot_be_substituted(self) -> None:
        for field, reason in (
            ("prior_complete_dialogue_sha256", "selected_prior_hash_chain_matches"),
            ("parent_dialogue_sha256", "selected_original_parent_hash_matches"),
        ):
            with self.subTest(field=field):
                original = self.expanded[field]
                self.expanded[field] = "f" * 64
                result = self._select()
                self.assertFalse(result["eligible"])
                self.assertIn(reason, result["reason_codes"])
                self.expanded[field] = original

    def test_persisted_incremental_counts_are_verified(self) -> None:
        for field, reason in (
            ("prior_turn_count", "selected_validation_prior_count_matches"),
            ("new_turn_count", "selected_validation_new_count_matches"),
        ):
            with self.subTest(field=field):
                original = self.validation[field]
                self.validation[field] += 1
                result = self._select()
                self.assertFalse(result["eligible"])
                self.assertIn(reason, result["reason_codes"])
                self.validation[field] = original

    def test_failed_report_still_blocks_multi_round_source(self) -> None:
        self.report["status"] = "failed"
        result = self._select()
        self.assertFalse(result["eligible"])
        self.assertIn("report_status_completed", result["reason_codes"])


if __name__ == "__main__":
    unittest.main()
