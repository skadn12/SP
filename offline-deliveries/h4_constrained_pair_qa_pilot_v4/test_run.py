"""Offline checks for the frozen V4 pair-selection diagnostic."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import run


class PairSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.v3, cls.view, cls.facts, cls.baseline = run.load_frozen()
        cls.pairs = run.enumerate_pairs(cls.facts)

    def selection(self, pair_ids: list[str] | None = None) -> dict:
        ids = pair_ids or [pair["pair_id"] for pair in self.pairs[:5]]
        return {"selected_options": [
            {"pair_id": pair_id, "orientations": ["as_stated", "as_stated"] if index < 2
             else ["as_stated", "opposite"]}
            for index, pair_id in enumerate(ids)], "abstain_reason": ""}

    def materialize(self, payload: dict) -> dict:
        return run.materialize(json.dumps(payload), self.facts, self.pairs, self.view,
                               strict_json=self.v3.strict_json)

    def test_frozen_baseline_and_reviewed_ids(self) -> None:
        self.assertEqual(len(self.facts), 13)
        self.assertEqual(self.baseline["failure_reason"], run.V3_FAILURE_REASON)
        self.assertEqual(self.baseline["shape_valid_questions"], 0)
        self.assertEqual(self.baseline["source_correct_questions"], 0)
        self.assertFalse(self.baseline["training_promotion"])

    def test_curated_pool_only_joins_distinct_cross_window_turns(self) -> None:
        self.assertEqual(len(self.pairs), len(run.PAIR_RELATIONS))
        actual = {tuple(sorted(pair["fact_ids"])) for pair in self.pairs}
        self.assertEqual(actual, set(run.PAIR_RELATIONS))
        self.assertNotIn(("F006", "F018"), actual)
        self.assertNotIn(("F009", "F018"), actual)
        self.assertNotIn(("F009", "F022"), actual)
        for pair in self.pairs:
            self.assertEqual(len(set(pair["fact_ids"])), 2)
            self.assertTrue(run.cross_window(*pair["refs"]))
            self.assertEqual(pair["relation"], run.PAIR_RELATIONS[tuple(sorted(pair["fact_ids"]))])

    def test_duplicate_reviewed_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate_reviewed_fact_id"):
            run.enumerate_pairs([*self.facts, self.facts[0]])

    def test_gold_and_evidence_are_derived_from_frozen_facts(self) -> None:
        result = self.materialize(self.selection())
        self.assertEqual(result["gold"], ["A", "B"])
        self.assertEqual(result["status"], "UNREVIEWED_CANDIDATE_NOT_GOLD")
        self.assertFalse(result["training_promotion"])
        self.assertTrue(result["candidate_id"].startswith("v4-ms-"))
        by_id = {fact["fact_id"]: fact for fact in self.facts}
        for label, pair in zip("ABCDE", self.pairs[:5], strict=True):
            clauses = result["option_evidence"][label]["clauses"]
            self.assertEqual([clause["fact_id"] for clause in clauses], pair["fact_ids"])
            self.assertEqual([clause["evidence_ref"] for clause in clauses], pair["refs"])
            self.assertEqual(clauses[0]["claim"], by_id[pair["fact_ids"][0]]["claim"])
            expected_second = (by_id[pair["fact_ids"][1]]["claim"] if label in "AB"
                               else by_id[pair["fact_ids"][1]]["opposite_claim"])
            self.assertEqual(clauses[1]["claim"], expected_second)

    def test_duplicate_json_key_unknown_and_reused_pair_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate_json_key"):
            run.materialize('{"selected_options":[],"selected_options":[],"abstain_reason":"x"}',
                            self.facts, self.pairs, self.view, strict_json=self.v3.strict_json)
        payload = self.selection()
        payload["selected_options"][0]["pair_id"] = "P999"
        with self.assertRaisesRegex(ValueError, "pair_id_unknown_or_reused"):
            self.materialize(payload)
        payload = self.selection()
        payload["selected_options"][1]["pair_id"] = payload["selected_options"][0]["pair_id"]
        with self.assertRaisesRegex(ValueError, "pair_id_unknown_or_reused"):
            self.materialize(payload)

    def test_abstention_requires_reason(self) -> None:
        result = self.materialize({"selected_options": [], "abstain_reason": "No suitable five pairs."})
        self.assertEqual(result["status"], "ABSTAIN")
        self.assertEqual(result["questions"], [])
        with self.assertRaisesRegex(ValueError, "abstain_reason_missing"):
            self.materialize({"selected_options": [], "abstain_reason": " "})

    def test_tampered_prompt_is_rejected_before_api(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(run, "HERE", Path(temporary)):
            run.prepare()
            prompt = Path(temporary) / "prepared" / "prompt.txt"
            prompt.write_text(prompt.read_text(encoding="utf-8") + "tampered", encoding="utf-8")
            with mock.patch.object(self.v3, "credential_client", side_effect=AssertionError("API called")):
                with mock.patch.object(run, "v3_module", return_value=self.v3):
                    with self.assertRaisesRegex(ValueError, "frozen_prompt_or_pair_pool_mismatch"):
                        run.run()
            self.assertFalse((Path(temporary) / "prepared" / "attempt.reserved.json").exists())

    def test_invalid_model_selection_preserves_raw_response(self) -> None:
        invalid_raw = '{"selected_options":[],"selected_options":[],"abstain_reason":"x"}'
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(run, "HERE", Path(temporary)):
            run.prepare()
            fake_client = SimpleNamespace(complete=lambda **_kwargs: SimpleNamespace(
                raw_text=invalid_raw, metadata={"usage": {"total_tokens": 7}}))
            with mock.patch.object(self.v3, "credential_client", return_value=fake_client):
                with mock.patch.object(run, "v3_module", return_value=self.v3):
                    with self.assertRaisesRegex(ValueError, "duplicate_json_key"):
                        run.run()
            stage = Path(temporary) / "prepared"
            self.assertEqual((stage / "response_raw.txt").read_text(encoding="utf-8"), invalid_raw)
            self.assertEqual(json.loads((stage / "failure.json").read_text())["status"], "VALIDATION_FAILED")
            self.assertFalse((stage / "candidate.json").exists())


if __name__ == "__main__":
    unittest.main()
