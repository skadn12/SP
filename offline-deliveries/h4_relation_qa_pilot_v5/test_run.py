"""Offline structural and provenance checks for V5; fixtures are not source reviews."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import run


class RelationPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.v3 = run.v3_module()
        cls.view = run.surface(cls.v3)
        cls.turns = {row["turn_ref"]: row for row in cls.view["turns"]}

    def relation(self, index: int, refs: tuple[str, str] = ("G02:S01:T003", "G02:S02:T001")) -> dict:
        return {"relation_id": f"R{index:03d}", "relation_type": "request_withdrawal",
                "actor": "Emily", "object_id": f"synthetic-{index}", "version": "", "scope": "the request and its status",
                "claim": f"Synthetic relation {index} holds.",
                "opposite_claim": f"Synthetic relation {index} does not hold.",
                "checkpoints": [{"ref": ref, "speaker": self.turns[ref]["speaker"],
                                 "quote": self.turns[ref]["text"]} for ref in refs]}

    def payload(self, count: int = 5) -> dict:
        return {"relations": [self.relation(i) for i in range(1, count + 1)],
                "insufficiency_reason": "" if count >= 5 else "Fewer than five relations."}

    def test_source_is_frozen_and_prompt_hides_metadata(self) -> None:
        self.assertEqual(self.view["history_sha256"], self.v3.R03_SHA256)
        self.assertEqual(self.view["g02_sha256"], self.v3.G02_SHA256)
        self.assertEqual(len(self.view["turns"]), 21)
        prompt = run.extraction_prompt(self.view)
        self.assertNotIn("event_key", prompt)
        self.assertNotIn("relative_time", prompt)
        self.assertNotIn("source_graph_sha256", prompt)

    def test_extraction_requires_two_exact_cross_window_quotes(self) -> None:
        valid = run.validate_extraction(json.dumps(self.payload()), self.view, strict_json=self.v3.strict_json)
        self.assertEqual(len(valid["relations"]), 5)
        self.assertFalse(valid["training_promotion"])
        bad = self.payload()
        bad["relations"][0]["checkpoints"][1]["quote"] = "NOT IN SPOKEN TURN"
        with self.assertRaisesRegex(ValueError, "checkpoint_speaker_or_quote_mismatch"):
            run.validate_extraction(json.dumps(bad), self.view, strict_json=self.v3.strict_json)
        bad = self.payload()
        bad["relations"][0]["checkpoints"][1] = copy.deepcopy(bad["relations"][0]["checkpoints"][0])
        with self.assertRaisesRegex(ValueError, "checkpoint_ref_invalid_or_reused"):
            run.validate_extraction(json.dumps(bad), self.view, strict_json=self.v3.strict_json)
        bad = self.payload()
        bad["relations"][0] = self.relation(1, ("G02:S01:T002", "G02:S01:T003"))
        with self.assertRaisesRegex(ValueError, "relation_not_cross_window"):
            run.validate_extraction(json.dumps(bad), self.view, strict_json=self.v3.strict_json)

    def test_duplicate_relation_and_ref_leak_rejected(self) -> None:
        bad = self.payload()
        bad["relations"][1]["relation_id"] = "R001"
        with self.assertRaisesRegex(ValueError, "relation_id_invalid_or_duplicate"):
            run.validate_extraction(json.dumps(bad), self.view, strict_json=self.v3.strict_json)
        bad = self.payload()
        bad["relations"][1]["object_id"] = bad["relations"][0]["object_id"]
        with self.assertRaisesRegex(ValueError, "duplicate_relation_key"):
            run.validate_extraction(json.dumps(bad), self.view, strict_json=self.v3.strict_json)
        bad = self.payload()
        bad["relations"][0]["claim"] += " G02:S01:T003"
        with self.assertRaisesRegex(ValueError, "turn_ref_leaked_into_proposition"):
            run.validate_extraction(json.dumps(bad), self.view, strict_json=self.v3.strict_json)
        with self.assertRaisesRegex(ValueError, "duplicate_json_key"):
            run.validate_extraction('{"relations":[],"relations":[],"insufficiency_reason":"x"}',
                                    self.view, strict_json=self.v3.strict_json)

    def test_review_is_hash_bound_and_cannot_rewrite_raw(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(run, "HERE", Path(temporary)):
            stage = Path(temporary) / "extraction"
            stage.mkdir()
            raw = json.dumps(self.payload())
            (stage / "response_raw.txt").write_text(raw, encoding="utf-8")
            pending = run.validate_extraction(raw, self.view, strict_json=self.v3.strict_json)
            run.write_new(stage / "pending_relations.json", pending)
            assessments = [{"relation_id": f"R{i:03d}", "decision": "supported_nontrivial",
                            "reason": "Synthetic fixture for review plumbing."} for i in range(1, 6)]
            review = {"status": "SOURCE_REVIEWED",
                      "attestation": "TWO_SPAN_CLAIM_AND_STRICT_CONTRADICTION_REVIEWED",
                      "reviewer": "test-only", "pending_relations_sha256": run.sha((stage / "pending_relations.json").read_bytes()),
                      "assessments": assessments}
            run.write_new(Path(temporary) / "reviewed_relations.json", review)
            accepted, _, _ = run.reviewed_relations(self.view, strict_json=self.v3.strict_json)
            self.assertEqual(len(accepted), 5)
            edited = copy.deepcopy(pending)
            edited["relations"][0]["claim"] = "Human rewrite."
            (stage / "pending_relations.json").write_text(json.dumps(edited), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pending_relations_not_raw_extraction"):
                run.reviewed_relations(self.view, strict_json=self.v3.strict_json)

    def test_composition_derives_gold_and_rejects_reuse(self) -> None:
        relations = self.payload()["relations"]
        selection = {"selected_options": [
            {"relation_id": f"R{i:03d}", "orientation": "as_stated" if i in (1, 3) else "opposite"}
            for i in range(1, 6)], "abstain_reason": ""}
        candidate = run.materialize(json.dumps(selection), relations, self.view, strict_json=self.v3.strict_json)
        self.assertEqual(candidate["gold"], ["A", "C"])
        self.assertEqual(candidate["options"]["B"], relations[1]["opposite_claim"])
        self.assertEqual(len(candidate["option_evidence"]["A"]["checkpoints"]), 2)
        self.assertEqual(candidate["status"], "UNREVIEWED_CANDIDATE_NOT_GOLD")
        selection["selected_options"][1]["relation_id"] = "R001"
        with self.assertRaisesRegex(ValueError, "relation_id_unknown_or_reused"):
            run.materialize(json.dumps(selection), relations, self.view, strict_json=self.v3.strict_json)
        abstain = run.materialize(json.dumps({"selected_options": [], "abstain_reason": "No five independent options."}),
                                  relations, self.view, strict_json=self.v3.strict_json)
        self.assertEqual(abstain["status"], "ABSTAIN")

    def test_frozen_prompt_tamper_blocks_api_and_raw_failure_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(run, "HERE", Path(temporary)):
            run.prepare_extraction()
            prompt = Path(temporary) / "extraction" / "prompt.txt"
            original = prompt.read_text(encoding="utf-8")
            prompt.write_text(original + "tampered", encoding="utf-8")
            with mock.patch.object(self.v3, "credential_client", side_effect=AssertionError("API called")):
                with mock.patch.object(run, "v3_module", return_value=self.v3):
                    with self.assertRaisesRegex(ValueError, "frozen_manifest_mismatch"):
                        run.run_stage("extraction")
            self.assertFalse((Path(temporary) / "extraction" / "attempt.reserved.json").exists())
            prompt.write_text(original, encoding="utf-8")
            invalid_raw = '{"relations":[],"relations":[],"insufficiency_reason":"x"}'
            fake = SimpleNamespace(complete=lambda **_kwargs: SimpleNamespace(raw_text=invalid_raw, metadata={"usage": {"total_tokens": 7}}))
            with mock.patch.object(self.v3, "credential_client", return_value=fake):
                with mock.patch.object(run, "v3_module", return_value=self.v3):
                    with self.assertRaisesRegex(ValueError, "duplicate_json_key"):
                        run.run_stage("extraction")
            self.assertEqual((Path(temporary) / "extraction" / "response_raw.txt").read_text(encoding="utf-8"), invalid_raw)
            self.assertEqual(json.loads((Path(temporary) / "extraction" / "failure.json").read_text())["status"], "VALIDATION_FAILED")


if __name__ == "__main__":
    unittest.main()
