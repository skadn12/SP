"""Exercise the offline join with real file boundaries and missing responses."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from reanalyze import analyze, question_sha256, sha256


class OfflineJoinTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "source"
        self.audit_dir = Path(self.temp.name) / "audits"
        self.audit_dir.mkdir()
        self.run = self.source / "probe_v3" / "run"
        (self.run / "responses").mkdir(parents=True)
        self.question = {"candidate_id": "q1", "root_id": "train-root", "question_type": "multi_select",
                         "options": {"A": "one", "B": "two", "C": "three"}, "gold": ["A"]}
        self.write(self.source / "candidates.json", {"candidates": [self.question]})
        variant = {"variant_id": "perm_00", "options": self.question["options"],
                   "display_to_canonical": {key: key for key in self.question["options"]},
                   "displayed_gold": ["A"]}
        self.requests = []
        for mode, seed in (("no_context", 7), ("full_context", 7)):
            self.requests.append({"request_id": f"q1__perm_00__{mode}__s00", "candidate_id": "q1",
                                  "question_type": "multi_select", "variant": variant, "mode": mode,
                                  "slot": 0, "seed": seed, "prompt_sha256": f"hash-{mode}"})
        self.write(self.run / "questions.json", {"candidates": [{k: v for k, v in self.question.items() if k != "gold"}]})
        self.write(self.run / "gold.json", {"q1": self.question["gold"]})
        self.write(self.run / "source.json", {"root_id": "train-root", "turns": []})
        self.write(self.run / "manifest.json", {"model": "frozen-model", "display_protocol": "custom",
                                                 "temperature": 0, "top_p": 1, "max_tokens": 32,
                                                 "rollouts_per_position": 1, "requests": self.requests,
                                                 "input_sha256": {name: sha256(self.run / name) for name in
                                                                  ("questions.json", "gold.json", "source.json")}})
        for request, raw in zip(self.requests, ("<box></box>", "<box>A,B</box>")):
            self.write(self.run / "responses" / (request["request_id"] + ".json"),
                       {"request_id": request["request_id"], "mode": request["mode"],
                        "variant_id": "perm_00", "seed": request["seed"],
                        "prompt_sha256": request["prompt_sha256"], "model_identifier": "frozen-model",
                        "metadata": {"temperature": 0, "top_p": 1, "max_tokens": 32,
                                     "structured_output_api": "fixture_api",
                                     "structured_output_version": "fixture_v1"}, "raw_output": raw})

    def analyze(self):
        return analyze(self.source, self.run, self.audit_dir)

    def write_passing_reviews(self):
        for kind in ("quality", "split", "permutation"):
            self.write(self.audit_dir / f"{kind}_audit_revision_2.json", {
                "questions": {"q1": {"status": "PASS", "question_sha256": question_sha256(self.question),
                                       "evidence_files": [{"path": str(self.run / "source.json"),
                                                           "sha256": sha256(self.run / "source.json")}],
                                       "evidence": ["Synthetic fixture review; not an actual approval."]}}})

    @staticmethod
    def write(path: Path, value: dict) -> None:
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_full_zero_partial_gain_survives_join(self):
        row = self.analyze()["questions"]["q1"]
        self.assertEqual(row["legacy_eligibility"]["difficulty_bucket"], "full_zero_review")
        self.assertEqual(row["new_policy_reason"], "FULL_ZERO_PARTIAL_GAIN")
        self.assertTrue(row["numeric_candidate"])
        self.assertEqual(row["metrics"]["partial_context_gain"], 0.5)
        self.assertEqual(row["new_policy_candidate_status"], "DIAGNOSTIC_PENDING")

    def test_missing_paired_response_blocks_metrics(self):
        (self.run / "responses" / (self.requests[1]["request_id"] + ".json")).unlink()
        row = self.analyze()["questions"]["q1"]
        self.assertEqual(row["new_policy_reason"], "PAIRING_INCOMPLETE")
        self.assertEqual(row["legacy_eligibility"]["difficulty_bucket"], "technical_excluded")
        self.assertIsNone(row["metrics"]["partial_context_gain"])
        self.assertFalse(row["numeric_candidate"])
        self.assertIn("MISSING_FULL_CONTEXT", row["per_permutation"][0]["pair_issues"])

    def test_different_seed_does_not_block_matched_slot(self):
        self.write_passing_reviews()
        manifest = json.loads((self.run / "manifest.json").read_text())
        manifest["requests"][1]["seed"] = 8
        self.write(self.run / "manifest.json", manifest)
        response_path = self.run / "responses" / (self.requests[1]["request_id"] + ".json")
        response = json.loads(response_path.read_text())
        response["seed"] = 8
        self.write(response_path, response)
        row = self.analyze()["questions"]["q1"]
        self.assertEqual(row["pairing_check_status"], "PASS")
        self.assertEqual(row["seed_pair_warning"], "SEEDS_DIFFER_FOR_EVERY_PAIR")
        self.assertEqual(row["new_policy_candidate_status"], "TRAINING_CANDIDATE")

    def test_stale_question_review_cannot_pass(self):
        self.write_passing_reviews()
        p = self.audit_dir / "quality_audit_revision_2.json"
        review = json.loads(p.read_text())
        review["questions"]["q1"]["question_sha256"] = "older-question-version"
        self.write(p, review)
        row = self.analyze()["questions"]["q1"]
        self.assertTrue(row["numeric_candidate"])
        self.assertEqual(row["quality_check_status"], "EVIDENCE_MISMATCH")
        self.assertEqual(row["new_policy_candidate_status"], "DIAGNOSTIC_PENDING")

    def test_changed_source_blocks_comparison_and_review(self):
        self.write_passing_reviews()
        self.write(self.run / "source.json", {"different": "dialogue"})
        row = self.analyze()["questions"]["q1"]
        self.assertEqual(row["pairing_check_status"], "INCOMPLETE")
        self.assertEqual(row["quality_check_status"], "EVIDENCE_MISMATCH")

    def test_decode_config_mismatch_blocks_comparison(self):
        response_path = self.run / "responses" / (self.requests[1]["request_id"] + ".json")
        response = json.loads(response_path.read_text())
        response["metadata"]["temperature"] = 0.8
        self.write(response_path, response)
        row = self.analyze()["questions"]["q1"]
        self.assertEqual(row["pairing_check_status"], "INCOMPLETE")
        self.assertFalse(row["numeric_candidate"])


if __name__ == "__main__":
    unittest.main()
