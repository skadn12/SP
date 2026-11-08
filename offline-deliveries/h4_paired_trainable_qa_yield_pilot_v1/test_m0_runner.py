"""Offline checks for source-bound M0 admission and fixed-band scoring."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import m0_runner
import review_gate
from common import sha_bytes, sha_file, write_once


class M0RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run_dir = Path(self.temp.name)
        self.root_id = "fixture-root"
        self.surface = {"root_id": self.root_id, "source_sha256": "a" * 64,
                        "turns": [
                            {"turn_ref": "P:T001", "speaker": "Ada", "text": "Plan alpha was proposed."},
                            {"turn_ref": "R01:S01:T002", "speaker": "Ada", "text": "Plan alpha was cancelled."},
                        ]}
        self.question = {"candidate_id": "fixture-ms-1", "question_type": "multi_select",
                         "question": "Which claims hold?", "options": {
                             "A": "The plan was proposed, then cancelled.",
                             "B": "The plan was cancelled, then proposed.",
                             "C": "The plan changed status.",
                             "D": "The plan never changed status."},
                         "gold": ["A", "C"]}
        self.artifact = json.dumps({"questions": [self.question]}).encode("utf-8")
        self.stage = self.run_dir / "roots" / self.root_id / "A" / "multi_select"
        self.stage.mkdir(parents=True)
        frozen = self.run_dir / "frozen" / self.root_id
        write_once(frozen / "surface.json", self.surface)
        write_once(frozen / "m0_parent.json", {"turns": self.surface["turns"][:1]})
        write_once(frozen / "m0_child.json", {"turns": self.surface["turns"][1:]})
        self.manifest = {"roots": {self.root_id: {
            "surface_path": str(frozen / "surface.json"), "source_sha256": "a" * 64,
            "parent_turn_count": 1}}, "question_bank_sha256": "b" * 64,
            "m0": {"displays": 6, "modes": ["no_context", "full_context"],
                   "seed": 20261004, "no_context_max_correct": 1,
                   "full_context_min_correct": 2, "full_context_max_correct": 4,
                   "minimum_gain": 2},
            "m0_identity": {"model_path": "/fixture/Qwen3-4B"}}
        inputs = review_gate.prepare_review_inputs(self.artifact, self.surface, "b" * 64)
        packet = inputs["packets"][0]
        review = review_gate.review_template(packet)
        review["reviewer"] = "fixture human"
        review["review_minutes"] = 4
        for row in review["option_reviews"]:
            row.update(verdict="supported" if row["label"] in self.question["gold"] else "contradicted",
                       same_scope=True, direct_evidence=True, reason="Checked both source turns.",
                       checked_facts=[
                           {"relation_key": "proposal", "ref": "P:T001", "quote": "Plan alpha was proposed."},
                           {"relation_key": "cancellation", "ref": "R01:S01:T002", "quote": "Plan alpha was cancelled."},
                       ])
        review["checks"] = {key: True for key in review_gate.CHECKS}
        review["dedup"].update(tested_relation_keys=["proposal", "cancellation"],
                               prior_bank_result="unique", within_root_result="unique",
                               reason="Distinct relation and no prior match.")
        review["decision"] = "pass_to_m0"
        result = review_gate.validate_review(packet, review, self.artifact, self.surface, "b" * 64)
        write_once(self.stage / "candidate_review.json", {"reviews": [review], "skipped": []})
        write_once(self.stage / "candidate_review_validated.json",
                   review_gate.aggregate_reviews([{"packet": packet, "review": review, "result": result}], []))

    def _admit(self, artifact: bytes) -> tuple[dict, dict]:
        with patch.object(m0_runner, "RUN", self.run_dir), patch.object(
                m0_runner, "_artifact", return_value=(artifact, None)):
            return m0_runner.admitted_candidate(
                self.manifest, self.root_id, "A", "multi_select", "fixture-ms-1")

    def test_only_exact_reviewed_original_is_admitted(self) -> None:
        question, binding = self._admit(self.artifact)
        self.assertEqual(question, self.question)
        self.assertEqual(binding["question_subtree_sha256"], review_gate._digest(self.question))
        edited = json.dumps({"questions": [{**self.question, "question": "Changed question"}]}).encode()
        with self.assertRaisesRegex(RuntimeError, "candidate_review_packet_mismatch"):
            self._admit(edited)

    def test_tampered_validated_result_is_rejected(self) -> None:
        path = self.stage / "candidate_review_validated.json"
        value = json.loads(path.read_text())
        value["reviews"][0]["result"]["question_sha256"] = "0" * 64
        path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "candidate_review_result_mismatch"):
            self._admit(self.artifact)

    def test_score_band_and_response_binding(self) -> None:
        output = self.run_dir / "m0"
        output.mkdir()
        plan = {"question_sha256": "a" * 64, "historical_source_sha256": "b" * 64,
                "realized_dialogue_sha256": "c" * 64, "display_count": 6,
                "request_count": 12, "seed": 20261004,
                "modes": ["no_context", "full_context"], "model": "/fixture/Qwen3-4B",
                "requests": []}
        results = []
        for index in range(12):
            mode = "no_context" if index % 2 == 0 else "full_context"
            variant = f"perm_{index // 2:02d}"
            correct = index == 0 or (mode == "full_context" and index // 2 < 3)
            score = {"scored": False, "correct": None} if index == 10 else {"scored": True, "correct": correct}
            request = {"request_id": f"q__{variant}__{mode}", "variant_id": variant,
                       "mode": mode, "display_to_canonical": {"A": "A"},
                       "prompt_sha256": sha_bytes(f"prompt-{index}".encode())}
            saved = {"request_id": request["request_id"], "variant_id": variant,
                     "mode": mode, "display_to_canonical": request["display_to_canonical"],
                     "question_sha256": plan["question_sha256"],
                     "historical_source_sha256": plan["historical_source_sha256"],
                     "realized_dialogue_sha256": plan["realized_dialogue_sha256"],
                     "prompt_sha256": request["prompt_sha256"], "prompt": f"prompt-{index}",
                     "score": score}
            write_once(output / f"solver_probe_response_{index:02d}.json", saved)
            plan["requests"].append(request)
            results.append({"request_id": request["request_id"], "variant_id": variant,
                            "mode": mode, "score": score})
        summary = {"status": "COMPLETE", "request_count": 12,
                   "question_sha256": plan["question_sha256"],
                   "historical_source_sha256": plan["historical_source_sha256"],
                   "realized_dialogue_sha256": plan["realized_dialogue_sha256"],
                   "results": results, "rates": {
                       "no_context": {"correct": 1, "parse_failures": 1, "total": 6},
                       "full_context": {"correct": 3, "parse_failures": 0, "total": 6}}}
        outcome = m0_runner.score_summary(summary, plan, output, self.manifest)
        self.assertTrue(outcome["m0_band_pass"])
        self.assertEqual(outcome["context_gain"], 2)
        self.assertEqual(outcome["parse_failures"]["no_context"], 1)
        saved = json.loads((output / "solver_probe_response_00.json").read_text())
        saved["question_sha256"] = "wrong"
        (output / "solver_probe_response_00.json").write_text(json.dumps(saved))
        with self.assertRaisesRegex(RuntimeError, "m0_response_binding_invalid"):
            m0_runner.score_summary(summary, plan, output, self.manifest)


if __name__ == "__main__":
    unittest.main()
