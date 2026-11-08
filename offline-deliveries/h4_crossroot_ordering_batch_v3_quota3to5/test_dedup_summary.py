import unittest
from pathlib import Path
import hashlib
import json
import tempfile

import generate
from generate import normalize_candidate_pool, prompt_for


ROOT_ID = "cc-train-episode-123456"


class DedupSummaryTests(unittest.TestCase):
    def test_ordering_candidates_preserve_topic_prompt_items_and_refs(self):
        candidate = {
            "candidate_id": "old-order-1",
            "root_id": ROOT_ID,
            "question_type": "ordering",
            "topic": "Appointment planning",
            "question": "Order the statements by transcript appearance.",
            "items": {"I2": "The appointment is confirmed.", "I1": "A session is proposed."},
            "item_evidence": {
                "I1": {"turn_ref": "R01:T1"},
                "I2": {"turn_ref": "R02:T4"},
            },
        }
        summary = normalize_candidate_pool({"questions": [candidate]}, ROOT_ID)
        self.assertEqual(summary, [{
            "candidate_id": "old-order-1",
            "question": "Order the statements by transcript appearance.",
            "topic": "Appointment planning",
            "question_type": "ordering",
            "evidence_refs": ["R01:T1", "R02:T4"],
            "item_text": {"I1": "A session is proposed.", "I2": "The appointment is confirmed."},
        }])

    def test_multiselect_candidates_use_option_evidence_and_options(self):
        candidate = {
            "candidate_id": "old-multi-7",
            "root_id": ROOT_ID,
            "question_type": "multi_select",
            "question": "Which statements were explicitly reported?",
            "options": {"A": "The plan was delayed.", "B": "The meeting was canceled."},
            "option_evidence": {
                "A": {"turn_ref": "R01:T5"},
                "B": {"turn_ref": "R01:T9"},
            },
        }
        summary = normalize_candidate_pool({"questions": [candidate]}, ROOT_ID)
        self.assertEqual(summary[0]["evidence_refs"], ["R01:T5", "R01:T9"])
        self.assertEqual(summary[0]["item_text"], {
            "A": "The plan was delayed.", "B": "The meeting was canceled.",
        })
        self.assertEqual(summary[0]["topic"], "")

    def test_prompt_binds_dedup_summary_and_uncovered_evidence_rule(self):
        summary = [{"candidate_id": "old-1", "question": "Q", "topic": "T",
                    "question_type": "ordering", "evidence_refs": ["R01:T1"], "item_text": {}}]
        instructions = (Path(__file__).with_name("instructions.txt")).read_text(encoding="utf-8")
        prompt = prompt_for(ROOT_ID, {"root_id": ROOT_ID, "turn_count": 1, "turns": []},
                            instructions,
                            ["new-1"], summary)
        self.assertIn("prior_candidate_dedup_summary", prompt)
        self.assertIn("old-1", prompt)
        self.assertIn("uncovered facts and evidence elsewhere", prompt)
        self.assertIn('"turns": []', prompt)

    def test_candidate_from_another_root_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "prior_candidate_root_mismatch"):
            normalize_candidate_pool({"questions": [{"root_id": "other-root"}]}, ROOT_ID)

    def test_offline_prepare_freezes_prior_pool_and_binds_prompt(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            output = base / "delivery"
            output.mkdir()
            original_here = generate.HERE
            generate.HERE = output
            try:
                (output / "instructions.txt").write_bytes(Path(original_here, "instructions.txt").read_bytes())
                turns = [{"turn_ref": f"{ROOT_ID}:T{i}", "text": f"Source fact {i}.", "scene_ref": "scene-1"}
                         for i in range(1, 7)]
                source = {"root_id": ROOT_ID, "turn_count": len(turns), "round_count": 1,
                          "round_boundaries": [], "turns": turns}
                source_path = base / "source.json"
                source_raw = json.dumps(source, ensure_ascii=False).encode()
                source_path.write_bytes(source_raw)
                prior = {"questions": [{
                    "candidate_id": "old-order-1", "root_id": ROOT_ID, "question_type": "ordering",
                    "topic": "Previously covered", "question": "Prior question?",
                    "items": {"I1": "Already covered fact."},
                    "item_evidence": {"I1": {"turn_ref": f"{ROOT_ID}:T1"}},
                }]}
                prior_path = base / "prior.json"
                prior_raw = json.dumps(prior, ensure_ascii=False, indent=2).encode()
                prior_path.write_bytes(prior_raw)
                manifest = {"sources": [{
                    "root_id": ROOT_ID, "source_path": str(source_path),
                    "source_sha256": hashlib.sha256(source_raw).hexdigest(),
                    "prior_candidates_path": str(prior_path),
                    "prior_candidates_sha256": hashlib.sha256(prior_raw).hexdigest(),
                }]}
                manifest_path = base / "manifest.json"
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

                generate.prepare(manifest_path)

                root_dir = output / ROOT_ID
                self.assertEqual((root_dir / "prior_candidates.json").read_bytes(), prior_raw)
                summary = json.loads((root_dir / "PRIOR_CANDIDATE_SUMMARY.json").read_text())
                self.assertEqual(summary["original_pool_sha256"], hashlib.sha256(prior_raw).hexdigest())
                _, _, frozen, _ = generate.frozen_request(ROOT_ID, "initial")
                self.assertIn("Previously covered", frozen["prompt"])
                self.assertIn("Source fact 6.", frozen["prompt"])
            finally:
                generate.HERE = original_here

    def test_offline_topup_includes_original_pool_and_initial_candidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            output = base / "delivery"
            output.mkdir()
            original_here = generate.HERE
            generate.HERE = output
            try:
                (output / "instructions.txt").write_bytes(Path(original_here, "instructions.txt").read_bytes())
                turns = [{"turn_ref": f"{ROOT_ID}:T{i}", "text": f"Source fact {i}.", "scene_ref": "scene-1"}
                         for i in range(1, 7)]
                source = {"root_id": ROOT_ID, "turn_count": len(turns), "round_count": 1,
                          "round_boundaries": [], "turns": turns}
                source_path = base / "source.json"
                source_raw = json.dumps(source, ensure_ascii=False).encode()
                source_path.write_bytes(source_raw)
                prior = {"questions": []}
                prior_path = base / "prior.json"
                prior_raw = json.dumps(prior).encode()
                prior_path.write_bytes(prior_raw)
                manifest = {"sources": [{
                    "root_id": ROOT_ID, "source_path": str(source_path),
                    "source_sha256": hashlib.sha256(source_raw).hexdigest(),
                    "prior_candidates_path": str(prior_path),
                    "prior_candidates_sha256": hashlib.sha256(prior_raw).hexdigest(),
                }]}
                manifest_path = base / "manifest.json"
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                generate.prepare(manifest_path)

                candidate = {
                    "candidate_id": ROOT_ID + "-order-001", "root_id": ROOT_ID,
                    "question_type": "ordering", "topic": "Initial topic", "question": "Initial question?",
                    "items": {f"I{i}": f"Source fact {i}." for i in range(1, 7)},
                    "item_evidence": {f"I{i}": {"turn_ref": f"{ROOT_ID}:T{i}"} for i in range(1, 7)},
                }
                root_dir = output / ROOT_ID
                (root_dir / "initial" / "VALIDATION.json").write_text(json.dumps({"stage": "initial"}))
                (root_dir / "initial" / "VALIDATED_CANDIDATES.json").write_text(
                    json.dumps({"questions": [candidate]}))

                generate.prepare_topup(ROOT_ID, "initial result leaves distinct source evidence")
                _, _, request, _ = generate.frozen_request(ROOT_ID, "topup")
                summary = generate.read_summary_file(root_dir / "topup" / "TOPUP_DEDUP_SUMMARY.json", ROOT_ID)
                self.assertEqual(request["candidate_ids"], [ROOT_ID + "-order-004", ROOT_ID + "-order-005"])
                self.assertEqual(len(summary), 1)
                self.assertEqual(summary[0]["topic"], "Initial topic")
                self.assertIn("Initial topic", request["prompt"])
                self.assertIn("Source fact 6.", request["prompt"])
                self.assertFalse((root_dir / "topup" / "RESPONSE.json").exists())
            finally:
                generate.HERE = original_here


if __name__ == "__main__":
    unittest.main()
