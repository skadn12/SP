from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from .. import common_sources as sources


class ObservationTests(unittest.TestCase):
    def result(self):
        row = {"root_id": "root-a", "candidate_id": "root-a:q1", "question_type": "multi_select",
               "canonical_gold": ["A", "B"], "context_mode": "full_context", "request_id": "r1",
               "display_prediction": ["B", "C"], "display_to_canonical": {"A": "C", "B": "A", "C": "B"},
               "canonical_prediction": ["A", "B"], "scored": True, "correct": True}
        second = {**deepcopy(row), "context_mode": "no_context", "request_id": "r2", "display_prediction": ["A"], "canonical_prediction": ["C"], "correct": False}
        return {"root_id": "root-a", "candidate_id": "root-a:q1", "question_type": "multi_select", "request_count": 2, "requests": [row, second]}

    def observe(self, result):
        return sources.recompute_observations(result, "multi_select", ["A", "B"], root_id="root-a", candidate_id="root-a:q1")

    def test_canonical_mapping_and_contexts_remain_separate(self):
        result = self.observe(self.result())
        self.assertEqual(result["full_context"]["canonical_prediction_counts"], [{"prediction": ["A", "B"], "count": 1}])
        self.assertEqual(result["full_context"]["correct_count"], 1)
        self.assertEqual(result["no_context"]["correct_count"], 0)
        self.assertEqual(result["no_context"]["request_count"], 1)

    def test_display_letter_cannot_be_mistaken_for_canonical(self):
        result = self.result()
        result["requests"][0]["canonical_prediction"] = ["B", "C"]
        with self.assertRaisesRegex(sources.SourceDrift, "remap_mismatch"):
            self.observe(result)

    def test_unknowns_not_fabricated_as_zero_correct(self):
        result = self.result()
        row = result["requests"][0]
        row.update(display_prediction=None, canonical_prediction=None, scored=False, correct=None)
        observed = self.observe(result)["full_context"]
        self.assertIsNone(observed["correct_count"])
        self.assertEqual(observed["unknown_unscored_count"], 1)
        self.assertEqual(observed["canonical_prediction_counts"], [{"prediction": None, "count": 1}])

    def test_unknown_context_and_duplicate_request_block(self):
        for mutation in (lambda x: x["requests"][1].update(context_mode="other"), lambda x: x["requests"][1].update(request_id="r1")):
            result = self.result()
            mutation(result)
            with self.assertRaises(sources.SourceDrift):
                self.observe(result)

    def test_inactive_history_whitelist_excludes_gold_options_counts_paths(self):
        dirty = {"root_id": "root-a", "candidate_id": "root-a:q1", "stem": "old stem", "gold": ["A"],
                 "options": {"A": "secret"}, "correct_count": 33, "source": "/secret/path",
                 "candidate_sha256": "hash", "solver_prompt_fingerprint": "fingerprint"}
        clean = sources.history_projection(dirty, status=sources.HISTORY)
        self.assertEqual(clean["stem"], "old stem")
        self.assertEqual(clean["solver_prompt_fingerprint"], "fingerprint")
        for forbidden in ("gold", "options", "correct_count", "source"):
            self.assertNotIn(forbidden, clean)

    def test_active_projection_independently_rejects_inactive_or_cross_root(self):
        for root, status in (("root-a", sources.HISTORY), ("root-b", sources.ACTIVE)):
            with self.assertRaises(sources.SourceDrift):
                sources.model_projection({"root_id": "root-a", "observation_cards": [{"root_id": root, "source_quality_status": status}]})

    def test_empty_active_feedback_is_explicit(self):
        self.assertEqual(sources.model_projection({"root_id": "root-a", "observation_cards": []}),
                         {"root_id": "root-a", "feedback_status": "NO_ACTIVE_FEEDBACK", "observation_cards": []})

    def test_actual_review_status_must_match_fixed_policy(self):
        root = "cc-train-episode-197635"
        policy = sources.SOURCE_POLICY[root]
        packet = {"root_id": root, "source_candidate_id": policy["source_candidate_id"], "historical_qa": {"question_type": "multi_select"}}
        review = {"candidate_id": policy["source_candidate_id"], "review_status": policy["expected_review_status"], "confirmed_source_defect": False, "historical_gold_changed": False}
        selected = {"root_id": root, "candidate_id": policy["source_candidate_id"], "selected": True, "manual_review": deepcopy(review)}
        self.assertEqual(sources.verify_eligibility(root, packet, review, selected, policy), sources.HISTORY)
        review["review_status"] = "REVIEWED_NO_CLEAR_GOLD_ERROR"
        with self.assertRaisesRegex(sources.SourceDrift, "review_status_mismatch"):
            sources.verify_eligibility(root, packet, review, selected, policy)

    def test_exact_source_selector_and_sha(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rows.jsonl"
            path.write_text('{"candidate_id":"x","value":1}\n{"candidate_id":"y","value":2}\n')
            reader = sources.SourceReader()
            identity = {"path": str(path), "sha256": sources.file_sha(path), "selector": {"candidate_id": "y"}}
            self.assertEqual(reader.record(identity)["value"], 2)
            with self.assertRaises(sources.SourceDrift):
                reader.record({**identity, "selector": {"candidate_id": "missing"}})
            with self.assertRaisesRegex(sources.SourceDrift, "sha_mismatch"):
                reader.record({**identity, "sha256": "0" * 64})

    def test_frozen_writes_resume_only_identical_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frozen.json"
            sources.write_frozen(path, raw=b'{"order":1}\n')
            sources.write_frozen(path, raw=b'{"order":1}\n')
            with self.assertRaises(sources.SourceDrift):
                sources.write_frozen(path, raw=b'{"order":1}')


class FrozenP1SourceTests(unittest.TestCase):
    @unittest.skipUnless((sources.P1 / "pairing_manifest.jsonl").exists(), "requires frozen local P1 corpus")
    def test_actual_six_source_qualification_and_three_input_copy(self):
        import jsonschema
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            prepared = sources.prepare_sources(run)
            self.assertEqual(prepared["status"], "PASS")
            manifest = sources.load(Path(prepared["manifest"]["path"]))
            self.assertEqual(len(manifest["roots"]), 6)
            self.assertEqual(set(prepared["inputs"]), set(sources.FIRST_ROOTS))
            schema = sources.load(Path(sources.__file__).parent / "schemas/feedback_observation_schema.json")
            for root in sources.FIRST_ROOTS:
                folder = run / "inputs" / root
                self.assertEqual((folder / "expanded_dialogue_clean.json").read_bytes(),
                                 (sources.P1 / "inputs/shared" / root / "expanded_dialogue_clean.json").read_bytes())
                payload = sources.load(folder / "observation_cards.json")
                jsonschema.validate(payload, schema)
                projected = sources.model_projection(payload)
                self.assertNotIn("/root/", json.dumps(projected))
                self.assertNotIn("reasoning_error", json.dumps(projected))
                if root.endswith("197635"):
                    self.assertEqual(payload["feedback_status"], "NO_ACTIVE_FEEDBACK")
                    self.assertEqual(payload["observation_cards"], [])
                    history = sources.load(folder / "history_only.json")
                    self.assertTrue(any(row["source_quality_status"] == sources.HISTORY for row in history["records"]))
                    for row in history["records"]:
                        self.assertNotIn("historical_gold", row)
                        self.assertNotIn("solver_observations", row)
                else:
                    card = payload["observation_cards"][0]
                    self.assertTrue(card["gold_version"]["derived"])
                    self.assertIsNone(card["source_cluster_id"])
                    self.assertEqual(set(card["solver_observations"]), set(sources.MODELS))
            # Byte-stable re-entry must preserve the same run inputs.
            self.assertEqual(sources.prepare_sources(run), prepared)


if __name__ == "__main__":
    unittest.main()
