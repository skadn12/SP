"""Offline v24 -> existing QA bridge checks. No provider or Solver calls."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import generate as adapter


class SequentialGenerationAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.expansion = self.base / "expansion"
        self.expansion.mkdir()
        self.root = "fixture-root"
        self.source = self.expansion / "sources_merged" / (self.root + ".json")
        self.turns = [
            {"turn_ref": "P:T001", "speaker": "Alex", "text": "The original history."},
            {"turn_ref": "R02:A:S01:T001", "speaker": "Sam", "text": "Accepted A progress."},
            {"turn_ref": "R02:B:S01:T001", "speaker": "Alex", "text": "Accepted independent B event."},
        ]
        adapter.write(self.source, {"root_id": self.root, "turns": deepcopy(self.turns)})
        self.merge = self.expansion / "MERGE_MANIFEST.json"
        adapter.write(self.merge, {"version": adapter.EXPANSION_VERSION, "status": "PASS",
            "sequence": "frozen_base -> accepted_A -> accepted_B",
            "source_rows": [{"root_id": self.root, "path": str(self.source),
                            "sha256": adapter.base.digest(self.source.read_bytes()),
                            "accepted_scenes": 2, "accepted_A_scenes": 1,
                            "accepted_B_scenes": 1, "shortfall_scenes": 4, "status": "PASS"}],
            "files": adapter.base.bindings([self.source])})
        self.audit = self.expansion / "FINAL_AUDIT.json"
        adapter.write(self.audit, {"status": "PASS", "failures": [], "run": str(self.expansion),
            "run_status": "COMPLETE_WITH_SHORTFALL", "roots": [self.root], "model": adapter.MODEL})
        adapter.write(self.expansion / "MANIFEST.json", {
            "version": adapter.EXPANSION_VERSION, "roots": [self.root],
            "target_solver_model": adapter.TARGET, "files": []})

    def prepare_offline_fixture(self):
        feedback = self.base / "feedback"
        adapter.write(feedback / "source_lock.json", {
            "status": "EXPLICIT_PREVIOUS_TRAINING_ROUND_ONLY",
            "merge_current_generation_feedback": False,
            "model_identity_source": {"sha256": adapter.v15.evaluator.TRAINED_AUDIT_SHA256}})
        prior = self.base / "prior.jsonl"
        prior.write_text(json.dumps({"candidate_id": "earlier-question", "root_id": self.root,
            "question_type": "multi_select", "question": "Prior prompt",
            "options": {"A": "A previous statement"}, "gold": ["A"],
            "full_partial_mean": 0.5, "solver_output": "DO NOT LEAK"}) + "\n")
        run = self.base / "generation"
        with patch.object(adapter, "_dependency_paths", return_value=({}, {Path(adapter.__file__)})), \
             patch.object(adapter, "DEFAULT_PRIOR", prior), \
             patch.object(adapter.legacy, "FEEDBACK", feedback), \
             patch.object(adapter.v15, "prior_candidates", return_value=([], [])), \
             patch.object(adapter.legacy.feedback, "build_feedback_suffix", return_value="frozen"), \
             patch.object(adapter.legacy.Caller, "call", side_effect=AssertionError("No provider calls in prepare")):
            result = adapter.prepare(run, self.merge, slots=2)
        return run, result

    def test_accepted_partial_sequential_context_remains_one_unchanged_source(self):
        manifest, sources, excluded = adapter.checked_sources(self.merge)
        self.assertEqual(excluded, [])
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["source"]["turns"], self.turns)
        self.assertEqual(sources[0]["accepted_scenes"], 2)
        self.assertEqual(sources[0]["merge_record"]["shortfall_scenes"], 4)
        self.assertIn("Each later scene saw all accepted", adapter.MERGED_INSTRUCTION)
        self.assertNotIn("independently from the same frozen base", adapter.MERGED_INSTRUCTION)
        self.assertIn("diagnostic, not an admission requirement", adapter.MERGED_INSTRUCTION)

    def test_final_audit_must_finish_before_prepare_creates_any_run(self):
        self.audit.unlink()
        run = self.base / "generation"
        with self.assertRaisesRegex(ValueError, "final_audit_missing"):
            adapter.prepare(run, self.merge)
        self.assertFalse(run.exists())

    def test_failed_or_other_run_audit_cannot_authorize_source(self):
        original = adapter.read(self.audit)
        for changes, code in (
            ({"status": "FAIL"}, "final_audit_not_pass"),
            ({"failures": ["changed_source"]}, "final_audit_not_pass"),
            ({"run": str(self.base / "other")}, "audit_run_mismatch"),
            ({"run_status": "RUNNING"}, "not_complete"),
            ({"roots": ["other-root"]}, "audit_root_mismatch"),
        ):
            with self.subTest(changes=changes):
                self.audit.write_text(json.dumps({**original, **changes}))
                with self.assertRaisesRegex(ValueError, code):
                    adapter.checked_sources(self.merge)
        self.audit.write_text(json.dumps(original))

    def test_nonsequential_or_old_version_manifest_is_rejected(self):
        original = adapter.read(self.merge)
        for changes, code in (
            ({"version": "h4_sequential_ab_v23"}, "version_mismatch"),
            ({"sequence": "independent A and B"}, "order_mismatch"),
            ({"status": "PENDING"}, "merge_not_pass"),
        ):
            with self.subTest(changes=changes):
                self.merge.write_text(json.dumps({**original, **changes}))
                with self.assertRaisesRegex(ValueError, code):
                    adapter.checked_sources(self.merge)

    def test_prepare_binds_final_audit_and_preserves_existing_qa_policy(self):
        run, result = self.prepare_offline_fixture()
        manifest = adapter.verify(run)
        bound = {row["path"] for row in manifest["files"]}
        self.assertIn(str(self.audit), bound)
        self.assertIn(str(self.merge), bound)
        self.assertEqual(manifest["expansion_final_audit_sha256"], adapter.base.digest(self.audit.read_bytes()))
        self.assertEqual(manifest["source_rows"][0]["source_sha256"], adapter.base.digest(self.source.read_bytes()))
        self.assertEqual(adapter.read(run / "sources" / self.source.name)["turns"], self.turns)
        self.assertFalse(manifest["branch_separated_question_pools"])
        self.assertFalse(manifest["semantic_multi_review"])
        self.assertTrue(manifest["ordering_independent_qwen_review"])
        self.assertEqual(manifest["selection_rule"], "complete_valid_full_set and 0 < full_partial_mean < 1")
        self.assertEqual(manifest["ground_calls"], 0)
        self.assertEqual(manifest["budget"], adapter.v15.budget(1, 2))
        prompt_rows = adapter.prompt_prior(adapter.read(run / "prior" / (self.root + ".json")))
        prompt_text = json.dumps(prompt_rows)
        self.assertNotIn("DO NOT LEAK", prompt_text)
        self.assertNotIn("full_partial_mean", prompt_text)
        self.assertNotIn("gold", prompt_text.lower())

    def test_mutating_audit_or_source_after_prepare_invalidates_generation(self):
        run, _ = self.prepare_offline_fixture()
        old = self.audit.read_bytes()
        self.audit.write_bytes(old + b" ")
        with self.assertRaises(ValueError):
            adapter.verify(run)
        self.audit.write_bytes(old)
        self.source.write_bytes(self.source.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            adapter.verify(run)

    def test_default_prior_is_current_v17_thirty_question_pool(self):
        self.assertEqual(adapter.DEFAULT_PRIOR,
            adapter.DELIVERIES / "h4_root_merged_context_v17/runs/merged_latest_02/training_pool/TRAINING_POOL.jsonl")
        self.assertEqual(len([line for line in adapter.DEFAULT_PRIOR.read_text().splitlines() if line.strip()]), 30)

    def test_generation_validation_review_and_dedup_reuse_v17_unchanged(self):
        previous = adapter.DELIVERIES / "h4_root_merged_context_v17/generate.py"
        historical = ast.parse(previous.read_text().replace(
            'f"{root}-v17-{tag}-merged-{code}-b01-{index:02d}"',
            'f"{root}-v24-{tag}-merged-{code}-b01-{index:02d}"'))
        current = ast.parse(Path(adapter.__file__).read_text())
        old_functions = {node.name: node for node in historical.body if isinstance(node, ast.FunctionDef)}
        new_functions = {node.name: node for node in current.body if isinstance(node, ast.FunctionDef)}
        for name in ("generate", "make_payload", "prompt_prior", "_prior_pool_rows"):
            with self.subTest(function=name):
                self.assertEqual(ast.dump(old_functions[name]), ast.dump(new_functions[name]))


if __name__ == "__main__":
    unittest.main()
