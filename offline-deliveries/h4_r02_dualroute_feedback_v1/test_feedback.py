"""Meaningful offline provenance and request-integration tests against frozen V4."""
import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import feedback


ROOT = "cc-train-episode-103866"


class FeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory()
        cls.compiled = Path(cls.workspace.name) / "compiled"
        cls.manifest = feedback.compile_feedback(cls.compiled, list(feedback.TRANSCRIPT_SHA256))
        cls.identity = cls.manifest["target_solver_identity_sha256"]

    @classmethod
    def tearDownClass(cls):
        cls.workspace.cleanup()

    def context(self, outdir=None):
        return feedback.load_feedback_context(outdir or self.compiled, ROOT,
                                              expected_target_identity_sha256=self.identity)

    def test_all_roots_have_balanced_sourced_and_gold_free_projections(self):
        self.assertEqual(len(self.manifest["records"]), 8 * 2 * 2)
        banned = {"gold", "correct_option", "options", "items", "models", "difficulty_bucket",
                  "context_gain", "primary_transition"}

        def check_keys(value):
            if isinstance(value, dict):
                self.assertFalse(banned.intersection(value))
                for child in value.values():
                    check_keys(child)
            elif isinstance(value, list):
                for child in value:
                    check_keys(child)

        for root in self.manifest["roots"]:
            context = feedback.load_feedback_context(self.compiled, root,
                                                      expected_target_identity_sha256=self.identity)
            self.assertEqual(len(set(context["suffixes"].values())), 4)
            for (route, role), projection in context["projections"].items():
                check_keys(projection)
                self.assertEqual(projection["route"], route)
                self.assertEqual(projection["role"], role)
                self.assertEqual(projection["target_solver_model"], "M_clean_policy")
                self.assertEqual({row["question_type"] for row in projection["historical_examples"]},
                                 {"multi_select", "ordering"})
                for example in projection["historical_examples"]:
                    self.assertTrue(example["historical_stem_unverified"])
                    self.assertTrue(example["historical_evidence_refs"])
                    self.assertTrue(example["source_turn_excerpts"])
                    self.assertTrue(all(row["text"] for row in example["source_turn_excerpts"]))
                self.assertLessEqual(len(context["suffixes"][(route, role)]), feedback.MAX_SUFFIX_CHARS)

    def test_real_request_text_receipt_and_immutability_for_both_roles_routes(self):
        original = {"model": "Qwen/Qwen3.6-35B-A3B", "messages": [
            {"role": "system", "content": "Generate dialogue."},
            {"role": "user", "content": "Frozen cumulative transcript and route plan."}]}
        before = copy.deepcopy(original)
        for route in feedback.ROUTES:
            for role in feedback.ROLES:
                output, receipt = feedback.append_feedback_to_request(
                    original, self.compiled, ROOT, route, role,
                    expected_target_identity_sha256=self.identity)
                self.assertEqual(original, before)
                text = output["messages"][-1]["content"]
                suffix = self.context()["suffixes"][(route, role)]
                self.assertEqual(text, original["messages"][-1]["content"] + suffix)
                self.assertEqual(feedback.sha256_bytes(text.encode()), receipt["request_user_text_sha256"])
                self.assertEqual(feedback.digest(output), receipt["request_payload_sha256"])
                self.assertEqual(receipt["target_solver_identity_sha256"], self.identity)
                self.assertTrue(receipt["feedback_appended"])
                self.assertFalse(receipt["current_generation_feedback_used"])
                with self.assertRaisesRegex(ValueError, "already_appended"):
                    feedback.append_feedback_to_request(output, self.compiled, ROOT, route, role,
                                                        expected_target_identity_sha256=self.identity)

    def test_wrong_target_identity_model_and_root_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "caller_target_identity_mismatch"):
            feedback.load_feedback_context(self.compiled, ROOT, expected_target_identity_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "target_solver_model_mismatch"):
            feedback.load_feedback_context(self.compiled, ROOT, "M0")
        with self.assertRaisesRegex(ValueError, "unknown_feedback_root"):
            feedback.load_feedback_context(self.compiled, "cc-train-episode-not-frozen")
        with self.assertRaisesRegex(ValueError, "caller_target_identity_required"):
            feedback.append_feedback_to_request({"prompt": "dialogue"}, self.compiled,
                                                ROOT, feedback.ROUTES[0], feedback.ROLES[0])

    def test_checkpoint_identity_is_more_than_model_name(self):
        identity = self.manifest["target_solver_identity"]
        self.assertEqual(feedback.digest(identity), self.identity)
        self.assertTrue(identity["merged_model_identity"]["files"])
        self.assertTrue(identity["base_model_identity"]["files"])
        for artifact in identity["adapter_files_verified"].values():
            self.assertEqual(feedback.sha256_file(Path(artifact["path"])), artifact["sha256"])

    def test_suffix_tamper_fails_even_if_local_manifest_hash_is_rewritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "changed"
            shutil.copytree(self.compiled, out)
            relative = f"{ROOT}/old_event_continuation__planner.txt"
            path = out / relative
            path.write_text(path.read_text() + "CURRENT GENERATED FEEDBACK", encoding="utf-8")
            manifest_path = out / "COMPILED_FEEDBACK.json"
            manifest = json.loads(manifest_path.read_text())
            for record in manifest["records"]:
                if record["path"] == relative:
                    record["sha256"] = feedback.sha256_file(path)
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "projection_not_from_frozen_sources"):
                self.context(out)

    def test_v4_history_and_source_lock_tamper_are_detected(self):
        for target in (f"{ROOT}/candidate_history.json", "source_lock.json"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                source = Path(tmp) / "v4_copy"
                shutil.copytree(feedback.DEFAULT_V4_DIR, source)
                path = source / target
                path.write_bytes(path.read_bytes() + b"\n")
                with self.assertRaisesRegex(ValueError, "source_hash_mismatch"):
                    feedback.compile_feedback(Path(tmp) / "output", [ROOT], compiled_v4_dir=source)
                self.assertFalse((Path(tmp) / "output").exists())

    def test_unrelated_current_generation_files_do_not_change_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "copy"
            shutil.copytree(self.compiled, out)
            (out / "CURRENT_GENERATION_FEEDBACK.json").write_text(
                '{"new_answer":"DO NOT LOAD THIS AS PREVIOUS TRAINING FEEDBACK"}')
            self.assertEqual(self.context()["suffixes"], self.context(out)["suffixes"])

    def test_existing_delivery_never_overwritten(self):
        with self.assertRaisesRegex(ValueError, "no_overwrite"):
            feedback.compile_feedback(self.compiled, [ROOT])


if __name__ == "__main__":
    unittest.main()
