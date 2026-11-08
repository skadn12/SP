"""Offline integration tests for exact previous-round feedback provenance."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

HERE = Path(__file__).resolve().parent

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

f = load("v91_feedback_test_adapter", HERE / "feedback_adapter.py")
c = load("v91_feedback_test_compiler", HERE / "feedback_compiler_v91.py")
FROZEN_V4_COMPILER = (HERE.parents[1] / "h4_dependency_plan_field_evidence_minfix_v94"
                      / "evidence/feedback_adapter/compiler_v4_original.py")
old = load("v4_feedback_test_original", FROZEN_V4_COMPILER)

class FeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temp.name) / "projection"
        cls.roots = list(c.ROOTS)
        cls.manifest = f.compile_feedback(cls.output, cls.roots)
        cls.identity = cls.manifest["target_solver_identity_sha256"]
    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def payload(self, root):
        return {"messages": [{"role": "user", "content": "Complete history\nINPUT_JSON\n" + json.dumps({"root_id": root})}], "model": "writer"}

    def test_all_40_feedback_receipts_and_identity(self):
        for root in self.roots:
            for route in f.ROUTES:
                for role in f.ROLES:
                    with self.subTest(root=root, route=route, role=role):
                        request = self.payload(root)
                        original = json.loads(json.dumps(request))
                        result, receipt = f.append_feedback_to_request(request, self.output, root, route, role,
                            expected_target_identity_sha256=self.identity)
                        text = result["messages"][0]["content"]
                        suffix = (self.output/root/f"{route}__{role}.txt").read_text()
                        self.assertEqual(request, original)
                        self.assertTrue(text.endswith(suffix))
                        self.assertEqual(receipt["root_id"], root)
                        self.assertEqual(receipt["request_payload_sha256"], f.digest(result))
                        self.assertEqual(receipt["request_user_text_sha256"], f.sha256_bytes(text.encode()))
                        self.assertEqual(receipt["before_user_text_sha256"], f.sha256_bytes(original["messages"][0]["content"].encode()))
                        self.assertEqual(receipt["suffix_sha256"], f.sha256_bytes(suffix.encode()))
                        self.assertEqual(receipt["source_lock_sha256"], f.sha256_file(self.output/"source_lock.json"))
                        self.assertEqual(receipt["target_solver_identity_sha256"], self.identity)
                        self.assertIs(receipt["current_generation_feedback_used"], False)
                        projection = json.loads(suffix.splitlines()[2])
                        self.assertEqual(projection["root_id"], root)
                        historical = f.read_json(f.DEFAULT_V4_DIR/root/"candidate_history.json")
                        ids = {row["candidate_id"] for row in historical}
                        self.assertTrue(set(row["candidate_id"] for row in projection["historical_examples"]) <= ids)

    def test_cross_root_payload_rejected(self):
        with self.assertRaisesRegex(ValueError, "request_feedback_root_mismatch"):
            f.append_feedback_to_request(self.payload(self.roots[0]), self.output, self.roots[1], f.ROUTES[1], "planner",
                expected_target_identity_sha256=self.identity)

    def test_missing_root_has_no_fallback(self):
        missing = "cc-train-episode-missing"
        with self.assertRaisesRegex(ValueError, "unknown_feedback_root"):
            f.load_feedback_context(self.output, missing, expected_target_identity_sha256=self.identity)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "unknown_feedback_root"):
                f.compile_feedback(Path(temp)/"output", [missing])
            self.assertFalse((Path(temp)/"output").exists())

    def test_full_history_and_old_schema_preserved(self):
        lock = f.read_json(f.DEFAULT_V4_DIR/"source_lock.json")
        matrix = old._read_jsonl(Path(lock["evaluation_sources"]["candidate_transition_matrix"]["path"]))
        exposure = {row["candidate_id"]: row for row in old._read_jsonl(Path(lock["evaluation_sources"]["candidate_training_exposure"]["path"]))}
        for root in self.roots:
            pool = {row["candidate_id"]: row for row in old._json(Path(lock["candidate_pools"][root]["path"]))}
            rows = sorted((row for row in matrix if row["root_id"] == root), key=lambda row: row["candidate_id"])
            expected = [old._candidate_history(row, pool[row["candidate_id"]], exposure[row["candidate_id"]]) for row in rows]
            actual = f.read_json(f.DEFAULT_V4_DIR/root/"candidate_history.json")
            self.assertEqual(actual, expected)
            self.assertEqual(f.read_json(f.DEFAULT_V4_DIR/root/"aggregate.json"), old._aggregate(expected))
            self.assertEqual(len(actual), c.ROOT_SPECS[root]["first_round_feedback"]["historical_candidate_count"])
            for role in ("writer", "miner", "planner"):
                for model in ("M0", "M_clean_policy", "M_inclusive_policy"):
                    suffix = old._suffix(root, old._aggregate(expected), expected, role, model)
                    self.assertEqual((f.DEFAULT_V4_DIR/root/f"{role}__{model}_feedback.txt").read_text(), suffix)

    def test_history_sha_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            clone = Path(temp)/"history"
            shutil.copytree(f.DEFAULT_V4_DIR, clone)
            p = clone/self.roots[0]/"candidate_history.json"
            p.write_text(p.read_text()+"\n")
            with self.assertRaisesRegex(ValueError, "source_hash_mismatch"):
                f._load_v4(clone, [self.roots[0]], f.TARGET)

    def test_transcript_and_evaluation_sha_bindings(self):
        lock = f.read_json(f.DEFAULT_V4_DIR/"source_lock.json")
        for root in self.roots:
            frozen = c.ROOT_SPECS[root]["first_round_feedback"]
            for key, path_field, hash_field in (
                    ("candidate_pools", "candidate_pool_path", "candidate_pool_sha256"),
                    ("historical_transcripts", "transcript_path", "transcript_sha256")):
                self.assertEqual(lock[key][root], {"path": frozen[path_field], "sha256": frozen[hash_field]})
                self.assertEqual(f._verify_artifact(lock[key][root]), lock[key][root])
        for artifact in lock["evaluation_sources"].values():
            self.assertEqual(f._verify_artifact(artifact), artifact)
        bad = dict(lock["historical_transcripts"][self.roots[0]], sha256="0"*64)
        with self.assertRaisesRegex(ValueError, "source_hash_mismatch"):
            f._verify_artifact(bad)

    def test_suffix_transplant_cannot_impersonate_root(self):
        with tempfile.TemporaryDirectory() as temp:
            clone = Path(temp)/"projection"
            shutil.copytree(self.output, clone)
            relative = f"{self.roots[0]}/{f.ROUTES[1]}__planner.txt"
            (clone/relative).write_bytes((clone/self.roots[1]/f"{f.ROUTES[1]}__planner.txt").read_bytes())
            manifest = f.read_json(clone/"COMPILED_FEEDBACK.json")
            for row in manifest["records"]:
                if row["path"] == relative:
                    row["sha256"] = f.sha256_file(clone/relative)
            (clone/"COMPILED_FEEDBACK.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "projection_not_from_frozen_sources"):
                f.load_feedback_context(clone, self.roots[0], expected_target_identity_sha256=self.identity)

    def test_wrong_target_identity_and_model_rejected(self):
        with self.assertRaisesRegex(ValueError, "caller_target_identity_mismatch"):
            f.load_feedback_context(self.output, self.roots[0], expected_target_identity_sha256="0"*64)
        with self.assertRaisesRegex(ValueError, "target_solver_model_mismatch"):
            f.load_feedback_context(self.output, self.roots[0], target_solver_model="M0", expected_target_identity_sha256=self.identity)

    def test_input_paths_complete(self):
        paths = set(f.input_paths())
        self.assertTrue(all(path.is_file() for path in paths))
        self.assertTrue({p for p in f.DEFAULT_V4_DIR.rglob("*") if p.is_file()} <= paths)
        for root in self.roots:
            for field in ("candidate_pool_path", "transcript_path", "candidate_transition_matrix_path", "candidate_training_exposure_path"):
                self.assertIn(Path(c.ROOT_SPECS[root]["first_round_feedback"][field]), paths)

if __name__ == "__main__":
    unittest.main(verbosity=2)
