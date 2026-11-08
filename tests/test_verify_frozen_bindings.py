"""Synthetic checks for the standalone frozen-binding verifier."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "tools/verify_frozen_bindings.py"
spec = importlib.util.spec_from_file_location("verify_frozen_bindings", SCRIPT)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class VerifyBindingsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "input.txt"
        self.data.write_bytes(b"synthetic fixture\n")
        self.digest = hashlib.sha256(self.data.read_bytes()).hexdigest()

    def manifest(self, value, name="manifest.json"):
        path = self.root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def issues(self, result):
        return {item["problem"] for item in result["issues"]}

    def test_nested_and_duplicate_bindings_hash_once(self):
        record = {"path": str(self.data), "sha256": self.digest, "size_bytes": 18}
        path = self.manifest({"first": record, "nested": [{"again": record}]})
        result = verifier.verify([path])
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["binding_occurrences"], 2)
        self.assertEqual(result["unique_resolved_paths"], 1)
        self.assertEqual(result["matched_files"], 1)

    def test_nesting_can_be_disabled(self):
        record = {"path": str(self.data), "sha256": self.digest}
        root_list = self.manifest([record, {"nested": record}])
        self.assertEqual(verifier.verify([root_list], recursive=False)["binding_occurrences"], 1)
        root_object = self.manifest({"nested": record}, "object.json")
        result = verifier.verify([root_object], recursive=False)
        self.assertIn("no_bindings_found", self.issues(result))

    def test_relative_path_requires_base_and_confines_symlinks(self):
        path = self.manifest({"path": "input.txt", "sha256": self.digest})
        self.assertIn("relative_base_required", self.issues(verifier.verify([path])))
        self.assertEqual(verifier.verify([path], relative_base=self.root)["status"], "PASS")
        escaped = self.manifest({"path": "../outside", "sha256": self.digest})
        self.assertIn("relative_path_escape", self.issues(verifier.verify([escaped], relative_base=self.root)))
        with tempfile.TemporaryDirectory() as outside:
            (self.root / "link").symlink_to(outside, target_is_directory=True)
            via_link = self.manifest({"path": "link/input", "sha256": self.digest})
            self.assertIn("relative_path_escape", self.issues(verifier.verify([via_link], relative_base=self.root)))

    def test_absolute_longest_prefix_mapping_and_boundary(self):
        inner = self.root / "inner"
        inner.mkdir()
        (inner / "input.txt").write_bytes(self.data.read_bytes())
        path = self.manifest({"path": "/old/project/input.txt", "sha256": self.digest})
        maps = [(Path("/old"), self.root / "wrong"), (Path("/old/project"), inner)]
        self.assertEqual(verifier.verify([path], mappings=maps)["status"], "PASS")
        self.assertEqual(verifier.resolve_binding("/old/project2/file", None, maps), self.root / "wrong/project2/file")
        self.assertEqual(verifier.resolve_binding("/older/file", None, maps), Path("/older/file"))

    def test_conflicting_hashes_and_sizes_fail_closed(self):
        path = self.manifest([
            {"path": str(self.data), "sha256": self.digest},
            {"path": str(self.data), "sha256": "0" * 64},
        ])
        self.assertIn("conflicting_bindings", self.issues(verifier.verify([path])))
        path = self.manifest([
            {"path": str(self.data), "sha256": self.digest, "size_bytes": 18},
            {"path": str(self.data), "sha256": self.digest, "size_bytes": 19},
        ])
        self.assertIn("conflicting_bindings", self.issues(verifier.verify([path])))

    def test_missing_directory_hash_and_size_mismatch(self):
        path = self.manifest([
            {"path": str(self.root / "missing"), "sha256": self.digest},
            {"path": str(self.root), "sha256": self.digest},
            {"path": str(self.data), "sha256": "f" * 64},
        ])
        self.assertEqual(self.issues(verifier.verify([path])), {"missing", "not_regular_file", "sha256_mismatch"})
        path = self.manifest({"path": str(self.data), "sha256": self.digest, "size_bytes": 1})
        self.assertIn("size_mismatch", self.issues(verifier.verify([path])))

    def test_bad_binding_and_unreadable_manifest_are_errors(self):
        path = self.manifest([{"path": str(self.data), "sha256": "wrong"},
                              {"path": str(self.data), "sha256": self.digest, "size_bytes": True}])
        result = verifier.verify([path, self.root / "absent.json"])
        self.assertEqual(self.issues(result), {"invalid_binding", "invalid_size", "manifest_unreadable_or_invalid"})
        self.assertEqual(result["status"], "FAIL")

    def test_each_manifest_must_have_a_binding(self):
        valid = self.manifest({"path": str(self.data), "sha256": self.digest})
        empty = self.manifest({"other": []}, "empty.json")
        result = verifier.verify([valid, empty])
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["matched_files"], 1)
        self.assertIn("no_bindings_found", self.issues(result))

    def test_cli_returns_failure_without_echoing_file_contents(self):
        path = self.manifest({"path": str(self.data), "sha256": "a" * 64})
        result = subprocess.run([sys.executable, str(SCRIPT), str(path)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("synthetic fixture", result.stdout)
        self.assertEqual(json.loads(result.stdout)["issues"][0]["problem"], "sha256_mismatch")

    def test_mapping_validation(self):
        for raw in ("old=new", "/old=", "/old=relative", "/old/../escape=/new"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                verifier.parse_mapping(raw)
        with self.assertRaisesRegex(ValueError, "duplicate_mapping_source"):
            verifier.verify([], mappings=[(Path("/old"), self.root), (Path("/old"), self.root / "other")])


if __name__ == "__main__":
    unittest.main()
