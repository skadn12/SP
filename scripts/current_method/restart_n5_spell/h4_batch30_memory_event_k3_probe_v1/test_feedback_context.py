from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ..core import sha256_file
from . import qa_generation_only as qa


class FeedbackContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source = self.base / "source"
        self.source.mkdir()
        self.write(self.source / "batch_plan.json", {"offline_only": True})
        self.output = self.base / "output"
        self.context = {
            "intent_suffix": "Human-frozen feedback guidance.",
            "slot_suffixes": ["Writer guidance."] * 3,
        }
        self.context_path = self.base / "context.json"
        self.write(self.context_path, self.context)
        self.mapping_path = self.base / "feedback.json"
        self.write(self.mapping_path, {"roots": {
            "root-1": {"path": "context.json", "sha256": sha256_file(self.context_path)},
        }})
        self.ready = [{
            "shard": "shard-00", "root_id": "root-1",
            "source_path": str(self.source / "expanded.json"), "source_sha256": "source-hash",
        }]

    @staticmethod
    def write(path: Path, value: object) -> None:
        path.write_text(json.dumps(value), encoding="utf-8")

    def args(self, feedback: Path | None) -> argparse.Namespace:
        return argparse.Namespace(
            source_run=self.source, output_run=self.output, shard=[], allow_partial=False,
            feedback_context_manifest=feedback, api_config=self.base / "unused-config.json",
            live=True, force=False,
        )

    def inventory(self, feedback: Path | None) -> dict:
        with patch.object(qa, "_source_inventory", return_value=(self.ready, [])):
            qa.inventory_phase(self.args(feedback))
        return json.loads((self.output / "manifest.json").read_text())

    def run_mock_generation(self, feedback: Path | None):
        root_dir = qa._root_dir(self.output, "shard-00", "root-1")
        root_dir.mkdir(parents=True, exist_ok=True)
        expanded = {"root_id": "root-1", "turns": []}
        report = {
            "root_id": "root-1", "candidate_count": 1, "valid_candidate_count": 1,
            "candidate_generation_calls": 6,
        }
        with patch.object(qa, "_client", return_value=object()) as client, \
                patch.object(qa, "_prepare_root", return_value=(root_dir, expanded)), \
                patch.object(qa, "generate_root_qa", return_value=report) as generate:
            result = qa.generate_phase(self.args(feedback))
        self.assertEqual(result["solver_calls"], 0)
        self.assertFalse(result["semantic_admission_applied"])
        self.assertFalse(result["difficulty_filter_applied"])
        client.assert_called_once()
        generate.assert_called_once()
        return generate.call_args.kwargs

    def test_no_feedback_preserves_manifest_and_generator_arguments(self) -> None:
        expected = qa._manifest(self.source, self.output, {"offline_only": True}, self.ready, [], [])
        actual = self.inventory(None)
        self.assertEqual(actual, expected)
        self.assertNotIn("feedback_context", actual)
        kwargs = self.run_mock_generation(None)
        self.assertNotIn("feedback_context", kwargs)

    def test_inventory_freezes_hashes_and_generate_passes_exact_text(self) -> None:
        manifest = self.inventory(self.mapping_path)
        frozen = manifest["feedback_context"]
        self.assertEqual(frozen["manifest_sha256"], sha256_file(self.mapping_path))
        self.assertEqual(frozen["roots"]["root-1"], {
            "path": str(self.context_path), "sha256": sha256_file(self.context_path),
        })
        # A separate generate invocation can consume the frozen manifest without
        # specifying the optional input again.
        kwargs = self.run_mock_generation(None)
        self.assertEqual(kwargs["feedback_context"], self.context)

    def test_missing_root_fails_inventory_before_output(self) -> None:
        self.ready.append({**self.ready[0], "root_id": "root-2"})
        with self.assertRaisesRegex(ValueError, "feedback_context_root_missing:root-2"):
            self.inventory(self.mapping_path)
        self.assertFalse(self.output.exists())

    def test_context_drift_blocks_before_client_or_generation(self) -> None:
        self.inventory(self.mapping_path)
        self.context_path.write_text("changed", encoding="utf-8")
        with patch.object(qa, "_client") as client, patch.object(qa, "generate_root_qa") as generate:
            with self.assertRaisesRegex(RuntimeError, "feedback_context_hash_drift:root-1"):
                qa.generate_phase(self.args(None))
        client.assert_not_called()
        generate.assert_not_called()

    def test_mapping_drift_blocks_before_client(self) -> None:
        self.inventory(self.mapping_path)
        self.mapping_path.write_text(self.mapping_path.read_text() + "\n", encoding="utf-8")
        with patch.object(qa, "_client") as client:
            with self.assertRaisesRegex(RuntimeError, "feedback_context_manifest_hash_drift"):
                qa.generate_phase(self.args(None))
        client.assert_not_called()

    def test_feedback_cannot_be_added_to_an_unfrozen_generation(self) -> None:
        self.inventory(None)
        with patch.object(qa, "_client") as client:
            with self.assertRaisesRegex(RuntimeError, "feedback_context_requires_inventory_freeze"):
                qa.generate_phase(self.args(self.mapping_path))
        client.assert_not_called()

    def test_invalid_context_is_not_frozen(self) -> None:
        self.write(self.context_path, {**self.context, "generation_allowed": False})
        self.write(self.mapping_path, {"roots": {"root-1": {
            "path": str(self.context_path), "sha256": sha256_file(self.context_path),
        }}})
        with self.assertRaisesRegex(ValueError, "feedback_context_schema_invalid:root-1"):
            self.inventory(self.mapping_path)
        self.assertFalse(self.output.exists())

    def test_context_drift_in_later_root_blocks_entire_start(self) -> None:
        self.ready.append({**self.ready[0], "root_id": "root-2"})
        second = self.base / "context-2.json"
        self.write(second, self.context)
        self.write(self.mapping_path, {"roots": {
            "root-1": {"path": str(self.context_path), "sha256": sha256_file(self.context_path)},
            "root-2": {"path": str(second), "sha256": sha256_file(second)},
        }})
        self.inventory(self.mapping_path)
        second.write_text("drift", encoding="utf-8")
        with patch.object(qa, "_client") as client, patch.object(qa, "generate_root_qa") as generate:
            with self.assertRaisesRegex(RuntimeError, "feedback_context_hash_drift:root-2"):
                qa.generate_phase(self.args(None))
        client.assert_not_called()
        generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
