from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ..core import sha256_file
from .qa_generation_only import (
    PROBE_VERSION,
    _manifest,
    _root_dir,
    _source_inventory,
)


class QAGenerationOnlyTests(unittest.TestCase):
    def test_manifest_disables_solver_and_admission(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source_run = Path(temp) / "source"
            source_run.mkdir()
            (source_run / "batch_plan.json").write_text("{}", encoding="utf-8")
            ready = [{
                "shard": "shard-00",
                "root_id": "root-1",
                "source_path": "/tmp/root-1.json",
                "source_sha256": "abc",
            }]
            manifest = _manifest(source_run, Path(temp) / "out", {}, ready, [], [])
            self.assertEqual(manifest["probe_version"], PROBE_VERSION)
            self.assertFalse(manifest["api_contract"]["solver_enabled"])
            self.assertEqual(manifest["api_contract"]["solver_calls"], 0)
            self.assertFalse(manifest["screening_contract"]["difficulty_filter"])
            self.assertFalse(manifest["screening_contract"]["semantic_model_admission"])
            self.assertTrue(manifest["lineage_contract"]["output_namespace_isolated"])

    def test_inventory_keeps_ready_and_missing_roots_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "expanded.json"
            source.write_text("{}", encoding="utf-8")
            ready_result = {
                "status": "EXPANSION_READY",
                "selected_source": str(source),
                "selected_source_sha256": sha256_file(source),
                "successful_attempt_root": "/tmp/attempt",
            }
            plan = {
                "shards": [{
                    "shard": "shard-00",
                    "run_root": str(Path(temp) / "source"),
                    "roots": [
                        {"root_id": "ready", "input_sha256": "in", "parent_dialogue_sha256": "parent"},
                        {"root_id": "missing", "input_sha256": "in2", "parent_dialogue_sha256": "parent2"},
                    ],
                }],
            }
            def fake_result(path: Path, **kwargs: object):
                return ready_result if "ready" in str(path) else None

            with patch(
                "scripts.current_method.restart_n5_spell.h4_batch30_memory_event_k3_probe_v1.qa_generation_only._existing_ready",
                side_effect=fake_result,
            ):
                ready, missing = _source_inventory(plan, [])
            self.assertEqual([row["root_id"] for row in ready], ["ready"])
            self.assertEqual([row["root_id"] for row in missing], ["missing"])

    def test_root_directory_isolated_by_shard_and_root(self) -> None:
        path = _root_dir(Path("/tmp/output"), "shard-00", "root:1")
        self.assertEqual(path, Path("/tmp/output/shards/shard-00/roots/root_1"))


if __name__ == "__main__":
    unittest.main()
