from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from ..core import canonical_json
from ..input_adapter import dialogue_content_hash
from .plan import (
    BATCH_PLAN_VERSION,
    build_batch_plan,
    load_cleaned_batch,
    select_successful_expansion_source,
    write_batch_plan,
)


def _root(root_id: str, source_path: Path) -> dict:
    turns = [{
        "turn_id": f"{root_id}:P01:T001",
        "session_id": f"{root_id}:parent_session_01",
        "speaker": "Alex",
        "text": "The plan remains open.",
        "partition": "parent",
        "source_id": root_id.removeprefix("cc-train-") or root_id,
        "source_session_index": 1,
        "source_turn_index": 1,
    }, {
        "turn_id": f"{root_id}:P01:T002",
        "session_id": f"{root_id}:parent_session_01",
        "speaker": "Blair",
        "text": "We can review it tomorrow.",
        "partition": "parent",
        "source_id": root_id.removeprefix("cc-train-") or root_id,
        "source_session_index": 1,
        "source_turn_index": 2,
    }]
    dialogue = {
        "dialogue_hash": dialogue_content_hash(turns),
        "session_count": 1,
        "turn_count": len(turns),
        "turns": turns,
    }
    return {
        "schema_version": "unified_dialogue_root_v1",
        "root_id": root_id,
        "sample_id": root_id,
        "lineage_group": root_id,
        "source": {
            "dataset": "fixture",
            "split": "test",
            "source_id": root_id.removeprefix("cc-train-") or root_id,
            "source_row_index": 0,
            "source_path": str(source_path.resolve()),
            "source_file_sha256": "0" * 64,
            "source_record_sha256": "1" * 64,
            "source_record_sha256_scope": "fixture",
        },
        "dialogue": dialogue,
        "session_metadata": [{
            "session_id": turns[0]["session_id"],
            "source_session_index": 1,
            "time_interval": "day 1",
            "summary": "A short planning exchange.",
            "turn_ids": [turn["turn_id"] for turn in turns],
        }],
        "source_metadata": {"fixture": True},
    }


def _manifest(root_dir: Path, roots: list[dict]) -> Path:
    records = []
    for index, root in enumerate(roots):
        path = root_dir / "roots" / root["root_id"] / "unified_root.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(root, ensure_ascii=False), encoding="utf-8")
        import hashlib
        records.append({
            "root_id": root["root_id"], "output_path": str(path),
            "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_record_sha256": "1" * 64, "source_row_index": index,
        })
    payload = {
        "manifest_version": "named_input_cleaning_batch_manifest_v1",
        "status": "passed", "expected_count": len(roots),
        "record_count": len(roots), "roots_directory": str(root_dir / "roots"),
        "global_checks": {
            "all_records_valid": True, "all_repeatable": True,
            "expected_count": True, "mapping_coverage_exact": True,
            "names_unique_within_each_root": True, "root_ids_unique": True,
            "source_unchanged": True,
        }, "records": records,
    }
    path = root_dir / "manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _write_expansion_fixture(
    base: Path, *, root_id: str, input_sha256: str,
    status: str = "completed", generation_ready: bool = True,
    promotion: bool = True, invalid_promotion: bool = False,
) -> Path:
    """Create a small report/freeze/output fixture without invoking runtimes."""

    expansion = base / "expansion"
    (expansion / "reports").mkdir(parents=True, exist_ok=True)
    (expansion / "promotion").mkdir(parents=True, exist_ok=True)
    parent_hash = "b" * 64
    expanded = {
        "expanded_dialogue_version": "simple_append_expanded_dialogue_v1",
        "root_id": root_id,
        "append_mode": "parent_tail_only",
        "parent_dialogue_sha256": parent_hash,
        "parent_turn_count": 1,
        "child_turn_count": 1,
        "turn_count": 2,
        "child_scene_refs": ["R01:S01"],
        "child_boundary": {
            "first_child_turn_index_1_based": 2,
            "last_parent_turn_ref": f"{root_id}:P01:T001",
            "first_child_turn_ref": "R01:S01:T001",
        },
        "turns": [
            {"turn_ref": f"{root_id}:P01:T001", "session_ref": "p",
             "scene_ref": None, "speaker": "Alex", "text": "Before",
             "origin": "source_grounded"},
            {"turn_ref": "R01:S01:T001", "session_ref": "c",
             "scene_ref": "R01:S01", "speaker": "Blair", "text": "After",
             "origin": "generated_extension"},
        ],
        "qa_fact_source": "complete_parent_plus_child",
        "memory_is_qa_fact_source": False,
    }
    expanded["complete_dialogue_sha256"] = hashlib.sha256(
        json.dumps(expanded, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")).hexdigest()
    raw_path = expansion / "expanded_dialogue.json"
    raw_path.write_text(json.dumps(expanded, ensure_ascii=False), encoding="utf-8")
    validation = {
        "valid": True, "errors": [], "parent_turn_count": 1,
        "child_turn_count": 1, "turn_count": 2,
    }
    (expansion / "expanded_dialogue_validation.json").write_text(
        json.dumps(validation), encoding="utf-8")
    if promotion:
        promoted = dict(expanded)
        if invalid_promotion:
            promoted["complete_dialogue_sha256"] = "c" * 64
        promoted_path = expansion / "promotion" / "promotable_expanded_dialogue.json"
        promoted_path.write_text(json.dumps(promoted, ensure_ascii=False), encoding="utf-8")
        promoted_validation = dict(validation)
        if invalid_promotion:
            promoted_validation["valid"] = False
            promoted_validation["errors"] = ["fixture_invalid"]
        (expansion / "promotion" / "promotable_expanded_dialogue_validation.json").write_text(
            json.dumps(promoted_validation), encoding="utf-8")
    report = {
        "status": status,
        "generation_ready": generation_ready,
        "root_id": root_id,
        "run_root": str(expansion.resolve()),
        "input_sha256": input_sha256,
        "child_add_verified": True,
        "event_bucket_only": True,
    }
    if promotion:
        report["promotable_expanded_dialogue"] = expanded if not invalid_promotion else {
            "root_id": root_id,
            "complete_dialogue_sha256": "d" * 64,
        }
        report["promotion_validation"] = {
            "valid": not invalid_promotion,
            "errors": [] if not invalid_promotion else ["fixture_invalid"],
        }
    (expansion / "reports" / "MEMORY_EVENT_CONTINUATION_LIVE_REPORT.json").write_text(
        json.dumps(report), encoding="utf-8")
    freeze = {
        "decision": "pass" if status == "completed" else "fail",
        "root_id": root_id,
        "input_sha256": input_sha256,
        "parent_dialogue_sha256": parent_hash,
    }
    (expansion / "run_freeze_manifest.json").write_text(
        json.dumps(freeze), encoding="utf-8")
    return expansion


class PlanTests(unittest.TestCase):
    def test_load_and_split_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source.jsonl"
            source.write_text("fixture\n", encoding="utf-8")
            roots = [_root(f"cc-train-episode-{i}", source) for i in range(4)]
            manifest = _manifest(base, roots)
            loaded = load_cleaned_batch(base, expected_count=4)
            self.assertEqual(loaded["manifest_path"], str(manifest.resolve()))
            plan = build_batch_plan(
                cleaned_batch=base, run_root=base / "plan",
                expected_count=4, shard_size=2,
                config=base / "config.json", tokenizer=base / "tokenizer",
                solver_model=base / "solver", credentials=base / "credentials",
            )
            self.assertEqual(plan["plan_version"], BATCH_PLAN_VERSION)
            self.assertEqual(plan["shard_count"], 2)
            self.assertEqual([s["root_count"] for s in plan["shards"]], [2, 2])
            self.assertEqual(plan["api_calls"], 0)
            self.assertEqual(plan["solver_calls"], 0)
            self.assertTrue(plan["mainline_untouched"])
            self.assertTrue(
                plan["expansion_contract"]["fixed_flags"]["event_bucket_only"])
            namespaces = {
                row["expansion"]["route"]["namespace"]
                for row in plan["roots"]
            }
            self.assertEqual(len(namespaces), 4)
            for row in plan["roots"]:
                command = row["expansion"]["command"]
                self.assertIn("--event-bucket-only", command)
                self.assertIn("--allow-insecure-memorax-http", command)
                expected_user_id = (
                    f"{row['expansion']['route']['namespace']}:"
                    f"{row['root_id']}"
                )
                self.assertEqual(
                    row["expansion"]["route"]["user_id"], expected_user_id)
            shard_commands = plan["shards"][0]["commands"]
            self.assertIsNone(shard_commands["qa_inventory"])
            self.assertIn("{selected_path}", shard_commands[
                "qa_inventory_command_template"])
            for stage in ("qa_smoke", "qa_batch"):
                command = shard_commands[stage]
                self.assertIn("--api-config", command)
                self.assertEqual(
                    command[command.index("--api-config") + 1],
                    str((base / "config.json").resolve()),
                )

    def test_write_plan_only_creates_new_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source.jsonl"
            source.write_text("fixture\n", encoding="utf-8")
            _manifest(base, [_root("cc-train-episode-1", source)])
            plan = build_batch_plan(
                cleaned_batch=base, run_root=base / "plan",
                expected_count=1, shard_size=1,
            )
            output = write_batch_plan(plan)
            self.assertTrue((output / "batch_plan.json").is_file())
            self.assertTrue((output / "shards/shard-00/manifest.json").is_file())
            self.assertEqual(json.loads((output / "batch_plan.json").read_text())["api_calls"], 0)

    def test_source_selector_prefers_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            expansion = _write_expansion_fixture(
                base, root_id="root-1", input_sha256="a" * 64)
            selected = select_successful_expansion_source(
                expansion, expected_root_id="root-1",
                expected_input_sha256="a" * 64,
                expected_parent_dialogue_sha256="b" * 64,
            )
            self.assertTrue(selected["eligible"])
            self.assertEqual(selected["source_kind"], "promotion")
            self.assertTrue(selected["selected_path"].endswith(
                "promotion/promotable_expanded_dialogue.json"))

    def test_source_selector_allows_only_success_gated_raw_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            expansion = _write_expansion_fixture(
                base, root_id="root-2", input_sha256="a" * 64, promotion=False)
            selected = select_successful_expansion_source(
                expansion, expected_root_id="root-2",
                expected_input_sha256="a" * 64,
                expected_parent_dialogue_sha256="b" * 64,
            )
            self.assertTrue(selected["eligible"])
            self.assertEqual(selected["source_kind"], "raw_success_fallback")

    def test_source_selector_rejects_failed_raw_even_when_structurally_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            expansion = _write_expansion_fixture(
                base, root_id="root-3", input_sha256="a" * 64,
                status="failed", generation_ready=False, promotion=False)
            selected = select_successful_expansion_source(
                expansion, expected_root_id="root-3",
                expected_input_sha256="a" * 64,
                expected_parent_dialogue_sha256="b" * 64,
            )
            self.assertFalse(selected["eligible"])
            self.assertIn("report_status_completed", selected["reason_codes"])
            self.assertIsNone(selected["selected_path"])

    def test_source_selector_does_not_fallback_when_promotion_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            expansion = _write_expansion_fixture(
                base, root_id="root-4", input_sha256="a" * 64,
                invalid_promotion=True)
            selected = select_successful_expansion_source(
                expansion, expected_root_id="root-4",
                expected_input_sha256="a" * 64,
                expected_parent_dialogue_sha256="b" * 64,
            )
            self.assertFalse(selected["eligible"])
            self.assertIn("selected_complete_hash_matches", selected["reason_codes"])
            self.assertIsNone(selected["selected_path"])


if __name__ == "__main__":
    unittest.main()
