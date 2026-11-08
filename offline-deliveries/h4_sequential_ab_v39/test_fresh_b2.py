"""Temporary mock A3 -> unused source selection -> B2 -> real audit.

All new dialogue and source rows are explicitly synthetic test fixtures. Socket
creation is prohibited and reservations/inventory stay in a temporary delivery.
No existing runtime or frozen test file is changed by this added regression.
"""
from contextlib import ExitStack
import inspect
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

# Import the local tokenizer stack before blocking socket construction: Python's
# SSL module defines subclasses of the real socket class during import.
from transformers import AutoTokenizer

import a3_continuation as a
import audit_sequential
import fresh_b2 as b
from test_sequential_flow import SequentialProvider, initialize_mock

r = b.r


def mock_sources(folder):
    folder.mkdir()
    soda = {"rows": [{"row_idx": i, "truncated_cells": [], "row": {
        "original_index": 99000000 + i, "split": "train", "relation": relation,
        "narrative": "MOCK ONLY: isolated request and response.",
        "speakers": ["A", "B"], "dialogue": ["MOCK ONLY: a request.", "MOCK ONLY: a response."]}}
        for i, relation in enumerate(("xNeed", "xWant", "xReact", "xNeed"))]}
    multiwoz = [{"dialogue_id": f"MOCK_ONLY_{i}.json", "services": [service], "turns": [
        {"turn_id": str(j), "speaker": "USER" if j % 2 == 0 else "SYSTEM",
         "utterance": f"MOCK ONLY: request or response {j}."} for j in range(12)]}
        for i, service in enumerate(("hotel", "restaurant", "taxi", "hotel"))]
    source = folder / "soda_train_sample.json"
    source.write_text(json.dumps(soda))
    (folder / "soda_train_sample_source.json").write_text(json.dumps({
        "url": "https://example.invalid/MOCK_ONLY/soda", "sha256": r.base.digest(source.read_bytes())}))
    source = folder / "multiwoz_train_sample.json"
    source.write_text(json.dumps(multiwoz))
    (folder / "multiwoz_train_source.json").write_text(json.dumps({
        "url": "https://example.invalid/MOCK_ONLY/multiwoz", "download_sha256": r.base.digest(source.read_bytes()),
        "download_bytes": source.stat().st_size}))


class B2ReplenishedTests(unittest.TestCase):
    def test_new_source_directory_preserves_A3_and_complete_B_flow_and_consumption(self):
        original_here = r.HERE
        with tempfile.TemporaryDirectory(prefix="B2_REPLENISHED_MOCK_ONLY_") as tmp, ExitStack() as stack:
            temp = Path(tmp)
            delivery = temp / "offline-deliveries/h4_sequential_ab_v38"
            delivery.mkdir(parents=True)
            (delivery / "runs").mkdir()
            (delivery / "POLICY.md").write_text("MOCK ONLY: replenished pool regression.\n")
            for path in original_here.glob("*.py"):
                shutil.copy2(path, delivery / path.name)
            unique = delivery.parent / "h4_r02_source_unique_v2"
            unique.mkdir()
            shutil.copy2(original_here.parent / "h4_r02_source_unique_v2/unique_events.py", unique / "unique_events.py")
            sources = temp / "MOCK_ONLY_PUBLIC_TRAIN_STRUCTURE"
            mock_sources(sources)
            source_before = {p: p.read_bytes() for p in sources.iterdir()}
            fake = SequentialProvider()
            stack.enter_context(patch.object(r, "HERE", delivery))
            stack.enter_context(patch.object(audit_sequential, "HERE", delivery))
            stack.enter_context(patch.object(audit_sequential, "r", r))
            stack.enter_context(patch.object(r, "LOCK_PATH", temp / "inventory.lock"))
            stack.enter_context(patch.object(r, "MIN_HTTP_GAP_SECONDS", 0))
            stack.enter_context(patch.object(r.Caller, "__init__",
                lambda caller, run: initialize_mock(caller, run, fake)))
            stack.enter_context(patch("socket.socket", side_effect=AssertionError("network_forbidden_in_offline_test")))
            collect = r.source_inventory.collect_prior
            prior_paths = {key: temp / key for key in inspect.signature(collect).parameters if key != "current_runs"}
            def isolated_inventory():
                return collect(delivery / "runs", **prior_paths)
            stack.enter_context(patch.object(r.source_inventory, "collect_prior", side_effect=isolated_inventory))
            parent = delivery / "runs/MOCK_A3"
            a.prepare(parent)
            report = a.live(parent)
            self.assertEqual(report["status"], "A3_COMPLETE")
            self.assertEqual(a.audit(parent)["status"], "PASS")
            parent_bytes = {p: p.read_bytes() for p in parent.rglob("*") if p.is_file()}
            parent_turns = r.base.read(parent / "roots" / a.ROOT / "A/cumulative_dialogue.json")["turns"]
            run = delivery / "runs/MOCK_B2"
            original_freezer, original_writer = r.source_pool.freeze_unused_source_subset, r.base.write
            prepared = b.prepare(run, parent, sources)
            self.assertIs(r.source_pool.freeze_unused_source_subset, original_freezer)
            self.assertIs(r.base.write, original_writer)
            self.assertEqual(prepared["assigned_B_slots"], 2)
            self.assertFalse((run / "SOURCE_RESERVATIONS.json").exists())
            manifest = b.verify(run)
            self.assertEqual(manifest["execution_entrypoint"], b.ENTRYPOINT)
            self.assertEqual(manifest["source_pool_directory"], str(sources))
            self.assertEqual(manifest["inherited_A"]["accepted_scenes"], 3)
            self.assertEqual(manifest["logical_max"], 34)
            self.assertEqual(manifest["http_max"], 68)
            self.assertIs(manifest["require_local_state_repair"], True)
            self.assertIs(manifest["training_pool_eligible"], False)
            sources_manifest = r.base.read(run / "dataset_sources.json")
            self.assertEqual(len(sources_manifest["items"]), 2)
            self.assertEqual({row["dataset"] for row in sources_manifest["items"]}, {"allenai/soda", "MultiWOZ_2.2"})
            self.assertTrue(all(Path(row["local_path"]).parent == sources for row in sources_manifest["files"]))
            bound = {Path(row["path"]) for row in manifest["files"]}
            self.assertTrue(set(source_before) <= bound)
            calls_before = len(fake.observed)
            def before_b_request():
                reservation = r.base.read(run / "SOURCE_RESERVATIONS.json")
                self.assertEqual(len(reservation["reservations"]), 2)
                self.assertTrue((run / "LIVE_STARTED.json").exists())
            fake.before_request = before_b_request
            result = b.live(run)
            self.assertEqual(result["accepted_scenes"], 2, result)
            self.assertEqual(result["inherited_A_scenes"], 3)
            self.assertEqual(result["new_A_calls"], 0)
            audited = b.audit(run)
            self.assertEqual(audited["status"], "PASS", audited["failures"])
            b_calls = fake.observed[calls_before:]
            for observation in b_calls:
                context = observation["context"]
                if observation["role"] in ("purpose_proposer", "dataset_extractor"):
                    continue
                history = context.get("cumulative_history", context.get("branch_history"))
                self.assertEqual(history[:len(parent_turns)], parent_turns)
                self.assertEqual(context["route"], r.ROUTES[1])
            source_result = r.base.read(run / "sources_merged" / f"{a.ROOT}.json")
            self.assertIs(source_result["training_pool_eligible"], False)
            self.assertIs(result["training_pool_eligible"], False)
            for path in (run / "requests").glob("*/PROOF.json"):
                proof = r.base.read(path)
                if proof["role"] == "state_extractor":
                    self.assertIs(proof["require_local_state_repair"], True)
                    self.assertEqual(proof["state_repair_policy"], r.state_contract.STATE_CITATION_PATCH_VERSION)
                if proof["role"] == "scene_reviewer":
                    self.assertEqual(proof["review_contract_version"], r.realization_review.REVIEW_VERSION)
            final = source_result["turns"]
            self.assertEqual(final[:len(parent_turns)], parent_turns)
            self.assertEqual(len(final), len(parent_turns) + 24)
            self.assertTrue(all(row["turn_ref"].startswith("R02:B:") for row in final[len(parent_turns):]))
            self.assertEqual({p: p.read_bytes() for p in parent_bytes}, parent_bytes)
            self.assertEqual({p: p.read_bytes() for p in source_before}, source_before)
            consumed = {r.source_inventory.event_key(row) for row in isolated_inventory()["accepted_uses"]}
            self.assertEqual(len(consumed), 2)
            later = r.source_pool.freeze_unused_source_subset(consumed, 3, source_dir=sources)
            self.assertTrue(later["items"])
            self.assertFalse(consumed & {r.source_inventory.event_key(row) for row in later["items"]})
            # Frozen local source byte changes cannot survive verify.
            path = sources / "soda_train_sample.json"
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "frozen_file_changed"):
                b.verify(run)

    def test_missing_source_directory_fails_before_any_prepare_or_model(self):
        with tempfile.TemporaryDirectory(prefix="B2_MISSING_SOURCE_MOCK_ONLY_") as tmp:
            temp = Path(tmp)
            with patch.object(r, "prepare", side_effect=AssertionError("should_not_prepare")):
                with self.assertRaisesRegex(ValueError, "replenished_source_directory_missing"):
                    b.prepare(temp / "run", temp / "parent", temp / "missing")


if __name__ == "__main__":
    unittest.main()
