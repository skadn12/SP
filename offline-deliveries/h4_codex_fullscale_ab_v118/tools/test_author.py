"""Isolated synthetic fixtures for author identity, sequencing and immutability."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("v118_author", Path(__file__).with_name("author.py"))
author = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(author)

ROOT = "cc-train-episode-12345"
EXECUTOR = "/fixture/author"
REVIEWER = "/fixture/reviewer"


class AuthorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="v118-author-fixture-")
        self.master = Path(self.temp.name)
        self.unit = self.master / "units/root12345"
        self.folder = self.unit / "roots/root12345"
        self.out = io.StringIO()
        author.save(self.master / "ROLE_ASSIGNMENTS.json", {"roots": [{"root_id": ROOT,
            "author": EXECUTOR, "reviewer": REVIEWER, "baseline_reviewers": ["/fixture/base1", "/fixture/base2"]}]})
        self.baseline = self.unit / "inputs/BASELINE.json"
        author.save(self.baseline, {"root_id": ROOT, "turns": [
            {"turn_ref": "P01:T001", "speaker": "A", "text": "Synthetic baseline A."},
            {"turn_ref": "P01:T002", "speaker": "B", "text": "Synthetic baseline B."}]})
        reviews = []
        for number in (1, 2):
            path = self.unit / f"evidence/baseline_review{number}.json"
            author.save(path, {"reviewer": f"/fixture/base{number}", "decision": "PASS"})
            reviews.append(author.binding(path))
        self.constraints = self.unit / "inputs/CONTINUATION_CONSTRAINTS.json"
        author.save(self.constraints, {"root_id": ROOT, "decision": "PASS",
            "continuation_constraints": ["Synthetic fixture: preserve both baseline speakers."],
            "baseline_binding": author.binding(self.baseline), "review_bindings": reviews})
        author.save(self.unit / "RUN_CONFIG.json", {"controller_version": "codex_sequential_ab_v1",
            "batch_id": "v118-synthetic-fixture", "roots": [{"root_id": ROOT,
                "baseline_path": str(self.baseline), "baseline_sha256": author.digest(self.baseline)}],
            "new_turn_count": {"min": 12, "max": 16}, "scene_sequence": list(author.sequential.SCENES),
            "scope_scene_refs": list(author.sequential.SCENES)})
        allocations = []
        for index in (1, 2, 3):
            scene = f"R02:B:S{index:02d}"
            path = self.unit / f"inputs/B_S{index:02d}_EVENT.json"
            author.save(path, {"turns": [{"turn_id": "0", "text": f"Synthetic native event {index}."}]})
            allocations.append({"root_id": ROOT, "scene_ref": scene, "source_id": f"fixture:{index}",
                "source_item_path": str(path), "source_item_sha256": author.digest(path)})
        selection = self.unit / "selection.json"
        author.save(selection, {"root_id": ROOT, "allocations": allocations})
        author.save(self.unit / "SOURCE_RESERVATIONS.json", {"allocation": [
            {**r, "selection_path": str(selection), "selection_sha256": author.digest(selection)} for r in allocations]})

    def tearDown(self):
        self.temp.cleanup()

    def call(self, fn, *args):
        with contextlib.redirect_stdout(self.out):
            return fn(self.master, ROOT, EXECUTOR, *args)

    def payload(self, scene):
        path = self.master / (scene.replace(":", "_") + ".json")
        author.save(path, {"turns": [{"speaker": "A" if i % 2 else "B",
            "text": f"Synthetic fixture {scene} line {i}."} for i in range(1, 13)]})
        return path

    def scene_review(self, scene):
        folder = author.sequential.scene_folder(self.unit, ROOT, scene)
        source_path = folder / "CUMULATIVE_SOURCE.json"
        source = author.read(source_path)
        author.save(folder / "SOURCE_REVIEW.json", {"root_id": ROOT, "scene_ref": scene,
            "source_sha256": author.digest(source_path), "reviewer": REVIEWER, "full_source_read": True,
            "turn_refs_read": [t["turn_ref"] for t in source["turns"]], "created_at_utc": author.now(),
            "decision": "READY", "findings": [], "checks": {
                k: {"status": "PASS", "reason": "SYNTHETIC FIXTURE ONLY"} for k in author.sequential.CHECKS}})

    def complete(self):
        for scene in author.sequential.SCENES:
            self.call(author.make_read, scene)
            self.call(author.scene_save, scene, self.payload(scene))
            self.scene_review(scene)

    def test_identity_and_predecessor_block_without_author_artifacts(self):
        with self.assertRaisesRegex(ValueError, "EXECUTOR_NOT_ASSIGNED_AUTHOR"):
            author.make_read(self.master, ROOT, REVIEWER, "R02:A:S01")
        with self.assertRaisesRegex(ValueError, "SCENE_ALREADY_ATTEMPTED_OR_NOT_READY|PREDECESSOR_NOT_READY"):
            self.call(author.make_read, "R02:A:S02")
        self.assertFalse((self.unit / "evidence/AUTHOR_R02_A_S01_READ_CONTEXT.json").exists())
        with self.assertRaisesRegex(ValueError, "ALL_SIX_SCENES_MUST_BE_READY"):
            self.call(author.make_read)

    def test_missing_and_empty_constraints_fail_closed(self):
        self.constraints.unlink()
        with self.assertRaises(FileNotFoundError):
            self.call(author.make_read, "R02:A:S01")
        author.save(self.constraints, {"root_id": ROOT, "decision": "PASS", "continuation_constraints": []})
        with self.assertRaisesRegex(ValueError, "ALL_CONTINUATION_CONSTRAINTS_REQUIRED"):
            self.call(author.make_read, "R02:A:S01")

    def test_scene_first_write_read_coverage_and_no_overwrite(self):
        scene = "R02:A:S01"
        path = self.payload(scene)
        with self.assertRaises(FileNotFoundError):
            self.call(author.scene_save, scene, path)
        self.call(author.make_read, scene)
        self.call(author.scene_save, scene, path)
        folder = author.sequential.scene_folder(self.unit, ROOT, scene)
        self.assertEqual((folder / "AUTHOR_PAYLOAD.json").read_bytes(), path.read_bytes())
        receipt = author.read(folder / "AUTHOR_RECEIPT.json")
        self.assertEqual(receipt["turn_refs_read"], ["P01:T001", "P01:T002"])
        self.assertEqual(len(receipt["all_continuation_constraints_read"]), 1)
        self.assertLessEqual((folder / "PAYLOAD_SERIALIZATION_BINDING.json").stat().st_mtime_ns,
                             (folder / "EXPANSION_DRAFT.json").stat().st_mtime_ns)
        old = author.digest(folder / "EXPANSION_DRAFT.json")
        with self.assertRaises(ValueError):
            self.call(author.scene_save, scene, path)
        self.assertEqual(author.digest(folder / "EXPANSION_DRAFT.json"), old)

    def test_malformed_first_payload_is_retained_and_root_stops(self):
        scene = "R02:A:S01"
        self.call(author.make_read, scene)
        path = self.master / "invalid.json"
        author.save(path, {"turns": [{"speaker": "NEW", "text": "invalid"}] * 12})
        with self.assertRaises(ValueError):
            self.call(author.scene_save, scene, path)
        folder = author.sequential.scene_folder(self.unit, ROOT, scene)
        self.assertTrue((folder / "AUTHOR_FAILURE.json").is_file())
        self.assertEqual((folder / "AUTHOR_PAYLOAD.json").read_bytes(), path.read_bytes())
        with self.assertRaisesRegex(ValueError, "ROOT_STOPPED"):
            self.call(author.make_read, scene)

    def test_full_chain_native_archive_anchor_order_and_qa(self):
        self.complete()
        self.call(author.make_read)
        b3 = author.sequential.scene_folder(self.unit, ROOT, "R02:B:S03")
        self.assertEqual((self.folder / "CUMULATIVE_SOURCE.json").read_bytes(), (b3 / "CUMULATIVE_SOURCE.json").read_bytes())
        rendered = (self.unit / "evidence/AUTHOR_R02_B_S01_FULL_READ.md").read_text()
        self.assertIn("NATIVE_EVENT_FULL", rendered)
        self.assertIn("SELECTION_FULL", rendered)
        source = author.read(self.folder / "CUMULATIVE_SOURCE.json")
        turns = {t["turn_ref"]: t for t in source["turns"]}
        anchor_payload = self.master / "anchor.json"
        slots = []
        for slot in (1, 2):
            slots.append({"slot_id": str(slot), "anchors": [
                {"anchor_id": f"S{slot}A{i}", "turn_ref": f"{s}:T001", "speaker": turns[f"{s}:T001"]["speaker"],
                 "full_turn_quote": turns[f"{s}:T001"]["text"]} for i, s in enumerate(author.sequential.SCENES, 1)]})
        author.save(anchor_payload, {"slots": slots})
        qa_payload = self.master / "qa.json"
        author.save(qa_payload, {"stem": "Select all supported within " + ", ".join(author.sequential.SCENES),
            "facts": [{"fact_id": f"F{i}", "source_turn_ref": f"{s}:T001", "true_text": "Synthetic true fact.",
                       "false_text": "Synthetic false fact."} for i, s in enumerate(author.sequential.SCENES, 1)],
            "ordering": [{"slot_id": str(slot["slot_id"]), "items": [{"item_id": f"I{i}", "anchor_id": a["anchor_id"],
                "text": "Synthetic ordering claim.", "support_quote": a["full_turn_quote"]} for i, a in enumerate(slot["anchors"], 1)]} for slot in slots]})
        with self.assertRaisesRegex(ValueError, "ANCHORS_MUST_BE_FROZEN_FIRST"):
            self.call(author.qa_save, qa_payload)
        self.call(author.anchor_save, anchor_payload)
        old = author.digest(self.folder / "ORDERING_ANCHOR_PLAN.json")
        with self.assertRaisesRegex(ValueError, "ANCHORS_ALREADY_FROZEN"):
            self.call(author.anchor_save, anchor_payload)
        self.call(author.qa_save, qa_payload)
        self.assertEqual(author.digest(self.folder / "ORDERING_ANCHOR_PLAN.json"), old)
        self.assertTrue(author.read(self.folder / "QA_AUTHOR_RECEIPT.json")["ordering_anchor_plan_saved_before_QA_DRAFT"])
        with self.assertRaisesRegex(ValueError, "FIRST_QA_PAYLOAD_ALREADY_SAVED"):
            self.call(author.qa_save, qa_payload)

    def test_source_and_roles_changed_after_read_are_rejected(self):
        self.call(author.make_read, "R02:A:S01")
        with self.baseline.open("a") as stream:
            stream.write(" ")
        with self.assertRaises(ValueError):
            self.call(author.scene_save, "R02:A:S01", self.payload("R02:A:S01"))


if __name__ == "__main__":
    unittest.main()
