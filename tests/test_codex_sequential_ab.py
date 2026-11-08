"""Exercise the A3/B3 gate with synthetic artifacts, without any model calls."""
import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import codex_sequential_ab as controller


ROOT = "cc-train-episode-123"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_batch(tmp_path):
    baseline = tmp_path / "inputs/root123_BASELINE.json"
    write(baseline, {"root_id": ROOT, "turns": [
        {"turn_ref": "P01:T001", "scene_ref": "P01", "session_ref": "P01", "speaker": "Alice", "text": "We planned a garden.", "origin": "baseline"},
        {"turn_ref": "P01:T002", "scene_ref": "P01", "session_ref": "P01", "speaker": "Bob", "text": "We have not planted it yet.", "origin": "baseline"}]})
    write(tmp_path / "RUN_CONFIG.json", {
        "controller_version": controller.VERSION, "scene_sequence": list(controller.SCENES),
        "new_turn_count": {"min": 12, "max": 16},
        "roots": [{"root_id": ROOT, "baseline_path": str(baseline), "baseline_sha256": controller.digest(baseline)}]})
    allocations = []
    for index, scene in enumerate(controller.SCENES[3:], 1):
        event = tmp_path / f"inputs/root123_B_S{index:02d}_EVENT.json"
        write(event, {"event": f"Reserved event {index}"})
        allocations.append({"root_id": ROOT, "scene_ref": scene, "source_id": f"soda:train:{index}",
                            "source_item_path": str(event), "source_item_sha256": controller.digest(event)})
    write(tmp_path / "SOURCE_RESERVATIONS.json", {"allocation": allocations})
    return tmp_path


def draft(batch, index, count=12):
    scene = controller.SCENES[index]
    previous = batch / "inputs/root123_BASELINE.json" if not index else controller.scene_folder(batch, ROOT, controller.SCENES[index - 1]) / "CUMULATIVE_SOURCE.json"
    value = {"root_id": ROOT, "scene_ref": scene, "author": "/author", "input_source_sha256": controller.digest(previous), "turns": [
        {"turn_ref": f"{scene}:T{i:03d}", "session_ref": scene, "scene_ref": scene,
         "speaker": "Alice" if i % 2 else "Bob", "text": f"Scene {index}, statement {i}.", "origin": "codex_session_generated"}
        for i in range(1, count + 1)]}
    if index >= 3:
        value["source_id"] = f"soda:train:{index - 2}"
    write(controller.scene_folder(batch, ROOT, scene) / "EXPANSION_DRAFT.json", value)


def review(batch, index, decision="READY", reviewer="/reviewer", omit_old=False):
    scene = controller.SCENES[index]
    folder = controller.scene_folder(batch, ROOT, scene)
    source = folder / "CUMULATIVE_SOURCE.json"
    turns = controller.read(source)["turns"]
    value = {"root_id": ROOT, "scene_ref": scene, "source_sha256": controller.digest(source),
             "reviewer": reviewer, "created_at_utc": controller.now(), "full_source_read": True,
             "turn_refs_read": [turn["turn_ref"] for turn in (turns[-12:] if omit_old else turns)],
             "checks": {name: {"status": "PASS", "reason": "Synthetic check fixture only."} for name in controller.CHECKS},
             "findings": [], "decision": decision}
    write(folder / "SOURCE_REVIEW.json", value)


def materialize(batch, index):
    return controller.materialize_scene(batch, ROOT, controller.SCENES[index])


class SequentialABTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.batch = make_batch(Path(self.temporary.name))

    def test_all_six_independently_reviewed_scenes_required(self):
        batch = self.batch
        baseline = controller.read(batch / "inputs/root123_BASELINE.json")["turns"]
        for index in range(6):
            draft(batch, index)
            result = materialize(batch, index)
            assert result["status"] == "STRUCTURAL_PASS_ONLY"
            source = controller.read(result["source_path"])
            assert source["turns"][:len(baseline)] == baseline
            assert len(source["turns"]) == len(baseline) + 12 * (index + 1)
            assert not controller.progress(batch)["full_expansion_complete"]
            review(batch, index)
            state = controller.progress(batch)
            assert state["roots"][0]["ready_scene_count"] == index + 1
            assert state["full_expansion_complete"] is (index == 5)
        assert state["solver_run"] is False and state["training_run"] is False


    def test_pending_review_blocks_then_allows_successor(self):
        batch = self.batch
        draft(batch, 0)
        assert materialize(batch, 0)["status"] == "STRUCTURAL_PASS_ONLY"
        draft(batch, 1)
        assert materialize(batch, 1)["status"] == "PREDECESSOR_NOT_READY"
        assert not (controller.scene_folder(batch, ROOT, controller.SCENES[1]) / "CUMULATIVE_SOURCE.json").exists()
        review(batch, 0)
        assert materialize(batch, 1)["status"] == "STRUCTURAL_PASS_ONLY"


    def test_invalid_scene_fails_once_and_stops_root_even_after_file_repair(self):
        batch = self.batch
        draft(batch, 0, count=11)
        assert materialize(batch, 0)["status"] == "STRUCTURAL_FAILURE"
        receipt_paths = list((controller.scene_folder(batch, ROOT, controller.SCENES[0]) / "receipts").glob("*.json"))
        original = {str(path): path.read_bytes() for path in receipt_paths}
        draft(batch, 0, count=12)
        assert materialize(batch, 0)["status"] == "ROOT_STOPPED"
        assert all(Path(path).read_bytes() == content for path, content in original.items())
        assert not controller.progress(batch)["full_expansion_complete"]


    def test_full_history_and_different_reviewer_required(self):
        for index, (reviewer, omit_old) in enumerate([("/author", False), ("/reviewer", True)]):
            batch = make_batch(self.batch / str(index))
            draft(batch, 0)
            materialize(batch, 0)
            review(batch, 0, reviewer=reviewer, omit_old=omit_old)
            state = controller.progress(batch)["roots"][0]
            assert state["root_stopped"] is True
            assert state["ready_scene_count"] == 0

    def test_hold_is_terminal_and_cannot_be_replaced_by_ready(self):
        batch = self.batch
        draft(batch, 0)
        materialize(batch, 0)
        review(batch, 0, decision="HOLD")
        assert controller.progress(batch)["roots"][0]["root_stopped"] is True
        review(batch, 0, decision="READY")
        assert controller.progress(batch)["roots"][0]["root_stopped"] is True
        assert materialize(batch, 1)["status"] == "ROOT_STOPPED"


    def test_accepted_review_and_source_cannot_be_changed(self):
        batch = self.batch
        draft(batch, 0)
        materialize(batch, 0)
        review(batch, 0)
        assert controller.progress(batch)["roots"][0]["ready_scene_count"] == 1
        folder = controller.scene_folder(batch, ROOT, controller.SCENES[0])
        value = controller.read(folder / "SOURCE_REVIEW.json")
        value["reviewer"] = "/another-reader"
        write(folder / "SOURCE_REVIEW.json", value)
        state = controller.progress(batch)["roots"][0]
        assert state["root_stopped"] is True
        assert state["scenes"][0]["reason"] == "accepted_review_was_modified"


    def test_existing_history_prefix_corruption_is_detected(self):
        batch = self.batch
        draft(batch, 0)
        materialize(batch, 0)
        path = controller.scene_folder(batch, ROOT, controller.SCENES[0]) / "CUMULATIVE_SOURCE.json"
        value = controller.read(path)
        value["turns"][0]["text"] = "Unauthorized historical rewrite."
        write(path, value)
        state = controller.progress(batch)["roots"][0]
        assert state["root_stopped"] is True
        assert state["scenes"][0]["reason"] == "cumulative_source_not_exact_prefix_plus_draft"


    def test_three_B_events_must_have_distinct_reservations(self):
        batch = self.batch
        reservations = controller.read(batch / "SOURCE_RESERVATIONS.json")
        reservations["allocation"][1]["source_id"] = reservations["allocation"][0]["source_id"]
        write(batch / "SOURCE_RESERVATIONS.json", reservations)
        for index in range(3):
            draft(batch, index)
            materialize(batch, index)
            review(batch, index)
        draft(batch, 3)
        result = materialize(batch, 3)
        assert result["status"] == "STRUCTURAL_FAILURE"
        assert result["reason"] == "duplicate_reserved_source_id"


    def test_materialization_is_idempotent_without_overwriting_files(self):
        batch = self.batch
        draft(batch, 0)
        result = materialize(batch, 0)
        folder = controller.scene_folder(batch, ROOT, controller.SCENES[0])
        before = {str(path): path.read_bytes() for path in folder.rglob("*") if path.is_file()}
        assert materialize(batch, 0)["status"] == "AWAITING_REVIEW"
        assert {str(path): path.read_bytes() for path in folder.rglob("*") if path.is_file()} == before
        assert Path(result["receipt_path"]).exists()

if __name__ == "__main__":
    unittest.main()
