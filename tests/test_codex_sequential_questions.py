"""Check that the new QA adapter covers all six A/B scenes."""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from test_codex_sequential_ab import ROOT, draft, make_batch, materialize, review, write
import codex_sequential_ab as controller
import codex_sequential_questions as questions


class SequentialQuestionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.batch = make_batch(Path(self.temporary.name))
        cfg_path = self.batch / "RUN_CONFIG.json"
        cfg = controller.read(cfg_path)
        cfg.update(batch_id="synthetic_cross_scene", seed_namespace="fixed-before-authoring",
                   multiselect_target_cardinalities=[2, 3, 4, 5], scope_scene_refs=list(controller.SCENES))
        write(cfg_path, cfg)
        for index in range(6):
            draft(self.batch, index)
            materialize(self.batch, index)
            review(self.batch, index)
        controller.progress(self.batch)
        self.folder = controller.root_folder(self.batch, ROOT)
        final = controller.scene_folder(self.batch, ROOT, controller.SCENES[-1])
        for filename in ("CUMULATIVE_SOURCE.json", "SOURCE_REVIEW.json"):
            (self.folder / filename).write_bytes((final / filename).read_bytes())
        source = controller.read(self.folder / "CUMULATIVE_SOURCE.json")
        turns = {turn["turn_ref"]: turn for turn in source["turns"]}
        facts = [{"fact_id": f"F{i}", "source_turn_ref": f"{scene}:T001",
                  "true_text": f"Fact from {scene}.", "false_text": f"Contradiction from {scene}."}
                 for i, scene in enumerate(controller.SCENES, 1)]
        anchors = [{"anchor_id": f"K{i}", "turn_ref": f"{scene}:T001", "speaker": "Alice",
                    "full_turn_quote": turns[f"{scene}:T001"]["text"]} for i, scene in enumerate(controller.SCENES, 1)]
        write(self.folder / "ORDERING_ANCHOR_PLAN.json", {
            "source_sha256": controller.digest(self.folder / "CUMULATIVE_SOURCE.json"),
            "slots": [{"slot_id": f"{slot:02d}", "anchors": anchors} for slot in range(1, 3)]})
        write(self.folder / "QA_DRAFT.json", {
            "facts": facts, "stem": "Across " + ", ".join(controller.SCENES) + ", select all supported statements.",
            "ordering": [{"slot_id": f"{slot:02d}", "items": [
                {"item_id": f"I{i}", "anchor_id": anchor["anchor_id"], "text": f"Statement for scene {i}.",
                 "support_quote": anchor["full_turn_quote"]} for i, anchor in enumerate(anchors, 1)]} for slot in range(1, 3)]})

    def publish(self):
        with contextlib.redirect_stdout(io.StringIO()):
            questions.publish_assets(self.batch)

    def test_all_six_scene_facts_and_orderings_are_published_once_per_root(self):
        self.publish()
        public = controller.read(self.folder / "PUBLIC_ASSETS.json")
        private = controller.read(self.folder / "PRIVATE_ASSET_DESIGN.json")
        self.assertEqual(public["scope_scene_refs"], list(controller.SCENES))
        self.assertEqual(len(public["statements"]), 12)
        self.assertEqual(len(public["ordering_questions"]), 2)
        self.assertEqual(len(private["facts"]), 6)
        for question in public["ordering_questions"]:
            self.assertTrue(all(scene in question["question"] for scene in controller.SCENES))

    def test_six_facts_cannot_cover_only_five_scenes(self):
        path = self.folder / "QA_DRAFT.json"
        value = controller.read(path)
        value["facts"][-1]["source_turn_ref"] = value["facts"][0]["source_turn_ref"]
        write(path, value)
        with self.assertRaisesRegex(ValueError, "facts_require_one_per_scene"):
            self.publish()
        self.assertFalse((self.folder / "PUBLIC_ASSETS.json").exists())

    def test_ordering_slot_requires_one_anchor_per_scene(self):
        path = self.folder / "ORDERING_ANCHOR_PLAN.json"
        value = controller.read(path)
        anchor = value["slots"][0]["anchors"][-1]
        scene = controller.SCENES[0]
        anchor.update(turn_ref=f"{scene}:T002", speaker="Bob", full_turn_quote="Scene 0, statement 2.")
        write(path, value)
        draft_path = self.folder / "QA_DRAFT.json"
        write(draft_path, controller.read(draft_path))
        with self.assertRaisesRegex(ValueError, "ordering_requires_one_anchor_per_scene"):
            self.publish()

    def test_order_anchor_can_use_six_scenes_but_cannot_use_baseline(self):
        source = controller.read(self.folder / "CUMULATIVE_SOURCE.json")
        turns = {turn["turn_ref"]: turn for turn in source["turns"]}
        positions = {turn["turn_ref"]: i for i, turn in enumerate(source["turns"])}
        anchor = {"turn_ref": f"{controller.SCENES[0]}:T001", "quote": "Scene 0, statement 1."}
        self.assertEqual(questions.anchor_key(anchor, turns, positions, list(controller.SCENES))[0], 2)
        with self.assertRaisesRegex(ValueError, "anchor_outside_public_scope"):
            questions.anchor_key({"turn_ref": "P01:T001", "quote": "We planned a garden."}, turns, positions, list(controller.SCENES))


if __name__ == "__main__":
    unittest.main()
