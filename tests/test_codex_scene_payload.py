"""Regression of the real author-payload -> six-scene gate, without model calls."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from test_codex_sequential_ab import ROOT, make_batch, review
import codex_scene_payload as payload
import codex_sequential_ab as controller


def messages():
    return [{"speaker": "Alice" if i % 2 else "Bob", "text": f"Fixture statement {i}."}
            for i in range(1, 15)]


class ScenePayloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.batch = make_batch(Path(self.temporary.name))

    def test_serialized_first_drafts_pass_all_six_materializations_and_review_gates(self):
        for index, scene in enumerate(controller.SCENES):
            target = payload.write_new(self.batch, ROOT, scene, "/author", {"messages": messages()})
            value = controller.read(target)
            self.assertEqual(value["turns"][0]["session_ref"], scene)
            self.assertEqual(value["turns"][-1]["turn_ref"], scene + ":T014")
            self.assertEqual([{"speaker": t["speaker"], "text": t["text"]} for t in value["turns"]], messages())
            if index >= 3:
                self.assertEqual(value["source_id"], f"soda:train:{index - 2}")
            self.assertEqual(controller.materialize_scene(self.batch, ROOT, scene)["status"], "STRUCTURAL_PASS_ONLY")
            if index < 5:
                with self.assertRaisesRegex(ValueError, "SCENE_NOT_READY_FOR_FIRST_DRAFT"):
                    payload.write_new(self.batch, ROOT, controller.SCENES[index + 1], "/author", {"messages": messages()})
            review(self.batch, index)
            state = controller.progress(self.batch)["roots"][0]
            self.assertEqual(state["ready_scene_count"], index + 1)
        self.assertTrue(state["full_expansion_complete"])
        self.assertEqual(state["scenes"][-1]["cumulative_turn_count"], 86)

    def test_bad_metadata_and_unknown_speaker_cannot_create_a_draft(self):
        for field, value in (("session_ref", ROOT + ":R02:A:S01"), ("speaker", "Charlie")):
            bad = messages()
            bad[0][field] = value
            with self.assertRaisesRegex(ValueError, "ONLY_SPEAKER_AND_TEXT_ARE_AUTHORED|INVALID_MESSAGE"):
                payload.write_new(self.batch, ROOT, controller.SCENES[0], "/author", {"messages": bad})
            self.assertFalse((controller.scene_folder(self.batch, ROOT, controller.SCENES[0]) / "EXPANSION_DRAFT.json").exists())

    def test_first_draft_and_failed_root_cannot_be_rewritten(self):
        target = payload.write_new(self.batch, ROOT, controller.SCENES[0], "/author", {"messages": messages()})
        original = target.read_bytes()
        with self.assertRaisesRegex(ValueError, "SCENE_NOT_READY_FOR_FIRST_DRAFT"):
            payload.write_new(self.batch, ROOT, controller.SCENES[0], "/author", {"messages": messages()})
        controller.materialize_scene(self.batch, ROOT, controller.SCENES[0])
        review(self.batch, 0, decision="HOLD")
        with self.assertRaisesRegex(ValueError, "ROOT_ALREADY_STOPPED_NO_RECOVERY"):
            payload.write_new(self.batch, ROOT, controller.SCENES[1], "/author", {"messages": messages()})
        self.assertEqual(original, target.read_bytes())


if __name__ == "__main__":
    unittest.main()
