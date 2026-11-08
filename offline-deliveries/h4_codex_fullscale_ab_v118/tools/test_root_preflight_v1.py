import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from root_preflight_v1 import SCENES, build_anchors, check_anchors, check_qa, execute_checked_steps, load_same_root


class SameRootTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.unit = Path(self.temp.name) / "root123"
        self.folder = self.unit / "roots/root123"
        self.folder.mkdir(parents=True)
        self.source_path = self.folder / "CUMULATIVE_SOURCE.json"
        self.root = "cc-train-episode-123"
        self.source = {"root_id": self.root, "turns": [
            {"turn_ref": scene + ":T001", "scene_ref": scene, "speaker": "A", "text": "Exact local text " + scene}
            for scene in SCENES]}
        self.source_path.write_text(json.dumps(self.source))
        _, _, self.turns = load_same_root(self.unit, self.root)
        self.selection = {"root_id": self.root, "slots": [{"slot_id": str(k), "anchors": [
            {"anchor_id": f"K{k}{i}", "turn_ref": scene + ":T001"} for i, scene in enumerate(SCENES, 1)]}
            for k in (1, 2)]}
        self.plan = build_anchors(self.selection, self.root, self.source_path, self.turns)

    def test_full_text_loaded_from_same_root(self):
        check_anchors(self.plan, self.root, self.source_path, self.turns)
        self.assertEqual(self.plan["slots"][0]["anchors"][0]["full_turn_quote"], self.source["turns"][0]["text"])

    def test_other_unit_rejected(self):
        with self.assertRaisesRegex(ValueError, "UNIT_ROOT"):
            load_same_root(self.unit, "cc-train-episode-124")

    def test_other_source_root_rejected(self):
        self.source["root_id"] = "cc-train-episode-124"
        self.source_path.write_text(json.dumps(self.source))
        with self.assertRaisesRegex(ValueError, "SOURCE_ROOT"):
            load_same_root(self.unit, self.root)

    def test_cross_root_parent_turn_rejected(self):
        self.source["turns"][0]["turn_ref"] = "cc-train-episode-124:P01:T001"
        self.source_path.write_text(json.dumps(self.source))
        with self.assertRaisesRegex(ValueError, "PARENT_TURN"):
            load_same_root(self.unit, self.root)

    def test_wrong_selection_root_rejected(self):
        self.selection["root_id"] = "cc-train-episode-124"
        with self.assertRaisesRegex(ValueError, "SELECTION_ROOT"):
            build_anchors(self.selection, self.root, self.source_path, self.turns)

    def test_injected_quote_in_selection_rejected(self):
        self.selection["slots"][0]["anchors"][0]["full_turn_quote"] = "Other root"
        with self.assertRaisesRegex(ValueError, "ONLY_IDS_AND_REFS"):
            build_anchors(self.selection, self.root, self.source_path, self.turns)

    def test_cross_root_quote_rejected(self):
        self.plan["slots"][1]["anchors"][0]["full_turn_quote"] = "Quote copied from another root"
        with self.assertRaisesRegex(ValueError, "FULL_TURN_MISMATCH"):
            check_anchors(self.plan, self.root, self.source_path, self.turns)

    def test_wrong_speaker_rejected(self):
        self.plan["slots"][0]["anchors"][0]["speaker"] = "B"
        with self.assertRaisesRegex(ValueError, "FULL_TURN_MISMATCH"):
            check_anchors(self.plan, self.root, self.source_path, self.turns)

    def test_source_hash_change_rejected(self):
        self.source_path.write_text(json.dumps(self.source) + "\n")
        with self.assertRaisesRegex(ValueError, "HASH_MISMATCH"):
            check_anchors(self.plan, self.root, self.source_path, self.turns)

    def test_duplicate_scene_rejected(self):
        self.plan["slots"][0]["anchors"][1] = copy.deepcopy(self.plan["slots"][0]["anchors"][0])
        with self.assertRaises(ValueError):
            check_anchors(self.plan, self.root, self.source_path, self.turns)

    def test_qa_quote_and_fact_refs(self):
        draft = {"facts": [{"fact_id": str(i), "source_turn_ref": scene + ":T001"} for i, scene in enumerate(SCENES)],
                 "ordering": [{"slot_id": str(k), "items": [{"item_id": f"I{i}", "anchor_id": f"K{k}{i}",
                  "support_quote": "Exact local text"} for i in range(1, 7)]} for k in (1, 2)]}
        check_qa(draft, self.plan, self.root, self.source_path, self.turns)
        draft["ordering"][1]["items"][0]["support_quote"] = "A quote from another root"
        with self.assertRaisesRegex(ValueError, "QUOTE_OUTSIDE"):
            check_qa(draft, self.plan, self.root, self.source_path, self.turns)


class SequenceTests(unittest.TestCase):
    def test_first_failure_stops(self):
        calls = []
        def runner(argv, **kw):
            calls.append(argv)
            raise subprocess.CalledProcessError(1, argv)
        with self.assertRaises(subprocess.CalledProcessError):
            execute_checked_steps([{"argv": ["first"]}, {"argv": ["second"]}], runner)
        self.assertEqual(calls, [["first"]])

    def test_missing_required_artifact_stops_before_execution(self):
        calls = []
        with self.assertRaisesRegex(ValueError, "PREDECESSOR_ARTIFACT_MISSING"):
            execute_checked_steps([{"argv": ["first"], "requires": [{"path": "/does-not-exist-root-preflight"}]}],
                                  lambda *args, **kw: calls.append(args))
        self.assertEqual(calls, [])

    def test_expected_output_required_before_next_step(self):
        calls = []
        def runner(argv, **kw):
            calls.append(argv); return subprocess.CompletedProcess(argv, 0)
        with self.assertRaisesRegex(ValueError, "EXPECTED_OUTPUT_MISSING"):
            execute_checked_steps([{"argv": ["first"], "expects": ["/does-not-exist-root-preflight"]},
                                   {"argv": ["second"]}], runner)
        self.assertEqual(calls, [["first"]])

    def test_two_successful_steps(self):
        def runner(argv, **kw):
            self.assertFalse(kw["shell"]); self.assertTrue(kw["check"])
            return subprocess.CompletedProcess(argv, 0)
        self.assertEqual(execute_checked_steps([{"name": "one", "argv": ["one"]},
                                                {"name": "two", "argv": ["two"]}], runner), ["one", "two"])


if __name__ == "__main__":
    unittest.main()
