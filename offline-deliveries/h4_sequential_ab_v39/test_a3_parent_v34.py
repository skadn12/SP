"""Offline A3 admission: real accepted A2 plus explicitly MOCK new S03.

No model or network calls. The full A3 wrapper runs using the existing scripted
transport, then artifact mutations test the independent parent admission gates.
"""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch

import a3_continuation as a
from test_sequential_flow import SequentialProvider, initialize_mock

r = a.r


def overwrite_mock(path, value):
    """Intentional tampering of temporary MOCK artifacts only."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


class A3ParentAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="A3_PARENT_MOCK_ONLY_")
        self.addCleanup(self.temp.cleanup)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.delivery = Path(self.temp.name) / "delivery"
        self.delivery.mkdir()
        (self.delivery / "runs").mkdir()
        (self.delivery / "POLICY.md").write_text("MOCK ONLY: offline admission regression.\n")
        for source in r.HERE.glob("*.py"):
            shutil.copy2(source, self.delivery / source.name)
        self.run = self.delivery / "runs/MOCK_ONLY"
        self.fake = SequentialProvider()
        self.stack.enter_context(patch.object(r, "HERE", self.delivery))
        self.stack.enter_context(patch.object(r, "LOCK_PATH", Path(self.temp.name) / "inventory.lock"))
        self.stack.enter_context(patch.object(r, "MIN_HTTP_GAP_SECONDS", 0))
        self.stack.enter_context(patch.object(r.Caller, "__init__",
            lambda caller, run: initialize_mock(caller, run, self.fake)))
        self.stack.enter_context(patch("socket.socket", side_effect=AssertionError("network_forbidden_in_offline_test")))
        self.before_calls = []
        def before_request():
            reservation = r.base.read(self.run / "SOURCE_RESERVATIONS.json")
            self.assertEqual(reservation["reservations"], [])
            self.assertTrue((self.run / "LIVE_STARTED.json").exists())
            self.before_calls.append(True)
        self.fake.before_request = before_request
        a.prepare(self.run)
        report = a.live(self.run)
        self.assertEqual(report["status"], "A3_COMPLETE", report)
        self.assertEqual(a.audit(self.run)["status"], "PASS")
        self.folder = self.run / "roots" / a.ROOT / "A"
        self.accepted = r.base.read(self.folder / "S03.ATTEMPTS.json")[-1]
        self.realizer = Path(self.accepted["request"])
        self.reviewer = Path(self.accepted["review_request"])

    def refresh_merge(self, *, refresh_audit=False):
        """Model a rebuilt file index; an old audit must not bless new artifacts."""
        path = self.run / "MERGE_MANIFEST.json"
        merge = r.base.read(path)
        merge["files"] = r.base.bindings([Path(row["path"]) for row in merge["files"]])
        overwrite_mock(path, merge)
        if refresh_audit:
            audit_path = self.run / "FINAL_AUDIT.json"
            audit = r.base.read(audit_path)
            # This is deliberately not a replayed audit. Tests below exercise
            # parent-side scope/binding checks even with an asserted audit.
            audit["manifest_sha256"] = r.base.digest(r.base.read(self.run / "MANIFEST.json"))
            audit["merge_manifest_sha256"] = r.base.digest(merge)
            overwrite_mock(audit_path, audit)

    def test_complete_mock_S03_requires_real_A2_prefix_and_no_source_reservations(self):
        admitted = r.checked_parent_a3(self.run, a.ROOT)
        turns = r.base.read(admitted["cumulative_path"])["turns"]
        inherited = a._history(self.run)
        self.assertEqual(turns[:120], inherited)
        self.assertEqual(r.base.digest(inherited), a.HISTORY_SHA)
        self.assertTrue(turns[120:])
        self.assertTrue(all(row["turn_ref"].startswith("R02:A:S03:") for row in turns[120:]))
        self.assertEqual(admitted["result"]["accepted_scenes"], 3)
        self.assertEqual(len(self.before_calls), 5)
        self.assertEqual({row["role"] for row in self.fake.observed},
                         {"state_extractor", "planner", "event_state_extractor", "realizer", "scene_reviewer"})

    def test_missing_or_shortfall_parent_audit_is_rejected(self):
        path = self.run / "FINAL_AUDIT.json"
        audit = r.base.read(path)
        audit["generation_status"] = "A3_SHORTFALL"
        overwrite_mock(path, audit)
        with self.assertRaises(ValueError):
            r.checked_parent_a3(self.run, a.ROOT)
        path.unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            r.checked_parent_a3(self.run, a.ROOT)

    def test_old_pass_audit_cannot_bless_reindexed_changed_raw_review(self):
        path = self.reviewer / "RESPONSE.json"
        response = r.base.read(path)
        raw = json.loads(response["text"])
        raw["reason"] = "MOCK ONLY: different response after the original audit."
        response["text"] = json.dumps(raw)
        overwrite_mock(path, response)
        self.refresh_merge()
        with self.assertRaisesRegex(ValueError, "audit|binding|changed"):
            r.checked_parent_a3(self.run, a.ROOT)

    def test_changed_model_dialogue_rejected_even_with_asserted_fresh_audit(self):
        path = self.realizer / "RESPONSE.json"
        response = r.base.read(path)
        raw = json.loads(response["text"])
        raw["turns"][0]["text"] = "MOCK ONLY: substituted model output."
        response["text"] = json.dumps(raw)
        overwrite_mock(path, response)
        self.refresh_merge(refresh_audit=True)
        with self.assertRaisesRegex(ValueError, "scene|dialogue|response|audit"):
            r.checked_parent_a3(self.run, a.ROOT)

    def test_pass_review_for_different_dialogue_is_not_reusable(self):
        path = self.reviewer / "REQUEST.json"
        payload = r.base.read(path)
        prefix, body = payload["messages"][0]["content"].split("\nREVIEW INPUT DATA:\n", 1)
        context, end = json.JSONDecoder().raw_decode(body)
        context["new_scene"]["turns"][0]["text"] = "MOCK ONLY: a different reviewed scene."
        payload["messages"][0]["content"] = prefix + "\nREVIEW INPUT DATA:\n" + json.dumps(context) + body[end:]
        overwrite_mock(path, payload)
        self.refresh_merge(refresh_audit=True)
        with self.assertRaisesRegex(ValueError, "review|binding|scene"):
            r.checked_parent_a3(self.run, a.ROOT)

    def test_pass_attempt_must_match_its_saved_acceptance(self):
        path = self.realizer / "ACCEPTANCE.json"
        record = r.base.read(path)
        record["status"] = "REJECT"
        overwrite_mock(path, record)
        self.refresh_merge(refresh_audit=True)
        with self.assertRaisesRegex(ValueError, "acceptance|audit"):
            r.checked_parent_a3(self.run, a.ROOT)

    def test_A2_cannot_be_presented_as_three_accepted_scenes(self):
        result = r.base.read(self.folder / "RESULT.json")
        result.update(accepted_scenes=2, new_accepted_scenes=0)
        result["scene_metadata"] = result["scene_metadata"][:2]
        report = r.base.read(self.run / "REPORT.json")
        report["result"] = result
        overwrite_mock(self.folder / "RESULT.json", result)
        overwrite_mock(self.run / "REPORT.json", report)
        self.refresh_merge(refresh_audit=True)
        with self.assertRaisesRegex(ValueError, "inventory|report|audit"):
            r.checked_parent_a3(self.run, a.ROOT)


if __name__ == "__main__":
    unittest.main()
