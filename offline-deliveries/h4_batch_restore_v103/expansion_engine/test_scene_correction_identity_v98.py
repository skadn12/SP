"""Integration: quote repair must not erase another explicit identity negative.

Hand-authored reviews of one unchanged temporary body; no HTTP or semantic
accuracy claim. Both complete validation results and correction gates persist.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import runner
import test_event_identity_v98 as identity_fixtures
from test_runner_semantic_integration import FakeCaller, generator_stub


class SceneCorrectionIdentityV98Tests(unittest.TestCase):
    def test_quote_retry_cannot_erase_elsewhere_nested_negative_or_required_true(self):
        fixture = identity_fixtures.EventIdentityV98Tests(); fixture.setUp()
        body, good = fixture.scene_fixture()
        body["turns"].extend([
            {"speaker": "Elena", "text": "The front cable started sticking on Tuesday."},
            {"speaker": "Milan", "text": "Bring the bicycle to the workshop entrance."},
            {"speaker": "Elena", "text": "I can leave it with you at nine tomorrow."},
            {"speaker": "Milan", "text": "The inspection bench will be free then."},
            {"speaker": "Elena", "text": "The rear brake still responds normally."},
            {"speaker": "Milan", "text": "I will check both cables during inspection."},
            {"speaker": "Elena", "text": "Please call before replacing any components."},
            {"speaker": "Milan", "text": "I will send the estimate before ordering parts."},
            {"speaker": "Elena", "text": "My contact number is on the repair form."},
            {"speaker": "Milan", "text": "The form can stay beside the workshop register."},
        ])
        # The reviewer explicitly judges the mapped new request missing. The
        # fixture body itself is irrelevant to whether format retry may erase
        # that judgment: a body rewrite would need a separate realizer request.
        first = deepcopy(good)
        first["checks"][0]["evidence"][0]["quote"] = "BAD QUOTE NOT IN ANY TURN"
        first["card_checks"][0]["comparison"]["request_acceptance"].update(
            required=True, verdict="revise", request_evidence=[])
        second = deepcopy(first)
        second["checks"][0]["evidence"] = deepcopy(good["checks"][0]["evidence"])
        second["card_checks"][0]["comparison"]["request_acceptance"].update(
            required=False, verdict="compatible", request_evidence=[], acceptance_evidence=[])
        snapshots = deepcopy((first, second, body))
        with tempfile.TemporaryDirectory() as tmp:
            caller = FakeCaller(tmp, [("realizer", body), ("scene_reviewer", first), ("scene_reviewer", second)])
            validator = runner.realization_review.validate_review
            with patch.object(runner, "make_request", side_effect=generator_stub), \
                 patch.object(runner.realization_review, "validate_review", wraps=validator) as full_validation:
                result, audits = runner.produce_scene(caller, Path(tmp), "r", "dataset_new_event",
                    fixture.history, [fixture.card], fixture.scene, [])
            self.assertIsNone(result)
            self.assertEqual(full_validation.call_count, 2)
            self.assertEqual([x["role"] for x in caller.calls], ["realizer", "scene_reviewer", "scene_reviewer"])
            self.assertEqual(audits[-1]["status"], "REVIEWER_FAILURE_NO_DIALOGUE_REWRITE")
            first_dir, second_dir = caller.calls[1]["folder"], caller.calls[2]["folder"]
            first_record = runner.base.read(first_dir / "REVIEW.json")
            second_full = runner.base.read(second_dir / "FULL_VALIDATION_BEFORE_CORRECTION_GATE.json")
            second_record = runner.base.read(second_dir / "REVIEW.json")
            progress = runner.base.read(second_dir / "CORRECTION_PROGRESS.json")
            self.assertFalse(first_record["schema_valid"])
            self.assertIn("request_acceptance:revise", "\n".join(first_record["failures"]))
            self.assertEqual(second_full["status"], "PASS")
            self.assertTrue(second_full["schema_valid"])
            self.assertEqual(second_record["status"], "REJECT")
            self.assertFalse(second_record["schema_valid"])
            self.assertEqual(progress["status"], "REJECT")
            self.assertTrue(progress["errors"])
            self.assertTrue(any("protected" in e["code"] or "negative" in e["code"]
                                or "applicability" in e["code"] for e in progress["errors"]), progress)
            for directory in (first_dir, second_dir):
                self.assertTrue((directory / "REVIEW.json").is_file())
                self.assertTrue((directory / "CORRECTION_PROGRESS.json").is_file())
            repair_payload = caller.calls[2]["payload"]["messages"][0]["content"]
            self.assertIn("correction_contract", repair_payload)
            self.assertIn("protected_judgments", repair_payload)
        self.assertEqual((first, second, body), snapshots)


if __name__ == "__main__": unittest.main()
