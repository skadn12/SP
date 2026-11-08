"""Offline runner and independent request replay for bounded scanner repairs."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import runner as r
import audit_sequential as audit


FROZEN = r.HERE.parent / "h4_sequential_ab_v34/runs/inherited_a3_b3_01/requests"


def frozen_case(prefix="007"):
    directory = next(FROZEN.glob(prefix + "-*"))
    payload = r.base.read(directory / "REQUEST.json")
    context = audit.decode_context(payload, "\nEVENT_STATE_INPUT_JSON\n")
    raw = json.loads(r.base.read(directory / "RESPONSE.json")["text"])
    # Explicitly labelled offline single-contract fixture, not a rewritten run.
    raw.pop("parent_goal_checks", None)
    raw["route_fulfillment"].pop("parent_goal_checks", None)
    return context, raw


class OfflineCaller:
    def __init__(self, run, responses):
        self.run, self.responses, self.calls = run, list(responses), []

    def call(self, label, payload, proof):
        value = self.responses.pop(0)
        if callable(value):
            value = value(payload)
        raw = value if isinstance(value, str) else json.dumps(value)
        directory = self.run / "requests" / f"{len(self.calls)+1:03d}-{label}"
        for name, content in (("REQUEST.json", payload), ("PROOF.json", proof),
                              ("RESPONSE.json", {"text": raw, "offline_fixture": True})):
            r.base.write(directory / name, content)
        self.calls.append((payload, proof, directory))
        return raw, directory


class RunnerScannerV35Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="v35_scanner_offline_")
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.context, self.raw = frozen_case()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch("socket.socket", side_effect=AssertionError("offline_test_no_network")))
        self.stack.enter_context(patch.object(r, "branch_history_ok", return_value=[]))
        self.stack.enter_context(patch.object(r, "history_proof", return_value={}))

    def execute(self, responses):
        c = self.context
        caller = OfflineCaller(self.run, responses)
        value = r.extract_event_state(caller, self.run, c["root_id"], c["route"],
            c["cumulative_history"], c["dataset_cards"], c["scene_plan"], c["latest_state"], 1,
            purpose_input=c.get("purpose_input"))
        return caller, value

    def valid_negative(self):
        raw = deepcopy(self.raw)
        for row in raw["route_fulfillment"]["history_goal_checks"]:
            if row["relation"] == "executes_or_prepares":
                row["relation"] = "serves_existing_goal"
        return raw

    def test_single_group_negative_passes_schema_but_remains_replan(self):
        caller, result = self.execute([self.valid_negative()])
        self.assertEqual(len(caller.calls), 1)
        self.assertTrue(r.event_contract.requires_replan(result))
        payload, proof, directory = caller.calls[0]
        context = audit.decode_context(payload, "\nEVENT_STATE_INPUT_JSON\n")
        self.assertNotIn("parent_goal_checks_template", context)
        self.assertIs(proof["require_parent_goal_checks"], False)
        self.assertEqual(proof["scanner_contract"], "single_goal_v35")
        self.assertFalse((directory / "PARENT_GOAL_DECISION_PROVENANCE.json").exists())
        with patch.object(audit, "r", r):
            replay = audit.replay_event_request_context(payload, self.context["root_id"],
                self.context["route"], self.context["scene_plan"], None)
        self.assertEqual(replay, context)

    def test_unrepairable_scan_does_not_spend_second_call(self):
        broken = self.valid_negative()
        broken["root_id"] = "wrong-root"
        caller, result = self.execute([broken])
        self.assertIsNone(result)
        self.assertEqual(len(caller.calls), 1)
        directory = caller.calls[0][2]
        spec = r.base.read(directory / "FORMAT_REPAIR_SPEC.json")
        self.assertTrue(spec["unrepairable_failures"])
        self.assertEqual(r.base.read(directory / "VALIDATION.json")["status"], "REJECT")

    def test_malformed_json_stops_without_a_second_call_or_runtime_error(self):
        caller, result = self.execute(["{not-json"])
        self.assertIsNone(result)
        self.assertEqual(len(caller.calls), 1)
        self.assertTrue(r.base.read(caller.calls[0][2] / "FORMAT_REPAIR_SPEC.json")["unrepairable_failures"])

    def test_nonobject_json_stops_without_a_second_call_or_runtime_error(self):
        caller, result = self.execute(["[]"])
        self.assertIsNone(result)
        self.assertEqual(len(caller.calls), 1)
        self.assertTrue(r.base.read(caller.calls[0][2] / "FORMAT_REPAIR_SPEC.json")["unrepairable_failures"])

    def patch_response(self, payload):
        return {"root_id": self.context["root_id"], "route": self.context["route"],
            "scene_id": self.context["scene_id"], "repair_patches": [
                {"op": "replace", "path": ["route_fulfillment", "history_goal_checks", index, "relation"],
                 "value": "serves_existing_goal"}
                for index, row in enumerate(self.raw["route_fulfillment"]["history_goal_checks"])
                if row["relation"] == "executes_or_prepares"]}

    def test_exact_enum_patch_preserves_negative_and_audits_compact_request(self):
        caller, result = self.execute([self.raw, self.patch_response])
        self.assertEqual(len(caller.calls), 2)
        self.assertTrue(r.event_contract.requires_replan(result))
        self.assertEqual(result["card_plan_checks"], self.raw["card_plan_checks"])
        initial, _proof, first_dir = caller.calls[0]
        payload, proof, directory = caller.calls[1]
        spec = r.base.read(first_dir / "FORMAT_REPAIR_SPEC.json")
        self.assertEqual(spec["repair_mode"], "scanner_exact_patch_v35")
        actual = audit.decode_context(payload, "\nEVENT_STATE_INPUT_JSON\n")
        self.assertNotIn("latest_state", actual)
        self.assertNotIn("route_fulfillment_template", actual)
        self.assertIn("SCANNER_PATCH_INPUT_JSON", payload["messages"][0]["content"])
        self.assertFalse((directory / "TIMELINE_PATCH_PROVENANCE.json").exists())
        self.assertTrue((directory / "SCANNER_PATCH_PROVENANCE.json").exists())
        self.assertTrue(r.base.read(directory / "FORMAT_REPAIR_PRESERVATION.json")["unchanged_negative_findings_required"])
        original = audit.decode_context(initial, "\nEVENT_STATE_INPUT_JSON\n")
        with patch.object(audit, "r", r):
            replay = audit.replay_event_request_context(payload, self.context["root_id"],
                self.context["route"], self.context["scene_plan"], spec, original)
            self.assertEqual(replay["latest_state"], original["latest_state"])
            self.assertEqual(replay["targeted_repair"], spec)
            tampered = deepcopy(payload)
            tampered["messages"][0]["content"] += "\nUse a different judgment."
            with self.assertRaisesRegex(ValueError, "event_local_patch_request_replay_changed"):
                audit.replay_event_request_context(tampered, self.context["root_id"],
                    self.context["route"], self.context["scene_plan"], spec, original)

    def test_patch_cannot_flip_negative_to_independent(self):
        def invalid(payload):
            response = self.patch_response(payload)
            response["repair_patches"][0]["value"] = "independent"
            return response
        caller, result = self.execute([self.raw, invalid])
        self.assertIsNone(result)
        self.assertEqual(len(caller.calls), 2)
        directory = caller.calls[1][2]
        self.assertEqual(r.base.read(directory / "FORMAT_REPAIR_MERGE.json")["status"], "REJECT")
        self.assertTrue(r.base.read(directory / "VALIDATION.json")["merge_failed"])
        self.assertFalse((directory / "SCANNER_PATCH_PROVENANCE.json").exists())

    def test_unrepairable_retry_request_is_rejected_before_replay(self):
        c = self.context
        payload = r.event_contract.build_request(c["root_id"], c["route"], c["scene_plan"],
            c["cumulative_history"], c["dataset_cards"], c["latest_state"],
            purpose_input=c.get("purpose_input"), require_goal_checks=True, require_card_checks=True)
        context = audit.decode_context(payload, "\nEVENT_STATE_INPUT_JSON\n")
        with patch.object(audit, "r", r), self.assertRaisesRegex(ValueError, "unrepairable"):
            audit.replay_event_request_context(payload, c["root_id"], c["route"], c["scene_plan"],
                {"repair_mode": "scanner_exact_patch_v35", "unrepairable_failures": ["unknown"]}, context)


if __name__ == "__main__":
    unittest.main()
