"""Exact state citation repair regressions; fixtures are not live model judgments."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import latest_state as state
import runner


class StateCitationPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        origin = Path(__file__).resolve().parent.parent / (
            "h4_sequential_ab_v36/experiments/fixed_soda_a3_01/requests")
        first = next(origin.glob("002-*"))
        request = json.loads((first / "REQUEST.json").read_text())
        cls.context = json.loads(request["messages"][0]["content"].split("\nSTATE_INPUT_JSON\n")[1])
        cls.original = json.loads(json.loads((first / "RESPONSE.json").read_text())["text"])
        cls.history = cls.context["cumulative_history"]
        cls.expected_paths = ["facts[1].evidence[0].quote",
            "facts[4].evidence[0].turn_ref", "facts[4].evidence[1].turn_ref",
            "facts[5].evidence[0].turn_ref", "facts[5].evidence[1].turn_ref"]

    def validate(self, value):
        return state.validate_state(value, self.context["root_id"], self.context["route"],
            self.context["scene_id"], self.history)

    def repair_for(self, value):
        with self.assertRaises(state.StateValidationError) as caught:
            self.validate(value)
        return state.build_state_repair(caught.exception, value, self.history,
            require_local_citation_patch=True)

    def setUp(self):
        self.repair = self.repair_for(self.original)
        # Synthetic correct leaf values for a mechanical replay regression only.
        by_ref = {turn["turn_ref"]: turn["text"] for turn in self.history}
        self.patch = {"citation_repairs": [{"path": self.expected_paths[0],
            "value": by_ref[self.original["facts"][1]["evidence"][0]["turn_ref"]]}] + [
            {"path": row["path"], "value": row["candidate_turn_refs"][0]}
            for row in self.repair["citation_reference_candidates"]]}

    def merge(self, value, repair=None):
        return state.merge_state_repair(self.original, value, repair or self.repair, self.history)

    def test_recorded_mixed_failure_changes_exactly_five_leaves(self):
        self.assertEqual(self.repair["required_paths"], self.expected_paths)
        self.assertEqual(self.repair["unrepairable_failures"], [])
        original = deepcopy(self.original)
        merged = self.merge(self.patch)
        self.validate(merged)
        expected = deepcopy(original)
        for row in self.patch["citation_repairs"]:
            citation, leaf = state._citation_leaf(expected, row["path"])
            citation[leaf] = row["value"]
        self.assertEqual(merged, expected)
        self.assertEqual(self.original, original)
        for row in self.patch["citation_repairs"]:
            before, leaf = state._citation_leaf(original, row["path"])
            after, _ = state._citation_leaf(merged, row["path"])
            sibling = "quote" if leaf == "turn_ref" else "turn_ref"
            self.assertEqual(before[sibling], after[sibling])
        self.assertEqual(merged["facts"][0]["event"], original["facts"][0]["event"])

    def test_unlisted_valid_leaf_or_claim_cannot_be_changed(self):
        for path in ["facts[0].evidence[0].turn_ref", "facts[1].evidence[0].turn_ref",
                     "facts[4].evidence[0].quote", "facts[0].event"]:
            altered = deepcopy(self.patch)
            altered["citation_repairs"].append({"path": path, "value": "unauthorized"})
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.merge(altered)

    def test_patch_coverage_shape_and_literal_values_fail_closed(self):
        missing = deepcopy(self.patch); missing["citation_repairs"].pop()
        duplicate = deepcopy(self.patch); duplicate["citation_repairs"].append(duplicate["citation_repairs"][0])
        bad_quote = deepcopy(self.patch); bad_quote["citation_repairs"][0]["value"] = "fabricated quotation"
        bad_ref = deepcopy(self.patch); bad_ref["citation_repairs"][1]["value"] = "not-a-history-reference"
        extra_field = deepcopy(self.patch); extra_field["citation_repairs"][0]["statement"] = "edited"
        empty_value = deepcopy(self.patch); empty_value["citation_repairs"][0]["value"] = " "
        for value in [missing, duplicate, bad_quote, bad_ref, extra_field, empty_value,
                      self.original, {**self.patch, "facts": []}, {"citation_repairs": []}]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.merge(value)

    def test_tampered_spec_cannot_authorize_a_valid_reference(self):
        altered = deepcopy(self.repair)
        path = "facts[0].evidence[0].turn_ref"
        altered["required_paths"].append(path)
        citation, _ = state._citation_leaf(self.original, path)
        altered["errors"].append({"code": "state_evidence_reference_invalid", "path": path,
            "actual": citation["turn_ref"]})
        with self.assertRaisesRegex(ValueError, "spec_invalid"):
            self.merge(self.patch, altered)

    def test_non_citation_error_prevents_a_repair_request(self):
        value = deepcopy(self.original)
        value["facts"][0]["status"] = "invented-status"
        repair = self.repair_for(value)
        self.assertTrue(repair["unrepairable_failures"])
        with self.assertRaisesRegex(ValueError, "unrepairable"):
            state.build_state_request(self.context["root_id"], self.context["route"],
                self.context["scene_id"], self.history, self.context["dataset_cards"], repair)
        with self.assertRaises(ValueError):
            state.merge_state_repair(value, self.patch, repair, self.history)

    def test_prompt_retains_full_history_and_only_patch_output_contract(self):
        request = state.build_state_request(self.context["root_id"], self.context["route"],
            self.context["scene_id"], self.history, self.context["dataset_cards"], self.repair,
            purpose_input=self.context["purpose_input"])
        text = request["messages"][0]["content"]
        body = json.loads(text.split("\nSTATE_INPUT_JSON\n")[1])
        self.assertEqual(body["cumulative_history"], self.history)
        self.assertEqual(body["targeted_repair"]["previous_raw"], self.original)
        self.assertEqual(body["purpose_input"], self.context["purpose_input"])
        self.assertIn("Return ONLY {citation_repairs", text)
        self.assertNotIn("Return exactly {root_id,route,scene_id,facts", text)

    def test_legacy_default_still_uses_original_mixed_error_behavior(self):
        with self.assertRaises(state.StateValidationError) as caught:
            self.validate(self.original)
        repair = state.build_state_repair(caught.exception, self.original, self.history)
        self.assertNotIn("repair_mode", repair)
        self.assertIn("complete corrected snapshot", repair["instruction"])

    def test_two_rejected_leaves_in_same_citation_are_validated_together(self):
        baseline = self.merge(self.patch)
        value = deepcopy(baseline)
        value["facts"][0]["evidence"][0] = {"turn_ref": "missing", "quote": ""}
        repair = self.repair_for(value)
        original_citation = baseline["facts"][0]["evidence"][0]
        model_patch = {"citation_repairs": [
            {"path": f"facts[0].evidence[0].{leaf}", "value": original_citation[leaf]}
            for leaf in ("turn_ref", "quote")]}
        merged = state.merge_state_repair(value, model_patch, repair, self.history)
        self.assertEqual(merged, baseline)

    def test_invalid_json_cannot_request_full_regeneration(self):
        with self.assertRaises(state.StateValidationError) as caught:
            self.validate('{"facts":')
        repair = state.build_state_repair(caught.exception, '{"facts":', self.history,
            require_local_citation_patch=True)
        self.assertTrue(repair["unrepairable_failures"])
        self.assertEqual(repair["required_paths"], [])

    def test_runner_merges_model_patch_and_stops_non_citation_failure(self):
        class Caller:
            def __init__(self, run, values):
                self.run, self.values, self.calls = run, iter(values), []

            def call(self, tag, payload, proof):
                self.calls.append((tag, payload, proof))
                return next(self.values), self.run / f"request-{len(self.calls)}"

        with tempfile.TemporaryDirectory() as folder, \
                patch.object(runner, "branch_history_ok", return_value=self.history), \
                patch.object(runner, "history_proof", return_value={}):
            run = Path(folder)
            caller = Caller(run, [self.original, self.patch])
            result = runner.extract_latest_state(caller, run, self.context["root_id"],
                self.context["route"], self.history, self.context["dataset_cards"], self.context["scene_id"],
                purpose_input=self.context["purpose_input"], require_local_state_repair=True)
            self.assertEqual(result, self.merge(self.patch))
            self.assertEqual(len(caller.calls), 2)
            self.assertEqual(caller.calls[1][2]["state_repair_policy"], state.STATE_CITATION_PATCH_VERSION)
            self.assertTrue((run / "request-1/FORMAT_REPAIR_SPEC.json").exists())
            record = json.loads((run / "request-2/FORMAT_REPAIR_MERGE.json").read_text())
            self.assertEqual(record["changed_paths"], self.expected_paths)
            invalid = deepcopy(self.original); invalid["facts"][0]["status"] = "bad"
            caller = Caller(run / "unsupported", [invalid])
            result = runner.extract_latest_state(caller, run, self.context["root_id"],
                self.context["route"], self.history, self.context["dataset_cards"], self.context["scene_id"],
                require_local_state_repair=True)
            self.assertIsNone(result)
            self.assertEqual(len(caller.calls), 1)


if __name__ == "__main__":
    unittest.main()
