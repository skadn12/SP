"""Local reference repair must not rewrite substantive state claims."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

import latest_state as state


class LocalStateRepairTests(unittest.TestCase):
    def setUp(self):
        parent = Path(__file__).resolve().parent.parent / "h4_sequential_ab_v26/runs/inherited_a2_b3_01"
        request = next((parent / "requests").glob("005-*/REQUEST.json"))
        self.context = json.JSONDecoder().raw_decode(json.loads(request.read_text())["messages"][0]["content"]
            .split("\nSTATE_INPUT_JSON\n")[1])[0]
        self.history = self.context["cumulative_history"]
        self.original = json.loads(json.loads((request.parent / "RESPONSE.json").read_text())["text"])
        with self.assertRaises(state.StateValidationError) as caught:
            self.validate(self.original)
        self.repair = state.build_state_repair(caught.exception, self.original, self.history)
        self.patch = {"reference_repairs": [
            {"path": row["path"], "turn_ref": row["candidate_turn_refs"][0]}
            for row in self.repair["citation_reference_candidates"]]}

    def validate(self, candidate):
        return state.validate_state(candidate, self.context["root_id"], self.context["route"],
                                    self.context["scene_id"], self.history)

    def merge(self, patch):
        return state.merge_state_repair(self.original, patch, self.repair, self.history)

    def test_real_recorded_failure_can_be_fixed_only_at_rejected_references(self):
        before = deepcopy(self.original)
        merged = self.merge(self.patch)
        self.validate(merged)
        allowed = {row["path"] for row in self.patch["reference_repairs"]}
        for i, fact in enumerate(before["facts"]):
            expected = deepcopy(fact)
            for j, evidence in enumerate(expected["evidence"]):
                path = f"facts[{i}].evidence[{j}].turn_ref"
                if path in allowed:
                    evidence["turn_ref"] = next(row["turn_ref"] for row in self.patch["reference_repairs"]
                                                if row["path"] == path)
            self.assertEqual(merged["facts"][i], expected)
        self.assertEqual(self.original, before)
        self.assertEqual(merged["conflicts"], before["conflicts"])
        self.assertEqual(merged["unknowns"], before["unknowns"])

    def test_unknown_missing_duplicate_or_unaffected_reference_patch_rejected(self):
        variants = []
        missing = deepcopy(self.patch); missing["reference_repairs"].pop(); variants.append(missing)
        duplicate = deepcopy(self.patch); duplicate["reference_repairs"].append(duplicate["reference_repairs"][0]); variants.append(duplicate)
        unknown = deepcopy(self.patch); unknown["reference_repairs"][0]["turn_ref"] = "nonexistent"; variants.append(unknown)
        wrong_quote = deepcopy(self.patch); wrong_quote["reference_repairs"][0]["turn_ref"] = self.history[-1]["turn_ref"]; variants.append(wrong_quote)
        extra = deepcopy(self.patch); extra["reference_repairs"].append({"path": "facts[0].evidence[0].turn_ref", "turn_ref": self.history[0]["turn_ref"]}); variants.append(extra)
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError): self.merge(value)

    def test_full_response_or_semantic_edits_cannot_be_disguised_as_patch(self):
        for value in [self.original, {**self.patch, "facts": []},
                      {"reference_repairs": [{**self.patch["reference_repairs"][0], "statement": "changed"}]}]:
            with self.assertRaises(ValueError): self.merge(value)

    def test_patch_request_keeps_full_history_and_has_one_output_contract(self):
        request = state.build_state_request(self.context["root_id"], self.context["route"],
            self.context["scene_id"], self.history, self.context["dataset_cards"], self.repair,
            purpose_input=self.context["purpose_input"])
        text = request["messages"][0]["content"]
        context = json.loads(text.split("\nSTATE_INPUT_JSON\n")[1])
        self.assertEqual(context["cumulative_history"], self.history)
        self.assertEqual(context["purpose_input"], self.context["purpose_input"])
        self.assertIn("Return ONLY {reference_repairs", text)
        self.assertNotIn("Return exactly {root_id,route,scene_id,facts", text)


if __name__ == "__main__": unittest.main()
