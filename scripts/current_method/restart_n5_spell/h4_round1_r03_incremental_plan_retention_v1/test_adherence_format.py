"""Offline recovery checks using saved failures and adverse reference cases."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from .adherence_format import PROVENANCE_KEY, VERSION, format_adherence, validate_formatting_provenance
from .retention import assess_retention
from .run import offline_network_guard


SOURCE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
              "h4_round1_r03_supplement_format_resume_v1_live_1/execution")


def saved_case(batch, call):
    paths = {"normalization": SOURCE / f"checks/batch_{batch}_normalization.json",
             "response": SOURCE / f"responses/{call:03d}_adherence.json",
             "request": SOURCE / f"requests/{call:03d}_adherence.json",
             "bundle": SOURCE / "input_freeze.json"}
    original_bytes = {key: path.read_bytes() for key, path in paths.items()}
    raw = json.loads(original_bytes["response"])
    request = json.loads(original_bytes["request"])
    inputs = json.loads(request["messages"][0]["content"].split("\nINPUT DATA\n", 1)[1])
    child = json.loads(original_bytes["normalization"])["child"]
    for scene, identity in zip(child["scenes"], inputs["actual_generated_dialogue_identity"]["scenes"]):
        scene["session_ref"] = identity["session_ref"]
    parent = json.loads(original_bytes["bundle"])["parent"]
    parent["dialogue"]["turns"] = inputs["authoritative_history"]
    return raw, child, inputs["downstream_conditions"], parent, inputs["planned_targets"], paths, original_bytes


class AdherenceFormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.network = offline_network_guard()
        cls.network.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.network.__exit__(None, None, None)

    def setUp(self):
        self.child = {"scenes": [{"scene_ref": "R03:S01", "dialogue_turns": [
            {"turn_ref": "R03:S01:T001", "speaker": "A", "text": "The first exact sentence."},
            {"turn_ref": "R03:S01:T002", "speaker": "B", "text": "The second exact sentence."},
        ]}]}
        self.raw = {"scene_audits": [{
            "scene_ref": "R03:S01", "goal_status": "partial", "content_decision": "retain",
            "participant_adherence": "pass", "history_consistency": "pass", "novel_progress": "pass",
            "actual_states": [{"statement": "The first exact sentence.", "stage": "arrangement",
                               "turn_refs": ["R03:S01:T001"]}],
            "goal_evidence_refs": ["R03:S01:T001"], "condition_results": [], "reason": "Original judgment.",
        }]}
        self.conditions = {"R03:S01": []}

    def format(self, raw=None):
        raw = self.raw if raw is None else raw
        before = deepcopy((raw, self.child, self.conditions))
        result = format_adherence(raw, child=self.child, requested_conditions=self.conditions)
        self.assertEqual((raw, self.child, self.conditions), before)
        self.assertEqual(result["raw_output"], before[0])
        self.assertEqual(result["formatter_version"], VERSION)
        if not result["valid"]:
            self.assertNotIn("formatted_output", result)
        return result

    def assert_invalid(self, error, raw=None):
        result = self.format(raw)
        self.assertFalse(result["valid"], result)
        self.assertTrue(any(error in value for value in result["errors"]), result)

    def test_saved_failures_format_without_changing_judgments_or_source_files(self):
        eligible = []
        needs_reaudit = []
        for batch, call in ((2, 5), (4, 9), (6, 13), (12, 16)):
            with self.subTest(batch=batch):
                raw, child, conditions, parent, plan, paths, original_bytes = saved_case(batch, call)
                original = json.loads(raw)
                before = deepcopy((child, conditions))
                result = format_adherence(raw, child=child, requested_conditions=conditions)
                self.assertTrue(result["valid"], result)
                self.assertEqual(result["raw_output"], raw)
                self.assertEqual((child, conditions), before)
                formatted = result["formatted_output"]
                for prior, current in zip(original["scene_audits"], formatted["scene_audits"]):
                    for key in set(prior) - {"actual_states", "goal_evidence_refs", "condition_results"}:
                        self.assertEqual(current[key], prior[key])
                    for old_state, new_state in zip(prior["actual_states"], current["actual_states"]):
                        self.assertEqual(new_state["statement"], old_state["statement"])
                        self.assertEqual(new_state["stage"], old_state["stage"])
                assessment = assess_retention(parent=parent, plan=plan, child=child,
                    raw_audit=formatted, requested_conditions=conditions)
                for row in assessment["decisions"]:
                    if row["scene_ref"] in result["semantic_reaudit_required_scene_refs"]:
                        self.assertFalse(row["retained"])
                        self.assertEqual(row["content_status"], "NEEDS_SEMANTIC_REAUDIT")
                        self.assertIn("semantic_reaudit_required", row["errors"])
                        needs_reaudit.append(row["scene_ref"])
                        if row["scene_ref"] == "R03:S04":
                            self.assertIn("novel_progress_not_pass", row["errors"])
                            self.assertIn("content_not_retain", row["errors"])
                    else:
                        self.assertEqual(row["errors"], [])
                        eligible.append(row["scene_ref"])
                removed = [change["original"] for change in result["transformations"]
                           if change["operation"] == "exclude_unrequested_condition"]
                self.assertEqual(removed, [condition for row in original["scene_audits"]
                                          for condition in row["condition_results"]])
                reloaded = json.loads(json.dumps(formatted))
                repeated = format_adherence(reloaded, child=child, requested_conditions=conditions)
                self.assertTrue(repeated["valid"], repeated)
                self.assertEqual(repeated["formatted_output"], formatted)
                self.assertEqual(repeated["transformations"], [])
                self.assertEqual(repeated["semantic_reaudit_required_scene_refs"],
                                 result["semantic_reaudit_required_scene_refs"])
                repeated_assessment = assess_retention(parent=parent, plan=plan, child=child,
                    raw_audit=repeated["formatted_output"], requested_conditions=conditions)
                self.assertEqual(repeated_assessment, assessment)
                self.assertEqual({key: path.read_bytes() for key, path in paths.items()}, original_bytes)
        self.assertEqual(eligible, ["R03:S01", "R03:S02"])
        self.assertEqual(needs_reaudit, ["R03:S03", "R03:S04", "R03:S05", "R03:S06", "R03:S07"])

    def test_valid_audit_is_unchanged(self):
        result = self.format()
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["formatted_output"]["scene_audits"], self.raw["scene_audits"])
        self.assertEqual(result["formatted_output"][PROVENANCE_KEY], {
            "formatter_version": VERSION, "semantic_reaudit_required_scene_refs": [], "transformations": []})
        self.assertFalse(result["semantic_reaudit_required"])
        self.assertEqual(result["transformations"], [])

    def test_goal_wrapper_and_duplicate_refs_preserve_full_original(self):
        row = self.raw["scene_audits"][0]
        row["goal_evidence_refs"] = [{"statement": "The first exact sentence.", "turn_refs": ["old:one"]},
                                     "old:one"]
        result = self.format()
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["formatted_output"]["scene_audits"][0]["goal_evidence_refs"], ["R03:S01:T001"])
        wrapper = next(change for change in result["transformations"] if change["operation"] == "flatten_goal_evidence")
        self.assertEqual(wrapper["original"], row["goal_evidence_refs"][0])

    def test_ref_text_map_cannot_be_inferred_from_a_suffix_or_position(self):
        self.raw["scene_audits"][0]["goal_evidence_refs"] = ["r02:s01:T002"]
        self.assert_invalid("unresolved_reference")

    def test_wrong_reference_requires_exact_complete_text(self):
        state = self.raw["scene_audits"][0]["actual_states"][0]
        state["turn_refs"] = ["old:one"]
        for text in ("first exact sentence", " The first exact sentence.", "A summary of the first sentence."):
            with self.subTest(text=text):
                state["statement"] = text
                self.assert_invalid("reference_requires_exact_unique_quote")

    def test_repeated_text_cannot_establish_alias(self):
        self.child["scenes"][0]["dialogue_turns"][1]["text"] = "The first exact sentence."
        self.raw["scene_audits"][0]["actual_states"][0]["turn_refs"] = ["old:one"]
        self.assert_invalid("reference_requires_exact_unique_quote")

    def test_multiple_supplied_refs_cannot_be_assigned_to_one_quote(self):
        self.raw["scene_audits"][0]["actual_states"][0]["turn_refs"] = ["old:one", "old:two"]
        self.assert_invalid("reference_requires_exact_unique_quote")

    def test_conflicting_aliases_are_rejected(self):
        row = self.raw["scene_audits"][0]
        row["actual_states"][0]["turn_refs"] = ["old:one"]
        row["goal_evidence_refs"] = [{"statement": "The second exact sentence.", "turn_refs": ["old:one"]}]
        self.assert_invalid("contradictory_reference_mapping")

    def test_valid_ref_with_contradictory_exact_quote_is_not_replaced(self):
        self.raw["scene_audits"][0]["actual_states"][0]["turn_refs"] = ["R03:S01:T002"]
        self.assert_invalid("contradictory_ref_text")

    def test_valid_ref_for_summary_does_not_require_verbatim_quote(self):
        self.raw["scene_audits"][0]["actual_states"][0]["statement"] = "An unchanged auditor summary."
        result = self.format()
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["formatted_output"]["scene_audits"], self.raw["scene_audits"])

    def test_aliases_never_leak_to_a_different_scene(self):
        second = deepcopy(self.child["scenes"][0])
        second["scene_ref"] = "R03:S02"
        for turn in second["dialogue_turns"]:
            turn["turn_ref"] = turn["turn_ref"].replace("S01", "S02")
        self.child["scenes"].append(second)
        self.conditions["R03:S02"] = []
        row = deepcopy(self.raw["scene_audits"][0])
        row["scene_ref"] = "R03:S02"
        row["actual_states"][0]["turn_refs"] = ["R03:S02:T001"]
        row["goal_evidence_refs"] = ["old:one"]
        self.raw["scene_audits"].append(row)
        self.raw["scene_audits"][0]["actual_states"][0]["turn_refs"] = ["old:one"]
        self.assert_invalid("unresolved_reference")
        self.raw["scene_audits"][0]["actual_states"][0]["turn_refs"] = ["R03:S02:T001"]
        self.assert_invalid("cross_scene_reference")

    def test_missing_requested_condition_is_not_synthesized(self):
        self.conditions["R03:S01"] = ["The exact requested condition."]
        self.assert_invalid("requested_condition_coverage")

    def test_missing_or_extra_scene_condition_map_is_rejected(self):
        self.conditions = {}
        self.assert_invalid("requested_conditions_scene_coverage")
        self.conditions = {"R03:S01": [], "R03:S02": []}
        self.assert_invalid("requested_conditions_scene_coverage")

    def test_requested_conditions_preserve_status_and_use_evidence_aliases(self):
        row = self.raw["scene_audits"][0]
        row["actual_states"][0]["turn_refs"] = ["old:one"]
        self.conditions["R03:S01"] = ["The exact requested condition."]
        row["condition_results"] = [{"condition": self.conditions["R03:S01"][0],
                                     "status": "established", "turn_refs": ["old:one"]}]
        result = self.format()
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["formatted_output"]["scene_audits"][0]["condition_results"],
                         [{"condition": self.conditions["R03:S01"][0], "status": "established",
                           "turn_refs": ["R03:S01:T001"]}])
        row["condition_results"].append(deepcopy(row["condition_results"][0]))
        self.assert_invalid("requested_condition_coverage")

    def test_extra_condition_payload_is_preserved_without_a_semantic_rewrite(self):
        row = self.raw["scene_audits"][0]
        row.update(content_decision="reject", novel_progress="fail")
        extra = {"condition": "Not requested.", "status": "not_established", "turn_refs": [],
                 "model_note": "An extra value must remain in the log."}
        row["condition_results"] = [extra]
        result = self.format()
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["transformations"][0]["original"], extra)
        output = result["formatted_output"]["scene_audits"][0]
        self.assertEqual(output["content_decision"], "reject")
        self.assertEqual(output["novel_progress"], "fail")
        self.assertTrue(result["semantic_reaudit_required"])
        self.assertEqual(result["semantic_reaudit_required_scene_refs"], ["R03:S01"])
        self.assertEqual(result["formatted_output"][PROVENANCE_KEY]["transformations"], result["transformations"])

    def test_missing_or_malformed_provenance_fields_are_rejected(self):
        self.raw["scene_audits"][0]["condition_results"] = [
            {"condition": "Not requested.", "status": "established", "turn_refs": ["R03:S01:T001"]}]
        formatted = self.format()["formatted_output"]
        for field in ("formatter_version", "semantic_reaudit_required_scene_refs", "transformations"):
            with self.subTest(missing=field):
                bad = deepcopy(formatted)
                del bad[PROVENANCE_KEY][field]
                self.assert_invalid("formatting_provenance_shape", bad)
        for value in (None, [], "not metadata"):
            with self.subTest(provenance=value):
                bad = deepcopy(formatted)
                bad[PROVENANCE_KEY] = value
                self.assert_invalid("formatting_provenance_shape", bad)
        for refs in (None, "R03:S01", ["R03:S02"], ["R03:S01", "R03:S01"], [[]]):
            with self.subTest(refs=refs):
                bad = deepcopy(formatted)
                bad[PROVENANCE_KEY]["semantic_reaudit_required_scene_refs"] = refs
                self.assert_invalid("formatting_provenance_scene_refs", bad)

    def test_required_reaudit_cannot_be_cleared_without_preserved_exclusion_evidence(self):
        self.raw["scene_audits"][0]["condition_results"] = [
            {"condition": "Not requested.", "status": "established", "turn_refs": ["R03:S01:T001"]}]
        formatted = self.format()["formatted_output"]
        for field in ("semantic_reaudit_required_scene_refs", "transformations"):
            with self.subTest(cleared=field):
                bad = deepcopy(formatted)
                bad[PROVENANCE_KEY][field] = []
                self.assert_invalid("formatting_provenance_semantic_coverage", bad)

    def test_exclusion_metadata_cannot_point_to_a_different_audit_row(self):
        self.raw["scene_audits"][0]["condition_results"] = [
            {"condition": "Not requested.", "status": "established", "turn_refs": ["R03:S01:T001"]}]
        formatted = self.format()["formatted_output"]
        for path in ("/scene_audits/1/condition_results/0", "/scene_audits/0/reason/0", "not a path"):
            with self.subTest(path=path):
                bad = deepcopy(formatted)
                bad[PROVENANCE_KEY]["transformations"][0]["path"] = path
                self.assert_invalid("formatting_provenance_exclusion_identity", bad)

    def test_failed_conversion_still_reports_inherited_semantic_review_requirement(self):
        self.raw["scene_audits"][0]["condition_results"] = [
            {"condition": "Not requested.", "status": "established", "turn_refs": ["R03:S01:T001"]}]
        formatted = self.format()["formatted_output"]
        formatted["scene_audits"][0]["goal_evidence_refs"] = ["nonexistent"]
        result = self.format(formatted)
        self.assertFalse(result["valid"])
        self.assertTrue(result["semantic_reaudit_required"])
        self.assertEqual(result["semantic_reaudit_required_scene_refs"], ["R03:S01"])

    def test_assessor_blocks_invalid_provenance_after_serialization(self):
        raw, child, conditions, parent, plan, _, _ = saved_case(2, 5)
        valid = format_adherence(raw, child=child, requested_conditions=conditions)["formatted_output"]
        for metadata in ({}, None, {"semantic_reaudit_required_scene_refs": []}):
            with self.subTest(metadata=metadata):
                bad = deepcopy(valid)
                bad[PROVENANCE_KEY] = metadata
                assessment = assess_retention(parent=parent, plan=plan, child=child,
                    raw_audit=json.loads(json.dumps(bad)), requested_conditions=conditions)
                self.assertTrue(all(not row["retained"] for row in assessment["decisions"]))

    def test_validator_accepts_raw_audit_without_formatting_metadata(self):
        self.assertEqual(validate_formatting_provenance(self.raw, {"R03:S01"}), [])

    def test_duplicate_keys_and_nonfinite_values_fail_with_raw_preserved(self):
        for raw in ('{"scene_audits":[],"scene_audits":[]}',
                    '{"scene_audits":[{"turn_refs":[],"turn_refs":[]}]}',
                    '{"scene_audits":NaN}', '{"scene_audits":Infinity}',
                    '{"scene_audits":1e999}', {"scene_audits": float("inf")}):
            with self.subTest(raw=raw):
                self.assert_invalid("adherence_format_parse", raw)

    def test_unknown_fields_and_missing_structure_are_not_silently_fixed(self):
        for field in ("extra", "actual_states", "goal_evidence_refs"):
            raw = deepcopy(self.raw)
            if field == "extra":
                raw["scene_audits"][0][field] = "Must not be silently discarded."
            else:
                del raw["scene_audits"][0][field]
            self.assert_invalid("audit_dimensions_missing_or_misaligned", raw)


if __name__ == "__main__":
    unittest.main()
