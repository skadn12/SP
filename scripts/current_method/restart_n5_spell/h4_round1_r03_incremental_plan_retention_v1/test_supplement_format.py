"""Offline checks for structural repair without semantic replacement."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from .planning import sha
from .root_planning import merge_root_patch, prepare_root_original
from .run import offline_network_guard
from .supplement_format import VERSION, format_root_supplement
from .test_root_planning import existing_phase_patch, independent_new_patch, root_fixture_bundle


PRIOR_RUN = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
                 "h4_round1_r03_aligned_prompts_single_root_v1_live_1")


class SupplementFormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.network = offline_network_guard()
        cls.network.__enter__()
        cls.bundle = root_fixture_bundle()
        cls.prepared = prepare_root_original(cls.bundle)

    @classmethod
    def tearDownClass(cls):
        cls.network.__exit__(None, None, None)

    def format(self, raw):
        raw_before, prepared_before = deepcopy(raw), sha(self.prepared)
        result = format_root_supplement(raw, prepared=self.prepared)
        self.assertEqual(raw, raw_before)
        self.assertEqual(sha(self.prepared), prepared_before)
        self.assertEqual(result["raw_output"], raw_before)
        self.assertEqual(result["formatter_version"], VERSION)
        if not result["valid"]:
            self.assertNotIn("formatted_output", result)
        return result

    def assert_invalid(self, raw, error):
        result = self.format(raw)
        self.assertFalse(result["valid"], result)
        self.assertTrue(any(error in entry for entry in result["errors"]), result)

    def test_01_actual_failed_response_repairs_and_merges_without_changing_original(self):
        bundle = json.loads((PRIOR_RUN / "inputs/bundle.json").read_text())
        prepared = prepare_root_original(bundle)
        saved = json.loads((PRIOR_RUN / "execution/increment_patch_and_diff.json").read_text())
        raw = saved["raw_patch"]
        before = deepcopy(raw), sha(prepared)
        result = format_root_supplement(raw, prepared=prepared)
        self.assertTrue(result["valid"], result)
        self.assertEqual(raw, before[0])
        self.assertEqual(sha(prepared), before[1])
        self.assertEqual(len(result["transformations"]), 7)
        self.assertEqual(sum(row["operation"] == "move_candidate_field"
                             for row in result["transformations"]), 5)
        merged = merge_root_patch(prepared, result["formatted_output"])
        self.assertTrue(merged["valid"], merged)
        self.assertEqual(merged["merged"]["original_plan"], prepared["original"])
        self.assertEqual(merged["diff"]["original_candidate_changes"], [])
        self.assertEqual(merged["diff"]["original_goal_changes"], [])
        self.assertEqual(len(merged["merged"]["added_phases"]), 2)
        repeated = format_root_supplement(result["formatted_output"], prepared=prepared)
        self.assertTrue(repeated["valid"], repeated)
        self.assertEqual(repeated["formatted_output"], result["formatted_output"])
        self.assertEqual(repeated["transformations"], [])

    def test_02_valid_legacy_output_is_unchanged(self):
        raw = independent_new_patch(self.bundle)
        raw["phases"] = existing_phase_patch(self.bundle)["phases"]
        result = self.format(raw)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["formatted_output"], raw)
        self.assertEqual(result["transformations"], [])
        self.assertTrue(merge_root_patch(self.prepared, result["formatted_output"])["valid"])

    def test_numeric_overflow_in_json_text_is_rejected(self):
        raw = independent_new_patch(self.bundle)
        raw["new_developments"][0]["candidate"]["source_basis"] = ["OVERFLOW_SENTINEL"]
        for number in ("1e999", "-1e999"):
            with self.subTest(number=number):
                text = json.dumps(raw).replace('"OVERFLOW_SENTINEL"', number)
                self.assert_invalid(text, "supplement_format_parse:")

    def test_03_fully_flat_candidate_gets_wrapper(self):
        raw = independent_new_patch(self.bundle)
        row = raw["new_developments"][0]
        candidate = row.pop("candidate")
        row.update(candidate)
        result = self.format(raw)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["formatted_output"]["new_developments"][0]["candidate"], candidate)
        self.assertEqual(result["transformations"][0]["operation"], "create_candidate_wrapper")
        self.assertTrue(merge_root_patch(self.prepared, result["formatted_output"])["valid"])

    def test_04_identical_repeated_field_collapses_but_conflict_rejects(self):
        raw = independent_new_patch(self.bundle)
        row = raw["new_developments"][0]
        row["interaction_beats"] = deepcopy(row["candidate"]["interaction_beats"])
        result = self.format(raw)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["transformations"][0]["operation"], "collapse_identical_candidate_field")
        row["interaction_beats"][0] = "A contradictory action must not be discarded."
        self.assert_invalid(raw, "conflicting_candidate_field")

    def test_05_unknown_or_missing_fields_are_not_dropped_or_filled(self):
        for location in ("top", "phase", "wrapper", "candidate", "dependency"):
            with self.subTest(location=location):
                raw = independent_new_patch(self.bundle)
                raw["phases"] = existing_phase_patch(self.bundle)["phases"]
                target = {"top": raw, "phase": raw["phases"][0],
                          "wrapper": raw["new_developments"][0],
                          "candidate": raw["new_developments"][0]["candidate"],
                          "dependency": raw["phases"][0]["dependencies"][0]}[location]
                target["unexpected"] = "Must not silently disappear."
                self.assertFalse(self.format(raw)["valid"])
        for field in ("goal", "actor_bindings", "required_transition"):
            raw = independent_new_patch(self.bundle)
            del raw["new_developments"][0]["candidate"][field]
            self.assert_invalid(raw, "candidate_fields")

    def test_06_duplicate_phase_ids_are_rejected_across_both_arrays(self):
        for identifier in ("phase:duplicate", "phase:duplicate:invalid"):
            raw = independent_new_patch(self.bundle)
            raw["phases"] = existing_phase_patch(self.bundle, ("ME03:C01",))["phases"]
            raw["phases"][0]["phase_id"] = identifier
            raw["new_developments"][0]["phase_id"] = identifier
            self.assert_invalid(raw, "duplicate_phase_id")

    def test_07_canonicalization_reserves_later_valid_ids(self):
        raw = independent_new_patch(self.bundle, count=2)
        rows = raw["new_developments"]
        rows[0]["phase_id"] = "phase:ND09:C01"
        rows[1]["phase_id"] = "phase:formatted_1"
        rows[1]["dependencies"] = [{"phase_id": rows[0]["phase_id"], "condition": "The first label exists."}]
        result = self.format(raw)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["phase_id_map"]["phase:ND09:C01"], "phase:formatted_1_2")
        self.assertEqual(result["formatted_output"]["new_developments"][1]["phase_id"], "phase:formatted_1")
        self.assertEqual(result["formatted_output"]["new_developments"][1]["dependencies"][0]["phase_id"],
                         "phase:formatted_1_2")
        self.assertTrue(merge_root_patch(self.prepared, result["formatted_output"])["valid"])

    def test_08_all_phase_dependencies_are_rewritten_across_arrays(self):
        raw = independent_new_patch(self.bundle, count=2)
        raw["phases"] = existing_phase_patch(self.bundle, ("ME03:C01", "ME07:C01"))["phases"]
        rows = raw["phases"] + raw["new_developments"]
        for index, row in enumerate(rows):
            row["phase_id"] = "phase:needs:format_" + str(index)
            if index:
                row["dependencies"] = [{"phase_id": earlier["phase_id"],
                                        "condition": "The specific earlier prerequisite is retained."}
                                       for earlier in rows[:index]]
        result = self.format(raw)
        self.assertTrue(result["valid"], result)
        expected = deepcopy(raw)
        for row in expected["phases"] + expected["new_developments"]:
            row["phase_id"] = result["phase_id_map"][row["phase_id"]]
            for dependency in row["dependencies"]:
                dependency["phase_id"] = result["phase_id_map"].get(dependency["phase_id"], dependency["phase_id"])
        self.assertEqual(result["formatted_output"], expected)
        self.assertTrue(merge_root_patch(self.prepared, result["formatted_output"])["valid"])

    def test_09_forward_self_and_unknown_references_fail_without_guessing(self):
        for target, error in (("phase:independent_2", "forward_dependency"),
                              ("phase:independent_1", "self_dependency"),
                              ("phase:missing", "unknown_dependency"),
                              ("initial:ND99:C01", "unknown_dependency"),
                              ("ND01:C01", "unknown_dependency")):
            raw = independent_new_patch(self.bundle, count=2)
            raw["new_developments"][0]["dependencies"] = [{"phase_id": target, "condition": "A prerequisite."}]
            self.assert_invalid(raw, error)

    def test_10_initial_phase_ids_and_missing_ids_are_never_repaired(self):
        for value in ("initial:ME03:C01", " initial:ME03:C01", "", " ", None, 1):
            raw = independent_new_patch(self.bundle)
            raw["new_developments"][0]["phase_id"] = value
            self.assertFalse(self.format(raw)["valid"], value)
        raw = independent_new_patch(self.bundle)
        del raw["new_developments"][0]["phase_id"]
        self.assert_invalid(raw, "phase_id_required")

    def test_11_source_hash_and_original_goal_are_immutable(self):
        for field, value, error in (("source_candidate_id", "ND99:C01", "unknown_source"),
                                    ("original_goal", "Replace the original objective.", "original_goal_changed")):
            raw = existing_phase_patch(self.bundle)
            raw["phases"][0][field] = value
            self.assert_invalid(raw, error)
        raw = independent_new_patch(self.bundle)
        raw["original_plan_sha256"] = "0" * 64
        self.assert_invalid(raw, "patch_original_sha_mismatch")

    def test_12_dependencies_and_conditions_are_not_synthesized(self):
        for value in (None, "", [], [{"phase_id": "initial:ME03:C01", "condition": " "}],
                      [{"phase_id": "initial:ME03:C01"}]):
            raw = existing_phase_patch(self.bundle)
            raw["phases"][0]["dependencies"] = value
            self.assertFalse(self.format(raw)["valid"], value)
        raw = independent_new_patch(self.bundle)
        del raw["new_developments"][0]["dependencies"]
        self.assert_invalid(raw, "dependencies_required")

    def test_13_duplicate_json_keys_and_non_json_numbers_fail(self):
        raw = independent_new_patch(self.bundle)
        serialized = json.dumps(raw)
        valid = self.format(serialized)
        self.assertTrue(valid["valid"], valid)
        self.assertEqual(valid["formatted_output"], raw)
        self.assert_invalid(serialized.replace('"status": "PATCH"', '"status": "PATCH", "status": "PATCH"'),
                            "duplicate_json_key")
        self.assert_invalid(serialized.replace('"dependencies": []', '"dependencies": [], "dependencies": []'),
                            "duplicate_json_key")
        self.assert_invalid(serialized.replace('"dependencies": []', '"dependencies": NaN'),
                            "invalid_json_constant")

    def test_14_formatting_does_not_override_actor_contract_checks(self):
        raw = independent_new_patch(self.bundle)
        row = raw["new_developments"][0]
        row["actor_bindings"] = row["candidate"].pop("actor_bindings")
        row["actor_bindings"][0]["actor_ref"] = "SP999"
        result = self.format(raw)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["formatted_output"]["new_developments"][0]["candidate"]["actor_bindings"],
                         row["actor_bindings"])
        self.assertFalse(merge_root_patch(self.prepared, result["formatted_output"])["valid"])

    def test_15_empty_no_supplement_and_bad_status(self):
        raw = independent_new_patch(self.bundle, count=0)
        raw["status"] = "NO_SUPPLEMENT"
        result = self.format(raw)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["formatted_output"], raw)
        for status in ("PATCH", "OTHER", None, []):
            raw["status"] = status
            self.assertFalse(self.format(raw)["valid"], status)

    def test_16_generated_id_cannot_accidentally_resolve_an_unknown_dependency(self):
        raw = independent_new_patch(self.bundle, count=2)
        raw["new_developments"][0]["phase_id"] = "phase:ND09:C01"
        raw["new_developments"][1]["dependencies"] = [
            {"phase_id": "phase:formatted_1", "condition": "This name was never supplied by the planner."}]
        self.assert_invalid(raw, "unknown_dependency")


if __name__ == "__main__":
    unittest.main()
