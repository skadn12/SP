"""Read-only real inputs and isolated deterministic selection fixtures."""
from copy import deepcopy
import json
import unittest

from .plan_normalization import merged_queue, normalize_queue
from .task_selection import VERSION, select_same_task
from .test_plan_normalization import ACTUAL, dep, entry


def inputs(originals, additions):
    registry = {}
    for item in originals:
        candidate = item["scene"]["source_candidate_id"]
        saved = deepcopy(item)
        saved["scene"].pop("phase_id")
        saved["scene"].pop("source_candidate_id")
        saved["scene"].pop("dependencies")
        registry[candidate] = saved
    return ({"candidate_registry": registry, "original_sha256": "FIXTURE-NOT-A-LIVE-PLAN"},
            {"valid": True, "merged": {"added_phases": deepcopy(additions)}})


def variant(candidate="ND09:C01", *, phase="phase:variant", **kwargs):
    value = entry(candidate, phase=phase, **kwargs)
    value["scene"]["increment_type"] = "independent_new_development"
    for layer in ("scene", "gate_scene"):
        value[layer]["before_state"] = "A different starting description."
    return value


class TaskSelectionTests(unittest.TestCase):
    def test_actual_13_entries_selects_original_without_mutating_inputs(self):
        read = lambda name: json.loads((ACTUAL / name).read_text())
        prepared = read("original_plan_new_policy_diagnostics.json")
        merged = read("increment_patch_and_diff.json")
        before = deepcopy((prepared, merged))
        result = select_same_task(prepared, merged)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual((result["before_count"], result["after_count"]), (13, 12))
        self.assertEqual(result["aliases"], [])
        self.assertEqual(result["candidate_aliases"], {})
        omitted, = result["unselected_variants"]
        self.assertEqual(omitted["candidate_id"], "ND09:C01")
        self.assertEqual(omitted["preferred_candidate_id"], "ND02:C01")
        self.assertEqual(omitted["status"], "NOT_SELECTED_SAME_TASK_VARIANT")
        self.assertFalse(omitted["is_exact_alias"])
        self.assertFalse(omitted["dependency_reference_replacement"])
        for name in ("required_participants", "before_state", "source_basis", "interaction_beats"):
            self.assertIn("/scene/" + name, {row["path"] for row in omitted["differences"]})
        self.assertEqual(result["deferred_dependencies"], [])
        self.assertEqual((prepared, merged), before)
        legacy = normalize_queue(merged_queue(prepared, merged))
        self.assertEqual(legacy["errors"], ["UNRESOLVED_EXECUTION_EQUIVALENCE"])
        self.assertEqual(legacy["outline_check"]["errors"], ["planner_outline_duplicate:S05:S13"])

    def test_three_core_fields_use_only_existing_case_and_whitespace_normalization(self):
        add = variant()
        for key in ("goal", "required_transition", "after_state"):
            add["scene"][key] = "  REVIEW   THE\nBOOK. "
        result = select_same_task(*inputs([entry("ND02:C01")], [add]))
        self.assertTrue(result["valid"])
        self.assertEqual(result["after_count"], 1)

    def test_detail_differences_do_not_require_execution_equivalence(self):
        for field, value in (("participants", ["Emily"]), ("required_participants", ["Emily"]),
                             ("before_state", "Only Emily has arrived."),
                             ("interaction_beats", ["Read aloud."]),
                             ("source_basis", ["A different historical locator."])):
            with self.subTest(field=field):
                add = variant()
                add["scene"][field] = value
                result = select_same_task(*inputs([entry("ND02:C01")], [add]))
                self.assertTrue(result["valid"])
                self.assertEqual(len(result["unselected_variants"]), 1)
                self.assertEqual(result["aliases"], [])

    def test_different_core_field_is_not_selected_as_same_task(self):
        for field in ("goal", "required_transition", "after_state"):
            with self.subTest(field=field):
                add = variant()
                add["scene"][field] = "Send the invitation after preparation."
                add["gate_scene"][field] = add["scene"][field]
                result = select_same_task(*inputs([entry("ND02:C01")], [add]))
                self.assertEqual(result["after_count"], 2)
                self.assertEqual(result["unselected_variants"], [])

    def test_original_identity_comes_from_registry_not_numeric_id(self):
        original = entry("ND99:C01")
        result = select_same_task(*inputs([original], [variant("ND01:C01")]))
        self.assertEqual(result["unselected_variants"][0]["preferred_candidate_id"], "ND99:C01")
        self.assertEqual(result["entry_provenance"]["initial:ND99:C01"]["origin"], "original_plan")

    def test_same_origin_uses_frozen_order_and_multiple_variants_are_stable(self):
        first, second = entry("ND99:C01"), variant("ND01:C01", phase="unused")
        third = variant("ND00:C01", phase="phase:third", participants=["Emily"])
        prepared, merged = inputs([first, second], [third])
        result = select_same_task(prepared, merged)
        self.assertEqual(result["after_count"], 1)
        self.assertEqual([row["preferred_candidate_id"] for row in result["unselected_variants"]],
                         ["ND99:C01", "ND99:C01"])
        self.assertEqual(select_same_task(prepared, merged), result)
        prepared, merged = inputs([], [variant("ND99:C01", phase="phase:first"),
                                      variant("ND01:C01", phase="phase:second", participants=["Emily"])])
        result = select_same_task(prepared, merged)
        self.assertEqual(result["queue"][0]["scene"]["source_candidate_id"], "ND99:C01")

    def test_exact_duplicate_remains_alias_and_rebinds_only_exact_dependency(self):
        alias = entry("ND09:C01", phase="phase:alias")
        alias["scene"]["increment_type"] = "independent_new_development"
        follow = entry("ND10:C01", phase="phase:follow", goal="Close book.",
                       dependencies=[dep("phase:alias")])
        result = select_same_task(*inputs([entry("ND02:C01")], [alias, follow]))
        self.assertTrue(result["valid"])
        self.assertEqual(result["unselected_variants"], [])
        self.assertEqual(result["candidate_aliases"], {"ND09:C01": "ND02:C01"})
        self.assertEqual(result["queue"][1]["scene"]["dependencies"], [dep("initial:ND02:C01")])

    def test_non_equivalent_dependencies_defer_transitively_without_rebinding(self):
        omitted = variant()
        follow = entry("ND10:C01", phase="phase:follow", goal="Close book.",
                       dependencies=[dep("phase:variant")])
        next_phase = entry("ND11:C01", phase="phase:later", goal="Put book away.",
                           dependencies=[dep("phase:follow")])
        unrelated = entry("ND12:C01", phase="phase:unrelated", goal="Prepare supper.")
        unrelated["scene"]["source_basis"] = ["Background mentions phase:variant and ND09:C01."]
        prepared, merged = inputs([entry("ND02:C01")], [omitted, follow, next_phase, unrelated])
        before = deepcopy(merged)
        result = select_same_task(prepared, merged)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual([row["phase_id"] for row in result["deferred_dependencies"]],
                         ["phase:follow", "phase:later"])
        self.assertEqual([row["scene"]["phase_id"] for row in result["queue"]],
                         ["initial:ND02:C01", "phase:unrelated"])
        self.assertEqual(result["phase_aliases"], {})
        self.assertEqual(merged, before)

    def test_same_core_predecessor_successor_stays_distinct(self):
        add = variant(dependencies=[dep("initial:ND02:C01")])
        result = select_same_task(*inputs([entry("ND02:C01")], [add]))
        self.assertEqual(result["after_count"], 2)
        self.assertEqual(result["unselected_variants"], [])
        self.assertEqual(result["preserved_distinctions"][0]["reasons"][0]["reason"],
                         "structured_predecessor_successor_phases")
        self.assertFalse(result["valid"])  # Existing outline rule is not relaxed.

    def test_distinct_verified_event_bindings_are_not_grouped(self):
        original, add = entry("ND02:C01"), variant()
        original["gate_scene"]["verified_track_ref"] = "VT01"
        add["gate_scene"]["verified_track_ref"] = "VT02"
        result = select_same_task(*inputs([original], [add]))
        self.assertEqual(result["after_count"], 2)
        self.assertEqual(result["unselected_variants"], [])
        self.assertTrue(result["preserved_distinctions"])

    def test_true_global_dependency_errors_still_block(self):
        for additions, expected in (
                ([variant(dependencies=[dep("phase:missing")])], "dangling_dependency"),
                ([variant(dependencies=[dep("phase:variant")])], "self_dependency"),
                ([variant(dependencies=[dep("phase:later")]),
                  variant("ND10:C01", phase="phase:later", dependencies=[dep("phase:variant")])],
                 "dependency_cycle")):
            with self.subTest(expected=expected):
                result = select_same_task(*inputs([entry("ND02:C01")], additions))
                self.assertFalse(result["valid"])
                self.assertTrue(any(error.startswith(expected) for error in result["errors"]))
                self.assertEqual(result["unselected_variants"], [])

    def test_sidecars_are_versioned_and_not_added_to_model_scene(self):
        result = select_same_task(*inputs([entry("ND02:C01")], [variant()]),
                                  source_identity={"fixture": "NOT_LIVE"})
        self.assertEqual(result["version"], VERSION)
        self.assertEqual(result["source_identity"], {"fixture": "NOT_LIVE"})
        self.assertEqual(len(result["raw_queue_sha256"]), 64)
        self.assertEqual(len(result["execution_queue_sha256"]), 64)
        for row in result["queue"]:
            self.assertNotIn("preferred_candidate_id", row["scene"])
            self.assertNotIn("NOT_SELECTED_SAME_TASK_VARIANT", row["scene"].values())
