"""Evidence scope and semantic re-audit boundaries, without remote calls."""
from copy import deepcopy
import json
import unittest

from .retention import (
    ADHERENCE_VERSION, INLINE_ADHERENCE_VERSION, INLINE_ADHERENCE_TEMPLATE,
    assess_retention, render_adherence,
)


class EvidenceScopeTests(unittest.TestCase):
    def setUp(self):
        self.current = "R03:S01:T001"
        self.historical = "root:P01:T001"
        self.parent = {"root_id": "root", "dialogue": {"turns": [
            {"turn_id": self.historical, "speaker": "A", "text": "An earlier appointment was established."}]}}
        self.child = {"root_id": "root", "round_index": 3, "scenes": [{
            "scene_ref": "R03:S01", "memory_event_id": "ME01", "dialogue_turns": [
                {"turn_ref": self.current, "speaker": "A", "text": "A new route is arranged today."}]}]}
        self.plan = {"scenes": [{"scene_ref": "R03:S01", "memory_event_id": "ME01"}]}
        self.conditions = {"R03:S01": []}
        self.row = {"scene_ref": "R03:S01", "goal_status": "partial", "content_decision": "retain",
            "participant_adherence": "pass", "history_consistency": "pass", "novel_progress": "pass",
            "actual_states": [{"statement": "A new route is arranged today.", "stage": "arrangement",
                               "turn_refs": [self.current]}],
            "goal_evidence_refs": [self.current], "history_evidence_refs": [self.historical],
            "condition_results": [], "reason": "The route is new relative to the old appointment."}
        self.audit = {"scene_audits": [self.row]}

    def assess(self, *, version=ADHERENCE_VERSION):
        before = deepcopy((self.parent, self.plan, self.child, self.audit, self.conditions))
        result = assess_retention(parent=self.parent, plan=self.plan, child=self.child,
            raw_audit=self.audit, requested_conditions=self.conditions, audit_version=version)
        self.assertEqual((self.parent, self.plan, self.child, self.audit, self.conditions), before)
        self.assertEqual(result["raw_audit"], self.audit)
        return result["decisions"][0]

    def test_history_comparison_and_mixed_goal_support_are_allowed(self):
        self.row["goal_evidence_refs"].append(self.historical)
        result = self.assess()
        self.assertEqual(result["errors"], [])
        self.assertTrue(result["retained"])

    def test_history_only_goal_cannot_claim_partial_or_realized_new_progress(self):
        self.row["goal_evidence_refs"] = [self.historical]
        for status in ("partial", "realized"):
            self.row["goal_status"] = status
            self.assertIn("goal_evidence", self.assess()["errors"])
        self.row["goal_status"] = "not_realized"
        self.assertTrue(self.assess()["retained"])

    def test_historical_facts_cannot_be_new_actual_states(self):
        self.row["actual_states"][0]["turn_refs"] = [self.historical]
        result = self.assess()
        self.assertIn("actual_state_shape_or_refs", result["errors"])
        self.assertFalse(result["retained"])

    def test_history_field_rejects_generated_or_fabricated_ids(self):
        for refs in ([self.current], ["invented:P01:T001"], [self.historical, self.historical], [{}]):
            with self.subTest(refs=refs):
                self.row["history_evidence_refs"] = refs
                self.assertIn("history_evidence", self.assess()["errors"])

    def test_current_batch_sibling_is_not_authoritative_history(self):
        sibling = deepcopy(self.child["scenes"][0])
        sibling["scene_ref"] = "R03:S02"
        sibling["dialogue_turns"][0].update(turn_ref="R03:S02:T001", text="A sibling scene offers another observation.")
        self.child["scenes"].append(sibling)
        self.plan["scenes"].append({"scene_ref": "R03:S02", "memory_event_id": "ME01"})
        second = deepcopy(self.row)
        second["scene_ref"] = "R03:S02"
        second["actual_states"][0]["turn_refs"] = ["R03:S02:T001"]
        second["goal_evidence_refs"] = ["R03:S02:T001"]
        self.audit["scene_audits"].append(second)
        self.conditions["R03:S02"] = []
        self.row["goal_evidence_refs"].append("R03:S02:T001")
        self.assertIn("goal_evidence", self.assess()["errors"])

    def test_explicit_prerequisite_can_use_already_established_history(self):
        condition = "The earlier appointment is established."
        self.conditions["R03:S01"] = [condition]
        self.row["condition_results"] = [{"condition": condition, "status": "established",
                                          "turn_refs": [self.historical]}]
        result = self.assess()
        self.assertTrue(result["retained"])
        self.assertEqual(result["established_conditions"], self.row["condition_results"])
        self.assertEqual(result["actual_states"][0]["turn_refs"], [self.current])
        self.row["condition_results"][0]["turn_refs"] = ["planned:ME01"]
        self.assertIn("condition_evidence", self.assess()["errors"])

    def test_v4_requires_history_field_while_saved_v3_reads_remain_valid(self):
        del self.row["history_evidence_refs"]
        self.assertIn("audit_dimensions_missing_or_misaligned", self.assess()["errors"])
        self.assertTrue(self.assess(version=INLINE_ADHERENCE_VERSION)["retained"])
        self.assertTrue(self.assess(version=None)["retained"])

    def test_invented_condition_requires_semantic_reaudit_without_formatter(self):
        self.row["condition_results"] = [{"condition": "An invented success criterion.",
                                          "status": "established", "turn_refs": [self.current]}]
        result = self.assess()
        self.assertEqual(result["content_status"], "NEEDS_SEMANTIC_REAUDIT")
        self.assertIn("semantic_reaudit_required", result["errors"])
        self.assertFalse(result["retained"])
        self.assertEqual(result["established_conditions"], [])

    def test_invalid_enums_and_extra_top_level_fields_are_contract_errors(self):
        self.audit["extra"] = "unsupported"
        self.assertIn("audit_top_level_shape", self.assess()["errors"])
        del self.audit["extra"]
        self.row["content_decision"] = "invalid"
        self.assertIn("content_decision_invalid", self.assess()["errors"])
        self.row["actual_states"] = {}
        self.assertIn("actual_state_shape_or_refs", self.assess()["errors"])

    def test_cleaned_conditions_cannot_remove_semantic_reaudit_guard(self):
        original = {"condition": "An invented success criterion.", "status": "established",
                    "turn_refs": [self.current]}
        self.audit["_formatting_provenance"] = {
            "formatter_version": "adherence_saved_formatting_v2",
            "semantic_reaudit_required_scene_refs": ["R03:S01"],
            "transformations": [{"operation": "exclude_unrequested_condition",
                                 "path": "/scene_audits/0/condition_results/0",
                                 "scene_ref": "R03:S01", "original": original}]}
        self.audit = json.loads(json.dumps(self.audit))
        result = self.assess()
        self.assertEqual(result["content_status"], "NEEDS_SEMANTIC_REAUDIT")
        self.assertEqual(result["errors"], ["semantic_reaudit_required"])
        self.assertFalse(result["retained"])
        self.assertEqual(result["established_conditions"], [])

    def test_malformed_or_contradictory_provenance_fails_closed(self):
        for provenance in ({}, {"formatter_version": "unknown"}, {
                "formatter_version": "adherence_saved_formatting_v2",
                "semantic_reaudit_required_scene_refs": [],
                "transformations": [{"operation": "exclude_unrequested_condition", "path": "/x",
                                     "scene_ref": "R03:S01", "original": {"condition": "invented"}}]}):
            with self.subTest(provenance=provenance):
                self.audit["_formatting_provenance"] = provenance
                result = self.assess()
                self.assertIn("formatting_provenance_invalid", result["errors"])
                self.assertEqual(result["content_status"], "NEEDS_SEMANTIC_REAUDIT")
                self.assertFalse(result["retained"])

    def test_history_alias_conflict_and_duplicate_id_fail_before_audit(self):
        self.parent["dialogue"]["turns"][0]["turn_ref"] = "conflicting:ref"
        with self.assertRaisesRegex(ValueError, "history_reference_identity"):
            self.assess()
        del self.parent["dialogue"]["turns"][0]["turn_ref"]
        self.parent["dialogue"]["turns"].append(deepcopy(self.parent["dialogue"]["turns"][0]))
        with self.assertRaisesRegex(ValueError, "history_reference_identity"):
            self.assess()

    def test_current_and_history_id_collision_is_not_progress(self):
        self.parent["dialogue"]["turns"][0]["turn_id"] = self.current
        self.row["history_evidence_refs"] = []
        self.assertIn("current_history_reference_collision", self.assess()["errors"])

    def test_renderer_scopes_evidence_without_changing_saved_v3(self):
        inputs = dict(parent=self.parent, plan=self.plan, child=self.child,
                      requested_conditions=self.conditions)
        new = render_adherence(**inputs)
        self.assertIn("history_evidence_refs", new["prompt"])
        self.assertIn("Historical support never proves NEW progress", new["prompt"])
        self.assertIn("invented conditions or additional success criteria must not influence", new["prompt"])
        old = render_adherence(**inputs, prompt_version=INLINE_ADHERENCE_VERSION)
        self.assertIn(INLINE_ADHERENCE_TEMPLATE, old["prompt"])
        self.assertNotIn("history_evidence_refs", old["prompt"])
        self.assertEqual(old["messages"][1], new["messages"][1])


if __name__ == "__main__":
    unittest.main()
