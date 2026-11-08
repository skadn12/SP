"""Explicit functional relation cannot be canceled by a global distinct label."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

import event_state as scan
import realization_review
from evidence_refs import materialize_citations
import test_goal_checks_v29 as old_fixture


class ParentGoalsV34Tests(unittest.TestCase):
    def setUp(self):
        f = old_fixture.HistoryGoalChecksV29Tests(); f.setUp()
        self.f = f.f
        self.raw = deepcopy(self.f.result)
        self.raw["parent_goal_checks"] = [
            {"constraint_id": "c1", "parent_goal": {"objective": "Complete the planned weekend trip",
                "evidence": [{"turn_ref": "P01:T001"}, {"turn_ref": "R02:A:S01:T001"}]},
             "functional_relation": "separate_objective", "reason": "Recovery of lost property has its own objective."},
            {"constraint_id": "c2", "parent_goal": {"objective": "Hold the planned family dinner",
                "evidence": [{"turn_ref": "R02:B:S01:T004"}]},
             "functional_relation": "context_only", "reason": "The dinner is background and not advanced by this scene."}]

    def validate(self, raw=None):
        return scan.validate(self.raw if raw is None else raw, self.f.root, self.f.route,
                             self.f.scene, self.f.history, require_goal_checks=True, require_parent_goal_checks=True)

    def negative(self, relation="verifies_or_follows_up"):
        self.raw["parent_goal_checks"][0].update(functional_relation=relation,
            reason="The main work checks the result of arrangements for the same concrete trip.")
        return self.raw

    def test_positive_and_context_preserve_legacy_reviewer_projection(self):
        prior = deepcopy(self.raw); prior.pop("parent_goal_checks")
        checked = self.validate()
        self.assertFalse(scan.requires_replan(checked))
        self.assertEqual(realization_review.review_event_context(checked),
                         realization_review.review_event_context(scan.validate(prior, self.f.root,
                             self.f.route, self.f.scene, self.f.history, require_goal_checks=True)))
        self.assertEqual(self.raw["parent_goal_checks"][0]["parent_goal"]["evidence"],
                         [{"turn_ref": "P01:T001"}, {"turn_ref": "R02:A:S01:T001"}])

    def test_functional_followup_blocks_even_global_and_legacy_comparisons_pass(self):
        for relation in ("executes_or_prepares", "verifies_or_follows_up", "uncertain"):
            self.negative(relation)
            before = deepcopy(self.raw)
            checked = self.validate()
            self.assertTrue(checked["route_fulfillment"]["satisfied"])
            self.assertTrue(scan.requires_replan(checked))
            binding = scan.binding_plan_checks(checked)
            self.assertEqual(len(binding), 1)
            self.assertTrue(binding[0]["parent_goal_check"])
            self.assertNotIn("development_index", binding[0])
            self.assertEqual(binding[0]["relation"], scan.PARENT_GOAL_RELATIONS[relation])
            self.assertEqual(self.raw, before)
            provenance = scan.parent_goal_decision_provenance(self.raw, self.f.history)
            self.assertEqual(provenance["raw_checks"], self.raw["parent_goal_checks"])
            self.assertTrue(provenance["requires_replan"])
            self.assertEqual(provenance["derived_checks"][0]["relation"], scan.PARENT_GOAL_RELATIONS[relation])
            self.assertFalse(provenance["semantic_certified"])

    def test_missing_duplicate_extra_and_untyped_relation_cannot_pass(self):
        variants = []
        missing = deepcopy(self.raw); missing.pop("parent_goal_checks")
        variants.append((missing, "parent_goal_checks_required"))
        omitted = deepcopy(self.raw); omitted["parent_goal_checks"].pop()
        variants.append((omitted, "parent_goal_check_coverage_invalid"))
        duplicate = deepcopy(self.raw); duplicate["parent_goal_checks"].append(deepcopy(duplicate["parent_goal_checks"][0]))
        variants.append((duplicate, "parent_goal_check_duplicate_id"))
        free_relation = deepcopy(self.raw); free_relation["parent_goal_checks"][0]["relation"] = "independent"
        variants.append((free_relation, "parent_goal_check_fields_invalid"))
        invalid = deepcopy(self.raw); invalid["parent_goal_checks"][0]["functional_relation"] = "independent"
        variants.append((invalid, "parent_goal_functional_relation_invalid"))
        for value, code in variants:
            with self.subTest(code=code), self.assertRaises(scan.EventStateValidationError) as caught:
                self.validate(value)
            self.assertIn(code, {e["code"] for e in caught.exception.errors})

    def test_null_objective_allowed_only_for_nonservice_or_uncertain(self):
        self.raw["parent_goal_checks"][0]["parent_goal"] = None
        self.validate()
        self.raw["parent_goal_checks"][0]["functional_relation"] = "uncertain"
        self.assertTrue(scan.requires_replan(self.validate()))
        self.raw["parent_goal_checks"][0]["functional_relation"] = "executes_or_prepares"
        with self.assertRaises(scan.EventStateValidationError) as caught:
            self.validate()
        self.assertIn("parent_goal_service_requires_cited_objective", {e["code"] for e in caught.exception.errors})

    def test_nonnull_parent_needs_exact_history_reference_even_when_positive(self):
        for spans in ([], [{"turn_ref": "P01:T001", "quote": "This never appeared."}], [{"turn_ref": "UNKNOWN"}]):
            value = deepcopy(self.raw); value["parent_goal_checks"][0]["parent_goal"]["evidence"] = spans
            with self.subTest(spans=spans), self.assertRaises(scan.EventStateValidationError):
                self.validate(value)

    def test_format_repair_preserves_functional_verdict_objective_and_valid_text(self):
        self.negative()
        self.raw["parent_goal_checks"][0]["parent_goal"]["evidence"] = []
        with self.assertRaises(scan.EventStateValidationError) as caught:
            self.validate()
        repair = scan.build_repair(caught.exception, self.raw, self.f.history)
        self.assertEqual(len(repair["original_negative_parent_goal_checks"]), 1)
        fixed = deepcopy(self.raw)
        fixed["parent_goal_checks"][0]["parent_goal"]["evidence"] = [{"turn_ref": "P01:T001"}]
        self.assertTrue(scan.validate_repair_preservation(self.raw, fixed, repair))
        self.validate(fixed)
        for key, value in (("functional_relation", "separate_objective"), ("reason", "Different explanation."),
                           ("parent_goal", {"objective": "A replacement unrelated event", "evidence": [{"turn_ref": "P01:T001"}]})):
            changed = deepcopy(fixed); changed["parent_goal_checks"][0][key] = value
            with self.subTest(key=key), self.assertRaises(scan.EventStateValidationError):
                scan.validate_repair_preservation(self.raw, changed, repair)

    def test_unknown_reference_can_be_fixed_without_rejudging_negative(self):
        self.negative()
        self.raw["parent_goal_checks"][0]["parent_goal"]["evidence"] = [{"turn_ref": "UNKNOWN"}]
        with self.assertRaises(scan.EventStateValidationError) as caught:
            self.validate()
        repair = scan.build_repair(caught.exception, self.raw, self.f.history)
        fixed = deepcopy(self.raw); fixed["parent_goal_checks"][0]["parent_goal"]["evidence"] = ["P01:T001"]
        self.assertTrue(scan.validate_repair_preservation(self.raw, fixed, repair))
        self.assertTrue(scan.requires_replan(self.validate(fixed)))

    def test_timeline_patch_archives_bound_row_and_preserves_other_parent_judgments(self):
        self.negative()
        self.raw["timeline"][0]["evidence"] = []
        self.raw["segments"][0]["none_reason"] = "This draft entry is being classified separately."
        with self.assertRaises(scan.EventStateValidationError) as caught:
            self.validate()
        repair = scan.build_repair(caught.exception, self.raw, self.f.history)
        self.assertEqual(repair["repair_mode"], "timeline_entry_patch")
        changes = [{"development_index": row["development_index"], "constraint_ids": []}
                   for row in self.raw["plan_checks"]]
        patch = {"timeline_repairs": [{"constraint_id": "c1", "choice": "planned_development"}],
                 "plan_check_repairs": changes}
        merged = scan.merge_event_repair(self.raw, patch, repair, self.f.history)
        checked = self.validate(merged)
        self.assertEqual(checked["parent_goal_checks"][0]["constraint_id"], "c1")
        archived = checked["nonhistorical_parent_goal_checks"][0]
        self.assertEqual(archived["source_constraint_id"], "c1")
        self.assertEqual(archived["entry"]["functional_relation"], "verifies_or_follows_up")
        # It is explicitly relocated nonhistory, so it cannot keep vetoing the current plan.
        self.assertFalse(scan.parent_goal_blockers(checked))
        changed = deepcopy(merged); changed["parent_goal_checks"][0]["reason"] = "Altered"
        with self.assertRaises(scan.EventStateValidationError):
            scan.validate_repair_preservation(self.raw, changed, repair)

    def test_request_is_opt_in_and_preserves_complete_history(self):
        fixture = Path(__file__).parent.parent / "h4_sequential_ab_v32/runs/inherited_a2_b3_01/requests/015-cc-train-episode-104814-B-S02-event-state-1-1/REQUEST.json"
        c = json.loads(json.loads(fixture.read_text())["messages"][0]["content"].split("EVENT_STATE_INPUT_JSON\n", 1)[1])
        args = (c["root_id"], c["route"], c["scene_plan"], c["cumulative_history"], c["dataset_cards"], c["latest_state"])
        strict = scan.build_request(*args, require_parent_goal_checks=True)
        context = json.loads(strict["messages"][0]["content"].split("EVENT_STATE_INPUT_JSON\n", 1)[1])
        self.assertEqual(context["cumulative_history"], c["cumulative_history"])
        self.assertIn("parent_goal_checks_template", context)
        old = scan.build_request(*args)
        self.assertNotIn("V34 B CONCRETE PARENT-OBJECTIVE", old["messages"][0]["content"])
        prior = deepcopy(self.raw); prior.pop("parent_goal_checks")
        self.assertFalse(scan.requires_replan(scan.validate(prior, self.f.root, self.f.route, self.f.scene,
                                                           self.f.history, require_goal_checks=True)))


if __name__ == "__main__":
    unittest.main()
