"""Derived route decisions retain model semantics and reject missing judgments."""
from copy import deepcopy
import unittest

import event_state as scan
import test_event_v19 as event_fixture


class RoutePolicyV21Tests(unittest.TestCase):
    def setUp(self):
        self.role = {"role": "dataset_new_event", "primary_event": "Interview train ticket",
            "primary_goal": "Reach the already planned interview", "relation_to_history": "old_event_substep",
            "serves_existing_event": True, "existing_event": "Internship interview", "new_progress": True,
            "reason": "Train booking serves the interview.", "correction": "Create an event with its own main goal.",
            "history_evidence": [{"turn_ref": "P01:T001"}]}
        self.route = "dataset_new_event"

    def compile(self, value=None):
        return scan.compile_route_fulfillment(self.role if value is None else value, self.route)

    def test_old_step_derived_negative_without_format_retry(self):
        original = deepcopy(self.role)
        canonical = self.compile()
        self.assertFalse(canonical["satisfied"])
        self.assertFalse(scan.route_fulfilled(self.role, self.route))
        self.assertEqual(self.role, original)
        self.assertNotIn("satisfied", self.role)
        self.assertEqual({k:v for k,v in canonical.items() if k != "satisfied"}, self.role)

    def test_independent_problem_derived_positive(self):
        self.role.update(primary_event="Wallet recovery", primary_goal="Recover lost property",
            relation_to_history="distinct_event", serves_existing_event=False, existing_event="",
            reason="Recovery is a separate problem on the same trip.", correction="")
        self.assertTrue(self.compile()["satisfied"])

    def test_a_progress_derived_positive(self):
        self.route = "old_event_continuation"
        self.role.update(role=self.route, relation_to_history="continuation", correction="")
        self.assertTrue(self.compile()["satisfied"])

    def test_no_new_progress_is_negative_without_default(self):
        self.role.update(relation_to_history="distinct_event", serves_existing_event=False,
            new_progress=False, existing_event="", correction="Add concrete new progress.")
        self.assertFalse(self.compile()["satisfied"])

    def test_missing_required_semantics_never_fill_defaults(self):
        for field in ("role", "relation_to_history", "serves_existing_event", "new_progress"):
            value = deepcopy(self.role)
            value.pop(field)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.compile(value)

    def test_wrong_semantic_types_and_inconsistent_relations_fail(self):
        for field, value in (("serves_existing_event", 1), ("new_progress", "true"),
                             ("relation_to_history", None), ("relation_to_history", "distinct_event")):
            invalid = deepcopy(self.role)
            invalid[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.compile(invalid)

    def test_explicit_legacy_true_cannot_overwrite_policy_negative(self):
        self.role["satisfied"] = True
        with self.assertRaisesRegex(ValueError, "route_fulfillment_verdict_inconsistent"):
            self.compile()
        self.assertIs(self.role["satisfied"], True)

    def test_consistent_legacy_flag_accepted_and_canonical_revalidates(self):
        self.role["satisfied"] = False
        self.assertEqual(self.compile(), self.role)
        self.assertEqual(self.compile(self.compile()), self.role)

    def test_provenance_binds_explicit_fields_policy_and_raw(self):
        receipt = scan.route_decision_provenance(self.role, self.route)
        self.assertEqual(receipt["version"], scan.ROUTE_POLICY_VERSION)
        self.assertEqual(receipt["inputs"], {key: self.role[key] for key in
            ("role", "relation_to_history", "serves_existing_event", "new_progress")})
        self.assertFalse(receipt["output"]["satisfied"])
        self.assertEqual(receipt["decision_source"], "program_policy_derived_from_explicit_model_fields")
        self.assertFalse(receipt["semantic_certified"])
        self.role["satisfied"] = False
        legacy = scan.route_decision_provenance(self.role, self.route)
        self.assertNotEqual(receipt["raw_role_sha256"], legacy["raw_role_sha256"])
        self.assertEqual(legacy["decision_source"], "legacy_model_boolean_checked_against_policy")

    def test_template_asks_semantics_without_redundant_flag(self):
        self.assertNotIn("satisfied", scan.route_output_template(self.route))
        self.assertIn("DO NOT output a separate satisfied flag", scan.ROUTE_CONTRACT)

    def test_scan_materializes_canonical_without_raw_mutation(self):
        fixture = event_fixture.EventV19Tests()
        fixture.setUp()
        fixture.as_b(independent=False)
        fixture.result["route_fulfillment"].pop("satisfied")
        raw = deepcopy(fixture.result)
        checked = fixture.validate()
        self.assertEqual(raw, fixture.result)
        self.assertFalse(checked["route_fulfillment"]["satisfied"])
        self.assertTrue(scan.requires_replan(checked))
        self.assertEqual(fixture.validate(checked), checked)

    def test_repair_of_missing_check_preserves_negative_semantics_without_flag(self):
        previous = {"route": self.route, "route_fulfillment": deepcopy(self.role), "plan_checks": []}
        repair = {"errors": [{"path": "plan_checks[*].development_index"}]}
        current = deepcopy(previous)
        current["route_fulfillment"].update(relation_to_history="distinct_event", serves_existing_event=False,
            existing_event="", correction="", reason="Changed to positive.")
        with self.assertRaisesRegex(scan.EventStateValidationError,
                                    "event_state_repair_changed_unaffected_negative_role"):
            scan.validate_repair_preservation(previous, current, repair)

    def test_missing_correction_can_be_added_without_flipping_negative_semantics(self):
        previous = {"route": self.route, "route_fulfillment": deepcopy(self.role), "plan_checks": []}
        previous["route_fulfillment"]["correction"] = ""
        repair = {"errors": [{"path": "route_fulfillment.correction"}]}
        current = deepcopy(previous)
        current["route_fulfillment"]["correction"] = self.role["correction"]
        self.assertTrue(scan.validate_repair_preservation(previous, current, repair))
        current["route_fulfillment"].update(relation_to_history="distinct_event", serves_existing_event=False,
            existing_event="", correction="")
        with self.assertRaisesRegex(scan.EventStateValidationError, "event_state_repair_changed_negative_semantics"):
            scan.validate_repair_preservation(previous, current, repair)

    def test_own_evidence_error_cannot_enable_old_step_to_distinct_flip(self):
        previous = {"route": self.route, "route_fulfillment": deepcopy(self.role), "plan_checks": []}
        repair = {"errors": [{"path": "route_fulfillment.history_evidence[0].turn_ref"}]}
        current = deepcopy(previous)
        current["route_fulfillment"].update(relation_to_history="distinct_event", serves_existing_event=False,
            existing_event="", correction="")
        with self.assertRaisesRegex(scan.EventStateValidationError, "event_state_repair_changed_negative_semantics"):
            scan.validate_repair_preservation(previous, current, repair)


if __name__ == "__main__":
    unittest.main()
