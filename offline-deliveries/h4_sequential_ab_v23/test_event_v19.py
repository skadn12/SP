"""Offline regression cases for review mistakes observed in the v18 pilot."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

import event_state as scan
from evidence_refs import citation_provenance, materialize_citations


class EventV19Tests(unittest.TestCase):
    def setUp(self):
        self.root, self.route = "fixture", "old_event_continuation"
        self.history = [
            {"turn_ref": "P01:T001", "text": "I plan to take the train and arrive Saturday morning."},
            {"turn_ref": "R02:A:S01:T001", "text": "I will book the hotel for the weekend trip."}]
        self.scene = {"scene_id": "S01", "developments": [
            "Emily newly reports she has booked the train ticket.",
            "James agrees to meet at the hotel after the morning arrival."],
            "event_definition": {"trigger": "Confirm the planned trip", "main_goal": "Confirm bookings",
                                 "history_relation": "continuation"}}
        self.role = {"role": self.route, "satisfied": True,
            "primary_event": "Trip confirmation", "primary_goal": "Confirm bookings",
            "relation_to_history": "continuation", "serves_existing_event": True,
            "existing_event": "Planned weekend trip", "new_progress": True,
            "reason": "New confirmation advances the plan.", "correction": "",
            "history_evidence": [{"turn_ref": "P01:T001"}]}
        self.result = {"root_id": self.root, "route": self.route, "scene_id": "S01",
            "segments": [{"segment_id": "P01", "none_reason": ""},
                         {"segment_id": "R02:A:S01", "none_reason": "Hotel proposal is relevant background, no separate constraint selected."}],
            "timeline": [{"constraint_id": "c1", "status": "planned", "evidence_basis": "reported",
                "statement": "Emily plans train travel with a morning arrival.",
                "evidence": [{"turn_ref": "P01:T001"}], "implication": "New dialogue can report later completion."}],
            "plan_checks": [{"development_index": index, "verdict": "compatible",
                "constraint_ids": ["c1"], "comparison_kind": "new_development",
                "reason": "The new scene coherently advances the plan."} for index in (1, 2)],
            "route_fulfillment": deepcopy(self.role)}

    def validate(self, result=None):
        return scan.validate(self.result if result is None else result, self.root,
                             self.route, self.scene, self.history)

    def error(self, result=None):
        with self.assertRaises(scan.EventStateValidationError) as caught:
            self.validate(result)
        return caught.exception

    def make_negative(self, index=0, kind="timing_ambiguity"):
        check = self.result["plan_checks"][index]
        check.update(verdict="revise", comparison_kind=kind,
            reason="Clarify arrival in the city versus the hotel.",
            conflict={"type": "timing_ambiguity", "historical_claim": "Morning arrival is planned.",
                      "proposed_claim": "The new plan mentions afternoon arrival.",
                      "correction": "Distinguish morning city arrival from afternoon hotel check-in."})
        return check

    def as_b(self, *, independent=True):
        self.route = "dataset_new_event"
        self.result["route"] = self.route
        self.result["route_fulfillment"].update(role=self.route, satisfied=independent,
            primary_event="Lost wallet" if independent else "Interview train ticket",
            primary_goal="Recover property" if independent else "Reach old interview",
            relation_to_history="distinct_event" if independent else "old_event_substep",
            serves_existing_event=not independent, existing_event="" if independent else "Existing interview",
            reason="Own property recovery goal" if independent else "Only supports the old interview",
            correction="" if independent else "Transfer the card to an event with an independent main goal.")

    def test_new_completion_is_accepted_without_old_completion_evidence(self):
        checked = self.validate()
        self.assertEqual(checked["timeline"][0]["status"], "planned")
        self.assertFalse(scan.requires_replan(checked))
        self.assertEqual(self.result["timeline"][0]["evidence"], [{"turn_ref": "P01:T001"}])

    def test_refs_materialize_exact_text_with_provenance_and_no_input_mutation(self):
        original = deepcopy(self.result)
        checked = self.validate()
        self.assertEqual(checked["timeline"][0]["evidence"][0]["quote"], self.history[0]["text"])
        receipt = citation_provenance(self.result, self.history)
        self.assertEqual(receipt["dereferenced_citations"][0]["selection"], "model_selected_reference")
        self.assertFalse(receipt["semantic_support_verified"])
        self.assertEqual(self.result, original)

    def test_unknown_reference_is_reviewer_error_and_not_silently_selected(self):
        self.result["timeline"][0]["evidence"][0]["turn_ref"] = "UNKNOWN"
        self.assertEqual(self.error().errors[0]["code"], "event_state_citation_materialization_invalid")

    def test_existing_model_quote_is_not_repaired(self):
        self.result["timeline"][0]["evidence"][0]["quote"] = "I already finished the trip."
        self.assertIn("event_state_evidence_quote_invalid", {e["code"] for e in self.error().errors})

    def test_timing_ambiguity_has_concrete_comparison_and_correction(self):
        negative = deepcopy(self.make_negative())
        checked = self.validate()
        self.assertTrue(scan.requires_replan(checked))
        self.assertEqual(scan.binding_plan_checks(checked), [negative])

    def test_new_progress_cannot_use_unsupported_past_assertion_reason(self):
        self.make_negative(kind="new_development")["conflict"]["type"] = "unsupported_prior_assertion"
        self.assertIn("event_state_new_progress_not_prior_assertion", {e["code"] for e in self.error().errors})

    def test_prior_history_assertion_with_precise_conflict_remains_binding(self):
        row = self.make_negative(kind="historical_assertion")
        row["conflict"].update(type="unsupported_prior_assertion",
            proposed_claim="Claims history already established ticket purchase.",
            correction="State it as a new completion, not a previously established fact.")
        self.assertTrue(scan.requires_replan(self.validate()))

    def test_compatible_is_compact_without_conflict(self):
        self.validate()
        self.result["plan_checks"][0]["conflict"] = None
        self.validate()
        self.assertEqual(len(scan.plan_checks_template(self.scene)[0]), 5)

    def test_train_for_existing_interview_is_valid_negative_with_actionable_feedback(self):
        self.as_b(independent=False)
        checked = self.validate()
        finding = scan.binding_plan_checks(checked)[0]
        self.assertTrue(finding["route_role_check"])
        self.assertIn("independent main goal", finding["correction"])

    def test_lost_wallet_on_old_trip_can_be_distinct_event(self):
        self.as_b(independent=True)
        self.result["route_fulfillment"]["history_evidence"] = []
        self.assertFalse(scan.requires_replan(self.validate()))

    def test_false_distinct_with_real_progress_is_reviewer_inconsistency(self):
        self.as_b(independent=True)
        self.result["route_fulfillment"].update(satisfied=False, correction="Do something else")
        self.assertIn("route_fulfillment_verdict_inconsistent", {e["code"] for e in self.error().errors})

    def test_distinct_event_cannot_also_serve_old_task(self):
        self.as_b(independent=True)
        self.result["route_fulfillment"].update(serves_existing_event=True, existing_event="Interview")
        self.assertIn("route_fulfillment_relation_service_inconsistent", {e["code"] for e in self.error().errors})

    def test_absence_claim_cannot_enter_evidenceless_timeline(self):
        self.result["timeline"].append({"constraint_id": "c2", "status": "uncertain", "evidence_basis": "inferred",
            "statement": "No booking existed.", "evidence": [], "implication": "New event."})
        self.assertIn("event_state_evidence_count_invalid", {e["code"] for e in self.error().errors})

    def test_missing_row_targeted_and_valid_negative_cannot_disappear(self):
        self.make_negative()
        self.result["plan_checks"].pop()
        error = self.error()
        repair = scan.build_repair(error, self.result, self.history)
        self.assertEqual([r["development_index"] for r in repair["missing_checks"]], [2])
        corrected = deepcopy(self.result)
        corrected["plan_checks"].append({"development_index": 2, "verdict": "compatible", "constraint_ids": [],
            "comparison_kind": "new_development", "reason": "No constraint conflicts with the new event."})
        self.assertTrue(scan.validate_repair_preservation(self.result, corrected, repair))
        self.validate(corrected)
        corrected["plan_checks"][0].update(verdict="compatible", comparison_kind="new_development")
        corrected["plan_checks"][0].pop("conflict")
        with self.assertRaises(scan.EventStateValidationError) as caught:
            scan.validate_repair_preservation(self.result, corrected, repair)
        self.assertEqual(caught.exception.errors[0]["code"], "event_state_repair_changed_unaffected_negative")

    def test_contradictory_role_can_be_fixed_in_reviewer_repair(self):
        self.as_b(independent=True)
        self.result["route_fulfillment"].update(satisfied=False, correction="Unclear false verdict.")
        repair = scan.build_repair(self.error(), self.result, self.history)
        corrected = deepcopy(self.result)
        corrected["route_fulfillment"].update(satisfied=True, correction="")
        self.assertTrue(scan.validate_repair_preservation(self.result, corrected, repair))
        self.validate(corrected)

    def test_unrelated_schema_repair_preserves_negative_role(self):
        self.as_b(independent=False)
        self.result["plan_checks"].pop()
        repair = scan.build_repair(self.error(), self.result, self.history)
        changed = deepcopy(self.result)
        changed["route_fulfillment"].update(satisfied=True)
        with self.assertRaises(scan.EventStateValidationError):
            scan.validate_repair_preservation(self.result, changed, repair)

    def test_prompt_request_enumerates_every_development_and_reference_contract(self):
        request = scan.build_request(self.root, self.route, self.scene, self.history, [], {})
        data = json.loads(request["messages"][0]["content"].split("\nEVENT_STATE_INPUT_JSON\n")[1])
        self.assertEqual(data["expected_development_indices"], [1, 2])
        self.assertEqual([row["development_index"] for row in data["plan_checks_template"]], [1, 2])
        self.assertNotIn("proposed_change", data["plan_checks_template"][0])
        self.assertIn("does not need an old completion quote", scan.PROMPT)
        self.assertIn("Recovering a newly lost wallet", scan.PROMPT)
        self.assertIn("references ONLY, without quote text", scan.PROMPT)

    def test_null_uncertainty_is_preserved_as_unspecified(self):
        self.result["timeline"][0]["uncertainty"] = None
        raw = deepcopy(self.result)
        checked = self.validate()
        self.assertIsNone(checked["timeline"][0]["uncertainty"])
        self.assertEqual(self.result, raw)
        omitted = deepcopy(self.result)
        omitted["timeline"][0].pop("uncertainty")
        self.assertEqual(scan.plan_diagnostics(checked), scan.plan_diagnostics(self.validate(omitted)))
        self.assertIn("neither asserts that no uncertainty exists", scan.PROMPT)

    def test_null_uncertainty_never_clears_negative(self):
        self.make_negative()
        self.result["timeline"][0]["uncertainty"] = None
        checked = self.validate()
        self.assertTrue(scan.requires_replan(checked))
        self.assertEqual(checked["plan_checks"], self.result["plan_checks"])

    def test_other_nonstring_uncertainty_types_still_fail(self):
        for value in (False, 0, [], {}):
            with self.subTest(value=value):
                self.result["timeline"][0]["uncertainty"] = value
                self.assertIn("event_state_uncertainty_invalid", {e["code"] for e in self.error().errors})

    def test_bare_reference_strings_are_equivalent_only_as_evidence(self):
        checked_objects = self.validate()
        self.result["timeline"][0]["evidence"] = ["P01:T001"]
        self.result["route_fulfillment"]["history_evidence"] = ["P01:T001"]
        self.assertEqual(self.validate(), checked_objects)
        self.assertEqual(self.result["plan_checks"][0]["constraint_ids"], ["c1"])

    def test_actual_v19_a_progress_responses_replay_without_judgment_changes(self):
        """Only codec/null acceptance changes; original Qwen judgments remain.

        This verifies structural replay, not semantic certification of the
        reviewer's morning/afternoon interpretation or fact extraction.
        """
        pilot = Path(__file__).resolve().parent.parent / "h4_sequential_ab_v19/probes/historical_failures_01"
        case = json.loads((pilot / "cases/138343_A_progress.json").read_text())
        context = case["context"]
        for request in ("003-138343_A_progress-1", "004-138343_A_progress-2"):
            with self.subTest(request=request):
                response = json.loads((pilot / "requests" / request / "RESPONSE.json").read_text())
                raw = json.loads(response["text"])
                before = deepcopy(raw)
                checked = scan.validate(raw, context["root_id"], context["route"],
                    context["scene_plan"], context["cumulative_history"])
                self.assertEqual(raw, before)
                self.assertEqual(checked["plan_checks"], raw["plan_checks"])
                self.assertEqual([row["verdict"] for row in checked["plan_checks"]], ["compatible"] * 4)
                self.assertFalse(scan.requires_replan(checked))
                for old, new in zip(raw["timeline"], checked["timeline"]):
                    self.assertEqual({k:v for k,v in old.items() if k != "evidence"},
                                     {k:v for k,v in new.items() if k != "evidence"})
                self.assertEqual(checked["route_fulfillment"]["reason"], raw["route_fulfillment"]["reason"])
                self.assertEqual(checked["route_fulfillment"]["satisfied"], raw["route_fulfillment"]["satisfied"])


if __name__ == "__main__":
    unittest.main()
