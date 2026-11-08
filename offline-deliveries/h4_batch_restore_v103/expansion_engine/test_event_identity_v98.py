"""Exposed v97 failures + authored v98 annotations; no model-accuracy claim.

No archive is modified. Adding an authored comparison to a COPY tests routing,
not that the model will discover its meaning or repair a future generation.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest

import event_identity as identity
import event_state as event
import planner_contract as planner
import realization_review as reviewer
import runner
from event_identity_test_fixtures import comparison, bind_comparison, identity as fixture_identity
from test_event_state_v92 import review_fixture


OLD = Path(__file__).resolve().parents[2] / "h4_semantic_review_mixed_pilot_v97"
REQUESTS = OLD / "expansion_engine/runs/four_roots_B_scene01/requests"


def archived(label, *, scene=False):
    folder = next(REQUESTS.glob(label))
    files = [folder / name for name in ("REQUEST.json", "RESPONSE.json", "PROOF.json")]
    hashes = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    payload = json.loads(files[0].read_text())
    marker = "REVIEW INPUT DATA:\n" if scene else "EVENT_STATE_INPUT_JSON\n"
    context, _ = json.JSONDecoder().raw_decode(payload["messages"][0]["content"].split(marker, 1)[1])
    response = json.loads(json.loads(files[1].read_text())["text"])
    return context, response, hashes


class EventIdentityV98Tests(unittest.TestCase):
    def setUp(self):
        self.history = [{"turn_ref": "r:P01:T001", "speaker": "Elena", "text": "The earlier repair is finished."},
                        {"turn_ref": "r:P01:T002", "speaker": "Milan", "text": "That job was completed last week."}]
        self.card = {"card_id": "service-seed", "guidance_level": "seed",
                     "actors": [{"actor_id": "a1", "role": "customer"}, {"actor_id": "a2", "role": "provider"}],
                     "steps": [{"step_id": "s1", "actor_id": "a1", "action": "request a spa service", "state": "requested"},
                               {"step_id": "s2", "actor_id": "a2", "action": "accept the service request", "state": "agreed"}],
                     "dependencies": [{"before": "s1", "after": "s2", "relation": "acceptance responds to request"}]}
        self.scene = {"scene_id": "S01", "actors": ["Elena", "Milan"], "dataset_card_id": "service-seed",
                      "developments": ["Elena requests a new bicycle brake repair from Milan.",
                                       "Milan accepts Elena's request to repair the bicycle brakes."],
                      "card_development_links": [
                          {"card_step_ids": ["s1"], "dependency_indices": [], "development_indices": [1]},
                          {"card_step_ids": ["s2"], "dependency_indices": [1], "development_indices": [2]}]}
        self.scene["event_identity"] = fixture_identity(self.card, self.history, self.scene)
        self.response = review_fixture(self.scene, self.history, [self.card])
        self.response["card_checks"][0]["function_check"]["comparison"] = self.positive_comparison()

    def plan_span(self, index):
        return {"development_index": index, "quote": self.scene["developments"][index - 1]}

    def positive_comparison(self, *, bound=True):
        result = comparison(self.card, self.history, self.scene)
        result.update(purpose={"source": "Request and accept a new personal service", "actual": "Request and accept a new repair service"},
                      start_state={"source": "Service not requested", "actual": "New brake repair not requested"},
                      end_state={"source": "Request accepted; work pending", "actual": "Brake repair accepted; work pending"},
                      current_trigger="New brake fault, separate from the completed old job")
        result["roles"] = [{"actor_id": "a1", "participant": "Elena", "role": "repair customer"},
                           {"actor_id": "a2", "participant": "Milan", "role": "repair provider"}]
        result["nearest_old_event"] = {"summary": "Earlier completed repair", "history_evidence": [
            {"turn_ref": self.history[0]["turn_ref"], "quote": self.history[0]["text"]}],
            "difference": "New brake fault and new current request; old job is finished"}
        result["request_acceptance"] = {"required": True, "verdict": "compatible",
                                        "request_evidence": [self.plan_span(1)], "acceptance_evidence": [self.plan_span(2)]}
        return bind_comparison(result, self.card) if bound else result

    def validate(self, response=None):
        return event.validate(response or self.response, "r", "dataset_new_event", self.scene, self.history, [self.card])

    def scene_fixture(self):
        realization = {"root_id": "r", "route": "dataset_new_event", "scene_id": "S01", "status": "scene",
                       "turns": [{"speaker": "Elena", "text": "This is a new brake fault. Will you repair my bicycle brakes?"},
                                 {"speaker": "Milan", "text": "Yes, I accept your request to repair the brakes."}],
                       "feedback_topic_ids": [], "no_scene_reason": ""}
        def span(index):
            return {"turn_ref": f"NEW:T{index:03d}", "quote": realization["turns"][index - 1]["text"]}
        comp = self.positive_comparison(bound=False)
        comp["request_acceptance"].update(request_evidence=[span(1)], acceptance_evidence=[span(2)])
        response = {"root_id": "r", "route": "dataset_new_event", "scene_id": "S01",
                    "checks": [{"development_index": i, "realized": True, "evidence": [span(i)]} for i in (1, 2)],
                    "new_event_or_progress": True, "consistent_with_history": True,
                    "duplicates_existing_event": False, "reason": "Authored positive: new requested repair and acceptance.",
                    "consistency_checks": [{"check_id": key, "verdict": "not_applicable",
                                            "reason": "OFFLINE fixture annotation", "evidence": []}
                                           for key in reviewer.CONSISTENCY_CHECK_IDS],
                    "card_checks": [
                        {"kind": "event_function", "id": self.card["card_id"], "realized": True,
                         "reason": "Authored seed adaptation preserves service request and acceptance; work remains pending.",
                         "comparison": comp, "evidence": [span(1), span(2)]},
                        {"kind": "step", "id": "s1", "realized": True, "evidence": [span(1)]},
                        {"kind": "step", "id": "s2", "realized": True, "evidence": [span(2)]},
                        {"kind": "dependency", "id": 1, "realized": True, "evidence": [span(1), span(2)]}]}
        return realization, response

    def validate_scene(self, realization, response):
        return reviewer.validate_review(response, "r", "dataset_new_event", self.scene, realization,
                                        history=self.history, dataset_card=self.card)

    def test_seed_domain_can_change_without_keyword_veto(self):
        self.assertEqual(planner.collect_event_identity_errors(self.scene, self.card, self.history), [])
        result = self.validate()
        self.assertFalse(event.requires_replan(result))
        realization, response = self.scene_fixture()
        checked = self.validate_scene(realization, response)
        self.assertEqual(checked["status"], "PASS")
        self.assertFalse(checked["semantic_certified"])

    def test_planner_requires_complete_identity_exact_history_and_actual_actor_ids(self):
        for mutation, code in [("missing", "event_identity_fields_invalid"), ("quote", "event_identity_quote_invalid"),
                               ("actor", "event_identity_actor_coverage_invalid"), ("person", "event_identity_participant_invalid")]:
            scene = deepcopy(self.scene)
            if mutation == "missing": scene.pop("event_identity")
            elif mutation == "quote": scene["event_identity"]["nearest_old_event"]["history_evidence"][0]["quote"] = "invented"
            elif mutation == "actor": scene["event_identity"]["roles"][0]["actor_id"] = "a3"
            else: scene["event_identity"]["roles"][0]["participant"] = "Other root person"
            self.assertIn(code, [e["code"] for e in planner.collect_event_identity_errors(scene, self.card, self.history)])

    def test_nearest_old_event_cannot_cite_plan_or_another_root(self):
        response = deepcopy(self.response)
        response["card_checks"][0]["function_check"]["comparison"]["nearest_old_event"]["history_evidence"][0]["turn_ref"] = "other:P01:T001"
        with self.assertRaises(event.EventStateValidationError) as caught: self.validate(response)
        self.assertIn("event_identity_reference_invalid", [e["code"] for e in caught.exception.errors])

    def test_current_request_evidence_cannot_be_historical_or_after_acceptance(self):
        for mutation in ("history", "later", "missing"):
            response = deepcopy(self.response)
            req = response["card_checks"][0]["function_check"]["comparison"]["request_acceptance"]
            if mutation == "history": req["request_evidence"] = [{"turn_ref": "r:P01:T001", "quote": self.history[0]["text"]}]
            elif mutation == "later": req.update(request_evidence=[self.plan_span(2)], acceptance_evidence=[self.plan_span(1)])
            else: req["request_evidence"] = []
            with self.assertRaises(event.EventStateValidationError): self.validate(response)

    def test_nested_explicit_request_negative_binds_with_all_other_positives(self):
        request = self.response["card_checks"][0]["function_check"]["comparison"]["request_acceptance"]
        request.update(verdict="revise", request_evidence=[])
        checked = self.validate()
        bindings = event.binding_plan_checks(checked)
        self.assertEqual([(x["check_kind"], x["dimension"]) for x in bindings], [("card_event_identity", "request_acceptance")])

    def test_explicit_experience_status_substitution_requires_negative(self):
        dim = self.response["card_checks"][0]["function_check"]["comparison"]["experience_state"]
        dim.update(source="completed_experience", actual="future_arrangement")
        with self.assertRaises(event.EventStateValidationError) as caught: self.validate()
        self.assertIn("event_identity_explicit_experience_status_mismatch", [e["code"] for e in caught.exception.errors])
        dim["verdict"] = "revise"
        self.assertTrue(event.requires_replan(self.validate()))

    def test_scene_requires_request_in_earlier_turn_and_preserves_negative_route(self):
        realization, response = self.scene_fixture()
        request = response["card_checks"][0]["comparison"]["request_acceptance"]
        request["request_evidence"] = deepcopy(request["acceptance_evidence"])
        malformed = self.validate_scene(realization, response)
        self.assertFalse(malformed["schema_valid"])
        self.assertEqual(runner.classify_review_failure(malformed), "reviewer_failure")
        request.update(verdict="revise", request_evidence=[])
        checked = self.validate_scene(realization, response)
        self.assertTrue(checked["schema_valid"])
        self.assertEqual(checked["status"], "REJECT")
        self.assertEqual(runner.classify_review_failure(checked), "realizer_repair")
        repair = runner.repair_for_review(checked, realization)
        self.assertEqual(repair["event_identity_findings"][0]["revise_dimensions"], ["request_acceptance"])
        self.assertNotIn("nearest_old_event", json.dumps(repair["event_identity_findings"]))

    def test_scene_nested_experience_negative_binds_and_routes_realizer(self):
        realization, response = self.scene_fixture()
        response["card_checks"][0]["comparison"]["experience_state"].update(
            source="completed_experience", actual="future_arrangement", verdict="revise")
        checked = self.validate_scene(realization, response)
        self.assertEqual(checked["status"], "REJECT")
        self.assertTrue(checked["schema_valid"])
        self.assertEqual(runner.classify_review_failure(checked), "realizer_repair")

    def test_request_contract_contains_full_comparison_shape_without_raising_output_limit(self):
        request = event.build_request("r", "dataset_new_event", self.scene, self.history, [self.card], {})
        self.assertEqual(request["max_tokens"], 4096)
        context, _ = json.JSONDecoder().raw_decode(request["messages"][0]["content"].split("EVENT_STATE_INPUT_JSON\n", 1)[1])
        shape = context["output_contract"]["card_checks"]["items"][0]["function_comparison_shape"]
        self.assertEqual(set(shape), identity.COMPARISON_FIELDS | {"source_binding"})
        prompt = reviewer.build_review_prompt("r", "dataset_new_event", self.scene, self.history,
                                              self.scene_fixture()[0], dataset_card=self.card)
        self.assertIn('"comparison"', prompt)
        self.assertEqual(context["cumulative_history"], self.history)

    def test_malformed_nested_values_report_errors_without_exceptions(self):
        for field, bad in [("roles", None), ("nearest_old_event", []), ("request_acceptance", []),
                           ("experience_state", {"source": [], "actual": {}, "verdict": []})]:
            comp = self.positive_comparison(bound=False); comp[field] = bad
            self.assertTrue(identity.collect_errors(comp, self.card, self.history, comparison=True, scene=self.scene))

    def test_v97_raws_fail_missing_comparison_without_semantic_auto_upgrade(self):
        for label in ("008-*19053*", "013-*9557*", "018-*104814*"):
            context, response, hashes = archived(label)
            with self.assertRaises(event.EventStateValidationError) as caught:
                event.validate(response, context["root_id"], context["route"], context["scene_plan"],
                               context["cumulative_history"], context["dataset_cards"])
            self.assertTrue(any("comparison" in e.get("missing", []) or "comparison" in e["path"]
                                for e in caught.exception.errors))
            self.assertEqual({f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in hashes}, hashes)

    def test_v97_19053_authored_experience_negative_routes_without_rewriting_old_plan(self):
        context, old, hashes = archived("008-*19053*")
        card, scene, history = context["dataset_cards"][0], context["scene_plan"], context["cumulative_history"]
        response = deepcopy(old); function = response["card_checks"][0]["function_check"]
        function["comparison"] = comparison(card, history, scene, bound=True)
        function["comparison"]["purpose"]["actual"] = "Finalize a future volunteer schedule"
        function["comparison"]["experience_state"].update(source="completed_experience", actual="future_arrangement", verdict="revise")
        function["reason"] = "Authored exposed-case negative: completed experience became future preparation."
        checked = event.validate(response, context["root_id"], context["route"], scene, history, [card])
        self.assertEqual(event.binding_plan_checks(checked)[0]["dimension"], "experience_state")
        self.assertEqual({f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in hashes}, hashes)

    def test_v97_9557_authored_missing_new_request_routes_at_plan_and_scene(self):
        context, old, _ = archived("013-*9557*")
        card, scene, history = context["dataset_cards"][0], context["scene_plan"], context["cumulative_history"]
        response = deepcopy(old); function = response["card_checks"][0]["function_check"]
        comp = comparison(card, history, scene, bound=True)
        comp["request_acceptance"] = {"required": True, "verdict": "revise", "request_evidence": [],
                                      "acceptance_evidence": [{"development_index": 2, "quote": scene["developments"][1]}],
                                      "source_paths": comp["request_acceptance"]["source_paths"]}
        function["comparison"] = comp
        function["reason"] = "Authored exposed-case negative: old risk request was accepted; presenting risks is not a new request."
        self.assertTrue(event.requires_replan(event.validate(response, context["root_id"], context["route"], scene, history, [card])))
        ctx, raw, hashes = archived("015-*9557*", scene=True)
        response = deepcopy(raw); function = response["card_checks"][0]
        function["comparison"] = comparison(ctx["dataset_card_guidance"], ctx["branch_history"], ctx["scene_plan"])
        function["comparison"]["request_acceptance"] = {"required": True, "verdict": "revise", "request_evidence": [],
            "acceptance_evidence": [{"turn_ref": "NEW:T006", "quote": "I formally accept the request to help lead the risk mitigation strategy for these areas."}]}
        result = reviewer.validate_review(response, ctx["root_id"], ctx["route"], ctx["scene_plan"], ctx["new_scene"],
                                         history=ctx["branch_history"], dataset_card=ctx["dataset_card_guidance"],
                                         latest_state=ctx["latest_state"], event_state=ctx["event_state"])
        self.assertTrue(result["schema_valid"], result["failures"])
        self.assertEqual(runner.classify_review_failure(result), "realizer_repair")
        self.assertEqual({f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in hashes}, hashes)

    def test_v97_104814_existing_function_negative_and_plan2_bad_quote_remain(self):
        for label in ("018-*104814*", "020-*104814*", "021-*104814*"):
            ctx, old, hashes = archived(label)
            card, scene, history = ctx["dataset_cards"][0], ctx["scene_plan"], ctx["cumulative_history"]
            response = deepcopy(old)
            response["card_checks"][0]["function_check"]["comparison"] = comparison(card, history, scene, bound=True)
            if label.startswith("018"):
                checked = event.validate(response, ctx["root_id"], ctx["route"], scene, history, [card])
                self.assertIn("card_event_function", [x["check_kind"] for x in event.binding_plan_checks(checked)])
            else:
                with self.assertRaises(event.EventStateValidationError) as caught:
                    event.validate(response, ctx["root_id"], ctx["route"], scene, history, [card])
                self.assertIn("event_state_plan_evidence_quote_invalid", [e["code"] for e in caught.exception.errors])
            self.assertEqual({f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in hashes}, hashes)

    def test_literal_quote_checks_cannot_certify_its_semantic_support(self):
        # Intentionally wrong authored purpose attached to exact real quotes.
        # Mechanical acceptance must remain explicitly uncertified, never a claim
        # that code solved the v97 model's evidence-entailment failure.
        realization, response = self.scene_fixture()
        response["card_checks"][0]["comparison"]["purpose"]["actual"] = "INTENTIONALLY WRONG: a completed flight"
        checked = self.validate_scene(realization, response)
        self.assertTrue(checked["schema_valid"])
        self.assertFalse(checked["semantic_certified"])


if __name__ == "__main__": unittest.main()
