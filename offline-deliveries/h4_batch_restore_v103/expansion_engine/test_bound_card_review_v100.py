"""Authored source-binding counterexamples plus immutable v99 raw regression.

These offline checks validate source identity/reference boundaries and binding
negative feedback. They do not certify semantic support or model reliability.
No archived response is migrated or rewritten.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import card_source_contract as source
import event_identity as identity
import event_state as event
import realization_review
import runner
from event_identity_test_fixtures import comparison, identity as authored_identity
from test_event_state_v92 import review_fixture
from test_runner_semantic_integration import FakeCaller
import test_event_identity_v98 as v98_identity_tests


class BoundCardReviewV100Tests(unittest.TestCase):
    def setUp(self):
        self.history = [
            {"turn_ref": "r:P01:T001", "speaker": "Elena", "text": "The old repair was completed last week."},
            {"turn_ref": "r:P01:T002", "speaker": "Milan", "text": "We can discuss a new problem today."},
        ]
        self.card = {
            "card_id": "new-service-seed", "card_sha256": "a" * 64, "guidance_level": "seed",
            "actors": [{"actor_id": "a1", "role": "customer"}, {"actor_id": "a2", "role": "provider"}],
            "steps": [
                {"step_id": "s1", "actor_id": "a1", "action": "request a spa service", "state": "requested"},
                {"step_id": "s2", "actor_id": "a2", "action": "accept the service request", "state": "agreed"},
            ],
            "dependencies": [{"before": "s1", "after": "s2", "relation": "acceptance responds to request"}],
        }
        self.scene = {
            "scene_id": "S01", "actors": ["Elena", "Milan"], "dataset_card_id": self.card["card_id"],
            "developments": ["Elena asks Milan to repair a new bicycle brake fault today.",
                             "Milan accepts Elena's new request to repair the brakes."],
            "card_development_links": [
                {"card_step_ids": ["s1"], "dependency_indices": [], "development_indices": [1]},
                {"card_step_ids": ["s2"], "dependency_indices": [1], "development_indices": [2]},
            ],
        }
        self.scene["event_identity"] = authored_identity(self.card, self.history, self.scene)
        self.review = review_fixture(self.scene, self.history, [self.card])
        self.comp = self.review["card_checks"][0]["function_check"]["comparison"]

    def validate(self, review=None):
        return event.validate(self.review if review is None else review, "r", "dataset_new_event",
                              self.scene, self.history, [self.card])

    def assert_invalid(self, review, code):
        with self.assertRaises(event.EventStateValidationError) as caught:
            self.validate(review)
        self.assertIn(code, [error["code"] for error in caught.exception.errors])
        return caught.exception.errors

    def test_v99_actual_copied_source_review_fails_new_plan_contract_without_editing_raw(self):
        old = Path(__file__).resolve().parents[2] / "h4_planner_schema_regression_v99/regression/requests"
        folder = next(old.glob("007-*25652*event-state*"))
        request_path, response_path = folder / "REQUEST.json", folder / "RESPONSE.json"
        before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (request_path, response_path)}
        request = json.loads(request_path.read_text())
        context, _ = json.JSONDecoder().raw_decode(request["messages"][0]["content"].split("EVENT_STATE_INPUT_JSON\n", 1)[1])
        raw = json.loads(response_path.read_text())["text"]
        with self.assertRaises(event.EventStateValidationError) as caught:
            event.validate(raw, context["root_id"], context["route"], context["scene_plan"],
                           context["cumulative_history"], context["dataset_cards"])
        self.assertTrue(any("comparison" in error["path"] and ("source_binding" in error.get("allowed", []) or error["code"].startswith("card_source"))
                            for error in caught.exception.errors), caught.exception.errors)
        self.assertEqual(before, {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (request_path, response_path)})

    def test_assigned_card_id_card_hash_and_projection_hash_all_bind(self):
        self.validate()
        for field in ("card_id", "card_sha256", "projection_sha256"):
            with self.subTest(field=field):
                bad = deepcopy(self.review)
                bad["card_checks"][0]["function_check"]["comparison"]["source_binding"][field] = "wrong-current-card-binding"
                errors = self.assert_invalid(bad, "card_source_binding_mismatch")
                self.assertTrue(any(error["path"].endswith("." + field) for error in errors))

    def test_source_binding_cannot_be_omitted_or_accept_foreign_fields(self):
        for mode in ("missing", "extra"):
            with self.subTest(mode=mode):
                bad = deepcopy(self.review)
                comp = bad["card_checks"][0]["function_check"]["comparison"]
                if mode == "missing":
                    del comp["source_binding"]
                    self.assert_invalid(bad, "event_identity_fields_invalid")
                else:
                    comp["source_binding"]["root_id"] = "other-root"
                    self.assert_invalid(bad, "card_source_binding_fields_invalid")

    def test_actual_history_metadata_wildcards_and_containers_are_not_source_paths(self):
        forbidden = ["/scene_plan/goal", "/event_identity/purpose", "/history/0/text", "/metadata/model",
                     "/actors/*/role", "/steps/-1/action", "/steps/0", "/card_id", "/card_sha256",
                     "/guidance_level", "/actors/0/actor_id", "/steps/0/step_id", "/steps/99/action",
                     "/steps/0/evidence/0/quote", "source:/steps/0/action", "/steps/00/action"]
        for pointer in forbidden:
            with self.subTest(pointer=pointer):
                bad = deepcopy(self.review)
                bad["card_checks"][0]["function_check"]["comparison"]["purpose"]["source_paths"] = [pointer]
                self.assert_invalid(bad, "card_source_path_not_allowed")

    def test_source_paths_are_arrays_of_distinct_exact_literal_leaves(self):
        path = source.build(self.card)["allowed_source_paths"][0]
        for value, code in [(path, "card_source_paths_not_array"), ([path, path], "card_source_path_duplicate"),
                            ([None], "card_source_path_not_allowed"), ([{}], "card_source_path_not_allowed")]:
            with self.subTest(value=value):
                bad = deepcopy(self.review)
                bad["card_checks"][0]["function_check"]["comparison"]["purpose"]["source_paths"] = value
                self.assert_invalid(bad, code)

    def test_nested_raw_metadata_never_changes_projection_hash_or_enters_prompt(self):
        original = source.build(self.card)
        polluted = deepcopy(self.card)
        sentinel = "V100_RAW_SOURCE_METADATA_MUST_STAY_OUT"
        polluted["raw_dialogue"] = sentinel
        polluted["actors"][0].update(source_speaker=sentinel, raw={"text": sentinel})
        polluted["steps"][0].update(evidence=[{"quote": sentinel}], raw_text=sentinel)
        polluted["dependencies"][0].update(evidence=[{"quote": sentinel}], source_metadata=sentinel)
        self.assertEqual(source.build(polluted), original)
        self.assertNotIn(sentinel, json.dumps(source.projection(polluted)))
        request = event.build_request("r", "dataset_new_event", self.scene, self.history, [polluted], {})
        self.assertNotIn(sentinel, request["messages"][0]["content"])
        changed = deepcopy(self.card)
        changed["steps"][0]["state"] = "reported_success"
        self.assertNotEqual(source.build(changed)["source_binding"]["projection_sha256"],
                            original["source_binding"]["projection_sha256"])

    def test_plan_source_free_prose_cannot_sneak_back_into_bound_pairs(self):
        bad = deepcopy(self.review)
        bad["card_checks"][0]["function_check"]["comparison"]["purpose"]["source"] = "candidate proposal repackaged as a source requirement"
        self.assert_invalid(bad, "event_identity_fields_invalid")

    def test_valid_source_references_do_not_erase_independent_step_negative(self):
        self.review["card_checks"][0]["step_checks"][0].update(verdict="revise", reason="Authored negative: action object was substituted.")
        result = self.validate()
        bindings = event.binding_plan_checks(result)
        self.assertTrue(event.requires_replan(result))
        self.assertEqual([row["check_kind"] for row in bindings], ["card_step"])
        self.assertFalse(event.derive_coverage(result, self.history)["semantic_certified"])

    def test_empty_source_paths_preserve_explicit_nested_negative(self):
        for key in ("purpose", "start_state", "end_state", "request_acceptance", "experience_state"):
            self.comp[key]["source_paths"] = []
        self.comp["request_acceptance"].update(required=True, verdict="revise", request_evidence=[], acceptance_evidence=[])
        self.review["card_checks"][0]["function_check"]["reason"] = "Source applicability is not explicit in this fixture; actual request is judged absent."
        result = self.validate()
        self.assertTrue(event.requires_replan(result))
        self.assertIn("request_acceptance", [row.get("dimension") for row in event.binding_plan_checks(result)])
        self.assertEqual(result["card_checks"][0]["function_check"]["comparison"]["request_acceptance"]["verdict"], "revise")

    def test_empty_paths_do_not_resolve_explicit_experience_status_conflict(self):
        self.comp["experience_state"].update(source_paths=[], source="completed_experience", actual="future_arrangement", verdict="compatible")
        self.assert_invalid(self.review, "event_identity_explicit_experience_status_mismatch")
        self.comp["experience_state"]["verdict"] = "revise"
        self.assertTrue(event.requires_replan(self.validate()))

    def test_seed_domain_adaptation_remains_allowed_and_uncertified(self):
        self.assertIn("spa", self.card["steps"][0]["action"])
        self.assertIn("bicycle", self.scene["developments"][0])
        result = self.validate()
        self.assertFalse(event.requires_replan(result))
        self.assertFalse(event.derive_coverage(result, self.history)["semantic_certified"])
        self.assertEqual(result, self.review)

    def test_plan_request_contains_exact_frozen_card_projection_and_bound_output_shape(self):
        request = event.build_request("r", "dataset_new_event", self.scene, self.history, [self.card], {})
        self.assertEqual(request["max_tokens"], 4096)
        context, _ = json.JSONDecoder().raw_decode(request["messages"][0]["content"].split("EVENT_STATE_INPUT_JSON\n", 1)[1])
        contract = context["output_contract"]["card_checks"]["items"][0]
        shape = contract["function_comparison_shape"]
        self.assertEqual(shape["source_binding"], source.build(self.card)["source_binding"])
        self.assertEqual(set(shape["purpose"]), {"source_paths", "actual"})
        self.assertEqual(context["cumulative_history"], self.history)
        self.assertIn(source.build(self.card)["source_binding"]["projection_sha256"], request["messages"][0]["content"])

    def test_scene_comparison_contract_is_unchanged(self):
        prior = Path(__file__).resolve().parents[2] / "h4_planner_schema_regression_v99/expansion_engine/event_identity.py"
        import importlib.util
        spec = importlib.util.spec_from_file_location("bound_test_frozen_v99_identity", prior)
        old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
        self.assertEqual(identity.comparison_schema(self.card, scene=True), old.comparison_schema(self.card, scene=True))
        f = v98_identity_tests.EventIdentityV98Tests(); f.setUp()
        scene, review = f.scene_fixture()
        checked = f.validate_scene(scene, review)
        self.assertEqual(checked["status"], "PASS")
        self.assertTrue(checked["schema_valid"])
        self.assertNotIn("source_binding", review["card_checks"][0]["comparison"])
        self.assertFalse(checked["semantic_certified"])

    def test_bad_pointer_cannot_be_deleted_in_actual_bounded_reviewer_retry(self):
        for mode in ("delete", "replace"):
            with self.subTest(mode=mode):
                bad = deepcopy(self.review)
                bad["card_checks"][0]["function_check"]["comparison"]["purpose"]["source_paths"] = ["/scene_plan/goal"]
                corrected = deepcopy(self.review)
                if mode == "delete":
                    corrected["card_checks"][0]["function_check"]["comparison"]["purpose"]["source_paths"] = []
                self.validate(corrected)  # Empty is legal alone; retry still has an evidence obligation.
                with tempfile.TemporaryDirectory() as td:
                    run = Path(td)
                    caller = FakeCaller(run, [("event_state_extractor", bad), ("event_state_extractor", corrected)])
                    with patch.object(runner, "branch_history_ok", return_value=self.history):
                        result = runner.extract_event_state(caller, run, "r", "dataset_new_event", self.history,
                                                            [self.card], self.scene, {"facts": []}, plan_attempt=1)
                    self.assertEqual(len(caller.calls), runner.EVENT_STATE_ATTEMPTS)
                    self.assertEqual(len(caller.responses), 0)
                    second = caller.calls[1]["folder"]
                    validation = json.loads((second / "VALIDATION.json").read_text())
                    progress = json.loads((second / "CORRECTION_PROGRESS.json").read_text())
                    if mode == "delete":
                        self.assertIsNone(result)
                        self.assertEqual(validation["status"], "REJECT")
                        self.assertEqual(progress["status"], "REJECT")
                        self.assertTrue((second / "FULL_VALIDATION_BEFORE_CORRECTION_GATE.json").exists())
                        self.assertTrue(any("deleted" in error["code"] for error in progress["errors"]), progress)
                    else:
                        self.assertIsNotNone(result)
                        self.assertEqual(validation["status"], "PASS")
                        self.assertEqual(progress["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
