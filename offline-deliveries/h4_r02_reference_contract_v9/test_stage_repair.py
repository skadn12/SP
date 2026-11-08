"""Regression contracts for stage-aware R02 repair in v5.

These tests use no provider API. They exercise the validation/dispatch helpers
and a fake Caller transport so failed scenes remain reproducible offline.
"""

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import runner
import realization_review


class StageRepairTests(unittest.TestCase):
    def test_mapping_and_null_shortfall_reported_together_without_modifying_response(self):
        from copy import deepcopy
        card = {"card_id": "c", "steps": [{"step_id": "s1"}], "dependencies": []}
        plan = {"root_id": "root", "route": runner.ROUTES[1], "shortfall_reason": None,
                "scenes": [{"scene_id": "S01", "dataset_card_id": "c",
                            "developments": ["A requests a booking.", "B records the request."],
                            "card_development_links": [{"card_step_ids": ["s1"], "dependency_indices": [1],
                                                        "development_indices": [1]}]}]}
        original = deepcopy(plan)
        with self.assertRaises(runner.planner_contract.MappingValidationError) as caught:
            runner.validate_plan(plan, "root", runner.ROUTES[1], [], [card], "S01")
        codes = {row["code"] for row in caught.exception.errors}
        self.assertIn("card_development_link_dependencies_invalid", codes)
        self.assertIn("shortfall_reason_invalid", codes)
        self.assertEqual(plan, original)

    def setUp(self):
        self.card = {
            "card_id": "card_scaffold",
            "guidance_level": "scaffold",
            "steps": [
                {"step_id": "s1", "action": "request booking", "state": "proposed"},
                {"step_id": "s2", "action": "receive option", "state": "reported_action"},
                {"step_id": "s3", "action": "decline booking", "state": "reported_action"},
            ],
            "dependencies": [
                {"before": "s1", "after": "s2", "relation": "order"},
                {"before": "s2", "after": "s3", "relation": "condition_change"},
            ],
        }
        self.scene = {
            "scene_id": "S02",
            "dataset_card_id": "card_scaffold",
            "developments": [
                "A traveler requests a train booking.",
                "The agent proposes a train option.",
                "The traveler declines booking and changes to a restaurant request.",
            ],
            "card_development_links": [
                {"card_step_ids": ["s1", "s2"], "dependency_indices": [1],
                 "development_indices": [1, 2]},
                {"card_step_ids": ["s3"], "dependency_indices": [2],
                 "development_indices": [3]},
            ],
        }

    def test_card_fidelity_requires_valid_step_and_dependency_coverage(self):
        runner.validate_card_plan_fidelity(self.scene, self.card)

        missing_dependency = json.loads(json.dumps(self.scene))
        missing_dependency["card_development_links"][1]["dependency_indices"] = []
        with self.assertRaises((ValueError, TypeError)):
            runner.validate_card_plan_fidelity(missing_dependency, self.card)

        invalid_card_reference = json.loads(json.dumps(self.scene))
        invalid_card_reference["card_development_links"][0]["card_step_ids"] = ["not-a-card-step"]
        with self.assertRaises((ValueError, TypeError)):
            runner.validate_card_plan_fidelity(invalid_card_reference, self.card)

    def test_card_fidelity_rejects_v4_style_unrelated_plan_without_card_links(self):
        unrelated_workspace_plan = {
            "scene_id": "S02",
            "developments": [
                "Emily sets up a temporary office.",
                "David introduces Emily to a team lead.",
            ],
        }
        with self.assertRaises((ValueError, TypeError)):
            runner.validate_card_plan_fidelity(unrelated_workspace_plan, self.card)

    def test_invalid_speaker_diagnostics_include_each_1_based_turn_and_name(self):
        turns = [
            {"speaker": "David", "text": "Allowed."},
            {"speaker": "Sarah", "text": "Not in the scene actor list."},
            {"speaker": "Emily", "text": "Allowed."},
            {"speaker": "Sarah", "text": "Still invalid."},
        ]
        details = runner.invalid_speaker_details(turns, {"David", "Emily"})
        self.assertEqual(details, [
            {"turn_position": 2, "speaker": "Sarah"},
            {"turn_position": 4, "speaker": "Sarah"},
        ])

    def test_review_failure_routes_only_valid_unmet_development_to_realizer(self):
        root, route = "root-1", "dataset_new_event"
        scene = {"scene_id": "S01", "developments": ["Emily confirms the booking."]}
        realization = {
            "root_id": root, "route": route, "scene_id": "S01", "status": "scene",
            "turns": [{"speaker": "Emily", "text": "I have not confirmed it yet."}],
        }
        valid_unmet = {
            "root_id": root, "route": route, "scene_id": "S01",
            "checks": [{"development_index": 1, "realized": False, "evidence": [],
                        "reason": "There is no confirmation."}],
            "new_event_or_progress": False, "consistent_with_history": True,
            "duplicates_existing_event": False, "reason": "The confirmation is missing.",
        }
        rejected = realization_review.validate_review(
            valid_unmet, root, route, scene, realization,
        )
        self.assertEqual(rejected["status"], "REJECT")
        self.assertEqual(runner.classify_review_failure(rejected), "realizer_repair")

        malformed = realization_review.validate_review(
            "{broken json", root, route, scene, realization,
        )
        self.assertEqual(runner.classify_review_failure(malformed), "reviewer_failure")

        fabricated_quote = json.loads(json.dumps(valid_unmet))
        fabricated_quote["checks"][0].update(
            realized=True,
            evidence=[{"turn_ref": "NEW:T001", "quote": "I confirmed the booking."}],
            reason="The quote supports confirmation.",
        )
        fabricated_quote.update(new_event_or_progress=True, reason="Confirmation was made.")
        invalid_evidence = realization_review.validate_review(
            fabricated_quote, root, route, scene, realization,
        )
        self.assertTrue(any("quote_not_exact" in failure for failure in invalid_evidence["failures"]))
        self.assertEqual(runner.classify_review_failure(invalid_evidence), "reviewer_failure")

    def test_invalid_review_evidence_blocks_rewrite_even_if_another_step_is_unmet(self):
        root, route = "root-2", "dataset_new_event"
        scene = {"scene_id": "S01", "developments": [
            "Emily confirms the booking.", "David records the confirmation."
        ]}
        realization = {
            "root_id": root, "route": route, "scene_id": "S01", "status": "scene",
            "turns": [{"speaker": "Emily", "text": "I have not confirmed it yet."}],
        }
        review = {
            "root_id": root, "route": route, "scene_id": "S01",
            "checks": [
                {"development_index": 1, "realized": False, "evidence": [],
                 "reason": "No confirmation is stated."},
                {"development_index": 2, "realized": True,
                 "evidence": [{"turn_ref": "NEW:T001", "quote": "David recorded the confirmation."}],
                 "reason": "This is a fabricated citation."},
            ],
            "new_event_or_progress": False, "consistent_with_history": True,
            "duplicates_existing_event": False, "reason": "The first step is missing.",
        }
        rejected = realization_review.validate_review(review, root, route, scene, realization)
        self.assertTrue(any("quote_not_exact" in failure for failure in rejected["failures"]))
        self.assertTrue(any(":not_realized:" in failure for failure in rejected["failures"]))
        self.assertEqual(runner.classify_review_failure(rejected), "reviewer_failure")


class CallerRetryTests(unittest.TestCase):
    class ProviderStatusError(Exception):
        def __init__(self, status_code):
            super().__init__(f"HTTP {status_code}")
            self.status_code = status_code

    def make_caller(self, run_dir, statuses, *, response_metadata=None):
        observations = {"calls": [], "attempt_payloads": []}

        class FakeClient:
            def __init__(self, on_http_attempt):
                self.on_http_attempt = on_http_attempt

            def complete(self, *, prompt, max_tokens, json_object):
                self.on_http_attempt(payload)
                observations["attempt_payloads"].append(payload)
                call_number = len(observations["calls"])
                observations["calls"].append((prompt, max_tokens, json_object))
                status = statuses[call_number] if call_number < len(statuses) else None
                if status is not None:
                    raise CallerRetryTests.ProviderStatusError(status)
                metadata = ({"returned_model": runner.MODEL, "finish_reason": "stop"}
                            if response_metadata is None else response_metadata[call_number])
                return SimpleNamespace(raw_text="{}", metadata=metadata)

        def factory(_profile, *, on_http_attempt):
            return FakeClient(on_http_attempt)

        caller = runner.Caller.__new__(runner.Caller)
        caller.run = Path(run_dir)
        caller.limit = 8
        caller.profile = object()
        caller.factory = factory
        caller.annotate = lambda value: value
        caller.capacity = lambda *_args: {"status": "PASS"}
        caller.tokenizer = object()
        caller.http_counter = 0
        caller.request_counter = 0
        payload = {
            "model": runner.MODEL,
            "messages": [{"role": "user", "content": "same frozen prompt"}],
        }
        observations["payload"] = payload
        return caller, payload, observations

    def test_503_and_504_get_exactly_one_identical_payload_retry(self):
        for status in (503, 504):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temp:
                caller, payload, observed = self.make_caller(temp, [status, None])
                raw, _directory = caller.call("same-slot", payload, {"role": "planner"})
                self.assertEqual(raw, "{}")
                self.assertEqual(len(observed["calls"]), 2)
                self.assertEqual(observed["attempt_payloads"], [payload, payload])
                self.assertEqual(observed["calls"][0], observed["calls"][1])
                self.assertEqual(caller.http_counter, 2)

                attempt_files = list(Path(temp).rglob("HTTP_ATTEMPT*.json"))
                self.assertEqual(len(attempt_files), 2)
                attempt_records = [json.loads(path.read_text()) for path in attempt_files]
                self.assertEqual(
                    {row.get("payload_sha256") for row in attempt_records},
                    {runner.base.digest(payload)},
                )
                response = json.loads((Path(temp) / "requests/001-same-slot/RESPONSE.json").read_text())
                self.assertEqual(
                    [attempt["http_status"] for attempt in response["attempts"]],
                    [status, None],
                )

    def test_non_retryable_status_is_not_replayed(self):
        with tempfile.TemporaryDirectory() as temp:
            caller, payload, observed = self.make_caller(temp, [502])
            with self.assertRaises(RuntimeError):
                caller.call("same-slot", payload, {"role": "planner"})
            self.assertEqual(len(observed["calls"]), 1)
            self.assertEqual(observed["attempt_payloads"], [payload])
            self.assertEqual(caller.http_counter, 1)

    def test_retry_limit_is_one_even_when_both_attempts_return_504(self):
        with tempfile.TemporaryDirectory() as temp:
            caller, payload, observed = self.make_caller(temp, [504, 504])
            with self.assertRaises(RuntimeError):
                caller.call("same-slot", payload, {"role": "planner"})
            self.assertEqual(len(observed["calls"]), 2)
            self.assertEqual(observed["attempt_payloads"], [payload, payload])
            self.assertEqual(caller.http_counter, 2)
            self.assertEqual(len(list(Path(temp).rglob("HTTP_ATTEMPT*.json"))), 2)

    def test_invalid_response_metadata_is_audited_without_overwriting_http_result(self):
        invalid_metadata = [
            {"returned_model": "unexpected/model", "finish_reason": "stop", "http_status": 200},
            {"returned_model": runner.MODEL, "finish_reason": "length", "http_status": 200},
        ]
        for metadata in invalid_metadata:
            with self.subTest(metadata=metadata), tempfile.TemporaryDirectory() as temp:
                caller, payload, observed = self.make_caller(
                    temp, [None], response_metadata=[metadata],
                )
                with self.assertRaisesRegex(RuntimeError, "provider_request_failed_after_retry_policy"):
                    caller.call("invalid-metadata", payload, {"role": "planner"})
                self.assertEqual(len(observed["calls"]), 1)
                directory = Path(temp) / "requests/001-invalid-metadata"
                self.assertTrue((directory / "ERROR.json").exists())
                self.assertEqual(len(list(directory.glob("HTTP_RESULT*.json"))), 1)
                result = json.loads(next(directory.glob("HTTP_RESULT*.json")).read_text())
                self.assertEqual(result["http_status"], 200)
                error = json.loads((directory / "ERROR.json").read_text())
                self.assertNotEqual(error.get("error_type"), "FileExistsError")
                self.assertEqual(len(error["attempts"]), 1)

    def test_non_retryable_response_status_keeps_original_status_and_terminal_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            metadata = {"returned_model": runner.MODEL, "finish_reason": "stop", "http_status": 502}
            caller, payload, observed = self.make_caller(
                temp, [None], response_metadata=[metadata],
            )
            with self.assertRaisesRegex(RuntimeError, "provider_request_failed_after_retry_policy"):
                caller.call("bad-response-status", payload, {"role": "planner"})
            self.assertEqual(len(observed["calls"]), 1)
            directory = Path(temp) / "requests/001-bad-response-status"
            error = json.loads((directory / "ERROR.json").read_text())
            self.assertEqual(error["attempts"][-1]["http_status"], 502)
            self.assertNotEqual(error.get("error_type"), "FileExistsError")
            self.assertEqual(len(list(directory.glob("HTTP_RESULT*.json"))), 1)

    def test_terminal_504_response_preserves_both_results_without_audit_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            metadata = {"returned_model": runner.MODEL, "finish_reason": "stop", "http_status": 504}
            caller, payload, observed = self.make_caller(
                temp, [None, None], response_metadata=[metadata, metadata],
            )
            with self.assertRaisesRegex(RuntimeError, "provider_request_failed_after_retry_policy"):
                caller.call("terminal-504", payload, {"role": "planner"})
            self.assertEqual(len(observed["calls"]), 2)
            self.assertEqual(observed["attempt_payloads"], [payload, payload])
            directory = Path(temp) / "requests/001-terminal-504"
            result_files = sorted(directory.glob("HTTP_RESULT*.json"))
            self.assertEqual(len(result_files), 2)
            self.assertEqual([json.loads(path.read_text())["http_status"] for path in result_files], [504, 504])
            error = json.loads((directory / "ERROR.json").read_text())
            self.assertEqual([item["http_status"] for item in error["attempts"]], [504, 504])
            self.assertEqual(error["automatic_retries"], 1)
            self.assertNotEqual(error.get("error_type"), "FileExistsError")


class ProduceSceneAuditTests(unittest.TestCase):
    def test_terminal_reviewer_schema_failure_writes_one_final_acceptance_without_dialogue_rewrite(self):
        root, route = "offline-root", runner.ROUTES[0]
        scene = {"scene_id": "S01", "actors": ["Emily", "David"],
                 "developments": ["Emily confirms an appointment."]}
        realization = {
            "root_id": root, "route": route, "scene_id": "S01", "status": "scene",
            "turns": [{"speaker": "Emily" if i % 2 == 0 else "David",
                       "text": f"New appointment discussion detail {i + 1}."} for i in range(12)],
            "feedback_topic_ids": [], "no_scene_reason": "",
        }
        observations = []
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)

            class FakeSceneCaller:
                def call(self, label, payload, proof):
                    observations.append((label, payload, proof))
                    directory = run / "requests" / f"{len(observations):03d}-{label}"
                    directory.mkdir(parents=True)
                    if "-realizer-" in label:
                        return json.dumps(realization), directory
                    # Keep both independent reviewer attempts schema-invalid.
                    return "{}", directory

            request = ({"model": runner.MODEL, "messages": [{"role": "user", "content": "offline request"}]},
                       {"role": "realizer"})
            with patch.object(runner, "make_request", return_value=request), patch.object(
                runner.duplicate_check, "inspect_duplicates", return_value={"status": "PASS", "reasons": [], "matches": []}
            ):
                accepted, audits = runner.produce_scene(
                    FakeSceneCaller(), run, root, route, [], [], scene, [],
                )
            self.assertIsNone(accepted)
            self.assertEqual(sum("-realizer-" in label for label, _, _ in observations), 1)
            self.assertEqual(sum("-review-" in label for label, _, _ in observations), runner.REVIEW_RETRIES + 1)
            self.assertEqual(len(audits), 1)
            self.assertEqual(audits[0]["status"], "REVIEWER_FAILURE_NO_DIALOGUE_REWRITE")
            self.assertIs(audits[0]["schema_valid"], False)
            acceptance_files = list(run.rglob("ACCEPTANCE.json"))
            self.assertEqual(len(acceptance_files), 1)
            self.assertEqual(json.loads(acceptance_files[0].read_text()), audits[0])
            review_files = list(run.rglob("REVIEW.json"))
            self.assertEqual(len(review_files), runner.REVIEW_RETRIES + 1)
            for path in review_files:
                self.assertIs(json.loads(path.read_text())["schema_valid"], False)


if __name__ == "__main__":
    unittest.main()
