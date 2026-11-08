"""Offline integration tests for actual collection flow, without model/network calls.

All dialogue below is explicitly synthetic test text and stays in temporary
directories. The fake transport records the same request/response artifact shape
used by the live caller; contracts, duplicate checks and history append are real.
"""
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import candidate_runner as c


ROOT = "MOCK-ONLY-root"
A, B = c.r.ROUTES
BASELINE = [
    {"turn_ref": f"MOCK:P:T{index:03d}", "speaker": "Emily" if index % 2 else "James",
     "text": f"MOCK ONLY historical statement {index}; this substantive baseline utterance is preserved exactly."}
    for index in range(1, 5)
]
CARD = {"card_id": "MOCK-card", "guidance_level": "seed",
        "actors": [{"actor_id": "visitor"}, {"actor_id": "helper"}],
        "core_premise": "MOCK ONLY distinct collection test interaction."}


def model_plan(route, scene_id="S01"):
    return {**c.identity(ROOT, route, scene_id), "actors": ["Emily", "James"],
            "goal": "MOCK ONLY make a concrete test decision.",
            "developments": ["MOCK ONLY discuss the option.", "MOCK ONLY agree the next action."],
            "dataset_card_id": CARD["card_id"] if route == B else None,
            "source_anchors": [] if route == B else [{"turn_ref": BASELINE[0]["turn_ref"],
                                                        "quote": BASELINE[0]["text"]}],
            "actor_mapping": ([{"card_actor_id": "visitor", "root_actor": "Emily"},
                               {"card_actor_id": "helper", "root_actor": "James"}] if route == B else [])}


def model_scene(route, scene_id="S01", length=12):
    return {**c.identity(ROOT, route, scene_id), "status": "scene",
            "turns": [{"speaker": "Emily" if index % 2 else "James",
                       "text": f"MOCK ONLY generated {scene_id} utterance {index}: a unique response used solely by offline tests."}
                      for index in range(1, length + 1)],
            "dataset_card_id": CARD["card_id"] if route == B else None,
            "feedback_topic_ids": [], "no_scene_reason": ""}


def model_review(route, scene_id="S01"):
    return {**c.identity(ROOT, route, scene_id), "verdict": "acceptable",
            "core_checks": {name: {"status": "ok", "reason": "MOCK ONLY test finding.",
                                    "evidence_refs": ["NEW:T001"]}
                            for name in c.contract.CORE_CHECKS},
            "plan_observations": [], "summary": "MOCK ONLY test diagnostic."}


class RecordedFakeCaller:
    def __init__(self, run, responses, *, assert_generated_before_review=True):
        self.run, self.responses = run, {key: list(values) for key, values in responses.items()}
        self.observed = []
        self.role_counts = defaultdict(int)
        self.assert_generated_before_review = assert_generated_before_review

    def call(self, label, data, proof):
        role = proof["role"]
        index = self.role_counts[role]
        self.role_counts[role] += 1
        context = json.JSONDecoder().raw_decode(data["messages"][0]["content"].split("\nCANDIDATE_INPUT_JSON\n", 1)[1])[0]
        number = len(self.observed) + 1
        directory = self.run / "requests" / f"{number:03d}-{label}"
        proof = {**proof, "payload_sha256": c.r.base.digest(data)}
        self.observed.append({"role": role, "payload": deepcopy(data), "proof": deepcopy(proof),
                              "context": context, "directory": directory})
        c.r.base.write(directory / "REQUEST.json", data)
        c.r.base.write(directory / "PROOF.json", proof)
        c.r.base.write(directory / "RESERVATION.json", {"request_number": number})
        c.r.base.write(directory / "HTTP_ATTEMPT.json", {"http_attempt_number": number})
        if role == "reviewer" and self.assert_generated_before_review:
            letter = "A" if proof["route"] == A else "B"
            slot = self.run / "candidates" / proof["root_id"] / letter / proof["scene_id"]
            assert list(slot.glob("GENERATED_*.json")), "review ran before raw candidate persistence"
        if role not in self.responses or index >= len(self.responses[role]):
            raise AssertionError(f"unexpected_fake_call:{role}:{index}")
        response = self.responses[role][index]
        if isinstance(response, Exception):
            c.r.base.write(directory / "TRANSPORT_ERROR.json", {"error": str(response)})
            raise response
        raw = response if isinstance(response, str) else json.dumps(response, ensure_ascii=False, indent=2)
        c.r.base.write(directory / "HTTP_RESPONSE_01.json", {"text": raw, "metadata": {"mock_only": True}})
        c.r.base.write(directory / "RESPONSE.json", {"text": raw, "metadata": {"mock_only": True}, "attempts": 1})
        return raw, directory


class CandidateCollectionIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)
        c.r.base.write(self.run / "feedback/COMPILED_FEEDBACK.json", {"target_solver_identity_sha256": "MOCK-feedback-hash"})
        self.feedback = patch.object(c.r.base.feedback, "append_feedback_to_request",
            side_effect=lambda payload, *args, **kwargs: (payload, {"mock_only": True}))
        self.feedback.start()
        self.addCleanup(self.feedback.stop)
        self.network = patch("socket.socket", side_effect=AssertionError("offline_tests_cannot_access_network"))
        self.network.start()
        self.addCleanup(self.network.stop)

    def collect(self, route, responses, *, scene_id="S01", history=None, baseline=None, run=None):
        run = self.run if run is None else run
        baseline = deepcopy(BASELINE) if baseline is None else baseline
        history = deepcopy(baseline) if history is None else history
        before = deepcopy((history, baseline, CARD))
        caller = RecordedFakeCaller(run, responses)
        records, cumulative = c.collect_slot(caller, run, ROOT, route, scene_id, history, baseline,
                                             deepcopy(CARD) if route == B else None)
        self.assertEqual((history, baseline, CARD), before)
        return records, cumulative, caller

    def standard(self, route, *, scene_id="S01", scene=None, review=None, planner=None):
        return {"planner": [model_plan(route, scene_id) if planner is None else planner],
                "realizer": [model_scene(route, scene_id) if scene is None else scene],
                "reviewer": [model_review(route, scene_id) if review is None else review]}

    def assert_pending(self, records, cumulative, generated_turns=12):
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["disposition"], "retained_pending_review")
        self.assertEqual(cumulative[:len(BASELINE)], BASELINE)
        self.assertEqual(len(cumulative), len(BASELINE) + generated_turns)
        self.assertFalse(records[0]["training_pool_eligible"])
        self.assertFalse(records[0]["decision"]["training_admitted"])
        self.assertFalse(records[0]["semantic_certified"])
        self.assertTrue(all(turn["origin"] == "provisional_candidate" and turn["training_pool_eligible"] is False
                            for turn in cumulative[len(BASELINE):]))

    def test_A_and_B_use_the_same_nonblocking_candidate_flow(self):
        for route in (A, B):
            with self.subTest(route=route):
                records, history, caller = self.collect(route, self.standard(route))
                self.assert_pending(records, history)
                self.assertEqual([item["role"] for item in caller.observed], ["planner", "realizer", "reviewer"])
                for request in caller.observed:
                    self.assertEqual(request["context"]["cumulative_history"], BASELINE)
                    self.assertEqual(request["proof"]["history_sha256"], c.r.base.digest(BASELINE))
                    self.assertEqual(request["context"]["dataset_card"], CARD if route == B else None)
                    self.assertFalse(request["proof"]["training_pool_eligible"])

    def test_invalid_planner_does_not_block_A_or_B_generation(self):
        for route in (A, B):
            with self.subTest(route=route):
                records, history, caller = self.collect(route, self.standard(route, planner='{"bad":"plan"}'))
                self.assert_pending(records, history)
                realizer = next(x for x in caller.observed if x["role"] == "realizer")
                self.assertIsNone(realizer["context"]["advisory_plan"])
                plan_advisory = c.r.base.read(self.run / "candidates" / ROOT / ("A" if route == A else "B") / "S01/PLAN_ADVISORY.json")
                self.assertFalse(plan_advisory["blocks_generation"])
                self.assertTrue(plan_advisory["issue"])

    def test_planner_transport_failure_does_not_block(self):
        responses = self.standard(B)
        responses["planner"] = [RuntimeError("MOCK HTTP failure after bounded retry")]
        records, history, caller = self.collect(B, responses)
        self.assert_pending(records, history)
        self.assertIsNone(caller.observed[1]["context"]["advisory_plan"])

    def test_short_dialogue_and_auxiliary_omissions_only_warn(self):
        scene = model_scene(B, length=2)
        del scene["feedback_topic_ids"]
        del scene["no_scene_reason"]
        records, history, _ = self.collect(B, self.standard(B, scene=scene))
        self.assert_pending(records, history, generated_turns=2)
        self.assertIn("turn_count_outside_suggested_12_16", records[0]["quality_warnings"])
        self.assertIn("auxiliary_field_missing:feedback_topic_ids", records[0]["quality_warnings"])

    def test_reviewer_missing_fields_keeps_candidate_and_avoids_repair_loop(self):
        review = {**c.identity(ROOT, B, "S01"), "verdict": "acceptable", "core_checks": {}}
        records, history, caller = self.collect(B, self.standard(B, review=review))
        self.assert_pending(records, history)
        self.assertFalse(records[0]["decision"]["audit_complete"])
        self.assertEqual(caller.role_counts["reviewer"], 1)
        self.assertEqual(caller.role_counts["realizer"], 1)

    def test_reviewer_transport_failure_keeps_saved_dialogue(self):
        records, history, caller = self.collect(B, self.standard(B, review=RuntimeError("MOCK reviewer timeout")))
        self.assert_pending(records, history)
        self.assertEqual(records[0]["review_error"], "MOCK reviewer timeout")
        self.assertIsNone(records[0]["review_request"])
        self.assertEqual(caller.role_counts["realizer"], 1)
        generated = c.r.base.read(self.run / "candidates" / ROOT / "B/S01/GENERATED_1.json")
        self.assertEqual(generated["realization"], model_scene(B))

    def test_explicit_core_problem_quarantines_without_appending_or_retry(self):
        review = model_review(B)
        review["verdict"] = "problem"
        review["core_checks"]["actor_actions"]["status"] = "problem"
        records, history, caller = self.collect(B, self.standard(B, review=review))
        self.assertEqual(records[0]["disposition"], "quarantined")
        self.assertEqual(history, BASELINE)
        self.assertEqual(caller.role_counts["realizer"], 1)
        self.assertEqual(records[0]["decision"]["raw_review_verdict"], "problem")
        self.assertFalse(records[0]["decision"]["semantic_certified"])
        self.assertTrue((self.run / "candidates" / ROOT / "B/S01/GENERATED_1.json").exists())

    def test_invalid_speaker_gets_one_mechanical_repair(self):
        invalid = model_scene(B)
        invalid["turns"][0]["speaker"] = "MOCK-unknown-speaker"
        responses = self.standard(B)
        responses["realizer"] = [invalid, model_scene(B)]
        records, history, caller = self.collect(B, responses)
        self.assertEqual([x["disposition"] for x in records], ["hard_reject", "retained_pending_review"])
        self.assertEqual(len(history), len(BASELINE) + 12)
        self.assertEqual(caller.role_counts["reviewer"], 1)
        second = [x for x in caller.observed if x["role"] == "realizer"][1]
        self.assertIn("dialogue_speaker_invalid", second["context"]["mechanical_repair"]["failures"])
        self.assertEqual(records[0]["realization"], invalid)

    def test_duplicate_copy_gets_mechanical_repair_and_no_early_review(self):
        copied = model_scene(A)
        copied["turns"] = [{"speaker": x["speaker"], "text": x["text"]} for x in BASELINE]
        responses = self.standard(A)
        responses["realizer"] = [copied, model_scene(A)]
        records, history, caller = self.collect(A, responses)
        self.assertEqual([x["disposition"] for x in records], ["hard_reject", "retained_pending_review"])
        self.assertEqual(records[0]["duplicate_result"]["status"], "REJECT")
        self.assertEqual(len(history), len(BASELINE) + 12)
        self.assertEqual([x["role"] for x in caller.observed], ["planner", "realizer", "realizer", "reviewer"])
        self.assertTrue(caller.observed[2]["context"]["mechanical_repair"]["duplicate_reasons"])

    def test_full_candidate_history_reaches_next_slot_unchanged(self):
        first, history, _ = self.collect(A, self.standard(A))
        records, cumulative, caller = self.collect(B, self.standard(B, scene_id="S02"),
                                                  scene_id="S02", history=history)
        self.assertEqual(cumulative[:len(history)], history)
        self.assertEqual(len(cumulative), len(BASELINE) + 24)
        for request in caller.observed:
            self.assertEqual(request["context"]["cumulative_history"], history)
            self.assertEqual(request["context"]["formal_baseline_turns"], len(BASELINE))
            self.assertEqual(request["proof"]["history_sha256"], c.r.base.digest(history))
        self.assertTrue(any(x["turn_ref"].startswith("CANDIDATE:A:") for x in history))
        self.assertFalse(first[0]["training_pool_eligible"])
        self.assertFalse(records[0]["training_pool_eligible"])

    def test_raw_response_is_identical_in_transport_generated_and_record_artifacts(self):
        scene = model_scene(B)
        raw = json.dumps(scene, ensure_ascii=False, indent=4) + "\n"
        review_raw = json.dumps(model_review(B), ensure_ascii=False, indent=3) + "\n"
        records, _, caller = self.collect(B, self.standard(B, scene=raw, review=review_raw))
        row = records[0]
        transport = c.r.base.read(Path(row["request"]) / "RESPONSE.json")
        generated = c.r.base.read(self.run / "candidates" / ROOT / "B/S01/GENERATED_1.json")
        record = c.r.base.read(self.run / "candidates" / ROOT / "B/S01/RECORD_1.json")
        self.assertEqual(transport["text"], raw)
        self.assertEqual(generated["raw"], raw)
        self.assertEqual(record["realization"], scene)
        self.assertEqual(record["raw_review"], review_raw)
        self.assertEqual(c.r.base.read(Path(row["review_request"]) / "RESPONSE.json")["text"], review_raw)

    def test_wrong_scene_review_cannot_quarantine_this_scene(self):
        review = model_review(B, "S99")
        review["verdict"] = "problem"
        review["core_checks"]["actor_actions"]["status"] = "problem"
        records, history, _ = self.collect(B, self.standard(B, review=review))
        self.assert_pending(records, history)
        self.assertTrue(any(x["code"] == "review_identity_mismatch"
                            for x in records[0]["decision"]["audit_issues"]))

    def test_132_turn_baseline_is_never_truncated_in_any_stage(self):
        baseline = [{"turn_ref": f"MOCK:FULL:T{index:03d}",
                     "speaker": "Emily" if index % 2 else "James",
                     "text": f"MOCK ONLY full history row {index}: preserved beginning, middle, and ending."}
                    for index in range(1, 133)]
        records, history, caller = self.collect(B, self.standard(B), baseline=baseline)
        self.assertEqual(history[:132], baseline)
        self.assertEqual(len(history), 144)
        for request in caller.observed:
            self.assertEqual(request["context"]["cumulative_history"], baseline)
            self.assertEqual(request["proof"]["history_turns"], 132)
            self.assertEqual(request["proof"]["formal_baseline_sha256"], c.r.base.digest(baseline))

    def test_repeated_wrong_source_binding_is_hard_rejected_and_not_reviewed(self):
        scene = model_scene(B)
        scene["dataset_card_id"] = "WRONG-CARD"
        responses = self.standard(B)
        responses["realizer"] = [scene, deepcopy(scene)]
        responses["reviewer"] = []
        records, history, caller = self.collect(B, responses)
        self.assertEqual([x["disposition"] for x in records], ["hard_reject", "hard_reject"])
        self.assertEqual(history, BASELINE)
        self.assertEqual(caller.role_counts["reviewer"], 0)
        self.assertTrue(all("dialogue_card_mismatch" in row["mechanical_failures"] for row in records))

    def test_no_scene_stops_without_review_or_inventing_dialogue(self):
        scene = {**c.identity(ROOT, B, "S01"), "status": "no_scene", "turns": [],
                 "no_scene_reason": "MOCK ONLY no coherent test scene."}
        responses = self.standard(B, scene=scene)
        responses["reviewer"] = []
        records, history, caller = self.collect(B, responses)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["disposition"], "hard_reject")
        self.assertEqual(history, BASELINE)
        self.assertEqual(caller.role_counts["realizer"], 1)
        self.assertEqual(caller.role_counts["reviewer"], 0)

    def test_audit_replays_real_raw_responses_after_mechanical_repair(self):
        invalid = model_scene(A)
        invalid["turns"][0]["speaker"] = "MOCK-unknown-speaker"
        responses = self.standard(A, review={**c.identity(ROOT, A, "S01"), "core_checks": {}})
        responses["realizer"] = [invalid, model_scene(A)]
        records, history, caller = self.collect(A, responses)
        counter = lambda text: len(text.split())
        manifest = {"routes": [A], "scenes_per_route": 1, "logical_max": 4, "http_max": 8}
        allocation = {"assignments": {ROOT: []}, "assigned_slots": 0}
        c.r.base.write(self.run / "SOURCE_RESERVATIONS.json", {
            "reservations": c.r.source_inventory._assignment_rows(self.run, allocation["assignments"])})
        c.r.base.write(self.run / "CANDIDATE_RECORDS.json", records)
        c.r.base.write(self.run / "FORMAL_BASELINE.json", {"turns": BASELINE})
        c.r.base.write(self.run / "CANDIDATE_HISTORY.json", {"turns": history})
        summary = c.metrics.summarize_records(records, count_text_tokens=counter, baseline_turns=BASELINE)
        c.r.base.write(self.run / "CANDIDATE_REPORT.json", {"metrics": summary,
            "logical_calls": len(caller.observed), "http_attempts": len(caller.observed)})
        c.r.base.write(self.run / "COLLECTION_BINDINGS.json", {
            "files": c.r.base.bindings([p for p in self.run.rglob("*") if p.is_file()])})
        # Preparation is tested separately against frozen A3. This test isolates
        # the real collection replay on a temporary mock-only baseline.
        with patch.object(c, "verify", return_value=(manifest, allocation)), patch.object(c, "ROOT", ROOT), \
                patch.object(c, "token_counter", return_value=counter):
            result = c.audit(self.run)
        self.assertEqual(result["status"], "ARTIFACT_REPLAY_VERIFIED")
        self.assertFalse(result["semantic_certified"])
        self.assertFalse(result["training_pool_eligible"])
        self.assertEqual(result["metrics"], summary)


if __name__ == "__main__":
    unittest.main()
