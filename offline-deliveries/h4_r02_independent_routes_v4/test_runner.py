"""Offline integration tests for the independent-route runner.

These tests exercise ``run_root`` with a deterministic caller.  They never
construct a provider client or make an HTTP/API request.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import runner


ROOT = "test-root"
ROUTE_A = runner.ROUTES[0]
ROUTE_B = runner.ROUTES[1]
A_ONLY = "A_ONLY_REVIEW_SENTINEL"


def _frozen_turns():
    return [
        {"turn_ref": "P:T001", "speaker": "Alice", "text": "Frozen substantive context one with enough detail."},
        {"turn_ref": "P:T002", "speaker": "Bob", "text": "Frozen substantive context two with enough detail."},
        {"turn_ref": "P:T003", "speaker": "Alice", "text": "Frozen substantive context three with enough detail."},
    ]


def _card(card_id="card-1"):
    return {
        "card_id": card_id,
        "card_sha256": "card-sha",
        "guidance_level": "seed",
        "actors": [{"actor_id": "new-a", "role": "participant"},
                   {"actor_id": "new-b", "role": "participant"}],
        "steps": [{"step_id": "step-1", "actor_id": "new-a", "action": "decides", "state": "new"}],
        "dependencies": [], "confusion_axes": [], "open_ends": [], "downgrade_reason": None,
    }


class FakeCaller:
    """Return scripted planner, realization and reviewer responses."""

    def __init__(self, run, *, duplicate_once=False, review_reject_once=False,
                 malformed_review_once=False):
        self.run = Path(run)
        self.duplicate_once = duplicate_once
        self.review_reject_once = review_reject_once
        self.malformed_review_once = malformed_review_once
        self.calls = []
        self.realizer_calls = {"A": 0, "B": 0}
        self.review_calls = {"A": 0, "B": 0}

    @staticmethod
    def _route_letter(payload):
        body = json.loads(payload["messages"][0]["content"])
        route = body.get("route")
        return ("A" if route == ROUTE_A else "B"), body

    def call(self, label, payload, proof):
        letter, body = self._route_letter(payload)
        role = body["role"]
        self.calls.append({"label": label, "role": role, "letter": letter,
                           "body": body, "proof": proof})
        directory = self.run / "requests" / label
        directory.mkdir(parents=True, exist_ok=True)
        if role == "planner":
            scene_id = body["target_scene_id"]
            card_id = body["dataset_cards"][0]["card_id"] if letter == "B" else None
            frozen = body["cumulative_history"]
            plan = {
                "root_id": body["root_id"], "route": body["route"],
                "scenes": [{
                    "scene_id": scene_id, "actors": ["Alice", "Bob"],
                    "source_anchors": [{"turn_ref": frozen[0]["turn_ref"],
                                         "quote": frozen[0]["text"]}],
                    "goal": "Make a concrete new decision.",
                    "developments": ["Alice makes a concrete decision.",
                                     "Bob records the resulting next step."],
                    "dataset_card_id": card_id, "feedback_topic_ids": [],
                }],
                "shortfall_reason": "",
            }
            return json.dumps(plan), directory

        if role == "realizer":
            self.realizer_calls[letter] += 1
            scene = body["scene_plan"]
            if (self.duplicate_once and letter == "A"
                    and self.realizer_calls[letter] == 1):
                # Three contiguous substantive turns copied from frozen history
                # must be rejected by duplicate_check.
                turns = [{"speaker": turn["speaker"], "text": turn["text"]}
                         for turn in body["cumulative_history"]] + [
                    {"speaker": "Alice", "text": f"A unique replacement turn {i}."}
                    if i % 2 else {"speaker": "Bob", "text": f"A unique replacement turn {i}."}
                    for i in range(4, 13)
                ]
            else:
                prefix = f"{letter} accepted scene {self.realizer_calls[letter]}"
                if letter == "A":
                    prefix += f" {A_ONLY}"
                turns = [{"speaker": "Alice" if i % 2 else "Bob",
                          "text": f"{prefix}: substantive new turn {i}."}
                         for i in range(1, 13)]
            result = {
                "root_id": body["root_id"], "route": body["route"],
                "scene_id": scene["scene_id"], "status": "scene", "turns": turns,
                "feedback_topic_ids": [], "no_scene_reason": "",
            }
            return json.dumps(result), directory

        if role == "scene_reviewer":
            self.review_calls[letter] += 1
            realization = body["new_scene"]
            if (self.malformed_review_once and letter == "B"
                    and self.review_calls[letter] == 1):
                return "{malformed review", directory
            if (self.review_reject_once and letter == "A"
                    and self.review_calls[letter] == 1):
                return json.dumps({
                    "root_id": body["root_id"], "route": body["route"],
                    "scene_id": body["scene_id"],
                    "checks": [
                        {"development_index": 1, "realized": False, "evidence": [],
                         "reason": "The first development is absent."},
                        {"development_index": 2, "realized": False, "evidence": [],
                         "reason": "The second development is absent."},
                    ],
                    "new_event_or_progress": False,
                    "consistent_with_history": True,
                    "duplicates_existing_event": True,
                    "reason": "The scene repeats the old event.",
                }), directory
            checks = []
            for index in (1, 2):
                checks.append({"development_index": index, "realized": True,
                               "evidence": [{"turn_index": index,
                                             "quote": realization["turns"][index - 1]["text"]}],
                               "reason": "The new turn explicitly realizes this development."})
            return json.dumps({
                "root_id": body["root_id"], "route": body["route"],
                "scene_id": body["scene_id"], "checks": checks,
                "new_event_or_progress": True, "consistent_with_history": True,
                "duplicates_existing_event": False,
                "reason": "The scene adds explicit progress.",
            }), directory

        raise AssertionError(f"unexpected mock role: {role}")


class RunnerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.run = Path(self.temp.name) / "run"
        (self.run / "sources").mkdir(parents=True)
        (self.run / "feedback").mkdir(parents=True)
        (self.run / "sources" / f"{ROOT}.json").write_text(
            json.dumps({"turns": _frozen_turns()}), encoding="utf-8"
        )

    def tearDown(self):
        self.temp.cleanup()

    def run_one(self, caller):
        return runner.run_root(caller, self.run, ROOT, [_card()], target_scenes=1)

    def make_request_stub(self, records):
        def fake_make_request(run, root, route, role, history, cards, scene_id,
                              *, scene=None, repair=None):
            body = {"root_id": root, "route": route, "role": role,
                    "target_scene_id": scene_id, "cumulative_history": history,
                    "dataset_cards": cards, "scene_plan": scene,
                    "targeted_repair": repair}
            records.append(body)
            return ({"model": runner.MODEL,
                     "messages": [{"role": "user", "content": json.dumps(body)}]},
                    {"root_id": root, "route": route, "role": role,
                     "scene_id": scene_id})
        return fake_make_request

    @staticmethod
    def review_request_stub(records):
        def fake_review_request(root, route, scene, history, realization, other_turns):
            body = {"root_id": root, "route": route, "role": "scene_reviewer",
                    "scene_id": scene["scene_id"], "branch_history": history,
                    "scene_plan": scene, "new_scene": realization,
                    "other_branch_added_turns_for_review_only": other_turns}
            records.append(body)
            return ({"model": runner.MODEL,
                     "messages": [{"role": "user", "content": json.dumps(body)}]},
                    {"root_id": root, "route": route, "role": "scene_reviewer",
                     "scene_id": scene["scene_id"]})
        return fake_review_request

    def test_branches_share_only_frozen_base_and_reviewer_sees_other_branch(self):
        requests, reviews = [], []
        caller = FakeCaller(self.run)
        with mock.patch.object(runner, "make_request", self.make_request_stub(requests)), \
             mock.patch.object(runner, "review_request", self.review_request_stub(reviews)):
            results = self.run_one(caller)
        self.assertEqual([r["accepted_scenes"] for r in results], [1, 1])
        b_generators = [x for x in requests if x["route"] == ROUTE_B]
        self.assertTrue(b_generators)
        self.assertTrue(all(A_ONLY not in json.dumps(x, ensure_ascii=False) for x in b_generators))
        self.assertTrue(reviews)
        b_review = next(x for x in reviews if x["route"] == ROUTE_B)
        self.assertIn(A_ONLY, json.dumps(b_review, ensure_ascii=False))
        self.assertNotIn(A_ONLY, json.dumps(b_review["branch_history"], ensure_ascii=False))
        self.assertEqual(b_review["branch_history"], _frozen_turns())
        self.assertEqual(b_review["other_branch_added_turns_for_review_only"][0]["turn_ref"],
                         "R02:A:S01:T001")
        self.assertIn(A_ONLY, b_review["other_branch_added_turns_for_review_only"][0]["text"])

    def test_duplicate_rejection_gets_at_most_one_realizer_rewrite(self):
        requests, reviews = [], []
        caller = FakeCaller(self.run, duplicate_once=True)
        with mock.patch.object(runner, "make_request", self.make_request_stub(requests)), \
             mock.patch.object(runner, "review_request", self.review_request_stub(reviews)):
            results = self.run_one(caller)
        a_result = results[0]
        self.assertEqual(a_result["accepted_scenes"], 1)
        self.assertEqual(caller.realizer_calls["A"], 2)
        self.assertEqual(len(a_result["scene_metadata"]), 1)
        attempts = json.loads((self.run / "roots" / ROOT / "A" / "S01.ATTEMPTS.json").read_text())
        self.assertEqual(attempts[0]["stage"], "duplicate")
        self.assertEqual(attempts[-1]["status"], "PASS")

    def test_missing_card_preserves_later_reserved_slot_without_dropping_branch(self):
        requests, reviews = [], []
        caller = FakeCaller(self.run)
        with mock.patch.object(runner, "make_request", self.make_request_stub(requests)), \
             mock.patch.object(runner, "review_request", self.review_request_stub(reviews)):
            results = runner.run_root(caller, self.run, ROOT, [None, _card("second-slot-card")], target_scenes=2)
        b = results[1]
        self.assertEqual(b["accepted_scenes"], 1)
        self.assertEqual(b["scene_metadata"][0]["scene_id"], "S02")
        self.assertEqual(b["scene_metadata"][0]["dataset_card_id"], "second-slot-card")
        self.assertEqual(b["failures"], [{"scene_id": "S01", "reason": "reserved_source_card_extraction_failed"}])
        b_requests = [r for r in requests if r["route"] == ROUTE_B]
        self.assertTrue(b_requests)
        self.assertEqual({r["target_scene_id"] for r in b_requests}, {"S02"})

    def test_review_rejection_and_malformed_review_are_fail_closed_with_one_rewrite(self):
        requests, reviews = [], []
        caller = FakeCaller(self.run, review_reject_once=True)
        with mock.patch.object(runner, "make_request", self.make_request_stub(requests)), \
             mock.patch.object(runner, "review_request", self.review_request_stub(reviews)):
            results = self.run_one(caller)
        self.assertEqual(results[0]["accepted_scenes"], 1)
        self.assertEqual(caller.realizer_calls["A"], 2)
        self.assertEqual(caller.review_calls["A"], 2)
        attempts = json.loads((self.run / "roots" / ROOT / "A" / "S01.ATTEMPTS.json").read_text())
        self.assertEqual(attempts[0]["status"], "REJECT")
        self.assertEqual(attempts[-1]["status"], "PASS")

        self.tearDown()
        self.setUp()
        requests, reviews = [], []
        caller = FakeCaller(self.run, malformed_review_once=True)
        with mock.patch.object(runner, "make_request", self.make_request_stub(requests)), \
             mock.patch.object(runner, "review_request", self.review_request_stub(reviews)):
            results = self.run_one(caller)
        self.assertEqual(results[1]["accepted_scenes"], 1)
        self.assertEqual(caller.realizer_calls["B"], 2)
        self.assertEqual(caller.review_calls["B"], 2)
        b_attempts = json.loads((self.run / "roots" / ROOT / "B" / "S01.ATTEMPTS.json").read_text())
        self.assertIn("review_json_invalid", " ".join(b_attempts[0]["failures"]))
        self.assertEqual(b_attempts[-1]["status"], "PASS")

    def test_transport_error_is_not_replayed(self):
        manifest = self.run / "MANIFEST.json"
        manifest.write_text(json.dumps({"http_max": 4}), encoding="utf-8")
        calls = []

        class Dev:
            @staticmethod
            def tokenizer_from(_path):
                return object()

        def profile():
            return object()

        def capacity(_profile, _tokenizer, _messages, _max_tokens):
            return {"status": "PASS"}

        def annotate(value):
            return value

        class Client:
            def complete(self, **_kwargs):
                calls.append("complete")
                raise OSError("simulated transport failure")

        def factory(_profile, on_http_attempt=None):
            calls.append("factory")
            return Client()

        with mock.patch.object(runner.base, "transport",
                              return_value=(profile, factory, annotate, capacity, Dev)):
            caller = runner.Caller(self.run)
            with self.assertRaisesRegex(RuntimeError, "no_automatic_retry"):
                caller.call("transport-failure", {
                    "model": runner.MODEL,
                    "messages": [{"role": "user", "content": "offline"}],
                }, {"test": True})
        self.assertEqual(calls, ["factory", "complete"])
        error = json.loads((self.run / "requests" / "001-transport-failure" / "ERROR.json").read_text())
        self.assertTrue(error["no_automatic_retry"])


if __name__ == "__main__":
    unittest.main()
