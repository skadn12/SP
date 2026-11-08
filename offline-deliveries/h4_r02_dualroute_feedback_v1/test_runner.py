from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import feedback
import runner


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.run_dir = Path(cls.tmp.name)
        feedback.compile_feedback(cls.run_dir / "feedback", runner.ROOTS)
        for root, (shard, _) in runner.SOURCE_SPECS.items():
            source = runner.PROJECT / f"restart-n5-runs/h4_unexpanded122_memory_event_k3_v1/shards/shard-{shard}/expansion/roots/{root}/promotion/promotable_expanded_dialogue.json"
            runner.write(cls.run_dir / "sources" / f"{root}.json", runner.read(source))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_four_route_roles_have_feedback_and_entire_history(self):
        root = runner.ROOTS[0]
        history = runner.read(self.run_dir / "sources" / f"{root}.json")["turns"]
        for route in runner.ROUTES:
            for role in ("planner", "realizer"):
                payload, proof = runner.make_request(self.run_dir, root, route, role, history)
                receipt = proof["feedback_receipt"]
                self.assertEqual((receipt["route"], receipt["role"]), (route, role))
                self.assertTrue(receipt["feedback_appended"])
                prompt = payload["messages"][0]["content"]
                body = prompt.split("\nINPUT_JSON\n")[1].split("\n[PREVIOUS_TRAINING_ROUND_EXPANSION_FEEDBACK]")[0]
                self.assertEqual(json.loads(body)["cumulative_history"], history)
                self.assertEqual(proof["history_sha256"], runner.digest(history))

    def test_prefix_changes_and_wrong_route_role_are_rejected(self):
        root = runner.ROOTS[0]
        history = runner.read(self.run_dir / "sources" / f"{root}.json")["turns"]
        changed = deepcopy(history)
        changed[0]["text"] = "rewritten history"
        for route, role, turns in [("wrong", "planner", history), (runner.ROUTES[0], "wrong", history),
                                   (runner.ROUTES[0], "planner", changed)]:
            with self.assertRaises(ValueError):
                runner.make_request(self.run_dir, root, route, role, turns)

    def test_accepted_A_is_in_B_request_and_stale_B_is_rejected(self):
        root = runner.ROOTS[0]
        history = runner.read(self.run_dir / "sources" / f"{root}.json")["turns"]
        realization = {"scene_id": "S01", "turns": [{"speaker": "Emily", "text": "MOCK_TEST_NEW_EVENT"}]}
        after_A = runner.append_scene(history, root, runner.ROUTES[0], realization)
        runner.verify_handoff(after_A, after_A)
        with self.assertRaises(ValueError):
            runner.verify_handoff(after_A, history)
        payload, proof = runner.make_request(self.run_dir, root, runner.ROUTES[1], "planner", after_A)
        self.assertIn("MOCK_TEST_NEW_EVENT", payload["messages"][0]["content"])
        self.assertEqual(proof["history_turns"], len(history) + 1)

    def test_fake_quote_rejected_and_no_scene_not_fabricated(self):
        root = runner.ROOTS[0]
        history = runner.read(self.run_dir / "sources" / f"{root}.json")["turns"]
        actors = sorted({t["speaker"] for t in history})[:2]
        scene = {"scene_id": "S01", "actors": actors, "source_anchors": [{"turn_ref": history[0]["turn_ref"], "quote": "not present"}],
                 "goal": "goal", "developments": ["one", "two"], "dataset_card_id": None, "feedback_topic_ids": []}
        plan = {"root_id": root, "route": runner.ROUTES[0], "scenes": [scene], "shortfall_reason": "shortfall"}
        with self.assertRaisesRegex(ValueError, "quote_not_exact"):
            runner.validate_plan(plan, root, runner.ROUTES[0], history, [])
        scene["source_anchors"][0]["quote"] = history[0]["text"]
        runner.validate_plan(plan, root, runner.ROUTES[0], history, [])
        scene["feedback_topic_ids"] = [0, 1]
        runner.validate_plan(plan, root, runner.ROUTES[0], history, [])
        no_scene = {"root_id": root, "route": runner.ROUTES[0], "scene_id": "S01", "status": "no_scene",
                    "turns": [], "feedback_topic_ids": [], "no_scene_reason": "cannot continue"}
        self.assertEqual(runner.validate_realization(no_scene, root, runner.ROUTES[0], scene)["turns"], [])

    def test_http_callback_matches_payload_and_raw_text_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            caller = runner.Caller.__new__(runner.Caller)
            caller.run = Path(tmp)
            caller.profile, caller.tokenizer = {}, None
            caller.counter = 0
            caller.http_counter = 0
            caller.reuse = {}
            import threading
            caller.lock = threading.Lock()
            caller.annotate = lambda value: value
            caller.capacity = lambda *args: {"status": "PASS"}
            payload = {"messages": [{"role": "user", "content": "mock"}], "model": runner.dataset_cards.MODEL}
            class Client:
                def __init__(self, profile, on_http_attempt):
                    self.callback = on_http_attempt
                def complete(self, **kwargs):
                    self.callback(payload)
                    from types import SimpleNamespace
                    return SimpleNamespace(raw_text='{"ok":true}', metadata={})
            caller.client_factory = Client
            raw, path = caller.call("mock", payload, {})
            self.assertEqual(raw, '{"ok":true}')
            self.assertTrue(runner.read(path / "HTTP_ATTEMPT.json")["actual_payload_equals_frozen"])
            self.assertEqual(runner.read(path / "RESPONSE.json")["text"], raw)
            caller.http_counter = runner.HTTP_MAX
            with self.assertRaisesRegex(ValueError, "budget_exhausted"):
                caller.call("overflow", payload, {})


if __name__ == "__main__":
    unittest.main()
