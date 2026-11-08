from copy import deepcopy
import hashlib
import json
import unittest

import grounded_task as g


class GroundedTaskTests(unittest.TestCase):
    def setUp(self):
        self.history = [
            {"speaker": "Noor", "text": "The cabinet will be repaired tomorrow.", "turn_ref": "P:T001"},
            {"speaker": "Wei", "text": "I can bring the tools.", "turn_ref": "P:T002"},
        ]
        self.card = {"card_id": "new_card", "source_id": "generic:42", "actors": [
            {"actor_id": "seeker", "role": "seeker"},
            {"actor_id": "helper", "role": "helper"}],
            "steps": [{"actor_id": "seeker", "action": "ask where to borrow a book",
                       "state": "requested", "evidence": [{"quote": "Where may I borrow the book?",
                       "unit_id": "turn:0", "source_speaker": "X", "char_start": 0, "char_end": 28}]}],
            "lineage": {"source_item_sha256": "source-binding"}}
        self.payload = {"model": "test-model", "messages": [{"role": "system", "content": "system"},
            {"role": "user", "content": "FULL_HISTORY_JSON\n" + json.dumps(self.history) +
             "\n[/PREVIOUS_TRAINING_ROUND_EXPANSION_FEEDBACK]"}]}
        self.proof = {"feedback_receipt": {"receipt": "frozen"}, "training_pool_eligible": False}

    def append(self, **kwargs):
        fields = dict(root="root-generic", route="dataset_new_event", scene_id="S17",
                      role="realizer", history=self.history, card=self.card)
        fields.update(kwargs)
        return g.append_current_task(self.payload, self.proof, **fields)

    def target(self, payload):
        content = payload["messages"][-1]["content"]
        return json.JSONDecoder().raw_decode(content.split(g.TARGET_MARKER, 1)[1])[0]

    def test_complete_original_evidence_follows_feedback_and_preserves_prior_messages(self):
        result, proof = self.append()
        content = result["messages"][-1]["content"]
        self.assertTrue(content.startswith(self.payload["messages"][-1]["content"]))
        self.assertLess(content.index("[/PREVIOUS_TRAINING_ROUND_EXPANSION_FEEDBACK]"), content.index(g.TAIL_MARKER))
        self.assertEqual(result["messages"][0], self.payload["messages"][0])
        target = self.target(result)
        self.assertEqual(target["source_card_with_original_evidence"], self.card)
        self.assertEqual(target["allowed_speakers"], ["Noor", "Wei"])
        self.assertEqual((target["root_id"], target["scene_id"], target["dataset_card_id"]),
                         ("root-generic", "S17", "new_card"))
        self.assertEqual(proof["feedback_receipt"], self.proof["feedback_receipt"])
        self.assertFalse(proof["training_pool_eligible"])

    def test_no_inputs_mutated_or_aliased(self):
        original = deepcopy((self.payload, self.proof, self.history, self.card))
        result, proof = self.append()
        result["messages"][0]["content"] = "changed"
        proof["feedback_receipt"]["receipt"] = "changed"
        self.assertEqual((self.payload, self.proof, self.history, self.card), original)

    def test_proof_binds_intermediate_final_tail_and_full_card(self):
        result, proof = self.append()
        canonical = lambda value: hashlib.sha256(json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        tail = result["messages"][-1]["content"][len(self.payload["messages"][-1]["content"]):]
        self.assertEqual(proof["feedback_receipt_scope"], "intermediate_before_tail")
        self.assertEqual(proof["pre_tail_payload_sha256"], canonical(self.payload))
        self.assertEqual(proof["final_payload_sha256"], canonical(result))
        self.assertEqual(proof["raw_card_sha256"], canonical(self.card))
        self.assertEqual(proof["source_card_with_evidence_sha256"], canonical(self.card))
        self.assertEqual(proof["final_task_sha256"], canonical(tail))
        self.assertEqual(proof["current_task_history_sha256"], canonical(self.history))

    def test_plan_mapping_is_advisory_and_absent_mapping_does_not_invent_roles(self):
        mapping = [{"card_actor_id": "seeker", "root_actor": "Wei"},
                   {"card_actor_id": "helper", "root_actor": "Noor"}]
        plan = {"actor_mapping": mapping, "goal": "Locate a book"}
        result, _ = self.append(plan=plan)
        self.assertEqual(self.target(result)["draft_actor_mapping"], mapping)
        self.assertEqual(self.target(result)["advisory_plan"], plan)
        absent, _ = self.append()
        self.assertIsNone(self.target(absent)["draft_actor_mapping"])
        self.assertIn("do not add mapping fields", absent["messages"][-1]["content"])
        self.assertIn("not evidence that events occurred", result["messages"][-1]["content"])

    def test_all_three_roles_keep_target_but_request_different_outputs(self):
        endings = {"planner": "Generate only the advisory plan now.",
                   "realizer": "Generate only the new scene now.",
                   "reviewer": "Generate only the advisory review now."}
        realization = {"root_id": "root-generic", "scene_id": "S17", "turns": []}
        for role, ending in endings.items():
            with self.subTest(role=role):
                result, proof = self.append(role=role, realization=realization)
                self.assertTrue(result["messages"][-1]["content"].endswith(ending + "\n"))
                self.assertEqual(self.target(result)["current_task_role"], role)
                self.assertEqual(self.target(result)["new_scene_to_diagnose"],
                                 realization if role == "reviewer" else None)
                self.assertEqual(proof["current_task_role"], role)

    def test_a_route_uses_history_without_fabricated_source(self):
        for role in g.ROLES:
            result, _ = self.append(route="old_event_continuation", card=None, role=role)
            self.assertIsNone(self.target(result)["source_card_with_original_evidence"])
            self.assertIsNone(self.target(result)["dataset_card_id"])
            self.assertIn("Continue an event grounded in the actual history", result["messages"][-1]["content"])

    def test_large_history_is_not_truncated(self):
        history = [{"speaker": "Wei", "turn_ref": f"P:T{i:04}", "text": f"Turn {i}: " + "正文" * 300}
                   for i in range(500)]
        self.payload["messages"][-1]["content"] = "HISTORY\n" + json.dumps(history, ensure_ascii=False)
        prefix = self.payload["messages"][-1]["content"]
        result, proof = self.append(history=history)
        self.assertEqual(result["messages"][-1]["content"][:len(prefix)], prefix)
        self.assertEqual(proof["current_task_history_turns"], 500)
        self.assertEqual(proof["current_task_history_sha256"], g.digest(history))

    def test_missing_card_and_invalid_role_fail_before_creating_request(self):
        for kwargs, message in [({"card": None}, "current_task_B_card_missing"),
                                ({"role": "unknown"}, "current_task_role_invalid")]:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                self.append(**kwargs)

    def test_tail_cannot_precede_a_later_assistant_message(self):
        self.payload["messages"].append({"role": "assistant", "content": "old answer"})
        with self.assertRaisesRegex(ValueError, "current_task_last_message_not_user"):
            self.append()


if __name__ == "__main__":
    unittest.main()
