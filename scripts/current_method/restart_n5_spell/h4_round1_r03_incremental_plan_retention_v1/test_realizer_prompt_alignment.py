"""Successful-route writing rules, input boundaries, and final HTTP messages."""
from copy import deepcopy
import json
import unittest

from .capacity import local_counter
from .prompts import TARGET_RULES, render_realizer
from .test_root_entry import RootCliHarness
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.prompts import (
    REALIZER_PARTICIPANT_RULES, REALIZER_TEXT_FREE_STYLE_RULES, realizer_arc_turn_rules,
)


class RealizerPromptAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.parent = {"root_id": "MOCK-root", "dialogue": {"turns": [
            {"turn_id": "P:T1", "speaker": "A", "text": "Private old opening. A second sentence."},
            {"turn_id": "P:T2", "speaker": "B", "text": "Private old response?"},
        ]}}
        self.plan = {"scenes": [{"phase_id": "initial:MOCK", "participants": ["A", "B"],
            "goal": "Complete the project", "before_state": "Unverified planning hypothesis",
            "required_transition": "Arrange a review", "after_state": "Project completed",
            "actor_bindings": [{"actor": "A", "binding": "Owns the project"}],
            "interaction_beats": ["A proposes a time", "B checks availability", "A records the arrangement"]}]}

    def test_shared_writing_rules_keep_actual_progress_policy(self):
        for mode in ("text_free", "full_parent"):
            with self.subTest(mode=mode):
                prompt = render_realizer(parent=self.parent, plan=self.plan, mode=mode)["prompt"]
                self.assertIn(REALIZER_PARTICIPANT_RULES, prompt)
                self.assertIn(realizer_arc_turn_rules(actual_progress_only=True), prompt)
                self.assertIn(TARGET_RULES, prompt)
                self.assertIn("Each required participant must speak", prompt)
                self.assertIn("8-turn scene is invalid", prompt)
                self.assertIn("do not\nadd another speaker merely to satisfy a count", prompt)
                self.assertIn("what actually happens\nin earlier output scenes", prompt)
                self.assertIn("Do not add output fields", prompt)
                for incompatible in ("REQUIRED END STATE must already be true",
                                     "as already realized Parent history",
                                     "ACTOR BINDINGS are frozen facts",
                                     "The frozen scene plans are the only authority for what must happen"):
                    self.assertNotIn(incompatible, prompt)

    def test_statistics_are_derived_locally_without_copying_historical_text(self):
        original = deepcopy((self.parent, self.plan))
        prompt = render_realizer(parent=self.parent, plan=self.plan)["prompt"]
        data = json.loads(prompt.rsplit("\n\nINPUT DATA\n", 1)[1])
        self.assertEqual(data["planned_phases_not_facts"], self.plan["scenes"])
        self.assertIn(REALIZER_TEXT_FREE_STYLE_RULES, prompt)
        tail = data["text_free_continuity_tail"]
        self.assertEqual(tail["turn_count"], 2)
        self.assertEqual(tail["last_speaker"], "B")
        self.assertEqual([row["speaker"] for row in tail["turn_shapes"]], ["A", "B"])
        self.assertEqual(tail["turn_shapes"][1]["ending"], "question")
        self.assertEqual(tail["turn_shapes"][0]["sentence_count"], 2)
        self.assertEqual(data["text_free_style_profile"]["speaker_turn_pattern"], "alternating")
        self.assertEqual(data["text_free_style_profile"]["source_turn_count"], 2)
        for row in self.parent["dialogue"]["turns"]:
            self.assertNotIn(row["text"], prompt)
        self.assertNotIn("authoritative_history", data)
        for row in tail["turn_shapes"]:
            self.assertEqual(set(row), {"turn_ref", "speaker", "character_count", "word_count", "sentence_count", "ending"})
        self.assertEqual((self.parent, self.plan), original)
        with self.assertRaisesRegex(ValueError, "text_free_shape_fields"):
            render_realizer(parent=self.parent, plan=self.plan,
                            continuity_shape={"turn_shapes": [{"text": "must not leak"}]})
        full = render_realizer(parent=self.parent, plan=self.plan, mode="full_parent")["prompt"]
        full_data = json.loads(full.rsplit("\n\nINPUT DATA\n", 1)[1])
        self.assertEqual(full_data["authoritative_history"], self.parent["dialogue"]["turns"])
        self.assertNotIn("text_free_style_profile", full_data)

    def test_formal_cli_counts_and_sends_aligned_messages_with_retained_prefix(self):
        with RootCliHarness() as harness:
            result = harness.invoke(case="no_supplement")
            realizer = [row for row in harness.requests if row["role"] == "realizer"]
            self.assertGreaterEqual(len(realizer), 2)
            counter = local_counter(harness.model["tokenizer_path"], harness.model["tokenizer_json_sha256"])
            for index, request in enumerate(realizer):
                prompt = request["messages"][0]["content"]
                self.assertEqual(request["prompt_version"], "incremental_retention_v1.realizer.successful_prompt_alignment_v3.text_free")
                self.assertIn(realizer_arc_turn_rules(actual_progress_only=True), prompt)
                self.assertIn(REALIZER_TEXT_FREE_STYLE_RULES, prompt)
                data = json.loads(prompt.rsplit("\n\nINPUT DATA\n", 1)[1])
                self.assertLessEqual(len(data["planned_phases_not_facts"]), 2)
                self.assertEqual(bool(data["retained_actual_prefix"]), index > 0)
                if index > 0:
                    self.assertTrue(data["text_free_continuity_tail"]["turn_shapes"][-1]["turn_ref"].startswith("R03:"))
                self.assertEqual(request["capacity"]["known_final_messages_tokens"], counter.count_tokens(request["messages"]))
                self.assertTrue(request["capacity"]["fits"])
                self.assertTrue(any(payload["messages"] == request["messages"] for payload in harness.http_payloads))
            self.assertGreater(result["retained_scene_count"], 0)
            self.assertTrue(any(status == "partial" for status in result["goal_statuses"].values()))


if __name__ == "__main__":
    unittest.main()
