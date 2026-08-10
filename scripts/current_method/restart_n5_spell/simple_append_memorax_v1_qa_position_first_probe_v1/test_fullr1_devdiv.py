from __future__ import annotations

import json
import unittest
from pathlib import Path

from .context import (
    RAW_DIALOGUE_CONTEXT_BUDGET_TOKENS,
    build_fresh_r1_context,
    select_fresh_r1_qa_context,
)
from .memory_context import compile_memory_overview, normalize_bucket_snapshot
from .payloads import MEMORAX_USER_NAMESPACE, build_bucket_search_payload
from .prompts import build_memory_augmented_planner_prompt
from .workflow import execute_single_round


PARENT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-inputs/"
    "conversation_chronicles_v1/roots/cc-train-episode-146700/"
    "unified_root.json")
EXPANDED = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "simple_append_memorax_v1_batch_remaining8_pilot_8/roots/"
    "cc-train-episode-146700/expansion/expanded_dialogue.json")
HISTORICAL_EXPANSION = EXPANDED.parent


def _tokens(text: str) -> int:
    return max(0, (len(text.encode("utf-8")) + 3) // 4)


def _empty_memory(root_id: str) -> dict:
    user_id = build_bucket_search_payload(root_id=root_id)["payload"]["user_id"]
    snapshot = normalize_bucket_snapshot({
        "success": True,
        "data": {"buckets": {
            "person": {}, "location": {}, "event": {},
            "item": {}, "dreaming": {},
        }},
    }, root_id=root_id, user_id=user_id)
    return compile_memory_overview(snapshot)


class _EmptyMemorax:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def add(self, payload, *, request_id):
        self.calls.append({"role": "add", "request_id": request_id})
        return {"success": True, "data": {"status": "completed"}}

    def search_buckets(self, payload, *, request_id):
        self.calls.append({"role": "search", "request_id": request_id})
        return {"success": True, "data": {"buckets": {
            "person": {}, "location": {}, "event": {},
            "item": {}, "dreaming": {},
        }}}

    def add_status(self, task_id, *, request_id):  # pragma: no cover
        raise AssertionError("synchronous fixture must not poll")


class FullR1DevelopmentDiversityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.parent = json.loads(PARENT.read_text(encoding="utf-8"))
        cls.expanded = json.loads(EXPANDED.read_text(encoding="utf-8"))

    def test_parent_under_budget_is_complete(self) -> None:
        context = build_fresh_r1_context(
            self.parent, token_counter=_tokens,
            budget_tokens=RAW_DIALOGUE_CONTEXT_BUDGET_TOKENS)
        self.assertTrue(context["full_parent_included"])
        self.assertEqual(
            context["selected_session_count"],
            self.parent["dialogue"]["session_count"])
        self.assertEqual(
            context["selected_turn_count"],
            self.parent["dialogue"]["turn_count"])

    def test_parent_over_budget_falls_back_without_splitting_session(self) -> None:
        context = build_fresh_r1_context(
            self.parent, token_counter=_tokens, budget_tokens=500)
        self.assertFalse(context["full_parent_included"])
        selected = set(context["selected_session_ids"])
        expected = [
            row for row in self.parent["dialogue"]["turns"]
            if row["session_id"] in selected
        ]
        self.assertEqual(expected, context["selected_turns"])
        self.assertEqual(
            context["selected_turns"][-1]["turn_id"],
            self.parent["dialogue"]["turns"][-1]["turn_id"])

    def test_qa_under_budget_uses_complete_accumulated_dialogue(self) -> None:
        view = select_fresh_r1_qa_context(
            self.expanded, token_counter=_tokens, budget_tokens=10_000)
        selection = view["qa_context_selection"]
        self.assertTrue(selection["full_parent_child_included"])
        self.assertEqual(view["turns"], self.expanded["turns"])

    def test_qa_over_budget_keeps_complete_sessions(self) -> None:
        view = select_fresh_r1_qa_context(
            self.expanded, token_counter=_tokens, budget_tokens=500)
        selection = view["qa_context_selection"]
        self.assertFalse(selection["full_parent_child_included"])
        selected_ids = set(selection["selected_session_ids"])
        original = self.expanded["turns"]
        expected = [
            row for row in original
            if str(row.get("session_ref") or row.get("session_id")) in selected_ids
        ]
        self.assertEqual(expected, view["turns"])

    def test_planner_adds_only_the_diversity_contract_to_schema(self) -> None:
        context = build_fresh_r1_context(
            self.parent, token_counter=_tokens)
        prompt = build_memory_augmented_planner_prompt(
            context, _empty_memory(self.parent["root_id"]))["prompt"]
        self.assertIn(
            "at least two scenes must introduce materially different concrete developments",
            prompt)
        self.assertIn('"planner_schema_version":"simple_append_planner_v1"', prompt)
        self.assertIn('"new_development"', prompt)
        self.assertNotIn('"action_id"', prompt)

    def test_experiment_uses_isolated_memorax_namespace(self) -> None:
        root_id = self.parent["root_id"]
        user_id = build_bucket_search_payload(
            root_id=root_id)["payload"]["user_id"]
        self.assertEqual(
            user_id, f"{MEMORAX_USER_NAMESPACE}:{root_id}")
        self.assertNotEqual(user_id, f"gxy-selfplay:{root_id}")

    def test_complete_shadow_round_passes_full_context_to_both_roles(self) -> None:
        planner_output = json.loads(
            (HISTORICAL_EXPANSION / "planner/output.json").read_text(
                encoding="utf-8"))
        realizer_output = json.loads(
            (HISTORICAL_EXPANSION / "realizer/output.json").read_text(
                encoding="utf-8"))
        seen: dict[str, dict] = {}

        def planner_call(prompt):
            seen["planner"] = dict(prompt)
            return planner_output

        def realizer_call(prompt):
            seen["realizer"] = dict(prompt)
            return realizer_output

        result = execute_single_round(
            parent=self.parent,
            memorax=_EmptyMemorax(),
            planner_call=planner_call,
            realizer_call=realizer_call,
            async_add=False,
            wait_for_memory_completion=True,
            raw_dialogue_token_counter=_tokens,
        )
        self.assertTrue(result["generation_ready"])
        context = result["selected_parent_context"]
        self.assertTrue(context["full_parent_included"])
        first_turn = self.parent["dialogue"]["turns"][0]["turn_id"]
        last_turn = self.parent["dialogue"]["turns"][-1]["turn_id"]
        self.assertIn(first_turn, seen["planner"]["prompt"])
        self.assertIn(last_turn, seen["planner"]["prompt"])
        self.assertIn(first_turn, seen["realizer"]["prompt"])
        self.assertIn(last_turn, seen["realizer"]["prompt"])
        self.assertTrue(
            result["qa_context_view"]["qa_context_selection"]
            ["full_parent_child_included"])


if __name__ == "__main__":
    unittest.main()
