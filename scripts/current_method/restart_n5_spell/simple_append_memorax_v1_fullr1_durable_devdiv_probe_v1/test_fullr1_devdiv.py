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
from .durable_contracts import (
    PLANNER_SCHEMA_VERSION,
    build_parent_anchor_registry,
    compile_round_plan,
    validate_planner_output,
)
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

    def _planner(self, scene_count: int = 3) -> dict:
        first_positions: list[int] = []
        seen: set[str] = set()
        for position, row in enumerate(self.parent["dialogue"]["turns"], 1):
            if row["session_id"] not in seen:
                seen.add(row["session_id"])
                first_positions.append(position)
        return {
            "planner_schema_version": PLANNER_SCHEMA_VERSION,
            "round_goal": "Advance distinct durable Parent developments.",
            "scenes": [{
                "goal": f"Advance root {index + 1}",
                "participants": ["Co-workers A", "Co-workers B"],
                "primary_anchor_span": {
                    "start_position": first_positions[index],
                    "end_position": first_positions[index],
                },
                "new_development": f"A distinct performed action {index + 1}",
                "interaction_beats": ["setup", "choice", "action"],
                "transition_from_previous": "A natural later transition.",
                "desired_outcome": f"Changed situation {index + 1}",
                "durable_outcome": f"A concrete arrangement {index + 1} is established.",
            } for index in range(scene_count)],
        }

    def _validate(self, value: dict) -> dict:
        context = build_fresh_r1_context(self.parent, token_counter=_tokens)
        return validate_planner_output(
            value,
            parent_speakers=context["parent_speakers"],
            parent_anchor_registry=build_parent_anchor_registry(
                self.parent["dialogue"]["turns"]),
        )

    def test_planner_adds_only_durable_diversity_fields(self) -> None:
        context = build_fresh_r1_context(
            self.parent, token_counter=_tokens)
        prompt = build_memory_augmented_planner_prompt(
            context, _empty_memory(self.parent["root_id"]),
            build_parent_anchor_registry(self.parent["dialogue"]["turns"]))[
                "prompt"]
        self.assertIn("one to six consecutive global_parent_position turns", prompt)
        self.assertIn(PLANNER_SCHEMA_VERSION, prompt)
        self.assertIn('"primary_anchor_span"', prompt)
        self.assertIn('"durable_outcome"', prompt)
        self.assertIn('"new_development"', prompt)
        self.assertNotIn('"action_id"', prompt)
        template = prompt.split("OUTPUT EXACTLY THIS SHAPE\n", 1)[-1]
        self.assertNotIn('"session_ref"', template)
        self.assertNotIn('"source_anchor"', template)

    def test_three_scenes_three_distinct_valid_anchors_pass(self) -> None:
        result = self._validate(self._planner(3))
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["unique_derived_anchor_session_count"], 3)

    def test_duplicate_derived_session_rejected(self) -> None:
        value = self._planner(3)
        value["scenes"][2]["primary_anchor_span"] = dict(
            value["scenes"][0]["primary_anchor_span"])
        result = self._validate(value)
        self.assertFalse(result["valid"])
        self.assertIn("planner_derived_anchor_sessions_not_unique", result["errors"])

    def test_unknown_parent_position_rejected(self) -> None:
        value = self._planner(3)
        value["scenes"][1]["primary_anchor_span"] = {
            "start_position": 10_000, "end_position": 10_000}
        result = self._validate(value)
        self.assertFalse(result["valid"])
        self.assertIn("scene_02:primary_anchor_span_not_parent", result["errors"])

    def test_two_scenes_two_distinct_anchors_pass(self) -> None:
        result = self._validate(self._planner(2))
        self.assertTrue(result["valid"], result["errors"])

    def test_three_nonadjacent_parent_roots_pass(self) -> None:
        value = self._planner(3)
        value["scenes"][0]["primary_anchor_span"] = {
            "start_position": 1, "end_position": 1}
        value["scenes"][1]["primary_anchor_span"] = {
            "start_position": 34, "end_position": 34}
        value["scenes"][2]["primary_anchor_span"] = {
            "start_position": 48, "end_position": 48}
        result = self._validate(value)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["derived_anchor_session_refs"], [
            "cc-train-episode-146700:parent_session_01",
            "cc-train-episode-146700:parent_session_04",
            "cc-train-episode-146700:parent_session_05",
        ])

    def test_missing_or_empty_durable_outcome_rejected(self) -> None:
        missing = self._planner(3)
        missing["scenes"][0].pop("durable_outcome")
        self.assertFalse(self._validate(missing)["valid"])
        empty = self._planner(3)
        empty["scenes"][0]["durable_outcome"] = "   "
        result = self._validate(empty)
        self.assertFalse(result["valid"])
        self.assertIn("scene_01:durable_outcome", result["errors"])

    def test_one_and_six_turn_spans_in_one_parent_session_pass(self) -> None:
        one = self._planner(3)
        self.assertTrue(self._validate(one)["valid"])
        six = self._planner(3)
        six["scenes"][0]["primary_anchor_span"] = {
            "start_position": 1, "end_position": 6}
        result = self._validate(six)
        self.assertTrue(result["valid"], result["errors"])
        anchor = result["materialized_anchors"][0]
        self.assertEqual(anchor["anchor_positions"], [1, 2, 3, 4, 5, 6])
        self.assertEqual(anchor["anchor_turn_refs"], [
            f"cc-train-episode-146700:P01:T{index:03d}"
            for index in range(1, 7)])
        self.assertEqual(
            anchor["derived_anchor_session_ref"],
            "cc-train-episode-146700:parent_session_01")

    def test_span_order_positive_and_size_guards(self) -> None:
        for span, error in (
            ({"start_position": 3, "end_position": 2},
             "scene_01:primary_anchor_span_order"),
            ({"start_position": 0, "end_position": 1},
             "scene_01:primary_anchor_span_order"),
            ({"start_position": 1, "end_position": 7},
             "scene_01:primary_anchor_span_too_long"),
        ):
            value = self._planner(3)
            value["scenes"][0]["primary_anchor_span"] = span
            result = self._validate(value)
            self.assertFalse(result["valid"])
            self.assertIn(error, result["errors"])

    def test_cross_session_and_child_positions_rejected(self) -> None:
        cross = self._planner(3)
        cross["scenes"][0]["primary_anchor_span"] = {
            "start_position": 8, "end_position": 9}
        result = self._validate(cross)
        self.assertFalse(result["valid"])
        self.assertIn(
            "scene_01:primary_anchor_span_cross_session", result["errors"])
        child = self._planner(3)
        child["scenes"][0]["primary_anchor_span"] = {
            "start_position": 63, "end_position": 63}
        result = self._validate(child)
        self.assertFalse(result["valid"])
        self.assertIn(
            "scene_01:primary_anchor_span_not_parent", result["errors"])

    def test_legacy_discrete_positions_are_schema_rejected(self) -> None:
        value = self._planner(3)
        value["scenes"][0]["source_anchor_positions"] = [1, 2]
        value["scenes"][0].pop("primary_anchor_span")
        result = self._validate(value)
        self.assertFalse(result["valid"])
        self.assertIn("scene_01:schema", result["errors"])

    def test_model_supplied_session_ref_is_schema_rejected(self) -> None:
        value = self._planner(3)
        value["scenes"][0]["session_ref"] = "forged"
        result = self._validate(value)
        self.assertFalse(result["valid"])
        self.assertIn("scene_01:schema", result["errors"])

    def test_compile_preserves_scene_refs_and_turn_budgets(self) -> None:
        context = build_fresh_r1_context(self.parent, token_counter=_tokens)
        compiled = compile_round_plan(
            self._planner(3), tail_context=context,
            parent_anchor_registry=build_parent_anchor_registry(
                self.parent["dialogue"]["turns"]),
        )
        self.assertEqual(
            [row["scene_ref"] for row in compiled["scenes"]],
            ["R01:S01", "R01:S02", "R01:S03"])
        self.assertEqual(
            compiled["round_turn_guidance"], {
                "soft_target": 48,
                "soft_range": [42, 54],
                "hard_minimum": 27,
                "hard_maximum": 66,
            })
        for scene in compiled["scenes"]:
            self.assertEqual(scene["turn_guidance"]["soft_range"], [14, 18])
            self.assertEqual(scene["turn_guidance"]["hard_minimum"], 9)
            self.assertIn("primary_anchor_span", scene)
            self.assertIn("anchor_positions", scene)
            self.assertIn("anchor_turn_refs", scene)
            self.assertIn("anchor_speakers", scene)
            self.assertIn("anchor_turn_texts", scene)
            self.assertIn("primary_anchor_materialization", scene)
            self.assertIn("derived_anchor_session_ref", scene)
            self.assertTrue(scene["durable_outcome"])
        self.assertEqual(
            compiled["scenes"][0]["primary_anchor_materialization"][0]["text"],
            self.parent["dialogue"]["turns"][0]["text"])
        self.assertEqual(
            compiled["scenes"][0]["anchor_speakers"],
            [self.parent["dialogue"]["turns"][0]["speaker"]])

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
        planner_output["planner_schema_version"] = PLANNER_SCHEMA_VERSION
        first_positions: list[int] = []
        seen_sessions: set[str] = set()
        for position, row in enumerate(self.parent["dialogue"]["turns"], 1):
            if row["session_id"] not in seen_sessions:
                seen_sessions.add(row["session_id"])
                first_positions.append(position)
        for index, scene in enumerate(planner_output["scenes"]):
            scene["primary_anchor_span"] = {
                "start_position": first_positions[index],
                "end_position": first_positions[index],
            }
            scene["durable_outcome"] = (
                f"Shadow durable arrangement {index + 1} is established.")
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
        self.assertIn('"primary_anchor_span"', seen["realizer"]["prompt"])
        self.assertIn('"primary_anchor_materialization"', seen["realizer"]["prompt"])
        self.assertIn('"derived_anchor_session_ref"', seen["realizer"]["prompt"])
        self.assertIn('"durable_outcome"', seen["realizer"]["prompt"])
        self.assertTrue(
            result["qa_context_view"]["qa_context_selection"]
            ["full_parent_child_included"])


if __name__ == "__main__":
    unittest.main()
