from __future__ import annotations

import json
import tempfile
import unittest
import urllib.error
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from ..live_backends import ChallengerAPIConfig
from ..simple_append_memorax_v1_fullr1_durable_devdiv_probe_v1.context import (
    build_fresh_r1_context,
)
from ..simple_append_v1.contracts import SCENE_HARD_MAXIMUM, SCENE_HARD_MINIMUM
from .contracts import (
    compile_round_plan, materialize_resolved_developments,
    validate_planner_continuation, validate_planner_output,
    validate_resolver_output,
)
from .preflight import _fixture_planner, _fixture_resolver, run_preflight
from .prompts import build_round_realizer_prompt
from .live_runtime import NoRetryOpenAIClient
from .settings import DEFAULT_INPUT, DEFAULT_OVERVIEW


class StateResolvedExpansionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.parent = json.loads(DEFAULT_INPUT.read_text(encoding="utf-8"))
        cls.tail = build_fresh_r1_context(
            cls.parent, token_counter=lambda text: max(1, len(text) // 4))
        cls.resolver = _fixture_resolver(cls.parent)
        cls.resolved = materialize_resolved_developments(
            cls.resolver, parent_turns=cls.parent["dialogue"]["turns"],
            root_id=cls.parent["root_id"],
            parent_dialogue_sha256=cls.parent["dialogue"]["dialogue_hash"])
        cls.planner = _fixture_planner(cls.resolved)

    def test_resolver_accepts_parent_refs_and_assigns_ids(self) -> None:
        validation = validate_resolver_output(
            self.resolver, parent_turns=self.parent["dialogue"]["turns"])
        self.assertTrue(validation["valid"], validation)
        self.assertEqual(
            [row["development_id"] for row in self.resolved["developments"]],
            ["D01", "D02", "D03"],
        )
        self.assertTrue(self.resolved["developments"][0]["evidence_source_turns"])

    def test_resolver_rejects_unknown_and_child_refs(self) -> None:
        for ref in ("missing:turn", "R01:S01:T001"):
            value = deepcopy(self.resolver)
            value["developments"][0]["evidence_turn_refs"] = [ref]
            validation = validate_resolver_output(
                value, parent_turns=self.parent["dialogue"]["turns"])
            self.assertFalse(validation["valid"])
            self.assertIn("development_01:evidence_ref_not_parent", validation["errors"])

    def test_resolver_rejects_empty_latest_state(self) -> None:
        value = deepcopy(self.resolver)
        value["developments"][0]["latest_state"] = ""
        validation = validate_resolver_output(
            value, parent_turns=self.parent["dialogue"]["turns"])
        self.assertFalse(validation["valid"])
        self.assertIn("development_01:latest_state", validation["errors"])

    def test_planner_rejects_unknown_development(self) -> None:
        value = deepcopy(self.planner)
        value["scenes"][0]["development_id"] = "D99"
        validation = validate_planner_output(
            value, resolved_set=self.resolved,
            parent_speakers=self.tail["parent_speakers"])
        self.assertFalse(validation["valid"])
        self.assertIn("scene_01:development_id_unknown", validation["errors"])

    def test_planner_rejects_duplicate_development(self) -> None:
        value = deepcopy(self.planner)
        value["scenes"][1]["development_id"] = value["scenes"][0]["development_id"]
        validation = validate_planner_output(
            value, resolved_set=self.resolved,
            parent_speakers=self.tail["parent_speakers"])
        self.assertFalse(validation["valid"])
        self.assertIn("planner_development_ids_not_unique", validation["errors"])

    def test_three_distinct_developments_compile(self) -> None:
        validation = validate_planner_output(
            self.planner, resolved_set=self.resolved,
            parent_speakers=self.tail["parent_speakers"])
        self.assertTrue(validation["valid"], validation)
        compiled = compile_round_plan(
            self.planner, resolved_set=self.resolved, tail_context=self.tail)
        self.assertEqual(
            [scene["development_id"] for scene in compiled["scenes"]],
            ["D01", "D02", "D03"],
        )
        self.assertFalse(compiled["state_resolution_contract"]["anchor_contract_used"])

    def test_obvious_latest_state_repetition_is_rejected(self) -> None:
        value = deepcopy(self.planner)
        value["scenes"][0]["new_development"] = (
            "A and B agree that regular practice should resume."
        )
        continuation = validate_planner_continuation(
            value, resolved_set=self.resolved)
        self.assertFalse(continuation["valid"], continuation)
        self.assertIn(
            "scene_01:new_development:obvious_latest_state_repetition",
            continuation["errors"],
        )

    def test_realizer_prompt_contains_frozen_latest_states(self) -> None:
        compiled = compile_round_plan(
            self.planner, resolved_set=self.resolved, tail_context=self.tail)
        prompt = build_round_realizer_prompt(
            compiled_plan=compiled, tail_context=self.tail)["prompt"]
        for row in self.resolved["developments"]:
            self.assertIn(row["latest_state"], prompt)
        self.assertIn("Treat that latest state as history", prompt)

    def test_length_and_context_contracts_are_unchanged(self) -> None:
        self.assertEqual(SCENE_HARD_MINIMUM, 9)
        self.assertEqual(SCENE_HARD_MAXIMUM, 48)
        self.assertTrue(self.tail["full_parent_included"])
        self.assertTrue(self.tail["complete_sessions_only"])
        self.assertTrue(self.tail["append_only"])

    def test_zero_api_preflight_passes_with_fake_token_counter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "new_run"
            report = run_preflight(
                input_path=DEFAULT_INPUT, overview_path=DEFAULT_OVERVIEW,
                run_root=run_root,
                api_config_path=Path("/root/autodl-tmp/gxy-selfplay/config.json"),
                tokenizer_path=Path("/root/autodl-tmp/Qwen3-4B"),
                token_counter=lambda text: max(1, len(text) // 4),
            )
            self.assertEqual(report["decision"], "pass", report)
            self.assertEqual(report["api_calls"], 0)
            self.assertFalse(run_root.exists())

    def test_transport_failure_is_not_retried(self) -> None:
        client = NoRetryOpenAIClient(ChallengerAPIConfig(
            base_url="https://example.invalid/v1", api_key="not-recorded",
            model="fixture-model", timeout_seconds=1,
        ))
        target = (
            "scripts.current_method.restart_n5_spell."
            "simple_append_memorax_v1_state_resolved_expansion_probe_v1."
            "live_runtime.urllib.request.urlopen"
        )
        with patch(target, side_effect=urllib.error.URLError("offline")) as request:
            with self.assertRaises(urllib.error.URLError):
                client.complete(prompt="{}", max_tokens=16)
        self.assertEqual(request.call_count, 1)


if __name__ == "__main__":
    unittest.main()
