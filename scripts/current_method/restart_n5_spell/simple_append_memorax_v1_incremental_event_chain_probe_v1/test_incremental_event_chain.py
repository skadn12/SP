"""Zero-API contract tests for the isolated incremental event-chain probe."""

from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from ..simple_append_memorax_v1_fullr1_durable_devdiv_probe_v1.context import (
    build_fresh_r1_context,
)
from .contracts import (
    apply_delta_updates, audit_episode_146700_initialization,
    compile_round_plan, materialize_event_chains,
    normalize_round_realizer_output, validate_delta_updater_output,
    validate_initializer_output, validate_planner_continuation,
    validate_planner_output,
)
from .preflight import (
    _fixture_delta, _fixture_initializer, _fixture_planner,
    _fixture_realizer, run_preflight,
)
from .initializer_live import run_initializer_only_preflight
from .local_event_contract import (
    audit_episode_146700_local_candidates, compile_parent_session_windows,
    materialize_local_candidates, validate_local_event_output,
)
from .local_event_prompts import build_local_event_prompt
from .local_event_stage_a import _fixture_outputs, run_stage_a_preflight
from .prompts import build_delta_updater_prompt, build_round_realizer_prompt
from .settings import DEFAULT_INPUT, DEFAULT_OVERVIEW


class IncrementalEventChainContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.parent = json.loads(DEFAULT_INPUT.read_text(encoding="utf-8"))
        cls.tail = build_fresh_r1_context(
            cls.parent, token_counter=lambda value: len(value.split()))
        cls.initializer = _fixture_initializer(cls.parent)
        cls.chain_set = materialize_event_chains(
            cls.initializer, parent_turns=cls.parent["dialogue"]["turns"],
            root_id=cls.parent["root_id"],
            parent_dialogue_sha256=cls.parent["dialogue"]["dialogue_hash"])
        cls.planner = _fixture_planner(cls.chain_set)
        cls.compiled = compile_round_plan(
            cls.planner, chain_set=cls.chain_set, tail_context=cls.tail)
        cls.normalized = normalize_round_realizer_output(
            _fixture_realizer(cls.compiled), compiled_plan=cls.compiled,
            parent_utterances=[row["text"] for row in cls.parent["dialogue"]["turns"]],
            parent_turn_refs=[row["turn_id"] for row in cls.parent["dialogue"]["turns"]])
        cls.child = cls.normalized["child"]
        cls.delta = _fixture_delta(cls.compiled, cls.child)

    def test_initializer_contract_and_deterministic_ids(self) -> None:
        result = validate_initializer_output(
            self.initializer, parent_turns=self.parent["dialogue"]["turns"])
        self.assertTrue(result["valid"], result)
        self.assertEqual(
            [row["chain_id"] for row in self.chain_set["chains"]],
            ["EC0001", "EC0002", "EC0003", "EC0004", "EC0005"])
        self.assertEqual(
            self.chain_set["chains"][0]["source_event_refs"],
            ["P01:E01", "P01:E02"])

    def test_initializer_rejects_empty_state_and_invalid_refs(self) -> None:
        empty = deepcopy(self.initializer)
        empty["session_event_candidates"][0]["events"][0]["event_state"] = ""
        self.assertFalse(validate_initializer_output(
            empty, parent_turns=self.parent["dialogue"]["turns"])["valid"])
        missing = deepcopy(self.initializer)
        missing["session_event_candidates"][0]["events"][0]["evidence_turn_refs"] = ["missing"]
        self.assertFalse(validate_initializer_output(
            missing, parent_turns=self.parent["dialogue"]["turns"])["valid"])
        child = deepcopy(self.initializer)
        child["session_event_candidates"][0]["events"][0]["evidence_turn_refs"] = ["R01:S01:T001"]
        self.assertFalse(validate_initializer_output(
            child, parent_turns=self.parent["dialogue"]["turns"])["valid"])

    def test_canary_semantic_checkpoint_passes_fixture(self) -> None:
        result = audit_episode_146700_initialization(self.chain_set)
        self.assertTrue(result["valid"], result)
        self.assertEqual(len(set(result["concrete_event_matches"].values())), 5)

    def test_canary_semantic_checkpoint_rejects_known_failures(self) -> None:
        merged = deepcopy(self.initializer)
        merged["continuation_links"].append({
            "earlier_session_ref": merged["session_event_candidates"][2]["session_ref"],
            "earlier_candidate_index": 1,
            "later_session_ref": merged["session_event_candidates"][4]["session_ref"],
            "later_candidate_index": 1,
        })
        merged_chains = materialize_event_chains(
            merged, parent_turns=self.parent["dialogue"]["turns"],
            root_id=self.parent["root_id"],
            parent_dialogue_sha256=self.parent["dialogue"]["dialogue_hash"])
        self.assertIn(
            "chain_03:apology_report_identity_merged",
            audit_episode_146700_initialization(merged_chains)["errors"])
        soccer = deepcopy(self.initializer)
        soccer["session_event_candidates"][0]["events"][1]["event_state"] = (
            "B suggested regular soccer practice.")
        soccer_chains = materialize_event_chains(
            soccer, parent_turns=self.parent["dialogue"]["turns"],
            root_id=self.parent["root_id"],
            parent_dialogue_sha256=self.parent["dialogue"]["dialogue_hash"])
        self.assertIn(
            "chain_01:soccer_acceptance_missing_from_state",
            audit_episode_146700_initialization(soccer_chains)["errors"])
        health = deepcopy(self.initializer)
        health["session_event_candidates"][3]["events"][0]["event_state"] = (
            "A plans to see a doctor.")
        health_chains = materialize_event_chains(
            health, parent_turns=self.parent["dialogue"]["turns"],
            root_id=self.parent["root_id"],
            parent_dialogue_sha256=self.parent["dialogue"]["dialogue_hash"])
        self.assertIn(
            "chain_04:accepted_coverage_missing_from_state",
            audit_episode_146700_initialization(health_chains)["errors"])

    def test_initializer_rejects_cross_window_evidence(self) -> None:
        invalid = deepcopy(self.initializer)
        invalid["session_event_candidates"][0]["events"][0]["evidence_turn_refs"] = [
            "cc-train-episode-146700:P02:T001"]
        result = validate_initializer_output(
            invalid, parent_turns=self.parent["dialogue"]["turns"])
        self.assertIn(
            "session_01:event_01:evidence_outside_local_session", result["errors"])

    def test_continuation_link_must_reference_existing_forward_candidate(self) -> None:
        backward = deepcopy(self.initializer)
        backward["continuation_links"][0] = {
            "earlier_session_ref": backward["session_event_candidates"][0]["session_ref"],
            "earlier_candidate_index": 2,
            "later_session_ref": backward["session_event_candidates"][0]["session_ref"],
            "later_candidate_index": 1,
        }
        result = validate_initializer_output(
            backward, parent_turns=self.parent["dialogue"]["turns"])
        self.assertIn("link_01:continuation_must_point_forward", result["errors"])
        unknown = deepcopy(self.initializer)
        unknown["continuation_links"][0]["later_candidate_index"] = 99
        result = validate_initializer_output(
            unknown, parent_turns=self.parent["dialogue"]["turns"])
        self.assertIn("link_01:candidate_ref_unknown", result["errors"])

    def test_coverage_word_form_is_normalized_without_semantic_classifier(self) -> None:
        result = audit_episode_146700_initialization(self.chain_set)
        self.assertNotIn(
            "chain_04:accepted_coverage_missing_from_state", result["errors"])

    def test_planner_accepts_three_distinct_existing_chains(self) -> None:
        result = validate_planner_output(
            self.planner, chain_set=self.chain_set,
            parent_speakers=self.tail["parent_speakers"])
        self.assertTrue(result["valid"], result)
        self.assertEqual(len(set(result["chain_ids"])), 3)

    def test_planner_rejects_unknown_and_duplicate_chain(self) -> None:
        unknown = deepcopy(self.planner)
        unknown["scenes"][0]["chain_id"] = "EC9999"
        self.assertFalse(validate_planner_output(
            unknown, chain_set=self.chain_set,
            parent_speakers=self.tail["parent_speakers"])["valid"])
        duplicate = deepcopy(self.planner)
        duplicate["scenes"][1]["chain_id"] = duplicate["scenes"][0]["chain_id"]
        duplicate["scenes"][1]["participants"] = duplicate["scenes"][0]["participants"]
        self.assertIn("planner_chain_ids_not_unique", validate_planner_output(
            duplicate, chain_set=self.chain_set,
            parent_speakers=self.tail["parent_speakers"])["errors"])

    def test_planner_repetition_gate(self) -> None:
        repeated = deepcopy(self.planner)
        repeated["scenes"][0]["new_development"] = self.chain_set["chains"][0]["current_state"]
        result = validate_planner_continuation(repeated, chain_set=self.chain_set)
        self.assertFalse(result["valid"])

    def test_realizer_receives_frozen_current_state(self) -> None:
        prompt = build_round_realizer_prompt(
            compiled_plan=self.compiled, tail_context=self.tail)["prompt"]
        for scene in self.compiled["scenes"]:
            self.assertIn(scene["event_chain"]["current_state"], prompt)

    def test_delta_contract_accepts_child_only_evidence(self) -> None:
        result = validate_delta_updater_output(
            self.delta, chain_set=self.chain_set,
            compiled_plan=self.compiled, child=self.child)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["updated_count"], 3)

    def test_delta_rejects_parent_and_wrong_scene_evidence(self) -> None:
        parent = deepcopy(self.delta)
        parent["chain_updates"][0]["evidence_turn_refs"] = [
            self.parent["dialogue"]["turns"][0]["turn_id"]]
        self.assertFalse(validate_delta_updater_output(
            parent, chain_set=self.chain_set,
            compiled_plan=self.compiled, child=self.child)["valid"])
        wrong = deepcopy(self.delta)
        wrong["chain_updates"][0]["evidence_turn_refs"] = [
            self.child["scenes"][1]["dialogue_turns"][0]["turn_ref"]]
        self.assertIn("update_01:evidence_outside_mapped_scene", validate_delta_updater_output(
            wrong, chain_set=self.chain_set,
            compiled_plan=self.compiled, child=self.child)["errors"])

    def test_delta_rejects_unknown_chain_and_planner_text_as_evidence(self) -> None:
        unknown = deepcopy(self.delta)
        unknown["chain_updates"][0]["chain_id"] = "EC9999"
        self.assertFalse(validate_delta_updater_output(
            unknown, chain_set=self.chain_set,
            compiled_plan=self.compiled, child=self.child)["valid"])
        planned = deepcopy(self.delta)
        planned["chain_updates"][0]["evidence_turn_refs"] = [
            self.compiled["scenes"][0]["new_development"]]
        self.assertFalse(validate_delta_updater_output(
            planned, chain_set=self.chain_set,
            compiled_plan=self.compiled, child=self.child)["valid"])

    def test_delta_does_not_modify_unused_chain(self) -> None:
        before = deepcopy(self.chain_set["chains"][3])
        history_lengths = [
            len(row["history"]) for row in self.chain_set["chains"][:3]]
        updated = apply_delta_updates(
            self.delta, chain_set=self.chain_set,
            compiled_plan=self.compiled, child=self.child)
        self.assertEqual(updated["chains"][3], before)
        self.assertTrue(all(
            len(row["history"]) == previous + 1
            for row, previous in zip(updated["chains"][:3], history_lengths)))

    def test_delta_prompt_has_no_planner_claim_text(self) -> None:
        prompt = build_delta_updater_prompt(
            chain_set=self.chain_set, compiled_plan=self.compiled,
            child=self.child)["prompt"]
        for scene in self.compiled["scenes"]:
            self.assertNotIn(scene["new_development"], prompt)
            self.assertNotIn(scene["durable_outcome"], prompt)

    def test_zero_api_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = run_preflight(
                input_path=DEFAULT_INPUT, overview_path=DEFAULT_OVERVIEW,
                run_root=Path(directory) / "absent-live-root",
                token_counter=lambda value: len(value.split()))
        self.assertEqual(report["decision"], "pass", report)
        self.assertEqual(report["api_calls"], 0)
        self.assertEqual(report["request_budget"]["total"], 4)
        self.assertEqual(report["request_budget"]["retries"], 0)
        self.assertEqual(report["request_budget"]["memorax"], 0)

    def test_initializer_only_preflight_freezes_all_downstream_calls_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = run_initializer_only_preflight(
                input_path=DEFAULT_INPUT, overview_path=DEFAULT_OVERVIEW,
                run_root=Path(directory) / "absent-initializer-root")
        self.assertEqual(report["decision"], "pass", report)
        self.assertEqual(report["request_budget"], {
            "chain_initializer": 1, "planner": 0, "round_realizer": 0,
            "delta_updater": 0, "total": 1, "retries": 0,
            "memorax": 0, "qa": 0, "solver": 0,
        })

    def test_stage_a_compiles_five_complete_parent_sessions(self) -> None:
        windows = compile_parent_session_windows(self.parent)
        self.assertEqual([row["window_id"] for row in windows], [
            "P01", "P02", "P03", "P04", "P05"])
        self.assertEqual(
            sum(row["turn_count"] for row in windows),
            self.parent["dialogue"]["turn_count"])

    def test_stage_a_fixtures_materialize_independent_candidates(self) -> None:
        windows = compile_parent_session_windows(self.parent)
        outputs = _fixture_outputs(windows)
        speakers = list(dict.fromkeys(
            row["speaker"] for row in self.parent["dialogue"]["turns"]))
        self.assertTrue(all(validate_local_event_output(
            output, window=window, parent_speakers=speakers)["valid"]
            for window, output in zip(windows, outputs)))
        candidates = materialize_local_candidates(list(zip(windows, outputs)))
        self.assertEqual([row["candidate_ref"] for row in candidates], [
            "P01:E01", "P02:E01", "P03:E01", "P04:E01", "P05:E01"])
        audit = audit_episode_146700_local_candidates(candidates)
        self.assertTrue(audit["valid"], audit)
        self.assertFalse(audit["linking_performed"])

    def test_stage_a_local_contract_rejects_other_session_ref(self) -> None:
        windows = compile_parent_session_windows(self.parent)
        output = _fixture_outputs(windows)[0]
        output["events"][0]["evidence_turn_refs"] = [
            "cc-train-episode-146700:P02:T001"]
        result = validate_local_event_output(
            output, window=windows[0],
            parent_speakers=self.tail["parent_speakers"])
        self.assertIn("event_01:evidence_outside_local_window", result["errors"])

    def test_stage_a_prompt_contains_no_memory_or_link_output(self) -> None:
        window = compile_parent_session_windows(self.parent)[0]
        prompt = build_local_event_prompt(window)["prompt"]
        self.assertNotIn("memorax", prompt.casefold())
        self.assertNotIn('"continuation_links"', prompt)
        self.assertIn("Do not create continuation links", prompt)

    def test_stage_a_preflight_is_zero_api_and_five_calls_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = run_stage_a_preflight(
                input_path=DEFAULT_INPUT, overview_path=DEFAULT_OVERVIEW,
                run_root=Path(directory) / "absent-stage-a-root")
        self.assertEqual(report["decision"], "pass", report)
        self.assertEqual(report["api_calls"], 0)
        self.assertEqual(report["request_budget"]["local_event_extractor"], 5)
        self.assertEqual(report["request_budget"]["linker"], 0)


if __name__ == "__main__":
    unittest.main()
