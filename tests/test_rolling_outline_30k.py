import unittest
import json
import ssl
import tempfile
import urllib.error
from pathlib import Path
from unittest.mock import patch

from scripts.current_method.restart_n5_spell.graph_expansion import (
    GRAPH_STAGE_ACCOUNTING_COVERAGE_ONLY,
    compile_graph_scene_contracts,
    graph_scene_coverage_report,
)
from scripts.current_method.restart_n5_spell.rolling_outline_30k import (
    normalize_outline_plan, remap_baseline_turns, remap_round_child_turns,
    validate_raw_outline_plan,
)
from scripts.current_method.restart_n5_spell.live_backends import (
    ChallengerAPIConfig, OpenAICompatibleChallengerClient,
)
from scripts.current_method.restart_n5_spell.run_rolling_outline_30k_canary import (
    _questioner_round_without_stopping_expansion,
    load_imported_raw_outline,
    select_baseline_turns,
)
from scripts.current_method.restart_n5_spell.orchestrator import CanaryOrchestrator
from scripts.current_method.restart_n5_spell.prompts import build_graph_realizer_prompt


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return json.dumps({"choices": [{"message": {"content": "{}"},
            "finish_reason": "stop"}], "usage": {"total_tokens": 1}}).encode()


class RollingOutline30kTests(unittest.TestCase):
    def setUp(self):
        self.turns = [
            {"turn_id": f"D{i}:1", "session_id": f"D{i}",
             "speaker": "John" if i % 2 else "Maria", "text": f"Fact {i}."}
            for i in range(1, 9)
        ]
        anchors = [f"T{i:04d}" for i in range(1, 9)]
        self.raw = {
            "title": "A bounded continuation",
            "expansion_objective": "Develop a new local project through revisions.",
            "allowed_new_characters": [
                {"name": "Asha Patel", "role": "project coordinator",
                 },
                {"name": "Luis Gray", "role": "site liaison",
                 },
            ],
            "allowed_locations": ["library", "city office", "school yard"],
            "time_window": "six weeks",
            "event_arcs": [
                {"summary": f"Arc {i}",
                 "stages": [f"arc {i} stage {j}" for j in range(1, 5)],
                 "anchor_aliases": [anchors[(i - 1) % len(anchors)]]}
                for i in range(1, 5)
            ],
            "scenes": [
                {"purpose": f"Scene {i}",
                 "participants": (["John", "Maria", "Asha Patel"] if i == 2 else
                                  ["John", "Maria", "Luis Gray"] if i == 5 else
                                  ["John", "Maria"]),
                 "anchor_aliases": [anchors[(i - 1) % len(anchors)]] if i == 1
                                    else [],
                 "handoff_requirement": "Continue to the next bounded action."}
                for i in range(1, 11)
            ],
        }

    def test_valid_plan_normalizes_to_existing_graph_contract(self):
        result = validate_raw_outline_plan(self.raw, cumulative_turns=self.turns)
        self.assertEqual("rolling_outline_accept", result["decision"])
        outline = normalize_outline_plan(self.raw, cumulative_turns=self.turns,
                                         round_index=1)
        contracts = compile_graph_scene_contracts(outline)
        self.assertEqual(10, len(contracts))
        self.assertEqual(4, len(outline["planned_event_arcs"]))
        self.assertEqual(4, len(outline["parent_anchors"]))
        self.assertTrue(all(contract["target_turn_range"] == [12, 16]
                            for contract in contracts))

    def test_unknown_anchor_is_rejected(self):
        self.raw["event_arcs"][0]["anchor_aliases"] = ["UNKNOWN"]
        result = validate_raw_outline_plan(self.raw, cumulative_turns=self.turns)
        self.assertEqual("rolling_outline_schema_invalid", result["decision"])
        self.assertIn("event_arc_anchors", result["reasons"])

    def test_existing_character_declaration_is_not_materialized_as_new(self):
        self.raw["allowed_new_characters"][0]["name"] = "John"
        for scene in self.raw["scenes"]:
            scene["participants"] = ["John" if name == "Asha Patel" else name
                                     for name in scene["participants"]]
            scene["participants"] = list(dict.fromkeys(scene["participants"]))
        result = validate_raw_outline_plan(self.raw, cumulative_turns=self.turns)
        self.assertEqual("rolling_outline_accept", result["decision"])
        outline = normalize_outline_plan(self.raw, cumulative_turns=self.turns,
                                         round_index=1)
        self.assertNotIn("John", [row["name"] for row in
                                  outline["allowed_new_characters"]])

    def test_unused_new_character_is_not_materialized(self):
        for scene in self.raw["scenes"]:
            scene["participants"] = [name for name in scene["participants"]
                                     if name != "Asha Patel"]
        result = validate_raw_outline_plan(self.raw, cumulative_turns=self.turns)
        self.assertEqual("rolling_outline_accept", result["decision"])
        outline = normalize_outline_plan(self.raw, cumulative_turns=self.turns,
                                         round_index=1)
        self.assertNotIn("Asha Patel", [row["name"] for row in
                                        outline["allowed_new_characters"]])

    def test_baseline_and_round_ids_are_disjoint(self):
        baseline = remap_baseline_turns([
            {"turn_id": "D1:1", "speaker": "John", "text": "Old."},
            {"turn_id": "C0001", "session_id": "session_01",
             "speaker": "Maria", "text": "Child."},
        ])
        child = remap_round_child_turns([
            {"turn_id": "C0001", "session_id": "session_01",
             "speaker": "John", "text": "New."}], round_index=1)
        ids = [row["turn_id"] for row in baseline + child]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual("B00C0001", baseline[1]["turn_id"])
        self.assertEqual("R01C0001", child[0]["turn_id"])

    def test_parent_only_baseline_excludes_existing_child(self):
        source = {"dialogue": {"parent_turns": [{"turn_id": "P1"}],
                               "child_turns": [{"turn_id": "C1"}]}}
        self.assertEqual([{"turn_id": "P1"}],
                         select_baseline_turns(source, "parent_only"))
        self.assertEqual(2, len(select_baseline_turns(source, "full")))

    def test_imported_outline_is_revalidated_and_attested(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw_outline.json"
            path.write_text(json.dumps(self.raw), encoding="utf-8")
            raw, attestation = load_imported_raw_outline(path, self.turns)
        self.assertEqual(self.raw, raw)
        self.assertEqual(64, len(attestation["source_sha256"]))
        self.assertFalse(attestation["semantic_regeneration_performed"])

    def test_scene_coverage_does_not_claim_semantic_completion(self):
        outline = normalize_outline_plan(self.raw, cumulative_turns=self.turns,
                                         round_index=1)
        contracts = compile_graph_scene_contracts(outline)
        sessions = [{"graph_scene_id": contract["scene_id"],
                     "covered_event_stage_ids":
                         contract["assigned_event_stage_ids"]}
                    for contract in contracts]
        report = graph_scene_coverage_report(outline, contracts, sessions, 4100)
        self.assertTrue(report["complete"])
        self.assertTrue(report["layers"]["stage_coverage_complete"])
        self.assertFalse(report["layers"]["semantic_transition_complete"])
        self.assertEqual([], report["authoritative_completed_event_stage_ids"])

    def test_scene_coverage_uses_run_local_total_target_and_hard_minimum(self):
        outline = normalize_outline_plan(self.raw, cumulative_turns=self.turns,
                                         round_index=1)
        contracts = compile_graph_scene_contracts(outline, coverage_only=True)
        sessions = [{"graph_scene_id": contract["scene_id"],
                     "covered_event_stage_ids":
                         contract["assigned_event_stage_ids"]}
                    for contract in contracts]
        report = graph_scene_coverage_report(
            outline, contracts, sessions, 3900,
            hard_minimum_child_tokens=3800,
            target_child_token_range=(4000, 4800))
        self.assertTrue(report["complete"])
        self.assertFalse(report["target_child_token_range_met"])
        self.assertEqual(3800, report["minimum_accepted_child_tokens"])
        too_short = graph_scene_coverage_report(
            outline, contracts, sessions, 3799,
            hard_minimum_child_tokens=3800,
            target_child_token_range=(4000, 4800))
        self.assertFalse(too_short["complete"])

    def test_coverage_contracts_do_not_create_arbitrary_hooks(self):
        outline = normalize_outline_plan(self.raw, cumulative_turns=self.turns,
                                         round_index=1)
        contracts = compile_graph_scene_contracts(outline, coverage_only=True)
        self.assertTrue(all(contract["new_unresolved_hook_policy"] ==
                            "forbidden" for contract in contracts))
        self.assertTrue(all(contract["stage_accounting_mode"] ==
                            GRAPH_STAGE_ACCOUNTING_COVERAGE_ONLY
                            for contract in contracts))

    def test_coverage_graph_session_two_does_not_require_hook(self):
        outline = normalize_outline_plan(self.raw, cumulative_turns=self.turns,
                                         round_index=1)
        contract = compile_graph_scene_contracts(
            outline, coverage_only=True)[1]
        prompt = build_graph_realizer_prompt(
            graph_scene_contract=contract, session_index=2,
            narrative_beats=[{"event_id": "E1"}], recent_sessions=[],
            continuity_summary={}, active_plans=[], pending_decisions=[],
            superseded_information=[], character_knowledge={},
            visibility_boundaries=[], unresolved_hooks=[],
            allowed_people=["John", "Maria"],
            session_location=contract["session_location"],
            allowed_event_arc_ids=contract["assigned_event_arc_ids"],
            forbidden_conflicts=[], parent_anchors=[])
        self.assertIn("no unresolved hook is required", prompt)
        self.assertIn("MUST both be []", prompt)

    def test_coverage_mode_tracks_progress_without_completed_state(self):
        outline = normalize_outline_plan(self.raw, cumulative_turns=self.turns,
                                         round_index=1)
        orchestrator = object.__new__(CanaryOrchestrator)
        orchestrator.config = {"mode": "live_graph_expansion",
            "graph_stage_accounting_mode":
                GRAPH_STAGE_ACCOUNTING_COVERAGE_ONLY}
        orchestrator.data = {"outline": outline}
        state = orchestrator._event_arc_state([{
            "covered_event_stage_ids": ["E1:stage_01"],
            "advanced_event_arc_ids": ["E1"],
        }])
        self.assertEqual(0, state["E1"]["completed_stage_count"])
        self.assertEqual(1, state["E1"]["covered_stage_count"])
        self.assertEqual("unknown_not_audited",
                         state["E1"]["semantic_transition_status"])

    def test_tls_handshake_eof_reconnects_once(self):
        config = ChallengerAPIConfig(base_url="https://example.invalid/v1",
            api_key="x", model="Qwen/Qwen3-32B", enable_thinking=False)
        client = OpenAICompatibleChallengerClient(config)
        error = urllib.error.URLError(ssl.SSLEOFError(8, "EOF occurred"))
        with patch("urllib.request.urlopen", side_effect=[error, _Response()]) as call:
            response = client.complete(prompt="x", max_tokens=8, json_object=True)
        self.assertEqual(2, call.call_count)
        self.assertEqual(1, response.metadata["transport_retry_count"])

    def test_questioner_transport_error_does_not_stop_expansion(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
                "scripts.current_method.restart_n5_spell."
                "run_rolling_outline_30k_canary._questioner_round",
                side_effect=urllib.error.URLError("timed out")):
            report, backend = _questioner_round_without_stopping_expansion(
                root=Path(tmp), round_index=2, cumulative=[], tokenizer=None,
                api=None, feedback={}, feedback_sha="f", snapshot={},
                project_diff="d", solver_backend="solver")
        self.assertEqual("questioner_transport_error_ambiguous",
                         report["questioner_decision"])
        self.assertFalse(report["transport_retry_performed"])
        self.assertEqual("solver", backend)


if __name__ == "__main__":
    unittest.main()
