import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from scripts.current_method.restart_n5_spell.simple_append_v1.multi_round_live import (
    DEFAULT_API_CONFIG,
    DEFAULT_BASELINE_QA_REPLAY,
    DEFAULT_BASELINE_RUN,
    DEFAULT_SOLVER_MODEL,
    DEFAULT_TOKENIZER,
    _round_focus_stats,
    _evaluate_round_qa,
    preflight,
)
from scripts.current_method.restart_n5_spell.simple_append_v1.prompts import (
    build_mcq_batch_prompt,
)
from scripts.current_method.restart_n5_spell.simple_append_v1.contracts import (
    compile_round_plan,
    normalize_round_realizer_output,
    validate_planner_output,
)
from scripts.current_method.restart_n5_spell.simple_append_v1.qa import (
    MCQ_BATCH_SCHEMA_VERSION,
    MCQ_FOCUS_WRITER_SCHEMA_VERSION,
    build_evidence_alias_registry,
    evaluate_focus_round_profile,
    partition_mcq_candidates,
)


class SimpleAppendMultiRoundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.round1 = json.loads(
            (DEFAULT_BASELINE_RUN / "expanded_dialogue.json").read_text())

    def test_focus_round_prompt_uses_batch_level_latest_round_quota(self):
        prompt = build_mcq_batch_prompt(self.round1, focus_round_index=1)["prompt"]
        self.assertIn('"focus_round_index":1', prompt)
        self.assertIn('"focus_alias_range":["D061","D108"]', prompt)
        self.assertIn("At least 4 questions must cite the current focus round",
                      prompt)
        self.assertIn('"coverage_role":"current_round_only"', prompt)
        self.assertIn("current_round_joint_history", prompt)
        self.assertIn("CURRENT FOCUS ROUND", prompt)
        self.assertIn("ORIGINAL PARENT", prompt)
        self.assertIn("EARLIER GENERATED CHILD", prompt)

        output = json.loads(prompt.split(
            "OUTPUT EXACTLY THIS SHAPE\n", 1)[1])
        roles = [row["coverage_role"] for row in output["questions"]]
        self.assertEqual([
            "current_round_only", "current_round_only",
            "current_round_joint_history", "current_round_joint_history",
            "parent_child", "parent_child",
            "general", "general",
        ], roles)

    def test_focus_prompt_partitions_each_authoritative_turn_exactly_once(self):
        expanded, _ = self._round4_artifacts()
        prompt = build_mcq_batch_prompt(
            expanded, focus_round_index=4)["prompt"]
        current_marker = (
            "CURRENT FOCUS ROUND — REQUIRED SOURCE FOR BOTH CURRENT-ROUND POOLS\n")
        parent_marker = (
            "ORIGINAL PARENT — REQUIRED HISTORY SOURCE FOR JOINT QUESTIONS\n")
        earlier_marker = (
            "EARLIER GENERATED CHILD — OPTIONAL HISTORY, NOT A PARENT SUBSTITUTE\n")
        current = json.loads(prompt.split(current_marker, 1)[1].split(
            "\n\n", 1)[0])
        parent = json.loads(prompt.split(parent_marker, 1)[1].split(
            "\n\n", 1)[0])
        earlier = json.loads(prompt.split(earlier_marker, 1)[1].split(
            "\n\n", 1)[0])
        refs = [row["evidence_ref"] for block in (current, parent, earlier)
                for row in block["turns"]]
        self.assertEqual(int(expanded["turn_count"]), len(refs))
        self.assertEqual(len(refs), len(set(refs)))
        self.assertEqual("D206", current["turns"][0]["evidence_ref"])
        self.assertEqual("D001", parent["turns"][0]["evidence_ref"])
        self.assertEqual("D061", earlier["turns"][0]["evidence_ref"])
        self.assertEqual(1, prompt.count('"evidence_ref":"D001"'))
        self.assertEqual(1, prompt.count('"evidence_ref":"D061"'))
        self.assertEqual(1, prompt.count('"evidence_ref":"D206"'))

    def test_round_focus_stats_distinguish_latest_from_earlier_evidence(self):
        value = {"mcq_batch_schema_version": MCQ_BATCH_SCHEMA_VERSION,
                 "questions": [{
                     "question_type": "causal",
                     "question": "Fixture question?",
                     "options": {"A": "one", "B": "two",
                                 "C": "three", "D": "four"},
                     "correct_option": "A",
                     "evidence_turn_refs": ["D001", "D108"],
                 }]}
        partition = partition_mcq_candidates(
            value, expanded_dialogue=self.round1,
            minimum_count=1, maximum_count=None,
            min_child_supported=1, min_joint_parent_child=1)
        stats = _round_focus_stats(
            partition, expanded=self.round1, round_index=1)
        self.assertEqual(1, stats["focus_round_supported_count"])
        self.assertEqual(1, stats["focus_round_joint_with_history_count"])

    @staticmethod
    def _round4_artifacts():
        root = Path(
            "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
            "conversation_chronicles_episode_61555_simple_append_v1_"
            "round_04_live_3/round_04")
        return (
            json.loads((root / "expanded_dialogue.json").read_text()),
            json.loads((root / "qa_evaluation/qa/writer_output.json").read_text()),
        )

    def test_round4_historical_child_questions_do_not_satisfy_focus_quota(self):
        expanded, raw = self._round4_artifacts()
        partition = partition_mcq_candidates(
            raw, expanded_dialogue=expanded, minimum_count=8,
            maximum_count=None, min_child_supported=4,
            min_joint_parent_child=2)
        self.assertTrue(partition["batch_profile_ready"], partition)
        focus = evaluate_focus_round_profile(
            partition, expanded_dialogue=expanded, focus_round_index=4)
        self.assertFalse(focus["focus_round_profile_ready"], focus)
        self.assertEqual(0, focus["focus_round_supported_count"])
        self.assertEqual(0, focus["focus_round_joint_with_history_count"])

    def test_four_current_and_two_joint_questions_satisfy_focus_quota(self):
        expanded, raw = self._round4_artifacts()
        aliases = build_evidence_alias_registry(expanded)["turn_ref_to_alias"]
        parent = aliases["cc-train-episode-61555:P01:T001"]
        current = aliases["R04:S01:T001"]
        for index, row in enumerate(raw["questions"]):
            row["evidence_turn_refs"] = (
                [parent, current] if index < 2 else
                [current] if index < 4 else [parent])
        partition = partition_mcq_candidates(
            raw, expanded_dialogue=expanded, minimum_count=8,
            maximum_count=None, min_child_supported=4,
            min_joint_parent_child=2)
        focus = evaluate_focus_round_profile(
            partition, expanded_dialogue=expanded, focus_round_index=4)
        self.assertTrue(partition["batch_profile_ready"], partition)
        self.assertTrue(focus["focus_round_profile_ready"], focus)
        self.assertEqual(4, focus["focus_round_supported_count"])
        self.assertEqual(2, focus["focus_round_joint_with_history_count"])

    def test_three_current_questions_fail_focus_quota(self):
        expanded, raw = self._round4_artifacts()
        aliases = build_evidence_alias_registry(expanded)["turn_ref_to_alias"]
        parent = aliases["cc-train-episode-61555:P01:T001"]
        current = aliases["R04:S01:T001"]
        for index, row in enumerate(raw["questions"]):
            row["evidence_turn_refs"] = (
                [parent, current] if index < 2 else
                [current] if index == 2 else [parent])
        partition = partition_mcq_candidates(
            raw, expanded_dialogue=expanded, minimum_count=8,
            maximum_count=None, min_child_supported=1,
            min_joint_parent_child=0)
        focus = evaluate_focus_round_profile(
            partition, expanded_dialogue=expanded, focus_round_index=4)
        self.assertFalse(focus["focus_round_profile_ready"], focus)
        self.assertIn("current_round_supported_below_minimum",
                      focus["focus_round_profile_errors"])

    def test_four_current_but_one_joint_fails_focus_quota(self):
        expanded, raw = self._round4_artifacts()
        aliases = build_evidence_alias_registry(expanded)["turn_ref_to_alias"]
        parent = aliases["cc-train-episode-61555:P01:T001"]
        current = aliases["R04:S01:T001"]
        for index, row in enumerate(raw["questions"]):
            row["evidence_turn_refs"] = (
                [parent, current] if index == 0 else
                [current] if index < 4 else [parent])
        partition = partition_mcq_candidates(
            raw, expanded_dialogue=expanded, minimum_count=8,
            maximum_count=None, min_child_supported=1,
            min_joint_parent_child=0)
        focus = evaluate_focus_round_profile(
            partition, expanded_dialogue=expanded, focus_round_index=4)
        self.assertFalse(focus["focus_round_profile_ready"], focus)
        self.assertIn("current_round_joint_with_history_below_minimum",
                      focus["focus_round_profile_errors"])

    def test_focus_profile_deficit_does_not_skip_local_solver(self):
        expanded, raw = self._round4_artifacts()
        split_raw = {
            "mcq_focus_writer_schema_version":
                MCQ_FOCUS_WRITER_SCHEMA_VERSION,
            "questions": [{
                **row,
                "coverage_role": (
                    "current_round_only" if index < 2 else
                    "current_round_joint_history" if index < 4 else
                    "parent_child" if index < 6 else
                    "general"),
            } for index, row in enumerate(raw["questions"])],
        }
        with tempfile.TemporaryDirectory() as temporary, patch(
                "scripts.current_method.restart_n5_spell.simple_append_v1."
                "multi_round_live._writer_call",
                return_value=(split_raw, {"usage": {"total_tokens": 1}})), patch(
                "scripts.current_method.restart_n5_spell.simple_append_v1."
                "multi_round_live.LocalSolverRunner") as local_solver, patch(
                "scripts.current_method.restart_n5_spell.simple_append_v1."
                "multi_round_live._evaluate_grounding",
                return_value=({"decision": "reject",
                               "semantic_correct_rate": 0.0}, [])):
            report = _evaluate_round_qa(
                round_root=Path(temporary), expanded=expanded,
                round_index=4, api_config=DEFAULT_API_CONFIG,
                tokenizer=None, solver_backend=None, manifest_sha="fixture")
        self.assertTrue(report["writer_admission_ready"], report)
        self.assertEqual("completed_top_up_required",
                         report["qa_stage_status"])
        self.assertTrue(report["top_up_required"])
        self.assertGreater(report["candidate_count"], 0)
        self.assertFalse(report["qa_pipeline_ready"])
        local_solver.assert_called_once()

    def test_preflight_is_zero_api_and_does_not_create_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "multi"
            report = preflight(
                baseline_run=DEFAULT_BASELINE_RUN,
                baseline_qa_replay=DEFAULT_BASELINE_QA_REPLAY,
                run_root=root, api_config=DEFAULT_API_CONFIG,
                tokenizer_path=DEFAULT_TOKENIZER,
                solver_model=DEFAULT_SOLVER_MODEL, total_rounds=3)
            self.assertEqual("pass", report["decision"], report)
            self.assertEqual(0, report["api_calls"])
            self.assertFalse(root.exists())

    def test_failed_round3_speaker_roster_shadow_and_declared_fix(self):
        historical_root = Path(
            "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
            "conversation_chronicles_episode_61555_simple_append_v1_"
            "multiround_live_1")
        round_root = historical_root / "round_03"
        historical_plan = json.loads(
            (round_root / "planner/compiled_round_plan.json").read_text())
        historical_realizer = json.loads(
            (round_root / "realizer/round_realizer_output.json").read_text())
        prior_expanded = json.loads(
            (historical_root / "round_02/expanded_dialogue.json").read_text())
        parent_utterances = [str(row["text"])
                             for row in prior_expanded["turns"]]

        historical = normalize_round_realizer_output(
            historical_realizer, compiled_plan=historical_plan,
            parent_utterances=parent_utterances)
        self.assertFalse(historical["valid"])
        self.assertEqual([
            "scene_02:turn_002:speaker",
            "scene_02:turn_006:speaker",
            "scene_02:turn_013:speaker",
        ], historical["errors"])

        planner_output = deepcopy(json.loads(
            (round_root / "planner/planner_output.json").read_text()))
        planner_output["scenes"][1]["participants"].append("Specialist D")
        tail_context = json.loads((round_root / "tail_context.json").read_text())
        planner_validation = validate_planner_output(
            planner_output, parent_speakers=tail_context["parent_speakers"])
        self.assertTrue(planner_validation["valid"], planner_validation)
        corrected_plan = compile_round_plan(
            planner_output, tail_context=tail_context, round_index=3)
        self.assertIn(
            "Specialist D",
            corrected_plan["speaker_roster_contract"]["round_speakers"])
        corrected = normalize_round_realizer_output(
            historical_realizer, compiled_plan=corrected_plan,
            parent_utterances=parent_utterances)
        self.assertTrue(corrected["valid"], corrected)
        self.assertEqual(48, corrected["turn_count"])


if __name__ == "__main__":
    unittest.main()
