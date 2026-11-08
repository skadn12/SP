"""Zero-API tests for the isolated Structural QA Writer v19 branch."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from ..simple_append_v1.qa import materialize_solver_candidate
from ..simple_append_v1.qa_live import build_solver_source
from .integrity import protected_manifests
from .structural_qa_live import preflight
from .structural_contract import (
    STRUCTURAL_QA_SCHEMA_VERSION,
    solver_partition,
    validate_structural_writer_output,
)
from .structural_prompts import build_structural_qa_prompt


EXPERIMENT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_146700_"
    "simple_append_memorax_fullr1_devdiv_probe_v1_experiment_1")
LIVE_V19 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_146700_"
    "qa_structural_v19_probe_v1_live_1")
LIVE_MEMORY_EXPANSION = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_91716_"
    "memory_event_continuation_recovery_live_2")


class StructuralQAV19Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.expanded = json.loads((
            EXPERIMENT / "expansion/expanded_dialogue.json").read_text())
        cls.overview = json.loads((
            EXPERIMENT / "expansion/memory/post_generation_overview.json"
        ).read_text())

    def _envelope(self, questions):
        empty = not questions
        return {
            "structural_qa_schema_version": STRUCTURAL_QA_SCHEMA_VERSION,
            "abstention_summary": {
                "estimated_skipped_count": 1 if empty else 0,
                "reason_codes": (["no_discriminative_structure"]
                                 if empty else []),
            },
            "questions": questions,
        }

    def _question(self, *, candidate_id="Q01", kind="temporal_sequence",
                  variable="event_order", refs=None, contract=None):
        refs = refs or ["D063", "D064", "D065"]
        contract = contract or {"events": [
            {"event": "A raises the problem", "turn_ids": [refs[0]]},
            {"event": "B proposes a step", "turn_ids": [refs[1]]},
            {"event": "A accepts it", "turn_ids": [refs[2]]},
        ]}
        return {
            "candidate_id": candidate_id,
            "question_type": kind,
            "source_scope": "child_only",
            "reasoning_structure": {
                "discriminative_variable": variable,
                "supporting_turn_ids": refs,
                "short_rationale": "The local order distinguishes the options.",
                "type_contract": contract,
            },
            "question": "Which order matches the dialogue?",
            "options": {
                "A": "A raises the problem, B proposes a step, A accepts it",
                "B": "B proposes a step, A raises the problem, A accepts it",
                "C": "A accepts it, B proposes a step, A raises the problem",
                "D": "A raises the problem, A accepts it, B proposes a step",
            },
            "gold_text": "A raises the problem, B proposes a step, A accepts it",
            "evidence_turn_ids": refs,
        }

    def test_valid_temporal_sequence(self):
        result = validate_structural_writer_output(
            self._envelope([self._question()]),
            expanded_dialogue=self.expanded)
        self.assertTrue(result["envelope_valid"])
        self.assertEqual(result["structural_valid_count"], 1)
        self.assertEqual(result["evidence_statistics"]["max"], 3)

    def test_valid_visibility_and_revision(self):
        visibility = self._question(
            candidate_id="Q01", kind="information_visibility",
            variable="knowledge_holder_at_timepoint", refs=["D063", "D064"],
            contract={
                "fact": "the missed kick happened",
                "timepoint": "before the new practice plan",
                "candidate_knowers": [
                    "Co-workers A", "Co-workers B", "Boss", "Specialist D"],
                "turn_ids": ["D063", "D064"],
            })
        visibility["question"] = (
            "Who knew that the missed kick happened at the timepoint before the new "
            "practice plan?")
        visibility["options"] = {
            "A": "Co-workers A", "B": "Co-workers B",
            "C": "Boss", "D": "Specialist D",
        }
        visibility["gold_text"] = "Co-workers A"
        revision = self._question(
            candidate_id="Q02", kind="information_revision",
            variable="old_vs_revised_plan", refs=["D063", "D064", "D065"],
            contract={
                "subject": "the practice plan",
                "old_state": "practice was not scheduled",
                "update": "B proposes restarting",
                "new_state": "A agrees to restart",
                "before_turn_ids": ["D063"],
                "update_turn_ids": ["D064"],
                "after_turn_ids": ["D065"],
            })
        revision["question"] = (
            "After B proposes restarting, what was the practice plan's new state?")
        revision["options"] = {
            "A": "A agrees to restart",
            "B": "practice was not scheduled",
            "C": "practice moves to weekends",
            "D": "the plan is cancelled",
        }
        revision["gold_text"] = "A agrees to restart"
        result = validate_structural_writer_output(
            self._envelope([visibility, revision]),
            expanded_dialogue=self.expanded)
        self.assertEqual(result["structural_valid_count"], 2)

    def test_fake_revision_and_missing_reasoning_rejected(self):
        row = self._question(
            kind="information_revision", variable="old_vs_revised_plan",
            contract={
                "subject": "the plan", "old_state": "", "update": "a choice",
                "new_state": "done",
                "before_turn_ids": ["D063"], "update_turn_ids": ["D064"],
                "after_turn_ids": ["D065"],
            })
        result = validate_structural_writer_output(
            self._envelope([row]), expanded_dialogue=self.expanded)
        self.assertIn("revision_states", result["items"][0]["structural_errors"])
        row = self._question()
        row["reasoning_structure"] = None
        result = validate_structural_writer_output(
            self._envelope([row]), expanded_dialogue=self.expanded)
        self.assertFalse(result["items"][0]["valid"])

    def test_actor_recall_cannot_masquerade_as_visibility(self):
        row = self._question(
            kind="information_visibility",
            variable="knowledge_holder_at_timepoint",
            refs=["D063", "D064"],
            contract={
                "fact": "Co-workers A missed the kick",
                "timepoint": "before the new practice plan",
                "candidate_knowers": [
                    "Co-workers A", "Co-workers B", "Boss", "Specialist D"],
                "turn_ids": ["D063", "D064"],
            })
        row["question"] = (
            "Who knew about the missed kick and encouraged A to resume practice?")
        row["options"] = {
            "A": "Co-workers A", "B": "Co-workers B",
            "C": "Boss", "D": "Specialist D",
        }
        row["gold_text"] = "Co-workers B"
        result = validate_structural_writer_output(
            self._envelope([row]), expanded_dialogue=self.expanded)
        self.assertIn(
            "visibility_question_not_knowledge_only",
            result["items"][0]["structural_errors"])

    def test_revision_dependency_and_temporal_answer_slots_are_enforced(self):
        revision = self._question(
            kind="information_revision", variable="old_vs_revised_plan",
            contract={
                "subject": "the plan", "old_state": "uncertain",
                "update": "B proposed practice", "new_state": "committed",
                "before_turn_ids": ["D063"], "update_turn_ids": ["D064"],
                "after_turn_ids": ["D065"],
            })
        revision["question"] = "What was the sequence of plans?"
        revision["options"] = {
            "A": "committed", "B": "uncertain", "C": "cancelled", "D": "deferred"}
        revision["gold_text"] = "committed"
        result = validate_structural_writer_output(
            self._envelope([revision]), expanded_dialogue=self.expanded)
        self.assertIn(
            "revision_question_not_update_focused",
            result["items"][0]["structural_errors"])

        dependency = self._question(
            kind="task_dependency_or_constraint", variable="prerequisite",
            refs=["D064", "D065"], contract={
                "prerequisite": "resume practice",
                "dependent_outcome": "regain confidence",
                "turn_ids": ["D064", "D065"],
            })
        dependency["question"] = "Who encouraged A to practice?"
        dependency["options"] = {
            "A": "Co-workers B", "B": "Boss", "C": "Specialist D", "D": "Friend C"}
        dependency["gold_text"] = "Co-workers B"
        result = validate_structural_writer_output(
            self._envelope([dependency]), expanded_dialogue=self.expanded)
        self.assertIn(
            "dependency_question_not_prerequisite_focused",
            result["items"][0]["structural_errors"])
        self.assertIn(
            "dependency_gold_not_prerequisite",
            result["items"][0]["structural_errors"])

        temporal = self._question()
        temporal["options"]["D"] = "A raises the problem and then stops"
        result = validate_structural_writer_output(
            self._envelope([temporal]), expanded_dialogue=self.expanded)
        self.assertIn(
            "temporal_option_not_event_permutation",
            result["items"][0]["structural_errors"])

    def test_evidence_limits_invalid_and_duplicate_refs(self):
        over = self._question(refs=[f"D{index:03d}" for index in range(63, 72)])
        over["reasoning_structure"]["type_contract"] = {"events": [
            {"event": f"event {index}", "turn_ids": [ref]}
            for index, ref in enumerate(over["evidence_turn_ids"], 1)]}
        result = validate_structural_writer_output(
            self._envelope([over]), expanded_dialogue=self.expanded)
        self.assertIn("evidence_over_limit", result["items"][0]["structural_errors"])

        duplicate = self._question(refs=["D063", "D064", "D064"])
        result = validate_structural_writer_output(
            self._envelope([duplicate]), expanded_dialogue=self.expanded)
        self.assertIn("evidence_duplicate_refs", result["items"][0]["structural_errors"])

        invalid = self._question(refs=["D063", "D064", "D999"])
        result = validate_structural_writer_output(
            self._envelope([invalid]), expanded_dialogue=self.expanded)
        self.assertIn("evidence_reference_invalid",
                      result["items"][0]["structural_errors"])

    def test_abstention_contract_accepts_zero_one_three_and_twelve(self):
        for count in (0, 1, 3, 12):
            with self.subTest(count=count):
                questions = []
                for index in range(1, count + 1):
                    row = self._question(candidate_id=f"Q{index:02d}")
                    row["question"] = f"Which order matches development {index}?"
                    questions.append(row)
                result = validate_structural_writer_output(
                    self._envelope(questions), expanded_dialogue=self.expanded)
                self.assertTrue(result["envelope_valid"])
                self.assertEqual(result["structural_valid_count"], count)
                self.assertEqual(result["structural_invalid_count"], 0)
                if count == 0:
                    self.assertTrue(result["zero_question_batch_valid"])
                    self.assertEqual(result["normalized_batch"]["questions"], [])

    def test_unexplained_empty_batch_is_rejected(self):
        value = self._envelope([])
        value["abstention_summary"] = {
            "estimated_skipped_count": 0,
            "reason_codes": [],
        }
        result = validate_structural_writer_output(
            value, expanded_dialogue=self.expanded)
        self.assertFalse(result["envelope_valid"])
        self.assertIn(
            "empty_batch_without_abstention_evidence",
            result["envelope_errors"])

    def test_exact_duplicate_and_malformed_options_quarantined(self):
        first = self._question(candidate_id="Q01")
        second = copy.deepcopy(first)
        second["candidate_id"] = "Q02"
        result = validate_structural_writer_output(
            self._envelope([first, second]), expanded_dialogue=self.expanded)
        self.assertEqual(result["structural_valid_count"], 1)
        self.assertTrue(any(error.startswith("exact_duplicate_overflow:")
                            for error in result["items"][1]["structural_errors"]))
        malformed = self._question()
        malformed["options"] = {"A": "one", "B": "two"}
        result = validate_structural_writer_output(
            self._envelope([malformed]), expanded_dialogue=self.expanded)
        self.assertIn("options", result["items"][0]["structural_errors"])

    def test_final_mcq_backward_compatibility(self):
        validation = validate_structural_writer_output(
            self._envelope([self._question()]),
            expanded_dialogue=self.expanded)
        normalized = validation["normalized_batch"]
        partition = solver_partition(validation)
        source = build_solver_source(self.expanded)
        qa = materialize_solver_candidate(
            normalized["questions"][0], index=1,
            expanded_dialogue=self.expanded,
            span_catalog=source["span_catalog"],
            candidate_status=partition["items"][0])
        self.assertEqual(set(qa["options"]), set("ABCD"))
        self.assertEqual(len(qa["evidence_turn_ids"]), 3)
        self.assertIn(qa["reference_answer_letter"], "ABCD")

    def test_remaining_question_type_contracts(self):
        fixtures = [
            ("state_change", "previous_vs_current_state", {
                "subject": "the plan", "old_state": "uncertain",
                "update": "a proposal is made",
                "new_state": "committed", "before_turn_ids": ["D063"],
                "update_turn_ids": ["D064"], "after_turn_ids": ["D065"]}),
            ("task_dependency_or_constraint", "prerequisite", {
                "prerequisite": "review the figures",
                "dependent_outcome": "submit the corrected report",
                "turn_ids": ["D095", "D096"]}),
            ("cross_turn_comparison", "same_dimension_comparison", {
                "comparison_dimension": "response to mistakes",
                "left": "A asks for help", "right": "D proposes a meeting",
                "left_turn_ids": ["D095"], "right_turn_ids": ["D079"]}),
            ("relationship_or_decision", "adopted_decision", {
                "adopted_decision": "restart practice",
                "considered_alternative": "add focus drills",
                "turn_ids": ["D064", "D065"]}),
            ("causal", "dialogue_specific_cause", {
                "dialogue_specific_cause": "A missed the kick",
                "competing_causes": ["the missed kick", "the new drill plan"],
                "turn_ids": ["D063", "D064"]}),
            ("cross_context_synthesis", "cross_context_synthesis", {
                "synthesis": "A applies advice in two settings",
                "evidence_groups": [
                    {"group": "practice", "turn_ids": ["D063"]},
                    {"group": "report", "turn_ids": ["D095"]},
                ]}),
        ]
        for kind, variable, contract in fixtures:
            with self.subTest(kind=kind):
                refs = []
                for value in contract.values():
                    if isinstance(value, list):
                        if value and isinstance(value[0], dict):
                            refs.extend(ref for group in value
                                        for ref in group.get("turn_ids", []))
                        elif value and all(str(row).startswith("D") for row in value):
                            refs.extend(value)
                refs = list(dict.fromkeys(refs))
                row = self._question(
                    kind=kind, variable=variable, refs=refs,
                    contract=contract)
                if kind == "state_change":
                    row["question"] = (
                        "After a proposal is made, what was the plan's new state?")
                    row["options"] = {
                        "A": "committed", "B": "uncertain",
                        "C": "cancelled", "D": "deferred",
                    }
                    row["gold_text"] = "committed"
                elif kind == "task_dependency_or_constraint":
                    row["question"] = (
                        "What had to happen before submit the corrected report?")
                    row["options"] = {
                        "A": "review the figures", "B": "cancel the report",
                        "C": "leave work", "D": "start a new report",
                    }
                    row["gold_text"] = "review the figures"
                result = validate_structural_writer_output(
                    self._envelope([row]), expanded_dialogue=self.expanded)
                self.assertEqual(result["structural_valid_count"], 1,
                                 result["items"])

    def test_more_than_twelve_and_source_scope_mismatch(self):
        questions = []
        for index in range(1, 14):
            row = self._question(candidate_id=f"Q{index:02d}")
            row["question"] = f"Which order is shown in case {index}?"
            questions.append(row)
        result = validate_structural_writer_output(
            self._envelope(questions), expanded_dialogue=self.expanded)
        self.assertEqual(result["structural_valid_count"], 12)
        self.assertIn("candidate_above_capacity_max",
                      result["items"][12]["structural_errors"])
        row = self._question()
        row["source_scope"] = "parent_only"
        result = validate_structural_writer_output(
            self._envelope([row]), expanded_dialogue=self.expanded)
        self.assertIn("source_scope_mismatch",
                      result["items"][0]["structural_errors"])

    def test_frozen_live_quarantine_inventory_and_visibility_shadow(self):
        frozen = json.loads((
            LIVE_V19 / "qa/writer_structural_validation.json").read_text())
        invalid = [row for row in frozen["items"] if not row["valid"]]
        self.assertEqual(len(invalid), 9)
        counts = {}
        for row in invalid:
            for error in row["structural_errors"]:
                counts[error] = counts.get(error, 0) + 1
        self.assertEqual(counts, {
            "type_contract_keys": 2,
            "discriminative_variable": 6,
            "source_scope_mismatch": 3,
            "evidence_duplicate_refs": 1,
            "evidence_over_limit": 4,
        })
        self.assertNotIn("supporting_evidence_mismatch", counts)
        self.assertNotIn("gold_text_binding", counts)

        raw = json.loads((LIVE_V19 / "qa/writer_output_raw.json").read_text())
        raw["structural_qa_schema_version"] = STRUCTURAL_QA_SCHEMA_VERSION
        shadow = validate_structural_writer_output(
            raw, expanded_dialogue=self.expanded)
        q07 = next(row for row in shadow["items"]
                   if row["candidate_id"] == "Q07")
        self.assertIn(
            "visibility_question_not_knowledge_only",
            q07["structural_errors"])

    def test_prompt_and_protected_integrity(self):
        prompt = build_structural_qa_prompt(self.expanded, self.overview)
        self.assertIn("capacity ceiling, never a target", prompt["prompt"])
        self.assertNotIn('"questions":[]', prompt["prompt"])
        self.assertNotIn('"questions": []', prompt["prompt"])
        self.assertIn("minimal sufficient", prompt["prompt"])
        self.assertNotIn("must generate 12", prompt["prompt"].casefold())
        self.assertNotIn("as many", prompt["prompt"].casefold())
        self.assertNotIn("fill the quota", prompt["prompt"].casefold())
        tail = prompt["prompt"][-500:]
        self.assertIn("FINAL TASK", tail)
        self.assertIn("Return zero only when", tail)
        self.assertNotIn("{", tail)
        self.assertNotIn('"questions"', tail)
        repo = Path(__file__).resolve().parents[4]
        self.assertTrue(protected_manifests(repo)["unchanged"])

    def test_explicit_input_override_preflight_is_zero_api(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = preflight(
                expanded_path=LIVE_MEMORY_EXPANSION / "expanded_dialogue.json",
                memory_overview_path=(
                    LIVE_MEMORY_EXPANSION /
                    "memory/post_generation_overview.json"),
                run_root=Path(temporary) / "new-qa-run",
                input_override=True,
            )
        self.assertEqual(report["decision"], "pass", report)
        self.assertEqual(report["api_calls"], 0)
        self.assertTrue(report["details"]["input_override"])
        self.assertEqual(report["checks"]["complete_dialogue_frozen"], True)


if __name__ == "__main__":
    unittest.main()
