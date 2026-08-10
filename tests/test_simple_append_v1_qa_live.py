import json
import copy
import tempfile
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.backends import BackendResponse
from scripts.current_method.restart_n5_spell.ledger import RequestLedger
from scripts.current_method.restart_n5_spell.simple_append_v1.qa import (
    MCQ_BATCH_SCHEMA_VERSION,
    MCQ_FOCUS_WRITER_SCHEMA_VERSION,
    QUESTION_TYPES,
    assess_option_surface_symmetry,
    build_evidence_alias_registry,
    retain_focus_writer_overflow,
    classify_balanced_no_context,
    classify_full_context_frontier,
    evaluate_qa_quota_status,
    materialize_solver_batch,
    materialize_solver_candidate,
    partition_mcq_candidates,
    normalize_focus_writer_batch,
    evaluate_focus_round_profile,
    permuted_options,
    resolve_evidence_turn_refs,
    validate_mcq_batch,
)
from scripts.current_method.restart_n5_spell.simple_append_v1.qa_live import (
    DEFAULT_API_CONFIG,
    DEFAULT_SOLVER_MODEL,
    DEFAULT_SOURCE_RUN,
    DEFAULT_TOKENIZER,
    LocalSolverRunner,
    build_solver_source,
    preflight,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(row["content"] for row in messages) + "\n<think>\n\n</think>\n\n"

    def encode(self, text, add_special_tokens=False):
        return list(text.encode())


class FakeSolver:
    model_identifier = "fake-solver"
    temperature = 0.7
    top_p = 0.9

    def generate(self, request):
        return BackendResponse("<box>A</box>", self.model_identifier,
                               {"sampling_seed": request.sampling_seed})


def batch_fixture(expanded, count=6):
    aliases = build_evidence_alias_registry(expanded)
    parent_ref = aliases["turn_ref_to_alias"][expanded["turns"][0]["turn_ref"]]
    child_ref = aliases["turn_ref_to_alias"][expanded["turns"][-1]["turn_ref"]]
    ordinary_types = sorted(QUESTION_TYPES - {"cross_context_synthesis"})
    return {"mcq_batch_schema_version": MCQ_BATCH_SCHEMA_VERSION,
            "questions": [{
                "question": f"Fixture question {index + 1}?",
                "question_type": ("cross_context_synthesis" if count == 1 else
                                  ordinary_types[index % len(ordinary_types)]),
                "options": {"A": f"correct {index}", "B": f"wrong b {index}",
                            "C": f"wrong c {index}", "D": f"wrong d {index}"},
                "correct_option": "A",
                "evidence_turn_refs": [
                    parent_ref, child_ref] if count == 1 else [
                    child_ref if index else parent_ref],
            } for index in range(count)]}


def orders(answers, rewards):
    return [{"order_id": f"o{index}", "gold_letter": letter,
             "valid_answer_letters": list(answers[index]),
             "valid_rewards": list(rewards[index])}
            for index, letter in enumerate("ABCD")]


def focus_writer_role(index):
    return (
        "current_round_only" if index < 2 else
        "current_round_joint_history" if index < 4 else
        "parent_child" if index < 6 else
        "general"
    )


class SimpleAppendQALiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expanded = json.loads(
            (DEFAULT_SOURCE_RUN / "expanded_dialogue.json").read_text())
        cls.source = build_solver_source(cls.expanded)

    def test_source_adapter_preserves_complete_parent_child(self):
        self.assertEqual(108, len(self.source["combined_dialogue"]))
        partitions = list(self.source["partition_by_turn"].values())
        self.assertEqual(60, partitions.count("original_parent"))
        self.assertEqual(48, partitions.count("promoted_child"))
        self.assertEqual(108, len(self.source["span_catalog"]["spans"]))

    def test_evidence_alias_registry_is_complete_and_preserves_boundary(self):
        aliases = build_evidence_alias_registry(self.expanded)
        self.assertEqual(108, aliases["turn_count"])
        self.assertEqual(60, aliases["parent_count"])
        self.assertEqual(48, aliases["child_count"])
        self.assertEqual("cc-train-episode-61555:P01:T001",
                         aliases["alias_to_turn_ref"]["D001"])
        self.assertEqual("cc-train-episode-61555:P05:T016",
                         aliases["alias_to_turn_ref"]["D060"])
        self.assertEqual("R01:S01:T001",
                         aliases["alias_to_turn_ref"]["D061"])
        self.assertEqual("R01:S03:T016",
                         aliases["alias_to_turn_ref"]["D108"])

    def test_alias_resolution_is_exact_and_never_repairs_malformed_ids(self):
        resolved = resolve_evidence_turn_refs(
            ["D061", "D108", "cc-train-episode-61555:P03:S03:T014"],
            expanded_dialogue=self.expanded)
        self.assertEqual(["R01:S01:T001", "R01:S03:T016"],
                         resolved["resolved_turn_refs"])
        self.assertEqual(["cc-train-episode-61555:P03:S03:T014"],
                         resolved["invalid_evidence_refs"])

    def test_program_reassigns_answer_letters(self):
        raw = batch_fixture(self.expanded)
        materialized = materialize_solver_batch(
            raw, expanded_dialogue=self.expanded,
            span_catalog=self.source["span_catalog"])
        self.assertEqual(6, len(materialized))
        self.assertGreater(len({row["reference_answer_letter"]
                                for row in materialized}), 1)
        for row in materialized:
            self.assertEqual(row["canonical_answer_text"],
                             row["options"][row["reference_answer_letter"]])
            self.assertTrue(row["writer_correct_option_ignored_after_text_binding"])

    def test_v3_candidate_pool_contract_enforces_scope_and_diversity(self):
        raw = batch_fixture(self.expanded, count=12)
        aliases = build_evidence_alias_registry(self.expanded)
        parent_ref = "D001"
        child_ref = "D108"
        for index, row in enumerate(raw["questions"]):
            if index < 2:
                row["question_type"] = "cross_context_synthesis"
                row["evidence_turn_refs"] = [parent_ref, child_ref]
            elif index < 6:
                row["evidence_turn_refs"] = [child_ref]
        result = validate_mcq_batch(
            raw, expanded_dialogue=self.expanded,
            minimum_count=8, maximum_count=12,
            min_child_supported=4, min_joint_parent_child=2,
            min_distinct_question_types=4)
        self.assertTrue(result["valid"], result)
        self.assertGreaterEqual(result["child_supported_question_count"], 4)
        self.assertGreaterEqual(result["joint_parent_child_question_count"], 2)
        self.assertGreaterEqual(result["question_type_count"], 4)

    def test_reasoning_type_and_source_scope_are_independent(self):
        raw = batch_fixture(self.expanded, count=1)
        raw["questions"][0]["question_type"] = "state_change"
        result = validate_mcq_batch(raw, expanded_dialogue=self.expanded)
        self.assertTrue(result["valid"], result)
        self.assertEqual(["parent_child"], result["source_scopes"])
        materialized = materialize_solver_batch(
            raw, expanded_dialogue=self.expanded,
            span_catalog=self.source["span_catalog"])
        self.assertEqual("state_change", materialized[0]["question_type"])
        self.assertEqual("parent_child", materialized[0]["source_scope"])

    def test_candidate_pool_above_twelve_is_rejected_without_truncation(self):
        raw = batch_fixture(self.expanded, count=14)
        parent_ref = "D001"
        child_ref = "D108"
        for index, row in enumerate(raw["questions"]):
            if index < 2:
                row["question_type"] = "cross_context_synthesis"
                row["evidence_turn_refs"] = [parent_ref, child_ref]
            elif index < 6:
                row["evidence_turn_refs"] = [child_ref]
        result = validate_mcq_batch(
            raw, expanded_dialogue=self.expanded,
            minimum_count=8, maximum_count=12,
            min_child_supported=4, min_joint_parent_child=2,
            min_distinct_question_types=4)
        self.assertFalse(result["valid"], result)
        self.assertIn("mcq_batch_question_count_above_maximum", result["errors"])
        self.assertEqual(14, result["question_count"])

    def test_option_surface_diagnostics_are_visible_but_nonblocking(self):
        value = assess_option_surface_symmetry({
            "A": "A balanced explanation based on the recorded events",
            "B": "This option is only a generic opposite",
            "C": "Another balanced explanation based on a different event",
            "D": "An unrelated answer with no comparable factual basis",
        }, "A")
        self.assertIn("asymmetric_giveaway_qualifier", value["warnings"])
        self.assertFalse(value["blocking"])

    def test_surplus_tail_is_retained_for_solver(self):
        raw = batch_fixture(self.expanded, count=13)
        value = {
            "mcq_focus_writer_schema_version":
                MCQ_FOCUS_WRITER_SCHEMA_VERSION,
            "questions": [{**row, "coverage_role": focus_writer_role(index)}
                          for index, row in enumerate(raw["questions"])],
        }
        original = copy.deepcopy(value)
        retained = retain_focus_writer_overflow(value)
        self.assertFalse(retained["applied"], retained)
        self.assertEqual("surplus_retained_for_solver", retained["reason"])
        self.assertEqual(13, len(retained["value"]["questions"]))
        self.assertEqual(13, retained["retained_question_count"])
        self.assertEqual(1, retained["overflow_question_count"])
        self.assertEqual(original, value)

    def test_surplus_roles_are_validated_over_the_complete_batch(self):
        raw = batch_fixture(self.expanded, count=13)
        value = {
            "mcq_focus_writer_schema_version":
                MCQ_FOCUS_WRITER_SCHEMA_VERSION,
            "questions": [{**row, "coverage_role": "general"}
                          for row in raw["questions"]],
        }
        retained = retain_focus_writer_overflow(value)
        self.assertFalse(retained["applied"])
        self.assertEqual("surplus_retained_for_solver", retained["reason"])
        normalized = normalize_focus_writer_batch(retained["value"])
        self.assertFalse(normalized["valid"])
        self.assertIn(
            "mcq_focus_writer_current_round_only_below_minimum",
            normalized["errors"])

    def test_twenty_eight_paid_candidates_all_reach_partition(self):
        raw = batch_fixture(self.expanded, count=28)
        value = {
            "mcq_focus_writer_schema_version":
                MCQ_FOCUS_WRITER_SCHEMA_VERSION,
            "questions": [{**row, "coverage_role": focus_writer_role(index)}
                          for index, row in enumerate(raw["questions"])],
        }
        retained = retain_focus_writer_overflow(value)
        normalized = normalize_focus_writer_batch(retained["value"])
        self.assertTrue(normalized["valid"], normalized)
        self.assertEqual(28, len(normalized["batch"]["questions"]))
        partition = partition_mcq_candidates(
            normalized["batch"], expanded_dialogue=self.expanded,
            minimum_count=8, maximum_count=None,
            min_child_supported=0, min_joint_parent_child=0)
        self.assertTrue(partition["envelope_valid"], partition)
        self.assertEqual(28, partition["solver_ready_count"])

    def test_pilot2_shadow_exceeds_current_candidate_range(self):
        path = Path(
            "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
            "conversation_chronicles_episode_61555_simple_append_v1_"
            "qa_only_pilot_2/qa/writer_output.json")
        raw = copy.deepcopy(json.loads(path.read_text()))
        raw["mcq_batch_schema_version"] = MCQ_BATCH_SCHEMA_VERSION
        for row in raw["questions"]:
            if row["question_type"] == "cross_parent_child":
                row["question_type"] = "cross_context_synthesis"
        result = validate_mcq_batch(
            raw, expanded_dialogue=self.expanded,
            minimum_count=8, maximum_count=12,
            min_child_supported=4, min_joint_parent_child=2,
            min_distinct_question_types=4)
        self.assertFalse(result["valid"], result)
        self.assertIn("mcq_batch_question_count_above_maximum", result["errors"])
        self.assertEqual(14, result["question_count"])
        self.assertEqual(10, result["source_scope_counts"]["parent_child"])

    def test_batch_scope_quota_has_no_per_item_binding(self):
        raw = batch_fixture(self.expanded, count=8)
        parent_ref = "D001"
        child_ref = "D108"
        for index, row in enumerate(raw["questions"]):
            row["evidence_turn_refs"] = (
                [parent_ref] if index < 2 else
                [child_ref] if index < 6 else [parent_ref, child_ref])
        result = partition_mcq_candidates(raw, expanded_dialogue=self.expanded)
        self.assertTrue(result["envelope_valid"], result)
        self.assertTrue(result["batch_profile_ready"], result)
        self.assertEqual(6, result["child_supported_count"])
        self.assertEqual(2, result["parent_child_count"])
        raw["questions"][6]["evidence_turn_refs"] = [child_ref]
        raw["questions"][7]["evidence_turn_refs"] = [child_ref]
        result = partition_mcq_candidates(raw, expanded_dialogue=self.expanded)
        self.assertFalse(result["batch_profile_ready"])
        self.assertIn("parent_child_below_minimum",
                      result["batch_profile_errors"])

    def test_focus_profile_counts_only_exact_current_round_evidence(self):
        raw = batch_fixture(self.expanded, count=8)
        parent_ref = "D001"
        current_ref = "D108"
        for index, row in enumerate(raw["questions"]):
            row["evidence_turn_refs"] = (
                [current_ref] if index < 2 else
                [parent_ref, current_ref] if index < 6 else [parent_ref])
        envelope = normalize_focus_writer_batch({
            "mcq_focus_writer_schema_version":
                MCQ_FOCUS_WRITER_SCHEMA_VERSION,
            "questions": [{
                **row,
                "coverage_role": focus_writer_role(index),
            } for index, row in enumerate(raw["questions"])],
        })
        partition = partition_mcq_candidates(
            raw, expanded_dialogue=self.expanded)
        focus = evaluate_focus_round_profile(
            partition, expanded_dialogue=self.expanded,
            focus_round_index=1,
            pool_assignments=envelope["pool_assignments"])
        self.assertTrue(focus["focus_round_profile_ready"], focus)
        self.assertEqual(6, focus["focus_round_supported_count"])
        self.assertEqual(4, focus["focus_round_joint_with_history_count"])

    def test_focus_writer_pool_label_cannot_replace_current_evidence(self):
        raw = batch_fixture(self.expanded, count=8)
        parent_ref = "D001"
        current_ref = "D108"
        for index, row in enumerate(raw["questions"]):
            row["evidence_turn_refs"] = (
                [parent_ref] if index == 0 else
                [current_ref] if index == 1 else
                [parent_ref, current_ref] if index < 6 else [parent_ref])
        envelope = normalize_focus_writer_batch({
            "mcq_focus_writer_schema_version":
                MCQ_FOCUS_WRITER_SCHEMA_VERSION,
            "questions": [{
                **row,
                "coverage_role": focus_writer_role(index),
            } for index, row in enumerate(raw["questions"])],
        })
        partition = partition_mcq_candidates(
            raw, expanded_dialogue=self.expanded)
        focus = evaluate_focus_round_profile(
            partition, expanded_dialogue=self.expanded,
            focus_round_index=1,
            pool_assignments=envelope["pool_assignments"])
        self.assertTrue(focus["focus_round_profile_ready"], focus)
        self.assertIn("Q01:current_round_only_missing_focus_evidence",
                      focus["coverage_role_warnings"])

    def test_focus_writer_four_role_envelope_flattens_without_trusting_labels(self):
        raw = batch_fixture(self.expanded, count=8)
        value = {
            "mcq_focus_writer_schema_version":
                MCQ_FOCUS_WRITER_SCHEMA_VERSION,
            "questions": [{
                **row,
                "coverage_role": focus_writer_role(index),
            } for index, row in enumerate(raw["questions"])],
        }
        normalized = normalize_focus_writer_batch(value)
        self.assertTrue(normalized["valid"], normalized)
        self.assertEqual(2, normalized["current_round_only_question_count"])
        self.assertEqual(
            2, normalized["current_round_joint_history_question_count"])
        self.assertEqual(2, normalized["parent_child_question_count"])
        self.assertEqual(2, normalized["general_question_count"])
        self.assertEqual(8, len(normalized["batch"]["questions"]))

    def test_focus_writer_rejects_too_few_current_round_slots(self):
        raw = batch_fixture(self.expanded, count=8)
        value = {
            "mcq_focus_writer_schema_version":
                MCQ_FOCUS_WRITER_SCHEMA_VERSION,
            "questions": [{
                **row,
                "coverage_role": (
                    "current_round_only" if index < 1 else
                    "current_round_joint_history" if index < 3 else
                    "parent_child" if index < 5 else
                    "general"),
            } for index, row in enumerate(raw["questions"])],
        }
        normalized = normalize_focus_writer_batch(value)
        self.assertFalse(normalized["valid"], normalized)
        self.assertIn("mcq_focus_writer_current_round_only_below_minimum",
                      normalized["errors"])

    def test_live3_shadow_single_array_removes_format_failure_but_audits_roles(self):
        run = Path(
            "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
            "conversation_chronicles_episode_61555_simple_append_v1_"
            "round_04_qa_focus_live_3")
        malformed = json.loads((run / "qa/writer_output.json").read_text())
        nested = malformed["current_round_only_questions"]
        self.assertEqual("current_round_joint_history_questions", nested[8])
        self.assertEqual("general_questions", nested[9][4])
        questions = (
            [{**row, "coverage_role": "current_round_only"}
             for row in nested[:8]] +
            [{**row, "coverage_role": "current_round_joint_history"}
             for row in nested[9][:4]] +
            [{**row, "coverage_role": "parent_child"}
             for row in nested[9][5][:2]] +
            [{**row, "coverage_role": "general"}
             for row in nested[9][5][2:]])
        normalized = normalize_focus_writer_batch({
            "mcq_focus_writer_schema_version":
                MCQ_FOCUS_WRITER_SCHEMA_VERSION,
            "questions": questions,
        })
        self.assertTrue(normalized["valid"], normalized)
        expanded = json.loads((Path(
            "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
            "conversation_chronicles_episode_61555_simple_append_v1_"
            "round_04_live_3/round_04/expanded_dialogue.json")).read_text())
        partition = partition_mcq_candidates(
            normalized["batch"], expanded_dialogue=expanded,
            minimum_count=8, maximum_count=None,
            min_child_supported=4, min_joint_parent_child=2)
        focus = evaluate_focus_round_profile(
            partition, expanded_dialogue=expanded, focus_round_index=4,
            pool_assignments=normalized["pool_assignments"])
        self.assertGreaterEqual(focus["focus_round_supported_count"], 4)
        self.assertGreaterEqual(
            focus["focus_round_joint_with_history_count"], 2)
        self.assertTrue(focus["focus_round_profile_ready"], focus)
        self.assertIn(
            "Q12:current_round_joint_history_missing_history_side",
            focus["coverage_role_warnings"])
        self.assertIn(
            "Q13:parent_child_missing_two_sided_evidence",
            focus["coverage_role_warnings"])

    def test_live5_shadow_separates_history_joint_from_parent_child(self):
        run = Path(
            "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
            "conversation_chronicles_episode_61555_simple_append_v1_"
            "round_04_qa_focus_live_5")
        value = json.loads((run / "qa/writer_output.json").read_text())
        value["mcq_focus_writer_schema_version"] = \
            MCQ_FOCUS_WRITER_SCHEMA_VERSION
        old_envelope = normalize_focus_writer_batch(value)
        self.assertFalse(old_envelope["valid"], old_envelope)
        self.assertIn("mcq_focus_writer_parent_child_below_minimum",
                      old_envelope["errors"])

        reclassified = copy.deepcopy(value)
        general = [row for row in reclassified["questions"]
                   if row["coverage_role"] == "general"]
        for row in general[:2]:
            row["coverage_role"] = "parent_child"
        envelope = normalize_focus_writer_batch(reclassified)
        self.assertTrue(envelope["valid"], envelope)
        expanded = json.loads((Path(
            "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
            "conversation_chronicles_episode_61555_simple_append_v1_"
            "round_04_live_3/round_04/expanded_dialogue.json")).read_text())
        partition = partition_mcq_candidates(
            envelope["batch"], expanded_dialogue=expanded,
            minimum_count=8, maximum_count=None,
            min_child_supported=4, min_joint_parent_child=2)
        focus = evaluate_focus_round_profile(
            partition, expanded_dialogue=expanded, focus_round_index=4,
            pool_assignments=envelope["pool_assignments"])
        self.assertGreaterEqual(
            focus["focus_round_joint_with_history_count"], 2)
        self.assertFalse(partition["batch_profile_ready"])
        self.assertIn("parent_child_below_minimum",
                      partition["batch_profile_errors"])
        self.assertTrue(any(
            "parent_child_missing_two_sided_evidence" in error
            for error in focus["writer_pool_contract_errors"]))

    def test_quota_deficit_requests_top_up_without_blocking_solver(self):
        quota = evaluate_qa_quota_status(
            {"batch_profile_errors": ["parent_child_below_minimum"]},
            {"focus_round_profile_errors": [
                "current_round_joint_with_history_below_minimum"]})
        self.assertTrue(quota["top_up_required"])
        self.assertFalse(quota["quota_profile_ready"])
        self.assertFalse(quota["quota_deficits_block_solver"])
        self.assertTrue(quota["existing_solver_candidates_preserved"])

    def test_live6_shadow_passes_actual_quotas_despite_role_warning(self):
        run = Path(
            "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
            "conversation_chronicles_episode_61555_simple_append_v1_"
            "round_04_qa_focus_live_6")
        value = json.loads((run / "qa/writer_output.json").read_text())
        envelope = normalize_focus_writer_batch(value)
        self.assertTrue(envelope["valid"], envelope)
        expanded = json.loads((Path(
            "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
            "conversation_chronicles_episode_61555_simple_append_v1_"
            "round_04_live_3/round_04/expanded_dialogue.json")).read_text())
        partition = partition_mcq_candidates(
            envelope["batch"], expanded_dialogue=expanded,
            minimum_count=8, maximum_count=None,
            min_child_supported=4, min_joint_parent_child=2)
        focus = evaluate_focus_round_profile(
            partition, expanded_dialogue=expanded, focus_round_index=4,
            pool_assignments=envelope["pool_assignments"])
        quota = evaluate_qa_quota_status(partition, focus)
        self.assertTrue(partition["batch_profile_ready"], partition)
        self.assertTrue(focus["focus_round_profile_ready"], focus)
        self.assertIn(
            "Q04:current_round_joint_history_missing_focus_side",
            focus["coverage_role_warnings"])
        self.assertTrue(quota["quota_profile_ready"])
        self.assertFalse(quota["top_up_required"])
        self.assertEqual(8, partition["solver_ready_count"])

    def test_live6_solver_replay_preflight_is_zero_api(self):
        source = Path(
            "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
            "conversation_chronicles_episode_61555_simple_append_v1_"
            "round_04_live_3/round_04")
        writer = Path(
            "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
            "conversation_chronicles_episode_61555_simple_append_v1_"
            "round_04_qa_focus_live_6/qa/writer_output.json")
        with tempfile.TemporaryDirectory() as temporary:
            replay_root = Path(temporary) / "replay"
            result = preflight(
                source_run=source, run_root=replay_root,
                api_config_path=DEFAULT_API_CONFIG,
                tokenizer_path=DEFAULT_TOKENIZER,
                solver_model=DEFAULT_SOLVER_MODEL,
                focus_round_index=4,
                writer_output_replay=writer)
            self.assertEqual("pass", result["decision"], result)
            self.assertEqual(0, result["api_calls"])
            self.assertEqual(
                8, result["details"]["writer_replay_solver_ready_count"])
            self.assertFalse(replay_root.exists())

    def test_pilot3_shadow_detects_excess_parent_only_and_duplicates(self):
        path = Path(
            "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
            "conversation_chronicles_episode_61555_simple_append_v1_"
            "qa_only_pilot_3/qa/writer_output.json")
        raw = copy.deepcopy(json.loads(path.read_text()))
        raw["mcq_batch_schema_version"] = MCQ_BATCH_SCHEMA_VERSION
        result = validate_mcq_batch(
            raw, expanded_dialogue=self.expanded,
            minimum_count=12, maximum_count=12,
            min_child_supported=4, min_joint_parent_child=2,
            min_distinct_question_types=4)
        self.assertFalse(result["valid"])
        self.assertIn("mcq_batch_question_count_above_maximum", result["errors"])
        self.assertIn("mcq_batch_missing_child_supported_question", result["errors"])
        self.assertTrue(any("duplicate_question" in error
                            for error in result["errors"]))

    def test_pilot4_candidate_partition_keeps_evidence_defects_solver_ready(self):
        path = Path(
            "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
            "conversation_chronicles_episode_61555_simple_append_v1_"
            "qa_only_pilot_4/qa/writer_output.json")
        raw = json.loads(path.read_text())
        raw["mcq_batch_schema_version"] = MCQ_BATCH_SCHEMA_VERSION
        for row in raw["questions"]:
            row.pop("candidate_slot_id", None)
        partition = partition_mcq_candidates(
            raw, expanded_dialogue=self.expanded)
        self.assertTrue(partition["envelope_valid"], partition)
        self.assertEqual(12, partition["solver_ready_count"])
        self.assertEqual(2, partition["evidence_ready_count"])
        self.assertEqual(10, partition["evidence_issue_count"])
        qa = materialize_solver_candidate(
            raw["questions"][1], index=2,
            expanded_dialogue=self.expanded,
            span_catalog=self.source["span_catalog"],
            candidate_status=partition["items"][1])
        self.assertTrue(qa["evidence_issues"])
        self.assertFalse(qa["evidence_ready"])

    def test_pilot5_shadow_preserves_failure_and_alias_fixture_closes_quota(self):
        path = Path(
            "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
            "conversation_chronicles_episode_61555_simple_append_v1_"
            "qa_only_pilot_5/qa/writer_output.json")
        historical = json.loads(path.read_text())
        historical["mcq_batch_schema_version"] = MCQ_BATCH_SCHEMA_VERSION
        original = partition_mcq_candidates(
            historical, expanded_dialogue=self.expanded)
        self.assertEqual(0, original["child_supported_count"])
        self.assertEqual(0, original["parent_child_count"])
        self.assertIn("evidence_reference_invalid",
                      original["items"][6]["evidence_issues"])

        # Explicit shadow fixture only: demonstrate the exact aliases a future
        # Writer can emit. This does not repair or rewrite the historical run.
        shadow = copy.deepcopy(historical)
        aliases = build_evidence_alias_registry(self.expanded)["turn_ref_to_alias"]
        for row in shadow["questions"]:
            row["evidence_turn_refs"] = [
                aliases.get(ref, ref) for ref in row["evidence_turn_refs"]]
        shadow["questions"][6]["evidence_turn_refs"] = ["D094", "D106"]
        shadow["questions"][7]["evidence_turn_refs"] = ["D049", "D106"]
        shadow["questions"][10]["evidence_turn_refs"] = ["D054", "D099"]
        shadow["questions"][11]["evidence_turn_refs"] = [
            "D029", "D045", "D054", "D106"]
        replay = partition_mcq_candidates(
            shadow, expanded_dialogue=self.expanded)
        self.assertTrue(replay["batch_profile_ready"], replay)
        self.assertEqual(4, replay["child_supported_count"])
        self.assertEqual(3, replay["parent_child_count"])
        self.assertEqual(12, replay["solver_ready_count"])

    def test_live_lower_bound_accepts_frozen_pilot6_twenty_question_pool(self):
        path = Path(
            "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
            "conversation_chronicles_episode_61555_simple_append_v1_"
            "qa_only_pilot_6/qa/writer_output.json")
        historical = json.loads(path.read_text())
        historical["mcq_batch_schema_version"] = MCQ_BATCH_SCHEMA_VERSION
        partition = partition_mcq_candidates(
            historical, expanded_dialogue=self.expanded,
            minimum_count=8, maximum_count=None)
        self.assertTrue(partition["envelope_valid"], partition)
        self.assertEqual(20, partition["solver_ready_count"])
        self.assertTrue(partition["batch_profile_ready"], partition)

    def test_structural_candidate_error_does_not_block_other_candidates(self):
        raw = batch_fixture(self.expanded, count=2)
        raw["questions"][1]["options"] = {"A": "only one option"}
        partition = partition_mcq_candidates(
            raw, expanded_dialogue=self.expanded,
            minimum_count=2, maximum_count=2,
            min_child_supported=1, min_joint_parent_child=0)
        self.assertTrue(partition["envelope_valid"])
        self.assertTrue(partition["items"][0]["solver_ready"])
        self.assertFalse(partition["items"][1]["solver_ready"])
        self.assertEqual(1, partition["solver_ready_count"])

    def test_aliases_resolve_to_canonical_ids_before_solver(self):
        raw = batch_fixture(self.expanded, count=1)
        partition = partition_mcq_candidates(
            raw, expanded_dialogue=self.expanded,
            minimum_count=1, maximum_count=1,
            min_child_supported=1, min_joint_parent_child=1)
        status = partition["items"][0]
        self.assertEqual(["D001", "D108"], status["valid_evidence_aliases"])
        self.assertEqual([
            "cc-train-episode-61555:P01:T001", "R01:S03:T016"],
            status["valid_evidence_turn_refs"])
        qa = materialize_solver_candidate(
            raw["questions"][0], index=1,
            expanded_dialogue=self.expanded,
            span_catalog=self.source["span_catalog"],
            candidate_status=status)
        self.assertEqual(status["valid_evidence_turn_refs"],
                         qa["evidence_turn_ids"])

    def test_invalid_alias_is_warning_and_does_not_block_solver(self):
        raw = batch_fixture(self.expanded, count=1)
        raw["questions"][0]["evidence_turn_refs"] = ["D999"]
        partition = partition_mcq_candidates(
            raw, expanded_dialogue=self.expanded,
            minimum_count=1, maximum_count=1,
            min_child_supported=0, min_joint_parent_child=0)
        item = partition["items"][0]
        self.assertTrue(item["solver_ready"])
        self.assertFalse(item["evidence_ready"])
        self.assertEqual(["D999"], item["invalid_evidence_turn_refs"])

    def test_k1_semantic_leakage_is_final(self):
        result = classify_balanced_no_context(
            orders((["A"], ["B"], ["C"], ["D"]),
                   ([1], [1], [1], [1])), samples_per_position=1)
        self.assertEqual("bad_ground_semantic_leakage", result["decision"])

    def test_k1_position_bias_is_final(self):
        result = classify_balanced_no_context(
            orders((["C"], ["C"], ["C"], ["C"]),
                   ([0], [0], [1], [0])), samples_per_position=1)
        self.assertEqual("no_context_position_bias", result["decision"])

    def test_k1_any_wrong_answer_passes(self):
        passed = classify_balanced_no_context(
            orders((["B"], ["A"], ["A"], ["B"]),
                   ([0], [0], [0], [1])), samples_per_position=1)
        boundary = classify_balanced_no_context(
            orders((["A"], ["B"], ["A"], ["B"]),
                   ([1], [1], [0], [0])), samples_per_position=1)
        self.assertEqual("pass_grounding", passed["decision"])
        self.assertEqual("pass_grounding", boundary["decision"])

    def test_k2_any_wrong_answer_passes_including_seven_of_eight(self):
        passed = classify_balanced_no_context(
            orders((["A", "B"], ["A", "A"], ["B", "D"], ["A", "B"]),
                   ([1, 0], [0, 0], [0, 1], [0, 0])), samples_per_position=2)
        seven_of_eight = classify_balanced_no_context(
            orders((["A", "B"], ["B", "A"], ["C", "A"], ["D", "B"]),
                   ([1, 0], [1, 1], [1, 1], [1, 1])), samples_per_position=2)
        self.assertEqual("pass_grounding", passed["decision"])
        self.assertEqual("pass_grounding", seven_of_eight["decision"])

    def test_full_context_spell_frontier_accepts_only_one_to_four_of_five(self):
        for correct in range(6):
            with self.subTest(correct=correct):
                result = classify_full_context_frontier(
                    [1] * correct + [0] * (5 - correct))
                self.assertEqual(f"{correct}/5", result["bucket"])
                self.assertEqual(1 <= correct <= 4, result["admitted"])
                self.assertEqual(
                    "retain_frontier_pre_audit" if 1 <= correct <= 4 else
                    "reject_easy" if correct == 5 else "reject_all_wrong",
                    result["decision"])

    def test_full_context_spell_frontier_requires_five_valid_rewards(self):
        for rewards in ([1, 0, 1], [1, 0, None, 1, 0], [True, 0, 0, 0, 0]):
            with self.subTest(rewards=rewards):
                result = classify_full_context_frontier(list(rewards))
                self.assertEqual("reject_responder_format", result["decision"])
                self.assertFalse(result["admitted"])

    def test_permutation_rotates_text_not_gold_identity(self):
        options = {"A": "one", "B": "two", "C": "three", "D": "four"}
        self.assertEqual({"A": "four", "B": "one", "C": "two", "D": "three"},
                         permuted_options(options, 1))

    def test_local_solver_runner_records_one_valid_choice(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = RequestLedger(root, "a" * 64, "local")
            runner = LocalSolverRunner(
                run_root=root, ledger=ledger, tokenizer=FakeTokenizer(),
                backend=FakeSolver(), dialogue=self.source["combined_dialogue"])
            qa = materialize_solver_batch(
                batch_fixture(self.expanded, count=1),
                expanded_dialogue=self.expanded,
                span_catalog=self.source["span_catalog"])[0]
            row = runner.one(qa=qa, options=qa["options"], mode="no_context",
                             order_id="original", slot=0)
            self.assertEqual("A", row["parsed_letter"])
            self.assertEqual(1, ledger.summary()["request_count"])

    def test_preflight_is_zero_api_and_does_not_create_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "qa-only"
            result = preflight(
                source_run=DEFAULT_SOURCE_RUN, run_root=run_root,
                api_config_path=DEFAULT_API_CONFIG,
                tokenizer_path=DEFAULT_TOKENIZER,
                solver_model=DEFAULT_SOLVER_MODEL)
            self.assertEqual("pass", result["decision"], result)
            self.assertEqual(0, result["api_calls"])
            self.assertFalse(run_root.exists())


if __name__ == "__main__":
    unittest.main()
