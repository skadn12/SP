import json
import tempfile
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.core import (
    balanced_grounding_decision,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.production_batch_v11 import (
    _candidate_paths,
    solver_guided_retry_required,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import (
    hash_tree,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.single_root_mcq_materializer import (
    build_solver_guided_full_distractor_prompt,
    build_distractor_validator,
    compile_solver_guided_variable_components,
    load_solver_guidance,
    preflight,
    solver_guided_retry_eligible,
    validate_factorial_components_with_shape_gate,
    validate_solver_guided_distractors_with_shape_gate,
)


PILOT40 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "slim_continuous_expansion_v11_end_to_end_batch_pilot_40")
PILOT40_SHA = "0e625fa4a8e57f280ae2fb709e4c56482244afbc373381c83c62ff119ba23c15"
CANDIDATE = PILOT40 / (
    "roots/cc-train-episode-61555/qa_candidates/candidate_02")
API_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def grounding_fixture(correct_count: int):
    rewards = [1] * correct_count + [0] * (12 - correct_count)
    rows = []
    for offset, letter in enumerate("ABCD"):
        chunk = rewards[offset * 3:(offset + 1) * 3]
        rows.append({
            "order_id": f"order_{letter}",
            "gold_letter": letter,
            "valid_answer_letters": ["A", "B", "C"],
            "valid_rewards": chunk,
        })
    return rows


class Pilot40SolverGuidedRecoveryTests(unittest.TestCase):
    def test_retry_is_limited_to_strict_grounding_failures(self):
        for decision in (
                "grounding_inconclusive", "bad_ground_semantic_leakage",
                "no_context_position_bias"):
            self.assertTrue(solver_guided_retry_eligible(decision))
            self.assertTrue(solver_guided_retry_required({
                "qa_pipeline_ready": False, "final_decision": decision}))
        for decision in (
                "retain_frontier", "reject_all_wrong", "bad_evidence_relevance",
                "frontier_audit_inconclusive", ""):
            self.assertFalse(solver_guided_retry_eligible(decision))
            self.assertFalse(solver_guided_retry_required({
                "qa_pipeline_ready": False, "final_decision": decision}))
        self.assertFalse(solver_guided_retry_required({
            "qa_pipeline_ready": True,
            "final_decision": "grounding_inconclusive"}))

    def test_historical_failure_loads_as_frozen_guidance(self):
        gold = load(CANDIDATE / "03_semantic_audit/qa/admitted_gold.json")
        guidance = load_solver_guidance(
            prior_mcq_run=CANDIDATE / "04_mcq_materializer",
            prior_spell_run=CANDIDATE / "05_spell_filter",
            expected_qa_id=gold["qa_id"],
            expected_question=gold["question"],
            expected_answer=gold["correct_answer_text"],
        )
        self.assertEqual("grounding_inconclusive",
                         guidance["prior_final_decision"])
        self.assertEqual(6, guidance["semantic_correct_count"])
        self.assertEqual(12, guidance["total_valid_rollouts"])
        self.assertEqual(gold["question"], guidance["prior_question"])
        self.assertEqual(set("ABCD"), set(guidance["prior_options"]))
        factors = guidance["no_context_component_selection_feedback"][
            "factor_selection_counts"]
        self.assertEqual(
            {"true_parent": 6, "false_parent": 6,
             "true_child": 12, "false_child": 0}, factors)
        self.assertEqual([
            guidance["prior_counterfactual_components"]["true_child_component"]
        ], guidance["required_shared_components"])
        self.assertEqual("child", guidance["locked_factor"])
        self.assertEqual("parent", guidance["variable_factor"])
        self.assertEqual(
            {"minimum": 8, "target": 12, "maximum": 16},
            guidance["variable_component_word_token_limits"])

    def test_guided_prompt_freezes_question_gold_and_prior_options_only(self):
        gold = load(CANDIDATE / "03_semantic_audit/qa/admitted_gold.json")
        source = load(CANDIDATE / "02_questioner/frozen/qa_source_view.json")
        catalog = load(CANDIDATE / "02_questioner/frozen/span_catalog.json")
        guidance = load_solver_guidance(
            prior_mcq_run=CANDIDATE / "04_mcq_materializer",
            prior_spell_run=CANDIDATE / "05_spell_filter",
            expected_qa_id=gold["qa_id"],
            expected_question=gold["question"],
            expected_answer=gold["correct_answer_text"],
        )
        prompt = build_solver_guided_full_distractor_prompt(
            frozen_dialogue=source["combined_dialogue"],
            span_catalog=catalog, base_qa=gold, solver_guidance=guidance)
        self.assertIn("SOLVER-GUIDED VARIABLE-COMPONENT", prompt)
        self.assertNotIn("FULL-OPTION DISTRACTOR", prompt)
        self.assertNotIn("complete independent trajectory", prompt)
        self.assertIn("not complete answers or complete two-time trajectories", prompt)
        self.assertIn(gold["question"], prompt)
        self.assertIn(guidance["prior_options"]["A"], prompt)
        self.assertIn('"semantic_correct_count":6', prompt)
        self.assertIn('"false_child":0', prompt)
        self.assertIn("no_context_component_selection_feedback", prompt)
        self.assertIn("required_shared_components", prompt)
        self.assertNotIn("raw_response", prompt)
        self.assertNotIn("request_events", prompt)
        self.assertTrue(prompt.rstrip().endswith(
            'OUTPUT ONLY {"counterfactual_variable_components":'
            '["...","...","..."]}'))

    def test_guided_validator_rejects_exact_reuse_of_prior_components(self):
        gold = load(CANDIDATE / "03_semantic_audit/qa/admitted_gold.json")
        guidance = load_solver_guidance(
            prior_mcq_run=CANDIDATE / "04_mcq_materializer",
            prior_spell_run=CANDIDATE / "05_spell_filter",
            expected_qa_id=gold["qa_id"],
            expected_question=gold["question"],
            expected_answer=gold["correct_answer_text"],
        )
        prior = guidance["prior_counterfactual_components"]
        validation = validate_factorial_components_with_shape_gate(
            {
                "counterfactual_parent_component": prior[
                    "counterfactual_parent_component"],
                "counterfactual_child_component": prior[
                    "counterfactual_child_component"],
            },
            core_qa=gold,
            span_catalog=load(
                CANDIDATE / "02_questioner/frozen/span_catalog.json"),
            placement={"answer_index": 3},
            solver_guidance=guidance,
        )
        self.assertEqual("distractor_shape_invalid", validation["decision"])
        self.assertEqual(
            ["counterfactual_parent_component",
             "counterfactual_child_component"],
            validation["solver_guided_reused_components"])

    def test_guided_full_option_validator_rejects_prior_option_set(self):
        gold = load(CANDIDATE / "03_semantic_audit/qa/admitted_gold.json")
        catalog = load(CANDIDATE / "02_questioner/frozen/span_catalog.json")
        guidance = load_solver_guidance(
            prior_mcq_run=CANDIDATE / "04_mcq_materializer",
            prior_spell_run=CANDIDATE / "05_spell_filter",
            expected_qa_id=gold["qa_id"],
            expected_question=gold["question"],
            expected_answer=gold["correct_answer_text"],
        )
        prior_component = guidance["prior_counterfactual_components"][
            "counterfactual_parent_component"]
        variable_components = [
            prior_component,
            "Co-workers B initially prioritized predictable monthly costs over premium purchases,",
            "Co-workers B initially preferred delaying expensive purchases until savings targets were met,",
        ]
        validation = validate_solver_guided_distractors_with_shape_gate(
            {"counterfactual_variable_components": variable_components},
            core_qa=gold,
            span_catalog=catalog, placement={"answer_index": 3},
            solver_guidance=guidance)
        self.assertEqual("distractor_shape_invalid", validation["decision"])
        self.assertEqual(["solver_guided_variable_component_reuses_prior"],
                         validation["reasons"])
        runtime_validator = build_distractor_validator(
            core_qa=gold, span_catalog=catalog,
            placement={"answer_index": 3}, solver_guidance=guidance)
        self.assertTrue(callable(runtime_validator))
        self.assertEqual(
            "distractor_shape_invalid",
            runtime_validator({
                "counterfactual_variable_components": variable_components
            })["decision"])

    def test_guided_full_options_must_keep_solver_dominant_true_factor(self):
        gold = load(CANDIDATE / "03_semantic_audit/qa/admitted_gold.json")
        catalog = load(CANDIDATE / "02_questioner/frozen/span_catalog.json")
        guidance = load_solver_guidance(
            prior_mcq_run=CANDIDATE / "04_mcq_materializer",
            prior_spell_run=CANDIDATE / "05_spell_filter",
            expected_qa_id=gold["qa_id"],
            expected_question=gold["question"],
            expected_answer=gold["correct_answer_text"],
        )
        compiled = compile_solver_guided_variable_components({
            "counterfactual_variable_components": [
                "Co-workers B initially prioritized predictable monthly costs over premium purchases,",
                "Co-workers B initially preferred delaying expensive purchases until savings targets were met,",
                "Co-workers B initially favored practical necessities before discretionary upgrades,",
            ]}, solver_guidance=guidance)
        self.assertEqual("solver_guided_components_compiled",
                         compiled["decision"])
        shared = guidance["required_shared_components"][0]
        self.assertTrue(all(value.endswith(shared)
                            for value in compiled["distractors"]))

    def test_variable_parent_component_cannot_include_later_clause(self):
        gold = load(CANDIDATE / "03_semantic_audit/qa/admitted_gold.json")
        guidance = load_solver_guidance(
            prior_mcq_run=CANDIDATE / "04_mcq_materializer",
            prior_spell_run=CANDIDATE / "05_spell_filter",
            expected_qa_id=gold["qa_id"],
            expected_question=gold["question"],
            expected_answer=gold["correct_answer_text"],
        )
        compiled = compile_solver_guided_variable_components({
            "counterfactual_variable_components": [
                "Co-workers B initially prioritized predictable monthly costs, later questioned the team event.",
                "Co-workers B initially preferred delaying expensive purchases until savings targets were met,",
                "Co-workers B initially favored practical necessities before discretionary upgrades,",
            ]}, solver_guidance=guidance)
        self.assertEqual("distractor_shape_invalid", compiled["decision"])
        self.assertIn("solver_guided_variable_component_contains_connector",
                      compiled["reasons"])

    def test_guided_preflight_is_zero_api_and_preserves_sources(self):
        before = hash_tree(PILOT40)["tree_sha256"]
        with tempfile.TemporaryDirectory() as temp:
            report = preflight(
                semantic_audit_run=CANDIDATE / "03_semantic_audit",
                qa_source_run=CANDIDATE / "02_questioner",
                run_root=Path(temp) / "fresh-guided-run",
                api_config_path=API_CONFIG, tokenizer_path=TOKENIZER,
                root_id="cc-train-episode-61555",
                prior_mcq_run=CANDIDATE / "04_mcq_materializer",
                prior_spell_run=CANDIDATE / "05_spell_filter",
            )
        self.assertTrue(report["valid"], report["failed_checks"])
        self.assertEqual(0, report["api_calls"])
        self.assertTrue(report["checks"]["solver_guidance_contract_valid"])
        self.assertEqual(before, hash_tree(PILOT40)["tree_sha256"])

    def test_batch_allocates_exactly_one_guided_variant(self):
        paths = _candidate_paths(Path("/tmp/batch/root"), 2)
        self.assertEqual(
            paths["base"] / "06_mcq_materializer_solver_guided",
            paths["mcq_solver_guided"])
        self.assertEqual(
            paths["base"] / "07_spell_filter_solver_guided",
            paths["spell_solver_guided"])

    def test_strict_grounding_threshold_is_unchanged(self):
        self.assertEqual(
            "pass_grounding",
            balanced_grounding_decision(grounding_fixture(4))["decision"])
        self.assertEqual(
            "grounding_inconclusive",
            balanced_grounding_decision(grounding_fixture(5))["decision"])

    def test_pilot40_remains_read_only(self):
        self.assertEqual(PILOT40_SHA, hash_tree(PILOT40)["tree_sha256"])


if __name__ == "__main__":
    unittest.main()
