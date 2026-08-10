from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1 import (
    live_canary_v11,
    live_qa_canary_1,
    production_batch_v11,
    single_root_mcq_materializer,
)
from scripts.current_method.restart_n5_spell.live_backends import ChallengerAPIConfig
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.planner_transport_v11 import (
    build_frozen_planner_transport_request_v11,
    validate_frozen_planner_request_v11,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import (
    hash_tree,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_validator_transport_v11 import (
    build_frozen_round_validator_transport_request_v11,
    validate_frozen_round_validator_request_v11,
)
from scripts.current_method.restart_n5_spell.core import sha256_file


RUNS = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
ROOTS = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-inputs/"
    "conversation_chronicles_v1/roots")
API_CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")
SOLVER = Path(
    "/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/"
    "accepted_models/solver_base_restart_stage2_merged")


class V11ProductionBatchTest(unittest.TestCase):
    @staticmethod
    def _transport_config() -> ChallengerAPIConfig:
        return ChallengerAPIConfig(
            base_url="https://example.invalid/v1", api_key="not-used",
            model="fixture-model", temperature=0.2, timeout_seconds=600,
            questioner_max_tokens=1024, enable_thinking=False)

    def test_batch_preflight_is_zero_api_and_does_not_create_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch_root = Path(tmp) / "fresh-batch"
            report = production_batch_v11.preflight_batch(
                batch_root=batch_root,
                root_ids=("cc-train-episode-216742", "cc-train-episode-41625"),
                api_config_path=API_CONFIG, tokenizer_path=TOKENIZER,
                solver_model=SOLVER)
            self.assertTrue(report["valid"], report["failed_checks"])
            self.assertEqual(0, report["api_calls"])
            self.assertEqual(0, report["network_attempts"])
            self.assertFalse(batch_root.exists())
            self.assertEqual(2, len(report["roots"]))
            self.assertEqual(
                "mark_only", production_batch_v11.EVIDENCE_MISMATCH_POLICY)
            self.assertEqual(2, report["request_budget"][
                "per_candidate_distractor_max"])
            self.assertEqual(1, report["request_budget"][
                "per_candidate_solver_guided_distractor_retry_max"])
            self.assertEqual(1, report["request_budget"][
                "solver_guided_retry_per_candidate_max"])
            self.assertEqual(
                production_batch_v11.PER_ROOT_PAID_REQUEST_LIMIT,
                report["request_budget"]["per_root_paid_max"])
            self.assertEqual(2, report["request_budget"][
                "per_candidate_questioner_max"])
            self.assertEqual(3, report["request_budget"]["qa_candidate_limit"])
            self.assertTrue(report["request_budget"]["stop_on_first_admitted"])
            self.assertEqual(18, report["request_budget"][
                "per_root_expansion_max"])

    def test_root_preflight_derives_identity_seed_and_true_tail(self):
        root_id = "cc-train-episode-216742"
        with tempfile.TemporaryDirectory() as tmp:
            report = production_batch_v11.preflight_root_expansion(
                root_id=root_id, input_path=ROOTS / root_id / "unified_root.json",
                run_root=Path(tmp) / "expansion", api_config_path=API_CONFIG,
                tokenizer_path=TOKENIZER)
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(root_id, report["root_id"])
        self.assertEqual(216742, report["orchestration_seed"])
        self.assertEqual(7, len(report["parent_tail_required"]))
        self.assertTrue(all(ref.startswith(root_id + ":")
                            for ref in report["parent_tail_required"]))
        self.assertEqual(600, report["api_configuration"]["timeout_seconds"])
        self.assertEqual(
            600, report["realizer_transport_configuration"][
                "read_timeout_seconds"])
        self.assertFalse(report["realizer_transport_configuration"]["streaming"])
        self.assertEqual(
            0, report["realizer_transport_configuration"][
                "automatic_retry_limit"])

    def test_live_realizer_fallback_uses_long_read_timeout(self):
        config = live_canary_v11.live_api_configuration_v11(API_CONFIG)
        self.assertEqual(600, config.timeout_seconds)

    def test_duplicate_root_does_not_form_a_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = production_batch_v11.preflight_batch(
                batch_root=Path(tmp) / "batch",
                root_ids=("cc-train-episode-216742", "cc-train-episode-216742"),
                api_config_path=API_CONFIG, tokenizer_path=TOKENIZER,
                solver_model=SOLVER)
        self.assertFalse(report["valid"])
        self.assertIn("at_least_two_distinct_roots", report["failed_checks"])

    def test_expansion_runtime_exposes_root_parameters(self):
        parameters = inspect.signature(
            live_canary_v11.execute_live_canary).parameters
        for name in ("root_id", "orchestration_seed", "run_id",
                     "parent_tail_refs"):
            self.assertIn(name, parameters)

    def test_non_61555_seed_is_frozen_by_planner_fingerprint(self):
        request = build_frozen_planner_transport_request_v11(
            config=self._transport_config(), prompt="fixture",
            authoritative_input_sha256="a" * 64,
            orchestration_seed=216742,
            prompt_schema_contract_sha256="b" * 64)
        validation = validate_frozen_planner_request_v11(request)
        self.assertTrue(validation["valid"], validation["checks"])

    def test_non_61555_seed_is_frozen_by_validator_fingerprint(self):
        request = build_frozen_round_validator_transport_request_v11(
            config=self._transport_config(), prompt="fixture",
            authoritative_input_sha256="a" * 64,
            orchestration_seed=41625,
            prompt_schema_contract_sha256="b" * 64)
        validation = validate_frozen_round_validator_request_v11(request)
        self.assertTrue(validation["valid"], validation["checks"])

    def test_progress_uses_configured_parent_tail(self):
        tail = ["other-root:P05:T010", "other-root:P05:T011"]
        result = {
            "compiled_round_plan": {"scenes": [{"scene_ref": "R01:S01"}]},
            "realized": [{"dialogue_turns": [{"text": "x"}]}],
            "contexts": {"R01:S01": {
                "parent_true_tail_refs": tail,
                "same_round_predecessor_refs": [],
                "same_round_scene_text_included": False,
            }},
        }
        progress = live_canary_v11.execution_progress_v11(
            result, parent_tail_refs=tail)
        self.assertTrue(progress["parent_tail_retained"])

    def test_questioner_preflight_accepts_dynamic_counts(self):
        source = RUNS / (
            "conversation_chronicles_episode_61555_slim_continuous_"
            "expansion_v11_live_canary_34")
        expanded = source / "expanded_dialogue_round_1.json"
        with tempfile.TemporaryDirectory() as tmp:
            report = live_qa_canary_1.preflight(
                source_run=source, run_root=Path(tmp) / "qa",
                api_config=API_CONFIG, tokenizer_path=TOKENIZER,
                expected_root_id="cc-train-episode-61555",
                expected_source_tree_sha256=hash_tree(source)["tree_sha256"],
                expected_expanded_file_sha256=sha256_file(expanded),
                expected_parent_turn_count=60, expected_child_turn_count=32)
        self.assertTrue(report["valid"], report["errors"])
        self.assertTrue(report["checks"]["parent_child_counts_consistent"])

    def test_child_evidence_coverage_is_soft_batch_metadata(self):
        rows = [{"qa_candidates": [
            {"status": "admitted", "qa": {
                "evidence_scope": "parent_only",
                "readiness": {"core_qa_ready": True}}},
            {"status": "quarantined", "qa": {
                "evidence_scope": "child_only",
                "readiness": {"core_qa_ready": True}}},
        ]}]
        coverage = production_batch_v11.summarize_evidence_scope_coverage(rows)
        self.assertEqual(1, coverage["evidence_scope_counts"]["parent_only"])
        self.assertEqual(1, coverage["evidence_scope_counts"]["child_only"])
        self.assertEqual(0.5, coverage["child_evidence_coverage_ratio"])
        self.assertFalse(coverage["child_evidence_coverage_low"])
        self.assertFalse(
            coverage["child_evidence_coverage_is_admission_gate"])
        self.assertTrue(production_batch_v11.candidate_generation_should_stop(
            rows[0]["qa_candidates"][0]))

    def test_mcq_materializer_preflight_is_generic_and_zero_api(self):
        audit = RUNS / (
            "conversation_chronicles_episode_61555_slim_v11_canary34_"
            "qa_semantic_audit_2")
        qa = RUNS / (
            "conversation_chronicles_episode_61555_slim_v11_canary34_"
            "qa_canary_5")
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "mcq"
            report = single_root_mcq_materializer.preflight(
                semantic_audit_run=audit, qa_source_run=qa,
                run_root=run_root, api_config_path=API_CONFIG,
                tokenizer_path=TOKENIZER,
                root_id="cc-train-episode-61555")
            self.assertTrue(report["valid"], report["failed_checks"])
            self.assertEqual(0, report["api_calls"])
            self.assertFalse(run_root.exists())
            self.assertIn("FACTORIAL COUNTERFACTUAL COMPONENT CONTRACT",
                          report["prompt"])
            self.assertIn("false-parent/false-child", report["prompt"])
            self.assertIn("factorial_components",
                          single_root_mcq_materializer.
                          DISTRACTOR_PROMPT_CONTRACT_VERSION)


if __name__ == "__main__":
    unittest.main()
