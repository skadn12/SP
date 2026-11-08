from __future__ import annotations

import copy
import tempfile
import json
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts import run_auto_graph_expansion_round as runner
from scripts.current_method import stage3_operation_controller as controller


def args_for(model: Path, *, allow: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        qa_operation_answer_type_diversification=True,
        solver_model=str(model),
        solver_adapter=str(runner.DEFAULT_ACTIVE_SOLVER_ADAPTER),
        accepted_stage2_solver_model=str(runner.DEFAULT_ACTIVE_SOLVER_MERGED),
        accepted_stage2_solver_adapter=str(runner.DEFAULT_ACTIVE_SOLVER_ADAPTER),
        fixed_solver_name=runner.DEFAULT_ACTIVE_SOLVER_NAME,
        policy_source_eval_dir="/tmp/no-stage64-policy-source",
        policy="explicit-policy.json",
        allow_nonaccepted_stage64_solver=allow,
    )


class Stage64SolverProvenanceTest(unittest.TestCase):
    def _closure_args(self, root: Path, **overrides: object) -> SimpleNamespace:
        config = root / "config.json"
        config.write_text("{}\n", encoding="utf-8")
        values: dict[str, object] = {
            "stage3_batch_closure": "dry-run",
            "stage3_max_hardening_retries": 2,
            "stage3_final_eval_seed": 20260712,
            "solver_eval_seed": 20260711,
            "seed": 20260710,
            "qa_operation_answer_type_diversification": True,
            "skip_eval": False,
            "allow_nonaccepted_stage64_solver": False,
            "max_model_len": 32768,
            "max_new_tokens": 80,
            "temperature": 0.7,
            "k5_eval_temperature": 1.0,
            "top_p": 0.9,
            "config": str(config),
            "accepted_stage2_solver_model": str(root / "accepted-stage2"),
            "base_model": str(root / "base-model"),
            "gpu_memory_utilization": 0.9,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_stage3_batch_closure_cli_defaults_are_off_and_independent(self) -> None:
        with mock.patch("sys.argv", ["run_auto"]):
            args = runner.parse_args()

        self.assertEqual(args.stage3_batch_closure, "off")
        self.assertIsNone(args.stage3_closure_output_dir)
        self.assertEqual(args.stage3_final_eval_seed, 20260712)
        self.assertEqual(args.stage3_max_hardening_retries, 2)

    def test_stage3_batch_closure_preconditions_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = (
                ({"skip_eval": True}, "incompatible with --skip-eval"),
                ({"stage3_final_eval_seed": 20260711}, "must differ"),
                (
                    {
                        "stage3_batch_closure": "execute",
                        "allow_nonaccepted_stage64_solver": True,
                    },
                    "forbids --allow-nonaccepted-stage64-solver",
                ),
                ({"max_model_len": 16384}, "max_model_len=32768"),
                ({"max_new_tokens": 81}, "max_new_tokens=80"),
                ({"k5_eval_temperature": 0.8}, "temperature=1.0"),
                ({"top_p": 0.8}, "top_p=0.9"),
                ({"stage3_max_hardening_retries": 3}, "between 0 and 2"),
                ({"config": str(root / "missing.json")}, "existing challenger --config"),
            )
            for overrides, message in cases:
                with self.subTest(overrides=overrides):
                    with self.assertRaisesRegex(RuntimeError, message):
                        runner.validate_stage3_batch_closure_args(
                            self._closure_args(root, **overrides)
                        )

    def test_stage3_batch_closure_command_is_thin_and_exact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._closure_args(root, stage3_max_hardening_retries=1)
            command = runner.build_stage3_batch_closure_command(
                args,
                mining_root=root / "mining",
                search_recalibration_dir=root / "mining" / "recalibrated",
                output_dir=root / "closure",
            )

        self.assertEqual(Path(command[1]).name, "stage3_batch_closure.py")
        self.assertIn("--dry-run", command)
        self.assertNotIn("--execute", command)
        self.assertEqual(command[command.index("--search-eval-seed") + 1], "20260711")
        self.assertEqual(command[command.index("--final-eval-seed") + 1], "20260712")
        self.assertEqual(command[command.index("--max-hardening-retries") + 1], "1")

    def test_stage64_recalibration_command_uses_dedicated_k5_temperature(self) -> None:
        args = Namespace(
            base_model="/model/base",
            solver_model="/model/stage2",
            solver_tokenizer="/model/base",
            accepted_stage2_solver_model="/model/stage2",
            seed=101,
            max_model_len=32768,
            max_new_tokens=80,
            temperature=0.7,
            k5_eval_temperature=1.15,
            top_p=0.9,
            gpu_memory_utilization=0.9,
            allow_nonaccepted_stage64_solver=False,
        )

        command = runner.build_stage64_recalibration_command(
            args,
            input_path=Path("/input.jsonl"),
            output_dir=Path("/output"),
            report_dir=Path("/reports"),
            solver_eval_seed=202,
        )

        self.assertEqual(command[command.index("--temperature") + 1], "1.15")
        self.assertEqual(command[command.index("--seed") + 1], "202")
        self.assertEqual(command[command.index("--generation-seed") + 1], "101")
        self.assertEqual(runner.effective_stage64_k5_eval_temperature(args), 1.15)

        args.k5_eval_temperature = 0.0
        fallback = runner.build_stage64_recalibration_command(
            args,
            input_path=Path("/input.jsonl"),
            output_dir=Path("/output"),
            report_dir=Path("/reports"),
            solver_eval_seed=202,
        )
        self.assertEqual(fallback[fallback.index("--temperature") + 1], "0.7")
        self.assertEqual(runner.effective_stage64_k5_eval_temperature(args), 0.7)

    def test_formal_stage3_generation_requires_four_thousand_token_delta(self) -> None:
        args = SimpleNamespace(
            qa_operation_answer_type_diversification=True,
            skip_generation=False,
            min_length_delta_tokens=3999,
            challenger_max_tokens=2048,
        )

        with self.assertRaisesRegex(RuntimeError, "min-length-delta-tokens >= 4000"):
            runner.enforce_formal_stage3_length_contract(args)

    def test_formal_stage3_length_contract_fails_before_creating_run_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "must-not-be-created"
            argv = [
                "run_auto",
                "--qa-operation-answer-type-diversification",
                "--run-root",
                str(run_root),
            ]
            with mock.patch("sys.argv", argv):
                parsed = runner.parse_args()
            with self.assertRaisesRegex(RuntimeError, "min-length-delta-tokens >= 4000"):
                runner.finalize_current_method_args(parsed)
            self.assertFalse(run_root.exists())

    def test_formal_stage3_generation_raises_challenger_completion_budget(self) -> None:
        args = SimpleNamespace(
            qa_operation_answer_type_diversification=True,
            skip_generation=False,
            min_length_delta_tokens=4000,
            challenger_max_tokens=2048,
        )

        contract = runner.enforce_formal_stage3_length_contract(args)

        self.assertTrue(contract["enforced_for_this_invocation"])
        self.assertEqual(contract["requested_challenger_max_tokens"], 2048)
        self.assertEqual(args.challenger_max_tokens, 8192)
        self.assertEqual(contract["effective_challenger_max_tokens"], 8192)

    def test_formal_stage3_larger_delta_gets_explicit_completion_headroom(self) -> None:
        args = SimpleNamespace(
            qa_operation_answer_type_diversification=True,
            skip_generation=False,
            min_length_delta_tokens=9000,
            challenger_max_tokens=4096,
        )

        contract = runner.enforce_formal_stage3_length_contract(args)

        self.assertEqual(args.challenger_max_tokens, 10536)
        self.assertEqual(contract["minimum_challenger_max_tokens"], 10536)

    def test_skip_generation_does_not_enforce_new_generation_length_contract(self) -> None:
        args = SimpleNamespace(
            qa_operation_answer_type_diversification=True,
            skip_generation=True,
            min_length_delta_tokens=0,
            challenger_max_tokens=2048,
        )

        contract = runner.enforce_formal_stage3_length_contract(args)

        self.assertFalse(contract["enforced_for_this_invocation"])
        self.assertEqual(contract["mode"], "existing_generation_recalibration")
        self.assertEqual(args.challenger_max_tokens, 2048)

    def test_skip_generation_finalization_allows_existing_zero_delta_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = [
                "run_auto",
                "--qa-operation-answer-type-diversification",
                "--skip-generation",
                "--policy",
                str(root / "existing-policy.json"),
                "--run-root",
                str(root / "run"),
                "--output-dir",
                str(root / "existing-generation"),
                "--report-dir",
                str(root / "reports"),
            ]
            with mock.patch("sys.argv", argv):
                finalized = runner.finalize_current_method_args(runner.parse_args())

        self.assertEqual(finalized.min_length_delta_tokens, 0)
        self.assertFalse(finalized.stage3_length_contract["enforced_for_this_invocation"])

    def test_formal_stage3_generation_summary_is_verified_and_audited(self) -> None:
        args = SimpleNamespace(
            qa_operation_answer_type_diversification=True,
            min_length_delta_tokens=4000,
            stage3_dialogue_protocol="segmented",
            stage3_length_contract={
                "mode": "formal_stage3_generation",
                "enforced_for_this_invocation": True,
            },
        )
        summary = {
            "generated_count": 2,
            "length_guard_enabled": True,
            "min_length_delta_tokens": 4000,
            "length_guard_target_delta": 4000,
            "rows_below_min_length_delta": 0,
            "rows_below_4000_delta": 0,
            "formal_mining_api_native_dialogue_accept_count": 2,
            "formal_mining_local_padding_accept_count": 0,
            "formal_mining_segmented_dialogue_accept_count": 2,
            "formal_mining_prefix_hash_chain_verified_count": 2,
            "formal_mining_final_qa_after_frozen_alignment_count": 2,
            "formal_mining_marker_scheduler_verified_count": 2,
            "formal_mining_exact_claim_preflight_verified_count": 2,
            "formal_mining_turn_duplicate_audit_version": (
                "formal_segment_turn_duplicate_audit_v1"
            ),
            "formal_mining_turn_duplicate_audit_verified_count": 2,
            "formal_mining_turn_duplicate_exact_pair_count_sum": 0,
            "formal_mining_turn_duplicate_near_pair_count_sum": 0,
            "formal_mining_turn_duplicate_phrase_pair_count_sum": 0,
            "formal_mining_opening_diversity_audit_version": (
                "formal_segment_opening_diversity_audit_v1"
            ),
            "formal_mining_opening_diversity_audit_verified_count": 2,
            "formal_mining_opening_diversity_concentrated_prefix_count_sum": 0,
        }

        audited = runner.verify_formal_stage3_generation_summary(
            args,
            summary,
            effective_challenger_max_tokens=8192,
        )

        self.assertTrue(audited["formal_stage3_length_contract"]["summary_verified"])
        self.assertEqual(audited["challenger_max_tokens"], 8192)

        bad_summary = dict(summary, rows_below_4000_delta=1)
        with self.assertRaisesRegex(RuntimeError, "rows_below_4000_delta=1"):
            runner.verify_formal_stage3_generation_summary(
                args,
                bad_summary,
                effective_challenger_max_tokens=8192,
            )

        padded_summary = dict(summary, formal_mining_local_padding_accept_count=1)
        with self.assertRaisesRegex(RuntimeError, "formal_mining_local_padding_accept_count=1"):
            runner.verify_formal_stage3_generation_summary(
                args,
                padded_summary,
                effective_challenger_max_tokens=8192,
            )

        nonnative_summary = dict(summary, formal_mining_api_native_dialogue_accept_count=1)
        with self.assertRaisesRegex(RuntimeError, "!= generated_count=2"):
            runner.verify_formal_stage3_generation_summary(
                args,
                nonnative_summary,
                effective_challenger_max_tokens=8192,
            )

        early_qa_summary = dict(
            summary,
            formal_mining_final_qa_after_frozen_alignment_count=1,
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "formal_mining_final_qa_after_frozen_alignment_count=1 != generated_count=2",
        ):
            runner.verify_formal_stage3_generation_summary(
                args,
                early_qa_summary,
                effective_challenger_max_tokens=8192,
            )

        duplicate_summary = dict(
            summary,
            formal_mining_turn_duplicate_phrase_pair_count_sum=1,
        )
        with self.assertRaisesRegex(RuntimeError, "duplicate pair count=1"):
            runner.verify_formal_stage3_generation_summary(
                args,
                duplicate_summary,
                effective_challenger_max_tokens=8192,
            )

        concentrated_opening_summary = dict(
            summary,
            formal_mining_opening_diversity_concentrated_prefix_count_sum=1,
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "concentrated opening prefix count=1",
        ):
            runner.verify_formal_stage3_generation_summary(
                args,
                concentrated_opening_summary,
                effective_challenger_max_tokens=8192,
            )

        empty_summary = dict(summary, generated_count=0, length_guard_target_delta=0)
        empty_audited = runner.verify_formal_stage3_generation_summary(
            args,
            empty_summary,
            effective_challenger_max_tokens=8192,
        )
        self.assertEqual(empty_audited["length_guard_target_delta"], 4000)

        long_form_args = SimpleNamespace(
            qa_operation_answer_type_diversification=True,
            min_length_delta_tokens=4000,
            stage3_dialogue_protocol="long_form",
            stage3_length_contract={
                "mode": "formal_stage3_generation",
                "enforced_for_this_invocation": True,
            },
        )
        long_form_summary = dict(
            summary,
            formal_mining_segmented_dialogue_accept_count=0,
            formal_mining_prefix_hash_chain_verified_count=0,
            formal_mining_marker_scheduler_verified_count=0,
            formal_mining_exact_claim_preflight_verified_count=0,
            formal_mining_turn_duplicate_audit_version="",
            formal_mining_turn_duplicate_audit_verified_count=0,
            formal_mining_turn_duplicate_phrase_pair_count_sum=0,
            formal_mining_opening_diversity_audit_version="",
            formal_mining_opening_diversity_audit_verified_count=0,
            formal_mining_opening_diversity_concentrated_prefix_count_sum=0,
        )
        long_form_audited = runner.verify_formal_stage3_generation_summary(
            long_form_args,
            long_form_summary,
            effective_challenger_max_tokens=8192,
        )
        self.assertTrue(
            long_form_audited["formal_stage3_length_contract"]["summary_verified"]
        )

    def test_formal_stage3_parent_admission_rejects_historical_padding(self) -> None:
        args = SimpleNamespace(
            qa_operation_answer_type_diversification=True,
            skip_generation=False,
        )
        dirty = {
            "sample_id": "dirty-parent",
            "dialogue": {
                "turns": [
                    {
                        "turn_id": "D1:1",
                        "speaker": "Analyst",
                        "text": runner.stage3_parent_builder.KNOWN_DETERMINISTIC_PADDING_TEXTS[0],
                    }
                ]
            },
        }

        with self.assertRaisesRegex(RuntimeError, "affected_rows=1/1"):
            runner.enforce_formal_stage3_parent_cleanliness(args, [dirty])

        clean = copy.deepcopy(dirty)
        clean["dialogue"]["turns"][0]["text"] = "A substantive parent turn."
        audit = runner.enforce_formal_stage3_parent_cleanliness(args, [clean])
        self.assertTrue(audit["enforced_for_this_invocation"])
        self.assertEqual(audit["affected_row_count"], 0)
        self.assertEqual(
            audit["known_padding_signature_sha256"],
            runner.stage3_parent_builder.deterministic_padding_signature_sha256(),
        )

    def test_generation_protocol_binds_clean_parent_dialogue_contract(self) -> None:
        protocol = controller.generation_protocol()
        parent_contract = protocol["parent_dialogue_contract"]
        self.assertFalse(parent_contract["historical_deterministic_padding_allowed"])
        self.assertEqual(
            parent_contract["cleanup_version"],
            runner.stage3_parent_builder.DETERMINISTIC_PADDING_CLEANUP_VERSION,
        )

    def test_accepted_base_restart_stage2_profile_passes(self) -> None:
        provenance = runner.enforce_stage64_solver_provenance(
            args_for(runner.DEFAULT_ACTIVE_SOLVER_MERGED)
        )
        self.assertEqual(provenance["profile"], "accepted_base_restart_stage2")
        self.assertFalse(provenance["diagnostic_override"])

    def test_legacy_solver_is_blocked(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "accepted base-restart Stage2"):
            runner.enforce_stage64_solver_provenance(args_for(runner.DEFAULT_R2_MERGED))

    def test_nonaccepted_override_is_explicitly_labeled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provenance = runner.enforce_stage64_solver_provenance(
                args_for(Path(tmp) / "other-model", allow=True)
            )
        self.assertEqual(provenance["profile"], "diagnostic_override")
        self.assertTrue(provenance["issues"])

    def test_explicit_accepted_reference_is_portable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            external = Path(tmp) / "accepted-stage2"
            args = args_for(external)
            args.accepted_stage2_solver_model = str(external)
            provenance = runner.enforce_stage64_solver_provenance(args)
        self.assertEqual(provenance["profile"], "accepted_base_restart_stage2")

    def test_batch_mining_stops_before_eval_when_semantic_gate_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = Namespace(
                output_dir=str(root / "generation"),
                run_root=str(root / "run"),
                report_dir=str(root / "reports"),
                stage64_mining_output_dir=str(root / "mining"),
                seed=7,
                skip_eval=False,
                base_model="/model/base",
                solver_model="/model/stage2",
                solver_tokenizer="/model/base",
                max_model_len=32768,
                gpu_memory_utilization=0.9,
                temperature=0.7,
                k5_eval_temperature=1.15,
                allow_nonaccepted_stage64_solver=False,
                stage64_min_reliable_frontier=300,
                run_training_pipeline=False,
                stage64_solver_provenance={"resolved_model": "/model/stage2"},
            )
            raw_audit = Path(args.output_dir) / "graph_conditioned_api_dialogue_raw.jsonl"
            raw_audit.parent.mkdir(parents=True, exist_ok=True)
            raw_audit.write_text(
                json.dumps(
                    {
                        "accepted": True,
                        "sample_id": "s1",
                        "api_graph_meta": {
                            "semantic_graph_v2_quality": {
                                "pass": True,
                                "gate_version": controller.REQUIRED_GRAPH_GATE_VERSION,
                            }
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            def fake_run(command: list[str], **_: object) -> None:
                output = Path(command[command.index("--output-dir") + 1])
                runner.write_json(
                    output / "option_shuffle_summary.json",
                    {
                        "quality": {"input_count": 1, "accepted_count": 0, "rejected_count": 1},
                        "shuffle": {"count": 0},
                    },
                )

            with mock.patch.object(runner.subprocess, "run", side_effect=fake_run) as run:
                summary = runner.run_stage64_robust_mining(
                    args,
                    [
                        {
                            "sample_id": "s1",
                            "native_generator_version": controller.REQUIRED_NATIVE_GENERATOR_VERSION,
                            "qa": {"question_type": "single_choice"},
                        }
                    ],
                )
            self.assertEqual(run.call_count, 1)
            self.assertEqual(summary["status"], "no_semantic_quality_passed")
            self.assertEqual(summary["robust_frontier_count"], 0)
            self.assertEqual(summary["requested_k5_eval_temperature"], 1.15)
            self.assertEqual(summary["effective_k5_eval_temperature"], 1.15)
            self.assertEqual(summary["k5_eval_temperature_source"], "k5_eval_temperature")

    def test_batch_mining_invokes_closure_and_embeds_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "reports").mkdir()
            args = Namespace(
                output_dir=str(root / "generation"),
                run_root=str(root / "run"),
                report_dir=str(root / "reports"),
                stage64_mining_output_dir=str(root / "mining"),
                stage3_closure_output_dir=str(root / "closure"),
                stage3_batch_closure="execute",
                stage3_final_eval_seed=20260712,
                stage3_max_hardening_retries=1,
                qa_operation_answer_type_diversification=True,
                seed=20260710,
                solver_eval_seed=20260711,
                skip_eval=False,
                base_model=str(root / "base-model"),
                solver_model=str(root / "accepted-stage2"),
                solver_tokenizer=str(root / "base-model"),
                accepted_stage2_solver_model=str(root / "accepted-stage2"),
                max_model_len=32768,
                max_new_tokens=80,
                gpu_memory_utilization=0.9,
                temperature=0.7,
                k5_eval_temperature=1.0,
                top_p=0.9,
                allow_nonaccepted_stage64_solver=False,
                stage64_min_reliable_frontier=300,
                run_training_pipeline=False,
                stage64_solver_provenance={"resolved_model": str(root / "accepted-stage2")},
                skip_stage3_operation_controller_update=True,
                stage3_operation_state_path=str(root / "operation-state.json"),
                config=str(root / "config.json"),
            )
            Path(args.config).write_text("{}\n", encoding="utf-8")

            def fake_run(command: list[str], **_: object) -> None:
                script = Path(command[1]).name
                output = Path(command[command.index("--output-dir") + 1])
                if script == "shuffle_stage64_options.py":
                    runner.write_json(
                        output / "option_shuffle_summary.json",
                        {
                            "quality": {"input_count": 1, "accepted_count": 1, "rejected_count": 0},
                            "shuffle": {"count": 1},
                        },
                    )
                elif script == "recalibrate_stage64_shuffled_frontier.py":
                    runner.write_json(
                        output / "recalibration_summary.json",
                        {"clean_count": 1, "frontier_count": 1},
                    )
                elif script == "stage3_batch_closure.py":
                    runner.write_json(
                        output / "closure_manifest.json",
                        {
                            "status": "complete",
                            "mode": "execute",
                            "counts": {
                                "search_clean": 1,
                                "search_original_robust_frontier": 1,
                                "search_hardening_eligible": 0,
                                "final_reliable_frontier": 2,
                            },
                        },
                    )
                else:
                    self.fail(f"Unexpected subprocess: {command}")

            with (
                mock.patch.object(
                    runner,
                    "attach_stage3_generation_protocol_from_raw_audit",
                    return_value={"sample_count": 1},
                ),
                mock.patch.object(runner.subprocess, "run", side_effect=fake_run) as run,
            ):
                summary = runner.run_stage64_robust_mining(
                    args,
                    [{"sample_id": "s1", "qa": {"question_type": "single_choice"}}],
                )

            self.assertEqual(run.call_count, 3)
            closure_call = run.call_args_list[2].args[0]
            self.assertEqual(Path(closure_call[1]).name, "stage3_batch_closure.py")
            self.assertIn("--execute", closure_call)
            closure = summary["stage3_batch_closure"]
            self.assertTrue(closure["invoked"])
            self.assertEqual(closure["status"], "complete")
            self.assertEqual(closure["counts"]["search_original_robust_frontier"], 1)
            self.assertEqual(closure["manifest"]["mode"], "execute")
            self.assertEqual(summary["search_robust_frontier_count"], 1)
            self.assertEqual(summary["final_reliable_frontier_count"], 2)
            self.assertEqual(summary["robust_frontier_count"], 2)
            self.assertEqual(
                summary["stage3_operation_controller"]["evaluation_phase"],
                "pre_hardening_search",
            )

    def test_stage64_defaults_promote_legacy_profile_before_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted_model = root / "accepted-model"
            accepted_adapter = root / "accepted-adapter"
            source = root / "policy-source"
            runner.write_json(source / "solver_eval_summary.json", {"model": str(accepted_model)})
            argv = [
                "run_auto",
                "--qa-operation-answer-type-diversification",
                "--min-length-delta-tokens",
                "4000",
                "--run-name",
                "stage64-test",
                "--run-root",
                str(root / "run"),
                "--output-dir",
                str(root / "generation"),
                "--report-dir",
                str(root / "reports"),
                "--parent-dir",
                str(root / "parents.jsonl"),
                "--policy-source-eval-dir",
                str(source),
                "--accepted-stage2-solver-model",
                str(accepted_model),
                "--accepted-stage2-solver-adapter",
                str(accepted_adapter),
            ]
            with mock.patch("sys.argv", argv):
                finalized = runner.finalize_current_method_args(runner.parse_args())
        self.assertEqual(Path(finalized.solver_model).resolve(), accepted_model.resolve())
        self.assertEqual(Path(finalized.solver_adapter).resolve(), accepted_adapter.resolve())
        self.assertEqual(finalized.fixed_solver_name, runner.DEFAULT_ACTIVE_SOLVER_NAME)
        self.assertEqual(finalized.max_model_len, 32768)
        self.assertEqual(finalized.min_length_delta_tokens, 4000)
        self.assertEqual(finalized.challenger_max_tokens, 8192)
        self.assertTrue(finalized.stage3_length_contract["enforced_for_this_invocation"])
        self.assertEqual(finalized.stage64_solver_provenance["profile"], "accepted_base_restart_stage2")


if __name__ == "__main__":
    unittest.main()
