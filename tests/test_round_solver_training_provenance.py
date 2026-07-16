from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts import run_auto_graph_expansion_round as auto_runner
from scripts import run_round_solver_training_pipeline as training


class RoundSolverTrainingProvenanceTest(unittest.TestCase):
    def test_prepared_stage3_train_file_bypasses_legacy_builder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = root / "stage3_train_mix_sft.jsonl"
            prepared.touch()
            args = SimpleNamespace(
                prepared_train_file=str(prepared),
                frontier=None,
                round_sft_train=None,
                frontier_report_dir=None,
            )
            with mock.patch.object(training, "run_command") as run_command:
                resolved = training.build_train_data(args, [], [], root / "reports")
            self.assertEqual(resolved, prepared)
            run_command.assert_not_called()

            args.frontier = ["unexpected.jsonl"]
            with self.assertRaisesRegex(ValueError, "cannot be combined"):
                training.build_train_data(args, [], [], root / "reports")

    def test_prepared_stage3_train_file_is_bound_to_passing_audit_and_sha(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = root / "sft_train.jsonl"
            prepared.write_text('{"messages": []}\n', encoding="utf-8")
            digest = training.file_sha256(prepared)
            summary = root / "train_mix_summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "schema_version": "stage3_train_mix_v1",
                        "status": "audit_passed",
                        "pass_criteria": {"heldout_overlap_zero": True},
                        "artifact_paths": {"sft_train": str(prepared)},
                        "output_artifacts": {"sft_train": {"sha256": digest}},
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                prepared_train_file=str(prepared),
                prepared_train_summary=str(summary),
            )
            audit = training.validate_prepared_train_audit(args, prepared)
            self.assertEqual(audit["sft_train_sha256"], digest)

            prepared.write_text('{"messages": [{"role": "user"}]}\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                training.validate_prepared_train_audit(args, prepared)

    def test_no_adapter_means_zero_initialized_lora_on_selected_base(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "accepted-stage2-merged"
            base.mkdir()
            provenance = training.validate_training_provenance(
                SimpleNamespace(base_model=str(base), init_adapter_dir=None)
            )

        self.assertEqual(provenance["initialization"], "zero_initialized_new_lora")
        self.assertIsNone(provenance["init_adapter_dir"])
        self.assertEqual(Path(provenance["resolved_base_model"]), base.resolve())

    def test_compatible_adapter_base_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            adapter = root / "adapter"
            base.mkdir()
            adapter.mkdir()
            (adapter / "adapter_model.pt").touch()
            (adapter / "adapter_config.json").write_text(
                json.dumps({"model": str(base)}), encoding="utf-8"
            )

            provenance = training.validate_training_provenance(
                SimpleNamespace(base_model=str(base), init_adapter_dir=str(adapter))
            )

        self.assertEqual(provenance["initialization"], "continued_from_compatible_adapter")
        self.assertEqual(Path(provenance["resolved_adapter_declared_base_model"]), base.resolve())

    def test_adapter_base_mismatch_fails_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            requested_base = root / "accepted-stage2-merged"
            adapter_base = root / "accepted-stage1-merged"
            adapter = root / "stage2-adapter"
            requested_base.mkdir()
            adapter_base.mkdir()
            adapter.mkdir()
            (adapter / "adapter_model.pt").touch()
            (adapter / "adapter_config.json").write_text(
                json.dumps({"model": str(adapter_base)}), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "Initial adapter base mismatch"):
                training.validate_training_provenance(
                    SimpleNamespace(base_model=str(requested_base), init_adapter_dir=str(adapter))
                )

    def test_validation_must_be_independent_or_explicitly_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            train_file = Path(tmp) / "train.jsonl"
            valid_file = Path(tmp) / "valid.jsonl"
            train_file.touch()
            valid_file.touch()

            with self.assertRaisesRegex(ValueError, "Choose an independent"):
                training.resolve_validation_file(
                    SimpleNamespace(valid_file=None, no_valid=False, skip_train=False, dry_run=False),
                    train_file,
                )
            with self.assertRaisesRegex(ValueError, "must be independent"):
                training.resolve_validation_file(
                    SimpleNamespace(
                        valid_file=str(train_file), no_valid=False, skip_train=False, dry_run=False
                    ),
                    train_file,
                )
            self.assertIsNone(
                training.resolve_validation_file(
                    SimpleNamespace(valid_file=None, no_valid=True, skip_train=False, dry_run=False),
                    train_file,
                )
            )
            self.assertEqual(
                training.resolve_validation_file(
                    SimpleNamespace(
                        valid_file=str(valid_file), no_valid=False, skip_train=False, dry_run=False
                    ),
                    train_file,
                ),
                valid_file,
            )

    def test_training_command_only_passes_nonempty_optional_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = Namespace(
                skip_train=False,
                adapter_output_dir=str(root / "adapter"),
                overwrite=False,
                dry_run=True,
                base_model=str(root / "accepted-stage2-merged"),
                init_adapter_dir=None,
                max_length=32768,
                epochs=1.0,
                batch_size=1,
                grad_accum=8,
                lr=1e-4,
                min_lr_ratio=0.1,
                warmup_ratio=0.05,
                weight_decay=0.0,
                max_grad_norm=1.0,
                lora_rank=16,
                lora_alpha=32,
                lora_dropout=0.05,
                eval_steps=5,
                save_steps=9999,
                max_valid_samples=64,
                seed=7,
            )
            with mock.patch.object(training, "run_command") as run_command:
                training.train_solver(args, root / "train.jsonl", None, root / "reports")

        command = run_command.call_args.args[0]
        self.assertNotIn("--init-adapter-dir", command)
        self.assertNotIn("--valid-file", command)
        self.assertEqual(command[command.index("--model") + 1], args.base_model)

    def test_stage3_auto_training_uses_stage2_merged_as_base_without_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_dir = root / "reports"
            auto_runner.write_json(report_dir / "quality_audit.json", {"pass": True})
            args = Namespace(
                run_training_pipeline=True,
                report_dir=str(report_dir),
                require_quality_audit_before_train=True,
                stage64_solver_provenance={"profile": "accepted_base_restart_stage2"},
                models_dir=str(root / "models"),
                output_dir=str(root / "generation"),
                train_mix_dir=str(root / "train_mix"),
                fixed_solver_name="solver_base_restart_stage2_accepted",
                solver_model=str(root / "accepted-stage2-merged"),
                solver_adapter=str(root / "stage2-adapter"),
                max_length=32768,
            )
            report = {
                "dev_or_heldout_frontier_count": 0,
                "frontier_count": 300,
                "policy_validation_pass": True,
                "policy_fallback_used": False,
                "seed_source_audit_pass": True,
            }
            with mock.patch.object(auto_runner.subprocess, "run") as run:
                auto_runner.maybe_run_training_pipeline(args, report)

        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--base-model") + 1], args.solver_model)
        self.assertIn("--no-valid", command)
        self.assertNotIn("--init-adapter-dir", command)
        self.assertNotIn(args.solver_adapter, command)


if __name__ == "__main__":
    unittest.main()
