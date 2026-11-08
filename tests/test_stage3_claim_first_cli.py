from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import run_auto_graph_expansion_round as runner


class Stage3ClaimFirstCliTest(unittest.TestCase):
    @staticmethod
    def _parse(*arguments: str):
        with mock.patch(
            "sys.argv",
            ["run_auto_graph_expansion_round.py", *arguments],
        ):
            return runner.parse_args()

    def test_defaults_are_disabled_and_preserve_three_to_six_contract(self) -> None:
        args = self._parse()

        self.assertFalse(args.claim_first_multiqa)
        self.assertEqual(args.claim_first_min_qas, 3)
        self.assertEqual(args.claim_first_max_qas, 6)
        self.assertEqual(args.claim_first_eval_per_mille, 100)

        native_args = runner.generation_args(args, {}, {})
        self.assertFalse(native_args.claim_first_multiqa)
        self.assertEqual(native_args.claim_first_min_qas, 3)
        self.assertEqual(native_args.claim_first_max_qas, 6)
        self.assertEqual(native_args.claim_first_eval_per_mille, 100)

    def test_opt_in_and_custom_bounds_are_forwarded_to_native_args(self) -> None:
        args = self._parse(
            "--claim-first-multiqa",
            "--claim-first-min-qas",
            "4",
            "--claim-first-max-qas",
            "5",
            "--claim-first-eval-per-mille",
            "250",
        )

        native_args = runner.generation_args(args, {}, {})
        self.assertTrue(native_args.claim_first_multiqa)
        self.assertEqual(native_args.claim_first_min_qas, 4)
        self.assertEqual(native_args.claim_first_max_qas, 5)
        self.assertEqual(native_args.claim_first_eval_per_mille, 250)

    def test_truth_first_requires_the_full_generate_by_construction_cli_contract(self) -> None:
        invalid_cases = (
            (
                ("--truth-first-generate-by-construction",),
                "requires --claim-first-multiqa",
            ),
            (
                (
                    "--truth-first-generate-by-construction",
                    "--claim-first-multiqa",
                ),
                "requires --qa-operation-answer-type-diversification",
            ),
            (
                (
                    "--truth-first-generate-by-construction",
                    "--claim-first-multiqa",
                    "--qa-operation-answer-type-diversification",
                ),
                "requires --stage3-dialogue-protocol discourse_plan",
            ),
        )
        for arguments, expected_message in invalid_cases:
            with self.subTest(arguments=arguments):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as caught:
                        self._parse(*arguments)
                self.assertEqual(caught.exception.code, 2)
                self.assertIn(expected_message, stderr.getvalue())

    def test_truth_first_valid_contract_is_forwarded_to_native_args(self) -> None:
        args = self._parse(
            "--truth-first-generate-by-construction",
            "--claim-first-multiqa",
            "--qa-operation-answer-type-diversification",
            "--stage3-dialogue-protocol",
            "discourse_plan",
            "--no-resume",
            "--require-fresh-stage3-run",
            "--claim-first-min-qas",
            "3",
            "--claim-first-max-qas",
            "6",
        )

        native_args = runner.generation_args(args, {}, {})

        self.assertTrue(args.truth_first_generate_by_construction)
        self.assertTrue(native_args.truth_first_generate_by_construction)
        self.assertTrue(native_args.claim_first_multiqa)
        self.assertTrue(native_args.qa_operation_answer_type_diversification)
        self.assertEqual(args.stage3_dialogue_protocol, "discourse_plan")
        self.assertEqual(native_args.claim_first_min_qas, 3)
        self.assertEqual(native_args.claim_first_max_qas, 6)

    def test_truth_first_requires_fresh_state_and_forbids_legacy_graph_reuse(self) -> None:
        base = (
            "--truth-first-generate-by-construction",
            "--claim-first-multiqa",
            "--qa-operation-answer-type-diversification",
            "--stage3-dialogue-protocol",
            "discourse_plan",
        )
        invalid_cases = (
            (base, "requires --no-resume"),
            ((*base, "--no-resume"), "requires --require-fresh-stage3-run"),
            (
                (*base, "--no-resume", "--require-fresh-stage3-run", "--fixed-stage3-graph-artifact", "old.json"),
                "forbids --fixed-stage3-graph-artifact",
            ),
            (
                (*base, "--no-resume", "--require-fresh-stage3-run", "--graph-proposal-cache"),
                "forbids --graph-proposal-cache",
            ),
            (
                (*base, "--no-resume", "--require-fresh-stage3-run", "--skip-generation"),
                "forbids --skip-generation",
            ),
        )
        for arguments, expected_message in invalid_cases:
            with self.subTest(arguments=arguments):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as caught:
                        self._parse(*arguments)
                self.assertEqual(caught.exception.code, 2)
                self.assertIn(expected_message, stderr.getvalue())

    def test_invalid_bounds_fail_during_argument_parsing(self) -> None:
        invalid_cases = (
            ("--claim-first-min-qas", "2"),
            ("--claim-first-max-qas", "7"),
            (
                "--claim-first-min-qas",
                "6",
                "--claim-first-max-qas",
                "5",
            ),
            ("--claim-first-eval-per-mille", "-1"),
            ("--claim-first-eval-per-mille", "1001"),
        )
        for arguments in invalid_cases:
            with self.subTest(arguments=arguments):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as caught:
                        self._parse("--claim-first-multiqa", *arguments)
                self.assertEqual(caught.exception.code, 2)

    def test_claim_first_is_compatible_with_fresh_no_resume_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "claim_first_n1"
            output_dir = run_root / "generation"
            args = self._parse(
                "--claim-first-multiqa",
                "--no-resume",
                "--require-fresh-stage3-run",
                "--run-root",
                str(run_root),
                "--output-dir",
                str(output_dir),
                "--stage3-operation-state-path",
                str(run_root / "stage3_operation_state.json"),
            )

            finalized = runner.finalize_current_method_args(args)

        self.assertTrue(finalized.claim_first_multiqa)
        self.assertTrue(finalized.no_resume)
        self.assertTrue(finalized.require_fresh_stage3_run)
        self.assertTrue(finalized.stage3_fresh_run_isolation["pass"])
        self.assertTrue(finalized.stage3_fresh_run_isolation["no_resume"])
    def test_smoke_operation_requires_isolated_formal_n1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "bridge_n1"
            args = self._parse(
                "--qa-operation-answer-type-diversification",
                "--claim-first-multiqa",
                "--train-only",
                "--n-train",
                "1",
                "--min-length-delta-tokens",
                "4000",
                "--stage3-smoke-operation",
                "bridge",
                "--no-resume",
                "--require-fresh-stage3-run",
                "--run-root",
                str(run_root),
                "--stage3-operation-state-path",
                str(run_root / "state" / "stage3_operation.json"),
            )
            finalized = runner.finalize_current_method_args(args)
        self.assertEqual(finalized.stage3_smoke_operation, "bridge")
        self.assertTrue(finalized.stage3_fresh_run_isolation["pass"])

        invalid = self._parse(
            "--qa-operation-answer-type-diversification",
            "--claim-first-multiqa",
            "--train-only",
            "--n-train",
            "2",
            "--min-length-delta-tokens",
            "4000",
            "--stage3-smoke-operation",
            "bridge",
            "--no-resume",
            "--require-fresh-stage3-run",
        )
        with self.assertRaisesRegex(RuntimeError, "n_train_is_one"):
            runner.finalize_current_method_args(invalid)


if __name__ == "__main__":
    unittest.main()
