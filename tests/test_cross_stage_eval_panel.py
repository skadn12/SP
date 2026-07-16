from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/current_method/cross_stage_eval_panel.py"
SPEC = importlib.util.spec_from_file_location("cross_stage_eval_panel", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
panel = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = panel
SPEC.loader.exec_module(panel)


def summary(outcomes: dict[str, bool]) -> dict[str, object]:
    n = len(outcomes)
    accuracy = sum(outcomes.values()) / n
    return {
        "n": n,
        "k_eval": 5,
        "solver_exact_accuracy_k1": accuracy,
        "solver_soft_accuracy_k5": accuracy,
        panel.FIRST_DRAW_OUTCOMES_KEY: outcomes,
    }


class CrossStageEvalPanelTest(unittest.TestCase):
    def test_stage2_net_one_of_thirty_is_not_claimed(self) -> None:
        sample_ids = [f"s{i}" for i in range(30)]
        baseline_correct = set(sample_ids[:16])
        candidate_correct = (baseline_correct - set(sample_ids[:5])) | set(sample_ids[16:22])
        original = {
            "stage1": {
                "stage2_frontier_heldout": summary(
                    {sample_id: sample_id in baseline_correct for sample_id in sample_ids}
                )
            },
            "stage2": {
                "stage2_frontier_heldout": summary(
                    {sample_id: sample_id in candidate_correct for sample_id in sample_ids}
                )
            },
        }

        report = panel.markdown_report(original, {})

        self.assertIn("净增 1/30", report)
        self.assertIn("discordant 为改善 6、退化 5", report)
        self.assertIn("McNemar exact p=1.0000", report)
        self.assertIn("不能判定阶段收益成立", report)
        self.assertIn("First-draw Accuracy", report)
        self.assertIn("不是另跑的 greedy K=1", report)
        self.assertIn("56.67% (n=30)", report)
        self.assertIn("Stage64 heldout (legacy)", report)

    def test_exact_mcnemar_uses_discordant_counts(self) -> None:
        self.assertEqual(panel.mcnemar_exact_p(6, 5), 1.0)
        self.assertAlmostEqual(panel.mcnemar_exact_p(7, 0), 0.015625)
        self.assertEqual(panel.mcnemar_exact_p(0, 0), 1.0)

    def test_collectors_load_first_draw_and_reject_stale_chat_k(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            panel_dir = Path(tmp)
            original_dir = panel_dir / "original_format/base/stage1_frontier_heldout"
            (original_dir / "reports").mkdir(parents=True)
            (original_dir / "generation").mkdir(parents=True)
            (original_dir / "reports/verify_summary.json").write_text(
                json.dumps({"n": 1, "solver_exact_accuracy_k1": 1.0}), encoding="utf-8"
            )
            (original_dir / "generation/verify_k5_eval.jsonl").write_text(
                json.dumps(
                    {
                        "sample_id": "sample-1",
                        "solver_eval": {"predictions": [{"exact_correct": True}]},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            chat_path = panel_dir / "chat_format/base/overall_summary.json"
            chat_path.parent.mkdir(parents=True)
            chat_path.write_text(json.dumps({"sampling_k": 5, "summary": {}}), encoding="utf-8")

            with (
                mock.patch.object(panel, "MODELS", {"base": Path("model")}),
                mock.patch.object(
                    panel, "ORIGINAL_DATASETS", {"stage1_frontier_heldout": Path("samples")}
                ),
            ):
                original = panel.collect_original(panel_dir)
                stale_chat = panel.collect_chat(panel_dir, expected_sampling_k=1)
                chat_path.write_text(json.dumps({"sampling_k": 1, "summary": {}}), encoding="utf-8")
                current_chat = panel.collect_chat(panel_dir, expected_sampling_k=1)

            self.assertEqual(
                original["base"]["stage1_frontier_heldout"][panel.FIRST_DRAW_OUTCOMES_KEY],
                {"sample-1": True},
            )
            self.assertEqual(stale_chat, {})
            self.assertIn("base", current_chat)

    def test_original_command_only_passes_seed_when_supported(self) -> None:
        args = argparse.Namespace(
            k_eval=5,
            max_model_len=32768,
            gpu_memory_utilization=0.9,
            seed=1234,
        )
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "downstream.py"
            script.write_text('parser.add_argument("-s", "--seed", type=int)\n', encoding="utf-8")
            supported = panel._original_command(
                script, args, Path("model"), Path("samples.jsonl"), Path("out")
            )
            script.write_text('parser.add_argument("--task")\n', encoding="utf-8")
            unsupported = panel._original_command(
                script, args, Path("model"), Path("samples.jsonl"), Path("out")
            )

        self.assertEqual(supported[-2:], ["--seed", "1234"])
        self.assertNotIn("--seed", unsupported)

    def test_stage3_cli_resolves_true_stage3_registry_and_diagnostics(self) -> None:
        args = panel.parse_args(
            [
                "--stage3-model",
                "/models/stage3",
                "--stage3-heldout",
                "/data/stage3_frontier_heldout.jsonl",
                "--stage3-chat-heldout",
                "/data/stage3_frontier_heldout_sft.jsonl",
                "--stage3-hard0",
                "/data/stage3_hard0.jsonl",
                "--stage3-easy5",
                "/data/stage3_easy5.jsonl",
            ]
        )

        spec = panel.resolve_panel_spec(args)

        self.assertTrue(spec.stage3_enabled)
        self.assertEqual(spec.models["stage3"], Path("/models/stage3"))
        self.assertNotIn("stage64", spec.models)
        self.assertEqual(
            spec.original_datasets["stage3_frontier_heldout"],
            Path("/data/stage3_frontier_heldout.jsonl"),
        )
        self.assertNotIn("stage64_frontier_heldout", spec.original_datasets)
        self.assertEqual(
            spec.final_chat_eval_set,
            "stage3_frontier_heldout_sft",
        )
        self.assertEqual(
            spec.diagnostics,
            (("stage3_hard0", "Stage3 hard0"), ("stage3_easy5", "Stage3 easy5")),
        )

    def test_stage3_cli_requires_model_and_heldout_and_chat_for_chat_run(self) -> None:
        with self.assertRaisesRegex(ValueError, "--stage3-model"):
            panel.resolve_panel_spec(panel.parse_args(["--stage3-heldout", "heldout.jsonl"]))
        with self.assertRaisesRegex(ValueError, "--stage3-chat-heldout"):
            panel.resolve_panel_spec(
                panel.parse_args(
                    [
                        "--stage3-model",
                        "model",
                        "--stage3-heldout",
                        "heldout.jsonl",
                        "--run-chat",
                    ]
                )
            )

    def test_stage3_report_is_nonlegacy_and_never_calls_stage3_stage64(self) -> None:
        sample_ids = [f"s{i}" for i in range(8)]
        stage2_outcomes = {sample_id: sample_id == "s0" for sample_id in sample_ids}
        stage3_outcomes = {sample_id: True for sample_id in sample_ids}
        original = {
            "stage2": {
                "stage3_frontier_heldout": summary(stage2_outcomes),
                "stage3_hard0": summary({"hard": False}),
                "stage3_easy5": summary({"easy": True}),
            },
            "stage3": {
                "stage3_frontier_heldout": summary(stage3_outcomes),
                "stage3_hard0": summary({"hard": True}),
                "stage3_easy5": summary({"easy": True}),
            },
        }
        spec = panel.stage3_panel_spec(
            stage3_model=Path("model"),
            stage3_heldout=Path("heldout.jsonl"),
            stage3_hard0=Path("hard0.jsonl"),
            stage3_easy5=Path("easy5.jsonl"),
        )

        report = panel.markdown_report(original, {}, spec=spec)

        self.assertIn("Stage3 solver 是否在可靠的 Stage3 heldout frontier 上超过 Stage2", report)
        self.assertIn("stage3 vs stage2", report)
        self.assertIn("Stage3 旧阶段保留诊断", report)
        self.assertIn("Stage3 Difficulty Diagnostics", report)
        self.assertIn("hard0/easy5 仅用于", report)
        self.assertNotIn("Stage3 heldout (legacy)", report)
        self.assertNotIn("Stage64", report)

    def test_default_cli_keeps_historical_stage64_mode(self) -> None:
        spec = panel.resolve_panel_spec(panel.parse_args([]))

        self.assertFalse(spec.stage3_enabled)
        self.assertEqual(spec.final_stage, "stage64")
        self.assertTrue(spec.final_legacy)
        self.assertIn("stage64", spec.models)
        self.assertIn("stage64_frontier_heldout", spec.original_datasets)

        calibrated = panel.markdown_report({}, {}, stage64_legacy=False)
        self.assertIn("Stage64 heldout (post-shuffle calibrated)", calibrated)

    def test_stage3_run_registry_builds_commands_without_running_vllm(self) -> None:
        args = panel.parse_args(
            [
                "--stage3-model",
                "/models/stage3",
                "--stage3-heldout",
                "/data/stage3.jsonl",
                "--run-original",
                "--skip-greedy-original",
            ]
        )
        spec = panel.resolve_panel_spec(args)
        args.panel_dir = Path("/tmp/stage3-panel")
        with mock.patch.object(panel, "run_cmd") as run_cmd:
            panel.run_original(
                args,
                models={"stage3": spec.models["stage3"]},
                datasets={"stage3_frontier_heldout": spec.original_datasets["stage3_frontier_heldout"]},
            )

        self.assertEqual(run_cmd.call_count, 1)
        command = run_cmd.call_args.args[0]
        self.assertIn("/models/stage3", command)
        self.assertIn("/data/stage3.jsonl", command)
        self.assertIn("/tmp/stage3-panel/original_format/stage3/stage3_frontier_heldout", command)


if __name__ == "__main__":
    unittest.main()
