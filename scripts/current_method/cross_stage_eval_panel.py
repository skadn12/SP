#!/usr/bin/env python3
"""Cross-stage evaluation panel for current-method self-play.

The panel is meant to answer one question consistently:

    Does solver stage N improve on the heldout frontier built for stage N?

By default this script only collects existing eval JSON files and writes a
Markdown report. Pass --run-original and/or --run-chat when you really want to
rerun vLLM evaluations.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PANEL_DIR = ROOT / "base_restart_current_method_validation/70_cross_stage_eval_panel"
DEFAULT_BASE_MODEL = Path("/root/autodl-tmp/Qwen3-4B")
DEFAULT_SEED = 20260615
DEFAULT_CHAT_SAMPLING_K = 1
FIRST_DRAW_OUTCOMES_KEY = "first_draw_correct_by_sample_id"
GREEDY_OUTCOMES_KEY = "greedy_k1_correct_by_sample_id"

MODELS: dict[str, Path] = {
    "base": DEFAULT_BASE_MODEL,
    "stage1": ROOT / "base_restart_current_method_validation/accepted_models/solver_base_restart_stage1_merged",
    "stage2": ROOT / "base_restart_current_method_validation/accepted_models/solver_base_restart_stage2_merged",
    "stage64": ROOT / "outputs/qwen3_4b_lora_stage64_r3_325f_shuffled_from_r2_ctx32000_merged",
}

ORIGINAL_DATASETS: dict[str, Path] = {
    "stage1_frontier_heldout": ROOT / "base_restart_current_method_validation/01_mining500_train/train_mix/frontier_heldout.jsonl",
    "stage2_frontier_heldout": ROOT / "base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/train_mix/frontier_heldout.jsonl",
    "stage64_frontier_heldout": ROOT
    / "base_restart_current_method_validation/69_stage64_api_selfplay500_ctx32768/"
    / "option_shuffled_seed20260710/splits/frontier_group_holdout_seed20260710/frontier_heldout.jsonl",
}

CHAT_INPUTS: tuple[Path, ...] = (
    ROOT
    / "base_restart_current_method_validation/69_stage64_api_selfplay500_ctx32768/"
    / "option_shuffled_seed20260710/splits/frontier_group_holdout_seed20260710/frontier_heldout_sft.jsonl",
    ROOT / "data/locomo_processed_datasets/_global_clean_v1_solver_r1_training_dataset/actual_sft/clean_v1_solver_r1_valid_chat.jsonl",
)

CLEAN_CHAT_INPUT = CHAT_INPUTS[1]


@dataclass(frozen=True)
class PanelSpec:
    """Resolved model/dataset registry for one historical or Stage3 panel."""

    models: dict[str, Path]
    original_datasets: dict[str, Path]
    chat_inputs: tuple[Path, ...]
    final_stage: str
    final_dataset: str
    final_title: str
    final_legacy: bool
    final_chat_eval_set: str | None
    diagnostics: tuple[tuple[str, str], ...] = ()

    @property
    def stage3_enabled(self) -> bool:
        return self.final_stage == "stage3"


def historical_panel_spec(
    *,
    stage64_heldout: Path | None = None,
    stage64_chat_heldout: Path | None = None,
    stage64_legacy: bool = True,
) -> PanelSpec:
    heldout = stage64_heldout or ORIGINAL_DATASETS["stage64_frontier_heldout"]
    chat_heldout = stage64_chat_heldout or CHAT_INPUTS[0]
    datasets = dict(ORIGINAL_DATASETS)
    datasets["stage64_frontier_heldout"] = heldout
    return PanelSpec(
        models=dict(MODELS),
        original_datasets=datasets,
        chat_inputs=(chat_heldout, CLEAN_CHAT_INPUT),
        final_stage="stage64",
        final_dataset="stage64_frontier_heldout",
        final_title="Stage64",
        final_legacy=stage64_legacy,
        final_chat_eval_set=chat_heldout.stem,
    )


def stage3_panel_spec(
    *,
    stage3_model: Path,
    stage3_heldout: Path,
    stage3_chat_heldout: Path | None = None,
    stage3_hard0: Path | None = None,
    stage3_easy5: Path | None = None,
) -> PanelSpec:
    models = {name: path for name, path in MODELS.items() if name != "stage64"}
    models["stage3"] = stage3_model
    datasets = {
        name: path
        for name, path in ORIGINAL_DATASETS.items()
        if name != "stage64_frontier_heldout"
    }
    datasets["stage3_frontier_heldout"] = stage3_heldout
    diagnostics: list[tuple[str, str]] = []
    if stage3_hard0 is not None:
        datasets["stage3_hard0"] = stage3_hard0
        diagnostics.append(("stage3_hard0", "Stage3 hard0"))
    if stage3_easy5 is not None:
        datasets["stage3_easy5"] = stage3_easy5
        diagnostics.append(("stage3_easy5", "Stage3 easy5"))
    chat_inputs = (stage3_chat_heldout, CLEAN_CHAT_INPUT) if stage3_chat_heldout else (CLEAN_CHAT_INPUT,)
    return PanelSpec(
        models=models,
        original_datasets=datasets,
        chat_inputs=chat_inputs,
        final_stage="stage3",
        final_dataset="stage3_frontier_heldout",
        final_title="Stage3",
        final_legacy=False,
        final_chat_eval_set=stage3_chat_heldout.stem if stage3_chat_heldout else None,
        diagnostics=tuple(diagnostics),
    )


def resolve_panel_spec(args: argparse.Namespace) -> PanelSpec:
    stage3_fields = (
        args.stage3_model,
        args.stage3_heldout,
        args.stage3_chat_heldout,
        args.stage3_hard0,
        args.stage3_easy5,
    )
    if any(value is not None for value in stage3_fields):
        missing = [
            flag
            for flag, value in (
                ("--stage3-model", args.stage3_model),
                ("--stage3-heldout", args.stage3_heldout),
            )
            if value is None
        ]
        if missing:
            raise ValueError("Stage3 mode requires " + " and ".join(missing))
        if args.run_chat and args.stage3_chat_heldout is None:
            raise ValueError("--run-chat in Stage3 mode requires --stage3-chat-heldout")
        return stage3_panel_spec(
            stage3_model=args.stage3_model,
            stage3_heldout=args.stage3_heldout,
            stage3_chat_heldout=args.stage3_chat_heldout,
            stage3_hard0=args.stage3_hard0,
            stage3_easy5=args.stage3_easy5,
        )
    return historical_panel_spec(
        stage64_heldout=args.stage64_heldout,
        stage64_chat_heldout=args.stage64_chat_heldout,
        stage64_legacy=not args.stage64_heldout_calibrated,
    )


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pct(value: Any) -> str:
    try:
        return f"{100.0 * float(value):.2f}%"
    except Exception:
        return "-"


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _supports_cli_option(script: Path, option: str) -> bool:
    """Check a Python argparse script without importing its heavyweight deps."""
    try:
        source = script.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(script))
    except (OSError, SyntaxError):
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if any(isinstance(arg, ast.Constant) and arg.value == option for arg in node.args):
            return True
    return False


def _original_command(
    script: Path,
    args: argparse.Namespace,
    model_path: Path,
    samples_path: Path,
    out_dir: Path,
    *,
    k_eval: int | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(script),
        "--task",
        "verify",
        "--samples",
        str(samples_path),
        "--out-dir",
        str(out_dir),
        "--base-model",
        str(DEFAULT_BASE_MODEL),
        "--solver-model",
        str(model_path),
        "--solver-tokenizer",
        str(DEFAULT_BASE_MODEL),
        "--k-eval",
        str(args.k_eval if k_eval is None else k_eval),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    if _supports_cli_option(script, "--seed"):
        cmd.extend(["--seed", str(args.seed)])
    return cmd


def run_original(
    args: argparse.Namespace,
    *,
    models: dict[str, Path] | None = None,
    datasets: dict[str, Path] | None = None,
) -> None:
    script = ROOT / "scripts/current_method/diagnostics_context_ablation.py"
    models = MODELS if models is None else models
    datasets = ORIGINAL_DATASETS if datasets is None else datasets
    if not _supports_cli_option(script, "--seed"):
        print(
            f"Note: {script.name} does not expose --seed; using its internal seed configuration.",
            file=sys.stderr,
        )
    for model_name, model_path in models.items():
        for dataset_name, samples_path in datasets.items():
            out_dir = args.panel_dir / "original_format" / model_name / dataset_name
            run_cmd(
                _original_command(script, args, model_path, samples_path, out_dir),
                cwd=ROOT,
            )
            if not args.skip_greedy_original:
                run_cmd(
                    _original_command(
                        script,
                        args,
                        model_path,
                        samples_path,
                        out_dir / "greedy_k1",
                        k_eval=1,
                    ),
                    cwd=ROOT,
                )


def run_chat(
    args: argparse.Namespace,
    *,
    models: dict[str, Path] | None = None,
    chat_inputs: tuple[Path, ...] | None = None,
) -> None:
    script = ROOT / "scripts/eval_v13_pilot_vllm.py"
    models = MODELS if models is None else models
    chat_inputs = CHAT_INPUTS if chat_inputs is None else chat_inputs
    for model_name, model_path in models.items():
        out_dir = args.panel_dir / "chat_format" / model_name
        cmd = [
            sys.executable,
            str(script),
            "--output-dir",
            str(out_dir),
            "--model",
            str(model_path),
            "--adapter-dir",
            str(model_path),
            "--max-length",
            str(args.max_length),
            "--max-model-len",
            str(args.max_model_len),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--sampling-k",
            str(args.chat_sampling_k),
            "--seed",
            str(args.seed),
            "--chunk-size",
            str(args.chunk_size),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--input",
        ]
        cmd.extend(str(path) for path in chat_inputs)
        run_cmd(cmd, cwd=ROOT)


def _read_first_draw_outcomes(path: Path) -> dict[str, bool]:
    outcomes: dict[str, bool] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            sample_id = row.get("sample_id")
            predictions = row.get("solver_eval", {}).get("predictions", [])
            first_correct = predictions[0].get("exact_correct") if predictions else None
            if not isinstance(sample_id, str) or not isinstance(first_correct, bool):
                continue
            if sample_id in outcomes:
                raise ValueError(f"Duplicate sample_id {sample_id!r} in {path}:{line_number}")
            outcomes[sample_id] = first_correct
    return outcomes


def collect_original(
    panel_dir: Path,
    *,
    models: dict[str, Path] | None = None,
    datasets: dict[str, Path] | None = None,
) -> dict[str, dict[str, Any]]:
    models = MODELS if models is None else models
    datasets = ORIGINAL_DATASETS if datasets is None else datasets
    out: dict[str, dict[str, Any]] = {}
    for model_name in models:
        out[model_name] = {}
        for dataset_name in datasets:
            p = panel_dir / "original_format" / model_name / dataset_name / "reports/verify_summary.json"
            if p.exists():
                summary = read_json(p)
                if not isinstance(summary, dict):
                    print(f"Skipping non-object summary: {p}", file=sys.stderr)
                    continue
                generation_path = p.parent.parent / "generation/verify_k5_eval.jsonl"
                if generation_path.exists():
                    summary[FIRST_DRAW_OUTCOMES_KEY] = _read_first_draw_outcomes(generation_path)
                greedy_summary_path = p.parent.parent / "greedy_k1/reports/verify_summary.json"
                greedy_generation_path = p.parent.parent / "greedy_k1/generation/verify_k5_eval.jsonl"
                if greedy_summary_path.exists() and greedy_generation_path.exists():
                    greedy_summary = read_json(greedy_summary_path)
                    summary["solver_greedy_accuracy_k1"] = greedy_summary.get("solver_exact_accuracy_k1")
                    summary[GREEDY_OUTCOMES_KEY] = _read_first_draw_outcomes(greedy_generation_path)
                out[model_name][dataset_name] = summary
    write_json(panel_dir / "original_format/original_panel_summary.json", out)
    return out


def collect_chat(
    panel_dir: Path,
    expected_sampling_k: int = DEFAULT_CHAT_SAMPLING_K,
    *,
    models: dict[str, Path] | None = None,
) -> dict[str, Any]:
    models = MODELS if models is None else models
    out: dict[str, Any] = {}
    for model_name in models:
        p = panel_dir / "chat_format" / model_name / "overall_summary.json"
        if p.exists():
            summary = read_json(p)
            actual_sampling_k = summary.get("sampling_k") if isinstance(summary, dict) else None
            if (
                not isinstance(actual_sampling_k, int)
                or isinstance(actual_sampling_k, bool)
                or actual_sampling_k != expected_sampling_k
            ):
                print(
                    f"Skipping stale/incompatible chat summary {p}: "
                    f"sampling_k={actual_sampling_k!r}, expected {expected_sampling_k}.",
                    file=sys.stderr,
                )
                continue
            out[model_name] = summary
    write_json(panel_dir / "chat_format/chat_panel_summary.json", out)
    return out


def _metric_with_n(original: dict[str, dict[str, Any]], model: str, dataset: str, key: str) -> str:
    summary = original.get(model, {}).get(dataset, {})
    value = summary.get(key)
    if value is None:
        return "-"
    n = summary.get("n")
    return f"{pct(value)} (n={n})" if isinstance(n, int) and not isinstance(n, bool) else pct(value)


def _chat_metric_with_n(chat: dict[str, Any], model: str, eval_set: str) -> str:
    by_set = chat.get(model, {}).get("summary", {}).get("by_eval_set", {})
    summary = by_set.get(eval_set, {})
    value = summary.get("accuracy")
    if value is None:
        return "-"
    n = summary.get("n")
    return f"{pct(value)} (n={n})" if isinstance(n, int) and not isinstance(n, bool) else pct(value)


def mcnemar_exact_p(improved: int, regressed: int) -> float:
    """Two-sided exact McNemar p-value for the two discordant cells."""
    if improved < 0 or regressed < 0:
        raise ValueError("discordant counts must be non-negative")
    discordant = improved + regressed
    if discordant == 0:
        return 1.0
    lower = min(improved, regressed)
    tail_numerator = sum(math.comb(discordant, i) for i in range(lower + 1))
    return min(1.0, 2.0 * tail_numerator / (2**discordant))


def paired_first_draw_comparison(
    original: dict[str, dict[str, Any]],
    candidate: str,
    baseline: str,
    dataset: str,
) -> dict[str, Any] | None:
    candidate_summary = original.get(candidate, {}).get(dataset, {})
    baseline_summary = original.get(baseline, {}).get(dataset, {})
    outcome_key = (
        GREEDY_OUTCOMES_KEY
        if isinstance(candidate_summary.get(GREEDY_OUTCOMES_KEY), dict)
        and isinstance(baseline_summary.get(GREEDY_OUTCOMES_KEY), dict)
        else FIRST_DRAW_OUTCOMES_KEY
    )
    candidate_outcomes = candidate_summary.get(outcome_key)
    baseline_outcomes = baseline_summary.get(outcome_key)
    if not isinstance(candidate_outcomes, dict) or not isinstance(baseline_outcomes, dict):
        return None
    if not candidate_outcomes or set(candidate_outcomes) != set(baseline_outcomes):
        return None
    if any(not isinstance(value, bool) for value in candidate_outcomes.values()) or any(
        not isinstance(value, bool) for value in baseline_outcomes.values()
    ):
        return None

    sample_ids = sorted(candidate_outcomes)
    improved = sum(candidate_outcomes[sid] and not baseline_outcomes[sid] for sid in sample_ids)
    regressed = sum(baseline_outcomes[sid] and not candidate_outcomes[sid] for sid in sample_ids)
    return {
        "n": len(sample_ids),
        "candidate_correct": sum(candidate_outcomes.values()),
        "baseline_correct": sum(baseline_outcomes.values()),
        "improved": improved,
        "regressed": regressed,
        "net": improved - regressed,
        "discordant": improved + regressed,
        "mcnemar_exact_p": mcnemar_exact_p(improved, regressed),
        "metric": "greedy_k1" if outcome_key == GREEDY_OUTCOMES_KEY else "sampled_first_draw",
    }


def stage_comparisons(
    *,
    stage64_legacy: bool = True,
    stage3_enabled: bool = False,
) -> tuple[tuple[str, str, str, str, bool], ...]:
    final = (
        ("Stage3", "stage3", "stage2", "stage3_frontier_heldout", False)
        if stage3_enabled
        else ("Stage64", "stage64", "stage2", "stage64_frontier_heldout", stage64_legacy)
    )
    return (
        ("Stage1", "stage1", "base", "stage1_frontier_heldout", False),
        ("Stage2", "stage2", "stage1", "stage2_frontier_heldout", False),
        final,
    )


def _net_phrase(stats: dict[str, Any]) -> str:
    net = int(stats["net"])
    if net > 0:
        return f"净增 {net}/{stats['n']}"
    if net < 0:
        return f"净减 {-net}/{stats['n']}"
    return f"净变化 0/{stats['n']}"


def _stage_judgment(stats: dict[str, Any], legacy: bool) -> str:
    if legacy:
        return "legacy heldout，仅作诊断，不能作为阶段增长证据"
    if stats["net"] > 0 and stats["mcnemar_exact_p"] < 0.05:
        return "配对证据支持阶段收益成立"
    return "未达到双侧 α=0.05，不能判定阶段收益成立"


def _stage_conclusion(
    original: dict[str, dict[str, Any]],
    title: str,
    candidate: str,
    baseline: str,
    dataset: str,
    legacy: bool,
) -> str:
    stats = paired_first_draw_comparison(original, candidate, baseline, dataset)
    if stats is None:
        return f"{title}：缺少同一批 sample_id 的逐样本 greedy/first-draw 结果，无法做配对判定。"
    n = stats["n"]
    candidate_accuracy = stats["candidate_correct"] / n
    baseline_accuracy = stats["baseline_correct"] / n
    conclusion = (
        f"{title}：基于 {stats['metric']}，{candidate} 为 {pct(candidate_accuracy)}，"
        f"{baseline} 为 {pct(baseline_accuracy)}（配对 n={n}），{_net_phrase(stats)}；"
        f"discordant 为改善 {stats['improved']}、退化 {stats['regressed']}，"
        f"McNemar exact p={stats['mcnemar_exact_p']:.4f}；{_stage_judgment(stats, legacy)}。"
    )
    if legacy:
        conclusion += (
            " 该 heldout 沿用了 shuffle 前的 frontier 标记，未在 shuffle 后用固定 Stage2 solver "
            "重新评估并重选 frontier。"
        )
    return conclusion


def _retention_conclusion(
    original: dict[str, dict[str, Any]],
    *,
    candidate: str = "stage64",
    title: str = "Stage64 旧阶段保留诊断",
) -> str:
    specs = (
        ("Stage1 heldout", "stage1", "stage1_frontier_heldout"),
        ("Stage2 heldout", "stage2", "stage2_frontier_heldout"),
    )
    parts: list[str] = []
    for label, baseline, dataset in specs:
        stats = paired_first_draw_comparison(original, candidate, baseline, dataset)
        if stats is None:
            parts.append(f"{label} 缺少可配对逐样本结果")
            continue
        parts.append(
            f"相对 {baseline} 在 {label} 上{_net_phrase(stats)}"
            f"（改善 {stats['improved']}、退化 {stats['regressed']}，"
            f"McNemar exact p={stats['mcnemar_exact_p']:.4f}）"
        )
    return title + "：" + "；".join(parts) + "。"


def markdown_report(
    original: dict[str, dict[str, Any]],
    chat: dict[str, Any],
    *,
    stage64_legacy: bool = True,
    spec: PanelSpec | None = None,
) -> str:
    spec = spec or historical_panel_spec(stage64_legacy=stage64_legacy)
    lines: list[str] = []
    if spec.final_legacy:
        final_suffix = " (legacy)"
    elif spec.stage3_enabled:
        final_suffix = ""
    else:
        final_suffix = " (post-shuffle calibrated)"
    if spec.stage3_enabled:
        final_goal = "Stage3 solver 是否在可靠的 Stage3 heldout frontier 上超过 Stage2。"
        retention_title = "Stage3 旧阶段保留诊断"
    else:
        final_goal = (
            "Stage64 的旧 heldout 标为 legacy，只用于诊断，不用于证明阶段增长。"
            if spec.final_legacy
            else "Stage64 使用显式声明为 post-shuffle calibrated 的 heldout。"
        )
        retention_title = "Stage64 旧阶段保留诊断"
    lines.extend(
        [
            "# 跨阶段能力增长评估面板",
            "",
            "这个面板用于验证 self-play 的阶段性收益：",
            "",
            "- Stage1 solver 是否在 Stage1 heldout frontier 上超过 base；",
            "- Stage2 solver 是否在 Stage2 heldout frontier 上超过 Stage1；",
            f"- {final_goal}",
            "",
            "## 结论",
            "",
        ]
    )
    comparisons = stage_comparisons(
        stage64_legacy=spec.final_legacy,
        stage3_enabled=spec.stage3_enabled,
    )
    for index, comparison in enumerate(comparisons, start=1):
        lines.append(f"{index}. {_stage_conclusion(original, *comparison)}")
    final_stage = comparisons[-1][1]
    lines.extend(
        [
            f"4. {_retention_conclusion(original, candidate=final_stage, title=retention_title)}",
            "",
            "## Original-format Verifier: First-draw Accuracy",
            "",
            "这里的 first-draw 是同一次 K-sample（默认 K=5）运行中的第一个样本，不是另跑的 greedy K=1。",
            "",
            f"| solver | Stage1 heldout | Stage2 heldout | {spec.final_title} heldout{final_suffix} |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for model in spec.models:
        lines.append(
            "| {m} | {s1} | {s2} | {final} |".format(
                m=model,
                s1=_metric_with_n(original, model, "stage1_frontier_heldout", "solver_exact_accuracy_k1"),
                s2=_metric_with_n(original, model, "stage2_frontier_heldout", "solver_exact_accuracy_k1"),
                final=_metric_with_n(original, model, spec.final_dataset, "solver_exact_accuracy_k1"),
            )
        )
    if any(
        original.get(model, {}).get(dataset, {}).get("solver_greedy_accuracy_k1") is not None
        for model in spec.models
        for dataset in spec.original_datasets
    ):
        lines.extend(
            [
                "",
                "## Original-format Verifier: Independent Greedy K1",
                "",
                f"| solver | Stage1 heldout | Stage2 heldout | {spec.final_title} heldout{final_suffix} |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for model in spec.models:
            lines.append(
                "| {m} | {s1} | {s2} | {final} |".format(
                    m=model,
                    s1=_metric_with_n(original, model, "stage1_frontier_heldout", "solver_greedy_accuracy_k1"),
                    s2=_metric_with_n(original, model, "stage2_frontier_heldout", "solver_greedy_accuracy_k1"),
                    final=_metric_with_n(original, model, spec.final_dataset, "solver_greedy_accuracy_k1"),
                )
            )
    lines.extend(
        [
            "",
            "## Original-format Verifier: K5 Soft Accuracy",
            "",
            f"| solver | Stage1 heldout | Stage2 heldout | {spec.final_title} heldout{final_suffix} |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for model in spec.models:
        lines.append(
            "| {m} | {s1} | {s2} | {final} |".format(
                m=model,
                s1=_metric_with_n(original, model, "stage1_frontier_heldout", "solver_soft_accuracy_k5"),
                s2=_metric_with_n(original, model, "stage2_frontier_heldout", "solver_soft_accuracy_k5"),
                final=_metric_with_n(original, model, spec.final_dataset, "solver_soft_accuracy_k5"),
            )
        )
    if spec.diagnostics:
        diagnostic_headers = " | ".join(label for _, label in spec.diagnostics)
        lines.extend(
            [
                "",
                f"## {spec.final_title} Difficulty Diagnostics",
                "",
                "hard0/easy5 仅用于观察难度迁移和异常退化，不参与 frontier 阶段收益判定。",
                "",
                f"| solver | {diagnostic_headers} |",
                "| --- | " + " | ".join("---:" for _ in spec.diagnostics) + " |",
            ]
        )
        for model in spec.models:
            metrics = " | ".join(
                _metric_with_n(original, model, dataset, "solver_exact_accuracy_k1")
                for dataset, _ in spec.diagnostics
            )
            lines.append(f"| {model} | {metrics} |")
    lines.extend(
        [
            "",
            "## Paired Greedy/First-draw Comparisons",
            "",
            "优先配对独立 greedy K1；若旧产物尚无 greedy 结果才回退到 sampled first-draw。p 值为 Python 标准库计算的双侧 exact McNemar。",
            "",
            "| comparison | metric | heldout | paired n | improved | regressed | net | McNemar exact p | 判定 |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for title, candidate, baseline, dataset, legacy in comparisons:
        stats = paired_first_draw_comparison(original, candidate, baseline, dataset)
        dataset_label = title + " heldout" + (" (legacy)" if legacy else "")
        if stats is None:
            lines.append(
                f"| {candidate} vs {baseline} | - | {dataset_label} | - | - | - | - | - | 缺少可配对逐样本结果 |"
            )
            continue
        lines.append(
            "| {candidate} vs {baseline} | {metric} | {dataset} | {n} | {up} | {down} | {net:+d} | {p:.4f} | {judgment} |".format(
                candidate=candidate,
                baseline=baseline,
                metric=stats["metric"],
                dataset=dataset_label,
                n=stats["n"],
                up=stats["improved"],
                down=stats["regressed"],
                net=stats["net"],
                p=stats["mcnemar_exact_p"],
                judgment=_stage_judgment(stats, legacy),
            )
        )
    if chat:
        chat_heldout = spec.final_chat_eval_set
        lines.extend(
            [
                "",
                "## Chat/SFT-format Eval",
                "",
                "这个评估使用 SFT chat 文件。它能检查训练格式和 clean retention，但不能替代 original-format frontier 校准。",
                "",
                f"| solver | {spec.final_title} SFT heldout{final_suffix} | clean retention |",
                "| --- | ---: | ---: |",
            ]
        )
        for model in spec.models:
            lines.append(
                "| {m} | {held} | {clean} |".format(
                    m=model,
                    held=(
                        _chat_metric_with_n(chat, model, chat_heldout)
                        if chat_heldout is not None
                        else "-"
                    ),
                    clean=_chat_metric_with_n(chat, model, "clean_v1_solver_r1_valid_chat"),
                )
            )
    if spec.stage3_enabled:
        lines.extend(
            [
                "",
                "## Stage3 验收闭环",
                "",
                "1. Stage3 heldout 必须来自 option balance 后由固定 Stage2 solver 重新评估的 reliable frontier；",
                "2. Stage3 与 Stage2 在相同 sample_id 上做 independent greedy K1 和 sampled K5 配对比较；",
                "3. Stage1/Stage2 heldout 用于 retention，hard0/easy5 仅用于难度诊断；",
                "4. 同时检查 clean retention 和外部泛化，避免用单一 heldout 掩盖遗忘。",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## 下一步修复",
                "",
                "Stage64 的正确闭环应该是：",
                "",
                "1. 在生成门禁中拒绝 exact duplicate / 多标签同答案；",
                "2. 清空旧 `solver_eval`，并对每题构造 cyclic option permutations；",
                "3. 固定使用 accepted base-restart Stage2 solver，显式记录 resolved checkpoint 与 seed；",
                "4. 通过严格解码健康门禁后，只保留每个排列都处于 1/5-4/5 的 robust frontier；",
                "5. 按 `parent_sample_id` 整组切 train/heldout，并冻结未参与模型选择的新 test；",
                "6. 训练 Stage64 solver，同时保留 clean replay/regression guard；",
                "7. 用 independent greedy K1、sampled K5、旧阶段 heldout 和外部集做配对评估。",
                "",
            ]
        )
    supported: list[str] = []
    not_established: list[str] = []
    for title, candidate, baseline, dataset, legacy in comparisons:
        if legacy:
            continue
        stats = paired_first_draw_comparison(original, candidate, baseline, dataset)
        if stats is not None and stats["net"] > 0 and stats["mcnemar_exact_p"] < 0.05:
            supported.append(title)
        else:
            not_established.append(title)
    supported_text = "、".join(supported) if supported else "无"
    not_established_text = "、".join(not_established) if not_established else "无"
    lines.extend(
        [
            f"按当前配对 greedy K1（缺失时回退 first-draw）exact 检验（双侧 α=0.05），有证据支持的阶段收益：{supported_text}；"
            f"尚不能判定成立：{not_established_text}。"
            + (
                "Stage3 heldout 不作 legacy 标记，其证据资格由上游 reliable-frontier 门禁保证。"
                if spec.stage3_enabled
                else "Stage64 需要先重新校准 heldout。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or summarize a cross-stage self-play eval panel.")
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL_DIR)
    parser.add_argument("--stage64-heldout", type=Path, default=ORIGINAL_DATASETS["stage64_frontier_heldout"])
    parser.add_argument("--stage64-chat-heldout", type=Path, default=CHAT_INPUTS[0])
    parser.add_argument(
        "--stage64-heldout-calibrated",
        action="store_true",
        help="Declare the supplied Stage64 heldout post-shuffle calibrated; legacy is the safe default.",
    )
    parser.add_argument(
        "--stage3-model",
        type=Path,
        help="Enable true Stage3 mode with this merged Stage3 solver checkpoint.",
    )
    parser.add_argument(
        "--stage3-heldout",
        type=Path,
        help="Reliable Stage3 frontier heldout recalibrated by the fixed Stage2 solver.",
    )
    parser.add_argument(
        "--stage3-chat-heldout",
        type=Path,
        help="Optional Stage3 heldout SFT/chat file; required with --run-chat in Stage3 mode.",
    )
    parser.add_argument("--stage3-hard0", type=Path, help="Optional Stage3 hard0 diagnostic dataset.")
    parser.add_argument("--stage3-easy5", type=Path, help="Optional Stage3 easy5 diagnostic dataset.")
    parser.add_argument("--run-original", action="store_true", help="Rerun original-format verifier evals.")
    parser.add_argument("--skip-greedy-original", action="store_true", help="With --run-original, skip the additional independent greedy K=1 pass.")
    parser.add_argument("--run-chat", action="store_true", help="Rerun chat/SFT vLLM evals.")
    parser.add_argument("--k-eval", type=int, default=5)
    parser.add_argument("--chat-sampling-k", type=int, default=DEFAULT_CHAT_SAMPLING_K)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-length", type=int, default=32000)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    try:
        spec = resolve_panel_spec(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.panel_dir.mkdir(parents=True, exist_ok=True)
    if args.run_original:
        run_original(args, models=spec.models, datasets=spec.original_datasets)
    if args.run_chat:
        run_chat(args, models=spec.models, chat_inputs=spec.chat_inputs)
    original = collect_original(
        args.panel_dir,
        models=spec.models,
        datasets=spec.original_datasets,
    )
    chat = collect_chat(
        args.panel_dir,
        expected_sampling_k=args.chat_sampling_k,
        models=spec.models,
    )
    report = markdown_report(original, chat, spec=spec)
    report_path = args.panel_dir / "cross_stage_eval_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
