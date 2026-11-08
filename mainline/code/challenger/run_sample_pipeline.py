from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one LoCoMo sample through the graph-aware QA data pipeline.")
    parser.add_argument("--raw-input", default="locomo10.json")
    parser.add_argument("--sample-index", type=int, required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--models", default="qwen3-4b,qwen3-8b")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--candidate-count", type=int, default=3)
    parser.add_argument("--augmentation-rounds", type=int, default=2)
    parser.add_argument("--request-timeout", type=int, default=600)
    parser.add_argument("--light-candidates-per-model", type=int, default=3)
    parser.add_argument("--full-candidates-per-model", type=int, default=5)
    parser.add_argument("--graph-ratio", type=float, default=0.5)
    parser.add_argument("--graph-min-dialogues", type=int, default=20)
    parser.add_argument("--graph-max-dialogues", type=int, default=60)
    parser.add_argument("--qa-per-dialogue", type=int, default=2)
    parser.add_argument("--candidate-pool-per-dialogue", type=int, default=6)
    parser.add_argument("--min-light-accuracy", type=float, default=0.3)
    parser.add_argument("--max-light-accuracy", type=float, default=0.7)
    parser.add_argument("--max-too-easy", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true", help="Run commands even when their primary output exists.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def first_json(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8").lstrip()
    obj, _ = json.JSONDecoder().raw_decode(text)
    return obj if isinstance(obj, dict) else {}


def collect_results(path: str | Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    results = payload.get("results") if isinstance(payload, dict) else None
    return [item for item in results if isinstance(item, dict)] if isinstance(results, list) else []


def collect_failures(path: str | Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    payload = load_json(path)
    failures = payload.get("failures") if isinstance(payload, dict) else None
    return [item for item in failures if isinstance(item, dict)] if isinstance(failures, list) else []


def processed_count(path: str | Path) -> int:
    if not Path(path).exists():
        return 0
    return len(collect_results(path)) + len(collect_failures(path))


def result_count(path: str | Path) -> int:
    if not Path(path).exists():
        return 0
    return len(collect_results(path))


def count_raw_qa(path: str | Path) -> int:
    payload = load_json(path)
    if not isinstance(payload, list) or not payload:
        return 0
    qa = payload[0].get("qa") if isinstance(payload[0], dict) else None
    return len(qa) if isinstance(qa, list) else 0


def label_counts(path: str | Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in collect_results(path):
        label = str(result.get("frontier_label") or "unknown")
        counts[label] = counts.get(label, 0) + 1
    return counts


def average_words(path: str | Path) -> float | None:
    results = collect_results(path)
    values = [
        result.get("final_dialogue_word_count")
        for result in results
        if isinstance(result.get("final_dialogue_word_count"), (int, float))
    ]
    return round(sum(values) / len(values), 4) if values else None


def graph_dialogue_count(success_count: int, args: argparse.Namespace) -> int:
    if success_count <= 0:
        return args.graph_min_dialogues
    value = int(success_count * args.graph_ratio)
    return min(args.graph_max_dialogues, max(args.graph_min_dialogues, value))


def command_to_string(command: list[str]) -> str:
    return " ".join(f'"{item}"' if " " in item else item for item in command)


def run_step(
    *,
    name: str,
    command: list[str],
    outputs: list[Path],
    args: argparse.Namespace,
    summary: dict[str, Any],
    is_complete=None,
) -> None:
    exists = all(path.exists() for path in outputs)
    complete = bool(exists and (is_complete() if is_complete is not None else True))
    if exists and not complete:
        print(f"[resume] {name}: outputs exist but are incomplete")
    if complete and not args.force:
        print(f"[skip] {name}: outputs complete")
        summary["steps"].append({"name": name, "status": "skipped", "outputs": [str(path) for path in outputs]})
        return
    print(f"[run] {name}")
    print(command_to_string(command))
    if args.dry_run:
        summary["steps"].append({"name": name, "status": "dry_run", "command": command})
        return
    start = time.time()
    completed = subprocess.run(command, cwd=Path.cwd(), check=False)
    elapsed = round(time.time() - start, 2)
    status = "completed" if completed.returncode == 0 else "failed"
    summary["steps"].append(
        {
            "name": name,
            "status": status,
            "returncode": completed.returncode,
            "elapsed_seconds": elapsed,
            "command": command,
            "outputs": [str(path) for path in outputs],
        }
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or f"locomo_sample{args.sample_index}"
    py = sys.executable

    raw_answered = workdir / f"{prefix}_raw_answered_list.json"
    round1 = workdir / f"{prefix}_round1.json"
    round1_eval = workdir / f"{prefix}_round1_model_eval_qwen3_4b_8b.json"
    round1_frontier = workdir / f"{prefix}_round1_frontier_qwen3_4b_8b.json"
    graph = workdir / f"{prefix}_round1_memory_graph.json"
    graph_summary = workdir / f"{prefix}_round1_memory_graph_summary.json"
    graph_hardened_prefix = workdir / f"{prefix}_graph_hardened_v3"
    graph_hardened = graph_hardened_prefix.with_suffix(".json")
    graph_hardened_report = graph_hardened_prefix.with_name(graph_hardened_prefix.name + "_report.json")
    graph_hardened_graph = workdir / f"{prefix}_graph_hardened_v3_memory_graph.json"
    graph_hardened_graph_summary = workdir / f"{prefix}_graph_hardened_v3_memory_graph_summary.json"
    multiqa_prefix = workdir / f"{prefix}_graph_multiqa_temporal_v3"
    multiqa = multiqa_prefix.with_suffix(".json")
    multiqa_report = multiqa_prefix.with_name(multiqa_prefix.name + "_report.json")
    multiqa_eval = workdir / f"{prefix}_graph_multiqa_temporal_v3_model_eval_qwen3_4b_8b.json"
    multiqa_frontier = workdir / f"{prefix}_graph_multiqa_temporal_v3_frontier_qwen3_4b_8b.json"
    merged = workdir / f"{prefix}_training_candidates.json"
    merged_report = workdir / f"{prefix}_training_candidates_report.json"
    pipeline_summary_path = workdir / f"{prefix}_pipeline_summary.json"

    summary: dict[str, Any] = {
        "sample_index": args.sample_index,
        "prefix": prefix,
        "workdir": str(workdir),
        "steps": [],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }

    run_step(
        name="extract_raw_answered",
        command=[
            py,
            "extract_raw_locomo_sample.py",
            "--input",
            args.raw_input,
            "--sample-index",
            str(args.sample_index),
            "--output",
            str(raw_answered),
            "--drop-empty-answer",
            "--as-list",
        ],
        outputs=[raw_answered],
        args=args,
        summary=summary,
    )

    qa_count = count_raw_qa(raw_answered) if raw_answered.exists() else 0
    run_step(
        name="surface_harden_round1",
        command=[
            py,
            "test.py",
            "--dataset",
            str(raw_answered),
            "--sample-index",
            "0",
            "--qa-start",
            "0",
            "--qa-count",
            str(qa_count),
            "--output",
            str(round1),
            "--continue-on-error",
            "--checkpoint-every",
            str(args.checkpoint_every),
            "--candidate-count",
            str(args.candidate_count),
            "--augmentation-rounds",
            str(args.augmentation_rounds),
            "--request-timeout",
            str(args.request_timeout),
            *(["--resume-output"] if args.resume and round1.exists() else []),
        ],
        outputs=[round1],
        args=args,
        summary=summary,
        is_complete=lambda: processed_count(round1) >= qa_count,
    )

    run_step(
        name="round1_model_eval",
        command=[
            py,
            "evaluate_model_difficulty.py",
            "--input",
            str(round1),
            "--output",
            str(round1_eval),
            "--models",
            args.models,
            "--candidates-per-model",
            str(args.full_candidates_per_model),
        ],
        outputs=[round1_eval],
        args=args,
        summary=summary,
        is_complete=lambda: result_count(round1_eval) >= result_count(round1),
    )

    run_step(
        name="round1_frontier",
        command=[
            py,
            "frontier_filter.py",
            "--input",
            str(round1_eval),
            "--output",
            str(round1_frontier),
        ],
        outputs=[round1_frontier],
        args=args,
        summary=summary,
        is_complete=lambda: result_count(round1_frontier) >= result_count(round1_eval),
    )

    success_count = len(collect_results(round1_frontier)) if round1_frontier.exists() else 0
    max_dialogues = graph_dialogue_count(success_count, args)
    max_too_easy = args.max_too_easy if args.max_too_easy is not None else max_dialogues + args.graph_min_dialogues
    summary["round1"] = {
        "raw_qa_count": qa_count,
        "successful_harden_count": success_count,
        "label_counts": label_counts(round1_frontier) if round1_frontier.exists() else {},
        "average_final_dialogue_word_count": average_words(round1_frontier) if round1_frontier.exists() else None,
        "graph_selected_dialogue_count": max_dialogues,
        "max_too_easy_for_merge": max_too_easy,
    }

    run_step(
        name="build_round1_memory_graph",
        command=[
            py,
            "build_dialogue_memory_graph.py",
            "--input",
            str(round1_frontier),
            "--output",
            str(graph),
            "--summary-output",
            str(graph_summary),
        ],
        outputs=[graph, graph_summary],
        args=args,
        summary=summary,
        is_complete=lambda: len(load_json(graph).get("graphs", [])) >= result_count(round1_frontier),
    )

    run_step(
        name="graph_aware_harden",
        command=[
            py,
            "graph_aware_hardening.py",
            "--source",
            str(round1_frontier),
            "--graph",
            str(graph),
            "--output-prefix",
            str(graph_hardened_prefix),
            "--max-dialogues",
            str(max_dialogues),
            "--turns-per-dialogue",
            "4",
            "--request-timeout",
            str(args.request_timeout),
            *(["--resume"] if args.resume else []),
        ],
        outputs=[graph_hardened, graph_hardened_report],
        args=args,
        summary=summary,
        is_complete=lambda: (
            first_json(graph_hardened_report).get("successful_graph_hardening_count", 0)
            + first_json(graph_hardened_report).get("failed_graph_hardening_count", 0)
        ) >= max_dialogues,
    )

    run_step(
        name="build_graph_hardened_memory_graph",
        command=[
            py,
            "build_dialogue_memory_graph.py",
            "--input",
            str(graph_hardened),
            "--output",
            str(graph_hardened_graph),
            "--summary-output",
            str(graph_hardened_graph_summary),
        ],
        outputs=[graph_hardened_graph, graph_hardened_graph_summary],
        args=args,
        summary=summary,
        is_complete=lambda: len(load_json(graph_hardened_graph).get("graphs", [])) >= result_count(graph_hardened),
    )

    run_step(
        name="graph_multiqa_temporal_v3",
        command=[
            py,
            "multi_qa_from_memory_graph.py",
            "--source",
            str(graph_hardened),
            "--graph",
            str(graph_hardened_graph),
            "--output-prefix",
            str(multiqa_prefix),
            "--max-dialogues",
            str(max_dialogues),
            "--qa-per-dialogue",
            str(args.qa_per_dialogue),
            "--candidate-pool-per-dialogue",
            str(args.candidate_pool_per_dialogue),
            "--allowed-evolution-types",
            "temporal_ordering",
            "--light-candidates-per-model",
            str(args.light_candidates_per_model),
            "--min-light-accuracy",
            str(args.min_light_accuracy),
            "--max-light-accuracy",
            str(args.max_light_accuracy),
            "--request-timeout",
            str(args.request_timeout),
            *(["--resume"] if args.resume else []),
        ],
        outputs=[multiqa, multiqa_report],
        args=args,
        summary=summary,
        is_complete=lambda: first_json(multiqa_report).get("attempted_dialogue_count", 0) >= max_dialogues,
    )

    run_step(
        name="multiqa_model_eval",
        command=[
            py,
            "evaluate_model_difficulty.py",
            "--input",
            str(multiqa),
            "--output",
            str(multiqa_eval),
            "--models",
            args.models,
            "--candidates-per-model",
            str(args.full_candidates_per_model),
        ],
        outputs=[multiqa_eval],
        args=args,
        summary=summary,
        is_complete=lambda: result_count(multiqa_eval) >= result_count(multiqa),
    )

    run_step(
        name="multiqa_frontier",
        command=[
            py,
            "frontier_filter.py",
            "--input",
            str(multiqa_eval),
            "--output",
            str(multiqa_frontier),
        ],
        outputs=[multiqa_frontier],
        args=args,
        summary=summary,
        is_complete=lambda: result_count(multiqa_frontier) >= result_count(multiqa_eval),
    )

    run_step(
        name="merge_training_candidates",
        command=[
            py,
            "merge_training_candidates.py",
            "--output",
            str(merged),
            "--report-output",
            str(merged_report),
            "--input",
            "hardened_base",
            str(round1_frontier),
            "--input",
            "graph_aware_multiqa",
            str(multiqa_frontier),
            "--keep-labels",
            "too_easy,medium,frontier,hard",
            "--max-too-easy",
            str(max_too_easy),
        ],
        outputs=[merged, merged_report],
        args=args,
        summary=summary,
        is_complete=lambda: bool(first_json(merged_report).get("training_candidate_count", 0)),
    )

    if merged_report.exists():
        summary["training_candidate_summary"] = first_json(merged_report)
    if multiqa_frontier.exists():
        summary["multiqa_frontier_label_counts"] = label_counts(multiqa_frontier)
    if graph_hardened_graph_summary.exists():
        summary["graph_hardened_memory_graph_summary"] = first_json(graph_hardened_graph_summary)

    save_json(pipeline_summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved pipeline summary to: {pipeline_summary_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
