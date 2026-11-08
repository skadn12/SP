from __future__ import annotations

import argparse
import gc
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_old_improved_gate200 import render_prompt as render_frontier_prompt  # noqa: E402
from run_old_improved_gate200 import score_prediction  # noqa: E402


DEFAULT_R3_MODEL = "outputs/qwen3_4b_lora_r3_frontier_plus_clean_replay_from_r2_ctx15360_full_merged"
DEFAULT_R5_MODEL = "outputs/qwen3_4b_lora_old_improved_r5_train_small_from_r3_ctx15360_merged"
DEFAULT_FRONTIER = "outputs/old_improved_gate200/frontier.jsonl"
DEFAULT_TRAIN_MIX = "outputs/old_improved_r5_train_small_data/train.jsonl"
DEFAULT_CLEAN_RETENTION = "data/locomo_processed_datasets/_global_clean_v1_solver_r1/sft_data__clean_v1_solver_r1_valid_chat.jsonl"
DEFAULT_HELDOUT = "eval_sets/locomo_s0_s3_strictfresh_heldout200_chat.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-train eval for old_improved_r5_train_small.")
    parser.add_argument("--r3_model", default=DEFAULT_R3_MODEL)
    parser.add_argument("--r5_model", default=DEFAULT_R5_MODEL)
    parser.add_argument("--base_tokenizer", default="/root/autodl-tmp/Qwen3-4B")
    parser.add_argument("--frontier_jsonl", default=DEFAULT_FRONTIER)
    parser.add_argument("--train_mix_jsonl", default=DEFAULT_TRAIN_MIX)
    parser.add_argument("--clean_retention_jsonl", default=DEFAULT_CLEAN_RETENTION)
    parser.add_argument("--heldout_jsonl", default=DEFAULT_HELDOUT)
    parser.add_argument("--longbench_v2_jsonl", default=None)
    parser.add_argument("--quality_jsonl", default=None)
    parser.add_argument("--output_dir", default="outputs/old_improved_r5_train_small_eval")
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.86)
    parser.add_argument("--heldout_limit", type=int, default=None)
    parser.add_argument(
        "--phase",
        choices=["all", "r5_after", "r3_before", "report"],
        default="all",
        help="Run all phases in subprocesses, one model phase, or compile report from phase outputs.",
    )
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
    return rows


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("metadata")
    return value if isinstance(value, dict) else {}


def render_chat_prompt(tokenizer: Any, row: dict[str, Any]) -> str:
    messages = [m for m in row.get("messages") or [] if isinstance(m, dict) and m.get("role") != "assistant"]
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return "\n\n".join(f"{m.get('role')}: {m.get('content')}" for m in messages) + "\nassistant:"


def parse_label(text: str) -> str | None:
    text = str(text or "").strip()
    match = re.match(r"^\s*([A-Z])(?:[\.\):]|$)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"(?:answer|option|答案|选项)\s*[:：]?\s*([A-Z])\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([A-D])\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def gold_label(row: dict[str, Any]) -> str | None:
    for message in row.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "assistant":
            return parse_label(str(message.get("content") or ""))
    return None


def load_vllm(model_path: str, tokenizer_path: str, args: argparse.Namespace):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    llm = LLM(
        model=model_path,
        tokenizer=tokenizer_path,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    return tokenizer, llm, params


def unload_vllm(llm: Any) -> None:
    try:
        del llm
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def generate(llm: Any, prompts: list[str], params: Any) -> list[str]:
    outputs = llm.generate(prompts, params)
    return [item.outputs[0].text.strip() if item.outputs else "" for item in outputs]


def eval_frontier_with_model(model_path: str, frontier: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    tokenizer, llm, params = load_vllm(model_path, args.base_tokenizer, args)
    result = eval_frontier_loaded(model_path, tokenizer, llm, params, frontier)
    unload_vllm(llm)
    return result


def eval_frontier_loaded(model_path: str, tokenizer: Any, llm: Any, params: Any, frontier: list[dict[str, Any]]) -> dict[str, Any]:
    prompts = [render_frontier_prompt(tokenizer, sample) for sample in frontier]
    predictions = generate(llm, prompts, params)
    rows = []
    correct = 0
    partials = []
    for sample, prediction in zip(frontier, predictions):
        exact, partial, parsed = score_prediction(sample, prediction)
        correct += int(exact)
        partials.append(float(partial))
        rows.append(
            {
                "sample_id": sample.get("sample_id"),
                "question_type": sample.get("qa", {}).get("question_type"),
                "attack_skill": sample.get("attack_plan", {}).get("attack_skill"),
                "gold": sample.get("qa", {}).get("gold"),
                "prediction": prediction,
                "parsed": parsed,
                "correct": exact,
                "partial_credit": round(partial, 6),
            }
        )
    n = len(rows)
    return {
        "model": model_path,
        "n": n,
        "correct": correct,
        "accuracy": round(correct / n, 6) if n else 0.0,
        "mean_partial_credit": round(sum(partials) / n, 6) if n else 0.0,
        "rows": rows,
    }


def eval_chat_mcq_with_model(model_path: str, rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    tokenizer, llm, params = load_vllm(model_path, args.base_tokenizer, args)
    result = eval_chat_mcq_loaded(model_path, tokenizer, llm, params, rows)
    unload_vllm(llm)
    return result


def eval_chat_mcq_loaded(model_path: str, tokenizer: Any, llm: Any, params: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    prompts = [render_chat_prompt(tokenizer, row) for row in rows]
    predictions = generate(llm, prompts, params)
    details = []
    correct = 0
    missing = 0
    for row, prediction in zip(rows, predictions):
        gold = gold_label(row)
        pred = parse_label(prediction)
        ok = bool(gold and pred == gold)
        correct += int(ok)
        missing += int(pred is None)
        details.append(
            {
                "record_id": row.get("record_id") or metadata(row).get("record_id"),
                "sample_name": row.get("sample_name") or metadata(row).get("sample_name"),
                "gold_label": gold,
                "pred_label": pred,
                "correct": ok,
                "prediction": prediction,
                "metadata": metadata(row),
            }
        )
    n = len(details)
    return {
        "model": model_path,
        "n": n,
        "correct": correct,
        "accuracy": round(correct / n, 6) if n else 0.0,
        "missing_prediction_label_count": missing,
        "rows": details,
    }


def subset_train_mix(train_rows: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    return [row for row in train_rows if metadata(row).get("mix_source") == source]


def before_frontier_accuracy(frontier: list[dict[str, Any]]) -> float:
    if not frontier:
        return 0.0
    correct = 0
    for sample in frontier:
        preds = sample.get("solver_eval", {}).get("predictions")
        if isinstance(preds, list) and preds:
            correct += int(bool(preds[0].get("exact_correct")))
    return round(correct / len(frontier), 6)


def external_eval_payload(path: str | None, name: str) -> dict[str, Any]:
    if not path:
        return {"name": name, "available": False, "reason": "no local jsonl path provided", "accuracy": 0.0}
    if not Path(path).exists():
        return {"name": name, "available": False, "reason": f"path not found: {path}", "accuracy": 0.0}
    return {"name": name, "available": False, "reason": "custom external schema evaluator not configured", "path": path, "accuracy": 0.0}


def recommendation(report: dict[str, Any], external_available: bool) -> str:
    frontier_ok = report["frontier_delta"] >= 0.25
    clean_ok = report["clean_retention"] >= 0.95
    old_ok = report["old_hard_replay_delta"] >= -0.05
    heldout_ok = report["locomo_heldout_delta"] >= -0.05
    if frontier_ok and clean_ok and old_ok and heldout_ok and external_available:
        return "old_improved_line_promising_continue_to_gate400_or_r6"
    if frontier_ok:
        return "overfit_to_internal_frontier_need_more_replay_or_validation"
    return "do_not_continue_training_old_improved_line"


def build_markdown(report: dict[str, Any], external: dict[str, Any]) -> str:
    lines = [
        "# old_improved_r5_train_small Eval",
        "",
        f"- train_count: {report['train_count']}",
        f"- frontier_before_accuracy: {report['frontier_before_accuracy']}",
        f"- frontier_after_accuracy: {report['frontier_after_accuracy']}",
        f"- frontier_delta: {report['frontier_delta']}",
        f"- clean_retention: {report['clean_retention']}",
        f"- clean_retention_before_r3: {report['clean_retention_before_r3']}",
        f"- old_hard_replay_accuracy: {report['old_hard_replay_accuracy']}",
        f"- old_hard_replay_before_r3: {report['old_hard_replay_before_r3']}",
        f"- locomo_heldout_accuracy: {report['locomo_heldout_accuracy']}",
        f"- locomo_heldout_before_r3: {report['locomo_heldout_before_r3']}",
        f"- longbench_v2_accuracy: {report['longbench_v2_accuracy']}",
        f"- quality_accuracy: {report['quality_accuracy']}",
        f"- recommendation: {report['recommendation']}",
        "",
        "## External Eval",
        "",
        f"- LongBench-v2: {external['longbench_v2']}",
        f"- QuALITY: {external['quality']}",
        "",
    ]
    return "\n".join(lines)


def phase_command(args: argparse.Namespace, phase: str) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--phase",
        phase,
        "--r3_model",
        args.r3_model,
        "--r5_model",
        args.r5_model,
        "--base_tokenizer",
        args.base_tokenizer,
        "--frontier_jsonl",
        args.frontier_jsonl,
        "--train_mix_jsonl",
        args.train_mix_jsonl,
        "--clean_retention_jsonl",
        args.clean_retention_jsonl,
        "--heldout_jsonl",
        args.heldout_jsonl,
        "--output_dir",
        args.output_dir,
        "--max_model_len",
        str(args.max_model_len),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--gpu_memory_utilization",
        str(args.gpu_memory_utilization),
    ]
    if args.heldout_limit is not None:
        cmd.extend(["--heldout_limit", str(args.heldout_limit)])
    if args.longbench_v2_jsonl:
        cmd.extend(["--longbench_v2_jsonl", args.longbench_v2_jsonl])
    if args.quality_jsonl:
        cmd.extend(["--quality_jsonl", args.quality_jsonl])
    return cmd


def run_all_phases(args: argparse.Namespace) -> int:
    for phase in ["r5_after", "r3_before", "report"]:
        subprocess.run(phase_command(args, phase), check=True)
    return 0


def load_common_eval_sets(args: argparse.Namespace) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    frontier = load_jsonl(args.frontier_jsonl)
    train_mix = load_jsonl(args.train_mix_jsonl)
    clean_rows = load_jsonl(args.clean_retention_jsonl)
    old_hard_rows = subset_train_mix(train_mix, "old_hard_replay")
    heldout_rows = load_jsonl(args.heldout_jsonl)
    if args.heldout_limit is not None:
        heldout_rows = heldout_rows[: args.heldout_limit]
    return frontier, train_mix, clean_rows, old_hard_rows, heldout_rows


def run_r5_after_phase(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frontier, train_mix, clean_rows, old_hard_rows, heldout_rows = load_common_eval_sets(args)

    frontier_before = before_frontier_accuracy(frontier)
    r5_tokenizer, r5_llm, r5_params = load_vllm(args.r5_model, args.base_tokenizer, args)
    frontier_after = eval_frontier_loaded(args.r5_model, r5_tokenizer, r5_llm, r5_params, frontier)
    clean_after = eval_chat_mcq_loaded(args.r5_model, r5_tokenizer, r5_llm, r5_params, clean_rows)
    old_after = eval_chat_mcq_loaded(args.r5_model, r5_tokenizer, r5_llm, r5_params, old_hard_rows)
    heldout_after = eval_chat_mcq_loaded(args.r5_model, r5_tokenizer, r5_llm, r5_params, heldout_rows)
    unload_vllm(r5_llm)

    payload = {
        "frontier_before_accuracy": frontier_before,
        "frontier_after": frontier_after,
        "clean_after": clean_after,
        "old_after": old_after,
        "heldout_after": heldout_after,
        "train_count": len(train_mix),
        "frontier_count": len([row for row in train_mix if metadata(row).get("mix_source") == "old_improved_gate200_frontier"]),
        "clean_replay_count": len([row for row in train_mix if metadata(row).get("mix_source") == "clean_replay"]),
        "old_hard_replay_count": len(old_hard_rows),
    }
    write_json(output_dir / "_r5_after_phase.json", payload)
    print(json.dumps({"phase": "r5_after", "frontier_after_accuracy": frontier_after["accuracy"]}, ensure_ascii=False, indent=2))
    return 0


def run_r3_before_phase(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, clean_rows, old_hard_rows, heldout_rows = load_common_eval_sets(args)
    r3_tokenizer, r3_llm, r3_params = load_vllm(args.r3_model, args.base_tokenizer, args)
    clean_before = eval_chat_mcq_loaded(args.r3_model, r3_tokenizer, r3_llm, r3_params, clean_rows)
    old_before = eval_chat_mcq_loaded(args.r3_model, r3_tokenizer, r3_llm, r3_params, old_hard_rows)
    heldout_before = eval_chat_mcq_loaded(args.r3_model, r3_tokenizer, r3_llm, r3_params, heldout_rows)
    unload_vllm(r3_llm)

    payload = {
        "clean_before": clean_before,
        "old_before": old_before,
        "heldout_before": heldout_before,
    }
    write_json(output_dir / "_r3_before_phase.json", payload)
    print(json.dumps({"phase": "r3_before", "clean_before_accuracy": clean_before["accuracy"]}, ensure_ascii=False, indent=2))
    return 0


def run_report_phase(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    r5_payload = json.loads((output_dir / "_r5_after_phase.json").read_text(encoding="utf-8"))
    r3_payload = json.loads((output_dir / "_r3_before_phase.json").read_text(encoding="utf-8"))

    frontier_before = r5_payload["frontier_before_accuracy"]
    frontier_after = r5_payload["frontier_after"]
    clean_after = r5_payload["clean_after"]
    old_after = r5_payload["old_after"]
    heldout_after = r5_payload["heldout_after"]
    clean_before = r3_payload["clean_before"]
    old_before = r3_payload["old_before"]
    heldout_before = r3_payload["heldout_before"]

    write_json(output_dir / "frontier_recovery.json", {"before_accuracy": frontier_before, "after": frontier_after})

    write_json(output_dir / "clean_retention.json", {"source": args.clean_retention_jsonl, "before_r3": clean_before, "after_r5_small": clean_after})

    write_json(output_dir / "old_hard_replay_eval.json", {"source": args.train_mix_jsonl, "before_r3": old_before, "after_r5_small": old_after})

    write_json(output_dir / "heldout_eval.json", {"source": args.heldout_jsonl, "before_r3": heldout_before, "after_r5_small": heldout_after})

    external = {
        "longbench_v2": external_eval_payload(args.longbench_v2_jsonl, "LongBench-v2"),
        "quality": external_eval_payload(args.quality_jsonl, "QuALITY"),
    }
    write_json(output_dir / "external_eval.json", external)

    report = {
        "experiment": "old_improved_r5_train_small",
        "base_solver": "r3",
        "train_count": r5_payload["train_count"],
        "frontier_count": r5_payload["frontier_count"],
        "clean_replay_count": r5_payload["clean_replay_count"],
        "old_hard_replay_count": r5_payload["old_hard_replay_count"],
        "frontier_before_accuracy": frontier_before,
        "frontier_after_accuracy": frontier_after["accuracy"],
        "frontier_delta": round(frontier_after["accuracy"] - frontier_before, 6),
        "clean_retention": clean_after["accuracy"],
        "clean_retention_before_r3": clean_before["accuracy"],
        "clean_retention_delta": round(clean_after["accuracy"] - clean_before["accuracy"], 6),
        "old_hard_replay_accuracy": old_after["accuracy"],
        "old_hard_replay_before_r3": old_before["accuracy"],
        "old_hard_replay_delta": round(old_after["accuracy"] - old_before["accuracy"], 6),
        "locomo_heldout_accuracy": heldout_after["accuracy"],
        "locomo_heldout_before_r3": heldout_before["accuracy"],
        "locomo_heldout_delta": round(heldout_after["accuracy"] - heldout_before["accuracy"], 6),
        "longbench_v2_accuracy": external["longbench_v2"]["accuracy"],
        "quality_accuracy": external["quality"]["accuracy"],
        "external_eval_available": bool(external["longbench_v2"]["available"] and external["quality"]["available"]),
        "recommendation": "",
    }
    report["recommendation"] = recommendation(report, report["external_eval_available"])
    write_json(output_dir / "report.json", report)
    (output_dir / "report.md").write_text(build_markdown(report, external), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved eval outputs to: {output_dir.resolve()}")
    return 0


def main() -> int:
    args = parse_args()
    if args.phase == "all":
        return run_all_phases(args)
    if args.phase == "r5_after":
        return run_r5_after_phase(args)
    if args.phase == "r3_before":
        return run_r3_before_phase(args)
    if args.phase == "report":
        return run_report_phase(args)
    raise ValueError(f"unknown phase: {args.phase}")


if __name__ == "__main__":
    raise SystemExit(main())
