from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_old_improved_gate200 import solver_user_prompt  # noqa: E402


DEFAULT_CLEAN_REPLAY_CANDIDATES = [
    "data/locomo_processed_datasets/_global_clean_v1_solver_r1/sft_data__clean_v1_solver_r1_train_chat.jsonl",
    "data/locomo_processed_datasets/_global_clean_v1_solver_r1_training_dataset/actual_sft/clean_v1_solver_r1_train_chat.jsonl",
    "curriculum/clean_v1_train_optioneval_k5_split_easy_stabilizer_sample.jsonl",
]

DEFAULT_OLD_HARD_REPLAY_CANDIDATES = [
    "eval_sets/graph_expansion_challenger_v1/r3_frontier_plus_clean_replay_train_ctx15360/r2_frontier_plus_clean_replay_sft_train.jsonl",
    "eval_sets/graph_expansion_challenger_v1/r2_frontier_plus_clean_replay_train_with_s0_s2/r2_frontier_plus_clean_replay_sft_train.jsonl",
    "eval_sets/graph_expansion_challenger_v1/train_v12_hard_combined/solver_r2_hard_sft_train.jsonl",
]

SYSTEM_PROMPT = (
    "You are a long-dialogue graph-memory QA solver. Use only the dialogue evidence and the listed "
    "options if present. Follow the requested output format exactly."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build small old_improved r5 SFT mix from audited gate200 frontier.")
    parser.add_argument("--frontier_jsonl", required=True)
    parser.add_argument("--clean_replay_source", default=None)
    parser.add_argument("--old_hard_replay_source", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_clean_replay", type=int, default=80)
    parser.add_argument("--num_old_hard_replay", type=int, default=20)
    parser.add_argument("--tokenizer_model", default="/root/autodl-tmp/Qwen3-4B")
    parser.add_argument("--seed", type=int, default=20260615)
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


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stable_id(value: str, length: int = 16) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("metadata")
    return value if isinstance(value, dict) else {}


def first_existing(candidates: list[str], explicit: str | None) -> str:
    if explicit:
        if not Path(explicit).exists():
            raise FileNotFoundError(f"source not found: {explicit}")
        return explicit
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError(f"none of the candidate sources exist: {candidates}")


def frontier_answer(sample: dict[str, Any]) -> str:
    qa = sample.get("qa", {})
    qtype = str(qa.get("question_type") or "")
    gold = [str(item) for item in qa.get("gold", [])]
    if qtype == "multi_select":
        return ", ".join(gold)
    if qtype == "ordering":
        return " > ".join(gold)
    if qtype == "evidence_selection":
        return ", ".join(gold)
    if qtype == "invalidated_statement_detection":
        return " | ".join(gold)
    if qtype == "state_reconstruction":
        return gold[0] if gold else ""
    return ", ".join(gold)


def frontier_to_chat(sample: dict[str, Any]) -> dict[str, Any]:
    sample_id = str(sample.get("sample_id"))
    attack = sample.get("attack_plan", {})
    qa = sample.get("qa", {})
    user = solver_user_prompt(sample)
    assistant = frontier_answer(sample)
    return {
        "record_id": stable_id(f"old_improved_frontier::{sample_id}"),
        "sample_name": sample_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {
            "mix_source": "old_improved_gate200_frontier",
            "mix_source_id": sample_id,
            "training_use": "primary_train",
            "question_type": qa.get("question_type"),
            "attack_skill": attack.get("attack_skill"),
            "reasoning_mechanism": attack.get("reasoning_mechanism"),
            "solver_soft_accuracy": sample.get("solver_eval", {}).get("soft_accuracy"),
            "solver_partial_credit": sample.get("solver_eval", {}).get("partial_credit"),
            "frontier_category": sample.get("frontier_label", {}).get("category"),
            "gold_answer": assistant,
        },
    }


def normalize_chat_record(row: dict[str, Any], mix_source: str, index: int, source_path: str) -> dict[str, Any] | None:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        return None
    meta = dict(metadata(row))
    source_id = str(row.get("record_id") or meta.get("record_id") or f"{mix_source}_{index}")
    meta.update(
        {
            "mix_source": mix_source,
            "mix_source_id": source_id,
            "mix_source_path": source_path,
        }
    )
    return {
        "record_id": stable_id(f"{mix_source}::{source_id}::{index}"),
        "sample_name": row.get("sample_name") or meta.get("sample_name") or f"{mix_source}_{index:04d}",
        "messages": messages,
        "metadata": meta,
    }


def select_clean_replay(rows: list[dict[str, Any]], n: int, source_path: str, rng: random.Random) -> list[dict[str, Any]]:
    normalized = [item for idx, row in enumerate(rows) if (item := normalize_chat_record(row, "clean_replay", idx, source_path))]
    stabilizers = [row for row in normalized if metadata(row).get("training_use") == "stabilizer_train"]
    others = [row for row in normalized if row not in stabilizers]
    rng.shuffle(stabilizers)
    rng.shuffle(others)
    return (stabilizers + others)[:n]


def select_old_hard_replay(rows: list[dict[str, Any]], n: int, source_path: str, rng: random.Random) -> list[dict[str, Any]]:
    normalized = [item for idx, row in enumerate(rows) if (item := normalize_chat_record(row, "old_hard_replay", idx, source_path))]
    hard = [
        row
        for row in normalized
        if str(metadata(row).get("training_use") or "") in {"challenge_train", "primary_train"}
        or "hard" in str(metadata(row).get("difficulty_for_solver") or "").lower()
        or "frontier" in str(metadata(row).get("difficulty_for_solver") or "").lower()
    ]
    others = [row for row in normalized if row not in hard]
    rng.shuffle(hard)
    rng.shuffle(others)
    return (hard + others)[:n]


def render_for_count(tokenizer: Any, record: dict[str, Any]) -> int:
    messages = [m for m in record.get("messages", []) if isinstance(m, dict) and m.get("role") != "assistant"]
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    else:
        text = "\n\n".join(str(m.get("content") or "") for m in messages)
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def distribution(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(metadata(row).get(key) or "unknown") for row in rows))


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# old_improved_r5_train_small_data",
        "",
        f"- frontier_count: {report['frontier_count']}",
        f"- clean_replay_count: {report['clean_replay_count']}",
        f"- old_hard_replay_count: {report['old_hard_replay_count']}",
        f"- total_train_count: {report['total_train_count']}",
        f"- max_prompt_tokens: {report['max_prompt_tokens']}",
        f"- clean_replay_source: {report['clean_replay_source']}",
        f"- old_hard_replay_source: {report['old_hard_replay_source']}",
        f"- recommendation: {report['recommendation']}",
        "",
        "## Question Type Distribution",
        "",
    ]
    for key, value in report["question_type_distribution"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Attack Skill Distribution", ""])
    for key, value in report["attack_skill_distribution"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frontier_rows = load_jsonl(args.frontier_jsonl)
    frontier_chat = [frontier_to_chat(row) for row in frontier_rows]
    clean_source = first_existing(DEFAULT_CLEAN_REPLAY_CANDIDATES, args.clean_replay_source)
    hard_source = first_existing(DEFAULT_OLD_HARD_REPLAY_CANDIDATES, args.old_hard_replay_source)
    clean_rows = select_clean_replay(load_jsonl(clean_source), args.num_clean_replay, clean_source, rng)
    hard_rows = select_old_hard_replay(load_jsonl(hard_source), args.num_old_hard_replay, hard_source, rng)

    train_rows = frontier_chat + clean_rows + hard_rows
    rng.shuffle(train_rows)
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in train_rows:
        user = next((m.get("content", "") for m in row.get("messages", []) if isinstance(m, dict) and m.get("role") == "user"), "")
        assistant = next((m.get("content", "") for m in row.get("messages", []) if isinstance(m, dict) and m.get("role") == "assistant"), "")
        key = stable_id(str(user) + "\n" + str(assistant), 24)
        if key in seen:
            continue
        seen.add(key)
        row.setdefault("metadata", {})["train_mix_hash"] = key
        deduped.append(row)
    train_rows = deduped

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model, trust_remote_code=True)
        prompt_tokens = [render_for_count(tokenizer, row) for row in train_rows]
        max_prompt_tokens = max(prompt_tokens) if prompt_tokens else 0
    except Exception:
        max_prompt_tokens = 0

    report = {
        "frontier_count": len(frontier_chat),
        "clean_replay_count": len(clean_rows),
        "old_hard_replay_count": len(hard_rows),
        "total_train_count": len(train_rows),
        "question_type_distribution": distribution(train_rows, "question_type"),
        "attack_skill_distribution": distribution(train_rows, "attack_skill"),
        "mix_source_distribution": distribution(train_rows, "mix_source"),
        "max_prompt_tokens": max_prompt_tokens,
        "frontier_jsonl": args.frontier_jsonl,
        "clean_replay_source": clean_source,
        "old_hard_replay_source": hard_source,
        "recommendation": "ready_to_train" if len(frontier_chat) >= 25 and len(clean_rows) == args.num_clean_replay else "inspect_train_mix",
    }

    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_json(output_dir / "train_mix_report.json", report)
    (output_dir / "train_mix_report.md").write_text(build_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved train mix to: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
