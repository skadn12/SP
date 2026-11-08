from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "You are a long-dialogue memory QA solver. Read the provided dialogue evidence carefully, "
    "track time, speakers, relations, and state updates, then choose the single correct option. "
    "Answer with only the option letter and the option text."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MCQ eval data from original LoCoMo QA with evidence windows.")
    parser.add_argument("--input", default="locomo10.json")
    parser.add_argument("--sample-index", type=int, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def stable_hash(parts: list[Any]) -> str:
    text = "\n".join(normalize(part) for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def parse_dia_id(value: str) -> tuple[int, int] | None:
    match = re.match(r"^D(\d+):(\d+)$", str(value).strip())
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def flatten_conversation(sample: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    conv = sample.get("conversation")
    if not isinstance(conv, dict):
        raise ValueError("sample conversation must be a dict")
    turns: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for session_index in range(1, 1000):
        session_key = f"session_{session_index}"
        if session_key not in conv:
            break
        session_time = conv.get(f"{session_key}_date_time")
        session = conv.get(session_key)
        if not isinstance(session, list):
            continue
        for turn_index, turn in enumerate(session, 1):
            if not isinstance(turn, dict):
                continue
            item = dict(turn)
            item["session_key"] = session_key
            item["session_time"] = session_time
            item["session_index"] = session_index
            item["turn_index"] = turn_index
            item["sort_key"] = (session_index, turn_index)
            turns.append(item)
            dia_id = str(item.get("dia_id") or "").strip()
            if dia_id:
                by_id[dia_id] = item
    return turns, by_id


def evidence_window(turns: list[dict[str, Any]], evidence_ids: list[str], window_size: int) -> list[dict[str, Any]]:
    positions = {str(turn.get("dia_id")): index for index, turn in enumerate(turns)}
    selected: dict[int, dict[str, Any]] = {}
    for evidence_id in evidence_ids:
        index = positions.get(str(evidence_id))
        if index is None:
            continue
        start = max(0, index - window_size)
        end = min(len(turns), index + window_size + 1)
        for pos in range(start, end):
            selected[pos] = turns[pos]
    return [selected[pos] for pos in sorted(selected)]


def dialogue_text(window_turns: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    previous_session: str | None = None
    for turn in window_turns:
        session_key = str(turn.get("session_key") or "")
        if session_key and session_key != previous_session:
            session_time = turn.get("session_time")
            lines.append(f"## {session_key} ({session_time})" if session_time else f"## {session_key}")
            previous_session = session_key
        dia_id = str(turn.get("dia_id") or "").strip()
        speaker = str(turn.get("speaker") or "Speaker").strip()
        text = str(turn.get("text") or "").strip()
        if text:
            lines.append(f"[{dia_id}] {speaker}: {text}" if dia_id else f"{speaker}: {text}")
    return "\n".join(lines)


def answer_pool(qa_items: list[dict[str, Any]], category: Any | None = None) -> list[str]:
    answers: list[str] = []
    for item in qa_items:
        if category is not None and item.get("category") != category:
            continue
        answer = str(item.get("answer") or "").strip()
        if answer and normalize(answer) not in {normalize(a) for a in answers}:
            answers.append(answer)
    return answers


def choose_distractors(rng: random.Random, qa_items: list[dict[str, Any]], answer: str, category: Any) -> list[str]:
    answer_norm = normalize(answer)
    candidates = answer_pool(qa_items, category=category) + answer_pool(qa_items, category=None)
    clean: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        norm = normalize(candidate)
        if not norm or norm == answer_norm:
            continue
        if norm in answer_norm or answer_norm in norm:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        clean.append(candidate)
    rng.shuffle(clean)
    return clean[:3]


def option_label(index: int) -> str:
    return chr(ord("A") + index)


def build_user_prompt(dialogue: str, question: str, options: list[str]) -> str:
    option_lines = [f"{option_label(index)}. {option}" for index, option in enumerate(options)]
    return "\n\n".join(
        [
            "Dialogue evidence:",
            dialogue,
            "Question:",
            question,
            "Options:",
            "\n".join(option_lines),
        ]
    )


def count_map(values: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    data = load_json(args.input)
    sample = data[args.sample_index]
    turns, by_id = flatten_conversation(sample)
    qa_items = [item for item in sample.get("qa", []) if isinstance(item, dict) and str(item.get("answer") or "").strip()]
    if args.limit is not None:
        qa_items = qa_items[: args.limit]

    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for qa_index, qa in enumerate(qa_items):
        question = str(qa.get("question") or "").strip()
        answer = str(qa.get("answer") or "").strip()
        evidence = [str(item) for item in qa.get("evidence", []) if str(item).strip()]
        if not question or not answer or not evidence:
            skipped.append({"qa_index": qa_index, "reason": "missing_question_answer_or_evidence"})
            continue
        distractors = choose_distractors(rng, qa_items, answer, qa.get("category"))
        if len(distractors) < 3:
            skipped.append({"qa_index": qa_index, "reason": "not_enough_distractors"})
            continue
        options = [answer] + distractors[:3]
        rng.shuffle(options)
        correct_index = options.index(answer)
        window_turns = evidence_window(turns, evidence, args.window_size)
        if not window_turns:
            skipped.append({"qa_index": qa_index, "reason": "evidence_not_found"})
            continue
        dtext = dialogue_text(window_turns)
        record_id = stable_hash(["locomo_original", args.sample_index, qa_index, question, answer])
        records.append(
            {
                "record_id": record_id,
                "sample_name": f"sample{args.sample_index}",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(dtext, question, options)},
                    {"role": "assistant", "content": f"{option_label(correct_index)}. {answer}"},
                ],
                "metadata": {
                    "record_id": record_id,
                    "sample_name": f"sample{args.sample_index}",
                    "source": "original_locomo_evidence_window_mcq",
                    "qa_index": qa_index,
                    "category": qa.get("category"),
                    "evidence": evidence,
                    "window_size": args.window_size,
                    "correct_option_index": correct_index,
                    "correct_option_label": option_label(correct_index),
                    "answer": answer,
                    "question": question,
                    "evidence_found_count": sum(1 for eid in evidence if eid in by_id),
                    "window_turn_count": len(window_turns),
                },
            }
        )

    output_prefix = Path(args.output_prefix)
    write_jsonl(output_prefix.with_name(output_prefix.name + "_chat.jsonl"), records)
    summary = {
        "input": args.input,
        "sample_index": args.sample_index,
        "sample_id": sample.get("sample_id"),
        "window_size": args.window_size,
        "answered_qa_count": len([item for item in sample.get("qa", []) if isinstance(item, dict) and str(item.get("answer") or "").strip()]),
        "selected_count": len(records),
        "skipped_count": len(skipped),
        "category_counts": count_map([record["metadata"].get("category") for record in records]),
        "average_window_turn_count": round(sum(record["metadata"]["window_turn_count"] for record in records) / len(records), 4) if records else None,
        "skipped": skipped[:50],
        "notes": [
            "This is an original LoCoMo evidence-window MCQ eval set, not graph-generated QA.",
            "Distractors are sampled from other original LoCoMo answers, preferably within the same category.",
        ],
    }
    write_json(output_prefix.with_name(output_prefix.name + "_summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved original LoCoMo MCQ eval to: {output_prefix.with_name(output_prefix.name + '_chat.jsonl').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
