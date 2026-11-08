from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract one top-level LoCoMo sample into the local batch result format.")
    parser.add_argument("--input", default="locomo10.json")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--qa-start", type=int, default=0)
    parser.add_argument("--qa-count", type=int, default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def session_number(key: str) -> int | None:
    match = re.fullmatch(r"session_(\d+)", key)
    return int(match.group(1)) if match else None


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def normalize_evidence(value: Any) -> list[str]:
    if value is None:
        return []
    raw_items = value if isinstance(value, list) else [value]
    evidence: list[str] = []
    for item in raw_items:
        for part in re.split(r"[;,]\s*", str(item)):
            part = part.strip()
            if part:
                evidence.append(part)
    return evidence


def normalize_turns(sample: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    conversation = sample.get("conversation")
    if not isinstance(conversation, dict):
        raise ValueError("sample.conversation must be an object.")
    dialogue: list[dict[str, Any]] = []
    time_context: list[dict[str, Any]] = []
    for key in sorted(conversation, key=lambda item: session_number(item) or 9999):
        number = session_number(key)
        if number is None:
            continue
        turns = conversation.get(key)
        if not isinstance(turns, list):
            continue
        session_time = conversation.get(f"{key}_date_time")
        evidence_ids: list[str] = []
        for turn_index, raw_turn in enumerate(turns, start=1):
            if not isinstance(raw_turn, dict):
                continue
            turn = dict(raw_turn)
            turn.setdefault("dia_id", f"D{number}:{turn_index}")
            turn["turn_index"] = turn_index
            turn["timeline_order"] = number
            turn["session_key"] = key
            if turn_index == 1 and session_time:
                turn["session_time"] = session_time
            evidence_ids.append(str(turn.get("dia_id")))
            dialogue.append(turn)
        time_context.append(
            {
                "timeline_order": number,
                "session_key": key,
                "session_number": number,
                "session_time": session_time,
                "evidence_ids": evidence_ids,
            }
        )
    return dialogue, time_context


def main() -> int:
    args = parse_args()
    payload = load_json(args.input)
    if not isinstance(payload, list):
        raise ValueError("LoCoMo input must be a list of samples.")
    sample = payload[args.sample_index]
    if not isinstance(sample, dict):
        raise ValueError("Selected sample must be an object.")
    dialogue, time_context = normalize_turns(sample)
    qas = sample.get("qa")
    if not isinstance(qas, list):
        raise ValueError("sample.qa must be a list.")
    qa_end = len(qas) if args.qa_count is None else min(len(qas), args.qa_start + args.qa_count)
    selected_qas = qas[args.qa_start:qa_end]
    sample_id = str(sample.get("sample_id", f"sample-{args.sample_index}"))
    dialogue_words = sum(word_count(str(turn.get("text", ""))) for turn in dialogue)

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for offset, qa in enumerate(selected_qas):
        qa_index = args.qa_start + offset
        if not isinstance(qa, dict):
            failures.append({"qa_index": qa_index, "reason": "qa_not_object"})
            continue
        normalized_qa = dict(qa)
        answer = normalized_qa.get("answer", normalized_qa.get("correct_answer_text"))
        if answer is None or not str(answer).strip():
            failures.append({"qa_index": qa_index, "reason": "answer_empty"})
            continue
        normalized_qa["evidence"] = normalize_evidence(normalized_qa.get("evidence"))
        normalized_qa.setdefault("correct_answer_text", answer)
        normalized_qa.setdefault("original_question", normalized_qa.get("question"))
        normalized_qa.setdefault("original_answer", answer)
        normalized_qa.setdefault("original_evidence", normalized_qa.get("evidence", []))
        normalized_qa.setdefault("question_evolved", False)
        results.append(
            {
                "dataset_path": str(Path(args.input).resolve()),
                "sample_index": args.sample_index,
                "sample_id": sample_id,
                "qa_index": qa_index,
                "qa": normalized_qa,
                "original_dialogue_word_count": dialogue_words,
                "dialogue_time_context": time_context,
                "final_dialogue": dialogue,
                "final_dialogue_word_count": dialogue_words,
                "word_count_delta": 0,
                "time_order_annotations": [],
                "word_count_comparison": {
                    "original": dialogue_words,
                    "final": dialogue_words,
                    "delta": 0,
                },
            }
        )

    output = args.output or f"locomo_sample{args.sample_index}_qa{args.qa_start}_{qa_end}_round1.json"
    batch_payload = {
        "dataset_path": str(Path(args.input).resolve()),
        "sample_index": args.sample_index,
        "sample_id": sample_id,
        "qa_start": args.qa_start,
        "requested_qa_count": len(selected_qas),
        "successful_qa_count": len(results),
        "qa_indices": [result["qa_index"] for result in results],
        "total_original_dialogue_word_count": dialogue_words * len(results),
        "total_final_dialogue_word_count": dialogue_words * len(results),
        "total_word_count_delta": 0,
        "failed_count": len(failures),
        "failures": failures,
        "results": results,
    }
    save_json(output, batch_payload)
    print(json.dumps(
        {
            "output": str(Path(output).resolve()),
            "sample_id": sample_id,
            "qa_count": len(results),
            "session_count": len(time_context),
            "dialogue_turn_count": len(dialogue),
            "dialogue_word_count": dialogue_words,
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
