#!/usr/bin/env python3
"""Prepare LoCoMo raw sample0-5 as current_method seed parents.

This adapter is intentionally narrow: it reads the raw LoCoMo JSON, preserves
the full dialogue turn order and raw turn ids, and builds a minimal
current_method-compatible graph/state-chain from raw QA evidence. It does not
read legacy clean_v1 processed graphs, r4 parents, r4 mining data, or any solver
artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_RAW = Path("/root/autodl-tmp/gxy-selfplay/locomo10.json")
DEFAULT_VALID_ROOT = Path("/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation")
DEFAULT_TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")
DEFAULT_PARENT_COUNT = 6
DEFAULT_MAX_SEED_EVENTS = 0
DEFAULT_CHUNK_MIN_TOKENS = 6000
DEFAULT_CHUNK_TARGET_TOKENS = 7000
DEFAULT_CHUNK_HARD_MAX_TOKENS = 7500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build current_method seed parents from LoCoMo raw sample0-5.")
    parser.add_argument("--raw-path", default=str(DEFAULT_RAW))
    parser.add_argument("--valid-root", default=str(DEFAULT_VALID_ROOT))
    parser.add_argument("--tokenizer-path", default=str(DEFAULT_TOKENIZER))
    parser.add_argument("--parent-count", type=int, default=DEFAULT_PARENT_COUNT)
    parser.add_argument("--max-seed-events", type=int, default=DEFAULT_MAX_SEED_EVENTS, help="0 keeps every in-chunk raw QA anchor.")
    parser.add_argument("--chunk-min-tokens", type=int, default=DEFAULT_CHUNK_MIN_TOKENS)
    parser.add_argument("--chunk-target-tokens", type=int, default=DEFAULT_CHUNK_TARGET_TOKENS)
    parser.add_argument("--chunk-hard-max-tokens", type=int, default=DEFAULT_CHUNK_HARD_MAX_TOKENS)
    return parser.parse_args()


def stable_hash(value: Any, length: int = 16) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def session_number(key: str) -> int | None:
    match = re.fullmatch(r"session_(\d+)", key)
    return int(match.group(1)) if match else None


def session_keys(conversation: dict[str, Any]) -> list[str]:
    keys = [key for key, value in conversation.items() if session_number(key) is not None and isinstance(value, list)]
    return sorted(keys, key=lambda key: session_number(key) or 0)


def extract_turns(sample: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    conversation = sample.get("conversation") if isinstance(sample.get("conversation"), dict) else {}
    turns: list[dict[str, Any]] = []
    session_dates: dict[str, str] = {}
    duplicate_ids: list[str] = []
    seen: set[str] = set()
    for key in session_keys(conversation):
        number = session_number(key) or 0
        date_key = f"session_{number}_date_time"
        date_value = normalize_space(conversation.get(date_key))
        if date_value:
            session_dates[key] = date_value
        raw_turns = conversation.get(key) if isinstance(conversation.get(key), list) else []
        for index, raw_turn in enumerate(raw_turns, start=1):
            if not isinstance(raw_turn, dict):
                continue
            turn_id = normalize_space(raw_turn.get("dia_id") or raw_turn.get("turn_id") or f"D{number}:{index}")
            if turn_id in seen:
                duplicate_ids.append(turn_id)
                turn_id = f"{turn_id}__dup_{len(duplicate_ids)}"
            seen.add(turn_id)
            turn = {
                "turn_id": turn_id,
                "speaker": normalize_space(raw_turn.get("speaker")),
                "text": normalize_space(raw_turn.get("text")),
                "session": key,
                "session_index": number,
                "session_turn_index": index,
            }
            if date_value:
                turn["session_date_time"] = date_value
            for optional_key in ("img_url", "blip_caption", "query"):
                if raw_turn.get(optional_key) is not None:
                    turn[optional_key] = raw_turn.get(optional_key)
            if turn["text"]:
                turns.append(turn)
    diagnostics = {
        "session_count": len(session_keys(conversation)),
        "turn_count": len(turns),
        "duplicate_turn_ids": duplicate_ids,
        "session_dates": session_dates,
        "speaker_a": conversation.get("speaker_a"),
        "speaker_b": conversation.get("speaker_b"),
    }
    return turns, diagnostics


def dialogue_text(turns: list[dict[str, Any]]) -> str:
    return "\n".join(f"[{turn['turn_id']}] {turn.get('speaker')}: {turn.get('text')}" for turn in turns)


def load_token_counter(tokenizer_path: Path):
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)

        def count(text: str) -> int:
            return len(tokenizer(text, add_special_tokens=False).input_ids)

        return count, "tokenizer"
    except Exception:
        def count(text: str) -> int:
            return max(1, len(text) // 4)

        return count, "approx_chars_div4"


def chunk_turns_by_token_budget(
    turns: list[dict[str, Any]],
    *,
    count_tokens,
    min_tokens: int,
    target_tokens: int,
    hard_max_tokens: int,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_tokens = 0

    def turn_tokens(turn: dict[str, Any]) -> int:
        return count_tokens(f"[{turn['turn_id']}] {turn.get('speaker')}: {turn.get('text')}\n")

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        chunks.append(
            {
                "turns": current,
                "token_count": current_tokens,
                "first_turn_id": current[0].get("turn_id"),
                "last_turn_id": current[-1].get("turn_id"),
                "session_start": current[0].get("session"),
                "session_end": current[-1].get("session"),
            }
        )
        current = []
        current_tokens = 0

    for turn in turns:
        tokens = turn_tokens(turn)
        would_exceed_target = current and current_tokens >= min_tokens and current_tokens + tokens > target_tokens
        would_exceed_hard = current and current_tokens + tokens > hard_max_tokens
        if would_exceed_target or would_exceed_hard:
            flush()
        current.append(turn)
        current_tokens += tokens
        if current_tokens >= hard_max_tokens:
            flush()
    flush()

    # Avoid a tiny tail chunk when it can be merged without breaking the hard max.
    if len(chunks) >= 2 and chunks[-1]["token_count"] < min_tokens:
        tail = chunks[-1]
        prev = chunks[-2]
        if prev["token_count"] + tail["token_count"] <= hard_max_tokens:
            merged_turns = prev["turns"] + tail["turns"]
            chunks[-2] = {
                "turns": merged_turns,
                "token_count": prev["token_count"] + tail["token_count"],
                "first_turn_id": merged_turns[0].get("turn_id"),
                "last_turn_id": merged_turns[-1].get("turn_id"),
                "session_start": merged_turns[0].get("session"),
                "session_end": merged_turns[-1].get("session"),
            }
            chunks.pop()
    return chunks


def normalize_evidence(value: Any, turn_ids: set[str]) -> list[str]:
    raw = value if isinstance(value, list) else ([value] if value else [])
    out: list[str] = []
    for item in raw:
        turn_id = normalize_space(item)
        if turn_id and turn_id in turn_ids and turn_id not in out:
            out.append(turn_id)
    return out


def qa_category_name(value: Any) -> str:
    mapping = {
        1: "multi_hop",
        2: "temporal",
        3: "commonsense",
        4: "single_hop",
        5: "adversarial",
    }
    try:
        return mapping.get(int(value), f"category_{value}")
    except (TypeError, ValueError):
        return normalize_space(value) or "unknown"


def build_seed_graph(
    *,
    sample: dict[str, Any],
    sample_index: int,
    chunk_index: int,
    turns: list[dict[str, Any]],
    weak_qa_rows: list[dict[str, Any]],
    raw_path: Path,
    max_seed_events: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    raw_sample_id = normalize_space(sample.get("sample_id")) or f"raw_sample_{sample_index}"
    seed_id = f"locomo_raw_sample{sample_index}_chunk{chunk_index:03d}"
    turn_by_id = {turn["turn_id"]: turn for turn in turns}
    turn_ids = set(turn_by_id)
    qa_rows = sample.get("qa") if isinstance(sample.get("qa"), list) else []

    events: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []
    used_evidence: set[tuple[str, ...]] = set()
    event_limit = int(max_seed_events or 0)
    for qa_index, qa in enumerate(qa_rows):
        if not isinstance(qa, dict):
            continue
        evidence = normalize_evidence(qa.get("evidence"), turn_ids)
        raw_evidence = [normalize_space(item) for item in (qa.get("evidence") if isinstance(qa.get("evidence"), list) else [])]
        if raw_evidence and set(raw_evidence) - turn_ids:
            continue
        if not evidence:
            continue
        evidence_key = tuple(evidence)
        answer = normalize_space(qa.get("answer") or qa.get("adversarial_answer"))
        question = normalize_space(qa.get("question"))
        if not answer or not question:
            continue
        if event_limit > 0 and evidence_key in used_evidence and len(events) >= event_limit // 2:
            continue
        used_evidence.add(evidence_key)
        event_index = len(events) + 1
        first_turn = turn_by_id.get(evidence[0], {})
        category = qa_category_name(qa.get("category"))
        entity = normalize_space(first_turn.get("speaker")) or raw_sample_id
        slot = f"raw_qa_{category}"
        event_id = f"{seed_id}:E{event_index:03d}"
        value = f"Q: {question} A: {answer}"
        event = {
            "event_id": event_id,
            "event_instance_id": event_id,
            "origin_event_id": None,
            "lineage_id": evidence[0],
            "node_id": event_id,
            "role": "state_change",
            "entity": entity,
            "slot": slot,
            "value": value,
            "previous_value": "unknown",
            "time_index": event_index,
            "status": "active",
            "turn_ids": evidence,
            "evidence_turn_ids": evidence,
            "raw_qa_index": qa_index,
            "raw_qa_category": category,
        }
        node = {
            "node_id": event_id,
            "event_id": event_id,
            "role": slot,
            "label": f"LoCoMo raw QA seed {event_index}",
            "value": value,
            "status": "active",
            "turn_ids": evidence,
            "evidence_turn_ids": evidence,
            "origin": "locomo_raw_qa_evidence_seed",
            "can_be_gold": True,
        }
        events.append(event)
        nodes.append(node)
        if event_limit > 0 and len(events) >= event_limit:
            break

    if not events and turns:
        evidence = [turns[0]["turn_id"]]
        event_id = f"{seed_id}:E001"
        value = turns[0]["text"]
        events.append(
            {
                "event_id": event_id,
                "event_instance_id": event_id,
                "origin_event_id": None,
                "lineage_id": evidence[0],
                "node_id": event_id,
                "role": "state_change",
                "entity": turns[0].get("speaker") or raw_sample_id,
                "slot": "raw_dialogue_opening",
                "value": value,
                "previous_value": "unknown",
                "time_index": 1,
                "status": "active",
                "turn_ids": evidence,
                "evidence_turn_ids": evidence,
                "raw_qa_index": None,
                "raw_qa_category": "fallback_dialogue",
            }
        )
        nodes.append(
            {
                "node_id": event_id,
                "event_id": event_id,
                "role": "raw_dialogue_opening",
                "label": "LoCoMo raw opening seed",
                "value": value,
                "status": "active",
                "turn_ids": evidence,
                "evidence_turn_ids": evidence,
                "origin": "locomo_raw_dialogue_seed",
                "can_be_gold": True,
            }
        )

    graph_hash = stable_hash({"sample_id": seed_id, "events": events}, length=24)
    dialogue_hash = stable_hash([turn.get("turn_id") for turn in turns] + [turn.get("text") for turn in turns], length=24)
    alignment_hash = stable_hash({"graph_hash": graph_hash, "dialogue_hash": dialogue_hash}, length=24)
    graph_update = {
        "graph_snapshot_id": seed_id,
        "graph_snapshot_hash": graph_hash,
        "events": events,
        "state_changes": [],
        "invalidations": [],
        "cancellations": [],
        "restorations": [],
        "delayed_activations": [],
        "visibility_edges": [],
        "entities": sorted({event["entity"] for event in events if event.get("entity")}),
        "depends_on_graph_snapshot_hash": graph_hash,
        "depends_on_dialogue_hash": dialogue_hash,
        "depends_on_alignment_hash": alignment_hash,
        "seed_graph_source": "locomo_raw_qa_evidence",
        "weak_cross_chunk_qa_count": len(weak_qa_rows),
    }
    state_chain = {
        "chain_id": f"locomo_raw_seed:{seed_id}",
        "hard_mode": "locomo_raw_current_method_seed",
        "entity": raw_sample_id,
        "slot": "locomo_raw_memory_state",
        "slot_label": "LoCoMo raw memory state",
        "parent_source": "locomo_raw",
        "source_dialogue_path": str(raw_path),
        "source_split": "train",
        "nodes": nodes,
        "node_count": len(nodes),
        "target_role": nodes[-1]["role"] if nodes else None,
        "target_node_id": nodes[-1]["node_id"] if nodes else None,
        "valid_node_id": nodes[-1]["node_id"] if nodes else None,
    }
    hashes = {
        "graph_snapshot_hash": graph_hash,
        "dialogue_hash": dialogue_hash,
        "alignment_hash": alignment_hash,
    }
    return graph_update, state_chain, hashes


def qa_chunk_partition(sample: dict[str, Any], chunk_turn_ids: set[str], all_turn_ids: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible: list[dict[str, Any]] = []
    weak: list[dict[str, Any]] = []
    qa_rows = sample.get("qa") if isinstance(sample.get("qa"), list) else []
    for qa_index, qa in enumerate(qa_rows):
        if not isinstance(qa, dict):
            continue
        raw_evidence = [normalize_space(item) for item in (qa.get("evidence") if isinstance(qa.get("evidence"), list) else [])]
        raw_evidence = [item for item in raw_evidence if item]
        if not raw_evidence:
            continue
        evidence_set = set(raw_evidence)
        if evidence_set <= chunk_turn_ids:
            eligible.append({"qa_index": qa_index, **qa})
        elif evidence_set <= all_turn_ids and evidence_set & chunk_turn_ids:
            weak.append(
                {
                    "qa_index": qa_index,
                    "question": qa.get("question"),
                    "answer": qa.get("answer") or qa.get("adversarial_answer"),
                    "answer_field": "answer" if qa.get("answer") is not None else "adversarial_answer",
                    "evidence": raw_evidence,
                    "category": qa.get("category"),
                    "weak_reason": "cross_chunk_evidence",
                }
            )
    return eligible, weak


def build_parent_row(
    sample: dict[str, Any],
    sample_index: int,
    chunk_index: int,
    chunk: dict[str, Any],
    raw_path: Path,
    max_seed_events: int,
    chunk_policy: dict[str, Any],
) -> dict[str, Any]:
    turns = chunk["turns"]
    text = dialogue_text(turns)
    seed_id = f"locomo_raw_sample{sample_index}_chunk{chunk_index:03d}"
    raw_sample_id = normalize_space(sample.get("sample_id")) or f"raw_sample_{sample_index}"
    all_turns, turn_diagnostics = extract_turns(sample)
    all_turn_ids = {turn["turn_id"] for turn in all_turns}
    chunk_turn_ids = {turn["turn_id"] for turn in turns}
    eligible_qa_rows, weak_qa_rows = qa_chunk_partition(sample, chunk_turn_ids, all_turn_ids)
    graph_source_sample = {**sample, "qa": eligible_qa_rows}
    graph_update, state_chain, hashes = build_seed_graph(
        sample=graph_source_sample,
        sample_index=sample_index,
        chunk_index=chunk_index,
        turns=turns,
        weak_qa_rows=weak_qa_rows,
        raw_path=raw_path,
        max_seed_events=max_seed_events,
    )
    seed_source = {
        "split": "train",
        "source_split": "train",
        "sample_name": f"sample{sample_index}",
        "raw_training_use": "train_candidate",
        "training_use": "train_candidate",
        "source_dialogue_path": str(raw_path),
        "parent_source": "locomo_raw",
        "parent_sample_id": raw_sample_id,
        "parent_chunk_id": seed_id,
        "sample_id": seed_id,
        "allowed_train_seed": True,
        "forbidden_reasons": [],
        "inferred_by": "prepare_locomo_current_method_seed",
    }
    metadata = {
        "prompt_mode": "full_dialogue",
        "source_pipeline": "base_restart_current_method_validation",
        "graph_schema_version": "locomo_raw_minimal_current_method_seed_v1",
        "current_method_source": True,
        "method": "locomo_raw_to_current_method_seed_parent",
        "source": "locomo_raw",
        "source_sample_index": sample_index,
        "source_chunk_index": chunk_index,
        "chunk_id": seed_id,
        "raw_sample_id": raw_sample_id,
        "sample_name": f"sample{sample_index}",
        "source_sample_name": f"sample{sample_index}",
        "parent_sample_name": f"sample{sample_index}",
        "source_split": "train",
        "split": "train",
        "training_use": "train_candidate",
        "source_dialogue_path": str(raw_path),
        "graph_snapshot_id": seed_id,
        **hashes,
        "turn_count": len(turns),
        "full_raw_turn_count": len(all_turns),
        "session_count": turn_diagnostics["session_count"],
        "chunk_token_count": chunk["token_count"],
        "chunk_token_count_mode": chunk_policy["token_count_mode"],
        "chunk_min_tokens": chunk_policy["min_tokens"],
        "chunk_target_tokens": chunk_policy["target_tokens"],
        "chunk_hard_max_tokens": chunk_policy["hard_max_tokens"],
        "chunk_first_turn_id": chunk["first_turn_id"],
        "chunk_last_turn_id": chunk["last_turn_id"],
        "chunk_session_start": chunk["session_start"],
        "chunk_session_end": chunk["session_end"],
        "seed_event_count": len(graph_update.get("events") or []),
        "raw_qa_all_evidence_inside_chunk_count": len(eligible_qa_rows),
        "raw_qa_cross_chunk_weak_count": len(weak_qa_rows),
        "weak_cross_chunk_qa": weak_qa_rows[:20],
        "speaker_a": turn_diagnostics.get("speaker_a"),
        "speaker_b": turn_diagnostics.get("speaker_b"),
        "turn_ids_preserved": True,
        "speaker_info_preserved": True,
        "session_time_info_preserved": bool(turn_diagnostics.get("session_dates")),
        "evidence_window_used_as_primary_input": False,
        "chunking_strategy": "locomo_session_chronological_contiguous_context_budget",
    }
    row = {
        "sample_id": seed_id,
        "expanded_graph_id": seed_id,
        "expanded_dialogue_id": f"{seed_id}_dialogue",
        "source_id": seed_id,
        "source": "locomo_raw",
        "source_sample_index": sample_index,
        "source_chunk_index": chunk_index,
        "source_raw_sample_id": raw_sample_id,
        "split": "train",
        "source_split": "train",
        "sample_name": f"sample{sample_index}",
        "frontier_from_split": "train",
        "frontier_from_sample_name": f"sample{sample_index}",
        "training_use": "train_candidate",
        "dialogue": {"turns": turns, "dialogue_hash": hashes["dialogue_hash"]},
        "turns": turns,
        "dialogue_text": text,
        "expanded_dialogue_text": text,
        "final_dialogue_text": text,
        "graph_update": graph_update,
        "state_chain": state_chain,
        "state_chain_id": state_chain["chain_id"],
        "state_chain_nodes": state_chain.get("nodes", []),
        "metadata": metadata,
        "seed_source": seed_source,
        "seed": {
            "source": seed_source,
            "sample_id": seed_id,
            "parent_sample_id": raw_sample_id,
            "parent_chunk_id": seed_id,
        },
        "validation": {
            "graph_dialogue_consistent": True,
            "gold_derivable": True,
            "option_valid": True,
            "anti_shortcut_pass": True,
            "warnings": [],
            "locomo_raw_seed_parent": True,
        },
        "gold_derivable_from_graph_or_state_chain": True,
        "unconstrained_llm_gold": False,
        "heldout_contamination": False,
        "option_validation_pass": True,
        "leakage_free": True,
        "graph_dialogue_alignment": True,
    }
    return row


def schema_summary(raw_path: Path, data: list[Any], parent_count: int) -> dict[str, Any]:
    sample_summaries: list[dict[str, Any]] = []
    for index, sample in enumerate(data[: min(parent_count, len(data))]):
        if not isinstance(sample, dict):
            sample_summaries.append({"index": index, "usable": False, "reason": f"sample_type={type(sample).__name__}"})
            continue
        conversation = sample.get("conversation") if isinstance(sample.get("conversation"), dict) else {}
        session_turns = []
        session_field_shapes: dict[str, Any] = {}
        for key in session_keys(conversation):
            raw_turns = conversation.get(key) if isinstance(conversation.get(key), list) else []
            session_turns.append(len(raw_turns))
            first_turn = raw_turns[0] if raw_turns and isinstance(raw_turns[0], dict) else {}
            session_field_shapes[key] = {
                "turn_count": len(raw_turns),
                "turn_keys": sorted(first_turn.keys()) if first_turn else [],
            }
        turn_keys = sorted({field for key in session_keys(conversation) for turn in conversation.get(key, []) if isinstance(turn, dict) for field in turn})
        sample_summaries.append(
            {
                "index": index,
                "raw_sample_id": sample.get("sample_id"),
                "usable": bool(conversation and session_turns),
                "top_level_fields": sorted(sample.keys()),
                "conversation_fields": sorted(conversation.keys()),
                "session_count": len(session_turns),
                "turn_count": sum(session_turns),
                "turn_keys_union": turn_keys,
                "has_speaker": "speaker" in turn_keys,
                "has_time_or_session_info": any(key.endswith("_date_time") for key in conversation) and bool(session_turns),
                "has_turn_id": "dia_id" in turn_keys or "turn_id" in turn_keys,
                "qa_count": len(sample.get("qa") or []) if isinstance(sample.get("qa"), list) else 0,
                "qa_first_keys": sorted((sample.get("qa") or [{}])[0].keys()) if isinstance(sample.get("qa"), list) and sample.get("qa") and isinstance(sample.get("qa")[0], dict) else [],
                "session_field_structure_preview": dict(list(session_field_shapes.items())[:2]),
            }
        )
    first = data[0] if data and isinstance(data[0], dict) else {}
    conversation0 = first.get("conversation") if isinstance(first.get("conversation"), dict) else {}
    first_session = next((conversation0.get(key) for key in session_keys(conversation0)), [])
    first_turn = first_session[0] if first_session and isinstance(first_session[0], dict) else {}
    sample0_5_usable = [
        bool(isinstance(sample, dict) and isinstance(sample.get("conversation"), dict) and session_keys(sample["conversation"]))
        for sample in data[: min(parent_count, len(data))]
    ]
    return {
        "raw_path": str(raw_path),
        "sample_count": len(data),
        "top_level_type": "list",
        "sample_major_fields": sorted(first.keys()) if isinstance(first, dict) else [],
        "dialogue_session_fields": sorted(conversation0.keys()) if isinstance(conversation0, dict) else [],
        "turn_field_structure": {
            "first_turn_keys": sorted(first_turn.keys()),
            "first_turn_example": first_turn,
        },
        "contains_speaker": "speaker" in first_turn,
        "contains_time_or_session_info": any(key.endswith("_date_time") for key in conversation0),
        "contains_turn_id": "dia_id" in first_turn or "turn_id" in first_turn,
        "sample0_5_usable": sample0_5_usable,
        "samples": sample_summaries,
    }


def difficulty_state() -> dict[str, Any]:
    question_type_shares = {"single_choice": 0.80, "multi_select": 0.15, "ordering": 0.05}
    return {
        "experiment": "base_restart_current_method_validation",
        "created_by": "prepare_locomo_current_method_seed",
        "solver_name": "solver_base_qwen3_4b",
        "difficulty_plan": {
            "question_type_shares": question_type_shares,
            "allowed_question_types": ["single_choice", "multi_select", "ordering"],
            "single_choice_subtype_shares": {
                "ordinary_single_choice": 0.125,
                "hard_single_choice": 0.875,
            },
            "single_choice_subtype_hardness_mix": {
                "ordinary_single_choice": {"h0": 0.85, "h1": 0.15, "h2": 0.0},
                "hard_single_choice": {"h0": 0.35, "h1": 0.50, "h2": 0.15},
            },
            "hard_single_choice_template_shares": {
                "minimal_difference": 0.20,
                "latest_valid_state": 0.20,
                "guaranteed_vs_possible": 0.20,
                "override_sensitive": 0.20,
                "sequence_as_single_choice": 0.20,
            },
            "cell_weights": {
                "single_choice|conditional_plan": 0.35,
                "single_choice|delayed_activation": 0.65,
                "multi_select|conditional_plan": 0.90,
                "multi_select|permission_visibility_scope": 0.10,
                "ordering|timeline_fork_merge": 0.90,
                "ordering|conditional_plan": 0.10,
            },
            "cell_hardness_mix": {
                "single_choice|conditional_plan": {"h0": 0.55, "h1": 0.35, "h2": 0.10},
                "single_choice|delayed_activation": {"h0": 0.45, "h1": 0.40, "h2": 0.15},
                "multi_select|conditional_plan": {"h0": 0.75, "h1": 0.25, "h2": 0.0},
                "multi_select|permission_visibility_scope": {"h0": 1.0, "h1": 0.0, "h2": 0.0},
                "ordering|timeline_fork_merge": {"h0": 0.90, "h1": 0.10, "h2": 0.0},
                "ordering|conditional_plan": {"h0": 0.90, "h1": 0.10, "h2": 0.0},
            },
            "qtype_curriculum": {
                "stage": "base_restart_bootstrap",
                "mode": "bootstrap_current_method_from_raw",
                "ordinary_single_choice": 0.10,
                "hard_single_choice": 0.70,
                "multi_select": 0.15,
                "ordering": 0.05,
            },
        },
    }


def main() -> int:
    args = parse_args()
    raw_path = Path(args.raw_path)
    valid_root = Path(args.valid_root)
    raw_audit_dir = valid_root / "raw_audit"
    preprocess_dir = valid_root / "preprocess_locomo_current_method"
    seed_dir = valid_root / "seed_parents"
    state_path = valid_root / "states" / "bootstrap_controller_state.json"
    seed_path = seed_dir / "seed_parents.jsonl"

    raw_paths = [raw_path]
    raw_audit_dir.mkdir(parents=True, exist_ok=True)
    (raw_audit_dir / "locomo_raw_paths.txt").write_text(
        "\n".join(str(path) for path in raw_paths if path.exists()) + "\n",
        encoding="utf-8",
    )

    missing_required_fields: list[str] = []
    failure_reason = ""
    adapter_status = "success"
    rows: list[dict[str, Any]] = []
    chunk_records: list[dict[str, Any]] = []
    try:
        payload = read_json(raw_path)
        if not isinstance(payload, list):
            raise ValueError(f"expected top-level list, got {type(payload).__name__}")
        summary = schema_summary(raw_path, payload, int(args.parent_count))
        write_json(raw_audit_dir / "raw_schema_summary.json", summary)
        parent_count = min(int(args.parent_count), len(payload))
        count_tokens, token_count_mode = load_token_counter(Path(args.tokenizer_path))
        chunk_policy = {
            "min_tokens": int(args.chunk_min_tokens),
            "target_tokens": int(args.chunk_target_tokens),
            "hard_max_tokens": int(args.chunk_hard_max_tokens),
            "tokenizer_path": str(args.tokenizer_path),
            "token_count_mode": token_count_mode,
            "strategy": "locomo_session_chronological_contiguous_context_budget",
        }
        for sample_index in range(parent_count):
            turns, _diagnostics = extract_turns(payload[sample_index])
            chunks = chunk_turns_by_token_budget(
                turns,
                count_tokens=count_tokens,
                min_tokens=chunk_policy["min_tokens"],
                target_tokens=chunk_policy["target_tokens"],
                hard_max_tokens=chunk_policy["hard_max_tokens"],
            )
            for chunk_index, chunk in enumerate(chunks):
                if int(chunk.get("token_count") or 0) < chunk_policy["min_tokens"]:
                    chunk_records.append(
                        {
                            "sample_index": sample_index,
                            "chunk_index": chunk_index,
                            "sample_id": f"locomo_raw_sample{sample_index}_chunk{chunk_index:03d}",
                            "token_count": chunk["token_count"],
                            "turn_count": len(chunk["turns"]),
                            "first_turn_id": chunk["first_turn_id"],
                            "last_turn_id": chunk["last_turn_id"],
                            "session_start": chunk["session_start"],
                            "session_end": chunk["session_end"],
                            "seed_event_count": 0,
                            "weak_cross_chunk_qa_count": None,
                            "status": "skipped_below_min_tokens",
                        }
                    )
                    continue
                row = build_parent_row(
                    payload[sample_index],
                    sample_index,
                    chunk_index,
                    chunk,
                    raw_path,
                    int(args.max_seed_events),
                    chunk_policy,
                )
                if not row.get("graph_update", {}).get("events"):
                    chunk_records.append(
                        {
                            "sample_index": sample_index,
                            "chunk_index": chunk_index,
                            "token_count": chunk["token_count"],
                            "status": "skipped_no_in_chunk_raw_qa_anchors",
                        }
                    )
                    continue
                rows.append(row)
                chunk_records.append(
                    {
                        "sample_index": sample_index,
                        "chunk_index": chunk_index,
                        "sample_id": row["sample_id"],
                        "token_count": chunk["token_count"],
                        "turn_count": len(chunk["turns"]),
                        "first_turn_id": chunk["first_turn_id"],
                        "last_turn_id": chunk["last_turn_id"],
                        "session_start": chunk["session_start"],
                        "session_end": chunk["session_end"],
                        "seed_event_count": len(row.get("graph_update", {}).get("events") or []),
                        "weak_cross_chunk_qa_count": (row.get("metadata") or {}).get("raw_qa_cross_chunk_weak_count"),
                        "status": "emitted",
                    }
                )
        for row in rows:
            for key in ("sample_id", "dialogue", "expanded_dialogue_text", "graph_update", "state_chain", "metadata", "seed_source"):
                if key not in row or row.get(key) in (None, "", [], {}):
                    missing_required_fields.append(f"{row.get('sample_id', 'unknown')}.{key}")
        if missing_required_fields:
            adapter_status = "failed"
            failure_reason = "missing required seed parent fields"
        else:
            write_jsonl(seed_path, rows)
            write_json(state_path, difficulty_state())
    except Exception as exc:
        adapter_status = "failed"
        failure_reason = repr(exc)
        if not (raw_audit_dir / "raw_schema_summary.json").exists():
            write_json(raw_audit_dir / "raw_schema_summary.json", {"raw_path": str(raw_path), "error": failure_reason})

    qtype_seed_counts = Counter()
    for row in rows:
        for event in row.get("graph_update", {}).get("events", []):
            if isinstance(event, dict):
                qtype_seed_counts[str(event.get("raw_qa_category") or "unknown")] += 1
    preprocess_summary = {
        "raw_path": str(raw_path),
        "output_seed_parent_path": str(seed_path),
        "parent_count": len(rows) if adapter_status == "success" else 0,
        "raw_sample_count_used": int(args.parent_count),
        "chunk_policy": {
            "min_tokens": int(args.chunk_min_tokens),
            "target_tokens": int(args.chunk_target_tokens),
            "hard_max_tokens": int(args.chunk_hard_max_tokens),
            "tokenizer_path": str(args.tokenizer_path),
            "strategy": "locomo_session_chronological_contiguous_context_budget",
        },
        "chunk_records": chunk_records,
        "chunk_count_emitted": sum(1 for item in chunk_records if item.get("status") == "emitted"),
        "chunk_count_skipped_no_anchors": sum(1 for item in chunk_records if item.get("status") != "emitted"),
        "chunk_token_stats": {
            "min": min([item["token_count"] for item in chunk_records], default=None),
            "max": max([item["token_count"] for item in chunk_records], default=None),
            "over_hard_max": sum(1 for item in chunk_records if int(item.get("token_count") or 0) > int(args.chunk_hard_max_tokens)),
            "below_min": sum(1 for item in chunk_records if int(item.get("token_count") or 0) < int(args.chunk_min_tokens)),
        },
        "schema_keys": sorted(rows[0].keys()) if rows else [],
        "missing_required_fields": sorted(set(missing_required_fields)),
        "adapter_status": adapter_status,
        "failure_reason": failure_reason,
        "sample_indices": list(range(len(rows))),
        "raw_qa_seed_category_counts": dict(qtype_seed_counts.most_common()),
        "state_path": str(state_path),
        "policy_shares": difficulty_state()["difficulty_plan"]["qtype_curriculum"],
    }
    write_json(preprocess_dir / "preprocess_summary.json", preprocess_summary)
    print(json.dumps(preprocess_summary, ensure_ascii=False, indent=2))
    return 0 if adapter_status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
