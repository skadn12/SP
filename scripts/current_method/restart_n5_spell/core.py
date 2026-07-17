from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

LETTERS = ("A", "B", "C", "D")
BOXED_ANSWER_PATTERN = re.compile(r"^\s*<box>\s*([ABCD])\s*</box>\s*$")
BOXED_PARSER_VERSION = "boxed_letter_v1"
LEGACY_PARSER_MODE = "legacy_audit_only"
BOXED_PARSER_MODE = BOXED_PARSER_VERSION
NO_CONTEXT_PERMUTATION_DIAGNOSTIC_VERSION = "no_context_permutation_diagnostic_v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def normalize_option(text: str) -> str:
    return " ".join(str(text).strip().casefold().split())


def parse_mcq_response(raw: str, options: dict[str, str]) -> dict[str, Any]:
    """Legacy free-text parser retained exclusively for historical-run audit/replay."""
    text = str(raw or "").strip()
    hits = set(re.findall(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", text.upper()))
    if len(hits) == 1:
        letter = next(iter(hits))
        return {"parse_status": "valid_letter", "parsed_letter": letter,
                "parsed_option_text": options[letter]}
    if len(hits) > 1:
        return {"parse_status": "parse_invalid", "reason": "conflicting_letters"}
    norm = normalize_option(text)
    matches = [k for k, v in options.items() if normalize_option(v) == norm]
    if len(matches) == 1:
        letter = matches[0]
        return {"parse_status": "valid_option_text", "parsed_letter": letter,
                "parsed_option_text": options[letter]}
    return {"parse_status": "parse_invalid", "reason": "no_unique_semantic_option"}


def parse_boxed_mcq_answer(raw: str, options: dict[str, str]) -> dict[str, Any]:
    """Strict admission parser: the complete response must be one boxed A-D letter."""
    match = BOXED_ANSWER_PATTERN.fullmatch(str(raw or ""))
    if not match:
        return {"parse_status": "parse_invalid", "reason": "not_strict_boxed_letter",
                "parser_version": BOXED_PARSER_VERSION}
    letter = match.group(1)
    return {"parse_status": "valid_boxed_letter", "parsed_letter": letter,
            "parsed_option_text": options[letter], "parser_version": BOXED_PARSER_VERSION}


def local_reward(parsed: dict[str, Any], canonical_answer_text: str) -> int | None:
    if not str(parsed.get("parse_status", "")).startswith("valid_"):
        return None
    return int(normalize_option(parsed["parsed_option_text"]) == normalize_option(canonical_answer_text))


def collect_valid_rollouts(
    generator: Callable[[int, int, int], str], options: dict[str, str], answer: str,
    requested_valid: int, *, base_seed: int = 0, max_retries: int = 1,
) -> dict[str, Any]:
    records, rewards = [], []
    invalid = retries = exhausted = 0
    for slot in range(requested_valid):
        obtained = False
        for retry in range(max_retries + 1):
            seed = base_seed + slot * (max_retries + 1) + retry
            raw = generator(slot, retry, seed)
            parsed = parse_boxed_mcq_answer(raw, options)
            reward = local_reward(parsed, answer)
            record = {"raw_response": raw, **parsed, "local_reward": reward,
                      "sampling_seed": seed, "target_valid_slot": slot,
                      "parse_retry_index": retry}
            records.append(record)
            if reward is not None:
                rewards.append(reward)
                obtained = True
                break
            invalid += 1
            if retry < max_retries:
                retries += 1
        if not obtained:
            exhausted += 1
    return {"requested_valid_rollouts": requested_valid,
            "obtained_valid_rollouts": len(rewards), "raw_generations": len(records),
            "parse_invalid_generations": invalid, "retry_generations": retries,
            "exhausted_target_slots": exhausted, "verified_rewards": rewards,
            "rollouts": records}


def classify_reward_group(rewards: list[int], expected_k: int = 5) -> dict[str, Any]:
    if len(rewards) != expected_k:
        return {"decision": "reject_responder_format", "mean": None, "std": None}
    mean = sum(rewards) / expected_k
    std = math.sqrt(sum((x - mean) ** 2 for x in rewards) / expected_k)
    decision = "reject_easy" if mean == 1 else "reject_all_wrong" if mean == 0 else "retain_frontier"
    return {"decision": decision, "mean": mean, "std": std}


def classify_frontier_pre_audit(rewards: list[int]) -> dict[str, Any]:
    result = classify_reward_group(rewards, expected_k=5)
    if result["decision"] == "retain_frontier":
        result["decision"] = "retain_frontier_pre_audit"
    return result


def full_context_gate(grounding: dict[str, Any], rewards: list[int] | None = None) -> dict[str, Any]:
    if grounding.get("decision") != "pass_grounding":
        return {"run_full_context":False, "decision":f"skipped_{grounding.get('decision', 'unknown')}"}
    if rewards is None:
        return {"run_full_context":True, "decision":"scheduled"}
    return {"run_full_context":True, **classify_frontier_pre_audit(rewards)}


def grounding_decision(initial: list[int], extra: list[int] | None = None) -> dict[str, Any]:
    if len(initial) != 3:
        return {"decision": "bad_no_context_eval", "needs_extension": False}
    c3 = sum(initial)
    if c3 <= 1:
        return {"decision": "pass_grounding", "needs_extension": False}
    if c3 == 3:
        return {"decision": "bad_ground", "needs_extension": False}
    if extra is None:
        return {"decision": "needs_extension", "needs_extension": True}
    if len(extra) != 2:
        return {"decision": "bad_no_context_eval", "needs_extension": False}
    return {"decision": "bad_ground" if sum(initial + extra) >= 3 else "pass_grounding",
            "needs_extension": False}


def option_audit(orders: list[dict[str, Any]]) -> dict[str, Any]:
    if any(len(o.get("valid_rewards", [])) != 5 for o in orders):
        return {"decision": "option_audit_inconclusive"}
    rates = [sum(o["valid_rewards"]) / 5 for o in orders]
    spread = max(rates) - min(rates)
    threshold = lambda value, target: value + 1e-12 >= target
    severe = (max(rates) >= .8 and min(rates) <= .2 and threshold(spread, .6)
              and len({o.get("gold_letter") for o in orders}) > 1
              and all(o.get("context_hash") == orders[0].get("context_hash") for o in orders))
    return {"decision": "bad_option_bias" if severe else
            "option_order_sensitive" if threshold(spread, .4) else "pass",
            "correct_rates": rates, "accuracy_range": spread}


def classify_no_context_permutation_bias(orders: list[dict[str, Any]]) -> dict[str, Any]:
    """Diagnose why a no-context bad-ground item remains answerable after reordering."""
    if len(orders) != 3 or {row.get("order_id") for row in orders} != {
            "original", "perm1", "perm2"}:
        return {"decision":"bad_ground_permutation_inconclusive",
                "diagnostic_version":NO_CONTEXT_PERMUTATION_DIAGNOSTIC_VERSION}
    summaries = []
    all_letters: list[str] = []
    all_rewards: list[int] = []
    for row in orders:
        letters = row.get("valid_answer_letters")
        rewards = row.get("valid_rewards")
        gold = row.get("gold_letter")
        if (not isinstance(letters, list) or not isinstance(rewards, list) or
                len(letters) != 3 or len(rewards) != 3 or
                any(letter not in "ABCD" for letter in letters) or
                any(value not in (0, 1) for value in rewards) or gold not in "ABCD"):
            return {"decision":"bad_ground_permutation_inconclusive",
                    "diagnostic_version":NO_CONTEXT_PERMUTATION_DIAGNOSTIC_VERSION}
        all_letters.extend(letters)
        all_rewards.extend(rewards)
        summaries.append({"order_id":row["order_id"], "gold_letter":gold,
            "valid_answer_letters":letters, "valid_rewards":rewards,
            "semantic_correct_rate":sum(rewards) / 3})
    gold_letters = {row["gold_letter"] for row in summaries}
    semantic_leakage = all(value == 1 for value in all_rewards) and len(gold_letters) == 3
    position_bias = len(set(all_letters)) == 1 and len(gold_letters) == 3
    decision = ("bad_ground_semantic_leakage" if semantic_leakage else
                "bad_ground_position_bias" if position_bias else
                "bad_ground_mixed_bias")
    return {"decision":decision,
        "diagnostic_version":NO_CONTEXT_PERMUTATION_DIAGNOSTIC_VERSION,
        "orders":summaries, "overall_semantic_correct_rate":sum(all_rewards) / 9,
        "answer_letter_counts":{letter:all_letters.count(letter) for letter in "ABCD"},
        "correct_semantic_followed_across_all_orders":semantic_leakage,
        "single_letter_selected_across_all_orders":position_bias}


def context_preflight(prompt_tokens: int, context_tokens: int, qa_tokens: int,
                      reserve: int, max_model_len: int) -> dict[str, Any]:
    total = prompt_tokens + context_tokens + qa_tokens + reserve
    return {"total_tokens": total, "solver_max_model_len": max_model_len,
            "decision": "pass" if total <= max_model_len else "reject_context_overflow",
            "truncated": False}
