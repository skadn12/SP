#!/usr/bin/env python3
"""Prepare shuffled displays and audit coverage; never certify semantic quality.

The caller must separately approve source support and question semantics. This
module preserves candidates, emits a restricted public view, and keeps scoring
labels private. A new display is a new evaluation condition: historical scores
must not be attached to it.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "question_presentation_gate_v1"
GOLD_COUNTS = (2, 3, 4, 5, 6)


def _field(candidate: dict[str, Any], name: str, alias: str) -> Any:
    if name in candidate and alias in candidate and candidate[name] != candidate[alias]:
        raise ValueError(f"Conflicting {name}/{alias}")
    value = candidate.get(name, candidate.get(alias))
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _validate(candidate: dict[str, Any]) -> tuple[str, str, str, str, dict[str, str], list[str]]:
    if not isinstance(candidate, dict):
        raise ValueError("candidate must be an object")
    candidate_id = _field(candidate, "candidate_id", "id")
    root_id = _field(candidate, "root_id", "root")
    kind = _field(candidate, "question_type", "type")
    operation = _field(candidate, "task_operation", "operation")
    if kind not in {"multi_select", "ordering"}:
        raise ValueError("question_type must be multi_select or ordering")
    if not isinstance(candidate.get("question"), str) or not candidate["question"].strip():
        raise ValueError("question must be a nonempty string")
    key = "options" if kind == "multi_select" else "items"
    choices = candidate.get(key)
    if not isinstance(choices, dict) or not all(
        isinstance(k, str) and k.strip() and isinstance(v, str) and v.strip()
        for k, v in choices.items()
    ):
        raise ValueError(f"{key} must map nonempty string IDs to nonempty strings")
    if kind == "multi_select" and len(choices) != 6:
        raise ValueError("multi_select must contain exactly six options")
    if kind == "ordering" and not 2 <= len(choices) <= 26:
        raise ValueError("ordering must contain between 2 and 26 items")
    gold = candidate.get("gold")
    if not isinstance(gold, list) or not all(isinstance(k, str) for k in gold):
        raise ValueError("gold must be a list of string IDs")
    if len(gold) != len(set(gold)):
        raise ValueError("gold must not contain duplicate IDs")
    if not set(gold).issubset(choices):
        raise ValueError("gold contains unknown IDs")
    if kind == "multi_select" and len(gold) not in GOLD_COUNTS:
        raise ValueError("multi_select gold count must be 2, 3, 4, 5, or 6")
    if kind == "ordering" and set(gold) != set(choices):
        raise ValueError("ordering gold must include every item exactly once")
    binding = candidate.get("source_binding")
    if not isinstance(binding, dict) or not isinstance(binding.get("sha256"), str):
        raise ValueError("source_binding must contain sha256")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", binding["sha256"]):
        raise ValueError("source_binding.sha256 must be a SHA-256 hex digest")
    return candidate_id, root_id, kind, operation, choices, gold


def lint_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Flag possible shortcuts for review, without deleting or rewriting text."""
    _, _, kind, _, choices, _ = _validate(candidate)
    flags: list[dict[str, Any]] = []
    if kind == "ordering":
        cue = re.compile(r"\b(?:first|initially|later|finally|last|lastly|concludes?|conclusion)\b|最初|后来|最后|首先|随后|总结", re.I)
        for item_id, value in choices.items():
            matches = [m.group() for m in cue.finditer(value)]
            if matches:
                flags.append({"code": "ORDERING_DISCOURSE_CUE", "item_id": item_id,
                              "matches": matches, "action": "semantic_review_required_do_not_auto_delete"})
        time = re.compile(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b|\b(?:1[0-2]|0?[1-9])\s*(?:a\.?m\.?|p\.?m\.?)\b|\b(?:noon|midnight)\b|(?:上午|下午|早上|晚上)?\d{1,2}点", re.I)
        if all(time.search(value) for value in choices.values()):
            flags.append({"code": "ALL_ORDERING_ITEMS_EXPOSE_CLOCK_TIME",
                          "action": "review_whether_source_is_needed"})
    else:
        quantity = re.compile(r"\b(?:select|choose|pick|mark)\s+(?:(?:exactly|the)\s+)?(?:[2-6]|two|three|four|five|six)\b|选(?:择|取)?\s*(?:恰好|正好)?\s*[二三四五六2-6]\s*(?:个|项|条)", re.I)
        matches = [m.group() for m in quantity.finditer(candidate["question"])]
        if matches:
            flags.append({"code": "QUESTION_REVEALS_SELECTION_COUNT", "matches": matches,
                          "action": "review_answer_count_hint"})
    return {"status": "mechanical_only", "semantically_approved": False,
            "flags": flags, "semantic_admission": "NOT_ASSESSED"}


def prepare_display(candidate: dict[str, Any], seed: int | str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (public, private), deterministically shuffled without inspecting gold.

    Nonidentity refers to the supplied choice order, not answer order. Independent
    shuffling can by chance match the answer order; this is not filtered using
    gold. The original candidate is never mutated.
    """
    candidate_id, root_id, kind, operation, choices, gold = _validate(candidate)
    if isinstance(seed, bool) or not isinstance(seed, (int, str)):
        raise ValueError("seed must be an integer or string")
    canonical_ids = list(choices)
    seed_material = json.dumps([SCHEMA_VERSION, seed, candidate_id, canonical_ids],
                               ensure_ascii=False, separators=(",", ":")).encode()
    rng = random.Random(int.from_bytes(hashlib.sha256(seed_material).digest(), "big"))
    permutation = canonical_ids[:]
    for _ in range(128):
        rng.shuffle(permutation)
        if permutation != canonical_ids:
            break
    else:  # Defensive fallback; independent of answer labels.
        permutation = canonical_ids[1:] + canonical_ids[:1]
    display_to_canonical = {chr(65 + i): key for i, key in enumerate(permutation)}
    canonical_to_display = {value: key for key, value in display_to_canonical.items()}
    choice_key = "options" if kind == "multi_select" else "items"
    binding = candidate["source_binding"]
    public_binding = {"sha256": binding["sha256"]}
    if isinstance(binding.get("path"), str):
        public_binding["path"] = binding["path"]
    if isinstance(binding.get("bytes"), int) and not isinstance(binding["bytes"], bool):
        public_binding["bytes"] = binding["bytes"]
    public = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate_id, "root_id": root_id,
        "question_type": kind, "task_operation": operation,
        "question": candidate["question"],
        choice_key: {label: choices[key] for label, key in display_to_canonical.items()},
        "source_binding": public_binding,
    }
    display_gold = [canonical_to_display[key] for key in gold]
    if kind == "multi_select":
        display_gold.sort()
    private = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate_id, "question_type": kind, "seed": seed,
        "candidate_sha256": hashlib.sha256(json.dumps(candidate, ensure_ascii=False,
            sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "public_sha256": hashlib.sha256(json.dumps(public, ensure_ascii=False,
            sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "display_to_canonical": display_to_canonical,
        "canonical_to_display": canonical_to_display,
        "canonical_gold": copy.deepcopy(gold), "display_gold": display_gold,
        "answer_comparison": "set" if kind == "multi_select" else "sequence",
        "nonidentity_permutation": permutation != canonical_ids,
        "lint": lint_candidate(candidate),
        "status": "mechanical_only", "semantically_approved": False,
        "historical_scores_reusable": False,
        "score_warning": "Changed display requires fresh Solver evaluation; do not transfer historical scores.",
    }
    return public, private


def restore_prediction(private: dict[str, Any], display_prediction: list[str]) -> list[str]:
    """Restore canonical IDs; MC answers are sets and ordering answers sequences."""
    if not isinstance(display_prediction, list) or not all(isinstance(x, str) for x in display_prediction):
        raise ValueError("display_prediction must be a list of string labels")
    if len(display_prediction) != len(set(display_prediction)):
        raise ValueError("prediction must not repeat labels")
    mapping = private["display_to_canonical"]
    if not set(display_prediction).issubset(mapping):
        raise ValueError("prediction contains unknown display labels")
    restored = [mapping[label] for label in display_prediction]
    return sorted(restored) if private["question_type"] == "multi_select" else restored


def is_correct_display_answer(private: dict[str, Any], display_prediction: list[str]) -> bool:
    restored = restore_prediction(private, display_prediction)
    gold = private["canonical_gold"]
    return set(restored) == set(gold) if private["question_type"] == "multi_select" else restored == gold


def coverage_report(candidates: list[dict[str, Any]], minimum_per_count: int = 1) -> dict[str, Any]:
    """Count validated MC candidates, without changing labels to fill quotas."""
    if isinstance(minimum_per_count, bool) or not isinstance(minimum_per_count, int) or minimum_per_count < 1:
        raise ValueError("minimum_per_count must be a positive integer")
    counts = {str(k): 0 for k in GOLD_COUNTS}
    invalid: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, candidate in enumerate(candidates):
        try:
            candidate_id, _, kind, _, _, gold = _validate(candidate)
            if candidate_id in seen:
                raise ValueError("duplicate candidate_id; repeated candidate cannot fill coverage")
            seen.add(candidate_id)
            if kind == "multi_select":
                counts[str(len(gold))] += 1
        except (ValueError, TypeError) as exc:
            invalid.append({"index": i, "error": str(exc)})
    deficits = {str(k): max(0, minimum_per_count - counts[str(k)]) for k in GOLD_COUNTS}
    return {"schema_version": SCHEMA_VERSION, "status": "mechanical_only",
            "semantically_approved": False, "minimum_per_count": minimum_per_count,
            "gold_count_histogram": counts, "deficits": deficits,
            "missing_gold_counts": [k for k in GOLD_COUNTS if deficits[str(k)] > 0],
            "coverage_complete": not any(deficits.values()) and not invalid,
            "invalid_candidates": invalid,
            "quota_policy": "Do not modify gold or relax semantic review to fill deficits.",
            "six_gold_note": "Six of six is a legitimate all-correct case, not distractor evidence; assess it within a mixed batch."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--require-coverage", action="store_true")
    parser.add_argument("--minimum-per-count", type=int, default=1)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    candidates = data.get("candidates") if isinstance(data, dict) else data
    if not isinstance(candidates, list):
        parser.error("input must be a candidate list or an object with candidates")
    coverage = coverage_report(candidates, args.minimum_per_count)
    public, private, rejected = [], [], []
    seen: set[str] = set()
    for i, candidate in enumerate(candidates):
        try:
            display, scoring = prepare_display(candidate, args.seed)
            if display["candidate_id"] in seen:
                raise ValueError("duplicate candidate_id")
            seen.add(display["candidate_id"])
            public.append(display)
            private.append(scoring)
        except (ValueError, TypeError) as exc:
            rejected.append({"index": i, "error": str(exc)})
    audit = {"schema_version": SCHEMA_VERSION, "status": "mechanical_only",
             "semantically_approved": False, "solver_ready_declared": False,
             "next_requirement": "Independent semantic admission and execution freeze are required.",
             "candidate_count": len(candidates), "prepared_count": len(public),
             "rejected": rejected, "coverage": coverage,
             "historical_scores_reusable": False}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for filename, value in (("PUBLIC_DISPLAYS.json", public), ("PRIVATE_SCORING.json", private),
                            ("PREPARATION_AUDIT.json", audit)):
        path = args.output_dir / filename
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        if filename == "PRIVATE_SCORING.json":
            path.chmod(0o600)
    print(json.dumps({"output_dir": str(args.output_dir), "prepared_count": len(public),
                      "coverage_complete": coverage["coverage_complete"],
                      "solver_ready_declared": False}, ensure_ascii=False))
    return 2 if rejected or (args.require_coverage and not coverage["coverage_complete"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
