#!/usr/bin/env python3
"""Run the legacy 6-mapping x 3-sample protocol on one frozen multi-select item."""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
MODEL = Path("/root/autodl-tmp/Qwen3-4B")
MAX_MODEL_LEN = 16384
GPU_MEMORY_UTILIZATION = 0.8
DISPLAY_SEED = 2026101903
SAMPLING_SEED_START = 2026101904
TEMPERATURE = 0.8
TOP_P = 0.9
MAX_NEW_TOKENS = 32
STOP = ["\n\n", "\n", "</answer>"]
LABELS = ["A", "B", "C", "D"]
QUESTION_KEY = "multi_select"
QUESTION_ID = "acc-104814-restaurant-match-multiselect-01"
OUTPUT_PATTERN = re.compile(r"\s*([A-D](?:\s*,\s*[A-D])*)\s*\Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def derived_seed(display_seed: int, question_id: str, variant: int) -> int:
    raw = f"{display_seed}|{question_id}|{variant}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def make_variants(item: dict[str, Any]) -> list[dict[str, Any]]:
    import random

    option_text = {row["id"]: row["text"] for row in item["options"]}
    option_text_by_id = {
        item["label_to_option_id"][label]: text for label, text in option_text.items()
    }
    option_ids = [item["label_to_option_id"][label] for label in LABELS]
    content_for_option = {
        option_id: "content_" + sha256_bytes(option_text[label].strip().encode("utf-8"))[:16]
        for label, option_id in item["label_to_option_id"].items()
    }
    option_for_content = {content: option_id for option_id, content in content_for_option.items()}
    ordered_content = [content_for_option[option_id] for option_id in option_ids]
    random.Random(derived_seed(DISPLAY_SEED, QUESTION_ID, -1)).shuffle(ordered_content)

    variants = []
    used: set[tuple[str, ...]] = set()
    for variant_index in range(6):
        rng = random.Random(derived_seed(DISPLAY_SEED, QUESTION_ID, variant_index))
        while True:
            order = list(ordered_content)
            rng.shuffle(order)
            if tuple(order) not in used:
                break
        used.add(tuple(order))
        display_to_option_id = {
            label: option_for_content[content] for label, content in zip(LABELS, order)
        }
        gold_option_ids = {
            item["label_to_option_id"][label] for label in item["gold_labels"]
        }
        gold_labels = sorted(
            label for label, option_id in display_to_option_id.items()
            if option_id in gold_option_ids
        )
        variants.append({
            "variant_id": f"perm_{variant_index:02d}",
            "display_to_option_id": display_to_option_id,
            "options": [
                {"id": label, "option_id": display_to_option_id[label],
                 "text": option_text_by_id[display_to_option_id[label]]}
                for label in LABELS
            ],
            "gold_labels": gold_labels,
        })
    return variants


def build_prompt(question: str, options: list[dict[str, str]], turns: list[dict[str, str]], tokenizer: Any) -> str:
    dialogue = "\n".join(
        f"[{row['turn_id']}] {row['speaker']}: {row['text']}" for row in turns
    ) or "[No dialogue supplied.]"
    rendered_options = "\n".join(f"{row['id']}. {row['text']}" for row in options)
    user = "\n".join([
        "Dialogue evidence:", dialogue, "", "Question type: multi_select",
        f"Question: {question}", "", "Options:", rendered_options, "",
        "Select all that apply. Return only selected option labels separated by commas, for example A,C.",
        "Answer only in the required format. Do not explain.",
    ])
    messages = [
        {"role": "system", "content": "You are a long-dialogue QA solver. Use only the supplied dialogue and options. Follow the requested answer format exactly."},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def prepare() -> None:
    for path in (HERE / "REQUESTS.json", HERE / "PERMUTATIONS.json", HERE / "PREFLIGHT.json"):
        if path.exists():
            raise SystemExit(f"Refusing to overwrite frozen preparation artifact: {path}")

    item_bytes = (BASE / "FORMAT_RESTORED_ITEMS.json").read_bytes()
    samples_bytes = (BASE / "SOLVER_SAMPLES.json").read_bytes()
    items = json.loads(item_bytes)
    samples = json.loads(samples_bytes)
    item = next(row for row in items if row["key"] == QUESTION_KEY)
    if item["candidate_id"] != QUESTION_ID or item["question_type"] != "multi_select":
        raise ValueError("frozen candidate identity/type mismatch")
    if [row["id"] for row in item["options"]] != LABELS:
        raise ValueError("this protocol adaptation requires exactly the frozen A-D options")
    if len(item["gold_labels"]) != 2:
        raise ValueError("unexpected Gold cardinality")

    sample_by_mode = {
        row["context_mode"]: row for row in samples
        if row["candidate_id"] == QUESTION_ID
    }
    if set(sample_by_mode) != {"full_eligible_R02", "no_context"}:
        raise ValueError("expected exact frozen full-context and no-context samples")
    variants = make_variants(item)
    if len({tuple(row["display_to_option_id"].values()) for row in variants}) != 6:
        raise ValueError("display mappings are not unique")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), trust_remote_code=True, local_files_only=True
    )
    requests = []
    for context_mode in ("full_eligible_R02", "no_context"):
        sample = sample_by_mode[context_mode]
        for variant in variants:
            for repeat_index in range(3):
                sample_index = int(variant["variant_id"].split("_")[1]) * 3 + repeat_index
                request_id = f"{context_mode}__{variant['variant_id']}__s{repeat_index:02d}"
                prompt = build_prompt(item["question"], variant["options"], sample["turns"], tokenizer)
                prompt_tokens = len(tokenizer(prompt, add_special_tokens=False).input_ids)
                if prompt_tokens + MAX_NEW_TOKENS > MAX_MODEL_LEN:
                    raise ValueError(f"prompt overflow: {request_id} ({prompt_tokens} tokens)")
                prompt_path = HERE / "prompts" / f"{request_id}.txt"
                prompt_path.write_text(prompt, encoding="utf-8")
                requests.append({
                    "request_id": request_id,
                    "candidate_id": QUESTION_ID,
                    "context_mode": context_mode,
                    "turn_count": len(sample["turns"]),
                    "variant_id": variant["variant_id"],
                    "sample_index": repeat_index,
                    "sample_index_within_condition": sample_index,
                    "sampling_seed": SAMPLING_SEED_START + sample_index,
                    "display_to_option_id": variant["display_to_option_id"],
                    "options": variant["options"],
                    "gold_labels": variant["gold_labels"],
                    "gold_option_ids": [item["label_to_option_id"][label] for label in item["gold_labels"]],
                    "prompt_path": str(prompt_path.relative_to(HERE)),
                    "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                    "prompt_token_count": prompt_tokens,
                    "source_sha256": item["source_sha256"],
                    "candidate_sha256": item["candidate_sha256"],
                    "candidate_status": item["hold_status"],
                    "global_formal_qa_eligible": item["scope_global_formal_qa_eligible"],
                })
    write_json(HERE / "PERMUTATIONS.json", {
        "schema_version": "legacy_mc_mapping_adaptation_v1",
        "algorithm": "day2 adapter.py variants: SHA256-derived seeds and unique content rebinding",
        "display_seed": DISPLAY_SEED,
        "candidate_id": QUESTION_ID,
        "label_count": len(LABELS),
        "variants": variants,
        "gold_labels_by_variant": {row["variant_id"]: row["gold_labels"] for row in variants},
    })
    write_json(HERE / "REQUESTS.json", requests)
    write_json(HERE / "MODEL_CONFIG.json", {
        "model": str(MODEL), "tokenizer": str(MODEL), "base_model_only": True,
        "temperature": TEMPERATURE, "top_p": TOP_P, "max_new_tokens": MAX_NEW_TOKENS,
        "stop_sequences": STOP, "max_model_len": MAX_MODEL_LEN,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "thinking": False, "offline": True, "automatic_retry": False,
        "sampling_seed_start": SAMPLING_SEED_START,
        "sampling_seed_rule": "same seed for corresponding mapping/repeat across full-context and no-context conditions",
        "legacy_protocol_model_difference": "legacy protocol used S0_M_clean merged weights; this run uses the requested base Qwen3-4B",
    })
    write_json(HERE / "PREFLIGHT.json", {
        "status": "PASS", "inference_started": False,
        "request_count": len(requests),
        "requests_per_context": {mode: sum(row["context_mode"] == mode for row in requests)
                                 for mode in sample_by_mode},
        "unique_variants": len(variants), "repeats_per_variant": 3,
        "input_bindings": {
            "FORMAT_RESTORED_ITEMS.json": sha256_bytes(item_bytes),
            "SOLVER_SAMPLES.json": sha256_bytes(samples_bytes),
        },
        "min_prompt_tokens": min(row["prompt_token_count"] for row in requests),
        "max_prompt_tokens": max(row["prompt_token_count"] for row in requests),
        "overflow": False, "truncation": False,
        "formal_result_eligible": False,
        "note": "Four-option adaptation of the legacy six-mapping, three-sample multi-select protocol; full R02 is the primary condition, no-context is a separate diagnostic control.",
    })
    print(json.dumps({"status": "PREPARED", "requests": len(requests),
                      "variants": len(variants), "prompt_tokens": [min(r["prompt_token_count"] for r in requests),
                                                                  max(r["prompt_token_count"] for r in requests)]}))


def score_answer(raw: str, row: dict[str, Any], finish_reason: str | None) -> dict[str, Any]:
    match = OUTPUT_PATTERN.fullmatch(raw.strip()) if isinstance(raw, str) else None
    parsed = [part.strip() for part in match.group(1).split(",")] if match else None
    valid = bool(parsed) and len(parsed) == len(set(parsed)) and finish_reason in ("stop", "eos")
    if not valid:
        return {"valid": False, "parsed": parsed, "exact_correct": False,
                "tp": None, "fp": None, "fn": None, "partial_score": 0.0,
                "partial_score_type": "legacy_penalized_multi_select",
                "invalid_policy": "fixed denominator; zero partial credit; item FP/FN unknown"}
    pred, gold = set(parsed), set(row["gold_labels"])
    tp, fp, fn = len(pred & gold), len(pred - gold), len(gold - pred)
    n, g = len(row["options"]), len(gold)
    partial = tp / g if g == n else tp / g - fp / (n - g)
    return {"valid": True, "parsed": parsed, "exact_correct": pred == gold,
            "tp": tp, "fp": fp, "fn": fn, "n": n, "g": g,
            "partial_score": partial,
            "partial_score_type": "legacy_penalized_multi_select"}


def run() -> None:
    if not (HERE / "REQUESTS.json").exists() or not (HERE / "PREFLIGHT.json").exists():
        raise SystemExit("Prepare and freeze requests before inference")
    if any((HERE / name).exists() for name in ("STARTED.json", "RAW_ANSWERS.json", "SCORES.json")):
        raise SystemExit("Refusing to overwrite prior inference/scoring outputs")
    preflight = read_json(HERE / "PREFLIGHT.json")
    if preflight.get("inference_started") is not False or preflight.get("status") != "PASS":
        raise ValueError("invalid preflight state")
    requests = read_json(HERE / "REQUESTS.json")
    prompts = []
    for row in requests:
        path = HERE / row["prompt_path"]
        prompt = path.read_text(encoding="utf-8")
        if sha256_bytes(prompt.encode("utf-8")) != row["prompt_sha256"]:
            raise ValueError(f"prompt hash mismatch: {row['request_id']}")
        prompts.append(prompt)

    write_json(HERE / "STARTED.json", {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "request_count": len(requests), "request_sha256": sha256_bytes((HERE / "REQUESTS.json").read_bytes()),
        "model": str(MODEL), "automatic_retry": False, "gold_read_by_solver": False,
    })

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    from vllm import LLM, SamplingParams

    raw_rows: list[dict[str, Any]] = []
    failed = False
    llm = None
    try:
        llm = LLM(model=str(MODEL), tokenizer=str(MODEL), trust_remote_code=True,
                  dtype="bfloat16", max_model_len=MAX_MODEL_LEN,
                  gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
                  seed=DISPLAY_SEED, enable_prefix_caching=False)
        batch_size = 6
        for start in range(0, len(requests), batch_size):
            request_batch = requests[start:start + batch_size]
            prompt_batch = prompts[start:start + batch_size]
            params = [SamplingParams(n=1, temperature=TEMPERATURE, top_p=TOP_P,
                                     max_tokens=MAX_NEW_TOKENS, stop=STOP,
                                     seed=row["sampling_seed"])
                      for row in request_batch]
            try:
                outputs = llm.generate(prompt_batch, params, use_tqdm=False)
            except BaseException as exc:
                failed = True
                for row in request_batch:
                    raw = {"request_id": row["request_id"], "status": "BACKEND_ERROR",
                           "error": f"{type(exc).__name__}: {exc}",
                           "context_mode": row["context_mode"], "variant_id": row["variant_id"],
                           "sample_index": row["sample_index"], "sampling_seed": row["sampling_seed"],
                           "automatic_retry": False}
                    write_json(HERE / "raw" / f"{row['request_id']}.json", raw)
                    raw_rows.append(raw)
                for row in requests[start + len(request_batch):]:
                    raw = {"request_id": row["request_id"], "status": "NOT_SUBMITTED",
                           "context_mode": row["context_mode"], "variant_id": row["variant_id"],
                           "sample_index": row["sample_index"], "sampling_seed": row["sampling_seed"],
                           "automatic_retry": False}
                    write_json(HERE / "raw" / f"{row['request_id']}.json", raw)
                    raw_rows.append(raw)
                break
            if len(outputs) != len(request_batch):
                failed = True
                raise RuntimeError(f"request/response count mismatch: {len(request_batch)} vs {len(outputs)}")
            for row, output in zip(request_batch, outputs):
                completion = output.outputs[0] if output.outputs else None
                raw = {
                    "request_id": row["request_id"], "status": "RETURNED",
                    "context_mode": row["context_mode"], "turn_count": row["turn_count"],
                    "variant_id": row["variant_id"], "sample_index": row["sample_index"],
                    "sampling_seed": row["sampling_seed"],
                    "prompt_sha256": row["prompt_sha256"],
                    "raw_prediction": completion.text if completion else "",
                    "finish_reason": completion.finish_reason if completion else None,
                    "completion_token_ids": list(completion.token_ids) if completion else [],
                    "candidate_status": "HOLD", "diagnostic_only": True,
                    "automatic_retry": False,
                }
                write_json(HERE / "raw" / f"{row['request_id']}.json", raw)
                raw_rows.append(raw)
            print(json.dumps({"status": "INFERENCE_PROGRESS",
                              "completed": min(start + batch_size, len(requests)),
                              "planned": len(requests)}), flush=True)
    finally:
        if llm is not None:
            del llm
            gc.collect()

    if len(raw_rows) != len(requests):
        raise RuntimeError(f"fixed denominator drift: {len(raw_rows)} vs {len(requests)}")
    write_json(HERE / "RAW_ANSWERS.json", raw_rows)
    scored = []
    for row, raw in zip(requests, raw_rows):
        if raw["status"] != "RETURNED":
            score = {"valid": False, "parsed": None, "exact_correct": False,
                     "tp": None, "fp": None, "fn": None, "partial_score": 0.0,
                     "partial_score_type": "legacy_penalized_multi_select",
                     "invalid_policy": "fixed denominator; zero partial credit; item FP/FN unknown"}
        else:
            score = score_answer(raw["raw_prediction"], row, raw["finish_reason"])
        predicted_option_ids = None
        if score["valid"]:
            predicted_option_ids = [row["display_to_option_id"][label] for label in score["parsed"]]
        scored.append({
            "request_id": row["request_id"], "context_mode": row["context_mode"],
            "variant_id": row["variant_id"], "sample_index": row["sample_index"],
            "display_to_option_id": row["display_to_option_id"],
            "gold_labels": row["gold_labels"], "gold_option_ids": row["gold_option_ids"],
            "raw_prediction": raw.get("raw_prediction"),
            "predicted_option_ids": predicted_option_ids,
            **score,
        })
    summaries = {}
    for mode in ("full_eligible_R02", "no_context"):
        rows = [row for row in scored if row["context_mode"] == mode]
        by_variant = {}
        for variant_index in range(6):
            variant_id = f"perm_{variant_index:02d}"
            variant_rows = [row for row in rows if row["variant_id"] == variant_id]
            semantic = [tuple(row["predicted_option_ids"] or []) for row in variant_rows if row["valid"]]
            by_variant[variant_id] = {
                "exact_correct": sum(row["exact_correct"] for row in variant_rows),
                "planned_repeats": len(variant_rows),
                "partial_score_mean_fixed_denominator": sum(row["partial_score"] for row in variant_rows) / len(variant_rows),
                "valid_repeats": sum(row["valid"] for row in variant_rows),
                "semantic_prediction_modes": len(set(semantic)),
                "repeat_agreement_fraction": (max(Counter(semantic).values()) / len(semantic)) if semantic else None,
            }
        semantic_predictions = [tuple(row["predicted_option_ids"] or []) for row in rows if row["valid"]]
        summaries[mode] = {
            "planned_calls": len(rows), "valid_calls": sum(row["valid"] for row in rows),
            "exact_calls": sum(row["exact_correct"] for row in rows),
            "exact_rate_fixed_denominator": sum(row["exact_correct"] for row in rows) / len(rows),
            "partial_score_mean_fixed_denominator": sum(row["partial_score"] for row in rows) / len(rows),
            "invalid_calls": sum(not row["valid"] for row in rows),
            "known_fp_total": sum(row["fp"] or 0 for row in rows if row["valid"]),
            "known_fn_total": sum(row["fn"] or 0 for row in rows if row["valid"]),
            "semantic_prediction_modes": len(set(semantic_predictions)),
            "semantic_option_selection_counts": {
                option_id: sum(option_id in (row["predicted_option_ids"] or []) for row in rows if row["valid"])
                for option_id in requests[0]["gold_option_ids"] + [
                    option_id for option_id in requests[0]["display_to_option_id"].values()
                    if option_id not in requests[0]["gold_option_ids"]
                ]
            },
            "display_label_selection_counts": {
                label: sum(label in (row["parsed"] or []) for row in rows if row["valid"])
                for label in LABELS
            },
            "permutation_repeat_summary": by_variant,
        }
    write_json(HERE / "SCORES.json", {
        "schema_version": "legacy_mc_permutation_repeat_scores_v1",
        "score_formula": "J = TP/g - FP/(n-g); invalid responses stay in denominator with zero partial credit and unknown item FP/FN",
        "candidate_id": QUESTION_ID, "candidate_status": "HOLD",
        "global_formal_qa_eligible": False, "diagnostic_only": True,
        "legacy_protocol_adaptation": {
            "mappings_per_condition": 6, "samples_per_mapping": 3,
            "main_condition": "full_eligible_R02", "control_condition": "no_context",
            "format_adaptation": "four options A-D; original wording and Gold frozen; option identity remapped by stable option ID",
            "model_adaptation": "base Qwen3-4B; legacy protocol used S0_M_clean merged weights",
            "automatic_retry": False, "training": False,
        },
        "summaries": summaries, "requests": scored,
        "inference_complete": not failed,
    })
    print(json.dumps({mode: {key: value for key, value in summary.items()
                             if key in ("planned_calls", "valid_calls", "exact_calls",
                                        "exact_rate_fixed_denominator", "partial_score_mean_fixed_denominator",
                                        "semantic_prediction_modes")}
                      for mode, summary in summaries.items()}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare-only", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        prepare()
    else:
        run()


if __name__ == "__main__":
    main()
