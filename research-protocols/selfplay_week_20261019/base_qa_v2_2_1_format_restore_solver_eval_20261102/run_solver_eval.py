#!/usr/bin/env python3
"""Restore legacy display labels in a separate copy and run diagnostic Solver eval."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
from pathlib import Path
from typing import Any

ROOT = Path("/root/autodl-tmp/gxy-selfplay")
WEEK = ROOT / "research-protocols/selfplay_week_20261019"
EXPERIMENT = WEEK / "base_qa_v2_2_1_format_restore_solver_eval_20261102"
ACCEPTANCE = WEEK / "base_qa_v2_2_1_acceptance_20261102"
MODEL = Path("/root/autodl-tmp/Qwen3-4B")
MAX_MODEL_LEN = 12288
GPU_MEMORY_UTILIZATION = 0.8
SEED = 20261031
STOP = ["\n\n", "\n", "</answer>"]

SPECS = [
    {
        "key": "full_order",
        "candidate": ACCEPTANCE / "outputs/attempt_03/cc-train-episode-104814/frozen_review/AUTHOR_CANDIDATE.json",
        "bundle": ACCEPTANCE / "outputs/attempt_03/cc-train-episode-104814/frozen_review",
        "review": ACCEPTANCE / "outputs/quality_104814/QUALITY_REVIEW.json",
        "legacy_type": "ordering",
    },
    {
        "key": "partial_order",
        "candidate": ACCEPTANCE / "outputs/additional_104814/candidate_01_partial_order/CANDIDATE.json",
        "bundle": ACCEPTANCE / "outputs/additional_104814/candidate_01_partial_order/frozen_review",
        "review": ACCEPTANCE / "outputs/quality_104814_partial/QUALITY_REVIEW.json",
        "legacy_type": "partial_order",
    },
    {
        "key": "multi_select",
        "candidate": ACCEPTANCE / "outputs/additional_104814/candidate_02_actor_outcome_multiselect/CANDIDATE.json",
        "bundle": ACCEPTANCE / "outputs/additional_104814/candidate_02_actor_outcome_multiselect/frozen_review",
        "review": ROOT / "outputs/quality_104814_multiselect/REVIEW.json",
        "legacy_type": "multi_select",
    },
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_json(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return digest_bytes(raw)


def load_item(spec: dict[str, Any]) -> dict[str, Any]:
    candidate_bytes = spec["candidate"].read_bytes()
    candidate = json.loads(candidate_bytes)
    bundle = spec["bundle"]
    source = read_json(bundle / "SOURCE.json")
    scope = read_json(bundle / "SOURCE_SCOPE.json")
    render = read_json(bundle / "RENDER_CONTRACT.json")
    review = read_json(spec["review"])
    if candidate.get("candidate_id") != render.get("candidate_id"):
        raise ValueError(f"candidate/render binding mismatch: {spec['key']}")
    if scope.get("formal_qa_eligible_turn_refs") is None:
        raise ValueError(f"no exact admitted turn list: {spec['key']}")
    allowed = set(scope["formal_qa_eligible_turn_refs"])
    source_turns = source.get("turns", [])
    selected = [
        {"turn_id": row["turn_ref"], "speaker": row["speaker"], "text": row["text"]}
        for row in source_turns if row.get("turn_ref") in allowed
    ]
    selected_refs = {row["turn_id"] for row in selected}
    if selected_refs != allowed:
        missing = sorted(allowed - selected_refs)
        raise ValueError(f"admitted turn view incomplete for {spec['key']}: {missing[:5]}")
    display_map = render["display_to_canonical"]
    if set(display_map.values()) != set(candidate["options"]):
        raise ValueError(f"frozen display map does not cover options: {spec['key']}")

    if candidate["question_type"] == "multi_select":
        label_to_id = {label: display_map[label] for label in sorted(display_map)}
        id_to_label = {option_id: label for label, option_id in label_to_id.items()}
    else:
        old_labels = sorted(display_map)
        label_to_id = {f"I{index + 1}": display_map[old] for index, old in enumerate(old_labels)}
        id_to_label = {option_id: label for label, option_id in label_to_id.items()}

    gold = candidate["gold_answer"]
    if candidate["question_type"] == "ordering":
        gold_labels: Any = [id_to_label[option_id] for option_id in gold["option_ids"]]
    elif candidate["question_type"] == "partial_order":
        gold_labels = [[id_to_label[before], id_to_label[after]] for before, after in gold["pairs"]]
    elif candidate["question_type"] == "multi_select":
        gold_labels = sorted(id_to_label[option_id] for option_id in gold["option_ids"])
    else:
        raise ValueError(f"unsupported type {candidate['question_type']}")

    options = [{"id": label, "text": candidate["options"][option_id]}
               for label, option_id in label_to_id.items()]
    cited_refs = {
        evidence["turn_ref"]
        for item in candidate.get("item_analysis", {}).values()
        for evidence in item.get("evidence", [])
    }
    if not cited_refs <= allowed:
        raise ValueError(f"candidate cites non-admitted evidence in {spec['key']}: {sorted(cited_refs - allowed)}")

    return {
        "key": spec["key"], "candidate_id": candidate["candidate_id"],
        "root_id": candidate["root_id"], "question_type": candidate["question_type"],
        "legacy_type": spec["legacy_type"], "question": candidate["question"],
        "options": options, "label_to_option_id": label_to_id,
        "gold_labels": gold_labels, "gold_answer": gold,
        "hold_status": review.get("overall_decision", review.get("overall_conclusion", "HOLD")),
        "candidate_path": str(spec["candidate"]),
        "candidate_sha256": digest_bytes(candidate_bytes),
        "source_sha256": scope["source_sha256"],
        "scope_sha256": digest_json(scope),
        "scope_global_formal_qa_eligible": scope.get("formal_qa_eligible"),
        "admitted_turn_count": len(selected),
        "admitted_turn_refs": [row["turn_id"] for row in selected],
        "source_turns": selected,
        "original_display_to_option_id": display_map,
    }


def build_prompt(item: dict[str, Any], turns: list[dict[str, str]], tokenizer: Any) -> str:
    dialogue = "\n".join(f"[{t['turn_id']}] {t['speaker']}: {t['text']}" for t in turns)
    if not dialogue:
        dialogue = "[No dialogue supplied.]"
    options = "\n".join(f"{row['id']}. {row['text']}" for row in item["options"])
    if item["question_type"] == "multi_select":
        format_hint = "Return only selected option labels separated by commas, for example A,C."
    elif item["question_type"] == "ordering":
        format_hint = "Return every item exactly once in order, using I-labels separated by >, for example I2>I1>I3."
    else:
        format_hint = "Return only the supported before/after pair(s) using <, for example I2<I1. Do not invent a total order."
    user = "\n".join([
        "Dialogue evidence:", dialogue, "", f"Question type: {item['legacy_type']}",
        f"Question: {item['question']}", "", "Items/options:", options, "", format_hint,
        "Answer only in the required format. Do not explain.",
    ])
    messages = [
        {"role": "system", "content": "You are a long-dialogue QA solver. Use only the supplied dialogue and items. Follow the requested answer format exactly."},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def extract_multi(text: str, valid: set[str]) -> list[str]:
    found = re.findall(r"(?<![A-Za-z0-9])([A-Z])(?![A-Za-z0-9])", text.upper())
    return [label for label in found if label in valid]


def extract_items(text: str, valid: set[str]) -> list[str]:
    found = re.findall(r"(?<![A-Za-z0-9])I\s*(\d+)(?![A-Za-z0-9])", text.upper())
    return [f"I{number}" for number in found if f"I{number}" in valid]


def extract_pairs(text: str, valid: set[str]) -> list[list[str]]:
    result = []
    for left, op, right in re.findall(r"(?<![A-Za-z0-9])(I\s*\d+)\s*([<>])\s*(I\s*\d+)(?![A-Za-z0-9])", text.upper()):
        before, after = re.sub(r"\s+", "", left), re.sub(r"\s+", "", right)
        if before not in valid or after not in valid or before == after:
            continue
        result.append([before, after] if op == "<" else [after, before])
    return result


def score(item: dict[str, Any], raw: str) -> dict[str, Any]:
    qtype = item["question_type"]
    valid = {row["id"] for row in item["options"]}
    if qtype == "multi_select":
        pred = extract_multi(raw, valid)
        pred_set, gold_set = set(pred), set(item["gold_labels"])
        precision = len(pred_set & gold_set) / len(pred_set) if pred_set else 0.0
        recall = len(pred_set & gold_set) / len(gold_set) if gold_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"parsed": pred, "exact_correct": pred_set == gold_set and len(pred) == len(pred_set),
                "partial_score": round(f1, 6), "score_type": "set_f1", "precision": precision, "recall": recall}
    if qtype == "ordering":
        pred = extract_items(raw, valid)
        gold = item["gold_labels"]
        gold_pairs = {(gold[i], gold[j]) for i in range(len(gold)) for j in range(i + 1, len(gold))}
        pred_unique = list(dict.fromkeys(pred))
        pred_pairs = {(pred_unique[i], pred_unique[j]) for i in range(len(pred_unique)) for j in range(i + 1, len(pred_unique))}
        correct_pairs = len(gold_pairs & pred_pairs)
        return {"parsed": pred, "exact_correct": pred == gold,
                "partial_score": round(correct_pairs / len(gold_pairs), 6) if gold_pairs else 0.0,
                "score_type": "correct_gold_pairwise_relations_over_all_gold_pairs",
                "item_completeness": round(len(set(pred)) / len(valid), 6),
                "predicted_pairwise_precision": round(correct_pairs / len(pred_pairs), 6) if pred_pairs else 0.0}
    pred_pairs_list = extract_pairs(raw, valid)
    pred_pairs = {tuple(pair) for pair in pred_pairs_list}
    gold_pairs = {tuple(pair) for pair in item["gold_labels"]}
    precision = len(pred_pairs & gold_pairs) / len(pred_pairs) if pred_pairs else 0.0
    recall = len(pred_pairs & gold_pairs) / len(gold_pairs) if gold_pairs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"parsed": pred_pairs_list, "exact_correct": pred_pairs == gold_pairs and len(pred_pairs_list) == len(pred_pairs),
            "partial_score": round(f1, 6), "score_type": "directed_pair_f1", "precision": precision, "recall": recall}


def make_samples(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples = []
    for item in items:
        for mode, turns in (("full_eligible_R02", item["source_turns"]), ("no_context", [])):
            samples.append({"sample_id": f"{item['key']}__{mode}", "candidate_id": item["candidate_id"],
                            "context_mode": mode, "turn_count": len(turns), "question_type": item["question_type"],
                            "question": item["question"], "options": item["options"], "gold_labels": item["gold_labels"],
                            "turns": turns})
    return samples


def validate_roundtrips(items: list[dict[str, Any]]) -> None:
    for item in items:
        mapping = item["label_to_option_id"]
        if item["question_type"] == "ordering":
            assert [mapping[label] for label in item["gold_labels"]] == item["gold_answer"]["option_ids"]
            assert score(item, ">".join(item["gold_labels"]))["exact_correct"]
            assert not score(item, ">".join(reversed(item["gold_labels"])))["exact_correct"]
        elif item["question_type"] == "partial_order":
            mapped = [[mapping[before], mapping[after]] for before, after in item["gold_labels"]]
            assert mapped == item["gold_answer"]["pairs"]
            assert score(item, ";".join("<".join(pair) for pair in item["gold_labels"]))["exact_correct"]
            assert not score(item, ">".join(item["gold_labels"][0]))["exact_correct"]
        else:
            assert set(mapping[label] for label in item["gold_labels"]) == set(item["gold_answer"]["option_ids"])
            assert score(item, ",".join(item["gold_labels"]))["exact_correct"]
            assert not score(item, ",".join(sorted(set(mapping) - set(item["gold_labels"])) ))["exact_correct"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true", help="Build and validate transformed packets without loading Qwen.")
    args = parser.parse_args()
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    items = [load_item(spec) for spec in SPECS]
    validate_roundtrips(items)
    samples = make_samples(items)
    for sample in samples:
        labels = {row["id"] for row in sample["options"]}
        if sample["question_type"] == "ordering" and any(not re.fullmatch(r"I\d+", label) for label in labels):
            raise ValueError("ordering labels must follow the legacy I1..In format")
        if sample["question_type"] == "partial_order" and any(not re.fullmatch(r"I\d+", label) for label in labels):
            raise ValueError("partial-order labels must follow the legacy I1..In format")
        if sample["question_type"] == "multi_select" and any(not re.fullmatch(r"[A-Z]", label) for label in labels):
            raise ValueError("multi-select labels must remain alphabetic")
    public_items = [{k: v for k, v in item.items() if k not in ("source_turns", "admitted_turn_refs")} for item in items]
    write_json(EXPERIMENT / "FORMAT_RESTORED_ITEMS.json", public_items)
    write_json(EXPERIMENT / "SOLVER_SAMPLES.json", samples)
    if args.prepare_only:
        print(json.dumps({"prepared_items": len(items), "samples": len(samples), "view_turn_counts": [x["admitted_turn_count"] for x in items]}, ensure_ascii=False))
        return 0

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True, local_files_only=True)
    prompts = [build_prompt(next(item for item in items if item["candidate_id"] == sample["candidate_id"]),
                            sample["turns"], tokenizer) for sample in samples]
    prompt_tokens = [len(tokenizer(prompt, add_special_tokens=False).input_ids) for prompt in prompts]
    if any(n + 64 > MAX_MODEL_LEN for n in prompt_tokens):
        raise SystemExit(f"Prompt overflow preflight: {prompt_tokens}")
    write_json(EXPERIMENT / "PROMPT_PREFLIGHT.json", {
        "model": str(MODEL), "tokenizer": str(MODEL), "max_model_len": MAX_MODEL_LEN,
        "prompt_token_counts": {sample["sample_id"]: count for sample, count in zip(samples, prompt_tokens)},
        "overflow": False, "truncation": False, "context_scope": "exact formal_qa_eligible_turn_refs from frozen SOURCE_SCOPE.json",
    })
    llm = LLM(model=str(MODEL), tokenizer=str(MODEL), trust_remote_code=True, dtype="bfloat16",
              max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
              seed=SEED, enable_prefix_caching=False)
    params = SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=64, stop=STOP, seed=SEED)
    generated = llm.generate(prompts, params)
    results = []
    for sample, prompt, token_count, request in zip(samples, prompts, prompt_tokens, generated):
        raw = request.outputs[0].text.strip()
        item = next(item for item in items if item["candidate_id"] == sample["candidate_id"])
        scored = score(item, raw)
        results.append({**{k: sample[k] for k in ("sample_id", "candidate_id", "context_mode", "turn_count", "question_type", "gold_labels")},
                        "prompt_token_count": token_count, "raw_prediction": raw, **scored,
                        "diagnostic_only": True, "candidate_hold_status": item["hold_status"]})
    config = {"model": str(MODEL), "tokenizer": str(MODEL), "base_model_only": True,
              "max_model_len": MAX_MODEL_LEN, "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
              "seed": SEED, "temperature": 0.0, "top_p": 1.0, "k": 1,
              "max_new_tokens": 64, "stop_sequences": STOP, "offline": True,
              "model_training": False, "retry": False}
    write_json(EXPERIMENT / "MODEL_CONFIG.json", config)
    write_json(EXPERIMENT / "SOLVER_RESULTS.json", {
        "schema_version": "format_restored_solver_diagnostic_v1",
        "interpretation": "All three source candidates remain HOLD. Solver outputs diagnose answerability/format only; they do not upgrade QA admission or establish a valid Solver weakness.",
        "global_formal_qa_eligible": False,
        "model_config": config,
        "items": [{"candidate_id": item["candidate_id"], "key": item["key"], "question_type": item["question_type"],
                   "hold_status": item["hold_status"], "source_sha256": item["source_sha256"],
                   "candidate_sha256": item["candidate_sha256"], "admitted_turn_count": item["admitted_turn_count"],
                   "label_to_option_id": item["label_to_option_id"], "gold_labels": item["gold_labels"],
                   "original_display_to_option_id": item["original_display_to_option_id"]} for item in items],
        "results": results,
        "summary": {"full_context_exact": sum(r["exact_correct"] for r in results if r["context_mode"] == "full_eligible_R02"),
                    "full_context_count": sum(r["context_mode"] == "full_eligible_R02" for r in results),
                    "no_context_exact": sum(r["exact_correct"] for r in results if r["context_mode"] == "no_context"),
                    "no_context_count": sum(r["context_mode"] == "no_context" for r in results)},
    })
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
