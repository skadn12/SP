#!/usr/bin/env python3
"""Local diagnostic evaluation of frozen fresh cardinality candidates."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent
SOURCE_EXPERIMENT = WEEK / "base_qa_v2_2_1_fresh_cardinality_20261103"
LEGACY_FILE = WEEK / "base_qa_v2_2_1_format_restore_solver_eval_20261102/multiselect_permutation_repeat_20261102/run_multiselect_permutation_eval.py"
spec = importlib.util.spec_from_file_location("legacy_cardinality_eval", LEGACY_FILE)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)

MODEL = Path("/root/autodl-tmp/Qwen3-4B")
MAX_MODEL_LEN = 16384
GPU_MEMORY_UTILIZATION = 0.8
DISPLAY_SEED = 2026110303
SAMPLING_SEED_START = 2026110304
TEMPERATURE = 0.8
TOP_P = 0.9
MAX_NEW_TOKENS = 32
STOP = ["\n\n", "\n", "</answer>"]
MODES = ("full_context", "no_context")
PROTOCOL = "6 unique label permutations x 3 samples per permutation x Full/No-context"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_sha(value) -> str:
    return sha(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def offline():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"


def prepare():
    for name in ("REQUESTS.json", "ITEMS_PRIVATE.json", "PERMUTATIONS_PRIVATE.json", "PREFLIGHT.json"):
        if (HERE / name).exists():
            raise RuntimeError(f"Preparation already frozen: {name}")
    offline()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True, local_files_only=True)
    complete_path = SOURCE_EXPERIMENT / "CANDIDATES_COMPLETE.json"
    complete_bytes = complete_path.read_bytes()
    complete = json.loads(complete_bytes)
    source_bindings = {row["root_id"]: row["source"] for row in read(SOURCE_EXPERIMENT / "INPUT_FREEZE.json")["roots"]}
    items, variants_by_id, requests, prompt_paths, bindings = [], {}, [], [], [
        {"path": str(complete_path), "sha256_bytes": sha(complete_bytes)},
        {"path": str(SOURCE_EXPERIMENT / "INTEGRITY_AUDIT.json"), "sha256_bytes": sha((SOURCE_EXPERIMENT / "INTEGRITY_AUDIT.json").read_bytes())},
        {"path": str(SOURCE_EXPERIMENT / "INTEGRITY_ADDENDUM.json"), "sha256_bytes": sha((SOURCE_EXPERIMENT / "INTEGRITY_ADDENDUM.json").read_bytes())},
        {"path": str(LEGACY_FILE), "sha256_bytes": sha(LEGACY_FILE.read_bytes())},
    ]
    eligible = [row for row in complete["candidates"] if row["answer_status"] == "GOLD_AGREEMENT"]
    assert len(eligible) == 11
    assert next(row for row in complete["candidates"] if row["candidate_id"] == "c_river")["answer_status"] == "HOLD_ANSWER_AMBIGUITY"

    for item_index, source_item in enumerate(eligible):
        qid, root_id = source_item["candidate_id"], source_item["root_id"]
        labels = list(source_item["options"])
        assert labels == [chr(ord("A") + i) for i in range(len(labels))]
        n, g = len(labels), source_item["gold_count"]
        assert 6 <= n <= 8 and 2 <= g <= 6 and len(source_item["display_gold"]["option_ids"]) == g
        assert source_item["all_true_control"] == (g == n)
        legacy.LABELS = labels
        legacy.OUTPUT_PATTERN = re.compile(r"\s*([" + "".join(labels) + r"](?:\s*,\s*[" + "".join(labels) + r"])*)\s*\Z")
        legacy.QUESTION_ID = qid

        source_file = Path(source_bindings[root_id]["path"])
        source_bytes = source_file.read_bytes()
        assert sha(source_bytes) == source_bindings[root_id]["sha256_bytes"]
        source = read(source_file)
        source_canonical_sha256 = canonical_sha(source)
        turns = [{"turn_id": row["turn_ref"], "speaker": row["speaker"], "text": row["text"]} for row in source["turns"]]
        assert turns
        label_to_option_id = source_item["display_to_option_id"]
        assert set(label_to_option_id) == set(labels)
        assert all(source_item["options"][label] for label in labels)
        item = {
            "candidate_id": qid, "root_id": root_id, "question": source_item["question"],
            "question_type": "multi_select",
            "options": [{"id": label, "text": source_item["options"][label]} for label in labels],
            "label_to_option_id": label_to_option_id,
            "gold_labels": source_item["display_gold"]["option_ids"],
            "gold_option_ids": sorted(source_item["display_to_option_id"][label] for label in source_item["display_gold"]["option_ids"]),
            "diagnostic_only": True, "formal_qa_eligible": False,
            "all_true_control": source_item["all_true_control"], "n": n, "g": g,
            "content_certificate_status": source_item["certificate_status"],
        }
        assert set(item["gold_option_ids"]) == set(source_item["canonical_gold"]["option_ids"])
        assert len(turns) == len({row["turn_id"] for row in turns})
        variants = legacy.make_variants(item)
        assert len(variants) == 6 and len({tuple(v["display_to_option_id"].values()) for v in variants}) == 6
        variants_by_id[qid] = variants
        items.append({**item, "turn_count": len(turns), "source_path": str(source_file), "source_sha256_bytes": sha(source_bytes),
                      "source_canonical_sha256": source_canonical_sha256, "new_target_vs_historical_ten": source_item["new_target_vs_historical_ten"]})
        bindings.extend([
            {"path": str(source_file), "sha256_bytes": sha(source_bytes)},
            {"path": str(SOURCE_EXPERIMENT / "bundles" / qid / "PUBLIC_PACKET.json"), "sha256_bytes": sha((SOURCE_EXPERIMENT / "bundles" / qid / "PUBLIC_PACKET.json").read_bytes())},
            {"path": str(SOURCE_EXPERIMENT / "comparisons" / f"{qid}.json"), "sha256_bytes": sha((SOURCE_EXPERIMENT / "comparisons" / f"{qid}.json").read_bytes())},
        ])
        for mode in MODES:
            for variant in variants:
                for repeat in range(3):
                    variant_index = int(variant["variant_id"].split("_")[1])
                    sample_index = variant_index * 3 + repeat
                    seed = SAMPLING_SEED_START + item_index * 18 + sample_index
                    rid = f"{qid}__{mode}__{variant['variant_id']}__s{repeat:02d}"
                    prompt = legacy.build_prompt(item["question"], variant["options"], turns if mode == "full_context" else [], tokenizer)
                    # The prompt is built only from the public stem/options/source turns.
                    # Literal words such as "Gold" can naturally occur in source text.
                    tokens = len(tokenizer(prompt, add_special_tokens=False).input_ids)
                    assert tokens + MAX_NEW_TOKENS <= MAX_MODEL_LEN, (rid, tokens)
                    prompt_path = HERE / "prompts" / f"{rid}.txt"
                    prompt_path.parent.mkdir(parents=True, exist_ok=True)
                    if prompt_path.exists():
                        assert prompt_path.read_text(encoding="utf-8") == prompt, f"staged prompt changed: {rid}"
                    else:
                        with prompt_path.open("x", encoding="utf-8") as stream:
                            stream.write(prompt)
                    prompt_paths.append({"path": str(prompt_path), "sha256": sha(prompt.encode())})
                    requests.append({
                        "request_id": rid, "candidate_id": qid, "root_id": root_id,
                        "context_mode": mode, "variant_id": variant["variant_id"], "repeat_index": repeat,
                        "sampling_seed": seed, "display_to_option_id": variant["display_to_option_id"],
                        "options": variant["options"], "gold_labels": variant["gold_labels"],
                        "gold_option_ids": item["gold_option_ids"], "n": n, "g": g,
                        "all_true_control": item["all_true_control"],
                        "prompt_path": str(prompt_path.relative_to(HERE)), "prompt_sha256": sha(prompt.encode()),
                        "prompt_token_count": tokens, "turn_count": len(turns) if mode == "full_context" else 0,
                        "source_sha256_bytes": sha(source_bytes), "source_canonical_sha256": source_canonical_sha256,
                        "diagnostic_only": True,
                    })
    assert len(items) == 11 and len(requests) == 396
    assert sum(item["all_true_control"] for item in items) == 1
    write(HERE / "ITEMS_PRIVATE.json", items)
    write(HERE / "PERMUTATIONS_PRIVATE.json", variants_by_id)
    write(HERE / "REQUESTS.json", requests)
    write(HERE / "INPUT_BINDINGS.json", {"files": bindings, "prompts": prompt_paths,
        "solver_adapter_sha256": sha(Path(__file__).read_bytes()), "source_experiment_manifest_sha256": sha((SOURCE_EXPERIMENT / "MANIFEST_FINAL.json").read_bytes())})
    write(HERE / "MODEL_CONFIG.json", {
        "model": str(MODEL), "tokenizer": str(MODEL), "base_model_only": True,
        "temperature": TEMPERATURE, "top_p": TOP_P, "max_new_tokens": MAX_NEW_TOKENS,
        "stop_sequences": STOP, "max_model_len": MAX_MODEL_LEN,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION, "dtype": "bfloat16",
        "thinking": False, "offline": True, "automatic_retry": False, "training": False,
        "display_seed": DISPLAY_SEED, "sampling_seed_start": SAMPLING_SEED_START,
        "sampling_seed_rule": "unique paired seed per item/mapping/repeat; paired between context modes",
        "permutations_per_item": 6, "repeats_per_permutation": 3,
        "context_modes": list(MODES), "planned_requests": len(requests),
        "prompt": "unchanged legacy multi-select prompt; complete frozen source turns; no author analysis, Gold or candidate identifier",
        "excluded_candidate": {"candidate_id": "c_river", "reason": "blind HOLD: answer membership depends on unidentified caller ownership"},
        "all_true_control": "c_meadow included separately; exact accuracy and recall reported; J undefined when n==g",
        "formal_qa_eligible": False, "diagnostic_only": True,
    })
    write(HERE / "PREFLIGHT.json", {
        "status": "PASS", "request_count": len(requests), "candidate_count": len(items),
        "candidate_ids": [i["candidate_id"] for i in items],
        "full_context_turn_counts": {i["candidate_id"]: i["turn_count"] for i in items},
        "all_options_counts": dict(Counter(str(i["n"]) for i in items)),
        "gold_counts": dict(Counter(str(i["g"]) for i in items)),
        "min_prompt_tokens": min(r["prompt_token_count"] for r in requests),
        "max_prompt_tokens": max(r["prompt_token_count"] for r in requests),
        "truncation": False, "gold_or_author_analysis_fields_passed_to_prompt_builder": False,
        "preflight_attempt_01": "Stopped before inference by an overbroad substring check; existing staged prompts were byte-checked and reused.",
        "excluded_hold_candidate": "c_river", "control_separate": "c_meadow",
        "requests_sha256": sha((HERE / "REQUESTS.json").read_bytes()),
        "diagnostic_only": True, "formal_qa_eligible": False,
    })
    print(json.dumps(read(HERE / "PREFLIGHT.json"), ensure_ascii=False), flush=True)


def run():
    if (HERE / "STARTED.json").exists():
        raise RuntimeError("Inference already started; no retries or overwrite")
    requests = read(HERE / "REQUESTS.json")
    preflight = read(HERE / "PREFLIGHT.json")
    assert preflight["status"] == "PASS" and preflight["requests_sha256"] == sha((HERE / "REQUESTS.json").read_bytes())
    bindings = read(HERE / "INPUT_BINDINGS.json")
    for row in bindings["files"]:
        assert sha(Path(row["path"]).read_bytes()) == row["sha256_bytes"], row["path"]
    for row in bindings["prompts"]:
        assert sha(Path(row["path"]).read_bytes()) == row["sha256"], row["path"]
    prompts = [(HERE / row["prompt_path"]).read_text(encoding="utf-8") for row in requests]
    assert all(sha(prompt.encode()) == row["prompt_sha256"] for row, prompt in zip(requests, prompts))
    offline()
    import torch
    import transformers
    import vllm
    write(HERE / "STARTED.json", {
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "planned_requests": len(requests),
        "requests_sha256": sha((HERE / "REQUESTS.json").read_bytes()), "vllm_version": vllm.__version__,
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "model": str(MODEL), "gold_read_by_solver": False, "automatic_retry": False,
    })
    from vllm import LLM, SamplingParams
    llm = None
    raw_rows = []
    failed = False
    try:
        llm = LLM(model=str(MODEL), tokenizer=str(MODEL), trust_remote_code=True,
                  dtype="bfloat16", max_model_len=MAX_MODEL_LEN,
                  gpu_memory_utilization=GPU_MEMORY_UTILIZATION, seed=DISPLAY_SEED,
                  enable_prefix_caching=False)
        batch_size = 6
        for start in range(0, len(requests), batch_size):
            batch = requests[start:start + batch_size]
            params = [SamplingParams(n=1, temperature=TEMPERATURE, top_p=TOP_P,
                        max_tokens=MAX_NEW_TOKENS, stop=STOP, seed=row["sampling_seed"]) for row in batch]
            try:
                outputs = llm.generate(prompts[start:start + batch_size], params, use_tqdm=False)
                assert len(outputs) == len(batch)
                for row, output in zip(batch, outputs):
                    completion = output.outputs[0] if output.outputs else None
                    raw = {"request_id": row["request_id"], "status": "RETURNED",
                           "raw_prediction": completion.text if completion else "",
                           "finish_reason": completion.finish_reason if completion else None,
                           "stop_reason": completion.stop_reason if completion else None,
                           "completion_token_ids": list(completion.token_ids) if completion else [],
                           "prompt_sha256": row["prompt_sha256"], "automatic_retry": False,
                           "diagnostic_only": True}
                    write(HERE / "raw" / f"{row['request_id']}.json", raw)
                    raw_rows.append(raw)
            except Exception as exc:
                failed = True
                write(HERE / "BACKEND_FAILURE.json", {"batch_start": start, "error": repr(exc), "automatic_retry": False})
                for row in batch:
                    raw = {"request_id": row["request_id"], "status": "BACKEND_ERROR", "error": repr(exc), "automatic_retry": False}
                    write(HERE / "raw" / f"{row['request_id']}.json", raw)
                    raw_rows.append(raw)
                for row in requests[start + len(batch):]:
                    raw = {"request_id": row["request_id"], "status": "NOT_SUBMITTED", "automatic_retry": False}
                    write(HERE / "raw" / f"{row['request_id']}.json", raw)
                    raw_rows.append(raw)
                break
            print(json.dumps({"status": "INFERENCE_PROGRESS", "completed": len(raw_rows), "planned": len(requests)}), flush=True)
    finally:
        if llm is not None:
            del llm
            gc.collect()
    if len(raw_rows) != len(requests):
        raise RuntimeError(f"fixed denominator drift: {len(raw_rows)} vs {len(requests)}")
    write(HERE / "RAW_ANSWERS.json", raw_rows)
    if not failed:
        write(HERE / "COMPLETED.json", {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "returned_requests": len(raw_rows), "automatic_retry": False})
    score()


def finalize_preflight():
    """Validate frozen requests/prompts after a preparation-stage summary error."""
    if (HERE / "PREFLIGHT.json").exists() or (HERE / "STARTED.json").exists():
        raise RuntimeError("Preflight or inference already started")
    requests = read(HERE / "REQUESTS.json")
    items = read(HERE / "ITEMS_PRIVATE.json")
    bindings = read(HERE / "INPUT_BINDINGS.json")
    for row in bindings["files"]:
        assert sha(Path(row["path"]).read_bytes()) == row["sha256_bytes"], row["path"]
    for row in bindings["prompts"]:
        assert sha(Path(row["path"]).read_bytes()) == row["sha256"], row["path"]
    assert len(requests) == 396 and len(items) == 11
    assert len({r["request_id"] for r in requests}) == len(requests)
    assert Counter(r["context_mode"] for r in requests) == {"full_context": 198, "no_context": 198}
    for row in requests:
        raw = (HERE / row["prompt_path"]).read_bytes()
        assert sha(raw) == row["prompt_sha256"]
        assert len(row["display_to_option_id"]) == row["n"]
        assert len(set(row["display_to_option_id"].values())) == row["n"]
        assert len(row["gold_option_ids"]) == row["g"]
        assert set(row["gold_labels"]) == {
            label for label, oid in row["display_to_option_id"].items() if oid in row["gold_option_ids"]
        }
    item_by_id = {i["candidate_id"]: i for i in items}
    assert set(item_by_id) == {r["candidate_id"] for r in requests}
    assert all(item_by_id[r["candidate_id"]]["turn_count"] > 0 for r in requests if r["context_mode"] == "full_context")
    preflight = {
        "status": "PASS", "request_count": len(requests), "candidate_count": len(items),
        "candidate_ids": list(item_by_id),
        "full_context_turn_counts": {qid: item["turn_count"] for qid, item in item_by_id.items()},
        "all_options_counts": dict(Counter(str(len(i["options"])) for i in items)),
        "gold_counts": dict(Counter(str(len(i["gold_option_ids"])) for i in items)),
        "min_prompt_tokens": min(r["prompt_token_count"] for r in requests),
        "max_prompt_tokens": max(r["prompt_token_count"] for r in requests),
        "truncation": False, "gold_or_author_analysis_fields_passed_to_prompt_builder": False,
        "preparation_recovery": "REQUESTS.json and staged prompts were frozen before metadata summary failed; recovery verified hashes/mappings and created the missing preflight receipt.",
        "excluded_hold_candidate": "c_river", "control_separate": "c_meadow",
        "requests_sha256": sha((HERE / "REQUESTS.json").read_bytes()),
        "diagnostic_only": True, "formal_qa_eligible": False,
    }
    write(HERE / "PREFLIGHT.json", preflight)
    print(json.dumps(preflight, ensure_ascii=False), flush=True)


def score_one(raw, request):
    if raw["status"] != "RETURNED":
        return {"valid": False, "parsed": None, "exact_correct": False, "tp": None, "fp": None, "fn": None,
                "J": None if request["n"] == request["g"] else 0.0, "recall": 0.0,
                "invalid_policy": "included in fixed denominator; zero partial credit; item FP/FN unknown"}
    labels = sorted(request["display_to_option_id"])
    legacy.LABELS = labels
    legacy.OUTPUT_PATTERN = re.compile(r"\s*([" + "".join(labels) + r"](?:\s*,\s*[" + "".join(labels) + r"])*)\s*\Z")
    value = legacy.score_answer(raw["raw_prediction"], request, raw["finish_reason"])
    if not value["valid"]:
        return {**value, "J": None if request["n"] == request["g"] else 0.0, "recall": 0.0}
    pred = {request["display_to_option_id"][label] for label in value["parsed"]}
    gold = set(request["gold_option_ids"])
    tp, fp, fn = len(pred & gold), len(pred - gold), len(gold - pred)
    exact = pred == gold
    recall = tp / len(gold)
    if request["n"] == request["g"]:
        j_value = None
    else:
        j_value = float(Fraction(tp, request["g"]) - Fraction(fp, request["n"] - request["g"]))
    return {"valid": True, "parsed": value["parsed"], "predicted_option_ids": sorted(pred),
            "exact_correct": exact, "tp": tp, "fp": fp, "fn": fn, "recall": recall, "J": j_value,
            "J_fraction": None if j_value is None else str(Fraction(tp, request["g"]) - Fraction(fp, request["n"] - request["g"]))}


def group_summary(rows, all_true=False):
    valid = [r for r in rows if r["valid"]]
    preds = [tuple(r["predicted_option_ids"]) for r in valid]
    exact = sum(r["exact_correct"] for r in rows)
    out = {
        "planned_calls": len(rows), "valid_calls": len(valid), "invalid_calls": len(rows) - len(valid),
        "exact_correct_calls": exact, "exact_accuracy_fixed_denominator": exact / len(rows),
        "known_fp_total": sum(r["fp"] or 0 for r in valid), "known_fn_total": sum(r["fn"] or 0 for r in valid),
        "semantic_prediction_modes": len(set(preds)),
        "modal_semantic_repeat_agreement": max(Counter(preds).values()) / len(valid) if valid else None,
    }
    if all_true:
        out["mean_J"] = None
        out["J_status"] = "UNDEFINED: n-g=0; report exact accuracy and recall only"
        out["mean_recall"] = sum(r["recall"] for r in rows) / len(rows)
    else:
        out["mean_J"] = sum(r["J"] for r in rows) / len(rows)
        out["J_status"] = "J=TP/g-FP/(n-g); invalid answers included as zero"
        out["mean_recall"] = sum(r["recall"] for r in rows) / len(rows)
    return out


def score():
    requests, raws = read(HERE / "REQUESTS.json"), read(HERE / "RAW_ANSWERS.json")
    assert len(requests) == len(raws) == 396
    scored = []
    for request, raw in zip(requests, raws):
        assert request["request_id"] == raw["request_id"]
        row = score_one(raw, request)
        scored.append({**{k: request[k] for k in ("request_id", "candidate_id", "root_id", "context_mode", "variant_id", "repeat_index", "sampling_seed", "display_to_option_id", "gold_labels", "gold_option_ids", "n", "g", "all_true_control")},
                       **row, "raw_prediction": raw.get("raw_prediction"), "finish_reason": raw.get("finish_reason")})
    items = read(HERE / "ITEMS_PRIVATE.json")
    by_question = {}
    for item in items:
        qid = item["candidate_id"]
        by_question[qid] = {}
        for mode in MODES:
            subset = [r for r in scored if r["candidate_id"] == qid and r["context_mode"] == mode]
            summary = group_summary(subset, item["all_true_control"])
            summary["by_permutation"] = {}
            for permutation in range(6):
                group = [r for r in subset if r["variant_id"] == f"perm_{permutation:02d}"]
                summary["by_permutation"][f"perm_{permutation:02d}"] = group_summary(group, item["all_true_control"])
            summary["option_counts"] = {oid: {
                "gold": oid in item["gold_option_ids"],
                "selected": sum(oid in (r.get("predicted_option_ids") or []) for r in subset if r["valid"]),
                "missed": sum(oid in item["gold_option_ids"] and oid not in (r.get("predicted_option_ids") or []) for r in subset if r["valid"]),
                "false_selected": sum(oid not in item["gold_option_ids"] and oid in (r.get("predicted_option_ids") or []) for r in subset if r["valid"]),
            } for oid in sorted(item["label_to_option_id"].values())}
            by_question[qid][mode] = summary
    summaries = {}
    for mode in MODES:
        rows = [r for r in scored if r["context_mode"] == mode]
        normal = [r for r in rows if not r["all_true_control"]]
        control = [r for r in rows if r["all_true_control"]]
        summaries[mode] = {
            "non_control": group_summary(normal), "all_true_control": group_summary(control, True),
            "question_macro_exact_accuracy_non_control": sum(by_question[i["candidate_id"]][mode]["exact_accuracy_fixed_denominator"] for i in items if not i["all_true_control"]) / sum(not i["all_true_control"] for i in items),
        }
    score_doc = {
        "diagnostic_only": True, "formal_qa_eligible": False,
        "formula": "J=TP/g-FP/(n-g), n>g only; invalid answers receive zero and remain in the fixed denominator",
        "all_true_control_policy": "When n==g, J is undefined; report exact accuracy, recall and invalid output separately. Do not use J=TP/g.",
        "context_comparison": "Full complete dialogue is primary; no-context is a diagnostic baseline.",
        "items": by_question, "context_summaries": summaries, "requests": scored,
    }
    write(HERE / "SCORES.json", score_doc)
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "finalize_preflight", "run", "score"))
    args = parser.parse_args()
    {"prepare": prepare, "finalize_preflight": finalize_preflight, "run": run, "score": score}[args.mode]()


if __name__ == "__main__":
    main()
