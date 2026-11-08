#!/usr/bin/env python3
"""Ten-root diagnostic adapter; reuse the frozen legacy prompt, shuffler and J scorer.

No generator changes, source approval, Gold changes, retries, or training.
"""
from __future__ import annotations

import argparse
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
INPUT = WEEK / "base_qa_v2_2_1_ten_root_20261103"
LEGACY_FILE = WEEK / "base_qa_v2_2_1_format_restore_solver_eval_20261102/multiselect_permutation_repeat_20261102/run_multiselect_permutation_eval.py"
spec = importlib.util.spec_from_file_location("legacy_mc_eval", LEGACY_FILE)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
legacy.LABELS = list("ABCDEF")
legacy.OUTPUT_PATTERN = re.compile(r"\s*([A-F](?:\s*,\s*[A-F])*)\s*\Z")
MODES = ("full_context", "no_context")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical_sha(value):
    return sha(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def offline():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"


def prepare():
    if (HERE / "REQUESTS.json").exists():
        raise RuntimeError("Preparation already frozen")
    offline()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(legacy.MODEL), trust_remote_code=True, local_files_only=True)
    manifest_path = INPUT / "shuffled_display_research/SHUFFLED_DISPLAY_MANIFEST.json"
    manifest = read(manifest_path)
    requests, items, bindings, permutations = [], [], [], {}
    for entry in manifest["rows"]:
        root_id = entry["root_id"]
        base = INPUT / "shuffled_display_research" / root_id
        packet_path = base / "DISPLAY_PACKET.json"
        mapping_path = base / "DISPLAY_MAPPING_PRIVATE.json"
        source_path = INPUT / "roots" / root_id / "FULL_SOURCE.json"
        candidate_path = INPUT / "roots" / root_id / "HISTORICAL_CANDIDATE.json"
        packet, mapping, source, candidate = map(read, (packet_path, mapping_path, source_path, candidate_path))
        assert canonical_sha(packet) == entry["packet_sha256"]
        assert packet["source"] == source
        assert canonical_sha(source) == packet["source_sha256"]
        assert sha(source_path.read_bytes()) == candidate["source_binding"]["sha256"]
        assert canonical_sha(candidate) == mapping["source_candidate_sha256"]
        assert mapping["display_to_canonical"] == entry["display_to_canonical"]
        assert set(mapping["canonical_gold"]) == {"option_" + label for label in candidate["gold"]}
        for label, option_id in mapping["display_to_canonical"].items():
            assert packet["options"][label] == candidate["options"][option_id.removeprefix("option_")]
        qid = packet["candidate_id"]
        item = {
            "candidate_id": qid, "root_id": root_id, "question": packet["question"],
            "question_type": packet["question_type"],
            "options": [{"id": label, "text": text} for label, text in packet["options"].items()],
            "label_to_option_id": mapping["display_to_canonical"],
            "gold_labels": mapping["display_gold"], "gold_option_ids": mapping["canonical_gold"],
            "diagnostic_only": True,
        }
        assert item["question_type"] == "multi_select" and list(packet["options"]) == legacy.LABELS
        turns = [{"turn_id": row["turn_ref"], "speaker": row["speaker"], "text": row["text"]} for row in source["turns"]]
        assert len({row["turn_id"] for row in turns}) == len(turns)
        legacy.QUESTION_ID = qid
        variants = legacy.make_variants(item)
        assert len({tuple(v["display_to_option_id"].values()) for v in variants}) == 6
        permutations[qid] = variants
        items.append(item)
        for path in (packet_path, mapping_path, source_path, candidate_path):
            bindings.append({"path": str(path), "sha256_bytes": sha(path.read_bytes())})
        for mode in MODES:
            for variant in variants:
                # Gold metadata is never passed to build_prompt or to the model.
                prompt = legacy.build_prompt(item["question"], variant["options"], turns if mode == "full_context" else [], tokenizer)
                assert qid not in prompt  # MC3 in private ID must not leak answer cardinality.
                tokens = len(tokenizer(prompt, add_special_tokens=False).input_ids)
                assert tokens + legacy.MAX_NEW_TOKENS <= legacy.MAX_MODEL_LEN, (qid, tokens)
                for repeat in range(3):
                    sample_index = int(variant["variant_id"].split("_")[1]) * 3 + repeat
                    rid = f"{qid}__{mode}__{variant['variant_id']}__s{repeat:02d}"
                    prompt_path = HERE / "prompts" / f"{rid}.txt"
                    prompt_path.parent.mkdir(parents=True, exist_ok=True)
                    with prompt_path.open("x", encoding="utf-8") as stream:
                        stream.write(prompt)
                    requests.append({
                        "request_id": rid, "candidate_id": qid, "root_id": root_id,
                        "context_mode": mode, "variant_id": variant["variant_id"], "repeat_index": repeat,
                        "sampling_seed": legacy.SAMPLING_SEED_START + sample_index,
                        "display_to_option_id": variant["display_to_option_id"], "options": variant["options"],
                        "gold_labels": variant["gold_labels"], "gold_option_ids": item["gold_option_ids"],
                        "prompt_path": str(prompt_path.relative_to(HERE)), "prompt_sha256": sha(prompt.encode()),
                        "prompt_token_count": tokens, "turn_count": len(turns) if mode == "full_context" else 0,
                        "source_sha256_bytes": sha(source_path.read_bytes()), "diagnostic_only": True,
                    })
    assert len(items) == 10 and len(requests) == 360
    write(HERE / "ITEMS_PRIVATE.json", items)
    write(HERE / "PERMUTATIONS_PRIVATE.json", permutations)
    write(HERE / "REQUESTS.json", requests)
    write(HERE / "INPUT_BINDINGS.json", {"files": bindings, "legacy_runner_sha256": sha(LEGACY_FILE.read_bytes()), "adapter_sha256": sha(Path(__file__).read_bytes())})
    write(HERE / "MODEL_CONFIG.json", {
        "model": str(legacy.MODEL), "tokenizer": str(legacy.MODEL), "base_model_only": True,
        "temperature": legacy.TEMPERATURE, "top_p": legacy.TOP_P, "max_new_tokens": legacy.MAX_NEW_TOKENS,
        "stop_sequences": legacy.STOP, "max_model_len": legacy.MAX_MODEL_LEN,
        "gpu_memory_utilization": legacy.GPU_MEMORY_UTILIZATION, "dtype": "bfloat16",
        "thinking": False, "offline": True, "automatic_retry": False, "training": False,
        "display_seed": legacy.DISPLAY_SEED, "sampling_seed_start": legacy.SAMPLING_SEED_START,
        "sampling_seed_rule": "same paired seed across context modes; 18 distinct seeds per candidate",
        "permutations": 6, "repeats_per_permutation": 3, "context_modes": MODES,
        "prompt": "unchanged legacy build_prompt; turn_ref adapted to turn_id; ID/Gold/author analysis excluded",
        "diagnostic_only": True, "formal_qa_eligible": False,
    })
    write(HERE / "PREFLIGHT.json", {
        "status": "PASS", "request_count": len(requests), "candidate_count": len(items),
        "min_prompt_tokens": min(r["prompt_token_count"] for r in requests),
        "max_prompt_tokens": max(r["prompt_token_count"] for r in requests),
        "truncation": False, "source_and_mapping_bindings_verified": True,
        "gold_or_candidate_id_in_solver_prompt": False,
        "request_sha256": sha((HERE / "REQUESTS.json").read_bytes()),
        "diagnostic_only": True, "formal_qa_eligible": False,
    })
    print(json.dumps(read(HERE / "PREFLIGHT.json")), flush=True)


def run():
    if (HERE / "STARTED.json").exists():
        raise RuntimeError("Inference already started; no automatic retries or overwrites")
    preflight = read(HERE / "PREFLIGHT.json")
    assert preflight["status"] == "PASS" and preflight["request_sha256"] == sha((HERE / "REQUESTS.json").read_bytes())
    for binding in read(HERE / "INPUT_BINDINGS.json")["files"]:
        assert sha(Path(binding["path"]).read_bytes()) == binding["sha256_bytes"]
    requests = read(HERE / "REQUESTS.json")
    prompts = [(HERE / r["prompt_path"]).read_text() for r in requests]
    assert all(sha(prompt.encode()) == r["prompt_sha256"] for r, prompt in zip(requests, prompts))
    offline()
    import vllm
    import torch
    import transformers
    write(HERE / "STARTED.json", {
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "planned_calls": len(requests),
        "vllm": vllm.__version__, "torch": torch.__version__, "transformers": transformers.__version__,
        "diagnostic_only": True, "gold_read_by_solver": False, "automatic_retry": False,
    })
    llm = vllm.LLM(model=str(legacy.MODEL), tokenizer=str(legacy.MODEL), trust_remote_code=True,
                   dtype="bfloat16", max_model_len=legacy.MAX_MODEL_LEN,
                   gpu_memory_utilization=legacy.GPU_MEMORY_UTILIZATION,
                   seed=legacy.DISPLAY_SEED, enable_prefix_caching=False)
    raws = []
    for start in range(0, len(requests), 6):
        batch = requests[start:start + 6]
        params = [vllm.SamplingParams(n=1, temperature=legacy.TEMPERATURE, top_p=legacy.TOP_P,
                  max_tokens=legacy.MAX_NEW_TOKENS, stop=legacy.STOP, seed=r["sampling_seed"]) for r in batch]
        try:
            outputs = llm.generate(prompts[start:start + 6], params, use_tqdm=False)
            assert len(outputs) == len(batch)
        except Exception as exc:
            write(HERE / "BACKEND_FAILURE.json", {"batch_start": start, "error": repr(exc), "automatic_retry": False})
            raise
        for row, output in zip(batch, outputs):
            completion = output.outputs[0] if output.outputs else None
            raw = {"request_id": row["request_id"], "status": "RETURNED",
                   "raw_prediction": completion.text if completion else "",
                   "finish_reason": completion.finish_reason if completion else None,
                   "stop_reason": completion.stop_reason if completion else None,
                   "completion_token_ids": list(completion.token_ids) if completion else [],
                   "prompt_sha256": row["prompt_sha256"], "diagnostic_only": True}
            write(HERE / "raw" / f"{row['request_id']}.json", raw)
            raws.append(raw)
        print(json.dumps({"status": "INFERENCE_PROGRESS", "completed": len(raws), "planned": len(requests)}), flush=True)
    write(HERE / "RAW_ANSWERS.json", raws)
    write(HERE / "COMPLETED.json", {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "returned_calls": len(raws), "diagnostic_only": True})
    score()


def summarize(rows):
    predictions = [tuple(sorted(r["predicted_option_ids"])) for r in rows if r["valid"]]
    return {
        "calls": len(rows), "valid_calls": sum(r["valid"] for r in rows),
        "exact_calls": sum(r["exact_correct"] for r in rows),
        "exact_accuracy": sum(r["exact_correct"] for r in rows) / len(rows),
        "mean_J": sum(r["partial_score"] for r in rows) / len(rows),
        "fp_total": sum(r["fp"] or 0 for r in rows), "fn_total": sum(r["fn"] or 0 for r in rows),
        "invalid_calls": sum(not r["valid"] for r in rows),
        "semantic_prediction_modes": len(set(predictions)),
        "modal_semantic_agreement": max(Counter(predictions).values()) / len(predictions) if predictions else None,
        "prediction_distribution": [{"option_ids": list(p), "count": n} for p, n in Counter(predictions).most_common()],
    }


def score():
    requests, raws = read(HERE / "REQUESTS.json"), read(HERE / "RAW_ANSWERS.json")
    assert len(requests) == len(raws) == 360
    rows = []
    for request, raw in zip(requests, raws):
        assert request["request_id"] == raw["request_id"]
        value = legacy.score_answer(raw["raw_prediction"], request, raw["finish_reason"])
        predicted = sorted(request["display_to_option_id"][label] for label in value["parsed"]) if value["valid"] else []
        if value["valid"]:
            g, n = len(request["gold_option_ids"]), len(request["options"])
            exact_J = Fraction(value["tp"], g) - Fraction(value["fp"], n-g) if n != g else Fraction(value["tp"], g)
            assert abs(float(exact_J) - value["partial_score"]) < 1e-12
            assert value["exact_correct"] == (set(predicted) == set(request["gold_option_ids"]))
            tags = (["extra_selection"] if value["fp"] else []) + (["missed_selection"] if value["fn"] else [])
        else:
            exact_J, tags = Fraction(0), ["format_or_finish_error"]
        rows.append({**{k: request[k] for k in ("request_id", "candidate_id", "root_id", "context_mode", "variant_id", "repeat_index", "sampling_seed", "display_to_option_id", "gold_labels", "gold_option_ids")},
                     **value, "raw_prediction": raw["raw_prediction"], "finish_reason": raw["finish_reason"],
                     "predicted_option_ids": predicted, "J_fraction": str(exact_J), "observed_error_types": tags})
    by_question = {}
    for item in read(HERE / "ITEMS_PRIVATE.json"):
        qid = item["candidate_id"]
        by_question[qid] = {}
        for mode in MODES:
            subset = [r for r in rows if r["candidate_id"] == qid and r["context_mode"] == mode]
            summary = summarize(subset)
            summary["by_permutation"] = {f"perm_{i:02d}": summarize([r for r in subset if r["variant_id"] == f"perm_{i:02d}"]) for i in range(6)}
            summary["option_counts"] = {oid: {"gold": oid in item["gold_option_ids"], "selected": sum(oid in r["predicted_option_ids"] for r in subset if r["valid"])} for oid in sorted(item["label_to_option_id"].values())}
            by_question[qid][mode] = summary
    scores = {"diagnostic_only": True, "formal_qa_eligible": False,
              "formula": "J=TP/g-FP/(n-g); invalid=0, fixed denominator; MC6 is a separate all-true control",
              "summaries": {mode: summarize([r for r in rows if r["context_mode"] == mode]) for mode in MODES},
              "by_question": by_question, "requests": rows}
    write(HERE / "SCORES.json", scores)
    print(json.dumps(scores["summaries"], ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run", "score"))
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "score": score}[args.mode]()


if __name__ == "__main__":
    main()
