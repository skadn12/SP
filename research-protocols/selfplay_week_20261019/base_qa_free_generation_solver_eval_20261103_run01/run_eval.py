#!/usr/bin/env python3
"""Evaluate three frozen free-authored drafts with existing local inference code."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent
PROJECT = WEEK.parent.parent
SOURCE = WEEK / "base_qa_free_generation_20261103_run01"
ENGINE_PATH = WEEK / "base_qa_v2_2_1_cardinality_solver_eval_20261103/run_cardinality_eval.py"
ORDER_PROMPT_PATH = WEEK / "base_qa_v2_2_1_format_restore_solver_eval_20261102/run_solver_eval.py"
SCORER_PATH = PROJECT / "offline-deliveries/h4_feedback_training_pool_v15_qa/evaluation/scoring.py"
PERMUTATION_PATH = SCORER_PATH.with_name("protocol.py")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


engine = module("existing_cardinality_inference", ENGINE_PATH)
order_prompt = module("existing_order_prompt", ORDER_PROMPT_PATH)
official = module("official_scoring", SCORER_PATH)
permutation_protocol = module("official_permutations", PERMUTATION_PATH)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(name, value):
    with (HERE / name).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def parse_and_score(raw, request):
    qtype = request["question_type"]
    labels = list(request["display_to_option_id"])
    delimiter = "," if qtype == "multi_select" else ">"
    alternative = "(?:" + "|".join(re.escape(label) for label in labels) + ")"
    pattern = re.compile(r"\s*" + alternative + r"(?:\s*" + re.escape(delimiter) + r"\s*" + alternative + r")*\s*\Z")
    text = raw.get("raw_prediction")
    valid = (raw.get("status") == "RETURNED" and isinstance(text, str)
             and raw.get("finish_reason") in ("stop", "eos")
             and bool(pattern.fullmatch(text)))
    parsed = [part.strip() for part in text.strip().split(delimiter)] if valid else None
    reason = None if valid else "not_complete_required_label_syntax_or_backend_error"
    if valid and len(parsed) != len(set(parsed)):
        valid, reason = False, "duplicate_label"
    if valid and qtype == "ordering" and set(parsed) != set(labels):
        valid, reason = False, "incomplete_order"
    if not valid:
        return {"valid": False, "parse_reason": reason, "parsed": parsed,
                "exact_correct": False, "J": 0.0, "J_fraction": "0",
                "predicted_option_ids": None, "tp": None, "fp": None, "fn": None,
                "recall": 0.0, "item_completeness": len(set(parsed or [])) / len(labels),
                "invalid_policy": "zero in fixed denominators; raw answer unchanged; no repair or retry"}
    field = "options" if qtype == "multi_select" else "items"
    question = {"question_type": qtype, "gold_content_ids": request["gold_option_ids"]}
    mapped_request = {"variant": {field: {option["id"]: option["text"] for option in request["options"]},
                                  "display_to_content": request["display_to_option_id"]}}
    # This is a frozen transport adapter after strict syntax validation, not
    # repair of a malformed model answer. The official scorer accepts a box.
    result = official.score_response(question, mapped_request, "<box>" + ",".join(parsed) + "</box>")
    assert result["scored"], result
    partial = Fraction(result["partial_score_fraction"])
    output = {"valid": True, "parse_reason": None, "parsed": parsed,
              "exact_correct": result["exact_match"], "J": float(partial), "J_fraction": str(partial),
              "predicted_option_ids": result["content_prediction"],
              "tp": result["tp"], "fp": result["fp"], "fn": result["fn"],
              "recall": result["tp"] / request["g"], "item_completeness": len(parsed) / len(labels),
              "official_scoring_status": result["scoring_status"]}
    if qtype == "ordering":
        output.update(correct_pairs=result["tp"], wrong_pairs=result["fp"],
                      total_pairs=request["g"], pair_accuracy=result["tp"] / request["g"],
                      pair_reversal_FN=result["fp"],
                      fn_note="official scorer uses fn=0 for complete order; reversed Gold pairs equal wrong_pairs")
    return output


def self_checks():
    multi = {"question_type": "multi_select", "display_to_option_id": dict(zip("ABCDEF", "abcdef")),
             "options": [{"id": key, "text": key} for key in "ABCDEF"], "gold_option_ids": ["a", "c"], "n": 6, "g": 2}
    raw = lambda text: {"status": "RETURNED", "finish_reason": "stop", "raw_prediction": text}
    assert parse_and_score(raw("A,C"), multi)["J_fraction"] == "1"
    assert parse_and_score(raw("A,B"), multi)["J_fraction"] == "1/4"
    assert not parse_and_score(raw("Answer: A,C"), multi)["valid"]
    assert not parse_and_score(raw("A,C,C"), multi)["valid"]
    assert not parse_and_score(raw("A,G"), multi)["valid"]
    ordering = {"question_type": "ordering", "display_to_option_id": {f"I{i}": str(i) for i in range(1, 7)},
                "options": [{"id": f"I{i}", "text": str(i)} for i in range(1, 7)],
                "gold_option_ids": list("123456"), "n": 30, "g": 15}
    assert parse_and_score(raw("I1>I2>I3>I4>I5>I6"), ordering)["J_fraction"] == "1"
    assert parse_and_score(raw("I6>I5>I4>I3>I2>I1"), ordering)["J_fraction"] == "-1"
    assert parse_and_score(raw("I2>I1>I3>I4>I5>I6"), ordering)["J_fraction"] == "13/15"
    assert not parse_and_score(raw("I1>I2>I3"), ordering)["valid"]
    assert not parse_and_score(raw("I1>I1>I3>I4>I5>I6"), ordering)["valid"]
    assert not parse_and_score(raw("I1,I2,I3,I4,I5,I6"), ordering)["valid"]


def prepare():
    engine.offline()
    self_checks()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(engine.MODEL), trust_remote_code=True, local_files_only=True)
    bindings = []
    # Bind every previous experiment artifact so no frozen input is overwritten.
    for path in sorted(SOURCE.iterdir()):
        if path.is_file():
            bindings.append({"path": str(path), "sha256_bytes": sha(path.read_bytes())})
    for path in (ENGINE_PATH, engine.LEGACY_FILE, ORDER_PROMPT_PATH, SCORER_PATH, PERMUTATION_PATH,
                 Path(__file__), engine.MODEL / "config.json", engine.MODEL / "tokenizer_config.json"):
        bindings.append({"path": str(path), "sha256_bytes": sha(path.read_bytes())})
    for name, digest in read(SOURCE / "MANIFEST_FINAL.json").items():
        assert sha((SOURCE / name).read_bytes()) == digest, name
    targets = read(SOURCE / "TARGETS_FROZEN.json")
    candidates = read(SOURCE / "CANDIDATES_FIRST_DRAFT.json")["candidates"]
    blind = read(SOURCE / "BLIND_COMPARISON.json")
    assert blind["answer_agreements"] == 3
    items, permutations, requests, prompt_bindings = [], {}, [], []
    for index, candidate in enumerate(candidates):
        qtype = "ordering" if candidate["question_type"] == "full_order" else "multi_select"
        field = "items" if qtype == "ordering" else "options"
        meta = targets["source_freeze"][candidate["root_id"]]
        source_file = Path(meta["path"])
        assert sha(source_file.read_bytes()) == candidate["source_sha256"] == meta["sha256"]
        source = read(source_file)
        assert len(source["turns"]) == meta["turn_count"]
        turns = [{"turn_id": row["turn_ref"], "speaker": row["speaker"], "text": row["text"]} for row in source["turns"]]
        labels = list(candidate[field])
        mapping = {label: candidate[field][label]["option_id"] for label in labels}
        canonical_gold = [mapping[label] for label in candidate["gold"]]
        target = next(row for row in targets["targets"] if row["candidate_id"] == candidate["candidate_id"])
        if qtype == "ordering":
            assert canonical_gold == target["gold_semantic_order"]
        else:
            assert set(canonical_gold) == set(target["gold_semantic_ids"])
        item = {"candidate_id": candidate["candidate_id"], "root_id": candidate["root_id"],
                "question_type": qtype, "legacy_type": qtype, "question": candidate["question"],
                "options": [{"id": label, "text": candidate[field][label]["text"]} for label in labels],
                "label_to_option_id": mapping, "gold_labels": candidate["gold"],
                "gold_option_ids": canonical_gold, "source_path": str(source_file),
                "source_sha256": meta["sha256"], "turn_count": len(turns),
                "diagnostic_only": True, "formal_qa_eligible": False}
        variant_count, repeats = (6, 3) if qtype == "multi_select" else (12, 1)
        item.update(permutations=variant_count, repeats_per_permutation=repeats)
        public_question = {"candidate_id": item["candidate_id"], "question_type": qtype,
                           field: {label: candidate[field][label]["text"] for label in labels}}
        public_variants = permutation_protocol.make_variants(public_question, seed=2026110351, count=variant_count)
        text_to_option_id = {candidate[field][label]["text"]: mapping[label] for label in labels}
        assert len(text_to_option_id) == len(labels)
        variants = []
        for public_variant in public_variants:
            rebound = {label: text_to_option_id[public_variant["content_to_text"][content_id]]
                       for label, content_id in public_variant["display_to_content"].items()}
            reverse = {option_id: label for label, option_id in rebound.items()}
            gold_labels = [reverse[oid] for oid in canonical_gold]
            variants.append({"variant_id": public_variant["variant_id"], "display_to_option_id": rebound,
                             "options": [{"id": label, "option_id": rebound[label], "text": text}
                                         for label, text in public_variant[field].items()],
                             "gold_labels": sorted(gold_labels) if qtype == "multi_select" else gold_labels})
        assert len({tuple(v["display_to_option_id"].values()) for v in variants}) == variant_count
        permutations[item["candidate_id"]] = variants
        items.append(item)
        bindings.append({"path": str(source_file), "sha256_bytes": meta["sha256"]})
        for mode in ("full_context", "no_context"):
            public_turns = turns if mode == "full_context" else []
            for variant_index, variant in enumerate(variants):
                if qtype == "multi_select":
                    prompt = engine.legacy.build_prompt(item["question"], variant["options"], public_turns, tokenizer)
                else:
                    prompt = order_prompt.build_prompt({"question_type": "ordering", "legacy_type": "ordering",
                                                       "question": item["question"], "options": variant["options"]}, public_turns, tokenizer)
                token_count = len(tokenizer(prompt, add_special_tokens=False).input_ids)
                assert token_count + engine.MAX_NEW_TOKENS <= engine.MAX_MODEL_LEN
                for repeat in range(repeats):
                    request_id = f"{item['candidate_id']}__{mode}__{variant['variant_id']}__s{repeat:02d}"
                    relative = Path("prompts") / f"{request_id}.txt"
                    (HERE / relative).parent.mkdir(exist_ok=True)
                    with (HERE / relative).open("x", encoding="utf-8") as stream:
                        stream.write(prompt)
                    prompt_bindings.append({"path": str(HERE / relative), "sha256": sha(prompt.encode())})
                    n = len(labels) if qtype == "multi_select" else len(labels) * (len(labels) - 1)
                    g = len(canonical_gold) if qtype == "multi_select" else n // 2
                    requests.append({"request_id": request_id, "candidate_id": item["candidate_id"],
                                     "root_id": item["root_id"], "question_type": qtype, "context_mode": mode,
                                     "variant_id": variant["variant_id"], "repeat_index": repeat,
                                     "sampling_seed": 2026110352 + index * 18 + variant_index * 3 + repeat,
                                     "display_to_option_id": variant["display_to_option_id"], "options": variant["options"],
                                     "gold_labels": variant["gold_labels"], "gold_option_ids": canonical_gold,
                                     "n": n, "g": g, "all_true_control": False,
                                     "prompt_path": str(relative), "prompt_sha256": sha(prompt.encode()),
                                     "prompt_token_count": token_count, "turn_count": len(public_turns),
                                     "source_sha256_bytes": meta["sha256"], "diagnostic_only": True})
    assert len(requests) == 96
    write("ITEMS_PRIVATE.json", items)
    write("PERMUTATIONS_PRIVATE.json", permutations)
    write("REQUESTS.json", requests)
    write("INPUT_BINDINGS.json", {"files": bindings, "prompts": prompt_bindings})
    write("MODEL_CONFIG.json", {"model": str(engine.MODEL), "base_model_only": True,
                               "temperature": engine.TEMPERATURE, "top_p": engine.TOP_P,
                               "max_new_tokens": engine.MAX_NEW_TOKENS, "max_model_len": engine.MAX_MODEL_LEN,
                               "gpu_memory_utilization": engine.GPU_MEMORY_UTILIZATION, "dtype": "bfloat16",
                               "stop_sequences": engine.STOP, "thinking": False, "offline": True,
                               "automatic_retry": False, "training": False,
                               "sampling_by_type": {"multi_select": {"permutations": 6, "repeats": 3},
                                                    "ordering": {"permutations": 12, "repeats": 1}},
                               "permutation_generator": str(PERMUTATION_PATH), "display_seed": 2026110351,
                               "planned_requests": len(requests),
                               "paired_sampling_seeds": True, "context_modes": ["full_context", "no_context"],
                               "prompt_builders": [str(engine.LEGACY_FILE), str(ORDER_PROMPT_PATH)],
                               "prompt_protocol_note": "Recent diagnostic bare-label prompt format retained; not identical to historical box-output prompt. Scoring and sampling follow historical standard.",
                               "question_language": "zh-CN", "source_language": "English",
                               "independent_questions": "one question only per prompt; no other candidate or answer is supplied",
                               "source_scope": "full frozen dialogue, no turn filtering or truncation",
                               "multi_select_scoring": "J=TP/g-FP/(n-g)",
                               "ordering_scoring": "directed pair J=(correct_pairs-wrong_pairs)/15; 30 possible directions,15 Gold",
                               "parser": "strict bare comma labels for multi-select; strict complete > chain for full order; validates before adapting to official scorer transport",
                               "invalid_policy": "zero scores in fixed denominators; preserve raw output without repair or retry",
                               "formal_qa_eligible": False, "diagnostic_only": True})
    write("PREFLIGHT.json", {"status": "PASS", "request_count": len(requests), "candidate_count": len(items),
                            "requests_sha256": sha((HERE / "REQUESTS.json").read_bytes()),
                            "self_checks": "11 scoring/parser boundary checks passed",
                            "min_prompt_tokens": min(row["prompt_token_count"] for row in requests),
                            "max_prompt_tokens": max(row["prompt_token_count"] for row in requests),
                            "truncation": False, "source_turn_counts": {i["candidate_id"]: i["turn_count"] for i in items},
                            "prior_artifacts_bound_and_unchanged": True, "no_cross_question_prompt_leakage": True})
    print(json.dumps(read(HERE / "PREFLIGHT.json"), ensure_ascii=False), flush=True)


def summarize(rows):
    valid = [row for row in rows if row["valid"]]
    answers = Counter(tuple(sorted(row["predicted_option_ids"]) if row["question_type"] == "multi_select"
                            else row["predicted_option_ids"]) for row in valid)
    exact = sum(row["exact_correct"] for row in rows)
    j = sum((Fraction(row["J_fraction"]) for row in rows), Fraction(0)) / len(rows)
    return {"planned_calls": len(rows), "valid_calls": len(valid), "invalid_calls": len(rows) - len(valid),
            "exact_correct_calls": exact, "exact_accuracy": exact / len(rows),
            "mean_J_fraction": str(j), "mean_J": float(j),
            "known_FP": sum(row["fp"] or 0 for row in valid), "known_FN": sum(row["fn"] or 0 for row in valid),
            "distinct_semantic_predictions": len(answers),
            "modal_valid_prediction_fraction": max(answers.values()) / len(valid) if valid else None,
            "mean_recall_or_pair_accuracy": sum(row["recall"] for row in rows) / len(rows)}


def score():
    requests, raw_rows = read(HERE / "REQUESTS.json"), read(HERE / "RAW_ANSWERS.json")
    assert len(requests) == len(raw_rows) == 96
    scored = []
    for request, raw in zip(requests, raw_rows):
        assert request["request_id"] == raw["request_id"]
        scored.append({**{key: request[key] for key in ("request_id", "candidate_id", "question_type", "context_mode", "variant_id", "repeat_index", "sampling_seed", "n", "g")},
                       **parse_and_score(raw, request), "raw_prediction": raw.get("raw_prediction"), "finish_reason": raw.get("finish_reason")})
    item_summaries = {}
    for item in read(HERE / "ITEMS_PRIVATE.json"):
        by_mode = {}
        for mode in ("full_context", "no_context"):
            rows = [row for row in scored if row["candidate_id"] == item["candidate_id"] and row["context_mode"] == mode]
            by_mode[mode] = summarize(rows)
            by_mode[mode]["by_permutation"] = {f"perm_{index:02d}": summarize([row for row in rows if row["variant_id"] == f"perm_{index:02d}"]) for index in range(item["permutations"])}
        item_summaries[item["candidate_id"]] = by_mode
    write("SCORES.json", {"diagnostic_only": True, "formal_qa_eligible": False,
                          "scorer_path": str(SCORER_PATH), "items": item_summaries, "requests": scored})
    print(json.dumps(item_summaries, ensure_ascii=False, indent=2), flush=True)


def run():
    self_checks()
    engine.HERE = HERE
    engine.score = score
    engine.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run", "score"))
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "score": score}[args.mode]()
