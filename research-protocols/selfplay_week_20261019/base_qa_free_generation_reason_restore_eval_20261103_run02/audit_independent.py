#!/usr/bin/env python3
"""Independent, read-only-input audit of the paired visible-reason rerun.

No experiment runner, model, baseline parser or official scorer is imported.
Use --preflight before inference. The final audit writes a new result once.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "base_qa_free_generation_solver_eval_20261103_run02"
MODES = ("full_context", "no_context")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def calculate(labels, request):
    """Calculate semantic scores directly from sets and ordered pairs."""
    invalid = {"valid": False, "exact_correct": False, "J_fraction": "0", "J": 0.0,
               "predicted_option_ids": None, "tp": None, "fp": None, "fn": None,
               "recall": 0.0}
    mapping = request["display_to_option_id"]
    if (not labels or len(labels) != len(set(labels)) or
            any(label not in mapping for label in labels)):
        return invalid
    if request["question_type"] == "ordering" and set(labels) != set(mapping):
        return invalid
    prediction = [mapping[label] for label in labels]
    gold = request["gold_option_ids"]
    if request["question_type"] == "ordering":
        gp = {(gold[i], gold[j]) for i in range(len(gold)) for j in range(i+1, len(gold))}
        pp = {(prediction[i], prediction[j]) for i in range(len(prediction))
              for j in range(i+1, len(prediction))}
        tp, fp, fn = len(gp & pp), len(pp - gp), 0
        partial = Fraction(tp - fp, math.comb(len(gold), 2))
        exact, recall = prediction == gold, tp / math.comb(len(gold), 2)
    else:
        gs, ps = set(gold), set(prediction)
        tp, fp, fn = len(gs & ps), len(ps - gs), len(gs - ps)
        partial = Fraction(tp, len(gs)) - Fraction(fp, len(mapping) - len(gs))
        exact, recall, prediction = gs == ps, tp / len(gs), sorted(ps)
    return {"valid": True, "exact_correct": exact, "J_fraction": str(partial),
            "J": float(partial), "predicted_option_ids": prediction,
            "tp": tp, "fp": fp, "fn": fn, "recall": recall}


def assess(raw, request, source_refs):
    """Separate designated final-answer validity from the reason/citation protocol."""
    zero = calculate(None, request)
    result = {"protocol_valid": False, "final_answer_valid": False,
              "answer_only": zero, "strict": zero, "final_answer": None}
    if (raw.get("status") != "RETURNED" or raw.get("finish_reason") not in ("stop", "eos")
            or not isinstance(raw.get("raw_prediction"), str)):
        return result
    try:
        obj = json.loads(raw["raw_prediction"], object_pairs_hook=unique_keys)
    except (ValueError, TypeError):
        return result
    if not isinstance(obj, dict):
        return result
    final = obj.get("final_answer")
    result["final_answer"] = final
    if not isinstance(final, str) or not final.startswith("<box>") or not final.endswith("</box>"):
        return result
    body = final[5:-6]
    labels = body.split(" ")
    if not body or any(not token for token in labels):
        return result
    answer = calculate(labels, request)
    result.update(final_answer_valid=answer["valid"], answer_only=answer)
    if not answer["valid"]:
        return result
    if list(obj) != ["item_evidence", "answer_reason", "final_answer"]:
        return result
    reason = obj["answer_reason"]
    evidence = obj["item_evidence"]
    if not isinstance(reason, str) or not 1 <= len(reason) <= 800 or not reason.strip():
        return result
    if not isinstance(evidence, dict) or set(evidence) != set(request["display_to_option_id"]):
        return result
    for ev in evidence.values():
        if not isinstance(ev, dict) or set(ev) != {"judgment", "turn_refs", "reason"}:
            return result
        if ev["judgment"] not in ("supported", "contradicted", "not_supported", "ambiguous"):
            return result
        explanation, refs = ev["reason"], ev["turn_refs"]
        if (not isinstance(explanation, str) or not 1 <= len(explanation) <= 320 or
                not explanation.strip()):
            return result
        if (not isinstance(refs, list) or len(refs) > 3 or
                any(not isinstance(ref, str) or ref not in source_refs for ref in refs)
                or len(refs) != len(set(refs))):
            return result
        if request["context_mode"] == "no_context" and refs:
            return result
        if request["context_mode"] == "full_context":
            if ev["judgment"] == "supported" and not refs:
                return result
            if request["question_type"] == "ordering" and not refs:
                return result
    result.update(protocol_valid=True, strict=answer)
    return result


def summarize(rows):
    count = len(rows)
    partial = sum((Fraction(row["J_fraction"]) for row in rows), Fraction()) / count
    valid = [row for row in rows if row["valid"]]
    predictions = Counter(tuple(row["predicted_option_ids"]) for row in valid)
    return {"planned_calls": count, "valid_calls": len(valid),
            "invalid_calls": count-len(valid),
            "exact_correct_calls": sum(row["exact_correct"] for row in rows),
            "exact_accuracy": sum(row["exact_correct"] for row in rows) / count,
            "mean_J_fraction": str(partial), "mean_J": float(partial),
            "known_FP": sum(row["fp"] or 0 for row in valid),
            "known_FN": sum(row["fn"] or 0 for row in valid),
            "distinct_semantic_predictions": len(predictions),
            "modal_valid_prediction_fraction": max(predictions.values()) / len(valid) if valid else None,
            "mean_recall_or_pair_accuracy": sum(row["recall"] for row in rows) / count}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    checks, errors = Counter(), []

    def check(value, name, detail=None):
        checks[name] += 1
        if not value:
            errors.append({"check": name, "detail": detail})

    requests, old = read(HERE / "REQUESTS.json"), read(BASE / "REQUESTS.json")
    items, old_items = read(HERE / "ITEMS_PRIVATE.json"), read(BASE / "ITEMS_PRIVATE.json")
    config, preflight = read(HERE / "MODEL_CONFIG.json"), read(HERE / "PREFLIGHT.json")
    check(len(requests) == len(old) == 396, "396_requests")
    check(items == old_items and len(items) == 12, "frozen_twelve_questions")
    check(read(HERE / "PERMUTATIONS_PRIVATE.json") == read(BASE / "PERMUTATIONS_PRIVATE.json"),
          "frozen_permutations")
    check(Counter(r["context_mode"] for r in requests) == {"full_context": 198, "no_context": 198},
          "paired_context_counts")
    check(config["model"] == read(BASE / "MODEL_CONFIG.json")["model"] == "/root/autodl-tmp/Qwen3-4B",
          "unchanged_model_checkpoint_path")
    check(config["temperature"] == .8 and config["top_p"] == .9 and config["top_k"] == -1
          and config["repetition_penalty"] == 1.0 and config["engine_seed"] == 2026110303,
          "unchanged_sampling")
    check(config["native_thinking"] is False and config["visible_reasons"] is True
          and config["structured_outputs"] is True and config["max_new_tokens"] == 6144
          and config["max_model_len"] == 16384 and config["stop_sequences"] is None,
          "explicit_joint_protocol_change")
    check(digest(HERE / "REQUESTS.json") == preflight["requests_sha256"], "preflight_request_hash")
    for relative, expected in read(BASE / "MANIFEST_FINAL.json")["files"].items():
        check(digest(BASE / relative) == expected, "unchanged_baseline_manifest", relative)
    for binding in read(HERE / "INPUT_BINDINGS.json")["files"]:
        check(digest(binding["path"]) == binding["sha256"], "bound_input_hash", binding["path"])
    changed_fields = {"prompt_path", "schema_path", "prompt_sha256", "prompt_token_count",
                      "schema_sha256", "baseline_prompt_sha256"}
    by_item = {item["candidate_id"]: item for item in items}
    source_cache, ref_cache = {}, {}
    for item in items:
        check(digest(item["source_path"]) == item["source_sha256"], "frozen_full_source", item["candidate_id"])
        source_cache[item["candidate_id"]] = read(item["source_path"])
    old_by_id = {row["request_id"]: row for row in old}
    check(len({r["request_id"] for r in requests}) == len(requests) and
          set(old_by_id) == {r["request_id"] for r in requests}, "identical_unique_request_ids")
    for row in requests:
        rid = row["request_id"]
        prior = old_by_id[rid]
        check({k: v for k, v in row.items() if k not in changed_fields} ==
              {k: v for k, v in prior.items() if k not in changed_fields},
              "paired_question_gold_mapping_seed", rid)
        check(row["baseline_prompt_sha256"] == prior["prompt_sha256"], "baseline_prompt_binding", rid)
        check(digest(HERE / row["prompt_path"]) == row["prompt_sha256"], "prompt_hash", rid)
        check(digest(HERE / row["schema_path"]) == row["schema_sha256"], "schema_hash", rid)
        check(row["prompt_token_count"] + config["max_new_tokens"] <= config["max_model_len"],
              "untruncated_context_budget", rid)
        item = by_item[row["candidate_id"]]
        turns = source_cache[row["candidate_id"]]["turns"] if row["context_mode"] == "full_context" else []
        refs = [turn["turn_ref"] for turn in turns]
        ref_cache[rid] = refs
        check(len(refs) == len(set(refs)), "unique_turn_references", rid)
        prompt = (HERE / row["prompt_path"]).read_text(encoding="utf-8")
        check(row["turn_count"] == len(turns), "unchanged_source_turn_count", rid)
        marker = "<|im_start|>user\n"
        check(prompt.count(marker) == 1, "single_public_user_message", rid)
        user = prompt.split(marker, 1)[-1].split("<|im_end|>", 1)[0]
        dialogue = "\n".join(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" for t in turns)
        prefix = ("Dialogue:\n" + dialogue + "\n\n") if turns else ""
        public = (prefix + f"Question ({item['question_type']}):\n{item['question']}\n\nDisplayed items:\n" +
                  "\n".join(f"{o['id']}. {o['text']}" for o in row["options"]) + "\n\n")
        check(user.startswith(public), "exact_public_dialogue_question_and_options", rid)
        tail = user[len(public):]
        check(tail.startswith("For each displayed label,") if turns else
              tail.startswith("No source dialogue or source references are supplied"), "neutral_instruction_prefix", rid)
        check("gold_option_ids" not in tail and "display_to_option_id" not in tail and
              "target_path" not in tail and "Requested operation:" not in tail,
              "no_private_metadata_or_added_operation_hint", rid)
        schema = read(HERE / row["schema_path"])
        check(schema["type"] == "object" and schema["additionalProperties"] is False and
              set(schema["required"]) == {"item_evidence", "answer_reason", "final_answer"} and
              set(schema["properties"]) == {"item_evidence", "answer_reason", "final_answer"},
              "schema_top_level", rid)
        evidence = schema["properties"]["item_evidence"]
        check(set(evidence["required"]) == set(row["display_to_option_id"]) == set(evidence["properties"])
              and evidence["additionalProperties"] is False, "schema_all_labels_no_gold_selection", rid)
        pattern = schema["properties"]["final_answer"]["pattern"]
        labels = list(row["display_to_option_id"])
        check(all(re.fullmatch(pattern, "<box>" + " ".join(seq) + "</box>")
                  for seq in ([label] for label in labels)) and
              re.fullmatch(pattern, "<box>" + " ".join(labels) + "</box>") is not None and
              re.fullmatch(pattern, "<box>" + " ".join(labels[::-1]) + "</box>") is not None and
              re.fullmatch(pattern, "<box>Z99</box>") is None,
              "schema_generic_cardinality_and_order", rid)
        for ev in evidence["properties"].values():
            citation = ev["properties"]["turn_refs"]
            check(set(ev["required"]) == {"judgment", "turn_refs", "reason"} and
                  ev["additionalProperties"] is False, "schema_evidence_fields", rid)
            if refs:
                check(citation["items"]["enum"] == refs and citation["maxItems"] == 3,
                      "schema_all_public_turn_refs", rid)
            else:
                check(citation["maxItems"] == 0, "schema_no_context_empty_refs", rid)
    preliminary = {"status": "PASS" if not errors else "FAIL", "checks": dict(checks), "errors": errors}
    if args.preflight:
        print(json.dumps(preliminary, ensure_ascii=False, indent=2))
        return

    runtime, old_runtime = read(HERE / "STARTED.json"), read(BASE / "STARTED.json")
    check(runtime["vllm_version"] == old_runtime["vllm_version"] == "0.21.0",
          "actual_same_vllm_0_21_0")
    check(runtime["planned_requests"] == 396 and runtime["automatic_retry"] is False,
          "actual_run_no_retry")
    runtime_note = {"baseline_vllm_version": old_runtime["vllm_version"],
                    "rerun_vllm_version": runtime["vllm_version"],
                    "same_runtime": runtime["vllm_version"] == old_runtime["vllm_version"],
                    "correction": "The prepared MODEL_CONFIG historical_differences runtime statement is inaccurate; actual STARTED records both show vLLM 0.21.0."}
    raw_rows, scores = read(HERE / "RAW_ANSWERS.json"), read(HERE / "SCORES.json")
    raw_by_id = {r["request_id"]: r for r in raw_rows}
    scored_by_id = {r["request_id"]: r for r in scores["requests"]}
    check(len(raw_rows) == len(scores["requests"]) == len(raw_by_id) == len(scored_by_id) == 396,
          "396_unique_returned_and_scored_requests")
    check(set(raw_by_id) == set(scored_by_id) == set(old_by_id), "complete_matching_request_sets")
    derived = []
    for row in requests:
        rid = row["request_id"]
        raw, observed = raw_by_id[rid], scored_by_id[rid]
        check(digest(HERE / "raw" / (rid + ".json")) ==
              hashlib.sha256((json.dumps(raw, ensure_ascii=False, indent=2) + "\n").encode()).hexdigest(),
              "preserved_per_call_raw", rid)
        check(raw["prompt_sha256"] == row["prompt_sha256"] and raw["automatic_retry"] is False,
              "raw_prompt_binding_and_no_retry", rid)
        independent = assess(raw, row, ref_cache[rid])
        for field in ("protocol_valid", "final_answer_valid", "final_answer"):
            check(observed.get(field) == independent[field], "independent_" + field, rid)
        for channel, actual in (("strict", observed), ("answer_only", observed.get("answer_only_score", {}))):
            expected = independent[channel]
            for field in ("valid", "exact_correct", "J_fraction", "predicted_option_ids", "tp", "fp", "fn"):
                if channel == "answer_only" and not actual and not expected["valid"]:
                    continue
                check(actual.get(field) == expected[field], "independent_" + channel + "_" + field,
                      {"request_id": rid, "expected": expected[field], "actual": actual.get(field)})
            if actual:
                check(abs(actual["J"] - expected["J"]) < 1e-12 and
                      abs(actual["recall"] - expected["recall"]) < 1e-12,
                      "independent_" + channel + "_numeric", rid)
        derived.append({**{k: row[k] for k in ("request_id", "candidate_id", "context_mode", "variant_id")},
                        **independent})
    item_summaries, comparisons = {}, {}
    baseline_scores = {r["request_id"]: r for r in read(BASE / "SCORES.json")["requests"]}
    for cid in by_item:
        item_summaries[cid] = {}
        for mode in MODES:
            chosen = [r for r in derived if r["candidate_id"] == cid and r["context_mode"] == mode]
            strict, answer = summarize([r["strict"] for r in chosen]), summarize([r["answer_only"] for r in chosen])
            item_summaries[cid][mode] = {"strict": strict, "final_answer_only": answer}
            reported = scores["items"][cid][mode]
            for field, expected in strict.items():
                actual = reported[field]
                check(abs(actual - expected) < 1e-12 if isinstance(expected, float) else actual == expected,
                      "independent_item_strict_" + field, {"candidate_id": cid, "mode": mode})
    for mode in MODES:
        chosen = [r for r in derived if r["context_mode"] == mode]
        comparisons[mode] = {"baseline": summarize([baseline_scores[r["request_id"]] for r in chosen])}
        for channel in ("strict", "answer_only"):
            current = summarize([r[channel] for r in chosen])
            transitions = Counter()
            for row in chosen:
                before, after = baseline_scores[row["request_id"]], row[channel]
                transitions[("correct" if before["exact_correct"] else "wrong") + "_to_" +
                            ("correct" if after["exact_correct"] else "wrong")] += 1
            current["paired_exact_transitions"] = dict(transitions)
            current["exact_accuracy_delta"] = current["exact_accuracy"] - comparisons[mode]["baseline"]["exact_accuracy"]
            current["mean_J_delta_fraction"] = str(Fraction(current["mean_J_fraction"]) -
                  Fraction(comparisons[mode]["baseline"]["mean_J_fraction"]))
            comparisons[mode][channel] = current
    destination = HERE / "INDEPENDENT_REASON_AUDIT.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump({"created_at_utc": datetime.now(timezone.utc).isoformat(),
                   "status": "PASS" if not errors else "FAIL",
                   "method": "Independent JSON parsing, manually checked public schema, semantic set and directed-pair Fraction scoring; no runner/scorer/model imports",
                   "checks": dict(checks), "errors": errors, "comparisons": comparisons,
                   "runtime_verification": runtime_note,
                   "items": item_summaries,
                   "bindings": {name: digest(HERE / name) for name in
                                ("REQUESTS.json", "RAW_ANSWERS.json", "SCORES.json", "MODEL_CONFIG.json")}},
                  stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "PASS" if not errors else "FAIL", "checks": sum(checks.values()),
                      "errors": errors, "comparisons": comparisons}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
