#!/usr/bin/env python3
"""Independently audit frozen run02 calls using only Python's standard library.

This file neither imports the experiment runner/scorer nor calls a model. Run
only after REQUESTS, RAW_ANSWERS and SCORES have been frozen. A pre-existing
audit output is never overwritten.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "base_qa_free_generation_20261103_run02"
MODES = ("full_context", "no_context")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    destination = HERE / "INDEPENDENT_SCORE_AUDIT.json"
    if destination.exists():
        raise RuntimeError("Independent score audit already frozen; refusing overwrite")
    required = ("REQUESTS.json", "RAW_ANSWERS.json", "SCORES.json", "ITEMS_PRIVATE.json",
                "PERMUTATIONS_PRIVATE.json", "INPUT_BINDINGS.json", "MODEL_CONFIG.json")
    for name in required:
        if not (HERE / name).is_file():
            raise RuntimeError(f"Results are not ready: missing {name}")

    requests = read(HERE / "REQUESTS.json")
    raw_rows = read(HERE / "RAW_ANSWERS.json")
    scores = read(HERE / "SCORES.json")
    items = read(HERE / "ITEMS_PRIVATE.json")
    permutations = read(HERE / "PERMUTATIONS_PRIVATE.json")
    config = read(HERE / "MODEL_CONFIG.json")
    errors, checks = [], Counter()

    def check(value, name, detail=None):
        checks[name] += 1
        if not value:
            errors.append({"check": name, "detail": detail})

    def check_hash(path, expected, name):
        path = Path(path)
        check(path.is_file() and digest(path) == expected, name, str(path))

    check(len(requests) == len(raw_rows) == len(scores["requests"]) == 396,
          "fixed_request_count_396")
    check(len(items) == 12, "twelve_candidates")
    check(Counter(item["question_type"] for item in items) ==
          {"multi_select": 10, "ordering": 2}, "ten_multiselect_two_ordering")
    check(Counter(row["context_mode"] for row in requests) ==
          {"full_context": 198, "no_context": 198}, "fixed_context_call_counts")
    check(len({row["request_id"] for row in requests}) == len(requests), "unique_request_ids")
    request_by_id = {row["request_id"]: row for row in requests}
    raw_by_id = {row["request_id"]: row for row in raw_rows}
    score_by_id = {row["request_id"]: row for row in scores["requests"]}
    check(len(raw_by_id) == len(raw_rows), "unique_raw_request_ids")
    check(len(score_by_id) == len(scores["requests"]), "unique_scored_request_ids")
    check(set(request_by_id) == set(raw_by_id) == set(score_by_id), "complete_matching_call_sets")
    item_by_id = {row["candidate_id"]: row for row in items}
    check(len(item_by_id) == 12, "unique_candidate_ids")
    check(set(item_by_id) == set(permutations) == set(scores["items"]), "matching_item_sets")

    # Verify every frozen source experiment entry, not just a filename pattern.
    manifest = read(SOURCE / "MANIFEST_FINAL.json")
    check(isinstance(manifest.get("files"), dict) and bool(manifest["files"]), "source_manifest_shape")
    for relative, expected in manifest["files"].items():
        check_hash(SOURCE / relative, expected, "source_manifest_hash")
    index = read(SOURCE / "QUESTION_INDEX.json")
    check({row["candidate_id"] for row in index} == set(item_by_id), "frozen_question_index_binding")
    source_cache = {}
    for entry in index:
        item = item_by_id[entry["candidate_id"]]
        folder = SOURCE / entry["item_path"]
        display, packet = read(folder / "DISPLAY_PRIVATE.json"), read(folder / "PUBLIC_PACKET.json")
        qtype = "ordering" if display["question_type"] == "full_order" else display["question_type"]
        check(item["question_type"] == qtype, "frozen_question_type", entry["candidate_id"])
        check(item["question"] == display["question"] == packet["question"],
              "frozen_stem_binding", entry["candidate_id"])
        check(item["root_id"] == display["root_id"] == packet["source"]["root_id"],
              "frozen_root_binding", entry["candidate_id"])
        displayed_text = {row["label"]: row["text"] for row in display["options"]}
        canonical_text = {row["id"]: row["text"] for row in item["options"]}
        check(canonical_text == displayed_text, "frozen_option_text_binding", entry["candidate_id"])
        check(item["label_to_option_id"] == display["label_to_option_id"],
              "frozen_option_identity_binding", entry["candidate_id"])
        same_gold = (item["gold_option_ids"] == display["semantic_gold"] if qtype == "ordering"
                     else set(item["gold_option_ids"]) == set(display["semantic_gold"]))
        check(same_gold, "frozen_semantic_gold_binding", entry["candidate_id"])
        check(Path(item["source_path"]) == Path(packet["source"]["path"]),
              "frozen_source_path_binding", entry["candidate_id"])
        check_hash(item["source_path"], packet["source"]["sha256"], "full_source_file_hash")
        source_cache[item["candidate_id"]] = read(item["source_path"])

    bindings = read(HERE / "INPUT_BINDINGS.json")
    for binding in bindings["files"]:
        check_hash(binding["path"], binding["sha256_bytes"], "input_file_hash")
    for binding in bindings["prompts"]:
        check_hash(binding["path"], binding["sha256"], "prompt_binding_hash")
    check(config["max_new_tokens"] == 32 and config["max_model_len"] == 16384,
          "unchanged_context_and_output_limits")
    check(config["temperature"] == 0.8 and config["top_p"] == 0.9,
          "unchanged_sampling_configuration")

    independent = []
    for rid, request in request_by_id.items():
        if rid not in raw_by_id or rid not in score_by_id:
            continue
        raw, observed = raw_by_id[rid], score_by_id[rid]
        item = item_by_id[request["candidate_id"]]
        mapping = request["display_to_option_id"]
        labels, ids = set(mapping), set(mapping.values())
        canonical_text = {item["label_to_option_id"][row["id"]]: row["text"] for row in item["options"]}
        check(len(mapping) == len(ids) == len(canonical_text), "mapping_bijection", rid)
        check(ids == set(canonical_text), "mapping_canonical_ids", rid)
        check(len(request["options"]) == len(mapping) and
              {row["id"] for row in request["options"]} == labels and
              all(mapping[row["id"]] in canonical_text and
                  canonical_text[mapping[row["id"]]] == row["text"] for row in request["options"]),
              "option_text_binding", rid)
        check(request["question_type"] == item["question_type"], "request_type_binding", rid)
        check(request["gold_option_ids"] == item["gold_option_ids"], "canonical_gold_binding", rid)
        gold = request["gold_option_ids"]
        k = len(labels)
        if request["question_type"] == "ordering":
            check([mapping[label] for label in request["gold_labels"]] == gold,
                  "display_gold_order_mapping", rid)
            check((request["n"], request["g"]) == (k * (k - 1), math.comb(k, 2)),
                  "dynamic_ordering_directed_pair_units", rid)
        else:
            check({mapping[label] for label in request["gold_labels"]} == set(gold),
                  "display_gold_set_mapping", rid)
            check(request["n"] == k and request["g"] == len(set(gold)) and 0 < request["g"] < k,
                  "multi_choice_units", rid)
        check_hash(HERE / request["prompt_path"], request["prompt_sha256"], "request_prompt_hash")
        prompt = (HERE / request["prompt_path"]).read_text(encoding="utf-8")
        source = source_cache[request["candidate_id"]]
        expected_turns = source["turns"] if request["context_mode"] == "full_context" else []
        check(request["turn_count"] == len(expected_turns), "request_turn_count", rid)
        if expected_turns:
            dialogue = "\n".join(f"[{row['turn_ref']}] {row['speaker']}: {row['text']}" for row in expected_turns)
            check(dialogue in prompt, "complete_untruncated_ordered_dialogue", rid)
        else:
            check("[No dialogue supplied.]" in prompt, "no_context_prompt", rid)
        check(item["question"] in prompt and all(row["text"] in prompt for row in request["options"]),
              "public_question_options_in_prompt", rid)
        check(request["prompt_token_count"] + config["max_new_tokens"] <= config["max_model_len"],
              "context_budget", rid)
        if raw.get("prompt_sha256"):
            check(raw["prompt_sha256"] == request["prompt_sha256"], "raw_prompt_binding", rid)

        # Independent scanner; no reused regex, parsing adapter or scorer imports.
        text = raw.get("raw_prediction")
        delimiter = "," if request["question_type"] == "multi_select" else ">"
        parsed = [part.strip() for part in text.strip().split(delimiter)] if isinstance(text, str) else None
        valid = bool(raw.get("status") == "RETURNED" and raw.get("finish_reason") in {"stop", "eos"}
                     and parsed and all(label in labels for label in parsed)
                     and len(parsed) == len(set(parsed)))
        if valid and request["question_type"] == "ordering":
            valid = set(parsed) == labels
        prediction = [mapping[label] for label in parsed] if valid else None
        tp = fp = fn = None
        exact, partial, accuracy = False, Fraction(), 0.0
        if valid:
            if request["question_type"] == "multi_select":
                pred_set, gold_set = set(prediction), set(gold)
                tp, fp, fn = len(pred_set & gold_set), len(pred_set - gold_set), len(gold_set - pred_set)
                partial = Fraction(tp, len(gold_set)) - Fraction(fp, k - len(gold_set))
                exact, prediction = pred_set == gold_set, sorted(pred_set)
                accuracy = tp / len(gold_set)
            else:
                gold_pairs = {(gold[i], gold[j]) for i in range(len(gold)) for j in range(i + 1, len(gold))}
                pred_pairs = {(prediction[i], prediction[j]) for i in range(k) for j in range(i + 1, k)}
                tp, fp, fn = len(gold_pairs & pred_pairs), len(pred_pairs - gold_pairs), 0
                partial = Fraction(tp - fp, math.comb(k, 2))
                exact, accuracy = prediction == gold, tp / math.comb(k, 2)
        expected = {"valid": valid, "exact_correct": exact, "J_fraction": str(partial),
                    "tp": tp, "fp": fp, "fn": fn, "predicted_option_ids": prediction}
        for field, value in expected.items():
            check(observed.get(field) == value, "independent_score_" + field,
                  {"request_id": rid, "expected": value, "actual": observed.get(field)})
        check(abs(observed["J"] - float(partial)) < 1e-12, "independent_score_numeric_J", rid)
        check(abs(observed["recall"] - accuracy) < 1e-12, "independent_recall_or_pair_accuracy", rid)
        if valid and request["question_type"] == "ordering":
            check((observed.get("correct_pairs"), observed.get("wrong_pairs"), observed.get("total_pairs")) ==
                  (tp, fp, math.comb(k, 2)), "dynamic_ordering_pair_statistics", rid)
        independent.append({"request_id": rid, "candidate_id": request["candidate_id"],
                            "context_mode": request["context_mode"], "variant_id": request["variant_id"],
                            "repeat_index": request["repeat_index"], **expected,
                            "recall_or_pair_accuracy": accuracy})

    def summarize(rows):
        exact = sum(row["exact_correct"] for row in rows)
        valid = [row for row in rows if row["valid"]]
        partial = sum((Fraction(row["J_fraction"]) for row in rows), Fraction()) / len(rows)
        predictions = Counter(tuple(row["predicted_option_ids"]) for row in valid)
        return {"planned_calls": len(rows), "valid_calls": len(valid), "invalid_calls": len(rows) - len(valid),
                "exact_correct_calls": exact, "exact_accuracy": exact / len(rows),
                "mean_J_fraction": str(partial), "mean_J": float(partial),
                "known_FP": sum(row["fp"] or 0 for row in valid),
                "known_FN": sum(row["fn"] or 0 for row in valid),
                "distinct_semantic_predictions": len(predictions),
                "modal_valid_prediction_fraction": max(predictions.values()) / len(valid) if valid else None,
                "mean_recall_or_pair_accuracy": sum(row["recall_or_pair_accuracy"] for row in rows) / len(rows)}

    summaries, planned_total = {}, 0
    for cid, item in item_by_id.items():
        k = len(item["options"])
        variants, repeats = (6, 3) if item["question_type"] == "multi_select" else (min(12, math.factorial(k)), 1)
        planned_total += 2 * variants * repeats
        check(item["permutations"] == variants and item["repeats_per_permutation"] == repeats,
              "item_dynamic_sampling_protocol", cid)
        check(len(permutations[cid]) == variants, "frozen_permutation_count", cid)
        frozen_variants = {row["variant_id"]: row for row in permutations[cid]}
        paired, summaries[cid] = {}, {}
        for mode in MODES:
            planned = [row for row in requests if row["candidate_id"] == cid and row["context_mode"] == mode]
            counted = Counter(row["variant_id"] for row in planned)
            check(set(counted) == {f"perm_{i:02d}" for i in range(variants)} and
                  set(counted.values()) == {repeats}, "dynamic_slot_protocol", {"candidate_id": cid, "mode": mode})
            orders = {tuple(row["display_to_option_id"].values()) for row in planned}
            check(len(orders) == variants, "unique_display_permutations", {"candidate_id": cid, "mode": mode})
            for row in planned:
                frozen = frozen_variants[row["variant_id"]]
                check(all(row[field] == frozen[field] for field in ("display_to_option_id", "options", "gold_labels")),
                      "frozen_variant_mapping_binding", row["request_id"])
                check(row["repeat_index"] in range(repeats), "repeat_index_range", row["request_id"])
                slot = (row["variant_id"], row["repeat_index"])
                if mode == "full_context":
                    check(slot not in paired, "unique_sampling_slot", row["request_id"])
                    paired[slot] = row
                else:
                    ref = paired.get(slot)
                    check(ref is not None and all(row[field] == ref[field] for field in
                          ("sampling_seed", "display_to_option_id", "gold_labels", "gold_option_ids", "options", "n", "g")),
                          "paired_context_mapping_and_sampling_seed", row["request_id"])
            rows = [row for row in independent if row["candidate_id"] == cid and row["context_mode"] == mode]
            summary = summarize(rows)
            observed_summary = scores["items"][cid][mode]
            for field, value in summary.items():
                actual = observed_summary.get(field)
                same = abs(value - actual) < 1e-12 if isinstance(value, float) and isinstance(actual, (int, float)) else value == actual
                check(same, "aggregate_" + field, {"candidate_id": cid, "mode": mode, "expected": value, "actual": actual})
            summary["by_permutation"] = {}
            for variant in counted:
                subset = [row for row in rows if row["variant_id"] == variant]
                variant_summary = summarize(subset)
                for field, value in variant_summary.items():
                    actual = observed_summary["by_permutation"][variant].get(field)
                    same = abs(value - actual) < 1e-12 if isinstance(value, float) and isinstance(actual, (int, float)) else value == actual
                    check(same, "permutation_aggregate_" + field,
                          {"candidate_id": cid, "mode": mode, "variant": variant, "expected": value, "actual": actual})
                summary["by_permutation"][variant] = variant_summary
            summaries[cid][mode] = summary
    check(planned_total == 396, "independently_derived_total_396")
    for mode in MODES:
        seeds = [row["sampling_seed"] for row in requests if row["context_mode"] == mode]
        check(len(seeds) == len(set(seeds)), "unique_seeds_within_context_mode", mode)

    result = {"schema_version": "independent_free_generation_solver_score_audit_v2",
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": "PASS" if not errors else "FAIL",
              "method": "Independent standard-library scanner, explicit ordered-pair sets and Fraction arithmetic; no imported runner, scorer or model calls",
              "scoring": {"multi_select": "TP/g - FP/(n-g)",
                          "ordering": "(correct_pairs - wrong_pairs)/C(k,2); variable k",
                          "invalid": "0 in fixed planned denominator; no repaired outputs",
                          "ordering_fn": "0 matches historical native scorer; wrong_pairs counts reversed Gold relations"},
              "checks": dict(checks), "errors": errors,
              "bound_inputs": {name: digest(HERE / name) for name in required},
              "source_manifest_sha256": digest(SOURCE / "MANIFEST_FINAL.json"),
              "item_summaries": summaries, "independent_requests": independent,
              "diagnostic_only": True, "formal_qa_eligible": False}
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"path": str(destination), "status": result["status"],
                      "checks": sum(checks.values()), "errors": errors}, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
