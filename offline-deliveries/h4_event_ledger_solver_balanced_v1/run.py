"""Balanced presentation probe for the four reviewed event-ledger questions."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path

from scripts.current_method.restart_n5_spell.h4_memory_light_gate_siliconflow_v1.run import (
    credential_client, provider_profile,
)
from scripts.current_method.restart_n5_spell.h4_round0_b0_k3_difficulty_v1.contract import score_output


BASE = Path(__file__).resolve().parents[1]
REVIEW = BASE / "h4_event_ledger_qa_review_repeat01"
SOURCE = BASE / "h4_memory_light_gate_siliconflow_qa_v1/prepared_run/frozen/qa_source.json"
OUT = Path(__file__).resolve().parent
ORDERING_ORDERS = ("ABCD", "DCBA", "ABDC", "CDBA", "BCAD", "DACB", "BDAC", "CADB")
MODEL = "Qwen/Qwen3.6-35B-A3B"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def orders_for(question_type, keys):
    if question_type == "ordering":
        assert len(keys) == 4
        return [tuple(keys[ord(char) - ord("A")] for char in order) for order in ORDERING_ORDERS]
    assert question_type == "multi_select" and len(keys) == 5
    return [tuple(keys[shift:] + keys[:shift]) for shift in range(5)] + [
        tuple(list(reversed(keys))[shift:] + list(reversed(keys))[:shift]) for shift in range(5)
    ]


def variants(candidate):
    qtype = candidate["question_type"]
    field = "items" if qtype == "ordering" else "options"
    keys = list(candidate[field])
    orders = orders_for(qtype, keys)
    assert len(set(orders)) == len(orders)
    result = []
    for index, order in enumerate(orders):
        mapping = dict(zip(keys, order))
        inverse = {canonical: display for display, canonical in mapping.items()}
        variant = {
            "variant_id": f"perm_{index:02d}",
            "display_to_canonical": mapping,
            field: {display: candidate[field][canonical] for display, canonical in mapping.items()},
            "displayed_gold": [inverse[key] for key in candidate["gold"]],
        }
        assert score_output("<box>" + ",".join(variant["displayed_gold"]) + "</box>", candidate, variant)["correct"]
        result.append(variant)
    assert all(sum(v["display_to_canonical"][display] == canonical for v in result) == 2
               for display in keys for canonical in keys)
    if qtype == "ordering":
        assert all(sum(order.index(a) < order.index(b) for order in orders) == 4
                   for a, b in itertools.combinations(keys, 2))
        adjacent = Counter((a, b) for order in orders for a, b in zip(order, order[1:]))
        assert all(adjacent[a, b] == 2 for a in keys for b in keys if a != b)
    return result


def transcript(source):
    return "\n".join(f"[{turn['turn_ref']}] {turn['speaker']}: {turn['text']}" for turn in source["turns"])


def prompt(candidate, variant, context, mode):
    qtype = candidate["question_type"]
    field = "items" if qtype == "ordering" else "options"
    lines = ["Answer the question using only the supplied dialogue when present.",
             "Return exactly one <box>...</box> pair, with no other text."]
    if mode == "full_context":
        lines += ["Dialogue:", context]
    else:
        assert mode == "no_context"
    lines += ["Question:", candidate["question"],
              "Items:" if qtype == "ordering" else "Options:",
              "\n".join(f"{key}. {text}" for key, text in variant[field].items()),
              ("Put all item IDs in chronological order, separated by commas inside the box."
               if qtype == "ordering" else
               "Put all supported option letters in the box, separated by commas.")]
    return "\n\n".join(lines)


def build():
    questions_path = REVIEW / "solver_questions.json"
    gold_path = REVIEW / "solver_gold.json"
    questions = read(questions_path)["candidates"]
    gold = read(gold_path)
    source = read(SOURCE)
    assert len(questions) == 4 and len(source["turns"]) == 188
    assert {q["candidate_id"] for q in questions} == set(gold)
    context = transcript(source)
    requests = []
    for question in questions:
        candidate = deepcopy(question)
        candidate["gold"] = gold[candidate["candidate_id"]]
        for variant in variants(candidate):
            for mode in ("no_context", "full_context"):
                request_id = f"{candidate['candidate_id']}__{variant['variant_id']}__{mode}"
                text = prompt(candidate, variant, context, mode)
                if mode == "no_context":
                    assert "Dialogue:" not in text and source["turns"][0]["text"] not in text
                assert "displayed_gold" not in text and "display_to_canonical" not in text
                requests.append({"request_id": request_id, "candidate_id": candidate["candidate_id"],
                                 "mode": mode, "variant": variant,
                                 "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                 "prompt": text})
    assert len(requests) == 76
    manifest = {"version": "h4_event_ledger_solver_balanced_v1", "model": MODEL,
                "question_count": 4, "request_count": len(requests), "rollouts_per_position": 1,
                "input_sha256": {str(path): sha(path) for path in (questions_path, gold_path, SOURCE)},
                "requests": [{key: request[key] for key in ("request_id", "candidate_id", "mode", "prompt_sha256")}
                             for request in requests],
                "ordering_orders": list(ORDERING_ORDERS), "training_promotion": False}
    return questions, gold, requests, manifest


def preflight():
    _, _, requests, manifest = build()
    path = OUT / "manifest.json"
    if path.exists():
        assert read(path) == manifest, "frozen_manifest_drift"
    else:
        write(path, manifest)
    return {"status": "PASS", "request_count": len(requests), "model": MODEL,
            "ordering_variants": 8, "multi_select_variants_per_question": 10}


def invoke(request):
    profile = deepcopy(provider_profile())
    profile["profile_id"] = "h4_event_ledger_solver_balanced_v1"
    profile["request_parameters"].update(temperature=0.0, max_tokens=128, enable_thinking=False)
    profile["request_parameters"].pop("response_format", None)
    profile["output_limit"] = 128
    attempts = []
    def on_attempt(payload):
        attempts.append({"model": payload.get("model"), "max_tokens": payload.get("max_tokens")})
    client = credential_client(profile, on_http_attempt=on_attempt)
    response = client.complete(prompt=request["prompt"], max_tokens=128, json_object=False)
    return response, attempts


def live():
    questions, gold, requests, manifest = build()
    assert (OUT / "manifest.json").exists(), "run_preflight_first"
    assert read(OUT / "manifest.json") == manifest, "frozen_manifest_drift"
    candidates = {q["candidate_id"]: {**q, "gold": gold[q["candidate_id"]]} for q in questions}
    pending = [request for request in requests if not (OUT / "responses" / (request["request_id"] + ".json")).exists()]
    errors = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(invoke, request): request for request in pending}
        for future in as_completed(futures):
            request = futures[future]
            try:
                response, attempts = future.result()
                candidate = candidates[request["candidate_id"]]
                scored = score_output(response.raw_text, candidate, request["variant"])
                metadata = response.metadata
                record = {"request_id": request["request_id"], "candidate_id": request["candidate_id"],
                          "question_type": candidate["question_type"], "mode": request["mode"],
                          "variant_id": request["variant"]["variant_id"],
                          "display_to_canonical": request["variant"]["display_to_canonical"],
                          "displayed_gold": request["variant"]["displayed_gold"],
                          "prompt_sha256": request["prompt_sha256"], "raw_output": response.raw_text,
                          "score": scored, "attempts": attempts,
                          "metadata": {key: metadata.get(key) for key in (
                              "actual_model", "finish_reason", "usage", "http_status", "completion_status")}}
                write(OUT / "responses" / (request["request_id"] + ".json"), record)
                print(json.dumps({"request_id": request["request_id"], "scored": scored["scored"],
                                  "correct": scored["correct"]}), flush=True)
            except Exception as exc:
                errors.append({"request_id": request["request_id"], "error_type": type(exc).__name__,
                               "error": str(exc)[:300]})
                print(json.dumps(errors[-1]), flush=True)
    if errors:
        write(OUT / "errors.json", errors)
    return summarize()


def summarize():
    _, _, requests, manifest = build()
    assert read(OUT / "manifest.json") == manifest, "frozen_manifest_drift"
    rows = [read(path) for path in sorted((OUT / "responses").glob("*.json"))] if (OUT / "responses").exists() else []
    by_id = {row["request_id"]: row for row in rows}
    assert len(rows) == len(by_id)
    assert set(by_id).issubset({request["request_id"] for request in requests})
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["candidate_id"], row["mode"]].append(row)
    per_question = []
    for cid in sorted({request["candidate_id"] for request in requests}):
        item = {"candidate_id": cid}
        for mode in ("no_context", "full_context"):
            group = grouped[cid, mode]
            item[mode] = {"correct": sum(row["score"]["correct"] is True for row in group),
                          "scored": sum(row["score"]["scored"] is True for row in group),
                          "completed": len(group), "expected": 8 if cid == "Q04" else 10,
                          "canonical_predictions": Counter(
                              ",".join(row["score"]["canonical_prediction"]) for row in group)}
        per_question.append(item)
    totals = Counter()
    for row in rows:
        usage = row["metadata"].get("usage") or {}
        totals["input_tokens"] += usage.get("prompt_tokens") or 0
        totals["output_tokens"] += usage.get("completion_tokens") or 0
    summary = {"status": "COMPLETE" if len(rows) == len(requests) else "INCOMPLETE",
               "model": MODEL, "completed": len(rows), "expected": len(requests),
               "per_question": per_question, "usage": dict(totals),
               "training_promotion": False}
    write(OUT / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "live", "summarize"))
    args = parser.parse_args()
    print(json.dumps({"preflight": preflight, "live": live, "summarize": summarize}[args.command](),
                     ensure_ascii=False, default=dict), flush=True)
