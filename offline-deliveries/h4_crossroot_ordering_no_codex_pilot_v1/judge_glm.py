"""Different-checkpoint strict quote-entailment audit of all rebound items."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import urllib.request

from judge import HERE, SOURCE, LABELS, ROOT_IDS, canonical, inputs, read, save_new, sha
from judge_deepseek import config


MODEL = "zai-org/GLM-5.2"
RUN = HERE / "semantic_review_glm_v4"


def prompt(root_id: str) -> tuple[str, list[str]]:
    rows, source, history = inputs(root_id)
    by_ref = {turn["turn_ref"]: turn for turn in source["turns"]}
    candidates = []
    for row in rows:
        candidates.append({
            "candidate_id": row["candidate_id"], "topic": row["topic"],
            "question": row["question"], "gold": row["gold"],
            "items": [{"label": label, "claim": row["items"][label],
                       "quote": row["item_evidence"][label]["quote"],
                       "source_turn": by_ref[row["item_evidence"][label]["turn_ref"]]}
                      for label in LABELS],
        })
    prior = [{key: item.get(key) for key in
              ("candidate_id", "question_type", "topic", "question", "evidence_refs", "item_text")}
             for item in history]
    instruction = """You are an evidence auditor independent of the question generator and a prior reviewer. Audit ALL supplied ordering candidates, including ones another model might have rejected.
For each item, decide whether the entire claim is supported by its exact quoted text and the source-turn speaker only. Do not use the adjacent dialogue, topic, question, historical candidates, or world knowledge to supply a missing referent. In particular, 'it', 'them', 'here', 'there', 'this', 'that', 'we', or an unspecified event/place/object do NOT identify a named target unless the quote itself names it. 'Not good at speaking' does not explicitly mean 'nervous'. A suggestion, intention, promise, decision, report, and completed action are distinct. If any person, place, object, time, state, or modality in the claim is merely plausible rather than entailed, set supported=false and name that exact unsupported span. Ignore the generator's evidence reason. Do not edit claims.
Separately decide whether all six claims share a natural topic, whether the whole question repeats a historical candidate's tested facts/event sequence, and whether it repeats another candidate in this current batch. Historical candidates and current candidates are supplied only for these whole-question duplication checks, never to rescue an item quote.
Return one JSON object only: {"reviews":[{"candidate_id":"...","items":{"I1":{"supported":true,"unsupported_spans":[],"reason":"..."},..."I6":{"supported":false,"unsupported_spans":["exact unsupported phrase"],"reason":"..."}},"coherent":true,"novel_vs_history":true,"distinct_within_batch":true,"reason":"..."},...]}. Each supplied ID and I1-I6 exactly once. A supported item must have no unsupported_spans; an unsupported item must have at least one. Use JSON booleans. No Markdown.
"""
    return instruction + "\nDATA:\n" + json.dumps({"root_id": root_id,
        "current_candidates": candidates, "historical_candidates": prior}, ensure_ascii=False), [r["candidate_id"] for r in rows]


def prepare() -> None:
    endpoint = config()["endpoint"]
    for suffix in ROOT_IDS:
        root_id = "cc-train-episode-" + suffix
        text, ids = prompt(root_id)
        payload = {"model": MODEL, "messages": [{"role": "user", "content": text}],
                   "temperature": 0, "max_tokens": 8192, "stream": False,
                   "response_format": {"type": "json_object"}}
        index = {"status": "PREPARED_NO_API_CALL", "root_id": root_id,
                 "candidate_ids": ids, "generator_model": "Qwen/Qwen3.6-35B-A3B",
                 "reviewer_provider": "siliconflow", "reviewer_model_requested": MODEL,
                 "endpoint": endpoint, "temperature": 0, "max_tokens": 8192, "retry_limit": 0,
                 "prompt_sha256": sha(text.encode("utf-8")),
                 "payload_sha256": sha(canonical(payload)),
                 "source_sha256": sha((SOURCE / root_id / "source.json").read_bytes()),
                 "label_candidate_file_sha256": sha((HERE / "LABEL_REBOUND_UNREVIEWED.json").read_bytes()),
                 "prior_summary_sha256": sha((SOURCE / root_id / "PRIOR_CANDIDATE_SUMMARY.json").read_bytes())}
        directory = RUN / root_id
        save_new(directory / "REQUEST.json", payload)
        save_new(directory / "INDEX.json", index)
    print("prepared 4 independent GLM review requests")


def execute(root_id: str) -> None:
    settings = config()
    directory = RUN / root_id
    index, payload = read(directory / "INDEX.json"), read(directory / "REQUEST.json")
    text, ids = prompt(root_id)
    if (index["candidate_ids"] != ids or index["prompt_sha256"] != sha(text.encode("utf-8"))
            or index["payload_sha256"] != sha(canonical(payload))
            or payload["messages"] != [{"role": "user", "content": text}]):
        raise ValueError("glm_frozen_request_changed")
    if (directory / "RESPONSE.json").exists() or (directory / "ERROR.json").exists():
        raise FileExistsError("glm_attempt_already_completed")
    save_new(directory / "ATTEMPT.json", {"status": "RESERVED_ONE_HTTP_ATTEMPT",
              "requested_at": datetime.now(timezone.utc).isoformat(),
              "payload_sha256": index["payload_sha256"]})
    request = urllib.request.Request(settings["endpoint"], data=canonical(payload),
                headers={"Content-Type": "application/json", "Authorization": "Bearer " + settings["secret"]},
                method="POST")
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            raw, http_status = response.read(), response.status
    except Exception as error:
        save_new(directory / "ERROR.json", {"status": "ERROR_NO_RETRY", "error_type": type(error).__name__,
                 "error": str(error), "payload_sha256": index["payload_sha256"]})
        raise
    save_new(directory / "RESPONSE.json", {"status": "RESPONSE_RECEIVED", "root_id": root_id,
             "http_status": http_status, "raw_response_sha256": sha(raw),
             "payload_sha256": index["payload_sha256"], "envelope": json.loads(raw)})
    print(json.dumps({"root_id": root_id, "response_sha256": sha(raw)}))


def parse(envelope: dict, ids: list[str]) -> tuple[dict, list[str]]:
    errors = []
    if envelope.get("model") != MODEL:
        errors.append("returned_model_mismatch")
    choices = envelope.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or choices[0].get("finish_reason") != "stop":
        return {}, errors + ["choices_invalid"]
    try:
        data = json.loads(choices[0]["message"]["content"])
    except (KeyError, TypeError, ValueError):
        return {}, errors + ["json_invalid"]
    rows = data.get("reviews") if isinstance(data, dict) else None
    if not isinstance(rows, list) or len(rows) != len(ids):
        return {}, errors + ["review_count_invalid"]
    by_id = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("candidate_id") not in ids:
            errors.append("candidate_id_invalid")
            continue
        candidate_id = row["candidate_id"]
        if candidate_id in by_id:
            errors.append("candidate_id_duplicate")
        by_id[candidate_id] = row
        items = row.get("items")
        if not isinstance(items, dict) or set(items) != set(LABELS):
            errors.append(f"{candidate_id}:item_set_invalid")
            continue
        for label, item in items.items():
            if not isinstance(item, dict) or type(item.get("supported")) is not bool:
                errors.append(f"{candidate_id}:{label}:verdict_invalid")
                continue
            details = item.get("unsupported_spans")
            if (not isinstance(details, list) or any(not isinstance(x, str) or not x.strip() for x in details)
                    or (item["supported"] and details) or (not item["supported"] and not details)
                    or not isinstance(item.get("reason"), str) or not item["reason"].strip()):
                errors.append(f"{candidate_id}:{label}:details_invalid")
        for field in ("coherent", "novel_vs_history", "distinct_within_batch"):
            if type(row.get(field)) is not bool:
                errors.append(f"{candidate_id}:{field}_invalid")
        if not isinstance(row.get("reason"), str) or not row["reason"].strip():
            errors.append(f"{candidate_id}:reason_invalid")
    if set(by_id) != set(ids):
        errors.append("review_ids_incomplete")
    return by_id if not errors else {}, errors


def finalize() -> None:
    result = {"status": "DIFFERENT_CHECKPOINT_SEMANTIC_REVIEW_COMPLETED",
              "reviewer_provider": "siliconflow", "reviewer_model_requested": MODEL,
              "no_codex_content_review": True, "solver_calls": 0, "selected_for_training": 0,
              "roots": []}
    for suffix in ROOT_IDS:
        root_id = "cc-train-episode-" + suffix
        directory = RUN / root_id
        index = read(directory / "INDEX.json")
        response_path = directory / "RESPONSE.json"
        if not response_path.exists():
            envelope, by_id, errors, response_sha = {}, {}, ["response_missing"], None
        else:
            response = read(response_path)
            envelope = response["envelope"]
            by_id, errors = parse(envelope, index["candidate_ids"])
            response_sha = sha(response_path.read_bytes())
            if response["payload_sha256"] != index["payload_sha256"]:
                errors.append("payload_binding_invalid")
        checks = []
        for candidate_id in index["candidate_ids"]:
            review = by_id.get(candidate_id)
            passed = bool(review and not errors and all(
                review["items"][label]["supported"] for label in LABELS)
                and all(review[field] for field in
                        ("coherent", "novel_vs_history", "distinct_within_batch")))
            checks.append({"candidate_id": candidate_id, "semantic_pass": passed,
                           "review": review, "schema_or_binding_errors": errors})
        result["roots"].append({"root_id": root_id, "candidate_count": len(checks),
            "semantic_pass_count": sum(row["semantic_pass"] for row in checks),
            "reviewer_model_returned": envelope.get("model"),
            "reviewer_response_id": envelope.get("id"), "usage": envelope.get("usage"),
            "prompt_sha256": index["prompt_sha256"], "payload_sha256": index["payload_sha256"],
            "response_file_sha256": response_sha, "source_sha256": index["source_sha256"],
            "schema_or_binding_errors": errors, "checks": checks})
    result["total_semantic_pass"] = sum(root["semantic_pass_count"] for root in result["roots"])
    save_new(RUN / "SEMANTIC_REVIEW.json", result)
    print(json.dumps({"semantic_pass": result["total_semantic_pass"],
                      "reviewed": sum(root["candidate_count"] for root in result["roots"])}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--execute", choices=["cc-train-episode-" + value for value in ROOT_IDS])
    group.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.execute:
        execute(args.execute)
    else:
        finalize()
