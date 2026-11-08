"""Adversarial second pass over first-pass accepted ordering candidates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import urllib.request

from judge import HERE, SOURCE, LABELS, canonical, read, save_new, sha
from judge_deepseek import MODEL, RUN as FIRST_RUN, config


RUN = HERE / "semantic_challenge_deepseek_v3"
ROOT_IDS = ["210046", "231899", "194259", "96544"]


def prompt(root_id: str) -> tuple[str, list[str]]:
    first = read(FIRST_RUN / "SEMANTIC_REVIEW.json")
    root_review = next(row for row in first["roots"] if row["root_id"] == root_id)
    accepted = [row["candidate_id"] for row in root_review["checks"] if row["semantic_pass"]]
    rows = [row for row in read(HERE / "LABEL_REBOUND_UNREVIEWED.json")
            if row["candidate_id"] in accepted]
    source = read(SOURCE / root_id / "source.json")
    by_ref = {turn["turn_ref"]: turn for turn in source["turns"]}
    data = []
    for row in rows:
        data.append({"candidate_id": row["candidate_id"], "items": [
            {"label": label, "claim": row["items"][label],
             "speaker": row["item_evidence"][label]["speaker"],
             "quote": row["item_evidence"][label]["quote"],
             "full_turn": by_ref[row["item_evidence"][label]["turn_ref"]]}
            for label in LABELS]})
    instruction = """Audit each claim against ONLY its quote and source speaker. Your job is to find hidden unsupported details missed by a prior reviewer; do not defer to any prior result. A claim passes only if every material assertion is explicit in the cited quote or follows directly from the speaker identity. Do not use neighboring dialogue, topic, or plausible context to fill missing actor, object, place, event, time, or status. Examples: 'I'm here' does not establish 'at the office'; 'I'll do it' does not identify what 'it' is; an offer or plan is not a completed action. If a claim needs a neighboring turn to resolve a pronoun or referent, fail it. Preserve distinction between suggestion, intention, promise, and report of completion. When uncertain, fail.
Return one JSON object: {"reviews":[{"candidate_id":"...","items":{"I1":{"entailed":true,"unsupported_details":[],"reason":"..."},..."I6":{"entailed":false,"unsupported_details":["specific missing claim"],"reason":"..."}}},...]}. Include every candidate and every I1-I6 exactly once. A true item must have an empty unsupported_details list. A false item must name at least one concrete unsupported detail. Use JSON booleans. No Markdown.
"""
    return instruction + "\nDATA:\n" + json.dumps(data, ensure_ascii=False), accepted


def prepare() -> None:
    for suffix in ROOT_IDS:
        root_id = "cc-train-episode-" + suffix
        text, ids = prompt(root_id)
        if not ids:
            continue
        payload = {"model": MODEL, "messages": [{"role": "user", "content": text}],
                   "temperature": 0, "max_tokens": 4096, "stream": False,
                   "response_format": {"type": "json_object"}}
        index = {"status": "PREPARED_NO_API_CALL", "root_id": root_id,
                 "candidate_ids": ids, "reviewer_model": MODEL,
                 "prior_review_sha256": sha((FIRST_RUN / "SEMANTIC_REVIEW.json").read_bytes()),
                 "prompt_sha256": sha(text.encode("utf-8")),
                 "payload_sha256": sha(canonical(payload)),
                 "source_sha256": sha((SOURCE / root_id / "source.json").read_bytes())}
        directory = RUN / root_id
        save_new(directory / "REQUEST.json", payload)
        save_new(directory / "INDEX.json", index)
    print("prepared adversarial checks for first-pass accepted roots")


def execute(root_id: str) -> None:
    settings = config()
    directory = RUN / root_id
    index, payload = read(directory / "INDEX.json"), read(directory / "REQUEST.json")
    text, ids = prompt(root_id)
    if (index["candidate_ids"] != ids or index["prompt_sha256"] != sha(text.encode("utf-8"))
            or index["payload_sha256"] != sha(canonical(payload))
            or index["prior_review_sha256"] != sha((FIRST_RUN / "SEMANTIC_REVIEW.json").read_bytes())):
        raise ValueError("challenge_request_changed")
    if (directory / "RESPONSE.json").exists() or (directory / "ERROR.json").exists():
        raise FileExistsError("challenge_attempt_already_completed")
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
            if not isinstance(item, dict) or type(item.get("entailed")) is not bool:
                errors.append(f"{candidate_id}:{label}:verdict_invalid")
                continue
            details = item.get("unsupported_details")
            if (not isinstance(details, list) or any(not isinstance(x, str) or not x.strip() for x in details)
                    or (item["entailed"] and details) or (not item["entailed"] and not details)
                    or not isinstance(item.get("reason"), str) or not item["reason"].strip()):
                errors.append(f"{candidate_id}:{label}:details_invalid")
    if set(by_id) != set(ids):
        errors.append("review_ids_incomplete")
    return by_id if not errors else {}, errors


def finalize() -> None:
    first = read(FIRST_RUN / "SEMANTIC_REVIEW.json")
    result = {"status": "ADVERSARIAL_SEMANTIC_CHALLENGE_COMPLETED",
              "reviewer_provider": "siliconflow", "reviewer_model_requested": MODEL,
              "first_pass_review_sha256": sha((FIRST_RUN / "SEMANTIC_REVIEW.json").read_bytes()),
              "no_codex_content_review": True, "solver_calls": 0, "selected_for_training": 0,
              "roots": []}
    for root in first["roots"]:
        root_id = root["root_id"]
        accepted = [row["candidate_id"] for row in root["checks"] if row["semantic_pass"]]
        if not accepted:
            continue
        directory = RUN / root_id
        index = read(directory / "INDEX.json")
        response_path = directory / "RESPONSE.json"
        if not response_path.exists():
            envelope, by_id, errors, response_sha = {}, {}, ["response_missing"], None
        else:
            response = read(response_path)
            envelope = response["envelope"]
            by_id, errors = parse(envelope, accepted)
            response_sha = sha(response_path.read_bytes())
            if response["payload_sha256"] != index["payload_sha256"]:
                errors.append("payload_binding_invalid")
        checks = []
        for candidate_id in accepted:
            review = by_id.get(candidate_id)
            passed = bool(review and not errors and all(
                review["items"][label]["entailed"] for label in LABELS))
            checks.append({"candidate_id": candidate_id, "challenge_pass": passed,
                           "review": review, "schema_or_binding_errors": errors})
        result["roots"].append({"root_id": root_id, "candidate_count": len(checks),
            "challenge_pass_count": sum(row["challenge_pass"] for row in checks),
            "reviewer_model_returned": envelope.get("model"),
            "reviewer_response_id": envelope.get("id"), "usage": envelope.get("usage"),
            "prompt_sha256": index["prompt_sha256"], "payload_sha256": index["payload_sha256"],
            "response_file_sha256": response_sha, "source_sha256": index["source_sha256"],
            "schema_or_binding_errors": errors, "checks": checks})
    result["total_challenge_pass"] = sum(root["challenge_pass_count"] for root in result["roots"])
    save_new(RUN / "CHALLENGE_REVIEW.json", result)
    print(json.dumps({"challenge_pass": result["total_challenge_pass"],
                      "first_pass": first["total_semantic_pass"]}))


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
