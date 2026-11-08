"""Separate DeepSeek reviewer attempt after TJU's pre-HTTP TLS failure."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.request

from judge import HERE, SOURCE, ROOT_IDS, canonical, parse_review, prompt, read, save_new, sha


MODEL = "deepseek-ai/DeepSeek-V3.2"
PROVIDER = "siliconflow"
CONFIG = Path("/root/autodl-tmp/gxy-selfplay/config.json")
RUN = HERE / "semantic_review_deepseek_v2"


def config() -> dict:
    settings = read(CONFIG)["api"]
    if settings.get("base_url") != "https://api.siliconflow.cn/v1" or not settings.get("api_key"):
        raise ValueError("siliconflow_config_invalid")
    return {"endpoint": settings["base_url"] + "/chat/completions", "secret": settings["api_key"]}


def prepare() -> None:
    endpoint = config()["endpoint"]
    for suffix in ROOT_IDS:
        root_id = "cc-train-episode-" + suffix
        text, candidate_ids = prompt(root_id)
        payload = {"model": MODEL, "messages": [{"role": "user", "content": text}],
                   "temperature": 0, "max_tokens": 8192, "stream": False,
                   "response_format": {"type": "json_object"}}
        index = {
            "status": "PREPARED_NO_API_CALL", "root_id": root_id,
            "candidate_ids": candidate_ids, "generator_model": "Qwen/Qwen3.6-35B-A3B",
            "reviewer_provider": PROVIDER, "reviewer_model_requested": MODEL,
            "endpoint": endpoint, "temperature": 0, "max_tokens": 8192, "retry_limit": 0,
            "prompt_sha256": sha(text.encode("utf-8")), "payload_sha256": sha(canonical(payload)),
            "source_sha256": sha((SOURCE / root_id / "source.json").read_bytes()),
            "label_candidate_file_sha256": sha((HERE / "LABEL_REBOUND_UNREVIEWED.json").read_bytes()),
            "prior_summary_sha256": sha((SOURCE / root_id / "PRIOR_CANDIDATE_SUMMARY.json").read_bytes()),
            "prior_tju_attempt": str(HERE / "semantic_review" / root_id / "ERROR.json") if
                (HERE / "semantic_review" / root_id / "ERROR.json").exists() else None,
        }
        directory = RUN / root_id
        save_new(directory / "REQUEST.json", payload)
        save_new(directory / "INDEX.json", index)
    print("prepared 4 independent DeepSeek review requests")


def execute(root_id: str) -> None:
    settings = config()
    directory = RUN / root_id
    index, payload = read(directory / "INDEX.json"), read(directory / "REQUEST.json")
    text, ids = prompt(root_id)
    if (index["candidate_ids"] != ids or index["prompt_sha256"] != sha(text.encode("utf-8"))
            or index["payload_sha256"] != sha(canonical(payload))
            or payload["messages"] != [{"role": "user", "content": text}]):
        raise ValueError("frozen_deepseek_request_changed")
    if (directory / "RESPONSE.json").exists() or (directory / "ERROR.json").exists():
        raise FileExistsError("review_attempt_already_completed")
    save_new(directory / "ATTEMPT.json", {"status": "RESERVED_ONE_HTTP_ATTEMPT",
              "requested_at": datetime.now(timezone.utc).isoformat(),
              "payload_sha256": index["payload_sha256"]})
    request = urllib.request.Request(
        settings["endpoint"], data=canonical(payload),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + settings["secret"]},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            raw = response.read()
            http_status = response.status
    except Exception as error:
        save_new(directory / "ERROR.json", {"status": "ERROR_NO_RETRY", "error_type": type(error).__name__,
                 "error": str(error), "payload_sha256": index["payload_sha256"]})
        raise
    envelope = json.loads(raw)
    save_new(directory / "RESPONSE.json", {"status": "RESPONSE_RECEIVED", "root_id": root_id,
             "http_status": http_status, "raw_response_sha256": sha(raw),
             "payload_sha256": index["payload_sha256"], "envelope": envelope})
    print(json.dumps({"root_id": root_id, "returned_model": envelope.get("model"),
                      "usage": envelope.get("usage"), "response_sha256": sha(raw)}))


def finalize() -> None:
    result = {"status": "AUTOMATIC_SEMANTIC_REVIEW_COMPLETED",
              "generator_model": "Qwen/Qwen3.6-35B-A3B", "reviewer_provider": PROVIDER,
              "reviewer_model_requested": MODEL, "no_codex_content_review": True,
              "solver_calls": 0, "selected_for_training": 0, "roots": []}
    for suffix in ROOT_IDS:
        root_id = "cc-train-episode-" + suffix
        directory = RUN / root_id
        index = read(directory / "INDEX.json")
        response_path = directory / "RESPONSE.json"
        if not response_path.exists():
            by_id, errors, envelope, response_sha = {}, ["response_missing"], {}, None
        else:
            response = read(response_path)
            response_sha = sha(response_path.read_bytes())
            envelope = response.get("envelope", {})
            by_id, errors = parse_review(envelope, index["candidate_ids"], MODEL)
            if response.get("payload_sha256") != index["payload_sha256"]:
                errors.append("response_payload_binding_invalid")
        checks = []
        for candidate_id in index["candidate_ids"]:
            review = by_id.get(candidate_id)
            passed = bool(review and not errors and all(
                review["items"][label]["supported"] for label in ("I1", "I2", "I3", "I4", "I5", "I6")
            ) and all(review[field] for field in
                      ("coherent", "novel_vs_history", "distinct_within_batch")))
            checks.append({"candidate_id": candidate_id, "semantic_pass": passed,
                           "review": review, "schema_or_binding_errors": errors})
        result["roots"].append({
            "root_id": root_id, "candidate_count": len(checks),
            "semantic_pass_count": sum(row["semantic_pass"] for row in checks),
            "reviewer_model_returned": envelope.get("model"),
            "reviewer_response_id": envelope.get("id"), "usage": envelope.get("usage"),
            "prompt_sha256": index["prompt_sha256"], "payload_sha256": index["payload_sha256"],
            "response_file_sha256": response_sha, "source_sha256": index["source_sha256"],
            "candidate_file_sha256": index["label_candidate_file_sha256"],
            "schema_or_binding_errors": errors, "checks": checks,
        })
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
