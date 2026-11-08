"""Second-stage repair using discovered outcomes, not reviewed event labels."""

from __future__ import annotations

import hashlib
import json

from run import HERE, SOURCE, SOURCE_SHA256, audit, credential_client, provider_profile, write_json


def main() -> None:
    for name in ("repair_request.json", "repair_response.json", "repair_audit.json"):
        if (HERE / name).exists():
            raise RuntimeError("refusing overwrite: " + name)
    source_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert source_sha == SOURCE_SHA256
    turns = json.loads(SOURCE.read_text(encoding="utf-8"))["turns"]
    previous = json.loads((HERE / "response.json").read_text(encoding="utf-8"))
    prior_events = json.loads(previous["raw_output"])["events"]
    targets = [{"target_index": i, "observed_outcome": event["observed_outcome"],
                "outcome_refs_as_returned": event["outcome_refs"]}
               for i, event in enumerate(prior_events)]
    dialogue = "\n".join(
        f"[{turn['turn_ref']}] {turn['speaker']}: {turn['text']}" for turn in turns
    )
    prompt = (
        "For each previously discovered later occurrence below, find the earlier "
        "proposal and acceptance for that SAME concrete in-person meeting or "
        "activity. The previous occurrence description and refs may be imperfect; "
        "check the source. Search for the SAME participants, place, and day/time "
        "when stated. Do not substitute an earlier general suggestion, introduction, "
        "interest, or plan for a different activity. A proposal must be made by "
        "the speaker at proposal_ref; acceptance must be by the speaker at "
        "acceptance_ref; proposal must precede acceptance and both must precede "
        "the observed occurrence. If no qualifying matched proposal/acceptance "
        "exists, set status to NOT_FOUND, give a brief reason, and leave event null. "
        "Use exact turn-ref strings without enclosing square brackets. Return "
        "only JSON: {\"results\":[{\"target_index\":0,\"status\":\"FOUND\","
        "\"reason\":\"...\",\"event\":{\"proposer\":\"...\","
        "\"accepter\":\"...\",\"proposal_ref\":\"...\","
        "\"acceptance_ref\":\"...\",\"plan_refs\":[],\"outcome_refs\":[],"
        "\"planned_place\":\"...\",\"planned_time\":\"...\","
        "\"observed_outcome\":\"...\"}}]}. Include one result for every target.\n\n"
        f"DISCOVERED OUTCOMES (no proposal/acceptance labels):\n"
        f"{json.dumps(targets, ensure_ascii=False)}\n\nDIALOGUE:\n{dialogue}"
    )
    profile = provider_profile()
    profile["profile_id"] = "h4_r03_relation_extraction_repair_v1"
    profile["request_parameters"].update(temperature=0, max_tokens=2000, enable_thinking=False)
    profile["output_limit"] = 2000
    write_json(HERE / "repair_request.json", {
        "source_path": str(SOURCE), "source_sha256": source_sha,
        "discovered_outcomes": targets, "manual_ledger_supplied": False,
        "request_parameters": profile["request_parameters"], "model": profile["model"],
        "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "training_promotion": False,
    })
    attempts = []
    client = credential_client(profile, on_http_attempt=lambda payload: attempts.append({
        "model": payload.get("model"), "max_tokens": payload.get("max_tokens"),
        "prompt_sha256": hashlib.sha256(payload["messages"][0]["content"].encode()).hexdigest(),
    }))
    try:
        response = client.complete(prompt=prompt, max_tokens=2000, json_object=True)
    except Exception as error:
        write_json(HERE / "repair_response.json", {
            "status": "ERROR", "error_type": type(error).__name__,
            "metadata": getattr(error, "metadata", None), "attempts": attempts,
            "training_promotion": False,
        })
        raise
    write_json(HERE / "repair_response.json", {
        "status": "RESPONSE_RECEIVED", "raw_output": response.raw_text,
        "metadata": response.metadata, "attempts": attempts,
        "source_sha256": source_sha, "training_promotion": False,
    })
    try:
        parsed = json.loads(response.raw_text)
        results = parsed.get("results") if isinstance(parsed, dict) else None
        error = None
    except (ValueError, TypeError) as exc:
        results, error = None, type(exc).__name__
    found = [row.get("event") for row in results if isinstance(row, dict)
             and row.get("status") == "FOUND"] if isinstance(results, list) else None
    checks = audit(found, turns)
    write_json(HERE / "repair_audit.json", {
        "parse_error": error, "results": results,
        "result_count": len(results) if isinstance(results, list) else None,
        "found_count": len(found) if isinstance(found, list) else None,
        "mechanical_checks": checks,
        "all_found_mechanical_pass": bool(checks) and all(row["mechanical_pass"] for row in checks),
        "semantic_review": "PENDING", "training_promotion": False,
    })
    print(json.dumps({"result_count": len(results) if isinstance(results, list) else None,
                      "found_count": len(found) if isinstance(found, list) else None,
                      "all_found_mechanical_pass": all(row["mechanical_pass"] for row in checks)}))


if __name__ == "__main__":
    main()
