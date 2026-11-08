"""One blind appointment-relation extraction from the frozen R03 dialogue."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

WORKTREE = Path("/root/autodl-tmp/gxy-selfplay/worktrees/h4-writer-coverage-v1")
sys.path.insert(0, str(WORKTREE))
from scripts.current_method.restart_n5_spell.h4_memory_light_gate_siliconflow_v1.run import (  # noqa: E402
    credential_client,
    provider_profile,
)

HERE = Path(__file__).resolve().parent
SOURCE = Path("/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_memory_light_gate_siliconflow_qa_v1/prepared_run/frozen/qa_source.json")
SOURCE_SHA256 = "b163bd9ceed0ec0cf4745e5c414fb5d7615788672c879c6283669aad78395113"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_prompt(turns: list[dict]) -> str:
    dialogue = "\n".join(
        f"[{turn['turn_ref']}] {turn['speaker']}: {turn['text']}" for turn in turns
    )
    return (
        "Extract every cross-scene appointment/activity relation in the dialogue below. "
        "A qualifying relation has (1) a person proposing a concrete in-person "
        "meeting or activity, (2) another person accepting it, and (3) a later "
        "scene showing the meeting or activity actually taking place. "
        "A mere future plan, reminder, or statement about a different past event "
        "is not an observed outcome. Exclude uncertain relations. Return at most "
        "eight distinct relations; do not invent any to fill the limit. "
        "Use only exact turn refs from the dialogue. The proposal_ref and "
        "acceptance_ref must be the turns that establish the respective acts; "
        "outcome_refs must directly show the later occurrence. Include plan_refs "
        "only for necessary extra support for the place or time. "
        "Return only a JSON object with an events array. Every event must have "
        "these exact keys: proposer, accepter, proposal_ref, acceptance_ref, "
        "plan_refs (array), outcome_refs (array), planned_place, planned_time, "
        "observed_outcome (brief factual text). Return {\"events\":[]} if none qualify.\n\n"
        f"DIALOGUE:\n{dialogue}"
    )


def audit(events: object, turns: list[dict]) -> list[dict]:
    index = {turn["turn_ref"]: (position, turn) for position, turn in enumerate(turns)}
    if not isinstance(events, list):
        return [{"error": "events_not_array", "mechanical_pass": False}]
    rows = []
    for event in events:
        if not isinstance(event, dict):
            rows.append({"error": "event_not_object", "mechanical_pass": False})
            continue
        refs = [event.get("proposal_ref"), event.get("acceptance_ref")]
        plan_refs = event.get("plan_refs")
        outcome_refs = event.get("outcome_refs")
        checks = {
            "required_fields": all(key in event for key in (
                "proposer", "accepter", "proposal_ref", "acceptance_ref",
                "plan_refs", "outcome_refs", "planned_place", "planned_time",
                "observed_outcome",
            )),
            "reference_shapes": (all(isinstance(ref, str) for ref in refs)
                                 and isinstance(plan_refs, list)
                                 and all(isinstance(ref, str) for ref in plan_refs)
                                 and isinstance(outcome_refs, list) and bool(outcome_refs)
                                 and all(isinstance(ref, str) for ref in outcome_refs)),
        }
        all_refs = refs + (plan_refs if isinstance(plan_refs, list) else []) + (
            outcome_refs if isinstance(outcome_refs, list) else []
        )
        checks["refs_exist"] = checks["reference_shapes"] and all(ref in index for ref in all_refs)
        if checks["refs_exist"]:
            proposal_pos, proposal_turn = index[refs[0]]
            acceptance_pos, acceptance_turn = index[refs[1]]
            checks["proposer_speaker_matches"] = proposal_turn["speaker"] == event.get("proposer")
            checks["accepter_speaker_matches"] = acceptance_turn["speaker"] == event.get("accepter")
            checks["distinct_people"] = event.get("proposer") != event.get("accepter")
            checks["proposal_precedes_acceptance"] = proposal_pos < acceptance_pos
            checks["acceptance_precedes_outcome"] = all(
                acceptance_pos < index[ref][0] for ref in outcome_refs
            )
            checks["outcome_in_later_scene"] = all(
                index[ref][1].get("scene_ref") != acceptance_turn.get("scene_ref")
                for ref in outcome_refs
            )
            checks["outcome_one_scene"] = len({index[ref][1].get("scene_ref") for ref in outcome_refs}) == 1
        rows.append({
            "event_index": len(rows), "checks": checks,
            "mechanical_pass": all(checks.values()),
            "note": "Reference and speaker checks do not establish semantic entailment.",
        })
    return rows


def main() -> None:
    for path in (HERE / "request.json", HERE / "response.json", HERE / "audit.json"):
        if path.exists():
            raise RuntimeError(f"refusing overwrite: {path.name}")
    actual_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if actual_sha != SOURCE_SHA256:
        raise RuntimeError("frozen source hash changed")
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    turns = source["turns"]
    assert len(turns) == 188 and len({turn["turn_ref"] for turn in turns}) == 188
    prompt = build_prompt(turns)
    profile = provider_profile()
    profile["profile_id"] = "h4_r03_relation_extraction_v1"
    profile["request_parameters"].update(temperature=0, max_tokens=2200, enable_thinking=False)
    profile["output_limit"] = 2200
    write_json(HERE / "request.json", {
        "source_path": str(SOURCE), "source_sha256": actual_sha,
        "source_turn_count": len(turns), "model": profile["model"],
        "request_parameters": profile["request_parameters"],
        "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "manual_ledger_supplied": False, "training_promotion": False,
    })
    attempts = []
    client = credential_client(profile, on_http_attempt=lambda payload: attempts.append({
        "model": payload.get("model"), "max_tokens": payload.get("max_tokens"),
        "prompt_sha256": hashlib.sha256(payload["messages"][0]["content"].encode()).hexdigest(),
    }))
    try:
        response = client.complete(prompt=prompt, max_tokens=2200, json_object=True)
    except Exception as error:
        write_json(HERE / "response.json", {
            "status": "ERROR", "error_type": type(error).__name__,
            "metadata": getattr(error, "metadata", None),
            "attempts": attempts, "training_promotion": False,
        })
        raise
    write_json(HERE / "response.json", {
        "status": "RESPONSE_RECEIVED", "raw_output": response.raw_text,
        "metadata": response.metadata, "attempts": attempts,
        "source_sha256": actual_sha, "training_promotion": False,
    })
    try:
        parsed = json.loads(response.raw_text)
        events = parsed.get("events") if isinstance(parsed, dict) else None
        error = None
    except (ValueError, TypeError) as exc:
        events, error = None, type(exc).__name__
    rows = audit(events, turns)
    write_json(HERE / "audit.json", {
        "parse_error": error, "event_count": len(events) if isinstance(events, list) else None,
        "events": events, "mechanical_checks": rows,
        "all_mechanical_pass": bool(rows) and all(row["mechanical_pass"] for row in rows),
        "semantic_review": "PENDING", "training_promotion": False,
    })
    print(json.dumps({"event_count": len(events) if isinstance(events, list) else None,
                      "all_mechanical_pass": all(row["mechanical_pass"] for row in rows)}))


if __name__ == "__main__":
    main()
