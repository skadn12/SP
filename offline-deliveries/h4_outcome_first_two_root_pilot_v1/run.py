"""Bounded, ledger-free outcome-first extraction on two frozen R02 dialogues."""

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
SOURCES = {
    "112812": (Path("/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_gate_downstream_solver_eval_v1/run/dialogues/cc-train-episode-112812.json"), "245f316a8ff57bcf74886d7223afdbfee815cc7f16554e586edd42943cfa9ea7"),
    "96462": (Path("/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_gate_downstream_solver_eval_v1/run/dialogues/cc-train-episode-96462.json"), "aed0e64230751f633c4359d6f8365b6d3f20ce12a5be3022aee97bb1c978e5f7"),
}


def write_new(path: Path, obj: object) -> None:
    with path.open("x", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, indent=2)
        file.write("\n")


def call(stage: str, root: str, prompt: str, source_path: Path, source_sha: str) -> dict:
    prefix = HERE / f"{root}_{stage}"
    profile = provider_profile()
    profile["profile_id"] = f"h4_outcome_first_two_root_pilot_v1_{root}_{stage}"
    profile["request_parameters"].update(temperature=0, max_tokens=2400, enable_thinking=False)
    profile["output_limit"] = 2400
    write_new(prefix.with_suffix(".request.json"), {
        "source_path": str(source_path), "source_sha256": source_sha,
        "model": profile["model"], "parameters": profile["request_parameters"],
        "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "manual_ledger_supplied": False, "training_promotion": False,
    })
    attempts = []
    client = credential_client(profile, on_http_attempt=lambda payload: attempts.append({
        "model": payload.get("model"), "max_tokens": payload.get("max_tokens"),
        "prompt_sha256": hashlib.sha256(payload["messages"][0]["content"].encode()).hexdigest(),
    }))
    try:
        response = client.complete(prompt=prompt, max_tokens=2400, json_object=True)
        record = {"status": "RESPONSE_RECEIVED", "raw_output": response.raw_text,
                  "metadata": response.metadata, "attempts": attempts,
                  "source_sha256": source_sha, "training_promotion": False}
    except Exception as error:
        record = {"status": "ERROR", "error_type": type(error).__name__,
                  "message": str(error), "metadata": getattr(error, "metadata", None),
                  "attempts": attempts, "training_promotion": False}
    write_new(prefix.with_suffix(".response.json"), record)
    return record


def main() -> None:
    for root, (source_path, expected_sha) in SOURCES.items():
        actual_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            raise RuntimeError(f"source changed: {root}")
        turns = json.loads(source_path.read_text(encoding="utf-8"))["turns"]
        dialogue = "\n".join(
            f"{turn['turn_ref']} | {turn['speaker']} | {turn['text']}" for turn in turns
        )
        first_prompt = (
            "Read the full dialogue. Discover concrete actions or states FIRST OBSERVED in the "
            "latest R02 scenes that could be connected to an earlier choice, proposal, plan, "
            "or stated problem. The later turn must say an action happened or a state is now "
            "true; a future intention, offer, arrangement, prediction, or reminder is NOT an "
            "observed outcome. Keep repeated descriptions of one underlying development "
            "together. Return at most six distinct developments, fewer if warranted. Do not "
            "infer that a meeting/class took place merely because participants arrived or "
            "were ready. Use exact turn_ref strings copied without brackets. Return JSON only: "
            '{"outcomes":[{"id":"O1","actor":"...","observed_state":"...",'
            '"outcome_refs":["..."],"observed_vs_planned":"OBSERVED"}]}. '
            "Every outcome ref must directly support the observed state. No earlier evidence "
            "is requested at this stage.\n\nDIALOGUE:\n" + dialogue
        )
        first = call("discover", root, first_prompt, source_path, actual_sha)
        if first["status"] != "RESPONSE_RECEIVED":
            continue
        try:
            outcomes = json.loads(first["raw_output"])["outcomes"]
        except (ValueError, KeyError, TypeError):
            continue
        targets = [{"id": o.get("id"), "actor": o.get("actor"),
                    "observed_state": o.get("observed_state"),
                    "outcome_refs": o.get("outcome_refs")}
                   for o in outcomes if isinstance(o, dict)]
        second_prompt = (
            "For each independently discovered later outcome below, trace backward to its "
            "SPECIFIC earlier cause or commitment in the full dialogue. Do not substitute "
            "a generic suggestion for a later concrete action, or turn a future plan into a "
            "completed act. Earlier and later evidence must involve the SAME activity, "
            "person, and relevant constraints. Record a distinct relation only if the source "
            "actually supports it. Multiple descriptions of the same change remain one "
            "cluster. For each target, mark FOUND or NOT_FOUND. The earlier_ref must support "
            "the claimed precursor; later_refs must support the observed status; bridge_refs "
            "only if needed. Return exact turn_ref strings copied without brackets. If the "
            "later state was misclassified as observed, mark NOT_FOUND with that reason. "
            "Do not invent absolute dates for UNKNOWN scenes. Return JSON only: "
            '{"relations":[{"id":"O1","status":"FOUND","kind":"plan_to_action",'
            '"actor":"...","earlier_claim":"...","earlier_ref":"...",'
            '"bridge_refs":[],"later_claim":"...","later_refs":["..."],'
            '"reason":"..."}]}. Include one row per target.\n\nDISCOVERED LATER OUTCOMES:\n'
            + json.dumps(targets, ensure_ascii=False) + "\n\nDIALOGUE:\n" + dialogue
        )
        call("backtrack", root, second_prompt, source_path, actual_sha)


if __name__ == "__main__":
    main()
