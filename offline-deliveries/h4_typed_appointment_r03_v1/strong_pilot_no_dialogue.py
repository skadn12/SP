"""Paired no-dialogue diagnostic for the reviewed R03 multi-select item."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
WORKTREE = Path("/root/autodl-tmp/gxy-selfplay/worktrees/h4-writer-coverage-v1")
sys.path.insert(0, str(WORKTREE))
from scripts.current_method.restart_n5_spell.h4_memory_light_gate_siliconflow_v1.run import (  # noqa: E402
    credential_client, provider_profile,
)


SOURCE = Path("/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_memory_light_gate_siliconflow_qa_v1/prepared_run/frozen/qa_source.json")
REFERENCE = HERE / "strong_pilot_v4.json"
OUT = HERE / "strong_pilot_no_dialogue.json"
LETTERS = "ABCDE"


def main() -> None:
    if OUT.exists():
        raise RuntimeError("output_exists")
    candidate_bytes = (HERE / "candidate_v3.json").read_bytes()
    candidate = json.loads(candidate_bytes)
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    source_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    candidate_sha = hashlib.sha256(candidate_bytes).hexdigest()
    if (source_sha != candidate["frozen_source_sha256"]
            or source_sha != reference["source_sha256"]
            or candidate_sha != reference["candidate_sha256"]
            or candidate["candidate_id"] != reference["candidate_id"]
            or len(reference["rows"]) != 5):
        raise RuntimeError("frozen_input_mismatch")

    profile = provider_profile()
    profile["profile_id"] = "h4_typed_appointment_r03_strong_no_dialogue"
    profile["request_parameters"].update(temperature=0, max_tokens=160, enable_thinking=False)
    profile["output_limit"] = 160
    if profile["model"] != reference["model"] or profile["request_parameters"]["temperature"] != reference["temperature"]:
        raise RuntimeError("reference_parameter_mismatch")

    attempts: list[dict[str, object]] = []
    client = credential_client(profile, on_http_attempt=lambda payload: attempts.append({
        "model": payload.get("model"), "max_tokens": payload.get("max_tokens")
    }))
    rows = []
    try:
        for index, prior in enumerate(reference["rows"]):
            mapping = prior["display_to_canonical"]
            if prior["variant_id"] != f"perm_{index:02d}" or set(mapping) != set(LETTERS) or set(mapping.values()) != set(LETTERS):
                raise RuntimeError("reference_permutation_mismatch")
            displayed = {label: candidate["options"][mapping[label]] for label in LETTERS}
            prompt = (
                "Answer the multi-select question using only the question and options. "
                "Check each entire option independently; one wrong clause makes it false. "
                "Return only a JSON object with an answer array of selected option labels. "
                "Select as many options as the evidence supports; do not explain.\n\n"
                f"QUESTION:\n{candidate['question']}\n\nOPTIONS:\n"
                + "\n".join(f"{key}. {option}" for key, option in displayed.items())
            )
            response = client.complete(prompt=prompt, max_tokens=160, json_object=True)
            raw = response.raw_text
            try:
                answer = json.loads(raw).get("answer")
            except (ValueError, AttributeError):
                answer = None
            valid = isinstance(answer, list) and len(answer) == len(set(answer)) and all(
                isinstance(key, str) and key in displayed for key in answer
            )
            canonical = sorted(mapping[key] for key in answer) if valid else []
            row = {
                "variant_id": prior["variant_id"], "display_to_canonical": mapping,
                "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "raw_output": raw, "valid": valid, "canonical_prediction": canonical,
                "correct": valid and canonical == sorted(candidate["gold"]),
                "metadata": {key: response.metadata.get(key) for key in (
                    "actual_model", "finish_reason", "usage", "http_status", "completion_status",
                    "raw_response"
                )},
            }
            rows.append(row)
            print(json.dumps({"variant_id": row["variant_id"], "correct": row["correct"],
                              "canonical_prediction": canonical}), flush=True)
    finally:
        OUT.write_text(json.dumps({
            "candidate_id": candidate["candidate_id"], "source_sha256": source_sha,
            "candidate_sha256": candidate_sha, "reference": REFERENCE.name,
            "condition": "question_and_options_only", "model": profile["model"],
            "request_parameters": profile["request_parameters"], "gold": candidate["gold"],
            "attempt_count": len(attempts), "rows": rows, "training_promotion": False,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
