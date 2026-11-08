"""Bounded strong-solver diagnostic for the reviewed typed appointment item."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
WORKTREE = Path("/root/autodl-tmp/gxy-selfplay/worktrees/h4-writer-coverage-v1")
sys.path.insert(0, str(WORKTREE))
from scripts.current_method.restart_n5_spell.h4_memory_light_gate_siliconflow_v1.run import (  # noqa: E402
    credential_client, provider_profile,
)


SOURCE = Path("/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_memory_light_gate_siliconflow_qa_v1/prepared_run/frozen/qa_source.json")
OUT = HERE / "strong_pilot_v4.json"
LETTERS = "ABCDE"


def main() -> None:
    if OUT.exists():
        raise RuntimeError("output_exists")
    source_bytes = SOURCE.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    candidate = json.loads((HERE / "candidate_v3.json").read_text(encoding="utf-8"))
    if source_sha != candidate["frozen_source_sha256"]:
        raise RuntimeError("source_sha_mismatch")
    source = json.loads(source_bytes)
    dialogue = "\n".join(
        f"[{turn['turn_ref']}] {turn['speaker']}: {turn['text']}"
        for turn in source["turns"]
    )
    orders = random.Random(20261003).sample(list(itertools.permutations(LETTERS)), 5)
    profile = provider_profile()
    profile["profile_id"] = "h4_typed_appointment_r03_strong_pilot_v4"
    profile["request_parameters"].update(temperature=0, max_tokens=160, enable_thinking=False)
    profile["output_limit"] = 160
    attempts: list[dict[str, object]] = []
    client = credential_client(profile, on_http_attempt=lambda payload: attempts.append({
        "model": payload.get("model"), "max_tokens": payload.get("max_tokens")
    }))
    rows = []
    try:
        for index, order in enumerate(orders):
            displayed = {label: candidate["options"][canonical]
                         for label, canonical in zip(LETTERS, order)}
            mapping = dict(zip(LETTERS, order))
            prompt = (
                "Answer the multi-select question using only the recorded dialogue. "
                "Check each entire option independently; one wrong clause makes it false. "
                "Return only a JSON object with an answer array of selected option labels. "
                "Select as many options as the evidence supports; do not explain.\n\n"
                f"DIALOGUE:\n{dialogue}\n\nQUESTION:\n{candidate['question']}\n\nOPTIONS:\n"
                + "\n".join(f"{key}. {text}" for key, text in displayed.items())
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
                "variant_id": f"perm_{index:02d}", "display_to_canonical": mapping,
                "raw_output": raw, "valid": valid, "canonical_prediction": canonical,
                "correct": valid and canonical == sorted(candidate["gold"]),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "metadata": {key: response.metadata.get(key) for key in (
                    "actual_model", "finish_reason", "usage", "http_status", "completion_status"
                )},
            }
            rows.append(row)
            print(json.dumps({"variant_id": row["variant_id"], "correct": row["correct"],
                              "canonical_prediction": canonical}), flush=True)
    finally:
        OUT.write_text(json.dumps({
            "candidate_id": candidate["candidate_id"], "source_sha256": source_sha,
            "candidate_sha256": hashlib.sha256((HERE / "candidate_v3.json").read_bytes()).hexdigest(),
            "model": profile["model"], "temperature": 0, "gold": candidate["gold"],
            "attempt_count": len(attempts), "rows": rows, "training_promotion": False,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
