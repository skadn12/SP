"""One-shot strong-solver check for the reviewed appointment binding candidate."""

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
EXPECTED_SOURCE_SHA = "b163bd9ceed0ec0cf4745e5c414fb5d7615788672c879c6283669aad78395113"
MANIFEST = HERE / "m0_balanced/run/manifest.json"
OUT = HERE / "strong_solver_results.json"


def main() -> None:
    if OUT.exists():
        raise RuntimeError("strong_solver_results_exist")
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED_SOURCE_SHA
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    questions = json.loads((HERE / "candidates.json").read_text(encoding="utf-8"))["candidates"]
    candidate = questions[0]
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    variants = {}
    for request in manifest["requests"]:
        if request["candidate_id"] == candidate["candidate_id"]:
            variants[request["variant"]["variant_id"]] = request["variant"]
    assert len(variants) == 10
    dialogue = "\n".join(
        f"[{turn['turn_ref']}] {turn['speaker']}: {turn['text']}"
        for turn in source["turns"]
    )
    profile = provider_profile()
    profile["profile_id"] = "h4_source_locked_hard_qa_r03_strong_solver_v1"
    profile["request_parameters"].update(temperature=0, max_tokens=160, enable_thinking=False)
    profile["output_limit"] = 160
    attempts = []
    client = credential_client(profile, on_http_attempt=lambda payload: attempts.append({
        "model": payload.get("model"), "max_tokens": payload.get("max_tokens")
    }))
    rows = []
    try:
        for variant_id, variant in sorted(variants.items()):
            options = "\n".join(f"{key}. {value}" for key, value in variant["options"].items())
            prompt = (
                "Answer the multi-select question using only the recorded dialogue. "
                "Check each entire option independently; one wrong clause makes the option false. "
                "Return only JSON of the form {\"answer\":[\"A\",\"C\"]}; do not explain.\n\n"
                f"DIALOGUE:\n{dialogue}\n\nQUESTION:\n{candidate['question']}\n\nOPTIONS:\n{options}"
            )
            response = client.complete(prompt=prompt, max_tokens=160, json_object=True)
            raw = response.raw_text
            parsed = json.loads(raw)
            answer = parsed.get("answer")
            valid = isinstance(answer, list) and len(answer) == len(set(answer)) and all(
                isinstance(key, str) and key in variant["options"] for key in answer
            )
            canonical = sorted(variant["display_to_canonical"][key] for key in answer) if valid else []
            rows.append({
                "variant_id": variant_id,
                "raw_output": raw,
                "valid": valid,
                "canonical_prediction": canonical,
                "correct": valid and canonical == sorted(candidate["gold"]),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "metadata": {key: response.metadata.get(key) for key in (
                    "actual_model", "finish_reason", "usage", "http_status", "completion_status"
                )},
            })
            print(json.dumps({"variant_id": variant_id, "correct": rows[-1]["correct"],
                              "canonical_prediction": canonical}), flush=True)
    finally:
        OUT.write_text(json.dumps({
            "candidate_id": candidate["candidate_id"],
            "source_sha256": EXPECTED_SOURCE_SHA,
            "model": profile["model"],
            "temperature": 0,
            "gold": candidate["gold"],
            "attempt_count": len(attempts),
            "rows": rows,
            "training_promotion": False,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
