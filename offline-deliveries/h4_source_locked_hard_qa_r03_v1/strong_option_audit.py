"""Ask the strong solver to cite evidence for each option of one reviewed item."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

WORKTREE = Path("/root/autodl-tmp/gxy-selfplay/worktrees/h4-writer-coverage-v1")
sys.path.insert(0, str(WORKTREE))
from scripts.current_method.restart_n5_spell.h4_memory_light_gate_siliconflow_v1.run import (  # noqa: E402
    credential_client, provider_profile,
)

HERE = Path(__file__).resolve().parent
SOURCE = Path("/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_memory_light_gate_siliconflow_qa_v1/prepared_run/frozen/qa_source.json")
EXPECTED_SHA = "b163bd9ceed0ec0cf4745e5c414fb5d7615788672c879c6283669aad78395113"
OUT = HERE / "strong_option_audit.json"


def main() -> None:
    if OUT.exists():
        raise RuntimeError("audit_exists")
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED_SHA
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    candidate = json.loads((HERE / "candidates.json").read_text(encoding="utf-8"))["candidates"][0]
    dialogue = "\n".join(
        f"[{turn['turn_ref']}] {turn['speaker']}: {turn['text']}"
        for turn in source["turns"]
    )
    options = "\n".join(f"{key}. {value}" for key, value in candidate["options"].items())
    prompt = (
        "Audit each entire option separately against the dialogue. For each option, "
        "return supported true or false, exact turn refs, and a brief quote identifying "
        "the person who first suggested each appointment. Do not assume a fixed number "
        "of true options. Return JSON only: {\"options\":{\"A\":{\"supported\":true,"
        "\"refs\":[\"...\"],\"reason\":\"...\"},...}}.\n\n"
        f"DIALOGUE:\n{dialogue}\n\nQUESTION:\n{candidate['question']}\n\nOPTIONS:\n{options}"
    )
    profile = provider_profile()
    profile["profile_id"] = "h4_source_locked_hard_qa_r03_strong_option_audit_v1"
    profile["request_parameters"].update(temperature=0, max_tokens=1600, enable_thinking=False)
    profile["output_limit"] = 1600
    attempts = []
    client = credential_client(profile, on_http_attempt=lambda payload: attempts.append({
        "model": payload.get("model"), "max_tokens": payload.get("max_tokens")
    }))
    response = client.complete(prompt=prompt, max_tokens=1600, json_object=True)
    result = {
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "attempt_count": len(attempts), "raw_output": response.raw_text,
        "metadata": {key: response.metadata.get(key) for key in (
            "actual_model", "finish_reason", "usage", "http_status", "completion_status"
        )},
        "training_promotion": False,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(response.raw_text)


if __name__ == "__main__":
    main()
