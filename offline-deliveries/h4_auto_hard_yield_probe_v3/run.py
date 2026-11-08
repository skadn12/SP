"""One bounded autonomous multi-select generation call with fact scaffolding."""
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

OUT = Path(__file__).resolve().parent
SOURCE = Path("/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_writer_coverage_v1/prepared_run/frozen/cc-train-episode-112812/qa_source.json")
FACTS = OUT.parent / "h4_auto_hard_yield_probe_v2/facts.json"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    facts = json.loads(FACTS.read_text(encoding="utf-8"))
    dialogue = "\n".join(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" for t in source["turns"])
    prompt = """Generate exactly one multi-select memory question from FACTS and DIALOGUE. Return only JSON. Select one coherent story/state chain yourself; do not mix unrelated calendar, budget, boss or sushi facts. The fact table is a human-supplied source aid, not a question plan. You choose the event relationship, option texts and gold. Use at least one R02 fact so the question cannot be answered from old scenes alone.

Use exactly five options A-E, each a conjunction of TWO independently checkable claims about actor, action status or result from the same story chain. Correct options must have both claims directly supported by cited turns. Incorrect options must contain one clearly false clause DIRECTLY CONTRADICTED by a quoted turn; silence or an undocumented event does not count as contradiction. Use two or three correct and two or three incorrect options. Do not use generic negations, far-fetched actions, guesses about missing events, or the same claim repeated in different wording. Avoid wording that gives the answer without reading context. Do not put self-corrections in rationale while leaving option text unchanged.

Output schema: {"candidates":[{"candidate_id":"Q1","question_type":"multi_select","question":"...","options":{"A":"...",...,"E":"..."},"gold":["..."],"option_claims":{"A":["first atomic claim","second atomic claim"],...},"option_evidence_refs":{"A":["exact turn_ref",...],...},"option_verdicts":{"A":"supported|contradicted",...},"option_counterevidence_quotes":{"false_option_key":"[exact turn_ref] exact quote",...},"selected_fact_ids":["F..",...],"requires_R02":true}]}. For every false option, include the exact source quote that contradicts its false clause. Do not fill a slot with an ungrounded option just to reach five; if five valid conjunctive options cannot be made, return {"candidates":[]}.

Before finalizing internally, reread the exact option strings and check each clause against the source. The gold must match the option text itself, not an intended rewrite.

FACTS:
""" + json.dumps(facts, ensure_ascii=False, indent=2) + "\n\nDIALOGUE:\n" + dialogue
    (OUT / "generation_prompt.txt").write_text(prompt, encoding="utf-8")
    profile = provider_profile()
    profile["profile_id"] = "h4_auto_hard_yield_probe_v3"
    profile["request_parameters"].update(temperature=0.25, max_tokens=4000, enable_thinking=False)
    profile["output_limit"] = 4000
    attempts = []
    client = credential_client(profile, on_http_attempt=lambda payload: attempts.append({
        "model": payload.get("model"), "max_tokens": payload.get("max_tokens")
    }))
    response = client.complete(prompt=prompt, max_tokens=4000, json_object=True)
    (OUT / "raw_response.txt").write_text(response.raw_text, encoding="utf-8")
    write_json(OUT / "generation_call.json", {
        "source": str(SOURCE), "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "facts": str(FACTS), "facts_sha256": hashlib.sha256(FACTS.read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "model": profile["model"], "temperature": 0.25, "max_tokens": 4000,
        "attempts": attempts,
        "metadata": {key: response.metadata.get(key) for key in (
            "actual_model", "finish_reason", "usage", "http_status", "completion_status"
        )},
    })
    parsed = json.loads(response.raw_text)
    write_json(OUT / "candidates.json", parsed)
    refs = {t["turn_ref"] for t in source["turns"]}
    checks = []
    for candidate in parsed.get("candidates", []):
        options = candidate.get("options", {})
        evid = candidate.get("option_evidence_refs", {})
        checks.append({
            "candidate_id": candidate.get("candidate_id"),
            "five_option_keys": set(options) == set("ABCDE"),
            "two_claims_each": all(len(candidate.get("option_claims", {}).get(k, [])) == 2 for k in "ABCDE"),
            "all_citations_exist": all(ref in refs for values in evid.values() for ref in values),
            "gold_keys_exist": set(candidate.get("gold", [])).issubset(options),
            "source_check_type": "deterministic_shape_and_ref_membership_only; semantic entailment manually audited separately",
        })
    write_json(OUT / "source_evidence_self_check.json", checks)
    print(json.dumps({"status": "done", "attempts": len(attempts),
                      "candidate_count": len(parsed.get("candidates", []))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
