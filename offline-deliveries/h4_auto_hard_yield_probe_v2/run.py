"""One-shot structured-facts autonomous question-yield probe."""
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


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    facts_path = OUT / "facts.json"
    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    assert source["root_id"] == facts["root_id"] and len(source["turns"]) == 137
    dialogue = "\n".join(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" for t in source["turns"])
    prompt = """Create hard, evidence-grounded memory questions from the FACTS and exact DIALOGUE. Return one JSON object with a `candidates` array and no prose. Generate at most three multi_select and two ordering questions. Return zero ordering questions if the available explicit event-time edges cannot support a unique four-event order. You must choose events, questions, option phrasing, item combinations, and gold yourself; the fact IDs are source aids, not a prescribed question plan.

Question quality: focus on actor, action status, first versus later occurrences, and multi-turn state transitions within the same theme. A wrong multi-select option must be directly contradicted by a cited fact, not merely unmentioned; do not use cheap negations or general knowledge. Include at least two supported and two contradicted options out of A-E. Do not use a sentence split into several options. Every option must be unambiguously true or false from the full dialogue. For ordering, the four events must be linked by at least three explicit temporal relations, with no pair left comparable only by their display position or by unknown scene dates. Do not turn plans into completed events. Do not repeat the same core fact set in another candidate. At least one R02 fact must be necessary for each candidate; otherwise omit it.

Use objects for `options` A-E and `items` I1-I4 (not arrays). Common fields: candidate_id, question_type, question, gold, source_fact_ids, requires_R02. For multi_select add options, option_evidence_refs, option_verdicts and a short option_rationale. For ordering add items, item_evidence_refs and `ordering_edges`, each [earlier_item,later_item,source_fact_relation]. Copy full turn_ref values exactly. Gold must be the exact selected option keys or four item keys in event chronology. Do not include hints such as 'correct' or 'first' in option text.

FACTS:
""" + json.dumps(facts, ensure_ascii=False, indent=2) + "\n\nDIALOGUE:\n" + dialogue
    (OUT / "generation_prompt.txt").write_text(prompt, encoding="utf-8")
    profile = provider_profile()
    profile["profile_id"] = "h4_auto_hard_yield_probe_v2"
    profile["request_parameters"].update(temperature=0.35, max_tokens=6000, enable_thinking=False)
    profile["output_limit"] = 6000
    attempts = []
    client = credential_client(profile, on_http_attempt=lambda payload: attempts.append({
        "model": payload.get("model"), "max_tokens": payload.get("max_tokens")
    }))
    response = client.complete(prompt=prompt, max_tokens=6000, json_object=True)
    (OUT / "raw_response.txt").write_text(response.raw_text, encoding="utf-8")
    write_json(OUT / "generation_call.json", {
        "source": str(SOURCE),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "facts_sha256": hashlib.sha256(facts_path.read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "model": profile["model"], "temperature": 0.35, "max_tokens": 6000,
        "attempts": attempts,
        "metadata": {key: response.metadata.get(key) for key in (
            "actual_model", "finish_reason", "usage", "http_status", "completion_status"
        )},
    })
    candidates = json.loads(response.raw_text)
    write_json(OUT / "candidates.json", candidates)
    print(json.dumps({"status": "done", "attempts": len(attempts),
                      "candidate_count": len(candidates["candidates"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
