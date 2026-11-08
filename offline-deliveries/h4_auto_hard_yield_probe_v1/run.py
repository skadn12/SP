"""One-shot autonomous question-yield probe on a frozen dialogue."""
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


def build_prompt(source: dict) -> str:
    dialogue = "\n".join(
        f"[{turn['turn_ref']}] {turn['speaker']}: {turn['text']}"
        for turn in source["turns"]
    )
    return """You are writing hard, evidence-grounded memory questions for a dialogue solver. Return one JSON object with a `candidates` array; no prose. Target up to three multi_select and three ordering questions. If a defensible question cannot be made, output fewer. Do not make single-choice questions.

Select event relationships autonomously from the dialogue. Focus on same-topic near misses: who actually acted, whether an action was proposed, scheduled, attempted or completed, which earlier plan was later executed, and explicit temporal dependencies. Distractors should be plausible because a similar event really occurs elsewhere in the dialogue, while remaining clearly false for this question. Avoid generic negations, absurd alternatives, world-knowledge shortcuts, and copies of a single sentence split into multiple options. Do not imply a meeting, class, purchase, appointment, or result happened when the text only plans it. Distinguish a second occurrence from the first; do not infer chronology from dates that are unknown.

Each multi_select candidate needs 5 options A-E and 2 or 3 gold options. Each ordering candidate needs 4 items I1-I4. Gold is the array of correct keys in chronological order for ordering, and an unordered set of supported keys for multi_select. At least two independent dialogue segments should be necessary to answer each question. For ordering, every pairwise edge must be evidenced by event chronology; sequence of mentions alone is insufficient where a speaker recalls older events. Make items comparable by the question's ordering basis. Avoid duplicating another candidate's core fact set.

For every option or item, include `evidence_refs` as a map from key to exact turn_ref arrays. For multi_select, include `option_verdicts` as a map from key to `supported` or `contradicted`, and add a short `rationale` explaining why each false option conflicts with the dialogue. For ordering, include `ordering_edges` as a list of [earlier_key,later_key,evidence_ref_for_earlier,evidence_ref_for_later] for the three adjacent gold edges. State `requires_later_round` as true only when a R02 fact is needed to answer the question. Required common fields: candidate_id (Q01...), question_type, question, options or items, gold, evidence_refs, requires_later_round. No answer hints in question wording.

DIALOGUE:
""" + dialogue


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    assert source["root_id"] == "cc-train-episode-112812"
    assert len(source["turns"]) == 137
    prompt = build_prompt(source)
    (OUT / "generation_prompt.txt").write_text(prompt, encoding="utf-8")
    profile = provider_profile()
    profile["profile_id"] = "h4_auto_hard_yield_probe_v1"
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
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "model": profile["model"],
        "temperature": 0.35,
        "max_tokens": 6000,
        "attempts": attempts,
        "metadata": {key: response.metadata.get(key) for key in (
            "actual_model", "finish_reason", "usage", "http_status", "completion_status"
        )},
    })
    write_json(OUT / "candidates.json", json.loads(response.raw_text))
    print(json.dumps({"status": "done", "attempts": len(attempts),
                      "candidate_count": len(json.loads(response.raw_text)["candidates"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
