"""Composer prompt for one audited candidate."""

from __future__ import annotations

import json
from typing import Any, Mapping

from scripts.current_method.restart_n5_spell.qa_composition_probe_v1.composer import _atomic_payload


def build_request(*, candidate: Mapping[str, Any], atomic_by_id: Mapping[str, Any]) -> dict[str, Any]:
    source_qids = list(candidate["source_qids"])
    source_qas = [atomic_by_id[qid] for qid in source_qids]
    payload = {
        "candidate_id": candidate["candidate_id"],
        "target": candidate["target"],
        "source_qids": source_qids,
        "source_contributions": candidate["source_contributions"],
        "unique_contribution_closure": candidate["unique_contribution_closure"],
        "atomic_qas": [_atomic_payload(qa) for qa in source_qas],
    }
    instructions = """You are an isolated Composite QA Composer. Return exactly one JSON object.

Use the supplied candidate target, but do not copy any supporting fact or
contribution wording into the question. The question should expose only the
target outcome/decision/change/state/plan/motivation. It must require the
unique contribution of every source QA.

The gold answer must combine every source contribution, using only facts
supported by the supplied atomic QAs. Create exactly four options in one
answer dimension. Every distractor must have the same overall structure and
similar length as the gold, and must mutate exactly one source contribution.
Use plausible alternatives from the supplied transcript when possible; never
invent a clearly unrelated fact. Do not make one option uniquely detailed,
long, qualified, or stylistically distinctive.

Return only these keys:
question, answer_dimension, options, correct_option, distractor_mutations,
is_mechanical_conjunction.

Do not return candidate IDs, source IDs, evidence groups, source contributions,
gold_text, or any other metadata. The program will deterministically attach
those fields from the frozen candidate after this response. The program will
also set `gold_text` equal to `options[correct_option]`.

The question must require the unique contribution of every source QA. Create
exactly four options in one answer dimension. Every distractor must have the
same overall structure and similar length as the gold, and must mutate exactly
one source contribution. Use only facts supported by the supplied atomic QAs.
`distractor_mutations` must map each non-gold option letter to exactly one
source QA id.

Return this shape:
{"question":"...","answer_dimension":"...",
"options":{"A":"...","B":"...","C":"...","D":"..."},
"correct_option":"A",
"distractor_mutations":{"B":"...","C":"...","D":"..."},
"is_mechanical_conjunction":false}"""
    return {
        "request_id": f"relation-aware-composer-{candidate['candidate_id']}",
        "candidate_id": candidate["candidate_id"],
        "payload": payload,
        "prompt": instructions + "\n\nINPUT_JSON:\n" + json.dumps(
            payload, ensure_ascii=False, indent=2, sort_keys=True),
    }
