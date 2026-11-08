"""Strict, isolated Composer prompt for relation-slot QA."""

from __future__ import annotations

import json
from typing import Any, Mapping

from scripts.current_method.restart_n5_spell.qa_composition_probe_v1.composer import (
    _atomic_payload,
)
from scripts.current_method.restart_n5_spell.qa_composition_probe_v1.contracts import AtomicQA


def build_request(
    *, pair: Mapping[str, Any], left: AtomicQA, right: AtomicQA,
    relation_contract: Mapping[str, Any], relation_audit: Mapping[str, Any],
) -> dict[str, Any]:
    source_turns = relation_audit.get("relation_source_turns") or {}
    required_groups = [
        {"source_qa_id": left.qa_id, "turn_refs": list(left.evidence_turn_refs)},
        {"source_qa_id": right.qa_id, "turn_refs": list(right.evidence_turn_refs)},
    ]
    contract = {
        "fact_A": relation_contract.get("fact_A"),
        "trigger": relation_contract.get("trigger"),
        "fact_B": relation_contract.get("fact_B"),
    }
    payload = {
        "pair_id": pair["pair_id"],
        "composition_type": "causal_revision_slot_tuple",
        "atomic_qa_1": _atomic_payload(left),
        "atomic_qa_2": _atomic_payload(right),
        "required_qa_ids": [left.qa_id, right.qa_id],
        "required_evidence_groups": required_groups,
        "required_evidence_turn_refs": sorted({
            *left.evidence_turn_refs, *right.evidence_turn_refs,
        }),
        "relation_contract": contract,
        "relation_evidence_groups": [
            {"group": name, "turn_refs": [str(ref) for ref in source_turns.get(name, [])]}
            for name in ("fact_A", "trigger", "fact_B")
        ],
        "pair_metadata": dict(pair),
        "slot_contract": {
            "initial_state": "fact_A: the original recipe/formulation, including its ratio",
            "trigger": "the observed problem that forced reconsideration",
            "revision": "fact_B: the concrete added/changed ingredient or action",
        },
    }
    instructions = """You are an isolated causal-revision QA Composer. Return exactly one JSON object.

The question must require the complete three-slot relation, in this order:
(1) the original formulation and its ratio, (2) the concrete problem that
prompted reconsideration, and (3) the concrete revision that was made.
The question must explicitly ask for all three, for example: "Starting from
the original ratio, what problem prompted the revision, and what was added?"
Do not ask only why the recipe changed, what the final recipe was, or what
action was taken.

The correct option must contain all three slots. Generate three near-neighbor
distractors of the same length and grammatical form. Every distractor must
also contain all three slots, but mutate exactly one slot:
- one distractor changes only the original ratio/formulation;
- one changes only the trigger/problem;
- one changes only the revision/action.
Do not use unrelated topics, participant roles, stylistic giveaways, "only",
or options that are much shorter/longer. Do not reveal which slot is mutated.
Mutations must be plausible alternatives, but must not be asserted by the
supplied evidence.

Use exactly four unique options in the same `causal_revision_tuple` answer
dimension. Copy the correct option exactly into gold_text and point
correct_option at it. Keep required_qa_ids, required_evidence_groups, and
required_evidence_turn_refs unchanged from the input. Use only supplied facts
for the gold answer; do not add an unsupported change to the gold answer.

Required keys:
pair_id, composition_type, question, answer_dimension, gold_text, options,
correct_option, required_qa_ids, required_evidence_turn_refs,
required_evidence_groups, relation_summary, relation_binding,
is_mechanical_conjunction.

Return this shape:
{"pair_id":"...","composition_type":"causal_revision_slot_tuple",
"question":"...","answer_dimension":"causal_revision_tuple",
"gold_text":"...","options":{"A":"...","B":"...","C":"...","D":"..."},
"correct_option":"A","required_qa_ids":["...","..."],
"required_evidence_turn_refs":["..."],"required_evidence_groups":[
{"source_qa_id":"...","turn_refs":["..."]},
{"source_qa_id":"...","turn_refs":["..."]}],
"relation_summary":"...","relation_binding":{"used_fact_A":true,
"used_trigger":true,"used_fact_B":true},"is_mechanical_conjunction":false}

The relation_binding booleans must all be true."""
    return {
        "request_id": f"targeted-distractor-composer-{pair['pair_id']}",
        "pair_id": pair["pair_id"],
        "composition_type": "causal_revision_slot_tuple",
        "payload": payload,
        "prompt": instructions + "\n\nINPUT_JSON:\n" + json.dumps(
            payload, ensure_ascii=False, indent=2, sort_keys=True),
    }
