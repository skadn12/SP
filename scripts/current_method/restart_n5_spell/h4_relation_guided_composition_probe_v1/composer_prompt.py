"""Relation-guided Composer request wrapper; no production Composer changes."""

from __future__ import annotations

import json
from typing import Any, Mapping

from scripts.current_method.restart_n5_spell.qa_composition_probe_v1.composer import _atomic_payload
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
    relation_groups = [
        {"group": name, "turn_refs": [str(ref) for ref in source_turns.get(name, [])]}
        for name in ("fact_A", "trigger", "fact_B")
    ]
    payload = {
        "pair_id": pair["pair_id"],
        "composition_type": "state_change",
        "atomic_qa_1": _atomic_payload(left),
        "atomic_qa_2": _atomic_payload(right),
        "required_qa_ids": [left.qa_id, right.qa_id],
        "required_evidence_groups": required_groups,
        "required_evidence_turn_refs": sorted(set(left.evidence_turn_refs + right.evidence_turn_refs)),
        "relation_contract": {
            "relation_type": relation_contract.get("relation_type"),
            "relation_id": relation_contract.get("relation_id"),
            "fact_A": relation_contract.get("fact_A"),
            "trigger": relation_contract.get("trigger"),
            "fact_B": relation_contract.get("fact_B"),
            "resolution": relation_contract.get("resolution"),
        },
        "relation_evidence_groups": relation_groups,
        "pair_metadata": dict(pair),
    }
    instructions = """You are an isolated Relation-guided Composite Question Composer. Return exactly one JSON object.

The relation is a causal revision. The composite question must require all three
parts of the supplied relation: the initial state in fact_A, the trigger/cause,
and the revised state in fact_B. It must ask why/how the change occurred, not
only what the final state was. It must not be a mechanical conjunction and must
not copy either atomic question or atomic gold answer.

Use exactly four unique options in one answer dimension. The correct option must
state the initial state, trigger, and revision as one causal relation. Keep the
required_qa_ids and required_evidence_groups unchanged. Include all required
output keys: pair_id, composition_type, question, answer_dimension, gold_text,
options, correct_option, required_qa_ids, required_evidence_turn_refs,
required_evidence_groups, relation_summary, relation_binding, is_mechanical_conjunction.

Necessity constraint: each atomic QA must contribute a unique relation role; do
not compose a pair when one atomic already covers fact_A, trigger, and fact_B.
The correct option text must be copied exactly into gold_text, and
correct_option must point to that exact option. Make all four options complete
causal explanations in the same `relation_statement` dimension. Do not use
transition-only distractors such as "changed from X to Y" or options that omit
the trigger/revision relation.

The output shape is strict: answer_dimension must be one scalar string such as
"cause" or "state"; options must be a separate object with exactly the four
keys A, B, C, and D. Never put the options object inside answer_dimension.
required_evidence_groups must be a separate array with exactly two objects,
each object containing source_qa_id and turn_refs, copied unchanged from the
input. required_evidence_turn_refs must be the union of those two groups.

Return this exact shape (replace values, preserve keys):
{"pair_id":"RGPAIR0001","composition_type":"state_change","question":"...",
"answer_dimension":"cause","gold_text":"...","options":{"A":"...","B":"...","C":"...","D":"..."},
"correct_option":"A","required_qa_ids":["...","..."],
"required_evidence_turn_refs":["..."],"required_evidence_groups":[
{"source_qa_id":"...","turn_refs":["..."]},{"source_qa_id":"...","turn_refs":["..."]}],
"relation_summary":"...","relation_binding":{"used_fact_A":true,"used_trigger":true,"used_fact_B":true},
"is_mechanical_conjunction":false}

The relation_binding object must contain booleans used_fact_A, used_trigger, and
used_fact_B, all true."""
    return {
        "request_id": f"relation-guided-composer-{pair['pair_id']}",
        "pair_id": pair["pair_id"],
        "composition_type": "state_change",
        "payload": payload,
        "prompt": instructions + "\n\nINPUT_JSON:\n" + json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
    }
