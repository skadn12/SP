"""Ordering-only Writer guidance and explicitly synthetic schema example."""

from __future__ import annotations

import json
from typing import Any, Mapping

from .contract import GROUNDED_VERSION, VERSION, ordering_schema, policy_config


def synthetic_example(node_count: int = 4) -> dict[str, Any]:
    policy_config({"version": VERSION, "node_count": node_count})
    sequence = ["I2", "I1", "I4", "I3", "I5", "I6"][:node_count]
    events = {
        "I1": ("Ava", "Ava started the scheduled inspection", "started"),
        "I2": ("Ava", "Ava decided to inspect the equipment", "decided"),
        "I3": ("Ben", "Ben completed the record filing", "completed"),
        "I4": ("Ava", "Ava completed the equipment inspection", "completed"),
        "I5": ("Ben", "Ben started the next scheduled review", "started"),
        "I6": ("Ben", "Ben completed the next scheduled review", "completed"),
    }
    refs = {item: f"EXAMPLE_ONLY_T{sequence.index(item) + 1}" for item in sequence}
    turns = [{"turn_ref": refs[item], "speaker": "Example narrator",
              "text": f"At {9 + index:02d}:00 on the same day, {events[item][1]}."}
             for index, item in enumerate(sequence)]
    nodes = [{"item_id": item, "item_text": events[item][1], "actor": events[item][0],
              "event_state": events[item][2], "evidence_refs": [refs[item]]}
             for item in (f"I{x}" for x in range(1, node_count + 1))]
    relations = [{"before": before, "after": after,
                  "evidence_refs": [refs[before], refs[after]],
                  "temporal_basis": f"The example gives {9 + index:02d}:00 and {10 + index:02d}:00 on the same day."}
                 for index, (before, after) in enumerate(zip(sequence, sequence[1:]))]
    row = {
        "question_type": "ordering", "ordering_contract_version": VERSION,
        "question": "In which chronological order did these specified decisions, starts, and completions occur?",
        "event_nodes": nodes, "temporal_relations": relations,
        "discriminative_variable": "Chronology of the explicitly stated event states",
        "evidence_groups": {"event_evidence": [turn["turn_ref"] for turn in turns]},
        "reasoning_groups_used": ["event_evidence"],
        "hypothesized_reasoning_type": "temporal", "intent_reference": "EXAMPLE_ONLY_INTENT",
        "gold": None,
    }
    return {"expanded": {"turns": turns}, "raw": {"candidates": [row]}}


def ordering_guidance(policy: Mapping[str, Any]) -> list[str]:
    config = policy_config(policy)
    if config["version"] == GROUNDED_VERSION:
        from ..h4_ordering_grounded_options_v2.guidance import ordering_guidance as grounded_guidance
        return grounded_guidance(config)
    if config["version"] != VERSION:
        return []
    n = config["node_count"]
    example = synthetic_example(n)
    return [
        "ORDERING EVENT CONTRACT (ordering only; single-choice and multi-select answers retain their original schema):",
        f"Use ordering_contract_version={VERSION}. Each ordering candidate has exactly {n} event_nodes, with unique item_id I1 through I{n}.",
        "Ask about the temporal order of event occurrences or state changes. Nodes are explicit event instances or state changes, never topics, turn numbers, or positions of mentions.",
        "actor names the actor of the event, not automatically the speaker. event_state must specify the intended, decided, started, completed, or other precise state supported by the evidence; a plan does not mean its intended action occurred. Keep the question and item_text on the same state basis.",
        "A recollection is not a new occurrence; reporting another person's action does not make it the speaker's action. For continuous events specify whether starts, completions, or another stated change are compared. Do not force an order between indistinguishable overlapping events.",
        "Each temporal relation states before, after, nonempty evidence_refs, and nonempty temporal_basis. Use dates, relative times, or sufficiently supported cross-sentence context. Do not add edges from usual workflows, common sense, turn order alone, or intent before/trigger/after group labels.",
        "Provide only supported temporal relations. There is no edge-count quota and no requirement to list all pairs. The program will require the proposed relations to be acyclic and determine one unique order; this is not certification of semantic truth.",
        "Every node and relation uses exact turn_ref strings from the current DIALOGUE. Do not use integer positions, near-matching text, or guessed refs. The program retains exact speaker/text excerpts; do not copy long excerpts into new fields.",
        "Keep discriminative_variable and evidence_groups, and the existing reasoning metadata. evidence_groups must be a nonempty object of lists of exact current turn refs. The program derives candidate-level evidence_refs from nodes in array order then relations, preserving first occurrence; do not output this extra flat list.",
        "For ordering, omit gold or set it to null. Do not output items, final_order, canonical_gold, derived_gold, gold_derivation, ordering_derivation, derivation_status, semantic_support_status, derivation_version, or any final permutation. The program alone projects items and Gold after graph checks. Unknown fields are rejected by the schema.",
        "If the dialogue cannot support an ordering candidate, omit it and let the existing missing-candidate and batch repair rules handle the shortage. Never fabricate a complete chain to meet the requested count.",
        "ORDERING JSON SCHEMA:", json.dumps(ordering_schema(n), ensure_ascii=False, separators=(",", ":")),
        "SYNTHETIC ORDERING FORMAT EXAMPLE ONLY: the following transcript and references belong solely to this example. They are not current-root facts or usable refs. Produce the requested mixed candidate batch from the actual DIALOGUE below.",
        json.dumps(example, ensure_ascii=False, separators=(",", ":")),
    ]
