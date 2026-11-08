"""One-call event selection with quoted evidence and one direction field."""
from __future__ import annotations

import json

from .contract import VERSION, ordering_schema
from ..h4_r02_ordering_temporal_contract_v1.contract import policy_config


def synthetic_example():
    turns = [
        {"turn_ref": "EXAMPLE_ONLY_T1", "speaker": "Ava",
         "text": "On June 2, 2026, I registered for the workshop."},
        {"turn_ref": "EXAMPLE_ONLY_T2", "speaker": "Ben",
         "text": "On June 4, 2026, I collected the exhibition tickets."},
        {"turn_ref": "EXAMPLE_ONLY_T3", "speaker": "Ava",
         "text": "On June 5, 2026, I finished the garden survey."},
        {"turn_ref": "EXAMPLE_ONLY_T4", "speaker": "Ben",
         "text": "On June 7, 2026, I booked the restaurant."},
    ]
    chronological_ids = ["I2", "I4", "I1", "I3"]
    claims = ["Ava registered for the workshop", "Ben collected the exhibition tickets",
              "Ava finished the garden survey", "Ben booked the restaurant"]
    by_id = dict(zip(chronological_ids, turns))
    nodes = [{"item_id": item, "item_text": claims[chronological_ids.index(item)],
              "actor": by_id[item]["speaker"], "event_state": "completed",
              "evidence_refs": [by_id[item]["turn_ref"]],
              "evidence_spans": [{"turn_ref": by_id[item]["turn_ref"], "quote": by_id[item]["text"]}]}
             for item in ("I1", "I2", "I3", "I4")]
    relations = [{"ordered_pair": [earlier, later],
                  "evidence_refs": [by_id[earlier]["turn_ref"], by_id[later]["turn_ref"]],
                  "evidence_spans": [{"turn_ref": by_id[item]["turn_ref"], "quote": by_id[item]["text"]}
                                     for item in (earlier, later)]}
                 for earlier, later in zip(chronological_ids, chronological_ids[1:])]
    row = {"question_type": "ordering", "ordering_contract_version": VERSION,
           "question": "In which order did these four actions occur in June of the same year?",
           "event_nodes": nodes, "temporal_relations": relations,
           "discriminative_variable": "Event occurrence dates, across independent activities",
           "evidence_groups": {"events": [turn["turn_ref"] for turn in turns]}, "gold": None}
    return {"expanded": {"turns": turns}, "raw": {"candidates": [row]}}


def ordering_guidance(policy):
    config = policy_config(policy)
    if config["version"] != VERSION:
        raise ValueError("GROUNDED_ORDERING_POLICY_REQUIRED")
    return [
        "GROUNDED ORDERING OPTIONS V2 (ordering only; keep the original SC/MS schema):",
        f"Use ordering_contract_version={VERSION} and exactly four event_nodes with item_id I1-I4 when a supported ordering is available.",
        "Within this same response, select source-supported event instances before writing the four options. An option needs a specific actor, action, and precise state. Keep actor, event_state, item_text, and the question consistent with those source facts. A plan is not a completed action. A question about another person's worry is not evidence that the speaker is worried.",
        "Bind every event and temporal relation to evidence_spans: exact {turn_ref,quote} substrings from DIALOGUE. evidence_refs must equal the set of refs represented by those spans. Use multiple spans and multiple refs whenever pronouns, reported speech, dates, or event identity require surrounding context. A short matching word is not adequate evidence for a whole event.",
        "Cite the sentences that establish the stated actor and state, not merely another sentence about the same topic. Do not promote ready-to-send into sent, or infer a previous submission solely because feedback is mentioned. Repeated recollections of one occurrence do not create extra events.",
        "Compare event occurrence or explicitly stated state-change times, not the order in which events are mentioned. Independent events can be ordered using supported dates or relative times; they need not be stages of one process. No edge follows merely from a usual workflow, intent before/trigger/after labels, or turn order.",
        "For each supported relation output only ordered_pair=[earlier_item_id,later_item_id], evidence_refs, and evidence_spans. ordered_pair[0] occurred earlier than ordered_pair[1]. Do not output before, after, temporal_basis, or a separate natural-language direction. The program derives those display fields and Gold from this one direction representation.",
        "A conversation-date header dates the conversation, not every past, future, remembered, or planned event mentioned in it. Optional date_anchors=[{segment_id,conversation_date}] must refer to the segment of at least one cited ref. They can anchor explicit today/yesterday/next-week expressions with the quoted context; do not copy the full date table or substitute a header for event-time evidence.",
        "Provide only supported comparisons. Overlapping or unknown event times must not be forced into a unique chain. If four distinct supported options cannot be ordered uniquely, omit ordering and declare its INSUFFICIENT_EVIDENCE abstention; this is allowed without a quota-filling repair. Keep supported SC/MS candidates.",
        "Keep discriminative_variable, evidence_groups, and existing reasoning metadata. Do not output items, final_order, canonical_gold, derived_gold, gold_derivation, ordering_derivation, derivation_status, semantic_support_status, derivation_version, or a flat candidate-level evidence_refs. Omit gold or set it to null.",
        "The program checks reference/quote authenticity and graph structure. Those checks do not certify semantic support, actor identity, state interpretation, or event chronology.",
        "ORDERING JSON SCHEMA:", json.dumps(ordering_schema(), ensure_ascii=False, separators=(",", ":")),
        "SYNTHETIC ORDERING FORMAT EXAMPLE ONLY: these independent events, people, and refs are not current-root evidence. All dates in this example share the same year.",
        json.dumps(synthetic_example(), ensure_ascii=False, separators=(",", ":")),
    ]
