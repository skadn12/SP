from __future__ import annotations

import json
from typing import Any, Mapping

from ..core import sha256_bytes


PROMPT_VERSION = "h4_qa_mining_contract_guided_candidate_generator_v1"
ROLES = ("before_state", "trigger", "after_state")


def _contract_evidence(contract: Mapping[str, Any], expanded: Mapping[str, Any]) -> list[str]:
    text_by_ref = {
        str(row.get("turn_ref")): f"{row.get('speaker')}: {row.get('text')}"
        for row in expanded.get("turns") or []
    }
    lines: list[str] = []
    groups = contract.get("required_evidence_groups") or {}
    for role in ROLES:
        refs = [str(ref) for ref in groups.get(role) or []]
        lines.append(f"{role}: {json.dumps(refs, ensure_ascii=False)}")
        for ref in refs:
            lines.append(f"  {ref} | {text_by_ref.get(ref, '[UNKNOWN REF]')}")
    return lines


def build_prompt(expanded: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    groups = {role: [str(ref) for ref in (contract.get("required_evidence_groups") or {}).get(role) or []]
              for role in ROLES}
    schema = {
        "candidates": [{
            "candidate_id": "I001-Q01",
            "question_type": "single_choice",
            "question": "Why did the later state follow from the earlier state and trigger?",
            "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
            "correct_option": "A",
            "gold": [],
            "evidence_refs": sum(groups.values(), []),
            "evidence_groups": groups,
            "required_reasoning": list(ROLES),
            "discriminative_variable": str(contract.get("discriminative_variable") or ""),
            "distractor_strategy": "same_topic_wrong_relation",
        }]
    }
    dialogue = "\n".join(
        f"{row.get('turn_ref')} | {row.get('speaker')} | {row.get('text')}"
        for row in expanded.get("turns") or []
    )
    prompt = "\n".join([
        "ROLE: H4 contract-guided QA candidate generator.",
        "Generate exactly four candidates for this one fixed intent: 2 single_choice, 1 multi_select, 1 ordering.",
        "The reasoning contract is an exact binding contract, not a suggestion.",
        "Every question and answer must require the before_state + trigger + after_state chain.",
        "Do not generate endpoint-only recall, topic-only, or unrelated distractor questions.",
        "For every candidate, copy evidence_groups and evidence_refs EXACTLY from the contract ledger below.",
        "Do not substitute, add, or remove turn refs. required_reasoning MUST be exactly [before_state, trigger, after_state].",
        "discriminative_variable MUST exactly equal the contract value.",
        "Distractors should be same-topic wrong relations, not random facts.",
        "single_choice uses options A-D and correct_option; multi_select uses options A-E and list gold; ordering uses items I1-I4 and list gold.",
        "Every row MUST include a non-empty question string.",
        "Return JSON only with exactly one top-level key candidates.",
        "EXACT CONTRACT:", json.dumps({
            "intent_id": contract.get("intent_id"),
            "intent_type": contract.get("intent_type"),
            "discriminative_variable": contract.get("discriminative_variable"),
            "required_evidence_groups": groups,
            "required_reasoning": list(ROLES),
        }, ensure_ascii=False, indent=2),
        "CONTRACT EVIDENCE LEDGER:", *_contract_evidence(contract, expanded),
        "OUTPUT SCHEMA:", json.dumps(schema, ensure_ascii=False, indent=2),
        "AUTHORITATIVE DIALOGUE:", dialogue,
    ])
    return {"prompt_version": PROMPT_VERSION, "prompt": prompt,
            "prompt_sha256": sha256_bytes(prompt.encode()),
            "candidate_minimums": {"single_choice": 2, "multi_select": 1, "ordering": 1},
            "exact_contract_groups": groups}


def build_repair_prompt(
    expanded: Mapping[str, Any], contract: Mapping[str, Any], raw: Mapping[str, Any] | None,
) -> dict[str, Any]:
    base = build_prompt(expanded, contract)
    prompt = "\n".join([
        "ROLE: H4 contract-guided candidate schema repair.",
        "Repair only the supplied candidate batch for the exact contract below.",
        "Return exactly 2 single_choice + 1 multi_select + 1 ordering.",
        "Preserve valid question/options/gold content where possible.",
        "Every row must contain all required fields. Copy evidence_groups, evidence_refs, required_reasoning, and discriminative_variable exactly from the contract; no other refs are allowed.",
        "Return JSON only with top-level key candidates.",
        "CONTRACT-GUIDED BASE PROMPT:", base["prompt"],
        "SUPPLIED BATCH:", json.dumps(raw or {}, ensure_ascii=False, indent=2),
    ])
    return {**base, "prompt_version": PROMPT_VERSION + "_repair",
            "prompt": prompt, "prompt_sha256": sha256_bytes(prompt.encode())}


def build_ordering_repair_prompt(
    expanded: Mapping[str, Any], contract: Mapping[str, Any],
    invalid_ordering: Mapping[str, Any] | None,
) -> dict[str, Any]:
    base = build_prompt(expanded, contract)
    prompt = "\n".join([
        "ROLE: H4 contract-guided ordering candidate repair.",
        "Return exactly one JSON object with key candidate.",
        "The candidate MUST be question_type ordering with exactly four distinct items I1,I2,I3,I4 and a four-ID gold permutation.",
        "Make four grounded state/event descriptions by distinguishing sub-events or consequences in the supplied contract evidence; do not invent unrelated facts.",
        "Copy evidence_groups, evidence_refs, required_reasoning, and discriminative_variable exactly from the contract.",
        "The question must require the before_state -> trigger -> after_state relation.",
        "CONTRACT-GUIDED BASE:", base["prompt"],
        "INVALID ORDERING TO REPAIR:", json.dumps(invalid_ordering or {}, ensure_ascii=False, indent=2),
    ])
    return {**base, "prompt_version": PROMPT_VERSION + "_ordering_repair",
            "prompt": prompt, "prompt_sha256": sha256_bytes(prompt.encode())}
