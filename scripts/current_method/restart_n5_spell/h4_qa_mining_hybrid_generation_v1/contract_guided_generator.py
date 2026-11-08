from __future__ import annotations

import json
from typing import Any, Mapping

from ..core import sha256_bytes
from .hybrid_contract import ROLES, contract_summary, expected_counts


def _dialogue(expanded: Mapping[str, Any]) -> str:
    from ..h4_writer_partial_memory_v1.sources import render_dialogue
    return render_dialogue(expanded)


def build_prompt(expanded: Mapping[str, Any], contract: Mapping[str, Any], *,
                 intent_guidance: bool = True,
                 ordering_policy: Mapping[str, Any] | None = None) -> dict[str, Any]:
    from ..h4_r02_ordering_temporal_contract_v1.contract import GROUNDED_VERSION, VERSION, policy_config
    from ..h4_r02_ordering_temporal_contract_v1.guidance import ordering_guidance
    from ..h4_writer_partial_memory_v1.sources import (
        MEMORY_ONLY, TRANSCRIPT_PLUS_MEMORY, memory_navigation_block,
        memory_ordering_guidance, validate_memory_contract, validate_source_envelope,
    )
    mode = validate_source_envelope(expanded)
    memory_only = mode == MEMORY_ONLY
    policy = policy_config(ordering_policy)
    grounded = policy["version"] == GROUNDED_VERSION
    temporal = policy["version"] != "legacy"
    if grounded and memory_only:
        raise ValueError("GROUNDED_ORDERING_REQUIRES_TRANSCRIPT")
    if memory_only:
        validate_memory_contract(contract)
        if not temporal:
            raise ValueError("WRITER_MEMORY_ONLY_REQUIRES_TEMPORAL_CONTRACT")
    # The opaque slot identity is the only Miner field consumed when guidance
    # is disabled. In particular OUTPUT SHAPE must not restore the old target.
    summary = (contract_summary(contract) if intent_guidance and not memory_only else {
        "intent_id": str(contract.get("intent_id") or ""),
        "discriminative_variable": "...",
    })
    schema = {"candidates": [{"question_type": "single_choice", "question": "...", "options": {"A": "...", "B": "...", "C": "...", "D": "..."}, "correct_option": "A", "gold": ["A"], "evidence_refs": [], "evidence_groups": {}, "reasoning_groups_used": ["before_state", "trigger"], "intent_reference": summary["intent_id"], "discriminative_variable": summary["discriminative_variable"], "hypothesized_reasoning_type": "causal", "distractor_strategy": "same_topic_wrong_relation"}]}
    if memory_only or grounded:
        schema["abstentions"] = [{"question_type": "ordering", "status": "INSUFFICIENT_EVIDENCE", "reason": "..."}]
    prompt = "\n".join([
        "ROLE: H4 hybrid QA mining contract-guided explorer.",
        ("Generate 2 single_choice and 1 multi_select candidates; target at most 1 supported ordering candidate. If ordering lacks sufficient evidence, explicitly abstain instead of filling its quota."
         if grounded else "Target 2 single_choice, 1 multi_select, and 1 ordering candidate for this Memory-only mixed batch. These are per-type upper limits."
         if memory_only else "Generate exactly 2 single_choice, 1 multi_select, and 1 ordering candidate for this intent."
         if intent_guidance else
         "Generate exactly 2 single_choice, 1 multi_select, and 1 ordering candidate for this batch."),
        ("Choose evidence-supported questions from the frozen Memory below. No transcript-derived intent or reasoning chain is provided."
         if memory_only else "Use the contract as guidance, not as a hard semantic filter: you may explore subsets or alternative evidence groups."
         if intent_guidance else
         "Choose a transcript-supported focus for this batch from the full dialogue and generate candidates using the same requested question types and counts."),
        ("Single-choice and multi-select candidates must be structurally grounded: evidence_refs must be exact memory_ref values and evidence_groups must be a subset of them. Ordering uses the same event contract below with Memory references."
         if memory_only else "Single-choice and multi-select candidates must be structurally grounded: evidence_refs must be real turn refs and evidence_groups must be a subset of them. Ordering uses the event contract below."
         if temporal else "Every candidate must be structurally grounded: evidence_refs must be real turn refs and evidence_groups must be a subset of them."),
        "Do not claim semantic validation. Record reasoning_groups_used and hypothesized_reasoning_type as metadata.",
        ("Single options are A-D; multi-select options A-E with list gold; ordering declares event_nodes and temporal_relations without a final answer."
         if temporal else "Single options are A-D; multi-select options A-E with list gold; ordering items I1-I4 with a permutation gold."),
        ("Return JSON only with candidates and optional abstentions. Only ordering may explicitly abstain: omit its candidate and use exactly {question_type:ordering,status:INSUFFICIENT_EVIDENCE,reason:<nonempty reason>}. An ordering abstention does not trigger repair and remains fixed if SC/MS needs repair. Structural errors in emitted candidates may trigger at most one repair; do not use abstention to conceal a malformed emitted candidate."
         if grounded else "Return JSON only with candidates and optional abstentions. If a type cannot meet its target with supported questions, emit fewer or no candidates for that type and one abstention object with exactly question_type, status=INSUFFICIENT_EVIDENCE, and a nonempty reason. Legal evidence-shortage abstentions do not trigger repair. Structural errors in emitted candidates may trigger at most one repair."
         if memory_only else "Return JSON only with one top-level key candidates."),
        *(["CONTRACT GUIDANCE:", json.dumps(summary, ensure_ascii=False, indent=2)]
          if intent_guidance and not memory_only else []),
        "OUTPUT SHAPE:", json.dumps(schema, ensure_ascii=False),
        *(memory_ordering_guidance(policy) if memory_only else ordering_guidance(policy)),
        *([memory_navigation_block(expanded)] if memory_only else ["DIALOGUE:", _dialogue(expanded)]),
    ])
    if mode == TRANSCRIPT_PLUS_MEMORY:
        prompt = "\n".join((prompt, memory_navigation_block(expanded)))
    version = ("h4_hybrid_contract_guided_ordering_temporal_v1" if temporal else "h4_hybrid_contract_guided_v1")
    if memory_only:
        version = "h4_hybrid_contract_guided_memory_only_temporal_v1"
    elif grounded:
        version = "h4_hybrid_contract_guided_ordering_grounded_v2"
    return {"prompt_version": version, "prompt": prompt,
            "prompt_sha256": sha256_bytes(prompt.encode()), "expected_counts": expected_counts("contract_guided")}


def build_repair_prompt(expanded: Mapping[str, Any], contract: Mapping[str, Any], raw: Any, source: str, *,
                        intent_guidance: bool = True,
                        ordering_policy: Mapping[str, Any] | None = None,
                        primary_abstentions: list[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    base = build_prompt(expanded, contract, intent_guidance=intent_guidance, ordering_policy=ordering_policy)
    from ..h4_writer_partial_memory_v1.sources import MEMORY_ONLY, source_mode
    from ..h4_r02_ordering_temporal_contract_v1.contract import GROUNDED_VERSION, policy_config
    grounded = policy_config(ordering_policy)["version"] == GROUNDED_VERSION
    if primary_abstentions and not grounded:
        raise ValueError("PRIMARY_ABSTENTIONS_REQUIRE_GROUNDED_POLICY")
    repair_instruction = (
        "REPAIR THE FOLLOWING BATCH TO THE STRUCTURAL SCHEMA. Preserve supported content where possible. Keep 2 single_choice and 1 multi_select targets; ordering may remain INSUFFICIENT_EVIDENCE. Do not invent facts to meet an ordering quota. SOURCE=contract_guided."
        if grounded else "REPAIR THE FOLLOWING BATCH TO THE STRUCTURAL SCHEMA AND REQUESTED TYPE LIMITS. Preserve content where possible. "
        "Use only the same frozen Memory above. Evidence shortages may remain INSUFFICIENT_EVIDENCE abstentions; never invent facts to fill a quota. SOURCE=contract_guided."
        if source_mode(expanded) == MEMORY_ONLY else
        "REPAIR THE FOLLOWING BATCH TO THE EXACT COUNTS AND STRUCTURAL SCHEMA. Preserve content where possible. SOURCE=contract_guided.")
    frozen_abstentions = []
    if grounded and primary_abstentions:
        if (not isinstance(primary_abstentions, list) or len(primary_abstentions) != 1 or
                any(not isinstance(item, Mapping) or set(item) != {"question_type", "status", "reason"} or
                    item.get("question_type") != "ordering" or item.get("status") != "INSUFFICIENT_EVIDENCE" or
                    not isinstance(item.get("reason"), str) or not item["reason"].strip()
                    for item in primary_abstentions)):
            raise ValueError("INVALID_FROZEN_PRIMARY_ABSTENTIONS")
        frozen_abstentions = ["FROZEN VALID PRIMARY ABSTENTIONS: Keep the following abstention unchanged. Do not emit or repair an ordering candidate even when SC/MS triggered this repair.",
                              json.dumps(primary_abstentions, ensure_ascii=False)]
    prompt = "\n".join([base["prompt"], repair_instruction, *frozen_abstentions, json.dumps(raw or {}, ensure_ascii=False)])
    version = (base["prompt_version"].replace("_v2", "_repair_v2") if grounded
               else base["prompt_version"].replace("_v1", "_repair_v1"))
    return {**base, "prompt_version": version, "prompt": prompt, "prompt_sha256": sha256_bytes(prompt.encode())}
