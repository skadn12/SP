from __future__ import annotations

import json
from typing import Any, Mapping

from ..core import sha256_bytes
from .hybrid_contract import expected_counts


def _dialogue(expanded: Mapping[str, Any]) -> str:
    from ..h4_writer_partial_memory_v1.sources import render_dialogue
    return render_dialogue(expanded)


def build_prompt(expanded: Mapping[str, Any], intent: Mapping[str, Any], *,
                 ordering_policy: Mapping[str, Any] | None = None) -> dict[str, Any]:
    from ..h4_r02_ordering_temporal_contract_v1.contract import VERSION, policy_config
    from ..h4_r02_ordering_temporal_contract_v1.guidance import ordering_guidance
    policy = policy_config(ordering_policy)
    if policy["version"] not in ("legacy", VERSION):
        raise ValueError("GROUNDED_ORDERING_V2_REQUIRES_CONTRACT_GUIDED")
    temporal = policy["version"] == VERSION
    schema = {"candidates": [{"question_type": "single_choice", "question": "...", "options": {"A": "...", "B": "...", "C": "...", "D": "..."}, "correct_option": "A", "gold": ["A"], "evidence_refs": ["valid-turn-ref"], "evidence_groups": {"evidence": ["valid-turn-ref"]}, "reasoning_groups_used": ["..."], "hypothesized_reasoning_type": "causal|temporal|decision|multi-hop|unknown", "intent_reference": str(intent.get("intent_id") or ""), "discriminative_variable": "...", "distractor_strategy": "same_topic_wrong_relation"}]}
    prompt = "\n".join([
        "ROLE: H4 hybrid QA mining free-exploration generator.",
        "Generate exactly 4 single_choice, 2 multi_select, and 2 ordering candidates for the fixed dialogue.",
        "Explore any grounded question that may expose Solver weaknesses: causal, temporal, decision, multi-hop, counterfactual, or other.",
        ("Do not force the intent contract or before/trigger/after chain. Single-choice and multi-select still provide evidence_refs and evidence_groups using only real turn refs. Ordering uses the event contract below."
         if temporal else "Do not force the intent contract or before/trigger/after chain. Still provide evidence_refs and evidence_groups using only real turn refs."),
        "Answers must be unique, distractors plausible, and the question must not copy an option verbatim.",
        "Record hypothesized_reasoning_type and reasoning_groups_used as metadata; these labels are not admission gates.",
        ("Return JSON only with one top-level key candidates. Single A-D; multi A-E + list gold; ordering event_nodes and temporal_relations without a final answer."
         if temporal else "Return JSON only with one top-level key candidates. Single A-D; multi A-E + list gold; ordering I1-I4 + permutation gold."),
        "INTENT LABEL ONLY:", json.dumps({"intent_id": intent.get("intent_id"), "intent_type": intent.get("intent_type")}, ensure_ascii=False),
        "OUTPUT SHAPE:", json.dumps(schema, ensure_ascii=False),
        *ordering_guidance(policy),
        "DIALOGUE:", _dialogue(expanded),
    ])
    return {"prompt_version": ("h4_hybrid_free_exploration_ordering_temporal_v1" if temporal else "h4_hybrid_free_exploration_v1"), "prompt": prompt,
            "prompt_sha256": sha256_bytes(prompt.encode()), "expected_counts": expected_counts("free_exploration")}


def build_repair_prompt(expanded: Mapping[str, Any], intent: Mapping[str, Any], raw: Any, *,
                        ordering_policy: Mapping[str, Any] | None = None) -> dict[str, Any]:
    base = build_prompt(expanded, intent, ordering_policy=ordering_policy)
    prompt = "\n".join([base["prompt"], "REPAIR THIS FREE-EXPLORATION BATCH TO EXACT COUNTS AND STRUCTURAL SCHEMA. Do not add semantic contract requirements.", json.dumps(raw or {}, ensure_ascii=False)])
    return {**base, "prompt_version": base["prompt_version"].replace("_v1", "_repair_v1"), "prompt": prompt, "prompt_sha256": sha256_bytes(prompt.encode())}
