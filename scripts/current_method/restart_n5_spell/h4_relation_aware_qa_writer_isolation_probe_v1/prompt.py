"""Build an isolated relation-aware wrapper around the unchanged QA Writer prompt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.current_method.restart_n5_spell.simple_append_v1.prompts import build_mcq_batch_prompt
from scripts.current_method.restart_n5_spell.simple_append_v1.qa import build_evidence_alias_registry


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _alias_groups(expanded: Mapping[str, Any], relation_audit: Mapping[str, Any]) -> dict[str, list[str]]:
    registry = build_evidence_alias_registry(expanded)
    canonical_to_alias = registry["turn_ref_to_alias"]
    source_turns = relation_audit.get("relation_source_turns") or {}
    return {
        key: [canonical_to_alias[ref] for ref in values if ref in canonical_to_alias]
        for key, values in source_turns.items() if isinstance(values, list)
    }


def build_relation_aware_writer_prompt(
    *, expanded: Mapping[str, Any], relation_contract: Mapping[str, Any],
    relation_audit: Mapping[str, Any], target_questions: int = 8,
    maximum_questions: int = 12, focus_round_index: int = 1,
) -> dict[str, Any]:
    base = build_mcq_batch_prompt(
        expanded, target_questions=target_questions,
        maximum_questions=maximum_questions, focus_round_index=focus_round_index)
    groups = _alias_groups(expanded, relation_audit)
    contract_view = {
        "relation_type": relation_contract.get("relation_type"),
        "relation_id": relation_contract.get("relation_id"),
        "fact_A": {
            "description": relation_contract.get("fact_A", {}).get("value"),
            "evidence_aliases": groups.get("fact_A", []),
        },
        "trigger": {
            "description": relation_contract.get("trigger", {}).get("content"),
            "evidence_aliases": groups.get("trigger", []),
        },
        "fact_B": {
            "description": relation_contract.get("fact_B", {}).get("value"),
            "evidence_aliases": groups.get("fact_B", []),
        },
        "resolution": {
            "description": relation_contract.get("resolution", {}).get("value"),
            "evidence_aliases": groups.get("resolution", []),
        },
    }
    wrapper = (
        "\n\nH4 RELATION-AWARE WRITER ISOLATION CONDITION B\n"
        "This is an isolated prompt wrapper. Keep the original JSON schema, option, "
        "evidence alias, and coverage-role contracts unchanged. Do not output the "
        "relation contract or its metadata as a question field.\n\n"
        "RELATION TARGET\n"
        + json.dumps(contract_view, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n\nRELATION-AWARE QUESTION REQUIREMENTS\n"
        "- Reserve at least two questions for this exact relation.\n"
        "- Prefer questions that require the initial state (fact_A), the cause/trigger, "
        "and the revised or final state (fact_B/resolution).\n"
        "- A relation-targeted question must cite evidence aliases from at least two "
        "distinct relation groups, and should use causal, information_revision, "
        "state_change, temporal_sequence, or cross_context_synthesis when accurate.\n"
        "- Do not ask only what the final choice was, what was added, or what happened "
        "in one turn. Ask why/how the state changed or how the trigger relates to the "
        "revision.\n"
        "- Do not make a mechanical conjunction of two independent answers. The answer "
        "must express the relation.\n"
        "- If a relation-targeted question can be answered after removing fact_A or "
        "fact_B, omit it rather than returning an easy atomic question.\n"
        "- Continue to generate the remaining ordinary questions only after the "
        "relation-targeted slots are covered."
    )
    prompt = str(base["prompt"]) + wrapper
    return {
        **base,
        "prompt_version": "h4_relation_aware_writer_isolation_v1",
        "prompt": prompt,
        "base_prompt_sha256": base.get("prompt_sha256"),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "relation_contract_sha256": _sha(relation_contract),
        "relation_audit_sha256": _sha(relation_audit),
        "relation_alias_groups": groups,
        "condition": "B_relation_aware",
    }
