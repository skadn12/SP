"""Deterministic post-generation audit for typed relation realization."""

from __future__ import annotations

import re
from typing import Any, Mapping


_STOPWORDS = {
    "a", "an", "and", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "the", "to", "with", "this", "that",
    "stage", "resolution", "state", "scene", "plan", "initial", "final",
    "current", "after", "before", "because", "then", "later", "open",
    "tentative", "contested", "revised", "provisional", "true", "now",
}
_CAUSAL_CUES = re.compile(
    r"\b(?:because|therefore|so|since|after|due to|as a result|which meant|to address|in response)\b",
    re.IGNORECASE,
)
_RESOLUTION_CUES = re.compile(
    r"\b(?:final|finally|resolved|settled|confirmed|committed|decided|agreed|ready)\b",
    re.IGNORECASE,
)
_RELATION_NOISE = {
    "additional", "but", "classmates", "during", "information", "initially",
    "may", "new", "not", "objection", "only", "provided", "requested",
    "reconsideration", "revision", "stakeholder", "that", "the", "trigger",
    "was", "when", "with",
}


def _tokens(value: Any) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", str(value or "").casefold())
        if len(token) > 2 and token not in _STOPWORDS
    }


def _scene_text(scene: Mapping[str, Any]) -> str:
    return " ".join(str(row.get("text") or "") for row in scene.get("dialogue_turns") or [])


def _evidence_refs(scene: Mapping[str, Any], terms: set[str]) -> list[str]:
    refs: list[str] = []
    for row in scene.get("dialogue_turns") or []:
        text_terms = _tokens(row.get("text"))
        if terms & text_terms:
            refs.append(str(row.get("turn_ref") or ""))
    return [ref for ref in refs if ref]


def _hit(terms: set[str], text: str, minimum: int = 1) -> bool:
    return len(terms & _tokens(text)) >= minimum


def audit_relation(child: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    scenes = [scene for scene in child.get("scenes") or [] if isinstance(scene, Mapping)]
    texts = [_scene_text(scene) for scene in scenes]
    fact_a_terms = _tokens(contract.get("fact_A", {}).get("value"))
    trigger_terms = _tokens(contract.get("trigger", {}).get("content"))
    fact_b_terms = _tokens(contract.get("fact_B", {}).get("value"))
    resolution_terms = _tokens(contract.get("resolution", {}).get("value"))
    early_trigger_terms = (trigger_terms - fact_a_terms) - _RELATION_NOISE
    early_revision_terms = (fact_b_terms - fact_a_terms) - _RELATION_NOISE
    # A small overlap is enough because the Realizer may paraphrase; the
    # contract still requires the typed relation and the trigger cue.
    checks = {
        "three_scenes": len(texts) == 3,
        "fact_A_present": bool(texts) and _hit(fact_a_terms, texts[0], minimum=2),
        "no_early_relation_leakage": bool(texts) and not (
            _hit(early_trigger_terms, texts[0], minimum=2) or
            _hit(early_revision_terms, texts[0], minimum=1)),
        "trigger_present": len(texts) >= 2 and _hit(trigger_terms, texts[1], minimum=2),
        "fact_B_present": len(texts) >= 2 and _hit(fact_b_terms, texts[1], minimum=2),
        "causal_link_present": len(texts) >= 2 and bool(_CAUSAL_CUES.search(texts[1])),
        "resolution_preserves_B": len(texts) == 3 and _hit(fact_b_terms, texts[2], minimum=2),
        "resolution_evidence_present": len(texts) == 3 and bool(_RESOLUTION_CUES.search(texts[2])),
    }
    relation_present = bool(
        checks["fact_A_present"] and checks["trigger_present"] and
        checks["fact_B_present"] and checks["causal_link_present"])
    state_chain_valid = bool(
        relation_present and checks["no_early_relation_leakage"] and
        checks["resolution_preserves_B"] and
        checks["resolution_evidence_present"])
    fact_a_refs = _evidence_refs(scenes[0], fact_a_terms) if scenes else []
    trigger_refs = _evidence_refs(scenes[1], trigger_terms) if len(scenes) >= 2 else []
    fact_b_refs = _evidence_refs(scenes[1], fact_b_terms) if len(scenes) >= 2 else []
    resolution_refs = _evidence_refs(scenes[2], resolution_terms) if len(scenes) == 3 else []
    return {
        "audit_version": "h4_relation_contract_audit_v1",
        "relation_id": contract.get("relation_id"),
        "relation_type": contract.get("relation_type"),
        "relation_present": relation_present,
        "relation_audit_pass": state_chain_valid,
        "trigger_supported": bool(checks["trigger_present"] and checks["causal_link_present"]),
        "state_chain_valid": state_chain_valid,
        "temporal_leakage_free": checks["no_early_relation_leakage"],
        "early_relation_leakage": not checks["no_early_relation_leakage"],
        "relation_source_turns": {
            "fact_A": fact_a_refs,
            "trigger": trigger_refs,
            "fact_B": fact_b_refs,
            "resolution": resolution_refs,
        },
        "checks": checks,
        "term_diagnostics": {
            "fact_A": sorted(fact_a_terms),
            "trigger": sorted(trigger_terms),
            "fact_B": sorted(fact_b_terms),
            "resolution": sorted(resolution_terms),
            "early_trigger_specific": sorted(early_trigger_terms),
            "early_revision_specific": sorted(early_revision_terms),
        },
        "limitations": [
            "Lexical overlap is a conservative transport audit, not a semantic proof.",
            "QA-side necessity requires a composite response and separate context ablations.",
        ],
    }
