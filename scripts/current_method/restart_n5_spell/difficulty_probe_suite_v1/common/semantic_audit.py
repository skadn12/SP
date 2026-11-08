from __future__ import annotations

import re
from typing import Any, Iterable, Mapping


def norm(text: Any) -> str:
    return " ".join(str(text or "").casefold().split())


def tokens(text: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", norm(text)))


def paraphrase_audit(original: Mapping[str, Any], variant: Mapping[str, Any]) -> dict[str, Any]:
    errors = []
    for key in ("options", "gold_letter", "gold_text", "evidence_turn_refs", "evidence_texts"):
        if original.get(key) != variant.get(key):
            errors.append(f"frozen_field_changed:{key}")
    q_overlap = len(tokens(original.get("question")) & tokens(variant.get("question")))
    return {"valid": not errors and norm(original.get("question")) != norm(variant.get("question")),
            "errors": errors, "question_token_overlap": q_overlap,
            "question_changed": norm(original.get("question")) != norm(variant.get("question"))}


def source_audit(original: Mapping[str, Any], variant: Mapping[str, Any]) -> dict[str, Any]:
    errors = []
    for key in ("question", "options", "gold_letter", "gold_text", "evidence_turn_refs"):
        if original.get(key) != variant.get(key):
            errors.append(f"frozen_field_changed:{key}")
    old_text = " ".join(original.get("evidence_texts") or [])
    new_text = " ".join(variant.get("evidence_texts") or [])
    overlap = len(tokens(old_text) & tokens(new_text)) / max(1, len(tokens(old_text) | tokens(new_text)))
    return {"valid": not errors and norm(old_text) != norm(new_text), "errors": errors,
            "evidence_token_jaccard": round(overlap, 4),
            "new_fact_warning": len(tokens(new_text) - tokens(old_text)) > max(3, int(len(tokens(old_text)) * .35))}


def state_audit(plan: Mapping[str, Any], realized: Mapping[str, Any]) -> dict[str, Any]:
    required = str(plan.get("required_transition") or "").casefold()
    text = " ".join(str(row.get("text") or "") for row in realized.get("dialogue_turns") or []).casefold()
    keywords = [word for word in re.findall(r"[a-z]{4,}", required) if word not in {"should", "must", "after", "before"}]
    hit = sum(word in text for word in keywords)
    return {"premature_closure": len(realized.get("dialogue_turns") or []) < 6,
            "required_transition_keyword_hits": hit, "required_keyword_count": len(keywords),
            "transition_supported_heuristically": bool(keywords) and hit >= max(1, len(keywords) // 3)}

