"""Deterministic display permutations and semantic-gold audits."""

from __future__ import annotations

from typing import Any, Mapping


SINGLE_KEYS = ("A", "B", "C", "D")
GOLD_REBINDING_AUDIT_VERSION = "h4_gold_rebinding_audit_v2_ordered_identity"


def _canonical_keys(question_type: str, count: int) -> tuple[str, ...]:
    if question_type == "single_choice" and count == 4:
        return SINGLE_KEYS
    if question_type == "multi_select" and 4 <= count <= 6:
        return tuple(chr(ord("A") + index) for index in range(count))
    if question_type == "ordering" and 4 <= count <= 6:
        return tuple(f"I{index + 1}" for index in range(count))
    return ()


def _rotate(mapping: Mapping[str, str], shift: int) -> dict[str, str]:
    keys = list(mapping)
    values = [str(mapping[key]) for key in keys]
    return {key: values[(index - shift) % len(keys)] for index, key in enumerate(keys)}


def _assert_unique_texts(mapping: Mapping[str, str], label: str) -> None:
    values = [" ".join(str(value).strip().casefold().split()) for value in mapping.values()]
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate_{label}_text")


def position_variants(candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    qtype = str(candidate.get("question_type") or "")
    if qtype == "single_choice":
        original = {str(key): str(value) for key, value in (candidate.get("options") or {}).items()}
        if tuple(original) != SINGLE_KEYS:
            raise ValueError("single_options_not_canonical")
        _assert_unique_texts(original, "option")
        correct_text = original[str(candidate.get("correct_option"))]
        variants: list[dict[str, Any]] = []
        for shift in range(len(SINGLE_KEYS)):
            options = _rotate(original, shift)
            gold = [key for key, value in options.items() if value == correct_text]
            if len(gold) != 1:
                raise ValueError("single_gold_rebinding_ambiguous")
            variants.append({
                "variant": f"shift_{shift}",
                "shift": shift,
                "display_kind": "option_position",
                "options": options,
                "gold": gold,
            })
        return variants
    if qtype == "multi_select":
        original = {str(key): str(value) for key, value in (candidate.get("options") or {}).items()}
        keys = _canonical_keys(qtype, len(original))
        if not keys or tuple(original) != keys:
            raise ValueError("multi_options_not_canonical")
        _assert_unique_texts(original, "option")
        original_gold = [str(key) for key in candidate.get("gold") or []]
        if not 2 <= len(original_gold) <= len(keys) or len(set(original_gold)) != len(original_gold) or any(key not in original for key in original_gold):
            raise ValueError("multi_gold_invalid_for_rebinding")
        gold_texts = {original[key] for key in original_gold}
        variants = []
        for shift in range(len(keys)):
            options = _rotate(original, shift)
            gold = [key for key, value in options.items() if value in gold_texts]
            if set(gold) != set(original_gold):
                # This compares semantic text membership, not labels.
                if {options[key] for key in gold} != gold_texts:
                    raise ValueError("multi_gold_rebinding_semantic_mismatch")
            variants.append({
                "variant": f"shift_{shift}",
                "shift": shift,
                "display_kind": "option_position",
                "options": options,
                "gold": gold,
            })
        return variants
    if qtype == "ordering":
        original = {str(key): str(value) for key, value in (candidate.get("items") or {}).items()}
        keys = _canonical_keys(qtype, len(original))
        if not keys or tuple(original) != keys:
            raise ValueError("ordering_items_not_canonical")
        _assert_unique_texts(original, "item")
        gold = [str(key) for key in candidate.get("gold") or []]
        if len(gold) != len(keys) or len(set(gold)) != len(gold) or set(gold) != set(keys):
            raise ValueError("ordering_gold_invalid_for_rebinding")
        keys = list(keys)
        variants = []
        for shift in range(len(keys)):
            display = keys[shift:] + keys[:shift]
            variants.append({
                "variant": f"shift_{shift}",
                "shift": shift,
                "display_kind": "item_display_order",
                "items": {key: original[key] for key in display},
                "gold": list(gold),
            })
        return variants
    raise ValueError(f"unsupported_question_type:{qtype}")


def audit_gold_rebinding(candidate: Mapping[str, Any], variants: list[Mapping[str, Any]]) -> dict[str, Any]:
    qtype = str(candidate.get("question_type") or "")
    errors: list[str] = []
    coverage: dict[str, set[str]] = {}
    original_gold_texts: set[str] = set()
    if qtype == "single_choice":
        original = {str(k): str(v) for k, v in (candidate.get("options") or {}).items()}
        original_gold_texts = {original[str(candidate.get("correct_option"))]}
    elif qtype == "multi_select":
        original = {str(k): str(v) for k, v in (candidate.get("options") or {}).items()}
        original_gold_texts = {original[str(k)] for k in candidate.get("gold") or []}
    elif qtype == "ordering":
        original = {str(k): str(v) for k, v in (candidate.get("items") or {}).items()}
        expected_keys = set(_canonical_keys(qtype, len(original)))
    else:
        return {"audit_version": GOLD_REBINDING_AUDIT_VERSION, "error_count": 1,
                "errors": ["unsupported_question_type"], "gold_rebinding_errors": 1}

    rows: list[dict[str, Any]] = []
    for variant in variants:
        displayed = variant.get("options") or variant.get("items") or {}
        rebound = [str(x) for x in variant.get("gold") or []]
        if qtype in {"single_choice", "multi_select"}:
            rebound_texts = {str(displayed[key]) for key in rebound if key in displayed}
            ok = rebound_texts == original_gold_texts
            for key in rebound:
                coverage.setdefault(key, set()).add(str(variant["variant"]))
        else:
            # Ordering gold is expressed as stable item IDs; display order is
            # allowed to rotate, but the complete gold permutation must remain
            # unchanged.  Do not infer validity from an arbitrary item's text.
            ok = (len(rebound) == len(expected_keys)
                  and len(set(rebound)) == len(rebound) and set(rebound) == expected_keys
                  and rebound == [str(item) for item in candidate.get("gold") or []]
                  and {str(key): str(value) for key, value in displayed.items()} == original)
        if not ok:
            errors.append(f"{variant.get('variant')}:semantic_gold_mismatch")
        rows.append({
            "variant": variant.get("variant"),
            "shift": variant.get("shift"),
            "rebound_gold": rebound,
            "rebound_gold_texts": sorted(
                str(displayed[key]) for key in rebound if key in displayed),
            "semantic_gold_preserved": ok,
        })
    container = candidate.get("items") if qtype == "ordering" else candidate.get("options")
    expected_variants = len(container or {})
    if len(variants) != expected_variants:
        errors.append(f"variant_count:{len(variants)}:{expected_variants}")
    return {
        "audit_version": GOLD_REBINDING_AUDIT_VERSION,
        "question_type": qtype,
        "original_gold_texts": sorted(original_gold_texts),
        "variants": rows,
        "coverage": {key: sorted(values) for key, values in coverage.items()},
        "gold_rebinding_errors": len(errors),
        "errors": errors,
        "pass": not errors,
    }
