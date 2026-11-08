from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from .hybrid_contract import expected_counts, provenance

QUESTION_TYPES = {"single_choice", "multi_select", "ordering"}
_SINGLE = tuple("ABCD")
_MULTI = tuple("ABCDE")
_ORDER = ("I1", "I2", "I3", "I4")


def _list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(
        isinstance(x, str) and x.strip() for x in value
    ) and len(set(value)) == len(value)


def _mapping(value: Any, keys: tuple[str, ...]) -> bool:
    return isinstance(value, Mapping) and set(value) == set(keys) and all(
        isinstance(value[k], str) and value[k].strip() for k in keys
    )


def _gold(value: Any) -> list[str]:
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    return [str(x) for x in value or []]


def normalize_batch(raw: Any, *, intent_id: str, source: str,
                    contract: Mapping[str, Any], expanded: Mapping[str, Any],
                    ordering_policy: Mapping[str, Any] | None = None,
                    ordering_context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Normalize only structural fields; no semantic grounding judgment is made."""
    from ..h4_r02_ordering_temporal_contract_v1.contract import (
        GROUNDED_VERSION, VERSION, derive_ordering, persist_derivation, policy_config,
    )
    from ..h4_writer_partial_memory_v1.sources import (
        MEMORY_ONLY, evidence_catalog, source_identity, source_mode, validate_memory_contract,
    )
    if source_mode(expanded) == MEMORY_ONLY:
        validate_memory_contract(contract)
    ordering_config = policy_config(ordering_policy)
    rows = raw.get("candidates") if isinstance(raw, Mapping) else None
    if not isinstance(rows, list):
        return {"valid": False, "errors": ["candidate_top_level_schema"], "candidates": [],
                "type_counts": {}}
    valid_refs = set(evidence_catalog(expanded))
    contract_groups = {str(k): {str(r) for r in v}
                       for k, v in (contract.get("required_evidence_groups") or {}).items()
                       if isinstance(v, list)}
    normalized: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    batch_errors: list[str] = []
    for index, raw_row in enumerate(rows, 1):
        row = raw_row if isinstance(raw_row, Mapping) else {}
        qtype = str(row.get("question_type") or "")
        counts[qtype] += 1
        errors: list[str] = []
        if (ordering_config["version"] == GROUNDED_VERSION and qtype != "ordering" and
                {"ordering_contract_version", "ordering_derivation", "derived_gold",
                 "canonical_gold", "gold_derivation", "derivation_status", "derivation_version",
                 "semantic_support_status", "program_temporal_relations", "evidence_authenticity_status"} & set(row)):
            errors.append("WRITER_ORDERING_AUTHORITY_ON_NONORDERING")
        ordering_result = None
        ordering_derivation = None
        if qtype == "ordering" and ordering_config["version"] != "legacy":
            context = dict(ordering_context or {})
            identity = {**context, "intent_id": intent_id, "source": source,
                        "raw_row_index": index,
                        "candidate_id": f"{context.get('root_id', '')}:{provenance(intent_id, source, index)['candidate_id']}"}
            ordering_result = derive_ordering(
                raw_row, expanded=expanded, policy=ordering_config, identity=identity)
            errors.extend(ordering_result["errors"])
            row = ordering_result["projected_row"]
            if context.get("root_dir") is None:
                errors.append("ORDERING_CONTEXT_MISSING")
            else:
                ordering_derivation = persist_derivation(
                    ordering_result["record"], root_dir=context["root_dir"])
        question = row.get("question")
        if qtype not in QUESTION_TYPES:
            errors.append("question_type_invalid")
        if not isinstance(question, str) or not question.strip():
            errors.append("question_missing")
        if qtype == "single_choice":
            if not _mapping(row.get("options"), _SINGLE): errors.append("single_options_invalid")
            correct = str(row.get("correct_option") or "")
            if correct not in _SINGLE: errors.append("correct_option_invalid")
            gold = [correct] if correct in _SINGLE else []
            options = dict(row.get("options") or {})
            items = {}
        elif qtype == "multi_select":
            if not _mapping(row.get("options"), _MULTI): errors.append("multi_options_invalid")
            gold = _gold(row.get("gold"))
            if not gold or any(x not in _MULTI for x in gold) or len(set(gold)) != len(gold):
                errors.append("multi_gold_invalid")
            options = dict(row.get("options") or {})
            items = {}
        elif qtype == "ordering":
            source_items = row.get("items") or row.get("options")
            order_ids = (tuple(f"I{x}" for x in range(1, ordering_config["node_count"] + 1))
                         if ordering_config["version"] != "legacy" else _ORDER)
            if not _mapping(source_items, order_ids): errors.append("ordering_items_invalid")
            gold = _gold(row.get("gold"))
            if len(gold) != len(order_ids) or set(gold) != set(order_ids):
                errors.append("ordering_gold_invalid")
            options = {}
            items = dict(source_items or {})
        else:
            gold, options, items = [], {}, {}
        evidence_refs = [str(x) for x in row.get("evidence_refs") or []]
        if not _list(evidence_refs): errors.append("evidence_refs_invalid")
        if any(ref not in valid_refs for ref in evidence_refs): errors.append("evidence_ref_unknown")
        groups_raw = row.get("evidence_groups")
        groups = {str(k): [str(x) for x in v]
                  for k, v in (groups_raw.items() if isinstance(groups_raw, Mapping) else [])
                  if isinstance(v, list)}
        if not groups: errors.append("evidence_groups_missing")
        group_refs = {ref for values in groups.values() for ref in values}
        if any(ref not in valid_refs for ref in group_refs): errors.append("evidence_group_ref_unknown")
        # Evidence groups are metadata for later alignment analysis.  The
        # structural grounding contract only requires that every referenced
        # turn exists; a model may list a group ref without repeating it in
        # the flat evidence_refs list.  Keep that discrepancy observable but
        # do not reject the candidate here.
        required = [str(x) for x in (row.get("required_reasoning") or
                                     row.get("reasoning_groups_used") or [])]
        variable = str(row.get("discriminative_variable") or "").strip()
        if not variable: errors.append("candidate_variable_missing")
        alignment = {key: bool(set(values) & contract_groups.get(key, set()))
                     for key, values in groups.items()}
        meta = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
        item = {
            **provenance(intent_id, source, index),
            "question_type": qtype,
            "question": question.strip() if isinstance(question, str) else "",
            "options": options,
            "items": items,
            "correct_option": str(row.get("correct_option") or "") if qtype == "single_choice" else None,
            "gold": gold,
            "evidence_refs": evidence_refs,
            "evidence_groups": groups,
            "reasoning_groups_used": [str(x) for x in row.get("reasoning_groups_used") or meta.get("reasoning_groups_used") or []],
            "hypothesized_reasoning_type": str(row.get("hypothesized_reasoning_type") or meta.get("hypothesized_reasoning_type") or "unknown"),
            "intent_reference": str(row.get("intent_reference") or meta.get("intent_reference") or intent_id),
            "discriminative_variable": variable,
            "distractor_strategy": str(row.get("distractor_strategy") or ""),
            "required_reasoning": required,
            "contract_group_alignment": alignment,
            "schema_errors": sorted(set(errors)),
            "schema_valid": not errors,
            "source_row_index": index,
        }
        if ordering_result is not None:
            item["ordering_contract_version"] = ordering_config["version"]
            item["ordering_derivation"] = ordering_derivation
            item["derivation_status"] = ordering_result["status"]
            item["semantic_support_status"] = "NOT_ASSESSED"
        if "evidence_source_mode" in expanded:
            item.update(source_identity(expanded))
            item["evidence_validation_scope"] = (
                "MEMORY_REFERENCE_EXISTENCE_ONLY" if source_mode(expanded) == MEMORY_ONLY
                else "TRANSCRIPT_REFERENCE_EXISTENCE_ONLY")
        normalized.append(item)
    required_counts = expected_counts(source)
    for qtype, expected in required_counts.items():
        if counts[qtype] != expected:
            batch_errors.append(f"candidate_count_mismatch:{qtype}:{counts[qtype]}:{expected}")
    if len(rows) != sum(required_counts.values()):
        batch_errors.append(f"candidate_total_mismatch:{len(rows)}:{sum(required_counts.values())}")
    return {"valid": not batch_errors and all(x["schema_valid"] for x in normalized),
            "errors": sorted(set(batch_errors)), "candidates": normalized,
            "type_counts": dict(counts), "expected_counts": required_counts}
