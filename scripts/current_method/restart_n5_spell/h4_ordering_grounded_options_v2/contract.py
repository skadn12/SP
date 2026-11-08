"""Check evidence authenticity and derive directions; do not certify semantics.

The v1 graph/identity path remains authoritative. This version validates a new
raw shape before adapting it to that graph implementation. The persisted raw
row is always the exact v2 model output, never the internal graph projection.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from ..core import canonical_json, sha256_bytes
from ..h4_r02_ordering_temporal_contract_v1 import contract as legacy
from ..h4_writer_partial_memory_v1.sources import MEMORY_ONLY, evidence_catalog, source_mode

VERSION = "ordering_grounded_options_v2"
DERIVATION_VERSION = "quoted_evidence_ordered_pair_v2"


def ordering_schema(node_count: int = 4) -> dict[str, Any]:
    legacy.policy_config({"version": VERSION, "node_count": node_count})
    schema = deepcopy(legacy.ordering_schema(node_count))
    properties = schema["properties"]
    properties["ordering_contract_version"] = {"const": VERSION}
    text = {"type": "string", "minLength": 1, "pattern": r"\S"}
    span = {"type": "object", "additionalProperties": False,
            "required": ["turn_ref", "quote"],
            "properties": {"turn_ref": text, "quote": text}}
    spans = {"type": "array", "minItems": 1, "uniqueItems": True, "items": span}
    anchors = {"type": "array", "minItems": 1, "uniqueItems": True,
               "items": {"type": "object", "additionalProperties": False,
                         "required": ["segment_id", "conversation_date"],
                         "properties": {"segment_id": text, "conversation_date": text}}}
    node = properties["event_nodes"]["items"]
    node["required"] += ["evidence_spans"]
    node["properties"].update(evidence_spans=spans, date_anchors=anchors)
    ids = node["properties"]["item_id"]
    properties["temporal_relations"]["items"] = {
        "type": "object", "additionalProperties": False,
        "required": ["ordered_pair", "evidence_refs", "evidence_spans"],
        "properties": {
            "ordered_pair": {"type": "array", "minItems": 2, "maxItems": 2,
                             "uniqueItems": True, "items": ids},
            "evidence_refs": node["properties"]["evidence_refs"],
            "evidence_spans": spans, "date_anchors": anchors}}
    return schema


def _quote_checks(entity, catalog, segments):
    errors, checked = [], []
    refs = entity.get("evidence_refs")
    refs = refs if isinstance(refs, list) else []
    valid_refs = {ref for ref in refs if isinstance(ref, str)}
    spans = entity.get("evidence_spans")
    spans = spans if isinstance(spans, list) else []
    quoted_refs = set()
    for span in spans:
        if not isinstance(span, Mapping):
            continue
        ref, quote = span.get("turn_ref"), span.get("quote")
        if not isinstance(ref, str) or ref not in valid_refs:
            errors.append("ORDERING_SPAN_REF_NOT_DECLARED")
        if isinstance(ref, str):
            quoted_refs.add(ref)
        source = catalog.get(ref) if isinstance(ref, str) else None
        source_text = source.get("text") if source else None
        if (not isinstance(quote, str) or not quote.strip() or
                not isinstance(source_text, str) or quote not in source_text):
            errors.append("ORDERING_QUOTE_NOT_EXACT_SOURCE_SUBSTRING")
        else:
            checked.append({"turn_ref": ref, "quote": quote,
                            "speaker": source.get("speaker"),
                            "substring_start": source_text.index(quote)})
    if valid_refs != quoted_refs:
        errors.append("ORDERING_QUOTE_REFERENCE_COVERAGE_MISMATCH")
    anchors = entity.get("date_anchors", [])
    if isinstance(anchors, list):
        for anchor in anchors:
            if not isinstance(anchor, Mapping):
                continue
            key = (anchor.get("segment_id"), anchor.get("conversation_date"))
            if not all(isinstance(value, str) for value in key) or key not in segments:
                errors.append("ORDERING_DATE_ANCHOR_UNKNOWN")
            elif not (valid_refs & segments[key]):
                errors.append("ORDERING_DATE_ANCHOR_OUTSIDE_EVIDENCE")
    return errors, checked


def _date_segments(expanded):
    result = {}
    turns = expanded.get("turns") or []
    for segment in expanded.get("conversation_date_segments") or []:
        if not isinstance(segment, Mapping):
            continue
        first, last = segment.get("first_turn_index_1_based"), segment.get("last_turn_index_1_based")
        key = (segment.get("segment_id"), segment.get("conversation_date"))
        if (not all(isinstance(value, str) for value in key) or
                type(first) is not int or type(last) is not int or
                not 1 <= first <= last <= len(turns)):
            continue
        members = turns[first-1:last]
        if (members[0].get("turn_ref") != segment.get("first_turn_ref") or
                members[-1].get("turn_ref") != segment.get("last_turn_ref")):
            continue
        result[key] = {turn.get("turn_ref") for turn in members}
    return result


def derive_ordering(raw_row: Any, *, expanded: Mapping[str, Any],
                    policy: Mapping[str, Any], identity: Mapping[str, Any]) -> dict[str, Any]:
    config = legacy.policy_config(policy)
    if config["version"] != VERSION:
        raise ValueError("GROUNDED_ORDERING_POLICY_REQUIRED")
    row = dict(raw_row) if isinstance(raw_row, Mapping) else {}
    schema_errors = [{"path": "/".join(str(x) for x in error.absolute_path),
                      "validator": error.validator, "message": error.message}
                     for error in Draft202012Validator(ordering_schema()).iter_errors(raw_row)]
    errors = ["ORDERING_SCHEMA_INVALID"] if schema_errors else []
    if row.get("ordering_contract_version") != VERSION:
        errors.append("ORDERING_CONTRACT_VERSION_MISMATCH")
    if source_mode(expanded) == MEMORY_ONLY:
        errors.append("GROUNDED_ORDERING_REQUIRES_TRANSCRIPT")
    catalog, segments = evidence_catalog(expanded), _date_segments(expanded)
    nodes = row.get("event_nodes") if isinstance(row.get("event_nodes"), list) else []
    relations = row.get("temporal_relations") if isinstance(row.get("temporal_relations"), list) else []
    checked_spans, evidence_errors = [], []
    for kind, entities in (("event_node", nodes), ("temporal_relation", relations)):
        for index, entity in enumerate(entities):
            if not isinstance(entity, Mapping):
                continue
            local_errors, quotes = _quote_checks(entity, catalog, segments)
            errors.extend(local_errors)
            evidence_errors.extend(local_errors)
            checked_spans.extend({"kind": kind, "index": index, **quote} for quote in quotes)

    # A program-only projection reuses the proven v1 DAG implementation. Never
    # accept old before/after fields as input to v2, even if its schema failed.
    graph_row = deepcopy(row)
    graph_row["ordering_contract_version"] = legacy.VERSION
    graph_row["event_nodes"] = [
        {key: value for key, value in node.items()
         if key not in ("evidence_spans", "date_anchors")}
        if isinstance(node, Mapping) else node for node in nodes]
    projected_relations = []
    for relation in relations:
        relation = relation if isinstance(relation, Mapping) else {}
        pair = relation.get("ordered_pair")
        pair = pair if isinstance(pair, list) and len(pair) == 2 else [None, None]
        projected_relations.append({"before": pair[0], "after": pair[1],
            "evidence_refs": relation.get("evidence_refs"),
            "temporal_basis": f"Program projection: {pair[0]} precedes {pair[1]}."})
    graph_row["temporal_relations"] = projected_relations
    result = legacy.derive_ordering(graph_row, expanded=expanded,
        policy={"version": legacy.VERSION, "node_count": 4}, identity=identity)
    errors = sorted(set(errors + result["errors"]))
    order = [] if errors else result["projected_row"]["gold"]
    record = result["record"]
    record.update(contract_version=VERSION, derivation_version=DERIVATION_VERSION,
        policy=config, raw_row=raw_row,
        raw_row_sha256=sha256_bytes(canonical_json(raw_row).encode()),
        event_nodes=nodes, temporal_relations=relations,
        program_temporal_relations=projected_relations,
        schema_errors=schema_errors, errors=errors,
        derivation_status="FAIL" if errors else "PASS", derived_gold=order,
        evidence_authenticity_status=("FAIL" if evidence_errors else
                                      "NOT_ASSESSED" if schema_errors else "PASS"),
        semantic_support_status="NOT_ASSESSED")
    record["evidence_projection"].update(
        support_scope="EXACT_QUOTE_AND_REFERENCE_EXISTENCE_ONLY",
        checked_spans=checked_spans,
        date_scope="CONVERSATION_DATE_ANCHORS_ONLY_NOT_EVENT_DATES")
    projected = {**row, "items": result["projected_row"]["items"], "gold": order,
                 "evidence_refs": result["projected_row"]["evidence_refs"]}
    return {"status": record["derivation_status"], "errors": errors,
            "projected_row": projected, "record": record}
