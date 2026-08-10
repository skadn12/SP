from __future__ import annotations

import inspect
import json
from typing import Any, Mapping, Sequence

from .core import canonical_json, sha256_bytes
from .questioner_contract import span_catalog_index


RELATION_MINER_PROMPT_VERSION = "spell_lineage_relation_miner_single_v1"
RELATION_MINER_SCHEMA_VERSION = "spell_lineage_relation_bundle_schema_v1"
RELATION_BUNDLE_VERSION = "spell_lineage_relation_bundle_v1"

RELATION_MINER_JSON_SCHEMA: dict[str, Any] = {
    "type":"object", "additionalProperties":False,
    "required":["relation_id", "parent_condition_span_ids",
                "child_consequence_span_ids", "relation_type",
                "relation_description", "joint_dependency_explanation"],
    "properties":{
        "relation_id":{"type":"string", "minLength":1},
        "parent_condition_span_ids":{"type":"array", "minItems":1,
            "maxItems":4, "uniqueItems":True,
            "items":{"type":"string", "minLength":1}},
        "child_consequence_span_ids":{"type":"array", "minItems":1,
            "maxItems":5, "uniqueItems":True,
            "items":{"type":"string", "minLength":1}},
        "relation_type":{"type":"string", "minLength":1},
        "relation_description":{"type":"string", "minLength":1},
        "joint_dependency_explanation":{"type":"string", "minLength":1},
    },
}


def relation_schema_sha256() -> str:
    return sha256_bytes(canonical_json(RELATION_MINER_JSON_SCHEMA).encode())


def relation_prompt_template_sha256() -> str:
    material = (RELATION_MINER_PROMPT_VERSION + RELATION_MINER_SCHEMA_VERSION +
                inspect.getsource(build_relation_miner_prompt) +
                canonical_json(RELATION_MINER_JSON_SCHEMA))
    return sha256_bytes(material.encode())


def build_relation_miner_prompt(*, arc_id: str, candidate_index: int,
                                lineage_view: Mapping[str, Any],
                                span_stage_membership: Mapping[str, Sequence[str]],
                                prior_relation: Mapping[str, Any] | None) -> str:
    catalog = span_catalog_index(dict(lineage_view["span_catalog"]))
    parent_ids = set(lineage_view["selected_parent_span_ids"])
    child_ids = set(lineage_view["selected_child_span_ids"])
    spans = [{"span_id":span_id,
              "source_partition":"parent" if span_id in parent_ids else "child",
              "event_stage_ids":list(span_stage_membership.get(span_id, ())),
              "speaker":span["speaker"], "exact_text":span["exact_text"]}
             for span_id, span in catalog.items()]
    prior = None
    if prior_relation is not None:
        prior = {key:prior_relation[key] for key in (
            "relation_id", "parent_condition_span_ids",
            "child_consequence_span_ids", "relation_type",
            "relation_description")}
    payload = {"arc_id":arc_id, "candidate_index":candidate_index,
        "required_relation_id":f"{arc_id}_R{candidate_index + 1}",
        "lineage_view_sha256":lineage_view["view_sha256"],
        "covered_event_stage_ids":lineage_view["covered_event_stage_ids"],
        "accepted_prior_relation_for_distinctness":prior,
        "lineage_span_catalog":spans}
    rules = [
        "Return exactly one JSON relation object matching the schema.",
        "Select an initial condition, plan, assumption, state, commitment, or information established in parent evidence.",
        "Select an action, discovery, revision, state change, or outcome that occurs in child evidence.",
        "The parent condition and child consequence must have a real semantic dependency, not merely share a topic.",
        "The complete relation must be underdetermined if either the parent evidence or child evidence is removed.",
        "Choose the smallest sufficient evidence span set from this lineage view.",
        "Use only listed span IDs and copy the required relation_id exactly.",
        "Do not generate a question, answer, distractor, option, answer letter, quote, or field outside the schema.",
        "If a prior accepted relation is supplied, choose a different joint relation with a different canonical support set.",
        "The dialogue text and span metadata are the only factual source.",
        "Return JSON only without Markdown or explanation outside the object.",
    ]
    return "\n\n".join([
        f"PROMPT_VERSION: {RELATION_MINER_PROMPT_VERSION}",
        "RELATION MINER RULES\n- " + "\n- ".join(rules),
        "FROZEN LINEAGE INPUT\n" + json.dumps(payload, ensure_ascii=False,
                                                sort_keys=True, indent=2),
        "OUTPUT JSON SCHEMA\n" + json.dumps(RELATION_MINER_JSON_SCHEMA,
                                             ensure_ascii=False, sort_keys=True,
                                             indent=2),
    ])


def _string_list(value: Any, minimum: int, maximum: int) -> bool:
    return (isinstance(value, list) and minimum <= len(value) <= maximum and
            all(isinstance(item, str) and item.strip() for item in value) and
            len(set(value)) == len(value))


def validate_relation_candidate(raw: Any, *, arc_id: str, candidate_index: int,
                                lineage_view: Mapping[str, Any],
                                prior_support_sets: Sequence[Sequence[str]] = ()) -> dict[str, Any]:
    schema_errors: list[str] = []
    required = set(RELATION_MINER_JSON_SCHEMA["required"])
    if not isinstance(raw, dict):
        return {"decision":"relation_schema_invalid",
                "reasons":["relation_schema_invalid"], "relation_bundle":None}
    if set(raw) != required:
        schema_errors.append("relation_schema_invalid")
    for key in ("relation_id", "relation_type", "relation_description",
                "joint_dependency_explanation"):
        if not isinstance(raw.get(key), str) or not raw[key].strip():
            schema_errors.append("relation_schema_invalid")
    if raw.get("relation_id") != f"{arc_id}_R{candidate_index + 1}":
        schema_errors.append("relation_schema_invalid")
    if not _string_list(raw.get("parent_condition_span_ids"), 1, 4):
        schema_errors.append("relation_schema_invalid")
    if not _string_list(raw.get("child_consequence_span_ids"), 1, 5):
        schema_errors.append("relation_schema_invalid")
    if schema_errors:
        return {"decision":"relation_schema_invalid",
                "reasons":sorted(set(schema_errors)), "relation_bundle":None}

    catalog = span_catalog_index(dict(lineage_view["span_catalog"]))
    parent_allowed = set(lineage_view["selected_parent_span_ids"])
    child_allowed = set(lineage_view["selected_child_span_ids"])
    parent_raw = list(raw["parent_condition_span_ids"])
    child_raw = list(raw["child_consequence_span_ids"])
    reasons: list[str] = []
    if any(span_id not in catalog for span_id in parent_raw + child_raw):
        reasons.append("relation_span_invalid")
    if any(span_id not in parent_allowed for span_id in parent_raw):
        reasons.append("relation_scope_invalid")
    if any(span_id not in child_allowed for span_id in child_raw):
        reasons.append("relation_scope_invalid")
    if set(parent_raw) & set(child_raw):
        reasons.append("relation_scope_invalid")
    if reasons:
        return {"decision":sorted(set(reasons))[0],
                "reasons":sorted(set(reasons)), "relation_bundle":None}

    order = {span_id:index for index, span_id in enumerate(catalog)}
    parent = sorted(parent_raw, key=order.__getitem__)
    child = sorted(child_raw, key=order.__getitem__)
    support = sorted(set(parent + child), key=order.__getitem__)
    support_set = frozenset(support)
    if any(support_set == frozenset(previous) for previous in prior_support_sets):
        return {"decision":"duplicate_relation_candidate",
                "reasons":["duplicate_relation_candidate"],
                "relation_bundle":None, "canonical_support_span_ids":support}
    bundle = {"relation_bundle_version":RELATION_BUNDLE_VERSION,
        "arc_id":arc_id, "candidate_index":candidate_index,
        "relation_id":raw["relation_id"],
        "parent_condition_span_ids":parent,
        "child_consequence_span_ids":child,
        "canonical_support_span_ids":support,
        "relation_type":raw["relation_type"].strip(),
        "relation_description":raw["relation_description"].strip(),
        "joint_dependency_explanation":raw["joint_dependency_explanation"].strip(),
        "source_lineage_view_sha256":lineage_view["view_sha256"]}
    bundle_sha = sha256_bytes(canonical_json(bundle).encode())
    return {"decision":"relation_accept", "reasons":[],
            "relation_bundle":bundle, "relation_bundle_sha256":bundle_sha,
            "canonical_support_span_ids":support}
