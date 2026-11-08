"""Versioned structural ordering derivation; semantic support is not assessed."""

from __future__ import annotations

import json
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from ..core import atomic_json, canonical_json, sha256_bytes, sha256_file


VERSION = "ordering_temporal_contract_v1"
GROUNDED_VERSION = "ordering_grounded_options_v2"
DERIVATION_VERSION = "unique_topological_order_v1"
REGISTRY_FILENAME = "ordering_runtime_context.json"
IDENTITY_FIELDS = ("run_id", "arm_id", "root_id", "intent_id", "source", "stage",
                   "request_id", "raw_row_index", "candidate_id", "raw_sha256")
PROGRAM_FIELDS = {"items", "evidence_refs", "final_order", "canonical_gold", "derived_gold",
                  "gold_derivation", "ordering_derivation", "derivation_status",
                  "semantic_support_status", "derivation_version"}


def policy_config(policy: Mapping[str, Any] | None) -> dict[str, Any]:
    if policy is None:
        return {"version": "legacy", "node_count": 4}
    if not isinstance(policy, Mapping) or set(policy) != {"version", "node_count"}:
        raise ValueError("ORDERING_POLICY_INVALID")
    version, count = policy["version"], policy["node_count"]
    if type(count) is not int or version not in ("legacy", VERSION, GROUNDED_VERSION):
        raise ValueError("ORDERING_POLICY_INVALID")
    if count not in ((4,) if version in ("legacy", GROUNDED_VERSION) else (4, 5, 6)):
        raise ValueError("ORDERING_POLICY_INVALID")
    return {"version": version, "node_count": count}


def ordering_schema(node_count: int = 4) -> dict[str, Any]:
    config = policy_config({"version": VERSION, "node_count": node_count})
    text = {"type": "string", "minLength": 1, "pattern": r"\S"}
    strings = {"type": "array", "items": text}
    refs = {**strings, "minItems": 1, "uniqueItems": True}
    ids = {"type": "string", "enum": [f"I{x}" for x in range(1, config["node_count"] + 1)]}
    metadata = {
        "type": "object", "additionalProperties": False,
        "properties": {"reasoning_groups_used": strings,
                       "hypothesized_reasoning_type": text, "intent_reference": text},
    }
    properties = {
        "question_type": {"const": "ordering"},
        "ordering_contract_version": {"const": VERSION},
        "question": text, "discriminative_variable": text,
        "event_nodes": {
            "type": "array", "minItems": node_count, "maxItems": node_count,
            "items": {"type": "object", "additionalProperties": False,
                      "required": ["item_id", "item_text", "actor", "event_state", "evidence_refs"],
                      "properties": {"item_id": ids, "item_text": text, "actor": text,
                                     "event_state": text, "evidence_refs": refs}},
        },
        "temporal_relations": {
            "type": "array",
            "items": {"type": "object", "additionalProperties": False,
                      "required": ["before", "after", "evidence_refs", "temporal_basis"],
                      "properties": {"before": ids, "after": ids, "evidence_refs": refs,
                                     "temporal_basis": text}},
        },
        "evidence_groups": {"type": "object", "minProperties": 1,
                            "additionalProperties": strings},
        "gold": {"type": "null"},
        "metadata": metadata,
        "reasoning_groups_used": strings, "required_reasoning": strings,
        "hypothesized_reasoning_type": text, "intent_reference": text,
        "distractor_strategy": {"type": "string"},
    }
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
            "additionalProperties": False, "required": ["question_type", "ordering_contract_version",
                "question", "event_nodes", "temporal_relations", "discriminative_variable",
                "evidence_groups"], "properties": properties}


def register_runtime_context(root_dir: Path, ordering_policy: Mapping[str, Any],
                             context: Mapping[str, Any]) -> dict[str, Any]:
    """Persist the trusted run/arm/root/request inventory used by verification."""
    config = policy_config(ordering_policy)
    payload = {**dict(context), "ordering_policy": config}
    requests = payload.get("requests")
    if requests is not None and not isinstance(requests, Mapping):
        raise ValueError("ORDERING_RUNTIME_REQUESTS_INVALID")
    previous_path = Path(root_dir) / REGISTRY_FILENAME
    if previous_path.exists():
        previous = load_runtime_context(root_dir)
        for key in ("ordering_policy", "run_id", "arm_id", "root_id"):
            if previous.get(key) != payload.get(key):
                raise ValueError(f"ORDERING_RUNTIME_IDENTITY_DRIFT:{key}")
        for request_id, request in (previous.get("requests") or {}).items():
            if not isinstance(requests, Mapping) or requests.get(request_id) != request:
                raise ValueError(f"ORDERING_RUNTIME_REQUEST_DRIFT:{request_id}")
    atomic_json(Path(root_dir) / REGISTRY_FILENAME, payload)
    return payload


def load_runtime_context(root_dir: Path) -> dict[str, Any]:
    path = Path(root_dir) / REGISTRY_FILENAME
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("ORDERING_RUNTIME_CONTEXT_INVALID")
    if not isinstance(value.get("ordering_policy"), Mapping):
        raise ValueError("ORDERING_RUNTIME_POLICY_MISSING")
    policy_config(value["ordering_policy"])
    return value


def resolve_runtime_context(root_dir: Path, ordering_policy: Mapping[str, Any] | None = None,
                            ordering_context: Mapping[str, Any] | None = None,
                            ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve policy from trusted configuration, never from candidate fields."""
    context = dict(ordering_context or {})
    if context.get("root_dir") is not None and Path(context["root_dir"]).resolve() != Path(root_dir).resolve():
        raise ValueError("ORDERING_RUNTIME_ROOT_DIRECTORY_MISMATCH")
    context["root_dir"] = Path(root_dir)
    path = Path(root_dir) / REGISTRY_FILENAME
    if path.exists():
        registry = load_runtime_context(root_dir)
        actual_policy = policy_config(registry["ordering_policy"])
        if ordering_policy is not None and policy_config(ordering_policy) != actual_policy:
            raise ValueError("ORDERING_RUNTIME_POLICY_MISMATCH")
        for key in ("run_id", "arm_id", "root_id"):
            if key in context and context[key] != registry.get(key):
                raise ValueError(f"ORDERING_RUNTIME_IDENTITY_MISMATCH:{key}")
            context[key] = registry.get(key)
        return actual_policy, context
    actual_policy = policy_config(ordering_policy)
    if actual_policy["version"] != "legacy":
        raise ValueError("ORDERING_RUNTIME_CONTEXT_MISSING")
    return actual_policy, context


def _sha(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def derive_ordering(raw_row: Any, *, expanded: Mapping[str, Any],
                    policy: Mapping[str, Any], identity: Mapping[str, Any]) -> dict[str, Any]:
    config = policy_config(policy)
    if config["version"] == GROUNDED_VERSION:
        from ..h4_ordering_grounded_options_v2.contract import derive_ordering as derive_grounded
        return derive_grounded(raw_row, expanded=expanded, policy=config, identity=identity)
    if config["version"] != VERSION:
        raise ValueError("ORDERING_DERIVATION_REQUIRES_NEW_POLICY")
    row = dict(raw_row) if isinstance(raw_row, Mapping) else {}
    errors: list[str] = []
    schema_errors = [{"path": "/".join(str(x) for x in error.absolute_path),
                      "validator": error.validator, "message": error.message}
                     for error in Draft202012Validator(ordering_schema(config["node_count"])).iter_errors(raw_row)]
    if schema_errors:
        errors.append("ORDERING_SCHEMA_INVALID")
    if row.get("ordering_contract_version") != VERSION:
        errors.append("ORDERING_CONTRACT_VERSION_MISMATCH")
    if "gold" in row and row["gold"] is not None:
        errors.append("WRITER_GOLD_FORBIDDEN")
    if PROGRAM_FIELDS & set(row):
        errors.append("WRITER_PROGRAM_FIELDS_FORBIDDEN")
    from ..h4_writer_partial_memory_v1.sources import (
        MEMORY_ONLY, evidence_catalog, source_identity, source_mode, transcript_envelope,
    )
    by_ref = evidence_catalog(expanded)
    memory_only = source_mode(expanded) == MEMORY_ONLY
    expected = [f"I{x}" for x in range(1, config["node_count"] + 1)]
    nodes = row.get("event_nodes") if isinstance(row.get("event_nodes"), list) else []
    edges = row.get("temporal_relations") if isinstance(row.get("temporal_relations"), list) else []
    ids = [node.get("item_id") for node in nodes if isinstance(node, Mapping)]
    string_ids = [node for node in ids if isinstance(node, str)]
    if len(nodes) != config["node_count"]:
        errors.append("ORDERING_NODE_COUNT_INVALID")
    if len(string_ids) != len(ids) or len(set(string_ids)) != len(ids):
        errors.append("ORDERING_NODE_ID_INVALID_OR_DUPLICATE")
    if set(string_ids) != set(expected):
        errors.append("ORDERING_NODE_ID_SET_INVALID")
    flat_refs: list[str] = []
    excerpts: list[dict[str, Any]] = []
    for kind, entities in (("event_node", nodes), ("temporal_relation", edges)):
        for index, entity in enumerate(entities):
            refs = entity.get("evidence_refs") if isinstance(entity, Mapping) else None
            if not isinstance(refs, list) or not refs:
                errors.append("ORDERING_EVIDENCE_MISSING")
                continue
            for ref in refs:
                if not isinstance(ref, str) or ref not in by_ref:
                    errors.append("ORDERING_EVIDENCE_REF_UNKNOWN")
                    continue
                if ref not in flat_refs:
                    flat_refs.append(ref)
                source_record = by_ref[ref]
                if memory_only:
                    excerpts.append({"kind": kind, "index": index, "memory_ref": ref,
                                     "bucket_type": source_record["bucket_type"],
                                     "key": source_record["key"], "content": source_record["content"]})
                else:
                    excerpts.append({"kind": kind, "index": index, "turn_ref": ref,
                                     "speaker": source_record.get("speaker"), "text": source_record.get("text")})
    graph: dict[str, set[str]] = {item_id: set() for item_id in expected}
    unique_edges: set[tuple[str, str]] = set()
    for edge in edges:
        if not isinstance(edge, Mapping):
            continue
        before, after = edge.get("before"), edge.get("after")
        if not isinstance(before, str) or not isinstance(after, str) or before not in graph or after not in graph:
            errors.append("ORDERING_EDGE_ENDPOINT_UNKNOWN")
            continue
        if before == after:
            errors.append("ORDERING_SELF_LOOP")
        unique_edges.add((before, after))
        graph[after].add(before)
    order: list[str] = []
    # prepare checks the entire graph for a cycle before a tie can be reported.
    sorter = TopologicalSorter(graph)
    try:
        sorter.prepare()
    except CycleError:
        errors.append("ORDERING_GRAPH_CYCLE")
    else:
        while sorter.is_active():
            ready = sorter.get_ready()
            if len(ready) != 1:
                errors.append("ORDER_NOT_DETERMINED_BY_CONSTRAINTS")
                break
            order.append(ready[0])
            sorter.done(ready[0])
    errors = sorted(set(errors))
    if errors:
        order = []
    items = {node["item_id"]: node["item_text"] for node in nodes
             if isinstance(node, Mapping) and isinstance(node.get("item_id"), str)
             and isinstance(node.get("item_text"), str)}
    projected = {**row, "items": items, "gold": order, "evidence_refs": flat_refs}
    record = {
        "contract_version": VERSION, "derivation_version": DERIVATION_VERSION,
        "policy": config, "identity": {key: identity.get(key) for key in IDENTITY_FIELDS},
        "raw_row": raw_row, "raw_row_sha256": _sha(raw_row),
        "event_nodes": nodes, "temporal_relations": edges,
        "schema_errors": schema_errors, "errors": errors,
        "derivation_status": "FAIL" if errors else "PASS", "semantic_support_status": "NOT_ASSESSED",
        "evidence_projection": {"policy": "nodes_then_relations_first_seen_exact_ref_v1",
                                "evidence_refs": flat_refs, "exact_excerpts": excerpts},
        "unique_edges": [list(edge) for edge in sorted(unique_edges)],
        "items": items, "items_sha256": _sha(items), "derived_gold": order,
    }
    if "evidence_source_mode" in expanded:
        record["evidence_source"] = source_identity(expanded)
        record["evidence_projection"]["reference_namespace"] = "memory_ref" if memory_only else "turn_ref"
        record["evidence_projection"]["support_scope"] = (
            "MEMORY_REFERENCE_EXISTENCE_ONLY" if memory_only else "TRANSCRIPT_REFERENCE_EXISTENCE_ONLY")
    if not memory_only:
        record["transcript_sha256"] = _sha(transcript_envelope(expanded))
    return {"status": record["derivation_status"], "errors": errors,
            "projected_row": projected, "record": record}


def persist_derivation(record: Mapping[str, Any], *, root_dir: Path) -> dict[str, Any]:
    identity = record["identity"]
    for field in IDENTITY_FIELDS:
        value = identity.get(field)
        if field == "raw_row_index":
            if type(value) is not int or value < 1:
                raise ValueError("ORDERING_IDENTITY_INVALID:raw_row_index")
        elif not isinstance(value, str) or not value:
            raise ValueError(f"ORDERING_IDENTITY_INVALID:{field}")
    request_id = identity["request_id"]
    if Path(request_id).name != request_id or request_id in (".", ".."):
        raise ValueError("ORDERING_REQUEST_ID_INVALID")
    relative = Path("ordering_derivations") / request_id / f"row-{identity['raw_row_index']:04d}.json"
    path = Path(root_dir) / relative
    if path.exists():
        if canonical_json(json.loads(path.read_text())) != canonical_json(record):
            raise ValueError("ORDERING_DERIVATION_DRIFT")
    else:
        atomic_json(path, record)
    return {"path": str(relative), "sha256": sha256_file(path), "identity": dict(identity)}


def _registry(context: Mapping[str, Any]) -> dict[str, Any] | None:
    root_dir = context.get("root_dir")
    if root_dir is None:
        return None
    path = Path(root_dir) / REGISTRY_FILENAME
    if not path.exists():
        return None
    return load_runtime_context(root_dir)


def verify_candidate(candidate: Mapping[str, Any], *, expanded: Mapping[str, Any],
                     policy: Mapping[str, Any] | None, context: Mapping[str, Any]) -> dict[str, Any]:
    if candidate.get("question_type") != "ordering":
        return {"status": "NOT_APPLICABLE"}
    config = policy_config(policy)
    registry = _registry(context)
    if registry is not None:
        if policy_config(registry.get("ordering_policy")) != config:
            raise ValueError("ORDERING_RUNTIME_POLICY_MISMATCH")
        for key in ("run_id", "arm_id", "root_id"):
            if key in context and registry.get(key) != context[key]:
                raise ValueError(f"ORDERING_RUNTIME_IDENTITY_MISMATCH:{key}")
    if config["version"] == "legacy":
        if candidate.get("ordering_contract_version") or candidate.get("ordering_derivation"):
            raise ValueError("ORDERING_NEW_CANDIDATE_IN_LEGACY_CONTEXT")
        return {"status": "LEGACY_EXPLICIT_CONTEXT"}
    if registry is None:
        raise ValueError("ORDERING_RUNTIME_CONTEXT_MISSING")
    if candidate.get("ordering_contract_version") != config["version"]:
        raise ValueError("ORDERING_CONTRACT_VERSION_MISMATCH")
    if config["version"] == GROUNDED_VERSION:
        if candidate.get("semantic_support_status") != "NOT_ASSESSED":
            raise ValueError("ORDERING_SEMANTIC_CERTIFICATION_FORBIDDEN")
        if candidate.get("derivation_status") != "PASS":
            raise ValueError("ORDERING_CANDIDATE_DERIVATION_STATUS_INVALID")
        if {"event_nodes", "temporal_relations", "program_temporal_relations",
            "evidence_projection", "evidence_authenticity_status", "final_order",
            "canonical_gold", "derived_gold", "gold_derivation", "derivation_version"} & set(candidate):
            raise ValueError("ORDERING_UNVERIFIED_CANDIDATE_FIELDS")
    metadata = candidate.get("ordering_derivation")
    if not isinstance(metadata, Mapping) or not context.get("root_dir"):
        raise ValueError("ORDERING_DERIVATION_MISSING")
    root_dir = Path(context["root_dir"]).resolve()
    path = (root_dir / str(metadata.get("path") or "")).resolve()
    if not path.is_relative_to(root_dir / "ordering_derivations") or not path.is_file():
        raise ValueError("ORDERING_DERIVATION_PATH_INVALID")
    if sha256_file(path) != metadata.get("sha256"):
        raise ValueError("ORDERING_DERIVATION_HASH_MISMATCH")
    record = json.loads(path.read_text())
    derivation_version = DERIVATION_VERSION
    if config["version"] == GROUNDED_VERSION:
        from ..h4_ordering_grounded_options_v2.contract import DERIVATION_VERSION as derivation_version
    if record.get("contract_version") != config["version"] or record.get("derivation_version") != derivation_version:
        raise ValueError("ORDERING_DERIVATION_VERSION_MISMATCH")
    if record.get("policy") != config or record.get("identity") != metadata.get("identity"):
        raise ValueError("ORDERING_DERIVATION_IDENTITY_MISMATCH")
    identity = record["identity"]
    for key in IDENTITY_FIELDS:
        if key in context and identity.get(key) != context[key]:
            raise ValueError(f"ORDERING_DERIVATION_CONTEXT_MISMATCH:{key}")
    root_id = str(identity.get("root_id") or "")
    source = candidate.get("generation_source")
    expected_identity = {
        "root_id": candidate.get("root_id") or context.get("root_id"),
        "intent_id": candidate.get("intent_id"), "source": source,
        "candidate_id": candidate.get("candidate_id"), "raw_row_index": candidate.get("source_row_index"),
    }
    for key, value in expected_identity.items():
        if identity.get(key) != value:
            raise ValueError(f"ORDERING_CANDIDATE_IDENTITY_MISMATCH:{key}")
    prefix = "C" if source == "contract_guided" else "F"
    expected_id = f"{root_id}:{identity['intent_id']}-{prefix}-Q{identity['raw_row_index']:02d}"
    if identity["candidate_id"] != expected_id:
        raise ValueError("ORDERING_CANDIDATE_ID_FORM_INVALID")
    expected_path = root_dir / "ordering_derivations" / identity["request_id"] / f"row-{identity['raw_row_index']:04d}.json"
    if path != expected_path.resolve():
        raise ValueError("ORDERING_DERIVATION_PATH_IDENTITY_MISMATCH")
    if registry is not None:
        for key in ("run_id", "arm_id", "root_id"):
            if identity.get(key) != registry.get(key):
                raise ValueError(f"ORDERING_REGISTRY_IDENTITY_MISMATCH:{key}")
        requests = registry.get("requests")
        if not isinstance(requests, Mapping) or identity["request_id"] not in requests:
            raise ValueError("ORDERING_REQUEST_NOT_REGISTERED")
        request = requests[identity["request_id"]]
        for key in ("intent_id", "source", "stage", "raw_sha256"):
            if not isinstance(request, Mapping) or request.get(key) != identity.get(key):
                raise ValueError(f"ORDERING_REGISTERED_REQUEST_MISMATCH:{key}")
    raw_path = root_dir / "raw" / "responses" / f"{identity['request_id']}.json.txt"
    if not raw_path.is_file():
        raise ValueError("ORDERING_RAW_RESPONSE_MISSING")
    if sha256_file(raw_path) != identity["raw_sha256"]:
        raise ValueError("ORDERING_RAW_RESPONSE_HASH_MISMATCH")
    from ..simple_append_v1.live_runtime import load_json_object_strict
    raw = load_json_object_strict(raw_path.read_text())
    rows = raw.get("candidates") if isinstance(raw, Mapping) else None
    index = identity["raw_row_index"] - 1
    if not isinstance(rows, list) or len(rows) <= index or rows[index] != record.get("raw_row"):
        raise ValueError("ORDERING_RAW_ROW_MISMATCH")
    result = derive_ordering(record.get("raw_row"), expanded=expanded, policy=config, identity=identity)
    if result["status"] != "PASS" or canonical_json(result["record"]) != canonical_json(record):
        raise ValueError("ORDERING_REDERIVATION_MISMATCH")
    if "evidence_source_mode" in expanded:
        from ..h4_writer_partial_memory_v1.sources import source_identity
        if any(candidate.get(key) != value for key, value in source_identity(expanded).items()):
            raise ValueError("ORDERING_CANDIDATE_SOURCE_IDENTITY_MISMATCH")
    projected = result["projected_row"]
    for key in ("items", "gold", "evidence_refs", "evidence_groups", "discriminative_variable"):
        if candidate.get(key) != projected.get(key):
            raise ValueError(f"ORDERING_PROJECTION_MISMATCH:{key}")
    if candidate.get("question") != projected.get("question", "").strip():
        raise ValueError("ORDERING_PROJECTION_MISMATCH:question")
    return {"status": "PASS", "semantic_support_status": "NOT_ASSESSED", "derivation_path": str(path)}
