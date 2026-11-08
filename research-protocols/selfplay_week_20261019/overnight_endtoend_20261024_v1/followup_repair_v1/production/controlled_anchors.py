"""Prospective reviewer-selected atomic citations; never repair an old review.

The reviewer still reads every utterance and makes every semantic decision.
The controller only resolves its chosen immutable anchor IDs to original text.
"""
from copy import deepcopy
import hashlib
import json

from production_core_v3 import EvidenceResolver, require


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def make_catalog(packet, source, scene):
    resolver = EvidenceResolver(packet, source, scene)
    anchors = []
    refs = list(resolver.source) + list(resolver.native)
    for index, ref in enumerate(refs):
        canonical, domain, turn, text = resolver.resolve(ref)
        require(isinstance(text, str) and bool(text.strip()), "ANCHOR_TEXT_EMPTY", reference=ref)
        payload = {"turn_ref": canonical, "domain": domain, "speaker": turn["speaker"], "quote": text}
        anchors.append({"anchor_id": f"E{index:04d}_{digest(payload)[:12]}", **payload})
    catalog = {"schema_version": "controlled_citation_catalog_v1", "root_id": packet["root_id"],
               "condition": packet["condition"], "scene_ref": scene,
               "source_sha256": digest(source), "source_turn_count": len(source["turns"]),
               "native_event_binding": resolver.native_binding, "anchors": anchors}
    return {**catalog, "catalog_id": digest(catalog)}


def expand_review(raw, catalog, packet, source, scene):
    """Return a NEW expanded object plus mechanical map, leaving raw unchanged."""
    require(catalog == make_catalog(packet, source, scene), "ANCHOR_CATALOG_SCOPE_OR_CONTENT_CHANGED")
    require(isinstance(raw, dict) and raw.get("catalog_id") == catalog["catalog_id"],
            "REVIEW_ANCHOR_CATALOG_MISMATCH")
    lookup = {item["anchor_id"]: item for item in catalog["anchors"]}
    expanded = deepcopy(raw)
    del expanded["catalog_id"]
    mappings = []

    def get_anchor(anchor_id):
        require(isinstance(anchor_id, str) and anchor_id in lookup, "UNKNOWN_REVIEW_ANCHOR_ID")
        return lookup[anchor_id]

    checks = expanded.get("checks")
    require(isinstance(checks, dict), "REVIEW_REQUIRED_DIMENSIONS_MISSING")
    for dimension, check in checks.items():
        evidence = check.get("evidence") if isinstance(check, dict) else None
        require(isinstance(evidence, list) and bool(evidence), "EMPTY_REVIEW_CHECK", dimension=dimension)
        for index, item in enumerate(evidence):
            require(isinstance(item, dict) and set(item) == {"anchor_id"}, "REVIEW_EVIDENCE_MUST_SELECT_ATOMIC_ANCHOR")
            anchor = get_anchor(item["anchor_id"])
            evidence[index] = {key: anchor[key] for key in ("turn_ref", "speaker", "quote")}
            mappings.append({"dimension": dimension, "evidence_index": index,
                             "anchor_id": item["anchor_id"], "turn_ref": anchor["turn_ref"],
                             "quote_sha256": hashlib.sha256(anchor["quote"].encode()).hexdigest()})
    findings = expanded.get("findings")
    require(isinstance(findings, list), "REVIEW_FINDINGS_OR_CODES_MISSING")
    for finding in findings:
        require(isinstance(finding, dict) and "turn_refs" not in finding
                and isinstance(finding.get("anchor_ids"), list) and bool(finding["anchor_ids"]),
                "FINDING_MUST_SELECT_ATOMIC_ANCHORS")
        finding["turn_refs"] = [get_anchor(key)["turn_ref"] for key in finding.pop("anchor_ids")]
    receipt = {"schema_version": "controlled_citation_expansion_v1", "catalog_id": catalog["catalog_id"],
               "raw_review_sha256": digest(raw), "expanded_review_sha256": digest(expanded),
               "mappings": mappings, "controller_chose_evidence": False,
               "controller_changed_semantic_fields": False, "semantic_correctness_established": False,
               "old_reviews_reclassified": False}
    return expanded, receipt
