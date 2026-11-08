"""Versioned program-only per-row admission; semantic reviews never enter this path."""
from __future__ import annotations

from collections import Counter
from typing import Mapping

from ..core import canonical_json, sha256_bytes
from ..h4_qa_mining_hybrid_generation_v1.candidate_contract import normalize_batch
from ..h4_qa_multiroot_k3_t08_probe_v1.validator import validate_candidate

VERSION = "partial_v1"
GROUNDED_VERSION = "partial_grounded_v2"
VERSIONS = (VERSION, GROUNDED_VERSION)
COUNTS = {"single_choice": 2, "multi_select": 1, "ordering": 1}
POLICY = {"version": VERSION, "minimum_counts": COUNTS, "per_slot_caps": COUNTS,
          "selection_order": "primary_then_repair_each_original_array_order",
          "duplicate_scope": "same_root_arm_slot_primary_and_repair",
          "duplicate_key": "casefold_whitespace_normalized_type_question_ordered_options_items_and_gold",
          "semantic_admission": False, "repair_prompt": "unchanged_full_batch_repair",
          "repair_limit": 1}


def admission_policy_config(version=VERSION):
    if version not in VERSIONS:
        raise ValueError("UNKNOWN_ADMISSION_POLICY")
    if version == VERSION:
        return dict(POLICY)
    return {**POLICY, "version": GROUNDED_VERSION,
            "repair_prompt": "preserve_primary_ordering_abstention",
            "abstention_types": ["ordering"],
            "abstention_status": "INSUFFICIENT_EVIDENCE",
            "abstention_is_semantic_certification": False}


def duplicate_key(candidate):
    def norm(value):
        return " ".join(str(value).casefold().split())
    kind = candidate["question_type"]
    entries = candidate.get("items") if kind == "ordering" else candidate.get("options")
    value = {"question_type": kind, "question": norm(candidate.get("question", "")),
             "entries": [(str(k), norm(v)) for k, v in (entries or {}).items()],
             "gold": sorted(candidate.get("gold") or []) if kind == "multi_select" else candidate.get("gold")}
    return sha256_bytes(canonical_json(value).encode())


def candidate_identity(candidate, *, root_id, stage, request_id, admission_policy=VERSION):
    admission_policy_config(admission_policy)
    row = dict(candidate)
    raw_id = str(row["candidate_id"])
    if not raw_id.startswith(root_id + ":"):
        row["source_candidate_id"] = raw_id
        raw_id = root_id + ":" + raw_id
    row.update(raw_candidate_id=raw_id, candidate_id=raw_id + ":" + stage,
               admission_identity_version=admission_policy, admission_stage=stage,
               admission_request_id=request_id, root_id=root_id,
               candidate_source="fresh_generated")
    return row


def verify_retained_candidate(candidate, *, expanded, policy, context, admission_policy=VERSION):
    """Verify the admitted alias without changing the original derivation identity."""
    admission_policy_config(admission_policy)
    if (admission_policy == GROUNDED_VERSION) != ((policy or {}).get("version") == "ordering_grounded_options_v2"):
        raise ValueError("GROUNDED_ORDERING_ADMISSION_POLICY_MISMATCH")
    if candidate.get("admission_identity_version") != admission_policy:
        raise ValueError("PARTIAL_ADMISSION_IDENTITY_VERSION")
    stage, raw_id = candidate.get("admission_stage"), candidate.get("raw_candidate_id")
    if stage not in ("primary", "repair") or candidate.get("candidate_id") != str(raw_id) + ":" + stage:
        raise ValueError("PARTIAL_ADMISSION_IDENTITY_INVALID")
    identity = (candidate.get("ordering_derivation") or {}).get("identity") or {}
    if candidate.get("question_type") == "ordering" and candidate.get("ordering_contract_version"):
        if identity.get("stage") != stage or identity.get("request_id") != candidate.get("admission_request_id"):
            raise ValueError("PARTIAL_ADMISSION_REQUEST_IDENTITY_INVALID")
    original = {**candidate, "candidate_id": raw_id}
    from ..h4_r02_ordering_temporal_contract_v1.contract import verify_candidate
    return verify_candidate(original, expanded=expanded, policy=policy, context=context)


def _response_errors(raw, *, root_id, expanded, parse_error, admission_policy=VERSION):
    if parse_error:
        return [str(parse_error)]
    if not isinstance(raw, Mapping) or not isinstance(raw.get("candidates"), list):
        return ["candidate_top_level_schema"]
    errors = []
    if "root_id" in raw and raw["root_id"] != root_id:
        errors.append("response_root_identity_mismatch")
    from .sources import source_mode, source_identity
    identity = source_identity(expanded)
    for field in ("transcript_sha256", "expanded_dialogue_sha256"):
        if field in raw and (source_mode(expanded) == "MEMORY_ONLY" or
                             raw[field] != identity.get("transcript_sha256")):
            errors.append("response_transcript_identity_mismatch:" + field)
    for field in ("memory_snapshot_sha256", "memory_projection_sha256"):
        if field in raw and raw[field] != identity.get(field):
            errors.append("response_memory_identity_mismatch:" + field)
    if "abstentions" in raw:
        allowed = ({"ordering"} if admission_policy == GROUNDED_VERSION else
                   set(COUNTS) if source_mode(expanded) == "MEMORY_ONLY" else set())
        if not allowed or not isinstance(raw["abstentions"], list):
            errors.append("abstentions_not_allowed_or_invalid")
        else:
            seen = set()
            for item in raw["abstentions"]:
                if (not isinstance(item, Mapping) or
                        set(item) != {"question_type", "status", "reason"} or
                        not isinstance(item.get("question_type"), str) or
                        item.get("question_type") not in allowed or
                        item.get("status") != "INSUFFICIENT_EVIDENCE" or
                        not isinstance(item.get("reason"), str) or not item["reason"].strip() or
                        item["question_type"] in seen):
                    errors.append("abstention_schema_invalid")
                    break
                seen.add(item["question_type"])
    return errors


def repair_required(raw, normalized, acceptance, *, expanded, admission_policy=VERSION):
    """Memory shortages with a legal per-type abstention do not request repair."""
    from .sources import source_mode
    admission_policy_config(admission_policy)
    if admission_policy == GROUNDED_VERSION:
        return bool(normalized.get("response_failure") or
                    normalized.get("abstention_errors") or
                    any(row["decision"] != "PASS" for row in normalized.get("row_checks") or []) or
                    acceptance["unresolved_minimums"])
    if source_mode(expanded) != "MEMORY_ONLY":
        return not acceptance["accepted"]
    if normalized.get("response_failure") or any(
            row["decision"] != "PASS" for row in normalized.get("row_checks") or []):
        return True
    if raw.get("candidates") == []:
        acceptance["abstention_reason"] = ("EXPLICIT_INSUFFICIENT_EVIDENCE" if raw.get("abstentions")
                                          else "UNSPECIFIED_EMPTY_RESPONSE")
        return False
    abstained = {row["question_type"] for row in (raw or {}).get("abstentions") or []}
    return any(kind not in abstained for kind in acceptance["missing_minimums"])


def normalize_stage(raw, *, intent_id, source, contract, expanded, root_id,
                    stage, request_id, parse_error=None, ordering_policy=None,
                    ordering_context=None, admission_policy=VERSION,
                    inherited_abstentions=None):
    """One malformed candidate cannot crash normalization of independent rows.

    Empty prefix placeholders retain the original raw row index. They are
    discarded, never reinterpreted as model candidates. No row is repaired.
    """
    admission_policy_config(admission_policy)
    grounded = admission_policy == GROUNDED_VERSION
    if grounded != ((ordering_policy or {}).get("version") == "ordering_grounded_options_v2"):
        raise ValueError("GROUNDED_ORDERING_ADMISSION_POLICY_MISMATCH")
    if inherited_abstentions and (not grounded or stage != "repair"):
        raise ValueError("INHERITED_ABSTENTIONS_REQUIRE_GROUNDED_REPAIR")
    inherited = [dict(item) for item in inherited_abstentions or []]
    if inherited:
        inherited_errors = _response_errors({"candidates": [], "abstentions": inherited},
            root_id=root_id, expanded=expanded, parse_error=None, admission_policy=admission_policy)
        if inherited_errors:
            raise ValueError("INHERITED_ABSTENTIONS_INVALID")
    errors = _response_errors(raw, root_id=root_id, expanded=expanded, parse_error=parse_error,
                              admission_policy=admission_policy)
    if errors:
        result = {"valid": False, "errors": errors, "candidates": [],
                  "response_failure": True, "row_checks": []}
        if grounded:
            result.update(admission_policy=admission_policy, effective_abstentions=inherited,
                          abstention_errors=[], blocked_abstentions=[])
        return result
    effective = {item["question_type"]: item for item in inherited}
    frozen_types = set(effective)
    abstention_errors, blocked = [], []
    if grounded:
        emitted_types = {row.get("question_type") for row in raw["candidates"]
                         if isinstance(row, Mapping) and isinstance(row.get("question_type"), str)}
        for item in raw.get("abstentions", []):
            kind = item["question_type"]
            if kind in frozen_types:
                continue  # Keep the primary reason, not a model-authored replacement.
            if kind in emitted_types:
                abstention_errors.append("abstention_conflicts_with_emitted_candidate:" + kind)
                blocked.append(dict(item))
            else:
                effective[kind] = dict(item)
    normalized, checks = [], []
    from .sources import evidence_catalog
    refs = set(evidence_catalog(expanded))
    for index, raw_row in enumerate(raw["candidates"], 1):
        try:
            if not isinstance(raw_row, Mapping):
                raise ValueError("candidate_row_not_object")
            if grounded and raw_row.get("question_type") in frozen_types:
                raise ValueError("candidate_conflicts_with_primary_abstention")
            result = normalize_batch({"candidates": [{}] * (index-1) + [raw_row]},
                intent_id=intent_id, source=source, contract=contract, expanded=expanded,
                ordering_policy=ordering_policy, ordering_context=ordering_context)
            row = candidate_identity(result["candidates"][-1], root_id=root_id,
                                     stage=stage, request_id=request_id,
                                     admission_policy=admission_policy)
            final_check = validate_candidate(row, valid_turn_refs=refs)
            row_errors = list(row.get("schema_errors") or []) + final_check["blocking_errors"]
            if not row_errors:
                verify_retained_candidate(row, expanded=expanded, policy=ordering_policy,
                    context={**dict(ordering_context or {}), "root_id": root_id},
                    admission_policy=admission_policy)
            row["schema_errors"] = sorted(set(row_errors))
            row["schema_valid"] = not row_errors
            normalized.append(row)
            checks.append({"raw_row_index": index, "candidate_id": row["candidate_id"],
                           "decision": "PASS" if not row_errors else "REJECT", "errors": row["schema_errors"]})
        except (ValueError, TypeError, KeyError, AttributeError, IndexError) as error:
            checks.append({"raw_row_index": index, "decision": "REJECT",
                           "errors": ["candidate_local_exception:" + type(error).__name__ + ":" + str(error)]})
    result = {"valid": all(row["decision"] == "PASS" for row in checks), "errors": [],
              "candidates": normalized, "response_failure": False, "row_checks": checks}
    if grounded:
        result.update(admission_policy=admission_policy, effective_abstentions=list(effective.values()),
                      abstention_errors=abstention_errors, blocked_abstentions=blocked)
        result["valid"] = result["valid"] and not abstention_errors
    return result


def admit_stage(normalized, *, retained=None, admission_policy=VERSION):
    admission_policy_config(admission_policy)
    if admission_policy == GROUNDED_VERSION and normalized.get("admission_policy") != admission_policy:
        raise ValueError("NORMALIZED_ADMISSION_POLICY_MISMATCH")
    result = list(retained or [])
    if admission_policy == GROUNDED_VERSION and any(
            row.get("admission_identity_version") != admission_policy for row in result):
        raise ValueError("RETAINED_ADMISSION_POLICY_MISMATCH")
    counts = Counter(row["question_type"] for row in result)
    seen = {duplicate_key(row) for row in result}
    selections = []
    for row in normalized.get("candidates") or []:
        if not row.get("schema_valid"):
            decision = "REJECT_PROGRAM"
        elif duplicate_key(row) in seen:
            decision = "DUPLICATE"
        elif counts[row["question_type"]] >= COUNTS[row["question_type"]]:
            decision = "TYPE_CAP_FILLED"
        else:
            decision = "RETAIN"
            result.append(row)
            counts[row["question_type"]] += 1
            seen.add(duplicate_key(row))
        selections.append({"candidate_id": row["candidate_id"], "decision": decision})
    missing = {kind: count-counts[kind] for kind, count in COUNTS.items() if count > counts[kind]}
    acceptance = {"policy": admission_policy, "accepted": not missing, "minimum_counts": dict(COUNTS),
        "per_slot_caps": dict(COUNTS), "valid_counts": dict(counts), "missing_minimums": missing,
        "retained_candidate_count": len(result), "row_selections": selections}
    if admission_policy == GROUNDED_VERSION:
        abstentions = list(normalized.get("effective_abstentions") or [])
        abstained = {item["question_type"] for item in abstentions}
        unresolved = {kind: count for kind, count in missing.items() if kind not in abstained}
        acceptance.update(accepted=not unresolved, full_quota_met=not missing,
            unresolved_minimums=unresolved,
            abstained_minimums={kind: count for kind, count in missing.items() if kind in abstained},
            effective_abstentions=abstentions, semantic_support_status="NOT_ASSESSED")
    return result, acceptance
