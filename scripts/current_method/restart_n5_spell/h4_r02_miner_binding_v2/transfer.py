"""Miner Binding v2, with one deterministically derived current evidence chain.

This module performs structural association and exact indexing only. It neither
selects evidence semantically nor evaluates explanation truth. Writer previews
are offline data artifacts; this adapter provides no Writer hook.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

from ..core import atomic_json, canonical_json, sha256_bytes
from ..h4_qa_mining_probe_v1.intent_contract import ROLE_KEYS
from ..h4_qa_mining_probe_v1.reasoning_contract import contract_has_three_disjoint_groups
from ..h4_r02_feedback_transfer_p2_2_v1.transfer import (
    COMMON_BASE_VERSION, MINER_TEMPLATE_VERSION, FIXED_POLICY,
    eligible_cards, model_cards, model_history, intent_item_schema,
    miner_output_schema as common_miner_output_schema,
    revise_miner_prompt as common_revise_miner_prompt,
)

TRANSFER_VERSION = "h4_r02_miner_binding_v2"
BINDING_SCHEMA_VERSION = "h4_r02_feedback_binding_schema_v2"
MINER_ENVELOPE_VERSION = "h4_r02_miner_binding_envelope_v2"
DERIVATION = "ORIGINAL_VALIDATED_INTENT_CHAIN_ONLY"
DELIVERY_STATUS = "NOT_SENT_TO_WRITER"
BINDING_FIELDS = {"intent_id", "status", "source_feedback_ids", "scope", "change_from_source"}
CHANGE_FIELDS = {"old_question_focus", "feedback_basis", "added_distinction"}
MINER_BINDING_RULES = """P2-2 MINER BINDING V2 OUTPUT RULES:
Return exactly the original three intents and feedback_bindings in one JSON object, with no other top-level fields. Preserve the original intent fields, enum values, three disjoint nonempty reasoning_chain groups using 1-based positions, and all original candidate families. Give every intent an explicit unique string intent_id; its binding must use that exact ID. Association never uses list positions.
Each binding has exactly five fields: intent_id, status, source_feedback_ids, scope, change_from_source. Do not output evidence_refs or target_distinction in a binding. The original intent's discriminative_variable is the current distinction, and its original reasoning_chain is the sole source of current evidence. The program will derive current evidence exactly from that validated chain; it will not search, supplement, trim, or repair the chain. Historical source_evidence_refs locate the old question only; they do not select evidence for a current intent.
PROPOSED_TRANSFER must cite exactly one eligible observation card from this root. scope and all three change_from_source strings must be nonempty. change_from_source is an object with exactly old_question_focus, feedback_basis, added_distinction, never a string. old_question_focus describes the actual relation or distinction in the historical question; distractor options are not facts. feedback_basis explicitly names the relevant supplied model key (M0, M_clean_policy, or M_inclusive_policy) and full_context/no_context observation and explains why that actual behavior motivates exploring this current distinction. You may cite actual prediction patterns; do not invent observations, thresholds, capability diagnoses, or rewrite counts. The program attaches the original complete observations and denominators. A generic claim that feedback suggests harder questions is not an explanation.
added_distinction must explain the concrete new or changed relation relative to the old question and how the current dialogue evidence is needed, while staying aligned with this original intent's discriminative_variable. It must not substitute a new target, merely change topics, paraphrase the old relation, or attach a source ID to an arbitrary subject. Cross-event or cross-topic transfer is allowed when the actual relation and reason are explained; no particular round or topic is required. scope must distinguish intent, expected purpose, arrangement, and completed outcomes, and must not treat correlated changes as a proved causal effect.
When no supported transfer is available, use status NO_SUPPORTED_TRANSFER, source_feedback_ids=[], scope="", and change_from_source={"old_question_focus":"","feedback_basis":"","added_distinction":""}. Preserve the original normal transcript-grounded intent. When feedback_status is NO_ACTIVE_FEEDBACK every binding must take this form; do not invent or borrow sources. An eligible root need not transfer all intents and has no fixed quota of NO_SUPPORTED_TRANSFER. Do not add capability judgments, Gold certification, or other binding fields.
"""
GUIDED_FACT_RULES = """GUIDED V2 PREVIEW FACT RULES:
The binding is a model proposal, not a verified fact. The current distinction and evidence come solely from the original validated intent and its original contract, through ORIGINAL_VALIDATED_INTENT_CHAIN_ONLY. The quoted speaker/text are exact current transcript values. Historical question options, Gold, source evidence references, and predictions do not establish new dialogue facts. Interpret change_from_source and scope as unverified explanations; preserve the evidence's distinction between expected uses, arrangements, and completed outcomes. The complete frozen transcript remains authoritative. This data preview adds no candidate admission rule and was NOT_SENT_TO_WRITER.
"""


def _digest(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode())


def binding_schema() -> dict[str, Any]:
    explanation = {"type": "object", "additionalProperties": False,
        "required": sorted(CHANGE_FIELDS),
        "properties": {field: {"type": "string"} for field in sorted(CHANGE_FIELDS)}}
    return {"$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": BINDING_SCHEMA_VERSION, "type": "object", "additionalProperties": False,
        "required": sorted(BINDING_FIELDS), "properties": {
            "intent_id": {"type": "string", "minLength": 1, "pattern": "\\S"},
            "status": {"enum": ["PROPOSED_TRANSFER", "NO_SUPPORTED_TRANSFER"]},
            "source_feedback_ids": {"type": "array", "maxItems": 1, "uniqueItems": True,
                "items": {"type": "string", "minLength": 1, "pattern": "\\S"}},
            "scope": {"type": "string"}, "change_from_source": explanation},
        "allOf": [{"if": {"properties": {"status": {"const": "PROPOSED_TRANSFER"}}},
            "then": {"properties": {"source_feedback_ids": {"minItems": 1},
                "scope": {"minLength": 1, "pattern": "\\S"},
                "change_from_source": {"properties": {field: {"minLength": 1, "pattern": "\\S"}
                    for field in sorted(CHANGE_FIELDS)}}}},
            "else": {"properties": {"source_feedback_ids": {"maxItems": 0}, "scope": {"const": ""},
                "change_from_source": {"properties": {field: {"const": ""} for field in sorted(CHANGE_FIELDS)}}}}}]}


def miner_output_schema() -> dict[str, Any]:
    schema = common_miner_output_schema(envelope=False)
    schema["required"].append("feedback_bindings")
    item = binding_schema()
    item.pop("$schema")
    item.pop("title")
    schema["properties"]["feedback_bindings"] = {"type": "array", "items": item,
        "description": "One binding for each explicitly identified original intent. Explanations require separate offline review."}
    return schema


def revise_miner_prompt(prompt_spec: Mapping[str, Any]) -> dict[str, Any]:
    """Reuse the common shape-only revision; version only the new envelope."""
    revised = common_revise_miner_prompt(prompt_spec, envelope=False)
    original_schema = json.dumps(common_miner_output_schema(envelope=False), ensure_ascii=False, indent=2)
    marker = "Return exactly one JSON object conforming to this JSON Schema (descriptions are not output values):\n"
    section = marker + original_schema + "\nNUMBERED AUTHORITATIVE DIALOGUE:"
    if revised["prompt"].count(section) != 1:
        raise ValueError("COMMON_MINER_SCHEMA_SOURCE_DRIFT")
    prompt = revised["prompt"].replace(section, marker + json.dumps(miner_output_schema(), ensure_ascii=False, indent=2)
        + "\nNUMBERED AUTHORITATIVE DIALOGUE:", 1)
    return {**revised, "prompt": prompt, "prompt_sha256": sha256_bytes(prompt.encode()),
        "prompt_version": MINER_TEMPLATE_VERSION + "+" + MINER_ENVELOPE_VERSION,
        "binding_schema_version": BINDING_SCHEMA_VERSION}


def validate_bindings(raw: Any, intent_result: Mapping[str, Any], contracts: list[Mapping[str, Any]], *,
                      expanded: Mapping[str, Any], root_id: str, cards: Any) -> dict[str, Any]:
    """Validate structure, exact IDs and source eligibility, never text meaning."""
    active_ids = {row["feedback_id"] for row in eligible_cards(cards, root_id=root_id)}
    envelope_errors = []
    if not isinstance(raw, Mapping):
        envelope_errors.append("ENVELOPE_NOT_OBJECT_OR_JSON_PARSE_FAILED")
        binding_rows = []
        raw_rows = []
    else:
        unknown = sorted(str(key) for key in set(raw) - {"intents", "feedback_bindings"})
        if unknown:
            envelope_errors.append("UNKNOWN_ENVELOPE_FIELDS:" + ",".join(unknown))
        if "intents" not in raw:
            envelope_errors.append("INTENTS_MISSING")
        if "feedback_bindings" not in raw:
            envelope_errors.append("BINDINGS_MISSING")
        elif not isinstance(raw["feedback_bindings"], list):
            envelope_errors.append("BINDINGS_NOT_ARRAY")
        binding_rows = raw.get("feedback_bindings") if isinstance(raw.get("feedback_bindings"), list) else []
        raw_rows = raw.get("intents") if isinstance(raw.get("intents"), list) else []
    envelope_fatal = any(error.startswith(("UNKNOWN_ENVELOPE_FIELDS", "ENVELOPE_NOT", "INTENTS_MISSING"))
        for error in envelope_errors)
    explicit_counts = Counter(row["intent_id"] for row in raw_rows if isinstance(row, Mapping)
        and isinstance(row.get("intent_id"), str) and row["intent_id"].strip())
    binding_counts = Counter(row["intent_id"] for row in binding_rows if isinstance(row, Mapping)
        and isinstance(row.get("intent_id"), str))
    contract_ids = {str(row["intent_id"]) for row in contracts if contract_has_three_disjoint_groups(row)}
    original_items = list(intent_result.get("intents") or [])
    normalized_counts = Counter(str(row.get("intent_id")) for row in original_items)
    valid_intents = {str(row["intent_id"]): row for row in original_items
        if row.get("valid") and str(row.get("intent_id")) in contract_ids}
    entries = []
    for index, binding in enumerate(binding_rows):
        errors = []
        if not isinstance(binding, Mapping):
            entries.append({"binding_index": index, "intent_id": None, "status": None,
                "mechanical_valid": False, "errors": ["BINDING_SCHEMA_INVALID", "BINDING_NOT_OBJECT"],
                "binding": deepcopy(binding)})
            continue
        identifier = binding.get("intent_id")
        intent_id = identifier if isinstance(identifier, str) else None
        change = binding.get("change_from_source")
        status = binding.get("status")
        sources = binding.get("source_feedback_ids")
        schema_errors = []
        if set(binding) != BINDING_FIELDS:
            schema_errors.append("BINDING_FIELDS_INVALID")
        if intent_id is None or not intent_id.strip():
            schema_errors.append("BINDING_INTENT_ID_INVALID")
        if not isinstance(status, str) or status not in {"PROPOSED_TRANSFER", "NO_SUPPORTED_TRANSFER"}:
            schema_errors.append("BINDING_STATUS_INVALID")
        if not isinstance(change, Mapping) or set(change) != CHANGE_FIELDS or not all(
                isinstance(change.get(field), str) for field in CHANGE_FIELDS):
            schema_errors.append("CHANGE_FROM_SOURCE_SCHEMA_INVALID")
        if not isinstance(binding.get("scope"), str):
            schema_errors.append("SCOPE_TYPE_INVALID")
        sources_structured = isinstance(sources, list) and all(isinstance(value, str) and value.strip() for value in sources)
        if not sources_structured or len(sources) != len(set(sources)) or len(sources) > 1:
            schema_errors.append("SOURCE_IDS_INVALID")
        if status == "NO_SUPPORTED_TRANSFER":
            if sources != [] or binding.get("scope") != "" or not isinstance(change, Mapping) or any(
                    change.get(field) != "" for field in CHANGE_FIELDS):
                schema_errors.append("NO_SUPPORTED_TRANSFER_NONEMPTY_FIELDS")
        elif status == "PROPOSED_TRANSFER":
            if not sources_structured or len(sources) != 1:
                schema_errors.append("SOURCE_IDS_INVALID")
            elif sources[0] not in active_ids:
                errors.append("SOURCE_NOT_ELIGIBLE_FOR_ROOT")
            if not active_ids:
                errors.append("NO_ACTIVE_FEEDBACK")
            if not isinstance(binding.get("scope"), str) or not binding["scope"].strip():
                schema_errors.append("SCOPE_MISSING")
            if isinstance(change, Mapping):
                for field in CHANGE_FIELDS:
                    if not isinstance(change.get(field), str) or not change[field].strip():
                        schema_errors.append(field.upper() + "_MISSING")
        if schema_errors:
            errors.extend(["BINDING_SCHEMA_INVALID", *schema_errors])
        if intent_id is not None and binding_counts[intent_id] > 1:
            errors.append("DUPLICATE_BINDING_INTENT_ID")
        if intent_id not in explicit_counts:
            errors.append("INTENT_ID_NOT_EXPLICIT")
        elif explicit_counts[intent_id] != 1:
            errors.append("DUPLICATE_RAW_INTENT_ID")
        if intent_id is not None and normalized_counts[intent_id] > 1:
            # An explicit raw ID must not bind to an earlier valid item whose
            # original canonicalizer happened to generate that same default ID.
            errors.append("AMBIGUOUS_NORMALIZED_INTENT_ID")
        if intent_id not in valid_intents:
            errors.append("INTENT_FILTERED" if intent_id in explicit_counts else "UNKNOWN_INTENT_ID")
        if envelope_fatal:
            errors.append("ENVELOPE_INVALID")
        entries.append({"binding_index": index, "intent_id": intent_id, "status": status,
            "mechanical_valid": not errors, "errors": sorted(set(errors)), "binding": deepcopy(dict(binding))})
    by_intent = {}
    for intent_id in valid_intents:
        matches = [row for row in entries if row.get("intent_id") == intent_id]
        selected = matches[0] if len(matches) == 1 and matches[0]["mechanical_valid"] else None
        if envelope_fatal:
            state = "ENVELOPE_INVALID"
        elif "BINDINGS_NOT_ARRAY" in envelope_errors:
            state = "BINDING_INVALID"
        elif not matches:
            state = "BINDING_MISSING"
        elif selected is None:
            state = "BINDING_INVALID"
        else:
            state = selected["status"]
        by_intent[intent_id] = {"intent_id": intent_id, "mechanical_valid": selected is not None,
            "binding_state": state, "model_declared_status": matches[0].get("status") if len(matches) == 1 else None,
            "fallback_reason": None if state == "PROPOSED_TRANSFER" else state,
            "binding": deepcopy(selected["binding"]) if selected else None,
            "binding_indices": [row["binding_index"] for row in matches]}
    return {"transfer_version": TRANSFER_VERSION, "binding_schema_version": BINDING_SCHEMA_VERSION,
        "root_id": root_id, "feedback_status": "ACTIVE_FEEDBACK" if active_ids else "NO_ACTIVE_FEEDBACK",
        "envelope_errors": envelope_errors, "bindings": entries, "by_intent": by_intent,
        "validation_scope": "STRUCTURE_AND_EXACT_ASSOCIATION_ONLY; explanation truth and evidence support require offline review",
        "valid_proposed_binding_count": sum(value["binding_state"] == "PROPOSED_TRANSFER" for value in by_intent.values()),
        "valid_no_supported_transfer_count": sum(value["binding_state"] == "NO_SUPPORTED_TRANSFER" for value in by_intent.values())}


class EvidenceDerivationError(ValueError):
    """The original validated chain cannot be copied from its frozen input."""


def derive_intent_evidence(intent: Mapping[str, Any], contract: Mapping[str, Any], *,
                           expanded: Mapping[str, Any], transcript_sha256: str | None = None) -> dict[str, Any]:
    """Reusable exact chain extraction, with no target or explanation lookup.

    Call only with an item validated by the original canonicalizer and its
    accepted original contract. Every current reference is copied once, in
    frozen transcript order. Wide original chains remain unchanged.
    """
    if not intent.get("valid"):
        raise EvidenceDerivationError("ORIGINAL_INTENT_INVALID")
    if not contract_has_three_disjoint_groups(contract) or set(contract.get("required_evidence_groups") or {}) != set(ROLE_KEYS):
        raise EvidenceDerivationError("ORIGINAL_REASONING_CONTRACT_INVALID")
    if str(intent.get("intent_id")) != str(contract.get("intent_id")):
        raise EvidenceDerivationError("INTENT_CONTRACT_ID_MISMATCH")
    groups = deepcopy(intent.get("reasoning_chain"))
    if groups != contract["required_evidence_groups"]:
        raise EvidenceDerivationError("INTENT_CONTRACT_CHAIN_MISMATCH")
    refs = {ref for group in groups.values() for ref in group}
    turns = list(expanded.get("turns") or [])
    counts = Counter(turn.get("turn_ref") for turn in turns)
    if any(counts[ref] != 1 for ref in refs):
        raise EvidenceDerivationError("FROZEN_TRANSCRIPT_REF_INTEGRITY_ERROR")
    selected = [turn for turn in turns if turn.get("turn_ref") in refs]
    evidence = [{key: deepcopy(turn.get(key)) for key in ("turn_ref", "speaker", "text")} for turn in selected]
    scenes = list(dict.fromkeys(str(turn.get("scene_ref") or turn.get("session_ref") or
        str(turn["turn_ref"]).rsplit(":T", 1)[0]) for turn in selected))
    return {"intent_id": str(intent["intent_id"]), "usable": True,
        "derived_target_distinction": intent["discriminative_variable"],
        "derived_evidence_groups": groups, "derived_current_evidence": evidence,
        "derivation": DERIVATION, "delivery_status": DELIVERY_STATUS,
        "intent_sha256": _digest(intent), "contract_sha256": _digest(contract),
        "transcript_sha256": transcript_sha256 or _digest(expanded),
        "transcript_sha256_basis": "frozen_file_bytes" if transcript_sha256 else "canonical_json",
        "transcript_canonical_sha256": _digest(expanded),
        "diagnostic_counts_only": {"evidence_count": len(evidence), "scene_or_session_count": len(scenes),
            "scene_or_session_refs": scenes, "no_count_threshold_or_gate": True}}


def derive_all_intent_evidence(intent_result: Mapping[str, Any], contracts: list[Mapping[str, Any]], *,
                               expanded: Mapping[str, Any], root_id: str,
                               transcript_sha256: str | None = None) -> dict[str, Any]:
    contract_map = {str(row["intent_id"]): row for row in contracts}
    entries = []
    for index, intent in enumerate(intent_result.get("intents") or []):
        identifier = str(intent["intent_id"])
        try:
            if not intent.get("valid"):
                raise EvidenceDerivationError("ORIGINAL_INTENT_INVALID")
            if identifier not in contract_map:
                raise EvidenceDerivationError("ORIGINAL_REASONING_CONTRACT_INVALID")
            item = derive_intent_evidence(intent, contract_map[identifier], expanded=expanded,
                transcript_sha256=transcript_sha256)
        except EvidenceDerivationError as error:
            item = {"intent_id": identifier, "usable": False, "error": str(error),
                "original_intent_errors": list(intent.get("errors") or []), "delivery_status": DELIVERY_STATUS,
                "intent_sha256": _digest(intent)}
        entries.append({"intent_index": index, **item})
    return {"transfer_version": TRANSFER_VERSION, "root_id": root_id, "derivation": DERIVATION,
        "delivery_status": DELIVERY_STATUS, "items": entries,
        "by_intent": {row["intent_id"]: row for row in entries if row["usable"]},
        "usable_intent_count": sum(row["usable"] for row in entries),
        "integrity_errors": [row for row in entries if row.get("error") == "FROZEN_TRANSCRIPT_REF_INTEGRITY_ERROR"]}


def compile_guided_brief(*, root_id: str, intent: Mapping[str, Any], contract: Mapping[str, Any] | None,
                         validation: Mapping[str, Any], derived: Mapping[str, Any] | None, cards: Any) -> dict[str, Any]:
    """Offline view; derived must come from the same reusable extraction above."""
    identifier = str(intent["intent_id"])
    item = (validation.get("by_intent", {}).get(identifier, {"binding_state": "BINDING_MISSING", "binding": None})
        if intent.get("valid") else {"binding_state": "ORIGINAL_INTENT_INVALID", "binding": None})
    state = item["binding_state"]
    usable = bool(derived and derived.get("usable"))
    proposed = state == "PROPOSED_TRANSFER" and usable
    binding = item.get("binding")
    sources = set(binding["source_feedback_ids"]) if proposed else set()
    return {"brief_version": TRANSFER_VERSION, "root_id": root_id, "intent_id": identifier,
        "delivery_status": DELIVERY_STATUS, "generation_source": "contract_guided",
        "original_intent": deepcopy(dict(intent)), "original_contract": deepcopy(contract),
        "binding_state": state, "binding": deepcopy(binding),
        "derived_intent_evidence": deepcopy(derived) if usable else None,
        "effective_feedback_mode": "CONCRETE_TRANSFER_PREVIEW" if proposed else "ORIGINAL_GUIDANCE_PREVIEW",
        "fallback_reason": None if proposed else ("EVIDENCE_DERIVATION_FAILED" if state == "PROPOSED_TRANSFER" else state),
        "observation_cards": [card for card in model_cards(cards, root_id=root_id) if card["feedback_id"] in sources],
        "fixed_policy": FIXED_POLICY, "fact_rules": GUIDED_FACT_RULES}


def compile_free_brief(*, root_id: str, intent_label: Mapping[str, Any], cards: Any,
                       history_only: Any, expanded: Mapping[str, Any]) -> dict[str, Any]:
    """Future Free boundary: label, root observations, full frozen transcript.

    No binding, change, scope, contract or Guided evidence argument is accepted.
    Projection discards any additional fields in intent_label.
    """
    projected = model_cards(cards, root_id=root_id)
    return {"brief_version": TRANSFER_VERSION, "root_id": root_id, "delivery_status": DELIVERY_STATUS,
        "generation_source": "free_exploration",
        "intent_label": {key: str(intent_label[key]) for key in ("intent_id", "intent_type")},
        "feedback_status": "ACTIVE_FEEDBACK" if projected else "NO_ACTIVE_FEEDBACK",
        "observation_cards": projected, "historical_tracking": model_history(history_only, root_id=root_id),
        "complete_transcript": deepcopy(dict(expanded)), "fixed_policy": FIXED_POLICY}


class TransferHooksV2:
    """Adapter for the formal shared Miner phase; never a Writer entry point."""
    def __init__(self, cards: Any, history_only: Any = None, *, transcript_sha256: str | None = None):
        self.cards = deepcopy(cards)
        self.history_only = deepcopy(history_only if history_only is not None else {"records": []})
        self.transcript_sha256 = transcript_sha256
        self.validations: dict[str, Any] = {}
        self.derived: dict[str, Any] = {}
        self.previews: dict[str, Any] = {}

    def miner_prompt(self, prompt_spec: Mapping[str, Any], *, expanded: Mapping[str, Any], root_id: str, root_dir: Path) -> dict[str, Any]:
        revised = revise_miner_prompt(prompt_spec)
        cards = model_cards(self.cards, root_id=root_id)
        payload = {"root_id": root_id, "feedback_status": "ACTIVE_FEEDBACK" if cards else "NO_ACTIVE_FEEDBACK",
            "observation_cards": cards, "historical_tracking": model_history(self.history_only, root_id=root_id)}
        prompt = revised["prompt"] + "\n" + MINER_BINDING_RULES + "\n" + FIXED_POLICY + "\nROOT OBSERVATIONS AND HISTORICAL TRACKING:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
        result = {**revised, "prompt": prompt, "prompt_sha256": sha256_bytes(prompt.encode()),
            "feedback_mode": TRANSFER_VERSION}
        atomic_json(Path(root_dir) / "miner_prompt_adaptation.json", {key: value for key, value in result.items() if key != "prompt"})
        return result

    def project_miner_response(self, raw: Any, *, expanded: Mapping[str, Any], root_id: str, root_dir: Path) -> Any:
        return {"intents": deepcopy(raw["intents"])} if isinstance(raw, Mapping) and "intents" in raw else raw

    def after_intents(self, raw: Any, intent_result: Mapping[str, Any], contracts: list[Mapping[str, Any]], *, expanded: Mapping[str, Any], root_id: str, root_dir: Path) -> None:
        validation = validate_bindings(raw, intent_result, contracts, expanded=expanded, root_id=root_id, cards=self.cards)
        derived = derive_all_intent_evidence(intent_result, contracts, expanded=expanded, root_id=root_id,
            transcript_sha256=self.transcript_sha256)
        contract_map = {str(row["intent_id"]): row for row in contracts}
        previews = [compile_guided_brief(root_id=root_id, intent=intent,
            contract=contract_map.get(str(intent["intent_id"])) if intent.get("valid") else None,
            validation=validation, derived=derived["by_intent"].get(str(intent["intent_id"])) if intent.get("valid") else None,
            cards=self.cards) for intent in intent_result.get("intents") or []]
        self.validations[root_id], self.derived[root_id], self.previews[root_id] = validation, derived, previews
        location = Path(root_dir)
        atomic_json(location / "feedback_bindings.json", {"root_id": root_id, "transfer_version": TRANSFER_VERSION,
            "binding_schema_version": BINDING_SCHEMA_VERSION,
            "feedback_bindings": deepcopy(raw.get("feedback_bindings")) if isinstance(raw, Mapping) else None,
            "full_response_sha256": _digest(raw),
            "intents_projection_sha256": _digest(self.project_miner_response(raw, expanded=expanded, root_id=root_id, root_dir=root_dir))})
        atomic_json(location / "binding_validation.json", validation)
        atomic_json(location / "derived_intent_evidence.json", derived)
        atomic_json(location / "guided_brief_preview.json", {"root_id": root_id, "transfer_version": TRANSFER_VERSION,
            "delivery_status": DELIVERY_STATUS, "items": previews})
