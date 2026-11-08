"""Deterministic, mode-isolated P2-2 feedback adaptation for the original generator.

This module does not call a model, canonicalize an intent, accept a candidate,
repair a batch, or select a source. Mechanical binding validation is not a
semantic or Gold verdict. All extra state lives in sidecars.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

from ..core import atomic_json, atomic_text, canonical_json, sha256_bytes
from ..h4_qa_mining_probe_v1.intent_contract import INTENT_TYPES, ROLE_KEYS, QUESTION_TYPES

COMMON_BASE_VERSION = "h4_r02_feedback_common_base_v1"
MINER_TEMPLATE_VERSION = "h4_r02_feedback_common_miner_schema_v1"
TRANSFER_VERSION = "h4_r02_feedback_transfer_p2_2_v1"
MODES = {"ORIGINAL_HOOK_DISABLED", "LEGACY_COMMON_BASE", "P2_2_CONCRETE_TRANSFER"}
BINDING_FIELDS = {
    "intent_id", "status", "source_feedback_ids", "target_distinction",
    "evidence_refs", "scope", "change_from_source",
}
CARD_FIELDS = (
    "root_id", "feedback_id", "source_candidate_id", "source_question_type",
    "source_question", "options", "items", "historical_gold", "gold_version",
    "source_quality_status", "historical_review_status", "review_status", "source_quality_caveat", "remaining_risks", "source_evidence_refs",
    "source_cluster_id", "source_cluster_status", "solver_observations",
)
HISTORY_FIELDS = (
    "root_id", "source_candidate_id", "question_type", "stem",
    "candidate_sha256", "solver_prompt_fingerprint", "source_quality_status",
)
FIXED_POLICY = """FIXED OBSERVATION USE POLICY:
1. Where full-context observations are better than no-context observations, explore distinctions jointly involving current people, events, states, or constraints only when the current transcript supports them. Do not infer that a capability has been mastered.
2. Where both contexts perform well, explore distinctions depending more on this particular dialogue. Do not infer cheating or leakage.
3. Where full-context predictions differ from the historical Gold, use the actual predictions only for eligible sources to look for verifiable confusion. Do not infer a missing capability or an unconditional need for harder questions.
4. Ineligible sources cannot guide generation through scores or Gold. Historical options include distractors and Gold is only a historical label. Facts and answers for every new question must come from the complete current frozen transcript; historical questions and predictions do not establish dialogue facts.
5. Do not invent score thresholds, difficulty grades, strategy enums, curricula, or model-capability fields. Do not require references to a particular dialogue round. Context/position/rollout observations concern the same historical question and are not independent questions.
"""
GUIDED_FACT_RULES = """GUIDED TRANSFER RULES:
The binding is a model proposal, not a verified fact. Its target_distinction must express the same distinction as the original intent. The quoted current evidence is copied verbatim and the complete current transcript remains authoritative. The original contract remains guidance; preserve the original mixed-batch quota and candidate schema. Do not output feedback metadata or self-certify feedback use. Historical Gold and options are not new facts. Preserve scope and avoid asserting unsupported results. This brief adds no candidate admission rule.
"""
MINER_BINDING_RULES = """P2-2 BINDING OUTPUT RULES:
Return the original three intents and a feedback_bindings array in one JSON object; these are its only top-level fields. Keep every original intent field and the original intent-type and candidate-family requirements. Give each intent an explicit unique string intent_id, and use that exact ID for its binding. Do not use list positions to associate a binding.
A PROPOSED_TRANSFER binding may cite at most one eligible observation card from this root. Its target_distinction, scope, and change_from_source must be non-empty and describe a distinction supported by the current transcript, aligned with that intent's discriminative_variable. Its unique evidence_refs must be exact current turn_ref strings inside the union of that same intent's three reasoning_chain groups; reasoning_chain itself still uses 1-based positions. Do not rewrite the observed counts. Eligibility does not certify historical Gold.
When no supported transfer is available, emit NO_SUPPORTED_TRANSFER with empty source_feedback_ids and evidence_refs and empty target_distinction, scope, and change_from_source strings. When feedback_status is NO_ACTIVE_FEEDBACK every binding must take this form. Preserve the original transcript-grounded intent and do not invent or borrow a source. No extra binding fields, capability judgments, or Gold-certification fields are allowed.
"""


def _digest(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode())


def intent_item_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["intent_id", "intent_type", "reasoning_chain", "discriminative_variable", "candidate_question_types"],
        "properties": {
            "intent_id": {"type": "string", "minLength": 1, "description": "A unique explicit ID for this intent."},
            "intent_type": {"type": "string", "enum": sorted(INTENT_TYPES)},
            "reasoning_chain": {
                "type": "object", "required": list(ROLE_KEYS),
                "properties": {role: {
                    "type": "array", "minItems": 1, "uniqueItems": True,
                    "items": {"type": "integer", "minimum": 1},
                    "description": "Actual 1-based positions in the numbered current dialogue; groups must be non-empty and disjoint.",
                } for role in ROLE_KEYS},
            },
            "discriminative_variable": {"type": "string", "minLength": 1, "description": "The concrete distinction to test, not a topic."},
            "candidate_question_types": {"type": "array", "minItems": 3, "maxItems": 3,
                "uniqueItems": True, "items": {"enum": list(QUESTION_TYPES)}},
        },
    }


def binding_schema() -> dict[str, Any]:
    properties = {
        "intent_id": {"type": "string", "minLength": 1},
        "status": {"enum": ["PROPOSED_TRANSFER", "NO_SUPPORTED_TRANSFER"]},
        "source_feedback_ids": {"type": "array", "maxItems": 1, "uniqueItems": True, "items": {"type": "string", "minLength": 1}},
        "target_distinction": {"type": "string"},
        "evidence_refs": {"type": "array", "uniqueItems": True, "items": {"type": "string", "minLength": 1}},
        "scope": {"type": "string"}, "change_from_source": {"type": "string"},
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema", "title": "P2-2 feedback binding v1",
        "type": "object", "additionalProperties": False, "required": sorted(BINDING_FIELDS), "properties": properties,
        "allOf": [{"if": {"properties": {"status": {"const": "PROPOSED_TRANSFER"}}},
            "then": {"properties": {
                "source_feedback_ids": {"minItems": 1}, "evidence_refs": {"minItems": 1},
                **{field: {"minLength": 1, "pattern": "\\S"} for field in ("target_distinction", "scope", "change_from_source")},
            }}, "else": {"properties": {
                "source_feedback_ids": {"maxItems": 0}, "evidence_refs": {"maxItems": 0},
                **{field: {"const": ""} for field in ("target_distinction", "scope", "change_from_source")},
            }}}],
    }


def miner_output_schema(*, envelope: bool = False) -> dict[str, Any]:
    schema = {"type": "object", "additionalProperties": False,
        "required": ["intents"], "properties": {"intents": {
            "type": "array", "minItems": 3, "maxItems": 3, "items": intent_item_schema()}}}
    if envelope:
        schema["required"].append("feedback_bindings")
        item = binding_schema()
        item.pop("$schema", None)
        item.pop("title", None)
        schema["properties"]["feedback_bindings"] = {"type": "array", "items": item,
            "description": "One binding for each explicitly identified original intent. Semantic support is reviewed separately."}
    return schema


def revise_miner_prompt(prompt_spec: Mapping[str, Any], *, envelope: bool = False) -> dict[str, Any]:
    """Versioned shape-only revision; never mutate the original default builder."""
    original = str(prompt_spec["prompt"])
    start_marker = "Return exactly one JSON object with this shape:\n"
    end_marker = "\nNUMBERED AUTHORITATIVE DIALOGUE:"
    if original.count(start_marker) != 1 or original.count(end_marker) != 1:
        raise ValueError("MINER_TEMPLATE_SOURCE_DRIFT")
    before, rest = original.split(start_marker, 1)
    old_shape, after = rest.split(end_marker, 1)
    # Confirm replacement targets the actual former concrete format example.
    expected = {"intents": [{"intent_id": "I001", "intent_type": "state_transition",
        "reasoning_chain": {"before_state": [12], "trigger": [31], "after_state": [42]},
        "discriminative_variable": "workload sustainability", "candidate_question_types": list(QUESTION_TYPES)}]}
    if json.loads(old_shape) != expected:
        raise ValueError("MINER_EXAMPLE_SOURCE_DRIFT")
    revised = before + "Return exactly one JSON object conforming to this JSON Schema (descriptions are not output values):\n" + json.dumps(miner_output_schema(envelope=envelope), ensure_ascii=False, indent=2) + end_marker + after
    return {**dict(prompt_spec), "prompt": revised, "prompt_sha256": sha256_bytes(revised.encode()),
        "prompt_version": MINER_TEMPLATE_VERSION + ("+p2_2_envelope" if envelope else ""),
        "original_prompt_sha256": sha256_bytes(original.encode()), "common_base_version": COMMON_BASE_VERSION}


def eligible_cards(cards: Any, *, root_id: str) -> list[dict[str, Any]]:
    rows = cards.get("observation_cards", []) if isinstance(cards, Mapping) else cards
    return [deepcopy(dict(row)) for row in (rows or []) if isinstance(row, Mapping)
        and row.get("root_id") == root_id
        and row.get("source_quality_status") == "ACTIVE_GUIDANCE_ELIGIBLE"
        and row.get("source_question_type") in {"single_choice", "multi_select"}
        and isinstance(row.get("feedback_id"), str) and row["feedback_id"]]


def model_cards(cards: Any, *, root_id: str) -> list[dict[str, Any]]:
    return [{field: deepcopy(row[field]) for field in CARD_FIELDS if field in row}
        for row in eligible_cards(cards, root_id=root_id)]


def model_history(history_only: Any, *, root_id: str) -> list[dict[str, Any]]:
    rows = history_only.get("records", []) if isinstance(history_only, Mapping) else history_only
    return [{field: deepcopy(row[field]) for field in HISTORY_FIELDS if field in row}
        for row in (rows or []) if isinstance(row, Mapping) and row.get("root_id") == root_id]


def _unique_strings(value: Any, *, allow_empty: bool = False) -> bool:
    return (isinstance(value, list) and (allow_empty or bool(value))
        and all(isinstance(x, str) and bool(x.strip()) for x in value)
        and len(set(value)) == len(value))


def validate_bindings(raw: Any, intent_result: Mapping[str, Any], contracts: list[Mapping[str, Any]], *, expanded: Mapping[str, Any], root_id: str, cards: Any) -> dict[str, Any]:
    active = eligible_cards(cards, root_id=root_id)
    active_ids = {row["feedback_id"] for row in active}
    envelope_errors: list[str] = []
    if not isinstance(raw, Mapping):
        envelope_errors.append("ENVELOPE_NOT_OBJECT_OR_JSON_PARSE_FAILED")
        bindings = None
    else:
        unknown = sorted(str(key) for key in set(raw) - {"intents", "feedback_bindings"})
        if unknown:
            envelope_errors.append("UNKNOWN_ENVELOPE_FIELDS:" + ",".join(unknown))
        if "feedback_bindings" not in raw:
            envelope_errors.append("BINDINGS_MISSING")
        elif not isinstance(raw["feedback_bindings"], list):
            envelope_errors.append("BINDINGS_NOT_ARRAY")
        bindings = raw.get("feedback_bindings")
    all_intents = list(intent_result.get("intents") or [])
    contract_by_id = {str(row.get("intent_id")): row for row in contracts}
    contract_ids = set(contract_by_id)
    valid_intents = {str(row["intent_id"]): row for row in all_intents if row.get("valid") and str(row.get("intent_id")) in contract_ids}
    raw_rows = raw.get("intents", []) if isinstance(raw, Mapping) else []
    explicit_counts = Counter(row["intent_id"] for row in raw_rows if isinstance(row, Mapping)
        and isinstance(row.get("intent_id"), str) and row["intent_id"].strip()) if isinstance(raw_rows, list) else Counter()
    binding_rows = bindings if isinstance(bindings, list) else []
    binding_counts = Counter(row["intent_id"] for row in binding_rows if isinstance(row, Mapping) and isinstance(row.get("intent_id"), str))
    refs = {str(row.get("turn_ref")) for row in expanded.get("turns") or []}
    entries: list[dict[str, Any]] = []
    for index, binding in enumerate(binding_rows):
        errors: list[str] = []
        if not isinstance(binding, Mapping):
            entries.append({"binding_index": index, "intent_id": None, "mechanical_valid": False, "errors": ["BINDING_NOT_OBJECT"]})
            continue
        identifier = binding.get("intent_id")
        intent_id = identifier if isinstance(identifier, str) else None
        if set(binding) != BINDING_FIELDS:
            errors.append("BINDING_FIELDS_INVALID")
        if intent_id is None or not intent_id.strip():
            errors.append("BINDING_INTENT_ID_INVALID")
        elif binding_counts[intent_id] > 1:
            errors.append("DUPLICATE_BINDING_INTENT_ID")
        if intent_id not in explicit_counts:
            errors.append("INTENT_ID_NOT_EXPLICIT")
        elif explicit_counts[intent_id] != 1:
            errors.append("DUPLICATE_RAW_INTENT_ID")
        if intent_id not in valid_intents:
            errors.append("INTENT_FILTERED" if intent_id in explicit_counts else "UNKNOWN_INTENT_ID")
        status = binding.get("status")
        if not isinstance(status, str) or status not in {"PROPOSED_TRANSFER", "NO_SUPPORTED_TRANSFER"}:
            errors.append("BINDING_STATUS_INVALID")
        sources, evidence = binding.get("source_feedback_ids"), binding.get("evidence_refs")
        if status == "NO_SUPPORTED_TRANSFER":
            if sources != [] or evidence != [] or any(binding.get(key) != "" for key in ("target_distinction", "scope", "change_from_source")):
                errors.append("NO_SUPPORTED_TRANSFER_NONEMPTY_FIELDS")
        elif status == "PROPOSED_TRANSFER":
            if not _unique_strings(sources) or len(sources) > 1:
                errors.append("SOURCE_IDS_INVALID")
            elif any(source not in active_ids for source in sources):
                errors.append("SOURCE_NOT_ELIGIBLE_FOR_ROOT")
            if not active:
                errors.append("NO_ACTIVE_FEEDBACK")
            for field in ("target_distinction", "scope", "change_from_source"):
                if not isinstance(binding.get(field), str) or not binding[field].strip():
                    errors.append(field.upper() + "_MISSING")
            if not _unique_strings(evidence):
                errors.append("EVIDENCE_REFS_INVALID")
            else:
                if any(ref not in refs for ref in evidence):
                    errors.append("UNKNOWN_CURRENT_REF")
                current_intent = valid_intents.get(intent_id, {})
                union = {ref for group in (current_intent.get("reasoning_chain") or {}).values() for ref in group}
                if any(ref not in union for ref in evidence):
                    errors.append("REF_OUTSIDE_INTENT_CHAIN")
                contract_union = {ref for group in (contract_by_id.get(intent_id, {}).get("required_evidence_groups") or {}).values() for ref in group}
                if any(ref not in contract_union for ref in evidence):
                    errors.append("REF_OUTSIDE_REASONING_CONTRACT")
        if any(error.startswith("UNKNOWN_ENVELOPE_FIELDS") or error.startswith("ENVELOPE_NOT") for error in envelope_errors):
            errors.append("ENVELOPE_INVALID")
        entries.append({"binding_index": index, "intent_id": intent_id, "status": status,
            "mechanical_valid": not errors, "errors": sorted(set(errors)), "binding": deepcopy(dict(binding))})
    by_intent: dict[str, Any] = {}
    for intent_id in valid_intents:
        matching = [entry for entry in entries if entry.get("intent_id") == intent_id]
        selected = matching[0] if len(matching) == 1 and matching[0]["mechanical_valid"] else None
        if not active:
            reason = "NO_ACTIVE_FEEDBACK"
        elif any(error.startswith("UNKNOWN_ENVELOPE_FIELDS") or error.startswith("ENVELOPE_NOT") for error in envelope_errors):
            reason = "ENVELOPE_INVALID"
        elif "BINDINGS_NOT_ARRAY" in envelope_errors:
            reason = "BINDING_INVALID"
        elif not matching:
            reason = "BINDING_MISSING"
        elif selected is None:
            reason = "BINDING_INVALID"
        elif selected["status"] == "NO_SUPPORTED_TRANSFER":
            reason = "NO_SUPPORTED_TRANSFER"
        else:
            reason = None
        by_intent[intent_id] = {"intent_id": intent_id, "mechanical_valid": selected is not None,
            "fallback_reason": reason, "binding": deepcopy(selected["binding"]) if selected else None,
            "binding_indices": [entry["binding_index"] for entry in matching]}
    return {"transfer_version": TRANSFER_VERSION, "root_id": root_id,
        "feedback_status": "ACTIVE_FEEDBACK" if active else "NO_ACTIVE_FEEDBACK",
        "envelope_errors": envelope_errors, "bindings": entries, "by_intent": by_intent,
        "validation_scope": "mechanical only; semantic alignment, evidence and Gold require offline review",
        "valid_proposed_binding_count": sum(value["fallback_reason"] is None for value in by_intent.values())}


def compile_free_brief(*, root_id: str, intent_label: Mapping[str, Any], cards: Any, history_only: Any) -> dict[str, Any]:
    """No binding argument or binding-derived state may enter this function."""
    projected = model_cards(cards, root_id=root_id)
    return {"brief_version": TRANSFER_VERSION, "root_id": root_id,
        "intent_label": {key: str(intent_label[key]) for key in ("intent_id", "intent_type")},
        "generation_source": "free_exploration", "assigned_feedback_mode": "P2_2_CONCRETE_TRANSFER",
        "effective_feedback_mode": "ROOT_OBSERVATIONS" if projected else "NO_ACTIVE_FEEDBACK",
        "fallback_reason": None if projected else "NO_ACTIVE_FEEDBACK",
        "feedback_status": "ACTIVE_FEEDBACK" if projected else "NO_ACTIVE_FEEDBACK",
        "observation_cards": projected, "historical_tracking": model_history(history_only, root_id=root_id),
        "fixed_policy": FIXED_POLICY}


def compile_guided_brief(*, root_id: str, contract: Mapping[str, Any], validation: Mapping[str, Any], expanded: Mapping[str, Any], cards: Any) -> dict[str, Any]:
    intent_id = str(contract["intent_id"])
    item = validation.get("by_intent", {}).get(intent_id, {"fallback_reason": "BINDING_MISSING", "binding": None})
    reason, binding = item.get("fallback_reason"), item.get("binding")
    brief = {"brief_version": TRANSFER_VERSION, "root_id": root_id, "intent_id": intent_id,
        "generation_source": "contract_guided", "assigned_feedback_mode": "P2_2_CONCRETE_TRANSFER",
        "effective_feedback_mode": "CONCRETE_TRANSFER" if reason is None else "GENERIC_WRITER_GUIDANCE",
        "fallback_reason": reason}
    if reason is None:
        ids = set(binding["source_feedback_ids"])
        ref_map = {str(turn.get("turn_ref")): turn for turn in expanded.get("turns") or []}
        brief.update({"binding": deepcopy(binding),
            "observation_cards": [card for card in model_cards(cards, root_id=root_id) if card["feedback_id"] in ids],
            "current_evidence": [{"turn_ref": ref, "speaker": ref_map[ref].get("speaker"), "text": ref_map[ref].get("text")} for ref in binding["evidence_refs"]],
            "fixed_policy": FIXED_POLICY, "fact_rules": GUIDED_FACT_RULES})
    return brief


def record_semantic_diagnostic(*, intent_id: str, judgment: str, reason: str, evidence_refs: list[str]) -> dict[str, Any]:
    """Record a supplied direct-review verdict; never infer from word similarity."""
    if judgment not in {"PASS", "FAIL", "UNDETERMINED", "NOT_APPLICABLE"}:
        raise ValueError("invalid semantic diagnostic judgment")
    return {"intent_id": intent_id, "BINDING_INTENT_MISMATCH": judgment,
        "diagnostic_code": "BINDING_INTENT_MISMATCH" if judgment == "FAIL" else None,
        "reason": reason, "evidence_refs": list(evidence_refs), "affects_candidate_admission": False,
        "review_method": "Codex direct transcript review"}


class TransferHooks:
    def __init__(self, mode: str, cards: Any, history_only: Any, generic_writer_suffix: str, legacy_intent_suffix: str = ""):
        if mode not in MODES or mode == "ORIGINAL_HOOK_DISABLED":
            raise ValueError("ORIGINAL_HOOK_DISABLED must use hooks=None; otherwise select an explicit adapted mode")
        self.mode = mode
        self.cards = deepcopy(cards)
        self.history_only = deepcopy(history_only)
        self.generic_writer_suffix = str(generic_writer_suffix)
        self.legacy_intent_suffix = str(legacy_intent_suffix)
        self.validations: dict[str, Any] = {}
        self.compiled: dict[tuple[str, str, str], dict[str, Any]] = {}

    def miner_prompt(self, prompt_spec: Mapping[str, Any], *, expanded: Mapping[str, Any], root_id: str, root_dir: Path) -> dict[str, Any]:
        revised = revise_miner_prompt(prompt_spec, envelope=self.mode == "P2_2_CONCRETE_TRANSFER")
        if self.mode == "LEGACY_COMMON_BASE":
            suffix = self.legacy_intent_suffix
        else:
            cards = model_cards(self.cards, root_id=root_id)
            payload = {"root_id": root_id, "feedback_status": "ACTIVE_FEEDBACK" if cards else "NO_ACTIVE_FEEDBACK",
                "observation_cards": cards, "historical_tracking": model_history(self.history_only, root_id=root_id)}
            suffix = MINER_BINDING_RULES + "\n" + FIXED_POLICY + "\nROOT OBSERVATIONS AND HISTORICAL TRACKING:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
        prompt = revised["prompt"] + ("\n" + suffix if suffix else "")
        result = {**revised, "prompt": prompt, "prompt_sha256": sha256_bytes(prompt.encode()), "feedback_mode": self.mode}
        atomic_json(Path(root_dir) / "miner_prompt_adaptation.json", {key: value for key, value in result.items() if key != "prompt"})
        return result

    def project_miner_response(self, raw: Any, *, expanded: Mapping[str, Any], root_id: str, root_dir: Path) -> Any:
        if self.mode == "P2_2_CONCRETE_TRANSFER" and isinstance(raw, Mapping) and "intents" in raw:
            return {"intents": deepcopy(raw["intents"])}
        return raw

    def after_intents(self, raw: Any, intent_result: Mapping[str, Any], contracts: list[Mapping[str, Any]], *, expanded: Mapping[str, Any], root_id: str, root_dir: Path) -> None:
        if self.mode == "LEGACY_COMMON_BASE":
            result = {"root_id": root_id, "mode": self.mode, "status": "NOT_APPLICABLE", "by_intent": {}, "bindings": [], "envelope_errors": []}
        else:
            result = validate_bindings(raw, intent_result, contracts, expanded=expanded, root_id=root_id, cards=self.cards)
        self.validations[root_id] = result
        atomic_json(Path(root_dir) / "feedback_bindings.json", {"root_id": root_id, "mode": self.mode,
            "feedback_bindings": deepcopy(raw.get("feedback_bindings")) if isinstance(raw, Mapping) else None,
            "full_response_sha256": _digest(raw), "intents_projection_sha256": _digest(self.project_miner_response(raw, expanded=expanded, root_id=root_id, root_dir=root_dir))})
        atomic_json(Path(root_dir) / "binding_validation.json", result)

    def writer_suffix(self, contract: Mapping[str, Any], generation_source: str, *, expanded: Mapping[str, Any], root_id: str, root_dir: Path) -> str:
        key = (root_id, str(contract["intent_id"]), generation_source)
        if key in self.compiled:
            return self.compiled[key]["rendered_suffix"]
        if generation_source not in {"contract_guided", "free_exploration"}:
            raise ValueError("unknown generation source")
        if self.mode == "LEGACY_COMMON_BASE":
            brief = {"brief_version": COMMON_BASE_VERSION, "root_id": root_id, "intent_id": key[1],
                "generation_source": generation_source, "assigned_feedback_mode": self.mode,
                "effective_feedback_mode": "GENERIC_WRITER_GUIDANCE", "fallback_reason": None}
            suffix = self.generic_writer_suffix
        elif generation_source == "free_exploration":
            # Explicit label projection before the compiler provides a data-dependency boundary.
            brief = compile_free_brief(root_id=root_id, intent_label={field: contract[field] for field in ("intent_id", "intent_type")}, cards=self.cards, history_only=self.history_only)
            suffix = "P2-2 FREE ROOT OBSERVATION BRIEF:\n" + json.dumps(brief, ensure_ascii=False, indent=2)
        else:
            if root_id not in self.validations:
                raise RuntimeError("binding validation must precede Writer brief compilation")
            brief = compile_guided_brief(root_id=root_id, contract=contract, validation=self.validations[root_id], expanded=expanded, cards=self.cards)
            suffix = ("P2-2 GUIDED CONCRETE FEEDBACK BRIEF:\n" + json.dumps(brief, ensure_ascii=False, indent=2)) if brief["fallback_reason"] is None else self.generic_writer_suffix
        entry = {**brief, "intent_id": key[1], "rendered_suffix": suffix,
            "brief_sha256": sha256_bytes(suffix.encode()), "structured_brief_sha256": _digest(brief),
            "contract_sha256": _digest(contract), "observation_cards_sha256": _digest(eligible_cards(self.cards, root_id=root_id))}
        self.compiled[key] = entry
        location = Path(root_dir) / "writer_briefs" / key[1]
        atomic_json(location / f"{generation_source}.json", entry)
        atomic_text(location / f"{generation_source}.txt", suffix)
        return suffix

    def writer_request(self, prompt_spec: Mapping[str, Any], contract: Mapping[str, Any], generation_source: str, attempt: Any, request_id: str, *, expanded: Mapping[str, Any], root_id: str, root_dir: Path) -> None:
        key = (root_id, str(contract["intent_id"]), generation_source)
        if key not in self.compiled:
            raise RuntimeError("Writer request without compiled brief")
        entry = self.compiled[key]
        suffix = entry["rendered_suffix"]
        prompt = str(prompt_spec["prompt"])
        if suffix and not prompt.endswith("\n" + suffix):
            raise RuntimeError("WRITER_BRIEF_PROMPT_MISMATCH")
        audit = {field: entry.get(field) for field in ("assigned_feedback_mode", "effective_feedback_mode", "fallback_reason", "root_id", "intent_id", "generation_source", "brief_sha256", "structured_brief_sha256", "contract_sha256", "observation_cards_sha256")}
        audit.update({"request_id": request_id, "attempt": attempt, "prompt_sha256": sha256_bytes(prompt.encode()),
            "compiled_brief_exact_suffix": True, "repair_reuses_compiled_bytes": True,
            "source_feedback_ids": list((entry.get("binding") or {}).get("source_feedback_ids") or []),
            "brief_path": str(Path(root_dir) / "writer_briefs" / key[1] / f"{generation_source}.json")})
        atomic_json(Path(root_dir) / "writer_briefs" / "requests" / f"{request_id}.json", audit)
