from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Sequence

from .core import LETTERS, canonical_json, sha256_bytes
from .questioner_contract import derive_local_difficulty, span_catalog_index, span_catalog_sha256


STRICT_ADMISSION_MODE = "strict_contract_v7"
SPELL_STYLE_MINIMAL_MODE = "spell_style_minimal_v1"
MINIMAL_NORMALIZATION_VERSION = "normalize_minimal_text_v1"
MINIMAL_MATERIALIZER_VERSION = "spell_style_minimal_materializer_v1"
LEGACY_OPTIONAL_SCHEMA_PROFILE = "unlabeled_span_ids_optional_evidence_v1"
SPAN_IDS_ONLY_SCHEMA_PROFILE = "span_ids_only_v1"

MINIMAL_REQUIRED_FIELDS = frozenset({
    "qa_id", "question", "correct_answer_text", "distractors",
    "answer_support_span_ids",
})
MINIMAL_OPTIONAL_FIELDS = frozenset({
    "answer_evidence_values", "distractor_support", "correct_answer_explanation",
    "distractor_explanations", "reasoning_steps",
})
MODEL_RESERVED_LOCAL_FIELDS = frozenset({
    "options", "reference_answer_letter", "reference_answer_text",
    "canonical_answer_text", "difficulty_level", "reasoning_type",
    "evidence_turn_ids", "evidence_session_ids", "exact_quote",
    "answer_supporting_quotes", "distractor_supporting_quotes", "target_letter",
    "final_reasoning_conclusion", "answer_support_type",
})
MINIMAL_PLACEHOLDER_MARKERS = (
    "placeholder", "insert answer", "lorem ipsum", "[answer]", "<answer>",
    "tbd", "todo",
)
MINIMAL_RAW_ARTIFACT_MARKERS = ("```", "<|im_start|>", "<|im_end|>", "\x00")


def normalize_minimal_text_v1(value: Any) -> str:
    """NFKC, trim, collapse whitespace, and casefold; no semantic normalization."""
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", normalized.strip()).casefold()


def _literal_substring(value: Any, text: Any) -> bool:
    needle = normalize_minimal_text_v1(value)
    return bool(needle) and needle in normalize_minimal_text_v1(text)


def _contains_reserved_field(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(MODEL_RESERVED_LOCAL_FIELDS & set(value)) or any(
            _contains_reserved_field(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_reserved_field(child) for child in value)
    return False


def _has_fixed_placeholder_or_artifact(value: str) -> bool:
    normalized = normalize_minimal_text_v1(value)
    return (any(marker in normalized for marker in MINIMAL_PLACEHOLDER_MARKERS) or
            any(marker in value for marker in MINIMAL_RAW_ARTIFACT_MARKERS))


def _stable_json_serializable(value: Any) -> bool:
    try:
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                   allow_nan=False)
    except (TypeError, ValueError):
        return False
    return True


def _complexity_profile(*, turns: int, sessions: int) -> str:
    if sessions >= 2:
        return "cross_session"
    if turns >= 2:
        return "single_session_multi_turn"
    return "single_turn"


def materialize_spell_style_minimal(
    item: Any, target_letter: str, *, span_catalog: dict[str, Any],
    strict_v7_reasons: Sequence[str] = (),
    reasoning_contract: dict[str, Any] | None = None,
    schema_profile: str = LEGACY_OPTIONAL_SCHEMA_PROFILE,
) -> tuple[dict[str, Any] | None, list[str], dict[str, Any]]:
    """Apply only structural, literal-reference, and mechanical-binding hard gates."""
    diagnostics: dict[str, Any] = {
        "schema_profile":schema_profile,
        "normalization_version": MINIMAL_NORMALIZATION_VERSION,
        "optional_field_status": {key:"missing" for key in sorted(MINIMAL_OPTIONAL_FIELDS)},
        "strict_v7_contract_pass": not bool(strict_v7_reasons),
        "strict_v7_contract_reasons": sorted(set(str(x) for x in strict_v7_reasons)),
    }
    if not isinstance(item, dict) or target_letter not in LETTERS or not isinstance(span_catalog, dict):
        return None, ["minimal_generated_schema_invalid"], diagnostics
    if schema_profile not in (LEGACY_OPTIONAL_SCHEMA_PROFILE, SPAN_IDS_ONLY_SCHEMA_PROFILE):
        return None, ["minimal_schema_profile_invalid"], diagnostics
    errors: list[str] = []
    try:
        catalog = span_catalog_index(span_catalog)
    except ValueError:
        return None, ["span_catalog_invalid"], diagnostics
    if not _stable_json_serializable(item):
        errors.append("minimal_generated_schema_invalid")
    if _contains_reserved_field(item):
        errors.append("model_supplied_local_binding_fields")
    allowed_fields = (MINIMAL_REQUIRED_FIELDS | MINIMAL_OPTIONAL_FIELDS
                      if schema_profile == LEGACY_OPTIONAL_SCHEMA_PROFILE else
                      MINIMAL_REQUIRED_FIELDS | {"reasoning_steps"})
    if set(item) - allowed_fields:
        errors.append("span_ids_only_additional_properties" if
                      schema_profile == SPAN_IDS_ONLY_SCHEMA_PROFILE else
                      "minimal_generated_schema_invalid")
    for key in MINIMAL_OPTIONAL_FIELDS:
        if key in item:
            diagnostics["optional_field_status"][key] = "present"

    for key in ("qa_id", "question", "correct_answer_text"):
        if not isinstance(item.get(key), str) or not item[key].strip():
            errors.append("minimal_generated_schema_invalid")
    distractors = item.get("distractors")
    if (not isinstance(distractors, list) or len(distractors) != 3 or
            any(not isinstance(value, str) or not value.strip() for value in distractors)):
        errors.append("bad_unlabeled_options")
        distractors = distractors if isinstance(distractors, list) else []
    correct = item.get("correct_answer_text")
    option_texts = [correct, *distractors] if isinstance(correct, str) and len(distractors) == 3 else []
    if option_texts and len({normalize_minimal_text_v1(value) for value in option_texts}) != 4:
        errors.append("bad_unlabeled_options")
    text_fields = [item.get("question"), *option_texts]
    if any(isinstance(value, str) and _has_fixed_placeholder_or_artifact(value)
           for value in text_fields):
        errors.append("minimal_placeholder_or_raw_artifact")

    answer_span_ids = item.get("answer_support_span_ids")
    if (not isinstance(answer_span_ids, list) or not answer_span_ids or
            any(not isinstance(span_id, str) or not span_id.strip() for span_id in answer_span_ids)):
        errors.append("minimal_generated_schema_invalid")
        answer_span_ids = []
    if any(span_id not in catalog for span_id in answer_span_ids):
        errors.append("bad_span_id")

    evidence_values = item.get("answer_evidence_values")
    normalized_answer_values: list[str] = []
    answer_value_literal_checks: list[dict[str, Any]] = []
    if evidence_values is not None:
        if not isinstance(evidence_values, list):
            errors.append("bad_answer_evidence_binding")
        else:
            for entry in evidence_values:
                if (not isinstance(entry, dict) or set(entry) != {"span_id", "value"} or
                        not isinstance(entry.get("span_id"), str) or not entry["span_id"].strip() or
                        not isinstance(entry.get("value"), str) or not entry["value"].strip()):
                    errors.append("bad_answer_evidence_binding")
                    continue
                span_id, value = entry["span_id"], entry["value"]
                normalized_answer_values.append(normalize_minimal_text_v1(value))
                if span_id not in catalog:
                    errors.append("bad_span_id")
                    literal = False
                else:
                    literal = _literal_substring(value, catalog[span_id]["exact_text"])
                    if not literal:
                        errors.append("bad_answer_evidence_binding")
                answer_value_literal_checks.append({"span_id":span_id, "value":value,
                    "literal_in_span":literal,
                    "literal_in_correct_answer":_literal_substring(value, correct or "")})

    distractor_support = item.get("distractor_support")
    distractor_value_literal_checks: list[dict[str, Any]] = []
    if distractor_support is not None:
        if not isinstance(distractor_support, list):
            errors.append("bad_distractor_support")
        else:
            seen_indexes: set[int] = set()
            for entry in distractor_support:
                if not isinstance(entry, dict) or not {"distractor_index", "span_ids"} <= set(entry) or \
                        set(entry) - {"distractor_index", "span_ids", "evidence_values"}:
                    errors.append("bad_distractor_support")
                    continue
                index = entry.get("distractor_index")
                if (not isinstance(index, int) or isinstance(index, bool) or index not in (0, 1, 2) or
                        index in seen_indexes):
                    errors.append("bad_distractor_support")
                    continue
                seen_indexes.add(index)
                span_ids = entry.get("span_ids")
                if (not isinstance(span_ids, list) or
                        any(not isinstance(span_id, str) or not span_id.strip() for span_id in span_ids)):
                    errors.append("bad_distractor_support")
                    continue
                if any(span_id not in catalog for span_id in span_ids):
                    errors.append("bad_span_id")
                values = entry.get("evidence_values", [])
                if (not isinstance(values, list) or
                        any(not isinstance(value, str) or not value.strip() for value in values)):
                    errors.append("bad_distractor_support")
                    continue
                for value in values:
                    literal = any(span_id in catalog and
                                  _literal_substring(value, catalog[span_id]["exact_text"])
                                  for span_id in span_ids)
                    if not literal:
                        errors.append("bad_distractor_support")
                    distractor_value_literal_checks.append({"distractor_index":index,
                        "value":value, "literal_in_supporting_span":literal,
                        "literal_in_distractor":bool(len(distractors) == 3 and
                            _literal_substring(value, distractors[index]))})

    correct_explanation = item.get("correct_answer_explanation")
    if correct_explanation is not None and not isinstance(correct_explanation, str):
        errors.append("minimal_optional_field_invalid")
    distractor_explanations = item.get("distractor_explanations")
    if (distractor_explanations is not None and
            (not isinstance(distractor_explanations, list) or len(distractor_explanations) != 3 or
             any(not isinstance(value, str) for value in distractor_explanations))):
        errors.append("minimal_optional_field_invalid")
    reasoning_steps = item.get("reasoning_steps")
    if (reasoning_steps is not None and
            (not isinstance(reasoning_steps, list) or
             any(not isinstance(value, str) for value in reasoning_steps))):
        errors.append("minimal_optional_field_invalid")
        reasoning_steps = []
    reasoning_steps = list(reasoning_steps or []) if isinstance(reasoning_steps, list) else []

    selected_spans = {span_id for span_id in answer_span_ids if span_id in catalog}
    ordered_spans = [span for span in catalog.values() if span["span_id"] in selected_spans]
    evidence_span_ids = [span["span_id"] for span in ordered_spans]
    evidence_turn_ids = list(dict.fromkeys(span["turn_id"] for span in ordered_spans))
    evidence_session_ids = list(dict.fromkeys(span["session_id"] for span in ordered_spans))
    distinct_values = len(set(normalized_answer_values))
    reasoning_type = str((reasoning_contract or {}).get("reasoning_type") or "")
    local_difficulty = derive_local_difficulty(
        distinct_spans=len(evidence_span_ids), distinct_turns=len(evidence_turn_ids),
        distinct_sessions=len(evidence_session_ids), distinct_values=distinct_values,
        reasoning_steps=len(reasoning_steps), reasoning_type=reasoning_type)
    diagnostics.update({
        "answer_span_count":len(evidence_span_ids),
        "answer_turn_count":len(evidence_turn_ids),
        "answer_session_count":len(evidence_session_ids),
        "distinct_answer_value_count":distinct_values,
        "reasoning_step_count":len(reasoning_steps),
        "cross_session":len(evidence_session_ids) >= 2,
        "suggested_local_difficulty":local_difficulty,
        "suggested_complexity_profile":_complexity_profile(
            turns=len(evidence_turn_ids), sessions=len(evidence_session_ids)),
        "answer_value_literal_checks":answer_value_literal_checks,
        "distractor_value_literal_checks":distractor_value_literal_checks,
    })
    if errors:
        return None, sorted(set(errors)), diagnostics

    options: dict[str, str] = {}
    cursor = 0
    for letter in LETTERS:
        if letter == target_letter:
            options[letter] = str(correct)
        else:
            options[letter] = str(distractors[cursor]); cursor += 1
    explanation_values = (distractor_explanations if isinstance(distractor_explanations, list)
                          and len(distractor_explanations) == 3 else ["", "", ""])
    explanation_map: dict[str, str] = {}
    cursor = 0
    for letter in LETTERS:
        if letter == target_letter:
            explanation_map[letter] = str(correct_explanation or "")
        else:
            explanation_map[letter] = str(explanation_values[cursor]); cursor += 1
    qa = {
        "qa_id":item["qa_id"], "question":item["question"], "options":options,
        "reference_answer_letter":target_letter,
        "reference_answer_text":options[target_letter],
        "canonical_answer_text":str(correct),
        "evidence_span_ids":evidence_span_ids,
        "evidence_turn_ids":evidence_turn_ids,
        "evidence_session_ids":evidence_session_ids,
        "reasoning_steps":reasoning_steps,
        "correct_answer_explanation":str(correct_explanation or ""),
        "distractor_explanations":explanation_map,
        "answer_evidence_values":list(evidence_values or []) if isinstance(evidence_values, list) else [],
        "distractor_support":list(distractor_support or []) if isinstance(distractor_support, list) else [],
        "strict_v7_diagnostics":{"decision":"accept" if not strict_v7_reasons else "reject",
            "reasons":sorted(set(str(value) for value in strict_v7_reasons))},
        "complexity_diagnostics":diagnostics,
        "optional_field_status":dict(diagnostics["optional_field_status"]),
        "difficulty_level":"unclassified",
        "admission_mode":SPELL_STYLE_MINIMAL_MODE,
        "minimal_schema_profile":schema_profile,
        "minimal_materializer_version":MINIMAL_MATERIALIZER_VERSION,
        "span_catalog_sha256":span_catalog_sha256(span_catalog),
        "local_option_binding":True,
    }
    return qa, [], diagnostics


def build_answer_evidence_contexts(
    qa: dict[str, Any], *, span_catalog: dict[str, Any],
    frozen_dialogue: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Build evidence-only/masked turns solely from answer support span IDs."""
    catalog = span_catalog_index(span_catalog)
    span_ids = qa.get("evidence_span_ids")
    if (not isinstance(span_ids, list) or not span_ids or
            any(span_id not in catalog for span_id in span_ids)):
        raise ValueError("minimal_evidence_span_integrity_failed")
    answer_turn_ids = {catalog[span_id]["turn_id"] for span_id in span_ids}
    seen_turn_ids: set[str] = set()
    ordered: list[dict[str, Any]] = []
    for turn in frozen_dialogue:
        turn_id = str(turn.get("turn_id") or "")
        if not turn_id or turn_id in seen_turn_ids:
            raise ValueError("minimal_frozen_dialogue_integrity_failed")
        seen_turn_ids.add(turn_id); ordered.append(dict(turn))
    if not answer_turn_ids <= seen_turn_ids:
        raise ValueError("minimal_evidence_turn_integrity_failed")
    evidence_only = [turn for turn in ordered if turn["turn_id"] in answer_turn_ids]
    evidence_masked = [turn for turn in ordered if turn["turn_id"] not in answer_turn_ids]
    return {"answer_turn_ids":[turn["turn_id"] for turn in evidence_only],
        "evidence_only":evidence_only, "evidence_masked":evidence_masked,
        "context_builder_sha256":sha256_bytes(canonical_json({
            "answer_turn_ids":[turn["turn_id"] for turn in evidence_only],
            "frozen_turn_ids":[turn["turn_id"] for turn in ordered]}).encode("utf-8"))}
