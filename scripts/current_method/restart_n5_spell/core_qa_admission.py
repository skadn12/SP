from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core import canonical_json, sha256_bytes
from .questioner_contract import span_catalog_index, span_catalog_sha256


CORE_QA_SCHEMA_VERSION = "single_item_core_qa_schema_v1"
CORE_QA_ADMISSION_VERSION = "single_item_core_qa_mechanical_admission_v1"
CORE_QA_OBJECT_VERSION = "single_item_core_qa_object_v1"

CORE_QA_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "question",
        "correct_answer_text",
        "answer_support_span_ids",
    ],
    "properties": {
        "question": {"type": "string", "minLength": 1},
        "correct_answer_text": {"type": "string", "minLength": 1},
        "answer_support_span_ids": {
            "type": "array",
            "minItems": 1,
            "maxItems": 6,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
    },
}

_REQUIRED_FIELDS = frozenset(CORE_QA_JSON_SCHEMA["required"])
_ARTIFACT_MARKERS = (
    "```",
    "<|im_start|>",
    "<|im_end|>",
    "\x00",
    "[insert answer]",
    "<answer>",
    "lorem ipsum",
    "placeholder",
    "tbd",
    "todo",
)
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "because", "been", "but",
    "by", "did", "do", "does", "for", "from", "had", "has", "have", "he",
    "her", "hers", "him", "his", "how", "i", "in", "is", "it", "its", "me",
    "my", "of", "on", "or", "our", "ours", "she", "so", "that", "the",
    "their", "theirs", "them", "they", "this", "to", "us", "was", "we",
    "were", "what", "when", "where", "which", "who", "why", "will", "with",
    "would", "you", "your", "yours",
})
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
    "first": "ordinal:1", "second": "ordinal:2", "third": "ordinal:3",
    "fourth": "ordinal:4", "fifth": "ordinal:5",
}
_ABSOLUTE_TIME_WORDS = frozenset({
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "monday", "tuesday",
    "wednesday", "thursday", "friday", "saturday", "sunday", "morning",
    "afternoon", "evening", "night", "noon", "midnight", "today", "tomorrow",
    "yesterday",
})
_RELATIVE_TIME_WORDS = frozenset({
    "after", "before", "earlier", "eventually", "finally", "first", "following",
    "initially", "later", "next", "previously", "prior", "subsequently", "then",
})
_CLOCK_OR_DATE_RE = re.compile(
    r"(?<!\w)(?:\d{1,4}(?:[:/\-]\d{1,2}){1,2}(?:\s*(?:a\.?m\.?|p\.?m\.?))?)(?!\w)",
    flags=re.IGNORECASE,
)
_DIGIT_RE = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)")
_WORD_RE = re.compile(r"[^\W_]+(?:['\u2019][^\W_]+)?", flags=re.UNICODE)


def core_qa_schema_sha256() -> str:
    return sha256_bytes(canonical_json(CORE_QA_JSON_SCHEMA).encode("utf-8"))


def core_qa_validator_sha256() -> str:
    """Hash the complete mechanical validator implementation used by a run."""
    return sha256_bytes(Path(__file__).read_bytes())


def _normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text.strip()).casefold()


def _words(value: Any) -> list[str]:
    return [token.casefold().replace("\u2019", "'") for token in
            _WORD_RE.findall(unicodedata.normalize("NFKC", str(value or "")))]


def _meaningful_tokens(value: Any) -> set[str]:
    return {token for token in _words(value)
            if token not in _STOPWORDS and len(token) >= 2}


def _has_artifact(value: str) -> bool:
    normalized = _normalize(value)
    return any(marker in value or marker in normalized for marker in _ARTIFACT_MARKERS)


def _phrase_occurs(phrase: str, text: str) -> bool:
    phrase_words = _words(phrase)
    text_words = _words(text)
    if not phrase_words or len(phrase_words) > len(text_words):
        return False
    width = len(phrase_words)
    return any(text_words[index:index + width] == phrase_words
               for index in range(len(text_words) - width + 1))


def _literal_keys(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    keys = {f"literal:{match.group(0).replace(' ', '')}"
            for match in _CLOCK_OR_DATE_RE.finditer(normalized)}
    for match in _DIGIT_RE.finditer(normalized):
        keys.add(f"number:{match.group(0).replace(',', '')}")
    for word in _words(normalized):
        if word in _NUMBER_WORDS:
            keys.add(f"number:{_NUMBER_WORDS[word]}")
        if word in _ABSOLUTE_TIME_WORDS:
            keys.add(f"time:{word}")
    return keys


def _explicit_catalog_speakers(text: str,
                               catalog: Mapping[str, Mapping[str, str]]) -> list[str]:
    speakers = sorted({span["speaker"] for span in catalog.values()},
                      key=lambda value: (_normalize(value), value))
    return [speaker for speaker in speakers if _phrase_occurs(speaker, text)]


def _selected_speaker_is_grounded(speaker: str,
                                  selected: Sequence[Mapping[str, str]]) -> bool:
    return any(_normalize(span["speaker"]) == _normalize(speaker) or
               _phrase_occurs(speaker, span["exact_text"])
               for span in selected)


def validate_core_qa_structure(raw: Any, *,
                               span_catalog: dict[str, Any]) -> dict[str, Any]:
    """Validate only the strict three-field model protocol.

    Schema and count failures are intentionally separate from admission failures so
    the caller may apply one format-only retry without retrying semantic content.
    """
    catalog = span_catalog_index(span_catalog)
    reasons: list[str] = []
    if not isinstance(raw, dict):
        return {"decision": "core_qa_schema_invalid",
                "reasons": ["core_qa_not_object"], "core_qa_candidate": None}
    if set(raw) != _REQUIRED_FIELDS:
        reasons.append("core_qa_field_set_invalid")
    question = raw.get("question")
    answer = raw.get("correct_answer_text")
    if not isinstance(question, str) or not question.strip():
        reasons.append("core_qa_question_invalid")
    elif _has_artifact(question):
        reasons.append("core_qa_question_artifact")
    if not isinstance(answer, str) or not answer.strip():
        reasons.append("core_qa_answer_invalid")
    elif _has_artifact(answer):
        reasons.append("core_qa_answer_artifact")
    support = raw.get("answer_support_span_ids")
    support_items_valid = (isinstance(support, list) and
        all(isinstance(span_id, str) and bool(span_id.strip())
            for span_id in support))
    if not support_items_valid:
        reasons.append("core_qa_support_invalid")
        support = support if isinstance(support, list) else []
    if not 1 <= len(support) <= 6:
        reasons.append("core_qa_support_count_invalid")
    if support_items_valid:
        if len(set(support)) != len(support):
            reasons.append("core_qa_duplicate_support")
        if any(span_id not in catalog for span_id in support):
            reasons.append("core_qa_unknown_support_span")
    if reasons:
        return {"decision": "core_qa_schema_invalid",
                "reasons": list(dict.fromkeys(reasons)),
                "core_qa_candidate": None}
    order = {span_id: index for index, span_id in enumerate(catalog)}
    candidate = {
        "question": question.strip(),
        "correct_answer_text": answer.strip(),
        "answer_support_span_ids": sorted(support, key=order.__getitem__),
    }
    return {"decision": "core_qa_schema_accept", "reasons": [],
            "core_qa_candidate": candidate}


def validate_core_qa_schema(raw: Any, *,
                            span_catalog: dict[str, Any]) -> dict[str, Any]:
    """Runner-facing schema layer with the canonical candidate under ``core_qa``."""
    result = validate_core_qa_structure(raw, span_catalog=span_catalog)
    return {**result, "core_qa": result.get("core_qa_candidate")}


def admit_core_qa(raw: Any, *, span_catalog: dict[str, Any],
                  qa_id: str | None = None,
                  known_speakers: Sequence[str] | None = None
                  ) -> dict[str, Any]:
    """Apply a lightweight deterministic gate, not semantic entailment checking."""
    if qa_id is not None and (not isinstance(qa_id, str) or not qa_id.strip()):
        raise ValueError("core_qa_program_qa_id_invalid")
    structure = validate_core_qa_structure(raw, span_catalog=span_catalog)
    if structure["decision"] != "core_qa_schema_accept":
        return {**structure,
                "admission_version": CORE_QA_ADMISSION_VERSION,
                "admission_scope": "mechanical_only_not_entailment",
                "mechanical_checks": None}

    catalog = span_catalog_index(span_catalog)
    candidate = structure["core_qa_candidate"]
    selected = [catalog[span_id]
                for span_id in candidate["answer_support_span_ids"]]
    evidence_text = "\n".join(
        f"{span['speaker']}: {span['exact_text']}" for span in selected)
    combined_qa_text = (candidate["question"] + " " +
                        candidate["correct_answer_text"])

    if known_speakers is None:
        speaker_catalog = catalog
    else:
        normalized_known = {_normalize(value) for value in known_speakers
                            if isinstance(value, str) and value.strip()}
        speaker_catalog = {span_id: span for span_id, span in catalog.items()
                           if _normalize(span["speaker"]) in normalized_known}
    explicit_speakers = _explicit_catalog_speakers(combined_qa_text,
                                                   speaker_catalog)
    ungrounded_speakers = [speaker for speaker in explicit_speakers
                           if not _selected_speaker_is_grounded(speaker, selected)]
    requested_literals = _literal_keys(combined_qa_text)
    evidence_literals = _literal_keys(evidence_text)
    ungrounded_literals = sorted(requested_literals - evidence_literals)
    relative_cues = sorted(set(_words(combined_qa_text)) & _RELATIVE_TIME_WORDS)
    distinct_turn_ids = list(dict.fromkeys(span["turn_id"] for span in selected))
    answer_tokens = sorted(_meaningful_tokens(candidate["correct_answer_text"]))
    evidence_tokens = _meaningful_tokens(evidence_text)
    answer_anchors = sorted(set(answer_tokens) & evidence_tokens)

    checks = {
        "check_scope": "mechanical_only_not_entailment",
        "selected_span_count": len(selected),
        "selected_turn_count": len(distinct_turn_ids),
        "selected_turn_ids": distinct_turn_ids,
        "selected_session_ids": list(dict.fromkeys(
            span["session_id"] for span in selected)),
        "explicit_catalog_speakers": explicit_speakers,
        "ungrounded_explicit_speakers": ungrounded_speakers,
        "subject_speaker_grounded": not ungrounded_speakers,
        "requested_number_or_absolute_time_literals": sorted(requested_literals),
        "evidence_number_or_absolute_time_literals": sorted(evidence_literals),
        "ungrounded_number_or_absolute_time_literals": ungrounded_literals,
        "number_and_absolute_time_grounded": not ungrounded_literals,
        "relative_temporal_cues": relative_cues,
        "relative_temporal_turn_requirement_met": (
            not relative_cues or len(distinct_turn_ids) >= 2),
        "answer_meaningful_tokens": answer_tokens,
        "answer_evidence_lexical_anchors": answer_anchors,
        "answer_has_evidence_lexical_anchor": bool(answer_anchors),
        "semantic_entailment_verified": False,
    }
    reasons: list[str] = []
    if ungrounded_speakers:
        reasons.append("core_qa_subject_not_grounded")
    if ungrounded_literals:
        reasons.append("core_qa_number_or_time_not_grounded")
    if relative_cues and len(distinct_turn_ids) < 2:
        reasons.append("core_qa_relative_time_requires_multiple_turns")
    if not answer_anchors:
        reasons.append("core_qa_answer_lexical_anchor_missing")

    core_qa = {
        "question": candidate["question"],
        "correct_answer_text": candidate["correct_answer_text"],
        "answer_support_span_ids": list(candidate["answer_support_span_ids"]),
    }
    if qa_id is not None:
        core_qa = {"qa_id": qa_id.strip(), **core_qa}
    core_qa_sha = sha256_bytes(canonical_json(core_qa).encode("utf-8"))
    return {
        "decision": "core_qa_accept" if not reasons else "core_qa_reject",
        "reasons": reasons,
        "admission_version": CORE_QA_ADMISSION_VERSION,
        "admission_scope": "mechanical_only_not_entailment",
        "schema_version": CORE_QA_SCHEMA_VERSION,
        "schema_sha256": core_qa_schema_sha256(),
        "validator_sha256": core_qa_validator_sha256(),
        "source_span_catalog_sha256": span_catalog_sha256(span_catalog),
        "core_qa_object_version": CORE_QA_OBJECT_VERSION,
        "core_qa": core_qa if not reasons else None,
        "rejected_core_qa": core_qa if reasons else None,
        "core_qa_sha256": core_qa_sha,
        "mechanical_checks": checks,
        "selected_evidence": [dict(span) for span in selected],
    }
