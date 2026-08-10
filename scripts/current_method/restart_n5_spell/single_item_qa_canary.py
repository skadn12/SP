from __future__ import annotations

import inspect
import json
import re
from typing import Any, Mapping, Sequence

from .contrastive_distractor_canary import materialize_contrastive_candidate
from .core import LETTERS, canonical_json, sha256_bytes
from .core_qa_admission import (
    CORE_QA_JSON_SCHEMA,
    CORE_QA_SCHEMA_VERSION,
    core_qa_schema_sha256,
    validate_core_qa_schema,
)
from .lineage_distractor_writer import validate_distractor_object
from .questioner_contract import span_catalog_index, span_catalog_sha256


SINGLE_ITEM_QA_CANARY_VERSION = "single_item_qa_format_canary_v1"
SINGLE_ITEM_QA_PROMPT_VERSION = "spell_single_item_core_qa_writer_v1"
SINGLE_ITEM_QA_SCHEMA_VERSION = CORE_QA_SCHEMA_VERSION
SINGLE_ITEM_QA_ID_VERSION = "single_item_program_qa_id_v1"
SINGLE_ITEM_ANSWER_PLACEMENT_VERSION = "single_item_answer_placement_v1"
SINGLE_ITEM_QA_MAX_COMPLETION_TOKENS = 1024
SINGLE_ITEM_QA_MIN_SUPPORT_SPANS = 1
SINGLE_ITEM_QA_MAX_SUPPORT_SPANS = 6
DISTRACTOR_SEMANTIC_ATTEMPT_CAP = 1


SINGLE_ITEM_QA_JSON_SCHEMA = CORE_QA_JSON_SCHEMA


def single_item_qa_schema_sha256() -> str:
    return core_qa_schema_sha256()


def single_item_qa_prompt_sha256() -> str:
    material = canonical_json({
        "prompt_version": SINGLE_ITEM_QA_PROMPT_VERSION,
        "schema_version": SINGLE_ITEM_QA_SCHEMA_VERSION,
        "schema_sha256": single_item_qa_schema_sha256(),
        "prompt_source": inspect.getsource(build_single_item_questioner_prompt),
        "compact_view_source": inspect.getsource(build_compact_dialogue_view),
    })
    return sha256_bytes(material.encode("utf-8"))


def single_item_qa_protocol_metadata() -> dict[str, Any]:
    """Return immutable protocol identifiers without embedding dialogue content."""
    return {
        "canary_version": SINGLE_ITEM_QA_CANARY_VERSION,
        "prompt_version": SINGLE_ITEM_QA_PROMPT_VERSION,
        "prompt_sha256": single_item_qa_prompt_sha256(),
        "schema_version": SINGLE_ITEM_QA_SCHEMA_VERSION,
        "schema_sha256": single_item_qa_schema_sha256(),
        "max_completion_tokens": SINGLE_ITEM_QA_MAX_COMPLETION_TOKENS,
        "support_span_bounds": {
            "min_items": SINGLE_ITEM_QA_MIN_SUPPORT_SPANS,
            "max_items": SINGLE_ITEM_QA_MAX_SUPPORT_SPANS,
        },
        "model_output_fields": list(SINGLE_ITEM_QA_JSON_SCHEMA["required"]),
        "qa_id_source": "program",
        "distractor_semantic_attempt_cap": DISTRACTOR_SEMANTIC_ATTEMPT_CAP,
    }


def build_compact_dialogue_view(span_catalog: Mapping[str, Any]) -> str:
    """Render each authoritative whole-turn span exactly once in catalog order."""
    catalog = span_catalog_index(dict(span_catalog))
    rows = []
    for span in catalog.values():
        rows.append(json.dumps({
            "span_id": span["span_id"],
            "turn_id": span["turn_id"],
            "session_id": span["session_id"],
            "speaker": span["speaker"],
            "text": span["exact_text"],
        }, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(rows)


def build_single_item_questioner_prompt(
    *,
    span_catalog: Mapping[str, Any],
    lineage_id: str,
) -> str:
    """Build the full-dialogue, single-object Questioner request."""
    lineage = str(lineage_id or "").strip()
    if not lineage:
        raise ValueError("single_item_lineage_id_invalid")
    compact_view = build_compact_dialogue_view(span_catalog)
    rules = [
        "Generate exactly one natural question and its canonical correct answer from the frozen dialogue below.",
        "Use only facts supported by the frozen dialogue; preserve the correct subject, time, event, and relation.",
        "Select the minimal sufficient evidence set using 1 to 6 listed span IDs.",
        "Every answer_support_span_id must be copied exactly from the frozen dialogue rows.",
        "Do not generate answer choices, distractors, explanations, reasoning steps, an answer letter, or a QA identifier.",
        "Return exactly the three fields in the schema. Additional fields are invalid.",
        "Return one strict JSON object only, with no Markdown fence, preamble, commentary, or trailing text.",
    ]
    return "\n\n".join([
        f"PROMPT_VERSION: {SINGLE_ITEM_QA_PROMPT_VERSION}",
        f"SOURCE_LINEAGE_ID: {lineage}",
        "SINGLE-ITEM CORE QA RULES\n- " + "\n- ".join(rules),
        "FULL FROZEN DIALOGUE\n" + compact_view,
        "OUTPUT JSON SCHEMA\n" + json.dumps(
            SINGLE_ITEM_QA_JSON_SCHEMA,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    ])


def validate_single_item_core_object(
    raw: Any,
    *,
    span_catalog: Mapping[str, Any],
) -> dict[str, Any]:
    """Adapt the canonical core schema validator to the canary task labels."""
    try:
        result = validate_core_qa_schema(raw, span_catalog=dict(span_catalog))
    except ValueError as error:
        return {
            "decision": "single_item_questioner_schema_invalid",
            "reasons": [f"core_qa_span_catalog_invalid:{error}"],
            "core_qa": None,
        }
    core_qa = result.get("core_qa")
    if result.get("decision") != "core_qa_schema_accept" or core_qa is None:
        return {
            **result,
            "decision": "single_item_questioner_schema_invalid",
            "core_qa": None,
            "core_schema_decision": result.get("decision"),
        }
    return {
        **result,
        "decision": "single_item_questioner_accept",
        "core_qa": core_qa,
        "core_schema_decision": result.get("decision"),
        **core_qa,
    }


def _lineage_slug(lineage_id: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(lineage_id).strip().casefold()).strip("_")
    if not slug:
        raise ValueError("single_item_lineage_id_invalid")
    return slug


def freeze_core_qa(
    raw: Any,
    *,
    lineage_id: str,
    sample_index: int,
    span_catalog: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach local-only fields after the model's exact core object is accepted."""
    if (not isinstance(sample_index, int) or isinstance(sample_index, bool) or
            sample_index < 0):
        raise ValueError("single_item_sample_index_invalid")
    validation = validate_single_item_core_object(raw, span_catalog=span_catalog)
    if validation["decision"] != "single_item_questioner_accept":
        raise ValueError("single_item_core_object_not_accepted")
    core = validation["core_qa"]
    return {
        "qa_id": f"{_lineage_slug(lineage_id)}_single_item_{sample_index + 1:03d}",
        "question": core["question"],
        "correct_answer_text": core["correct_answer_text"],
        "answer_support_span_ids": list(core["answer_support_span_ids"]),
        "reasoning_steps": [],
    }


def frozen_core_qa_metadata(
    *,
    frozen_core_qa: Mapping[str, Any],
    lineage_id: str,
    sample_index: int,
    span_catalog: Mapping[str, Any],
) -> dict[str, Any]:
    """Record local ID provenance separately from the minimal QA payload."""
    expected = freeze_core_qa(
        {
            "question": frozen_core_qa.get("question"),
            "correct_answer_text": frozen_core_qa.get("correct_answer_text"),
            "answer_support_span_ids": frozen_core_qa.get(
                "answer_support_span_ids"),
        },
        lineage_id=lineage_id,
        sample_index=sample_index,
        span_catalog=span_catalog,
    )
    if canonical_json(dict(frozen_core_qa)) != canonical_json(expected):
        raise ValueError("frozen_core_qa_integrity_mismatch")
    seed = canonical_json({
        "version": SINGLE_ITEM_QA_ID_VERSION,
        "lineage_id": lineage_id,
        "sample_index": sample_index,
    })
    return {
        "qa_id_version": SINGLE_ITEM_QA_ID_VERSION,
        "qa_id": expected["qa_id"],
        "qa_id_seed_sha256": sha256_bytes(seed.encode("utf-8")),
        "qa_id_content_dependent": False,
        "core_qa_sha256": sha256_bytes(canonical_json(expected).encode("utf-8")),
        "span_catalog_sha256": span_catalog_sha256(dict(span_catalog)),
    }


def deterministic_answer_placement(
    lineage_id: str,
    sample_index: int,
) -> dict[str, Any]:
    """Choose a reproducible answer slot without inspecting QA or option text."""
    if (not isinstance(sample_index, int) or isinstance(sample_index, bool) or
            sample_index < 0):
        raise ValueError("single_item_sample_index_invalid")
    lineage = str(lineage_id or "").strip()
    if not lineage:
        raise ValueError("single_item_lineage_id_invalid")
    seed_material = canonical_json({
        "version": SINGLE_ITEM_ANSWER_PLACEMENT_VERSION,
        "lineage_id": lineage,
        "sample_index": sample_index,
    })
    digest = sha256_bytes(seed_material.encode("utf-8"))
    seed = int(digest[:8], 16)
    return {
        "answer_placement_version": SINGLE_ITEM_ANSWER_PLACEMENT_VERSION,
        "placement_seed": seed,
        "target_letter": LETTERS[seed % len(LETTERS)],
        "content_dependent": False,
    }


def materialize_single_item_mcq(
    *,
    core_qa: Mapping[str, Any],
    distractors: Sequence[str],
    placement: Mapping[str, Any],
    span_catalog: Mapping[str, Any],
) -> dict[str, Any]:
    """Delegate the full MCQ to the existing authoritative minimal materializer."""
    target_letter = placement.get("target_letter")
    if (target_letter not in LETTERS or
            placement.get("content_dependent") is not False):
        raise ValueError("single_item_answer_placement_invalid")
    validation = validate_distractor_object(
        {"distractors": list(distractors)},
        correct_answer_text=str(core_qa.get("correct_answer_text") or ""),
    )
    if validation["decision"] != "distractor_accept":
        raise ValueError(f"distractor_not_accepted:{validation['reasons']}")
    candidate = materialize_contrastive_candidate(
        base_qa=core_qa,
        distractors=validation["distractors"],
        target_letter=str(target_letter),
        span_catalog=dict(span_catalog),
    )
    return {
        **candidate,
        "placement": dict(placement),
        "distractor_semantic_attempt_count": 1,
        "distractor_semantic_retry_allowed": False,
    }


def classify_candidate_disposition(final_decision: str | None) -> dict[str, Any]:
    decision = str(final_decision or "")
    if decision == "bad_ground":
        return {
            "final_decision": decision,
            "disposition": "discard_whole_qa_and_resample",
            "discard_whole_qa": True,
            "resample_required": True,
            "same_qa_distractor_retry_allowed": False,
        }
    return {
        "final_decision": decision,
        "disposition": "terminal_no_same_qa_distractor_retry",
        "discard_whole_qa": decision not in {"retain_frontier", ""},
        "resample_required": False,
        "same_qa_distractor_retry_allowed": False,
    }
