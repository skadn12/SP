from __future__ import annotations

from typing import Any, Sequence

from .core import canonical_json, sha256_bytes


SPAN_CATALOG_VERSION = "dialogue_span_catalog_v1"
REASONING_CONTRACT_VERSION = "questioner_reasoning_contract_v1"
QUESTIONER_SCHEMA_VERSION = "questioner_unlabeled_span_ids_v1"

STRUCTURAL_SUPPORT_ROLES = frozenset({
    "answer_value", "supporting_fact", "temporal_relation", "causal_relation",
    "ordering_relation",
})

DEFAULT_REASONING_TYPES = (
    "multi_turn_combination",
    "cross_session_relation",
    "temporal_or_causal",
)


def build_span_catalog(frozen_dialogue: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Build one deterministic whole-turn span for every immutable dialogue turn."""
    spans: list[dict[str, str]] = []
    seen_turn_ids: set[str] = set()
    seen_span_ids: set[str] = set()
    for turn in frozen_dialogue:
        if not isinstance(turn, dict):
            raise ValueError("span_catalog_invalid_turn")
        turn_id = str(turn.get("turn_id") or "").strip()
        speaker = str(turn.get("speaker") or "").strip()
        exact_text = str(turn.get("text") or "")
        session_id = str(turn.get("session_id") or
                         (turn_id.split(":", 1)[0] if ":" in turn_id else "parent")).strip()
        if not turn_id or turn_id in seen_turn_ids or not speaker or not exact_text or not session_id:
            raise ValueError("span_catalog_invalid_turn")
        span_id = f"{turn_id}#S0001"
        if span_id in seen_span_ids:
            raise ValueError("span_catalog_duplicate_span_id")
        spans.append({"span_id":span_id, "turn_id":turn_id, "session_id":session_id,
                      "speaker":speaker, "exact_text":exact_text})
        seen_turn_ids.add(turn_id); seen_span_ids.add(span_id)
    return {"span_catalog_version":SPAN_CATALOG_VERSION, "spans":spans}


def span_catalog_index(catalog: dict[str, Any]) -> dict[str, dict[str, str]]:
    if catalog.get("span_catalog_version") != SPAN_CATALOG_VERSION:
        raise ValueError("span_catalog_version_invalid")
    spans = catalog.get("spans")
    if not isinstance(spans, list) or not spans:
        raise ValueError("span_catalog_empty")
    result: dict[str, dict[str, str]] = {}
    for span in spans:
        if (not isinstance(span, dict) or
                set(span) != {"span_id", "turn_id", "session_id", "speaker", "exact_text"} or
                any(not isinstance(span.get(key), str) or not span[key]
                    for key in ("span_id", "turn_id", "session_id", "speaker", "exact_text")) or
                span["span_id"] in result):
            raise ValueError("span_catalog_invalid_span")
        result[span["span_id"]] = dict(span)
    return result


def span_catalog_sha256(catalog: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(catalog).encode("utf-8"))


def default_reasoning_contracts(target_letters: Sequence[str]) -> list[dict[str, Any]]:
    letters = tuple(str(x).upper() for x in target_letters)
    if len(letters) != 3 or any(letter not in "ABCD" for letter in letters):
        raise ValueError("questioner_contract_requires_three_target_letters")
    common = {"reasoning_contract_version":REASONING_CONTRACT_VERSION,
              "minimum_distinct_answer_spans":2,
              "minimum_distinct_answer_turns":2,
              "minimum_distinct_evidence_values":2,
              "minimum_reasoning_steps":2}
    return [
        {**common, "item_index":0, "target_letter":letters[0],
         "reasoning_type":"multi_turn_combination",
         "minimum_distinct_answer_sessions":1},
        {**common, "item_index":1, "target_letter":letters[1],
         "reasoning_type":"cross_session_relation",
         "minimum_distinct_answer_sessions":2},
        {**common, "item_index":2, "target_letter":letters[2],
         "reasoning_type":"temporal_or_causal",
         "minimum_distinct_answer_sessions":2},
    ]


def validate_reasoning_contracts(contracts: Sequence[dict[str, Any]]) -> None:
    if len(contracts) != 3:
        raise ValueError("questioner_reasoning_contract_invalid")
    expected = default_reasoning_contracts(
        tuple(str(contract.get("target_letter") or "") for contract in contracts))
    if canonical_json(list(contracts)) != canonical_json(expected):
        raise ValueError("questioner_reasoning_contract_invalid")


def reasoning_contract_sha256(contracts: Sequence[dict[str, Any]]) -> str:
    validate_reasoning_contracts(contracts)
    return sha256_bytes(canonical_json(list(contracts)).encode("utf-8"))


def single_reasoning_contract_sha256(contract: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(contract).encode("utf-8"))


def derive_local_difficulty(*, distinct_spans: int, distinct_turns: int,
                            distinct_sessions: int, distinct_values: int,
                            reasoning_steps: int, reasoning_type: str) -> str | None:
    higher_structure = reasoning_type in {
        "temporal_or_causal", "temporal_relation", "causal_relation", "ordering_relation"}
    if (distinct_spans >= 3 and distinct_turns >= 3 and distinct_sessions >= 2 and
            distinct_values >= 3 and reasoning_steps >= 3 and higher_structure):
        return "L3"
    if (distinct_spans >= 2 and distinct_turns >= 2 and distinct_sessions >= 2 and
            distinct_values >= 2 and reasoning_steps >= 2):
        return "L2"
    if (distinct_spans >= 2 and distinct_turns >= 2 and distinct_values >= 2 and
            reasoning_steps >= 2):
        return "L1"
    return None


def contract_is_satisfied(contract: dict[str, Any], *, distinct_spans: int,
                          distinct_turns: int, distinct_sessions: int,
                          distinct_values: int, reasoning_steps: int) -> bool:
    return all((
        distinct_spans >= int(contract["minimum_distinct_answer_spans"]),
        distinct_turns >= int(contract["minimum_distinct_answer_turns"]),
        distinct_sessions >= int(contract["minimum_distinct_answer_sessions"]),
        distinct_values >= int(contract["minimum_distinct_evidence_values"]),
        reasoning_steps >= int(contract["minimum_reasoning_steps"]),
    ))


def model_span_catalog_view(catalog: dict[str, Any]) -> list[dict[str, str]]:
    """Compact prompt view; local turn/session mappings stay outside the model protocol."""
    return [{"span_id":span["span_id"], "speaker":span["speaker"],
             "exact_text":span["exact_text"]}
            for span in span_catalog_index(catalog).values()]


def model_reasoning_contract_view(contracts: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    validate_reasoning_contracts(contracts)
    return [{key:value for key,value in contract.items() if key != "target_letter"}
            for contract in contracts]
