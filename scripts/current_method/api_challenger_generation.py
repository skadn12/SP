from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

try:
    from current_method import seed_audit
except ModuleNotFoundError:
    from scripts.current_method import seed_audit


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
OLD_METHOD_SCRIPT = SCRIPTS / "old_method/run_old_improved_gate400_sample0_5.py"
REQUIRED_CHALLENGER_KEYS = ("attack_plan", "graph_update", "dialogue", "qa")


def _load_old_gate_module() -> Any:
    module_name = "current_method_reused_gate400_api_generation"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, OLD_METHOD_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load current-method API generation source: {OLD_METHOD_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


old_gate = _load_old_gate_module()
base = old_gate.base
_ORIGINAL_BASE_BUILD_QA_AND_GRAPH = getattr(base, "build_qa_and_graph", None)
_ORIGINAL_OLD_GATE_BUILD_REWRITE_PROMPT = getattr(old_gate, "build_rewrite_prompt", None)
LETTERS = getattr(base, "LETTERS", "ABCDEFGHIJKLMNOPQRSTUVWXYZ")


CURRENT_METHOD_GRAPH_VALUE_KEYS = (
    "value",
    "to",
    "restored_state",
    "cancelled_state",
    "invalidated_statement",
    "final_true_state",
    "planned_state",
    "activation_state",
)
COMMON_TURN_ID_FIELD_NAMES = (
    "turn_ids",
    "evidence_turn_ids",
    "activation_turn_ids",
    "invalidation_turn_ids",
    "cancellation_turn_ids",
    "restoration_turn_ids",
    "event_turn_ids",
    "turn_id",
)
TURN_COVERAGE_CLEANUP_WARNING_PREFIX = "turn_coverage_cleanup"
GRAPH_DIALOGUE_CONSISTENCY_WARNING = "graph_dialogue_consistency_failed"
TURN_COVERAGE_CLEANUP_RESET_STATS_KEY = "_turn_coverage_cleanup_reset_stats"


def _normalized_dialogue_turn(turn: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(turn, dict):
        return None
    turn_id = _legacy_turn_id(turn)
    if not turn_id:
        return None
    return {
        "dia_id": turn_id,
        "turn_id": turn_id,
        "speaker": base.normalize_space(turn.get("speaker") or "Speaker"),
        "text": base.normalize_space(turn.get("text")),
    }


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_challenger_config(config_path: str | Path) -> dict[str, str | None]:
    return old_gate.load_config_payload(config_path).get("api", {}) if hasattr(old_gate, "load_config_payload") else {}


def parse_json_response(text: str) -> tuple[dict[str, Any] | None, bool]:
    candidate, parsed = old_gate.json_candidate_from_text(text)
    return (candidate, parsed) if isinstance(candidate, dict) or not parsed else (None, False)


def require_payload_keys(payload: dict[str, Any] | None) -> dict[str, Any]:
    parsed_payload, validation_error = old_gate.validate_challenger_payload(payload)
    if parsed_payload is None:
        raise ValueError(f"Missing current_method challenger payload keys: {validation_error}")
    return parsed_payload


def call_challenger_api(args: argparse.Namespace, prompt_payload: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    return old_gate.call_challenger_api(args, prompt_payload)


def load_partial_samples(path: str | Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    return old_gate.read_partial_samples(Path(path), args)


def append_partial_sample(path: str | Path, sample: dict[str, Any]) -> None:
    old_gate.append_partial_sample(Path(path), sample)


def record_failed_api_call(path: str | Path, record: dict[str, Any]) -> None:
    old_gate.append_jsonl_record(Path(path), record)


def _normalize_turn_ids(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item or "").strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _legacy_turn_id(turn: dict[str, Any]) -> str | None:
    if not isinstance(turn, dict):
        return None
    value = str(turn.get("turn_id") or turn.get("dia_id") or "").strip()
    return value or None


def _visible_turn_ids_from_legacy_turns(turns: Any) -> set[str]:
    if not isinstance(turns, list):
        return set()
    visible_turn_ids: set[str] = set()
    for turn in turns:
        turn_id = _legacy_turn_id(turn)
        if turn_id:
            visible_turn_ids.add(turn_id)
    return visible_turn_ids


def _filter_turn_ids_to_visible(turn_ids: Any, visible_turn_ids: set[str]) -> list[str]:
    return [turn_id for turn_id in _normalize_turn_ids(turn_ids) if turn_id in visible_turn_ids]


def _filter_nodes_to_visible_turns(
    nodes: list[dict[str, Any]],
    visible_turn_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    filtered_nodes: list[dict[str, Any]] = []
    input_turn_refs = 0
    kept_turn_refs = 0
    for node in nodes:
        if not isinstance(node, dict):
            continue
        normalized_turn_ids = _normalize_turn_ids(node.get("turn_ids"))
        input_turn_refs += len(normalized_turn_ids)
        filtered_turn_ids = [turn_id for turn_id in normalized_turn_ids if turn_id in visible_turn_ids]
        kept_turn_refs += len(filtered_turn_ids)
        if not filtered_turn_ids:
            continue
        filtered_node = copy.deepcopy(node)
        filtered_node["turn_ids"] = filtered_turn_ids
        filtered_nodes.append(filtered_node)
    stats = {
        "input_nodes": len(nodes),
        "kept_nodes": len(filtered_nodes),
        "dropped_nodes": len(nodes) - len(filtered_nodes),
        "input_turn_refs": input_turn_refs,
        "kept_turn_refs": kept_turn_refs,
        "dropped_turn_refs": input_turn_refs - kept_turn_refs,
    }
    return filtered_nodes, stats


def _legacy_row_turn_coverage_summary(row: dict[str, Any]) -> dict[str, Any]:
    final_dialogue = row.get("final_dialogue") if isinstance(row.get("final_dialogue"), list) else []
    visible_turn_ids = _visible_turn_ids_from_legacy_turns(final_dialogue)
    state_chain_nodes = row.get("state_chain_nodes")
    if not isinstance(state_chain_nodes, list):
        state_chain = row.get("state_chain") if isinstance(row.get("state_chain"), dict) else {}
        state_chain_nodes = state_chain.get("nodes") if isinstance(state_chain.get("nodes"), list) else []
    node_turn_ref_count = 0
    missing_node_turn_ref_count = 0
    nodes_with_missing_refs = 0
    for node in state_chain_nodes:
        if not isinstance(node, dict):
            continue
        normalized_turn_ids = _normalize_turn_ids(node.get("turn_ids"))
        node_turn_ref_count += len(normalized_turn_ids)
        missing_turn_ids = [turn_id for turn_id in normalized_turn_ids if turn_id not in visible_turn_ids]
        missing_node_turn_ref_count += len(missing_turn_ids)
        if missing_turn_ids:
            nodes_with_missing_refs += 1
    return {
        "final_dialogue_turn_count": len(visible_turn_ids),
        "state_chain_node_count": len(state_chain_nodes),
        "node_turn_ref_count": node_turn_ref_count,
        "missing_node_turn_ref_count": missing_node_turn_ref_count,
        "nodes_with_missing_refs": nodes_with_missing_refs,
    }


def _sample_visible_turns(sample: dict[str, Any]) -> tuple[set[str], dict[str, str]]:
    dialogue = sample.get("dialogue") if isinstance(sample.get("dialogue"), dict) else {}
    turns = dialogue.get("turns")
    visible_turn_ids: set[str] = set()
    turn_text_by_id: dict[str, str] = {}
    if not isinstance(turns, list):
        return visible_turn_ids, turn_text_by_id
    for turn in turns:
        turn_id = _legacy_turn_id(turn) if isinstance(turn, dict) else None
        if not turn_id:
            continue
        visible_turn_ids.add(turn_id)
        turn_text_by_id[turn_id] = base.normalize_text((turn or {}).get("text"))
    return visible_turn_ids, turn_text_by_id


def _turn_sort_key(turn_id: str) -> tuple[Any, ...]:
    text = str(turn_id or "").strip()
    match = base.re.match(r"^([A-Za-z]+)(\d+):(\d+)$", text)
    if match:
        return (0, match.group(1), int(match.group(2)), int(match.group(3)), text)
    colon_match = base.re.match(r"^(.*?):(\d+)$", text)
    if colon_match:
        prefix = colon_match.group(1)
        try:
            suffix = int(colon_match.group(2))
        except ValueError:
            suffix = 10**9
        return (1, prefix, suffix, text)
    return (2, text)


def _token_overlap_ratio(left: str, right: str) -> float:
    left_tokens = {token for token in left.split() if token}
    right_tokens = {token for token in right.split() if token}
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = left_tokens & right_tokens
    return len(overlap) / max(1, min(len(left_tokens), len(right_tokens)))


def _text_support_score(turn_text: str, expected_text: str) -> float:
    turn_norm = base.normalize_text(turn_text)
    expected_norm = base.normalize_text(expected_text)
    if not turn_norm or not expected_norm:
        return 0.0
    if turn_norm == expected_norm:
        return 1.0
    if expected_norm in turn_norm or turn_norm in expected_norm:
        return 0.92
    overlap_tokens = len(set(turn_norm.split()) & set(expected_norm.split()))
    overlap_ratio = _token_overlap_ratio(turn_norm, expected_norm)
    if overlap_ratio >= 0.75:
        return min(0.9, overlap_ratio)
    if overlap_ratio >= 0.6 and overlap_tokens >= 2:
        return 0.8
    if overlap_ratio >= 0.5 and overlap_tokens >= 3:
        return 0.72
    return 0.0


def _text_supports_expected(turn_text: str, expected_text: str) -> bool:
    return _text_support_score(turn_text, expected_text) > 0.0


def _visible_supporting_turn_ids(
    turn_ids: Any,
    visible_turn_ids: set[str],
    turn_text_by_id: dict[str, str],
    expected_text: str | None = None,
) -> tuple[list[str], dict[str, Any]]:
    normalized_turn_ids = _normalize_turn_ids(turn_ids)
    visible_ids = [turn_id for turn_id in normalized_turn_ids if turn_id in visible_turn_ids]
    expected_norm = base.normalize_text(expected_text) if expected_text is not None else ""
    if expected_norm:
        kept_ids = [
            turn_id
            for turn_id in visible_ids
            if _text_supports_expected(turn_text_by_id.get(turn_id, ""), expected_norm)
        ]
    else:
        kept_ids = list(visible_ids)
    kept_ids = sorted(dict.fromkeys(kept_ids), key=_turn_sort_key)
    stats = {
        "input_count": len(normalized_turn_ids),
        "visible_count": len(visible_ids),
        "supporting_count": len(kept_ids),
        "dropped_missing_count": len(normalized_turn_ids) - len(visible_ids),
        "dropped_unsupported_count": len(visible_ids) - len(kept_ids),
    }
    return kept_ids, stats


def _find_visible_supporting_turns_for_text(sample: dict[str, Any], expected_text: str) -> list[dict[str, Any]]:
    if not base.normalize_text(expected_text):
        return []
    dialogue = sample.get("dialogue") if isinstance(sample.get("dialogue"), dict) else {}
    turns = dialogue.get("turns")
    if not isinstance(turns, list):
        return []
    supporting_turns: list[dict[str, Any]] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        turn_id = _legacy_turn_id(turn)
        if not turn_id:
            continue
        turn_text = base.normalize_space(turn.get("text"))
        score = _text_support_score(turn_text, expected_text)
        if score <= 0.0:
            continue
        supporting_turns.append(
            {
                "turn_id": turn_id,
                "text": turn_text,
                "score": round(score, 4),
                "sort_key": _turn_sort_key(turn_id),
            }
        )
    supporting_turns.sort(key=lambda item: (item["sort_key"], -float(item["score"]), str(item["turn_id"])))
    return supporting_turns


def _sample_state_chain_nodes(sample: dict[str, Any]) -> list[dict[str, Any]]:
    state_chain_nodes = sample.get("state_chain_nodes")
    if isinstance(state_chain_nodes, list):
        return [node for node in state_chain_nodes if isinstance(node, dict)]
    state_chain = sample.get("state_chain") if isinstance(sample.get("state_chain"), dict) else {}
    nodes = state_chain.get("nodes")
    if isinstance(nodes, list):
        return [node for node in nodes if isinstance(node, dict)]
    return []


def _normalized_candidate_text(value: Any) -> str:
    text = base.normalize_space(value)
    if len(text) < 8 or text.endswith("?"):
        return ""
    return text


def _graph_candidate_context(sample: dict[str, Any]) -> dict[str, Any]:
    graph_update = sample.get("graph_update") if isinstance(sample.get("graph_update"), dict) else {}
    state_changes = graph_update.get("state_changes")
    to_texts: set[str] = set()
    from_texts: set[str] = set()
    false_reason_type_by_text: dict[str, str] = {}
    false_reason_text_by_text: dict[str, str] = {}
    positive_reason_type_by_text: dict[str, str] = {}
    if isinstance(state_changes, list):
        for state_change in state_changes:
            if not isinstance(state_change, dict):
                continue
            to_text = base.normalize_text(state_change.get("to"))
            from_text = base.normalize_text(state_change.get("from"))
            if to_text:
                to_texts.add(to_text)
                positive_reason_type_by_text.setdefault(to_text, "state_change_to")
            if from_text:
                from_texts.add(from_text)
                false_reason_type_by_text.setdefault(from_text, "expired")
                false_reason_text_by_text.setdefault(from_text, "state replaced by a later supported value")
    invalidated_texts: set[str] = set()
    final_texts: set[str] = set()
    cancellations = graph_update.get("cancellations")
    if isinstance(cancellations, list):
        for cancellation in cancellations:
            if isinstance(cancellation, dict):
                cancelled_state = base.normalize_text(cancellation.get("cancelled_state"))
                if cancelled_state:
                    invalidated_texts.add(cancelled_state)
                    false_reason_type_by_text[cancelled_state] = "canceled"
                    false_reason_text_by_text[cancelled_state] = "explicitly canceled by a later event"
    invalidations = graph_update.get("invalidations")
    if isinstance(invalidations, list):
        for invalidation in invalidations:
            if not isinstance(invalidation, dict):
                continue
            stale_state = base.normalize_text(invalidation.get("invalidated_statement"))
            final_state = base.normalize_text(invalidation.get("final_true_state"))
            if stale_state:
                invalidated_texts.add(stale_state)
                false_reason_type_by_text[stale_state] = "invalidated"
                false_reason_text_by_text[stale_state] = "explicitly invalidated by later evidence"
            if final_state:
                final_texts.add(final_state)
                positive_reason_type_by_text[final_state] = "final_true_state"
    restorations = graph_update.get("restorations")
    if isinstance(restorations, list):
        for restoration in restorations:
            if isinstance(restoration, dict):
                restored_state = base.normalize_text(restoration.get("restored_state"))
                if restored_state:
                    final_texts.add(restored_state)
                    positive_reason_type_by_text[restored_state] = "restored"
    delayed_activations = graph_update.get("delayed_activations")
    if isinstance(delayed_activations, list):
        for activation in delayed_activations:
            if not isinstance(activation, dict):
                continue
            activation_state = base.normalize_text(activation.get("activation_state"))
            planned_state = base.normalize_text(activation.get("planned_state"))
            if activation_state:
                final_texts.add(activation_state)
                positive_reason_type_by_text[activation_state] = "delayed_activation"
            if planned_state and planned_state != activation_state:
                false_reason_type_by_text.setdefault(planned_state, "delayed_activation")
                false_reason_text_by_text.setdefault(planned_state, "planned state is not yet the active final state")
    return {
        "to_texts": to_texts,
        "from_texts": from_texts,
        "stale_from_only_texts": from_texts - to_texts,
        "invalidated_texts": invalidated_texts,
        "final_texts": final_texts,
        "false_reason_type_by_text": false_reason_type_by_text,
        "false_reason_text_by_text": false_reason_text_by_text,
        "positive_reason_type_by_text": positive_reason_type_by_text,
    }


def _candidate_source_priority(source: str) -> int:
    return {
        "graph_event": 0,
        "state_chain_node": 1,
        "state_change": 2,
        "qa_option": 3,
        "dialogue_turn": 4,
    }.get(str(source or ""), 99)


def extract_visible_event_candidates(sample: dict[str, Any]) -> list[dict[str, Any]]:
    qa = sample.get("qa") if isinstance(sample.get("qa"), dict) else {}
    graph_update = sample.get("graph_update") if isinstance(sample.get("graph_update"), dict) else {}
    visible_turn_ids, turn_text_by_id = _sample_visible_turns(sample)
    evidence_turn_ids = set(_normalize_turn_ids(qa.get("evidence_turn_ids")))
    graph_context = _graph_candidate_context(sample)
    candidates: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()

    def add_candidate(
        *,
        candidate_id: str,
        source: str,
        text: str,
        raw_turn_ids: Any,
        entity: str | None = None,
        slot: str | None = None,
        value: str | None = None,
        candidate_kind: str | None = None,
        metadata: dict[str, Any] | None = None,
        allow_scan_without_declared_turns: bool = False,
    ) -> None:
        candidate_text = _normalized_candidate_text(text or value)
        if not candidate_text:
            return
        supporting_turns = _find_visible_supporting_turns_for_text(sample, candidate_text)
        supporting_turn_id_set = {str(turn["turn_id"]) for turn in supporting_turns}
        declared_visible_turn_ids = [turn_id for turn_id in _normalize_turn_ids(raw_turn_ids) if turn_id in visible_turn_ids]
        if declared_visible_turn_ids:
            support_turn_ids = [turn_id for turn_id in declared_visible_turn_ids if turn_id in supporting_turn_id_set]
        elif allow_scan_without_declared_turns:
            support_turn_ids = [str(turn["turn_id"]) for turn in supporting_turns]
        else:
            support_turn_ids = []
        if not support_turn_ids:
            return
        support_turn_ids = sorted(dict.fromkeys(support_turn_ids), key=_turn_sort_key)
        earliest_turn_id = support_turn_ids[0]
        earliest_turn_sort_key = _turn_sort_key(earliest_turn_id)
        normalized_text = base.normalize_text(candidate_text)
        overlap_evidence_turn_ids = [turn_id for turn_id in support_turn_ids if turn_id in evidence_turn_ids]
        graph_false_reason = None
        graph_positive_reason = None
        if normalized_text in graph_context["invalidated_texts"]:
            graph_false_reason = "invalidated_or_cancelled_state"
        elif candidate_kind == "from_state" and normalized_text in graph_context["stale_from_only_texts"]:
            graph_false_reason = "replaced_from_state"
        if overlap_evidence_turn_ids:
            graph_positive_reason = "qa_evidence_overlap"
        elif normalized_text in graph_context["final_texts"]:
            graph_positive_reason = "final_or_restored_state"
        elif candidate_kind == "to_state" and normalized_text in graph_context["to_texts"]:
            graph_positive_reason = "state_change_to"
        false_reason_type = graph_context["false_reason_type_by_text"].get(normalized_text)
        false_reason_text = graph_context["false_reason_text_by_text"].get(normalized_text)
        positive_reason_type = graph_context["positive_reason_type_by_text"].get(normalized_text)
        dedupe_key = (source, normalized_text, earliest_turn_id)
        if dedupe_key in seen_keys:
            return
        seen_keys.add(dedupe_key)
        support_texts = [turn_text_by_id.get(turn_id, "") for turn_id in support_turn_ids if turn_text_by_id.get(turn_id, "")]
        candidate = {
            "candidate_id": candidate_id,
            "text": candidate_text,
            "value": value or candidate_text,
            "turn_ids": support_turn_ids,
            "earliest_turn_id": earliest_turn_id,
            "earliest_turn_sort_key": earliest_turn_sort_key,
            "source": source,
            "entity": entity,
            "slot": slot,
            "support_texts": support_texts,
            "normalized_text": normalized_text,
            "support_turn_count": len(support_turn_ids),
            "overlap_evidence_turn_ids": overlap_evidence_turn_ids,
            "overlap_evidence_count": len(overlap_evidence_turn_ids),
            "source_priority": _candidate_source_priority(source),
            "graph_false_reason": graph_false_reason,
            "graph_positive_reason": graph_positive_reason,
            "false_reason_type": false_reason_type,
            "false_reason_text": false_reason_text,
            "positive_reason_type": positive_reason_type,
        }
        if candidate_kind:
            candidate["candidate_kind"] = candidate_kind
        if metadata:
            candidate.update(metadata)
        candidates.append(candidate)

    options = qa.get("options")
    if isinstance(options, list):
        for index, option in enumerate(options):
            if not isinstance(option, dict):
                continue
            add_candidate(
                candidate_id=f"qa_option:{option.get('id') or index}",
                source="qa_option",
                text=str(option.get("text") or ""),
                raw_turn_ids=option.get("turn_ids"),
                value=str(option.get("text") or ""),
                metadata={"option_id": str(option.get("id") or index)},
            )

    events = graph_update.get("events")
    if isinstance(events, list):
        for index, event in enumerate(events):
            if not isinstance(event, dict):
                continue
            add_candidate(
                candidate_id=f"graph_event:{event.get('event_id') or index}",
                source="graph_event",
                text=_item_value(event),
                raw_turn_ids=event.get("turn_ids"),
                value=_item_value(event),
                metadata={
                    "event_id": str(event.get("event_id") or index),
                    "node_id": event.get("node_id"),
                    "status": event.get("status"),
                    "role": event.get("role"),
                },
            )

    state_changes = graph_update.get("state_changes")
    if isinstance(state_changes, list):
        for index, state_change in enumerate(state_changes):
            if not isinstance(state_change, dict):
                continue
            common = {
                "entity": state_change.get("entity"),
                "slot": state_change.get("slot"),
                "metadata": {
                    "state_change_index": index,
                    "from": state_change.get("from"),
                    "to": state_change.get("to"),
                },
            }
            add_candidate(
                candidate_id=f"state_change:{index}:to",
                source="state_change",
                text=str(state_change.get("to") or ""),
                raw_turn_ids=state_change.get("evidence_turn_ids"),
                value=str(state_change.get("to") or ""),
                entity=common["entity"],
                slot=common["slot"],
                candidate_kind="to_state",
                metadata=common["metadata"],
            )
            add_candidate(
                candidate_id=f"state_change:{index}:from",
                source="state_change",
                text=str(state_change.get("from") or ""),
                raw_turn_ids=[],
                value=str(state_change.get("from") or ""),
                entity=common["entity"],
                slot=common["slot"],
                candidate_kind="from_state",
                metadata=common["metadata"],
                allow_scan_without_declared_turns=True,
            )

    invalidations = graph_update.get("invalidations")
    if isinstance(invalidations, list):
        for index, invalidation in enumerate(invalidations):
            if not isinstance(invalidation, dict):
                continue
            add_candidate(
                candidate_id=f"invalidation:{index}:stale",
                source="graph_event",
                text=str(invalidation.get("invalidated_statement") or ""),
                raw_turn_ids=invalidation.get("invalidation_turn_ids"),
                value=str(invalidation.get("invalidated_statement") or ""),
                candidate_kind="invalidated_state",
                metadata={
                    "false_reason_type": "invalidated",
                    "false_reason_text": "explicitly invalidated by later evidence",
                },
                allow_scan_without_declared_turns=True,
            )
            add_candidate(
                candidate_id=f"invalidation:{index}:final",
                source="graph_event",
                text=str(invalidation.get("final_true_state") or ""),
                raw_turn_ids=invalidation.get("invalidation_turn_ids"),
                value=str(invalidation.get("final_true_state") or ""),
                candidate_kind="final_true_state",
                metadata={
                    "positive_reason_type": "final_true_state",
                },
                allow_scan_without_declared_turns=True,
            )

    cancellations = graph_update.get("cancellations")
    if isinstance(cancellations, list):
        for index, cancellation in enumerate(cancellations):
            if not isinstance(cancellation, dict):
                continue
            add_candidate(
                candidate_id=f"cancellation:{index}",
                source="graph_event",
                text=str(cancellation.get("cancelled_state") or ""),
                raw_turn_ids=cancellation.get("turn_ids"),
                value=str(cancellation.get("cancelled_state") or ""),
                candidate_kind="cancelled_state",
                metadata={
                    "false_reason_type": "canceled",
                    "false_reason_text": "explicitly canceled by a later event",
                },
                allow_scan_without_declared_turns=True,
            )

    restorations = graph_update.get("restorations")
    if isinstance(restorations, list):
        for index, restoration in enumerate(restorations):
            if not isinstance(restoration, dict):
                continue
            add_candidate(
                candidate_id=f"restoration:{index}",
                source="graph_event",
                text=str(restoration.get("restored_state") or ""),
                raw_turn_ids=restoration.get("turn_ids"),
                value=str(restoration.get("restored_state") or ""),
                candidate_kind="restored_state",
                metadata={
                    "positive_reason_type": "restored",
                },
                allow_scan_without_declared_turns=True,
            )

    delayed_activations = graph_update.get("delayed_activations")
    if isinstance(delayed_activations, list):
        for index, activation in enumerate(delayed_activations):
            if not isinstance(activation, dict):
                continue
            add_candidate(
                candidate_id=f"delayed_activation:{index}:planned",
                source="graph_event",
                text=str(activation.get("planned_state") or ""),
                raw_turn_ids=activation.get("activation_turn_ids"),
                value=str(activation.get("planned_state") or ""),
                candidate_kind="planned_state",
                metadata={
                    "false_reason_type": "delayed_activation",
                    "false_reason_text": "planned state is not yet the active final state",
                },
                allow_scan_without_declared_turns=True,
            )
            add_candidate(
                candidate_id=f"delayed_activation:{index}:active",
                source="graph_event",
                text=str(activation.get("activation_state") or ""),
                raw_turn_ids=activation.get("activation_turn_ids"),
                value=str(activation.get("activation_state") or ""),
                candidate_kind="activation_state",
                metadata={
                    "positive_reason_type": "delayed_activation",
                },
                allow_scan_without_declared_turns=True,
            )

    for index, node in enumerate(_sample_state_chain_nodes(sample)):
        add_candidate(
            candidate_id=f"state_chain_node:{node.get('node_id') or index}",
            source="state_chain_node",
            text=_item_value(node),
            raw_turn_ids=node.get("turn_ids"),
            value=_item_value(node),
            entity=node.get("entity"),
            slot=node.get("slot"),
            metadata={
                "node_id": node.get("node_id"),
                "status": node.get("status"),
                "role": node.get("role"),
            },
        )

    dialogue = sample.get("dialogue") if isinstance(sample.get("dialogue"), dict) else {}
    turns = dialogue.get("turns")
    if isinstance(turns, list):
        for index, turn in enumerate(turns):
            if not isinstance(turn, dict):
                continue
            turn_id = _legacy_turn_id(turn)
            if not turn_id:
                continue
            add_candidate(
                candidate_id=f"dialogue_turn:{turn_id}",
                source="dialogue_turn",
                text=str(turn.get("text") or ""),
                raw_turn_ids=[turn_id],
                value=str(turn.get("text") or ""),
                metadata={
                    "speaker": turn.get("speaker"),
                    "turn_index": index,
                },
            )

    candidates.sort(
        key=lambda candidate: (
            int(candidate.get("source_priority") or 99),
            candidate.get("earliest_turn_sort_key") or (99, ""),
            -int(candidate.get("support_turn_count") or 0),
            str(candidate.get("candidate_id") or ""),
        )
    )
    return candidates


def _dedupe_candidate_list(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for candidate in sorted(
        candidates,
        key=lambda candidate: (
            int(candidate.get("source_priority") or 99),
            candidate.get("earliest_turn_sort_key") or (99, ""),
            -int(candidate.get("support_turn_count") or 0),
            str(candidate.get("candidate_id") or ""),
        ),
    ):
        key = (
            str(candidate.get("normalized_text") or ""),
            str(candidate.get("earliest_turn_id") or ""),
        )
        if not key[0] or key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def build_generation_candidate_bank(sample: dict[str, Any]) -> dict[str, Any]:
    candidates = extract_visible_event_candidates(sample)
    ordering_candidates = _dedupe_candidate_list(
        [
            candidate
            for candidate in candidates
            if not candidate.get("false_reason_type")
        ]
    )
    positive_candidates = _dedupe_candidate_list(
        [
            candidate
            for candidate in candidates
            if not candidate.get("false_reason_type")
            and (
                candidate.get("positive_reason_type")
                or int(candidate.get("overlap_evidence_count") or 0) > 0
                or str(candidate.get("source") or "") in {"graph_event", "state_change", "state_chain_node"}
            )
        ]
    )
    negative_candidates = _dedupe_candidate_list(
        [
            candidate
            for candidate in candidates
            if candidate.get("false_reason_type") and candidate.get("false_reason_text")
        ]
    )
    ordering_distinct_turn_count = len(
        {
            str(candidate.get("earliest_turn_id") or "")
            for candidate in ordering_candidates
            if str(candidate.get("earliest_turn_id") or "")
        }
    )
    bank = {
        "ordering_candidates": [
            {
                "candidate_id": candidate.get("candidate_id"),
                "text": candidate.get("text"),
                "value": candidate.get("value"),
                "turn_ids": list(candidate.get("turn_ids") or []),
                "earliest_turn_id": candidate.get("earliest_turn_id"),
                "earliest_turn_sort_key": candidate.get("earliest_turn_sort_key"),
                "source": candidate.get("source"),
                "entity": candidate.get("entity"),
                "slot": candidate.get("slot"),
                "support_texts": list(candidate.get("support_texts") or []),
            }
            for candidate in ordering_candidates
        ],
        "multi_select_positive_candidates": [
            {
                "candidate_id": candidate.get("candidate_id"),
                "text": candidate.get("text"),
                "value": candidate.get("value"),
                "turn_ids": list(candidate.get("turn_ids") or []),
                "earliest_turn_id": candidate.get("earliest_turn_id"),
                "earliest_turn_sort_key": candidate.get("earliest_turn_sort_key"),
                "source": candidate.get("source"),
                "entity": candidate.get("entity"),
                "slot": candidate.get("slot"),
                "support_texts": list(candidate.get("support_texts") or []),
            }
            for candidate in positive_candidates
        ],
        "multi_select_negative_candidates": [
            {
                "candidate_id": candidate.get("candidate_id"),
                "text": candidate.get("text"),
                "value": candidate.get("value"),
                "turn_ids": list(candidate.get("turn_ids") or []),
                "earliest_turn_id": candidate.get("earliest_turn_id"),
                "earliest_turn_sort_key": candidate.get("earliest_turn_sort_key"),
                "source": candidate.get("source"),
                "entity": candidate.get("entity"),
                "slot": candidate.get("slot"),
                "support_texts": list(candidate.get("support_texts") or []),
                "false_reason": candidate.get("false_reason_text"),
                "false_reason_type": candidate.get("false_reason_type"),
            }
            for candidate in negative_candidates
        ],
        "ordering_ready": ordering_distinct_turn_count >= 4,
        "multi_select_ready": len(positive_candidates) >= 2 and len(negative_candidates) >= 2,
        "ordering_candidate_count": len(ordering_candidates),
        "ordering_distinct_earliest_turn_count": ordering_distinct_turn_count,
        "multi_select_positive_count": len(positive_candidates),
        "multi_select_negative_count": len(negative_candidates),
    }
    return bank


def _candidate_bank_not_ready_reasons(bank: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if int(bank.get("ordering_distinct_earliest_turn_count") or 0) < 4:
        reasons.append("ordering_insufficient_distinct_candidates")
    if int(bank.get("multi_select_positive_count") or 0) < 2:
        reasons.append("multi_select_insufficient_positive_candidates")
    if int(bank.get("multi_select_negative_count") or 0) < 2:
        reasons.append("multi_select_insufficient_safe_negative_candidates")
    return reasons


def _refresh_sample_validation(sample: dict[str, Any]) -> dict[str, Any]:
    refreshed = copy.deepcopy(sample)
    refreshed["validation"] = base.validate_sample(refreshed)
    return refreshed


def _generation_consistency_meta(sample: dict[str, Any]) -> dict[str, Any]:
    meta = sample.get("generation_consistency_repair") if isinstance(sample.get("generation_consistency_repair"), dict) else {}
    warnings = meta.get("warnings")
    if not isinstance(warnings, list):
        meta["warnings"] = []
    sample["generation_consistency_repair"] = meta
    return meta


def _register_generation_consistency_warning(meta: dict[str, Any], warning: str) -> None:
    warnings = meta.get("warnings")
    if not isinstance(warnings, list):
        warnings = []
    if warning not in warnings:
        warnings.append(warning)
    meta["warnings"] = warnings


def _mark_generation_consistency_invalid(sample: dict[str, Any], warning: str) -> None:
    meta = _generation_consistency_meta(sample)
    _register_generation_consistency_warning(meta, warning)
    meta["valid_after_visible_support_guard"] = False
    validation = sample.get("validation") if isinstance(sample.get("validation"), dict) else {}
    validation_warnings = validation.get("warnings")
    if not isinstance(validation_warnings, list):
        validation_warnings = []
    if warning not in validation_warnings:
        validation_warnings.append(warning)
    validation["warnings"] = validation_warnings
    validation["graph_dialogue_consistent"] = False
    details = validation.get("details") if isinstance(validation.get("details"), dict) else {}
    validation_checks = details.get("validation_checks") if isinstance(details.get("validation_checks"), dict) else {}
    validation_checks["graph_dialogue_consistency"] = False
    details["validation_checks"] = validation_checks
    validation["details"] = details
    sample["validation"] = validation
    sample["graph_dialogue_alignment"] = False


def _generation_programmatic_builder_meta(sample: dict[str, Any]) -> dict[str, Any]:
    meta = sample.get("generation_programmatic_builder") if isinstance(sample.get("generation_programmatic_builder"), dict) else {}
    sample["generation_programmatic_builder"] = meta
    return meta


def _select_ordering_candidates_from_bank(bank: dict[str, Any], rng: Any | None = None) -> list[dict[str, Any]]:
    raw_candidates = list(bank.get("ordering_candidates") or [])
    if len(raw_candidates) < 4:
        return []
    candidates = [copy.deepcopy(candidate) for candidate in raw_candidates if isinstance(candidate, dict)]
    candidates.sort(
        key=lambda candidate: (
            candidate.get("earliest_turn_sort_key") or (99, ""),
            int(_candidate_source_priority(str(candidate.get("source") or ""))),
            str(candidate.get("candidate_id") or ""),
        )
    )
    if rng is not None and len(candidates) > 4:
        prefix = candidates[: min(len(candidates), 8)]
        sample_size = min(4, len(prefix))
        selected = list(rng.sample(prefix, sample_size))
        selected.sort(key=lambda candidate: (candidate.get("earliest_turn_sort_key") or (99, ""), str(candidate.get("candidate_id") or "")))
        if len(selected) == 4:
            return selected
    return candidates[:4]


def _select_multi_select_candidates_from_bank(bank: dict[str, Any], rng: Any | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    positive_candidates = [copy.deepcopy(candidate) for candidate in (bank.get("multi_select_positive_candidates") or []) if isinstance(candidate, dict)]
    negative_candidates = [copy.deepcopy(candidate) for candidate in (bank.get("multi_select_negative_candidates") or []) if isinstance(candidate, dict)]
    positive_candidates.sort(
        key=lambda candidate: (
            -int(len(candidate.get("turn_ids") or [])),
            candidate.get("earliest_turn_sort_key") or (99, ""),
            str(candidate.get("candidate_id") or ""),
        )
    )
    negative_candidates.sort(
        key=lambda candidate: (
            int(_candidate_source_priority(str(candidate.get("source") or ""))),
            candidate.get("earliest_turn_sort_key") or (99, ""),
            str(candidate.get("false_reason_type") or ""),
            str(candidate.get("candidate_id") or ""),
        )
    )
    if rng is not None:
        if len(positive_candidates) > 2:
            positive_candidates = list(rng.sample(positive_candidates[: min(len(positive_candidates), 6)], 2))
        else:
            positive_candidates = positive_candidates[:2]
        if len(negative_candidates) > 2:
            negative_candidates = list(rng.sample(negative_candidates[: min(len(negative_candidates), 6)], 2))
        else:
            negative_candidates = negative_candidates[:2]
    else:
        positive_candidates = positive_candidates[:2]
        negative_candidates = negative_candidates[:2]
    positive_candidates.sort(key=lambda candidate: (candidate.get("earliest_turn_sort_key") or (99, ""), str(candidate.get("candidate_id") or "")))
    negative_candidates.sort(key=lambda candidate: (candidate.get("earliest_turn_sort_key") or (99, ""), str(candidate.get("candidate_id") or "")))
    return positive_candidates, negative_candidates


def _candidate_to_node(candidate: dict[str, Any], *, role_prefix: str, index: int) -> dict[str, Any]:
    false_reason_type = str(candidate.get("false_reason_type") or "").strip()
    status = false_reason_type or str(candidate.get("positive_reason_type") or candidate.get("source") or role_prefix)
    return {
        "node_id": str(candidate.get("candidate_id") or f"{role_prefix}_{index}"),
        "role": str(candidate.get("source") or f"{role_prefix}_{index}"),
        "label": str(candidate.get("source") or role_prefix),
        "value": str(candidate.get("value") or candidate.get("text") or ""),
        "turn_ids": list(candidate.get("turn_ids") or []),
        "status": status,
    }


def _scaffold_sample_from_bundle(bundle: Any) -> dict[str, Any]:
    return {
        "dialogue": {"turns": [copy.deepcopy(turn) for turn in getattr(bundle, "turns", []) if isinstance(turn, dict)]},
        "graph_update": copy.deepcopy((getattr(bundle, "row", {}) or {}).get("graph_update") or {}),
        "qa": copy.deepcopy((getattr(bundle, "row", {}) or {}).get("qa") or {}),
        "state_chain_nodes": copy.deepcopy((getattr(bundle, "row", {}) or {}).get("state_chain_nodes") or []),
        "state_chain": copy.deepcopy((getattr(bundle, "row", {}) or {}).get("state_chain") or {}),
    }


def _source_turn_lookup_from_bundle(bundle: Any) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for turn in getattr(bundle, "turns", []) or []:
        normalized = _normalized_dialogue_turn(turn) if isinstance(turn, dict) else None
        if normalized:
            lookup[normalized["turn_id"]] = normalized
    return lookup


def _select_dialogue_turns_for_candidate_bank(
    bundle: Any,
    selected_turn_ids: list[str],
    rng: Any,
    *,
    radius: int = 1,
    max_turns: int = 20,
) -> list[dict[str, Any]]:
    turns = base.turn_window(bundle, selected_turn_ids, rng, radius=radius, max_turns=max_turns)
    expanded_turns, _stats = _expand_dialogue_turns_with_source_refs(
        turns,
        selected_turn_ids,
        _source_turn_lookup_from_bundle(bundle),
    )
    return expanded_turns


def _structured_candidate_pool(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed_sources = {"graph_event", "state_change", "state_chain_node", "qa_option"}
    return [candidate for candidate in candidates if str(candidate.get("source") or "") in allowed_sources]


def try_rebuild_ordering_from_visible_candidates(sample: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    rebuilt = copy.deepcopy(sample)
    qa = rebuilt.get("qa") if isinstance(rebuilt.get("qa"), dict) else {}
    question_type = str(qa.get("question_type") or "unknown")
    report: dict[str, Any] = {
        "sample_id": rebuilt.get("sample_id"),
        "question_type": question_type,
        "rebuilt": False,
        "reason": None,
        "selected_candidate_ids": [],
        "visible_order": [],
    }
    if question_type != "ordering":
        report["reason"] = "question_type_not_ordering"
        return rebuilt, report
    meta = _generation_programmatic_builder_meta(rebuilt)
    bank = sample.get("generation_candidate_bank") if isinstance(sample.get("generation_candidate_bank"), dict) else build_generation_candidate_bank(rebuilt)
    selected_candidates = _select_ordering_candidates_from_bank(bank)
    if len(selected_candidates) < 4:
        meta.update(
            {
                "applied": False,
                "question_type": "ordering",
                "not_rebuilt_reason": "ordering_insufficient_distinct_visible_candidates",
                "candidate_count": int(bank.get("ordering_candidate_count") or 0),
                "selected_candidate_ids": [str(candidate.get("candidate_id") or "") for candidate in selected_candidates],
            }
        )
        report["reason"] = "ordering_insufficient_distinct_visible_candidates"
        report["selected_candidate_ids"] = meta["selected_candidate_ids"]
        rebuilt["generation_candidate_bank"] = bank
        return rebuilt, report
    option_candidates = sorted(
        selected_candidates,
        key=lambda candidate: (
            str(candidate.get("normalized_text") or ""),
            candidate.get("earliest_turn_sort_key") or (99, ""),
            str(candidate.get("candidate_id") or ""),
        ),
    )
    options: list[dict[str, Any]] = []
    label_by_candidate_id: dict[str, str] = {}
    for index, candidate in enumerate(option_candidates):
        label = LETTERS[index]
        candidate_id = str(candidate.get("candidate_id") or f"candidate_{index}")
        label_by_candidate_id[candidate_id] = label
        options.append(
            {
                "id": label,
                "text": str(candidate.get("text") or candidate.get("value") or ""),
                "turn_ids": list(candidate.get("turn_ids") or []),
            }
        )
    visible_order = [
        label_by_candidate_id[str(candidate.get("candidate_id") or "")]
        for candidate in sorted(
            selected_candidates,
            key=lambda candidate: (
                candidate.get("earliest_turn_sort_key") or (99, ""),
                str(candidate.get("candidate_id") or ""),
            ),
        )
    ]
    evidence_turn_ids = sorted(
        {
            turn_id
            for candidate in selected_candidates
            for turn_id in (candidate.get("turn_ids") or [])
            if str(turn_id).strip()
        },
        key=_turn_sort_key,
    )
    qa["options"] = options
    qa["gold"] = visible_order
    qa["gold_source"] = "earliest_visible_support_turn"
    qa["evidence_turn_ids"] = evidence_turn_ids
    meta.update(
        {
            "applied": True,
            "question_type": "ordering",
            "gold_source": "earliest_visible_support_turn",
            "selected_candidate_ids": [str(candidate.get("candidate_id") or "") for candidate in selected_candidates],
            "visible_order": visible_order,
            "candidate_count": int(bank.get("ordering_candidate_count") or 0),
            "not_rebuilt_reason": None,
        }
    )
    rebuilt["qa"] = qa
    rebuilt["generation_programmatic_builder"] = meta
    rebuilt["generation_candidate_bank"] = bank
    report.update(
        {
            "rebuilt": True,
            "selected_candidate_ids": list(meta.get("selected_candidate_ids") or []),
            "visible_order": visible_order,
        }
    )
    return rebuilt, report


def try_rebuild_multi_select_from_visible_candidates(sample: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    rebuilt = copy.deepcopy(sample)
    qa = rebuilt.get("qa") if isinstance(rebuilt.get("qa"), dict) else {}
    question_type = str(qa.get("question_type") or "unknown")
    report: dict[str, Any] = {
        "sample_id": rebuilt.get("sample_id"),
        "question_type": question_type,
        "rebuilt": False,
        "reason": None,
        "selected_positive_ids": [],
        "selected_negative_ids": [],
    }
    if question_type != "multi_select":
        report["reason"] = "question_type_not_multi_select"
        return rebuilt, report
    meta = _generation_programmatic_builder_meta(rebuilt)
    bank = sample.get("generation_candidate_bank") if isinstance(sample.get("generation_candidate_bank"), dict) else build_generation_candidate_bank(rebuilt)
    selected_positive, selected_negative = _select_multi_select_candidates_from_bank(bank)
    if len(selected_positive) < 2:
        meta.update(
            {
                "applied": False,
                "question_type": "multi_select",
                "not_rebuilt_reason": "multi_select_insufficient_positive_candidates",
                "candidate_count": int(bank.get("multi_select_positive_count") or 0),
                "selected_positive_ids": [str(candidate.get("candidate_id") or "") for candidate in selected_positive],
                "selected_negative_ids": [],
            }
        )
        report["reason"] = "multi_select_insufficient_positive_candidates"
        report["selected_positive_ids"] = list(meta.get("selected_positive_ids") or [])
        rebuilt["generation_candidate_bank"] = bank
        return rebuilt, report
    if len(selected_negative) < 2:
        meta.update(
            {
                "applied": False,
                "question_type": "multi_select",
                "not_rebuilt_reason": "multi_select_insufficient_safe_negative_candidates",
                "candidate_count": int(bank.get("multi_select_negative_count") or 0),
                "selected_positive_ids": [str(candidate.get("candidate_id") or "") for candidate in selected_positive],
                "selected_negative_ids": [str(candidate.get("candidate_id") or "") for candidate in selected_negative],
            }
        )
        report["reason"] = "multi_select_insufficient_safe_negative_candidates"
        report["selected_positive_ids"] = list(meta.get("selected_positive_ids") or [])
        report["selected_negative_ids"] = list(meta.get("selected_negative_ids") or [])
        rebuilt["generation_candidate_bank"] = bank
        return rebuilt, report
    selected_candidates = selected_positive + selected_negative
    option_candidates = sorted(
        selected_candidates,
        key=lambda candidate: (
            str(candidate.get("normalized_text") or ""),
            int(candidate.get("source_priority") or 99),
            candidate.get("earliest_turn_sort_key") or (99, ""),
            str(candidate.get("candidate_id") or ""),
        ),
    )
    options: list[dict[str, Any]] = []
    positive_candidate_ids = {str(candidate.get("candidate_id") or "") for candidate in selected_positive}
    selected_negative_ids = [str(candidate.get("candidate_id") or "") for candidate in selected_negative]
    gold_labels: list[str] = []
    for index, candidate in enumerate(option_candidates):
        label = LETTERS[index]
        candidate_id = str(candidate.get("candidate_id") or f"candidate_{index}")
        if candidate_id in positive_candidate_ids:
            gold_labels.append(label)
        options.append(
            {
                "id": label,
                "text": str(candidate.get("text") or candidate.get("value") or ""),
                "turn_ids": list(candidate.get("turn_ids") or []),
            }
        )
    evidence_turn_ids = sorted(
        {
            turn_id
            for candidate in selected_positive
            for turn_id in (candidate.get("turn_ids") or [])
            if str(turn_id).strip()
        },
        key=_turn_sort_key,
    )
    qa["options"] = options
    qa["gold"] = sorted(gold_labels)
    qa["gold_source"] = "visible_graph_candidate_labels"
    qa["evidence_turn_ids"] = evidence_turn_ids
    meta.update(
        {
            "applied": True,
            "question_type": "multi_select",
            "gold_source": "visible_graph_candidate_labels",
            "selected_positive_ids": [str(candidate.get("candidate_id") or "") for candidate in selected_positive],
            "selected_negative_ids": selected_negative_ids,
            "candidate_count": int(bank.get("multi_select_positive_count") or 0) + int(bank.get("multi_select_negative_count") or 0),
            "not_rebuilt_reason": None,
        }
    )
    rebuilt["qa"] = qa
    rebuilt["generation_programmatic_builder"] = meta
    rebuilt["generation_candidate_bank"] = bank
    report.update(
        {
            "rebuilt": True,
            "selected_positive_ids": list(meta.get("selected_positive_ids") or []),
            "selected_negative_ids": selected_negative_ids,
        }
    )
    return rebuilt, report


def enforce_ordering_visible_support(sample: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    guarded = copy.deepcopy(sample)
    qa = guarded.get("qa") if isinstance(guarded.get("qa"), dict) else {}
    question_type = str(qa.get("question_type") or "unknown")
    report: dict[str, Any] = {
        "sample_id": guarded.get("sample_id"),
        "question_type": question_type,
        "applied": question_type == "ordering",
        "all_options_supported": False,
        "gold_match_visible_order": False,
        "gold_mismatch_visible_order": False,
        "option_no_support": False,
        "gold_option_no_support": False,
        "warnings": [],
        "invalidated_by_guard": False,
    }
    if question_type != "ordering":
        return guarded, report
    meta = _generation_consistency_meta(guarded)
    options = qa.get("options")
    original_gold = [str(item) for item in (qa.get("gold") or [])]
    per_option_earliest_visible_turn: dict[str, str] = {}
    per_option_supporting_turn_ids: dict[str, list[str]] = {}
    supported_option_ids: list[str] = []
    option_no_support_ids: list[str] = []
    if isinstance(options, list):
        for index, option in enumerate(options):
            if not isinstance(option, dict):
                continue
            option_id = str(option.get("id") or f"option_{index}")
            supporting_turns = _find_visible_supporting_turns_for_text(guarded, str(option.get("text") or ""))
            supporting_turn_ids = [str(turn["turn_id"]) for turn in supporting_turns]
            option["turn_ids"] = supporting_turn_ids
            per_option_supporting_turn_ids[option_id] = supporting_turn_ids
            if supporting_turn_ids:
                per_option_earliest_visible_turn[option_id] = supporting_turn_ids[0]
                supported_option_ids.append(option_id)
            else:
                option_no_support_ids.append(option_id)
                _register_generation_consistency_warning(meta, "ordering_option_has_no_visible_support")
    visible_order = [
        option_id
        for option_id, _turn_id in sorted(
            per_option_earliest_visible_turn.items(),
            key=lambda item: (_turn_sort_key(item[1]), item[0]),
        )
    ]
    meta["ordering_visible_order"] = visible_order
    meta["ordering_original_gold"] = original_gold
    meta["per_option_earliest_visible_turn"] = per_option_earliest_visible_turn
    meta["ordering_supporting_turn_ids_by_option"] = per_option_supporting_turn_ids
    meta["ordering_all_options_supported"] = len(option_no_support_ids) == 0 and bool(per_option_supporting_turn_ids)
    if option_no_support_ids:
        _mark_generation_consistency_invalid(guarded, "ordering_option_has_no_visible_support")
        report["invalidated_by_guard"] = True
    if meta["ordering_all_options_supported"] and visible_order == original_gold:
        meta["ordering_gold_matches_visible_support"] = True
        report["gold_match_visible_order"] = True
    elif meta["ordering_all_options_supported"]:
        meta["ordering_gold_matches_visible_support"] = False
        _register_generation_consistency_warning(meta, "ordering_gold_mismatch_from_visible_support")
        report["gold_mismatch_visible_order"] = True
    else:
        meta["ordering_gold_matches_visible_support"] = False
    meta["valid_after_visible_support_guard"] = meta.get("valid_after_visible_support_guard", True)
    report["all_options_supported"] = bool(meta["ordering_all_options_supported"])
    report["option_no_support"] = bool(option_no_support_ids)
    report["warnings"] = list(meta.get("warnings") or [])
    report["ordering_visible_order"] = visible_order
    report["ordering_original_gold"] = original_gold
    report["per_option_earliest_visible_turn"] = per_option_earliest_visible_turn
    report["options_with_no_support"] = option_no_support_ids
    report["supported_option_ids"] = supported_option_ids
    guarded["qa"] = qa
    guarded["generation_consistency_repair"] = meta
    return guarded, report


def enforce_multi_select_visible_support(sample: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    guarded = copy.deepcopy(sample)
    qa = guarded.get("qa") if isinstance(guarded.get("qa"), dict) else {}
    question_type = str(qa.get("question_type") or "unknown")
    report: dict[str, Any] = {
        "sample_id": guarded.get("sample_id"),
        "question_type": question_type,
        "applied": question_type == "multi_select",
        "all_options_supported": False,
        "option_no_support": False,
        "gold_option_no_support": False,
        "warnings": [],
        "invalidated_by_guard": False,
    }
    if question_type != "multi_select":
        return guarded, report
    meta = _generation_consistency_meta(guarded)
    options = qa.get("options")
    gold_labels = {str(item) for item in (qa.get("gold") or [])}
    options_with_no_support: list[str] = []
    gold_options_with_no_support: list[str] = []
    supporting_turn_ids_by_option: dict[str, list[str]] = {}
    if isinstance(options, list):
        for index, option in enumerate(options):
            if not isinstance(option, dict):
                continue
            option_id = str(option.get("id") or f"option_{index}")
            had_turn_ids = bool(_normalize_turn_ids(option.get("turn_ids")))
            expected_text = str(option.get("text") or "")
            if not had_turn_ids and not base.normalize_text(expected_text):
                continue
            supporting_turns = _find_visible_supporting_turns_for_text(guarded, expected_text)
            supporting_turn_ids = [str(turn["turn_id"]) for turn in supporting_turns]
            option["turn_ids"] = supporting_turn_ids
            supporting_turn_ids_by_option[option_id] = supporting_turn_ids
            if not supporting_turn_ids:
                options_with_no_support.append(option_id)
                _register_generation_consistency_warning(meta, "multi_select_option_has_no_visible_support")
                if option_id in gold_labels:
                    gold_options_with_no_support.append(option_id)
    meta["multi_select_supporting_turn_ids_by_option"] = supporting_turn_ids_by_option
    meta["multi_select_options_with_no_support"] = options_with_no_support
    meta["multi_select_all_options_supported"] = len(options_with_no_support) == 0 and bool(supporting_turn_ids_by_option)
    if gold_options_with_no_support:
        meta["multi_select_gold_options_with_no_support"] = gold_options_with_no_support
        _mark_generation_consistency_invalid(guarded, "multi_select_gold_option_has_no_visible_support")
        report["invalidated_by_guard"] = True
    else:
        meta["multi_select_gold_options_with_no_support"] = []
        meta["valid_after_visible_support_guard"] = meta.get("valid_after_visible_support_guard", True)
    report["all_options_supported"] = bool(meta["multi_select_all_options_supported"])
    report["option_no_support"] = bool(options_with_no_support)
    report["gold_option_no_support"] = bool(gold_options_with_no_support)
    report["warnings"] = list(meta.get("warnings") or [])
    report["options_with_no_support"] = options_with_no_support
    report["gold_options_with_no_support"] = gold_options_with_no_support
    guarded["qa"] = qa
    guarded["generation_consistency_repair"] = meta
    return guarded, report


def _mark_turn_cleanup_invalid(sample: dict[str, Any], warning: str) -> None:
    validation = sample.get("validation") if isinstance(sample.get("validation"), dict) else {}
    warnings = validation.get("warnings")
    if not isinstance(warnings, list):
        warnings = []
    if warning not in warnings:
        warnings.append(warning)
    validation["warnings"] = warnings
    validation["graph_dialogue_consistent"] = False
    details = validation.get("details") if isinstance(validation.get("details"), dict) else {}
    validation_checks = details.get("validation_checks") if isinstance(details.get("validation_checks"), dict) else {}
    validation_checks["graph_dialogue_consistency"] = False
    details["validation_checks"] = validation_checks
    validation["details"] = details
    sample["validation"] = validation
    sample["graph_dialogue_alignment"] = False
    cleanup = sample.get("turn_coverage_cleanup") if isinstance(sample.get("turn_coverage_cleanup"), dict) else {}
    cleanup_warnings = cleanup.get("warnings") if isinstance(cleanup.get("warnings"), list) else []
    if warning not in cleanup_warnings:
        cleanup_warnings.append(warning)
    cleanup["warnings"] = cleanup_warnings
    cleanup["valid_after_cleanup"] = False
    sample["turn_coverage_cleanup"] = cleanup


def _item_turn_ids(item: dict[str, Any]) -> list[str]:
    for key in ("turn_ids", "evidence_turn_ids", "activation_turn_ids", "invalidation_turn_ids", "turn_id"):
        turn_ids = _normalize_turn_ids(item.get(key))
        if turn_ids:
            return turn_ids
    return []


def _item_value(item: dict[str, Any]) -> str:
    for key in CURRENT_METHOD_GRAPH_VALUE_KEYS:
        value = base.normalize_space(item.get(key))
        if value:
            return value
    return ""


def _item_role(item: dict[str, Any], *, default_prefix: str, index: int) -> str:
    return str(
        item.get("role")
        or item.get("slot")
        or item.get("entity")
        or item.get("node_id")
        or item.get("event_id")
        or f"{default_prefix}_{index}"
    )


def _legacy_turns_from_current_method(row: dict[str, Any]) -> list[dict[str, Any]] | None:
    dialogue = row.get("dialogue") if isinstance(row.get("dialogue"), dict) else {}
    turns = dialogue.get("turns")
    if not isinstance(turns, list):
        return None
    legacy_turns: list[dict[str, Any]] = []
    for idx, turn in enumerate(turns):
        if not isinstance(turn, dict):
            continue
        turn_id = str(turn.get("turn_id") or turn.get("dia_id") or f"T{idx + 1}")
        legacy_turns.append(
            {
                "dia_id": turn_id,
                "turn_id": turn_id,
                "speaker": base.normalize_space(turn.get("speaker") or "Speaker"),
                "text": base.normalize_space(turn.get("text")),
            }
        )
    return legacy_turns or None


def _legacy_nodes_from_current_method(row: dict[str, Any]) -> list[dict[str, Any]]:
    graph_update = row.get("graph_update") if isinstance(row.get("graph_update"), dict) else {}
    raw_groups: list[tuple[str, list[Any]]] = []
    for key in (
        "events",
        "state_changes",
        "invalidations",
        "restorations",
        "cancellations",
        "delayed_activations",
    ):
        value = graph_update.get(key)
        if isinstance(value, list):
            raw_groups.append((key, value))
    qa = row.get("qa") if isinstance(row.get("qa"), dict) else {}
    if qa.get("gold") and qa.get("evidence_turn_ids"):
        gold_values = qa.get("gold")
        if not isinstance(gold_values, list):
            gold_values = [gold_values]
        raw_groups.append(
            (
                "qa_gold",
                [
                    {
                        "role": qa.get("question_type") or "qa_gold",
                        "value": value,
                        "turn_ids": qa.get("evidence_turn_ids"),
                    }
                    for value in gold_values
                ],
            )
        )

    nodes: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for group_name, items in raw_groups:
        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            value = _item_value(item)
            if len(value) < 8 or value.endswith("?"):
                continue
            turn_ids = _item_turn_ids(item)
            if not turn_ids:
                continue
            key = (base.normalize_text(value), tuple(turn_ids))
            if not key[0] or key in seen:
                continue
            nodes.append(
                {
                    "node_id": item.get("node_id") or item.get("event_id") or f"{group_name}_{index}",
                    "role": _item_role(item, default_prefix=group_name, index=index),
                    "label": item.get("role") or group_name,
                    "value": value,
                    "turn_ids": turn_ids,
                    "status": item.get("status") or group_name,
                }
            )
            seen.add(key)
    return nodes


def _conversation_entity(row: dict[str, Any]) -> str | None:
    graph_update = row.get("graph_update") if isinstance(row.get("graph_update"), dict) else {}
    entities = graph_update.get("entities")
    if isinstance(entities, list):
        for entity in entities:
            if isinstance(entity, dict) and entity.get("type") == "conversation_memory":
                entity_id = str(entity.get("entity_id") or "").strip()
                if entity_id:
                    return entity_id
    seed = row.get("seed") if isinstance(row.get("seed"), dict) else {}
    conversation_id = str(seed.get("locomo_conversation_id") or "").strip()
    return conversation_id or None


def _augment_seed_row_for_old_gate(row: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(row, dict):
        return row
    augmented = copy.deepcopy(row)
    is_current_method_row = all(isinstance(row.get(key), dict) for key in ("dialogue", "graph_update", "qa"))
    if not isinstance(augmented.get("final_dialogue"), list):
        legacy_turns = _legacy_turns_from_current_method(row)
        if legacy_turns:
            augmented["final_dialogue"] = legacy_turns
    final_dialogue = augmented.get("final_dialogue") if isinstance(augmented.get("final_dialogue"), list) else []
    visible_turn_ids = _visible_turn_ids_from_legacy_turns(final_dialogue)
    raw_legacy_nodes = _legacy_nodes_from_current_method(row)
    filtered_legacy_nodes, node_turn_filter_stats = _filter_nodes_to_visible_turns(raw_legacy_nodes, visible_turn_ids)
    if is_current_method_row:
        augmented["state_chain_nodes"] = copy.deepcopy(filtered_legacy_nodes)
        state_chain = augmented.get("state_chain") if isinstance(augmented.get("state_chain"), dict) else {}
        state_chain["nodes"] = copy.deepcopy(filtered_legacy_nodes)
        seed_source = row.get("seed_source") if isinstance(row.get("seed_source"), dict) else {}
        if not state_chain.get("source_dialogue_path") and seed_source.get("source_dialogue_path"):
            state_chain["source_dialogue_path"] = seed_source.get("source_dialogue_path")
        augmented["state_chain"] = state_chain
    elif filtered_legacy_nodes:
        if not isinstance(augmented.get("state_chain_nodes"), list):
            augmented["state_chain_nodes"] = copy.deepcopy(filtered_legacy_nodes)
        state_chain = augmented.get("state_chain") if isinstance(augmented.get("state_chain"), dict) else {}
        if not isinstance(state_chain.get("nodes"), list):
            state_chain["nodes"] = copy.deepcopy(filtered_legacy_nodes)
        seed_source = row.get("seed_source") if isinstance(row.get("seed_source"), dict) else {}
        if not state_chain.get("source_dialogue_path") and seed_source.get("source_dialogue_path"):
            state_chain["source_dialogue_path"] = seed_source.get("source_dialogue_path")
        if state_chain:
            augmented["state_chain"] = state_chain
    if not augmented.get("state_entity"):
        conversation_entity = _conversation_entity(row)
        if conversation_entity:
            augmented["state_entity"] = conversation_entity
    if not augmented.get("state_slot"):
        augmented["state_slot"] = "memory_state"
    turn_coverage_summary = _legacy_row_turn_coverage_summary(augmented)
    augmented["_current_method_old_gate_adapter"] = {
        "final_dialogue_turn_count": turn_coverage_summary["final_dialogue_turn_count"],
        "raw_legacy_node_count": len(raw_legacy_nodes),
        "filtered_legacy_node_count": len(filtered_legacy_nodes),
        "node_turn_filter_stats": node_turn_filter_stats,
        "turn_coverage_summary": turn_coverage_summary,
    }
    return augmented


def _augment_seed_rows_for_old_gate(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_augment_seed_row_for_old_gate(row) for row in seed_rows]


def _count_stats(values: list[int]) -> dict[str, float]:
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0}
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(sum(values) / len(values), 4),
    }


def _source_turn_lookup_from_row(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_turns = row.get("final_dialogue")
    if not isinstance(raw_turns, list):
        raw_turns = _legacy_turns_from_current_method(row) or []
    lookup: dict[str, dict[str, Any]] = {}
    for turn in raw_turns:
        normalized = _normalized_dialogue_turn(turn) if isinstance(turn, dict) else None
        if normalized:
            lookup[normalized["turn_id"]] = normalized
    return lookup


def _build_source_turn_lookup_by_source_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    source_turn_lookup_by_source_id: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        lookup = _source_turn_lookup_from_row(row)
        for key in (row.get("sample_id"), row.get("record_id")):
            key_text = str(key or "").strip()
            if key_text and key_text not in source_turn_lookup_by_source_id:
                source_turn_lookup_by_source_id[key_text] = lookup
    return source_turn_lookup_by_source_id


def _referenced_turn_ids_from_payload(qa: dict[str, Any], graph_update: dict[str, Any]) -> list[str]:
    referenced_turn_ids: list[str] = []

    def add_turn_ids(value: Any) -> None:
        for turn_id in _normalize_turn_ids(value):
            if turn_id not in referenced_turn_ids:
                referenced_turn_ids.append(turn_id)

    add_turn_ids(qa.get("evidence_turn_ids"))
    options = qa.get("options")
    if isinstance(options, list):
        for option in options:
            if isinstance(option, dict):
                add_turn_ids(option.get("turn_ids"))
    events = graph_update.get("events")
    if isinstance(events, list):
        for event in events:
            if isinstance(event, dict):
                add_turn_ids(event.get("turn_ids"))
    state_changes = graph_update.get("state_changes")
    if isinstance(state_changes, list):
        for state_change in state_changes:
            if isinstance(state_change, dict):
                add_turn_ids(state_change.get("evidence_turn_ids"))
    for group_name in ("invalidations", "restorations", "cancellations", "delayed_activations"):
        items = graph_update.get(group_name)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            for field_name in COMMON_TURN_ID_FIELD_NAMES:
                add_turn_ids(item.get(field_name))
    return sorted(referenced_turn_ids, key=_turn_sort_key)


def _expand_dialogue_turns_with_source_refs(
    dialogue_turns: Any,
    referenced_turn_ids: list[str],
    source_turn_lookup: dict[str, dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized_turns: list[dict[str, Any]] = []
    visible_turn_ids: set[str] = set()
    if isinstance(dialogue_turns, list):
        for turn in dialogue_turns:
            normalized = _normalized_dialogue_turn(turn) if isinstance(turn, dict) else None
            if not normalized or normalized["turn_id"] in visible_turn_ids:
                continue
            normalized_turns.append(normalized)
            visible_turn_ids.add(normalized["turn_id"])
    missing_before = [turn_id for turn_id in referenced_turn_ids if turn_id not in visible_turn_ids]
    added_turn_ids: list[str] = []
    for turn_id in missing_before:
        source_turn = (source_turn_lookup or {}).get(turn_id)
        if source_turn and turn_id not in visible_turn_ids:
            normalized_turns.append(copy.deepcopy(source_turn))
            visible_turn_ids.add(turn_id)
            added_turn_ids.append(turn_id)
    normalized_turns.sort(key=lambda turn: _turn_sort_key(turn.get("turn_id") or ""))
    missing_after = [turn_id for turn_id in referenced_turn_ids if turn_id not in visible_turn_ids]
    stats = {
        "dialogue_turn_count_before": len(visible_turn_ids) - len(added_turn_ids),
        "dialogue_turn_count_after": len(visible_turn_ids),
        "referenced_turn_count": len(referenced_turn_ids),
        "missing_turn_ids_before": missing_before,
        "added_turn_ids": added_turn_ids,
        "added_turn_count": len(added_turn_ids),
        "missing_turn_ids_after": missing_after,
        "recovered_missing_turn_count": len(missing_before) - len(missing_after),
    }
    return normalized_turns, stats


def _programmatic_ordering_from_candidate_bank(
    bundle: Any,
    bank: dict[str, Any],
    skill: str,
    rng: Any,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[str]] | None:
    selected_candidates = _select_ordering_candidates_from_bank(bank, rng=rng)
    if len(selected_candidates) < 4:
        return None
    option_candidates = sorted(
        selected_candidates,
        key=lambda candidate: (
            str(base.normalize_text(candidate.get("text") or candidate.get("value") or "")),
            candidate.get("earliest_turn_sort_key") or (99, ""),
            str(candidate.get("candidate_id") or ""),
        ),
    )
    options: list[dict[str, Any]] = []
    label_by_candidate_id: dict[str, str] = {}
    for index, candidate in enumerate(option_candidates):
        label = LETTERS[index]
        candidate_id = str(candidate.get("candidate_id") or f"candidate_{index}")
        label_by_candidate_id[candidate_id] = label
        options.append(
            {
                "id": label,
                "text": str(candidate.get("text") or candidate.get("value") or ""),
                "turn_ids": list(candidate.get("turn_ids") or []),
            }
        )
    ordered_candidates = sorted(
        selected_candidates,
        key=lambda candidate: (
            candidate.get("earliest_turn_sort_key") or (99, ""),
            str(candidate.get("candidate_id") or ""),
        ),
    )
    gold = [label_by_candidate_id[str(candidate.get("candidate_id") or "")] for candidate in ordered_candidates]
    evidence_turn_ids = sorted(
        {
            turn_id
            for candidate in selected_candidates
            for turn_id in (candidate.get("turn_ids") or [])
            if str(turn_id).strip()
        },
        key=_turn_sort_key,
    )
    question = (
        "Order the option letters by the first dialogue turn that supports each event, from earliest to latest. "
        "Answer as letters separated by '>'."
    )
    qa = {
        "question_type": "ordering",
        "question": question,
        "options": options,
        "gold": gold,
        "gold_source": "programmatic_earliest_visible_turn",
        "evidence_turn_ids": evidence_turn_ids,
    }
    nodes = [_candidate_to_node(candidate, role_prefix="ordering_candidate", index=index) for index, candidate in enumerate(ordered_candidates, start=1)]
    graph_update = base.make_graph_update(bundle=bundle, nodes=nodes, evidence_turn_ids=evidence_turn_ids, skill=skill)
    visible_order = [str(label) for label in gold]
    graph_update["generation_candidate_bank"] = copy.deepcopy(bank)
    qa["generation_candidate_bank_qtype"] = "ordering"
    turns = _select_dialogue_turns_for_candidate_bank(bundle, evidence_turn_ids, rng, radius=1, max_turns=20)
    qa["generation_programmatic_builder"] = {
        "applied": True,
        "question_type": "ordering",
        "gold_source": "earliest_visible_support_turn",
        "selected_candidate_ids": [str(candidate.get("candidate_id") or "") for candidate in ordered_candidates],
        "visible_order": visible_order,
    }
    return qa, graph_update, turns, evidence_turn_ids


def _programmatic_multi_select_from_candidate_bank(
    bundle: Any,
    bank: dict[str, Any],
    skill: str,
    rng: Any,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[str]] | None:
    selected_positive, selected_negative = _select_multi_select_candidates_from_bank(bank, rng=rng)
    if len(selected_positive) < 2 or len(selected_negative) < 2:
        return None
    selected_candidates = selected_positive + selected_negative
    option_candidates = sorted(
        selected_candidates,
        key=lambda candidate: (
            int(_candidate_source_priority(str(candidate.get("source") or ""))),
            candidate.get("earliest_turn_sort_key") or (99, ""),
            str(candidate.get("candidate_id") or ""),
        ),
    )
    positive_candidate_ids = {str(candidate.get("candidate_id") or "") for candidate in selected_positive}
    negative_reason_by_candidate_id = {
        str(candidate.get("candidate_id") or ""): {
            "false_reason": candidate.get("false_reason"),
            "false_reason_type": candidate.get("false_reason_type"),
        }
        for candidate in selected_negative
    }
    options: list[dict[str, Any]] = []
    gold_labels: list[str] = []
    for index, candidate in enumerate(option_candidates):
        label = LETTERS[index]
        candidate_id = str(candidate.get("candidate_id") or f"candidate_{index}")
        option = {
            "id": label,
            "text": str(candidate.get("text") or candidate.get("value") or ""),
            "turn_ids": list(candidate.get("turn_ids") or []),
        }
        if candidate_id in positive_candidate_ids:
            gold_labels.append(label)
        else:
            option.update(negative_reason_by_candidate_id.get(candidate_id, {}))
        options.append(option)
    evidence_turn_ids = sorted(
        {
            turn_id
            for candidate in selected_positive
            for turn_id in (candidate.get("turn_ids") or [])
            if str(turn_id).strip()
        },
        key=_turn_sort_key,
    )
    qa = {
        "question_type": "multi_select",
        "question": "Select exactly the two options that remain supported active graph states. Answer with option letters only.",
        "options": options,
        "gold": sorted(gold_labels),
        "gold_source": "programmatic_candidate_bank_labels",
        "evidence_turn_ids": evidence_turn_ids,
    }
    graph_nodes = [
        _candidate_to_node(candidate, role_prefix="multi_select_candidate", index=index)
        for index, candidate in enumerate(option_candidates, start=1)
    ]
    graph_update = base.make_graph_update(bundle=bundle, nodes=graph_nodes, evidence_turn_ids=evidence_turn_ids, skill=skill)
    graph_update["generation_candidate_bank"] = copy.deepcopy(bank)
    qa["generation_programmatic_builder"] = {
        "applied": True,
        "question_type": "multi_select",
        "gold_source": "visible_graph_candidate_labels",
        "selected_positive_ids": [str(candidate.get("candidate_id") or "") for candidate in selected_positive],
        "selected_negative_ids": [str(candidate.get("candidate_id") or "") for candidate in selected_negative],
    }
    turns = _select_dialogue_turns_for_candidate_bank(
        bundle,
        sorted(
            {
                turn_id
                for candidate in selected_candidates
                for turn_id in (candidate.get("turn_ids") or [])
                if str(turn_id).strip()
            },
            key=_turn_sort_key,
        ),
        rng,
        radius=1,
        max_turns=20,
    )
    return qa, graph_update, turns, evidence_turn_ids


def _build_qa_and_graph_with_turn_ref_coverage(
    bundle: Any,
    qtype: str,
    skill: str,
    rng: Any,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[str]]:
    if _ORIGINAL_BASE_BUILD_QA_AND_GRAPH is None:
        raise RuntimeError("old_gate base.build_qa_and_graph is unavailable")
    source_scaffold_sample = _scaffold_sample_from_bundle(bundle)
    candidate_bank = build_generation_candidate_bank(source_scaffold_sample)
    custom_result: tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[str]] | None = None
    if qtype == "ordering" and candidate_bank.get("ordering_ready") is True:
        custom_result = _programmatic_ordering_from_candidate_bank(bundle, candidate_bank, skill, rng)
    elif qtype == "multi_select" and candidate_bank.get("multi_select_ready") is True:
        custom_result = _programmatic_multi_select_from_candidate_bank(bundle, candidate_bank, skill, rng)
    if custom_result is not None:
        qa, graph_update, turns, evidence_turn_ids = custom_result
    else:
        qa, graph_update, turns, evidence_turn_ids = _ORIGINAL_BASE_BUILD_QA_AND_GRAPH(bundle, qtype, skill, rng)
        if isinstance(graph_update, dict):
            graph_update["generation_candidate_bank"] = copy.deepcopy(candidate_bank)
        if isinstance(qa, dict):
            qa["generation_programmatic_builder"] = {
                "applied": False,
                "question_type": qtype,
                "not_rebuilt_reasons": _candidate_bank_not_ready_reasons(candidate_bank),
            }
    referenced_turn_ids = _referenced_turn_ids_from_payload(
        qa if isinstance(qa, dict) else {},
        graph_update if isinstance(graph_update, dict) else {},
    )
    expanded_turns, _ = _expand_dialogue_turns_with_source_refs(
        turns,
        referenced_turn_ids,
        {
            turn_id: normalized
            for turn_id, turn in getattr(bundle, "turn_map", {}).items()
            if (normalized := _normalized_dialogue_turn(turn))
        },
    )
    return qa, graph_update, expanded_turns, evidence_turn_ids


def _build_rewrite_prompt_with_consistency_constraints(
    bundle: Any,
    scaffold: dict[str, Any],
    index: int,
    qtype: str,
    skill: str,
    seed_text: str,
) -> dict[str, Any]:
    if _ORIGINAL_OLD_GATE_BUILD_REWRITE_PROMPT is None:
        raise RuntimeError("old_gate build_rewrite_prompt is unavailable")
    prompt = _ORIGINAL_OLD_GATE_BUILD_REWRITE_PROMPT(bundle, scaffold, index, qtype, skill, seed_text)
    constraints = list(prompt.get("constraints") or [])
    extra_constraints = [
        "keep qa.gold, qa.evidence_turn_ids, and every qa.options[*].turn_ids semantically aligned to the scaffold",
        "every turn_id referenced anywhere in qa or graph_update must be present in dialogue.turns of the same payload",
        "do not collapse two different scaffold options into paraphrases of the same supported event",
        "do not cite or imply support from dialogue turns that are absent from dialogue.turns",
    ]
    if qtype == "ordering":
        extra_constraints.extend(
            [
                "for ordering, each option must stay grounded in its own option turn_ids and describe a distinct event",
                "for ordering, qa.gold must still match the earliest visible supporting dialogue turn for each option",
            ]
        )
    if qtype == "multi_select":
        extra_constraints.extend(
            [
                "for multi_select, every option with turn_ids must remain visibly supported by at least one dialogue turn in dialogue.turns",
                "for multi_select, do not rewrite unsupported distractors into duplicates of gold-supported options",
            ]
        )
    for constraint in extra_constraints:
        if constraint not in constraints:
            constraints.append(constraint)
    prompt["constraints"] = constraints
    prompt["current_method_consistency_contract"] = {
        "dialogue_turn_ids": [
            str(turn.get("turn_id") or "")
            for turn in ((scaffold.get("dialogue") or {}).get("turns") or [])
            if isinstance(turn, dict) and str(turn.get("turn_id") or "").strip()
        ],
        "qa_evidence_turn_ids": list(_normalize_turn_ids((scaffold.get("qa") or {}).get("evidence_turn_ids"))),
        "qa_option_turn_ids": {
            str(option.get("id") or f"option_{index}"): list(_normalize_turn_ids(option.get("turn_ids")))
            for index, option in enumerate(((scaffold.get("qa") or {}).get("options") or []))
            if isinstance(option, dict)
        },
    }
    return prompt


def _expected_text_from_item(item: dict[str, Any]) -> str | None:
    for key in CURRENT_METHOD_GRAPH_VALUE_KEYS:
        value = base.normalize_space(item.get(key))
        if value:
            return value
    return None


def _assign_turn_id_field(container: dict[str, Any], field_name: str, kept_turn_ids: list[str]) -> None:
    if field_name == "turn_id":
        container[field_name] = kept_turn_ids[0] if kept_turn_ids else None
        return
    if field_name == "event_turn_ids" and not isinstance(container.get(field_name), list):
        container[field_name] = kept_turn_ids[0] if kept_turn_ids else None
        return
    container[field_name] = kept_turn_ids


def _register_cleanup_warning(
    cleanup: dict[str, Any],
    sample: dict[str, Any],
    warning: str,
    *,
    invalid: bool,
) -> None:
    warnings = cleanup.get("warnings") if isinstance(cleanup.get("warnings"), list) else []
    if warning not in warnings:
        warnings.append(warning)
    cleanup["warnings"] = warnings
    if invalid:
        _mark_turn_cleanup_invalid(sample, warning)


def reset_turn_coverage_cleanup_artifacts(sample: dict[str, Any]) -> dict[str, Any]:
    reset_sample = copy.deepcopy(sample)
    existing_stats = (
        reset_sample.get(TURN_COVERAGE_CLEANUP_RESET_STATS_KEY)
        if isinstance(reset_sample.get(TURN_COVERAGE_CLEANUP_RESET_STATS_KEY), dict)
        else {}
    )
    reset_stats: dict[str, Any] = {
        "cleanup_artifact_reset_applied": True,
        "cleanup_artifact_reset_count": 1,
        "stale_cleanup_warning_removed_count": int(existing_stats.get("stale_cleanup_warning_removed_count") or 0),
        "stale_cleanup_invalid_flag_reset_count": int(existing_stats.get("stale_cleanup_invalid_flag_reset_count") or 0),
    }

    had_cleanup_block = isinstance(reset_sample.get("turn_coverage_cleanup"), dict)
    if had_cleanup_block:
        reset_sample.pop("turn_coverage_cleanup", None)

    validation = reset_sample.get("validation") if isinstance(reset_sample.get("validation"), dict) else None
    validation_warnings = list(validation.get("warnings") or []) if validation else []
    removed_cleanup_warnings = [
        warning
        for warning in validation_warnings
        if isinstance(warning, str) and warning.startswith(TURN_COVERAGE_CLEANUP_WARNING_PREFIX)
    ]
    if removed_cleanup_warnings:
        reset_stats["stale_cleanup_warning_removed_count"] += len(removed_cleanup_warnings)
    preserved_warnings = [
        warning
        for warning in validation_warnings
        if not (isinstance(warning, str) and warning.startswith(TURN_COVERAGE_CLEANUP_WARNING_PREFIX))
    ]

    should_refresh_graph_dialogue = had_cleanup_block or bool(removed_cleanup_warnings)
    if should_refresh_graph_dialogue and validation is not None:
        try:
            refreshed_validation = base.validate_sample(reset_sample)
        except Exception:
            refreshed_validation = None
        if isinstance(refreshed_validation, dict):
            desired_graph_dialogue_consistent = refreshed_validation.get("graph_dialogue_consistent") is True
            if validation.get("graph_dialogue_consistent") is not desired_graph_dialogue_consistent:
                reset_stats["stale_cleanup_invalid_flag_reset_count"] += 1
            validation["graph_dialogue_consistent"] = desired_graph_dialogue_consistent

            details = validation.get("details") if isinstance(validation.get("details"), dict) else {}
            validation_checks = details.get("validation_checks") if isinstance(details.get("validation_checks"), dict) else {}
            if validation_checks.get("graph_dialogue_consistency") is not desired_graph_dialogue_consistent:
                reset_stats["stale_cleanup_invalid_flag_reset_count"] += 1
            validation_checks["graph_dialogue_consistency"] = desired_graph_dialogue_consistent
            details["validation_checks"] = validation_checks
            validation["details"] = details

            if "graph_dialogue_alignment" in reset_sample and reset_sample.get("graph_dialogue_alignment") is not desired_graph_dialogue_consistent:
                reset_stats["stale_cleanup_invalid_flag_reset_count"] += 1
                reset_sample["graph_dialogue_alignment"] = desired_graph_dialogue_consistent

            preserved_warnings = [
                warning
                for warning in preserved_warnings
                if warning != GRAPH_DIALOGUE_CONSISTENCY_WARNING
            ]
            refreshed_warnings = refreshed_validation.get("warnings") if isinstance(refreshed_validation.get("warnings"), list) else []
            if GRAPH_DIALOGUE_CONSISTENCY_WARNING in refreshed_warnings:
                preserved_warnings.append(GRAPH_DIALOGUE_CONSISTENCY_WARNING)

        validation["warnings"] = preserved_warnings
        reset_sample["validation"] = validation
    elif validation is not None:
        validation["warnings"] = preserved_warnings
        reset_sample["validation"] = validation

    reset_sample[TURN_COVERAGE_CLEANUP_RESET_STATS_KEY] = reset_stats
    return reset_sample


def _cleanup_turn_id_field(
    *,
    sample: dict[str, Any],
    cleanup: dict[str, Any],
    container: dict[str, Any],
    field_name: str,
    visible_turn_ids: set[str],
    turn_text_by_id: dict[str, str],
    expected_text: str | None,
) -> tuple[list[str], dict[str, Any]]:
    kept_turn_ids, stats = _visible_supporting_turn_ids(
        container.get(field_name),
        visible_turn_ids,
        turn_text_by_id,
        expected_text=expected_text,
    )
    _assign_turn_id_field(container, field_name, kept_turn_ids)
    cleanup["trimmed_ref_count"] += stats["dropped_missing_count"] + stats["dropped_unsupported_count"]
    cleanup["missing_ref_count"] += stats["dropped_missing_count"]
    cleanup["unsupported_ref_count"] += stats["dropped_unsupported_count"]
    return kept_turn_ids, stats


def diagnose_generated_sample_evidence_window(
    sample: dict[str, Any],
    *,
    source_turn_lookup: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    qa = sample.get("qa") if isinstance(sample.get("qa"), dict) else {}
    graph_update = sample.get("graph_update") if isinstance(sample.get("graph_update"), dict) else {}
    question_type = str(qa.get("question_type") or "unknown")
    attack_plan = sample.get("attack_plan") if isinstance(sample.get("attack_plan"), dict) else {}
    visible_turn_ids, turn_text_by_id = _sample_visible_turns(sample)
    source_turn_lookup = source_turn_lookup or {}
    source_turn_ids = set(source_turn_lookup)
    total_missing_ref_count = 0
    total_recoverable_missing_ref_count = 0
    total_unsupported_ref_count = 0

    def missing_and_recoverable(turn_ids: list[str]) -> tuple[list[str], list[str]]:
        missing = [turn_id for turn_id in turn_ids if turn_id not in visible_turn_ids]
        recoverable = [turn_id for turn_id in missing if turn_id in source_turn_ids]
        return missing, recoverable

    evidence_turn_ids = _normalize_turn_ids(qa.get("evidence_turn_ids"))
    evidence_visible = [turn_id for turn_id in evidence_turn_ids if turn_id in visible_turn_ids]
    evidence_missing, evidence_recoverable = missing_and_recoverable(evidence_turn_ids)
    total_missing_ref_count += len(evidence_missing)
    total_recoverable_missing_ref_count += len(evidence_recoverable)

    options_diagnostic: list[dict[str, Any]] = []
    ordering_earliest: dict[str, str] = {}
    ordering_missing_support: list[str] = []
    multi_select_missing_support: list[dict[str, Any]] = []
    original_gold_labels = {str(item) for item in (qa.get("gold") or [])} if isinstance(qa.get("gold"), list) else set()
    options = qa.get("options")
    if isinstance(options, list):
        for index, option in enumerate(options):
            if not isinstance(option, dict):
                continue
            option_turn_ids = _normalize_turn_ids(option.get("turn_ids"))
            visible_option_turn_ids = [turn_id for turn_id in option_turn_ids if turn_id in visible_turn_ids]
            missing_option_turn_ids, recoverable_option_turn_ids = missing_and_recoverable(option_turn_ids)
            supporting_turn_ids, _stats = _visible_supporting_turn_ids(
                option_turn_ids,
                visible_turn_ids,
                turn_text_by_id,
                expected_text=str(option.get("text") or ""),
            )
            unsupported_visible_turn_ids = [turn_id for turn_id in visible_option_turn_ids if turn_id not in supporting_turn_ids]
            total_missing_ref_count += len(missing_option_turn_ids)
            total_recoverable_missing_ref_count += len(recoverable_option_turn_ids)
            total_unsupported_ref_count += len(unsupported_visible_turn_ids)
            option_id = str(option.get("id") or f"option_{index}")
            option_diag = {
                "id": option_id,
                "text": str(option.get("text") or ""),
                "turn_ids": option_turn_ids,
                "visible_turn_ids": visible_option_turn_ids,
                "missing_turn_ids": missing_option_turn_ids,
                "recoverable_missing_turn_ids": recoverable_option_turn_ids,
                "supporting_visible_turn_ids": supporting_turn_ids,
                "unsupported_visible_turn_ids": unsupported_visible_turn_ids,
                "is_gold": option_id in original_gold_labels,
            }
            if supporting_turn_ids:
                option_diag["earliest_visible_support_turn_id"] = sorted(supporting_turn_ids, key=_turn_sort_key)[0]
                ordering_earliest[option_id] = option_diag["earliest_visible_support_turn_id"]
            elif question_type == "ordering" and option_turn_ids:
                ordering_missing_support.append(option_id)
            if question_type == "multi_select" and option_turn_ids and not supporting_turn_ids:
                multi_select_missing_support.append(
                    {
                        "id": option_id,
                        "text": str(option.get("text") or ""),
                        "is_gold": option_id in original_gold_labels,
                    }
                )
            options_diagnostic.append(option_diag)

    graph_diagnostic: dict[str, list[dict[str, Any]]] = {
        "events": [],
        "state_changes": [],
        "invalidations": [],
        "restorations": [],
        "cancellations": [],
        "delayed_activations": [],
    }

    def record_graph_item(group_name: str, index: int, item: dict[str, Any], field_name: str, expected_text: str | None) -> None:
        nonlocal total_missing_ref_count, total_recoverable_missing_ref_count, total_unsupported_ref_count
        raw_turn_ids = _normalize_turn_ids(item.get(field_name))
        if not raw_turn_ids:
            return
        supporting_turn_ids, _stats = _visible_supporting_turn_ids(
            raw_turn_ids,
            visible_turn_ids,
            turn_text_by_id,
            expected_text=expected_text,
        )
        visible_item_turn_ids = [turn_id for turn_id in raw_turn_ids if turn_id in visible_turn_ids]
        missing_item_turn_ids, recoverable_item_turn_ids = missing_and_recoverable(raw_turn_ids)
        unsupported_visible_turn_ids = [turn_id for turn_id in visible_item_turn_ids if turn_id not in supporting_turn_ids]
        total_missing_ref_count += len(missing_item_turn_ids)
        total_recoverable_missing_ref_count += len(recoverable_item_turn_ids)
        total_unsupported_ref_count += len(unsupported_visible_turn_ids)
        graph_diagnostic[group_name].append(
            {
                "index": index,
                "field": field_name,
                "turn_ids": raw_turn_ids,
                "visible_turn_ids": visible_item_turn_ids,
                "missing_turn_ids": missing_item_turn_ids,
                "recoverable_missing_turn_ids": recoverable_item_turn_ids,
                "supporting_visible_turn_ids": supporting_turn_ids,
                "unsupported_visible_turn_ids": unsupported_visible_turn_ids,
                "expected_text": expected_text,
            }
        )

    events = graph_update.get("events")
    if isinstance(events, list):
        for index, event in enumerate(events):
            if isinstance(event, dict):
                record_graph_item("events", index, event, "turn_ids", _expected_text_from_item(event))
    state_changes = graph_update.get("state_changes")
    if isinstance(state_changes, list):
        for index, state_change in enumerate(state_changes):
            if isinstance(state_change, dict):
                record_graph_item(
                    "state_changes",
                    index,
                    state_change,
                    "evidence_turn_ids",
                    base.normalize_space(state_change.get("to") or state_change.get("from")),
                )
    for group_name in ("invalidations", "restorations", "cancellations", "delayed_activations"):
        items = graph_update.get(group_name)
        if not isinstance(items, list):
            continue
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            expected_text = _expected_text_from_item(item)
            for field_name in COMMON_TURN_ID_FIELD_NAMES:
                if _normalize_turn_ids(item.get(field_name)):
                    record_graph_item(group_name, index, item, field_name, expected_text)

    diagnosis_cause = "clean"
    if total_missing_ref_count > 0 and total_unsupported_ref_count > 0:
        diagnosis_cause = "mixed_window_and_semantic"
    elif total_missing_ref_count > 0:
        diagnosis_cause = "window_exclusion"
    elif total_unsupported_ref_count > 0:
        diagnosis_cause = "semantic_or_rewrite_drift"

    diagnostic: dict[str, Any] = {
        "sample_id": sample.get("sample_id"),
        "source_id": sample.get("source_id"),
        "question_type": question_type,
        "attack_skill": str(attack_plan.get("attack_skill") or "unknown"),
        "dialogue_turn_ids": sorted(visible_turn_ids, key=_turn_sort_key),
        "dialogue_turn_count": len(visible_turn_ids),
        "source_turn_lookup_available": bool(source_turn_lookup),
        "source_turn_count": len(source_turn_lookup),
        "qa_evidence_turn_ids": {
            "turn_ids": evidence_turn_ids,
            "visible_turn_ids": evidence_visible,
            "missing_turn_ids": evidence_missing,
            "recoverable_missing_turn_ids": evidence_recoverable,
        },
        "qa_options": options_diagnostic,
        "graph_update": graph_diagnostic,
        "total_missing_ref_count": total_missing_ref_count,
        "total_recoverable_missing_ref_count": total_recoverable_missing_ref_count,
        "total_unsupported_ref_count": total_unsupported_ref_count,
        "diagnosis_cause": diagnosis_cause,
    }
    if question_type == "ordering":
        visible_recomputed_order = [
            option["id"]
            for option in sorted(
                [option for option in options_diagnostic if option.get("supporting_visible_turn_ids")],
                key=lambda option: (
                    _turn_sort_key(str(option.get("earliest_visible_support_turn_id") or "")),
                    str(option.get("id") or ""),
                ),
            )
        ]
        diagnostic["ordering"] = {
            "original_gold": [str(item) for item in (qa.get("gold") or [])],
            "visible_recomputed_order": visible_recomputed_order,
            "per_option_earliest_visible_support_turn": ordering_earliest,
            "options_with_no_visible_support": ordering_missing_support,
            "gold_matches_visible_order": visible_recomputed_order == [str(item) for item in (qa.get("gold") or [])]
            if len(visible_recomputed_order) == len(options_diagnostic)
            else False,
        }
    if question_type == "multi_select":
        diagnostic["multi_select"] = {
            "options_with_no_visible_semantic_support": multi_select_missing_support,
            "unsupported_gold_options": [
                option["id"]
                for option in multi_select_missing_support
                if option.get("is_gold") is True
            ],
        }
    return diagnostic


def repair_generated_sample_evidence_window(
    sample: dict[str, Any],
    *,
    source_turn_lookup: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    repaired = copy.deepcopy(sample)
    before = diagnose_generated_sample_evidence_window(repaired, source_turn_lookup=source_turn_lookup)
    qa = repaired.get("qa") if isinstance(repaired.get("qa"), dict) else {}
    graph_update = repaired.get("graph_update") if isinstance(repaired.get("graph_update"), dict) else {}
    dialogue = repaired.get("dialogue") if isinstance(repaired.get("dialogue"), dict) else {}
    referenced_turn_ids = _referenced_turn_ids_from_payload(qa, graph_update)
    expanded_turns, expansion_stats = _expand_dialogue_turns_with_source_refs(
        dialogue.get("turns"),
        referenced_turn_ids,
        source_turn_lookup,
    )
    if expanded_turns:
        dialogue["turns"] = expanded_turns
        repaired["dialogue"] = dialogue
    after = diagnose_generated_sample_evidence_window(repaired, source_turn_lookup=source_turn_lookup)
    warnings: list[str] = []
    ordering_original_gold: list[str] | None = None
    ordering_recomputed_gold: list[str] | None = None
    safe_to_rewrite_gold = None
    if str(qa.get("question_type") or "") == "ordering":
        ordering_after = after.get("ordering") if isinstance(after.get("ordering"), dict) else {}
        ordering_original_gold = list(ordering_after.get("original_gold") or [])
        ordering_recomputed_gold = list(ordering_after.get("visible_recomputed_order") or [])
        if ordering_after.get("options_with_no_visible_support"):
            warnings.append("ordering_option_has_no_visible_support")
        if ordering_recomputed_gold and ordering_original_gold and ordering_recomputed_gold != ordering_original_gold:
            warnings.append("ordering_gold_mismatch_after_visible_trim")
            safe_to_rewrite_gold = False
    if str(qa.get("question_type") or "") == "multi_select":
        multi_select_after = after.get("multi_select") if isinstance(after.get("multi_select"), dict) else {}
        if multi_select_after.get("options_with_no_visible_semantic_support"):
            warnings.append("multi_select_option_has_no_visible_support")
    repair_meta: dict[str, Any] = {
        "applied": True,
        "question_type": str(qa.get("question_type") or "unknown"),
        "source_turn_lookup_available": bool(source_turn_lookup),
        "dialogue_turn_count_before": before.get("dialogue_turn_count"),
        "dialogue_turn_count_after": after.get("dialogue_turn_count"),
        "missing_ref_count_before": before.get("total_missing_ref_count"),
        "missing_ref_count_after": after.get("total_missing_ref_count"),
        "unsupported_ref_count_after": after.get("total_unsupported_ref_count"),
        "diagnosis_cause_before": before.get("diagnosis_cause"),
        "diagnosis_cause_after": after.get("diagnosis_cause"),
        "warnings": warnings,
        **expansion_stats,
    }
    if ordering_original_gold is not None:
        repair_meta["ordering_original_gold"] = ordering_original_gold
    if ordering_recomputed_gold is not None:
        repair_meta["ordering_recomputed_gold"] = ordering_recomputed_gold
    if safe_to_rewrite_gold is not None:
        repair_meta["safe_to_rewrite_gold"] = safe_to_rewrite_gold
    repaired["generation_consistency_repair"] = repair_meta
    repair_report = {
        "sample_id": repaired.get("sample_id"),
        "question_type": repair_meta["question_type"],
        "added_turn_count": int(repair_meta.get("added_turn_count") or 0),
        "missing_ref_count_before": int(repair_meta.get("missing_ref_count_before") or 0),
        "missing_ref_count_after": int(repair_meta.get("missing_ref_count_after") or 0),
        "unsupported_ref_count_after": int(repair_meta.get("unsupported_ref_count_after") or 0),
        "fixed_missing_refs_before_cleanup": (
            int(repair_meta.get("missing_ref_count_before") or 0) > 0
            and int(repair_meta.get("missing_ref_count_after") or 0) == 0
        ),
        "ordering_gold_mismatch": "ordering_gold_mismatch_after_visible_trim" in warnings,
        "ordering_option_no_visible_support": "ordering_option_has_no_visible_support" in warnings,
        "multi_select_option_no_visible_support": "multi_select_option_has_no_visible_support" in warnings,
        "warnings": warnings,
        "diagnosis_cause_after": repair_meta.get("diagnosis_cause_after"),
    }
    return repaired, repair_report


def cleanup_generated_sample_turn_refs(sample: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    cleaned = reset_turn_coverage_cleanup_artifacts(sample)
    qa = cleaned.get("qa") if isinstance(cleaned.get("qa"), dict) else {}
    graph_update = cleaned.get("graph_update") if isinstance(cleaned.get("graph_update"), dict) else {}
    visible_turn_ids, turn_text_by_id = _sample_visible_turns(cleaned)
    question_type = str(qa.get("question_type") or "unknown")
    cleanup: dict[str, Any] = {
        "applied": True,
        "valid_after_cleanup": True,
        "warnings": [],
        "visible_turn_count": len(visible_turn_ids),
        "trimmed_ref_count": 0,
        "missing_ref_count": 0,
        "unsupported_ref_count": 0,
        "question_type": question_type,
    }
    cleaned["turn_coverage_cleanup"] = cleanup

    any_graph_support_visible = False
    ordering_option_turns: list[tuple[str, list[str]]] = []

    original_qa_evidence = _normalize_turn_ids(qa.get("evidence_turn_ids"))
    if original_qa_evidence:
        expected_text = None
        gold = qa.get("gold")
        if question_type == "state_reconstruction":
            if isinstance(gold, list) and gold:
                expected_text = str(gold[0])
            elif gold is not None:
                expected_text = str(gold)
        kept_turn_ids, _stats = _cleanup_turn_id_field(
            sample=cleaned,
            cleanup=cleanup,
            container=qa,
            field_name="evidence_turn_ids",
            visible_turn_ids=visible_turn_ids,
            turn_text_by_id=turn_text_by_id,
            expected_text=expected_text,
        )
        if not kept_turn_ids:
            _register_cleanup_warning(
                cleanup,
                cleaned,
                "turn_coverage_cleanup_warning:qa_evidence_turn_ids_empty_after_cleanup",
                invalid=question_type in {"evidence_selection"},
            )

    options = qa.get("options")
    if isinstance(options, list):
        for index, option in enumerate(options):
            if not isinstance(option, dict):
                continue
            original_option_turn_ids = _normalize_turn_ids(option.get("turn_ids"))
            if not original_option_turn_ids:
                continue
            kept_turn_ids, _stats = _cleanup_turn_id_field(
                sample=cleaned,
                cleanup=cleanup,
                container=option,
                field_name="turn_ids",
                visible_turn_ids=visible_turn_ids,
                turn_text_by_id=turn_text_by_id,
                expected_text=str(option.get("text") or ""),
            )
            if question_type == "ordering":
                ordering_option_turns.append((str(option.get("id") or f"option_{index}"), kept_turn_ids))
            if not kept_turn_ids:
                _register_cleanup_warning(
                    cleanup,
                    cleaned,
                    f"turn_coverage_cleanup_warning:qa_option_turn_ids_empty:{index}",
                    invalid=question_type in {"multi_select", "ordering"},
                )

    events = graph_update.get("events")
    if isinstance(events, list):
        for index, event in enumerate(events):
            if not isinstance(event, dict):
                continue
            original_turn_ids = _normalize_turn_ids(event.get("turn_ids"))
            if not original_turn_ids:
                continue
            kept_turn_ids, _stats = _cleanup_turn_id_field(
                sample=cleaned,
                cleanup=cleanup,
                container=event,
                field_name="turn_ids",
                visible_turn_ids=visible_turn_ids,
                turn_text_by_id=turn_text_by_id,
                expected_text=_expected_text_from_item(event),
            )
            if kept_turn_ids:
                any_graph_support_visible = True
            else:
                _register_cleanup_warning(
                    cleanup,
                    cleaned,
                    f"turn_coverage_cleanup_warning:graph_event_turn_ids_empty:{index}",
                    invalid=True,
                )

    state_changes = graph_update.get("state_changes")
    if isinstance(state_changes, list):
        for index, state_change in enumerate(state_changes):
            if not isinstance(state_change, dict):
                continue
            original_turn_ids = _normalize_turn_ids(state_change.get("evidence_turn_ids"))
            if not original_turn_ids:
                continue
            expected_text = base.normalize_space(state_change.get("to") or state_change.get("from"))
            kept_turn_ids, _stats = _cleanup_turn_id_field(
                sample=cleaned,
                cleanup=cleanup,
                container=state_change,
                field_name="evidence_turn_ids",
                visible_turn_ids=visible_turn_ids,
                turn_text_by_id=turn_text_by_id,
                expected_text=expected_text,
            )
            if kept_turn_ids:
                any_graph_support_visible = True
            else:
                _register_cleanup_warning(
                    cleanup,
                    cleaned,
                    f"turn_coverage_cleanup_warning:state_change_evidence_empty:{index}",
                    invalid=True,
                )

    for group_name in ("invalidations", "restorations", "cancellations", "delayed_activations"):
        items = graph_update.get(group_name)
        if not isinstance(items, list):
            continue
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            expected_text = _expected_text_from_item(item)
            for field_name in COMMON_TURN_ID_FIELD_NAMES:
                original_turn_ids = _normalize_turn_ids(item.get(field_name))
                if not original_turn_ids:
                    continue
                kept_turn_ids, _stats = _cleanup_turn_id_field(
                    sample=cleaned,
                    cleanup=cleanup,
                    container=item,
                    field_name=field_name,
                    visible_turn_ids=visible_turn_ids,
                    turn_text_by_id=turn_text_by_id,
                    expected_text=expected_text,
                )
                if kept_turn_ids:
                    any_graph_support_visible = True
                else:
                    _register_cleanup_warning(
                        cleanup,
                        cleaned,
                        f"turn_coverage_cleanup_warning:{group_name}_{field_name}_empty:{index}",
                        invalid=True,
                    )

    if question_type == "ordering":
        if any(not kept_turn_ids for _option_id, kept_turn_ids in ordering_option_turns):
            _register_cleanup_warning(
                cleanup,
                cleaned,
                "turn_coverage_cleanup_failed:ordering_option_has_no_visible_support",
                invalid=True,
            )
        else:
            recomputed_gold = [
                option_id
                for option_id, kept_turn_ids in sorted(
                    ordering_option_turns,
                    key=lambda item: (_turn_sort_key(sorted(item[1], key=_turn_sort_key)[0]), item[0]),
                )
            ]
            original_gold = [str(item) for item in (qa.get("gold") or [])]
            if recomputed_gold != original_gold:
                _register_cleanup_warning(
                    cleanup,
                    cleaned,
                    "turn_coverage_cleanup_failed:ordering_gold_mismatch_after_visible_trim",
                    invalid=True,
                )
    elif question_type == "evidence_selection":
        if original_qa_evidence and not _normalize_turn_ids(qa.get("evidence_turn_ids")):
            _register_cleanup_warning(
                cleanup,
                cleaned,
                "turn_coverage_cleanup_failed:evidence_selection_has_no_visible_evidence",
                invalid=True,
            )
    elif question_type == "multi_select":
        if isinstance(options, list):
            for index, option in enumerate(options):
                if not isinstance(option, dict):
                    continue
                original_turn_ids = _normalize_turn_ids((sample.get("qa") or {}).get("options", [])[index].get("turn_ids")) if isinstance((sample.get("qa") or {}).get("options"), list) and index < len((sample.get("qa") or {}).get("options")) and isinstance((sample.get("qa") or {}).get("options")[index], dict) else []
                if original_turn_ids and not _normalize_turn_ids(option.get("turn_ids")):
                    _register_cleanup_warning(
                        cleanup,
                        cleaned,
                        "turn_coverage_cleanup_failed:multi_select_option_has_no_visible_support",
                        invalid=True,
                    )
                    break
    elif question_type == "state_reconstruction":
        qa_evidence_after = _normalize_turn_ids(qa.get("evidence_turn_ids"))
        if not qa_evidence_after and not any_graph_support_visible:
            _register_cleanup_warning(
                cleanup,
                cleaned,
                "turn_coverage_cleanup_failed:state_reconstruction_has_no_visible_support",
                invalid=True,
            )

    coverage_audit = _build_post_generation_turn_coverage_audit([cleaned])
    if int(coverage_audit.get("rows_with_missing_refs") or 0) > 0:
        _register_cleanup_warning(
            cleanup,
            cleaned,
            "turn_coverage_cleanup_failed:residual_missing_refs_after_cleanup",
            invalid=True,
        )

    cleanup["valid_after_cleanup"] = cleaned.get("validation", {}).get("graph_dialogue_consistent") is True
    reset_stats = (
        cleaned.pop(TURN_COVERAGE_CLEANUP_RESET_STATS_KEY)
        if isinstance(cleaned.get(TURN_COVERAGE_CLEANUP_RESET_STATS_KEY), dict)
        else {
            "cleanup_artifact_reset_applied": True,
            "cleanup_artifact_reset_count": 1,
            "stale_cleanup_warning_removed_count": 0,
            "stale_cleanup_invalid_flag_reset_count": 0,
        }
    )
    cleanup["artifact_reset"] = {
        "applied": bool(reset_stats.get("cleanup_artifact_reset_applied")),
        "count": int(reset_stats.get("cleanup_artifact_reset_count") or 0),
        "stale_cleanup_warning_removed_count": int(reset_stats.get("stale_cleanup_warning_removed_count") or 0),
        "stale_cleanup_invalid_flag_reset_count": int(reset_stats.get("stale_cleanup_invalid_flag_reset_count") or 0),
    }
    cleanup_report = {
        "sample_id": cleaned.get("sample_id"),
        "question_type": question_type,
        "valid_after_cleanup": cleanup["valid_after_cleanup"],
        "warnings": list(cleanup.get("warnings") or []),
        "trimmed_ref_count": int(cleanup.get("trimmed_ref_count") or 0),
        "missing_ref_count": int(cleanup.get("missing_ref_count") or 0),
        "unsupported_ref_count": int(cleanup.get("unsupported_ref_count") or 0),
        "cleanup_artifact_reset_applied": bool(reset_stats.get("cleanup_artifact_reset_applied")),
        "cleanup_artifact_reset_count": int(reset_stats.get("cleanup_artifact_reset_count") or 0),
        "stale_cleanup_warning_removed_count": int(reset_stats.get("stale_cleanup_warning_removed_count") or 0),
        "stale_cleanup_invalid_flag_reset_count": int(reset_stats.get("stale_cleanup_invalid_flag_reset_count") or 0),
    }
    cleaned["qa"] = qa
    cleaned["graph_update"] = graph_update
    cleaned["turn_coverage_cleanup"] = cleanup
    return cleaned, cleanup_report


def _build_turn_cleanup_summary(cleanup_reports: list[dict[str, Any]]) -> dict[str, Any]:
    by_question_type: dict[str, dict[str, int]] = {}
    warning_counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    cleanup_invalid_count = 0
    valid_after_cleanup_count = 0
    trimmed_ref_total = 0
    missing_ref_total = 0
    unsupported_ref_total = 0
    cleanup_artifact_reset_count = 0
    stale_cleanup_warning_removed_count = 0
    stale_cleanup_invalid_flag_reset_count = 0
    for report in cleanup_reports:
        question_type = str(report.get("question_type") or "unknown")
        stats = by_question_type.setdefault(
            question_type,
            {
                "n": 0,
                "cleanup_invalid_count": 0,
                "trimmed_ref_total": 0,
                "missing_ref_total": 0,
                "unsupported_ref_total": 0,
            },
        )
        stats["n"] += 1
        trimmed = int(report.get("trimmed_ref_count") or 0)
        missing = int(report.get("missing_ref_count") or 0)
        unsupported = int(report.get("unsupported_ref_count") or 0)
        stats["trimmed_ref_total"] += trimmed
        stats["missing_ref_total"] += missing
        stats["unsupported_ref_total"] += unsupported
        trimmed_ref_total += trimmed
        missing_ref_total += missing
        unsupported_ref_total += unsupported
        cleanup_artifact_reset_count += int(report.get("cleanup_artifact_reset_count") or 0)
        stale_cleanup_warning_removed_count += int(report.get("stale_cleanup_warning_removed_count") or 0)
        stale_cleanup_invalid_flag_reset_count += int(report.get("stale_cleanup_invalid_flag_reset_count") or 0)
        if report.get("valid_after_cleanup") is True:
            valid_after_cleanup_count += 1
        else:
            cleanup_invalid_count += 1
            stats["cleanup_invalid_count"] += 1
        warnings = list(report.get("warnings") or [])
        warning_counts.update(warnings)
        if warnings and len(examples) < 20:
            examples.append(
                {
                    "sample_id": report.get("sample_id"),
                    "question_type": question_type,
                    "warnings": warnings,
                    "trimmed_ref_count": trimmed,
                    "missing_ref_count": missing,
                    "unsupported_ref_count": unsupported,
                }
            )
    return {
        "samples": len(cleanup_reports),
        "cleanup_applied": len(cleanup_reports),
        "cleanup_artifact_reset_applied": True,
        "cleanup_artifact_reset_count": cleanup_artifact_reset_count,
        "stale_cleanup_warning_removed_count": stale_cleanup_warning_removed_count,
        "stale_cleanup_invalid_flag_reset_count": stale_cleanup_invalid_flag_reset_count,
        "cleanup_invalid_count": cleanup_invalid_count,
        "valid_after_cleanup_count": valid_after_cleanup_count,
        "trimmed_ref_total": trimmed_ref_total,
        "missing_ref_total": missing_ref_total,
        "unsupported_ref_total": unsupported_ref_total,
        "by_question_type": by_question_type,
        "warning_counts": dict(sorted(warning_counts.items())),
        "examples": examples,
    }


def _build_generation_consistency_repair_summary(
    repair_reports: list[dict[str, Any]],
    cleanup_reports: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    by_question_type: dict[str, dict[str, int]] = {}
    warning_counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    ordering_gold_mismatch_count = 0
    ordering_option_no_visible_support_count = 0
    multi_select_option_no_visible_support_count = 0
    samples_fixed_before_cleanup = 0
    samples_left_for_cleanup_invalid = 0
    cleanup_invalid_by_sample_id = {
        str(report.get("sample_id")): report
        for report in (cleanup_reports or [])
        if report.get("valid_after_cleanup") is not True
    }
    for report in repair_reports:
        question_type = str(report.get("question_type") or "unknown")
        stats = by_question_type.setdefault(
            question_type,
            {
                "n": 0,
                "samples_fixed_before_cleanup": 0,
                "ordering_gold_mismatch_count": 0,
                "ordering_option_no_visible_support_count": 0,
                "multi_select_option_no_visible_support_count": 0,
            },
        )
        stats["n"] += 1
        warning_counts.update(list(report.get("warnings") or []))
        if report.get("fixed_missing_refs_before_cleanup") is True:
            samples_fixed_before_cleanup += 1
            stats["samples_fixed_before_cleanup"] += 1
        if report.get("ordering_gold_mismatch") is True:
            ordering_gold_mismatch_count += 1
            stats["ordering_gold_mismatch_count"] += 1
        if report.get("ordering_option_no_visible_support") is True:
            ordering_option_no_visible_support_count += 1
            stats["ordering_option_no_visible_support_count"] += 1
        if report.get("multi_select_option_no_visible_support") is True:
            multi_select_option_no_visible_support_count += 1
            stats["multi_select_option_no_visible_support_count"] += 1
        if str(report.get("sample_id")) in cleanup_invalid_by_sample_id:
            samples_left_for_cleanup_invalid += 1
        if (report.get("warnings") or []) and len(examples) < 20:
            examples.append(
                {
                    "sample_id": report.get("sample_id"),
                    "question_type": question_type,
                    "warnings": list(report.get("warnings") or []),
                    "added_turn_count": int(report.get("added_turn_count") or 0),
                    "missing_ref_count_before": int(report.get("missing_ref_count_before") or 0),
                    "missing_ref_count_after": int(report.get("missing_ref_count_after") or 0),
                    "unsupported_ref_count_after": int(report.get("unsupported_ref_count_after") or 0),
                    "diagnosis_cause_after": report.get("diagnosis_cause_after"),
                }
            )
    return {
        "samples": len(repair_reports),
        "ordering_gold_mismatch_count": ordering_gold_mismatch_count,
        "ordering_option_no_visible_support_count": ordering_option_no_visible_support_count,
        "multi_select_option_no_visible_support_count": multi_select_option_no_visible_support_count,
        "samples_retried": 0,
        "samples_fixed_before_cleanup": samples_fixed_before_cleanup,
        "samples_left_for_cleanup_invalid": samples_left_for_cleanup_invalid,
        "by_question_type": by_question_type,
        "warning_counts": dict(sorted(warning_counts.items())),
        "examples": examples,
    }


def _build_generation_visible_support_guard_summary(guard_reports: list[dict[str, Any]]) -> dict[str, Any]:
    by_question_type: dict[str, dict[str, int]] = {}
    examples: list[dict[str, Any]] = []
    ordering_samples = 0
    multi_select_samples = 0
    ordering_all_options_supported = 0
    ordering_gold_match_visible_order = 0
    ordering_gold_mismatch_visible_order = 0
    ordering_option_no_support = 0
    multi_select_all_options_supported = 0
    multi_select_option_no_support = 0
    for report in guard_reports:
        question_type = str(report.get("question_type") or "unknown")
        stats = by_question_type.setdefault(
            question_type,
            {
                "n": 0,
                "all_options_supported": 0,
                "gold_match_visible_order": 0,
                "gold_mismatch_visible_order": 0,
                "option_no_support": 0,
                "gold_option_no_support": 0,
                "invalidated_by_guard": 0,
            },
        )
        stats["n"] += 1
        if report.get("all_options_supported") is True:
            stats["all_options_supported"] += 1
        if report.get("gold_match_visible_order") is True:
            stats["gold_match_visible_order"] += 1
        if report.get("gold_mismatch_visible_order") is True:
            stats["gold_mismatch_visible_order"] += 1
        if report.get("option_no_support") is True:
            stats["option_no_support"] += 1
        if report.get("gold_option_no_support") is True:
            stats["gold_option_no_support"] += 1
        if report.get("invalidated_by_guard") is True:
            stats["invalidated_by_guard"] += 1
        if question_type == "ordering":
            ordering_samples += 1
            if report.get("all_options_supported") is True:
                ordering_all_options_supported += 1
            if report.get("gold_match_visible_order") is True:
                ordering_gold_match_visible_order += 1
            if report.get("gold_mismatch_visible_order") is True:
                ordering_gold_mismatch_visible_order += 1
            if report.get("option_no_support") is True:
                ordering_option_no_support += 1
        elif question_type == "multi_select":
            multi_select_samples += 1
            if report.get("all_options_supported") is True:
                multi_select_all_options_supported += 1
            if report.get("option_no_support") is True:
                multi_select_option_no_support += 1
        if report.get("warnings") and len(examples) < 20:
            examples.append(
                {
                    "sample_id": report.get("sample_id"),
                    "question_type": question_type,
                    "warnings": list(report.get("warnings") or []),
                    "all_options_supported": report.get("all_options_supported"),
                    "gold_match_visible_order": report.get("gold_match_visible_order"),
                    "gold_mismatch_visible_order": report.get("gold_mismatch_visible_order"),
                    "option_no_support": report.get("option_no_support"),
                    "gold_option_no_support": report.get("gold_option_no_support"),
                    "options_with_no_support": list(report.get("options_with_no_support") or []),
                    "ordering_visible_order": list(report.get("ordering_visible_order") or []),
                    "ordering_original_gold": list(report.get("ordering_original_gold") or []),
                }
            )
    return {
        "samples": len(guard_reports),
        "ordering_samples": ordering_samples,
        "multi_select_samples": multi_select_samples,
        "ordering_all_options_supported": ordering_all_options_supported,
        "ordering_gold_match_visible_order": ordering_gold_match_visible_order,
        "ordering_gold_mismatch_visible_order": ordering_gold_mismatch_visible_order,
        "ordering_option_no_support": ordering_option_no_support,
        "multi_select_all_options_supported": multi_select_all_options_supported,
        "multi_select_option_no_support": multi_select_option_no_support,
        "by_question_type": by_question_type,
        "examples": examples,
    }


def _build_generation_programmatic_builder_summary(builder_reports: list[dict[str, Any]]) -> dict[str, Any]:
    ordering_samples = 0
    ordering_rebuilt = 0
    ordering_not_rebuilt = 0
    multi_select_samples = 0
    multi_select_rebuilt = 0
    multi_select_not_rebuilt = 0
    not_rebuilt_reasons: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for report in builder_reports:
        question_type = str(report.get("question_type") or "unknown")
        rebuilt = report.get("rebuilt") is True
        if question_type == "ordering":
            ordering_samples += 1
            if rebuilt:
                ordering_rebuilt += 1
            else:
                ordering_not_rebuilt += 1
        elif question_type == "multi_select":
            multi_select_samples += 1
            if rebuilt:
                multi_select_rebuilt += 1
            else:
                multi_select_not_rebuilt += 1
        reason = str(report.get("reason") or "")
        if reason and reason not in {"question_type_not_ordering", "question_type_not_multi_select"}:
            not_rebuilt_reasons[reason] += 1
        if len(examples) < 20 and (rebuilt or reason):
            examples.append(
                {
                    "sample_id": report.get("sample_id"),
                    "question_type": question_type,
                    "rebuilt": rebuilt,
                    "reason": report.get("reason"),
                    "selected_candidate_ids": list(report.get("selected_candidate_ids") or []),
                    "selected_positive_ids": list(report.get("selected_positive_ids") or []),
                    "selected_negative_ids": list(report.get("selected_negative_ids") or []),
                    "visible_order": list(report.get("visible_order") or []),
                }
            )
    return {
        "samples": len(builder_reports),
        "ordering_samples": ordering_samples,
        "ordering_rebuilt": ordering_rebuilt,
        "ordering_not_rebuilt": ordering_not_rebuilt,
        "multi_select_samples": multi_select_samples,
        "multi_select_rebuilt": multi_select_rebuilt,
        "multi_select_not_rebuilt": multi_select_not_rebuilt,
        "not_rebuilt_reasons": dict(sorted(not_rebuilt_reasons.items())),
        "examples": examples,
    }


def _sample_candidate_bank(sample: dict[str, Any]) -> dict[str, Any]:
    existing_top_level = sample.get("generation_candidate_bank") if isinstance(sample.get("generation_candidate_bank"), dict) else None
    if isinstance(existing_top_level, dict):
        return copy.deepcopy(existing_top_level)
    graph_update = sample.get("graph_update") if isinstance(sample.get("graph_update"), dict) else {}
    existing = graph_update.get("generation_candidate_bank") if isinstance(graph_update.get("generation_candidate_bank"), dict) else None
    if isinstance(existing, dict):
        return copy.deepcopy(existing)
    return build_generation_candidate_bank(sample)


def _count_stats_from_bank_values(values: list[int]) -> dict[str, float]:
    return _count_stats(values)


def _build_generation_candidate_bank_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    ordering_candidate_counts: list[int] = []
    multi_select_positive_counts: list[int] = []
    multi_select_negative_counts: list[int] = []
    ordering_ready_count = 0
    multi_select_ready_count = 0
    ordering_samples = 0
    multi_select_samples = 0
    not_ready_reasons: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for sample in samples:
        qa = sample.get("qa") if isinstance(sample.get("qa"), dict) else {}
        question_type = str(qa.get("question_type") or "unknown")
        bank = _sample_candidate_bank(sample)
        ordering_candidate_counts.append(int(bank.get("ordering_candidate_count") or 0))
        multi_select_positive_counts.append(int(bank.get("multi_select_positive_count") or 0))
        multi_select_negative_counts.append(int(bank.get("multi_select_negative_count") or 0))
        reasons: list[str] = []
        if question_type == "ordering":
            ordering_samples += 1
            if bank.get("ordering_ready") is True:
                ordering_ready_count += 1
            else:
                reasons.append("ordering_insufficient_distinct_candidates")
        elif question_type == "multi_select":
            multi_select_samples += 1
            if bank.get("multi_select_ready") is True:
                multi_select_ready_count += 1
            else:
                if int(bank.get("multi_select_positive_count") or 0) < 2:
                    reasons.append("multi_select_insufficient_positive_candidates")
                if int(bank.get("multi_select_negative_count") or 0) < 2:
                    reasons.append("multi_select_insufficient_safe_negative_candidates")
        for reason in reasons:
            not_ready_reasons[reason] += 1
        if len(examples) < 20 and (question_type in {"ordering", "multi_select"}):
            examples.append(
                {
                    "sample_id": sample.get("sample_id"),
                    "question_type": question_type,
                    "ordering_candidate_count": int(bank.get("ordering_candidate_count") or 0),
                    "ordering_ready": bool(bank.get("ordering_ready")),
                    "multi_select_positive_count": int(bank.get("multi_select_positive_count") or 0),
                    "multi_select_negative_count": int(bank.get("multi_select_negative_count") or 0),
                    "multi_select_ready": bool(bank.get("multi_select_ready")),
                    "not_ready_reasons": reasons,
                }
            )
    return {
        "samples": len(samples),
        "ordering_samples": ordering_samples,
        "multi_select_samples": multi_select_samples,
        "ordering_candidate_count_stats": _count_stats_from_bank_values(ordering_candidate_counts),
        "multi_select_positive_count_stats": _count_stats_from_bank_values(multi_select_positive_counts),
        "multi_select_negative_count_stats": _count_stats_from_bank_values(multi_select_negative_counts),
        "ordering_ready_count": ordering_ready_count,
        "multi_select_ready_count": multi_select_ready_count,
        "not_ready_reasons": dict(sorted(not_ready_reasons.items())),
        "examples": examples,
    }


def _build_adapter_summary(input_rows: list[dict[str, Any]], augmented_rows: list[dict[str, Any]]) -> dict[str, Any]:
    turn_counts: list[int] = []
    node_counts: list[int] = []
    rows_with_final_dialogue = 0
    rows_with_state_chain_nodes = 0
    missing_node_turn_ref_total_after_filter = 0
    rows_with_missing_node_turn_refs_after_filter = 0
    raw_legacy_node_count_total = 0
    filtered_legacy_node_count_total = 0
    dropped_legacy_node_count_total = 0
    for row in augmented_rows:
        adapter_meta = row.get("_current_method_old_gate_adapter") if isinstance(row.get("_current_method_old_gate_adapter"), dict) else {}
        coverage = adapter_meta.get("turn_coverage_summary") if isinstance(adapter_meta.get("turn_coverage_summary"), dict) else {}
        final_dialogue_turn_count = int(coverage.get("final_dialogue_turn_count") or 0)
        state_chain_node_count = int(coverage.get("state_chain_node_count") or 0)
        missing_ref_count = int(coverage.get("missing_node_turn_ref_count") or 0)
        turn_counts.append(final_dialogue_turn_count)
        node_counts.append(state_chain_node_count)
        if final_dialogue_turn_count > 0:
            rows_with_final_dialogue += 1
        if state_chain_node_count > 0:
            rows_with_state_chain_nodes += 1
        missing_node_turn_ref_total_after_filter += missing_ref_count
        if missing_ref_count > 0:
            rows_with_missing_node_turn_refs_after_filter += 1
        raw_count = int(adapter_meta.get("raw_legacy_node_count") or 0)
        filtered_count = int(adapter_meta.get("filtered_legacy_node_count") or 0)
        raw_legacy_node_count_total += raw_count
        filtered_legacy_node_count_total += filtered_count
        dropped_legacy_node_count_total += max(0, raw_count - filtered_count)
    return {
        "input_rows": len(input_rows),
        "augmented_rows": len(augmented_rows),
        "rows_with_final_dialogue": rows_with_final_dialogue,
        "rows_with_state_chain_nodes": rows_with_state_chain_nodes,
        "final_dialogue_turn_count_stats": _count_stats(turn_counts),
        "state_chain_node_count_stats": _count_stats(node_counts),
        "missing_node_turn_ref_total_after_filter": missing_node_turn_ref_total_after_filter,
        "rows_with_missing_node_turn_refs_after_filter": rows_with_missing_node_turn_refs_after_filter,
        "raw_legacy_node_count_total": raw_legacy_node_count_total,
        "filtered_legacy_node_count_total": filtered_legacy_node_count_total,
        "dropped_legacy_node_count_total": dropped_legacy_node_count_total,
        "note": "current_method rows are filtered so legacy state_chain_nodes only reference final_dialogue-visible turn_ids",
    }


def _collect_missing_refs(
    missing_refs: list[dict[str, Any]],
    *,
    field: str,
    value: Any,
    visible_turn_ids: set[str],
) -> int:
    missing_count = 0
    for turn_id in _normalize_turn_ids(value):
        if turn_id not in visible_turn_ids:
            missing_count += 1
            if len(missing_refs) < 20:
                missing_refs.append({"field": field, "turn_id": turn_id})
    return missing_count


def _build_post_generation_turn_coverage_audit(samples: list[dict[str, Any]]) -> dict[str, Any]:
    by_question_type: dict[str, dict[str, int]] = {}
    examples: list[dict[str, Any]] = []
    rows_with_missing_refs = 0
    missing_ref_total = 0
    valid_after_cleanup_samples = 0
    valid_after_cleanup_rows_with_missing_refs = 0
    valid_after_cleanup_missing_ref_total = 0
    for sample in samples:
        dialogue = sample.get("dialogue") if isinstance(sample.get("dialogue"), dict) else {}
        visible_turn_ids = _visible_turn_ids_from_legacy_turns(dialogue.get("turns"))
        qa = sample.get("qa") if isinstance(sample.get("qa"), dict) else {}
        graph_update = sample.get("graph_update") if isinstance(sample.get("graph_update"), dict) else {}
        question_type = str(qa.get("question_type") or "unknown")
        is_valid_after_cleanup = (
            (sample.get("turn_coverage_cleanup") or {}).get("valid_after_cleanup") is True
            if isinstance(sample.get("turn_coverage_cleanup"), dict)
            else sample.get("validation", {}).get("graph_dialogue_consistent") is True
        )
        if is_valid_after_cleanup:
            valid_after_cleanup_samples += 1
        stats = by_question_type.setdefault(
            question_type,
            {"n": 0, "rows_with_missing_refs": 0, "missing_ref_total": 0},
        )
        stats["n"] += 1

        missing_refs: list[dict[str, Any]] = []
        sample_missing_count = 0
        sample_missing_count += _collect_missing_refs(
            missing_refs,
            field="qa.evidence_turn_ids",
            value=qa.get("evidence_turn_ids"),
            visible_turn_ids=visible_turn_ids,
        )
        options = qa.get("options")
        if isinstance(options, list):
            for index, option in enumerate(options):
                if isinstance(option, dict):
                    sample_missing_count += _collect_missing_refs(
                        missing_refs,
                        field=f"qa.options[{index}].turn_ids",
                        value=option.get("turn_ids"),
                        visible_turn_ids=visible_turn_ids,
                    )
        events = graph_update.get("events")
        if isinstance(events, list):
            for index, event in enumerate(events):
                if isinstance(event, dict):
                    sample_missing_count += _collect_missing_refs(
                        missing_refs,
                        field=f"graph_update.events[{index}].turn_ids",
                        value=event.get("turn_ids"),
                        visible_turn_ids=visible_turn_ids,
                    )
        state_changes = graph_update.get("state_changes")
        if isinstance(state_changes, list):
            for index, change in enumerate(state_changes):
                if isinstance(change, dict):
                    sample_missing_count += _collect_missing_refs(
                        missing_refs,
                        field=f"graph_update.state_changes[{index}].evidence_turn_ids",
                        value=change.get("evidence_turn_ids"),
                        visible_turn_ids=visible_turn_ids,
                    )
        invalidations = graph_update.get("invalidations")
        if isinstance(invalidations, list):
            for index, invalidation in enumerate(invalidations):
                if isinstance(invalidation, dict):
                    sample_missing_count += _collect_missing_refs(
                        missing_refs,
                        field=f"graph_update.invalidations[{index}].invalidation_turn_ids",
                        value=invalidation.get("invalidation_turn_ids"),
                        visible_turn_ids=visible_turn_ids,
                    )
        restorations = graph_update.get("restorations")
        if isinstance(restorations, list):
            for index, restoration in enumerate(restorations):
                if isinstance(restoration, dict):
                    sample_missing_count += _collect_missing_refs(
                        missing_refs,
                        field=f"graph_update.restorations[{index}].turn_ids",
                        value=restoration.get("turn_ids"),
                        visible_turn_ids=visible_turn_ids,
                    )
        cancellations = graph_update.get("cancellations")
        if isinstance(cancellations, list):
            for index, cancellation in enumerate(cancellations):
                if isinstance(cancellation, dict):
                    sample_missing_count += _collect_missing_refs(
                        missing_refs,
                        field=f"graph_update.cancellations[{index}].turn_ids",
                        value=cancellation.get("turn_ids"),
                        visible_turn_ids=visible_turn_ids,
                    )
        delayed_activations = graph_update.get("delayed_activations")
        if isinstance(delayed_activations, list):
            for index, activation in enumerate(delayed_activations):
                if isinstance(activation, dict):
                    sample_missing_count += _collect_missing_refs(
                        missing_refs,
                        field=f"graph_update.delayed_activations[{index}].activation_turn_ids",
                        value=activation.get("activation_turn_ids"),
                        visible_turn_ids=visible_turn_ids,
                    )

        missing_ref_total += sample_missing_count
        stats["missing_ref_total"] += sample_missing_count
        if sample_missing_count > 0:
            rows_with_missing_refs += 1
            stats["rows_with_missing_refs"] += 1
            if is_valid_after_cleanup:
                valid_after_cleanup_rows_with_missing_refs += 1
                valid_after_cleanup_missing_ref_total += sample_missing_count
            if len(examples) < 20:
                examples.append(
                    {
                        "sample_id": sample.get("sample_id"),
                        "question_type": question_type,
                        "dialogue_turn_count": len(visible_turn_ids),
                        "missing_refs": missing_refs,
                    }
                )
    return {
        "samples": len(samples),
        "rows_with_missing_refs": rows_with_missing_refs,
        "missing_ref_total": missing_ref_total,
        "valid_after_cleanup_samples": valid_after_cleanup_samples,
        "valid_after_cleanup_rows_with_missing_refs": valid_after_cleanup_rows_with_missing_refs,
        "valid_after_cleanup_missing_ref_total": valid_after_cleanup_missing_ref_total,
        "by_question_type": by_question_type,
        "examples": examples,
    }


def _weights_to_quotas(weights: dict[str, Any], *, total: int = 1000) -> list[tuple[str, int]]:
    values: list[tuple[str, float]] = []
    for key, value in sorted((weights or {}).items()):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric > 0:
            values.append((str(key), numeric))
    if not values:
        raise ValueError("Compiled policy has no positive weights.")
    denom = sum(value for _key, value in values)
    allocations: list[tuple[str, int, float]] = []
    remaining = total
    for key, value in values:
        raw = total * value / denom
        base_count = int(raw)
        allocations.append((key, base_count, raw - base_count))
        remaining -= base_count
    allocations.sort(key=lambda item: item[2], reverse=True)
    extra = {key: 0 for key, _base, _frac in allocations}
    for key, _base, _frac in allocations[:remaining]:
        extra[key] += 1
    by_key = {key: count + extra[key] for key, count, _frac in allocations}
    return [(key, max(1, by_key[key])) for key, _value in values]


def _patch_old_gate_for_current_method(
    *,
    run_name: str,
    seed_scope: str,
    fixed_solver_name: str,
    compiled_policy: dict[str, Any],
) -> None:
    old_gate.EXPERIMENT = run_name
    old_gate.SEED_SCOPE = seed_scope
    old_gate.FIXED_SOLVER_NAME = fixed_solver_name
    old_gate.ALLOWED_SAMPLE_NAMES = set(seed_audit.DEFAULT_ALLOWED_SAMPLE_NAMES)
    old_gate.extract_seed_source = lambda row: seed_audit.extract_seed_source(
        row,
        allowed_sample_names=seed_audit.DEFAULT_ALLOWED_SAMPLE_NAMES,
    )
    if _ORIGINAL_BASE_BUILD_QA_AND_GRAPH is not None:
        base.build_qa_and_graph = _build_qa_and_graph_with_turn_ref_coverage
    if _ORIGINAL_OLD_GATE_BUILD_REWRITE_PROMPT is not None:
        old_gate.build_rewrite_prompt = _build_rewrite_prompt_with_consistency_constraints
    base.ATTACK_SKILL_QUOTAS = _weights_to_quotas(compiled_policy.get("attack_skill_weights") or {})
    base.QUESTION_TYPE_QUOTAS = _weights_to_quotas(compiled_policy.get("question_type_weights") or {})


def _generation_args(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    partial_samples: Path,
    failed_api_calls: Path,
    compiled_policy: dict[str, Any],
) -> argparse.Namespace:
    config_args = argparse.Namespace(
        config=str(args.config),
        challenger_base_url=getattr(args, "challenger_base_url", None),
        challenger_model=getattr(args, "challenger_model", None),
        challenger_api_key=getattr(args, "challenger_api_key", None),
    )
    config_args = old_gate.apply_config_defaults(config_args)
    return argparse.Namespace(
        output_dir=str(output_dir),
        num_samples=int(args.n_train),
        k_eval=5,
        max_length=int(args.max_length),
        max_largest_question_type_share=float(args.max_largest_question_type_share),
        max_largest_attack_skill_share=float(args.max_largest_attack_skill_share),
        seed=int(args.seed),
        eval_backend=getattr(args, "eval_backend", "auto"),
        base_model=str(getattr(args, "base_model", None) or "/root/autodl-tmp/Qwen3-4B"),
        max_model_len=int(args.max_model_len),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(getattr(args, "temperature", 0.7)),
        top_p=float(getattr(args, "top_p", 0.9)),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        limit_seed_rows=getattr(args, "limit_seed_rows", None),
        config=str(args.config),
        challenger_base_url=config_args.challenger_base_url,
        challenger_model=config_args.challenger_model,
        challenger_api_key=config_args.challenger_api_key,
        llm_timeout=int(args.llm_timeout),
        max_llm_retries=int(args.max_llm_retries),
        max_api_failures=int(args.max_api_failures),
        challenger_temperature=float(getattr(args, "challenger_temperature", 0.25)),
        challenger_max_tokens=int(getattr(args, "challenger_max_tokens", 4096)),
        checkpoint_every=int(args.checkpoint_every),
        partial_samples=str(partial_samples),
        failed_api_calls=str(failed_api_calls),
        no_resume=bool(getattr(args, "no_resume", False)),
        quiet=bool(getattr(args, "quiet", False)),
        compiled_policy=compiled_policy,
    )


def _gold_answer_for_sft(sample: dict[str, Any]) -> str:
    qa = sample.get("qa") if isinstance(sample.get("qa"), dict) else {}
    gold = qa.get("gold")
    if isinstance(gold, list):
        if qa.get("question_type") == "ordering":
            return " > ".join(str(item) for item in gold)
        if qa.get("question_type") == "invalidated_statement_detection":
            return " | ".join(str(item) for item in gold)
        return ", ".join(str(item) for item in gold)
    return str(gold or "")


def build_sft_records(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sample in samples:
        records.append(
            {
                "record_id": sample.get("sample_id"),
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a long-dialogue graph-memory QA solver. Use only the dialogue evidence "
                            "and follow the requested output format exactly."
                        ),
                    },
                    {"role": "user", "content": base.solver_user_prompt(sample)},
                    {"role": "assistant", "content": _gold_answer_for_sft(sample)},
                ],
                "metadata": {
                    "method": "current_method",
                    "question_type": (sample.get("qa") or {}).get("question_type") if isinstance(sample.get("qa"), dict) else None,
                    "attack_skill": (sample.get("attack_plan") or {}).get("attack_skill")
                    if isinstance(sample.get("attack_plan"), dict)
                    else None,
                    "seed_scope": seed_audit.SEED_SCOPE,
                },
            }
        )
    return records


def generate_api_sample(bundle: Any, index: int, qtype: str, skill: str, rng: Any) -> dict[str, Any]:
    sample = old_gate.build_sample(bundle, index, qtype, skill, rng)
    if sample.get("challenger_api_used") is not True:
        raise RuntimeError(f"Challenger API was not used for generation index {index}.")
    return sample


def api_failure_summary(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(record.get("error_type") or "unknown") for record in records)
    return {
        "api_failure_count": len(records),
        "api_missing_keys_count": int(counts.get("missing_required_keys", 0)),
        "api_invalid_json_count": int(counts.get("invalid_json", 0)),
        "api_timeout_count": int(counts.get("timeout", 0)),
        "skipped_seed_count": len(records),
    }


def _read_failed_api_calls(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _touch_jsonl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("", encoding="utf-8")


def write_generation_outputs(
    *,
    output_dir: Path,
    samples: list[dict[str, Any]],
    compiled_policy: dict[str, Any],
    fixed_solver_name: str,
    seed_source: str | Path,
    partial_samples: Path,
    failed_api_calls: Path,
    requested_count: int,
    adapter_summary_path: Path | None = None,
    adapter_summary: dict[str, Any] | None = None,
    candidate_bank_summary_path: Path | None = None,
    candidate_bank_summary: dict[str, Any] | None = None,
    programmatic_builder_summary_path: Path | None = None,
    programmatic_builder_summary: dict[str, Any] | None = None,
    repair_summary_path: Path | None = None,
    repair_summary: dict[str, Any] | None = None,
    visible_support_guard_summary_path: Path | None = None,
    visible_support_guard_summary: dict[str, Any] | None = None,
    cleanup_summary_path: Path | None = None,
    cleanup_summary: dict[str, Any] | None = None,
    post_generation_turn_coverage_audit_path: Path | None = None,
    post_generation_turn_coverage_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    valid_samples = [sample for sample in samples if base.base_valid(sample)]
    write_jsonl(output_dir / "samples.jsonl", samples)
    write_jsonl(output_dir / "valid_samples.jsonl", valid_samples)
    write_jsonl(output_dir / "sft_train.jsonl", build_sft_records(samples))
    write_jsonl(output_dir / "sft_eval_dev.jsonl", [])
    write_jsonl(output_dir / "sft_eval_heldout.jsonl", [])
    write_jsonl(output_dir / "graph_updates.jsonl", [sample.get("graph_update", {}) for sample in samples])
    write_jsonl(output_dir / "expanded_graphs.jsonl", [sample.get("graph_update", {}) for sample in samples])
    write_jsonl(output_dir / "expanded_dialogues.jsonl", [sample.get("dialogue", {}) for sample in samples])
    _touch_jsonl(partial_samples)
    _touch_jsonl(failed_api_calls)
    failures = _read_failed_api_calls(failed_api_calls)
    api_stats = api_failure_summary(failures)
    summary = {
        "method": "current_method",
        "generation_source": "current_method.api_challenger_generation",
        "generation_logic_source": "validated_api_challenger_generation_derived",
        "legacy_split_parallel_generator_used": False,
        "local_template_fallback_used": False,
        "policy_id": compiled_policy.get("policy_id"),
        "policy_source": compiled_policy.get("policy_source"),
        "seed_source": str(seed_source),
        "seed_scope": seed_audit.SEED_SCOPE,
        "fixed_solver": fixed_solver_name,
        "requested_count": requested_count,
        "generated_count": len(samples),
        "valid_count": len(valid_samples),
        "clean_count": sum(1 for sample in samples if base.is_clean(sample)),
        "challenger_api_called_count": sum(1 for sample in samples if sample.get("challenger_api_used") is True),
        "challenger_generation_sources": dict(Counter(str(sample.get("generation_source") or "unknown") for sample in samples)),
        "partial_samples": str(partial_samples),
        "failed_api_calls": str(failed_api_calls),
        **api_stats,
    }
    if adapter_summary_path is not None:
        summary["adapter_summary_path"] = str(adapter_summary_path)
    if isinstance(adapter_summary, dict):
        summary["adapter_rows_with_missing_node_turn_refs_after_filter"] = int(
            adapter_summary.get("rows_with_missing_node_turn_refs_after_filter") or 0
        )
        summary["adapter_missing_node_turn_ref_total_after_filter"] = int(
            adapter_summary.get("missing_node_turn_ref_total_after_filter") or 0
        )
    if candidate_bank_summary_path is not None:
        summary["generation_candidate_bank_summary_path"] = str(candidate_bank_summary_path)
    if isinstance(candidate_bank_summary, dict):
        summary["candidate_bank_ordering_ready_count"] = int(candidate_bank_summary.get("ordering_ready_count") or 0)
        summary["candidate_bank_multi_select_ready_count"] = int(candidate_bank_summary.get("multi_select_ready_count") or 0)
    if programmatic_builder_summary_path is not None:
        summary["generation_programmatic_builder_summary_path"] = str(programmatic_builder_summary_path)
    if isinstance(programmatic_builder_summary, dict):
        summary["programmatic_builder_ordering_rebuilt"] = int(
            programmatic_builder_summary.get("ordering_rebuilt") or 0
        )
        summary["programmatic_builder_ordering_not_rebuilt"] = int(
            programmatic_builder_summary.get("ordering_not_rebuilt") or 0
        )
        summary["programmatic_builder_multi_select_rebuilt"] = int(
            programmatic_builder_summary.get("multi_select_rebuilt") or 0
        )
        summary["programmatic_builder_multi_select_not_rebuilt"] = int(
            programmatic_builder_summary.get("multi_select_not_rebuilt") or 0
        )
    if repair_summary_path is not None:
        summary["generation_consistency_repair_summary_path"] = str(repair_summary_path)
    if isinstance(repair_summary, dict):
        summary["generation_consistency_samples_fixed_before_cleanup"] = int(
            repair_summary.get("samples_fixed_before_cleanup") or 0
        )
        summary["generation_consistency_samples_left_for_cleanup_invalid"] = int(
            repair_summary.get("samples_left_for_cleanup_invalid") or 0
        )
        summary["generation_consistency_ordering_gold_mismatch_count"] = int(
            repair_summary.get("ordering_gold_mismatch_count") or 0
        )
        summary["generation_consistency_multi_select_option_no_visible_support_count"] = int(
            repair_summary.get("multi_select_option_no_visible_support_count") or 0
        )
    if visible_support_guard_summary_path is not None:
        summary["generation_visible_support_guard_summary_path"] = str(visible_support_guard_summary_path)
    if isinstance(visible_support_guard_summary, dict):
        summary["visible_support_guard_ordering_gold_mismatch_visible_order"] = int(
            visible_support_guard_summary.get("ordering_gold_mismatch_visible_order") or 0
        )
        summary["visible_support_guard_ordering_option_no_support"] = int(
            visible_support_guard_summary.get("ordering_option_no_support") or 0
        )
        summary["visible_support_guard_multi_select_option_no_support"] = int(
            visible_support_guard_summary.get("multi_select_option_no_support") or 0
        )
    if cleanup_summary_path is not None:
        summary["post_generation_turn_cleanup_summary_path"] = str(cleanup_summary_path)
    if isinstance(cleanup_summary, dict):
        summary["cleanup_invalid_count"] = int(cleanup_summary.get("cleanup_invalid_count") or 0)
        summary["valid_after_cleanup_count"] = int(cleanup_summary.get("valid_after_cleanup_count") or 0)
        summary["cleanup_missing_ref_total"] = int(cleanup_summary.get("missing_ref_total") or 0)
        summary["cleanup_unsupported_ref_total"] = int(cleanup_summary.get("unsupported_ref_total") or 0)
    if post_generation_turn_coverage_audit_path is not None:
        summary["post_generation_turn_coverage_audit_path"] = str(post_generation_turn_coverage_audit_path)
    if isinstance(post_generation_turn_coverage_audit, dict):
        summary["post_generation_rows_with_missing_refs"] = int(
            post_generation_turn_coverage_audit.get("rows_with_missing_refs") or 0
        )
        summary["post_generation_missing_ref_total"] = int(
            post_generation_turn_coverage_audit.get("missing_ref_total") or 0
        )
        summary["valid_after_cleanup_rows_with_missing_refs"] = int(
            post_generation_turn_coverage_audit.get("valid_after_cleanup_rows_with_missing_refs") or 0
        )
    write_json(output_dir / "generation_summary.json", summary)
    return summary


def generate_samples(
    *,
    seed_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    compiled_policy: dict[str, Any],
    output_dir: str | Path,
    partial_samples: str | Path,
    failed_api_calls: str | Path,
    seed_source: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_dir = Path(output_dir)
    partial_samples = Path(partial_samples)
    failed_api_calls = Path(failed_api_calls)
    output_dir.mkdir(parents=True, exist_ok=True)
    _touch_jsonl(failed_api_calls)
    _patch_old_gate_for_current_method(
        run_name=str(args.run_name),
        seed_scope=seed_audit.SEED_SCOPE,
        fixed_solver_name=str(args.fixed_solver_name),
        compiled_policy=compiled_policy,
    )
    gen_args = _generation_args(
        args,
        output_dir=output_dir,
        partial_samples=partial_samples,
        failed_api_calls=failed_api_calls,
        compiled_policy=compiled_policy,
    )
    input_seed_rows = seed_rows
    seed_rows = _augment_seed_rows_for_old_gate(seed_rows)
    source_turn_lookup_by_source_id = _build_source_turn_lookup_by_source_id(seed_rows)
    adapter_summary = _build_adapter_summary(input_seed_rows, seed_rows)
    adapter_summary_path = output_dir / "current_method_to_old_gate_adapter_summary.json"
    write_json(adapter_summary_path, adapter_summary)
    if int(adapter_summary.get("missing_node_turn_ref_total_after_filter") or 0) != 0:
        raise RuntimeError(
            "current_method -> old_gate adapter produced state_chain nodes referencing absent final_dialogue turns"
        )
    old_gate.ACTIVE_ARGS = gen_args
    try:
        samples = old_gate.generate_samples(seed_rows, gen_args)
    except old_gate.ApiFailureLimitExceeded:
        raise
    if len(samples) < int(args.n_train):
        raise RuntimeError(f"Generated {len(samples)}/{args.n_train}; partial checkpoint remains at {partial_samples}")
    for sample in samples:
        sample["generation_candidate_bank"] = _sample_candidate_bank(sample)
    candidate_bank_summary = _build_generation_candidate_bank_summary(samples)
    candidate_bank_summary_path = output_dir / "generation_candidate_bank_summary.json"
    write_json(candidate_bank_summary_path, candidate_bank_summary)
    repaired_samples: list[dict[str, Any]] = []
    repair_reports: list[dict[str, Any]] = []
    for sample in samples:
        source_turn_lookup = source_turn_lookup_by_source_id.get(str(sample.get("source_id") or ""))
        repaired_sample, repair_report = repair_generated_sample_evidence_window(
            sample,
            source_turn_lookup=source_turn_lookup,
        )
        repaired_samples.append(repaired_sample)
        repair_reports.append(repair_report)
    samples = repaired_samples
    builder_samples: list[dict[str, Any]] = []
    builder_reports: list[dict[str, Any]] = []
    for sample in samples:
        question_type = str(((sample.get("qa") or {}) if isinstance(sample.get("qa"), dict) else {}).get("question_type") or "unknown")
        if question_type == "ordering":
            builder_sample, builder_report = try_rebuild_ordering_from_visible_candidates(sample)
        elif question_type == "multi_select":
            builder_sample, builder_report = try_rebuild_multi_select_from_visible_candidates(sample)
        else:
            builder_sample = copy.deepcopy(sample)
            builder_report = {
                "sample_id": builder_sample.get("sample_id"),
                "question_type": question_type,
                "rebuilt": False,
                "reason": None,
                "selected_candidate_ids": [],
                "selected_positive_ids": [],
                "selected_negative_ids": [],
                "visible_order": [],
            }
        builder_sample = _refresh_sample_validation(builder_sample)
        builder_samples.append(builder_sample)
        builder_reports.append(builder_report)
    samples = builder_samples
    programmatic_builder_summary = _build_generation_programmatic_builder_summary(builder_reports)
    programmatic_builder_summary_path = output_dir / "generation_programmatic_builder_summary.json"
    write_json(programmatic_builder_summary_path, programmatic_builder_summary)
    guarded_samples: list[dict[str, Any]] = []
    visible_support_guard_reports: list[dict[str, Any]] = []
    for sample in samples:
        question_type = str(((sample.get("qa") or {}) if isinstance(sample.get("qa"), dict) else {}).get("question_type") or "unknown")
        if question_type == "ordering":
            guarded_sample, guard_report = enforce_ordering_visible_support(sample)
        elif question_type == "multi_select":
            guarded_sample, guard_report = enforce_multi_select_visible_support(sample)
        else:
            guarded_sample = copy.deepcopy(sample)
            guard_report = {
                "sample_id": guarded_sample.get("sample_id"),
                "question_type": question_type,
                "applied": False,
                "all_options_supported": False,
                "gold_match_visible_order": False,
                "gold_mismatch_visible_order": False,
                "option_no_support": False,
                "gold_option_no_support": False,
                "warnings": [],
                "invalidated_by_guard": False,
            }
        guarded_samples.append(guarded_sample)
        visible_support_guard_reports.append(guard_report)
    samples = guarded_samples
    visible_support_guard_summary = _build_generation_visible_support_guard_summary(visible_support_guard_reports)
    visible_support_guard_summary_path = output_dir / "generation_visible_support_guard_summary.json"
    write_json(visible_support_guard_summary_path, visible_support_guard_summary)
    cleaned_samples: list[dict[str, Any]] = []
    cleanup_reports: list[dict[str, Any]] = []
    for sample in samples:
        sample = reset_turn_coverage_cleanup_artifacts(sample)
        cleaned_sample, cleanup_report = cleanup_generated_sample_turn_refs(sample)
        cleaned_samples.append(cleaned_sample)
        cleanup_reports.append(cleanup_report)
    samples = cleaned_samples
    repair_summary = _build_generation_consistency_repair_summary(repair_reports, cleanup_reports)
    repair_summary_path = output_dir / "generation_consistency_repair_summary.json"
    write_json(repair_summary_path, repair_summary)
    cleanup_summary = _build_turn_cleanup_summary(cleanup_reports)
    cleanup_summary_path = output_dir / "post_generation_turn_cleanup_summary.json"
    write_json(cleanup_summary_path, cleanup_summary)
    post_generation_turn_coverage_audit = _build_post_generation_turn_coverage_audit(samples)
    post_generation_turn_coverage_audit_path = output_dir / "post_generation_turn_coverage_audit.json"
    write_json(post_generation_turn_coverage_audit_path, post_generation_turn_coverage_audit)
    summary = write_generation_outputs(
        output_dir=output_dir,
        samples=samples,
        compiled_policy=compiled_policy,
        fixed_solver_name=str(args.fixed_solver_name),
        seed_source=seed_source,
        partial_samples=partial_samples,
        failed_api_calls=failed_api_calls,
        requested_count=int(args.n_train),
        adapter_summary_path=adapter_summary_path,
        adapter_summary=adapter_summary,
        candidate_bank_summary_path=candidate_bank_summary_path,
        candidate_bank_summary=candidate_bank_summary,
        programmatic_builder_summary_path=programmatic_builder_summary_path,
        programmatic_builder_summary=programmatic_builder_summary,
        repair_summary_path=repair_summary_path,
        repair_summary=repair_summary,
        visible_support_guard_summary_path=visible_support_guard_summary_path,
        visible_support_guard_summary=visible_support_guard_summary,
        cleanup_summary_path=cleanup_summary_path,
        cleanup_summary=cleanup_summary,
        post_generation_turn_coverage_audit_path=post_generation_turn_coverage_audit_path,
        post_generation_turn_coverage_audit=post_generation_turn_coverage_audit,
    )
    return samples, summary


def load_samples(path: str | Path) -> list[dict[str, Any]]:
    return seed_audit.read_jsonl(path)
