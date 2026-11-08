from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


HARDEN_TEST_DIR = Path("harden-test")
if HARDEN_TEST_DIR.exists():
    sys.path.insert(0, str(HARDEN_TEST_DIR.resolve()))

import dialogue_hardening  # noqa: E402
import test as generation_test  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Graph-aware dialogue hardening by adding structured future sessions.")
    parser.add_argument("--source", required=True, help="Source batch JSON with final_dialogue.")
    parser.add_argument("--graph", required=True, help="Memory graph JSON from build_dialogue_memory_graph.py.")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--models", default="qwen3-4b,qwen3-8b")
    parser.add_argument("--challenger-model", default=None)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--request-timeout", type=int, default=None)
    parser.add_argument("--light-request-timeout", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--max-dialogues", type=int, default=20)
    parser.add_argument("--turns-per-dialogue", type=int, default=4)
    parser.add_argument(
        "--adversarial-memory-pressure",
        action="store_true",
        help="Ask the challenger to add override/conflict/distractor chains instead of only benign graph facts.",
    )
    parser.add_argument(
        "--policy-file",
        default=None,
        help="Optional challenger-generated policy JSON; validated dialogue hardening directives are injected into the prompt.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def save_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_policy_directives(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    payload = load_json(path)
    policy = payload.get("validated_policy") if isinstance(payload.get("validated_policy"), dict) else payload
    directives = policy.get("dialogue_hardening_directives") if isinstance(policy, dict) else None
    if not isinstance(directives, dict):
        return {}
    allowed_keys = {"global", "by_candidate_type"}
    return {key: copy.deepcopy(value) for key, value in directives.items() if key in allowed_keys}


def collect_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results = payload.get("results")
    if isinstance(results, list):
        return [item for item in results if isinstance(item, dict)]
    if isinstance(payload.get("final_dialogue"), list):
        return [payload]
    return []


def result_key_from_values(sample_id: Any, qa_index: Any) -> tuple[str, int]:
    return str(sample_id), int(qa_index)


def result_key(result: dict[str, Any]) -> tuple[str, int]:
    return result_key_from_values(result.get("sample_id", result.get("sample_index", "")), result.get("qa_index", -1))


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def parse_session_date(value: Any) -> datetime | None:
    text = str(value or "")
    match = re.search(r"\b(\d{1,2})\s+([A-Za-z]+),\s*(20\d{2})\b", text)
    if not match:
        return None
    months = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }
    month = months.get(match.group(2).lower())
    if not month:
        return None
    return datetime(int(match.group(3)), month, int(match.group(1)), 10, 0)


def format_session_time(value: datetime) -> str:
    return f"10:00 am on {value.day} {value.strftime('%B')}, {value.year}"


def latest_session_info(dialogue: list[dict[str, Any]]) -> tuple[int, str, str]:
    latest_order = 0
    latest_key = "session_1"
    latest_time = ""
    latest_date: datetime | None = None
    for turn in dialogue:
        if not isinstance(turn, dict):
            continue
        order = int(turn.get("timeline_order", 0) or 0)
        if order > latest_order:
            latest_order = order
            latest_key = str(turn.get("session_key") or f"session_{order}")
        parsed = parse_session_date(turn.get("session_time"))
        if parsed and (latest_date is None or parsed > latest_date):
            latest_date = parsed
            latest_time = str(turn.get("session_time") or "")
    if latest_date is None:
        latest_date = datetime(2023, 1, 1, 10, 0)
    return latest_order or 1, latest_key, format_session_time(latest_date + timedelta(days=14))


def next_dia_prefix(dialogue: list[dict[str, Any]]) -> int:
    max_prefix = 0
    for turn in dialogue:
        if not isinstance(turn, dict):
            continue
        for key in ("dia_id", "turn_ref", "turn_id"):
            value = str(turn.get(key) or "")
            match = re.match(r"^D(\d+):\d+$", value)
            if match:
                max_prefix = max(max_prefix, int(match.group(1)))
    return max_prefix + 1


def next_session_key(dialogue: list[dict[str, Any]], fallback_order: int) -> str:
    max_session = 0
    for turn in dialogue:
        if not isinstance(turn, dict):
            continue
        match = re.match(r"^session_(\d+)$", str(turn.get("session_key") or ""))
        if match:
            max_session = max(max_session, int(match.group(1)))
    return f"session_{max(max_session + 1, fallback_order)}"


def graph_sort_key(graph: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        -int(graph.get("timeline_event_count", 0)),
        -int(graph.get("state_update_count", 0)),
        -int(graph.get("person_relation_count", 0)),
        int(graph.get("qa_index", 999999)),
    )


def compact_graph(graph: dict[str, Any]) -> dict[str, Any]:
    events = graph.get("timeline_events", [])
    relations = graph.get("person_relations", [])
    updates = graph.get("state_updates", [])
    return {
        "qa_index": graph.get("qa_index"),
        "source_question": graph.get("source_question"),
        "timeline_events": [
            {
                "event_id": event.get("event_id"),
                "turn_id": event.get("turn_id"),
                "speaker": event.get("speaker"),
                "quote": event.get("quote"),
                "session_key": event.get("session_key"),
                "session_time": event.get("session_time"),
                "time_expression": event.get("time_expression"),
            }
            for event in events[:12]
            if isinstance(event, dict)
        ],
        "person_nodes": graph.get("person_nodes", []),
        "person_relations": [
            {
                "source": relation.get("source"),
                "target": relation.get("target"),
                "relation_type": relation.get("relation_type"),
                "evidence_quotes": relation.get("evidence_quotes", [])[:2],
            }
            for relation in relations[:6]
            if isinstance(relation, dict)
        ],
        "state_updates": updates[:6] if isinstance(updates, list) else [],
    }


def build_messages(
    result: dict[str, Any],
    graph: dict[str, Any],
    *,
    turns_per_dialogue: int,
    adversarial_memory_pressure: bool = False,
    policy_directives: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    dialogue = result.get("final_dialogue", [])
    latest_order, _, next_time = latest_session_info(dialogue)
    speakers = [
        node.get("name")
        for node in graph.get("person_nodes", [])
        if isinstance(node, dict) and node.get("is_speaker") and node.get("name")
    ]
    if not speakers:
        speakers = sorted({str(turn.get("speaker")) for turn in dialogue if isinstance(turn, dict) and turn.get("speaker")})
    qa = result.get("qa", {}) if isinstance(result.get("qa"), dict) else {}
    payload = {
        "task": "Add a graph-aware future dialogue session that expands the memory graph.",
        "original_qa_to_preserve": {
            "question": qa.get("question"),
            "answer": qa.get("answer", qa.get("correct_answer_text")),
            "evidence": qa.get("evidence"),
        },
        "allowed_speakers": speakers,
        "next_session": {
            "timeline_order": latest_order + 1,
            "session_key": f"session_{latest_order + 1}",
            "session_time": next_time,
        },
        "memory_graph": compact_graph(graph),
        "dialogue_tail": [
            {
                "turn_id": turn.get("dia_id", turn.get("turn_ref", turn.get("turn_index"))),
                "speaker": turn.get("speaker"),
                "text": turn.get("text"),
                "session_key": turn.get("session_key"),
                "session_time": turn.get("session_time"),
            }
            for turn in dialogue[-10:]
            if isinstance(turn, dict)
        ],
        "requirements": [
            f"Return exactly {turns_per_dialogue} new turns.",
            "The new turns must form one coherent future session after all existing sessions.",
            "Add at least two concrete new events with time cues such as yesterday, last week, next month, recently, or this weekend.",
            "Add at least one state update that changes, advances, or finalizes an existing plan, preference, activity, project, relationship, or location.",
            "Add at least one relation-bearing turn involving two people, such as support, advice, collaboration, shared experience, or asking about the other person.",
            "You may mention one new third person only if they are introduced through one existing speaker's report; do not make the new person a speaker.",
            "Do not contradict existing timeline events or person states.",
            "Do not change the answer to the original QA. The original answer must remain unique.",
            "Avoid generic encouragement or filler. Each turn must add graph facts usable for future QA.",
            "Return strict JSON only.",
        ],
        "schema": {
            "new_turns": [
                {
                    "speaker": "one of allowed_speakers",
                    "text": "specific future-session turn text",
                    "graph_ops": ["new_event", "state_update", "person_relation"],
                    "new_event_summary": "concise event fact",
                    "state_update_summary": "optional state update",
                    "relation_summary": "optional relation fact",
                    "answer_safety_note": "why original QA answer remains unchanged",
                }
            ]
        },
    }
    if policy_directives:
        payload["challenger_policy_directives"] = {
            "source": "solver_feedback_generated_policy",
            "dialogue_hardening_directives": policy_directives,
            "instruction": (
                "Follow these challenger-generated directives when they are compatible with the preservation and validity requirements. "
                "If a directive conflicts with answer uniqueness or evidence grounding, preserve validity first."
            ),
        }
        payload["requirements"].extend([
            "Use the challenger_policy_directives to choose which graph facts, conflicts, relation changes, or state updates to add.",
            "When following a policy directive, make the resulting event chain explicit enough for later QA generation and validation.",
        ])

    if adversarial_memory_pressure:
        payload["task"] = "Add an adversarial graph-aware future dialogue session that creates harder long-memory reasoning chains."
        payload["adversarial_memory_pressure"] = {
            "goal": "Make future QA require tracking final state under long-context distractors, while preserving the original QA answer.",
            "preferred_chain_patterns": [
                "old_state_or_plan -> complication/correction/reschedule -> final/current state",
                "old answer-like distractor -> explicit override/cancellation -> unique final answer",
                "person A report -> person B response/advice -> later consequence for A or B",
                "setup in an early session -> near-miss distractor on same topic -> later true outcome",
                "earlier preference/intent -> cancellation/change -> final decision",
            ],
            "important": [
                "The distractor facts must be plausible and same-topic, but globally wrong for the final state.",
                "Use explicit words such as changed, cancelled, postponed, instead, latest, final, currently, or by the end when natural.",
                "Make the final/current state explicit in a later turn after the correction, not only implied.",
                "Do not leave two equally valid final answers for the same person/topic chain.",
                "Do not merely add encouragement; add facts that can become MCQ options.",
            ],
        }
        payload["requirements"].extend([
            "Create at least one three-step memory chain: an old plan/preference/state, a later complication or correction, and a final/current state.",
            "Include at least one plausible same-topic distractor fact from a different speaker, different session, or earlier state; it must not become the final answer.",
            "At least one turn must explicitly override, cancel, reschedule, narrow, or revise an earlier memory.",
            "At least one later turn must state the final/current outcome after the override, not just imply it.",
            "For the strongest chain, make the state_update_summary follow this shape: old_state -> correction_or_complication -> final_current_state.",
            "Keep the corrected/final chain on one person-topic pair so later QA has a unique answer.",
            "Prefer reusing existing people, places, activities, projects, or relations from the memory graph so the new facts connect to prior sessions.",
        ])

    return [
        {
            "role": "system",
            "content": (
                "You are a graph-aware dialogue hardening module. "
                "You extend dialogue memory with structured future events while preserving the original QA answer. "
                "Return strict JSON only."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def normalize_new_turns(payload: Any, allowed_speakers: set[str], turns_per_dialogue: int) -> list[dict[str, Any]]:
    raw_turns = payload.get("new_turns") if isinstance(payload, dict) else payload
    if not isinstance(raw_turns, list):
        raise ValueError("graph hardening response must contain new_turns list")
    turns: list[dict[str, Any]] = []
    for raw_turn in raw_turns[:turns_per_dialogue]:
        if not isinstance(raw_turn, dict):
            continue
        speaker = normalize_space(raw_turn.get("speaker"))
        text = normalize_space(raw_turn.get("text"))
        if allowed_speakers and speaker not in allowed_speakers:
            continue
        if len(text.split()) < 8:
            continue
        graph_ops = raw_turn.get("graph_ops")
        if not isinstance(graph_ops, list):
            graph_ops = []
        turns.append(
            {
                "speaker": speaker,
                "text": text,
                "graph_ops": [str(item) for item in graph_ops],
                "new_event_summary": normalize_space(raw_turn.get("new_event_summary")),
                "state_update_summary": normalize_space(raw_turn.get("state_update_summary")),
                "relation_summary": normalize_space(raw_turn.get("relation_summary")),
                "answer_safety_note": normalize_space(raw_turn.get("answer_safety_note")),
            }
        )
    if len(turns) < max(2, min(3, turns_per_dialogue)):
        raise ValueError(f"not_enough_valid_new_turns:{len(turns)}")
    return turns


def generate_turns(
    *,
    result: dict[str, Any],
    graph: dict[str, Any],
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    request_timeout: int,
    turns_per_dialogue: int,
    adversarial_memory_pressure: bool = False,
    policy_directives: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str, Any]:
    dialogue = result.get("final_dialogue", [])
    allowed_speakers = {
        str(node.get("name"))
        for node in graph.get("person_nodes", [])
        if isinstance(node, dict) and node.get("is_speaker") and node.get("name")
    }
    if not allowed_speakers:
        allowed_speakers = {str(turn.get("speaker")) for turn in dialogue if isinstance(turn, dict) and turn.get("speaker")}
    raw_text = generation_test.call_chat_completion(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=build_messages(
            result,
            graph,
            turns_per_dialogue=turns_per_dialogue,
            adversarial_memory_pressure=adversarial_memory_pressure,
            policy_directives=policy_directives,
        ),
        temperature=temperature,
        request_timeout=request_timeout,
    )
    payload = generation_test.extract_json_payload(raw_text)
    return normalize_new_turns(payload, allowed_speakers, turns_per_dialogue), raw_text, payload


def apply_turns(result: dict[str, Any], turns: list[dict[str, Any]]) -> dict[str, Any]:
    updated = copy.deepcopy(result)
    dialogue = [copy.deepcopy(turn) for turn in updated.get("final_dialogue", []) if isinstance(turn, dict)]
    latest_order, _, session_time = latest_session_info(dialogue)
    new_order = latest_order + 1
    session_key = next_session_key(dialogue, new_order)
    dia_prefix = next_dia_prefix(dialogue)
    for offset, raw_turn in enumerate(turns, start=1):
        dialogue.append(
            {
                "speaker": raw_turn["speaker"],
                "dia_id": f"D{dia_prefix}:{offset}",
                "text": raw_turn["text"],
                "turn_index": len(dialogue) + 1,
                "timeline_order": new_order,
                "session_key": session_key,
                "session_time": session_time if offset == 1 else None,
                "source": "graph_aware_hardening",
                "graph_ops": raw_turn.get("graph_ops", []),
                "new_event_summary": raw_turn.get("new_event_summary"),
                "state_update_summary": raw_turn.get("state_update_summary"),
                "relation_summary": raw_turn.get("relation_summary"),
                "answer_safety_note": raw_turn.get("answer_safety_note"),
            }
        )
    updated["final_dialogue"] = dialogue
    updated["final_dialogue_word_count"] = generation_test.count_dialogue_text_units(dialogue)
    original = updated.get("original_dialogue_word_count")
    if isinstance(original, int):
        updated["word_count_delta"] = updated["final_dialogue_word_count"] - original
        updated["word_count_comparison"] = {
            "before": original,
            "after": updated["final_dialogue_word_count"],
            "delta": updated["word_count_delta"],
        }
    updated["graph_aware_hardening"] = {
        "enabled": True,
        "added_turn_count": len(turns),
        "added_session_key": session_key,
        "added_timeline_order": new_order,
        "added_dia_prefix": f"D{dia_prefix}",
        "added_session_time": session_time,
        "new_turns": turns,
    }
    updated.pop("model_eval", None)
    updated.pop("frontier_label", None)
    return updated


def build_summary(payload: dict[str, Any]) -> dict[str, Any]:
    results = collect_results(payload)
    attempted = [
        result for result in results
        if isinstance(result.get("graph_aware_hardening"), dict)
    ]
    failed = [
        item for item in payload.get("graph_aware_hardening_failures", [])
        if isinstance(item, dict)
    ]
    added_turns = [
        len(result.get("graph_aware_hardening", {}).get("new_turns", []))
        for result in attempted
    ]
    errors: dict[str, int] = {}
    for item in failed:
        reason = str(item.get("error") or "unknown").split(":")[0]
        errors[reason] = errors.get(reason, 0) + 1
    return {
        "total_input_count": len(results),
        "successful_graph_hardening_count": len(attempted),
        "failed_graph_hardening_count": len(failed),
        "total_added_turn_count": sum(added_turns),
        "average_added_turn_count": round(sum(added_turns) / len(added_turns), 4) if added_turns else None,
        "failure_reason_counts": errors,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }




def reset_previous_graph_hardening(payload: dict[str, Any]) -> None:
    """Prepare a fresh hardening round from an already graph-hardened source.

    Earlier rounds may already contain graph_aware_hardening metadata. For a new
    output prefix, that metadata is history, not evidence that this round is
    complete. Resume should only skip records from the current output file.
    """
    for result in collect_results(payload):
        previous = result.get("graph_aware_hardening")
        if not isinstance(previous, dict):
            continue
        history = result.get("graph_aware_hardening_history")
        if not isinstance(history, list):
            history = []
        history.append(copy.deepcopy(previous))
        result["graph_aware_hardening_history"] = history
        result.pop("graph_aware_hardening", None)


def checkpoint(output_prefix: Path, payload: dict[str, Any]) -> None:
    payload["graph_aware_hardening_summary"] = build_summary(payload)
    save_json(output_prefix.with_suffix(".json"), payload)
    save_json(output_prefix.with_name(output_prefix.name + "_report.json"), payload["graph_aware_hardening_summary"])


def main() -> int:
    args = parse_args()
    output_prefix = Path(args.output_prefix)
    result_path = output_prefix.with_suffix(".json")
    if args.resume and result_path.exists():
        output_payload = load_json(result_path)
    else:
        source_payload = load_json(args.source)
        output_payload = copy.deepcopy(source_payload)
        reset_previous_graph_hardening(output_payload)
        output_payload["graph_aware_hardening_source"] = {
            "source_path": args.source,
            "graph_path": args.graph,
        }
        output_payload["graph_aware_hardening_failures"] = []

    graph_payload = load_json(args.graph)
    graphs = [graph for graph in graph_payload.get("graphs", []) if isinstance(graph, dict)]
    graphs.sort(key=graph_sort_key)
    if args.max_dialogues is not None:
        graphs = graphs[: args.max_dialogues]
    results = collect_results(output_payload)
    result_by_key = {result_key(result): result for result in results}
    completed = {
        int(result.get("qa_index", -1))
        for result in results
        if isinstance(result.get("graph_aware_hardening"), dict)
    }
    failed = {
        int(item.get("qa_index", -1))
        for item in output_payload.get("graph_aware_hardening_failures", [])
        if isinstance(item, dict)
    }
    api_config = dialogue_hardening.build_config(args)
    policy_directives = load_policy_directives(args.policy_file)
    if args.policy_file:
        output_payload.setdefault("graph_aware_hardening_source", {})["policy_file"] = args.policy_file
        output_payload["graph_aware_hardening_policy_directives"] = policy_directives

    for graph in graphs:
        qa_index = int(graph.get("qa_index", -1))
        if qa_index in completed or qa_index in failed:
            continue
        result = result_by_key.get(result_key_from_values(graph.get("sample_id", graph.get("sample_index", "")), qa_index))
        if result is None:
            continue
        print(f"Graph-aware hardening qa_index={qa_index}")
        try:
            turns, raw_text, raw_payload = generate_turns(
                result=result,
                graph=graph,
                base_url=api_config["base_url"],
                api_key=api_config["api_key"],
                model=api_config["challenger_model"],
                temperature=args.temperature,
                request_timeout=api_config["request_timeout"],
                turns_per_dialogue=args.turns_per_dialogue,
                adversarial_memory_pressure=args.adversarial_memory_pressure,
                policy_directives=policy_directives,
            )
            updated = apply_turns(result, turns)
            if policy_directives:
                updated["graph_aware_hardening"]["policy_directives_used"] = policy_directives
            updated["graph_aware_hardening"]["raw_generation_response"] = raw_text
            updated["graph_aware_hardening"]["raw_generation_payload"] = raw_payload
            result.clear()
            result.update(updated)
        except Exception as error:  # noqa: BLE001
            output_payload.setdefault("graph_aware_hardening_failures", []).append(
                {
                    "qa_index": qa_index,
                    "sample_id": graph.get("sample_id"),
                    "error": str(error),
                }
            )
        checkpoint(output_prefix, output_payload)

    checkpoint(output_prefix, output_payload)
    print(json.dumps(output_payload["graph_aware_hardening_summary"], ensure_ascii=False, indent=2))
    print(f"Saved graph-aware hardening result to: {output_prefix.with_suffix('.json').resolve()}")
    print(f"Saved graph-aware hardening report to: {output_prefix.with_name(output_prefix.name + '_report.json').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
