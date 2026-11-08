from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any


HARDEN_TEST_DIR = Path("harden-test")
if HARDEN_TEST_DIR.exists():
    sys.path.insert(0, str(HARDEN_TEST_DIR.resolve()))

import answer_semantics  # noqa: E402
import dialogue_hardening  # noqa: E402
import evolved_qa_validation  # noqa: E402
import qa_evolution  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multiple QA items from dialogue memory graphs.")
    parser.add_argument("--source", required=True, help="Source batch JSON with final_dialogue.")
    parser.add_argument("--graph", required=True, help="Memory graph JSON from build_dialogue_memory_graph.py.")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--models", default="qwen3-4b,qwen3-8b")
    parser.add_argument("--challenger-model", default=None)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--solver-base-url", default=None, help="OpenAI-compatible endpoint for solver light eval only.")
    parser.add_argument("--solver-api-key", default=None, help="API key for solver light eval endpoint.")
    parser.add_argument("--request-timeout", type=int, default=None)
    parser.add_argument("--light-request-timeout", type=int, default=None)
    parser.add_argument("--critic-temperature", type=float, default=0.1)
    parser.add_argument("--option-temperature", type=float, default=0.25)
    parser.add_argument("--light-temperature", type=float, default=0.3)
    parser.add_argument("--max-dialogues", type=int, default=20)
    parser.add_argument("--qa-per-dialogue", type=int, default=3)
    parser.add_argument("--candidate-pool-per-dialogue", type=int, default=8)
    parser.add_argument("--light-candidates-per-model", type=int, default=5)
    parser.add_argument("--evidence-window-size", type=int, default=2, help="Turns before/after evidence turns used for oracle evidence-window diagnostic eval.")
    parser.add_argument("--solver-eval-view", choices=("full", "window", "both"), default="both", help="Use full dialogue, oracle evidence window, or both for solver light eval.")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--min-light-accuracy", type=float, default=None)
    parser.add_argument("--max-light-accuracy", type=float, default=0.7)
    parser.add_argument("--allowed-evolution-types", default=None)
    parser.add_argument(
        "--candidate-type-weights",
        default=None,
        help=(
            "Comma-separated candidate_type=weight policy, e.g. "
            "cross_session_chain=0.35,cross_session_bridge=0.25,conflict_chain=0.15,"
            "relation_chain=0.15,state_update=0.10,temporal_pair=0"
        ),
    )
    parser.add_argument("--include-easy", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-api-failures", action="store_true")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def save_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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


def short(text: Any, limit: int = 96) -> str:
    value = evolved_qa_validation.normalize_space(text)
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def answer_label(text: Any, limit: int = 72) -> str:
    value = evolved_qa_validation.normalize_space(text)
    if not value:
        return ""
    value = re.sub(r"^(i|we|he|she|they)\s+(just\s+|also\s+|finally\s+|currently\s+)?", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\b(btw|by the way)\b\.?", "", value, flags=re.IGNORECASE)
    value = value.strip(" .,!?:;\"'")
    return short(value, limit)


def option_label(text: Any, limit: int = 86) -> str:
    value = answer_label(text, limit)
    value = re.sub(r"^[A-Z][A-Za-z0-9_ -]{0,30}'s latest update:\s*", "", value)
    return value.strip(" .,!?:;\"'")


def is_actionable_state_update_quote(value: Any) -> bool:
    text = evolved_qa_validation.normalize_space(value)
    if len(text) < 18:
        return False
    lowered = text.lower()
    if text.endswith("?") or re.search(r"\b(did you|have you|do you|are you|can you|could you|would you|what about|how about|i wonder)\b", lowered):
        return False
    if re.search(r"\b(would be|could be|should be|might be|great idea|great way|good way|sounds like|that's awesome|that is awesome|that's great|that is great|you should|maybe you|i think you)\b", lowered):
        return False
    return bool(re.search(
        r"\b(i('| a)m|we('| a)re|i have|i've|we have|we've|planning|planned|decided|changed|cancelled|canceled|postponed|rescheduled|instead|latest|final|currently|started|finished|found|went|going|working|moved|adopted|took|taking|will|this weekend|next month|last week|recently)\b",
        lowered,
    ))


def relation_label(value: Any) -> str:
    return str(value or "relationship").replace("_", " ").strip()


def normalize_option_value(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip(" \t\r\n")


def graph_sort_key(graph: dict[str, Any]) -> tuple[int, int, int]:
    return (
        -int(graph.get("timeline_event_count", 0)),
        -int(graph.get("person_relation_count", 0)),
        int(graph.get("qa_index", 999999)),
    )


def dialogue_quote_map(dialogue: list[dict[str, Any]]) -> dict[str, str]:
    quotes: dict[str, str] = {}
    for turn in dialogue:
        if not isinstance(turn, dict):
            continue
        turn_id = str(turn.get("dia_id") or turn.get("turn_ref") or turn.get("turn_id") or turn.get("turn_index") or "").strip()
        text = evolved_qa_validation.normalize_space(turn.get("text"))
        if turn_id and text:
            quotes[turn_id] = text
    return quotes


def exact_quote(turn_id: Any, fallback: Any, quote_map: dict[str, str] | None) -> str:
    key = str(turn_id or "").strip()
    if quote_map and key in quote_map:
        return quote_map[key]
    return evolved_qa_validation.normalize_space(fallback)


def candidate_base(index: int, question: str, answer: str, evidence: list[str], quotes: list[dict[str, str]], etype: str) -> dict[str, Any]:
    return {
        "candidate_index": index,
        "evolved_question": question,
        "evolved_answer": answer,
        "evolved_evidence": evidence,
        "evidence_quotes": quotes,
        "evolution_type": etype,
        "why_harder": "Generated from a structured dialogue memory graph instead of a single source QA.",
        "source_facts": [
            {"turn_id": item["turn_id"], "fact": item["quote"]}
            for item in quotes
        ],
        "expected_reasoning": "",
        "_graph_generated": True,
    }


def event_sort_key(event: dict[str, Any]) -> tuple[int, int]:
    order = event.get("order")
    if isinstance(order, (list, tuple)) and len(order) >= 2:
        return int(order[0] or 999999), int(order[1] or 999999)
    return 999999, 999999


def sorted_events(graph: dict[str, Any]) -> list[dict[str, Any]]:
    events = [
        event for event in graph.get("timeline_events", [])
        if isinstance(event, dict) and event.get("turn_id") and event.get("quote") and event.get("phrase")
    ]
    return sorted(events, key=event_sort_key)


def event_by_id(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(event.get("event_id")): event for event in sorted_events(graph) if event.get("event_id")}


def append_graph_distractors(candidate: dict[str, Any], values: list[Any]) -> None:
    answer = normalize_option_value(candidate.get("evolved_answer"))
    existing = [str(item) for item in candidate.get("_graph_distractor_pool", []) if str(item or "").strip()]
    seen = {option_label(answer).lower(), answer.lower(), *[item.lower() for item in existing]}
    pool = list(existing)
    for value in values:
        label = option_label(value)
        if not label or label.lower() in seen:
            continue
        if answer_semantics.is_semantically_equivalent_answer(answer, label, str(candidate.get("evolved_question") or "")):
            continue
        seen.add(label.lower())
        pool.append(label)
    if pool:
        candidate["_graph_distractor_pool"] = pool


def append_priority_graph_distractors(candidate: dict[str, Any], values: list[Any]) -> None:
    answer = normalize_option_value(candidate.get("evolved_answer"))
    existing = [str(item) for item in candidate.get("_graph_priority_distractor_pool", []) if str(item or "").strip()]
    normal_existing = [str(item) for item in candidate.get("_graph_distractor_pool", []) if str(item or "").strip()]
    seen = {option_label(answer).lower(), answer.lower(), *[item.lower() for item in existing]}
    priority_pool = list(existing)
    for value in values:
        label = option_label(value)
        if not label or label.lower() in seen:
            continue
        if answer_semantics.is_semantically_equivalent_answer(answer, label, str(candidate.get("evolved_question") or "")):
            continue
        seen.add(label.lower())
        priority_pool.append(label)
    if priority_pool:
        candidate["_graph_priority_distractor_pool"] = priority_pool
    append_graph_distractors(candidate, priority_pool + normal_existing)


def extract_update_chains(graph: dict[str, Any]) -> list[dict[str, Any]]:
    events_by_id = event_by_id(graph)
    chains = []
    for update in graph.get("state_updates", []):
        if not isinstance(update, dict):
            continue
        previous_turn = str(update.get("previous_turn_id") or "")
        current_turn = str(update.get("current_turn_id") or "")
        if not previous_turn or not current_turn:
            continue
        previous_event = events_by_id.get(str(update.get("previous_event_id"))) or {"turn_id": previous_turn, "quote": update.get("previous_quote")}
        current_event = events_by_id.get(str(update.get("current_event_id"))) or {"turn_id": current_turn, "quote": update.get("current_quote")}
        if not is_actionable_state_update_quote(current_event.get("quote") or update.get("current_quote")):
            continue
        person = update.get("person") or current_event.get("speaker") or "the speaker"
        chains.append(
            {
                "chain_type": "update_chain",
                "person": person,
                "old_state": previous_event,
                "trigger_event": previous_event,
                "new_state": current_event,
                "turn_ids": [previous_turn, current_turn],
                "session_keys": [previous_event.get("session_key"), current_event.get("session_key")],
                "distractor_answer": f"{person}'s latest update: {answer_label(update.get('previous_quote'), 70)}",
            }
        )
    return chains


def extract_relation_chains(graph: dict[str, Any]) -> list[dict[str, Any]]:
    chains = []
    for relation in graph.get("person_relations", []):
        if not isinstance(relation, dict):
            continue
        evidence_quotes = [
            item for item in relation.get("evidence_quotes", [])
            if isinstance(item, dict) and item.get("turn_id") and item.get("quote")
        ]
        if len(evidence_quotes) < 2:
            continue
        chains.append(
            {
                "chain_type": "relation_chain",
                "source": relation.get("source"),
                "target": relation.get("target"),
                "relation_type": relation.get("relation_type") or "relationship",
                "interaction": evidence_quotes[0],
                "response": evidence_quotes[-1],
                "turn_ids": [str(item.get("turn_id")) for item in evidence_quotes[:3]],
                "evidence_quotes": evidence_quotes[:3],
            }
        )
    return chains


def extract_cross_session_chains(graph: dict[str, Any]) -> list[dict[str, Any]]:
    events = sorted_events(graph)
    chains = []
    for first_index, setup in enumerate(events):
        for bridge_index, bridge in enumerate(events[first_index + 1 :], start=first_index + 1):
            for result in events[bridge_index + 1 :]:
                turn_ids = {setup.get("turn_id"), bridge.get("turn_id"), result.get("turn_id")}
                if len(turn_ids) < 3 or setup.get("session_key") == result.get("session_key"):
                    continue
                shared_speaker = setup.get("speaker") and setup.get("speaker") == result.get("speaker")
                bridge_diff_speaker = bridge.get("speaker") and bridge.get("speaker") != setup.get("speaker")
                if not (shared_speaker or bridge_diff_speaker):
                    continue
                chains.append(
                    {
                        "chain_type": "cross_session_chain",
                        "setup": setup,
                        "bridge": bridge,
                        "result": result,
                        "turn_ids": [str(setup["turn_id"]), str(bridge["turn_id"]), str(result["turn_id"])],
                        "session_keys": [setup.get("session_key"), bridge.get("session_key"), result.get("session_key")],
                    }
                )
                if len(chains) >= 20:
                    return chains
    return chains


def extract_memory_graph_chains(graph: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    update_chains = extract_update_chains(graph)
    relation_chains = extract_relation_chains(graph)
    cross_session_chains = extract_cross_session_chains(graph)
    conflict_chains = [
        {**chain, "chain_type": "conflict_chain"}
        for chain in update_chains
        if option_label(chain.get("old_state", {}).get("quote")) and option_label(chain.get("new_state", {}).get("quote"))
    ]
    return {
        "update_chain": update_chains,
        "relation_chain": relation_chains,
        "conflict_chain": conflict_chains,
        "cross_session_chain": cross_session_chains,
    }


def state_chain_candidates(graph: dict[str, Any], chains: list[dict[str, Any]], start_index: int, limit: int, quote_map: dict[str, str] | None = None) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    distractors = [
        f"{chain.get('person')}'s latest update: {answer_label(chain.get('new_state', {}).get('quote'), 70)}"
        for chain in chains
    ]
    for chain in chains[:limit]:
        old_state = chain.get("old_state", {})
        new_state = chain.get("new_state", {})
        person = str(chain.get("person") or "the speaker")
        previous_turn, current_turn = chain["turn_ids"][:2]
        question = (
            f"After the earlier context \"{short(old_state.get('quote'), 70)}\", "
            f"what latest/current state should be remembered for {person}?"
        )
        answer = f"{person}'s latest update: {answer_label(new_state.get('quote'), 70)}"
        quotes = [
            {"turn_id": previous_turn, "quote": exact_quote(previous_turn, old_state.get("quote"), quote_map)},
            {"turn_id": current_turn, "quote": exact_quote(current_turn, new_state.get("quote"), quote_map)},
        ]
        candidate = candidate_base(start_index + len(candidates), question, answer, [previous_turn, current_turn], quotes, "final_state")
        candidate["expected_reasoning"] = (
            f"The chain starts at {previous_turn} and the later cross-session update is {current_turn}, "
            f"so the final remembered state is the later one."
        )
        candidate["_graph_candidate_type"] = "update_chain"
        candidate["_graph_chain"] = chain
        append_graph_distractors(candidate, [item for item in distractors if item != answer] + [chain.get("distractor_answer")])
        candidates.append(candidate)
    return candidates


def conflict_chain_candidates(graph: dict[str, Any], chains: list[dict[str, Any]], start_index: int, limit: int, quote_map: dict[str, str] | None = None) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    distractors = [chain.get("old_state", {}).get("quote") for chain in chains] + [chain.get("new_state", {}).get("quote") for chain in chains]
    for chain in chains[:limit]:
        old_state = chain.get("old_state", {})
        new_state = chain.get("new_state", {})
        previous_turn, current_turn = chain["turn_ids"][:2]
        person = str(chain.get("person") or "the speaker")
        question = (
            f"Which later update should override the earlier memory \"{short(old_state.get('quote'), 72)}\" "
            f"for {person}?"
        )
        answer = answer_label(new_state.get("quote"), 86)
        quotes = [
            {"turn_id": previous_turn, "quote": exact_quote(previous_turn, old_state.get("quote"), quote_map)},
            {"turn_id": current_turn, "quote": exact_quote(current_turn, new_state.get("quote"), quote_map)},
        ]
        candidate = candidate_base(start_index + len(candidates), question, answer, [previous_turn, current_turn], quotes, "conflict_resolution")
        candidate["expected_reasoning"] = (
            f"{previous_turn} is the earlier memory, but {current_turn} is later and therefore supplies the current state."
        )
        candidate["_graph_candidate_type"] = "conflict_chain"
        candidate["_graph_chain"] = chain
        append_graph_distractors(candidate, [item for item in distractors if item != new_state.get("quote")])
        candidates.append(candidate)
    return candidates


def relation_chain_candidates(graph: dict[str, Any], chains: list[dict[str, Any]], start_index: int, limit: int, quote_map: dict[str, str] | None = None) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    all_relation_events: list[str] = []
    for other_chain in chains:
        for item in other_chain.get("evidence_quotes", []):
            if isinstance(item, dict) and item.get("quote"):
                all_relation_events.append(answer_label(item.get("quote"), 86))
    for chain in chains[:limit]:
        source = str(chain.get("source") or "")
        target = str(chain.get("target") or "")
        if not source or not target:
            continue
        evidence_quotes = [
            {"turn_id": str(item.get("turn_id")), "quote": exact_quote(item.get("turn_id"), item.get("quote"), quote_map)}
            for item in chain.get("evidence_quotes", [])
            if isinstance(item, dict) and item.get("turn_id") and item.get("quote")
        ][:3]
        if len(evidence_quotes) < 2:
            continue
        relation = relation_label(chain.get("relation_type"))
        anchor = evidence_quotes[0]
        response = evidence_quotes[-1]
        answer = answer_label(response.get("quote"), 86)
        if not answer:
            continue
        question = (
            f"After the earlier interaction clue involving {source} and {target} in {anchor['turn_id']}, "
            f"what later concrete response or event involving them should be remembered?"
        )
        candidate = candidate_base(
            start_index + len(candidates),
            question,
            answer,
            [item["turn_id"] for item in evidence_quotes],
            evidence_quotes,
            "person_relation_temporal",
        )
        candidate["expected_reasoning"] = (
            f"{anchor['turn_id']} establishes the earlier {relation} clue between {source} and {target}; "
            f"{response['turn_id']} gives the later concrete response/event to remember."
        )
        candidate["_graph_candidate_type"] = "relation_chain"
        candidate["_graph_chain"] = chain
        forbidden = {answer.lower(), answer_label(anchor.get("quote"), 86).lower()}
        append_graph_distractors(candidate, [item for item in all_relation_events if item and item.lower() not in forbidden])
        candidates.append(candidate)
    return candidates




def turn_order_lookup(graph: dict[str, Any]) -> dict[str, tuple[int, int]]:
    lookup: dict[str, tuple[int, int]] = {}
    for event in sorted_events(graph):
        turn_id = str(event.get("turn_id") or "")
        if turn_id:
            lookup[turn_id] = event_sort_key(event)
    return lookup


def chain_final_order(chain: dict[str, Any], order_lookup: dict[str, tuple[int, int]]) -> tuple[int, int]:
    turn_ids = [str(item) for item in chain.get("turn_ids", []) if item is not None]
    for turn_id in reversed(turn_ids):
        if turn_id in order_lookup:
            return order_lookup[turn_id]
    return 999999, 999999


def multi_update_conflict_candidates(
    graph: dict[str, Any],
    chains: list[dict[str, Any]],
    start_index: int,
    limit: int,
    quote_map: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    order_lookup = turn_order_lookup(graph)
    by_person: dict[str, list[dict[str, Any]]] = {}
    for chain in chains:
        person = str(chain.get("person") or "the speaker")
        by_person.setdefault(person, []).append(chain)

    candidates: list[dict[str, Any]] = []
    all_state_labels = [
        f"{str(chain.get('person') or 'the speaker')}'s latest update: {answer_label(chain.get('new_state', {}).get('quote'), 70)}"
        for chain in chains
    ]
    for person, person_chains in sorted(by_person.items(), key=lambda item: (-len(item[1]), item[0])):
        ordered = sorted(person_chains, key=lambda item: chain_final_order(item, order_lookup))
        if len(ordered) < 2:
            continue
        # Use the earliest state, an intermediate/follow-up state, and the latest state.
        old_chain = ordered[0]
        middle_chain = ordered[-2]
        final_chain = ordered[-1]
        old_state = old_chain.get("old_state", {})
        middle_state = middle_chain.get("new_state", {})
        final_state = final_chain.get("new_state", {})
        old_turn = str(old_chain.get("turn_ids", [""])[0] or "")
        middle_turn = str(middle_chain.get("turn_ids", ["", ""])[-1] or "")
        final_turn = str(final_chain.get("turn_ids", ["", ""])[-1] or "")
        evidence = []
        for turn_id in [old_turn, middle_turn, final_turn]:
            if turn_id and turn_id not in evidence:
                evidence.append(turn_id)
        if len(evidence) < 3:
            continue
        answer = f"{person}'s latest update: {answer_label(final_state.get('quote'), 70)}"
        if not answer_label(final_state.get("quote")):
            continue
        question = (
            f"For {person}, the dialogue first suggested \"{short(old_state.get('quote'), 58)}\" "
            f"and later introduced the competing update \"{short(middle_state.get('quote'), 58)}\". "
            "By the final/current update, what should be remembered?"
        )
        quote_by_turn = {
            old_turn: old_state.get("quote"),
            middle_turn: middle_state.get("quote"),
            final_turn: final_state.get("quote"),
        }
        quotes = [
            {"turn_id": turn_id, "quote": exact_quote(turn_id, quote_by_turn.get(turn_id), quote_map)}
            for turn_id in evidence
        ]
        candidate = candidate_base(start_index + len(candidates), question, answer, evidence, quotes, "conflict_resolution")
        candidate["expected_reasoning"] = (
            f"{old_turn} gives an old memory for {person}, {middle_turn} is a later competing update, "
            f"and {final_turn} is the latest/current update that overrides both."
        )
        candidate["_graph_candidate_type"] = "conflict_chain"
        candidate["_graph_chain"] = {
            "chain_type": "multi_update_conflict_chain",
            "person": person,
            "turn_ids": evidence,
            "old_chain": old_chain,
            "middle_chain": middle_chain,
            "final_chain": final_chain,
        }
        priority_distractors = [
            f"{person}'s latest update: {answer_label(old_state.get('quote'), 70)}",
            f"{person}'s latest update: {answer_label(middle_state.get('quote'), 70)}",
        ]
        append_priority_graph_distractors(candidate, priority_distractors)
        append_graph_distractors(candidate, [item for item in all_state_labels if item != answer])
        candidates.append(candidate)
        if len(candidates) >= limit:
            break
    return candidates



def final_overwrite_state_candidates(
    graph: dict[str, Any],
    chains: list[dict[str, Any]],
    start_index: int,
    limit: int,
    quote_map: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    order_lookup = turn_order_lookup(graph)
    by_person: dict[str, list[dict[str, Any]]] = {}
    for chain in chains:
        person = str(chain.get("person") or "the speaker")
        by_person.setdefault(person, []).append(chain)

    all_state_labels = [
        f"{str(chain.get('person') or 'the speaker')}'s latest update: {answer_label(chain.get('new_state', {}).get('quote'), 70)}"
        for chain in chains
    ]
    candidates: list[dict[str, Any]] = []
    for person, person_chains in sorted(by_person.items(), key=lambda item: (-len(item[1]), item[0])):
        ordered = sorted(person_chains, key=lambda item: chain_final_order(item, order_lookup))
        if not ordered:
            continue
        first_chain = ordered[0]
        final_chain = ordered[-1]
        first_state = first_chain.get("old_state", {})
        prior_state = final_chain.get("old_state", {})
        final_state = final_chain.get("new_state", {})
        first_turn = str(first_chain.get("turn_ids", [""])[0] or "")
        prior_turn = str(final_chain.get("turn_ids", [""])[0] or "")
        final_turn = str(final_chain.get("turn_ids", ["", ""])[-1] or "")
        final_label = answer_label(final_state.get("quote"), 70)
        if not final_turn or not final_label:
            continue
        evidence: list[str] = []
        for turn_id in [first_turn, prior_turn, final_turn]:
            if turn_id and turn_id not in evidence:
                evidence.append(turn_id)
        if len(evidence) < 2:
            continue
        quote_by_turn = {
            first_turn: first_state.get("quote"),
            prior_turn: prior_state.get("quote"),
            final_turn: final_state.get("quote"),
        }
        answer = f"{person}'s latest update: {final_label}"
        if len(evidence) >= 3:
            question = (
                f"For {person}, after the earlier memory \"{short(first_state.get('quote'), 58)}\" "
                f"and the later revision \"{short(prior_state.get('quote'), 58)}\", "
                "what is the final/current state by the end of the dialogue?"
            )
        else:
            question = (
                f"For {person}, the dialogue earlier suggested \"{short(prior_state.get('quote'), 70)}\". "
                "What is the final/current state by the end of the dialogue?"
            )
        quotes = [
            {"turn_id": turn_id, "quote": exact_quote(turn_id, quote_by_turn.get(turn_id), quote_map)}
            for turn_id in evidence
        ]
        candidate = candidate_base(start_index + len(candidates), question, answer, evidence, quotes, "final_state")
        candidate["expected_reasoning"] = (
            f"Track the state updates for {person} in order and use {final_turn}, the latest evidence turn, "
            "as the final/current state."
        )
        candidate["_graph_candidate_type"] = "update_chain"
        candidate["_graph_chain"] = {
            "chain_type": "final_overwrite_state_chain",
            "person": person,
            "turn_ids": evidence,
            "first_chain": first_chain,
            "final_chain": final_chain,
        }
        priority_distractors = [
            f"{person}'s latest update: {answer_label(first_state.get('quote'), 70)}",
            f"{person}'s latest update: {answer_label(prior_state.get('quote'), 70)}",
        ]
        append_priority_graph_distractors(candidate, priority_distractors)
        append_graph_distractors(candidate, [item for item in all_state_labels if item != answer])
        candidates.append(candidate)
        if len(candidates) >= limit:
            break
    return candidates


def relation_state_update_candidates(
    graph: dict[str, Any],
    relation_chains: list[dict[str, Any]],
    update_chains: list[dict[str, Any]],
    start_index: int,
    limit: int,
    quote_map: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    order_lookup = turn_order_lookup(graph)
    updates_by_person: dict[str, list[dict[str, Any]]] = {}
    for chain in update_chains:
        person = str(chain.get("person") or "")
        if person:
            updates_by_person.setdefault(person, []).append(chain)
    for items in updates_by_person.values():
        items.sort(key=lambda item: chain_final_order(item, order_lookup))

    candidates: list[dict[str, Any]] = []
    state_distractors = [
        f"{chain.get('person') or 'the speaker'}'s latest update: {answer_label(chain.get('new_state', {}).get('quote'), 70)}"
        for chain in update_chains
    ]
    for relation in relation_chains:
        source = str(relation.get("source") or "")
        target = str(relation.get("target") or "")
        people = [person for person in [source, target] if person]
        if not people:
            continue
        evidence_quotes = [
            {"turn_id": str(item.get("turn_id")), "quote": exact_quote(item.get("turn_id"), item.get("quote"), quote_map)}
            for item in relation.get("evidence_quotes", [])
            if isinstance(item, dict) and item.get("turn_id") and item.get("quote")
        ][:2]
        if not evidence_quotes:
            continue
        for person in people:
            person_updates = updates_by_person.get(person, [])
            if not person_updates:
                continue
            latest = person_updates[-1]
            final_turn = str(latest.get("turn_ids", ["", ""])[-1] or "")
            if not final_turn:
                continue
            final_quote = latest.get("new_state", {}).get("quote")
            answer = f"{person}'s latest update: {answer_label(final_quote, 70)}"
            if not answer_label(final_quote):
                continue
            evidence = [item["turn_id"] for item in evidence_quotes]
            if final_turn not in evidence:
                evidence.append(final_turn)
            if len(evidence) < 2:
                continue
            question = (
                f"After the earlier interaction clue involving {source} and {target}, "
                f"what later/current update about {person} should be remembered rather than only the relation clue?"
            )
            quotes = evidence_quotes + [{"turn_id": final_turn, "quote": exact_quote(final_turn, final_quote, quote_map)}]
            candidate = candidate_base(start_index + len(candidates), question, answer, evidence, quotes, "person_relation_temporal")
            candidate["expected_reasoning"] = (
                f"The relation clue appears in {evidence_quotes[0]['turn_id']}; the later/current update for {person} is in {final_turn}."
            )
            candidate["_graph_candidate_type"] = "relation_chain"
            candidate["_graph_chain"] = {
                "chain_type": "relation_state_update_chain",
                "relation_chain": relation,
                "update_chain": latest,
                "turn_ids": evidence,
            }
            person_state_distractors: list[str] = []
            latest_old_quote = latest.get("old_state", {}).get("quote")
            if answer_label(latest_old_quote):
                person_state_distractors.append(f"{person}'s latest update: {answer_label(latest_old_quote, 70)}")
            for prior_update in person_updates[:-1]:
                prior_new_quote = prior_update.get("new_state", {}).get("quote")
                prior_old_quote = prior_update.get("old_state", {}).get("quote")
                if answer_label(prior_new_quote):
                    person_state_distractors.append(f"{person}'s latest update: {answer_label(prior_new_quote, 70)}")
                if answer_label(prior_old_quote):
                    person_state_distractors.append(f"{person}'s latest update: {answer_label(prior_old_quote, 70)}")
            append_priority_graph_distractors(candidate, person_state_distractors)
            append_graph_distractors(candidate, [item for item in state_distractors if item != answer])
            candidates.append(candidate)
            if len(candidates) >= limit:
                return candidates
    return candidates


def cross_session_chain_candidates_from_chains(graph: dict[str, Any], chains: list[dict[str, Any]], start_index: int, limit: int, quote_map: dict[str, str] | None = None) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    result_distractors = [chain.get("result", {}).get("phrase") for chain in chains]
    bridge_distractors = [chain.get("bridge", {}).get("phrase") for chain in chains]
    for chain in chains[:limit]:
        setup = chain.get("setup", {})
        bridge = chain.get("bridge", {})
        result = chain.get("result", {})
        setup_turn, bridge_turn, result_turn = chain["turn_ids"][:3]
        question = (
            f"Given the setup \"{short(setup.get('phrase'), 62)}\" and the bridge "
            f"\"{short(bridge.get('phrase'), 62)}\", what later cross-session outcome followed?"
        )
        answer = answer_label(result.get("phrase"), 86)
        quotes = [
            {"turn_id": setup_turn, "quote": exact_quote(setup_turn, setup.get("quote"), quote_map)},
            {"turn_id": bridge_turn, "quote": exact_quote(bridge_turn, bridge.get("quote"), quote_map)},
            {"turn_id": result_turn, "quote": exact_quote(result_turn, result.get("quote"), quote_map)},
        ]
        candidate = candidate_base(start_index + len(candidates), question, answer, [setup_turn, bridge_turn, result_turn], quotes, "evidence_bridge")
        candidate["expected_reasoning"] = f"{setup_turn} gives the setup, {bridge_turn} gives the bridge, and {result_turn} is the later outcome."
        candidate["_graph_candidate_type"] = "cross_session_chain"
        candidate["_graph_chain"] = chain
        append_priority_graph_distractors(candidate, [item for item in result_distractors if item != result.get("phrase")])
        append_graph_distractors(candidate, [item for item in bridge_distractors if item != result.get("phrase")])
        candidates.append(candidate)
    return candidates


def display_turn_id(turn: dict[str, Any]) -> str:
    for key in ("dia_id", "turn_ref", "turn_id", "turn_index"):
        value = turn.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def turn_session_key(turn: dict[str, Any]) -> str:
    return str(turn.get("session_key") or turn.get("session") or "").strip()


def turn_id_aliases(turn: dict[str, Any]) -> list[str]:
    aliases: list[str] = []
    for key in ("dia_id", "turn_ref", "turn_id", "turn_index"):
        value = turn.get(key)
        if value is not None and str(value).strip():
            aliases.append(str(value).strip())
    session_key = turn_session_key(turn)
    turn_index = turn.get("turn_index")
    session_match = re.search(r"(\d+)$", session_key)
    if session_match and turn_index is not None and str(turn_index).strip():
        aliases.append(f"D{session_match.group(1)}:{str(turn_index).strip()}")
    clean_aliases: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        if alias and alias not in seen:
            seen.add(alias)
            clean_aliases.append(alias)
    return clean_aliases


def build_evidence_window_dialogue(
    dialogue: list[dict[str, Any]],
    candidate: dict[str, Any],
    *,
    window_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if window_size < 0:
        return dialogue, {"enabled": False, "reason": "negative_window_size"}
    clean_dialogue = [turn for turn in dialogue if isinstance(turn, dict)]
    id_to_index: dict[str, int] = {}
    for index, turn in enumerate(clean_dialogue):
        for alias in turn_id_aliases(turn):
            id_to_index.setdefault(alias, index)
    evidence_ids = [str(item) for item in candidate.get("evolved_evidence", []) if item is not None]
    chain = candidate.get("_graph_chain") if isinstance(candidate.get("_graph_chain"), dict) else {}
    chain_turn_ids = [str(item) for item in chain.get("turn_ids", []) if item is not None]
    target_ids = []
    seen_ids = set()
    for turn_id in evidence_ids + chain_turn_ids:
        if turn_id and turn_id not in seen_ids:
            seen_ids.add(turn_id)
            target_ids.append(turn_id)

    selected_indices: set[int] = set()
    missing_ids = []
    for turn_id in target_ids:
        index = id_to_index.get(turn_id)
        if index is None:
            missing_ids.append(turn_id)
            continue
        start = max(0, index - window_size)
        end = min(len(clean_dialogue), index + window_size + 1)
        selected_indices.update(range(start, end))

    # Preserve compact session/time anchors so relative dates remain interpretable.
    selected_sessions = {
        turn_session_key(clean_dialogue[index])
        for index in selected_indices
        if turn_session_key(clean_dialogue[index])
    }
    for index, turn in enumerate(clean_dialogue):
        if turn_session_key(turn) in selected_sessions and (
            index == 0
            or turn_session_key(clean_dialogue[index - 1]) != turn_session_key(turn)
        ):
            selected_indices.add(index)

    if not selected_indices:
        return clean_dialogue, {
            "enabled": False,
            "reason": "no_evidence_turns_found",
            "original_turn_count": len(clean_dialogue),
            "window_turn_count": len(clean_dialogue),
            "missing_turn_ids": missing_ids,
        }

    window_dialogue = [copy.deepcopy(clean_dialogue[index]) for index in sorted(selected_indices)]
    return window_dialogue, {
        "enabled": True,
        "window_size": window_size,
        "original_turn_count": len(clean_dialogue),
        "window_turn_count": len(window_dialogue),
        "evidence_turn_ids": evidence_ids,
        "chain_turn_ids": chain_turn_ids,
        "missing_turn_ids": missing_ids,
    }


def candidate_result_with_dialogue_view(candidate_result: dict[str, Any], dialogue_view: list[dict[str, Any]], window_meta: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(candidate_result)
    updated["final_dialogue_full_turn_count"] = len(candidate_result.get("final_dialogue", [])) if isinstance(candidate_result.get("final_dialogue"), list) else None
    updated["final_dialogue"] = copy.deepcopy(dialogue_view)
    updated["evaluation_dialogue_view"] = window_meta
    return updated


def temporal_candidates(graph: dict[str, Any], start_index: int, limit: int, quote_map: dict[str, str] | None = None) -> list[dict[str, Any]]:
    events = sorted_events(graph)
    pairs: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for left_index, first in enumerate(events):
        for right_index, second in enumerate(events[left_index + 1 :], start=left_index + 1):
            if first.get("turn_id") == second.get("turn_id"):
                continue
            gap = right_index - left_index
            if gap < 1:
                continue
            pairs.append((gap, first, second))
    pairs.sort(key=lambda item: (-item[0], tuple(item[1].get("order") or (999999, 999999))))

    candidates: list[dict[str, Any]] = []
    for offset, (_, first, second) in enumerate(pairs[:limit]):
        index = start_index + offset
        first_phrase = short(first.get("phrase"))
        second_phrase = short(second.get("phrase"))
        question = f"Which dialogue event was mentioned earlier: {first_phrase} or {second_phrase}?"
        answer = str(first.get("phrase"))
        quotes = [
            {"turn_id": str(first["turn_id"]), "quote": exact_quote(first["turn_id"], first.get("quote"), quote_map)},
            {"turn_id": str(second["turn_id"]), "quote": exact_quote(second["turn_id"], second.get("quote"), quote_map)},
        ]
        candidate = candidate_base(
            index,
            question,
            answer,
            [str(first["turn_id"]), str(second["turn_id"])],
            quotes,
            "temporal_ordering",
        )
        candidate["expected_reasoning"] = (
            f"{first.get('turn_id')} appears before {second.get('turn_id')} in the dialogue timeline, "
            f"so {answer} is the earlier mentioned event."
        )
        candidate["_graph_candidate_type"] = "temporal_pair"
        candidates.append(candidate)
    return candidates


def final_state_candidates(graph: dict[str, Any], start_index: int, limit: int, quote_map: dict[str, str] | None = None) -> list[dict[str, Any]]:
    updates = [item for item in graph.get("state_updates", []) if isinstance(item, dict)]
    candidates: list[dict[str, Any]] = []
    for offset, update in enumerate(updates[:limit]):
        previous_turn = str(update.get("previous_turn_id") or "")
        current_turn = str(update.get("current_turn_id") or "")
        if not previous_turn or not current_turn:
            continue
        person = str(update.get("person") or "the speaker")
        current_quote = str(update.get("current_quote") or "")
        previous_quote = str(update.get("previous_quote") or "")
        if not is_actionable_state_update_quote(current_quote):
            continue
        index = start_index + len(candidates)
        question = f"By the latest relevant dialogue session, what was {person}'s updated state after the earlier context \"{short(previous_quote, 70)}\"?"
        answer = f"{person}'s latest update: {answer_label(current_quote, 70)}"
        quotes = [
            {"turn_id": previous_turn, "quote": exact_quote(previous_turn, previous_quote, quote_map)},
            {"turn_id": current_turn, "quote": exact_quote(current_turn, current_quote, quote_map)},
        ]
        candidate = candidate_base(index, question, answer, [previous_turn, current_turn], quotes, "final_state")
        candidate["expected_reasoning"] = (
            f"The earlier context is in {previous_turn}. The later relevant update is in {current_turn}: {current_quote}."
        )
        candidate["_graph_candidate_type"] = "state_update"
        append_priority_graph_distractors(candidate, [f"{person}'s latest update: {answer_label(previous_quote, 70)}"])
        append_graph_distractors(candidate, [
            f"{item.get('person') or 'the speaker'}'s latest update: {answer_label(item.get('current_quote'), 70)}"
            for item in updates
            if item is not update
        ])
        candidates.append(candidate)
    return candidates


def relation_candidates(graph: dict[str, Any], start_index: int, limit: int, quote_map: dict[str, str] | None = None) -> list[dict[str, Any]]:
    relations = [item for item in graph.get("person_relations", []) if isinstance(item, dict)]
    candidates: list[dict[str, Any]] = []
    for relation in relations[:limit]:
        evidence_quotes = [
            {"turn_id": str(item.get("turn_id")), "quote": exact_quote(item.get("turn_id"), item.get("quote"), quote_map)}
            for item in relation.get("evidence_quotes", [])
            if isinstance(item, dict) and item.get("turn_id") and item.get("quote")
        ][:3]
        evidence = [item["turn_id"] for item in evidence_quotes]
        if len(evidence) < 2:
            continue
        source = str(relation.get("source") or "")
        target = str(relation.get("target") or "")
        relation_type = str(relation.get("relation_type") or "relationship")
        if not source or not target:
            continue
        index = start_index + len(candidates)
        question = f"Which two people are linked by multiple dialogue clues involving {relation_type.replace('_', ' ')}?"
        answer = f"{source} and {target}"
        candidate = candidate_base(index, question, answer, evidence, evidence_quotes, "multi_hop_attribute")
        candidate["expected_reasoning"] = (
            f"The evidence turns repeatedly link {source} and {target} through {relation_type.replace('_', ' ')}."
        )
        candidate["_graph_candidate_type"] = "person_relation"
        candidates.append(candidate)
    return candidates


def cross_session_bridge_candidates(graph: dict[str, Any], start_index: int, limit: int, quote_map: dict[str, str] | None = None) -> list[dict[str, Any]]:
    events = sorted_events(graph)
    candidates: list[dict[str, Any]] = []
    for first_index, first in enumerate(events):
        for middle_index, middle in enumerate(events[first_index + 1 :], start=first_index + 1):
            for final in events[middle_index + 1 :]:
                if len({first.get("turn_id"), middle.get("turn_id"), final.get("turn_id")}) < 3:
                    continue
                if first.get("session_key") == final.get("session_key"):
                    continue
                index = start_index + len(candidates)
                question = (
                    f"Across the dialogue timeline, what event connected the earlier context "
                    f"\"{short(first.get('phrase'), 70)}\" to the later outcome \"{short(final.get('phrase'), 70)}\"?"
                )
                answer = answer_label(middle.get("phrase"), 70)
                quotes = [
                    {"turn_id": str(first["turn_id"]), "quote": exact_quote(first["turn_id"], first.get("quote"), quote_map)},
                    {"turn_id": str(middle["turn_id"]), "quote": exact_quote(middle["turn_id"], middle.get("quote"), quote_map)},
                    {"turn_id": str(final["turn_id"]), "quote": exact_quote(final["turn_id"], final.get("quote"), quote_map)},
                ]
                candidate = candidate_base(
                    index,
                    question,
                    answer,
                    [str(first["turn_id"]), str(middle["turn_id"]), str(final["turn_id"])],
                    quotes,
                    "multi_hop_attribute",
                )
                candidate["expected_reasoning"] = (
                    f"{first.get('turn_id')} gives the earlier context, {middle.get('turn_id')} supplies the bridge event, "
                    f"and {final.get('turn_id')} gives the later outcome."
                )
                candidate["_graph_candidate_type"] = "cross_session_bridge"
                candidates.append(candidate)
                if len(candidates) >= limit:
                    return candidates
    return candidates


def enrich_graph_distractors(graph: dict[str, Any], candidates: list[dict[str, Any]]) -> None:
    event_options = [answer_label(event.get("phrase"), 86) for event in sorted_events(graph)]
    state_options = [
        f"{item.get('person') or 'the speaker'}'s latest update: {answer_label(item.get('current_quote'), 70)}"
        for item in graph.get("state_updates", [])
        if isinstance(item, dict)
    ]
    relation_options = [
        relation_label(item.get("relation_type"))
        for item in graph.get("person_relations", [])
        if isinstance(item, dict)
    ]
    generic_relation_options = ["support", "collaboration", "question about", "shared experience", "co mentioned"]
    for candidate in candidates:
        if len(candidate.get("_graph_distractor_pool", [])) >= 3:
            continue
        evidence = {str(item) for item in candidate.get("evolved_evidence", [])}
        non_evidence_events = [
            answer_label(event.get("phrase"), 86)
            for event in sorted_events(graph)
            if str(event.get("turn_id")) not in evidence
        ]
        candidate_type = str(candidate.get("_graph_candidate_type") or "")
        evolution_type = str(candidate.get("evolution_type") or "")
        if evolution_type == "person_relation_temporal":
            append_graph_distractors(candidate, non_evidence_events + state_options + event_options)
        elif evolution_type in {"final_state", "conflict_resolution"}:
            append_graph_distractors(candidate, state_options + non_evidence_events + event_options)
        elif candidate_type in {"cross_session_chain", "cross_session_bridge", "temporal_pair"}:
            append_graph_distractors(candidate, non_evidence_events + event_options)
        else:
            append_graph_distractors(candidate, state_options + relation_options + non_evidence_events + event_options)


CANDIDATE_TYPE_ORDER = (
    "cross_session_chain",
    "cross_session_bridge",
    "conflict_chain",
    "relation_chain",
    "update_chain",
    "state_update",
    "temporal_pair",
)




def candidate_fingerprint(candidate: dict[str, Any]) -> tuple[str, str, tuple[str, ...]]:
    question = evolved_qa_validation.normalize_space(candidate.get("evolved_question")).lower()
    answer = evolved_qa_validation.normalize_space(candidate.get("evolved_answer")).lower()
    evidence = tuple(str(item) for item in candidate.get("evolved_evidence", []) if item is not None)
    return question, answer, evidence


def dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for candidate in candidates:
        key = candidate_fingerprint(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def candidate_type_quotas(candidate_type_weights: dict[str, float], candidate_pool: int) -> dict[str, int]:
    weighted = {
        key: float(candidate_type_weights.get(key, 0.0))
        for key in CANDIDATE_TYPE_ORDER
        if float(candidate_type_weights.get(key, 0.0)) > 0
    }
    if not weighted or candidate_pool <= 0:
        return {}
    ranked = sorted(weighted.items(), key=lambda item: (-item[1], CANDIDATE_TYPE_ORDER.index(item[0])))
    if len(ranked) >= candidate_pool:
        return {key: 1 for key, _ in ranked[:candidate_pool]}

    total = sum(weighted.values())
    raw = {key: value * candidate_pool / total for key, value in weighted.items()}
    quotas = {key: max(1, int(raw[key])) for key in weighted}

    while sum(quotas.values()) > candidate_pool:
        reducible = [key for key, value in quotas.items() if value > 1]
        if not reducible:
            break
        key = min(reducible, key=lambda item: (weighted[item], raw[item] - int(raw[item])))
        quotas[key] -= 1

    while sum(quotas.values()) < candidate_pool:
        key = max(weighted, key=lambda item: (raw[item] - int(raw[item]), weighted[item]))
        quotas[key] += 1
    return quotas


def select_weighted_candidates(
    buckets: dict[str, list[dict[str, Any]]],
    candidate_pool: int,
    candidate_type_weights: dict[str, float],
) -> list[dict[str, Any]]:
    quotas = candidate_type_quotas(candidate_type_weights, candidate_pool)
    selected: list[dict[str, Any]] = []
    used_ids: set[int] = set()
    active_types = [key for key in CANDIDATE_TYPE_ORDER if quotas.get(key, 0) > 0]

    # Interleave operator types so early high-yield buckets cannot fill the per-dialogue
    # selected quota before update/conflict/state candidates are even validated.
    for offset in range(max(quotas.values(), default=0)):
        for candidate_type in active_types:
            if offset >= quotas.get(candidate_type, 0):
                continue
            items = buckets.get(candidate_type, [])
            if offset >= len(items):
                continue
            candidate = items[offset]
            selected.append(candidate)
            used_ids.add(id(candidate))
            if len(selected) >= candidate_pool:
                return selected

    fill_order = sorted(
        [key for key in CANDIDATE_TYPE_ORDER if candidate_type_weights.get(key, 0.0) > 0],
        key=lambda key: (-candidate_type_weights.get(key, 0.0), CANDIDATE_TYPE_ORDER.index(key)),
    )
    for candidate_type in fill_order:
        for candidate in buckets.get(candidate_type, []):
            if id(candidate) in used_ids:
                continue
            selected.append(candidate)
            used_ids.add(id(candidate))
            if len(selected) >= candidate_pool:
                return selected
    return selected


def generate_graph_candidates(
    graph: dict[str, Any],
    candidate_pool: int,
    dialogue: list[dict[str, Any]] | None = None,
    candidate_type_weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    quote_map = dialogue_quote_map(dialogue or [])
    chains = extract_memory_graph_chains(graph)

    if candidate_type_weights:
        buckets: dict[str, list[dict[str, Any]]] = {}
        next_index = 0

        def add_bucket(candidate_type: str, items: list[dict[str, Any]]) -> None:
            nonlocal next_index
            items = dedupe_candidates(items)
            buckets[candidate_type] = items
            next_index += len(items)

        add_bucket("cross_session_chain", cross_session_chain_candidates_from_chains(graph, chains["cross_session_chain"], next_index, candidate_pool, quote_map))
        add_bucket("cross_session_bridge", cross_session_bridge_candidates(graph, next_index, candidate_pool, quote_map))
        add_bucket(
            "conflict_chain",
            multi_update_conflict_candidates(graph, chains["update_chain"], next_index, candidate_pool, quote_map)
            + conflict_chain_candidates(graph, chains["conflict_chain"], next_index, candidate_pool, quote_map),
        )
        add_bucket(
            "relation_chain",
            relation_state_update_candidates(graph, chains["relation_chain"], chains["update_chain"], next_index, candidate_pool, quote_map)
            + relation_chain_candidates(graph, chains["relation_chain"], next_index, candidate_pool, quote_map),
        )
        add_bucket(
            "update_chain",
            final_overwrite_state_candidates(graph, chains["update_chain"], next_index, candidate_pool, quote_map)
            + state_chain_candidates(graph, chains["update_chain"], next_index, candidate_pool, quote_map),
        )
        add_bucket("state_update", final_state_candidates(graph, next_index, candidate_pool, quote_map))
        add_bucket("temporal_pair", temporal_candidates(graph, next_index, candidate_pool, quote_map))
        candidates = dedupe_candidates(select_weighted_candidates(buckets, candidate_pool, candidate_type_weights))
        enrich_graph_distractors(graph, candidates)
        return candidates[:candidate_pool]

    candidates: list[dict[str, Any]] = []
    hard_quota = max(1, candidate_pool // 4)
    candidates.extend(multi_update_conflict_candidates(graph, chains["update_chain"], len(candidates), hard_quota, quote_map))
    candidates.extend(final_overwrite_state_candidates(graph, chains["update_chain"], len(candidates), hard_quota, quote_map))
    candidates.extend(state_chain_candidates(graph, chains["update_chain"], len(candidates), hard_quota, quote_map))
    candidates.extend(conflict_chain_candidates(graph, chains["conflict_chain"], len(candidates), hard_quota, quote_map))
    candidates.extend(relation_state_update_candidates(graph, chains["relation_chain"], chains["update_chain"], len(candidates), hard_quota, quote_map))
    candidates.extend(relation_chain_candidates(graph, chains["relation_chain"], len(candidates), hard_quota, quote_map))
    candidates.extend(cross_session_chain_candidates_from_chains(graph, chains["cross_session_chain"], len(candidates), hard_quota, quote_map))
    if len(candidates) < candidate_pool:
        candidates.extend(final_state_candidates(graph, len(candidates), max(1, hard_quota), quote_map))
    if len(candidates) < candidate_pool:
        candidates.extend(cross_session_bridge_candidates(graph, len(candidates), max(1, hard_quota), quote_map))
    remaining = max(0, candidate_pool - len(candidates))
    if remaining:
        candidates.extend(temporal_candidates(graph, len(candidates), remaining, quote_map))
    candidates = dedupe_candidates(candidates)[:candidate_pool]
    enrich_graph_distractors(graph, candidates)
    return candidates


def compact_graph_option_context(candidate: dict[str, Any], graph: dict[str, Any], limit: int = 24) -> dict[str, Any]:
    evidence_turns = {str(item) for item in candidate.get("evolved_evidence", []) if item is not None}
    chain = candidate.get("_graph_chain") if isinstance(candidate.get("_graph_chain"), dict) else {}
    chain_turns = {str(item) for item in chain.get("turn_ids", []) if item is not None}
    forbidden_turns = sorted(evidence_turns | chain_turns)
    priority_raw_pool = candidate.get("_graph_priority_distractor_pool", [])
    raw_pool = candidate.get("_graph_distractor_pool", [])
    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    pool_source_items: list[tuple[Any, str]] = []
    if isinstance(priority_raw_pool, list):
        pool_source_items.extend((item, "candidate_priority_graph_pool") for item in priority_raw_pool)
    if isinstance(raw_pool, list):
        pool_source_items.extend((item, "candidate_graph_pool") for item in raw_pool)
    for item, source in pool_source_items:
        option = normalize_option_value(item)
        if not option or option.lower() in seen:
            continue
        seen.add(option.lower())
        pool.append({"option": option, "source": source})
    for event in sorted_events(graph):
        turn_id = str(event.get("turn_id") or "")
        if turn_id in evidence_turns or turn_id in chain_turns:
            continue
        option = answer_label(event.get("phrase"), 86)
        if option and option.lower() not in seen:
            seen.add(option.lower())
            pool.append({
                "option": option,
                "source": "timeline_event",
                "turn_id": turn_id,
                "speaker": event.get("speaker"),
                "session_key": event.get("session_key"),
            })
    for update in graph.get("state_updates", []):
        if not isinstance(update, dict):
            continue
        turns = {str(update.get("previous_turn_id") or ""), str(update.get("current_turn_id") or "")}
        if turns & (evidence_turns | chain_turns):
            continue
        option = f"{update.get('person') or 'the speaker'}'s latest update: {answer_label(update.get('current_quote'), 70)}"
        if option and option.lower() not in seen:
            seen.add(option.lower())
            pool.append({
                "option": option,
                "source": "state_update",
                "turn_ids": sorted(turns),
                "person": update.get("person"),
            })
    for relation in graph.get("person_relations", []):
        if not isinstance(relation, dict):
            continue
        option = relation_label(relation.get("relation_type"))
        if option and option.lower() not in seen:
            seen.add(option.lower())
            pool.append({
                "option": option,
                "source": "person_relation",
                "source_person": relation.get("source"),
                "target_person": relation.get("target"),
            })
    return {
        "candidate_type": candidate.get("_graph_candidate_type"),
        "evolution_type": candidate.get("evolution_type"),
        "question": candidate.get("evolved_question"),
        "answer": candidate.get("evolved_answer"),
        "evidence_turns": sorted(evidence_turns),
        "chain": chain,
        "forbidden_sources": {
            "turn_ids": forbidden_turns,
            "rules": [
                "Do not use any option semantically equivalent to the answer.",
                "Use superseded states from the same conflict/final-state chain only when the question explicitly asks for latest/current/final state.",
                "Do not use partial-list or substring variants of the correct answer.",
                "Prefer distractors from different speakers, sessions, relation types, or independent state updates.",
            ],
        },
        "distractor_pool": pool[:limit],
    }


def build_compact_graph_option_messages(candidate: dict[str, Any], graph: dict[str, Any]) -> list[dict[str, str]]:
    payload = {
        "task": "Choose safe MCQ options for a graph-grounded dialogue QA using a compact memory-graph context.",
        "compact_graph_context": compact_graph_option_context(candidate, graph),
        "requirements": [
            "Return exactly 4 options.",
            "Include exactly one option semantically equivalent to the answer.",
            "Choose exactly 3 distractors from distractor_pool whenever possible.",
            "If a pool item needs minor cleanup for grammar, preserve its meaning and source.",
            "Distractors must be globally wrong for the question, not merely less complete.",
            "Distractors must not come from forbidden turn_ids unless they are explicitly superseded states in a latest/current/final-state question.",
            "Distractors must not be paraphrases, substrings, supersets, or partial-list variants of the answer.",
            "Do not invent new dialogue facts outside the compact graph context.",
            "Return strict JSON only.",
        ],
        "output_schema": {"options": ["correct answer or distractor", "option", "option", "option"]},
    }
    return [
        {"role": "system", "content": "You create adversarial but safe MCQ options from structured memory-graph evidence. Return strict JSON only."},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def generate_compact_graph_mcq_options(
    candidate: dict[str, Any],
    graph: dict[str, Any],
    *,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    request_timeout: int,
) -> list[str]:
    raw_text = qa_evolution.generation_test.call_chat_completion(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=build_compact_graph_option_messages(candidate, graph),
        temperature=temperature,
        request_timeout=request_timeout,
    )
    payload = qa_evolution.generation_test.extract_json_payload(raw_text)
    options = payload.get("options") if isinstance(payload, dict) else payload
    if not isinstance(options, list):
        raise ValueError("Compact graph option generator did not return a list.")
    return [qa_evolution.generation_test.normalize_option_text(str(option)) for option in options]


def graph_mcq_options(candidate: dict[str, Any]) -> tuple[list[str], int] | None:
    raw_pool = candidate.get("_graph_distractor_pool", [])
    priority_pool = candidate.get("_graph_priority_distractor_pool", [])
    if not isinstance(raw_pool, list):
        raw_pool = []
    if not isinstance(priority_pool, list):
        priority_pool = []
    if not raw_pool and not priority_pool:
        return None
    answer = normalize_option_value(candidate.get("evolved_answer"))
    question = str(candidate.get("evolved_question") or "")
    ordered_pool: list[tuple[str, Any]] = []
    priority_seen: set[str] = set()
    for item in priority_pool:
        option = normalize_option_value(item)
        if not option:
            continue
        priority_seen.add(option.lower())
        ordered_pool.append(("priority", item))
    for item in raw_pool:
        option = normalize_option_value(item)
        if not option or option.lower() in priority_seen:
            continue
        ordered_pool.append(("graph", item))

    distractors: list[str] = []
    priority_used = 0
    seen = {answer.lower()}
    for source, item in ordered_pool:
        option = normalize_option_value(item)
        if not option or option.lower() in seen:
            continue
        if answer_semantics.is_semantically_equivalent_answer(answer, option, question):
            continue
        answer_norm = answer.lower().rstrip(".")
        option_norm = option.lower().rstrip(".")
        if answer_norm.startswith(option_norm) or option_norm.startswith(answer_norm):
            continue
        if evolved_qa_validation.is_partial_list_answer(answer, option):
            continue
        seen.add(option.lower())
        distractors.append(option)
        if source == "priority":
            priority_used += 1
        if len(distractors) >= 3:
            break
    candidate["_graph_priority_distractor_used_count"] = priority_used
    if len(distractors) < 3:
        return None
    correct_index = int(hashlib.md5(f"{question}\n{answer}".encode("utf-8")).hexdigest()[:8], 16) % 4
    options: list[str] = []
    distractor_index = 0
    for index in range(4):
        if index == correct_index:
            options.append(answer)
        else:
            options.append(distractors[distractor_index])
            distractor_index += 1
    return options, correct_index


def split_allowed_types(value: str | None) -> set[str] | None:
    if not value:
        return None
    allowed = {item.strip() for item in value.split(",") if item.strip()}
    return allowed or None


def parse_candidate_type_weights(value: str | None) -> dict[str, float] | None:
    if not value:
        return None
    weights: dict[str, float] = {}
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid candidate type weight '{item}'. Expected candidate_type=weight.")
        key, raw_weight = item.split("=", 1)
        key = key.strip()
        if key not in CANDIDATE_TYPE_ORDER:
            raise ValueError(f"Unknown candidate_type '{key}'. Supported: {', '.join(CANDIDATE_TYPE_ORDER)}")
        weight = float(raw_weight.strip())
        if weight < 0:
            raise ValueError(f"candidate_type weight must be non-negative for {key}")
        weights[key] = weight
    return weights or None


def is_api_failure_reason(reason: Any) -> bool:
    text = str(reason or "").lower()
    return any(
        marker in text
        for marker in (
            "api request failed",
            "responsetimeout",
            "response stream timeout",
            "urlopen error",
            "connection",
            "actively refused",
            "积极拒绝",
            "no such file or directory",
        )
    )


def failure_summary(candidate: dict[str, Any], reason: str, **extra: Any) -> dict[str, Any]:
    payload = {
        "candidate_index": candidate.get("candidate_index"),
        "candidate_type": candidate.get("_graph_candidate_type"),
        "evolution_type": candidate.get("evolution_type"),
        "selected": False,
        "status": "rejected",
        "rejection_reason": reason,
        "candidate": candidate,
    }
    payload.update(extra)
    return payload


def summarize(payload: dict[str, Any]) -> dict[str, Any]:
    candidate_summaries = [item for item in payload.get("candidate_summaries", []) if isinstance(item, dict)]
    results = [item for item in payload.get("results", []) if isinstance(item, dict)]
    attempted = len({item.get("source_qa_index") for item in candidate_summaries})
    generated = len(candidate_summaries)
    rule_safe = sum(1 for item in candidate_summaries if item.get("rule_safe"))
    critic_safe = sum(1 for item in candidate_summaries if item.get("critic_safe"))
    option_safe = sum(1 for item in candidate_summaries if item.get("option_safe"))
    option_source_counts: dict[str, int] = {}
    for item in candidate_summaries:
        source = item.get("option_source")
        if source:
            option_source_counts[str(source)] = option_source_counts.get(str(source), 0) + 1
    selected = len(results)
    failures: dict[str, int] = {}
    for item in candidate_summaries:
        if item.get("selected"):
            continue
        reason = str(item.get("rejection_reason") or item.get("status") or "unknown").split(";")[0].strip()
        failures[reason] = failures.get(reason, 0) + 1
    type_counts: dict[str, int] = {}
    for result in results:
        etype = str(result.get("qa", {}).get("evolution_type") or "unknown")
        type_counts[etype] = type_counts.get(etype, 0) + 1
    light_values = [
        float(result["graph_multiqa"]["light_eval_accuracy"])
        for result in results
        if isinstance(result.get("graph_multiqa", {}).get("light_eval_accuracy"), (int, float))
    ]
    full_context_values = [
        float(result["graph_multiqa"]["full_context_accuracy"])
        for result in results
        if isinstance(result.get("graph_multiqa", {}).get("full_context_accuracy"), (int, float))
    ]
    evidence_window_values = [
        float(result["graph_multiqa"]["evidence_window_accuracy"])
        for result in results
        if isinstance(result.get("graph_multiqa", {}).get("evidence_window_accuracy"), (int, float))
    ]
    retrieval_gaps = [
        float(result["graph_multiqa"]["retrieval_gap"])
        for result in results
        if isinstance(result.get("graph_multiqa", {}).get("retrieval_gap"), (int, float))
    ]

    def summarize_group(items: list[dict[str, Any]]) -> dict[str, Any]:
        group_generated = len(items)
        group_rule_safe = sum(1 for item in items if item.get("rule_safe"))
        group_critic_safe = sum(1 for item in items if item.get("critic_safe"))
        group_option_safe = sum(1 for item in items if item.get("option_safe"))
        group_selected = sum(1 for item in items if item.get("selected"))
        group_light_values = [
            float(item.get("light_eval_accuracy"))
            for item in items
            if isinstance(item.get("light_eval_accuracy"), (int, float))
        ]
        group_full_values = [
            float(item.get("full_context_accuracy"))
            for item in items
            if isinstance(item.get("full_context_accuracy"), (int, float))
        ]
        group_window_values = [
            float(item.get("evidence_window_accuracy"))
            for item in items
            if isinstance(item.get("evidence_window_accuracy"), (int, float))
        ]
        group_retrieval_gaps = [
            float(item.get("retrieval_gap"))
            for item in items
            if isinstance(item.get("retrieval_gap"), (int, float))
        ]
        group_failures: dict[str, int] = {}
        for item in items:
            if item.get("selected"):
                continue
            reason = str(item.get("rejection_reason") or item.get("status") or "unknown").split(";")[0].strip()
            group_failures[reason] = group_failures.get(reason, 0) + 1
        group_option_sources: dict[str, int] = {}
        for item in items:
            source = item.get("option_source")
            if source:
                group_option_sources[str(source)] = group_option_sources.get(str(source), 0) + 1
        return {
            "generated_count": group_generated,
            "rule_safe_count": group_rule_safe,
            "rule_safe_rate": round(group_rule_safe / group_generated, 4) if group_generated else None,
            "critic_safe_count": group_critic_safe,
            "critic_safe_rate": round(group_critic_safe / group_generated, 4) if group_generated else None,
            "option_safe_count": group_option_safe,
            "option_safe_rate": round(group_option_safe / group_generated, 4) if group_generated else None,
            "selected_count": group_selected,
            "selected_rate": round(group_selected / group_generated, 4) if group_generated else None,
            "average_light_eval_accuracy": round(sum(group_light_values) / len(group_light_values), 4) if group_light_values else None,
            "average_full_context_accuracy": round(sum(group_full_values) / len(group_full_values), 4) if group_full_values else None,
            "average_evidence_window_accuracy": round(sum(group_window_values) / len(group_window_values), 4) if group_window_values else None,
            "average_retrieval_gap": round(sum(group_retrieval_gaps) / len(group_retrieval_gaps), 4) if group_retrieval_gaps else None,
            "option_source_counts": group_option_sources,
            "failure_reason_counts": dict(sorted(group_failures.items(), key=lambda kv: (-kv[1], kv[0]))[:10]),
        }

    by_candidate_type: dict[str, list[dict[str, Any]]] = {}
    by_evolution_type: dict[str, list[dict[str, Any]]] = {}
    for item in candidate_summaries:
        by_candidate_type.setdefault(str(item.get("candidate_type") or "unknown"), []).append(item)
        by_evolution_type.setdefault(str(item.get("evolution_type") or "unknown"), []).append(item)

    return {
        "attempted_dialogue_count": attempted,
        "generated_candidate_count": generated,
        "rule_safe_count": rule_safe,
        "rule_safe_rate": round(rule_safe / generated, 4) if generated else None,
        "critic_safe_count": critic_safe,
        "critic_safe_rate": round(critic_safe / generated, 4) if generated else None,
        "option_safe_count": option_safe,
        "option_safe_rate": round(option_safe / generated, 4) if generated else None,
        "option_source_counts": option_source_counts,
        "selected_qa_count": selected,
        "selected_by_evolution_type": type_counts,
        "average_light_eval_accuracy": round(sum(light_values) / len(light_values), 4) if light_values else None,
        "average_full_context_accuracy": round(sum(full_context_values) / len(full_context_values), 4) if full_context_values else None,
        "average_evidence_window_accuracy": round(sum(evidence_window_values) / len(evidence_window_values), 4) if evidence_window_values else None,
        "average_retrieval_gap": round(sum(retrieval_gaps) / len(retrieval_gaps), 4) if retrieval_gaps else None,
        "failure_reason_counts": failures,
        "operator_report_by_candidate_type": {
            key: summarize_group(items)
            for key, items in sorted(by_candidate_type.items())
        },
        "operator_report_by_evolution_type": {
            key: summarize_group(items)
            for key, items in sorted(by_evolution_type.items())
        },
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }


def checkpoint(output_prefix: Path, payload: dict[str, Any]) -> None:
    payload["graph_multiqa_summary"] = summarize(payload)
    save_json(output_prefix.with_suffix(".json"), payload)
    save_json(output_prefix.with_name(output_prefix.name + "_report.json"), payload["graph_multiqa_summary"])


def main() -> int:
    args = parse_args()
    output_prefix = Path(args.output_prefix)
    result_path = output_prefix.with_suffix(".json")
    if args.resume and result_path.exists():
        output_payload = load_json(result_path)
    else:
        output_payload = {
            "source_path": args.source,
            "graph_path": args.graph,
            "generation_mode": "memory_graph_multiqa",
            "results": [],
            "candidate_summaries": [],
        }
    if args.retry_api_failures:
        previous_summaries = [
            item for item in output_payload.get("candidate_summaries", [])
            if isinstance(item, dict)
        ]
        kept_summaries = [
            item for item in previous_summaries
            if item.get("selected") or not is_api_failure_reason(item.get("rejection_reason"))
        ]
        removed_count = len(previous_summaries) - len(kept_summaries)
        output_payload["candidate_summaries"] = kept_summaries
        output_payload["api_failure_retry"] = {
            "enabled": True,
            "removed_previous_api_failure_count": removed_count,
        }
        if removed_count:
            print(f"Retrying {removed_count} previous API-failed candidates.")

    source_payload = load_json(args.source)
    graph_payload = load_json(args.graph)
    source_by_key = {result_key(result): result for result in collect_results(source_payload)}
    graphs = [graph for graph in graph_payload.get("graphs", []) if isinstance(graph, dict)]
    graphs.sort(key=graph_sort_key)
    if args.max_dialogues is not None:
        graphs = graphs[: args.max_dialogues]

    api_config = dialogue_hardening.build_config(args)
    solver_base_url = args.solver_base_url or api_config["base_url"]
    solver_api_key = args.solver_api_key or api_config["api_key"]
    solver_models = dialogue_hardening.split_models(args.models)
    allowed_evolution_types = split_allowed_types(args.allowed_evolution_types)
    candidate_type_weights = parse_candidate_type_weights(args.candidate_type_weights)
    output_payload["candidate_type_weights"] = candidate_type_weights
    completed = {
        (item.get("source_qa_index"), item.get("candidate_index"))
        for item in output_payload.get("candidate_summaries", [])
        if isinstance(item, dict)
    }
    selected_per_source: dict[Any, int] = {}
    for result in output_payload.get("results", []):
        if isinstance(result, dict):
            source_index = result.get("graph_multiqa", {}).get("source_qa_index")
            selected_per_source[source_index] = selected_per_source.get(source_index, 0) + 1

    for graph in graphs:
        source_key = result_key_from_values(graph.get("sample_id", graph.get("sample_index", "")), graph.get("qa_index", -1))
        source_result = source_by_key.get(source_key)
        if not source_result:
            continue
        source_index = graph.get("qa_index")
        if selected_per_source.get(source_index, 0) >= args.qa_per_dialogue:
            continue
        dialogue = source_result.get("final_dialogue", [])
        original_qa = source_result.get("qa", {}) if isinstance(source_result.get("qa"), dict) else {}
        candidates = generate_graph_candidates(
            graph,
            args.candidate_pool_per_dialogue,
            dialogue,
            candidate_type_weights=candidate_type_weights,
        )
        if allowed_evolution_types is not None:
            candidates = [
                candidate for candidate in candidates
                if str(candidate.get("evolution_type") or "") in allowed_evolution_types
            ]
        print(f"Graph multi-QA qa_index={source_index}: {len(candidates)} candidates")
        for candidate in candidates:
            candidate_key = (source_index, candidate.get("candidate_index"))
            if candidate_key in completed:
                continue
            if selected_per_source.get(source_index, 0) >= args.qa_per_dialogue:
                break
            dialogue_view, evidence_window = build_evidence_window_dialogue(
                dialogue,
                candidate,
                window_size=args.evidence_window_size,
            )
            rule = evolved_qa_validation.validate_evolved_qa_rules(candidate, dialogue, original_qa)
            if not rule.get("rule_safe"):
                output_payload["candidate_summaries"].append(
                    failure_summary(candidate, str(rule.get("rejection_reason") or "rule_rejected"), source_qa_index=source_index, rule_safe=False, evidence_window=evidence_window, rule_validation=rule)
                )
                checkpoint(output_prefix, output_payload)
                continue
            try:
                critic = evolved_qa_validation.validate_evolved_qa_with_critic(
                    candidate,
                    dialogue,
                    original_qa=original_qa,
                    base_url=api_config["base_url"],
                    api_key=api_config["api_key"],
                    model=api_config["challenger_model"],
                    temperature=args.critic_temperature,
                    request_timeout=api_config["request_timeout"],
                )
                if not critic.get("critic_safe"):
                    output_payload["candidate_summaries"].append(
                        failure_summary(candidate, str(critic.get("reason") or "critic_rejected"), source_qa_index=source_index, rule_safe=True, critic_safe=False, evidence_window=evidence_window, rule_validation=rule, critic=critic)
                    )
                    checkpoint(output_prefix, output_payload)
                    continue
                graph_options = graph_mcq_options(candidate)
                option_source = "graph" if graph_options is not None else "compact_graph_llm"
                if graph_options is None:
                    raw_options = generate_compact_graph_mcq_options(
                        candidate,
                        graph,
                        base_url=api_config["base_url"],
                        api_key=api_config["api_key"],
                        model=api_config["challenger_model"],
                        temperature=args.option_temperature,
                        request_timeout=api_config["request_timeout"],
                    )
                    ordered_options, correct_index = qa_evolution.reorder_options(
                        candidate["evolved_question"],
                        candidate["evolved_answer"],
                        raw_options,
                    )
                else:
                    ordered_options, correct_index = graph_options
                option_validation = evolved_qa_validation.validate_evolved_mcq_options(
                    candidate,
                    ordered_options,
                    dialogue,
                    semantic_equivalent=answer_semantics.is_semantically_equivalent_answer,
                )
                if not option_validation.get("option_safe"):
                    output_payload["candidate_summaries"].append(
                        failure_summary(candidate, str(option_validation.get("rejection_reason") or "option_rejected"), source_qa_index=source_index, rule_safe=True, critic_safe=True, option_safe=False, option_source=option_source, evidence_window=evidence_window, rule_validation=rule, critic=critic, option_validation=option_validation)
                    )
                    checkpoint(output_prefix, output_payload)
                    continue
                candidate_result = qa_evolution.apply_evolved_candidate_to_result(
                    source_result,
                    candidate,
                    ordered_options,
                    correct_index,
                )
                eval_candidate_result = candidate_result_with_dialogue_view(candidate_result, dialogue_view, evidence_window)
                light_eval_full = None
                light_eval_window = None
                if args.solver_eval_view in {"full", "both"}:
                    light_eval_full = qa_evolution.evaluate_light_candidate(
                        candidate_result,
                        model_names=solver_models,
                        base_url=solver_base_url,
                        api_key=solver_api_key,
                        request_timeout=args.light_request_timeout or api_config["request_timeout"],
                        temperature=args.light_temperature,
                        max_retries=args.max_retries,
                        candidates_per_model=args.light_candidates_per_model,
                    )
                if args.solver_eval_view in {"window", "both"}:
                    light_eval_window = qa_evolution.evaluate_light_candidate(
                        eval_candidate_result,
                        model_names=solver_models,
                        base_url=solver_base_url,
                        api_key=solver_api_key,
                        request_timeout=args.light_request_timeout or api_config["request_timeout"],
                        temperature=args.light_temperature,
                        max_retries=args.max_retries,
                        candidates_per_model=args.light_candidates_per_model,
                    )
                full_context_accuracy = light_eval_full.get("average_accuracy") if isinstance(light_eval_full, dict) else None
                evidence_window_accuracy = light_eval_window.get("average_accuracy") if isinstance(light_eval_window, dict) else None
                if isinstance(full_context_accuracy, (int, float)):
                    light_accuracy = full_context_accuracy
                else:
                    light_accuracy = evidence_window_accuracy
                retrieval_gap = None
                if isinstance(full_context_accuracy, (int, float)) and isinstance(evidence_window_accuracy, (int, float)):
                    retrieval_gap = round(float(evidence_window_accuracy) - float(full_context_accuracy), 4)
                light_eval = {
                    "solver_eval_view": args.solver_eval_view,
                    "selection_accuracy_source": "full_context" if isinstance(full_context_accuracy, (int, float)) else "evidence_window",
                    "average_accuracy": light_accuracy,
                    "full_context_accuracy": full_context_accuracy,
                    "evidence_window_accuracy": evidence_window_accuracy,
                    "retrieval_gap": retrieval_gap,
                    "full_context_eval": light_eval_full,
                    "evidence_window_eval": light_eval_window,
                }
                if (
                    args.min_light_accuracy is not None
                    and isinstance(light_accuracy, (int, float))
                    and float(light_accuracy) < args.min_light_accuracy
                ):
                    output_payload["candidate_summaries"].append(
                        failure_summary(candidate, "light_eval_too_hard_or_unstable", source_qa_index=source_index, rule_safe=True, critic_safe=True, option_safe=True, option_source=option_source, evidence_window=evidence_window, light_eval_accuracy=light_accuracy, full_context_accuracy=full_context_accuracy, evidence_window_accuracy=evidence_window_accuracy, retrieval_gap=retrieval_gap, rule_validation=rule, critic=critic, option_validation=option_validation, light_eval=light_eval)
                    )
                    checkpoint(output_prefix, output_payload)
                    continue
                if (
                    not args.include_easy
                    and isinstance(light_accuracy, (int, float))
                    and float(light_accuracy) > args.max_light_accuracy
                ):
                    output_payload["candidate_summaries"].append(
                        failure_summary(candidate, "light_eval_too_easy", source_qa_index=source_index, rule_safe=True, critic_safe=True, option_safe=True, option_source=option_source, evidence_window=evidence_window, light_eval_accuracy=light_accuracy, full_context_accuracy=full_context_accuracy, evidence_window_accuracy=evidence_window_accuracy, retrieval_gap=retrieval_gap, rule_validation=rule, critic=critic, option_validation=option_validation, light_eval=light_eval)
                    )
                    checkpoint(output_prefix, output_payload)
                    continue
                candidate_result["qa_index"] = len(output_payload["results"])
                candidate_result["source_qa_index"] = source_index
                candidate_result["graph_multiqa"] = {
                    "source_qa_index": source_index,
                    "source_question": graph.get("source_question"),
                    "candidate_index": candidate.get("candidate_index"),
                    "candidate_type": candidate.get("_graph_candidate_type"),
                    "light_eval_accuracy": light_accuracy,
                    "full_context_accuracy": full_context_accuracy,
                    "evidence_window_accuracy": evidence_window_accuracy,
                    "retrieval_gap": retrieval_gap,
                    "rule_validation": rule,
                    "critic": critic,
                    "option_validation": option_validation,
                    "option_source": option_source,
                    "evidence_window": evidence_window,
                    "light_eval": light_eval,
                }
                output_payload["results"].append(candidate_result)
                output_payload["candidate_summaries"].append(
                    {
                        "source_qa_index": source_index,
                        "candidate_index": candidate.get("candidate_index"),
                        "candidate_type": candidate.get("_graph_candidate_type"),
                        "evolution_type": candidate.get("evolution_type"),
                        "selected": True,
                        "status": "selected",
                        "rule_safe": True,
                        "critic_safe": True,
                        "option_safe": True,
                        "option_source": option_source,
                        "evidence_window": evidence_window,
                        "light_eval_accuracy": light_accuracy,
                        "full_context_accuracy": full_context_accuracy,
                        "evidence_window_accuracy": evidence_window_accuracy,
                        "retrieval_gap": retrieval_gap,
                        "rule_validation": rule,
                        "critic": critic,
                        "option_validation": option_validation,
                        "light_eval": light_eval,
                        "candidate": candidate,
                    }
                )
                selected_per_source[source_index] = selected_per_source.get(source_index, 0) + 1
                checkpoint(output_prefix, output_payload)
            except Exception as error:  # noqa: BLE001
                output_payload["candidate_summaries"].append(
                    failure_summary(candidate, str(error), source_qa_index=source_index, rule_safe=True, evidence_window=evidence_window, rule_validation=rule)
                )
                checkpoint(output_prefix, output_payload)

    checkpoint(output_prefix, output_payload)
    print(json.dumps(output_payload["graph_multiqa_summary"], ensure_ascii=False, indent=2))
    print(f"Saved graph multi-QA result to: {output_prefix.with_suffix('.json').resolve()}")
    print(f"Saved graph multi-QA report to: {output_prefix.with_name(output_prefix.name + '_report.json').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
