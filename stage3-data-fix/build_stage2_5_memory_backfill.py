#!/usr/bin/env python3
"""Build Stage2.5 LoCoMo memory graphs and backfill existing Stage2 canonical parents.

The graph builder reads only:
  - conversation
  - event_summary
  - observation

It deliberately does not read qa or session_summary for graph construction.

For each existing Stage2 parent, only source-memory events whose complete evidence
turn set is already present in that parent's dialogue are injected. This prevents
future/unseen LoCoMo sessions from leaking into an existing chunk-derived branch.
"""
from __future__ import annotations

import argparse
import collections
import copy
import difflib
import hashlib
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

USED_LINEAGES = {"conv-26", "conv-30", "conv-41", "conv-42", "conv-43", "conv-44"}
STOPWORDS = set(
    """a an the and or but if then to of in on at for with from by as is are was were
    be been being has have had do does did this that these those it its his her their
    our my your who whom whose which what when where why how into over under after
    before during while about up down out off again further can could may might must
    shall should will would just very really also so than too only own same such no
    nor not recently currently conversation""".split()
)
LEAK_RE = re.compile(r"raw_qa_|(?:^|[\s\"])(?:Q|A):\s", re.I)
TURN_ID_RE = re.compile(r"D\d+:\d+")


def stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def norm_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def norm_tokens(text: Any) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", norm_space(text).lower())
        if token not in STOPWORDS and len(token) > 1
    ]


def text_similarity(left: Any, right: Any) -> float:
    left_tokens = norm_tokens(left)
    right_tokens = norm_tokens(right)
    left_set = set(left_tokens)
    right_set = set(right_tokens)
    if not left_set or not right_set:
        return 0.0
    overlap = len(left_set & right_set) / min(len(left_set), len(right_set))
    jaccard = len(left_set & right_set) / len(left_set | right_set)
    sequence = difflib.SequenceMatcher(
        None, " ".join(left_tokens), " ".join(right_tokens)
    ).ratio()
    return 0.45 * overlap + 0.35 * jaccard + 0.20 * sequence


def evidence_ids(raw: Any) -> List[str]:
    if isinstance(raw, str):
        value = raw.strip()
        return [value] if TURN_ID_RE.fullmatch(value) else []
    if isinstance(raw, list):
        result: List[str] = []
        for item in raw:
            result.extend(evidence_ids(item))
        return result
    return []


def flatten_source_dialogue(record: Mapping[str, Any]) -> List[Dict[str, Any]]:
    conversation = record["conversation"]
    sessions: List[Tuple[int, str]] = []
    for key in conversation:
        match = re.fullmatch(r"session_(\d+)", key)
        if match:
            sessions.append((int(match.group(1)), key))
    sessions.sort()

    turns: List[Dict[str, Any]] = []
    for session_index, key in sessions:
        session_date_time = conversation.get(f"session_{session_index}_date_time")
        for turn_index, turn in enumerate(conversation[key], 1):
            turns.append(
                {
                    "turn_id": f"D{session_index}:{turn_index}",
                    "speaker": turn.get("speaker"),
                    "text": norm_space(turn.get("text")),
                    "session_index": session_index,
                    "turn_index": turn_index,
                    "session_date_time": session_date_time,
                }
            )
    return turns


def infer_status(text: str) -> str:
    value = text.lower()
    if any(term in value for term in ("cancelled", "canceled", "called off", "no longer plans")):
        return "canceled"
    if any(
        term in value
        for term in (
            "plans to",
            "planning to",
            "is planning",
            "hopes to",
            "wants to",
            "would like to",
            "intends to",
            "is considering",
            "are considering",
            "going to",
            "will ",
        )
    ):
        return "planned"
    if any(
        term in value
        for term in (
            "failed",
            "did not pass",
            "was rejected",
            "lost his job",
            "lost her job",
            "lost their job",
        )
    ):
        return "completed_negative"
    if any(
        term in value
        for term in (
            "used to",
            "formerly",
            "previously",
            "last year",
            "years ago",
            "as a kid",
            "when she was",
            "when he was",
        )
    ):
        return "historical"
    if re.search(
        r"\b(attended|went|visited|returned|joined|started|began|launched|finished|"
        r"completed|won|lost|bought|adopted|moved|signed up|painted|ran|met|spoke|"
        r"gave|reached out|opened|graduated|passed|took|organized|donated)\b",
        value,
    ):
        return "completed"
    return "active"


def infer_event_type(text: str) -> str:
    value = text.lower()
    rules = [
        ("employment", ("job", "banker", "door dash", "workplace", "employed", "unemployed", "career")),
        ("education", ("school", "education", "class", "course", "degree", "university", "college", "study", "studying", "test", "exam", "certification")),
        ("travel", ("trip", "travel", "visited", "visit ", "returned from", "vacation", "camping", "beach", "paris", "rome", "mountains", "forest")),
        ("community_service", ("volunteer", "charity", "shelter", "donated", "mentorship", "community", "toy drive", "service-focused")),
        ("creative_project", ("paint", "painting", "screenplay", "book", "blog", "photograph", "photo", "camera", "dance piece", "choreography")),
        ("business", ("business", "store", "studio", "campaign", "wholesaler", "advertising", "ad campaign", "customer")),
        ("relationship", ("friend", "relationship", "single", "married", "wife", "husband", "family", "kids", "children", "aunt", "mentor")),
        ("health_wellbeing", ("health", "stress", "anxiety", "therapy", "mental", "injur", "hospital", "self-care", "relax", "destress")),
        ("preference_interest", ("favorite", "likes", "loves", "enjoys", "passion", "interested", "hobby")),
        ("possession", ("owns", "bought", "has a ", "got a ", "car", "pet", "dog", "cat", "turtle")),
        ("location", ("lives in", "moved to", "moved from", "office", "location")),
        ("plan_goal", ("plans to", "planning to", "hopes to", "wants to", "goal", "aspiration", "considering", "going to")),
        ("life_event", ("birthday", "lost", "won", "joined", "started", "began", "attended", "went", "visited", "returned", "graduated", "passed", "failed")),
    ]
    for event_type, terms in rules:
        if any(term in value for term in terms):
            return event_type
    return "memory_fact"


def memory_value(entity: str, text: str) -> str:
    proposition = norm_space(text)
    pattern = re.compile(rf"^{re.escape(entity)}(?:'s)?\s+", re.I)
    compact = pattern.sub("", proposition, count=1)
    return compact or proposition


def existing_event_text(event: Mapping[str, Any]) -> str:
    provenance = event.get("provenance") or {}
    return " ".join(
        str(value or "")
        for value in (
            event.get("entity"),
            event.get("slot"),
            event.get("value"),
            provenance.get("supporting_span"),
        )
    )


def duplicate_to_existing(
    source_event: Mapping[str, Any], existing_events: Sequence[Mapping[str, Any]]
) -> Tuple[float, Mapping[str, Any] | None]:
    best_score = 0.0
    best_event: Mapping[str, Any] | None = None
    source_entity = str(source_event.get("entity") or "").casefold()
    for existing in existing_events:
        if source_entity != str(existing.get("entity") or "").casefold():
            continue
        score = text_similarity(source_event.get("memory_text"), existing_event_text(existing))
        if score > best_score:
            best_score = score
            best_event = existing
    return best_score, best_event


def build_lineage_memory_graph(
    record: Mapping[str, Any]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    lineage = record["sample_id"]
    turns = flatten_source_dialogue(record)
    turn_map = {turn["turn_id"]: turn for turn in turns}
    turn_order = {turn["turn_id"]: index + 1 for index, turn in enumerate(turns)}

    atomic_events: List[Dict[str, Any]] = []
    by_session_person: MutableMapping[Tuple[int, str], List[Dict[str, Any]]] = collections.defaultdict(list)
    seen_atomic = set()

    # Deliberate boundary: record["qa"] is never read.
    for key, observation_by_person in record.get("observation", {}).items():
        match = re.fullmatch(r"session_(\d+)_observation", key)
        if not match:
            continue
        session_index = int(match.group(1))
        for person, items in observation_by_person.items():
            if not isinstance(items, list):
                continue
            for item_index, item in enumerate(items, 1):
                if not isinstance(item, list) or len(item) < 2:
                    continue
                claim = norm_space(item[0])
                event_evidence_ids = list(dict.fromkeys(evidence_ids(item[1])))
                if not claim or not event_evidence_ids:
                    continue
                if any(turn_id not in turn_map for turn_id in event_evidence_ids):
                    continue
                duplicate_key = (person.casefold(), claim.casefold(), tuple(event_evidence_ids))
                if duplicate_key in seen_atomic:
                    continue
                seen_atomic.add(duplicate_key)
                event_type = infer_event_type(claim)
                event_id = (
                    f"{lineage}:OBS:{session_index:02d}:{item_index:03d}:"
                    f"{stable_hash(duplicate_key)[:8]}"
                )
                event = {
                    "event_id": event_id,
                    "node_type": "atomic_memory_event",
                    "entity": person,
                    "event_type": event_type,
                    "slot": event_type,
                    "value": memory_value(person, claim),
                    "memory_text": claim,
                    "status": infer_status(claim),
                    "session_index": session_index,
                    "session_date_time": turn_map[event_evidence_ids[0]].get("session_date_time"),
                    "evidence_turn_ids": event_evidence_ids,
                    "supporting_spans": [turn_map[turn_id]["text"] for turn_id in event_evidence_ids],
                    "required_for_dialogue": False,
                    "provenance": {
                        "source": "locomo_observation",
                        "source_lineage_key": lineage,
                        "qa_used": False,
                        "verified_turn_ids_exist": True,
                    },
                }
                event["global_time_index"] = min(turn_order[turn_id] for turn_id in event_evidence_ids)
                atomic_events.append(event)
                by_session_person[(session_index, person)].append(event)

    summary_events: List[Dict[str, Any]] = []
    rejects: List[Dict[str, Any]] = []
    for key, event_summary in record.get("event_summary", {}).items():
        match = re.fullmatch(r"events_session_(\d+)", key)
        if not match:
            continue
        session_index = int(match.group(1))
        summary_date = event_summary.get("date")
        for person, items in event_summary.items():
            if person == "date" or not isinstance(items, list):
                continue
            for item_index, raw_summary in enumerate(items, 1):
                summary = norm_space(raw_summary)
                if not summary:
                    continue
                candidates = by_session_person.get((session_index, person), [])
                ranked = sorted(
                    (
                        (text_similarity(summary, candidate["memory_text"]), candidate)
                        for candidate in candidates
                    ),
                    key=lambda pair: pair[0],
                    reverse=True,
                )
                linked_events = [candidate for score, candidate in ranked if score >= 0.24][:2]
                if not linked_events and ranked and ranked[0][0] >= 0.14:
                    linked_events = [ranked[0][1]]
                summary_evidence_ids = list(
                    dict.fromkeys(
                        turn_id
                        for candidate in linked_events
                        for turn_id in candidate["evidence_turn_ids"]
                    )
                )

                if not summary_evidence_ids:
                    same_session_turns = [
                        turn
                        for turn in turns
                        if turn["session_index"] == session_index and turn["speaker"] == person
                    ]
                    ranked_turns = sorted(
                        (
                            (text_similarity(summary, turn["text"]), turn)
                            for turn in same_session_turns
                        ),
                        key=lambda pair: pair[0],
                        reverse=True,
                    )
                    if ranked_turns and ranked_turns[0][0] >= 0.16:
                        summary_evidence_ids = [ranked_turns[0][1]["turn_id"]]

                if not summary_evidence_ids:
                    rejects.append(
                        {
                            "source_lineage_key": lineage,
                            "session_index": session_index,
                            "entity": person,
                            "summary_text": summary,
                            "reason": "no_turn_grounded_evidence",
                        }
                    )
                    continue

                event_type = infer_event_type(summary)
                event_id = (
                    f"{lineage}:SUM:{session_index:02d}:{item_index:03d}:"
                    f"{stable_hash((person, summary))[:8]}"
                )
                event = {
                    "event_id": event_id,
                    "node_type": "session_event_summary",
                    "entity": person,
                    "event_type": event_type,
                    "slot": event_type,
                    "value": memory_value(person, summary),
                    "memory_text": summary,
                    "status": infer_status(summary),
                    "session_index": session_index,
                    "session_date_time": summary_date,
                    "evidence_turn_ids": summary_evidence_ids,
                    "supporting_spans": [turn_map[turn_id]["text"] for turn_id in summary_evidence_ids],
                    "summarizes_event_ids": [event["event_id"] for event in linked_events],
                    "required_for_dialogue": False,
                    "provenance": {
                        "source": "locomo_event_summary",
                        "source_lineage_key": lineage,
                        "qa_used": False,
                        "alignment_method": "observation_or_turn_lexical_alignment",
                    },
                }
                event["global_time_index"] = min(turn_order[turn_id] for turn_id in summary_evidence_ids)
                summary_events.append(event)

    events = sorted(
        atomic_events + summary_events,
        key=lambda event: (
            event["session_index"],
            0 if event["node_type"] == "atomic_memory_event" else 1,
            event["event_id"],
        ),
    )
    edges = [
        {
            "edge_type": "summarizes",
            "source_event_id": summary_event["event_id"],
            "target_event_id": atomic_event_id,
        }
        for summary_event in summary_events
        for atomic_event_id in summary_event["summarizes_event_ids"]
    ]
    graph = {
        "schema_version": "locomo_lineage_memory_graph_v1",
        "graph_id": (
            f"{lineage}:full_memory_graph:"
            f"{stable_hash([event['event_id'] for event in events])[:16]}"
        ),
        "source_lineage_key": lineage,
        "speakers": [
            record["conversation"].get("speaker_a"),
            record["conversation"].get("speaker_b"),
        ],
        "source_turn_count": len(turns),
        "events": events,
        "edges": edges,
        "event_count": len(events),
        "atomic_event_count": len(atomic_events),
        "summary_event_count": len(summary_events),
        "metadata": {
            "qa_used_for_graph_construction": False,
            "fact_authority": "conversation_turns",
            "candidate_sources": ["event_summary", "observation"],
            "session_summary_used": False,
        },
    }
    return graph, turns, rejects


def enhance_parent(
    parent: Mapping[str, Any], lineage_graph: Mapping[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    output = copy.deepcopy(parent)
    parent_turn_ids = {turn["turn_id"] for turn in output["dialogue"]["turns"]}
    existing_events = copy.deepcopy(output["semantic_parent_graph"]["events"])
    added_events: List[Dict[str, Any]] = []
    duplicate_edges: List[Dict[str, Any]] = []
    duplicate_merge_count = 0

    for source_event in lineage_graph["events"]:
        source_evidence = set(source_event["evidence_turn_ids"])
        if not source_evidence.issubset(parent_turn_ids):
            continue
        similarity, existing_match = duplicate_to_existing(source_event, existing_events)
        if similarity >= 0.78 and existing_match is not None:
            duplicate_merge_count += 1
            mutable_match = next(
                event
                for event in existing_events
                if event.get("event_id") == existing_match.get("event_id")
            )
            provenance = mutable_match.setdefault("provenance", {})
            provenance.setdefault("memory_backfill_matches", []).append(
                {
                    "source_memory_event_id": source_event["event_id"],
                    "similarity": round(similarity, 4),
                    "evidence_turn_ids": source_event["evidence_turn_ids"],
                    "memory_text": source_event["memory_text"],
                }
            )
            mutable_match["evidence_turn_ids"] = list(
                dict.fromkeys(
                    (mutable_match.get("evidence_turn_ids") or [])
                    + source_event["evidence_turn_ids"]
                )
            )
            duplicate_edges.append(
                {
                    "edge_type": "same_as",
                    "source_event_id": source_event["event_id"],
                    "target_event_id": mutable_match["event_id"],
                    "similarity": round(similarity, 4),
                }
            )
        else:
            event_copy = copy.deepcopy(source_event)
            event_copy["required_for_dialogue"] = False
            event_copy["provenance"]["backfilled_into_parent_sample_id"] = output["sample_id"]
            added_events.append(event_copy)

    combined_events = existing_events + added_events
    event_ids = [event["event_id"] for event in combined_events]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError(f"duplicate event IDs after backfill: {output['sample_id']}")
    included_ids = set(event_ids)
    inherited_edges = [
        copy.deepcopy(edge)
        for edge in lineage_graph.get("edges", [])
        if edge["source_event_id"] in included_ids and edge["target_event_id"] in included_ids
    ]

    old_sample_id = output["sample_id"]
    base_graph_id = output["semantic_parent_graph"].get("graph_id")
    graph_id_seed = {
        "base_graph_id": base_graph_id,
        "lineage_memory_graph_id": lineage_graph["graph_id"],
        "event_ids": event_ids,
    }
    new_graph_id = (
        f"{old_sample_id}:memory_backfilled:{stable_hash(graph_id_seed)[:16]}"
    )
    visible_source_count = len(added_events) + duplicate_merge_count
    new_graph = {
        "schema_version": "semantic_memory_graph_v4_backfilled",
        "graph_id": new_graph_id,
        "parent_graph_id": base_graph_id,
        "graph_role": "stage2_canonical_plus_source_memory_backfill",
        "events": combined_events,
        "edges": inherited_edges + duplicate_edges,
        "event_count": len(combined_events),
        "existing_verified_event_count": len(existing_events),
        "source_memory_event_added_count": len(added_events),
        "source_memory_duplicate_merged_count": duplicate_merge_count,
        "source_memory_full_event_count": lineage_graph["event_count"],
        "source_memory_visible_event_count": visible_source_count,
        "source_memory_unseen_event_count": lineage_graph["event_count"] - visible_source_count,
        "metadata": {
            "qa_used_for_graph_construction": False,
            "backfill_policy": "inject_only_if_all_evidence_turn_ids_exist_in_parent_dialogue",
            "unseen_source_sessions_injected": False,
            "base_stage2_graph_preserved": True,
            "lineage_memory_graph_id": lineage_graph["graph_id"],
        },
    }
    new_graph["graph_sha256"] = stable_hash(new_graph)
    new_sample_id = (
        f"stage3_parent_v4_"
        f"{stable_hash((old_sample_id, new_graph['graph_sha256']))[:20]}"
    )

    output["schema_version"] = "stage3_parent_v4_memory_backfilled"
    output["base_parent_sample_id"] = old_sample_id
    output["sample_id"] = new_sample_id
    output["source_id"] = new_sample_id
    output["semantic_parent_graph"] = new_graph
    output["memory_backfill_audit"] = {
        "base_parent_sample_id": old_sample_id,
        "source_lineage_key": output["lineage_group"],
        "parent_dialogue_turn_count": len(output["dialogue"]["turns"]),
        "full_lineage_memory_event_count": lineage_graph["event_count"],
        "visible_source_event_count": visible_source_count,
        "added_event_count": len(added_events),
        "duplicate_merged_count": duplicate_merge_count,
        "unseen_source_event_count": lineage_graph["event_count"] - visible_source_count,
        "all_added_evidence_present": all(
            set(event["evidence_turn_ids"]).issubset(parent_turn_ids)
            for event in added_events
        ),
        "qa_used": False,
        "raw_qa_leakage_count": 0,
    }
    audit_row = copy.deepcopy(output["memory_backfill_audit"])
    audit_row["new_parent_sample_id"] = new_sample_id
    audit_row["new_graph_id"] = new_graph_id
    audit_row["new_event_count"] = len(combined_events)
    audit_row["old_event_count"] = len(existing_events)
    return output, audit_row


def validate_outputs(
    lineage_graphs: Sequence[Mapping[str, Any]],
    enhanced_parents: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    for graph in lineage_graphs:
        event_ids = [event["event_id"] for event in graph["events"]]
        if len(event_ids) != len(set(event_ids)):
            issues.append({"scope": graph["graph_id"], "code": "duplicate_event_ids"})
        for event in graph["events"]:
            if LEAK_RE.search(json.dumps(event, ensure_ascii=False)):
                issues.append(
                    {
                        "scope": graph["graph_id"],
                        "code": "qa_leakage",
                        "event_id": event["event_id"],
                    }
                )
    for parent in enhanced_parents:
        graph = parent["semantic_parent_graph"]
        event_ids = [event["event_id"] for event in graph["events"]]
        if len(event_ids) != len(set(event_ids)):
            issues.append({"scope": parent["sample_id"], "code": "duplicate_event_ids"})
        parent_turn_ids = {turn["turn_id"] for turn in parent["dialogue"]["turns"]}
        for event in graph["events"]:
            provenance = event.get("provenance") or {}
            if provenance.get("source") in {"locomo_observation", "locomo_event_summary"}:
                if not set(event.get("evidence_turn_ids") or []).issubset(parent_turn_ids):
                    issues.append(
                        {
                            "scope": parent["sample_id"],
                            "code": "missing_parent_evidence",
                            "event_id": event["event_id"],
                        }
                    )
                if LEAK_RE.search(json.dumps(event, ensure_ascii=False)):
                    issues.append(
                        {
                            "scope": parent["sample_id"],
                            "code": "qa_leakage",
                            "event_id": event["event_id"],
                        }
                    )
    return issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--locomo", required=True, type=Path)
    parser.add_argument("--parents", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    locomo_records = read_json(args.locomo)
    parent_rows = read_jsonl(args.parents)

    lineage_graphs: Dict[str, Dict[str, Any]] = {}
    source_dialogues: Dict[str, List[Dict[str, Any]]] = {}
    rejects: List[Dict[str, Any]] = []
    for record in locomo_records:
        lineage = record.get("sample_id")
        if lineage not in USED_LINEAGES:
            continue
        graph, turns, graph_rejects = build_lineage_memory_graph(record)
        lineage_graphs[lineage] = graph
        source_dialogues[lineage] = turns
        rejects.extend(graph_rejects)

    missing_lineages = sorted(
        {parent["lineage_group"] for parent in parent_rows} - set(lineage_graphs)
    )
    if missing_lineages:
        raise ValueError(f"missing LoCoMo source lineages: {missing_lineages}")

    enhanced_parents: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    for parent in parent_rows:
        enhanced, audit = enhance_parent(parent, lineage_graphs[parent["lineage_group"]])
        enhanced_parents.append(enhanced)
        audit_rows.append(audit)

    lineage_rows = [
        {
            "schema_version": "stage2_5_lineage_memory_root_v1",
            "source_lineage_key": lineage,
            "source_dialogue": {
                "turn_count": len(source_dialogues[lineage]),
                "turns": source_dialogues[lineage],
            },
            "memory_graph": lineage_graphs[lineage],
        }
        for lineage in sorted(lineage_graphs)
    ]

    issues = validate_outputs(
        [row["memory_graph"] for row in lineage_rows], enhanced_parents
    )
    if issues:
        raise ValueError(f"output validation failed with {len(issues)} issues: {issues[:5]}")

    write_jsonl(args.output_dir / "stage2_5_lineage_memory_graphs.jsonl", lineage_rows)
    write_jsonl(args.output_dir / "stage3_parent_v4_memory_backfilled.jsonl", enhanced_parents)
    write_jsonl(args.output_dir / "stage2_5_backfill_audit.jsonl", audit_rows)
    write_jsonl(args.output_dir / "stage2_5_backfill_rejects.jsonl", rejects)

    per_lineage = {}
    for lineage, graph in sorted(lineage_graphs.items()):
        parent_audits = [row for row in audit_rows if row["source_lineage_key"] == lineage]
        per_lineage[lineage] = {
            "full_source_turn_count": len(source_dialogues[lineage]),
            "full_memory_event_count": graph["event_count"],
            "atomic_event_count": graph["atomic_event_count"],
            "summary_event_count": graph["summary_event_count"],
            "parent_count": len(parent_audits),
            "mean_visible_source_events_per_parent": round(
                statistics.mean(row["visible_source_event_count"] for row in parent_audits),
                3,
            ),
            "min_visible_source_events_per_parent": min(
                row["visible_source_event_count"] for row in parent_audits
            ),
            "max_visible_source_events_per_parent": max(
                row["visible_source_event_count"] for row in parent_audits
            ),
        }

    old_counts = [row["old_event_count"] for row in audit_rows]
    new_counts = [row["new_event_count"] for row in audit_rows]
    added_counts = [row["added_event_count"] for row in audit_rows]
    summary = {
        "schema_version": "stage2_5_backfill_summary_v1",
        "input": {
            "locomo_record_count": len(locomo_records),
            "used_lineages": sorted(lineage_graphs),
            "stage2_parent_count": len(parent_rows),
        },
        "construction_policy": {
            "graph_inputs": ["conversation", "event_summary", "observation"],
            "qa_used": False,
            "session_summary_used": False,
            "parent_injection_rule": "all evidence turn IDs must exist in parent dialogue",
            "unseen_source_sessions_injected": False,
        },
        "lineage_graphs": per_lineage,
        "parent_backfill": {
            "parent_count": len(audit_rows),
            "old_event_count_mean": round(statistics.mean(old_counts), 3),
            "new_event_count_mean": round(statistics.mean(new_counts), 3),
            "new_event_count_min": min(new_counts),
            "new_event_count_max": max(new_counts),
            "added_event_count_mean": round(statistics.mean(added_counts), 3),
            "total_events_added": sum(added_counts),
            "total_duplicates_merged": sum(
                row["duplicate_merged_count"] for row in audit_rows
            ),
        },
        "quality": {
            "validation_issue_count": len(issues),
            "event_summary_reject_count": len(rejects),
            "qa_leakage_count": 0,
            "all_added_evidence_present": all(
                row["all_added_evidence_present"] for row in audit_rows
            ),
        },
    }
    with (args.output_dir / "stage2_5_backfill_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
