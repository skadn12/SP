#!/usr/bin/env python3
import json
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent
RAW_QA_SLOT_RE = re.compile(r"^raw_qa_", re.I)
QA_WRAPPER_RE = re.compile(r"(?:^|\s)Q\s*:\s*.+?(?:^|\s)A\s*:\s*", re.I | re.S)
BACKFILLED_NODE_TYPES = {"atomic_memory_event", "session_event_summary"}


def read_jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main():
    lineages = read_jsonl(OUT / "stage2_5_lineage_memory_graphs.jsonl")
    parents = read_jsonl(OUT / "stage3_parent_v4_memory_backfilled.jsonl")
    audits = read_jsonl(OUT / "stage2_5_backfill_audit.jsonl")
    summary = json.load((OUT / "stage2_5_backfill_summary.json").open(encoding="utf-8"))

    assert len(lineages) == 6
    assert len(parents) == 77
    assert len(audits) == 77
    assert summary["quality"]["validation_issue_count"] == 0
    assert summary["quality"]["qa_leakage_count"] == 0
    assert summary["parent_backfill"]["new_event_count_mean"] > 100

    event_count = 0
    backfilled_event_count = 0
    missing_backfilled_evidence_count = 0
    raw_qa_slot_count = 0
    qa_wrapper_count = 0
    duplicate_event_id_row_count = 0
    event_counts = []
    for parent in parents:
        turns = {turn["turn_id"] for turn in parent["dialogue"]["turns"]}
        graph = parent["semantic_parent_graph"]
        event_counts.append(len(graph["events"]))
        event_count += len(graph["events"])
        event_ids = [event["event_id"] for event in graph["events"]]
        if len(event_ids) != len(set(event_ids)):
            duplicate_event_id_row_count += 1
        for event in graph["events"]:
            if event.get("node_type") in BACKFILLED_NODE_TYPES:
                backfilled_event_count += 1
                if not set(event.get("evidence_turn_ids") or []).issubset(turns):
                    missing_backfilled_evidence_count += 1
            if RAW_QA_SLOT_RE.search(str(event.get("slot") or "")):
                raw_qa_slot_count += 1
            for value in (
                event.get("value"),
                event.get("memory_text"),
                event.get("supporting_span"),
                *(event.get("supporting_spans") or []),
            ):
                if isinstance(value, str) and QA_WRAPPER_RE.search(value):
                    qa_wrapper_count += 1

    assert event_count == 9105
    assert backfilled_event_count == 8720
    assert min(event_counts) == 96
    assert max(event_counts) == 148
    assert missing_backfilled_evidence_count == 0
    assert raw_qa_slot_count == 0
    assert qa_wrapper_count == 0
    assert duplicate_event_id_row_count == 0

    print(
        "PASS: 6 lineage graphs; 77 enhanced parents; 9105 total events; "
        "8720 backfilled events; 0 missing evidence IDs; 0 raw_qa slots; "
        "0 Q:/A: wrappers; 0 duplicate event-id rows"
    )


if __name__ == "__main__":
    main()
