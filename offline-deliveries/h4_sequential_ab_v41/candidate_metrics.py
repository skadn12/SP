"""Body-token accounting for collection, never a semantic or training gate.

One record represents one Realizer response, including rejected attempts. Raw
model verdicts are bound, not converted to acceptance. Diagnostic replay is
reported separately and cannot increase fresh collection yield.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable


SCHEMA_VERSION = "h4.candidate_metrics.v41.1"
DISPOSITIONS = {
    "retained": "retained", "quarantined": "quarantined", "rejected": "rejected",
    "RETAINED_PENDING_REVIEW": "retained", "QUARANTINED": "quarantined",
    "REJECTED": "rejected", "retained_pending_review": "retained", "hard_reject": "rejected",
}
ROUTES = {
    "A": "A", "B": "B", "old_event_continuation": "A", "dataset_new_event": "B",
}
BUCKETS = ("generated", "retained", "quarantined", "rejected", "admitted")


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _counts():
    return {"scenes": 0, "turns": 0, "body_tokens": 0}


def _buckets():
    return {name: _counts() for name in BUCKETS}


def _add(target, source):
    for key in target:
        target[key] += source[key]


def measure_turns(turns, count_text_tokens: Callable[[str], int]):
    """Count dialogue text only; malformed entries are explicitly unmeasurable.

This intentionally does not estimate tokens in broken JSON, prompt text,
speaker labels, metadata or reviewer output. A malformed response may still
have readable turns, which count as generated/rejected but never clean yield.
"""
    if not isinstance(turns, list):
        return {"turns": 0, "body_tokens": 0}, False
    valid = True
    measured = {"turns": 0, "body_tokens": 0}
    for turn in turns:
        text = turn.get("text") if isinstance(turn, dict) else None
        if not isinstance(text, str) or not text.strip():
            valid = False
            continue
        tokens = count_text_tokens(text)
        if type(tokens) is not int or tokens < 0:
            raise ValueError("invalid_body_token_count")
        measured["turns"] += 1
        measured["body_tokens"] += tokens
    return measured, valid


def summarize_records(records: Iterable[dict], *, count_text_tokens: Callable[[str], int],
                      baseline_turns=(), scope="fresh_collection"):
    """Summarize immutable collection records without admitting any training data.

Required record keys: candidate_id, route, disposition, realization,
provenance. Provenance is fresh_collection or diagnostic_replay. Optional
raw_review is retained as a digest plus its exact model status. If an explicit
review_status is supplied, it must equal raw_review.status (or None if absent).

Generated totals count all measurable attempts, including duplicates and
rejected repairs. Retained, quarantined and rejected are disjoint outcomes;
only retained contributes to the provisional candidate cumulative length.
"""
    if scope not in ("fresh_collection", "offline_diagnostic"):
        raise ValueError("invalid_collection_scope")
    records = list(records)
    totals, diagnostic = _buckets(), _buckets()
    routes = {letter: _buckets() for letter in ("A", "B")}
    diagnostic_routes = {letter: _buckets() for letter in ("A", "B")}
    rows, seen = [], set()
    unmeasurable = {"fresh_collection": 0, "diagnostic_replay": 0}
    attempts = {"fresh_collection": 0, "diagnostic_replay": 0}
    for record in records:
        candidate_id = record.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in seen:
            raise ValueError("invalid_or_duplicate_candidate_id")
        seen.add(candidate_id)
        route = ROUTES.get(record.get("route"))
        if route is None:
            raise ValueError("invalid_candidate_route")
        disposition = DISPOSITIONS.get(record.get("disposition"))
        if disposition is None:
            raise ValueError("invalid_candidate_disposition")
        provenance = record.get("provenance")
        if provenance not in attempts:
            raise ValueError("candidate_provenance_required")
        if scope == "offline_diagnostic" and provenance != "diagnostic_replay":
            raise ValueError("diagnostic_run_cannot_claim_fresh_candidate")
        if record.get("training_pool_eligible", False) is not False or record.get("admitted", False) is not False:
            raise ValueError("candidate_collection_cannot_admit_training_data")
        raw_review = record.get("raw_review")
        parsed_review = raw_review
        if isinstance(raw_review, str):
            try:
                parsed_review = json.loads(raw_review)
            except (ValueError, TypeError):
                parsed_review = None
        raw_status = parsed_review.get("status") if isinstance(parsed_review, dict) else None
        raw_verdict = parsed_review.get("verdict", raw_status) if isinstance(parsed_review, dict) else None
        if "review_status" in record and record["review_status"] != raw_status:
            raise ValueError("raw_review_verdict_rewritten")
        if "review_verdict" in record and record["review_verdict"] != raw_verdict:
            raise ValueError("raw_review_verdict_rewritten")
        realization = record.get("realization")
        turns = realization.get("turns") if isinstance(realization, dict) else None
        measured, readable = measure_turns(turns, count_text_tokens)
        if disposition in ("retained", "quarantined") and (not readable or not measured["turns"]):
            raise ValueError("retained_candidate_has_unreadable_dialogue")
        counts = {"scenes": int(measured["turns"] > 0), **measured}
        attempts[provenance] += 1
        unmeasurable[provenance] += int(not readable)
        bucket_set = totals if provenance == "fresh_collection" else diagnostic
        route_set = routes if provenance == "fresh_collection" else diagnostic_routes
        for bucket in ("generated", disposition):
            _add(bucket_set[bucket], counts)
            _add(route_set[route][bucket], counts)
        rows.append({
            "candidate_id": candidate_id, "route": route, "disposition": disposition,
            "provenance": provenance, "record_sha256": digest(record),
            "realization_sha256": digest(realization), "raw_review_sha256": digest(raw_review),
            "raw_review_status": raw_status, "raw_review_verdict": raw_verdict,
            "body_measurable": readable, **counts,
            "training_pool_eligible": False,
        })
    baseline, readable = measure_turns(list(baseline_turns), count_text_tokens)
    if not readable:
        raise ValueError("baseline_dialogue_unreadable")
    projected = {key: baseline[key] + totals["retained"][key] for key in baseline}
    return {
        "schema_version": SCHEMA_VERSION, "scope": scope,
        "body_token_definition": "Sum local frozen tokenizer encode(turn.text, add_special_tokens=False); metadata and API tokens excluded.",
        "generated_definition": "All measurable Realizer attempts, including rejected drafts and duplicates; not clean candidate yield.",
        "retained_definition": "Mechanically usable pending candidates; semantic or training acceptance is not implied.",
        "diagnostic_policy": "Reused diagnostic responses never increase fresh collection yield or formal baseline.",
        **totals, "by_route": routes,
        "diagnostic_replay": {**diagnostic, "by_route": diagnostic_routes},
        "attempt_counts": attempts, "attempts_with_unmeasurable_body": unmeasurable,
        "formal_accepted_baseline": baseline,
        "formal_accepted_cumulative": dict(baseline),
        "provisional_candidate_cumulative": projected,
        "candidate_rows": rows, "records_sha256": digest(records),
        "training_pool_eligible": False, "semantic_certified": False,
        "training_admitted_scenes": 0, "training_admitted_body_tokens": 0,
    }


def verify_summary(report, records, *, count_text_tokens, baseline_turns=(), scope="fresh_collection"):
    """Recompute accounting, bindings and verdict preservation; no semantic PASS."""
    expected = summarize_records(records, count_text_tokens=count_text_tokens,
                                 baseline_turns=baseline_turns, scope=scope)
    if report != expected:
        raise ValueError("candidate_accounting_or_artifact_binding_changed")
    return {"status": "ARTIFACTS_CONSISTENT", "semantic_certified": False,
            "training_pool_eligible": False, "records": len(expected["candidate_rows"])}


def bind_files(paths):
    """Bind exact bytes, useful for raw model responses and immutable decisions."""
    return [{"path": str(path), "sha256": digest(path.read_bytes())}
            for path in sorted({Path(value).resolve() for value in paths})]


def verify_files(bindings):
    seen = set()
    for row in bindings:
        path = Path(row["path"]).resolve()
        if path in seen:
            raise ValueError("duplicate_candidate_artifact_binding")
        seen.add(path)
        if digest(path.read_bytes()) != row["sha256"]:
            raise ValueError("candidate_artifact_changed:" + str(path))
    return len(seen)
