"""Validate v116 baseline reviews and seal the fixed two-reader selection.

This module checks artifact bindings and declared decisions. It does not judge
whether a reader's semantic interpretation is correct, and never calls a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path


CHECKS = (
    "identity_coreference", "gender_relationship_titles", "fact_state_time",
    "unsupported_reconciliation",
)
RANK = {"PASS": 0, "WARNING": 1, "HOLD": 2}
CLASSIFICATIONS = {
    "HARD_CONTRADICTION": "HOLD",
    "UNRESOLVED_IDENTITY_OR_TITLE": "HOLD",
    "UNRESOLVED_CORE_EVENT": "HOLD",
    "UNRESOLVED_STATE_TIME": "WARNING",
    "ATTRIBUTED_UNVERIFIED_REPORT": "WARNING",
    "UNDERSPECIFIED_ATTRIBUTE": "WARNING",
    "COMPATIBLE_TIME_CHANGE": "PASS",
    "COMPATIBLE_NONEXCLUSIVE": "PASS",
}
REVIEW_FIELDS = {
    "schema_version", "policy_version", "root_id", "baseline_path",
    "baseline_sha256", "plan_sha256", "policy_sha256", "reviewer",
    "started_at_utc", "finished_at_utc", "full_source_read",
    "covered_turn_refs", "checks", "findings", "continuation_constraints",
    "decision",
}
FINDING_FIELDS = {
    "id", "category", "classification", "disposition", "reason", "citations",
    "temporal_evidence",
}


class ReviewValidationError(ValueError):
    """An invalid artifact blocks selection rather than disappearing from it."""


def _require(condition, message):
    if not condition:
        raise ReviewValidationError(message)


def _text(value, name):
    _require(isinstance(value, str) and bool(value.strip()), f"{name}: nonempty string required")
    return value


def _keys(value, expected, name):
    _require(isinstance(value, dict) and set(value) == set(expected), f"{name}: field set mismatch")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read(path):
    path = Path(path)
    try:
        payload = path.read_bytes()
        value = json.loads(payload, object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReviewValidationError(f"cannot read JSON artifact {path}: {error}") from error
    _require(isinstance(value, dict), f"{path}: JSON object required")
    return value, hashlib.sha256(payload).hexdigest()


def _sha(value, name):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             f"{name}: SHA256 required")
    return value


def _file_binding(path, expected, name):
    _text(path, f"{name}.path")
    _require(Path(path).is_absolute(), f"{name}: absolute path required")
    _sha(expected, f"{name}.sha256")
    try:
        actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as error:
        raise ReviewValidationError(f"{name}: unavailable bound file {path}") from error
    _require(actual == expected, f"{name}: SHA256 mismatch")


def _timestamp(value, name):
    _text(value, name)
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ReviewValidationError(f"{name}: invalid timestamp") from error
    _require(result.tzinfo is not None and result.utcoffset() == timedelta(0),
             f"{name}: UTC timezone required")
    return result


def _load_plan(plan_path):
    plan, plan_sha = _read(plan_path)
    _require(plan.get("schema_version") == "codex_baseline_screen_plan_v1", "invalid plan schema")
    _require(plan.get("candidate_denominator") == 8, "fixed candidate denominator must be eight")
    _require(plan.get("selection_max") == 4, "fixed selection maximum must be four")
    _require(plan.get("reviews_per_candidate") == 2, "exactly two readers required")
    _require(plan.get("status") == "FROZEN_BEFORE_THIS_BATCH_FULL_TEXT_SEMANTIC_READS",
             "plan was not declared frozen before semantic reads")
    created = _timestamp(plan.get("created_at_utc"), "plan.created_at_utc")
    _require(created <= datetime.now(timezone.utc), "plan timestamp is in the future")
    _file_binding(plan.get("policy_path"), plan.get("policy_sha256"), "policy")
    candidates = plan.get("candidates")
    _require(isinstance(candidates, list) and len(candidates) == 8, "exactly eight candidate records required")
    _require(all(isinstance(row, dict) for row in candidates), "invalid candidate record")
    order = [row.get("root_id") for row in candidates]
    _require(all(isinstance(root, str) and re.fullmatch(r"cc-train-episode-\d+", root) for root in order),
             "invalid candidate root identity")
    _require(len(set(order)) == 8 and plan.get("candidate_order") == order, "candidate order mismatch or duplicate")
    _require(order == sorted(order, key=lambda root: int(root.rsplit("-", 1)[-1])), "candidate order is not numeric ascending")
    excluded = plan.get("excluded_previous_full_validation_roots", [])
    _require(isinstance(excluded, list) and not (set(order) & set(excluded)), "candidate overlaps excluded roots")
    for row in candidates:
        _file_binding(row.get("baseline_path"), row.get("baseline_sha256"), f"baseline {row['root_id']}")
    bindings = plan.get("bindings")
    _require(isinstance(bindings, list), "plan bindings must be a list")
    for binding in bindings:
        _require(isinstance(binding, dict), "invalid plan binding")
        _file_binding(binding.get("path"), binding.get("sha256"), "plan input binding")
    return {"plan": plan, "sha256": plan_sha, "path": str(Path(plan_path).resolve()), "created": created}


def _worst(statuses):
    return max(statuses, key=RANK.__getitem__, default="PASS")


def _citation(citation, turns, name):
    _keys(citation, {"turn_ref", "speaker", "quote"}, name)
    ref = _text(citation["turn_ref"], f"{name}.turn_ref")
    _require(ref in turns, f"{name}: unknown turn_ref")
    _require(citation["speaker"] == turns[ref]["speaker"], f"{name}: speaker mismatch")
    quote = _text(citation["quote"], f"{name}.quote")
    _require(quote in turns[ref]["text"], f"{name}: quote is not an exact source substring")


def _validate_review(context, review_path):
    review, review_sha = _read(review_path)
    _keys(review, REVIEW_FIELDS, "review")
    plan = context["plan"]
    _require(review["schema_version"] == "codex_baseline_review_v1", "invalid review schema")
    _require(review["policy_version"] == "v116", "invalid review policy version")
    _require(review["plan_sha256"] == context["sha256"], "review plan binding mismatch")
    _require(review["policy_sha256"] == plan["policy_sha256"], "review policy binding mismatch")
    candidates = {row["root_id"]: row for row in plan["candidates"]}
    _require(isinstance(review["root_id"], str) and review["root_id"] in candidates,
             "review root is outside the frozen candidates")
    candidate = candidates[review["root_id"]]
    _require(review["baseline_path"] == candidate["baseline_path"], "review baseline path mismatch")
    _require(review["baseline_sha256"] == candidate["baseline_sha256"], "review baseline SHA256 mismatch")
    source, source_sha = _read(candidate["baseline_path"])
    _require(source_sha == candidate["baseline_sha256"], "baseline changed while validating")
    _require(source.get("root_id") == review["root_id"], "baseline root mismatch")
    rows = source.get("turns")
    _require(isinstance(rows, list) and bool(rows) and all(isinstance(row, dict) for row in rows),
             "baseline must contain turns")
    refs = []
    for row in rows:
        refs.append(_text(row.get("turn_ref"), "source.turn_ref"))
        _text(row.get("speaker"), "source.speaker")
        _text(row.get("text"), "source.text")
    _require(len(refs) == len(set(refs)), "duplicate source turn_ref")
    _require(candidate.get("baseline_turn_count") == len(rows), "baseline turn count mismatch")
    _require(candidate.get("baseline_speakers") == sorted({row["speaker"] for row in rows})
             and len(candidate["baseline_speakers"]) == 2, "baseline speaker set mismatch")
    scenes = {row.get("scene_ref") for row in rows if isinstance(row.get("scene_ref"), str)
              and row["scene_ref"].startswith("R01:")}
    _require(scenes == {"R01:S01", "R01:S02", "R01:S03"}, "baseline must contain exactly three R01 scenes")
    _require(review["full_source_read"] is True, "full source reading must be declared")
    _require(review["covered_turn_refs"] == refs, "full source coverage must preserve exact original order")
    reviewer = _text(review["reviewer"], "reviewer")
    _require(reviewer == reviewer.strip(), "reviewer identity has surrounding whitespace")
    started = _timestamp(review["started_at_utc"], "started_at_utc")
    finished = _timestamp(review["finished_at_utc"], "finished_at_utc")
    _require(context["created"] <= started <= finished <= datetime.now(timezone.utc),
             "review time order must be plan <= start <= finish <= now")
    _keys(review["checks"], CHECKS, "checks")
    for category, check in review["checks"].items():
        _keys(check, {"complete", "status", "reason"}, f"check {category}")
        _require(check["complete"] is True, f"check {category}: incomplete")
        _require(isinstance(check["status"], str) and check["status"] in RANK,
                 f"check {category}: invalid status")
        _text(check["reason"], f"check {category}.reason")
    findings = review["findings"]
    _require(isinstance(findings, list), "findings must be a list")
    ids = set()
    by_category = {category: [] for category in CHECKS}
    turns = dict(zip(refs, rows))
    for finding in findings:
        _keys(finding, FINDING_FIELDS, "finding")
        finding_id = _text(finding["id"], "finding.id")
        _require(finding_id not in ids, "duplicate finding id")
        ids.add(finding_id)
        category = finding["category"]
        _require(isinstance(category, str) and category in CHECKS, "invalid finding category")
        classification = finding["classification"]
        _require(isinstance(classification, str) and classification in CLASSIFICATIONS, "invalid finding classification")
        disposition = CLASSIFICATIONS[classification]
        _require(finding["disposition"] == disposition, "finding classification/disposition mismatch")
        _text(finding["reason"], "finding.reason")
        citations = finding["citations"]
        _require(isinstance(citations, list) and bool(citations), "every finding needs source citations")
        for citation in citations:
            _citation(citation, turns, "finding citation")
        if disposition == "HOLD":
            _require(len({citation["turn_ref"] for citation in citations}) >= 2,
                     "HOLD finding needs at least two distinct source turns")
        temporal = finding["temporal_evidence"]
        _require(isinstance(temporal, list), "temporal_evidence must be a list")
        for citation in temporal:
            _citation(citation, turns, "temporal evidence")
        if classification == "COMPATIBLE_TIME_CHANGE":
            _require(bool(temporal), "compatible time change needs explicit temporal evidence")
        by_category[category].append(disposition)
    for category in CHECKS:
        _require(review["checks"][category]["status"] == _worst(by_category[category]),
                 f"check {category}: status does not aggregate its findings")
    expected = _worst(check["status"] for check in review["checks"].values())
    _require(review["decision"] == expected, "review decision does not aggregate checks")
    constraints = review["continuation_constraints"]
    _require(isinstance(constraints, list), "continuation_constraints must be a string list")
    for constraint in constraints:
        _text(constraint, "continuation constraint")
    _require(expected == "PASS" or bool(constraints), "non-PASS review requires continuation constraints")
    return review, review_sha


def validate_review(plan_path, review_path):
    """Return the unchanged validated review, or raise ReviewValidationError."""
    return _validate_review(_load_plan(plan_path), review_path)[0]


def _encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _write_pair(paths_and_payloads):
    """Reserve both names exclusively before writing; never replace a file."""
    opened = []
    try:
        for path, payload in paths_and_payloads:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            opened.append((path, descriptor, payload))
    except OSError:
        for path, descriptor, _ in opened:
            os.close(descriptor)
            path.unlink()
        raise
    pending = list(opened)
    try:
        while pending:
            path, descriptor, payload = pending.pop(0)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
    except BaseException:
        for _, descriptor, _ in pending:
            os.close(descriptor)
        for path, _, _ in opened:
            path.unlink(missing_ok=True)
        raise


def finalize_screen(plan_path, review_paths, output_dir):
    """Require all sixteen valid reviews and save the first eligible four roots.

    A valid HOLD is retained and excludes its root. Any invalid, duplicate or
    missing review blocks the entire finalization before outputs are created.
    """
    context = _load_plan(plan_path)
    plan = context["plan"]
    review_paths = [Path(path).resolve() for path in review_paths]
    _require(len(review_paths) == 16, "all eight candidates need exactly two reviews")
    _require(len(set(review_paths)) == 16, "duplicate review path")
    grouped = {root: [] for root in plan["candidate_order"]}
    identities = set()
    for path in review_paths:
        review, sha = _validate_review(context, path)
        identity = (review["root_id"], review["reviewer"])
        _require(identity not in identities, "duplicate reviewer for root; exactly one review per reader")
        identities.add(identity)
        grouped[review["root_id"]].append((path, sha, review))
    _require(all(len(rows) == 2 for rows in grouped.values()), "missing or excess reviews for candidate root")
    output_dir = Path(output_dir).resolve()
    results_path = output_dir / "SCREEN_RESULTS.json"
    selected_path = output_dir / "SELECTED_ROOTS.json"
    _require(not results_path.exists() and not selected_path.exists(), "screen outputs already exist; no overwrite")
    roots = []
    selected = []
    for candidate in plan["candidates"]:
        rows = sorted(grouped[candidate["root_id"]], key=lambda row: row[2]["reviewer"])
        decision = _worst(row[2]["decision"] for row in rows)
        eligible = decision != "HOLD"
        is_selected = eligible and len(selected) < plan["selection_max"]
        constraints = list(dict.fromkeys(constraint for _, _, review in rows
                                         for constraint in review["continuation_constraints"]))
        entry = {
            "root_id": candidate["root_id"], "baseline_path": candidate["baseline_path"],
            "baseline_sha256": candidate["baseline_sha256"], "decision": decision,
            "eligible": eligible, "selected": is_selected, "continuation_constraints": constraints,
            "reviews": [{"path": str(path), "sha256": sha, "reviewer": review["reviewer"],
                         "decision": review["decision"],
                         "continuation_constraints": review["continuation_constraints"]}
                        for path, sha, review in rows],
        }
        roots.append(entry)
        if is_selected:
            selected.append(entry)
    common = {
        "batch_id": plan["batch_id"], "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_path": context["path"], "plan_sha256": context["sha256"],
        "policy_path": plan["policy_path"], "policy_sha256": plan["policy_sha256"],
        "candidate_denominator": 8, "reviews_per_candidate": 2, "validated_review_count": 16,
        "selection_max": 4, "selected_root_count": len(selected),
        "semantic_correctness_certified_by_program": False, "external_semantic_certification": False,
        "generation_started": False, "solver_started": False,
        "training_started": False, "pool_merge_started": False,
    }
    results = {"schema_version": "codex_baseline_screen_results_v1", **common,
               "eligible_root_count": sum(root["eligible"] for root in roots), "roots": roots}
    results_payload = _encoded(results)
    selection = {"schema_version": "codex_baseline_selected_roots_v1", **common,
                 "screen_results_path": str(results_path),
                 "screen_results_sha256": hashlib.sha256(results_payload).hexdigest(), "roots": selected}
    # Recheck frozen inputs immediately before persisting derived decisions.
    fresh = _load_plan(plan_path)
    _require(fresh["sha256"] == context["sha256"], "plan changed during finalization")
    for path in review_paths:
        _, actual_sha = _read(path)
        _require(any(actual_sha == sha and path == saved_path for rows in grouped.values()
                     for saved_path, sha, _ in rows), "review changed during finalization")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_pair([(results_path, results_payload), (selected_path, _encoded(selection))])
    return results, selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--plan", type=Path, required=True)
    validate.add_argument("--review", type=Path, required=True)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--plan", type=Path, required=True)
    finalize.add_argument("--reviews", type=Path, nargs="+", required=True)
    finalize.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "validate":
        review = validate_review(args.plan, args.review)
        print(json.dumps({"root_id": review["root_id"], "reviewer": review["reviewer"],
                          "decision": review["decision"], "mechanical_validation": "PASS"}))
    else:
        results, _ = finalize_screen(args.plan, args.reviews, args.output_dir)
        print(json.dumps({"selected_root_count": results["selected_root_count"],
                          "validated_review_count": results["validated_review_count"]}))


if __name__ == "__main__":
    main()
