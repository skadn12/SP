"""Offline audit for the two exposed Planner-schema regression roots.

This checker deliberately does not import the expansion engine and never calls
the provider.  It binds the proposed regression pair to the *same* v98
historical context (history, branch hash, card and latest state), while keeping
the old v98 response/retry artifacts labelled as exposed regression evidence.
It writes only ``REGRESSION_INPUT_BINDING_AUDIT.json`` when run.

The eventual live trial may use this receipt as a gate.  A pair that is
accepted here is a regression reproduction, not a fresh source yield and must
not increase the source inventory or READY count.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any


DELIVERY = Path(__file__).resolve().parents[1]
V98_RUN = DELIVERY.parent / "h4_event_identity_correction_pilot_v98" / "expansion_engine" / "runs" / "four_roots_B_scene01"
V99_ENGINE = DELIVERY / "expansion_engine"
V99_EVIDENCE = DELIVERY / "evidence"
ROOTS = ("cc-train-episode-19053", "cc-train-episode-25652")
MAX_TOKENS = 4096
ROOT_TURNS = {"cc-train-episode-19053": 80, "cc-train-episode-25652": 91}
V98_REQUEST_RE = re.compile(
    r"^(?P<seq>\d+)-(?P<root>cc-train-episode-(?:19053|25652))-B-S01-planner-(?P<attempt>[12])$"
)


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fail(rows: list[dict[str, Any]], root: str, code: str, **detail: Any) -> None:
    rows.append({"root": root, "code": code, **detail})


def decode_context(request: dict[str, Any]) -> tuple[dict[str, Any], str, int]:
    messages = request.get("messages")
    if not isinstance(messages, list) or len(messages) != 1 or not isinstance(messages[0], dict):
        raise ValueError("planner_request_messages_invalid")
    text = messages[0].get("content")
    if not isinstance(text, str):
        raise ValueError("planner_request_content_missing")
    marker = "\nINPUT_JSON\n"
    if text.count(marker) != 1:
        raise ValueError("planner_input_marker_count_invalid")
    suffix = text.split(marker, 1)[1]
    context, end = json.JSONDecoder().raw_decode(suffix)
    if not isinstance(context, dict):
        raise ValueError("planner_input_context_not_object")
    return context, text, len(suffix) - end


def strip_retry_context(context: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(context)
    value.pop("targeted_repair", None)
    return value


def audit_v98_root(root: str, failures: list[dict[str, Any]]) -> dict[str, Any]:
    request_root = V98_RUN / "requests"
    matches = []
    for path in request_root.iterdir():
        m = V98_REQUEST_RE.match(path.name)
        if m and m.group("root") == root:
            matches.append((int(m.group("attempt")), path, m))
    matches.sort(key=lambda item: item[0])
    if [x[0] for x in matches] != [1, 2]:
        fail(failures, root, "v98_planner_attempts_not_exactly_1_and_2",
             observed=[x[0] for x in matches])
        return {"root": root, "status": "FAIL", "attempts": []}

    attempts: list[dict[str, Any]] = []
    context_without_retry: dict[str, Any] | None = None
    frozen_source = read(V98_RUN / "sources" / f"{root}.json")["turns"]
    frozen_state = read(V98_RUN / "roots" / root / "B" / "S01.LATEST_STATE.json")
    frozen_feedback = (V98_RUN / "feedback" / root / "dataset_new_event__planner.txt").read_text(encoding="utf-8")
    for attempt, directory, match in matches:
        try:
            request = read(directory / "REQUEST.json")
            proof = read(directory / "PROOF.json")
            validation = read(directory / "VALIDATION.json")
            response = read(directory / "RESPONSE.json")
            context, text, trailing = decode_context(request)
        except Exception as error:
            fail(failures, root, "v98_request_decode_failed", attempt=attempt,
                 error=f"{type(error).__name__}:{error}")
            continue

        expected_context = {
            "root_id": root,
            "route": "dataset_new_event",
            "role": "planner",
            "target_scene_id": "S01",
        }
        for key, expected in expected_context.items():
            if context.get(key) != expected:
                fail(failures, root, "v98_context_identity_mismatch", attempt=attempt,
                     field=key, actual=context.get(key), expected=expected)
        history = context.get("cumulative_history")
        if not isinstance(history, list) or len(history) != ROOT_TURNS[root]:
            fail(failures, root, "v98_history_length_mismatch", attempt=attempt,
                 actual=len(history) if isinstance(history, list) else None,
                 expected=ROOT_TURNS[root])
        elif len({x.get("turn_ref") for x in history if isinstance(x, dict)}) != len(history):
            fail(failures, root, "v98_history_turn_ref_duplicate", attempt=attempt)
        if history != frozen_source:
            fail(failures, root, "v98_full_frozen_history_mismatch", attempt=attempt)
        if context.get("latest_state") != frozen_state:
            fail(failures, root, "v98_latest_state_artifact_mismatch", attempt=attempt)
        if not frozen_feedback or text.count(frozen_feedback) != 1:
            fail(failures, root, "v98_frozen_feedback_not_present_exactly_once", attempt=attempt)
        if digest(request) != proof.get("payload_sha256"):
            fail(failures, root, "v98_request_payload_hash_mismatch", attempt=attempt)
        cards = context.get("dataset_cards")
        if not isinstance(cards, list) or len(cards) != 1:
            fail(failures, root, "v98_card_binding_not_one", attempt=attempt,
                 count=len(cards) if isinstance(cards, list) else None)
        for field in ("branch_base_sha256", "history_sha256", "latest_state_sha256"):
            if field == "history_sha256":
                actual = digest(history)
            elif field == "branch_base_sha256":
                # v98 proof's branch hash is the frozen prefix; the context
                # carries the same value under branch_base_sha256.
                actual = context.get(field)
            else:
                actual = digest(context.get("latest_state"))
            expected = proof.get(field)
            if expected != actual:
                fail(failures, root, "v98_context_proof_hash_mismatch", attempt=attempt,
                     field=field, actual=actual, expected=expected)
        if proof.get("root_id") != root or proof.get("route") != "dataset_new_event" or proof.get("role") != "planner":
            fail(failures, root, "v98_proof_identity_mismatch", attempt=attempt)
        if proof.get("other_branch_in_generator_context") is not False:
            fail(failures, root, "v98_other_branch_context_not_false", attempt=attempt)
        if request.get("max_tokens") != MAX_TOKENS:
            fail(failures, root, "v98_max_tokens_changed", attempt=attempt,
                 actual=request.get("max_tokens"), expected=MAX_TOKENS)
        if validation.get("status") != "REJECT":
            fail(failures, root, "v98_exposed_failure_status_changed", attempt=attempt,
                 actual=validation.get("status"))
        errors = validation.get("errors", [])
        if not any(isinstance(error, dict) and error.get("path") == "scene.event_identity"
                   for error in errors):
            fail(failures, root, "v98_exposed_event_identity_error_missing", attempt=attempt)
        if response.get("text") is None:
            fail(failures, root, "v98_response_text_missing", attempt=attempt)
        if attempt == 1 and "targeted_repair" in context:
            fail(failures, root, "v98_initial_context_has_retry_repair", attempt=attempt)
        if attempt == 2 and not isinstance(context.get("targeted_repair"), dict):
            fail(failures, root, "v98_retry_context_missing_targeted_repair", attempt=attempt)
        common = strip_retry_context(context)
        if context_without_retry is None:
            context_without_retry = common
        elif context_without_retry != common:
            fail(failures, root, "v98_retry_context_changed_outside_targeted_repair", attempt=attempt)
        attempts.append({
            "attempt": attempt,
            "request_dir": str(directory),
            "request_sha256": digest(request),
            "payload_sha256_from_proof": proof.get("payload_sha256"),
            "context_sha256": digest(context),
            "context_without_targeted_repair_sha256": digest(common),
            "history_sha256": proof.get("history_sha256"),
            "branch_base_sha256": proof.get("branch_base_sha256"),
            "latest_state_sha256": proof.get("latest_state_sha256"),
            "card_ids": [x.get("card_id") for x in cards] if isinstance(cards, list) else [],
            "trailing_chars_after_input_json": trailing,
            "validation_status": validation.get("status"),
            "validation_error_paths": [x.get("path") for x in errors if isinstance(x, dict)],
            "targeted_repair_present": "targeted_repair" in context,
            "exact_full_history_bound": history == frozen_source,
            "exact_latest_state_artifact_bound": context.get("latest_state") == frozen_state,
            "exact_frozen_feedback_once": bool(frozen_feedback) and text.count(frozen_feedback) == 1,
        })
    status = "PASS" if len(attempts) == 2 and not any(x["root"] == root for x in failures) else "FAIL"
    return {"root": root, "status": status, "attempts": attempts,
            "paired_context_sha256": digest(context_without_retry) if context_without_retry else None}


def audit_v98_seal(failures: list[dict[str, Any]]) -> dict[str, Any]:
    manifest_path = V98_RUN.parents[2] / "COMPLETE_MANIFEST.json"
    manifest = read(manifest_path)
    if manifest.get("status") not in {"PASS", "SEALED"}:
        fail(failures, "both", "v98_seal_status_not_pass", actual=manifest.get("status"))
    by_path = {row["path"]: row for row in manifest["files"]}
    wanted: set[Path] = set()
    for root in ROOTS:
        wanted.update({V98_RUN / "sources" / f"{root}.json",
                       V98_RUN / "roots" / root / "B" / "S01.LATEST_STATE.json",
                       V98_RUN / "feedback" / root / "dataset_new_event__planner.txt"})
        for path in (V98_RUN / "requests").glob(f"*-{root}-B-S01-planner-*"):
            wanted.update(path / name for name in ("REQUEST.json", "PROOF.json", "VALIDATION.json", "RESPONSE.json"))
    rows = []
    for path in sorted(wanted):
        bound = by_path.get(str(path))
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        okay = isinstance(bound, dict) and bound.get("sha256") == actual
        if not okay:
            fail(failures, "both", "v98_sealed_input_hash_mismatch", path=str(path),
                 expected=bound.get("sha256") if isinstance(bound, dict) else None, actual=actual)
        rows.append({"path": str(path), "sha256": actual, "sealed_match": okay})
    return {"manifest_path": str(manifest_path),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "bound_regression_file_count": len(rows), "files": rows}


def selection_status() -> dict[str, Any]:
    path = V99_ENGINE / "SOURCE_SELECTION.json"
    if not path.exists():
        return {"status": "PENDING", "reason": "v99_source_selection_missing", "path": str(path)}
    selection = read(path)
    observed = selection.get("roots")
    expected = list(ROOTS)
    # This is intentionally a gate, not an automatic rewrite.  v99 currently
    # carries the inherited four-root selection until the pair test installs a
    # separate regression binding.
    return {"status": "PASS" if observed == expected else "PENDING",
            "reason": None if observed == expected else "selection_is_not_pair_only",
            "path": str(path), "observed_roots": observed, "expected_roots": expected,
            "selection_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def result_status() -> dict[str, Any]:
    path = V99_EVIDENCE / "REGRESSION_SCHEMA_TEST_RESULTS_FINAL.json"
    if not path.exists():
        return {"status": "PENDING", "reason": "offline_schema_test_result_missing", "path": str(path)}
    value = read(path)
    if not isinstance(value, dict):
        return {"status": "FAIL", "reason": "offline_schema_test_result_not_object", "path": str(path)}
    # Do not trust a test's PASS flag alone.  Preserve the result binding and
    # expose its expected no-live/no-yield claims for a later human audit.
    return {"status": "PRESENT", "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "top_level_keys": sorted(value), "declared_live": value.get("live"),
            "declared_http_calls": value.get("http_calls"), "declared_new_scenes": value.get("new_scenes"),
            "declared_fresh_yield": value.get("fresh_yield")}


def audit() -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    seal = audit_v98_seal(failures)
    roots = [audit_v98_root(root, failures) for root in ROOTS]
    selection = selection_status()
    result = result_status()
    if selection["status"] != "PASS":
        # A pair-specific offline test can proceed without replacing the
        # inherited runner selection, so this remains a pending live gate.
        pass
    live_budget = {
        "offline_preview_logical_calls": 0,
        "offline_preview_http_attempts": 0,
        "actual_paired_regression_max_logical_calls": 16,
        "actual_paired_regression_max_http_attempts": 32,
        "actual_paired_regression_calculation": {
            "roots": 2, "arms_per_root": 2,
            "planner_attempts_per_cell": 2, "event_review_attempts_per_cell": 2,
            "transport_retries": 1,
            "per_cell_logical_upper_bound": 4,
            "total_logical_upper_bound": 16,
            "http_multiplier": 2,
        },
        "hard_stop": [
            "offline script performs no provider request",
            "if planner attempt 2 is rejected, stop that root and do not call realizer",
            "first structurally valid nonempty Planner response receives bounded event review; then cell stops",
            "no question, solver, selected, training or source-promotion calls",
            "regression outputs never increment fresh source yield or READY inventory",
        ],
    }
    return {
        "schema_version": "h4.v99.paired_regression_input_binding_audit.v1",
        "status": "PASS" if not failures else "FAIL",
        "live_gate": "DEFER_TO_REGRESSION_PAIRED_PREFLIGHT_AUDIT" if not failures else "BLOCKED",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "v98_run": str(V98_RUN),
        "v99_delivery": str(DELIVERY),
        "pair_roots": list(ROOTS),
        "classification": "exposed_prior_planner_failure_regression; never fresh yield",
        "v98_seal_binding": seal,
        "frozen_baseline": roots,
        "selection_gate": selection,
        "selection_scope_note": "Inherited production selection is diagnostic only; actual exposed pair is frozen separately in regression/REGRESSION_MANIFEST.json, never production prepare.",
        "offline_schema_test": result,
        "budget_and_stop": live_budget,
        "failures": failures,
        "semantic_certified": False,
        "independent_scope": "v98 request naming, context hashes, card/state/history binding and bounded call policy; no model semantic judgment",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=V99_EVIDENCE / "REGRESSION_INPUT_BINDING_AUDIT.json")
    args = parser.parse_args()
    output = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "live_gate": output["live_gate"],
                      "failures": len(output["failures"]), "output": str(args.output)}, ensure_ascii=False))
    raise SystemExit(0 if output["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
