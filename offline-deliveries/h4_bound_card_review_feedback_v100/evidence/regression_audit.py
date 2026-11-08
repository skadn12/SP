"""Independent stdlib audit for the exposed v100 review/replan regression.

Checks raw artifacts, source projections, frozen inputs and bounded endpoints.
Does not import engine code, call a model, or certify semantic judgments.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parents[1]
RUN = BASE / "regression"
OLD = BASE.parent / "h4_planner_schema_regression_v99"
OLD_RUN = OLD / "regression"
LEGACY = BASE.parent / "h4_r02_dualroute_feedback_v1"
ROOTS = ["cc-train-episode-19053", "cc-train-episode-25652"]
ORIGIN_PLAN_PREFIX = {ROOTS[0]: "003", ROOTS[1]: "006"}
STAGES = ("initial_event_review", "semantic_replan", "post_replan_event_review")
PROJECTION_FIELDS = ("card_id", "card_sha256", "guidance_level", "actors", "steps", "dependencies")
LEAF_FIELDS = {"actors": ("role",), "steps": ("action", "state"), "dependencies": ("relation",)}
NESTED_FIELDS = {"actors": ("actor_id", "role"),
                 "steps": ("step_id", "actor_id", "action", "state"),
                 "dependencies": ("before", "after", "relation")}


def read(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value: Any, *, ascii_only: bool = False) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=ascii_only, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def check(condition: bool, code: str, failures: list[dict[str, Any]], **detail: Any) -> None:
    if not condition:
        failures.append({"code": code, **detail})


def request_context(payload: dict[str, Any], *, event: bool) -> dict[str, Any]:
    marker = "\nEVENT_STATE_INPUT_JSON\n" if event else "\nINPUT_JSON\n"
    messages = payload["messages"]
    assert len(messages) == 1 and messages[0]["role"] == "user", "request_message_shape_invalid"
    text = messages[0]["content"]
    assert text.count(marker) == 1, "request_context_marker_count_invalid"
    value = json.JSONDecoder().raw_decode(text.split(marker, 1)[1])[0]
    assert isinstance(value, dict), "request_context_not_object"
    return value


def origin_files(root: str) -> tuple[Path, Path]:
    short = root.rsplit("-", 1)[1]
    prepared = OLD_RUN / "prepared" / f"{short}_treatment.json"
    found = list((OLD_RUN / "requests").glob(ORIGIN_PLAN_PREFIX[root] + "-*/CANDIDATE_PLAN.json"))
    assert len(found) == 1, "origin_candidate_plan_not_unique"
    return prepared, found[0]


def card_contract(card: dict[str, Any]) -> dict[str, Any]:
    projection = {field: card.get(field) for field in PROJECTION_FIELDS}
    for collection, fields in NESTED_FIELDS.items():
        rows = card.get(collection)
        projection[collection] = [{field: row[field] for field in fields if field in row}
                                  for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    leaves = []
    for name, fields in LEAF_FIELDS.items():
        rows = projection.get(name)
        for index, row in enumerate(rows if isinstance(rows, list) else []):
            if not isinstance(row, dict):
                continue
            for field in fields:
                value = row.get(field)
                if isinstance(value, str) and value.strip():
                    leaves.append({"source_path": f"/{name}/{index}/{field}", "value": value})
    return {
        "source_projection": projection,
        "source_binding": {"card_id": projection["card_id"], "card_sha256": projection["card_sha256"],
                           "projection_sha256": digest(projection, ascii_only=True)},
        "source_leaves": leaves, "allowed_source_paths": [row["source_path"] for row in leaves],
    }


def audit_event_context(ctx: dict[str, Any], original: dict[str, Any], plan: dict[str, Any],
                        failures: list[dict[str, Any]], **label: Any) -> None:
    check(ctx.get("root_id") == original["root_id"] and ctx.get("route") == original["route"]
          and ctx.get("scene_id") == "S01", "event_identity_changed", failures, **label)
    check(ctx.get("cumulative_history") == original["cumulative_history"],
          "event_full_history_changed", failures, **label)
    check(ctx.get("latest_state") == original["latest_state"], "event_state_changed", failures, **label)
    check(ctx.get("scene_plan") == plan["scenes"][0], "event_plan_not_expected_candidate", failures, **label)
    expected = [card_contract(card) for card in original["dataset_cards"]]
    observed = ctx.get("card_source_contracts")
    check(isinstance(observed, list) and len(observed) == len(expected),
          "bound_contract_count_invalid", failures, **label)
    if isinstance(observed, list):
        for index, contract in enumerate(expected):
            actual = observed[index] if index < len(observed) else {}
            for field, value in contract.items():
                check(isinstance(actual, dict) and actual.get(field) == value,
                      "bound_card_source_contract_changed", failures, index=index, field=field, **label)


def original_context(root: str) -> dict[str, Any]:
    prepared, _ = origin_files(root)
    return request_context(read(prepared)["payload"], event=False)


def audit_review_output(value: dict[str, Any], original: dict[str, Any],
                        failures: list[dict[str, Any]], **label: Any) -> None:
    cards = {card["card_id"]: card for card in original["dataset_cards"]}
    observed_cards = value.get("card_checks", [])
    check(isinstance(observed_cards, list) and
          [row.get("card_id") for row in observed_cards if isinstance(row, dict)] == list(cards),
          "validated_review_card_coverage_changed", failures, **label)
    for row in observed_cards if isinstance(observed_cards, list) else []:
        card = cards.get(row.get("card_id")) if isinstance(row, dict) else None
        if card is None:
            continue
        expected = card_contract(card)
        comparison = row.get("function_check", {}).get("comparison", {})
        check(comparison.get("source_binding") == expected["source_binding"],
              "validated_review_source_binding_changed", failures, **label)
        for dimension in ("purpose", "start_state", "end_state", "request_acceptance", "experience_state"):
            pair = comparison.get(dimension)
            paths = pair.get("source_paths") if isinstance(pair, dict) else None
            check(isinstance(paths, list) and len(paths) == len(set(paths))
                  and all(isinstance(path, str) and path in expected["allowed_source_paths"] for path in paths),
                  "validated_review_source_paths_invalid", failures, dimension=dimension, **label)
            if dimension in {"purpose", "start_state", "end_state"}:
                check(isinstance(pair, dict) and set(pair) == {"source_paths", "actual"}
                      and isinstance(pair.get("actual"), str) and bool(pair["actual"].strip()),
                      "validated_review_has_free_source_text_or_invalid_actual", failures,
                      dimension=dimension, **label)


def frozen_audit(manifest: dict[str, Any], failures: list[dict[str, Any]], *, live: bool) -> dict[str, Any]:
    expected = {"version": "v100_bound_card_review_feedback_regression_v1",
                "max_initial_review_attempts_per_root": 2, "max_semantic_replan_rounds_per_root": 1,
                "max_structural_planner_attempts_per_semantic_round": 2,
                "max_post_replan_review_attempts_per_root": 2,
                "max_logical_requests": 12, "http_max": 24, "max_tokens": 4096,
                "fresh_source_count": 0, "new_source_reservations": 0, "training_allowed": False}
    for key, value in expected.items():
        check(manifest.get(key) == value, "manifest_value_changed", failures,
              field=key, expected=value, actual=manifest.get(key))
    cells = manifest.get("cells", [])
    check([row.get("root_id") for row in cells] == ROOTS, "regression_roots_changed", failures)
    files = manifest.get("files", [])
    bindings = {row["path"]: row["sha256"] for row in files}
    check(len(bindings) == len(files), "duplicate_manifest_path", failures)
    changed = [{"path": path, "expected": value, "actual": sha(path) if Path(path).is_file() else None}
               for path, value in bindings.items() if not Path(path).is_file() or sha(path) != value]
    check(not changed, "manifest_frozen_file_changed", failures, files=changed)
    required = {BASE / "evidence/regression_driver.py", BASE / "PRECALL_PLAN_ZH.md",
                BASE / "PRESERVATION_BEFORE.json", RUN / "provider_profile.json",
                RUN / "CAPACITY_PREFLIGHT.json"}
    required.update((BASE / "expansion_engine").glob("*.py"))
    required.update(LEGACY.glob("*.py"))
    for root in ROOTS:
        required.update(origin_files(root))
        required.add(RUN / "sources" / f"{root}.json")
    missing = sorted(str(path) for path in required if str(path.resolve()) not in bindings)
    check(not missing, "required_dependency_not_frozen", failures, paths=missing)

    protection = read(BASE / "PRESERVATION_BEFORE.json")
    protected_changed = []
    for row in protection["files"]:
        if not Path(row["path"]).is_file() or sha(row["path"]) != row["sha256"]:
            protected_changed.append(row["path"])
    check(not protected_changed, "historical_protected_file_changed", failures, paths=protected_changed)

    qa_files = sorted((BASE / "qa_engine").glob("*.py"))
    qa_changed = [path.name for path in qa_files if not (OLD / "qa_engine" / path.name).is_file()
                  or sha(path) != sha(OLD / "qa_engine" / path.name)]
    check(len(qa_files) == 15 and not qa_changed, "qa_engine_changed_outside_scope", failures,
          count=len(qa_files), paths=qa_changed)
    check(sha(BASE / "expansion_engine/realization_review.py") == sha(OLD / "expansion_engine/realization_review.py"),
          "scene_review_engine_changed_outside_scope", failures)

    prepared_summary = []
    for cell in cells:
        root = cell["root_id"]
        for key in ("input", "prepared_request"):
            row = cell[key]
            check(sha(row["path"]) == row["sha256"] == bindings.get(str(Path(row["path"]).resolve())),
                  "cell_input_binding_mismatch", failures, root=root, field=key)
        data = read(cell["input"]["path"])
        original_prepared, original_plan = origin_files(root)
        original = original_context(root)
        check(data.get("context") == original and data.get("initial_plan") == read(original_plan),
              "historical_regression_input_changed", failures, root=root)
        check(data.get("exposed_regression_only") is True, "input_exposure_label_missing", failures, root=root)
        origins = {row["path"]: row["sha256"] for row in data.get("provenance", [])}
        check(origins == {str(original_prepared): sha(original_prepared), str(original_plan): sha(original_plan)},
              "historical_input_provenance_incomplete", failures, root=root)
        check(read(RUN / "sources" / f"{root}.json")["turns"] == original["cumulative_history"],
              "frozen_regression_source_history_changed", failures, root=root)
        for old_feedback in (OLD_RUN / "feedback").rglob("*"):
            if old_feedback.is_file():
                current = RUN / "feedback" / old_feedback.relative_to(OLD_RUN / "feedback")
                check(current.is_file() and sha(current) == sha(old_feedback),
                      "frozen_feedback_copy_changed", failures, root=root, path=str(current))
        payload = read(cell["prepared_request"]["path"])["payload"]
        ctx = request_context(payload, event=True)
        audit_event_context(ctx, original, data["initial_plan"], failures, root=root, stage="prepared")
        check("targeted_repair" not in ctx and payload.get("max_tokens") == 4096,
              "prepared_initial_request_is_retry_or_budget_changed", failures, root=root)
        prepared_summary.append({"root_id": root, "context_sha256": digest(original),
                                 "initial_plan_sha256": digest(data["initial_plan"]),
                                 "payload_sha256": digest(payload), "history_turns": len(original["cumulative_history"]),
                                 "prepared_file_sha256": cell["prepared_request"]["sha256"]})
    capacity = read(RUN / "CAPACITY_PREFLIGHT.json")
    check(capacity.get("status") == "PASS" and capacity.get("network_calls") == 0,
          "capacity_preflight_not_offline_pass", failures)
    check([row.get("root_id") for row in capacity.get("cells", [])] == ROOTS,
          "capacity_preflight_roots_incomplete", failures)
    for row in capacity.get("cells", []):
        cell = next((value for value in cells if value["root_id"] == row.get("root_id")), {})
        check(row.get("capacity", {}).get("status") == "PASS"
              and row.get("capacity", {}).get("truncation") is False
              and row.get("prepared_request_sha256") == cell.get("prepared_request", {}).get("sha256"),
              "capacity_prepared_input_binding_failed", failures, root=row.get("root_id"))
    forbidden = [str(RUN / name) for name in ("SOURCE_RESERVATIONS.json", "UNIQUE_EVENT_ALLOCATION.json",
                  "dataset_cards_result.json", "shards", "source_reviews", "materialized_sources") if (RUN / name).exists()]
    check(not forbidden, "regression_contains_fresh_source_artifacts", failures, paths=forbidden)
    if not live:
        check(not (RUN / "LIVE_STARTED.json").exists() and not (RUN / "requests").exists(),
              "preflight_already_has_live_artifacts", failures)
    return {"frozen_file_count": len(files), "historical_protected_file_count": len(protection["files"]),
            "unchanged_qa_files": len(qa_files), "prepared_cases": prepared_summary}


def review_artifact(root: str, stage: str) -> dict[str, Any] | None:
    path = RUN / "roots" / root / (stage + ".json")
    return read(path) if path.exists() else None


def live_audit(manifest: dict[str, Any], failures: list[dict[str, Any]]) -> dict[str, Any]:
    requests = sorted((RUN / "requests").glob("*"))
    check(len(requests) <= 12, "logical_request_budget_exceeded", failures, actual=len(requests))
    counts: dict[str, Counter] = defaultdict(Counter)
    role_counts = Counter()
    stage_order: dict[str, list[str]] = defaultdict(list)
    prior_common: dict[tuple[str, str], dict[str, Any]] = {}
    actual_candidate: dict[str, dict[str, Any]] = {}
    initial_diagnostics: dict[str, list[dict[str, Any]]] = {}
    previous_planner_raw: dict[str, str] = {}
    stage_requests: dict[tuple[str, str], list[Path]] = defaultdict(list)
    http_numbers, rows, token_rows = [], [], []
    input_by_root = {cell["root_id"]: read(cell["input"]["path"]) for cell in manifest["cells"]}
    for number, directory in enumerate(requests, 1):
        check(directory.name.startswith(f"{number:03d}-"), "logical_request_sequence_invalid", failures,
              request=directory.name)
        payload, proof = read(directory / "REQUEST.json"), read(directory / "PROOF.json")
        root, stage, role = proof.get("regression_root"), proof.get("regression_stage"), proof.get("role")
        check(root in ROOTS and stage in STAGES, "request_regression_identity_invalid", failures,
              request=directory.name, root=root, stage=stage)
        check(proof.get("root_id") == root and proof.get("route") == "dataset_new_event"
              and proof.get("regression_only") is True and proof.get("fresh_source") is False,
              "request_proof_exposure_binding_invalid", failures, request=directory.name)
        check(proof.get("payload_sha256") == digest(payload) and payload.get("max_tokens") == 4096,
              "request_payload_hash_or_budget_changed", failures, request=directory.name)
        check(role == ("planner" if stage == "semantic_replan" else "event_state_extractor"),
              "unexpected_role_or_endpoint_called", failures, request=directory.name, role=role)
        counts[root][stage] += 1; role_counts[role] += 1; stage_order[root].append(stage)
        stage_requests[(root, stage)].append(directory)
        original = input_by_root[root]["context"]
        ctx = request_context(payload, event=role == "event_state_extractor")
        if role == "event_state_extractor":
            expected_plan = input_by_root[root]["initial_plan"] if stage == "initial_event_review" else actual_candidate.get(root)
            check(expected_plan is not None, "post_review_without_actual_valid_candidate", failures,
                  request=directory.name)
            if expected_plan:
                audit_event_context(ctx, original, expected_plan, failures, root=root, request=directory.name)
                check(proof.get("scene_plan_sha256") == digest(expected_plan["scenes"][0]),
                      "review_plan_proof_hash_changed", failures, request=directory.name)
            check(proof.get("plan_attempt") == (0 if stage == "initial_event_review" else 1),
                  "review_plan_attempt_changed", failures, request=directory.name)
            if counts[root][stage] == 1:
                check("targeted_repair" not in ctx, "initial_stage_request_has_repair", failures,
                      request=directory.name)
                if stage == "initial_event_review":
                    cell = next(value for value in manifest["cells"] if value["root_id"] == root)
                    prepared = read(cell["prepared_request"]["path"])["payload"]
                    check(payload == prepared, "first_review_not_exact_prepared", failures, request=directory.name)
            else:
                check(isinstance(ctx.get("targeted_repair"), dict), "review_retry_missing_repair", failures,
                      request=directory.name)
            common = dict(ctx); common.pop("targeted_repair", None)
            key = (root, stage)
            if key in prior_common:
                check(common == prior_common[key], "review_retry_changed_context_outside_repair", failures,
                      request=directory.name)
            prior_common[key] = common
            validation_path = directory / "VALIDATION.json"
            if validation_path.exists() and stage == "initial_event_review":
                validation = read(validation_path)
                if validation.get("status") == "PASS":
                    initial_diagnostics[root] = [value for value in validation.get("plan_diagnostics", [])
                                                 if value.get("requires_replan") is True]
        else:
            check(counts[root]["initial_event_review"] > 0, "replan_without_initial_review", failures,
                  request=directory.name)
            initial = review_artifact(root, "initial_event_review")
            check(isinstance(initial, dict) and initial["summary"].get("structurally_valid") is True
                  and initial["summary"].get("requires_replan") is True,
                  "replan_without_binding_negative_review", failures, request=directory.name)
            feedback = read(RUN / "roots" / root / "SEMANTIC_REPLAN_FEEDBACK.json")
            check(feedback.get("failure") == "event_state_requires_replan"
                  and feedback.get("previous_response") == input_by_root[root]["initial_plan"]
                  and feedback.get("plan_checks") == initial_diagnostics.get(root),
                  "semantic_feedback_not_original_binding_negatives", failures, request=directory.name)
            check(ctx.get("event_state") == initial.get("review") if isinstance(initial, dict) else False,
                  "replan_missing_exact_negative_snapshot", failures, request=directory.name)
            repaired = ctx.get("targeted_repair")
            if counts[root][stage] == 1:
                check(repaired == feedback, "first_replan_feedback_not_exact_production_artifact", failures,
                      request=directory.name)
            else:
                check(isinstance(repaired, dict) and repaired.get("previous_response") == previous_planner_raw.get(root),
                      "structural_retry_previous_response_not_actual", failures, request=directory.name)
            frozen_ctx = dict(ctx); frozen_ctx.pop("targeted_repair", None); frozen_ctx.pop("event_state", None)
            check(frozen_ctx == original, "replan_original_input_changed", failures, request=directory.name)
            fb = (RUN / "feedback" / root / "dataset_new_event__planner.txt").read_text(encoding="utf-8")
            check(payload["messages"][0]["content"].count(fb) == 1, "replan_frozen_feedback_changed", failures,
                  request=directory.name)
            if (directory / "CANDIDATE_PLAN.json").exists():
                actual_candidate[root] = read(directory / "CANDIDATE_PLAN.json")
                check(json.loads(read(directory / "RESPONSE.json")["text"]) == actual_candidate[root],
                      "candidate_plan_changed_raw_response", failures, request=directory.name)
            if (directory / "RESPONSE.json").exists():
                previous_planner_raw[root] = read(directory / "RESPONSE.json")["text"]

        attempts = sorted(directory.glob("HTTP_ATTEMPT*.json")); statuses = []
        check(len(attempts) <= 2, "http_retry_budget_exceeded", failures, request=directory.name)
        for attempt_number, attempt_path in enumerate(attempts, 1):
            attempt = read(attempt_path); http_numbers.append(attempt.get("http_attempt_number"))
            check(attempt.get("actual_payload_equals_frozen") is True and attempt.get("payload_sha256") == digest(payload),
                  "actual_http_payload_changed", failures, request=directory.name, attempt=attempt_number)
            result = read(directory / f"HTTP_RESULT_{attempt_number:02d}.json")
            statuses.append(result.get("http_status"))
            if attempt_number > 1:
                check(statuses[attempt_number - 2] in {503, 504}, "unallowed_transport_retry", failures,
                      request=directory.name)
        if (directory / "RESPONSE.json").exists():
            metadata = read(directory / "RESPONSE.json").get("metadata", {})
            check(metadata.get("finish_reason") == "stop", "incomplete_success_response", failures,
                  request=directory.name)
            token_rows.append({key: metadata.get(key) for key in ("input_tokens", "output_tokens", "total_tokens")})
        rows.append({"request": directory.name, "root_id": root, "stage": stage, "role": role,
                     "http_attempts": len(attempts), "http_statuses": statuses})
    check(sorted(http_numbers) == list(range(1, len(http_numbers) + 1)),
          "http_attempt_sequence_invalid", failures)
    check(len(http_numbers) <= 24, "global_http_budget_exceeded", failures, actual=len(http_numbers))
    for root in ROOTS:
        count = counts[root]
        check(all(count[stage] <= 2 for stage in STAGES) and sum(count.values()) <= 6,
              "per_root_stage_budget_exceeded", failures, root=root, counts=dict(count))
        observed = stage_order[root]
        check(observed == sorted(observed, key=STAGES.index), "stage_order_or_second_semantic_round_invalid",
              failures, root=root, stages=observed)
    report = read(RUN / "REPORT.json")
    check(report.get("status") == "COMPLETE" and [cell.get("root_id") for cell in report.get("cells", [])] == ROOTS,
          "report_not_complete_two_cases", failures)
    check(report.get("logical_requests") == len(requests) and report.get("http_attempts") == len(http_numbers),
          "report_request_counts_wrong", failures)
    for key in ("new_sources", "new_dialogue_turns", "ready_sources", "new_questions", "solver_calls", "training_jobs"):
        check(report.get(key) == 0, "regression_claims_production_output", failures, field=key, actual=report.get(key))
    check(report.get("fresh_yield_measured") is False and report.get("semantic_reliability_certified") is False,
          "regression_overclaims_fresh_yield_or_reliability", failures)
    for cell in report.get("cells", []):
        root = cell["root_id"]
        check(cell.get("semantic_replan_rounds") ==
              (1 if (RUN / "roots" / root / "SEMANTIC_REPLAN_FEEDBACK.json").exists() else 0),
              "report_semantic_round_count_wrong", failures, root=root)
        check(len(cell.get("planner_attempts", [])) <= counts[root]["semantic_replan"],
              "report_planner_count_wrong", failures, root=root)
        for report_key, stage in (("initial_review", "initial_event_review"),
                                  ("post_replan_review", "post_replan_event_review")):
            summary = cell.get(report_key)
            if isinstance(summary, dict):
                check(summary.get("logical_requests") == counts[root][stage],
                      "report_stage_call_count_wrong", failures, root=root, stage=stage)
                binding = summary.get("artifact", {})
                check(binding.get("path") and sha(binding["path"]) == binding.get("sha256"),
                      "report_review_artifact_binding_changed", failures, root=root, stage=stage)
                if binding.get("path"):
                    actual = read(binding["path"])
                    check({key: value for key, value in summary.items() if key != "artifact"} == actual.get("summary"),
                          "report_stage_summary_not_artifact", failures, root=root, stage=stage)
                    if actual.get("review") is not None:
                        audit_review_output(actual["review"], input_by_root[root]["context"], failures,
                                            root=root, stage=stage)
                        last = stage_requests[(root, stage)][-1]
                        validation = read(last / "VALIDATION.json")
                        check(validation.get("status") == "PASS"
                              and validation.get("event_state_sha256") == digest(actual["review"])
                              and validation.get("requires_replan") == summary.get("requires_replan"),
                              "review_artifact_not_actual_validated_response", failures, root=root, stage=stage)
                        raw_value = json.loads(read(last / "RESPONSE.json")["text"])
                        check(raw_value == actual["review"], "review_artifact_changed_raw_response", failures,
                              root=root, stage=stage)
        if cell.get("endpoint_review_without_replan") is True:
            last = cell.get("post_replan_review") or cell.get("initial_review")
            check(isinstance(last, dict) and last.get("structurally_valid") is True
                  and last.get("requires_replan") is False,
                  "report_endpoint_has_no_matching_machine_review", failures, root=root)
    return {"logical_requests": len(requests), "http_attempts": len(http_numbers), "role_counts": dict(role_counts),
            "per_root_stages": {root: dict(counts[root]) for root in ROOTS}, "requests": rows,
            "provider_tokens": {key: sum(row[key] or 0 for row in token_rows)
                                for key in ("input_tokens", "output_tokens", "total_tokens")},
            "report_sha256": sha(RUN / "REPORT.json")}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--phase", choices=("preflight", "live"), required=True)
    parser.add_argument("--output", type=Path); args = parser.parse_args()
    manifest_path = RUN / "REGRESSION_MANIFEST.json"; failures: list[dict[str, Any]] = []; summary = {}
    try:
        manifest = read(manifest_path)
        summary["frozen_inputs"] = frozen_audit(manifest, failures, live=args.phase == "live")
        if args.phase == "live": summary["live"] = live_audit(manifest, failures)
    except Exception as error:
        failures.append({"code": "audit_exception", "error": f"{type(error).__name__}:{error}"})
    output = {"version": "v100_independent_regression_audit_v1", "phase": args.phase,
              "status": "PASS" if not failures else "FAIL", "manifest_path": str(manifest_path),
              "manifest_sha256": sha(manifest_path) if manifest_path.is_file() else None,
              "audited_at_utc": datetime.now(timezone.utc).isoformat(), "audit_script_sha256": sha(__file__),
              "summary": summary, "failures": failures, "provider_calls_by_auditor": 0,
              "auditor_imports_engine": False, "semantic_certified": False,
              "classification": "exposed treatment-only regression; no fresh-yield measurement or paired causal claim"}
    path = args.output or BASE / "evidence" / f"REGRESSION_{args.phase.upper()}_AUDIT.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2); stream.write("\n")
    print(json.dumps({"phase": args.phase, "status": output["status"], "failures": len(failures),
                      "manifest_sha256": output["manifest_sha256"], "output": str(path)}))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__": main()
