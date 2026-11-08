"""Replay finished original-plan diagnostic, without editing requests or verdicts."""
from pathlib import Path
import importlib.util
import json


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    delivery = Path(__file__).resolve().parents[1]
    probe = module("v94_probe_replay", delivery / "evidence/plan_review_probe.py")
    r = probe.r
    audit = module("v94_audit_replay", delivery / "expansion_engine/audit_run.py")
    report = r.base.read(probe.RUN / "REPORT.json")
    manifest = r.base.read(probe.RUN / "MANIFEST.json")
    r.base.verify_bindings(manifest["files"])
    probe.protected_inputs()
    for pair in r.base.read(probe.RUN / "EXECUTION_CODE_SNAPSHOT.json")["files"]:
        r.base.verify_bindings([pair["original"], pair["snapshot"]])
    cases = {case["case_id"]: case for case in [probe.case_inputs(*args) for args in probe.CASES]}
    requests, attempts, failed_requests = 0, 0, 0
    for row in report["rows"]:
        case = cases[row["case_id"]]
        result = None
        for attempt in row["attempts"]:
            failed = "request" not in attempt
            if failed:
                paths = list((probe.RUN / "requests").glob(f"*-{case['case_id']}-plan-review-{attempt['attempt']}"))
                r.base.require(len(paths) == 1, "failed_request_not_preserved")
                path = paths[0]
            else:
                path = Path(attempt["request"])
                r.base.verify_bindings([attempt["original_response"]])
            request, proof = (r.base.read(path / name) for name in ("REQUEST.json", "PROOF.json"))
            r.base.require(request["model"] == r.MODEL, "requested_model_changed")
            r.base.require(proof["payload_sha256"] == r.base.digest(request), "payload_changed")
            context = audit.decode_context(request, "\nEVENT_STATE_INPUT_JSON\n")
            audit.review_instruction_tail(request, "event_state_extractor", context, [case["card"]])
            for key, expected in (("root_id", case["root_id"]), ("route", r.ROUTES[1]),
                                  ("scene_plan", case["scene"]), ("cumulative_history", case["history"]),
                                  ("latest_state", case["state"])):
                r.base.require(context[key] == expected, "original_input_changed:" + key)
            r.base.require(context["dataset_cards"] == [r.planner_contract.mapping_context(case["card"])["card_guidance"]],
                           "assigned_card_changed")
            for http in path.glob("HTTP_ATTEMPT*.json"):
                evidence = r.base.read(http)
                r.base.require(evidence["payload_sha256"] == r.base.digest(request), "http_payload_changed")
                r.base.require(evidence["actual_payload_equals_frozen"] is True, "http_payload_not_frozen")
                attempts += 1
            if failed:
                r.base.require(attempt["status"] == "TRANSPORT_FAILURE", "failed_request_status_changed")
                r.base.require((path / "ERROR.json").exists(), "failed_request_error_missing")
                r.base.require(not (path / "RESPONSE.json").exists(), "failed_completion_promoted_to_valid_response")
                failed_requests += 1
                result = None
                continue
            raw = r.base.read(path / "RESPONSE.json")["text"]
            try:
                result = r.event_contract.validate(raw, case["root_id"], r.ROUTES[1], case["scene"],
                                                  case["history"], cards=[case["card"]])
                r.base.require(result == r.parse_object(raw), "raw_review_was_rewritten")
                expected = {"status": "VALID_REVIEW", "requires_replan": r.event_contract.requires_replan(result),
                            "diagnostics": r.event_contract.plan_diagnostics(result), "semantic_certified": False}
            except (ValueError, TypeError) as error:
                result = None
                expected = {"status": "INVALID_REVIEW", "errors": r.event_contract.build_repair(error, raw,
                    history=case["history"])["errors"], "semantic_certified": False}
            r.base.require(expected == attempt["validation"] == r.base.read(path / "VALIDATION.json"),
                           "validation_replay_changed")
            requests += 1
        r.base.require(row["review_complete"] == (result is not None), "review_completion_changed")
        found = probe.expected_observation(case, result) if result else None
        r.base.require(row["expected_narrow_observation_found"] == found, "diagnostic_changed")
    r.base.require(attempts == report["http_attempts"], "http_count_changed")
    result = {"status": "PASS", "valid_or_invalid_reviews_replayed": requests,
              "failed_completion_requests_verified": failed_requests, "http_artifacts_verified": attempts,
              "request_artifact_bindings": r.base.bindings(list((probe.RUN / "requests").rglob("*.json"))),
              "semantic_certified": False, "model_calls": 0,
              "scope": "source/card/full-history/request-tail/raw-response binding and validation replay"}
    r.base.write(delivery / "evidence/DIAGNOSTIC_ARTIFACT_AUDIT.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
