"""Read-only replay of finished diagnostic requests, not semantic certification."""
from pathlib import Path
import importlib.util
import json


def main():
    delivery = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("frozen_review_probe", delivery / "evidence/review_probe.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    r = probe.r
    spec = importlib.util.spec_from_file_location("frozen_review_audit", delivery / "expansion_engine/audit_run.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    run = probe.RUN
    manifest = r.base.read(run / "MANIFEST.json")
    report = r.base.read(run / "REPORT.json")
    r.base.verify_bindings(manifest["files"])
    for name in (delivery / "PRESERVATION_BEFORE.json",
                 delivery.parent / "h4_card_fidelity_event_consistency_minfix_v92/PRESERVATION_BEFORE.json"):
        r.base.verify_bindings(r.base.read(name)["files"])
    for row in r.base.read(run / "EXECUTION_CODE_SNAPSHOT.json")["files"]:
        r.base.verify_bindings([row["original"], row["snapshot"]])
    cards = r.base.read(probe.OLD_RUN / "dataset_cards_result.json")["accepted_cards"]
    by_source = {card["source_id"]: card for card in cards}
    allocation = r.base.read(probe.OLD_RUN / "UNIQUE_EVENT_ALLOCATION.json")
    count = 0
    rows = []
    for row in report["rows"]:
        number = int(row["root_id"].rsplit("-", 1)[1])
        case = probe.inputs(number, row["kind"], by_source, allocation)
        parsed = None
        for attempt in row["attempts"]:
            if "request" not in attempt:
                continue
            directory = Path(attempt["request"])
            r.base.verify_bindings([attempt["original_response"]])
            payload = r.base.read(directory / "REQUEST.json")
            proof = r.base.read(directory / "PROOF.json")
            r.base.require(proof["payload_sha256"] == r.base.digest(payload), "request_changed")
            raw = r.base.read(directory / "RESPONSE.json")["text"]
            saved = r.base.read(directory / "VALIDATION.json")
            if row["kind"] == "plan":
                context = audit.decode_context(payload, "\nEVENT_STATE_INPUT_JSON\n")
                audit.review_instruction_tail(payload, "event_state_extractor", context, [case["card"]])
                r.base.require(context["cumulative_history"] == case["history"], "history_changed")
                try:
                    parsed = r.event_contract.validate(raw, case["root_id"], r.ROUTES[1], case["scene"],
                                                      case["history"], cards=[case["card"]])
                    expected = {"status": "VALID_REVIEW", "requires_replan": r.event_contract.requires_replan(parsed),
                                "diagnostics": r.event_contract.plan_diagnostics(parsed), "semantic_certified": False}
                except (ValueError, TypeError) as error:
                    parsed = None
                    expected = {"status": "INVALID_REVIEW",
                                "errors": r.event_contract.build_repair(error, raw, history=case["history"])["errors"],
                                "semantic_certified": False}
            else:
                context = audit.decode_context(payload, "\nREVIEW INPUT DATA:\n")
                audit.review_instruction_tail(payload, "scene_reviewer", context)
                r.base.require(context["branch_history"] == case["history"], "history_changed")
                expected = r.realization_review.validate_review(raw, case["root_id"], r.ROUTES[1],
                    case["scene"], case["realization"], dataset_card=case["card"], latest_state=case["state"],
                    history=case["history"])
                parsed = expected["parsed"] if expected["schema_valid"] else None
            r.base.require(saved == expected == attempt["validation"], "review_validation_replay_changed")
            r.base.require(context["scene_plan"] == case["scene"], "plan_changed")
            count += 1
        r.base.require(row["review_complete"] == (parsed is not None), "review_completion_changed")
        found = probe.expected_observation(number, row["kind"], parsed) if parsed else None
        r.base.require(row["expected_narrow_observation_found"] == found, "narrow_diagnostic_changed")
        rows.append({"root_id": row["root_id"], "kind": row["kind"], "review_complete": parsed is not None})
    result = {"status": "PASS", "requests_replayed": count, "rows": rows,
              "semantic_certified": False, "http_calls": 0,
              "scope": "exact input/history/tail/raw-response binding and validation replay only"}
    r.base.write(delivery / "evidence/DIAGNOSTIC_ARTIFACT_AUDIT.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
