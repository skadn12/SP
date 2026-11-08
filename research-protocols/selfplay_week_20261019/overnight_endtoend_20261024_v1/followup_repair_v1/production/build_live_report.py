#!/usr/bin/env python3
"""Read-only audit and exclusive report for the authorized single v1 probe."""
from datetime import datetime
from pathlib import Path

from completion_guard import validate_completion
from production_core import EvidenceResolver, bind, read, require, write
from run_production import HERE, OET1, load_module


def main():
    run = HERE / "live_probe_24719_generic_v1"
    integrity = load_module("oet1_live_probe_receipt_check", OET1 / "scripts/production_integrity.py")
    calls = []
    for path in sorted((run / "cli_calls").glob("*/RESULT.json")):
        result, events = integrity.validate_call_result(path)
        completion = validate_completion(result, events)
        calls.append({"call_id": path.parent.name, "result_binding": bind(path),
                      "raw_response": result["raw_response"], "request": result["request"],
                      "backend": result["backend"], "finished_at_utc": result["finished_at_utc"],
                      "usage": result["usage"], "provenance_check": "PASS",
                      "normal_completion_check": completion})
    protocol = read(run / "PROTOCOL.json")
    for value in protocol["code_bindings"]:
        require(bind(value["path"]) == value, "LIVE_PROTOCOL_CODE_CHANGED")
    unit = run / "sources/cc-train-episode-24719/GENERIC"
    sd = unit / "OET1_A_S01"
    packet = read(unit / "SOURCE_AUTHOR_INPUT.json")
    source = read(sd / "CUMULATIVE_SOURCE.json")
    review = read(sd / "REVIEW.json")
    resolver = EvidenceResolver(packet, source, "OET1:A:S01")
    exact, mismatches = [], []
    for dimension, check in review["checks"].items():
        for evidence in check["evidence"]:
            ref, domain, turn, text = resolver.resolve(evidence["turn_ref"])
            if evidence["quote"] in text:
                exact.append({"dimension": dimension, "turn_ref": ref})
            else:
                elsewhere = [t["turn_ref"] for t in source["turns"] if evidence["quote"] in t["text"]]
                mismatches.append({"dimension": dimension, "reported_turn_ref": ref,
                                   "reported_quote": evidence["quote"], "actual_referenced_turn_text": text,
                                   "exact_quote_found_elsewhere": elsewhere, "reference_rewritten": False})
    starts = [read(path)["started_at_utc"] for path in (run / "cli_calls").glob("*/attempt_*/START.json")]
    failure = read(sd / "FAILURE.json")
    started = min(starts)
    finished = failure["at_utc"]
    report = {"schema_version": "oet1_followup_live_probe_report_v1",
              "status": "STOPPED_STRICT_EVIDENCE_REJECTION", "protocol_binding": bind(run / "PROTOCOL.json"),
              "new_protocol_separate_from_old_frozen_experiment": True, "old_response_recovery_used": False,
              "root_id": "cc-train-episode-24719", "condition": "GENERIC", "planned_scenes": 6,
              "scenes_attempted": 1, "scenes_accepted": 0, "remaining_scenes_not_attempted": 5,
              "model_calls": len(calls), "transport_attempts": len(starts), "semantic_retries": 0,
              "started_at_utc": started, "finished_at_utc": finished,
              "elapsed_seconds": (datetime.fromisoformat(finished) - datetime.fromisoformat(started)).total_seconds(),
              "initial_turn_count": len(packet["full_initial_source"]["turns"]),
              "new_turn_count": len(read(sd / "DRAFT.json")["turns"]), "actual_source_turn_count": len(source["turns"]),
              "reported_review_source_turn_count": review["source_turn_count"], "raw_review_decision": review["decision"],
              "raw_review_failure_codes": review["failure_codes"], "strict_admission": "REJECT",
              "rejection_code": failure["error_code"], "exact_quote_anchors": len(exact),
              "mismatched_quote_anchors": mismatches, "calls": calls,
              "prefix_proof": bind(sd / "PREFIX_PROOF.json"), "failure": bind(sd / "FAILURE.json"),
              "new_complete_sources": 0, "new_questions": 0,
              "sandbox_transport_blocked": False, "frozen_stop_gate_bypassed": False,
              "source_text_correctness_proven": False, "native_resolver_live_tested": False,
              "interpretation": "New source/review calls completed normally; review evidence was invalid. No true content failure is inferred from this evidence mismatch alone. No failed review field was rewritten and no new semantic attempt was made."}
    write(HERE / "LIVE_PROBE_REPORT.json", report)
    print(f"{report['status']}: {len(calls)} calls, {report['elapsed_seconds']:.2f}s, {len(mismatches)} mismatched evidence anchors")


if __name__ == "__main__":
    main()
