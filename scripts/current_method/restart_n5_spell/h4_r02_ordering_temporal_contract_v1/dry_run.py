"""Actual scheduler with injected fake clients in temporary runs; never network."""
from __future__ import annotations

import tempfile
import urllib.error
from pathlib import Path
from unittest.mock import patch

from ..core import atomic_json, sha256_file


def exercise(source):
    from . import runtime, dependencies
    from .test_runtime import seed_run, synthetic_candidates, FakeTokenPreflight, FakeClient

    scenarios = []
    expected = {
        "success": (12, 12, False, 48),
        "ordering_missing": (24, 24, False, 48),
        "malformed_json": (24, 24, False, 0),
        "known_429_rejected": (13, 13, False, 48),
        "timeout_unknown": (1, 1, True, 0),
        "caught_verifier_error": (0, 0, True, 0),
        "repair_capacity_block": (1, 1, True, 0),
        "graph_failure_both_stages": (18, 18, False, 24),
    }
    for scenario in expected:
        with tempfile.TemporaryDirectory(prefix="ordering-offline-") as tmp:
            target = seed_run(tmp, source_run=source)
            calls, verifications = [], []
            tokens = FakeTokenPreflight(block_repair=scenario == "repair_capacity_block")

            def reply(request):
                if scenario == "timeout_unknown":
                    return TimeoutError("OFFLINE_SYNTHETIC_UNKNOWN")
                if scenario == "known_429_rejected" and len(calls) == 1:
                    return urllib.error.HTTPError("https://example.invalid", 429, "OFFLINE_REJECTION", {}, None)
                if scenario == "malformed_json":
                    return '{"candidates": ['
                repair = "REPAIR THE FOLLOWING BATCH" in request["prompt"]
                counts = (2, 1, 0) if scenario in ("ordering_missing", "repair_capacity_block") and not repair else (2, 1, 1)
                result = synthetic_candidates(request["prompt"], counts)
                if scenario == "graph_failure_both_stages":
                    for row in result["candidates"]:
                        if row.get("ordering_contract_version"):
                            row["temporal_relations"] = []
                return result

            def verifier():
                verifications.append(True)
                if scenario == "caught_verifier_error" and len(verifications) >= 2:
                    raise ValueError("OFFLINE_FROZEN_IDENTITY_GUARD")

            def execute():
                return runtime.execute_schedule(target,
                    client_factory=lambda row: FakeClient(calls, reply),
                    verifier=verifier, token_preflight=tokens, backend_mode="mock")

            stopped, reason, result = False, None, None
            with dependencies.forbid_network(), \
                    patch("urllib.request.urlopen", side_effect=AssertionError("urllib forbidden in offline exercise")), \
                    patch("scripts.current_method.restart_n5_spell.live_backends.OpenAICompatibleChallengerClient", side_effect=AssertionError("real client construction forbidden")), \
                    patch("scripts.current_method.restart_n5_spell.h4_qa_multiroot_k3_t08_probe_v1.generation.OpenAICompatibleChallengerClient", side_effect=AssertionError("real client construction forbidden")):
                try:
                    result = execute()
                except BaseException as error:
                    stopped, reason = True, type(error).__name__ + ":" + str(error)
                calls_before_resume = len(calls)
                preserved = {str(p.relative_to(target)): sha256_file(p)
                             for p in target.glob("arms/**/*") if p.is_file()}
                if not stopped:
                    assert execute() == result
                else:
                    try:
                        execute()
                    except BaseException:
                        pass
                    else:
                        raise AssertionError("guarded run unexpectedly resumed")
                assert len(calls) == calls_before_resume
            reservations = len(list(target.glob("arms/**/ledger/requests/*.json")))
            stages = runtime.rows(target / "reports/runtime_stage_records.jsonl")
            finals = runtime.rows(target / "reports/runtime_final_candidates.jsonl")
            checks = {
                "expected_calls_reservations_stop_and_final_count": (len(calls), reservations, stopped, len(finals)) == expected[scenario],
                "resume_no_resend": len(calls) == calls_before_resume,
                "resume_artifacts_preserved": preserved == {str(p.relative_to(target)): sha256_file(p) for p in target.glob("arms/**/*") if p.is_file()},
                "no_http_attempts": not runtime.rows(target / "audit/http_events.jsonl"),
                "both_arms_intent_guidance": all("CONTRACT GUIDANCE:" in c["prompt"] for c in calls),
                "no_scms_quality_block": all("BEGIN_SCMS_QUALITY_GUIDANCE_V1" not in c["prompt"] for c in calls),
                "unchanged_output_limit": all(c["max_tokens"] == 8192 for c in calls),
            }
            if not stopped:
                checks.update(
                    stage_rows_match_reservations=len(stages) == reservations,
                    unique_stage_identity=len({r["global_request_id"] for r in stages}) == len(stages),
                    separate_final_analysis_identity=len({tuple(r["analysis_identity"]) for r in finals}) == len(finals),
                    every_stage_keeps_raw_and_parse_status=all("raw_parsed" in r and "parse_error" in r for r in stages),
                    both_arms_alignment_available=all(r.get("intent_guidance") is True and r.get("contract_alignment_analysis") == "AVAILABLE_FOR_OFFLINE_CONTENT_REVIEW" for r in stages + finals),
                    finalizer_artifacts_exist=all((runtime.root_path(target, arm, root) / name).is_file()
                        for arm in runtime.ARMS for root in runtime.ROOTS
                        for name in ("candidate_generation.json", "candidate_pool.json", "validity_audit.json", "candidates_valid.json", "permutations.json")),
                    future_limits=result["primary_calls"] == 12 and result["repair_calls"] <= 12,
                )
                new_ordering = [row for row in finals if row["arm_id"] == runtime.ARMS[1] and row["question_type"] == "ordering"]
                checks["new_ordering_has_program_sidecar"] = all(row.get("ordering_derivation") for row in new_ordering)
            else:
                expected_reason = {"timeout_unknown": "unknown_service_completion",
                    "caught_verifier_error": "OFFLINE_FROZEN_IDENTITY_GUARD",
                    "repair_capacity_block": "TOKEN_CONTEXT_EXCEEDED"}[scenario]
                checks.update(expected_guard_reason=expected_reason in reason,
                              no_final_qa=not finals,
                              partial_stage_archive_retained=bool(list(target.glob("arms/**/stage_calls/*/*.json"))))
            scenarios.append({"scenario": scenario, "status": "PASS" if all(checks.values()) else "FAIL",
                "checks": checks, "synthetic_client_calls": len(calls), "synthetic_reservations": reservations,
                "synthetic_stage_rows": len(stages), "synthetic_final_candidates": len(finals),
                "stopped": stopped, "reason": reason, "verifier_checks": len(verifications),
                "token_checks": len(tokens.calls), "resume_resends": len(calls) - calls_before_resume,
                "real_client_constructions": 0, "actual_network_calls": 0,
                "synthetic_output_retained_as_experiment_QA": False})
    report = {"status": "PASS" if all(x["status"] == "PASS" for x in scenarios) else "FAIL",
        "scenarios": scenarios, "repository_modules": sorted(map(str, dependencies.loaded_repository_files())),
        "method": "New execute_schedule with injected FakeClient -> original RequestLedger -> shared execute_writer_batch -> original finalizer. All runs are temporary; no execute_live or real client/HTTP invocation.",
        "external_calls": 0}
    atomic_json(Path(source) / "audit/dry_run_report.json", report)
    return report
