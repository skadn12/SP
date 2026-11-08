"""Read-only frozen-executor gate reproduction. Never enables live calls.

All new files stay in this experiment directory. The original global SQLite
ledger is opened read-only. No tokenizer/model weights or credential is read.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path.insert(0, str(PROJECT))
NETWORK_EVENTS = []


def deny_network(event, args):
    if event.startswith("socket."):
        NETWORK_EVENTS.append(event)
        raise AssertionError("OFFLINE_NETWORK_FORBIDDEN")


sys.addaudithook(deny_network)

from scripts.current_method import base_qa_executor_v2 as executor
from scripts.current_method import base_qa_pipeline_v1 as qa
from scripts.current_method import base_qa_capability_probe_v1 as probe
from scripts.current_method import qa_api_transport_v2 as transport


def budget_snapshot():
    path = probe.CANONICAL / probe.LEDGER_NAME
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(r) for r in db.execute("SELECT * FROM attempts ORDER BY attempt_id")]
        tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    for row in rows:
        if row["receipt"]:
            row["receipt"] = transport._strict_json(row["receipt"])
    return {"global_limit": 12, "attempts_consumed": len(rows), "remaining": 12 - len(rows),
            "attempts": rows, "tables": tables}


def main():
    baseline = qa.read(HERE / "BEFORE_MANIFEST.json")
    for binding in baseline["files"]:
        if qa.legacy.file_binding(binding["path"]) != binding:
            raise RuntimeError("FROZEN_OR_GLOBAL_STATE_CHANGED_DURING_OFFLINE_PREFLIGHT")
    before = budget_snapshot()
    assert before["attempts_consumed"] == 4 and before["remaining"] == 8
    newest = HERE.parent / "base_qa_proxy_updated_key_probe_20261030_v1/PROBE_SUMMARY.json"
    evidence = qa.legacy.file_binding(newest)
    observed = qa.read(newest)
    assert observed["http_status"] == 200 and observed["returned_model"] == transport.EXPECTED_MODEL
    actual = {"endpoint": transport.ENDPOINT, "request_model": transport.MODEL,
              "response_model": observed["returned_model"], "output_parameter": "max_tokens",
              "max_output_tokens": 4096, "max_tokens_hard_limit_confirmed": False,
              "reasoning_in_output_budget_confirmed": False, "tokenizer_compatible": False,
              "context_window_tokens": None, "evidence_binding": evidence,
              "evidence_ref": str(newest), "false_means_unverified_not_disproved": True}
    probe.durable_write(HERE / "OBSERVED_CAPABILITY_NOT_VERIFIED.json", actual)
    reproductions = []

    # This constructor reads a manifest before constructing any ledger/client.
    with patch.object(executor, "Ledger") as ledger_factory, \
         patch.object(transport, "_read_credential") as credential_reader, \
         patch.object(transport, "SingleAttemptHTTPTransport") as client_factory:
        try:
            executor.Executor(probe.CANONICAL, mode="dry-run")
        except FileNotFoundError as error:
            reproductions.append({"test": "frozen_constructor_against_actual_canonical_directory",
                "result": "EXPECTED_HOLD", "error_type": type(error).__name__,
                "reason": "EXPERIMENT.json absent; directory is occupied by original budget takeover guard",
                "opened_new_ledger": False, "network_client_constructed": False})
        else:
            raise AssertionError("Unexpected usable canonical v2 experiment")
        ledger_factory.assert_not_called()
        credential_reader.assert_not_called()
        client_factory.assert_not_called()

    try:
        transport.validate_capability(actual)
    except transport.TransportError as error:
        reproductions.append({"test": "frozen_transport_actual_capability_facts",
                             "result": "EXPECTED_HOLD", "reason": error.reason})
    else:
        raise AssertionError("Unverified capability unexpectedly passed")

    # A test-only object isolates the original _authorize method. It does not
    # instantiate Executor, Ledger or transport, nor overwrite canonical state.
    # Unproved evidence stays false/null. No success-path capability is forged.
    with tempfile.TemporaryDirectory(dir=HERE, prefix="gate_fixture_") as directory:
        temp = Path(directory)
        cap = temp / "NOT_VERIFIED_CAPABILITY.json"
        probe.durable_write(cap, actual)
        subject = object.__new__(executor.Executor)
        subject.state = probe.CANONICAL.resolve()
        subject.manifest_sha = qa.legacy.fingerprint(qa.read(HERE / "TASK_AND_BUDGET_PLAN.json"))
        approval = {"test_fixture_only": True, "approved": True,
                    "experiment_sha256": subject.manifest_sha, "max_attempts": 12,
                    "max_output_tokens": 4096, "roots": list(executor.ROOTS),
                    "approval_reference": "Current overnight authorization; offline gate reproduction only",
                    "capability_binding": qa.legacy.file_binding(cap)}
        approval_path = temp / "OFFLINE_FIXTURE_ONLY.json"
        probe.durable_write(approval_path, approval)
        with patch.object(executor, "Ledger") as ledger_factory, \
             patch.object(transport, "_read_credential") as credential_reader, \
             patch.object(transport, "SingleAttemptHTTPTransport") as client_factory:
            try:
                subject._authorize(approval_path, cap)
            except executor.ExecutionHold as error:
                assert str(error) == "service/local tokenizer compatibility is unverified"
                reproductions.append({"test": "frozen_authorize_with_real_unverified_evidence",
                    "result": "EXPECTED_HOLD", "reason": str(error),
                    "fake_capability_flags_used": False, "fixture_files_removed": True})
            else:
                raise AssertionError("Unverified tokenizer/context unexpectedly passed")
            ledger_factory.assert_not_called()
            credential_reader.assert_not_called()
            client_factory.assert_not_called()

    v2_expected = {"metadata", "roots", "attempts"}
    actual_tables = set(before["tables"]) - {"sqlite_sequence"}
    assert actual_tables == {"policy", "attempts"} and actual_tables != v2_expected
    reproductions.append({"test": "readonly_ledger_schema_comparison",
        "result": "EXPECTED_HOLD", "v2_expected_business_tables": sorted(v2_expected),
        "authoritative_actual_business_tables": sorted(actual_tables),
        "new_or_reset_budget_created": False,
        "note": "Do not instantiate the v2 Ledger on the authoritative database to test this."})

    # A 64-token cut-off observation cannot establish these independent gates.
    missing = [
        {"gate": "tokenizer_compatible", "observed": "UNVERIFIED", "required": True,
         "line": "base_qa_executor_v2.py:180"},
        {"gate": "max_tokens_hard_limit_confirmed_at_4096", "observed": "UNVERIFIED", "required": True,
         "line": "base_qa_executor_v2.py:185"},
        {"gate": "reasoning_in_output_budget_confirmed", "observed": "UNVERIFIED", "required": True,
         "line": "base_qa_executor_v2.py:186"},
        {"gate": "verified_remote_context_window", "observed": None, "required": ">4096, evidence-backed",
         "line": "base_qa_executor_v2.py:193"},
        {"gate": "one_authoritative_budget_and_proxy_transport_integration", "observed": "NOT_CONNECTED",
         "required": "Reuse original global12 without resetting or bypassing any stage",
         "line": "base_qa_executor_v2.py:134,142,152"}]
    after = budget_snapshot()
    assert after == before
    changed = [b["path"] for b in baseline["files"] if qa.legacy.file_binding(b["path"]) != b]
    assert not changed and not NETWORK_EVENTS
    final = {"status": "OFFLINE_HOLD_HARD_SAFETY_GATES", "gate_reproductions": reproductions,
             "hard_blockers": missing, "network_audit_events": NETWORK_EVENTS,
             "new_model_calls": 0, "real_credential_reads": 0,
             "model_weights_loaded": False, "solver_runs": 0, "training_runs": 0,
             "unchanged_existing_files": len(baseline["files"]), "changed_existing_files": changed,
             "budget_used": after["attempts_consumed"], "budget_remaining": after["remaining"],
             "synthetic_length_probe": "SKIPPED: cannot resolve independent hard blockers; obey immediate stop",
             "completed_roots": 0, "started_roots": 0, "generated_candidates": 0,
             "candidate_holds": 0, "experiment_holds": 1,
             "finished_shanghai": datetime.now(timezone(timedelta(hours=8))).isoformat()}
    probe.durable_write(HERE / "OFFLINE_PREFLIGHT_RESULT.json", final)
    probe.durable_write(HERE / "GLOBAL_BUDGET_AFTER.json", after)
    probe.durable_write(HERE / "PRESERVATION_CHECK.json", {
        "files_checked": len(baseline["files"]), "changed_files": changed,
        "authoritative_ledger_byte_hash_unchanged": True, "new_model_calls": 0})
    print(json.dumps(final, ensure_ascii=False))


if __name__ == "__main__":
    main()
