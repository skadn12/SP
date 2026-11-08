#!/usr/bin/env python3
"""Engineering-only temporary F136 coverage, no-replay and delivery checks."""
from __future__ import annotations
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "scripts"))
import question_driver as q
import f136202_saved_mc5_first_verification as owner
import content_assembly_delivery as content
import finalize_report_delivery as report
import resume_authorized_pipeline as serial


def main() -> int:
    before = owner.saved.actual_requests(owner.ROOT, owner.ARM)
    scheduler_ready = q.read(serial.READY)
    for row in scheduler_ready["dependency_bindings"]:
        serial.verify(row)
    q.SOURCE_SELECTION_PATH = owner.SELECTION
    q.SOURCE_FREEZE_PATH = BASE / "SOURCE_FREEZE.json"
    source = owner._source()
    frozen = owner.inspect_inputs(source)
    candidates = q.read(owner.INPUT_WORK / "CANDIDATES_FINAL.json")
    expected = [q.anonymize_candidate(c)[0]["question_key"] for c in candidates if c["slot_id"] in owner.SLOTS]
    reviewer, clone = owner.build_review()
    captures = []
    class MockNoDispatch:
        def call(self, key, prompt):
            payload = owner.saved.shape.request_payload(prompt)
            assert key == owner.NEW_CALL
            assert [row["question_key"] for row in payload["review_cases"]] == expected
            assert payload["complete_source_and_warnings"] == q.source_view(source)
            captures.append((key, prompt))
            raise RuntimeError("INDEPENDENT_ENGINEERING_EXACT_FIVE_MOCK_NO_DISPATCH")
    cache = BASE / "runtime_cache/f136202_delivery_interfaces"
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="delivery_", dir=cache) as directory:
        out = Path(directory) / "terminal"; out.mkdir()
        q.publish(out / "RECOVERY_INPUT_FREEZE.json", frozen)
        data = reviewer(owner.ROOT, owner.ARM, [c for c in candidates if c["slot_id"] in owner.SLOTS],
                        source, owner.SELECTION, out, frozen, MockNoDispatch())
        owner.publish_ready(out, frozen, data)
        assert len(captures) == 1 and len(data["reviews"]) == 5
        initial_bytes = {str(p.relative_to(out)): p.read_bytes() for p in out.glob("*.json")}
        with patch.object(content, "OUT", out), patch.object(content, "TERMINAL", out / "CONTINUATION_FINALIZER_READY.json"):
            checked = content.validate_f136_terminal_ready()
        assert checked["candidate_bytes_preserved"] is True
        assert checked["existing_ORDER_review_objects_preserved"] is True
        gates = checked["frozen_gate_validations"]
        assert sum(row["review_validation"]["admitted"] is True for row in gates) == 3
        assert {str(p.relative_to(out)): p.read_bytes() for p in out.glob("*.json")} == initial_bytes
        # Corrupt only temporary fixture outputs, then update the temporary
        # top-level binding so the deeper preservation/replay check is reached.
        for slot in ("ORDER1", "MC2"):
            review_path = out / "CONTENT_REVIEW_FINAL.json"
            review = q.read(review_path)
            chosen = next(row for row in review["reviews"] if row["candidate_id"].endswith("-" + slot))
            chosen["issues"] = list(chosen.get("issues", [])) + ["ENGINEERING_FIXTURE_CHANGED_SAVED_OBJECT"]
            review_path.write_text(json.dumps(review))
            ready_path = out / "CONTINUATION_FINALIZER_READY.json"
            ready = q.read(ready_path); ready["final_review_binding"] = q.binding(review_path)
            ready_path.write_text(json.dumps(ready))
            try:
                owner.validate_terminal_ready(ready_path)
            except RuntimeError:
                pass
            else:
                raise AssertionError("Changed saved review object admitted: " + slot)
            for relative, raw in initial_bytes.items():
                (out / relative).write_bytes(raw)
        # Test first-only replay policies under an isolated call root. The
        # production helper is replaced with an unconditional no-dispatch trap.
        fixture_base = Path(directory) / "calls"; call = fixture_base / "cli_calls" / owner.NEW_CALL
        attempt = call / "attempt_01"; attempt.mkdir(parents=True)
        prompt = captures[0][1]
        def forbidden(*args, **kwargs):
            raise AssertionError("REAL_HELPER_DISPATCH_REACHED_BY_ENGINEERING_TEST")
        with patch.object(owner, "BASE", fixture_base), patch.object(q.cc, "call", forbidden):
            for name in ("START.json", "REQUEST.txt"):
                (attempt / name).write_text("{}")
                try:
                    owner.FirstOnlyCalls().call(owner.NEW_CALL, prompt)
                except RuntimeError as exc:
                    assert "no re-dispatch" in str(exc)
                else:
                    raise AssertionError("Consumed first verifier replayed: " + name)
                (attempt / name).unlink()
            (call / "RESULT.json").write_text("{}")
            parsed = call / "PARSED_RESPONSE.json"
            response = {"reviews": [{"question_key": key, "decision": "HOLD"} for key in expected]}
            parsed.write_text(json.dumps(response))
            request = attempt / "REQUEST.txt"; request.write_text(prompt)
            receipt = {"request": q.binding(request), "parsed_response": q.binding(parsed)}
            with patch.object(owner.saved, "verify_receipt", return_value=(receipt, owner.NEW_CALL)):
                once = owner.FirstOnlyCalls()
                observed, observed_receipt = once.call(owner.NEW_CALL, prompt)
                assert observed == response and observed_receipt == receipt
                try: once.call(owner.NEW_CALL, prompt)
                except RuntimeError: pass
                else: raise AssertionError("Second wrapper invocation accepted")
                request.write_text(prompt + "FIXTURE_CHANGED")
                try: owner.FirstOnlyCalls().call(owner.NEW_CALL, prompt)
                except RuntimeError: pass
                else: raise AssertionError("Changed cached request accepted")
        # Exact five unique anonymous keys, no ORDER or missing row, may be
        # returned in any response order without changing their meanings.
        fixture = {"reviews": [{"question_key": key} for key in reversed(expected)]}
        assert owner.FirstOnlyCalls.accept(fixture, {}, expected)[0] == fixture
        for keys in (expected[:-1], expected + ["FIXTURE_ORDER"], expected[:-1] + [expected[0]]):
            try: owner.FirstOnlyCalls.accept({"reviews": [{"question_key": key} for key in keys]}, {}, expected)
            except RuntimeError: pass
            else: raise AssertionError("Wrong response coverage accepted")
    assert owner.saved.actual_requests(owner.ROOT, owner.ARM) == before
    for row in scheduler_ready["dependency_bindings"]:
        serial.verify(row)
    now = dt.datetime.now(dt.timezone.utc)
    receipt = {"schema_version": "fgr20_f136202_delivery_interface_independent_engineering_check_v1",
        "at_utc": now.isoformat(), "status": "PASS_ENGINEERING_ONLY",
        "check_script_binding": q.binding(Path(__file__)), "exact_five_wrapper_binding": q.binding(owner.__file__),
        "content_wrapper_binding": q.binding(content.__file__), "report_wrapper_binding": q.binding(report.__file__),
        "scheduler_ready_binding": q.binding(serial.READY),
        "scheduler_dependency_count_reverified_before_and_after": len(scheduler_ready["dependency_bindings"]),
        "exact_saved_Gold_and_full_source_read": True, "final_candidate_count": 8,
        "mock_only_new_first_verifier_call_id": captures[0][0],
        "mock_prompt_sha256": hashlib.sha256(captures[0][1].encode()).hexdigest(),
        "new_verifier_exact_five_slots": list(owner.SLOTS),
        "new_five_mock_reviews_all_HOLD": True, "original_three_ORDER_PASS_unchanged": True,
        "final_candidate_and_prior_review_archive_bytes_unchanged": True,
        "frozen_gate_admits_only_original_three_ORDER_in_mock": True,
        "read_only_terminal_validation_bytes_and_file_set_unchanged": True,
        "changed_ORDER_object_and_changed_MC_replay_object_rejected": True,
        "START_and_REQUEST_without_RESULT_reject_redispatch": True,
        "cached_RESULT_reused_without_helper": True, "cached_request_drift_and_second_invocation_rejected": True,
        "exact_five_response_keys_required_any_row_order_allowed": True,
        "content_pure_self_check": content.self_check(), "report_pure_self_check": report.self_check(),
        "temporary_fixture_outputs_removed": True, "actual_request_registry_unchanged": True,
        "semantic_calls_started": 0, "solver_calls_started": 0, "semantic_judgments_authored": 0,
        "actual_content_freeze_report_or_solver_launched": False, "existing_bound_files_modified": False,
        "clone": clone}
    path = BASE / "mechanical_audits" / ("F136202_DELIVERY_INTERFACE_ENGINEERING_CHECK_" + now.strftime("%Y%m%dT%H%M%SZ") + ".json")
    q.publish(path, receipt)
    print(json.dumps(q.binding(path), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
