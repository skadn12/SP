#!/usr/bin/env python3
"""CPU-only regressions using REAL frozen responses; no model or GPU calls.

The two transport integration tests replay historical responses in temporary
TEST_ONLY receipt trees. They are controller tests, not new production results.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
from production_core import (CHECKS, SCENES, EvidenceResolver, ProductionError, bind,
                             build_cumulative, read, safe_log, validate_review,
                             validate_zero_change, write)
from run_production import HERE, OET1, ProductionPipeline, owned_output, preflight, read_scope_receipt
from completion_guard import validate_completion
from run_production_v2 import ProductionPipeline as ProductionPipelineV2

FROZEN_BEFORE = None
FROZEN_AFTER = None


def frozen_snapshot():
    paths = set()
    for directory in ("inputs", "sources", "cli_calls", "scripts"):
        paths.update(path for path in (OET1 / directory).rglob("*") if path.is_file())
    paths.update(path for path in OET1.glob("*") if path.is_file())
    return [bind(path) for path in sorted(paths)]


def setUpModule():
    global FROZEN_BEFORE
    FROZEN_BEFORE = frozen_snapshot()


def tearDownModule():
    global FROZEN_AFTER
    FROZEN_AFTER = frozen_snapshot()
    if FROZEN_BEFORE != FROZEN_AFTER:
        raise AssertionError("FROZEN_HISTORICAL_FILES_CHANGED")


def fixture(root, arm, scene, *, raw=False):
    root = "cc-train-episode-" + str(root)
    unit = OET1 / "sources" / root / arm
    sd = unit / scene.replace(":", "_")
    packet = read(unit / "SOURCE_AUTHOR_INPUT.json")
    response_path = (OET1 / "cli_calls" / f"{root}__{arm}__{scene.replace(':', '_')}__REVIEW" /
                     "PARSED_RESPONSE.json") if raw else sd / "REVIEW.json"
    return packet, read(sd / "CUMULATIVE_SOURCE.json"), read(response_path)


class CoreRegression(unittest.TestCase):
    def positive(self):
        return fixture(24719, "GENERIC", "OET1:B:S02", raw=True)

    def assert_code(self, code, function, *args, **kwargs):
        with self.assertRaises(ProductionError) as caught:
            function(*args, **kwargs)
        self.assertEqual(caught.exception.code, code)
        return caught.exception

    def test_safe_logger_accepts_caller_reserved_fields(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            path = Path(temporary) / "PROGRESS.jsonl"
            row = safe_log(path, "EVENT", at_utc="historical-time", kind="caller-kind", emit=False)
            self.assertEqual(read(path), row)
            self.assertEqual(row["kind"], "EVENT")
            self.assertEqual(row["caller_kind"], "caller-kind")
            self.assertEqual(row["caller_at_utc"], "historical-time")
            self.assertNotEqual(row["at_utc"], "historical-time")
            self.assertEqual(path.read_text().count('"at_utc"'), 1)

    def test_explicit_count_and_bool_schemas(self):
        self.assertTrue(validate_zero_change(0, schema="count"))
        self.assertTrue(validate_zero_change(False, schema="boolean"))
        for value in (False, True, 0.0, -1, "0", None):
            with self.subTest(schema="count", value=value):
                self.assert_code("INVALID_NONNEGATIVE_COUNT", validate_zero_change, value, schema="count")
        for value in (0, 1, None, "false"):
            with self.subTest(schema="boolean", value=value):
                self.assert_code("INVALID_BOOLEAN", validate_zero_change, value, schema="boolean")
        self.assert_code("NONZERO_CHANGE_COUNT", validate_zero_change, 1)
        self.assert_code("CHANGE_FLAG_TRUE", validate_zero_change, True, schema="boolean")

    def test_real_stale_count_reconstructed_only_in_new_object(self):
        packet, historical, review = fixture(25652, "TARGETED", "OET1:A:S01")
        sd = OET1 / "sources/cc-train-episode-25652/TARGETED/OET1_A_S01"
        previous_path = Path(packet["source_binding"]["path"])
        previous = read(previous_path)
        repaired, proof = build_cumulative(packet, previous, read(sd / "DRAFT.json"), "OET1:A:S01",
                                          bind(previous_path), bind(sd / "DRAFT.json"))
        self.assertEqual(historical["previous_turn_count"], 159)
        self.assertEqual(repaired["previous_turn_count"], 171)
        self.assertEqual(len(repaired["turns"]), 185)
        self.assertEqual(repaired["turns"], historical["turns"])
        self.assertEqual(repaired["turns"][:171], previous["turns"])
        self.assertTrue(proof["exact_previous_turn_prefix"] and proof["exact_draft_turn_suffix"])
        self.assertEqual(review["decision"], "HOLD")
        self.assert_code("REVIEW_SEMANTIC_ADMISSION_REJECTED", validate_review,
                         review, packet, repaired, "OET1:A:S01")

    def test_cumulative_detects_prefix_content_change(self):
        packet, _, _ = self.positive()
        root = OET1 / "sources/cc-train-episode-24719/GENERIC"
        previous_path = root / "OET1_B_S01/CUMULATIVE_SOURCE.json"
        draft_path = root / "OET1_B_S02/DRAFT.json"
        previous = read(previous_path)
        previous["turns"][0]["text"] += " changed"
        self.assert_code("INITIAL_SOURCE_PREFIX_CHANGED", build_cumulative, packet, previous,
                         read(draft_path), "OET1:B:S02", bind(previous_path), bind(draft_path))

    def test_real_canonical_native_recovery(self):
        packet, source, response = fixture(24719, "GENERIC", "OET1:B:S01", raw=True)
        receipt = validate_review(response, packet, source, "OET1:B:S01")
        self.assertTrue(receipt["semantic_admitted"])
        self.assertGreater(receipt["quote_counts_by_domain"]["assigned_native_record"], 0)

    def test_real_array_native_alias(self):
        packet, source, response = self.positive()
        original = deepcopy(response)
        receipt = validate_review(response, packet, source, "OET1:B:S02")
        self.assertEqual(response, original)
        self.assertEqual(receipt["quote_counts_by_domain"], {"cumulative_source": 21, "assigned_native_record": 1})
        native = [item for item in receipt["checked_quotes"] if item["domain"] == "assigned_native_record"]
        self.assertEqual(native[0]["canonical_turn_ref"], "MUL0564.json:7")
        self.assertEqual(native[0]["raw_turn_ref"], "assigned_native_event.native_record.turns[7]")

    def test_all_native_aliases_are_one_domain(self):
        packet, source, _ = self.positive()
        resolver = EvidenceResolver(packet, source, "OET1:B:S02")
        refs = ["MUL0564.json:7", "MUL0564.json:turn_id=7", "assigned_native_event.native_record.turns[7]"]
        resolutions = [resolver.resolve(ref) for ref in refs]
        self.assertTrue(all(result == resolutions[0] for result in resolutions))
        self.assertEqual(len(resolver.native), len(packet["native_events"][1]["native_record"]["turns"]))
        self.assertEqual(resolutions[0][3], "The TR0254 leaves Cambridge at 17:21.")

    def test_native_array_alias_uses_position_not_turn_id(self):
        packet, source, _ = self.positive()
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            packet = deepcopy(packet)
            record = deepcopy(packet["native_events"][1]["native_record"])
            record["turns"][7]["turn_id"] = "seventy"
            path = Path(temporary) / "native.json"
            write(path, record)
            packet["native_events"][1].update(native_event_binding=bind(path), native_record=record)
            canonical = EvidenceResolver(packet, source, "OET1:B:S02").resolve(
                "assigned_native_event.native_record.turns[7]")[0]
            self.assertEqual(canonical, "MUL0564.json:seventy")

    def test_unassigned_native_and_unsafe_aliases_rejected(self):
        packet, source, _ = self.positive()
        resolver = EvidenceResolver(packet, source, "OET1:B:S02")
        for ref in ("MUL0526.json:9", "MUL0564.json:turn_id=999", "assigned_native_event.native_record.turns[-1]",
                    "assigned_native_event.native_record.turns[07]", "assigned_native_event.native_record.turns[7].frames",
                    "native_events[0].native_record.turns[7]", "MUL0564.json.frames.state"):
            with self.subTest(ref=ref):
                self.assert_code("UNKNOWN_OR_UNAUTHORIZED_EVIDENCE_REF", resolver.resolve, ref)
        self.assert_code("NATIVE_ARRAY_INDEX_OUT_OF_RANGE", resolver.resolve,
                         "assigned_native_event.native_record.turns[999]")

    def test_no_native_domain_for_route_a(self):
        packet, source, _ = fixture(24719, "TARGETED", "OET1:A:S01")
        resolver = EvidenceResolver(packet, source, "OET1:A:S01")
        self.assertFalse(resolver.native)
        self.assert_code("UNKNOWN_OR_UNAUTHORIZED_EVIDENCE_REF", resolver.resolve, "MUL0564.json:7")

    def test_duplicate_source_ref_rejected(self):
        packet, source, _ = self.positive()
        source["turns"].append(deepcopy(source["turns"][0]))
        self.assert_code("DUPLICATE_SOURCE_REFS", EvidenceResolver, packet, source, "OET1:B:S02")

    def test_native_packet_and_bound_record_must_match(self):
        packet, source, _ = self.positive()
        packet["native_events"][1]["native_record"]["turns"][0]["utterance"] += " changed"
        self.assert_code("PACKET_NATIVE_RECORD_DIFFERS_FROM_BOUND_RECORD", EvidenceResolver, packet, source, "OET1:B:S02")

    def test_native_hash_binding_rejected(self):
        packet, source, _ = self.positive()
        packet["native_events"][1]["native_event_binding"]["sha256"] = "0" * 64
        self.assert_code("BINDING_CHANGED", EvidenceResolver, packet, source, "OET1:B:S02")

    def test_real_frames_state_quote_still_rejected(self):
        packet, source, response = fixture(31457, "GENERIC", "OET1:B:S02", raw=True)
        error = self.assert_code("QUOTE_NOT_EXACT_CONTIGUOUS_SUBSTRING", validate_review,
                                 response, packet, source, "OET1:B:S02")
        self.assertEqual(error.details["canonical_reference"], "MUL0870.json:14")
        self.assertEqual(error.details["domain"], "assigned_native_record")

    def test_real_bad_read_count_still_rejected(self):
        packet, source, response = fixture(29753, "TARGETED", "OET1:B:S03", raw=True)
        error = self.assert_code("REVIEW_SOURCE_COUNT_MISMATCH", validate_review, response, packet, source, "OET1:B:S03")
        self.assertEqual(error.details, {"expected": 258, "reported": 260})
        self.assertEqual(response["source_turn_count"], 260)
        self.assertEqual(response["decision"], "FAIL")

    def test_three_real_content_failures_have_valid_evidence_but_no_admission(self):
        cases = [(25652, "GENERIC", "OET1:A:S03", "UNRESOLVED_PROJECT_YEAR_BINDING"),
                 (29753, "GENERIC", "OET1:A:S01", "UNSUPPORTED_CURRENT_CLINIC_APPOINTMENT_STATUS"),
                 (31457, "TARGETED", "OET1:A:S02", "TIME_ORDER_CONTRADICTION")]
        for root, arm, scene, code in cases:
            with self.subTest(root=root, arm=arm, scene=scene):
                packet, source, response = fixture(root, arm, scene)
                receipt = validate_review(response, packet, source, scene, require_ready=False)
                self.assertFalse(receipt["semantic_admitted"])
                self.assertIn(code, response["failure_codes"])
                self.assert_code("REVIEW_SEMANTIC_ADMISSION_REJECTED", validate_review, response, packet, source, scene)

    def test_bad_quote_speaker_and_unknown_finding_rejected(self):
        for mutation, code in (("quote", "QUOTE_NOT_EXACT_CONTIGUOUS_SUBSTRING"),
                               ("empty_quote", "EMPTY_EVIDENCE_QUOTE"),
                               ("speaker", "EVIDENCE_SPEAKER_MISMATCH"),
                               ("finding", "UNKNOWN_OR_UNAUTHORIZED_EVIDENCE_REF")):
            with self.subTest(mutation=mutation):
                packet, source, response = self.positive()
                evidence = response["checks"]["time"]["evidence"][0]
                if mutation == "quote": evidence["quote"] += " NEVER_IN_ORIGINAL"
                if mutation == "empty_quote": evidence["quote"] = ""
                if mutation == "speaker": evidence["speaker"] = "UNKNOWN_SPEAKER"
                if mutation == "finding": response["findings"].append({"severity": "WARNING", "turn_refs": ["NOT_ASSIGNED:1"]})
                self.assert_code(code, validate_review, response, packet, source, "OET1:B:S02")

    def test_full_read_independence_and_integer_count(self):
        for field, value, code in (("full_source_read", False, "REVIEW_FULL_SOURCE_READ_REQUIRED"),
                                   ("full_source_read", 1, "REVIEW_FULL_SOURCE_READ_REQUIRED"),
                                   ("independent_of_source_author", False, "REVIEW_INDEPENDENCE_REQUIRED"),
                                   ("source_turn_count", True, "REVIEW_SOURCE_COUNT_MISMATCH"),
                                   ("source_turn_count", 258.0, "REVIEW_SOURCE_COUNT_MISMATCH"),
                                   ("source_or_gold_ambiguity_failure", 0, "REVIEW_AMBIGUITY_BOOLEAN_REQUIRED")):
            with self.subTest(field=field, value=value):
                packet, source, response = self.positive()
                response[field] = value
                self.assert_code(code, validate_review, response, packet, source, "OET1:B:S02")

    def test_missing_review_dimension(self):
        packet, source, response = self.positive()
        del response["checks"]["time"]
        self.assert_code("REVIEW_REQUIRED_DIMENSIONS_MISSING", validate_review, response, packet, source, "OET1:B:S02")

    def test_every_semantic_gate_independently_rejects(self):
        for mutation in ("HOLD", "FAIL", "ambiguity", "check", "blocking", "failure_codes"):
            with self.subTest(mutation=mutation):
                packet, source, response = self.positive()
                if mutation in ("HOLD", "FAIL"): response["decision"] = mutation
                if mutation == "ambiguity": response["source_or_gold_ambiguity_failure"] = True
                if mutation == "check": response["checks"]["time"]["status"] = "FAIL"
                if mutation == "blocking": response["findings"].append({"severity": "BLOCKING", "turn_refs": []})
                if mutation == "failure_codes": response["failure_codes"] = ["TEST_FAILURE"]
                self.assert_code("REVIEW_SEMANTIC_ADMISSION_REJECTED", validate_review, response, packet, source, "OET1:B:S02")

    def test_read_scope_never_claims_independent_audit(self):
        path = OET1 / "sources/cc-train-episode-24719/GENERIC/SOURCE_AUTHOR_INPUT.json"
        receipt = read_scope_receipt([path])
        self.assertFalse(receipt["independent_review_completed"] or receipt["semantic_audit_pass"])
        self.assertTrue(receipt["artifacts"][0]["read_all_bytes"])
        self.assertEqual(receipt["artifacts"][0]["binding"], bind(path))

    def test_preflight_without_model_or_gpu(self):
        result = preflight("cc-train-episode-24719", "GENERIC")
        self.assertEqual(result["status"], "PASS_MECHANICAL_PREFLIGHT_ONLY")
        self.assertEqual(result["source_turn_count"], 188)
        self.assertFalse(result["model_call_executed"] or result["gpu_required"] or result["sandbox_connectivity_tested"])

    def test_frozen_output_directory_forbidden(self):
        self.assert_code("OUTPUT_OUTSIDE_NEW_REPAIR_DIRECTORY", owned_output, OET1 / "sources")
        self.assert_code("OUTPUT_OUTSIDE_NEW_REPAIR_DIRECTORY", owned_output, HERE)


class HistoricalReplayTransport:
    """TEST-ONLY CPU replay. Never creates a process or claims a real new call."""
    def __init__(self, run_dir, *, same_thread=False):
        self.run_dir = Path(run_dir)
        self.calls = 0
        self.same_thread = same_thread

    def __call__(self, call_id, prompt):
        self.calls += 1
        cdir = self.run_dir / "cli_calls" / call_id
        if (cdir / "RESULT.json").exists():
            result = read(cdir / "RESULT.json")
            return read(result["parsed_response"]["path"]), result
        historical = OET1 / "cli_calls" / call_id / "RESULT.json"
        original = read(historical)
        response = read(original["parsed_response"]["path"])
        attempt = cdir / "attempt_01"
        attempt.mkdir(parents=True)
        (attempt / "REQUEST.txt").write_text(prompt, encoding="utf-8")
        raw_text = Path(original["raw_response"]["path"]).read_text(encoding="utf-8")
        (attempt / "LAST_MESSAGE.txt").write_text(raw_text, encoding="utf-8")
        (attempt / "STDERR.log").write_text("TEST_ONLY_REPLAY_NO_MODEL_CALL\n", encoding="utf-8")
        thread = "TEST_FIXTURE_" + ("BOTH_ROLES" if self.same_thread else call_id.rsplit("__", 1)[1])
        events = [{"type": "thread.started", "thread_id": thread}, {"type": "turn.completed", "usage": {}}]
        (attempt / "EVENTS.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
        write(attempt / "START.json", {"request": bind(attempt / "REQUEST.txt"), "test_only_historical_replay": True})
        terminal = {"finished_at_utc": "2000-01-01T00:00:00+00:00", "returncode": 0, "timed_out": False,
                    "backend": {"thread_id": thread, "model": "TEST_ONLY_NO_MODEL", "model_provider": "CPU_FIXTURE_REPLAY"},
                    "test_only_historical_replay": True, "historical_response_binding": original["raw_response"]}
        for key, name in (("request", "REQUEST.txt"), ("raw_response", "LAST_MESSAGE.txt"),
                          ("events", "EVENTS.jsonl"), ("stderr", "STDERR.log")):
            terminal[key] = bind(attempt / name)
        write(attempt / "TERMINAL.json", terminal)
        parsed = write(cdir / "PARSED_RESPONSE.json", response)
        result = {**terminal, "terminal_binding": bind(attempt / "TERMINAL.json"),
                  "parsed_response": parsed, "response_received": True}
        write(cdir / "RESULT.json", result)
        return response, result


class PipelineRegression(unittest.TestCase):
    def test_v2_completion_guard_normal_and_three_failure_modes(self):
        with tempfile.TemporaryDirectory(prefix="TEST_ONLY_", dir=HERE) as temporary:
            transport = HistoricalReplayTransport(temporary)
            pipeline = ProductionPipelineV2(temporary, transport)
            result = pipeline.run_source("cc-train-episode-24719", "TARGETED", through_scene="OET1:A:S01")
            self.assertEqual(result["status"], "SOURCE_PREFIX_COMPLETE")
            self.assertTrue((Path(temporary) / "COMPLETION_PROTOCOL_V2.json").exists())
            sd = Path(temporary) / "sources/cc-train-episode-24719/TARGETED/OET1_A_S01"
            result = read(sd / "AUTHOR_RESULT.json")
            events = {"turn_completed": True, "thread_id": result["backend"]["thread_id"]}
            self.assertEqual(validate_completion(result, events)["status"], "PASS_BACKEND_NORMAL_COMPLETION")
            terminal = read(result["terminal_binding"]["path"])
            for field, value, code in (("returncode", 1, "BACKEND_NONZERO_OR_UNKNOWN_EXIT"),
                                       ("timed_out", True, "BACKEND_TIMED_OUT_OR_UNKNOWN")):
                changed = deepcopy(result)
                changed[field] = value
                terminal_copy = deepcopy(terminal)
                terminal_copy[field] = value
                path = Path(temporary) / f"negative_{field}.json"
                write(path, terminal_copy)
                changed["terminal_binding"] = bind(path)
                with self.assertRaises(ProductionError) as caught:
                    validate_completion(changed, events)
                self.assertEqual(caught.exception.code, code)
            with self.assertRaises(ProductionError) as caught:
                validate_completion(result, {"turn_completed": False})
            self.assertEqual(caught.exception.code, "BACKEND_TURN_NOT_COMPLETED")
            changed = deepcopy(result)
            changed["timed_out"] = 0
            with self.assertRaises(ProductionError) as caught:
                validate_completion(changed, events)
            self.assertEqual(caught.exception.code, "RESULT_TERMINAL_COMPLETION_MISMATCH")

    def test_accepted_resume_rechecks_real_call_provenance(self):
        with tempfile.TemporaryDirectory(prefix="TEST_ONLY_", dir=HERE) as temporary:
            transport = HistoricalReplayTransport(temporary)
            pipeline = ProductionPipeline(temporary, transport)
            pipeline.run_source("cc-train-episode-24719", "TARGETED", through_scene="OET1:A:S01")
            sd = Path(temporary) / "sources/cc-train-episode-24719/TARGETED/OET1_A_S01"
            raw_path = Path(read(sd / "REVIEW_RESULT.json")["raw_response"]["path"])
            raw_path.write_text(raw_path.read_text() + " ")
            with self.assertRaises(AssertionError):
                pipeline.run_source("cc-train-episode-24719", "TARGETED", through_scene="OET1:A:S01")
            self.assertEqual(transport.calls, 2)

    def test_same_actual_author_reviewer_thread_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="TEST_ONLY_", dir=HERE) as temporary:
            transport = HistoricalReplayTransport(temporary, same_thread=True)
            result = ProductionPipeline(temporary, transport).run_source("cc-train-episode-24719", "TARGETED",
                                                                         through_scene="OET1:A:S01")
            self.assertEqual(result["status"], "STOP_FAILURE")
            failure = read(Path(temporary) / "sources/cc-train-episode-24719/TARGETED/OET1_A_S01/FAILURE.json")
            self.assertEqual(failure["error_code"], "SOURCE_SELF_REVIEW")

    def test_resume_after_validation_before_acceptance(self):
        with tempfile.TemporaryDirectory(prefix="TEST_ONLY_", dir=HERE) as temporary:
            transport = HistoricalReplayTransport(temporary)
            pipeline = ProductionPipeline(temporary, transport)
            import run_production
            original_write = run_production.write
            def interrupt_before_acceptance(path, value):
                if Path(path).name == "ACCEPTED.json":
                    raise KeyboardInterrupt("TEST_ONLY_PROCESS_INTERRUPTION")
                return original_write(path, value)
            with patch("run_production.write", side_effect=interrupt_before_acceptance):
                with self.assertRaises(KeyboardInterrupt):
                    pipeline.run_source("cc-train-episode-24719", "TARGETED", through_scene="OET1:A:S01")
            sd = Path(temporary) / "sources/cc-train-episode-24719/TARGETED/OET1_A_S01"
            self.assertTrue((sd / "REVIEW_VALIDATION.json").exists())
            self.assertFalse((sd / "FAILURE.json").exists())
            validation_before = bind(sd / "REVIEW_VALIDATION.json")
            result = pipeline.run_source("cc-train-episode-24719", "TARGETED", through_scene="OET1:A:S01")
            self.assertEqual(result["status"], "SOURCE_PREFIX_COMPLETE")
            self.assertEqual(bind(sd / "REVIEW_VALIDATION.json"), validation_before)

    def test_shared_entrypoint_acceptance_and_resume_with_historical_replay(self):
        with tempfile.TemporaryDirectory(prefix="TEST_ONLY_", dir=HERE) as temporary:
            transport = HistoricalReplayTransport(temporary)
            pipeline = ProductionPipeline(temporary, transport)
            result = pipeline.run_source("cc-train-episode-24719", "TARGETED", through_scene="OET1:A:S01")
            self.assertEqual(result["status"], "SOURCE_PREFIX_COMPLETE")
            self.assertEqual(transport.calls, 2)
            resumed = pipeline.run_source("cc-train-episode-24719", "TARGETED", through_scene="OET1:A:S01")
            self.assertEqual(resumed["status"], "SOURCE_PREFIX_COMPLETE")
            self.assertEqual(transport.calls, 2)
            sd = Path(temporary) / "sources/cc-train-episode-24719/TARGETED/OET1_A_S01"
            receipt = read(sd / "ACCEPTED.json")
            self.assertEqual(receipt["status"], "READY")
            self.assertTrue(read(sd / "REVIEW_VALIDATION.json")["semantic_admitted"])
            request = read(sd / "REVIEW_RESULT.json")["request"]["path"]
            self.assertIn('"mechanical_prefix_and_count_proof"', Path(request).read_text())
            self.assertIn('"native_quote_domain"', Path(request).read_text())

    def test_shared_entrypoint_real_semantic_failure_stops_without_retry(self):
        with tempfile.TemporaryDirectory(prefix="TEST_ONLY_", dir=HERE) as temporary:
            transport = HistoricalReplayTransport(temporary)
            pipeline = ProductionPipeline(temporary, transport)
            result = pipeline.run_source("cc-train-episode-29753", "GENERIC", through_scene="OET1:A:S01")
            self.assertEqual(result["status"], "STOP_FAILURE")
            self.assertEqual(transport.calls, 2)
            sd = Path(temporary) / "sources/cc-train-episode-29753/GENERIC/OET1_A_S01"
            failure = read(sd / "FAILURE.json")
            self.assertEqual(failure["error_code"], "REVIEW_SEMANTIC_ADMISSION_REJECTED")
            self.assertFalse(failure["semantic_retry"])
            self.assertFalse((sd / "ACCEPTED.json").exists())
            self.assertEqual(read(sd / "REVIEW.json")["decision"], "HOLD")
            resumed = pipeline.run_source("cc-train-episode-29753", "GENERIC", through_scene="OET1:A:S01")
            self.assertEqual(resumed["status"], "STOP_EXISTING_FAILURE_NO_RETRY")
            self.assertEqual(transport.calls, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if args.report:
        output = owned_output(args.report)
        snapshot = {"before": FROZEN_BEFORE, "after": FROZEN_AFTER,
                    "unchanged": FROZEN_BEFORE == FROZEN_AFTER}
        manifest_path = output.with_name(output.stem + "_FROZEN_INPUT_MANIFEST.json")
        write(manifest_path, snapshot)
        report = {"schema_version": "oet1_followup_production_repair_report_v1",
                  "status": "PASS_CPU_REGRESSION_ONLY" if result.wasSuccessful() else "FAIL_REGRESSION",
                  "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
                  "elapsed_seconds": round(time.monotonic() - started, 3),
                  "model_calls": 0, "gpu_calls": 0, "new_production_dialogues": 0, "new_production_questions": 0,
                  "integration_transport": "TEMPORARY_TEST_ONLY_HISTORICAL_RESPONSE_REPLAY_NOT_REAL_MODEL_EXECUTION",
                  "frozen_historical_files_checked": len(FROZEN_BEFORE or []),
                  "frozen_historical_files_unchanged": FROZEN_BEFORE == FROZEN_AFTER,
                  "historical_manifest": bind(manifest_path),
                  "entrypoint": bind(HERE / "run_production.py"), "shared_core": bind(HERE / "production_core.py"),
                  "future_entrypoint_v2": bind(HERE / "run_production_v2.py"),
                  "completion_guard": bind(HERE / "completion_guard.py"),
                  "tests": bind(__file__),
                  "fixed": ["SAFE_RESERVED_LOG_FIELDS", "EXACT_CUMULATIVE_COUNTS_BEFORE_REVIEW",
                            "SHARED_HASH_BOUND_NATIVE_UTTERANCE_RESOLVER", "STRICT_UNMODIFIED_REVIEW_READ_COUNTS",
                            "SEMANTIC_FAIL_HOLD_REMAIN_REJECTED", "EXPLICIT_BOOL_VS_COUNT_SCHEMAS",
                            "MECHANICAL_SCOPE_RECEIPTS_NEVER_CLAIM_INDEPENDENT_AUDIT",
                            "V2_REQUIRES_ZERO_EXIT_NO_TIMEOUT_AND_TURN_COMPLETED"],
                  "historical_results_rewritten": False, "frozen_stop_gate_bypassed": False,
                  "limitations": ["New author/review prompts define a NEW protocol; cannot retroactively change OET1 results.",
                                  "No actual model/sandbox connectivity or production stability is established by CPU replay.",
                                  "Source runner is integrated; question generation and evaluation are outside this repair.",
                                  "Three actual semantic failures and 260/258 review mismatch still require rejection.",
                                  "Independent closeout semantic audit remains uncompleted when sandbox launch is blocked."]}
        write(output, report)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
