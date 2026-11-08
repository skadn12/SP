#!/usr/bin/env python3
"""V3 CPU regression: old suite + whitespace, version binding and CLI checks.

No model calls or GPU work. All replay/mutation files are TEST_ONLY temporary
controller fixtures and are removed after each test.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

sys.dont_write_bytecode = True
import test_repair as legacy
import production_core_v3 as core
from run_production_v3 import HERE, OET1, ProductionPipeline, owned_output

BEFORE = None
AFTER = None
PRESERVED_BEFORE = None
PRESERVED_AFTER = None


def specified_historical_snapshot():
    expected = core.read(HERE / "FROZEN_INPUT_MANIFEST.json")["before"]
    actual = [core.bind(item["path"]) for item in expected]
    core.require(actual == expected, "SPECIFIED_HISTORICAL_MANIFEST_CHANGED")
    return actual


def preserved_v1_v2_snapshot():
    names = ["production_core.py", "run_production.py", "run_production_v2.py", "completion_guard.py",
             "test_repair.py", "REPAIR_REPORT.json", "REPAIR_REPORT_V2.json", "FROZEN_INPUT_MANIFEST.json",
             "REPAIR_REPORT_V2_FROZEN_INPUT_MANIFEST.json", "LIVE_PREFLIGHT_24719_GENERIC.json",
             "LIVE_PROBE_REPORT.json", "LIVE_PROBE_24719_GENERIC.log"]
    paths = [HERE / name for name in names]
    paths.extend(path for path in (HERE / "live_probe_24719_generic_v1").rglob("*") if path.is_file())
    return [core.bind(path) for path in sorted(paths)]


def setUpModule():
    global BEFORE, PRESERVED_BEFORE
    BEFORE = specified_historical_snapshot()
    PRESERVED_BEFORE = preserved_v1_v2_snapshot()


def tearDownModule():
    global AFTER, PRESERVED_AFTER
    AFTER = specified_historical_snapshot()
    PRESERVED_AFTER = preserved_v1_v2_snapshot()
    if BEFORE != AFTER or PRESERVED_BEFORE != PRESERVED_AFTER:
        raise AssertionError("SPECIFIED_PROTECTED_FILES_CHANGED")


class WhitespaceReviewReplay(legacy.HistoricalReplayTransport):
    """Deliberately invalid TEST_ONLY response with coherent fixture provenance."""
    def __init__(self, run_dir, quote):
        super().__init__(run_dir)
        self.quote = quote

    def __call__(self, call_id, prompt):
        response, result = super().__call__(call_id, prompt)
        if not call_id.endswith("__REVIEW"):
            return response, result
        response = deepcopy(response)
        response["checks"]["time"]["evidence"][0]["quote"] = self.quote
        Path(result["raw_response"]["path"]).write_text(json.dumps(response), encoding="utf-8")
        Path(result["parsed_response"]["path"]).write_text(json.dumps(response), encoding="utf-8")
        terminal = core.read(result["terminal_binding"]["path"])
        result["raw_response"] = core.bind(result["raw_response"]["path"])
        result["parsed_response"] = core.bind(result["parsed_response"]["path"])
        terminal["raw_response"] = result["raw_response"]
        terminal["test_only_deliberate_invalid_quote_mutation"] = True
        Path(result["terminal_binding"]["path"]).write_text(json.dumps(terminal), encoding="utf-8")
        result["terminal_binding"] = core.bind(result["terminal_binding"]["path"])
        result["test_only_deliberate_invalid_quote_mutation"] = True
        Path(self.run_dir / "cli_calls" / call_id / "RESULT.json").write_text(json.dumps(result), encoding="utf-8")
        return response, result


class V3Regression(unittest.TestCase):
    def positive(self):
        return legacy.fixture(24719, "GENERIC", "OET1:B:S02", raw=True)

    def assert_code(self, code, function, *args, **kwargs):
        with self.assertRaises(core.ProductionError) as caught:
            function(*args, **kwargs)
        self.assertEqual(caught.exception.code, code)
        return caught.exception

    def test_empty_space_newline_tab_and_unicode_whitespace_rejected(self):
        for quote in ("", " ", "   ", "\n", "\r\n\t  ", "\u00a0", "\u3000"):
            with self.subTest(quote=repr(quote)):
                packet, source, response = self.positive()
                response["checks"]["time"]["evidence"][0]["quote"] = quote
                self.assert_code("EMPTY_EVIDENCE_QUOTE", core.validate_review,
                                 response, packet, source, "OET1:B:S02")
                self.assertEqual(response["checks"]["time"]["evidence"][0]["quote"], quote)

    def test_nonempty_exact_historical_quotes_pass_unchanged(self):
        packet, source, response = self.positive()
        original = deepcopy(response)
        receipt = core.validate_review(response, packet, source, "OET1:B:S02")
        self.assertTrue(receipt["semantic_admitted"])
        self.assertEqual(response, original)
        self.assertEqual(receipt["schema_version"], "oet1_followup_review_validation_v3")
        self.assertEqual(receipt["checker_binding"], core.bind(HERE / "production_core_v3.py"))

    def test_nonempty_quote_with_spaces_is_exact_and_not_trimmed(self):
        packet, source, response = self.positive()
        evidence = next(e for check in response["checks"].values() for e in check["evidence"]
                        if e["turn_ref"] == "assigned_native_event.native_record.turns[7]")
        evidence["quote"] = " leaves "
        receipt = core.validate_review(response, packet, source, "OET1:B:S02")
        native = next(item for item in receipt["checked_quotes"] if item["domain"] == "assigned_native_record")
        self.assertEqual(native["quote_sha256"], hashlib.sha256(b" leaves ").hexdigest())
        self.assertEqual(evidence["quote"], " leaves ")

    def test_adding_nonexistent_surrounding_spaces_does_not_pass_by_trimming(self):
        packet, source, response = self.positive()
        evidence = next(e for check in response["checks"].values() for e in check["evidence"]
                        if e["turn_ref"] == "assigned_native_event.native_record.turns[7]")
        evidence["quote"] = " The TR0254 leaves Cambridge at 17:21. "
        self.assert_code("QUOTE_NOT_EXACT_CONTIGUOUS_SUBSTRING", core.validate_review,
                         response, packet, source, "OET1:B:S02")

    def test_real_frames_state_count_and_content_failures_still_rejected(self):
        cases = [(31457, "GENERIC", "OET1:B:S02", True, "QUOTE_NOT_EXACT_CONTIGUOUS_SUBSTRING"),
                 (29753, "TARGETED", "OET1:B:S03", True, "REVIEW_SOURCE_COUNT_MISMATCH"),
                 (25652, "GENERIC", "OET1:A:S03", False, "REVIEW_SEMANTIC_ADMISSION_REJECTED"),
                 (29753, "GENERIC", "OET1:A:S01", False, "REVIEW_SEMANTIC_ADMISSION_REJECTED"),
                 (31457, "TARGETED", "OET1:A:S02", False, "REVIEW_SEMANTIC_ADMISSION_REJECTED")]
        for root, arm, scene, raw, code in cases:
            with self.subTest(root=root, arm=arm, scene=scene):
                packet, source, response = legacy.fixture(root, arm, scene, raw=raw)
                self.assert_code(code, core.validate_review, response, packet, source, scene)

    def test_v3_pipeline_rejects_whitespace_preserves_raw_and_stops(self):
        for quote in (" ", "\n\t"):
            with self.subTest(quote=repr(quote)), tempfile.TemporaryDirectory(prefix="TEST_ONLY_V3_", dir=HERE) as temporary:
                transport = WhitespaceReviewReplay(temporary, quote)
                pipeline = ProductionPipeline(temporary, transport)
                result = pipeline.run_source("cc-train-episode-24719", "TARGETED", through_scene="OET1:A:S01")
                self.assertEqual(result["status"], "STOP_FAILURE")
                sd = Path(temporary) / "sources/cc-train-episode-24719/TARGETED/OET1_A_S01"
                self.assertEqual(core.read(sd / "FAILURE.json")["error_code"], "EMPTY_EVIDENCE_QUOTE")
                self.assertEqual(core.read(sd / "REVIEW.json")["checks"]["time"]["evidence"][0]["quote"], quote)
                self.assertFalse((sd / "ACCEPTED.json").exists())
                self.assertEqual(pipeline.run_source("cc-train-episode-24719", "TARGETED", through_scene="OET1:A:S01")["status"],
                                 "STOP_EXISTING_FAILURE_NO_RETRY")
                self.assertEqual(transport.calls, 2)

    def test_v3_pipeline_passes_exact_replay_and_binds_v3_core_hash(self):
        with tempfile.TemporaryDirectory(prefix="TEST_ONLY_V3_", dir=HERE) as temporary:
            transport = legacy.HistoricalReplayTransport(temporary)
            pipeline = ProductionPipeline(temporary, transport)
            result = pipeline.run_source("cc-train-episode-24719", "TARGETED", through_scene="OET1:A:S01")
            self.assertEqual(result["status"], "SOURCE_PREFIX_COMPLETE")
            protocol = core.read(Path(temporary) / "PROTOCOL_V3.json")
            for name in ("run_production_v3.py", "production_core_v3.py", "completion_guard_v3.py"):
                self.assertIn(core.bind(HERE / name), protocol["code_bindings"])
            self.assertTrue(protocol["evidence_quote_requires_nonwhitespace_character"])
            self.assertFalse(protocol["evidence_quote_stripped_or_rewritten"])
            self.assertFalse((Path(temporary) / "PROTOCOL.json").exists())
            self.assertEqual(pipeline.run_source("cc-train-episode-24719", "TARGETED", through_scene="OET1:A:S01")["status"],
                             "SOURCE_PREFIX_COMPLETE")
            self.assertEqual(transport.calls, 2)

    def test_v3_cannot_open_old_protocol_directory(self):
        for protocol_name in ("PROTOCOL.json", "COMPLETION_PROTOCOL_V2.json"):
            with self.subTest(protocol=protocol_name), tempfile.TemporaryDirectory(prefix="TEST_ONLY_V3_", dir=HERE) as temporary:
                core.write(Path(temporary) / protocol_name, {"test_only": True})
                transport = legacy.HistoricalReplayTransport(temporary)
                pipeline = ProductionPipeline(temporary, transport)
                self.assert_code("OLD_PROTOCOL_RUN_DIRECTORY_REQUIRES_NEW_V3_DIRECTORY", pipeline.run_source,
                                 "cc-train-episode-24719", "TARGETED", through_scene="OET1:A:S01")
                self.assertEqual(transport.calls, 0)
                self.assertFalse((Path(temporary) / "PROTOCOL_V3.json").exists())

    def test_v3_readonly_cli_uses_v3_validation(self):
        packet, source, response = self.positive()
        with tempfile.TemporaryDirectory(prefix="TEST_ONLY_V3_", dir=HERE) as temporary:
            directory = Path(temporary)
            paths = {}
            for name, value in (("packet", packet), ("source", source), ("review", response)):
                paths[name] = directory / (name + ".json")
                core.write(paths[name], value)
            command = [sys.executable, "-B", str(HERE / "run_production_v3.py"), "inspect-review",
                       "--packet", str(paths["packet"]), "--source", str(paths["source"]),
                       "--review", str(paths["review"]), "--scene", "OET1:B:S02"]
            positive = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(positive.returncode, 0, positive.stdout + positive.stderr)
            self.assertEqual(json.loads(positive.stdout)["checker_binding"], core.bind(HERE / "production_core_v3.py"))
            response["checks"]["time"]["evidence"][0]["quote"] = " "
            paths["review"].write_text(json.dumps(response), encoding="utf-8")
            negative = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(negative.returncode, 1)
            self.assertEqual(json.loads(negative.stdout)["error_code"], "EMPTY_EVIDENCE_QUOTE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    output = owned_output(args.report)
    started = time.monotonic()
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(legacy),
                               unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    groups = Counter("OET1_top_level_direct_files" if Path(item["path"]).parent == OET1
                     else Path(item["path"]).relative_to(OET1).parts[0] for item in BEFORE or [])
    manifest = {"scope": "Only exact file paths listed in FROZEN_INPUT_MANIFEST.json before; not all of OET1",
                "historical_before": BEFORE, "historical_after": AFTER,
                "v1_v2_and_live_before": PRESERVED_BEFORE, "v1_v2_and_live_after": PRESERVED_AFTER}
    manifest_path = output.with_name(output.stem + "_SPECIFIED_FILES_MANIFEST.json")
    core.write(manifest_path, manifest)
    report = {"schema_version": "oet1_followup_whitespace_quote_repair_report_v3",
              "status": "PASS_CPU_REGRESSION_ONLY" if result.wasSuccessful() else "FAIL_REGRESSION",
              "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
              "legacy_regression_tests": 29, "v3_additional_tests": result.testsRun - 29,
              "elapsed_seconds": round(time.monotonic() - started, 3),
              "model_calls": 0, "gpu_calls": 0, "new_production_sources": 0, "new_production_questions": 0,
              "entrypoint_version": "v3", "entrypoint": core.bind(HERE / "run_production_v3.py"),
              "bound_core": core.bind(HERE / "production_core_v3.py"),
              "bound_completion_guard": core.bind(HERE / "completion_guard_v3.py"),
              "tests": core.bind(__file__), "readme": core.bind(HERE / "README_ZH.md"),
              "quote_requirement": "isinstance(quote, str) and bool(quote.strip()); validate original quote in text without modifying it",
              "new_run_protocol_filename": "PROTOCOL_V3.json", "old_run_directory_reuse_allowed": False,
              "specified_historical_paths_checked": len(BEFORE or []),
              "specified_historical_path_groups": dict(groups), "specified_historical_paths_unchanged": BEFORE == AFTER,
              "specified_historical_scope": "OET1 top-level direct files + recursive inputs/sources/cli_calls/scripts, exact 873-path existing manifest only",
              "whole_oet1_immutability_claimed": False,
              "other_directories_not_covered": ["training", "questions", "question_reviews", "closeout_cli",
                                                 "finite_finish_logs_v3", "followup_repair_v1", "wave_results", "wave_results_v4"],
              "v1_v2_code_reports_and_live_paths_checked": len(PRESERVED_BEFORE or []),
              "v1_v2_code_reports_and_live_paths_unchanged": PRESERVED_BEFORE == PRESERVED_AFTER,
              "specified_files_manifest": core.bind(manifest_path),
              "historical_results_reclassified": False,
              "limitations": ["CPU-only controller and read-only CLI regressions; no new production stability claim.",
                              "Whitespace rejection does not fix citation selection mistakes or semantic source failures.",
                              "The specified historical manifest excludes unlisted OET1 subdirectories."]}
    core.write(output, report)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
