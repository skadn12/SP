"""Formatter entry checks using private temporary ledgers and mocked HTTP only."""
from copy import deepcopy
import json
import unittest

from .fixtures import example_budget
from .prompts import TARGET_RULES, render_initial_planner
from .root_scope import prepare_bundle, render_for_bundle, with_supplement_formatting
from .run import offline_network_guard
from .supplement_format import VERSION
from .test_readiness import Response
from .test_root_entry import RootCliHarness, mixed_root_patch
from .test_root_planning import root_fixture_bundle
from .test_supplement_alignment import read_json
from ..h4_round1_r03_feedback_view_v2.templates import PLANNER_DEVELOPMENT_GUIDANCE_V1


class FormattingRootCliHarness(RootCliHarness):
    def __init__(self, *, use_formatter=True):
        self.use_formatter = use_formatter

    def __enter__(self):
        super().__enter__()
        self.original_bundle = deepcopy(self.bundle)
        if self.use_formatter:
            self.bundle = with_supplement_formatting(self.bundle)
            self.pin_fixture_approval()
        return self

    def http_open(self, request, **kwargs):
        actual = self.requests[-1]
        if actual["role"] != "supplement":
            return super().http_open(request, **kwargs)
        payload = json.loads(request.data)
        self.http_payloads.append(payload)
        assert request.full_url == self.model["endpoint"]
        assert payload["messages"] == actual["messages"]
        content = mixed_root_patch(self.bundle)
        content["phases"][0]["phase_id"] = "phase:source:followup"
        row = content["new_developments"][0]
        row["phase_id"] = "phase:new:independent"
        for field in ("before_state", "required_transition", "after_state",
                      "interaction_beats", "actor_bindings"):
            row[field] = row["candidate"].pop(field)
        self.raw_supplement = json.dumps(content)
        if self.case == "duplicate_key":
            self.raw_supplement = self.raw_supplement.replace(
                '"status": "PATCH"', '"status": "PATCH", "status": "PATCH"', 1)
        elif self.case == "overflow_number":
            marker = "FORMATTER_OVERFLOW_PROBE"
            content["new_developments"][0]["candidate"]["source_basis"] = [marker]
            self.raw_supplement = json.dumps(content).replace(json.dumps(marker), "1e999")
        return Response(json.dumps({"choices": [{"finish_reason": "stop", "message": {
            "content": self.raw_supplement}}]}).encode())


class SupplementFormatEntryTests(unittest.TestCase):
    def test_default_initial_restores_legacy_output_and_preserves_guidance(self):
        with offline_network_guard():
            bundle = root_fixture_bundle()
            view = bundle["shared_feedback_view"]
            kwargs = {
                "record": {**deepcopy(view), "performance": deepcopy(view["solver_observations"])},
                "dialogue": {"root_id": bundle["parent"]["root_id"], "turns": [
                    {"turn_ref": row["turn_id"], "speaker": row["speaker"], "text": row["text"]}
                    for row in bundle["parent"]["dialogue"]["turns"]]},
                "memory_candidates": bundle["candidates"],
                "verified_tracks": bundle["verified_tracks"],
            }
            default = render_initial_planner(**kwargs)
            explicit = render_initial_planner(planner_contract_version="legacy_initial_v1", **kwargs)
        self.assertEqual(default, explicit)
        self.assertEqual(default["prompt_version"], "incremental_retention_v1.initial_planner")
        self.assertNotIn("planner_contract_identity", default)
        self.assertEqual(default["prompt"].count(PLANNER_DEVELOPMENT_GUIDANCE_V1), 1)
        self.assertEqual(default["prompt"].count(TARGET_RULES), 1)
        self.assertIn('"selected_continuations"', default["prompt"])
        self.assertIn('"new_development_candidates"', default["prompt"])
        self.assertNotIn('"source_candidate_ref"', default["prompt"])

    def test_formatter_selection_changes_metadata_only_and_keeps_messages_exact(self):
        with offline_network_guard():
            bundle = root_fixture_bundle()
            before = deepcopy(bundle)
            formatted_bundle = with_supplement_formatting(bundle)
            expected = deepcopy(before)
            expected["supplement_format_version"] = VERSION
            self.assertEqual(formatted_bundle, expected)
            self.assertEqual(bundle, before)
            prepared = prepare_bundle(bundle)
            formatted_prepared = prepare_bundle(formatted_bundle)
            self.assertEqual(formatted_prepared, prepared)
            original_prompt = render_for_bundle(bundle, example_budget(), prepared)
            formatted_prompt = render_for_bundle(formatted_bundle, example_budget(), formatted_prepared)
        self.assertEqual(formatted_prompt, original_prompt)
        self.assertNotIn(VERSION, json.dumps(formatted_prompt["messages"]))

    def test_mock_http_formats_then_runs_gate_realizer_and_adherence(self):
        with FormattingRootCliHarness() as harness:
            bundle_before = deepcopy(harness.bundle)
            original_before = deepcopy(harness.original_bundle)
            report = harness.invoke()
            self.assertEqual(report["status"], "EXECUTION_ENDED_GOALS_RECORDED_SEPARATELY")
            self.assertEqual(report["retained_scene_count"], 8)
            self.assertEqual(report["attempted_request_counts"], {
                "planner": 0, "supplement": 1, "gate": 8, "realizer": 4,
                "adherence": 4, "closure": 0})
            self.assertEqual(len(harness.http_payloads), 17)
            run_dir = harness.root / "MOCK_run"
            raw = read_json(run_dir / "responses/001_supplement.json")
            formatting = read_json(run_dir / "supplement_formatting.json")
            formatted = read_json(run_dir / "formatted_patch.json")
            merged = read_json(run_dir / "increment_patch_and_diff.json")
            self.assertIsInstance(raw, str)
            self.assertEqual(raw, harness.raw_supplement)
            self.assertEqual(formatting["raw_output"], raw)
            self.assertEqual(merged["raw_patch"], raw)
            self.assertTrue(formatting["valid"], formatting)
            self.assertEqual(formatting["formatted_output"], formatted)
            self.assertEqual(merged["formatted_patch"], formatted)
            self.assertEqual(len(formatting["transformations"]), 7)
            self.assertEqual(sum(row["operation"] == "move_candidate_field"
                                 for row in formatting["transformations"]), 5)
            self.assertEqual(merged["merged"]["original_plan"], original_before["original_plan"])
            self.assertEqual(merged["diff"]["original_candidate_changes"], [])
            self.assertEqual(merged["diff"]["original_goal_changes"], [])
            self.assertEqual(harness.bundle, bundle_before)
            self.assertEqual(harness.original_bundle, original_before)

    def test_duplicate_keys_survive_transport_and_are_rejected_before_gate(self):
        with FormattingRootCliHarness() as harness:
            report = harness.invoke(case="duplicate_key")
            self.assertEqual(report["status"], "SUPPLEMENT_INVALID_STOP")
            self.assertEqual([row["role"] for row in harness.requests], ["supplement"])
            run_dir = harness.root / "MOCK_run"
            self.assertEqual(read_json(run_dir / "responses/001_supplement.json"), harness.raw_supplement)
            self.assertEqual(harness.raw_supplement.count('"status": "PATCH"'), 2)
            formatting = read_json(run_dir / "supplement_formatting.json")
            self.assertEqual(formatting["raw_output"], harness.raw_supplement)
            self.assertFalse(formatting["valid"])
            self.assertTrue(any("duplicate_json_key:status" in error for error in formatting["errors"]))
            self.assertFalse((run_dir / "formatted_patch.json").exists())

    def test_unknown_formatter_version_is_rejected_before_claim_or_request(self):
        for value in (None, "root_supplement_formatting_unknown"):
            with self.subTest(version=value), FormattingRootCliHarness() as harness:
                harness.bundle["supplement_format_version"] = value
                with self.assertRaisesRegex(ValueError, "supplement_formatter_requires_explicit_original_root_contract"):
                    harness.invoke()
                harness.assert_unclaimed()

    def test_legacy_entry_still_rejects_same_raw_format_without_adapter_selection(self):
        with FormattingRootCliHarness(use_formatter=False) as harness:
            report = harness.invoke()
            self.assertEqual(report["status"], "SUPPLEMENT_INVALID_STOP")
            self.assertEqual([row["role"] for row in harness.requests], ["supplement"])
            run_dir = harness.root / "MOCK_run"
            result = read_json(run_dir / "increment_patch_and_diff.json")
            self.assertIn("patch_phase_1_identity", result["errors"])
            self.assertIn("root_new_development_1_shape", result["errors"])
            self.assertFalse((run_dir / "supplement_formatting.json").exists())
            self.assertFalse((run_dir / "formatted_patch.json").exists())

    def test_overflow_number_is_rejected_before_it_can_enter_gate_content(self):
        with FormattingRootCliHarness() as harness:
            report = harness.invoke(case="overflow_number")
            self.assertEqual(report["status"], "SUPPLEMENT_INVALID_STOP")
            self.assertEqual([row["role"] for row in harness.requests], ["supplement"])
            run_dir = harness.root / "MOCK_run"
            formatting = read_json(run_dir / "supplement_formatting.json")
            self.assertFalse(formatting["valid"])
            self.assertEqual(formatting["raw_output"], harness.raw_supplement)
            self.assertFalse((run_dir / "formatted_patch.json").exists())


if __name__ == "__main__":
    unittest.main()
