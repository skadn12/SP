from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.backends import BackendResponse
from scripts.current_method.restart_n5_spell.core import atomic_json, sha256_file
from scripts.current_method.restart_n5_spell.ledger import RequestLedger
from scripts.current_method.restart_n5_spell.lineage_three_stage_canary import (
    ApiBudget,
    run_single_object_task,
)


class _Backend:
    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.requests = []
        config = type("Config", (), {
            "model": "fixture-model",
            "base_url": "fixture://local",
            "temperature": 0.0,
        })()
        self.client = type("Client", (), {"config": config})()
        self.max_tokens = 1024

    def generate_batch(self, request):
        self.requests.append(request)
        return BackendResponse(
            next(self._responses),
            "fixture-model",
            {"usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
            }},
        )


def _lineage_view() -> dict:
    return {
        "combined_dialogue": [{
            "turn_id": "P001",
            "session_id": "parent-session",
            "speaker": "Alice",
            "text": "Alice submitted the report.",
        }],
        "span_catalog": {
            "P001#S0": {
                "turn_id": "P001",
                "text": "Alice submitted the report.",
            },
        },
        "view_sha256": "fixture-view-sha256",
    }


class SingleObjectTaskBudgetTest(unittest.TestCase):
    def _ledger(self, root: Path) -> RequestLedger:
        manifest = root / "config/run_manifest.json"
        atomic_json(manifest, {"run_id": "single-object-task-budget-test"})
        return RequestLedger(root, sha256_file(manifest), "mock")

    def test_custom_limits_label_and_format_retry_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = self._ledger(root)
            malformed = "MALFORMED_RAW_MUST_NOT_REENTER_PROMPT {"
            accepted = json.dumps({"value": "ok"})
            backend = _Backend([malformed, accepted])

            result = run_single_object_task(
                run_root=root,
                ledger=ledger,
                backend=backend,
                budget=ApiBudget(
                    normal_caps={"single_item_questioner": 1},
                    format_retry_cap=1,
                    physical_cap=2,
                ),
                stage="single_item_questioner",
                logical_request_id="single-item-question",
                semantic_prompt="Generate one core QA JSON object.",
                lineage_view=_lineage_view(),
                validator=lambda raw: {
                    "decision": "single_item_questioner_accept",
                    "value": raw["value"],
                },
                schema_retry_decisions=(
                    "single_item_questioner_schema_invalid",
                ),
                ledger_budget_limit=7,
                ledger_total_budget_limit=9,
                json_invalid_decision="single_item_questioner_json_invalid",
            )

            self.assertTrue(result["accepted"])
            self.assertEqual(result["physical_request_count"], 2)
            self.assertTrue(result["format_retry_used"])
            self.assertEqual(
                result["attempts"][0]["validation"]["decision"],
                "single_item_questioner_json_invalid",
            )
            self.assertNotIn(malformed, backend.requests[1].prompt)
            self.assertNotIn("MALFORMED_RAW_MUST_NOT_REENTER_PROMPT",
                             backend.requests[1].prompt)

            for request_id in (
                    "single-item-question-a00", "single-item-question-a01"):
                state = ledger.read(request_id)
                self.assertEqual(state["budget_limit"], 7)
                self.assertEqual(state["total_budget_limit"], 9)
            retry_state = ledger.read("single-item-question-a01")
            self.assertEqual(
                retry_state["request_metadata"]["retry_mode"],
                "full_object_regeneration",
            )

    def test_default_ledger_limits_remain_twenty_four(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = self._ledger(root)
            backend = _Backend([json.dumps({"value": "ok"})])

            result = run_single_object_task(
                run_root=root,
                ledger=ledger,
                backend=backend,
                budget=ApiBudget(
                    normal_caps={"qa_writer": 1},
                    format_retry_cap=0,
                    physical_cap=1,
                ),
                stage="qa_writer",
                logical_request_id="legacy-defaults",
                semantic_prompt="Generate one object.",
                lineage_view=_lineage_view(),
                validator=lambda raw: {
                    "decision": "qa_accept",
                    "value": raw["value"],
                },
                schema_retry_decisions=("qa_schema_invalid",),
            )

            self.assertTrue(result["accepted"])
            state = ledger.read("legacy-defaults-a00")
            self.assertEqual(state["budget_limit"], 24)
            self.assertEqual(state["total_budget_limit"], 24)

    def test_directed_retry_suffix_factory_replaces_default_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = self._ledger(root)
            backend = _Backend([
                json.dumps({"value": "too_many_anchors"}),
                json.dumps({"value": "ok"}),
            ])
            result = run_single_object_task(
                run_root=root, ledger=ledger, backend=backend,
                budget=ApiBudget(normal_caps={"qa_writer": 1},
                                 format_retry_cap=1, physical_cap=2),
                stage="qa_writer", logical_request_id="directed-retry",
                semantic_prompt="Planner overview JSON.",
                lineage_view=_lineage_view(),
                validator=lambda raw: {
                    "decision": "qa_accept" if raw["value"] == "ok" else
                    "qa_schema_invalid"},
                schema_retry_decisions=("qa_schema_invalid",),
                retry_prompt_suffix_factory=lambda raw, validation: (
                    "DIRECTED ANCHOR REPAIR: previous=" + raw["value"]),
            )
            self.assertTrue(result["accepted"])
            self.assertIn("DIRECTED ANCHOR REPAIR: previous=too_many_anchors",
                          backend.requests[1].prompt)
            retry = ledger.read("directed-retry-a01")
            self.assertEqual("directed_schema_repair",
                             retry["request_metadata"]["retry_mode"])

    def test_explicit_three_attempt_budget_reaches_third_response(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = self._ledger(root)
            backend = _Backend([
                json.dumps({"value": "invalid_1"}),
                json.dumps({"value": "invalid_2"}),
                json.dumps({"value": "ok"}),
            ])
            result = run_single_object_task(
                run_root=root, ledger=ledger, backend=backend,
                budget=ApiBudget(normal_caps={"distractor_writer": 1},
                                 format_retry_cap=2, physical_cap=3),
                stage="distractor_writer", logical_request_id="three-attempts",
                semantic_prompt="Generate three distractors.",
                lineage_view=_lineage_view(),
                validator=lambda raw: {
                    "decision": "distractor_accept" if raw["value"] == "ok"
                    else "distractor_shape_invalid"},
                schema_retry_decisions=("distractor_shape_invalid",),
                max_attempts=3, ledger_budget_limit=3,
                ledger_total_budget_limit=3,
            )
            self.assertTrue(result["accepted"])
            self.assertEqual(3, result["physical_request_count"])
            third = ledger.read("three-attempts-a02")
            self.assertEqual("three-attempts-a01",
                             third["request_metadata"]["retry_of_request_id"])


if __name__ == "__main__":
    unittest.main()
