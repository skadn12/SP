"""Synthetic offline tests for identity hiding, preserved failures and honest costs."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from . import reporting as r


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def candidate():
    return {"question_type": "ordering", "question": "Which event came first?", "items": {"I1": "intends to act"},
            "event_nodes": [{"item_id": "I1", "actor": "A", "event_state": "planned"}],
            "temporal_relations": [{"before": "I1", "after": "I2", "temporal_basis": "The plan preceded the action."}],
            "candidate_id": "secret-generated-id", "intent_reference": "intent_1", "model": "secret-model"}


def stage(which="primary", arm=r.ARMS[0], valid=True, replaced=False):
    return {"arm_id": arm, "root_id": r.ROOTS[0], "slot_id": "intent_1", "stage": which,
            "request_id": "request-" + which, "raw_parsed": {"candidates": [candidate()]},
            "replaced_by_repair": replaced,
            "validation": {"candidates": [{**candidate(), "source_row_index": 1, "schema_valid": valid,
                                             "schema_errors": [] if valid else ["unsupported_schema"], "gold": ["I1", "I2"]}],
                           "minimum_acceptance": {"accepted": valid}}}


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.run = Path(self.temp.name) / "run"
        self.source = Path(self.temp.name) / "source"
        self.source_policy = {"dimensions": ["GOLD_ITEM_SUPPORT"], "ordering_review": {},
                              "denominators": {}, "future_denominators": {}, "dedup_rubric": {},
                              "cost_policy": {}, "quality_sequence": ["old"], "post_quality": {},
                              "completion": {}}
        write(self.source / "review_policy.json", self.source_policy)
        for root in r.ROOTS:
            write(self.run / "inputs/shared" / root / "expanded_dialogue_clean.json", {"turns": [{"turn_ref": "T1", "text": "A planned to act."}]})
            write(self.run / "inputs/shared" / root / "history_only.json", {"records": []})

    def tearDown(self):
        self.temp.cleanup()

    def ledger(self, name, *, usage=None, outcome_only=False):
        base = self.run / "arms" / r.ARMS[0] / "qa/shards/shard-00/roots" / r.ROOTS[0] / "ledger"
        stage_name = "repair" if "repair" in name else "primary"
        metadata = {"usage": usage, "actual_model": "returned-version", "finish_reason": "stop",
                    "http_attempt_count": 1, "transport_retry_count": 0, "rate_limit_retry_count": 0}
        if not outcome_only:
            write(base / "requests" / (name + ".json"), {"request_id": name, "status": "completed",
                  "request_metadata": {"stage": stage_name, "slot_id": "intent_1"}, "response_metadata": metadata})
        write(base / "transport_outcomes" / (name + ".json"), metadata)

    def test_prepare_is_offline_idempotent_and_never_authors_review(self):
        with patch("socket.create_connection", side_effect=AssertionError("network prohibited")):
            r.prepare_review_assets(self.run, self.source)
            original = (self.run / "review_policy.json").read_bytes()
            r.prepare_review_assets(self.run, self.source)
            summary = r.report(self.run)
        self.assertEqual(original, (self.run / "review_policy.json").read_bytes())
        self.assertEqual((self.run / "reports/quality_review.jsonl").read_text(), "")
        self.assertEqual(summary["content_status"], "NOT_EVALUATED")
        self.assertEqual(summary["runtime_external_calls"], 0)
        self.assertIsNone(summary["effective_ordering_cost"])
        schema = r._read(self.run / "policy_snapshot/review_schemas/quality_review.schema.json")
        edge = schema["properties"]["edge_checks"]["items"]
        self.assertIn("basis_direction_case", edge["required"])
        self.assertIn("BASIS_TEXT_RIGHT_EDGE_REVERSED", edge["properties"]["basis_direction_case"]["enum"])

    def test_existing_policy_is_not_silently_replaced(self):
        r.prepare_review_assets(self.run, self.source)
        write(self.run / "review_policy.json", {"different": True})
        with self.assertRaisesRegex(ValueError, "existing_review_asset_differs"):
            r.prepare_review_assets(self.run, self.source)

    def test_views_include_rejected_replaced_and_repair_without_identity(self):
        r.prepare_review_assets(self.run, self.source)
        primary = stage(valid=False, replaced=True)
        repair = stage("repair")
        other = stage(arm=r.ARMS[1])
        broken = {**stage("repair", arm=r.ARMS[1]), "raw_parsed": None, "parse_error": "invalid JSON"}
        r._jsonl(self.run / "reports/runtime_stage_records.jsonl", [primary, repair, other, broken])
        final = {**repair, **repair["validation"]["candidates"][0]}
        r._jsonl(self.run / "reports/runtime_final_candidates.jsonl", [final])
        summary = r.report(self.run)
        views = r._rows(self.run / "reports/quality_review_views.jsonl")
        mapping = r._rows(self.run / "audit/quality_view_mapping.jsonl")
        self.assertEqual(len(views), 3)
        self.assertEqual(len(summary["nonreviewable_requests"]), 1)
        self.assertEqual(len({row["view_id"] for row in views}), 3)
        self.assertEqual([row["view_id"] for row in views], sorted(row["view_id"] for row in views))
        for row in views:
            serialized = json.dumps(row)
            for hidden in ("arm_id", "secret-model", "secret-generated-id", "program_schema_valid", "intent_reference"):
                self.assertNotIn(hidden, serialized)
            self.assertIn("temporal_relations", row["candidate"])
            self.assertNotEqual(row["complete_transcript_sha256"], "UNKNOWN")
        self.assertTrue(mapping[0]["replaced_by_repair"])
        self.assertFalse(mapping[0]["final_validator_accepted"])
        self.assertTrue(mapping[1]["final_validator_accepted"])
        self.assertEqual((self.run / "audit/quality_view_mapping.jsonl").stat().st_mode & 0o777, 0o600)
        self.assertTrue(all(row["content_PASS"] == "UNKNOWN" for row in summary["groups"]))

    def test_all_calls_accounted_once_and_missing_usage_not_zero(self):
        self.ledger("root-generator-intent_1", usage={"prompt_tokens": 100, "completion_tokens": 20,
                    "total_tokens": 120, "completion_tokens_details": {"reasoning_tokens": 5}})
        self.ledger("root-repair-intent_1", usage={"prompt_tokens": 200, "completion_tokens": 30,
                    "total_tokens": 230})
        self.ledger("root-generator-intent_2", outcome_only=True)
        costs = r.call_accounting(self.run)
        self.assertEqual(costs["total"]["logical_calls"], 3)
        self.assertEqual(costs["total"]["http_attempts"]["value"], 3)
        self.assertEqual(costs["total"]["total_tokens"]["known_subtotal"], 350)
        self.assertEqual(costs["total"]["total_tokens"]["value"], "UNKNOWN")
        self.assertEqual(costs["total"]["total_tokens"]["unknown_calls"], 1)
        self.assertEqual(costs["total"]["reasoning_tokens"]["known_subtotal"], 5)
        self.assertEqual(costs["total"]["verified_money"]["value"], "UNKNOWN")
        self.assertEqual(costs["calls"][0]["usage"]["completion_tokens_details"], {"reasoning_tokens": 5})
        self.assertIsNone(costs["total"]["tokens_per_effective_ordering"])
        self.assertEqual({row["stage"] for row in costs["calls"]}, {"primary", "repair"})

    def test_partial_technical_stop_keeps_completed_yield_and_unsent_slots_distinct(self):
        r.prepare_review_assets(self.run, self.source)
        completed = stage()
        self.ledger("root-generator-intent_1", usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120})
        final = {**completed, **completed["validation"]["candidates"][0]}
        failed = {**stage(arm=r.ARMS[1]), "raw_parsed": None, "parse_error": "technical failure", "validation": {}}
        r._jsonl(self.run / "reports/runtime_stage_records.jsonl", [completed, failed])
        r._jsonl(self.run / "reports/runtime_final_candidates.jsonl", [final])
        base = self.run / "arms" / r.ARMS[1] / "qa/shards/shard-00/roots" / r.ROOTS[0] / "ledger"
        write(base / "transport_outcomes/failure.json", {
            "stage": "primary", "slot_id": "intent_1", "http_attempt_count": 1,
            "service_completion_status": "KNOWN_HTTP_REJECTION", "http_status": 400})
        write(self.run / "reports/runtime_summary.json", {"status": "STOPPED", "external_calls": 2})
        write(self.run / "audit/execution_stop.json", {"reason": "TECHNICAL_FAILURE_NO_REPAIR"})
        summary = r.report(self.run)
        self.assertEqual(summary["attempted_primary_slots"], 2)
        self.assertEqual(summary["unattempted_primary_slots"], 10)
        self.assertEqual(summary["technical_failure_calls"], 1)
        self.assertEqual(summary["costs"]["total_tokens"]["value"], "UNKNOWN")
        self.assertEqual(summary["costs"]["total_tokens"]["known_subtotal"], 120)
        self.assertEqual(summary["execution_stop_reason"], "TECHNICAL_FAILURE_NO_REPAIR")
        slots = r._read(self.run / "reports/slot_outcomes_and_collateral.json")
        self.assertEqual(sum(s["status"] == "NOT_RUN" for s in slots), 10)
        failure = next(s for s in slots if s["status"] == "TECHNICAL_FAILURE")
        self.assertEqual(failure["http_attempts"]["value"], 1)
        self.assertIsNone(failure["whole_batch_rejected"])
        self.assertEqual(sum(s["final_program_accepted"] for s in slots), 1)
        failed_groups = [g for g in summary["groups"] if g["root_id"] == r.ROOTS[0] and g["arm_id"] == r.ARMS[1] and g["layer"] == "primary"]
        self.assertTrue(all(g["parse_failed_requests"] == 0 and g["technical_failed_requests"] == 1 for g in failed_groups))

    def test_authored_pass_is_not_auto_certified_and_duplicates_block(self):
        r.prepare_review_assets(self.run, self.source)
        r._jsonl(self.run / "reports/runtime_stage_records.jsonl", [stage()])
        r.report(self.run)
        view_id = r._rows(self.run / "reports/quality_review_views.jsonl")[0]["view_id"]
        authored = {"view_id": view_id, "judgment": "PASS"}
        r._jsonl(self.run / "reports/quality_review.jsonl", [authored])
        summary = r.report(self.run)
        self.assertEqual(summary["status"], "PENDING_OFFLINE_DIAGNOSTICS")
        self.assertEqual(summary["authored_rows_pending_validation"], 1)
        self.assertTrue(all(row["deduplicated_supported_count"] == "UNKNOWN" for row in summary["groups"]))
        r._jsonl(self.run / "reports/quality_review.jsonl", [authored, authored])
        self.assertEqual(r.report(self.run)["status"], "BLOCKED_REVIEW_IDENTITY")


if __name__ == "__main__":
    unittest.main()
