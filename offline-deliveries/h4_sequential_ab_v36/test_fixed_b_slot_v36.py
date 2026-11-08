"""Offline wrapper tests: scripted transport, socket forbidden, /tmp artifacts only."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch

import fixed_b_slot as f
from test_sequential_flow import SequentialProvider, initialize_mock

r = f.r


class FixedBSlotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="B36_FIXED_SLOT_MOCK_ONLY_")
        self.addCleanup(self.temp.cleanup)
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        delivery = Path(self.temp.name) / "delivery"; delivery.mkdir()
        for source in r.HERE.glob("*.py"): shutil.copyfile(source, delivery / source.name)
        shutil.copyfile(r.HERE / "POLICY.md", delivery / "POLICY.md")
        self.run = delivery / "experiments/MOCK_ONLY"
        self.fake = SequentialProvider()
        self.stack.enter_context(patch.object(r, "HERE", delivery))
        self.stack.enter_context(patch.object(r, "LOCK_PATH", Path(self.temp.name) / "lock"))
        self.stack.enter_context(patch.object(r, "MIN_HTTP_GAP_SECONDS", 0))
        def mock_caller(caller, run):
            initialize_mock(caller, run, self.fake)
            caller.limit = f.HTTP_MAX
        self.stack.enter_context(patch.object(r.Caller, "__init__", mock_caller))
        self.stack.enter_context(patch("socket.socket", side_effect=AssertionError("network_forbidden_offline_test")))
        self.stack.enter_context(patch.object(r.source_inventory, "reserve", side_effect=AssertionError("experiment_must_not_reserve")))
        f.prepare(self.run)

    def test_real_admission_and_prepare_do_not_call_provider(self):
        self.assertEqual(f.audit(self.run)["status"], "PASS")
        self.assertEqual(self.fake.observed, [])
        self.assertEqual(len(r.route_base_history(self.run, f.ROOT, f.ROUTE)), 132)
        self.assertEqual(r.base.read(self.run / "FROZEN_CARD.json")["card_id"], f.CARD_ID)
        self.assertTrue(all(not (self.run / x).exists() for x in f.FORBIDDEN_OUTPUTS))

    def test_full_scripted_live_and_real_replay_quarantines_output(self):
        report = f.live(self.run)
        self.assertEqual(report["status"], "EXPERIMENT_B_ACCEPTED", report)
        self.assertEqual(report["logical_calls"], 6)
        audit = f.audit(self.run)
        self.assertEqual(audit["status"], "PASS", audit)
        self.assertEqual(audit["logical_calls"], 6)
        self.assertEqual(audit["http_attempts"], 6)
        self.assertGreater(audit["artifact_replays"], 15)
        candidate = r.base.read(self.run / "experiment_candidates" / (f.ROOT + ".json"))
        self.assertFalse(candidate["training_pool_eligible"])
        self.assertEqual(candidate["new_accepted_scenes"], 1)
        self.assertEqual(report["production_added_B_scenes"], 0)
        self.assertEqual(r.base.read(self.run / "SOURCE_INVENTORY_BEFORE.json"), r.base.read(self.run / "SOURCE_INVENTORY_AFTER.json"))
        self.assertTrue(all(not (self.run / x).exists() for x in f.FORBIDDEN_OUTPUTS))
        for row in self.fake.observed:
            if row["role"] != "purpose_proposer":
                context = row["context"]
                self.assertEqual(len(context.get("cumulative_history", context.get("branch_history"))), 132)
        with self.assertRaisesRegex(ValueError, "cannot_restart"): f.live(self.run)

    def test_negative_scanner_replans_without_new_source_or_old_plan(self):
        self.fake.mode = "B_role_replan"
        report = f.live(self.run)
        self.assertEqual(report["status"], "EXPERIMENT_B_ACCEPTED", report)
        self.assertEqual(report["logical_calls"], 8)
        audit = f.audit(self.run)
        self.assertEqual(audit["status"], "PASS", audit)
        self.assertEqual(self.fake.role_counts["planner"], 2)
        self.assertTrue(report["source_inventory_unchanged"])

    def test_negative_local_scanner_patch_preserves_full_history_and_replans(self):
        self.fake.mode = "B_event_preservation_repair"
        report = f.live(self.run)
        self.assertEqual(report["status"], "EXPERIMENT_B_ACCEPTED", report)
        self.assertEqual(report["logical_calls"], 9)
        audit = f.audit(self.run)
        self.assertEqual(audit["status"], "PASS", audit)
        self.assertEqual(len(list(self.run.glob("requests/*/FORMAT_REPAIR_PRESERVATION.json"))), 1)
        self.assertEqual(self.fake.role_counts["planner"], 2)

    def test_logical_budget_precedes_transport(self):
        delegate = unittest.mock.Mock()
        scoped = f.ScopedCaller(delegate, self.run)
        scoped.calls = f.LOGICAL_MAX
        with self.assertRaisesRegex(ValueError, "logical_budget_exhausted"):
            scoped.call("unused", {}, {})
        delegate.call.assert_not_called()

    def test_missing_actor_binding_is_not_silently_filled(self):
        original = self.fake.planner
        def omit_binding(context):
            value = original(context)
            value["scenes"][0].pop("card_actor_bindings")
            return value
        self.fake.planner = omit_binding
        report = f.live(self.run)
        self.assertEqual(report["status"], "EXPERIMENT_B_SHORTFALL", report)
        self.assertEqual(report["logical_calls"], 4)
        self.assertNotIn("realizer", self.fake.role_counts)
        self.assertEqual(f.audit(self.run)["status"], "PASS")

    def test_raw_http_tamper_rejected_even_with_reindexed_outputs(self):
        f.live(self.run)
        path = next(self.run.glob("requests/*/HTTP_RESPONSE_01.json"))
        payload = r.base.read(path); payload["text"] += " "
        path.write_text(json.dumps(payload))
        index_path = self.run / "EXPERIMENT_OUTPUT_MANIFEST.json"
        index = r.base.read(index_path)
        index["files"] = r.base.bindings([Path(row["path"]) for row in index["files"]])
        index_path.write_text(json.dumps(index))
        audit = f.audit(self.run)
        self.assertEqual(audit["status"], "FAIL")
        self.assertIn("raw_http_response_changed", audit["failures"])

    def test_time_order_tamper_rejected_after_output_reindex(self):
        f.live(self.run)
        path = sorted(self.run.glob("requests/*/HTTP_RESULT_01.json"))[0]
        row = r.base.read(path)
        row["http_completed_at_utc"] = "2099-01-01T00:00:00+00:00"
        path.write_text(json.dumps(row))
        index_path = self.run / "EXPERIMENT_OUTPUT_MANIFEST.json"
        index = r.base.read(index_path)
        index["files"] = r.base.bindings([Path(row["path"]) for row in index["files"]])
        index_path.write_text(json.dumps(index))
        audit = f.audit(self.run)
        self.assertEqual(audit["status"], "FAIL")
        self.assertIn("http_gap_under_minimum", audit["failures"])

    def test_changed_source_card_fails_immutable_admission(self):
        card_path = self.run / "FROZEN_CARD.json"
        value = r.base.read(card_path); value["steps"][0]["actor_id"] = "actor_2"
        card_path.write_text(json.dumps(value))
        with self.assertRaises(ValueError): f.verify(self.run)
        self.assertEqual(f.audit(self.run)["status"], "FAIL")

    def test_reviewer_cannot_rebind_planned_source_actor(self):
        from test_actor_review_v29 import strict_fixture, validate
        scene, card, realization, raw = strict_fixture()
        scene["card_actor_bindings"] = [{"actor_id": "requester", "participant": "James"},
                                         {"actor_id": "provider", "participant": "Emily"}]
        before = deepcopy(raw)
        checked = validate(scene, card, realization, raw)
        self.assertEqual(checked["status"], "REJECT")
        self.assertTrue(checked["schema_valid"])
        self.assertTrue(any(error.startswith("review_card_actor_binding_disagrees_with_plan")
                            for error in checked["failures"]))
        self.assertEqual(raw, before)

    def test_production_ledger_in_experiment_is_rejected(self):
        r.base.write(self.run / "SOURCE_RESERVATIONS.json", {"reservations": []})
        with self.assertRaisesRegex(ValueError, "production_leak"): f.verify(self.run)


if __name__ == "__main__": unittest.main()
