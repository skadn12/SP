"""Offline isolation checks for the exact A2 -> new A S03 wrapper."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import a3_continuation as a


class A3ContinuationTests(unittest.TestCase):
    def test_prepare_and_audit_preserve_exact_A2_without_API(self):
        (a.r.HERE / "runs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="a3_offline_test_", dir=a.r.HERE / "runs") as tmp:
            run = Path(tmp); run.rmdir()
            with patch.object(a.r, "Caller", side_effect=AssertionError("No API allowed")):
                result = a.prepare(run)
                self.assertEqual(result["input_turns"], 120)
                self.assertEqual(result["new_scene_slots"], ["S03"])
                self.assertEqual(a.audit(run)["generation_status"], "PREPARED_NO_LIVE_CALLS")
                self.assertEqual(len(a._history(run)), 120)
                parent = a.r.PARENT_RUN / "roots" / a.ROOT / "A"
                folder = run / "roots" / a.ROOT / "A"
                for source in parent.iterdir():
                    if source.name.startswith(("S01.", "S02.")):
                        self.assertEqual((folder / source.name).read_bytes(), source.read_bytes())
                self.assertFalse((folder / "S03.PLAN.json").exists())
                self.assertFalse((folder / "S03.LATEST_STATE.json").exists())
                history = a._history(run)
                payload = a.r.state_contract.build_state_request(a.ROOT, a.ROUTE, a.SCENE, history, [])
                proof = {"root_id": a.ROOT, "route": a.ROUTE, "scene_id": a.SCENE, "role": "state_extractor"}
                self.assertEqual(a._scope_payload(payload, proof, history)["inherited_A2_history_sha256"], a.HISTORY_SHA)
                # Model input has only accepted turn text, never frozen failed plan/scan records.
                c = json.loads(payload["messages"][0]["content"].split("STATE_INPUT_JSON\n", 1)[1])
                self.assertEqual(c["cumulative_history"], history)
                self.assertFalse(c.get("targeted_repair"))
                shortfall = {"root_id": a.ROOT, "route": a.ROUTE, "scenes": [],
                             "shortfall_reason": "No concrete progress selected in this synthetic offline run."}
                latest = a.r.base.read(parent / "S03.LATEST_STATE.json")
                class OfflineCaller:
                    def call(self, label, payload, proof):
                        a._scope_payload(payload, proof, history)
                        return json.dumps(shortfall), run / "requests/001-offline-planner"
                with patch.object(a.r, "extract_latest_state", return_value=latest), patch.object(a.r, "produce_scene") as realizer:
                    flow = a._flow(run, OfflineCaller())
                self.assertEqual(flow["accepted_scenes"], 2)
                self.assertEqual(flow["new_accepted_scenes"], 0)
                self.assertEqual(flow["output_history_sha256"], a.HISTORY_SHA)
                realizer.assert_not_called()

    def test_scope_rejects_changed_history_B_or_non_S03_stages(self):
        history = [{"turn_ref": "R02:A:S02:T001", "speaker": "Emily", "text": "Accepted history."}]
        payload = {"messages": [{"content": "\nINPUT_JSON\n" + json.dumps({"cumulative_history": history,
                                                                                 "dataset_cards": []})}]}
        proof = {"root_id": a.ROOT, "route": a.ROUTE, "scene_id": a.SCENE, "role": "planner"}
        for key, value in (("route", "dataset_new_event"), ("scene_id", "S01"), ("role", "solver")):
            changed = {**proof, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                a._scope_payload(payload, changed, history)
        with self.assertRaisesRegex(ValueError, "full_history_changed"):
            a._scope_payload(payload, proof, history + [deepcopy(history[0])])

    def test_binding_negative_prevents_realizer_and_keeps_old_plan_unaccepted(self):
        history = [{"turn_ref": "P01:T001", "speaker": "Emily", "text": "Old history."}] * 120
        inherited = {"scene_metadata": [{"scene_id": "S01"}, {"scene_id": "S02"}]}
        plan = {"root_id": a.ROOT, "route": a.ROUTE, "scenes": [{"scene_id": "S03"}], "shortfall_reason": ""}
        written = {}
        class Caller:
            def call(self, label, payload, proof):
                return json.dumps(plan), Path("/tmp/offline_A3_plan")
        with patch.object(a, "_history", return_value=history), \
             patch.object(a.r.base, "read", return_value=inherited), \
             patch.object(a.r.base, "write", side_effect=lambda p, v: written.__setitem__(str(p), deepcopy(v))), \
             patch.object(a.r, "extract_latest_state", return_value={}), \
             patch.object(a.r, "make_request", return_value=({}, {})), \
             patch.object(a, "_scope_payload", return_value={}), \
             patch.object(a.r, "validate_plan", return_value=plan), \
             patch.object(a.r.evidence_refs, "citation_provenance", return_value={}), \
             patch.object(a.r, "extract_event_state", return_value={}), \
             patch.object(a.r.event_contract, "requires_replan", return_value=True), \
             patch.object(a.r.event_contract, "plan_diagnostics", return_value=[]), \
             patch.object(a.r.event_contract, "binding_plan_checks", return_value=[{"verdict": "revise"}]), \
             patch.object(a.r, "produce_scene") as realizer:
            result = a._flow(Path("/tmp/offline_A3"), Caller())
        self.assertEqual(result["accepted_scenes"], 2)
        self.assertEqual(result["failures"][0]["reason"], "event_state_requires_replan_after_limit")
        self.assertFalse(any(p.endswith("S03.PLAN.json") for p in written))
        self.assertFalse(any(p.endswith("S03.json") for p in written))
        realizer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
