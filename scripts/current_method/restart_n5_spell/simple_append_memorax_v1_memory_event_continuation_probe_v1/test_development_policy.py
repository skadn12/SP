from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import json
import unittest
from unittest.mock import patch

from . import development_policy as policy
from .contracts import (compile_planner_output, build_planner_outline_pairs,
                        materialize_planner_output, validate_realizer_plan_isolation)
from .profiles import DEVELOPMENT_GAIN_PROFILE, LEGACY_PROFILE
from .test_memory_event_continuation import (
    PARENT_PATH, _split_plan, _adherence_reviews,
)
from .workflow import execute_round


def memory():
    # Exact narrow historical failure: intention was mistaken for submission.
    return {"buckets": {"event": [{"memory_facts": [
        {"memory_ref": "GM:E006:C002", "text": "Emily completed the revisions by adding details to the budget and adjusting the timeline for realism. She expressed confidence that the changes would make the proposal more credible and easier to approve, stating she intended to send the revised version back to her boss that afternoon."},
        {"memory_ref": "GM:E006:C003", "text": "After finishing the revisions, Emily reviewed her proposal one last time, feeling confident enough to submit it after lunch. She prepared a backup plan in case the boss had concerns, noting that she did not want to rely on luck this time but wanted to be ready for any outcome."},
        {"memory_ref": "GM:E004:C001", "text": "Emily flew solo last weekend."},
    ]}]}}


def plan(count=3):
    return compile_planner_output(_split_plan(0, count), candidates=[],
                                  scene_quota=3)


def gate_review(ref, relation="NEW_PROPOSAL"):
    return {"candidate_ref": ref, "relation": relation,
            "identity_match": {k: None for k in ("actor", "action", "object", "occurrence", "target_state")},
            "completion_parts": [{"claim": "Emily sends her revised proposal", "state": "unknown", "evidence_indices": []}],
            "evidence": [], "reason": "Independent proposed arrangement based on established working relationship; absence is not proof of noncompletion."}


def outline(plan_value):
    return {"candidate_reviews": [{
        "candidate_ref": c["candidate_ref"], "quality": "useful",
        "expected_information": c["after_state"],
        "target_after_state": c["after_state"],
        "coverage": ["new_information"], "precondition_status": "supported",
        "depends_on": [], "evidence": [], "reason": "Specific new result established in scene.",
    } for c in policy.candidate_views(plan_value)],
        "pair_reviews": [{"pair_ref": p["pair_ref"], "decision": "distinct", "reason": "Independent concrete actions."}
                         for p in build_planner_outline_pairs(plan_value)]}


class DevelopmentPolicyTest(unittest.TestCase):
    def test_intention_cannot_be_declared_completed_with_planned_state(self):
        p = plan(1)
        r = gate_review("ND01:C01", "REALIZED")
        r["identity_match"] = dict.fromkeys(r["identity_match"], True)
        r["evidence"] = [{"memory_ref": "GM:E006:C002", "quote": "she intended to send the revised version back to her boss that afternoon"}]
        r["completion_parts"][0].update(state="planned", evidence_indices=[1])
        out = policy.validate_gate({"candidate_reviews": [r]}, plan=p, memory_view=memory())
        self.assertFalse(out["valid"])
        r["relation"] = "PLANNED_OR_INTENDED"
        self.assertEqual(policy.validate_gate({"candidate_reviews": [r]}, plan=p, memory_view=memory())["passed_candidate_refs"], ["ND01:C01"])

    def test_unrelated_ref_and_unknown_are_not_noncompletion(self):
        p = plan(1)
        r = gate_review("ND01:C01", "UNKNOWN")
        out = policy.validate_gate({"candidate_reviews": [r]}, plan=p, memory_view=memory())
        self.assertTrue(out["valid"])
        self.assertEqual(out["passed_candidate_refs"], ["ND01:C01"])
        self.assertEqual(out["deferred_candidate_refs"], ["ND01:C01"])
        self.assertEqual(out["normalized_candidate_reviews"][0]["decision"], "defer")
        r["evidence"] = [{"memory_ref": "GM:E004:C001", "quote": "she intended to send the revised version"}]
        self.assertFalse(policy.validate_gate({"candidate_reviews": [r]}, plan=p, memory_view=memory())["valid"])

    def test_completion_may_use_multiple_memory_sentences(self):
        p = plan(1)
        m = memory()
        m["buckets"]["event"][0]["memory_facts"].extend([
            {"memory_ref": "M1", "text": "Emily sent the revised proposal."},
            {"memory_ref": "M2", "text": "She then shared that backup plan with David."}])
        r = gate_review("ND01:C01", "REALIZED")
        r["identity_match"] = dict.fromkeys(r["identity_match"], True)
        r["evidence"] = [{"memory_ref": "M1", "quote": "Emily sent the revised proposal."}, {"memory_ref": "M2", "quote": "She then shared that backup plan with David."}]
        r["completion_parts"] = [{"claim": "submission", "state": "completed", "evidence_indices": [1]}, {"claim": "sharing", "state": "completed", "evidence_indices": [1, 2]}]
        out = policy.validate_gate({"candidate_reviews": [r]}, plan=p, memory_view=m)
        self.assertTrue(out["valid"])
        self.assertEqual(out["rejected_candidate_refs"], ["ND01:C01"])

    def test_low_gain_does_not_fill_quota_and_state_cannot_be_strengthened(self):
        p = plan()
        raw = outline(p)
        raw["candidate_reviews"][0]["quality"] = "low"
        raw["candidate_reviews"][1]["quality"] = "unknown"
        a = policy.validate_outline(raw, plan=p, memory_view=memory())
        selected = policy.select_candidates(["ND01:C01", "ND02:C01", "ND03:C01"], audit=a, limit=3)
        self.assertEqual(selected["selected_candidate_refs"], ["ND03:C01"])
        self.assertEqual(selected["unfilled_scene_count"], 2)
        raw["candidate_reviews"][2]["target_after_state"] = "full mastery instead of clearer understanding"
        self.assertFalse(policy.validate_outline(raw, plan=p, memory_view=memory())["valid"])

    def test_useful_admission_beats_original_rank_no_new_person_quota(self):
        p = plan()
        raw = outline(p)
        raw["candidate_reviews"][0]["quality"] = "low"
        raw["candidate_reviews"][1]["quality"] = "low"
        a = policy.validate_outline(raw, plan=p, memory_view=memory())
        self.assertEqual(policy.select_candidates(["ND01:C01", "ND02:C01", "ND03:C01"], audit=a, limit=1)["selected_candidate_refs"], ["ND03:C01"])

    def test_dependency_order_and_missing_prerequisite(self):
        p = plan(2)
        raw = outline(p)
        raw["candidate_reviews"][0].update(depends_on=["ND02:C01"], precondition_status="conditional")
        a = policy.validate_outline(raw, plan=p, memory_view=memory())
        self.assertEqual(policy.select_candidates(["ND01:C01", "ND02:C01"], audit=a, limit=2)["selected_candidate_refs"], ["ND02:C01", "ND01:C01"])
        source = policy.materialization_input(p, ["ND02:C01", "ND01:C01"])
        materialized = policy.annotate_selected(materialize_planner_output(
            source, passed_candidate_refs=["ND02:C01", "ND01:C01"], scene_count=2), a, round_index=2)
        self.assertEqual(materialized["scenes"][1]["depends_on_scene_refs"], ["R02:S01"])
        self.assertTrue(validate_realizer_plan_isolation(raw_plan=source,
            materialized_plan=materialized, passed_candidate_refs=["ND02:C01", "ND01:C01"], scene_count=2)["valid"])
        self.assertEqual(policy.select_candidates(["ND01:C01"], audit=a, limit=2)["selected_candidate_refs"], [])
        raw["candidate_reviews"][1].update(depends_on=["ND01:C01"], precondition_status="conditional")
        a = policy.validate_outline(raw, plan=p, memory_view=memory())
        self.assertFalse(policy.select_candidates(["ND01:C01", "ND02:C01"], audit=a, limit=2)["valid"])

    def test_single_candidate_quality_coverage_and_duplicate_handling(self):
        p = plan(1)
        self.assertEqual(policy.outline_prompt(plan=p, memory_view=memory())["candidate_count"], 1)
        raw = outline(p)
        raw["candidate_reviews"] = []
        self.assertFalse(policy.validate_outline(raw, plan=p, memory_view=memory())["valid"])
        p = plan(2)
        raw = outline(p)
        raw["pair_reviews"][0]["decision"] = "duplicate"
        a = policy.validate_outline(raw, plan=p, memory_view=memory())
        self.assertEqual(len(policy.select_candidates(["ND01:C01", "ND02:C01"], audit=a, limit=2)["selected_candidate_refs"]), 1)

    def test_profile_and_live_cli_preflight_dispatch(self):
        from . import live
        self.assertIsNone(LEGACY_PROFILE.development_policy)
        self.assertEqual(DEVELOPMENT_GAIN_PROFILE.worst_case_requests()["total_model_requests"], 7)
        with self.assertRaises(ValueError):
            replace(DEVELOPMENT_GAIN_PROFILE, realizer_batch_size=1,
                    sequential_batch_promotion=True)
        with patch.object(live, "preflight", return_value={"decision": "pass"}) as preflight, patch.object(live, "execute_live") as execute, patch("builtins.print"):
            live.main(["--profile", "development_gain_v2", "--preflight-only"])
            self.assertEqual(preflight.call_args.kwargs["execution_profile"], DEVELOPMENT_GAIN_PROFILE)
            execute.assert_not_called()

    def test_malformed_model_fields_are_invalid_not_exceptions(self):
        p = plan(1)
        for field in ("relation", "identity_match", "completion_parts", "evidence", "reason"):
            for bad in ([], {}, 42):
                if field == "evidence" and bad == []:
                    continue  # Empty evidence is valid for UNKNOWN.
                row = gate_review("ND01:C01", "UNKNOWN")
                row[field] = bad
                with self.subTest(stage="gate", field=field, bad=bad):
                    result = policy.validate_gate({"candidate_reviews": [row]}, plan=p, memory_view=memory())
                    self.assertFalse(result["valid"])
        for bad in ([], {}):
            row = gate_review("ND01:C01", "UNKNOWN")
            row["completion_parts"][0]["state"] = bad
            self.assertFalse(policy.validate_gate({"candidate_reviews": [row]}, plan=p, memory_view=memory())["valid"])
        for field in ("quality", "precondition_status", "depends_on", "coverage"):
            for bad in ({}, 42):
                raw = outline(p)
                raw["candidate_reviews"][0][field] = bad
                with self.subTest(stage="outline", field=field, bad=bad):
                    self.assertFalse(policy.validate_outline(raw, plan=p, memory_view=memory())["valid"])

    def test_new_profile_cannot_accidentally_use_remote_memory_or_live_transport(self):
        from .live import execute_live
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "not-created"
            with patch("socket.socket", side_effect=AssertionError("network forbidden")), self.assertRaisesRegex(RuntimeError, "development_gain_live_disabled"):
                execute_live(report={"decision": "pass"}, input_path=Path("unused"),
                    run_root=run, config_path=Path("unused"), tokenizer_path=Path("unused"),
                    credentials_path=Path("unused"), allow_insecure_http=False,
                    execution_profile=DEVELOPMENT_GAIN_PROFILE)
            self.assertFalse(run.exists())

    def test_same_workflow_five_mock_calls_no_api_memory_or_retry(self):
        parent = json.loads(PARENT_PATH.read_text())
        speakers = list(dict.fromkeys(t["speaker"] for t in parent["dialogue"]["turns"]))
        artifacts, calls = {}, []

        class FrozenMemory:
            calls = []
            def search_buckets(self, *args, **kwargs):
                return {"success": True, "data": {"buckets": {"person": {}, "location": {}, "event": {"event-a": {"display_key": "project", "content": [f"{speakers[0]} plans to submit a proposal."]}}, "item": {}, "dreaming": {}}}}
            def add(self, *args, **kwargs):
                raise AssertionError("offline fixture must not write Memory")
            def add_status(self, *args, **kwargs):
                raise AssertionError("offline fixture must not poll")

        def planner(prompt):
            calls.append("planner")
            self.assertIn("AUTHORITATIVE PARENT TRANSCRIPT", prompt["prompt"])
            self.assertIn(parent["dialogue"]["turns"][0]["text"], prompt["prompt"])
            raw = _split_plan(0, 3)
            for row in raw["new_development_candidates"]:
                row.pop("participants")
                row["speaker_refs"] = ["SP01", "SP02"]
                row["actor_bindings"] = [{"actor_ref": "SP01", "binding": f"{speakers[0]} owns the specific action"}]
            return raw

        def gate(prompt):
            calls.append("gate")
            self.assertNotIn(parent["dialogue"]["turns"][0]["text"], prompt["prompt"])
            reviews = [gate_review(c["candidate_ref"]) for c in policy.candidate_views(artifacts["planner/compiled_plan"])]
            ref, quote = next(iter(policy.memory_facts(artifacts["memory/global_state_view"]).items()))
            for review in reviews:
                review["evidence"] = [{"memory_ref": ref, "quote": quote}]
            return {"candidate_reviews": reviews}

        def audit(prompt):
            calls.append("outline")
            raw = outline(artifacts["planner/compiled_plan"])
            raw["candidate_reviews"][0]["quality"] = "low"
            return raw

        def realizer(prompt):
            calls.append("realizer")
            self.assertFalse(prompt["full_parent_included"])
            self.assertNotIn(parent["dialogue"]["turns"][0]["text"], prompt["prompt"])
            return {"scenes": [{"dialogue_turns": [{"speaker": speakers[j % 2], "text": f"Synthetic scene {i} independently generated reply {j}."} for j in range(14)]} for i in range(2)]}

        def adherence(prompt):
            calls.append("adherence")
            self.assertIn("exact_target_after_state", prompt["prompt"])
            return _adherence_reviews(2)

        with patch("socket.socket", side_effect=AssertionError("no network in mock")):
            result = execute_round(parent=parent, memorax=FrozenMemory(),
                verifier_call=lambda p: self.fail("verifier forbidden"),
                planner_call=planner, planner_repair_call=lambda p: self.fail("no deficit repair"),
                semantic_gate_call=gate, outline_semantic_call=audit,
                realizer_call=realizer, adherence_call=adherence,
                scene_replacement_call=lambda p: self.fail("no replacement"),
                execution_profile=replace(DEVELOPMENT_GAIN_PROFILE, feedback_view_version=None),
                reuse_existing_namespace=True, no_memorax_writes=True,
                artifact_sink=lambda name, value: artifacts.update({name: deepcopy(value)}),
                token_counter=lambda text: max(1, len(text) // 4))
        self.assertEqual(calls, ["planner", "gate", "outline", "realizer", "adherence"])
        self.assertEqual(len(result["child_dialogue"]["scenes"]), 2)
        self.assertEqual(artifacts["planner/outline_semantic_selection_audit"]["unfilled_scene_count"], 1)


if __name__ == "__main__":
    unittest.main()
