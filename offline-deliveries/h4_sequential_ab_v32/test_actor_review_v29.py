"""Explicit same-Qwen actor judgments; code checks only frozen claims/identities."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

import realization_review as review

ROOT = "actor-test-root"
ROUTE = "dataset_new_event"


def fixture():
    scene = {"scene_id": "S01", "actors": ["Emily", "James"],
        "goal": "Emily requests James's opinion about a museum detour.",
        "developments": ["Emily asks James whether the museum is worth the detour.",
                         "James provides his perspective on the museum's worth."],
        "dataset_card_id": "card-test", "source_anchors": [],
        "card_development_links": [{"card_step_ids": ["s1", "s2"],
            "dependency_indices": [1], "development_indices": [1, 2]}]}
    card = {"card_id": "card-test", "guidance_level": "scaffold",
        "actors": [{"actor_id": "requester", "role": "Requester"},
                   {"actor_id": "provider", "role": "Provider"}],
        "steps": [{"step_id": "s1", "actor_id": "requester", "action": "Requests an opinion", "state": "proposed"},
                  {"step_id": "s2", "actor_id": "provider", "action": "Provides an opinion", "state": "reported_action"}],
        "dependencies": [{"before": "s1", "after": "s2", "relation": "prerequisite"}]}
    realization = {"root_id": ROOT, "route": ROUTE, "scene_id": "S01", "status": "scene",
        "turns": [{"speaker": "Emily", "text": "James, is the museum worth a detour?"},
                  {"speaker": "James", "text": "I think the exhibits are worth the extra walk."}]}
    raw = {"root_id": ROOT, "route": ROUTE, "scene_id": "S01",
        "checks": [{"development_index": i, "realized": True,
                    "evidence": [{"turn_ref": f"NEW:T{i:03d}"}]} for i in (1, 2)],
        "new_event_or_progress": True, "consistent_with_history": True,
        "duplicates_existing_event": False, "reason": "A new museum advice event.",
        "route_fulfillment": {"role": ROUTE, "primary_event": "Museum advice",
            "primary_goal": "Decide whether a new museum detour is worthwhile",
            "relation_to_history": "distinct_event", "serves_existing_event": False,
            "existing_event": "", "new_progress": True, "reason": "It has its own goal.",
            "correction": "", "history_evidence": [],
            "new_evidence": [{"turn_ref": "NEW:T002"}]},
        "card_checks": [{"kind": "step", "id": "s1", "realized": True,
                         "evidence": [{"turn_ref": "NEW:T001"}]},
                        {"kind": "step", "id": "s2", "realized": True,
                         "evidence": [{"turn_ref": "NEW:T002"}]},
                        {"kind": "dependency", "id": 1, "realized": True,
                         "evidence": [{"turn_ref": "NEW:T001"}, {"turn_ref": "NEW:T002"}]}]}
    return scene, card, realization, raw


def actor_check(expected_actor, expected_action, ref, **changes):
    return {"expected_actor": expected_actor, "expected_action": expected_action,
        "actor": expected_actor, "action": "Performs the required behavior.",
        "evidence_mode": "direct_speech", "evidence": [{"turn_ref": ref}],
        "satisfied": True, "reason": "The cited actor performs this action."} | changes


def strict_fixture():
    scene, card, realization, raw = fixture()
    for row, actor in zip(raw["checks"], ("Emily", "James")):
        row["actor_checks"] = [actor_check(actor, scene["developments"][row["development_index"]-1],
                                             f'NEW:T{row["development_index"]:03d}')]
    raw["card_actor_bindings"] = [{"card_actor_id": "requester", "actor": "Emily"},
                                  {"card_actor_id": "provider", "actor": "James"}]
    for row, step, actor in zip(raw["card_checks"], card["steps"], ("Emily", "James")):
        row["actor_checks"] = [actor_check(actor, step["action"], "NEW:T001" if actor == "Emily" else "NEW:T002")]
    return scene, card, realization, raw


def validate(scene, card, realization, raw, **kwargs):
    return review.validate_review(raw, ROOT, ROUTE, scene, realization,
        dataset_card=card, history=[], require_actor_checks=True, **kwargs)


def repair_spec(scene, card, realization, raw):
    return review.build_format_repair(validate(scene, card, realization, raw), raw,
        scene, [], realization, dataset_card=card, require_actor_checks=True)


def patch(updates):
    return {"root_id": ROOT, "route": ROUTE, "scene_id": "S01", "repair_updates": updates}


class ActorReviewTests(unittest.TestCase):
    def test_legacy_contract_remains_opt_in_and_strict_requires_actor_checks(self):
        scene, card, realization, raw = fixture()
        self.assertEqual(review.validate_review(raw, ROOT, ROUTE, scene, realization,
            dataset_card=card, history=[])["status"], "PASS")
        result = validate(scene, card, realization, raw)
        self.assertFalse(result["schema_valid"])
        self.assertTrue(any("actor_checks_count_invalid" in e for e in result["schema_failures"]))

    def test_complete_explicit_actor_checks_accept_without_mutation(self):
        scene, card, realization, raw = strict_fixture()
        before = deepcopy(raw)
        result = validate(scene, card, realization, raw)
        self.assertEqual(result["status"], "PASS", result["failures"])
        self.assertEqual(raw, before)
        self.assertFalse(result["semantic_certified"])
        self.assertTrue(any("actor_checks" in r["path"]
            for r in result["citation_provenance"]["dereferenced_citations"]))

    def test_v28_actor_swap_pattern_rejected_by_direct_speaker_identity(self):
        scene, card, realization, raw = strict_fixture()
        raw["checks"][1]["actor_checks"][0]["evidence"] = [{"turn_ref": "NEW:T001"}]
        result = validate(scene, card, realization, raw)
        self.assertEqual(result["status"], "REJECT")
        self.assertTrue(any("direct_speaker_actor_mismatch" in e for e in result["schema_failures"]))

    def test_reported_action_does_not_equate_speaker_with_action_actor(self):
        scene, card, realization, raw = strict_fixture()
        realization["turns"][1] = {"speaker": "Emily", "text": "James just told me he thinks the exhibits are worth the extra walk."}
        for row in (raw["checks"][1], raw["card_checks"][1]):
            row["actor_checks"][0]["evidence_mode"] = "reported_action"
        result = validate(scene, card, realization, raw)
        self.assertEqual(result["status"], "PASS", result["failures"])
        self.assertFalse(result["semantic_certified"])

    def test_explicit_actor_negative_overrides_positive_realized_and_route(self):
        scene, card, realization, raw = strict_fixture()
        raw["checks"][1]["actor_checks"][0].update(satisfied=False, actor="", evidence=[],
            reason="James never offers the requested perspective.")
        result = validate(scene, card, realization, raw)
        self.assertTrue(result["schema_valid"], result["schema_failures"])
        self.assertEqual(result["status"], "REJECT")
        self.assertTrue(any("not_satisfied" in e for e in result["failures"]))

    def test_expected_action_literal_frozen_and_observed_actor_mismatch_rejects(self):
        scene, card, realization, raw = strict_fixture()
        item = raw["checks"][1]["actor_checks"][0]
        item.update(actor="Emily", evidence=[{"turn_ref": "NEW:T001"}])
        result = validate(scene, card, realization, raw)
        self.assertTrue(result["schema_valid"], result["schema_failures"])
        self.assertTrue(any("expected_observed_actor_mismatch" in e for e in result["failures"]))
        item["expected_action"] = "Emily provides her perspective instead."
        self.assertTrue(any("expected_action_source_mismatch" in e
            for e in validate(scene, card, realization, raw)["schema_failures"]))

    def test_card_step_actor_cannot_switch_abstract_binding(self):
        scene, card, realization, raw = strict_fixture()
        raw["card_checks"][1]["actor_checks"][0].update(expected_actor="Emily", actor="Emily",
            evidence=[{"turn_ref": "NEW:T001"}])
        result = validate(scene, card, realization, raw)
        self.assertTrue(any("expected_actor_binding_mismatch" in e for e in result["schema_failures"]))

    def test_citation_repair_cannot_flip_nested_negative_or_recast_expectation(self):
        scene, card, realization, raw = strict_fixture()
        item = raw["checks"][1]["actor_checks"][0]
        item.update(satisfied=False, evidence=[{"turn_ref": "NEW:T999"}])
        spec = repair_spec(scene, card, realization, raw)
        repaired = deepcopy(raw["checks"][1])
        repaired["actor_checks"][0]["evidence"] = [{"turn_ref": "NEW:T002"}]
        merged = review.merge_review_repair(raw, patch({"checks": [repaired]}), spec)
        self.assertEqual(validate(scene, card, realization, merged)["status"], "REJECT")
        repaired["actor_checks"][0]["satisfied"] = True
        with self.assertRaisesRegex(ValueError, "negative_verdict_changed"):
            review.merge_review_repair(raw, patch({"checks": [repaired]}), spec)
        repaired["actor_checks"][0]["expected_actor"] = "Emily"
        with self.assertRaisesRegex(ValueError, "actor_expectation_changed"):
            review.merge_review_repair(raw, patch({"checks": [repaired]}), spec)

    def test_citation_repair_cannot_disguise_direct_actor_as_a_report(self):
        scene, card, realization, raw = strict_fixture()
        raw["checks"][1]["actor_checks"][0]["evidence"] = [{"turn_ref": "NEW:T001"}]
        spec = repair_spec(scene, card, realization, raw)
        changed = deepcopy(raw["checks"][1])
        changed["actor_checks"][0]["evidence_mode"] = "reported_action"
        with self.assertRaisesRegex(ValueError, "actor_semantics_changed"):
            review.merge_review_repair(raw, patch({"checks": [changed]}), spec)
        changed = deepcopy(raw["checks"][1])
        changed["actor_checks"][0]["evidence"] = [{"turn_ref": "NEW:T002"}]
        merged = review.merge_review_repair(raw, patch({"checks": [changed]}), spec)
        self.assertEqual(validate(scene, card, realization, merged)["status"], "PASS")

    def test_recorded_v28_s03_quote_is_emily_not_james(self):
        # Contract simulation around an immutable recorded mismatch; this is
        # not a fresh Qwen judgment or a claim to replay missing new fields.
        path = Path(__file__).resolve().parent.parent / "h4_sequential_ab_v28/runs/inherited_a2_b3_01/roots/cc-train-episode-104814/B"
        before = (path / "S03.json").read_bytes()
        realization = json.loads(before)
        scene = json.loads((path / "S03.PLAN.json").read_text())["scenes"][0]
        claim = actor_check("James", scene["developments"][3], "NEW:T007")
        materialized = review.evidence_refs.materialize_citations(claim, [], realization["turns"])
        failures, schema_failures = [], []
        review._validate_actor_checks([materialized], scene["developments"][3], realization["turns"],
            "development[4]", failures, schema_failures)
        self.assertEqual(realization["turns"][6]["speaker"], "Emily")
        self.assertTrue(any("direct_speaker_actor_mismatch" in e for e in schema_failures))
        self.assertEqual((path / "S03.json").read_bytes(), before)

    def test_binding_repair_adds_missing_binding_preserves_existing_positive_binding(self):
        scene, card, realization, raw = strict_fixture()
        missing = raw["card_actor_bindings"].pop()
        spec = repair_spec(scene, card, realization, raw)
        good = raw["card_actor_bindings"] + [missing]
        merged = review.merge_review_repair(raw, patch({"card_actor_bindings": good}), spec)
        self.assertEqual(validate(scene, card, realization, merged)["status"], "PASS")
        bad = deepcopy(good); bad[0]["actor"] = "James"
        with self.assertRaisesRegex(ValueError, "card_actor_binding_changed"):
            review.merge_review_repair(raw, patch({"card_actor_bindings": bad}), spec)

    def test_prompt_fixes_action_strings_and_explicit_report_mode(self):
        scene, card, realization, _ = strict_fixture()
        prompt = review.build_review_prompt(ROOT, ROUTE, scene, [], realization,
            dataset_card=card, require_actor_checks=True)
        shape = json.JSONDecoder().raw_decode(prompt.split("\nOUTPUT SHAPE:\n")[1])[0]
        self.assertEqual(shape["checks"][1]["actor_checks"][0]["expected_action"], scene["developments"][1])
        self.assertEqual(shape["card_checks"][1]["actor_checks"][0]["expected_action"], card["steps"][1]["action"])
        self.assertNotIn("actor_checks", shape["card_checks"][2])
        self.assertIn("Speaker, action actor, recipient", prompt)
        self.assertIn("reported_action", prompt)


class GoalReviewIntegrationTests(unittest.TestCase):
    def test_goal_repair_preserves_prior_blocker_and_independent_judgments(self):
        scene, card, realization, raw = strict_fixture()
        role = raw["route_fulfillment"]
        role["history_goal_checks"] = [
            {"constraint_id": "c1", "historical_goal": "Dinner", "relation": "serves_existing_goal", "reason": "New work is dinner supplies."},
            {"constraint_id": "c2", "historical_goal": "Workshop", "relation": "independent", "reason": "Different concrete purpose."}]
        result = {"schema_failures": ["review_history_goal_check_coverage_invalid:route_fulfillment.history_goal_checks"]}
        spec = review.build_format_repair(result, raw, scene, [], realization,
            dataset_card=card, require_actor_checks=True, require_goal_checks=True)
        self.assertIn("route_fulfillment", spec["repair_fields"])
        changed = deepcopy(role)
        changed["history_goal_checks"][0]["relation"] = "independent"
        with self.assertRaisesRegex(ValueError, "negative_verdict_changed"):
            review.merge_review_repair(raw, patch({"route_fulfillment": changed}), spec)
        changed = deepcopy(role)
        changed["history_goal_checks"].append({"constraint_id": "c3", "historical_goal": "Other task", "relation": "independent", "reason": "Unrelated work."})
        merged = review.merge_review_repair(raw, patch({"route_fulfillment": changed}), spec)
        self.assertEqual(merged["route_fulfillment"]["history_goal_checks"][:2], role["history_goal_checks"])

    def test_fixed_timeline_goal_vetoes_globally_positive_review(self):
        scene, card, realization, raw = strict_fixture()
        history = [{"turn_ref": "P01:T001", "speaker": "James", "text": "I am planning a family dinner this weekend."}]
        event_state = {"root_id": ROOT, "route": ROUTE, "scene_id": "S01", "timeline": [
            {"constraint_id": "c1", "status": "planned", "evidence_basis": "reported",
             "statement": "James is planning a family dinner.",
             "evidence": [{"turn_ref": "P01:T001", "quote": history[0]["text"]}], "implication": "Upcoming goal."}]}
        raw["event_checks"] = [{"constraint_id": "c1", "verdict": "consistent",
                                  "history_evidence": [], "new_evidence": []}]
        raw["route_fulfillment"]["history_goal_checks"] = [{"constraint_id": "c1",
            "historical_goal": "Plan the existing family dinner", "relation": "serves_existing_goal",
            "reason": "The actual new work arranges supplies for that dinner."}]
        result = review.validate_review(raw, ROOT, ROUTE, scene, realization, dataset_card=card,
            history=history, event_state=event_state, require_actor_checks=True, require_goal_checks=True)
        self.assertTrue(result["schema_valid"], result["schema_failures"])
        self.assertEqual(result["status"], "REJECT")
        self.assertFalse(result["parsed"]["route_fulfillment"]["satisfied"])
        self.assertTrue(result["route_decision_provenance"]["history_goal_policy"]["global_satisfied"])
        self.assertEqual(result["route_decision_provenance"]["history_goal_policy"]["blocking_checks"][0]["constraint_id"], "c1")
        prompt = review.build_review_prompt(ROOT, ROUTE, scene, history, realization, dataset_card=card,
            event_state=event_state, require_actor_checks=True, require_goal_checks=True)
        shape = json.JSONDecoder().raw_decode(prompt.split("\nOUTPUT SHAPE:\n")[1])[0]
        self.assertEqual(shape["route_fulfillment"]["history_goal_checks"][0]["constraint_id"], "c1")
        payload = json.JSONDecoder().raw_decode(prompt.split("\nREVIEW INPUT DATA:\n")[1])[0]
        self.assertNotIn("implication", payload["event_state"]["timeline"][0])


if __name__ == "__main__":
    unittest.main()
