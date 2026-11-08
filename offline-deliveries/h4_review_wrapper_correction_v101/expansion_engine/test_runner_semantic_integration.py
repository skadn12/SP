"""Offline routing tests with manually annotated review fixtures, never model accuracy.

FakeCaller writes only temporary request artifacts and performs no network I/O.
The tests exercise production gate validation and bounded retry/repair routing.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import runner as r
import audit_run as a
from event_identity_test_fixtures import comparison


ROOT = "cc-train-episode-fixture"
B = r.ROUTES[1]
A = r.ROUTES[0]
HISTORY = [{"turn_ref": "P:S01:T001", "speaker": "Elena",
            "text": "We still need to choose a lunch venue."}]
CARD = {
    "card_id": "fixture-card", "card_sha256": "a" * 64, "guidance_level": "scaffold",
    "actors": [{"actor_id": "a1", "role": "requester"}, {"actor_id": "a2", "role": "provider"}],
    "steps": [{"step_id": "s1", "actor_id": "a1", "action": "request venue details", "state": "requested"},
              {"step_id": "s2", "actor_id": "a2", "action": "provide venue details", "state": "reported"}],
    "dependencies": [{"before": "s1", "after": "s2", "relation": "respond to request"}],
    "confusion_axes": [], "open_ends": [], "downgrade_reason": "",
}
SCENE = {
    "scene_id": "S01", "actors": ["Elena", "Milan"],
    "developments": ["Elena asks Milan for details of a new lunch venue.",
                     "Milan provides its address and opening hours."],
    "dataset_card_id": CARD["card_id"],
    "card_development_links": [
        {"card_step_ids": ["s1"], "dependency_indices": [], "development_indices": [1]},
        {"card_step_ids": ["s2"], "dependency_indices": [1], "development_indices": [2]},
    ],
}


def plan_span(index):
    return {"development_index": index, "quote": SCENE["developments"][index - 1]}


def event_fixture():
    return {
        "root_id": ROOT, "route": B, "scene_id": "S01",
        "segments": [{"segment_id": "P:S01", "none_reason": "No earlier visit constrains this new request."}],
        "timeline": [],
        "plan_checks": [{"development_index": index, "verdict": "compatible", "constraint_ids": [],
                         "reason": "This coherent new request may establish a new event.",
                         "issue_type": "coherent_new_event", "plan_evidence": [plan_span(index)]}
                        for index in (1, 2)],
        "card_checks": [{"card_id": CARD["card_id"],
                         "function_check": {"verdict": "compatible", "reason": "The whole event remains a venue inquiry.",
                                            "plan_evidence": [plan_span(1)], "comparison": comparison(CARD, HISTORY, SCENE, bound=True)},
                         "step_checks": [{"step_id": "s" + str(index), "mapped_development_indices": [index],
                                          "verdict": "compatible", "reason": "The mapped action has the correct role and state.",
                                          "plan_evidence": [plan_span(index)]} for index in (1, 2)],
                         "dependency_checks": [{"dependency_index": 1, "mapped_development_indices": [2],
                                                "verdict": "compatible", "reason": "The provider responds to the requester.",
                                                "plan_evidence": [plan_span(2)]}]}],
        "consistency_checks": [{"category": category, "verdict": "compatible",
                                "reason": "No conflict appears in this dimension of the proposed event.",
                                "plan_evidence": [plan_span(1)]}
                               for category in sorted(r.event_contract.CONSISTENCY_CATEGORIES)],
    }


class FakeCaller:
    def __init__(self, run, responses):
        self.run, self.responses, self.calls = Path(run), list(responses), []

    def call(self, label, payload, proof):
        role, raw = self.responses.pop(0)
        if role != proof["role"]:
            raise AssertionError(f"expected {role}, received {proof['role']}")
        folder = self.run / "requests" / f"{len(self.calls) + 1:03d}-{label}"
        folder.mkdir(parents=True)
        self.calls.append({"role": role, "payload": deepcopy(payload), "proof": deepcopy(proof),
                           "folder": folder})
        return json.dumps(raw), folder


def generator_stub(run, root, route, role, history, cards, scene_id, **options):
    context = {"root_id": root, "route": route, "role": role,
               "targeted_repair": deepcopy(options.get("repair"))}
    return {"model": r.MODEL, "messages": [{"role": "user", "content": json.dumps(context)}]}, {
        "root_id": root, "route": route, "role": role, "scene_id": scene_id,
    }


class RunnerSemanticIntegrationTests(unittest.TestCase):
    def prepare(self, run):
        r.base.write(Path(run) / "sources" / f"{ROOT}.json", {"turns": HISTORY})

    def validated(self, fixture):
        return r.event_contract.validate(fixture, ROOT, B, SCENE, HISTORY, cards=[CARD])

    def test_card_semantic_negative_replans_before_any_realizer(self):
        negative, positive = event_fixture(), event_fixture()
        negative["card_checks"][0]["step_checks"][0].update(
            verdict="revise", reason="The requester discusses an unrelated project instead of venue details.")
        negative = self.validated(negative)
        positive = self.validated(positive)
        plan = {"root_id": ROOT, "route": B, "scenes": [SCENE], "shortfall_reason": ""}
        with tempfile.TemporaryDirectory() as tmp:
            self.prepare(tmp)
            caller = FakeCaller(tmp, [("planner", plan), ("planner", plan)])
            entered = []
            def production(*args, **kwargs):
                entered.append(len(caller.calls))
                return None, [{"status": "TEST_ONLY_NO_OUTPUT"}]
            with patch.object(r, "extract_latest_state", return_value={"facts": []}), \
                 patch.object(r, "make_request", side_effect=generator_stub), \
                 patch.object(r, "validate_plan", side_effect=lambda raw, *args, **kwargs: json.loads(raw)), \
                 patch.object(r, "extract_event_state", side_effect=[negative, positive]), \
                 patch.object(r, "produce_scene", side_effect=production):
                r.run_root(caller, Path(tmp), ROOT, [CARD], routes=(B,))
            self.assertEqual(entered, [2])
            self.assertEqual([call["role"] for call in caller.calls], ["planner", "planner"])
            repair = json.loads(caller.calls[1]["payload"]["messages"][0]["content"])["targeted_repair"]
            self.assertEqual(repair["plan_checks"][0]["check_kind"], "card_step")
            self.assertEqual(repair["plan_checks"][0]["step_id"], "s1")
            self.assertEqual(r.base.read(caller.calls[0]["folder"] / "EVENT_STATE_DECISION.json")["status"], "REPLAN")
            self.assertEqual(r.base.read(caller.calls[1]["folder"] / "EVENT_STATE_DECISION.json")["status"], "PASS")

    def test_absence_only_negative_retries_same_reviewer_without_automatic_pass(self):
        bad = event_fixture()
        bad["plan_checks"][0].update(verdict="revise", issue_type="unsupported_historical_assertion",
                                     reason="This is a new event not supported by prior turns.")
        with tempfile.TemporaryDirectory() as tmp:
            self.prepare(tmp)
            caller = FakeCaller(tmp, [("event_state_extractor", bad), ("event_state_extractor", event_fixture())])
            result = r.extract_event_state(caller, Path(tmp), ROOT, B, HISTORY, [CARD], SCENE,
                                           {"facts": []}, plan_attempt=1)
            self.assertEqual(result, event_fixture())
            self.assertEqual(len(caller.calls), 2)
            rejected = r.base.read(caller.calls[0]["folder"] / "VALIDATION.json")
            self.assertEqual(rejected["status"], "REJECT")
            self.assertIn("event_state_history_absence_not_contradiction", {error["code"] for error in rejected["errors"]})
            context = a.decode_context(caller.calls[1]["payload"], "\nEVENT_STATE_INPUT_JSON\n")
            self.assertEqual(context["targeted_repair"]["previous_raw"], json.dumps(bad))
            self.assertEqual(context["scene_plan"], SCENE)
            self.assertEqual(r.base.read(caller.calls[1]["folder"] / "VALIDATION.json")["status"], "PASS")

    def test_repeated_absence_only_review_stops_at_existing_budget(self):
        bad = event_fixture()
        bad["plan_checks"][0].update(verdict="revise", issue_type="unsupported_historical_assertion",
                                     reason="This is a new event not supported by prior turns.")
        with tempfile.TemporaryDirectory() as tmp:
            self.prepare(tmp)
            caller = FakeCaller(tmp, [("event_state_extractor", bad), ("event_state_extractor", bad)])
            self.assertIsNone(r.extract_event_state(caller, Path(tmp), ROOT, B, HISTORY, [CARD], SCENE,
                                                   {"facts": []}, plan_attempt=1))
            self.assertEqual(len(caller.calls), r.EVENT_STATE_ATTEMPTS)
            self.assertTrue(all(r.base.read(call["folder"] / "VALIDATION.json")["status"] == "REJECT"
                                for call in caller.calls))

    def scene_response(self, route, coherent):
        texts = (["Shall I book the table for 12:30?", "Yes, please book it for 12:30.",
                  "All set. I have secured the table for 12:30."] if coherent else
                 ["I will call to lock in the booking.", "With our celebration booked, we can relax.",
                  "I will confirm once the reservation is secured."])
        texts += [f"We can discuss lunch detail number {number} after that." for number in range(4, 13)]
        return {"root_id": ROOT, "route": route, "scene_id": "S01", "status": "scene",
                "turns": [{"speaker": "Elena" if number % 2 else "Milan", "text": text}
                          for number, text in enumerate(texts)],
                "feedback_topic_ids": [], "no_scene_reason": ""}

    def scene_review(self, realization, contradicted):
        def span(number):
            return {"turn_ref": f"NEW:T{number:03d}", "quote": realization["turns"][number - 1]["text"]}
        return {
            "root_id": ROOT, "route": A, "scene_id": "S01",
            "checks": [{"development_index": index, "realized": True, "evidence": [span(index)]}
                       for index in (1, 2)],
            "new_event_or_progress": True, "consistent_with_history": True,
            "duplicates_existing_event": False, "reason": "Manual fixture annotation of booking progression.",
            "consistency_checks": [
                {"check_id": "participant_identity", "verdict": "supported", "reason": "The two actors address each other.",
                 "evidence": [span(1), span(2)]},
                {"check_id": "itinerary_and_time", "verdict": "not_applicable", "reason": "No travel itinerary.", "evidence": []},
                {"check_id": "action_status_progression", "verdict": "contradicted" if contradicted else "supported",
                 "reason": "Completion is asserted between pending claims." if contradicted else "Authorization is followed by a new explicit completion report.",
                 "evidence": [span(1), span(2), span(3)]},
            ],
        }

    def test_internal_scene_negative_repairs_realizer_with_exact_findings(self):
        rejected, accepted = self.scene_response(A, False), self.scene_response(A, True)
        scene = deepcopy(SCENE)
        scene["dataset_card_id"], scene["card_development_links"] = None, []
        with tempfile.TemporaryDirectory() as tmp:
            self.prepare(tmp)
            caller = FakeCaller(tmp, [("realizer", rejected), ("scene_reviewer", self.scene_review(rejected, True)),
                                      ("realizer", accepted), ("scene_reviewer", self.scene_review(accepted, False))])
            with patch.object(r, "make_request", side_effect=generator_stub):
                result, audits = r.produce_scene(caller, Path(tmp), ROOT, A, HISTORY, [], scene, [])
            self.assertEqual(result, accepted)
            self.assertEqual([audit["status"] for audit in audits], ["REJECT", "PASS"])
            self.assertEqual([call["role"] for call in caller.calls], ["realizer", "scene_reviewer", "realizer", "scene_reviewer"])
            repair = json.loads(caller.calls[2]["payload"]["messages"][0]["content"])["targeted_repair"]
            findings = repair["scene_consistency_findings"]
            self.assertEqual(len(findings), 1)
            self.assertEqual(findings[0]["check_id"], "action_status_progression")
            self.assertEqual(findings[0]["evidence"], self.scene_review(rejected, True)["consistency_checks"][2]["evidence"])
            self.assertEqual(repair["previous_response"], rejected)

    def test_missing_scene_checks_retries_reviewer_not_dialogue(self):
        accepted = self.scene_response(A, True)
        bad_review, good_review = self.scene_review(accepted, False), self.scene_review(accepted, False)
        del bad_review["consistency_checks"]
        scene = deepcopy(SCENE)
        scene["dataset_card_id"], scene["card_development_links"] = None, []
        with tempfile.TemporaryDirectory() as tmp:
            self.prepare(tmp)
            caller = FakeCaller(tmp, [("realizer", accepted), ("scene_reviewer", bad_review), ("scene_reviewer", good_review)])
            with patch.object(r, "make_request", side_effect=generator_stub):
                result, audits = r.produce_scene(caller, Path(tmp), ROOT, A, HISTORY, [], scene, [])
            self.assertEqual(result, accepted)
            self.assertEqual([call["role"] for call in caller.calls], ["realizer", "scene_reviewer", "scene_reviewer"])
            self.assertTrue(caller.calls[2]["proof"]["review_format_repair"])
            self.assertEqual(audits[0]["status"], "PASS")


class AuditTailBindingTests(unittest.TestCase):
    def fixture(self, role):
        prefix = {"model": r.MODEL, "messages": [{"role": "user", "content": "Frozen input and historical feedback."}]}
        payload = deepcopy(prefix)
        tail = (r.planner_contract.final_planner_instruction() if role == "planner" else r.final_realizer_instruction())
        payload["messages"][0]["content"] += tail
        receipt = {"before_payload_sha256": r.base.digest(prefix),
                   "before_text_sha256": r.base.digest(prefix["messages"][0]["content"].encode()),
                   "tail_sha256": r.base.digest(tail.encode()), "final_payload_sha256": r.base.digest(payload),
                   "final_text_sha256": r.base.digest(payload["messages"][0]["content"].encode())}
        return prefix, payload, {role + "_tail_receipt": receipt}

    def test_planner_and_realizer_tail_replay_restores_exact_feedback_payload(self):
        for role in ("planner", "realizer"):
            prefix, payload, proof = self.fixture(role)
            restored, text = a.feedback_prefix(payload, proof, role, {})
            self.assertEqual(restored, prefix)
            self.assertEqual(text, prefix["messages"][0]["content"])

    def test_missing_tail_or_any_sha_change_rejected(self):
        for role in ("planner", "realizer"):
            for key in ("tail_sha256", "final_payload_sha256", "final_text_sha256", "before_payload_sha256", "before_text_sha256"):
                prefix, payload, proof = self.fixture(role)
                proof[role + "_tail_receipt"][key] = "0" * 64
                with self.subTest(role=role, key=key), self.assertRaises(ValueError):
                    a.feedback_prefix(payload, proof, role, {})
            prefix, payload, proof = self.fixture(role)
            with self.assertRaises(ValueError):
                a.feedback_prefix(payload, {}, role, {})
            payload["messages"][0]["content"] = prefix["messages"][0]["content"]
            with self.assertRaises(ValueError):
                a.feedback_prefix(payload, proof, role, {})


class AuditSelectedCardBindingTests(unittest.TestCase):
    def prior_run(self):
        return (r.DELIVERIES / "h4_planner_cardinality_expansion_batch_v91"
                / "expansion_engine/runs/ten_roots_B_scene01")

    def test_actual_v91_cards_keep_reserved_root_scene_mapping_and_failed_slot(self):
        run = self.prior_run()
        slots, cards = a.selected_slot_cards(run, r.base.read(run / "UNIQUE_EVENT_ALLOCATION.json"))
        self.assertEqual(len(cards), 9)
        self.assertEqual(len(slots), 10)
        self.assertEqual(sum(bool(guidance) for guidance in slots.values()), 9)
        self.assertEqual(slots[("cc-train-episode-24719", B, "S01")], [])
        self.assertTrue(all(not guidance[0]["source_is_root_evidence"]
                            for guidance in slots.values() if guidance))

    def test_duplicate_and_unallocated_selected_cards_rejected(self):
        prior = self.prior_run()
        original = r.base.read(prior / "dataset_cards_result.json")
        allocation = r.base.read(prior / "UNIQUE_EVENT_ALLOCATION.json")
        for mutation in ("duplicate", "unallocated"):
            result = deepcopy(original)
            if mutation == "duplicate":
                result["accepted_cards"].append(deepcopy(result["accepted_cards"][0]))
                expected = "selected_source_card_duplicate"
            else:
                result["accepted_cards"][0]["source_id"] = "not-allocated"
                expected = "selected_card_source_not_allocated"
            with tempfile.TemporaryDirectory() as tmp:
                r.base.write(Path(tmp) / "dataset_cards_result.json", result)
                with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, expected):
                    a.selected_slot_cards(Path(tmp), allocation)


if __name__ == "__main__":
    unittest.main()
