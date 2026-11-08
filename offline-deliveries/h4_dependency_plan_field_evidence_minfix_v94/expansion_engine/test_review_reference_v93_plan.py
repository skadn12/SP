"""Bind reference reminders to actual immutable plans, not assumed step indices.

Archived model failures below stay failures. These offline tests demonstrate
input/evidence integrity, not that future model reviews reliably find meaning.
"""
from copy import deepcopy
import json
from pathlib import Path
import unittest

import event_state as event
from test_event_state_v92 import review_fixture


DELIVERIES = Path(__file__).resolve().parents[2]
V91_REQUESTS = (DELIVERIES / "h4_planner_cardinality_expansion_batch_v91"
                / "expansion_engine/runs/ten_roots_B_scene01/requests")
V92_REQUESTS = (DELIVERIES / "h4_card_fidelity_event_consistency_minfix_v92"
                / "evidence/model_review_probe/requests")


def archive_context(root):
    path = sorted(V91_REQUESTS.glob(f"*-cc-train-episode-{root}-B-S01-event-state-*-1/REQUEST.json"))[0]
    request = json.loads(path.read_text())
    return json.loads(request["messages"][0]["content"].split("EVENT_STATE_INPUT_JSON\n", 1)[1])


def parse_request(request):
    text = request["messages"][0]["content"].split("EVENT_STATE_INPUT_JSON\n", 1)[1]
    context, end = json.JSONDecoder().raw_decode(text)
    return context, text[end:]


class V93PlanReferenceTests(unittest.TestCase):
    def setUp(self):
        self.context = archive_context("20432")
        self.scene = self.context["scene_plan"]
        self.cards = self.context["dataset_cards"]
        self.history = self.context["cumulative_history"]

    def test_actual_four_developments_cover_six_distinct_steps_without_renumbering(self):
        before = deepcopy((self.scene, self.cards))
        table = event.review_reference_context(self.scene, self.cards)
        self.assertEqual(table["allowed_development_indices"], [1, 2, 3, 4])
        self.assertEqual(table["numbered_developments"], [
            {"development_index": index, "text": text}
            for index, text in enumerate(self.scene["developments"], 1)])
        steps = {row["step_id"]: row for row in table["actual_card_targets"][0]["step_targets"]}
        self.assertEqual({key: row["mapped_development_indices"] for key, row in steps.items()},
                         {"s1": [1], "s2": [2], "s3": [2], "s4": [3], "s5": [4], "s6": [4]})
        self.assertEqual(steps["s2"]["mapped_developments"], steps["s3"]["mapped_developments"])
        self.assertEqual(steps["s5"]["mapped_developments"], steps["s6"]["mapped_developments"])
        self.assertEqual(steps["s6"]["mapped_developments"][0]["text"], self.scene["developments"][3])
        self.assertIn("(s5)", self.scene["developments"][3])
        self.assertIn("(s6)", self.scene["developments"][3])
        self.assertEqual(before, (self.scene, self.cards))

    def test_dependency_table_retains_relation_mapping_instead_of_endpoint_union(self):
        table = event.review_reference_context(self.scene, self.cards)
        deps = table["actual_card_targets"][0]["dependency_targets"]
        self.assertEqual([row["mapped_development_indices"] for row in deps], [[1], [2], [3]])
        self.assertEqual(deps[0]["card_requirement"], self.cards[0]["dependencies"][0])
        # In the actual first dependency, before s1 is at dev1 and after s2 at
        # dev2. The original relation-bearing assignment remains dev1 only.
        self.assertEqual(deps[0]["mapped_developments"], [table["numbered_developments"][0]])
        other = archive_context("38601")
        other_table = event.review_reference_context(other["scene_plan"], other["dataset_cards"])
        self.assertEqual([row["mapped_development_indices"]
                          for row in other_table["actual_card_targets"][0]["dependency_targets"]],
                         [[2], [3], [4]])

    def test_source_allowlist_and_text_bytes_survive_request_and_reference_table(self):
        cards = deepcopy(self.cards)
        sentinel = "SOURCE_DIALOGUE_SENTINEL_93_SHOULD_NEVER_ENTER_CONTEXT"
        cards[0].update(source_dialogue=sentinel, source_anchors=[{"quote": sentinel}])
        cards[0]["steps"][0]["source_quote"] = sentinel
        cards[0]["dependencies"][0]["source_evidence"] = sentinel
        scene = deepcopy(self.scene)
        scene["developments"][0] = "Emily’s exact words: ‘new campus’ — then a literal … and ... remain data (s1)."
        before = deepcopy((scene, cards, self.history))
        request = event.build_request(self.context["root_id"], "dataset_new_event", scene,
                                      self.history, cards, self.context["latest_state"])
        context, suffix = parse_request(request)
        table = json.loads(suffix.split("ACTUAL REVIEW REFERENCE TABLE\n", 1)[1])
        self.assertNotIn(sentinel, request["messages"][0]["content"])
        self.assertEqual(context["scene_plan"], scene)
        self.assertEqual(context["cumulative_history"], self.history)
        self.assertEqual(table["numbered_developments"][0]["text"].encode("utf-8"),
                         scene["developments"][0].encode("utf-8"))
        self.assertEqual(table["actual_card_targets"][0]["step_targets"][0]["card_requirement"],
                         {key: cards[0]["steps"][0][key] for key in ("step_id", "actor_id", "action", "state")})
        self.assertEqual(before, (scene, cards, self.history))
        self.assertTrue(table["reference_enumeration_only_not_semantic_approval"])

    def test_current_reminder_and_actual_table_follow_unchanged_failed_raw_review(self):
        failed_raw = {
            "note": "OLD_REVIEW_DATA_SENTINEL with invalid index 6; CURRENT EVENT REVIEW CHECK: is data.",
            "plan_checks": [{"development_index": 6, "quote": "Elena confirms..."}],
        }
        repair = {"previous_raw": failed_raw, "instruction": "Return a complete corrected review."}
        before = deepcopy(repair)
        request = event.build_request(self.context["root_id"], "dataset_new_event", self.scene,
                                      self.history, self.cards, self.context["latest_state"], repair=repair)
        context, suffix = parse_request(request)
        self.assertEqual(context["targeted_repair"], repair)
        self.assertEqual(context["targeted_repair"]["previous_raw"], failed_raw)
        self.assertEqual(repair, before)
        self.assertTrue(suffix.startswith("\nCURRENT EVENT REVIEW CHECK:\n"))
        self.assertNotIn("OLD_REVIEW_DATA_SENTINEL", suffix)
        table = json.loads(suffix.split("ACTUAL REVIEW REFERENCE TABLE\n", 1)[1])
        self.assertEqual(table, event.review_reference_context(self.scene, self.cards))
        self.assertEqual(table["allowed_development_indices"], [1, 2, 3, 4])
        self.assertNotEqual(table["allowed_development_indices"][-1], failed_raw["plan_checks"][0]["development_index"])

    def test_invalid_map_is_rejected_before_the_table_or_request_can_legitimize_it(self):
        mutations = (
            ("index-six", lambda scene: scene["card_development_links"][0].update(development_indices=[6]),
             "card_development_link_indices_invalid"),
            ("missing-step", lambda scene: scene["card_development_links"][1].update(card_step_ids=["s2"]),
             "card_steps_not_fully_mapped"),
            ("wrong-card", lambda scene: scene.update(dataset_card_id="unassigned"), "dataset_card_id_mismatch"),
            ("oversized", lambda scene: scene["developments"].extend(["Another new step.", "Sixth new step."]),
             "scene_developments_count_invalid"),
        )
        for name, mutate, code in mutations:
            with self.subTest(mutation=name):
                scene = deepcopy(self.scene)
                mutate(scene)
                before = deepcopy(scene)
                for operation in (lambda: event.review_reference_context(scene, self.cards),
                                  lambda: event.build_request(self.context["root_id"], "dataset_new_event", scene,
                                                             self.history, self.cards, self.context["latest_state"])):
                    with self.assertRaises(event.EventStateValidationError) as caught:
                        operation()
                    self.assertIn(code, [row["code"] for row in caught.exception.errors])
                    self.assertEqual(before, scene)

    def test_actual_v92_ellipsis_and_wrong_indices_remain_invalid_without_rewriting(self):
        cases = (("194259", "event_state_plan_evidence_quote_invalid"),
                 ("20432", "event_state_plan_evidence_index_invalid"),
                 ("38601", "event_state_card_mapped_developments_mismatch"))
        for root, expected_code in cases:
            with self.subTest(root=root):
                request_dir = sorted(V92_REQUESTS.glob(f"*-cc-train-episode-{root}-plan-review-2"))[0]
                context = json.loads(json.loads((request_dir / "REQUEST.json").read_text())[
                    "messages"][0]["content"].split("EVENT_STATE_INPUT_JSON\n", 1)[1])
                raw = json.loads((request_dir / "RESPONSE.json").read_text())["text"]
                before = raw.encode("utf-8")
                with self.assertRaises(event.EventStateValidationError) as caught:
                    event.validate(raw, context["root_id"], "dataset_new_event", context["scene_plan"],
                                   context["cumulative_history"], cards=context["dataset_cards"])
                self.assertIn(expected_code, [row["code"] for row in caught.exception.errors])
                self.assertEqual(before, raw.encode("utf-8"))

    def test_literal_ellipsis_can_be_evidence_but_invented_ellipsis_cannot(self):
        scene = deepcopy(self.scene)
        scene["developments"][0] = "Emily says ‘college ... near coast’ before giving a new preference (s1)."
        response = review_fixture(scene, self.history, self.cards, root=self.context["root_id"])
        response["plan_checks"][0]["plan_evidence"] = [{"development_index": 1, "quote": "college ... near coast"}]
        event.validate(response, self.context["root_id"], "dataset_new_event", scene, self.history, cards=self.cards)
        response["plan_checks"][0]["plan_evidence"][0]["quote"] = "Emily says ... new preference"
        with self.assertRaises(event.EventStateValidationError) as caught:
            event.validate(response, self.context["root_id"], "dataset_new_event", scene, self.history, cards=self.cards)
        self.assertIn("event_state_plan_evidence_quote_invalid", [row["code"] for row in caught.exception.errors])

    def test_reference_reminder_never_overrides_a_substantive_card_negative(self):
        response = review_fixture(self.scene, self.history, self.cards, root=self.context["root_id"])
        response["card_checks"][0]["dependency_checks"][0].update(
            verdict="revise", reason="The mapped development states intent but does not express its required relation to the later question.")
        table = event.review_reference_context(self.scene, self.cards)
        result = event.validate(response, self.context["root_id"], "dataset_new_event", self.scene,
                                self.history, cards=self.cards)
        self.assertTrue(table["reference_enumeration_only_not_semantic_approval"])
        self.assertTrue(event.requires_replan(result))
        self.assertEqual(event.binding_plan_checks(result)[0]["dependency_index"], 1)
        self.assertEqual(event.binding_plan_checks(result)[0]["mapped_development_indices"], [1])


if __name__ == "__main__":
    unittest.main()
