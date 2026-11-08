"""Current-scene evidence assignment and final-message capacity, entirely offline."""
import json
import unittest

from .capacity import future_audit_reserve, local_counter, proposed_budget
from .retention import (
    ADHERENCE_VERSION, INLINE_ADHERENCE_VERSION, LEGACY_ADHERENCE_VERSION, LEGACY_ADHERENCE_TEMPLATE,
    render_adherence,
)
from .run import offline_network_guard


def example(turns=2):
    child = {"root_id": "MOCK-root", "round_index": 3,
             "schema_version": "memory_event_continuation_child_v1", "scenes": []}
    for number in (5, 6):
        ref = f"R03:S{number:02d}"
        child["scenes"].append({"scene_ref": ref, "session_ref": ref,
            "memory_event_id": f"ME{number:02d}", "dialogue_turns": [
                {"turn_ref": f"{ref}:T{index:03d}", "speaker": "MOCK speaker",
                 "text": f"MOCK scene {number} observation {index}"}
                for index in range(1, turns + 1)]})
    plan = {"scenes": [{key: value for key, value in scene.items() if key != "dialogue_turns"}
                       for scene in child["scenes"]]}
    return {"parent": {"root_id": "MOCK-root", "dialogue": {"turns": []}},
            "plan": plan, "child": child, "requested_conditions": {"R03:S05": [], "R03:S06": []}}


class InlineAdherenceTests(unittest.TestCase):
    def test_dialogue_evidence_is_inline_and_conditions_explicit(self):
        inputs = example()
        inputs["requested_conditions"]["R03:S06"] = ["An exact required arrangement is established."]
        rendered = render_adherence(**inputs, prompt_version=INLINE_ADHERENCE_VERSION)
        self.assertEqual(rendered["prompt_version"], "incremental_retention_v1." + INLINE_ADHERENCE_VERSION)
        data = json.loads(rendered["messages"][0]["content"].split("\n\nINPUT DATA\n")[1])
        self.assertNotIn("actual_generated_dialogue_identity", data)
        self.assertEqual(data["downstream_conditions"], inputs["requested_conditions"])
        inline = json.loads(rendered["messages"][1]["content"])
        for actual, expected in zip(inline["scenes"], inputs["child"]["scenes"]):
            self.assertEqual(actual["scene_ref"], expected["scene_ref"])
            self.assertEqual(actual["dialogue_turns"], [{"turn_ref": turn["turn_ref"],
                "generated_turn": {"speaker": turn["speaker"], "text": turn["text"]}}
                for turn in expected["dialogue_turns"]])
        rules = rendered["messages"][0]["content"].split("\n\nINPUT DATA\n")[0]
        self.assertIn("goal_evidence_refs is an array of unique strings", rules)
        self.assertIn("Copy each ID verbatim", rules)
        self.assertIn("condition_results MUST be []", rules)
        self.assertIn("once each, copying condition text verbatim", rules)

    def test_raw_turn_objects_are_preserved_verbatim(self):
        inputs = example(turns=1)
        raw_one = '{ "speaker" : "MOCK speaker", "text" : "MOCK scene 5 observation 1", "extra": ["\\u4e2d", {"nested":true}] }'
        raw_two = '{"text":"MOCK scene 6 observation 1","speaker":"MOCK speaker"}'
        raw = ' \n{"scenes":[{"dialogue_turns":[' + raw_one + ']},{"dialogue_turns":[' + raw_two + ']}]} \n'
        rendered = render_adherence(**inputs, generated_response=raw)
        annotated = rendered["messages"][1]["content"]
        self.assertEqual(annotated.count(raw_one), 1)
        self.assertEqual(annotated.count(raw_two), 1)
        self.assertNotIn("\\\\u4e2d", annotated)

    def test_duplicate_json_keys_and_raw_child_mismatch_rejected(self):
        inputs = example(turns=1)
        raw = {"scenes": [{"dialogue_turns": [{"speaker": turn["speaker"], "text": turn["text"]}
                           for turn in scene["dialogue_turns"]]} for scene in inputs["child"]["scenes"]]}
        duplicate = json.dumps(raw).replace('"speaker": "MOCK speaker"', '"speaker":"BAD","speaker":"MOCK speaker"', 1)
        with self.assertRaisesRegex(ValueError, "duplicate_key"):
            render_adherence(**inputs, generated_response=duplicate)
        raw["scenes"][1]["dialogue_turns"][0]["text"] = "Different saved dialogue"
        with self.assertRaisesRegex(ValueError, "child_mismatch"):
            render_adherence(**inputs, generated_response=raw)

    def test_incorrect_turn_assignment_cannot_be_sent(self):
        inputs = example()
        inputs["child"]["scenes"][1]["dialogue_turns"][0]["turn_ref"] = "R02:S01:T001"
        with self.assertRaisesRegex(ValueError, "turn_identity"):
            render_adherence(**inputs)

    def test_missing_conditions_and_unknown_prompt_version_fail_closed(self):
        inputs = example()
        inputs["requested_conditions"].pop("R03:S05")
        with self.assertRaisesRegex(ValueError, "requested_conditions_shape"):
            render_adherence(**inputs)
        with self.assertRaisesRegex(ValueError, "unsupported_adherence_prompt_version"):
            render_adherence(**example(), prompt_version="unknown")

    def test_explicit_legacy_renderer_keeps_raw_message_and_rules(self):
        inputs = example()
        raw = json.dumps({"scenes": []})
        rendered = render_adherence(**inputs, generated_response=raw, prompt_version=LEGACY_ADHERENCE_VERSION)
        self.assertEqual(rendered["messages"][1]["content"], raw)
        self.assertIn(LEGACY_ADHERENCE_TEMPLATE, rendered["messages"][0]["content"])
        self.assertNotIn("EVIDENCE OUTPUT CONTRACT", rendered["messages"][0]["content"])

    def test_raw_segment_capacity_covers_maximum_turns_and_escaping(self):
        with offline_network_guard():
            counter = local_counter()
            inputs = example(turns=48)
            for scene in inputs["child"]["scenes"]:
                for turn in scene["dialogue_turns"]:
                    turn["text"] += ' quote " backslash \\ and\nline'
            raw = json.dumps({"scenes": [{"dialogue_turns": [
                {"speaker": turn["speaker"], "text": turn["text"]} for turn in scene["dialogue_turns"]]
                } for scene in inputs["child"]["scenes"]]}, ensure_ascii=True)
            budget = proposed_budget()
            self.assertLessEqual(counter.count_text(raw), budget["max_output_tokens"]["realizer"])
            reserve = future_audit_reserve(parent=inputs["parent"], plan=inputs["plan"],
                conditions=inputs["requested_conditions"], budget=budget, counter=counter)
            rendered = render_adherence(**inputs, generated_response=raw)
            self.assertLessEqual(counter.count_tokens(rendered["messages"]),
                reserve["known_final_messages_tokens"] + reserve["future_content_bound_tokens"])
            self.assertTrue(reserve["fits"])
            self.assertEqual(reserve["maximum_raw_turn_segments"], 96)
            self.assertLess(reserve["future_content_bound_tokens"], 8 * budget["max_output_tokens"]["realizer"])
            legacy = future_audit_reserve(parent=inputs["parent"], plan=inputs["plan"],
                conditions=inputs["requested_conditions"], budget=budget, counter=counter,
                prompt_version=LEGACY_ADHERENCE_VERSION)
            self.assertEqual(legacy["future_content_bound_tokens"], 8192 + 2 * counter.max_token_bytes)


if __name__ == "__main__":
    unittest.main()
