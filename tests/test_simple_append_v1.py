import json
import unittest
from copy import deepcopy
from pathlib import Path

from scripts.current_method.restart_n5_spell.simple_append_v1 import (
    append_expansion_round,
    assemble_expanded_dialogue,
    build_evidence_alias_registry,
    build_mcq_batch_prompt,
    build_planner_prompt,
    build_round_realizer_prompt,
    build_tail_context,
    build_tail_context_from_expanded_dialogue,
    compile_round_plan,
    normalize_round_realizer_output,
    shadow_project_flat_realizer_v1,
    validate_expanded_dialogue,
    validate_appended_round,
    validate_mcq_batch,
    validate_planner_output,
)
from scripts.current_method.restart_n5_spell.simple_append_v1.contracts import (
    PLANNER_SCHEMA_VERSION,
    REALIZER_SCHEMA_VERSION,
)
from scripts.current_method.restart_n5_spell.simple_append_v1.preflight import run
from scripts.current_method.restart_n5_spell.simple_append_v1.qa import (
    MCQ_BATCH_SCHEMA_VERSION,
)


PARENT_PATH = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-inputs/"
    "conversation_chronicles_v1/roots/cc-train-episode-61555/unified_root.json"
)


def planner_fixture(parent_speakers):
    a, b = parent_speakers[:2]
    return {
        "planner_schema_version": PLANNER_SCHEMA_VERSION,
        "round_goal": "Continue the final parent situation through a concrete outcome.",
        "scenes": [{
            "goal": f"Develop scene {index}.",
            "participants": [a, b] if index == 1 else [a, b, "Visitor C"],
            "new_development": f"A distinct development occurs in scene {index}.",
            "interaction_beats": ["A proposal is made.", "A constraint appears.",
                                  "The speakers choose a response."],
            "transition_from_previous": ("Continue the parent tail." if index == 1
                                         else f"Continue scene {index - 1}."),
            "desired_outcome": f"Scene {index} produces a distinct outcome.",
        } for index in range(1, 4)],
    }


def realizer_fixture(compiled, turns_per_scene=16):
    dialogues = {}
    counts = ([turns_per_scene] * len(compiled["scenes"])
              if isinstance(turns_per_scene, int) else list(turns_per_scene))
    for scene, scene_turn_count in zip(compiled["scenes"], counts):
        speakers = scene["participants"]
        dialogues[scene["scene_ref"]] = [{
                "speaker": speakers[index % len(speakers)],
                "text": f"Unique {scene['scene_ref']} fixture utterance {index + 1}.",
            } for index in range(scene_turn_count)]
    return {"realizer_schema_version": REALIZER_SCHEMA_VERSION,
            "scene_dialogues": dialogues}


class SimpleAppendV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent = json.loads(PARENT_PATH.read_text(encoding="utf-8"))
        cls.context = build_tail_context(cls.parent, session_count=2)
        cls.plan = planner_fixture(cls.context["parent_speakers"])
        cls.compiled = compile_round_plan(cls.plan, tail_context=cls.context)

    def normalized_child(self):
        return normalize_round_realizer_output(
            realizer_fixture(self.compiled), compiled_plan=self.compiled,
            parent_utterances=[row["text"] for row in self.parent["dialogue"]["turns"]])

    def test_tail_context_uses_last_two_complete_sessions(self):
        self.assertEqual(2, self.context["selected_session_count"])
        self.assertEqual(
            [row["session_id"] for row in self.parent["session_metadata"][-2:]],
            self.context["selected_session_ids"])
        self.assertEqual(self.parent["dialogue"]["turns"][-1]["turn_id"],
                         self.context["parent_tail_turn_id"])

    def test_planner_is_small_sequential_outline_without_v11_fields(self):
        result = validate_planner_output(
            self.plan, parent_speakers=self.context["parent_speakers"])
        self.assertTrue(result["valid"], result)
        text = json.dumps(self.plan)
        for forbidden in ("placement_choice_id", "action_id", "memory_id",
                          "candidate_update_hints", "dependency_edges"):
            self.assertNotIn(forbidden, text)

    def test_planner_allows_naturally_connected_new_speaker(self):
        self.assertTrue(validate_planner_output(
            self.plan, parent_speakers=self.context["parent_speakers"])["valid"])
        bad = deepcopy(self.plan)
        bad["scenes"][0]["participants"] = ["New A", "New B"]
        result = validate_planner_output(
            bad, parent_speakers=self.context["parent_speakers"])
        self.assertIn("scene_01:not_connected_to_parent_speaker", result["errors"])

    def test_planner_allows_five_declared_speakers_for_a_long_scene(self):
        value = deepcopy(self.plan)
        value["scenes"][1]["participants"] = [
            self.context["parent_speakers"][0],
            self.context["parent_speakers"][1],
            "Connected Specialist C",
            "Connected Manager D",
            "Connected Client E",
        ]
        result = validate_planner_output(
            value, parent_speakers=self.context["parent_speakers"])
        self.assertTrue(result["valid"], result)

    def test_compiler_assigns_three_sequential_scene_refs(self):
        self.assertEqual(["R01:S01", "R01:S02", "R01:S03"],
                         [row["scene_ref"] for row in self.compiled["scenes"]])
        self.assertEqual([None, "R01:S01", "R01:S02"],
                         [row["predecessor_scene_ref"]
                          for row in self.compiled["scenes"]])
        self.assertEqual("parent_tail_only", self.compiled["append_mode"])
        self.assertEqual(self.context["parent_tail_turn_id"],
                         self.compiled["source_parent_tail_turn_id"])
        self.assertTrue(all(row["turn_guidance"]["hard_minimum"] == 9
                            for row in self.compiled["scenes"]))
        roster = self.compiled["speaker_roster_contract"]
        self.assertEqual("simple_append_speaker_roster_v1",
                         roster["speaker_roster_contract_version"])
        self.assertEqual(
            {row["scene_ref"]: row["participants"]
             for row in self.compiled["scenes"]},
            roster["scene_core_speakers"])
        self.assertEqual(
            list(dict.fromkeys(
                participant
                for scene in self.compiled["scenes"]
                for participant in scene["participants"])),
            roster["round_speakers"])

    def test_prompts_freeze_the_declared_speaker_roster(self):
        planner_prompt = build_planner_prompt(self.context)["prompt"]
        self.assertIn("complete speaking cast for the round", planner_prompt)
        self.assertIn("If a new specialist will speak, name them now", planner_prompt)
        realizer_prompt = build_round_realizer_prompt(
            compiled_plan=self.compiled, tail_context=self.context)["prompt"]
        self.assertIn('"frozen_speaker_roster"', realizer_prompt)
        self.assertIn("exact whitelist for every speaker field", realizer_prompt)
        for speaker in self.compiled["speaker_roster_contract"]["round_speakers"]:
            self.assertIn(speaker, realizer_prompt)

    def test_one_round_realizer_returns_three_long_scenes(self):
        result = self.normalized_child()
        self.assertTrue(result["valid"], result)
        self.assertEqual(48, result["turn_count"])
        self.assertEqual(3, result["long_scene_count"])
        self.assertEqual([], result["warnings"])

    def test_round_roster_speaker_can_enter_earlier_scene(self):
        fixture = realizer_fixture(self.compiled)
        fixture["scene_dialogues"]["R01:S01"][1]["speaker"] = "Visitor C"
        result = normalize_round_realizer_output(
            fixture, compiled_plan=self.compiled,
            parent_utterances=[row["text"] for row in self.parent["dialogue"]["turns"]])
        self.assertTrue(result["valid"], result)
        self.assertIn(
            "scene_01:round_roster_speaker_outside_core:Visitor C",
            result["warnings"])

    def test_unplanned_round_speaker_is_still_rejected(self):
        fixture = realizer_fixture(self.compiled)
        fixture["scene_dialogues"]["R01:S01"][1]["speaker"] = "Unknown D"
        result = normalize_round_realizer_output(
            fixture, compiled_plan=self.compiled,
            parent_utterances=[row["text"] for row in self.parent["dialogue"]["turns"]])
        self.assertFalse(result["valid"])
        self.assertIn("scene_01:turn_002:speaker", result["errors"])

    def test_same_scene_exact_duplicate_blocks_but_cross_scene_duplicate_warns(self):
        same_scene = realizer_fixture(self.compiled)
        same_scene["scene_dialogues"]["R01:S01"][1]["text"] = \
            same_scene["scene_dialogues"]["R01:S01"][0]["text"]
        blocked = normalize_round_realizer_output(
            same_scene, compiled_plan=self.compiled,
            parent_utterances=[row["text"] for row in self.parent["dialogue"]["turns"]])
        self.assertFalse(blocked["valid"])
        self.assertIn("scene_01:turn_002:child_exact_duplicate", blocked["errors"])

        cross_scene = realizer_fixture(self.compiled)
        cross_scene["scene_dialogues"]["R01:S02"][1]["text"] = \
            cross_scene["scene_dialogues"]["R01:S01"][0]["text"]
        admitted = normalize_round_realizer_output(
            cross_scene, compiled_plan=self.compiled,
            parent_utterances=[row["text"] for row in self.parent["dialogue"]["turns"]])
        self.assertTrue(admitted["valid"], admitted)
        self.assertEqual(1, admitted["cross_scene_exact_duplicate_count"])
        self.assertEqual({
            "first_turn_ref": "R01:S01:T001",
            "duplicate_turn_ref": "R01:S02:T002",
        }, admitted["cross_scene_exact_duplicates"][0])

    def test_single_parent_exact_copy_warns_but_excessive_copy_blocks(self):
        parent_turns = self.parent["dialogue"]["turns"]
        one_copy = realizer_fixture(self.compiled)
        one_copy["scene_dialogues"]["R01:S02"][0]["text"] = parent_turns[0]["text"]
        admitted = normalize_round_realizer_output(
            one_copy, compiled_plan=self.compiled,
            parent_utterances=[row["text"] for row in parent_turns],
            parent_turn_refs=[row["turn_id"] for row in parent_turns])
        self.assertTrue(admitted["valid"], admitted)
        self.assertEqual(1, admitted["parent_exact_copy_count"])
        self.assertFalse(admitted["parent_exact_copy_excessive"])
        self.assertEqual({
            "duplicate_turn_ref": "R01:S02:T001",
            "source_turn_refs": [parent_turns[0]["turn_id"]],
        }, admitted["parent_exact_copies"][0])

        excessive = realizer_fixture(self.compiled)
        for index in range(3):
            excessive["scene_dialogues"]["R01:S02"][index]["text"] = \
                parent_turns[index]["text"]
        blocked = normalize_round_realizer_output(
            excessive, compiled_plan=self.compiled,
            parent_utterances=[row["text"] for row in parent_turns],
            parent_turn_refs=[row["turn_id"] for row in parent_turns])
        self.assertFalse(blocked["valid"])
        self.assertEqual(3, blocked["parent_exact_copy_count"])
        self.assertTrue(blocked["parent_exact_copy_excessive"])
        self.assertIn("realizer_parent_exact_copy_excessive", blocked["errors"])

    def test_scene_still_requires_at_least_one_core_participant(self):
        fixture = realizer_fixture(self.compiled)
        fixture["scene_dialogues"]["R01:S01"] = [{
            "speaker": "Visitor C",
            "text": f"Visitor-only scene fixture turn {index + 1}.",
        } for index in range(16)]
        result = normalize_round_realizer_output(
            fixture, compiled_plan=self.compiled,
            parent_utterances=[row["text"] for row in self.parent["dialogue"]["turns"]])
        self.assertFalse(result["valid"])
        self.assertIn("scene_01:core_speaker_coverage", result["errors"])

    def test_canary3_round_roster_shadow_replay(self):
        canary = Path(
            "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
            "conversation_chronicles_episode_61555_simple_append_v1_live_canary_3")
        historical = json.loads(
            (canary / "realizer/round_realizer_output.json").read_text())
        historical_plan = json.loads(
            (canary / "planner/compiled_round_plan.json").read_text())
        result = normalize_round_realizer_output(
            historical, compiled_plan=historical_plan,
            parent_utterances=[row["text"] for row in self.parent["dialogue"]["turns"]])
        self.assertTrue(result["valid"], result)
        self.assertEqual(48, result["turn_count"])
        self.assertTrue(any("Colleague C" in warning
                            for warning in result["warnings"]))

    def test_turn_limits_are_flexible_but_have_hard_safety_guards(self):
        result = normalize_round_realizer_output(
            realizer_fixture(self.compiled, turns_per_scene=12),
            compiled_plan=self.compiled,
            parent_utterances=[row["text"] for row in self.parent["dialogue"]["turns"]])
        self.assertTrue(result["valid"], result)
        self.assertIn("round_soft_turn_target_not_met", result["warnings"])
        too_short = normalize_round_realizer_output(
            realizer_fixture(self.compiled, turns_per_scene=5),
            compiled_plan=self.compiled,
            parent_utterances=[row["text"] for row in self.parent["dialogue"]["turns"]])
        self.assertFalse(too_short["valid"])
        self.assertIn("realizer_round_too_short", too_short["errors"])
        one_short_scene = normalize_round_realizer_output(
            realizer_fixture(self.compiled, turns_per_scene=[8, 19, 19]),
            compiled_plan=self.compiled,
            parent_utterances=[row["text"] for row in self.parent["dialogue"]["turns"]])
        self.assertFalse(one_short_scene["valid"])
        self.assertIn("scene_01:hard_turn_guard", one_short_scene["errors"])

    def test_long_scene_can_use_round_budget_without_old_24_turn_cap(self):
        result = normalize_round_realizer_output(
            realizer_fixture(self.compiled, turns_per_scene=[17, 18, 30]),
            compiled_plan=self.compiled,
            parent_utterances=[row["text"] for row in self.parent["dialogue"]["turns"]])
        self.assertTrue(result["valid"], result)
        self.assertEqual(65, result["turn_count"])
        self.assertNotIn("scene_03:hard_turn_guard", result["errors"])

    def test_scene_safety_cap_still_rejects_runaway_single_scene(self):
        result = normalize_round_realizer_output(
            realizer_fixture(self.compiled, turns_per_scene=[49, 9, 9]),
            compiled_plan=self.compiled,
            parent_utterances=[row["text"] for row in self.parent["dialogue"]["turns"]])
        self.assertFalse(result["valid"])
        self.assertIn("scene_01:hard_turn_guard", result["errors"])

    def test_parent_is_unchanged_and_child_is_appended_at_tail(self):
        parent_before = deepcopy(self.parent)
        child = self.normalized_child()["child"]
        expanded = assemble_expanded_dialogue(parent=self.parent, child=child)
        validation = validate_expanded_dialogue(expanded, parent=self.parent)
        self.assertTrue(validation["valid"], validation)
        self.assertEqual(parent_before, self.parent)
        self.assertEqual(60, expanded["parent_turn_count"])
        self.assertEqual(48, expanded["child_turn_count"])
        self.assertEqual(self.context["parent_tail_turn_id"],
                         child["source_parent_tail_turn_id"])
        self.assertEqual(61, expanded["child_boundary"][
            "first_child_turn_index_1_based"])
        self.assertEqual("R01:S01:T001", expanded["child_boundary"][
            "first_child_turn_ref"])

    def test_later_round_uses_accumulated_tail_and_appends_without_rewrite(self):
        round1_child = self.normalized_child()["child"]
        round1 = assemble_expanded_dialogue(
            parent=self.parent, child=round1_child)
        tail = build_tail_context_from_expanded_dialogue(round1, session_count=2)
        self.assertEqual(["R01:S02:SESSION", "R01:S03:SESSION"],
                         tail["selected_session_ids"])
        self.assertEqual("R01:S03:T016", tail["parent_tail_turn_id"])
        plan = planner_fixture(tail["parent_speakers"])
        compiled = compile_round_plan(plan, tail_context=tail, round_index=2)
        normalized = normalize_round_realizer_output(
            realizer_fixture(compiled), compiled_plan=compiled,
            parent_utterances=[row["text"] for row in round1["turns"]])
        self.assertTrue(normalized["valid"], normalized)
        round2 = append_expansion_round(
            prior_expanded=round1, child=normalized["child"])
        validation = validate_appended_round(
            round2, prior_expanded=round1)
        self.assertTrue(validation["valid"], validation)
        self.assertEqual(round1["turns"], round2["turns"][:108])
        self.assertEqual(156, round2["turn_count"])
        self.assertEqual(96, round2["child_turn_count"])
        self.assertEqual("R02:S01:T001",
                         round2["latest_round_boundary"]["first_turn_ref"])

    def test_prompts_send_tail_once_and_qa_sees_complete_sample(self):
        child = self.normalized_child()["child"]
        expanded = assemble_expanded_dialogue(parent=self.parent, child=child)
        planner = build_planner_prompt(self.context)
        realizer = build_round_realizer_prompt(
            compiled_plan=self.compiled, tail_context=self.context)
        mcq = build_mcq_batch_prompt(expanded)
        self.assertEqual(1, realizer["prompt"].count(
            self.context["selected_turns"][-1]["text"]))
        self.assertEqual(3, planner["prompt"].count(
            '"goal":"distinct goal for scene'))
        self.assertIn("Copy every existing Parent speaker name exactly", planner["prompt"])
        self.assertIn("participants as its core focus", realizer["prompt"])
        self.assertIn("Never invent a speaker absent from the complete round cast",
                      realizer["prompt"])
        self.assertIn('"scene_ref":"R01:S03"', realizer["prompt"])
        output_contract = realizer["prompt"].split(
            "OUTPUT EXACTLY THIS SHAPE WITH EVERY PLANNED SCENE KEY", 1)[1]
        self.assertNotIn('"scenes":[', output_contract)
        self.assertIn('"scene_dialogues":{', output_contract)
        self.assertIn(self.parent["dialogue"]["turns"][0]["text"], mcq["prompt"])
        self.assertIn('"evidence_ref":"D102"', mcq["prompt"])
        self.assertNotIn('"turn_ref":"R01:S03:T010"', mcq["prompt"])
        self.assertIn("Create at least 8 grounded", mcq["prompt"])
        self.assertIn("no question-count upper admission gate", mcq["prompt"])
        self.assertEqual("simple_append_full_dialogue_mcq_batch_prompt_v18",
                         mcq["prompt_version"])
        self.assertIn("identify one factual answer slot", mcq["prompt"])
        self.assertIn("changing only one material fact", mcq["prompt"])
        self.assertIn("Do not swap in an unrelated event", mcq["prompt"])
        self.assertIn("equally plausible to a solver", mcq["prompt"])
        self.assertIn("generic or extreme distractors", mcq["prompt"])
        self.assertIn("soft preference, not a rejection gate", mcq["prompt"])
        self.assertLess(planner["token_estimate"], 10000)
        self.assertLess(realizer["token_estimate"], 15000)

    def test_mcq_gate_is_batch_level_not_every_question_child(self):
        child = self.normalized_child()["child"]
        expanded = assemble_expanded_dialogue(parent=self.parent, child=child)
        aliases = build_evidence_alias_registry(expanded)
        parent_ref = aliases["turn_ref_to_alias"][
            expanded["turns"][0]["turn_ref"]]
        child_ref = aliases["turn_ref_to_alias"][
            expanded["turns"][-1]["turn_ref"]]
        batch = {
            "mcq_batch_schema_version": MCQ_BATCH_SCHEMA_VERSION,
            "questions": [{
                "question_type": "causal",
                "question": "What parent fact is stated?",
                "options": {"A": "One", "B": "Two", "C": "Three", "D": "Four"},
                "correct_option": "A", "evidence_turn_refs": [parent_ref],
            }, {
                "question_type": "state_change",
                "question": "What child fact is stated?",
                "options": {"A": "Alpha", "B": "Beta", "C": "Gamma", "D": "Delta"},
                "correct_option": "B", "evidence_turn_refs": [child_ref],
            }],
        }
        result = validate_mcq_batch(batch, expanded_dialogue=expanded,
                                    expected_count=2)
        self.assertTrue(result["valid"], result)
        self.assertEqual(1, result["child_supported_question_count"])
        self.assertFalse(result["every_question_must_use_child"])

    def test_nested_realizer_scene_shape_is_rejected(self):
        nested = {
            "realizer_schema_version": REALIZER_SCHEMA_VERSION,
            "scenes": [{"scene_ref": "R01:S01", "dialogue_turns": []}],
        }
        result = normalize_round_realizer_output(
            nested, compiled_plan=self.compiled,
            parent_utterances=[row["text"] for row in self.parent["dialogue"]["turns"]])
        self.assertFalse(result["valid"])
        self.assertIn("realizer_top_level_schema", result["errors"])

    def test_canary2_flat_output_shadow_projects_without_admission(self):
        canary = Path(
            "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
            "conversation_chronicles_episode_61555_simple_append_v1_live_canary_2")
        historical = json.loads((canary / "realizer/round_realizer_output.json").read_text())
        historical_plan = json.loads((canary / "planner/compiled_round_plan.json").read_text())
        shadow = shadow_project_flat_realizer_v1(historical)
        self.assertTrue(shadow["shadow_projection_only"])
        self.assertFalse(shadow["authoritative_admission_allowed"])
        result = normalize_round_realizer_output(
            shadow["projection"], compiled_plan=historical_plan,
            parent_utterances=[row["text"] for row in self.parent["dialogue"]["turns"]])
        self.assertTrue(result["valid"], result)
        self.assertEqual([13, 11, 15], [len(scene["dialogue_turns"])
                         for scene in result["child"]["scenes"]])
        self.assertEqual(39, result["turn_count"])
        self.assertIn("round_soft_turn_target_not_met", result["warnings"])

    def test_offline_preflight_closes_without_api(self):
        report = run(parent_path=PARENT_PATH)
        self.assertEqual("passed", report["status"])
        self.assertEqual(0, report["physical_api_requests"])
        self.assertTrue(all(report["checks"].values()), report["checks"])


if __name__ == "__main__":
    unittest.main()
