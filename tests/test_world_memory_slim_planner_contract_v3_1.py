from __future__ import annotations

import json
import unittest
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from scripts.current_method.restart_n5_spell.unified_dialogue_root import read_unified_parent
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.controlled_expansion import (
    compile_action_menu_v3,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.interaction_links import (
    compile_source_interaction_links,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.planner_contract_v3_1 import (
    FROZEN_PLANNER_FIELDS, PLANNER_ACTION_VIEW_VERSION,
    PLANNER_CONTRACT_VERSION, compile_allowed_proposal_bundles,
    compile_planner_action_view, validate_planner_action_view,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.planner_v3 import (
    PLANNER_PROMPT_VERSION_V3, PLANNER_RETRY_PROMPT_VERSION_V3,
    PLANNER_SCHEMA_VERSION_V3, build_planner_v3_prompt,
    build_planner_v3_retry_prompt,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_plan_validation import (
    compile_validate_round_plan,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.source_memory_initializer_v1 import (
    authoritative_affordance_view, initialize_source_world_memory,
)


INPUT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-inputs/conversation_chronicles_v1/"
    "roots/cc-train-episode-61555/unified_root.json")
HISTORICAL_V7 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_world_memory_slim_fresh_e2e_v7")
HISTORICAL_V7_SHA = "4d0fea20040bf849678f56bd6e4f9c88c5cfaa94f16e7381e16ccd0eeba2e4aa"
HISTORICAL_V8 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_world_memory_slim_fresh_e2e_v8")


def _sessions(turns):
    grouped = defaultdict(list)
    for turn in turns:
        grouped[turn["session_id"]].append(turn)
    return [{"session_id": key, "turns": value}
            for key, value in sorted(grouped.items())]


def _speakers(refs):
    return [
        {"participant_ref": refs[0], "scene_role": "primary",
         "intents": ["raise one concrete issue", "seek a usable next step"]},
        {"participant_ref": refs[1], "scene_role": "supporting",
         "intents": ["test the proposal against a practical constraint"]},
    ]


def _scene(refs, *, proposals, bundle_id=None, suffix="one"):
    if bundle_id is not None:
        proposal_selection = {
            "selection_type": "bundle",
            "bundle_id": bundle_id,
            "proposal_contents": {
                row["action_id"]: row["proposed_content"] for row in proposals
            },
        }
    elif proposals:
        if len(proposals) != 1:
            raise ValueError("test_fixture_single_scene_requires_one_proposal")
        proposal_selection = {
            "selection_type": "single_action",
            "action_id": proposals[0]["action_id"],
            "proposed_content": proposals[0]["proposed_content"],
        }
    else:
        proposal_selection = {"selection_type": "none"}
    return {
        "goal": f"Develop a concrete extension {suffix}.",
        "speaker_participants": _speakers(refs),
        "interaction_beats": ["The issue is raised.", "An assumption is tested.",
                              "A concrete direction emerges."],
        "continuity_requirements": ["Preserve authoritative source facts."],
        "proposal_selection": proposal_selection,
        "desired_outcome": "A natural, independently reviewable exchange.",
    }


def _plan(refs, scenes):
    return {
        "planner_schema_version": PLANNER_SCHEMA_VERSION_V3,
        "scene_budget": {"minimum": 2, "target": 3, "maximum": 4},
        "round_arc": {"starting_situation": "A practical issue remains open.",
                      "central_tension": "The participants prefer different approaches.",
                      "ending_direction": "They leave with a concrete direction."},
        "scene_goals": scenes,
    }


class PlannerContractV31Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = json.loads(INPUT.read_text(encoding="utf-8"))
        cls.parent = read_unified_parent(INPUT)
        cls.source = initialize_source_world_memory(cls.root, cls.parent)
        cls.memory = cls.source["memory"]
        cls.refs = [row["entity_ref"] for row in cls.memory["entities"]]
        cls.links = compile_source_interaction_links(
            root_id=cls.root["root_id"], sessions=_sessions(cls.parent["dialogue"]["turns"]),
            speaker_to_entity=cls.source["speaker_to_entity_ref"])
        cls.menu = compile_action_menu_v3(
            authoritative_affordance_view(cls.memory), root_id=cls.root["root_id"],
            interaction_links=cls.links)
        cls.roles = [{"entity_ref": row["entity_ref"], "display_name": row["display_name"]}
                     for row in cls.memory["entities"]]
        cls.view = compile_planner_action_view(cls.menu, role_table=cls.roles)
        cls.bundles = compile_allowed_proposal_bundles(cls.menu)
        cls.action_by_operation = {row["operation"]: row for row in cls.menu["actions"]}
        cls.history_before = hash_tree(HISTORICAL_V7)["tree_sha256"]
        cls.v8_history_before = hash_tree(HISTORICAL_V8)["tree_sha256"]
        cls.historical = [json.loads(
            (HISTORICAL_V7 / f"attempts/planner-v3-a{index:02d}.json").read_text(
                encoding="utf-8"))["parsed_response"] for index in (1, 2, 3)]

    @classmethod
    def tearDownClass(cls):
        assert hash_tree(HISTORICAL_V7)["tree_sha256"] == cls.history_before
        assert hash_tree(HISTORICAL_V8)["tree_sha256"] == cls.v8_history_before

    def _validate(self, plan, *, memory=None, menu=None, view=None, bundles=None):
        return compile_validate_round_plan(
            plan, action_menu=menu or self.menu,
            authoritative_memory=memory or self.memory,
            interaction_links=self.links, round_index=1,
            planner_action_view=view or self.view,
            allowed_bundles=bundles or self.bundles)

    def _corrected(self):
        event = self.action_by_operation["add_event"]
        person = self.action_by_operation["add_person"]
        relation = self.action_by_operation["add_relation"]
        return _plan(self.refs, [
            _scene(self.refs, proposals=[{
                "action_id": event["action_id"],
                "proposed_content": {"description": "They schedule a short planning meeting.",
                                     "narrative_time_expression": "the following afternoon"},
            }], suffix="one"),
            _scene(self.refs, proposals=[{
                "action_id": person["action_id"],
                "proposed_content": {"display_role": "a neutral scheduling coordinator"},
            }, {
                "action_id": relation["action_id"],
                "proposed_content": {"relation_type": "collaborates_with"},
            }], bundle_id="B0001", suffix="two"),
            _scene(self.refs, proposals=[], suffix="three"),
        ])

    def test_versions_are_v33(self):
        self.assertEqual("slim_world_planner_contract_v3_3", PLANNER_CONTRACT_VERSION)
        self.assertEqual("planner_action_view_v1", PLANNER_ACTION_VIEW_VERSION)
        self.assertEqual("slim_world_planner_v3_prompt_v4", PLANNER_PROMPT_VERSION_V3)
        self.assertEqual("slim_world_planner_v3_retry_prompt_v4", PLANNER_RETRY_PROMPT_VERSION_V3)

    def test_action_view_hides_internal_and_frozen_fields(self):
        self.assertTrue(validate_planner_action_view(self.menu, self.view)["valid"])
        for internal, projected in zip(self.menu["actions"], self.view["actions"]):
            self.assertNotIn("operation", projected)
            self.assertNotIn("operation_class", projected)
            self.assertNotIn("payload_schema", projected)
            self.assertNotIn("frozen_payload", projected)
            writable = projected["writable_payload_schema"]
            self.assertIs(writable["additionalProperties"], False)
            self.assertFalse(set(writable["properties"]) & FROZEN_PLANNER_FIELDS)
            self.assertFalse(set(writable["properties"]) &
                             set((internal.get("frozen_payload") or {}).keys()))

    def test_program_frozen_refs_are_merged_without_planner_echo(self):
        corrected = self._corrected()
        result = self._validate(corrected)
        self.assertTrue(result["valid"], result)
        event = result["compiled_round_plan"]["scenes"][0]["compiled_proposals"][0]
        self.assertEqual(self.refs, event["compiled_payload"]["participant_refs"])
        self.assertNotIn("participant_refs",
                         corrected["scene_goals"][0]["proposal_selection"]["proposed_content"])

    def test_frozen_field_echo_and_unknown_field_are_rejected(self):
        plan = self._corrected()
        content = plan["scene_goals"][0]["proposal_selection"]["proposed_content"]
        content["participant_refs"] = self.refs
        content["invented"] = "no"
        result = self._validate(plan)
        self.assertTrue(any(value.startswith("frozen_field_present") for value in result["errors"]))
        self.assertTrue(any(value.startswith("unknown_writable_field") for value in result["errors"]))

    def test_zero_or_one_proposal_scene_and_corrected_fixture_pass(self):
        result = self._validate(self._corrected())
        self.assertTrue(result["valid"], result)
        self.assertEqual([1, 2, 0], [len(row["compiled_proposals"])
                                    for row in result["compiled_round_plan"]["scenes"]])

    def test_legal_bundle_compiles_dependency_and_atomic_group(self):
        result = self._validate(self._corrected())
        proposals = result["compiled_round_plan"]["scenes"][1]["compiled_proposals"]
        self.assertEqual([proposals[0]["proposal_ref"]], proposals[1]["proposal_dependencies"])
        self.assertEqual(proposals[0]["atomic_group_ref"], proposals[1]["atomic_group_ref"])
        self.assertEqual("B0001", proposals[1]["proposal_bundle_id"])
        self.assertTrue(proposals[1]["compiled_payload"]["target_ref"].startswith("R01:S02:TMP:"))

    def test_live_schema_cannot_express_free_pair_or_three_proposals(self):
        plan = self._corrected()
        plan["scene_goals"][1]["proposal_selection"] = {
            "selection_type": "single_action",
            "action_id": self.action_by_operation["add_person"]["action_id"],
            "proposed_content": {"display_role": "a coordinator"},
            "second_action": self.action_by_operation["add_shared_task"]["action_id"],
        }
        failed = self._validate(plan)
        self.assertTrue(any(value.startswith("invalid_proposal_selection_shape")
                            for value in failed["errors"]))
        plan = self._corrected()
        plan["scene_goals"][0]["proposal_selection"]["proposal_contents"] = {
            "A0001": {}, "A0002": {}, "A0003": {}}
        failed = self._validate(plan)
        self.assertTrue(any(value.startswith("single_action_has_bundle_fields")
                            for value in failed["errors"]))

    def test_same_action_across_scenes_is_rejected(self):
        plan = self._corrected()
        plan["scene_goals"][2]["proposal_selection"] = deepcopy(
            plan["scene_goals"][0]["proposal_selection"])
        result = self._validate(plan)
        self.assertTrue(any(value.startswith("duplicate_action_id") for value in result["errors"]))

    def test_different_action_ids_of_same_operation_are_allowed(self):
        menu = deepcopy(self.menu)
        event = deepcopy(self.action_by_operation["add_event"])
        event["action_id"] = "A0099"
        menu["actions"].append(event)
        view = compile_planner_action_view(menu, role_table=self.roles)
        bundles = compile_allowed_proposal_bundles(menu)
        plan = self._corrected()
        plan["scene_goals"][1] = _scene(self.refs, proposals=[{
            "action_id": "A0099", "proposed_content": {
                "description": "They hold a distinct follow-up review."}}], suffix="two")
        result = self._validate(plan, menu=menu, view=view, bundles=bundles)
        self.assertTrue(result["valid"], result)

    def test_duplicate_round_fingerprint_is_rejected(self):
        menu = deepcopy(self.menu)
        event = deepcopy(self.action_by_operation["add_event"])
        event["action_id"] = "A0099"
        menu["actions"].append(event)
        view = compile_planner_action_view(menu, role_table=self.roles)
        bundles = compile_allowed_proposal_bundles(menu)
        plan = self._corrected()
        plan["scene_goals"][1] = _scene(self.refs, proposals=[{
            "action_id": "A0099", "proposed_content": deepcopy(
                plan["scene_goals"][0]["proposal_selection"]["proposed_content"])}], suffix="two")
        result = self._validate(plan, menu=menu, view=view, bundles=bundles)
        self.assertEqual("planner_repairable", result["classification"])
        self.assertTrue(any("duplicate_round_proposal" in value for value in result["errors"]))

    def test_exact_existing_fact_collision_but_distinct_event_passes(self):
        plan = self._corrected()
        event_action = self.action_by_operation["add_event"]
        payload = deepcopy(event_action["frozen_payload"])
        payload.update(plan["scene_goals"][0]["proposal_selection"]["proposed_content"])
        memory = deepcopy(self.memory)
        record = {"memory_ref": "existing:M999999", "logical_key": "fixture:event",
                  "record_type": "event", "operation": "add_event", "payload": payload,
                  "verification_status": "source_grounded", "authoritative_version": 1}
        memory["authoritative_records"].append(record)
        memory["active_head_index"]["fixture:event"] = record["memory_ref"]
        collision = self._validate(plan, memory=memory)
        self.assertTrue(any(value.startswith("existing_fact_collision")
                            for value in collision["errors"]))
        distinct = deepcopy(plan)
        distinct["scene_goals"][0]["proposal_selection"]["proposed_content"][
            "description"] = "They schedule a genuinely different review meeting."
        self.assertTrue(self._validate(distinct, memory=memory)["valid"])

    def test_relation_uses_canonical_enum_and_frozen_endpoints(self):
        plan = self._corrected()
        relation_id = self.action_by_operation["add_relation"]["action_id"]
        plan["scene_goals"][1]["proposal_selection"]["proposal_contents"][
            relation_id]["relation_type"] = "coworker"
        valid = self._validate(plan)
        self.assertTrue(valid["valid"], valid)
        payload = valid["compiled_round_plan"]["scenes"][1]["compiled_proposals"][1][
            "compiled_payload"]
        self.assertEqual(self.refs[0], payload["source_ref"])
        self.assertTrue(payload["target_ref"].startswith("R01:S02:TMP:"))
        plan["scene_goals"][1]["proposal_selection"]["proposal_contents"][relation_id][
            "relation_type"] = "colleague"
        invalid = self._validate(plan)
        self.assertTrue(any(value.startswith("invalid_canonical_enum")
                            for value in invalid["errors"]))

    def test_initial_and_retry_prompts_only_use_projection(self):
        prompt = build_planner_v3_prompt(
            root_id=self.root["root_id"], role_table=self.roles,
            action_menu=self.menu, context_summary={"summary": "safe"},
            planner_action_view=self.view, allowed_bundles=self.bundles)
        self.assertIn(PLANNER_PROMPT_VERSION_V3.rsplit("_prompt", 1)[0], PLANNER_PROMPT_VERSION_V3)
        self.assertNotIn('"frozen_payload"', prompt)
        self.assertNotIn('"payload_schema":', prompt)
        self.assertIn('"writable_payload_schema"', prompt)
        retry = build_planner_v3_retry_prompt(
            root_id=self.root["root_id"], frozen_action_menu=self.menu,
            frozen_role_table=self.roles, frozen_context={"secret": "OLD RESPONSE TEXT"},
            frozen_scene_budget={"minimum": 2, "target": 3, "maximum": 4},
            errors=["frozen_field_present:scene=1:action=A0001:field=participant_refs"],
            relevant_action_ids=["A0001"],
            frozen_authoritative_memory_sha256="m" * 64,
            frozen_interaction_links_sha256="i" * 64,
            frozen_round_context_sha256="c" * 64,
            frozen_planner_action_view=self.view,
            frozen_allowed_bundles=self.bundles)
        self.assertNotIn("OLD RESPONSE TEXT", retry)
        self.assertNotIn('"frozen_payload"', retry)
        self.assertNotIn('"payload_schema":', retry)
        self.assertIn("frozen_field_present", retry)

    def test_initial_and_scene_count_retry_end_with_two_scene_minimum_example(self):
        prompt = build_planner_v3_prompt(
            root_id=self.root["root_id"], role_table=self.roles,
            action_menu=self.menu, context_summary={"summary": "safe"},
            planner_action_view=self.view, allowed_bundles=self.bundles)
        initial_example = json.loads(prompt.rsplit("\n", 1)[-1])
        self.assertEqual(2, len(initial_example["scene_goals"]))
        for scene in initial_example["scene_goals"]:
            self.assertEqual(2, len(scene["speaker_participants"]))
            self.assertIsInstance(scene["continuity_requirements"], list)
        self.assertNotEqual("none", initial_example["scene_goals"][0][
            "proposal_selection"]["selection_type"])
        self.assertEqual("none", initial_example["scene_goals"][1][
            "proposal_selection"]["selection_type"])

        retry = build_planner_v3_retry_prompt(
            root_id=self.root["root_id"], frozen_action_menu=self.menu,
            frozen_role_table=self.roles, frozen_context={"omitted": True},
            frozen_scene_budget={"minimum": 2, "target": 3, "maximum": 4},
            errors=["planner_scene_count"], relevant_action_ids=["A0003", "A0007"],
            frozen_authoritative_memory_sha256="m" * 64,
            frozen_interaction_links_sha256="i" * 64,
            frozen_round_context_sha256="c" * 64,
            frozen_planner_action_view=self.view,
            frozen_allowed_bundles=self.bundles)
        retry_example = json.loads(retry.rsplit("\n", 1)[-1])
        self.assertEqual(2, len(retry_example["scene_goals"]))
        self.assertIn("never return a one-scene plan", retry)
        action_material = retry.split(
            "FROZEN COMPLETE PLANNER ACTION VIEW\n", 1)[1].split(
                "\n\nFROZEN COMPLETE ATOMIC BUNDLE VIEW", 1)[0]
        self.assertEqual(len(self.view["actions"]), len(json.loads(action_material)))

    def test_retry_instructions_make_speaker_and_continuity_repairs_explicit(self):
        retry = build_planner_v3_retry_prompt(
            root_id=self.root["root_id"], frozen_action_menu=self.menu,
            frozen_role_table=self.roles, frozen_context={},
            frozen_scene_budget={"minimum": 2, "target": 3, "maximum": 4},
            errors=["scene:1:speaker_count", "scene:1:continuity_requirements"],
            relevant_action_ids=["A0001"],
            frozen_planner_action_view=self.view,
            frozen_allowed_bundles=self.bundles)
        self.assertIn("2-4 distinct participant_ref", retry)
        self.assertIn("must be a JSON array of non-empty strings", retry)
        self.assertIn("FROZEN ROLE TABLE", retry)

    def test_historical_v8_cardinality_shadow_replay_and_minimum_fix(self):
        attempts = [json.loads(
            (HISTORICAL_V8 / f"attempts/planner-v3-a{index:02d}.json").read_text(
                encoding="utf-8"))["parsed_response"] for index in (1, 2, 3)]
        results = [self._validate(plan) for plan in attempts]
        self.assertIn("planner_scene_count", results[0]["errors"])
        self.assertTrue(all(not result["valid"] for result in results))
        self.assertTrue(any("schema" in error for result in results
                            for error in result["errors"]))
        self.assertTrue(self._validate(self._corrected())["valid"])
        self.assertEqual(self.v8_history_before, hash_tree(HISTORICAL_V8)["tree_sha256"])

    def test_projection_corruption_is_local_invariant(self):
        corrupt = deepcopy(self.view)
        corrupt["actions"][0]["writable_payload_schema"]["properties"][
            "participant_refs"] = {"type": "array"}
        result = self._validate(self._corrected(), view=corrupt)
        self.assertEqual("local_invariant_failure", result["classification"])

    def test_speaker_connectivity_problem_is_scene_infeasible(self):
        plan = self._corrected()
        plan["scene_goals"][0]["speaker_participants"][0]["participant_ref"] = "unknown:E"
        result = self._validate(plan)
        self.assertEqual("scene_infeasible", result["classification"])

    def test_historical_v7_attempts_are_rejected_by_live_v33_schema(self):
        results = [self._validate(plan) for plan in self.historical]
        self.assertTrue(all(not result["valid"] for result in results))
        self.assertTrue(all(any("schema" in error for error in result["errors"])
                            for result in results))

    def test_historical_v7_is_read_only(self):
        self.assertEqual(HISTORICAL_V7_SHA, self.history_before)
        self.assertEqual(self.history_before, hash_tree(HISTORICAL_V7)["tree_sha256"])


if __name__ == "__main__":
    unittest.main()
