from __future__ import annotations

import json
import unittest
from copy import deepcopy

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.complete_dialogue_view import (
    compose_complete_dialogue_view, validate_complete_dialogue_view,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.context_compiler_v2 import (
    ContextInvariantFailure, compile_scene_context_v2, validate_context_manifest_v2,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.continuous_round_plan_v1 import (
    compile_continuous_round_plan, validate_continuous_round_plan,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.narrative_order import (
    build_narrative_order_manifest, validate_narrative_order_manifest,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.offline_preflight_v11 import (
    INPUT, V10, build_cumulative_fixture, build_round_blueprint_fixture,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.inputs import (
    load_unified_root_read_only,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.parent_anchor_compiler import (
    compile_anchor_registry, validate_anchor_registry, validate_placement_boundary,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.placement_choice_registry import (
    compile_placement_choice_registry, compile_source_anchor_registry,
    resolve_placement_choice, resolve_source_anchor,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.planning_state_view import (
    audit_interaction_relation_separation, compile_planning_state_view,
    validate_planning_state_view,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.structural_expansion_gate import (
    compute_promotion_closure, normalize_exact_text, simulate_dependency_runtime,
    validate_connected_jump, validate_realized_expansion_structure,
    validate_round_blueprint_structure,
)


class ContinuousExpansionV11PhaseATest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.v10_before = hash_tree(V10)["tree_sha256"]
        cls.loaded = load_unified_root_read_only(INPUT)
        cls.input_before = cls.loaded["input_sha256"]
        cls.memory = json.loads((V10 / "world_memory_final.json").read_text())
        cls.links = json.loads((V10 / "interaction_links_final.json").read_text())
        cls.child = json.loads((V10 / "child_dialogue_promotable.json").read_text())
        cls.menu = json.loads((V10 / "affordance_menu.json").read_text())
        cls.old_plan = json.loads((V10 / "planner_output.json").read_text())
        cls.speaker_map = {row["display_name"]: row["entity_ref"]
                           for row in cls.memory["entities"]
                           if row.get("introduced_round") == 0}
        cls.complete = compose_complete_dialogue_view(
            original_parent=cls.loaded["parent"],
            promoted_rounds=[{"round_index": 1, "child": cls.child}],
            speaker_to_entity_ref=cls.speaker_map)
        cls.planning = compile_planning_state_view(
            memory=cls.memory, interaction_links=cls.links)
        cls.anchors = compile_anchor_registry(
            complete_dialogue=cls.complete, planning_state=cls.planning)
        cls.source_anchors = compile_source_anchor_registry(cls.anchors)
        cls.placement_choices = compile_placement_choice_registry(
            complete_dialogue=cls.complete, anchor_registry=cls.anchors)
        cls.tail = next(row for row in cls.anchors["anchors"]
                        if row["anchor_type"] == "tail_anchor")
        cls.gap = next(row for row in cls.anchors["anchors"]
                       if row["anchor_type"] == "historical_gap_anchor")
        selection = cls.old_plan["scene_goals"][1]["proposal_selection"]
        cls.blueprint = build_round_blueprint_fixture(
            source_anchor_id=cls.tail["anchor_id"],
            placement_choice_id="PC_TAIL_AFTER",
            entity_refs=cls.tail["linked_entity_refs"],
            single_action_selection=selection)
        cls.compiled = compile_continuous_round_plan(
            cls.blueprint, action_menu=cls.menu,
            source_anchor_registry=cls.source_anchors,
            placement_choice_registry=cls.placement_choices, round_index=2)

    def _plan_validation(self, plan):
        return validate_continuous_round_plan(
            plan, source_anchor_registry=self.source_anchors,
            placement_choice_registry=self.placement_choices,
            action_menu=self.menu,
            authoritative_entity_refs={row["entity_ref"]
                                       for row in self.planning["entities"]})

    def _gate(self, compiled):
        return validate_round_blueprint_structure(
            compiled_plan=compiled, source_anchor_registry=self.source_anchors,
            placement_choice_registry=self.placement_choices,
            complete_dialogue=self.complete, planning_state=self.planning)

    def _outputs_contexts(self):
        outputs, contexts = {}, {}
        for scene in self.compiled["scenes"]:
            contexts[scene["scene_ref"]] = compile_scene_context_v2(
                scene=scene, complete_dialogue=self.complete,
                source_anchor_registry=self.source_anchors,
                placement_choice_registry=self.placement_choices,
                planning_state=self.planning,
                same_round_scenes=outputs)
            refs = [row["participant_ref"] for row in scene["speaker_participants"]]
            outputs[scene["scene_ref"]] = {
                "scene_ref": scene["scene_ref"],
                "dialogue_turns": [{
                    "turn_ref": f"{scene['scene_ref']}:T{index:03d}",
                    "speaker_ref": refs[index % 2],
                    "text": f"Unique fixture statement {scene['scene_ref']} number {index}.",
                } for index in range(1, 7)]}
        return outputs, contexts

    def test_01_complete_cumulative_125(self):
        value = build_cumulative_fixture(self.loaded["parent"], self.speaker_map)
        self.assertEqual(value["turn_count"], 125)
        self.assertEqual([row["turn_count"] for row in value["rounds"]], [30, 35])

    def test_02_complete_parent_and_earlier_round_preserved(self):
        value = build_cumulative_fixture(self.loaded["parent"], self.speaker_map)
        refs = value["generation_order_turn_refs"]
        self.assertEqual(refs[:60], [row["turn_id"]
                                    for row in self.loaded["parent"]["dialogue"]["turns"]])
        self.assertEqual(len(refs[60:90]), 30)
        self.assertTrue(validate_complete_dialogue_view(value)["valid"])

    def test_03_generation_and_narrative_order_are_separate(self):
        value = build_cumulative_fixture(self.loaded["parent"], self.speaker_map)
        self.assertEqual(set(value["generation_order_turn_refs"]),
                         set(value["narrative_order_turn_refs"]))
        self.assertTrue(value["stable_refs_preserved"])

    def test_04_planning_state_excludes_parent_utterances(self):
        for field in ("active_events", "open_threads", "tasks", "key_states",
                      "knowledge_asymmetries", "causal_links"):
            self.assertFalse(any(row.get("record_type") == "parent_utterance"
                                 for row in self.planning[field]))
        self.assertTrue(validate_planning_state_view(self.planning)["valid"])

    def test_05_planning_state_filters_non_authoritative(self):
        memory = deepcopy(self.memory)
        memory["authoritative_records"].append({
            "memory_ref": "pending:M", "logical_key": "pending", "record_type": "event",
            "verification_status": "model_claimed", "payload": {"text": "leak"}})
        view = compile_planning_state_view(memory=memory, interaction_links=[])
        self.assertFalse(any(row["memory_ref"] == "pending:M" for row in view["active_events"]))

    def test_06_empty_structured_state_is_valid(self):
        source = {"root_id": "r", "entities": [], "authoritative_records": [],
                  "active_head_index": {}}
        view = compile_planning_state_view(memory=source)
        self.assertTrue(validate_planning_state_view(view)["valid"])
        self.assertTrue(view["absence_is_not_novelty_evidence"])

    def test_07_interaction_relation_separation(self):
        audit = audit_interaction_relation_separation(self.planning)
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["cooccurrence_relation_upgrade_count"], 0)

    def test_08_empty_relation_graph_allowed(self):
        source = deepcopy(self.planning)
        source["explicit_relations"] = []
        self.assertTrue(audit_interaction_relation_separation(source)["valid"])

    def test_09_explicit_relation_requires_support(self):
        source = deepcopy(self.planning)
        if source["explicit_relations"]:
            source["explicit_relations"][0]["support_turn_refs"] = []
            self.assertFalse(audit_interaction_relation_separation(source)["valid"])

    def test_10_anchor_registry_refs_exist(self):
        result = validate_anchor_registry(self.anchors, complete_dialogue=self.complete)
        self.assertTrue(result["valid"], result["errors"])
        types = {row["anchor_type"] for row in self.anchors["anchors"]}
        self.assertTrue({"tail_anchor", "historical_gap_anchor", "interaction_anchor"} <= types)

    def test_11_planner_cannot_invent_program_refs(self):
        value = deepcopy(self.blueprint)
        value["scenes"][0]["turn_ref"] = "invented:T1"
        self.assertFalse(self._plan_validation(value)["valid"])

    def test_12_placement_valid_variants(self):
        for placement, anchor in (("after", self.tail), ("before", self.tail),
                                  ("between", self.gap), ("parallel", self.tail)):
            self.assertTrue(validate_placement_boundary(
                anchor=anchor, placement=placement,
                complete_dialogue=self.complete)["valid"])

    def test_13_placement_missing_boundaries_rejected(self):
        for placement, missing, anchor in (
                ("after", "after_turn_ref", self.tail),
                ("before", "before_turn_ref", self.tail),
                ("between", "before_turn_ref", self.gap)):
            value = deepcopy(anchor)
            value["boundary"][missing] = None
            self.assertFalse(validate_placement_boundary(
                anchor=value, placement=placement,
                complete_dialogue=self.complete)["valid"])

    def test_14_between_reverse_rejected(self):
        value = deepcopy(self.gap)
        boundary = value["boundary"]
        boundary["after_turn_ref"], boundary["before_turn_ref"] = (
            boundary["before_turn_ref"], boundary["after_turn_ref"])
        self.assertFalse(validate_placement_boundary(
            anchor=value, placement="between", complete_dialogue=self.complete)["valid"])

    def test_15_parallel_interval_required(self):
        value = deepcopy(self.tail)
        value["session_ref"] = None
        value["boundary"] = {"after_turn_ref": None, "before_turn_ref": None}
        self.assertFalse(validate_placement_boundary(
            anchor=value, placement="parallel", complete_dialogue=self.complete)["valid"])

    def test_16_between_future_is_constraint_only(self):
        scene = deepcopy(self.compiled["scenes"][0])
        scene.update({
            "source_anchor_id": self.gap["anchor_id"],
            "source_anchor": resolve_source_anchor(
                self.source_anchors, self.gap["anchor_id"]),
            "placement_choice_id": "PC_GAP_0001",
            "placement_choice": resolve_placement_choice(
                self.placement_choices, "PC_GAP_0001"),
            "placement": "between", "predecessor_scene_refs": [],
                      "dependency_scene_refs": []})
        context = compile_scene_context_v2(
            scene=scene, complete_dialogue=self.complete,
            source_anchor_registry=self.source_anchors,
            placement_choice_registry=self.placement_choices,
            planning_state=self.planning)
        self.assertTrue(context["following_context_constraint_refs"])
        future = [row for row in context["p0_required_dialogue"]
                  if row["turn_ref"] in context["following_context_constraint_refs"]]
        self.assertTrue(all(row["knowledge_role"] ==
                            "following_context_constraint_only" for row in future))

    def test_17_scene_count_5_6_7_pass(self):
        selection = self.old_plan["scene_goals"][1]["proposal_selection"]
        for count in (5, 6, 7):
            plan = build_round_blueprint_fixture(
                source_anchor_id=self.tail["anchor_id"],
                placement_choice_id="PC_TAIL_AFTER",
                entity_refs=self.tail["linked_entity_refs"],
                single_action_selection=selection, scene_count=count)
            self.assertTrue(self._plan_validation(plan)["valid"], (count, self._plan_validation(plan)))

    def test_18_scene_count_4_8_rejected(self):
        selection = self.old_plan["scene_goals"][1]["proposal_selection"]
        for count in (4, 8):
            plan = build_round_blueprint_fixture(
                source_anchor_id=self.tail["anchor_id"],
                placement_choice_id="PC_TAIL_AFTER",
                entity_refs=self.tail["linked_entity_refs"],
                single_action_selection=selection, scene_count=count)
            self.assertFalse(self._plan_validation(plan)["valid"])

    def test_19_target_added_turns_contract(self):
        self.assertEqual(self.blueprint["target_added_turns"],
                         {"minimum": 30, "target": 36, "maximum": 45})

    def test_20_mainline_branch_merge_dag_passes(self):
        result = self._gate(self.compiled)
        self.assertTrue(result["valid"], result["errors"])
        self.assertGreaterEqual(result["longest_hard_dependency_chain"], 3)

    def test_21_self_or_future_dependency_rejected(self):
        value = deepcopy(self.compiled)
        value["scenes"][0]["dependency_edges"] = [{
            "predecessor_scene_ref": value["scenes"][0]["scene_ref"],
            "dependency_type": "continues", "hard_dependency": True}]
        self.assertFalse(self._gate(value)["valid"])

    def test_22_cycle_rejected(self):
        value = deepcopy(self.compiled)
        value["scenes"][0]["dependency_edges"] = [{
            "predecessor_scene_ref": value["scenes"][-1]["scene_ref"],
            "dependency_type": "continues", "hard_dependency": True}]
        self.assertFalse(self._gate(value)["valid"])

    def test_23_isolated_scene_rejected(self):
        value = deepcopy(self.compiled)
        value["scenes"][-1]["dependency_edges"] = []
        self.assertFalse(self._gate(value)["valid"])

    def test_24_short_dependency_chain_rejected(self):
        value = deepcopy(self.compiled)
        for scene in value["scenes"]:
            scene["dependency_edges"] = [edge for edge in scene["dependency_edges"]
                                         if edge["dependency_type"] == "parallel_to"]
        self.assertFalse(self._gate(value)["valid"])

    def test_25_valid_connected_jumps(self):
        linked = {"entity_refs": self.tail["linked_entity_refs"][:1],
                  "relation_refs": [], "event_refs": [], "task_refs": [],
                  "thread_refs": [], "state_refs": []}
        for kind in ("time_skip", "location_shift", "related_character_shift",
                     "connected_new_event", "connected_new_entity", "parallel_thread"):
            self.assertTrue(validate_connected_jump(
                link_type=kind, linked_existing_refs=linked,
                planning_state=self.planning,
                introduces_new_entity=kind == "connected_new_entity",
                introduces_new_event=kind == "connected_new_event")["valid"])

    def test_26_unrelated_jump_rejected(self):
        linked = {key: [] for key in ("entity_refs", "relation_refs", "event_refs",
                                      "task_refs", "thread_refs", "state_refs")}
        result = validate_connected_jump(
            link_type="location_shift", linked_existing_refs=linked,
            planning_state=self.planning, introduces_new_event=True)
        self.assertFalse(result["valid"])
        self.assertTrue(result["unrelated_topic_jump"])

    def test_27_unknown_new_person_entry_rejected(self):
        linked = {"entity_refs": ["unknown:E"], "relation_refs": [],
                  "event_refs": [], "task_refs": [], "thread_refs": [], "state_refs": []}
        self.assertFalse(validate_connected_jump(
            link_type="connected_new_entity", linked_existing_refs=linked,
            planning_state=self.planning, introduces_new_entity=True)["valid"])

    def test_28_interaction_cannot_be_friend_relation(self):
        linked = {"entity_refs": [],
                  "relation_refs": [self.planning["interaction_links"][0]["interaction_link_ref"]],
                  "event_refs": [], "task_refs": [], "thread_refs": [], "state_refs": []}
        result = validate_connected_jump(
            link_type="relationship_followup", linked_existing_refs=linked,
            planning_state=self.planning)
        self.assertFalse(result["valid"])
        self.assertIn("interaction_miscast_as_explicit_relation", result["errors"])

    def test_29_parent_true_tail_retained(self):
        context = compile_scene_context_v2(
            scene=self.compiled["scenes"][0], complete_dialogue=self.complete,
            source_anchor_registry=self.source_anchors,
            placement_choice_registry=self.placement_choices,
            planning_state=self.planning)
        expected = [f"cc-train-episode-61555:P05:T{index:03d}" for index in range(10, 17)]
        self.assertEqual(context["parent_true_tail_refs"], expected)
        self.assertTrue(context["parent_true_tail_fully_retained"])

    def test_30_scene2_sees_scene1_and_scene3_sees_scene1_scene2(self):
        outputs, contexts = self._outputs_contexts()
        refs = [scene["scene_ref"] for scene in self.compiled["scenes"]]
        self.assertIn(refs[0], contexts[refs[1]]["same_round_predecessor_refs"])
        self.assertTrue(set(refs[:2]).issubset(
            contexts[refs[2]]["same_round_predecessor_refs"]))
        self.assertTrue(all(validate_context_manifest_v2(row)["valid"]
                            for row in contexts.values()))

    def test_31_context_p0_overflow_is_invariant_failure(self):
        with self.assertRaises(ContextInvariantFailure):
            compile_scene_context_v2(
                scene=self.compiled["scenes"][0], complete_dialogue=self.complete,
                source_anchor_registry=self.source_anchors,
                placement_choice_registry=self.placement_choices,
                planning_state=self.planning,
                token_budget=10)

    def test_32_only_p2_is_trimmed(self):
        context = compile_scene_context_v2(
            scene=self.compiled["scenes"][0], complete_dialogue=self.complete,
            source_anchor_registry=self.source_anchors,
            placement_choice_registry=self.placement_choices,
            planning_state=self.planning,
            token_budget=4000)
        self.assertEqual(context["trimming_policy"], "P2_only")
        self.assertTrue(context["dropped_p2_turn_refs"])
        self.assertTrue(set(context["p0_required_turn_refs"]).issubset(
            context["selected_context_turn_refs"]))

    def test_33_frozen_context_sha_deterministic(self):
        args = dict(scene=self.compiled["scenes"][0], complete_dialogue=self.complete,
                    source_anchor_registry=self.source_anchors,
                    placement_choice_registry=self.placement_choices,
                    planning_state=self.planning)
        self.assertEqual(compile_scene_context_v2(**args)["frozen_context_sha256"],
                         compile_scene_context_v2(**args)["frozen_context_sha256"])

    def test_34_failed_predecessor_blocks_descendants_parallel_can_continue(self):
        refs = [scene["scene_ref"] for scene in self.compiled["scenes"]]
        statuses = {ref: "not_run" for ref in refs}
        statuses[refs[0]], statuses[refs[1]] = "generated", "scene_failed"
        result = simulate_dependency_runtime(self.compiled, statuses)
        self.assertIn(refs[2], result["blocked_scene_refs"])
        self.assertNotIn(refs[3], result["blocked_scene_refs"])

    def test_35_promotion_requires_hard_predecessor(self):
        refs = [scene["scene_ref"] for scene in self.compiled["scenes"]]
        result = compute_promotion_closure(
            self.compiled, requested_promoted_scene_refs=[refs[1]],
            committed_scene_refs=[])
        self.assertFalse(result["valid"])

    def test_36_independent_parallel_can_promote(self):
        refs = [scene["scene_ref"] for scene in self.compiled["scenes"]]
        result = compute_promotion_closure(
            self.compiled, requested_promoted_scene_refs=[refs[3]],
            committed_scene_refs=[])
        self.assertTrue(result["valid"])

    def test_37_narrative_actions_compile_no_pre_review_proposal(self):
        self.assertTrue(all(scene["primary_action"] for scene in self.blueprint["scenes"]))
        self.assertTrue(all(scene["compiled_proposals"] == []
                            for scene in self.compiled["scenes"]))

    def test_38_narrative_action_is_not_a_memory_claim(self):
        scene = self.compiled["scenes"][1]
        self.assertEqual(scene["durable_update_compilation_stage"],
                         "post_round_validation")
        self.assertNotIn("operation", scene["primary_action"])

    def test_39_normalized_duplicate_rules(self):
        self.assertEqual(normalize_exact_text('  “Hello   World”  '), "hello world")
        self.assertEqual(normalize_exact_text("ＡＢＣ"), "abc")

    def test_40_realized_gate_rejects_parent_exact_duplicate(self):
        outputs, contexts = self._outputs_contexts()
        first = self.compiled["scenes"][0]["scene_ref"]
        outputs[first]["dialogue_turns"][0]["text"] = self.complete["turns"][0]["text"]
        narrative = build_narrative_order_manifest(
            compiled_plan=self.compiled, complete_dialogue=self.complete)
        self.assertFalse(validate_realized_expansion_structure(
            compiled_plan=self.compiled, scene_outputs=outputs,
            context_manifests=contexts, complete_dialogue=self.complete,
            narrative_order_manifest=narrative)["valid"])

    def test_41_action_type_reuse_is_not_proposal_fingerprint_failure(self):
        value = deepcopy(self.blueprint)
        value["scenes"][2]["primary_action"] = deepcopy(
            value["scenes"][0]["primary_action"])
        value["scenes"][2]["primary_action"]["arguments"]["dimension"] = "different"
        self.assertTrue(self._plan_validation(value)["valid"])

    def test_42_duplicate_anchor_placement_development_rejected(self):
        value = deepcopy(self.compiled)
        value["scenes"][1]["new_development"] = value["scenes"][0]["new_development"]
        self.assertFalse(self._gate(value)["valid"])

    def test_43_semantic_review_never_claimed_ready(self):
        gate = self._gate(self.compiled)
        self.assertFalse(gate["semantic_novelty_review_ready"])

    def test_44_narrative_manifest_append_only(self):
        value = build_narrative_order_manifest(
            compiled_plan=self.compiled, complete_dialogue=self.complete)
        self.assertTrue(validate_narrative_order_manifest(value)["valid"])
        self.assertEqual(value["storage_mode"], "append_only")
        self.assertEqual(value["historical_transcript_rewrite_count"], 0)

    def test_45_narrative_manifest_has_branch_dependency_fields(self):
        value = build_narrative_order_manifest(
            compiled_plan=self.compiled, complete_dialogue=self.complete)
        self.assertTrue(all("dependency_scene_refs" in row and "branch_role" in row
                            for row in value["scenes"]))

    def test_46_historical_v10_and_input_unchanged(self):
        self.assertEqual(hash_tree(V10)["tree_sha256"], self.v10_before)
        self.assertEqual(load_unified_root_read_only(INPUT)["input_sha256"], self.input_before)

    def test_47_no_live_or_legacy_module_import(self):
        import ast
        from scripts.current_method.restart_n5_spell.world_memory_slim_v1 import (
            offline_preflight_v11,
        )
        forbidden = {"live_backends", "live_fresh_e2e_v7", "evidence_scope",
                     "transactional_binder", "stateful_graph_expansion",
                     "stateful_outline_planner", "stateful_graph_orchestrator"}
        files = [
            "complete_dialogue_view.py", "planning_state_view.py",
            "parent_anchor_compiler.py", "context_compiler_v2.py",
            "continuous_round_plan_v1.py", "structural_expansion_gate.py",
            "narrative_order.py", "offline_preflight_v11.py",
        ]
        found = set()
        package = offline_preflight_v11.PACKAGE
        for name in files:
            tree = ast.parse((package / name).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    found.update(alias.name.rsplit(".", 1)[-1] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    found.add(node.module.rsplit(".", 1)[-1])
        self.assertFalse(forbidden & found)

    def test_48_complete_dialogue_narrative_insertion_does_not_change_storage(self):
        root_id = self.loaded["parent"]["sample_id"]
        child = {"child_id": root_id + ":insert", "root_id": root_id,
                 "surface": "promotable", "scenes": [{
                     "scene_ref": "R09:S01", "session_ref": "R09:S01:SESSION",
                     "dialogue_turns": [{"turn_ref": "R09:S01:T001",
                                         "speaker_ref": next(iter(self.speaker_map.values())),
                                         "text": "A structurally inserted fixture turn."}]}]}
        manifest = {"ordered_scene_refs": ["R09:S01"], "scenes": [{
            "scene_ref": "R09:S01", "intended_narrative_position": 0.5}]}
        value = compose_complete_dialogue_view(
            original_parent=self.loaded["parent"],
            promoted_rounds=[{"round_index": 1, "child": child,
                              "narrative_order_manifest": manifest}],
            speaker_to_entity_ref=self.speaker_map)
        self.assertEqual(value["generation_order_turn_refs"][-1], "R09:S01:T001")
        self.assertEqual(value["narrative_order_turn_refs"][1], "R09:S01:T001")

    def test_49_before_between_parallel_get_narrative_keys(self):
        value = deepcopy(self.compiled)
        variants = ["PC_TAIL_AFTER", "PC_TAIL_BEFORE",
                    "PC_GAP_0001", "PC_PARALLEL_TAIL"]
        for scene, choice_id in zip(value["scenes"], variants):
            choice = resolve_placement_choice(self.placement_choices, choice_id)
            scene["placement_choice_id"] = choice_id
            scene["placement_choice"] = choice
            scene["placement"] = choice["placement"]
        manifest = build_narrative_order_manifest(
            compiled_plan=value, complete_dialogue=self.complete)
        keyed = {row["placement"] for row in manifest["scenes"]
                 if row["narrative_order_key"]}
        self.assertTrue({"after", "before", "between", "parallel"} <= keyed)

    def test_50_same_round_normalized_duplicate_rejected(self):
        outputs, contexts = self._outputs_contexts()
        refs = [scene["scene_ref"] for scene in self.compiled["scenes"]]
        source = outputs[refs[0]]["dialogue_turns"][0]["text"]
        outputs[refs[1]]["dialogue_turns"][0]["text"] = f'  “{source}”  '
        narrative = build_narrative_order_manifest(
            compiled_plan=self.compiled, complete_dialogue=self.complete)
        result = validate_realized_expansion_structure(
            compiled_plan=self.compiled, scene_outputs=outputs,
            context_manifests=contexts, complete_dialogue=self.complete,
            narrative_order_manifest=narrative)
        self.assertFalse(result["valid"])

    def test_51_planner_cannot_preallocate_target_version(self):
        value = deepcopy(self.blueprint)
        value["scenes"][0]["primary_action"]["arguments"]["authoritative_version"] = 1
        result = self._plan_validation(value)
        self.assertFalse(result["valid"])

    def test_52_new_event_connected_to_existing_task(self):
        task_ref = self.planning["tasks"][0]["memory_ref"]
        linked = {"entity_refs": [], "relation_refs": [], "event_refs": [],
                  "task_refs": [task_ref], "thread_refs": [], "state_refs": []}
        result = validate_connected_jump(
            link_type="connected_new_event", linked_existing_refs=linked,
            planning_state=self.planning, introduces_new_event=True)
        self.assertTrue(result["valid"], result["errors"])


if __name__ == "__main__":
    unittest.main()
