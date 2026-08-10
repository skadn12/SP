from __future__ import annotations

import json
import unittest
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from scripts.current_method.restart_n5_spell.core import sha256_file
from scripts.current_method.restart_n5_spell.unified_dialogue_root import read_unified_parent
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.controlled_expansion import (
    compile_action_menu_v3,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.interaction_links import (
    compile_source_interaction_links,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.live_role_runtime import (
    load_json_object_strict,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.planner_contract_v3_1 import (
    PLANNER_BUNDLE_VERSION,
    PLANNER_CONTRACT_VERSION,
    compile_allowed_proposal_bundles,
    compile_planner_action_view,
    validate_bundle_selection_registry,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.planner_legacy_bundle_audit import (
    classify_legacy_bundle_plan,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.planner_v3 import (
    PLANNER_PROMPT_VERSION_V3,
    PLANNER_RETRY_PROMPT_VERSION_V3,
    PLANNER_SCHEMA_VERSION_V3,
    build_planner_v3_prompt,
    build_planner_v3_retry_prompt,
    planner_proposal_selection_schema,
    validate_planner_v3,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_plan_validation import (
    compile_validate_round_plan,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.semantic_import_audit_v7 import (
    audit_v7_recursive_import_graph,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.source_memory_initializer_v1 import (
    authoritative_affordance_view,
    initialize_source_world_memory,
)


PROJECT = Path("/root/autodl-tmp/gxy-selfplay/current_method_github")
MODULE = PROJECT / "scripts/current_method/restart_n5_spell/world_memory_slim_v1"
INPUT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-inputs/conversation_chronicles_v1/"
    "roots/cc-train-episode-61555/unified_root.json")
V9 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_world_memory_slim_fresh_e2e_v9")

PROTECTED_SHAS = {
    "realizer_v3.py": "b6ae9dab749cfcb7eb39d7e0f833b9e2d87e2fcab8fd5e7cd4ecb8b16be8a463",
    "dialogue_quality.py": "6cc41bbf5392e6e6a4da7002aba185e60f932f95defd5f493bcd134d4ced9380",
    "claim_validator_v2.py": "78666101a85593368fd723999c97f60b5e5d11caca12a03e3e89f142d685d090",
    "evidence_scope.py": "e45d8e77ae1e725f4f1650b68b7ac86e2de6248dd2ee348a1b68f53da374862a",
    "transactional_binder.py": "b0fbeb0ba4a0d2a8a53c7b8c8045df6b2d7aa19e3b32a061dd4b993801c11f2d",
    "append_only_memory.py": "2911b70dc020e56ceea4dbe0aad4507cd72f87d51b20e7354e9a25ec7bf1618d",
    "promotion.py": "63759ee59ac18d6b7f65c82b0e26d83cd2921f309bb525a81818d87bb72ab9a0",
    "provenance_ledger.py": "cacaa44ef195288d0c6683715e7f23cc8739c36a9635cff2e9d6485d0c226a69",
}


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
         "intents": ["test one practical constraint"]},
    ]


def _scene(refs, selection, suffix):
    return {
        "goal": f"Develop extension {suffix}.",
        "speaker_participants": _speakers(refs),
        "interaction_beats": ["Raise the issue.", "Test one assumption.",
                              "Reach a concrete outcome."],
        "continuity_requirements": ["Preserve authoritative source facts."],
        "proposal_selection": selection,
        "desired_outcome": "A natural independently reviewable exchange.",
    }


class PlannerBundleSelectionV33Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v9_before = hash_tree(V9)["tree_sha256"]
        cls.input_before = sha256_file(INPUT)
        root = json.loads(INPUT.read_text(encoding="utf-8"))
        parent = read_unified_parent(INPUT)
        source = initialize_source_world_memory(root, parent)
        cls.root = root
        cls.memory = source["memory"]
        cls.refs = [row["entity_ref"] for row in cls.memory["entities"]]
        cls.links = compile_source_interaction_links(
            root_id=root["root_id"], sessions=_sessions(parent["dialogue"]["turns"]),
            speaker_to_entity=source["speaker_to_entity_ref"])
        cls.menu = compile_action_menu_v3(
            authoritative_affordance_view(cls.memory), root_id=root["root_id"],
            interaction_links=cls.links)
        cls.roles = [{"entity_ref": row["entity_ref"],
                      "display_name": row["display_name"]}
                     for row in cls.memory["entities"]]
        cls.view = compile_planner_action_view(cls.menu, role_table=cls.roles)
        cls.bundles = compile_allowed_proposal_bundles(cls.menu)
        cls.by_operation = {row["operation"]: row for row in cls.menu["actions"]}
        cls.bundle = cls.bundles["bundles"][0]

    @classmethod
    def tearDownClass(cls):
        assert cls.v9_before == hash_tree(V9)["tree_sha256"]
        assert cls.input_before == sha256_file(INPUT)

    def corrected_plan(self):
        person = self.by_operation["add_person"]
        relation = self.by_operation["add_relation"]
        event = self.by_operation["add_event"]
        return {
            "planner_schema_version": PLANNER_SCHEMA_VERSION_V3,
            "scene_budget": {"minimum": 2, "target": 3, "maximum": 4},
            "round_arc": {
                "starting_situation": "A practical issue remains open.",
                "central_tension": "The participants prefer different approaches.",
                "ending_direction": "They leave with a concrete direction.",
            },
            "scene_goals": [
                _scene(self.refs, {
                    "selection_type": "bundle", "bundle_id": "B0001",
                    "proposal_contents": {
                        person["action_id"]: {
                            "display_role": "a neutral scheduling coordinator"},
                        relation["action_id"]: {
                            "relation_type": "collaborates_with"},
                    },
                }, "one"),
                _scene(self.refs, {
                    "selection_type": "single_action",
                    "action_id": event["action_id"],
                    "proposed_content": {
                        "description": "They schedule a short planning meeting.",
                        "narrative_time_expression": "the following afternoon",
                    },
                }, "two"),
                _scene(self.refs, {"selection_type": "none"}, "three"),
            ],
        }

    def validate(self, plan, *, bundles=None, view=None):
        return compile_validate_round_plan(
            plan, action_menu=self.menu, authoritative_memory=self.memory,
            interaction_links=self.links, round_index=1,
            planner_action_view=view or self.view,
            allowed_bundles=bundles or self.bundles)

    def test_versions_and_oneof_are_frozen(self):
        self.assertEqual("slim_world_planner_contract_v3_3", PLANNER_CONTRACT_VERSION)
        self.assertEqual("planner_bundle_selection_v1", PLANNER_BUNDLE_VERSION)
        self.assertEqual("slim_world_planner_v3_prompt_v4", PLANNER_PROMPT_VERSION_V3)
        self.assertEqual("slim_world_planner_v3_retry_prompt_v4",
                         PLANNER_RETRY_PROMPT_VERSION_V3)
        schema = planner_proposal_selection_schema()
        self.assertEqual(3, len(schema["oneOf"]))
        self.assertTrue(all(row["additionalProperties"] is False
                            for row in schema["oneOf"]))

    def test_none_selection_and_extra_field(self):
        plan = self.corrected_plan()
        self.assertTrue(self.validate(plan)["valid"])
        plan["scene_goals"][2]["proposal_selection"]["action_id"] = "A0001"
        result = self.validate(plan)
        self.assertTrue(any(error.startswith("invalid_proposal_selection_shape")
                            for error in result["errors"]))

    def test_single_selection_and_bundle_fields(self):
        plan = self.corrected_plan()
        self.assertTrue(self.validate(plan)["valid"])
        plan["scene_goals"][1]["proposal_selection"]["bundle_id"] = "B0001"
        result = self.validate(plan)
        self.assertIn("single_action_has_bundle_fields:scene=2", result["errors"])
        self.assertIn("single_action_illegally_carries_bundle:scene=2", result["errors"])

    def test_single_selection_with_proposal_contents_is_rejected(self):
        plan = self.corrected_plan()
        plan["scene_goals"][1]["proposal_selection"]["proposal_contents"] = {}
        result = self.validate(plan)
        self.assertIn("single_action_has_bundle_fields:scene=2", result["errors"])

    def test_bundle_complete_missing_extra_and_top_action(self):
        self.assertTrue(self.validate(self.corrected_plan())["valid"])
        required = self.bundle["required_action_ids"]
        for missing in required:
            plan = self.corrected_plan()
            del plan["scene_goals"][0]["proposal_selection"]["proposal_contents"][missing]
            self.assertTrue(any(error.startswith("incomplete_bundle_actions")
                                for error in self.validate(plan)["errors"]))
        plan = self.corrected_plan()
        plan["scene_goals"][0]["proposal_selection"]["proposal_contents"]["A9999"] = {}
        self.assertTrue(any(error.startswith("extra_bundle_actions")
                            for error in self.validate(plan)["errors"]))
        plan = self.corrected_plan()
        plan["scene_goals"][0]["proposal_selection"]["action_id"] = required[0]
        self.assertIn("bundle_has_single_action_fields:scene=1",
                      self.validate(plan)["errors"])

    def test_unknown_bundle_is_rejected(self):
        plan = self.corrected_plan()
        plan["scene_goals"][0]["proposal_selection"]["bundle_id"] = "B9999"
        self.assertTrue(any(error.startswith("unknown_bundle_id")
                            for error in self.validate(plan)["errors"]))

    def test_bundle_only_actions_cannot_be_single_or_split(self):
        required = self.bundle["required_action_ids"]
        plan = self.corrected_plan()
        contents = plan["scene_goals"][0]["proposal_selection"]["proposal_contents"]
        plan["scene_goals"][0]["proposal_selection"] = {
            "selection_type": "single_action", "action_id": required[0],
            "proposed_content": contents[required[0]],
        }
        plan["scene_goals"][1]["proposal_selection"] = {
            "selection_type": "single_action", "action_id": required[1],
            "proposed_content": contents[required[1]],
        }
        result = self.validate(plan)
        self.assertEqual("planner_repairable", result["classification"])
        self.assertEqual(2, sum(error.startswith("bundle_only_action_selected_singly")
                                for error in result["errors"]))
        self.assertTrue(any(error.startswith("split_bundle_across_scenes")
                            for error in result["errors"]))

    def test_bundle_and_actions_are_round_unique(self):
        plan = self.corrected_plan()
        plan["scene_goals"][1]["proposal_selection"] = deepcopy(
            plan["scene_goals"][0]["proposal_selection"])
        result = self.validate(plan)
        self.assertTrue(any(error.startswith("duplicate_bundle_id")
                            for error in result["errors"]))
        self.assertTrue(any(error.startswith("bundle_action_reused_elsewhere")
                            for error in result["errors"]))

    def test_unregistered_pair_is_not_expressible_in_live_shape(self):
        plan = self.corrected_plan()
        plan["scene_goals"][1]["proposal_selection"] = {
            "selection_type": "single_action",
            "action_id": self.by_operation["create_state"]["action_id"],
            "proposed_content": {},
            "second_action_id": self.by_operation["introduce_external_constraint"]["action_id"],
        }
        result = self.validate(plan)
        self.assertTrue(any(error.startswith("invalid_proposal_selection_shape")
                            for error in result["errors"]))

    def test_duplicate_json_keys_are_rejected_before_dict_overwrite(self):
        with self.assertRaisesRegex(ValueError, "duplicate_json_object_key"):
            load_json_object_strict('{"scene_goals":[],"scene_goals":[]}')
        with self.assertRaisesRegex(ValueError, "duplicate_json_object_key"):
            load_json_object_strict(
                '{"proposal_selection":{"selection_type":"none",'
                '"selection_type":"single_action"}}')
        with self.assertRaisesRegex(ValueError, "duplicate_bundle_action_key:A0007"):
            load_json_object_strict(
                '{"proposal_contents":{"A0007":{},"A0007":{},"A0003":{}}}')
        with self.assertRaisesRegex(ValueError, "duplicate_json_object_key:description"):
            load_json_object_strict(
                '{"proposed_content":{"description":"one","description":"two"}}')

    def test_bundle_registry_corruption_is_local_invariant(self):
        corrupt = deepcopy(self.bundles)
        corrupt["bundles"][0]["required_action_ids"].append("A9999")
        result = self.validate(self.corrected_plan(), bundles=corrupt)
        self.assertEqual("local_invariant_failure", result["classification"])
        corrupt = deepcopy(self.bundles)
        relation_id = self.bundle["required_action_ids"][1]
        corrupt["bundles"][0]["frozen_payload_overrides"][relation_id][
            "relation_type"] = "sibling"
        self.assertFalse(validate_bundle_selection_registry(
            self.menu, corrupt, self.view)["valid"])
        corrupt = deepcopy(self.bundles)
        corrupt["bundles"][0]["dependency_edges"][0]["depends_on_action_ids"] = [
            corrupt["bundles"][0]["dependency_edges"][0]["action_id"]]
        self.assertFalse(validate_bundle_selection_registry(
            self.menu, corrupt, self.view)["valid"])
        corrupt = deepcopy(self.bundles)
        duplicate = deepcopy(corrupt["bundles"][0])
        duplicate["bundle_id"] = "B0002"
        corrupt["bundles"].append(duplicate)
        result = self.validate(self.corrected_plan(), bundles=corrupt)
        self.assertEqual("local_invariant_failure", result["classification"])

    def test_corrected_bundle_expands_in_required_order_and_internal_contract(self):
        result = self.validate(self.corrected_plan())
        self.assertTrue(result["valid"], result)
        proposals = result["compiled_round_plan"]["scenes"][0]["compiled_proposals"]
        self.assertEqual(self.bundle["required_action_ids"],
                         [row["action_id"] for row in proposals])
        self.assertEqual(["R01:S01:P01", "R01:S01:P02"],
                         [row["proposal_ref"] for row in proposals])
        self.assertEqual(["R01:S01:P01"], proposals[1]["proposal_dependencies"])
        self.assertEqual(proposals[0]["atomic_group_ref"],
                         proposals[1]["atomic_group_ref"])
        self.assertEqual(proposals[0]["temporary_refs"][0],
                         proposals[1]["compiled_payload"]["target_ref"])
        internal_keys = {
            "proposal_ref", "action_id", "operation", "operation_class",
            "compiled_payload", "connected_memory_refs", "target_memory_ref",
            "pre_state", "authoritative_version", "proposal_fingerprint",
            "temporary_refs", "proposal_dependencies", "atomic_group_ref",
            "proposal_bundle_id", "lifecycle", "provenance",
        }
        self.assertEqual(internal_keys, set(proposals[0]))
        for forbidden in ("proposal_selection", "selection_type", "bundle_id",
                          "proposal_contents"):
            self.assertNotIn(forbidden, proposals[0])

    def test_prompts_only_use_v33_selection_contract_and_full_frozen_views(self):
        initial = build_planner_v3_prompt(
            root_id=self.root["root_id"], role_table=self.roles,
            action_menu=self.menu, context_summary={"summary": "safe"},
            planner_action_view=self.view, allowed_bundles=self.bundles)
        retry = build_planner_v3_retry_prompt(
            root_id=self.root["root_id"], frozen_action_menu=self.menu,
            frozen_role_table=self.roles, frozen_context={"old": "omitted"},
            frozen_scene_budget={"minimum": 2, "target": 3, "maximum": 4},
            errors=["split_bundle_across_scenes:bundle=B0001"],
            relevant_action_ids=["A0007"],
            frozen_planner_action_view=self.view,
            frozen_allowed_bundles=self.bundles)
        for prompt in (initial, retry):
            self.assertEqual(0, prompt.count("extension_proposals"))
            self.assertEqual(0, prompt.count("proposal_bundle_id"))
            self.assertIn("proposal_selection", prompt)
            self.assertIn("bundle_only", prompt)
            self.assertIn("indivisible", prompt)
        self.assertIn(json.dumps(self.view["actions"], ensure_ascii=False,
                                 separators=(",", ":")), retry)
        self.assertIn("required_action_views", retry)

    def test_historical_v9_attempts_have_required_root_causes(self):
        audits = []
        for index in (1, 2, 3):
            record = json.loads((V9 / f"attempts/planner-v3-a{index:02d}.json").read_text())
            audit = classify_legacy_bundle_plan(
                record["parsed_response"], allowed_bundles=self.bundles,
                planner_action_view=self.view)
            self.assertFalse(audit["valid_live_v3_3"])
            self.assertFalse(audit["migration_performed"])
            audits.append(audit)
        self.assertTrue(any(error.startswith("invalid_unregistered_action_pair")
                            for error in audits[0]["errors"]))
        for audit in audits[1:]:
            for prefix in ("single_action_illegally_carries_bundle",
                           "incomplete_bundle_actions",
                           "split_bundle_across_scenes",
                           "bundle_only_action_selected_singly"):
                self.assertTrue(any(error.startswith(prefix)
                                    for error in audit["errors"]), audit)

    def test_live_validator_rejects_historical_schema_without_migration(self):
        record = json.loads((V9 / "attempts/planner-v3-a01.json").read_text())
        result = validate_planner_v3(
            record["parsed_response"], action_menu=self.menu,
            authoritative_entity_refs=set(self.refs),
            planner_action_view=self.view, allowed_bundles=self.bundles)
        self.assertFalse(result["valid"])
        self.assertTrue(any(error.endswith(":schema") for error in result["errors"]))

    def test_protected_contract_modules_and_recursive_import_graph_are_unchanged(self):
        for relative, expected in PROTECTED_SHAS.items():
            self.assertEqual(expected, sha256_file(MODULE / relative), relative)
        audit = audit_v7_recursive_import_graph(
            project_root=PROJECT, runner_path=MODULE / "live_fresh_e2e_v7.py")
        self.assertTrue(audit["valid"], audit)
        self.assertEqual(0, audit["forbidden_import_count"])


if __name__ == "__main__":
    unittest.main()
