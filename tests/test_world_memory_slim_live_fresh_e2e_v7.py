from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from scripts.current_method.restart_n5_spell.core import sha256_file
from scripts.current_method.restart_n5_spell.world_memory_slim_v1 import live_fresh_e2e_v7 as runner
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.append_only_memory import (
    apply_scene_commit,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.controlled_expansion import (
    compile_action_menu_v3,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.evidence_scope import (
    build_committer_scope_prompt,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.planner_v3 import (
    PLANNER_SCHEMA_VERSION_V3,
    build_planner_v3_retry_prompt,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.round_plan_validation import (
    compile_validate_round_plan,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.scene_local_refs import (
    dialogue_sha256,
    normalize_full_realizer_output,
    reconstruct_claim_only_output,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.semantic_import_audit_v7 import (
    audit_v7_recursive_import_graph,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.source_memory_initializer_v1 import (
    authoritative_affordance_view,
    initialize_source_world_memory,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.transactional_binder import (
    build_allocation_plan,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.v7_manifest import (
    directory_manifest,
    implementation_manifest,
    project_working_tree_manifest_sha256,
)


ROOT_ID = "slim-v7-test-root"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
V6_RUNNER = (PROJECT_ROOT / "scripts/current_method/restart_n5_spell/"
             "world_memory_slim_v1/live_fresh_e2e.py")


def source_fixture() -> tuple[dict, dict]:
    turns = [
        {"turn_id": f"{ROOT_ID}:P01:T001", "session_id": f"{ROOT_ID}:parent_01",
         "speaker": "A", "text": "I have the timetable in front of me."},
        {"turn_id": f"{ROOT_ID}:P01:T002", "session_id": f"{ROOT_ID}:parent_01",
         "speaker": "B", "text": "Then we can compare the two practical options."},
        {"turn_id": f"{ROOT_ID}:P01:T003", "session_id": f"{ROOT_ID}:parent_01",
         "speaker": "A", "text": "The earlier option leaves less preparation time."},
        {"turn_id": f"{ROOT_ID}:P01:T004", "session_id": f"{ROOT_ID}:parent_01",
         "speaker": "B", "text": "Let us keep the decision open for now."},
    ]
    material = json.dumps(turns, ensure_ascii=False, sort_keys=True).encode("utf-8")
    dialogue_hash = hashlib.sha256(material).hexdigest()
    root = {"root_id": ROOT_ID, "sample_id": ROOT_ID,
            "dialogue": {"dialogue_hash": dialogue_hash, "turns": deepcopy(turns)}}
    parent = {"sample_id": ROOT_ID, "lineage_group": ROOT_ID,
              "dialogue": {"dialogue_hash": dialogue_hash, "turns": deepcopy(turns)}}
    return root, parent


def plan_fixture() -> tuple[dict, dict, list[str]]:
    root, parent = source_fixture()
    source = initialize_source_world_memory(root, parent)
    memory = source["memory"]
    menu = compile_action_menu_v3(
        authoritative_affordance_view(memory), root_id=ROOT_ID, interaction_links=[])
    action = next(row for row in menu["actions"] if row["operation"] == "add_event")
    refs = [row["entity_ref"] for row in memory["entities"]]
    speakers = [
        {"participant_ref": refs[0], "scene_role": "primary",
         "intents": ["raise one concrete scheduling issue", "seek a usable decision"]},
        {"participant_ref": refs[1], "scene_role": "supporting",
         "intents": ["test the option against a practical constraint"]},
    ]
    plan = {
        "planner_schema_version": PLANNER_SCHEMA_VERSION_V3,
        "scene_budget": {"minimum": 2, "target": 3, "maximum": 4},
        "round_arc": {
            "starting_situation": "Two colleagues still need a practical next step.",
            "central_tension": "Their preferred timing differs.",
            "ending_direction": "They leave with a concrete shared development.",
        },
        "scene_goals": [
            {"goal": "Work through a concrete scheduling development.",
             "speaker_participants": deepcopy(speakers),
             "interaction_beats": ["A raises the timing issue.",
                                   "B tests it against preparation needs.",
                                   "They identify a concrete shared development."],
             "continuity_requirements": ["Do not alter the source exchange."],
             "proposal_selection": {
                 "selection_type": "single_action",
                 "action_id": action["action_id"],
                 "proposed_content": {
                     "description": "They arrange a short planning meeting."},
             },
             "desired_outcome": "A reviewable new event is stated naturally."},
            {"goal": "Clarify how they will communicate afterward.",
             "speaker_participants": deepcopy(speakers),
             "interaction_beats": ["A asks for a concise follow-up.",
                                   "B distinguishes urgent from optional details.",
                                   "They close without asserting another persistent fact."],
             "continuity_requirements": ["Do not introduce another memory proposal."],
             "proposal_selection": {"selection_type": "none"},
             "desired_outcome": "The exchange closes naturally without another proposal."},
        ],
    }
    return plan, menu, refs


def raw_dialogue(scene_number: int, speakers: list[str], *, bad_evidence: bool = False) -> dict:
    turns = [{"turn_key": f"t{index:03d}",
              "speaker_ref": speakers[(index - 1) % 2],
              "text": f"Scene {scene_number} original extension utterance {index} develops a practical option."}
             for index in range(1, 8)]
    claims = [] if scene_number == 2 else [{
        "proposal_ref": "R01:S01:P01", "realization_status": "realized",
        "candidate_evidence_turn_keys": ["t999" if bad_evidence else "t007"],
    }]
    return {"dialogue_turns": turns, "proposal_claims": claims}


class ScriptedCaller:
    def __init__(self, responses: dict[str, list[dict]]):
        self.responses = {key: [deepcopy(row) for row in value]
                          for key, value in responses.items()}
        self.records: list[dict] = []
        self.prompts: list[dict] = []

    def call_json(self, *, request_id: str, role: str, attempt: int,
                  prompt: str, prompt_version: str, max_tokens: int):
        self.prompts.append({"request_id": request_id, "role": role, "attempt": attempt,
                             "prompt": prompt, "prompt_version": prompt_version})
        if not self.responses.get(role):
            raise AssertionError(f"unexpected role request: {role}:{request_id}")
        value = self.responses[role].pop(0)
        logical = {"planner_v3": "planner", "realizer_v3": "realizer",
                   "evidence_committer_scope_v1": "evidence_committer"}[role]
        self.records.append({"logical_role": logical, "attempt": attempt,
                             "prompt_tokens_local": len(prompt) // 4,
                             "completion_tokens_local": 10,
                             "total_tokens_local": len(prompt) // 4 + 10})
        return deepcopy(value), json.dumps(value), []


def committed_review() -> dict:
    return {"proposal_reviews": [{
        "proposal_ref": "R01:S01:P01", "review_decision": "confirmed",
        "verification_status": "semantically_verified",
        "accepted_evidence_turn_refs": ["R01:S01:T007"],
        "reason_code": "supported",
    }], "scope_status": "clean", "world_consistency_status": "clean"}


def scope_only_review() -> dict:
    return {"proposal_reviews": [], "scope_status": "clean",
            "world_consistency_status": "clean"}


class LiveV7RunnerTests(unittest.TestCase):
    def _run(self, responses: dict[str, list[dict]]):
        root, parent = source_fixture()
        temp = tempfile.TemporaryDirectory(prefix="slim-v7-runner-test-")
        root_path = Path(temp.name)
        run_root = root_path / "run"
        run_root.mkdir()
        input_path = root_path / "input.json"
        input_path.write_text(json.dumps(root), encoding="utf-8")
        v6_root = root_path / "v6"
        v6_root.mkdir()
        (v6_root / "immutable.txt").write_text("historical", encoding="utf-8")
        freeze = {
            "git_head": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                text=True, capture_output=True, check=True).stdout.strip(),
            "implementation_manifest_sha256": implementation_manifest(PROJECT_ROOT)[
                "implementation_manifest_sha256"],
            "project_working_tree_manifest_sha256": project_working_tree_manifest_sha256(PROJECT_ROOT),
            "input_sha256": sha256_file(input_path),
            "parent_dialogue_sha256": runner._semantic_sha(parent["dialogue"]),
            "v6_directory_manifest_sha256": directory_manifest(v6_root)[
                "directory_manifest_sha256"],
        }
        caller = ScriptedCaller(responses)
        summary = runner._execute_v7(
            root=root, parent=parent, run_root=run_root, caller=caller,
            limits=runner.V7RunLimits(), freeze_manifest=freeze,
            input_path=input_path, historical_v6_root=v6_root,
            model_config={"mode": "scripted-no-network"})
        return temp, run_root, caller, summary

    def test_end_to_end_offline_harness_commits_and_calls_scope_only_committer(self):
        plan, _, speakers = plan_fixture()
        responses = {
            "planner_v3": [plan],
            "realizer_v3": [raw_dialogue(1, speakers), raw_dialogue(2, speakers)],
            "evidence_committer_scope_v1": [committed_review(), scope_only_review()],
        }
        temp, run_root, caller, summary = self._run(responses)
        self.addCleanup(temp.cleanup)
        self.assertTrue(summary["generation_ready"], summary)
        self.assertTrue(summary["fresh_e2e_ready"], summary)
        self.assertEqual(2, summary["api_usage"]["roles"]["evidence_committer"]["physical_requests"])
        self.assertIn("MODE\nscope-only", [row["prompt"] for row in caller.prompts
                                             if row["role"] == "evidence_committer_scope_v1"][1])
        self.assertEqual(["R01:S01:T001", "R01:S01:T002"], [
            row["turn_ref"] for row in json.loads(
                (run_root / "child_dialogue_full.json").read_text())["scenes"][0]["dialogue_turns"][:2]])
        self.assertEqual(1, summary["memory"]["generated_extension_count"])
        journal = (run_root / "run_commit_journal.jsonl").read_text()
        self.assertIn('"status":"PREPARED"', journal)
        self.assertIn('"status":"COMMITTED"', journal)

    def test_planner_compiler_payload_error_routes_to_planner_retry(self):
        plan, _, speakers = plan_fixture()
        invalid = deepcopy(plan)
        invalid["scene_goals"][0]["proposal_selection"]["proposed_content"] = {}
        responses = {
            "planner_v3": [invalid, plan],
            "realizer_v3": [raw_dialogue(1, speakers), raw_dialogue(2, speakers)],
            "evidence_committer_scope_v1": [committed_review(), scope_only_review()],
        }
        temp, _, caller, summary = self._run(responses)
        self.addCleanup(temp.cleanup)
        self.assertEqual(2, len(summary["planner"]["attempts"]))
        self.assertEqual("planner_repairable",
                         summary["planner"]["attempts"][0]["validation"]["classification"])
        self.assertEqual(2, sum(row["role"] == "planner_v3" for row in caller.prompts))

    def test_claim_only_retry_freezes_dialogue_and_only_exposes_local_refs(self):
        plan, _, speakers = plan_fixture()
        repaired = {"proposal_claims": [{
            "proposal_ref": "R01:S01:P01", "realization_status": "realized",
            "candidate_evidence_turn_refs": ["R01:S01:T007"],
        }]}
        responses = {
            "planner_v3": [plan],
            "realizer_v3": [raw_dialogue(1, speakers, bad_evidence=True), repaired,
                             raw_dialogue(2, speakers)],
            "evidence_committer_scope_v1": [committed_review(), scope_only_review()],
        }
        temp, _, caller, summary = self._run(responses)
        self.addCleanup(temp.cleanup)
        scene = summary["scenes"][0]
        self.assertEqual(1, scene["claim_only_retry_count"])
        self.assertTrue(scene["realizer_attempts"][1]["frozen_dialogue_check"][
            "dialogue_sha_unchanged"])
        self.assertEqual(scene["realizer_attempts"][0]["dialogue_sha256"],
                         scene["realizer_attempts"][1]["dialogue_sha256"])
        retry_prompt = next(row["prompt"] for row in caller.prompts
                            if row["prompt_version"].endswith("claim_retry_prompt_v1"))
        self.assertIn("R01:S01:T007", retry_prompt)
        self.assertNotIn("t999", retry_prompt)
        self.assertNotIn('"dialogue_turns"', retry_prompt.split("OUTPUT ONLY THIS OBJECT")[-1])

    def test_committer_contract_retry_keeps_semantic_input_sha(self):
        plan, _, speakers = plan_fixture()
        responses = {
            "planner_v3": [plan],
            "realizer_v3": [raw_dialogue(1, speakers), raw_dialogue(2, speakers)],
            "evidence_committer_scope_v1": [
                {"wrong": "shape"}, committed_review(), scope_only_review()],
        }
        temp, _, _, summary = self._run(responses)
        self.addCleanup(temp.cleanup)
        attempts = summary["scenes"][0]["committer_attempts"]
        self.assertEqual(2, len(attempts))
        self.assertEqual(1, len({row["semantic_input_sha256"] for row in attempts}))

    def test_local_invariant_stops_before_following_scene(self):
        plan, _, _ = plan_fixture()
        root, parent = source_fixture()
        with tempfile.TemporaryDirectory(prefix="slim-v7-invariant-test-") as temp_name:
            base = Path(temp_name)
            run_root = base / "run"; run_root.mkdir()
            input_path = base / "input.json"; input_path.write_text(json.dumps(root))
            v6_root = base / "v6"; v6_root.mkdir(); (v6_root / "x").write_text("x")
            freeze = {"git_head": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
                capture_output=True, check=True).stdout.strip(),
                "implementation_manifest_sha256": implementation_manifest(PROJECT_ROOT)["implementation_manifest_sha256"],
                "project_working_tree_manifest_sha256": project_working_tree_manifest_sha256(PROJECT_ROOT),
                "input_sha256": sha256_file(input_path),
                "parent_dialogue_sha256": runner._semantic_sha(parent["dialogue"]),
                "v6_directory_manifest_sha256": directory_manifest(v6_root)["directory_manifest_sha256"]}
            caller = ScriptedCaller({"planner_v3": [plan]})
            with patch.object(runner, "_execute_scene",
                              side_effect=runner.LocalInvariantFailure("broken_registry")) as execute:
                summary = runner._execute_v7(
                    root=root, parent=parent, run_root=run_root, caller=caller,
                    limits=runner.V7RunLimits(), freeze_manifest=freeze,
                    input_path=input_path, historical_v6_root=v6_root,
                    model_config={"mode": "scripted-no-network"})
            self.assertEqual(1, execute.call_count)
            self.assertEqual("local_invariant_failure", summary["failure_stage"])
            self.assertEqual(1, len(caller.records))


class LiveV7ContractTests(unittest.TestCase):
    def test_recursive_graph_has_no_v6_semantic_modules_and_v6_runner_is_unchanged(self):
        audit = audit_v7_recursive_import_graph(
            project_root=PROJECT_ROOT, runner_path=Path(runner.__file__))
        self.assertTrue(audit["valid"], audit)
        self.assertEqual(0, audit["forbidden_import_count"])
        self.assertEqual("69763e2493f88eee8e74fb4a6492013bccc52b5109c5ce5ba888839bc406ea03",
                         sha256_file(V6_RUNNER))

    def test_scene_compiler_payload_failure_is_planner_repairable(self):
        plan, menu, refs = plan_fixture()
        source = initialize_source_world_memory(*source_fixture())
        invalid = deepcopy(plan)
        invalid["scene_goals"][0]["proposal_selection"]["proposed_content"] = {}
        result = compile_validate_round_plan(
            invalid, action_menu=menu, authoritative_memory=source["memory"],
            interaction_links=[{"entity_refs": refs}], round_index=1)
        self.assertFalse(result["valid"])
        self.assertEqual("planner_repairable", result["classification"])

    def test_full_retry_regenerates_refs_but_claim_retry_freezes_them(self):
        _, _, speakers = plan_fixture()
        first = normalize_full_realizer_output(raw_dialogue(1, speakers), scene_ref="R01:S01")
        second_raw = raw_dialogue(1, speakers)
        second_raw["dialogue_turns"][0]["text"] = "A genuinely regenerated first utterance."
        second = normalize_full_realizer_output(second_raw, scene_ref="R01:S01")
        self.assertEqual([row["turn_ref"] for row in first["normalized_output"]["dialogue_turns"]],
                         [row["turn_ref"] for row in second["normalized_output"]["dialogue_turns"]])
        self.assertNotEqual(first["dialogue_sha256"], second["dialogue_sha256"])
        frozen = first["normalized_output"]["dialogue_turns"]
        rebuilt = reconstruct_claim_only_output(
            frozen_dialogue_turns=frozen,
            claim_only_output={"proposal_claims": first["normalized_output"]["proposal_claims"]})
        self.assertTrue(rebuilt["dialogue_sha_unchanged"])
        self.assertEqual(dialogue_sha256(frozen), rebuilt["dialogue_sha256_after"])

    def test_valid_realized_claim_on_final_attempt_is_not_discarded(self):
        from scripts.current_method.restart_n5_spell.world_memory_slim_v1.claim_validator_v2 import (
            realizer_retry_route,
        )
        route = realizer_retry_route(
            json_parsed=True, dialogue_quality={"valid": True},
            claim_validation={"valid": True},
            claims=[{"realization_status": "realized"}],
            requests_used=3, maximum_requests=3)
        self.assertEqual("submit_to_committer", route)

    def test_planner_retry_uses_frozen_hashes_without_repeating_parent_context(self):
        _, menu, refs = plan_fixture()
        prompt = build_planner_v3_retry_prompt(
            root_id=ROOT_ID, frozen_action_menu=menu,
            frozen_role_table=[{"entity_ref": refs[0]}, {"entity_ref": refs[1]}],
            frozen_context={"parent_turns_untrusted": [{"text": "DO NOT REPEAT ME"}]},
            frozen_scene_budget={"minimum": 2, "target": 3, "maximum": 4},
            errors=["proposal_payload_missing:description"],
            relevant_action_ids=[menu["actions"][0]["action_id"]],
            frozen_authoritative_memory_sha256="m" * 64,
            frozen_interaction_links_sha256="i" * 64,
            frozen_round_context_sha256="c" * 64)
        self.assertNotIn("DO NOT REPEAT ME", prompt)
        self.assertIn(f'"entity_ref":"{refs[0]}"', prompt)
        self.assertIn(f'"entity_ref":"{refs[1]}"', prompt)
        self.assertIn("c" * 64, prompt)
        output_example = json.loads(prompt.rsplit("\n", 1)[-1])
        self.assertEqual("scene_goals", list(output_example)[-1])
        self.assertEqual(2, len(output_example["scene_goals"]))

    def test_committer_prompt_has_distinct_proposal_and_scope_only_shapes(self):
        plan, menu, refs = plan_fixture()
        source = initialize_source_world_memory(*source_fixture())
        compiled = compile_validate_round_plan(
            plan, action_menu=menu, authoritative_memory=source["memory"],
            interaction_links=[{"entity_refs": refs}], round_index=1)[
                "compiled_round_plan"]["scenes"][0]
        proposal = compiled["compiled_proposals"][0]
        claim = [{"proposal_ref": proposal["proposal_ref"],
                  "realization_status": "realized",
                  "candidate_evidence_turn_refs": ["R01:S01:T001"]}]
        dialogue = [{"turn_ref": "R01:S01:T001", "speaker_ref": refs[0],
                     "text": "A new supported development is stated."}]
        proposal_prompt = build_committer_scope_prompt(
            compiled_proposals=[proposal], dialogue_turns=dialogue,
            proposal_claims=claim, authoritative_world={"root_id": ROOT_ID})
        proposal_shape = json.loads(proposal_prompt.split("OUTPUT EXACTLY\n")[-1])
        self.assertEqual([proposal["proposal_ref"]], [
            row["proposal_ref"] for row in proposal_shape["proposal_reviews"]])
        scope_prompt = build_committer_scope_prompt(
            compiled_proposals=[], dialogue_turns=dialogue,
            proposal_claims=[], authoritative_world={"root_id": ROOT_ID})
        self.assertEqual([], json.loads(scope_prompt.split("OUTPUT EXACTLY\n")[-1])[
            "proposal_reviews"])

    def test_temporary_person_is_mentioned_or_reported_then_becomes_authoritative_only_on_commit(self):
        plan, menu, refs = plan_fixture()
        source = initialize_source_world_memory(*source_fixture())
        person = next(row for row in menu["actions"] if row["operation"] == "add_person")
        relation = next(row for row in menu["actions"] if row["operation"] == "add_relation")
        plan["scene_goals"][0]["proposal_selection"] = {
            "selection_type": "bundle",
            "bundle_id": "B0001",
            "proposal_contents": {
                person["action_id"]: {"display_role": "a scheduling coordinator"},
                relation["action_id"]: {"relation_type": "collaborates_with"},
            },
        }
        compiled_result = compile_validate_round_plan(
            plan, action_menu=menu, authoritative_memory=source["memory"],
            interaction_links=[{"entity_refs": refs}], round_index=1)
        self.assertTrue(compiled_result["valid"], compiled_result)
        scene = compiled_result["compiled_round_plan"]["scenes"][0]
        proposal = scene["compiled_proposals"][0]
        allocation = build_allocation_plan(
            root_id=ROOT_ID, round_index=1, scene_index=1,
            compiled_proposals=[proposal],
            high_water_marks=source["memory"]["high_water_marks"])
        claim = [{"proposal_ref": proposal["proposal_ref"],
                  "realization_status": "realized",
                  "candidate_evidence_turn_refs": ["R01:S01:T001"]}]
        review = {"proposal_reviews": [{
            "proposal_ref": proposal["proposal_ref"],
            "review_decision": "confirmed", "verification_status": "semantically_verified",
            "accepted_evidence_turn_refs": ["R01:S01:T001"], "reason_code": "supported"}],
            "scope_status": "clean", "world_consistency_status": "clean"}
        committed = apply_scene_commit(
            source["memory"], scene_ref="R01:S01", round_index=1, scene_index=1,
            compiled_proposals=[proposal], proposal_claims=claim,
            committer_output=review, allocation_plan=allocation,
            evidence_final_turn_refs={proposal["proposal_ref"]: ["R01:S01:T001"]},
            evidence_text_sha256={proposal["proposal_ref"]: ["a" * 64]})
        self.assertEqual(len(source["memory"]["entities"]) + 1,
                         len(committed["entities"]))
        self.assertEqual("semantically_verified", committed["entities"][-1][
            "verification_status"])
        invalid = deepcopy(plan)
        invalid["scene_goals"][0]["proposal_selection"]["proposal_contents"][
            person["action_id"]]["scene_presence"] = "speaker"
        rejected = compile_validate_round_plan(
            invalid, action_menu=menu, authoritative_memory=source["memory"],
            interaction_links=[{"entity_refs": refs}], round_index=1)
        self.assertFalse(rejected["valid"])
        self.assertEqual("planner_repairable", rejected["classification"])


if __name__ == "__main__":
    unittest.main()
