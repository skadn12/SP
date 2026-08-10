from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.binder import bind_scene
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.memory import (
    commit_confirmed_delta,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.prompts import (
    EVIDENCE_COMMITTER_PROMPT_VERSION,
    EVIDENCE_COMMITTER_RETRY_PROMPT_VERSION,
    build_emergent_evidence_committer_prompt,
    build_evidence_committer_prompt,
    build_evidence_committer_retry_prompt,
    compact_evidence_committer_retry_errors,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import (
    hash_tree,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.schemas import (
    EVIDENCE_COMMITTER_SCHEMA_VERSION,
    validate_evidence_committer_output,
)


HISTORICAL_V4_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_world_memory_slim_fresh_e2e_v4")


class EvidenceCommitterRetryV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.memory = json.loads(
            (HISTORICAL_V4_RUN / "world_memory_initial.json").read_text(
                encoding="utf-8"))
        cls.plan = json.loads(
            (HISTORICAL_V4_RUN / "planner_output.json").read_text(encoding="utf-8"))
        cls.compiled = json.loads(
            (HISTORICAL_V4_RUN / "compiled_deltas.json").read_text(encoding="utf-8"))
        cls.realizer = json.loads(
            (HISTORICAL_V4_RUN / "scenes/scene_01/realizer_output.json").read_text(
                encoding="utf-8"))
        cls.delta = cls.compiled["scenes"][0]["proposed_graph_deltas"][0]
        cls.scene = {
            "scene_key": "scene_01",
            "goal": cls.plan["scene_goals"][0]["goal"],
            "participants": cls.plan["scene_goals"][0]["participants"],
            "related_memory_ids": cls.delta["related_memory_ids"],
        }
        cls.historical_outputs = [json.loads(
            (HISTORICAL_V4_RUN /
             f"attempts/committer-scene_01-D0001-a{index:02d}.json").read_text(
                 encoding="utf-8"))["parsed_response"] for index in (1, 2, 3)]

    def _planned_prompt(self) -> str:
        return build_evidence_committer_prompt(
            proposed_delta=self.delta,
            dialogue_turns=self.realizer["dialogue_turns"],
            realizer_claims=self.realizer["claimed_graph_updates"],
            memory_view=self.memory)

    def test_planned_prompt_ends_with_exact_seven_field_template(self) -> None:
        prompt = self._planned_prompt()
        self.assertIn(EVIDENCE_COMMITTER_PROMPT_VERSION, prompt)
        self.assertNotIn('"decisions":', prompt)
        template = json.loads(prompt.splitlines()[-1])
        self.assertEqual(set(template), {
            "evidence_committer_schema_version", "proposed_delta_id", "decision",
            "confirmed_post_state", "evidence_turn_keys", "reason",
            "semantic_status",
        })
        self.assertIn("otherwise empty string", template["confirmed_post_state"])

    def test_emergent_prompt_uses_the_same_exact_contract(self) -> None:
        proposal = {
            "proposal_id": f"{self.memory['root_id']}:emergent:scene_01:E0001",
            "root_id": self.memory["root_id"], "source_scene": "scene_01",
            "operation": "open_thread", "description": "Discuss next week.",
            "candidate_evidence_turn_keys": [
                self.realizer["dialogue_turns"][0]["turn_key"]],
            "semantic_status": "model_claimed", "formal_thread_id": None,
            "authoritative": False,
        }
        prompt = build_emergent_evidence_committer_prompt(
            pending_proposal=proposal,
            dialogue_turns=self.realizer["dialogue_turns"],
            candidate_evidence_turns=[self.realizer["dialogue_turns"][0]],
            memory_view=self.memory)
        self.assertNotIn('"decisions":', prompt)
        self.assertEqual(len(json.loads(prompt.splitlines()[-1])), 7)

    def test_v4_outputs_receive_specific_extra_field_and_value_diagnostics(self) -> None:
        for output in self.historical_outputs:
            result = validate_evidence_committer_output(output)
            self.assertFalse(result["valid"])
            self.assertIn("evidence_committer_unexpected_field:decisions",
                          result["errors"])
            self.assertIn("evidence_committer_values", result["errors"])

    def test_v4_outputs_are_not_silently_repaired_but_copy_can_shadow_validate(self) -> None:
        for output in self.historical_outputs:
            self.assertFalse(validate_evidence_committer_output(output)["valid"])
            shadow = copy.deepcopy(output)
            shadow.pop("decisions")
            shadow["confirmed_post_state"] = ""
            result = validate_evidence_committer_output(shadow)
            self.assertTrue(result["valid"], result["errors"])
            self.assertEqual(shadow["decision"], "rejected")
            self.assertEqual(shadow["semantic_status"], "rejected")

    def test_retry_diagnostics_are_compact_and_actionable(self) -> None:
        errors = validate_evidence_committer_output(
            self.historical_outputs[0])["errors"]
        compact = compact_evidence_committer_retry_errors(errors)
        self.assertEqual(compact, [
            "remove_all_unexpected_fields_including_decisions",
            "confirmed_post_state_must_be_a_string_and_all_references_must_be_valid",
        ])

    def test_retry_prompt_omits_rejected_response_and_keeps_template_last(self) -> None:
        base = self._planned_prompt()
        errors = validate_evidence_committer_output(
            self.historical_outputs[0])["errors"]
        retry = build_evidence_committer_retry_prompt(
            frozen_base_prompt=base, errors=errors)
        self.assertIn(EVIDENCE_COMMITTER_RETRY_PROMPT_VERSION, retry)
        self.assertIn("previous rejected response is intentionally omitted", retry)
        self.assertNotIn(self.historical_outputs[0]["reason"], retry)
        self.assertNotIn('"decisions":', retry)
        self.assertEqual(len(json.loads(retry.splitlines()[-1])), 7)

    def test_valid_rejected_decision_reaches_pending_not_authoritative_memory(self) -> None:
        rejected = copy.deepcopy(self.historical_outputs[0])
        rejected.pop("decisions")
        rejected["confirmed_post_state"] = ""
        binding = bind_scene(
            root_id=self.memory["root_id"], round_index=1,
            scene_goal=self.scene, realizer_output=self.realizer,
            deltas=[self.delta], committer_outputs=[rejected])
        result = commit_confirmed_delta(
            self.memory, self.delta, rejected, binding,
            scene_key="scene_01", round_index=1, dry_run=True)
        self.assertEqual(result["memory_version"], self.memory["memory_version"])
        self.assertEqual(result["current_states"], self.memory["current_states"])
        self.assertEqual(result["pending_memory_updates"][-1]["semantic_status"],
                         "rejected")
        self.assertFalse(result["memory_update_log"][-1]["authoritative_applied"])

    def test_confirmed_post_state_null_is_never_accepted(self) -> None:
        output = {
            "evidence_committer_schema_version": EVIDENCE_COMMITTER_SCHEMA_VERSION,
            "proposed_delta_id": self.delta["proposed_delta_id"],
            "decision": "rejected", "confirmed_post_state": None,
            "evidence_turn_keys": [], "reason": "Independent rejection.",
            "semantic_status": "rejected",
        }
        self.assertIn("evidence_committer_values",
                      validate_evidence_committer_output(output)["errors"])

    def test_shadow_replay_preserves_historical_v4_run(self) -> None:
        before = hash_tree(HISTORICAL_V4_RUN)["tree_sha256"]
        _ = self._planned_prompt()
        self.assertEqual(before, hash_tree(HISTORICAL_V4_RUN)["tree_sha256"])
        self.assertEqual(before,
                         "d425b935f31bcbf95b15cb9ed1d62c82906f1dfdde56e8769552a0c6ef037d0e")


if __name__ == "__main__":
    unittest.main()
