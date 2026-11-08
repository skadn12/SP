"""Final adherence example/validator checks; generated text is always synthetic."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest

from .retention import render_adherence, assess_retention
from .run import offline_network_guard
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.contracts import (
    normalize_realizer_output,
)


SOURCE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/") / (
    "h4_round1_r03_same_task_selection_resume_v1_live_1/execution"
)


def exercise_adherence_alignment():
    """Return exportable MOCK evidence without a backend, ledger, or file writes."""
    with offline_network_guard():
        source_paths = {
            "bundle": SOURCE / "input_freeze.json",
            "selected_queue": SOURCE / "normalized_execution_queue.json",
        }
        source_bytes = {name: path.read_bytes() for name, path in source_paths.items()}
        bundle = json.loads(source_bytes["bundle"])
        queue = json.loads(source_bytes["selected_queue"])
        parent = bundle["parent"]
        plan = {"scenes": [
            {**deepcopy(entry["scene"]), "scene_ref": f"R03:S{index:02d}"}
            for index, entry in enumerate(queue[:2], 1)
        ]}
        generated = {"scenes": [{"dialogue_turns": [
            {
                "speaker": scene["required_participants"][
                    index % len(scene["required_participants"])
                ],
                "text": (
                    f"OFFLINE MOCK {scene['scene_ref']} turn {index}: "
                    "This synthetic pending arrangement is not a real observation."
                ),
            }
            for index in range(14)
        ]} for scene in plan["scenes"]]}
        normalized = normalize_realizer_output(
            generated, plan=plan, parent=parent, round_index=3
        )
        if not normalized["valid"]:
            raise AssertionError(normalized)
        child = normalized["child"]
        for scene in child["scenes"]:
            scene["session_ref"] = scene["scene_ref"]

        conditions_by_phase = {}
        for entry in queue:
            for dependency in entry["scene"]["dependencies"]:
                values = conditions_by_phase.setdefault(dependency["phase_id"], [])
                if dependency["condition"] not in values:
                    values.append(dependency["condition"])
        conditions = {
            scene["scene_ref"]: conditions_by_phase.get(scene["phase_id"], [])
            for scene in plan["scenes"]
        }
        generated_raw = json.dumps(generated, ensure_ascii=False)
        rendered = render_adherence(
            parent=parent, plan=plan, child=child,
            requested_conditions=conditions, generated_response=generated_raw,
            prompt_version="adherence.raw_response_v2",
        )
        text = rendered["messages"][0]["content"]
        decode = json.JSONDecoder().raw_decode
        # Parse the examples actually sent by the renderer, not a separate mock schema.
        example = decode(text.split("Return exactly ", 1)[1])[0]
        state_example = decode(text.split("turn_refs, each ", 1)[1])[0]
        condition_example = decode(text.split("condition_results\ncontains ", 1)[1])[0]
        audit_rows = []
        for scene in child["scenes"]:
            first_turn = scene["dialogue_turns"][0]
            ref = first_turn["turn_ref"]
            row = deepcopy(example["scene_audits"][0])
            row.update(
                scene_ref=scene["scene_ref"], goal_status="partial",
                content_decision="retain", participant_adherence="pass",
                history_consistency="pass", novel_progress="pass",
                actual_states=[{
                    **deepcopy(state_example), "statement": first_turn["text"],
                    "stage": "arrangement", "turn_refs": [ref],
                }],
                goal_evidence_refs=[ref],
                condition_results=[{
                    **deepcopy(condition_example), "condition": condition,
                    "status": "established", "turn_refs": [ref],
                } for condition in conditions[scene["scene_ref"]]],
                reason="OFFLINE MOCK contract check, not a semantic observation.",
            )
            audit_rows.append(row)

        cases = {}
        first_history_turn = parent["dialogue"]["turns"][0]
        history_ref = first_history_turn.get("turn_ref") or first_history_turn["turn_id"]
        for name in (
            "partial", "not_realized", "history_ref_as_generated_evidence",
            "wrong_evidence_field", "unknown_goal",
        ):
            raw_audit = {"scene_audits": deepcopy(audit_rows)}
            for row in raw_audit["scene_audits"]:
                if name == "not_realized":
                    row.update(goal_status="not_realized", goal_evidence_refs=[])
                elif name == "history_ref_as_generated_evidence":
                    row["actual_states"][0]["turn_refs"] = [history_ref]
                elif name == "wrong_evidence_field":
                    row["supporting_memory_refs"] = row.pop("goal_evidence_refs")
                elif name == "unknown_goal":
                    row.update(goal_status="UNKNOWN", goal_evidence_refs=[])
            assessment = assess_retention(
                parent=parent, plan=plan, child=child, raw_audit=raw_audit,
                requested_conditions=conditions,
            )
            cases[name] = {"raw_audit_MOCK": raw_audit, "assessment_MOCK": assessment}

        return {
            "fixture_notice": (
                "Historical inputs are immutable actual artifacts; all generated "
                "dialogue and adherence judgments below are MOCK fixtures. "
                "No R03 observation or real semantic success is asserted."
            ),
            "source_identity": {
                name: {"path": str(source_paths[name]),
                       "sha256": hashlib.sha256(data).hexdigest()}
                for name, data in source_bytes.items()
            },
            "policy_id": bundle["policy_id"],
            "realizer_mode": bundle["realizer_mode"],
            "history_turn_count": len(parent["dialogue"]["turns"]),
            "selected_queue_count": len(queue),
            "source_candidate_ids": [scene["source_candidate_id"] for scene in plan["scenes"]],
            "prompt_version": rendered["prompt_version"],
            "final_messages_MOCK": rendered["messages"],
            "parsed_output_example": example,
            "parsed_state_example": state_example,
            "parsed_condition_example": condition_example,
            "plan": plan,
            "generated_response_MOCK": generated_raw,
            "normalized_child_MOCK": child,
            "requested_conditions": conditions,
            "cases": cases,
            "real_external_calls": 0,
            "ledger_accesses": 0,
            "file_writes": 0,
        }


class AdherenceContractAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = exercise_adherence_alignment()

    def decisions(self, name):
        return self.evidence["cases"][name]["assessment_MOCK"]["decisions"]

    def test_final_renderer_examples_match_current_fields(self):
        evidence = self.evidence
        self.assertEqual(evidence["prompt_version"], "incremental_retention_v1.adherence.raw_response_v2")
        self.assertEqual(set(evidence["parsed_output_example"]["scene_audits"][0]), {
            "scene_ref", "goal_status", "content_decision", "participant_adherence",
            "history_consistency", "novel_progress", "actual_states",
            "goal_evidence_refs", "condition_results", "reason",
        })
        self.assertEqual(set(evidence["parsed_state_example"]), {"statement", "stage", "turn_refs"})
        self.assertEqual(set(evidence["parsed_condition_example"]), {"condition", "status", "turn_refs"})
        self.assertEqual(evidence["history_turn_count"], 143)
        self.assertEqual(evidence["selected_queue_count"], 12)
        self.assertEqual(evidence["source_candidate_ids"], ["ME08:C01", "ME07:C01"])
        self.assertEqual(evidence["realizer_mode"], "text_free")
        self.assertEqual(len(evidence["final_messages_MOCK"]), 2)
        first = evidence["final_messages_MOCK"][0]["content"]
        data = json.loads(first.rsplit("\n\nINPUT DATA\n", 1)[1])
        self.assertEqual(len(data["authoritative_history"]), 143)
        self.assertEqual(data["planned_targets"], evidence["plan"])
        self.assertEqual(data["downstream_conditions"], evidence["requested_conditions"])
        self.assertEqual(evidence["final_messages_MOCK"][1]["content"], evidence["generated_response_MOCK"])
        rules = first.split("\n\nINPUT DATA\n", 1)[0]
        self.assertNotIn("supporting_memory_refs", rules)
        self.assertNotIn("evidence_turn_refs", rules)
        self.assertEqual(evidence["real_external_calls"], 0)
        self.assertEqual(evidence["ledger_accesses"], 0)
        self.assertEqual(evidence["file_writes"], 0)

    def test_partial_goal_with_progress_can_retain(self):
        for decision in self.decisions("partial"):
            self.assertEqual(decision["goal_status"], "partial")
            self.assertEqual(decision["errors"], [])
            self.assertTrue(decision["retained"])

    def test_not_realized_with_progress_and_empty_goal_refs_can_retain(self):
        for decision in self.decisions("not_realized"):
            self.assertEqual(decision["goal_status"], "not_realized")
            self.assertEqual(decision["errors"], [])
            self.assertTrue(decision["retained"])

    def test_history_refs_cannot_impersonate_generated_scene_evidence(self):
        for decision in self.decisions("history_ref_as_generated_evidence"):
            self.assertIn("actual_state_shape_or_refs", decision["errors"])
            self.assertFalse(decision["retained"])

    def test_wrong_evidence_field_remains_rejected(self):
        for decision in self.decisions("wrong_evidence_field"):
            self.assertIn("audit_dimensions_missing_or_misaligned", decision["errors"])
            self.assertIn("goal_evidence", decision["errors"])
            self.assertFalse(decision["retained"])

    def test_unknown_goal_is_not_promoted(self):
        for decision in self.decisions("unknown_goal"):
            self.assertEqual(decision["goal_status"], "UNKNOWN")
            self.assertIn("goal_information_insufficient", decision["errors"])
            self.assertFalse(decision["retained"])


if __name__ == "__main__":
    unittest.main()
