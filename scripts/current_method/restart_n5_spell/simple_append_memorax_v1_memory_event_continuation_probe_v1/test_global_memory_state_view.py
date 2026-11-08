from __future__ import annotations

import unittest

from .contracts import (build_semantic_gate_context,
                         validate_semantic_gate_output)
from .memory_context import (
    BUCKET_SNAPSHOT_VERSION,
    compile_global_memory_state_view,
)
from .prompts import build_semantic_continuation_gate_prompt


def _snapshot():
    return {
        "bucket_snapshot_version": BUCKET_SNAPSHOT_VERSION,
        "root_id": "root-1",
        "snapshot_sha256": "snapshot-sha",
        "buckets": {
            "person": {
                "a": {"canonical_key": "a", "display_key": "A",
                      "content": ["A completed the school call."]},
            },
            "location": {},
            "event": {
                "school call": {
                    "canonical_key": "school call",
                    "display_key": "School call",
                    "content": ["The school confirmed volunteers.",
                                "The schedule remains pending."],
                },
            },
            "item": {
                "shared doc": {
                    "canonical_key": "shared doc",
                    "display_key": "Shared doc",
                    "content": [{"attribute": "owner", "value": "A"}],
                },
            },
            "dreaming": {},
        },
    }


class GlobalMemoryStateViewTest(unittest.TestCase):
    def test_view_keeps_all_bucket_state_without_transcript(self):
        view = compile_global_memory_state_view(_snapshot())
        self.assertEqual(view["root_id"], "root-1")
        self.assertEqual(view["bucket_counts"]["event"], 1)
        self.assertEqual(view["buckets"]["event"][0]["memory_facts"], [
            {"memory_ref": "GM:E001:C001",
             "text": "The school confirmed volunteers."},
            {"memory_ref": "GM:E001:C002",
             "text": "The schedule remains pending."},
        ])
        self.assertEqual(view["buckets"]["person"][0]["memory_facts"], [
            {"memory_ref": "GM:P001:C001",
             "text": "A completed the school call."}])
        self.assertNotIn("dialogue", view)

    def test_fast_gate_context_contains_global_view_and_no_parent_text(self):
        parent = {"dialogue": {"turns": [{
            "turn_id": "root-1:P01:T001", "session_id": "root-1:p01",
            "speaker": "A", "text": "PRIVATE PARENT TEXT",
        }]}}
        candidates = [{"memory_event_id": "ME01",
                       "memory_event_key": "school call",
                       "memory_event_display": "School call",
                       "memory_event_content": ["hint"]}]
        plan = {"candidate_pool": [{
            "memory_event_id": "ME01", "development_mode": "continuation",
            "next_step_candidates": [{
                "before_state": "before", "required_transition": "new",
                "after_state": "after", "actor_bindings": [],
                "interaction_beats": [],
            }],
        }]}
        view = compile_global_memory_state_view(_snapshot())
        context = build_semantic_gate_context(
            parent=parent, candidates=candidates, plan=plan,
            memory_only_fast=True, global_memory_state_view=view)
        self.assertEqual(context["global_memory_state_view"]["root_id"],
                         "root-1")
        self.assertNotIn("PRIVATE PARENT TEXT", str(context))
        prompt = build_semantic_continuation_gate_prompt(gate_context=context)
        self.assertIn("GLOBAL MEMORY STATE VIEW", prompt["prompt"])
        self.assertIn("all explicit states in every bucket", prompt["prompt"])

    def test_memory_reject_requires_existing_memory_fact_ref(self):
        parent = {"dialogue": {"turns": []}}
        plan = {"candidate_pool": [{
            "memory_event_id": "ME01",
            "next_step_candidates": [{
                "before_state": "pending",
                "required_transition": "complete the call",
                "after_state": "call complete",
            }],
        }]}
        view = compile_global_memory_state_view(_snapshot())
        good = {"candidate_reviews": [{
            "candidate_ref": "ME01:C01",
            "parent_relation": "REALIZED",
            "decision": "reject",
            "duplicate_with_parent": True,
            "supporting_memory_refs": ["GM:E001:C001"],
            "reason": "The cited fact explicitly says the call is complete.",
        }]}
        result = validate_semantic_gate_output(
            good, plan=plan, parent=parent, memory_only_fast=True,
            global_memory_state_view=view)
        self.assertTrue(result["valid"], result["errors"])
        bad = {"candidate_reviews": [{
            **good["candidate_reviews"][0],
            "supporting_memory_refs": [],
        }]}
        result = validate_semantic_gate_output(
            bad, plan=plan, parent=parent, memory_only_fast=True,
            global_memory_state_view=view)
        self.assertFalse(result["valid"])
        self.assertIn("gate_review_1_reject_evidence_required",
                      result["errors"])

    def test_memory_supporting_ref_must_exist(self):
        parent = {"dialogue": {"turns": []}}
        plan = {"candidate_pool": [{
            "memory_event_id": "ME01",
            "next_step_candidates": [{"required_transition": "new"}],
        }]}
        view = compile_global_memory_state_view(_snapshot())
        value = {"candidate_reviews": [{
            "candidate_ref": "ME01:C01",
            "parent_relation": "REALIZED",
            "decision": "reject",
            "duplicate_with_parent": True,
            "supporting_memory_refs": ["GM:E999:C001"],
            "reason": "unsupported",
        }]}
        result = validate_semantic_gate_output(
            value, plan=plan, parent=parent, memory_only_fast=True,
            global_memory_state_view=view)
        self.assertFalse(result["valid"])
        self.assertIn("gate_review_1_evidence_not_memory", result["errors"])


if __name__ == "__main__":
    unittest.main()
