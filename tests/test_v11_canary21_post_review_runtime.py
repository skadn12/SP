"""Regression for Canary21's continuation/source variable collision."""

from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path

from scripts.current_method.restart_n5_spell.world_memory_slim_v1.expanded_dialogue_assembly import (
    assemble_expanded_dialogue,
    validate_expanded_dialogue,
)
from scripts.current_method.restart_n5_spell.world_memory_slim_v1.provenance import hash_tree


CANARY21 = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_61555_slim_continuous_expansion_"
    "v11_live_canary_21"
)
CANARY21_TREE_SHA = (
    "848f55c53e56adf8a3172cd982660b653a4d818ccc021035ca4bd4e5cc45f7e0"
)
PARENT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-inputs/"
    "conversation_chronicles_v1/roots/cc-train-episode-61555/unified_root.json"
)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class Canary21PostReviewRuntimeTests(unittest.TestCase):
    def test_continuation_loop_does_not_overwrite_source_initialization(self) -> None:
        source = Path(
            "scripts/current_method/restart_n5_spell/world_memory_slim_v1/"
            "live_canary_v11.py"
        ).read_text(encoding="utf-8")
        self.assertIsNone(re.search(
            r'^\s*source = str\(action\.get\("continuation_of"\)',
            source, flags=re.MULTILINE))
        self.assertIn(
            'continuation_source = str(action.get("continuation_of")', source)

    def test_canary21_frozen_artifacts_can_materialize_complete_dialogue(self) -> None:
        parent = load(PARENT)
        child_path = CANARY21 / "child_dialogue_promotable.json"
        child = load(child_path)
        source = load(CANARY21 / "source_initialization.json")
        expanded = assemble_expanded_dialogue(
            original_parent=parent,
            promotable_child=child,
            narrative_order_manifest=load(CANARY21 / "narrative_order_manifest.json"),
            speaker_to_entity_ref=source["speaker_to_entity_ref"],
            parent_path=PARENT,
            parent_sha256=hashlib.sha256(PARENT.read_bytes()).hexdigest(),
            child_path=child_path,
            child_sha256=hashlib.sha256(child_path.read_bytes()).hexdigest(),
        )
        validation = validate_expanded_dialogue(
            expanded, original_parent=parent, promotable_child=child)
        self.assertTrue(validation["valid"], validation["errors"])
        self.assertEqual(validation["parent_turn_count"], 60)
        self.assertEqual(validation["child_turn_count"], 38)
        self.assertEqual(validation["turn_count"], 98)

    def test_canary21_history_is_unchanged(self) -> None:
        self.assertEqual(hash_tree(CANARY21)["tree_sha256"], CANARY21_TREE_SHA)


if __name__ == "__main__":
    unittest.main()
