from __future__ import annotations

from scripts.current_method.restart_n5_spell.planner_realizer_temporal_boundary_probe_v1.compact_probe_v2 import (
    _compact_boundary_block,
)


def test_compact_boundary_block_keeps_open_scenes_and_locks_actions() -> None:
    block = _compact_boundary_block({"scenes": [{}, {}, {}]})
    assert "open/provisional" in block
    assert "too thick" in block
    assert "do not apply, brush, coat, baste" in block
    assert "only here may the glaze be applied" in block


def test_compact_boundary_block_does_not_dump_future_beats() -> None:
    block = _compact_boundary_block({"scenes": [{"interaction_beats": ["future beat"]}, {}]})
    assert "future beat" not in block
