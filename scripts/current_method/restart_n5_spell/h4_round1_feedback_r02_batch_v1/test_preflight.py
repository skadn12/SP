from __future__ import annotations

from .preflight import _remap_intent


def test_remap_preserves_zero_round_and_changes_only_transport_identity() -> None:
    source = {
        "root_id": "cc-train-episode-1",
        "round_index": 0,
        "user_id": "old:cc-train-episode-1",
        "session_id": "cc-train-episode-1:memorax:R00",
        "payload": {
            "user_id": "old:cc-train-episode-1",
            "session_id": "cc-train-episode-1:memorax:R00",
            "metadata": {"root_id": "cc-train-episode-1", "round_index": 0},
            "messages": [{"role": "user", "content": "hello"}],
        },
    }
    remapped = _remap_intent(
        source, root_id="cc-train-episode-1",
        user_id="new:cc-train-episode-1", branch="feedback")
    assert remapped["round_index"] == 0
    assert remapped["user_id"] == "new:cc-train-episode-1"
    assert remapped["payload"]["user_id"] == "new:cc-train-episode-1"
    assert remapped["session_id"] == source["session_id"]
    assert remapped["payload"]["messages"] == source["payload"]["messages"]
    assert remapped["live_submission"] is False
