from __future__ import annotations

import copy
import hashlib

from .adapter import (
    FEEDBACK_PROMPT_VERSION,
    assert_dialogue_only_memorax_payload,
    inject_feedback_card,
    override_feedback_session,
)


CARD = {
    "feedback_version": "round1_feedback_compiler_v1",
    "root_id": "cc-train-episode-254852",
    "target": {
        "anchor_refs": ["R01:S02:T001"],
        "constraints": ["append-only continuation"],
        "desired_event_operator": ["show a later trigger"],
        "frontier_status": "emerging_frontier_both",
        "gap_type": "temporal_precedence",
        "source_candidate_sha256": "0" * 64,
    },
}


def test_injects_abstract_card_without_mutating_input():
    prompt = {"prompt_version": "base_v1", "prompt": "AUTHORITATIVE TRANSCRIPT"}
    original = copy.deepcopy(prompt)
    result = inject_feedback_card(prompt, CARD)
    assert prompt == original
    assert result["prompt_version"] == FEEDBACK_PROMPT_VERSION
    assert "R01:S02:T001" in result["prompt"]
    assert "<box>" not in result["prompt"]
    assert result["feedback_card_exposed"] is True


def test_session_override_rehashes_payload_and_keeps_r02_only_payload():
    intent = {
        "root_id": "cc-train-episode-254852",
        "round_index": 2,
        "session_id": "cc-train-episode-254852:memorax:R02",
        "payload_sha256": "old",
        "request_id": "old",
        "payload": {
            "session_id": "cc-train-episode-254852:memorax:R02",
            "feedback_card_exposed": True,
            "feedback_card_sha256": "old-hash",
            "metadata": {"round_index": 2, "feedback_card_exposed": True},
            "messages": [{"content": "R02 only"}],
            "window_context": [{"content": "R02 only"}],
        },
    }
    result = override_feedback_session(intent, root_id=intent["root_id"])
    assert result["session_id"].endswith(":R02-feedback")
    assert result["payload"]["session_id"] == result["session_id"]
    assert "feedback_card_exposed" not in result["payload"]
    assert "feedback_card_sha256" not in result["payload"]
    assert "feedback_card_exposed" not in result["payload"]["metadata"]
    assert result["feedback_card_exposed"] is True
    expected = hashlib.sha256(
        __import__("json").dumps(result["payload"], ensure_ascii=False,
                                  sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert result["payload_sha256"] == expected
    assert result["request_id"].endswith(expected[:12])


def test_payload_guard_rejects_feedback_control_fields():
    try:
        assert_dialogue_only_memorax_payload({
            "messages": [], "window_context": [],
            "metadata": {"feedback_card": {"gap_type": "ordering"}},
        })
    except ValueError as error:
        assert str(error).startswith("feedback_control_in_memorax_payload:")
    else:
        raise AssertionError("feedback payload marker was not rejected")
