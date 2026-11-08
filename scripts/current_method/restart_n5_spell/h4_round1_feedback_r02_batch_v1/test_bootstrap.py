from __future__ import annotations

from .bootstrap import _validate_intent


def test_validate_intent_rejects_live_submission_and_r02_refs() -> None:
    base = {
        "live_submission": False,
        "root_id": "cc-train-episode-1",
        "round_index": 1,
        "user_id": "gxy-selfplay:round1-paired-feedback:x:cc-train-episode-1",
        "session_id": "cc-train-episode-1:memorax:R01",
        "turn_refs": ["R01:S01:T001"],
        "payload": {"user_id": "gxy-selfplay:round1-paired-feedback:x:cc-train-episode-1"},
    }
    _validate_intent(base, root_id="cc-train-episode-1",
                     user_id=base["user_id"], round_index=1)
    bad = {**base, "turn_refs": ["R02:S01:T001"]}
    try:
        _validate_intent(bad, root_id="cc-train-episode-1",
                         user_id=base["user_id"], round_index=1)
    except RuntimeError as error:
        assert "r02_ref" in str(error)
    else:
        raise AssertionError("R02 refs must be rejected")
