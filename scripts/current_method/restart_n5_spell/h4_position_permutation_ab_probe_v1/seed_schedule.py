"""Deterministic paired seeds for the position-effect probe.

Context mode, displayed permutation, and output mode are intentionally absent
from the seed payload.  This keeps paired conditions comparable and makes a
resume produce the same sampling stream.
"""

from __future__ import annotations

import hashlib
import json


SEED_SCHEDULE_VERSION = "h4_position_ab_paired_seed_v1"


def stable_seed(root_id: str, candidate_id: str, slot_id: int) -> int:
    if isinstance(slot_id, bool):
        raise ValueError("slot_id_must_be_nonnegative_integer")
    try:
        slot = int(slot_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("slot_id_must_be_nonnegative_integer") from exc
    if slot < 0:
        raise ValueError("slot_id_must_be_nonnegative")
    payload = json.dumps(
        {"version": SEED_SCHEDULE_VERSION, "root_id": str(root_id),
         "candidate_id": str(candidate_id), "slot_id": slot},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32 - 1)


__all__ = ["SEED_SCHEDULE_VERSION", "stable_seed"]
