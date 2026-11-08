"""Stable paired sampling seeds for the K=3 probe.

The seed intentionally excludes context mode and display position.  The same
three slot seeds are therefore paired across both context modes and all
display permutations; only the prompt/input changes.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


SEED_SCHEDULE_VERSION = "h4_k3_paired_seed_v1"


def stable_seed(root_id: str, candidate_id: str, slot_id: int) -> int:
    if isinstance(slot_id, bool):
        raise ValueError("slot_id_must_be_nonnegative_integer")
    try:
        slot = int(slot_id)
    except (TypeError, ValueError) as error:
        raise ValueError("slot_id_must_be_nonnegative_integer") from error
    if slot < 0:
        raise ValueError("slot_id_must_be_nonnegative")
    payload = json.dumps(
        {
            "version": SEED_SCHEDULE_VERSION,
            "root_id": str(root_id),
            "candidate_id": str(candidate_id),
            "slot_id": slot,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    # vLLM accepts a 32-bit seed.  Avoid Python's process-randomized hash().
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32 - 1)


def seed_schedule(root_id: str, candidate_id: str, rollout_count: int = 3) -> list[dict[str, Any]]:
    if isinstance(rollout_count, bool):
        raise ValueError("rollout_count_must_be_positive")
    try:
        count = int(rollout_count)
    except (TypeError, ValueError) as error:
        raise ValueError("rollout_count_must_be_positive") from error
    if count < 1:
        raise ValueError("rollout_count_must_be_positive")
    return [
        {
            "slot_id": slot,
            "seed": stable_seed(root_id, candidate_id, slot),
            "paired_factors": ["context_mode", "display_position"],
        }
        for slot in range(count)
    ]
