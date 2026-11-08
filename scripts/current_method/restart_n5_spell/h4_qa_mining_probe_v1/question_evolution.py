"""Phase-0 evolution interface.

Question evolution is deliberately not executed in this probe.  The output is
a stable stub so a later Phase 1 can add at most one revision round without
changing the Phase-0 artifacts.
"""

from __future__ import annotations

from typing import Any, Mapping


def make_phase0_evolution_stubs(
    candidates: list[Mapping[str, Any]], difficulty: Mapping[str, Any]
) -> list[dict[str, Any]]:
    buckets = {str(row.get("candidate_id")): row.get("difficulty_bucket")
               for row in difficulty.get("candidate_summaries") or []}
    return [{
        "candidate_id": row.get("candidate_id"),
        "lineage": [row.get("candidate_id")],
        "round": 0,
        "max_round": 1,
        "eligible_for_phase1_evolution": buckets.get(str(row.get("candidate_id")))
            in {"hard_candidate", "unresolved_full_zero", "unresolved"},
        "executed": False,
        "reason": "phase0_freeze_no_evolution",
    } for row in candidates]
