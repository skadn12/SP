from __future__ import annotations

from typing import Any, Mapping


GUIDED_COUNTS = {"single_choice": 2, "multi_select": 1, "ordering": 1}
FREE_COUNTS = {"single_choice": 4, "multi_select": 2, "ordering": 2}
ROLES = ("before_state", "trigger", "after_state")


def expected_counts(source: str) -> dict[str, int]:
    return dict(GUIDED_COUNTS if source == "contract_guided" else FREE_COUNTS)


def source_prefix(source: str) -> str:
    return "C" if source == "contract_guided" else "F"


def contract_summary(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "intent_id": str(contract.get("intent_id") or ""),
        "intent_type": str(contract.get("intent_type") or ""),
        "discriminative_variable": str(contract.get("discriminative_variable") or ""),
        "required_evidence_groups": {
            str(k): [str(x) for x in v]
            for k, v in (contract.get("required_evidence_groups") or {}).items()
            if isinstance(v, list)
        },
    }


def provenance(intent_id: str, source: str, ordinal: int) -> dict[str, Any]:
    return {
        "candidate_id": f"{intent_id}-{source_prefix(source)}-Q{ordinal:02d}",
        "intent_id": intent_id,
        "generation_source": source,
    }

