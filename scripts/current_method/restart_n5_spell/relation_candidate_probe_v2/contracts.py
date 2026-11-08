"""Contracts for a transcript-first proposal-to-outcome probe."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


RELATION_TYPE = "proposal_outcome"
VALID_ROLES = {"proposal", "outcome"}
VALID_STATUSES = {"eligible", "uncertain", "rejected"}
VALID_CONFIDENCES = {"high", "medium", "uncertain"}


@dataclass(frozen=True)
class Turn:
    turn_index: int
    turn_ref: str
    speaker: str
    text: str
    origin: str
    root_id: str

    @property
    def scope(self) -> str:
        return "parent_only" if self.origin == "source_grounded" else "child_only"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"scope": self.scope}


@dataclass(frozen=True)
class Fact:
    fact_id: str
    root_id: str
    event_key: str
    event_summary: str
    actor: str
    actors: list[str]
    relation_role: str
    value: str
    turn_ref: str
    turn_index: int
    speaker: str
    evidence_text: str
    origin: str
    cue: str
    strength: str = "high"
    source_structured_refs: list[str] = field(default_factory=list)
    value_tokens: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RelationCandidate:
    relation_id: str
    root_id: str
    relation_type: str
    event_key: str
    event_summary: str
    proposal: dict[str, Any]
    outcome: dict[str, Any]
    value_relation: str
    turn_distance: int
    source_structured_refs: list[str]
    confidence: str
    reason: str
    status: str
    cross_parent_child: bool
    scope: str
    validation_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Rejection:
    pair_id: str
    root_id: str
    proposal_fact_id: str | None
    outcome_fact_id: str | None
    reason: str
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

