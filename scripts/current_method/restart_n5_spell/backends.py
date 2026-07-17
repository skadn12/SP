from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class RealizerRequest:
    request_id: str
    session_index: int
    attempt: int
    prompt: str
    prompt_sha256: str


@dataclass(frozen=True)
class QuestionerRequest:
    request_id: str
    batch_index: int
    assigned_gold_letters: tuple[str, str, str]
    prompt: str
    prompt_sha256: str
    valid_turn_ids: tuple[str, ...]
    valid_session_ids: tuple[str, ...]


@dataclass(frozen=True)
class SolverRequest:
    request_id: str
    qa_id: str
    evaluation_mode: str
    order_id: str
    target_slot: int
    retry_index: int
    sampling_seed: int
    options: dict[str, str]
    canonical_answer_text: str
    prompt_sha256: str
    prompt: str = ""


@dataclass(frozen=True)
class BackendResponse:
    raw_text: str
    model_identifier: str
    metadata: dict[str, Any] = field(default_factory=dict)


class RealizerBackend(Protocol):
    def generate_session(self, request: RealizerRequest) -> BackendResponse: ...


class QuestionerBackend(Protocol):
    def generate_batch(self, request: QuestionerRequest) -> BackendResponse: ...


class SolverBackend(Protocol):
    def generate(self, request: SolverRequest) -> BackendResponse: ...
