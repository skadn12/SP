"""Backend-neutral runner interfaces for the slim pipeline."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from .schemas import validate_evidence_committer_output


class EvidenceCommitterUnavailable(RuntimeError):
    """Raised when semantic commitment is requested without a configured role."""


class EvidenceCommitterRunner(Protocol):
    def commit(self, *, prompt: str, request: Mapping[str, Any]) -> Mapping[str, Any]:
        ...


def run_evidence_committer(
    *, prompt: str, request: Mapping[str, Any],
    runner: EvidenceCommitterRunner | None,
) -> dict[str, Any]:
    """Invoke an explicitly supplied runner; never fall back to local semantics."""
    if runner is None:
        raise EvidenceCommitterUnavailable(
            "evidence_committer_runner_not_configured; claim remains model_claimed")
    output = dict(runner.commit(prompt=prompt, request=request))
    validation = validate_evidence_committer_output(output)
    if not validation["valid"]:
        raise ValueError("evidence_committer_output_invalid:" +
                         ",".join(validation["errors"]))
    return output

