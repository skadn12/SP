"""Immutable-contract, crash-recoverable local Solver request ledger."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from ..backends import BackendResponse
from ..core import atomic_json, canonical_json, sha256_bytes, sha256_file
from .io_utils import load_json, safe_name, write_once_json, write_once_text


LEDGER_VERSION = "h4_round0_request_ledger_v2"
RESPONSE_ENVELOPE_VERSION = "h4_round0_response_envelope_v1"


class AmbiguousSolverRequest(RuntimeError):
    pass


class RecordedSolverFailure(RuntimeError):
    def __init__(self, message: str, *, cached: bool):
        super().__init__(message)
        self.cached = cached


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def request_cache_key(contract: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(dict(contract)).encode("utf-8"))


class Round0RequestLedger:
    def __init__(self, run_root: Path, *, solver_contract_sha256: str):
        self.run_root = run_root
        self.root = run_root / "request_ledger"
        self.raw_prompts = run_root / "raw_prompts"
        self.raw_responses = run_root / "raw_responses"
        self.solver_contract_sha256 = solver_contract_sha256
        for path in (self.root, self.raw_prompts, self.raw_responses):
            path.mkdir(parents=True, exist_ok=True)

    def _state_path(self, root_id: str, cache_key: str) -> Path:
        return self.root / safe_name(root_id) / f"{cache_key}.json"

    def _response_path(self, cache_key: str) -> Path:
        return self.raw_responses / f"{cache_key}.json"

    def read(self, root_id: str, cache_key: str) -> dict[str, Any] | None:
        path = self._state_path(root_id, cache_key)
        return load_json(path) if path.is_file() else None

    @staticmethod
    def _assert_state_contract(
        state: Mapping[str, Any], frozen: Mapping[str, Any], prompt_sha: str,
    ) -> None:
        if state.get("ledger_version") != LEDGER_VERSION:
            raise RuntimeError("unsupported_request_ledger_version")
        if state.get("request_contract") != dict(frozen):
            raise RuntimeError(f"request_contract_drift:{state.get('request_cache_key')}")
        prompt_path = Path(str(state["prompt_path"]))
        if not prompt_path.is_file():
            raise RuntimeError(f"raw_prompt_missing:{state.get('request_cache_key')}")
        observed_prompt = prompt_path.read_text(encoding="utf-8")
        if sha256_bytes(observed_prompt.encode("utf-8")) != prompt_sha:
            raise RuntimeError(f"raw_prompt_hash_drift:{state.get('request_cache_key')}")

    def _load_envelope(self, cache_key: str) -> tuple[Path, dict[str, Any]] | None:
        path = self._response_path(cache_key)
        if not path.is_file():
            return None
        envelope = load_json(path)
        if envelope.get("response_envelope_version") != RESPONSE_ENVELOPE_VERSION:
            raise RuntimeError(f"response_envelope_version_mismatch:{cache_key}")
        if envelope.get("request_cache_key") != cache_key:
            raise RuntimeError(f"response_envelope_cache_key_mismatch:{cache_key}")
        raw = str(envelope.get("raw_text") or "")
        if sha256_bytes(raw.encode("utf-8")) != envelope.get("raw_response_sha256"):
            raise RuntimeError(f"raw_response_hash_drift:{cache_key}")
        if not isinstance(envelope.get("response_metadata"), Mapping):
            raise RuntimeError(f"response_envelope_metadata_invalid:{cache_key}")
        return path, envelope

    @staticmethod
    def _response(envelope: Mapping[str, Any], *, reused: bool) -> BackendResponse:
        metadata = dict(envelope.get("response_metadata") or {})
        if reused:
            metadata["reused"] = True
        return BackendResponse(
            str(envelope.get("raw_text") or ""),
            str(envelope["model_identifier"]),
            metadata,
        )

    @staticmethod
    def _finish_attempt(
        state: Mapping[str, Any], *, status: str, fields: Mapping[str, Any],
    ) -> dict[str, Any]:
        attempts = [dict(row) for row in state.get("attempts") or []]
        if not attempts or attempts[-1].get("status") != "reserved":
            raise RuntimeError("active_reserved_attempt_missing")
        attempts[-1] = {**attempts[-1], "status": status, **dict(fields)}
        return {**dict(state), "attempts": attempts}

    def _complete_from_envelope(
        self,
        state_path: Path,
        state: Mapping[str, Any],
        envelope_path: Path,
        envelope: Mapping[str, Any],
        *,
        recovered: bool,
    ) -> dict[str, Any]:
        completed_at = _now()
        attempts = [dict(row) for row in state.get("attempts") or []]
        if not attempts:
            raise RuntimeError("response_without_request_attempt")
        if int(envelope.get("attempt_id", -1)) != int(attempts[-1]["attempt_id"]):
            raise RuntimeError("response_envelope_attempt_id_mismatch")
        if attempts[-1].get("status") in {"reserved", "interrupted"}:
            attempts[-1] = {
                **attempts[-1],
                "status": "completed",
                "completed_at": completed_at,
                "completed_by_recovery": recovered,
            }
        elif attempts[-1].get("status") != "completed":
            raise RuntimeError("response_envelope_attempt_state_invalid")
        completed = {
            **dict(state),
            "status": "completed",
            "attempts": attempts,
            "completed_at": completed_at,
            "completed_by_recovery": recovered,
            "response_envelope_path": str(envelope_path),
            "response_envelope_sha256": sha256_file(envelope_path),
            "raw_response_sha256": envelope["raw_response_sha256"],
            "model_identifier": envelope["model_identifier"],
            "response_metadata": dict(envelope.get("response_metadata") or {}),
        }
        for key in ("error", "error_type", "failed_at", "interrupted_at"):
            completed.pop(key, None)
        atomic_json(state_path, completed)
        return completed

    def _completed_response(
        self, state: Mapping[str, Any], cache_key: str,
    ) -> BackendResponse:
        loaded = self._load_envelope(cache_key)
        if loaded is None:
            raise RuntimeError(f"completed_response_envelope_missing:{cache_key}")
        path, envelope = loaded
        if sha256_file(path) != state.get("response_envelope_sha256"):
            raise RuntimeError(f"response_envelope_hash_drift:{cache_key}")
        if envelope.get("model_identifier") != state.get("model_identifier"):
            raise RuntimeError(f"response_model_identifier_drift:{cache_key}")
        return self._response(envelope, reused=True)

    @staticmethod
    def _reserve_attempt(state: Mapping[str, Any]) -> dict[str, Any]:
        attempts = [dict(row) for row in state.get("attempts") or []]
        attempt_id = len(attempts) + 1
        attempts.append({
            "attempt_id": attempt_id,
            "status": "reserved",
            "reserved_at": _now(),
        })
        reserved = {
            **dict(state),
            "status": "reserved",
            "active_attempt_id": attempt_id,
            "attempts": attempts,
            "solver_generation_count": attempt_id,
        }
        for key in ("error", "error_type", "failed_at", "interrupted_at"):
            reserved.pop(key, None)
        return reserved

    def recover_orphaned_requests(self) -> dict[str, int]:
        """Recover nonterminal attempts while the caller holds the run lock."""
        report = {
            "orphan_request_count": 0,
            "response_envelope_recovered_count": 0,
            "retryable_request_count": 0,
        }
        for state_path in sorted(self.root.glob("*/*.json")):
            state = load_json(state_path)
            if state.get("ledger_version") != LEDGER_VERSION:
                raise RuntimeError(f"unsupported_request_ledger_version:{state_path}")
            request_contract = state.get("request_contract") or {}
            if request_contract.get("solver_contract_sha256") != self.solver_contract_sha256:
                raise RuntimeError(f"orphan_solver_contract_mismatch:{state_path}")
            cache_key = str(state.get("request_cache_key") or "")
            if request_cache_key(request_contract) != cache_key:
                raise RuntimeError(f"orphan_request_cache_key_mismatch:{state_path}")
            self._assert_state_contract(
                state, request_contract, str(state.get("prompt_sha256") or ""),
            )
            status = str(state.get("status") or "")
            if status == "completed":
                self._completed_response(state, cache_key)
                continue
            if status == "failed":
                attempts = list(state.get("attempts") or [])
                if not attempts or attempts[-1].get("status") != "failed":
                    raise RuntimeError(f"failed_request_attempt_invalid:{state_path}")
                continue
            if status not in {"reserved", "interrupted", "retryable"}:
                raise AmbiguousSolverRequest(
                    f"unsupported_request_state:{cache_key}:{status}"
                )
            report["orphan_request_count"] += 1
            loaded = self._load_envelope(cache_key)
            if loaded is not None:
                envelope_path, envelope = loaded
                self._complete_from_envelope(
                    state_path, state, envelope_path, envelope, recovered=True,
                )
                report["response_envelope_recovered_count"] += 1
                continue
            attempts = [dict(row) for row in state.get("attempts") or []]
            if attempts and attempts[-1].get("status") == "reserved":
                attempts[-1] = {
                    **attempts[-1],
                    "status": "abandoned",
                    "abandoned_at": _now(),
                    "abandon_reason": "process_ended_without_response_envelope",
                }
            retryable = {
                **state,
                "status": "retryable",
                "attempts": attempts,
                "recovered_at": _now(),
                "recovery_action": "retry_exact_contract_and_seed",
            }
            atomic_json(state_path, retryable)
            report["retryable_request_count"] += 1
        return report

    def invoke(
        self,
        *,
        contract: Mapping[str, Any],
        prompt: str,
        call: Callable[[], BackendResponse],
    ) -> tuple[BackendResponse, bool]:
        frozen = dict(contract)
        if frozen.get("solver_contract_sha256") != self.solver_contract_sha256:
            raise RuntimeError("request_solver_contract_sha_mismatch")
        cache_key = request_cache_key(frozen)
        root_id = str(frozen["root_id"])
        state_path = self._state_path(root_id, cache_key)
        prompt_sha = sha256_bytes(prompt.encode("utf-8"))
        if frozen.get("prompt_sha256") != prompt_sha:
            raise RuntimeError("request_prompt_hash_mismatch")
        existing = self.read(root_id, cache_key)
        if existing is not None:
            self._assert_state_contract(existing, frozen, prompt_sha)
            status = existing.get("status")
            if status == "completed":
                return self._completed_response(existing, cache_key), True
            if status == "failed":
                raise RecordedSolverFailure(
                    f"recorded_solver_failure:{cache_key}:{existing.get('error')}",
                    cached=True,
                )
            if status != "retryable":
                raise AmbiguousSolverRequest(
                    f"ambiguous_solver_request:{cache_key}:{status}"
                )
            state = self._reserve_attempt(existing)
        else:
            prompt_path = self.raw_prompts / f"{prompt_sha}.txt"
            write_once_text(prompt_path, prompt)
            state = self._reserve_attempt({
                "ledger_version": LEDGER_VERSION,
                "request_cache_key": cache_key,
                "request_contract": frozen,
                "created_at": _now(),
                "prompt_path": str(prompt_path),
                "prompt_sha256": prompt_sha,
                "attempts": [],
                "solver_generation_count": 0,
            })
        atomic_json(state_path, state)
        try:
            response = call()
        except Exception as error:
            failed_at = _now()
            failed = self._finish_attempt(
                state,
                status="failed",
                fields={
                    "failed_at": failed_at,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            failed.update({
                "status": "failed",
                "failed_at": failed_at,
                "error_type": type(error).__name__,
                "error": str(error),
            })
            atomic_json(state_path, failed)
            raise RecordedSolverFailure(
                f"solver_generation_failed:{cache_key}:{type(error).__name__}:{error}",
                cached=False,
            ) from error
        except BaseException as error:
            interrupted_at = _now()
            interrupted = self._finish_attempt(
                state,
                status="interrupted",
                fields={
                    "interrupted_at": interrupted_at,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            interrupted.update({
                "status": "interrupted",
                "interrupted_at": interrupted_at,
                "error_type": type(error).__name__,
                "error": str(error),
            })
            atomic_json(state_path, interrupted)
            raise
        envelope_path = self._response_path(cache_key)
        envelope = {
            "response_envelope_version": RESPONSE_ENVELOPE_VERSION,
            "request_cache_key": cache_key,
            "attempt_id": state["active_attempt_id"],
            "raw_text": response.raw_text,
            "raw_response_sha256": sha256_bytes(response.raw_text.encode("utf-8")),
            "model_identifier": response.model_identifier,
            "response_metadata": dict(response.metadata or {}),
        }
        write_once_json(envelope_path, envelope, with_sha=False)
        self._complete_from_envelope(
            state_path, state, envelope_path, envelope, recovered=False,
        )
        return response, False

    def record_decision(
        self, root_id: str, cache_key: str, decision: Mapping[str, Any],
    ) -> None:
        path = self._state_path(root_id, cache_key)
        state = load_json(path)
        if state.get("status") != "completed":
            raise RuntimeError(f"cannot_score_noncompleted_request:{cache_key}")
        frozen = dict(decision)
        if "decision" in state:
            if state["decision"] != frozen:
                raise RuntimeError(f"request_decision_drift:{cache_key}")
            return
        atomic_json(path, {
            **state,
            "decision": frozen,
            "decision_recorded_at": _now(),
        })

    def summary(self) -> dict[str, int]:
        states = self.states()
        attempts = [attempt for state in states for attempt in state.get("attempts") or []]
        return {
            "request_count": len(states),
            "completed_request_count": sum(row.get("status") == "completed" for row in states),
            "failed_request_count": sum(row.get("status") == "failed" for row in states),
            "ambiguous_request_count": sum(
                row.get("status") not in {"completed", "failed"} for row in states
            ),
            "actual_solver_generation_count": len(attempts),
            "retried_generation_count": sum(max(len(row.get("attempts") or []) - 1, 0) for row in states),
            "abandoned_attempt_count": sum(row.get("status") == "abandoned" for row in attempts),
            "interrupted_attempt_count": sum(row.get("status") == "interrupted" for row in attempts),
            "recovered_request_count": sum(
                bool(row.get("recovered_at")) or bool(row.get("completed_by_recovery"))
                for row in states
            ),
            "recovered_response_envelope_count": sum(
                bool(row.get("completed_by_recovery")) for row in states
            ),
        }

    def states(self) -> list[dict[str, Any]]:
        return [load_json(path) for path in sorted(self.root.glob("*/*.json"))]


__all__ = [
    "AmbiguousSolverRequest",
    "LEDGER_VERSION",
    "RESPONSE_ENVELOPE_VERSION",
    "RecordedSolverFailure",
    "Round0RequestLedger",
    "request_cache_key",
]
