from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .backends import BackendResponse
from .core import atomic_json, atomic_text, canonical_json, sha256_bytes


class AmbiguousRequestError(RuntimeError):
    pass


class RequestLedger:
    """Crash-safe request state shared by mock and future live backends."""

    def __init__(self, run_root: Path, manifest_sha256: str, backend_mode: str):
        self.root = run_root / "ledger"
        self.state_dir = self.root / "requests"
        self.raw_dir = run_root / "raw" / "backend_responses"
        self.events = self.root / "request_events.jsonl"
        self.manifest_sha256 = manifest_sha256
        self.backend_mode = backend_mode
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, request_id: str) -> Path:
        return self.state_dir / f"{request_id}.json"

    def read(self, request_id: str) -> dict[str, Any] | None:
        path = self._path(request_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _event(self, record: dict[str, Any]) -> None:
        self.events.parent.mkdir(parents=True, exist_ok=True)
        with self.events.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            import os
            os.fsync(handle.fileno())

    def invoke(self, *, request_id: str, intent: str, attempt: int, prompt_sha256: str,
               call: Callable[[], BackendResponse], budget_class: str,
               budget_limit: int | None, request_metadata: dict[str, Any] | None = None,
               total_budget_limit: int | None = None,
               backend_identity: dict[str, Any] | None = None) -> BackendResponse:
        request_metadata = dict(request_metadata or {})
        backend_identity = dict(backend_identity or {})
        request_metadata_sha256 = sha256_bytes(canonical_json(request_metadata).encode())
        backend_identity_sha256 = sha256_bytes(canonical_json(backend_identity).encode())
        existing = self.read(request_id)
        if existing:
            if existing["status"] == "completed":
                expected = {"intent":intent, "attempt":attempt, "prompt_sha256":prompt_sha256,
                    "manifest_sha256":self.manifest_sha256, "backend_mode":self.backend_mode,
                    "budget_class":budget_class,
                    "request_metadata_sha256":request_metadata_sha256,
                    "backend_identity_sha256":backend_identity_sha256}
                drift = {key:(existing.get(key), value) for key, value in expected.items()
                         if existing.get(key) != value}
                if drift:
                    raise RuntimeError(f"request contract drift for {request_id}: {drift}")
                raw = Path(existing["raw_response_path"]).read_text(encoding="utf-8")
                if sha256_bytes(raw.encode()) != existing.get("raw_response_sha256"):
                    raise RuntimeError(f"raw response hash drift for {request_id}")
                return BackendResponse(raw, existing["model_identifier"],
                    {**(existing.get("response_metadata") or {}), "reused": True})
            raise AmbiguousRequestError(
                f"request {request_id} is {existing['status']}; refusing automatic resend")
        peer_states = [json.loads(path.read_text(encoding="utf-8")) for path in self.state_dir.glob("*.json")]
        reserved_for_class = sum(x.get("budget_class") == budget_class for x in peer_states)
        if budget_limit is not None and reserved_for_class >= budget_limit:
            raise RuntimeError(f"budget exhausted for {budget_class}: {reserved_for_class}/{budget_limit}")
        if total_budget_limit is not None and len(peer_states) >= total_budget_limit:
            raise RuntimeError(f"total request budget exhausted: {len(peer_states)}/{total_budget_limit}")
        now = datetime.now(timezone.utc).isoformat()
        reservation = {"request_id": request_id, "intent": intent, "attempt": attempt,
            "prompt_sha256": prompt_sha256, "manifest_sha256": self.manifest_sha256,
            "backend_mode": self.backend_mode, "status": "reserved", "created_at": now,
            "budget_class": budget_class, "budget_limit": budget_limit,
            "total_budget_limit":total_budget_limit,
            "request_metadata":request_metadata,
            "request_metadata_sha256":request_metadata_sha256,
            "backend_identity":backend_identity,
            "backend_identity_sha256":backend_identity_sha256,
            "budget_reservation_index": reserved_for_class + 1,
            "paid_generation_count": 0, "mock_generation_count": 0,
        }
        atomic_json(self._path(request_id), reservation)
        self._event(reservation)
        response = call()
        raw_path = self.raw_dir / f"{request_id}.txt"
        atomic_text(raw_path, response.raw_text)
        completed = {**reservation, "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "raw_response_path": str(raw_path),
            "raw_response_sha256": sha256_bytes(response.raw_text.encode()),
            "model_identifier": response.model_identifier,
            "response_metadata": response.metadata,
            "mock_generation_count": int(self.backend_mode == "mock")}
        if self.backend_mode == "live":
            completed["paid_generation_count"] = 1
        atomic_json(self._path(request_id), completed)
        self._event(completed)
        return response

    def completed(self) -> list[dict[str, Any]]:
        return [json.loads(path.read_text(encoding="utf-8")) for path in self.state_dir.glob("*.json")
                if json.loads(path.read_text(encoding="utf-8")).get("status") == "completed"]

    def record_decision(self, request_id: str, decision: str) -> None:
        state = self.read(request_id)
        if not state or state.get("status") != "completed":
            raise RuntimeError(f"cannot finalize non-completed request {request_id}")
        if "final_decision" in state:
            if state["final_decision"] != decision:
                raise RuntimeError(f"decision drift for {request_id}")
            return
        state["final_decision"] = decision
        state["decision_recorded_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(self._path(request_id), state)
        self._event({**state, "event": "decision_recorded"})

    def summary(self) -> dict[str, int]:
        states = [json.loads(p.read_text(encoding="utf-8")) for p in self.state_dir.glob("*.json")]
        return {"request_count": len(states),
            "completed_request_count": sum(x.get("status") == "completed" for x in states),
            "ambiguous_request_count": sum(x.get("status") != "completed" for x in states),
            "mock_backend_call_count": sum(int(x.get("mock_generation_count", 0)) for x in states),
            "paid_api_generation_count": sum(int(x.get("paid_generation_count", 0)) for x in states)}
