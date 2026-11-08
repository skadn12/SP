"""Deterministic I/O helpers for immutable Round-0 artifacts."""

from __future__ import annotations

import errno
import fcntl
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..core import atomic_json, atomic_text, canonical_json, sha256_bytes, sha256_file


class RunLock:
    """Non-blocking process lock held for the complete live Solver phase."""

    def __init__(self, run_root: Path):
        self.path = run_root / "solver.run.lock"
        self._handle = None

    def __enter__(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            handle.close()
            if error.errno in {errno.EACCES, errno.EAGAIN}:
                raise RuntimeError(f"solver_run_lock_busy:{self.path}") from error
            raise
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        self._handle = handle
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._handle is None:
            return
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def verify_sha_sidecar(path: Path) -> str:
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file():
        raise RuntimeError(f"sha256_sidecar_missing:{path}")
    expected = sidecar.read_text(encoding="utf-8").strip()
    observed = sha256_file(path)
    if expected != observed:
        raise RuntimeError(f"sha256_sidecar_mismatch:{path}:{expected}:{observed}")
    return observed


def safe_name(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return text or "item"


def jsonl_text(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )


def write_once_json(path: Path, value: Any, *, with_sha: bool = True) -> None:
    expected = canonical_json(value)
    if path.exists():
        observed = canonical_json(load_json(path))
        if observed != expected:
            raise RuntimeError(f"immutable_artifact_drift:{path}")
    else:
        atomic_json(path, value)
    if with_sha:
        write_once_text(path.with_name(path.name + ".sha256"), sha256_file(path) + "\n")


def write_once_jsonl(
    path: Path, rows: Iterable[Mapping[str, Any]], *, with_sha: bool = True,
) -> None:
    text = jsonl_text(rows)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"immutable_artifact_drift:{path}")
    else:
        atomic_text(path, text)
    if with_sha:
        write_once_text(path.with_name(path.name + ".sha256"), sha256_file(path) + "\n")


def write_once_text(path: Path, value: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != value:
            raise RuntimeError(f"immutable_artifact_drift:{path}")
    else:
        atomic_text(path, value)


def stable_hash_int(*parts: Any) -> int:
    payload = canonical_json([str(part) for part in parts]).encode("utf-8")
    return int.from_bytes(bytes.fromhex(sha256_bytes(payload))[:8], "big")


def stable_seed(root_id: str, candidate_id: str, slot_id: int) -> int:
    from . import SEED_SCHEDULE_VERSION

    if isinstance(slot_id, bool) or int(slot_id) < 0:
        raise ValueError("slot_id_must_be_nonnegative_integer")
    payload = {
        "version": SEED_SCHEDULE_VERSION,
        "root_id": str(root_id),
        "candidate_id": str(candidate_id),
        "slot_id": int(slot_id),
    }
    digest = sha256_bytes(canonical_json(payload).encode("utf-8"))
    return int.from_bytes(bytes.fromhex(digest)[:8], "big") % (2**32 - 1)


def prepare_run_root(run_root: Path, source_run: Path) -> None:
    from . import PROBE_VERSION

    run_root = run_root.resolve()
    identity = {
        "probe_version": PROBE_VERSION,
        "run_root": str(run_root),
        "source_run": str(source_run.resolve()),
        "source_read_only": True,
        "historical_runs_read_only": True,
    }
    if run_root.exists() and not (run_root / "run_identity.json").is_file():
        raise RuntimeError(f"run_root_exists_without_identity:{run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    write_once_json(run_root / "run_identity.json", identity)
    for name in (
        "request_ledger",
        "raw_prompts",
        "raw_responses",
        "candidate_results",
        "logs",
    ):
        (run_root / name).mkdir(exist_ok=True)
