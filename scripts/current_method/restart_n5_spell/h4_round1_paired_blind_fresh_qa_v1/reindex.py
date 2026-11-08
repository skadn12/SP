"""Offline correction of the paired QA batch candidate identity index.

The live batch is immutable.  This command derives a new versioned run from
its raw candidate rows and request ledgers; it does not call the backend.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .batch import (
    BRANCHES,
    CONTROL_PATHS,
    FEEDBACK_PATHS,
    SMOKE_CONTROL,
    SMOKE_FEEDBACK,
    SMOKE_ROOT,
    RUN_BASE,
    _json,
    _offline_finalize_branch,
    _request_states,
    _scope_candidate,
    load_json,
    load_obj,
    preflight,
    run_batch,
)


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_tree_merge(src: Path, dst: Path) -> None:
    if not src.is_dir():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            _copy_tree_merge(item, target)
        else:
            _copy_file(item, target)


def _raw_without_scoped_id(rows: list[dict]) -> list[dict]:
    result = []
    for row in rows:
        item = dict(row)
        # Force the corrected global row ordinal in _scope_candidate.  The
        # previous ID is retained in source_candidate_id for lineage.
        item.pop("candidate_id", None)
        result.append(item)
    return result


def _request_dirs(source_branch: Path, root_id: str, branch: str) -> list[Path]:
    if root_id == SMOKE_ROOT:
        if branch == "control":
            return [SMOKE_CONTROL]
        smoke_feedback_initial = RUN_BASE / "h4_round1_paired_blind_fresh_qa_v1_live_2" / "feedback"
        return [smoke_feedback_initial, SMOKE_FEEDBACK]
    return [source_branch]


def _logical_request_id(row: dict) -> str:
    request_id = str(row.get("request_id") or "")
    metadata = row.get("request_metadata") or {}
    recovery_of = metadata.get("recovery_of") or metadata.get("recovery_of_request_id")
    if recovery_of:
        request_id = str(recovery_of)
    if "-attempt-" in request_id:
        request_id = request_id.rsplit("-attempt-", 1)[0]
    if "-repair-" in request_id:
        request_id = request_id.replace("-repair-", "-generator-")
    return request_id


def _effective_request_states(*dirs: Path) -> list[dict]:
    """Fold a completed repair over its reserved generator request.

    A reserved request is a real technical attempt, but once its explicitly
    recorded repair succeeds it must not keep a pair in a false nonterminal
    state.  Unrepaired reserved requests remain in the returned list so the
    final report still fails closed.
    """
    raw = _request_states(*dirs)
    groups: dict[str, list[dict]] = {}
    for row in raw:
        groups.setdefault(_logical_request_id(row), []).append(row)
    selected: list[dict] = []
    for logical_id, rows in sorted(groups.items()):
        completed = [x for x in rows if x.get("status") == "completed"]
        if completed:
            selected.extend(completed)
            continue
        selected.extend(rows)
    return selected


def derive(source: Path, output: Path) -> dict:
    if output.exists():
        raise RuntimeError(f"output_exists:{output}")
    output.mkdir(parents=True)

    # Copy frozen pair/preflight inputs byte-for-byte.  Their sidecars remain
    # valid because this operation does not change the file contents.
    for name in (
        "preflight_report.json",
        "paired_qa_contract.json",
        "batch_request_plan.json",
        "paired_dialogue_manifest.jsonl",
        "paired_dialogue_summary.json",
        "feedback_memory_contract_audit.json",
        "quarantine_root_manifest.jsonl",
    ):
        if (source / name).is_file():
            _copy_file(source / name, output / name)
            if (source / f"{name}.sha256").is_file():
                _copy_file(source / f"{name}.sha256", output / f"{name}.sha256")

    frozen_order = load_obj(source / "preflight_report.json")["frozen_success_order"]
    for branch in BRANCHES:
        for root_id in frozen_order:
            src = source / branch / "roots" / root_id
            if not (src / "candidate_pool.json").is_file():
                raise RuntimeError(f"missing_branch_artifact:{branch}:{root_id}")
            dst = output / branch / "roots" / root_id
            dst.mkdir(parents=True, exist_ok=True)

            # Preserve raw request evidence for the derived prompt audit.
            for dirname in ("raw", "raw_prompts", "raw_responses", "ledger"):
                _copy_tree_merge(src / dirname, dst / dirname)
            if root_id == SMOKE_ROOT:
                # The reused smoke branch is reference-only in live_4.  Copy
                # both its initial and recovery artifacts into the derived
                # audit tree without changing either source run.
                smoke_dirs = [SMOKE_CONTROL] if branch == "control" else [
                    RUN_BASE / "h4_round1_paired_blind_fresh_qa_v1_live_2" / "feedback",
                    SMOKE_FEEDBACK,
                ]
                for smoke_dir in smoke_dirs:
                    for dirname in ("raw", "raw_prompts", "raw_responses", "ledger"):
                        _copy_tree_merge(smoke_dir / dirname, dst / dirname)
                source_ref = src / "source_reference.json"
                if source_ref.is_file():
                    _copy_file(source_ref, dst / "source_reference.json")
            for name in ("request_events.jsonl", "request_summary.json"):
                if (src / name).is_file():
                    _copy_file(src / name, dst / name)
            if (src / "dialogue_source.json").is_file():
                _copy_file(src / "dialogue_source.json", dst / "dialogue_source.json")

            raw_rows = _raw_without_scoped_id(load_json(src / "candidate_pool.json"))
            valid_rows = _raw_without_scoped_id(load_json(src / "candidates_valid.json"))
            validity = load_obj(src / "validity_audit.json")
            dialogue = load_obj(src / "dialogue_source.json")
            request_dirs = _request_dirs(src, root_id, branch)
            raw_states = _request_states(*request_dirs)
            states = _effective_request_states(*request_dirs)
            result = _offline_finalize_branch(
                branch_dir=dst,
                root_id=root_id,
                branch=branch,
                dialogue=dialogue,
                raw_rows=raw_rows,
                valid_rows=valid_rows,
                validity=validity,
                request_states=states,
                source_run=str(src),
                reused_by_reference=(root_id == SMOKE_ROOT),
            )
            result["reindexed_from"] = str(src)
            result["raw_attempt_count"] = len(raw_states)
            result["raw_nonterminal_attempt_count"] = sum(x.get("status") != "completed" for x in raw_states)
            result["effective_request_count"] = len(states)
            result["superseded_nonterminal_attempt_count"] = max(
                0, result["raw_nonterminal_attempt_count"] - result["nonterminal_request_count"]
            )
            _json(dst / "request_attempt_collapse.json", {
                "raw_attempt_count": len(raw_states),
                "raw_nonterminal_attempt_count": result["raw_nonterminal_attempt_count"],
                "effective_request_count": len(states),
                "effective_nonterminal_request_count": result["nonterminal_request_count"],
                "superseded_nonterminal_attempt_count": result["superseded_nonterminal_attempt_count"],
                "policy": "completed repair/attempt-02 replaces its reserved logical request; all raw ledgers remain preserved",
                "raw_request_ids": [x.get("request_id") for x in raw_states],
                "effective_request_ids": [x.get("request_id") for x in states],
            })
            stored = {k: v for k, v in result.items() if k not in {"request_states", "candidate_rows", "valid_rows", "gates"}}
            _json(dst / "branch_result.json", stored)

    # Reuse the production aggregator.  Because every branch_result exists,
    # run_batch only reads local files and writes corrected aggregate audits.
    return run_batch(output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    args = parser.parse_args(argv)
    result = derive(args.source_run.resolve(), args.output_run.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
