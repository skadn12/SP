"""Offline-only preflight for a paired Round-1 Memory reconstruction smoke.

This module deliberately stops before any Memorax or model request.  It freezes
the first compiler-selected root, resolves the historical successful R02
control lineage, and creates remapped Parent/R01 Add plans for two empty future
namespaces.  The plans are inputs to a later, explicitly authorized live
bootstrap runner.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..core import canonical_json, sha256_file
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.payloads import (
    memorax_user_id,
)


# Keep these paths local to the isolated implementation.  Importing the
# compiler package here would also import unrelated runtime constants.
RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
DEFAULT_COMPILER_RUN = RUN_BASE / "h4_round1_feedback_compiler_v1_live_1"
DEFAULT_CONTINUATION_RUNS = RUN_BASE
DEFAULT_RUN_ROOT = RUN_BASE / (
    "h4_round1_feedback_memory_reconstruction_smoke_v1_live_1")
EXPECTED_ROOT_COUNT = 70
EXPECTED_FORMAL_COUNT = 10
VERSION = "h4_round1_feedback_memory_reconstruction_smoke_v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _verify_sidecar(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(f"source_missing:{path}")
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file():
        raise RuntimeError(f"source_sidecar_missing:{path}")
    expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
    actual = sha256_file(path)
    if expected != actual:
        raise RuntimeError(f"source_sha_mismatch:{path}")
    return actual


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    path.with_name(path.name + ".sha256").write_text(
        sha256_file(path) + "\n", encoding="utf-8")


def _semantic_projection(intent: Mapping[str, Any]) -> dict[str, Any]:
    """Remove transport identity/timestamps while retaining logical content."""
    payload = copy.deepcopy(dict(intent.get("payload") or {}))

    def scrub(value: Any, key: str | None = None) -> Any:
        if isinstance(value, Mapping):
            return {
                str(k): scrub(v, str(k))
                for k, v in value.items()
                if str(k) not in {
                    "user_id", "request_id", "task_id", "timestamp",
                    "payload_sha256", "api_key", "authorization",
                }
            }
        if isinstance(value, list):
            return [scrub(v, key) for v in value]
        return value

    projected_payload = scrub(payload)
    return {
        "round_index": int(intent.get("round_index") or 0),
        "session_id": str(intent.get("session_id") or ""),
        "source_dialogue_sha256": str(
            intent.get("source_dialogue_sha256") or ""),
        "turn_refs": [str(x) for x in intent.get("turn_refs") or []],
        "payload": projected_payload,
    }


def _remap_intent(
    intent: Mapping[str, Any], *, root_id: str, user_id: str,
    branch: str,
) -> dict[str, Any]:
    """Make a deterministic new-namespace Add plan without changing content."""
    result = copy.deepcopy(dict(intent))
    payload = result.setdefault("payload", {})
    payload["user_id"] = user_id
    metadata = payload.setdefault("metadata", {})
    metadata["root_id"] = root_id
    result["root_id"] = root_id
    result["user_id"] = user_id
    result["request_id"] = (
        f"{VERSION}-{branch}-{root_id}-r{int(result.get('round_index') or 0):02d}-"
        f"{_canonical_sha(_semantic_projection(intent))[:12]}"
    )
    # The historical logical R00/R01 session boundaries remain unchanged.
    result["task_id"] = None
    result["payload_sha256"] = _canonical_sha(payload)
    result["reconstruction_branch"] = branch
    result["live_submission"] = False
    result["api_key_in_artifact"] = False
    return result


def _root_state_candidates(
    continuation_runs: Path, root_id: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    pattern = "h4_round1_cumulative_continuation_fresh_mining_v1_live_*"
    for run in sorted(continuation_runs.glob(pattern)):
        for state_path in run.glob(
            f"shards/*/roots/{root_id}/root_state.json"):
            state = _load(state_path)
            if (
                state.get("status") == "COMPLETED"
                and state.get("memory_write_verified") is True
                and state.get("memory_visibility_verified") is True
                and int(state.get("qa_calls") or 0) == 0
                and int(state.get("solver_calls") or 0) == 0
            ):
                candidates.append({
                    "run_root": str(run.resolve()),
                    "state_path": str(state_path.resolve()),
                    "state_sha256": sha256_file(state_path),
                    "state": state,
                })
    return candidates


def _resolve_control_reference(
    continuation_runs: Path, root_id: str,
) -> dict[str, Any]:
    candidates = _root_state_candidates(continuation_runs, root_id)
    # A reused run points to the original successful state.  The authority is
    # the unique completed state without a prior_reference wrapper.
    authorities = [x for x in candidates if not x["state"].get("prior_reference")]
    if len(authorities) != 1:
        raise RuntimeError(
            f"control_authority_not_unique:{root_id}:{len(authorities)}")
    selected = authorities[0]
    state = selected["state"]
    report_path = Path(str(
        state.get("report_path") or "")).resolve()
    if not report_path.is_file():
        raise RuntimeError(f"control_report_missing:{root_id}")
    report = _load(report_path)
    if (
        report.get("status") != "completed"
        or report.get("root_id") != root_id
        or report.get("memory_write_verified") is not True
        or report.get("memory_visibility_verified") is not True
        or int(report.get("qa_calls") or 0) != 0
        or int(report.get("solver_calls") or 0) != 0
    ):
        raise RuntimeError(f"control_report_not_verified:{root_id}")
    attempt_root = Path(str(state.get("attempt_root") or "")).resolve()
    intent_path = attempt_root / "memory" / "child_add_intent.json"
    completion_path = attempt_root / "memory" / "child_add_completion.json"
    delta_path = attempt_root / "child_dialogue.json"
    cumulative_path = attempt_root / "promotion" / "promotable_expanded_dialogue.json"
    for path in (intent_path, completion_path, delta_path, cumulative_path):
        if not path.is_file():
            raise RuntimeError(f"control_artifact_missing:{path}")
    return {
        "authority_run_root": selected["run_root"],
        "state_path": selected["state_path"],
        "state_sha256": selected["state_sha256"],
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "attempt_root": str(attempt_root),
        "r02_control_delta_path": str(delta_path),
        "r02_control_delta_sha256": sha256_file(delta_path),
        "r02_control_cumulative_path": str(cumulative_path),
        "r02_control_cumulative_sha256": sha256_file(cumulative_path),
        "r02_control_add_intent_path": str(intent_path),
        "r02_control_add_intent_sha256": sha256_file(intent_path),
        "r02_control_add_completion_path": str(completion_path),
        "r02_control_add_completion_sha256": sha256_file(completion_path),
        "old_user_id": state.get("prior_user_id"),
        "old_r02_session_id": state.get("r02_session_id"),
        "qa_calls": int(report.get("qa_calls") or 0),
        "solver_calls": int(report.get("solver_calls") or 0),
        "memorax_calls": int(report.get("memorax_calls") or 0),
        "authority_selection": "unique_completed_state_without_prior_reference",
    }


def _source_parent_path(r01_dialogue_path: Path) -> Path:
    # The promoted complete dialogue carries the immutable parent prefix and
    # its parent_dialogue_sha256.  Keep the physical source path, while the
    # parent prefix boundary remains explicit in the audit.
    candidate = r01_dialogue_path.parents[2] / "expanded_dialogue.json"
    if candidate.is_file():
        return candidate
    return r01_dialogue_path


def build_preflight(
    *, run_root: Path, compiler_run: Path, continuation_runs: Path,
) -> dict[str, Any]:
    if run_root.exists() and any(run_root.iterdir()):
        raise RuntimeError(f"output_run_not_empty:{run_root}")
    run_root.mkdir(parents=True, exist_ok=True)

    required = [
        compiler_run / "selected_10_roots.jsonl",
        compiler_run / "root_feedback_cards.jsonl",
        compiler_run / "r01_only_memory_fork_plan.json",
        compiler_run / "feedback_prompt_token_budget.json",
        compiler_run / "final_feedback_compiler_report.json",
    ]
    source_sha = {str(path): _verify_sidecar(path) for path in required}
    selected = _load_jsonl(required[0])
    cards = _load_jsonl(required[1])
    fork_plan = _load(required[2])
    budget = _load(required[3])
    final = _load(required[4])
    if len(selected) != EXPECTED_FORMAL_COUNT:
        raise RuntimeError(f"formal_root_count:{len(selected)}")
    if final.get("status") != "ROUND1_FEEDBACK_COMPILER_COMPLETE":
        raise RuntimeError("compiler_not_complete")
    first = selected[0]
    root_id = str(first.get("root_id") or "")
    if not root_id:
        raise RuntimeError("selected_root_id_missing")
    plan_rows = [x for x in fork_plan.get("roots") or []
                 if str(x.get("root_id") or "") == root_id]
    if len(plan_rows) != 1:
        raise RuntimeError(f"fork_plan_root_not_unique:{root_id}")
    plan = plan_rows[0]
    if len(cards) != EXPECTED_FORMAL_COUNT or not any(
            str(x.get("root_id") or "") == root_id for x in cards):
        raise RuntimeError("feedback_card_selection_mismatch")
    if str(plan.get("fork_plan_version") or "").endswith("_v1") is False:
        raise RuntimeError("fork_plan_version_missing")

    root_manifest_path = continuation_runs / (
        "h4_round1_cumulative_continuation_fresh_mining_v1_live_47"
    ) / "round1_root_manifest.jsonl"
    _verify_sidecar(root_manifest_path)
    root_manifest = {
        str(x.get("root_id")): x for x in _load_jsonl(root_manifest_path)}
    if root_id not in root_manifest:
        raise RuntimeError(f"round1_root_not_found:{root_id}")
    lineage = root_manifest[root_id]
    canonical_dialogue = Path(str(lineage.get("canonical_dialogue_path") or ""))
    if not canonical_dialogue.is_file() or sha256_file(canonical_dialogue) != str(
            lineage.get("canonical_dialogue_sha256") or ""):
        raise RuntimeError(f"canonical_dialogue_sha_drift:{root_id}")
    if "r02" in canonical_dialogue.as_posix().casefold():
        raise RuntimeError("r02_in_canonical_input_path")

    parent_intent_path = Path(plan["r01_only_reconstruction"]["parent_add_intent_path"])
    r01_intent_path = Path(plan["r01_only_reconstruction"]["r01_add_intent_path"])
    r01_completion_path = Path(plan["r01_only_reconstruction"]["r01_add_completion_path"])
    for path, expected in (
        (parent_intent_path, plan["r01_only_reconstruction"]["parent_add_intent_sha256"]),
        (r01_intent_path, plan["r01_only_reconstruction"]["r01_add_intent_sha256"]),
        (r01_completion_path, plan["r01_only_reconstruction"]["r01_add_completion_sha256"]),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"r01_source_sha_drift:{path}")
    parent_intent = _load(parent_intent_path)
    r01_intent = _load(r01_intent_path)
    r01_completion = _load(r01_completion_path)
    parent_round = parent_intent.get("round_index")
    r01_round = r01_intent.get("round_index")
    if (
        parent_round is None or int(parent_round) != 0
        or r01_round is None or int(r01_round) != 1
        or r01_intent.get("session_id") != plan.get("source_r01_session_id")
        or r01_completion.get("verified_complete") is not True
    ):
        raise RuntimeError("r01_memory_contract_invalid")
    if any(not str(ref).startswith("R01:") for ref in r01_intent.get("turn_refs") or []):
        raise RuntimeError("r01_intent_contains_non_r01_ref")
    if any(str(ref).startswith("R02") for ref in r01_intent.get("turn_refs") or []):
        raise RuntimeError("r02_ref_in_r01_intent")

    control = _resolve_control_reference(continuation_runs, root_id)
    branches = plan.get("branches") or {}
    if set(branches) != {"control", "feedback"}:
        raise RuntimeError("fork_branches_incomplete")
    mapping = {}
    for branch in ("control", "feedback"):
        namespace = str((branches[branch] or {}).get("namespace") or "")
        if not namespace:
            raise RuntimeError(f"namespace_missing:{branch}")
        user_id = memorax_user_id(root_id, namespace=namespace)
        mapping[branch] = {
            "root_id": root_id,
            "namespace": namespace,
            "user_id": user_id,
            "r01_session_id": plan.get("source_r01_session_id"),
            "r02_session_id": (branches[branch] or {}).get("r02_session_id"),
            "feedback_card_exposed": bool(
                (branches[branch] or {}).get("feedback_card_exposed")),
            "fork_plan_version": plan.get("fork_plan_version"),
        }
    if mapping["control"]["feedback_card_exposed"] or not mapping["feedback"]["feedback_card_exposed"]:
        raise RuntimeError("branch_feedback_exposure_invalid")
    if mapping["control"]["namespace"] == mapping["feedback"]["namespace"]:
        raise RuntimeError("branch_namespace_collision")

    semantic = {
        "historical_parent": {
            "path": str(parent_intent_path),
            "sha256": sha256_file(parent_intent_path),
            "semantic_payload_sha256": _canonical_sha(_semantic_projection(parent_intent)),
        },
        "historical_r01": {
            "path": str(r01_intent_path),
            "sha256": sha256_file(r01_intent_path),
            "semantic_payload_sha256": _canonical_sha(_semantic_projection(r01_intent)),
        },
        "branches": {},
    }
    reconstruction_plans = {}
    for branch in ("control", "feedback"):
        user_id = mapping[branch]["user_id"]
        parent_plan = _remap_intent(
            parent_intent, root_id=root_id, user_id=user_id, branch=branch)
        r01_plan = _remap_intent(
            r01_intent, root_id=root_id, user_id=user_id, branch=branch)
        reconstruction_plans[branch] = {
            "parent": parent_plan,
            "r01": r01_plan,
        }
        semantic["branches"][branch] = {
            "parent_semantic_payload_sha256": _canonical_sha(
                _semantic_projection(parent_plan)),
            "r01_semantic_payload_sha256": _canonical_sha(
                _semantic_projection(r01_plan)),
            "parent_matches_history": _canonical_sha(
                _semantic_projection(parent_plan)) == semantic["historical_parent"]["semantic_payload_sha256"],
            "r01_matches_history": _canonical_sha(
                _semantic_projection(r01_plan)) == semantic["historical_r01"]["semantic_payload_sha256"],
        }

    if not all(
        semantic["branches"][b][key]
        for b in ("control", "feedback")
        for key in ("parent_matches_history", "r01_matches_history")
    ):
        raise RuntimeError("semantic_payload_reconstruction_drift")

    card = next(x for x in cards if str(x.get("root_id")) == root_id)
    report = {
        "preflight_version": f"{VERSION}_offline_preflight_v1",
        "status": "PASS",
        "decision": "pass",
        "created_at": _now(),
        "offline_only": True,
        "memorax_calls": 0,
        "api_calls": 0,
        "solver_calls": 0,
        "qa_calls": 0,
        "semantic_auditor_calls": 0,
        "dev_test_access": 0,
        "r02_generation": 0,
        "selected_order_index": 0,
        "root_id": root_id,
        "root_count_in_compiler": len(selected),
        "compiler_run": str(compiler_run.resolve()),
        "compiler_source_sha256": source_sha,
        "canonical_dialogue": {
            "path": str(canonical_dialogue.resolve()),
            "sha256": sha256_file(canonical_dialogue),
            "complete_dialogue_sha256": lineage.get("complete_dialogue_sha256"),
            "parent_dialogue_path": str(_source_parent_path(canonical_dialogue).resolve()),
            "parent_dialogue_sha256": lineage.get("round0_dialogue_sha256"),
            "r01_only_input": True,
        },
        "memory_source": {
            "parent_add_intent": str(parent_intent_path.resolve()),
            "r01_add_intent": str(r01_intent_path.resolve()),
            "r01_add_completion": str(r01_completion_path.resolve()),
            "r01_write_verified": True,
        },
        "control_reference": control,
        "namespace_mapping": mapping,
        "feedback_card": {
            "path": str((compiler_run / "root_feedback_cards.jsonl").resolve()),
            "root_id": root_id,
            "card_sha256": _canonical_sha(card),
            "feedback_version": card.get("feedback_version"),
        },
        "token_budget_reference": {
            "path": str(required[3].resolve()),
            "target_api_model": budget.get("target_api_model"),
            "expansion_input_plus_reserve": budget.get("expansion_input_plus_reserve"),
            "future_solver_limit": budget.get("future_solver_limit"),
        },
        "hard_gates": {
            "first_selected_root_used": True,
            "control_authority_unique": True,
            "old_control_read_only": True,
            "new_namespaces_not_submitted": True,
            "r01_payloads_semantically_reproducible": True,
            "control_reconstruction_r02_add": 0,
            "feedback_reconstruction_r02_add": 0,
            "r02_control_read_only_frozen": True,
            "r02_control_semantics_used_for_generation": False,
            "test_content_read": False,
        },
    }

    _write(run_root / "preflight_report.json", report)
    _write(run_root / "selected_smoke_root.json", {
        "selection_source": str((compiler_run / "selected_10_roots.jsonl").resolve()),
        "selection_index": 0,
        "root": first,
    })
    _write(run_root / "namespace_mapping.json", mapping)
    _write(run_root / "frozen_control_reference.json", control)
    _write(run_root / "control_fingerprint_audit.json", {
        "status": "PASS",
        "historical_control": control,
        "old_control_not_rewritten": True,
        "qa_calls": 0,
        "solver_calls": 0,
    })
    _write(run_root / "semantic_payload_comparison.json", semantic)
    _write(run_root / "r01_reconstruction_equivalence_report.json", {
        "status": "ANCHOR_EQUIVALENT",
        "exact_semantic_payload_parity": True,
        "control_vs_feedback_parity": True,
        "historical_vs_control_parity": True,
        "historical_vs_feedback_parity": True,
        "r02_control_content_semantics_read": False,
    })
    _write(run_root / "control_reconstruction" / "parent_add_intent.json",
           reconstruction_plans["control"]["parent"])
    _write(run_root / "control_reconstruction" / "r01_child_add_intent.json",
           reconstruction_plans["control"]["r01"])
    _write(run_root / "feedback_reconstruction" / "parent_add_intent.json",
           reconstruction_plans["feedback"]["parent"])
    _write(run_root / "feedback_reconstruction" / "r01_child_add_intent.json",
           reconstruction_plans["feedback"]["r01"])
    _write(run_root / "memory_write_count_audit.json", {
        "status": "PASS",
        "control_reconstruction": {"parent_add": 0, "r01_child_add": 0, "r02_add": 0},
        "feedback_reconstruction": {"parent_add": 0, "r01_child_add": 0, "r02_add": 0},
        "offline_plans_only": True,
    })
    _write(run_root / "final_feedback_memory_smoke_report.json", {
        "status": "OFFLINE_PREFLIGHT_PASS",
        "next_status": "READY_FOR_LIVE_R01_BOOTSTRAP_SMOKE",
        "root_id": root_id,
        "memorax_calls": 0,
        "api_calls": 0,
        "r02_generation": 0,
        "r02_write": 0,
        "feedback_adherence": "NOT_RUN",
        "feedback_leakage": "NOT_RUN",
        "note": "Live bootstrap and feedback R02 remain explicitly unrun.",
    })
    (run_root / "final_feedback_memory_smoke_report.md").write_text(
        "# Round-1 Feedback Memory Reconstruction Smoke\n\n"
        "- Status: `OFFLINE_PREFLIGHT_PASS`\n"
        f"- Root: `{root_id}`\n"
        "- R01 semantic parity: `PASS`\n"
        "- Memorax/API calls: `0`\n"
        "- R02 generation/write: `0/0`\n"
        "- Next status: `READY_FOR_LIVE_R01_BOOTSTRAP_SMOKE`\n",
        encoding="utf-8",
    )
    md = run_root / "final_feedback_memory_smoke_report.md"
    md.with_name(md.name + ".sha256").write_text(
        sha256_file(md) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--compiler-run", type=Path, default=DEFAULT_COMPILER_RUN)
    parser.add_argument("--continuation-runs", type=Path,
                        default=DEFAULT_CONTINUATION_RUNS)
    args = parser.parse_args(argv)
    report = build_preflight(
        run_root=args.run_root.resolve(), compiler_run=args.compiler_run.resolve(),
        continuation_runs=args.continuation_runs.resolve())
    print(json.dumps({
        "status": report["status"],
        "decision": report["decision"],
        "root_id": report["root_id"],
        "memorax_calls": report["memorax_calls"],
        "next_status": "READY_FOR_LIVE_R01_BOOTSTRAP_SMOKE",
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
