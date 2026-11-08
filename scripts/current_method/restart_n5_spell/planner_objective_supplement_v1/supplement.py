from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.frozen_loader import (
    load_dialogue,
    parent_view,
)
from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.io_utils import (
    prepare_empty,
    read_json,
    sha256_file,
    write_json,
    write_jsonl,
)
from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.live_adapter import (
    LiveJSONClient,
    extract_json,
)
from scripts.current_method.restart_n5_spell.scene_complexity_probe_v1.analyzer import analyze_chain
from scripts.current_method.restart_n5_spell.planner_objective_probe_v1.contracts import (
    OBJECTIVES,
    validate_objective_plan,
)
from scripts.current_method.restart_n5_spell.planner_objective_probe_v1.probe import (
    _prompt,
    _repair_prompt,
)


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
DEFAULT_SNAPSHOT = RUN_BASE / "simple_append_memorax_v1_batch_pilot_3_5roots/roots/cc-train-episode-216742/qa"
DEFAULT_CASE_ID = "POS07"
DEFAULT_ROOT_ID = "cc-train-episode-216742"
DEFAULT_EVENT_KEY = "breaking things in house"
PROBE_VERSION = "planner_objective_supplement_v1"


def _load_input(snapshot: Path, root_id: str, event_key: str, case_id: str) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    dialogue_path = snapshot / "frozen" / "expanded_dialogue.json"
    memory_path = snapshot / "frozen" / "memory_overview.json"
    if not dialogue_path.exists() or not memory_path.exists():
        raise FileNotFoundError(f"frozen_snapshot_missing:{snapshot}")
    dialogue = load_dialogue(dialogue_path)
    if str(dialogue.get("root_id") or "") != root_id:
        raise ValueError(f"root_id_mismatch:{dialogue.get('root_id')}:{root_id}")
    memory = read_json(memory_path)
    events = list(memory.get("events") or [])
    event = next((row for row in events
                  if str(row.get("canonical_key") or row.get("display_key") or "") == event_key), None)
    if not event:
        raise ValueError(f"event_missing:{event_key}")
    case = {
        "case_id": case_id,
        "root_id": root_id,
        "event_key": event_key,
        "event_content": [str(value) for value in (event.get("content") or [])],
        "snapshot_path": str(snapshot),
        "dialogue_path": str(dialogue_path),
        "memory_path": str(memory_path),
        "dialogue_sha256": sha256_file(dialogue_path),
        "memory_sha256": sha256_file(memory_path),
    }
    parent = parent_view(dialogue)
    speakers = sorted({str(row.get("speaker") or "")
                       for row in parent["dialogue"]["turns"]
                       if str(row.get("speaker") or "")})
    return case, parent, speakers


def _run_objective(client: LiveJSONClient, case: dict[str, Any], parent: dict[str, Any],
                   speakers: list[str], objective_key: str, case_id: str) -> dict[str, Any]:
    objective = OBJECTIVES[objective_key]
    attempts: list[dict[str, Any]] = []
    value: Any = {}
    status = "failed"
    error = None
    try:
        raw, meta = client.complete(_prompt(case, parent, objective))
        attempts.append(meta)
        value = extract_json(raw)
        status = "pass"
    except Exception as exc:
        error = f"{type(exc).__name__}:{exc}"
    contract = validate_objective_plan(value, objective_key=objective_key,
                                       case_id=case_id, valid_speakers=speakers)
    if not contract["valid"] and isinstance(value, dict):
        try:
            raw, meta = client.complete(_repair_prompt(
                case, parent, objective, value, list(contract["errors"])))
            attempts.append(meta)
            repaired = extract_json(raw)
            repaired_contract = validate_objective_plan(
                repaired, objective_key=objective_key, case_id=case_id,
                valid_speakers=speakers)
            value, contract = repaired, repaired_contract
            status = "pass_repaired" if contract["valid"] else "repair_invalid"
        except Exception as exc:
            status = "repair_failed"
            error = f"{type(exc).__name__}:{exc}"
    return {
        **case,
        "objective": objective_key,
        "valid_speakers": speakers,
        "api": {"status": status, "attempts": attempts, **({"error": error} if error else {})},
        "plan": value,
        "contract": contract,
        "analysis": analyze_chain(value) if contract["valid"] else {},
    }


def _report(metrics: dict[str, Any], records: list[dict[str, Any]]) -> str:
    lines = [
        "# Planner Objective Supplement v1",
        "",
        "Single-root O0/O1 Planner-only supplement on frozen episode-61555.",
        "",
        f"- Event: `{metrics['source']['event_key']}`",
        f"- Complete pair: `{metrics['paired_complete']}`",
        f"- O0 complex: `{metrics['by_objective']['O0']['complex_chain']}`",
        f"- O1 complex: `{metrics['by_objective']['O1']['complex_chain']}`",
        f"- Planner API calls: `{metrics['planner_calls']}`",
        "- Realizer/QA/Solver/Writer/Memorax calls: `0`",
        "",
        "## Contract outcomes",
        "",
    ]
    for row in records:
        lines.append(f"- `{row['objective']}`: api={row['api']['status']}, valid={row['contract']['valid']}, errors={row['contract'].get('errors', [])}, complex={row['analysis'].get('complex_chain')}")
    lines.extend(["", "This supplement does not modify the mainline or the frozen five-root batch.", ""])
    return "\n".join(lines)


def run(*, run_root: Path, config: Path, snapshot: Path, root_id: str,
        event_key: str, case_id: str) -> dict[str, Any]:
    prepare_empty(run_root)
    case, parent, speakers = _load_input(snapshot, root_id, event_key, case_id)
    client = LiveJSONClient(config, max_tokens=5000, temperature=0.0)
    records = [_run_objective(client, case, parent, speakers, key, case_id) for key in OBJECTIVES]
    by_objective = {key: {
        "valid": bool(next(row for row in records if row["objective"] == key)["contract"]["valid"]),
        "complex_chain": bool(next(row for row in records if row["objective"] == key)["analysis"].get("complex_chain")),
    } for key in OBJECTIVES}
    complete = all(row["contract"]["valid"] for row in records)
    metrics = {
        "probe_version": PROBE_VERSION,
        "source": {
            "snapshot": str(snapshot),
            "dialogue_sha256": case["dialogue_sha256"],
            "memory_sha256": case["memory_sha256"],
            "root_id": root_id,
            "event_key": event_key,
        },
        "by_objective": by_objective,
        "paired_complete": complete,
        "planner_calls": client.calls,
        "api_calls": client.calls,
        "realizer_calls": 0,
        "qa_writer_calls": 0,
        "solver_calls": 0,
        "writer_calls": 0,
        "memorax_search_calls": 0,
        "memorax_add_calls": 0,
    }
    write_json(run_root / "inputs.json", {"case": case, "speakers": speakers})
    write_jsonl(run_root / "planner_cases.jsonl", records)
    write_jsonl(run_root / "audits.jsonl", [{
        "objective": row["objective"], "contract": row["contract"],
        "analysis": row["analysis"], "api": row["api"],
    } for row in records])
    write_json(run_root / "metrics.json", metrics)
    write_json(run_root / "manifest.json", {
        "probe_version": PROBE_VERSION, "isolated": True,
        "source_snapshot_read_only": str(snapshot),
        "api_role_allowlist": ["planner"], "planner_calls": client.calls,
        "realizer_calls": 0, "qa_writer_calls": 0, "solver_calls": 0,
        "writer_calls": 0, "memorax_search_calls": 0, "memorax_add_calls": 0,
        "outputs": ["inputs.json", "planner_cases.jsonl", "audits.jsonl", "metrics.json", "report.md", "manifest.json"],
    })
    (run_root / "report.md").write_text(_report(metrics, records), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="single-root Planner objective supplement")
    parser.add_argument("--config", type=Path, default=Path("/root/autodl-tmp/gxy-selfplay/config.json"))
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--root-id", default=DEFAULT_ROOT_ID)
    parser.add_argument("--case-id", default=DEFAULT_CASE_ID)
    parser.add_argument("--event-key", default=DEFAULT_EVENT_KEY)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = (args.run_root or RUN_BASE / f"planner_objective_supplement_v1_{stamp}").resolve()
    print(json.dumps({"run_root": str(run_root), "metrics": run(
        run_root=run_root, config=args.config, snapshot=args.snapshot.resolve(),
        root_id=args.root_id, event_key=args.event_key, case_id=args.case_id,
    )}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
