from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.frozen_loader import (
    load_dialogue, parent_view,
)
from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.io_utils import (
    prepare_empty, read_json, sha256_file, write_json, write_jsonl,
)
from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.live_adapter import (
    LiveJSONClient, extract_json,
)
from scripts.current_method.restart_n5_spell.scene_complexity_probe_v1.analyzer import analyze_chain

from .contracts import (FIXED_MAX_BEATS, FIXED_MIN_BEATS, FIXED_SCENE_COUNT,
                        OBJECTIVES, validate_objective_plan)
from .probe import (_objective_summary, _paired_comparison, _prompt,
                    _repair_prompt, _select_events)


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
PROBE_VERSION = "planner_objective_probe_v1_batch_stability_v1"

SNAPSHOTS = (
    Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/simple_append_memorax_v1_qa_option_v2_13836_live_1"),
    Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/conversation_chronicles_episode_146700_qa_fact_first_probe_v1_live_1"),
    Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/conversation_chronicles_episode_178392_3round_qa_budget16384_live_1"),
    Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/conversation_chronicles_episode_41625_child_promotion_qa_live_1"),
    Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/conversation_chronicles_episode_91716_memory_event_continuation_qa_live_1"),
)


def _load_cases(max_roots: int) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    cases: list[dict[str, Any]] = []
    parents: dict[str, dict[str, Any]] = {}
    seen_roots: set[str] = set()
    for snapshot in SNAPSHOTS:
        dialogue_path = snapshot / "frozen" / "expanded_dialogue.json"
        memory_path = snapshot / "frozen" / "memory_overview.json"
        if not dialogue_path.exists() or not memory_path.exists():
            raise FileNotFoundError(f"frozen_snapshot_missing:{snapshot}")
        dialogue = load_dialogue(dialogue_path)
        root_id = str(dialogue["root_id"] or "")
        if not root_id or root_id in seen_roots:
            raise ValueError(f"duplicate_or_empty_root_id:{root_id}")
        seen_roots.add(root_id)
        memory = read_json(memory_path)
        selected = _select_events(list(memory.get("events") or []), 1)
        if not selected:
            continue
        event = selected[0]
        case_id = f"POB{len(cases) + 1:02d}"
        cases.append({
            "case_id": case_id, "root_id": root_id,
            "case_namespace": f"{root_id}:{event.get('canonical_key')}",
            "event_key": str(event.get("canonical_key") or event.get("display_key") or ""),
            "event_content": [str(value) for value in (event.get("content") or [])],
            "snapshot_path": str(snapshot), "dialogue_path": str(dialogue_path),
            "memory_path": str(memory_path),
            "dialogue_sha256": sha256_file(dialogue_path),
            "memory_sha256": sha256_file(memory_path),
        })
        parents[case_id] = parent_view(dialogue)
        if len(cases) >= max_roots:
            break
    return cases, parents


def _batch_conclusion(paired: Mapping[str, Any]) -> dict[str, Any]:
    count = int(paired.get("paired_complete_cases") or 0)
    outcomes = list(paired.get("pair_outcomes") or [])
    positive = sum(int(row.get("objective_gain") or 0) > 0 for row in outcomes)
    negative = sum(int(row.get("objective_gain") or 0) < 0 for row in outcomes)
    unchanged = count - positive - negative
    if count < 3:
        label = "INSUFFICIENT_COMPLETE_PAIRS"
        reason = f"Only {count} complete O0/O1 pair(s); at least 3 are required for a batch-direction decision."
    elif positive >= 2 and negative == 0:
        label = "OBJECTIVE_CONTROL_BATCH_SIGNAL"
        reason = f"Objective control improved {positive}/{count} complete pairs with no negative pair."
    elif positive > negative:
        label = "OBJECTIVE_CONTROL_MIXED_POSITIVE"
        reason = f"Objective control improved {positive}/{count} pairs, reduced {negative}, and left {unchanged} unchanged."
    elif positive == negative:
        label = "NO_STABLE_OBJECTIVE_GAIN"
        reason = f"Positive and negative pair effects are tied ({positive} each); {unchanged} are unchanged."
    else:
        label = "OBJECTIVE_CONTROL_UNSTABLE_OR_HARMFUL"
        reason = f"Negative effects ({negative}) exceed positive effects ({positive})."
    return {"label": label, "reason": reason, "positive_pairs": positive,
            "negative_pairs": negative, "unchanged_pairs": unchanged}


def _report(metrics: Mapping[str, Any], records: list[Mapping[str, Any]]) -> str:
    def render(value: Any) -> str:
        return "n/a" if value is None else f"{float(value):.3f}"
    lines = [
        "# Planner Objective Probe v1 — Multi-root Batch Stability", "",
        "Five distinct frozen roots contribute one transition-affording event each. Capacity is fixed at 3 scenes and 3–7 beats; only O0/O1 objective text differs.", "",
        "## Dataset", "",
        f"- Distinct roots: {metrics['dataset']['number_of_roots']}",
        f"- Events: {metrics['dataset']['number_of_events']}",
        f"- Planner cases: {metrics['dataset']['number_of_planner_cases']}",
        f"- Complete O0/O1 pairs: {metrics['paired_comparison']['paired_complete_cases']}", "",
        "## Objective comparison", "",
        "| Objective | Valid | Complex chain | Final at last | Trigger rate | Reconsideration | Revision |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in OBJECTIVES:
        row = metrics["by_objective"][key]
        lines.append(f"| {key} | {row['valid_planner_cases']}/{row['planner_cases']} | {render(row['complex_chain_rate'])} | {render(row['final_at_last_rate'])} | {render(row['trigger_backed_transition_rate'])} | {render(row['reconsideration_rate'])} | {render(row['revision_rate'])} |")
    lines.extend(["", "## Paired outcomes", ""])
    for row in metrics["paired_comparison"]["pair_outcomes"]:
        case = next(item for item in records if item["case_id"] == row["case_id"])
        lines.append(f"- `{row['case_id']}` / `{case['root_id']}` / `{case['event_key']}`: O0={row['O0_complex']}, O1={row['O1_complex']}, gain={row['objective_gain']:+d}")
    if metrics["paired_comparison"]["excluded_case_ids"]:
        lines.append(f"- Excluded incomplete pairs: {metrics['paired_comparison']['excluded_case_ids']}")
    conclusion = metrics["conclusion"]
    lines.extend([
        "", "## Conclusion", "", f"Verdict: **{conclusion['label']}**", "",
        conclusion["reason"], "",
        "## Recommended next step", "",
        ("Run an isolated Realizer-adherence probe on accepted O1 plans."
         if conclusion["label"] == "OBJECTIVE_CONTROL_BATCH_SIGNAL"
         else "Do not modify the mainline; inspect incomplete/negative pairs and refine Planner contracts first."),
    ])
    return "\n".join(lines) + "\n"


def run_batch(*, run_root: Path, config: Path | None, live: bool, max_roots: int) -> dict[str, Any]:
    prepare_empty(run_root)
    cases, parents = _load_cases(max_roots)
    client = LiveJSONClient(config, max_tokens=5000, temperature=0.0) if live and config else None
    records: list[dict[str, Any]] = []
    for case in cases:
        parent = parents[case["case_id"]]
        valid_speakers = sorted({str(row.get("speaker") or "")
                                 for row in parent["dialogue"]["turns"]
                                 if str(row.get("speaker") or "")})
        for key, objective in OBJECTIVES.items():
            value: Any = {}
            api_meta: dict[str, Any] = {"status": "not_run_no_live", "attempts": []}
            if client is not None:
                try:
                    raw, first_meta = client.complete(_prompt(case, parent, objective))
                    value = extract_json(raw)
                    api_meta = {"status": "pass", "attempts": [first_meta]}
                except Exception as exc:
                    api_meta = {"status": "failed", "attempts": [],
                                "error": f"{type(exc).__name__}:{exc}"}
            contract = validate_objective_plan(
                value, objective_key=key, case_id=case["case_id"],
                valid_speakers=valid_speakers)
            if client is not None and not contract["valid"] and isinstance(value, Mapping):
                try:
                    raw, repair_meta = client.complete(_repair_prompt(
                        case, parent, objective, value, list(contract["errors"])))
                    repaired = extract_json(raw)
                    repaired_contract = validate_objective_plan(
                        repaired, objective_key=key, case_id=case["case_id"],
                        valid_speakers=valid_speakers)
                    api_meta["attempts"].append(repair_meta)
                    api_meta["status"] = "pass_repaired" if repaired_contract["valid"] else "repair_invalid"
                    value, contract = repaired, repaired_contract
                except Exception as exc:
                    api_meta["status"] = "repair_failed"
                    api_meta["repair_error"] = f"{type(exc).__name__}:{exc}"
            analysis = analyze_chain(value) if contract["valid"] else {}
            records.append({**case, "objective": key, "valid_speakers": valid_speakers,
                            "api": api_meta, "plan": value, "contract": contract,
                            "analysis": analysis})
    by_objective = {key: _objective_summary([row for row in records if row["objective"] == key])
                    for key in OBJECTIVES}
    paired = _paired_comparison(records)
    calls = client.calls if client else 0
    metrics = {
        "probe_version": PROBE_VERSION,
        "dataset": {"number_of_roots": len({case["root_id"] for case in cases}),
                    "number_of_events": len(cases), "number_of_planner_cases": len(records),
                    "root_ids": [case["root_id"] for case in cases]},
        "fixed_capacity": {"scene_count": FIXED_SCENE_COUNT, "min_beats": FIXED_MIN_BEATS,
                           "max_beats": FIXED_MAX_BEATS},
        "by_objective": by_objective, "paired_comparison": paired,
        "conclusion": _batch_conclusion(paired), "api_calls": calls, "planner_calls": calls,
        "realizer_calls": 0, "solver_calls": 0, "writer_calls": 0,
        "memorax_search_calls": 0, "memorax_add_calls": 0,
        "limitations": ["Planner-only; Realizer adherence is not tested.",
                        "Five roots are a small stability batch.",
                        "Temperature=0 is shared, but the provider exposes no explicit sampling seed."],
    }
    write_jsonl(run_root / "inputs.jsonl", cases)
    write_jsonl(run_root / "planner_cases.jsonl", records)
    write_jsonl(run_root / "audits.jsonl", [{"case_id": row["case_id"], "root_id": row["root_id"],
        "objective": row["objective"], "contract": row["contract"],
        "analysis": row["analysis"]} for row in records])
    write_json(run_root / "metrics.json", metrics)
    write_json(run_root / "manifest.json", {"probe_version": PROBE_VERSION, "isolated": True,
        "historical_runs_read_only": True, "root_namespace_isolated": True,
        "planner_calls": calls, "realizer_calls": 0, "solver_calls": 0,
        "writer_calls": 0, "memorax_search_calls": 0, "memorax_add_calls": 0})
    (run_root / "report.md").write_text(_report(metrics, records), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="multi-root Planner objective stability batch")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--config", type=Path, default=Path("/root/autodl-tmp/gxy-selfplay/config.json"))
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--max-roots", type=int, default=5)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = (args.run_root or RUN_BASE / f"planner_objective_probe_v1_batch_{stamp}").resolve()
    metrics = run_batch(run_root=run_root, config=args.config, live=args.live,
                        max_roots=max(1, min(args.max_roots, len(SNAPSHOTS))))
    print(json.dumps({"run_root": str(run_root), "metrics": metrics}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
