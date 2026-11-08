from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.frozen_loader import (
    DEFAULT_RUN, load_frozen_inputs, parent_view,
)
from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.io_utils import (
    prepare_empty, read_json, read_jsonl, write_json, write_jsonl,
)
from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.live_adapter import (
    LiveJSONClient, extract_json,
)

from .analyzer import analyze_chain, summarize
from .contracts import SETTINGS, Setting, validate_plan


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
HISTORICAL_P4 = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/difficulty_probe_suite_v1_20260821T210535Z/p4a_question_aware_state/plans.json")
PROBE_VERSION = "scene_complexity_probe_v1"
EVENT_SELECTION_RULE = "prefer_transition_affording_event_text_then_source_order_v1"
TRANSITION_AFFORDANCE_TERMS = (
    "plan", "consider", "decid", "choos", "agree", "propos",
    "organize", "arrang", "responsib",
)


def _case(case_index: int, event: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_id": f"SC{case_index:02d}",
        "event_key": str(event.get("canonical_key") or event.get("display_key") or f"event-{case_index}"),
        "event_content": [str(value) for value in (event.get("content") or [])],
    }


def _select_events(events: list[Mapping[str, Any]], max_events: int) -> list[Mapping[str, Any]]:
    def score(row: Mapping[str, Any]) -> int:
        text = " ".join([str(row.get("canonical_key") or ""),
                         *[str(value) for value in (row.get("content") or [])]]).casefold()
        return sum(term in text for term in TRANSITION_AFFORDANCE_TERMS)
    ranked = sorted(enumerate(events), key=lambda item: (-score(item[1]), item[0]))
    return [row for _, row in ranked[:max_events]]


def _prompt(case: Mapping[str, Any], parent: Mapping[str, Any], setting: Setting) -> str:
    speakers = sorted({str(row.get("speaker") or "") for row in parent["dialogue"]["turns"] if str(row.get("speaker") or "")})
    event_context = {"event_key": case["event_key"], "event_content": case["event_content"], "canonical_speakers": speakers}
    return f"""You are the Planner in an isolated scene-capacity diagnostic. Do not write dialogue. Do not add QA facts. Plan one coherent continuation of the frozen event below. The same planning objective is used in settings A/B/C; only capacity differs. Attempt a natural state-evolution chain when the event supports it. Do not manufacture arbitrary reversals, false history, or unrelated distractors.

CAPACITY SETTING
- setting: {setting.key}
- exactly {setting.scene_count} scenes
- {setting.min_beats} to {setting.max_beats} interaction beats per scene

DIAGNOSTIC OBJECTIVE
Track exactly one concrete state slot across all scenes. Preserve the distinction between a value change (for example venue=A to venue=B), a stage change (tentative to contested to final), and ordinary event progress (preparing to completed). A meaningful revision needs a concrete trigger such as a constraint, new requirement, new information, external event, or stakeholder objection. Mere "changed mind" is not a trigger. Do not mark a final resolution before the last scene. If a complex chain is unsupported, return the simplest truthful chain; do not fake complexity.

FROZEN EVENT INPUT
{json.dumps(event_context, ensure_ascii=False, indent=2)}

The output is a probe-local diagnostic annotation, not a production Planner schema or a new persistent state system. Use exact canonical speaker names in action text. Return only JSON with no markdown:
{{
  "case_id": "{case['case_id']}",
  "setting": "{setting.key}",
  "round_goal": "...",
  "scenes": [
    {{
      "scene_index": 1,
      "primary_action": "proposal|challenge|tradeoff|reconsideration|revision|resolution|event_progress|other",
      "state_slot": "same concrete slot name in every scene",
      "before_value": "...",
      "before_stage": "unset|tentative|contested|reconsidering|revised|final|in_progress|completed|other",
      "trigger": {{"type": "constraint|new_requirement|new_information|external_event|stakeholder_objection|none", "content": "concrete cause, or explicitly no new trigger"}},
      "transition_type": "proposal|challenge|tradeoff|reconsideration|revision|resolution|event_progress|other",
      "required_transition": "...",
      "after_value": "...",
      "after_stage": "unset|tentative|contested|reconsidering|revised|final|in_progress|completed|other",
      "resolution_status": "open|provisional|final",
      "interaction_beats": ["..."]
    }}
  ]
}}"""


def _repair_prompt(
    case: Mapping[str, Any], parent: Mapping[str, Any], setting: Setting,
    plan: Mapping[str, Any], errors: list[str],
) -> str:
    return _prompt(case, parent, setting) + f"""

REPAIR ONLY
The previous diagnostic plan failed these deterministic contract checks:
{json.dumps(errors, ensure_ascii=False)}

Previous plan:
{json.dumps(dict(plan), ensure_ascii=False, indent=2)}

Return one complete corrected JSON object. Preserve the same grounded event and
state slot. Fix scene count, beat count, enums, and adjacent state continuity.
Do not add complexity merely to satisfy the analyzer."""


def _historical_reference() -> dict[str, Any]:
    if not HISTORICAL_P4.exists():
        return {"path": str(HISTORICAL_P4), "available": False}
    value = read_json(HISTORICAL_P4)
    cases = value if isinstance(value, list) else []
    return {
        "path": str(HISTORICAL_P4), "available": True,
        "planner_cases": len(cases),
        "scene_counts": [len((row.get("plan") or {}).get("scenes") or []) for row in cases if isinstance(row, Mapping)],
        "note": "Historical P4 includes fallback materialization and is reference-only, not an A/B/C causal condition.",
    }


def _paired_comparison(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in records:
        by_case.setdefault(str(row.get("case_id") or ""), {})[str(row.get("setting") or "")] = row
    complete = [case_id for case_id, rows in by_case.items()
                if all(rows.get(key, {}).get("contract", {}).get("valid") for key in SETTINGS)]
    rates = {}
    for key in SETTINGS:
        values = [bool(by_case[case_id][key].get("analysis", {}).get("complex_chain")) for case_id in complete]
        rates[key] = sum(values) / len(values) if values else None
    return {
        "paired_complete_cases": len(complete), "paired_case_ids": complete,
        "excluded_case_ids": sorted(set(by_case) - set(complete)),
        "complex_chain_rate_by_setting": rates,
    }


def _conclusion(paired: Mapping[str, Any]) -> dict[str, Any]:
    rates = dict(paired.get("complex_chain_rate_by_setting") or {})
    if not paired.get("paired_complete_cases") or any(rates.get(key) is None for key in SETTINGS):
        return {"code": "D", "label": "insufficient_valid_data", "reason": "No event has valid Planner output in all three paired settings."}
    a, b, c = float(rates["A"]), float(rates["B"]), float(rates["C"])
    b_gain, c_gain = b > a, c > a
    if b_gain and not c_gain:
        return {"code": "B", "label": "per_scene_budget_likely", "reason": "Only increased per-scene capacity improved complex-chain rate."}
    if c_gain and not b_gain:
        return {"code": "A", "label": "scene_count_likely", "reason": "Only adding scenes improved complex-chain rate."}
    if b_gain and c_gain:
        return {"code": "D", "label": "both_capacity_factors", "reason": "Both capacity interventions improved complex-chain rate."}
    return {"code": "C", "label": "planner_objective_or_other",
            "reason": f"Across {paired['paired_complete_cases']} fully valid paired event(s), neither capacity intervention improved complex-chain rate; capacity is not supported as the sole bottleneck."}


def _report(metrics: Mapping[str, Any], records: list[Mapping[str, Any]]) -> str:
    lines = [
        "# Scene Complexity Diagnostic Probe v1", "",
        "Research question: Does the current scene budget/granularity prevent the generation of complex state evolution?", "",
        "This is a Planner-only paired diagnostic. No dialogue, QA, Solver, Writer, Realizer, or Memorax call is made.", "",
        "## Dataset", "",
        f"- Roots: {metrics['dataset']['number_of_roots']}",
        f"- Frozen events: {metrics['dataset']['number_of_events']}",
        f"- Planner cases: {metrics['dataset']['number_of_planner_cases']}",
        f"- Fully valid A/B/C paired events: {metrics['paired_comparison']['paired_complete_cases']}",
        f"- Historical P4 reference: `{metrics['dataset']['historical_reference']['path']}`", "",
        "## Setting comparison", "",
        "| Setting | Valid cases | Scenes/case | Complex chain rate | State value changes | Premature closure | Trigger rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    def render(value: Any) -> str:
        return "n/a" if value is None else f"{float(value):.3f}"
    for key in SETTINGS:
        row = metrics["by_setting"][key]
        lines.append(
            f"| {key} | {row['valid_planner_cases']}/{row['planner_cases']} | {render(row['scenes_per_case'])} | "
            f"{render(row['complex_chain_rate'])} | {render(row['mean_state_value_changes'])} | "
            f"{render(row['premature_closure_rate'])} | {render(row['trigger_backed_transition_rate'])} |")
    lines.extend(["", "## Transition and scene-role distribution", ""])
    for key in SETTINGS:
        row = metrics["by_setting"][key]
        lines.append(f"- **{key}**: transitions={row['transition_type_distribution']}; roles={row['scene_role_distribution']}; reconsideration_rate={render(row['reconsideration_rate'])}; revision_rate={render(row['revision_rate'])}")
    lines.extend(["", "## Failure examples", ""])
    failures = [row for row in records if not row.get("contract", {}).get("valid") or not row.get("analysis", {}).get("complex_chain")]
    if not failures:
        lines.append("None in this pilot.")
    for row in failures[:8]:
        analysis = row.get("analysis") or {}
        reasons = []
        if not row.get("contract", {}).get("valid"): reasons.append("invalid_contract")
        if analysis.get("simple_completion_chain"): reasons.append("simple_completion_chain")
        if analysis.get("premature_closure"): reasons.append("premature_closure")
        if analysis and not analysis.get("role_requirement_met"): reasons.append("missing_reconsideration_or_revision")
        if analysis and not analysis.get("all_post_initial_changes_trigger_backed"): reasons.append("missing_trigger")
        lines.append(f"- `{row.get('case_id')}:{row.get('setting')}`: {', '.join(reasons) or 'complexity_requirements_not_met'}")
    conclusion = metrics["conclusion"]
    lines.extend([
        "", "## Conclusion", "",
        f"Classification: **{conclusion['code']} — {conclusion['label']}**", "",
        conclusion["reason"], "",
        "This pilot diagnoses Planner plan capacity only. It does not establish that Realizer can faithfully realize a complex plan.",
    ])
    return "\n".join(lines) + "\n"


def run_probe(*, run_root: Path, config: Path | None, live: bool, max_events: int,
              replay_from: Path | None = None) -> dict[str, Any]:
    prepare_empty(run_root)
    data = load_frozen_inputs(DEFAULT_RUN)
    all_events = list((data.get("memory_overview") or {}).get("events") or [])
    events = _select_events(all_events, max_events)
    cases = [_case(index, event) for index, event in enumerate(events, 1)]
    parent = parent_view(data["dialogue"])
    if replay_from is not None and live:
        raise ValueError("replay_from_and_live_are_mutually_exclusive")
    client = LiveJSONClient(config, max_tokens=5000, temperature=0.0) if live and config else None
    records: list[dict[str, Any]] = []
    if replay_from is not None:
        source_records = read_jsonl(replay_from / "planner_cases.jsonl")
        if not source_records:
            raise ValueError(f"replay_source_planner_cases_missing:{replay_from}")
        cases_by_id: dict[str, dict[str, Any]] = {}
        for source in source_records:
            key = str(source.get("setting") or "")
            if key not in SETTINGS:
                continue
            setting = SETTINGS[key]
            case_id = str(source.get("case_id") or "")
            value = source.get("plan") if isinstance(source.get("plan"), Mapping) else {}
            contract = validate_plan(value, setting=setting, case_id=case_id)
            analysis = analyze_chain(value) if contract["valid"] else {}
            case = {"case_id": case_id, "event_key": source.get("event_key"),
                    "event_content": source.get("event_content") or []}
            cases_by_id[case_id] = case
            records.append({**source, **case, "setting": key, "contract": contract,
                            "analysis": analysis, "replayed_from": str(replay_from)})
        cases = [cases_by_id[key] for key in sorted(cases_by_id)]
    else:
        for case in cases:
          for setting in SETTINGS.values():
            api_meta: dict[str, Any] = {"status": "not_run_no_live", "attempts": []}
            value: Any = {}
            if client is not None:
                try:
                    raw, first_meta = client.complete(_prompt(case, parent, setting))
                    value = extract_json(raw)
                    api_meta = {"status": "pass", "attempts": [first_meta]}
                except Exception as exc:
                    api_meta = {"status": "failed", "attempts": [], "error": f"{type(exc).__name__}:{exc}"}
            contract = validate_plan(value, setting=setting, case_id=str(case["case_id"]))
            if client is not None and not contract["valid"] and isinstance(value, Mapping):
                try:
                    raw, repair_meta = client.complete(_repair_prompt(
                        case, parent, setting, value, list(contract["errors"])))
                    repaired = extract_json(raw)
                    repaired_contract = validate_plan(
                        repaired, setting=setting, case_id=str(case["case_id"]))
                    api_meta["attempts"].append(repair_meta)
                    api_meta["status"] = "pass_repaired" if repaired_contract["valid"] else "repair_invalid"
                    value, contract = repaired, repaired_contract
                except Exception as exc:
                    api_meta["status"] = "repair_failed"
                    api_meta["repair_error"] = f"{type(exc).__name__}:{exc}"
            analysis = analyze_chain(value) if contract["valid"] else {}
            records.append({
                **case, "setting": setting.key,
                "capacity": {"scene_count": setting.scene_count, "min_beats": setting.min_beats, "max_beats": setting.max_beats},
                "api": api_meta, "plan": value, "contract": contract, "analysis": analysis,
            })
    by_setting = {key: summarize([row for row in records if row["setting"] == key]) for key in SETTINGS}
    paired = _paired_comparison(records)
    generation_planner_calls = (
        sum(len((row.get("api") or {}).get("attempts") or []) for row in records)
        if replay_from is not None else (client.calls if client else 0))
    metrics = {
        "probe_version": PROBE_VERSION,
        "research_question": "Does the current scene budget/granularity prevent the generation of complex state evolution?",
        "dataset": {
            "number_of_roots": 1 if cases else 0, "number_of_events": len(cases),
            "number_of_planner_cases": len(records),
            "event_selection_rule": EVENT_SELECTION_RULE,
            "scenes_per_setting": {key: setting.scene_count for key, setting in SETTINGS.items()},
            "historical_reference": _historical_reference(),
        },
        "by_setting": by_setting, "paired_comparison": paired, "conclusion": _conclusion(paired),
        "api_calls": generation_planner_calls, "planner_calls": generation_planner_calls,
        "analysis_execution_api_calls": client.calls if client else 0,
        "replay_source": str(replay_from) if replay_from is not None else None,
        "realizer_calls": 0, "solver_calls": 0, "writer_calls": 0,
        "memorax_search_calls": 0, "memorax_add_calls": 0,
        "limitations": [
            "Planner-only; Realizer fidelity is not tested.",
            "Small paired diagnostic; batch stability is not established.",
            "All settings use the same model and temperature=0, but the provider API exposes no explicit sampling seed.",
        ],
    }
    write_jsonl(run_root / "inputs.jsonl", [{"frozen_inputs": data["inputs"], "cases": cases,
        "event_selection_rule": EVENT_SELECTION_RULE,
        "settings": {key: setting.__dict__ for key, setting in SETTINGS.items()}}])
    write_jsonl(run_root / "planner_cases.jsonl", records)
    write_jsonl(run_root / "audits.jsonl", [{"case_id": row["case_id"], "setting": row["setting"], "contract": row["contract"], "analysis": row["analysis"]} for row in records])
    write_json(run_root / "metrics.json", metrics)
    write_json(run_root / "manifest.json", {
        "probe_version": PROBE_VERSION, "isolated": True, "historical_runs_read_only": True,
        "api_role_allowlist": ["planner"], "planner_calls": generation_planner_calls,
        "analysis_execution_api_calls": client.calls if client else 0,
        "replay_source": str(replay_from) if replay_from is not None else None,
        "realizer_calls": 0, "solver_calls": 0, "writer_calls": 0,
        "memorax_search_calls": 0, "memorax_add_calls": 0,
        "outputs": ["inputs.jsonl", "planner_cases.jsonl", "audits.jsonl", "metrics.json", "report.md", "manifest.json"],
    })
    (run_root / "report.md").write_text(_report(metrics, records), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="isolated Planner-only scene complexity diagnostic")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--config", type=Path, default=Path("/root/autodl-tmp/gxy-selfplay/config.json"))
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--max-events", type=int, default=2)
    parser.add_argument("--replay-from", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = (args.run_root or RUN_BASE / f"scene_complexity_probe_v1_{stamp}").resolve()
    metrics = run_probe(run_root=run_root, config=args.config, live=args.live,
                        max_events=max(1, args.max_events), replay_from=args.replay_from)
    print(json.dumps({"run_root": str(run_root), "metrics": metrics}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
