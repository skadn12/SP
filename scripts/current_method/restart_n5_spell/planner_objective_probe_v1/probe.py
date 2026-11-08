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
    prepare_empty, read_jsonl, write_json, write_jsonl,
)
from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.live_adapter import (
    LiveJSONClient, extract_json,
)
from scripts.current_method.restart_n5_spell.scene_complexity_probe_v1.analyzer import (
    analyze_chain, summarize,
)
from .contracts import (FIXED_MAX_BEATS, FIXED_MIN_BEATS, FIXED_SCENE_COUNT,
                        OBJECTIVES, Objective, validate_objective_plan)


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
PROBE_VERSION = "planner_objective_probe_v1"
EVENT_SELECTION_RULE = "prefer_transition_affording_event_text_then_source_order_v1"
AFFORDANCE_TERMS = ("plan", "consider", "decid", "choos", "agree",
                    "propos", "organize", "arrang", "responsib")


def _select_events(events: list[Mapping[str, Any]], max_events: int) -> list[Mapping[str, Any]]:
    def score(row: Mapping[str, Any]) -> int:
        text = " ".join([str(row.get("canonical_key") or ""),
                         *[str(value) for value in (row.get("content") or [])]]).casefold()
        return sum(term in text for term in AFFORDANCE_TERMS)
    ranked = sorted(enumerate(events), key=lambda item: (-score(item[1]), item[0]))
    return [row for _, row in ranked[:max_events]]


def _cases(data: Mapping[str, Any], max_events: int) -> list[dict[str, Any]]:
    events = list((data.get("memory_overview") or {}).get("events") or [])
    return [{
        "case_id": f"PO{index:02d}",
        "event_key": str(event.get("canonical_key") or event.get("display_key") or f"event-{index}"),
        "event_content": [str(value) for value in (event.get("content") or [])],
    } for index, event in enumerate(_select_events(events, max_events), 1)]


def _output_schema(case_id: str, objective_key: str) -> str:
    return f"""{{
  "case_id": "{case_id}",
  "setting": "{objective_key}",
  "round_goal": "...",
  "scenes": [
    {{
      "scene_index": 1,
      "primary_action": "proposal|challenge|tradeoff|reconsideration|revision|resolution|event_progress|other",
      "state_slot": "same concrete state slot in every scene",
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


def _prompt(case: Mapping[str, Any], parent: Mapping[str, Any], objective: Objective) -> str:
    speakers = sorted({str(row.get("speaker") or "") for row in parent["dialogue"]["turns"] if str(row.get("speaker") or "")})
    event = {"event_key": case["event_key"], "event_content": case["event_content"],
             "canonical_speakers": speakers}
    return f"""You are the Planner in an isolated fixed-capacity objective diagnostic. Do not write dialogue and do not add QA facts. Plan one coherent continuation of the frozen event. O0 and O1 use identical capacity, schema, model, and decoding parameters; only the objective instruction differs.

FIXED CAPACITY
- exactly {FIXED_SCENE_COUNT} scenes
- {FIXED_MIN_BEATS} to {FIXED_MAX_BEATS} interaction beats per scene

OBJECTIVE {objective.key}: {objective.label}
{objective.instruction}

COMMON TRUTHFULNESS RULES
Track exactly one concrete state slot. Keep value change, stage change, and ordinary event progress distinct. Every meaningful change after the initial proposal must have a concrete trigger such as a constraint, new requirement, new information, external event, or stakeholder objection. "Changed mind" alone is not a trigger. Never mark a final resolution before the last scene. Do not invent false prior history, arbitrary reversals, unrelated distractors, or non-canonical speakers. If the requested structure is unsupported, prefer the simplest truthful chain; do not fake objective adherence.
Use only the exact names in canonical_speakers. Do not invent Classmates C or any other role. `event_progress` is legal only as transition_type/primary_action; it is NEVER a legal trigger.type. If confirmation or availability is newly learned, use trigger.type=`new_information`.

FROZEN EVENT INPUT
{json.dumps(event, ensure_ascii=False, indent=2)}

This is a probe-local diagnostic annotation, not a production Planner schema or persistent state system. Return only JSON with no markdown:
{_output_schema(str(case['case_id']), objective.key)}"""


def _repair_prompt(case: Mapping[str, Any], parent: Mapping[str, Any], objective: Objective,
                   plan: Mapping[str, Any], errors: list[str]) -> str:
    return _prompt(case, parent, objective) + f"""

REPAIR ONLY
The previous plan failed deterministic contract checks:
{json.dumps(errors, ensure_ascii=False)}

Previous plan:
{json.dumps(dict(plan), ensure_ascii=False, indent=2)}

Return one complete corrected object. Fix only scene/beat count, enums, and adjacent-state continuity. Preserve the same event, state slot, and substantive choices. Do not add reconsideration, revision, resolution, or triggers merely to improve the complexity score."""


def _paired_comparison(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in records:
        by_case.setdefault(str(row.get("case_id") or ""), {})[str(row.get("objective") or "")] = row
    complete = [case_id for case_id, rows in by_case.items()
                if all(rows.get(key, {}).get("contract", {}).get("valid") for key in OBJECTIVES)]
    outcomes = []
    for case_id in complete:
        o0 = bool(by_case[case_id]["O0"].get("analysis", {}).get("complex_chain"))
        o1 = bool(by_case[case_id]["O1"].get("analysis", {}).get("complex_chain"))
        outcomes.append({"case_id": case_id, "O0_complex": o0, "O1_complex": o1,
                         "objective_gain": int(o1) - int(o0)})
    def rate(key: str) -> float | None:
        values = [float(row[f"{key}_complex"]) for row in outcomes]
        return sum(values) / len(values) if values else None
    return {
        "paired_complete_cases": len(complete), "paired_case_ids": complete,
        "excluded_case_ids": sorted(set(by_case) - set(complete)),
        "O0_complex_chain_rate": rate("O0"),
        "O1_complex_chain_rate": rate("O1"),
        "mean_objective_gain": (sum(row["objective_gain"] for row in outcomes) / len(outcomes)) if outcomes else None,
        "pair_outcomes": outcomes,
    }


def _conclusion(paired: Mapping[str, Any]) -> dict[str, Any]:
    if not paired.get("paired_complete_cases"):
        return {"label": "INSUFFICIENT_DATA", "reason": "No event has valid O0 and O1 Planner outputs."}
    o0 = float(paired["O0_complex_chain_rate"] or 0.0)
    o1 = float(paired["O1_complex_chain_rate"] or 0.0)
    if o1 > o0:
        if int(paired["paired_complete_cases"]) < 3:
            return {"label": "OBJECTIVE_CONTROL_PILOT_SIGNAL",
                    "reason": f"Across only {paired['paired_complete_cases']} complete pair(s), fixed-capacity explicit state-evolution instructions raised complex-chain rate from {o0:.3f} to {o1:.3f}; this is directional evidence, not batch validation."}
        return {"label": "OBJECTIVE_CONTROL_SUPPORTED",
                "reason": f"At fixed capacity, explicit state-evolution instructions raised complex-chain rate from {o0:.3f} to {o1:.3f}."}
    if o1 == o0 == 0.0:
        return {"label": "OBJECTIVE_PROMPT_INSUFFICIENT",
                "reason": "Neither objective produced a valid complex chain at fixed capacity."}
    if o1 == o0:
        return {"label": "NO_ATTRIBUTABLE_OBJECTIVE_GAIN",
                "reason": f"Both objectives produced the same complex-chain rate ({o0:.3f})."}
    return {"label": "OBJECTIVE_CONTROL_HARMFUL_OR_UNSTABLE",
            "reason": f"Explicit control reduced complex-chain rate from {o0:.3f} to {o1:.3f}."}


def _objective_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    result = summarize(rows)
    valid = [row.get("analysis") or {} for row in rows if row.get("contract", {}).get("valid")]
    result["final_at_last_rate"] = (sum(bool(row.get("final_resolution_at_last_scene")) for row in valid) / len(valid)) if valid else None
    return result


def _report(metrics: Mapping[str, Any], records: list[Mapping[str, Any]]) -> str:
    def render(value: Any) -> str:
        return "n/a" if value is None else f"{float(value):.3f}"
    lines = [
        "# Planner Objective Probe v1", "",
        "Research question: At fixed scene capacity, does an explicit state-evolution objective reduce the Planner's rapid-closure bias?", "",
        "Capacity is fixed at 3 scenes and 3–7 interaction beats for both objectives. This is Planner-only.", "",
        "## Dataset", "",
        f"- Roots: {metrics['dataset']['number_of_roots']}",
        f"- Frozen events: {metrics['dataset']['number_of_events']}",
        f"- Planner cases: {metrics['dataset']['number_of_planner_cases']}",
        f"- Fully valid O0/O1 pairs: {metrics['paired_comparison']['paired_complete_cases']}", "",
        "## Objective comparison", "",
        "| Objective | Valid | Complex-chain rate | Final-at-last rate | Trigger rate | Reconsideration | Revision |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in OBJECTIVES:
        row = metrics["by_objective"][key]
        lines.append(
            f"| {key} | {row['valid_planner_cases']}/{row['planner_cases']} | {render(row['complex_chain_rate'])} | "
            f"{render(row['final_at_last_rate'])} | {render(row['trigger_backed_transition_rate'])} | "
            f"{render(row['reconsideration_rate'])} | {render(row['revision_rate'])} |")
    paired = metrics["paired_comparison"]
    lines.extend([
        "", "## Strict paired endpoint", "",
        f"- O0 complex-chain rate: {render(paired['O0_complex_chain_rate'])}",
        f"- O1 complex-chain rate: {render(paired['O1_complex_chain_rate'])}",
        f"- Mean objective gain: {render(paired['mean_objective_gain'])}", "",
        "## Failure examples", "",
    ])
    failures = [row for row in records if not row.get("contract", {}).get("valid") or not row.get("analysis", {}).get("complex_chain")]
    if not failures:
        lines.append("None in this pilot.")
    for row in failures[:8]:
        analysis = row.get("analysis") or {}
        reasons = []
        if not row.get("contract", {}).get("valid"): reasons.append("invalid_contract")
        if analysis and not analysis.get("role_requirement_met"): reasons.append("missing_reconsideration_or_revision")
        if analysis and not analysis.get("final_resolution_at_last_scene"): reasons.append("missing_final_resolution")
        if analysis and not analysis.get("all_post_initial_changes_trigger_backed"): reasons.append("missing_trigger")
        if analysis.get("premature_closure"): reasons.append("premature_closure")
        lines.append(f"- `{row.get('case_id')}:{row.get('objective')}`: {', '.join(reasons) or 'complexity_requirements_not_met'}")
    conclusion = metrics["conclusion"]
    lines.extend([
        "", "## Conclusion", "", f"Verdict: **{conclusion['label']}**", "",
        conclusion["reason"], "",
        "This pilot measures plan structure only; Realizer fidelity and batch stability remain untested.",
    ])
    return "\n".join(lines) + "\n"


def run_probe(*, run_root: Path, config: Path | None, live: bool, max_events: int,
              replay_from: Path | None = None) -> dict[str, Any]:
    prepare_empty(run_root)
    data = load_frozen_inputs(DEFAULT_RUN)
    cases = _cases(data, max_events)
    parent = parent_view(data["dialogue"])
    if replay_from is not None and live:
        raise ValueError("replay_from_and_live_are_mutually_exclusive")
    client = LiveJSONClient(config, max_tokens=5000, temperature=0.0) if live and config else None
    records: list[dict[str, Any]] = []
    if replay_from is not None:
        sources = read_jsonl(replay_from / "planner_cases.jsonl")
        if not sources:
            raise ValueError(f"replay_source_planner_cases_missing:{replay_from}")
        cases_by_id: dict[str, dict[str, Any]] = {}
        for source in sources:
            key = str(source.get("objective") or "")
            if key not in OBJECTIVES:
                continue
            case_id = str(source.get("case_id") or "")
            value = source.get("plan") if isinstance(source.get("plan"), Mapping) else {}
            valid_speakers = sorted({str(row.get("speaker") or "") for row in parent["dialogue"]["turns"] if str(row.get("speaker") or "")})
            contract = validate_objective_plan(value, objective_key=key, case_id=case_id,
                                               valid_speakers=valid_speakers)
            analysis = analyze_chain(value) if contract["valid"] else {}
            case = {"case_id": case_id, "event_key": source.get("event_key"),
                    "event_content": source.get("event_content") or []}
            cases_by_id[case_id] = case
            records.append({**source, **case, "objective": key, "contract": contract,
                            "analysis": analysis, "replayed_from": str(replay_from)})
        cases = [cases_by_id[key] for key in sorted(cases_by_id)]
    else:
        for case in cases:
            for key, objective in OBJECTIVES.items():
                api_meta: dict[str, Any] = {"status": "not_run_no_live", "attempts": []}
                value: Any = {}
                if client is not None:
                    try:
                        raw, first_meta = client.complete(_prompt(case, parent, objective))
                        value = extract_json(raw)
                        api_meta = {"status": "pass", "attempts": [first_meta]}
                    except Exception as exc:
                        api_meta = {"status": "failed", "attempts": [],
                                    "error": f"{type(exc).__name__}:{exc}"}
                valid_speakers = sorted({str(row.get("speaker") or "") for row in parent["dialogue"]["turns"] if str(row.get("speaker") or "")})
                contract = validate_objective_plan(
                    value, objective_key=key, case_id=str(case["case_id"]),
                    valid_speakers=valid_speakers)
                if client is not None and not contract["valid"] and isinstance(value, Mapping):
                    try:
                        raw, repair_meta = client.complete(_repair_prompt(
                            case, parent, objective, value, list(contract["errors"])))
                        repaired = extract_json(raw)
                        repaired_contract = validate_objective_plan(
                            repaired, objective_key=key, case_id=str(case["case_id"]),
                            valid_speakers=valid_speakers)
                        api_meta["attempts"].append(repair_meta)
                        api_meta["status"] = "pass_repaired" if repaired_contract["valid"] else "repair_invalid"
                        value, contract = repaired, repaired_contract
                    except Exception as exc:
                        api_meta["status"] = "repair_failed"
                        api_meta["repair_error"] = f"{type(exc).__name__}:{exc}"
                analysis = analyze_chain(value) if contract["valid"] else {}
                records.append({
                    **case, "objective": key,
                    "capacity": {"scene_count": FIXED_SCENE_COUNT,
                                 "min_beats": FIXED_MIN_BEATS, "max_beats": FIXED_MAX_BEATS},
                    "api": api_meta, "plan": value, "contract": contract, "analysis": analysis,
                })
    by_objective = {key: _objective_summary([row for row in records if row["objective"] == key])
                    for key in OBJECTIVES}
    paired = _paired_comparison(records)
    generation_calls = (
        sum(len((row.get("api") or {}).get("attempts") or []) for row in records)
        if replay_from is not None else (client.calls if client else 0))
    metrics = {
        "probe_version": PROBE_VERSION,
        "research_question": "At fixed scene capacity, does an explicit state-evolution objective reduce rapid closure?",
        "dataset": {"number_of_roots": 1 if cases else 0, "number_of_events": len(cases),
                    "number_of_planner_cases": len(records), "event_selection_rule": EVENT_SELECTION_RULE},
        "fixed_capacity": {"scene_count": FIXED_SCENE_COUNT, "min_beats": FIXED_MIN_BEATS,
                           "max_beats": FIXED_MAX_BEATS},
        "by_objective": by_objective, "paired_comparison": paired, "conclusion": _conclusion(paired),
        "api_calls": generation_calls, "planner_calls": generation_calls,
        "analysis_execution_api_calls": client.calls if client else 0,
        "replay_source": str(replay_from) if replay_from else None,
        "realizer_calls": 0, "solver_calls": 0, "writer_calls": 0,
        "memorax_search_calls": 0, "memorax_add_calls": 0,
        "limitations": [
            "Planner-only; Realizer fidelity is not tested.",
            "Small paired diagnostic; batch stability is not established.",
            "Same model and temperature=0 are used, but the provider exposes no explicit sampling seed.",
        ],
    }
    write_jsonl(run_root / "inputs.jsonl", [{"frozen_inputs": data["inputs"], "cases": cases,
        "objectives": {key: objective.__dict__ for key, objective in OBJECTIVES.items()},
        "fixed_capacity": metrics["fixed_capacity"]}])
    write_jsonl(run_root / "planner_cases.jsonl", records)
    write_jsonl(run_root / "audits.jsonl", [{"case_id": row["case_id"],
        "objective": row["objective"], "contract": row["contract"],
        "analysis": row["analysis"]} for row in records])
    write_json(run_root / "metrics.json", metrics)
    write_json(run_root / "manifest.json", {
        "probe_version": PROBE_VERSION, "isolated": True, "historical_runs_read_only": True,
        "api_role_allowlist": ["planner"], "planner_calls": generation_calls,
        "analysis_execution_api_calls": client.calls if client else 0,
        "replay_source": str(replay_from) if replay_from else None,
        "realizer_calls": 0, "solver_calls": 0, "writer_calls": 0,
        "memorax_search_calls": 0, "memorax_add_calls": 0,
        "outputs": ["inputs.jsonl", "planner_cases.jsonl", "audits.jsonl",
                    "metrics.json", "report.md", "manifest.json"],
    })
    (run_root / "report.md").write_text(_report(metrics, records), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="isolated fixed-capacity Planner objective probe")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--config", type=Path,
                        default=Path("/root/autodl-tmp/gxy-selfplay/config.json"))
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--max-events", type=int, default=2)
    parser.add_argument("--replay-from", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = (args.run_root or RUN_BASE / f"planner_objective_probe_v1_{stamp}").resolve()
    metrics = run_probe(run_root=run_root, config=args.config, live=args.live,
                        max_events=max(1, args.max_events), replay_from=args.replay_from)
    print(json.dumps({"run_root": str(run_root), "metrics": metrics}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
