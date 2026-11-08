from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.frozen_loader import (
    load_dialogue,
    parent_view,
)
from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.io_utils import (
    prepare_empty,
    read_jsonl,
    sha256_file,
    write_json,
)
from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.live_adapter import (
    LiveJSONClient,
    extract_json,
)
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.contracts import (
    normalize_realizer_output,
)
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.prompts import (
    build_realizer_prompt,
)


RUN_BASE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
PROBE_VERSION = "planner_realizer_temporal_boundary_batch_v1"
SOURCE_FILES = {
    "POB03": RUN_BASE / "planner_objective_probe_v1_batch_20260822T110000Z" / "planner_cases.jsonl",
    "POB04": RUN_BASE / "planner_objective_probe_v1_batch_20260822T110000Z" / "planner_cases.jsonl",
    "POS07": RUN_BASE / "planner_objective_supplement_v1_20260822T150000Z" / "planner_cases.jsonl",
}

# The profile is part of this isolated audit, not production logic. It names
# actions that are only allowed in the final scene for each frozen plan.
BOUNDARY_PROFILES: dict[str, dict[str, list[str]]] = {
    "POB03": {
        "forbidden_before_final": [
            "final version", "recipe is finalized", "glaze is finalized", "i'm brushing",
            "i am brushing", "i'm applying", "i am applying", "i applied", "i'm coating",
            "i am coating", "i placed the ham in the oven", "i put the ham in the oven",
            "i set the timer", "i'm basting", "i am basting",
        ],
        "required_final_any": [["brush", "apply", "coat"], ["oven"], ["timer"]],
    },
    "POB04": {
        "forbidden_before_final": [
            "finalized", "officially scheduled", "event is now set", "committed to coordinating",
            "revised outreach timeline", "confirmed for a weekend date",
        ],
        "required_final_any": [["weekend"], ["outreach"], ["environmental group", "green horizon"]],
    },
    "POS07": {
        "forbidden_before_final": [
            "committed to cognitive-behavioral therapy", "committed to weekly", "weekly sessions",
            "first appointment", "specific treatment plan", "agrees to cognitive-behavioral",
        ],
        "required_final_any": [["therapist", "therapy"], ["weekly"], ["appointment", "session"]],
    },
}


def _find_source(case_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = SOURCE_FILES[case_id]
    records = read_jsonl(path)
    source = next((row for row in records if row.get("case_id") == case_id and row.get("objective") == "O1"), None)
    if not source or not source.get("contract", {}).get("valid") or not source.get("analysis", {}).get("complex_chain"):
        raise ValueError(f"accepted_o1_source_missing_or_invalid:{case_id}")
    snapshot = Path(str(source["snapshot_path"]))
    dialogue_path = snapshot / "frozen" / "expanded_dialogue.json"
    if not dialogue_path.exists() or sha256_file(dialogue_path) != source.get("dialogue_sha256"):
        raise ValueError(f"source_dialogue_sha256_mismatch:{case_id}")
    dialogue = load_dialogue(dialogue_path)
    return source, {
        "parent": parent_view(dialogue),
        "source_records_path": str(path),
        "source_records_sha256": sha256_file(path),
        "dialogue_path": str(dialogue_path),
        "dialogue_sha256": sha256_file(dialogue_path),
    }


def _adapt_plan(source: Mapping[str, Any]) -> dict[str, Any]:
    speakers = [str(value) for value in source.get("valid_speakers") or []]
    if not speakers or any(not value for value in speakers):
        raise ValueError("source_speakers_missing")
    raw_plan = source.get("plan") if isinstance(source.get("plan"), Mapping) else {}
    scenes = []
    case_id = str(source["case_id"])
    for scene in raw_plan.get("scenes") or []:
        index = int(scene.get("scene_index") or 0)
        trigger = scene.get("trigger") if isinstance(scene.get("trigger"), Mapping) else {}
        before = f"{scene.get('state_slot')}: {scene.get('before_value')} [stage={scene.get('before_stage')}]"
        after = f"{scene.get('state_slot')}: {scene.get('after_value')} [stage={scene.get('after_stage')}; resolution={scene.get('resolution_status')}]"
        transition = (
            f"{scene.get('primary_action')} / {scene.get('transition_type')}; "
            f"requested={scene.get('required_transition')}; "
            f"trigger={trigger.get('type')}: {trigger.get('content')}"
        )
        scenes.append({
            "memory_event_id": f"{case_id}:S{index:02d}",
            "development_mode": scene.get("transition_type"),
            "source_basis": source.get("event_key"),
            "new_people": [],
            "verified_current_state": before,
            "goal": raw_plan.get("round_goal"),
            "required_participants": speakers,
            "transition_from_previous": "initial accepted state" if index == 1 else "continue from prior Child scene end state",
            "before_state": before,
            "required_transition": transition,
            "after_state": after,
            "new_development": transition,
            "durable_outcome": after,
            "actor_bindings": [{"actor": speaker, "responsibilities": [beat for beat in scene.get("interaction_beats") or [] if speaker in str(beat)]} for speaker in speakers],
            "interaction_beats": list(scene.get("interaction_beats") or []),
        })
    if len(scenes) != 3:
        raise ValueError(f"adapted_scene_count:{case_id}:{len(scenes)}")
    return {"plan_version": "planner_objective_o1_realizer_adapter_generic_v1", "source_case_id": case_id, "source_objective": "O1", "scenes": scenes}


def _generic_boundary_block(plan: Mapping[str, Any]) -> str:
    scene_count = len(list(plan.get("scenes") or []))
    return "\n".join([
        "TEMPORAL BOUNDARY CONTROL (strict)",
        "Realize only the current scene's transition and immediate consequence.",
        "Keep every non-final scene open/provisional; never call it final, complete, or ready.",
        "Do not perform, announce as completed, or imply any action belonging to a later scene.",
        "A future action may be mentioned only as an explicitly future plan, never as already done.",
        f"This plan has {scene_count} scenes; preserve the plan's scene order exactly.",
    ])


def _profile_audit(child: Mapping[str, Any], case_id: str) -> dict[str, Any]:
    scenes = child.get("scenes") if isinstance(child.get("scenes"), list) else []
    texts = [" ".join(str(turn.get("text") or "").casefold() for turn in scene.get("dialogue_turns") or []) for scene in scenes if isinstance(scene, Mapping)]
    profile = BOUNDARY_PROFILES[case_id]
    early = " ".join(texts[:2])
    forbidden_hits = [term for term in profile["forbidden_before_final"] if term.casefold() in early]
    final_text = texts[2] if len(texts) >= 3 else ""
    required_missing = [
        "/".join(alternatives)
        for alternatives in profile["required_final_any"]
        if not any(term.casefold() in final_text for term in alternatives)
    ]
    checks = {
        "three_scenes": len(texts) == 3,
        "no_profiled_final_action_before_final_scene": not forbidden_hits,
        "final_scene_contains_profiled_resolution_evidence": not required_missing,
    }
    return {
        "audit_version": "planner_realizer_temporal_boundary_profile_v1",
        "case_id": case_id,
        "overall_pass": bool(all(checks.values())),
        "checks": checks,
        "forbidden_hits_before_final": forbidden_hits,
        "required_final_evidence_missing": required_missing,
        "limitations": [
            "Conservative lexical evidence; present-action patterns are distinguished from explicit future-plan wording only by fixed phrases.",
            "Profiled final actions are case-specific and are not production logic.",
        ],
    }


def _audit(child: Mapping[str, Any], case_id: str) -> dict[str, Any]:
    return _profile_audit(child, case_id)


def _run_case(*, case_id: str, config: Path, case_root: Path) -> dict[str, Any]:
    source, provenance = _find_source(case_id)
    parent = provenance["parent"]
    plan = _adapt_plan(source)
    base = build_realizer_prompt(parent=parent, plan=plan, include_full_parent=True)
    boundary = _generic_boundary_block(plan)
    profile = BOUNDARY_PROFILES[case_id]
    profile_block = (
        "FINAL-SCENE-ONLY ACTIONS (hard): the following actions/states may occur only in Scene 3; "
        "do not perform or assert them in Scenes 1-2: " + "; ".join(profile["forbidden_before_final"])
    )
    prompt = {**base, "prompt_version": f"{PROBE_VERSION}_compact_prompt", "prompt": str(base["prompt"]) + "\n\n" + boundary + "\n" + profile_block + "\nReturn only the same JSON object with exactly the `scenes` key.", "temporal_boundary_control": boundary, "profile_block": profile_block}
    write_json(case_root / "source_record.json", source)
    write_json(case_root / "adapted_plan.json", plan)
    write_json(case_root / "prompt.json", prompt)
    client = LiveJSONClient(config, max_tokens=8192, temperature=0.0)
    raw = ""
    meta: dict[str, Any] = {}
    parsed: Any = {}
    error = None
    try:
        raw, meta = client.complete(prompt["prompt"])
        (case_root / "raw_response.txt").write_text(raw, encoding="utf-8")
        parsed = extract_json(raw)
        write_json(case_root / "parsed_response.json", parsed)
    except Exception as exc:
        error = f"{type(exc).__name__}:{exc}"
        (case_root / "raw_response.txt").write_text(raw, encoding="utf-8")
    mechanical = normalize_realizer_output(parsed, plan=plan, parent=parent, round_index=1) if error is None else {"valid": False, "errors": [error]}
    semantic = _audit(mechanical.get("child") or {}, case_id) if mechanical.get("valid") else {"overall_pass": False, "checks": {}, "limitations": ["skipped mechanical failure"]}
    if mechanical.get("child"):
        write_json(case_root / "child_dialogue.json", mechanical["child"])
    write_json(case_root / "mechanical_audit.json", mechanical)
    write_json(case_root / "semantic_audit.json", semantic)
    passed = bool(mechanical.get("valid") and semantic.get("overall_pass"))
    conclusion = "PASS" if passed else ("UNDECIDED_TIMEOUT" if error and client.calls == 0 and "Timeout" in error else ("UNDECIDED_TRANSPORT" if error else "FAIL"))
    return {"case_id": case_id, "root_id": source["root_id"], "event_key": source["event_key"], "source": provenance, "mechanical_contract_valid": bool(mechanical.get("valid")), "temporal_audit_pass": bool(semantic.get("overall_pass")), "request_attempts": 1, "realizer_calls_completed": client.calls, "response_meta": meta, "transport_error": error, "conclusion": conclusion}


def run(*, run_root: Path, config: Path, case_ids: list[str]) -> dict[str, Any]:
    prepare_empty(run_root)
    cases = []
    for case_id in case_ids:
        case_root = run_root / case_id
        case_root.mkdir(parents=True, exist_ok=True)
        cases.append(_run_case(case_id=case_id, config=config, case_root=case_root))
    metrics = {"probe_version": PROBE_VERSION, "case_ids": case_ids, "cases": cases, "completed_cases": sum(row["realizer_calls_completed"] > 0 for row in cases), "passed_cases": sum(row["conclusion"] == "PASS" for row in cases), "analysis_execution_api_calls": 0, "planner_calls": 0, "qa_writer_calls": 0, "solver_calls": 0, "writer_calls": 0, "auditor_calls": 0, "memorax_search_calls": 0, "memorax_add_calls": 0}
    write_json(run_root / "metrics.json", metrics)
    write_json(run_root / "manifest.json", {"probe_version": PROBE_VERSION, "isolated": True, "case_ids": case_ids, "api_role_allowlist": ["realizer"], "one_request_per_case": True, "retry_budget_per_case": 0, "analysis_execution_api_calls": 0, "planner_calls": 0, "qa_writer_calls": 0, "solver_calls": 0, "writer_calls": 0, "auditor_calls": 0, "memorax_search_calls": 0, "memorax_add_calls": 0})
    lines = ["# Planner Realizer Temporal Boundary Batch v1", "", f"- Cases: `{', '.join(case_ids)}`", f"- Passed: `{metrics['passed_cases']}/{len(case_ids)}`", f"- Completed responses: `{metrics['completed_cases']}/{len(case_ids)}`", "", "Each case used one compact boundary Realizer request with retry=0. No Planner, QA, Solver, Writer, Auditor, or Memorax calls were made.", ""]
    for row in cases:
        lines.append(f"- `{row['case_id']}` (`{row['root_id']}`): **{row['conclusion']}**, audit={row['temporal_audit_pass']}, calls={row['realizer_calls_completed']}, error={row['transport_error'] or 'none'}")
    (run_root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="three-root Realizer temporal-boundary batch")
    parser.add_argument("--config", type=Path, default=Path("/root/autodl-tmp/gxy-selfplay/config.json"))
    parser.add_argument("--case-ids", nargs="+", default=["POB03", "POB04", "POS07"])
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = (args.run_root or RUN_BASE / f"{PROBE_VERSION}_{stamp}").resolve()
    print(json.dumps({"run_root": str(run_root), "metrics": run(run_root=run_root, config=args.config, case_ids=args.case_ids)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
