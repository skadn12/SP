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
SOURCE_BATCH = RUN_BASE / "planner_objective_probe_v1_batch_20260822T110000Z"
CASE_ID = "POB03"
OBJECTIVE = "O1"
PROBE_VERSION = "planner_realizer_adherence_canary_v1"


def _load_source() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    records_path = SOURCE_BATCH / "planner_cases.jsonl"
    records = read_jsonl(records_path)
    source = next((row for row in records
                   if row.get("case_id") == CASE_ID and row.get("objective") == OBJECTIVE), None)
    if not source:
        raise ValueError("accepted_o1_source_missing")
    if not source.get("contract", {}).get("valid") or not source.get("analysis", {}).get("complex_chain"):
        raise ValueError("source_is_not_accepted_complex_o1")
    snapshot = Path(str(source.get("snapshot_path") or ""))
    dialogue_path = snapshot / "frozen" / "expanded_dialogue.json"
    if not dialogue_path.exists():
        raise FileNotFoundError(f"source_dialogue_missing:{dialogue_path}")
    if sha256_file(dialogue_path) != source.get("dialogue_sha256"):
        raise ValueError("source_dialogue_sha256_mismatch")
    dialogue = load_dialogue(dialogue_path)
    parent = parent_view(dialogue)
    return source, parent, {
        "source_records_path": str(records_path),
        "source_records_sha256": sha256_file(records_path),
        "dialogue_path": str(dialogue_path),
        "dialogue_sha256": sha256_file(dialogue_path),
    }


def adapt_plan(source: Mapping[str, Any]) -> dict[str, Any]:
    raw_plan = source.get("plan") if isinstance(source.get("plan"), Mapping) else {}
    speakers = [str(value) for value in source.get("valid_speakers") or []]
    if speakers != ["Classmates A", "Classmates B"]:
        raise ValueError(f"unexpected_canary_speakers:{speakers}")
    scenes = []
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
        actor_bindings = [{
            "actor": speaker,
            "responsibilities": [beat for beat in scene.get("interaction_beats") or []
                                 if speaker in str(beat)],
        } for speaker in speakers]
        scenes.append({
            "memory_event_id": f"{CASE_ID}:S{index:02d}",
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
            "actor_bindings": actor_bindings,
            "interaction_beats": list(scene.get("interaction_beats") or []),
        })
    if len(scenes) != 3:
        raise ValueError(f"adapted_scene_count:{len(scenes)}")
    return {
        "plan_version": "planner_objective_o1_realizer_adapter_v1",
        "source_case_id": CASE_ID,
        "source_objective": OBJECTIVE,
        "scenes": scenes,
    }


SCENE_GROUPS = (
    {
        "ratio": ("two parts", "2 parts", "two-to-one", "2:1"),
        "honey": ("honey",),
        "mustard": ("mustard",),
        "vinegar": ("vinegar",),
        "brown_sugar": ("brown sugar",),
        "garlic": ("garlic",),
        "paprika": ("paprika",),
    },
    {
        "thickness_trigger": ("too thick", "thicker than", "overly thick"),
        "water_revision": ("water",),
        "revised_consistency": ("thin", "spreadable", "consistency"),
    },
    {
        "ham": ("ham",),
        "glaze": ("glaze",),
        "application": ("brush", "apply", "coat", "spread"),
        "oven": ("oven",),
        "timer_or_baste": ("timer", "30 minutes", "half an hour", "baste"),
    },
)

EARLY_FINALITY_TERMS = (
    "final version", "recipe is complete", "we've nailed it", "we are done",
)
FINAL_APPLICATION_TERMS = ("brush", "apply", "coat", "baste")


def _find_evidence(turns: list[Mapping[str, Any]], alternatives: tuple[str, ...]) -> list[str]:
    refs = []
    for turn in turns:
        text = str(turn.get("text") or "").casefold()
        if any(term.casefold() in text for term in alternatives):
            refs.append(str(turn.get("turn_ref") or ""))
    return [ref for ref in refs if ref]


def semantic_audit(child: Mapping[str, Any]) -> dict[str, Any]:
    scenes = child.get("scenes") if isinstance(child.get("scenes"), list) else []
    scene_audits = []
    for index, (scene, groups) in enumerate(zip(scenes, SCENE_GROUPS), 1):
        turns = scene.get("dialogue_turns") if isinstance(scene, Mapping) else []
        turns = turns if isinstance(turns, list) else []
        evidence = {name: _find_evidence(turns, alternatives)
                    for name, alternatives in groups.items()}
        scene_audits.append({
            "scene_index": index,
            "scene_ref": scene.get("scene_ref") if isinstance(scene, Mapping) else None,
            "required_group_evidence": evidence,
            "all_required_groups_realized": all(evidence.values()),
        })
    all_text = [
        " ".join(str(turn.get("text") or "").casefold()
                 for turn in scene.get("dialogue_turns") or [])
        for scene in scenes if isinstance(scene, Mapping)
    ]
    boundaries = {
        "water_not_before_revision_scene": len(all_text) == 3 and "water" not in all_text[0],
        "oven_only_in_final_scene": len(all_text) == 3 and "oven" not in all_text[0] and "oven" not in all_text[1] and "oven" in all_text[2],
        "no_explicit_finality_before_final_scene": len(all_text) == 3 and not any(
            term in " ".join(all_text[:2]) for term in EARLY_FINALITY_TERMS),
        "final_application_only_in_final_scene": len(all_text) == 3 and not any(
            term in " ".join(all_text[:2]) for term in FINAL_APPLICATION_TERMS),
        "timer_only_in_final_scene": len(all_text) == 3 and "timer" not in all_text[0] and "timer" not in all_text[1] and "timer" in all_text[2],
    }
    if len(scenes) >= 2:
        scene2 = scenes[1].get("dialogue_turns") or []
        thick_refs = _find_evidence(scene2, SCENE_GROUPS[1]["thickness_trigger"])
        water_refs = _find_evidence(scene2, SCENE_GROUPS[1]["water_revision"])
        def turn_number(ref: str) -> int:
            return int(ref.rsplit("T", 1)[-1]) if "T" in ref else 10**9
        boundaries["trigger_precedes_revision"] = bool(
            thick_refs and water_refs and min(map(turn_number, thick_refs)) < min(map(turn_number, water_refs)))
    else:
        boundaries["trigger_precedes_revision"] = False
    return {
        "audit_version": "planner_objective_realizer_semantic_evidence_v1",
        "scene_audits": scene_audits,
        "stage_boundary_checks": boundaries,
        "overall_pass": bool(
            len(scene_audits) == 3
            and all(row["all_required_groups_realized"] for row in scene_audits)
            and all(boundaries.values())
        ),
        "limitations": [
            "This is deterministic lexical evidence, not an LLM semantic judgment.",
            "The source frozen event overlaps some accepted plan beats, so novelty is not tested.",
        ],
    }


def _report(metrics: Mapping[str, Any], mechanical: Mapping[str, Any],
            semantic: Mapping[str, Any]) -> str:
    lines = [
        "# Planner Realizer Adherence Canary v1", "",
        f"Verdict: **{metrics['conclusion']}**", "",
        f"- Source: `{CASE_ID}:{OBJECTIVE}`",
        f"- Realizer calls: `{metrics['realizer_calls']}`",
        f"- Mechanical contract valid: `{mechanical.get('valid')}`",
        f"- Semantic evidence pass: `{semantic.get('overall_pass')}`",
        "- Planner/QA/Solver/Writer/Auditor/Memorax calls: `0`", "",
        "## Stage evidence", "",
    ]
    for scene in semantic.get("scene_audits") or []:
        lines.append(f"- Scene {scene['scene_index']}: pass={scene['all_required_groups_realized']}; evidence={scene['required_group_evidence']}")
    lines.extend(["", f"Boundary checks: `{semantic.get('stage_boundary_checks')}`", "",
                  "This canary measures plan-to-dialogue adherence only. It does not establish source novelty, grounding, QA difficulty, or mainline readiness.", ""])
    return "\n".join(lines)


def run(*, run_root: Path, config: Path) -> dict[str, Any]:
    prepare_empty(run_root)
    source, parent, provenance = _load_source()
    adapted = adapt_plan(source)
    prompt = build_realizer_prompt(parent=parent, plan=adapted, include_full_parent=True)
    write_json(run_root / "source_record.json", source)
    write_json(run_root / "adapted_plan.json", adapted)
    write_json(run_root / "prompt.json", prompt)
    client = LiveJSONClient(config, max_tokens=8192, temperature=0.0)
    raw = ""
    meta: dict[str, Any] = {}
    parsed: Any = {}
    transport_error = None
    request_attempts = 1
    try:
        raw, meta = client.complete(str(prompt["prompt"]))
        (run_root / "raw_response.txt").write_text(raw, encoding="utf-8")
        parsed = extract_json(raw)
        write_json(run_root / "parsed_response.json", parsed)
    except Exception as exc:
        transport_error = f"{type(exc).__name__}:{exc}"
        (run_root / "raw_response.txt").write_text(raw, encoding="utf-8")
    mechanical = (
        normalize_realizer_output(parsed, plan=adapted, parent=parent, round_index=1)
        if transport_error is None else {"valid": False, "errors": [transport_error]}
    )
    semantic = semantic_audit(mechanical.get("child") or {}) if mechanical.get("valid") else {
        "overall_pass": False, "scene_audits": [], "stage_boundary_checks": {},
        "limitations": ["Semantic evidence audit skipped because the mechanical contract failed."],
    }
    if mechanical.get("child"):
        write_json(run_root / "child_dialogue.json", mechanical["child"])
    write_json(run_root / "mechanical_audit.json", mechanical)
    write_json(run_root / "semantic_adherence_audit.json", semantic)
    passed = bool(mechanical.get("valid") and semantic.get("overall_pass"))
    metrics = {
        "probe_version": PROBE_VERSION,
        "source": provenance,
        "source_case_id": CASE_ID,
        "source_objective": OBJECTIVE,
        "request_attempts": request_attempts,
        "realizer_calls": client.calls,
        "realizer_response_meta": meta,
        "transport_error": transport_error,
        "mechanical_contract_valid": bool(mechanical.get("valid")),
        "semantic_evidence_pass": bool(semantic.get("overall_pass")),
        "conclusion": "REALIZER_ADHERENCE_CANARY_PASS" if passed else "REALIZER_ADHERENCE_CANARY_FAIL",
        "planner_calls": 0,
        "qa_writer_calls": 0,
        "solver_calls": 0,
        "writer_calls": 0,
        "auditor_calls": 0,
        "memorax_search_calls": 0,
        "memorax_add_calls": 0,
    }
    write_json(run_root / "metrics.json", metrics)
    write_json(run_root / "manifest.json", {
        "probe_version": PROBE_VERSION, "isolated": True,
        "source_runs_read_only": True, "api_role_allowlist": ["realizer"],
        "request_attempts": request_attempts, "realizer_calls": client.calls,
        "retry_budget": 0, "planner_calls": 0, "qa_writer_calls": 0,
        "solver_calls": 0, "writer_calls": 0, "auditor_calls": 0,
        "memorax_search_calls": 0, "memorax_add_calls": 0,
    })
    (run_root / "report.md").write_text(_report(metrics, mechanical, semantic), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="single-call Realizer adherence canary")
    parser.add_argument("--config", type=Path, default=Path("/root/autodl-tmp/gxy-selfplay/config.json"))
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = (args.run_root or RUN_BASE / f"planner_realizer_adherence_canary_v1_{stamp}").resolve()
    print(json.dumps({"run_root": str(run_root), "metrics": run(run_root=run_root, config=args.config)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
