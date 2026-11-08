"""Append one coherent state-transition round to the frozen 89-turn dialogue.

The prior expanded dialogue is treated as immutable parent context. Three new
Realizer-only scenes extend the bakery-trial decision as
``unresolved -> revised -> resolved``. The result uses the append-only
multi-round assembly contract and does not call Planner, QA, Solver, or
Memorax.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.io_utils import (
    prepare_empty,
    write_json,
)
from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.live_adapter import (
    LiveJSONClient,
)
from scripts.current_method.restart_n5_spell.planner_realizer_temporal_boundary_probe_v1.scene_release_probe_v2 import (
    build_single_scene_prompt_v2,
)
from scripts.current_method.restart_n5_spell.planner_realizer_temporal_boundary_probe_v1.scene_release_probe_v3 import (
    _safe_extract,
)
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.contracts import (
    normalize_realizer_output,
)
from scripts.current_method.restart_n5_spell.simple_append_v1.assembly import (
    append_expansion_round,
    validate_appended_round,
)


PROBE_VERSION = "named_scene_release_expansion_episode_178392_round2_v1"
ROUND_INDEX = 2
EXPECTED_SPEAKERS = {"Maya", "Jordan"}
FORBIDDEN_ALIASES = (
    "Classmates A", "Classmates B", "Classmate A", "Classmate B",
    "Neighbors A", "Neighbors B", "Neighbor A", "Neighbor B",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def load_prior(path: Path) -> dict[str, Any]:
    expanded = load_json(path)
    turns = expanded.get("turns")
    if not isinstance(turns, list) or len(turns) != 89:
        raise ValueError(f"unexpected_prior_turn_count:{len(turns or [])}")
    speakers = {str(row.get("speaker") or "") for row in turns}
    if speakers != EXPECTED_SPEAKERS:
        raise ValueError(f"unexpected_speakers:{sorted(speakers)}")
    blob = json.dumps(expanded, ensure_ascii=False)
    if any(alias in blob for alias in FORBIDDEN_ALIASES):
        raise ValueError("forbidden_alias_in_prior")
    if expanded.get("qa_fact_source") != "complete_parent_plus_child":
        raise ValueError("prior_not_complete_parent_plus_child")
    parent = {
        "root_id": str(expanded.get("root_id") or ""),
        "dialogue": {
            "dialogue_hash": str(expanded.get("complete_dialogue_sha256") or ""),
            "turns": [
                {
                    "turn_id": str(row["turn_ref"]),
                    "session_id": str(row["session_ref"]),
                    "speaker": str(row["speaker"]),
                    "text": str(row["text"]),
                }
                for row in turns
            ],
        },
    }
    return {"expanded": expanded, "parent": parent}


def build_chain_plan(parent: Mapping[str, Any]) -> dict[str, Any]:
    root_id = str(parent["root_id"])
    participants = ["Maya", "Jordan"]
    return {
        "plan_version": "named_career_trial_followup_state_chain_v1",
        "round_index": ROUND_INDEX,
        "chain_type": "unresolved_revised_resolved",
        "root_id": root_id,
        "participants": participants,
        "prior_state_basis": ["R01:S03:T001", "R01:S03:T005", "R01:S03:T007", "R01:S03:T008"],
        "scenes": [
            {
                "scene_index": 1,
                "memory_event_id": f"{root_id}:CHAIN:CAREER:R02:S01",
                "development_mode": "unresolved",
                "source_basis": ["R01:S03:T001", "R01:S03:T005", "R01:S03:T007"],
                "required_participants": participants,
                "participants": participants,
                "before_state": (
                    "Maya has scheduled a bakery trial and defined criteria for enjoyment, pace, "
                    "growth, team support, and physical demands, but has not gathered actual trial evidence."
                ),
                "required_transition": (
                    "Maya completes an initial trial shift and reports mixed evidence: she enjoys the "
                    "creative work but finds the pace or physical demands difficult; neither a final "
                    "bakery commitment nor a rejection is allowed yet."
                ),
                "after_state": (
                    "The first trial has produced mixed observations and Maya's career fit remains unresolved."
                ),
                "handoff_to_next": "mixed trial evidence requiring a revised evaluation plan",
                "actor_bindings": [
                    {"actor": "Maya", "binding": "Maya performs the first trial and reports her own observations."},
                    {"actor": "Jordan", "binding": "Jordan helps separate positive evidence from unresolved concerns."},
                ],
                "interaction_beats": [
                    "Maya reports what she enjoyed during the first bakery shift.",
                    "Maya also reports a concrete difficulty with pace, hours, or physical demands.",
                    "Jordan records the mixed evidence and confirms that no final decision is made yet.",
                ],
            },
            {
                "scene_index": 2,
                "memory_event_id": f"{root_id}:CHAIN:CAREER:R02:S02",
                "development_mode": "revised",
                "source_basis": ["R02:S01:after_state"],
                "required_participants": participants,
                "participants": participants,
                "before_state": (
                    "Maya's first bakery trial produced both creative satisfaction and concerns about "
                    "pace, sustainability, or physical demands; the original evaluation is inconclusive."
                ),
                "required_transition": (
                    "Because the mixed evidence is not enough for a decision, Maya and Jordan revise "
                    "the evaluation plan: they seek a more suitable shift or mentoring arrangement and "
                    "prioritize sustainable hours and team support while keeping restaurant work conditional."
                ),
                "after_state": (
                    "A revised bakery follow-up plan is active, with adjusted criteria for sustainability "
                    "and support rather than an immediate permanent commitment."
                ),
                "handoff_to_next": "revised follow-up shift and sustainability criteria",
                "actor_bindings": [
                    {"actor": "Maya", "binding": "Maya explains which trial concern requires the revision."},
                    {"actor": "Jordan", "binding": "Jordan helps formulate the revised follow-up test."},
                ],
                "interaction_beats": [
                    "Maya identifies the concern that the first criteria did not resolve.",
                    "Jordan proposes an adjusted shift or mentoring check focused on sustainable work.",
                    "Maya accepts the revised plan and keeps the restaurant path as a later fallback.",
                ],
            },
            {
                "scene_index": 3,
                "memory_event_id": f"{root_id}:CHAIN:CAREER:R02:S03",
                "development_mode": "resolved",
                "source_basis": ["R02:S02:after_state"],
                "required_participants": participants,
                "participants": participants,
                "before_state": (
                    "Maya has a revised bakery follow-up plan and has not yet decided whether the "
                    "bakery environment is sustainable for her next step."
                ),
                "required_transition": (
                    "Maya completes the revised follow-up, compares the new observations with the first "
                    "trial, and resolves the immediate decision by continuing with a bounded bakery "
                    "trial period and a recorded review rule; her entire long-term career is not declared settled."
                ),
                "after_state": (
                    "Maya has an evidence-based immediate bakery plan with a review point and a conditional "
                    "restaurant fallback, resolving the current trial decision without overclaiming a lifetime career choice."
                ),
                "handoff_to_next": "bounded bakery continuation plan with review rule",
                "actor_bindings": [
                    {"actor": "Maya", "binding": "Maya compares evidence and chooses the bounded immediate plan."},
                    {"actor": "Jordan", "binding": "Jordan helps document the review rule and fallback condition."},
                ],
                "interaction_beats": [
                    "Maya reports the revised follow-up observations.",
                    "They compare the first and second sets of evidence against Maya's goals.",
                    "Maya chooses a bounded continuation plan and records when to reconsider restaurants.",
                ],
            },
        ],
    }


def build_chain_prompt(*, parent: Mapping[str, Any], prior: list[Mapping[str, Any]],
                       current: Mapping[str, Any], index: int) -> dict[str, Any]:
    base = build_single_scene_prompt_v2(
        parent=parent, prior=prior, current=current, index=index)
    chain = f"""

STATE-CHAIN CONTRACT (strict, round 2)
This is scene {index} of exactly 3 in a second continuous career-trial round.
The complete previous dialogue, including R01:S01-S03, is authoritative
history. Only the current scene's state change may become true now. Do not
invent later-round outcomes early. Preserve exact names Maya and Jordan; never
use A/B, Classmates, Neighbors, user, or assistant labels.
Scene 1 must report mixed first-trial evidence without a final decision.
Scene 2 must revise the evaluation/follow-up plan because of that mixed evidence.
Scene 3 must compare the follow-up evidence and resolve only the immediate
bounded bakery continuation decision; do not claim Maya's lifetime career is
settled or that a restaurant job has been obtained.
"""
    return {
        **base,
        "prompt": base["prompt"] + chain,
        "prompt_version": f"{PROBE_VERSION}_prompt",
        "round_index": ROUND_INDEX,
        "future_plans_included": False,
    }


def renumber(scene: Mapping[str, Any], index: int, root_id: str,
             event_id: str) -> dict[str, Any]:
    ref = f"R{ROUND_INDEX:02d}:S{index:02d}"
    return {
        "scene_ref": ref,
        "session_ref": f"{root_id}:r{ROUND_INDEX:02d}:s{index:02d}",
        "memory_event_id": event_id,
        "dialogue_turns": [
            {"turn_ref": f"{ref}:T{position:03d}",
             "speaker": str(row.get("speaker") or ""),
             "text": str(row.get("text") or "")}
            for position, row in enumerate(scene.get("dialogue_turns") or [], 1)
        ],
    }


def audit_chain(child: Mapping[str, Any]) -> dict[str, Any]:
    scenes = list(child.get("scenes") or [])
    texts = [" ".join(str(row.get("text") or "") for row in scene.get("dialogue_turns") or []).casefold()
             for scene in scenes]
    anchor_groups = [
        ("trial", "shift", "creative", "pace", "physical", "mixed"),
        ("revise", "revised", "adjust", "mentor", "sustainable", "support"),
        ("compare", "evidence", "continue", "review", "fallback", "plan"),
    ]
    anchor_pass = [any(term in text for term in group)
                   for text, group in zip(texts, anchor_groups)]
    forbidden = [
        (1, ("final decision", "permanent commitment", "career is settled", "restaurant job obtained")),
        (2, ("permanent commitment", "career is settled", "final career", "restaurant job obtained")),
    ]
    forbidden_hits = [
        {"scene": index, "term": term}
        for index, terms in forbidden if index <= len(texts)
        for term in terms
        if term in texts[index - 1]
        and not (term == "final decision" and re.search(
            r"\b(?:no|not)\s+(?:a\s+)?final decision\b", texts[index - 1]))
    ]
    speakers = sorted({str(row.get("speaker")) for scene in scenes
                       for row in scene.get("dialogue_turns") or []})
    checks = {
        "three_scenes": len(scenes) == 3,
        "anchor_coverage": anchor_pass == [True, True, True],
        "early_resolution_forbidden": not forbidden_hits,
        "round_two_refs": all(str(scene.get("scene_ref", "")).startswith("R02:S") for scene in scenes),
        "speaker_set": speakers == ["Jordan", "Maya"],
        "no_forbidden_aliases": not any(alias in json.dumps(child, ensure_ascii=False)
                                         for alias in FORBIDDEN_ALIASES),
    }
    return {
        "audit_version": "named_career_trial_followup_state_chain_audit_v1",
        "chain_type": "unresolved_revised_resolved",
        "round_index": ROUND_INDEX,
        "checks": checks,
        "anchor_pass": anchor_pass,
        "forbidden_hits": forbidden_hits,
        "overall_pass": bool(all(checks.values())),
    }


def preflight(*, prior_path: Path, run_root: Path, config: Path) -> dict[str, Any]:
    errors: list[str] = []
    if not prior_path.exists():
        errors.append("prior_expanded_missing")
    if run_root.exists():
        errors.append("run_root_must_be_new")
    if not config.exists():
        errors.append("config_missing")
    prior = None
    plan = None
    try:
        loaded = load_prior(prior_path)
        prior, parent = loaded["expanded"], loaded["parent"]
        plan = build_chain_plan(parent)
    except Exception as exc:
        errors.append(f"prior_or_plan:{type(exc).__name__}:{exc}")
    return {
        "probe_version": PROBE_VERSION,
        "decision": "pass" if not errors else "fail",
        "failed_checks": errors,
        "api_calls": 0,
        "prior_path": str(prior_path),
        "prior_sha256": sha256_file(prior_path) if prior_path.exists() else None,
        "prior_turn_count": len((prior or {}).get("turns") or []),
        "run_root": str(run_root),
        "config_path": str(config),
        "round_index": ROUND_INDEX,
        "chain_plan": plan,
        "future_plans_in_any_scene_prompt": False,
        "one_request_per_scene": True,
        "retry_budget": 0,
        "memorax_calls": 0,
        "qa_calls": 0,
    }


def run_live(*, prior_path: Path, run_root: Path, config: Path) -> dict[str, Any]:
    check = preflight(prior_path=prior_path, run_root=run_root, config=config)
    if check["decision"] != "pass":
        raise ValueError("preflight_failed:" + ",".join(check["failed_checks"]))
    prepare_empty(run_root)
    write_json(run_root / "preflight.json", check)
    loaded = load_prior(prior_path)
    prior_expanded, parent = loaded["expanded"], loaded["parent"]
    plan = build_chain_plan(parent)
    write_json(run_root / "chain_plan.json", plan)
    write_json(run_root / "prior_frozen_manifest.json", {
        "path": str(prior_path),
        "sha256": sha256_file(prior_path),
        "turn_count": len(prior_expanded["turns"]),
        "complete_dialogue_sha256": prior_expanded.get("complete_dialogue_sha256"),
        "prior_run_modified": False,
    })
    client = LiveJSONClient(config, max_tokens=4096, temperature=0.0)
    prior_child: list[dict[str, Any]] = []
    scene_records: list[dict[str, Any]] = []
    for index, current in enumerate(plan["scenes"], 1):
        scene_root = run_root / f"scene_{index:02d}"
        scene_root.mkdir(parents=True, exist_ok=True)
        prompt = build_chain_prompt(parent=parent, prior=prior_child,
                                    current=current, index=index)
        write_json(scene_root / "prompt.json", prompt)
        raw, meta, parsed, error = "", {}, {}, None
        try:
            raw, meta = client.complete(prompt["prompt"])
            (scene_root / "raw_response.txt").write_text(raw, encoding="utf-8")
            parsed, envelope_mode = _safe_extract(raw)
            write_json(scene_root / "parsed_response.json", parsed)
        except Exception as exc:
            envelope_mode = None
            error = f"{type(exc).__name__}:{exc}"
            (scene_root / "raw_response.txt").write_text(raw, encoding="utf-8")
        normalized = (normalize_realizer_output(
            parsed, plan={"scenes": [current]}, parent=parent,
            round_index=ROUND_INDEX) if error is None else {"valid": False, "errors": [error]})
        if normalized.get("valid") and normalized.get("child", {}).get("scenes"):
            scene = renumber(normalized["child"]["scenes"][0], index,
                             str(parent["root_id"]), str(current["memory_event_id"]))
            prior_child.append(scene)
            write_json(scene_root / "child_scene.json", scene)
        write_json(scene_root / "mechanical_audit.json", normalized)
        record = {
            "scene_index": index,
            "request_attempts": 1,
            "realizer_calls_completed": client.calls,
            "transport_error": error,
            "mechanical_contract_valid": bool(normalized.get("valid")),
            "response_meta": meta,
            "envelope_mode": envelope_mode,
        }
        write_json(scene_root / "metrics.json", record)
        scene_records.append(record)
        if error or not normalized.get("valid"):
            break
    child = {
        "schema_version": "memory_event_continuation_child_v1",
        "root_id": parent["root_id"],
        "round_index": ROUND_INDEX,
        "scenes": prior_child,
    }
    write_json(run_root / "child_dialogue_round2.json", child)
    audit = audit_chain(child)
    write_json(run_root / "state_chain_audit_round2.json", audit)
    appended = None
    append_validation: dict[str, Any] = {"valid": False, "errors": ["child_not_complete"]}
    if len(prior_child) == 3 and audit["overall_pass"]:
        appended = append_expansion_round(prior_expanded=prior_expanded, child=child)
        append_validation = validate_appended_round(appended, prior_expanded=prior_expanded)
        write_json(run_root / "expanded_dialogue_round2.json", appended)
        write_json(run_root / "expanded_dialogue_round2_validation.json", append_validation)
    usage = [row.get("response_meta", {}).get("usage") or {} for row in scene_records]
    prompt_tokens = sum(int(row.get("prompt_tokens") or 0) for row in usage)
    completion_tokens = sum(int(row.get("completion_tokens") or 0) for row in usage)
    status = "completed" if append_validation.get("valid") else "failed"
    report = {
        "probe_version": PROBE_VERSION,
        "status": status,
        "failure_stage": None if status == "completed" else f"scene_{len(prior_child) + 1:02d}",
        "prior_path": str(prior_path),
        "prior_turn_count": len(prior_expanded["turns"]),
        "new_round": ROUND_INDEX,
        "new_child_turn_count": sum(len(s.get("dialogue_turns") or []) for s in prior_child),
        "appended_turn_count": len(appended.get("turns") or []) if appended else 0,
        "realizer_calls": client.calls,
        "llm_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "memorax_calls": 0,
        "qa_calls": 0,
        "state_chain_audit": audit,
        "expanded_dialogue_validation": append_validation,
        "prior_run_modified": False,
    }
    write_json(run_root / "report.json", report)
    (run_root / "report.md").write_text(
        f"# Episode 178392 Round-2 Scene-Release Expansion\n\n"
        f"Status: **{status}**\n\n"
        f"Prior turns: {report['prior_turn_count']}; new Child turns: {report['new_child_turn_count']}; "
        f"appended turns: {report['appended_turn_count']}; Realizer calls: {client.calls}; "
        f"LLM tokens: {prompt_tokens + completion_tokens}.\n\n"
        f"State-chain audit: `{audit['overall_pass']}`; append validation: `{append_validation.get('valid')}`.\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-expanded", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    check = preflight(prior_path=args.prior_expanded, run_root=args.run_root,
                      config=args.config)
    if args.preflight_only or not args.live:
        print(json.dumps(check, ensure_ascii=False, indent=2))
        return 0 if check["decision"] == "pass" else 1
    result = run_live(prior_path=args.prior_expanded, run_root=args.run_root,
                      config=args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
