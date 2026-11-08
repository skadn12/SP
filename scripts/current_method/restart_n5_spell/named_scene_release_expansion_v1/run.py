"""Generate one named-speaker state-chain expansion with scene release.

This module is intentionally isolated from the production Memorax expansion
runner.  It freezes one Parent-grounded three-scene chain and releases only
the current scene plan to the Realizer.  Each later request receives the
already generated prior Child scenes, so the output can be audited as
unresolved -> revised -> resolved rather than as three parallel continuations.
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
    _renumber,
    build_single_scene_prompt_v2,
)
from scripts.current_method.restart_n5_spell.planner_realizer_temporal_boundary_probe_v1.scene_release_probe_v3 import (
    _safe_extract,
)
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.contracts import (
    normalize_realizer_output,
)
from scripts.current_method.restart_n5_spell.simple_append_v1.assembly import (
    assemble_expanded_dialogue,
    validate_expanded_dialogue,
)


PROBE_VERSION = "named_scene_release_expansion_v1"
MAX_TOKENS = 4096
EXPECTED_SPEAKERS = {"Nora", "Caleb"}
FORBIDDEN_ALIASES = ("Neighbors A", "Neighbors B", "Neighbor A", "Neighbor B")


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def load_parent(input_path: Path) -> dict[str, Any]:
    root = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(root, Mapping):
        raise ValueError("input_root_not_object")
    dialogue = root.get("dialogue")
    if not isinstance(dialogue, Mapping) or not isinstance(dialogue.get("turns"), list):
        raise ValueError("input_dialogue_missing")
    turns = list(dialogue["turns"])
    speakers = {str(row.get("speaker") or "").strip() for row in turns
                if isinstance(row, Mapping)}
    if speakers != EXPECTED_SPEAKERS:
        raise ValueError(f"unexpected_speakers:{sorted(speakers)}")
    if any(alias in json.dumps(root, ensure_ascii=False)
           for alias in FORBIDDEN_ALIASES):
        raise ValueError("forbidden_alias_in_parent")
    return {
        "root_id": str(root.get("root_id") or ""),
        "dialogue": {
            "dialogue_hash": str(dialogue.get("dialogue_hash") or ""),
            "turns": turns,
        },
    }


def build_chain_plan(parent: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze a single Parent-grounded gardening state chain."""
    root_id = str(parent["root_id"])
    participants = ["Nora", "Caleb"]
    scenes = [
        {
            "scene_index": 1,
            "memory_event_id": f"{root_id}:CHAIN:GARDEN:S01",
            "development_mode": "unresolved",
            "source_basis": [
                "cc-train-episode-91716:P04:T001",
                "cc-train-episode-91716:P04:T011",
            ],
            "required_participants": participants,
            "participants": participants,
            "before_state": (
                "Nora is still struggling with recurring weeds and has asked Caleb "
                "for practical advice; no method has been tested yet."
            ),
            "required_transition": (
                "Caleb proposes a diluted vinegar method, and Nora agrees to test "
                "it on a small weed patch while protecting her good plants."
            ),
            "after_state": (
                "Nora and Caleb have an agreed small-patch test plan using diluted "
                "vinegar, with protection of good plants as an explicit constraint."
            ),
            "handoff_to_next": "small-patch test plan with plant-protection constraint",
            "actor_bindings": [
                {"actor": "Caleb", "binding": "Caleb proposes and explains the initial method."},
                {"actor": "Nora", "binding": "Nora accepts the test plan and owns the garden."},
            ],
            "interaction_beats": [
                "Nora describes the recurring weeds and asks Caleb to help choose a method.",
                "Caleb explains a diluted vinegar mixture and the need to avoid good plants.",
                "Nora agrees to test the mixture on a small patch before using it broadly.",
            ],
        },
        {
            "scene_index": 2,
            "memory_event_id": f"{root_id}:CHAIN:GARDEN:S02",
            "development_mode": "revised",
            "source_basis": ["R01:S01:after_state"],
            "required_participants": participants,
            "participants": participants,
            "before_state": (
                "After Scene 1, Nora and Caleb have agreed to test diluted vinegar "
                "on a small weed patch while protecting good plants."
            ),
            "required_transition": (
                "During the small-patch test, they notice the spray could reach a "
                "good plant, so they revise the plan to a targeted brush-on method "
                "and mark the patch for a careful follow-up."
            ),
            "after_state": (
                "The original spray plan has been revised to a targeted brush-on "
                "method because protecting good plants is now a confirmed constraint."
            ),
            "handoff_to_next": "revised targeted brush-on method and marked test patch",
            "actor_bindings": [
                {"actor": "Nora", "binding": "Nora reports the test constraint from her garden."},
                {"actor": "Caleb", "binding": "Caleb revises the method to protect good plants."},
            ],
            "interaction_beats": [
                "Nora reports that the small-patch spray spread close to a good plant.",
                "Caleb recognizes the protection constraint and suggests brushing the mixture directly onto weeds.",
                "Nora and Caleb mark the patch and agree to use the revised targeted method.",
            ],
        },
        {
            "scene_index": 3,
            "memory_event_id": f"{root_id}:CHAIN:GARDEN:S03",
            "development_mode": "resolved",
            "source_basis": ["R01:S02:after_state"],
            "required_participants": participants,
            "participants": participants,
            "before_state": (
                "After Scene 2, the targeted brush-on method is the active plan and "
                "the marked test patch is ready for careful application."
            ),
            "required_transition": (
                "Nora and Caleb apply the revised brush-on method to the marked patch, "
                "check the surrounding good plants, and record the resulting garden-care method."
            ),
            "after_state": (
                "The marked weed patch has been treated with the revised targeted method "
                "without contacting the good plants, and the method is documented for future care."
            ),
            "handoff_to_next": "resolved targeted method documented",
            "actor_bindings": [
                {"actor": "Nora", "binding": "Nora applies the revised method in her garden."},
                {"actor": "Caleb", "binding": "Caleb checks the result and records the method."},
            ],
            "interaction_beats": [
                "Nora applies the mixture with a brush only to the marked weed patch.",
                "Caleb checks the surrounding good plants and confirms they were avoided.",
                "Nora and Caleb record the targeted method and agree it resolves this patch's problem.",
            ],
        },
    ]
    return {
        "plan_version": "named_garden_state_chain_contract_v1",
        "chain_type": "unresolved_revised_resolved",
        "root_id": root_id,
        "participants": participants,
        "scenes": scenes,
    }


def build_chain_prompt(*, parent: Mapping[str, Any], prior: list[Mapping[str, Any]],
                       current: Mapping[str, Any], index: int) -> dict[str, Any]:
    base = build_single_scene_prompt_v2(
        parent=parent, prior=prior, current=current, index=index)
    chain_block = f"""

STATE-CHAIN CONTRACT (strict)
This is scene {index} of exactly 3 in one continuous garden problem.
Only the current scene's state change may become true now. Do not resolve the
whole garden problem before scene 3. Do not introduce any later-scene result.
The current scene must begin from CURRENT PLAN before_state and end at its
after_state. The next scene will receive this scene's actual dialogue as
authoritative history. Preserve exact names Nora and Caleb; never use aliases.
For scene 1, keep the weed problem unresolved and only establish the small-patch
test plan. For scene 2, the test creates the plant-protection constraint and the
method must be revised; do not claim the weeds are already resolved. For scene 3,
apply the revised method and record the resolution.
"""
    return {**base, "prompt": base["prompt"] + chain_block,
            "chain_type": "unresolved_revised_resolved",
            "future_plans_included": False}


def _scene_text(scene: Mapping[str, Any]) -> str:
    return " ".join(str(row.get("text") or "") for row in scene.get("dialogue_turns") or [])


def audit_chain(child: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    scenes = list(child.get("scenes") or [])
    texts = [_scene_text(scene).casefold() for scene in scenes]
    anchor_groups = [
        ("test", "small patch", "diluted", "vinegar"),
        ("revise", "revised", "brush", "good plant", "protect"),
        ("apply", "marked", "record", "document", "treated"),
    ]
    anchor_pass = [any(term in text for term in group)
                   for text, group in zip(texts, anchor_groups)]
    early_forbidden = [
        (1, ("resolved", "problem is solved", "fully treated", "documented method")),
        (2, ("problem is solved", "fully resolved", "final result", "without contacting")),
    ]
    forbidden_hits = []
    for index, terms in early_forbidden:
        if index <= len(texts):
            forbidden_hits.extend(
                {"scene": index, "term": term}
                for term in terms if term in texts[index - 1])
    continuity_pass = len(scenes) == 3 and all(
        str(plan["scenes"][i].get("source_basis", [""])[0]).startswith("R01:S")
        for i in (1, 2)
    )
    checks = {
        "three_scenes": len(scenes) == 3,
        "anchor_coverage": anchor_pass == [True, True, True],
        "early_resolution_forbidden": not forbidden_hits,
        "state_chain_contract": continuity_pass,
        "speaker_set": sorted({str(row.get("speaker")) for scene in scenes
                                for row in scene.get("dialogue_turns") or []}) == ["Caleb", "Nora"],
    }
    return {
        "audit_version": "named_state_chain_audit_v1",
        "chain_type": "unresolved_revised_resolved",
        "checks": checks,
        "anchor_pass": anchor_pass,
        "forbidden_hits": forbidden_hits,
        "overall_pass": bool(all(checks.values())),
    }


def preflight(*, input_path: Path, run_root: Path, config: Path) -> dict[str, Any]:
    errors: list[str] = []
    if not input_path.exists():
        errors.append("input_missing")
    if run_root.exists():
        errors.append("run_root_must_be_new")
    if not config.exists():
        errors.append("config_missing")
    try:
        parent = load_parent(input_path)
        plan = build_chain_plan(parent)
    except Exception as exc:
        parent = None
        plan = None
        errors.append(f"parent_or_plan:{type(exc).__name__}:{exc}")
    return {
        "probe_version": PROBE_VERSION,
        "decision": "pass" if not errors else "fail",
        "failed_checks": errors,
        "api_calls": 0,
        "input_path": str(input_path),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest() if input_path.exists() else None,
        "run_root": str(run_root),
        "config_path": str(config),
        "parent_turn_count": len((parent or {}).get("dialogue", {}).get("turns") or []),
        "chain_plan": plan,
        "future_plans_in_any_scene_prompt": False,
        "one_request_per_scene": True,
        "retry_budget": 0,
        "memorax_calls": 0,
        "qa_calls": 0,
    }


def run_live(*, input_path: Path, run_root: Path, config: Path) -> dict[str, Any]:
    check = preflight(input_path=input_path, run_root=run_root, config=config)
    if check["decision"] != "pass":
        raise ValueError("preflight_failed:" + ",".join(check["failed_checks"]))
    prepare_empty(run_root)
    write_json(run_root / "preflight.json", check)
    parent = load_parent(input_path)
    plan = build_chain_plan(parent)
    write_json(run_root / "chain_plan.json", plan)
    client = LiveJSONClient(config, max_tokens=MAX_TOKENS, temperature=0.0)
    prior: list[dict[str, Any]] = []
    scene_records: list[dict[str, Any]] = []
    for index, current in enumerate(plan["scenes"], 1):
        scene_root = run_root / f"scene_{index:02d}"
        scene_root.mkdir(parents=True, exist_ok=True)
        prompt = build_chain_prompt(parent=parent, prior=prior,
                                    current=current, index=index)
        write_json(scene_root / "prompt.json", prompt)
        raw = ""
        error = None
        meta: dict[str, Any] = {}
        parsed: dict[str, Any] = {}
        calls_before = client.calls
        try:
            raw, meta = client.complete(prompt["prompt"])
            (scene_root / "raw_response.txt").write_text(raw, encoding="utf-8")
            parsed, envelope_mode = _safe_extract(raw)
            write_json(scene_root / "parsed_response.json", parsed)
        except Exception as exc:
            envelope_mode = None
            error = f"{type(exc).__name__}:{exc}"
            (scene_root / "raw_response.txt").write_text(raw, encoding="utf-8")
        normalized = (
            normalize_realizer_output(parsed, plan={"scenes": [current]},
                                      parent=parent, round_index=1)
            if error is None else {"valid": False, "errors": [error]}
        )
        if normalized.get("valid") and normalized.get("child", {}).get("scenes"):
            scene = _renumber(normalized["child"]["scenes"][0], index,
                              str(parent["root_id"]), str(current["memory_event_id"]))
            prior.append(scene)
            write_json(scene_root / "child_scene.json", scene)
        write_json(scene_root / "mechanical_audit.json", normalized)
        record = {
            "scene_index": index,
            "request_attempts": 1,
            "realizer_calls_completed": client.calls - calls_before,
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
        "round_index": 1,
        "scenes": prior,
    }
    write_json(run_root / "child_dialogue.json", child)
    chain_audit = audit_chain(child, plan)
    write_json(run_root / "state_chain_audit.json", chain_audit)
    expanded = None
    expanded_validation: dict[str, Any] = {"valid": False, "errors": ["child_not_complete"]}
    if len(prior) == 3 and chain_audit["overall_pass"]:
        expanded = assemble_expanded_dialogue(parent=parent, child=child)
        expanded_validation = validate_expanded_dialogue(expanded, parent=parent)
        write_json(run_root / "expanded_dialogue.json", expanded)
        write_json(run_root / "expanded_dialogue_validation.json", expanded_validation)
    usage = [row.get("response_meta", {}).get("usage") or {} for row in scene_records]
    prompt_tokens = sum(int(row.get("prompt_tokens") or 0) for row in usage)
    completion_tokens = sum(int(row.get("completion_tokens") or 0) for row in usage)
    status = "completed" if expanded_validation.get("valid") else "failed"
    report = {
        "probe_version": PROBE_VERSION,
        "status": status,
        "failure_stage": None if status == "completed" else f"scene_{len(prior) + 1:02d}",
        "run_root": str(run_root),
        "input_path": str(input_path),
        "root_id": parent["root_id"],
        "scene_count": len(prior),
        "child_turn_count": sum(len(s.get("dialogue_turns") or []) for s in prior),
        "expanded_turn_count": len(expanded.get("turns") or []) if expanded else 0,
        "realizer_calls": client.calls,
        "llm_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "memorax_calls": 0,
        "qa_calls": 0,
        "state_chain_audit": chain_audit,
        "expanded_dialogue_validation": expanded_validation,
        "historical_runs_modified": False,
    }
    write_json(run_root / "report.json", report)
    (run_root / "report.md").write_text(
        f"# Named Scene-Release Expansion v1\n\nStatus: **{status}**\n\n"
        f"Scenes: {len(prior)}/3; Child turns: {report['child_turn_count']}; "
        f"Realizer calls: {client.calls}; LLM tokens: {prompt_tokens + completion_tokens}.\n\n"
        f"State-chain audit: `{chain_audit['overall_pass']}`; Expanded validation: `{expanded_validation.get('valid')}`.\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    check = preflight(input_path=args.input, run_root=args.run_root,
                      config=args.config)
    if args.preflight_only or not args.live:
        print(json.dumps(check, ensure_ascii=False, indent=2))
        return 0 if check["decision"] == "pass" else 1
    result = run_live(input_path=args.input, run_root=args.run_root,
                      config=args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
