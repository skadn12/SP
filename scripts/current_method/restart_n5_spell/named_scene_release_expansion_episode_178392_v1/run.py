"""Run the validated scene-release chain on a second, human-speaker root.

This is an isolated adapter around ``named_scene_release_expansion_v1``.  It
keeps the production expansion and QA paths untouched while changing only the
frozen input, names, and Parent-grounded career-transition chain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.current_method.restart_n5_spell.named_scene_release_expansion_v1 import run as base


PROBE_VERSION = "named_scene_release_expansion_episode_178392_v1"
SPEAKERS = {"Maya", "Jordan"}
ALIASES = (
    "Classmates A", "Classmates B", "Classmate A", "Classmate B",
    "Neighbors A", "Neighbors B", "Neighbor A", "Neighbor B",
)


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
    if speakers != SPEAKERS:
        raise ValueError(f"unexpected_speakers:{sorted(speakers)}")
    if any(alias in json.dumps(root, ensure_ascii=False) for alias in ALIASES):
        raise ValueError("forbidden_alias_in_parent")
    return {
        "root_id": str(root.get("root_id") or ""),
        "dialogue": {
            "dialogue_hash": str(dialogue.get("dialogue_hash") or ""),
            "turns": turns,
        },
    }


def build_chain_plan(parent: Mapping[str, Any]) -> dict[str, Any]:
    root_id = str(parent["root_id"])
    participants = ["Maya", "Jordan"]
    return {
        "plan_version": "named_career_state_chain_contract_v1",
        "chain_type": "unresolved_revised_resolved",
        "root_id": root_id,
        "participants": participants,
        "scenes": [
            {
                "scene_index": 1,
                "memory_event_id": f"{root_id}:CHAIN:CAREER:S01",
                "development_mode": "unresolved",
                "source_basis": [
                    "cc-train-episode-178392:P05:T008",
                    "cc-train-episode-178392:P05:T009",
                    "cc-train-episode-178392:P02:T009",
                ],
                "required_participants": participants,
                "participants": participants,
                "before_state": (
                    "Maya has left the cashier job because of stress and unhappiness, "
                    "but has not chosen her next culinary step; she enjoys baking and cooking."
                ),
                "required_transition": (
                    "Jordan helps Maya turn that interest into a concrete search: she will "
                    "compare bakery and restaurant roles and identify where to apply, without "
                    "claiming that a new job is secured yet."
                ),
                "after_state": (
                    "Maya has an explicit initial culinary-search plan, with bakery and "
                    "restaurant options still open and no application outcome yet."
                ),
                "handoff_to_next": "open bakery-versus-restaurant search plan",
                "actor_bindings": [
                    {"actor": "Maya", "binding": "Maya owns the career decision and describes her culinary interest."},
                    {"actor": "Jordan", "binding": "Jordan helps structure the initial comparison and search."},
                ],
                "interaction_beats": [
                    "Maya explains that quitting solved the immediate stress but left her without a next step.",
                    "Jordan connects her stated love of baking and cooking to two possible role types.",
                    "Maya agrees to compare the options before applying, keeping the decision unresolved.",
                ],
            },
            {
                "scene_index": 2,
                "memory_event_id": f"{root_id}:CHAIN:CAREER:S02",
                "development_mode": "revised",
                "source_basis": ["R01:S01:after_state"],
                "required_participants": participants,
                "participants": participants,
                "before_state": (
                    "Maya is comparing bakery and restaurant roles as possible culinary next steps, "
                    "but has not applied."
                ),
                "required_transition": (
                    "New information about schedules and experience requirements makes the broad "
                    "search impractical, so Maya and Jordan revise it to a staged bakery-first "
                    "trial while preserving a later restaurant option."
                ),
                "after_state": (
                    "The open two-track search has been revised to a bakery-first trial plan, "
                    "with restaurant work deferred until Maya has more experience."
                ),
                "handoff_to_next": "revised bakery-first trial plan",
                "actor_bindings": [
                    {"actor": "Maya", "binding": "Maya reports the new constraints and accepts the revised order."},
                    {"actor": "Jordan", "binding": "Jordan helps compare the constraints and supports the revision."},
                ],
                "interaction_beats": [
                    "Maya reports that the restaurant options require experience or schedules she cannot start with.",
                    "Jordan suggests using a bakery trial as a lower-risk first step while keeping the restaurant goal.",
                    "Maya revises the plan and specifies what evidence she will use to judge the trial.",
                ],
            },
            {
                "scene_index": 3,
                "memory_event_id": f"{root_id}:CHAIN:CAREER:S03",
                "development_mode": "resolved",
                "source_basis": ["R01:S02:after_state"],
                "required_participants": participants,
                "participants": participants,
                "before_state": (
                    "Maya has chosen a bakery-first trial as the revised transition plan, "
                    "while the restaurant option remains a later possibility."
                ),
                "required_transition": (
                    "Maya completes the bakery-first step by submitting a targeted application "
                    "and scheduling a trial shift; she and Jordan record the criteria for deciding "
                    "whether to pursue restaurant work next."
                ),
                "after_state": (
                    "Maya's immediate career transition is resolved into a documented bakery trial "
                    "with a clear follow-up decision rule, rather than an unsupported claim that "
                    "her long-term career is already settled."
                ),
                "handoff_to_next": "documented bakery trial and restaurant follow-up rule",
                "actor_bindings": [
                    {"actor": "Maya", "binding": "Maya submits the application and commits to the trial."},
                    {"actor": "Jordan", "binding": "Jordan helps record the follow-up criteria."},
                ],
                "interaction_beats": [
                    "Maya submits an application to a bakery that matches the revised plan.",
                    "Jordan helps her schedule a trial shift and define what to evaluate.",
                    "They record the trial and the condition for revisiting restaurant work later.",
                ],
            },
        ],
    }


def build_chain_prompt(*, parent: Mapping[str, Any], prior: list[Mapping[str, Any]],
                       current: Mapping[str, Any], index: int) -> dict[str, Any]:
    base_prompt = base.build_single_scene_prompt_v2(
        parent=parent, prior=prior, current=current, index=index)
    chain_block = f"""

STATE-CHAIN CONTRACT (strict)
This is scene {index} of exactly 3 in one continuous career-transition problem.
Only the current scene's state change may become true now. Do not resolve the
whole transition before scene 3 and do not introduce any later-scene result.
The current scene must begin from CURRENT PLAN before_state and end at its
after_state. Later scenes receive the actual prior dialogue as authoritative
history. Preserve exact names Maya and Jordan; never use aliases.
Scene 1 establishes only the open bakery-versus-restaurant search. Scene 2
must introduce a concrete schedule/experience constraint and revise that plan.
Scene 3 completes the bakery-first trial step and records a bounded follow-up;
do not claim Maya's entire long-term career is settled.
"""
    return {
        **base_prompt,
        "prompt": base_prompt["prompt"] + chain_block,
        "chain_type": "unresolved_revised_resolved",
        "future_plans_included": False,
    }


def audit_chain(child: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    scenes = list(child.get("scenes") or [])
    texts = [" ".join(str(row.get("text") or "") for row in scene.get("dialogue_turns") or []).casefold()
             for scene in scenes]
    anchor_groups = [
        ("cashier", "baking", "cooking", "bakery", "restaurant", "compare"),
        ("schedule", "experience", "constraint", "revise", "revised", "trial"),
        ("application", "apply", "trial shift", "record", "document", "follow-up"),
    ]
    anchor_pass = [any(term in text for term in group)
                   for text, group in zip(texts, anchor_groups)]
    early_forbidden = [
        (1, ("application submitted", "hired", "new job secured", "career is settled")),
        (2, ("application submitted", "hired", "career is settled", "final job")),
    ]
    forbidden_hits = []
    for index, terms in early_forbidden:
        if index <= len(texts):
            forbidden_hits.extend(
                {"scene": index, "term": term}
                for term in terms if term in texts[index - 1])
    checks = {
        "three_scenes": len(scenes) == 3,
        "anchor_coverage": anchor_pass == [True, True, True],
        "early_resolution_forbidden": not forbidden_hits,
        "state_chain_contract": len(scenes) == 3,
        "speaker_set": sorted({str(row.get("speaker")) for scene in scenes
                                for row in scene.get("dialogue_turns") or []}) == ["Jordan", "Maya"],
    }
    return {
        "audit_version": "named_career_state_chain_audit_v1",
        "chain_type": "unresolved_revised_resolved",
        "checks": checks,
        "anchor_pass": anchor_pass,
        "forbidden_hits": forbidden_hits,
        "overall_pass": bool(all(checks.values())),
    }


def main() -> int:
    # Patch only the imported isolated runner; no production module is changed.
    base.PROBE_VERSION = PROBE_VERSION
    base.EXPECTED_SPEAKERS = SPEAKERS
    base.FORBIDDEN_ALIASES = ALIASES
    base.load_parent = load_parent
    base.build_chain_plan = build_chain_plan
    base.build_chain_prompt = build_chain_prompt
    base.audit_chain = audit_chain

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    check = base.preflight(input_path=args.input, run_root=args.run_root,
                           config=args.config)
    if args.preflight_only or not args.live:
        print(json.dumps(check, ensure_ascii=False, indent=2))
        return 0 if check["decision"] == "pass" else 1
    result = base.run_live(input_path=args.input, run_root=args.run_root,
                           config=args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
