"""Zero-API hard preflight for incremental event-chain Expansion."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..core import atomic_json, atomic_text, sha256_file
from ..live_backends import load_challenger_config
from ..simple_append_memorax_v1_fullr1_durable_devdiv_probe_v1.context import (
    build_fresh_r1_context,
)
from ..simple_append_memorax_v1_fullr1_durable_devdiv_probe_v1.memory_context import (
    MEMORY_OVERVIEW_VERSION,
)
from ..simple_append_v1.assembly import assemble_expanded_dialogue, validate_expanded_dialogue
from ..simple_append_v1.contracts import SCENE_HARD_MAXIMUM, SCENE_HARD_MINIMUM
from .contracts import (
    DELTA_UPDATER_SCHEMA_VERSION, INITIALIZER_SCHEMA_VERSION,
    PLANNER_SCHEMA_VERSION, apply_delta_updates,
    audit_episode_146700_initialization, compile_round_plan,
    materialize_event_chains, normalize_round_realizer_output,
    validate_delta_updater_output, validate_initializer_output,
    validate_planner_continuation, validate_planner_output,
)
from .prompts import (
    build_chain_initializer_prompt, build_delta_updater_prompt,
    build_planner_prompt, build_round_realizer_prompt,
)
from .settings import (
    CONTEXT_LIMIT, DEFAULT_API_CONFIG, DEFAULT_INPUT, DEFAULT_OVERVIEW,
    DEFAULT_REPORT_ROOT, DEFAULT_RUN_ROOT, DEFAULT_TOKENIZER,
    DELTA_UPDATER_MAX_TOKENS, FROZEN_INPUT_SHA256, FROZEN_OVERVIEW_SHA256,
    INITIALIZER_MAX_TOKENS, LIVE_TIMEOUT_SECONDS, PLANNER_MAX_TOKENS,
    ROUND_REALIZER_MAX_TOKENS, protected_integrity_snapshot, source_manifest,
)


PREFLIGHT_VERSION = "simple_append_incremental_event_chain_preflight_v1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected_object:{path}")
    return value


def _fixture_initializer(parent: Mapping[str, Any]) -> dict[str, Any]:
    speakers = list(dict.fromkeys(
        str(row["speaker"]) for row in parent["dialogue"]["turns"]))
    root = str(parent["root_id"])
    sessions = list(dict.fromkeys(
        str(row["session_id"]) for row in parent["dialogue"]["turns"]))
    session = lambda index: sessions[index - 1]
    return {
        "chain_initializer_schema_version": INITIALIZER_SCHEMA_VERSION,
        "session_event_candidates": [
            {"session_ref": session(1), "events": [
                {
                    "event_summary": "A missed B's soccer kick and felt disappointed",
                    "event_state": "A missed B's kick and remained disappointed about the miss.",
                    "participants": speakers[:2],
                    "evidence_turn_refs": [f"{root}:P01:T001", f"{root}:P01:T003"],
                },
                {
                    "event_summary": "B proposed resuming soccer practice and A accepted",
                    "event_state": "A has accepted B's suggestion to resume regular soccer practice.",
                    "participants": speakers[:2],
                    "evidence_turn_refs": [f"{root}:P01:T006", f"{root}:P01:T007"],
                },
            ]},
            {"session_ref": session(2), "events": [{
                "event_summary": "A told B that the store has a new cashier",
                "event_state": "A knows the store has a new cashier but has not worked with her yet.",
                "participants": speakers[:2],
                "evidence_turn_refs": [f"{root}:P02:T001", f"{root}:P02:T003"],
            }]},
            {"session_ref": session(3), "events": [{
                "event_summary": "A apologized to a friend after being rude",
                "event_state": "A apologized to the friend and felt better afterward.",
                "participants": speakers[:2],
                "evidence_turn_refs": [f"{root}:P03:T003", f"{root}:P03:T005"],
            }]},
            {"session_ref": session(4), "events": [{
                "event_summary": "A became ill, planned a doctor visit, and obtained shift coverage",
                "event_state": "A plans to see a doctor and B has agreed to provide work coverage.",
                "participants": speakers[:2],
                "evidence_turn_refs": [f"{root}:P04:T005", f"{root}:P04:T006"],
            }]},
            {"session_ref": session(5), "events": [{
                "event_summary": "A placed a call about a report mistake and waited for the boss",
                "event_state": "A placed an urgent call, is waiting for the boss to answer, and intends to disclose the report mistake and take responsibility.",
                "participants": speakers[:2],
                "evidence_turn_refs": [
                    f"{root}:P05:T001", f"{root}:P05:T003", f"{root}:P05:T009",
                ],
            }]},
        ],
        "continuation_links": [{
            "earlier_session_ref": session(1), "earlier_candidate_index": 1,
            "later_session_ref": session(1), "later_candidate_index": 2,
        }],
    }


def _fixture_planner(chain_set: Mapping[str, Any]) -> dict[str, Any]:
    chains = list(chain_set["chains"])
    return {
        "planner_schema_version": PLANNER_SCHEMA_VERSION,
        "round_goal": "Advance three established chains into distinct concrete consequences.",
        "scenes": [
            {
                "chain_id": chains[0]["chain_id"],
                "goal": "Show the first resumed practice and a practical obstacle.",
                "participants": list(chains[0]["participants"]),
                "new_development": "A and B begin practice and discover the field is unavailable.",
                "interaction_beats": ["arrive", "discover closure", "choose another field"],
                "transition_from_previous": "Continue after the Parent tail.",
                "desired_outcome": "They secure another location.",
                "durable_outcome": "A and B book another field for the next session.",
            },
            {
                "chain_id": chains[1]["chain_id"],
                "goal": "Show the doctor and coverage plan's concrete result.",
                "participants": list(chains[1]["participants"]),
                "new_development": "A reports medical guidance after B completes the covered shift.",
                "interaction_beats": ["share guidance", "confirm shift", "set return condition"],
                "transition_from_previous": "Shift to the next workday.",
                "desired_outcome": "A has a recovery and return arrangement.",
                "durable_outcome": "A remains off work until the agreed return date.",
            },
            {
                "chain_id": chains[2]["chain_id"],
                "goal": "Move the pending disclosure to the boss's response.",
                "participants": list(chains[2]["participants"]),
                "new_development": "A completes the disclosure and receives a correction deadline.",
                "interaction_beats": ["boss answers", "A explains", "deadline is set"],
                "transition_from_previous": "Return to the unresolved report issue.",
                "desired_outcome": "The correction has an owner and deadline.",
                "durable_outcome": "A must submit the corrected report by the deadline.",
            },
        ],
    }


def _fixture_realizer(compiled: Mapping[str, Any]) -> dict[str, Any]:
    dialogues = {}
    for scene_index, scene in enumerate(compiled["scenes"], 1):
        participants = list(scene["participants"])
        dialogues[scene["scene_ref"]] = [{
            "speaker": participants[index % len(participants)],
            "text": f"Offline event-chain scene {scene_index} progression turn {index + 1}.",
        } for index in range(9)]
    return {
        "realizer_schema_version": "simple_append_round_realizer_v2",
        "scene_dialogues": dialogues,
    }


def _fixture_delta(
    compiled: Mapping[str, Any], child: Mapping[str, Any],
) -> dict[str, Any]:
    child_by_scene = {str(row["scene_ref"]): row for row in child["scenes"]}
    updates = []
    for scene in compiled["scenes"]:
        child_scene = child_by_scene[str(scene["scene_ref"])]
        updates.append({
            "chain_id": scene["chain_id"],
            "update_status": "updated",
            "new_current_state": f"Offline new state for {scene['chain_id']}.",
            "evidence_turn_refs": [child_scene["dialogue_turns"][-1]["turn_ref"]],
        })
    return {
        "delta_updater_schema_version": DELTA_UPDATER_SCHEMA_VERSION,
        "chain_updates": updates,
        "untracked_new_event_candidates": [],
    }


def run_preflight(
    *, input_path: Path = DEFAULT_INPUT, overview_path: Path = DEFAULT_OVERVIEW,
    run_root: Path = DEFAULT_RUN_ROOT, api_config_path: Path = DEFAULT_API_CONFIG,
    tokenizer_path: Path = DEFAULT_TOKENIZER,
    token_counter: Callable[[str], int] | None = None,
) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "run_root_absent": not run_root.exists(),
        "input_exists": input_path.is_file(),
        "overview_exists": overview_path.is_file(),
        "pre_generation_overview_only": overview_path.name == "pre_generation_overview.json",
        "api_config_exists": api_config_path.is_file(),
        "tokenizer_exists": tokenizer_path.exists(),
        "input_sha_frozen": input_path.is_file() and sha256_file(input_path) == FROZEN_INPUT_SHA256,
        "overview_sha_frozen": overview_path.is_file() and sha256_file(overview_path) == FROZEN_OVERVIEW_SHA256,
        "api_calls_zero": True,
        "memorax_calls_zero": True,
        "qa_solver_calls_zero": True,
    }
    details: dict[str, Any] = {}
    prompts: dict[str, dict[str, Any]] = {}
    parent: dict[str, Any] = {}
    overview: dict[str, Any] = {}
    tail: dict[str, Any] = {}
    chain_set: dict[str, Any] = {}
    compiled: dict[str, Any] = {}
    child: dict[str, Any] = {}
    try:
        if token_counter is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
            counter = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
        else:
            counter = token_counter
        parent = _load(input_path)
        overview = _load(overview_path)
        tail = build_fresh_r1_context(parent, token_counter=counter)
        checks["authoritative_parent_valid"] = tail["parent_turn_count"] == 62
        checks["context_selection_unchanged"] = (
            tail["full_parent_included"] is True and
            tail["complete_sessions_only"] is True and
            tail["raw_dialogue_context_budget_tokens"] == 4096 and
            tail["parent_tail_turn_id"] == parent["dialogue"]["turns"][-1]["turn_id"]
        )
        checks["memorax_overview_contract_unchanged"] = (
            overview.get("memory_overview_version") == MEMORY_OVERVIEW_VERSION and
            overview.get("root_id") == parent.get("root_id") and
            overview.get("memory_is_authoritative") is False and
            overview.get("memory_is_qa_fact_source") is False
        )
        initializer = _fixture_initializer(parent)
        initializer_validation = validate_initializer_output(
            initializer, parent_turns=parent["dialogue"]["turns"])
        checks["initializer_fixture_valid"] = initializer_validation["valid"]
        invalid_state = json.loads(json.dumps(initializer))
        invalid_state["session_event_candidates"][0]["events"][0]["event_state"] = ""
        checks["initializer_empty_current_state_rejected"] = not validate_initializer_output(
            invalid_state, parent_turns=parent["dialogue"]["turns"])["valid"]
        invalid_ref = json.loads(json.dumps(initializer))
        invalid_ref["session_event_candidates"][0]["events"][0]["evidence_turn_refs"] = ["missing-parent-ref"]
        checks["initializer_missing_ref_rejected"] = not validate_initializer_output(
            invalid_ref, parent_turns=parent["dialogue"]["turns"])["valid"]
        child_ref = json.loads(json.dumps(initializer))
        child_ref["session_event_candidates"][0]["events"][0]["evidence_turn_refs"] = ["R01:S01:T001"]
        checks["initializer_child_ref_rejected"] = not validate_initializer_output(
            child_ref, parent_turns=parent["dialogue"]["turns"])["valid"]
        cross_window = json.loads(json.dumps(initializer))
        cross_window["session_event_candidates"][0]["events"][0]["evidence_turn_refs"] = [
            f"{parent['root_id']}:P02:T001"]
        checks["initializer_cross_window_ref_rejected"] = not validate_initializer_output(
            cross_window, parent_turns=parent["dialogue"]["turns"])["valid"]
        chain_set = materialize_event_chains(
            initializer, parent_turns=parent["dialogue"]["turns"],
            root_id=parent["root_id"],
            parent_dialogue_sha256=parent["dialogue"]["dialogue_hash"])
        initializer_audit = audit_episode_146700_initialization(chain_set)
        checks["initializer_canary_semantic_checkpoints_valid"] = initializer_audit["valid"]
        checks["deterministic_chain_ids"] = [
            row["chain_id"] for row in chain_set["chains"]
        ] == ["EC0001", "EC0002", "EC0003", "EC0004", "EC0005"]
        checks["soccer_continuation_linked"] = (
            len(chain_set["chains"][0]["source_event_refs"]) == 2)
        checks["concrete_events_remain_independent"] = (
            len(set(initializer_audit["concrete_event_matches"].values())) == 5)
        checks["coverage_word_form_normalized"] = (
            "accepted_coverage_missing_from_state" not in " ".join(
                initializer_audit["errors"]))
        prompts["initializer"] = build_chain_initializer_prompt(
            tail_context=tail, memory_overview=overview)
        planner = _fixture_planner(chain_set)
        planner_validation = validate_planner_output(
            planner, chain_set=chain_set, parent_speakers=tail["parent_speakers"])
        continuation = validate_planner_continuation(planner, chain_set=chain_set)
        checks["planner_fixture_valid"] = planner_validation["valid"]
        checks["planner_continuation_fixture_valid"] = continuation["valid"]
        checks["three_distinct_chain_ids"] = len(set(planner_validation["chain_ids"])) == 3
        unknown = json.loads(json.dumps(planner))
        unknown["scenes"][0]["chain_id"] = "EC9999"
        checks["planner_unknown_chain_rejected"] = not validate_planner_output(
            unknown, chain_set=chain_set, parent_speakers=tail["parent_speakers"])["valid"]
        duplicate = json.loads(json.dumps(planner))
        duplicate["scenes"][1]["chain_id"] = duplicate["scenes"][0]["chain_id"]
        duplicate["scenes"][1]["participants"] = duplicate["scenes"][0]["participants"]
        checks["planner_duplicate_chain_rejected"] = not validate_planner_output(
            duplicate, chain_set=chain_set, parent_speakers=tail["parent_speakers"])["valid"]
        compiled = compile_round_plan(planner, chain_set=chain_set, tail_context=tail)
        prompts["planner"] = build_planner_prompt(
            chain_set=chain_set, parent_speakers=tail["parent_speakers"])
        prompts["realizer"] = build_round_realizer_prompt(
            compiled_plan=compiled, tail_context=tail)
        checks["realizer_receives_frozen_current_state"] = all(
            scene["event_chain"]["current_state"] in prompts["realizer"]["prompt"]
            for scene in compiled["scenes"])
        normalized = normalize_round_realizer_output(
            _fixture_realizer(compiled), compiled_plan=compiled,
            parent_utterances=[row["text"] for row in parent["dialogue"]["turns"]],
            parent_turn_refs=[row["turn_id"] for row in parent["dialogue"]["turns"]])
        checks["round_realizer_fixture_valid"] = normalized["valid"]
        child = normalized["child"]
        expanded = assemble_expanded_dialogue(parent=parent, child=child)
        expanded_validation = validate_expanded_dialogue(expanded, parent=parent)
        checks["append_only_assembly_valid"] = expanded_validation["valid"]
        checks["append_only_contract_unchanged"] = (
            expanded.get("append_mode") == "parent_tail_only" and
            expanded.get("parent_dialogue_sha256") == parent["dialogue"]["dialogue_hash"] and
            [row["turn_ref"] for row in expanded["turns"][:62]] ==
            [row["turn_id"] for row in parent["dialogue"]["turns"]]
        )
        checks["turn_budget_contract_unchanged"] = (
            SCENE_HARD_MINIMUM == 9 and SCENE_HARD_MAXIMUM == 48 and
            compiled["round_turn_guidance"] == {
                "soft_target": 48, "soft_range": [42, 54],
                "hard_minimum": 27, "hard_maximum": 66,
            }
        )
        delta = _fixture_delta(compiled, child)
        delta_validation = validate_delta_updater_output(
            delta, chain_set=chain_set, compiled_plan=compiled, child=child)
        checks["delta_fixture_valid"] = delta_validation["valid"]
        parent_evidence = json.loads(json.dumps(delta))
        parent_evidence["chain_updates"][0]["evidence_turn_refs"] = [
            parent["dialogue"]["turns"][0]["turn_id"]]
        checks["delta_parent_evidence_rejected"] = not validate_delta_updater_output(
            parent_evidence, chain_set=chain_set,
            compiled_plan=compiled, child=child)["valid"]
        unknown_update = json.loads(json.dumps(delta))
        unknown_update["chain_updates"][0]["chain_id"] = "EC9999"
        checks["delta_unknown_chain_rejected"] = not validate_delta_updater_output(
            unknown_update, chain_set=chain_set,
            compiled_plan=compiled, child=child)["valid"]
        wrong_scene = json.loads(json.dumps(delta))
        wrong_scene["chain_updates"][0]["evidence_turn_refs"] = [
            child["scenes"][1]["dialogue_turns"][0]["turn_ref"]]
        checks["delta_cross_scene_evidence_rejected"] = not validate_delta_updater_output(
            wrong_scene, chain_set=chain_set,
            compiled_plan=compiled, child=child)["valid"]
        before_unused = json.loads(json.dumps(chain_set["chains"][3]))
        before_history_lengths = [
            len(row["history"]) for row in chain_set["chains"][:3]]
        updated = apply_delta_updates(
            delta, chain_set=chain_set, compiled_plan=compiled, child=child)
        checks["delta_history_appended"] = all(
            len(row["history"]) == before + 1
            for row, before in zip(updated["chains"][:3], before_history_lengths))
        checks["unused_chain_unchanged"] = updated["chains"][3] == before_unused
        prompts["delta_updater"] = build_delta_updater_prompt(
            chain_set=chain_set, compiled_plan=compiled, child=child)
        checks["delta_prompt_excludes_planner_claims"] = all(
            scene["new_development"] not in prompts["delta_updater"]["prompt"] and
            scene["durable_outcome"] not in prompts["delta_updater"]["prompt"]
            for scene in compiled["scenes"])
        checks["anchor_contract_removed"] = all(
            token not in prompts["planner"]["prompt"]
            for token in ("primary_anchor_span", "source_anchor_positions", "session_ref")
        )
        budgets = {
            "initializer": INITIALIZER_MAX_TOKENS,
            "planner": PLANNER_MAX_TOKENS,
            "realizer": ROUND_REALIZER_MAX_TOKENS,
            "delta_updater": DELTA_UPDATER_MAX_TOKENS,
        }
        for role, prompt in prompts.items():
            checks[f"{role}_prompt_fits"] = (
                counter(prompt["prompt"]) + budgets[role] <= CONTEXT_LIMIT)
    except Exception as error:
        details["chain_error"] = f"{type(error).__name__}:{error}"
        expected = (
            "authoritative_parent_valid", "context_selection_unchanged",
            "memorax_overview_contract_unchanged", "initializer_fixture_valid",
            "initializer_canary_semantic_checkpoints_valid",
            "initializer_empty_current_state_rejected", "initializer_missing_ref_rejected",
            "initializer_child_ref_rejected", "initializer_cross_window_ref_rejected",
            "deterministic_chain_ids", "soccer_continuation_linked",
            "concrete_events_remain_independent", "coverage_word_form_normalized",
            "planner_fixture_valid", "planner_continuation_fixture_valid",
            "three_distinct_chain_ids", "planner_unknown_chain_rejected",
            "planner_duplicate_chain_rejected", "realizer_receives_frozen_current_state",
            "round_realizer_fixture_valid", "append_only_assembly_valid",
            "append_only_contract_unchanged", "turn_budget_contract_unchanged",
            "delta_fixture_valid", "delta_parent_evidence_rejected",
            "delta_unknown_chain_rejected", "delta_cross_scene_evidence_rejected",
            "delta_history_appended", "unused_chain_unchanged",
            "delta_prompt_excludes_planner_claims", "anchor_contract_removed",
            "initializer_prompt_fits", "planner_prompt_fits", "realizer_prompt_fits",
            "delta_updater_prompt_fits",
        )
        for name in expected:
            checks.setdefault(name, False)
    config = None
    try:
        config = replace(
            load_challenger_config(api_config_path),
            timeout_seconds=LIVE_TIMEOUT_SECONDS)
        checks["model_configured"] = bool(config.model)
        checks["timeout_frozen_600"] = config.timeout_seconds == 600
    except Exception as error:
        checks["model_configured"] = False
        checks["timeout_frozen_600"] = False
        details["api_config_error"] = f"{type(error).__name__}:{error}"
    manifest = source_manifest()
    failed = sorted(key for key, value in checks.items() if not value)
    return {
        "preflight_version": PREFLIGHT_VERSION,
        "decision": "pass" if not failed else "fail",
        "checks": checks,
        "failed_checks": failed,
        "details": details,
        "input_path": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path) if input_path.is_file() else None,
        "overview_path": str(overview_path.resolve()),
        "overview_sha256": sha256_file(overview_path) if overview_path.is_file() else None,
        "run_root": str(run_root.resolve()),
        "root_id": parent.get("root_id") if parent else None,
        "api_config_path": str(api_config_path.resolve()),
        "api_config_sha256": sha256_file(api_config_path) if api_config_path.is_file() else None,
        "tokenizer_path": str(tokenizer_path.resolve()),
        "api_configuration": ({
            "model": config.model, "base_url": config.base_url,
            "temperature": config.temperature,
            "timeout_seconds": config.timeout_seconds,
            "api_key_recorded": False,
        } if config is not None else None),
        "source_manifest": manifest,
        "protected_integrity": protected_integrity_snapshot(),
        "prompt_manifests": {
            role: {key: value for key, value in prompt.items() if key != "prompt"}
            for role, prompt in prompts.items()
        },
        "request_budget": {
            "chain_initializer": 1, "planner": 1, "round_realizer": 1,
            "delta_updater": 1, "total": 4, "retries": 0,
            "memorax": 0, "qa": 0, "solver": 0,
        },
        "max_completion_tokens": {
            "chain_initializer": INITIALIZER_MAX_TOKENS,
            "planner": PLANNER_MAX_TOKENS,
            "round_realizer": ROUND_REALIZER_MAX_TOKENS,
            "delta_updater": DELTA_UPDATER_MAX_TOKENS,
        },
        "api_calls": 0,
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Incremental Event Chain Hard Preflight", "",
        f"- Decision: `{report['decision']}`",
        f"- API calls: `{report['api_calls']}`",
        f"- Run root: `{report['run_root']}`", "", "## Checks", "",
    ]
    lines.extend(
        f"- {key}: `{str(value).lower()}`" for key, value in report["checks"].items())
    lines.extend(["", "No run root or outbound request was created.", ""])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--overview", type=Path, default=DEFAULT_OVERVIEW)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    args = parser.parse_args(argv)
    report = run_preflight(
        input_path=args.input.resolve(), overview_path=args.overview.resolve(),
        run_root=args.run_root.resolve(), api_config_path=args.api_config.resolve(),
        tokenizer_path=args.tokenizer.resolve())
    atomic_json(args.report_root / "INCREMENTAL_EVENT_CHAIN_PREFLIGHT.json", report)
    atomic_text(
        args.report_root / "INCREMENTAL_EVENT_CHAIN_PREFLIGHT.md", _markdown(report))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["decision"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PREFLIGHT_VERSION", "_markdown", "run_preflight"]
