"""Zero-API hard preflight for latest-state-aware Expansion."""

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
from ..simple_append_v1.assembly import (
    assemble_expanded_dialogue, validate_expanded_dialogue,
)
from ..simple_append_v1.contracts import (
    SCENE_HARD_MAXIMUM, SCENE_HARD_MINIMUM,
)
from .contracts import (
    PLANNER_SCHEMA_VERSION, RESOLVER_SCHEMA_VERSION,
    compile_round_plan, materialize_resolved_developments,
    normalize_round_realizer_output, validate_planner_continuation,
    validate_planner_output, validate_resolver_output,
)
from .prompts import (
    build_planner_prompt, build_round_realizer_prompt,
    build_state_resolver_prompt,
)
from .settings import (
    CONTEXT_LIMIT, DEFAULT_API_CONFIG, DEFAULT_INPUT, DEFAULT_OVERVIEW,
    DEFAULT_REPORT_ROOT, DEFAULT_RUN_ROOT, DEFAULT_TOKENIZER,
    FROZEN_INPUT_SHA256, FROZEN_OVERVIEW_SHA256, LIVE_TIMEOUT_SECONDS,
    PLANNER_MAX_TOKENS, RESOLVER_MAX_TOKENS, ROUND_REALIZER_MAX_TOKENS,
    protected_integrity_snapshot, source_manifest,
)


PREFLIGHT_VERSION = "simple_append_state_resolved_expansion_preflight_v1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected_object:{path}")
    return value


def _fixture_resolver(parent: Mapping[str, Any]) -> dict[str, Any]:
    turns = list(parent["dialogue"]["turns"])
    speakers = list(dict.fromkeys(str(row["speaker"]) for row in turns))
    return {
        "resolver_schema_version": RESOLVER_SCHEMA_VERSION,
        "developments": [
            {
                "event_summary": "soccer practice decision",
                "participants": speakers[:2],
                "latest_state": "A and B have already agreed that regular practice should resume.",
                "evidence_turn_refs": [turns[5]["turn_id"], turns[6]["turn_id"]],
            },
            {
                "event_summary": "health and work coverage",
                "participants": speakers[:2],
                "latest_state": "A planned a doctor visit and B already agreed to cover work.",
                "evidence_turn_refs": [turns[37]["turn_id"], turns[38]["turn_id"]],
            },
            {
                "event_summary": "report mistake disclosure",
                "participants": speakers[:2],
                "latest_state": "A intended to notify the boss and take responsibility.",
                "evidence_turn_refs": [turns[49]["turn_id"], turns[55]["turn_id"]],
            },
        ],
    }


def _fixture_planner(resolved: Mapping[str, Any]) -> dict[str, Any]:
    tracks = list(resolved["developments"])
    return {
        "planner_schema_version": PLANNER_SCHEMA_VERSION,
        "round_goal": "Advance three established developments into distinct consequences.",
        "scenes": [
            {
                "development_id": tracks[0]["development_id"],
                "goal": "Show the first resumed practice and a new practical obstacle.",
                "participants": list(tracks[0]["participants"]),
                "new_development": "A and B begin practice and discover the field is unavailable.",
                "interaction_beats": ["arrive to practice", "discover closure", "make a new plan"],
                "transition_from_previous": "Continue after the Parent tail.",
                "desired_outcome": "They secure a different practice location.",
                "durable_outcome": "A and B book a different field for the next practice.",
            },
            {
                "development_id": tracks[1]["development_id"],
                "goal": "Show the concrete result of the doctor and coverage plan.",
                "participants": list(tracks[1]["participants"]),
                "new_development": "A reports the doctor's instruction while B confirms the completed shift.",
                "interaction_beats": ["share medical guidance", "confirm coverage", "set return condition"],
                "transition_from_previous": "Shift to the next workday.",
                "desired_outcome": "A has a concrete recovery and return arrangement.",
                "durable_outcome": "A will remain off work until the doctor-cleared return date.",
            },
            {
                "development_id": tracks[2]["development_id"],
                "goal": "Move from intended disclosure to the boss's concrete response.",
                "participants": list(tracks[2]["participants"]),
                "new_development": "A receives the boss's response and is assigned a correction deadline.",
                "interaction_beats": ["relay the response", "clarify correction", "accept deadline"],
                "transition_from_previous": "Return to the unresolved report issue.",
                "desired_outcome": "The report correction has an owner and deadline.",
                "durable_outcome": "A must submit the corrected report by the agreed deadline.",
            },
        ],
    }


def _fixture_realizer(compiled: Mapping[str, Any]) -> dict[str, Any]:
    dialogues = {}
    for scene_index, scene in enumerate(compiled["scenes"], 1):
        participants = list(scene["participants"])
        dialogues[scene["scene_ref"]] = [{
            "speaker": participants[index % len(participants)],
            "text": f"Offline fixture scene {scene_index} progression turn {index + 1}.",
        } for index in range(9)]
    return {
        "realizer_schema_version": "simple_append_round_realizer_v2",
        "scene_dialogues": dialogues,
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
        "api_config_exists": api_config_path.is_file(),
        "tokenizer_exists": tokenizer_path.exists(),
        "input_sha_frozen": input_path.is_file() and sha256_file(input_path) == FROZEN_INPUT_SHA256,
        "overview_sha_frozen": overview_path.is_file() and sha256_file(overview_path) == FROZEN_OVERVIEW_SHA256,
        "api_calls_zero": True,
        "memorax_calls_zero": True,
        "qa_solver_calls_zero": True,
    }
    details: dict[str, Any] = {}
    resolver_prompt: dict[str, Any] = {}
    planner_prompt: dict[str, Any] = {}
    realizer_prompt: dict[str, Any] = {}
    parent: dict[str, Any] = {}
    overview: dict[str, Any] = {}
    tail: dict[str, Any] = {}
    resolved: dict[str, Any] = {}
    compiled: dict[str, Any] = {}
    try:
        if token_counter is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, local_files_only=True)
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
        resolver_prompt = build_state_resolver_prompt(
            tail_context=tail, memory_overview=overview)
        resolver_fixture = _fixture_resolver(parent)
        resolver_validation = validate_resolver_output(
            resolver_fixture, parent_turns=parent["dialogue"]["turns"])
        checks["resolver_fixture_valid"] = resolver_validation["valid"]
        resolved = materialize_resolved_developments(
            resolver_fixture, parent_turns=parent["dialogue"]["turns"],
            root_id=parent["root_id"],
            parent_dialogue_sha256=parent["dialogue"]["dialogue_hash"])
        checks["deterministic_development_ids"] = [
            row["development_id"] for row in resolved["developments"]
        ] == ["D01", "D02", "D03"]
        planner_prompt = build_planner_prompt(
            resolved_set=resolved, parent_speakers=tail["parent_speakers"])
        planner_fixture = _fixture_planner(resolved)
        planner_validation = validate_planner_output(
            planner_fixture, resolved_set=resolved,
            parent_speakers=tail["parent_speakers"])
        continuation = validate_planner_continuation(
            planner_fixture, resolved_set=resolved)
        checks["planner_fixture_valid"] = planner_validation["valid"]
        checks["planner_continuation_fixture_valid"] = continuation["valid"]
        checks["three_distinct_development_ids"] = (
            len(set(planner_validation["development_ids"])) == 3)
        compiled = compile_round_plan(
            planner_fixture, resolved_set=resolved, tail_context=tail)
        realizer_prompt = build_round_realizer_prompt(
            compiled_plan=compiled, tail_context=tail)
        checks["realizer_receives_frozen_latest_state"] = all(
            row["latest_state"] in realizer_prompt["prompt"]
            for row in resolved["developments"][:3])
        normalized = normalize_round_realizer_output(
            _fixture_realizer(compiled), compiled_plan=compiled,
            parent_utterances=[row["text"] for row in parent["dialogue"]["turns"]],
            parent_turn_refs=[row["turn_id"] for row in parent["dialogue"]["turns"]])
        checks["round_realizer_fixture_valid"] = normalized["valid"]
        expanded = assemble_expanded_dialogue(parent=parent, child=normalized["child"])
        expanded_validation = validate_expanded_dialogue(expanded, parent=parent)
        checks["append_only_assembly_valid"] = expanded_validation["valid"]
        checks["append_only_contract_unchanged"] = (
            expanded.get("append_mode") == "parent_tail_only" and
            expanded.get("parent_dialogue_sha256") ==
            parent["dialogue"]["dialogue_hash"] and
            expanded_validation.get("parent_turn_count") ==
            parent["dialogue"]["turn_count"] and
            [row["turn_ref"] for row in expanded["turns"][:62]] ==
            [row["turn_id"] for row in parent["dialogue"]["turns"]])
        checks["turn_budget_contract_unchanged"] = (
            SCENE_HARD_MINIMUM == 9 and SCENE_HARD_MAXIMUM == 48 and
            compiled["round_turn_guidance"] == {
                "soft_target": 48, "soft_range": [42, 54],
                "hard_minimum": 27, "hard_maximum": 66,
            }
        )
        checks["anchor_contract_removed"] = all(
            token not in planner_prompt["prompt"]
            for token in ("primary_anchor_span", "source_anchor_positions", "session_ref")
        )
        checks["resolver_prompt_fits"] = (
            counter(resolver_prompt["prompt"]) + RESOLVER_MAX_TOKENS <= CONTEXT_LIMIT)
        checks["planner_prompt_fits"] = (
            counter(planner_prompt["prompt"]) + PLANNER_MAX_TOKENS <= CONTEXT_LIMIT)
        checks["realizer_prompt_fits"] = (
            counter(realizer_prompt["prompt"]) + ROUND_REALIZER_MAX_TOKENS <= CONTEXT_LIMIT)
    except Exception as error:
        details["chain_error"] = f"{type(error).__name__}:{error}"
        for name in (
            "authoritative_parent_valid", "context_selection_unchanged",
            "memorax_overview_contract_unchanged", "resolver_fixture_valid",
            "deterministic_development_ids", "planner_fixture_valid",
            "planner_continuation_fixture_valid", "three_distinct_development_ids",
            "realizer_receives_frozen_latest_state", "round_realizer_fixture_valid",
            "append_only_assembly_valid", "append_only_contract_unchanged",
            "turn_budget_contract_unchanged", "anchor_contract_removed",
            "resolver_prompt_fits", "planner_prompt_fits", "realizer_prompt_fits",
        ):
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
            "resolver": {key: value for key, value in resolver_prompt.items() if key != "prompt"},
            "planner": {key: value for key, value in planner_prompt.items() if key != "prompt"},
            "realizer": {key: value for key, value in realizer_prompt.items() if key != "prompt"},
        },
        "request_budget": {
            "state_resolver": 1, "planner": 1, "round_realizer": 1,
            "total": 3, "retries": 0, "memorax": 0, "qa": 0, "solver": 0,
        },
        "max_completion_tokens": {
            "state_resolver": RESOLVER_MAX_TOKENS,
            "planner": PLANNER_MAX_TOKENS,
            "round_realizer": ROUND_REALIZER_MAX_TOKENS,
        },
        "api_calls": 0,
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# State-resolved Expansion Hard Preflight", "",
        f"- Decision: `{report['decision']}`",
        f"- API calls: `{report['api_calls']}`",
        f"- Run root: `{report['run_root']}`", "", "## Checks", "",
    ]
    lines.extend(
        f"- {key}: `{str(value).lower()}`"
        for key, value in report["checks"].items())
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
    atomic_json(args.report_root / "STATE_RESOLVED_EXPANSION_PREFLIGHT.json", report)
    atomic_text(args.report_root / "STATE_RESOLVED_EXPANSION_PREFLIGHT.md", _markdown(report))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["decision"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
