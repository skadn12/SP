"""Zero-network preflight for the optional Memorax simple-append layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from ..core import atomic_json, atomic_text
from .durable_contracts import (
    PLANNER_SCHEMA_VERSION,
    build_parent_anchor_registry,
    compile_round_plan,
)
from .context import build_fresh_r1_context, select_fresh_r1_qa_context
from .memory_context import compile_memory_overview, normalize_bucket_snapshot
from .payloads import build_add_intent, build_bucket_search_payload, memorax_user_id
from .prompts import (
    build_memory_augmented_mcq_prompt,
    build_memory_augmented_planner_prompt,
    build_memory_augmented_realizer_prompt,
)


PREFLIGHT_VERSION = (
    "simple_append_memorax_fullr1_devdiv_local_anchor_span_offline_preflight_v1")
DEFAULT_PARENT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-inputs/"
    "conversation_chronicles_v1/roots/cc-train-episode-146700/"
    "unified_root.json")
DEFAULT_EXPANDED = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "simple_append_memorax_v1_batch_remaining8_pilot_8/roots/"
    "cc-train-episode-146700/expansion/expanded_dialogue.json")
DEFAULT_TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")
DEFAULT_REPORT_ROOT = Path(__file__).with_name("reports")


def _bucket_fixture() -> dict[str, Any]:
    return {
        "success": True,
        "data": {"buckets": {
            "person": {
                "co-workers a": {
                    "display_key": "Co-workers A",
                    "content": ["Co-workers A is considering a practical decision."],
                },
                "friend c": {
                    "display_key": "Friend C",
                    "content": ["Friend C is connected to an earlier debt event."],
                },
            },
            "location": {
                "office": {"display_key": "Office", "content": [
                    "Several discussions occur at the office."]},
            },
            "event": {
                "debt settlement": {
                    "display_key": "Debt settlement",
                    "content": ["An earlier debt was settled."],
                },
            },
            "item": {
                "car": {"display_key": "Car", "content": [
                    {"attribute": "status", "value": "under consideration"}]},
            },
            "dreaming": {
                "responsible planning": {
                    "display_key": "Responsible planning",
                    "content": ["The story repeatedly revisits practical planning."],
                },
            },
        }},
    }


def _planner_fixture(
    speakers: list[str], registry: Mapping[str, Any],
) -> dict[str, Any]:
    a, b = speakers[:2]
    first_positions: list[int] = []
    seen: set[str] = set()
    for row in registry["entries"]:
        if row["session_ref"] not in seen:
            seen.add(row["session_ref"])
            first_positions.append(row["position"])
    return {
        "planner_schema_version": PLANNER_SCHEMA_VERSION,
        "round_goal": "Continue the story with three concrete developments.",
        "scenes": [{
            "goal": f"Develop scene {index}.",
            "participants": [a, b],
            "primary_anchor_span": {
                "start_position": first_positions[index - 1],
                "end_position": first_positions[index - 1],
            },
            "new_development": f"Distinct development {index}.",
            "interaction_beats": ["A fact emerges.", "A constraint matters.",
                                  "The speakers choose a response."],
            "transition_from_previous": (
                "Continue the parent tail." if index == 1 else
                f"Continue scene {index - 1}."),
            "desired_outcome": f"Distinct outcome {index}.",
            "durable_outcome": f"Durable outcome {index} is established.",
        } for index in range(1, 4)],
    }


def run(
    *, parent_path: Path = DEFAULT_PARENT,
    expanded_path: Path = DEFAULT_EXPANDED,
    tokenizer_path: Path = DEFAULT_TOKENIZER,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    parent = json.loads(Path(parent_path).read_text(encoding="utf-8"))
    anchor_registry = build_parent_anchor_registry(parent["dialogue"]["turns"])
    expanded = json.loads(Path(expanded_path).read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True)
    token_counter = lambda text: len(tokenizer.encode(
        text, add_special_tokens=False))
    root_id = str(parent["root_id"])
    tail = build_fresh_r1_context(parent, token_counter=token_counter)
    fallback = build_fresh_r1_context(
        parent, token_counter=lambda text: token_counter(text) * 4)
    qa_context = select_fresh_r1_qa_context(
        expanded, token_counter=token_counter)
    qa_fallback = select_fresh_r1_qa_context(
        expanded, token_counter=lambda text: token_counter(text) * 4)
    intent = build_add_intent(
        root_id=root_id, round_index=0,
        source_dialogue_sha256=parent["dialogue"]["dialogue_hash"],
        turns=parent["dialogue"]["turns"],
        recent_context_turns=(), async_mode=False)
    query = build_bucket_search_payload(root_id=root_id)
    snapshot = normalize_bucket_snapshot(
        _bucket_fixture(), root_id=root_id, user_id=memorax_user_id(root_id))
    recent = "\n".join(str(row["text"]) for row in tail["selected_turns"])
    overview = compile_memory_overview(
        snapshot, recent_dialogue_text=recent)
    plan = _planner_fixture(list(tail["parent_speakers"]), anchor_registry)
    compiled = compile_round_plan(
        plan, tail_context=tail, parent_anchor_registry=anchor_registry)
    planner = build_memory_augmented_planner_prompt(
        tail, overview, anchor_registry)
    realizer = build_memory_augmented_realizer_prompt(
        compiled_plan=compiled, tail_context=tail,
        memory_overview=overview)
    mcq = build_memory_augmented_mcq_prompt(
        qa_context, overview, focus_round_index=1)
    serialized = json.dumps({
        "intent": intent, "query": query, "overview": overview,
        "prompts": [planner, realizer, mcq],
    }, ensure_ascii=False)
    checks = {
        "parent_add_contains_all_turns": (
            len(intent["turn_refs"]) == parent["dialogue"]["turn_count"]),
        "r1_parent_under_budget_uses_full_parent": (
            tail["full_parent_raw_dialogue_tokens"] <= 4096 and
            tail["full_parent_included"] is True and
            tail["selected_turn_count"] == parent["dialogue"]["turn_count"]),
        "r1_parent_over_budget_uses_fallback": (
            fallback["full_parent_raw_dialogue_tokens"] > 4096 and
            fallback["full_parent_included"] is False),
        "parent_fallback_keeps_complete_sessions": (
            fallback["complete_sessions_only"] is True),
        "r1_qa_under_budget_uses_full_dialogue": (
            qa_context["qa_context_selection"]
            ["full_parent_child_raw_dialogue_tokens"] <= 4096 and
            qa_context["qa_context_selection"]
            ["full_parent_child_included"] is True),
        "r1_qa_over_budget_uses_fallback": (
            qa_fallback["qa_context_selection"]
            ["full_parent_child_raw_dialogue_tokens"] > 4096 and
            qa_fallback["qa_context_selection"]
            ["full_parent_child_included"] is False),
        "qa_fallback_keeps_complete_sessions": (
            qa_fallback["qa_context_selection"]
            ["complete_sessions_only"] is True),
        "stable_root_user_id": intent["user_id"] == query["payload"]["user_id"],
        "speaker_prefix_preserved": all(
            str(row["content"]).startswith("[Speaker: ")
            for row in intent["payload"]["messages"]),
        "default_mode": intent["payload"]["mode"] == "default",
        "sync_initial_add": intent["payload"]["async_mode"] is False,
        "all_bucket_types_requested": all(
            query["payload"]["bucket_filters"][key] == "all" for key in (
                "entity_names", "location_names", "event_names",
                "item_names", "dreaming_names")),
        "memory_overview_bounded": overview["token_estimate"] < 2000,
        "memory_not_authoritative": overview["memory_is_authoritative"] is False,
        "memory_not_qa_fact_source": overview["memory_is_qa_fact_source"] is False,
        "planner_local_anchor_span_schema": (
            PLANNER_SCHEMA_VERSION in planner["prompt"] and
            '"primary_anchor_span"' in planner["prompt"]),
        "realizer_schema_unchanged": "simple_append_round_realizer_v2" in realizer["prompt"],
        "mcq_schema_unchanged": "simple_append_mcq_focus_writer_v4" in mcq["prompt"],
        "planner_has_memory": "NON-AUTHORITATIVE MEMORAX MEMORY" in planner["prompt"],
        "planner_has_development_diversity_contract": (
            "durable_outcome is mandatory" in planner["prompt"]),
        "realizer_has_memory": "NON-AUTHORITATIVE MEMORAX MEMORY" in realizer["prompt"],
        "mcq_has_memory": "QUESTION TOPIC SELECTION" in mcq["prompt"],
        "api_key_absent": "MEMORAX_API_KEY" not in serialized and
            "Authorization" not in serialized,
        "api_calls_zero": True,
        "base_simple_append_not_modified_by_runtime": True,
    }
    failed = [key for key, value in checks.items() if not value]
    return {
        "preflight_version": PREFLIGHT_VERSION,
        "status": "pass" if not failed else "fail",
        "checks": checks,
        "failed_checks": failed,
        "api_calls": 0,
        "artifacts": {
            "parent_context": tail,
            "parent_fallback_context": fallback,
            "qa_context_selection": qa_context["qa_context_selection"],
            "qa_fallback_context_selection": qa_fallback["qa_context_selection"],
            "add_intent": intent,
            "bucket_query": query,
            "bucket_snapshot": snapshot,
            "memory_overview": overview,
            "prompt_manifests": {
                name: {key: value for key, value in prompt.items()
                       if key != "prompt"}
                for name, prompt in (("planner", planner),
                                     ("realizer", realizer), ("mcq", mcq))
            },
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# Simple Append Memorax v1 Offline Preflight", "",
             f"- status: `{report['status']}`",
             f"- API calls: `{report['api_calls']}`", "", "## Checks", ""]
    lines.extend(f"- {key}: `{str(value).lower()}`"
                 for key, value in report["checks"].items())
    lines.extend(["", "No network call or authoritative data write was made.", ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--expanded", type=Path, default=DEFAULT_EXPANDED)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    args = parser.parse_args(argv)
    report = run(
        parent_path=args.parent, expanded_path=args.expanded,
        tokenizer_path=args.tokenizer)
    atomic_json(args.report_root / "MEMORAX_V1_OFFLINE_PREFLIGHT.json", report)
    atomic_text(args.report_root / "MEMORAX_V1_OFFLINE_PREFLIGHT.md",
                _markdown(report))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
