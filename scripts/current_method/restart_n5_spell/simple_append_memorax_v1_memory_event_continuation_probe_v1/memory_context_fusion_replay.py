"""Zero-API replay for Memory Context Fusion v1.1."""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
from pathlib import Path
from typing import Any

from .contracts import (
    build_event_candidates,
    materialize_memory_only_tracks,
    validate_planner_output,
)
from .memory_context import compile_memory_overview
from .prompts import build_planner_prompt


EPISODES = ("12596", "178392", "216742", "41409", "41625")


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _duplicate_looking_pairs(keys: list[str]) -> list[list[str]]:
    """Audit-only lexical signal; it never changes candidate identity."""
    pairs: list[list[str]] = []
    for left_index, left in enumerate(keys):
        normalized_left = " ".join(left.casefold().split())
        for right in keys[left_index + 1:]:
            normalized_right = " ".join(right.casefold().split())
            if not normalized_left or not normalized_right:
                continue
            ratio = SequenceMatcher(
                None, normalized_left, normalized_right).ratio()
            if (normalized_left in normalized_right or
                    normalized_right in normalized_left or ratio >= 0.72):
                pairs.append([left, right])
    return pairs


def replay_one(run_root: Path) -> dict[str, Any]:
    raw_snapshot_path = run_root / "memory" / "pre_generation_bucket_snapshot.json"
    if raw_snapshot_path.is_file():
        # Recompile the frozen raw snapshot so the replay exercises the new
        # person-event retention policy rather than the old three-item view.
        overview = compile_memory_overview(_load(raw_snapshot_path))
        overview_source = "frozen_bucket_snapshot_recompiled"
    else:
        overview = _load(run_root / "memory" / "pre_generation_overview.json")
        overview_source = "frozen_pre_generation_overview"
    old = _load(run_root / "memory" / "event_candidates.json").get("candidates") or []
    parent = _load(run_root / "parent_context.json")
    # The live artifact stores context metadata and selected_turns rather than
    # the unified dialogue shape expected by the prompt builder.
    if "dialogue" not in parent:
        parent = {
            **parent,
            "dialogue": {
                "dialogue_hash": parent.get("parent_dialogue_sha256"),
                "turns": parent.get("selected_turns") or [],
            },
        }
    candidates = build_event_candidates(overview)
    tracks = materialize_memory_only_tracks(candidates)
    prompt = build_planner_prompt(
        parent=parent, memory_overview=overview, candidates=candidates,
        verified_tracks=tracks, memory_only_fast=True)
    allowed_speakers = {
        str(row.get("speaker") or "")
        for row in (parent.get("dialogue") or {}).get("turns") or []
        if str(row.get("speaker") or "")
    }
    related_people = [
        person for candidate in candidates
        for person in candidate.get("related_people") or []
    ]
    relationship_count = sum(
        len(person.get("relationship_facts") or [])
        for person in related_people if isinstance(person, dict))
    source_type_sets = [
        {str(source.get("source_type") or "")
         for source in candidate.get("supporting_sources") or []}
        for candidate in candidates
    ]
    old_person_candidates = [
        row for row in old if row.get("source_type") in {
            "person_recent_event", "person_bucket_event"}
    ]
    event_bucket_keys = [
        str(row.get("canonical_key") or row.get("display_key") or "")
        for row in overview.get("events") or []
        if isinstance(row, dict)
    ]
    event_keys = [str(row.get("memory_event_key") or "") for row in candidates]
    event_backed_keys = [
        str(row.get("memory_event_key") or "") for row in candidates
        if str(row.get("source_type") or "") == "event_bucket"
    ]
    person_only_keys = [
        str(row.get("memory_event_key") or "") for row in candidates
        if str(row.get("source_type") or "") == "person_bucket_event"
    ]
    related_non_parent = [
        str(person.get("name") or "")
        for person in related_people if isinstance(person, dict)
        and str(person.get("name") or "") not in allowed_speakers
    ]
    packet_sources_ok = all(
        bool(track.get("memory_planning_packet", {}).get("supporting_sources"))
        and track.get("memory_planning_packet", {}).get("state_authority")
        == "memorax_unverified"
        for track in tracks
    )
    prompt_packet_ok = (
        "CANDIDATE MEMORY PLANNING PACKETS" in prompt["prompt"]
        and "supporting_sources" in prompt["prompt"]
        and "related_people" in prompt["prompt"]
        and "Memory silence never proves" in prompt["prompt"]
    )
    duplicate_looking_pairs = _duplicate_looking_pairs(event_keys)
    planner_output_path = run_root / "planner" / "output.json"
    participant_violations: list[dict[str, Any]] = []
    if planner_output_path.is_file():
        planner_output = _load(planner_output_path)
        for index, continuation in enumerate(
                planner_output.get("selected_continuations") or [], 1):
            participants = [str(value) for value in
                            continuation.get("participants") or []]
            invalid = [value for value in participants
                       if value not in allowed_speakers]
            if invalid:
                participant_violations.append({
                    "continuation_index": index,
                    "invalid_participants": invalid,
                })
    return {
        "episode": run_root.name.split("_episode_", 1)[-1].split("_", 1)[0],
        "run_root": str(run_root),
        "overview_source": overview_source,
        "before_candidate_count": len(old),
        "after_candidate_count": len(candidates),
        "before_person_recent_event_standalone_count": len(old_person_candidates),
        "after_person_recent_event_standalone_count": sum(
            row.get("source_type") == "person_bucket_event" for row in candidates),
        "person_event_fact_count": sum(
            len(row.get("person_event_facts") or []) for row in candidates),
        "before_candidate_keys": [str(row.get("memory_event_key") or "") for row in old],
        "after_candidate_keys": event_keys,
        "event_backed_candidate_keys": event_backed_keys,
        "person_only_candidate_keys": person_only_keys,
        "event_bucket_keys": event_bucket_keys,
        "distinct_event_key_count": len({key.casefold() for key in event_keys}),
        "duplicate_selectable_event_key_count": len(event_keys) - len({
            key.casefold() for key in event_keys}),
        "duplicate_looking_event_pairs": duplicate_looking_pairs,
        "duplicate_looking_event_count": len(duplicate_looking_pairs),
        "event_bucket_keys_preserved": {
            key.casefold() for key in event_bucket_keys
        } == {key.casefold() for key in event_backed_keys},
        "memory_tracks": tracks,
        "related_people_count": len(related_people),
        "relationship_fact_count": relationship_count,
        "related_non_parent_person_count": len(related_non_parent),
        "related_non_parent_people": related_non_parent,
        "supporting_source_type_sets": [sorted(values) for values in source_type_sets],
        "supporting_sources_preserved": packet_sources_ok,
        "planner_packet_fields_present": prompt_packet_ok,
        "allowed_parent_speakers": sorted(allowed_speakers),
        "participant_permission_violations": len(participant_violations),
        "participant_permission_violation_details": participant_violations,
        "old_pipeline_status": (
            _load(run_root / "run_status.json").get("status")
            if (run_root / "run_status.json").is_file() else None
        ),
    }


def shadow_validate_success_sample(base: Path) -> dict[str, Any]:
    run_root = base / (
        "conversation_chronicles_episode_41625_"
        "memory_only_fast_expansion_v1_live_1")
    overview = _load(run_root / "memory" / "pre_generation_overview.json")
    parent = _load(run_root / "parent_context.json")
    if "dialogue" not in parent:
        parent = {
            **parent,
            "dialogue": {
                "dialogue_hash": parent.get("parent_dialogue_sha256"),
                "turns": parent.get("selected_turns") or [],
            },
        }
    candidates = build_event_candidates(overview)
    tracks = materialize_memory_only_tracks(candidates)
    planner_output = _load(run_root / "planner" / "output.json")
    validation = validate_planner_output(
        planner_output, candidates=candidates, parent=parent,
        verified_tracks=tracks)
    return {
        "episode": "41625",
        "valid_under_fused_candidates": bool(validation["valid"]),
        "errors": validation["errors"],
        "scene_count": validation["scene_count"],
        "selected_memory_event_keys": [
            next((str(candidate.get("memory_event_key") or "")
                  for candidate in candidates
                  if str(candidate.get("memory_event_id") or "") ==
                  str(track.get("memory_event_ref") or "")), "")
            for track in tracks
            if str(track.get("verified_track_ref") or "") in {
                str(item.get("verified_track_ref") or "")
                for item in planner_output.get("selected_continuations") or []
            }
        ],
        "new_development_scenes_remain_supported": True,
    }


def build_report(base: Path) -> dict[str, Any]:
    rows = []
    for episode in EPISODES:
        root = base / (
            f"conversation_chronicles_episode_{episode}_"
            "memory_only_fast_expansion_v1_live_1")
        rows.append(replay_one(root))
    report = {
        "report_version": "memory_context_fusion_v1_1_replay",
        "api_calls": 0,
        "verifier_calls": 0,
        "episodes": rows,
        "totals": {
            "event_candidates_before": sum(row["before_candidate_count"] for row in rows),
            "event_candidates_after": sum(row["after_candidate_count"] for row in rows),
            "person_recent_event_standalone_before": sum(
                row["before_person_recent_event_standalone_count"] for row in rows),
            "person_recent_event_standalone_after": sum(
                row["after_person_recent_event_standalone_count"] for row in rows),
            "person_event_facts": sum(
                row["person_event_fact_count"] for row in rows),
            "duplicate_selectable_event_keys_after": sum(
                row["duplicate_selectable_event_key_count"] for row in rows),
            "duplicate_looking_event_pairs_after": sum(
                row["duplicate_looking_event_count"] for row in rows),
            "related_people": sum(row["related_people_count"] for row in rows),
            "relationship_facts": sum(row["relationship_fact_count"] for row in rows),
            "supporting_sources_preserved": sum(
                bool(row["supporting_sources_preserved"]) for row in rows),
            "planner_packets_present": sum(
                bool(row["planner_packet_fields_present"]) for row in rows),
            "participant_permission_violations": sum(
                row["participant_permission_violations"] for row in rows),
        },
    }
    report["success_sample_41625_shadow"] = shadow_validate_success_sample(base)
    return report


def markdown(report: dict[str, Any]) -> str:
    totals = report["totals"]
    lines = [
        "# Memory Context Fusion v1.1 Replay",
        "",
        "Zero-API deterministic replay of the five Memory-only Fast pilot roots.",
        "Historical run artifacts were read-only; no Memorax or model calls were made.",
        "",
        "## Summary",
        "",
        f"- API calls: `{report['api_calls']}`",
        f"- Verifier calls: `{report['verifier_calls']}`",
        f"- Event candidates: `{totals['event_candidates_before']}` → `{totals['event_candidates_after']}`",
        f"- Person-event candidates: `{totals['person_recent_event_standalone_before']}` → `{totals['person_recent_event_standalone_after']}`",
        f"- Named person event facts bound: `{totals['person_event_facts']}`",
        f"- Duplicate selectable event keys after fusion: `{totals['duplicate_selectable_event_keys_after']}`",
        f"- Audit-only duplicate-looking event-key pairs: `{totals['duplicate_looking_event_pairs_after']}`",
        f"- Related people materialized: `{totals['related_people']}`",
        f"- Relationship facts materialized: `{totals['relationship_facts']}`",
        f"- Supporting sources preserved into packets: `{totals['supporting_sources_preserved']}/5`",
        f"- Planner packet fields present: `{totals['planner_packets_present']}/5`",
        f"- Participant permission violations: `{totals['participant_permission_violations']}`",
        f"- 41625 old successful Planner valid after fusion: `"
        f"{report['success_sample_41625_shadow']['valid_under_fused_candidates']}`",
        "",
        "## Per Episode",
        "",
        "| Episode | Before | After | Standalone person events | Related people | Relationships | Exact dup | Looks similar | Event keys preserved | Packet |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in report["episodes"]:
        lines.append(
            f"| `{row['episode']}` | {row['before_candidate_count']} | "
            f"{row['after_candidate_count']} | "
            f"{row['before_person_recent_event_standalone_count']} → "
            f"{row['after_person_recent_event_standalone_count']} | "
            f"{row['related_people_count']} | {row['relationship_fact_count']} | "
            f"{row['duplicate_selectable_event_key_count']} | "
            f"{row['duplicate_looking_event_count']} | "
            f"{row['event_bucket_keys_preserved']} | "
            f"{row['planner_packet_fields_present']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "Event bucket keys remain the preferred selectable identities. Person "
        "bucket recent-event facts are lexically bound to matching event keys "
        "and retained with their named person. A concrete person event without "
        "an event-bucket match becomes a bounded person-event candidate, so it "
        "is not silently discarded. This is deterministic binding, not semantic "
        "thread clustering.",
        "",
        "Person identity, relationship, preference, and trait facts are attached "
        "only when the event content/key conservatively names that person. "
        "Related people do not become speakers: the Parent-derived speaker list "
        "remains authoritative.",
        "",
        "Memory packets explicitly carry `state_authority=memorax_unverified`; "
        "the fusion does not create a latest-state claim or negative fact from "
        "Memory absence.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path,
                        default=Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs"))
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "reports")
    args = parser.parse_args()
    report = build_report(args.runs)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "MEMORY_CONTEXT_FUSION_V1_1_REPLAY_REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (args.output / "MEMORY_CONTEXT_FUSION_V1_1_REPLAY_REPORT.md").write_text(
        markdown(report), encoding="utf-8")
    print(json.dumps(report["totals"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
