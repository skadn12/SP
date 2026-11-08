"""Zero-API replay for person-bucket event binding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .contracts import (
    build_person_event_candidates,
    materialize_memory_only_tracks,
)
from .memory_context import compile_memory_overview
from .prompts import build_planner_prompt


DEFAULT_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_216742_"
    "memory_event_continuation_fallback_pool_v1_live_1")


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def replay(run_root: Path) -> dict[str, Any]:
    raw_path = run_root / "memory" / "pre_generation_bucket_snapshot.json"
    if not raw_path.is_file():
        raise FileNotFoundError(str(raw_path))
    snapshot = _load(raw_path)
    overview = compile_memory_overview(snapshot)
    candidates = build_person_event_candidates(overview)
    tracks = materialize_memory_only_tracks(candidates)
    parent_path = run_root / "parent_context.json"
    parent = _load(parent_path) if parent_path.is_file() else {
        "dialogue": {"turns": []}}
    if "dialogue" not in parent:
        parent = {**parent, "dialogue": {
            "turns": parent.get("selected_turns") or []}}
    prompt = build_planner_prompt(
        parent=parent, memory_overview=overview, candidates=candidates,
        verified_tracks=tracks, memory_only_fast=True,
        strict_speaker_refs=True)
    event_candidates = [
        row for row in candidates
        if row.get("source_type") == "event_bucket"]
    person_candidates = [
        row for row in candidates
        if row.get("source_type") == "person_bucket_event"]
    return {
        "report_version": "person_bucket_event_binding_replay_v1",
        "api_calls": 0,
        "run_root": str(run_root),
        "raw_snapshot_sha256": snapshot.get("snapshot_sha256"),
        "overview_sha256": overview.get("memory_overview_sha256"),
        "overview_version": overview.get("memory_overview_version"),
        "event_bucket_candidate_count": len(event_candidates),
        "person_bucket_candidate_count": len(person_candidates),
        "selectable_candidate_count": len(candidates),
        "duplicate_memory_event_keys": len(candidates) - len({
            str(row.get("memory_event_key") or "").casefold()
            for row in candidates}),
        "person_event_fact_count": sum(
            len(row.get("person_event_facts") or []) for row in candidates),
        "candidates": candidates,
        "tracks": tracks,
        "planner_prompt_version": prompt.get("prompt_version"),
        "planner_prompt_character_count": len(prompt.get("prompt") or ""),
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Person Bucket Event Binding Replay",
        "",
        "Zero-API deterministic replay. The historical run and frozen bucket "
        "snapshot were read-only; no Memorax, Planner, Realizer, or Solver "
        "calls were made.",
        "",
        f"- Event-bucket candidates: `{report['event_bucket_candidate_count']}`",
        f"- Person-only candidates: `{report['person_bucket_candidate_count']}`",
        f"- Selectable candidates: `{report['selectable_candidate_count']}`",
        f"- Duplicate candidate keys: `{report['duplicate_memory_event_keys']}`",
        f"- Bound named person event facts: `{report['person_event_fact_count']}`",
        f"- Planner prompt version: `{report['planner_prompt_version']}`",
        f"- API calls: `{report['api_calls']}`",
        "",
        "## Candidates",
        "",
        "| Ref | Source | Memory key | Named person event facts | Supporting source types |",
        "|---|---|---|---:|---|",
    ]
    for row in report["candidates"]:
        facts = row.get("person_event_facts") or []
        sources = sorted({str(source.get("source_type") or "")
                          for source in row.get("supporting_sources") or []})
        lines.append(
            f"| `{row.get('memory_event_id')}` | `{row.get('source_type')}` | "
            f"`{row.get('memory_event_key')}` | {len(facts)} | "
            f"{', '.join(f'`{source}`' for source in sources)} |")
        for fact in facts:
            lines.append(
                f"|  |  |  | `{fact.get('person_display')}` / "
                f"`{fact.get('event_tag')}` | {fact.get('fact_text')} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "Selectable continuation identities come only from person-bucket event "
        "facts. Facts from different people are conservatively merged when "
        "their concrete local event wording overlaps. Event buckets remain in "
        "the frozen snapshot for audit but are not materialized into Planner "
        "candidates. The binding does not infer a latest state or replace the "
        "Parent transcript.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "reports")
    args = parser.parse_args()
    report = replay(args.run_root)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "PERSON_BUCKET_EVENT_BINDING_REPLAY.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8")
    (args.output / "PERSON_BUCKET_EVENT_BINDING_REPLAY.md").write_text(
        markdown(report), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "event_bucket_candidate_count", "person_bucket_candidate_count",
        "selectable_candidate_count", "duplicate_memory_event_keys",
        "person_event_fact_count", "api_calls")}, ensure_ascii=False,
        indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
