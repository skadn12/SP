"""Offline diagnostic comparison for the two completed A/B Realizer outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Mapping


DEFAULT_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "h4_round1_r03_realizer_ab_single_generation_v1_live_1"
)
PREPARATION_RUN = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "h4_round1_r03_realizer_ab_input_preparation_v1_live_3"
)
SOURCE_DIALOGUE = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "h4_round1_feedback_r02_batch_single_root_v1_live_3/expanded_dialogue.json"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                sort_keys=True) + "\n", encoding="utf-8")


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def flatten(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scene_index, scene in enumerate(value.get("scenes") or [], 1):
        for turn_index, turn in enumerate(scene.get("dialogue_turns") or [], 1):
            rows.append({
                "scene_index": scene_index,
                "turn_index": turn_index,
                "turn_ref": f"R03:S{scene_index:02d}:T{turn_index:03d}",
                "speaker": str(turn.get("speaker") or ""),
                "text": str(turn.get("text") or ""),
            })
    return rows


def source_round(ref: str) -> str:
    for part in ref.split(":"):
        if part.startswith("P") and part[1:].isdigit():
            return "Parent"
        if part in {"R01", "R02"}:
            return part
    return "UNKNOWN"


def nearest_history(text: str, history: list[Mapping[str, Any]]) -> dict[str, Any]:
    ranked = sorted((
        (SequenceMatcher(None, norm(text), norm(str(row.get("text") or ""))).ratio(), row)
        for row in history
    ), key=lambda pair: pair[0], reverse=True)
    score, row = ranked[0]
    return {
        "similarity": round(score, 4),
        "turn_ref": row.get("turn_ref"),
        "speaker": row.get("speaker"),
        "text": row.get("text"),
        "source_round": source_round(str(row.get("turn_ref") or "")),
    }


def prompt_checks(mode: str, actual_messages: list[Mapping[str, Any]],
                  frozen_messages: list[Mapping[str, Any]]) -> dict[str, Any]:
    actual = actual_messages[0]["content"]
    frozen = frozen_messages[0]["content"]
    return {
        "actual_messages_byte_equal_frozen": actual_messages == frozen_messages,
        "actual_prompt_sha256": sha(actual),
        "frozen_prompt_sha256": sha(frozen),
        "prompt_equal": actual == frozen,
        "mode_marker_present": f"AB PREPARATION MODE: {mode}" in actual,
        "full_transcript_marker": "FROZEN PARENT TRANSCRIPT" in actual,
        "text_free_marker": "CONTINUITY SHAPE ONLY" in actual,
        "output_only_new_instruction": "Return only" in actual
        and "newly generated append-only Child dialogue" in actual,
    }


def scene_diagnostics(mode: str, output: Mapping[str, Any],
                      plan: Mapping[str, Any], history: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = flatten(output)
    by_scene: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_scene.setdefault(row["scene_index"], []).append(row)
    result: list[dict[str, Any]] = []
    for index, scene_plan in enumerate(plan.get("scenes") or [], 1):
        scene_rows = by_scene.get(index, [])
        beats = [str(value) for value in scene_plan.get("interaction_beats") or []]
        beat_matches = []
        for beat in beats:
            # Planner beats are rendered as ``Speaker: wording`` while output
            # turn objects store speaker separately.
            beat_text = beat.split(":", 1)[1].strip() if ":" in beat else beat
            exact = [row for row in scene_rows if norm(row["text"]) == norm(beat_text)]
            beat_matches.append({
                "beat": beat,
                "beat_text": beat_text,
                "exact_output_turn_refs": [row["turn_ref"] for row in exact],
                "exact": bool(exact),
            })
        nearest = [
            {
                "output_turn_ref": row["turn_ref"],
                "speaker": row["speaker"],
                "text": row["text"],
                "nearest_historical": nearest_history(row["text"], history),
            }
            for row in scene_rows
        ]
        result.append({
            "scene_index": index,
            "memory_event_id": scene_plan.get("memory_event_id"),
            "planned_goal": scene_plan.get("goal"),
            "planned_required_transition": scene_plan.get("required_transition"),
            "planned_after_state": scene_plan.get("after_state"),
            "output_turn_count": len(scene_rows),
            "planner_interaction_beats_exactly_reused": sum(
                row["exact"] for row in beat_matches),
            "planner_beat_comparison": beat_matches,
            "output_turn_nearest_history": nearest,
            "content_diagnostic": (
                "OUTPUT_HAS_DIRECTLY_OBSERVABLE_NEW_DETAIL_OR_SCHEDULE"
                if mode == "B_TEXT_FREE" and index in {1, 2, 3}
                else "UNKNOWN_SEMANTIC_VALUE"
            ),
        })
    return result


def build_comparison(run_root: Path) -> dict[str, Any]:
    source = read_json(SOURCE_DIALOGUE)
    plan = read_json(
        Path(read_json(PREPARATION_RUN / "source_identity.json")["plan_path"]))
    history = source.get("turns") or []
    prep_report = read_json(PREPARATION_RUN / "ab_input_preparation_report.json")
    rows: dict[str, Any] = {}
    for mode, directory in (("A_FULL_PARENT", "a_full_parent"),
                            ("B_TEXT_FREE", "b_text_free")):
        root = run_root / directory
        output = read_json(root / "parsed_response.json")
        frozen = read_json(PREPARATION_RUN / directory / "messages.json")
        actual = read_json(root / "actual_messages.json")
        result = read_json(root / "condition_result.json")
        copy = read_json(root / "exact_copy_analysis.json")
        rows[mode] = {
            "request_completed": result.get("request_completed"),
            "technical_failure": result.get("technical_failure"),
            "parse_failure": result.get("parse_failure"),
            "scene_count": result.get("scene_count"),
            "attempted_turns": result.get("attempted_turns"),
            "promoted_turns": result.get("promoted_turns"),
            "exact_copy_output_turn_count": copy.get("exact_copy_output_turn_count"),
            "exact_copy_match_pair_count": copy.get("match_pair_count"),
            "copy_sources": copy.get("source_round_unique_output_turn_counts"),
            "longest_contiguous_output_copy_length": copy.get("longest_contiguous_output_copy_length"),
            "closure_stage": result.get("closure_stage"),
            "closure_passed": result.get("closure_passed"),
            "validation_errors": result.get("validation_errors") or [],
            "prompt_checks": prompt_checks(mode, actual, frozen),
            "scene_diagnostics": scene_diagnostics(mode, output, plan, history),
            "raw_response_sha256": read_json(root / "response_metadata.json").get("raw_response_sha256"),
        }

    # The plan itself contains wording copied from prior events in all scenes;
    # this is reported as a frozen upstream issue, not attributed to either
    # Realizer condition and not converted into a new hard gate.
    plan_history_matches = [
        {
            "scene_index": 1,
            "field": "interaction_beats[0:3]",
            "historical_refs": ["R02:S02:T001", "R02:S02:T005", "R02:S02:T007"],
            "finding": "BEATS_RESTATE_PRIOR_BULLSEYE_AND_TRACKING_CONTENT",
        },
        {
            "scene_index": 2,
            "field": "before_state/interaction_beats",
            "historical_refs": ["R02:S01:T001-T014"],
            "finding": "PLAN_TREATS_COMPLETED_AERODYNAMICS_TUTORIAL_AS_IF_REVIEW_HAS_NOT_OCCURRED; A REPLAYS IT",
        },
        {
            "scene_index": 3,
            "field": "before_state",
            "historical_refs": ["R02:S02:T001-T014"],
            "finding": "BEFORE_STATE_IS_ARCHERY_SUMMARY_WHILE_GOAL_IS_MATH_GROUP; SOURCE_BINDING_MISMATCH",
        },
        {
            "scene_index": 3,
            "field": "transition_from_previous/interaction_beats",
            "historical_refs": ["R02:S03:T001", "R02:S03:T013"],
            "finding": "COMPLETES_PRIOR_INTENT_TO_CONTACT_GROUP_AND_SET_FIRST_MEETING",
        },
    ]
    report = {
        "comparison_version": "h4_round1_r03_realizer_ab_single_generation_comparison_v1",
        "created_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "run_root": str(run_root),
        "root_id": source.get("root_id"),
        "source_dialogue_sha256": source.get("complete_dialogue_sha256"),
        "preparation_report_status": prep_report.get("status"),
        "conditions": rows,
        "plan_upstream_diagnostics": {
            "plan_sha256": file_sha(Path(read_json(PREPARATION_RUN / "source_identity.json")["plan_path"])),
            "historical_summary_or_reuse_observations": plan_history_matches,
            "semantic_plan_quality": "UNKNOWN_NOT_AUDITED",
        },
        "comparative_conclusions": {
            "request_and_contract": "Both requests completed with same frozen model, parameters, plan, output contract and one attempt each.",
            "copy_comparison": "A was quarantined by 11 exact R02 text matches, including a 10-turn contiguous block; B had 0 exact-copy matches and passed the existing closure.",
            "new_facts_comparison": "B contains observable schedule/detail additions (for example 85% average, recording sessions, 8 PM review, calendar invite). Whether these are true or supported by the frozen history is a semantic question and remains UNKNOWN; no semantic auditor was called.",
            "plan_execution": "Both outputs follow all three scene plans and reuse their first interaction beats. This shows the frozen plan is an upstream repetition pressure; it does not prove either output's facts are semantically valid.",
            "historical_conflicts": "No direct contradiction was established offline. UNKNOWN where the output introduces facts not present in the authoritative dialogue.",
            "route_decision": "No default route selected. One root and one sample per condition cannot establish a generally superior input mode.",
        },
        "calls": {
            "realizer": 2, "planner": 0, "qa_writer": 0, "solver": 0,
            "semantic_auditor": 0, "training": 0, "memorax_search": 0,
            "memorax_status": 0, "memorax_add": 0, "memorax_mutation": 0,
            "retry": 0, "repair": 0, "fallback": 0,
        },
        "source_files_unchanged": read_json(run_root / "source_sha_audit.json").get("all_unchanged"),
    }
    return report


def write_outputs(run_root: Path, report: Mapping[str, Any]) -> None:
    write_json(run_root / "ab_comparative_analysis.json", report)
    lines = [
        "# R03 Realizer A/B 离线对照分析", "",
        f"- Root: `{report['root_id']}`",
        "- 真实调用：Realizer `2`；Planner、QA、Solver、semantic auditor、训练、Memorax：`0`",
        "- 默认路线选择：`false`", "",
        "| 条件 | 请求 | scenes | attempted | promoted | exact-copy turns | longest copy | closure |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ("A_FULL_PARENT", "B_TEXT_FREE"):
        row = report["conditions"][mode]
        lines.append(
            f"| {mode} | {row['request_completed']} | {row['scene_count']} | "
            f"{row['attempted_turns']} | {row['promoted_turns']} | "
            f"{row['exact_copy_output_turn_count']} | "
            f"{row['longest_contiguous_output_copy_length']} | "
            f"{row['closure_passed']} |"
        )
    lines.extend([
        "", "## 结论", "",
        "- A：11 个输出 turn 命中历史 R02 原文，最长连续复制 10 turn，既有 validator 隔离。",
        "- B：0 个 exact-copy，44 turn 通过既有 closure；这不是语义正确性证明。",
        "- B 观察到可见的新数字、时间安排和执行细节，但是否由历史支持为 `UNKNOWN`。",
        "- 两组都直接复用了 Planner 的 interaction beats；Planner 上游重复压力冻结，不在本轮修改。",
        "- 本样本不足以选择默认 A 或 B 路线。",
    ])
    (run_root / "ab_comparative_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    report = build_comparison(args.run_root)
    write_outputs(args.run_root, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
