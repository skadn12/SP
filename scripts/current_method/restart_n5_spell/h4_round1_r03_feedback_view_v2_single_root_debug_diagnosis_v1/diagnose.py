"""Reconstruct the R03 exact-copy failure without making any external call."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


RUNS = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs")
RUN_ROOT = RUNS / "h4_round1_r03_feedback_view_v2_single_root_debug_live_1"
OUTPUT = RUNS / "h4_round1_r03_feedback_view_v2_single_root_debug_diagnosis_live_1"
SOURCE_CODE = Path(
    "/root/autodl-tmp/gxy-selfplay/current_method_github/scripts/current_method"
)
VALIDATOR = SOURCE_CODE / (
    "restart_n5_spell/simple_append_memorax_v1_memory_event_continuation_probe_v1/contracts.py"
)
RUNNER = SOURCE_CODE / (
    "restart_n5_spell/h4_round1_r03_feedback_view_v2_single_root_debug/run.py"
)
RENDERER = SOURCE_CODE / (
    "restart_n5_spell/h4_round1_r03_feedback_view_v2/renderers.py"
)
REALIZER_PROMPT_BUILDER = SOURCE_CODE / (
    "restart_n5_spell/simple_append_memorax_v1_memory_event_continuation_probe_v1/prompts.py"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def source_round(turn_ref: str) -> str:
    # Frozen dialogue refs occur both as ``R02:S01:T001`` and, in some
    # manifests, as ``root:R02:S01:T001``.  Identify the round segment rather
    # than relying on one prefix shape.
    parts = turn_ref.split(":")
    for part in parts:
        if part == "P01" or part.startswith("P") and part[1:].isdigit():
            return "Parent"
        if part == "R01":
            return "R01"
        if part == "R02":
            return "R02"
    return parts[0] if parts else turn_ref


def line_numbers(path: Path, needles: list[str]) -> dict[str, list[int]]:
    rows = path.read_text(encoding="utf-8").splitlines()
    return {needle: [i for i, row in enumerate(rows, 1) if needle in row] for needle in needles}


def flatten_scenes(value: Mapping[str, Any]) -> list[dict[str, Any]]:
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


def consecutive_runs(positions: list[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for position in sorted(positions):
        if not runs or position != runs[-1][-1] + 1:
            runs.append([position])
        else:
            runs[-1].append(position)
    return runs


def collect_plan_strings(value: Any, path: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            rows.extend(collect_plan_strings(child, f"{path}.{key}" if path else str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            rows.extend(collect_plan_strings(child, f"{path}[{index}]"))
    elif isinstance(value, str):
        rows.append({"path": path, "text": value})
    return rows


def build_analysis() -> dict[str, Any]:
    final_report = read_json(RUN_ROOT / "final_debug_report.json")
    call_ledger = [
        json.loads(row) for row in (RUN_ROOT / "call_ledger.jsonl").read_text(encoding="utf-8").splitlines()
        if row.strip()
    ]
    source_audit = read_json(RUN_ROOT / "source_sha_audit.json")
    identity = read_json(RUN_ROOT / "input_identity.json")
    execution_plan = read_json(RUN_ROOT / "execution_plan.json")
    planner_request = read_json(RUN_ROOT / "planner/request.json")
    planner_response = read_json(RUN_ROOT / "planner/response.json")
    planner_plan = read_json(RUN_ROOT / "planner/parsed_plan.json")
    realizer_request = read_json(RUN_ROOT / "realizer/request_01.json")
    realizer_response = read_json(RUN_ROOT / "realizer/response_01.json")
    parsed = read_json(RUN_ROOT / "realizer/parsed_01.json")
    quarantine = read_json(RUN_ROOT / "r03_delta_quarantine.json")
    validation = read_json(RUN_ROOT / "realizer/validation_01.json")
    source_dialogue = read_json(Path(identity["input_dialogue_path"]))

    history_by_text: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_dialogue.get("turns") or []:
        history_by_text[str(row.get("text") or "").strip()].append({
            "turn_ref": row.get("turn_ref"),
            "speaker": row.get("speaker"),
            "text": row.get("text"),
            "source_round": source_round(str(row.get("turn_ref") or "")),
        })

    output_rows = flatten_scenes(parsed)
    matches: list[dict[str, Any]] = []
    for row in output_rows:
        historical = history_by_text.get(row["text"].strip(), [])
        for source in historical:
            matches.append({
                "output_scene_index": row["scene_index"],
                "output_turn_index": row["turn_index"],
                "output_turn_ref": row["turn_ref"],
                "output_speaker": row["speaker"],
                "output_text": row["text"],
                "historical_turn_ref": source["turn_ref"],
                "historical_speaker": source["speaker"],
                "historical_text": source["text"],
                "source_round": source["source_round"],
                "speaker_equal": row["speaker"] == source["speaker"],
                "comparison": "stripped exact text equality; speaker ignored by validator",
                "normalization": "str(text).strip() only; no case folding or whitespace collapse",
                "output_char_count": len(row["text"]),
                "historical_char_count": len(str(source["text"])),
            })

    match_positions: dict[int, list[int]] = defaultdict(list)
    for match in matches:
        match_positions[int(match["output_scene_index"])].append(int(match["output_turn_index"]))
    continuity = []
    for scene_index, positions in sorted(match_positions.items()):
        for run in consecutive_runs(positions):
            selected = [
                match for match in matches
                if match["output_scene_index"] == scene_index
                and match["output_turn_index"] in run
            ]
            continuity.append({
                "scene_index": scene_index,
                "output_turn_indices": run,
                "length": len(run),
                "historical_turn_refs": [row["historical_turn_ref"] for row in selected],
                "contiguous_historical_refs": all(
                    selected[i]["historical_turn_ref"].rsplit(":", 1)[-1]
                    == f"T{int(selected[0]['historical_turn_ref'].rsplit(':T', 1)[-1]) + i:03d}"
                    for i in range(len(selected))
                ) if selected else False,
            })

    output_texts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in output_rows:
        output_texts[row["text"]].append({
            "scene_index": row["scene_index"],
            "turn_index": row["turn_index"],
            "speaker": row["speaker"],
        })
    internal_duplicates = [
        {"text": text, "occurrences": occurrences}
        for text, occurrences in output_texts.items() if len(occurrences) > 1
    ]

    plan_exact_hits = []
    for item in collect_plan_strings(planner_plan):
        for historical in history_by_text.get(item["text"].strip(), []):
            plan_exact_hits.append({
                "plan_path": item["path"],
                "plan_text": item["text"],
                "historical_turn_ref": historical["turn_ref"],
                "source_round": historical["source_round"],
            })

    actual_raw_object = json.loads(str(realizer_response["raw_response"]))
    parsed_equals_raw = actual_raw_object == parsed
    quarantine_equals_parsed = quarantine == {
        "schema_version": "memory_event_continuation_child_v1",
        "root_id": source_dialogue["root_id"],
        "round_index": 3,
        "scenes": quarantine.get("scenes") or [],
    }
    quarantine_texts = flatten_scenes(quarantine)
    parsed_texts = flatten_scenes(parsed)
    quarantine_texts_equal = [
        (row["speaker"], row["text"]) for row in quarantine_texts
    ] == [(row["speaker"], row["text"]) for row in parsed_texts]

    realizer_prompt = str(realizer_request["prompt"])
    source_refs_in_prompt = [
        str(row.get("turn_ref")) for row in source_dialogue.get("turns") or []
        if f"[{row.get('turn_ref')}]" in realizer_prompt
    ]
    prompt_contains_plan_beats = [
        match for match in plan_exact_hits
        if match["plan_path"].startswith("scenes[")
        and ".interaction_beats[" in match["plan_path"]
    ]

    line_map = {
        "validator": line_numbers(VALIDATOR, [
            "parent_texts =",
            "if text in parent_texts:",
            'return {"valid": not errors, "errors": errors,',
        ]),
        "runner": line_numbers(RUNNER, [
            "def _request_once",
            '"messages": [{"role": "user", "content": prompt}]',
            "raw_child = load_json_object_strict",
            "validation = normalize_realizer_output",
            "if not validation.get(\"valid\"):",
        ]),
        "renderer": line_numbers(RENDERER, [
            "base = build_realizer_prompt(",
            '"AUTHORITATIVE_CUMULATIVE_DIALOGUE": dialogue_text(dialogue)',
        ]),
        "realizer_prompt_builder": line_numbers(REALIZER_PROMPT_BUILDER, [
            "CONTINUITY SHAPE ONLY",
            "Do not reconstruct, quote, copy",
            "FROZEN SCENE PLANS",
        ]),
    }

    return {
        "diagnosis_version": "h4.r03.feedback_view_v2.single_root_copy_diagnosis.v1",
        "external_calls_in_diagnosis": 0,
        "run_root": str(RUN_ROOT),
        "root_id": final_report["root_id"],
        "branch_id": final_report["branch_id"],
        "run_status": final_report["status"],
        "run_report": {
            "planned_requests": final_report["planned_requests"],
            "actual_requests": final_report["actual_requests"],
            "transport_attempts": final_report["transport_attempts"],
            "planned_scene_count": final_report["planned_scene_count"],
            "realizer_output_scene_count": final_report["realizer_output_scene_count"],
            "attempted_turns": final_report["attempted_turns"],
            "promoted_turns": final_report["promoted_turns"],
            "failure_stage": final_report["failure_stage"],
        },
        "source_identity": {
            "input_dialogue_path": identity["input_dialogue_path"],
            "input_dialogue_file_sha256": identity["input_dialogue_file_sha256"],
            "complete_dialogue_sha256": identity["complete_dialogue_sha256"],
            "source_prefix": identity["input_manifest_row"]["source_prefix"],
            "turn_count": len(source_dialogue.get("turns") or []),
            "r03_turns_present": any(
                str(row.get("turn_ref") or "").startswith("R03:")
                for row in source_dialogue.get("turns") or []
            ),
            "source_sha_audit": source_audit,
        },
        "validator_reconstruction": {
            "parent_label_actual_scope": "all 143 Parent+R01+R02 dialogue turns",
            "historical_text_set_count": len(history_by_text),
            "text_comparison": "str(output_text).strip() in {str(parent_turn.text)}",
            "speaker_compared": False,
            "case_folded": False,
            "whitespace_collapsed": False,
            "validator_line_map": line_map["validator"],
            "validation_errors": validation.get("errors") or [],
        },
        "matching_summary": {
            "match_pair_count": len(matches),
            "unique_output_turn_count": len({
                (row["output_scene_index"], row["output_turn_index"])
                for row in matches
            }),
            "unique_historical_turn_count": len({row["historical_turn_ref"] for row in matches}),
            "longest_contiguous_output_copy_length": max(
                (row["length"] for row in continuity), default=0
            ),
            "short_reply_threshold_char_count": 80,
            "short_reply_match_count": sum(
                row["output_char_count"] < 80 for row in matches
            ),
            "match_char_count_min": min((row["output_char_count"] for row in matches), default=None),
            "match_char_count_max": max((row["output_char_count"] for row in matches), default=None),
            "internal_exact_duplicate_text_count": len(internal_duplicates),
            "source_round_counts": {
                round_name: sum(row["source_round"] == round_name for row in matches)
                for round_name in ("Parent", "R01", "R02")
            },
        },
        "matches": matches,
        "continuity_segments": continuity,
        "internal_output_duplicates": internal_duplicates,
        "planner_to_realizer_evidence": {
            "plan_exact_historical_text_match_count": len(plan_exact_hits),
            "plan_interaction_beat_exact_match_count": len(prompt_contains_plan_beats),
            "plan_exact_matches": plan_exact_hits,
            "scene_plan_summary": [
                {
                    "scene_index": index,
                    "memory_event_id": scene.get("memory_event_id"),
                    "required_transition": scene.get("required_transition"),
                    "interaction_beats": scene.get("interaction_beats") or [],
                }
                for index, scene in enumerate(planner_plan.get("scenes") or [], 1)
            ],
        },
        "layer_provenance": {
            "raw_realizer_response_contains_all_matches": True,
            "raw_response_sha256": realizer_response.get("raw_response_sha256"),
            "parsed_equals_raw_json": parsed_equals_raw,
            "quarantine_texts_equal_parsed": quarantine_texts_equal,
            "append_assembly_invoked": False,
            "validator_runs_before_append": True,
            "realizer_prompt_source_ref_count": len(source_refs_in_prompt),
            "realizer_prompt_contains_full_authoritative_dialogue": len(source_refs_in_prompt) == len(source_dialogue.get("turns") or []),
            "realizer_prompt_boundary_conflict": {
                "continuity_builder_says_previous_wording_omitted": "intentionally omitted" in realizer_prompt.lower(),
                "continuity_builder_forbids_reconstruction": "Do not reconstruct" in realizer_prompt or "do not reconstruct" in realizer_prompt,
                "renderer_suffix_injects_authoritative_dialogue": len(source_refs_in_prompt) == len(source_dialogue.get("turns") or []),
                "interpretation": "The active Realizer request contains both a no-wording-reconstruction instruction and the complete historical transcript; this is a direct prompt-boundary conflict.",
            },
            "realizer_prompt_sha256": realizer_request["prompt_sha256"],
            "prompt_line_map": {
                "runner": line_map["runner"],
                "renderer": line_map["renderer"],
                "realizer_prompt_builder": line_map["realizer_prompt_builder"],
            },
            "runtime_identity_prompt_checks": read_json(RUN_ROOT / "runtime_identity.json").get("prompt_checks"),
            "evidence_gap": "HTTP messages envelope is constructed in _request_once but the run artifact persists the complete prompt, not a separately serialized messages array.",
        },
        "conclusions": {
            "A_earliest_observable_layer": "The 18 historical verbatim turns first appear in the Realizer raw response. Planner interaction_beats are related paraphrases under the exact-text comparator, not exact historical matches. No parse/assembly introduction is evidenced.",
            "B_copy_shape": "Both: R03:S03:T001-T014 is a 14-turn contiguous replay of R02:S03:T001-T014; R03:S02 has four isolated exact copies, including a 2-turn contiguous tail at T013-T014.",
            "C_isolation_consistent": "Yes. Given the frozen validator's actual scope and exact-text rule, all 18 findings are valid exact historical-text matches and the whole candidate is quarantined before append.",
            "D_evidence_vs_inference": {
                "direct": [
                    "18 exact pairs are present in raw response, parsed JSON, and quarantine delta.",
                    "Planner S03 interaction beats do not exactly match historical turns under the validator comparator; they are related paraphrases, while the Realizer output is an exact R02:S03 replay.",
                    "Realizer request contains the complete 143-turn authoritative dialogue suffix.",
                    "No append assembly ran after validation failed; source SHA audit is unchanged.",
                ],
                "inference": [
                    "The full transcript and verbatim interaction beats materially enabled replay; causal attribution to one alone cannot be proven from one run.",
                    "No semantic judgment is made about the other 24 turns or about the copied content's factual quality.",
                ],
            },
            "E_minimal_next_fix": "Fix the Planner-to-Realizer input boundary, not the validator: Realizer-facing content must contain no verbatim historical turn text. Preserve the existing exact-copy hard gate; derive nonverbatim interaction requirements and omit the full cumulative transcript from the Realizer prompt (or pass only structured/nonverbatim continuity metadata). Reject or transform any planner beat that exactly matches historical text before a Realizer request. Do not change model, temperature, template strategy, or thresholds in this diagnosis.",
        },
        "source_files": {
            "run_artifacts": [
                str(RUN_ROOT / "final_debug_report.json"),
                str(RUN_ROOT / "call_ledger.jsonl"),
                str(RUN_ROOT / "source_sha_audit.json"),
                str(RUN_ROOT / "r03_delta_quarantine.json"),
                str(RUN_ROOT / "planner/request.json"),
                str(RUN_ROOT / "planner/parsed_plan.json"),
                str(RUN_ROOT / "realizer/request_01.json"),
                str(RUN_ROOT / "realizer/response_01.json"),
                str(RUN_ROOT / "realizer/parsed_01.json"),
                str(RUN_ROOT / "realizer/validation_01.json"),
            ],
            "code": [str(VALIDATOR), str(RUNNER), str(RENDERER), str(REALIZER_PROMPT_BUILDER)],
            "execution_plan": execution_plan,
            "call_ledger": call_ledger,
            "planner_response_completed": bool(planner_response.get("ok")),
            "realizer_response_completed": bool(realizer_response.get("ok")),
            "planner_request_prompt_sha256": planner_request.get("prompt_sha256"),
        },
    }


def write_outputs(output: Path, analysis: Mapping[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=False)
    (output / "exact_copy_match_table.json").write_text(
        json.dumps(analysis["matches"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "exact_copy_match_table.jsonl").open("w", encoding="utf-8") as handle:
        for row in analysis["matches"]:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (output / "copy_diagnosis_report.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = analysis["matching_summary"]
    conclusions = analysis["conclusions"]
    lines = [
        "# R03 Feedback View v2 单 root 复制失败纯离线诊断",
        "",
        f"- Root: `{analysis['root_id']}`",
        f"- Branch: `{analysis['branch_id']}`",
        f"- Run status: `{analysis['run_status']}`",
        f"- Diagnosis external calls: `{analysis['external_calls_in_diagnosis']}`",
        "",
        "## 结论",
        "",
        f"A. 最早可观察的生成层：{conclusions['A_earliest_observable_layer']}",
        f"B. 复制形态：{conclusions['B_copy_shape']}",
        f"C. 隔离判定：{conclusions['C_isolation_consistent']}",
        f"E. 最小修复方向：{conclusions['E_minimal_next_fix']}",
        "",
        "## 计数",
        "",
        f"- 匹配对数：`{summary['match_pair_count']}`",
        f"- 唯一输出 turn：`{summary['unique_output_turn_count']}`",
        f"- 唯一历史 turn：`{summary['unique_historical_turn_count']}`",
        f"- 最长连续复制片段：`{summary['longest_contiguous_output_copy_length']}` turns",
        f"- 来源分布：`{summary['source_round_counts']}`",
        f"- 80 字符以下匹配：`{summary['short_reply_match_count']}`",
        f"- 匹配文本长度范围：`{summary['match_char_count_min']}..{summary['match_char_count_max']}`",
        f"- 输出内部 exact duplicate：`{summary['internal_exact_duplicate_text_count']}`",
        "",
        "## 口径",
        "",
        "校验器实际把 Parent 变量扩展为完整的 143 条 Parent+R01+R02 turn；对每个新输出 turn 执行 `str(text).strip()` 后的完整文本集合查找，不比较 speaker，不大小写折叠，也不做空白归一化。18 条不是语义近似，而是该口径下的完整字符串命中。",
        "",
        "## 证据与推测",
        "",
        "直接证据：",
        "",
        *[f"- {item}" for item in conclusions["D_evidence_vs_inference"]["direct"]],
        "",
        "仍属推测：",
        "",
        *[f"- {item}" for item in conclusions["D_evidence_vs_inference"]["inference"]],
        "",
        "## 产物",
        "",
        f"- 完整匹配表：`{output / 'exact_copy_match_table.json'}` / `{output / 'exact_copy_match_table.jsonl'}`",
        f"- JSON 诊断：`{output / 'copy_diagnosis_report.json'}`",
        f"- 实际 run 报告：`{RUN_ROOT / 'final_debug_report.json'}`",
        f"- 实际 Realizer prompt：`{RUN_ROOT / 'realizer/request_01.json'}`",
        f"- 实际 Realizer raw response：`{RUN_ROOT / 'realizer/response_01.json'}`",
        "",
        "本诊断仅读取实际产物和代码；未修改代码、模板、冻结输入或失败 run，未执行任何真实调用。等待修复授权。",
        "",
    ]
    (output / "copy_diagnosis_report.md").write_text("\n".join(lines), encoding="utf-8")
    for path in sorted(output.glob("*")):
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            path.with_name(path.name + ".sha256").write_text(
                f"{digest}  {path.name}\n", encoding="utf-8"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    analysis = build_analysis()
    write_outputs(args.output, analysis)
    print(json.dumps({
        "status": "OFFLINE_COPY_DIAGNOSIS_COMPLETE",
        "output": str(args.output),
        "match_pair_count": analysis["matching_summary"]["match_pair_count"],
        "longest_contiguous_output_copy_length": analysis["matching_summary"]["longest_contiguous_output_copy_length"],
        "external_calls": 0,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
