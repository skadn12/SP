"""Render manually authored old-R02 ordering diagnoses; no API or semantic gate.

The semantic decisions live in diagnostics/old_r02_ordering_manual_notes.json.
This script checks coverage and references, and quotes source turns verbatim. It
does not infer semantic validity from schema success, reference presence or order.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

ROOT_IDS = [f"cc-train-episode-{x}" for x in (122711, 136202, 197635, 198480, 254852, 94289)]
SOURCE = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round1_r03_r1_aligned_feedback_r02_qa_generation_live_1/shards/shard-00/roots")
DEFAULT_OUTPUT = Path("/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_r02_feedback_ablation_p0_p1_v1_live_1/diagnostics")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(output: Path):
    notes_path = Path(__file__).parent / "diagnostics/old_r02_ordering_manual_notes.json"
    notes = json.loads(notes_path.read_text())
    rows, inventory, seen = [], [], set()
    for root_id in ROOT_IDS:
        source_dir = SOURCE / root_id
        candidate_path = source_dir / "candidates_valid.json"
        transcript_path = source_dir / "expanded_dialogue_clean.json"
        transcript = json.loads(transcript_path.read_text())
        turns = {t["turn_ref"]: t for t in transcript["turns"]}
        candidates = [c for c in json.loads(candidate_path.read_text()) if c["question_type"] == "ordering"]
        inventory.append({"root_id": root_id, "ordering_count": len(candidates), "full_transcript_turns_read": len(turns), "candidate_source_path": str(candidate_path), "candidate_source_sha256": sha(candidate_path), "full_transcript_path": str(transcript_path), "full_transcript_sha256": sha(transcript_path)})
        for candidate in candidates:
            cid = candidate["candidate_id"]
            note = notes[cid]
            seen.add(cid)
            def expand(ref):
                return f"{root_id}:{ref}" if ref.startswith("P") else ref
            item_reviews = []
            for label, status, refs, assessment in note["items"]:
                refs = [expand(r) for r in refs]
                assert all(r in turns for r in refs), (cid, refs)
                item_reviews.append({"item_id": label, "original_text": candidate["items"][label], "status": status, "assessment": assessment, "reviewed_evidence_refs": refs, "evidence_quotes": [{"turn_ref": r, "speaker": turns[r]["speaker"], "text": turns[r]["text"]} for r in refs]})
            assert {x["item_id"] for x in item_reviews} == set(candidate["items"]), cid
            rows.append({"candidate_id": cid, "root_id": root_id, "question": candidate["question"], "generation_source": candidate["generation_source"], "original_gold": candidate["gold"], "original_items": candidate["items"], "original_evidence_refs": candidate["evidence_refs"], "candidate_source_path": str(candidate_path), "candidate_source_sha256": sha(candidate_path), "full_transcript_path": str(transcript_path), "full_transcript_sha256": sha(transcript_path), "full_transcript_turn_count": len(turns), "review_status": "content_reviewed_against_full_transcript_by_codex_no_api", "reviewer_kind": "Codex direct reading; not an independent human rater", "review_method": "Read all source transcript turns and each original candidate. Semantic judgments manually authored; script only validates coverage/refs and renders evidence quotes.", "gold_assessment": note["verdict"], "ordering_basis": note["basis"], "item_reviews": item_reviews, "order_uniqueness": note["uniqueness"], "problem_categories": note["categories"], "content_findings": note["finding"], "correction_recommendation_not_applied": note["correction"], "original_reference_assessment": note["ref_note"], "unknowns": note.get("unknowns", []), "applied_to_original_or_p1": False})
    assert len(rows) == 52, len(rows)
    assert seen == set(notes), (set(notes) - seen, seen - set(notes))
    output.mkdir(parents=True, exist_ok=True)
    output_file = output / "old_r02_ordering_review.jsonl"
    output_file.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    counts = dict(collections.Counter(r["gold_assessment"] for r in rows))
    summary = {"coverage": {"reviewed": len(rows), "expected": 52, "all_full_transcripts_read": True, "total_transcript_turns": sum(x["full_transcript_turns_read"] for x in inventory)}, "gold_assessment_counts": counts, "categories_are_nonexclusive": True, "category_counts": dict(collections.Counter(c for r in rows for c in r["problem_categories"])), "source_inventory": inventory, "manual_notes_sha256": sha(notes_path), "renderer_sha256": sha(Path(__file__)), "jsonl_sha256": sha(output_file), "reviewer": "Codex direct content reading; no review API and no independent human adjudication", "limitations": ["Content review is not original program admission and is not applied to P1 inputs, labels, filtering, generation or scoring.", "A source quote or schema pass does not establish a correct Gold.", "Dialogue mention order, actual event time and planned actions are distinguished. Where chronology or completion is absent, no unique corrected permutation is invented.", "All 52 were directly reviewed. Unstated real-world times, intended references, actual unreported completions and independent human adjudication remain unknown."]}
    (output / "old_r02_ordering_review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    md = ["# 旧对齐 R02 全量 ordering 内容诊断", "", "覆盖 **52/52** 道 ordering，完整读取六 root 的 **880/880** 个 transcript turn。原始各 root 数量为 8、8、9、9、9、9。本报告为 Codex 直接阅读后的逐题内容核查，不是独立人类评审，也不是程序 schema/引用存在性检查的语义结论；没有调用审核 API。", "", "所有原 Gold、历史题目、transcript 与本次 P1 输入/标签/评分均未修改。下述修正只作建议，不是新增准入。`corrected R02` 是扩充路线名称，不代表 QA Gold 正确。", "", "判定同时考虑题干时间口径、实际事件和陈述时间、计划与已完成状态、同一事项的重复回顾。只有明确支持时才给有条件的修正排列；没有唯一证据时保留未知。", "", "## 覆盖与判定", "", "|root|ordering 核查|完整 transcript turns|", "|---|---:|---:|"]
    md.extend(f"|{x['root_id']}|{x['ordering_count']}/{x['ordering_count']}|{x['full_transcript_turns_read']}|" for x in inventory)
    md.extend(["", "|Gold 判定|数量|", "|---|---:|"])
    md.extend(f"|{key}|{value}|" for key, value in counts.items())
    md.extend(["", "`supported_with_reference_or_scope_defect` 表示可给出支持原 Gold 的具体指称，但存在引用或范围缺陷；不能等同于无歧义的高质量题。`underdetermined` 表示文本不足以确认唯一原 Gold。`contradicted_gold` 表示存在可定位的反证；`unsupported_item` 表示至少一个必要条目未被支持。类别可以重叠。", "", "独立人类裁决、未陈述的真实事件时间、作者原本希望指向哪次重复事件、计划活动实际完成情况均没有外部核实。正文 unknowns 按题记录。", ""])
    for row in rows:
        md.extend([f"## {row['candidate_id']}", "", f"**题干**：{row['question']}", "", f"**原 Gold**：{' → '.join(row['original_gold'])}；**判定**：{row['gold_assessment']}", "", f"**排序依据**：{row['ordering_basis']}", "", "|条目|原文|证据/状态|", "|---|---|---|"])
        for item in row['item_reviews']:
            refs = ', '.join(item['reviewed_evidence_refs']) or '完整 transcript 无对应完成事件'
            md.append(f"|{item['item_id']}|{item['original_text']}|{item['status']}：{item['assessment']}；refs：{refs}|")
        md.extend(["", f"**顺序唯一性**：{row['order_uniqueness']}", "", f"**问题类别**：{', '.join(row['problem_categories'])}", "", f"**内容判断**：{row['content_findings']}", "", f"**原引用核对**：{row['original_reference_assessment']}", "", f"**修正建议（未应用）**：{row['correction_recommendation_not_applied']}", "", f"**仍未知**：{'；'.join(row['unknowns']) if row['unknowns'] else '无影响本条明确判断的未核查内容；未做独立人类复核。'}", ""])
    (output / "old_r02_ordering_review.md").write_text('\n'.join(md) + '\n')
    print(json.dumps({"coverage": summary["coverage"], "verdicts": counts, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    render(parser.parse_args().output)
