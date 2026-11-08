"""Render immutable live responses with their public tasks and cited source text."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import repair


def main():
    out = Path(__file__).resolve().parent / "outputs"
    freeze = repair.read(out / "RAW_RESPONSE_FREEZE.json")
    for binding in freeze["bindings"]:
        repair.check(binding)
    packet = repair.read(out / "REASON_REVIEW_PACKET.json")
    review_path = out / "SEMANTIC_REVIEW_AI.json"
    reviews = repair.read(review_path)["reviews"] if review_path.exists() else []
    by_id = {row["request_id"]: row for row in reviews}
    rows = sorted(packet["requests"], key=lambda row: (row["question_type"] != "multi_select", row["request_id"]))
    lines = ["# B1 真实理由输出样例", "",
        "这些是恢复理由接口后的真实模型返回，不是手写格式示例。模型为本轮 B1，完整对话输入，原生 thinking 关闭，要求输出可核查的简短解题说明。",
        "四个请求在运行前已固定；零自动重试。该批用于接口验证，不用于估计总体准确率或证明反馈收益。",
        f"模型加载及四次生成合计 {freeze['wall_seconds']:.2f} 秒；每次输出上限 3072 tokens。", "",
        "逐项审核为独立 AI 审核，并非人类认证。模型原文与后续审核意见分开记录。", ""]
    counts = {"requests": len(rows), "returned": 0, "output_tokens": 0, "prompt_tokens": 0}
    for record in rows:
        request = record["request"]
        repair.check(record["raw_binding"])
        repair.check(record["source_binding"])
        raw = repair.read(record["raw_binding"]["path"])
        source = repair.read(record["source_binding"]["path"])
        counts["returned"] += raw["status"] == "RETURNED"
        counts["output_tokens"] += len(raw.get("output_token_ids", []))
        counts["prompt_tokens"] += len(raw.get("prompt_token_ids", []))
        lines.extend([f"## {record['candidate_id']}", "", f"题型：{record['question_type']}；root：{record['root_id']}", "",
            request["question"], ""])
        lines.extend(f"- {label}. {text}" for label, text in request["display"].items())
        lines.extend(["", "### 模型原始输出", "", "```json", raw["raw_output"], "```", "",
            f"原始记录：`{record['raw_binding']['path']}`", "",
            "### 模型实际引用的对话", ""])
        cited = {ref for item in record.get("parsed", {}).get("item_evidence", {}).values() for ref in item["turn_refs"]}
        for turn in source["turns"]:
            if turn["turn_ref"] in cited:
                lines.append(f"- [{turn['turn_ref']}] {turn['speaker']}: {turn['text']}")
        lines.extend(["", "### 程序一致性检查", "", "```json",
            json.dumps(record.get("mechanical_consistency", {}), ensure_ascii=False, indent=2), "```", ""])
        if record["request_id"] in by_id:
            lines.extend(["### 独立 AI 审核记录", "", "```json",
                json.dumps(by_id[record["request_id"]], ensure_ascii=False, indent=2), "```", ""])
        else:
            lines.extend(["独立来源／理由审核尚待完成。", ""])
    path = out / "EXAMPLES_ZH.md"
    with path.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    repair.save(out / "LIVE_EXECUTION_STATUS.json", {
        "updated_at_beijing": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "LIVE_REASONS_SAVED", "model_role": "B1",
        "gpu_execution_performed": True, "new_model_calls": len(rows),
        "automatic_retry": False, "wall_seconds": freeze["wall_seconds"],
        "counts": counts, "semantic_review_available": review_path.exists(),
        "formal_weakness_admission": False, "causal_training_benefit_established": False,
        "examples": repair.bind(path), "raw_freeze": repair.bind(out / "RAW_RESPONSE_FREEZE.json"),
        "review": repair.bind(review_path) if review_path.exists() else None})
    print(json.dumps({"examples": str(path), "counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
