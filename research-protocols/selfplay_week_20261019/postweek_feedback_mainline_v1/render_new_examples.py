"""Render the two sealed responses with their actual displayed labels and reviews."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "question_execution"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    packet = read(OUT / "REASON_REVIEW_PACKET.json")
    scores = read(OUT / "SCORES.json")
    reviews = read(OUT / "SEMANTIC_REVIEW_AI.json")
    by_score = {row["candidate_id"]: row for row in scores["records"]}
    by_review = {row["request_id"]: row for row in reviews["reviews"]}
    lines = [
        "# 两道反馈驱动新题：真实返回与审核", "",
        f"固定B1模型，两题各一次，最终完全正确 {scores['correct']}/{scores['requests']}。完整来源，原始返回先封存，之后读取私有Gold评分。", "",
        "这是Train侧接口与任务实现验证。可见简短理由不等于内部思维；两题有明确任务提示，不能据此证明难度提高、稳定弱点或反馈因果收益。", "",
        "本次理由审核者也参与过题目内容审核，因此不声明Gold盲审。审核意见与原始模型文本分开保存。", "",
    ]
    for record in packet["requests"]:
        score = by_score[record["candidate_id"]]
        row = record.get("request")
        if row is None:
            row = next(row for row in read(OUT / "PUBLIC_REQUESTS.json")["requests"] if row["request_id"] == record["request_id"])
        raw = read(record["raw_binding"]["path"])
        review = by_review[record["request_id"]]
        inverse = {canonical: display for display, canonical in row["display_to_canonical"].items()}
        display_gold = [inverse[label] for label in score["gold"]]
        lines += [f"## {record['candidate_id']}", "", f"操作：`{row['task_operation']}`；root：`{row['root_id']}`。", "", row["question"], ""]
        lines += [f"- {label}. {text}" for label, text in row["display"].items()]
        lines += ["", f"显示标签Gold：`{' '.join(display_gold)}`；模型显示答案：`{' '.join(record.get('answer_labels', []))}`；完全正确：{score['exact']}。", "", "### 原始模型返回", "", "```json", raw.get("raw_output") or json.dumps(raw, ensure_ascii=False), "```", ""]
        if "parsed" in record:
            source = read(row["source_path"])
            turns = {turn["turn_ref"]: turn for turn in source["turns"]}
            refs = list(dict.fromkeys(ref for item in record["parsed"]["item_evidence"].values() for ref in item["turn_refs"]))
            lines += ["### 模型实际引用的原文", ""]
            for ref in refs:
                turn = turns[ref]
                lines += [f"- [{ref}] {turn['speaker']}: {turn['text']}"]
            lines += [""]
        lines += ["### 独立AI理由审核", "", "```json", json.dumps(review, ensure_ascii=False, indent=2), "```", ""]
    with (OUT / "REPORT_ZH.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    print(json.dumps({"rendered": len(packet["requests"]), "path": str(OUT / "REPORT_ZH.md")}))


if __name__ == "__main__":
    main()
