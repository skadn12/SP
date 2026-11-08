"""Summarize this completed diagnostic evaluation; never rewrite frozen inputs."""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "base_qa_free_generation_20261103_run02"
PREVIOUS = HERE.parent / "base_qa_free_generation_solver_eval_20261103_run01"
read = lambda p: json.loads(p.read_text(encoding="utf-8"))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def write(name, value):
    with (HERE / name).open("x", encoding="utf-8") as f:
        f.write(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def aggregate(rows):
    total = len(rows)
    j = sum((Fraction(r["J_fraction"]) for r in rows), Fraction()) / total
    return {"calls": total, "exact_correct": sum(r["exact_correct"] for r in rows),
            "exact_accuracy": sum(r["exact_correct"] for r in rows) / total,
            "mean_J_fraction": str(j), "mean_J": float(j),
            "valid": sum(r["valid"] for r in rows), "invalid": sum(not r["valid"] for r in rows),
            "known_FP": sum(r["fp"] or 0 for r in rows), "known_FN": sum(r["fn"] or 0 for r in rows)}


def main():
    scores = read(HERE / "SCORES.json")
    diagnostics = read(HERE / "DIAGNOSTICS.json")
    items = read(HERE / "ITEMS_PRIVATE.json")
    audit = read(HERE / "INDEPENDENT_SCORE_AUDIT.json")
    assert audit["status"] == "PASS" and not audit["errors"]
    assert (HERE / "COMPLETED.json").exists()
    assert (HERE / "ERROR_ANALYSIS.md").exists()
    checks = []
    for path, expected in read(SOURCE / "INPUT_BINDINGS.json")["protected_inputs"].items():
        checks.append({"path": path, "expected": expected, "actual": sha(Path(path)), "match": sha(Path(path)) == expected})
    for name, expected in read(PREVIOUS / "MANIFEST_FINAL.json").items():
        path = PREVIOUS / name
        checks.append({"path": str(path), "expected": expected, "actual": sha(path), "match": sha(path) == expected})
    assert all(r["match"] for r in checks)
    write("HISTORICAL_INTEGRITY.json", {"status": "PASS", "checks": checks,
                                       "protected_generation_inputs": len(read(SOURCE / "INPUT_BINDINGS.json")["protected_inputs"]),
                                       "previous_solver_run_unchanged": True})
    modes = ("full_context", "no_context")
    summary = {}
    for mode in modes:
        rows = [r for r in scores["requests"] if r["context_mode"] == mode]
        summary[mode] = {"call_micro": aggregate(rows),
                         "by_question_type": {q: aggregate([r for r in rows if r["question_type"] == q]) for q in ("multi_select", "ordering")},
                         "question_macro_exact": sum(scores["items"][i["candidate_id"]][mode]["exact_accuracy"] for i in items) / len(items),
                         "question_macro_J": sum(scores["items"][i["candidate_id"]][mode]["mean_J"] for i in items) / len(items)}
    title = {1: "辅导员联系进度", 2: "咨询步骤排序（5项）", 3: "已发送／已收到通信", 4: "照片呈现排序（3项）", 5: "观赛调整", 6: "志愿安排待答复事项", 7: "发言分工草稿", 8: "成功火车／餐桌预订", 9: "育儿会谈最终时段", 10: "有参考号的成功预订", 11: "仍未发送的草稿", 12: "已提出的加薪理由"}
    table = ["| 题号 | Root | 任务 | Gold（冻结展示） | Full 整题正确 | Full 平均 J | 无原文整题正确 | 无原文平均 J |",
             "|---:|---|---|---|---:|---:|---:|---:|"]
    for item in items:
        cid = item["candidate_id"]
        full, nc = (scores["items"][cid][m] for m in modes)
        gold = ("," if item["question_type"] == "multi_select" else ">").join(item["gold_labels"])
        table.append(f"| {item['number']} | {item['root_id'].split('-')[-1]} | {title[item['number']]} | `{gold}` | {full['exact_correct_calls']}/{full['planned_calls']} | {full['mean_J']:.4f} | {nc['exact_correct_calls']}/{nc['planned_calls']} | {nc['mean_J']:.4f} |")
    full = summary["full_context"]["call_micro"]
    nc = summary["no_context"]["call_micro"]
    summary.update(created_at_utc=datetime.now(timezone.utc).isoformat(), candidate_count=len(items), root_count=6,
                   diagnostic_only=True, formal_qa_eligible=False,
                   question_numbers={i["number"]: i["candidate_id"] for i in items})
    write("RUN_SUMMARY.json", summary)
    lines = ["# 自由出题 Run02：本地 Qwen3-4B 真实评估", "",
             f"已完成全部12道冻结首稿、6个roots、396次真实本地推理。Full整题正确{full['exact_correct']}/{full['calls']}（{full['exact_accuracy']:.2%}），平均J={full['mean_J']:.6f}；无原文{nc['exact_correct']}/{nc['calls']}（{nc['exact_accuracy']:.2%}），平均J={nc['mean_J']:.6f}。", "",
             "这些是研究首稿的诊断结果，内容盲审一致与正式素材使用资格分别记录。本轮没有新出题、改写、修改Gold或自行批准素材。", "",
             "## 逐题整题分与部分分", "", *table, "",
             "表中整题正确是程序打乱后的多次作答计数；J为逐次部分分的均值，不是正确率或F1。完整题干、全部选项、Gold和原标签答案分布见 [QUESTIONS_WITH_RESULTS.md](QUESTIONS_WITH_RESULTS.md)。", "",
             "## 固定协议与映射", "",
             "- 模型：`/root/autodl-tmp/Qwen3-4B`，基础权重，bf16；vLLM本地离线加载，不使用学校API。temperature=0.8，top_p=0.9，max_new_tokens=32，max_model_len=16384，关闭thinking。Prompt和停止串沿用上一批诊断评估。",
             "- 每道多选6种唯一选项排列×3次采样；5项排序12种排列×1次；3项排序穷举6种排列×1次。每个条件198次；Full与No-context按排列及种子配对。",
             "- 保持语义option_id，逐排列重绑字母／I标签和Gold；打乱不读取Gold。展示种子2026110351；采样种子`2026110352+题目索引*36+排列索引*3+重复索引`，避免跨题种子重叠。底层引擎种子2026110303。",
             "- Full输入完整冻结对话（218–233轮），只传turn_ref、speaker、text及本题题干选项；不传Gold、证书、warnings、审核摘要或其他题目。最长9393 token，无截断。Reviewer曾看到来源warnings，本轮Solver没有；二者输入条件并不相同。",
             "- 同模型Codex独立上下文盲审记录沿用首稿实验，不等同跨模型验证；本轮不重跑Reviewer。",
             "- 多选部分分严格为`J=TP/g−FP/(n−g)`；本批均有错误选项，无全对项分母例外。排序将答案转换为两两先后关系，`J=(正确对数−反转对数)/C(k,2)`；5项分母10、3项分母3。",
             "- 所有非法回答按固定分母计零，原文保留，不补项、不提取修复、不重采。Full一条排序缺项；无原文11条多选输出带`Answer:`或未给标签，均单列格式错误。没有后端失败。", "",
             "## 结果解释", "",
             "Full有5题在各自全部排列下整题100%正确（题4、5、6、9、10）；其余7题有错误。题3和8仅3/18正确，题2为0/12，在通信／预订状态筛选和咨询流程排序任务上出现可观察错误；是否源于长对话检索覆盖、状态绑定或其他原因仍需进一步验证。具体错选、漏选、反转及原文核查见 [ERROR_ANALYSIS.md](ERROR_ANALYSIS.md)。", "",
             "题4无原文仍5/6正确，题5无原文12/18正确，支持现有语义／常识捷径问题；这两题不能作为困难推理能力的证据。题6、9、10的Full表现高于无原文，但其最终状态有直接重述，高正确率也不能证明L3/L4复杂关系发现改善。题9无原文有11条格式错误，不能把该题全部差值解释为纯推理上的上下文收益；总体No-context差值也包含输出协议因素。", "",
             "多选同排列三次重复的语义答案一致率与跨排列差异分别保留在DIAGNOSTICS.json；一致也可能稳定答错。排序每排列只有一次，无法估计同排列重复一致性。题8与12有明显排列相关差异，但重排也同时改变采样种子，本轮只能称排列／采样联合敏感，不能单独证明位置导致错误。", "",
             "报告同时给出按调用微平均与按题宏平均，避免排序调用数不同造成混淆。", "",
             "| 条件 | 调用微平均整题率 | 调用微平均J | 题目宏平均整题率 | 题目宏平均J |",
             "|---|---:|---:|---:|---:|"]
    for mode in modes:
        s = summary[mode]
        lines.append(f"| {mode} | {s['call_micro']['exact_accuracy']:.4%} | {s['call_micro']['mean_J']:.6f} | {s['question_macro_exact']:.4%} | {s['question_macro_J']:.6f} |")
    lines += ["", "## 可复核交付", "",
              "- `QUESTIONS_WITH_RESULTS.md`：完整12题、Gold、Full/No-context和答案分布。",
              "- `ANSWERS_ALL.csv`：396次原始回答、逐次J、整题结果和回绑后的原题标签。",
              "- `REQUESTS.json`、`PERMUTATIONS_PRIVATE.json`、`prompts/`：每次展示、稳定ID、同步Gold映射、种子和实际输入。",
              "- `RAW_ANSWERS.json`、`raw/`：原始模型输出、token和结束原因。",
              "- `SCORES.json`、`DIAGNOSTICS.json`：逐次评分、每项漏选错选、两两反转、逐排列与重复统计。",
              "- `INDEPENDENT_SCORE_AUDIT.json`：标准库独立复算全部396次，包含映射与聚合验证，结果PASS。",
              "- `INPUT_INTEGRITY_FINAL.json`、`HISTORICAL_INTEGRITY.json`：冻结题目、Gold、来源、历史版本与上一批评估哈希未变。",
              "", "本轮可提取状态更新、人物／事件绑定、长清单漏选和流程排序的候选反馈；没有做Evidence-only或证据消融，且单题样本有限，尚不能将这些候选反馈称为已定位的因果弱点。无训练、无扩充流程变更。", ""]
    write("REPORT.md", "\n".join(lines))
    # Freeze the finished delivery, excluding volatile caches and the manifest itself.
    files = {str(p.relative_to(HERE)): sha(p) for p in sorted(HERE.rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts and p.name != "MANIFEST_FINAL.json"}
    write("MANIFEST_FINAL.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(), "files": files})
    print(json.dumps({"full": full, "no_context": nc, "manifest_files": len(files), "audit": "PASS"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
