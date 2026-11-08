"""CPU-only descriptive report from committed raw, scores and blind review."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "postweek_source_citation_control_v1"
REPORT_PATH = HERE / "RESULTS_ZH.md"


def read(p: Path):
    return json.loads(p.read_text())


def save(p: Path, value) -> None:
    with p.open("x") as f:
        if isinstance(value, str):
            f.write(value)
        else:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def main() -> None:
    if REPORT_PATH.exists():
        raise RuntimeError("REPORT_EXISTS_NO_OVERWRITE")
    report = read(HERE / "outputs/scoring/UNBLINDED_MECHANICAL_REPORT.json")
    raw_freeze = read(HERE / "outputs/B1/RAW_RESPONSE_FREEZE.json")
    key = read(PARENT / "private/PRIVATE_SCORE_KEY.json")
    rows = report["individual_responses"]
    errors = []
    for row in rows:
        if row["condition"] != "C1" or row["exact_order_correct"]:
            continue
        raw = read(HERE / "outputs/B1/responses" / (row["request_id"] + ".json"))
        obj = json.loads(raw["raw_output"])
        gold = key["roots"][row["root_id"]]["variants"][row["display_variant"]]
        has_all_anchors = row["all_first_sufficient_anchors_cited"]
        categories = Counter(r["error_category"] for r in row["semantic_item_reviews"].values() if not r["sufficient"])
        errors.append({
            "request_id": row["request_id"], "root_id": row["root_id"],
            "display_variant": row["display_variant"], "replicate": row["replicate"],
            "predicted_display_order": obj["final_answer"],
            "gold_display_order": gold["gold_display_order"],
            "first_sufficient_anchor_hits": row["first_sufficient_anchor_hits"],
            "all_six_correct_anchors_cited_but_final_order_wrong": has_all_anchors,
            "output_equals_display_label_order": obj["final_answer"] == list("ABCDEF"),
            "citation_error_categories_from_blind_review": dict(categories),
            "description": "All six first-sufficient evidence anchors are cited, but final order conflicts with their adjudicated chronology." if has_all_anchors else "Citation/source-state errors coexist with an incorrect final ordering; no unique causal attribution.",
        })
    pair_deltas = Counter(p["exact_delta_C1_minus_C0"] for p in report["matched_pairs"])
    residual = {
        "schema_version": "source_citation_control_residual_descriptive_analysis_v1",
        "scope": "Post-unblinding descriptive classification, not a new admission rule or causal mechanism test.",
        "C1_incorrect_requests": len(errors),
        "C1_all_correct_anchors_but_wrong_order": sum(e["all_six_correct_anchors_cited_but_final_order_wrong"] for e in errors),
        "C1_wrong_order_equal_to_display_label_order": sum(e["output_equals_display_label_order"] for e in errors),
        "matched_pair_exact_outcomes": {"improved": pair_deltas[1], "worsened": pair_deltas[-1], "unchanged": pair_deltas[0]},
        "error_records": errors,
    }
    save(HERE / "RESIDUAL_ERROR_ANALYSIS.json", residual)
    overall = report["overall"]
    c0, c1 = overall["C0"], overall["C1"]
    byroot = []
    for r in report["by_root"]:
        byroot.append(f"| {r['root_id'].rsplit('-', 1)[-1]} | {r['C0']['exact_correct']}/4 | {r['C1']['exact_correct']}/4 | {r['exact_accuracy_delta_C1_minus_C0'] * 100:+.0f} 个百分点 | {r['C0']['anchor_hits']}/24 → {r['C1']['anchor_hits']}/24 |")
    text = f"""# Train-only 来源证据辅助对照：执行结果

本轮在这三个已暴露的 Train 题上，**没有观察到 C1 的收益**。C0 整题答对 {c0['exact_correct']}/12（{c0['exact_accuracy']:.2%}），C1 为 {c1['exact_correct']}/12（{c1['exact_accuracy']:.2%}），下降 16.67 个百分点；两组首次充分证据引用命中率都为 60/72（83.33%）。这不支持把给出证据片段作为当前已经有效的修复。

## 固定指标

| 指标 | C0：完整对话 | C1：完整对话＋审核证据片段 |
|---|---:|---:|
| 整题排序正确 | 5/12（41.67%） | 3/12（25.00%） |
| 两两排序关系正确 | 165/180（91.67%） | 154/180（85.56%） |
| 首次充分证据引用命中 | 60/72（83.33%） | 60/72（83.33%） |
| 引用充分：AI 代理盲审辅助读出 | 60/72（83.33%） | 60/72（83.33%） |
| JSON schema 有效 | 12/12 | 12/12 |

12 个配对中，整题正确性改善 1 对、变差 3 对、不变 8 对。180 个两两关系、72 个题项和12个请求均来自每组仅3道题的置换与重复，不能当成相应数量的独立题目。

| Root | C0 整题正确 | C1 整题正确 | C1−C0 | 首次证据命中：C0 → C1 |
|---|---:|---:|---:|---:|
{chr(10).join(byroot)}

## 可观察到的错误

C1 的 9 次整题错误中，**5 次已引用全部六个正确的首次充分证据，却仍给出错误排序**。这证明至少这些记录不能仅归因于没有找到证据；证据定位与最终排序的一致性仍有问题。它不等于已证明内部推理机制。

具体例子：136202 的 perm_01 两个种子下，C1 都引用了六个正确锚点，却输出 `A-B-C-D-E-F`；冻结正确展示顺序为 `B-C-F-A-E-D`。这是按展示顺序作答的具体错误现象，尚不能据本轮证明证据面板导致了该偏差。

147997 中两组全部题项引用都正确，但 C0 仍只有 2/4 整题正确，C1 只有 1/4；156136 中也仍有引用定位、请求与已完成状态混淆。C1 帮助所有题目的假设未获支持。全部题项的模型判断均为 supported；这组排序题的题项本来均有支持，所以本轮没有检验区分 supported／contradicted／not_supported／ambiguous 的能力。

## 执行与复核

- 固定 B1 Day5 step-56，3 roots × 2 标签映射 × 2 种子 × C0/C1，共 24/24 请求，逐请求按冻结随机顺序执行，零重试。
- 推理阶段 wall {raw_freeze['wall_seconds']:.2f} 秒，含模型加载 {raw_freeze['model_load_seconds']:.2f} 秒；准备、审计和后处理用时不计入此数字。
- 输入 223,792 tokens，输出 7,501 tokens，共 231,293 tokens。模型返回均为 stop，未触达输出长度上限。
- 原始响应先封存，再机械计分；144 个题项按相同语义项／判断／引用集合去重成 40 组，分别由三个独立 Codex 代理盲审，再封存语义记录，最后解盲。
- **语义审核是 AI 代理盲审，不是真人审核。** 原方案要求的人工语义审核尚未由真人完成，相关语义读出作为辅助结果。机械排序和锚点计分不依赖该代理审核。
- 来源 ID 后处理修复及审核方式见 `POSTPROCESSING_NOTES_ZH.md`；修复未改动任何冻结输入、原始响应或模型预算。

## 结论与后续

本轮负结果表明：不能把“缺少明确证据片段”当成这三道题的充分解释。下一项值得验证的目标是证据到排序的转换：用结构化的逐项锚点及事件状态生成最终顺序，并检查输出是否与所引证据一致。任何新调用需另行固定协议；不从本轮直接宣称获得正式可复现弱点、训练数据准入或 self-play 反馈因果收益。

只有三个已暴露 Train 题，且 C1 是证据片段和局部上下文的整体辅助，147997 的上下文含与先后关系相关的线索。因此不作显著性、未见题泛化或纯检索机制结论，也不据此概括证据辅助普遍有害。尚需在新独立 Train roots 上复现，才适合形成正式反馈卡。

## 文件

- `outputs/B1/RAW_RESPONSE_FREEZE.json`：24 个原始响应的封存清单、时长。
- `outputs/scoring/BLIND_SCORE_FREEZE.json`：机械评分封存。
- `outputs/blind_review/SEMANTIC_REVIEW_FREEZE.json`：代理盲审封存。
- `outputs/scoring/UNBLINDED_MECHANICAL_REPORT.json`：全部指标、逐 root／配对／请求结果。
- `RESIDUAL_ERROR_ANALYSIS.json`：9 次 C1 错误的逐条分类。
- `RAW_EXECUTION_AUDIT.json`、`BLIND_REVIEW_V2_AUDIT.json`、`FINAL_RESULTS_AUDIT.json`：独立审计。
"""
    save(REPORT_PATH, text)
    print(json.dumps({"status": "DESCRIPTIVE_REPORT_WRITTEN", "path": str(REPORT_PATH),
        "C0_exact": c0["exact_correct"], "C1_exact": c1["exact_correct"],
        "C1_wrong_despite_all_six_anchors": residual["C1_all_correct_anchors_but_wrong_order"]}))


if __name__ == "__main__":
    main()
