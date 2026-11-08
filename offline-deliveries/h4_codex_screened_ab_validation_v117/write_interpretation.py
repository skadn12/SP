"""Render a Chinese interpretation only from completed v117 artifacts.

This additive reporting tool does not run Solver, alter frozen content, train, or
merge a pool. Default invocation validates inputs without writing; use --write
only after the batch coordinator explicitly authorizes report generation.
"""
from argparse import ArgumentParser
from collections import Counter
from fractions import Fraction
from pathlib import Path
import hashlib
import json


HERE = Path(__file__).resolve().parent
TYPE_LABELS = (
    ("multi_select_gold2_5", "Gold2–5 多选"),
    ("gold2", "Gold2"), ("gold3", "Gold3"),
    ("gold4", "Gold4"), ("gold5", "Gold5"),
    ("ordering", "排序"),
)


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_bindings(bindings):
    for binding in bindings:
        path = Path(binding["path"])
        require(path.is_file() and sha(path) == binding["sha256"],
                f"Input binding mismatch: {path}")


def metric_cell(row):
    if not row["calls"]:
        return "—"
    return f"{row['correct']}/{row['calls']}（{row['accuracy']:.1%}）"


def load_completed():
    final = read(HERE / "FINAL_STATUS.json")
    behavior = read(HERE / "BEHAVIOR_DIAGNOSTICS.json")
    summary = read(HERE / "CONTENT_SUMMARY.json")
    pool = read(HERE / "evaluation/EVALUATION_POOL.json")["questions"]
    selected = read(HERE / "SELECTED_CANDIDATES.json")
    source_audit = read(HERE / "SOURCE_PROVENANCE_AUDIT.json")
    export_audit = read(HERE / "EVIDENCE_EXPORT_AUDIT.json")
    solver_audit = read(HERE / "INDEPENDENT_SOLVER_AUDIT.json")
    gate = read(HERE / "BASELINE_GATE.json")
    check_bindings([gate["selection_authority"], *gate["screen_bindings"]])
    screen = read(HERE.parent / "h4_codex_baseline_screen_v116/SCREEN_RESULTS.json")
    require(final["status"] == "COMPLETE_SERIALIZED_A3B3_VALIDATION_NO_TRAINING",
            "Final delivery is not complete")
    require(final["content"] == summary, "Final content summary differs from source")
    require(final["training_started"] is False and selected["training_started"] is False,
            "This report is restricted to a no-new-training batch")
    require(behavior["training_started"] is False and behavior["pool_merge_performed"] is False,
            "Unexpected training or merge in diagnostics")
    for name, audit in (("source", source_audit), ("export", export_audit), ("solver", solver_audit)):
        require(audit["status"] == "PASS" and audit["errors"] == [], f"{name} audit incomplete")
    require(final["source_provenance_audit_status"] == "PASS", "Final source audit mismatch")
    require(export_audit["evaluation_pool_sha256"] == sha(HERE / "evaluation/EVALUATION_POOL.json"),
            "Export audit does not bind the current evaluation pool")
    check_bindings(behavior["input_bindings"])
    roots = sorted(gate["fixed_root_ids"])
    require(roots == sorted(r["root_id"] for r in summary["roots"]), "Fixed root set changed")
    require(roots == sorted(r["root_id"] for r in screen["roots"] if r["selected"]),
            "v116 selection changed")
    require(summary["attempted_roots"] == len(roots), "Root denominator mismatch")
    require(summary["complete_A3B3_roots"] == sum(r["full_expansion_complete"] for r in summary["roots"]),
            "Complete root count mismatch")
    qmap = {q["candidate_id"]: q for q in pool}
    require(len(qmap) == len(pool) == summary["new_questions"] + summary["gold6_controls"],
            "Question count mismatch")
    require(export_audit["candidate_count"] == len(pool), "Export candidate count mismatch")
    for side in ("base", "trained"):
        solver = final["solver"][side]
        require(solver["call_set_errors"] == [], f"{side} call set invalid")
        qs = {q["candidate_id"]: q for q in solver["questions"]}
        require(set(qs) == set(qmap), f"{side} question set mismatch")
        require(all(not q["data_errors"] and q["actual_full_calls"] == q["expected_full_calls"]
                    for q in qs.values()), f"{side} incomplete question evaluation")
        records = behavior["sides"][side]["responses"]
        require(len({r["request_id"] for r in records}) == len(records), f"{side} duplicate diagnostic calls")
        for diagnostic_key, metric_key in (("gold2_5", "multi_select_gold2_5"),
                ("gold6_control", "gold6_control"), *((f"gold{k}", f"gold{k}") for k in (2, 3, 4, 5))):
            group = behavior["sides"][side]["groups"][diagnostic_key]
            metric = solver["by_type"][metric_key]
            require(group["calls"] == metric["calls"], f"{side} {diagnostic_key} call mismatch")
            require(group["error_kinds"].get("exact", 0) == metric["correct"],
                    f"{side} {diagnostic_key} exact mismatch")
        require(solver["valid_calls"] == sum(solver["by_type"][k]["calls"]
                for k in ("multi_select_gold2_5", "gold6_control", "ordering")), f"{side} total mismatch")
        require(sorted(solver["by_root"]) == roots, f"{side} root metric set mismatch")
        for key, _ in (*TYPE_LABELS, ("gold6_control", "Gold6")):
            metric = solver["by_type"][key]
            require(sum(solver["by_root"][r][key]["correct"] for r in roots) == metric["correct"],
                    f"{side} {key} root correct mismatch")
            require(sum(solver["by_root"][r][key]["calls"] for r in roots) == metric["calls"],
                    f"{side} {key} root calls mismatch")
            require(metric["accuracy"] == metric["correct"] / metric["calls"], f"{side} accuracy mismatch")
    chosen = selected["questions"]
    expected_chosen = {q["candidate_id"] for q in final["solver"]["trained"]["questions"]
                       if q["selected_by_policy"] and qmap[q["candidate_id"]]["role"] != "control6"}
    require({q["candidate_id"] for q in chosen} == expected_chosen, "Selection set mismatch")
    require(all(q == qmap[q["candidate_id"]] and q["role"] != "control6" for q in chosen),
            "Selected candidate differs from frozen pool")
    require(len(chosen) == len(expected_chosen) == final["selected_new_question_count"],
            "Selected count mismatch")
    require(dict(Counter(q["question_type"] for q in chosen)) == final["selected_type_counts"],
            "Selected type count mismatch")
    return final, behavior, screen, qmap, chosen, source_audit, export_audit


def render(data):
    final, behavior, screen, qmap, chosen, source_audit, export_audit = data
    summary = final["content"]
    metrics = final["solver"]
    roots = sorted(r["root_id"] for r in summary["roots"])
    selected_types = Counter(q["question_type"] for q in chosen)
    selected_gold = Counter(len(q["gold"]) for q in chosen if q["question_type"] == "multi_select")
    selected_by_root = Counter(q["root_id"] for q in chosen)
    excluded = [q for q in metrics["trained"]["questions"]
                if qmap[q["candidate_id"]]["role"] != "control6" and not q["selected_by_policy"]]
    excluded_reasons = Counter(q["selection_reason"] for q in excluded)
    at_one = sum(Fraction(q["full_partial_mean"]) == 1 for q in excluded)
    at_zero = sum(Fraction(q["full_partial_mean"]) == 0 for q in excluded)
    lengths = final["dialogue_lengths"]
    count = len(qmap)
    baseline_warnings = sum(r["decision"] == "WARNING" for r in screen["roots"] if r["selected"])
    delta_multi = metrics["trained"]["by_type"]["multi_select_gold2_5"]["accuracy"] - metrics["base"]["by_type"]["multi_select_gold2_5"]["accuracy"]
    delta_order = metrics["trained"]["by_type"]["ordering"]["accuracy"] - metrics["base"]["by_type"]["ordering"]["accuracy"]
    lines = ["# v117 结果解释与质量边界", "",
        f"固定 {summary['attempted_roots']} 个 root 完成 {summary['complete_A3B3_roots']} 个 A3＋B3；"
        f"{summary['ready_scene_count']}/{summary['expected_scene_count']} 场 READY，新增 {summary['accepted_new_turns']} 条发言。"
        f"{count}/{count} 题内容审核通过：Gold2/3/4/5 各 {summary['multi_gold_counts']['2']} 题、"
        f"Gold6 对照 {summary['gold6_controls']} 题、排序 {summary['content_pass_type_counts']['ordering']} 题。首稿保留，未因结果换 root、事件或改写题目。", "",
        f"前置 v116 漏斗是 {screen['candidate_denominator']} 个完整基线双人筛查 → {screen['eligible_root_count']} 个可续写 → "
        f"按冻结上限固定 {screen['selected_root_count']} 个进入本轮。{summary['complete_A3B3_roots']}/{summary['attempted_roots']} "
        f"是筛选后固定样本的流程完成率，不能写成任意 root 成功率，也不能将未扩充的其余基线算作扩充失败。", "",
        f"M0 与已有 M_clean_policy 各完成 {metrics['base']['valid_calls']}、{metrics['trained']['valid_calls']} 次有效回答，"
        "独立计分复算 PASS。本批仅比较现有两个模型，没有新训练，也没有合并训练池。", "",
        "|题型|M0 exact／调用|已有 M_clean_policy exact／调用|", "|---|---:|---:|"]
    for key, label in TYPE_LABELS:
        lines.append(f"|{label}|{metric_cell(metrics['base']['by_type'][key])}|{metric_cell(metrics['trained']['by_type'][key])}|")
    lines += ["", f"本批已有训练模型相对 M0：Gold2–5 多选差 {delta_multi * 100:+.1f} 个百分点，排序差 {delta_order * 100:+.1f} 个百分点。"
        "这是本批描述性结果；未测得新的训练收益，题目不同也不支持直接推断相对上一批的难度升降。", "",
        f"Gold6 仅作单列对照：M0 {metric_cell(metrics['base']['by_type']['gold6_control'])}，"
        f"已有训练模型 {metric_cell(metrics['trained']['by_type']['gold6_control'])}。全选在这组即正确，不进入新增训练候选或 Gold2–5 准确率。", "",
        "|root|M0 多选 Gold2–5|已有训练模型多选 Gold2–5|M0 排序|已有训练模型排序|新增入选候选|", "|---|---:|---:|---:|---:|---:|"]
    for root in roots:
        cells = [metric_cell(metrics[side]["by_root"][root][key])
                 for key in ("multi_select_gold2_5", "ordering") for side in ("base", "trained")]
        lines.append(f"|{root}|{'|'.join(cells)}|{selected_by_root[root]}|")
    lines += ["", f"按既定完整有效调用且 0<mean_partial<1 规则，保留 {len(chosen)} 道新增候选："
        f"多选 {selected_types['multi_select']}、排序 {selected_types['ordering']}；"
        f"入选多选 Gold2/3/4/5 分别为 {', '.join(str(selected_gold[k]) for k in (2,3,4,5))}。"
        f"未入选新增题 {len(excluded)} 道，其中 mean_partial=1 为 {at_one} 道、=0 为 {at_zero} 道。"
        f"完整原因计数：`{json.dumps(dict(excluded_reasons), ensure_ascii=False, sort_keys=True)}`。未入选不自动等于内容错误；入选仍是未入池候选。", ""]
    for side, label in (("base", "M0"), ("trained", "已有训练模型")):
        group = behavior["sides"][side]["groups"]["gold2_5"]
        kinds = group["error_kinds"]
        lines.append(f"{label} 的 Gold2–5 共 {group['calls']} 次回答：仅误选 {kinds.get('extra_only', 0)} 次、仅漏选 {kinds.get('missing_only', 0)} 次、"
            f"误选且漏选 {kinds.get('extra_and_missing', 0)} 次；选择数超过 Gold 数 {group['more_than_gold_selected']} 次，全选 {group['all_six_selected']} 次。"
            "误选指选了来源反驳的选项，漏选指漏掉来源支持项；这些计数描述行为，不能单独证明错误由模型或题目哪一方造成。")
        lines.append("")
    lines += [f"内容通过表示本次全文审核认为题干范围、人物归属、计划/完成状态、每个正误命题与来源一致；错项要求明确反驳，不以 unknown 冒充 false。"
        f"排序检查六个新场景的对话出现顺序及合理替代匹配。无损导出审计 PASS，核对 {export_audit['counts']['source_citations_checked']} 条引用。"
        "机械程序保护引用、哈希、计数和原始计分，不会自动证明语义真值；内容通过也不等于每题困难或绝无歧义。", "",
        "当前材料主要弱点是难度与自然度：不少多选是直接识别列名、数字、标签、时间或明确对立的请求状态；"
        "排序测的是对话出现顺序，每场一个锚点，场景与主题本身可能提供线索。B 路线多次采用朋友转述服务查询、澄清并记笔记的相似结构，"
        "部分对话反复解释 pending/planned/confirmed，具有流程痕迹。答案可追溯与表达自然、干扰项有区分度需分别评估。", "",
        f"本批只有 {len(roots)} 个六事实 family：每 root 的 Gold2/3/4/5 和 Gold6 复用同一组六事实，"
        f"不是 {summary['content_pass_type_counts']['multi_select']} 组独立多选知识。多选 6 个置换×3 次回答、排序 12 个置换×1 次回答都相关，"
        "两道排序也共用同一对话；不据调用数计算独立样本量或声称显著性。", "",
        f"四个 root 均已有训练侧与 Codex 曝光，不是未见 holdout。前置筛查中固定 root 有 {baseline_warnings} 个 WARNING；"
        f"本批逐场 actor_identity 检查 FAIL {summary['actor_identity_failed_scenes']}/{summary['actor_identity_checked_scenes']}、"
        f"WARNING {summary['actor_identity_warning_scenes']} 不撤销基线继承限制，也不是逐句人物正确率。"
        "作者与非作者审核者继承同一模型及会话背景，不构成异模型、外部语义认证或严格盲审。", "",
        f"来源程序审计 PASS 仍保留 {len(source_audit['warnings'])} 条警告。回执字段不完全统一：57679 A1 无独立同期 payload，A2/A3/B1 为明确标注的晚归档；"
        "B2/B3 则另有物化前 payload/serializer 绑定，不能与前几场混称。68549 A2 初始 read-context 身份误用、75223 A1 冗余错误身份读取、"
        "主作者误开自身审核入口被拒绝等，均保留原记录并另作补充；补充说明不能冒充同期凭证。详见 [流程证据边界](PROCEDURE_LIMITATIONS_ZH.md) "
        "和 evidence 中的身份/来源补充，不据机械 PASS 声称全部执行证据完全相同。", ""]
    if lengths:
        body = [r["dialogue_tokens_speakers_and_body"] for r in lengths]
        new_body = [r["new_dialogue_tokens_speakers_and_body"] for r in lengths]
        lines += [f"完整累计对话正文＋speaker 平均 {final['mean_dialogue_tokens_speakers_and_body']:.2f} tokens（{min(body)}–{max(body)}），"
            f"新增部分平均 {sum(new_body)/len(new_body):.2f} tokens。按本地 Qwen3-4B tokenizer 计算，不含 turn 编号。"
            f"计划冻结到自动交付 {final['timing']['batch_setup_to_delivery_seconds']/60:.2f} 分钟，Solver 包装器时间见 FINAL_STATUS.json；"
            "最终复核/封存耗时另计，并行收据不能相加作为批次墙钟。精确 Codex 模型 ID、调用 usage 和费用不可得，文本 tokens 不等于计费消耗。", ""]
    lines += ["后续可保持当前冻结规则扩大预先固定 root 的流程验证，统一新批次回执，并增加自然度、互动结构及干扰项区分度的独立评审。"
        "保留本批全部失败与限制，不依 Solver 分数回改本批来源、题目或 Gold；候选入池与训练应另行留痕。", "",
        "文件入口：[自动统计](REPORT_ZH.md)、[最终机器统计](FINAL_STATUS.json)、[误选漏选诊断](BEHAVIOR_DIAGNOSTICS.json)、"
        "[来源审计](SOURCE_PROVENANCE_AUDIT.json)、[引用导出审计](EVIDENCE_EXPORT_AUDIT.json)、"
        "[Solver 独立复算](INDEPENDENT_SOLVER_AUDIT.json)、[筛选候选](SELECTED_CANDIDATES.json)。"
        "完整对话和题答见 roots/root*/FULL_DIALOGUE.md、QUESTIONS_AND_ANSWERS.md；全部首稿/审核在 scenes/，原始回答在 evaluation/run/{base,trained}/responses/。", ""]
    interpretation = "\n".join(lines)
    entries = ["# v117 阅读入口", "", f"本轮固定 {summary['attempted_roots']} 个基线筛选后 root，"
        f"完成 {summary['complete_A3B3_roots']} 个 A3＋B3，保留 {len(chosen)} 道未入池候选。未启动新训练、未合并训练池。", "",
        "先读 [结果解释与质量边界](INTERPRETATION_ZH.md)，再看 [自动统计报告](REPORT_ZH.md)。"
        "8→6→4 的基线筛选、同族相关性、既有曝光及流程记录缺口均须保留。", "",
        "|root|完整累计对话|题目、答案与证据|", "|---|---|---|"]
    for root in roots:
        sub = "roots/root" + root.rsplit("-", 1)[1]
        entries.append(f"|{root}|[完整对话]({sub}/FULL_DIALOGUE.md)|[题答]({sub}/QUESTIONS_AND_ANSWERS.md)|")
    entries += ["", "[内容质量短评](evidence/CONTENT_QUALITY_REVIEW_ZH.md) 与 [流程证据边界](PROCEDURE_LIMITATIONS_ZH.md) "
        "解释自然度、题目难度和回执限制。全部首稿、失败、来源审核及原始 Solver 回答均保留。", ""]
    return interpretation, "\n".join(entries)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Exclusively create both Markdown reports after coordinator authorization")
    args = parser.parse_args()
    data = load_completed()
    interpretation, read_first = render(data)
    if args.write:
        targets = [(HERE / "INTERPRETATION_ZH.md", interpretation), (HERE / "READ_FIRST.md", read_first)]
        require(all(not p.exists() for p, _ in targets), "Report already exists; exclusive writes required")
        for path, contents in targets:
            with path.open("x") as handle:
                handle.write(contents)
    print(json.dumps({"input_validation": "PASS", "reports_written": args.write,
        "selected_candidate_count": len(data[4]), "interpretation_characters": len(interpretation)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
