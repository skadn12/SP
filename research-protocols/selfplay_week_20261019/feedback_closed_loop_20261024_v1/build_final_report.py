"""Build the final report only after numeric and target audits are complete."""
import datetime
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent


def read(name):
    return json.loads((P / name).read_text())


def bind(name):
    path = (P / name).resolve()
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def main():
    analysis = read("execution/ANALYSIS_SUMMARY.json")
    numeric = read("DIFFICULTY_COMPARISON.json")
    target = read("TARGET_ERROR_SUMMARY.json")
    audit = read("RESULT_INTEGRITY_AUDIT.json")
    # Audit schemas are deliberately preserved; the final report binds the original.
    assert audit["status"] == "PASS"
    assert target["status"] == "PASS_STRUCTURAL_AND_QUOTE_CHECKS"
    groups = {r["group"]: r for r in numeric["group_comparisons"]}
    primary = groups["NON_MC6_PRIMARY"]
    assert primary["F"]["correct"] == 12 and primary["G"]["correct"] == 8
    denom = analysis["denominators"]
    assert denom["returned_requests"] == 128 and denom["valid_responses"] == 121
    realized = target["summaries"]["REALIZED_NON_MC6"]
    assert realized["semantic_questions"] == 24
    calls = realized["dependent_display_responses"]
    final_hit = realized["confirmed_final_mechanism_error"]
    pattern = realized["target_final_pattern"]
    reason_only = realized["reason_only_target_error"]
    citation_target = realized["citation_only_target_error"]
    table = []
    labels = {"NON_MC6_PRIMARY": "主比较：非全选多选＋排序", "MC2_5_PRIMARY": "多选 Gold2–5", "ORDERING": "排序", "MC6_CONTROL": "Gold6 全选控制"}
    for name in ("NON_MC6_PRIMARY", "MC2_5_PRIMARY", "ORDERING", "MC6_CONTROL"):
        r = groups[name]
        f, g = r["F"], r["G"]
        table.append(f"| {labels[name]} | {f['semantic_questions']} | {f['correct']}/{f['requests']}（{100*f['exact_accuracy']:.2f}%） | {g['correct']}/{g['requests']}（{100*g['exact_accuracy']:.2f}%） | {r['F_minus_G']['exact_accuracy_pp']:+.2f} pp |")
    root_rows = [f"| {r['root_id'].split('-')[-1]} | {r['F']['correct']}/14 | {r['G']['correct']}/14 | {r['F_minus_G']['exact_accuracy_pp']:+.2f} pp |" for r in numeric["root_primary_comparisons"]]
    candidate_ids = ", ".join(f"`{cid}`" for cid in target["confirmed_final_error_candidate_ids"]) or "无"
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    text = f"""# 不训练的反馈闭环实验：最终报告

完成时间（UTC）：{now}

## 结论

本轮已完成“既有回答错误诊断＋出题反馈 → 一轮对话扩充 → 反馈/通用两臂出题 → 内容审核 → 固定 Solver 评估 → 目标错误复核”的流程闭环。**没有训练模型，也没有修改正式训练池、Dev/Test。**

**本轮不支持“反馈指导使题目总体更难”。** 同源、同 root/操作/Gold 配额的主比较中，反馈臂 F 正确率 21.43%，通用臂 G 为 14.29%，F 比 G 高 7.14 个百分点；多选持平，排序 F 更容易。这里是4个发现集 Train roots上的描述性结果，不能据此宣布反馈无用或稳定有效。

## 实际完成规模与流程

- roots：218519、222937、225575、229686。v125 父来源后追加一轮 R03，每 root 48 turn，共192新 turn；最终来源共914 turn。
- A 为旧事件续写3场，B为预留 Train 原生事件忠实转述3场；按 A→B累计为同一来源，非独立分支。每场8 turn，旧前缀逐项保留。
- F/G共用相同扩充来源。F接收19张已审核公开候选弱点卡及出题反馈；G接收相同题型、操作、质量标准和Gold配额，隔离弱点卡、预期错误与F题。卡片实际交付、读取和计划使用已记入消费 ledger。
- 每臂32语义题：Gold2、3、4、5、6各4道，多选共20道；三种排序操作各4道，排序共12道。总计64语义题，每题两种程序打乱展示，128次请求。
- 每槽一份初稿、至多一次内容修订。初审53 PASS／11 HOLD／0 FAIL；11题一次修订后全部通过。64题初审独立推导Gold均与作者一致；HOLD集中在来源必要性和题面捷径，而非已确认Gold错误。
- 全部64题和目标判据均在 Solver 启动前冻结。Solver分数出现后没有改题、换Gold、追加样本或选择性重试。
- 目标实现预审为27/32 PASS、5 HOLD；主比较的F非全选题24/28实现原卡窄触发。目标HOLD仍照常参与难度主比较，其错答不算原卡命中。

## 固定 Solver 与作答记录

固定模型：`restart-n5-runs/day5_B1_schedule_v1/merged_model`，即既有B1模型；本轮不训练，也不是base模型。FULL完整来源，THINK_OFF，输出 `item_evidence → answer_reason → <box>答案</box>`。这些是模型原始可见结构化解释，不等同于已证实忠实的隐藏思考轨迹。

temperature=0.8、top_p=0.9、top_k=-1、repetition_penalty=1、engine seed=0；batch8、context16384、最多输出6144、prefix cache关闭。实际最大输入9579 tokens，无输入截断。生成阶段不读Gold，原始输出全部封存后才评分。

128/128请求均提交并返回，均正常stop，无长度终止或后端失败；121/128满足完整作答合同。原始运行墙钟为{analysis['raw_wall_seconds']:.2f}秒（含本次模型启动），累计输入{analysis['total_prompt_tokens']:,} tokens、输出{analysis['total_output_tokens']:,} tokens；这不是整轮人工式编写/审核总耗时。

## 难度主结果

分母为已冻结的有效内容题之全部展示请求。每道语义题重复两次，**不是128道独立新题**。下表“每臂题数”按语义题，“正确率”按两展示请求；F−G为正表示反馈题在本轮更容易。

| 范围 | 每臂题数 | F：反馈指导 | G：通用对照 | F−G |
|---|---:|---:|---:|---:|
{chr(10).join(table)}

主比较只含每臂28语义题／56展示。MC6全真题单列，不能用它提高干扰项能力或弱点题难度的结论。始终全选在Gold2–5子集的exact基线为0；全选控制的正确并不证明能排除混淆项。

只统计合同有效回答时，主比较 F=12/52（23.08%）、G=8/53（15.09%）；排序exact F=5/20（25.00%）、G=1/21（4.76%）。排序pairwise严格口径 F=242/360（67.22%）、G=234/360（65.00%）；合同有效条件为242/300（80.67%）与234/315（74.29%）。两种口径均未出现F更难的总体方向。

多选Gold2–5宏平均 precision/recall/F1：F=75.05%／97.03%／83.51%，G=74.64%／93.13%／82.07%。F累计误选35、漏选3；G误选33、漏选5。这里的计数跨两展示，不能把它们视为独立语义错误实例。两臂都以过选而非漏选为主；仅“选得多”还不是已定位的推理机制。

## 语义题、root与排列稳定性

28个匹配语义槽中，按两展示平均exact：F更难5槽、相同16槽、更容易7槽。主比较两展示均对 F=3/28、G=2/28；至少一次对 F=9/28、G=6/28；两展示均有效且原始标签预测一致 F=7/28、G=9/28。

| root | F主比较正确 | G主比较正确 | F−G |
|---|---:|---:|---:|
{chr(10).join(root_rows)}

总差异集中在229686；其余三个root的主比较exact相同。不能把128次展示当独立样本做显著性论证，也不能从单个root外推。temperature0.8与不同排列同时存在，因此排列不一致不是纯选项位置因果效应。

## 作答合同问题与答案本身

7个合同无效全部出现在排序题：5次排序项不全，1次重复引用，1次排序项无证据锚点；不应统称为JSON格式失败。F为4次，G为3次。

另做不改变官方分数的final-answer-only诊断：F主比较13/56、G8/56；F排序6/24、G1/24。差异是F-222937-ORDER2的perm_01完整排序正确，但一项没有证据锚点，严格合同未通过。因此准确率下降不能全部解释成关系推理难度；本轮去掉证据合同惩罚仍没有反馈更难的方向。详见 `ANSWER_CONTRACT_DIAGNOSTIC.json`。

## 预设弱点是否出现

后审使用启动前冻结的具体标签/排序对和排除条件，逐一检查32道F题、64次展示，包括最终正确的作答；使用原目标预审员复核，并非新的独立审核样本。

主探针中，实现原卡目标的24语义题／{calls}展示：

- 最终答案出现预设目标标签/排序对错误：{pattern['responses']}/{calls}展示，涉及{pattern['semantic_questions_any_display']}/24题；这只是目标相关错答，不自动认定机制。
- 可见解释足以支持同一窄机制、且最终目标关系也错：{final_hit['responses']}/{calls}展示，涉及{final_hit['semantic_questions_any_display']}/24题；两展示均出现的为{final_hit['semantic_questions_both_displays']}/24题，覆盖{len(final_hit['roots_with_any'])}个发现root。按全部28个F非控制生成槽算，至少一次此类命中的产出为{final_hit['semantic_questions_any_display']}/28。
- 整题严格正确但目标理由错误：{reason_only['responses']}展示、{reason_only['semantic_questions_any_display']}题，单列，不改写最终正确率。
- 仅特定目标阶段引用错误：{citation_target['responses']}展示、{citation_target['semantic_questions_any_display']}题，单列；普通漏引或坐标错不自动算该目标。

至少一次确认最终目标机制错误的题：{candidate_ids}。

具体确认支持的是**新题中的局部可观察错误**。不能从同题两种排列宣布稳定弱点，也不能据F单臂机制后审声称比G多产出机制命中；G没有同卡的配套机制审核。引用不足、取值错误、机制不明和目标HOLD均保持分离，完整记录见 `TARGET_ERROR_SUMMARY.json`。

普通引用错位或支撑不足仍很频繁：24道已实现目标的非控制题中，44/48次展示存在citation_defect，23/24题至少一次。这个AI语义审核标签不同于7次严格合同无效，也不等同于原卡机制错误；它不推翻已审核Gold，但限制了将结构化解释整体当成可靠证据链的做法。

## 完整可见输出示例

四个示例在启动前选定，未按结果挑选，保留题面、全部选项、展示Gold/预测、原始标签映射和完整原始可见输出：`EXAMPLES_ZH.md` 与 `EXAMPLES_PRIVATE.json`。

其中 F-218519-MC2的perm_00，展示B选项声称时间差为“两小时十五分钟”；模型理由实际算出“两小时四十五分钟”，仍标为supported并选入，最终展示Gold为C F，预测为B C F。这是可直接检查的“计算结果与支持判断不一致”局部错误。G对应题也出现单位/支持判断错误，因此这项观察本身不能替代F优于G的收益证明。

## 限制与下一步

1. 两臂共享已经受中性关系目标指导的R03扩充来源，只比较同源出题指导；没有单独测量扩充反馈因果收益，也没有严格匹配的旧来源前后题对，不能把旧30题16/30直接与本轮相减。
2. F/G的操作与配额匹配，但题目不是等难平行题，作者生成有随机性。通用质量提醒发出时G初稿已冻结、F部分仍未冻结，时点差异已在 `AUTHORING_PROCESS_DEVIATIONS.json` 披露；初稿有效率差不能当反馈因果收益。
3. 作者与审核者属于同模型家族，隔离上下文不能消除共同偏差；这是AI内容审核，不是外部人类审核。218519、229686继续标为EDITED_SYNTHETIC，题目按冻结来源判定，不宣称恢复原始真实事实。
4. 当前是4个发现Train roots上的开发检验，尚未建立未见root的正式复现。未训练，因此没有“弱点被修复”或验证集提升结论。

下一步应优先将可见的具体错误转成可否证的窄反馈，并制作同关系“保留触发／移除触发”的新题对，在保留Train roots上复核，固定相同预算与审核。尤其把“算对数值仍接受不符断言”“把请求/计划当已完成”“将错误阶段引文作为完成证据”分开，分别定义正确行为和排除条件；不靠更长提示或单纯降低exact作为成功标准。当前结果不支持直接宣称反馈有难度收益，也不需要为了这一轮的验证去训练模型。

## 文件导航

- 对话与来源冻结：`sources/`、`SOURCE_FREEZE.json`、`SOURCE_ASSEMBLY_READBACK_CHECK.json`。
- 初稿/失败/一次修订：`question_authoring/F/`、`question_authoring/G/`、`CANDIDATES_INITIAL_PRIVATE.json`、`CONTENT_REVIEW_INITIAL.json`。
- 最终题与内容审核：`CANDIDATES_PRIVATE.json`、`CONTENT_REVIEW_AI.json`、`CONTENT_FREEZE.json`。
- 启动前目标标准：`TARGET_PRE_SOLVER_FREEZE.json`、`target_review/*/PRE_SOLVER_TARGET_REVIEW.json`。
- 实际模型输入与完整输出：`execution/prompts/`、`execution/raw/`、`execution/RAW_RESPONSE_FREEZE.json`。
- 官方逐题评分与汇总：`execution/SCORES.json`、`execution/ANALYSIS_SUMMARY.json`。
- 主对照与分层表：`DIFFICULTY_COMPARISON.json`、`DIFFICULTY_TABLES_ZH.md`。
- 机制后审：`target_review/*/POST_SOLVER_TARGET_REVIEW.json`、`TARGET_ERROR_SUMMARY.json`。
- 反馈实际消费及过程偏差：`FEEDBACK_CONSUMPTION_LEDGER.json`、`PROVENANCE_SUMMARY_ZH.md`、`AUTHORING_PROCESS_DEVIATIONS.json`。
- 可复查例子：`EXAMPLES_ZH.md`；完整性核查：`RESULT_INTEGRITY_AUDIT.json`。
"""
    with (P / "REPORT_ZH.md").open("x") as stream:
        stream.write(text)
    status = {"schema_version": "feedback_closed_loop_final_status_v1", "status": "COMPLETE_NO_TRAINING",
        "completed_at_utc": now, "training": False, "formal_training_pool_modified": False,
        "dev_test_modified": False, "completed_steps": ["feedback_consumption", "one_round_expansion", "matched_F_G_authoring", "content_review_and_single_revision", "pre_solver_target_freeze", "128_fixed_solver_requests", "raw_freeze_then_scoring", "post_solver_target_review", "difficulty_comparison", "report"],
        "feedback_harder_supported_this_run": False, "primary_F_correct": 12, "primary_G_correct": 8,
        "primary_requests_each_arm": 56, "primary_semantic_questions_each_arm": 28,
        "interpretation": "Operational loop complete; descriptive feedback difficulty benefit not supported. Local observed target errors are not stable unseen-root reproduction or repair.",
        "bindings": [bind(n) for n in ["REPORT_ZH.md", "execution/ANALYSIS_SUMMARY.json", "DIFFICULTY_COMPARISON.json", "TARGET_ERROR_SUMMARY.json", "RESULT_INTEGRITY_AUDIT.json", "FEEDBACK_CONSUMPTION_LEDGER.json", "ANSWER_CONTRACT_DIAGNOSTIC.json", "EXAMPLES_ZH.md"]]}
    with (P / "FINAL_STATUS.json").open("x") as stream:
        json.dump(status, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": status["status"], "feedback_harder_supported": False,
        "local_confirmed_final_mechanism_questions": final_hit["semantic_questions_any_display"]}))


if __name__ == "__main__":
    main()
