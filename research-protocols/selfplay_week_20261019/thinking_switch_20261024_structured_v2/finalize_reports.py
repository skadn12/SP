"""Additive, offline report assembly. Does not alter frozen generation artifacts."""
import collections
import hashlib
import json
from pathlib import Path

S = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def binding(path):
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verify(b):
    got = binding(Path(b["path"]))
    assert got["sha256"] == b["sha256"], b["path"]
    if "bytes" in b:
        assert got["bytes"] == b["bytes"], b["path"]


def main():
    comparison = read(S / "execution/COMPARISON.json")
    packet_path = S / "semantic_audit/SEMANTIC_AUDIT_PACKET.json"
    packet = read(packet_path)
    contract = read(S / "semantic_audit/REVIEW_CONTRACT.json")
    expected = {r["request_id"]: (c, r) for c in packet["cases"] for r in c["responses"]}
    records, review_bindings = [], []
    for reviewer, key in [("reviewer_a", "records"), ("reviewer_b", "reviews")]:
        path = S / "semantic_audit" / reviewer / "REVIEWS.json"
        doc = read(path)
        verify(doc["packet_binding"])
        assert doc["packet_binding"]["sha256"] == binding(packet_path)["sha256"]
        review_bindings.append(binding(path))
        for record in doc[key]:
            assert all(field in record for field in contract["per_response_fields"])
            case, response = expected[record["request_id"]]
            assert record["mode"] == response["mode"]
            assert record["candidate_id"] == case["candidate_id"]
            assert (record["final_correct"] == "PASS") == response["strict"]["exact"]
            assert record.get("verified_display_answer", record.get("source_verified_gold")) == case["display_gold"]
            assert len(record["item_checks"]) == len(case["display"])
            assert {item["label"] for item in record["item_checks"]} == set(case["display"])
            for b in [case["source_binding"], case["source_transcript_binding"], response["raw_binding"]]:
                verify(b)
            records.append({"reviewer_id": reviewer, **record})
    assert len(records) == len(expected) == len({r["request_id"] for r in records}) == 12
    assert {r["request_id"] for r in records} == set(expected)
    fields = ["final_correct", "final_reasons_supported", "citations_supported", "item_final_consistent", "thinking_factuality", "thinking_adds_verifiable_error_beyond_own_final", "thinking_adds_verifiable_correct_check_beyond_own_final"]
    counts = {mode: {field: dict(collections.Counter(r[field] for r in records if r["mode"] == mode)) for field in fields} for mode in ["THINK_OFF", "THINK_ON"]}
    on_records = [r for r in records if r["mode"] == "THINK_ON"]
    corrections = [r["request_id"] for r in on_records if r.get("self_corrections_or_rejected_hypotheses", r.get("thinking_self_corrections", []))]
    local_cases = [r["request_id"] for r in on_records if r["thinking_adds_verifiable_error_beyond_own_final"] == "OBSERVED"]
    assert len(local_cases) == 1
    local = next(r for r in records if r["request_id"] == local_cases[0])
    assert all(e["feedback_eligible"] is False and e["final_role_correct"] is True for e in local["thinking_new_error_evidence"])
    summary = {
        "schema_version": "fixed_semantic_audit_summary_v1",
        "status": "COMPLETE_FIXED_SUBSET",
        "selection_binding": binding(S / "execution/SEMANTIC_AUDIT_SELECTION.json"),
        "packet_binding": binding(packet_path),
        "review_bindings": review_bindings,
        "coverage": {"questions": 6, "responses": 12, "roots": 4, "display": "perm_00", "item_checks": sum(len(r["item_checks"]) for r in records), "missing_time_slots_not_replaced": 2},
        "coverage_and_bindings_pass": True,
        "source_gold_concerns_in_fixed_subset": [],
        "per_mode": counts,
        "incremental_evidence": {
            "extra_correct_check_responses": {"observed": 5, "denominator": 6, "interpretation": "Includes minor source checks and time-format conversions; not five new substantive deductions or failure diagnoses."},
            "trace_local_role_inconsistency": {"count": 1, "request_ids": local_cases, "final_role_correct": True, "feedback_eligible": False},
            "new_error_beyond_own_final_still_unresolved_or_adopted_in_final": {"confirmed_count": 0, "denominator": 6, "uncertainty_note": "One response has UNKNOWN thinking factuality for an unsupported identity aside; zero confirmed cases is not proof of absence."},
            "new_formally_reproduced_stable_weakness": {"count": 0, "denominator": 6},
            "responses_with_explicit_self_corrections": {"count": len(corrections), "request_ids": corrections},
            "errors_already_in_own_final_not_incremental": ["PDF1-218519-MC2__perm_00__FULL__THINK_ON__structured_v2", "PDF1-229686-MC3__perm_00__FULL__THINK_ON__structured_v2"],
        },
        "disclosure": {"reviewers": "Two separate AI reviewers on disjoint assigned questions, shared workspace, Gold visible; not external blind review or duplicate inter-rater agreement.", "limits": "Only the preselected quantity/clock subset; no inference to all 30 questions or all seven operations; visible traces are not proof of internal faithfulness."},
        "normalized_records": records,
    }
    dump(S / "semantic_audit/SUMMARY.json", summary)
    semantic_text = """# 固定子集语义审核结论

生成前选定每个 root 的首个数量题与钟面排序题，两个 root 缺排序题，保留缺槽。因此实际为 **6题、12响应、68个逐项检查**，不是从成功响应里挑选。

两位独立 AI 审核员分别审核互不重叠的三题，均读完整来源、OFF/ON全文并逐项核验。Gold可见、共享workspace，不是外部盲审；每题只由一位审核员审核，没有双评一致性估计。绑定、覆盖、Gold与来源一致性检查通过，仅覆盖这个子集。

| 语义维度（响应级，所有逐项均通过才记PASS） | OFF | ON |
|---|---:|---:|
| 最终答案正确 | 1/6 | 5/6 |
| 所有最终理由均受来源与题意支持 | 0/6 | 4/6 |
| 所有最终引用均支撑其声称内容 | 1/6 | 0/6 |
| 逐项判断/时刻与最终答案一致 | 5/6 | 6/6 |

以上是固定子集人工语义检查，不是整批准确率。理由维度采用严格字面口径：OFF 的218519排序题钟值与答案全对，但把尚未发生的出行写成“departed”，因此该响应全理由PASS未通过；这是状态措辞风险，不是钟面排序失败。

## thinking 增量是什么

- 6个ON响应中5个出现自身final没有展开的正确核验信息。有些只是具体字段、12/24小时制换算等微小增量，不能称为5个新的实质推导或弱点。
- 1个ON响应（229686-ORDER1）在轨迹里把“08:15前到达”两次误述为“08:15前离开”。final已正确写arrival且排序正确；只记录为局部角色不一致，不认定持续错误、最终采用或正式弱点。
- **已确认的、超出自身final且最终仍未解决或被采纳的新错误证据：0/6；本轮新正式复现弱点：0。** 另1例的thinking factuality为UNKNOWN；零项已确认证据不能解释为已证明没有其他错误，也不证明thinking从不提供诊断信息。
- 4个ON响应有明确自我纠正，不能把候选错误直接当作未解决错误。218519-MC2的“15分钟=0.5小时”、229686-MC3的日期混算/差值错误在自身final已可见，因此不算thinking新增诊断。
- 222937-MC5的身份旁支采纳情况不清楚，thinking factuality保留UNKNOWN，未强行认定新错误或全对。

## 对反馈的含义

答对、理由正确、引用支撑、轨迹局部一致性必须分别记录。最明显的待核查问题是**合法引用ID并不保证语义支撑**：12份响应中11份至少有一处错位或不足。它不意味着所有引用都错，也不意味着Gold错误。

thinking可以继续作为诊断支路保存，以帮助核对计算与自纠；目前不能因过程更长或答案改善就自动升级正式弱点、切换生产默认、宣称Challenger已理解弱点或证明反馈训练收益。正式弱点仍需不同展示/来源下复现，反馈收益仍需独立对照。

详见 `SUMMARY.json` 的统一记录及 `reviewer_a/`、`reviewer_b/` 的逐项证据。
"""
    (S / "semantic_audit/REPORT_ZH.md").write_text(semantic_text, encoding="utf-8")
    rows = {(r["mode"], r["group"]): r for r in comparison["main"]}
    a, b = rows[("THINK_OFF", "ALL")], rows[("THINK_ON", "ALL")]
    ratio = b["mean_output_tokens"] / a["mean_output_tokens"]
    table = []
    for group, label in [("MC2_5_PRIMARY", "多选 Gold2–5（主分析）"), ("ORDERING", "排序"), ("MC6_CONTROL", "多选 Gold6（控制）"), ("ALL", "全部")]:
        x, y = rows[("THINK_OFF", group)], rows[("THINK_ON", group)]
        table.append(f'| {label} | {x["correct"]}/{x["requests"]}（{x["exact_accuracy"]:.1%}） | {y["correct"]}/{y["requests"]}（{y["exact_accuracy"]:.1%}） |')
    off_ops = {r["task_operation"]: r for r in comparison["breakdowns"]["task_operation"] if r["mode"] == "THINK_OFF"}
    on_ops = {r["task_operation"]: r for r in comparison["breakdowns"]["task_operation"] if r["mode"] == "THINK_ON"}
    op_table = [f'| {op} | {off_ops[op]["correct"]}/{off_ops[op]["requests"]} | {on_ops[op]["correct"]}/{on_ops[op]["requests"]} |' for op in sorted(off_ops)]
    report = f"""# B1 thinking 开关：完成报告（2026-10-24）

**当前B1（基于Qwen3-4B）确实能够生成thinking。结构兼容对照中，开启后60/60响应生成非空、闭合的thinking；严格完全正确率33/60→44/60。收益集中在数量/时钟比较，尚未证明thinking带来新的正式弱点或反馈训练收益。**

## 设计与分母

固定4个历史Train roots、30道去重题，每题两个既定排列；每种模式60次、合计120次新请求。20道多选（Gold2–5共16题，Gold6共4题），10道排序。两排列不能当作独立题，同root题目也可能相关。仅使用FULL输入，没有训练、扩充或更改Gold；未使用Dev/Test。

两组正文、来源、题目、显示、seed、采样参数及预算完全配对，仅切换聊天模板`enable_thinking`。双方共同使用reasoning-aware最终JSON约束：`reasoning_parser=qwen3, enable_in_reasoning=False`；这里False指thinking阶段不套JSON grammar，并非关闭thinking。详见冻结的`PROTOCOL_ZH.md`和加载凭据。

模型：`restart-n5-runs/day5_B1_schedule_v1/merged_model`；tokenizer：`/root/autodl-tmp/Qwen3-4B`。temperature=0.8，top_p=0.9，top_k=-1，输出上限6144，context=16384，最大prompt7830；无输入截断、无重试补样。先封存全量原始响应，后读取Gold评分。

## 正式结果

| 分组（作答次数，非独立题数） | thinking关 | thinking开 |
|---|---:|---:|
{chr(10).join(table)}

主指标要求最终合同有效、正常stop且答案完全正确；3次ON无效响应全部留在分母并计错。Gold6控制列取模型真实完全正确率，未使用历史代码仅反映Gold全选占比的误导字段。

- 合同有效：OFF 60/60；ON 57/60。实际非空闭合thinking：OFF 0/60；ON 60/60。
- 同一展示配对：改善16次、退化5次、双对28次、双错11次；按每题两排列累计，10题改善、3题退化、17题持平。
- 两排列均答对：14/30→20/30；恢复canonical标签后两次预测一致：18/30→22/30（分母始终30）。
- Gold2–5多选macro F1：0.8382→0.8991；排序pairwise：0.8368→0.8895。均为辅助指标，不替代完全正确率。

## 哪些操作改善

| 操作 | OFF | ON |
|---|---:|---:|
{chr(10).join(op_table)}

数量比较为4题×两排列，显式时钟排序为2题×两排列，都是小样本。操作表包含Gold6控制：MC_CONSTRAINT_CONJUNCTION的16个请求中含8个Gold6请求，不能将该整格解读成非全选能力结果；非全选主结果应看上表Gold2–5行。话语重建反而从3/8降到2/8；不能概括为所有题型都提升。

## 成本和失败

平均总生成tokens：392.6→1141.87，**约{ratio:.2f}倍**；中位数380→928。ON平均实际thinking为686.93 tokens（按原始token IDs中thinking边界计数）。这是输出token成本对比，未测得完整计费成本。

本轮GPU整轮运行约331.89秒（5分32秒），包含双方共用模型加载及交错批次；没有单请求latency，不能宣称某一臂的独立耗时或按比例换算。

ON的三次失败均保留，未修复、未替换：

1. `PDF1-222937-MC4 / perm_01`：输出达6144上限，thinking已闭合，截断发生在最终JSON阶段。
2. `PDF1-222937-ORDER3 / perm_00`：排序漏项（INCOMPLETE_ORDERING）。
3. `PDF1-225575-MC6 / perm_01`：重复引用（DUPLICATE_REFERENCE）；可读答案虽对，严格仍计错。

因此ON可读答案次指标45/60，不等于正式44/60。详见`execution/FORMAT_FAILURES.json`。

## 语义审核与诊断收益

预选6题×两模式的12响应由两位AI审核员分别读完整来源、逐项核验，共68个item检查。Gold可见、共享workspace，不是外部盲审或双评一致性检验。两个缺失的时钟题槽保留，不替换。

固定子集中答案正确1/6→5/6，所有最终理由受支持0/6→4/6，但所有引用语义均充分仅OFF 1/6、ON 0/6；11/12响应至少有一处引用错位或不足。严格自动评分中的引用合法性与这里的语义支撑不是同一个标准。

6个ON中5个有额外正确核验信息，部分只是微小字段或时间制换算；4个有明确自纠。仅1个出现final之外的局部arrival/leave角色误述，但final已正确、无排序错误关联，不能晋升弱点。**已确认的新增且最终未解决/采用错误证据0/6；新正式复现弱点0。** 另1例thinking factuality为UNKNOWN，不把零项已确认证据解释成没有其他错误。15分钟换算、日期混算等另一些错误在final本身已能看到，不算thinking新增信息。详细边界见`semantic_audit/REPORT_ZH.md`。

## 前置自由生成实验也完整保留

`../thinking_switch_20261024_v1/`是独立前置工程探查，两组都不加结构约束。原JSON合同有效率双方均0/60；最终JSON内可读合法答案正确15/60→47/60只是次指标。失败主要涉及item_evidence结构/引用要求，不能解释为两组推理都为零。两臂平均输出196.47→898.8 tokens，整轮约281.15秒。

v1格式失败后才新建当前v2，共同加入兼容thinking的最终JSON约束。两轮分开报告，v2不覆盖v1，不跨版本归因或合并分数。v1原计划语义审核已明确延期至本轮v2，不宣称v1完成审核。

## 当前决定及未完成的因果证明

本轮完成thinking可运行性、配对答题表现、原始过程保存及固定子集语义审核。**支持继续把thinking作为诊断探索支路保存，但暂不据此切换生产默认或自动形成正式弱点。** 下一项证据应是：来源/引用与计算问题独立复现后，冻结弱点卡，比较Challenger有反馈与无反馈生成任务的针对性及后续训练收益。

这次没有新训练、没有完成上述反馈对照，也不能证明泛化、内部推理忠实性或“模型已经理解自身弱点”。仅30题、4个Train roots，每题两依赖展示，结论是探索性的。

## 文件与核验

- `execution/COMPARISON.json`：正式数值主源。该文件在语义审核完成前生成，其中semantic_audit的PENDING仅为当时快照；最终审核状态看`FINAL_STATUS.json`与`semantic_audit/SUMMARY.json`，不回写历史文件。
- `EXAMPLES_ZH.md`：预选第一root两道题，完整thinking与final原文（含一题两组均错）。
- `execution/raw/`：全部120原始响应、token IDs和后端返回；`execution/RAW_RESPONSE_FREEZE.json`绑定先封存再评分。
- `INDEPENDENT_DESIGN_CHECK.json`：配对设计检查通过；`INDEPENDENT_RESULTS_CHECK.json`：21项结果复算通过。
- `CLOSURE_CHECK.json`：916次绑定检查、674唯一文件，120次均返回，119stop+1length。完整性通过不等于全部答案有效。
- `OLD_P_FINAL_MANIFEST_CHECK.json`：旧P的771产物+4来源，共775文件hash/bytes全部未变。
- `FINAL_MANIFEST.json`与`FINAL_MANIFEST_CHECK.json`：最终交付文件清单及校验。
"""
    (S / "REPORT_ZH.md").write_text(report, encoding="utf-8")
    (S / "README_ZH.md").write_text("""# B1 thinking开关实验交付

先读 [完成报告](REPORT_ZH.md)，原始过程见 [预选完整示例](EXAMPLES_ZH.md)，固定子集的引用/理由审查见 [语义审核结论](semantic_audit/REPORT_ZH.md)。

- 正式数值：`execution/COMPARISON.json`；语义最终状态：`semantic_audit/SUMMARY.json`与`FINAL_STATUS.json`。
- 冻结设计：`PROTOCOL_ZH.md`、`execution/PROTOCOL_FREEZE.json`；完整原始响应：`execution/raw/`。
- 独立检查：`INDEPENDENT_DESIGN_CHECK.json`、`INDEPENDENT_RESULTS_CHECK.json`、`CLOSURE_CHECK.json`。
- 前置自由生成实验保留于相邻`thinking_switch_20261024_v1/`，两轮不可合并或替换。
- 最终文件哈希：`FINAL_MANIFEST.json`，校验结果：`FINAL_MANIFEST_CHECK.json`。

本次只完成冻结题的thinking开关探索及语义审核，未改旧题、Gold、模型或训练池；未启动训练。thinking可运行且本小样本准确率提高，但新增正式弱点和反馈训练因果收益尚未建立。
""", encoding="utf-8")
    dump(S / "FINAL_STATUS.json", {
        "status": "COMPLETE_FOR_REGISTERED_THINKING_SWITCH_EXPERIMENT",
        "semantic_audit_status": "COMPLETE_FIXED_SUBSET",
        "semantic_summary_binding": binding(S / "semantic_audit/SUMMARY.json"),
        "comparison_binding": binding(S / "execution/COMPARISON.json"),
        "semantic_snapshot_note": "COMPARISON.semantic_audit is an earlier pending snapshot; this additive status supersedes only that status, not numbers.",
        "generation_requests": 120, "unique_questions": 30, "roots": 4,
        "decision": "CONTINUE_DIAGNOSTIC_EXPLORATION_NO_PRODUCTION_DEFAULT_SWITCH",
        "new_formal_weaknesses": 0, "feedback_training_causal_benefit_proven": False,
        "new_training_started": False, "gold_or_training_pool_changed": False,
    })
    print(json.dumps({"status": "REPORTS_WRITTEN", "semantic_coverage": summary["coverage"], "counts": counts, "explicit_self_correction_responses": len(corrections)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
