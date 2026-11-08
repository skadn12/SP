# V2.1 真实生成复盘与 V2.2 方案

日期：2026-11-02。状态：**分析及方案已交付；V2.2 未实施，下一轮实验未启动**。

本次通读复核两个 root 的375轮完整来源、12个R02场景与全部16条原候选关系，另检19项补充关系假设。结论是：限定来源中的高阶关系供给受限，现有选择也未要求L3/L4；两道冻结目标从一开始就是L1。尚未证实存在“漏掉同范围安全L3/L4”“把已有困难关系排到简单关系后”或“L3目标被构造退化成L1”。104814选项冻结失配另有确定代码根因。

## 交付索引

| 用户要求 | 文件 | 内容 |
|---|---|---|
| 真实生成失败根因报告 | [ROOT_CAUSE_REPORT.md](ROOT_CAUSE_REPORT.md) | A/B/C/D证据强度、完整来源结构、两题生成链、NO_SAFE解释与研究限制 |
| 候选关系难度分析 | [CANDIDATE_DIFFICULTY_ANALYSIS.md](CANDIDATE_DIFFICULTY_ANALYSIS.md) / [JSON](CANDIDATE_DIFFICULTY_ANALYSIS.json) | 全16条关系的证据、依赖、状态、最简推理、替代路径、自然价值与等级 |
| V2.1阶段定位 | [V2_1_STAGE_DIAGNOSIS.md](V2_1_STAGE_DIAGNOSIS.md) | 代码与prompt定位、必要性与难度区别、冻结故障链和测试覆盖缺口 |
| V2.2精确修改方案 | [V2_2_PRECISE_CHANGE_PLAN.md](V2_2_PRECISE_CHANGE_PLAN.md) | 复用/修改边界、字段与选择契约、题面条件化审计、规范渲染方案与未来回归 |
| 下一轮真实实验计划 | [NEXT_REAL_EXPERIMENT_PLAN.md](NEXT_REAL_EXPERIMENT_PLAN.md) | 同两个root及固定范围、授权前提、≤4题、统一质量标准和停止条件 |

## 完整复核附录

- [103866逐轮与逐关系分析](root_audits/ROOT_103866_ANALYSIS.md) / [JSON](root_audits/ROOT_103866_ANALYSIS.json)：197轮，六场景，七原关系、九补充假设。
- [104814逐轮与逐关系分析](root_audits/ROOT_104814_ANALYSIS.md) / [JSON](root_audits/ROOT_104814_ANALYSIS.json)：178轮，六场景，九原关系、十补充假设。
- [只读基线](READ_ONLY_BASELINE.json) / [收尾完整性核验](INTEGRITY_CHECK.json)：旧run与三份pipeline共89文件逐字节哈希复核；375轮覆盖与引文、16关系及文档链接核验。此核验不是新生成能力回归实验。

## 原始证据入口

- [历史运行报告](../base_qa_v2_1_dual_root_real_20261101/REPORT_ZH.md)、[指标](../base_qa_v2_1_dual_root_real_20261101/METRICS.json)、[生成账本](../base_qa_v2_1_dual_root_real_20261101/GENERATION_LEDGER.json)。
- [103866作者Discovery](../base_qa_v2_1_dual_root_real_20261101/cc-train-episode-103866/author/DISCOVERY.json)、[TARGET](../base_qa_v2_1_dual_root_real_20261101/cc-train-episode-103866/author/TARGET_01.json)、[最终质量记录](../base_qa_v2_1_dual_root_real_20261101/cc-train-episode-103866/review_01/FINAL_QUALITY_DECISION.json)。
- [104814作者Discovery](../base_qa_v2_1_dual_root_real_20261101/cc-train-episode-104814/author/DISCOVERY.json)、[TARGET](../base_qa_v2_1_dual_root_real_20261101/cc-train-episode-104814/author/TARGET_01.json)、[冻结诊断](../base_qa_v2_1_dual_root_real_20261101/cc-train-episode-104814/review_01/FREEZE_ORDER_DIAGNOSTIC.json)、[最终质量记录](../base_qa_v2_1_dual_root_real_20261101/cc-train-episode-104814/review_01/FINAL_QUALITY_DECISION.json)。
- [精确版本资格状态](../base_qa_v2_1_dual_root_real_20261101/MATERIAL_AUTHORIZATION_STATUS.json)、[已有来源准入记录](../day3_production_v1/inventory/PRIORITY_SOURCE_ADMISSION.json)、[已有完整素材审核](../day3_production_v1/inventory/PRIORITY_FULL_SOURCE_REVIEW.json)。

## 本轮边界与下一步

旧代码和历史实验未改动；没有生成新QA/Gold/选项，没有运行新的QA Reviewer、学校API、本地Qwen或训练，没有修改扩充流程。分析协作者仅做只读证据复核，不是新一轮题目盲审。已有素材审核可复用，但正式使用资格仍为pending；这份报告不构成批准。

等待确认的是**是否实施本方案并完成回归**。正式素材授权仍须在后续真实生成前落实；若没有同范围安全L3/L4，应报告NO_SAFE并停止，不扩大工程或重复改题求PASS。
