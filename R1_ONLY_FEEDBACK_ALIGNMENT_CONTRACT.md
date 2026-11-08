# R1-only 起点与第一轮训练数据生成流程对齐

本文件记录 2026-09-13 用户确认的工作路线。这里的“第一轮基线”指训练前实际生产训练数据的流程，不是早期 direct-MCQ 原型，也不是后来名为 aligned 的 R03 实验。

## 起点

- 共同起点是原始 Parent + 已冻结 R01，以及对应的 R1-only Memory。
- 原有 R02 / R03 扩充不进入新实验的初始 Memory 或初始累计对话。新实验从这个起点生成的后继轮次应单独编号、保存和追加。
- 70 个训练 source roots 由 `h4_round1_cumulative_continuation_fresh_mining_v1_live_47/round1_root_manifest.jsonl` 定位；使用其中指向的原始 R01 source 文件，不使用该 cumulative run 后来生成的 R02。
- `h4_round1_train_r01_only_rebuild_v1_live_1` 是后续 60-root 重建：历史报告 59 PASS、1 quarantine。它不包含先前 10-root pilot，也不能据此声称那 6 个比较 root 已有可复用的干净远端 Memory。
- 冻结的 R01 后 Memory 快照和历史 Add 回执可用于追溯。当时成功不证明远端 namespace 现在仍为 R1-only。已写入 R02 的 namespace 不作为新的 R1-only 起点。

## 允许增加的反馈

保留并实现固定反馈模板、程序编译的反馈输入、Planner / Intent Miner / QA prompt 的反馈指导以及来源追踪。模板与动态观测分开保存；模型不能修改 policy、虚构缺失观测或从正确率推导能力诊断。

新增反馈段不改变原 prompt 的事实来源、输出 schema、题型配额、调度或审核方式。历史 QA/Gold/行为反馈仅是指导；事实和最终 Gold 仍由当前分支冻结 transcript 支持。不能把旧 R02/R03 分支的事件当成新分支的已发生事实。

第一轮 QA 原有三个 intent 和原有 intent enum 继续使用。先前 SUCCESSOR_FRONTIER / COVERAGE_GAP / FRESH_R03 的强制槽位、NO_SAFE 跳过调度、强制 R03 引用及额外去重硬门，不属于这个严格对照入口。可借鉴它们的内容指导，不移植其执行合同。

## 要保持的基线行为

| 阶段 | 对齐要求 |
| --- | --- |
| Memory / Planner | 复用真实第一轮 `--memory-only-fast --event-bucket-only` 入口及其配置；Memory-only Fast 不等于删除原扩充 Gate。候选数、场景数、输入范围、预算不能借反馈之名改变。 |
| Realizer | 3 场景一次生成、默认每场 14 turns、硬范围 9–48；人物、隔离、adherence 和追加规则保持。输入按真实轮次对齐：首次 R01 扩充带 Parent 原文；从 R1-only 起点生成 R02 时使用 text-free 连续性/风格统计，不带 Parent/R01 历史对白。两组采用相同输入边界。 |
| QA | 复用 `h4_batch30_memory_event_k3_probe_v1/qa_generation_only.py` → `h4_qa_multiroot_k3_t08_probe_v1/generation.py::generate_root_qa`。 |
| QA 主批次 | 每个有效 intent 各调用 contract_guided（2 单选 + 1 多选 + 1 排序）和 free_exploration（4 单选 + 2 多选 + 2 排序）。 |
| QA repair | 对每个 intent × source 的混合 batch，结构有效题少于最低 2 单选 + 1 多选 + 1 排序时，整 batch repair 一次；free 主批次目标虽为 4/2/2，接受最低也为 2/1/1。不是按 root 缺某题型才定向补题。 |
| QA 审核 | 使用原结构 validator 和 permutation 检查；不新增 QA semantic auditor / 语义准入门。扩充阶段原有审核另算，不一并删除。 |
| 模型参数 | QA 使用实际第一轮 Qwen/Qwen3-14B、temperature 0.1、thinking false、Intent 4096 / Writer 8192 tokens。其他阶段以各自真实 invocation 为准。 |
| Solver | 保留同一模型、候选池、上下文模式、seed / permutation / 计分规则；候选先冻结再评估。反馈导致候选改变是实验结果，不是理由去改评估条件。 |

用户此前明确要求“有连续关系的场景一起生成，其他场景分批”。这个功能保留为有名字的共同配置改进；启用时反馈组和无反馈组必须一起采用。它与历史第一轮的一次生成行为不同，报告中必须列出。不能把 development_mode=continuation 自动等同于场景间依赖，也不能宣称现有 continuation_first 已完整实现此要求。

同输入、同程序规则不保证相同成功 intent 数、repair 次数或 QA 数；这些会随模型输出变化。对齐的是调度规则和预算，而不是强行凑出相同实际调用数。

## 目前落实范围

- 已用历史产物核对第一轮 QA builders：102 个 Intent Miner prompt、606 个 primary QA prompt 均可逐字复现。详见 `restart-n5-runs/h4_r01_start_feedback_alignment_qa_audit_v1.md`。
- QA 原入口新增可选 `--feedback-context-manifest`；不传该参数保持原路由。传入时仅通过既有 `feedback_context` 接口追加冻结文本，并记录、验证各 root 输入哈希。
- 反馈模板位于同目录模块的 `feedback_guidance_v1.md`。旧 R03 模板和结果保留为历史版本，不回写为“已对齐”。
- 起点与 QA 接口确认不代表 Planner / Realizer 所有入口已经完成对齐，也不代表已开始新的远端 Memory 写入或模型生成。

本文件取代此前 `h4_round1_r03_route_alignment_audit_v3.md` 中“只改反馈已完成”、repair 条件以及 continuation_first 完整性的结论。历史报告保留，以便追溯。

2026-09-14 输入边界纠正：本次 R02 的 aligned profiles 曾错误强制 `full_parent`，覆盖后继轮次的 text-free 设置，实际产物出现逐句复制历史对白。已将有反馈/无反馈两个 profile 恢复为 `text_free`；历史调用及失败结果保留，不能追认为修复后产物。依据为历史 `h4_round1_cumulative_continuation_fresh_mining_v1_live_25` 的 R02 Realizer prompt / manifest；逐句比对记录见 `restart-n5-runs/h4_round1_r03_r1_aligned_realizer_copy_audit.json`。此次修复不调用模型、不写入 R02 Memory。
