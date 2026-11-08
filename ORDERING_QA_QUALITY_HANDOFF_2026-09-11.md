# 排序题质量问题交接

更新时间：2026-09-11 UTC。本文是离线交接，不是新一轮审核通过报告。

仓库：`/root/autodl-tmp/gxy-selfplay/current_method_github`

## 1. 接手前先读的结论与约束

1. 已发现排序题存在 gold 顺序错误、选项不受对话支持、说话人错误、计划被写成已完成、顺序不唯一、证据引用不匹配等问题。不能把 Solver 与 gold 不一致一律解释为模型能力不足。
2. 实际进入 Clean/Inclusive 的排序题分别为 159 条，并集 191 个 candidate ID。旧审核表分别标记 77/80 条问题，但已经发现旧表漏判，因此 48.43%/50.31% 只能称为“旧审核表标记比例”，不是已确认的真实错误率。
3. Dev/Test 排序题也做过审核及修正 gold 复评。修正后的 Full 准确率略高于 Base，但提升很小，且 Dev context gain 仍下降。不能声称“所有指标都没有下降”或“错数据没有影响”。
4. 用户希望保留现有已训练模型，从第二轮治理 QA 数据；不希望为了此事重训第一轮、重新生成既有扩充或改动已加入 Memorax 的内容。
5. 用户要求全自动 self-play。不能把人工审核、人工仲裁作为运行闭环的必经步骤。也不能把另一个模型审核通过、多模型一致或模型生成的证据链当作独立真值。
6. 此次只交接。未启动 Solver、QA Writer、外部 API、Memorax、训练或额外 semantic auditor 服务，未修改任何旧结果。本次对原对话的核查仍是 Codex 的语义判断，不是独立人工真值。

命名注意：这里讨论的第一轮已训练模型，其工件路径使用 `h4_round0_b0_*`。不要与历史 `h4_round1_paired_blind_solver_*` 中 Control 276 / Feedback 254 的比较混为一谈。本文不统计后者，也不按候选位置强行配对。

## 2. 实际训练数据的规模

来源是实际训练使用的 `h4_round0_b0_matched_training_v1_live_2`，不是整个候选池，也不是 Dev/Test manifest。

| 数据组成 | Clean | Inclusive |
| --- | ---: | ---: |
| 全部训练记录 | 386 | 386 |
| ordering | 159 | 159 |
| multi_select | 172 | 172 |
| single_choice | 55 | 55 |

排序题按 `candidate_id` 去重：共享 127 条，Clean 独有 32 条，Inclusive 独有 32 条，并集为 `127 + 32 + 32 = 191`。两份训练文件合计 318 条排序记录不等于 318 个独立候选。

本次从 manifest 和实际 train JSONL 重算了上述数量，并核对两份 manifest、两份 train 文件的 SHA256 均与旧审核报告记录一致。身份和文件一致性不等于语义正确性。

### 2.1 旧训练审核表的标记结果，待校准

| 范围 | 排序题数 | invalid_gold | unsupported | ambiguous | 问题合计 | 问题比例 | gold_status=valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean 实际训练 | 159 | 59 | 12 | 6 | 77 | 48.43% | 82 |
| Inclusive 实际训练 | 159 | 60 | 12 | 8 | 80 | 50.31% | 79 |
| 两者候选并集 | 191 | 71 | 16 | 9 | 96 | 50.26% | 95 |

这里“问题合计”是 `invalid_gold + unsupported + ambiguous`，不是单指 gold 排列错误。折算到全部训练记录，旧表标记的问题排序题占 Clean `77/386 = 19.95%`、Inclusive `80/386 = 20.73%`。

不能据此说“一半全部训练数据错误”，也不能说“其余约 80% 全部正确”：其他题型未在这次排序专项中全面复核，排序审核本身也有错判。

| 排序题成员范围 | 候选数 | 旧表标记问题数 |
| --- | ---: | ---: |
| 两策略共享 | 127 | 61 |
| Clean 独有 | 32 | 16 |
| Inclusive 独有 | 32 | 19 |

### 2.2 evidence mismatch 不能与 gold 错误混算

| 范围 | evidence_status=mismatch | 其中 gold_status=valid |
| --- | ---: | ---: |
| Clean | 112 | 52 |
| Inclusive | 114 | 49 |
| 候选并集 | 138 | 60 |

这些是旧表的两个交叉维度，不能直接与“问题合计”相加。引用错误可能与错误 gold 共存，也可能是完整对话支持答案、仅给定引用选错。

旧训练流程不把 evidence refs 作为独立字段提供给训练模型，因此仅引用错不自动等于错误监督目标。但审核标为 `valid + mismatch` 也不自动可靠，见第 5 节漏判。

此外，manifest 中的 `EVIDENCE_CLEAN` 是旧筛选口径，不是完整题意和 gold 的正确性证书。Clean 的 159 条排序题全部带此标记，旧复核仍标出了 77 条问题。

训练审核入口：[旧汇总报告][train-report]、[逐题审核表][train-review]、[Clean 问题表][clean-errors]、[Inclusive 问题表][inclusive-errors]。

## 3. Dev/Test：审核过哪些，哪些进入了复评

以下是既有审核表和复评工件中的记录，不代表本次重新逐题确认了 Dev/Test 的全部语义判定。

| 范围 | 原排序题数 | 原 gold 判可用 | invalid_gold 修正 | ambiguous | unsupported | 复评保留 | 排除 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dev | 119 | 51 | 62 | 5 | 1 | 113 | 6 |
| Test | 126 | 63 | 52 | 11 | 0 | 115 | 11 |
| 合计 | 245 | 114 | 114 | 16 | 1 | 228 | 17 |

具体处理口径：

- `gold_status=valid`：保留原 gold。
- `gold_status=invalid_gold`：使用审核表给出的 `corrected_order`。
- `ambiguous / unsupported`：排除，不强造唯一答案。
- `evidence_mismatch` 但 gold 判可用：仍进入评估，Dev 29 条、Test 34 条，合计 63 条。
- Dev/Test 共有 169 条被标记引用不匹配，这包含与其他错误类别重叠的题，不是额外排除 169 条。

旧表中“gold 和引用都通过”的 `primary_status=valid` 为 Dev 22、Test 29，共 51 条；不能把这个 51 当作“只有 51 条答案可用”。原 gold 判可用的数量是 114，修正后参与评估的是 228。

包括未专项重审的其他题型，Dev 总候选从 485 保留 479，Test 从 507 保留 496。非排序题沿用原 gold。没有证据可以因此声称“整个 Dev/Test 已经无错误”。

来源：[Dev/Test 审核报告][eval-audit-report]、[Dev/Test 逐题表][eval-review]、[Dev 离线复评报告][dev-rescore]、[Test 评估报告][test-report]。

## 4. 模型结果：Full 略升，不等于所有指标都改善

下面是报告中的 ordering 汇总，使用当前审核修正后的 gold 和上述保留子集。Full/No 为百分比，Gain 和差值为百分点。`context gain = Full accuracy - No-context accuracy`。

| 集合 | 模型 | Full | No | Gain | Full 相对 Base | Gain 相对 Base |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Dev，113 题 | Base / M0 | 39.97 | 15.93 | 24.04 | 0 | 0 |
| Dev，113 题 | Clean | 40.71 | 16.89 | 23.82 | +0.74 | -0.22 |
| Dev，113 题 | Inclusive | 40.49 | 17.92 | 22.57 | +0.52 | -1.47 |
| Test，115 题 | Base / M0 | 38.04 | 16.74 | 21.30 | 0 | 0 |
| Test，115 题 | Clean | 38.33 | 15.22 | 23.12 | +0.29 | +1.81 |
| Test，115 题 | Inclusive | 38.19 | 16.30 | 21.88 | +0.14 | +0.58 |

ordering 每题、每种 context 有 12 条计分请求，所以每模型 Dev Full 分母为 1356，Test 为 1380。这不是 1356/1380 道独立问题，也不是 root-equal 结果。

- Dev Full 正确请求：Base 542，Clean 552，Inclusive 549。
- Test Full 正确请求：Base 525，Clean 529，Inclusive 527。
- 当前 Full 增益对应净增加 Dev 10/7 条、Test 4/2 条正确请求，不能写成多做对相同数量的独立题。
- 本次未计算显著性或按 root 聚类的置信区间，不能把小差异称为稳定提升。

### 4.1 gold 改动能改变“退化”的方向

在相同的 113 道 Dev 排序题和相同模型输出上，仅改变计分 gold：

| 计分 gold | Base Full | Clean Full | Inclusive Full | Clean - Base | Inclusive - Base |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原 gold，同一保留子集 | 41.59% | 40.71% | 40.78% | -0.88 pp | -0.81 pp |
| 审核修正 gold，同一保留子集 | 39.97% | 40.71% | 40.49% | +0.74 pp | +0.52 pp |

数据来自 Dev 报告的 `original_gold_same_subset.by_question_type.ordering` 与 `by_question_type.ordering`。这是计分变化，不是模型变强；此处符号翻转并不证明审核修正后的 gold 已绝对正确。

过程也必须区分：Dev `live_7` 是复用既有输出的离线重计分；历史 Test `live_2` 为 Clean/Inclusive 实际新增过 25836 次 Solver 请求，Base 复用。不能把整段历史 Test 工作说成零推理离线复评。本次交接没有新增推理。

### 4.2 如何解释“训练有错，结果没降”

可确认的是：当前计分口径下 Full 没有下降，旧训练审核发现较多问题，但真实问题率仍待校准。

不能确认的是：错数据没有害处、模型已自动纠错、能力已稳定提升，或者它们导致了某个具体退化。有效监督和错误监督可能相互抵消，损害也可能表现为“本来可以提高更多”，这些目前只是解释假设。

历史训练使用 Qwen3-4B、LoRA rank 16、学习率 0.0001、每策略 386 条、2 epochs、193 optimizer steps，见 [训练配方][recipe]。没有做同配置“原始训练数据 vs 修正训练数据”的重训对照。用户当前不希望为此重训第一轮，不应擅自启动该对照。

## 5. 可以直接打开原对话核查的样例

以下 ID 都是实际 train 候选。本次只是有针对性的核查，不能把少数样例的发现率外推为总体错误率。旧表未被回写。

### 5.1 94289：计划被写成已完成，审核仍放行

- Candidate：`cc-train-episode-94289:I003-F-Q08`，同时进入 Clean/Inclusive。
- I2：`She started journaling to process her emotions.`
- 对应 `R01:S03:T007` 原文：`I'm going to start making a list of things I can do when I feel overwhelmed. Cleaning is the first one, but I'm thinking about adding exercise and maybe even journaling.`
- 后文讨论写日记的好处，没有落实“已经开始”的事实。不能用调整顺序补救这个事件状态错误。
- 旧表：`gold_status=valid`，gold 为 `I1,I3,I4,I2`；备注同时写 `the journal is only proposed`，与放行判定冲突。
- 本次发现：该选项至少存在 unsupported 问题，不能仅归为引用错。
- [题目与完整对话文本][case-94289]；[原始对话 JSON][dialogue-94289]。候选池在原始对话同目录的 `candidate_pool.json`。

### 5.2 63323：说话人反了，两条训练候选都漏判

- Candidates：`cc-train-episode-63323:I001-F-Q07`、`cc-train-episode-63323:I002-F-Q07`，均进入两策略。
- I2 分别写 `James asks Emily if she has seen it.` / `James asks if she has seen it.`
- `P01:T002` 是 James 问 `What movie did you get?`。
- `P01:T003` 是 Emily 说 `I got "The Hunger Games." Have you seen it?`，问是否看过的人不是 James。
- 全对话中 James 还有问近期看了什么电影、今晚是否要看的发言，但不能等价替代该选项。
- 旧表将两条都标成 `valid + mismatch`。这是说话人/事件支持问题，不是单纯移动 evidence ref 就解决。
- [题目与完整对话文本][case-63323]；[原始对话 JSON][dialogue-63323]。

额外注意：同一原对话 `P01:T004` 中 James 表示以前看过该电影，`P04:T004` 又表示没看过。说明 source dialogue 本身也可能不一致；QA 相对于 source 的正确性与 source 自身一致性是两个问题。没有统计这种源对话问题的总体比例，也不据此重做扩充。

### 5.3 103866：事件有支持，但 gold 颠倒了一对顺序

- Candidate：`cc-train-episode-103866:I001-F-Q08`，同时进入 Clean/Inclusive。
- I2：James 问最近做了什么新菜，对应 `P05:T008`。
- I1：Emily 说做了 vegan lasagna，对应 `P05:T009`。
- I4：James 提议本周末露营，对应 `R01:S02:T001`。
- I3：Emily 说会带露营装备，对应 `R01:S02:T004`。
- 原 gold：`I2,I1,I3,I4`；对这些发言顺序，原文支持 `I2,I1,I4,I3`。
- 旧表对此标为 `invalid_gold`，同时记录前两个给定 evidence refs 不相关。这与前两例“选项不成立但放行”是不同错误层次。
- [题目与完整对话文本][case-103866]；[原始对话 JSON][dialogue-103866]。

### 5.4 75223：不能只按给定 evidence 位置推翻 gold

- Candidate：`cc-train-episode-75223:I002-F-Q06`。
- 后面的扩充段再次提到与 Tyler 交谈，但 `P01:T009` 已有 `Maybe I should try talking to him before trying to get revenge.`，早于 `P02:T010` 的 `Forgiveness is key.`。
- 若只看给定的较晚 evidence，就会误判交谈想法一定晚于宽恕建议。至少不能据这个理由推翻原顺序。
- [题目与完整对话文本][case-75223]。该例用于约束审核方法，不将它宣称为经过全维度认证的完美题。

前三个漏判候选是 94289 一条、63323 两条。不要简单把旧问题并集 `96 + 3` 当作最终准确问题数：其他行仍可能存在漏判或误判，分类规则也尚未完全校准。

## 6. 审核自身的可靠性问题

旧文件名称包含 `manual`，报告状态为 `COMPLETE_FULL_DIALOGUE_MANUAL_REVIEW`，逐行 `review_basis` 为 `full_dialogue_manual`。这些字段只说明旧流程记录了什么，不能据此声称有独立人类专家完整审核并确认正确。

本次重新统计验证了 191 行覆盖与计数、训练文件哈希以及旧裁决表哈希。旧汇总脚本验证了候选身份、状态合法性、gold 排列、manifest 与实际 assistant target 的一致性。这些检查不会发现“计划变已完成”“问话人写反”等语义错误。

因此交接时必须同时保留两个事实：

- 旧数据足以说明存在需要处理的质量风险，有可追溯的错题样例。
- 旧表给出的比例和修正 gold 仍是可被推翻的审核判断，不能把统计闭合误当作语义复核可靠。

不能通过覆写旧表掩盖历史。后续如修订，应产生新版本、逐题记录旧判定/新判定/证据/原因，并保留旧模型输出，必要时对同一批输出离线重计分。

## 7. 自动 self-play 的治理讨论停在哪里

尚未在本次任务中实施或验证第二轮排序质量门控。此前讨论过的方案是候选设计，不是已证明有效的解决办法。

| 检查层 | 能检查什么 | 不能证明什么 |
| --- | --- | --- |
| 程序结构检查 | schema、引用存在、引用原文一致、ID 完整、排列合法、显示与 canonical 映射 | 事件是否成立、语义支持、实际发生顺序 |
| 隐藏 gold 的独立模型检查 | 尝试发现说话人/状态错误、独立解答、给出不确定关系 | gold 绝对正确；与生成器可能共享偏差 |
| 反例检查或多次检查 | 寻找另一合法顺序、冲突原话、重复事件 | 多模型一致不等于独立真值 |
| 基于确定规则生成答案 | 在确实可计算的任务内给出确定 gold | 不能把更简单的任务冒充原来的复杂语义排序 |

“原话在对话中的出现顺序”可以由位置确定，但“事件实际发生顺序”不能直接按 turn ID 排。模型提取的时间关系交给程序拓扑排序，只能验证这组关系是否有唯一解，不能证明模型提取的关系是真的。

用户坚持全自动，因此人工不能作为默认回退。可讨论的不确定题处理是自动隔离/弃用/重新生成，而不是让另一个模型改 gold 后无条件入库。无论如何，保留开放语义出题就必须承认残余标签噪声，不能承诺零错误。

后续反馈至少要分开：

- QA 质量问题：题干、选项、gold、证据或源对话冲突。
- Solver 解题问题：在当前质量检查支持题目可用的前提下，Solver 仍失败。
- 技术失败：解析、超时、输出格式等，不混成语义能力错误。

质量过滤后的样本应称为“自动检查通过”，而非“已确认绝对正确”。还需记录按 root/题型/来源的通过率及难度变化，避免只保留易验证题后将分数提升误解为能力提升。

## 8. 下一位接手者的任务边界

1. 先读本文和关联报告，从实际工件继续，不重新猜项目状态。保留现有修改和历史结果。
2. 优先把第 5 节样例整理为质量检查的回归案例，区分结构问题、语义问题、gold 问题、源对话冲突和审核错判。不要直接重跑全部生成或 Solver。
3. 设计第二轮 QA 准入时，明确哪些检查可确定计算、哪些仅模型估计、不确定如何处理；不能以“加审核模型”作为正确性保证，也不能擅自把复杂事件排序改成引用位置排序。
4. 如需要重新校准旧裁决或实现/试跑门控，单独明确范围和版本。当前请求只授权交接，不授权新外部调用、改训练集、重训或改 Memorax。
5. 未来更新指标必须使用相同候选子集、相同 gold 版本比较模型。Test 已被查看并用于讨论，不应再称为从未接触的盲测集；不要把 Dev/Test 题及答案放入训练或 QA 反馈示例。

本次输出仅本文。没有回写上述三条漏判、没有生成新的“最终错误率”、没有修复训练文件、没有开展第二轮训练。此处记录的治理方向不代表用户已经批准具体实现。

## 9. 工件索引

以下绝对路径可直接在同一机器使用。目录内的旧文件名保留原样，不代表认可其 `manual`/`clean` 命名所暗示的可靠性。

- [实际训练数据目录][train-data]：`m_clean_policy_train.jsonl`、`m_inclusive_policy_train.jsonl`，以及对应的 `*_training_manifest.jsonl`。
- [训练运行目录][train-run]：`clean_adapter/`、`inclusive_adapter/`、`training_recipe.json`、`final_formal_training_report.json`。
- [训练审核目录][train-audit-dir]：最终逐题 TSV、两策略问题 TSV、`roots/` 完整对话阅读材料；逐题行内有 `dialogue_path`、`candidate_pool_path`。
- [Dev/Test 原始评估 manifest 目录][eval-data]：`dev_full_eval_manifest.jsonl`、`test_full_eval_manifest.jsonl`，注意是 `live_1`。
- [Dev/Test 审核目录][eval-audit-dir]：`final_manual_review.tsv`、`final_ordering_manual_review_report.json`。
- [Dev 复评目录][dev-rescore-dir]：使用 `live_7`，含逐题 `*_rows.jsonl`、排除列表及汇总；其中只有 Test Base，没有 Test 两个训练模型的最终对比。
- [Test 评估目录][test-dir]：使用 `live_2`，含三模型最终汇总和 Clean/Inclusive 原始输出。
- [审核相关代码][review-code]：`train_review_materials.py`、`train_manual_adjudications.tsv`、`finalize_train_review.py`、`complete_manual_review.py`、`offline_rescore.py`。不要为阅读而直接运行这些入口，部分会写文件；`review.py` 包含模型运行逻辑。

[train-report]: /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_train_ordering_full_manual_review_v1_live_1/final_train_ordering_manual_review_report.json
[train-review]: /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_train_ordering_full_manual_review_v1_live_1/final_train_ordering_manual_review.tsv
[clean-errors]: /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_train_ordering_full_manual_review_v1_live_1/clean_ordering_training_errors.tsv
[inclusive-errors]: /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_train_ordering_full_manual_review_v1_live_1/inclusive_ordering_training_errors.tsv
[eval-audit-report]: /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_dev_test_ordering_review_v1_live_1/final_ordering_manual_review_report.json
[eval-review]: /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_dev_test_ordering_review_v1_live_1/final_manual_review.tsv
[dev-rescore]: /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_ordering_corrected_offline_rescore_v1_live_7/final_offline_rescore_report.json
[test-report]: /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_common_test_eval_v1_live_2/final_test_evaluation_report.json
[recipe]: /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_b0_formal_training_v1_live_2/training_recipe.json
[case-94289]: /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_train_ordering_full_manual_review_v1_live_1/roots/cc-train-episode-94289.txt
[dialogue-94289]: /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_unexpanded122_qa_generation_only_v1/shards/shard-56/roots/cc-train-episode-94289/expanded_dialogue_clean.json
[case-63323]: /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_train_ordering_full_manual_review_v1_live_1/roots/cc-train-episode-63323.txt
[dialogue-63323]: /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_unexpanded122_qa_generation_only_v1/shards/shard-47/roots/cc-train-episode-63323/expanded_dialogue_clean.json
[case-103866]: /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_train_ordering_full_manual_review_v1_live_1/roots/cc-train-episode-103866.txt
[dialogue-103866]: /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_unexpanded122_qa_generation_only_v1/shards/shard-00/roots/cc-train-episode-103866/expanded_dialogue_clean.json
[case-75223]: /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_train_ordering_full_manual_review_v1_live_1/roots/cc-train-episode-75223.txt
[train-data]: /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_b0_matched_training_v1_live_2
[train-run]: /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_b0_formal_training_v1_live_2
[train-audit-dir]: /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_train_ordering_full_manual_review_v1_live_1
[eval-data]: /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_b0_matched_training_v1_live_1
[eval-audit-dir]: /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_dev_test_ordering_review_v1_live_1
[dev-rescore-dir]: /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_ordering_corrected_offline_rescore_v1_live_7
[test-dir]: /root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_common_test_eval_v1_live_2
[review-code]: /root/autodl-tmp/gxy-selfplay/current_method_github/scripts/current_method/restart_n5_spell/h4_ordering_full_review_v1
