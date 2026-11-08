# 下一代有限反馈出题：作者交付规范

本文件仅规定交付结构，不是已读取卡片、已完成计划或已生成题目的回执。必须等主代理明确释放新版卡路径与 `SELECTION_FREEZE.json` 后开始。遵循同目录 `PROTOCOL_FREEZE.json`；主批 32 题位/120 请求与本批最多 4 道新题分列。

## 固定分工与准入边界

- author_a：`cc-train-episode-218519`、`cc-train-episode-222937`，作者 `/root/followup_production_pipeline_audit`。
- author_b：`cc-train-episode-225575`、`cc-train-episode-229686`，作者 `/root/followup_production_pipeline_audit/historical_fixture_checks`。
- 每 root 最多 1 稿；选定卡片与维度以释放的 selection 为准。没有观测目标保留 `NO_OBSERVED_TARGET`，素材不足保留 `MATERIAL_SHORTFALL`。
- 最多一轮独立内容审核后修订，不自审。不新增 Solver 请求、不训练、不扩充对话。
- 新题只能证明反馈被具体使用并获得内容支持；不能据此宣称弱点复现、正式弱点准入或因果训练收益。

## 必须顺序完成的操作

1. 实际完整读取释放卡片、所选维度的真实受审错误响应、实际正例/反例、完整绑定来源，并核对主批题目以避免重复同一具体关系。只在读完后写读取回执。
2. 每 root 冻结一条来源专属计划；保存其文件 hash/创建顺序，随后才写新题初稿。无观测目标或素材不足仍须有一条计划和原因。
3. 生成候选并通过现有机械门禁；六选多选 gold 2–5，排序 4–6 项且来源蕴含唯一全序。沿用四种多选操作/三种排序操作和所有来源可靠性门禁。
4. 将初稿交独立非作者审查。保留初审记录，必要时至多修订一轮并保存逐字段变更。
5. 独立终审完成后才写最终 `CANDIDATES_PRIVATE.json` 别名。不得提前把草稿写成最终准入结果。

## 每组文件

- `AUTHOR_READ_RECEIPT.json`：包含 `author_id`、`card_ids_read`、`full_sources_read: true`、`source_card_binding`、`sources`（完整文件 `binding`、按原顺序的 `turn_refs_read`）、`reviewed_response_ids_read`、`positive_counterexamples_read: [{request_id, display_item, note}]`。正向反例必须同 root、同 selected_dimension，且该项 judgment/citation/reason 均 PASS，不是目标错误 affected 项；可来自 OBSERVED 响应中的正确单项。旧 `positive_counterexample_response_ids_read` 可保留但不作为准入门。`sources[].binding` 必须等于 selection 该 root 的完整 path/hash/bytes。可保留协议/selection绑定。回执必须是实际读取的记录。
- `TASK_PLANS.json`：`plans` 数组，每 root 一条。选中计划含 `candidate_id`、`root_id`、`card_id`、`task_operation`、非空字符串 `source_supported_target_relation`、`predicted_error`、`positive_counterexample`。短缺项写 `status`、`reason`。只针对所选来源，不能把错误回答直接当成真实事实。
- `CANDIDATES_PRIVATE_DRAFT.json`：初稿数组或 `{candidates: [...]}`；保留原稿不覆盖。
- `REVISION_LOG.json`：独立初审后的修订次数、相关 candidate、改动理由/字段与 Gold 是否改变；没有修订时也明确记录 0 轮。
- `CANDIDATES_PRIVATE.json`：独立审核后最终候选数组，与初稿保持同一候选集合；未准入的内容也保留审核状态，不按 Solver 分数删选。

独立审核由主代理安排写入 `review_a/`、`review_b/` 的 `INITIAL_CONTENT_REVIEW_AI.json` 与 `CONTENT_REVIEW_AI.json`；作者不写自己的准入审核。

## 候选的反馈连结

沿用主批 `production_author_candidate_v1` 字段与完整 `source_binding`、逐项确切引文/判断/理由、排序边及来源论证。每题设置 `solver_evaluated: false`、`formal_weakness_admission: false`。

`feedback_target` 沿用 `card_id`、`dimension`、`predicted_error`、`competing_explanation`、`source_specific_realization`、`positive_counterexample`，新增非空 `reviewed_request_ids`（必须来自当前 root 所选卡的真实 OBSERVED 支持响应）与 `previous_candidate_ids` 以追溯旧例。正例/反例必须对应实际受审核内容，按回执中的 request_id + display_item 明确定位。必须同 root、同 selected_dimension，单项判断、引用、理由三项全 PASS，且不在该响应的目标错误 affected 项中。不能以整题 NOT_OBSERVED 代替逐项正确核查，也不能要求整条响应绝对没有其他错误。

新题可复用来源事实，但必须换用主批未使用的具体关系/例子。不能仅改措辞或选项顺序来充数。公开证据选项不要直接给出可脱离来源判定的完整引文；不泄露 gold 数量；排序不加入人为序号、起初/后来/最终等位置捷径。来源不足时不编造、不开新扩充、不借外部知识补全。

## 机械检查和独立内容检查

使用 `production_question_gate.bind_candidates` 绑定 selection 源，`audit_candidate` 检查精确引文和结构；机械通过只证明格式/引用匹配，不能替代语义审核。独立审查须重读相关证据、重建 Gold，检查完整来源、目标错误的具体实现、独立正确反例、与主批不同具体关系、无来源捷径及竞争解释。作者与审核者身份必须分离。

最后由主代理运行同目录 `assemble.py`，保留独立审核与短缺项，汇总实际反馈消费；主批与本批数量始终分列。本文件不触发 assemble 或 Solver。

## 已释放 selection 的正向对照例外

225575 的 QUANTITY_OR_TIME 无判断/引用/理由全部 PASS 的单项，不得制造 PPP。按 SELECTION_FREEZE 的正向策略，允许 judgment+reason PASS 而 citation FAIL 的实际单项；回执必须写 actual_citation_supported: false、comparison_scope: JUDGMENT_AND_REASON_ONLY_WITH_REVIEWER_SOURCE_RECONSTRUCTION，并给 additional_source_evidence: [{turn_ref, speaker, quote}]；独立审核须重构来源证据。其他 root 优先 PPP。此例外不改写原语义审核结论。
