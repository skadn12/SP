# 本轮统一出题合同（无诊断信息）

这是固定Solver无训练的同源生成扩样复现。每个作者读取分配的完整来源（不是仅追加部分），每root正好8个槽：MC2 / MC3 / MC4 / MC5 / MC6 / ORDER1 / ORDER2 / ORDER3。每槽一初稿，内容审核后最多一次修订；不得看Solver结果、原题、另一个作者的产物后改题或增加题。来源为两人累积对话，R03追加6场；日期跨场不默认同周。传入source_status及warnings必须保持。

## 数量、质量和形式
- MC2：multi_select，MC_QUANTITATIVE_RELATION，6项，恰2正确。
- MC3：multi_select，MC_CONSTRAINT_CONJUNCTION，6项，恰3正确。
- MC4：multi_select，MC_EVIDENCE_SELECTION，6项，恰4正确；可问哪些给定turn片段足以支持指定命题，但题面须清楚判定单位是单turn还是片段组合；引用坐标和内容需逐字准确。
- MC5：multi_select，MC_CAUSAL_SUPPORT，6项，恰5正确；需要明示的因果/目的支持，不用世界知识臆测理由。
- MC6：multi_select，MC_CONSTRAINT_CONJUNCTION，6项全部正确，单列全选控制，不宣称提升了干扰项质量。
- ORDER1：ordering，ORDER_EXPLICIT_TIME，6项；明确按日内钟面值排序（不同日期不当持续时间），项目不露直接时刻值，六值无并列。
- ORDER2：ordering，ORDER_EXPLICIT_PROCEDURE，6项；operation_class=source_defined_dependency，按来源规定的程序而非常识模板。可包含明确未执行但来源规定的末步，题意不能冒充已发生。需确定唯一全序和来源特定依赖。
- ORDER3：ordering，ORDER_DISCOURSE_RECONSTRUCTION，6项；operation_class=discourse_reconstruction，按实际话语贡献出现位置，精确定位完整贡献，不能混同事件现实发生顺序。

问题及选项用英文，内部审核字段可中文。标签A至F，Gold集合/序列私有；公开题面不可提示Gold数量。题干应有可比较的长度和复杂度（约30–90词），单选项约10–35词，六项长度/细节尽量匹配。不得把事实同义改写凑成多个正确项。错误项改变一个可核验关系，其余详情不变，避免荒谬数字/明显否定/长度暗示。所有正确项和错误项均提供足够的逐字来源证据和完整解释；“未提及”不能偷换为明确为假，按题面supported标准裁决。

每题至少一个必要前提或必要排序边依赖R03新turn，注明`r03_necessity`具体去掉新证据为何不能推出完整Gold；不能仅说引用了R03。可跨旧新来源组合，不能依赖warn中冲突和未口述字段。不得引入新来源事实。整题完全重复才排除，不扩大为同事实家族排除。

排序项目不用first/later/finally/最初/后来/总结等编辑排名词；真正必要的提问/响应/依赖关系可保留。项目的canonical写入顺序可任意；最终显示由程序随机打乱两次且不能等于Gold。不要作者手工靠ABCF排列制造难度。

## 先计划再出题
先创建`TASK_PLANS.json`：每root/槽的操作、来源关系、必要R03前提、预定质量验收；冻结后写题。记录完整来源读取回执，包括hash和完整turn refs。不能自检冒充独立审核。

## 候选格式
输出`CANDIDATES_DRAFT.json`，JSON list。每题：
```
candidate_id, arm, slot_id, root_id, author_id, question_type, task_operation,
question, options或items(恰A..F), gold(list), source_binding(path/sha256/bytes),
r03_necessity, item_analysis
```
MC的每个item_analysis：
```
evidence:[{turn_ref,speaker,quote}],should_select:boolean,rationale,changed_relation:null或描述
```
排序item_analysis：evidence+rationale；再加ordering_analysis：
```
operation_class:clock_value|source_defined_dependency|discourse_reconstruction,
source_dependence_rationale,
edges:[{before,after,evidence:[{turn_ref,speaker,quote}],rationale}],
mention_order_boundary,editorial_cue_policy,common_sense_boundary
```
五条相邻边要定义Gold唯一全序，但每边也要真实来源支持；不能先造Gold再编边。引用quote应原始text完整或逐字子串，不可拼接改写。

用项目已有`scripts/current_method/production_question_gate.py`的audit_candidate做机械预检，修正工程引用错误不等于独立语义审核。独立内容审核前尚可修正JSON语法/引用录入，任何实质命题修改须保留工程日志。结果保留全部槽，无法安全产出则记录SLOT_FAILURES.json，不能默默换预算。只写分配的作者目录。
