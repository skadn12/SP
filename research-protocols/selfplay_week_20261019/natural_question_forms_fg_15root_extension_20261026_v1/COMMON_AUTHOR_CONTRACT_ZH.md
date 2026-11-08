# 本批作者合同：自然问法 F/G 多选题

## 固定设计
每个 root、每个 arm 恰有五个槽：MC2、MC3、MC4、MC5、MC6。每槽 6 项 A–F，Gold 数分别为 2、3、4、5、6。操作分别是数量关系、约束合取、证据选择、因果/目的支持、全真控制。MC6 单独报告，不纳入主要 J 汇总。不得生成排序题或新增操作。题目和选项用英文；审核说明可用中文。

F 接收原公开反馈卡与原 questioner feedback，G 不接收任何反馈卡、F 计划、题目、Gold 或成绩。两臂读取同一完整来源及该来源警告。作者不得读取本批 Solver 输出、对侧作者文件或旧题/旧计划。

## 出题要求
先为每个槽冻结任务计划，再写初稿。可根据当前完整来源重新选择实例，不强制复制历史题；自然、具体地提出所需内容，选项直接回答问题。可问计算、可行安排、行动、原因、证据或满足条件的实例，不需要统一写成“哪些陈述受支持”，也不必包装成台账。形式变化不能引入额外能力、新事实或额外难度来源。

严格使用该槽所规定的操作、6 个选项和 Gold 数量。每项须有准确逐字来源证据、选择判定和理由；数值须独立计算；因果/目的须有明示支持；证据槽须明确证据单位。不得因目标配额提示 Gold，不得泄露 F 卡或具体错误机制，不得把全部解题事实直接写进题面。source warnings 的边界逐字保留；不能将未说明、含糊、提议或计划提升为事实或完成状态。

候选字段至少包括：candidate_id, arm, slot_id, root_id, author_id, question_type, task_operation, question, options, gold, source_binding, source_status, source_warnings, r03_necessity, item_analysis。MC 的 item_analysis 对 A–F 各提供 evidence[{turn_ref,speaker,quote}], should_select, rationale。r03_necessity 须说明删去哪些 R03 证据会使完整答案集无法推出；不能只说引用了 R03。

## 审查、修订和冻结
先盲推 Gold、再对作者答案核对。至少由非作者独立审核每题题意、Gold、逐项证据、操作对齐、自然度、完整来源必要性及反馈目标（仅 F）。初稿问题最多一次内容修订；保持完整日志。无法可靠解决的槽按 HOLD/FAIL 保留，不换题、不按 Solver 结果返修。通过后冻结最终题目、审核绑定、显示排列、请求与 prompt 清单，再做精确 tokenizer/context 预检；完整 prompt 加 6,144 输出预留不得超过 16,384 tokens，禁止截断来源。
