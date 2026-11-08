# v111 固定执行协议

本批沿用已冻结的 v105 CONTENT_CONTRACT 和当前 controller，不改变语义门槛，不针对结果救题。此次是下一批 12 个训练侧 root 的受控生产，不是 untouched test。

## 作者与审核角色

- 事件匹配及正文：group1 /root/scale_group1，group2 /root/scale_group2，group3 /root/scale_group3。
- 完整来源审核和 READY 后 QA 作者：group1 /root/scale_group2，group2 /root/scale_group3，group3 /root/scale_group1。
- 资产与最终公开题来源审核：group1 /root/scale_group3，group2 /root/scale_group1，group3 /root/scale_group2。
- 语义审核者与该题作者不同；公开审核不读取该 root 的 QA_DRAFT、私有设计、PRIVATE_GOLD、私有锚点或当前 Solver 分数。角色均继承同一模型和共享项目背景，不能声称完全盲审、外部独立认证或异模型一致性。

## 固定阶段

1. bootstrap 固定 root、隔离检查及互斥事件分区；匹配者完整阅读基线和事件，保存选源，不换 root。
2. initialize 固定 source 预留及输入；只使用投影后的 TRAINING_FEEDBACK.json，不向作者开放原始混合反馈。作者须记录匹配反馈是否存在。
3. 一次新增 12–16 turns，旧史前缀不改；记录真实开始与结束时间及输入哈希，HOLD 和缺槽不重写。
4. 非正文作者完整来源审核八项；BLOCKING 则 HOLD，不再出题。
5. READY 来源先保存两槽各六个不同 turn 的排序锚点计划，再写一次六事实真假资产与排序题。
6. 第三角色仅读完整来源和匿名资产，保存源判断；控制器据此按固定规则组合 Gold2/3/4/5 及 Gold6 对照。
7. 第三角色完整读最终公开题及来源，保存源判断后，控制器才从原设计构造私有 Gold 并交叉核对。失败、歧义和未知均原样排除，不能改 Gold 救题。
8. 记录完整分母、去重、输入哈希、计时和统一stage ledger后封存内容。之后复用 v15 full-only 两模型配对评估：多选18调用/题，排序12调用/题，每模型至多1368，不自动重试。
9. 由已有 M_clean_policy 的完整有效表现按 0<mean_partial<1 筛选候选；Gold6不进入新训练候选。只绑定原 replay，不混合、不训练。
10. 候选按预先固定 namespace=v111_training_presentation_v1 用 SHA256 派生 Fisher–Yates 重排展示并同步答案，一题一条；仅一次 seed，不搜索种子，不按分布拒绝排列，不改候选或正确内容。另存新视图并独立只读核对。

## 可追溯性与边界

- 每阶段角色、来源/公开文件哈希、完整有序turn_refs、private_answers_read_before_save、真实时间必须记录；receipt 保存后不改。机械阶段使用PROGRAMMATIC，语义阶段使用CURRENT_CODEX_SESSION。
- 生成模型精确ID、采样参数、usage和费用未暴露的字段写null，不猜测。并行组共享时间不可相加当作总wall。
- 使用既有训练侧历史反馈；本批不新增validation/test反馈或事后闭环。当前训练侧12root尚未产生的Solver分数不用于选源、扩充、语义审核或救题。
- 内容及模型评估和训练导出分别封存；所有历史封存目录保持只读。此任务不启动新训练。
