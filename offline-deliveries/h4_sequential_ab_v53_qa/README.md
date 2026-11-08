# v53 答案分布与显示修复

current arm：六选项须有 2–4 个 supports，其余显式 contradicts；沿用最多一次机械修复。排序固定一句题干与可选精确场景范围，禁止顺序摘要。完整上下文、冻结上一轮反馈、旧重复策略及 Solver 18/12 次评估机制保持。

原始候选记录保留 Gold 和证据，用于审计。DISPLAY_CANDIDATES.jsonl 与 SELECTED_QUESTIONS.md 输出 Gold-blind 随机摆放后的题面与映射 Gold，单批与批次汇总均导出并可重放验证。显示结果仍为 provisional，不含 runnable training messages，正式训练之前还须做内容准入与训练输入构建。

82 项离线测试通过；新规则套用旧六批上下文的12个请求均通过容量检查，最高38,316/40,960。没有进行真实出题、Solver或训练。本实现不能证明未来反例在语义上正确或产率提高。新root/修复请求仍需实测容量。

旧题审计与43题呈现样例在 ../h4_answer_distribution_v53。旧两批冻结结果均未修改；旧全选题仍保持真实Gold，不作为新规范合格题。
