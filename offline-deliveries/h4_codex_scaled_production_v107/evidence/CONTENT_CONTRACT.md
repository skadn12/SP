# v105 受控 Codex 扩充与联合出题内容合同

适用本轮主线程配置的 root。沿用 v104 的 QA_POLICY、作者及两层来源审核定义；本合同把已发现的错误替代、单事件和施事问题落实为执行顺序，不改历史规则或历史产物。继承依据：`h4_batch_codex_benchmark_v104/AUTHOR_INSTRUCTIONS.md` 与其 `evidence/QA_POLICY.md`、`REVIEW_AND_QA_INSTRUCTIONS.md`、`QUESTION_REVIEW_INSTRUCTIONS.md`。

## 1. 正文首稿与来源准入

1. 作者先记 UTC 开始时间，完整读本 root 的冻结 baseline、指定事件及主线程指定历史反馈。保留旧 turns 逐字前缀；只写一次 suffix，不换源，不为过审改稿。
2. 新增自然英文 12–16 turns，单一场景 `R02:C:S01`；仅用主线程许可的既有人物。时间、地点、身份、报告、计划和完成状态遵从完整旧史，不从新素材回忆倒灌旧史。
3. `EXPANSION_DRAFT.json`：`root_id/source_id/event_mapping/state_notes/turns`。每 turn 恰含 `turn_ref/session_ref/scene_ref/speaker/text/origin`；turn_ref 从 `R02:C:S01:T001` 连续，session_ref 与 scene_ref 均为 `R02:C:S01`，origin=`codex_session_generated`。state_notes 使用 `turn_ref/quote/constraint`，quote 为真实连续原文。
4. 保存作者 receipt：真实起止时间、读过文件、全文读取声明、作者角色、attempt=1、反馈使用及限制。机械物化只验证结构／前缀，不代表语义 READY。
5. 非该 root 正文作者完整读冻结累计来源和事件，先保存 `SOURCE_REVIEW.json`。逐旧 session／scene 与新 scene 检查八项：time、location、actor_identity、plan_vs_completion、cross_scene_continuity、semantic_repetition、naturalness、event_fidelity。
6. SOURCE_REVIEW 必含 `root_id/source_sha256/reviewer/full_source_read/turn_refs_read/checks/findings/decision/started_at_utc/finished_at_utc/limitations`。turn_refs_read 为完整有序列表；每 check 有 status/reason；finding 有 severity、turn_refs、exact_quotes、reason。实质矛盾 BLOCKING → HOLD，停止该 root QA。WARNING 保留，含糊事实不得据此写成精确答案。

## 2. 每 root 目标与不可强凑

仅 READY 来源进入 QA。每 root 目标是 Gold2／3／4／5 各一题、Gold6 对照一题、排序两个槽位；全部为一次首稿，缺额或失败原样记录。六真父题需要六个互不重复的明确真事实；不足就保留失败，不补零、不编造。目标数量不构成通过依据，同 family 题目不当独立 root 样本。

## 3. 多选：先审错误版本，再组合

1. 作者保存六事实资产：`fact_id/true_text/false_text/evidence/mutation`。evidence 为 `turn_ref/quote` 列表；mutation 写明唯一变更属性及两值。false_text 可为 null。真句整句成立；假句必须在同人物、事件、时间、条件、模态及状态下被明确排除，不能靠未提及、常识或擅加 only/never。
2. 给非该题作者的资产审核者只提供完整来源、明确 scene 范围及匿名公开陈述：`statement_id/proposition`；不提供 true/false 标签、预期答案数或私有映射。审核者逐项保存以下结构后，才允许组合：
   `{statement_id, proposition, relation: supports|contradicts|unknown, scope_valid, fact_distinct, reason, citations:[{turn_ref,speaker,quote}]}`。
3. 六真版本均须 supports；每个被采用的错误版本须经独立来源判断为 contradicts。unknown 只记录缺额，不进入真假已知组合；不能临时改词、换证据或改答案救题。吃吐司不排除吃麦片；开启安防时紧张不排除关闭时也紧张。反义词替换本身不证明互斥。
4. 每子题六选项、六个不同事实各选一个版本。Gold2／3／4／5 分别需 4／3／2／1 个已通过错误版本；不够就缺该槽。选择与标签置换只用预先固定的装配规则，不用当前 Solver 分数。
5. 最终候选先保存公开文件，不含 Gold、事实映射、证据或审核结果。公开多选字段恰为：
   `candidate_id/root_id/question_type=multi_select/question/options(A–F)/source_path/source_sha256`。
6. 非该题作者先读完整来源和最终公开整题，逐项保存来源判断；proposition 必须等于公开选项原句，relation 仍是 supports／contradicts／unknown，并检查 stem_valid、scope_valid、fact_distinct 与重复／纠缠选项。不得按预期 Gold 数填判断。
7. 该最终来源判断保存后，才从未改的原装配设计构造独立文件 `PRIVATE_GOLD.json`：`candidate_id/role/gold/evidence`，并绑定 public 与来源判断哈希。Gold 必须与全部 supports 标签恰相等；unknown、判断冲突或证据问题则排除整题，不以审核结果反改原设计答案。

## 4. 排序：固定锚点、单事件、人物状态保持

1. 写题前单独保存 `ORDERING_ANCHOR_PLAN.json`：`root_id/source_sha256/slots`；每 slot 有 slot_id=01／02、六 anchors；每 anchor 有 `anchor_id=a1..a6/turn_ref/speaker/full_turn_quote`。每槽同一新 scene 的六个不同 turn；完整 quote 与 speaker 逐字匹配。两槽可以重叠，须披露相关性及重叠数。
2. 每 item 只能表达自身固定完整 turn 内一个事件或言语行为，不把填杯和喝水合并一项，不跨 turn 拼接行动。保留 asks／offers／reports／proposes、否定、推测、未来、条件与完成状态。
3. 结果／状态不自动推出个人行动：物品“已洗好”不能在施事未固定时写成某人“亲自洗过”；某物在某人旁边不自动变成在长椅旁。完整来源仍不能明确支持人物或行动状态时，拒绝该项／槽，不能移动锚点或改 Gold。
4. 作者 items 字段：`item_id=I1..I6/anchor_id/text/support_quote`；每槽恰覆盖 a1..a6，展示顺序打乱。support_quote 是自身 turn 的非空逐字连续子串。所有公开陈述用名字或明确角色，禁止独立词 he/she/him/her/his/hers/himself/herself；原文引文不改。
5. 公开排序字段恰为 `candidate_id/root_id/question_type=ordering/chronology_kind=conversation_order/question/items(I1–I6)/source_path/source_sha256`。题干固定：
   `Order the statements, using left-to-right clause order within a turn, as they APPEAR IN THE SUPPLIED DIALOGUE within the R02:C:S01 scene.`
   目标是话语出现顺序，不是客观事件发生顺序；不得泄漏 turn 号、锚点、答案或证据。
6. 审核者先读公开题与完整来源，保存每项 `proposition/supported/actor_ok/modality_ok/state_ok/neutral_stem_ok/order_anchor/support_refs/alternative_matching_anchors/reason`，字段沿用 v104。order_anchor 为 `{turn_ref,quote}`，quote 在该 turn 恰出现一次。
7. 必须穷尽公开 scene 内每项全部合理语义匹配，包括同 turn 的其他匹配；全文用于解释人物指代。所有匹配组合都产生同一严格无重叠顺序，才能 unique_order=true；不能只看字面唯一或生成者私有锚点。题级保存 `coherent/unique_order/neutral_stem_ok/derived_order/alternative_orders/reason`。独立锚点还须落在写题前固定 turn 内。
8. 来源判断先保存，随后才读取／构造原设计私 Gold 交叉核对；排列冲突、单事件违规或支持不足均排除，不移动题句、不换锚点、不改答案救题。

## 5. 记录边界

全文审核与最终题面判断均绑定实际 source/public SHA-256，列全 turn_refs_read；逐字引用须属于所报 turn、speaker 一致。公开与私有文件分开；保留首稿、失败、原答案、判断、实际计时和任何最终裁决。脚本只能机械核对字段、哈希、引用和已记录锚点组合；**人工／模型语义判断不能写成程序自动认证**。

receipt 如实披露作者／审阅角色、是否看私有答案、六真设计意图及 Solver 分数。角色交叉仍是同一继承 Codex 模型和共享上下文，不是不同模型认证。未测 Solver 不报难度或正确率，未训练不报训练收益。本合同不新增训练、API 或自动生产授权。
