# A3＋B3 恢复验证内容合同 v112

本批为两条已暴露训练root的探索性流程恢复；不训练，不声称外部语义认证或严格后端对照。

## 扩充
- 固定旧P＋R01基线，按A1→A2→A3→B1→B2→B3顺序累计。每场12–16条自然英文发言，保留完整已接纳历史逐字前缀。
- A推进已存在的事件，不能只重复愿望或重新完成旧阶段。B每场使用一个独立永久预留的新数据集事件，形成区别于A旧事物流的新主目的。
- 只用既有speaker。允许建立明确的当前新事实，不能倒灌新素材的职业、亲属、时间和完成状态为既有历史；第三方服务可在朋友对话中明确转述。
- 每场作者先完整读取累计来源；首稿只保存一次，不修稿或换源。非该场作者完整读累计来源和B原生事件，检查time/location/actor_identity/plan_vs_completion/cross_scene_continuity/semantic_repetition/naturalness/event_fidelity，明确READY或HOLD。
- 只有上一场READY且hash绑定无误才进入下一场。HOLD停止该root依赖后续，另一root继续。所有失败和未执行槽位保留，不算零成本或已完成。
- 只有六场全READY才标full_expansion_complete。原始R01三场不能算本批A三场。

## 联合出题
- 仅六场全READY的root进入QA；一root一个六事实family，每个新scene恰一真事实，fact需source_turn_ref。完整范围为六个R02 scene，原历史仍供上下文。
- 每事实允许一个错误替代；要求同范围明确contradicts，unknown不得充作错项。先匿名资产审核，再按冻结seed规则组合Gold2/3/4/5各一题及Gold6对照，不足允许缺额。
- 每root两排序槽位，各预先冻结六个不同完整turn，每新scene各一个；保存ORDERING_ANCHOR_PLAN后才写QA_DRAFT。每item只一事件/言语行为，保留人物、模态和完成状态。
- 排序范围明确为六个新scene的对话出现顺序。非作者检查完整公开范围中的所有合理匹配，须全部得到同一严格顺序。不得换锚点救题。
- 非题目作者只读完整来源和公开资产，保存ASSET_SOURCE_JUDGMENTS；组合后再只读最终公开题保存PUBLIC_SOURCE_JUDGMENTS，之后程序构造原设计private Gold并交叉核对。不得读取当前solver分数救题。

## 评估与报告
- 固定v15 full-only协议，M0与既有M_clean_policy配对，multi_select每题6展示×3回答，ordering每题12展示×1回答。最大每模型228调用，无自动重试。
- Gold6单列对照，不进入新训练候选。候选仍按完整有效调用且0<mean_partial<1筛选。无新训练。
- 完整对话、各阶段稿件和失败、题面/答案/证据、按root/路线/scene计数、时间、输入哈希与限制均交付。
- 同一继承模型的不同代理不算异模型或外部独立认证；实际模型精确ID/调用usage不可得则null，文本tokens不冒充账单usage。
