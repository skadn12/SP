# V2.2 精确修改方案（待确认，未实施）

## 1. 修改决策

本轮根因并不支持无差别重写六个模块。拟新增一个隔离的 V2.2 薄层及新 prompt，保留 V1/V2/V2.1 和所有冻结产物；不开发模型/API 执行器，不改 Reviewer 整体 schema，不改两条对话扩充路线。主要工作放在 **Discovery 的关系搜索与冻结前选择**，其次是 Construction 的复杂依赖保留；Evidence Audit 复用旧引擎并补查询条件化分析，Option/Freeze 做一个边界修复。

| 模块 | 修改幅度 | 修改理由 | 复用范围 |
|---|---|---|---|
| Discovery | 核心 prompt 与关系卡增强 | 有自然关系漏枚举；缺高阶搜索记录 | 现有事件ID、来源引用、身份与完整扫描结构 |
| Ranking | 新增轻量选择契约 | 现有 SELECT 没有 L3/L4 目标约束 | 现有自然性/不确定性/Gold 可核验前置条件 |
| Construction | 定向增强 | 103866 题干提供状态使证书缩短；无复杂桥保留机制 | 先冻 Gold、动态题型、真实错误路径、静态选项筛查 |
| Evidence Audit | 补充语义输入，不重写路径算法 | ledger 完整不等于题干条件下完整 | AND/OR、最小支持集、逐轮与语义组删除、全源扫描 |
| Difficulty Gate | 新增明确 rubric 和阶段对照 | 两个 L1 目标被送往困难题审核 | 既有机械验证只作正确性检查 |
| Option/Freeze | 新版本窄边界修复 | 104814 持久化使显示标签漂移 | 原 strict parser、raw bytes freeze、receipt、Gold 比较 |

文件建议：新增 `scripts/current_method/base_qa_pipeline_v2_2.py`、`base_qa_prompts_v2_2/`、定向测试文件及版本说明。具体目录沿项目现有习惯；这些是未来计划，本轮没有创建它们。

## 2. 难度契约：按最容易的可靠路径，不按作者最长证书

| 级别 | 必须如何区分 | 排除的伪难度 |
|---|---|---|
| L0 | 单轮可直接读出目标，包括直接陈述的原因、更新或总结 | 该轮提到三个历史场景，也仍可能只是提取 |
| L1 | 局部明确数值、日期或序列上的基础计算 | 算式拆成多个步骤、单位换算、引用两轮不能自动升档 |
| L2 | 跨轮/跨场景找回并绑定信息或对照状态，但没有不可替代的高阶关系操作 | 两个独立字段、人物状态清单、同话题事件拼接 |
| L3 | 至少一个真实且不可替代的跨场景事件/人物/状态关系操作；需恢复身份、保持/更新状态、处理关联约束而产生新结论 | 只把答案位置放远、题干把桥告诉读者、最终单轮已完整总结 |
| L4 | 多场景链上的状态与约束互相耦合，必须组合才能得出唯一结论；移除关键链节或解除约束会改变可行结论 | 独立事实AND、线性步骤多、场景数多、需要读长文 |

同一个信息需求如既有长状态链又有单轮总结，按总结路径定低等级。若复杂性只在未授权来源中成立，状态是 `SCOPE_BLOCKED`，不能提升当前等级。正式证据资格与实际可见答案捷径另记：parent/R01虽不可充当正式前提，既然会完整展示，也必须检查其中是否直接泄露答案或提供更简路径；不能因禁止引用就把实际可读捷径算成不存在。如果最简路径或身份仍有争议，标 `UNRESOLVED`，不在困难产出中计数。没有安全 L3/L4 时输出 `NO_SAFE_HARD_RELATION`；保留低层关系作为诊断，不为填题降级目标。

这个 gate 只能验证**预期结构难度**。未来本地 Solver 实验才能判定模型是否真的会错；不能在 prompt、人工 rubric 或同模型盲审之后提前声称“已证实困难”。

## 3. Discovery：从自然信息目标搜索必要桥

### 3.1 事件与状态表示

在现有事件 ledger 上增加变更条目，不重建来源格式：

```text
event_id / actor / object / scene / evidence_refs
state_before -> action_or_observation -> state_after
plan_status: proposed / requested / confirmed / cancelled / completed / unresolved
identity_link: same_event / explicit_related_event / independent_event / unknown
identity_basis_refs / uncertainty / scope_status
```

状态值必须有来源支持；没有转变就记录 persistence，不编造“计划→完成”。同人物、相似活动、同Tuesday不能代替事件身份桥；“没有提到完成”也不能自动当作“明确未完成”。保留父层与 R01 的 context-only 标识，它们可帮助解读但不能偷偷成为目标前提。

### 3.2 有限的定向搜索过程

完整扫描后，围绕来源确实支持的自然需求，做一次跨场景定向检查：

1. 找同一事件的更新、取消、完成、冲突、知识/经验状态保持及明确关联事件。
2. 检查前置条件、资源/人物/时序约束是否真正共同决定某个行动或状态。
3. 为每个初步目标主动寻找本地直述、末轮摘要和另一条更短路径。
4. 记录成立、不成立及不确定的关系；不能只归档最后 SELECT，也不能把所有场景两两笛卡尔比较。
5. 区分“没有发现”“不支持身份”“本地摘要可答”“只是检索”“缺范围许可”，分别保留原因。

新 prompt 应优先要求寻找关系，而不是先找可相减的数字。允许发现一个真 L3、零个 L4；不预设每 root 都有复杂关系，不要求凑九条或任意数量。

### 3.3 关系卡（拟新增字段）

```text
relation_id, information_need, event_ids, source_scope_digest
evidence_distribution: scene_ids / turn_refs / source_types
identity_and_relation_basis
dependency_graph: fact_nodes / state_transition_edges / constraint_edges
reasoning_steps: supported relational operations, not prose subdivisions
state_changes_and_persistence
minimal_support_paths, alternative_paths, direct_summary_bypasses
natural_value_and_reason
minimum_route_level, level_rationale, uncertainty
critical_bridges: operation / evidence / why indispensable
scope_status, decision, rejection_reason
```

程序只验字段绑定、图引用和报告一致性；自然关系与最简路径仍需语义分析，不能用 schema PASS 冒充发现成功。

## 4. Ranking：先准入与真实性，再选合格难度

不引入一个调权重后总分足够就能抵消来源风险的黑箱分数。采用可审核的两阶段选择：

- **资格层**：精确来源范围合法、身份和关系有据、自然需求成立、Gold 可唯一核验、无未决替代答案。任一不满足就不进入正式构题池。
- **困难目标层**：审查最容易可信路线是否仍为 L3/L4，关键桥在完整来源及预期题面下是否不可替代。L0/L1/L2 保留为低层诊断池，不获得困难 SELECT。

对合格池，优先选择自然用途清晰、关系确定、具有可诊断状态/身份错误的目标；同等条件下再考虑复杂性。不要强制 L4 压过更可信自然的 L3。记录 `selection_receipt`：全部候选、排除依据、入选理由、关键桥、未入选者、预算。没有更强候选就诚实 NO_SAFE，而不是把简单算术包装为困难关系。

这修复的是已证明的**选择目标缺口**；不是宣称上轮已经存在一个排序颠倒的例子。

## 5. TARGET、Construction 与 Evidence Audit

### 5.1 冻结前后应保持的内容

TARGET 在题干/选项前冻结现有 Gold 和语义，同时增加：

- `target_difficulty_level` 与最简路线理由；
- `critical_relational_bridges` 与对应身份/状态/约束证据；
- `source_fact`、`question_stipulation`、`optional_background` 三类输入区分；
- 已知替代路径与摘要旁路，以及尚未排除的不确定性。

不能先冻结带缺口的复杂链，再由题干添加关键事实“补证”；起算规则一类必要外加条件要如实标注。若自然问题必须靠显式条件才成立，要重新评定其信息价值和难度，不把条件本身当来源推理成果。

### 5.2 构题阶段

先围绕信息需求写自然问题，再按目标语义选择单选/多选/排序及选项数量。避免用多选把数个独立事实打包，或用排序把显而易见的日常步骤冒充状态链。

干扰项优先来自真实可定位的错误路径：错误身份延续、把旧计划当最新状态、错误把请求当批准、把独立事件合并、端点/对象绑定错误。引用存在不代表干扰合理，需说明“读者为何可能走错”及“为何与Gold不构成歧义”。静态长度/数值极值筛查继续保留，但不能靠关键词黑名单代替语义审查。

成题后分别核验**允许证据范围中的正式推导**，以及 **实际全部可见来源（包括context-only的parent/R01）＋实际题干＋实际选项** 下的最简答题路径。后者不授予来源资格，但决定实际包是否仍需要复杂推理：

1. Gold 语义、计算/排序和canonical option映射保持一致；
2. 任一关键桥是否被题干提供、被选项排除法绕过、被末轮总结替代；
3. 删除某个 turn 与删除同一语义前提的全部替代支持分开做；
4. 比较 `target_level` 与 `constructed_minimum_route_level`，保存具体缩短路径；
5. 不满足目标就保留本次产物和 HOLD；不循环改写同题直到通过。

### 5.3 复用必要性引擎的边界

继续用 V2 的 AND/OR antichain、逐轮删除、组删除与全源扫描。补充两层支持标注：**来源单独支持的语义**与**查询条件下得出答案的最短路径**。题干事实和选项约束是已给信息，不能从审计输入中隐去。

103866 应保留两个数值组的真实必要性，同时明确确认状态并非数值必需；104814 应保留条件下的两前提加法，同时明确它不符合困难目标。两例均是未来回归负例，不改写成通过的历史案例。

## 6. Option/Freeze：新版本统一入口

最小修复选用**首次冻结前的稳定ID规范化＋私有渲染契约**。不能只改出包或只改比较器。

1. 在新 V2.2 边界复制 candidate，仅按稳定 option ID 排序 `options` 对象键；不修改调用方原对象。首次 packet/hash/receipt 生成前完成，重复规范化应幂等。
2. 冻结规范化 candidate；重新读回后再生成并校验实际送审 packet。新封装路径的所有入口（freeze、packet、parse、compare）都验同一规范序和版本，禁止旁路传入另一个未规范内存对象。
3. 冻结私有 `render_contract` sidecar：schema/renderer version、seed、candidate ID、预审 candidate digest、source-scope digest、public packet digest、精确 display→canonical 双射。候选 digest 的定义排除 sidecar 自身，避免哈希自引用。新增外层 V2.2 私有 freeze/envelope 绑定候选、契约与内层 V2.1 receipt 的摘要；内层 receipt 及公共 projection 保持原字段，不能向旧 receipt 追加契约字段。V2.1 740–744 对 receipt 做精确字段集合校验，追加字段会被拒绝。
4. 新 V2.2 wrapper 先校验外层 envelope、契约和规范化对象，再把精确旧 schema 的内层 receipt/参数投影送入现有 V2.1 raw/parser/comparator；旧内部 renderer 每次获得同一规范化对象，重建结果必须与契约逐项相同。公共 Reviewer projection 保持原样。若实现改成新公共渲染/receipt schema，则 parser/comparator/receipt 验证器必须同步升级，不能半升级或 monkeypatch 历史模块。
5. private contract、Gold、TARGET、canonical私有分析仍不向 Reviewer公开；公共包继续白名单。契约、文件读回或 packet 任一失配就停止送审/比较。外层同时冻结一个小型 implementation manifest，按字节SHA绑定V2.2薄层、normalizer/renderer入口、新prompts及被复用的legacy依赖；freeze/read/replay均核验。V2.1现有 `implementation_bindings`（952–956附近）不覆盖新文件，版本字符串不能替代这些绑定。
6. 只规范对象键；**不能排序 ordering Gold、expected_answer、推理步骤、原始轮次等有序数组**。Gold 改变不得改变公共选项渲染，但私有冻结绑定须拦截审后 Gold 替换。
7. 历史104814 raw、packet、freeze和HOLD一律保留；不通过迁移或重解释旧raw宣称修复成功。

规范序解决此根因；私有契约提供持久化边界的可核验绑定。它不解决复杂关系供给、难度和正式素材许可，这些独立判定。

## 7. 未来定向回归与验收（未实施、未运行）

| 测试族 | 正例/反例与预期 |
|---|---|
| 难度分级 | L0直述、L1时间差/加法、L2字段绑定被保留但不得困难SELECT；跨场景且不可替代状态桥的受控L3、耦合约束L4可进入候选池 |
| 摘要旁路 | 多场景历史链最后单轮重述时降级；不能删除摘要或隐藏容易证据来升难度 |
| 身份与范围 | 同人物独立事件、相近时间、同地名不合并；R01-only关键前提保持SCOPE_BLOCKED；不得推断AM/PM等未决值 |
| 选择 | 真L3与自然L1并存时选前者；只有L1时NO_SAFE_HARD_RELATION；不以数量或总分抵消资格条件 |
| 构造保留 | TARGET有关键桥，stem给出该桥或options可独立排除时被识别；保持自然问题的真L3不过度拒绝 |
| 证据 | 单turn冗余但语义组必要、多个不同充分集合、题干预设、等价重述、完整source旁路，分别记录 |
| 顺序稳定性 | 历史四选项等小fixture穷举24排列，compact/pretty、sort_keys开关、冻结读回、独立进程重载后packet/hash/mapping恒等；测试规范化幂等且输入不变。大选项集采用固定逆序、边界排列与有界性质检查，不枚举26! |
| 题型映射 | 单选/多选/排序，2及26选项边界用有界样例；排序Gold特意不按ID排序，仍正确roundtrip |
| 篡改/隔离 | question/option文本或ID、seed、scope、renderer版本、mapping、packet、receipt任一变更均封闭失败；私有字段不进入公共包 |
| raw及历史 | 继续保留正常/字段错误/重复key/非法UTF8/多余字段/未知label等严格解析；不修raw、不重试；V1/V2/V2.1 fixture行为不变，104814原始HOLD不变 |

受控正例仅是未来工程测试素材，不能获得正式 QA 证据资格，也不能算真实生成成功率。历史合格样例需按新定义重定级；不能因为历史PASS就标L3。回归结果与真实复杂关系发现、最终盲审通过分别报告。

验收次序：最小模块与prompt变更→定向回归→旧版兼容回归→精确来源授权检查→用户确认的有限真实实验。任何一个环节失败均保留结果，不扩展为新执行器、探针或无限候选搜索。
