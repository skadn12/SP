# Base QA Generation V2.2：分阶段语义工作契约

本目录是直接由 Codex 执行的离线语义工作指令，不是自动调用模型的执行器。它把 Discovery 从单次关系提取扩为可追溯的分阶段工作；代码只核验提交的结构、绑定、图与冻结边界，不能凭字段齐全证明发现完整、关系真实或出题能力提高。

## 顺序及传递材料

| 阶段 | 指令 | 必须传入 | 产物与后续用途 |
|---|---|---|---|
| 1 | `inventory.txt` | 完整实际可见来源、精确来源绑定、外部准入范围 | 稳定事件身份、状态观察/变化、逐轮扫描；保留不确定性 |
| 2 | `typed_relation_sweep.txt` | 同一完整来源、inventory | 十类关系检查记录、全部候选、排除理由和未检查范围 |
| 3 | `composition_and_counterpath.txt` | 同一来源、inventory、全部关系卡 | 有依据的关系组合、依赖图、替代充分路径和摘要旁路 |
| 4 | `ranking.txt` | 上述全部记录和来源资格 | 按最容易可信路线分级；只选择合格 L3/L4，否则 `NO_SAFE_L3_L4` |
| 5 | `target.txt` | 已冻结选择记录及唯一合格关系 | 构题前冻结语义答案、关键桥、图、难度和证据；不含题干/选项 |
| 6 | `construction.txt` | 不变 TARGET、同一来源和全部反路径 | 一次构题、动态题型、真实错误路径干扰项；不改目标 |
| 7 | `postconstruction_audit.txt` | TARGET、题面、所有选项、冻结映射及同一完整来源 | 重新核验实际答题路径、难度是否保持、桥是否被题干/选项替代 |

数据格式以 `../base_qa_schema_v2_2.json` 和 V2.2 实现的机械校验为准。旧版事件、证据、TARGET 和 candidate 的兼容字段不自行改名。阶段 artifact 的 SHA 由实现计算；Challenger 不应填造哈希、外部审核身份或来源批准。Reviewer 继续使用现有 V2.1 严格公共 schema、原始输出保留及先盲答冻结再比较的协议；本目录不新增 Reviewer 协议。作者的 postconstruction audit 不等于独立盲审。

Discovery envelope 的精确字段为 `schema_version: base_qa_generation_v2_2:discovery`、`root_id`、`source_sha256`、`source_scope_sha256`、`events`、`relations`、`source_scan`、`coverage`。TARGET 使用 `selection_sha256` 绑定选择收据，并原样复制入选关系的 `dependency_graph` 与 `path_audit`。候选的 `query_path_audit` 另外绑定 `question_sha256`、`options_sha256`，记录 `critical_bridges_preserved`、`supplied_bridge_ids`、`minimum_route_level`；合格困难候选分别要求真实的 `true`、空数组、`L3/L4`，不满足的首轮分析必须保留为 HOLD，不能伪填通过值。

## 覆盖与边界

Typed sweep 必须逐类记录 `cross_scene_event_identity`、`actor_action_binding`、`event_state_update`、`temporal_plan_revision`、`multi_actor_roles`、`object_attribute_binding`、`quantity_unit_scope_constraint`、`causal_relation`、`joint_event_constraint`、`composed_state_chain`。每类要区别“实际检查后未发现”和“未检查”，保留对应场景/事件范围、候选数和理由。完整逐轮扫描不等于穷尽关系发现，图内路径枚举也不等于完整自然语言证明。

事件状态 `status_state` 与 chronology 分开：未发送、未批准、已完成不自动构成不同独立事件间的时间顺序。同人物、相似名称和相近时间不支持同事件合并。关系组合必须有真实耦合；独立事实包、多个数字、更多场景或拆细算式不能提高难度。

高阶 operation 还必须有可执行 `rule:{type,expected_result}`，不能靠 `constraint_apply` 等标签把 `24−6` 升成 L3。本版仅支持三类离线规则：`precedence_closure` 从明确前置边求闭包与唯一顺序；`guarded_transition` 对明确身份事件的状态按有据条件执行更新；`constraint_filter` 在至少两个真正耦合变量的域和约束上筛选可行解。fact 可附 `symbolic_value` 保存有引文依据的边、状态/更新、域/约束。执行结果必须与 `expected_result` 相等；来源到符号值的提取仍是作者语义判断，执行通过不是自然语言蕴涵证明。无法忠实表达为三类规则的自然关系保留为 `UNRESOLVED`/诊断，不强行编码，也不据此宣称素材中不存在复杂关系。

未支持的原始关系卡仍放在 `relations` 中，使用 `representation_status:UNSUPPORTED`、`decision:HOLD/REJECT` 和非空 `representation_reason`，保留原始关系/证据字段；`dependency_graph`、`path_audit` 可缺失或原样保留其未支持形式。这类卡不能进入 hard pool，但不会仅因不可执行而使其他合格关系所在的整轮无效。可建模关系使用 `MODELED` 或省略该字段，必须满足完整图/路径契约。

TARGET 另冻结 `answer_projection:{path:[字符串键或非负整数索引,...],expected_answer:...}`，从图执行器的 `symbolic_answer` 取出语义答案。空路径表示整个结果。代码必须要求投影结果、TARGET 的 `expected_answer`、projection 的 `expected_answer` 完全相等，避免图算一件事、Gold 冻结另一件事；schema 只检查路径结构，真实取值和相等由执行器核验。选项生成不得改变投影。

当前执行器的投影难度验收范围比通用路径语法更窄：`precedence_closure` 仅支持 `['order']` 的完整排序；`constraint_filter` 仅支持 `['solutions']` 的完整唯一赋值；`guarded_transition` 支持 `['state','values',...查询字段]`，反向核验每次计入难度的更新是否影响该字段或后续必要 guard。修改无关状态字段不能给当前问题升难度。代码还检查跨变量约束是否真正筛掉可行解，并剔除不影响结果的冗余 typed 输入。未支持投影或与查询无关的链保留为 HOLD，不通过更换目标或强行编码绕过；schema 的通用 path 形状不等于执行器支持所有投影。

已确认充分的替代路径必须进入依赖图的支持分支。仅在 `counterpaths` 中声明另一条 L3/L4 路径、却继续用原图声称证据必要，会阻止困难准入；低阶充分旁路仍直接降低难度。图外未决路径保持 HOLD。

## 两种来源范围

正式证据范围取决于已有外部准入记录和精确版本；实际可见范围决定读者有哪些更简单的答案路线。parent/R01 等 context-only 材料可影响实际难度，但不会因此自动获得证据资格。禁止通过隐藏摘要、删去上下文或自批合成材料让困难目标看起来成立。当前未授权 roots 不得因离线机制测试通过而启动正式 QA。

## 难度与一次构题

L0 单轮直接提取；L1 局部事实/简单计算；L2 跨轮检索但无不可替代高阶关系操作；L3 必须有真实、必要的跨场景事件/人物/状态关系操作；L4 必须组合多关系或耦合约束/状态链。按最容易可信的充分路径分级，不能选择最长证书定级。不确定就不标成合格困难目标。

难度门槛在 TARGET 冻结前执行；成题后把题干条件、选项排除和全部可见来源共同纳入审计，防止复杂桥被题面直接告诉读者。自然题型和选项数由关系决定，干扰项引用真实错误路径。一个 candidate 失败就保留并 HOLD；不得同题反复改写直到通过。允许零题。

## 能力结论限制

本轮合成 fixture 与历史冻结案例只验证离线机制和故障边界。不能用测试通过、候选数或 schema PASS 宣称 Discovery 已能稳定发现真实 L3/L4，不能提前声称本地 Qwen 会出错。正式能力验证仍需授权来源、有限真实生成和真实独立盲审；Solver、训练、学校 API 和两条扩充路线不属于这些 prompts 的执行范围。
