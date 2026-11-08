# 一次增量计划与未完成目标保留 v1

当前默认初始 Planner 恢复原输出合同，并保留已补回的生成指导。
root 补充仍沿用原只追加合同；在补充返回后用程序整理字段层级和阶段编号，
再进入原校验与合并流程。接入和离线回放见
[SUPPLEMENT_FORMATTING.md](SUPPLEMENT_FORMATTING.md)。
[SHARED_PLANNER_OUTPUT.md](SHARED_PLANNER_OUTPUT.md) 所述共享原始输出合同仅保留为显式历史版本。
Adherence 默认使用 `adherence.scoped_evidence_v4`，区分本场事实与历史证据，
并阻止删过自创条件的审核直接晋级；真实编号仍附在对应对白旁。
输出字段约束和旧审核结果的离线整理边界见 [ADHERENCE_EVIDENCE.md](ADHERENCE_EVIDENCE.md)。
下文保留历史 v1 政策说明。

显式 policy：`expanded_single_plan_incremental_retention_v1`。
这是版本化的新政策，不代表旧合同原本支持保留未达最终目标的场景。
本次只落地实现和离线回归，不授权任何真实请求。

## 范围与最小改动

只新增本目录，不改旧源码、QA schema、旧 profile、模板副本、保护清单和历史 run。
原失败计划通过冻结 `variant_selection.value` 接入；其原始响应、旧验证和账本仍不变。
新入口 `run.py -> workflow.execute_policy` 显式使用本 policy；旧 `workflow.execute_round` 不改路由。
代码复用原 Planner 校验/编译/候选 materializer、事件 ID 对照、outline 机械检查、
Semantic Gate 输出校验、Realizer 规范化/复制检查和 scene-level 机械晋级。

| 文件 | 作用 |
| --- | --- |
| planning.py | 新 profile；保留完整旧诊断；只对精确 future-only 目标标签分流；只追加阶段的不可变 patch |
| budget.py | 完整预算校验、root/round 持久账本、请求前记账与整批检查容量预留 |
| prompts.py | 固定增量模板、目标语义说明、Planner 接口适配、双模式 Realizer 和 Gate renderer |
| retention.py | 同一 adherence 调用的版本化维度、引用检查、实际条件依赖与机械晋级适配 |
| workflow.py | 冻结计划入口，单次补充、选初始最多六场、补充场景不受六场截断、分批执行 |
| run.py | 默认 mock；真实路径独立授权/SHA/预算门和无重试单次传输；无 Memory 能力 |
| fixtures.py / test_policy.py | 原真实失败输入 + 明确标注的合成模型响应与安全回归 |
| preflight.py | 新目录交付、现有受影响测试、冻结文件 SHA 前后对照 |

## 新旧规则

| 检查点 | expanded_single_plan_v1（不变） | 本 policy |
| --- | --- | --- |
| 生效完整方案 | 1 | 1；补充是增量版本，不是备选方案 |
| 初始候选/执行 | continuation≤3、新池≤8、执行≤6 | 相同；入口消费冻结方案，不重跑初始 Planner |
| future_only_after_state | continuation 可在 Gate 前导致整个方案失败 | 原判定留档；目标诊断与硬错误分开；最多一次 pre-Gate 补充 |
| 补充次数 | 无此入口 | root+round 共一次；失败/超时/进程退出均不恢复次数 |
| 最终场景数 | 最多6 | 初始最多6 + 零至多个已通过补充阶段，受明确预算限制 |
| Realizer 批次 | 初始最多3批 | 每批最多2场，无固定全轮3批截断 |
| 未达目标 | 旧 adherence/恢复/晋级规则 | 目标保留原值；新进展和其他检查通过才可保留 |
| 审计 | 原合同 | 同一 adherence 角色的新维度，不增加另一个审核模型 |
| 自动恢复 | 旧配置仍可复现 | 冻结计划新入口没有 repair/continuation/replacement；不把失败改名另一次补充 |
| Gate 上下文 | 旧模式可 Memory-only | 新 policy 的 Gate 显式读取权威历史与已保留文本；Memory不发任何请求 |
| 事实记录 | 旧路径 | 只有实际保留对白；计划与审计摘要绝不进入事实输入 |

`memory_only_fast` 是继承的原 Planner/候选输入身份；新入口不运行 event-verifier，
Gate 的实际覆盖由显式 `gate_context_mode` 决定，不宣称 Memory 摘要为事实。
非目标类失败在新入口保守隔离/停止，不引入新恢复。旧 profile 的恢复能力与上限没有被删除。
初始 Planner renderer 的目标措辞已同步，但仅做接口回归；本次冻结失败方案入口不会调度它。

## 计划与场景身份

保存 `original_plan` + 原始 SHA、旧 validator 结果、增量 `patch`、新 `merged` 版本和差异。
补充仅可针对具有目标诊断的稳定候选。原 goal、after_state、无关候选与原方案不改写。
每个补充场景沿用源候选 ME/VT，另有唯一 `phase_id`；不是伪造 ND 候选。
依赖用 `{phase_id, condition}` 指向先前阶段，不能前向引用或循环。
phase 的执行 scene/turn refs 沿用既有 R03:Sxx:Txxx 语法，独立 branch 和完整快照 SHA 在外层身份中保存。
不能跨快照解释同名编号；原文编号保留，失败文本不进入后续历史。原 append-only 累计拼接器和校验器直接复用。

## 同一个 adherence 调用的维度

| 字段 | 取值/用途 |
| --- | --- |
| goal_status | realized / partial / not_realized / UNKNOWN，不修改原目标 |
| content_decision | retain / reject / insufficient |
| participant_adherence | pass / fail / UNKNOWN |
| history_consistency | pass / fail / UNKNOWN |
| novel_progress | pass / fail / UNKNOWN，重复意向不算进展 |
| actual_states | statement、intention/arrangement/action/completed、实际 turn_refs |
| goal_evidence_refs | 原目标达成/进展依据；partial/realized 必须有可定位引用 |
| condition_results | 同一调用审计事先给出的后续条件，established/not_established/UNKNOWN + 引用 |
| reason | 审计理由，原始输出始终保存 |

目标 partial/not_realized + 有依据的新进展 + 其他维度 pass → 政策可保留候选 → 原复制/依赖机械检查 → 保留。
未知维度、只有总 FAIL、无合法引用、复制、绑定漂移或历史冲突 → 不晋级。
单独生成的 policy eligibility view 供旧机械晋级器使用，绝不重写原 audit 为 realized。
模型的 “pass/established” 仍是判断，不是真值证明；本次没有语义准确性验证。

后场依赖必须由前场实际引用支持的 `established` 条件满足。
“前场保留”不等于“前场目标完成”；邀请只拟好，不能把发送条件视为已成立。
未明确表达的潜在依赖仍有语义风险：UNKNOWN，本次没有新增自动依赖推断模块。

## 预算与持久性

真实入口必须提供所有字段，不采用旧16次默认值：

- `total_requests`：本轮本 root 新请求总上限；历史初始 Planner 1 次另外记账。
- `role_requests`：planner / supplement / gate / realizer / adherence / closure 各自上限。
- `max_input_tokens`、`max_output_tokens`（每角色）、`context_window_tokens`。
- `max_generated_tokens`：按预留输出上限保守计费；`max_cumulative_tokens`：累计长度上限。
- `scene_output_reserve_tokens`：每场新增对白容量，单独为审计输入预留空间。

冻结初始计划入口要求新增 `planner=0`，`supplement≤1`；Planner/补充输出上限8192。
开始前预留 closure；每批预留所有 Gate + 1 Realizer + 1 adherence，才会请求生成。
closure 是本地确定性收口，模型预算仅保守占位而不会实际调用新审核角色。
任一资源不足 → `DEFERRED_BUDGET`，计划保留，不截断历史，不跳过审计晋级。
预算推导的场景安全界与初始6场上限分别记录。预算是上限，不是填满配额。
未使用的保留请求预算不退款：可能更早延后场景，不会超额调用。

持久账本在请求前 fsync；锁在稳定文件上。请求状态不明也消费机会。
真实 CLI registry 固定在项目 `restart-n5-runs/policy_state/<policy>/`，不随 run/branch 切换。
同 root+round 二次完整执行拒绝；本版本重启后不自动续发任何不明状态请求。
预算样例是大容量 mock 用例，绝不是当前40960服务端配置或真实授权。
真实入口须有本地 tokenizer 文件/SHA，只本地加载，按 chat template 计数；不探测服务端容量。

## 运行入口（真实命令未执行）

离线回归（新目录，socket connect/connect_ex/sendto/create_connection 阻断）：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m scripts.current_method.restart_n5_spell.h4_round1_r03_incremental_plan_retention_v1.preflight --output /ABS/NEW_OFFLINE_DIRECTORY
```

后续真实运行模板，必须再次明确授权、重新冻结身份与预算；本次未执行：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m scripts.current_method.restart_n5_spell.h4_round1_r03_incremental_plan_retention_v1.run --execute --bundle /ABS/AUTHORIZED_BUNDLE.json --budget /ABS/AUTHORIZED_BUDGET.json --model-config /ABS/AUTHORIZED_MODEL_CONFIG.json --authorization /ABS/EXPLICIT_AUTHORIZATION.json --run-dir /ABS/NEW_REAL_RUN_DIRECTORY
```

授权文件需 `explicitly_authorized=true`、policy/identity、bundle/budget/model_config SHA，
以及 `implementation_manifest()` 的逐文件当前 SHA。bundle 需要四份 frozen_sources 的路径/SHA及一致的解析内容；
fixture 标签或 offline branch 不能直接实跑。model config 必需 endpoint、model、temperature、timeout_seconds、
api_key_env、本地 tokenizer_path、tokenizer_json_sha256。凭据仅从环境读取，不能写入报告或 messages。
无授权/缺预算/源 SHA 不一致时在构建真实请求之前拒绝。

## 交付边界

offline_failed_plan_path 中全为明确命名的 mocks，不生成正式 r03_delta 或累计新R03。
保留对白样例只在 `local_retained_dialogue_fixture.json`；后续 Memory 可用文本接口只接受保留对白列表，
本目录没有 Memorax 操作接口。旧 Parent/R01/R02/旧R03和失败账本完全只读。
真实模型质量、8192服务端支持、HTTP实连、审计语义准确性：NOT_TESTED。
旧 protected-manifest 差异不更新 expected SHA，不借本次测试重建旧保护基线。
