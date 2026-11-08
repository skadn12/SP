# 当前扩充与 QA 工作交接

更新时间：2026-08-21

本文只记录当前代码和最近实际运行结果。历史 probe、诊断支线和已废弃方案不作为当前主流程。

## 1. 当前结论

当前有两条需要区分的路线：

1. **扩充路线**：基础合同在 `simple_append_v1`，当前 Memorax 驱动的最近成功 R1 在 `simple_append_memorax_v1_memory_event_continuation_probe_v1`。
2. **QA 路线**：正式 QA 以完整 Parent + Child 对话为事实来源；`qa_memory_only_probe_v1` 只是验证“只给 Memory 出题”的诊断支线，不能替代正式 QA。

最近一次实际完成的扩充是：

```text
conversation_chronicles_episode_41625_memory_only_fast_expansion_v1_live_1
```

它完成了 Memory 驱动的单次 R1，并成功写入 Child；这证明单 root 的链路可以跑通，但还没有证明多 root、连续多轮和批量运行已经稳定。

## 2. 扩充流程

### 2.1 基础扩充合同

代码目录：

```text
scripts/current_method/restart_n5_spell/simple_append_v1
```

核心流程：

```text
authoritative Parent
  -> Planner 生成少量 scene plan
  -> Round Realizer 一次生成整轮 Child scenes
  -> 本地结构、参与者、场景顺序和长度检查
  -> Parent + Child append-only assembly
  -> 通过检查的 Child 才进入 promotion / Add
```

关键文件：

| 文件 | 职责 |
| --- | --- |
| `contracts.py` | Planner、Realizer、scene 和长度合同 |
| `context.py` | Parent、尾部上下文和计划上下文组装 |
| `prompts.py` | Planner / Realizer prompt |
| `assembly.py` | Parent 与 Child 的追加组装、lineage |
| `live_runtime.py` | 一次扩充运行的 API 编排 |
| `live_canary.py` | 单次 live 入口 |
| `additional_round_live.py` | 后续轮次入口 |
| `multi_round_live.py` | 多轮实验入口 |
| `preflight.py` | 零 API 合同和输入检查 |
| `qa.py` / `qa_live.py` | 该基础分支附带的 QA 入口，不代表当前 Memory-only QA |

### 2.2 当前 Memorax 驱动 R1

代码目录：

```text
scripts/current_method/restart_n5_spell/simple_append_memorax_v1_memory_event_continuation_probe_v1
```

当前使用的实际链路：

```text
Fresh Parent
  -> Parent Add（首次 namespace）
  -> Memorax Bucket Search / overview
  -> 提取 concrete memory event candidates
  -> Planner 选择不同事件并生成 next-step candidates
  -> Semantic Continuation Gate 筛选重复/已发生的发展
  -> 程序选择第一个 PASS candidate
  -> candidate isolation，只有 selected candidate 进入 Realizer
  -> Round Realizer 一次生成整轮 Child
  -> Adherence / state-transition / 结构检查
  -> 通过的 Child 才 Child Add
  -> Post Search 检查新发展是否可再次检索
```

关键文件：

| 文件 | 职责 |
| --- | --- |
| `live.py` / `live_runtime.py` | 当前支线 live 编排和预算控制 |
| `workflow.py` | 扩充阶段顺序和运行状态 |
| `incremental_round.py` | R1/R2 轮次入口及增量状态 |
| `single_event_line.py` | 单事件线实验入口；不要误认为所有扩充都必须使用它 |
| `prompts.py` | 当前 Planner / Realizer 输入合同 |
| `contracts.py` | Planner、candidate、scene、Realizer 合同 |
| `context.py` / `memory_context.py` | Memory overview、事件候选和受限上下文 |
| `scene_continuation.py` | 失败 scene 的定向续写/替换逻辑 |
| `scene_continuation_live.py` | 定向续写 live runner |
| `scene_continuation_audit_recovery.py` | Adherence 失败后的恢复编排 |
| `scene_replacement.py` | 不可修复 scene 的替换逻辑 |
| `realizer_adherence.py` | Realizer 是否执行 selected plan 的结构化审计 |
| `realizer_adherence_live.py` | Adherence Auditor live 入口 |
| `realizer_adherence_offline.py` | 对既有 Child 的零 API 审计 |
| `promotion.py` | 只将通过审核的 Child 推进到写入阶段 |
| `async_write.py` / `write_state.py` | Memorax Add 异步任务、状态落盘和恢复；不重复提交未确认任务 |

### 2.3 最近成功扩充结果

运行目录：

```text
/root/autodl-tmp/gxy-selfplay/restart-n5-runs/conversation_chronicles_episode_41625_memory_only_fast_expansion_v1_live_1
```

报告：

```text
.../reports/MEMORY_EVENT_CONTINUATION_LIVE_REPORT.json
```

已确认：

```text
Planner = 1
Semantic Gate = 1
Realizer = 1
retries = 0
Event Verifier = 0
Parent Add = 1
Child Add = 1
child_add_verified = true
generation_ready = true
```

该次 token 统计：

```text
Planner:  1,891 prompt +   773 completion = 2,664
Gate:     9,151 prompt +   159 completion = 9,310
Realizer: 3,910 prompt + 1,994 completion = 5,904
总 LLM token: 17,878
```

这只是一次成功的 R1，不应表述为“批量、多轮已经稳定”。

## 3. QA 正式流程

正式 QA 的事实源仍然是完整对话，而不是 Memory 摘要：

```text
完整 Parent + Child dialogue
  -> QA Writer 生成 MCQ
  -> 程序做 schema / options / gold / evidence / duplicate 检查
  -> no-context Solver grounding
  -> 只有 grounding 通过的题进入 full-context Solver K=5
  -> 按 frontier/admission 规则录取训练题
```

正式 QA 代码主要在：

```text
scripts/current_method/restart_n5_spell/simple_append_v1
```

以及 Memorax 基础包装：

```text
scripts/current_method/restart_n5_spell/simple_append_memorax_v1
```

关键文件：

| 文件 | 职责 |
| --- | --- |
| `simple_append_v1/prompts.py` | QA Writer prompt 和题型合同 |
| `simple_append_v1/qa.py` | 题目 schema、gold/options 绑定、证据和录取基础逻辑 |
| `simple_append_v1/qa_live.py` | Writer 调用和 QA live 编排 |
| `simple_append_v1/qa_candidate_replay.py` | 对冻结 Writer 输出做零 API replay |
| `simple_append_memorax_v1/qa_sources.py` | Parent + Child 完整 QA 输入及 Memory topic hints |
| `simple_append_memorax_v1/qa_admission.py` | QA admission 辅助逻辑 |
| `simple_append_memorax_v1/frontier_k5_replay.py` | K=5 frontier replay |
| `simple_append_memorax_v1/qa_live.py` | Memorax 包装分支的 QA 入口 |
| `simple_append_memorax_v1/qa.py` | 包装分支 QA 辅助合同 |

### 3.1 Solver

当前本地 Solver 使用：

```text
Tokenizer / base model:
/root/autodl-tmp/Qwen3-4B

Trained/merged solver:
/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/training/models/merged
```

Solver 阶段：

1. **no-context grounding**：不提供完整对话，检查题目是否只凭题干/选项或泄漏线索就能答对；出现明显 semantic leakage 则拒绝。
2. **full-context K=5**：只有 grounding 通过的题才用完整对话重复回答 5 次。
3. 使用现有 frontier admission（当前标识为 `spell_frontier_1_to_4_of_5`）决定是否录取；不是“full-context 全对就一定录取”。

证据 ID 不匹配通常作为 evidence/审计标记处理；结构错误、gold/options 无法绑定等仍可阻塞题目。

## 4. 最近 QA 诊断支线

代码目录：

```text
scripts/current_method/restart_n5_spell/simple_append_memorax_v1_qa_memory_only_probe_v1
```

它是隔离实验，Writer 只看冻结 Memorax overview，不看完整对话；完整对话只用于离线 truth audit。不要把它和正式 QA 合并。

最近运行：

```text
/root/autodl-tmp/gxy-selfplay/restart-n5-runs/conversation_chronicles_episode_41625_qa_memory_only_probe_v1_live_2
```

结果：

```text
Writer calls = 1
retries = 0
finish_reason = stop
prompt = 1,692
completion = 1,545
total = 3,237
raw questions = 12
mechanically valid = 12
transcript-verified usable = 6
admitted before Solver = 0
```

本地 Solver replay：

```text
/root/autodl-tmp/gxy-selfplay/restart-n5-runs/conversation_chronicles_episode_41625_qa_memory_only_probe_v1_solver_replay_1
```

结果：

```text
verified candidates = 6
no-context grounding pass = 3
full-context K=5 = 3
admitted = 0
paid API calls = 0
```

结论：Memory-only QA 证明 Writer 可以从 Memory 生成结构化候选，但目前不能替代完整对话 QA，也没有产出可录取训练题。

## 5. Memorax 在当前流程中的边界

Memorax 的职责：

```text
长期事件/人物线索召回
提供 Planner 的可继续事件候选和摘要
Child Add 后检查新增发展是否可再次检索
```

Memorax 不是 authoritative transcript，也不保证提供本地 canonical turn provenance。最终事实仍以本地 Parent/Child dialogue 为准，尤其是正式 QA 的 evidence 和 gold。

核心接口和状态文件：

```text
simple_append_memorax_v1/client.py
simple_append_memorax_v1/async_write.py
simple_append_memorax_v1/async_recovery.py
simple_append_memorax_v1/write_state.py
simple_append_memorax_v1/generation_registry.py
```

Add 的 task_id 必须先落盘；处理超时不能直接重复 Add。明确确认未写入才允许新的 attempt，否则保持 pending/ambiguous 并查询原 task。

## 6. 当前不要混用的支线

以下是诊断或历史路线，不得当作当前主流程入口：

```text
simple_append_memorax_v1_qa_position_first_probe_v1
simple_append_memorax_v1_qa_fact_first_probe_v1
simple_append_memorax_v1_incremental_event_chain_probe_v1
simple_append_memorax_v1_state_resolution_probe_v1
simple_append_memorax_v1_qa_memory_only_probe_v1
```

这些支线用于定位 evidence、事件粒度、Memory state resolution、position-first 或 Memory-only QA 的问题；它们的 prompt、schema、报告不能反向覆盖正式路线。

完整支线登记见：

```text
EXPERIMENT_BRANCH_REGISTRY.md
```

## 7. 当前能力边界

已经实际验证：

```text
单 root 的 Memory-driven R1：可以完成并写回 Child
一次整轮 Planner / Gate / Realizer：可以运行
candidate isolation：有独立检查
Child Add 后 Post Search：有检查入口
正式 QA 的 Writer -> Solver 链路：代码已存在
```

尚未宣称稳定：

```text
多 root 批量扩充
连续 R2/R3 多轮稳定续写
每轮都能保持事件多样性
Memory-only QA 可以产出可录取训练题
QA 全流程大规模录取率
```

因此下一步若继续实验，必须先明确是：

```text
扩充 robustness pilot
或
正式完整对话 QA
或
Memory-only QA 诊断
```

不能把三者的输入、prompt、报告和 admission 结果混在一起。

## 8. 推荐入口顺序

### 只做零 API 检查

```bash
cd /root/autodl-tmp/gxy-selfplay/current_method_github
python -m scripts.current_method.restart_n5_spell.simple_append_memorax_v1.preflight
python -m scripts.current_method.restart_n5_spell.simple_append_v1.preflight
```

具体 runner 参数以对应目录 README 和 `--help` 为准；不要从历史 run 目录 resume。

### 做一次扩充 R1

使用当前 Memory 驱动目录的 fresh-run 入口：

```text
scripts/current_method/restart_n5_spell/simple_append_memorax_v1_memory_event_continuation_probe_v1/live.py
```

必须使用新的 run root 和新的 Memorax namespace；已写过 Parent/Child 的 namespace 不要重复 Add。

### 做正式 QA

使用完整 Parent + Child 的 QA 输入和正式 QA runner，不要使用 `qa_memory_only_probe_v1` 的 Writer 输入替代它。完成 Writer 结构检查后再进入本地 Solver grounding 和 K=5。

## 9. 运行纪律

每次实验都要保存：

```text
implementation / working-tree SHA
input dialogue SHA
prompt/schema SHA
run root
Memorax namespace
API calls / retries
prompt/completion/total tokens
failure_stage
```

禁止：

```text
覆盖历史 run
把 shadow/replay 结果冒充真实成功
未确认写入就重复 Add
把诊断支线 prompt 复制回主线
未通过 Child 审核就写入 Memorax
```
