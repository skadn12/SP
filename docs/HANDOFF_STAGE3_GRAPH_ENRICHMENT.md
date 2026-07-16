# Stage3 图谱补全交接（2026-07-13）

## 当前目标

不要重跑 Stage1 或 Stage2。以已验收的 Stage2 solver 和现有 77 条 Stage3 parent 为起点，把 Stage1 未纳入的 LoCoMo 原始记忆补进每条 parent graph，随后单独发起一轮新的 Stage3 图谱→对话→QA 生成、验证和 frontier 筛选。

用户的架构约束：

- API challenger 负责生成图谱、扩写对话和 QA（每题 4 个有区分度的选项）。
- 本地代码只负责 schema / 对齐 / 长度 / 泄漏 / 拒绝校验；不得从事实本地拼 QA、生成 gold 或替 API 补写对话。
- 图谱是跨轮持久记忆：下一轮必须继承、合并并增量扩展，不是每条样本独立的小图。
- 仅在本地工作；不要提交或推送 GitHub。

## 已确认的根因

旧 Stage1 的原始 seed adapter 是
[`prepare_locomo_current_method_seed.py`](scripts/current_method/prepare_locomo_current_method_seed.py)。

它只取 `locomo10.json` 的前 6 个原始样本：`conv-26, conv-30, conv-41, conv-42, conv-43, conv-44`，然后：

1. 只读取 `conversation` 和 `qa`；**未把** `event_summary`、`observation`、`session_summary` 用于建图。
2. 将对话按时间连续切为 6k–7.5k token chunk。21 个 chunk 中 16 个被发出，5 个因不足 6k token 被跳过。
3. 只使用 evidence 完全落在当前 chunk 内的 QA；跨 chunk evidence 被排除。
4. 直接把 QA 变为事件值：`Q: <question> A: <answer>`。

因此旧图谱不是从原始叙事中提炼出的长期记忆，而是 QA anchor 的子集；还发生了答案泄漏。预处理统计在
[`preprocess_summary.json`](base_restart_current_method_validation/preprocess_locomo_current_method/preprocess_summary.json)：16 个 seed chunk、875 个 raw-QA anchor。

后续单个 challenger 样本又只围绕少量选中 anchor / 新增事件建立 `graph_update`，没有持续携带原始 seed 的完整记忆，这解释了现有 parent graph 稀疏、与长对话脱节的问题。

## 应实施的补全方案

### 输入与映射

- Stage3 parent 输入：
  [`stage3_parent_v3.jsonl`](base_restart_current_method_validation/172_stage2_canonical_graph_rebuild_v3_semantic_audited_final/stage3_parent_v3.jsonl)，共 77 条。
- 原始来源：[`locomo10.json`](locomo10.json)。
- 通过每条 parent 的 provenance / lineage 中的
  `locomo_raw_sample{N}_chunk{MMM}` 映射回 raw 样本索引 `N`。
- 补全时读取 raw record 的 `conversation`、`event_summary`、`observation`；**严禁读取 `qa`（包括 adversarial_answer）**。

### 构图与合并

对每个 Stage3 parent：

1. 从原始记录提取有 provenance 的语义事实：事件、稳定属性/偏好、关系、计划、状态变化、时间锚点。
2. 使用原始 `D#: #` turn id（若来自 `event_summary`，保留 session/date 来源并尽可能关联对应 turns）。
3. 与现有 canonical parent graph 做实体归一、语义去重和时间去重；旧 graph event 不删除。
4. 为新增事实写稳定 ID、`origin=raw_backfill`、raw sample / session / turn provenance，并记录 merge/dedupe 统计。
5. 输出 enriched parent graph 和审计报告；无有效 provenance、含 QA 文本、无 evidence、或实体冲突时必须拒绝该条/该事件，不可静默猜测。

建议先生成一个独立的 enriched-parent JSONL 和每条 graph artifact，再由新 Stage3 run 读取；不要原地覆盖 canonical Stage2 文件。

## 生成前必须修的图谱防线

当前 API 图谱扩展器：
[`native_api_graph_dialogue_expansion.py`](scripts/current_method/native_api_graph_dialogue_expansion.py)。

已发现一个明确 bug：`normalize_current_event_plan` 会把多个不同的 access-scope 事件错误归一成相同的 `Nate / gaming room / private` 事件，导致 API 对话的事实冲突，随后对齐器拒绝。

修复要求：

- 保留可见性/权限语义（主体、客体、范围、条件、时态），不能用固定默认值覆盖。
- 在调用对话 API **前**，针对 normalized event plan 做 collision gate：若同一 canonical slot、同一时态下出现互斥 value，拒绝该计划并要求图谱 API 修复。
- 保留每个 graph artifact；重试必须复用该 artifact，而不是换一张图。

## 不要混用的现有运行

当前旧运行目录：
`base_restart_current_method_validation/216_stage3_per_target_inherited_graph_smoke_n10_v22`。

截至交接时无对应生成进程在运行；该目录有 11 条 raw 输出、2 条 validated 输出。它使用的是补全前的稀疏 parent graph，**不可与 enriched parent 的生成结果混入同一个训练/frontier 集**。保留它仅作失败案例和回归测试。

先前 v21 还存在“n=10 target 共用一个固定 artifact”的 target-mismatch 问题，也不要 resume。

## 关键路径

- 原始数据：`locomo10.json`
- Stage2 训练数据：
  `base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/train_mix/frontier_train.jsonl`
- 验收的 Stage2 solver：
  `base_restart_current_method_validation/accepted_models/solver_base_restart_stage2_merged`
- canonical Stage3 parent（77 条）：
  `base_restart_current_method_validation/172_stage2_canonical_graph_rebuild_v3_semantic_audited_final/stage3_parent_v3.jsonl`
- Stage3 canonical graph 构造：
  `scripts/current_method/materialize_stage3_canonical_graph.py`
- graph schema / 校验：`scripts/current_method/semantic_graph_v2.py`
- Stage2 canonical rebuild：
  `scripts/current_method/rebuild_stage2_canonical_graphs.py`
- Stage3 调度入口：`scripts/run_auto_graph_expansion_round.py`
- 图谱-对话-API QA 生成器：
  `scripts/current_method/native_api_graph_dialogue_expansion.py`
- 质量/前沿校验：`scripts/current_method/stage64_quality.py`

## 建议实施顺序

1. 新建独立脚本/模块：从 parent provenance 读取 raw record，输出 `raw_backfill_graph`；只读、可审计。
2. 实现 canonical graph merge，写出 `stage3_parent_v3_enriched.jsonl`；运行 QA-leak、provenance、dedupe、冲突和事件数量审计。
3. 修复 normalizer 的 visibility collapse，并加入 pre-dialogue collision gate；补充单元测试。
4. 用极小 smoke（建议 10 target）验证每 target 独立 artifact、graph persistence、对话覆盖率和 QA 四选项结构。
5. smoke 通过后，另开全量 Stage3 运行目录；只用 enriched parent，之后再做 solver K=5 frontier 筛选。

## 验收标准

- enriched graph 不含 `Q:`、`A:`、`answer`、`adversarial_answer` 或任何 raw QA 文本。
- 每个新增事件都可回溯至 raw sample + session/turn 或 `event_summary`/`observation` 来源。
- 图谱明显大于旧 parent，但不是无 provenance 的泛化偏好；重复事件被合并而非膨胀。
- 每个 target 有自己的持久 graph artifact；重试不改变图。
- 图谱→对话覆盖通过、无互斥事实、扩写满足长度门槛。
- QA 在对话生成之后由 API 提出且为四选一；本地只验证。
