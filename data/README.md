# 数据包说明

本目录保存当前 graph-conditioned self-play 主线的可复现数据。数据按
`LoCoMo 原始数据 → Stage1 → Stage2 → Stage2.5 memory backfill → 当前 Stage3`
组织。模型权重、checkpoint、运行缓存和完整历史工作区不在本数据包内。

## 快速定位

| 阶段 | 主要数据 | Graph 位置 | 数据含义 |
|---|---|---|---|
| 原始数据 | `locomo10.json` | 原始 observation / event summary | LoCoMo 多 session 长对话、人物、时间、memory annotations 和原始 QA |
| Stage1 | `base_restart_current_method_validation/01_mining500_train/train_mix/train_mix.jsonl` | 每行 `graph_update` | 第一轮 self-play 训练混合数据，包含扩充对话、QA、gold、solver eval 和 graph update |
| Stage2 | `base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/train_mix/train_mix.jsonl` | 每行 `graph_update` | 使用 Stage1 solver 挖掘并 repair/top-up 后的第二轮训练混合数据 |
| Stage2 parent | `base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/train_mix/frontier_train.jsonl` | 每行 `graph_update` | 当前 Stage3 parent 的直接 Stage2 来源；主要是 Stage1 solver 的边界样本 |
| Stage2.5 | `stage2_5_lineage_memory_graphs.jsonl` | 每行 `memory_graph` | 从 LoCoMo conversation、observation、event summary 构造的 lineage 级完整 memory graph，不使用原始 QA |
| 当前 Stage3 | `stage3_parent_v4_memory_backfilled.jsonl` | 每行 `semantic_parent_graph` | 当前正式 parent 输入：清理后的 Stage2 对话、canonical graph 和可见 LoCoMo memory backfill |
| Run234 | `run234_safe_pool_scene_canary_n5/frozen_truth_plans.json` | `core_graph` / truth plan / `safe_expansion_pool` | 从 Stage3 parent 冻结出的五个 canary 生成计划 |

## 数据流

```text
locomo10.json
  └─ 原始长对话、observation、event summary
       ↓
01_mining500_train/train_mix/train_mix.jsonl
  └─ Stage1 graph-conditioned self-play 数据；graph_update
       ↓
05_stage2_repair_topup_from_stage1/train_mix/
  ├─ train_mix.jsonl       Stage2 最终训练混合
  └─ frontier_train.jsonl  Stage3 parent 的直接来源
       ↓ canonical graph rebuild
stage2_5_lineage_memory_graphs.jsonl
  └─ 从 LoCoMo 回填的完整 lineage memory graph
       ↓ 按 evidence turn 可见性合并
stage3_parent_v4_memory_backfilled.jsonl
  └─ 当前 Stage3 数据；semantic_parent_graph
       ↓ Run234 冻结选择
run234_safe_pool_scene_canary_n5/frozen_truth_plans.json
```

## Stage1

目录：`base_restart_current_method_validation/01_mining500_train/train_mix/`

- `train_mix.jsonl`：最终训练混合，共 114 条。
- `frontier_train.jsonl`：84 条 boundary / near-miss 困难样本。
- `clean_replay.jsonl`：30 条 5/5 clean replay 样本。
- `frontier_heldout.jsonl`：不进入训练的 frontier 留出集。
- `sft_train.jsonl`：转换后的监督微调格式。
- `train_mix_summary.json`：数量、题型和难度分布。

每行把对话、QA 与 graph 放在一起。核心字段包括：

```text
dialogue
question / qa / options / gold_answer
solver_eval
graph_update.events
graph_update.state_changes
graph_update.cancellations
graph_update.invalidations
graph_update.restorations
graph_update.delayed_activations
graph_update.visibility_edges
graph_update.graph_snapshot_id / graph_snapshot_hash
```

## Stage2

最终版本位于：
`base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/train_mix/`

- `train_mix.jsonl`：repair/top-up 后的 Stage2 最终训练混合，共 108 条。
- `frontier_train.jsonl`：Stage3 parent 的直接上游数据。
- `clean_replay.jsonl`：Stage2 clean replay。
- `frontier_heldout.jsonl`：Stage2 留出 frontier。
- `near_miss_aux_pool.jsonl`：辅助 near-miss 数据，不等同于正式 train mix。

Stage2 graph 同样位于每行的 `graph_update`。Stage3 迁移时不会盲目继承旧
graph，而会执行 `stage2_canonical_graph_rebuild_v3`，保留有对话证据的 canonical
semantic events，排除 legacy QA 或不合格 graph events。

## Stage2.5 Memory Backfill

`stage2_5_lineage_memory_graphs.jsonl` 每行对应一个 LoCoMo lineage，graph 位于
`memory_graph`：

```text
source_lineage_key
source_dialogue
memory_graph.events
memory_graph.edges
```

构造输入仅包括 conversation、observation 和 event summary；不使用 LoCoMo 原始
QA，也不使用 session summary。合并到某个 parent 时，只有全部
`evidence_turn_ids` 都存在于该 parent 对话中的事件才可见。

## 当前 Stage3 Parent

`stage3_parent_v4_memory_backfilled.jsonl` 是当前实际使用的正式输入：

- 77 条 parent；
- 每条 graph 有 96–148 个事件，平均 118.247；
- graph events 总数 9,105；
- 新增 memory events 8,720；
- QA leakage 计数为 0。

每行主要字段：

```text
dialogue.turns              清理后的长对话
turns                       对话兼容副本
semantic_parent_graph       当前 parent graph
source_snapshot             Stage2 文件、行号、sample ID 和 hash
provenance                  LoCoMo seed 与生成来源
memory_backfill_audit       回填数量、证据完整性和泄漏审计
```

`semantic_parent_graph.events` 同时包含：

- `stage2_rebuild:*`：从 Stage2 重建并保留的 canonical semantic events；
- `*:OBS:*`：LoCoMo atomic observation memory events；
- `*:SUM:*`：LoCoMo event-summary memory events。

## Run234

`run234_safe_pool_scene_canary_n5/` 保存五个 isolated scene canary：

- `frozen_truth_plans.json`：冻结的 core graph、truth、parent provenance 和 safe pool；
- `api_scene_calls.jsonl`：场景调用记录与响应；
- `final_report.json`：最终验收报告；
- `machine_audit.json` / `manual_audit.json`：自动与人工审计；
- `manifest.json` / `state.json`：运行配置和状态。

Run234 不调用 graph extraction API，也不修改 parent graph；每个 scene 只从冻结的
core graph 与 `safe_expansion_pool` 生成对话。

## 完整性与限制

`DATA_MANIFEST.tsv` 给出仓库中每个数据文件的字节数、SHA-256 和相对路径，可用于
下载后的完整性验证。

本数据包不包含：

- 模型权重、adapter、optimizer 或 checkpoint；
- 完整 29G 历史实验工作区；
- 临时 API 缓存、日志和重复运行输出；
- 超过 GitHub 普通单文件限制的候选池。
