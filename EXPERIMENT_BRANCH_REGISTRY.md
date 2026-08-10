# Experiment Branch Registry

本文件是 `current_method_github` 的实验支线总登记。它解决的是“代码支线、运行目录和报告分散”问题，不替代各支线自己的冻结报告或 run manifest。

## 使用规则

每创建一个新支线或新 live probe，必须先在本表增加一行，至少记录：

```text
branch_id
base branch/module
scope
input/root
new run root
API policy and actual calls
freeze/preflight report
final report
status
next action
```

约定：

- 历史 run 目录只读；失败 run 不 resume、不覆盖、不删除。
- `API=0` 表示离线实现/preflight/shadow；不能写成“live 成功”。
- “代码存在”不等于“已进入 Git”；以 `git status` 和 commit 为准。
- 报告中的 SHA 是审计证据；不能用当前 working tree SHA 代替历史 freeze SHA。
- 新支线优先使用新 sibling module；主线和历史 sibling 不现场修改。

## 当前支线索引

| ID | 代码模块 | 主要输入/范围 | 运行与报告 | API | 状态 | 下一步 |
|---|---|---|---|---:|---|---|
| `WMS-V1` | `scripts/current_method/restart_n5_spell/world_memory_slim_v1` | `cc-train-episode-61555`，Slim Planner/Realizer/Memory 链路及 v1-v10 历史 | 运行目录：`/root/autodl-tmp/gxy-selfplay/restart-n5-runs/conversation_chronicles_episode_61555_world_memory_slim_fresh_e2e_v1` 至 `v10`；模块报告：`world_memory_slim_v1/reports/` | 多次历史 live | 历史冻结，只读 | 不覆盖；新实验另建 run root |
| `WMS-V11` | 同上模块的 v11 报告/runner | continuous expansion 设计、Planner probes、Phase A/B1 | `world_memory_slim_v1/reports/SLIM_CONTINUOUS_EXPANSION_V11_*`、`V11_*`；相关 run 以 `slim_continuous_expansion_v11_*` 开头 | 依各 run | 历史/离线与 canary 混合，按各报告判定 | 先查对应报告，不把 probe 当完整 canary |
| `MEMORAX-BASE` | `scripts/current_method/restart_n5_spell/simple_append_memorax_v1` | 简化扩充、Memorax 写入、batch/pilot | 运行目录以 `simple_append_memorax_v1_*` 开头；报告分布在 sibling `reports/` 和 run `reports/` | 依各 run | 已有多轮 pilot，不能混用状态 | 复用成功 run 的 frozen dialogue，避免重复扩充 |
| `QA-V19` | `simple_append_memorax_v1_fullr1_devdiv_qa_structural_v19_probe_v1` | `episode-146700` QA Writer、结构门、Solver replay | `/root/autodl-tmp/gxy-selfplay/restart-n5-runs/conversation_chronicles_episode_146700_qa_structural_v19*`；模块报告含 `STRUCTURAL_QA_V19_*` | 依各 run | 历史失败/诊断，全部只读 | 不从失败 Writer response 直接当成功 fixture |
| `QA-STRUCTURE-FIRST` | `simple_append_memorax_v1_qa_structure_first_probe_v1` | structure/evidence-first Writer 试验 | `/root/autodl-tmp/gxy-selfplay/restart-n5-runs/conversation_chronicles_episode_146700_qa_structure_first_probe_v1_live_1`；模块 `reports/STRUCTURE_FIRST_QA_PREFLIGHT.*` | 1 次历史 Writer | 历史冻结；用于对比 | 不修改，作为结构发现基线 |
| `QA-POSITION-FIRST` | `simple_append_memorax_v1_qa_position_first_probe_v1` | position-first Writer、distractor provenance、full dialogue | `/root/autodl-tmp/gxy-selfplay/restart-n5-runs/conversation_chronicles_episode_146700_qa_position_distractor_provenance_probe_v2_live_1`；模块 `POSITION_*_PREFLIGHT.*` | 1 次历史 Writer | 历史冻结 | 不运行 Solver 以外的隐式链路 |
| `QA-LOCAL-RECOGNIZER` | `simple_append_memorax_v1_qa_position_first_probe_v1/local_structure_*` | 5 Parent windows + 3 Child windows 的 Structure Recognizer | `/root/autodl-tmp/gxy-selfplay/restart-n5-runs/conversation_chronicles_episode_146700_qa_local_structure_recognizer_probe_v1_live_1`；[live report](../restart-n5-runs/conversation_chronicles_episode_146700_qa_local_structure_recognizer_probe_v1_live_1/reports/LOCAL_STRUCTURE_RECOGNIZER_LIVE_REPORT.md)；[semantic audit](../restart-n5-runs/conversation_chronicles_episode_146700_qa_local_structure_recognizer_probe_v1_live_1/reports/LOCAL_STRUCTURE_RECOGNIZER_SEMANTIC_AUDIT.md) | 8 次 Recognizer，0 retry | 已完成，结论是语义类型识别仍不稳定；run 与代码冻结 | 不进入 Local MCQ Writer，先解决结构识别/类型绑定 |
| `QA-FACT-FIRST` | `simple_append_memorax_v1_qa_fact_first_probe_v1` | frozen facts；same-subject 与 participant-connected 两次零 API rebundle | live facts：`conversation_chronicles_episode_146700_qa_fact_first_probe_v1_live_1`；participant run：`conversation_chronicles_episode_146700_qa_fact_first_participant_rebundle_v1_offline`；[participant audit](../restart-n5-runs/conversation_chronicles_episode_146700_qa_fact_first_participant_rebundle_v1_offline/PARTICIPANT_REBUNDLE_SEMANTIC_AUDIT.md) | live 8 次 Fact Extractor；两次 rebundle API=0 | participant connectivity 恢复 C02 和 58,389 个 cross-subject combinations，但独立主题仅 4→5 | Case B；`ready_for_reverse_question_writer=false`；停止继续修 bundler，转回检查 Expansion 的跨阶段持续事实链 |

## 最近一次 QA 支线结论

`QA-FACT-FIRST` 最终 participant-connected 诊断：

```text
43 frozen grounded facts
81,652 participant-connected combinations
58,389 new cross-subject combinations
60 materialized bundles
C02 coverage: 0 -> 8
12 semantic-usable variants
5 independent themes
Fact Extractor / Writer / Solver / Expansion / Memorax calls = 0
```

这条支线已经排除了 `earliest 12` 和 `same subject` 两个明显机械限制。
跨人物连接确实恢复了 Specialist D/Boss 相关候选，但多数新增组合只是人物有交集、剧情并不连续；独立主题只从约 4 增到约 5。按冻结停止条件，不再引入第三种 bundler heuristic，也不启动 Reverse Writer。

## 代码与 Git 状态

当前工作树仍包含大量历史实验的 `modified` 和 `untracked` 文件。最近 Git HEAD 是旧 checkpoint；因此：

- 本登记文件和各支线文件目前是工作树记录，不代表已提交到 Git。
- 需要回滚或复现时，应使用支线目录中的 `BASELINE.md`、`*_PREFLIGHT.json`、`run_freeze_manifest.json`、`POST_RUN_INTEGRITY.json` 和 run 目录 manifest。
- 不要使用 `git checkout`、`reset`、`clean` 来“整理支线”，这些会破坏未提交实验现场。

## 新支线登记模板

复制以下段落追加到“当前支线索引”：

```text
| `NEW-ID` | `scripts/.../new_sibling` | frozen input/root | `/root/.../new_run`；preflight：`...`；final：`...` | 0 或实际次数 | planned / preflight / live / failed / complete / frozen | 明确唯一下一步 |
```

新 run 必须同时保存：

```text
run_freeze_manifest.json
request ledger / raw responses
preflight report
summary report
post-run integrity report
```

## 状态含义

| 状态 | 含义 |
|---|---|
| `planned` | 只有方案，没有执行 |
| `preflight` | 零 API 检查中或已通过 |
| `live` | 真实请求正在进行；代码已冻结 |
| `failed` | 失败现场已保存，不得自动重跑 |
| `complete` | 本轮链路完成，按报告判定 readiness |
| `frozen` | 历史审计对象，只读 |

最后更新：2026-08-09。更新本文件不代表启动任何 API 或改变历史 run。
