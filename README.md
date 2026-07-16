# Current Method Self-Play 代码包

这是“当前方法”的干净 GitHub 代码包，包含从 parent pool 生成 self-play 样本、API challenger 扩展对话、Stage3 API QA 多样化、质量验证、可靠 frontier 累积、train mix 构建、solver 训练与评测所需的主线代码。

注意：这里整理的是**当前方法主线代码**，不是整个工作区的 r1-r5 历史归档。模型权重、运行缓存、API key 和 149G 工作区内容都没有放进来。

`data/` 目录已经加入当前方法实验线路的可复现数据子集，包括原始 `locomo10.json`、切分后的 seed / parent rows、stage1、stage2、stage2 repair/topup、Stage64 API self-play 与 Stage64 题目类型多样化 probe 的关键 JSON/JSONL、报告和状态文件。数据目录不是完整 29G 工作区镜像；具体包含和排除内容见 `data/README.md` 与 `data/DATA_MANIFEST.tsv`。

## 当前方法实验流程

```text
1. 用本地 Qwen3-4B 从 Stage2 evidence turns 重建 canonical graph
   -> 2. fail-closed 审计并产出 stage3_parent_v3
   -> 3. 构建 challenger policy / curriculum
   -> 4. API 生成 graph event plan
   -> 5. API 生成 appended dialogue + qa_proposal
   -> 6. 本地验证 graph/dialogue/QA/长度/泄漏
   -> 7. 输出 clean_samples / reject logs / summary
   -> 8. solver eval + frontier selection
   -> 9. 构建 train mix
   -> 10. 可选：LoRA 训练、merge、vLLM 评测
```

Stage3 继续 accepted Stage2 solver 的 self-play。历史 CLI 名 `--qa-operation-answer-type-diversification` 和部分 `stage64_*` artifact 保留兼容，但新实验结论应称为 Stage3。当前 schema 仍是 `single_choice_mcq`，正式 target 在 API 前固定 `qa_operation`、合法 `answer_type`、reasoning subtype/template、hop count 和 hard-negative strategy，并在 solver feedback hardening 后保持这些字段不变。

完整的 Stage2 -> Stage3 执行顺序、门禁和命令见 [`docs/STAGE3_SELFPLAY_ROUND.md`](docs/STAGE3_SELFPLAY_ROUND.md)。

Stage3 graph payload schema 仍是 `semantic_graph_v2`，但 cache key / gate namespace 已升级为 `semantic_graph_v16`。v16 保留 v15 的 reasoning-ledger、temporal anti-shortcut、formal-evidence 和 neutral/filler 隔离，并新增 `visibility` / `relation_lookup` 的固定八事件 role layout、字段等式和 parent-collision correction；retry 只引用事件 ID 和字段关系，不重复 parent 原文。历史 v15 run173 为 5/7，失败集中在这两类。v16 定向验证中，visibility 为 1/1，relation_lookup 为 1/1（后者使用一次 validator-guided correction）；这只是定向证据，正式 dialogue/QA 前仍需新目录中的 fresh 7-operation v16 graph-only gate。通过后再做小批量 mining，生成后立即 shuffle/balance，用固定 accepted Stage2 solver 对全部 cyclic positions 做 search K5；再对 easy5/hard0/position-sensitive 做一次冻结 graph/dialogue/gold 的 QA-only feedback rewrite，并以独立 seed 做 final cyclic K5。

## 目录结构

```text
.
  README.md
  MANIFEST.json
  requirements.txt
  configs/
    config.example.json
  docs/
    CODE_MAP.md
    CROSS_STAGE_EVAL_PANEL.md
    STAGE3_SELFPLAY_ROUND.md
    STAGE64_RELIABLE_FRONTIER_MINING.md
    GITHUB.md
  scripts/
    run_auto_graph_expansion_round.py
    generate_r2_llm_graph_expansion_challenger.py
    generate_v13_pilot.py
    run_graph_expansion_challenger_v1.py
    eval_v13_pilot.py
    eval_v13_pilot_vllm.py
    run_round_solver_training_pipeline.py
    build_r2_frontier_plus_clean_replay_train.py
    merge_custom_lora_adapter.py
    old_method/
      run_old_improved_gate200.py
      run_old_improved_gate400_sample0_5.py
    current_method/
      *.py
      stage64_quality.py
  train_qwen3_lora_sft.py
```

## 快速开始

先复制示例配置：

```bash
cp configs/config.example.json config.json
```

然后填写：

- `api.api_key`
- `api.base_url`
- `api.model`

正式生成前，先用本地 Qwen3-4B 重建并严格审计 Stage2 canonical graph。这个步骤不调用付费 API；模型请求只包含 evidence turn ID、speaker 和 text，不包含旧 graph 的 entity/slot/value/status/time，也不包含旧 QA、options 或 gold。下面的实际 artifact 路径命令从主工作区 `/root/autodl-tmp/gxy-selfplay` 运行：

```bash
python scripts/current_method/rebuild_stage2_canonical_graphs.py \
  --input base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/train_mix/frontier_train.jsonl \
  --output-dir base_restart_current_method_validation/172_stage2_canonical_graph_rebuild_v3_semantic_audited_final \
  --model /root/autodl-tmp/Qwen3-4B \
  --max-rows 0 \
  --minimum-events-per-parent 2 \
  --semantic-exclusion-jsonl data/stage2_semantic_audit_exclusions_v1.jsonl \
  --forbidden-jsonl base_restart_current_method_validation/01_mining500_train/train_mix/frontier_heldout.jsonl \
  --forbidden-jsonl base_restart_current_method_validation/04_stage2_train_from_stage1/train_mix/frontier_heldout.jsonl \
  --forbidden-jsonl base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/train_mix/frontier_heldout.jsonl

python scripts/current_method/audit_stage2_canonical_rebuild.py \
  --source base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/train_mix/frontier_train.jsonl \
  --rebuild-dir base_restart_current_method_validation/172_stage2_canonical_graph_rebuild_v3_semantic_audited_final \
  --forbidden-jsonl base_restart_current_method_validation/01_mining500_train/train_mix/frontier_heldout.jsonl \
  --forbidden-jsonl base_restart_current_method_validation/04_stage2_train_from_stage1/train_mix/frontier_heldout.jsonl \
  --forbidden-jsonl base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/train_mix/frontier_heldout.jsonl \
  --output base_restart_current_method_validation/172_stage2_canonical_graph_rebuild_v3_semantic_audited_final/canonical_rebuild_audit.json
```

正式 artifact schema 是 `stage2_canonical_graph_rebuild_v3`。当前语义审计后产物包含 `77` 个 canonical graphs、`385` 个 canonical events 和 `77` 个 `stage3_parent_v3`；171 个 exclusions 中包含独立语义审计隔离的 18 个 events，eligible event rejection、row-atomic rejection 和 heldout exact overlap 均为 `0`。原始 78 行中的 1 行没有任何 clean linked evidence，因此整行隔离，未用无链接对话补事件；`fallback_event_count=0`。重建命令默认禁用 fallback，正式运行不要添加兼容参数 `--allow-dialogue-fallback`。保留的长 structured parent atoms 及其 `depends_on` 对 challenger 仍然可读，省略的只是 parent dialogue utterance；终态误链规则也已收紧为严格 entity/fact-word 匹配。`rebuild_summary.json` 仍明确记录 `training_authorized=false`。

历史目录 `161_stage2_canonical_graph_rebuild_v2_final` 虽通过当时的结构审计，但后续语义复审发现 11 个坏 events，因此不可作为 Stage3 parent，也不可用于 graph-only smoke。

然后才运行 7 条 graph-only API smoke：

```bash
python scripts/current_method/stage3_graph_only_smoke.py \
  --input base_restart_current_method_validation/172_stage2_canonical_graph_rebuild_v3_semantic_audited_final/stage3_parent_v3.jsonl \
  --output-dir runs/stage3_graph_smoke \
  --n 7 \
  --max-attempts-per-sample 2 \
  --minimum-pass-rate 0.70 \
  --graph-semantic-retries 1 \
  --confirm-api-calls
```

The v16 graph-only gate freezes the full operation/reasoning target before
each request and uses one validator-guided correction. Historical v14 `7/7`
and v15 run173 `5/7` results are not v16 full-gate evidence. The v16 targeted
visibility and relation checks both pass, but failed raw graphs remain outside
mining; only rows from a newly passed seven-operation v16 graph gate may enter
the next stage.

`--parent-dir` 需要指向本次运行要使用的 parent pool JSONL，例如 mined frontier pool。包内只有 `data/README.md` 列出的精选复现数据，不保证包含示例命令所需的完整 parent pool。

开启兼容 flag 后，主入口不会再走普通“当前位置 K5 -> frontier”路径。生成结束后会自动运行 semantic gate、全部 cyclic positions K5 和解码健康门禁，产物写入 `<run-root>/stage64_mining/`。单批自动训练被硬阻止；先跨批累积至少 300 条 robust frontier，再统一做 `lineage_group` split，并在 frontier 过滤后从已评估位置重物化均衡 gold。

运行回归测试时要显式覆盖两组目录：

```bash
python -m unittest discover -s tests -p 'test_*.py'
python -m unittest discover -s scripts/current_method -p 'test_*.py'
```

## 核心代码说明

### 总入口

| 文件 | 实验流程位置 | 作用 |
| --- | --- | --- |
| `scripts/run_auto_graph_expansion_round.py` | 全流程入口 | 当前方法主入口。负责读取 parent rows、构建 policy、调用 native/API generator、写报告，并可选触发 solver eval、frontier selection 和训练流程。 |
| `scripts/current_method/native_challenger_generation.py` | 生成入口分发 | native current-method generator 的分发层。`graph_conditioned_api_dialogue` 模式会转到 `native_api_graph_dialogue_expansion.py`。 |
| `scripts/current_method/native_api_graph_dialogue_expansion.py` | challenger 生成核心 | 当前 API self-play 的核心文件。包含 interleaved target ledger、reasoning signature、graph/event-plan API prompt、validator-guided semantic retry、`semantic_graph_v16` cache isolation、visibility/relation fixed-role contract、formal evidence exact-claim 与 neutral graph-fact isolation、dialogue/API QA、temporal anti-shortcut、alignment、长度和泄漏验证。 |

### API 调用与历史兼容依赖

| 文件 | 实验流程位置 | 作用 |
| --- | --- | --- |
| `scripts/generate_r2_llm_graph_expansion_challenger.py` | API 底层兼容 | 复用旧 self-play 里的 API 配置、JSON 解析、`curl` POST helper、默认 challenger model/base_url。当前方法 API 调用会复用这里的 `post_json_with_curl`。 |
| `scripts/generate_v13_pilot.py` | 兼容工具 | 历史 v13 pilot 的兼容子集，提供稳定 shuffle、quota 计算、leak pattern、chat record 格式化等函数。 |
| `scripts/run_graph_expansion_challenger_v1.py` | 兼容工具 | graph expansion v1 的兼容子集，主要给旧 generator 提供常量和轻量工具。 |

### Policy、课程和难度控制

| 文件 | 实验流程位置 | 作用 |
| --- | --- | --- |
| `scripts/current_method/challenger_policy_update.py` | policy 生成/更新 | 调用 API 或规则逻辑生成 challenger policy，控制问题类型、攻击技能和分布。 |
| `scripts/current_method/challenger_policy_autodesign.py` | policy 自动设计 | 根据已有结果自动设计/调整 challenger policy。 |
| `scripts/current_method/difficulty_controller.py` | closed-loop 难度控制 | 根据上一轮 solver 表现调整下一轮生成分布，避免题型过度集中或 frontier collapse。 |
| `scripts/current_method/proposer_bandit.py` | bandit proposer | 用 bandit 方式调度候选 cell、难度和分布。 |
| `scripts/current_method/run_adaptive_smoke_hardening_loop.py` | 自适应 smoke loop | 自动跑小批量 smoke、分析失败原因、调整 hardening 参数。 |

### Stage64 / 题目类型多样化

| 文件 | 实验流程位置 | 作用 |
| --- | --- | --- |
| `scripts/current_method/native_api_graph_dialogue_expansion.py` | Stage64 API 主实现 | `--qa-operation-answer-type-diversification` 的核心实现。负责把每个样本分配到不同 `qa_operation` / `answer_type`，要求 API 生成对应题型的最终 `qa_proposal`，并做本地验证。 |
| `scripts/current_method/stage64_quality.py` | Stage64 语义门禁 | 在 graph cache 写入/复用前检查 semantic slot、问题式 value 和 operation-specific graph prerequisites；在 QA 形成后统一检查 answer surface、问题/答案类型、operation/subtype、multi-anchor reasoning 和同实体/同 slot hard negative。 |
| `scripts/current_method/semantic_graph_v2.py` | Stage3 graph schema | 对 API graph delta 做不修复、不补齐的 canonical schema 与 operation-specific 验证；payload schema 名与 `semantic_graph_v16` cache/gate namespace 有意分离。 |
| `scripts/current_method/stage3_graph_only_smoke.py` | Stage3 graph-only gate | 在生成 dialogue/QA 和加载 solver 前，用小批付费 API 调用验证 graph 通过率，支持 validator-guided correction，并保存 raw reject。 |
| `scripts/current_method/stage3_operation_controller.py` | Stage3 operation controller | 直接消费 cyclic K5 aggregate，更新下一批 `qa_operation|answer_type` 配额，不额外重跑 solver。 |
| `scripts/current_method/stage3_solver_feedback_mining.py` | Stage3 solver-feedback QA mining | 分类四类 cyclic K5 结果，冻结 graph/dialogue/gold/reasoning target，导出 harden/soften/position-repair 请求和诊断/retention manifest。 |
| `scripts/current_method/stage3_batch_closure.py` | Stage3 batch closure | 可恢复地执行一次 QA feedback rewrite、候选合并、reshuffle 和独立 final cyclic K5；不会自动训练。 |
| `scripts/current_method/qa_operation_answer_type_anchor_diversification_probe.py` | Stage64 probe / 分析 | 用于 anchor-first QA operation / answer type diversification 的 probe、诊断和数据整理，主要分析题型覆盖、anchor 使用和答案类型分布。 |

### Seed、数据准备和审计

| 文件 | 实验流程位置 | 作用 |
| --- | --- | --- |
| `scripts/current_method/prepare_locomo_current_method_seed.py` | seed 准备 | 从 LoCoMo / parent data 准备 current-method seed rows，并记录 provenance。 |
| `scripts/current_method/seed_audit.py` | seed 审计 | 检查 train/dev/heldout provenance，防止 forbidden seed 或 split 泄漏。 |
| `scripts/current_method/rebuild_stage2_canonical_graphs.py` | Stage2 canonical graph 重建 | 用本地 Qwen3-4B 从 evidence-only 请求重新抽取事件；旧 graph/QA 语义标签不进入模型请求，旧事件逐个重抽取或隔离，支持独立语义审计 exclusion manifest，默认禁用 dialogue fallback，输出 `stage2_canonical_graph_rebuild_v3` 与正式 `stage3_parent_v3.jsonl`。 |
| `scripts/current_method/audit_stage2_canonical_rebuild.py` | canonical rebuild 严格审计 | 只读、fail-closed 地核对源事件处置、请求字段、canonical graph、parent 一致性和 heldout exact overlap；不修复或准入数据。 |
| `scripts/current_method/build_stage3_parent_v2.py` | legacy parent 迁移兼容 | 仅保留给旧产物迁移和诊断；它不能重建被污染的 Stage2 graph，不是正式 Stage3 parent 入口。 |
| `scripts/current_method/api_challenger_generation.py` | 基础生成/验证适配 | 当前方法复用的基础 adapter，包含 JSONL 读写、样本验证、solver eval 接口、clean/valid 判断等公共逻辑。 |

### Frontier、报告和诊断

| 文件 | 实验流程位置 | 作用 |
| --- | --- | --- |
| `scripts/current_method/frontier_selection.py` | frontier selection | 根据 solver 评测结果筛 frontier、strict frontier、boundary frontier 等。 |
| `scripts/current_method/report_builder.py` | 报告生成 | 汇总生成质量、frontier yield、题型分布、solver 表现、长度统计等。 |
| `scripts/current_method/cross_stage_eval_panel.py` | 跨阶段能力验证 | 汇总或复跑 Stage1/Stage2/Stage64 heldout 上的 solver 对比，验证 self-play 是否真的逐阶段提升。 |
| `scripts/current_method/recalibrate_stage64_shuffled_frontier.py` | Stage64 shuffle 后重校准 | 清空旧 solver/frontier 字段，用固定 Stage2 solver 重评估 shuffled clean samples，并重新筛 frontier / hard0 / easy5 / grouped heldout。 |
| `scripts/current_method/shuffle_stage64_options.py` | 可复现 option balance | 先拒绝重复/歧义选项，再在 option arity × attack skill 内确定性平衡 gold label；绝不携带旧 `solver_eval`。 |
| `scripts/current_method/external_generalization_report.py` | 外部评估审计 | 按 `record_id` 做配对统计、exact McNemar 和 gold-label 分层，避免把小样本 bucket 差异写成因果收益。 |
| `scripts/current_method/accumulate_stage64_reliable_frontier.py` | 跨批 frontier 累积 | 只接受健康、语义合格、全部位置稳定的 frontier；达到默认 300 条后做 lineage-group split，再从已评估 cyclic positions 重物化 balanced train/heldout。 |
| `scripts/current_method/write_base_restart_validation_summary.py` | 验证汇总 | 为 base restart / current-method validation 运行写总览 summary。 |
| `scripts/current_method/diagnostics_context_ablation.py` | 诊断实验 | 做 context ablation，分析 solver 对 parent context / evidence window 的依赖。 |
| `scripts/current_method/eval_api_14b_samples.py` | API solver 评测 | 用 API 模型评测样本，可用于快速 sanity check 或外部 judge。 |

### Train mix / mining 数据构建

| 文件 | 实验流程位置 | 作用 |
| --- | --- | --- |
| `scripts/current_method/build_base_restart_train_mix.py` | train mix 构建 | base restart 早期 train mix 构建脚本。 |
| `scripts/current_method/build_base_restart_stage2_train_mix.py` | stage2 train mix 构建 | 当前方法 stage2 train mix 构建，支持 frontier、clean replay、top-up 等组合。 |
| `scripts/current_method/build_base_restart_stage2_repair_train_mix.py` | repair train mix 构建 | 用于 stage2 repair/top-up 的 train mix 构建。 |
| `scripts/current_method/build_stage3_train_mix.py` | Stage3 train mix | 强制消费 training-ready manifest，正式 heldout 只做排除，选择 Stage1/Stage2 replay，派生 train-only validation 并写 SHA256 provenance。 |
| `scripts/current_method/build_chained_parent_expansion_probe.py` | chained parent probe | 构建 chained parent expansion 相关 probe。 |
| `scripts/current_method/build_chained_parent_v2_from_child_v2.py` | chained parent v2 | 从 child v2 构建 parent v2，用于链式 parent/child 数据实验。 |
| `scripts/build_r2_frontier_plus_clean_replay_train.py` | 训练数据构建 wrapper | 根据 frontier + clean replay 构建 solver SFT train JSONL。当前训练 wrapper 会调用它。 |

### 可选训练、merge 和评测

| 文件 | 实验流程位置 | 作用 |
| --- | --- | --- |
| `scripts/run_round_solver_training_pipeline.py` | 训练总入口 | 支持 audited prepared train file；校验 summary/SHA256 和 checkpoint base，训练新 LoRA、merge adapter、运行 smoke/full eval。 |
| `train_qwen3_lora_sft.py` | LoRA SFT 训练 | 基础 Qwen3 LoRA SFT 训练脚本。 |
| `scripts/merge_custom_lora_adapter.py` | adapter merge | 将 LoRA adapter merge 到 base model。 |
| `scripts/eval_v13_pilot.py` | MCQ 评测工具 | 非 vLLM / 通用评测逻辑和解析工具。 |
| `scripts/eval_v13_pilot_vllm.py` | vLLM 评测 | 使用 vLLM 跑 solver MCQ eval，适合批量评测 frontier 和 clean retention。 |

## Stage64 API 模式

开启参数：

```bash
--qa-operation-answer-type-diversification
```

### Stage64 是什么

Stage64 的贡献是 **single-choice MCQ 内部的语义题型多样化**。它把当前方法从“主要生成一种宽泛的单选状态判断题”，推进到“按 QA operation 和 answer type 明确控制不同推理操作”。它目前不等同于 multi-select / ordering 等输出 schema 多样化。

它重点解决的是：同样是长对话状态追踪，训练数据不应该总是问“哪个选项仍然为真”，而应该覆盖更多真实推理操作，例如：

- 查实体关系：某人做了什么、拥有什么、在哪里、计划什么；
- 查时间更新：计划/状态最终落在哪个时间点；
- 做桥接推理：需要连接两个事件或两段对话才能回答；
- 做比较：比较两个对象、状态或事件的差异；
- 查可见性/权限：谁知道、谁能看到、信息对谁有效；
- 做计数：有多少人、多少对象、多少事件满足条件；
- 查因果/结果：某个原因导致了什么结果，或者某个行动带来什么后果。

因此 Stage64 的核心不是“多加一层验证”，而是**把题目空间从单一模板扩展成 operation-aware / answer-type-aware 的题型空间**。本地验证只是为了保证 API 生成的新题型真的合格。

换句话说：

```text
普通当前方法：主要围绕当前状态判断构造题目
Stage64 当前方法：显式控制 qa_operation + answer_type，生成更多类型的题目

旧本地 Stage64：本地根据 graph anchor 机械派生不同题型 QA
当前 API Stage64：API 生成这些不同题型的最终 QA，本地验证题型、答案类型和图证据是否合格
```

这样做的原因是：self-play 里的 challenger 应该来自更强模型，产生更自然、更有攻击性的多题型问题；但 API 生成又天然不稳定，所以需要本地质量门来保证题型、答案类型和 graph anchor 对齐。

### 相比旧逻辑做了什么改变

1. **题目类型空间改变**

   这是 Stage64 最重要的改变。它不再只依赖普通 single-choice 状态判断，而是显式引入 `qa_operation` 作为题目类型控制轴，并引入 `answer_type` 作为答案类型控制轴。

   目标是让生成数据覆盖更多推理操作，而不是让 solver 只学习一种“读到最后选仍然为真的选项”的模式。

2. **新增 QA operation / answer type 控制**

   Stage64 会给每个样本分配目标 `qa_operation` 和偏好的 `answer_type`，例如：

   - `temporal_state_update`
   - `bridge`
   - `comparison`
   - `visibility`
   - `count`
   - `cause_consequence`
   - `relation_lookup`

   同时控制答案类型，例如：

   - `ObjectOrCommonNoun`
   - `DateOrTime`
   - `Number`
   - `Event`
   - `LocationOrScope`
   - `Person`
   - `GroupOrOrg`
   - `ProperNounOrNamedEntity`
   - `YesNo`

   本地代码会检查 API 返回的 `qa_operation`、`answer_type`、options 和 gold 是否一致，并验证选项表面确实符合声明的 answer type。

3. **最终 QA 来源改变**

   旧逻辑里，Stage64 会在本地根据 graph anchor 派生 question/options/gold。当前逻辑要求 API 直接返回最终 `qa_proposal`，本地不再重建最终 QA。

4. **新增 API `qa_proposal` schema**

   API 必须返回完整结构，而不是只返回自然语言问题。必需字段包括：

   - `qa_operation`
   - `qa_subtype`
   - `answer_type`
   - `answer_format`
   - `question` / `question_text`
   - `options`
   - `gold`
   - `query_plan`
   - `anchor_refs`

5. **新增本地题型强校验**

   本地验证不是为了替 API 写题，而是为了确认 API 写出来的题真的属于目标题型。它会拒绝以下情况：

   - `gold` 不是选项 id；
   - gold option 不唯一；
   - 规范化后出现完全重复选项，或 gold 文本同时对应多个 label；
   - option 的 `answer_type` 不一致，或表面答案不是对应类型，例如 Number 选项仍是一整句问题；
   - `count` 没有返回 Number；
   - `qa_operation` 不在允许列表；
   - 问题语义和 `qa_operation` 明显错配；
   - `anchor_refs` / `source_event_instance_id` 无法对齐 graph；
   - question/options 里泄漏内部词；
   - 题目太泛化，例如只问 “Which event is ...”。

6. **新增 solver-visible artifact 拦截**

   Stage64 会拦截 API 容易生成的内部状态语言，例如：

   - `raw_qa`
   - `anchor`
   - `event_instance`
   - `graph_anchor`
   - `latest confirmed detail`
   - `confirmed detail`
   - `final active state`
   - `settled as true`
   - `Q:` / `A:` wrapper

   这些词如果出现在 solver 可见的问题、选项或答案里，会导致样本被 reject。

7. **新增长度增长守卫**

   当前 Stage64 继承主流程的 length guard，要求扩展后的 dialogue 相对 parent 有足够 token 增长，例如当前 smoke test 使用 `--min-length-delta-tokens 4000`。

8. **新增 reject 日志和题型分布统计**

   每轮运行会记录：

   - API 调用次数；
   - reject 原因；
   - `qa_operation` 分布；
   - `answer_type` 分布；
   - schema pass 计数；
   - timeout 计数；
   - 长度增长统计。

   这些信息写在 `generation_summary.json` 和 `graph_conditioned_api_dialogue_rejects.jsonl` 里。

9. **新增 QA-only repair，减少 API token 浪费**

   Stage64 API 生成里最贵的是长对话扩展。如果 appended dialogue 已经通过 graph/dialogue alignment 和长度验证，但最终 `qa_proposal` 只是因为 schema、`gold`、`answer_type`、`qa_operation`、`query_plan` 或 `anchor_refs` 的小错误被拒绝，直接整条重跑会浪费大量 token。

   当前代码增加了一次短 prompt 的 QA-only repair：

   ```text
   1. API 生成 graph event plan + appended dialogue + qa_proposal
   2. 本地验证 graph/dialogue alignment、长度和 Stage64 QA
   3. 如果 dialogue 已合格，但 Stage64 QA validator 报 qa_proposal 类错误
   4. 只把失败原因、event candidates、已验证 appended dialogue 和旧 qa_proposal 发回 API
   5. API 只重写 qa_proposal，不重写 dialogue
   6. 修复后的 qa_proposal 仍必须重新通过完整 Stage64 validator
   ```

   这个机制的作用是降低 API 失败样本带来的 token 浪费。它不会把 Stage64 改回本地模板派生：最终 question/options/gold 仍然必须来自 API，本地只是把失败原因反馈给 API，并在修复后继续严格验证。

### Stage64 没有做什么

Stage64 当前**不做本地最终 QA 派生**。也就是说，本地代码不会自己根据 graph anchor 重新生成 question/options/gold。它只验证 API 给出的 `qa_proposal` 是否满足目标题目类型、答案类型和 graph evidence 约束。

这点很重要：如果本地重新派生最终 QA，就会退回模板化 challenger，不再是我们想要的 API self-play。

开启后，API challenger 必须生成最终 `qa_proposal`，包括：

- `qa_operation`
- `qa_subtype`
- `answer_type`
- `answer_format`
- `question` / `question_text`
- `options`
- `gold`
- `query_plan`
- `anchor_refs`

本地代码只负责：

- schema 验证；
- graph reference / anchor reference 校验；
- `qa_operation` 和 `answer_type` 匹配检查；
- solver-visible artifact 拦截；
- 长度增长检查；
- graph/dialogue alignment 检查；
- reject / retry / 落盘。

本地不会在 Stage64 模式里重新派生最终 QA。这样可以保证 challenger 由更强 API 模型生成，同时用本地质量门兜底。

## 跨阶段能力增长评估

为了证明 self-play 不是只在训练集上变好，需要用跨阶段 heldout 面板验证：

```text
Stage1 solver 在 Stage1 heldout frontier 上 > base
Stage2 solver 在 Stage2 heldout frontier 上 > Stage1
Stage64 solver 在 Stage64 heldout frontier 上 > Stage2
```

对应脚本：

```bash
python scripts/current_method/cross_stage_eval_panel.py \
  --panel-dir base_restart_current_method_validation/70_cross_stage_eval_panel
```

默认只汇总已有评测 JSON 并写 `cross_stage_eval_report.md`，不会重新跑 GPU。需要重跑时再加 `--run-original` 或 `--run-chat`。

当前配对面板结论是：Stage1 在小规模 validation heldout 上有正向证据；Stage2 只是 `16/30 -> 17/30`，双侧 exact McNemar `p=1.0`，不能判定阶段收益成立；旧 Stage64 heldout 是 legacy，不能作为能力增长证据。详细解释见 `docs/CROSS_STAGE_EVAL_PANEL.md`。

Stage64 shuffle 后重校准入口：

```bash
python scripts/current_method/recalibrate_stage64_shuffled_frontier.py \
  --input /path/to/new_stage64/shuffled/clean_samples.jsonl \
  --output-dir /path/to/new_stage64/recalibrated \
  --solver-model /path/to/accepted_models/solver_base_restart_stage2_merged \
  --accepted-stage2-solver-model /path/to/accepted_models/solver_base_restart_stage2_merged \
  --temperature 1.0 \
  --top-p 0.9 \
  --permutation-mode cyclic \
  --group-level lineage_group \
  --max-model-len 32768
```

这一步不会重新调用 API。它会先拒绝 exact duplicate / 多标签同答案，并执行 answer type、operation/subtype、semantic slot 和 hard-negative 语义门禁；随后确认 resolved checkpoint 是 accepted base-restart Stage2，对每道题的全部 cyclic option positions 分别评估，并只保留每个排列都处于 `1/5-4/5` 的 robust frontier。完整流程见 `docs/STAGE64_RELIABLE_FRONTIER_MINING.md`。

截至 2026-07-11 的最终审计结论：旧 492 条中，结构门禁可留下 429 条，但完整语义门禁留下 0 条；此前对这 429 条做的全排列诊断也得到 0 条 robust frontier、368 条 robust easy5、5 条 robust hard0 和 56 条 position-sensitive。旧 408、shuffle 后 325、单排列重校准 15 条以及 12 条 train 都不能用于训练或论文结论。权威审计目录是 `base_restart_current_method_validation/76_stage64_final_semantic_quality_audit`，全排列诊断目录是 `base_restart_current_method_validation/73_stage64_permutation_robust_recalibration`。

新 Stage3 run 默认自动执行 search shuffle 和 cyclic K5，关键路径是 `<run-root>/stage64_mining/shuffled` 与 `recalibrated`。完整 QA feedback closure 是显式 opt-in：传 `--stage3-batch-closure execute` 后，正式累计输入改为 `<run-root>/stage64_mining/batch_closure/final_reliable_batch`。`--skip-eval` 与 closure 不兼容，也不会产生 frontier。

## 质量门

生成器会拒绝以下样本：

- 出现 solver 可见的内部词，例如 `raw_qa`、`anchor`、`event_instance`、`latest confirmed detail`、`settled as true`。
- API 返回的 QA schema 不合法。
- `qa_operation` 和 `answer_type` 不匹配。
- `qa_operation` 与实际问题或 `qa_subtype` 不匹配。
- `count` 没有返回 Number 类型。
- Number/DateOrTime/YesNo 等选项表面不符合声明类型，或 option 只是复制了一遍问题。
- gold option id 无效或不唯一。
- 规范化后选项文本完全重复，或 gold 文本映射到多个 label。
- option 引用 `raw_qa_*` 泛化槽、使用无关事实作 distractor，或 bridge/comparison/count/cause 缺少多锚点推理。
- graph reference 缺失或无法对齐。
- graph/dialogue alignment 失败。
- 长度增长低于配置目标。

当前原则是：

```text
API 负责生成候选和语言多样性
本地代码负责底线验证和质量控制
```

也就是说，本地规则是护栏，不是模板机。它拦截泄漏、错配、重复和不合格样本，但不强行固定问题措辞。

## 重要输出文件

每次运行会写入 `--output-dir`：

- `clean_samples.jsonl`
- `valid_samples.jsonl`
- `samples.jsonl`
- `samples.partial.jsonl`
- `generation_summary.json`
- `graph_conditioned_api_dialogue_raw.jsonl`
- `graph_conditioned_api_dialogue_rejects.jsonl`
- `graph_conditioned_api_dialogue_summary.json`

这些都是运行产物，不要提交到 GitHub。

## 依赖

仅生成数据时，大部分逻辑只依赖 Python 标准库；如果要做 tokenizer 级长度统计，建议安装 `transformers`。

训练和 vLLM 评测需要更重的依赖，见 `requirements.txt`。

## GitHub 发布说明

这个目录应该作为独立小仓库提交，不要把整个工作区提交上去。

推荐流程见：

```text
docs/GITHUB.md
```

提交前确认：

- 不提交 `config.json`。
- 不提交 API key。
- 不提交 `runs/`、`outputs/`、`eval_sets/`、`eval_reports/`。
- 不提交模型权重、checkpoint 或 `.safetensors` 文件。
