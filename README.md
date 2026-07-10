# Current Method Self-Play 代码包

这是“当前方法”的干净 GitHub 代码包，包含从 parent pool 生成 self-play 样本、API challenger 扩展对话、Stage64 API QA 多样化、质量验证、frontier 选择、train mix 构建、可选 solver 训练与评测所需的主线代码。

注意：这里整理的是**当前方法主线代码**，不是整个工作区的 r1-r5 历史归档。模型权重、运行缓存、API key 和 149G 工作区内容都没有放进来。

`data/` 目录已经加入当前方法实验线路的可复现数据子集，包括原始 `locomo10.json`、切分后的 seed / parent rows、stage1、stage2、stage2 repair/topup、Stage64 API self-play 与 Stage64 题目类型多样化 probe 的关键 JSON/JSONL、报告和状态文件。数据目录不是完整 29G 工作区镜像；具体包含和排除内容见 `data/README.md` 与 `data/DATA_MANIFEST.tsv`。

## 当前方法实验流程

```text
1. 准备 parent pool / seed rows
   -> 2. 构建 challenger policy / curriculum
   -> 3. API 生成 graph event plan
   -> 4. API 生成 appended dialogue + qa_proposal
   -> 5. 本地验证 graph/dialogue/QA/长度/泄漏
   -> 6. 输出 clean_samples / reject logs / summary
   -> 7. solver eval + frontier selection
   -> 8. 构建 train mix
   -> 9. 可选：LoRA 训练、merge、vLLM 评测
```

Stage64 模式是当前方法中的题目类型扩展模块：它把原来比较单一的状态类/选择类问题，扩展成按 `qa_operation` 和 `answer_type` 控制的多题型 self-play 数据。API challenger 负责生成最终 `qa_proposal`，本地代码负责检查这些新题型是否真的满足目标类型和图证据约束。

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
    current_method/
      *.py
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

运行 10 条 Stage64 API smoke test：

```bash
python scripts/run_auto_graph_expansion_round.py \
  --run-name stage64_api_selfplay10 \
  --run-root runs/stage64_api_selfplay10/run \
  --output-dir runs/stage64_api_selfplay10 \
  --report-dir runs/stage64_api_selfplay10/reports \
  --parent-dir /path/to/topup_combined_frontier_pool.jsonl \
  --generation-backend native_current_method \
  --native-challenger-mode graph_conditioned_api_dialogue \
  --native-curriculum-mode r3_long_easy_v2 \
  --qa-operation-answer-type-diversification \
  --train-only \
  --n-train 10 \
  --max-llm-retries 1 \
  --max-attempts-per-sample 4 \
  --max-api-failures 50 \
  --min-added-turns 24 \
  --min-length-delta-tokens 4000 \
  --generation-concurrency 1 \
  --llm-timeout 180
```

`--parent-dir` 需要指向已有 parent pool JSONL，例如 mined frontier pool。这个代码包不包含 parent data。

## 核心代码说明

### 总入口

| 文件 | 实验流程位置 | 作用 |
| --- | --- | --- |
| `scripts/run_auto_graph_expansion_round.py` | 全流程入口 | 当前方法主入口。负责读取 parent rows、构建 policy、调用 native/API generator、写报告，并可选触发 solver eval、frontier selection 和训练流程。 |
| `scripts/current_method/native_challenger_generation.py` | 生成入口分发 | native current-method generator 的分发层。`graph_conditioned_api_dialogue` 模式会转到 `native_api_graph_dialogue_expansion.py`。 |
| `scripts/current_method/native_api_graph_dialogue_expansion.py` | challenger 生成核心 | 当前 API self-play 的核心文件。包含 graph/event-plan API prompt、dialogue API prompt、Stage64 `qa_proposal` 处理、alignment、schema、长度和泄漏验证、reject 记录、样本落盘。 |

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
| `scripts/current_method/qa_operation_answer_type_anchor_diversification_probe.py` | Stage64 probe / 分析 | 用于 anchor-first QA operation / answer type diversification 的 probe、诊断和数据整理，主要分析题型覆盖、anchor 使用和答案类型分布。 |

### Seed、数据准备和审计

| 文件 | 实验流程位置 | 作用 |
| --- | --- | --- |
| `scripts/current_method/prepare_locomo_current_method_seed.py` | seed 准备 | 从 LoCoMo / parent data 准备 current-method seed rows，并记录 provenance。 |
| `scripts/current_method/seed_audit.py` | seed 审计 | 检查 train/dev/heldout provenance，防止 forbidden seed 或 split 泄漏。 |
| `scripts/current_method/api_challenger_generation.py` | 基础生成/验证适配 | 当前方法复用的基础 adapter，包含 JSONL 读写、样本验证、solver eval 接口、clean/valid 判断等公共逻辑。 |

### Frontier、报告和诊断

| 文件 | 实验流程位置 | 作用 |
| --- | --- | --- |
| `scripts/current_method/frontier_selection.py` | frontier selection | 根据 solver 评测结果筛 frontier、strict frontier、boundary frontier 等。 |
| `scripts/current_method/report_builder.py` | 报告生成 | 汇总生成质量、frontier yield、题型分布、solver 表现、长度统计等。 |
| `scripts/current_method/cross_stage_eval_panel.py` | 跨阶段能力验证 | 汇总或复跑 Stage1/Stage2/Stage64 heldout 上的 solver 对比，验证 self-play 是否真的逐阶段提升。 |
| `scripts/current_method/write_base_restart_validation_summary.py` | 验证汇总 | 为 base restart / current-method validation 运行写总览 summary。 |
| `scripts/current_method/diagnostics_context_ablation.py` | 诊断实验 | 做 context ablation，分析 solver 对 parent context / evidence window 的依赖。 |
| `scripts/current_method/eval_api_14b_samples.py` | API solver 评测 | 用 API 模型评测样本，可用于快速 sanity check 或外部 judge。 |

### Train mix / mining 数据构建

| 文件 | 实验流程位置 | 作用 |
| --- | --- | --- |
| `scripts/current_method/build_base_restart_train_mix.py` | train mix 构建 | base restart 早期 train mix 构建脚本。 |
| `scripts/current_method/build_base_restart_stage2_train_mix.py` | stage2 train mix 构建 | 当前方法 stage2 train mix 构建，支持 frontier、clean replay、top-up 等组合。 |
| `scripts/current_method/build_base_restart_stage2_repair_train_mix.py` | repair train mix 构建 | 用于 stage2 repair/top-up 的 train mix 构建。 |
| `scripts/current_method/build_chained_parent_expansion_probe.py` | chained parent probe | 构建 chained parent expansion 相关 probe。 |
| `scripts/current_method/build_chained_parent_v2_from_child_v2.py` | chained parent v2 | 从 child v2 构建 parent v2，用于链式 parent/child 数据实验。 |
| `scripts/build_r2_frontier_plus_clean_replay_train.py` | 训练数据构建 wrapper | 根据 frontier + clean replay 构建 solver SFT train JSONL。当前训练 wrapper 会调用它。 |

### 可选训练、merge 和评测

| 文件 | 实验流程位置 | 作用 |
| --- | --- | --- |
| `scripts/run_round_solver_training_pipeline.py` | 训练总入口 | 可选流程：构建 train data、训练 LoRA、merge adapter、运行 smoke/full eval。 |
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

Stage64 最大的贡献是**题目类型多样化**。它把当前方法从“主要生成一种宽泛的单选状态判断题”，推进到“按 QA operation 和 answer type 明确控制的多类型题目生成”。

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
   - `PersonOrEntity`
   - `YesNo`

   本地代码会检查 API 返回的 `qa_operation`、`answer_type`、options 和 gold 是否一致。

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
   - option 的 `answer_type` 不一致；
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

当前面板结论是：Stage1 和 Stage2 的逐阶段收益已经成立；Stage64 虽然在 option shuffle 后做过标签重平衡和基于旧 frontier 标记的重新切分，但没有重新用固定 Stage2 solver 评估并重选 frontier，所以当前 heldout 不能作为严格能力增长证据。详细解释见 `docs/CROSS_STAGE_EVAL_PANEL.md`。

## 质量门

生成器会拒绝以下样本：

- 出现 solver 可见的内部词，例如 `raw_qa`、`anchor`、`event_instance`、`latest confirmed detail`、`settled as true`。
- API 返回的 QA schema 不合法。
- `qa_operation` 和 `answer_type` 不匹配。
- `count` 没有返回 Number 类型。
- gold option id 无效或不唯一。
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
