# Structured QA Solver 接口交接

本文档交接当前已经验证过的 **multi-select / ordering structured QA Solver replay** 接口。

它是 QA-only、probe-only 的结构化 Solver 协议，不是生产 QA Writer，也不替换 single-choice 主线。

## 1. 当前接口位置

可复用的协议模块：

```text
scripts/current_method/restart_n5_spell/structured_qa_box_protocol.py
```

178392 的完整 replay runner（包含本地 vLLM 加载、题目重绑定、5 次 rollout 和报告）：

```text
scripts/current_method/restart_n5_spell/episode_178392_structured_qa_position_probe_v1/run.py
```

零 API 测试：

```text
scripts/current_method/restart_n5_spell/test_structured_qa_box_protocol.py
```

协议模块可以被其他 structured QA 工作直接导入；`episode_178392.../run.py` 是冻结题集的专用 replay，不要把它当成通用 QA 生成器。

## 2. 支持的题型和输出合同

Solver 最终输出必须是唯一一对 `<box>...</box>`，且不能有任何盒子外文本。

多选题：

```text
<box>A,C</box>
```

当前协议支持 A–E 的非空子集（共 31 个合法答案）。

排序题：

```text
<box>I2,I4,I1,I3</box>
```

排序题必须完整覆盖题目传入的所有 item IDs，且每个 ID 只出现一次。

当前 parser 不接受：

```text
A,C
Answer: A,C
I2>I4>I1>I3
<box>A,C</box>\nExplanation: ...
```

这些均属于 `parse_failure`。如果 box 合法但答案与 gold 不同，则属于 `reasoning_error`，不能混为格式失败。

## 3. 可复用的 Python 接口

```python
from scripts.current_method.restart_n5_spell.structured_qa_box_protocol import (
    BOXED_PROTOCOL_VERSION,
    STRUCTURED_GUIDED_DECODING_VERSION,
    STRUCTURED_SOLVER_MAX_TOKENS,
    STRUCTURED_SOLVER_TEMPERATURE,
    classify_structured_result,
    evaluate_multi_select,
    evaluate_ordering,
    parse_structured_box,
    structured_box_choices,
)
```

### 3.1 生成 guided-decoding 的合法答案集合

```python
choices = structured_box_choices("multi_select")
# ["<box>A</box>", "<box>B</box>", ..., "<box>A,B,C,D,E</box>"]

choices = structured_box_choices(
    "ordering",
    allowed_item_ids=["I1", "I2", "I3", "I4"],
)
# 4 个 item 时返回 24 个完整排列
```

`choices` 只限制输出语法和有限答案空间，不知道 gold，也不替 Solver 做语义判断。

### 3.2 解析一次 Solver 输出

```python
parsed = parse_structured_box(
    raw_output,
    question_type="multi_select",
)

parsed = parse_structured_box(
    raw_output,
    question_type="ordering",
    allowed_item_ids=["I1", "I2", "I3", "I4"],
)
```

成功时的关键字段：

```json
{
  "raw_output": "<box>A,C</box>",
  "parsed_answer": ["A", "C"],
  "parse_status": "success",
  "failure_type": null,
  "parser_version": "structured_qa_box_v1"
}
```

失败时：

```json
{
  "raw_output": "Answer: A,C",
  "parsed_answer": [],
  "parse_status": "parse_failure",
  "failure_type": "parse_failure",
  "parse_reason": "..."
}
```

### 3.3 绑定 gold 并分类失败

```python
result = classify_structured_result(
    parsed,
    gold=["A", "B", "C"],
    question_type="multi_select",
)
```

规则：

```text
parse_status != success
→ failure_type = parse_failure

box 合法但 prediction != gold
→ failure_type = reasoning_error

box 合法且 prediction == gold
→ failure_type = null
```

不要从 raw text 自己重新猜答案，也不要把裸文本修补成 box；否则会把格式问题和推理能力混在一起。

## 4. 评价指标

多选题：

```python
metrics = evaluate_multi_select(predicted, gold)
```

返回：

```text
exact_match   # 当前 admission 主指标
precision     # 辅助指标
recall        # 辅助指标
jaccard       # 辅助指标
failure_type  # null 或 reasoning_error
```

排序题：

```python
metrics = evaluate_ordering(predicted, gold)
```

返回：

```text
exact_order_match  # 主指标
pairwise_accuracy  # 辅助指标
failure_type       # null 或 reasoning_error
```

`parse_failure` 不计入推理难度；汇总时必须单独统计。

## 5. 本地 vLLM 调用约定

当前结构化 probe 使用：

```text
temperature = 0.0
max_tokens = 32
stop = ["</box>"]
include_stop_str_in_output = true
guided_decoding = GuidedDecodingParams(choice=choices)
```

Prompt 必须使用 Qwen tokenizer 的 chat template，而不是把可读文本直接交给引擎：

```python
prompt = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": STRUCTURED_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
```

`build_chat_prompt` 已在专用 runner 中实现。其他任务如果复用它，应保持 `enable_thinking=False`；否则 Qwen 可能先输出思维内容，破坏 strict box 合同。

这组参数只属于 structured Solver probe，不覆盖 single-choice 的 `single_choice_solver_max_tokens` 或生产 Solver 配置。

## 6. 选项重绑定/打乱

多选题可以复制题目后重排 option 文本，并根据文本反算新的 gold：

```text
原题 gold = ["A", "C"]
重排后必须重新得到 gold_after_rebinding
```

不要把旧字母直接沿用到重排题。

排序题的 item ID 是题目合同的一部分；改变展示顺序不改变 gold 的 item ID 顺序。不要把排序题当成单选字母题评价。

冻结题目不能被 replay runner 原地修改；重排结果只写入新的 run 目录。

## 7. 当前已验证结果

最终的 zero-API/local replay 产物是：

```text
/root/autodl-tmp/gxy-selfplay/restart-n5-runs/
conversation_chronicles_episode_178392_structured_qa_box_protocol_replay_local_v4/
```

报告：

```text
.../reports/STRUCTURED_BOX_REPLAY_REPORT.md
.../reports/STRUCTURED_BOX_REPLAY_REPORT.json
```

v4 使用 chat template + finite-choice guided decoding，450 次本地生成：

```text
parse_failure = 0
finish_reason=stop = 450
reasoning_error = 250

no-context exact = 80/225  (35.6%)
full-context exact = 120/225 (53.3%)
no-context MSQ = 33.3%
full-context MSQ = 56.7%
no-context ordering = 40.0%
full-context ordering = 46.7%
```

因此这次协议修复验证的是：**输出格式和 parser 已稳定，剩余错误主要是推理错误，而不是截断/格式污染。** 这不等于题目已经进入生产 admission。

## 8. 其他工作如何调用

推荐调用顺序：

```text
1. 准备 frozen question，不改 Writer 产物
2. 对当前 option/item 顺序建立 choices
3. 用 chat template 渲染 prompt
4. 使用结构化参数生成一次 raw output
5. parse_structured_box
6. classify_structured_result
7. 单独汇总 parse_failure 与 reasoning_error
8. 只用 exact_match / exact_order_match 做当前 admission
```

如果任务只需要离线解析已有输出，不要加载模型，直接调用 parser/evaluator。

如果任务需要复用完整 runner，使用：

```bash
python -m scripts.current_method.restart_n5_spell.episode_178392_structured_qa_position_probe_v1.run \
  --expanded <frozen-expanded-dialogue.json> \
  --questions <frozen-structured-questions.json> \
  --run-root <new-run-root>
```

`--run-root` 必须是不存在的新目录；runner 不 resume、不覆盖历史 run。

## 9. 明确禁止的复用方式

不要把以下生产文件当成 structured interface：

```text
scripts/current_method/restart_n5_spell/simple_append_v1/qa.py
scripts/current_method/restart_n5_spell/simple_append_v1/qa_live.py
scripts/current_method/restart_n5_spell/simple_append_v1/prompts.py
```

它们属于 single-choice/生产 QA 路线，不能因为名字相似而替换本协议。

本接口也不负责：

```text
QA Writer 生成
题目 admission
Expansion / Planner / Realizer
Memorax
训练
```

不要修改历史 run、生产 QA、single-choice solver、Planner、Realizer 或 Memory 来适配 structured probe。

## 10. 最小自检

```bash
cd /root/autodl-tmp/gxy-selfplay/current_method_github
python -m compileall scripts/current_method/restart_n5_spell/structured_qa_box_protocol.py \
  scripts/current_method/restart_n5_spell/episode_178392_structured_qa_position_probe_v1/run.py
python -m pytest -q scripts/current_method/restart_n5_spell/test_structured_qa_box_protocol.py
git diff --check
```

以上自检不调用 API。若 pytest 环境不可用，至少执行 compileall 和直接导入/fixture assertions，并在报告中标记环境原因。

## 11. 版本边界

```text
BOXED_PROTOCOL_VERSION = structured_qa_box_v1
STRUCTURED_GUIDED_DECODING_VERSION = finite_choice_set_v1
STRUCTURED_SOLVER_MAX_TOKENS = 32
STRUCTURED_SOLVER_TEMPERATURE = 0.0
```

任何要修改 box 合同、允许的答案集合、parser 严格性或 structured 生成参数的工作，都应新建独立 probe 或先更新本交接文档；不要静默改动后让其他任务继续使用旧假设。
