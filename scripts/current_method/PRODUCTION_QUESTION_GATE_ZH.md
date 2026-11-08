# 批量题目来源绑定与审核接口

`production_question_gate.py` 补充现有 `question_presentation_gate.py`，不修改任何历史冻结文件。

## 顺序

1. 在 `SELECTION_FREEZE.json` 中固定 `roots`：每条包含 `root_id`、绝对 `path`、`sha256`，建议同时包含 `bytes`。来源 JSON 须有同一 `root_id` 及 `turns`（唯一 `turn_ref`、`speaker`、`text`）。
2. 作者写候选，来源绑定可省略。程序从选取冻结唯一派生绝对路径、SHA256 和字节数。作者相对路径可规范化；作者提供的 SHA 与冻结不一致直接报错，不悄悄替换素材。
3. 在自动绑定后的完整候选上计算 `fingerprint`；作者与独立审核者均引用这个哈希。审核后改一字会使绑定失效。
4. 机械审核逐项和逐边检查引文是对应说话人、轮次的逐字非空子串。排序图必须无环、有唯一全序且与 Gold 一致；图中关系是否被原文蕴含仍须独立语义审核。
5. 独立审核者完成全部显式检查项。`validate_review` 只在机械检查通过、独立身份、候选/来源绑定完整且各必要语义项均 PASS 时签发 `admitted`。这代表审核者判断，绝不代表程序证明语义正确。

## 候选

沿用现有 `candidate_id`、`root_id`、`author_id`、`question_type`、`task_operation`、`question`、`options` 或 `items`、`gold`。`item_analysis` 须正好覆盖全部项，每项 `evidence` 为非空 `[{turn_ref,speaker,quote}]`；多选另有布尔 `should_select`，与 Gold 成员一致。

排序的 `ordering_analysis` 提供：

```json
{
  "operation_class": "clock_value",
  "source_dependence_rationale": "说明哪些值或关系必须从本对话提取或绑定",
  "edges": [
    {"before": "C", "after": "A", "evidence": [
      {"turn_ref": "T003", "speaker": "Alice", "quote": "逐字原文"}
    ], "rationale": "解释原文为什么支持这个比较"}
  ]
}
```

`operation_class` 为 `clock_value`（钟点数值）、`occurrence_chronology`（事件实际发生）、`discourse_reconstruction`（话语/交互重建）、`version_evolution`（版本演变）或 `source_defined_dependency`（来源明示依赖）。已明确的旧操作名可自动推定；`ORDER_EXPLICIT_PROCEDURE` 的含义不唯一，应显式填写，不能默认推断为流程依赖。

## 排序来源依赖的检查边界

程序只报告每项**作者最早引证轮次**，不是已经核验的首次相关提及。多项同轮有并列时不虚构轮内顺序。Gold 与最早引证轮次同序仅触发审核解释，不自动拒绝。

审核者必须定位首次相关提及并区分操作：话语重建的正确答案本来就可能等于话语顺序；实际事件排序若恰好与提及顺序相同，需要解释是否出现可绕过目标证据的捷径。常识流程能否独立完成、编辑性提示词、可直接排序的显示钟点均单独审核。保留真正来源内容，不因命中 first/later 等词就机械删除。

`source_dependence_rationale` 是作者说明，不能替代 `source_dependence` 独立检查。来源内容不足时保留 HOLD 缺额，不强造关系、偷换操作或靠修改 Gold 补齐配额。

## 审核记录

`build_review_packet` 提供逐题模板。必须字段：

- `candidate_id`、`candidate_sha256`、`source_binding`、`author_id`、`reviewer_id`（与作者不同）。
- `decision` 为 `PASS`、`HOLD` 或 `FAIL`。
- `checks` 每项为 `{decision,rationale}`，理由须为非空的实际审核说明。
- 通用项：`source_support`、`gold_truth`。
- 多选：`option_distinctness`，另有 `distractor_relation`；六 Gold 改为 `six_gold_control`。
- 排序：`operation_alignment`、`unique_total_order`、`mention_order`、`common_sense`、`editorial_cues`、`source_dependence`。

可以额外保存 `reconstructed_gold`、`initial_content_decision`、`issues`、`source_dependence_review` 和逐项复核。审核者可见 Gold 属于可见答案的独立 AI 审核，不是人类盲审；身份分离的声明也不是共享文件系统上的技术盲法。

## 调用

```python
from scripts.current_method.production_question_gate import (
    bind_candidates, build_review_packet, audit_candidate, validate_review,
)
bound, binding_audit = bind_candidates(drafts, selection_path)
packet = build_review_packet(bound[0], selection_path)
admission = validate_review(bound[0], independent_review, selection_path)
```

```bash
python scripts/current_method/production_question_gate.py \
  --input DRAFT.json --selection SELECTION_FREEZE.json --output-dir new_gate_dir
```

可追加 `--reviews CONTENT_REVIEW_AI.json` 检查准入。目录必须不存在；所有派生候选和审核包权限为 0600，不能当公开 Solver 输入。公开题面仍由现有 presentation gate 的白名单输出。返回码 0 表示本次要求的机械检查（或提供审核后的全部准入）通过；未提供审核时 `admitted_count` 始终为 0。

验证：`python -m unittest scripts.current_method.test_production_question_gate scripts.current_method.test_question_presentation_gate`。覆盖素材漂移/错根、错误引文/说话人、绑定替换、排序有环/非唯一/Gold不一致、同轮引证并列、合法话语重合、缺审核项、作者自审、候选复写、六 Gold 控制和防覆盖写。
