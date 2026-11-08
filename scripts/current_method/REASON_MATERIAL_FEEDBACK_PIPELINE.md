# 理由与素材反馈主流程入口

`reason_material_feedback_pipeline.py` 把已经审核的 Train 理由与素材缺口接到按 root 隔离的规划、来源交接和来源准入。它不调用模型，不把写出提示词计为模型消费，不修改旧冻结实验。

## 准备交接输入

```bash
python scripts/current_method/reason_material_feedback_pipeline.py prepare \
  --config /absolute/path/CONFIG.json \
  --output-dir /absolute/path/new_handoff_directory
```

每个文件 binding 的形式均为：

```json
{"path":"/absolute/path/file.json","sha256":"真实文件SHA256","bytes":123}
```

配置契约：

```json
{
  "schema_version": "reason_material_feedback_pipeline_v1_config",
  "batch_id": "train_reason_material_1",
  "namespace": "RMF1",
  "reason_packet": {"path":"/absolute/path/INPUT_PACKET.json","sha256":"...","bytes":123},
  "shortfall_ledger": {"path":"/absolute/path/ADMISSION_LEDGER_FROZEN.json","sha256":"...","bytes":123},
  "roots": [
    {
      "root_id": "cc-train-episode-136202",
      "source_binding": {"path":"/absolute/path/source.json","sha256":"...","bytes":123},
      "task_operations": [{"operation_id":"source_status_ordering","question_type":"ordering"}]
    },
    {
      "root_id": "cc-train-episode-104814",
      "source_binding": {"path":"/absolute/path/original_source.json","sha256":"...","bytes":123},
      "task_operations": [{"operation_id":"explicit_event_time","question_type":"ordering"}],
      "material_targets": [
        {"slot_id":"DIVV3-R2-ORD2","material_goal":"在新对话中自然补充相关未来事件的共同日期与明确时间，保留提议、确认和完成的区别。"}
      ]
    }
  ]
}
```

`reason_packet` 接 `postweek_reason_feedback_repair_v1/repair.py export-feedback` 的实际 `INPUT_PACKET.json`，逐条核对原始模型输出、审核和来源版本。`unknown` 审核拒绝；`fail` 可以作为已经审核的错误观察保留，不变成事实或正式弱点。未选理由在回执中列明。

SHORTFALL 必须有独立 `review_binding`，且对应审核条目为 `PASS_SHORTFALL` 或 `SOURCE_SHORTFALL`。显式 `material_goal` 必须由当前消费者根据该缺口给出；程序不以字符串规则猜测事实关系。素材缺口不等于 Solver 弱点。

同 root 理由和缺口必须绑定同一源版本。有素材目标路由为 `EXTEND`，否则为 `REUSE`。所有 `task_operations` 原样保留，禁止其中夹带 Gold、答案或目标 Gold 数量。

每 root 产物：

- `CHALLENGER_INPUT.json/.txt`：完整来源、本 root 审核理由、素材缺口、task operations，供实际 Challenger 规划使用。
- `SOURCE_AUTHOR_INPUT.json/.txt`：完整来源、必要事实关系目标与路线约束，不含 Solver 理由、未来答案及 Gold 数量。
- 批次 `PROGRAM_READ_RECEIPT.json`：只证明程序读取了绑定输入；`model_calls=0`、`llm_consumption_proven=false`。

新生成 `EXTEND` 还需在对应 root 配置中提供三个 B 场的 `native_events`，每条包含 `scene_ref`、`source_id`、`split:"train"`、`binding`、`reserved_before_authoring:true`。没有预留时交接包明确 `author_input_ready=false`；不能绕过该状态生成新 B 场。来源池永久预留由上游执行，本控制器只核对给定绑定与声明。

## 来源验收

```bash
python scripts/current_method/reason_material_feedback_pipeline.py accept-source \
  --handoff-dir /absolute/path/handoff/cc-train-episode-136202 \
  --submission /absolute/path/SUBMISSION.json
```

REUSE submission：

```json
{
  "root_id":"cc-train-episode-136202",
  "mode":"REUSE",
  "author_input_binding":{"path":"/absolute/path/SOURCE_AUTHOR_INPUT.json","sha256":"...","bytes":123},
  "source_binding":{"path":"/absolute/path/unchanged_source.json","sha256":"...","bytes":123},
  "source_author_id":"真实来源作者或有来源依据的作者角色标识",
  "review_binding":{"path":"/absolute/path/current_source_review.json","sha256":"...","bytes":123}
}
```

当前完整来源审核契约：

```json
{
  "source_binding":{"path":"/absolute/path/source.json","sha256":"...","bytes":123},
  "reviewer_id":"实际审核者标识",
  "reviewer_type":"ai_agent",
  "independent_of_source_author":true,
  "full_source_read":true,
  "turn_refs_read":["必须逐项列出完整累计来源的所有turn_ref"],
  "decision":"READY",
  "checks": {
    "time":{"status":"PASS","reason":"实际审核说明"},
    "location":{"status":"PASS","reason":"实际审核说明"},
    "actor_identity":{"status":"PASS","reason":"实际审核说明"},
    "plan_vs_completion":{"status":"PASS","reason":"实际审核说明"},
    "cross_scene_continuity":{"status":"PASS","reason":"实际审核说明"},
    "semantic_repetition":{"status":"PASS","reason":"实际审核说明"},
    "naturalness":{"status":"PASS_WITH_WARNING","reason":"实际局限说明"},
    "event_fidelity":{"status":"PASS","reason":"实际审核说明"}
  },
  "findings":[],
  "created_at_utc":"实际带时区的审核时间"
}
```

审核者必须不同于来源作者。程序校验全文覆盖与文件绑定，但语义结论仍是声明审核者的判断；不把程序验收称为独立人工审核。

新 EXTEND submission 包含 `root_id`、`mode:"EXTEND"`、`author_input_binding`、按输入顺序的六条 `scenes`，以及最终 `material_realization_review_binding`。每场记录有 `scene_ref`、`draft_binding`、`source_binding`、`review_binding`。draft 字段为 `root_id`、`scene_ref`、`author_id`、`created_at_utc`、`input_source_binding`、`author_input_binding`、`turns`；B 场再带 `native_event_binding` 和 `native_event_source_id`。每场12–16 turns，turn包含 `turn_ref`、`scene_ref`、`session_ref`、`speaker`、`text`。审核再额外绑定 `draft_binding`。下一场作者时间不得早于上一场 READY 审核时间。

## 复用已有合格扩充后代

为了补素材，先检查已存在的审核后代，不重复生成相同素材。当前支持原有 T4R 冻结格式的显式导入，历史文件不改写。

```json
{
  "root_id":"cc-train-episode-104814",
  "mode":"EXTEND",
  "author_input_binding":{"path":"/absolute/path/current/SOURCE_AUTHOR_INPUT.json","sha256":"...","bytes":123},
  "existing_extension_import":{
    "namespace":"T4R",
    "source_freeze_binding":{"path":"/absolute/path/day3_targeted_4root_v1/SOURCE_FREEZE.json","sha256":"...","bytes":123},
    "source_author_id":"有历史依据的原作者标识",
    "current_source_review_binding":{"path":"/absolute/path/current_full_source_review.json","sha256":"...","bytes":123},
    "material_realization_review_binding":{"path":"/absolute/path/current_material_review.json","sha256":"...","bytes":123}
  }
}
```

程序从原 SOURCE_FREEZE 读取独立全文 QA、六场原始 draft/parent review/acceptance、原生 Train 事件预留，验证累计来源与时间前驱。仍要求本轮完整来源审核及具体缺口实现审核。历史 draft 不会被补写成读取了本轮提示词。

素材实现审核格式：

```json
{
  "source_binding":{"path":"/absolute/path/accepted_descendant_source.json","sha256":"...","bytes":123},
  "reviewer_id":"实际审核者标识",
  "reviewer_type":"ai_agent",
  "full_source_read":true,
  "targets":[
    {
      "slot_id":"DIVV3-R2-ORD2",
      "status":"READY",
      "notes":"结合全部证据解释为何现在存在唯一可比时间关系，以及仍保留的计划/完成边界。",
      "evidence":[{"turn_ref":"T4R:A:S01:T003","quote":"来源中实际存在的逐字片段"}]
    }
  ]
}
```

新扩充与导入后代都必须通过素材实现审核，不能仅以通用来源审核代替题型素材充分性。导入通过时输出 `mode=REUSE_EXISTING_EXTENSION`，明确 `existing_source_generated_for_current_feedback=false`。

两条来源路径最终统一输出 `QUESTION_READY_SOURCE.json`，包含实际来源binding、task operations、来源审核及来源版本链。它代表可以开始出题，尚不代表题目内容通过、模型消费过提示词、正式弱点复现、训练准入或收益成立。

## CPU 验证

```bash
python -m unittest scripts.current_method.test_reason_material_feedback_pipeline -v
```

测试覆盖REUSE、新六场扩充、旧后代导入、输入隔离、Gold字段阻断、未知/未完成理由审核、原始理由篡改、来源版本变化、自审、全文漏读、前驱未就绪、事件替换、原前缀变化及素材仍不足。
