# Stage64 Reliable Frontier Mining

## 目标

Stage64 的正式训练单元不是“某一次排列下恰好答错几次”的样本，而是对选项位置、解码后端和 provenance 都稳健的 frontier。

## 必须通过的门禁

1. API QA schema、graph reference、evidence 和 anti-shortcut validation 通过。
2. 选项文本非空、按 case/punctuation/whitespace 规范化后完全唯一；gold 文本只对应一个 label。
3. Number/DateOrTime/YesNo 等选项表面必须符合声明类型，不能把问题本身当作 option。
4. `qa_operation`、`qa_subtype` 和问题语义必须一致；bridge/comparison/count/cause 至少需要两个 reasoning anchors。
5. option graph refs 必须使用具体 semantic slots，不能使用 `raw_qa_*`；hard negatives 必须与 gold 共享实体、关系、slot 或推理链，而不是无关事实。
6. 固定 solver 必须解析到 `accepted_models/solver_base_restart_stage2_merged` 的真实目标，Stage64 runner 会拒绝 legacy R2 checkpoint 和不匹配的 policy provenance。
7. 每个 completion 数量正确、非空、无 replacement character；single-choice 输出必须以合法 label 开头，parsed 与前缀一致。
8. 对 N 个选项生成 N 个 cyclic permutations，使 gold 各处于每个 label 一次。
9. 默认要求每个 permutation 的 K5 bucket 都在 `1/5-4/5`，否则标为 easy、hard0 或 position-sensitive diagnostic。
10. 新 Stage3 产物按 `lineage_group` 整组切分；正式 heldout/test 不能参与模型选择。
11. 正式 target 在 API 前冻结 operation、answer type、reasoning subtype/template、hop count 和 hard-negative strategy；QA-only hardening 不得改变这些字段。
12. temporal anti-shortcut target 的 gold 必须早于最高 time_index，且更晚的同槽无效更新必须同时出现在 reasoning path 和 distractor refs。

Stage64 graph plan 会在 dialogue/QA 生成和 cache 写入之前单独过一次语义门禁。cache key 同时包含 `qa_operation`、`answer_type` 和门禁版本；命中 cache 时还会复验，旧规则留下的坏 plan 会被删除并重新生成，不会在重试中循环复用。

## 可复现 option balance

```bash
python scripts/current_method/shuffle_stage64_options.py \
  --input /path/to/clean_samples.jsonl \
  --output-dir /path/to/shuffled \
  --seed 20260711
```

脚本在 `option arity × attack skill` 内平衡 gold label，移除所有旧 eval 字段，并把结构或语义不合格题写入 `quality_rejected_samples.jsonl`。

## 只做质量审计

```bash
python scripts/current_method/recalibrate_stage64_shuffled_frontier.py \
  --input /path/to/shuffled/clean_samples.jsonl \
  --output-dir /path/to/quality_audit \
  --quality-audit-only
```

Lexical near-duplicate 默认只报告，因为日期、计数和实体的 minimal-pair distractor 可以是合法难例；只有明确需要时才传 `--reject-near-duplicate-options`。

## 旧 492 条最终审计

2026-07-11 的两层审计结果：

- 只做重复/歧义结构门禁：429/492 留下；其中全部 cyclic-position 评估得到 0 robust frontier、368 robust easy5、5 robust hard0、56 position-sensitive。
- 加入完整语义门禁：0/492 留下。主要失败为 `generic_raw_qa_slot=481`、operation/subtype 或 question 语义错配、answer surface 错配和无关 distractor。

因此旧 408 frontier、shuffle 后 325 train、单排列 15 frontier 和 12 train 都禁止用于正式训练。

## permutation-robust 重校准

```bash
python scripts/current_method/recalibrate_stage64_shuffled_frontier.py \
  --input /path/to/shuffled/clean_samples.jsonl \
  --output-dir /path/to/recalibrated \
  --solver-model /path/to/accepted_models/solver_base_restart_stage2_merged \
  --accepted-stage2-solver-model /path/to/accepted_models/solver_base_restart_stage2_merged \
  --solver-tokenizer /path/to/Qwen3-4B \
  --k-eval 5 \
  --temperature 1.0 \
  --top-p 0.9 \
  --max-new-tokens 80 \
  --permutation-mode cyclic \
  --min-frontier-permutation-share 1.0 \
  --group-level lineage_group \
  --seed 20260711 \
  --max-model-len 32768
```

发布包可部署在任意路径；`--accepted-stage2-solver-model` 是 provenance 参考路径，`--solver-model` 是本次实际请求加载的路径。两者必须解析到同一个 checkpoint。主入口同样提供 `--accepted-stage2-solver-adapter`，避免把 legacy adapter 写入父模型 provenance。K5 的 temperature、top-p、max-new-tokens 和 seed 会写入 summary；跨批累积要求这些参数完全一致。

关键输出：

- `quality_rejected_samples.jsonl`
- `stage2_recalibration_permutation_eval_rows.jsonl`
- `frontier.jsonl`
- `splits/frontier_lineage_group_holdout_seed*/frontier_train.jsonl`
- `diagnostic_hard0_samples.jsonl`
- `diagnostic_easy5_samples.jsonl`
- `diagnostic_position_sensitive_samples.jsonl`
- `reports/solver_decoding_health.json`
- `recalibration_summary.json`

顶层不会再写容易误用的全量 `sft_train.jsonl`。若确实需要全量 clean chat，只能显式传 `--write-all-clean-sft`，输出名为 `clean_samples_sft.jsonl`。

## Mining 策略

旧 492 条的 graph/event plan 本身几乎全部使用 `raw_qa_*` 泛化槽，QA-only repair 不能修复 graph 和已经实现的 dialogue，所以这批数据需要废弃。下一轮必须重新调用 API，从 graph-first 阶段生成 semantic slots 和 operation-specific hard negatives，再生成 dialogue 与 QA。

`scripts/run_auto_graph_expansion_round.py` 在 Stage64 flag 下默认自动执行该批次的 semantic gate、option balance 和 cyclic recalibration，输出到 `<run-root>/stage64_mining/`；不会再发布 current-position frontier，也不会直接启动训练。传 `--skip-eval` 时只运行生成后质量/平衡阶段。

不要一次生成几千条。先用 7 条覆盖七类 operation 的完整 smoke，再扩到每批 25-50 条。每批立即执行 semantic gate、option balance、search cyclic eval、一次 QA-only solver-feedback rewrite 和独立 final cyclic eval，只累计 final robust frontier；根据 easy5、hard0、position-sensitive 和 reject reason 反馈下一批 graph objective。达到至少 300 条 reliable frontier 后，再按 `lineage_group` 分组切 train/heldout，并冻结独立 test。

跨批累积使用：

```bash
python scripts/current_method/accumulate_stage64_reliable_frontier.py \
  --recalibration-dir /path/to/batch01/stage64_mining/batch_closure/final_reliable_batch \
  --recalibration-dir /path/to/batch02/stage64_mining/batch_closure/final_reliable_batch \
  --output-dir /path/to/stage64_accumulated \
  --min-frontier 300 \
  --heldout-ratio 0.20 \
  --seed 20260711
```

阈值未达到时只写 `frontier_pool.jsonl` 和 `accumulation_summary.json`，不会创建可误用的 train split。达到阈值且 balance/leakage gate 全部通过后才写 `training_ready_manifest.json`；正式 split 在 `splits/frontier_lineage_group_holdout_seed*/`。
