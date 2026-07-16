# 跨阶段能力增长评估面板

这个面板支持两种模式：历史 Stage64 复现，以及真正的 Stage3 评估。Stage3 模式回答三个配对问题：Stage1 是否超过 base、Stage2 是否超过 Stage1、Stage3 是否在 reliable Stage3 heldout 上超过固定的 accepted Stage2。

## 评估口径

- `sampled_first_draw` 是一次 K-sample 运行中的第一个 completion，不是独立 greedy K1。
- 新版 `--run-original` 默认额外运行独立 `K=1, temperature=0`，报告优先使用 `greedy_k1`；可用 `--skip-greedy-original` 关闭。
- K5 soft 是五次 exact correctness 的平均，不是 pass@5。
- 所有阶段比较按相同 `sample_id` 配对，并报告 improved、regressed 和双侧 exact McNemar p 值。
- Stage1/Stage2 heldout 参与过模型选择，属于 validation；论文主结果还需要冻结、未参与选择的新 test。

## 使用

只汇总已有产物：

```bash
python scripts/current_method/cross_stage_eval_panel.py \
  --panel-dir base_restart_current_method_validation/70_cross_stage_eval_panel
```

重新运行 sampled K5 和独立 greedy K1：

```bash
python scripts/current_method/cross_stage_eval_panel.py \
  --panel-dir base_restart_current_method_validation/70_cross_stage_eval_panel \
  --run-original \
  --k-eval 5 \
  --seed 20260711 \
  --max-model-len 32768
```

旧 chat summaries 使用 `sampling_k=5`，而新版 chat 默认 `sampling_k=1`。collector 会拒绝参数不匹配的旧产物，防止静默混用。

## 当前可支持的结论

现有 original-format 产物尚未单独跑 greedy K1，因此下面是 sampled first-draw 配对诊断：

| comparison | paired n | improved | regressed | net | exact p | 判断 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Stage1 vs base on Stage1 heldout | 22 | 7 | 0 | +7 | 0.0156 | 小样本 validation 正向证据 |
| Stage2 vs Stage1 on Stage2 heldout | 30 | 6 | 5 | +1 | 1.0000 | 尚未证明阶段收益 |
| old Stage64 vs Stage2 on legacy Stage64 heldout | 83 | 2 | 4 | -2 | 0.6875 | legacy，仅诊断 |

因此不能再写“Stage1 和 Stage2 的逐阶段收益都已成立”。准确表述是：Stage1 有小样本 validation 证据；Stage2 当前只净增 1/30；Stage64 尚无可靠 heldout。

## 旧 Stage64 产物为什么是 legacy

旧 408 frontier 同时受以下问题影响：

1. 使用的是历史 legacy R2 checkpoint，不是 base-restart accepted Stage2；
2. K1/K5 解码产生大量乱码，宽松 parser 会从任意文本中捞取 A-D；
3. gold 极度偏 A；
4. option shuffle 后沿用了 shuffle 前的 eval 标签；
5. 492 条中有 63 条规范化后完全重复选项，45 条存在 gold 多标签歧义；
6. 481 条选项仍依赖 `raw_qa_*` 泛化槽，且 answer type / operation/subtype / distractor 语义大量错配。

71 目录用正确 accepted Stage2 做了健康的单排列重评，得到 15 条 frontier 候选，但其中 2 条仍是重复/歧义选项。73 目录对结构合格的 429 条做全部 cyclic positions，结果为 0 robust frontier、368 robust easy5、5 robust hard0、56 position-sensitive。76 目录加入完整语义门禁后，旧 492 条 0 条合格。因此这些结果只能用于诊断，不能作为正式 train/heldout。

## 新 Stage64 heldout 接入

完成 permutation-robust 重校准后，显式传入新路径：

```bash
python scripts/current_method/cross_stage_eval_panel.py \
  --panel-dir /path/to/new_panel \
  --stage64-heldout /path/to/frontier_heldout.jsonl \
  --stage64-chat-heldout /path/to/frontier_heldout_sft.jsonl \
  --stage64-heldout-calibrated \
  --run-original
```

只有在以下条件全部满足时才能使用 `--stage64-heldout-calibrated`：选项唯一性 gate 通过、accepted Stage2 固定、cyclic permutations 完成、所有排列均为 frontier、按 `parent_sample_id` 切分且 test 未参与模型选择。

## Stage3 模式

Stage3 训练完成后使用：

```bash
python scripts/current_method/cross_stage_eval_panel.py \
  --panel-dir /path/to/stage3_panel \
  --stage3-model /path/to/stage3_merged \
  --stage3-heldout /path/to/frontier_heldout.jsonl \
  --stage3-chat-heldout /path/to/frontier_heldout_sft.jsonl \
  --stage3-hard0 /path/to/diagnostic_hard0_samples.jsonl \
  --stage3-easy5 /path/to/diagnostic_easy5_samples.jsonl \
  --run-original \
  --run-chat \
  --k-eval 5 \
  --seed 20260711 \
  --max-model-len 32768
```

Stage3 模式注册 `base/stage1/stage2/stage3` 四个模型，主比较为 Stage3 vs Stage2。报告仍保留 Stage1/Stage2 retention，并且不会把新结果写成 Stage64 或 legacy。可靠 heldout 必须按 `lineage_group` 整组切分，正式 heldout 不能参与 train mix 或训练期 validation。
