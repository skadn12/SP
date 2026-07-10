# 跨阶段能力增长评估面板

这个文档说明如何验证 self-play 是否真的带来逐阶段能力增长。

核心问题是：

```text
Stage1 solver 在 Stage1 heldout 上是否超过 base？
Stage2 solver 在 Stage2 heldout 上是否超过 Stage1？
Stage64 solver 在 Stage64 heldout 上是否超过 Stage2？
```

只看训练集准确率或单轮 frontier yield 不够。每一阶段都需要一个由上一阶段 solver 校准出来的 heldout frontier，然后用固定评测方式比较不同 solver。

## 脚本

```text
scripts/current_method/cross_stage_eval_panel.py
```

默认行为是读取已有评测 JSON，汇总成 Markdown 报告，不会重新跑模型：

```bash
python scripts/current_method/cross_stage_eval_panel.py \
  --panel-dir base_restart_current_method_validation/70_cross_stage_eval_panel
```

输出：

```text
base_restart_current_method_validation/70_cross_stage_eval_panel/cross_stage_eval_report.md
```

如果需要重新跑 original-format verifier：

```bash
python scripts/current_method/cross_stage_eval_panel.py \
  --panel-dir base_restart_current_method_validation/70_cross_stage_eval_panel \
  --run-original \
  --k-eval 5 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90
```

如果需要重新跑 SFT chat-format eval：

```bash
python scripts/current_method/cross_stage_eval_panel.py \
  --panel-dir base_restart_current_method_validation/70_cross_stage_eval_panel \
  --run-chat \
  --max-length 32000 \
  --max-model-len 32768 \
  --chunk-size 8 \
  --gpu-memory-utilization 0.90
```

## 当前面板结论

当前已经跑出的面板显示：

| solver | Stage1 heldout | Stage2 heldout | Stage64 heldout |
| --- | ---: | ---: | ---: |
| base | 45.45% | 43.33% | 97.59% |
| stage1 | 77.27% | 53.33% | 97.59% |
| stage2 | 63.64% | 56.67% | 97.59% |
| stage64 | 36.36% | 16.67% | 95.18% |

可以得到三个判断：

1. Stage1 的能力增长成立：`77.27% > 45.45%`。
2. Stage2 的能力增长成立，但幅度较小：`56.67% > 53.33%`。
3. Stage64 当前 heldout 不能作为能力增长证据，因为 base、Stage1、Stage2 在这个 heldout 上已经接近满分。

## Stage64 当前问题

Stage64 这轮确实做了 option shuffle，也做了 gold label 重平衡和基于旧 `frontier_label` 的 train/heldout 重新切分，用来避免 gold label 分布偏置。但这不是“重新用 solver 评估后再筛 frontier”。

shuffle 之后，如果继续沿用 shuffle 前的 `solver_eval` 和 frontier 标记，就会出现一个问题：

```text
样本的选项位置变了，
但 frontier/heldout 难度标签没有重新用固定 solver 评估。
```

因此当前 Stage64 heldout 已经不是严格的 Stage2-calibrated frontier。它可以用于检查格式、泄漏、gold label 是否平衡和 clean retention，但不能用于证明 Stage64 solver 超过 Stage2 solver。

## 正确闭环

Stage64 后续应按这个流程重做：

```text
1. 生成 Stage64 API self-play 样本
2. option shuffle
3. 清空或忽略旧 solver_eval
4. 用固定 Stage2 solver 重新评估 shuffle 后样本
5. 根据新的评估结果重新筛 frontier / hard0 / easy5
6. 从重新筛出的 frontier 切 train / heldout
7. 训练 Stage64 solver
8. 在新的 Stage64 heldout 上比较 Stage64 solver 和 Stage2 solver
9. 同时在 Stage1/Stage2 heldout 与外部合并集上检查遗忘
```

论文里可以把 Stage1/Stage2 作为已经成立的逐阶段增长证据；Stage64 在完成重新校准前，应描述为“题目类型多样化模块与诊断结果”，不要直接写成已经证明第三阶段能力增长。
