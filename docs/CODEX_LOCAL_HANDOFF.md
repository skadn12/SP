# Local Codex Handoff

最后更新：2026-08-10

这份文档把服务器上的当前 self-play 项目交接给本地 Codex。它描述的是
**当前工作快照和已知失败现场**，不是“整个服务器工作区已经备份”。

## 1. 代码仓库

GitHub：<https://github.com/skadn12/SP>

交接文档创建前的代码基线提交：

```text
ec8c30aec6dab0452a4e6bb6f510bae739788dc0
```

克隆：

```bash
git clone https://github.com/skadn12/SP.git
cd SP
```

本地部署说明：[`LOCAL_SETUP.md`](LOCAL_SETUP.md)。

GitHub 中包含当前源代码、测试、`data/` 精选数据、配置模板和实验支线登记。
以下内容按设计没有进入 GitHub：API key、Memorax credentials、模型权重、完整
run 目录、缓存和未整理的报告。

## 2. 服务和本地资源

当前扩充/QA 编排仍使用远程服务：

```text
Planner / Realizer / QA Writer：配置的 OpenAI-compatible API
Memorax：配置的 HTTP API
```

本地需要自己创建未跟踪的 `config.json`，参考：
`configs/config.example.json`。Memorax 凭据单独配置，不要写入 Git。

Solver 可以本地运行。已上传的完整 merged Solver 位于：

<https://huggingface.co/xinyigong/oldSP>

本地已有训练模型的服务器路径是：

```text
/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/training/models/merged
```

训练数据 Dataset（私有）：

<https://huggingface.co/datasets/xinyigong/SP>

Dataset 当前目录：

```text
curated_data/
solver_training/stage2_repair_topup/
expansion_inputs/
freeze_manifests/
```

Qwen 和 Memorax 仍可用 API，因此本地不必下载 Planner/Realizer 模型权重。

## 3. 当前主支线

当前需要继续观察的代码模块：

```text
scripts/current_method/restart_n5_spell/simple_append_memorax_v1_memory_event_continuation_probe_v1
```

链路：

```text
Fresh Parent
→ Memorax Parent Add / Bucket Search
→ concrete event discovery
→ transcript verification
→ Planner
→ Semantic Continuation Gate
→ candidate selection / isolation
→ one-call Round Realizer
→ Child Add
→ post-generation Bucket Search
```

这条支线明确不包含：Event Chain、State Ledger、QA、Solver、R2/R3 和 Training。
每个 episode 必须使用新的 Memorax namespace；不要复用已经写过 Parent 或 Child 的
namespace。

默认模块自带的示例 namespace 是旧测试值，只能用于离线 preflight；真实运行必须
显式传入新的 namespace 和新的 run root。

## 4. 最近一次 fresh multi-episode 结果

汇总报告（服务器只读现场）：

```text
/root/autodl-tmp/gxy-selfplay/restart-n5-runs/
memory_event_continuation_robustness_v1_batch_1/reports/
MEMORY_EVENT_CONTINUATION_ROBUSTNESS_BATCH_REPORT.md
```

结果：

```text
fresh episodes                 3
event discovery >= 3           3/3
Planner completed               3/3
Gate complete                  2/3
Realizer called                2/3
Realizer contract success      0/2
Child Add                      0/3
Post Search                    0/3
candidate isolation failures   0
memory visibility verified     0/3 (未到达)
end-to-end pipeline success    0/3
```

三个只读 run：

```text
/root/autodl-tmp/gxy-selfplay/restart-n5-runs/
conversation_chronicles_episode_12596_memory_event_continuation_robustness_v1_batch_1

/root/autodl-tmp/gxy-selfplay/restart-n5-runs/
conversation_chronicles_episode_178392_memory_event_continuation_robustness_v1_batch_1

/root/autodl-tmp/gxy-selfplay/restart-n5-runs/
conversation_chronicles_episode_216742_memory_event_continuation_robustness_v1_batch_1
```

逐 episode 结论：

| Episode | 结果 | 真实失败点 |
| --- | --- | --- |
| `12596` | Gate 通过 | Realizer 只返回 1 scene / 9 turns，要求 3 scenes |
| `178392` | Gate 合同失败 | 6 个 review 缺少 `reason`，不是合法语义 reject |
| `216742` | Gate 通过 | Realizer 返回 `[9, 8, 8]`，后两场低于 hard minimum 9 |

这批实验没有产生合法 Child，不能把其中的 Realizer 输出当作训练数据，不能
执行 Child Add 补写，也不能在原 run 上 repair 或 resume。

## 5. 下一步建议

先做零 API 诊断，不要立即再次 live：

1. 检查 `178392` 的 Gate 原始响应，确认 `reason` 是模型漏字段还是本地合同过严。
2. 检查 `12596` 和 `216742` 的 Realizer prompt、materialized input、原始输出和
   parser，确认是模型没有遵守 3-scene/9-turn 约束，还是计数/归一化问题。
3. 只允许必要的机械合同修复；不要改变 Memorax discovery、Planner 发散、Gate
   语义标准或 candidate isolation。
4. 通过离线测试后，使用新的 episode 和新的 namespace 做一次新的 bounded R1；
   不复用上述失败 run。
5. 只有 Realizer 合同通过后才做 Child Add，再检查 Post Search 是否看到新发展。

不要因为这批失败就进入 R2/R3、QA、Solver、Training 或批量生产。

## 6. 运行前检查

从仓库根目录运行包模块，不要直接执行 `.py` 文件：

```bash
python -m scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.live \
  --preflight-only \
  --allow-insecure-memorax-http
```

确认：

```text
API calls = 0
Memorax calls = 0
Solver calls = 0
run root 不存在或是全新目录
namespace 不曾写入 Parent/Child
```

真实运行后保存：

```text
run_freeze_manifest.json
run_status.json
request ledger
raw responses
prompt/schema manifests
validation results
summary report
```

## 7. 重要保护规则

- 历史 v6-v10、v11 probes、QA支线和上述 batch run 只读。
- 失败 run 不删除、不覆盖、不 resume、不把失败响应变成 fixture 成功。
- 不把 `config.json`、Token、模型权重或原始 Memorax 响应提交到 Git。
- 代码改动只进入新的 sibling module，除非明确确认是主线修复。
- 每个新支线先登记到 `EXPERIMENT_BRANCH_REGISTRY.md`。
- 运行结束要记录 input SHA、代码 SHA、namespace、API 次数和失败阶段。

## 8. 服务器现场和远程备份

完整服务器工作区没有整体备份。当前最重要的外部现场是：

```text
/root/autodl-tmp/gxy-selfplay/restart-n5-inputs
/root/autodl-tmp/gxy-selfplay/restart-n5-freezes
/root/autodl-tmp/gxy-selfplay/restart-n5-runs
/root/autodl-tmp/gxy-selfplay/eval_reports
```

它们应单独同步到本地磁盘或对象存储；GitHub 只负责代码版本，不能替代运行
现场和 Memorax bucket 快照。
