三模型评估使用 v15 原生排列、提示、解码和评分；仅如实更换 Codex 来源声明，并注册 v127 已核验、已合并的 round2 模型。

M0、M_clean_policy、M_v125_newonly 使用同一份 DEV 后 TEST 请求表。多选每题 6 个随机展示排列 × 3 次采样；排序每题 12 个随机展示排列 × 1 次。固定 49 题目标时，每模型 798 调用，三模型 2394 调用。内容审核不通过的缺额保留，不用 Solver 分数筛题。

先由 QA 流程完成并冻结两 split，在本批次根目录写 QA_FREEZE.json：status 为 PASS/FROZEN/QA_FROZEN，files 是带 path、sha256 的完整文件绑定，必须包含 delivery/{dev,test}/QUESTIONS_WITH_ANSWERS.json 和对应 normalized CUMULATIVE_SOURCE.json。若同时提供 split/global EVALUATION_POOL.json，也必须冻结且与最终题目完全相同。路径可以为绝对路径或相对本批次目录。

CPU/tokenizer 预检命令：

```bash
/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python /root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_codex_held_roots_eval_v134/evaluation/evaluate.py prepare
/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python /root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_codex_held_roots_eval_v134/evaluation/evaluate.py verify
```

独立审核者核对全部来源、语义审核、固定候选、无 Gold 泄漏、模型身份、共享请求和工具封存，然后写 evidence/INDEPENDENT_EVALUATION_PREFLIGHT_AUDIT.json。该文件必须 status=PASS，audited_files 必须绑定本次 evaluation/PREFLIGHT_FREEZE.json；路径相对本批次目录或绝对路径。

独立审核通过后才可执行一次：

```bash
/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python /root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_codex_held_roots_eval_v134/evaluation/evaluate.py verify --require-independent
/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python /root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_codex_held_roots_eval_v134/evaluation/run_all.py
```

run_all.py 顺序执行 base/clean/new 子进程，每个模型退出后再加载下一个；已启动的侧不自动重跑。可单独运行 evaluate.py live --side base/clean/new，但同样强制独立预检通过并禁止重复启动。两训练模型路径均已合并 adapter，不再次加载 adapter。

CPU 报告可独立重建到新目录，不触发推理：

```bash
/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python /root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_codex_held_roots_eval_v134/evaluation/report.py --output-dir /tmp/v134-report-review
```

默认结果在 evaluation/reports/：逐调用、逐题 JSON/CSV、31 个固定 heldout root（无题保留 null）、DEV/TEST/type/Gold 分层、解析的全选基线和三组配对差值。缺失、重复和非法调用 exact=0，其他不确定指标保留上下界；不以有效调用替换计划分母。按调用、题目等权、已有题 root 等权分别命名。

Gold6 是全选对照；全选在 Gold2–5 的 exact 和 signed partial 均为 0。排序同时报告 exact 与 pair accuracy=(correct pairs)/15，signed partial 则为 (correct−wrong)/15，两者不同。排列、重复采样、Gold 家族和同 root 题目互相关联。该增量描述性评估不证明扩充或训练的因果收益，也不将对话出现顺序视为复杂真实时间推理。

prepare/live/report 均不改旧批次、旧题、旧权重或训练池。原生 selected_questions 仅作为历史兼容产物保留，完全不用于本次筛选。
