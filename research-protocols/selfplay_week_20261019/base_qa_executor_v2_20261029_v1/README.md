# 基础 QA 执行器 v2：离线交付

本目录为 `base_qa_executor_v2_20261029_v1`，独立于冻结的基础出题 v1 和 preflight。项目没有 Git 元数据，本次采用新模块、新目录、SHA-256 清单实现版本隔离，没有覆盖旧入口。

- 执行器：`scripts/current_method/base_qa_executor_v2.py`
- 持久预算：`scripts/current_method/qa_api_budget_v2.py`
- 单次 HTTP 适配：`scripts/current_method/qa_api_transport_v2.py`
- 完整说明：[REPORT_ZH.md](REPORT_ZH.md)
- 新预算：[CALL_BUDGET_V2.json](CALL_BUDGET_V2.json)
- 账本格式：[LEDGER_FORMAT_ZH.md](LEDGER_FORMAT_ZH.md)
- 最终验证：[offline_checks_v2_final/TEST_RESULT.json](offline_checks_v2_final/TEST_RESULT.json)
- 冻结清单：[DELIVERY_MANIFEST.json](DELIVERY_MANIFEST.json)

## 安全默认值

`run` 默认 dry-run：不构造 HTTP 调用器、不读取凭据、不调用 tokenizer、不占预算。`--live` 只是必要开关，仍须具备与实验清单绑定的批准文件和经外部证据确认的服务能力。随交付提供的两个模板均不可执行。当前状态 **HOLD**。

只允许三个已批准 root，每 root 一次候选机会、最多四个阶段。一次 live 试验共用唯一固定目录的 SQLite 账本，F/G 不能通过分别初始化目录而各用 12 次额度。新代码没有旧出题器回退、Solver、训练或扩充入口。

## 离线使用

在项目根目录运行。`--state-dir` 必须是尚不存在的新目录；以下命令只创建清单和返回 dry-run 计划，不开始出题：

```bash
python -B -m scripts.current_method.base_qa_executor_v2 init \
  --state-dir /tmp/base_qa_executor_v2_dry_check \
  --selection research-protocols/selfplay_week_20261019/base_qa_p0_20261027_v1/offline_smoke_v1/SOURCE_SELECTION_THREE_ROOTS.json \
  --arm G

python -B -m scripts.current_method.base_qa_executor_v2 run \
  --state-dir /tmp/base_qa_executor_v2_dry_check \
  --root cc-train-episode-109164
```

重跑离线测试需要新的输出目录，避免覆盖检查产物：

```bash
python -B research-protocols/selfplay_week_20261019/base_qa_executor_v2_20261029_v1/RUN_OFFLINE_CHECKS.py \
  --output /tmp/base_qa_executor_v2_offline_recheck
```

该测试入口安装禁止 socket/DNS 的 Python audit hook，并通过 `sitecustomize.py` 传给 Python 子进程。若出现被阻止的真实网络尝试，检查仍判失败；Mock HTTP 对象和虚构服务响应不算真实请求。

## 产物界限

`offline_checks_v2_final/MOCK_SYNTHETIC_ONLY/` 使用 v1 单元测试的合成对话和手写响应，包含测试用的 `FROZEN_QA`、请求、账本。它们不是三个真实 roots 上新生成的 QA，不进入数据集、不作为模型质量证据。`REAL_SOURCES_DRY_RUN_ONLY/` 只绑定原来源并返回 dry-run，未生成关系或题目。

`offline_checks_v1/` 是接入过程中的先前离线检查；最终交付以 `offline_checks_v2_final/` 和交付 SHA 清单为准。先前目录不用于恢复 live 运行。

本次未创建 canonical `live_state`，没有已授权的 live 账本，没有读取学校凭据内容。得到单独明确批准、解决服务能力证据和 tokenizer 兼容性后，才能准备真实试验；本交付不自动启动。
