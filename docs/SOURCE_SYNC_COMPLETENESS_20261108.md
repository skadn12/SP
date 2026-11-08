# 2026-11-08 相关源码补齐

首轮提交 `0f0f75d` 是选定主线快照，并没有包含所有相关源码。
本轮补齐此前遗漏的实现、测试、通用提示词、作者/盲审合同、schema 和方法说明，
在现有 `skadn12/SP` 项目中发布，并将同一完成提交同步到发布分支与默认 `main`。

## 核对范围与文件清单

| 来源 | 本轮处理 | 逐文件 hash 和排除记录 |
| --- | --- | --- |
| 主仓库 `scripts/current_method/restart_n5_spell/` 及遗漏测试 | 补齐原先未跟踪的实验实现和测试，包括 writer comparison、反馈、置换、消融等包 | [REPO_SOURCE_COMPLETENESS_20261108.json](REPO_SOURCE_COMPLETENESS_20261108.json) |
| 工作区 `research-protocols/selfplay_week_20261019/` | 补齐周目录内自写源码及通用方法资源，包括 F 补充、结构探索、候选准入、Solver 前置及正式处理工具 | [WEEK_SOURCE_COMPLETENESS_20261108.json](WEEK_SOURCE_COMPLETENESS_20261108.json) |
| 工作区 `offline-deliveries/` | 补齐各实验的通用构造、评估、审核与交付工具，保留不同实验版本的实现 | [OFFLINE_SOURCE_COMPLETENESS_20261108.json](OFFLINE_SOURCE_COMPLETENESS_20261108.json) |
| 工作区根、`scripts/`、`external_eval/`、`tests/`、`stage3-data-fix/`、`mainline/code/`、`memory-tests/`、`recovery-tools/` | 补齐支持源码，并与仓库已有同名文件逐字节比较 | [WORKSPACE_SOURCE_COMPLETENESS_20261108.json](WORKSPACE_SOURCE_COMPLETENESS_20261108.json) |

每项清单记录纳入文件的发布路径、SHA-256 和大小。少量脚本同时包含通用算法与真实事实、Gold 或审核常量；为保留源文件原貌，这类混合脚本整体排除并在清单中记录，依赖它们的历史运行仍需原现场文件。补齐依据文件内容及调用关系，
不只检查文件是否已被 Git 跟踪，也不只按最近修改时间挑选。
历史实验实现收入代码归档，不代表它们被本轮训练调用或被提升为当前方法。

## 保持实现和来源关系

提取的源文件保持原始字节，不修改出题路线、准入规则、Gold、评分公式、模型、
训练配方或调用门禁。主仓库版本保持为规范位置；工作区中同名但内容不同的镜像，
保存在 `archives/workspace-source-variants/`，并记录来源，不覆盖规范版本。
本地原文件、运行现场和未跟踪数据均保留。

工作区根不是 Git 仓库。原工作区的 `research-protocols/` 与 `offline-deliveries/`
在 `current_method_github/` 旁边；本代码包把它们按原相对目录保存在仓库内。
复现历史运行需恢复相应工作区布局、绝对路径和冻结输入。只添加源码不会使
缺少数据、manifest、模型或预算账本的历史运行自动可用。

## 明确排除的内容

- 训练、验证、测试数据及具体候选题、Gold、对话、实际审核结果、模型请求和响应。
- 仅把上述实际数据写成 Python 字面量再落盘的物化脚本；不能只因扩展名为 `.py`
  就把它们当成通用实现。通用算法和合成测试 fixture 则纳入源码。
- 密钥、私有运行配置、日志、结果报告、checkpoint、权重、缓存、虚拟环境、第三方插件。
- 冗余冻结代码副本、临时执行目录、备份，以及独立 Git worktree 和 `SP-820-publish`
  的另一个项目快照。它们不覆盖本次规范源码位置。
- 已识别但尚未核实再分发许可的第三方参考源码；对应来源与哈希见清单。
- 首轮整理时产生的 `training/r2_20261108/` 临时副本：其 13 份源码与已发布的原目录
  内容相同，不重复收录。

具体排除路径及原因见对应清单。本次是上述范围的源码同步，不是服务器文件系统备份；
仓库既有 `data/` 历史资产仍保留在 Git 历史中。

## 验证

本轮验证记录见 [SOURCE_SYNC_VALIDATION_20261108.json](SOURCE_SYNC_VALIDATION_20261108.json)。
验证包括源码语法、清单与 Git index 的 hash/大小、源文件字节一致性、凭据扫描，
以及在独立源码目录中执行所选合成和 mock 测试。全量历史集成测试仍需其外部输入，
不把静态编译或部分测试通过表述为所有历史实验已复现。

本次没有启动出题、Solver、训练、外部基准或已取消的过夜任务。
归档的历史合同和授权说明不构成重新执行这些任务的授权。
