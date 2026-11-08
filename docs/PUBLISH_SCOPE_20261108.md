# 2026-11-08 当前方法源码发布

发布目标是现有项目 `skadn12/SP`，分支为
`publish/current-method-r2-20261108`。这是当前研究实现的增量源码快照；
`main` 保持原提交，便于审查后再合并。

## 收录范围

- `scripts/current_method/`：对话扩展、QA 生成、证据和呈现门禁、反馈处理及对应测试；
  补齐扩展入口实际 import 的 feedback compiler、renderer 和 QA prompt。
- `scripts/codex_*.py` 及测试：离线题目构造和不可变审核工具；补齐动态依赖的 public parser。
- `research-protocols/selfplay_week_20261019/`：11-04/11-05 F/G 作者合同、
  批次准备和筛选/导出源码，以及 11-08 R2 训练工具和其 manifest 绑定的源码依赖。
- `external_eval/`：MultiRC/BAMBOO 独立 smoke runner、BAMBOO 官方 prompt 与来源清单。
- 配置、依赖声明和代码说明。W&B 属于训练依赖，凭据由运行环境提供。

新增提取文件的来源和 SHA-256 见同目录 `*_SOURCE_PROVENANCE_20261108.json`。
源代码保持现场字节，不调整出题路线、准入标准、Gold、评分公式、模型或训练配方。
README 下半部分的 Stage3 说明属于历史文档。

## 目录及复现限制

服务器工作区根目录不是 Git 仓库；代码仓库位于其 `current_method_github/`。
原工作区的 `research-protocols/` 与 `offline-deliveries/` 是代码仓库的同级目录。
本次把提取的源码按这两个目录的原相对结构保存在仓库内，便于审查和传输。
例如本轮训练源码位于：

```text
research-protocols/selfplay_week_20261019/
  training_set_rebuild_20261108_overnight_run01/final_training/
```

复原原现场时，应把仓库中的 `research-protocols/` 和 `offline-deliveries/`
源码映射到工作区同级目录，再单独准备原始输入和 manifest。训练工具保留了
原现场的绝对路径及冻结 hash；直接换目录执行会因路径、输入或 hash 不符停止。
这份快照不声称可以在空白机器上直接执行完整实验，不为迁移方便改写冻结合同。

本次新增提交不含训练/验证/测试数据、生成题目、模型响应、运行日志、结果报告、
checkpoint、模型权重、缓存、虚拟环境或密钥。仓库原先已经跟踪的历史 `data/`
和历史文档保留，因此整个 Git 历史及完整分支归档仍含原有历史资产。
本地其余独立实验目录保持原样，未将它们全部视作当前方法的依赖。

## 离线检查

逐项验证记录见 [PUBLISH_VALIDATION_20261108.json](PUBLISH_VALIDATION_20261108.json)。

- Codex 批次/证据/顺序门禁：55 项通过。
- QA 生产、呈现、反馈、API transport/budget：80 项通过。
- 可独立运行的 base-QA pipeline、schema、option freeze 和 mock transport：199 项通过。
- 扩展主模块在仅含 Git index 的独立目录中，8 个模块导入通过；禁用网络和子进程。
- 发布源码编译、JSON 解析、来源 hash、凭据扫描在提交前核查。

历史 fixture 集成测试未全部通过：部分测试依赖未发布的实验来源、候选、旧审核文件和
SQLite 预算账本。未伪造这些输入或修改门禁来使测试通过。通用 `unittest discover`
还需正确的 package 上下文；按模块执行的上述自包含测试通过。两份按原字节提取的
QA revision 源文件保留了原有文件末尾空行，属于已知 whitespace 提醒。

本次没有启动生成、Solver、训练或新的基准评估。历史协议和授权记录用于解释源码，
不代表重新授权启动已取消或已结束的任务。
