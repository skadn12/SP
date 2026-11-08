# Self-play 当前工作线

本仓库整理长期对话扩展、多选与排序题生成、Solver 反馈、数据准入、LoRA 训练和评估所需的当前工作线源码。源码与合同按原路径和原字节保留；部分依赖仍位于历史命名目录，不应仅凭目录名判断其是否在当前流程中使用。

## 代码入口

| 内容 | 入口或目录 |
| --- | --- |
| 对话扩展与记忆、Planner、Gate、Realizer | `scripts/current_method/restart_n5_spell/simple_append_memorax_v1_memory_event_continuation_probe_v1/` |
| QA、反馈、呈现和质量检查 | `scripts/current_method/` 与 `scripts/current_method/restart_n5_spell/` 中相应实现 |
| F/G 出题和审查 | `research-protocols/selfplay_week_20261019/training_set_rebuild_20261104_run01/`、`training_set_rebuild_20261105_FG_exploration_run01/` 和 `training_set_rebuild_20261105_F_supplement_run01/` |
| 结构探索与来源审查 | `research-protocols/selfplay_week_20261019/training_set_structure_pilot_plan_20261106_v1/` |
| R2 候选准入、训练与验证 | `research-protocols/selfplay_week_20261019/training_set_rebuild_20261108_overnight_run01/final_training/` |
| 训练器 | `train_qwen3_lora_sft.py` |
| Formal Solver | `research-protocols/selfplay_week_20261019/training_set_rebuild_20261108_overnight_run01/solver_eval_20261108_strict128_formal/` |
| MultiRC/BAMBOO smoke | `external_eval/smoke_eval.py` |

这些路径是源码索引，不是新的运行指令。应按目标实验的冻结合同、manifest、输入哈希和实际完成记录确定调用链。历史实验的目录名、脚本或合同不构成重新运行的授权。

## 当前研究约束

训练集建设需要同时关注题面自然度、任务结构多样性、与历史题的语义重复，以及 Solver 反馈的作用。来源身份独立、题目重复和操作覆盖是不同的检查维度；结构普通不等于题目质量不合格，也不能将未知历史状态自动转换为来源否决。最新方法口径见 [方法说明](docs/CURRENT_METHOD_20261108.md)。

已完成的 11 月 R2 训练使用固定的训练和验证输入；验证及模型身份必须按对应运行合同核对。本仓库不含这些数据、Gold、模型、adapter、账本、实际响应或凭据，不能单独复现该次运行。训练和验证的源码保持原始行为；清理仓库没有启动训练、生成或 Solver。

## 文件布局与外部输入

服务器工作区根不是 Git 仓库。原工作区的 `research-protocols/` 和 `offline-deliveries/` 与 `current_method_github/` 同级；这里按原工作区相对路径保存相关源码。历史脚本中的绝对路径、代码哈希和执行门禁未被改写。复现时需在授权环境中恢复正确布局及外部输入；直接从空目录运行可能因缺少文件或哈希不符而停止。

依赖声明见 `requirements.txt`，API 配置示例见 `configs/config.example.json`。凭据通过运行环境或单独的私有配置提供，禁止提交。MultiRC、BAMBOO 和模型文件由使用者按各自许可另行取得；BAMBOO 上游来源与哈希见 [方法说明](docs/CURRENT_METHOD_20261108.md)。

外部 smoke 的历史 `r1/r2` 标记对应早期 stage1/stage2，不等同于本次 M-clean/R2。其现有 runner 在 tokenizer 支持时设置 `enable_thinking=False`，因此不能把该 smoke 的输出描述为保留 CoT；本次没有更改这个历史评估协议或本轮训练验证协议。

## 来源与限制

- [扩展依赖来源](docs/EXPANSION_SOURCE_PROVENANCE_20261108.json)、[F/G 来源](docs/FG_SOURCE_PROVENANCE_20261108.json)、[R2 来源](docs/R2_SOURCE_PROVENANCE_20261108.json)。
- [本次清理范围和验证](docs/CLEANUP_20261108.md)。
- 实际数据、样本题、Gold、对话、私有审核、模型响应、权重、日志、运行现场、密钥及第三方未核实许可的文件未纳入当前文件树。部分历史脚本本身混有实际事实或审核，须在受控环境中单独保管。
- 旧文件从当前 `main` 文件树移除，既有 Git 历史仍保留；SP-820 为独立仓库，本次不修改。
