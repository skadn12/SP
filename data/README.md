# 数据说明

这个目录保存当前方法实验线路的可复现数据子集。它不是完整 29G 工作区镜像，而是围绕论文/复现实验需要整理出的主线数据包。

## 包含内容

- `locomo10.json`：原始 LOCOMO 数据入口。
- `base_restart_current_method_validation/preprocess_locomo_current_method/`：预处理摘要。
- `base_restart_current_method_validation/seed_parents/`：切分后的 parent / seed rows。
- `base_restart_current_method_validation/raw_audit/`：原始数据与 seed provenance 审计。
- `base_restart_current_method_validation/accepted_models/`：stage1 / stage2 accepted solver 记录。
- `base_restart_current_method_validation/reports/`：stage1、stage2 总结与审计报告。
- `base_restart_current_method_validation/states/`：stage2 相关 controller state。
- `base_restart_current_method_validation/00_smoke50_check/`：初始 smoke 数据与验证报告。
- `base_restart_current_method_validation/01_mining500_train/`：stage1 mining、frontier、train mix 和训练摘要。
- `base_restart_current_method_validation/02_stage2_smoke50_with_stage1_solver/`：使用 stage1 solver 的 stage2 smoke 数据。
- `base_restart_current_method_validation/03_stage2_guarded_mining300/`：stage2 guarded mining 主数据、frontier、eval 和报告。
- `base_restart_current_method_validation/04_stage2_train_from_stage1/`：stage2 train mix 与训练摘要。
- `base_restart_current_method_validation/05_stage2_repair_topup_from_stage1/`：stage2 repair / topup 数据、frontier 和 train mix。
- `base_restart_current_method_validation/64_qa_operation_answer_type_anchor_diversification_api_resume120/`：Stage64 API self-play 数据。
- `base_restart_current_method_validation/64_qa_operation_answer_type_anchor_diversification_probe/`：Stage64 题目类型多样化 probe 数据。

## 没有包含什么

- 模型权重、checkpoint、adapter、merged model。
- 大型训练缓存和本地运行缓存。
- 完整 29G 历史实验工作区。
- 超过 GitHub 普通仓库单文件限制的候选池，例如：
  `base_restart_current_method_validation/01_mining500_train/mining/topup_combined_candidate_pool.jsonl`
  约 121M，未放入仓库。

## 文件清单

`DATA_MANIFEST.tsv` 记录每个已纳入文件的大小、SHA256 和相对路径。

