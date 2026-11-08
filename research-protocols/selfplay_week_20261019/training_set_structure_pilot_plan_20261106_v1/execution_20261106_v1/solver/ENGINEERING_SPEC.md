# 本轮评估接口实现约束

工程准备可在出题期间做；不得读取候选内容进行选择，不得生成请求、加载模型或推理，直至 controller 冻结全部质量、实际结构、历史重叠结果。

以 `training_set_rebuild_20261105_three_root_G_retry_eval_v1/model_baseline/adapter.py` 的三型、final-answer-only 函数及 `training_set_rebuild_20261106_wording_solver_eval_v1/adapter.py` 的最小适配方式为入口。复用其历史本地执行循环，不修改任何历史代码、来源、答案、评分或导出文件。

新 adapter 位于本目录。后续 controller 将提供 `INPUT_REVIEW_SEAL.json`，包括不可变文件绑定、每题 raw draft 路径与哈希、candidate_id、root、四组 group、quality_status、gold_agreement、独立结构/重叠结果与协议标记。质量 PASS 且 Gold 一致才进入评估；普通 D 类或历史重复绝不能当质量失败排除。流程有偏差的可保留描述性行为，但须带标记，不混作干净因果对照。

输入在内存规范化，不改首稿：兼容单对象 / candidates[]、`gold` / `gold_option_ids`、source_hash/source_sha256 等已有别名。partial_order 原稿保持 gold_option_ids=[] 与 gold_pairs，调用旧接口时仅在内存映射为 directed pairs。保留稳定 option_id、查询域与原题全部可见内容，不把作者证书/卡/Gold/历史结果放进 prompt。旧 arm gate 用代理值校验后恢复 OLD_F/OLD_G/NEW_F/NEW_G；root×group 最多一题。

固定 M-clean 和 tokenizer 见本目录 `MODEL_IDENTITY_AUDIT.json`，使用流式哈希再次校验已绑定文件。统一 Full/No-context、每型 min(6,k!) 个 Gold-independent 不重复展示 ×3次，engine seed2026110303，temperature0.8、top_p0.9、bf16，batch12，总窗口16384、输出上限6144。命名空间 structure_pilot_20261106_v1；具体显示seed和逐请求seed须在推理前冻结，并在 Full/No-context 配对复用。不自动重试、不补跑、不额外消融、不导出训练集、不训练。

统一可见格式说明应按本题 k 与实际类型生成。完整排序明确全部 k 项恰好一次，格式例子列全 k 个标签。partial_order 说明完整的所问有向对集合、包含蕴含的传递关系、不强排未约束对，并与原 query domain 对齐。multi_select 可见格式不暗示 Gold 数。保持历史简短证据/理由 JSON 接口及 final-answer-only 主评分；辅助字段格式问题不决定答题分。

CPU preflight 检查全部源/题/映射/展示/评分/格式例子和 token 长度，源绝不截断；超长明确 EVAL_HOLD 并冻结，不让整个批次静默丢失。推理前冻结精确代码、模型/tokenizer、来源/首稿/审查、prompt/schema、展示排列、逐请求seed、scorer 与一次性台账，最多1152调用。一次执行的中断要保留已返回 raw，未执行记 missing；不将基础设施缺失算模型答错，不擅自继续未完成调用。

多选/全序 J、偏序 directed-pair P/R/F1 分型报告，整题准确率另报。无效最终答案在固定已执行计划分母计0。不得混合 J/F1 做一个未经说明的总难度分。提供必要的离线语义 fixture（正确答案轮换回绑、错选/反序、偏序漏传递边、最终答案和辅助格式分离、重复/无效标签、固定输入篡改与台账不可重试）。工程测试不得产生实验模型调用或改历史 artifacts。

工程交付包括 adapter、必要离线验证、变更说明及输入seal格式。只进行 CPU 准备，返回后由 controller 在审查冻结后执行 prepare/run。
