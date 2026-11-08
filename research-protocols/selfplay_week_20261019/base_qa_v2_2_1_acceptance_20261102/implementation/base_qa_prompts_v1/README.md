# P0 基础 QA 出题与审核 v1

入口：`python -B -m scripts.current_method.base_qa_pipeline_v1`，在项目根目录执行。
这是新增的离线状态机和模型材料接口，没有网络、模型或训练调用入口。历史生成器和 runner 没有改道；复现历史继续使用原入口，新 P0 试验显式使用此入口。

每个 run 只容纳一个来源、一个关系和一道题。修订时新建 run，不能覆盖旧结果。F/G 仅保留运行元数据；二者共用来源、Prompt 和审核标准，均拒绝 QA 弱点反馈。旧扩充机制、完整来源、固定 Solver 和评分均未修改。

## 离线阶段

```bash
python -B -m scripts.current_method.base_qa_pipeline_v1 prepare \
  --selection /absolute/path/to/frozen_selection.json \
  --root cc-train-episode-109164 --arm F --run-dir /absolute/path/to/NEW_RUN

python -B -m scripts.current_method.base_qa_pipeline_v1 freeze-relation \
  --run-dir /absolute/path/to/NEW_RUN --input /absolute/path/to/relation.json

python -B -m scripts.current_method.base_qa_pipeline_v1 freeze-candidate \
  --run-dir /absolute/path/to/NEW_RUN --input /absolute/path/to/candidate.json

python -B -m scripts.current_method.base_qa_pipeline_v1 freeze-blind \
  --run-dir /absolute/path/to/NEW_RUN --input /absolute/path/to/blind_solution.json

python -B -m scripts.current_method.base_qa_pipeline_v1 finalize \
  --run-dir /absolute/path/to/NEW_RUN --input /absolute/path/to/comparison_review.json

python -B -m scripts.current_method.base_qa_pipeline_v1 prepare-display \
  --run-dir /absolute/path/to/NEW_RUN --seed trial_01 \
  --output-dir /absolute/path/to/NEW_DISPLAY
```

这些命令只准备或检查 JSON，**不会替用户调用模型**。每一步模型输出须在获准后由独立调用产生。`prepare` 生成 DISCOVERY_PACKET；冻结关系后才生成 AUTHOR_PACKET；冻结候选后才生成 BLIND_PACKET；冻结独立答案后才生成 COMPARISON_PACKET。盲审包选项也以不依赖 Gold 的方式重排，回答使用稳定 item ID。

`NO_SAFE_QA` 是关系发现的合法结果：`{"status":"NO_SAFE_QA","reason":"具体原因"}`。该 run 不再生成题目。

`finalize` 只有 `PASS_STATIC_REVIEW` 才生成 FROZEN_QA 和哈希回执；HOLD 返回退出码 2。HOLD、不一致、歧义、证据不足均不能通过增加某方权重自动解决。`prepare-display` 必须读取通过审核且哈希未变化的 FROZEN_QA；输出 PUBLIC_QUESTION 和 PRIVATE_MAPPING，后者不能传给 Solver。

程序还验证来源、来源选择、来源警告和实现/Prompt 的哈希。修改实现后，应创建新版本 run；不能拿新代码继续旧冻结 run。文件采用独占创建，避免静默覆盖。哈希和阶段约束是可追溯接口，不是进程级文件隔离；运行 Reviewer 时仍须提供独立会话和限制材料。

## JSON 契约

关系对象字段见 [discovery.txt](discovery.txt)。其中 `expected_answer` 在题型与选项之前冻结，不含 A/B 等位置；至少两个候选必要事实，每个都有精确引用。步骤只引用先前节点，全部事实必须进入最终结论。必要性组包含同一前提的重复表述，说明删组后哪个判断欠定。连通图只验证作者提出的结构，不证明关系自然或证据必要。

候选字段见 [author.txt](author.txt)。`relation_sha256` 使用 `production_question_gate.fingerprint(relation)`；`options` 以稳定 ID 为键，Gold 引用 ID；`item_analysis` 对每项记录语义答案、状态、依据及错误原因。单选恰好一个 Gold，多选至少一个真项和一个非真项，排序包含所有项且证据边构成唯一全序。不设题型比例、六选项或 Gold 数量配额。

数值答案使用 `{"value":"2/5","unit":"ratio"}` 等类型化对象。每个数值答案索引都必须被精确计算覆盖；支持有理数求和、差、乘积、比值及 ISO 时间间隔。单位必须匹配。代码验证运算和答案映射；数字是否从原文正确抽取仍由独立 Reviewer 审核，不能把普通算术检查当作来源真实性证明。

独立答案字段见 [blind.txt](blind.txt)。调用方在发出请求前提供实际 packet SHA 和固定身份元数据；响应需给独立答案、自己找到的引用、推理，以及显式空 `ambiguities: []` 才可能继续准入。模型标识当前仅被记录和校验非空，尚未由真实 API transport 回执认证；未来适配器须核对响应模型、请求材料和 usage，不应让模型自行填写可信身份。

比较审核字段见 [comparison.txt](comparison.txt)。九个语义检查必须分别有理由，不能从一个粗粒度 PASS 补齐。每个必要性组都需要独立判断删组欠定、单组不足、无冗余旁路。来源全文只出现一次，`removal_contexts` 中的 turn_ref 清单可按原序还原各组上下文。这是见过完整来源后的静态反事实检查，**不是盲删证据实验**。来源 warnings 可能复述被删除事实，未来 Solver ablation 不得直接使用该审核包。

通过标准只是 `PASS_STATIC_REVIEW`，输出始终标注 `empirical_quality_established: false`。需要真实生成、无上下文和独立删证据试验，才能进一步判断自然性、捷径和必要性。

## 批次诊断与错误分类

```bash
python -B -m scripts.current_method.base_qa_pipeline_v1 audit-batch \
  --input /absolute/path/to/candidates.json --output /absolute/path/to/NEW_AUDIT.json
```

输入为候选列表或 `{"candidates":[...]}`；必须是含题干、选项和 Gold 的有效记录。报告题型、Gold 数量、原始前 k 项模式、题干开头、引号选项及真假项长度。排序不纳入真假项长度对比。该工具不判断语义捷径可利用程度，也不为平衡分布修改正确答案。

`diagnose_response` 只增加错误诊断：无效标签、排序漏项、单选多答，以及外部核实了数值正确但选项错误的情况；不自动补答案，不替换或改变原有评分。新试验应保存多个展示 seed，并让 Full/No-context 对同一排列成对比较。

软件测试：`python -B -m unittest scripts.current_method.test_base_qa_pipeline_v1 scripts.current_method.test_production_question_gate scripts.current_method.test_question_presentation_gate`。其中合成样例仅是程序测试夹具，不是新数据集或模型生成的高质量 QA。
