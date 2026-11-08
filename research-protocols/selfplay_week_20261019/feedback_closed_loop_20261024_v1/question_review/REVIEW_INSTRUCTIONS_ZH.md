# 独立内容审核（两阶段，臂别隐藏）

仅读分配的PUBLIC_PACKET.json以及其中绑定的完整来源、SOURCE_FREEZE.json warnings。第一阶段禁止打开AUTHOR_KEY_PACKET.json、私有mapping、作者目录、诊断卡、Solver结果。公开包隐藏F/G、作者、预期错误与Gold。按完整来源和题面，对每项支持/不支持或唯一排序边独立裁决，先落盘`INDEPENDENT_GOLD.json`，每题含review_id、independent_gold、item_judgments（每项源turn refs+具体理由）、needed_r03_refs、source_specificity、initial_quality_issues。回执记录完整来源hash/turn refs。不能只看引文或旧段落，不能把常识假设当来源明示。

第一阶段保存后才能打开AUTHOR_KEY_PACKET.json。逐项比作者Gold和证据，检查quote逐字、speaker、scope、state与R03必要性。需要按以下required checks落具体PASS/HOLD/FAIL+rationale：
- 全部：source_support、gold_truth。
- MC：option_distinctness；若Gold6用six_gold_control，其余用distractor_relation。
- 排序：operation_alignment、unique_total_order、mention_order、common_sense、editorial_cues、source_dependence。

排序要核第一相关贡献位置（而非仅作者所引的最早位置），clock排序不能混成现实日历行程；程序排序可含来源规定但未完成步骤，不能冒称发生；话语排序是实际贡献位置，顺序与mention一致可以是预期操作。明确六项无并列、五条边真实成立，不能机械图合法便PASS。至少一条必要边依赖R03。若展示项脱离对话也能仅靠通用流程全排，需HOLD。删除first/finally等编辑暗示，但不机械删真正必要语义。

MC逐个解释所有负例改变的关系，不能将未提到等同为明确否定。Gold2/3/4/5/6和操作配额仅机械合同，不是语义准入理由。每题至少一个必要新前提，去掉R03后该题完整Gold不再可确定。题面若要求单片段证据，某片段必须独立充分，不能借全文其他turn替它补齐。

写`REVIEW_DECISIONS.json`：
```
{"reviewer_id":"实际代理ID","reviews":[
 {"review_id":"RQ-...","decision":"PASS|HOLD|FAIL","independent_gold":[...],
 "reconstructed_gold":[...],"checks":{"source_support":{"decision":"PASS","rationale":"具体"},...},
 "r03_necessity":{"decision":"PASS|HOLD","rationale":"必要前提/边及反事实删去测试","turn_refs":[...]},
 "issues":[],"required_revision":[],"gold_discrepancy_resolution":null}
]}
```
必须保留初始独立Gold；若第二阶段发现自己推导需修正，说明具体证据，不能默默覆盖。不同意作者答案则HOLD并给可执行修订建议，不擅改Gold或题面。每槽最多一次作者内容修订，保留初稿和所有失败。无需读取历史实验、外网、调用模型或编造程序生成PASS；语义结论必须你逐题判断。共享文件系统/同Codex家族限制需披露，不能声称外部人审或严格模型独立。
