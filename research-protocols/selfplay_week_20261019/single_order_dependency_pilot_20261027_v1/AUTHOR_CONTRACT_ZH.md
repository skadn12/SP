# 单选与排序关系诊断合同

这是已有多选质量诊断的独立分支，来源不变，不做F/G对比或训练。每root尝试一道单选和一道排序，共六题；先读完整来源和warnings，找真实关系，再冻结TASK_PLANS。无可靠关系应HOLD，不添加原文没有的事实，不按Solver成绩改题。

单选为single_choice，4个自然、信息量和措辞相近的选项，仅一个正确。使用稳定item_01等ID，Gold为包含一个ID的list。不能使用“以上皆是/皆非”，题干不复述决定答案的事实；避免正确项独有详细限定、假项明显荒谬，或靠生活常识即可唯一选择。目标是需要至少两个来自不同scene的必要前提才能裁决选项，而非把旧多选真项随便留一项。

排序为ordering，4–6个实际存在、可唯一确定完整顺序的事件/计划动作/比较对象，Gold为全部稳定ID的有序list。数量按来源关系决定，不强凑长度或类型比例。题干明确按发生、约定执行、最终计划，或来源支持的具体数值排序；不能混淆计划与完成，不能从文本出现顺序推断事件时间，不能把独立日历强行连成同一天。选项文本不直接给出其排列用日期、钟点、序号或“首先/然后”等提示，也不使用A导致B这类常识顺序即可确定整题的明显链。允许真实的约定依赖，但必须有对话特定关系。展示前打乱，不按Gold存序作为模型标签。

所有题JSON字段：candidate_id、root_id、question_type、author_id、source_binding、question、options{stable_id:text}、gold[list]、item_analysis{stable_id:{should_select:bool或ordering中为True,rationale,evidence:[{turn_ref,speaker,quote}]}}、dependency_certificate。

dependency_certificate包含 premise_groups[{group_id,turn_refs,premise,why_necessary,removal_effect}]、joint_inference、single_group_insufficient:true、redundant_turn_refs:[refs]。single_choice额外target_option（正确项）；ordering额外target_pair:[before_id,after_id]，这一对的顺序须通过两个以上必要前提联合确定，不能每个前提各自完整说明pair顺序。排序仍应审核整条序列唯一性，不能只审这一对。所有groups跨至少两个scene，refs按source顺序且互不交叠；冗余复述必须归入同一删除组。删组后目标应在逻辑上无法确定，单给任一组不足推出目标；若还可从其余文段恢复则补组或HOLD。引用数量不证明必要性。

独立审核先只看question/options/source/warnings推Gold并冻结，然后才读作者Gold/证书。审核检查gold_truth、source_support、joint_necessity、redundancy_bypass、question_options_shortcut、distractor_style（排序看item_style和无显式顺序提示）、not_independent_fact_bundle、ordering_total_order（排序）。每题review记录candidate_sha256（规范JSON fingerprint）、reviewer_id!=author_id、gold_blind:true、independent_gold、decision、checks{名称:{decision,rationale}}、ablation_review[{group_id,removed_turn_refs,target_underdetermined_after_removal,rationale}]。PASS只是语义审核通过，不是实证质量认证。

至多一次源依据修订，保留draft/review/HOLD/修订版本。冻结后用Full、No-context、Parent(pre-R03)、Extension(R03)、Evidence-only、删每组、只给单组诊断。每题三种与Gold无关排列；同题各排列和条件使用同一采样seed，不能把三个展示当独立n。结果只按最终答案评主分，引用合规另列；排序报完全正确、pairwise accuracy、LCS/n。消融分数是保留Full答案的诊断，不是删后重新判Gold的正确率。不得把Full低分、No低分、引用通过或删组下降单独当质量成功。
