# 独立盲审要求（本轮Solver前）
只读分配的BLIND_QUESTIONS.json、其中的完整来源以及本说明。禁止作者目录、私有mapping/Gold/分析、反馈包、历史问题、模型结果。整份来源全读，保留读取回执；按每项事实独立推导Gold，不猜配额。

题目是自然直接问内容、选项直接回答问题。不要求日志/清单等包装；短题干若范围清楚不是失败。检查source_support,gold_truth,option_distinctness,distractor_relation,natural_concrete_task,source_dependence,r03_necessity,operation_alignment。每维度{decision:PASS|HOLD|FAIL,rationale:具体理由}。重复问法仅诊断，不因重复单独HOLD；无新增难度配额。MC4逐项证据单位清晰，Gold依赖完整来源，不能凭题面全部事实自答；MC5必须是原文明示理由/目的，不以推测动机充当来源原因。

输出INDEPENDENT_KEY_INITIAL.json：{reviewer_id,reviewer_is_human:false,gold_and_arm_blind:true,frozen_before_author_gold:true,reviews:[{review_id,independent_gold:[A..F],decision,options:{A:{selected:boolean,evidence:[{turn_ref,speaker,quote}],rationale},...F},checks:{上述8维度},issues:[],natural_task_description}]}。
证据逐字原文子串、说话者、坐标准确。每个选项都记录证据及解释，包括错误项。不能复制作者分析。必须记录具体R03必要关系，非只列引用。
完成并写入KEY_FREEZE.json（key文件path/sha256/bytes及timestamp）后再报告，不查看作者Gold。必要时作者一次修订后，追加REVIEW_REVISION1.json验证（保持初稿key和freeze，不覆盖）。没有Solver前结果可用。
