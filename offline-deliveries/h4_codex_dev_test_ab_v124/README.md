# v124 验证集与测试集扩充出题

固定31个root：dev15、test16。逐root执行A3+B3，完整root目标为Gold2/3/4/5各一、Gold6对照一、排序两题。现有benchmark和训练池保持原样。本批仅生成和内容审核，不运行Solver。

先完成并封存dev，再按同一冻结规则处理test。各root可因真实来源或内容问题HOLD，全部纳入固定分母。所有成功和失败都保留在units/root*/。最终delivery/提供对话、题目、证据及总账。

原生B事件沿用MultiWOZ TRAIN池，root来自原固定dev/test划分；并非单独的MultiWOZ dev/test语料。test中4root有历史v106曝光，见evidence/EXPOSURE_DISCLOSURE.json。
