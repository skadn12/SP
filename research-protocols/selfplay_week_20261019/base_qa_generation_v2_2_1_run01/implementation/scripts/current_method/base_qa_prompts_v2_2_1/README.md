# V2.2.1 离线出题契约

从 V2.2 七阶段 Prompt 独立复制并修订，保留事件身份、L3/L4、必要证据、全部替代路径、Gold、自然性、干扰项及来源准入要求。新增 `blind.txt` 严格 typed-answer 协议。

答案分四层：Discovery 即冻结的 `query`；TARGET 的 `semantic_answer:{kind,value}`；Construction 的稳定 `option_id` 与 `gold_answer`；Renderer 一次冻结的展示标签映射。每项 `semantic_value` 是原始 JSON 含义，不再用内部列表包装限制题型。语义支持 scalar/object/set/sequence/relation_set，题型支持单选、多选、完整排序和部分顺序。

允许全局多个解，但当前查询答案必须唯一；完整排序仍须唯一完整序列。查询必须单独重做依赖删除，不能凭无关全局变化继承 L3/L4。每个计入难度的操作还须证明对同一查询有实际贡献：删除跨变量耦合、跳过状态更新或去除新增前序推导后回放，答案不变的操作不得抬高难度。未知事实先后不是 false，量词 all/any 与自然信息需求保持一致。

`base_qa_schema_v2_2_1.json` 根定义校验 Discovery，可复用 `$defs/target`、`candidate`、`query`、`semantic_answer`、`task_answer`、`blind_response`。Schema 只约束结构；来源准入、语义、必要性、唯一性、哈希与映射需程序核验和独立盲审。Unsupported 关系保留诊断，不强塞三类执行规则。

无题型配额、固定题干或选项槽位；自然信息需求开放搜索补充 typed sweep。没有可靠关系或自然形式可输出 NO_SAFE_L3_L4／NO_SAFE_QA。离线 fixture／历史回归不批准正式素材，也不证明真实题目多样性、认知任务丰富性或 Solver 难度。
