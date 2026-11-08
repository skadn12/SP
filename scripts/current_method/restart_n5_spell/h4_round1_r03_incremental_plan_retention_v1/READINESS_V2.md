# 增量扩充运行准备 v2：T1-T7

实现身份 `incremental_run_readiness_v2`；policy 仍为
`expanded_single_plan_incremental_retention_v1`。本次不改变计划目标、一次补充、
候选选择或 retention/admission 的语义。旧 README、旧 preflight/live_4 是历史说明，
新检查入口是 `readiness.py`，不能以旧 234 项报告替代本次结果。

## T1：角色容量与数据限制

供应商容量依据是用户提供的 Qwen/Qwen3-14B 页面“128K”说明。
没有保存的截图位图，也没有建立服务端精确整数证据；采用本地保守 128000，
不把本地 tokenizer 的 model_max_length=131072 当成供应商服务容量。
公开页面查阅没有认证 API、容量探针或 tokenizer 下载。

旧实跑 execution_plan 明确同一 Qwen/Qwen3-14B、temperature=0.25、timeout=600。
新提案按角色分别冻结，真实入口要求 model 与每个 role_capacity 对应。
thinking=false，未新增 thinking_budget 请求字段；思考额外预留为0。

| 角色 | 输入上限 | 输出预留 | 本地安全余量 | 工作窗口 |
| --- | ---: | ---: | ---: | ---: |
| 正常 Planner（本次0次） | 119552 | 8192 | 256 | 128000 |
| 增量 Planner | 119552 | 8192 | 256 | 128000 |
| Gate | 125696 | 2048 | 256 | 128000 |
| Realizer | 119552 | 8192 | 256 | 128000 |
| adherence | 123648 | 4096 | 256 | 128000 |

256是明确保留的本地服务包装不确定性余量，不声称是服务端实测开销。
messages 自身的 chat template 已整体计数一次；没有再扣第二份输出或思考预算。
服务器 8192/128000 接受情况仍为 NOT_TESTED。

累计对话限制提案仍填100000，但新单位明确是本地 tokenizer 的规范化对话 JSON
token 数，不把旧100000字节估界改写成历史实际token数。该数据额度也需一并授权。
Solver、QA、训练数据长度合同没有读取为API默认、更没有改写或放宽。

## T2：无损输入投影

保留完整原方案一份：3个continuation、8个new-development候选，目标、beats、
原文本不改。完整 candidate_registry 仍保存在后台并供 Gate/materializer 使用。
模型视图只把按类型/内容完全相等的重复值换成同一 INPUT DATA 内的 `$input_ref`。
稳定ID、VT/ME/人物绑定、依赖、状态来源以及不相等的独有字段保留。
字段映射覆盖每个叶字段；重建结果与源对象严格相等，不以“看起来相似”去重。

143条历史按连续组和显式列名渲染，原 turn ref、speaker、text、顺序与轮次保留；
历史中本来重复的话也保留。原始 scene_ref 在已有规范化父对象中编码于 turn_id，
另交付逐turn原始 scene/session 与模型可见编号的对应核验，不另造事实索引或摘要。
纯后台路径只进sidecar；新增空R03状态明确为[]。共享反馈文本与输出字段不重写。

新旧包装 token 对照 A/B/C 统一使用同一历史、身份和预算数据：
A为可按旧SHA复现的空反馈；B只补同一原真实反馈；C在B基础上无损投影。
另列新待授权预算/身份下的最终messages计数，不把A到C全部算作去重收益。
无token压缩指标，也不据长度推断模型质量改善。

## T3：正式反馈链

`bundle.build_failed_bundle -> run -> workflow -> render_supplement -> transport`。
正式构造不依赖 fixtures；fixture反向复用正式构造，仅追加mock标签。
以原 root/source candidate/cluster/observation snapshot 锁定历史反馈；读取原实际
Planner请求清单与prompt，并复用现有共享compiler，要求逐字段内容相等。
`validate_bundle_feedback` 在入口及workflow中发送、账本创建、机会消费之前执行。
删除了缺失时静默补{}；缺items、身份/SHA错配等fail-closed。
本来没有的旧证据、M0或分子分母不编造；没有把此要求扩散到Fresh。

完整反馈包含原ordering题、4个items、历史Gold、4条证据、Clean/Inclusive及快照身份。
Clean full10/12、no0/12；Inclusive full9/12、no1/12。模型仍只接收原共享视图。
真实CLI/HTTP构造测试的发送端被mock；完整对象从来源到最终HTTP payload相等，
不只是检查关键词。临时mock授权与状态不导出为生产可用文件。

## T4：容量检查与审计预留

本地4B目录仅用作tokenizer。词表SHA与Qwen官方14B固定commit的tokenizer.json一致：
https://huggingface.co/Qwen/Qwen3-14B/blob/d17b97ca036704c156b50686b3f4551e224bbe03/tokenizer.json
使用本地chat template、enable_thinking=false整体计数，保存库版本与文件/模板SHA。
供应商是否使用完全相同chat template仍是NOT_TESTED，不冒充服务器精确计数。
HTTP序列化字节仅作独立诊断，不参与新入口的token调度。

审计读取完整历史、计划、实际原始对白及本地turn编号，职责/输出维度/晋级不变。
实际 Realizer response 作为单独一个消息原文提供一次，不再对字符串二次JSON转义。
另提供scene/turn数组位置对应的现有规范化编号，未创建新的事实摘要。
生成前用同一最终renderer计数已知内容；每场最多48个合法turn编号的开销提前计入。
未来内容预留=本批Realizer输出8192+两端token边界余量256，
后者来自当前词表最大token字节长度128的两端预留，不是旧JSON字节估界。
同一词表/原始消息直嵌是此界的前提，包装服务差异另保留256；不作任意模型的通用上界。

第一次补充之前检查已知首批Gate/Realizer/审计容量及调用额度；每一批生成前再次预留
全部Gate、Realizer和adherence额度，并检查增长后的实际保留前缀。
发送任何请求前按最终messages重数；生成后审计发送前再检查真实材料。
无法装载即延后/停止并保存超额量；不截断、不跳过审计、不强行晋级。
未来所有内容/依赖未知，因此不是“8场必定成功”的承诺。

## T5：HTTP错误证据

当前 SingleShotTransport 委托 request_once；保持原请求参数、一次open，无跳转/重试。
`response_received`正文与状态先经安全落盘，然后才解析JSON/业务结构。
非2xx、200非JSON、空body、业务失败、超时/无响应分别留证；有部分body时明确不完整。
追踪头包含 x-siliconcloud-trace-id；不留Authorization或API key。
无敏感正文完整留存，非UTF8用base64；脱敏记录原因。
成功正文若含凭据须改写才能留存，则保守失败，不把被改写的对白当模型原文推进。
transport completed不代表schema/计划/QA成功；workflow另存业务失败或领域校验结果。

## T6：统一离线回归与保护

本版本执行受影响policy、投影、feedback、HTTP、真实CLI构造及旧质量链单测，
另跑QA revision-1固定schema检查。不跑全仓606项、不改旧expected SHA。
所有socket连接被阻断；HTTP open以本地响应模拟。HTTP入口测试使用独立临时registry，
没有读取/创建/重置真实root registry。完整mock质量链不生成正式r03_delta或新累计R03。
仅保存明确标注的fixture；原失败方案、R02历史、旧runs与保护源码做前后SHA核验。
修改前快照在implementation_before中，逐文件身份如实标明是否对应live_4；
不把快照时已更新的fixture声称为原旧文件。历史原请求/数据/SHA不覆盖。
旧QA protected-manifest provenance问题未修复，仍单列待办。

## T7：等待预算授权

| Purpose | 最多请求数 | 粒度 |
| --- | ---: | --- |
| 历史正常Planner | 已有1，不计入新增预算 | 原失败完整方案，绝不重跑 |
| 本次正常Planner | 0 | 无调用 |
| 增量Planner | 1 | 唯一一次pre-Gate patch |
| Gate | 8 | 每个尝试候选/补充阶段各一次 |
| Realizer | 4 | 每批1-2场，最多四批，不强制用满 |
| adherence | 4 | 每批生成对应一批审计，生成前预留 |
| 本地closure | 1个虚拟预留槽 | 确定性本地拼接校验，不是模型请求 |

真实请求总上限17；ledger total_requests=18是17模型+1本地槽。
输出预留总上限73728模型tokens，ledger max_generated_tokens=73729包含1个旧本地占位。
总额和每purpose额度同时生效，不挪用未用额度、未用预留不退款。
理想mock链可执行6初始+2独立补充；串行依赖的2个补充可能需要两批，
或Gate拒绝/内容失败消耗预算时，后续场景会延后。这17次能安全收口已有结果，
但不保证完成所有计划目标或固定产出8场。本次不建议为凑场景增加额度。
所有repair/retry/replacement、QA/Solver/training、Memory操作0。

交付中的budget/model/bundle均待授权。真实入口需要另外明确授权文件，绑定当前
代码manifest、bundle/budget/model SHA；缺失会在构造transport前拒绝。
未执行命令单独保存。本次停止，不消费唯一增量机会。

后续待办保持：122711一次补充的完整链验证；单root收口后再决定其他roots；新R03
冻结后重新准备Composition/Fresh（旧14次仍暂停）；QA去重历史覆盖验收；恢复Targeted前
处理eligibility/兼容性；后续Solver、第二轮数据/配置冻结和训练评测。
