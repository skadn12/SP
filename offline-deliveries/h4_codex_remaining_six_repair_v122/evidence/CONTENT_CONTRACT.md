# v122：剩余六个失败 root 的固定补救协议

本批仅处理74644、103866、135466、146700、197635、254852，分母固定6；此前22个救回不重复生成。v118–v121全部只读，旧失败不改判。沿用下列内容规则；其中原批范围文字仅为历史背景，本批范围以本段和REPAIR_SCOPE为准。

来源：74644从v121已修候选继续，仅允许一条text补丁纠正婚礼已告知却被否认的矛盾，保留事故修订和完整补丁链，双审；其他五个不新增文本补丁，仍fresh双审。197635、254852累计来源保持EDITED_SYNTHETIC。135466可在正式source author阶段提出与原事件一致的A线程约束，禁止继承矛盾的knee映射；NEW_SELECTION须在完整source证据上固定A1联系尝试/A2对应责任联系线程阶段/A3实习线程，不能要求联系必须成功或获得原谅。

前缀：103866只候选保留A三场；146700候选保留六场及原QA/anchors；254852只候选保留已审A1，A2原文作为单独历史未审候选导入，保留真实旧作者，不虚称新作者。所有候选前缀都经本批assigned reviewer全文复核。其余从A1开始。

预算：每root本批一次来源候选、每新scene一次语义首稿、每完整root一次新QA语义首稿；146700零新QA稿，254852A2零新scene稿、一次fresh语义审核。原稿继承不重写，所有失败保留，不追加至通过。scene/asset/public审核预检归档首个payload并锁定语义，仅允许可唯一确定的缺candidate_id补齐；其余reason/relation/citation变化不伪称机械修复。这三类正式save须同hash预检PASS、内部再验后才归档RAW。独立exec预检成功后才save，禁止shellfallthrough。来源审核沿用一次不可重交机制：审核者先在内存中调用既有validator核验完整schema，再正式保存；正式保存保留RAW及失败，不允许重交。

新场每场12–16自然英文turn，只原有speaker；增加语言门防止中文稿误过。不放宽原生事件、自然性、多样性、事实支撑、排序唯一性标准。146700新assigned reviewer重新asset审和fresh public审，不能追认旧失败RAW。

# v118 全训练侧 A3＋B3 扩充与联合出题

本批固定68个clean-policy训练root；开发/测试在root、原始source_id及source_record_sha256三级隔离。不新训练、不合并正式训练池。13个已有完整A3＋B3复用，v116两条HOLD保留；其余53个全纳入，不以筛选后的小样本替代全量。20432候选必须应用v114隔离及推荐覆盖。

## 新批事前适配

25条基线不满足先前“恰两speaker、恰三R01场”的模板条件。本批不改基线，允许使用所有原有speaker，仍禁止新增发言人物。已有三场R01的root，A1/A2/A3各延续对应原场；缺少原场的root，事前在selection.json固定三条真实已有事件线程及逐字来源，允许同一旧事件的不同阶段连续推进，不能虚构缺失R01场。该组单独报告，不冒充与v117完全同条件。

八条验证root事前固定，覆盖标准组和适配组；不换root凑通过数。完成首波有效流程核验后顺序推进其余固定root。语义HOLD保留；机械实现问题保留错误及修复版本，复验后继续，不能用改schema或重写语义首稿抹去失败。

## 扩充

固定P＋R01原文，A1→A2→A3→B1→B2→B3累计；每场12–16条自然英文发言，只提交speaker/text，统一序列化。每场一次语义首稿，原文前缀逐字不变。作者先完整读累计来源和全部继承约束；B还读完整预留原生事件。每个B场使用一条独立未消费MultiWOZ训练事件，选择前固定不相交候选分区，全文兼容性匹配后永久预留，失败/未执行也不得释放。

A推进旧事件当前阶段，不能回退已完成行动或只复述愿望。B保留原生请求、条件、结果边界，有区别于A的新目的；服务信息可由既有speaker明确转述，不把朋友改为服务人员，不倒灌新的职业、关系或状态为旧事实。原生事件中的未指定属性保持未指定。

非该稿作者完整读累计来源、约束及B完整原生事件，逐项判断time/location/actor_identity/plan_vs_completion/cross_scene_continuity/semantic_repetition/naturalness/event_fidelity，并逐条核对继承约束。只有READY且绑定有效才能继续。HOLD停止该root后续依赖，不换源、不修首稿，其他root继续。

场景须含回应导致的解释、选择或后续行动；不能反复确认一个事实凑长度。三场B不得全部复用“顾问转述—朋友核对—写清单”同一骨架。略机械、信息密集记naturalness WARNING；严重循环无推进、新增矛盾、篡改原生事件则FAIL/HOLD。多样性不能靠虚构原生事实取得。每条警告/阻断应引用具体turn。

## 联合出题

六场全READY后，完整累计来源出一个六事实family，每新scene恰一事实。一个事实至多一个错误替代；同一范围须被明确contradicts，unknown不能作错项。错项尽量只改一个可核验属性，保持相近语义类型/粒度，避免荒诞或无关选项；直接反义本身不判无效，难度以冻结后Solver实测，不据分数返工。匿名资产来源先审后固定seed组合Gold2/3/4/5各一及Gold6对照，允许不足缺额。

两排序槽位各先固定六个完整turn，每新scene一个；ORDERING_ANCHOR_PLAN早于QA_DRAFT。每项只含一个事件/言语行为，保留人物、模态、完成状态。公开范围为六新scene“对话出现顺序”；非作者审遍合理语义匹配，所有匹配组合须同一严格顺序，不换锚点救题。

实际预定非作者先读完整来源与PUBLIC_ASSETS，保存ASSET_SOURCE_JUDGMENTS；compose后实际重读PUBLIC_QUESTIONS，保存PUBLIC_SOURCE_JUDGMENTS；之后才构造原设计PRIVATE_GOLD并交叉核对。不读私有答案或当前Solver分数救题。程序不能证明模型内部未读，保留实际阅读声明及可见输入。导出保留所有引用、relation/reason，不截成首引；逐字quote/speaker/turn/source-hash检查。

## 统一证据与评价

新批所有上下文、完整打印输入、原始payload/judgment均在batch内独占写；payload和serializer绑定必须先于draft/物化。作者/审核者使用ROLE_ASSIGNMENTS中实际代理身份，读收据统一full_source_read、turn_refs_read、all_continuation_constraints_read、constraints_binding、真实起止时间及输入哈希；禁止代他人建立read-context。共享文件系统下此为可审计声明，不是密码学执行者认证。

验证和生产均保持v15配对Solver：M0及既有M_clean_policy，多选6展示×3回答，排序12展示×1回答，无自动重试。Gold6单列且不入候选；完整有效调用且0<等展示partial均分<1方入候选。完全正确率、partial、误选/漏选按题型/Gold/root分别报告，调用及同family变体相关，不能冒充独立样本量。

4/4或本批通过率不说明任意root成功率，所有root已训练/会话曝光，不称unseen holdout。代理作者和审核者继承同一模型，不是外部/异模型认证或严格盲审。精确模型ID、usage和成本当前不可得则null；产物tokens不是调用usage。内容可靠度、自然度、难度、训练价值分别报告。本批交付完整对话、题答证据、全部阶段稿/失败、原始Solver、逐root总账及封存清单。
