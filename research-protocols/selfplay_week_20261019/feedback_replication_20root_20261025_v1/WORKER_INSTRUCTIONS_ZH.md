# 已授权的 16 新 root／20 root 合计过夜 F/G 复现

用户的明确指令：**新增16个，加在一起20个，挂上过夜实验**。你是后台执行者，必须真实完成扩充→独立审核→F/G出题→独立Gold与内容审核→固定B1作答→统计报告，不要以计划或进度说明结束。允许修复工程问题；不得改变冻结方法、反馈、root名册、预算和模型，不训练，不修改Dev/Test，不重跑原4。不要向用户询问许可。所有操作仅限新任务目录及其必要GPU进程，原实验和模型只读。

## 必读路径和固定约束

新任务目录：`/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/feedback_replication_20root_20261025_v1`（以下P）。仓库：`/root/autodl-tmp/gxy-selfplay`（REPO），研究周目录为P.parent（W）。

原实验E=`W/feedback_closed_loop_20261024_v1`。只读复用原4：218519、222937、225575、229686。原FINAL_MANIFEST.json SHA256=`fc71556232d16133d3ba9aa28459d2a8c21836cfed31b54e7132df838a6f05af`，772个封存文件。原128条作答直接复用，不做新model call。旧结果不利于反馈也必须保留。

先读并核验：`INPUT_LOCK.json`（bindings不可变）、`CONFIG.json`、`PLAN.json`、`PROTOCOL_ZH.md`、`ANALYSIS_PLAN.json`、`FEEDBACK_TRANSFER_ADAPTER_ZH.md`、`SELECTION_FREEZE.json`、`PARENT_SOURCE_SELECTION.json`、`SOURCE_PREPARATION_RECEIPT.json`、`RUNNER_ADAPTATION_RECEIPT.json`。16个root已按固定SHA排序及父source≤8192token筛选，不能更换。它们不是M-clean未见root，只是未参与此次4root诊断。

`SUPERVISOR_STATUS.json`含启动和12h deadline，每次阶段/每30–60秒更新`TASK_STATUS.json`及`PROGRESS.jsonl`。最后45min停止开新生成（FINALIZE_REQUEST）；最后20min不再开新Solver/工程修复，保存现有结果及PARTIAL报告。`STOP_REQUEST`立即停止。工程失败连续3次须收口；不凭空成功。不要启动旁支、训练或重复旧实验。

## 当前已备好的可用工具

1. `scripts/codex_calls.py`：真实fresh Codex CLI text-only calls，嵌入完整授权数据，禁止工具；JSON-only，记录prompt/raw/events/stderr/usage/实际backend/threadid/hash、PID进程registry；最大3次**无语义响应**的transport重试。返回非JSON、错误引用或不合格内容均消耗该次语义预算，不能伪装transport失败重新调用。中断START无RESULT必须显式恢复分类，不可静默重生。`call(call_id,prompt)`返回(parsed_object,receipt)，复用已完成receipt；作者和审核均fresh独立thread，不能用你这个管理线程亲自写题/来源或给语义PASS。并发最多3个call；不要 override 默认模型或reasoning。实际默认此前为gpt-6.1-sol/hua/ultra，每call核实。
2. `solver_runner.py`：已将旧runner泛化为CONFIG16 root/256题/512作答、CL2 namespace、允许source有效子集；固定完整输入FULL、B1、THINK_OFF、6144输出/16384总长、温度.8/top_p.9/batch8、2程序排列。不要修改冻结runner。使用当前Python `/root/miniconda3/bin/python`，依次`prepare`、`run`、`analyze`，日志落P。其import的旧structured-v2及所有模型文件只读。
3. `REPO/scripts/current_method/production_question_gate.py` 的audit_candidate/required_checks/validate_review是真实机械门，不会证明语义。导入时将其目录加入sys.path。无需新训练依赖。
4. 旧E的`assemble_sources.py`、`assemble_content.py`、`summarize_difficulty.py`可只读学习schema，但有4root/218519特判，禁止直接执行。旧O=`W/overnight_endtoend_20261024_v1`的production_driver/control/questions可只读参考接口，但O是双来源、14turn×6、7槽且含训练，不能照跑。你应在P/scripts创建有限、可恢复、幂等的`production_driver.py`与汇总脚本。
5. `launch.py`、`supervisor.py`、`validate_final.py`负责持久管理。可运行`python validate_final.py --describe`查看最终report/audit/manifest接口。不要修改这几个冻结文件。所有新增代码先py_compile及必要静态检查，然后执行，不为工程验证调用旧题Solver。

### 并行交接（启动时的明确文件所有权）

父会话的独立工程协作者正在交付两个尚未纳入INPUT_LOCK的文件：`scripts/source_driver.py`（来源准备协作者负责）和`analysis_replication.py`（统计设计协作者负责）。**不要同时编辑这两个文件。**来源脚本交付后会出现`SOURCE_DRIVER_READY.json`，其绑定hash核验、py_compile后可直接运行；分析脚本交付后可运行`python analysis_replication.py --self-check`（纯内存、0模型）核验，其输出接口已定`REPLICATION_COMPARISON.json`、`REPLICATION_TABLES_ZH.md`、`REPLICATION_ROOT_DIFFERENCES.csv`。

先读取`ENGINEERING_HANDOFF.json`的UTC截止；在其截止前，若来源脚本尚未ready，就独立实现**出题/审核**driver及内容汇总（你负责`question_driver.py`等其他新文件），不要空等，也不要启动第二份来源作者。来源ready后尽快启动该唯一来源driver，最多3并发；读取其真实状态及冻结最终source。超过交接截止仍未ready，先检查是否确有活工程进程/文件更新，然后可在另一个新文件名`source_driver_recovery.py`实现必要等价source阶段，并记录偏差；仍禁止覆盖协作者文件、重复任何已存在CLI call_id或已经生成的root。统计脚本若未交付，同理先推进其他工作，最迟Solver前必须冻结实际统计脚本binding，不能看分数后改主要指标。

## 阶段一：锁定验证与只读完整性

检查INPUT_LOCK每binding、16 packets、父来源、48个预留事件及原封存772文件。生成`ORIGINAL_INTEGRITY_BEFORE.json`，字段`status:PASS, unchanged:true, original_root_ids:[4个完整root_id], original_manifest_binding:{path,sha256,bytes}, verified_artifacts:772`，可加逐文件摘要；最终重复验证并写`ORIGINAL_INTEGRITY_AFTER.json`。不能修改原sealed文件。任何输入漂移则BLOCKED收口。

## 阶段二：每root共享来源R03扩充与独立审核

读取`inputs/cc-train-episode-{n}_SOURCE_AUTHOR_INPUT.json`、其完整父source和每个native_event.binding完整record。source作者只读这份**中性**材料包，不给F卡、预期错误、Gold配额或旧Solver结果。不要传整个P protocol给source作者。旧source_authoring(root222937)的输出schema只读参考可用；信息都应嵌入prompt，不准作者访问文件。

每root一次初稿调用可返回完整六场，但必须累计读取前情，A1→A2→A3→B1→B2→B3，每场恰8turns，**新增48turns**；每场平均英文词数≤18，保留父turn对象逐一完全相等。scene_ref和turn_ref为`R03:A:S01:T001`等。只用父源已有人物，不发明新命名人物/性别；不把愿望变成确认或完成。A延续已有事件；B三场分别忠实报告其永久预留的完整Train native enquiry，必须读原生全部turn，保存请求/拒绝/服务回复/预订状态、关键数值与时刻，不把三单独查询编成一趟旅行；允许在48turn预算内忠实压缩，不要求逐字复述原生所有语句。不能发明服务回复或修复继承冲突。保留每条说话人/quote锚点及native映射。

source reviewer fresh独立调用，读取完整父+48新增+完整native records，中性packet，不看F/G。逐维检查time/location/actor_identity/plan_vs_completion/cross_scene_continuity/semantic_repetition/naturalness/event_fidelity与逐字来源anchor。明确新增加的blocking和继承warnings，不能因产量或预期题型而PASS。初稿后**最多一次实质修订**，保留SCENES_DRAFT、初审、SCENES_REVISION_01、recheck及所有失败；工程性JSON/ID标准化须日志，不得偷改命题。修订使用fresh source author，回传旧草稿/独立审核，不给出题卡。无通过则停止该root后续，留缺额，不替换root或native events。

由程序生成最终`sources/cc-train-episode-{n}.json`，prefix exact、parent/new/total turn数及累计数正确，保留synthetic状态、native provenance、review warnings。每个合格root最终source同供F/G。写`SOURCE_SELECTION_FINAL.json`（schema沿E，split train，roots为通过source的root_id/path/sha256/bytes；失败roots不把父source冒充最终source）、`SOURCE_FREEZE.json`、`SOURCE_FAILURES.json`。16全部source失败则直接PARTIAL，无Solver。

source内容通过后可并行该root出题；最终统一selection与freeze必须在机械gate/solver前稳定冻结。分root中间selection可存独立路径用于阶段内机械检查，最终统一gate重新复核。

## 阶段三：隔离F/G计划、出题、审核、一次修订

读取共同合同`question_authoring/COMMON_AUTHOR_CONTRACT_ZH.md`。每root每arm用fresh作者上下文，8槽固定为MC2、MC3、MC4、MC5、MC6、ORDER1、ORDER2、ORDER3，六项A–F及对应Gold/operation**完全沿合同**。程序设置metadata/author_id/source_binding必须诚实记录，不由管理者生成语义内容。

**G只收**共同合同、分配完整source、source_status/warnings、六项schema与操作配额。不得收F卡、迁移adapter、旧/新Solver错误、F题/计划、原4结果、实验方向或期待F/G谁难。G作者、G审核不复用F thread。

**F额外收**`question_authoring/F/PUBLIC_FEEDBACK_INPUT.json`的完整19张原冻结卡和迁移adapter（旧观察不等于新root已复现）。不要把事后G错误分析或新理由不一致审计加入F。每槽TASK_PLANS记录feedback_id、trigger_realization、predicted_observable_error、counterexample、target_acceptance_test；不能实现触发写TARGET_HOLD，内容有效题仍可评估难度，不把目标命中冒充已复现弱点。

先每root/arm一次计划call，保存TASK_PLANS并hash冻结，再独立draft call接收该计划+同样的授权输入。可同arm一次出8槽，以减少上下文重复，但不可跨root省略完整source。初稿每槽一份，不根据生成难度再抽题；无法产出记SLOT_FAILURES。题目格式完整参考共同合同和E/CANDIDATES_PRIVATE.json的schema（只给schema，不给旧题内容）。每题必要R03前提/排序边；英文题面30–90词/项10–35词为原描述性质量要求；真项distinct、假项改变一个精确关系、有支持/反驳anchor；排序6节点/5相邻边、唯一来源支持全序，显示由程序打乱，不能带first/later/finally等编辑顺序捷径。ORDER1时钟值不含显示时刻，ORDER2source_defined_dependency，ORDER3实际话语贡献顺序；不要把三者偷换。

**独立审核流程**：先隐藏arm、feedback、作者Gold/item_analysis预判，只给完整source/warnings和匿名题面/选项，fresh reviewer重建每题independent_gold、逐项或逐边引用理由，保存后不可覆盖；再给另一fresh reviewer先前保存的独立Gold与作者key/evidence，核对来源支持、gold_truth、distinctness、distractor_relation或six_gold_control，以及排序所有required_checks和R03必要性。程序按匿名mapping恢复candidate_id并绑定准确fingerprint，不给审核者F/G对方。明确是同Codex家族AI审核，不声称人审或完全盲审。不能程序对比Gold相等就自动PASS。

初审非PASS槽最多一次content revision，提供本槽审核缺陷、原计划和本arm授权输入，不提供另arm/Solver。保留初稿、初审、一次修订、独立重新核验；修订者thread_id另外记录，candidate author_id可保留原作者作为lineage键但不得冒称修订同thread。非JSON返回也消耗初稿，最多仍只可用一次允许的修订，不无限工程重试语义。已PASS不改题。

调用gate.audit_candidate及gate.validate_review。`CONTENT_REVIEW_AI.json`每record沿E/runner schema：candidate_id/candidate_sha256/source_binding/author_id/reviewer_id/decision/checks（每required字段decision+rationale）/independent_gold/reconstructed_gold/r03_necessity/完整读取与隔离声明，reviewer_id!=author_id。不完整或不可靠=HOLD/FAIL。保留所有实际生成候选与最终review，未生成槽独立失败表，不用placeholder冒充题。

最终写`CANDIDATES_PRIVATE.json` list、`CONTENT_REVIEW_AI.json`、`CONTENT_GATE_FINAL.json`、`CONTENT_FREEZE.json`（bindings至少含前两者、SOURCE_SELECTION_FINAL、SOURCE_FREEZE、protocol/config/common、全部计划/作者/审核/修订/调用回执、失败表，status FROZEN_BEFORE_SOLVER）。`question_authoring/F|G/AUTHOR_INPUT.json`记录实际许可输入、读取回执、thread隔离；`FEEDBACK_CONSUMPTION_LEDGER.json`记录F卡使用/触发实现/未实现，不用未来分数倒改。

**仅整题完全重复排除**：按type、空白归一question、忽略标签和排序的六项文本判断；不能扩大到同事实/主题排除。相同整题先按固定root/slot/arm顺序保留其一、另HOLD并保存duplicate disposition；不可去重后重出。内容冻结后不再改语义。

## 阶段四五：固定Solver预检与真实执行

所有内容和目标冻结后，按`python solver_runner.py prepare`检查完整prompt预算与structuredAPI。不给Solver私人Gold/反馈；runner已使用公共请求和私有评分分离。上下文不能截断。若某候选预算超限，明确其工程失败/缺额，不借机重新改写题或更换root；如需要排除请求，必须在Solver首次调用前追加不可覆盖的长度处置冻结并在独立审核中披露，保持原content版本不变，不改冻结runner。谨慎先做相同prompt逻辑的CPU长度预检查，以减少prepare中途不可覆写文件。

真实B1=`/root/autodl-tmp/gxy-selfplay/restart-n5-runs/day5_B1_schedule_v1/merged_model`；tokenizer=`/root/autodl-tmp/Qwen3-4B`。不是base，也不是仅M-clean。FULL、THINK_OFF、item_evidence→answer_reason→<box>answer</box>不变。两预定程序排列属于同语义重复测量；sorting显示不得已是正确Gold顺序。无自动语义重试。执行run并等真实GPU作答/RAW_RESPONSE_FREEZE，再analyze；保存完整prompt/raw文本/引用/最终选择/Gold对齐、token、finish_reason、运行时间。已完成raw绝不重跑，遇到中断须按原runner不可重放政策保存PARTIAL。

## 阶段六：新增16主要复现 + 原4 + 合并20

读取ANALYSIS_PLAN全部规定，不能只报有利子集。至少生成`REPLICATION_COMPARISON.json`、`REPLICATION_TABLES_ZH.md`及逐rootCSV。使用真实SCORES.records；按strict.root_id/arm/slot_id/variant连接；原4读取E/execution/SCORES.json。MC6单列，主为MC2–5+3排序。保留旧strict口径全冻结请求汇总；另外用两臂内容有效且各两排列真实返回的共同槽做配对难度（未返回不填0，已返回但格式无效按strict0；失效/缺额独列）。清楚列2排列分母与独立语义题数。

new16 primary、old4 discovery、combined20 secondary分开；精确率、answer-only、MC P/R/F1/FP/FN、排序pairwise、格式有效率、排列一致性、root/操作/Gold分层、产出率/拒绝原因/target命中均报告。root配对F−G（正=F更容易），按整root bootstrap10000次seed20261025和95%区间，不独立重采样排列或选项。缺共同槽root的差值null，不补0；保留全部计划root完成表，明确CI有效root数及缺额选择偏差。固定两个排列不能证明稳健弱点；别把判断-选择不一致简单说成输出box typo。

不能把共享、受中性反馈目标指导的source归因成扩充反馈因果收益；此次没有训练收益结论，也不证明模型正式复现全部卡片。F更难/相当/更容易/不确定均照实。旧F/G原4净差仅集中229686，必须检查新16是否一致、避免合计盖过新增。

## 最终交付与收口

`FINAL_REPORT_ZH.md`自包含说明真实完成规模、各阶段/失败/耗时/费用usage（如可得）、新16及合计20表、置信范围、负结果、来源/审核/模型局限及下一步。`FINAL_REPORT.json`与`FINAL_ARTIFACT_MANIFEST.json`精确接口见`python validate_final.py --describe`。原4完整性最终receipt按前述字段并binding到report。最终报告JSON冻结后调用fresh independent closeout reviewer，仅审核已有真实产物/统计，输出`INDEPENDENT_CLOSEOUT_AUDIT.json`：verdict PASS|FAIL + report_sha256 + checks/issues；report仅存audit路径，manifest绑定两者防hash循环。可用helper.call(...,closeout=True)做已有产物审计，不能借此截止后新生成。

最终manifest绑定reportJSON/MD、独立audit、原4完整性、所有核心数据/内容/调用/评分/统计/失败记录；运行validate_final核验。512次新请求全部真实返回且阶段/审核全通过才能COMPLETED。若缺额、失败或时间到，诚实PARTIAL/BLOCKED，仍保存可复现产物和manifest；不要等待用户答复，不无限续跑。先完成可完成的已授权工作，不只解释困难。

**现在开始执行：**校验锁定输入，按并行交接文件所有权先构建有限出题/审核driver，通过工程检查；来源脚本ready即开启前三个source作者fresh调用，最多3并发。每个root就地保存进度，不等全部代码优美才开始。你这个管理线程只组织/修复代码/聚合，不自己产语义来源/题/Gold/审核PASS。后台监督器会在你未完成即结束时续跑同一管理thread，但独立作者/审核始终fresh。
