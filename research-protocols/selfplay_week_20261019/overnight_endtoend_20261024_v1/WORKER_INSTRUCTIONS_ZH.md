# 用户授权的过夜任务

你正在独立后台 Codex 会话中执行已获用户授权的工作。工作区 `/root/autodl-tmp/gxy-selfplay`，本任务根目录为本文件所在目录。用户希望过夜争取完成：不同 root 的扩充与出题、冻结训练池、普通/定向配对训练评估和可复现交付。允许写代码、运行 Codex 作者/独立审核子进程、使用现有 GPU 和模型；无需再请求批准。不得发送消息给外部人员、删除原有模型、修改以前的冻结实验、暴露密钥或自动升级购买服务。

## 先读和立即推进

1. 读本目录 `PLAN.json`、`PRODUCTION_HANDOFF_ZH.md`、`TRAINING_HANDOFF_ZH.md`，后两者若尚不存在先推进独立准备，稍后读取。主任务状态写本目录 `TASK_STATUS.json`，每阶段开始和完成时更新；先写版本化正式协议，不能只写计划就停止。
2. 已完成的 QFP2 不要重跑：`../postweek_question_firstpass_v2/FINAL_REPORT_ZH.md`。FBC1 成对来源、反馈卡及扩充机制位于 `../postweek_feedback_control_4root_v1/`。生产规则以 `../postweek_question_presentation_repair_v1/QUESTION_AUTHOR_RULES_V2.txt` 和 `scripts/current_method/question_presentation_gate_v2.py` 为准。
3. 主控脚本本身不产生模型数据。已有控制器没有 API 调用并不表示扩充已完成。实际作者和独立审核使用已验证可用的 Codex CLI：`/root/.vscode-server/extensions/openai.chatgpt-26.51002.51308-linux-x64/bin/linux-x86_64/codex`。非交互模式 `exec --skip-git-repo-check --sandbox read-only --color never --json -o <唯一输出文件> -`，通过 stdin 传具体指令。不要将本主控指令复制给数据作者。最多同时 3 个数据/审核子进程，主控占一个；主控可用 workspace-write/danger-full-access，数据作者和审核只读并仅返回 JSON，由主控不可变落盘。不要强行指定新模型，记录实际后端。完整保存每次请求/返回、哈希与时间。

## 固定范围与预算

- 期限由 `SUPERVISOR_STATUS.json` 指定，至多12小时；预留末尾至少45分钟做评估或收口，不能无限修提示词。
- 新 Train roots 固定为 24719、25652、29753、31457、38601、43599、45849、50175，完整 ID 加 `cc-train-episode-`。来源名单与内容在 `offline-deliveries/h4_codex_train_pool_v125/data/ROOT_MANIFEST.json` 和 `.../sources/<root>.json`。验证 Train 白名单、hash、完整 turn。它们均曾被 M-clean 历史看到，称“不同于前小试的新扩充 roots”，不能称模型从未见过的 roots。
- 每个 root 做 GENERIC 和 TARGETED 两臂，初始源/事件素材相同，两路线各3场，合计每臂6场；每场12–16发言，按既有正式作者/审核流程逐场审核再接续。最多16完整新来源、96场、112个新题槽位。两臂生成预算、材料、操作槽位与模板约束相同，差异仅预冻结反馈。TARGETED 反馈只可来自已核验的 Train 作答/来源不足卡，必须标注 exploratory；不能升级为已证明的正式模型弱点。
- 每源 MC2/3/4/5/6、ORDER1/2 各1题。MC6为全对控制，主训练与主收益分析排除，公开不得泄漏数量或控制标签。MC2–5保留准确gold数配额，不按 solver 成绩筛题。排序公开选项必须真正打乱，顺序目标有来源唯一依据。删除数量/时间词等只可在冻结作者规则许可范围内，不能事后将失败改为成功。
- 每个场景/题槽一次不可变语义尝试；只允许明确的传输失败重试，重试前记录原返回和原因。返回成功但内容失败不得重试或悄悄改写。结构/语义失败必须记录为 FAIL/SHORTFALL，保留 planned 分母。修复工程代码可重跑未成功执行的阶段，不能改已封存内容。
- 先跑前4个新 root 的工程门，按 QFP2 总准入>=90%、每root>=85%、每槽>=75%、source/Gold歧义失败0。若失败，停止自动扩大，将问题和剩余槽如实记录；可使用已独立审核的QFP2主任务另做明确标识的pilot，但不得冒充本次新root端到端通过。若通过，继续另外4root。独立质量门不可被最终solver分数覆盖。

## 作者与审核隔离

- 数据作者只见完整源、允许事件素材、条件反馈和规则，不见旧题、私有答案或solver回答。禁止使用固定模板程序批量编造对话来冒充 Codex 生成。
- 独立审核不能由同一数据作者自审，使用新CLI会话；来源审核和题目审核有详细证据，先推导独立答案再对比作者Gold。明确披露AI审核和条件可见，不能称人工盲审。
- 逐项验证引文是原文连续子串、turn存在、人物/场景/状态绑定、所有选项真假和唯一Gold、任务操作、多样性覆盖。机械校验不代替语义审核。按 root/arm/槽位汇总首次准入与失败。
- 弱点反馈只采用来源核验后的判断；QFP2解释样本0/16严格通过，不能将solver解释直接作为训练金标。原始可见解释不代表忠实内部CoT。

## 训练与公平比较

- 只在新的独立冻结训练协议中训练，不改FBC1/QFP2原来未训练的事实。明确本轮为探索性单次配对实验，不冒充正式A2/B2门已经通过。
- 两新模型都从同一个既有 B1 merged checkpoint 开始，分别注入相同设置的新 LoRA。不能先训 GENERIC 再在其结果上训 TARGETED。记录基座/adapter/数据/代码/seed全谱系和hash。
- 默认使用本次审核通过且两臂共同root/slot匹配的 MC2–5 与排序题；可把QFP2每臂24道主任务作为预先声明的既有数据层，分层报告新/旧；原24 foundation如确需使用必须两臂完全相同并审计来源/Gold。旧133题候选池不能直接当定向反馈数据。
- 每题预冻结2种选项排列、2epoch，保持双射及相应答案一致；两臂题量、优化器步数、超参数、起点、共有基础题匹配，token差异报告。内容不足用共同匹配子集，不填假数据、不重复过采样某一臂凑数。
- 完整来源输入零截断。先真实tokenize全部prompt+target，结合现有runtime选择安全窗口，做最长样本smoke；若OOM仅可降低batch/用gradient accumulation而保持等效优化预算；不能偷偷截断历史或缩短源。记录原始失败。
- GPU单实例flock，顺序训练两臂。磁盘仅约9.9GB，根盘约2.7GB；旧模型不删除、不重复复制。临时merge放 `/dev/shm/gxy_overnight_20261024_v1/`，逐臂评估后只清理本任务可重建临时文件，耐久存adapter、优化器/调度器/RNG恢复状态和日志。不要把唯一checkpoint只留内存盘。
- W&B环境当前没有key，不要求用户补key。用offline模式记录，保存本地train loss、lr、gradient finite/norm、tokens、step、显存、耗时、数据和模型版本，若SDK缺失优先本地日志并明确未接入，不能伪称已同步。优先完成离线W&B。
- 每20个optimizer step在预先冻结的共同Dev子集跑真实最终答案exact/partial（MC）及pairwise（排序），不是只记valid loss；不足20步也做最终评估。跟踪不选最佳分数checkpoint，以预冻结固定终点比较。监控可以轻量固定子集，但最终全冻结评估范围一致。验证题不能进入Challenger反馈、生成或训练。
- 已有训练脚本/Day7评估入口优先复用，但旧冻结runner绑定旧池不能直接对新池伪装旧run。必要时只在本目录新建适配runner，增加真正checkpoint/resume与Dev评估。

## 评估和结论

- 在任何新训练前冻结共同评估题与root，沿既有Dev边界，主控生成过程不读取评估题级内容；禁止使用Test题级结果反哺。B1、B1+GENERIC、B1+TARGETED在完全相同题/排列/输出规则上比较；现成B1缓存仅在prompt/checkpoint/seed确实一致且hash绑定时复用，否则重测。
- 报告主任务MC2–5、排序、MC6控制分开，整题/部分指标、按root配对、排列一致性、新旧题分层、训练拟合与未参与生成的评估分开。高训练分数不证明泛化，题难不证明反馈有效。可见reason若用于诊断需保存和核验，不把原始解释直接入池。
- “没有明显收益／证据不足”是允许的完整实验结论。不要为求正结果反复生成、换测试集、改终点或追加训练。

## 进程、恢复与完成交付

- 本任务已有外部supervisor持久保活；不要再创建重复supervisor、无限自举、定时任务或自我修改监督器。自己的子任务应可恢复，完成一个阶段就落不可变manifest/receipt；校验hash后才跳过已完成阶段。工作记录每阶段更新到 `TASK_STATUS.json`，长阶段每5分钟以上有可见进度日志。
- 主控等待长工具任务时每60秒以内轮询一次并更新阶段心跳。发现 `FINALIZE_REQUEST` 文件后停止新增生成，优先评估已有可行产物并收口；不得开始预计越过deadline的新训练。
- 遇到单root语义失败，保留并继续其他独立root；达不到扩批门停止扩批、报告其影响。关键公平性/数据隔离/答案审核不通过不得强行训练。API/认证/资源无法恢复时，有限3次工程重试后收口，不能整晚空转。
- 本目录最终必须有 `FINAL_REPORT_ZH.md`、`FINAL_REPORT.json`、`FINAL_ARTIFACT_MANIFEST.json`。机器报告至少含 `overall_status`（COMPLETED/PARTIAL/BLOCKED）、`objective_completed`（布尔）、各phase状态、准确计数/失败、模型/指标路径、结论边界和剩余工作；manifest内 `artifacts` 为 `[{path,sha256,bytes}]`，路径必须真实且不自包含manifest本身。不要把不存在模型或未做评估写为完成。
- `overall_status=COMPLETED`只在新增root生产验收、正式池冻结、两臂训练、共同评估和报告均真实完成时使用；任何缺项报告PARTIAL或BLOCKED并提供可靠下一步。若只有QFP2 pilot训练完成，也必须说明新增root主目标未完成。
- 最后由全新只读CLI审查报告/模型/指标/哈希与范围声明。将验收保存在 `INDEPENDENT_CLOSEOUT_AUDIT.json`。只有证据真实才结束；不要只写计划、建议或等待用户回答。

### 机器可验收字段约定（必须匹配监督器）

`FINAL_REPORT.json.phases` 为对象，以下6个键各有 `status` 与证据路径：`freeze_rules_and_balanced_protocol`、`different_root_paired_source_extension`、`independent_source_and_question_review`、`freeze_matched_training_pool`、`paired_training_with_metrics_and_dev_every20steps`、`frozen_evaluation_and_delivery`。真实完成的status用大写 `COMPLETED`；缺项不可标完成。

完成报告另含 `training_arms.GENERIC` 和 `training_arms.TARGETED`，各有绝对文件路径 `adapter_path`（真实权重文件，不是目录）与 `result_path`（训练结果JSON）；`common_evaluation.summary_path` 为共同评估汇总文件绝对路径，`common_evaluation.models` 必须为 `['B1','B1_GENERIC','B1_TARGETED']`。这三种逻辑名称对应真实checkpoint和同题评估，不能仅改名冒充已评估。

`INDEPENDENT_CLOSEOUT_AUDIT.json` 必须含 `verdict`（PASS/FAIL/HOLD）与 `report_sha256`（审查的最终机器报告hash）。完成审计后不要再更改报告，除非重新审计。最终manifest绑定机器报告、中文报告、独立审计、所有训练模型/结果与共同评估汇总，以及协议/训练池/相关关键证据；manifest本身不自包含。若审计未通过，只能PARTIAL/BLOCKED如实收口。监督器不会仅凭最终文字宣称完成就退出。
