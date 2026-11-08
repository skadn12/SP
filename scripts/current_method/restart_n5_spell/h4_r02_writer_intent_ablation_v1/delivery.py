"""Human-readable preparation deliverables; never sends requests."""
import json
from pathlib import Path
from ..core import atomic_text,atomic_json
from .constants import VERSION,REPO,CODE,CONFIG,DEFAULT_RUN,ARMS

def write_delivery(run):
 run=Path(run);read=lambda n:json.loads((run/n).read_text())
 pre=read('audit/preflight_report.json');tests=read('audit/test_report.json');tokens=read('audit/token_budget_summary.json');identity=read('audit/token_identity.json');coverage=read('audit/dependency_coverage_report.json');pres=read('audit/source_preservation_report.json')
 if pre['status']!='PASS':raise RuntimeError('DELIVERY_REQUIRES_PASSED_PREPARATION')
 from .reporting import report
 review=report(run)
 if (review.get('status')!='PREPARED_NOT_RUN' or review.get('errors') or
     list(run.glob('arms/**/ledger/requests/*.json'))):
  raise RuntimeError('PREPARATION_DELIVERY_REQUIRES_ZERO_REAL_STATE_AND_CLEAN_REPORT')
 status='''RUN_ID = h4_r02_writer_intent_ablation_v1_live_1
RUN_STATUS = PREPARED_NOT_RUN
PREPARATION_STATUS = READY_FOR_AUTHORIZATION
ACTUAL_EXTERNAL_CALLS = 0
PRIMARY_WRITER_CALLS = 0
REPAIR_CALLS = 0
MINER_CALLS = 0
SOLVER_CALLS = 0
LIVE_STATUS = NOT_RUN

12 primary 请求已离线准备；未来最多12次整批替换repair，总上限24次逻辑调用，失败占预算。当前任务未授权live。没有创建已授权标记，没有后台任务。六条原V3 intent与历史NOT_READY/评分原样保留。
'''
 atomic_text(run/'RUN_STATUS.md',status)
 prefix=f'env PYTHONPATH={REPO} /root/miniconda3/bin/python -B -m scripts.current_method.restart_n5_spell.{VERSION}.run'
 commands=f'''当前状态 PREPARED_NOT_RUN；以下 live 命令仅在下一条明确真实执行指令之后使用。本轮没有执行它，也没有创建授权标记或后台调度。

未来真实执行（固定目录、固定配置、固定12格次序）：

```bash
{prefix} live --run-dir {run} --config {CONFIG} --execute-after-authorization
```

断点恢复使用完全相同命令。仅复用同臂、同root、同slot、同stage、同request_id且身份一致的已完成缓存；known HTTP rejection失败占预算且不重发；reservation/付费完成未知或sticky guardrail存在时立即停，不自动repair、重发、改ID或删除账本。无需也不得改为live_2。若发生冻结漂移，先报告原因，不重冻结来绕过原任务。

离线重新核验准备身份（已冻结时只核验，不重新生成模板或覆盖结果）：

```bash
{prefix} prepare --run-dir {run}
```

离线报告／质量审阅材料：

```bash
{prefix} report --run-dir {run}
```

共享入口为 generation.execute_writer_batch；参数 source='contract_guided'，A intent_guidance=True，B=False，无feedback suffix或hook；同一finalize_writer_candidates在各root-arm分别执行原validator与Gold置换检查。外层 execute_live/execute_schedule 仅负责编排、身份/预算/unknown守卫；没有另写生成、repair或准入器。

计划12次primary，最多12次repair，总最多24逻辑调用；每臂6+≤6、每root-arm3+≤3、每格1+≤1。repair紧跟自己的primary，按原minima触发、整批替换、不合并、不回退、不因final validator拒绝再修。每次输出预算8192，reserve512。HTTP 429沿原client最多重试2次，HTTP attempts/retries与逻辑次数分列，不补采样。

每次新请求前复核冻结依赖、配置、输入和实际prompt；primary须匹配本轮冻结字节。repair依赖真实primary解析结果，未来才生成并按本臂模板、原raw=None规则、实际wrapper精确检查容量。超容、未知完成、预算/依赖/模板漂移等停止并记账，不能截断或改上限绕过。质量判断、去重和错误传播按review_policy进行，ordering单列；程序通过不代表内容通过。

生成及QA诊断完成后停止。上述命令不授权Miner、Free、Solver、Planner、Realizer、Memory、训练、v4/P2-2或额外审核API。live子命令必须显式出现；默认mode是prepare，无模型请求。
'''
 atomic_text(run/'EXECUTE_AFTER_AUTHORIZATION.md',commands)
 body=f'''准备已完成，状态 **READY_FOR_AUTHORIZATION / PREPARED_NOT_RUN**。本轮外部调用全部为零。

| 检查 | 结果 |
|---|---|
| 六条原始V3 intent | 两root各3原槽全部保留；intent_1/2/3、类型、变量、证据链均未改 |
| A原builder字节身份 | 六份primary及原repair基线一致，无新feedback suffix |
| B输入隔离 | contract语义、特定引用、OUTPUT SHAPE旧变量、repair/normalizer回填均清除；源语义变异不影响B |
| Writer执行 | 两臂与原generate_root_qa共用execute_writer_batch和finalize_writer_candidates；原路径仍Miner→Guided+Free |
| 原规则 | 2SC+1MS+1ordering minima、最多一次整批替换repair、最终validator/Gold重绑定保持 |
| 无网络测试 | {tests['tests_run']}项 PASS；日志见audit/test_report.json |
| primary离线请求 | 12份完整prompt与request spec，12份精确token容量检查PASS |
| 动态repair | 规则/builder及synthetic继承、超容保护测试已冻结；真实repair prompt 0份 |
| 依赖 | AST闭包{len(coverage['repository_files'])}仓库文件；dry-run实际加载{len(coverage['observed_repository_modules'])}文件，覆盖PASS |
| 来源保全 | {pres['file_count']}个登记文件逐一SHA/size复核，变化{pres['changed_count']}；非历史全量重扫 |

唯一实验方法差异是是否提供原Miner完整intent包（考点分配、类型与证据链）。必要模板差异共有四处：for this intent改为for this batch；contract软指导句替换为一条中性自主选材说明；删除CONTRACT GUIDANCE段；OUTPUT SHAPE中旧变量改为普通...占位符。不是只删除JSON段。完整对白、格式、候选字段和题型目标保持相同。B仍由Writer自己生成变量、真实refs与候选证据组；opaque slot ID只作归档身份。B内部empty contract groups产生的alignment false原样保留，离线对齐分析NOT_APPLICABLE，不作为质量失败；未伪造semantic_valid/contract_valid。

输入仅来自指定SOURCE_RUN原V3输出与完整R02 transcript。history_only、provenance仅离线跟踪，未进入任何prompt；没有cards、成绩、bindings、briefs、额外共同反馈政策或Codex修正示例。历史Miner 1/6和NOT_READY保持，当前审计示例完全不作为模型输入。

Writer配置来自指定WRITER_REFERENCE的generation及实际primary/repair请求核对：Qwen/Qwen3-14B、temperature0.1、thinking=false、不发送seed、8192输出、180秒、原429 retry limit2。两臂generation_source均contract_guided。原HTTP客户端的既有传输规则保留，不把Miner4096套到Writer；本轮没有访问/models或其他网络端点。

12份primary的实际单user聊天wrapper输入共 **{tokens['primary_input_tokens']} tokens**：A={tokens['by_arm'][ARMS[0]]}，B={tokens['by_arm'][ARMS[1]]}。primary输出预算上限98,304 tokens；若12格都触发repair，另有最多98,304输出预算，repair输入随尚未产生的primary响应而未知。这是未来预算而非已耗usage；实际成本/金额未知，同次数不表示同token成本。

复用了本地不可变tokenizer revision {identity['tokenizer']['revision']} 与chat template；每份input+8192+512不超过既有服务证据的{identity['context_limit']}上下文上限。service evidence采集时间保存在源manifest（2026-09-15）；本轮日期为2026-09-16，未重新查询服务。既有同服务wrapper核验{identity['operational_wrapper_verification']['verified_completed_count']}条匹配，服务不暴露checkpoint/tokenizer revision，不能把本地离线核验称为服务当前状态认证。模型能力、Gold语义和QA质量尚未评估。

repair真实prompt无法事先冻结，本轮冻结的是同臂builder、输入原primary解析对象的规则（JSON失败时raw=None仍走原{{}}渲染）、一次整批替换和动态容量机制。synthetic response仅用于测试，不归档为真实QA。未来实际repair必须逐次准确计数；容量不足即停，不裁剪任何内容、不改输出上限。已完成缓存按全局身份复用，相同B模板三次仍为三个不同预定调用；两臂隔离ledger/cache/token结果，未知完成状态停止后续。

预冻结质量规则以候选+完整对白为第一阶段材料，尽量隐藏arm/原intent；质量完成冻结后才开放去重/覆盖与A错误传播分析。标记Codex offline diagnostic，不宣称独立盲审或人工真值。主指标是各root-arm最终结构接受且质量PASS的SC/MS数及每3预定槽产量，root等权；质量PASS比例另列其明确分母。ordering单列，同时保留primary SC/MS和ordering触发repair的间接影响；全部失败、不可评、被替换primary及repair都留在对应层。

去重由内容阅读决定对象/事件/答案关系是否相同，指纹仅辅助；最终质量PASS SC/MS重复簇数作去重产量，不回写准入或补采样。history_only是有限stem/fingerprint池，缺options/Gold不作无根据语义去重。成本计入该臂全部实际生成，包括失败与被替换primary；零产量UNDEFINED_ZERO_YIELD，usage缺失明确不完整。本轮只有空审阅模板，没有QA结果或结论。

准确未来命令、固定12/24预算、恢复方式和停止条件见[EXECUTE_AFTER_AUTHORIZATION.md](../EXECUTE_AFTER_AUTHORIZATION.md)。本次没有运行真实Writer、Miner、Free、Solver、Memory或其他服务；没有创建已授权标记，没有安排后台任务，没有实现v4。至此停止，真实运行等待后续明确指令。
'''
 atomic_text(run/'reports/preparation_report.md',body)
 inventory=['EXPERIMENT_PLAN.md','RUN_STATUS.md','execution_contract.json','implementation_manifest.json','input_manifest.json','schedule.json','slot_call_manifest.jsonl','review_policy.json','audit/field_visibility_report.md','audit/prompt_diff_report.json','audit/repair_inheritance_report.json','audit/dependency_coverage_report.json','audit/test_report.json','audit/token_identity.json','audit/token_report.json','audit/token_budget_summary.json','audit/repair_capacity_test.json','audit/preflight_report.json','audit/frozen_manifest.json','audit/source_preservation_report.json','reports/preparation_report.md','EXECUTE_AFTER_AUTHORIZATION.md']
 atomic_text(run/'ARTIFACT_INDEX.md','当前状态 PREPARED_NOT_RUN。真实调用为零；所有离线模拟仅为测试夹具。\n\n'+'\n'.join(f'- [{p}]({p})' for p in inventory)+'\n\n六份原baseline见policy_snapshot/baseline_guided_prompts，12份primary见policy_snapshot/primary_requests；原输入在inputs/shared，冻结代码在policy_snapshot/frozen_code，repair真实prompt尚不存在。\n')
 atomic_json(run/'audit/delivery_check.json',dict(status='PASS',required_artifacts_present=all((run/p).is_file() for p in inventory),actual_external_calls=0,real_request_ledger_files=len(list(run.glob('arms/**/ledger/requests/*.json'))),live_authorization_marker_created=False,background_process_scheduled=False))
 return pre
