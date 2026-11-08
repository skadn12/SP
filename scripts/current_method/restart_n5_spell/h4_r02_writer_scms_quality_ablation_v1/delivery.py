"""Preparation delivery only. Never generates or authorizes real responses."""
import json
from pathlib import Path
from ..core import atomic_json,atomic_text
from .constants import VERSION,REPO,CONFIG,ARMS
from .guidance import guidance_text,GUIDANCE_SHA256


def write_delivery(run):
    run=Path(run);read=lambda n:json.loads((run/n).read_text())
    pre=read('audit/preflight_report.json');tests=read('audit/test_report.json');tokens=read('audit/token_budget_summary.json');identity=read('audit/token_identity.json');coverage=read('audit/dependency_coverage_report.json');preserved=read('audit/source_preservation_report.json')
    if pre['status']!='PASS':raise RuntimeError('DELIVERY_REQUIRES_PASSED_PREPARATION')
    from .reporting import report
    reviewed=report(run)
    if reviewed.get('status')!='PREPARED_NOT_RUN' or reviewed.get('errors') or list(run.glob('**/ledger/requests/*.json')):raise RuntimeError('PREPARATION_DELIVERY_REQUIRES_ZERO_REAL_STATE_AND_CLEAN_REPORT')
    atomic_text(run/'RUN_STATUS.md','''RUN_STATUS = PREPARED_NOT_RUN
PREPARATION_STATUS = READY_FOR_AUTHORIZATION
LIVE_STATUS = PREPARED_NOT_RUN
ACTUAL_EXTERNAL_CALLS = 0
PRIMARY_WRITER_CALLS = 0
REPAIR_CALLS = 0
MINER_CALLS = 0
FREE_CALLS = 0
SOLVER_CALLS = 0
PLANNER_CALLS = 0
REALIZER_CALLS = 0
MEMORY_CALLS = 0
TRAINING_CALLS = 0
EXTRA_REVIEW_API_CALLS = 0

本轮仅离线准备。未来12 primary＋最多12 repair，总最多24逻辑调用，失败计入。真实运行需要下一条明确指令。没有授权标记或后台任务；没有使用旧响应作本轮结果。\n''')
    prefix=f'env PYTHONPATH={REPO} /root/miniconda3/bin/python -B -m scripts.current_method.restart_n5_spell.{VERSION}.run'
    atomic_text(run/'EXECUTE_AFTER_AUTHORIZATION.md',f'''本轮状态PREPARED_NOT_RUN，下面live命令仅供后续明确真实执行指令使用。本轮不执行，不创建授权文件，不安排后台任务。

首先离线核验已冻结身份（已冻结时仅复核，不覆盖准备或真实结果）：

```bash
{prefix} prepare --run-dir {run}
```

后续明确授权后，真实执行命令：

```bash
{prefix} live --run-dir {run} --config {CONFIG} --execute-after-authorization
```

恢复使用同一live命令、同目录/配置和全局身份。固定12 primary顺序见schedule.json，最多每格一次紧随的repair，总24逻辑上限，失败占预算；每臂6+≤6、每root-arm3+≤3。原HTTP429最多重试2次，HTTP尝试与逻辑调用分列。完成缓存只按本轮同arm/root/slot/stage/request_id复用，不读旧run响应、不按prompt SHA跨格合并。

两臂都source='contract_guided'、intent_guidance=True、hooks=None，通过原execute_writer_batch及各root-arm的finalize_writer_candidates。BASE suffix为空，QUALITY suffix为policy_snapshot/scms_quality_guidance_v1.txt（sha256 {GUIDANCE_SHA256}）；原文本不裁剪，质量块不混入反馈或例题。primary与repair都追加同一块且仅一次。

每次发送前复核冻结代码/输入/配置/manifest/预算/prompt及精确token容量。真实repair原文待primary响应产生，使用本臂原解析对象构建，None沿原{{}}逻辑；包含实际raw、intent和完整指导，8192 output+512reserve不变。超容或冻结漂移、预算异常、未知完成/付费状态和sticky guardrail立刻停，不repair或重发、不删ledger、不换ID/live_2、不改上限绕过。明确HTTP拒绝按原失败规则记账；没有额外采样或语义重试。

原2SC+1MS+1ordering minima触发一次整批替换repair；不合并、不挑选、不回收失败primary。final validator拒绝不再repair，单Gold MS原拒绝不补Gold，ordering失败仍可能导致整批SC/MS丢弃。排序显式规则不变，不保证mixed输出不受间接影响。

离线报告入口（无服务请求）：

```bash
{prefix} report --run-dir {run}
```

未来先审所有primary/repair可解析候选、冻结质量，再做两臂原intent对照、去重/共享事实覆盖/错误传播与有限history对照。报告输出SC、MS、合计及ordering分层；差值QUALITY−BASE，root等权。按新review_policy和空schema填写，完成状态通过完整性检查决定，不能把未审项当通过。生成与诊断后停止，不运行Solver或其他角色。命令不授权Miner、Free、Planner、Realizer、Memory、训练、v4/P2-2、排序修复或扩样。
''')
    body=f'''离线准备已完成：**READY_FOR_AUTHORIZATION / PREPARED_NOT_RUN**。外部调用0，真实Writer primary/repair、Miner、Solver及其他角色均0；未来效果指标保持NOT_RUN，没有QA质量或收益结果。

唯一新增方法要求是任务书冻结的SC/MS质量指导块：明确题干范围、原intent只是指导而非事实、逐选项支持、单选唯一且资格词有据、多选all-and-only且至少两个真实Gold、人物/阶段/意向与完成/事件时间/因果强度不混淆。块全文见下方；它是提示，不是新增程序语义validator，也不证明模型实际执行了内部自检。

BASE_WRITER与SCMS_QUALITY_WRITER都传原完整contract、完整对白，显式intent_guidance=True，source=contract_guided、hooks=None。仅QUALITY使用原slot_suffix追加质量块；BASE suffix为空。原execute_writer_batch/finalize_writer_candidates直接复用，共享generation/Guided builder/normalizer/validator/Gold代码本轮未改。

| 检查 | 结果 |
|---|---|
| 数据 | 两root各3原槽，全部6条原V3 intent/原ID/type/variable/refs保留，无筛选修正 |
| 来源 | 指定SOURCE_WRITER逐字复制，并与SOURCE_MINER原raw/parse/contracts/derived/schema及冻结输入交叉核验 |
| BASE身份 | 六份重新渲染BASE与上一轮A冻结prompt及实际发出prompt双重字节相同 |
| QUALITY差异 | 每份严格为BASE加一个换行和同一3358-byte固定块，去除后逐字还原BASE |
| repair继承 | 同synthetic raw与None均核验原repair＋唯一相同块；真实repair尚0份 |
| 两臂intent语义 | manifest→实际True参数→prompt→normalizer→sidecar→报告均适用；未继承旧第二臂关闭或NA路径 |
| 原生成规则 | 2SC/1MS/1ordering minima、超额合法行、最多一次整批替换、final拒绝不重试、单Gold MS终检拒绝保持 |
| 无网络回归 | {tests['tests_run']}项通过；必要mock dry-run通过 |
| token | 12份primary全部精确wrapper核验通过，input+8192+512≤{identity['context_limit']} |
| 执行依赖 | {len(coverage['repository_files'])}个仓库闭包文件，dry-run观测{len(coverage['observed_repository_modules'])}个，覆盖通过 |
| 历史保全 | 登记{preserved['file_count']}个来源/共享依赖逐一SHA+size复核，变化{preserved['changed_count']}；非全历史扫描 |

排序的显式指令、schema、配额、repair与validator未改。新指导虽只对SC/MS提出内容要求，同一个mixed响应的长度、排序输出和repair概率仍可能间接受影响，未来须记录。没有排序Gold修复、部分接收SC/MS或程序补选/裁剪Gold。

12份primary确定输入预算共{tokens['primary_input_tokens']:,} tokens，BASE={tokens['by_arm'][ARMS[0]]:,}，QUALITY={tokens['by_arm'][ARMS[1]]:,}；primary输出上限98,304。条件repair最多12次、输出上限另98,304，实际repair输入未知。当前input/output/total usage全部0；未来金额无可靠账单依据为未知。每次primary/repair output上限8192、reserve512不变。

复用原本地tokenizer revision {identity['tokenizer']['revision']}、chat template及2026-09-15采集的service evidence；已有wrapper核验{identity['operational_wrapper_verification']['verified_completed_count']}条completed记录匹配。本轮未访问/models、未下载tokenizer或查询服务当前状态；托管模型revision未暴露，离线通过不能冒称当前服务认证。两臂synthetic超容repair均被阻断，实际repair发送前还需基于真实primary输出逐次精确token计数，不能截断或改上限。

质量规则沿用八维、逐项Gold/全部选项、FAIL优先/其次UNDETERMINED聚合及结构未评；第一阶段隐藏臂/原intent。冻结质量后两臂都进行原intent对照与源错误逐例诊断；错误选项本身不自动算错误传播。SC、MS与合计分别列primary/repair/final，ordering另列；固定三个槽/root-arm与六槽/arm，配对差QUALITY−BASE，两个root等权。新增完成条件核对全部审阅、去重与共享事实覆盖等材料，不沿用旧硬编码pending。空审阅文件不包含未来QA判断。

完整答案关系去重与共享事实覆盖分别诊断，簇数不等于独立知识点；历史池仅复制原history_only（9/11条），未扩入上一轮56题且不输入模型。成本包括本臂全部mixed调用、失败/被替换primary/repair与ordering，不伪拆SC/MS专属tokens；零产量未定义、usage缺失不算完整效率。未来若ordering或其他题型短缺丢弃可用SC/MS，需同时报告质量数与重复关系，不能直接称独特覆盖损失。

固定未来预算12 primary＋最多12 repair，总≤24逻辑调用，失败计入；恢复仅同一身份。本轮无需启动真实服务，已停止。准确未来命令见EXECUTE_AFTER_AUTHORIZATION.md，等待后续明确任务指令。原Miner评分及历史56题全部保持；未实施v4/P2-2或其他实验。

冻结块文件为policy_snapshot/scms_quality_guidance_v1.txt，UTF-8、LF换行、无末尾换行，sha256 `{GUIDANCE_SHA256}`。完整原文：

```text
{guidance_text(run)}
```
'''
    atomic_text(run/'reports/preparation_report.md',body)
    names=['EXPERIMENT_PLAN.md','RUN_STATUS.md','execution_contract.json','implementation_manifest.json','input_manifest.json','schedule.json','slot_call_manifest.jsonl','review_policy.json','policy_snapshot/CODEX_WORK_ORDER.md','policy_snapshot/scms_quality_guidance_v1.txt','policy_snapshot/repair_policy.json','audit/prompt_pairing_report.json','audit/repair_inheritance_report.json','audit/dependency_coverage_report.json','audit/test_report.json','audit/token_identity.json','audit/token_report.json','audit/token_budget_summary.json','audit/preflight_report.json','audit/frozen_manifest.json','audit/source_preservation_report.json','reports/preparation_report.md','EXECUTE_AFTER_AUTHORIZATION.md']
    assert all((run/n).is_file() for n in names)
    atomic_text(run/'ARTIFACT_INDEX.md','状态PREPARED_NOT_RUN；本轮实际调用0。\n\n'+'\n'.join(f'- [{n}]({n})' for n in names)+'\n\n完整输入位于inputs/shared；12份请求位于policy_snapshot/primary_requests；冻结代码位于policy_snapshot/frozen_code。真实repair prompt未产生，synthetic仅为测试夹具。\n')
    atomic_json(run/'audit/delivery_check.json',{'status':'PASS','required_artifacts_present':True,'actual_external_calls':0,'real_request_ledger_files':len(list(run.glob('**/ledger/requests/*.json'))),'live_authorization_marker_created':False,'background_process_scheduled':False})
    return pre
