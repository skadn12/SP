"""Render Day 1 actual work; never present planned slots as generated data."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json

BASE=Path(__file__).resolve().parent


def main():
    ledger=json.loads((BASE/'reviews/ADMISSION_LEDGER.json').read_text())
    valid=json.loads((BASE/'VALIDATION.json').read_text())
    audit=json.loads((BASE/'audit/INDEPENDENT_DAY1_AUDIT.json').read_text())
    assert valid['status']=='PASS' and audit['status']=='PASS'
    now=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    counts=ledger['status_counts']
    pass_n=counts.get('PASS_INTERNAL_CONTENT',0)
    hold_n=sum(v for k,v in counts.items() if k.startswith('HOLD'))
    fail_n=sum(v for k,v in counts.items() if k.startswith('FAIL'))
    assert pass_n+hold_n+fail_n==113
    lines=[]
    for typ,label in [('multi_select','多选'),('ordering','排序')]:
        c=ledger['by_type'][typ]
        lines.append(f"| {label} | {sum(c.values())} | {c.get('PASS_INTERNAL_CONTENT',0)} | {sum(v for k,v in c.items() if k.startswith('HOLD'))} | {sum(v for k,v in c.items() if k.startswith('FAIL'))} |")
    report=f'''# 第1天完成记录

完成时间：{now}，北京时间。状态：**DAY1_COMPLETE**。本文件报告实际已做工作，后续执行使用新目录，不覆盖本目录封存。

第一天目标已完成：实验设计和起点已绑定，评估边界明确，弱点与memory数据结构可验证，首批113道旧多选/排序完成来源先行重审。没有启动新的Solver评估、对话扩充或训练；语义重审由三个内部模型任务实际完成。

## 实验设计与起点

- A为多样化、无Solver失败反馈；B为多样化＋真实训练侧诊断反馈，初始70%定向、30%自由探索。
- A1/B1从同一M-clean开始，A2/B2分别从本分支第一阶段模型继续。旧Round2保留为回归比较。
- 已对base、M-clean合并模型及adapter、旧Round2合并模型及adapter逐文件计算SHA256；不复制权重，也不重新合并。
- 固定20＋10个Train root，memory试验取第一波前6个。A/B各180个计划题位、各90多选＋90排序，Gold2/3/4/5目标23/23/22/22。**这是题位安排，生成题数为0。**
- 冻结了调用/单次修订预算、Gold无关的选项置换、同等训练资源比较和两次更新的模型谱系。最终训练数据、实际token/微步日程、控制器实现及运行参数须在执行前另行预检冻结；设计文件不等于现已跑通生产控制器。旧训练器会一次性token化且可能截断超长输入，新包装器必须先执行零截断检查并实现逐epoch置换，不能直接以旧行为冒充新协议。

入口：[PROTOCOL_V1.json](PROTOCOL_V1.json)、[MODEL_BINDINGS.json](MODEL_BINDINGS.json)、[SLOT_PLAN.json](SLOT_PLAN.json)。文件名与私有映射可见性等机械澄清保存在MECHANICAL_CORRECTIONS.json；未依据新Solver分数变更设计。

## 数据与评估边界

权威划分101root=70Train＋15DEV＋16TEST。现有第二轮用了68Train；未入池的194259、87273仍是已暴露Train。旧DEV/TEST标记为历史暴露回归集，不能称全新测试；DEV只允许预定汇总指标选模型，TEST不反馈生成。真正未见root的前瞻评估集尚未建立，作为后续泛化主张的门槛保留。

同题改写、共享事实题族及所有排列不得跨训练/反馈/锁定评估边界；题级错误进入memory后，不再算未反馈迁移证据。当前角色限制是工作流规范，不是文件系统权限隔离。详见[evaluation/README_ZH.md](evaluation/README_ZH.md)。

## 首批历史题的实际重审

113题=51多选＋62排序，覆盖56个Train root；原123题中的10单选单列。56份完整source已原字节复制并绑定。公开重审包不含Gold、作者证据和旧分数；三个新任务先从完整source重建、保存，之后才由父任务核对旧Gold与逐字引文。

| 类型 | 重审题数 | 内部内容通过 | 暂缓HOLD | 失败FAIL |
|---|---:|---:|---:|---:|
{chr(10).join(lines)}
| 合计 | 113 | {pass_n} | {hold_n} | {fail_n} |

通过要求原题有明确支持、排序唯一、逐字证据有效且独立重建与原Gold一致。HOLD包括重复锚点导致多种顺序、计划/完成或人物归属不清、宽泛类别或因果关系证据不足；不把不确定强判为错误。14道FAIL中，11道为内部重建与原Gold不一致，3道为来源审核失败，均保留具体证据供后续裁决。原题、原Gold均未改，未按Solver分数反向筛题。

这批是历史表现筛选过的Train候选，不能把其通过率当所有旧题的总体质量。内部审核共享模型/文件系统，不是外部盲审。PASS_INTERNAL_CONTENT表示可进入训练池准备，**当前训练准入仍为0**，需再通过格式/映射、token化、去重、预算及最终池冻结。初始INVENTORY保留当时PENDING状态，最新结论以ADMISSION_LEDGER为准。

这批候选不能整体复用。旧题20%是后续训练混合目标，当前4道内容候选不保证能满足；若数量不足，应另行扩大重审范围或以新版本对两臂统一调整比例，不能反复复制少量题凑数或降低审核标准。

- [逐题结论CSV](reviews/ADMISSION_LEDGER.csv)
- [完整审核账本及证据入口](reviews/ADMISSION_LEDGER.json)
- [内部内容通过的原题](reviews/PASS_CONTENT_CANDIDATES.json)
- [来源先审规则](SOURCE_FIRST_REVIEW_POLICY.md)

## 弱点与memory准备

已分开观察错误、原因假说与内容/协议缺陷。至少3个Train root复现并有受控探针支持才升级原因状态；频次按root/题族去重。Gold错误、题意歧义、格式错误及缺证据不算模型能力弱点。

弱点memory和对话事实memory采用独立schema。弱点memory只给B Challenger，初始化快照为0失败、0原因，未把v140的DEV现象伪装成当前S0的训练侧弱点。事实memory只允许已审核前缀、原文引用、有效状态和修订关系，禁止未来事件或按当前题Gold定制。schema边界测试27项通过。非空memory首次运行前仍需核查改善方向、模型谱系与状态时间先后；事实memory生产方案的定量采纳/裁决条件须在小试读分前固定。

入口：[诊断协议](diagnosis/DIAGNOSIS_PROTOCOL_ZH.md)、[空弱点memory](WEAKNESS_MEMORY_B_v000.json)、[事实memory schema](diagnosis/FACT_MEMORY_SCHEMA.json)。

## 下一步已准备的输入

从现有已审核v125版本按root/type及固定哈希选择，准备20个Train root的39题：19多选＋20排序，无成绩选择。它们与本次重审的历史113题并非同一批题目/来源版本，不能互相套用审核结论。162896缺一道多选，保留缺额。已分出公开输入与私有Gold，计划直接答案582次，实际执行0次；结构化依据支路另设预算，保持Solver先答、评审后见Gold。

下一步在新目录验证运行接口并冻结请求，采集S0回答及小批结构化依据，形成第一版非空弱点memory；同时实施6root事实memory三臂小试。旧题HOLD/FAIL另留修复账本，不阻塞使用现有可靠训练版本做诊断。

入口：[诊断输入计划](diagnosis_inputs/INPUT_PLAN.json)、[公开39题](diagnosis_inputs/PUBLIC_QUESTIONS.json)。

## 验证与交接

父级机械核对{valid['check_count']}项、0错误；独立内部审计PASS，详见[VALIDATION.json](VALIDATION.json)与[audit/INDEPENDENT_DAY1_AUDIT.json](audit/INDEPENDENT_DAY1_AUDIT.json)。这些检查验证来源绑定、映射边界和统计，不把语义真值或访问隔离当已获外部认证。

COMPLETE_MANIFEST.json绑定本目录文件；本周后续实验另建目录，保留本目录为第1天审计快照。
'''
    with (BASE/'REPORT_ZH.md').open('x',encoding='utf-8') as f:f.write(report)
    status={'status':'DAY1_COMPLETE','completed_at_beijing':now,'goals':{
        'AB_design_and_model_start_bound':True,'root_and_evaluation_boundary_bound':True,
        'weakness_taxonomy_and_memory_schemas_validated':True,'legacy_source_review_complete':True,
        'next_diagnosis_inputs_prepared':True},
        'legacy_review':{'questions':113,'roots':56,'content_pass':pass_n,'hold':hold_n,'fail':fail_n,'training_admitted':0},
        'future_question_slots':360,'new_generated_questions':0,'next_diagnosis_questions':39,'next_diagnosis_planned_direct_calls':582,
        'new_solver_calls':0,'new_training_runs':0,'semantic_reviewer_tasks':3,
        'future_gates':['Runtime implementation/preflight and exact request freeze before Solver calls.',
                        'Nonempty memory state chronology, model lineage and actual improvement direction must be validated before admission.',
                        'Actual source/memory pilot quality criteria and exact parameters fixed before pilot execution.',
                        'Dynamic legacy format mapping, tokenization and paired training budget freeze before training.',
                        'New prospective unseen-root benchmark before claiming unseen-root generalization.']}
    with (BASE/'FINAL_STATUS.json').open('x',encoding='utf-8') as f:json.dump(status,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(status,ensure_ascii=False))


if __name__=='__main__':main()
