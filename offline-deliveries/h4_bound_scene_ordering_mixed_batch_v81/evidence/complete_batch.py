"""Write actual final summary only after independent result/union checks."""
from pathlib import Path
import sys,json,hashlib
HOME=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HOME))
from common import read,write,bind,now
E=HOME/'evidence'
audits={name:read(E/(name+'.json')) for name in ['GENERATION_AUDIT','INPUT_AUDIT','RESULT_AUDIT','UNION_AUDIT']}
assert audits['GENERATION_AUDIT']['status'] in ['PASS','PASS_WITH_GENERATION_AND_CONTENT_SHORTFALLS']
assert all(audits[name]['status']=='PASS' for name in ['INPUT_AUDIT','RESULT_AUDIT','UNION_AUDIT'])
new=read(HOME/'AUDITED_NEW_CANDIDATES.json');cumulative=read(HOME/'CUMULATIVE_AUDITED_NEW_CANDIDATES.json')
details=read(E/'RESULT_DETAILS.json');report=read(HOME/'MIXED_EVALUATION_REPORT.json')
allocation=read(HOME/'multiselect/ALLOCATION.json')
diagnosis=read(E/'ORDERING_FAILURE_DIAGNOSIS.json')
protected=read(HOME/'PRESERVATION_BEFORE.json')
assert all(bind(Path(p))['sha256']==digest for p,digest in protected['files'].items())
frozen={p['path']:p['sha256'] for p in read(HOME/'BATCH_PREPARATION.COMPLETE.json')['files']}
assert all(bind(p)['sha256']==frozen[str(p)] for p in [*HOME.glob('*.py'),*HOME.glob('*.md')])
assert bind(HOME/'SCENE_PLANS.json')==read(HOME/'CONFIG.json')['ordering_anchor_plan']
assert not list(HOME.rglob('__pycache__'))
lines=['本轮 v81 已完成固定单场景与原文锚点实验。多选继续使用冻结 v79 引擎，完整来源仍只有 194259／130221 两个 root；未扩对话、未训练、未正式入库。','',
 f"多选内容审核通过 {allocation['assembled_child_count']} 道新题、2 道旧 Gold6 对照。本轮新入选 {new['new_question_count']} 道，累计候选 {cumulative['candidate_count']} 道。",'',
 '| 多选 Gold 数 | 本轮新题内容通过 | 本轮新入选 | 累计新候选 |','|---|---:|---:|---:|']
for g in range(2,6):
 k=str(g);lines.append(f"| {g} | {allocation['assembled_distribution'].get(k,0)} | {new['gold_distribution'].get(k,0)} | {cumulative['gold_distribution'].get(k,0)} |")
excluded=[q for q in details['questions'] if q['role']=='new_child' and not q['models']['trained']['selected_by_policy']]
lines+=['','通过内容审核但未入选的新题：'+ '；'.join(f"{q['candidate_id']}（Gold{q['gold_count']}，训练后完全正确率 {q['models']['trained']['full_exact_rate']}，原因 {q['models']['trained']['selection_reason']}）" for q in excluded)+'。']
lines+=['','| 比较范围 | 训练前 M0 | 第一轮训练后 M_clean_policy |','|---|---:|---:|']
for role,label in [('new_child','本轮新多选题'),('historical_control','旧 Gold6 对照'),('all_questions','全部评测题')]:
 values=[details['metrics'][side][role] for side in ['base','trained']]
 text=[f"{v['exact_correct']}/{v['calls']} = {v['exact_percent']:.2f}%" if v['calls'] else '未测评' for v in values]
 lines.append(f'| {label} | {text[0]} | {text[1]} |')
lines+=['',f"两模型各 {report['calls_per_model']} 次作答，使用相同完整题目、显示排列、种子、解码和评分。难度筛选仍按训练后平均部分分严格位于 0 与 1 之间。每题选择原因和全部 raw 回答见 RESULT_DETAILS／PAIRED_CALLS。跨批题目组成不同，不能仅凭不同批正确率断言受控难度变化或训练退化。",'',
 '排序安排 4 道：24 处固定证据全部精确，所有锚点都在公开单场景内，0 个历史整套六项重复。3 道 writer Gold 与锚点顺序一致；网球题将 T003 排在 T002 前，被直接拒绝，未自动修正。另 3 道的核心事实、状态和范围内顺序唯一性通过完整来源审核，但 12 个题项出现原文未确证的性别代词，全部因 actor_ok 失败排除，0 道进入 Solver。','',
 '相对 v80，准确引用从 16/24 提升到 24/24，历史整组六项重复从 4/4 降到 0/4。本轮四题未再出现上述引用、场景锚点和整套重复问题；这一小样本尚不能证明稳定生成能力。当前失败集中于人物指称和一道 Gold 顺序错误；保留场景及审核方法，不再同时修改多选。','',
 '130221 的唯一补写仍未得到第4个可靠反例：联系其他机构或雇专业策划可与原文活动并存，不能把缺少证据当作明确矛盾。Gold2 因反例不足缺一题；Gold3 可用旧组合已在去重历史中，不能当作新增重复计算。','',
 '下一步最小修复可把这三道事实/顺序已通过的失败稿仅作人物指称中性化，另存新候选并重新独立审核、测评。保留原稿和原判，不纠正那道错误 Gold 的题目，不放宽来源或唯一顺序标准。当前批次没有实施该后续修复。','',
 '两组多选完整来源审核及排序审核都在打开当前私有 Gold 前保存，历史全对父题先验及规划背景按记录披露；不称作全新盲审。固定证据和程序审计不替代语义正确性。候选共享父题事实，整套文本去重不等于语义去重、独立事实创新。','',
 '独立审计：'+ '；'.join(f"{name} {a['status']}（{a['check_count']} 检查）" for name,a in audits.items())+'。',
 f"v74–v80 共 {len(protected['files'])} 个受保护文件保持字节不变。所有成功/失败稿、来源、模型输出和审核入口见 ARTIFACT_INDEX.md。"]
with (E/'RESULTS.md').open('x') as f:f.write('\n'.join(lines)+'\n')
paths=sorted(p for p in HOME.rglob('*') if p.is_file() and p.name!='RUN_COMPLETE.json')
record={'completed_at_utc':now(),'status':'COMPLETED_WITH_ORDERING_GENERATION_SHORTFALL',
 'new_selected_count':new['new_question_count'],'cumulative_candidate_count':cumulative['candidate_count'],
 'cumulative_multiselect_gold_distribution':cumulative['gold_distribution'],
 'generation_http_calls':audits['GENERATION_AUDIT']['totals']['actual_generation_http_calls'],
 'solver_calls_per_model':report['calls_per_model'],'total_solver_calls':2*report['calls_per_model'],
 'ordering_new_selected_count':sum(q['question_type']=='ordering' for q in new['questions']),
 'training_started':False,'formal_training_admission':False,'merged_into_baseline':False,
 'preserved_previous_file_count':len(protected['files']),'frozen_code_and_plan_unchanged':True,
 'files':[bind(p) for p in paths]}
write(E/'RUN_COMPLETE.json',record)
print({k:v for k,v in record.items() if k!='files'})
