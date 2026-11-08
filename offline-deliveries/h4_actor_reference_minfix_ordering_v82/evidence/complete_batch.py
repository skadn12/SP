from pathlib import Path
from fractions import Fraction
import sys
HOME=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HOME))
from common import read,write,bind,now,sha
import batch
batch.verify_frozen(HOME)
audits={n:read(HOME/'evidence'/(n+'.json')) for n in ['INPUT_AUDIT','RESULT_AUDIT','UNION_AUDIT']}
assert all(a['status']=='PASS' for a in audits.values())
protected=read(HOME/'PRESERVATION_BEFORE.json')['files']
assert all(sha(p)==s for p,s in protected.items())
assert not list(HOME.rglob('__pycache__'))
new=read(HOME/'AUDITED_NEW_CANDIDATES.json');union=read(HOME/'CUMULATIVE_AUDITED_NEW_CANDIDATES.json')
details=read(HOME/'evidence/RESULT_DETAILS.json');review=read(HOME/'ordering/CONTENT_REVIEW.json')
report=read(HOME/'MIXED_EVALUATION_REPORT.json')
lines=['v82 已完成三道排序失败稿的人物指称最小修复。共调整12个题项，保留题干、标签、原文、引用和原 Gold；错误 Gold 的第四道题未参与本次修复。未调用生成模型，多选方法及17道累计多选候选不变。','',
 f"独立完整来源审核通过 {review['accepted_count']}/3 道；难度筛选最终入选 {new['new_question_count']} 道排序。累计候选 {union['candidate_count']} 道，类型分布 {union['type_distribution']}；多选 Gold2/3/4/5 数量仍为4/4/3/6。",'',
 '| 模型 | 完全正确率 | 平均部分分 |','|---|---:|---:|']
for side,label in [('base','训练前 M0'),('trained','第一轮训练后 M_clean_policy')]:
    m=details['metrics'][side]['all_questions']
    lines.append(f"| {label} | {m['exact_correct']}/{m['calls']} = {m['exact_percent']:.2f}% | {m['partial_mean']} |")
lines+=['','| 题目 | 训练前完全正确率 | 训练后完全正确率 | 入选 |','|---|---:|---:|---|']
for q in details['questions']:
    values=[q['models'][s] for s in ['base','trained']]
    lines.append(f"| {q['candidate_id']} | {values[0]['full_exact_rate']} | {values[1]['full_exact_rate']} | {'是' if values[1]['selected_by_policy'] else '否：'+values[1]['selection_reason']} |")
lines+=['',f"两模型各 {report['calls_per_model']} 次调用，每道题12次独立种子排列，提示、来源、排列、种子和评分保持配对。难度入选规则仍为所有回答有效且训练后平均部分分严格位于0和1之间。完全正确率是描述指标，不直接决定入选。",'',
 '人物档案已完成4个人物的姓名、角色和原文事实整理；性别及第三人称代词没有可靠设定的部分保留未知。Liam怀孕、被称为准妈妈是有引用的事实，不据此填入he/she。档案仅供后续生成规划，当前Solver仍只看到原始完整对话及公开题面。','',
 '新审核员在独立上下文中完整阅读149/91轮来源，先保存本轮判断，再由程序与保留的私有Gold交叉检查。历史失败稿修复来源公开披露，不称为从未见过原材料的普遍盲审证明。人物档案和程序校验不替代语义审核。','',
 '本轮是有针对性的三题修复实验，不能据此声称批量排序已可靠生成。v81没有排序Solver测评，因此不能声称排序准确率相对v81提高或下降。新增候选仍依赖两份来源及既有事实；扩规模还需新来源和独立抽检。','',
 '本轮未训练、未正式入库，也未修改旧baseline。下一轮可以在相同场景/引用/审核规则下，把无可靠代词的人物默认使用姓名作为writer约束，做小批混合生成验证。', '',
 f"独立输入、结果、累计导出审计均通过；{len(protected)}个历史受保护文件字节不变。全部题目、逐项修改、来源、审核和模型回答入口见ARTIFACT_INDEX.md。"]
with (HOME/'evidence/RESULTS.md').open('x') as f:f.write('\n'.join(lines)+'\n')
files=sorted(p for p in HOME.rglob('*') if p.is_file() and p.name!='RUN_COMPLETE.json')
complete={'completed_at_utc':now(),'status':'COMPLETE','new_ordering_selected':new['new_question_count'],
 'content_usable_ordering':review['accepted_count'],'cumulative_candidate_count':union['candidate_count'],
 'type_distribution':union['type_distribution'],'multiselect_gold_distribution':union['gold_distribution'],
 'generation_http_calls':0,'solver_calls_per_model':report['calls_per_model'],
 'training_started':False,'formal_training_admission':False,'preserved_previous_file_count':len(protected),
 'files':[bind(p) for p in files]}
write(HOME/'evidence/RUN_COMPLETE.json',complete)
print({k:v for k,v in complete.items() if k!='files'})
