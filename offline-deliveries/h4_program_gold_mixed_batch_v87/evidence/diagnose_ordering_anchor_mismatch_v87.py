"""Explain saved anchor crosschecks without changing source judgments or answers."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import json, hashlib
BATCH=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def bind(p):return {'path':str(Path(p).resolve()),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
if (BATCH/'evidence/RUN_COMPLETE.json').exists():raise ValueError('already_sealed')
content_path=BATCH/'ordering/CONTENT_REVIEW.json';judgments_path=BATCH/'ordering/SOURCE_JUDGMENTS.json'
content=read(content_path);judgments={q['candidate_id']:q for q in read(judgments_path)['questions']}
files={Path(__file__).resolve(),content_path,judgments_path};questions=[]
for check in content['questions']:
 root=check['root_id'];run=BATCH/'ordering'/('root'+root.split('-')[-1]+'_ordering')
 plan_path,gold_path,pool_path=(run/name for name in ('ANCHOR_PLAN.json','GOLD_PLAN.json','POOL.json'))
 files.update((plan_path,gold_path,pool_path))
 plans={q['candidate_id']:q for q in read(plan_path)['questions']};answers={q['candidate_id']:q for q in read(gold_path)['questions']};pool={q['candidate_id']:q for q in read(pool_path)['questions']}
 qid=check['candidate_id'];plan=plans[qid];answer=answers[qid];q=pool[qid];labels=[]
 for label,fixed in plan['anchors'].items():
  independent=check['order_anchor_spans'].get(label);position=answer['anchor_positions'][label]
  if independent is None:
   kind='NO_RESOLVED_INDEPENDENT_ANCHOR'
  elif independent['turn_ref']!=fixed['turn_ref']:
   kind='DIFFERENT_SOURCE_TURN_LABEL_MAPPING'
  elif not(position['quote_start']<=independent['char_start'] and independent['char_end']<=position['quote_end']):
   kind='SAME_TURN_OUTSIDE_FIXED_QUOTE_SPAN'
  else:kind='FIXED_LABEL_ANCHOR_MATCH'
  matches=[other for other,a in plan['anchors'].items() if independent is not None and a['turn_ref']==independent['turn_ref'] and a['quote']==independent['quote']]
  labels.append({'label':label,'public_item':q['items'][label],'classification':kind,'fixed_anchor':fixed,'independent_primary_anchor':independent,'exact_other_fixed_quote_labels':[x for x in matches if x!=label],'saved_native_item_judgment':{k:judgments[qid]['items'][label][k] for k in ('supported','actor_ok','modality_ok','state_ok')}})
 counts=Counter(r['classification'] for r in labels)
 every_moved_matches_other=all(r['exact_other_fixed_quote_labels'] for r in labels if r['classification']=='DIFFERENT_SOURCE_TURN_LABEL_MAPPING')
 questions.append({'candidate_id':qid,'root_id':root,'content_status':check['status'],'labels':labels,'label_classification_counts':dict(counts),'all_moved_labels_match_another_fixed_quote_exactly':every_moved_matches_other,'saved_native_unique_order':judgments[qid]['unique_order'],'independent_order':check['computed_order'],'program_gold':answer['gold'],'independent_order_agrees_with_program_gold':check['computed_order']==answer['gold'],'writer_labels_in_chronological_number_order':check['computed_order']==['I1','I2','I3','I4','I5','I6'],'diagnostic_conclusion':'The writer reassigned source statements among immutable labels; every differing independent primary anchor exactly matches another fixed quote. This is not a same-turn span-containment-only exclusion.' if counts['DIFFERENT_SOURCE_TURN_LABEL_MAPPING'] and every_moved_matches_other and not counts['SAME_TURN_OUTSIDE_FIXED_QUOTE_SPAN'] else 'Saved independent anchors remain within the corresponding fixed quotes.' if not check['reasons'] else 'See separate observed mismatch kinds; no automatic semantic recertification.'})
summary={'mechanically_accepted_questions':len(questions),'independently_supported_and_unique_questions':sum(q['saved_native_unique_order'] and all(all(i['saved_native_item_judgment'].values()) for i in q['labels']) for q in questions),'questions_with_true_cross_turn_label_reassignment':sum(q['label_classification_counts'].get('DIFFERENT_SOURCE_TURN_LABEL_MAPPING',0)>0 for q in questions),'cross_turn_reassigned_labels':sum(q['label_classification_counts'].get('DIFFERENT_SOURCE_TURN_LABEL_MAPPING',0) for q in questions),'questions_with_same_turn_span_only_failure':sum(q['label_classification_counts'].get('SAME_TURN_OUTSIDE_FIXED_QUOTE_SPAN',0)>0 for q in questions),'independent_unique_but_program_answer_mismatch_questions':sum(q['saved_native_unique_order'] and not q['independent_order_agrees_with_program_gold'] for q in questions)}
record={'version':'v87_ordering_label_mismatch_diagnostic_v1','created_at_utc':datetime.now(timezone.utc).isoformat(),'private_diagnostic_not_for_independent_reviewer':True,'summary':summary,'questions':questions,'inputs':[bind(p) for p in sorted(files)],'authoritative_judgments_changed':False,'question_or_gold_edits':[],'limitations':['Based on the already saved independent semantic primary anchors; no additional review or replacement of a saved judgment.','The two rejected103866 questions have one independent source order each; the failure is disagreement with the pre-frozen answer under changed label mapping, not lack of an order.','Earlier stage diagnostic unique_semantic_order rejection category includes program-answer mismatch. This appended record disambiguates independent uniqueness from answer agreement.','This does not authorize relabeling items or regenerating Gold for rejected questions.']}
p=BATCH/'evidence/ORDERING_LABEL_MISMATCH_DIAGNOSTIC.json'
with p.open('x') as f:json.dump(record,f,ensure_ascii=False,indent=2);f.write('\n')
lines=['v87 排序标签错位诊断','','103866 两题均被 writer 按对话时间重新编号为 I1–I6：第一题错5个label，第二题错3个label。八个错误标签的独立语义锚点都位于不同turn，并逐字匹配另一标签的固定引文。**没有同turn仅引文字符范围超界而误拒的案例。**','','四道进入独立审核的题目均被判定内容有依据且顺序唯一；103866两题的实际唯一顺序与随机冻结标签的程序Gold不一致，135466两题通过。这里应区分“源上顺序唯一”与“源上顺序符合冻结答案”。','','| Root / slot | 不同turn错位label数 | 同turn范围超界数 | 独立顺序唯一 | 与程序Gold一致 | 内容 |','|---|---:|---:|---|---|---|']
for q in questions:lines.append('| '+' | '.join([q['root_id'].split('-')[-1]+' / '+q['candidate_id'].rsplit('-',1)[-1],str(q['label_classification_counts'].get('DIFFERENT_SOURCE_TURN_LABEL_MAPPING',0)),str(q['label_classification_counts'].get('SAME_TURN_OUTSIDE_FIXED_QUOTE_SPAN',0)),str(q['saved_native_unique_order']),str(q['independent_order_agrees_with_program_gold']),q['content_status']])+' |')
lines+=['','题目、Gold、独立判断及既有阶段诊断均未修改。该诊断含私有交叉资料，不提供给未完成独立判断的审核者。']
with p.with_suffix('.md').open('x') as f:f.write('\n'.join(lines)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
