"""Describe saved source-ID predicate crosschecks; never alter a judgment or answer."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import json,hashlib
BATCH=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def bind(p):return {'path':str(Path(p).resolve()),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
def run():
 if (BATCH/'evidence/RUN_COMPLETE.json').exists():raise ValueError('already_sealed')
 content_path=BATCH/'ordering/CONTENT_REVIEW.json';judgments_path=BATCH/'ordering/SOURCE_JUDGMENTS.json';complete_path=BATCH/'ordering/CONTENT_REVIEW.COMPLETE.json'
 if not complete_path.exists():raise ValueError('content_review_not_complete')
 content=read(content_path);judgments={q['candidate_id']:q for q in read(judgments_path)['questions']}
 files={Path(__file__).resolve(),content_path,judgments_path,complete_path};questions=[]
 for check in content['questions']:
  root=check['root_id'];run_dir=BATCH/'ordering'/('root'+root.rsplit('-',1)[1]+'_ordering')
  plan_path,gold_path,pool_path,writer_path=(run_dir/name for name in ('ANCHOR_PLAN.json','GOLD_PLAN.json','POOL.json','WRITER_PLAN.json'))
  files.update((plan_path,gold_path,pool_path,writer_path))
  plans={q['candidate_id']:q for q in read(plan_path)['questions']};answers={q['candidate_id']:q for q in read(gold_path)['questions']};pool={q['candidate_id']:q for q in read(pool_path)['questions']};writers={q['candidate_id']:q for q in read(writer_path)['questions']}
  qid=check['candidate_id'];plan=plans[qid];answer=answers[qid];q=pool[qid];labels=[]
  response_path=next((run_dir/'requests').glob('*/RESPONSE.json'));files.add(response_path)
  raw=next(r for r in json.loads(read(response_path)['text'])['questions'] if r is not None and r['candidate_id']==qid)
  records={r['anchor_id']:r['predicate'] for r in raw['records']};prefixes={r['anchor_id']:r['statement_prefix'] for r in writers[qid]['anchor_records']}
  for label,fixed in plan['anchors'].items():
   independent=check.get('order_anchor_spans',{}).get(label);position=answer['anchor_positions'][label]
   if independent is None:kind='NO_RESOLVED_INDEPENDENT_ANCHOR'
   elif independent['turn_ref']!=fixed['turn_ref']:kind='PRIMARY_MATCH_DIFFERENT_SOURCE_TURN'
   elif not(position['quote_start']<=independent['char_start'] and independent['char_end']<=position['quote_end']):kind='SAME_TURN_OUTSIDE_FIXED_QUOTE_SPAN'
   else:kind='FIXED_SOURCE_ID_ANCHOR_MATCH'
   other=[other for other,a in plan['anchors'].items() if independent is not None and a['turn_ref']==independent['turn_ref'] and a['quote']==independent['quote'] and other!=label]
   assembled=prefixes[fixed['turn_ref']]+records[fixed['turn_ref']]
   labels.append({'label':label,'anchor_id':fixed['turn_ref'],'writer_predicate':records[fixed['turn_ref']],'frozen_statement_prefix':prefixes[fixed['turn_ref']],'public_item':q['items'][label],'exact_id_projection_matches_public_item':assembled==q['items'][label],'classification':kind,'fixed_anchor':fixed,'independent_primary_anchor':independent,'exact_other_fixed_quote_labels':other,'saved_native_item_judgment':{k:judgments[qid]['items'][label][k] for k in ('supported','actor_ok','modality_ok','state_ok')}})
  counts=Counter(r['classification'] for r in labels)
  questions.append({'candidate_id':qid,'root_id':root,'content_status':check['status'],'labels':labels,'label_classification_counts':dict(counts),'saved_native_unique_order':judgments[qid]['unique_order'],'independent_order':check.get('computed_order'),'program_gold':answer['gold'],'independent_order_agrees_with_program_gold':check.get('computed_order')==answer['gold'],'diagnostic_conclusion':'At least one independently identified primary event differs from the fixed source-ID anchor; inspect saved semantic judgments. The program mapping was not repaired.' if counts['PRIMARY_MATCH_DIFFERENT_SOURCE_TURN'] else 'At least one independent evidence span extends beyond the fixed quote in the same turn; this is distinct from a cross-turn event mismatch.' if counts['SAME_TURN_OUTSIDE_FIXED_QUOTE_SPAN'] else 'All resolved primary source anchors fit the corresponding frozen quotes.' if counts['FIXED_SOURCE_ID_ANCHOR_MATCH']==6 else 'Some items lack a resolved independent primary anchor; no semantic success is inferred.'})
 summary={'mechanically_accepted_questions':len(questions),'independently_supported_and_unique_questions':sum(q['saved_native_unique_order'] and all(all(i['saved_native_item_judgment'].values()) for i in q['labels']) for q in questions),'questions_with_cross_turn_primary_anchor_mismatch':sum(q['label_classification_counts'].get('PRIMARY_MATCH_DIFFERENT_SOURCE_TURN',0)>0 for q in questions),'cross_turn_mismatched_items':sum(q['label_classification_counts'].get('PRIMARY_MATCH_DIFFERENT_SOURCE_TURN',0) for q in questions),'questions_with_same_turn_span_failure':sum(q['label_classification_counts'].get('SAME_TURN_OUTSIDE_FIXED_QUOTE_SPAN',0)>0 for q in questions),'independent_unique_but_program_answer_mismatch_questions':sum(q['saved_native_unique_order'] and not q['independent_order_agrees_with_program_gold'] for q in questions),'program_source_id_projection_mismatches':sum(not label['exact_id_projection_matches_public_item'] for q in questions for label in q['labels'])}
 record={'version':'v88_source_id_anchor_mismatch_diagnostic_v1','created_at_utc':datetime.now(timezone.utc).isoformat(),'private_diagnostic_not_for_independent_reviewer':True,'summary':summary,'questions':questions,'inputs':[bind(p) for p in sorted(files)],'authoritative_judgments_changed':False,'question_or_gold_edits':[],'limitations':['Based only on already saved independent source primary anchors and official crosschecks. This is not an independent recertification.','The program assigns private labels from anchor_id. A predicate may still describe the wrong event; exact ID coverage and correct program assembly do not certify meaning.','A different independent primary turn, same-turn quote-span failure, lack of unique order and disagreement with program Gold are separate conditions.','No rejected item is relabeled, rewritten or given a new Gold. Missing review results are not passes.']}
 p=BATCH/'evidence/ORDERING_SOURCE_ID_MISMATCH_DIAGNOSTIC.json'
 with p.open('x') as f:json.dump(record,f,ensure_ascii=False,indent=2);f.write('\n')
 lines=['v88 排序 source-ID 对应诊断','','本记录仅解释已保存的独立审核和私有交叉结果。谓词按原始 anchor_id 与冻结姓名前缀组装，不根据含义换位，也不修正 Gold。','','| Root / slot | 不同 turn 主锚点不符 | 同 turn 引文范围超界 | 独立顺序唯一 | 与程序 Gold 一致 | 内容 |','|---|---:|---:|---|---|---|']
 for q in questions:lines.append('| '+' | '.join([q['root_id'].rsplit('-',1)[1]+' / '+q['candidate_id'].rsplit('-',1)[1],str(q['label_classification_counts'].get('PRIMARY_MATCH_DIFFERENT_SOURCE_TURN',0)),str(q['label_classification_counts'].get('SAME_TURN_OUTSIDE_FIXED_QUOTE_SPAN',0)),str(q['saved_native_unique_order']),str(q['independent_order_agrees_with_program_gold']),q['content_status']])+' |')
 lines+=['','该记录含私有锚点与 Gold，不提供给尚未完成独立判断的审核者。完整原始谓词、组装结果与证据绑定见同名 JSON。']
 with p.with_suffix('.md').open('x') as f:f.write('\n'.join(lines)+'\n')
 return summary
if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False,indent=2))
