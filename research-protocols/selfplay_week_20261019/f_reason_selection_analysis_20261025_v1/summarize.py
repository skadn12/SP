import json,hashlib
from pathlib import Path
from collections import Counter
A=Path(__file__).parent;E=A.parent/'feedback_closed_loop_20261024_v1'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
def ck(name,condition):
 checks.append(dict(name=name,passed=bool(condition)))
 assert condition,name
for b in read(A/'INPUT_BINDINGS.json'):ck('input binding '+b['path'],sha(Path(b['path']))==b['sha256'])
M=read(E/'FINAL_MANIFEST.json')
for b in M['files']:ck('original sealed '+b['path'],sha(E/b['path'])==b['sha256'])
allrows=[];response_rows=[]
for root in ['218519','222937','225575','229686']:
 B=A/'reviews'/root;P=read(B/'PACKET.json');R=read(B/'REVIEW.json');T={x['turn_ref']:x['text'] for x in read(Path(P['source_binding']['path']))['turns']};by={r['candidate_id']:r for r in R['records']}
 ck(root+' eight questions',len(by)==8)
 for row in P['records']:
  c=row['candidate'];rr=by[c['candidate_id']];rev={x['request_id']:x for x in rr['responses']};ck(c['candidate_id']+' two repeats',len(rev)==2)
  joined=[]
  for x in row['responses']:
   r=rev[x['request_id']];st=x['score']['strict'];ck(x['request_id']+' raw binding',sha(Path(x['raw_binding']['path']))==x['raw_binding']['sha256'])
   l1labels=set()
   for i,ev in enumerate(r['evidence']):
    ck(x['request_id']+f' evidence{i} raw',bool(ev['raw_quote']) and ev['raw_quote'] in x['raw_output'])
    ck(x['request_id']+f' evidence{i} mapping',[x['presentation']['canonical_to_display'][v] for v in ev['canonical_labels']]==ev['display_labels'])
    for sc in ev['source_checks']:ck(x['request_id']+f' evidence{i} source '+sc['turn_ref'],bool(sc['source_quote']) and sc['source_quote'] in T[sc['turn_ref']])
    if ev['type']=='L1':l1labels.update(ev['canonical_labels'])
   ck(x['request_id']+' L1 evidence',bool(l1labels)==r['l1_local_correct_reason_wrong_decision'])
   gold=set(c['gold']);pred=set(st['prediction'] or [])
   affected=[lab for lab in sorted(l1labels) if (lab in gold)!=(lab in pred)]
   if l1labels:ck(x['request_id']+' L1 is MC',c['question_type']=='multi_select')
   z=dict(root_id=root,candidate_id=c['candidate_id'],slot_id=c['slot_id'],request_id=x['request_id'],exact=st['exact'],valid=st['valid'],review=r,l1_labels=sorted(l1labels),l1_final_error_labels=affected,l1_affects_final_item=bool(affected),raw_binding=x['raw_binding'])
   joined.append(z);response_rows.append(z)
  allrows.append(dict(candidate_id=c['candidate_id'],root_id=root,slot_id=c['slot_id'],responses=joined))
ck('all F covered',len(response_rows)==64 and len(allrows)==32 and len({x['request_id'] for x in response_rows})==64)
def flag_summary(flag):
 selected=[r for r in response_rows if r['review'][flag]]
 return dict(responses=len(selected),semantic_questions=len({r['candidate_id'] for r in selected}),request_ids=[r['request_id'] for r in selected])
affected=[r for r in response_rows if r['l1_affects_final_item']]
local=[r for r in response_rows if r['review']['l1_local_correct_reason_wrong_decision']]
mc=[r for r in response_rows if r['slot_id'].startswith('MC') and r['slot_id']!='MC6']
screen=read(A/'MECHANICAL_SCREEN.json');mechanical={}
for arm in ['F','G']:
 m=[x for x in screen if x['arm']==arm and x['slot_id'].startswith('MC')]
 mechanical[arm]=dict(mc_responses=len(m),any_judgment_box_disagreement=sum(bool(x['judgment_final_mismatch_labels']) for x in m),gold_correct_judgment_bad_box=sum(bool(x['gold_correct_judgment_bad_final_labels']) for x in m),all_six_judgments_gold_correct_but_bad_final=sum(x['all_six_judgments_gold_correct'] and not x['exact'] for x in m))
summary=dict(schema_version='f_reason_selection_summary_v1',new_solver_calls=0,training=False,scope='F 4 Train roots, 32 semantic questions, 64 dependent display responses; FULL B1 THINK_OFF',overall=dict(responses=64,semantic_questions=32,strict_correct=sum(r['exact'] for r in response_rows)),mc_gold2_to5=dict(responses=len(mc),correct=sum(r['exact'] for r in mc),wrong=sum(not r['exact'] for r in mc)),L1_local_correct_reason_wrong_decision=dict(**flag_summary('l1_local_correct_reason_wrong_decision'),unique_local_items=sum(len(r['l1_labels']) for r in local)),L1_same_item_wrong_in_final=dict(responses=len(affected),semantic_questions=len({r['candidate_id'] for r in affected}),unique_local_items=sum(len(r['l1_final_error_labels']) for r in affected),share_of_noncontrol_MC_responses=len(affected)/len(mc),share_of_noncontrol_MC_wrong=len(affected)/sum(not r['exact'] for r in mc),request_ids=[r['request_id'] for r in affected]),L1_local_judgment_conflict_but_same_item_box_correct=dict(responses=len(local)-len(affected),request_ids=[r['request_id'] for r in local if not r['l1_affects_final_item']]),L2_correct_judgment_wrong_box=flag_summary('l2_correct_judgment_wrong_box'),L3_whole_visible_solution_correct_wrong_box=flag_summary('l3_whole_visible_solution_correct_wrong_box'),all_six_time_values_correct_but_order_wrong=flag_summary('retrieval_correct_but_order_wrong'),answer_correct_contract_failure=flag_summary('answer_correct_contract_failure'),uncertain=flag_summary('uncertain'),mechanical_F_G=mechanical,by_root={root:dict(responses=sum(r['root_id']==root for r in response_rows),L1=sum(r['root_id']==root and r['review']['l1_local_correct_reason_wrong_decision'] for r in response_rows),L1_affects_final=sum(r['root_id']==root and r['l1_affects_final_item'] for r in response_rows)) for root in ['218519','222937','225575','229686']},limitations=['局部正确理由不等于整题推理正确，更不等于隐藏思考正确。','两排列是重复测量，9次final冲突对应7道语义题。','L2机械筛仅检验judgment与box/Gold，语义完整性由F逐条审核补充。','F/G L1没有同口径完整比较，不能根据旧G taxonomy计数直接比较发生率。','事后非盲分析；不改变正式分数，不证明训练或反馈因果收益。'],original_experiment_manifest_sha256=sha(E/'FINAL_MANIFEST.json'))
ck('main claim 9 responses',len(affected)==9)
ck('main claim 7 questions',len({r['candidate_id'] for r in affected})==7)
ck('no L2/L3',summary['L2_correct_judgment_wrong_box']['responses']==summary['L3_whole_visible_solution_correct_wrong_box']['responses']==0)
(A/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
(A/'RESPONSE_INDEX.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in response_rows))
(A/'VALIDATION.json').write_text(json.dumps(dict(status='PASS',check_count=len(checks),checks=checks),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
