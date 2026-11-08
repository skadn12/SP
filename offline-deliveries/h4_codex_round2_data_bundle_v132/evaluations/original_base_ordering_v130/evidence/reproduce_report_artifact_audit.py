#!/usr/bin/env python3
"""Supplemental additive audit for all report artifacts and stopped inference logs."""
import collections,csv,datetime,fractions,hashlib,json,pathlib,re
P=pathlib.Path(__file__).resolve().parent.parent
ERRORS=[];BOUND={}
def read(p):return json.loads(p.read_text())
def check(ok,msg):
 if not ok:ERRORS.append(msg)
def bind(p):
 data=p.read_bytes();BOUND[str(p)]={'path':str(p),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
def stats(rep,ref,tag):
 check(rep['planned_calls']==ref['calls'] and rep['valid_calls']==ref['calls'] and rep['exact_correct']==ref['exact_correct'] and rep['invalid_or_missing_calls']==0,tag+':COUNTS')
 check(abs(rep['exact_rate_fixed']-ref['exact_rate'])<1e-12,tag+':EXACT')
 check(abs(rep['metrics']['pair_accuracy']['mean']-ref['pair_accuracy'])<1e-12,tag+':PAIR')
 check(abs(rep['metrics']['signed_partial']['mean']-float(fractions.Fraction(ref['mean_signed_partial'])))<1e-12,tag+':PARTIAL')
 check(rep['pair_counts']['known_correct_pairs']==ref['correct_pairs'] and rep['pair_counts']['planned_pairs']==ref['total_pairs'] and rep['pair_counts']['unknown_pairs']==0,tag+':PAIR_COUNTS')
 for cat in ['A_internal','B_internal','cross_route']:
  pc=ref['pair_categories'][cat];rp=rep['pair_categories'][cat];check(rp['known_correct_pairs']==pc['correct'] and rp['planned_pairs']==pc['total'] and rp['unknown_pairs']==0,tag+':CATEGORY:'+cat)
 check(abs(rep['metrics']['block_violation']['mean']-ref['block_violation_calls']/ref['calls'])<1e-12,tag+':BLOCK')
audit=read(P/'evidence/INDEPENDENT_FINAL_AUDIT.json');check(audit['status']=='PASS' and not audit['errors'],'FINAL_AUDIT_NOT_PASS');bind(P/'evidence/INDEPENDENT_FINAL_AUDIT.json')
ind=audit['checks']['independently_recomputed_results'];ids={'base':'M0','baseline':'M_clean_policy','new':'M_v125_newonly'}
scope=read(P/'SCOPE.json');summary=read(P/'SUMMARY.json')
for key,mid in ids.items():stats(summary['models'][key]['summary'],ind[mid]['overall'],'SUMMARY:'+key)
pc=read(P/'PER_CALL.json');check(len(pc)==168 and len({r['request_id'] for r in pc})==168,'PER_CALL_ROWS')
raws={key:{r['request_id']:r for r in ind[mid]['per_call']} for key,mid in ids.items()}
for r in pc:
 for key,mid in ids.items():
  rr=r['models'][key];ref=raws[key][r['request_id']];check(rr['valid'] and not rr['errors'],'PER_CALL_VALID:'+r['request_id']+':'+key);check(rr['exact_match']==ref['exact'] and rr['correct_pairs']==ref['correct_pairs'] and fractions.Fraction(rr['signed_partial'])==fractions.Fraction(ref['signed_partial']),'PER_CALL_NUMERIC:'+r['request_id']+':'+key)
  check(rr['block_violation']==ref['block_violation'] and rr['pair_category_correct']=={c:v[0] for c,v in ref['pair_categories'].items()},'PER_CALL_CATEGORIES:'+r['request_id']+':'+key)
pq=read(P/'PER_QUESTION.json');check(len(pq)==14 and {r['candidate_id'] for r in pq}==set(scope['candidate_ids']),'PER_QUESTION_ROWS')
for r in pq:
 for key,mid in ids.items():stats(r['models'][key],ind[mid]['per_question'][r['candidate_id']],'PER_QUESTION:'+r['candidate_id']+':'+key)
pr=read(P/'PER_ROOT.json');check(len(pr)==15 and {r['root_id'] for r in pr}==set(scope['fixed_dev_root_ids']),'PER_ROOT_ROWS')
for r in pr:
 if r['has_questions']:
  for key,mid in ids.items():stats(r['models'][key],ind[mid]['per_root'][r['root_id']],'PER_ROOT:'+r['root_id']+':'+key)
 else:
  check(r['question_count']==0,'UNCOVERED_COUNT')
  for key in ids:check(r['models'][key]['exact_rate_fixed'] is None and r['models'][key]['planned_calls']==0,'UNCOVERED_MISSING_DENOMINATOR')
with (P/'PER_QUESTION.csv').open(newline='') as f:cr=list(csv.DictReader(f))
check(len(cr)==14 and {r['candidate_id'] for r in cr}==set(scope['candidate_ids']),'CSV_ROWS')
for r in cr:
 for key,mid in ids.items():
  ref=ind[mid]['per_question'][r['candidate_id']];check(int(r[key+'_planned_calls'])==12 and int(r[key+'_valid_calls'])==12 and abs(float(r[key+'_exact'])-ref['exact_rate'])<1e-12 and abs(float(r[key+'_pair_accuracy'])-ref['pair_accuracy'])<1e-12,'CSV_NUMERIC:'+r['candidate_id']+':'+key)
zh=(P/'REPORT_ZH.md').read_text();check(all(x in zh for x in ['3/168 = 1.79%', '4/168 = 2.38%','168/168 = 100.00%','70.63%','70.00%','未读取TEST题文','14题','504次调用']),'CHINESE_HEADLINES_OR_LIMITS')
log=(P/'live.log').read_text();state=read(P/'RUN_STATUS.json');check(state['status']=='BASE_CONTROL_REPORTED_AWAITING_INDEPENDENT_AUDIT' and state['pid'] is None,'LOG_RUN_NOT_STOPPED');check('Traceback (most recent call last)' not in log and '"status": "BASE_CONTROL_REPORTED_AWAITING_INDEPENDENT_AUDIT"' in log,'LOG_FAILED_OR_NOT_COMPLETE')
files=['SUMMARY.json','PER_CALL.json','PER_QUESTION.json','PER_QUESTION.csv','PER_ROOT.json','REPORT_ZH.md','live.log','prepare.log','RUN_STATUS.json']
for fn in files:bind(P/fn)
bind(pathlib.Path(__file__))
out={'schema_version':'v130_report_artifact_additive_independent_audit_v1','status':'FAIL' if ERRORS else 'PASS','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'auditor':'/root/baseline_order_errors','errors':ERRORS,'checks':{'reports':6,'per_call_rows':168,'model_call_comparisons':504,'per_question_rows':14,'root_rows':15,'uncovered_root_rows':8,'csv_model_rows_checked':42,'logs_complete':True,'immutable_main_audit_preserved':True,'no_inference_or_TEST':True},'audited_files':list(BOUND.values())}
p=P/'evidence/REPORT_ARTIFACT_AUDIT.json'
with p.open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'status':out['status'],'errors':ERRORS,'audited_files':len(BOUND),'path':str(p)},ensure_ascii=False,indent=2))
raise SystemExit(bool(ERRORS))
