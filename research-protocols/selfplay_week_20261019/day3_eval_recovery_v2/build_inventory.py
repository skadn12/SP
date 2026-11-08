import collections, hashlib, json
from pathlib import Path
from datetime import datetime,timezone
W=Path('/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019')
O=W/'day3_eval_recovery_v2'
B=Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries')
V124=B/'h4_codex_dev_test_ab_v124';V134=B/'h4_codex_held_roots_eval_v134'
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def canon(q):return hashlib.sha256(json.dumps(q,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(n,d):
 p=O/n
 if p.exists():raise RuntimeError('Refusing overwrite '+str(p))
 p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def count(qs):
 roots=sorted({q['root_id'] for q in qs})
 return dict(questions=len(qs),roots=len(roots),root_ids=roots,by_type=dict(collections.Counter(q['question_type'] for q in qs)),mc_gold_count=dict(sorted(collections.Counter(str(len(q['gold'])) for q in qs if q['question_type']=='multi_select').items())),primary_gold2to5_mc=sum(q['question_type']=='multi_select' and 2<=len(q['gold'])<=5 for q in qs),ordering=sum(q['question_type']=='ordering' for q in qs),gold6_controls=sum(q['question_type']=='multi_select' and len(q['gold'])==6 for q in qs))
roots={r['root_id']:r for r in load(W/'day1/evaluation/ROOT_BOUNDARY.json')['roots']}
train=set(load(W/'day1/TRAIN_ROOTS.json')['root_ids'])
stock={};bindings=[]
for name,p in [('v124_dev',V124/'delivery/dev/QUESTIONS_WITH_ANSWERS.json'),('v124_test',V124/'delivery/test/QUESTIONS_WITH_ANSWERS.json'),('v134_increment',V134/'EVALUATION_POOL.json'),('extended_dev',V134/'delivery/dev/EXTENDED_QUESTIONS_WITH_ANSWERS.json'),('extended_test',V134/'delivery/test/EXTENDED_QUESTIONS_WITH_ANSWERS.json')]:
 stock[name]=load(p)['questions'];bindings.append(bind(p))
extended=stock['extended_dev']+stock['extended_test']
original=stock['v124_dev']+stock['v124_test']+stock['v134_increment']
oi={q['candidate_id']:q for q in original};ei={q['candidate_id']:q for q in extended}
exact=len(oi)==len(original)==len(extended)==len(ei) and set(oi)==set(ei) and all(canon(q)==canon(oi[i]) for i,q in ei.items())
qa=[]
for p in [V124/'MASTER_FREEZE.json',V134/'QA_FREEZE.json']:
 d=load(p);bindings.append(bind(p));bad=[]
 for item in d['files']:
  t=Path(item['path'])
  if not t.is_file() or sha(t)!=item['sha256']:bad.append(dict(path=str(t),expected=item['sha256'],actual=sha(t) if t.is_file() else None))
 qa.append(dict(manifest=str(p),checked=len(d['files']),failures=bad))
incids={q['candidate_id'] for q in stock['v134_increment']};rows=[]
for q in extended:
 r=roots.get(q['root_id'],{});p=Path(q['source_path']);actual=sha(p) if p.is_file() else None
 rows.append(dict(candidate_id=q['candidate_id'],root_id=q['root_id'],split=q['split'],question_type=q['question_type'],gold_count=len(q['gold']) if q['question_type']=='multi_select' else None,role=q.get('role'),origin='v134_increment' if q['candidate_id'] in incids else 'v124',record_sha256=canon(q),source_path=str(p),source_sha256_expected=q.get('source_sha256'),source_sha256_actual=actual,source_hash_match=actual is not None and actual==q.get('source_sha256'),original_split_match=r.get('original_split')==q['split'],root_in_train70=q['root_id'] in train,historical_training_diagnostic_rows=r.get('historical_train_diagnostic_row_count'),v125_training_used=r.get('v125_training_used'),historical_role=r.get('current_role'),historically_exposed=True,intended_reporting='GOLD6_CONTROL_ONLY' if q['question_type']=='multi_select' and len(q['gold'])==6 else 'HISTORICAL_REGRESSION_CANDIDATE',current_fact_family_leakage_check='PENDING_NOT_REPLACED_BY_ROOT_DISJOINTNESS',current_semantic_dedup='PENDING_SHARED_FACTS_VARIANTS_NOT_INDEPENDENT',current_request_freeze='PENDING',current_admission='NOT_YET_ADMITTED'))
sealed=W/'day3_eval_clearance_v1';m=load(sealed/'FINAL_MANIFEST.json');bad=[]
for item in m['files']:
 p=Path(item['path']);p=p if p.is_absolute() else sealed/p
 if not p.is_file() or sha(p)!=item['sha256']:bad.append(str(p))
s=dict(schema_version='historical_regression_inventory_v2',created_at_utc=datetime.now(timezone.utc).isoformat(),status='MECHANICAL_INVENTORY_COMPLETE_CURRENT_CONTENT_BOUNDARY_ADMISSION_PENDING',selection_basis='All exact records in pre-existing extended DEV/TEST libraries; no scores or model answers used.',counts=count(extended),by_split={k:count([q for q in extended if q['split']==k]) for k in ['dev','test']},stock_counts={k:count(v) for k,v in stock.items()},exact_extended_equals_v124_135_plus_v134_49=exact,distinct_candidate_ids=len(ei),source_hash_failures=[r['candidate_id'] for r in rows if not r['source_hash_match']],split_mismatches=[r['candidate_id'] for r in rows if not r['original_split_match']],root_overlap_train70=sorted({r['root_id'] for r in rows if r['root_in_train70']}),historical_diagnostic_root_rows=sum(r['historical_training_diagnostic_rows'] for r in {r['root_id']:r for r in rows}.values()),qa_manifest_checks=qa,qa_bindings_pass_does_not_replace_new_semantic_or_leakage_admission=True,sealed_v1_files_checked=len(m['files']),sealed_v1_failures=bad,newly_admitted_regression_questions=0,new_family_supplement_preserved=dict(questions=4,multi_select=4,ordering=0,roots=3,requests=72),training_candidates_unchanged=187,new_solver_calls=0,new_training_runs=0,training_start_go=False,pending=['Score-blind current training/replay/base/diagnosis/feedback fact-family leakage check.','Semantic related-family grouping; Gold variants/permutations are not independent.','Reuse exact source/QA scopes and retain historical conflicts disclosure.','Freeze common requests, mapping, zero-truncation and model identities before evaluation.'],input_bindings=bindings)
assert exact
assert not s['source_hash_failures']
assert not s['split_mismatches']
assert not s['root_overlap_train70']
assert not bad
write('HISTORICAL_CANDIDATE_METADATA.json',dict(schema_version='historical_regression_candidate_metadata_v2',questions=rows))
write('INVENTORY_SUMMARY.json',s)
print(json.dumps({k:s[k] for k in ['status','counts','exact_extended_equals_v124_135_plus_v134_49','qa_manifest_checks','sealed_v1_files_checked','sealed_v1_failures','newly_admitted_regression_questions']},ensure_ascii=False))
