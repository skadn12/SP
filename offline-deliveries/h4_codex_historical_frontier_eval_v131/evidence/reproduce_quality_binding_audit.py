"""Additive v131 preflight binding for late historical applicability evidence."""
import collections,datetime,hashlib,json,pathlib
P=pathlib.Path(__file__).resolve().parent.parent
Q=P/'evidence/HISTORICAL_QUALITY_CROSSCHECK.json'
expected='5773f6deef5e8187ef68517a6d144eaf4e11d002d646cb319ee1b391e851fdb9'
errors=[];bindings={}
def bind(p,digest=None,size=None):
 p=pathlib.Path(p).resolve();raw=p.read_bytes();actual=hashlib.sha256(raw).hexdigest();bindings[str(p)]={'path':str(p),'sha256':actual,'bytes':len(raw)}
 if digest is not None and actual!=digest:errors.append('HASH:'+str(p))
 if size is not None and len(raw)!=size:errors.append('SIZE:'+str(p))
def ck(ok,message):
 if not ok:errors.append(message)
bind(__file__);bind(Q,expected);q=json.loads(Q.read_text())
for r in q['audited_files']:bind(r['path'],r['sha256'],r.get('bytes'))
bind(P/'manifest.json');manifest=json.loads((P/'manifest.json').read_text());rows=q['rows'];selected={r['candidate_id']:r for r in manifest['candidates']}
ck(q['status']=='READONLY_CROSSCHECK_COMPLETE','QUALITY_LOOKUP_INCOMPLETE')
ck(len(rows)==123 and {r['candidate_id'] for r in rows}==set(selected),'EXACT_123_CANDIDATE_SET')
ck(collections.Counter(r['quality_state'] for r in rows)=={'UNKNOWN':121,'HISTORICAL_ACCEPT_ONLY':2},'QUALITY_STATE_COUNTS')
ck(q['summary']=={'selected_questions':123,'selected_roots':58,'question_type_counts':{'multi_select':51,'ordering':62,'single_choice':10},'known_bound_nonvalid_gold_count':0,'historical_same_content_accept_count':2,'no_applicable_semantic_review_found_count':121,'unknown_is_not_valid':True,'all123_retained_for_evaluation':True,'gold_modified':False,'new_llm_audit_calls':0},'QUALITY_SUMMARY')
for r in rows:
 cid=r['candidate_id'];s=selected[cid];ck(all(r[k]==s[k] for k in ['root_id','question_type']),'ROW_IDENTITY:'+cid);ck(r['first_round_training_exposure']=='UNEXPOSED_TO_BOTH' and r['gold_status']=='unknown_not_newly_adjudicated','QUALITY_SCOPE:'+cid);ck(not r['bound_historical_nonvalid_gold_labels'],'UNEXPECTED_BOUND_INVALID:'+cid)
 for review in r['applicable_historical_reviews']:
  bind(review['review_path'],review['review_file_sha256']);ck(review['semantic_status']=='accept' and all(review['field_equality'].values()) and all(review['source_equality'].values()),'HISTORICAL_REVIEW_APPLICABILITY:'+cid)
bind(P/'evidence/INDEPENDENT_PREFLIGHT_AUDIT.json');prior=json.loads((P/'evidence/INDEPENDENT_PREFLIGHT_AUDIT.json').read_text());ck(prior['status']=='PASS' and not prior['errors'],'PRIMARY_PREFLIGHT_NOT_PASS')
ck(not (P/'RUN_STATUS.json').exists() and not (P/'responses').exists(),'QUALITY_BINDING_AFTER_INFERENCE_START')
result={'schema_version':'v131_independent_quality_binding_preflight_v1','status':'FAIL' if errors else 'PASS','errors':errors,'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'auditor':'/root/baseline_order_errors','scope':'Artifact integrity and selected-population consistency only; existing model accept is not a fresh Gold truth certification. No inference or TEST content.','quality_lookup_sha256':expected,'checks':{'questions':123,'unknown':121,'historical_accept_only':2,'all123_retained':True,'no_new_semantic_judgment':True,'all_quality_input_hashes_verified':True},'audited_files':list(bindings.values()),'new_inference_calls':0}
destination=P/'evidence/QUALITY_PREFLIGHT_AUDIT.json'
with destination.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'status':result['status'],'errors':errors,'bound_files':len(bindings),'path':str(destination)}));raise SystemExit(bool(errors))
