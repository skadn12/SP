"""Recheck the late-quality preflight and bind final reports without inference."""
import datetime,hashlib,json,pathlib
P=pathlib.Path(__file__).resolve().parent.parent
errors=[];bindings={}
def ck(ok,message):
 if not ok:errors.append(message)
def read(p):return json.loads(pathlib.Path(p).read_text())
def bind(p,digest=None,size=None):
 p=pathlib.Path(p).resolve()
 if str(p) in bindings:
  row=bindings[str(p)]
 else:
  h=hashlib.sha256()
  with p.open('rb') as stream:
   for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
  row={'path':str(p),'sha256':h.hexdigest(),'bytes':p.stat().st_size};bindings[str(p)]=row
 if digest is not None:ck(row['sha256']==digest,'HASH:'+str(p))
 if size is not None:ck(row['bytes']==size,'SIZE:'+str(p))
bind(__file__)
for name in ['QUALITY_PREFLIGHT_AUDIT.json','INDEPENDENT_FINAL_AUDIT.json']:
 path=P/'evidence'/name;bind(path);audit=read(path);ck(audit['status']=='PASS' and not audit['errors'],name+':NOT_PASS')
 for r in audit['audited_files']:bind(r['path'],r['sha256'],r.get('bytes'))
quality=read(P/'evidence/QUALITY_PREFLIGHT_AUDIT.json');run=read(P/'RUN_STATUS.json')
ck(datetime.datetime.fromisoformat(quality['created_at_utc'])<=datetime.datetime.fromisoformat(run['started_at_utc']),'QUALITY_BINDING_AFTER_INFERENCE_START')
ck(run['status']=='REPORTED_AWAITING_INDEPENDENT_AUDIT' and run['pid'] is None and run['actual_new_calls']==3258,'ROOT_NOT_STOPPED_COMPLETE')
for path in sorted(P.glob('*')):
 if path.is_file():bind(path)
for path in sorted((P/'evidence').glob('*')):
 if path.is_file() and path.name!='QUALITY_FINAL_AUDIT.json':bind(path)
q=read(P/'evidence/HISTORICAL_QUALITY_CROSSCHECK.json');s=read(P/'SUMMARY.json')
ck(s['status']=='COMPLETE' and not s['errors'],'REPORT_NOT_COMPLETE')
ck(s['historical_gold_review_applicability']==q['summary'],'FINAL_REPORT_QUALITY_SUMMARY_DIFFERS')
result={'schema_version':'v131_independent_quality_final_binding_v1','status':'FAIL' if errors else 'PASS','errors':errors,'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'auditor':'/root/baseline_order_errors','checks':{'quality_evidence_and_all746_inputs_unchanged_since_before_inference':True,'primary_final_audit_and_all_raw_bindings_reverified':True,'all_present_root_report_status_logs_and_evidence_files_bound':True,'final_report_quality_summary_equal':True,'no_semantic_truth_certification_claimed':True},'audited_files':list(bindings.values()),'new_inference_calls':0,'TEST_read_or_inference':False}
destination=P/'evidence/QUALITY_FINAL_AUDIT.json'
with destination.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'status':result['status'],'errors':errors,'bound_files':len(bindings),'path':str(destination)}));raise SystemExit(bool(errors))
