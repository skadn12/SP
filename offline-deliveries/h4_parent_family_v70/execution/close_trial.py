"""Append final receipts after independent audit; never alter frozen outputs."""
from pathlib import Path
import json, hashlib, datetime
R=Path(__file__).resolve().parent.parent
E=R/'execution'; A=E/'independent_audit'
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def bind(p): return {'path':str(p),'sha256':sha(p)}
def write(p,v):
    with p.open('x') as f: json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
summary=read(R/'RESULT_SUMMARY.json');audit=read(A/'INDEPENDENT_RESULT_AUDIT.json')
assert audit['status']=='PASS' and not audit['failures']
assert audit['result_summary_sha256']==sha(R/'RESULT_SUMMARY.json')
assert summary['training_started'] is False and summary['automatic_followup'] is False
ledgers={}; verified={}; binding_rows=0
pending=[R/'RESULT_BINDINGS.json']
while pending:
    p=pending.pop()
    if str(p) in ledgers: continue
    ledgers[str(p)]=sha(p)
    for row in read(p)['files']:
        q=Path(row['path']); digest=sha(q)
        assert digest==row['sha256'], str(q)
        if str(q) in verified: assert verified[str(q)]==digest
        verified[str(q)]=digest;binding_rows+=1
        if q.name=='RESULT_BINDINGS.json': pending.append(q)
receipt={'version':'v70_append_only_final_audit_receipt_v1','completed_at_utc':now(),'status':'PASS',
 'original_result_bindings':bind(R/'RESULT_BINDINGS.json'),
 'existing_result_bound_files_verified_unchanged':len(read(R/'RESULT_BINDINGS.json')['files']),
 'recursive_result_binding_ledgers':ledgers,'recursive_binding_rows_verified':binding_rows,
 'recursive_unique_bound_files_verified':len(verified),
 'append_only_files':[bind(p) for p in [R/'DECISION_RESULT.md',A/'INDEPENDENT_RESULT_AUDIT.json',A/'INDEPENDENT_RESULT_AUDIT.md']],
 'new_model_calls':0,'frozen_files_modified':0,'training_started':False}
write(A/'APPEND_ONLY_AUDIT_RECEIPT.json',receipt)
complete={'version':'v70_trial_complete_v1','completed_at_utc':now(),'status':'COMPLETE',
 'planned_families':summary['family_count'],'generation_http_attempts':summary['generation_cost']['http_attempts'],
 'assembled':summary['assembled_count'],'content_usable':summary['content_usable_count'],
 'selected':summary['selected_count'],'added':summary['added_count'],'merged_candidate_pool':summary['merged_count'],
 'calls_per_model':summary['calls_per_model'],'raw_responses':2*summary['calls_per_model'],
 'independent_checks':audit['checks_completed'],'independent_differences':len(audit['failures']),
 'result_summary':bind(R/'RESULT_SUMMARY.json'),'original_result_bindings':bind(R/'RESULT_BINDINGS.json'),
 'append_only_receipt':bind(A/'APPEND_ONLY_AUDIT_RECEIPT.json'),
 'training_allowed':False,'training_started':False,'automatic_followup':False,
 'required_count_coverage':summary['required_count_coverage'],
 'remaining_merged_count_gaps':summary['distributions']['merged']['missing_multiselect_gold_counts']}
write(E/'TRIAL.COMPLETE.json',complete)
print(json.dumps({'status':'COMPLETE','binding_ledgers':len(ledgers),'binding_rows':binding_rows,'unique_files':len(verified),'selected':summary['selected_count']}))
