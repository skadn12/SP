"""Append closure only after independent audit passes; verify frozen ancestry."""
from pathlib import Path
import datetime, hashlib, json
R=Path(__file__).resolve().parent.parent
A=R/'execution/independent_audit'
CACHE={}
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    p=Path(p).resolve()
    if p not in CACHE:
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(1048576),b''):h.update(b)
        CACHE[p]=h.hexdigest()
    return CACHE[p]
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def write(p,v):
    with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def main():
    audit_path=A/'INDEPENDENT_RESULT_AUDIT.json'
    audit=read(audit_path);summary=read(R/'RESULT_SUMMARY.json')
    assert audit['status']=='PASS' and not audit['failures']
    assert audit['result_summary_sha256']==sha(R/'RESULT_SUMMARY.json')
    assert summary['training_started'] is False and summary['automatic_followup'] is False
    ledgers={};verified={};rows=0;pending=[R/'RESULT_BINDINGS.json']
    while pending:
        p=pending.pop().resolve()
        if str(p) in ledgers:continue
        ledgers[str(p)]=sha(p)
        for r in read(p)['files']:
            q=Path(r['path']);q=q if q.is_absolute() else p.parent/q
            assert sha(q)==r['sha256'],str(q)
            verified[str(q.resolve())]=sha(q);rows+=1
            if q.name=='RESULT_BINDINGS.json':pending.append(q)
    additions={Path(__file__).resolve()}|{p for p in A.rglob('*') if p.is_file() and p.suffix in {'.json','.md','.py'} and str(p.resolve()) not in verified}
    receipt={'version':'v72_append_only_audit_receipt_v1','status':'PASS','completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'original_result_bindings':bind(R/'RESULT_BINDINGS.json'),'audit':bind(audit_path),
        'recursive_result_binding_ledgers':ledgers,'recursive_binding_rows_verified':rows,'recursive_unique_bound_files_verified':len(verified),
        'append_only_files':[bind(p) for p in sorted(additions)],'new_model_calls':0,'frozen_files_modified':0}
    rp=A/'APPEND_ONLY_AUDIT_RECEIPT.json';write(rp,receipt)
    result={'version':'v72_trial_complete_v1','status':'COMPLETE','completed_at_utc':receipt['completed_at_utc'],
        'generation_http_attempts':summary['generation_cost']['http_attempts'],'assembled':summary['assembled_count'],
        'content_usable':summary['content_usable_count'],'selected':summary['selected_count'],'added':summary['added_count'],
        'merged_candidate_pool':summary['merged_count'],'calls_per_model':summary['calls_per_model'],
        'independent_checks':audit['checks_completed'],'independent_differences':0,'result_summary':bind(R/'RESULT_SUMMARY.json'),
        'original_result_bindings':bind(R/'RESULT_BINDINGS.json'),'append_only_receipt':bind(rp),
        'training_started':False,'automatic_followup':False,'required_count_coverage':summary['required_count_coverage']}
    write(R/'execution/TRIAL.COMPLETE.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
