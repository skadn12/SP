#!/usr/bin/env python3
"""CPU-only settlement of a terminal one-shot run; never launches inference."""
import collections
import datetime
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_new(name, value):
    with (HERE/name).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')

def summary(rows):
    returned = [r for r in rows if r['execution_status']=='RETURNED']
    n = len(returned)
    return {'planned':len(rows), 'returned':n, 'missing':len(rows)-n,
            'valid':sum(bool(r['valid']) for r in returned),
            'invalid_executed':sum(not r['valid'] for r in returned),
            'exact_correct':sum(bool(r['exact_correct']) for r in returned),
            'exact_executed':sum(bool(r['exact_correct']) for r in returned)/n if n else None,
            'mean_J_executed':sum(r.get('J',0) for r in returned)/n if n else None,
            'mean_J_complete_plan':sum(r.get('J',0) for r in returned)/n if n==len(rows) and n else None,
            'protocol_valid_executed':sum(bool(r['protocol_valid']) for r in returned)/n if n else None}

def main():
    procs=subprocess.check_output(['ps','-eo','args'],text=True).splitlines()
    assert not any('run_formal.py run' in p and p.lstrip().startswith('python') for p in procs), 'RUN_STILL_LIVE'
    assert (HERE/'STARTED.json').exists() and not (HERE/'COMPLETED.json').exists()
    spec=importlib.util.spec_from_file_location('formal', HERE/'run_formal.py')
    runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    mod=runner.load_adapter()
    requests=read(HERE/'REQUESTS.json'); rows=[]; raws=[]; raw_bindings=[]
    known={r['request_id'] for r in requests}
    assert len(known)==len(requests)==4608
    assert all(p.stem in known for p in (HERE/'raw').glob('*.json'))
    for r in requests:
        for kind in ('prompt','schema'):
            assert sha(HERE/r[kind+'_path'])==r[kind+'_sha256'], 'ASSET_CHANGED'
        path=HERE/'raw'/(r['request_id']+'.json')
        if path.exists():
            raw=read(path); assert raw['request_id']==r['request_id']
            assert raw['prompt_sha256']==r['prompt_sha256']
            raw_bindings.append({'path':str(path), 'sha256':sha(path)})
        else:
            raw={'request_id':r['request_id'], 'status':'MISSING',
                 'reason':'terminal process absent; no persisted response; infrastructure missing, not model error',
                 'automatic_retry':False}
        checked=mod.validate(raw,r,schema=read(HERE/r['schema_path']))
        # The historical invalid() helper passes an empty label domain and
        # divides by zero.  Use its same official scorer with the real frozen
        # request, so invalid responses receive zero without changing metrics.
        ans=checked.get('answer_only_score') or mod._old.h.baseline.parse_and_score({'status':'INVALID'},r)
        rows.append({**{k:r[k] for k in ('request_id','candidate_id','root_id','arm','question_type','context_mode','variant_id','repeat_index')},
                     **ans, **checked, 'execution_status':raw['status'],
                     'raw_prediction':raw.get('raw_prediction'), 'finish_reason':raw.get('finish_reason')})
        raws.append(raw)
    by_item={cid:{m:summary([r for r in rows if r['candidate_id']==cid and r['context_mode']==m]) for m in ('full_context','no_context')}
             for cid in sorted({r['candidate_id'] for r in requests})}
    report={'status':'INTERRUPTED_SCORED', 'scored_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'primary_scoring':'frozen final-answer-only scorer; invalid executed answers score zero',
            'missing_policy':'excluded from executed denominators; incomplete Full-context candidates ineligible',
            'protocol_is_training_gate':False, 'automatic_retry':False,
            'request_count':len(rows), 'summary':{m:summary([r for r in rows if r['context_mode']==m]) for m in ('full_context','no_context')},
            'by_type':{q:{m:summary([r for r in rows if r['question_type']==q and r['context_mode']==m]) for m in ('full_context','no_context')} for q in ('multi_select','ordering')},
            'by_arm':{a:{m:summary([r for r in rows if r['arm']==a and r['context_mode']==m]) for m in ('full_context','no_context')} for a in ('F','G')},
            'by_item':by_item,'requests':rows, 'request_sha256':sha(HERE/'REQUESTS.json'),
            'runner_sha256':sha(HERE/'run_formal.py'), 'scoring_adapter_sha256':sha(runner.ADAPTER_FILE),
            'raw_bindings':raw_bindings,
            'limits':['Only returned request prefix observed; no unbiased F/G or batch-wide accuracy claim.',
                      'All 642 unresolved historical lineage records remain UNKNOWN/PENDING.']}
    write_new('INTERRUPTED.json',{'status':'INTERRUPTED','process_confirmed_absent':True,'original_session_handle':'62349 (unknown process id on poll)',
              'returned':len(raw_bindings),'planned':len(rows),'missing':len(rows)-len(raw_bindings),'cause':'process/session disappeared; exact termination cause not established','automatic_retry':False})
    write_new('RAW_ANSWERS.json',raws)
    write_new('SCORES.json',report)
    print(json.dumps({'status':report['status'],'summary':report['summary'],'complete_full_candidates':sum(v['full_context']['missing']==0 for v in by_item.values()),
                      'solver_value_candidates':[k for k,v in by_item.items() if v['full_context']['missing']==0 and 0<v['full_context']['mean_J_complete_plan']<1]},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
