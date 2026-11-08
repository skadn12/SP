"""CPU integrity preflight for a separate internal reviewer; no inference/import of runner.

The script is prepared by setup actor. An independent executing task must inspect
it and pass its actual identity. This does not constitute external independence.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

BATCH = Path(__file__).resolve().parent.parent
NATIVE_SHA = {
 'run_solver.py': '50f2b797fe8a6560d3fc792f34acbec40309dbda462744817b93a453785a6462',
 'protocol.py': '2314e7759ab483e75e28275dfff91f1e50f7b5f00cc13f5b904e18864d0d0e19',
 'scoring.py': '2d0e7411dcd7b1c520bdbaecfef5259d48f036ce071a98ac3d6e633be2017ad2'}
PUBLIC = {'candidate_id','root_id','question_type','question','options','items',
          'source_path','source_sha256','split','chronology_kind'}
MODELS = {'clean': ('M_clean_policy', '/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_common_dev_eval_v1_live_3/clean_merged_model'),
          'new': ('M_v125_newonly','/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_codex_combined_sft_v125_merged')}

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main(batch, auditor, output):
    if output.exists(): raise SystemExit('REFUSE_OVERWRITE:'+str(output))
    ev=batch/'evaluation'
    if any((ev/'run'/s/'STARTED.json').exists() for s in MODELS):
        raise SystemExit('PREFLIGHT_TOO_LATE_INFERENCE_ALREADY_STARTED')
    errors=[]; bindings={}; checks=0
    def check(value, code, **detail):
        nonlocal checks
        checks+=1
        if not value: errors.append({'code':code,**detail})
    def read(path):
        path=Path(path).resolve(); bindings[str(path)]=digest(path)
        return json.loads(path.read_text())
    def bound(rows, relative=None):
        for row in rows:
            p=Path(row['path'])
            if not p.is_absolute(): p=relative/p
            check(p.is_file() and digest(p)==row['sha256'],'BINDING_CHANGED',path=str(p))
            if p.is_file():
                bindings[str(p.resolve())]=digest(p)
                if 'bytes' in row: check(p.stat().st_size==row['bytes'],'BOUND_SIZE',path=str(p))
    design=read(batch/'PLAN.json'); validation=read(batch/'evidence/INPUT_VALIDATION.json')
    preflight=read(ev/'PREFLIGHT_FREEZE.json'); tools=read(ev/'TOOLS_FREEZE.json')
    bound(preflight['files']); bound(tools['files'])
    native=batch.parent/'h4_feedback_training_pool_v15_qa/evaluation'
    for name,h in NATIVE_SHA.items(): check(digest(native/name)==h,'NATIVE_SHA',file=name)
    pool=read(ev/'EVALUATION_POOL.json')['questions']; admission=read(batch/'private/ADMITTED_QUESTIONS.json')['questions']
    scope=read(ev/'SCOPE.json'); plan=read(ev/'SHARED_REQUESTS.json'); qs={q['candidate_id']:q for q in pool}
    check(len(qs)==len(pool)==len(admission)==validation['admitted_count'],'ADMITTED_COUNT')
    check(validation['status']=='PASS' and validation['planned_slots']==design['planned_slots'] and
          validation['rejected_count']+len(pool)==design['planned_slots'],'ADMISSION_ACCOUNTING')
    check(len({q['slot_id'] for q in pool})==len(pool),'UNIQUE_SLOT_IDS')
    expected={q['candidate_id']:q for q in admission}
    for q in pool:
        original=expected[q['candidate_id']]
        for k,v in original.items(): check(q.get(k)==v,'ADMITTED_VALUE_CHANGED',candidate=q['candidate_id'],field=k)
        check(q['admission_status']=='PASS' and q['split']=='dev','ADMISSION_AND_SPLIT')
        check(q['pilot_arm'] in {'new_structure','old_structure_control'},'ARM_VALID')
    requests=plan['requests']; by=defaultdict(list)
    check(plan['seed']==scope['seed']==2026100800,'SEED')
    check(len({r['request_id'] for r in requests})==len(requests)==scope['calls_per_model'],'REQUEST_SET')
    check(len(requests)==sum(18 if q['question_type']=='multi_select' else 12 for q in pool),'DYNAMIC_CALL_COUNT')
    for r in requests: by[r['candidate_id']].append(r)
    for cid,q in qs.items():
        v,s=(6,3) if q['question_type']=='multi_select' else (12,1)
        rows=by[cid]
        check({(r['variant']['variant_id'],r['slot']) for r in rows}=={(f'perm_{i:02d}',j) for i in range(v) for j in range(s)} and len(rows)==v*s,'VARIANT_SLOT_SET',candidate=cid)
        field='options' if q['question_type']=='multi_select' else 'items'
        contents={'content_'+hashlib.sha256(t.strip().encode()).hexdigest()[:16] for t in q[field].values()}
        orders=set()
        for row in rows:
            variant=row['variant']; mapping=variant['display_to_content']; shown=variant[field]
            check(set(mapping)==set(shown) and set(mapping.values())==contents,'DISPLAY_BIJECTION')
            check(all(mapping[k]=='content_'+hashlib.sha256(t.strip().encode()).hexdigest()[:16] for k,t in shown.items()),'DISPLAY_CONTENT_HASH')
            orders.add(tuple(mapping.values()))
        check(len(orders)==v,'UNIQUE_PERMUTATIONS',candidate=cid)
    for side,(name,path) in MODELS.items():
        run=ev/'run'/side; manifest=read(run/'manifest.json'); public=read(run/'questions.json')['candidates']; gold=read(run/'gold.json')
        check(manifest['requests']==requests and manifest['decoder_contract']==plan['decoder_contract'],'PAIRED_PLAN',side=side)
        check((manifest['solver_model'],manifest['model'])==(name,path),'FROZEN_MODEL_IDENTITY',side=side)
        identity=read(run/'solver_identity.json'); bound(identity['checkpoint_files'])
        check(identity.get('model_path')==path and identity.get('model_identifier')==name,'MODEL_IDENTITY')
        if side=='new': check(identity.get('adapter_already_merged') is True,'ALREADY_MERGED_NO_EXTRA_ADAPTER')
        check(len(public)==len(pool),'PUBLIC_COUNT')
        for q in public:
            check(set(q)<=PUBLIC,'PRIVATE_FIELD_IN_PUBLIC',candidate=q['candidate_id'])
            original=qs[q['candidate_id']]
            check(q=={k:v for k,v in original.items() if k in PUBLIC},'PUBLIC_CONTENT')
        for cid,q in qs.items():
            field='options' if q['question_type']=='multi_select' else 'items'
            want=['content_'+hashlib.sha256(q[field][label].strip().encode()).hexdigest()[:16] for label in q['gold']]
            check(gold[cid]==want,'GOLD_CONTENT_BINDING',candidate=cid)
        for i,r in enumerate(requests):
            check(r['seed']==2026100800+i,'PER_REQUEST_SEED')
            prompt=run/'prompts'/(r['request_id']+'.txt')
            check(digest(prompt)==r['prompt_sha256'],'PROMPT_HASH')
            text=prompt.read_text(); source=read(run/'sources'/(r['root_id']+'.json'))
            check(all(turn['text'] in text for turn in source['turns']),'FULL_SOURCE_UNTRUNCATED',request=r['request_id'])
            check(r['prompt_tokens']+32<=16384,'CONTEXT_BOUND')
        for filename,h in manifest['code_sha256'].items(): check(digest(filename)==h,'MANIFEST_CODE_SHA')
        for filename,h in manifest['input_sha256'].items():
            file=Path(manifest['pool_path']) if filename=='pool' else run/filename
            check(digest(file)==h,'MANIFEST_INPUT_SHA')
    bindings[str(Path(__file__).resolve())]=digest(__file__)
    result={'status':'PASS' if not errors else 'FAIL','actual_auditor':auditor,
            'created_at_utc':datetime.now(timezone.utc).isoformat(),'checks':checks,'errors':errors,
            'independent_internal_actor':True,'external_independence_claimed':False,
            'inference_executed':False,'candidate_count':len(pool),'calls_per_model':len(requests),
            'audited_files':[{'path':p,'sha256':h} for p,h in sorted(bindings.items())]}
    with output.open('x') as f: json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'status':result['status'],'checks':checks,'errors':len(errors),'path':str(output)}))
    return bool(errors)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',type=Path,default=BATCH)
    p.add_argument('--auditor',required=True);p.add_argument('--output',type=Path)
    a=p.parse_args();raise SystemExit(main(a.batch.resolve(),a.auditor,
        (a.output or a.batch/'evidence/INDEPENDENT_EVALUATION_PREFLIGHT_AUDIT.json').resolve()))
