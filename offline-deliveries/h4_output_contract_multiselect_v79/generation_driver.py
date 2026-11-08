"""Freeze bounded initial parent requests and at most one reviewed supplement per family."""
import argparse
import json
from pathlib import Path
import traceback
import asset_generation as assets
HERE=Path(__file__).resolve().parent
VERSION='v79_bounded_parent_family_execution_v1'

def write(path,value):assets.write(path,value)
def verify_plan(home=HERE):
    home=Path(home).resolve();plan=assets.read(home/'EXECUTION_PLAN.json')
    assets.require(plan.get('version')==VERSION and 0<plan.get('max_http',0)<=8
        and plan.get('training_allowed') is False and plan.get('automatic_followup') is False,'execution_contract_changed')
    assets.base.verify_bindings(plan['files'])
    config=assets.read(home/'CONFIG.json')
    assets.require(len(plan['slots'])==len(config['slots']) and plan['max_http']==config['max_http']==2*len(config['slots']) and {r['slot_id'] for r in plan['slots']}=={s['slot_id'] for s in config['slots']},'execution_families_changed')
    expected={home/'assets'/s['slot_id'] for s in config['slots']}
    assets.require(set((home/'assets').iterdir())==expected and all(p.is_dir() for p in expected)
        and {Path(row['run']).resolve() for row in plan['slots']}==expected,'unregistered_or_missing_executed_family')
    for row in plan['slots']:assets.verify(row['run'])
    return plan

def prepare(home=HERE):
    home=Path(home).resolve();config=assets.read(home/'CONFIG.json')
    for slot in config['slots']:assets.get_slot(config,slot['slot_id'])
    assets.require(not (home/'PREPARATION.STARTED.json').exists(),'preparation_already_started')
    write(home/'PREPARATION.STARTED.json',{'time':assets.v15.now(),'training_allowed':False})
    rows=[]
    try:
        provider,_,annotate,capacity,dev=assets.base.transport()
        tokenizer=dev.tokenizer_from(str(assets.base.TOKENIZER))
        for slot in config['slots']:
            sid=slot['slot_id'];run=home/'assets'/sid
            assets.prepare(home/'CONFIG.json',sid,run=run)
            payload=assets.read(run/'rounds/initial/INITIAL_REQUEST.json');profile=assets.read(run/'provider_profile.json')
            profile['output_limit']=payload['max_tokens'];profile['request_parameters']['max_tokens']=payload['max_tokens']
            cap=annotate(capacity(profile,tokenizer,payload['messages'],payload['max_tokens']))
            write(run/'CAPACITY.json',cap);assets.require(cap['status']=='PASS','initial_capacity_blocked:'+sid)
            rows.append({'slot_id':sid,'family_id':sid,'root_id':slot['root_id'],'run':str(run),
                'manifest':assets.binding(run/'MANIFEST.json'),'capacity':assets.binding(run/'CAPACITY.json'),
                'required_context_tokens':cap['required_context_tokens']})
            print(json.dumps({'event':'prepared','slot_id':sid,'context_tokens':cap['required_context_tokens']}),flush=True)
        files=set(home.glob('*.py'))|{home/'CONFIG.json'}
        files.update(p for p in home.glob('*.md'))
        files.update(p for p in (home/'execution').glob('*.json'))
        for row in rows:
            run=Path(row['run']);files.update((Path(row['manifest']['path']),Path(row['capacity']['path']),run/'rounds/initial/INITIAL_REQUEST.json',run/'rounds/initial/INITIAL_PROOF.json'))
        plan={'version':VERSION,'time':assets.v15.now(),'slots':rows,'max_http':config['max_http'],
            'max_questions':config['max_questions'],'max_calls_per_model':config['max_calls_per_model'],
            'training_allowed':False,'automatic_followup':False,'files':assets.base.bindings(files)}
        write(home/'EXECUTION_PLAN.json',plan)
        write(home/'PREPARATION.COMPLETE.json',{'time':assets.v15.now(),'plan':assets.binding(home/'EXECUTION_PLAN.json'),'slots':len(rows),'http_attempts':0})
        verify_plan(home);return plan
    except BaseException as e:
        write(home/'PREPARATION.ERROR.json',{'time':assets.v15.now(),'error':repr(e),'traceback':traceback.format_exc(),'completed':rows,'automatic_retry':False});raise

def generate(home=HERE):
    home=Path(home).resolve();plan=verify_plan(home)
    assets.require(not (home/'GENERATION.STARTED.json').exists(),'generation_already_started')
    write(home/'GENERATION.STARTED.json',{'time':assets.v15.now(),'plan':assets.binding(home/'EXECUTION_PLAN.json'),'automatic_retry':False})
    rows=[]
    try:
        for slot in plan['slots']:
            verify_plan(home);result=assets.generate(slot['run'])
            row={k:result[k] for k in ('slot_id','family_id','root_id','status','fact_count','counterversion_count','http_attempts','requests_prepared')}
            row['completion']=assets.binding(Path(slot['run'])/'rounds/initial/GENERATION.COMPLETE.json');rows.append(row)
            assets.require(sum(r['http_attempts'] for r in rows)<=plan['max_http'],'global_http_budget_exceeded')
            print(json.dumps({'event':'family_complete',**row}),flush=True)
        result={'time':assets.v15.now(),'plan':assets.binding(home/'EXECUTION_PLAN.json'),'slots':rows,
            'http_attempts':sum(r['http_attempts'] for r in rows),'training_allowed':False,'automatic_followup':False,
            'mechanical_repairs':0,'semantic_retries':0,'transport_retries':0}
        write(home/'INITIAL_GENERATION.COMPLETE.json',result);return result
    except BaseException as e:
        write(home/'GENERATION.ERROR.json',{'time':assets.v15.now(),'error':repr(e),'traceback':traceback.format_exc(),'completed':rows,'automatic_retry':False});raise


def finalize(home=HERE):
    home=Path(home).resolve();plan=verify_plan(home)
    assets.require(not (home/'GENERATION.COMPLETE.json').exists(),'global_generation_already_finalized')
    # Validate every actual round before completing any family. A later family's
    # missing review must not leave earlier families partially finalized.
    for slot in plan['slots']:
        run=Path(slot['run']);rounds=run/'rounds'
        assets.require((rounds/'initial').is_dir(),'initial_round_missing:'+slot['slot_id'])
        entries=list(rounds.iterdir());actual={p.name for p in entries}
        assets.require(actual <= {'initial','supplement'} and all(p.is_dir() for p in entries),
            'unexpected_round_directory:'+slot['slot_id'])
        for rid in ('initial','supplement'):
            if rid in actual:assets.reviewed_round(run,rid)
    rows=[]
    for slot in plan['slots']:
        run=Path(slot['run'])
        c=assets.read(run/'GENERATION.COMPLETE.json') if (run/'GENERATION.COMPLETE.json').exists() else assets.finalize(run)
        assets.verify(run,complete=True)
        row={k:c[k] for k in ('slot_id','family_id','root_id','status','fact_count','counterversion_count','http_attempts','requests_prepared','bounded_supplements')}
        row['completion']=assets.binding(run/'GENERATION.COMPLETE.json');rows.append(row)
    assets.require(sum(r['http_attempts'] for r in rows)<=plan['max_http'],'global_http_budget_exceeded')
    result={'time':assets.v15.now(),'plan':assets.binding(home/'EXECUTION_PLAN.json'),'slots':rows,
        'http_attempts':sum(r['http_attempts'] for r in rows),'bounded_supplements':sum(r['bounded_supplements'] for r in rows),
        'training_allowed':False,'automatic_followup':False,'mechanical_repairs':0,'semantic_retries':0,'transport_retries':0}
    write(home/'GENERATION.COMPLETE.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','generate','finalize','verify'));a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='generate':generate()
    elif a.action=='finalize':finalize()
    else:verify_plan();print('PASS')
