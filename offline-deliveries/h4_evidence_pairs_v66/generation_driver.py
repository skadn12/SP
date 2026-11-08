"""One bounded cross-root asset run. No automatic replay or training."""
import argparse
import json
from pathlib import Path
import traceback
import asset_generation as assets

HERE=Path(__file__).resolve().parent

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')

def verify_plan():
    plan=assets.read(HERE/'EXECUTION_PLAN.json')
    for row in plan['files']:
        assets.require(assets.sha(row['path'])==row['sha256'],'execution_input_changed:'+row['path'])
    return plan

def prepare():
    assets.require(not (HERE/'PREPARATION.STARTED.json').exists(),'preparation_already_started')
    write(HERE/'PREPARATION.STARTED.json',{'time':assets.v15.now(),'training_allowed':False})
    config=assets.read(HERE/'CONFIG.json');rows=[]
    provider,_,annotate,capacity,dev=assets.base.transport()
    tokenizer=dev.tokenizer_from(str(assets.base.TOKENIZER))
    try:
        for slot in config['slots']:
            sid=slot['slot_id'];run=HERE/'assets'/sid
            prepared=assets.prepare(HERE/'CONFIG.json',sid)
            payload=assets.read(run/'INITIAL_REQUEST.json')
            profile=assets.read(run/'provider_profile.json')
            profile['output_limit']=payload['max_tokens'];profile['request_parameters']['max_tokens']=payload['max_tokens']
            cap=annotate(capacity(profile,tokenizer,payload['messages'],payload['max_tokens']))
            write(run/'CAPACITY.json',cap)
            assets.require(cap['status']=='PASS','initial_capacity_blocked:'+sid)
            rows.append({'slot_id':sid,'root_id':slot['root_id'],'run':str(run),'manifest':assets.binding(run/'MANIFEST.json'),'capacity':assets.binding(run/'CAPACITY.json'),'required_context_tokens':cap['required_context_tokens']})
            print(json.dumps({'event':'prepared','slot_id':sid,'context_tokens':cap['required_context_tokens']}),flush=True)
        files=[HERE/'CONFIG.json',HERE/'EXECUTION_DECISION.md',HERE/'PROPOSED_DESIGN.md',HERE/'ROOT_INVENTORY.json',*sorted(HERE.glob('*.py'))]
        for row in rows:files += [Path(row['manifest']['path']),Path(row['capacity']['path'])]
        plan={'version':'v66_execution_v1','time':assets.v15.now(),'slots':rows,'max_http':20,'max_questions':10,'max_calls_per_model':180,'training_allowed':False,'files':[assets.binding(p) for p in files]}
        write(HERE/'EXECUTION_PLAN.json',plan)
        write(HERE/'PREPARATION.COMPLETE.json',{'time':assets.v15.now(),'plan':assets.binding(HERE/'EXECUTION_PLAN.json'),'slots':len(rows),'http_attempts':0})
    except BaseException as e:
        write(HERE/'PREPARATION.ERROR.json',{'time':assets.v15.now(),'error':repr(e),'traceback':traceback.format_exc(),'completed':rows,'automatic_retry':False});raise

def generate():
    plan=verify_plan();assets.require(not (HERE/'GENERATION.STARTED.json').exists(),'generation_already_started')
    write(HERE/'GENERATION.STARTED.json',{'time':assets.v15.now(),'plan':assets.binding(HERE/'EXECUTION_PLAN.json'),'automatic_retry':False})
    rows=[]
    try:
        for slot in plan['slots']:
            verify_plan();run=Path(slot['run'])
            result=assets.generate(run)
            row={k:result[k] for k in ('slot_id','root_id','status','fact_count','http_attempts','requests_prepared')}
            row['completion']=assets.binding(run/'GENERATION.COMPLETE.json');rows.append(row)
            assets.require(sum(x['http_attempts'] for x in rows)<=20,'global_http_budget_exceeded')
            print(json.dumps({'event':'slot_complete',**row}),flush=True)
        write(HERE/'GENERATION.COMPLETE.json',{'time':assets.v15.now(),'plan':assets.binding(HERE/'EXECUTION_PLAN.json'),'slots':rows,'http_attempts':sum(x['http_attempts'] for x in rows),'training_allowed':False,'automatic_followup':False})
    except BaseException as e:
        write(HERE/'GENERATION.ERROR.json',{'time':assets.v15.now(),'error':repr(e),'traceback':traceback.format_exc(),'completed':rows,'automatic_retry':False});raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','generate','verify']);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='generate':generate()
    else:verify_plan();print('PASS')
