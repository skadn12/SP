"""Independent CPU-only preflight. Synthetic files are confined to audit temp dirs."""
from pathlib import Path
import copy, contextlib, datetime, hashlib, importlib.util, io, json, shutil, sys, tempfile
sys.dont_write_bytecode=True
AUDIT=Path(__file__).resolve().parent;WORK=AUDIT.parent;EX=WORK/'execution'
sys.path.insert(0,str(EX))
import run_pilot as run
import score_pilot as score
import provider_client as provider
CHECKS=[]
def check(ok,label):
    assert ok,label
    CHECKS.append(label)
def rejects(fn,label):
    try:fn()
    except (AssertionError,ValueError,TypeError,KeyError,IndexError):CHECKS.append(label);return
    raise AssertionError('DID_NOT_REJECT:'+label)
def put(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');return p
def binding(p):return {'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def fake_role(name,payload,schema,cap,directory):
    """No inference; test only controller/scorer interfaces with artificial strings."""
    d=Path(directory);speakers=sorted({t['speaker'] for t in payload.get('context',[])})
    if name=='planner':value={'plan':['A synthetic CPU-only step.'],'visible_speakers':speakers}
    elif name=='realizer':
        value={'turns':[{'speaker':speakers[i%len(speakers)],'text':hashlib.sha256((str(d)+'/'+str(i)).encode()).hexdigest()} for i in range(12)]}
    elif name=='source_reviewer':
        value={k:payload[k] for k in ('root_id','branch_id','scene_index','candidate_id','candidate_sha256','accepted_prefix_sha256')}
        value.update(decision='PASS',critical_major_error_count=0,new_progress_count=1,
                     duplicate_turn_ratio=payload['mechanical_duplicate_turn_ratio'],old_history_linkage_count=0,
                     all_plausible_prior_states_checked=True,
                     evidence_refs=[{'kind':'new_progress','candidate_turn_index':0,'quote':payload['candidate']['turns'][0]['text'],'prefix_turn_index':None,'prefix_quote':None,'explanation':'CPU synthetic fixture only.'}])
    elif name=='memory_update':
        value={'keep_fact_ids':[f['fact_id'] for f in payload['previous_approved_memory']],'new_facts':[],'supersede_fact_ids':[]}
    elif name=='memory_review':
        value={k:payload[k] for k in ('root_id','snapshot_sha256','full_source_sha256')}
        value.update(decision='PASS',all_facts_reviewed=True,accepted_fact_ids=[f['fact_id'] for f in payload['candidate_memory']['facts']],coverage_status='BOUNDED',review_notes=['CPU synthetic fixture only.'])
    else:raise AssertionError(name)
    put(d/'RESPONSE.json',value);put(d/'ROLE_RECEIPT.json',{'status':'SYNTHETIC_CPU_ONLY','role':name})
    return value

def main():
    check(provider.strict_json('{"a":1}')=={'a':1},'strict JSON accepts ordinary object')
    for text in ('{"a":1,"a":2}','{"a":NaN}','{"a":Infinity}'):
        rejects(lambda text=text:provider.strict_json(text),'strict JSON rejects '+text)
    check(not provider.tool_activity({'a':[0,0.0,False,None]}),'zero nested tool usage is inactive')
    check(provider.tool_activity({'a':[0,{'tokens':1}]}),'nonzero nested tool usage is detected')
    roots=run.ROOTS
    check(len(roots)==6 and len(set(roots))==6 and run.PROTOCOL['planned_scenes']==54,'fixed six roots and54 slots')
    index=run.read(WORK/'initial_memory/INITIAL_MEMORY_INDEX.json')
    check([r['root_id'] for r in index['roots']]==roots,'initial memory root selection identical')
    initial={}
    for root in roots:
        src=run.OLD/'sources'/(root+'.json');d=WORK/'initial_memory'/root
        m=run.memory_gate(d/'FACT_MEMORY_SNAPSHOT.json',d/'SEMANTIC_REVIEW.json',src);initial[root]=m
        check(True,root+' exact-hash initial memory gate')
        row=next(x for x in index['roots'] if x['root_id']==root)
        for key in ('snapshot','review','role_call','original_candidate'):
            check(run.sha(row[key]['path'])==row[key]['sha256'],root+' index binding '+key)
        source=run.read(src);sessions=list(dict.fromkeys(t['session_ref'] for t in source['turns']))
        recent=[{k:t[k] for k in ('turn_ref','session_ref','speaker','text')} for t in source['turns'] if t['session_ref'] in sessions[-2:]]
        for arm in run.ARMS:
            c=run.context(source,arm,m)
            expected=[{k:t[k] for k in ('turn_ref','session_ref','speaker','text')} for t in source['turns']] if arm=='full_history' else recent
            check(c['context']==expected,root+' '+arm+' correct complete-session context')
            check((c['prefix_fact_memory']==run.memory_projection(m)) if arm==run.ARMS[1] else c['prefix_fact_memory'] is None,root+' '+arm+' memory isolation')
            check(all(set(t)=={'turn_ref','session_ref','speaker','text'} for t in c['context']),root+' '+arm+' context metadata limited')
    for func in (run.duplicate_ratio,score.duplicate_ratio):
        check(func([{'text':'prior unique content'}],[{'text':'prior unique content'}])==1,'exact prior duplicate '+func.__module__)
        check(func([], [{'text':'abc 123'},{'text':'abc 123'}])==.5,'earlier-new duplicate '+func.__module__)
        check(func([], [{'text':'Hello!'},{'text':'entirely novel text'}])==0,'greeting denominator exclusion '+func.__module__)
        rejects(lambda func=func:func([], [{'text':'Hello!'}]),'zero denominator rejection '+func.__module__)
    cand={'turns':[{'speaker':'A','text':'A new explicit action.'}]};prefix={'turns':[{'text':'older source quote'},{'text':'recent source quote'}]};ids={'root_id':'test-root','candidate_sha256':'a'*64}
    base={**ids,'decision':'PASS','critical_major_error_count':0,'new_progress_count':1,'old_history_linkage_count':0,'all_plausible_prior_states_checked':True,'duplicate_turn_ratio':0,'evidence_refs':[{'kind':'new_progress','candidate_turn_index':0,'quote':'explicit action','prefix_turn_index':None,'prefix_quote':None}]}
    check(run.check_review(copy.deepcopy(base),ids,cand,prefix,0,1),'valid exact cited review passes')
    for field,value in [('root_id','other'),('critical_major_error_count',True),('all_plausible_prior_states_checked',False),('new_progress_count',2)]:
        bad=copy.deepcopy(base);bad[field]=value
        rejects(lambda bad=bad:run.check_review(bad,ids,cand,prefix,0,1),'review rejects bad '+field)
    bad=copy.deepcopy(base);bad['evidence_refs']*=2
    rejects(lambda:run.check_review(bad,ids,cand,prefix,0,1),'review rejects duplicate evidence')
    for field,value in [('quote','nonexistent'),('quote',7),('prefix_turn_index',1)]:
        bad=copy.deepcopy(base);bad['evidence_refs'][0][field]=value
        rejects(lambda bad=bad:run.check_review(bad,ids,cand,prefix,0,1),'review rejects invalid evidence '+field+' '+str(value))
    bad=copy.deepcopy(base);bad['old_history_linkage_count']=1;bad['evidence_refs'].append({'kind':'old_history_linkage','candidate_turn_index':0,'quote':'new','prefix_turn_index':1,'prefix_quote':'recent'})
    rejects(lambda:run.check_review(bad,ids,cand,prefix,0,1),'review rejects linkage to recent boundary')
    check(not run.check_review(copy.deepcopy(base),ids,cand,prefix,.3,1),'duplicate threshold cannot pass')
    old_work,old_pilot,old_role,old_orig=run.WORK,run.PILOT,run.role,score.original_source
    try:
        with tempfile.TemporaryDirectory(prefix='runtime_cpu_',dir=AUDIT) as tmp:
            w=Path(tmp);run.WORK=w;run.PILOT=w/'pilot';run.role=fake_role
            score.original_source=lambda work,root:run.OLD/'sources'/(root+'.json')
            for root in roots:shutil.copytree(old_work/'initial_memory'/root,w/'initial_memory'/root)
            put(w/'EXECUTION_FREEZE.json',{'source_hashes':{r:run.sha(run.OLD/'sources'/(r+'.json')) for r in roots}})
            empty=score.score(w,run.PROTOCOL)
            check(len(empty['receipts'])==54 and all(empty['arms'][a]['status_counts']['MISSING']==18 for a in run.ARMS),'empty scoreboard retains54 denominator')
            with contextlib.redirect_stdout(io.StringIO()):
                for root in roots:
                    for arm in run.ARMS:run.run_branch(root,arm)
            check(len(list((w/'pilot/branches').glob('*/scene_*/SCENE_RECEIPT.json')))==54,'mock full controller persists54 receipts')
            s=score.score(w,run.PROTOCOL)
            check(all(s['arms'][a]['accepted_initial_scenes']==18 for a in run.ARMS),'independent scorer accepts all54 valid synthetic scenes')
            check(len(s['maintenance_memory_gates'])==12 and all(x['exact_hash_semantic_gate_pass'] for x in s['maintenance_memory_gates']),'twelve sibling memory gates exact-prefix reviewed')
            check(s['quality_gates_all_pass'] and s['recent_arm_ceiling_case'],'frozen quality thresholds and recent ceiling')
            check(s['cost_status']=='NO_COST_CONCLUSION' and not s['memory_adopted'] and s['selected_context_policy']=='full_history','unknown initial cost forces no adoption and full fallback')
            gate=w/'pilot/branches'/(roots[0]+'__'+run.ARMS[1])/'scene_01/MEMORY_GATE.json'
            original=gate.read_bytes();bad=run.read(gate);bad['decision']='FAIL';put(gate,bad)
            badscore=score.score(w,run.PROTOCOL)
            check(not badscore['quality_gates']['all_used_maintenance_memory_semantic_pass'],'independent scorer rejects failed sibling maintenance gate')
            gate.write_bytes(original)
            receipt=w/'pilot/branches'/(roots[0]+'__full_history')/'scene_01/SCENE_RECEIPT.json';rr=run.read(receipt)
            out=Path(rr['accepted_output_prefix_binding']['path']);altered=run.read(out);altered['turns'][-1]['text']='tampered';put(out,altered)
            ss=score.score(w,run.PROTOCOL)
            check(ss['arms']['full_history']['status_counts']['HOLD']==3 and ss['arms']['full_history']['accepted_initial_scenes']==15,'tampered accepted prefix blocks initial scene and descendants')
        with tempfile.TemporaryDirectory(prefix='runtime_initial_failure_',dir=AUDIT) as tmp:
            w=Path(tmp);run.WORK=w;run.PILOT=w/'pilot';put(w/'EXECUTION_FREEZE.json',{'source_hashes':{r:run.sha(run.OLD/'sources'/(r+'.json')) for r in roots}})
            mg=run.memory_gate
            def fail_memory(*args):raise ValueError('SYNTHETIC_INITIAL_GATE_FAILURE')
            def no_roles(*args):raise AssertionError('ROLE_CALLED_AFTER_GATE_FAILURE')
            run.memory_gate=fail_memory;run.role=no_roles
            try:
                with contextlib.redirect_stdout(io.StringIO()):run.run_branch(roots[0],run.ARMS[1])
            finally:run.memory_gate=mg
            rows=[run.read(p) for p in sorted((w/'pilot/branches').glob('*/scene_*/SCENE_RECEIPT.json'))]
            check([r['status'] for r in rows]==['HOLD','MISSING','MISSING'],'initial memory gate failure still persists all3 terminal slots')
    finally:run.WORK=old_work;run.PILOT=old_pilot;run.role=old_role;score.original_source=old_orig
    result={'status':'PASS','auditor':'/root/frozen_audit','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'check_count':len(CHECKS),'checks':CHECKS,'model_calls':0,'network_calls':0,'synthetic_scenes_are_experiment_outcomes':False,'sealed_input_mutations':0,'reviewed_files':[binding(EX/n) for n in ('run_pilot.py','provider_client.py','score_pilot.py')],'audit_script':binding(Path(__file__))}
    put(AUDIT/'RUNTIME_CPU_CHECKS.json',result)
    print(json.dumps({'status':'PASS','check_count':len(CHECKS),'model_calls':0,'network_calls':0}))
if __name__=='__main__':main()
