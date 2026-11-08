"""Additive paired v15 evaluation v3: canonical JSON label-key order for export.

This tool is frozen separately before use; it never modifies generation rules,
question text, answers or scores. Native runner/decoder/permutations unchanged.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import os
import sys

sys.dont_write_bytecode = True
B = Path(__file__).resolve().parents[1]
NATIVE = B.parent / 'h4_feedback_training_pool_v15_qa/evaluation'
SCOPE = B / 'evaluations/paired_v1'
MODELS = {'base':'M0', 'trained':'M_clean_policy'}
POLICY = 'v15_full_only_random_mcq6x3_order12x1_v1'
SEED = 2026100800
sys.path.insert(0, str(B/'tools'))
import scene_controller_v120 as scene


def need(ok, why):
    if not ok: raise ValueError(why)
def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p): return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def verify(rows):
    for r in rows: need(sha(r['path'])==r['sha256'], 'BOUND_FILE_CHANGED:'+r['path'])
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
def unique(rows):
    d={}
    for r in rows:
        need(r['path'] not in d or d[r['path']]==r,'CONFLICTING_BINDING');d[r['path']]=r
    return [d[k] for k in sorted(d)]
def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def adapter_text():
    text=(NATIVE/'protocol.py').read_text()
    old='if provenance.get("requested_model") != "Qwen/Qwen3.6-35B-A3B":'
    loc='CONTRACT_DIR = Path(__file__).resolve().parent.parent'
    need(text.count(old)==text.count(loc)==1,'NATIVE_PROTOCOL_CHANGED')
    return text.replace(old,'if (provenance.get("backend") != "CURRENT_CODEX_SESSION" or provenance.get("source_batch_id") != '+repr(B.name)+'):').replace(loc,'CONTRACT_DIR = Path('+repr(str(NATIVE.parent))+')')


def native():
    sys.path.insert(0,str(NATIVE))
    obj=module('v121_native_solver_runner',NATIVE/'run_solver.py')
    need(obj.POLICY_VERSION==POLICY and obj.SEED==SEED,'NATIVE_POLICY_CHANGED')
    path=SCOPE/'adapter_protocol.py'
    need(path.read_text()==adapter_text(),'ADAPTER_NOT_PROVENANCE_ONLY')
    obj.validate_pool=module('v121_provenance_protocol',path).validate_pool
    return obj


def freeze():
    need(not (B/'EVALUATION_TOOLS_FREEZE_V3.json').exists(),'EVALUATION_TOOLS_ALREADY_FROZEN')
    need(not SCOPE.exists() and not list((B/'evaluations').glob('**/*STARTED.json')),'EVALUATION_ALREADY_STARTED_BEFORE_TOOL_FREEZE')
    verify(read(B/'MASTER_FREEZE.json')['files'])
    verify(read(B/'EVALUATION_TOOLS_FREEZE.json')['files'])
    verify(read(B/'EVALUATION_TOOLS_FREEZE_V2.json')['files'])
    need(read(B/'evidence/EVALUATION_CANONICAL_LABEL_AUDIT_A_V1.json')['pass'] is True,'INDEPENDENT_CANONICAL_ADAPTER_AUDIT_REQUIRED')
    sys.path.insert(0,str(NATIVE))
    obj=module('v121_solver_binding_runner',NATIVE/'run_solver.py')
    need(obj.POLICY_VERSION==POLICY and obj.SEED==SEED,'NATIVE_POLICY_CHANGED')
    probe=obj.load_probe('v121_solver_binding_probe')
    paths=[Path(__file__),B/'scripts/codex_evidence_export.py',B/'evaluation_tools/export_canonical_labels_v3.py',B/'evaluation_tools/evaluate_v121_terminal_v2.py',B/'EVALUATION_TOOLS_FREEZE_V2.json',*NATIVE.glob('*.py'),*NATIVE.parent.glob('*.py'),
           obj.BASE_RUNNER,obj.MODEL_AUDIT,obj.TRAINED_AUDIT,B/'MASTER_FREEZE.json',B/'ROLE_ASSIGNMENTS.json',B/'EVALUATION_TOOLS_FREEZE.json',B/'evaluation_tools/evaluate_v121.py']
    for imported in list(sys.modules.values()):
        val=getattr(imported,'__file__',None)
        if val:
            p=Path(val).resolve()
            if p.suffix=='.py' and p.is_relative_to(probe.WORKTREE.resolve()):paths.append(p)
    save(B/'EVALUATION_TOOLS_FREEZE_V3.json',{'status':'FROZEN_BEFORE_SOLVER_PREPARATION_OR_CALLS','created_at_utc':now(),
        'files':unique([bind(p) for p in paths]),'native_policy':POLICY,'seed':SEED,'paired_models':MODELS,
        'automatic_retry':False,'score_guided_rewriting':False,'training_or_pool_merge':False,
        'supersedes_unused_tooling_freeze':bind(B/'EVALUATION_TOOLS_FREEZE_V2.json'),'change_reason':'Canonicalize only in-memory JSON object iteration order, preserving exact six label/text pairs, source, Gold, evidence, native scoring and generation.','independent_audit':bind(B/'evidence/EVALUATION_CANONICAL_LABEL_AUDIT_A_V1.json'),'generation_master_unchanged':True,'binding_scope':'Additive evaluation stage only; no generation runtime change.'})
    print(json.dumps({'status':'EVALUATION_TOOLING_FROZEN','native_policy':POLICY}))


def tooling():
    verify(read(B/'EVALUATION_TOOLS_FREEZE_V3.json')['files']);verify(read(B/'MASTER_FREEZE.json')['files'])


def export(folder):
    obj=module('v121_canonical_label_exporter',B/'evaluation_tools/export_canonical_labels_v3.py')
    folder=Path(folder).resolve();questions=read(folder/'PUBLIC_QUESTIONS.json')['questions']
    expected=folder/'CUMULATIVE_SOURCE.json';unit=folder.parents[1]
    bases=[unit,B,B.parent.parent]
    valid=[base for base in bases if all((Path(q['source_path']) if Path(q['source_path']).is_absolute() else base/q['source_path']).resolve()==expected for q in questions)]
    need(valid,'NO_CANONICAL_SOURCE_DIRECTORY')
    previous=Path.cwd()
    try:
        os.chdir(valid[0]);out=obj.export_root_questions(folder)
    finally:os.chdir(previous)
    for q in out:
        q['source_path']=str(expected)
        need(q['source_sha256']==sha(expected),'EXPORT_SOURCE_HASH')
    return out



def review_failure_terminal(unit, folder, root):
    failure=read(folder/'QA_REVIEW_FAILURE.json')
    role=next(r for r in read(B/'ROLE_ASSIGNMENTS.json')['roots'] if r['root_id']==root)
    reviewer=role['reviewer']
    directory=unit/'evidence/reads'/('public_root_'+hashlib.sha256(reviewer.encode()).hexdigest()[:16])
    need(failure.get('schema_version')=='v121_terminal_qa_review_failure_v1' and failure.get('root_id')==root and failure.get('status')=='QA_REVIEW_SCHEMA_HOLD','REVIEW_FAILURE_IDENTITY:'+root)
    need(failure.get('actual_public_reviewer')==reviewer and failure.get('terminal') is True and failure.get('no_second_semantic_judgment') is True,'REVIEW_FAILURE_NOT_TERMINAL:'+root)
    need(failure.get('original_raw_binding')==bind(directory/'RAW_JUDGMENT.json') and failure.get('save_failure_binding')==bind(directory/'SAVE_FAILURE.json'),'REVIEW_FAILURE_CANONICAL_BINDINGS:'+root)
    saved=read(directory/'SAVE_FAILURE.json')
    need(saved.get('status')=='FIRST_JUDGMENT_REJECTED_PRESERVED' and saved.get('executor_id')==reviewer and saved.get('raw_judgment_binding')==failure['original_raw_binding'],'REVIEW_FAILURE_ORIGIN:'+root)
    need(not (directory/'SAVE_RECEIPT.json').exists() and not (folder/'PUBLIC_SOURCE_JUDGMENTS.json').exists() and not (folder/'PRIVATE_GOLD.json').exists(),'REVIEW_FAILURE_LATER_SUCCESS_OR_GOLD:'+root)
    corrected=failure.get('corrected_payload_binding')
    need(isinstance(corrected,dict),'REVIEW_FAILURE_CORRECTED_PAYLOAD_BINDING:'+root)
    verify([corrected])
    original=read(directory/'RAW_JUDGMENT.json');value=read(corrected['path'])
    public={q['candidate_id']:q for q in read(folder/'PUBLIC_QUESTIONS.json')['questions']}
    added=[]
    for cid,item in value.get('questions',{}).items():
        before=original.get('questions',{}).get(cid,{})
        if 'candidate_id' not in before and 'candidate_id' in item:
            need(cid in public and public[cid]['question_type']=='multi_select' and item['candidate_id']==cid,'REVIEW_FAILURE_NON_MECHANICAL_IDENTIFIER:'+root)
            del item['candidate_id'];added.append(cid)
    need(added and value==original and saved.get('error')=='PUBLIC_JUDGMENT_CANDIDATE_MISMATCH','REVIEW_FAILURE_HAS_SEMANTIC_OR_OTHER_CHANGE:'+root)
    return {'root_id':root,'status':'QA_REVIEW_SCHEMA_HOLD','questions':0,'failure_binding':bind(folder/'QA_REVIEW_FAILURE.json'),'missing_ids_only':added}


def prepare():
    tooling();need(not SCOPE.exists(),'EVALUATION_SCOPE_EXISTS_NO_OVERWRITE')
    questions=[];terminal=[];source_groups={};bindings=[bind(B/'EVALUATION_TOOLS_FREEZE_V3.json')]
    for row in read(B/'MANIFEST.json')['roots']:
        root=row['root_id']
        source_groups[root]={'source_status':row['source_status'],'source_finalize_binding':row['source_finalize_binding']}
        if not row['prefix_reuse_approved']:
            terminal.append({'root_id':root,'status':'PREFIX_HOLD' if row.get('source_ready', True) else 'SOURCE_HOLD','questions':0});continue
        unit=Path(row['unit_path']);state=scene.inspect_root(unit,scene.load_config(unit),root,record_reviews=False)
        folder=scene.root_folder(unit,root)
        if state['root_stopped']:
            terminal.append({'root_id':root,'status':'SCENE_HOLD','ready_scenes':state['ready_scene_count'],'questions':0})
        else:
            need(state['full_expansion_complete'],'WAIT_FOR_SIX_SCENES:'+root)
            if (folder/'QA_REVIEW_FAILURE.json').exists():
                terminal.append(review_failure_terminal(unit,folder,root))
            elif (folder/'QA_AUTHOR_FAILURE.json').exists():
                terminal.append({'root_id':root,'status':'QA_FIRST_ATTEMPT_FAILED','questions':0})
            else:
                need((folder/'FINAL_CROSSCHECK.json').is_file(),'WAIT_FOR_FINALIZE:'+root)
                for action in ['COMPOSE','FINALIZE']:
                    p=unit/'evidence'/('GUARDED_'+action+'_RECEIPT.json');receipt=read(p)
                    need(receipt['status']=='GUARDED_STAGE_COMPLETED' and receipt['gate']['pass'] is True and receipt['downstream_returncode']==0,'GUARDED_STAGE_NOT_SUCCESSFUL:'+root+':'+action)
                out=export(folder);cross=read(folder/'FINAL_CROSSCHECK.json')
                need({q['candidate_id'] for q in out}=={q['candidate_id'] for q in cross['results'] if q['status']=='CONTENT_PASS'},'CONTENT_PASS_COVERAGE')
                questions+=out;terminal.append({'root_id':root,'status':'FINALIZED','questions':len(out),'unfilled_slots':read(folder/'COMPOSITION_RECEIPT.json')['unfilled_slots']})
        bindings += [bind(p) for p in unit.rglob('*.json') if p.is_file()]
    need(len({q['candidate_id'] for q in questions})==len(questions),'DUPLICATE_CANDIDATES')
    SCOPE.mkdir(parents=True)
    with (SCOPE/'adapter_protocol.py').open('x') as f:f.write(adapter_text())
    save(SCOPE/'EVALUATION_POOL.json',{'status':'SOURCE_FIRST_CONTENT_FROZEN_FOR_PAIRED_SOLVER',
        'provenance':{'backend':'CURRENT_CODEX_SESSION','source_batch_id':B.name,'exact_model_id':None,'same_inherited_model':True},
        'source_groups':source_groups,'questions':questions})
    bindings += [bind(SCOPE/'adapter_protocol.py'),bind(SCOPE/'EVALUATION_POOL.json')]
    if questions:native().validate_pool(SCOPE/'EVALUATION_POOL.json')
    count=sum(18 if q['question_type']=='multi_select' else 12 for q in questions)
    save(SCOPE/'ADAPTER_BINDINGS.json',{'created_at_utc':now(),'files':unique(bindings),'terminal_roots':terminal,
        'candidate_count':len(questions),'calls_per_model':count,'models':MODELS,'policy':POLICY,'seed':SEED,
        'gold6_separate':True,'automatic_retry':False,'training_or_pool_merge':False})
    print(json.dumps({'status':'PAIRED_POOL_PREPARED','questions':len(questions),'calls_per_model':count}))


def frozen():
    tooling();a=read(SCOPE/'ADAPTER_BINDINGS.json');verify(a['files']);need(a['candidate_count']>0,'EMPTY_POOL');return a


def preflight():
    a=frozen();need(not (SCOPE/'run').exists(),'PREFLIGHT_ALREADY_EXISTS')
    obj=native()
    for side,model in MODELS.items():obj.preflight(SCOPE/'EVALUATION_POOL.json',SCOPE/'run'/side,solver_model=model)
    mm=[read(SCOPE/'run'/side/'manifest.json')for side in MODELS]
    need(all(m['request_count']==a['calls_per_model'] for m in mm),'PREFLIGHT_CALL_COUNT')
    need(mm[0]['requests']==mm[1]['requests'] and mm[0]['decoder_contract']==mm[1]['decoder_contract'],'PAIRED_REQUESTS_DIFFER')
    for name in ['questions.json','gold.json','canonical_bindings.json']:
        need(read(SCOPE/'run/base'/name)==read(SCOPE/'run/trained'/name),'PAIRED_INPUTS_DIFFER:'+name)
    maximum=max(r['prompt_tokens'] for m in mm for r in m['requests']);need(maximum+32<=16384,'PROMPT_EXCEEDS_NATIVE_CONTEXT')
    paths=[]
    for side in MODELS:
        run=SCOPE/'run'/side
        paths += [run/n for n in ['manifest.json','questions.json','gold.json','canonical_bindings.json','solver_identity.json']]
        paths += list((run/'sources').glob('*.json'))+list((run/'prompts').glob('*.txt'))
    save(SCOPE/'PAIRED_PREFLIGHT.json',{'status':'PASS','created_at_utc':now(),'files':[bind(p) for p in paths],
        'adapter_binding':bind(SCOPE/'ADAPTER_BINDINGS.json'),'calls_per_model':a['calls_per_model'],'max_prompt_tokens':maximum})
    print(json.dumps({'status':'PAIRED_PREFLIGHT_PASS','calls_per_model':a['calls_per_model'],'max_prompt_tokens':maximum}))


def live(side):
    a=frozen();pre=read(SCOPE/'PAIRED_PREFLIGHT.json');verify(pre['files']);verify([pre['adapter_binding']])
    run=SCOPE/'run'/side;obj=native();need(Path(sys.prefix)==obj.EXPECTED_VENV,'USE_PROVEN_LOCAL4B_VENV')
    obj._verify_manifest(run,read(run/'manifest.json'))
    save(SCOPE/(side.upper()+'_STARTED.json'),{'started_at_utc':now(),'side':side,'automatic_retry':False})
    obj.live(run);obj.report(run);verify(pre['files']);verify(a['files'])
    need(read(run/'COMPLETE.json')['actual_call_count']==a['calls_per_model'],'INCOMPLETE_CALLS')
    save(SCOPE/(side.upper()+'_COMPLETE.json'),{'completed_at_utc':now(),'side':side,'calls':a['calls_per_model'],
        'results_binding':bind(run/'RESULTS.json'),'training_or_pool_merge':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','prepare','preflight','live']);p.add_argument('--side',choices=list(MODELS));a=p.parse_args()
    if a.action=='live':need(a.side,'SIDE_REQUIRED');live(a.side)
    else:globals()[a.action]()
