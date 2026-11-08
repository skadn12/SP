#!/usr/bin/env python3
"""Bound G136202 saved final-Gold schema recovery and first verifier only.

Global call-scope read attestations stay present. Exact missing row copies and
one exact R03 key alias are reversible. No new Gold, author, plan or revision is
permitted. Semantic admission remains a fresh independent verifier decision.
"""
from __future__ import annotations
import argparse
import ast
import copy
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import types
from typing import Any
sys.dont_write_bytecode=True
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent))
import question_driver as q
import review_schema_adapter as rsa
import saved_revision_gold_verification as saved
ROOT='cc-train-episode-136202'
ARM='G'
INPUT_WORK=BASE/'question_plan_continuation/136202/G'
SELECTION=BASE/'SOURCE_SELECTION_FINAL.json'
EXPECTED_SAVED='d85e11b5db07befe1aab07ebdcc6fa238bc6f36dfff54a44e50c382a704ba5b0'


def verify_code():
    saved.verify_code()
    if hashlib.sha256(Path(saved.__file__).read_bytes()).hexdigest()!=EXPECTED_SAVED:
        raise RuntimeError('Frozen first-verifier engineering helper drift')


def normalize_saved_global_blind(raw: Any):
    if (not isinstance(raw,dict) or set(raw)!={'reviews','complete_source_read','solver_results_read'}
            or raw.get('complete_source_read') is not True or raw.get('solver_results_read') is not False
            or not isinstance(raw.get('reviews'),list) or len(raw['reviews'])!=8):
        raise RuntimeError('Only the exact saved eight-review global true/false schema may be recovered')
    value=copy.deepcopy(raw);changes=[]
    for index,row in enumerate(value['reviews']):
        if not isinstance(row,dict) or any(key in row for key in ('complete_source_read','solver_results_read')):
            raise RuntimeError('A per-record attestation already exists; no broadcast, conflict resolution or overwrite allowed')
        r03=row.get('r03_necessity')
        if (not isinstance(r03,dict) or set(r03)!={'decision','removal_reason','turn_refs'}
                or not isinstance(r03['removal_reason'],str) or not r03['removal_reason'].strip()
                or not isinstance(r03['turn_refs'],list) or not r03['turn_refs']):
            raise RuntimeError('Saved final R03 object is not the exact complete one-key alias variant')
        normalized={('rationale' if key=='removal_reason' else key):copy.deepcopy(v) for key,v in r03.items()}
        row['r03_necessity']=normalized
        changes.append({'review_index':index,'question_key':row.get('question_key'),'kind':'exact_key_alias',
            'field':'r03_necessity','authored_original_value':copy.deepcopy(r03),'normalized_value':copy.deepcopy(normalized),
            'mapping':{'removal_reason':'rationale'}})
        for key in ('complete_source_read','solver_results_read'):
            row[key]=copy.deepcopy(raw[key])
            changes.append({'review_index':index,'question_key':row.get('question_key'),'kind':'call_scope_attestation_copy',
                'field':key,'original_field_absent':True,'source_global_path':[key],'source_global_value':raw[key],
                'normalized_value':row[key],'global_original_retained':True})
    reversed_value=copy.deepcopy(value)
    for change in reversed(changes):
        row=reversed_value['reviews'][change['review_index']]
        if change['kind']=='call_scope_attestation_copy':
            if row[change['field']] is not change['source_global_value']:
                raise RuntimeError('Global attestation copy value changed')
            del row[change['field']]
        else:
            restored={('removal_reason' if key=='rationale' else key):copy.deepcopy(v)
                      for key,v in row['r03_necessity'].items()}
            if restored!=change['authored_original_value']:
                raise RuntimeError('R03 key alias is not exactly reversible')
            row['r03_necessity']=restored
    if reversed_value!=raw or value['complete_source_read'] is not True or value['solver_results_read'] is not False:
        raise RuntimeError('Saved call-global object did not survive exact reverse mapping')
    return value,changes


def inspect_inputs(source):
    verify_code()
    required=('OUTCOME.json','CONTINUATION_FINALIZER_READY.json','TASK_PLANS.json','TASK_PLANS_FREEZE.json',
        'CANDIDATES_DRAFT_CALL.json','CANDIDATES_REVISION_01_RESPONSE.json','CANDIDATES_REVISION_01_CALL.json',
        'CANDIDATES_REVISION_01.json','CANDIDATES_FINAL.json','CONTENT_REVIEW_FINAL.json','SLOT_FAILURES.json',
        'BLIND_GOLD_revision_01.json','BLIND_GOLD_CALL_revision_01.json','BLIND_GOLD_VALIDATION_revision_01.json')
    if any(not (INPUT_WORK/name).is_file() for name in required):
        raise RuntimeError('Saved final-revision workflow has not reached a real terminal surface')
    outcome=q.read(INPUT_WORK/'OUTCOME.json');ready=q.read(INPUT_WORK/'CONTINUATION_FINALIZER_READY.json')
    if ((outcome.get('root_id'),outcome.get('arm'),outcome.get('source_binding'))!=(ROOT,ARM,source['binding'])
            or outcome.get('max_content_revisions')!=1 or outcome.get('status')!='COMPLETED_WITH_HOLES'):
        raise RuntimeError('Saved final-revision identity/source/budget/terminal state drift')
    if (ready.get('root_id'),ready.get('arm'),ready.get('final_work_dir'))!=(ROOT,ARM,str(INPUT_WORK)):
        raise RuntimeError('Original terminal READY identity drift')
    for key in ('final_outcome_binding','driver_outcome_binding','final_candidates_binding','final_review_binding','slot_failures_binding'):
        q.verify_binding(ready[key])
    if ready['final_candidates_binding']!=q.binding(INPUT_WORK/'CANDIDATES_FINAL.json'):
        raise RuntimeError('Original READY does not bind exact final candidates')
    reviews=q.read(INPUT_WORK/'CONTENT_REVIEW_FINAL.json').get('reviews',[])
    if len(reviews)!=8 or any(x.get('decision')=='PASS' for x in reviews):
        raise RuntimeError('This recovery cannot rereview or replace any previously passing candidate')
    initial,initial_id=saved.verify_receipt(INPUT_WORK/'CANDIDATES_DRAFT_CALL.json')
    revision,revision_id=saved.verify_receipt(INPUT_WORK/'CANDIDATES_REVISION_01_CALL.json')
    gold,gold_id=saved.verify_receipt(INPUT_WORK/'BLIND_GOLD_CALL_revision_01.json')
    requests=saved.actual_requests(ROOT,ARM);saved.verify_budget(requests,ROOT,ARM,gold_id)
    if ({x['call_id'] for x in requests['author']}!={initial_id}
            or {x['call_id'] for x in requests['revision']}!={revision_id}):
        raise RuntimeError('Actual consumed author/revision IDs differ from saved receipts')
    if requests['verify'] and not (saved.work_dir(ROOT,ARM)/'RECOVERY_INPUT_FREEZE.json').exists():
        raise RuntimeError('Any preexisting verifier without this recovery freeze blocks first-verifier recovery')
    plan_path=INPUT_WORK/'TASK_PLANS.json';plan_freeze_path=INPUT_WORK/'TASK_PLANS_FREEZE.json'
    plan=q.read(plan_path);plan_freeze=q.read(plan_freeze_path);q.verify_binding(plan_freeze['plans_binding'])
    if plan_freeze['plans_binding']!=q.binding(plan_path) or plan_freeze.get('errors'):
        raise RuntimeError('Actual eight-slot preauthoring plan is not valid and frozen')
    _,plan_errors=q.validate_plans(plan,ARM,ROOT,source['source'],None)
    if plan_errors or {x.get('slot_id') for x in plan.get('plans',[])}!=set(q.SLOTS):
        raise RuntimeError('Actual frozen plan is incomplete or invalid')
    revision_raw=q.read(INPUT_WORK/'CANDIDATES_REVISION_01_RESPONSE.json')
    if revision_raw!=q.read(revision['parsed_response']['path']):
        raise RuntimeError('Saved final revision response differs from actual accepted author response')
    normalized,bad,_=saved.ORIGINAL_NORMALIZE_CANDIDATES(revision_raw,ARM,ROOT,
        initial['backend']['thread_id'],source,plan,q.binding(SELECTION),revision['backend']['thread_id'])
    candidates=q.read(INPUT_WORK/'CANDIDATES_FINAL.json')
    if bad or candidates!=normalized or candidates!=q.read(INPUT_WORK/'CANDIDATES_REVISION_01.json'):
        raise RuntimeError('Final candidate must be actual final revision exactly; no initial fallback or new rewrite')
    if (len(candidates)!=8 or {c.get('slot_id') for c in candidates}!=set(q.SLOTS)
            or any(not q.basic_solver_shape(c,source) or c.get('content_revision_count')!=1
                   or c.get('revision_author_thread_id')!=revision['backend']['thread_id'] for c in candidates)):
        raise RuntimeError('Final revision shape or actual revision lineage is incomplete')
    raw=q.read(INPUT_WORK/'BLIND_GOLD_revision_01.json')
    if raw!=q.read(gold['parsed_response']['path']):
        raise RuntimeError('Saved final blind response differs from its actual accepted parsed response')
    payload=saved.shape.request_payload(Path(gold['request']['path']).read_text())
    if (payload.get('root_id')!=ROOT or payload.get('complete_source_and_warnings')!=q.source_view(source)
            or payload.get('anonymous_questions')!=[q.anonymize_candidate(c)[0] for c in candidates]):
        raise RuntimeError('Actual blind request does not exactly embed the complete source and these final candidates')
    old_map,old_errors=saved.ORIGINAL_NORMALIZE_BLIND(copy.deepcopy(raw),candidates,source['source'])
    expected=[]
    for c in candidates:expected.extend(['blind_r03_invalid:'+c['candidate_id'],'blind_read_attestation_invalid:'+c['candidate_id']])
    if old_errors!=expected or q.read(INPUT_WORK/'BLIND_GOLD_VALIDATION_revision_01.json').get('errors')!=expected:
        raise RuntimeError('Defects exceed the exact one-key R03 alias plus global attestation scope variant')
    adapted,changes=normalize_saved_global_blind(raw)
    blind_map,errors=saved.ORIGINAL_NORMALIZE_BLIND(copy.deepcopy(adapted),candidates,source['source'])
    if errors or len(changes)!=24 or set(blind_map)!={c['candidate_id'] for c in candidates}:
        raise RuntimeError('Reversible structure adaptation still fails frozen Gold/evidence/read/ref checks')
    if {k:v['independent_gold'] for k,v in old_map.items()}!={k:v['independent_gold'] for k,v in blind_map.items()}:
        raise RuntimeError('Independent saved Gold changed during mechanical schema adaptation')
    threads=[x['backend']['thread_id'] for x in (initial,revision,gold)]
    if len(set(threads))!=3:raise RuntimeError('Initial author, revision author and saved Gold reviewer threads are not fresh independent')
    paths=[INPUT_WORK/name for name in required]+[SELECTION,q.SOURCE_FREEZE_PATH,BASE/'REVIEW_SCHEMA_ADAPTER_READY.json']
    return {'schema_version':'fgr20_saved_revision_gold_first_verifier_input_v1','serialization_variant':'g136202_global_attestation_one_r03_alias',
        'root_id':ROOT,'arm':ARM,'source_binding':source['binding'],'input_work_dir':str(INPUT_WORK),
        'input_bindings':[q.binding(path) for path in paths],'actual_plan_path':str(plan_path),'actual_plan_freeze_path':str(plan_freeze_path),
        'actual_stage_requests_before_recovery':{k:v for k,v in requests.items() if k!='verify'},
        'existing_verifier_call_ids':sorted({x['call_id'] for x in requests['verify']}),
        'allowed_new_call_id':saved.call_id(ROOT,ARM),'saved_gold_call_id':gold_id,
        'saved_gold_receipt_binding':q.binding(INPUT_WORK/'BLIND_GOLD_CALL_revision_01.json'),
        'actual_initial_author_thread_id':threads[0],'actual_revision_author_thread_id':threads[1],'actual_saved_gold_thread_id':threads[2],
        'prior_blind_validation_errors':old_errors,'alias_transformations':changes,'adapted_blind_validation_errors':errors,
        'final_candidate_count':8,'independent_gold_values_unchanged':True,'final_revision_retained_without_initial_fallback':True,
        'global_original_attestations_retained':True,'no_new_gold_call':True,'no_new_plan_or_author_call':True,
        'no_second_revision':True,'semantic_admission_asserted':False}


class AdaptReview(ast.NodeTransformer):
    def __init__(self):self.changes=[]
    def visit_Call(self,node):
        if (isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name)
                and node.func.value.id=='rsa' and node.func.attr=='normalize_blind_aliases'
                and len(node.args)==1 and isinstance(node.args[0],ast.Name) and node.args[0].id=='raw_gold'):
            self.changes.append('saved_raw_gold_exact_global_scope_and_r03_alias_normalizer')
            return ast.copy_location(ast.Call(func=ast.Name(id='normalize_saved_global_blind',ctx=ast.Load()),args=node.args,keywords=[]),node)
        if (isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name)
                and node.func.value.id=='q' and node.func.attr=='binding' and len(node.args)==1
                and ast.unparse(node.args[0])=='rsa.__file__'):
            self.changes.append('honest_new_wrapper_binding_for_runtime_schema_log')
            node.args[0]=ast.Name(id='__file__',ctx=ast.Load())
        return self.generic_visit(node)


def build_review():
    verify_code();namespace=dict(vars(saved));namespace.update(__file__=__file__,normalize_saved_global_blind=normalize_saved_global_blind)
    tree=ast.parse(inspect.getsource(saved.review_batch));adapter=AdaptReview();tree=ast.fix_missing_locations(adapter.visit(tree))
    if adapter.changes!=['saved_raw_gold_exact_global_scope_and_r03_alias_normalizer',
            'honest_new_wrapper_binding_for_runtime_schema_log','honest_new_wrapper_binding_for_runtime_schema_log']:
        raise RuntimeError('Frozen saved-verifier clone no longer has the exact bounded engineering hook locations')
    exec(compile(tree,'<g136202_bound_global_saved_gold_review_clone>','exec'),namespace)
    publish=types.FunctionType(saved.publish_ready.__code__,namespace,saved.publish_ready.__name__,saved.publish_ready.__defaults__,saved.publish_ready.__closure__)
    return namespace['review_batch'],publish,{'changes':adapter.changes,'adapted_review_ast_sha256':hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()}


def scoped_dedicated_driver(argv,cwd):
    names={Path(__file__).name,'unused_failed_lineage_adapter.py'}
    if not argv or 'python' not in Path(argv[0]).name or 'run' not in argv:return None
    script=next((x for x in argv[1:] if x.endswith('.py')),None)
    if not script or Path(script).name not in names:return None
    actual=Path(script)
    if not actual.is_absolute():actual=Path(cwd)/actual
    actual=actual.resolve()
    try:actual.relative_to(BASE)
    except ValueError:return None
    return actual


def assert_idle():
    saved.assert_idle()
    # The frozen helper knows older families; additionally cover both newly
    # authorized dedicated wrappers. Never infer safety solely from call count.
    parents=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            argv=[x.decode(errors='replace') for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
            actual=scoped_dedicated_driver(argv,(proc/'cwd').resolve())
            stat=(proc/'stat').read_text().split()
            if actual and len(stat)>2 and stat[2]!='Z':
                parents.append({'pid':int(proc.name),'argv':argv,'resolved_script':str(actual)})
        except OSError:continue
    if parents:raise RuntimeError('Another dedicated driver parent is live: '+str(parents))


def prepare(source):
    out=saved.work_dir(ROOT,ARM);out.mkdir(parents=True,exist_ok=True);path=out/'RECOVERY_INPUT_FREEZE.json'
    if path.exists():
        frozen=q.read(path)
        if (frozen.get('root_id'),frozen.get('arm'),frozen.get('source_binding'),frozen.get('serialization_variant'))!=(ROOT,ARM,source['binding'],'g136202_global_attestation_one_r03_alias'):
            raise RuntimeError('Existing bounded global-scope recovery freeze identity drift')
        for value in frozen['input_bindings']:q.verify_binding(value)
        saved.verify_budget(saved.actual_requests(ROOT,ARM),ROOT,ARM,frozen['saved_gold_call_id'])
        return frozen
    frozen=inspect_inputs(source);q.publish(path,frozen);return frozen


def run(source,do_run):
    out=saved.work_dir(ROOT,ARM);out.mkdir(parents=True,exist_ok=True)
    with (out/'.continuation.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ready_path=out/'CONTINUATION_FINALIZER_READY.json'
        if ready_path.exists():
            ready=q.read(ready_path)
            if (ready.get('root_id'),ready.get('arm'),ready.get('driver_binding'))!=(ROOT,ARM,q.binding(__file__)):
                raise RuntimeError('Cached global-scope recovery READY identity drift')
            for key,value in ready.items():
                if key.endswith('_binding'):q.verify_binding(value)
            return ready
        if do_run and (BASE/'CONTENT_FREEZE.json').exists():raise RuntimeError('Content already frozen')
        frozen=prepare(source);reviewer,publisher,changes=build_review()
        q.publish(out/'DRIVER_ADAPTATION.json',{'schema_version':'fgr20_global_final_gold_scope_recovery_v1',
            'wrapper_binding':q.binding(__file__),'frozen_saved_verifier_helper_binding':q.binding(saved.__file__),
            'frozen_question_driver_binding':q.binding(q.__file__),'frozen_RAM_adapter_binding':q.binding(rsa.__file__),
            'input_freeze_binding':q.binding(out/'RECOVERY_INPUT_FREEZE.json'),'clone_changes':changes,
            'only_possible_new_call_id':saved.call_id(ROOT,ARM),'no_new_gold_or_author_or_plan_or_revision':True,
            'review_stage':'revision_01','global_original_attestations_retained':True,'semantic_admission_asserted':False})
        if not do_run:return {'root_id':ROOT,'arm':ARM,'preparation_binding':q.binding(out/'RECOVERY_INPUT_FREEZE.json'),'semantic_calls_started':0}
        if q.cc.stop_generation():raise RuntimeError('Generation cutoff before still-unused first verifier')
        assert_idle()
        with (saved.ROOT/'.family_call_start.lock').open('a+') as family:
            fcntl.flock(family,fcntl.LOCK_EX|fcntl.LOCK_NB);assert_idle()
            rsa.install(q.binding(out/'DRIVER_ADAPTATION.json'))
            data=reviewer(ROOT,ARM,q.read(INPUT_WORK/'CANDIDATES_FINAL.json'),source,SELECTION,out,frozen,q.cc)
        saved.verify_budget(saved.actual_requests(ROOT,ARM),ROOT,ARM,frozen['saved_gold_call_id'])
        return publisher(ROOT,ARM,INPUT_WORK,out,frozen,data)


def self_check(source):
    before=saved.actual_requests(ROOT,ARM);frozen=inspect_inputs(source)
    raw=q.read(INPUT_WORK/'BLIND_GOLD_revision_01.json');adapted,changes=normalize_saved_global_blind(raw)
    assert len(changes)==24 and adapted['complete_source_read'] is True and adapted['solver_results_read'] is False
    for name in (Path(__file__).name,'unused_failed_lineage_adapter.py'):
        relative=['/root/miniconda3/bin/python','scripts/'+name,'run']
        absolute=['/root/miniconda3/bin/python',str(BASE/'scripts'/name),'run']
        assert scoped_dedicated_driver(relative,BASE)==(BASE/'scripts'/name).resolve()
        assert scoped_dedicated_driver(absolute,BASE)==(BASE/'scripts'/name).resolve()
        assert scoped_dedicated_driver(relative,Path('/tmp')) is None
    for defect in ('global_false','row_conflict','missing_alias','extra_alias'):
        bad=copy.deepcopy(raw)
        if defect=='global_false':bad['complete_source_read']=False
        elif defect=='row_conflict':bad['reviews'][0]['solver_results_read']=True
        elif defect=='missing_alias':del bad['reviews'][0]['r03_necessity']['removal_reason']
        else:bad['reviews'][0]['r03_necessity']['rationale']='ENGINEERING_CONFLICT_FIXTURE'
        try:normalize_saved_global_blind(bad)
        except RuntimeError:pass
        else:raise AssertionError('Unapproved structure variation was repaired')
    reviewer,publisher,clone=build_review();cache=BASE/'runtime_cache/g136202_global_gold_selfcheck';cache.mkdir(parents=True,exist_ok=True)
    calls=[]
    class Mock:
        def call(self,key,prompt):
            assert key==saved.call_id(ROOT,ARM)
            payload=saved.shape.request_payload(prompt)
            assert payload['complete_source_and_warnings']==q.source_view(source) and len(payload['review_cases'])==8
            assert all(x['saved_independent_gold']['complete_source_read'] is True and x['saved_independent_gold']['solver_results_read'] is False for x in payload['review_cases'])
            calls.append({'call_id':key,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()})
            raise RuntimeError('ENGINEERING_MOCK_G136202_FIRST_VERIFIER_NO_DISPATCH')
    old=(q.normalize_blind,q.normalize_candidates,q.validate_g_prompt)
    try:
        rsa.install()
        with tempfile.TemporaryDirectory(prefix='global_',dir=cache) as d:
            out=Path(d);data=reviewer(ROOT,ARM,q.read(INPUT_WORK/'CANDIDATES_FINAL.json'),source,SELECTION,out,frozen,Mock())
            assert len(calls)==1 and all(x['decision']=='HOLD' for x in data['reviews'])
            assert len(data['slot_failures'])==8 and all(not x['admitted'] for x in data['gate_validations'])
            assert not (out/'CONTENT_VERIFIER_CALL_revision_01.json').exists()
            saved_reference=q.read(out/'SAVED_REVISION_GOLD_REFERENCE.json')
            assert saved_reference['copied_as_new_call'] is False
            mapped=q.read(out/'MECHANICAL_SAVED_REVISION_GOLD_INPUT.json')
            assert mapped['complete_source_read'] is True and mapped['solver_results_read'] is False
    finally:q.normalize_blind,q.normalize_candidates,q.validate_g_prompt=old
    assert saved.actual_requests(ROOT,ARM)==before
    return {'status':'PASS','semantic_calls_started':0,'root_id':ROOT,'arm':ARM,'saved_final_revision_candidates':8,
        'saved_blind_Gold_rows':8,'reversible_changes':24,'old_parser_error_count':16,'new_frozen_parser_errors':[],
        'global_original_attestations_retained':True,'gold_decisions_and_evidence_values_unchanged':True,
        'conflicting_missing_or_extra_aliases_rejected':True,'mock_only_first_verifier_reached':calls,
        'dedicated_driver_relative_absolute_and_outside_cwd_guard_checks':'PASS',
        'mock_no_response_hold_count':8,'new_plan_author_Gold_revision_calls':0,'semantic_admission_asserted':False,'clone':clone}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['inspect','prepare','run','self-check']);args=parser.parse_args()
    q.SOURCE_SELECTION_PATH=SELECTION;q.SOURCE_FREEZE_PATH=BASE/'SOURCE_FREEZE.json';_,sources=q.load_selection(SELECTION);source=sources[ROOT]
    if args.command=='self-check':print(json.dumps(self_check(source),ensure_ascii=False));return 0
    value=inspect_inputs(source) if args.command=='inspect' else run(source,args.command=='run')
    print(json.dumps(value,ensure_ascii=False));return 0


if __name__=='__main__':raise SystemExit(main())
