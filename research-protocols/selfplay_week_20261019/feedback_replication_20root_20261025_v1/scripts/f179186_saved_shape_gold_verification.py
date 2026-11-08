#!/usr/bin/env python3
"""F179186 actual saved shape final Gold: its first verifier only.

Eight exact removal_reason-to-rationale aliases are reversible. Existing final
revision, questions, Gold, read declarations and all semantic values stay
unchanged. No planner, author, revision or Gold call is available here.
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
sys.dont_write_bytecode=True
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent))
import question_driver as q
import review_schema_adapter as rsa
import saved_revision_gold_verification as saved
ROOT='cc-train-episode-179186'
ARM='F'
INPUT_WORK=BASE/'question_candidate_shape_continuation/179186/F'
PLAN_WORK=BASE/'question_plan_continuation/179186/F'
SELECTION=BASE/'SOURCE_SELECTION_FINAL.json'
VARIANT='f179186_shape_final_revision_one_r03_alias_no_global_copy'
EXPECTED_SAVED='d85e11b5db07befe1aab07ebdcc6fa238bc6f36dfff54a44e50c382a704ba5b0'
INSPECTION={'path':str(BASE/'mechanical_audits/F179186_SAVED_SHAPE_REVISION_GOLD_INSPECTION.json'),
 'sha256':'5e563f9283492b65cd232a05b49bed2d03ec75b3b243166df93d999f6ce6be96','bytes':30123}
INDEPENDENT_INSPECTION={'path':str(BASE/'mechanical_audits/F179186_SAVED_SHAPE_REVISION_GOLD_INDEPENDENT_ENGINEERING_INSPECTION.json'),
 'sha256':'e6361b384d59363e08a4c3d114c9c5463c05afe56fa202073736e084d7e5927c','bytes':3546}
SERIAL_STEPS=['shape_77882_G','shape_136202_F','saved_179186_G','saved_136202_G',
 'saved_109164_G','unused_77882_F','unused_failed_109164_F']
PARENT_NAMES={'source_driver.py','source_driver_recovery.py','source_call_recovery.py','question_driver.py',
 'question_driver_recovery.py','question_driver_continued.py','question_plan_schema_continuation.py',
 'question_plan_schema_continuation_v2.py','question_plan_schema_continuation_v3.py',
 'question_candidate_shape_continuation.py','question_unfilled_slot_continuation.py',
 'review_schema_adapter.py','review_schema_adapter_v2.py','unused_failed_lineage_adapter.py',
 'saved_revision_gold_verification.py','g136202_saved_global_gold_verification.py',
 'g109164_saved_global7_gold_verification.py','production_driver.py','resume_authorized_pipeline.py',
 'f179186_saved_shape_gold_verification.py'}


def verify_code():
    saved.verify_code()
    if hashlib.sha256(Path(saved.__file__).read_bytes()).hexdigest()!=EXPECTED_SAVED:
        raise RuntimeError('Frozen saved-first-verifier helper drift')
    for binding in (INSPECTION,INDEPENDENT_INSPECTION):q.verify_binding(binding)
    original=q.read(INSPECTION['path'])
    if (original.get('root_id'),original.get('arm'),original.get('reversible_change_count'),
        original.get('existing_verifier_request_count'))!=(ROOT,ARM,8,0):
        raise RuntimeError('Exact prior engineering inspection identity drift')
    for binding in original['input_bindings']:q.verify_binding(binding)


def normalize_saved_blind(raw):
    if not isinstance(raw,dict) or set(raw)!={'reviews'} or not isinstance(raw['reviews'],list) or len(raw['reviews'])!=8:
        raise RuntimeError('Only the exact saved reviews-only eight-record object is supported')
    value=copy.deepcopy(raw);changes=[];keys=[]
    for index,row in enumerate(value['reviews']):
        if (not isinstance(row,dict) or not isinstance(row.get('question_key'),str) or not row['question_key']
                or row.get('complete_source_read') is not True or row.get('solver_results_read') is not False):
            raise RuntimeError('All saved per-record declarations must already be explicit true/false')
        keys.append(row['question_key']);r03=row.get('r03_necessity')
        if (not isinstance(r03,dict) or set(r03)!={'decision','removal_reason','turn_refs'}
                or r03.get('decision') not in {'PASS','HOLD','FAIL'}
                or not isinstance(r03['removal_reason'],str) or not r03['removal_reason'].strip()
                or not isinstance(r03['turn_refs'],list) or not r03['turn_refs']
                or not all(isinstance(ref,str) and ref for ref in r03['turn_refs'])):
            raise RuntimeError('Only the complete one-key R03 alias is supported; no values can be repaired')
        mapped={('rationale' if key=='removal_reason' else key):copy.deepcopy(item) for key,item in r03.items()}
        row['r03_necessity']=mapped
        changes.append({'review_index':index,'question_key':row['question_key'],'kind':'exact_key_alias',
            'field':'r03_necessity','mapping':{'removal_reason':'rationale'},
            'authored_original_value':copy.deepcopy(r03),'normalized_value':copy.deepcopy(mapped),
            'decision_and_all_values_changed':False})
    if len(set(keys))!=8:raise RuntimeError('Saved anonymous question keys must remain unique')
    inverse=copy.deepcopy(value)
    for change in changes:
        row=inverse['reviews'][change['review_index']]
        restored={('removal_reason' if key=='rationale' else key):copy.deepcopy(item) for key,item in row['r03_necessity'].items()}
        if restored!=change['authored_original_value']:raise RuntimeError('R03 one-key alias inverse failed')
        row['r03_necessity']=restored
    if inverse!=raw:raise RuntimeError('The entire authored saved object is not exactly recoverable')
    return value,changes


def inspect_inputs(source):
    verify_code();outcome=q.read(INPUT_WORK/'OUTCOME.json');ready=q.read(INPUT_WORK/'CONTINUATION_FINALIZER_READY.json')
    if ((outcome.get('root_id'),outcome.get('arm'),outcome.get('source_binding'))!=(ROOT,ARM,source['binding'])
            or outcome.get('status')!='COMPLETED_WITH_HOLES' or outcome.get('max_content_revisions')!=1):
        raise RuntimeError('Actual final shape outcome identity/source/budget drift')
    if ((ready.get('root_id'),ready.get('arm'),ready.get('continuation_kind'),ready.get('final_work_dir'))
            !=(ROOT,ARM,'mechanical_saved_candidate_shape_recovery',str(INPUT_WORK))):
        raise RuntimeError('Actual prior shape continuation READY drift')
    for key,value in ready.items():
        if key.endswith('_binding'):q.verify_binding(value)
    if ready['final_candidates_binding']!=q.binding(INPUT_WORK/'CANDIDATES_FINAL.json'):
        raise RuntimeError('Prior shape READY does not bind exact final candidates')
    initial,initial_id=saved.verify_receipt(PLAN_WORK/'CANDIDATES_DRAFT_CALL.json')
    revision,revision_id=saved.verify_receipt(INPUT_WORK/'CANDIDATES_REVISION_01_CALL.json')
    gold,gold_id=saved.verify_receipt(INPUT_WORK/'BLIND_GOLD_CALL_revision_01.json')
    requests=saved.actual_requests(ROOT,ARM);saved.verify_budget(requests,ROOT,ARM,gold_id)
    if ({row['call_id'] for row in requests['author']}!={initial_id}
            or {row['call_id'] for row in requests['revision']}!={revision_id}
            or gold_id!='q20_gold_f_179186_candidate_shape_01_revision_01'):
        raise RuntimeError('Actual sole author/revision and final Gold identities drift')
    if requests['verify'] and not (saved.work_dir(ROOT,ARM)/'RECOVERY_INPUT_FREEZE.json').exists():
        raise RuntimeError('Existing verifier before recovery freeze blocks first-only continuation')
    plan=q.read(PLAN_WORK/'TASK_PLANS.json');plan_freeze=q.read(PLAN_WORK/'TASK_PLANS_FREEZE.json')
    q.verify_binding(plan_freeze['plans_binding'])
    if (plan_freeze['plans_binding']!=q.binding(PLAN_WORK/'TASK_PLANS.json') or plan_freeze.get('errors')
            or plan.get('root_id')!=ROOT or plan.get('arm')!=ARM
            or {row.get('slot_id') for row in plan.get('plans',[])}!=set(q.SLOTS)):
        raise RuntimeError('Actual original frozen eight-slot plan drift')
    revision_raw=q.read(INPUT_WORK/'CANDIDATES_REVISION_01_RESPONSE.json')
    if revision_raw!=q.read(revision['parsed_response']['path']):raise RuntimeError('Revision differs from actual receipt')
    candidates=q.read(INPUT_WORK/'CANDIDATES_FINAL.json')
    normalized,bad,_=saved.ORIGINAL_NORMALIZE_CANDIDATES(copy.deepcopy(revision_raw),ARM,ROOT,
        initial['backend']['thread_id'],source,plan,q.binding(SELECTION),revision['backend']['thread_id'])
    if (bad or candidates!=normalized or candidates!=q.read(INPUT_WORK/'CANDIDATES_REVISION_01.json')
            or len(candidates)!=8 or [row.get('slot_id') for row in candidates]!=list(q.SLOTS)
            or any(not q.basic_solver_shape(row,source) or row.get('content_revision_count')!=1
                   or row.get('revision_author_thread_id')!=revision['backend']['thread_id'] for row in candidates)):
        raise RuntimeError('Final candidates are not exactly the actual sole final revision')
    reviews=q.read(INPUT_WORK/'CONTENT_REVIEW_FINAL.json').get('reviews',[])
    if len(reviews)!=8 or any(row.get('decision')!='HOLD' for row in reviews):
        raise RuntimeError('Recovery cannot modify any prior semantic PASS/FAIL review')
    raw=q.read(INPUT_WORK/'BLIND_GOLD_revision_01.json')
    if raw!=q.read(gold['parsed_response']['path']):raise RuntimeError('Saved Gold differs from real response')
    payload=saved.shape.request_payload(Path(gold['request']['path']).read_text())
    if payload!={'root_id':ROOT,'complete_source_and_warnings':q.source_view(source),
                 'anonymous_questions':[q.anonymize_candidate(row)[0] for row in candidates]}:
        raise RuntimeError('Saved actual final Gold request source/anonymous candidate drift')
    if [row.get('question_key') for row in raw.get('reviews',[])]!=[q.anonymize_candidate(row)[0]['question_key'] for row in candidates]:
        raise RuntimeError('Saved anonymous Gold row alignment drift')
    old_map,old_errors=saved.ORIGINAL_NORMALIZE_BLIND(copy.deepcopy(raw),candidates,source['source'])
    expected=['blind_r03_invalid:'+row['candidate_id'] for row in candidates]
    if old_errors!=expected or q.read(INPUT_WORK/'BLIND_GOLD_VALIDATION_revision_01.json').get('errors')!=expected:
        raise RuntimeError('Saved final Gold defects exceed exactly the eight one-key aliases')
    adapted,changes=normalize_saved_blind(raw)
    blind_map,errors=saved.ORIGINAL_NORMALIZE_BLIND(copy.deepcopy(adapted),candidates,source['source'])
    if errors or len(changes)!=8 or set(blind_map)!={row['candidate_id'] for row in candidates}:
        raise RuntimeError('Alias-only adapted data fail the frozen Gold/evidence/reference checks')
    if {key:row['independent_gold'] for key,row in old_map.items()}!={key:row['independent_gold'] for key,row in blind_map.items()}:
        raise RuntimeError('Saved independent Gold value changed')
    threads=[receipt['backend']['thread_id'] for receipt in (initial,revision,gold)]
    if len(set(threads))!=3:raise RuntimeError('Actual initial/revision/Gold threads are not independent')
    paths=[Path(row['path']) for row in q.read(INSPECTION['path'])['input_bindings']]
    paths += [Path(INSPECTION['path']),Path(INDEPENDENT_INSPECTION['path']),BASE/'F179186_SAVED_SHAPE_REVISION_GOLD_HANDOFF.json']
    return {'schema_version':'fgr20_saved_revision_gold_first_verifier_input_v1','serialization_variant':VARIANT,
        'root_id':ROOT,'arm':ARM,'source_binding':source['binding'],'input_work_dir':str(INPUT_WORK),
        'input_bindings':[q.binding(path) for path in sorted(set(paths),key=str)],
        'actual_plan_path':str(PLAN_WORK/'TASK_PLANS.json'),'actual_plan_freeze_path':str(PLAN_WORK/'TASK_PLANS_FREEZE.json'),
        'actual_stage_requests_before_recovery':{key:rows for key,rows in requests.items() if key!='verify'},
        'existing_verifier_call_ids':sorted({row['call_id'] for row in requests['verify']}),
        'allowed_new_call_id':saved.call_id(ROOT,ARM),'saved_gold_call_id':gold_id,
        'saved_gold_receipt_binding':q.binding(INPUT_WORK/'BLIND_GOLD_CALL_revision_01.json'),
        'actual_initial_author_thread_id':threads[0],'actual_revision_author_thread_id':threads[1],'actual_saved_gold_thread_id':threads[2],
        'prior_blind_validation_errors':old_errors,'alias_transformations':changes,'adapted_blind_validation_errors':errors,
        'final_candidate_count':8,'independent_gold_values_unchanged':True,'final_revision_retained_without_initial_fallback':True,
        'no_global_attestation_copy':True,'no_new_gold_call':True,'no_new_plan_or_author_call':True,
        'no_second_revision':True,'semantic_admission_asserted':False}


class AdaptReview(ast.NodeTransformer):
    def __init__(self):self.changes=[]
    def visit_Call(self,node):
        if (isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name)
                and node.func.value.id=='rsa' and node.func.attr=='normalize_blind_aliases'
                and len(node.args)==1 and isinstance(node.args[0],ast.Name) and node.args[0].id=='raw_gold'):
            self.changes.append('saved_raw_gold_exact_eight_one_key_alias_normalizer')
            return ast.copy_location(ast.Call(func=ast.Name(id='normalize_saved_blind',ctx=ast.Load()),args=node.args,keywords=[]),node)
        if (isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name)
                and node.func.value.id=='q' and node.func.attr=='binding' and len(node.args)==1
                and ast.unparse(node.args[0])=='rsa.__file__'):
            self.changes.append('honest_new_wrapper_binding_for_runtime_schema_log');node.args[0]=ast.Name(id='__file__',ctx=ast.Load())
        return self.generic_visit(node)


def copy_exact_final_candidates(path):
    raw=(INPUT_WORK/'CANDIDATES_FINAL.json').read_bytes()
    if path.exists():
        if path.read_bytes()!=raw:raise RuntimeError('Exact final candidate byte copy drift')
    else:
        with path.open('xb') as stream:stream.write(raw)
    if path.read_bytes()!=raw:raise RuntimeError('Final candidate bytes changed')


def build_review():
    verify_code();namespace=dict(vars(saved));namespace.update(__file__=__file__,normalize_saved_blind=normalize_saved_blind)
    tree=ast.parse(inspect.getsource(saved.review_batch));adapter=AdaptReview();tree=ast.fix_missing_locations(adapter.visit(tree))
    expected=['saved_raw_gold_exact_eight_one_key_alias_normalizer','honest_new_wrapper_binding_for_runtime_schema_log',
              'honest_new_wrapper_binding_for_runtime_schema_log']
    if adapter.changes!=expected:raise RuntimeError('Frozen saved verifier clone hook locations drift')
    exec(compile(tree,'<f179186_saved_shape_final_gold_review_clone>','exec'),namespace)
    original_publish=types.FunctionType(saved.publish_ready.__code__,namespace,saved.publish_ready.__name__,saved.publish_ready.__defaults__,saved.publish_ready.__closure__)
    def publisher(root,arm,input_work,out,frozen,data):
        if (root,arm,input_work)!=(ROOT,ARM,INPUT_WORK):raise RuntimeError('Dedicated publisher identity drift')
        if len(data['reviews'])!=8:raise RuntimeError('Fresh verifier coverage differs from exact final eight')
        copy_exact_final_candidates(out/'CANDIDATES_FINAL.json')
        original=q.publish
        def publish(path,value):
            path=Path(path)
            if path==out/'CONTINUATION_FINALIZER_READY.json':
                value={**value,'serialization_variant':VARIANT,'original_final_candidate_bytes_unchanged':True}
            return original(path,value)
        q.publish=publish
        try:original_publish(root,arm,input_work,out,frozen,data)
        finally:q.publish=original
        # Return the same immutable object written by the exact READY hook.
        # No semantic data or review verdict is edited.
        return q.read(out/'CONTINUATION_FINALIZER_READY.json')
    return namespace['review_batch'],publisher,{'changes':adapter.changes,
        'adapted_review_ast_sha256':hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),
        'publisher':'frozen publisher after exact original byte copy; exact READY marker added before first immutable write'}


def publish_variant_ready(root,arm,input_work,out,frozen,data):
    # Compatibility alias for engineering integration; direct build_review
    # publishers perform the same exact READY-only immutable first-write hook.
    _,publisher,_=build_review()
    return publisher(root,arm,input_work,out,frozen,data)


def scoped_parent(argv,cwd):
    if not argv or 'python' not in Path(argv[0]).name:return None
    script=next((item for item in argv[1:] if item.endswith('.py')),None)
    if not script or Path(script).name not in PARENT_NAMES:return None
    if 'run' not in argv and Path(script).name not in {'production_driver.py','question_driver.py'}:return None
    resolved=Path(script) if Path(script).is_absolute() else Path(cwd)/script
    resolved=resolved.resolve()
    try:resolved.relative_to(BASE)
    except ValueError:return None
    return resolved


def serial7_handoff():
    directory=BASE/'worker_events/authorized_pipeline_resume';matches=[]
    for path in directory.glob('HANDOFF_*.json'):
        row=q.read(path)
        if (row.get('status')=='SEMANTIC_SEQUENCE_TERMINAL_HANDOFF' and [item.get('step') for item in row.get('steps',[])]==SERIAL_STEPS
                and not row.get('cutoff') and all(item.get('status') in {'REUSED_TERMINAL_OUTPUT','TERMINAL_OUTPUT_OBSERVED'} for item in row['steps'])):
            matches.append(q.binding(path))
    if len(matches)!=1:raise RuntimeError('Exact seven-step terminal handoff is not uniquely present; parent must wait')
    return matches[0]


def decode_proc_cmdline(raw):
    return [part.decode(errors='replace') for part in raw.split(b'\0') if part]


def assert_idle():
    saved.assert_idle();parents=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            argv=decode_proc_cmdline((proc/'cmdline').read_bytes())
            actual=scoped_parent(argv,(proc/'cwd').resolve());stat=(proc/'stat').read_text().split()
            if actual and len(stat)>2 and stat[2]!='Z':parents.append({'pid':int(proc.name),'argv':argv,'resolved_script':str(actual)})
        except (OSError,ValueError):continue
    if parents:raise RuntimeError('Every prior serial pipeline/driver parent must be terminal: '+str(parents))


def prepare(source):
    out=saved.work_dir(ROOT,ARM);out.mkdir(parents=True,exist_ok=True);path=out/'RECOVERY_INPUT_FREEZE.json'
    if path.exists():
        frozen=q.read(path)
        if (frozen.get('root_id'),frozen.get('arm'),frozen.get('source_binding'),frozen.get('serialization_variant'))!=(ROOT,ARM,source['binding'],VARIANT):
            raise RuntimeError('Existing exact saved-shape input freeze drift')
        for binding in frozen['input_bindings']:q.verify_binding(binding)
        saved.verify_budget(saved.actual_requests(ROOT,ARM),ROOT,ARM,frozen['saved_gold_call_id']);return frozen
    frozen=inspect_inputs(source);q.publish(path,frozen);return frozen


class FirstOnlyCalls:
    def __init__(self):self.used=False
    def call(self,key,prompt):
        if self.used or key!=saved.call_id(ROOT,ARM):raise RuntimeError('Only one dedicated first verifier helper invocation is allowed')
        self.used=True
        directory=BASE/'cli_calls'/key
        if (directory/'RESULT.json').exists():
            receipt,actual_id=saved.verify_receipt(directory/'RESULT.json')
            if actual_id!=key or Path(receipt['request']['path']).read_text()!=prompt:raise RuntimeError('Completed verifier prompt/receipt drift')
            return q.read(receipt['parsed_response']['path']),receipt
        if any(directory.glob('attempt_*/REQUEST.txt')) or any(directory.glob('attempt_*/START.json')):
            raise RuntimeError('Previously consumed or interrupted verifier lacks success receipt; no re-dispatch permitted')
        return q.cc.call(key,prompt)


def run(source,do_run):
    out=saved.work_dir(ROOT,ARM);out.mkdir(parents=True,exist_ok=True)
    with (out/'.continuation.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);ready_path=out/'CONTINUATION_FINALIZER_READY.json'
        if ready_path.exists():
            ready=q.read(ready_path)
            if (ready.get('root_id'),ready.get('arm'),ready.get('driver_binding'),ready.get('serialization_variant'))!=(ROOT,ARM,q.binding(__file__),VARIANT):
                raise RuntimeError('Cached dedicated first-verifier identity drift')
            for key,value in ready.items():
                if key.endswith('_binding'):q.verify_binding(value)
            return ready
        if do_run and (BASE/'CONTENT_FREEZE.json').exists():raise RuntimeError('Content already frozen')
        frozen=prepare(source);reviewer,publisher,clone=build_review()
        q.publish(out/'DRIVER_ADAPTATION.json',{'schema_version':'fgr20_shape_final_saved_gold_first_verifier_engineering_v1',
            'wrapper_binding':q.binding(__file__),'frozen_saved_verifier_helper_binding':q.binding(saved.__file__),
            'frozen_question_driver_binding':q.binding(q.__file__),'frozen_RAM_adapter_binding':q.binding(rsa.__file__),
            'input_freeze_binding':q.binding(out/'RECOVERY_INPUT_FREEZE.json'),'clone_changes':clone,
            'only_possible_new_call_id':saved.call_id(ROOT,ARM),'only_saved_gold_eight_single_alias_changes':True,
            'no_global_attestation_copy':True,'no_new_gold_author_plan_or_revision':True,'no_second_verifier_semantic_attempt':True,
            'run_only_after_serial7_terminal_and_all_parents_registry_idle':True,'semantic_admission_asserted':False})
        if not do_run:return {'root_id':ROOT,'arm':ARM,'preparation_binding':q.binding(out/'RECOVERY_INPUT_FREEZE.json'),'semantic_calls_started':0}
        if q.cc.stop_generation():raise RuntimeError('Generation cutoff before first verifier')
        assert_idle();handoff=serial7_handoff();q.publish(out/'SERIAL7_TERMINAL_PREREQUISITE.json',{'handoff_binding':handoff,'all_parents_and_registry_idle_verified':True})
        with (saved.ROOT/'.family_call_start.lock').open('a+') as family:
            fcntl.flock(family,fcntl.LOCK_EX|fcntl.LOCK_NB);assert_idle()
            if q.cc.stop_generation() or (BASE/'CONTENT_FREEZE.json').exists():raise RuntimeError('Content/generation cutoff at call startup')
            data=reviewer(ROOT,ARM,q.read(INPUT_WORK/'CANDIDATES_FINAL.json'),source,SELECTION,out,frozen,FirstOnlyCalls())
        saved.verify_budget(saved.actual_requests(ROOT,ARM),ROOT,ARM,frozen['saved_gold_call_id'])
        return publisher(ROOT,ARM,INPUT_WORK,out,frozen,data)


def self_check(source):
    before=saved.actual_requests(ROOT,ARM);frozen=inspect_inputs(source);raw=q.read(INPUT_WORK/'BLIND_GOLD_revision_01.json')
    mapped,changes=normalize_saved_blind(raw);assert len(changes)==8 and set(mapped)=={'reviews'}
    for defect in ('global','attestation_missing','attestation_conflict','missing_alias','mixed_alias','extra_r03','missing_record','duplicate_key'):
        bad=copy.deepcopy(raw)
        if defect=='global':bad['complete_source_read']=True
        elif defect=='attestation_missing':del bad['reviews'][0]['solver_results_read']
        elif defect=='attestation_conflict':bad['reviews'][0]['complete_source_read']=False
        elif defect=='missing_alias':del bad['reviews'][0]['r03_necessity']['removal_reason']
        elif defect=='mixed_alias':bad['reviews'][0]['r03_necessity']['rationale']='ENGINEERING_CONFLICT_FIXTURE'
        elif defect=='extra_r03':bad['reviews'][0]['r03_necessity']['extra']=True
        elif defect=='missing_record':bad['reviews'].pop()
        else:bad['reviews'][0]['question_key']=bad['reviews'][1]['question_key']
        try:normalize_saved_blind(bad)
        except RuntimeError:pass
        else:raise AssertionError('Unapproved missing/conflicting schema was repaired: '+defect)
    for name in ('resume_authorized_pipeline.py','review_schema_adapter_v2.py',Path(__file__).name):
        encoded=b'\0'.join(item.encode() for item in ['/root/miniconda3/bin/python','scripts/'+name,'run'])+b'\0'
        assert decode_proc_cmdline(encoded)==['/root/miniconda3/bin/python','scripts/'+name,'run']
        assert scoped_parent(decode_proc_cmdline(encoded),BASE)==(BASE/'scripts'/name).resolve()
        assert scoped_parent(['/root/miniconda3/bin/python','scripts/'+name,'run'],BASE)==(BASE/'scripts'/name).resolve()
        assert scoped_parent(['/root/miniconda3/bin/python',str(BASE/'scripts'/name),'run'],BASE)==(BASE/'scripts'/name).resolve()
        assert scoped_parent(['/root/miniconda3/bin/python','scripts/'+name,'run'],Path('/tmp')) is None
    reviewer,publisher,clone=build_review();calls=[];candidates=q.read(INPUT_WORK/'CANDIDATES_FINAL.json')
    class Mock:
        def call(self,key,prompt):
            assert key==saved.call_id(ROOT,ARM);payload=saved.shape.request_payload(prompt)
            assert payload['root_id']==ROOT and payload['complete_source_and_warnings']==q.source_view(source) and len(payload['review_cases'])==8
            assert [row['question_key'] for row in payload['review_cases']]==[q.anonymize_candidate(c)[0]['question_key'] for c in candidates]
            assert all('feedback_target' not in row and 'arm' not in row and 'candidate_id' not in row for row in payload['review_cases'])
            calls.append({'call_id':key,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()})
            raise RuntimeError('ENGINEERING_MOCK_F179186_FIRST_SAVED_VERIFIER_NO_DISPATCH')
    cache=BASE/'runtime_cache/f179186_saved_shape_gold_selfcheck';cache.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='saved_shape_',dir=cache) as directory:
        out=Path(directory);q.publish(out/'RECOVERY_INPUT_FREEZE.json',frozen)
        data=reviewer(ROOT,ARM,candidates,source,SELECTION,out,frozen,Mock())
        assert len(calls)==1 and len(data['reviews'])==8 and all(row['decision']=='HOLD' for row in data['reviews'])
        assert len(data['slot_failures'])==8 and all(not row['admitted'] for row in data['gate_validations'])
        assert not (out/'CONTENT_VERIFIER_CALL_revision_01.json').exists()
        saved_reference=q.read(out/'SAVED_REVISION_GOLD_REFERENCE.json');assert saved_reference['copied_as_new_call'] is False
        assert q.read(out/'MECHANICAL_SAVED_REVISION_GOLD_INPUT.json')==mapped
        normalization=q.read(out/'MECHANICAL_SAVED_REVISION_GOLD_NORMALIZATION.json')
        assert normalization['adapter_binding']==q.binding(__file__) and len(normalization['changes'])==8
        original_publish=q.publish
        ready=publisher(ROOT,ARM,INPUT_WORK,out,frozen,data)
        assert q.publish is original_publish and ready==q.read(out/'CONTINUATION_FINALIZER_READY.json')
        assert ready['serialization_variant']==VARIANT and ready['driver_binding']==q.binding(__file__) and ready['adapter_binding']==q.binding(rsa.__file__)
        assert (out/'CANDIDATES_FINAL.json').read_bytes()==(INPUT_WORK/'CANDIDATES_FINAL.json').read_bytes()
        assert q.read(out/'OUTCOME.json')['status']=='COMPLETED_WITH_HOLES'
    assert saved.actual_requests(ROOT,ARM)==before
    return {'status':'PASS','semantic_calls_started':0,'root_id':ROOT,'arm':ARM,'saved_final_revision_candidates':8,
        'saved_blind_Gold_rows':8,'reversible_single_key_alias_changes':8,'global_attestation_copies':0,
        'old_parser_error_count':8,'new_frozen_parser_errors':[],'eight_negative_schema_fixtures_rejected':True,
        'complete_saved_source_and_actual_final_anonymous_candidates_bound':True,'Gold_evidence_decisions_and_values_unchanged':True,
        'final_candidate_bytes_unchanged':True,'mock_only_first_verifier_reached':calls,'mock_no_response_HOLD_count':8,
        'serial7_and_v2_relative_absolute_outside_path_guards_checked':True,'proc_cmdline_NUL_separation_checked':True,
        'direct_publisher_exact_immutable_READY_return_and_publish_restore_checked':True,'new_plan_author_Gold_revision_calls':0,
        'semantic_admission_asserted':False,'clone':clone}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['inspect','prepare','run','self-check']);args=parser.parse_args()
    q.SOURCE_SELECTION_PATH=SELECTION;q.SOURCE_FREEZE_PATH=BASE/'SOURCE_FREEZE.json';_,sources=q.load_selection(SELECTION);source=sources[ROOT]
    if args.command=='self-check':print(json.dumps(self_check(source),ensure_ascii=False));return 0
    value=inspect_inputs(source) if args.command=='inspect' else run(source,args.command=='run')
    print(json.dumps(value,ensure_ascii=False));return 0


if __name__=='__main__':raise SystemExit(main())
