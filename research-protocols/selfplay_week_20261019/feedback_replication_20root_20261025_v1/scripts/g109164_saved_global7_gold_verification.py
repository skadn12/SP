#!/usr/bin/env python3
"""G109164: seven actual saved revision Gold records, first verifier only.

ORDER2 is an original author SLOT_FAILURE notice archived untouched with its
prior HOLD review; only the seven actual questions enter final candidates.
MC4's saved R03 decision FAIL is preserved for the fresh verifier. No statement,
Gold, semantic decision, candidate, author or revision is generated here.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
sys.dont_write_bytecode=True
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent))
import question_driver as q
import review_schema_adapter as rsa
import saved_revision_gold_verification as saved
import g136202_saved_global_gold_verification as global8
ROOT='cc-train-episode-109164'
ARM='G'
SLOTS=['MC2','MC3','MC4','MC5','MC6','ORDER1','ORDER3']
INPUT_WORK=BASE/'question_plan_continuation/109164/G'
SELECTION=BASE/'SOURCE_SELECTION_FINAL.json'
VARIANT='g109164_global_attestation_seven_saved_revision_Gold_ORDER2_failure_retained'
EXPECTED_GLOBAL8='d83d57650c7b1186219880cf703d31fc807d3e6fe2ca77aa0ba2748739e5d061'


def verify_code():
    global8.verify_code()
    if hashlib.sha256(Path(global8.__file__).read_bytes()).hexdigest()!=EXPECTED_GLOBAL8:
        raise RuntimeError('Frozen global-attestation engineering helper drift')


def normalize_saved_seven(raw):
    if (not isinstance(raw,dict) or set(raw)!={'reviews','complete_source_read','solver_results_read'}
            or raw.get('complete_source_read') is not True or raw.get('solver_results_read') is not False
            or not isinstance(raw.get('reviews'),list) or len(raw['reviews'])!=7):
        raise RuntimeError('Only this complete seven-record saved call-global schema is allowed')
    for row in raw['reviews']:
        if (not isinstance(row,dict) or any(k in row for k in ('complete_source_read','solver_results_read'))
                or not isinstance(row.get('r03_necessity'),dict)
                or set(row['r03_necessity'])!={'decision','removal_reason','r03_turn_refs'}):
            raise RuntimeError('Existing/conflicting per-record declaration or R03 shape cannot be repaired')
    value,aliases=rsa.normalize_blind_aliases(raw)
    if len(aliases)!=7:raise RuntimeError('All seven records require the exact complete frozen R03 key mapping')
    changes=copy.deepcopy(aliases)
    for change in changes:change['kind']='exact_frozen_r03_key_alias'
    for index,row in enumerate(value['reviews']):
        for key in ('complete_source_read','solver_results_read'):
            row[key]=copy.deepcopy(raw[key])
            changes.append({'review_index':index,'question_key':row.get('question_key'),
                'kind':'call_scope_attestation_copy','field':key,'original_field_absent':True,
                'source_global_path':[key],'source_global_value':raw[key],'normalized_value':row[key],
                'global_original_retained':True})
    reversed_value=copy.deepcopy(value)
    for change in reversed(changes):
        row=reversed_value['reviews'][change['review_index']]
        if change['kind']=='call_scope_attestation_copy':
            if row[change['field']] is not change['source_global_value']:raise RuntimeError('Read declaration value changed')
            del row[change['field']]
        else:
            restored={({'rationale':'removal_reason','turn_refs':'r03_turn_refs'}).get(k,k):copy.deepcopy(v)
                      for k,v in row['r03_necessity'].items()}
            if restored!=change['authored_original_value']:raise RuntimeError('R03 alias inverse is not exact')
            row['r03_necessity']=restored
    if reversed_value!=raw:raise RuntimeError('Full saved call object is not exactly recoverable')
    if value['complete_source_read'] is not True or value['solver_results_read'] is not False:
        raise RuntimeError('Original call-global declarations changed')
    return value,changes


def verify_budget(requests,saved_gold_id=None):
    for stage,n in [('plan',1),('author',1),('revision',1),('gold',2)]:
        ids={x['call_id'] for x in requests[stage]}
        if len(ids)!=n:raise RuntimeError(f'Wrong actual consumed {stage} call count: {ids}')
    for stage in ('plan','author'):
        if any(row['slot_ids']!=list(q.SLOTS) for row in requests[stage]):
            raise RuntimeError('Original plan/initial author must retain the actual eight-slot assignment')
    if any(row['slot_ids']!=SLOTS for row in requests['revision']):
        raise RuntimeError('The only actual revision must cover exactly the existing seven slots')
    if saved_gold_id and saved_gold_id not in {x['call_id'] for x in requests['gold']}:
        raise RuntimeError('Saved final Gold receipt is not its actual consumed call')
    if {x['call_id'] for x in requests['verify']}-{saved.call_id(ROOT,ARM)}:
        raise RuntimeError('A content verifier was already requested for this root/arm')


def inspect_inputs(source):
    verify_code()
    required=['OUTCOME.json','CONTINUATION_FINALIZER_READY.json','TASK_PLANS.json','TASK_PLANS_FREEZE.json',
        'CANDIDATES_DRAFT_RESPONSE.json','CANDIDATES_DRAFT_CALL.json','CANDIDATES_DRAFT.json',
        'CANDIDATES_REVISION_01_RESPONSE.json','CANDIDATES_REVISION_01_CALL.json','CANDIDATES_REVISION_01.json',
        'CANDIDATES_FINAL.json','CONTENT_REVIEW_FINAL.json','SLOT_FAILURES.json','BLIND_GOLD_revision_01.json',
        'BLIND_GOLD_CALL_revision_01.json','BLIND_GOLD_VALIDATION_revision_01.json']
    if any(not (INPUT_WORK/x).is_file() for x in required):raise RuntimeError('Actual terminal saved-revision inputs incomplete')
    outcome=q.read(INPUT_WORK/'OUTCOME.json');old_ready=q.read(INPUT_WORK/'CONTINUATION_FINALIZER_READY.json')
    if ((outcome.get('root_id'),outcome.get('arm'),outcome.get('source_binding'))!=(ROOT,ARM,source['binding'])
            or outcome.get('status')!='COMPLETED_WITH_HOLES' or outcome.get('max_content_revisions')!=1):
        raise RuntimeError('Original terminal identity/source or consumed revision budget drift')
    if (old_ready.get('root_id'),old_ready.get('arm'),old_ready.get('final_work_dir'))!=(ROOT,ARM,str(INPUT_WORK)):
        raise RuntimeError('Original finalizer READY identity drift')
    for key in ('final_outcome_binding','driver_outcome_binding','final_candidates_binding','final_review_binding','slot_failures_binding'):
        q.verify_binding(old_ready[key])
    if old_ready['final_candidates_binding']!=q.binding(INPUT_WORK/'CANDIDATES_FINAL.json'):
        raise RuntimeError('Original finalizer does not bind actual final candidate bytes')
    old_reviews=q.read(INPUT_WORK/'CONTENT_REVIEW_FINAL.json').get('reviews',[])
    if len(old_reviews)!=8 or any(x.get('decision')=='PASS' for x in old_reviews):
        raise RuntimeError('This continuation cannot rereview any previously passing candidate')
    initial,initial_id=saved.verify_receipt(INPUT_WORK/'CANDIDATES_DRAFT_CALL.json')
    revision,revision_id=saved.verify_receipt(INPUT_WORK/'CANDIDATES_REVISION_01_CALL.json')
    gold,gold_id=saved.verify_receipt(INPUT_WORK/'BLIND_GOLD_CALL_revision_01.json')
    requests=saved.actual_requests(ROOT,ARM);verify_budget(requests,gold_id)
    if ({r['call_id'] for r in requests['author']}!={initial_id} or {r['call_id'] for r in requests['revision']}!={revision_id}):
        raise RuntimeError('Actual consumed author/revision call identity mismatch')
    if requests['verify'] and not (saved.work_dir(ROOT,ARM)/'RECOVERY_INPUT_FREEZE.json').exists():
        raise RuntimeError('Any first verifier request predating this recovery freeze blocks continuation')
    plan_path=INPUT_WORK/'TASK_PLANS.json';plan=q.read(plan_path);plan_freeze=q.read(INPUT_WORK/'TASK_PLANS_FREEZE.json')
    q.verify_binding(plan_freeze['plans_binding'])
    if plan_freeze['plans_binding']!=q.binding(plan_path) or plan_freeze.get('errors'):
        raise RuntimeError('Actual preauthoring eight-slot plan is invalid or not frozen')
    _,plan_errors=q.validate_plans(plan,ARM,ROOT,source['source'],None)
    if plan_errors or {p.get('slot_id') for p in plan['plans']}!=set(q.SLOTS):raise RuntimeError('Frozen plan identity/slots invalid')
    initial_raw=q.read(INPUT_WORK/'CANDIDATES_DRAFT_RESPONSE.json')
    if initial_raw!=q.read(initial['parsed_response']['path']):raise RuntimeError('Initial author raw object differs from actual response')
    initial_candidates,initial_bad,_=saved.ORIGINAL_NORMALIZE_CANDIDATES(initial_raw,ARM,ROOT,initial['backend']['thread_id'],source,plan,q.binding(SELECTION))
    if initial_bad or initial_candidates!=q.read(INPUT_WORK/'CANDIDATES_DRAFT.json'):
        raise RuntimeError('Initial candidate/failure-notice lineage differs from actual authored response')
    revision_raw=q.read(INPUT_WORK/'CANDIDATES_REVISION_01_RESPONSE.json')
    if revision_raw!=q.read(revision['parsed_response']['path']):raise RuntimeError('Saved final revision object differs from actual response')
    revised,bad,_=saved.ORIGINAL_NORMALIZE_CANDIDATES(revision_raw,ARM,ROOT,initial['backend']['thread_id'],source,plan,q.binding(SELECTION),revision['backend']['thread_id'])
    if bad!=[{'slot_id':'ORDER2','reason':'candidate_not_returned_by_author'}] or revised!=q.read(INPUT_WORK/'CANDIDATES_REVISION_01.json'):
        raise RuntimeError('Final revision must be exactly seven authored slots with only unrequested ORDER2 absent')
    candidates=q.read(INPUT_WORK/'CANDIDATES_FINAL.json');by_slot={c['slot_id']:c for c in candidates}
    covered=[by_slot[s] for s in SLOTS]
    if len(candidates)!=8 or set(by_slot)!=set(q.SLOTS) or revised!=covered:
        raise RuntimeError('Final covered candidates are not exactly the seven real final revisions')
    if any(not q.basic_solver_shape(c,source) or c.get('content_revision_count')!=1 or c.get('revision_author_thread_id')!=revision['backend']['thread_id'] for c in covered):
        raise RuntimeError('Covered final candidate shape or actual revision lineage incomplete')
    failure=by_slot['ORDER2'];initial_failure=next(c for c in initial_candidates if c['slot_id']=='ORDER2')
    if (failure!=initial_failure or failure.get('candidate_status')!='SLOT_FAILURE'
            or q.basic_solver_shape(failure,source) or any(k in failure for k in ('question','items','gold'))):
        raise RuntimeError('Uncovered ORDER2 must remain exactly its original authored failure notice without a question or Gold')
    old_order2=next((x for x in old_reviews if x.get('candidate_id')==failure['candidate_id']),None)
    if not old_order2 or old_order2.get('decision')!='HOLD':raise RuntimeError('Original uncovered ORDER2 review must remain HOLD')
    raw=q.read(INPUT_WORK/'BLIND_GOLD_revision_01.json')
    if raw!=q.read(gold['parsed_response']['path']):raise RuntimeError('Saved blind final response differs from actual receipt')
    payload=saved.shape.request_payload(Path(gold['request']['path']).read_text())
    if (payload.get('root_id')!=ROOT or payload.get('complete_source_and_warnings')!=q.source_view(source)
            or payload.get('anonymous_questions')!=[q.anonymize_candidate(c)[0] for c in covered]):
        raise RuntimeError('Saved blind actual request must exactly embed complete source and only the seven covered final candidates')
    keys=[q.anonymize_candidate(c)[0]['question_key'] for c in covered]
    if [x.get('question_key') for x in raw.get('reviews',[])]!=keys:
        raise RuntimeError('Saved seven review rows do not align one-to-one with the actual anonymous request')
    old_map,old_errors=saved.ORIGINAL_NORMALIZE_BLIND(copy.deepcopy(raw),covered,source['source'])
    expected=[]
    for c in covered:expected.extend(['blind_r03_invalid:'+c['candidate_id'],'blind_read_attestation_invalid:'+c['candidate_id']])
    if old_errors!=expected or q.read(INPUT_WORK/'BLIND_GOLD_VALIDATION_revision_01.json').get('errors')!=expected:
        raise RuntimeError('Blind defects exceed only the known seven-record alias/global-scope variant')
    adapted,changes=normalize_saved_seven(raw);blind_map,errors=saved.ORIGINAL_NORMALIZE_BLIND(copy.deepcopy(adapted),covered,source['source'])
    if errors or len(changes)!=21 or set(blind_map)!={c['candidate_id'] for c in covered}:
        raise RuntimeError('Saved seven Gold/edge/evidence/full-read data fail frozen mechanical checks after structural mapping')
    if {k:v['independent_gold'] for k,v in old_map.items()}!={k:v['independent_gold'] for k,v in blind_map.items()}:
        raise RuntimeError('A saved independent Gold value changed')
    mc4=next(c for c in covered if c['slot_id']=='MC4')
    if blind_map[mc4['candidate_id']]['r03_necessity'].get('decision')!='FAIL':
        raise RuntimeError('The actual saved MC4 R03 FAIL must remain intact, not silently rehabilitated')
    threads=[x['backend']['thread_id'] for x in (initial,revision,gold)]
    if len(set(threads))!=3:raise RuntimeError('Actual author, revision and saved Gold threads are not independent')
    paths=[INPUT_WORK/x for x in required]+[SELECTION,q.SOURCE_FREEZE_PATH,BASE/'REVIEW_SCHEMA_ADAPTER_READY.json',BASE/'G136202_SAVED_GLOBAL_GOLD_VERIFICATION_READY.json']
    return {'schema_version':'fgr20_saved_revision_gold_first_verifier_input_v1','serialization_variant':VARIANT,
        'root_id':ROOT,'arm':ARM,'source_binding':source['binding'],'input_work_dir':str(INPUT_WORK),
        'input_bindings':[q.binding(x) for x in paths],'actual_plan_path':str(plan_path),'actual_plan_freeze_path':str(INPUT_WORK/'TASK_PLANS_FREEZE.json'),
        'actual_stage_requests_before_recovery':{k:v for k,v in requests.items() if k!='verify'},
        'existing_verifier_call_ids':sorted({r['call_id'] for r in requests['verify']}),'allowed_new_call_id':saved.call_id(ROOT,ARM),
        'saved_gold_call_id':gold_id,'saved_gold_receipt_binding':q.binding(INPUT_WORK/'BLIND_GOLD_CALL_revision_01.json'),
        'actual_initial_author_thread_id':threads[0],'actual_revision_author_thread_id':threads[1],'actual_saved_gold_thread_id':threads[2],
        'prior_blind_validation_errors':old_errors,'alias_transformations':changes,'adapted_blind_validation_errors':errors,
        'original_final_record_count':8,'final_candidate_count':7,'covered_final_revision_candidate_count':7,
        'unadmitted_original_failure_notice_count':1,'covered_slot_ids':SLOTS,'uncovered_slot_ids':['ORDER2'],
        'uncovered_order2_original_failure_notice_and_HOLD_preserved':True,'saved_MC4_r03_decision_FAIL_preserved':True,
        'independent_gold_values_unchanged':True,'final_revision_retained_without_initial_fallback':True,
        'global_original_attestations_retained':True,'no_new_gold_call':True,'no_new_plan_or_author_call':True,
        'no_second_revision':True,'semantic_admission_asserted':False}


def build_review():
    verify_code();reviewer,_,adaptation=global8.build_review()
    namespace=reviewer.__globals__
    namespace['normalize_saved_global_blind']=normalize_saved_seven
    namespace['__file__']=__file__
    return reviewer,{'frozen_global8_review_clone':adaptation,'replacement_pure_function':'normalize_saved_seven','honest_log_wrapper_binding':q.binding(__file__)}


def assert_idle():
    global8.assert_idle()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            argv=[x.decode(errors='replace') for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
            if not argv or 'python' not in Path(argv[0]).name or 'run' not in argv:continue
            script=next((x for x in argv[1:] if x.endswith('.py')),None)
            if not script or Path(script).name!=Path(__file__).name:continue
            actual=Path(script)
            if not actual.is_absolute():actual=(proc/'cwd').resolve()/actual
            actual.resolve().relative_to(BASE)
            stat=(proc/'stat').read_text().split()
            if len(stat)>2 and stat[2]!='Z':raise RuntimeError('Another G109 dedicated verifier parent is running')
        except (OSError,ValueError):continue


def publish_original_bytes(path,name):
    original=(INPUT_WORK/name).read_bytes()
    if path.exists():
        if path.read_bytes()!=original:raise RuntimeError('Original archive byte copy drift')
    else:
        with path.open('xb') as handle:handle.write(original)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=hashlib.sha256(original).hexdigest():
        raise RuntimeError('Original archive bytes were changed')


def publish_ready(out,frozen,data):
    all_candidates=q.read(INPUT_WORK/'CANDIDATES_FINAL.json');old_reviews=q.read(INPUT_WORK/'CONTENT_REVIEW_FINAL.json')['reviews']
    failure=next(c for c in all_candidates if c['slot_id']=='ORDER2');old_order2=next(r for r in old_reviews if r['candidate_id']==failure['candidate_id'])
    fresh={r['candidate_id']:r for r in data['reviews']}
    if set(fresh)!={c['candidate_id'] for c in all_candidates if c['slot_id'] in SLOTS}:raise RuntimeError('Fresh verifier result altered covered slot scope')
    covered=[c for c in all_candidates if c['slot_id'] in SLOTS]
    # The unfilled ORDER2 notice is provenance, not a candidate. Keep it in a
    # separately bound archive so downstream assembly cannot count its shell.
    archive_failures=[x for x in q.read(INPUT_WORK/'SLOT_FAILURES.json') if x['slot_id']=='ORDER2']
    publish_original_bytes(out/'ORIGINAL_FINAL_CANDIDATES_ARCHIVE.json','CANDIDATES_FINAL.json')
    publish_original_bytes(out/'ORIGINAL_CONTENT_REVIEW_FINAL_ARCHIVE.json','CONTENT_REVIEW_FINAL.json')
    q.publish(out/'UNADMITTED_FAILURE_ARCHIVE.json',{'schema_version':'fgr20_unadmitted_failure_archive_v1',
        'root_id':ROOT,'arm':ARM,'slot_id':'ORDER2','candidate':copy.deepcopy(failure),
        'prior_review':copy.deepcopy(old_order2),'failure_records':archive_failures,
        'admitted_as_candidate':False,'semantic_question_present':False,
        'preserved_values_equal_original':True,
        'original_candidate_file_binding':q.binding(INPUT_WORK/'CANDIDATES_FINAL.json'),
        'original_review_file_binding':q.binding(INPUT_WORK/'CONTENT_REVIEW_FINAL.json'),
        'original_candidates_archive_binding':q.binding(out/'ORIGINAL_FINAL_CANDIDATES_ARCHIVE.json'),
        'original_reviews_archive_binding':q.binding(out/'ORIGINAL_CONTENT_REVIEW_FINAL_ARCHIVE.json')})
    reviews=[fresh[c['candidate_id']] for c in covered]
    original_order2_failures=[x for x in q.read(INPUT_WORK/'SLOT_FAILURES.json') if x['slot_id']=='ORDER2']
    if not original_order2_failures:raise RuntimeError('Uncovered ORDER2 original failure records disappeared')
    failures=original_order2_failures+data['slot_failures']
    q.publish(out/'CANDIDATES_FINAL.json',covered)
    q.publish(out/'CONTENT_REVIEW_FINAL.json',{'reviews':reviews,'mechanical_audits':data['mechanical_audits'],
        'all_reviews_ai_family_not_human':True,'review_stage':'revision_01','initial_reviews_retained_at':str(INPUT_WORK),
        'covered_slot_ids':SLOTS,'uncovered_slots_archived_not_admitted':['ORDER2']})
    q.publish(out/'FINAL_REVIEW_GATE.json',{'results':data['gate_validations'],'covered_slot_ids':SLOTS,
        'uncovered_slots':[{'slot_id':'ORDER2','status':'ORIGINAL_AUTHOR_SLOT_FAILURE_RETAINED','admitted':False}],
        'semantic_admission_basis':'fresh independent verifier on only the seven actual saved blind questions plus frozen mechanical gate'})
    q.publish(out/'SLOT_FAILURES.json',failures)
    outcome={'schema_version':'fgr20_question_root_arm_v1','root_id':ROOT,'arm':ARM,'source_binding':frozen['source_binding'],
        'status':'COMPLETED_WITH_HOLES','candidate_ids':[c['candidate_id'] for c in covered],
        'candidate_bindings':[{'candidate_id':c['candidate_id'],'candidate_sha256':q.gate.fingerprint(c)} for c in covered],
        'slot_failures':failures,'calls':[data['verification_call']] if data['verification_call'] else [],'max_content_revisions':1,
        'review_stage':'revision_01','continuation_kind':saved.KIND,'serialization_variant':VARIANT,
        'prior_outcome_binding':q.binding(INPUT_WORK/'OUTCOME.json'),'prior_slot_failures_binding':q.binding(INPUT_WORK/'SLOT_FAILURES.json'),
        'final_candidates_binding':q.binding(out/'CANDIDATES_FINAL.json'),'review_binding':q.binding(out/'CONTENT_REVIEW_FINAL.json'),
        'slot_failures_binding':q.binding(out/'SLOT_FAILURES.json'),'plan_freeze_binding':q.binding(frozen['actual_plan_freeze_path']),
        'saved_blind_gold_receipt_binding':frozen['saved_gold_receipt_binding'],'covered_slot_ids':SLOTS,'uncovered_slot_ids':['ORDER2'],
        'no_new_gold_call':True,'no_new_plan_or_author_call':True,'no_second_revision':True,'final_revision_retained_without_initial_fallback':True,
        'uncovered_ORDER2_original_failure_notice_and_review_preserved':True,
        'unadmitted_failure_archive_binding':q.binding(out/'UNADMITTED_FAILURE_ARCHIVE.json'),
        'original_final_candidates_archive_binding':q.binding(out/'ORIGINAL_FINAL_CANDIDATES_ARCHIVE.json'),
        'uncovered_order2_original_review_binding':q.binding(INPUT_WORK/'CONTENT_REVIEW_FINAL.json'),
        'final_candidates_exclude_unadmitted_failure_shell':True}
    q.publish(out/'OUTCOME.json',outcome);q.publish(out/'CONTINUATION_INTEGRATION_OUTCOME.json',outcome)
    ready={'schema_version':'fgr20_question_continuation_finalizer_ready_v1','root_id':ROOT,'arm':ARM,
        'continuation_kind':saved.KIND,'serialization_variant':VARIANT,'review_stage':'revision_01','original_work_dir':str(INPUT_WORK),
        'initial_work_dir':str(INPUT_WORK),'main_original_work_dir':str(BASE/'question_authoring/G/roots/109164'),'final_work_dir':str(out),
        'actual_plan_path':frozen['actual_plan_path'],'actual_plan_freeze_path':frozen['actual_plan_freeze_path'],
        'original_outcome_binding':q.binding(INPUT_WORK/'OUTCOME.json'),'supersedes_ready_binding':q.binding(INPUT_WORK/'CONTINUATION_FINALIZER_READY.json'),
        'final_outcome_binding':q.binding(out/'CONTINUATION_INTEGRATION_OUTCOME.json'),'driver_outcome_binding':q.binding(out/'OUTCOME.json'),
        'final_candidates_binding':q.binding(out/'CANDIDATES_FINAL.json'),'final_review_binding':q.binding(out/'CONTENT_REVIEW_FINAL.json'),
        'slot_failures_binding':q.binding(out/'SLOT_FAILURES.json'),'input_freeze_binding':q.binding(out/'RECOVERY_INPUT_FREEZE.json'),
        'unadmitted_failure_archive_binding':q.binding(out/'UNADMITTED_FAILURE_ARCHIVE.json'),
        'original_final_candidates_archive_binding':q.binding(out/'ORIGINAL_FINAL_CANDIDATES_ARCHIVE.json'),
        'uncovered_order2_original_review_binding':q.binding(INPUT_WORK/'CONTENT_REVIEW_FINAL.json'),
        'saved_gold_receipt_reference_binding':q.binding(out/'SAVED_REVISION_GOLD_REFERENCE.json'),
        'normalization_binding':q.binding(out/'MECHANICAL_SAVED_REVISION_GOLD_NORMALIZATION.json'),'gate_validation_binding':q.binding(out/'FINAL_REVIEW_GATE.json'),
        'driver_binding':q.binding(__file__),'adapter_binding':q.binding(rsa.__file__),'covered_slot_ids':SLOTS,'uncovered_slot_ids':['ORDER2'],
        'no_new_gold_call':True,'no_new_plan_or_author_call':True,'no_second_revision':True,
        'initial_candidate_and_reviews_not_overwritten':True,'final_revision_retained_without_initial_fallback':True,
        'uncovered_ORDER2_original_failure_notice_and_review_preserved':True,
        'final_candidates_exclude_unadmitted_failure_shell':True,'original_final_candidate_bytes_unchanged':True}
    q.publish(out/'CONTINUATION_FINALIZER_READY.json',ready);return ready


def prepare(source):
    out=saved.work_dir(ROOT,ARM);out.mkdir(parents=True,exist_ok=True);path=out/'RECOVERY_INPUT_FREEZE.json'
    if path.exists():
        frozen=q.read(path)
        if (frozen.get('root_id'),frozen.get('arm'),frozen.get('source_binding'),frozen.get('serialization_variant'))!=(ROOT,ARM,source['binding'],VARIANT):raise RuntimeError('Existing seven-slot freeze identity drift')
        for value in frozen['input_bindings']:q.verify_binding(value)
        verify_budget(saved.actual_requests(ROOT,ARM),frozen['saved_gold_call_id']);return frozen
    frozen=inspect_inputs(source);q.publish(path,frozen);return frozen


def run(source,do_run):
    out=saved.work_dir(ROOT,ARM);out.mkdir(parents=True,exist_ok=True)
    with (out/'.continuation.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cached=out/'CONTINUATION_FINALIZER_READY.json'
        if cached.exists():
            ready=q.read(cached)
            if (ready.get('root_id'),ready.get('arm'),ready.get('driver_binding'))!=(ROOT,ARM,q.binding(__file__)):raise RuntimeError('Cached seven-slot verifier identity drift')
            for key,value in ready.items():
                if key.endswith('_binding'):q.verify_binding(value)
            return ready
        if do_run and (BASE/'CONTENT_FREEZE.json').exists():raise RuntimeError('Content already frozen')
        frozen=prepare(source);reviewer,clone=build_review()
        partition=inspect_inputs(source)
        q.publish(out/'FINAL7_PARTITION_ADAPTATION.json',{'schema_version':'fgr20_final_seven_question_partition_v1',
            'wrapper_binding':q.binding(__file__),'input_freeze_binding':q.binding(out/'RECOVERY_INPUT_FREEZE.json'),
            'original_final_record_count':8,'final_real_candidate_count':7,'covered_slot_ids':SLOTS,'uncovered_slot_ids':['ORDER2'],
            'covered_candidates_equal_actual_revision_list':True,'unadmitted_original_failure_notice_retained_separately':True,
            'actual_input_inspection':partition,'semantic_admission_asserted':False})
        adaptation_path=out/'DRIVER_ADAPTATION_FINAL7.json'
        q.publish(adaptation_path,{'schema_version':'fgr20_global_seven_final_gold_scope_recovery_v1',
            'wrapper_binding':q.binding(__file__),'frozen_global8_helper_binding':q.binding(global8.__file__),
            'frozen_saved_verifier_helper_binding':q.binding(saved.__file__),'frozen_RAM_adapter_binding':q.binding(rsa.__file__),
            'input_freeze_binding':q.binding(out/'RECOVERY_INPUT_FREEZE.json'),'clone_changes':clone,
            'prior_preparation_adaptation_binding':q.binding(out/'DRIVER_ADAPTATION.json') if (out/'DRIVER_ADAPTATION.json').exists() else None,
            'final7_partition_adaptation_binding':q.binding(out/'FINAL7_PARTITION_ADAPTATION.json'),
            'only_possible_new_call_id':saved.call_id(ROOT,ARM),'covered_slot_ids':SLOTS,'uncovered_slots_retained_HOLD':['ORDER2'],
            'no_new_gold_author_plan_or_revision':True,'semantic_admission_asserted':False})
        if not do_run:return {'root_id':ROOT,'arm':ARM,'preparation_binding':q.binding(out/'RECOVERY_INPUT_FREEZE.json'),'semantic_calls_started':0}
        if q.cc.stop_generation():raise RuntimeError('Generation cutoff before the unused first content verifier')
        assert_idle()
        with (saved.ROOT/'.family_call_start.lock').open('a+') as family:
            fcntl.flock(family,fcntl.LOCK_EX|fcntl.LOCK_NB);assert_idle();rsa.install(q.binding(adaptation_path))
            covered=[c for c in q.read(INPUT_WORK/'CANDIDATES_FINAL.json') if c['slot_id'] in SLOTS]
            data=reviewer(ROOT,ARM,covered,source,SELECTION,out,frozen,q.cc)
        verify_budget(saved.actual_requests(ROOT,ARM),frozen['saved_gold_call_id'])
        return publish_ready(out,frozen,data)


def self_check(source):
    before=saved.actual_requests(ROOT,ARM);frozen=inspect_inputs(source)
    raw=q.read(INPUT_WORK/'BLIND_GOLD_revision_01.json');mapped,changes=normalize_saved_seven(raw)
    assert len(changes)==21 and next(x for x in mapped['reviews'] if x['question_key']=='Qc148b6e2c27f50')['r03_necessity']['decision']=='FAIL'
    for defect in ('global_false','row_conflict','missing_alias','extra_alias','missing_record'):
        bad=copy.deepcopy(raw)
        if defect=='global_false':bad['complete_source_read']=False
        elif defect=='row_conflict':bad['reviews'][0]['solver_results_read']=True
        elif defect=='missing_alias':del bad['reviews'][0]['r03_necessity']['removal_reason']
        elif defect=='missing_record':bad['reviews'].pop()
        else:bad['reviews'][0]['r03_necessity']['rationale']='ENGINEERING_CONFLICT_FIXTURE'
        try:normalize_saved_seven(bad)
        except RuntimeError:pass
        else:raise AssertionError('Unapproved missing/conflicting record was repaired')
    reviewer,clone=build_review();covered=[c for c in q.read(INPUT_WORK/'CANDIDATES_FINAL.json') if c['slot_id'] in SLOTS];calls=[]
    class Mock:
        def call(self,key,prompt):
            assert key==saved.call_id(ROOT,ARM)
            payload=saved.shape.request_payload(prompt)
            assert payload['complete_source_and_warnings']==q.source_view(source) and len(payload['review_cases'])==7
            assert [x['question_key'] for x in payload['review_cases']]==[q.anonymize_candidate(c)[0]['question_key'] for c in covered]
            assert next(x for x in payload['review_cases'] if x['question_key']=='Qc148b6e2c27f50')['saved_independent_gold']['r03_necessity']['decision']=='FAIL'
            calls.append({'call_id':key,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()})
            raise RuntimeError('ENGINEERING_MOCK_SEVEN_SAVED_GOLD_FIRST_VERIFIER_NO_DISPATCH')
    old=(q.normalize_blind,q.normalize_candidates,q.validate_g_prompt);cache=BASE/'runtime_cache/g109164_global7_gold_selfcheck';cache.mkdir(parents=True,exist_ok=True)
    try:
        rsa.install()
        with tempfile.TemporaryDirectory(prefix='seven_',dir=cache) as d:
            out=Path(d);q.publish(out/'RECOVERY_INPUT_FREEZE.json',frozen)
            data=reviewer(ROOT,ARM,covered,source,SELECTION,out,frozen,Mock())
            assert len(calls)==1 and len(data['reviews'])==7 and all(x['decision']=='HOLD' for x in data['reviews'])
            ready=publish_ready(out,frozen,data)
            assert q.read(out/'CANDIDATES_FINAL.json')==q.read(INPUT_WORK/'CANDIDATES_REVISION_01.json')
            assert (out/'ORIGINAL_FINAL_CANDIDATES_ARCHIVE.json').read_bytes()==(INPUT_WORK/'CANDIDATES_FINAL.json').read_bytes()
            assert (out/'ORIGINAL_CONTENT_REVIEW_FINAL_ARCHIVE.json').read_bytes()==(INPUT_WORK/'CONTENT_REVIEW_FINAL.json').read_bytes()
            old_order2=next(x for x in q.read(INPUT_WORK/'CONTENT_REVIEW_FINAL.json')['reviews'] if x['candidate_id'].endswith('-ORDER2'))
            archive=q.read(out/'UNADMITTED_FAILURE_ARCHIVE.json')
            assert archive['prior_review']==old_order2 and archive['prior_review']['decision']=='HOLD'
            assert archive['candidate']==next(c for c in q.read(INPUT_WORK/'CANDIDATES_FINAL.json') if c['slot_id']=='ORDER2')
            assert len(q.read(out/'CONTENT_REVIEW_FINAL.json')['reviews'])==7
            assert all(not x['candidate_id'].endswith('-ORDER2') for x in q.read(out/'CONTENT_REVIEW_FINAL.json')['reviews'])
            assert len(q.read(out/'CANDIDATES_FINAL.json'))==7
            assert ready['unadmitted_failure_archive_binding']==q.binding(out/'UNADMITTED_FAILURE_ARCHIVE.json')
            assert not (out/'CONTENT_VERIFIER_CALL_revision_01.json').exists()
    finally:q.normalize_blind,q.normalize_candidates,q.validate_g_prompt=old
    assert saved.actual_requests(ROOT,ARM)==before
    return {'status':'PASS','semantic_calls_started':0,'root_id':ROOT,'arm':ARM,'covered_actual_saved_Gold_records':7,
        'uncovered_ORDER2_original_failure_notice_and_HOLD_archived_unchanged':True,
        'final_candidates_exactly_actual_seven_revision_records':True,'original_candidate_review_file_archive_bytes_unchanged':True,
        'saved_MC4_R03_FAIL_preserved_in_verifier_prompt':True,'reversible_changes':21,'original_parser_errors':14,
        'mapped_frozen_parser_errors':[],'mock_only_first_verifier_reached':calls,'mock_no_response_covered_HOLD_count':7,
        'new_plan_author_Gold_revision_calls':0,'semantic_admission_asserted':False,'clone':clone}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['inspect','prepare','run','self-check']);args=parser.parse_args()
    q.SOURCE_SELECTION_PATH=SELECTION;q.SOURCE_FREEZE_PATH=BASE/'SOURCE_FREEZE.json';_,sources=q.load_selection(SELECTION);source=sources[ROOT]
    if args.command=='self-check':print(json.dumps(self_check(source),ensure_ascii=False));return 0
    value=inspect_inputs(source) if args.command=='inspect' else run(source,args.command=='run')
    print(json.dumps(value,ensure_ascii=False));return 0


if __name__=='__main__':raise SystemExit(main())
