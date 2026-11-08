#!/usr/bin/env python3
"""Use one consumed failed initial-author identity for six unused revisions.

This is restricted to the exact bound F109164 failed attempt. Its raw response
is never admitted, parsed into candidates, or marked successful. Top-level error
events remain explicit. Frozen drivers and original artifacts stay unchanged.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any
sys.dont_write_bytecode=True
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent))
import question_driver as q
import question_unfilled_slot_continuation as u
import review_schema_adapter as r
ROOT='cc-train-episode-109164'
ARM='F'
CALL='q20_author_f_109164_plan_schema_01'
SLOTS=['MC2','MC3','MC4','MC6','ORDER1','ORDER3']
INPUT_WORK=BASE/'question_plan_continuation/109164/F'
DIRECTORY=BASE/'unused_failed_lineage_adapter/109164/F'
EXPECTED_UNUSED='0bf0fedd04e85d23edc1df0d013ef0f8f82c2303d2ffd885a6fafa315621ea34'
EXPECTED_REVIEW='7fb0938e8bf007ee62bfdbe6938fd26baff28568c35b7eec92eafa59124f44ec'
INSPECTION={'path':str(BASE/'mechanical_audits/F109164_UNUSED_REVISION_INSPECTION.json'),'sha256':'a9a567028961145126a218b839275c71f0b8266bc4b64534f093f628bee12108','bytes':3031}
TOP_ERROR='Reconnecting... 1/5 (stream disconnected before completion: Transport error: network error: error decoding response body)'


def verify_files():
    for module,expected in [(u,EXPECTED_UNUSED),(r,EXPECTED_REVIEW)]:
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Frozen engineering adapter binding drift')
    q.verify_binding(INSPECTION)
    inspection=q.read(INSPECTION['path'])
    if (inspection['root_id'],inspection['arm'],inspection['actual_initial_author_call_id'])!=(ROOT,ARM,CALL):
        raise RuntimeError('Restricted failed initial-author inspection identity drift')
    return inspection


def classify_failed_initial(budget: dict[str,Any]):
    inspection=verify_files()
    if (budget.get('root_id'),budget.get('arm'))!=(ROOT,ARM):
        raise RuntimeError('Failed-lineage exception is restricted to F109164')
    calls=budget.get('initial_author_calls',{})
    if set(calls)!={CALL} or len(calls[CALL])!=1:
        raise RuntimeError('Exactly the original single consumed initial REQUEST is required')
    request=calls[CALL][0]
    if request['requested_slot_ids']!=SLOTS:
        raise RuntimeError('Consumed author assignment is not the exact six valid requested slots')
    q.verify_binding(request['request_binding'])
    if request['request_binding']!=q.read(inspection['actual_initial_terminal_binding']['path'])['request']:
        raise RuntimeError('Actual budget request and failed terminal request differ')
    if (BASE/'cli_calls'/CALL/'RESULT.json').exists():
        raise RuntimeError('Failed initial unexpectedly acquired a success RESULT; do not reinterpret it')
    q.verify_binding(inspection['actual_initial_terminal_binding'])
    terminal=q.read(inspection['actual_initial_terminal_binding']['path'])
    for name in ('request','events','stderr','raw_response'):
        q.verify_binding(terminal[name])
    backend=terminal.get('backend',{})
    if backend!=inspection['actual_initial_backend']:
        raise RuntimeError('Failed initial backend differs from its immutable inspected terminal')
    if any(backend.get(k)!=v for k,v in {'thread_id':'01a16fa3-5865-7923-a0fd-3a53fdf6d566','model_override':None,'model':'gpt-6.1-sol','model_provider':'hua','effort':'ultra'}.items()):
        raise RuntimeError('Failed initial lacks its actual default-backend fresh identity')
    events=[json.loads(line) for line in Path(terminal['events']['path']).read_text().splitlines()]
    started=[e for e in events if e.get('type')=='thread.started']
    if len(started)!=1 or started[0].get('thread_id')!=backend['thread_id']:
        raise RuntimeError('Actual failed EVENTS fresh thread identity mismatch')
    top_errors=[e for e in events if e.get('type')=='error']
    if top_errors!=[{'type':'error','message':TOP_ERROR}]:
        raise RuntimeError('Failed-lineage adapter cannot classify an unexpected top-level error')
    diagnostics=[]
    for event in events:
        if event.get('type') not in {'item.started','item.updated','item.completed'}:continue
        item=event.get('item',{})
        if item.get('type') in {'reasoning','agent_message'}:continue
        if (event.get('type')!='item.completed' or item.get('type')!='error'
                or item.get('id') not in {'item_0','item_1'} or item.get('message')!=q._CONFIG_WARNING):
            raise RuntimeError('Unexpected forbidden item in consumed failed initial attempt')
        diagnostics.append({'event_type':event['type'],'item_type':item['type'],'id':item['id']})
    if diagnostics!=terminal.get('forbidden_tool_items') or [d['id'] for d in diagnostics]!=['item_0','item_1']:
        raise RuntimeError('Exact known startup diagnostics do not match failed terminal')
    completed=[e for e in events if e.get('type')=='turn.completed']
    messages=[e for e in events if e.get('type')=='item.completed' and e.get('item',{}).get('type')=='agent_message']
    if len(completed)!=1 or completed[0].get('usage')!=terminal.get('usage') or not messages:
        raise RuntimeError('Consumed semantic attempt lacks its actual completed message/usage trace')
    if terminal.get('timed_out') is not False or terminal.get('stopped') is not False:
        raise RuntimeError('Interrupted/stopped failed attempt requires different recovery classification')
    for rec in budget.get('revision_requests',[]):
        if rec['call_id']!=u.unused_call_id('revision',ARM,ROOT) or rec['requested_slot_ids']!=SLOTS:
            raise RuntimeError('Revision budget was used by another request or slot assignment')
    classification={'schema_version':'fgr20_consumed_failed_semantic_lineage_v1',
        'classification':'consumed_failed_semantic','root_id':ROOT,'arm':ARM,
        'initial_author_call_id':CALL,'initial_author_thread_id':backend['thread_id'],
        'actual_initial_terminal_binding':inspection['actual_initial_terminal_binding'],
        'actual_initial_events_binding':terminal['events'],'actual_initial_request_binding':terminal['request'],
        'actual_initial_raw_response_binding':terminal['raw_response'],
        'actual_initial_backend':backend,'actual_initial_usage':terminal.get('usage'),
        'top_level_errors_preserved':top_errors,'known_startup_diagnostics_preserved':diagnostics,
        'initial_request_consumed':True,'initial_attempt_success':False,
        'initial_raw_admitted_as_content':False,'raw_parsed_into_candidates':False,
        'original_event_policy_failure_not_erased':True,'transport_retry_authorized':False,
        'new_initial_author_or_planner_authorized':False,
        'only_potential_unused_revision_slots':SLOTS,'invalid_plan_slots_never_reassigned':['MC5','ORDER2'],
        'inspection_binding':INSPECTION,'adapter_binding':q.binding(__file__),
        'semantic_admission_asserted':False}
    return backend['thread_id'],classification


def install_lineage():
    old=u.actual_initial_author
    def restricted(budget):
        if (budget.get('root_id'),budget.get('arm'))!=(ROOT,ARM):
            return old(budget)
        thread,classification=classify_failed_initial(budget)
        classification_path=DIRECTORY/'CONSUMED_FAILED_SEMANTIC_LINEAGE.json'
        q.publish(classification_path,classification)
        return thread,{'initial_author_call_id':CALL,
            'actual_initial_terminal_binding':classification['actual_initial_terminal_binding'],
            'initial_author_backend':classification['actual_initial_backend'],
            'initial_raw_response_binding':classification['actual_initial_raw_response_binding'],
            'initial_response_classification':'consumed_failed_semantic',
            'initial_failure_classification_binding':q.binding(classification_path),
            'initial_raw_admitted_as_content':False,'initial_attempt_success':False}
    u.actual_initial_author=restricted
    return old


def inspect_budget(source):
    original=install_lineage()
    try:
        value=u.inspect_budget(ROOT,ARM,source,INPUT_WORK)
    finally:u.actual_initial_author=original
    if value['eligible_slot_ids']!=SLOTS or value['revision_requests']:
        raise RuntimeError('Only the exact six unused assigned slots may be revised once')
    for slot in ('MC5','ORDER2'):
        row=next(x for x in value['slot_dispositions'] if x['slot_id']==slot)
        if row['disposition']!='invalid_or_missing_frozen_plan_no_replan':
            raise RuntimeError('An invalid frozen plan slot was reintroduced')
    return value


def self_check(source):
    budget=inspect_budget(source)
    prompt=u.revision_prompt(ROOT,ARM,source,q.read(budget['actual_plan_path']),SLOTS,budget,INPUT_WORK)
    assert u.assignment(prompt)=={'root_id':ROOT,'arm':ARM,'slots':SLOTS}
    classification=classify_failed_initial(u.actual_request_budget(ROOT,ARM))[1]
    assert classification['classification']=='consumed_failed_semantic' and classification['initial_attempt_success'] is False
    trace=[]
    original_lineage=u.actual_initial_author
    original_out=u.CONTINUATIONS
    original_call=q.cc.call
    original_stop=q.cc.stop_generation
    original_active=u.qs.active_calls
    cache=BASE/'runtime_cache/unused_failed_lineage_selfcheck';cache.mkdir(parents=True,exist_ok=True)
    try:
        install_lineage()
        def mock_call(key,prompt,**kwargs):
            trace.append({'call_id':key,'assignment':u.assignment(prompt)})
            raise RuntimeError('ENGINEERING_MOCK_SINGLE_UNUSED_REVISION_REACHED_NO_DISPATCH')
        q.cc.call=mock_call;q.cc.stop_generation=lambda:False;u.qs.active_calls=lambda:[]
        with tempfile.TemporaryDirectory(prefix='six_slots_',dir=cache) as tmp:
            u.CONTINUATIONS=Path(tmp)
            ready=u.run_one(ROOT,ARM,source,BASE/'SOURCE_SELECTION_FINAL.json',INPUT_WORK)
            outcome=q.read(ready['final_outcome_binding']['path'])
            assert len(trace)==1 and trace[0]['call_id']==u.unused_call_id('revision',ARM,ROOT)
            assert trace[0]['assignment']['slots']==SLOTS
            assert outcome['eligible_slot_ids']==SLOTS
            assert not q.read(ready['final_candidates_binding']['path'])
    finally:
        u.actual_initial_author=original_lineage;u.CONTINUATIONS=original_out
        q.cc.call=original_call;q.cc.stop_generation=original_stop;u.qs.active_calls=original_active
    return {'status':'PASS','semantic_calls_started':0,'classification':'consumed_failed_semantic',
        'eligible_slot_ids':SLOTS,'invalid_plan_slots_still_excluded':['MC5','ORDER2'],
        'mock_frozen_unused_driver_reached_exact_single_revision':trace,
        'new_plan_or_initial_author_or_gold_or_verify_calls':0,
        'original_initial_raw_admitted_as_content':False,'initial_attempt_success':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['inspect','self-check','run'])
    args=parser.parse_args()
    verify_files()
    q.SOURCE_SELECTION_PATH=BASE/'SOURCE_SELECTION_FINAL.json';q.SOURCE_FREEZE_PATH=BASE/'SOURCE_FREEZE.json'
    _,sources=q.load_selection(q.SOURCE_SELECTION_PATH);source=sources[ROOT]
    if args.command=='self-check':print(json.dumps(self_check(source),ensure_ascii=False));return 0
    if args.command=='inspect':print(json.dumps(inspect_budget(source),ensure_ascii=False));return 0
    if (BASE/'CONTENT_FREEZE.json').exists() or q.cc.stop_generation():
        raise RuntimeError('No fresh unused revision may start after content/generation cutoff')
    launch_path=DIRECTORY/'RUNTIME_LINEAGE_ADAPTER_LAUNCH.json'
    q.publish(launch_path,{'schema_version':'fgr20_failed_lineage_runtime_launch_v1',
        'adapter_binding':q.binding(__file__),'frozen_unused_driver_binding':q.binding(u.__file__),
        'frozen_review_schema_adapter_binding':q.binding(r.__file__),
        'classification':'consumed_failed_semantic','root_id':ROOT,'arm':ARM,
        'only_unused_revision_slots':SLOTS,'old_raw_not_admitted':True,
        'parent_must_schedule_after_other_wrapper_families_idle':True})
    r.install(q.binding(launch_path));install_lineage()
    result=u.run_one(ROOT,ARM,source,q.SOURCE_SELECTION_PATH,INPUT_WORK)
    print(json.dumps(result,ensure_ascii=False));return 0


if __name__=='__main__':raise SystemExit(main())
