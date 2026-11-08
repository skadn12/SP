#!/usr/bin/env python3
"""OET1 provenance/isolation/prefix checks and narrowly scoped crash recovery.

No semantic data is authored or revised. Repair may only reconstruct mechanical
AUTHOR_CONSUMPTION metadata for an accepted scene or copy its exact accepted
last cumulative-source bytes to a missing FINAL_SOURCE path.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
SCENES=[f'OET1:{r}:S{i:02}' for r in 'AB' for i in range(1,4)]
NON_TOOL_ITEMS={'agent_message','reasoning','error'}

def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def bind(p):
    p=Path(p).resolve(); raw=p.read_bytes()
    return {'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
def verify(value):
    expected={k:value[k] for k in ['path','sha256','bytes']}
    assert bind(value['path'])==expected,('BINDING_CHANGED',value['path'])
    return Path(value['path'])
def parse_raw(path):
    s=Path(path).read_text(encoding='utf-8').strip()
    if s.startswith('```') and s.endswith('```'): s='\n'.join(s.splitlines()[1:-1])
    return json.loads(s)
def exclusive_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    return bind(path)

def inspect_events(path, *, sealed):
    observations=[];counts=Counter();thread_id=None;turn_completed=False;invalid=[]
    for index,line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(),1):
        try: event=json.loads(line)
        except ValueError:
            if sealed: invalid.append({'line':index,'error':'INVALID_SEALED_EVENT_JSON'})
            continue
        kind=event.get('type');item=event.get('item',{});itype=item.get('type')
        counts[f'{kind}:{itype}']+=1
        if kind=='thread.started':thread_id=event.get('thread_id')
        if kind=='turn.completed':turn_completed=True
        if itype and itype not in NON_TOOL_ITEMS:
            observations.append({'events_path':str(Path(path).resolve()),'line':index,
                'event_type':kind,'item_type':itype,'item_id':item.get('id'),
                'event_sha256':hashlib.sha256(line.encode('utf-8')).hexdigest()})
    return {'thread_id':thread_id,'turn_completed':turn_completed,'counts':dict(counts),
        'tool_events':observations,'invalid_events':invalid}

def validate_call_result(result_path,prompt=None):
    """Reusable guard for v2: validates all receipt bindings and actual tool use."""
    result_path=Path(result_path); result=read(result_path)
    for key in ['raw_response','parsed_response','terminal_binding','events','stderr','request']:
        verify(result[key])
    terminal=read(result['terminal_binding']['path'])
    for key in ['raw_response','events','stderr','request']:
        assert result[key]==terminal[key],('RESULT_TERMINAL_BINDING_MISMATCH',key)
    assert terminal['backend']==result['backend'],'RESULT_TERMINAL_BACKEND_MISMATCH'
    start_path=Path(result['terminal_binding']['path']).with_name('START.json'); start=read(start_path)
    verify(start['request']);assert start['request']==result['request'],'START_REQUEST_CHANGED'
    assert result.get('response_received') is True,'RESPONSE_RECEIPT_NOT_COMPLETE'
    assert parse_raw(result['raw_response']['path'])==read(result['parsed_response']['path']),'PARSED_RESPONSE_DIFFERS_FROM_RAW'
    if prompt is not None:
        expected=prompt.encode('utf-8'); raw_request=Path(result['request']['path']).read_bytes()
        assert raw_request==expected,'CURRENT_PROMPT_DIFFERS_FROM_FROZEN_REQUEST'
    events=inspect_events(result['events']['path'],sealed=True)
    assert not events['invalid_events'],('INVALID_EVENT_JSON',events['invalid_events'])
    assert not events['tool_events'],('FORBIDDEN_OR_UNRECOGNIZED_TOOL_EVENTS',events['tool_events'])
    assert result['backend'].get('thread_id')==events['thread_id'] and bool(events['thread_id']),'SESSION_BINDING_MISMATCH'
    assert result['backend'].get('model') and result['backend'].get('model_provider'),'MISSING_OBSERVED_BACKEND'
    return result,events

def audit(*,repair_metadata=False):
    failures=[];warnings=[];calls=[];active=[];tools=[];repairs=[];sources=[];by_raw={}
    def fail(scope,error): failures.append({'scope':scope,'error':str(error)})
    preparation=read(BASE/'inputs/PRODUCTION_PREPARATION_RECEIPT.json')
    for value in preparation['artifacts']:
        try:verify(value)
        except Exception as error:fail('frozen_preparation',error)
    freeze=BASE/'PRODUCTION_INPUT_FREEZE_V1.json'
    if freeze.exists():
        for value in read(freeze)['bindings']:
            try:verify(value)
            except Exception as error:fail('production_input_freeze',error)
    for cdir in sorted((BASE/'cli_calls').glob('*')):
        if not cdir.is_dir():continue
        for events in sorted(cdir.glob('attempt_*/EVENTS.jsonl')):
            terminal_path=events.with_name('TERMINAL.json');sealed=terminal_path.exists()
            info=inspect_events(events,sealed=sealed);tools.extend({'call_id':cdir.name,**x} for x in info['tool_events'])
            if info['invalid_events']:fail(str(events),info['invalid_events'])
            if not sealed:
                active.append({'call_id':cdir.name,'events_path':str(events.resolve()),'snapshot_counts':info['counts']});continue
            try:
                terminal=read(terminal_path)
                for key in ['events','stderr','request','raw_response']:
                    if key in terminal:verify(terminal[key])
                start=read(events.with_name('START.json'));verify(start['request'])
                assert start['request']==terminal['request'],'START_TERMINAL_REQUEST_MISMATCH'
                pid=events.with_name('PID.json'); assert pid.exists(),'MISSING_PROCESS_PID_RECEIPT'
                if events.with_name('TRANSPORT_FAILURE.json').exists():
                    transport=read(events.with_name('TRANSPORT_FAILURE.json'))
                    assert transport['original']==terminal,'TRANSPORT_FAILURE_CHANGED_ORIGINAL_RETURN'
                    raw=terminal.get('raw_response')
                    assert not raw or not Path(raw['path']).read_text().strip(),'TRANSPORT_RETRY_AFTER_SEMANTIC_RESPONSE'
            except Exception as error:fail(str(terminal_path),error)
        rp=cdir/'RESULT.json'
        if rp.exists():
            try:
                result,info=validate_call_result(rp);by_raw[result['raw_response']['path']]=result
                calls.append({'call_id':cdir.name,'result_binding':bind(rp),'backend':result['backend'],'event_counts':info['counts']})
            except Exception as error:fail(str(rp),error)
    roots=read(BASE/'PLAN.json')['new_train_roots']
    for root in roots:
        for arm in ['GENERIC','TARGETED']:
            unit=BASE/'sources'/root/arm;packet=read(unit/'SOURCE_AUTHOR_INPUT.json');accepted=[]
            previous_path=Path(packet['source_binding']['path'])
            for scene in SCENES:
                sd=unit/scene.replace(':','_');dp=sd/'DRAFT.json';cp=sd/'CUMULATIVE_SOURCE.json';rp=sd/'REVIEW.json';ap=sd/'ACCEPTED.json'
                if not dp.exists():break
                try:
                    draft=read(dp); previous=read(previous_path)
                    assert draft['input_source_binding']==bind(previous_path),'DRAFT_PREDECESSOR_BINDING_CHANGED'
                    assert draft['author_input_binding']==bind(unit/'SOURCE_AUTHOR_INPUT.json'),'AUTHOR_PACKET_CHANGED'
                    verify(draft['cli_raw_binding']);author_result=by_raw.get(draft['cli_raw_binding']['path'])
                    assert author_result is not None,'AUTHOR_CLI_PROVENANCE_NOT_VALIDATED'
                    response=parse_raw(draft['cli_raw_binding']['path'])
                    assert author_result['backend']['thread_id']==draft['author_id'],'DRAFT_AUTHOR_SESSION_MISMATCH'
                    assert response['all_source_turns_read'] is True and response['source_turn_count']==len(previous['turns']),'AUTHOR_SOURCE_SCOPE_MISMATCH'
                    assert len(response['turns'])==len(draft['turns'])==14,'DRAFT_TURN_COUNT_MISMATCH'
                    for index,(original,turn) in enumerate(zip(response['turns'],draft['turns']),1):
                        assert original['speaker']==turn['speaker'] and original['text']==turn['text'],'DRAFT_SEMANTIC_TEXT_DIFFERS_FROM_AUTHOR_RAW'
                        assert turn['turn_ref']==f'{scene}:T{index:03}' and turn['scene_ref']==scene and turn['speaker'] in packet['speakers'],'DRAFT_TURN_METADATA_INVALID'
                    if cp.exists():
                        cumulative=read(cp)
                        assert cumulative['turns']==previous['turns']+draft['turns'],'CUMULATIVE_PREFIX_OR_APPEND_CHANGED'
                        assert cumulative['input_source_binding']==bind(previous_path) and cumulative['draft_binding']==bind(dp),'CUMULATIVE_INPUT_BINDINGS_CHANGED'
                    else:
                        assert not ap.exists(),'ACCEPTED_WITHOUT_CUMULATIVE_SOURCE';break
                    if scene.startswith('OET1:B:'):
                        native=packet['native_events'][SCENES.index(scene)-3]
                        assert draft['native_event_binding']==native['native_event_binding'],'NATIVE_EVENT_BINDING_CHANGED'
                        verify(native['native_event_binding'])
                    if rp.exists():
                        review=read(rp);verify(review['cli_raw_binding'])
                        reviewer_result=by_raw.get(review['cli_raw_binding']['path']);assert reviewer_result is not None,'REVIEW_CLI_PROVENANCE_NOT_VALIDATED'
                        assert review['reviewer_id']==reviewer_result['backend']['thread_id'] and review['reviewer_id']!=draft['author_id'],'SOURCE_REVIEW_NOT_INDEPENDENT'
                        assert review['source_binding']==bind(cp) and review['draft_binding']==bind(dp),'REVIEW_CONTENT_BINDING_CHANGED'
                        raw_review=parse_raw(review['cli_raw_binding']['path'])
                        for key,value in raw_review.items():assert review[key]==value,('REVIEW_SEMANTIC_VALUE_CHANGED',key)
                        assert review['turn_refs_read']==[t['turn_ref'] for t in cumulative['turns']],'REVIEW_FULL_TURN_METADATA_MISMATCH'
                        turn_text={t['turn_ref']:t['text'] for t in cumulative['turns']}
                        for check in review['checks'].values():
                            assert check['reason'] and check['evidence'],'SOURCE_REVIEW_EVIDENCE_MISSING'
                            for ev in check['evidence']:assert ev['quote'] and ev['quote'] in turn_text[ev['turn_ref']],'SOURCE_REVIEW_QUOTE_NOT_CONTIGUOUS'
                    if not ap.exists():break
                    accepted_receipt=read(ap)
                    assert accepted_receipt['status']=='READY' and accepted_receipt['review']==bind(rp) and accepted_receipt['source']==bind(cp),'ACCEPTANCE_RECEIPT_CHANGED'
                    assert review['decision']=='READY' and review['independent_of_source_author'] is True,'ACCEPTED_FAILED_REVIEW'
                    assert all(v['status'] in ['PASS','WARNING','PASS_WITH_WARNING'] for v in review['checks'].values()),'ACCEPTED_FAILED_DIMENSION'
                    assert not any(f.get('severity')=='BLOCKING' for f in review.get('findings',[])),'ACCEPTED_BLOCKING_FINDING'
                    consumption=sd/'AUTHOR_CONSUMPTION.json'
                    expected={'full_source_read':True,'scope_attestation':response,
                        'turn_refs_read':[t['turn_ref'] for t in previous['turns']],
                        'controller_expanded_turn_refs':True,'input':bind(previous_path),
                        'packet':bind(unit/'SOURCE_AUTHOR_INPUT.json'),'draft':bind(dp),'call_receipt':author_result}
                    if not consumption.exists() and repair_metadata:
                        try:exclusive_json(consumption,expected);repairs.append({'action':'RECONSTRUCT_MISSING_AUTHOR_CONSUMPTION_MECHANICAL_METADATA','binding':bind(consumption),'author_raw_binding':draft['cli_raw_binding'],'accepted_receipt':bind(ap)})
                        except FileExistsError:pass
                    if consumption.exists():assert read(consumption)==expected,'AUTHOR_CONSUMPTION_CHANGED'
                    else:warnings.append({'scope':str(consumption),'reason':'MISSING_MECHANICAL_METADATA; safely recoverable from validated original raw and accepted scene'})
                    accepted.append({'scene':scene,'accepted':bind(ap),'draft':bind(dp),'review':bind(rp),'source':bind(cp)})
                    previous_path=cp
                except Exception as error:
                    fail(str(sd),error);break
            final=unit/'FINAL_SOURCE.json'
            if len(accepted)==6:
                if not final.exists() and repair_metadata:
                    try:
                        with final.open('xb') as stream:stream.write(previous_path.read_bytes())
                        repairs.append({'action':'COPY_EXACT_ACCEPTED_FINAL_CUMULATIVE_BYTES','binding':bind(final),'source_binding':bind(previous_path),'accepted_receipt':accepted[-1]['accepted']})
                    except FileExistsError:pass
                if final.exists():
                    if final.read_bytes()!=previous_path.read_bytes():fail(str(final),'FINAL_SOURCE_NOT_EXACT_LAST_ACCEPTED_CUMULATIVE_BYTES')
                else:warnings.append({'scope':str(final),'reason':'MISSING_FINAL_SOURCE; safely recoverable from accepted last scene exact bytes'})
            elif final.exists():fail(str(final),'FINAL_SOURCE_WITHOUT_ALL_SIX_VALID_ACCEPTANCES')
            sources.append({'root_id':root,'condition':arm,'validated_accepted_scenes':len(accepted),'accepted_scenes':accepted,'final_source':bind(final) if final.exists() else None})
    if tools:fail('cli_tool_isolation',f'{len(tools)} actual tool or unrecognized item events; preserved original paths/line/event hashes')
    status='FAIL' if failures else 'PASS_SNAPSHOT_ACTIVE_CALLS_REMAIN' if active else 'PASS'
    return {'schema_version':'oet1_production_engineering_integrity_v1','created_at_utc':now(),'status':status,
        'verdict':'FAIL' if failures else 'PASS','snapshot_only':bool(active),'completed_calls':len(calls),'active_calls':active,
        'calls':calls,'sources':sources,'actual_tool_events':tools,'failures':failures,'warnings':warnings,
        'mechanical_repairs':repairs,'semantic_revisions':0,'author_or_review_cli_calls_launched':0,
        'provenance_guard':bind(BASE/'scripts/production_driver_v2.py') if (BASE/'scripts/production_driver_v2.py').exists() else None,
        'checker_binding':bind(Path(__file__)),
        'interpretation':'Engineering integrity and observed isolation only; not independent semantic content/Gold admission. Active calls/new outputs need later audit. Expanded turn_refs are disclosed controller metadata; original model read attestation and raw remain available.'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repair-metadata',action='store_true');ap.add_argument('--output');args=ap.parse_args()
    result=audit(repair_metadata=args.repair_metadata)
    if args.output:
        binding=exclusive_json(args.output,result)
        print(json.dumps({'status':result['status'],'verdict':result['verdict'],'completed_calls':result['completed_calls'],
            'active_calls':len(result['active_calls']),'failures':len(result['failures']),'repairs':len(result['mechanical_repairs']),'binding':binding},ensure_ascii=False))
    else:print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
