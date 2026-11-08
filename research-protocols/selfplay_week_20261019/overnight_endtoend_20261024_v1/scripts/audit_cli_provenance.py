#!/usr/bin/env python3
"""Read-only provenance/isolation audit. Optional immutable output via --output.

The audit does not generate text, call Codex, or treat a source-read attestation
as proof of faithful reasoning. It verifies recorded process and content bytes.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def bind(p):
    p=Path(p).resolve(); raw=p.read_bytes()
    return {'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
def verify(binding):
    expected={k:binding[k] for k in ['path','sha256','bytes']}
    assert bind(binding['path'])==expected,('BINDING_CHANGED',binding['path'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output');args=ap.parse_args()
    failures=[]; calls=[]; active=[]; tool_events=[]
    for cdir in sorted((BASE/'cli_calls').glob('*')):
        if not cdir.is_dir(): continue
        observations=[]
        for events in sorted(cdir.glob('attempt_*/EVENTS.jsonl')):
            counts=Counter(); thread=None; completed=False
            for index,line in enumerate(events.read_text(encoding='utf-8').splitlines(),1):
                try: event=json.loads(line)
                except ValueError:
                    if (events.parent/'TERMINAL.json').exists(): failures.append({'call':cdir.name,'error':'INVALID_SEALED_EVENT_JSON','line':index})
                    continue
                etype=event.get('type');item=event.get('item',{});itype=item.get('type');counts[str(etype)+':'+str(itype)]+=1
                if etype=='thread.started':thread=event.get('thread_id')
                if etype=='turn.completed':completed=True
                if itype and itype not in {'agent_message','reasoning','error','plan','todo_list'}:
                    tool_events.append({'call':cdir.name,'events_path':str(events.resolve()),'line':index,'event_type':etype,'item_type':itype,'item_id':item.get('id')})
            terminal_path=events.parent/'TERMINAL.json'
            if terminal_path.exists():
                terminal=read(terminal_path)
                for key in ['events','stderr','request','raw_response']:
                    if key in terminal:
                        try: verify(terminal[key])
                        except Exception as e: failures.append({'call':cdir.name,'error':str(e),'binding':key})
                observations.append({'events':bind(events),'thread_id':thread,'turn_completed':completed,'sealed':True,'counts':dict(counts)})
            else:
                # Unsealed event files are live and their current bytes are not
                # asserted to be a stable generation artifact.
                active.append({'call':cdir.name,'events_path':str(events.resolve()),'thread_id':thread,'counts_at_snapshot':dict(counts)})
        rp=cdir/'RESULT.json'
        if rp.exists():
            result=read(rp)
            try:
                for key in ['raw_response','parsed_response','terminal_binding','events','stderr','request']: verify(result[key])
                text=Path(result['raw_response']['path']).read_text().strip()
                if text.startswith('```') and text.endswith('```'):text='\n'.join(text.splitlines()[1:-1])
                assert json.loads(text)==read(result['parsed_response']['path']), 'PARSED_RESPONSE_DIFFERS_FROM_RAW'
                backend=result['backend']; assert backend.get('thread_id'),'MISSING_SESSION_ID'
                assert backend.get('model') and backend.get('model_provider'),'MISSING_ACTUAL_BACKEND'
            except Exception as e: failures.append({'call':cdir.name,'error':str(e)})
            calls.append({'call':cdir.name,'result':bind(rp),'backend':result.get('backend'),'attempt_observations':observations})
    for draft_path in sorted((BASE/'sources').glob('*/*/OET1_*/DRAFT.json')):
        review_path=draft_path.with_name('REVIEW.json')
        if not review_path.exists():continue
        draft=read(draft_path); review=read(review_path)
        try:
            assert draft.get('author_id') and review.get('reviewer_id') and draft['author_id']!=review['reviewer_id'],'SOURCE_AUTHOR_REVIEWER_SESSION_NOT_INDEPENDENT'
            verify(draft['cli_raw_binding']);verify(review['cli_raw_binding'])
            assert review['source_binding']==bind(draft_path.with_name('CUMULATIVE_SOURCE.json')),'REVIEW_SOURCE_DRIFT'
            assert review['draft_binding']==bind(draft_path),'REVIEW_DRAFT_DRIFT'
        except Exception as e:failures.append({'path':str(draft_path),'error':str(e)})
    status='FAIL' if failures or tool_events else 'PASS_COMPLETED_CALLS_ACTIVE_REMAIN' if active else 'PASS'
    result={'schema_version':'oet1_readonly_cli_provenance_audit_v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'status':status,'completed_calls':len(calls),'active_calls':active,'completed_call_bindings':calls,
        'forbidden_or_unrecognized_tool_events':tool_events,'failures':failures,
        'scope':'Observed recorded CLI calls; new calls and active unsealed bytes require a later audit. This is engineering provenance review, not content/Gold review.',
        'checker_binding':bind(Path(__file__))}
    if args.output:
        p=Path(args.output);p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x',encoding='utf-8') as f:f.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'status':status,'completed_calls':len(calls),'active_calls':len(active),'failures':len(failures),'forbidden_tool_events':len(tool_events),'binding':bind(p)},ensure_ascii=False))
    else:print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
