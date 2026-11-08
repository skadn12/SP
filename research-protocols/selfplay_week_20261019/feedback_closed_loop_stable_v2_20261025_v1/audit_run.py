"""Preserve every call, including abandoned duplicate-worker artifacts."""
from common import *
from event_ledger import classify
from codex_client import backend

def metadata(thread):
    b=backend(thread)
    if b['actual_model'] and b['actual_provider']:return b
    for p in (BASE/'runtime/codex_home/sessions').glob('**/*%s*.jsonl'%thread):
        for line in p.read_text().splitlines():
            try:e=json.loads(line)
            except ValueError:continue
            if e.get('type')=='session_meta':b['actual_provider']=e.get('payload',{}).get('model_provider')
            if e.get('type')=='turn_context':b['actual_model']=e.get('payload',{}).get('model')
        b['session_binding']=bind(p)
    return b

def run():
    if load(BASE/'RUN_STATUS.json').get('active_semantic_calls'):raise RuntimeError('audit_only_after_semantic_calls_finish')
    rows=[];orphans=[]
    for d in sorted((BASE/'cli_calls').iterdir()):
        if not d.is_dir():continue
        attempts=[]
        for a in sorted(d.glob('attempt_*')):
            if not a.is_dir():continue
            final=(a/'LAST_MESSAGE.txt').read_text() if (a/'LAST_MESSAGE.txt').exists() else ''
            events=(a/'EVENTS.jsonl').read_text() if (a/'EVENTS.jsonl').exists() else ''
            terminal=load(a/'TERMINAL.json') if (a/'TERMINAL.json').exists() else None
            er=classify(events,final,returncode=terminal.get('returncode',0) if terminal else 0,timed_out=terminal.get('timed_out',False) if terminal else False)
            # Event completeness alone cannot recover an unknown process result.
            er['exit_status_attested']=terminal is not None
            atomic(a/'FINAL_EVENT_AUDIT.json',er)
            be=metadata(er['thread_id']);atomic(a/'FINAL_BACKEND_AUDIT.json',be)
            attempts.append({'directory':str(a),'files':{p.name:bind(p) for p in a.iterdir() if p.is_file()},
              'event_audit':bind(a/'FINAL_EVENT_AUDIT.json'),'backend_audit':bind(a/'FINAL_BACKEND_AUDIT.json'),
              'semantic_response_present':bool(final.strip()) or any(x['class'] in ('agent_message','reasoning') for x in er['event_classes'])})
        if not (d/'RECEIPT.json').exists():
            orphan={'accepted':False,'response':None,'failure_reasons':['duplicate_worker_or_interruption_without_terminal_receipt'],
              'directory':str(d),'at':now(),'forensic_only_no_semantic_retry':True,'attempts':[]}
            atomic(d/'RECEIPT.json',orphan);orphans.append(str(d))
        r=load(d/'RECEIPT.json')
        rows.append({'call_id':d.name,'accepted':r['accepted'],'failure_reasons':r['failure_reasons'],'receipt':bind(d/'RECEIPT.json'),
          'raw_prompt':bind(d/'REQUEST.txt') if (d/'REQUEST.txt').exists() else None,
          'output_schema':bind(d/'SCHEMA.json') if (d/'SCHEMA.json').exists() else None,'attempts':attempts})
    out={'at':now(),'calls':rows,'orphan_failures_preserved':orphans,
      'call_count':len(rows),'accepted_calls':sum(x['accepted'] for x in rows),'failed_calls':sum(not x['accepted'] for x in rows),
      'all_prompts_present':all(r['raw_prompt'] for r in rows),'new_semantic_calls':0,'training':False}
    atomic(BASE/'CALL_AUDIT.json',out);return out

if __name__=='__main__':print(json_compact({k:v for k,v in run().items() if k!='calls'}))
