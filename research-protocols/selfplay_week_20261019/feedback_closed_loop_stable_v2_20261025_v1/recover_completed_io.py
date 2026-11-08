"""Recover completed CLI I/O from a paused driver, without a new model call."""
import os,signal
from common import *
from event_ledger import classify
from codex_client import backend

def run():
    status_doc=load(BASE/'RUN_STATUS.json');recovered=[]
    for name,x in status_doc.get('active_semantic_calls',{}).items():
        p=pathlib.Path('/proc/%s/stat'%x['pid'])
        if not p.exists():raise RuntimeError('cannot_attest_exit_status:'+name)
        fields=p.read_text().rsplit(')',1)[1].split()
        if fields[0]!='Z':raise RuntimeError('wait_for_original_call_completion:'+name)
        wait_status=int(fields[49]);rc=os.waitstatus_to_exitcode(wait_status)
        d=BASE/'cli_calls'/name;a=d/f"attempt_{x['attempt']:02d}"
        final=(a/'LAST_MESSAGE.txt').read_text() if (a/'LAST_MESSAGE.txt').exists() else ''
        c=classify((a/'EVENTS.jsonl').read_text(),final,returncode=rc);atomic(a/'EVENT_RECEIPT.json',c)
        be=backend(c['thread_id']);atomic(a/'BACKEND.json',be)
        reasons=list(c['failure_reasons']);obj=None
        try:
            obj=json.loads(final)
            import jsonschema
            jsonschema.validate(obj,load(d/'SCHEMA.json'))
        except Exception as e:reasons.append('response_schema:'+str(e)[:400])
        if be['actual_model']!='gpt-6.1-sol' or be['actual_provider']!='hua':reasons.append('backend_unattested')
        ar={'at':now(),'accepted':not reasons,'failure_reasons':reasons,'semantic_response':bool(final.strip()),
          'returncode':rc,'timed_out':False,'backend':be,'event_receipt':bind(a/'EVENT_RECEIPT.json'),
          'observed_process_wait_status':wait_status,'recovered_io_only':True}
        atomic(a/'TERMINAL.json',ar)
        r={'accepted':not reasons,'response':obj,'failure_reasons':reasons,'attempts':[ar],'directory':str(d),
          'request':bind(d/'REQUEST.txt'),'schema':bind(d/'SCHEMA.json'),'fresh_session':True,
          'semantic_attempts':int(bool(final.strip())),'at':now(),'io_recovery_without_semantic_retry':True}
        atomic(d/'RECEIPT.json',r);recovered.append({'call_id':name,'receipt':bind(d/'RECEIPT.json')})
    atomic(BASE/'COMPLETED_IO_RECOVERY.json',{'at':now(),'recovered':recovered,'semantic_calls_added':0,'reason':'Pause driver to simplify future schema; all original children allowed to finish.'})
    pid=load(BASE/'OWNER.json')['pid'];os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
    status('questions',active_semantic_calls={})
    print(json_compact({'recovered':len(recovered),'stopped_driver':pid}))

if __name__=='__main__':run()
