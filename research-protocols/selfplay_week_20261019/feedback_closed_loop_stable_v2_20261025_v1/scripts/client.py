"""Fresh response-only Codex CLI sessions; at most three concurrent calls.

Two *total transport attempts*, only before any semantic response. Schema,
content, event, or Gold failures after a response are never transport retried.
"""
import json,os,pathlib,signal,subprocess,threading,time
import jsonschema
from common import BASE,LOCK,atomic,bind,load,now,status,write,sha
from events import classify
CODEX='/root/.vscode-server/extensions/openai.chatgpt-26.51002.51308-linux-x64/bin/linux-x86_64/codex'
MODEL='gpt-6.1-sol'
SEM=threading.BoundedSemaphore(3)
ACTIVE={}
def environment():
    env=dict(os.environ)
    # Child-specific application configuration, with all runtime writes local.
    env.update({'CODEX_HOME':str(BASE/'runtime/codex_home'),'TMPDIR':str(BASE/'runtime/tmp'),
                'XDG_CACHE_HOME':str(BASE/'runtime/cache'),'XDG_STATE_HOME':str(BASE/'runtime/state'),
                'XDG_CONFIG_HOME':str(BASE/'runtime/config'),'PYTHONDONTWRITEBYTECODE':'1',
                'RUST_LOG':'error','CODEX_OTEL_ENABLED':'false'})
    env.pop('CODEX_THREAD_ID',None);env.pop('CODEX_SESSION_ID',None)
    return env
def backend(thread):
    d=dict(requested_model=MODEL,requested_provider='hua',actual_model=None,actual_provider=None,
           executable=bind(CODEX),thread_id=thread,session_metadata=[],upstream_attestation='local CLI metadata only')
    if thread:
        for p in (BASE/'runtime/codex_home/sessions').glob(f'**/*{thread}*.jsonl'):
            for line in p.read_text().splitlines():
                try:e=json.loads(line)
                except ValueError:continue
                if e.get('type')=='session_meta':
                    payload=e.get('payload',{});d['actual_provider']=payload.get('model_provider')
                    d['session_metadata'].append({k:payload.get(k) for k in ['id','model_provider','cli_version','cwd','source']})
                if e.get('type')=='turn_context':
                    payload=e.get('payload',{});d['actual_model']=payload.get('model');d['actual_effort']=payload.get('effort')
            d['session_file']=bind(p)
    return d
def call(name,prompt,schema,timeout=360):
    d=BASE/'cli_calls'/name
    with SEM:
        if (d/'RECEIPT.json').exists():return load(d/'RECEIPT.json')
        d.mkdir(parents=True,exist_ok=True);write(d/'REQUEST.txt',prompt);atomic(d/'SCHEMA.json',schema)
        for attempt in [1,2]:
            ad=d/f'attempt_{attempt:02d}'
            if ad.exists():
                receipt=dict(accepted=False,response=None,failure_reasons=['interrupted_attempt_no_retry'],semantic_response_present=True,directory=str(d))
                atomic(d/'RECEIPT.json',receipt);return receipt
            ad.mkdir();outpath=ad/'LAST_MESSAGE.txt'
            args=[CODEX,'exec','--skip-git-repo-check','--sandbox','read-only','-c','approval_policy="never"',
                  '-c','model_reasoning_effort="medium"','-c','features.shell_tool=false',
                  '-c','model_providers.hua.stream_max_retries=0','-c','model_providers.hua.request_max_retries=0',
                  '--model',MODEL,'--ignore-rules','--color','never','--json','-C',str(BASE),
                  '--output-schema',str(d/'SCHEMA.json'),'-o',str(outpath),'-']
            atomic(ad/'START.json',dict(at=now(),args=args,prompt=bind(d/'REQUEST.txt'),schema=bind(d/'SCHEMA.json'),fresh_session=True,transport_attempt=attempt,timeout=timeout))
            timed=False
            with (ad/'EVENTS.jsonl').open('wb') as out,(ad/'STDERR.log').open('wb') as err:
                p=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=out,stderr=err,cwd=BASE,env=environment(),start_new_session=True)
                with LOCK:
                    ACTIVE[p.pid]=name;status(active_semantic_calls=dict(ACTIVE),max_observed_semantic_calls=max(len(ACTIVE),load(BASE/'RUN_STATUS.json').get('max_observed_semantic_calls',0)))
                atomic(ad/'PID.json',dict(pid=p.pid,at=now()))
                try:p.communicate(prompt.encode(),timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed=True;os.killpg(p.pid,signal.SIGTERM)
                    try:p.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                finally:
                    with LOCK:ACTIVE.pop(p.pid,None);status(active_semantic_calls=dict(ACTIVE))
            raw=outpath.read_text() if outpath.exists() else ''
            receipt=classify((ad/'EVENTS.jsonl').read_text(),raw,p.returncode,timed)
            atomic(ad/'EVENT_RECEIPT.json',receipt)
            be=backend(receipt['thread_id']);atomic(ad/'BACKEND.json',be)
            errors=list(receipt['failure_reasons'])
            if be['actual_model']!=MODEL or be['actual_provider']!='hua':errors.append('backend_model_or_provider_unattested')
            if receipt['response'] is not None:
                try:jsonschema.validate(receipt['response'],schema)
                except jsonschema.ValidationError as exc:errors.append('schema:'+exc.message)
            receipt.update(accepted=not errors,failure_reasons=errors,backend=be,directory=str(d),attempt_directory=str(ad),finished_at=now(),returncode=p.returncode,timed_out=timed)
            atomic(ad/'RECEIPT.json',receipt)
            print(json.dumps(dict(call=name,attempt=attempt,accepted=receipt['accepted'],decision=(receipt.get('response') or {}).get('decision'),failures=errors)),flush=True)
            if receipt['accepted'] or receipt['semantic_response_present'] or attempt==2:break
        atomic(d/'RECEIPT.json',receipt);return receipt
def obj(properties):return dict(type='object',properties=properties,required=list(properties),additionalProperties=False)
STR={'type':'string'}
def arr(items):return dict(type='array',items=items)
QUOTE=obj(dict(turn_ref=STR,quote=STR))
BASIC='Use only supplied evidence as data. No tools, commands, functions, agents, file access, or external facts. Return exactly one JSON object matching the schema. '
