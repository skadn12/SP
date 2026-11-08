#!/usr/bin/env python3
"""Audited one-shot Codex semantic calls. No automatic retries; three global slots."""
from __future__ import annotations
import concurrent.futures, datetime, fcntl, hashlib, json, os, pathlib, re, signal, subprocess, threading, time
from event_policy import classify_events
BASE=pathlib.Path(__file__).resolve().parents[1]
CODEX='/root/.vscode-server/extensions/openai.chatgpt-26.51002.51308-linux-x64/bin/linux-x86_64/codex'
MODEL='gpt-6.1-sol'
LOCK=threading.RLock()
SEM=threading.BoundedSemaphore(3)
ACTIVE={}

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(data): return hashlib.sha256(data).hexdigest()
def load(p): return json.loads(pathlib.Path(p).read_text())
def bind(p):
 p=pathlib.Path(p).resolve(); b=p.read_bytes(); return {'path':str(p),'sha256':digest(b),'bytes':len(b)}
def atomic(p,obj):
 p=pathlib.Path(p).resolve()
 if BASE not in p.parents: raise ValueError('write outside resume directory')
 p.parent.mkdir(parents=True,exist_ok=True)
 with LOCK:
  t=p.with_name(p.name+f'.{os.getpid()}.{threading.get_ident()}.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');os.replace(t,p)
def start_token(pid):
 try: return pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
 except OSError: return None

def status(phase=None,**changes):
 with LOCK:
  lock=(BASE/'.run_status.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
  try:
   p=BASE/'RUN_STATUS.json';d=load(p) if p.exists() else {}
   if phase:d['phase']=phase
   d.update(changes)
   registered={}
   for f in list((BASE/'cli_calls').glob('*/PID.json'))+list((BASE/'process_registry').glob('*.json')):
    try:
     x=load(f);pid=x['pid']
     if x.get('start_token')==start_token(pid) and start_token(pid) is not None:registered[pid]=x
    except (OSError,ValueError,KeyError):pass
   for pid,x in ACTIVE.items():
    if start_token(pid)==x['start_token']:registered[pid]=x
   registered.pop(os.getpid(),None)
   parents={}
   for f in pathlib.Path('/proc').glob('[0-9]*/stat'):
    try:
     pid=int(f.parent.name);parts=f.read_text().rsplit(')',1)[1].split();parents[pid]=(int(parts[1]),parts[19])
    except (OSError,ValueError,IndexError):pass
   owned=set(registered)|{os.getpid()}
   for _ in range(8):
    added={pid for pid,(ppid,tok) in parents.items() if ppid in owned}-owned
    if not added:break
    for pid in added:registered.setdefault(pid,{'pid':pid,'start_token':parents[pid][1],'parent_pid':parents[pid][0],'kind':'descendant'})
    owned|=added
   d.update(updated_at_utc=now(),active_children=list(registered.values()),worker={'pid':os.getpid(),'start_token':start_token(os.getpid())},semantic_parallel_limit=3,training=False,new_semantic_calls_started=len(list((BASE/'cli_calls').glob('*/START.json'))),new_semantic_calls_finished=len(list((BASE/'cli_calls').glob('*/RECEIPT.json'))))
   d['notes']=['Recovery closed; failed roots and missing slots retained; no training.' if d.get('phase')=='closeout' else 'Authorized recovery executing; counts reflect persisted receipts and live process start tokens.']
   atomic(p,d)
  finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

def strict_response(raw,classification,returncode=0,timed_out=False):
 reasons=list(classification['failure_reasons']);obj=None
 if returncode: reasons.append(f'exit_code:{returncode}')
 if timed_out: reasons.append('timeout')
 try:
  obj=json.loads(raw)
  if not isinstance(obj,(dict,list)): reasons.append('response_not_object_or_list')
 except (ValueError,TypeError): reasons.append('invalid_or_missing_response_json')
 msgs=classification['evidence']['agent_text_events']
 if not msgs or msgs[-1]['item']['text'].strip()!=raw.strip(): reasons.append('last_message_event_mismatch')
 if not any(e.get('type')=='turn.completed' for e in classification['raw_events']): reasons.append('missing_turn_completed')
 return obj,list(dict.fromkeys(reasons))

def backend_evidence(events):
 thread=next((e.get('thread_id') for e in events if e.get('type')=='thread.started'),None)
 d={'requested_model':MODEL,'provider_configuration':'hua','configured_base_url':'https://api.fullcupai.com/','wire_api':'responses','thread_id':thread,'executable':CODEX,'executable_sha256':bind(CODEX)['sha256'],'actual_model':None,'actual_provider':None,'model_attestation':'unavailable'}
 if thread:
  for p in pathlib.Path('/root/.codex/sessions').glob(f'**/*{thread}*.jsonl'):
   evidence=[]
   for line in p.open():
    try: e=json.loads(line)
    except ValueError: continue
    if e.get('type')=='session_meta':
     payload=e.get('payload',{});d['actual_provider']=payload.get('model_provider');d['cli_version']=payload.get('cli_version'); evidence.append(e)
    elif e.get('type')=='turn_context':
     payload=e.get('payload',{});d['actual_model']=payload.get('model');d['actual_effort']=payload.get('effort');evidence.append(e)
   d['session_metadata']=evidence;d['session_file']=str(p)
   if d['actual_model']:d['model_attestation']='local_codex_session_turn_context; upstream routing not independently observable'
 return d

def call(name,prompt,timeout=1200):
 """An existing completed receipt is reused; interrupted calls fail, never retried."""
 d=BASE/'cli_calls'/name
 with SEM:
  if (d/'RECEIPT.json').exists(): return load(d/'RECEIPT.json')
  if d.exists():
   r={'accepted':False,'failure_reasons':['interrupted_existing_call_no_retry'],'directory':str(d),'response':None};atomic(d/'RECEIPT.json',r);return r
  d.mkdir(parents=True)
  (d/'REQUEST.txt').write_text(prompt)
  args=[CODEX,'exec','--skip-git-repo-check','--sandbox','danger-full-access','-c','approval_policy="never"','-c','model_reasoning_effort="medium"','--model',MODEL,'--color','never','--json','-C',str(BASE),'-o',str(d/'LAST_MESSAGE.txt'),'-']
  atomic(d/'START.json',{'at':now(),'args':args,'request':bind(d/'REQUEST.txt'),'timeout_seconds':timeout})
  timed_out=False
  with (d/'EVENTS.jsonl').open('wb') as out,(d/'STDERR.log').open('wb') as err:
   p=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=out,stderr=err,start_new_session=True,cwd=BASE)
   child={'pid':p.pid,'start_token':start_token(p.pid),'started_at_utc':now(),'call_id':name,'kind':'semantic'}
   atomic(d/'PID.json',child)
   with LOCK:ACTIVE[p.pid]=child;status()
   try: p.communicate(prompt.encode(),timeout=timeout)
   except subprocess.TimeoutExpired:
    timed_out=True;os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
  raw=(d/'LAST_MESSAGE.txt').read_text() if (d/'LAST_MESSAGE.txt').exists() else ''
  c=classify_events((d/'EVENTS.jsonl').read_text(),timed_out=timed_out,backend_error=p.returncode!=0)
  atomic(d/'EVENT_CLASSIFICATION.json',c)
  obj,reasons=strict_response(raw,c,p.returncode,timed_out)
  backend=backend_evidence(c['raw_events']);atomic(d/'BACKEND.json',backend)
  if backend['actual_model']!=MODEL:reasons.append('backend_model_unattested_or_mismatch')
  r={'accepted':not reasons,'finished_at_utc':now(),'returncode':p.returncode,'timed_out':timed_out,'failure_reasons':reasons,'response':obj,'backend':backend,'directory':str(d),'hashes':{n:bind(d/n) for n in ['REQUEST.txt','EVENTS.jsonl','STDERR.log','BACKEND.json']}}
  if (d/'LAST_MESSAGE.txt').exists():r['hashes']['LAST_MESSAGE.txt']=bind(d/'LAST_MESSAGE.txt')
  atomic(d/'RECEIPT.json',r)
  with LOCK:ACTIVE.pop(p.pid,None);status()
  print(json.dumps({'call':name,'accepted':r['accepted'],'decision':obj.get('decision') if isinstance(obj,dict) else None,'failures':reasons}),flush=True)
  return r
