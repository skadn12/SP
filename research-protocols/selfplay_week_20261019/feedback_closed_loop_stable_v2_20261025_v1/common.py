"""Run-local persistence and immutable input access. No predecessor imports."""
from __future__ import annotations
import datetime, fcntl, hashlib, json, os, pathlib, threading

BASE = pathlib.Path(__file__).resolve().parent
PRE = BASE.parent / 'feedback_replication_20root_20261025_v1'
LOCK = threading.RLock()
PROTOCOL = json.loads((BASE / 'PROTOCOL_LOCK.json').read_text())
ROOTS = PROTOCOL['targets']['new_train_roots']
PILOT = ROOTS[:4]

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def load(p): return json.loads(pathlib.Path(p).read_text())
def sha(b): return hashlib.sha256(b).hexdigest()
def bind(p):
    p = pathlib.Path(p).resolve(); b = p.read_bytes()
    return {'path': str(p), 'sha256': sha(b), 'bytes': len(b)}
def verify(b):
    if bind(b['path']) != {k:b[k] for k in ('path','sha256','bytes')}:
        raise ValueError('immutable_binding_mismatch:' + b['path'])
def local(p):
    p = pathlib.Path(p).resolve()
    if BASE not in p.parents: raise ValueError('write_outside_run:' + str(p))
    return p
def atomic(p, obj):
    p = local(p); p.parent.mkdir(parents=True, exist_ok=True)
    with LOCK:
        t = p.with_name(p.name + '.%s.%s.tmp' % (os.getpid(),threading.get_ident()))
        with t.open('w') as f:
            json.dump(obj,f,ensure_ascii=False,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(t,p)
def write(p, text):
    p=local(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
def status(phase=None, **changes):
    with LOCK:
        with (BASE/'.status.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            p=BASE/'RUN_STATUS.json';d=load(p) if p.exists() else {}
            high=max(d.get('maximum_parallel_semantic_calls',0),changes.get('maximum_parallel_semantic_calls',0))
            incident=BASE/'CONCURRENCY_INCIDENT.json'
            if incident.exists():high=max(high,load(incident)['observed_simultaneous_semantic_calls'])
            if phase: d['phase']=phase
            d.update(changes); d.update(updated_at_utc=now(),semantic_parallel_limit=3,training=False,sealed_old_reruns=0,maximum_parallel_semantic_calls=high)
            atomic(p,d)
def json_compact(obj): return json.dumps(obj,ensure_ascii=False,separators=(',',':'))
