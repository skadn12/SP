"""Workspace-confined persistence and fixed input provenance."""
import datetime, hashlib, json, os, pathlib, threading
BASE = pathlib.Path(__file__).resolve().parents[1]
LOCK = threading.RLock()
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def load(path): return json.loads(pathlib.Path(path).read_text())
def sha(data): return hashlib.sha256(data).hexdigest()
def bind(path):
    path = pathlib.Path(path).resolve(); data = path.read_bytes()
    return dict(path=str(path), sha256=sha(data), bytes=len(data))
def confined(path):
    path = pathlib.Path(path).resolve()
    if BASE not in path.parents: raise ValueError('write outside authorized workspace: '+str(path))
    return path
def atomic(path, data):
    path = confined(path); path.parent.mkdir(parents=True, exist_ok=True)
    with LOCK:
        tmp = path.with_name(path.name+f'.{os.getpid()}.{threading.get_ident()}.tmp')
        with tmp.open('w') as f:
            json.dump(data, f, ensure_ascii=False, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
def write(path, text):
    path = confined(path); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text)
def status(phase=None, **changes):
    with LOCK:
        p = BASE/'RUN_STATUS.json'; d = load(p) if p.exists() else {}
        if phase: d['phase'] = phase
        d.update(changes, updated_at_utc=now(), semantic_parallel_limit=3, training=False, sealed_old_reruns=0)
        atomic(p,d)
def protocol(): return load(BASE/'PROTOCOL_LOCK.json')
def inputs():
    bs = load(BASE/'INPUT_BINDINGS.json')['bindings']
    freeze = next(b for b in bs if b['path'].endswith('/SOURCE_FREEZE.json'))
    outcomes = {x['root_id']:x for x in load(freeze['path'])['outcomes']}
    return outcomes
def source_input(root):
    d = pathlib.Path(inputs()[root]['fixed_initial_draft']['path'])
    return load(d.parents[3]/'inputs'/f'{root}_SOURCE_AUTHOR_INPUT.json')
def checked(binding):
    actual = bind(binding['path'])
    if actual != binding: raise ValueError('immutable binding drift: '+binding['path'])
    return load(binding['path'])
def integrity():
    results=[]
    for b in load(BASE/'INPUT_BINDINGS.json')['bindings']:
        a=bind(b['path']); results.append(dict(expected=b, actual=a, passed=a==b))
    return dict(at=now(), passed=all(x['passed'] for x in results), bindings=results)
