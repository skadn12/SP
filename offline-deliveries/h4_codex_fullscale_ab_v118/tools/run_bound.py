"""Verify and record additive runtime implementations around preserved tools."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
B = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(B / 'tools'))


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p): return {'path': str(Path(p).resolve()), 'sha256': sha(p)}
def verify(row): assert sha(row['path']) == row['sha256'], 'BOUND_RUNTIME_FILE_CHANGED:' + row['path']


def main():
    bindings = []
    for name in ['MASTER_FREEZE.json', 'evidence/EVALUATION_TOOL_BINDINGS.json',
                 'evidence/DELIVERY_TOOL_BINDINGS.json', 'evidence/REVIEW_IDENTITY_V2_BINDING.json',
                 'evidence/PATH_COMPATIBILITY_V2_BINDING.json', 'evidence/BOUND_RUNTIME_BINDING.json']:
        p = B / name; data = read(p); bindings.append(bind(p))
        for item in data.get('files', []): verify(item); bindings.append(item)
        if 'new_helper_binding' in data:
            verify(data['new_helper_binding']); bindings.append(data['new_helper_binding'])
    args = sys.argv[1:]
    assert len(args) >= 4
    stage = args[0] + '_' + args[1]
    wave = args[args.index('--wave') + 1]
    if '--side' in args: stage += '_' + args[args.index('--side') + 1]
    p = B / 'evidence/RUNTIME' / (wave + '_' + stage + '.json')
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump({'status': 'BOUND_RUNTIME_INVOCATION_STARTED', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
                   'argv': args, 'actual_executor': '/root', 'files': bindings, 'self_binding': bind(__file__),
                   'scope': 'All additive source-path export/review/runtime versions verified alongside frozen original tools.'}, f, indent=2); f.write('\n')
    import runtime_v2
    runtime_v2.main()
    with p.with_name(p.stem + '_COMPLETE.json').open('x') as f:
        json.dump({'status': 'COMPLETE', 'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'invocation_binding': bind(p)}, f, indent=2); f.write('\n')


if __name__ == '__main__': main()
