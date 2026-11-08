"""Real offline capacity preflight; all preparation artifacts stay in a temporary directory."""
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
import argparse, hashlib, json, os, sys, tempfile
from unittest.mock import patch
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.dont_write_bytecode = True
BATCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BATCH))
import ordering_generation as generation
from common import bind, read
parser = argparse.ArgumentParser()
parser.add_argument('--suffix', default='initial')
args = parser.parse_args()
output = BATCH / ('evidence/ORDERING_CAPACITY_PREFLIGHT_v87_' + args.suffix + '.json')
assert not output.exists(), 'refuse_overwrite:' + str(output)
rows = []
with ExitStack() as guards:
    for target in ('socket.socket.connect', 'socket.socket.connect_ex', 'socket.create_connection', 'socket.getaddrinfo', 'subprocess.Popen'):
        guards.enter_context(patch(target, side_effect=AssertionError('External operation prohibited')))
    config = read(BATCH / 'CONFIG.json')
    with tempfile.TemporaryDirectory(prefix='v87_capacity_preflight_') as temporary:
        folder = Path(temporary)
        (folder / 'CONFIG.json').write_text(json.dumps(config))
        error = None
        try:
            generation.prepare(folder)
        except Exception as exc:
            error = repr(exc)
        for slot in config['ordering_slots']:
            run = folder / 'ordering' / slot['slot_id']
            row = {'root_id': slot['root_id'], 'slot_id': slot['slot_id']}
            for filename, key in [('CAPACITY.json', 'capacity'), ('PRIOR.json', 'prior')]:
                if (run / filename).exists():
                    value = read(run / filename)
                    row[key if key == 'capacity' else 'distinct_prior_whole_sets'] = value if key == 'capacity' else len(value)
            rows.append(row)
        assert not list(folder.rglob('HTTP_ATTEMPT.json'))
        assert not list(folder.rglob('GENERATION.STARTED.json'))
        prepared = (folder / 'ordering/PREPARATION.COMPLETE.json').exists()
record = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'status': 'PASS' if prepared and error is None else 'BLOCKED_CAPACITY_OR_PREPARATION',
    'error': error, 'roots': rows, 'real_offline_prepare': True,
    'network_and_subprocess_prohibited': True, 'new_model_calls': 0, 'new_http_calls': 0,
    'config': bind(BATCH / 'CONFIG.json'), 'generation_code': bind(BATCH / 'ordering_generation.py'),
    'test_script': bind(Path(__file__)),
    'limitation': 'Local tokenizer proxy and provider-profile context guard only; service exact context, reasoning and output limits remain unknown where the capacity contract states so.'}
with output.open('x') as stream:
    json.dump(record, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(json.dumps(record, ensure_ascii=False))
if record['status'] != 'PASS':
    raise SystemExit(1)
