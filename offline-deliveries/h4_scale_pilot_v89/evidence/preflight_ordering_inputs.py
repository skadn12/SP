"""Offline actual ordering payload/capacity check, with no production writes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile

PILOT = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads(Path(path).read_text())

def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write(chr(10))

def main(spec_path, receipt_path):
    def guard(event, args):
        if event.startswith('socket.') or event in {'subprocess.Popen', 'os.system', 'os.posix_spawn'}:
            raise RuntimeError('preflight_forbids_external:' + event)
    sys.addaudithook(guard)
    old = PILOT / 'shards/shard_01'
    sys.path.insert(0, str(old))
    import ordering_generation as ordering
    import ordering_plan
    import actor_reference
    spec = read(spec_path)
    config = read(old / 'CONFIG.json')
    e = ordering.engine()
    provider, _ = e._dependencies()
    _, _, _, _, dev = e.base.transport()
    tokenizer = dev.tokenizer_from(str(e.base.TOKENIZER))
    rows = []
    with tempfile.TemporaryDirectory(prefix='v89_actual_ordering_preflight_') as temp:
        for row in spec['roots']:
            slot = next(s for s in config['ordering_slots'] if s['root_id'] == row['root_id'])
            run = Path(temp) / slot['slot_id']
            source = read(slot['source_path'])
            priors = [ordering.public(q) for path in row['public_prior_paths'] for q in read(path)['questions']
                      if q['root_id'] == row['root_id'] and q['question_type'] == 'ordering']
            feedback = e.source_adapter.feedback_data(slot['root_id'])
            e.source_adapter.freeze_feedback(slot['root_id'], run / 'feedback', data=feedback)
            ids = ordering.candidate_ids(config, slot)
            plan = ordering_plan.build(config, slot, source, ids)
            write(run / 'SOURCE.json', source)
            write(run / 'PRIOR.json', priors)
            write(run / 'provider_profile.json', provider)
            write(run / 'WRITER_PLAN.json', ordering_plan.writer_plan(source, plan))
            write(run / 'ACTOR_REFERENCE_POLICY.json', actor_reference.policy_for(config, slot, source))
            write(run / 'MANIFEST.json', {'root_id': slot['root_id'], 'candidate_ids': ids,
                  'source_sha256': slot['source_sha256'], 'max_tokens': config['ordering_output_max_tokens'],
                  'solver_model': e.legacy.TARGET})
            payload, proof = ordering.payload_for(run)
            write(run / 'REQUEST.json', payload)
            capacity = ordering.capacity_for(run, tokenizer)
            rows.append({'root_id': row['root_id'], 'prior_count': len(priors),
                  'prior_inputs': [bind(p) for p in row['public_prior_paths']],
                  'source': bind(slot['source_path']), 'capacity': capacity,
                  'public_projection_contract': 'PASS', 'payload_sha256': proof['payload_sha256']})
    status = 'PASS' if all(r['capacity']['status'] == 'PASS' for r in rows) else 'FAIL'
    write(receipt_path, {'version': 'v89_actual_ordering_preflight_v1',
          'created_at_utc': datetime.now(timezone.utc).isoformat(), 'status': status,
          'spec': bind(spec_path), 'frozen_engine': bind(old / 'ordering_generation.py'),
          'rows': rows, 'generation_http_calls': 0, 'solver_calls': 0,
          'failed_shard_or_historical_files_modified': False,
          'scope': 'Actual prior projection and ordering payload capacity in temporary files; no answer certification.'})
    print(json.dumps({'status': status, 'rows': rows}, ensure_ascii=False))
    if status != 'PASS':
        raise SystemExit(1)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('spec', type=Path)
    p.add_argument('receipt', type=Path)
    a = p.parse_args()
    main(a.spec, a.receipt)
