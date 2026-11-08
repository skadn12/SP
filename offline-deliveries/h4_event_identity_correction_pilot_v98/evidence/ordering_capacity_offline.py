"""Render v98 request against real sealed v96 sources/history without HTTP."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import sys
import tempfile

HOME = Path(__file__).resolve().parents[1]
OLD = HOME.parent / 'h4_v95_ready_mixed_qa_v96'
sys.dont_write_bytecode = True
sys.path.insert(0, str(HOME / 'qa_engine'))
import ordering_generation as generation
import ordering_prior
from common import bind


def read(path): return json.loads(Path(path).read_text())
def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    e = generation.engine(); rows = []
    _, _, _, _, dev = e.base.transport(); tokenizer = dev.tokenizer_from(str(e.base.TOKENIZER))
    for number in (25652, 19708):
        old_run = OLD / f'shards/shard_01/ordering/root{number}_ordering'
        root = f'cc-train-episode-{number}'
        slot = next(s for s in read(OLD / 'shards/shard_01/CONFIG.json')['ordering_slots'] if s['root_id'] == root)
        feedback = e.source_adapter.feedback_data(root)
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            e.source_adapter.freeze_feedback(root, run / 'feedback', data=feedback)
            compact, compact_files = e.source_adapter.prior_candidates(root, feedback,
                run / 'feedback' / root / 'candidate_history.json', e.v15)
            priors, receipt, _ = ordering_prior.collect(root,
                run / 'feedback' / root / 'candidate_history.json', ordering_prior.configured_paths(slot),
                compact_rows=compact, compact_files=compact_files)
            for name in ('SOURCE.json', 'WRITER_PLAN.json', 'ACTOR_REFERENCE_POLICY.json', 'provider_profile.json'):
                (run / name).write_bytes((old_run / name).read_bytes())
            manifest = read(old_run / 'MANIFEST.json')
            save(run / 'MANIFEST.json', manifest); save(run / 'PRIOR.json', priors); save(run / 'PRIOR_COVERAGE.json', receipt)
            payload, proof = generation.payload_for(run); save(run / 'REQUEST.json', payload)
            cap = generation.capacity_for(run, tokenizer)
            content = payload['messages'][0]['content']
            source = read(run / 'SOURCE.json')
            assert all(t['text'] in content for t in source['turns'])
            assert feedback['writer_suffix'] in content
            for q in priors:
                assert json.dumps(q['question'], ensure_ascii=False)[1:-1] in content
                assert all(json.dumps(text, ensure_ascii=False)[1:-1] in content for text in q['items'].values())
            rows.append({'root_id': root, 'capacity': cap, 'full_source_turns': len(source['turns']),
                'historical_candidates': receipt['historical_candidate_count'],
                'historical_ordering_candidates': receipt['historical_ordering_candidate_count'],
                'all_supplied_public_ordering_rows': len(priors), 'prior_item_counts': receipt['historical_item_counts'],
                'full_source_and_full_feedback': True, 'gold_or_evidence_in_public_prior_projection': False,
                'request_sha256': e.base.digest(payload), 'http_calls': 0})
    result = {'version': 'v98_actual_old_source_prior_capacity_rehearsal_v1',
              'status': 'PASS' if all(r['capacity']['status'] == 'PASS' for r in rows) else 'BLOCKED',
              'scope': 'real sealed v96 complete sources plus historical feedback and all new ordering priors; v98 roots must independently pass prepare capacity',
              'no_model_calls': True, 'rows': rows, 'script': bind(__file__)}
    save(HOME / 'evidence/ORDERING_CAPACITY_OFFLINE.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    assert result['status'] == 'PASS'


if __name__ == '__main__': main()
