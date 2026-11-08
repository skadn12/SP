"""Bind actual complete expansion into the separate conditional QA plan."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

HOME = Path(__file__).resolve().parents[1]
RUN = HOME / 'expansion_engine/runs/four_roots_B_scene01'
def read(p): return json.loads(p.read_text())
def bind(p): return {'path': str(p.resolve()), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}

def main():
    scope = read(HOME / 'PRECALL_PLAN.json')
    audit = read(RUN / 'FINAL_AUDIT.json')
    assert audit['status'] == 'PASS' and audit['failures'] == [] and audit['semantic_certified'] is False
    assert read(RUN / 'MANIFEST.json')['roots'] == scope['candidate_roots']
    report = read(RUN / 'REPORT.json')
    assert {r['root_id'] for r in report['routes']} == set(scope['candidate_roots'])
    plan = {'version': 'v98_four_root_semantic_review_mixed_v1',
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'scope_frozen_before_calls': bind(HOME / 'PRECALL_PLAN.json'),
            'candidate_root_budget': 4, 'candidate_roots': scope['candidate_roots'],
            'source_split': 'train', 'question_types': ['multi_select', 'ordering'],
            'training_allowed': False, 'formal_training_admission': False,
            'max_generation_http_total': 12, 'max_solver_calls_per_model_total': 456,
            'exclude_repeated_roots': [], 'exclude_known_hold_roots': [],
            'upstream_expansion_run': {name: bind(RUN / name) for name in
                                     ('MANIFEST.json', 'REPORT.json', 'FINAL_AUDIT.json')},
            'selection_disclosure': scope['selection_bias'],
            'roots_are_first_attempted_in_history': False,
            'note': 'Actual post-expansion bindings only; does not revise PRECALL scope or certify READY.'}
    with (HOME / 'PILOT_PLAN.json').open('x') as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': 'ACTUAL_EXPANSION_BOUND', 'roots': scope['candidate_roots']}))

if __name__ == '__main__': main()
