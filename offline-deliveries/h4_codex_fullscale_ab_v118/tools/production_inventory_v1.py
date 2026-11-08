"""Report attempted sources and review/procedure failures without rescuing them."""
from collections import Counter
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from status import population


def read(path):
    return json.loads(Path(path).read_text())


def inventory():
    batch = HERE.parent
    state = population()
    rows = []
    for root in state['roots']:
        unit = batch / 'units' / ('root' + root['root_id'].rsplit('-', 1)[1])
        folder = unit / 'roots' / unit.name
        row = {key: root[key] for key in ['root_id', 'wave', 'action', 'status', 'terminal']}
        row['valid_ready_scenes'] = root.get('ready_scenes', 0)
        row['materialized_scenes'] = len(list(folder.glob('scenes/*/CUMULATIVE_SOURCE.json')))
        reviews = [read(p) for p in folder.glob('scenes/*/SOURCE_REVIEW.json')]
        row['historical_ready_reviews_retained'] = sum(r['decision'] == 'READY' for r in reviews)
        row['six_scene_reviews_ready'] = len(reviews) == 6 and all(r['decision'] == 'READY' for r in reviews)
        row['qa_procedure_hold'] = (unit / 'evidence/QA_REVIEW_EXECUTION_HOLD.json').exists()
        row['failure_details'] = []
        for review in reviews:
            if review['decision'] == 'HOLD':
                row['failure_details'].append({'phase': 'scene', 'scene_ref': review['scene_ref'],
                    'failed_checks': {name: c['reason'] for name, c in review['checks'].items() if c['status'] == 'FAIL'}})
        for path in unit.glob('evidence/late_source_recheck_*/LATE_SOURCE_HOLD.json'):
            value = read(path)
            row['failure_details'].append({'phase': 'late_source_recheck', 'evidence_path': str(path),
                'failed_checks': {name: c['reason'] for name, c in value['checks'].items() if c['status'] == 'FAIL'}})
        for path in [*folder.glob('scenes/*/AUTHOR_FAILURE.json'), folder / 'QA_AUTHOR_FAILURE.json',
                     unit / 'evidence/QA_REVIEW_EXECUTION_HOLD.json']:
            if path.is_file():
                value = read(path)
                row['failure_details'].append({'phase': 'qa_procedure' if path.name == 'QA_REVIEW_EXECUTION_HOLD.json' else 'author_failure',
                                              'reason': value.get('reason', value.get('error')), 'evidence_path': str(path)})
        if root['status'] in {'NEW_BASELINE_HOLD', 'HISTORICAL_BASELINE_HOLD', 'MATCHING_HOLD'}:
            row['failure_details'].append({'phase': root['status'].lower()})
        row['source_only_complete_qa_hold'] = row['six_scene_reviews_ready'] and row['qa_procedure_hold']
        rows.append(row)
    return {'population_denominator': state['population_denominator'], 'terminal_roots': state['terminal_roots'],
            'statuses': state['statuses'],
            'new_materialized_scenes': sum(r['materialized_scenes'] for r in rows),
            'new_six_scene_reviews_ready_roots': sum(r['six_scene_reviews_ready'] for r in rows),
            'source_only_complete_qa_hold_roots': [r['root_id'] for r in rows if r['source_only_complete_qa_hold']],
            'new_valid_ready_scenes': sum(r['valid_ready_scenes'] for r in rows if r['action'] not in {'REUSE', 'HISTORICAL_HOLD'}),
            'failure_phases': dict(Counter(d['phase'] for r in rows for d in r['failure_details'])),
            'scope': 'Retained accepted reviews are historical decisions; a later root HOLD does not remain an eligible question source.',
            'roots': rows}


if __name__ == '__main__':
    data = inventory()
    print(json.dumps(data if '--full' in sys.argv else {k: v for k, v in data.items() if k != 'roots'},
                     ensure_ascii=False, indent=2))
