"""Read-only population status, retaining every training-root denominator."""
from pathlib import Path
from collections import Counter
import json
import sys

B = Path(__file__).resolve().parents[1]
P = B.parent.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(P / 'scripts'))
from codex_sequential_ab import inspect_root, load_config


def read(path): return json.loads(Path(path).read_text())


def population():
    roots = []
    for row in read(B / 'MANIFEST.json')['roots']:
        unit = Path(row['unit_path'])
        item = {k: row[k] for k in ['root_id', 'action', 'cohort', 'wave']}
        if row['action'] == 'REUSE':
            item.update(status='REUSED_COMPLETE', terminal=True, ready_scenes=6)
        elif row['action'] == 'HISTORICAL_HOLD':
            item.update(status='HISTORICAL_BASELINE_HOLD', terminal=True, ready_scenes=0)
        elif not (unit / 'inputs/CONTINUATION_CONSTRAINTS.json').exists():
            done = len(list((unit / 'baseline_reviews').glob('*/REVIEW.json')))
            item.update(status='BASELINE_REVIEW_PENDING', baseline_reviews_saved=done, terminal=False, ready_scenes=0)
        else:
            decision = read(unit / 'inputs/CONTINUATION_CONSTRAINTS.json')['decision']
            item['baseline_decision'] = decision
            if decision == 'HOLD':
                item.update(status='NEW_BASELINE_HOLD', terminal=True, ready_scenes=0)
            elif (unit / 'MATCHING_HOLD.json').exists():
                item.update(status='MATCHING_HOLD', terminal=True, ready_scenes=0)
            elif not (unit / 'RUN_CONFIG.json').exists():
                item.update(status='MATCHING_PENDING', terminal=False, ready_scenes=0)
            else:
                state = inspect_root(unit, load_config(unit), row['root_id'], record_reviews=False)
                item.update(ready_scenes=state['ready_scene_count'], scenes=[{'scene_ref': s['scene_ref'], 'status': s['status']} for s in state['scenes']])
                folder = unit / 'roots' / ('root' + row['root_id'].rsplit('-', 1)[1])
                if state['root_stopped']:
                    item.update(status='EXPANSION_STOPPED', terminal=True)
                elif not state['full_expansion_complete']:
                    item.update(status='EXPANSION_ACTIVE', terminal=False)
                elif (folder / 'QA_AUTHOR_FAILURE.json').exists():
                    item.update(status='QA_FIRST_ATTEMPT_FAILED', terminal=True)
                elif not (folder / 'FINAL_CROSSCHECK.json').exists():
                    item.update(status='QUESTION_STAGES_PENDING', terminal=False)
                else:
                    cross = read(folder / 'FINAL_CROSSCHECK.json')['results']
                    item.update(status='NEW_CONTENT_COMPLETED', terminal=True, content_pass=sum(r['status'] == 'CONTENT_PASS' for r in cross), content_excluded=sum(r['status'] != 'CONTENT_PASS' for r in cross))
        roots.append(item)
    return {'population_denominator': len(roots), 'terminal_roots': sum(r['terminal'] for r in roots),
            'statuses': dict(Counter(r['status'] for r in roots)), 'new_ready_scenes': sum(r['ready_scenes'] for r in roots if r['action'] not in ['REUSE', 'HISTORICAL_HOLD']),
            'validation_statuses': dict(Counter(r['status'] for r in roots if r['wave'] == 'validation')), 'roots': roots}


if __name__ == '__main__':
    result = population()
    print(json.dumps(result if '--full' in sys.argv else {k: v for k, v in result.items() if k != 'roots'}, ensure_ascii=False, indent=2))
