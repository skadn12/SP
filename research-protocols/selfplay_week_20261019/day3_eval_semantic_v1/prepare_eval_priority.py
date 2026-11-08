#!/usr/bin/env python3
"""Prioritize admitted material without changing any prelocked evaluation slot."""
import hashlib
import json
from collections import Counter
from pathlib import Path

WEEK = Path('/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019')
PREP = WEEK / 'day3_common_evaluation_v1'
OUT = Path(__file__).resolve().parent


def bind(path):
    data = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def main():
    lock_path = WEEK / 'day3_production_v1/evaluation/EVAL_SOURCE_FAMILY_LOCK.json'
    lock = json.loads(lock_path.read_text())
    status_path = PREP / 'PREPARATION_STATUS.json'
    status = json.loads(status_path.read_text())
    # Preserve the original source/slot ordering; no outcome-dependent sorting.
    rows = status['slot_material_decisions'] if 'slot_material_decisions' in status else None
    if rows is None:
        candidates = [(k, v) for k, v in status.items()
                      if isinstance(v, list) and len(v) == 186
                      and v and isinstance(v[0], dict) and 'slot_id' in v[0]]
        assert len(candidates) == 1, [k for k, _ in candidates]
        rows = candidates[0][1]
    assert len(rows) == len({r['slot_id'] for r in rows}) == 186
    counts = Counter(r['status'] for r in rows)
    assert counts == {'MATERIAL_SUPPORTED_PENDING_EXCLUSION': 60,
                      'MATERIAL_AMBIGUOUS_HOLD': 98, 'SOURCE_SHORTFALL': 28}
    priority = []
    retained = []
    for row in rows:
        item = {k: row[k] for k in ('slot_id', 'root_id', 'split', 'question_type',
                                    'status', 'rationale_zh', 'material_witnesses')}
        if row['status'] == 'MATERIAL_SUPPORTED_PENDING_EXCLUSION':
            item['next_action'] = 'Inventory actual families; resolve relevant historical support; perform five semantic exclusion checks before selection.'
            priority.append(item)
        else:
            item['action'] = 'Retain existing material disposition; do not replace root, source, scope or append a suffix to fill this slot.'
            retained.append(item)
    output = {
        'schema_version': 'common_eval_material_priority_queue_v1',
        'status': 'WORK_QUEUE_ONLY_NO_FAMILY_CLEARANCE_OR_NEW_QUESTIONS',
        'source_lock': bind(lock_path),
        'preparation_status': bind(status_path),
        'priority_rule': lock['root_scope']['selection_rule'],
        'priority_slots': priority,
        'retained_material_dispositions': retained,
        'counts': dict(counts),
        'reconstruction_scope': 'Resolve concrete related uncertainty for selected candidates; 993 legacy gaps are not a mandatory global rerun. Any relevant unresolved overlap remains HOLD.',
        'limits': ['60 supported material slots are not 60 eligible or accepted questions.',
                   'Hash ordering and source/root separation do not prove semantic independence.',
                   'This queue contains no new question, Gold, Solver result or training result.'],
        'new_evaluation_questions': 0,
        'new_solver_calls': 0,
        'new_training_runs': 0,
    }
    path = OUT / 'COMMON_EVAL_WORK_QUEUE.json'
    data = (json.dumps(output, ensure_ascii=False, indent=2) + '\n').encode()
    if path.exists() and path.read_bytes() != data:
        raise RuntimeError('Refusing to overwrite previous work queue')
    path.write_bytes(data)
    print(json.dumps({'priority_slots': len(priority), 'retained': len(retained),
                      'new_evaluation_questions': 0, 'path': str(path)}))


if __name__ == '__main__':
    main()
