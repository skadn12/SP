"""Traceable exact-statement deduplication, separate from Solver selection.

Historical warnings do not blacklist unselected drafts. Only formal selected
exports / documented training exposure and accepted duplicates in this batch
produce hard exclusions. No semantic similarity threshold is used.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def statement_signature(values):
    if isinstance(values, dict):
        values = values.values()
    return tuple(sorted(' '.join(text.casefold().split()) for text in values))


def selected_history(deliveries, root, exclude, export_paths=None):
    rows, files = [], []
    paths = export_paths if export_paths is not None else Path(deliveries).rglob('SELECTED_QUESTIONS.json')
    for path in sorted(paths):
        if path.is_relative_to(exclude):
            continue
        obj = json.loads(path.read_text())
        matches = [q for q in obj.get('questions', []) if q.get('root_id') == root]
        files.append(path)
        if not matches:
            continue
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        result_by_id, historical_model = {}, obj.get('solver_model', 'historical_not_inferred')
        historical_policy = obj.get('policy_version', obj.get('selection_policy'))
        if obj.get('source_result'):
            result_path = Path(obj['source_result'])
            if not result_path.is_absolute():
                result_path = Path(deliveries).parent / result_path
            result = json.loads(result_path.read_text())
            files.append(result_path)
            records = result['questions']
            result_by_id = records if isinstance(records, dict) else {q['candidate_id']: q for q in records}
            historical_model = result.get('solver_model', historical_model)
            historical_policy = result.get('policy_version', historical_policy)
        for q in matches:
            values = q.get('options') or q.get('items')
            if not isinstance(values, dict) or not all(isinstance(v, str) for v in values.values()):
                raise ValueError('selected_history_missing_statement_text:' + str(path))
            selection = q.get('training_selection', result_by_id.get(q['candidate_id'], {}))
            if selection.get('selected_by_policy', True) is not True:
                raise ValueError('selected_export_contains_unselected:' + str(path))
            rows.append({'candidate_id': q['candidate_id'], 'root_id': root,
                'question_type': q['question_type'], 'question': q['question'], 'facts': list(values.values()),
                'history_status': 'FORMAL_HISTORICAL_SELECTED_EXPORT', 'hard_dedup': True,
                'source_path': str(path.resolve()), 'source_sha256': sha,
                'policy_version': selection.get('policy_version', historical_policy),
                'solver_model': historical_model,
                'training_exposure_claimed': False})
    return rows, files


def deduplicate(candidates, historical, current=()):
    """Return accepted drafts and traceable rejection reasons; keep raw drafts."""
    blocked, batch = {}, {}
    for row in historical:
        if row.get('hard_dedup') is True and row.get('facts'):
            blocked.setdefault(statement_signature(row['facts']), []).append(row)
    for q in current:
        batch.setdefault(statement_signature(q.get('options') or q.get('items')), q['candidate_id'])
    accepted, rejected = [], []
    for q in candidates:
        key = statement_signature(q.get('options') or q.get('items'))
        if key in blocked:
            rejected.append({'candidate_id': q['candidate_id'], 'errors': ['duplicate_historical_selected_statement_set'],
                'matches': [{'candidate_id': row['candidate_id'], 'source_path': row['source_path'],
                             'source_sha256': row['source_sha256'], 'history_status': row['history_status']}
                            for row in blocked[key]], 'draft': q})
        elif key in batch:
            rejected.append({'candidate_id': q['candidate_id'], 'errors': ['duplicate_same_batch_statement_set'],
                'matches': [{'candidate_id': batch[key]}], 'draft': q})
        else:
            accepted.append(q)
            batch[key] = q['candidate_id']
    return accepted, rejected
