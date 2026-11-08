"""Lossless public ordering history without historical Gold or evidence fields."""
from copy import deepcopy
from pathlib import Path
from common import bind, need, read

VERSION = 'v98_complete_public_ordering_priors_v1'
PUBLIC_FIELDS = {'candidate_id', 'root_id', 'question_type', 'question', 'items',
                 'chronology_kind', 'source_path', 'source_sha256'}


def project(row, root, source_binding, *, candidate_id=None):
    need(row.get('root_id', root) == root and row.get('question_type') == 'ordering',
         'ordering_prior_identity')
    items = row.get('items')
    need(isinstance(items, dict) and len(items) >= 2
         and all(isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in items.items()),
         'ordering_prior_public_items')
    qid = candidate_id or row.get('candidate_id')
    need(isinstance(qid, str) and qid and isinstance(row.get('question'), str) and row['question'].strip(),
         'ordering_prior_public_identity')
    result = {'candidate_id': qid, 'root_id': root, 'question_type': 'ordering',
              'question': row['question'], 'items': deepcopy(items),
              'source_path': source_binding['path'], 'source_sha256': source_binding['sha256']}
    if 'chronology_kind' in row:
        result['chronology_kind'] = row['chronology_kind']
    need(set(result) <= PUBLIC_FIELDS, 'ordering_prior_private_field_leak')
    return result


def configured_paths(slot):
    """An explicit empty extras list must never discard the mandatory baseline."""
    need(isinstance(slot.get('prior_pool_path'), str), 'ordering_prior_baseline_required')
    extras = slot.get('prior_pool_paths', [])
    need(isinstance(extras, list) and all(isinstance(p, str) for p in extras),
         'ordering_prior_extra_paths_invalid')
    return list(dict.fromkeys([slot['prior_pool_path'], *extras]))


def collect(root, history_path, prior_paths, *, compact_rows=(), compact_files=()):
    history_path = Path(history_path).resolve()
    history = read(history_path)
    need(isinstance(history, list) and bool(history), 'ordering_prior_complete_history_required')
    need(len({h['candidate_id'] for h in history}) == len(history)
         and all(h['root_id'] == root for h in history), 'ordering_prior_history_identity')
    rows = []; files = [history_path]; origins = []
    history_ordering = [h for h in history if h['historical_question']['question_type'] == 'ordering']
    for h in history_ordering:
        rows.append(project(h['historical_question'], root, bind(history_path), candidate_id=h['candidate_id']))
        origins.append({'candidate_id': h['candidate_id'], 'kind': 'complete_previous_training_round_history',
                        'source': bind(history_path)})
    seen = {(q['candidate_id'], q['question'], tuple(q['items'].values())) for q in rows}
    for row in compact_rows:
        if row.get('question_type') != 'ordering':
            continue
        key = (row['candidate_id'], row['question'], tuple(row['facts']))
        if key in seen:
            continue
        need(isinstance(row.get('facts'), list) and len(row['facts']) >= 2, 'ordering_compact_prior_facts')
        data = {**row, 'root_id': root, 'items': {f'I{i}': text for i, text in enumerate(row['facts'], 1)}}
        origin = {'path': row['source_path'], 'sha256': row['source_sha256']}
        need(bind(origin['path']) == origin, 'ordering_compact_prior_source_changed')
        rows.append(project(data, root, origin))
        origins.append({'candidate_id': row['candidate_id'], 'kind': 'existing_historical_public_adapter',
                        'original_item_labels_unavailable_reindexed_without_text_changes': True, 'source': origin})
        seen.add(key)
    files.extend(Path(p).resolve() for p in compact_files)
    for value in prior_paths:
        path = Path(value).resolve(); pool = read(path); files.append(path)
        pool = pool['questions'] if isinstance(pool, dict) else pool
        need(isinstance(pool, list), 'ordering_prior_pool_schema')
        for row in pool:
            if row.get('root_id') == root and row.get('question_type') == 'ordering':
                # Preserve every original row and all original public item text.
                # Duplicate IDs across archives remain visible and auditable.
                rows.append(project(row, root, bind(path)))
                origins.append({'candidate_id': row['candidate_id'], 'kind': 'configured_public_pool',
                                'source': bind(path)})
    receipt = {'version': VERSION, 'root_id': root, 'historical_candidate_count': len(history),
               'historical_ordering_candidate_count': len(history_ordering),
               'historical_ordering_candidate_ids': [h['candidate_id'] for h in history_ordering],
               'supplied_public_prior_rows': len(rows), 'public_fields': sorted(PUBLIC_FIELDS),
               'all_historical_ordering_rows_retained': True, 'historical_item_counts': [len(q['items']) for q in rows],
               'source_truncated': False, 'feedback_truncated': False,
               'historical_gold_or_evidence_supplied': False, 'row_origins': origins,
               'source_files': [bind(p) for p in files]}
    return rows, receipt, files
