"""Project historical question text, including rejected writer drafts, for dedup."""
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

PROJECT = Path('/root/autodl-tmp/gxy-selfplay')
DELIVERIES = PROJECT / 'offline-deliveries'
HERE = Path(__file__).resolve().parent
ROOTS = ['cc-train-episode-' + str(n) for n in (104814, 130221, 160549, 194259)]
PRUNED = {'h4_scale_pilot_v89', 'solver', 'base', 'trained', 'logs', 'feedback',
          '__pycache__', '.git', 'test', 'dev'}


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    p = Path(path).resolve()
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def walk(value, pointer='$'):
    if isinstance(value, dict):
        yield pointer, value
        for key, child in value.items():
            if isinstance(child, (dict, list)):
                yield from walk(child, pointer + '.' + key)
            elif key in ('text', 'content') and isinstance(child, str):
                raw = child.strip()
                if raw.startswith('```') and raw.endswith('```'):
                    raw = raw.split('\n', 1)[1].rsplit('```', 1)[0].strip()
                if raw.startswith(('{', '[')):
                    try:
                        parsed = json.loads(raw)
                    except ValueError:
                        continue
                    yield from walk(parsed, pointer + '.' + key + '[parsed_json]')
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk(child, pointer + '[' + str(index) + ']')


def texts(value, kind):
    if isinstance(value, dict) and all(isinstance(x, str) for x in value.values()):
        return value
    if isinstance(value, list):
        rows = {}
        for i, x in enumerate(value):
            if isinstance(x, dict):
                x = x.get('proposition', x.get('text', x.get('statement')))
            if not isinstance(x, str):
                return None
            rows[('I' + str(i + 1)) if kind == 'ordering' else chr(65 + i)] = x
        return rows
    return None


def main():
    inventory = read(HERE.parent / 'inventory/DISCOVERY_RECEIPT.json')
    paths = {Path(row['path']) for row in inventory['scan_files']}
    for directory, subdirs, files in os.walk(DELIVERIES):
        subdirs[:] = sorted(d for d in subdirs if d not in PRUNED)
        for name in files:
            if not name.endswith('.json'):
                continue
            upper = name.upper()
            if any(word in upper for word in ('POOL', 'CANDIDATE', 'PUBLIC')) or name in ('questions.json', 'RESPONSE.json', 'response.json', 'parsed.json'):
                paths.add(Path(directory) / name)
    by_root = {root: {} for root in ROOTS}
    source_bindings, errors, skips = {}, [], []
    scanned = 0
    for path in sorted(paths):
        if 'h4_scale_pilot_v89' in path.parts:
            continue
        raw = path.read_text()
        scanned += 1
        if not any(root in raw for root in ROOTS):
            continue
        try:
            value = json.loads(raw)
        except ValueError as exc:
            errors.append({'path': str(path), 'error': str(exc)})
            continue
        for pointer, row in walk(value):
            root = row.get('root_id')
            if root not in ROOTS:
                continue
            kind = row.get('question_type')
            if kind not in ('multi_select', 'ordering'):
                continue
            field = 'items' if kind == 'ordering' else 'options'
            statements = texts(row.get(field) or row.get('options') or row.get('items'), kind)
            if not statements or not row.get('candidate_id') or not isinstance(row.get('question'), str):
                if row.get('options') or row.get('items'):
                    skips.append({'path': str(path), 'pointer': pointer, 'candidate_id': row.get('candidate_id'), 'reason': 'public_question_schema_incomplete'})
                continue
            pub = {k: row[k] for k in ('candidate_id', 'root_id', 'question_type', 'question')}
            pub[field] = statements
            if kind == 'ordering':
                pub['chronology_kind'] = row.get('chronology_kind', 'unspecified_historical_public_inventory')
            if row.get('source_path'):
                pub['source_path'] = row['source_path']
            if row.get('source_sha256'):
                pub['source_sha256'] = row['source_sha256']
            key = (kind, tuple(sorted(' '.join(t.split()).casefold() for t in statements.values())))
            lineage = {'path': str(path), 'json_pointer': pointer, 'candidate_id': row['candidate_id'],
                       'raw_response': path.name.lower() == 'response.json'}
            source_bindings.setdefault(str(path), bind(path))
            if key not in by_root[root]:
                by_root[root][key] = {'question': pub, 'occurrences': []}
            by_root[root][key]['occurrences'].append(lineage)
    outputs, metadata = [], []
    for root in ROOTS:
        groups = list(by_root[root].values())
        rows = [group['question'] for group in groups]
        used_paths = sorted({v['path'] for group in groups for v in group['occurrences']})
        output = {'version': 'v89_historical_public_whole_sets_v1', 'root_id': root,
            'questions': rows, 'inputs': [source_bindings[p] for p in used_paths],
            'projection': 'Only candidate identity, question, options/items, chronology and source metadata. No Gold, evidence, rationale, review judgments or Solver scores.',
            'whole_set_compaction': 'Exact whitespace-normalized, casefolded statement multiset by question type; first verbatim representative retained. No semantic merging.',
            'accepted_and_rejected_pool_drafts_included': True,
            'raw_writer_response_drafts_included': True,
            'historical_private_fields_encountered': True, 'current_private_artifacts_read': False,
            'semantic_certification_claimed': False}
        path = HERE / ('HISTORICAL_PUBLIC_PRIORS_root' + root.rsplit('-', 1)[1] + '.json')
        if path.exists():
            assert read(path) == output, 'existing public projection changed: ' + str(path)
        else:
            with path.open('x') as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
                f.write('\n')
        outputs.append(bind(path))
        metadata.append({'root_id': root, 'unique_whole_sets': len(groups),
            'counts_by_type': dict(Counter(r['question_type'] for r in rows)),
            'six_item_counts_by_type': dict(Counter(r['question_type'] for r in rows if len(r.get('items', r.get('options', {}))) == 6)),
            'raw_response_occurrences': sum(v['raw_response'] for g in groups for v in g['occurrences']),
            'groups': [{'candidate_id': g['question']['candidate_id'], 'occurrences': g['occurrences']} for g in groups]})
    receipt = {'version': 'v89_public_prior_discovery_receipt_v1', 'builder': bind(__file__),
        'files_scanned': scanned, 'selected_input_files': len(source_bindings),
        'parse_errors': errors, 'incomplete_schema_rows': skips, 'outputs': outputs, 'roots': metadata,
        'bounded_search_scope': 'Known pool files plus historical *.json filenames containing POOL/CANDIDATE/PUBLIC and questions/RESPONSE/response/parsed JSON; named solver/base/trained/logs/feedback/test/dev directories and v89 excluded.',
        'model_calls': 0, 'historical_writes': 0}
    path = HERE / 'PUBLIC_PRIOR_DISCOVERY_RECEIPT_FIRST_FOUR.json'
    with path.open('x') as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'files_scanned': scanned, 'selected_input_files': len(source_bindings),
        'parse_errors': len(errors), 'incomplete_schema_rows': len(skips),
        'roots': [{k: v for k, v in x.items() if k != 'groups'} for x in metadata]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
