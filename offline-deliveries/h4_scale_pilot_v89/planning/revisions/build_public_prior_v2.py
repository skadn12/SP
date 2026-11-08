"""Complete historical public ordering metadata without changing public text.

Missing source fields bind the historical carrier for lineage only. They do not
assert that the carrier is the dialogue source or certify any answer.
"""
import copy
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BATCH = HERE.parents[1]
PLANNING = HERE.parent
ROOT_NUMBERS = (130221, 160549, 194259, 104814)
REQUIRED = ('candidate_id', 'root_id', 'question_type', 'chronology_kind',
            'question', 'items', 'source_path', 'source_sha256')


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path, obj):
    with Path(path).open('x') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def whole_set(row):
    field = 'items' if row['question_type'] == 'ordering' else 'options'
    return (row['question_type'], tuple(sorted(
        ' '.join(value.split()).casefold() for value in row[field].values())))


def main():
    # Import and call only the installed engine's pure public projection. No
    # generation preparation, transport, feedback freeze or model calls occur.
    sys.path.insert(0, str(BATCH / 'engine'))
    import ordering_generation

    discovery_path = PLANNING / 'PUBLIC_PRIOR_DISCOVERY_RECEIPT_FIRST_FOUR.json'
    discovery = read(discovery_path)
    root_lineage = {r['root_id']: r for r in discovery['roots']}
    before = [bind(PLANNING / f'HISTORICAL_PUBLIC_PRIORS_root{n}.json')
              for n in ROOT_NUMBERS]
    roots = []
    outputs = []
    total_missing = Counter()
    all_repairs = []
    for n, original_binding in zip(ROOT_NUMBERS, before):
        original = read(original_binding['path'])
        revised = copy.deepcopy(original)
        revised['version'] = 'v89_historical_public_whole_sets_v2_metadata_contract'
        revised['previous_public_projection'] = original_binding
        revised['missing_ordering_metadata_policy'] = (
            'Unspecified chronology remains explicit. Missing source metadata '
            'binds the first historical inventory/response carrier recorded '
            'for the retained verbatim representative, for lineage only; this '
            'does not prove its dialogue source or certify its answer. '
            'Existing metadata and every public statement are unchanged.')
        groups = root_lineage[original['root_id']]['groups']
        assert len(groups) == len(original['questions'])
        registered_inputs = {r['path']: r for r in original['inputs']}
        root_missing = Counter()
        repairs = []
        ordering_count = 0
        for index, (old, current, group) in enumerate(zip(
                original['questions'], revised['questions'], groups)):
            assert old['candidate_id'] == group['candidate_id']
            if current['question_type'] != 'ordering':
                assert old == current
                continue
            ordering_count += 1
            missing = [field for field in REQUIRED if field not in current]
            assert set(missing) <= {'source_path', 'source_sha256'}
            if missing:
                occurrence = group['occurrences'][0]
                assert occurrence['candidate_id'] == old['candidate_id']
                carrier = registered_inputs[occurrence['path']]
                assert bind(carrier['path']) == carrier
                replacements = {'source_path': carrier['path'],
                                'source_sha256': carrier['sha256']}
                for field in missing:
                    current[field] = replacements[field]
                    root_missing[field] += 1
                repairs.append({'question_index': index,
                    'candidate_id': old['candidate_id'], 'root_id': old['root_id'],
                    'missing_fields': missing, 'carrier': carrier,
                    'carrier_json_pointer': occurrence['json_pointer'],
                    'raw_response_carrier': occurrence['raw_response'],
                    'metadata_role': 'historical_public_text_lineage_only',
                    'original_dialogue_source_proven': False,
                    'public_question_text_changed': False,
                    'private_gold_or_evidence_projected': False})
            projected = ordering_generation.public(current)
            assert tuple(projected) == REQUIRED
            assert all(isinstance(projected[k], str) and projected[k].strip()
                       for k in REQUIRED if k != 'items')
            assert projected['question_type'] == 'ordering'
            assert projected['root_id'] == original['root_id']
            assert isinstance(projected['items'], dict) and projected['items']
            assert all(isinstance(k, str) and isinstance(v, str) and v.strip()
                       for k, v in projected['items'].items())
            assert Path(projected['source_path']).is_absolute()
            assert len(projected['source_sha256']) == 64
            assert all(c in '0123456789abcdef' for c in projected['source_sha256'])
            assert all(current[key] == value for key, value in old.items())
            assert set(current) == set(old) | set(missing)
        assert len(original['questions']) == len(revised['questions'])
        assert [whole_set(q) for q in original['questions']] == [
            whole_set(q) for q in revised['questions']]
        assert [q['candidate_id'] for q in original['questions']] == [
            q['candidate_id'] for q in revised['questions']]
        revised['ordering_metadata_lineage_repairs'] = repairs
        output = HERE / f'HISTORICAL_PUBLIC_PRIORS_root{n}_v2.json'
        write(output, revised)
        outputs.append(bind(output))
        root_receipt = {'root_id': original['root_id'],
            'all_public_questions': len(revised['questions']),
            'ordering_questions_contract_checked': ordering_count,
            'ordering_questions_repaired': len(repairs),
            'missing_fields_before': dict(root_missing),
            'missing_fields_after': {}, 'original': original_binding,
            'revised': outputs[-1], 'question_text_and_statements_identical': True,
            'exact_whole_set_exclusion_coverage_identical': True,
            'ordering_generation_public_contract_passed': True,
            'lineage_repairs': repairs}
        roots.append(root_receipt)
        total_missing.update(root_missing)
        all_repairs.extend(repairs)
    assert before == [bind(row['path']) for row in before]
    receipt = {'version': 'v89_public_prior_metadata_repair_receipt_v2',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'builder': bind(__file__), 'discovery_lineage_receipt': bind(discovery_path),
        'actual_public_contract_code': bind(BATCH / 'engine/ordering_generation.py'),
        'required_ordering_fields': list(REQUIRED), 'roots': roots,
        'outputs': outputs, 'ordering_questions_repaired': len(all_repairs),
        'missing_fields_before': dict(total_missing), 'missing_fields_after': {},
        'old_projections_unchanged': True, 'historical_writes': 0,
        'private_historical_question_rows_read_in_this_revision': False,
        'current_private_artifacts_read': False, 'gold_or_evidence_projected': False,
        'model_calls': 0, 'http_calls': 0, 'solver_calls': 0,
        'limitation': 'Fallback metadata proves public-text carrier lineage '
                      'only. It is not an original dialogue-source binding '
                      'and is never question/answer certification.'}
    write(HERE / 'PUBLIC_PRIOR_METADATA_REPAIR_RECEIPT_v2.json', receipt)
    print(json.dumps({k: receipt[k] for k in (
        'ordering_questions_repaired', 'missing_fields_before',
        'missing_fields_after', 'outputs')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
