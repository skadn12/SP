"""Seal completed descriptive artifacts without changing frozen production files."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]
SHARD = HOME / 'shards/shard_01'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def write(path, value):
    assert not path.exists(), f'append-only artifact already exists: {path}'
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    outputs = [HOME / name for name in ('PRESERVATION_AFTER.json', 'FINAL_AUDIT.json', 'COMPLETE_MANIFEST.json')]
    assert not any(path.exists() for path in outputs)
    second_reader = read(HOME / 'evidence/FINAL_REPORT_SECOND_READER.json')
    assert second_reader['status'] == 'PASS'
    independent = read(HOME / 'evidence/INDEPENDENT_RESULT_AUDIT.json')
    assert independent['status'] == 'PASS'
    qa = read(HOME / 'evidence/ACTUAL_QA_TRANSPORT_PRIOR_AUDIT.json')
    assert qa['transport_integrity_status'] == 'PASS' and qa['actual_http_attempts'] == 4
    before = read(HOME / 'PRESERVATION_BEFORE.json')
    mismatches = [row for row in before['files'] if not Path(row['path']).is_file() or sha(row['path']) != row['sha256']]
    assert len(before['files']) == before['protected_count'] == 767 and not mismatches
    write(outputs[0], {'status': 'PASS', 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
                       'protected_count': 767, 'unchanged_count': 767, 'mismatches': [],
                       'before': bind(HOME / 'PRESERVATION_BEFORE.json'),
                       'prior_delivery': before['prior_delivery'],
                       'scope': 'All 767 sealed v95 files including its manifest; older chained files were not redundantly rehashed here.'})
    pool = read(SHARD / 'EVALUATION_POOL.json')['questions']
    selected = read(SHARD / 'SELECTED_CANDIDATES.json')['questions']
    new = read(HOME / 'NEW_SELECTED_CANDIDATES.json')
    expected = [q for q in selected if q['question_type'] == 'ordering' or len(q['gold']) < 6]
    assert new['questions'] == expected and new['question_count'] == len(expected) == 8
    assert new['training_allowed'] is False and new['formal_training_admission'] is False
    assert Counter(q['question_type'] for q in expected) == Counter({'multi_select': 7, 'ordering': 1})
    assert Counter(len(q['gold']) for q in expected if q['question_type'] == 'multi_select') == Counter({2: 2, 3: 1, 4: 2, 5: 2})
    assert len(pool) == 11 and len(selected) == 10
    audit_paths = [HOME / name for name in ('COMPLETE_RESULTS_ZH.md', 'NEW_SELECTED_CANDIDATES.json',
                   'evidence/COMPLETE_STAGE_RESULTS.json', 'evidence/INDEPENDENT_RESULT_AUDIT.json',
                   'evidence/ACTUAL_QA_TRANSPORT_PRIOR_AUDIT.json', 'evidence/FINAL_REPORT_SECOND_READER.json',
                   'evidence/QUALITY_SYNTHESIS_ZH.md', 'PRESERVATION_AFTER.json')]
    audit_paths += [SHARD / name for name in ('BATCH_REVIEW.COMPLETE.json', 'EVALUATION_PREPARATION.COMPLETE.json',
                    'TRAINED_EVALUATION.COMPLETE.json', 'BASE_EVALUATION.COMPLETE.json', 'MIXED_EVALUATION_REPORT.json')]
    write(outputs[1], {'status': 'PASS', 'scope': 'Artifact integrity, source/QA transport evidence, independent raw-score replay, export projection and report consistency. Not certification of automatic semantic generation or large-scale reliability.',
                       'completed_at_utc': datetime.now(timezone.utc).isoformat(),
                       'generation_http_calls': 4, 'solver_calls_per_model': 192,
                       'solver_calls_total': 384, 'content_usable_including_controls': 11,
                       'selected_including_controls': 10, 'new_selected': 8,
                       'new_multiselect_by_gold': {'2': 2, '3': 1, '4': 2, '5': 2},
                       'new_ordering': 1, 'training_started': False, 'formal_training_admission': False,
                       'preserved_v95_files': 767,
                       'known_limits': ['Only two of six expansion roots independently READY.',
                                        'Final source reviews not completely blind; explicit disclosures preserved.',
                                        'Eight multiselect variants share two six-fact parent families.',
                                        'Ordering historical public question text priors empty in actual requests.',
                                        'Two ordering drafts violate name-only policy; one further source-support exclusion.',
                                        'One initial finish-review command failed on missing prerequisite private audit serialization; original log retained; no model retry or question/Gold edit.',
                                        'Existing report role label historical_six_supported_control means current fresh-parent Gold6 controls in this run.'],
                       'files': [bind(path) for path in audit_paths]})
    paths = sorted(path for path in HOME.rglob('*') if path.is_file() and '__pycache__' not in path.parts and path != outputs[2])
    manifest = {'status': 'SEALED', 'sealed_at_utc': datetime.now(timezone.utc).isoformat(),
                'listed_file_count_excluding_this_manifest': len(paths),
                'protected_file_count_including_this_manifest': len(paths) + 1,
                'excluded': ['__pycache__/**', 'COMPLETE_MANIFEST.json self-hash'],
                'files': [bind(path) for path in paths]}
    write(outputs[2], manifest)
    assert all(sha(row['path']) == row['sha256'] for row in manifest['files'])
    print(json.dumps({'status': 'SEALED', 'listed_files': len(paths), 'protected_total': len(paths)+1,
                      'manifest_sha256': sha(outputs[2]), 'v95_unchanged': 767}, ensure_ascii=False))


if __name__ == '__main__':
    main()
