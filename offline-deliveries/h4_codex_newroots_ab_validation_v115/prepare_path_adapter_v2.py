"""Preserve failed preparation and normalize only evaluation source paths.

The immutable authoring files and the original frozen wrappers stay unchanged.
No Solver request or semantic correction is performed here.
"""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p): return {'path': str(Path(p).resolve()), 'sha256': sha(p)}
def save(p, value):
    with Path(p).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')
def now(): return datetime.now(timezone.utc).isoformat()

def main():
    freeze_path = HERE/'evidence/PATH_ADAPTER_TOOLING_FREEZE.json'
    freeze = read(freeze_path)
    assert freeze['status'] == 'FROZEN_BEFORE_PATH_NORMALIZATION_AND_SOLVER'
    for row in freeze['files']:
        assert sha(row['path']) == row['sha256']
    assert not list((HERE/'evaluation').glob('*_STARTED.json'))
    assert not (HERE/'evaluation/run').exists()
    spec = importlib.util.spec_from_file_location('v115_original_evaluate', HERE/'evaluate.py')
    evaluation = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = evaluation
    spec.loader.exec_module(evaluation)
    evaluation.verify_tooling()
    original_pool = HERE/'evaluation/EVALUATION_POOL.json'
    original_bindings = HERE/'evaluation/ADAPTER_BINDINGS.json'
    bindings = read(original_bindings)
    evaluation.verify(bindings['files'])
    pool = read(original_pool)
    normalized = copy.deepcopy(pool)
    changes = []
    for old, new in zip(pool['questions'], normalized['questions']):
        source = Path(old['source_path'])
        resolved = source if source.is_absolute() else PROJECT/source
        resolved = resolved.resolve()
        assert resolved.is_file() and sha(resolved) == old['source_sha256']
        new['source_path'] = str(resolved)
        assert {k:v for k,v in old.items() if k != 'source_path'} == {k:v for k,v in new.items() if k != 'source_path'}
        changes.append({'candidate_id': old['candidate_id'], 'before': old['source_path'], 'after': new['source_path'], 'source_sha256': old['source_sha256']})
    candidate = HERE/'evaluation/PATH_NORMALIZED_POOL_CANDIDATE.json'
    save(candidate, normalized)
    evaluation.native().validate_pool(candidate)
    archive = HERE/'evaluation/failed_prepare_v1'
    archive.mkdir(exist_ok=False)
    old_pool_hash = sha(original_pool)
    old_bindings_hash = sha(original_bindings)
    original_pool.rename(archive/original_pool.name)
    original_bindings.rename(archive/original_bindings.name)
    assert sha(archive/original_pool.name) == old_pool_hash
    assert sha(archive/original_bindings.name) == old_bindings_hash
    save(original_pool, normalized)
    new_bindings = copy.deepcopy(bindings)
    replacements = [row for row in new_bindings['files'] if row['path'] == str(original_pool)]
    assert len(replacements) == 1
    replacements[0]['sha256'] = sha(original_pool)
    new_bindings['files'] += [bind(Path(__file__)), bind(freeze_path)]
    new_bindings['path_representation_adapter'] = {
        'version': 2, 'original_prepare_failed_before_solver': True,
        'changed_candidate_fields': ['source_path'],
        'original_pool': bind(archive/'EVALUATION_POOL.json'),
        'original_bindings': bind(archive/'ADAPTER_BINDINGS.json'),
        'freeze': bind(freeze_path), 'created_at_utc': now()}
    save(original_bindings, new_bindings)
    evaluation.verify(new_bindings['files'])
    evaluation.native().validate_pool(original_pool)
    save(HERE/'evaluation/PREPARE_PATH_ADAPTER_RECEIPT.json', {
        'status': 'PASS', 'completed_at_utc': now(), 'solver_calls': 0,
        'first_prepare_error': 'ValueError: SOURCE_PATH_OR_SHA_INVALID:h4_codex_newroots_ab_validation_v115-ms-605e70e7c92958a4',
        'cause': 'Native protocol requires absolute source_path; authoring controller records a project-relative path. All files and content hashes already match.',
        'changes': changes, 'candidate_count': len(normalized['questions']),
        'source_question_gold_evidence_scoring_changed': False,
        'original_11_tooling_bindings_unchanged': True,
        'input_bindings': [bind(archive/'EVALUATION_POOL.json'), bind(archive/'ADAPTER_BINDINGS.json'), bind(original_pool), bind(original_bindings), bind(candidate), bind(Path(__file__)), bind(freeze_path)]})
    print(json.dumps({'status': 'PREPARE_PATH_NORMALIZATION_PASS', 'questions': len(normalized['questions']), 'calls_per_model': bindings['calls_per_model'], 'solver_calls': 0}))

if __name__ == '__main__': main()
