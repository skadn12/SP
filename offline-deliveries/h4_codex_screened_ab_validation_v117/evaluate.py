"""Evaluate sealed A3+B3 content with the unchanged native v15 Solver runner."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
NATIVE = HERE.parent / 'h4_feedback_training_pool_v15_qa/evaluation'
sys.path.insert(0, str(PROJECT / 'scripts'))
sys.path.insert(0, str(NATIVE))
sys.path.insert(0, str(HERE))
from codex_sequential_ab import inspect_root, load_config
from codex_evidence_export import export_root_questions
from prepare_path_adapter_v2 import normalize_questions
from source_review import verify_batch_gate, verify_scene_constraints

def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f: json.dump(x, f, ensure_ascii=False, indent=2); f.write('\n')
def bind(p): return {'path': str(Path(p).resolve()), 'sha256': sha(p)}
def verify(rows):
    for r in rows:
        if sha(r['path']) != r['sha256']: raise ValueError('BOUND_FILE_CHANGED:' + r['path'])
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m
def verify_tooling():
    frozen = read(HERE / 'evidence/TOOLING_FREEZE.json')
    assert frozen['status'] == 'FROZEN_BEFORE_QUESTION_AUTHORING'
    expected = {str((HERE / name).resolve()) for name in
                ('evaluate.py', 'audit_solver.py', 'deliver.py', 'source_review.py', 'prepare_path_adapter_v2.py')}
    expected.add(str((PROJECT / 'scripts/codex_evidence_export.py').resolve()))
    assert expected.issubset({row['path'] for row in frozen['files']}), 'TOOLING_FREEZE_INCOMPLETE'
    verify(frozen['files'])
    return frozen

def native():
    n = module('v117_native_runner', NATIVE / 'run_solver.py')
    n.validate_pool = module('v117_protocol', HERE / 'evaluation/adapter_protocol.py').validate_pool
    return n

def prepare():
    verify_tooling()
    cfg = load_config(HERE)
    questions = []; bindings = verify_batch_gate(cfg)
    for row in cfg['roots']:
        rid = row['root_id']; folder = HERE / 'roots' / ('root' + rid.rsplit('-', 1)[1])
        state = inspect_root(HERE, cfg, rid)
        for review_path in sorted((folder / 'scenes').glob('*/SOURCE_REVIEW.json')):
            review = read(review_path)
            verify_scene_constraints(rid, review['scene_ref'], review)
        if not state['full_expansion_complete']: continue
        src = folder / 'CUMULATIVE_SOURCE.json'
        final = folder / 'scenes/B_S03/CUMULATIVE_SOURCE.json'
        assert src.read_bytes() == final.read_bytes()
        questions.extend(export_root_questions(folder))
        for f in folder.rglob('*.json'): bindings.append(bind(f))
    assert questions, 'NO_CONTENT_PASS_QUESTIONS'
    questions, path_changes = normalize_questions(questions)
    cap = read(HERE / 'PRECALL_PLAN.json')['solver_calls_per_model_max']
    count = sum(18 if q['question_type'] == 'multi_select' else 12 for q in questions)
    assert count <= cap
    original = (NATIVE / 'protocol.py').read_text()
    guard = 'if provenance.get("requested_model") != "Qwen/Qwen3.6-35B-A3B":'
    replacement = ('if (provenance.get("backend") != "CURRENT_CODEX_SESSION" '
                   'or provenance.get("source_batch_id") != "h4_codex_screened_ab_validation_v117"):')
    assert original.count(guard) == 1
    original = original.replace(guard, replacement)
    old = 'CONTRACT_DIR = Path(__file__).resolve().parent.parent'
    assert original.count(old) == 1
    adapted = original.replace(old, 'CONTRACT_DIR = Path(' + repr(str(NATIVE.parent)) + ')')
    target = HERE / 'evaluation/adapter_protocol.py'; target.parent.mkdir(exist_ok=True)
    with target.open('x') as f: f.write(adapted)
    save(HERE / 'evaluation/EVALUATION_POOL.json', {'status': 'SEALED_CONTENT_READY_FOR_SOLVER_EVALUATION',
         'provenance': {'backend': 'CURRENT_CODEX_SESSION', 'source_batch_id': HERE.name,
                        'exact_model_id': None, 'same_inherited_model_and_shared_background': True},
         'questions': questions})
    path_receipt = HERE / 'evaluation/PREPARE_PATH_ADAPTER_RECEIPT.json'
    save(path_receipt, {'status': 'PATH_REPRESENTATION_CHECKS_PASS', 'created_at_utc': now(),
         'mode': 'PROSPECTIVE_BEFORE_FIRST_NATIVE_VALIDATION', 'solver_calls': 0,
         'candidate_count': len(questions), 'changes': path_changes,
         'changed_candidate_fields': ['source_path'],
         'source_question_gold_evidence_scoring_changed': False,
         'prior_failed_prepare_claimed': False,
         'adapter_binding': bind(HERE / 'prepare_path_adapter_v2.py')})
    for f in (HERE / 'PRECALL_PLAN.json', HERE / 'RUN_CONFIG.json', HERE / 'SOURCE_RESERVATIONS.json',
              HERE / 'EXECUTION_INPUTS_FROZEN.json', HERE / 'EXPOSURE_AUDIT.json',
              HERE / 'evidence/CONTENT_CONTRACT.md', Path(__file__), target,
              HERE / 'evaluation/EVALUATION_POOL.json', PROJECT / 'scripts/codex_sequential_ab.py',
              PROJECT / 'scripts/codex_sequential_questions.py', PROJECT / 'scripts/codex_scene_payload.py', PROJECT / 'scripts/codex_evidence_export.py',
              HERE / 'evidence/TOOLING_FREEZE.json', HERE / 'source_review.py',
              HERE / 'prepare_path_adapter_v2.py', path_receipt):
        bindings.append(bind(f))
    bindings += [bind(f) for f in NATIVE.glob('*.py')]
    bindings += [bind(f) for f in NATIVE.parent.glob('*.py')]
    save(HERE / 'evaluation/ADAPTER_BINDINGS.json', {'created_at_utc': now(), 'files': bindings,
         'candidate_count': len(questions), 'calls_per_model': count, 'training_started': False,
         'changes': ['Generator provenance guard accepts this Codex batch.',
                     'Contract directory points to unchanged native mechanical gates.',
                     'Only source_path representation normalized prospectively before first native validation.'],
         'path_representation_adapter': {'version': 2, 'mode': 'PROSPECTIVE',
                                        'receipt': bind(path_receipt), 'prior_failed_prepare_claimed': False},
         'native_generation_decoding_permutations_scoring_selection_changed': False})
    native().validate_pool(HERE / 'evaluation/EVALUATION_POOL.json')
    print(json.dumps({'status': 'EVALUATION_READY', 'questions': len(questions), 'calls_per_model': count}))

def preflight():
    verify_tooling()
    b = read(HERE / 'evaluation/ADAPTER_BINDINGS.json'); verify(b['files'])
    n = native()
    for side, model in (('base', 'M0'), ('trained', 'M_clean_policy')):
        n.preflight(HERE / 'evaluation/EVALUATION_POOL.json', HERE / 'evaluation/run' / side, solver_model=model)
    paths = [f for f in (HERE / 'evaluation/run').rglob('*') if f.is_file()]
    manifests = [read(HERE / 'evaluation/run' / s / 'manifest.json') for s in ('base', 'trained')]
    assert all(m['request_count'] == b['calls_per_model'] for m in manifests)
    for name in ('questions.json', 'gold.json'):
        assert read(HERE / 'evaluation/run/base' / name) == read(HERE / 'evaluation/run/trained' / name)
    for m in manifests:
        assert max(r['prompt_tokens'] for r in m['requests']) < 16384
    save(HERE / 'evaluation/PAIRED_PREFLIGHT.json', {'status': 'PASS', 'created_at_utc': now(),
         'files': [bind(p) for p in paths], 'adapter_binding': bind(HERE / 'evaluation/ADAPTER_BINDINGS.json'),
         'candidate_count': b['candidate_count'], 'calls_per_model': b['calls_per_model'],
         'max_prompt_tokens': max(r['prompt_tokens'] for m in manifests for r in m['requests']),
         'training_started': False})
    print('PAIRED_PREFLIGHT_PASS', b['calls_per_model'])

def live(side):
    verify_tooling()
    b = read(HERE / 'evaluation/ADAPTER_BINDINGS.json'); verify(b['files'])
    pre = read(HERE / 'evaluation/PAIRED_PREFLIGHT.json'); verify(pre['files']); verify([pre['adapter_binding']])
    save(HERE / 'evaluation' / (side.upper() + '_STARTED.json'), {'started_at_utc': now(), 'side': side})
    n = native(); run = HERE / 'evaluation/run' / side
    n.live(run); n.report(run)
    verify(pre['files']); verify([pre['adapter_binding']])
    verify(b['files'])
    save(HERE / 'evaluation' / (side.upper() + '_COMPLETE.json'), {'completed_at_utc': now(), 'side': side,
         'actual_calls': read(run / 'COMPLETE.json')['actual_call_count'], 'training_started': False})

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=['prepare', 'preflight', 'live'])
    p.add_argument('--side', choices=['base', 'trained']); args = p.parse_args()
    if args.phase == 'prepare': prepare()
    elif args.phase == 'preflight': preflight()
    else:
        assert args.side; live(args.side)
