"""Per-wave paired evaluation using the unchanged native v15 runner."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
B = Path(__file__).resolve().parents[1]
P = B.parent.parent
NATIVE = B.parent / 'h4_feedback_training_pool_v15_qa/evaluation'
sys.path.insert(0, str(P / 'scripts'))
sys.path.insert(0, str(NATIVE))
from codex_sequential_ab import inspect_root, load_config
from codex_evidence_export import export_root_questions


def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p): return {'path': str(Path(p).resolve()), 'sha256': sha(p)}
def save(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f: json.dump(d, f, ensure_ascii=False, indent=2); f.write('\n')
def verify(rows):
    for row in rows: assert sha(row['path']) == row['sha256'], row['path']
def module(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    obj = importlib.util.module_from_spec(spec); sys.modules[name] = obj; spec.loader.exec_module(obj)
    return obj
def native(scope):
    obj = module('v118_native_runner', NATIVE / 'run_solver.py')
    obj.validate_pool = module('v118_native_protocol', scope / 'adapter_protocol.py').validate_pool
    return obj


def tooling():
    frozen = read(B / 'evidence/EVALUATION_TOOL_BINDINGS.json')
    verify(frozen['files']); verify(read(B / 'MASTER_FREEZE.json')['files'])
    return frozen


def prepare(wave):
    tooling()
    manifest = read(B / 'MANIFEST.json')
    roots = [r for r in manifest['roots'] if r['wave'] == wave and r['action'] in ['SCREEN', 'INHERIT_SCREEN']]
    assert roots
    scope = B / 'evaluations' / wave
    questions, terminal, bindings = [], [], [bind(B / 'MANIFEST.json'), bind(B / 'MASTER_FREEZE.json'), bind(B / 'ROLE_ASSIGNMENTS.json')]
    for row in roots:
        unit = Path(row['unit_path']); cp = unit / 'inputs/CONTINUATION_CONSTRAINTS.json'
        constraints = read(cp); bindings.append(bind(cp))
        if constraints['decision'] == 'HOLD':
            terminal.append({'root_id': row['root_id'], 'status': 'BASELINE_HOLD'}); continue
        if (unit / 'MATCHING_HOLD.json').exists():
            terminal.append({'root_id': row['root_id'], 'status': 'MATCHING_HOLD'})
            bindings.append(bind(unit / 'MATCHING_HOLD.json')); continue
        verify(read(unit / 'EXECUTION_INPUTS_FROZEN.json')['files'])
        cfg = load_config(unit)
        state = inspect_root(unit, cfg, row['root_id'])
        folder = unit / 'roots' / ('root' + row['root_id'].rsplit('-', 1)[1])
        if state['root_stopped']:
            terminal.append({'root_id': row['root_id'], 'status': 'EXPANSION_STOPPED', 'ready_scenes': state['ready_scene_count']})
        else:
            assert state['full_expansion_complete'], 'WAIT_FOR_SIX_SCENES:' + row['root_id']
            if (folder / 'QA_AUTHOR_FAILURE.json').exists():
                terminal.append({'root_id': row['root_id'], 'status': 'QA_FIRST_ATTEMPT_FAILED'})
            else:
                cross = read(folder / 'FINAL_CROSSCHECK.json')
                assert cross['private_sha256'] == sha(folder / 'PRIVATE_GOLD.json')
                exported = export_root_questions(folder)
                for q in exported:
                    path = Path(q['source_path']).resolve()
                    assert sha(path) == q['source_sha256']
                    # Absolute path spelling only; source, question and Gold unchanged.
                    q['source_path'] = str(path)
                questions.extend(exported)
                terminal.append({'root_id': row['root_id'], 'status': 'CONTENT_REVIEW_COMPLETED', 'content_pass': len(exported)})
        bindings.extend(bind(p) for p in unit.rglob('*.json'))
    assert len(terminal) == len(roots)
    original = (NATIVE / 'protocol.py').read_text()
    guard = 'if provenance.get("requested_model") != "Qwen/Qwen3.6-35B-A3B":'
    assert original.count(guard) == 1
    original = original.replace(guard, 'if (provenance.get("backend") != "CURRENT_CODEX_SESSION" or provenance.get("source_batch_id") != ' + repr(B.name) + '):')
    old = 'CONTRACT_DIR = Path(__file__).resolve().parent.parent'
    assert original.count(old) == 1
    original = original.replace(old, 'CONTRACT_DIR = Path(' + repr(str(NATIVE.parent)) + ')')
    scope.mkdir(parents=True, exist_ok=True)
    with (scope / 'adapter_protocol.py').open('x') as f: f.write(original)
    save(scope / 'EVALUATION_POOL.json', {'status': 'CONTENT_FROZEN_FOR_PAIRED_SOLVER', 'provenance': {'backend': 'CURRENT_CODEX_SESSION', 'source_batch_id': B.name, 'exact_model_id': None, 'same_inherited_model': True}, 'questions': questions})
    bindings += [bind(p) for p in NATIVE.glob('*.py')] + [bind(p) for p in NATIVE.parent.glob('*.py')]
    bindings += [bind(__file__), bind(scope / 'adapter_protocol.py'), bind(scope / 'EVALUATION_POOL.json'), bind(B / 'evidence/EVALUATION_TOOL_BINDINGS.json')]
    count = sum(18 if q['question_type'] == 'multi_select' else 12 for q in questions)
    assert count <= len(roots) * 114
    save(scope / 'ADAPTER_BINDINGS.json', {'created_at_utc': now(), 'wave': wave, 'files': bindings, 'terminal_roots': terminal, 'candidate_count': len(questions), 'calls_per_model': count, 'native_decoding_permutations_scoring_selection_unchanged': True, 'training_started': False})
    if questions: native(scope).validate_pool(scope / 'EVALUATION_POOL.json')
    print(json.dumps({'status': 'EVALUATION_READY' if questions else 'EMPTY_CONTENT_POOL', 'wave': wave, 'roots': len(roots), 'questions': len(questions), 'calls_per_model': count}))


def preflight(wave):
    tooling(); scope = B / 'evaluations' / wave
    binding = read(scope / 'ADAPTER_BINDINGS.json'); verify(binding['files'])
    assert binding['candidate_count'] > 0
    obj = native(scope)
    for side, model in [('base', 'M0'), ('trained', 'M_clean_policy')]:
        obj.preflight(scope / 'EVALUATION_POOL.json', scope / 'run' / side, solver_model=model)
    manifests = [read(scope / 'run' / side / 'manifest.json') for side in ['base', 'trained']]
    assert all(m['request_count'] == binding['calls_per_model'] for m in manifests)
    for n in ['questions.json', 'gold.json']: assert read(scope / 'run/base' / n) == read(scope / 'run/trained' / n)
    maximum = max(r['prompt_tokens'] for m in manifests for r in m['requests'])
    assert maximum < 16384
    files = [bind(p) for p in (scope / 'run').rglob('*') if p.is_file()]
    save(scope / 'PAIRED_PREFLIGHT.json', {'status': 'PASS', 'created_at_utc': now(), 'files': files, 'adapter_binding': bind(scope / 'ADAPTER_BINDINGS.json'), 'candidate_count': binding['candidate_count'], 'calls_per_model': binding['calls_per_model'], 'max_prompt_tokens': maximum, 'training_started': False})
    print(json.dumps({'status': 'PAIRED_PREFLIGHT_PASS', 'wave': wave, 'max_prompt_tokens': maximum, 'calls_per_model': binding['calls_per_model']}))


def live(wave, side):
    tooling(); scope = B / 'evaluations' / wave
    adapter = read(scope / 'ADAPTER_BINDINGS.json'); verify(adapter['files'])
    pre = read(scope / 'PAIRED_PREFLIGHT.json'); verify(pre['files']); verify([pre['adapter_binding']])
    save(scope / (side.upper() + '_STARTED.json'), {'started_at_utc': now(), 'side': side, 'automatic_retry': False})
    obj = native(scope); run = scope / 'run' / side
    obj.live(run); obj.report(run)
    verify(pre['files']); verify(adapter['files'])
    save(scope / (side.upper() + '_COMPLETE.json'), {'completed_at_utc': now(), 'side': side, 'actual_calls': read(run / 'COMPLETE.json')['actual_call_count'], 'training_started': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'preflight', 'live'])
    p.add_argument('--wave', choices=['validation', 'production'], required=True)
    p.add_argument('--side', choices=['base', 'trained'])
    a = p.parse_args()
    if a.action == 'prepare': prepare(a.wave)
    elif a.action == 'preflight': preflight(a.wave)
    else:
        assert a.side; live(a.wave, a.side)
