"""Posthoc wording comparison; preserve old display positions and sampling seeds."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone

sys.dont_write_bytecode = True
BATCH = Path(__file__).resolve().parent.parent
PROJECT = BATCH.parent.parent
NATIVE = BATCH.parent / 'h4_feedback_training_pool_v15_qa/evaluation'
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(NATIVE))

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def bind(p): return {'path': str(Path(p).resolve()), 'sha256': sha(p)}
def save(p, x):
    with Path(p).open('x') as f:
        json.dump(x, f, ensure_ascii=False, indent=2); f.write('\n')
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    out = importlib.util.module_from_spec(spec); sys.modules[name] = out
    spec.loader.exec_module(out)
    return out
def verify(rows):
    for row in rows: assert sha(row['path']) == row['sha256'], row['path']
def native():
    n = module('v114_paired_native_runner', BATCH / 'tools/paired_native_runner.py')
    n.validate_pool = module('v114_protocol', BATCH / 'tools/adapter_protocol.py').validate_pool
    return n

def prepare():
    verify(read(BATCH / 'OLD_BATCH_PROTECTION.json')['files'])
    r = read(BATCH / 'reviews/REVISION_SOURCE_REVIEW.json')
    assert r['status'] == 'PASS' and r['errors'] == []
    verify(r['file_bindings'])
    draft = read(BATCH / 'REVISION_DRAFT.json')
    qs = draft['questions']; assert len(qs) == 2
    left, right = qs
    assert left['question'] == right['question'] and left['gold'] == right['gold']
    assert [k for k in left['options'] if left['options'][k] != right['options'][k]] == ['B']
    assert right['option_evidence']['B']['relation'] == 'contradicts'
    assert 'B' not in right['gold']
    save(BATCH / 'evaluation/EVALUATION_POOL.json', {
        'provenance': {'backend': 'CURRENT_CODEX_SESSION', 'source_batch_id': BATCH.name,
                       'posthoc_wording_comparison': True, 'exact_model_id': None},
        'questions': qs})
    original = (NATIVE / 'protocol.py').read_text()
    guard = 'if provenance.get("requested_model") != "Qwen/Qwen3.6-35B-A3B":'
    assert original.count(guard) == 1
    adapted = original.replace(guard, 'if (provenance.get("backend") != "CURRENT_CODEX_SESSION" or provenance.get("source_batch_id") != "h4_codex_wording_minfix_v114"):')
    line = 'CONTRACT_DIR = Path(__file__).resolve().parent.parent'
    assert adapted.count(line) == 1
    adapted = adapted.replace(line, 'CONTRACT_DIR = Path(' + repr(str(NATIVE.parent)) + ')')
    with (BATCH / 'tools/adapter_protocol.py').open('x') as f: f.write(adapted)
    original = (NATIVE / 'run_solver.py').read_text()
    here_line = 'HERE = Path(__file__).resolve().parent'
    assert original.count(here_line) == 1
    adapted = original.replace(here_line, 'HERE = Path(' + repr(str(NATIVE)) + ')')
    variant_line = 'return make_variants(question, seed=SEED, count=count), count'
    assert adapted.count(variant_line) == 1
    adapted = adapted.replace(variant_line,
        'paired_question = dict(question)\n'
        '    paired_question["candidate_id"] = question["original_candidate_id"]\n'
        '    return make_variants(paired_question, seed=SEED, count=count), count')
    seed_line = '"seed": SEED + len(requests), "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),'
    assert adapted.count(seed_line) == 1
    pair_path = repr(str(BATCH / 'PAIRING_INPUTS.json'))
    adapted = adapted.replace(seed_line,
        '"seed": read(Path(' + pair_path + '))["original_request_seeds"][variant["variant_id"] + "|" + str(slot)], "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),')
    with (BATCH / 'tools/paired_native_runner.py').open('x') as f: f.write(adapted)
    save(BATCH / 'tools/ADAPTER_CHANGES.json', {
        'native_runner': bind(NATIVE / 'run_solver.py'),
        'native_protocol': bind(NATIVE / 'protocol.py'),
        'changes': ['Native HERE points at original dependency directory.',
                    'Generator provenance guard permits this diagnostic batch.',
                    'Variants use original candidate ID as position-shuffle alias; content IDs still hash actual text.',
                    'Sampling seeds inherit matching original variant/slot, for both versions.'],
        'generation_decoding_model_scoring_selection_changed': False,
        'native_originals_modified': False})
    paths = [BATCH / name for name in ('PRECALL_PLAN.json', 'PAIRING_INPUTS.json',
              'REVISION_DRAFT.json', 'PUBLIC_REVISIONS.json', 'ASSET_REAUDIT_SUMMARY.json',
              'WORDING_RULE_ADDENDUM.md', 'OLD_BATCH_PROTECTION.json', 'evaluation/EVALUATION_POOL.json')]
    paths += list((BATCH / 'reviews').glob('*.json'))
    paths += list((BATCH / 'inputs').glob('*.json'))
    paths += list((BATCH / 'tools').glob('*.py')) + [BATCH / 'tools/ADAPTER_CHANGES.json']
    paths += list(NATIVE.glob('*.py')) + list(NATIVE.parent.glob('*.py'))
    save(BATCH / 'EXECUTION_FROZEN.json', {'created_at_utc': now(), 'files': [bind(p) for p in paths],
        'questions': 2, 'pairs': 1, 'calls_per_model': 36, 'calls_both_models': 72,
        'all_content_and_tools_frozen_before_new_solver_calls': True})
    native().validate_pool(BATCH / 'evaluation/EVALUATION_POOL.json')
    print('PREPARED', 2, 'questions', 36, 'calls/model')

def preflight():
    freeze = read(BATCH / 'EXECUTION_FROZEN.json'); verify(freeze['files'])
    n = native()
    for side, model in [('base', 'M0'), ('trained', 'M_clean_policy')]:
        n.preflight(BATCH / 'evaluation/EVALUATION_POOL.json', BATCH / 'evaluation/run' / side, solver_model=model)
    check_pairing()

def check_pairing():
    pair = read(BATCH / 'PAIRING_INPUTS.json')
    aliases = pair['candidate_aliases']; old = {r['variant']['variant_id'] + '|' + str(r['slot']): r for r in pair['original_requests']}
    manifests = {side: read(BATCH / 'evaluation/run' / side / 'manifest.json') for side in ['base', 'trained']}
    assert manifests['base']['requests'] == manifests['trained']['requests']
    run = BATCH / 'evaluation/run/base'
    questions = read(BATCH / 'evaluation/EVALUATION_POOL.json')['questions']
    qmap = {q['candidate_id']: q for q in questions}
    by_key = {}
    for r in manifests['base']['requests']:
        q = qmap[r['candidate_id']]; key = r['variant']['variant_id'] + '|' + str(r['slot']); prior = old[key]
        assert aliases[r['candidate_id']] == prior['candidate_id']
        assert r['seed'] == prior['seed']
        for label in 'ABCDEF':
            old_prop = prior['variant']['options'][label]
            canonical = next(k for k, v in questions[0]['options'].items() if v == old_prop)
            assert r['variant']['options'][label] == q['options'][canonical]
        prompt = (run / 'prompts' / (r['request_id'] + '.txt')).read_text()
        old_prompt = (Path(pair['original_manifest_path']).parent / 'prompts' / (prior['request_id'] + '.txt')).read_text()
        old_text = questions[0]['options']['B']; new_text = questions[1]['options']['B']
        assert prompt == (old_prompt if q['comparison_arm'] == 'original_replay' else old_prompt.replace(old_text, new_text))
        by_key.setdefault(key, []).append(r)
    assert len(by_key) == 18 and all(len(rs) == 2 and rs[0]['seed'] == rs[1]['seed'] for rs in by_key.values())
    for name in ['questions.json', 'gold.json', 'canonical_bindings.json']:
        assert read(BATCH / 'evaluation/run/base' / name) == read(BATCH / 'evaluation/run/trained' / name)
    paths = [p for p in (BATCH / 'evaluation/run').rglob('*') if p.is_file()]
    save(BATCH / 'evaluation/PAIRED_PREFLIGHT.json', {
        'status': 'PASS', 'created_at_utc': now(), 'pairs': 1, 'calls_per_model': 36,
        'same_seeds_display_positions_source_and_stem': True,
        'original_replay_prompts_identical_to_v113': True,
        'revised_prompts_differ_by_exactly_one_option_text': True,
        'max_prompt_tokens': max(r['prompt_tokens'] for r in manifests['base']['requests']),
        'files': [bind(p) for p in paths]})
    print('PAIRED_PREFLIGHT_PASS', 36)

def live(side):
    verify(read(BATCH / 'EXECUTION_FROZEN.json')['files'])
    verify(read(BATCH / 'evaluation/PAIRED_PREFLIGHT.json')['files'])
    save(BATCH / 'evaluation' / (side.upper() + '_STARTED.json'), {'started_at_utc': now(), 'side': side})
    n = native(); run = BATCH / 'evaluation/run' / side
    n.live(run); n.report(run)
    verify(read(BATCH / 'EXECUTION_FROZEN.json')['files'])
    verify(read(BATCH / 'evaluation/PAIRED_PREFLIGHT.json')['files'])
    save(BATCH / 'evaluation' / (side.upper() + '_COMPLETE.json'), {
        'completed_at_utc': now(), 'side': side,
        'actual_calls': read(run / 'COMPLETE.json')['actual_call_count'], 'training_started': False})
    print(side, 'COMPLETE', 36)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=['prepare', 'preflight', 'live'])
    p.add_argument('--side', choices=['base', 'trained']); a = p.parse_args()
    if a.phase == 'prepare': prepare()
    elif a.phase == 'preflight': preflight()
    else:
        assert a.side; live(a.side)
