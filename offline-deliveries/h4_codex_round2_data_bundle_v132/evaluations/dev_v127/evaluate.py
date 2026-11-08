"""Additive v124-dev / M_clean vs M_v125 adapter for the frozen v15 evaluator."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
NATIVE = REPO / 'offline-deliveries/h4_feedback_training_pool_v15_qa/evaluation'
DEV = REPO / 'offline-deliveries/h4_codex_dev_test_ab_v124'
TRAIN = REPO / 'offline-deliveries/h4_codex_train_pool_v125'
TRAINING = REPO / 'offline-deliveries/h4_codex_training_v126'
NEW_PATH = REPO / 'restart-n5-runs/h4_codex_combined_sft_v125_merged'
POLICY = 'v15_full_only_random_mcq6x3_order12x1_v1'
MODELS = {'baseline': 'M_clean_policy', 'new': 'M_v125_newonly'}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    p = Path(path).resolve()
    return {'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size}


def verify(rows):
    for row in rows:
        path = Path(row['path'])
        require(path.is_file() and sha(path) == row['sha256'], 'BOUND_FILE_CHANGED:' + str(path))


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def protocol_text():
    text = (NATIVE / 'protocol.py').read_text()
    old = 'if provenance.get("requested_model") != "Qwen/Qwen3.6-35B-A3B":'
    loc = 'CONTRACT_DIR = Path(__file__).resolve().parent.parent'
    require(text.count(old) == text.count(loc) == 1, 'NATIVE_PROTOCOL_CHANGED')
    return text.replace(old, 'if (provenance.get("backend") != "CURRENT_CODEX_SESSION" or provenance.get("source_batch_id") != "h4_codex_dev_test_ab_v124"):').replace(loc, 'CONTRACT_DIR = Path(' + repr(str(NATIVE.parent)) + ')')


def new_identity():
    identity = read(HERE / 'model_tools/NEW_MODEL_IDENTITY.json')
    require(identity['model_identifier'] == MODELS['new'] and identity['model_path'] == str(NEW_PATH), 'NEW_MODEL_IDENTITY_WRONG_PATH_OR_NAME')
    require(identity['tokenizer_path'] == '/root/autodl-tmp/Qwen3-4B', 'NEW_TOKENIZER_CHANGED')
    require(identity['adapter_path'] == str(REPO / 'restart-n5-runs/h4_codex_combined_sft_v125/adapter'), 'NEW_IDENTITY_USES_WRONG_ADAPTER')
    require(identity['adapter_enabled'] is True and identity['adapter_already_merged'] is True, 'NEW_MODEL_NOT_MERGED')
    require(identity['checkpoint_files'], 'NEW_MODEL_CHECKPOINT_BINDINGS_MISSING')
    require(sha(identity['audit_path']) == identity['audit_sha256'], 'NEW_MODEL_AUDIT_CHANGED')
    return identity


def native():
    sys.path.insert(0, str(NATIVE))
    obj = module('v127_native_v15_runner', NATIVE / 'run_solver.py')
    require(obj.POLICY_VERSION == POLICY and obj.SEED == 2026100800, 'NATIVE_POLICY_CHANGED')
    require((HERE / 'adapter_protocol.py').read_text() == protocol_text(), 'PROVENANCE_ADAPTER_CHANGED')
    obj.validate_pool = module('v127_dev_protocol', HERE / 'adapter_protocol.py').validate_pool
    original_identity, original_validate = obj.model_identity, obj._validate_model
    obj.MODEL_PATHS[MODELS['new']] = NEW_PATH

    def identity(model='M_clean_policy'):
        return new_identity() if model == MODELS['new'] else original_identity(model)

    def validate(value):
        if value.get('model_identifier') != MODELS['new']:
            return original_validate(value)
        require(value == new_identity(), 'NEW_IDENTITY_DOES_NOT_MATCH_FIXED_AUDIT')
        for row in value['checkpoint_files']:
            path = Path(row['path'])
            require(path.is_file() and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'NEW_CHECKPOINT_CHANGED:' + str(path))

    obj.model_identity, obj._validate_model = identity, validate
    return obj


def prepare():
    require(not (HERE / 'SCOPE.json').exists() and not (HERE / 'run').exists(), 'EVALUATION_ALREADY_PREPARED')
    require(read(TRAINING / 'FINAL_STATUS.json')['status'] == 'TRAINED_AND_AUDITED_AWAITING_DEV_EVALUATION', 'TRAINING_NOT_COMPLETE_AND_AUDITED')
    # Verify predecessor seals. Only dev dialogue/question contents are read.
    for base in [TRAIN, TRAINING, REPO / 'restart-n5-runs/h4_codex_combined_sft_v125']:
        for row in read(base / 'COMPLETE_MANIFEST.json')['files']:
            require(sha(base / row['path']) == row['sha256'], 'PREDECESSOR_SEAL_CHANGED:' + str(base / row['path']))
    manifest = read(DEV / 'MANIFEST.json')
    fixed = [r['root_id'] for r in manifest['roots'] if r['split'] == 'dev']
    original_path = DEV / 'delivery/dev/QUESTIONS_WITH_ANSWERS.json'
    questions = read(original_path)['questions']
    require(len(fixed) == 15 and len(questions) == 48, 'FIXED_DEV_POPULATION_CHANGED')
    covered = sorted({q['root_id'] for q in questions})
    require(len(covered) == 7 and set(covered) <= set(fixed), 'COVERED_DEV_ROOTS_CHANGED')
    require(len({q['candidate_id'] for q in questions}) == 48, 'DUPLICATE_DEV_CANDIDATE')
    require(sum(q['question_type'] == 'multi_select' for q in questions) == 34, 'DEV_TYPE_COUNTS_CHANGED')
    train_roots = {r['root_id'] for r in read(TRAIN / 'data/ROOT_MANIFEST.json')}
    require(train_roots.isdisjoint(fixed), 'TRAIN_DEV_ROOT_OVERLAP')
    require(all(q['split'] == 'dev' and q['evaluation_only'] is True for q in questions), 'NON_DEV_QUESTION')
    for q in questions:
        require(sha(q['source_path']) == q['source_sha256'], 'DEV_SOURCE_CHANGED')
    # The exact pool must remain a lossless copy, including Gold and evidence.
    save(HERE / 'EVALUATION_POOL.json', {'status': 'FROZEN_DEV_ONLY_NO_SELECTION', 'provenance': {
        'backend': 'CURRENT_CODEX_SESSION', 'source_batch_id': 'h4_codex_dev_test_ab_v124',
        'exact_model_id': None, 'same_inherited_model': True}, 'questions': copy.deepcopy(questions)})
    (HERE / 'adapter_protocol.py').write_text(protocol_text())
    save(HERE / 'SCOPE.json', {'schema_version': 'v127_fixed_dev_pair_v1', 'fixed_dev_root_ids': fixed,
        'covered_dev_root_ids': covered, 'question_count': 48, 'question_type_counts': {'multi_select': 34, 'ordering': 14},
        'gold2345_multi_select_count': 27, 'gold6_control_count': 7, 'calls_per_model': 780, 'total_calls': 1560,
        'dev_binding': bind(original_path), 'heldout_manifest_binding': bind(DEV / 'MANIFEST.json'),
        'dev_test_complete_manifest_binding': bind(DEV / 'COMPLETE_MANIFEST.json'),
        'training_package_binding': bind(TRAIN / 'COMPLETE_MANIFEST.json'), 'training_run_binding': bind(TRAINING / 'COMPLETE_MANIFEST.json'),
        'models': MODELS, 'policy_version': POLICY, 'seed': 2026100800,
        'test_evaluation': False, 'score_based_filtering': False,
        'selection_diagnostics': 'Native selected_questions retained only as diagnostic; no heldout records filtered.',
        'population_limit': 'Fixed15 dev roots; scores cover7 roots with48 delivered questions. Correlated permutations/rollouts are not independent data.'})
    runner = native()
    runner.validate_pool(HERE / 'EVALUATION_POOL.json')
    for side, model in MODELS.items():
        runner.preflight(HERE / 'EVALUATION_POOL.json', HERE / 'run' / side, solver_model=model)
    manifests = [read(HERE / 'run' / side / 'manifest.json') for side in MODELS]
    require(all(m['request_count'] == 780 and m['candidate_count'] == 48 for m in manifests), 'REQUEST_COUNT_MISMATCH')
    require(manifests[0]['requests'] == manifests[1]['requests'] and manifests[0]['decoder_contract'] == manifests[1]['decoder_contract'], 'PAIRED_REQUESTS_OR_DECODER_DIFFER')
    require(len({r['prompt_sha256'] for r in manifests[0]['requests']}) == 372, 'UNIQUE_PROMPTS_CHANGED')
    require(max(r['prompt_tokens'] for r in manifests[0]['requests']) == 8607, 'PROMPT_LENGTH_PLAN_CHANGED')
    for name in ['questions.json', 'gold.json', 'canonical_bindings.json']:
        require(read(HERE / 'run/baseline' / name) == read(HERE / 'run/new' / name), 'PAIRED_CANONICAL_INPUTS_DIFFER:' + name)
    # Include dynamically imported native backend/contract code in tool freeze.
    paths = {Path(__file__), HERE / 'run_pair.py', HERE / 'adapter_protocol.py', HERE / 'report_paired.py', runner.BASE_RUNNER, *NATIVE.glob('*.py'), *NATIVE.parent.glob('*.py')}
    for mod in list(sys.modules.values()):
        filename = getattr(mod, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == '.py' and path.is_relative_to(REPO / 'worktrees/h4-gate-downstream-v1'):
                paths.add(path)
    save(HERE / 'TOOLS_FREEZE.json', {'schema_version': 'v127_pre_inference_tools_v1', 'status': 'FROZEN_BEFORE_FIRST_INFERENCE',
        'created_at_utc': now(), 'files': [bind(p) for p in sorted(paths)], 'model_identity_adapter': 'Explicit new audited model identity; native baseline identity preserved.',
        'provenance_adapter': 'Only generator check and native gate directory path differ from v15 protocol.',
        'policy_version': POLICY, 'seed': 2026100800, 'native_algorithm_changed': False})
    inputs = [HERE / 'SCOPE.json', HERE / 'EVALUATION_POOL.json', HERE / 'TOOLS_FREEZE.json', HERE / 'model_tools/NEW_MODEL_IDENTITY.json', HERE / 'evidence/MODEL_MERGE_AUDIT.json']
    for side in MODELS:
        inputs.extend(p for p in (HERE / 'run' / side).rglob('*') if p.is_file())
    inputs.extend([original_path, DEV / 'MANIFEST.json', DEV / 'COMPLETE_MANIFEST.json', TRAIN / 'COMPLETE_MANIFEST.json', TRAINING / 'COMPLETE_MANIFEST.json'])
    inputs.extend(Path(q['source_path']) for q in questions)
    save(HERE / 'PREFLIGHT_FREEZE.json', {'status': 'PAIRED_PREFLIGHT_READY_FOR_INDEPENDENT_AUDIT', 'files': [bind(p) for p in sorted(set(inputs))],
        'candidate_count': 48, 'request_count_per_model': 780, 'maximum_prompt_tokens': 8607, 'unique_prompts_per_model': 372,
        'automatic_retry': False, 'test_evaluation': False, 'full_only': True})
    print(json.dumps({'status': 'PAIRED_PREFLIGHT_PREPARED', 'calls_per_model': 780, 'questions': 48}))


def frozen():
    verify(read(HERE / 'TOOLS_FREEZE.json')['files'])
    verify(read(HERE / 'PREFLIGHT_FREEZE.json')['files'])
    report = read(HERE / 'evidence/INDEPENDENT_PREFLIGHT_AUDIT.json')
    require(report['status'] == 'PASS', 'INDEPENDENT_PREFLIGHT_NOT_PASS')
    verify(report['audited_files'])


def live(side):
    frozen()
    runner = native()
    require(Path(sys.prefix) == runner.EXPECTED_VENV, 'USE_PROVEN_LOCAL4B_VENV')
    # Native live executes each frozen request once with structured decoding.
    runner.live(HERE / 'run' / side)
    runner.report(HERE / 'run' / side)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='phase', required=True)
    sub.add_parser('prepare')
    run = sub.add_parser('live')
    run.add_argument('--side', choices=MODELS, required=True)
    sub.add_parser('verify')
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare()
    elif args.phase == 'live':
        live(args.side)
    else:
        frozen()
        print('PASS: frozen paired evaluation inputs, tools and independent preflight')


if __name__ == '__main__':
    main()
