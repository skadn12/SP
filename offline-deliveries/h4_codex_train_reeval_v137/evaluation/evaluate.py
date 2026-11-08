"""Paired evaluation of all 363 actually trained questions; training fit, not heldout generalization.

Only ``live`` starts inference. Prepare is tokenizer/CPU only. Old experiments,
model audits, native permutation/scoring code and QA records remain immutable.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BATCH = HERE.parent
REPO = BATCH.parent.parent
NATIVE = REPO / 'offline-deliveries/h4_feedback_training_pool_v15_qa/evaluation'
V127 = REPO / 'offline-deliveries/h4_codex_dev_eval_v127'
V124 = REPO / 'offline-deliveries/h4_codex_dev_test_ab_v124'
V133 = REPO / 'offline-deliveries/h4_codex_held_roots_retry_v133'
TRAIN_PACKAGE = REPO / 'offline-deliveries/h4_codex_train_pool_v125'
TRAIN_ROOTS = TRAIN_PACKAGE / 'data/ROOT_MANIFEST.json'
POLICY = 'v15_full_only_random_mcq6x3_order12x1_v1'
SEED = 2026100800
BATCH_ID = 'h4_codex_train_reeval_v137'
MODELS = {'clean': 'M_clean_policy', 'new': 'M_v125_newonly'}
NEW_PATH = REPO / 'restart-n5-runs/h4_codex_combined_sft_v125_merged'
NEW_IDENTITY = V127 / 'model_tools/NEW_MODEL_IDENTITY.json'
NEW_IDENTITY_SHA = '64d0059000cd7942163fccd52e8031f024b13c31a9a4b8ba8d129df446d4c2a7'
NATIVE_RUNNER_SHA = '50f2b797fe8a6560d3fc792f34acbec40309dbda462744817b93a453785a6462'
NATIVE_PROTOCOL_SHA = '2314e7759ab483e75e28275dfff91f1e50f7b5f00cc13f5b904e18864d0d0e19'
NATIVE_SCORING_SHA = '2d0e7411dcd7b1c520bdbaecfef5259d48f036ce071a98ac3d6e633be2017ad2'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def verify(rows, relative_to=None):
    require(isinstance(rows, list) and rows, 'NONEMPTY_FILE_BINDINGS_REQUIRED')
    for row in rows:
        path = Path(row['path'])
        if not path.is_absolute():
            require(relative_to is not None, 'ABSOLUTE_BINDING_REQUIRED')
            path = relative_to / path
        require(path.is_file() and sha(path) == row['sha256'], 'BOUND_FILE_CHANGED:' + str(path))
        if 'bytes' in row:
            require(path.stat().st_size == row['bytes'], 'BOUND_FILE_SIZE_CHANGED:' + str(path))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def protocol_text():
    require(sha(NATIVE / 'protocol.py') == NATIVE_PROTOCOL_SHA, 'NATIVE_PROTOCOL_CHANGED')
    text = (NATIVE / 'protocol.py').read_text()
    old = 'if provenance.get("requested_model") != "Qwen/Qwen3.6-35B-A3B":'
    loc = 'CONTRACT_DIR = Path(__file__).resolve().parent.parent'
    require(text.count(old) == text.count(loc) == 1, 'NATIVE_PROTOCOL_ADAPTER_TARGET_CHANGED')
    return text.replace(old, 'if (provenance.get("backend") != "CURRENT_CODEX_SESSION" or provenance.get("source_batch_id") != ' + repr(BATCH_ID) + '):').replace(loc, 'CONTRACT_DIR = Path(' + repr(str(NATIVE.parent)) + ')')


def new_identity():
    require(sha(NEW_IDENTITY) == NEW_IDENTITY_SHA, 'V127_NEW_IDENTITY_CHANGED')
    value = read(NEW_IDENTITY)
    require(value['model_identifier'] == MODELS['new'] and value['model_path'] == str(NEW_PATH), 'NEW_MODEL_PATH_OR_NAME_CHANGED')
    require(value['tokenizer_path'] == '/root/autodl-tmp/Qwen3-4B', 'NEW_TOKENIZER_CHANGED')
    require(value['adapter_path'] == str(REPO / 'restart-n5-runs/h4_codex_combined_sft_v125/adapter'), 'NEW_ADAPTER_CHANGED')
    require(value['adapter_enabled'] is True and value['adapter_already_merged'] is True, 'NEW_MODEL_NOT_ALREADY_MERGED')
    require(sha(value['audit_path']) == value['audit_sha256'], 'NEW_MODEL_MERGE_AUDIT_CHANGED')
    return value


def native():
    require(sha(NATIVE / 'run_solver.py') == NATIVE_RUNNER_SHA, 'NATIVE_RUNNER_CHANGED')
    require(sha(NATIVE / 'scoring.py') == NATIVE_SCORING_SHA, 'NATIVE_SCORING_CHANGED')
    sys.path.insert(0, str(NATIVE))
    obj = module('v137_native_v15_runner', NATIVE / 'run_solver.py')
    require(obj.POLICY_VERSION == POLICY and obj.SEED == SEED, 'NATIVE_POLICY_CHANGED')
    require((HERE / 'adapter_protocol.py').read_text() == protocol_text(), 'PROVENANCE_ADAPTER_CHANGED')
    obj.validate_pool = module('v137_protocol', HERE / 'adapter_protocol.py').validate_pool
    original_identity, original_validate = obj.model_identity, obj._validate_model
    obj.MODEL_PATHS[MODELS['new']] = NEW_PATH

    def identity(model='M_clean_policy'):
        return new_identity() if model == MODELS['new'] else original_identity(model)

    def validate(value):
        if value.get('model_identifier') != MODELS['new']:
            return original_validate(value)
        require(value == new_identity(), 'NEW_IDENTITY_DIFFERS_FROM_V127_FIXED_AUDIT')
        verify(value['checkpoint_files'])

    obj.model_identity, obj._validate_model = identity, validate
    return obj


def qa_inputs():
    """Losslessly evaluate the effective canonical questions actually used in SFT."""
    manifest_path = TRAIN_PACKAGE / 'COMPLETE_MANIFEST.json'
    verify(read(manifest_path)['files'], TRAIN_PACKAGE)
    original = read(TRAIN_PACKAGE / 'data/COMBINED_QUESTIONS.json')['questions']
    actual_path = TRAIN_PACKAGE / 'data/TRAIN_NEW_ONLY.jsonl'
    lineage_path = TRAIN_PACKAGE / 'data/QUESTION_LINEAGE.jsonl'
    actual = [json.loads(line) for line in actual_path.read_text().splitlines() if line.strip()]
    lineage = [json.loads(line) for line in lineage_path.read_text().splitlines() if line.strip()]
    roots = sorted(r['root_id'] for r in read(TRAIN_ROOTS))
    require(len(roots) == len(set(roots)) == 68, 'FIXED_TRAIN_ROOT_COUNT_CHANGED')
    require(len(original) == len(actual) == len(lineage) == 363, 'ACTUAL_TRAIN_COUNT_CHANGED')
    ids = {q['candidate_id'] for q in original}
    require(len(ids) == 363 and ids == {q['candidate_id'] for q in actual} == {q['candidate_id'] for q in lineage}, 'ACTUAL_TRAIN_IDS_DIFFER')
    require({q['root_id'] for q in original} == set(roots), 'ACTUAL_TRAIN_ROOTS_DIFFER')
    heldout = read(V124 / 'MANIFEST.json')['roots']
    require(set(roots).isdisjoint(r['root_id'] for r in heldout), 'TRAIN_HELDOUT_ROOT_OVERLAP')
    questions = copy.deepcopy(original)
    for q in questions:
        q['split'] = 'train'
        q['evaluation_only'] = False
        q['actually_trained'] = True
        require(sha(q['source_path']) == q['source_sha256'], 'TRAIN_SOURCE_CHANGED:' + q['candidate_id'])
    counts = Counter(q['question_type'] for q in questions)
    gold_counts = Counter(len(q['gold']) for q in questions if q['question_type'] == 'multi_select')
    require(counts == {'multi_select': 238, 'ordering': 125}, 'TRAIN_TYPE_COUNTS_CHANGED')
    require(gold_counts == {2: 65, 3: 59, 4: 57, 5: 57}, 'TRAIN_GOLD_COUNTS_CHANGED')
    paths = [manifest_path, TRAIN_ROOTS, actual_path, lineage_path, TRAIN_PACKAGE / 'data/COMBINED_QUESTIONS.json', V124 / 'MANIFEST.json']
    paths.extend(Path(q['source_path']) for q in questions)
    quota = {root: {'split': 'train', 'question_count': sum(q['root_id'] == root for q in questions)} for root in roots}
    return questions, {'train': roots}, {'train': roots}, {'train': 363}, quota, paths


def prepare():
    require(not (HERE / 'SCOPE.json').exists() and not (HERE / 'run').exists(), 'EVALUATION_ALREADY_PREPARED_NO_OVERWRITE')
    questions, incremental, full_roots, split_counts, quota, inputs = qa_inputs()
    adapter_path = HERE / 'adapter_protocol.py'
    require(adapter_path.exists() and adapter_path.read_text() == protocol_text(), 'PROVENANCE_ADAPTER_CHANGED')
    runner = native()
    require(Path(sys.prefix) == runner.EXPECTED_VENV, 'USE_PROVEN_LOCAL4B_VENV')
    # Validate all sources, semantic gates and all three complete weight bindings
    # before constructing an immutable run directory. No model is loaded.
    for model in MODELS.values():
        runner._validate_model(runner.model_identity(model))
    save(HERE / 'EVALUATION_POOL.json', {'status': 'FROZEN_ALL_CONTENT_PASS_NO_SOLVER_SELECTION',
        'provenance': {'backend': 'CURRENT_CODEX_SESSION', 'source_batch_id': BATCH_ID,
                       'exact_model_id': None, 'same_inherited_model': True}, 'questions': questions})
    runner.validate_pool(HERE / 'EVALUATION_POOL.json')
    for side, model in MODELS.items():
        runner.preflight(HERE / 'EVALUATION_POOL.json', HERE / 'run' / side, solver_model=model)
    manifests = {side: read(HERE / 'run' / side / 'manifest.json') for side in MODELS}
    first = manifests['clean']
    count = sum(18 if q['question_type'] == 'multi_select' else 12 for q in questions)
    for side, manifest in manifests.items():
        require(manifest['candidate_count'] == len(questions) and manifest['request_count'] == count, 'DYNAMIC_REQUEST_COUNTS_DIFFER:' + side)
        require(manifest['requests'] == first['requests'] and manifest['decoder_contract'] == first['decoder_contract'], 'PAIRED_REQUESTS_OR_DECODER_DIFFER:' + side)
        for name in ['questions.json', 'gold.json', 'canonical_bindings.json']:
            require(read(HERE / 'run' / side / name) == read(HERE / 'run/clean' / name), 'PAIRED_INPUTS_DIFFER:' + side + ':' + name)
    maximum = max(r['prompt_tokens'] for r in first['requests'])
    save(HERE / 'SHARED_REQUESTS.json', {'policy_version': POLICY, 'seed': SEED, 'requests': first['requests'],
        'decoder_contract': first['decoder_contract'], 'candidate_order': [q['candidate_id'] for q in questions],
        'ordering': 'Frozen actual-training canonical order; one paired plan for both models.'})
    save(HERE / 'SCOPE.json', {'schema_version': 'v137_frozen_actual_training_fit_v1', 'batch_id': BATCH_ID,
        'fixed_incremental_roots': incremental, 'fixed_full_heldout_roots': full_roots,
        'covered_incremental_roots': sorted({q['root_id'] for q in questions}), 'root_quotas': quota,
        'fixed_training_roots': full_roots, 'target_question_count': 363, 'question_count': len(questions), 'question_count_by_split': split_counts,
        'question_type_counts': dict(Counter(q['question_type'] for q in questions)),
        'calls_per_model': count, 'total_planned_calls': count * len(MODELS),
        'runtime_scheduling': {'batch_size': 8, 'identical_scheduling_for_both_models': True, 'serial_numerical_equivalence_claimed': False},
        'models': MODELS, 'policy_version': POLICY, 'seed': SEED, 'maximum_prompt_tokens': maximum,
        'unique_prompts_per_model': len({r['prompt_sha256'] for r in first['requests']}),
        'score_based_filtering': False, 'automatic_retry': False, 'full_context_only': True,
        'gold6_is_control': True, 'gold6_present': False, 'no_new_training': True, 'actual_training_pool_frozen_before_solver': True,
        'population_limit': 'All 363 actual training questions on 68 train roots; this measures fit on seen questions, not heldout generalization. Permutations, Gold families and same-root questions are correlated.',
        'qa_input_bindings': [bind(p) for p in sorted(set(inputs))]})
    paths = {p for p in HERE.glob('*.py')} | {p for p in (BATCH / 'evidence').glob('*.py')} | {runner.BASE_RUNNER, *NATIVE.glob('*.py'), *NATIVE.parent.glob('*.py')}
    for manifest in manifests.values():
        paths.update(Path(p) for p in manifest['code_sha256'])
    for obj in list(sys.modules.values()):
        filename = getattr(obj, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == '.py' and path.is_relative_to(REPO / 'worktrees/h4-gate-downstream-v1'):
                paths.add(path)
    report_helpers = V127 / 'report_paired.py'
    paths.add(report_helpers)
    save(HERE / 'TOOLS_FREEZE.json', {'status': 'FROZEN_BEFORE_FIRST_INFERENCE', 'created_at_utc': now(),
        'files': [bind(p) for p in sorted(paths)], 'native_algorithm_changed': False,
        'adapter_changes': ['Generator provenance and absolute native gate path only.',
                            'Audited v127 merged round2 identity added; original M0/M_clean validators preserved.'],
        'policy_version': POLICY, 'seed': SEED})
    inputs.extend([HERE / n for n in ['SCOPE.json', 'EVALUATION_POOL.json', 'SHARED_REQUESTS.json', 'TOOLS_FREEZE.json']])
    inputs.extend([BATCH / 'PROTOCOL_ZH.md', BATCH / 'evidence/INPUT_VALIDATION.json'])
    inputs.extend([NEW_IDENTITY, Path(new_identity()['audit_path'])])
    for side in MODELS:
        inputs.extend(p for p in (HERE / 'run' / side).rglob('*') if p.is_file())
        identity = read(HERE / 'run' / side / 'solver_identity.json')
        for field in ['audit_path', 'tokenizer_audit_path']:
            if identity.get(field):
                inputs.append(Path(identity[field]))
    save(HERE / 'PREFLIGHT_FREEZE.json', {'status': 'PREPARED_AWAITING_INDEPENDENT_PREFLIGHT',
        'created_at_utc': now(), 'files': [bind(p) for p in sorted(set(inputs))],
        'candidate_count': len(questions), 'calls_per_model': count, 'models': MODELS,
        'maximum_prompt_tokens': maximum, 'automatic_retry': False, 'inference_started': False})
    print(json.dumps({'status': 'PREPARED_AWAITING_INDEPENDENT_PREFLIGHT', 'questions': len(questions), 'calls_per_model': count}))


def frozen(require_independent=False, check_models=True):
    verify(read(HERE / 'TOOLS_FREEZE.json')['files'])
    verify(read(HERE / 'PREFLIGHT_FREEZE.json')['files'])
    verify(read(TRAIN_PACKAGE / 'COMPLETE_MANIFEST.json')['files'], TRAIN_PACKAGE)
    plan = read(HERE / 'SHARED_REQUESTS.json')
    runner = native()
    for side in MODELS:
        run = HERE / 'run' / side
        manifest = read(run / 'manifest.json')
        require(manifest['requests'] == plan['requests'] and manifest['decoder_contract'] == plan['decoder_contract'], 'SHARED_PLAN_CHANGED:' + side)
        if check_models:
            runner._verify_manifest(run, manifest)
    if require_independent:
        report = read(BATCH / 'evidence/INDEPENDENT_EVALUATION_PREFLIGHT_AUDIT.json')
        require(report.get('status') == 'PASS', 'INDEPENDENT_PREFLIGHT_NOT_PASS')
        verify(report['audited_files'], BATCH)
        bound = {str((BATCH / r['path']).resolve()) for r in report['audited_files']}
        require(str((HERE / 'PREFLIGHT_FREEZE.json').resolve()) in bound, 'INDEPENDENT_AUDIT_NOT_BOUND_TO_THIS_PREFLIGHT')
    return runner


def live(side):
    module('v137_batched_live', HERE / 'batched_live.py').live(sys.modules[__name__], side)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='phase', required=True)
    commands.add_parser('prepare')
    verify_cmd = commands.add_parser('verify')
    verify_cmd.add_argument('--require-independent', action='store_true')
    live_cmd = commands.add_parser('live')
    live_cmd.add_argument('--side', choices=MODELS, required=True)
    commands.add_parser('report')
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare()
    elif args.phase == 'verify':
        frozen(require_independent=args.require_independent)
        print('PASS: frozen inputs, tools, shared plan and full model hashes' + ('; independent audit PASS' if args.require_independent else '; independent audit not checked'))
    elif args.phase == 'live':
        live(args.side)
    else:
        raise SystemExit(module('v137_report', HERE / 'report.py').main())


if __name__ == '__main__':
    main()
