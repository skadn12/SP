"""Independent v79 input/result audit. Standard library only; no model calls.

This file deliberately imports no batch, native scoring/selection/protocol or
mixed_evaluation implementation. It reconstructs the documented contracts.
Run --inputs while evaluation is active; run without it only after both sides
and the mixed report are complete. Outputs are append-only evidence JSON.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import traceback
import os
import sys

HOME = Path(__file__).resolve().parent.parent
PROJECT = HOME.parent.parent
SEED = 2026100800
MODELS = {
    'trained': ('M_clean_policy', str(PROJECT / 'restart-n5-runs/h4_round0_common_dev_eval_v1_live_3/clean_merged_model')),
    'base': ('M0', '/root/autodl-tmp/Qwen3-4B'),
}
ALLOWED_DIFFERENCES = {'prepared_at_utc', 'model', 'solver_model', 'solver_identity'}
POLICY = 'r02_groundless_full_only_random_position_v2'
BASE_AUDIT = PROJECT / 'offline-deliveries/h4_multiselect_partial_admission_v1/followup_model_identity.json'
BASE_AUDIT_SHA = '59e23710e2373a00aef3b7d50d117d4089eea47cd17fd16a00eebcc83fba0035'
TRAINED_AUDIT = PROJECT / 'restart-n5-runs/h4_round0_trained_solver_train_reeval_v1_live_1/model_identity_report.json'
TRAINED_AUDIT_SHA = 'add007d4e4f937f2ea1b7e124ccac3c53e5eda5b8bbed1beab3dd4b0855cadd0'


def audit_guard(event, args):
    if event.startswith('socket.') or event in {'subprocess.Popen','os.system','os.posix_spawn'}:
        raise RuntimeError('audit_forbids_external:' + event)
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(args[0])).resolve()
        mode, flags = args[1:3]
        writing = ((isinstance(mode,str) and any(c in mode for c in 'wax+'))
                   or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
        if writing and path not in {HOME/'evidence/INPUT_AUDIT.json', HOME/'evidence/RESULT_AUDIT.json'}:
            raise RuntimeError('audit_forbids_artifact_write:' + str(path))


sys.dont_write_bytecode = True
sys.addaudithook(audit_guard)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stamp(path):
    s = Path(path).stat()
    return {'size': s.st_size, 'mtime_ns': s.st_mtime_ns, 'ctime_ns': s.st_ctime_ns,
            'inode': s.st_ino, 'device': s.st_dev}


def now():
    return datetime.now(timezone.utc).isoformat()


def dt(value):
    result = datetime.fromisoformat(value)
    if result.utcoffset() is None:
        raise ValueError('naive_timestamp')
    return result


class Audit:
    def __init__(self):
        self.checks = []
        self.files = {}
        self.weights = {}

    def check(self, condition, name, detail=None):
        row = {'check': name, 'status': 'PASS' if condition else 'FAIL'}
        if detail is not None:
            row['detail'] = detail
        self.checks.append(row)
        return condition

    def bind(self, path):
        path = Path(path).resolve()
        row = {'path': str(path), 'sha256': digest(path)}
        self.files[str(path)] = row
        return row

    def read(self, path):
        self.bind(path)
        return json.loads(Path(path).read_text())

    def text(self, path):
        self.bind(path)
        return Path(path).read_text()

    def weight(self, row, cached):
        path = Path(row['path']).resolve()
        before = stamp(path)
        prior = cached.get(str(path))
        if prior and prior['stat_after'] == before and prior['sha256'] == row['sha256']:
            value = prior['sha256']
            method = 'full_sha256_from_input_audit_with_unchanged_size_mtime_ctime_inode_device'
        else:
            value = digest(path)
            method = 'full_sha256_recomputed_from_file_bytes'
        after = stamp(path)
        self.check(before == after and after['size'] == row['bytes'] and value == row['sha256'],
                   'checkpoint_bytes:' + str(path))
        self.weights[str(path)] = {'path': str(path), 'sha256': value,
                                  'stat_before': before, 'stat_after': after, 'verification': method}


def content_ids(question):
    field = 'options' if question['question_type'] == 'multi_select' else 'items'
    labels = list('ABCDEF') if field == 'options' else ['I' + str(i) for i in range(1, 7)]
    return field, labels, {label: 'content_' + hashlib.sha256(question[field][label].strip().encode()).hexdigest()[:16]
                           for label in labels}


def variants(question):
    """Recreate the specified seeded displays without accepting Gold input."""
    field, labels, by_label = content_ids(question)
    content = [by_label[label] for label in labels]
    texts = {by_label[label]: question[field][label] for label in labels}
    qid = question['candidate_id']
    def rng(index):
        raw = hashlib.sha256((str(SEED) + '|' + qid + '|' + str(index)).encode()).digest()
        return random.Random(int.from_bytes(raw[:8], 'big'))
    rng(-1).shuffle(content)
    used = set()
    output = []
    for i in range(6 if field == 'options' else 12):
        generator = rng(i)
        while True:
            order = content[:]
            generator.shuffle(order)
            if tuple(order) not in used:
                break
        used.add(tuple(order))
        output.append({'variant_id': f'perm_{i:02d}', 'canonical_order_content_ids': content[:],
                       'display_order_content_ids': order,
                       field: {label: texts[cid] for label, cid in zip(labels, order)},
                       'display_to_content': dict(zip(labels, order)), 'content_to_text': texts})
    return output


def score(raw, question, request, gold):
    """Parse raw display IDs and score canonical sets/pairs using exact fractions."""
    if not isinstance(raw, str):
        raise ValueError('raw_output_not_text')
    match = re.fullmatch(r'\s*<box>\s*(.*?)\s*</box>\s*', raw, flags=re.S)
    if match is None:
        raise ValueError('raw_output_not_exact_box')
    inside = match.group(1).strip()
    labels = [] if not inside else [part.strip().upper() for part in inside.split(',')]
    if any(re.fullmatch(r'[A-Z][A-Z0-9_]*', label) is None for label in labels) or len(set(labels)) != len(labels):
        raise ValueError('malformed_or_duplicate_display_ids')
    mapping = request['variant']['display_to_content']
    if not set(labels) <= set(mapping):
        raise ValueError('unknown_display_ids')
    prediction = [mapping[label] for label in labels]
    if len(set(prediction)) != len(prediction):
        raise ValueError('duplicate_content')
    if question['question_type'] == 'multi_select':
        predicted, correct = set(prediction), set(gold)
        tp, fp, fn = len(predicted & correct), len(predicted - correct), len(correct - predicted)
        partial = Fraction(tp, len(correct))
        if len(correct) < len(mapping):
            partial -= Fraction(fp, len(mapping) - len(correct))
        exact = predicted == correct
        canonical_prediction = sorted(predicted)
    else:
        if len(prediction) != len(gold) or set(prediction) != set(gold):
            raise ValueError('incomplete_order')
        positions = {cid: i for i, cid in enumerate(prediction)}
        total = len(gold) * (len(gold) - 1) // 2
        fp = sum(positions[a] > positions[b] for i, a in enumerate(gold) for b in gold[i + 1:])
        tp, fn = total - fp, 0
        partial = Fraction(tp - fp, total)
        exact, canonical_prediction = prediction == gold, prediction
    return {'parse_status': 'success', 'parse_reason': None, 'scored': True,
            'display_prediction': labels, 'content_prediction': canonical_prediction,
            'content_gold': gold, 'tp': tp, 'fp': fp, 'fn': fn, 'exact_match': exact,
            'partial_score': str(partial), 'partial_score_fraction': str(partial), 'scoring_status': 'SCORED'}


def select_independent(rows, question):
    count = 6 if question['question_type'] == 'multi_select' else 12
    expected = [f'perm_{i:02d}' for i in range(count)]
    means = {vid: sum((Fraction(r['independent_score']['partial_score_fraction']) for r in rows if r['variant_id'] == vid), Fraction()) /
             sum(r['variant_id'] == vid for r in rows) for vid in expected}
    mean = sum(means.values(), Fraction()) / count
    exact_rate = Fraction(sum(r['independent_score']['exact_match'] for r in rows), len(rows))
    return {'policy_version': POLICY, 'ground_used': False, 'selected_by_policy': 0 < mean < 1,
            'selection_reason': 'FULL_PARTIAL_IN_OPEN_INTERVAL' if 0 < mean < 1 else
                                ('FULL_PARTIAL_NONPOSITIVE' if mean <= 0 else 'FULL_PARTIAL_PERFECT_NO_NEW_HARD_ITEM'),
            'full_partial_mean': str(mean), 'permutation_means': {k: str(v) for k, v in means.items()},
            'permutation_score_min': str(min(means.values())), 'permutation_score_max': str(max(means.values())),
            'permutation_score_range': str(max(means.values()) - min(means.values())),
            'full_exact_rate': str(exact_rate), 'context_gain': None, 'data_errors': [],
            'candidate_id': question['candidate_id'], 'root_id': question['root_id'],
            'question_type': question['question_type'], 'expected_full_calls': 18 if count == 6 else 12,
            'actual_full_calls': len(rows)}


def aggregate(rows):
    n = len(rows)
    successes = sum(r['independent_score']['exact_match'] for r in rows)
    return {'calls': n, 'exact_correct': successes,
            'exact_accuracy': str(Fraction(successes, n)) if n else None,
            'partial_mean': str(sum((Fraction(r['independent_score']['partial_score_fraction']) for r in rows), Fraction()) / n) if n else None}


def selfcheck(a):
    q = {'question_type': 'multi_select'}
    req = {'variant': {'display_to_content': {c: c for c in 'ABCDEF'}}}
    gold = ['A', 'C']
    a.check(score('<box>A,C</box>', q, req, gold)['partial_score_fraction'] == '1', 'independent_algorithm_exact_case')
    a.check(score('<box>A,B,C,D,E,F</box>', q, req, gold)['partial_score_fraction'] == '0', 'independent_algorithm_all_selected_g2_case')
    a.check(score('<box>B,D,E,F</box>', q, req, gold)['partial_score_fraction'] == '-1', 'independent_algorithm_false_only_case')
    a.check(score('<box>A</box>', q, req, list('ABCDEF'))['partial_score_fraction'] == '1/6', 'independent_algorithm_g6_case')
    rejected = False
    try:
        score('<box>A,A</box>', q, req, gold)
    except ValueError:
        rejected = True
    a.check(rejected, 'independent_algorithm_duplicate_rejected')
    q = {'question_type': 'ordering'}
    req = {'variant': {'display_to_content': {'I'+str(i): str(i) for i in range(1, 7)}}}
    a.check(score('<box>I6,I5,I4,I3,I2,I1</box>', q, req, list('123456'))['partial_score_fraction'] == '-1', 'independent_algorithm_ordering_reverse_case')


def audit_inputs(a, cached):
    config = a.read(HOME / 'CONFIG.json')
    pool = a.read(HOME / 'EVALUATION_POOL.json')
    prepared = a.read(HOME / 'EVALUATION_PREPARATION.COMPLETE.json')
    frozen = a.read(HOME / 'BATCH_REVIEW.COMPLETE.json')
    questions = pool['questions']
    ids = [q['candidate_id'] for q in questions]
    candidate_count = len(questions)
    expected_calls = sum(18 if q['question_type'] == 'multi_select' else 12 for q in questions)
    a.check(candidate_count > 0 and len(ids) == len(set(ids)), 'unique_content_usable_questions')
    child_review = a.read(HOME / 'multiselect/FINAL_CONTENT_REVIEW.COMPLETE.json')
    ordering_pool = a.read(HOME / 'ordering/EVALUATION_POOL.json')['questions']
    assembled = a.read(HOME / 'multiselect/POOL.json')['questions']
    expected_pool = [q for q in assembled if q['candidate_id'] in child_review['usable_candidate_ids']] + ordering_pool
    a.check(questions == expected_pool, 'evaluation_pool_matches_frozen_review_usable_questions')
    distribution = {str(g): sum(len(q['gold']) == g for q in questions) for g in range(2, 7)}
    members = {r['candidate_id']:r for r in a.read(HOME / 'multiselect/FAMILY_SIDECAR.json')['members']}
    a.check(all(len(q['gold']) == members[q['candidate_id']]['gold_count'] for q in questions if q['question_type']=='multi_select'), 'content_gold_distribution_matches_assembly_roles', distribution)
    a.check(pool.get('training_allowed') is False and pool.get('formal_training_admission') is False and
            all(q['training_allowed'] is False for q in questions), 'pool_no_training_admission')
    controls = {s['parent_candidate_id'] for s in config['multiselect_config']['slots']}
    a.check({q['candidate_id'] for q in questions if len(q['gold']) == 6} == controls, 'gold6_are_exact_historical_control_ids')
    a.check(prepared['status'] == 'PASS' and prepared['candidate_count'] == candidate_count and prepared['calls_per_model'] == expected_calls and
            prepared['training_started'] is False and prepared['automatic_rerun'] is False, 'preparation_counts_and_no_rerun')
    a.check(frozen['status'] == 'FROZEN' and frozen['before_any_solver'] is True and frozen['solver_scores_used'] is False and
            frozen['training_allowed'] is False and dt(frozen['completed_at_utc']) <= dt(prepared['prepared_at_utc']), 'source_freeze_precedes_evaluation')
    for container, name in ((frozen, 'batch_review'), (prepared, 'evaluation_preparation')):
        rows = container['files']
        a.check(len({str(Path(r['path']).resolve()) for r in rows}) == len(rows), name + ':unique_frozen_files')
        for row in rows:
            a.check(a.bind(row['path'])['sha256'] == row['sha256'], name + ':file_binding:' + row['path'])
    manifests, golds = {}, {}
    base = a.read(BASE_AUDIT)
    trained = a.read(TRAINED_AUDIT)['models']['M_clean_policy']
    a.check(digest(BASE_AUDIT) == BASE_AUDIT_SHA and digest(TRAINED_AUDIT) == TRAINED_AUDIT_SHA, 'fixed_historical_model_audits')
    all_expected_requests = []
    for q in questions:
        field, labels, cid_by_label = content_ids(q)
        a.check(list(q[field]) == labels and len(set(cid_by_label.values())) == 6, 'six_unique_content_ids:' + q['candidate_id'])
        for v in variants(q):
            for slot in range(3 if field == 'options' else 1):
                all_expected_requests.append({'request_id': q['candidate_id'] + '__' + v['variant_id'] + f'__full_context__s{slot:02d}',
                                              'candidate_id': q['candidate_id'], 'root_id': q['root_id'], 'question_type': q['question_type'],
                                              'variant': v, 'mode': 'full_context', 'slot': slot, 'seed': SEED + len(all_expected_requests)})
    for side, (model, model_path) in MODELS.items():
        run = HOME / 'solver' / side
        m = a.read(run / 'manifest.json'); manifests[side] = m
        identity = a.read(run / 'solver_identity.json')
        a.check(m['status'] == 'PREFLIGHT_PASS' and m['solver_model'] == model and m['model'] == model_path and
                m['tokenizer'] == '/root/autodl-tmp/Qwen3-4B' and m['policy_version'] == 'v15_full_only_random_mcq6x3_order12x1_v1' and
                m['execution'] == 'full_only_no_ground_v1', side + ':fixed_model_tokenizer_policy')
        a.check(m['training_started'] is False and m['training_promotion'] is False and m['new_generation_calls'] == 0, side + ':no_training_or_new_generation')
        a.check(m['random_display'] == {'version': 'independent_seeded_permutation_v2', 'seed': SEED, 'gold_blind': True,
                                     'cyclic_rebinding': False, 'shape_filter': False}, side + ':gold_blind_random_display_contract')
        decoder = {'temperature': .8, 'top_p': .9, 'max_tokens': 32, 'enable_thinking': False,
                   'structured_output_api': 'structured_outputs=StructuredOutputsParams(choice)',
                   'structured_output_version': 'finite_choice_set_v1', 'vllm_version': '0.21.0',
                   'stop': ['</box>'], 'include_stop_str_in_output': True}
        a.check(m['decoder_contract'] == decoder, side + ':decoder_contract')
        a.check(m['candidate_count'] == candidate_count and m['request_count'] == len(m['requests']) == len(all_expected_requests) == expected_calls <= config['max_solver_calls_per_model'], side + ':exact_call_budget')
        a.check(identity == m['solver_identity'] and identity['model_identifier'] == model and identity['model_path'] == model_path and
                identity['tokenizer_path'] == m['tokenizer'], side + ':identity_manifest_binding')
        if side == 'base':
            expected_checkpoints = base['files']
            expected_audit_path, expected_audit_hash = str(BASE_AUDIT), BASE_AUDIT_SHA
        else:
            expected_checkpoints = [{'path': str(Path(model_path) / r['name']), 'name': r['name'], 'bytes': r['size_bytes'], 'sha256': r['sha256']}
                                    for r in trained['merged_model_identity']['files']]
            expected_checkpoints += [r for r in base['files'] if r['name'] in {'tokenizer.json', 'tokenizer_config.json', 'merges.txt', 'vocab.json'}]
            expected_audit_path, expected_audit_hash = str(TRAINED_AUDIT), TRAINED_AUDIT_SHA
            a.check(identity.get('adapter_already_merged') is True and identity.get('adapter_enabled') is True and
                    identity['adapter_path'] == trained['adapter_path'] and identity['model_fingerprint_sha256'] == trained['model_fingerprint_sha256'], side + ':trained_adapter_identity')
        a.check(identity['checkpoint_files'] == expected_checkpoints and identity['audit_path'] == expected_audit_path and identity['audit_sha256'] == expected_audit_hash and
                identity['tokenizer_audit_path'] == str(BASE_AUDIT) and identity['tokenizer_audit_sha256'] == BASE_AUDIT_SHA, side + ':identity_matches_fixed_audit')
        for row in identity['checkpoint_files']:
            if row['path'] not in a.weights:
                a.weight(row, cached)
        for path, value in m['code_sha256'].items():
            a.check(a.bind(path)['sha256'] == value, side + ':native_code_unchanged:' + path)
        for name, value in m['input_sha256'].items():
            path = HOME / 'EVALUATION_POOL.json' if name == 'pool' else run / name
            a.check(a.bind(path)['sha256'] == value, side + ':input_hash:' + name)
        a.check(Path(m['pool_path']).resolve() == HOME / 'EVALUATION_POOL.json' and m['pool_sha256'] == digest(HOME / 'EVALUATION_POOL.json'), side + ':pool_path_and_sha')
        a.check(a.read(run / 'questions.json')['candidates'] == [{k: v for k, v in q.items() if k not in {'gold', 'gold_content_ids'}} for q in questions], side + ':questions_exact_pool_projection')
        gold = a.read(run / 'gold.json'); golds[side] = gold
        expected_gold = {q['candidate_id']: [content_ids(q)[2][label] for label in q['gold']] for q in questions}
        a.check(gold == expected_gold, side + ':gold_exact_content_binding')
        expected_bindings = []
        for q in questions:
            field, labels, cid_by_label = content_ids(q)
            expected_bindings.append({'candidate_id': q['candidate_id'], 'root_id': q['root_id'], 'content_id_by_label': cid_by_label,
                                      'gold_content_ids': expected_gold[q['candidate_id']],
                                      'evidence_by_content_id': {cid_by_label[label]: q['option_evidence' if field == 'options' else 'item_evidence'][label] for label in labels}})
        a.check(a.read(run / 'canonical_bindings.json')['candidates'] == expected_bindings, side + ':canonical_evidence_binding')
        requests = m['requests']
        a.check(len({r['request_id'] for r in requests}) == len(requests), side + ':unique_request_ids')
        by_qid = {q['candidate_id']: q for q in questions}
        for actual, expected in zip(requests, all_expected_requests):
            rid = actual['request_id']
            a.check(set(actual) == set(expected) | {'prompt_sha256', 'prompt_tokens'} and all(actual.get(k) == v for k, v in expected.items()) and
                    type(actual['seed']) is int and type(actual['slot']) is int, side + ':independent_request_reconstruction:' + rid)
            prompt = a.text(run / 'prompts' / (rid + '.txt'))
            a.check(hashlib.sha256(prompt.encode()).hexdigest() == actual['prompt_sha256'] and type(actual['prompt_tokens']) is int and
                    0 < actual['prompt_tokens'] <= 16352, side + ':prompt_hash_capacity:' + rid)
            q = by_qid[actual['candidate_id']]
            source = a.read(q['source_path'])
            dialogue_block = 'Dialogue:\n' + '\n'.join('[' + t['turn_ref'] + '] ' + t['speaker'] + ': ' + t['text'] for t in source['turns'])
            field = 'options' if q['question_type'] == 'multi_select' else 'items'
            item_block = ('Options' if field == 'options' else 'Items') + ':\n' + '\n'.join(label + '. ' + text for label, text in actual['variant'][field].items())
            a.check(dialogue_block in prompt and ('Question (' + q['question_type'] + '):\n' + q['question']) in prompt and item_block in prompt,
                    side + ':complete_source_and_public_question_in_prompt:' + rid)
            system = 'You are a strict structured dialogue QA solver. Use only the supplied dialogue when it is present. Return the final answer in exactly one <box>...</box> pair. Do not provide explanations, reasoning, or any text outside the box.'
            trailing = 'Output only selected letters separated by commas, e.g. <box>A,C</box>.' if field == 'options' else 'Output only ordered item IDs separated by commas, e.g. <box>I2,I4,I1,I3</box>.'
            user_text = '\n\n'.join([dialogue_block, 'Question (' + q['question_type'] + '):\n' + q['question'], item_block, trailing])
            exact_public_prompt = '<|im_start|>system\n' + system + '<|im_end|>\n<|im_start|>user\n' + user_text + '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
            a.check(prompt == exact_public_prompt, side + ':exact_gold_free_full_prompt_reconstruction:' + rid)
            a.check(a.bind(q['source_path'])['sha256'] == q['source_sha256'] == digest(run / 'sources' / (q['root_id'] + '.json')),
                    side + ':full_source_version:' + rid)
    strip = lambda m: {k: v for k, v in m.items() if k not in ALLOWED_DIFFERENCES}
    a.check(strip(manifests['trained']) == strip(manifests['base']), 'paired_manifests_only_allowed_model_differences')
    a.check(golds['trained'] == golds['base'], 'paired_gold_content_identical')
    return {'config': config, 'pool': pool, 'questions': questions, 'manifests': manifests, 'golds': golds,
            'prepared': prepared, 'frozen': frozen, 'gold_distribution': distribution, 'controls': controls,
            'candidate_count':candidate_count, 'expected_calls':expected_calls, 'assembled':assembled, 'ordering_pool':ordering_pool}


def audit_results(a, ctx):
    questions = ctx['questions']; by_qid = {q['candidate_id']: q for q in questions}
    runtime = a.read(HOME / 'evidence/EVALUATION_RUNTIME_RECEIPT.json')
    expected_env = {}
    a.check(runtime['working_directory'] == str(HOME) and runtime['python'] == str(PROJECT / '.venv-local4b/bin/python') and
            runtime['python_flags'] == ['-B'] and runtime['additional_environment'] == expected_env and runtime['removed_environment'] == ['PYTHONPATH'] and
            runtime['solver_automatic_replays'] == 0 and runtime['frozen_code_edits'] == [] and runtime['training_performed'] is False,
            'runtime_clean_import_path_no_frozen_edits_or_solver_replays')
    expected_actions = ['freeze', 'prepare-evaluation', 'live-trained', 'live-base', 'report']
    a.check([r['action'] for r in runtime['commands']] == expected_actions, 'runtime_single_sequential_actions')
    for command in runtime['commands']:
        expected = 'env -u PYTHONPATH ' + runtime['python'] + ' -B batch.py ' + command['action']
        a.check(command['command'] == expected, 'runtime_exact_invocation:' + command['action'])
        binding = command['stdout_stderr_log']
        a.check(a.bind(binding['path'])['sha256'] == binding['sha256'], 'runtime_invocation_log_binding:' + command['action'])
    for binding in [runtime[k] for k in ('preparation', 'trained_complete', 'base_complete', 'report')] + runtime['recoveries']:
        a.check(a.bind(binding['path'])['sha256'] == binding['sha256'], 'runtime_artifact_binding:' + binding['path'])
    a.check(runtime['recoveries'] == [], 'no_runtime_import_recovery')
    sides = {}; selections = {}
    for side, m in ctx['manifests'].items():
        run = HOME / 'solver' / side
        complete = a.read(run / 'COMPLETE.json')
        start = a.read(run / 'STARTED.json')
        stage_complete = a.read(HOME / (side.upper() + '_EVALUATION.COMPLETE.json'))
        stage_start = a.read(HOME / (side.upper() + '_EVALUATION.STARTED.json'))
        a.check(complete['status'] == 'COMPLETE' and complete['actual_call_count'] == ctx['expected_calls'] and complete['training_started'] is False and
                complete['training_promotion'] is False and not (run / 'FAILED.json').exists() and not (HOME / (side.upper() + '_EVALUATION.ERROR.json')).exists(), side + ':execution_complete_no_failure')
        a.check(start['manifest_sha256'] == digest(run / 'manifest.json') and dt(ctx['frozen']['completed_at_utc']) < dt(start['started_at_utc']) and
                stage_start['training_started'] is False and stage_start['calls_planned'] == ctx['expected_calls'] and stage_complete['status'] == 'COMPLETE' and
                stage_complete['calls'] == ctx['expected_calls'] and stage_complete['training_started'] is False, side + ':started_after_frozen_and_stage_counts')
        files = sorted((run / 'responses').glob('*.json'))
        rows = [a.read(p) for p in files]
        plans = {r['request_id']: r for r in m['requests']}
        a.check(len(rows) == len(files) == ctx['expected_calls'] and Counter(r['request_id'] for r in rows) == Counter({rid: 1 for rid in plans}), side + ':one_response_per_plan_no_retry')
        replayed = []
        for path, record in zip(files, rows):
            rid = record['request_id']; request = plans[rid]; qid = request['candidate_id']; q = by_qid[qid]
            a.check(path.stem == rid and all(record[k] == request[k] for k in ('candidate_id', 'root_id', 'question_type', 'mode', 'slot', 'seed')) and
                    record['variant_id'] == request['variant']['variant_id'], side + ':record_plan_identity:' + rid)
            a.check(record['evaluation_valid'] is True and record['data_errors'] == [] and record['solver_model'] == m['solver_model'] and
                    record['model_path'] == m['model'], side + ':record_model_and_validity:' + rid)
            metadata = record['metadata']
            a.check(metadata['finish_reason'] in {'stop', 'eos'} and metadata['sampling_seed'] == request['seed'] and
                    metadata['prompt_token_count'] == request['prompt_tokens'] and all(metadata[k] == m['decoder_contract'][k] for k in
                    ('temperature', 'top_p', 'max_tokens', 'structured_output_api', 'structured_output_version')),
                    side + ':runtime_seed_decoder_prompt_tokens:' + rid)
            for key in ('retry_index', 'retry_count', 'transport_retry_count'):
                if key in metadata:
                    a.check(metadata[key] == 0, side + ':no_retry_metadata:' + rid + ':' + key)
            independent = score(record['raw_output'], q, request, ctx['golds'][side][qid])
            a.check(independent == record['score'], side + ':raw_score_independent_exact_replay:' + rid)
            replayed.append({k: record[k] for k in ('request_id', 'candidate_id', 'root_id', 'question_type', 'variant_id', 'slot', 'seed')} |
                            {'independent_score': independent, 'response': a.bind(path)})
        results = a.read(run / 'RESULTS.json')
        summaries = []
        for q in questions:
            qid = q['candidate_id']; qr = [r for r in replayed if r['candidate_id'] == qid]
            expected_pairs = {(f'perm_{i:02d}', j) for i in range(6) for j in range(3)} if q['question_type'] == 'multi_select' else {(f'perm_{i:02d}', 0) for i in range(12)}
            a.check(len(qr) == len(expected_pairs) and {(r['variant_id'], r['slot']) for r in qr} == expected_pairs and
                    len({r['seed'] for r in qr}) == len(qr), side + ':all_permutations_slots_and_seeds:' + qid)
            summary = select_independent(qr, q)
            a.check([r for r in results['questions'] if r['candidate_id'] == qid] == [summary], side + ':independent_selection:' + qid)
            summaries.append(summary)
        chosen = [r['candidate_id'] for r in summaries if r['selected_by_policy']]
        a.check(results['policy_version'] == m['policy_version'] and results['solver_model'] == m['solver_model'] and results['model_path'] == m['model'] and
                results['ground_calls'] == 0 and results['call_set_errors'] == [] and len(results['questions']) == len(questions) and results['selected_questions'] == chosen,
                side + ':results_complete_no_ground_and_exact_selection_ids')
        sides[side] = {'calls': len(rows), 'overall': aggregate(replayed),
                       'new_children': aggregate([r for r in replayed if r['candidate_id'] not in ctx['controls']]),
                       'historical_controls': aggregate([r for r in replayed if r['candidate_id'] in ctx['controls']]),
                       'by_gold_count': {str(g): aggregate([r for r in replayed if len(by_qid[r['candidate_id']]['gold']) == g]) for g in range(2, 7)},
                       'per_question': summaries, 'calls_replayed': replayed, 'selected_ids': chosen}
        selections[side] = chosen
    a.check(dt(a.read(HOME / 'TRAINED_EVALUATION.COMPLETE.json')['time']) <= dt(a.read(HOME / 'BASE_EVALUATION.STARTED.json')['time']), 'trained_completes_before_base_begins')
    l = {r['request_id']: r for r in sides['trained']['calls_replayed']}; r = {r['request_id']: r for r in sides['base']['calls_replayed']}
    a.check(set(l) == set(r) and all(all(l[rid][k] == r[rid][k] for k in ('request_id','candidate_id','root_id','question_type','variant_id','slot','seed')) for rid in l), 'paired_actual_response_identity_and_seed')
    selected_file = a.read(HOME / 'SELECTED_CANDIDATES.json')
    selected_questions = [q for q in questions if q['candidate_id'] in selections['trained']]
    a.check(selected_file['questions'] == selected_questions and selected_file['provisional'] is True and selected_file['training_allowed'] is False and
            selected_file['formal_training_admission'] is False, 'selected_pool_exact_trained_policy_and_provisional')
    mixed = a.read(HOME / 'MIXED_EVALUATION_REPORT.json')
    a.check(mixed['status'] == 'PASS' and mixed['evaluation_status'] == 'COMPLETED' and mixed['calls_per_model'] == ctx['expected_calls'] and
            mixed['training_started'] is False and mixed['formal_training_admission'] is False and
            mixed['independent_result_audit_required_before_admission'] is True and mixed['selected_candidate_ids'] == selections['trained'] and
            mixed['selected_new_candidate_count'] == sum(q['candidate_id'] not in ctx['controls'] for q in selected_questions), 'mixed_report_counts_selection_no_training')
    multi = mixed['by_type']['multi_select']; ordering = mixed['by_type']['ordering']
    a.check(multi['content_usable'] == sum(q['question_type']=='multi_select' for q in questions) and multi['content_excluded'] == len(ctx['assembled'])-sum(q['question_type']=='multi_select' for q in questions) and multi['mechanically_assembled'] == len(ctx['assembled']) and
            multi['solver_selected'] == len(selected_questions) and multi['selection_yield_among_content_usable'] == str(Fraction(sum(q['question_type']=='multi_select' for q in selected_questions), sum(q['question_type']=='multi_select' for q in questions))),
            'mixed_multiselect_candidate_and_selection_counts')
    ordering_mechanical = a.read(HOME / 'ordering/GENERATION.COMPLETE.json')['candidate_count']
    ordering_review = a.read(HOME / 'ordering/CONTENT_REVIEW.json')
    a.check(ordering['content_usable'] == len(ctx['ordering_pool']) == ordering_review['accepted_count'] and
            ordering['content_excluded'] == ordering_review['rejected_count'] and ordering['mechanically_assembled'] == ordering_mechanical and
            ordering['solver_selected'] == 0 and all(ordering['models'][side] == aggregate([]) for side in MODELS),
            'mixed_ordering_zero_usable_with_mechanical_and_content_exclusion_counts')
    for side in sides:
        a.check(mixed['by_type']['multi_select']['models'][side] == sides[side]['overall'], 'mixed_report_aggregate_matches_independent:' + side)
        for g in range(2, 7):
            a.check(mixed['by_type']['multi_select']['by_gold_count'][str(g)]['models'][side] == sides[side]['by_gold_count'][str(g)], 'mixed_report_gold_aggregate_matches_independent:' + side + ':' + str(g))
    for g in range(2, 7):
        row = multi['by_gold_count'][str(g)]
        a.check(row['content_usable'] == sum(len(q['gold']) == g for q in questions) and
                row['solver_selected'] == sum(len(q['gold']) == g for q in selected_questions) and
                row['role'] == ('historical_six_supported_control' if g == 6 else 'new_child'), 'gold_distribution_selection_and_role:' + str(g))
    pairs = a.read(HOME / 'PAIRED_CALLS.json')
    a.check(pairs['training_started'] is False and len(pairs['calls']) == ctx['expected_calls'] and
            Counter(row['request_id'] for row in pairs['calls']) == Counter({rid: 1 for rid in l}), 'paired_calls_complete_no_training')
    for pair in pairs['calls']:
        rid = pair['request_id']
        for side in ('trained', 'base'):
            independent = l[rid] if side == 'trained' else r[rid]
            recorded = pair[side]; scored = independent['independent_score']
            a.check(all(pair[k] == independent[k] == recorded[k] for k in ('request_id','candidate_id','root_id','question_type','variant_id','slot','seed')) and
                    recorded['exact'] == scored['exact_match'] and recorded['partial'] == scored['partial_score_fraction'] and
                    all(recorded[k] == scored[k] for k in ('tp','fp','fn')) and recorded['selected_count'] == len(scored['display_prediction']) and
                    recorded['gold_count'] == len(scored['content_gold']) and recorded['response'] == independent['response'], 'paired_call_independent_score:' + side + ':' + rid)
    for row in mixed['files']:
        a.check(a.bind(row['path'])['sha256'] == row['sha256'], 'mixed_report_final_binding:' + row['path'])
    details = a.read(HOME / 'evidence/RESULT_DETAILS.json')
    a.check(details['report'] == a.bind(HOME / 'MIXED_EVALUATION_REPORT.json'), 'result_details_report_binding')
    for side, values in sides.items():
        for name, key in (('new_child', 'new_children'), ('historical_control', 'historical_controls'), ('all_questions', 'overall')):
            expected = values[key]; actual = details['metrics'][side][name]
            a.check(all(actual[k] == expected[k] for k in ('calls', 'exact_correct', 'partial_mean')) and
                    actual['exact_fraction'] == expected['exact_accuracy'] and
                    actual['exact_percent'] == round(100 * float(Fraction(expected['exact_accuracy'])), 4),
                    'result_details_role_metrics:' + side + ':' + name)
        for question in details['questions']:
            qid = question['candidate_id']; q = by_qid[qid]
            summary = next(r for r in values['per_question'] if r['candidate_id'] == qid)
            a.check(question['root_id'] == q['root_id'] and question['gold_count'] == len(q['gold']) and
                    question['role'] == ('historical_control' if qid in ctx['controls'] else 'new_child') and
                    question['models'][side] == {k: summary[k] for k in ('full_exact_rate', 'full_partial_mean', 'selected_by_policy', 'selection_reason')},
                    'result_details_per_question:' + side + ':' + qid)
    a.check(len(details['questions']) == len(questions) and {r['candidate_id'] for r in details['questions']} == set(by_qid), 'result_details_question_coverage')
    return {'models': sides, 'selected_by_gold_count': {str(g): sum(len(q['gold']) == g for q in selected_questions) for g in range(2, 7)},
            'runtime_execution_audit': {'status': 'PASS', 'added_environment': expected_env, 'removed_environment':['PYTHONPATH'],
                                         'frozen_code_and_call_plan_unchanged': True,
                                         'receipt': a.bind(HOME / 'evidence/EVALUATION_RUNTIME_RECEIPT.json'),
                                         'basis': 'Exact clean-environment invocation/log bindings, frozen preparations, independently reconstructed plans and all actual response identities.'},
            'selected_new_count': sum(q['candidate_id'] not in ctx['controls'] for q in selected_questions),
            'selected_historical_controls_count': sum(q['candidate_id'] in ctx['controls'] for q in selected_questions),
            'selected_new_ids': [q['candidate_id'] for q in selected_questions if q['candidate_id'] not in ctx['controls']],
            'selected_historical_control_ids': [q['candidate_id'] for q in selected_questions if q['candidate_id'] in ctx['controls']]}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--inputs', action='store_true'); args = parser.parse_args()
    output = HOME / 'evidence' / ('INPUT_AUDIT.json' if args.inputs else 'RESULT_AUDIT.json')
    if output.exists():
        raise RuntimeError('refuse_overwrite:' + str(output))
    a = Audit(); report = {'version': 'v79_independent_result_audit_v1', 'created_at_utc': now(),
                           'phase': 'inputs_only' if args.inputs else 'paired_results',
                           'independent_implementation': True, 'native_or_mixed_score_functions_imported': False,
                           'new_model_or_http_calls': 0, 'training_performed': False,
                           'frozen_artifacts_modified': False, 'source_answer_semantics_recertified_by_this_audit': False}
    try:
        selfcheck(a)
        cached = {}
        if not args.inputs and (HOME / 'evidence/INPUT_AUDIT.json').exists():
            prior = a.read(HOME / 'evidence/INPUT_AUDIT.json')
            cached = {r['path']: r for r in prior.get('checkpoint_verification', [])}
        ctx = audit_inputs(a, cached)
        report['candidate_count'] = len(ctx['questions'])
        report['content_gold_distribution'] = ctx['gold_distribution']
        report['expected_calls_per_model'] = ctx['expected_calls']
        report['ordering_candidate_count'] = sum(q['question_type']=='ordering' for q in ctx['questions'])
        if not args.inputs:
            report.update(audit_results(a, ctx))
    except BaseException as error:
        a.check(False, 'audit_execution_exception', {'error': repr(error), 'traceback': traceback.format_exc()})
    report['status'] = 'FAIL' if any(c['status'] == 'FAIL' for c in a.checks) else 'PASS'
    report['checks'] = a.checks
    report['failed_checks'] = [c for c in a.checks if c['status'] == 'FAIL']
    report['check_count'] = len(a.checks)
    report['checkpoint_verification'] = list(a.weights.values())
    report['files'] = sorted(a.files.values(), key=lambda r: r['path']) + [{'path': str(Path(__file__).resolve()), 'sha256': digest(__file__)}]
    report['limitations'] = ['Result and execution audit does not replace source semantic review.',
                             'The four new six-option combinations reuse per-fact replacement sentences from v78 and remain dependent siblings; no new root or independent fact is established.',
                             'No ordering candidate passed source review in this batch; ordering difficulty/reliability cannot be estimated.',
                             'On-disk records establish the frozen invocation contract and executed call set; unsaved process environment cannot be reconstructed retroactively.']
    with output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(json.dumps({'path': str(output), 'status': report['status'], 'checks': report['check_count'], 'failed_checks': report['failed_checks']}, ensure_ascii=False))
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
