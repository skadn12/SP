"""Read-only reconstruction of paired Solver requests and raw response results.

Never imports evaluator, scorer, selection code, or a model. A tokenizer is used
only to reconstruct public prompts and count input tokens. An explicit --record
path is the only persistent write; it uses exclusive creation.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import random
import re
import sys

sys.dont_write_bytecode = True
SEED = 2026100800
BASE = Path('/root/autodl-tmp/Qwen3-4B')
TRAINED = Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_common_dev_eval_v1_live_3/clean_merged_model')
AUDIT_SHAS = {'base': '59e23710e2373a00aef3b7d50d117d4089eea47cd17fd16a00eebcc83fba0035',
              'trained': 'add007d4e4f937f2ea1b7e124ccac3c53e5eda5b8bbed1beab3dd4b0855cadd0'}
SYSTEM = ('You are a strict structured dialogue QA solver. Use only the supplied dialogue when it is present. '
          'Return the final answer in exactly one <box>...</box> pair. Do not provide explanations, '
          'reasoning, or any text outside the box.')


def read(path):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError('duplicate JSON key: ' + key)
            out[key] = value
        return out
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8388608), b''):
            h.update(chunk)
    return h.hexdigest()


def cid(text):
    return 'content_' + hashlib.sha256(text.strip().encode()).hexdigest()[:16]


def average(values):
    return sum(values, Fraction()) / len(values) if values else None


def variants(q):
    field = 'options' if q['question_type'] == 'multi_select' else 'items'
    labels = list(q[field])
    canonical = [cid(q[field][x]) for x in labels]
    texts = {cid(t): t for t in q[field].values()}
    def generator(index):
        h = hashlib.sha256(f"{SEED}|{q['candidate_id']}|{index}".encode()).digest()
        return random.Random(int.from_bytes(h[:8], 'big'))
    generator(-1).shuffle(canonical)
    seen = set()
    for index in range(6 if field == 'options' else 12):
        rng = generator(index)
        while True:
            display = canonical[:]
            rng.shuffle(display)
            if tuple(display) not in seen:
                seen.add(tuple(display))
                break
        yield {'variant_id': f'perm_{index:02d}', 'canonical_order_content_ids': canonical[:],
               'display_order_content_ids': display, field: dict(zip(labels, [texts[x] for x in display])),
               'display_to_content': dict(zip(labels, display)), 'content_to_text': texts}


def raw_score(q, display, raw):
    """Parse independently, then use set arithmetic or pair inversions."""
    field = 'options' if q['question_type'] == 'multi_select' else 'items'
    gold = [cid(q[field][label]) for label in q['gold']]
    if not isinstance(raw, str):
        return {'status': 'PARSE_FAILURE'}
    box = re.fullmatch(r'<box>\s*([^<>]*?)\s*</box>', raw.strip(), re.S)
    if box is None:
        return {'status': 'PARSE_FAILURE'}
    labels = [x.strip().upper() for x in box[1].split(',')] if box[1].strip() else []
    if len(labels) != len(set(labels)) or any(re.fullmatch(r'[A-Z][A-Z0-9_]*', x) is None for x in labels):
        return {'status': 'PARSE_FAILURE'}
    if set(labels) - set(display['display_to_content']):
        return {'status': 'INVALID_ID'}
    prediction = [display['display_to_content'][x] for x in labels]
    if len(prediction) != len(set(prediction)):
        return {'status': 'DUPLICATE_CONTENT_ID'}
    if field == 'options':
        ps, gs = set(prediction), set(gold)
        tp, fp, fn = len(ps & gs), len(ps - gs), len(gs - ps)
        partial = Fraction(tp, len(gs)) - (Fraction(fp, 6 - len(gs)) if len(gs) < 6 else 0)
        exact, prediction = ps == gs, sorted(ps)
    else:
        if len(prediction) != 6 or set(prediction) != set(gold):
            return {'status': 'INCOMPLETE_ORDER'}
        pos = {v: k for k, v in enumerate(prediction)}
        fp = sum(pos[a] > pos[b] for a, b in itertools.combinations(gold, 2))
        tp, fn = 15 - fp, 0
        partial, exact = Fraction(15 - 2 * fp, 15), prediction == gold
    return {'status': 'SCORED', 'partial': partial, 'exact': exact, 'tp': tp, 'fp': fp, 'fn': fn,
            'display_prediction': labels, 'content_prediction': prediction, 'content_gold': gold}


def prompt(tokenizer, q, v, source):
    multi = q['question_type'] == 'multi_select'
    field, heading = ('options', 'Options') if multi else ('items', 'Items')
    dialogue = '\n'.join(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" for t in source['turns'])
    answer_rule = ('Output only selected letters separated by commas, e.g. <box>A,C</box>.' if multi else
                   'Output only ordered item IDs separated by commas, e.g. <box>I2,I4,I1,I3</box>.')
    body = '\n\n'.join(['Dialogue:\n' + dialogue, f"Question ({q['question_type']}):\n{q['question']}",
                         heading + ':\n' + '\n'.join(f'{k}. {t}' for k, t in v[field].items()), answer_rule])
    return tokenizer.apply_chat_template([{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': body}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)


def run(directory, preflight_only=False, only_side=None):
    started = datetime.now(timezone.utc).isoformat()
    errors, checked, counts = [], {}, Counter()
    def check(condition, message):
        if not condition:
            errors.append(message)
        return bool(condition)
    def bind(path, expected=None):
        path = Path(path).resolve()
        if str(path) not in checked:
            checked[str(path)] = digest(path)
        if expected is not None:
            check(checked[str(path)] == expected, 'hash:' + str(path))
        return checked[str(path)]
    version = directory.name.split('_v', 1)[1].split('_', 1)[0]
    batch = directory.parent / ('h4_codex_scaled_production_v' + version)
    expected = read(batch / 'EVALUATION_COUNTS.json')['evaluation_counts']
    pool_document = read(directory / 'EVALUATION_POOL.json')
    pool = pool_document['questions']
    check(len(pool) == expected['candidate_count'] == len({q['candidate_id'] for q in pool}), 'pool count')
    check(Counter(q['question_type'] for q in pool) == expected['type_counts'], 'type counts')
    check(Counter(str(len(q['gold'])) for q in pool if q['question_type'] == 'multi_select') == expected['multi_gold_counts'], 'Gold counts')
    cap = read(batch / 'PRECALL_PLAN.json')['solver_calls_per_model_max']
    check(expected['calls_per_model'] <= cap, 'predeclared call cap')
    check(expected['calls_per_model'] == sum(18 if q['question_type'] == 'multi_select' else 12 for q in pool), 'frozen call formula')
    for manifest_path in [batch / 'COMPLETE_MANIFEST.json', directory / 'PREEXECUTION_CODE_BINDINGS.json',
                          directory / 'ADAPTER_BINDINGS.json', directory / 'PAIRED_PREFLIGHT.json']:
        if not manifest_path.exists():
            check(False, 'missing:' + str(manifest_path)); continue
        manifest = read(manifest_path); bind(manifest_path)
        for section in ('files', 'source_batch_files'):
            for row in manifest.get(section, []):
                path = Path(row['path']); path = path if path.is_absolute() else manifest_path.parent / path
                bind(path, row['sha256'])
                if 'bytes' in row:
                    check(path.stat().st_size == row['bytes'], 'bytes:' + str(path))
    preflight = read(directory / 'PAIRED_PREFLIGHT.json')
    check(preflight.get('status') == 'PASS' and preflight.get('evaluation_counts') == expected
          and preflight.get('candidate_count') == len(pool) and preflight.get('calls_per_model') == expected['calls_per_model']
          and preflight.get('same_prompts_permutations_seeds_decoder_scoring') is True
          and preflight.get('training_started') is False, 'preflight contract')
    check(preflight.get('adapter_binding', {}).get('path') == str(directory / 'ADAPTER_BINDINGS.json'), 'preflight adapter path')
    bind(directory / 'ADAPTER_BINDINGS.json', preflight.get('adapter_binding', {}).get('sha256'))
    public = read(batch / 'NEW_PUBLIC_QUESTIONS.json')['questions'] + read(batch / 'CONTROL_PUBLIC_QUESTIONS.json')['questions']
    private = read(batch / 'NEW_PRIVATE_GOLD.json')['questions'] + read(batch / 'CONTROL_PRIVATE_GOLD.json')['questions']
    public_by = {q['candidate_id']: q for q in public}; private_by = {q['candidate_id']: q for q in private}
    check(len(public_by) == len(public) == len(private_by) == len(private) == len(pool)
          and set(public_by) == set(private_by) == {q['candidate_id'] for q in pool}, 'sealed public/private pool sets')
    for q in pool:
        original, secret = public_by[q['candidate_id']], private_by[q['candidate_id']]
        for key in ('candidate_id', 'root_id', 'question_type', 'question', 'options', 'items', 'source_sha256'):
            if key in original: check(q.get(key) == original[key], 'sealed public field:' + q['candidate_id'] + ':' + key)
        check(q['gold'] == secret['gold'] and q['role'] == secret['role'], 'sealed Gold/role:' + q['candidate_id'])
        check(Path(q['source_path']).resolve() == (directory.parent.parent / original['source_path']).resolve(), 'sealed source path:' + q['candidate_id'])
    provenance = pool_document['provenance']
    check(provenance.get('backend') == 'CURRENT_CODEX_SESSION' and provenance.get('exact_model_id') is None
          and provenance.get('source_batch_id') == batch.name, 'honest generator provenance')
    claim = provenance.get('source_batch_manifest', {})
    check(claim.get('path') == str(batch / 'COMPLETE_MANIFEST.json'), 'generator content seal path')
    bind(batch / 'COMPLETE_MANIFEST.json', claim.get('sha256'))
    manifests = {side: read(directory / 'run' / side / 'manifest.json') for side in ('base', 'trained')}
    b, t = manifests['base'], manifests['trained']
    exclusions = {'prepared_at_utc', 'model', 'solver_model', 'solver_identity'}
    check({k: v for k, v in b.items() if k not in exclusions} == {k: v for k, v in t.items() if k not in exclusions}, 'paired manifests')
    check(b['requests'] == t['requests'], 'paired request objects')
    for name in ('questions.json', 'gold.json', 'canonical_bindings.json'):
        left, right = directory / 'run/base' / name, directory / 'run/trained' / name
        check(left.read_bytes() == right.read_bytes(), 'paired bytes:' + name)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(BASE), local_files_only=True)
    runs, summaries = {}, {}
    for side, model_name, model_path in [('base', 'M0', BASE), ('trained', 'M_clean_policy', TRAINED)]:
        if only_side and side != only_side: continue
        root = directory / 'run' / side; manifest = manifests[side]
        check(manifest['solver_model'] == model_name and manifest['model'] == str(model_path), 'model identity:' + side)
        check(manifest['tokenizer'] == str(BASE), 'tokenizer:' + side)
        check(manifest['execution'] == 'full_only_no_ground_v1', 'full only:' + side)
        check(manifest['policy_version'] == 'v15_full_only_random_mcq6x3_order12x1_v1', 'protocol:' + side)
        check(manifest['random_display'] == {'version': 'independent_seeded_permutation_v2', 'seed': SEED,
                  'gold_blind': True, 'cyclic_rebinding': False, 'shape_filter': False}, 'display contract:' + side)
        for flag in ('training_started', 'training_promotion'):
            check(manifest[flag] is False, flag + ':' + side)
        check(manifest['new_generation_calls'] == 0, 'no generation:' + side)
        decoder = manifest['decoder_contract']
        for key, value in {'temperature': .8, 'top_p': .9, 'max_tokens': 32, 'enable_thinking': False,
                           'structured_output_api': 'structured_outputs=StructuredOutputsParams(choice)',
                           'structured_output_version': 'finite_choice_set_v1', 'stop': ['</box>'],
                           'include_stop_str_in_output': True}.items():
            check(decoder.get(key) == value, 'decoder:' + side + ':' + key)
        identity = read(root / 'solver_identity.json')
        check(identity == manifest['solver_identity'] and identity['model_identifier'] == model_name, 'bound identity:' + side)
        check(identity['model_path'] == str(model_path) and identity['tokenizer_path'] == str(BASE)
              and identity['audit_sha256'] == AUDIT_SHAS[side] and identity['tokenizer_audit_sha256'] == AUDIT_SHAS['base'],
              'fixed checkpoint identity:' + side)
        if side == 'trained':
            check(identity.get('model_fingerprint_sha256') == '039fa9969ecff33b167ecf62a947fac872ac13905d05947ca2e7825c43d1ca90'
                  and identity.get('adapter_enabled') is True and identity.get('adapter_already_merged') is True, 'fixed trained adapter identity')
        for key in ('audit', 'tokenizer_audit'):
            bind(identity[key + '_path'], identity[key + '_sha256'])
        fixed_base_files = read(identity['tokenizer_audit_path'])['files']
        if side == 'base':
            fixed_files = fixed_base_files
        else:
            fixed_trained = read(identity['audit_path'])['models'][model_name]
            check(fixed_trained['inference_model_path'] == str(model_path)
                  and fixed_trained['model_fingerprint_sha256'] == identity['model_fingerprint_sha256'], 'fixed trained audit linkage')
            fixed_files = [{'path': str(model_path / row['name']), 'name': row['name'], 'bytes': row['size_bytes'], 'sha256': row['sha256']}
                           for row in fixed_trained['merged_model_identity']['files']]
            fixed_files += [row for row in fixed_base_files if row['name'] in {'tokenizer.json', 'tokenizer_config.json', 'merges.txt', 'vocab.json'}]
        check(identity['checkpoint_files'] == fixed_files, 'fixed audit checkpoint linkage:' + side)
        for row in fixed_files:
            check(Path(row['path']).stat().st_size == row['bytes'], 'checkpoint size:' + row['path'])
            bind(row['path'], row['sha256'])
        for path, h in manifest['code_sha256'].items(): bind(path, h)
        for path, h in manifest['input_sha256'].items(): bind(manifest['pool_path'] if path == 'pool' else root / path, h)
        check(Path(manifest['pool_path']).resolve() == (directory / 'EVALUATION_POOL.json').resolve()
              and manifest['pool_sha256'] == bind(directory / 'EVALUATION_POOL.json')
              and manifest['generator_provenance'] == provenance, 'run pool/provenance:' + side)
        request_ids = [r['request_id'] for r in manifest['requests']]
        check(len(set(request_ids)) == len(request_ids) == manifest['request_count'] == expected['calls_per_model'], 'request count:' + side)
        request_by = {r['request_id']: r for r in manifest['requests']}
        for index, request in enumerate(manifest['requests']):
            check(type(request['seed']) is int and request['seed'] == SEED + index, 'seed:' + side + ':' + request['request_id'])
        raw_by = {}
        if not preflight_only:
            for path in sorted((root / 'responses').glob('*.json')):
                row = read(path); bind(path)
                check(row['request_id'] not in raw_by and path.stem == row['request_id'], 'duplicate response:' + str(path))
                raw_by[row['request_id']] = row
            check(set(raw_by) == set(request_by), 'response set:' + side)
            complete = read(root / 'COMPLETE.json'); bind(root / 'COMPLETE.json'); bind(root / 'STARTED.json')
            check(complete['status'] == 'COMPLETE' and complete['actual_call_count'] == len(raw_by), 'completion:' + side)
            check(not (root / 'FAILED.json').exists(), 'failed run:' + side)
            check(read(root / 'STARTED.json')['manifest_sha256'] == bind(root / 'manifest.json'), 'started binding:' + side)
            native = read(root / 'RESULTS.json'); bind(root / 'RESULTS.json')
            native_by = {q['candidate_id']: q for q in native['questions']}
        question_rows, call_rows = [], []
        gold_file = read(root / 'gold.json'); bindings = {q['candidate_id']: q for q in read(root / 'canonical_bindings.json')['candidates']}
        for q in pool:
            qid = q['candidate_id']; field = 'options' if q['question_type'] == 'multi_select' else 'items'
            label_map = {label: cid(value) for label, value in q[field].items()}; gold = [label_map[x] for x in q['gold']]
            check(gold_file[qid] == gold == bindings[qid]['gold_content_ids'], 'canonical Gold:' + side + ':' + qid)
            check(bindings[qid]['content_id_by_label'] == label_map, 'canonical labels:' + side + ':' + qid)
            source = read(q['source_path']); bind(q['source_path'], q['source_sha256'])
            bind(root / 'sources' / (q['root_id'] + '.json'), q['source_sha256'])
            per_display, qscores, correct, valid = {}, [], 0, 0
            planned = [x for x in manifest['requests'] if x['candidate_id'] == qid]
            expected_variants = list(variants(q)); rollouts = range(3 if field == 'options' else 1)
            check({(x['variant']['variant_id'], x['slot']) for x in planned} == {(v['variant_id'], k) for v in expected_variants for k in rollouts}, 'display/slot set:' + qid)
            for v in expected_variants:
                text = prompt(tokenizer, q, v, source); encoded = len(tokenizer.encode(text, add_special_tokens=False)); values = []
                for slot in rollouts:
                    rid = f"{qid}__{v['variant_id']}__full_context__s{slot:02d}"; r = request_by[rid]
                    check(r['variant'] == v, 'permutation:' + side + ':' + rid)
                    check(r['candidate_id'] == qid and r['root_id'] == q['root_id'] and r['question_type'] == q['question_type'] and r['mode'] == 'full_context', 'request identity:' + side + ':' + rid)
                    path = root / 'prompts' / (rid + '.txt')
                    check(path.read_bytes() == text.encode(), 'public prompt:' + side + ':' + rid); bind(path, r['prompt_sha256'])
                    check(type(r['prompt_tokens']) is int and r['prompt_tokens'] == encoded and encoded + 32 <= 16384, 'prompt tokens:' + side + ':' + rid)
                    other = directory / 'run' / ('trained' if side == 'base' else 'base') / 'prompts' / path.name
                    check(path.read_bytes() == other.read_bytes(), 'paired prompt bytes:' + rid)
                    counts['requests_checked'] += 1
                    if preflight_only: continue
                    row = raw_by[rid]; meta = row.get('metadata', {}); before = len(errors)
                    for key, value in {'request_id': rid, 'candidate_id': qid, 'root_id': q['root_id'], 'solver_model': model_name,
                        'model_path': str(model_path), 'question_type': q['question_type'], 'variant_id': v['variant_id'], 'mode': 'full_context', 'slot': slot, 'seed': r['seed']}.items():
                        check(row.get(key) == value, 'response identity:' + side + ':' + rid + ':' + key)
                    for key in ('temperature', 'top_p', 'max_tokens', 'structured_output_api', 'structured_output_version'):
                        check(meta.get(key) == decoder[key], 'raw decoder:' + side + ':' + rid + ':' + key)
                    check(type(row.get('seed')) is int and type(row.get('slot')) is int
                          and type(meta.get('sampling_seed')) is int and type(meta.get('prompt_token_count')) is int
                          and meta['sampling_seed'] == r['seed'] and meta['prompt_token_count'] == encoded, 'raw tokens/seed:' + side + ':' + rid)
                    if 'enable_thinking' in meta: check(meta['enable_thinking'] is False, 'raw thinking:' + side + ':' + rid)
                    check(type(meta.get('completion_token_count')) is int and 0 < meta['completion_token_count'] <= 32, 'completion tokens:' + side + ':' + rid)
                    check(meta.get('finish_reason') in ('stop', 'eos'), 'unfinished:' + side + ':' + rid)
                    score = raw_score(q, v, row.get('raw_output')); native_score = row.get('score', {})
                    check(score['status'] == native_score.get('scoring_status'), 'raw status:' + side + ':' + rid)
                    check(score['status'] == 'SCORED', 'unscored:' + side + ':' + rid)
                    if score['status'] == 'SCORED':
                        check(native_score.get('parse_status') == 'success' and native_score.get('parse_reason') is None
                              and native_score.get('scored') is True, 'native parse flags:' + side + ':' + rid)
                        for key in ('tp', 'fp', 'fn', 'display_prediction', 'content_prediction', 'content_gold'):
                            check(score[key] == native_score.get(key), 'raw score:' + side + ':' + rid + ':' + key)
                        check(score['exact'] is native_score.get('exact_match') and str(score['partial']) == native_score.get('partial_score_fraction') == native_score.get('partial_score'), 'raw fraction:' + side + ':' + rid)
                    is_valid = len(errors) == before and score['status'] == 'SCORED'
                    check(row.get('evaluation_valid') is is_valid and (not is_valid or row.get('data_errors') == []), 'raw validity:' + side + ':' + rid)
                    counts['raw_responses_parsed_and_rescored'] += 1
                    counts['valid_responses'] += is_valid
                    if is_valid:
                        values.append(score['partial']); qscores.append(score['partial']); correct += score['exact']; valid += 1
                    call_rows.append({'candidate_id': qid, 'question_type': q['question_type'], 'gold_count': len(q['gold']), 'valid': is_valid,
                                      'exact': score.get('exact', False), 'partial': str(score['partial']) if 'partial' in score else None})
                if not preflight_only: per_display[v['variant_id']] = average(values)
            if preflight_only: continue
            complete_q = valid == len(expected_variants) * len(rollouts)
            mean = average(list(per_display.values())) if complete_q else None
            selected = 0 < mean < 1 if mean is not None else None
            exact = Fraction(correct, valid) if valid else None
            n = native_by[qid]
            reason = ('EVALUATION_INCOMPLETE_OR_INVALID' if mean is None else 'FULL_PARTIAL_IN_OPEN_INTERVAL' if selected else
                      'FULL_PARTIAL_NONPOSITIVE' if mean <= 0 else 'FULL_PARTIAL_PERFECT_NO_NEW_HARD_ITEM')
            for key, value in {'selected_by_policy': selected, 'full_partial_mean': str(mean) if mean is not None else None,
                'permutation_means': {k: str(v) for k, v in per_display.items() if v is not None}, 'full_exact_rate': str(exact) if exact is not None else None,
                'selection_reason': reason, 'ground_used': False, 'expected_full_calls': len(expected_variants) * len(rollouts), 'actual_full_calls': len(planned)}.items():
                check(n.get(key) == value, 'question aggregate:' + side + ':' + qid + ':' + key)
            if complete_q:
                check(n.get('permutation_score_min') == str(min(per_display.values()))
                      and n.get('permutation_score_max') == str(max(per_display.values()))
                      and n.get('permutation_score_range') == str(max(per_display.values()) - min(per_display.values())), 'display range:' + side + ':' + qid)
            check(n['policy_version'] == 'r02_groundless_full_only_random_position_v2' and n['data_errors'] == [], 'selection contract:' + side + ':' + qid)
            counts['question_aggregates_recomputed'] += 1
            question_rows.append({'candidate_id': qid, 'root_id': q['root_id'], 'question_type': q['question_type'], 'gold_count': len(q['gold']),
                                  'calls': len(planned), 'valid': valid, 'exact_correct': correct, 'exact_rate': str(exact),
                                  'partial_equal_display_mean': str(mean) if mean is not None else None, 'selected_by_policy': selected})
        if preflight_only: continue
        selected_ids = [x['candidate_id'] for x in question_rows if x['selected_by_policy']]
        check(native['selected_questions'] == selected_ids and native['call_set_errors'] == [], 'native selected IDs:' + side)
        runs[side] = question_rows
        groups = {}
        for category in ['Gold2-5', 'Gold6_control', 'ordering']:
            qs = [x for x in question_rows if (category == 'Gold2-5' and x['question_type'] == 'multi_select' and 2 <= x['gold_count'] <= 5)
                   or (category == 'Gold6_control' and x['question_type'] == 'multi_select' and x['gold_count'] == 6)
                   or (category == 'ordering' and x['question_type'] == 'ordering')]
            total, exact = sum(x['valid'] for x in qs), sum(x['exact_correct'] for x in qs)
            groups[category] = {'questions': len(qs), 'exact_correct': exact, 'valid_calls': total,
                                'exact_rate': str(Fraction(exact, total)) if total else None,
                                'question_equal_partial_mean': str(average([Fraction(x['partial_equal_display_mean']) for x in qs]))
                                if qs and all(x['partial_equal_display_mean'] is not None for x in qs) else None}
        summaries[side] = groups
    selected = []
    if not preflight_only:
        selected = [q for q in pool if next(x for x in runs['trained'] if x['candidate_id'] == q['candidate_id'])['selected_by_policy']
                    and not (q['question_type'] == 'multi_select' and len(q['gold']) == 6)] if 'trained' in runs else []
        audit_path = directory / 'AUDIT_RESULTS.json'
        if audit_path.exists():
            audit = read(audit_path); bind(audit_path)
            check(audit['status'] == 'VALID' and audit['data_errors'] == [], 'frozen audit status')
            for side in runs:
                aq = {x['candidate_id']: x for x in audit['models'][side]['questions']}
                for q in runs[side]:
                    a = aq[q['candidate_id']]
                    check(a['selected_by_policy'] is q['selected_by_policy'] and a['partial_mean']['fraction'] == q['partial_equal_display_mean']
                          and a['exact_correct'] == q['exact_correct'] and a['valid_calls'] == q['valid'], 'frozen audit question:' + side + ':' + q['candidate_id'])
        package_path = directory / 'training_candidate_package/FRONTIER_MANIFEST.json'
        if package_path.exists() and 'trained' in runs:
            package = read(package_path); bind(package_path)
            check([q['candidate_id'] for q in selected] == [q['candidate_id'] for q in package['candidates']], 'candidate package set/order')
            check(package['candidate_count'] == len(selected) and package['training_execution_authorized_by_this_file'] is False
                  and package['family_split_unit'] == 'root' and package['related_variants_are_not_independent'] is True, 'package contract')
            training_rows = [json.loads(line) for line in (package_path.parent / 'frontier_candidates.jsonl').read_text().splitlines()]
            check(len(training_rows) == len(selected), 'candidate export row count')
            for q, exported, metadata in zip(selected, training_rows, package['candidates']):
                field = 'options' if q['question_type'] == 'multi_select' else 'items'
                source = read(q['source_path']); heading = 'Options' if field == 'options' else 'Items'
                answer_rule = ('Output only selected letters separated by commas, e.g. <box>A,C</box>.' if field == 'options' else
                               'Output only ordered item IDs separated by commas, e.g. <box>I2,I4,I1,I3</box>.')
                chunks = ['Dialogue:\n' + '\n'.join(f"[{turn['turn_ref']}] {turn['speaker']}: {turn['text']}" for turn in source['turns']),
                          f"Question ({q['question_type']}):\n{q['question']}", heading + ':\n' + '\n'.join(f'{k}. {v}' for k, v in q[field].items()), answer_rule]
                messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': '\n\n'.join(chunks)},
                            {'role': 'assistant', 'content': '<box>' + ','.join(q['gold']) + '</box>'}]
                check(exported == {'messages': messages}, 'candidate export messages:' + q['candidate_id'])
                chat_sha = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                check(metadata['messages_sha256'] == chat_sha and metadata['canonical_gold'] == q['gold']
                      and metadata['root_id'] == q['root_id'] and metadata['source_sha256'] == q['source_sha256'], 'candidate export metadata:' + q['candidate_id'])
        selected_path = directory / 'SELECTED_CANDIDATES.json'
        if selected_path.exists() and 'trained' in runs:
            saved = read(selected_path); bind(selected_path)
            native_selected = [q['candidate_id'] for q in runs['trained'] if q['selected_by_policy']]
            check(native_selected == [q['candidate_id'] for q in saved['questions']], 'audit selected IDs')
        for manifest_path in [directory / 'EVALUATION_COMPLETE_MANIFEST.json', directory / 'training_candidate_package/COMPLETE_MANIFEST.json']:
            if not manifest_path.exists(): continue
            manifest = read(manifest_path); bind(manifest_path)
            for row in manifest['files']:
                path = Path(row['path']); path = path if path.is_absolute() else manifest_path.parent / path
                bind(path, row['sha256'])
                if 'bytes' in row: check(path.stat().st_size == row['bytes'], 'sealed bytes:' + str(path))
    method_matches = {}
    if version == '109':
        historical = directory.parent / 'h4_codex_scaled_production_v108_solver_eval'
        for name in ('evaluate.py', 'audit_results.py', 'build_package.py'):
            path = historical / name; bind(path); bind(directory / name)
            method_matches[name] = path.read_text().replace('v108', 'v109') == (directory / name).read_text()
            check(method_matches[name], 'v108 method namespace-only:' + name)
    return {'status': 'PASS' if not errors else 'FAIL', 'reviewer': '/root/scale_group2', 'started_at_utc': started,
            'finished_at_utc': datetime.now(timezone.utc).isoformat(), 'directory': str(directory), 'preflight_only': preflight_only,
            'only_side': only_side,
            'counts': dict(counts), 'summaries': summaries, 'question_aggregates': runs,
            'candidate_count': len(selected), 'candidate_ids': [q['candidate_id'] for q in selected],
            'candidate_type_counts': dict(Counter(q['question_type'] for q in selected)),
            'candidate_multi_gold_counts': dict(Counter(str(len(q['gold'])) for q in selected if q['question_type'] == 'multi_select')),
            'v108_method_namespace_only_equal': method_matches,
            'scope': ['All sealed source-batch and execution-code file hash bindings, frozen candidate denominator and call budget.',
                'Every paired public prompt byte, reconstructed chat template/token count, independent content-based display permutation, request slot and seed.',
                'Both fixed checkpoint audit hashes and their complete checkpoint-file lists, current checkpoint hashes and tokenizer identity.',
                'Every raw response identity, decoder metadata, finish reason, token counts, exact-box parse, set/pair score and rational fraction.',
                'Every question equal-display aggregation, exact-match rate, display range, native frontier selection and optional main-audit crosscheck.',
                'Final Gold2-5/ordering candidate selection, Gold6 exclusion, exported canonical chat messages and package manifests when present.'],
            'checked_file_sha256': checked, 'audit_script_sha256': digest(__file__), 'errors': sorted(set(errors)),
            'model_calls_started_by_this_reviewer': 0, 'semantic_gold_correctness_certified': False,
            'gold6_controls_excluded_from_new_candidates': True,
            'limitations': ['Execution, parsing, exact arithmetic, pairing and selection audit only; no new semantic Gold review.',
                'Reviewer shares inherited Codex model and project context and previously authored/reviewed part of content; this is no external independent-model certification.',
                'Sources belong to the training side. This is not an untouched-test performance estimate.',
                'Raw records omit a separate model_identifier; frozen native live code checked it, while saved model_path/solver_model, identity audits and checkpoint hashes are rechecked.',
                'Completion token counts need not be equal across models because generated replies differ; both are checked against the same 32-token limit.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--record', type=Path)
    parser.add_argument('--only-side', choices=('base', 'trained'))
    args = parser.parse_args()
    if args.record:
        assert not args.preflight_only and not args.only_side
        assert all((args.directory / name).exists() for name in ('AUDIT_RESULTS.json', 'SELECTED_CANDIDATES.json',
                   'EVALUATION_COMPLETE_MANIFEST.json', 'training_candidate_package/COMPLETE_MANIFEST.json'))
    result = run(args.directory.resolve(), args.preflight_only, args.only_side)
    if args.record:
        with args.record.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in {'checked_file_sha256', 'question_aggregates', 'candidate_ids', 'limitations'}}, ensure_ascii=False))
    if result['errors']: raise SystemExit(1)
