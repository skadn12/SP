import json, re, hashlib
from collections import Counter
from fractions import Fraction
from pathlib import Path
from datetime import datetime, timezone

HERE = Path('/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/base_qa_free_generation_solver_eval_20261103_run01')
read = lambda p: json.loads(p.read_text())
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
requests = read(HERE / 'REQUESTS.json')
raw_rows = read(HERE / 'RAW_ANSWERS.json')
scores = read(HERE / 'SCORES.json')
items = read(HERE / 'ITEMS_PRIVATE.json')
errors, checks = [], Counter()

def check(value, name, detail=None):
    checks[name] += 1
    if not value:
        errors.append({'check': name, 'detail': detail})

check(len(requests) == len(raw_rows) == len(scores['requests']) == 96, 'fixed_request_count')
check(len({r['request_id'] for r in requests}) == len(requests), 'unique_request_ids')
request_by_id = {r['request_id']: r for r in requests}
raw_by_id = {r['request_id']: r for r in raw_rows}
score_by_id = {r['request_id']: r for r in scores['requests']}
check(set(request_by_id) == set(raw_by_id) == set(score_by_id), 'complete_matching_call_sets')
item_by_id = {r['candidate_id']: r for r in items}
check(len(items) == 3, 'three_candidates')
bindings = read(HERE / 'INPUT_BINDINGS.json')
for binding in bindings['files']:
    check(digest(Path(binding['path'])) == binding['sha256_bytes'], 'input_file_hash', binding['path'])
for binding in bindings['prompts']:
    check(digest(Path(binding['path'])) == binding['sha256'], 'prompt_binding_hash', binding['path'])

independent = []
for rid, request in request_by_id.items():
    raw, observed = raw_by_id[rid], score_by_id[rid]
    item = item_by_id[request['candidate_id']]
    mapping = request['display_to_option_id']
    labels, ids = set(mapping), set(mapping.values())
    canonical_text = {item['label_to_option_id'][r['id']]: r['text'] for r in item['options']}
    check(len(mapping) == len(ids) == len(canonical_text), 'mapping_bijection', rid)
    check(ids == set(canonical_text), 'mapping_canonical_ids', rid)
    check(all(mapping[r['id']] in canonical_text and canonical_text[mapping[r['id']]] == r['text'] for r in request['options']), 'option_text_binding', rid)
    check(request['gold_option_ids'] == item['gold_option_ids'], 'canonical_gold_binding', rid)
    gold = request['gold_option_ids']
    if request['question_type'] == 'ordering':
        check([mapping[x] for x in request['gold_labels']] == gold, 'display_gold_order_mapping', rid)
        check((request['n'], request['g']) == (30, 15), 'ordering_directed_pair_units', rid)
    else:
        check({mapping[x] for x in request['gold_labels']} == set(gold), 'display_gold_set_mapping', rid)
        check(request['n'] == len(labels) and request['g'] == len(set(gold)), 'multi_choice_units', rid)
    check(digest(HERE / request['prompt_path']) == request['prompt_sha256'], 'request_prompt_hash', rid)
    prompt = (HERE / request['prompt_path']).read_text()
    source = read(Path(item['source_path']))
    expected_turns = source['turns'] if request['context_mode'] == 'full_context' else []
    check(request['turn_count'] == len(expected_turns), 'request_turn_count', rid)
    if expected_turns:
        check(all(f"[{row['turn_ref']}] {row['speaker']}: {row['text']}" in prompt for row in expected_turns), 'complete_untruncated_dialogue', rid)
    else:
        check('[No dialogue supplied.]' in prompt, 'no_context_prompt', rid)
    check(request['prompt_token_count'] + 32 <= 16384, 'context_budget', rid)
    if raw.get('prompt_sha256'):
        check(raw['prompt_sha256'] == request['prompt_sha256'], 'raw_prompt_binding', rid)
    text = raw.get('raw_prediction')
    delimiter = ',' if request['question_type'] == 'multi_select' else '>'
    # Independent scanner: no reused implementation or scorer imports.
    parsed = [x.strip() for x in text.strip().split(delimiter)] if isinstance(text, str) else None
    valid = bool(raw.get('status') == 'RETURNED' and raw.get('finish_reason') in {'stop', 'eos'}
                 and parsed and all(x in labels for x in parsed) and len(parsed) == len(set(parsed)))
    if valid and request['question_type'] == 'ordering':
        valid = set(parsed) == labels
    prediction = [mapping[x] for x in parsed] if valid else None
    tp = fp = fn = None
    exact, j, pair_accuracy = False, Fraction(), 0.0
    if valid:
        if request['question_type'] == 'multi_select':
            pred_set, gold_set = set(prediction), set(gold)
            tp, fp, fn = len(pred_set & gold_set), len(pred_set-gold_set), len(gold_set-pred_set)
            j = Fraction(tp, len(gold_set)) - Fraction(fp, len(labels)-len(gold_set))
            exact = pred_set == gold_set
            prediction = sorted(pred_set)
            pair_accuracy = tp / len(gold_set)
        else:
            gold_pairs = {(gold[i], gold[k]) for i in range(len(gold)) for k in range(i+1, len(gold))}
            pred_pairs = {(prediction[i], prediction[k]) for i in range(len(prediction)) for k in range(i+1, len(prediction))}
            tp, fp, fn = len(gold_pairs & pred_pairs), len(pred_pairs-gold_pairs), 0
            j = Fraction(tp-fp, len(gold_pairs))
            exact = prediction == gold
            pair_accuracy = tp / len(gold_pairs)
    expected = {'valid': valid, 'exact_correct': exact, 'J_fraction': str(j), 'tp': tp, 'fp': fp, 'fn': fn,
                'predicted_option_ids': prediction}
    for field, value in expected.items():
        check(observed.get(field) == value, 'independent_score_' + field, {'request_id': rid, 'expected': value, 'actual': observed.get(field)})
    check(abs(observed['J'] - float(j)) < 1e-12, 'independent_score_numeric_J', rid)
    independent.append({'request_id': rid, 'candidate_id': request['candidate_id'], 'context_mode': request['context_mode'],
                        'variant_id': request['variant_id'], 'repeat_index': request['repeat_index'],
                        **expected, 'recall_or_pair_accuracy': pair_accuracy})

summaries = {}
for cid, item in item_by_id.items():
    qtype = item['question_type']
    expected_variants, expected_repeats = (6, 3) if qtype == 'multi_select' else (12, 1)
    paired = {}
    summaries[cid] = {}
    for mode in ['full_context', 'no_context']:
        planned = [r for r in requests if r['candidate_id'] == cid and r['context_mode'] == mode]
        counted = Counter(r['variant_id'] for r in planned)
        check(len(counted) == expected_variants and set(counted.values()) == {expected_repeats}, 'legacy_slot_protocol', {'candidate_id': cid, 'mode': mode})
        orders = {tuple(r['display_to_option_id'].values()) for r in planned}
        check(len(orders) == expected_variants, 'unique_display_permutations', {'candidate_id': cid, 'mode': mode})
        for r in planned:
            key = (r['variant_id'], r['repeat_index'])
            if mode == 'full_context':
                paired[key] = r
            else:
                ref = paired[key]
                check(all(r[k] == ref[k] for k in ['sampling_seed', 'display_to_option_id', 'gold_labels', 'options']), 'paired_context_conditions', r['request_id'])
        rows = [r for r in independent if r['candidate_id'] == cid and r['context_mode'] == mode]
        exact = sum(r['exact_correct'] for r in rows)
        mean_j = sum((Fraction(r['J_fraction']) for r in rows), Fraction()) / len(rows)
        summary = {'calls': len(rows), 'valid_calls': sum(r['valid'] for r in rows), 'exact_correct_calls': exact,
                   'mean_J_fraction': str(mean_j), 'mean_J': float(mean_j),
                   'known_FP': sum(r['fp'] or 0 for r in rows), 'known_FN': sum(r['fn'] or 0 for r in rows)}
        observed_summary = scores['items'][cid][mode]
        for field in ['valid_calls', 'exact_correct_calls', 'mean_J_fraction', 'known_FP', 'known_FN']:
            check(summary[field] == observed_summary[field], 'aggregate_' + field, {'candidate_id': cid, 'mode': mode})
        summaries[cid][mode] = summary

result = {'schema_version': 'independent_free_generation_solver_score_audit_v1',
          'created_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'PASS' if not errors else 'FAIL',
          'method': 'Independent standard-library scanner and Fraction arithmetic; no imported evaluation modules or model calls',
          'scoring': {'multi_select': 'TP/g - FP/(n-g)', 'ordering': '(concordant-discordant)/15 = directed-pair J',
                      'invalid': '0 in fixed planned denominator; no repaired outputs', 'ordering_fn': '0 as in historical native scorer; wrong_pairs reports reversed Gold relations'},
          'checks': dict(checks), 'errors': errors,
          'bound_inputs': {name: digest(HERE/name) for name in ['REQUESTS.json','RAW_ANSWERS.json','SCORES.json','ITEMS_PRIVATE.json']},
          'item_summaries': summaries, 'independent_requests': independent,
          'diagnostic_only': True, 'formal_qa_eligible': False}
destination = HERE / 'INDEPENDENT_SCORE_AUDIT.json'
with destination.open('x') as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(json.dumps({'path': str(destination), 'status': result['status'], 'errors': errors, 'summaries': summaries}, ensure_ascii=False, indent=2))
