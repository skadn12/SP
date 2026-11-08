"""Independent post-freeze audit; never imports the scorer or inference runtime."""
from __future__ import annotations

import argparse
import collections
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
DAY2 = HERE.parent / 'day2'
ARMS = ('control', 'treatment')
COHORTS = ('discovery', 'replication')
CHECKS, ISSUES, HASH_CACHE = [], [], {}


def check(name, condition, detail=None):
    ok = bool(condition)
    CHECKS.append({'name': name, 'pass': ok})
    if not ok:
        ISSUES.append({'name': name, 'detail': detail})
    return ok


def need(name, condition):
    if not check(name, condition):
        raise RuntimeError(name)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in HASH_CACHE:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        HASH_CACHE[key] = digest.hexdigest()
    return HASH_CACHE[key]


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def bindings(rows, prefix):
    for index, row in enumerate(rows):
        path = Path(row['path'])
        need(f'{prefix}:exists:{index}', path.is_file())
        need(f'{prefix}:bytes:{index}', path.stat().st_size == row['bytes'])
        need(f'{prefix}:hash:{index}', sha(path) == row['sha256'])


def time(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def equal(name, actual, expected):
    if isinstance(expected, float) and isinstance(actual, (float, int)):
        return check(name, math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), {'actual': actual, 'expected': expected})
    return check(name, actual == expected, {'actual': actual, 'expected': expected})


def compare_fields(name, actual, expected):
    for key, value in expected.items():
        if isinstance(value, dict):
            check(name + ':' + key + ':dict', isinstance(actual.get(key), dict))
            compare_fields(name + ':' + key, actual.get(key, {}), value)
        else:
            equal(name + ':' + key, actual.get(key), value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate key')
        result[key] = value
    return result


def parse_raw(raw, row):
    import jsonschema
    if raw['status'] != 'RETURNED':
        return None, False
    try:
        value = json.loads(raw['raw_output'], object_pairs_hook=unique_object,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite')))
        jsonschema.Draft7Validator(row['json_schema']).validate(value)
        allowed = set(row['display']['items'])
        if len(value['evidence']) != 6 or len({x['label'] for x in value['evidence']}) != 6 or {x['label'] for x in value['evidence']} != allowed:
            raise ValueError('incomplete evidence labels')
        if len(value['answer']) != 6 or len(set(value['answer'])) != 6 or set(value['answer']) != allowed:
            raise ValueError('incomplete answer')
    except Exception:
        return None, False
    return value, raw.get('metadata', {}).get('finish_reason') in ('stop', 'eos')


def citation(value, source, mapping, gold):
    entries = []
    if value is not None:
        by_ref = {turn['turn_ref']: (i, turn['text']) for i, turn in enumerate(source['turns'])}
        for item in value['evidence']:
            found = by_ref.get(item['turn_ref'])
            offsets = []
            if found and item['quote']:
                offsets = [i for i in range(len(found[1])) if found[1].startswith(item['quote'], i)]
            entries.append({'display_label': item['label'], 'content_id': mapping[item['label']],
                            'declared_status': item['status'], 'turn_ref': item['turn_ref'], 'quote': item['quote'],
                            'quote_exact_in_cited_turn': bool(offsets), 'exact_occurrence_count': len(offsets),
                            'source_turn_index': found[0] if found else None, 'quote_start_offsets': offsets,
                            'position': [found[0], offsets[0]] if found and len(offsets) == 1 else None,
                            'semantic_support': 'NOT_REVIEWED'})
    known = len(entries) == 6 and all(x['position'] is not None for x in entries) and len({tuple(x['position']) for x in entries}) == 6
    ordered = sorted(entries, key=lambda x: tuple(x['position'])) if known else []
    display = [x['display_label'] for x in ordered] if known else None
    contents = [x['content_id'] for x in ordered] if known else None
    return {'computable': known, 'entries': entries,
            'reason': None if known else ('NO_STRICT_SCHEMA_OBJECT' if value is None else 'NONEXACT_NONUNIQUE_OR_TIED_POSITIONS'),
            'citation_sorted_display': display, 'citation_sorted_content': contents,
            'answer_consistent_with_own_citation_positions': value['answer'] == display if known else None,
            'citation_position_order_matches_gold': contents == gold if known else None, 'semantic_support': 'NOT_REVIEWED'}


def independently_score(row, raw, variant, canonical, gold_labels, source):
    mapping = variant['display_to_content']
    gold = [canonical[x] for x in gold_labels]
    value, valid = parse_raw(raw, row)
    predicted = [mapping[x] for x in value['answer']] if value else None
    exact = bool(valid and predicted == gold)
    correct_pairs = sum(predicted.index(gold[i]) < predicted.index(gold[j]) for i in range(6) for j in range(i + 1, 6)) if valid else 0
    own = citation(value, source, mapping, gold)
    derived = own['citation_position_order_matches_gold'] if valid and own['computable'] else None
    return {**{k: row[k] for k in ('request_id', 'candidate_id', 'root_id', 'probe_arm', 'cohort', 'pair_id', 'variant_id', 'seed')},
            'status': raw['status'], 'valid': valid, 'exact': exact,
            'prediction_display': value['answer'] if value else None, 'prediction_content': predicted, 'gold_content': gold,
            'correct_pairs': correct_pairs, 'total_pairs': 15, 'pair_accuracy': correct_pairs / 15,
            'citation_observation': own,
            'external_citation_sort': {'status': 'KNOWN' if derived is not None else 'UNKNOWN', 'exact': derived,
                                       'rescued_wrong_to_right': bool(derived is True and not exact),
                                       'changed_exact_to_wrong': bool(derived is False and exact)},
            'format_finish_reason': raw.get('metadata', {}).get('finish_reason'),
            'response_committed_at_utc': raw['response_committed_at_utc']}


def aggregate(records):
    n = len(records)
    good = [r for r in records if r['valid'] and r['citation_observation']['computable']]
    exact = sum(r['exact'] for r in records)
    pairs = sum(r['correct_pairs'] for r in records)
    valid = sum(r['valid'] for r in records)
    return {'calls': n, 'questions': len({r['candidate_id'] for r in records}), 'roots': len({r['root_id'] for r in records}),
            'valid_calls': valid, 'invalid_calls': n - valid, 'exact_calls': exact, 'exact_rate': exact / n,
            'correct_pairs': pairs, 'total_pairs': 15 * n, 'pair_accuracy': pairs / (15 * n),
            'citation_positions_computable_valid_calls': len(good),
            'answer_consistent_with_own_citation_positions': sum(r['citation_observation']['answer_consistent_with_own_citation_positions'] for r in good),
            'citation_consistency_fixed_denominator': n, 'citation_consistency_unknown_calls': n - len(good),
            'citation_position_order_matches_gold': sum(r['citation_observation']['citation_position_order_matches_gold'] for r in good),
            'external_citation_sort': {'fixed_denominator': n,
                                       'known_calls': sum(r['external_citation_sort']['exact'] is not None for r in records),
                                       'unknown_calls': sum(r['external_citation_sort']['exact'] is None for r in records),
                                       'exact_calls': sum(r['external_citation_sort']['exact'] is True for r in records),
                                       'rescued_wrong_to_right': sum(r['external_citation_sort']['rescued_wrong_to_right'] for r in records),
                                       'changed_exact_to_wrong': sum(r['external_citation_sort']['changed_exact_to_wrong'] for r in records)},
            'exact_quotes': sum(e['quote_exact_in_cited_turn'] for r in records for e in r['citation_observation']['entries']),
            'expected_citations': 6 * n}


def audit(run):
    run = run.resolve()
    manifest_path, freeze_path = run / 'manifest.json', run / 'RAW_RESPONSE_FREEZE.json'
    m, freeze = read(manifest_path), read(freeze_path)
    need('frozen_before_any_gold_parse', freeze['status'] == 'COMMITTED')
    bindings(freeze['files'] + [freeze['manifest_binding']], 'raw_freeze')
    rows = m['requests']
    need('108_requests', len(rows) == m['request_count'] == 108)
    ids = {row['request_id'] for row in rows}
    need('108_unique_ids', len(ids) == 108)
    expected_paths = {run / 'responses' / (rid + '.json') for rid in ids}
    need('freeze_exact108_terminalrecords', len(freeze['files']) == 108 and {Path(b['path']).resolve() for b in freeze['files']} == expected_paths)
    need('response_directory_exact_denominator', set((run / 'responses').glob('*.json')) == expected_paths)
    raw = {}
    for row in rows:
        rid = row['request_id']
        value = read(run / 'responses' / (rid + '.json'))
        need('terminal_status:' + rid, value['status'] in ('RETURNED', 'BACKEND_ERROR', 'NOT_SUBMITTED', 'MISSING'))
        need('terminal_before_freeze:' + rid, time(value['response_committed_at_utc']) <= time(freeze['created_at_utc']))
        need('raw_ids:' + rid, all(value[k] == row[k] for k in ('request_id', 'root_id', 'candidate_id')))
        raw[rid] = value
    # Only after all terminal bytes and commitment times have been verified may
    # this independent audit parse the scores, mappings and private Gold.
    gold_first_read_at = datetime.now(timezone.utc).isoformat()
    bindings([m['private_answer_binding']], 'private_gold_bytes')
    private = {q['candidate_id']: q for q in read(m['private_answer_binding']['path'])['questions']}
    scores, scoring_started = read(run / 'SCORES.json'), read(run / 'SCORING_STARTED.json')
    check('score_firstgold_after_rawfreeze', time(scoring_started['gold_first_read_at_utc']) >= time(freeze['created_at_utc']))
    equal('score_goldtimestamp', scores['gold_first_read_at_utc'], scoring_started['gold_first_read_at_utc'])
    check('scoretime_after_goldread', time(scores['scored_at_utc']) >= time(scores['gold_read_completed_at_utc']) >= time(scores['gold_first_read_at_utc']))
    bindings([scoring_started['raw_response_freeze_binding'], scoring_started['private_answer_binding']], 'scoring_started_binding')
    for key in ('manifest_binding', 'raw_response_freeze_binding', 'scoring_code_binding', 'public_questions_binding', 'display_mappings_binding', 'private_answer_binding', 'protocol_binding'):
        bindings([scores[key]], 'score_binding:' + key)
    bindings(m['code_bindings'], 'frozen_code')
    bindings(m['model_bindings'], 'model')
    bindings(m['input_bindings'], 'inputs')
    prior_path = DAY2 / 'COMPLETE_MANIFEST.json'
    prior = read(prior_path)
    check('old_day2_sealed', prior['sealed'] is True and prior['status'] == 'SOLVER_DIAGNOSIS_MILESTONE_COMPLETE')
    bindings([{**b, 'path': str(DAY2 / b['path'])} for b in prior['files']], 'old_day2_snapshot')
    pre = read(run / 'PREFLIGHT_FREEZE.json')
    bindings(pre['files'], 'preflight')
    policy, policy_freeze = read(HERE / 'PROTOCOL.json'), read(HERE / 'POLICY_FREEZE.json')
    bindings(policy_freeze['files'] + [policy_freeze['prior_sealed_snapshot']], 'policy')
    start = read(run / 'STARTED.json')
    check('policy_before_live', time(policy_freeze['created_at_utc']) < time(start['started_at_utc']))
    bindings([start['manifest_binding'], start['independent_audit_binding']], 'live_start')
    check('live_gold_not_read', start['gold_opened'] is False and freeze['gold_opened_by_live'] is False)
    public = {q['candidate_id']: q for q in read(run / 'PUBLIC_QUESTIONS.json')['questions']}
    maps = read(run / 'private/DISPLAY_MAPPINGS.json')
    frozen_tokenids = read(run / 'PROMPT_TOKEN_IDS.json')
    check('tokenid_denominator', set(frozen_tokenids) == ids)
    check('mapping_denominator', set(maps['requests']) == ids)
    all_old = read(DAY2 / 'solver/evidence_v2/PUBLIC_QUESTIONS.json')['questions']
    old_ordering = [q for q in all_old if q['question_type'] == 'ordering']
    discovery = set(policy['selection']['discovery_root_ids'])
    other = sorted((q for q in old_ordering if q['root_id'] not in discovery), key=lambda q: hashlib.sha256(('2026101900|order-construction-replication|' + q['candidate_id']).encode()).hexdigest())[:6]
    selected = {q['candidate_id']: 'discovery' for q in old_ordering if q['root_id'] in discovery}
    selected.update({q['candidate_id']: 'replication' for q in other})
    check('selection_fixed_hash_rule', set(public) == set(selected) and len(public) == 9)
    sources = {}
    for qid, q in public.items():
        expected_canonical = {label: 'content_' + hashlib.sha256(text.strip().encode()).hexdigest()[:16] for label, text in q['items'].items()}
        equal('canonical_from_text:' + qid, maps['public_to_canonical'][qid], expected_canonical)
        check('private_public_equal:' + qid, {k: private[qid][k] for k in q} == q)
        source_path = run / 'sources' / (q['root_id'] + '.json')
        check('source_hash:' + qid, sha(source_path) == q['source_sha256'])
        source = read(source_path)
        check('source_root_unique_refs:' + qid, source['root_id'] == q['root_id'] and len({t['turn_ref'] for t in source['turns']}) == len(source['turns']))
        sources[q['root_id']] = source
        # Independently replay the frozen native rebinding algorithm, rather
        # than trusting mapping text alone or importing the generation code.
        labels = list(q['items'])
        seed_for = lambda variant: int.from_bytes(hashlib.sha256(f"2026101903|{qid}|{variant}".encode()).digest()[:8], 'big')
        contents = list(expected_canonical.values())
        random.Random(seed_for(-1)).shuffle(contents)
        used, expected_variants = set(), {}
        for index in range(6):
            rng = random.Random(seed_for(index))
            while True:
                order = contents[:]; rng.shuffle(order)
                if tuple(order) not in used:
                    break
            used.add(tuple(order))
            expected_variants[f'perm_{index:02d}'] = dict(zip(labels, order))
        for row in (r for r in rows if r['candidate_id'] == qid):
            equal('native_rebinding:' + row['request_id'], maps['requests'][row['request_id']]['display_to_content'], expected_variants[row['variant_id']])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(m['tokenizer'], local_files_only=True)
    parsed_scores = {r['request_id']: r for r in scores['requests']}
    need('score_exact108records', len(scores['requests']) == 108 and set(parsed_scores) == ids)
    records, known_usage = [], collections.Counter()
    per_arm_usage = {arm: collections.Counter() for arm in ARMS}
    grouped = collections.defaultdict(dict)
    for row in rows:
        rid, qid, arm = row['request_id'], row['candidate_id'], row['probe_arm']
        check('row_selection_cohort:' + rid, row['cohort'] == selected[qid])
        check('row_six_labels:' + rid, set(row['display']['items']) == {'I1', 'I2', 'I3', 'I4', 'I5', 'I6'})
        check('row_strict_uniqueItems:' + rid, row['json_schema']['properties']['answer']['uniqueItems'] is True)
        variant = maps['requests'][rid]
        expected_mapping = {label: 'content_' + hashlib.sha256(text.strip().encode()).hexdigest()[:16] for label, text in row['display']['items'].items()}
        equal('mapping_text:' + rid, variant['display_to_content'], expected_mapping)
        check('mapping_bijection:' + rid, len(set(expected_mapping.values())) == 6 and set(expected_mapping.values()) == set(maps['public_to_canonical'][qid].values()))
        seed = int.from_bytes(hashlib.sha256(('2026101905|' + row['pair_id']).encode()).digest()[:8], 'big') % (2**31 - 1)
        equal('matched_seed:' + rid, row['seed'], seed)
        prompt = (run / 'prompts' / (rid + '.txt')).read_text()
        check('promptsha:' + rid, hashlib.sha256(prompt.encode()).hexdigest() == row['prompt_sha256'])
        equal('prompt_encoding:' + rid, tokenizer.encode(prompt, add_special_tokens=False), frozen_tokenids[rid])
        check('full_source_in_prompt:' + rid, all(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" in prompt for t in sources[row['root_id']]['turns']))
        response = raw[rid]
        if response['status'] == 'RETURNED':
            equal('response_promptIDs:' + rid, response['prompt_token_ids'], frozen_tokenids[rid])
            equal('rawdecode:' + rid, tokenizer.decode(response['completion_token_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=False), response['raw_output'])
            md = response['metadata']
            check('raw_runtime_contract:' + rid, md['temperature'] == .8 and md['top_p'] == .9 and md['max_tokens'] == 1024 and md['enable_thinking'] is False and response['gold_read_by_solver'] is False)
            for token_key, id_key in (('prompt_token_count', 'prompt_token_ids'), ('completion_token_count', 'completion_token_ids')):
                equal('usage:' + rid + ':' + token_key, md[token_key], len(response[id_key]))
                known_usage[token_key] += len(response[id_key]); per_arm_usage[arm][token_key] += len(response[id_key])
            check('output_budget:' + rid, len(response['completion_token_ids']) <= 1024)
            check('response_before_freeze_after_submit:' + rid, time(response['request_submitted_at_utc']) <= time(response['response_committed_at_utc']) <= time(freeze['created_at_utc']))
            for key in ('seed', 'variant_id', 'prompt_sha256'):
                equal('raw_row:' + rid + ':' + key, response[key], row[key])
            equal('raw_model:' + rid, response['model_path'], m['model'])
        record = independently_score(row, response, variant, maps['public_to_canonical'][qid], private[qid]['gold'], sources[row['root_id']])
        compare_fields('score:' + rid, parsed_scores[rid], record)
        check('parse_error_matches_validity:' + rid, (parsed_scores[rid]['parse_error'] is None) == record['valid'])
        for key, path in (('response_binding', run / 'responses' / (rid + '.json')), ('source_binding', run / 'sources' / (row['root_id'] + '.json')), ('display_mappings_binding', run / 'private/DISPLAY_MAPPINGS.json')):
            equal('record_binding:' + rid + ':' + key, parsed_scores[rid][key], bind(path))
        grouped[row['pair_id']][arm] = (row, record, prompt)
        records.append(record)
    check('54matchedpairs', len(grouped) == 54 and all(set(v) == set(ARMS) for v in grouped.values()))
    pairs, transitions = [], collections.Counter()
    marker = '<|im_end|>\n<|im_start|>assistant'
    for pair_id, values in sorted(grouped.items()):
        cr, control, cp = values['control']; tr, treatment, tp = values['treatment']
        for key in ('candidate_id', 'root_id', 'cohort', 'variant_id', 'seed', 'display', 'json_schema', 'max_output_tokens'):
            equal('pair_design:' + pair_id + ':' + key, cr[key], tr[key])
        point = cp.index(marker)
        equal('only_intervention_diff:' + pair_id, tp, cp[:point] + policy['intervention_text'] + cp[point:])
        transition = ('correct' if control['exact'] else 'wrong') + '_to_' + ('correct' if treatment['exact'] else 'wrong')
        transitions[transition] += 1
        pairs.append({**{k: cr[k] for k in ('pair_id', 'candidate_id', 'root_id', 'cohort', 'variant_id', 'seed')},
                      'control_request_id': cr['request_id'], 'treatment_request_id': tr['request_id'],
                      'control_valid': control['valid'], 'treatment_valid': treatment['valid'],
                      'control_exact': control['exact'], 'treatment_exact': treatment['exact'],
                      'exact_net_change': int(treatment['exact']) - int(control['exact']),
                      'correct_pair_net_change': treatment['correct_pairs'] - control['correct_pairs'], 'transition': transition})
    expected_contrasts = {'pairs': 54, 'transitions': {k: transitions[k] for k in ('correct_to_correct', 'correct_to_wrong', 'wrong_to_correct', 'wrong_to_wrong')}, 'records': pairs}
    compare_fields('matchedpairs', scores['matched_pairs'], expected_contrasts)
    arms = {arm: aggregate([r for r in records if r['probe_arm'] == arm]) for arm in ARMS}
    cohorts = {cohort: {arm: aggregate([r for r in records if r['probe_arm'] == arm and r['cohort'] == cohort]) for arm in ARMS} for cohort in COHORTS}
    compare_fields('arm_aggregate', scores['by_arm'], arms)
    compare_fields('cohort_aggregate', scores['by_cohort'], cohorts)
    roots = []
    for root in sorted({r['root_id'] for r in records}):
        subset = [r for r in records if r['root_id'] == root]
        summaries = {arm: aggregate([r for r in subset if r['probe_arm'] == arm]) for arm in ARMS}
        check('root_fixed6_perarm:' + root, all(s['calls'] == 6 for s in summaries.values()))
        roots.append({'root_id': root, 'cohort': subset[0]['cohort'], **summaries,
                      'exact_net_change': summaries['treatment']['exact_calls'] - summaries['control']['exact_calls'],
                      'correct_pair_net_change': summaries['treatment']['correct_pairs'] - summaries['control']['correct_pairs']})
    equal('root_aggregate', scores['by_root'], roots)
    rep = cohorts['replication']
    positive = [r['root_id'] for r in roots if r['cohort'] == 'replication' and r['exact_net_change'] > 0]
    conditions = {'at_least_three_positive_replication_roots': len(positive) >= 3,
                  'positive_replication_aggregate_exact_net_change': rep['treatment']['exact_calls'] > rep['control']['exact_calls'],
                  'replication_pair_correct_count_not_decreased': rep['treatment']['correct_pairs'] >= rep['control']['correct_pairs'],
                  'replication_citation_answer_consistent_count_net_positive': rep['treatment']['answer_consistent_with_own_citation_positions'] > rep['control']['answer_consistent_with_own_citation_positions'],
                  'both_arms_all_strict_valid': all(s['valid_calls'] == s['calls'] for s in arms.values())}
    equal('five_predeclared_gate_conditions', scores['primary_endpoint']['conditions'], conditions)
    equal('gate_met', scores['primary_endpoint']['met'], all(conditions.values()))
    equal('positive_reproots', scores['primary_endpoint']['positive_replication_roots'], positive)
    root_difference = sum(r['treatment']['exact_rate'] - r['control']['exact_rate'] for r in roots if r['cohort'] == 'replication') / 6
    equal('primary_root_equal_exact_difference', scores['primary_replication_root_equal_exact_difference'], root_difference)
    check('no_mechanism_or_training_claim', scores['cause_auto_published'] is False and scores['weakness_memory_upgraded'] is False and scores['training_started'] is False and scores['new_training_questions_generated'] == 0 and scores['memory_three_arm_executed_scenes'] == 0 and scores['use_as_final_eval'] is False)
    execution_path = run / ('COMPLETE.json' if (run / 'COMPLETE.json').exists() else 'FAILED.json')
    execution = read(execution_path)
    submitted_batches, batch_wall = [], 0.
    for path in sorted((run / 'batches').glob('*_SUBMITTED.json')):
        batch = read(path); submitted_batches.extend(batch['request_ids'])
        check('no_auto_retry:' + path.name, batch['automatic_retry'] is False)
        completion = Path(str(path).replace('_SUBMITTED.json', '_COMPLETE.json'))
        if completion.exists():
            done = read(completion); batch_wall += done['wall_seconds']
            equal('batch_ids:' + path.name, batch['request_ids'], done['request_ids'])
            check('batch_time:' + path.name, time(done['completed_at_utc']) >= time(batch['submitted_at_utc']))
        for parameter in batch['sampling_params']:
            if isinstance(parameter, str):
                check('backend_json_compat:' + path.name, 'choice=None' in parameter and 'uniqueItems' not in parameter and "json={'type': 'object'" in parameter)
    check('no_duplicate_submissions', len(submitted_batches) == len(set(submitted_batches)))
    returned = [r for r in raw.values() if r['status'] == 'RETURNED']
    statuses = collections.Counter(r['status'] for r in raw.values())
    known_prompt, known_completion = known_usage['prompt_token_count'], known_usage['completion_token_count']
    cost_expected = {'planned_calls': 108, 'returned_calls': len(returned), 'submitted_attempts': len(submitted_batches),
                     'not_submitted_calls': statuses['NOT_SUBMITTED'], 'missing_terminal_calls': statuses['MISSING'],
                     'requests_with_unknown_usage': statuses['BACKEND_ERROR'], 'known_prompt_tokens': known_prompt,
                     'known_completion_tokens': known_completion, 'known_total_tokens': known_prompt + known_completion,
                     'live_wall_seconds': execution['wall_seconds'], 'model_load_wall_seconds': execution['model_load_wall_seconds']}
    compare_fields('cost', scores['cost'], cost_expected)
    for arm in ARMS:
        compare_fields('cost:' + arm, scores['cost']['by_arm'][arm], {
            'known_prompt_tokens': per_arm_usage[arm]['prompt_token_count'], 'known_completion_tokens': per_arm_usage[arm]['completion_token_count'],
            'planned_calls': 54, 'returned_calls': sum(raw[r['request_id']]['status'] == 'RETURNED' for r in rows if r['probe_arm'] == arm)})
    check('live_wall_covers_batches_plus_load', execution['wall_seconds'] + 1e-6 >= batch_wall + (execution['model_load_wall_seconds'] or 0))
    equal('execution_cost_binding', scores['cost']['execution_binding'], bind(execution_path))
    equal('total_valid', scores['valid_calls'], sum(r['valid'] for r in records))
    equal('total_exact', scores['exact_calls'], sum(r['exact'] for r in records))
    for key, val in (('denominator_calls', 108), ('denominator_questions', 9), ('denominator_roots', 9)):
        equal(key, scores[key], val)
    return {'gold_first_read_by_audit_at_utc': gold_first_read_at, 'raw_status_counts': dict(statuses),
            'by_arm': arms, 'by_cohort': cohorts, 'by_root': roots, 'matched_pairs': expected_contrasts,
            'primary_replication_root_equal_exact_difference': root_difference,
            'continue_gate': {'conditions': conditions, 'met': all(conditions.values()), 'mechanism_confirmed': False},
            'cost': {**cost_expected, 'sum_completed_batch_wall_seconds': batch_wall},
            'bindings': [bind(manifest_path), bind(freeze_path), bind(run / 'SCORES.json'), bind(prior_path), bind(__file__)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=HERE / 'run')
    parser.add_argument('--out', type=Path, default=HERE / 'audit/RESULTS_AUDIT.json')
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError('NO_OVERWRITE:' + str(args.out))
    result = audit(args.run)
    output = {'status': 'PASS' if not ISSUES else 'ISSUES_FOUND',
              'audit_scope': 'Independent strict raw scoring, fixed denominators, literal citation positions, five predeclared gate conditions, provenance and actual cost; semantic support and internal mechanism are not certified.',
              'created_at_utc': datetime.now(timezone.utc).isoformat(), 'checks_count': len(CHECKS),
              'issues_count': len(ISSUES), 'issues': ISSUES, **result, 'checks': CHECKS}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(json.dumps({key: output[key] for key in ('status', 'checks_count', 'issues_count')}, ensure_ascii=False))
    if ISSUES:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
