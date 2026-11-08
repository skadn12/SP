"""Independent stdlib replay of v89 typed exports; never calls a model or writes.

This module does not import the export writer, production engines, or the four
shard auditors. The saved PASS audit is required, then raw responses and final
pool projections are checked again using a separate implementation.
"""
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re


def require(ok, message):
    if not ok:
        raise ValueError('export_audit:' + message)


def read(path):
    def pairs(rows):
        value = {}
        for key, item in rows:
            require(key not in value, 'duplicate_json_key:' + key)
            value[key] = item
        return value
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda value: require(False, 'nonfinite_json:' + value))


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def check_binding(value):
    require(isinstance(value, dict) and set(value) == {'path', 'sha256'}, 'binding_schema')
    require(Path(value['path']).is_absolute() and bind(value['path']) == value, 'binding_changed:' + value['path'])


def statement_key(question):
    kind = question['question_type']
    require(kind in ('ordering', 'multi_select'), 'unknown_type')
    field = 'items' if kind == 'ordering' else 'options'
    labels = ['I' + str(n) for n in range(1, 7)] if kind == 'ordering' else list('ABCDEF')
    require(isinstance(question[field], dict) and set(question[field]) == set(labels), 'six_statement_schema')
    require(all(isinstance(t, str) and t.strip() for t in question[field].values()), 'empty_statement')
    texts = tuple(sorted(' '.join(t.split()).casefold() for t in question[field].values()))
    require(len(set(texts)) == 6, 'duplicate_normalized_statement')
    require(isinstance(question['gold'], list) and len(set(question['gold'])) == len(question['gold'])
            and set(question['gold']) <= set(labels), 'gold_schema')
    require((2 <= len(question['gold']) <= 6) if kind == 'multi_select' else set(question['gold']) == set(labels), 'gold_cardinality')
    return question['root_id'], kind, texts


def is_control(q):
    return q['question_type'] == 'multi_select' and len(q['gold']) == 6


def raw_score(question, request, response):
    field = 'items' if question['question_type'] == 'ordering' else 'options'
    identities = {label: 'content_' + hashlib.sha256(text.strip().encode()).hexdigest()[:16]
                  for label, text in question[field].items()}
    correct = [identities[label] for label in question['gold']]
    variant = request['variant']; mapping = variant['display_to_content']
    require(set(mapping) == set(question[field]) and set(mapping.values()) == set(identities.values()), 'variant_mapping')
    require(variant[field] == {label: next(question[field][original] for original, cid in identities.items() if cid == content)
                             for label, content in mapping.items()}, 'display_statement_mapping')
    raw = response['raw_output']
    match = re.fullmatch(r'\s*<box>\s*(.*?)\s*</box>\s*', raw, re.S) if isinstance(raw, str) else None
    require(match is not None, 'raw_box_syntax')
    display = [] if not match.group(1).strip() else [item.strip().upper() for item in match.group(1).strip().split(',')]
    require(len(display) == len(set(display)) and set(display) <= set(mapping), 'prediction_labels')
    predicted = [mapping[label] for label in display]
    if question['question_type'] == 'multi_select':
        tp = len(set(predicted) & set(correct)); fp = len(set(predicted) - set(correct))
        score = Fraction(tp, len(correct)) - (Fraction(fp, 6 - len(correct)) if len(correct) < 6 else 0)
        exact = set(predicted) == set(correct)
    else:
        require(len(predicted) == 6 and set(predicted) == set(correct), 'incomplete_order')
        position = {cid: i for i, cid in enumerate(predicted)}
        reversed_pairs = sum(position[a] > position[b] for i, a in enumerate(correct) for b in correct[i + 1:])
        score = Fraction(15 - 2 * reversed_pairs, 15); exact = predicted == correct
    require(response['score']['partial_score_fraction'] == str(score)
            and response['score']['exact_match'] is exact, 'saved_raw_score')
    return score, exact


def shard_replay(shard, audit_binding):
    shard = Path(shard).resolve(); check_binding(audit_binding); audit = read(audit_binding['path'])
    require(Path(audit_binding['path']).parent == shard / 'evidence', 'audit_outside_shard')
    require(audit.get('status') == 'PASS' and audit.get('phase') == 'paired_results'
            and audit.get('home') == str(shard) and audit.get('failed_checks') == [], 'result_audit_pass_required')
    require(audit.get('checks') and all(c.get('status') == 'PASS' for c in audit['checks']), 'result_check_failure')
    auditor_home = Path(__file__).resolve().parent
    auditor_names = ('audit_shard.py', 'audit_contract_v89.py', 'audit_results_core_v89.py', 'audit_generation_core_v89.py')
    require(audit['auditor_files'] == [bind(auditor_home / name) for name in auditor_names], 'four_frozen_shard_auditor_bindings')
    for value in audit['files'] + audit['auditor_files'] + [audit['passing_input_audit']]:
        check_binding(value)
    input_audit = read(audit['passing_input_audit']['path'])
    require(input_audit.get('status') == 'PASS' and input_audit.get('phase') == 'inputs_only'
            and input_audit.get('auditor_files') == audit['auditor_files'], 'matching_passing_input_audit')
    questions = read(shard / 'EVALUATION_POOL.json')['questions']
    by_id = {q['candidate_id']: q for q in questions}
    require(len(by_id) == len(questions), 'duplicate_pool_id')
    controls = sorted(q['candidate_id'] for q in questions if is_control(q))
    require(controls == audit['controls'], 'historical_controls_exact')
    for q in questions:
        statement_key(q)
    selected = {}; side_summaries = {}
    for side in ('trained', 'base'):
        run = shard / 'solver' / side; manifest = read(run / 'manifest.json')
        plans = manifest['requests']; ids = [r['request_id'] for r in plans]
        require(len(ids) == len(set(ids)), 'duplicate_request')
        actual = {p.stem for p in (run / 'responses').glob('*.json')}
        require(actual == set(ids), 'response_inventory')
        groups = {qid: [] for qid in by_id}
        for plan in plans:
            response = read(run / 'responses' / (plan['request_id'] + '.json'))
            require(plan['candidate_id'] in by_id, 'unknown_request_candidate')
            require(all(response[k] == plan[k] for k in ('request_id', 'candidate_id', 'root_id', 'question_type', 'slot', 'seed'))
                    and response['variant_id'] == plan['variant']['variant_id'], 'response_identity')
            value, exact = raw_score(by_id[plan['candidate_id']], plan, response)
            groups[plan['candidate_id']].append((plan['variant']['variant_id'], plan['slot'], value, exact))
        summaries = []; chosen = []
        for q in questions:
            rows = groups[q['candidate_id']]; permutations = 12 if q['question_type'] == 'ordering' else 6
            slots = 1 if permutations == 12 else 3
            expected = {(f'perm_{i:02d}', j) for i in range(permutations) for j in range(slots)}
            require(len(rows) == len(expected) and {(r[0], r[1]) for r in rows} == expected, 'permutation_slot_coverage')
            mean = sum((r[2] for r in rows), Fraction()) / len(rows)
            exact_rate = Fraction(sum(r[3] for r in rows), len(rows)); admission = 0 < mean < 1
            summary = next((r for r in audit['models'][side]['per_question'] if r['candidate_id'] == q['candidate_id']), None)
            require(summary is not None and summary['full_partial_mean'] == str(mean)
                    and summary['full_exact_rate'] == str(exact_rate) and summary['selected_by_policy'] is admission,
                    'audited_question_summary_replay')
            summaries.append({'candidate_id': q['candidate_id'], 'full_partial_mean': str(mean),
                              'full_exact_rate': str(exact_rate), 'selected_by_policy': admission})
            if admission:
                chosen.append(q['candidate_id'])
        require(chosen == audit['models'][side]['selected_ids'], 'audited_selection_replay')
        selected[side] = chosen; side_summaries[side] = summaries
    output = [q for q in questions if q['candidate_id'] in selected['trained'] and not is_control(q)]
    require([q['candidate_id'] for q in output] == audit['selected_new_ids'], 'selected_new_ids_disagree')
    require(len(output) == audit['selected_new_count'], 'selected_new_count')
    require(read(shard / 'SELECTED_CANDIDATES.json')['questions'] ==
            [q for q in questions if q['candidate_id'] in selected['trained']], 'selected_pool_projection')
    return {'questions': output, 'side_summaries': side_summaries, 'controls': controls,
            'all_selected_new_ids': [q['candidate_id'] for q in output]}


def counts(questions):
    typed = {kind: sum(q['question_type'] == kind for q in questions) for kind in ('multi_select', 'ordering')}
    gold = {str(g): sum(q['question_type'] == 'multi_select' and len(q['gold']) == g for q in questions) for g in range(2, 6)}
    return typed, gold, dict(sorted(Counter(q['root_id'] for q in questions).items()))


def verify_pool(pool, questions):
    require(pool['questions'] == questions, 'exact_question_projection')
    typed, gold, roots = counts(questions)
    require(pool['candidate_count'] == len(questions) and pool['by_type'] == pool['type_distribution'] == typed
            and pool['gold_distribution'] == pool['multiselect_gold_distribution'] == gold
            and pool['root_distribution'] == roots and pool['source_root_count'] == len(roots), 'pool_counts')
    require(all(pool.get(k) is False for k in ('training_allowed', 'training_started', 'formal_training_admission', 'merged_into_baseline')), 'training_forbidden')
    require(not any(is_control(q) for q in questions), 'multiselect_control_leaked')


def audit_exports(home, baseline, shards, new_pool=None, cumulative_pool=None):
    """`shards` is an ordered list of {home, result_audit} explicit bindings."""
    home = Path(home).resolve(); baseline = Path(baseline).resolve()
    new_pool = read(home / 'AUDITED_NEW_CANDIDATES.json') if new_pool is None else new_pool
    cumulative_pool = read(home / 'CUMULATIVE_AUDITED_NEW_CANDIDATES.json') if cumulative_pool is None else cumulative_pool
    raw_new = []; replays = []
    for entry in shards:
        replay = shard_replay(entry['home'], entry['result_audit']); raw_new.extend(replay['questions'])
        replays.append({'home': entry['home'], 'result_audit': entry['result_audit'],
                        'selected_new_ids': replay['all_selected_new_ids'], 'historical_controls_excluded': replay['controls'],
                        'side_summaries': replay['side_summaries']})
    require(len({q['candidate_id'] for q in raw_new}) == len(raw_new), 'cross_shard_candidate_id_collision')
    unique = []; seen = {}; removed = []
    for q in raw_new:
        key = statement_key(q)
        if key in seen:
            removed.append({'candidate_id': q['candidate_id'], 'retained_candidate_id': seen[key], 'reason': 'same_root_type_all_six_normalized_statements'})
        else:
            unique.append(q); seen[key] = q['candidate_id']
    verify_pool(new_pool, unique)
    require(new_pool['selected_new_ids_before_deduplication'] == [q['candidate_id'] for q in raw_new]
            and new_pool['duplicates_removed'] == removed and new_pool['shards'] == shards, 'new_export_provenance')
    prior = read(baseline); previous = prior['questions']
    require(len(previous) == prior['candidate_count'] and not any(is_control(q) for q in previous), 'baseline_count_or_control')
    union = []; union_seen = {}; union_ids = {}; union_removed = []
    for q in previous + unique:
        qid = q['candidate_id']; key = statement_key(q)
        require(qid not in union_ids or union_ids[qid] == q, 'same_id_different_question')
        union_ids[qid] = q
        if key in union_seen:
            union_removed.append({'candidate_id': qid, 'retained_candidate_id': union_seen[key], 'reason': 'same_root_type_all_six_normalized_statements'})
        else:
            union.append(q); union_seen[key] = qid
    verify_pool(cumulative_pool, union)
    require(cumulative_pool['baseline'] == bind(baseline) and cumulative_pool['duplicates_removed'] == union_removed,
            'cumulative_provenance')
    return {'status': 'PASS', 'independent_raw_solver_replay': True, 'production_scoring_imported': False,
            'new_count': len(unique), 'cumulative_count': len(union), 'new_distribution': counts(unique),
            'cumulative_distribution': counts(union), 'shards': replays, 'baseline': bind(baseline),
            'source_semantics_recertified': False, 'new_model_calls': 0, 'training_performed': False}
