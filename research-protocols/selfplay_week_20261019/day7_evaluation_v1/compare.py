"""Independently recompute and compare three fixed TEST endpoint traces on CPU.

This module uses only the standard library. It never imports the model runtime
or native scorer, loads weights, sends requests, or releases item outcomes to
Challenger. All raw packets must already be frozen before private Gold is read.
"""
from __future__ import annotations

import collections
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import re
import statistics


HERE = Path(__file__).resolve().parent
ROLES = ('S0', 'A1', 'B1')
EXPECTED_GROUPS = {
    ('historical115_TEST', 'HISTORICAL_REGRESSION_PRIMARY'): (64, 1032),
    ('historical115_TEST', 'GOLD6_CONTROL_ONLY'): (11, 198),
    ('previous2MC_TEST', 'SEPARATE_PREVIOUS_MC_SUPPLEMENT'): (2, 36),
    ('supplement12_TEST', None): (12, 180),
}
STRICT_BOX = re.compile(r'<box>([A-Z][A-Z0-9_]*(?:,[A-Z][A-Z0-9_]*)*)</box>\Z')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(row) for row in Path(path).read_text().splitlines() if row]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def verify_bindings(rows):
    for row in rows:
        require(binding(row['path']) == {key: row[key] for key in ('path', 'sha256', 'bytes')},
                'BOUND_FILE_CHANGED:' + row['path'])


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def unique_by(rows, key):
    result = {row[key]: row for row in rows}
    require(len(result) == len(rows), 'DUPLICATE_' + key)
    return result


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


def root_uncertainty(root_values):
    """Descriptive root resampling only; one/two roots get no interval."""
    values = list(root_values)
    result = {'root_count': len(values), 'root_equal_mean': mean(values),
              'root_mean_min': min(values), 'root_mean_max': max(values),
              'scope': 'Descriptive across historically exposed roots; no causal or unseen-root inference'}
    if len(values) < 3:
        result.update(status='INTERVAL_NOT_REPORTED_TOO_FEW_ROOTS', lower_95=None, upper_95=None)
    else:
        rng = random.Random(20261022)
        means = sorted(mean(rng.choices(values, k=len(values))) for _ in range(2000))
        result.update(status='DESCRIPTIVE_ROOT_BOOTSTRAP', method='percentile_root_cluster_bootstrap',
                      replicates=2000, seed=20261022, lower_95=means[49], upper_95=means[1949],
                      estimand='root_equal_mean',
                      repeated_calls_independent=False)
    return result


def check_number(actual, expected, message):
    # The two paths use exact Fraction arithmetic before converting to float;
    # aggregate means can differ by the last floating point rounding bit.
    require(isinstance(actual, (float, int)) and abs(actual - expected) <= 1e-12, message)


def independent_score(raw, request, gold, variant, choices, role, model_directory):
    """Strict frozen output validity, bijective remapping, and independent math."""
    require(raw['request_id'] == request['request_id'], 'RAW_REQUEST_ID_DRIFT')
    require(raw['status'] in ('RETURNED', 'BACKEND_ERROR', 'NOT_SUBMITTED'), 'UNKNOWN_RAW_STATUS')
    mapping = variant['display_to_content']
    require(len(mapping) == len(set(mapping.values())), 'DISPLAY_MAP_NOT_BIJECTIVE')
    g = gold['gold_content_ids']
    require(g and len(g) == len(set(g)) and set(g) <= set(mapping.values()), 'INVALID_GOLD_CONTENT_IDS')
    if raw['status'] == 'RETURNED':
        require(raw['seed'] == request['sampling']['seed']
                and raw['metadata']['sampling'] == request['sampling']
                and raw['prompt_sha256'] == request['prompt_sha256']
                and raw['model_role'] == role
                and raw['model_directory'] == model_directory
                and raw['metadata']['prompt_token_count'] == request['prompt_tokens'],
                'RETURNED_RAW_REQUEST_CONTRACT_DRIFT')
    output = raw.get('raw_output')
    match = STRICT_BOX.fullmatch(output) if isinstance(output, str) else None
    labels = match.group(1).split(',') if match else None
    syntax_ok = (labels is not None and len(labels) == len(set(labels))
                 and all(label in mapping for label in labels))
    valid = (raw['status'] == 'RETURNED'
             and raw.get('metadata', {}).get('finish_reason') in ('stop', 'eos')
             and syntax_ok and output in choices)
    prediction = [mapping[label] for label in labels] if syntax_ok else None
    if request['question_type'] == 'ordering':
        valid = valid and len(prediction) == len(g) and set(prediction) == set(g)
    metrics = {}
    partial = Fraction(0)
    exact = False
    if valid:
        if request['question_type'] == 'multi_select':
            predset, goldset = set(prediction), set(g)
            tp, fp, fn = len(predset & goldset), len(predset - goldset), len(goldset - predset)
            partial = Fraction(tp, len(goldset))
            if len(goldset) != len(mapping):
                partial -= Fraction(fp, len(mapping) - len(goldset))
            exact = predset == goldset
            prediction = sorted(predset)
            metrics = {'tp': tp, 'fp': fp, 'fn': fn,
                       'jaccard': tp / len(predset | goldset),
                       'overselected': len(predset) > len(goldset),
                       'selected_count': len(predset), 'gold_count': len(goldset)}
        else:
            rank = {item: index for index, item in enumerate(prediction)}
            pairs = [(a, b) for index, a in enumerate(g) for b in g[index + 1:]]
            correct = sum(rank[a] < rank[b] for a, b in pairs)
            partial = Fraction(2 * correct - len(pairs), len(pairs))
            exact = prediction == g
            metrics = {'pair_accuracy': correct / len(pairs)}
    return {'valid': bool(valid), 'exact': bool(exact), 'partial_score': float(partial),
            'prediction_content_ids': prediction if valid else None,
            'metrics': metrics, 'raw_status': raw['status'],
            'finish_reason': raw.get('metadata', {}).get('finish_reason')}


def prediction_key(row):
    prediction = row['prediction_content_ids']
    return tuple(prediction) if row['valid'] else None


def stability(rows):
    """Separate invalids, stochastic repeat changes, and permutation changes."""
    valid = [row for row in rows if row['valid']]
    counts = collections.Counter(prediction_key(row) for row in valid)
    mode_count = max(counts.values()) if counts else 0
    by_variant = collections.defaultdict(list)
    by_sample = collections.defaultdict(list)
    for row in rows:
        by_variant[row['variant_id']].append(row)
        by_sample[row['sample_index']].append(row)
    repeated_pairs = permutation_pairs = 0
    repeated_valid_pairs = permutation_valid_pairs = 0
    repeated_agreements = permutation_agreements = 0
    for groups, repeat in ((by_variant.values(), True), (by_sample.values(), False)):
        for group in groups:
            for index, left in enumerate(group):
                for right in group[index + 1:]:
                    # Align equal sample indices across different display maps,
                    # and distinct repeated samples within one map.
                    if repeat:
                        repeated_pairs += 1
                    else:
                        permutation_pairs += 1
                    both_valid = left['valid'] and right['valid']
                    agreement = both_valid and prediction_key(left) == prediction_key(right)
                    if repeat:
                        repeated_valid_pairs += both_valid
                        repeated_agreements += agreement
                    else:
                        permutation_valid_pairs += both_valid
                        permutation_agreements += agreement
    return {
        'valid_content_modes': len(counts),
        'modal_prediction_calls': mode_count,
        'modal_prediction_fraction_all_planned_calls': mode_count / len(rows),
        'modal_prediction_fraction_valid_calls': mode_count / len(valid) if valid else None,
        'within_mapping_repeat_pairs_planned': repeated_pairs,
        'within_mapping_repeat_pairs_both_valid': repeated_valid_pairs,
        'within_mapping_repeat_prediction_agreements': repeated_agreements,
        'within_mapping_repeat_agreement_fixed_pairs': repeated_agreements / repeated_pairs if repeated_pairs else None,
        'across_mapping_same_sample_pairs_planned': permutation_pairs,
        'across_mapping_same_sample_pairs_both_valid': permutation_valid_pairs,
        'across_mapping_same_sample_prediction_agreements': permutation_agreements,
        'across_mapping_same_sample_agreement_fixed_pairs': permutation_agreements / permutation_pairs if permutation_pairs else None,
        'interpretation': 'Descriptive prediction agreement after content-ID remapping; pairs are dependent and invalid pairs contribute zero agreement. Different mappings may have different frozen seeds, so this is not a pure mapping effect.',
    }


def independently_recalculate(role, manifest, scored, rawfreeze, requests, gold, maps, choices):
    stored_requests = unique_by(scored['requests'], 'request_id')
    stored_questions = unique_by(scored['questions'], 'candidate_id')
    require(set(stored_requests) == {row['request_id'] for row in requests}, 'SCORED_REQUEST_IDS_DRIFT')
    require(set(stored_questions) == set(gold), 'SCORED_QUESTION_IDS_DRIFT')
    expected_paths = {str((HERE / 'models' / role / 'responses' / (row['request_id'] + '.json')).resolve()) for row in requests}
    require({row['path'] for row in rawfreeze['files']} == expected_paths
            and len(rawfreeze['files']) == len(requests), 'RAW_FREEZE_FILE_SET_DRIFT')
    byquestion = collections.defaultdict(list)
    independently_scored = {}
    for request in requests:
        rid, cid = request['request_id'], request['candidate_id']
        path = HERE / 'models' / role / 'responses' / (rid + '.json')
        raw = read(path)
        result = independent_score(raw, request, gold[cid], maps[request['prompt_id']],
                                   choices[request['constraint_key']], role,
                                   manifest['model_directory'])
        stored = stored_requests[rid]
        require(stored['valid'] == result['valid'] and stored['exact'] == result['exact'],
                'NATIVE_VALIDITY_OR_EXACT_DISAGREES:' + role + ':' + rid)
        check_number(stored['partial_score'], result['partial_score'], 'NATIVE_PARTIAL_DISAGREES:' + rid)
        verify_bindings([stored['response_binding']])
        require(stored['response_binding'] == binding(path), 'SCORED_RESPONSE_BINDING_DRIFT')
        for key in ('candidate_id', 'cohort_id', 'root_id', 'question_type', 'reporting_role', 'variant_id'):
            require(stored[key] == request[key], 'SCORED_REQUEST_METADATA_DRIFT:' + key)
        if result['valid']:
            native_prediction = stored['score']['content_prediction']
            require(native_prediction == result['prediction_content_ids'], 'NATIVE_CONTENT_REMAP_DISAGREES:' + rid)
            if 'prediction_content_ids' in stored:
                require(stored['prediction_content_ids'] == result['prediction_content_ids'], 'PREDICTION_ALIAS_DRIFT:' + rid)
            for key, value in result['metrics'].items():
                if key in stored['metrics']:
                    require(stored['metrics'][key] == value, 'NATIVE_AUXILIARY_METRIC_DISAGREES:' + rid + ':' + key)
        record = {**request, **result}
        independently_scored[rid] = record
        byquestion[cid].append(record)
    questions = {}
    for cid, rows in byquestion.items():
        q = stored_questions[cid]
        first = rows[0]
        exact_rate = mean(row['exact'] for row in rows)
        partial = mean(row['partial_score'] for row in rows)
        check_number(q['exact_rate'], exact_rate, 'NATIVE_QUESTION_EXACT_AGGREGATE_DISAGREES:' + cid)
        check_number(q['partial_score_mean'], partial, 'NATIVE_QUESTION_PARTIAL_AGGREGATE_DISAGREES:' + cid)
        require(q['planned_requests'] == len(rows)
                and q['valid_requests'] == sum(row['valid'] for row in rows)
                and q['any_correct'] == any(row['exact'] for row in rows)
                and q['all_correct'] == all(row['exact'] for row in rows), 'NATIVE_QUESTION_COUNTS_DISAGREE:' + cid)
        for key in ('cohort_id', 'root_id', 'question_type', 'reporting_role'):
            require(q[key] == first[key], 'NATIVE_QUESTION_METADATA_DRIFT:' + key)
        stable = stability(rows)
        if 'display_permutation_consistency' in q:
            check_number(q['display_permutation_consistency'], stable['modal_prediction_fraction_all_planned_calls'],
                         'NATIVE_QUESTION_CONSISTENCY_DISAGREES:' + cid)
        questions[cid] = {**{key: first[key] for key in ('candidate_id', 'cohort_id', 'root_id', 'question_type', 'reporting_role')},
                          'planned_requests': len(rows), 'valid_requests': sum(row['valid'] for row in rows),
                          'exact_rate': exact_rate, 'partial_score_mean': partial,
                          'any_correct': any(row['exact'] for row in rows),
                          'all_correct': all(row['exact'] for row in rows), 'stability': stable}
    return {'questions': questions, 'requests': independently_scored}


def group_ids(questions):
    """No pooled89-question primary score: four disjoint reporting groups."""
    groups = {}
    for (cohort, reporting), (expected_q, expected_r) in EXPECTED_GROUPS.items():
        ids = [cid for cid, q in questions.items()
               if q['cohort_id'] == cohort and (reporting is None or q['reporting_role'] == reporting)]
        require(len(ids) == expected_q
                and sum(questions[cid]['planned_requests'] for cid in ids) == expected_r,
                'REPORT_GROUP_DENOMINATOR_CHANGED:' + cohort + '/' + str(reporting))
        reporting_values = {questions[cid]['reporting_role'] for cid in ids}
        require(len(reporting_values) == 1, 'SUPPLEMENT_REPORTING_ROLE_NOT_UNIFORM')
        report_name = next(iter(reporting_values))
        for qtype in ('ALL', 'multi_select', 'ordering'):
            selected = [cid for cid in ids if qtype == 'ALL' or questions[cid]['question_type'] == qtype]
            if selected:
                groups[cohort + '/' + report_name + '/' + qtype] = selected
    require(sum(value[0] for value in EXPECTED_GROUPS.values()) == 89
            and sum(value[1] for value in EXPECTED_GROUPS.values()) == 1446, 'EXPECTED_TOTALS_CORRUPTED')
    covered = [cid for name, ids in groups.items() if name.endswith('/ALL') for cid in ids]
    require(len(covered) == len(set(covered)) == 89 and set(covered) == set(questions), 'REPORT_GROUPS_NOT_PARTITION')
    # Marginal views inherit the same cohort boundaries; absent Gold counts are
    # listed explicitly in group_availability rather than assigned fake scores.
    primary_groups = list(groups.items())
    for name, ids in primary_groups:
        if not name.endswith('/ALL'):
            continue
        prefix = name[:-len('/ALL')]
        for n in range(2, 6):
            selected = [cid for cid in ids if questions[cid]['question_type'] == 'multi_select'
                        and questions[cid]['gold_count'] == n]
            if selected:
                groups[prefix + '/multi_select_gold_' + str(n)] = selected
        selected = [cid for cid in ids if questions[cid]['question_type'] == 'multi_select'
                    and questions[cid]['gold_count'] == 6]
        if selected:
            groups[prefix + '/multi_select_gold_6_control'] = selected
        for n in (4, 5, 6):
            selected = [cid for cid in ids if questions[cid]['question_type'] == 'ordering'
                        and questions[cid]['arity'] == n]
            if selected:
                groups[prefix + '/ordering_arity_' + str(n)] = selected
        if ids and questions[ids[0]]['cohort_id'] == 'supplement12_TEST':
            for design in sorted({questions[cid]['planned_design'] for cid in ids}):
                selected = [cid for cid in ids if questions[cid]['planned_design'] == design]
                groups[prefix + '/planned_design_' + design] = selected
            for relation in sorted({questions[cid]['relation_class'] for cid in ids}):
                selected = [cid for cid in ids if questions[cid]['relation_class'] == relation]
                groups[prefix + '/relation_class_' + str(relation)] = selected
    return groups


def enrich_question_metadata(models, public, gold, frozen):
    admission_path = HERE.parent / 'day3_eval_supplement_v2/CONTENT_ADMISSION.json'
    require(any(row == binding(admission_path) for row in frozen['input_bindings']),
            'SUPPLEMENT_DESIGN_METADATA_NOT_IN_COMMON_FREEZE')
    admission = unique_by(read(admission_path)['records'], 'candidate_id')
    supplement_ids = {cid for cid, q in public.items() if q['cohort_id'] == 'supplement12_TEST'}
    require(set(admission) == supplement_ids, 'SUPPLEMENT_DESIGN_IDS_DIFFER_FROM_FROZEN_TEST')
    availability = {}
    for role, model in models.items():
        for cid, q in model['questions'].items():
            item = public[cid]
            require(all(q[key] == item[key] for key in ('candidate_id', 'root_id', 'cohort_id', 'question_type')),
                    'PUBLIC_QUESTION_IDENTITY_DRIFT')
            field = 'options' if q['question_type'] == 'multi_select' else 'items'
            q['arity'] = len(item[field])
            q['gold_count'] = len(gold[cid]['gold_content_ids'])
            q['planned_design'] = None
            q['relation_class'] = None
            if cid in supplement_ids:
                record = admission[cid]
                require(record['status'] == 'CONTENT_ACCEPTED'
                        and record['question_type'] == q['question_type']
                        and record['root_id'] == q['root_id']
                        and record['arity'] == q['arity'], 'SUPPLEMENT_CONTENT_METADATA_DRIFT')
                q['planned_design'] = record['planned_design']
                q['relation_class'] = record.get('relation_class')
        if role == 'S0':
            for cohort, reporting in sorted({(q['cohort_id'], q['reporting_role']) for q in model['questions'].values()}):
                qs = [q for q in model['questions'].values() if (q['cohort_id'], q['reporting_role']) == (cohort, reporting)]
                counts = {str(n): sum(q['question_type'] == 'multi_select' and q['gold_count'] == n for q in qs) for n in range(2, 6)}
                availability[cohort + '/' + reporting] = {
                    'mc_gold_count_2_to_5_question_counts': counts,
                    'gold6_control_question_count': sum(q['question_type'] == 'multi_select' and q['gold_count'] == 6 for q in qs),
                    'empty_marginals_have_no_score': True}
    return {'supplement_design_source': binding(admission_path),
            'classification_policy': 'Only frozen upstream planned_design/relation_class; no classification inferred from responses',
            'cohort_marginal_availability': availability,
            'within_across_scene_pairs': {'status': 'UNAVAILABLE',
                                         'reason': 'Minimal frozen Gold packet lacks per-item scene anchors. They are not reconstructed from model outcomes.'}}


def summarize_group(model, ids):
    qs = [model['questions'][cid] for cid in ids]
    idset = set(ids)
    rs = [row for row in model['requests'].values() if row['candidate_id'] in idset]
    roots = collections.defaultdict(list)
    for q in qs:
        roots[q['root_id']].append(q['exact_rate'])
    valid = [row for row in rs if row['valid']]
    output = {
        'questions': len(qs), 'roots': len(roots), 'planned_requests': len(rs),
        'returned_requests': sum(row['raw_status'] == 'RETURNED' for row in rs),
        'valid_requests': len(valid), 'invalid_requests': len(rs) - len(valid),
        'raw_status_counts': dict(collections.Counter(row['raw_status'] for row in rs)),
        'finish_reason_counts_returned': dict(collections.Counter(str(row['finish_reason']) for row in rs if row['raw_status'] == 'RETURNED')),
        'exact_requests': sum(row['exact'] for row in rs),
        'call_weighted_exact_fixed_denominator': mean(row['exact'] for row in rs),
        'question_equal_exact': mean(q['exact_rate'] for q in qs),
        'root_equal_exact': mean(mean(values) for values in roots.values()),
        'root_descriptive_uncertainty': root_uncertainty(mean(values) for values in roots.values()),
        'question_equal_partial': mean(q['partial_score_mean'] for q in qs),
        'call_weighted_partial_fixed_denominator': mean(row['partial_score'] for row in rs),
        'any_correct_questions': sum(q['any_correct'] for q in qs),
        'all_correct_questions': sum(q['all_correct'] for q in qs),
        'no_correct_questions': sum(not q['any_correct'] for q in qs),
        'all_correct_questions_fraction': mean(q['all_correct'] for q in qs),
        'any_correct_questions_fraction': mean(q['any_correct'] for q in qs),
        'valid_exact_requests': sum(row['exact'] for row in valid),
        'valid_partial_positive_requests': sum(row['partial_score'] > 0 for row in valid),
        'valid_partial_zero_requests': sum(row['partial_score'] == 0 for row in valid),
        'valid_partial_negative_requests': sum(row['partial_score'] < 0 for row in valid),
        'partial_credit_note': 'Partial score is a signed penalty-adjusted score, not the fraction of partially correct questions. Invalid calls contribute zero to the fixed planned denominator and are reported separately.',
        'question_equal_modal_prediction_fraction_all_calls': mean(q['stability']['modal_prediction_fraction_all_planned_calls'] for q in qs),
        'question_equal_modal_prediction_fraction_valid_calls': mean(q['stability']['modal_prediction_fraction_valid_calls'] for q in qs if q['stability']['modal_prediction_fraction_valid_calls'] is not None),
        'question_equal_within_mapping_repeat_agreement_fixed_pairs': mean(q['stability']['within_mapping_repeat_agreement_fixed_pairs'] for q in qs if q['stability']['within_mapping_repeat_agreement_fixed_pairs'] is not None),
        'question_equal_across_mapping_same_sample_agreement_fixed_pairs': mean(q['stability']['across_mapping_same_sample_agreement_fixed_pairs'] for q in qs if q['stability']['across_mapping_same_sample_agreement_fixed_pairs'] is not None),
    }
    for key in ('tp', 'fp', 'fn', 'jaccard', 'overselected', 'selected_count', 'gold_count', 'pair_accuracy'):
        vals = [row['metrics'][key] for row in valid if key in row['metrics']]
        output[key + '_valid_call_mean'] = mean(vals)
        output[key + '_valid_call_denominator'] = len(vals)
    # Do not merge invalid ordering traces into pairwise factual error rates.
    output['ordering_invalid_requests'] = sum(not row['valid'] for row in rs if row['question_type'] == 'ordering')
    output['within_across_scene_pairs'] = {'status': 'UNAVAILABLE', 'reason': 'Frozen Gold lacks per-item scene anchors; no outcome-derived reconstruction'}
    return output


def compare_group(left, right, ids):
    deltas = []
    partial_deltas = []
    root_deltas = collections.defaultdict(list)
    transitions = collections.Counter()
    content_agreement = both_valid = 0
    per_question = []
    for cid in ids:
        a, b = left['questions'][cid], right['questions'][cid]
        for key in ('candidate_id', 'cohort_id', 'root_id', 'question_type', 'reporting_role', 'planned_requests'):
            require(a[key] == b[key], 'PAIRED_QUESTION_METADATA_DRIFT:' + key)
        delta = a['exact_rate'] - b['exact_rate']
        partial_delta = a['partial_score_mean'] - b['partial_score_mean']
        deltas.append(delta)
        partial_deltas.append(partial_delta)
        root_deltas[a['root_id']].append(delta)
        per_question.append({'candidate_id': cid, 'root_id': a['root_id'],
                             'exact_rate_left': a['exact_rate'], 'exact_rate_right': b['exact_rate'],
                             'exact_rate_difference': delta, 'partial_difference': partial_delta})
    ids = set(ids)
    for rid, a in left['requests'].items():
        if a['candidate_id'] not in ids:
            continue
        b = right['requests'][rid]
        for key in ('candidate_id', 'cohort_id', 'root_id', 'question_type', 'reporting_role', 'variant_id', 'sample_index', 'prompt_id', 'prompt_sha256', 'sampling', 'constraint_key'):
            require(a[key] == b[key], 'PAIRED_REQUEST_METADATA_DRIFT:' + key)
        astat = 'invalid' if not a['valid'] else ('correct' if a['exact'] else 'wrong')
        bstat = 'invalid' if not b['valid'] else ('correct' if b['exact'] else 'wrong')
        transitions[bstat + '_to_' + astat] += 1
        both_valid += a['valid'] and b['valid']
        content_agreement += a['valid'] and b['valid'] and prediction_key(a) == prediction_key(b)
    return {'questions': len(deltas), 'roots': len(root_deltas),
            'planned_paired_requests': sum(transitions.values()),
            'question_equal_exact_difference': mean(deltas),
            'root_equal_exact_difference': mean(mean(values) for values in root_deltas.values()),
            'paired_root_descriptive_uncertainty': root_uncertainty(mean(values) for values in root_deltas.values()),
            'question_equal_partial_difference': mean(partial_deltas),
            'improved_questions': sum(delta > 0 for delta in deltas),
            'worsened_questions': sum(delta < 0 for delta in deltas),
            'unchanged_questions': sum(delta == 0 for delta in deltas),
            'paired_request_transition_direction': 'right_to_left',
            'paired_request_transitions': dict(transitions),
            'both_valid_paired_requests': both_valid,
            'content_prediction_agreement_calls': content_agreement,
            'content_prediction_agreement_fraction_both_valid': content_agreement / both_valid if both_valid else None,
            'question_equal_estimates_are_points_only': True,
            'per_question_changes_private_not_for_challenger': per_question}


def percent(value):
    return 'NA' if value is None else f'{100 * value:.2f}%'


def report_text(result):
    standard = lambda name: name.endswith(('/ALL', '/multi_select', '/ordering'))
    text = [
        '# 第七天固定终点 TEST 评估', '',
        '本次比较 S0（M-clean）、A1、B1 三个固定终点；每模型89题、1,446次计划作答，共4,338次。A2/B2未产出，当前完成一次训练阶段。', '',
        '三个来源使用独立分母，historical115 的 Gold6 控制题另列；不提供合并89题作为主指标。所有来源有历史暴露，来源之间存在 root 重叠，不能解释为全新 root 泛化。', '',
        '“整题正确率”先对同一题的固定置换与重复取均值，再对题等权平均；“部分得分”为多选罚错分或排序 Kendall 分，范围[-1,1]。二者均保留无效作答在计划分母内。另列有效排序作答的成对顺序正确率，无效输出单独计数。', '',
        '| 评估组/题型 | 题数 | 计划作答/模型 | S0 整题正确率 | A1 整题正确率 | B1 整题正确率 |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for name, s0 in result['models']['S0']['groups'].items():
        if not standard(name):
            continue
        a1 = result['models']['A1']['groups'][name]
        b1 = result['models']['B1']['groups'][name]
        text.append(f"| {name} | {s0['questions']} | {s0['planned_requests']} | {percent(s0['question_equal_exact'])} | {percent(a1['question_equal_exact'])} | {percent(b1['question_equal_exact'])} |")
    text += ['', '| 评估组/题型 | S0 部分得分 | A1 部分得分 | B1 部分得分 | S0/A1/B1 无效作答数 |',
             '|---|---:|---:|---:|---:|']
    for name, s0 in result['models']['S0']['groups'].items():
        if not standard(name):
            continue
        a1 = result['models']['A1']['groups'][name]
        b1 = result['models']['B1']['groups'][name]
        text.append(f"| {name} | {s0['question_equal_partial']:.4f} | {a1['question_equal_partial']:.4f} | {b1['question_equal_partial']:.4f} | {s0['invalid_requests']}/{a1['invalid_requests']}/{b1['invalid_requests']} |")
    text += ['', '下表单元格为“所有固定作答全对题数 / 至少一次全对题数”，分母均为该评估组题数；与请求级正确率含义不同。', '',
             '| 评估组 | 题数 | S0 | A1 | B1 |', '|---|---:|---:|---:|---:|']
    for name, s0 in result['models']['S0']['groups'].items():
        if not name.endswith('/ALL'):
            continue
        a1 = result['models']['A1']['groups'][name]
        b1 = result['models']['B1']['groups'][name]
        text.append(f"| {name} | {s0['questions']} | {s0['all_correct_questions']}/{s0['any_correct_questions']} | {a1['all_correct_questions']}/{a1['any_correct_questions']} | {b1['all_correct_questions']}/{b1['any_correct_questions']} |")
    text += ['', '排序成对准确率仅以有效排序作答为分母：', '',
             '| 评估组 | S0 成对准确率(有效作答数) | A1 成对准确率(有效作答数) | B1 成对准确率(有效作答数) |',
             '|---|---:|---:|---:|']
    for name, s0 in result['models']['S0']['groups'].items():
        if not name.endswith('/ordering'):
            continue
        a1 = result['models']['A1']['groups'][name]
        b1 = result['models']['B1']['groups'][name]
        text.append(f"| {name} | {percent(s0['pair_accuracy_valid_call_mean'])}({s0['pair_accuracy_valid_call_denominator']}) | {percent(a1['pair_accuracy_valid_call_mean'])}({a1['pair_accuracy_valid_call_denominator']}) | {percent(b1['pair_accuracy_valid_call_mean'])}({b1['pair_accuracy_valid_call_denominator']}) |")
    text += ['', '多选题按 Gold 数量拆分（仅列实际有题的分层）：', '',
             '| 评估组 | Gold 数量 | 题数 | S0 整题正确率 | A1 整题正确率 | B1 整题正确率 |',
             '|---|---:|---:|---:|---:|---:|']
    for name, s0 in result['models']['S0']['groups'].items():
        if '/multi_select_gold_' not in name:
            continue
        a1 = result['models']['A1']['groups'][name]
        b1 = result['models']['B1']['groups'][name]
        gold_n = name.split('/multi_select_gold_', 1)[1].replace('_control', '（控制）')
        text.append(f"| {name.rsplit('/multi_select_gold_', 1)[0]} | {gold_n} | {s0['questions']} | {percent(s0['question_equal_exact'])} | {percent(a1['question_equal_exact'])} | {percent(b1['question_equal_exact'])} |")
    text += ['', '补充集的冻结出题设计分层表现：', '',
             '| planned_design | 题数 | S0 | A1 | B1 |', '|---|---:|---:|---:|---:|']
    for name, s0 in result['models']['S0']['groups'].items():
        if '/planned_design_' not in name:
            continue
        a1 = result['models']['A1']['groups'][name]
        b1 = result['models']['B1']['groups'][name]
        design = name.rsplit('/planned_design_', 1)[1]
        text.append(f"| {design} | {s0['questions']} | {percent(s0['question_equal_exact'])} | {percent(a1['question_equal_exact'])} | {percent(b1['question_equal_exact'])} |")
    text += ['', 'B1−A1 / A1−S0 / B1−S0 的题等权差异、root 等权差异、逐题升降和配对作答变化记录在 PAIRED_AGGREGATE.json/comparisons。planned_design/relation_class 标签完全来自冻结的 CONTENT_ADMISSION.json，没有按模型回答结果重分类。', '',
             '跨置换内容一致性、同置换重复一致性和有效作答 FP/FN，见 PAIRED_AGGREGATE.json 的 models/groups。该文件也记录逐题变化与配对作答变化，属于私有评估记录，不可导出给 Challenger。', '',
             '主表为题等权点估计；JSON另给 root 等权均值及至少3个 root 时的描述性 cluster bootstrap 区间，1/2个 root 仅列均值和范围。该区间针对 root 等权指标，与题等权主指标不同。重复作答、同 root 题目以及跨来源重叠均具有依赖性，不能将4,338次作答视为独立实验样本。B1−A1仅为这一次匹配训练比较的描述差异，不能单独证明反馈的因果收益或两轮 self-play 成功。TEST结果不用于挑选 checkpoint、修题或生成反馈。', '',
             '独立算分使用冻结原始输出、有限答案空间及展示到内容的双射映射，未调用模型或原评分函数；结果与各终点评分包逐请求和逐题核对一致。', '']
    return '\n'.join(text)


def main():
    freeze_path = HERE / 'TEST_REQUEST_FREEZE.json'
    frozen = read(freeze_path)
    verify_bindings(frozen['input_bindings'] + frozen['artifact_bindings'] + [frozen['wrapper']])
    requests = lines(HERE / 'TEST_REQUESTS.jsonl')
    unique_by(requests, 'request_id')
    require(len(requests) == 1446 and all(row['split'] == 'test' and row['branch'] == 'direct' for row in requests), 'TEST_BOUNDARY_OR_REQUEST_DENOMINATOR_CHANGED')
    # Finish checking every sealed raw packet before opening any private Gold.
    sealed = {}
    bindings = [binding(__file__), binding(freeze_path)]
    manifests = {}
    for role in ROLES:
        directory = HERE / 'models' / role
        aggregate = read(directory / 'AGGREGATE.json')
        require(aggregate['status'] in ('COMPLETE', 'FAILED_FIXED_DENOMINATOR'), 'ENDPOINT_NOT_SCORED:' + role)
        verify_bindings([aggregate['execution_manifest'], aggregate['raw_response_freeze'], aggregate['private_scores']])
        manifest = read(directory / 'EXECUTION_MANIFEST.json')
        require(manifest['model_role'] == role and manifest['request_freeze'] == binding(freeze_path), 'ENDPOINT_TEST_FREEZE_DRIFT')
        verify_bindings([manifest['request_freeze'], manifest['endpoint_proof'],
                         manifest['historical_DEV_identity_manifest']]
                        + manifest['endpoint_evidence_bindings'] + manifest['model_bindings']
                        + manifest['tokenizer_bindings'] + manifest['code_bindings'])
        require(Path(manifest['model_directory']).is_dir(), 'ENDPOINT_MODEL_DIRECTORY_MISSING')
        rawfreeze = read(directory / 'RAW_RESPONSE_FREEZE.json')
        require(rawfreeze['request_freeze'] == binding(freeze_path)
                and rawfreeze['execution_manifest'] == binding(directory / 'EXECUTION_MANIFEST.json')
                and rawfreeze['planned_request_count'] == 1446
                and rawfreeze['gold_opened_by_live'] is False, 'RAW_FREEZE_CONTRACT_DRIFT')
        verify_bindings(rawfreeze['files'])
        sealed[role] = aggregate, rawfreeze
        manifests[role] = manifest
        bindings += [binding(directory / 'AGGREGATE.json'), aggregate['execution_manifest'],
                     aggregate['raw_response_freeze'], aggregate['private_scores']]
    gold = unique_by(lines(HERE / 'private/TEST_GOLD.jsonl'), 'candidate_id')
    public = unique_by(lines(HERE / 'TEST_PUBLIC_QUESTIONS.jsonl'), 'candidate_id')
    maps = read(HERE / 'TEST_DISPLAY_MAPS.json')
    choices = read(HERE / 'FINITE_CHOICES.json')
    require(len(gold) == 89, 'TEST_GOLD_QUESTION_DENOMINATOR_CHANGED')
    models = {}
    for role in ROLES:
        aggregate, rawfreeze = sealed[role]
        scored = read(HERE / 'models' / role / 'private/SCORED_REQUESTS.json')
        model = independently_recalculate(role, manifests[role], scored, rawfreeze, requests, gold, maps, choices)
        require(len(model['questions']) == 89 and len(model['requests']) == 1446, 'ENDPOINT_RECALCULATED_DENOMINATOR_CHANGED')
        model['status'] = aggregate['status']
        models[role] = model
    metadata = enrich_question_metadata(models, public, gold, frozen)
    for model in models.values():
        groups = group_ids(model['questions'])
        model['groups'] = {name: summarize_group(model, ids) for name, ids in groups.items()}
    groups = group_ids(models['S0']['questions'])
    comparisons = {}
    for left, right in (('B1', 'A1'), ('A1', 'S0'), ('B1', 'S0')):
        require(set(models[left]['requests']) == set(models[right]['requests'])
                and set(models[left]['questions']) == set(models[right]['questions']), 'PAIRED_IDS_DIFFER')
        comparisons[left + '_minus_' + right] = {
            name: {'difference_direction': left + '_minus_' + right,
                   **compare_group(models[left], models[right], ids)}
            for name, ids in groups.items()}
    result = {
        'schema_version': 'day7_fixed_endpoint_independently_recalculated_TEST_v1',
        'status': 'COMPLETE' if all(model['status'] == 'COMPLETE' for model in models.values()) else 'COMPLETE_WITH_FAILED_FIXED_DENOMINATOR_ENDPOINT',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'primary_comparison': 'B1_minus_A1/historical115_TEST/HISTORICAL_REGRESSION_PRIMARY/ALL',
        'primary_question_count': 64, 'primary_planned_requests_per_model': 1032,
        'total_questions_per_model_inventory_only': 89,
        'total_requests_per_model': 1446, 'total_planned_requests': 4338,
        'missing_models': {'A2': 'NOT_PRODUCED', 'B2': 'NOT_PRODUCED'},
        'completed_training_stages': 1,
        'paired_by': 'Identical frozen question/request IDs, prompt bytes, display maps, sample indices, sampling and seeds across fixed endpoints',
        'item_mapping_change_audit': {'status': 'UNCHANGED_BY_FROZEN_PAIRED_DESIGN',
                                      'question_ids_per_endpoint': 89,
                                      'request_prompt_mapping_ids_per_endpoint': 1446,
                                      'cross_endpoint_question_or_prompt_mapping_changes': 0,
                                      'audit_basis': 'Same shared TEST request freeze, display maps and public question artifacts; paired request IDs, prompt IDs/hashes, mappings, and item identity validated.'},
        'independent_recalculation': {'status': 'PASS', 'requests': 4338, 'question_endpoint_pairs': 267,
                                      'native_scorer_called': False, 'model_requests': 0,
                                      'strict_validity': 'RETURNED plus stop/eos plus exact frozen finite choice plus bijective content remapping and complete ordering',
                                      'invalid_partial_contribution': 0,
                                      'partial_score_formulas': {'multi_select': 'TP/G-FP/(N-G); Gold=N control TP/G', 'ordering': '(correct_pairs-wrong_pairs)/all_gold_pairs'}},
        'models': {role: {'status': model['status'], 'groups': model['groups']} for role, model in models.items()},
        'comparisons': comparisons,
        'question_equal_estimates_are_points_only': True,
        'root_uncertainty_policy': 'Descriptive root-equal cluster bootstrap only for groups with at least3 roots; 1/2 roots report mean/range only. Not a CI on question-equal primary estimates.',
        'metadata_views': metadata,
        'cohort_metric_key': 'cohort/requesting-origin and reporting role are kept separate; never pool across these groups for a primary score',
        'no_pooled_89_question_primary_metric': True,
        'no_unseen_root_claim': True, 'historically_exposed': True, 'overlapping_roots_across_cohorts': True,
        'no_causal_feedback_claim': True, 'checkpoint_selected_by_test': False,
        'challenger_release': False, 'private_item_changes_not_for_challenger': True,
        'claim_limit': 'One completed training stage on historically exposed, root-overlapping evaluation cohorts; descriptive paired differences do not establish feedback causality, unseen-root generalization, or a two-stage self-play cycle.',
        'bindings': bindings,
    }
    report = HERE / 'REPORT_ZH.md'
    with report.open('x') as stream:
        stream.write(report_text(result))
    result['report'] = binding(report)
    save(HERE / 'PAIRED_AGGREGATE.json', result)
    print(json.dumps({key: result[key] for key in ('status', 'primary_question_count', 'total_requests_per_model', 'independent_recalculation', 'models')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
