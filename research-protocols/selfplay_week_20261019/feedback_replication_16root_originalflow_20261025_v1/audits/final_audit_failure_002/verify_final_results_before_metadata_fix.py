#!/usr/bin/env python3
"""Independent CPU audit. Run only after the parent confirms final outputs sealed.

This script never generates model requests and never calls the analysis functions
under review. It reads sealed inputs, reapplies frozen mechanical validation,
rebuilds scores/aggregates independently, and writes audit receipts only.
"""
from __future__ import annotations

import argparse
import ast
import collections
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
from datetime import datetime, timezone
from fractions import Fraction

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parents[1]
WEEK = BASE.parent
OLD = WEEK / 'feedback_closed_loop_20261024_v1'
PRIOR = WEEK / 'feedback_replication_20root_resume_20261025_v1'
INITIAL = WEEK / 'feedback_replication_20root_20261025_v1'
EXPECTED_COUNTS = {'original4': 128, 'prior_recovery': 146, 'current_continuation': 298}
CURRENT_SOLVER_SHA = 'd40b421782b039a7ef32fcad4a5092a7936a5b708e1584911f38a579bf906c91'
COMBINED_SHA = 'a9049d378a1ecb025391079949ec0fb78dd74b463dd5acce1a5e49b2a45a7275'


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    p = Path(path).resolve()
    content = p.read_bytes()
    return {'path': str(p), 'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}


def atomic_json(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


class AuditError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def mean(values):
    vals = list(values)
    return math.fsum(vals) / len(vals) if vals else None


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def differences(actual, expected, path='', found=None):
    """Compare every field, with small tolerance for independently summed floats."""
    found = [] if found is None else found
    if len(found) >= 50:
        return found
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            found.append({'path': path, 'actual': actual, 'expected_type': 'dict'})
        else:
            if set(actual) != set(expected):
                found.append({'path': path, 'extra_fields': sorted(set(actual) - set(expected)),
                              'missing_fields': sorted(set(expected) - set(actual))})
            for key in expected.keys() & actual.keys():
                differences(actual[key], expected[key], path + '.' + key, found)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            found.append({'path': path, 'actual_length': len(actual) if isinstance(actual, list) else None,
                          'expected_length': len(expected)})
        else:
            for i, (a, b) in enumerate(zip(actual, expected)):
                differences(a, b, f'{path}[{i}]', found)
    elif isinstance(expected, bool):
        if not isinstance(actual, bool) or actual != expected:
            found.append({'path': path, 'actual': actual, 'expected': expected})
    elif isinstance(expected, (int, float)):
        if (not isinstance(actual, (int, float)) or isinstance(actual, bool)
                or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)):
            found.append({'path': path, 'actual': actual, 'expected': expected})
    elif actual != expected:
        found.append({'path': path, 'actual': actual, 'expected': expected})
    return found


def equal(actual, expected, label):
    diff = differences(actual, expected, label)
    require(not diff, json.dumps({'comparison': label, 'differences': diff}, ensure_ascii=False))


def final_answer_only(raw, presentation, kind):
    """Parse the diagnostic independently, without evidence-contract validation."""
    text = raw['raw_output'] if 'raw_output' in raw else raw.get('text', '')
    text = text.replace('<|im_end|>', '').strip()
    if text.startswith('<think>') and '</think>' in text:
        text = text.split('</think>', 1)[1].lstrip()
    try:
        obj = json.loads(text)
        answer = obj['final_answer']
        require(isinstance(answer, str) and answer.startswith('<box>') and answer.endswith('</box>'),
                'unusable final box')
        inner = answer[5:-6]
        labels = inner.split(' ')
        require(1 <= len(labels) <= 6 and all(s in list('ABCDEF') for s in labels), 'unusable final labels')
        require(len(set(labels)) == len(labels), 'duplicate final labels')
        if kind == 'ordering':
            require(sorted(labels) == list('ABCDEF'), 'incomplete final ordering')
        pred = [presentation['display_to_canonical'][label] for label in labels]
        gold = presentation['canonical_gold']
        correct = pred == gold if kind == 'ordering' else set(pred) == set(gold)
        return {'valid': True, 'exact': correct, 'prediction': pred}
    except (AuditError, ValueError, TypeError, KeyError, AttributeError):
        return {'valid': False, 'exact': False, 'prediction': []}


def own_strict_score(row, presentation, labels, valid):
    """Independent canonical exact/FP/FN/pairwise arithmetic after frozen validation."""
    kind = row['question_type']
    pred = [presentation['display_to_canonical'][s] for s in labels] if valid else []
    if kind == 'multi_select':
        pred.sort()
    gold = presentation['canonical_gold']
    result = {key: row[key] for key in ('request_id', 'candidate_id', 'root_id', 'cohort',
                                       'condition', 'question_type', 'task_operation')}
    result.update(valid=valid,
                  exact=bool(valid and (set(pred) == set(gold) if kind == 'multi_select' else pred == gold)),
                  prediction=pred, gold=gold, display_prediction=labels if valid else [],
                  display_gold=presentation['display_gold'], gold_count=len(gold) if kind == 'multi_select' else None,
                  false_positive=None, false_negative=None, pairwise_correct=None,
                  pairwise_total=None, allselect_exact=None)
    if kind == 'multi_select':
        result.update(false_positive=len(set(pred).difference(gold)),
                      false_negative=len(set(gold).difference(pred)),
                      allselect_exact=len(gold) == len(presentation['display_to_canonical']))
    else:
        pairs = [(gold[i], gold[j]) for i in range(len(gold)) for j in range(i + 1, len(gold))]
        result.update(pairwise_total=len(pairs), pairwise_correct=0 if not valid else
                      sum(pred.index(left) < pred.index(right) for left, right in pairs))
    return result


def partial_score(score):
    if score['question_type'] == 'ordering':
        return {'pairwise_accuracy': score['pairwise_correct'] / score['pairwise_total']}
    pred, gold = set(score['prediction']), set(score['gold'])
    tp = len(pred & gold)
    return {'precision': tp / len(pred) if pred else 0,
            'recall': tp / len(gold), 'f1': 2 * tp / (len(pred) + len(gold))}


def percentile_of_sorted(sorted_values, probability):
    rank = probability * (len(sorted_values) - 1)
    lo, hi = math.floor(rank), math.ceil(rank)
    return sorted_values[lo] * (hi - rank) + sorted_values[hi] * (rank - lo) if hi != lo else sorted_values[lo]


def root_bootstrap(root_values):
    if not root_values:
        return None
    rng = random.Random(20261025)
    n = len(root_values)
    # Uniform inverse-CDF sampling independently reproduces the specified stream.
    draws = sorted(math.fsum(root_values[int(rng.random() * n)] for _ in range(n)) / n for __ in range(10000))
    return {'resamples': 10000, 'seed': 20261025, 'unit': 'root',
            'percentile95': [percentile_of_sorted(draws, .025), percentile_of_sorted(draws, .975)]}


def independent_paired(records, planned_roots, slots, include_bootstrap):
    cells = collections.defaultdict(dict)
    for row in records:
        key = (row['root_id'], row['slot_id'], row['arm'])
        require(row['variant_id'] not in cells[key], 'duplicate variant in independent paired table')
        cells[key][row['variant_id']] = row
    roots, all_slots = [], []
    for root in planned_roots:
        common = []
        for slot in slots:
            arm_cells = [cells.get((root, slot, arm), {}) for arm in ('F', 'G')]
            if not all(set(cell) == {'perm_00', 'perm_01'} and all(r['returned'] for r in cell.values())
                       for cell in arm_cells):
                continue
            values = []
            for cell in arm_cells:
                require(len({r['candidate_id'] for r in cell.values()}) == 1, 'multiple candidates per paired arm-slot')
                values.append((Fraction(sum(int(r['exact']) for r in cell.values()), 2),
                               Fraction(sum(int(r['answer_only']['exact']) for r in cell.values()), 2)))
            f, af = values[0]
            g, ag = values[1]
            common.append({'slot_id': slot, 'F': float(f), 'G': float(g), 'difference': float(f-g),
                           'answer_F': float(af), 'answer_G': float(ag), 'answer_difference': float(af-ag)})
        roots.append({'root_id': root, 'common_slot_count': len(common), 'common_slots': common,
                      **{key: mean(s[key] for s in common) for key in ('F', 'G', 'difference', 'answer_difference')}})
        all_slots.extend(common)
    eligible = [r for r in roots if r['common_slot_count'] > 0]
    complete = [r for r in eligible if r['common_slot_count'] == len(slots)]
    return {'slots': slots, 'planned_roots': len(planned_roots), 'eligible_roots': len(eligible),
            'common_root_slots': len(all_slots), 'paired_returned_responses': 4 * len(all_slots),
            **{key: mean(r[key] for r in eligible) for key in ('F', 'G', 'difference')},
            'bootstrap': root_bootstrap([r['difference'] for r in eligible]) if include_bootstrap else None,
            'answer_only_difference': mean(r['answer_difference'] for r in eligible),
            'complete_slot_roots': len(complete),
            'complete_slot_root_difference': mean(r['difference'] for r in complete),
            'pooled_common_slot_difference_descriptive': mean(s['difference'] for s in all_slots),
            'roots': roots}


def independent_descriptive(records, slots_all, primary):
    groups = [('ALL', slots_all), ('NON_MC6', primary), ('MC2_5', ['MC2', 'MC3', 'MC4', 'MC5']),
              ('ORDERING', ['ORDER1', 'ORDER2', 'ORDER3']), ('MC6_CONTROL', ['MC6'])]
    groups += [(s, [s]) for s in slots_all]
    groups += [('PRESENTATION_perm_00', slots_all), ('PRESENTATION_perm_01', slots_all)]
    result = []
    for arm in ('F', 'G'):
        for group, slots in groups:
            prepared = [r for r in records if r['arm'] == arm and r['slot_id'] in slots
                        and (not group.startswith('PRESENTATION_') or r['variant_id'] == group[len('PRESENTATION_'):])]
            rows = [r for r in prepared if r['returned']]
            mc = [r for r in rows if r['question_type'] == 'multi_select']
            order = [r for r in rows if r['question_type'] == 'ordering']
            valid_order = [r for r in order if r['valid']]
            candidates = collections.defaultdict(list)
            for row in rows:
                candidates[row['candidate_id']].append(row)
            pairs = [pair for pair in candidates.values() if len(pair) == 2]
            identical = sum(all(r['valid'] for r in pair) and
                            (set(pair[0]['prediction']) == set(pair[1]['prediction'])
                             if pair[0]['question_type'] == 'multi_select' else
                             pair[0]['prediction'] == pair[1]['prediction']) for pair in pairs)
            greater = mean(len(r['prediction']) > len(r['gold']) for r in mc)
            less = mean(len(r['prediction']) < len(r['gold']) for r in mc)
            mc_partials = [partial_score(r) for r in mc]
            result.append({'arm': arm, 'group': group, 'prepared': len(prepared), 'returned': len(rows),
                           'contract_valid': sum(r['valid'] for r in rows),
                           'strict_exact_count': sum(r['exact'] for r in rows),
                           'strict_exact': mean(r['exact'] for r in rows),
                           'answer_only_exact': mean(r['answer_only']['exact'] for r in rows),
                           'mc_false_positive_labels': sum(r['false_positive'] for r in mc),
                           'mc_false_negative_labels': sum(r['false_negative'] for r in mc),
                           'mc_overselection_response_rate': greater,
                           'mc_underselection_response_rate': less,
                           'mc_contains_false_positive_rate': mean(r['false_positive'] > 0 for r in mc),
                           'mc_contains_false_negative_rate': mean(r['false_negative'] > 0 for r in mc),
                           'mc_label_count_greater_than_gold_rate': greater,
                           'mc_label_count_less_than_gold_rate': less,
                           'mc_precision': mean(r['precision'] for r in mc_partials),
                           'mc_recall': mean(r['recall'] for r in mc_partials),
                           'mc_f1': mean(r['f1'] for r in mc_partials),
                           'order_pairwise_strict': ratio(sum(r['pairwise_correct'] for r in order), 15 * len(order)),
                           'order_pairwise_contract_valid_only': ratio(sum(r['pairwise_correct'] for r in valid_order), 15 * len(valid_order)),
                           'semantic_pairs': len(pairs),
                           'both_exact': sum(all(r['exact'] for r in pair) for pair in pairs),
                           'at_least_one_exact': sum(any(r['exact'] for r in pair) for pair in pairs),
                           'both_valid_canonical_predictions_identical': identical,
                           'canonical_predictions_identical': identical})
    return result


def audit():
    checks, snapshots = {}, {}

    def snapshot(path):
        b = binding(path)
        snapshots[b['path']] = b
        return b

    def checked(b):
        equal(binding(b['path']), b, 'bound_file:' + b['path'])
        snapshots[b['path']] = b

    required = [BASE / 'scripts/solver_continue.py', BASE / 'scripts/combined_analysis.py',
                BASE / 'ORIGINAL_ANALYSIS_PLAN.json', BASE / 'ANALYSIS_CONTINUATION_FREEZE.json',
                BASE / 'CONTINUATION_MANIFEST.json', BASE / 'QUESTION_FREEZE.json',
                BASE / 'execution/PREPARATION.json', BASE / 'execution/RAW_RESPONSE_FREEZE.json',
                BASE / 'COMBINED_ANALYSIS.json', BASE / 'NORMALIZED_SCORES.json', BASE / 'ROOT_DIFFERENCES.csv',
                Path(__file__)]
    require(all(p.exists() for p in required), 'sealed final artifacts are incomplete')
    for p in required:
        snapshot(p)
    require(binding(BASE / 'scripts/solver_continue.py')['sha256'] == CURRENT_SOLVER_SHA, 'unreviewed final Solver script')
    require(binding(BASE / 'scripts/combined_analysis.py')['sha256'] == COMBINED_SHA, 'unreviewed final combined analysis script')
    for key in ('original_plan', 'read_only_copy', 'continuation_manifest'):
        checked(read(BASE / 'ANALYSIS_CONTINUATION_FREEZE.json')[key])
    checks['reviewed_scripts_and_frozen_plan_bindings'] = 'PASS'

    # Importing exposes only frozen helper functions; no run/prepare/analyze/LLM call.
    spec = importlib.util.spec_from_file_location('final_audit_frozen_continuation', BASE / 'scripts/solver_continue.py')
    sc = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = sc
    spec.loader.exec_module(sc)
    prep, public_current, tokens = sc.public()
    raw_freeze = read(BASE / 'execution/RAW_RESPONSE_FREEZE.json')
    sc.frozen.check_raw(raw_freeze, public_current, tokens)
    require(raw_freeze['requests'] == EXPECTED_COUNTS['current_continuation'], 'unexpected final current request count')
    require(raw_freeze['returned'] == EXPECTED_COUNTS['current_continuation'] and raw_freeze['failure'] is None,
            'current run did not return all298 requests successfully')
    checked(prep['private_scoring_binding'])
    private_current = read(BASE / 'execution/PRIVATE_SCORING.json')
    for key in ('candidate_binding', 'selection_binding', 'question_freeze_binding'):
        checked(private_current[key])
    for b in private_current['private_bindings'] + private_current['reused_shard_bindings']:
        checked(b)
    checks['frozen_public_preparation_and_raw_verifier'] = 'PASS'

    cohorts, shard_results = {}, []
    for folder, cohort in ((OLD, 'original4'), (PRIOR, 'prior_recovery'), (BASE, 'current_continuation')):
        docs = {}
        for name in ('SCORES.json', 'RAW_RESPONSE_FREEZE.json', 'PUBLIC_REQUESTS.json', 'PRIVATE_SCORING.json'):
            path = folder / 'execution' / name
            snapshot(path)
            docs[name] = read(path)
        stored = docs['SCORES.json']['records']
        require(len(stored) == EXPECTED_COUNTS[cohort], 'unexpected shard size:' + cohort)
        public_rows = docs['PUBLIC_REQUESTS.json']['requests']
        private_rows = docs['PRIVATE_SCORING.json']['requests']
        public = {r['request_id']: r for r in public_rows}
        private = {r['request_id']: r['presentation'] for r in private_rows}
        require(len(public) == len(private) == len(stored), 'duplicate/missing public/private requests:' + cohort)
        equal([r['request_id'] for r in stored], [r['request_id'] for r in public_rows], 'score_public_order:' + cohort)
        equal(set(public), set(private), 'public_private_ids:' + cohort)
        normalized, errors = [], collections.Counter()
        answer_only_rows_compared = 0
        for record in stored:
            rid = record['request_id']
            row, presentation = public[rid], private[rid]
            raw_binding = record['raw_binding'] if 'raw_binding' in record else record['raw']
            checked(raw_binding)
            raw = read(raw_binding['path'])
            returned = raw['status'] == 'RETURNED' if 'status' in raw else record.get('returned') is True and isinstance(raw.get('text'), str)
            labels, valid = [], False
            try:
                sc.need(returned and raw.get('finish_reason') == 'stop', 'BACKEND_OR_LENGTH_FAILURE')
                sc.need(not raw.get('capture_errors'), 'CAPTURE_ERRORS')
                thinking, final, boundary = sc.v2.split_output(raw['raw_output'] if 'raw_output' in raw else raw.get('text', ''))
                sc.need(not thinking, 'UNEXPECTED_NONEMPTY_THINKING')
                parsed, labels = sc.v2.legal_answer(final, row)
                sc.orig.base.validate_output(final, row, read(row['schema_path']))
                valid = True
            except Exception as exc:
                errors[str(exc)] += 1
            score = own_strict_score(row, presentation, labels, valid)
            equal(sc.orig.base.score_record(row, presentation, labels, valid), score, 'independent_vs_frozen_score:' + rid)
            if 'variant_id' in record['strict']:
                score['variant_id'] = row['variant_id']
            equal(record['strict'], score, 'strict_score:' + rid)
            diagnostic = final_answer_only(raw, presentation, row['question_type'])
            if 'answer_only' in record:
                original_diag = record['answer_only']
                equal(original_diag['valid'], diagnostic['valid'], 'historical_answer_only_valid:' + rid)
                equal(original_diag['exact'], diagnostic['exact'], 'historical_answer_only_exact:' + rid)
                pred = set if row['question_type'] == 'multi_select' else list
                equal(pred(original_diag['prediction']), pred(diagnostic['prediction']), 'historical_answer_only_prediction:' + rid)
                answer_only_rows_compared += 1
            if 'partial' in record:
                equal(record['partial'], partial_score(score), 'partial_score:' + rid)
            normalized.append({'cohort': cohort, 'request_id': rid, 'root_id': row['root_id'],
                               'candidate_id': row['candidate_id'], 'arm': row['arm'], 'slot_id': row['slot_id'],
                               'variant_id': row['variant_id'], 'returned': returned, 'valid': valid,
                               'exact': score['exact'], 'answer_only': diagnostic, 'prediction': score['prediction'],
                               'gold': score['gold'], 'gold_count': score['gold_count'],
                               'question_type': score['question_type'], 'partial': record.get('partial', {}),
                               'false_positive': score['false_positive'], 'false_negative': score['false_negative'],
                               'pairwise_correct': score['pairwise_correct'], 'pairwise_total': score['pairwise_total'],
                               'raw_binding': raw_binding, 'errors': record.get('errors', []),
                               'finish_reason': raw.get('finish_reason')})
        require(sum(r['returned'] for r in normalized) == len(normalized), 'nonreturned final shard row:' + cohort)
        cohorts[cohort] = normalized
        shard_results.append({'cohort': cohort, 'rows': len(normalized), 'returned': sum(r['returned'] for r in normalized),
                              'strict_valid': sum(r['valid'] for r in normalized),
                              'strict_exact': sum(r['exact'] for r in normalized),
                              'answer_only_exact': sum(r['answer_only']['exact'] for r in normalized),
                              'all_raw_hashes_checked': True, 'strict_pipeline_and_independent_score_parity': True,
                              'strict_errors': dict(errors), 'stored_answer_only_full_semantic_parity_rows': answer_only_rows_compared})
        print(json.dumps({'completed_cohort': cohort, 'rows': len(normalized)}), flush=True)
    checks['strict_scoring_all572_rows'] = 'PASS'
    all_rows = cohorts['original4'] + cohorts['prior_recovery'] + cohorts['current_continuation']
    merged = cohorts['prior_recovery'] + cohorts['current_continuation']
    require(len(all_rows) == 572, 'all-shard total is not572')
    require(len({r['request_id'] for r in all_rows}) == 572, 'duplicate request ID across shards')
    require(len({(r['root_id'], r['arm'], r['slot_id'], r['variant_id']) for r in all_rows}) == 572,
            'duplicate root/arm/slot/variant across shards')
    admitted = {s['candidate_id'] for s in read(BASE / 'QUESTION_FREEZE.json')['slots'] if s['status'] == 'PASS'}
    equal({r['candidate_id'] for r in merged}, admitted, 'full_PASS_admitted_candidate_cohort')
    equal(read(BASE / 'NORMALIZED_SCORES.json'), {'records': all_rows}, 'NORMALIZED_SCORES')
    checks['normalized_all_fields_and_answer_only_all572_rows'] = 'PASS'

    plan = read(BASE / 'ORIGINAL_ANALYSIS_PLAN.json')
    slots_all = [s['slot'] for s in plan['slots_per_root_per_arm']]
    primary = plan['metrics']['primary_scope']
    roots16 = read(BASE / 'CONTINUATION_MANIFEST.json')['roots']
    old_roots = sorted({r['root_id'] for r in cohorts['original4']})
    actual = read(BASE / 'COMBINED_ANALYSIS.json')
    expected = {'adaptive_recovery': True, 'positive_difference_means': 'F easier; negative means F harder',
                'two_displays_are_dependent': True,
                'new16_primary': independent_paired(merged, roots16, primary, True),
                'new16_MC2_5_secondary': independent_paired(merged, roots16, ['MC2', 'MC3', 'MC4', 'MC5'], True),
                'new16_ordering_secondary': independent_paired(merged, roots16, ['ORDER1', 'ORDER2', 'ORDER3'], True),
                'new16_MC6_control': independent_paired(merged, roots16, ['MC6'], False),
                'original4_frozen': independent_paired(cohorts['original4'], old_roots, primary, False),
                'combined20_secondary': independent_paired(all_rows, old_roots + roots16, primary, True),
                'new16_per_operation': {s: independent_paired(merged, roots16, [s], False) for s in slots_all},
                'descriptive_new16': independent_descriptive(merged, slots_all, primary),
                'descriptive_original4': independent_descriptive(cohorts['original4'], slots_all, primary),
                'initial16_reported_run': {'status': 'PARTIAL', 'source_pass': 5, 'content_candidates': 63,
                                          'content_admitted': 49, 'prepared_requests': 98, 'returned': 0,
                                          'basis': [binding(INITIAL / p) for p in ('FINAL_REPORT.json', 'SOURCE_FREEZE.json', 'CONTENT_GATE_FINAL.json', 'execution/RAW_RESPONSE_FREEZE.json')],
                                          'limitation': 'Initial run itself included source recovery; this is its recorded closeout, not a claim that all work stayed inside an unchanged initial budget.'},
                'shard_counts': EXPECTED_COUNTS,
                'score_bindings': [binding(p / 'execution/SCORES.json') for p in (OLD, PRIOR, BASE)]}
    require(isinstance(actual.get('at'), str), 'missing analysis timestamp')
    equal({k: v for k, v in actual.items() if k != 'at'}, expected, 'COMBINED_ANALYSIS')
    checks['all_paired_primary_secondary_per_operation_root_slot_values'] = 'PASS'
    checks['all_four_root_bootstrap_intervals_seed20261025_10000'] = 'PASS'
    checks['all60_descriptive_arm_group_rows'] = 'PASS'
    checks['cohort_and_original_budget_metadata'] = 'PASS'

    with (BASE / 'ROOT_DIFFERENCES.csv').open(newline='') as handle:
        reader = csv.DictReader(handle)
        columns = ['root_id', 'common_slot_count', 'F', 'G', 'difference', 'answer_difference']
        equal(reader.fieldnames, columns, 'root_csv_columns')
        csv_roots = []
        for row in reader:
            csv_roots.append({'root_id': row['root_id'], 'common_slot_count': int(row['common_slot_count']),
                              **{k: None if row[k] == '' else float(row[k]) for k in columns[2:]}})
    equal(csv_roots, [{key: row[key] for key in columns} for row in expected['combined20_secondary']['roots']], 'ROOT_DIFFERENCES_CSV')
    checks['all20_root_csv_rows_including_missing_root'] = 'PASS'

    diagnostic = read(OLD / 'ANSWER_CONTRACT_DIAGNOSTIC.json')
    snapshot(OLD / 'ANSWER_CONTRACT_DIAGNOSTIC.json')
    for summary in diagnostic['summaries']:
        g = summary['group']
        ss = primary if g == 'NON_MC6_PRIMARY' else ['MC2', 'MC3', 'MC4', 'MC5'] if g == 'MC2_5_PRIMARY' else ['ORDER1', 'ORDER2', 'ORDER3'] if g == 'ORDERING' else ['MC6']
        rr = [r for r in cohorts['original4'] if r['arm'] == summary['arm'] and r['slot_id'] in ss]
        rebuilt = {'requests': len(rr), 'strict_correct': sum(r['exact'] for r in rr),
                   'final_only_correct': sum(r['answer_only']['exact'] for r in rr),
                   'final_only_valid': sum(r['answer_only']['valid'] for r in rr)}
        equal({k: summary[k] for k in rebuilt}, rebuilt, 'original4_frozen_answer_diagnostic')
    checks['original4_frozen_answer_only_summary_parity'] = 'PASS'
    for b in list(snapshots.values()):
        equal(binding(b['path']), b, 'stable_snapshot:' + b['path'])
    checks['all_reviewed_artifacts_unchanged_during_audit'] = 'PASS'
    selected = {}
    for key in ('new16_primary', 'original4_frozen', 'combined20_secondary'):
        selected[key] = {k: v for k, v in expected[key].items() if k != 'roots'}
    return {'checks': checks, 'shards': shard_results, 'rows': len(all_rows), 'results': selected,
            'bindings': list(snapshots.values()),
            'primary_no_common_roots': [r['root_id'] for r in expected['new16_primary']['roots'] if not r['common_slot_count']]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sealed-results-confirmed', action='store_true', help='Parent has notified that new rawfreeze, scores and combined outputs are final.')
    args = parser.parse_args()
    if not args.sealed_results_confirmed:
        parser.error('Do not read current outcomes before parent confirms sealed final artifacts.')
    started = now()
    report = {'schema_version': 'fg_final_result_independent_cpu_audit_v1', 'started_at': started,
              'reviewer_id': 'codex_collaboration:/root/g_author',
              'scope': 'Strict numerical results/provenance only; no mechanism review.',
              'method': 'Independent canonical arithmetic, diagnostic parser, grouping and inverse-CDF root resampling; reapply frozen output-contract validation. No calls to combined_analysis aggregation functions.',
              'new_model_calls': 0, 'mechanism_reviewed': False,
              'numeric_tolerance': {'relative': 1e-12, 'absolute': 1e-12}}
    try:
        report.update(audit())
        report.update(status='PASS', findings=[])
    except Exception as exc:
        report.update(status='FAIL', findings=[{'type': type(exc).__name__, 'message': str(exc)}])
    report['completed_at'] = now()
    # Preserve the prior receipt byte-for-byte; attach stale provenance in the new receipt.
    cpu_path = BASE / 'execution/COMBINED_ANALYSIS_CPU_VERIFICATION.json'
    superseded = BASE / 'execution/prepare_failure_001/COMBINED_ANALYSIS_CPU_VERIFICATION_superseded.json'
    if not superseded.exists():
        superseded.parent.mkdir(parents=True, exist_ok=True)
        superseded.write_bytes(cpu_path.read_bytes())
    old_cpu = read(superseded)
    report['superseded_cpu_verification'] = {'binding': binding(superseded),
                                             'snapshot_script_binding': old_cpu.get('script'),
                                             'note': 'Retained prior CPU snapshot predates final analysis-script updates and current-shard execution; it is superseded, not evidence for final572 rows.'}
    cpu = {'schema_version': 'fg_final_three_shard_cpu_verification_v1', 'status': report['status'],
           'at': now(), 'method': report['method'], 'new_model_calls': 0,
           'rows_verified': report.get('rows', 0), 'results': report.get('shards', []),
           'checks': report.get('checks', {}), 'findings': report['findings'],
           'script': binding(BASE / 'scripts/combined_analysis.py'),
           'solver_script': binding(BASE / 'scripts/solver_continue.py'),
           'audit_script': binding(Path(__file__)),
           'superseded_snapshot': report['superseded_cpu_verification'],
           'audit_report_path': str(BASE / 'audits/FINAL_RESULT_AUDIT.json')}
    atomic_json(cpu_path, cpu)
    report['final_cpu_verification_binding'] = binding(cpu_path)
    atomic_json(BASE / 'audits/FINAL_RESULT_AUDIT.json', report)
    print(json.dumps({'status': report['status'], 'rows': report.get('rows'), 'findings': report['findings'],
                      'audit': binding(BASE / 'audits/FINAL_RESULT_AUDIT.json'), 'cpu_verification': binding(cpu_path)}), flush=True)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
