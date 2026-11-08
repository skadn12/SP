"""Independent CPU-only final artifact/score audit of the fixed v127 dev pair."""
from __future__ import annotations
import collections
import csv
import datetime
import hashlib
import importlib.util
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent.parent
REPO = HERE.parent.parent
NATIVE = REPO / 'offline-deliveries/h4_feedback_training_pool_v15_qa/evaluation'
PLAN = REPO / 'offline-deliveries/h4_codex_training_v126/evidence/DEV_EVALUATION_PLAN_AUDIT.json'
DEV = REPO / 'offline-deliveries/h4_codex_dev_test_ab_v124'
EXPECTED_REQUEST_SHA = 'c7c65c81d82b2afd482ec2541768b6fe4fa5c9553b5072a927d2ed7e252fff5a'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def mean(values):
    return sum(values, Fraction()) / len(values) if values else None


def close(actual, expected, label):
    if actual is None or expected is None:
        if actual is not None or expected is not None:
            raise AssertionError(label + ': null mismatch')
    elif not math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError(label + ': value mismatch')


def main():
    tracked = {}
    errors = []
    checked = {}
    metric_digest = {}
    current_status = None

    def require(condition, label):
        if not condition:
            raise AssertionError(label)

    def track(path):
        path = Path(path).resolve()
        actual = sha(path)
        require(str(path) not in tracked or tracked[str(path)] == actual, 'INPUT_CHANGED:' + str(path))
        tracked[str(path)] = actual
        return actual

    def read(path):
        track(path)
        return json.loads(Path(path).read_text())

    def strata(question):
        names = ['all48', 'main41_excluding_gold6'] if question['question_type'] == 'ordering' or len(question['gold']) < 6 else ['all48']
        if question['question_type'] == 'ordering':
            return names + ['ordering14']
        return names + ['multi_select_all34', 'gold' + str(len(question['gold'])),
                        'multi_select_gold6_control7' if len(question['gold']) == 6 else 'multi_select_gold2345_27']

    def fixed_metric(rows, key):
        values = [Fraction(r[key]) for r in rows]
        return mean(values)

    def audit_calls_summary(actual, rows, label):
        total = len(rows)
        require(actual['planned_calls'] == actual['valid_calls'] == total
                and actual['invalid_or_missing_calls'] == 0, label + ': counts')
        exact = sum(r['exact'] for r in rows)
        require(actual['exact_correct'] == exact and Fraction(actual['exact_rate_fraction']) == Fraction(exact, total), label + ': exact')
        close(actual['exact_rate_fixed'], Fraction(exact, total), label + ': exact point')
        for key, value in [('signed_partial', fixed_metric(rows, 'partial'))]:
            m = actual[key]
            require(Fraction(m['mean_fraction']) == value and m['planned_denominator'] == m['defined_count'] == total
                    and m['undefined_count'] == 0, label + ': ' + key)
            close(m['mean'], value, label + ': ' + key + ' point')
            close(m['lower'], value, label + ': ' + key + ' lower')
            close(m['upper'], value, label + ': ' + key + ' upper')
        ms = [r for r in rows if r['qtype'] == 'multi_select']
        order = [r for r in rows if r['qtype'] == 'ordering']
        def detailed(section, key, values):
            m = actual[section][key]
            value = mean(values)
            require(Fraction(m['mean_fraction']) == value and m['defined_count'] == m['planned_denominator'] == len(values)
                    and m['undefined_count'] == 0, label + ': ' + section + '/' + key)
            close(m['mean'], value, label + ': detail point')
        if ms:
            require(actual['multi_select']['planned_calls'] == len(ms), label + ': ms calls')
            values = {'tp': [r['score']['tp'] for r in ms], 'fp': [r['score']['fp'] for r in ms],
                      'fn': [r['score']['fn'] for r in ms], 'predicted_count': [len(r['score']['content_prediction']) for r in ms],
                      'predicted_minus_gold_count': [len(r['score']['content_prediction']) - r['g'] for r in ms],
                      'all6_rate': [int(len(r['score']['content_prediction']) == 6) for r in ms],
                      'overselection_rate': [int(len(r['score']['content_prediction']) > r['g']) for r in ms],
                      'underselection_rate': [int(len(r['score']['content_prediction']) < r['g']) for r in ms]}
            for key, vals in values.items():
                detailed('multi_select', key, [Fraction(v) for v in vals])
            for key in ['tp', 'fp', 'fn']:
                require(actual['multi_select']['known_' + key + '_total'] == sum(r['score'][key] for r in ms), label + ': totals')
        if order:
            require(actual['ordering']['planned_calls'] == len(order), label + ': order calls')
            for key, vals in {'pair_accuracy': [Fraction(r['score']['tp'], 15) for r in order],
                              'correct_pairs': [Fraction(r['score']['tp']) for r in order],
                              'wrong_pairs': [Fraction(r['score']['fp']) for r in order]}.items():
                detailed('ordering', key, vals)

    def audit_candidate_aggregate(actual, ids, candidate_stats, label):
        require(actual['candidate_count'] == len(ids), label + ': candidate count')
        if not ids:
            require(actual['exact_rate_equal_candidate'] is None and actual['signed_partial_equal_candidate'] is None, label + ': uncovered root measured')
            return
        total = sum(candidate_stats[cid]['n'] for cid in ids)
        exact = mean([candidate_stats[cid]['exact'] for cid in ids])
        partial = mean([candidate_stats[cid]['partial'] for cid in ids])
        require(actual['planned_calls'] == actual['valid_calls'] == total
                and Fraction(actual['exact_rate_equal_candidate_fraction']) == exact
                and Fraction(actual['signed_partial_equal_candidate_fraction']) == partial, label + ': candidate aggregate')
        close(actual['exact_rate_equal_candidate'], exact, label + ': candidate exact point')
        close(actual['signed_partial_equal_candidate'], partial, label + ': candidate partial point')
        close(actual['signed_partial_equal_candidate_lower'], partial, label + ': candidate partial lower')
        close(actual['signed_partial_equal_candidate_upper'], partial, label + ': candidate partial upper')

    try:
        track(__file__)
        status = read(HERE / 'RUN_STATUS.json')
        current_status = status['status']
        require(current_status == 'EVALUATION_REPORTED_AWAITING_INDEPENDENT_AUDIT', 'CONTROLLER_NOT_TERMINAL')
        require(status['summary_status'] == 'COMPLETE' and status['actual_calls'] == 1560 and status['pid'] is None, 'CONTROLLER_NOT_COMPLETE')
        require(status['test_evaluation'] is False and status['automatic_retry'] is False, 'TEST_OR_RETRY_ENABLED')
        preflight = read(HERE / 'evidence/INDEPENDENT_PREFLIGHT_AUDIT.json')
        require(preflight['status'] == 'PASS' and preflight['errors'] == [], 'PREFLIGHT_NOT_PASS')
        require(preflight['audited_files'], 'PREFLIGHT_BINDINGS_EMPTY')
        for row in preflight['audited_files']:
            require(track(row['path']) == row['sha256'], 'PREFLIGHT_BINDING_CHANGED:' + row['path'])
        plan = read(PLAN)
        planned = plan['readonly_checks']['plan_requests']
        require(digest(planned) == EXPECTED_REQUEST_SHA and len(planned) == 780, 'REQUEST_PLAN_CHANGED')
        canonical = read(DEV / 'delivery/dev/QUESTIONS_WITH_ANSWERS.json')['questions']
        qmap = {q['candidate_id']: q for q in canonical}
        require(len(canonical) == len(qmap) == 48, 'CANONICAL_48_CHANGED')
        scope = read(HERE / 'SCOPE.json')
        fixed = scope['fixed_dev_root_ids']
        require(len(fixed) == len(set(fixed)) == 15 and len(scope['covered_dev_root_ids']) == 7, 'ROOT_POPULATION')
        scoring = module('v127_final_independent_scoring', NATIVE / 'scoring.py')
        selection = module('v127_final_independent_selection', NATIVE / 'selection.py')
        report_summary = read(HERE / 'PAIRED_SUMMARY.json')
        require(report_summary['status'] == 'COMPLETE' and report_summary['errors'] == [], 'PAIRED_REPORT_NOT_COMPLETE')
        require(report_summary['planned_calls_total'] == 1560 and report_summary['planned_calls_per_model'] == 780, 'REPORT_BUDGET')
        require(report_summary['population'] == {'questions': 48, 'multi_select': 34, 'ordering': 14,
                'multi_select_gold_counts': {'2': 7, '3': 6, '4': 7, '5': 7, '6': 7}, 'covered_roots': 7, 'fixed_dev_roots': 15}, 'REPORT_POPULATION')
        all_rows, perq = {}, {}
        native_results = {}
        for side, model in [('baseline', 'M_clean_policy'), ('new', 'M_v125_newonly')]:
            run = HERE / 'run' / side
            require(not (run / 'FAILED.json').exists(), 'NATIVE_FAILED_ATTEMPT:' + side)
            start = read(run / 'STARTED.json')
            manifest = read(run / 'manifest.json')
            require(start['manifest_sha256'] == track(run / 'manifest.json'), 'STARTED_MANIFEST_BINDING')
            require(manifest['requests'] == planned and manifest['solver_model'] == model, 'MANIFEST_CHANGED')
            complete = read(run / 'COMPLETE.json')
            require(complete['status'] == 'COMPLETE' and complete['actual_call_count'] == 780, 'NATIVE_INCOMPLETE')
            records = [read(p) for p in sorted((run / 'responses').glob('*.json'))]
            require(len(records) == 780 and selection.audit_complete_call_set(manifest, records) == [], 'NATIVE_CALL_SET_NOT_COMPLETE')
            observed = {r['request_id']: r for r in records}
            golds = read(run / 'gold.json')
            rows = []
            for request in planned:
                rid, cid = request['request_id'], request['candidate_id']
                response = observed[rid]
                require((run / 'responses' / (rid + '.json')).is_file(), 'RESPONSE_FILENAME_ID_MISMATCH')
                require(response['solver_model'] == model and response['model_path'] == manifest['model']
                        and response['mode'] == 'full_context' and response['evaluation_valid'] is True
                        and response['data_errors'] == [], 'RESPONSE_MODEL_OR_VALIDITY')
                metadata = response['metadata']
                for key in ['temperature', 'top_p', 'max_tokens', 'structured_output_api', 'structured_output_version']:
                    require(metadata.get(key) == manifest['decoder_contract'][key], 'RESPONSE_DECODER_MISMATCH:' + key)
                require(metadata.get('finish_reason') in ['stop', 'eos'], 'INCOMPLETE_GENERATION')
                require(metadata.get('sampling_seed') == request['seed']
                        and metadata.get('prompt_token_count') == request['prompt_tokens'], 'RESPONSE_RUNTIME_INPUT_METADATA')
                require(type(metadata.get('completion_token_count')) is int
                        and 0 < metadata['completion_token_count'] <= manifest['decoder_contract']['max_tokens'], 'RESPONSE_COMPLETION_TOKEN_BUDGET')
                question = qmap[cid]
                score = scoring.score_response({**question, 'gold_content_ids': golds[cid]}, request, response['raw_output'])
                require(score == response['score'] and score['scored'] is True, 'RAW_RESPONSE_SCORE_MISMATCH')
                # Recompute the formula directly from canonical predictions, independent of saved tp/fp/fn.
                predicted, gold = score['content_prediction'], golds[cid]
                if question['question_type'] == 'multi_select':
                    tp = len(set(predicted) & set(gold)); fp = len(set(predicted) - set(gold)); fn = len(set(gold) - set(predicted))
                    partial = Fraction(tp, len(gold)) - (Fraction(fp, 6 - len(gold)) if len(gold) < 6 else 0)
                    require((tp, fp, fn) == (score['tp'], score['fp'], score['fn']), 'MULTISELECT_COUNTS')
                    exact = set(predicted) == set(gold)
                else:
                    rank = {v: i for i, v in enumerate(predicted)}
                    inversions = sum(rank[a] > rank[b] for i, a in enumerate(gold) for b in gold[i + 1:])
                    require(score['tp'] == 15 - inversions and score['fp'] == inversions, 'ORDERING_PAIR_COUNTS')
                    partial = Fraction(15 - 2 * inversions, 15)
                    exact = predicted == gold
                require(Fraction(score['partial_score_fraction']) == partial and score['exact_match'] == exact, 'DIRECT_SCORE_FORMULA')
                rows.append({'request_id': rid, 'candidate_id': cid, 'root_id': question['root_id'], 'qtype': question['question_type'],
                             'g': len(question['gold']), 'variant_id': request['variant']['variant_id'], 'slot': request['slot'], 'seed': request['seed'],
                             'prompt_sha256': request['prompt_sha256'], 'score': score, 'partial': partial, 'exact': int(exact)})
            all_rows[side] = rows
            byq = {}
            native_expected = []
            for question in canonical:
                cid = question['candidate_id']
                items = [r for r in rows if r['candidate_id'] == cid]
                n = 18 if question['question_type'] == 'multi_select' else 12
                require(len(items) == n, 'CANDIDATE_CALL_COUNT')
                byq[cid] = {'n': n, 'exact': Fraction(sum(r['exact'] for r in items), n), 'partial': mean([r['partial'] for r in items])}
                variants = [f'perm_{i:02d}' for i in range(6 if question['question_type'] == 'multi_select' else 12)]
                scores = {v: [r['partial'] for r in items if r['variant_id'] == v] for v in variants}
                native = selection.select(scores, variants, exact_rate=byq[cid]['exact'], data_errors=[])
                native.update(candidate_id=cid, root_id=question['root_id'], question_type=question['question_type'], expected_full_calls=n, actual_full_calls=n)
                native_expected.append(native)
            perq[side] = byq
            result = read(run / 'RESULTS.json')
            require(result == {'policy_version': manifest['policy_version'], 'solver_model': model, 'model_path': manifest['model'],
                    'ground_calls': 0, 'questions': native_expected, 'call_set_errors': [],
                    'selected_questions': [r['candidate_id'] for r in native_expected if r['selected_by_policy'] is True]}, 'NATIVE_RESULTS_RECOMPUTATION')
            native_results[side] = result
            require(report_summary['models'][side]['solver_model'] == model
                    and report_summary['models'][side]['model_path'] == manifest['model'], 'REPORT_MODEL_LABEL')
            audit_calls_summary(report_summary['models'][side]['call_summary'], rows, side + '/all')
        require([r['request_id'] for r in all_rows['baseline']] == [r['request_id'] for r in all_rows['new']], 'PAIR_CALL_SET')
        per_question = read(HERE / 'PER_QUESTION.json')
        require(len(per_question) == 48 and {q['candidate_id'] for q in per_question} == set(qmap), 'PER_QUESTION_POPULATION')
        for q in per_question:
            cid = q['candidate_id']; original = qmap[cid]
            require(q['root_id'] == original['root_id'] and q['question_type'] == original['question_type']
                    and q['gold_count'] == len(original['gold']) and set(q['strata']) == set(strata(original)), 'PER_QUESTION_METADATA')
            for side in ['baseline', 'new']:
                rows = [r for r in all_rows[side] if r['candidate_id'] == cid]
                audit_calls_summary(q[side], rows, side + '/' + cid)
                means, exacts = [], []
                for vid in sorted({r['variant_id'] for r in rows}):
                    vr = [r for r in rows if r['variant_id'] == vid]
                    audit_calls_summary(q[side]['permutations'][vid], vr, side + '/' + vid)
                    means.append(mean([r['partial'] for r in vr])); exacts.append(Fraction(sum(r['exact'] for r in vr), len(vr)))
                diagnostics = q[side]['permutation_diagnostics']
                require(diagnostics['all_permutations_complete'] is True and diagnostics['planned_permutations'] == len(means), 'PERMUTATION_COUNTS')
                for key, value in {'signed_partial_min': min(means), 'signed_partial_max': max(means), 'signed_partial_range': max(means)-min(means),
                                   'exact_rate_fixed_min': min(exacts), 'exact_rate_fixed_max': max(exacts), 'exact_rate_fixed_range': max(exacts)-min(exacts)}.items():
                    close(diagnostics[key], value, 'PERMUTATION_DIAGNOSTIC')
            delta = q['paired_delta']
            exact = perq['new'][cid]['exact'] - perq['baseline'][cid]['exact']
            partial = perq['new'][cid]['partial'] - perq['baseline'][cid]['partial']
            require(Fraction(delta['exact_rate_delta_fraction']) == exact and Fraction(delta['signed_partial_delta_fraction']) == partial, 'PER_QUESTION_DELTA')
        for name, summary in report_summary['strata'].items():
            ids = [cid for cid, q in qmap.items() if name in strata(q)]
            require(summary['candidate_count'] == len(ids), 'STRATUM_POPULATION:' + name)
            metric_digest[name] = {'candidate_count': len(ids)}
            for side in ['baseline', 'new']:
                audit_candidate_aggregate(summary['models'][side]['equal_candidate'], ids, perq[side], name + '/' + side)
                audit_calls_summary(summary['models'][side]['pooled_calls'], [r for r in all_rows[side] if r['candidate_id'] in ids], name + '/' + side + '/pooled')
                metric_digest[name][side] = {'exact': str(mean([perq[side][cid]['exact'] for cid in ids])),
                                             'signed_partial': str(mean([perq[side][cid]['partial'] for cid in ids]))}
            expected_exact = mean([perq['new'][cid]['exact'] - perq['baseline'][cid]['exact'] for cid in ids])
            expected_partial = mean([perq['new'][cid]['partial'] - perq['baseline'][cid]['partial'] for cid in ids])
            close(summary['paired_equal_candidate_delta']['exact_rate'], expected_exact, name + ': exact delta')
            for key in ['signed_partial', 'signed_partial_lower', 'signed_partial_upper']:
                close(summary['paired_equal_candidate_delta'][key], expected_partial, name + ': partial delta')
        require(set(report_summary['strata']) == {'all48', 'main41_excluding_gold6', 'multi_select_all34', 'multi_select_gold2345_27',
                'multi_select_gold6_control7', 'gold2', 'gold3', 'gold4', 'gold5', 'gold6', 'ordering14'}, 'ALL_GOLD_STRATA_PRESENT')
        roots = read(HERE / 'PER_ROOT.json')
        require(len(roots) == 15 and {r['root_id'] for r in roots} == set(fixed), 'ALL15_ROOTS_REPORTED')
        for root in roots:
            require(set(root['strata']) == set(report_summary['strata']), 'ROOT_STRATA_MISSING')
            ids = [cid for cid, q in qmap.items() if q['root_id'] == root['root_id']]
            require(root['candidate_count'] == len(ids) and root['has_questions'] == bool(ids), 'ROOT_COVERAGE')
            for name, values in root['strata'].items():
                subset = [cid for cid in ids if name in strata(qmap[cid])]
                for side in ['baseline', 'new']:
                    audit_candidate_aggregate(values[side], subset, perq[side], name + '/' + root['root_id'] + '/' + side)
                exact = mean([perq['new'][cid]['exact'] - perq['baseline'][cid]['exact'] for cid in subset])
                partial = mean([perq['new'][cid]['partial'] - perq['baseline'][cid]['partial'] for cid in subset])
                close(values['paired_delta']['exact_rate'], exact, 'ROOT_EXACT_DELTA')
                for key in ['signed_partial', 'signed_partial_lower', 'signed_partial_upper']:
                    close(values['paired_delta'][key], partial, 'ROOT_PARTIAL_DELTA')
        require(set(report_summary['root_macro']) == set(report_summary['strata']), 'ROOT_MACRO_STRATA_MISSING')
        for name, actual in report_summary['root_macro'].items():
            rootids = [root for root in fixed if any(q['root_id'] == root and name in strata(q) for q in canonical)]
            require(actual['covered_root_denominator'] == len(rootids) and actual['fixed_dev_root_population'] == 15
                    and actual['uncovered_roots_not_scored'] == 15 - len(rootids), 'ROOT_MACRO_DENOMINATOR')
            values = {}
            for side in ['baseline', 'new']:
                values[side] = {}
                for key in ['exact', 'partial']:
                    values[side][key] = mean([mean([perq[side][cid][key] for cid, q in qmap.items() if q['root_id'] == root and name in strata(q)]) for root in rootids])
                close(actual['models'][side]['exact_rate_equal_root'], values[side]['exact'], 'ROOT_MACRO_EXACT')
                for key in ['signed_partial_equal_root', 'signed_partial_equal_root_lower', 'signed_partial_equal_root_upper']:
                    close(actual['models'][side][key], values[side]['partial'], 'ROOT_MACRO_PARTIAL')
            close(actual['paired_delta_equal_root']['exact_rate'], values['new']['exact'] - values['baseline']['exact'], 'ROOT_MACRO_EXACT_DELTA')
            for key in ['signed_partial', 'signed_partial_lower', 'signed_partial_upper']:
                close(actual['paired_delta_equal_root'][key], values['new']['partial'] - values['baseline']['partial'], 'ROOT_MACRO_PARTIAL_DELTA')
        paircalls = read(HERE / 'PER_CALL.json')
        require(len(paircalls) == 780 and {r['request_id'] for r in paircalls} == {r['request_id'] for r in planned}, 'PER_CALL_PAIR_POPULATION')
        for record in paircalls:
            rid = record['request_id']
            rows = {side: next(r for r in all_rows[side] if r['request_id'] == rid) for side in ['baseline', 'new']}
            require(record['matched_key_equal'] is True, 'PER_CALL_MATCH_FLAG')
            for key in ['candidate_id', 'variant_id', 'slot', 'seed', 'prompt_sha256']:
                require(rows['baseline'][key] == rows['new'][key] == record['baseline'][key] == record['new'][key], 'PER_CALL_PAIR_KEY')
            for side in ['baseline', 'new']:
                require(record[side]['valid'] is True and record[side]['exact_match'] == rows[side]['exact']
                        and Fraction(record[side]['signed_partial']) == rows[side]['partial'], 'PER_CALL_SCORE')
            require(record['exact_delta'] == rows['new']['exact'] - rows['baseline']['exact']
                    and Fraction(record['signed_partial_delta_fraction']) == rows['new']['partial'] - rows['baseline']['partial'], 'PER_CALL_DELTA')
        csvpath = HERE / 'PER_QUESTION.csv'
        track(csvpath)
        csvrows = list(csv.DictReader(csvpath.read_text().splitlines()))
        require(len(csvrows) == 48 and {r['candidate_id'] for r in csvrows} == set(qmap), 'CSV_48_COVERAGE')
        for row in csvrows:
            cid = row['candidate_id']
            for side in ['baseline', 'new']:
                close(row[side + '_exact'], perq[side][cid]['exact'], 'CSV_EXACT')
                close(row[side + '_signed_partial'], perq[side][cid]['partial'], 'CSV_PARTIAL')
        for binding in report_summary['input_bindings']:
            require(track(binding['path']) == binding['sha256'], 'REPORT_INPUT_CHANGED')
        for name in ['controller.log', 'verify.log', 'baseline.log', 'new.log', 'report.log', 'REPORT_ZH.md',
                     'evidence/INDEPENDENT_PREFLIGHT_AUDIT_ATTEMPT1_FAIL.json']:
            track(HERE / name)
        checked = {'full48_preserved': True, 'fixed15_roots_covered7_roots': True, 'gold2345_and_gold6_separate': True,
                   'planned_calls_per_model': 780, 'total_responses': 1560, 'native_completeness_errors': [],
                   'all1560_raw_scores_recomputed': True, 'all1560_formulas_independently_recomputed': True,
                   'native_RESULTS_all48_recomputed': True, 'per_question_per_call_csv_root_metrics_verified': True,
                   'equal_candidate_pooled_call_equal_root_weights_verified': True,
                   'all_gold2_3_4_5_6_strata_present': True, 'all15_roots_including8_uncovered_reported': True,
                   'selection_diagnostics_do_not_filter_population': True, 'response_retry_or_failed_attempt_artifacts': 0,
                   'frozen_models_tools_inputs_unchanged': True, 'test_inference_calls': 0,
                   'test_content_read_by_audit': False, 'model_weights_loaded_by_audit': False,
                   'training_started_by_audit': False, 'gpu_allocated_by_audit': False}
    except Exception as exc:
        errors.append({'type': type(exc).__name__, 'message': str(exc)})
    changed = [path for path, value in tracked.items() if sha(path) != value]
    if changed:
        errors.append({'type': 'INPUT_CHANGED_DURING_AUDIT', 'paths': changed})
    result = {'schema_version': 'v127_independent_final_dev_audit_v1', 'status': 'PASS' if not errors else 'FAIL',
              'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'auditor': '/root/repair_v122_audit',
              'controller_status': current_status, 'errors': errors, 'checks': checked,
              'independent_descriptive_metrics': metric_digest, 'inputs_changed_during_audit': changed,
              'audited_files': [{'path': path, 'sha256': value} for path, value in sorted(tracked.items())],
              'limitations': ['Descriptive heldout dev comparison only;7 of15 fixed dev roots have48 questions.',
                              'Gold families,questions,permutations and rollouts within roots are correlated;1560 responses are not1560 independent samples.',
                              'Gold6 controls are separate from27 main multi-select questions;reporting all48 combines task types and controls.',
                              'Frozen questions/gold were not semantically rewritten or re-selected;this artifact audit is not fresh external semantic certification.',
                              'Test metadata/exposure history remains as disclosed in v124;no test question/source content or test inference is part of this run.'],
              'preflight_audit_attempt_history': 'Attempt1 FAIL was an audit-script absolute-vs-relative historical manifest path compatibility error,corrected before successful preflight and before first inference. That failed report is retained;formal inference sides each executed once with no score retry.'}
    output = HERE / 'evidence/INDEPENDENT_FINAL_AUDIT.json'
    with output.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'status': result['status'], 'errors': errors, 'path': str(output), 'sha256': sha(output),
                      'audited_files': len(tracked), 'independent_metrics': metric_digest}, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == '__main__':
    raise SystemExit(main())
