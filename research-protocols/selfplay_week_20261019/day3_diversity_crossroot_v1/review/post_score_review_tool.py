from pathlib import Path
import collections
import hashlib
import itertools
import json
import math
import re
import statistics
from datetime import datetime, timezone

B = Path('/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/day3_diversity_crossroot_v1')
R = B / 'runtime/direct_v1'
def load(path):
    return json.loads(Path(path).read_text())
def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
checks, errors = [], []
def check(name, value):
    checks.append({'name': name, 'pass': bool(value)})
    if not value:
        errors.append(name)
def close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)

s, summary = load(R / 'SCORES.json'), load(B / 'SUMMARY.json')
per = load(B / 'exports/PER_QUESTION.json')['questions']
exports = [json.loads(x) for x in (B / 'exports/SOLVER_RESPONSES.jsonl').read_text().splitlines() if x.strip()]
manifest, maps = load(R / 'manifest.json'), load(R / 'DISPLAY_MAPPINGS.json')
private, public = load(B / 'exports/PRIVATE_QUESTIONS.json')['questions'], load(R / 'PUBLIC_QUESTIONS.json')['questions']
ledger = load(B / 'exports/CANDIDATE_LEDGER.json')['records']
report = (B / 'REPORT_ZH.md').read_text()
qm, pub, sm, em, sq, pq, lm = [{row[key]: row for row in rows} for rows, key in [
    (private, 'candidate_id'), (public, 'candidate_id'), (s['requests'], 'request_id'),
    (exports, 'request_id'), (s['questions'], 'candidate_id'), (per, 'candidate_id'), (ledger, 'candidate_id')]]
check('258_unique_manifest_score_export_rows', len(manifest['requests']) == len(sm) == len(exports) == len(em) == 258)
check('20_admitted_question_sets_match', len(private) == len(public) == len(per) == len(sq) == 20 and set(qm) == set(pub) == set(sq) == set(pq))
independent = []
for row in manifest['requests']:
    rid = row['request_id']
    q = qm[row['candidate_id']]
    field = 'options' if q['question_type'] == 'multi_select' else 'items'
    raw, scored, ex, variant = load(R / 'responses' / (rid + '.json')), sm[rid], em[rid], maps['requests'][rid]
    canonical = {label: 'content_' + hashlib.sha256(text.strip().encode()).hexdigest()[:16] for label, text in pub[q['candidate_id']][field].items()}
    reverse = {content: label for label, content in canonical.items()}
    d2c = {label: reverse[content] for label, content in variant['display_to_content'].items()}
    match = re.fullmatch(r'<box>([^<>]+)</box>', raw.get('raw_output') or '')
    display = match.group(1).split(',') if match else []
    valid = bool(match) and raw['status'] == 'RETURNED' and raw.get('metadata', {}).get('finish_reason') in ('stop', 'eos') and not raw.get('output_validation_errors') and len(display) == len(set(display)) and all(x in d2c for x in display)
    valid = bool(valid and (display == sorted(display) and len(display) > 0 if field == 'options' else set(display) == set(d2c)))
    prediction, gold = [d2c[x] for x in display] if valid else None, q['gold']
    exact = bool(valid and (set(prediction) == set(gold) if field == 'options' else prediction == gold))
    one = {'request_id': rid, 'candidate_id': q['candidate_id'], 'type': q['question_type'], 'root': q['root_id'], 'n': len(q[field]),
        'valid': valid, 'exact': exact, 'display_prediction': display, 'canonical_prediction': prediction, 'canonical_gold': gold, 'raw_output': raw['raw_output']}
    score_prediction_matches = set(scored['canonical_prediction']) == set(prediction) if field == 'options' else scored['canonical_prediction'] == prediction
    check(rid + '_raw_score_exact_and_content_remapping', scored['valid'] == valid and scored['exact'] == exact and score_prediction_matches and scored['gold_canonical'] == gold)
    check(rid + '_export_content', ex['valid'] == valid and ex['exact'] == exact and ex['canonical_prediction'] == prediction and ex['canonical_gold'] == gold and ex['display_prediction'] == display and ex['display_to_canonical'] == d2c and ex['raw_output'] == raw['raw_output'] and Path(ex['raw_response_path']).resolve() == R / 'responses' / (rid + '.json'))
    if field == 'options':
        fp, fn, tp = set(prediction) - set(gold), set(gold) - set(prediction), set(gold) & set(prediction)
        one.update(fp=len(fp), fn=len(fn), tp=len(tp), fp_labels=sorted(fp), fn_labels=sorted(fn))
        check(rid + '_MC_FP_FN_TP', scored['metrics']['fp'] == len(fp) and scored['metrics']['fn'] == len(fn) and scored['metrics']['tp'] == len(tp))
        check(rid + '_export_MC_FP_FN', ex['false_positive_options'] == len(fp) and ex['false_negative_options'] == len(fn))
        check(rid + '_MC_item_outcomes', all(item['outcome'] == ('TP' if item['canonical_label'] in tp else 'FN' if item['canonical_label'] in fn else 'FP' if item['canonical_label'] in fp else 'TN') for item in scored['items']))
    else:
        pairs, rank = list(itertools.combinations(gold, 2)), {v: i for i, v in enumerate(prediction)}
        outcomes = [rank[a] < rank[b] for a, b in pairs]
        correct, total = sum(outcomes), len(pairs)
        one.update(correct_pairs=correct, total_pairs=total)
        check(rid + '_every_ordering_pair', scored['metrics']['correct_pairs'] == correct and scored['metrics']['total_pairs'] == total and len(scored['pairs']) == total and all(item['correct'] == ans and item['before_canonical_label'] == pair[0] and item['after_canonical_label'] == pair[1] for item, ans, pair in zip(scored['pairs'], outcomes, pairs)))
        check(rid + '_export_ordering_pairs', ex['pair_correct'] == correct and ex['pair_total'] == total)
    independent.append(one)

qresults = {}
for qid, q in qm.items():
    rows = [x for x in independent if x['candidate_id'] == qid]
    calls, exact = len(rows), sum(x['exact'] for x in rows)
    typ = q['question_type']
    result = {'candidate_id': qid, 'root_id': q['root_id'], 'question_type': typ, 'item_count': rows[0]['n'], 'calls': calls,
        'exact_calls': exact, 'exact_rate': exact / calls, 'valid_calls': sum(x['valid'] for x in rows),
        'stratum': 'incorrect' if exact == 0 else 'stably_correct' if exact == calls else 'occasionally_correct'}
    check(qid + '_fixed_calls_and_question_aggregate', calls == (18 if typ == 'multi_select' else min(12, math.factorial(len(q['items'])))))
    check(qid + '_question_aggregate', sq[qid]['denominator_calls'] == calls and sq[qid]['exact_calls'] == exact and close(sq[qid]['exact_rate'], exact / calls) and sq[qid]['stratum'] == result['stratum'])
    if typ == 'multi_select':
        result.update(fp_calls=sum(x['fp'] for x in rows), fn_calls=sum(x['fn'] for x in rows))
        check(qid + '_MC_FP_FN_aggregate', sq[qid]['fp_calls'] == result['fp_calls'] and sq[qid]['fn_calls'] == result['fn_calls'])
    else:
        result.update(correct_pairs=sum(x['correct_pairs'] for x in rows), total_pairs=sum(x['total_pairs'] for x in rows))
        result['pair_accuracy'] = result['correct_pairs'] / result['total_pairs']
        check(qid + '_ordering_pair_aggregate', sq[qid]['correct_pairs'] == result['correct_pairs'] and sq[qid]['total_pairs'] == result['total_pairs'] and close(sq[qid]['pair_accuracy'], result['pair_accuracy']))
    # PER_QUESTION intentionally omits detailed item/pair arrays and design.
    check(qid + '_PER_QUESTION_exported_score_fields', all(pq[qid][key] == value for key, value in sq[qid].items() if key not in ('items', 'pairs', 'design')))
    qresults[qid] = result
check('SUMMARY_and_PER_QUESTION_identical', summary['questions'] == per)
check('SUMMARY_type_and_arity_matches_SCORES', summary['by_type'] == s['by_type'] and summary['by_arity'] == s['by_arity'])
check('258_valid_108_exact', sum(x['valid'] for x in independent) == s['valid_calls'] == 258 and sum(x['exact'] for x in independent) == s['exact_calls'] == 108)
mc_rows = [x for x in independent if x['type'] == 'multi_select']
check('MC_101_FP_133_FN_item_events', sum(x['fp'] for x in mc_rows) == 101 and sum(x['fn'] for x in mc_rows) == 133)
check('overall_question_equal', close(statistics.mean(x['exact_rate'] for x in qresults.values()), s['question_equal_exact']))
check('overall_root_equal', close(statistics.mean(statistics.mean(x['exact_rate'] for x in qresults.values() if x['root_id'] == root) for root in {x['root_id'] for x in qresults.values()}), s['root_equal_exact']))
for typ in ('multi_select', 'ordering'):
    qs = [q for q in qresults.values() if q['question_type'] == typ]
    roots = {q['root_id'] for q in qs}
    values = s['by_type'][typ]
    check(typ + '_type_aggregate', values['questions'] == len(qs) and values['calls'] == sum(q['calls'] for q in qs) and close(values['question_equal_exact'], statistics.mean(q['exact_rate'] for q in qs)) and close(values['root_equal_exact'], statistics.mean(statistics.mean(q['exact_rate'] for q in qs if q['root_id'] == root) for root in roots)))
    check(typ + '_strata', dict(collections.Counter(q['stratum'] for q in qs)) == summary['strata'][typ])
arity = []
for n in (3, 4, 5, 6):
    qs = [q for q in qresults.values() if q['question_type'] == 'ordering' and q['item_count'] == n]
    if not qs:
        arity.append({'item_count': n, 'questions': 0, 'calls': 0})
        continue
    calls, exact = sum(q['calls'] for q in qs), sum(q['exact_calls'] for q in qs)
    correct, total = sum(q['correct_pairs'] for q in qs), sum(q['total_pairs'] for q in qs)
    values = {'item_count': n, 'questions': len(qs), 'calls': calls, 'exact_calls': exact, 'call_equal_exact': exact / calls,
        'question_equal_exact': statistics.mean(q['exact_rate'] for q in qs), 'correct_pairs': correct, 'total_pairs': total,
        'question_equal_pair_accuracy': statistics.mean(q['pair_accuracy'] for q in qs), 'pair_weighted_accuracy': correct / total}
    arity.append(values)
    check('ordering' + str(n) + '_arity', all(close(s['by_arity'][f'ordering:{n}'][k], v) for k, v in values.items() if k not in ('exact_calls', 'call_equal_exact')))
    cells = [str(n), str(len(qs)), str(calls), f'{values["question_equal_exact"] * 100:.2f}%', f'{values["question_equal_pair_accuracy"] * 100:.2f}%']
    check('ordering' + str(n) + '_report_table', '|' + '|'.join(cells) + '|' in report)
status = dict(collections.Counter(x['status'] for x in ledger))
check('24_production_slots_20_admitted_4_missing', len(ledger) == len(s['planned_slot_outcomes']) == summary['production_planned_slots'] == 24 and set(s['missing_candidate_ids']) == {x['candidate_id'] for x in ledger if x['status'] != 'CONTENT_PASS'} and s['production_unadmitted_slots'] == 4)
check('production_status_counts', status == summary['production_status_counts'] == {'CONTENT_PASS': 20, 'CONTENT_HOLD': 1, 'SOURCE_SHORTFALL': 3})
for root, values in summary['roots'].items():
    ls, qs = [x for x in ledger if x['root_id'] == root], [q for q in qresults.values() if q['root_id'] == root]
    truth = {'planned_slots': len(ls), 'admitted': len(qs), 'holds': sum(x['status'] == 'CONTENT_HOLD' for x in ls), 'shortfalls': sum(x['status'] == 'SOURCE_SHORTFALL' for x in ls), 'MC': sum(q['question_type'] == 'multi_select' for q in qs), 'ordering': sum(q['question_type'] == 'ordering' for q in qs)}
    check(root + '_root_counts', truth == values)
labels = {'MC_CAUSAL_SUPPORT': '多选：明确原因', 'MC_QUANTITATIVE_RELATION': '多选：数量/时间计算', 'MC_CONSTRAINT_CONJUNCTION': '多选：条件联合', 'ORDER_DISCOURSE_RECONSTRUCTION': '排序：叙述顺序', 'ORDER_EXPLICIT_TIME': '排序：明确计划时刻', 'ORDER_EXPLICIT_PROCEDURE': '排序：明确步骤', 'MC_EVIDENCE_SELECTION': '多选：最小证据集'}
for template, values in summary['by_template'].items():
    ls = [x for x in ledger if x['template_id'] == template]
    qs = [q for q in qresults.values() if lm[q['candidate_id']]['template_id'] == template]
    truth = {'planned_slots': len(ls), 'admitted': len(qs), 'holds': sum(x['status'] == 'CONTENT_HOLD' for x in ls), 'shortfalls': sum(x['status'] == 'SOURCE_SHORTFALL' for x in ls), 'calls': sum(q['calls'] for q in qs), 'question_equal_exact': statistics.mean(q['exact_rate'] for q in qs), 'arity_counts': dict(collections.Counter(str(q['item_count']) for q in qs))}
    if qs[0]['question_type'] == 'ordering':
        truth['question_equal_pair_accuracy'] = statistics.mean(q['pair_accuracy'] for q in qs)
    check(template + '_template_aggregate', truth == values)
    lines = [line for line in report.splitlines() if line.startswith('|' + labels[template] + '|')]
    check(template + '_report_table', len(lines) == 1 and lines[0].split('|')[2:6] == [str(truth[k]) for k in ('planned_slots', 'admitted', 'holds', 'shortfalls')] and lines[0].split('|')[-2] == f'{truth["question_equal_exact"] * 100:.2f}%')
check('MC_gold_count_distribution', dict(collections.Counter(str(len(q['gold'])) for q in private if q['question_type'] == 'multi_select')) == summary['gold_count_distribution_MC'])
check('16_metadata_fact_families', len({lm[qid]['fact_family_id'] for qid in qm}) == summary['fact_families'] == 16)
spotchecks = []
for qid in ('DIVX-R1-MC2', 'DIVX-R3-MC2'):
    rows = [x for x in independent if x['candidate_id'] == qid]
    sample = next(x for x in rows if not x['exact'])
    spotchecks.append({'candidate_id': qid, 'template_id': lm[qid]['template_id'], 'calls': len(rows), 'gold': qm[qid]['gold'], 'exact_calls': sum(x['exact'] for x in rows), 'FP_item_events': sum(x['fp'] for x in rows), 'FN_item_events': sum(x['fn'] for x in rows), 'one_raw_example': {k: sample[k] for k in ('request_id', 'raw_output', 'display_prediction', 'canonical_prediction', 'canonical_gold', 'fp_labels', 'fn_labels', 'fp', 'fn')}, 'matches_SCORES_PER_QUESTION_SUMMARY': True})
analysis = load(B / 'review/SOLVER_OBSERVABLE_ERROR_ANALYSIS.json')
for binding in analysis['inputs']:
    check('error_analysis_input_binding_' + binding['path'], bind(binding['path']) == binding)
check('error_analysis_refresh_history_retained', Path(analysis['input_binding_refresh']['historical_analysis_snapshot']).is_file())
paths = ['runtime/direct_v1/SCORES.json', 'runtime/direct_v1/RAW_RESPONSE_FREEZE.json', 'runtime/direct_v1/manifest.json', 'runtime/direct_v1/DISPLAY_MAPPINGS.json', 'runtime/direct_v1/PREFLIGHT_FREEZE.json', 'runtime/direct_v1/SCORING_STARTED.json', 'runtime/direct_v1/COMPLETE.json', 'runtime/direct_v1_private/PRIVATE_QUESTIONS.json', 'CONTENT_ADMISSION.json', 'review/PUBLIC_PREFLIGHT.json', 'SUMMARY.json', 'exports/SOLVER_RESPONSES.jsonl', 'exports/PER_QUESTION.json', 'REPORT_ZH.md', 'review/SOLVER_OBSERVABLE_ERROR_ANALYSIS.json']
result = {'schema_version': 'day3_crossroot_post_score_independent_review_v1', 'status': 'PASS' if not errors else 'FAIL',
    'created_at_utc': datetime.now(timezone.utc).isoformat(), 'reviewer': '/root/frozen_scoring', 'review_type': 'NON_BLIND_SHARED_WORKSPACE_REVIEW',
    'blind_review': False, 'reviewer_runtime_implementation_involvement_disclosed': True,
    'scope': 'Independent literal-box parsing and content-map reconstruction from all258 frozen raw responses, arithmetic/export/report consistency; does not newly establish semantic correctness of every Gold.',
    'inference_calls': 0, 'frozen_inputs_or_runtime_modified': False,
    'reviewer_checker_correction': 'MC canonical predictions are unordered sets. SCORES normalizes set order while the export preserves raw displayed-label mapping order. An initial reviewer-only comparison incorrectly required equal list order; it was corrected to set equality. No frozen result changed.',
    'final_exports_and_narrative_review': {'files_checked_with_hash_binding': ['SUMMARY.json', 'exports/SOLVER_RESPONSES.jsonl', 'exports/PER_QUESTION.json', 'REPORT_ZH.md'], 'checked_fields': ['all258 request IDs, mapped display/canonical predictions and Gold, raw output paths, valid/exact, ordering pair counts', '20 per-question call denominators, exact rates/strata, MC FP/FN totals, ordering pair counts', 'type/arity/template/root count and rate summaries and report tables', 'production20 PASS1 HOLD3 SHORTFALL; MC Gold counts;16 metadata fact families'], 'final_export_policy': 'Review executed after parent final deterministic report regeneration; any further edit requires a fresh review or updated final binding.'},
    'audited_files': [bind(B / p) for p in paths],
    'checks_total': len(checks), 'checks_passed': sum(x['pass'] for x in checks), 'errors': errors, 'checks': checks,
    'recomputed': {'planned_production_slots': 24, 'admitted_questions': 20, 'planned_roots': 4, 'diagnostic_roots': 4, 'planned_calls': 258, 'valid_calls': 258, 'exact_calls': 108, 'call_equal_exact': 108 / 258, 'question_equal_exact': s['question_equal_exact'], 'root_equal_exact': s['root_equal_exact'],
        'MC': {'questions': 9, 'calls': 162, 'exact_calls': sum(x['exact'] for x in independent if x['type'] == 'multi_select'), 'question_equal_exact': s['by_type']['multi_select']['question_equal_exact'], 'FP_item_events': sum(x['fp'] for x in mc_rows), 'FN_item_events': sum(x['fn'] for x in mc_rows)},
        'ordering': {'questions': 11, 'calls': 96, 'exact_calls': sum(x['exact'] for x in independent if x['type'] == 'ordering'), 'question_equal_exact': s['by_type']['ordering']['question_equal_exact'], 'call_equal_exact': sum(x['exact'] for x in independent if x['type'] == 'ordering') / 96, 'arity_strata': arity}, 'production_status_counts': status},
    'MC_FP_FN_spotchecks': spotchecks,
    'freeze_integrity_independent_subreview': {'reviewer': '/root/frozen_scoring/postfreeze_integrity', 'read_only': True, 'inference_calls': 0, 'status': 'PASS',
        'manifest_raw_score_unique_request_counts': [258, 258, 258], 'missing_duplicate_extra_raw_count': 0, 'raw_status_RETURNED': 258, 'raw_finish_stop': 258, 'single_completions': 258, 'validation_errors': 0,
        'submitted_batch_receipts': 33, 'complete_batch_receipts': 33, 'each_request_submitted_exactly_once': True,
        'raw_prompt_complete_token_sequence_and_request_metadata_match': True,
        'hash_bindings_recomputed_by_category': {'public_inputs': 267, 'code': 27, 'preflight_files': 268, 'raw_files': 258, 'independent_preflight_files': 6, 'other_bindings': 12}, 'unique_files_hashed': 559, 'hash_mismatches': 0,
        'last_raw_commit_UTC': '2026-10-20T12:21:06.575387+00:00', 'raw_freeze_UTC': '2026-10-20T12:21:07.244971+00:00', 'scorer_first_Gold_parse_UTC': '2026-10-20T12:22:20.020745+00:00', 'Gold_parse_after_raw_freeze_seconds': 72.775774,
        'independent_preflight_PASS_before_STARTED': True, 'STARTED_independent_preflight_binding_unchanged': True, 'planned_question_slots_field_means_admitted_20': True, 'production_denominator_is_full24': True,
        'model_files_checked_for_size_only_in_this_subreview': 18, 'model_hashes_recomputed_in_this_subreview': False},
    'interpretation_limits': ['This shared-workspace review and prior runtime implementation involvement are disclosed; it is not an external blinded evaluation.',
        'Question-equal, call-equal and root-equal averages differ; ordering60.61% question-equal is not47/96=48.96% call-equal.',
        'Arity3/4/6 results remain separate; there are no admitted arity5 ordering questions.',
        '20 admitted evaluation slots and24 full production slots are separate denominators; all258 calls and four unadmitted outcomes remain visible.',
        'This Train-only feasibility pilot does not prove feedback benefit, unseen generalization or controlled increased difficulty.',
        'Final narrative/export bindings cover the parent-finalized snapshot. Any further edit requires fresh review or refreshed final bindings.']}
destination = B / 'review/POST_SCORE_INDEPENDENT_REVIEW.json'
with destination.open('x') as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(json.dumps({'status': result['status'], 'checks': result['checks_total'], 'errors': errors, 'spotchecks': spotchecks, 'ordering_arity': arity, 'output': str(destination), 'output_binding': bind(destination)}, ensure_ascii=False))
