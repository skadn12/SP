"""Check completed v115 delivery only after parent authorizes final review.

No producer, controller, evaluation, or training function is imported/called.
This script exclusively creates DELIVERY_REVIEW.json. It never edits reports.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import itertools
import json
import re

BATCH = Path(__file__).resolve().parent.parent
ERRORS = []
BINDINGS = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    BINDINGS[str(path)] = sha(path)
    return json.loads(path.read_text())


def read_text(path):
    path = Path(path)
    BINDINGS[str(path)] = sha(path)
    return path.read_text()


def check(condition, where):
    if not condition:
        ERRORS.append(where)


def same(actual, expected, where):
    check(actual == expected, where)


def verify(items, where):
    for item in items:
        same(sha(item['path']), item['sha256'], where + ':' + item['path'])


def by_id(rows):
    out = {r['candidate_id']: r for r in rows}
    same(len(out), len(rows), 'unique_candidate_ids')
    return out


def audit_bindings(name, field, expected_count=None):
    doc = read(BATCH / name)
    check(doc.get('status', 'PASS') == 'PASS', name + ':status')
    check(not doc.get('errors', []) and not doc.get('issues', []), name + ':no_errors')
    verify(doc[field], name)
    if expected_count is not None:
        same(len(doc[field]), expected_count, name + ':binding_count')
    return doc


def raw_side(side, pool, final, behavior):
    run = BATCH / 'evaluation/run' / side
    manifest = read(run / 'manifest.json')
    complete = read(run / 'COMPLETE.json')
    wrapper = read(BATCH / 'evaluation' / (side.upper() + '_COMPLETE.json'))
    native = read(run / 'RESULTS.json')
    gold = read(run / 'gold.json')
    canonical = by_id(read(run / 'canonical_bindings.json')['candidates'])
    for cid, q in pool.items():
        field = 'options' if q['question_type'] == 'multi_select' else 'items'
        ids = {label: 'content_' + hashlib.sha256(text.strip().encode('utf-8')).hexdigest()[:16] for label, text in q[field].items()}
        same(gold[cid], [ids[label] for label in q['gold']], side + ':pool_label_to_canonical_Gold:' + cid)
        same(canonical[cid]['content_id_by_label'], ids, side + ':native_label_content_binding:' + cid)
        evidence_field = 'option_evidence' if field == 'options' else 'item_evidence'
        same(canonical[cid]['evidence_by_content_id'], {ids[label]: value for label, value in q[evidence_field].items()}, side + ':native_complete_evidence_binding:' + cid)
    same(manifest['request_count'], 456, side + ':request_count')
    same(manifest['candidate_count'], 28, side + ':candidate_count')
    same(complete['status'], 'COMPLETE', side + ':complete')
    same(complete['actual_call_count'], 456, side + ':actual_calls')
    same(wrapper['actual_calls'], 456, side + ':wrapper_calls')
    same(wrapper['training_started'], False, side + ':no_training')
    same(native['call_set_errors'], [], side + ':call_set_errors')
    requests = {r['request_id']: r for r in manifest['requests']}
    same(len(requests), 456, side + ':unique_requests')
    metrics = defaultdict(lambda: {'correct': 0, 'calls': 0, 'cids': set()})
    perroot = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'calls': 0}))
    scores = defaultdict(lambda: defaultdict(list))
    exacts = defaultdict(list)
    behaviors = []
    seen = set()
    for path in sorted((run / 'responses').glob('*.json')):
        row = read(path)
        request_id = row['request_id']
        check(request_id not in seen and request_id in requests, side + ':response_request:' + request_id)
        seen.add(request_id)
        request = requests[request_id]
        cid = row['candidate_id']
        q = pool[cid]
        for key in ('candidate_id', 'root_id', 'question_type', 'seed', 'slot'):
            same(row[key], request[key], side + ':response_identity:' + key + ':' + request_id)
        same(row['variant_id'], request['variant']['variant_id'], side + ':variant:' + request_id)
        check(row['evaluation_valid'] is True, side + ':evaluation_valid:' + request_id)
        match = re.fullmatch(r'\s*<box>\s*([^<>]*)\s*</box>\s*', row['raw_output'])
        if not match:
            ERRORS.append(side + ':raw_format:' + request_id)
            continue
        labels = [x.strip().upper() for x in match.group(1).split(',') if x.strip()]
        mapping = request['variant']['display_to_content']
        same(len(labels), len(set(labels)), side + ':unique_prediction_labels:' + request_id)
        check(all(label in mapping for label in labels), side + ':valid_prediction_labels:' + request_id)
        prediction = [mapping[label] for label in labels]
        expected = gold[cid]
        field = 'options' if q['question_type'] == 'multi_select' else 'items'
        same(request['variant']['content_to_text'], {'content_' + hashlib.sha256(value.strip().encode('utf-8')).hexdigest()[:16]: value for value in q[field].values()}, side + ':variant_content_text:' + request_id)
        if q['question_type'] == 'multi_select':
            fp = len(set(prediction) - set(expected))
            fn = len(set(expected) - set(prediction))
            score = Fraction(len(set(prediction) & set(expected)), len(expected))
            if len(expected) < 6:
                score -= Fraction(fp, 6 - len(expected))
            exact = set(prediction) == set(expected)
            names = ['gold6_control'] if len(expected) == 6 else ['multi_select_gold2_5', 'gold' + str(len(expected))]
            kind = 'extra_and_missing' if fp and fn else 'extra_only' if fp else 'missing_only' if fn else 'exact'
            behaviors.append({'request_id': request_id, 'candidate_id': cid, 'gold_count': len(expected),
                              'selected_count': len(set(prediction)), 'false_positive_count': fp,
                              'false_negative_count': fn, 'error_kind': kind})
        else:
            same(set(prediction), set(expected), side + ':ordering_labels:' + request_id)
            same(len(prediction), 6, side + ':ordering_size:' + request_id)
            ranks = {label: i for i, label in enumerate(prediction)}
            wrong = sum(ranks[a] > ranks[b] for a, b in itertools.combinations(expected, 2))
            score = Fraction(15 - 2 * wrong, 15)
            exact = prediction == expected
            names = ['ordering']
        same(Fraction(row['score']['partial_score_fraction']), score, side + ':raw_partial:' + request_id)
        same(row['score']['exact_match'], exact, side + ':raw_exact:' + request_id)
        scores[cid][row['variant_id']].append(score)
        exacts[cid].append(exact)
        for name in names:
            metrics[name]['calls'] += 1
            metrics[name]['correct'] += exact
            metrics[name]['cids'].add(cid)
            perroot[q['root_id']][name]['calls'] += 1
            perroot[q['root_id']][name]['correct'] += exact
    same(seen, set(requests), side + ':complete_response_coverage')
    same(len(seen), 456, side + ':response_count')
    comparisons = {}
    for name, value in metrics.items():
        actual = {'calls': value['calls'], 'correct': value['correct'], 'accuracy': value['correct'] / value['calls'], 'question_count': len(value['cids'])}
        same(final['solver'][side]['by_type'][name], actual, side + ':final_metric:' + name)
        comparisons[name] = actual
    for rid, groups in perroot.items():
        for name, v in groups.items():
            actual = {**v, 'accuracy': v['correct'] / v['calls']}
            same(final['solver'][side]['by_root'][rid][name], actual, side + ':final_root_metric:' + rid + ':' + name)
    same(final['solver'][side]['valid_calls'], 456, side + ':final_valid_calls')
    native_questions = by_id(native['questions'])
    same(set(native_questions), set(pool), side + ':native_candidate_coverage')
    same(final['solver'][side]['questions'], native['questions'], side + ':final_native_question_records')
    selected = set()
    for cid, groups in scores.items():
        ms = pool[cid]['question_type'] == 'multi_select'
        same(len(groups), 6 if ms else 12, side + ':display_count:' + cid)
        check(all(len(v) == (3 if ms else 1) for v in groups.values()), side + ':answers_per_display:' + cid)
        mean = sum((sum(v, Fraction()) / len(v) for v in groups.values()), Fraction()) / len(groups)
        exact_mean = Fraction(sum(exacts[cid]), len(exacts[cid]))
        r = native_questions[cid]
        same(Fraction(r['full_partial_mean']), mean, side + ':native_partial_mean:' + cid)
        same(Fraction(r['full_exact_rate']), exact_mean, side + ':native_exact_rate:' + cid)
        same(r['selected_by_policy'], 0 < mean < 1, side + ':native_admission:' + cid)
        if 0 < mean < 1 and pool[cid]['role'] != 'control6':
            selected.add(cid)
    recorded_behavior = behavior['sides'][side]
    same(recorded_behavior['responses'], behaviors, side + ':behavior_all_raw_rows')
    behavior_groups = {}
    for name, subset in [('gold2_5', [r for r in behaviors if r['gold_count'] < 6]),
                         ('gold6_control', [r for r in behaviors if r['gold_count'] == 6]),
                         *[('gold' + str(k), [r for r in behaviors if r['gold_count'] == k]) for k in (2, 3, 4, 5)]]:
        v = {'calls': len(subset), 'error_kinds': dict(Counter(r['error_kind'] for r in subset)),
             'all_six_selected': sum(r['selected_count'] == 6 for r in subset),
             'more_than_gold_selected': sum(r['selected_count'] > r['gold_count'] for r in subset),
             'any_false_positive': sum(r['false_positive_count'] > 0 for r in subset),
             'any_false_negative': sum(r['false_negative_count'] > 0 for r in subset),
             'total_false_positives': sum(r['false_positive_count'] for r in subset),
             'total_false_negatives': sum(r['false_negative_count'] for r in subset)}
        same(recorded_behavior['groups'][name], v, side + ':behavior_group:' + name)
        behavior_groups[name] = v
    return {'calls': len(seen), 'metrics': comparisons, 'behavior_groups': behavior_groups, 'selected_new_ids': sorted(selected)}


def report_disclosures(report, interpretation):
    text = report + '\n' + interpretation
    tests = {
        '31457_serious_inherited_baseline_conflict': r'(?s)31457.{0,500}(?:去世|已故|死亡|deceased|James.s body).{0,500}(?:矛盾|冲突|conflict)',
        'trained_roots_previously_exposed': r'(?:已暴露训练|已有训练曝光|训练.*暴露|训练.*曝光)',
        'family_and_repeated_calls_correlated': r'(?s)(?:family|事实族|同源|同一root|同一 root).{0,400}(?:相关|独立)',
        'role_separation_not_external_certification': r'(?s)(?:角色分离|作者与审核|同一.*模型).{0,250}(?:外部.*认证|外部.*审核)',
        'path_adapter_content_unchanged': r'(?s)(?:路径|source_path).{0,350}(?:不改|未改|不变|相同|无损)',
        'no_training_or_merge': r'(?s)(?:未合并|没有合并|不合并).{0,150}(?:未启动训练|没有训练|未训练)',
        'tokens_not_actual_usage': r'(?:文本 tokens 不是实际调用消耗|文本.*不是.*usage|tokens.*非.*usage|tokens.*不.*账单)',
    }
    out = {}
    for name, pattern in tests.items():
        match = re.search(pattern, text, re.I)
        check(match is not None, 'report_disclosure:' + name)
        out[name] = {'present': match is not None, 'excerpt': match.group(0) if match else None}
    check('INTERPRETATION' not in interpretation or len(interpretation) > 200, 'interpretation_not_empty_placeholder')
    return out


def check_author_archives():
    manifest = read(BATCH / 'evidence/authoring_45849_55788/ARCHIVE_MANIFEST.json')
    same(manifest['file_count'], 31, 'author_archive:31_actual_files')
    same(len(manifest['files']), 31, 'author_archive:file_count_matches')
    for item in manifest['files']:
        same(sha(item['archive_path']), item['archive_sha256'], 'author_archive:bytecopy:' + item['archive_path'])
        same(item['original_sha256'], item['archive_sha256'], 'author_archive:original_copy_hash')
    verify(manifest['root_artifact_bindings'], 'author_archive:root_artifacts')
    return {'archived_actual_files': 31, 'bound_original_root_artifacts': len(manifest['root_artifact_bindings'])}


def main():
    required = ['FINAL_STATUS.json', 'REPORT_ZH.md', 'INTERPRETATION_ZH.md', 'BEHAVIOR_DIAGNOSTICS.json', 'SELECTED_CANDIDATES.json', 'INDEPENDENT_SOLVER_AUDIT.json']
    missing = [name for name in required if not (BATCH / name).exists()]
    if missing:
        raise SystemExit('WAIT_FOR_PARENT_COMPLETED_DELIVERY: ' + ','.join(missing))
    started = datetime.now(timezone.utc).isoformat()
    final = read(BATCH / 'FINAL_STATUS.json')
    summary = read(BATCH / 'CONTENT_SUMMARY.json')
    same(final['content'], summary, 'final_content_summary_verbatim')
    for key, value in {'attempted_roots': 4, 'complete_A3B3_roots': 4, 'expected_scene_count': 24,
                       'attempted_scene_count': 24, 'ready_scene_count': 24, 'attempted_new_turns': 330,
                       'accepted_new_turns': 330, 'new_questions': 24, 'gold6_controls': 4,
                       'excluded_public_questions': 0}.items():
        same(summary[key], value, 'content:' + key)
    same(summary['content_pass_type_counts'], {'multi_select': 20, 'ordering': 8}, 'content:type_counts')
    same(summary['multi_gold_counts'], {str(k): 4 for k in (2, 3, 4, 5, 6)}, 'content:gold_distribution')
    same(final['status'], 'COMPLETE_SERIALIZED_A3B3_VALIDATION_NO_TRAINING', 'final_status')
    same(final['training_started'], False, 'final_no_training')
    same(final['external_semantic_certification'], False, 'final_no_external_certification')
    frozen = read(BATCH / 'EXECUTION_INPUTS_FROZEN.json')
    tooling = read(BATCH / 'evidence/TOOLING_FREEZE.json')
    history = read(BATCH / 'evidence/HISTORICAL_PROTECTION.json')
    for name, doc, count in [('frozen32', frozen, 32), ('tooling11', tooling, 11), ('history7249', history, 7249)]:
        same(len(doc['files']), count, name + ':count')
        verify(doc['files'], name)
    same(final['historical_files_verified_unchanged'], 7249, 'final_history_count')
    path_freeze = read(BATCH / 'evidence/PATH_ADAPTER_TOOLING_FREEZE.json')
    verify(path_freeze['files'], 'path_addendum')
    preexec = read(BATCH / 'evidence/PREEXECUTION_CODE_BINDINGS.json')
    verify(preexec['files'], 'preexecution')
    source_audit = audit_bindings('SOURCE_PROVENANCE_AUDIT.json', 'input_bindings')
    evidence_audit = audit_bindings('EVIDENCE_EXPORT_AUDIT.json', 'checked_input_files')
    solver_audit = audit_bindings('INDEPENDENT_SOLVER_AUDIT.json', 'input_bindings')
    supplement = audit_bindings('RAW_AUDIT_SUPPLEMENT.json', 'input_bindings')
    same(supplement['raw_audit_sha256'], sha(BATCH / 'INDEPENDENT_SOLVER_AUDIT.json'), 'raw_supplement:audit_binding')
    same(supplement['raw_log_sha256'], sha(supplement['raw_log_path']), 'raw_supplement:log_binding')
    for row in supplement['bound_rows_checked']:
        same(sha(row['path']), row['expected_sha256'], 'raw_supplement:each_bound_row:' + row['path'])
    check(all(row['reported_and_saved_before_base_start'] for row in supplement['pre_solver_time_checks']), 'raw_supplement:pre_solver_chronology')
    behavior = audit_bindings('BEHAVIOR_DIAGNOSTICS.json', 'input_bindings')
    author_archive = check_author_archives()
    for k, v in {'root_denominator': 4, 'scene_denominator': 24, 'full_ready_root_count': 4, 'ready_scene_count': 24, 'total_new_turns': 330}.items():
        same(source_audit[k], v, 'source_audit:' + k)
    same(evidence_audit['candidate_count'], 28, 'evidence_audit:count')
    same(sha(BATCH / 'evidence/audit_evidence_export_independent.py'), evidence_audit['audit_code_sha256'], 'evidence_audit:code_binding')
    for side in ('base', 'trained'):
        same(solver_audit['results'][side], {'calls': 456, 'candidates': 28, 'errors': 0}, 'solver_audit:' + side)
    adapter = read(BATCH / 'evaluation/ADAPTER_BINDINGS.json')
    verify(adapter['files'], 'adapter_bindings')
    paired = read(BATCH / 'evaluation/PAIRED_PREFLIGHT.json')
    verify(paired['files'], 'paired_preflight')
    verify([paired['adapter_binding']], 'paired_adapter')
    same(paired['status'], 'PASS', 'paired_status')
    same(paired['calls_per_model'], 456, 'paired_calls')
    pool = by_id(read(BATCH / 'evaluation/EVALUATION_POOL.json')['questions'])
    same(len(pool), 28, 'pool_count')
    for name in ('questions.json', 'gold.json'):
        same(read(BATCH / 'evaluation/run/base' / name), read(BATCH / 'evaluation/run/trained' / name), 'paired:' + name)
    sides = {side: raw_side(side, pool, final, behavior) for side in ('base', 'trained')}
    selected = read(BATCH / 'SELECTED_CANDIDATES.json')
    selected_ids = set(by_id(selected['questions']))
    same(selected_ids, set(sides['trained']['selected_new_ids']), 'selection_exact_rule_and_control_exclusion')
    check(all(q['role'] != 'control6' for q in selected['questions']), 'selection_no_Gold6')
    same(final['selected_new_question_count'], len(selected_ids), 'final_selected_count')
    same(final['selected_type_counts'], dict(Counter(q['question_type'] for q in selected['questions'])), 'final_selected_types')
    same(selected['training_started'], False, 'selected_no_training')
    same(behavior['training_started'], False, 'behavior_no_training')
    same(behavior['pool_merge_performed'], False, 'behavior_no_merge')
    for q in selected['questions']:
        same(q, pool[q['candidate_id']], 'selected_full_question_verbatim:' + q['candidate_id'])
    # Recompute the reported text lengths using the same explicitly named
    # local tokenizer, independently of the delivery producer.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('/root/autodl-tmp/Qwen3-4B', local_files_only=True)
    lengths = []
    for root in summary['roots']:
        rid = root['root_id']
        folder = BATCH / 'roots' / ('root' + rid.rsplit('-', 1)[1])
        turns = read(folder / 'CUMULATIVE_SOURCE.json')['turns']
        new_turns = [t for t in turns if t.get('scene_ref', '').startswith('R02:')]
        plain = '\n'.join(t['speaker'] + ': ' + t['text'] for t in turns)
        refs = '\n'.join('[' + t['turn_ref'] + '] ' + t['speaker'] + ': ' + t['text'] for t in turns)
        new = '\n'.join(t['speaker'] + ': ' + t['text'] for t in new_turns)
        lengths.append({'root_id': rid, 'turns': len(turns), 'dialogue_tokens_with_refs_and_speakers': len(tok.encode(refs, add_special_tokens=False)), 'dialogue_tokens_speakers_and_body': len(tok.encode(plain, add_special_tokens=False)), 'new_turns': len(new_turns), 'new_dialogue_tokens_speakers_and_body': len(tok.encode(new, add_special_tokens=False))})
        check((folder / 'FULL_DIALOGUE.md').is_file() and (folder / 'QUESTIONS_AND_ANSWERS.md').is_file(), rid + ':human_readable_artifacts')
        same(len(list((folder / 'scenes').glob('*/EXPANSION_DRAFT.json'))), 6, rid + ':six_drafts')
        for path in (folder / 'scenes').glob('*/SOURCE_REVIEW.json'):
            review = read(path)
            same(review['decision'], 'READY', rid + ':scene_READY:' + path.parent.name)
    same(final['dialogue_lengths'], lengths, 'all_dialogue_tokens_recomputed')
    same(final['mean_dialogue_tokens_speakers_and_body'], sum(r['dialogue_tokens_speakers_and_body'] for r in lengths) / 4, 'mean_dialogue_tokens')
    same(sum(r['new_turns'] for r in lengths), 330, 'source_new_turns')
    # Failure/adaptation records remain visible and exactly bound.
    path_receipt = read(BATCH / 'evaluation/PREPARE_PATH_ADAPTER_RECEIPT.json')
    verify(path_receipt['input_bindings'], 'path_adapter_receipt')
    failure = read(BATCH / 'evidence/PREPARE_V1_FAILURE_RECORD.json')
    same(failure['exit_code'], 1, 'prepare_failure_preserved')
    same(failure['solver_calls'], 0, 'prepare_failure_no_calls')
    same(path_receipt['source_question_gold_evidence_scoring_changed'], False, 'path_adapter_no_content_change')
    adapter_receipt = read(BATCH / 'SOURCE_PROVENANCE_AUDIT_SCHEMA_ADAPTER_RECEIPT.json')
    original = BATCH / 'SOURCE_PROVENANCE_AUDIT_FIRST_SCHEMA.json'
    same(sha(original), source_audit['delivery_schema_adapter']['original_report_sha256'], 'source_audit_first_schema_retained')
    same(sha(adapter_receipt['old_path']), adapter_receipt['old_sha256'], 'source_audit_schema_receipt:old')
    same(sha(adapter_receipt['new_path']), adapter_receipt['new_sha256'], 'source_audit_schema_receipt:new')
    same(adapter_receipt['errors'], [], 'source_audit_schema_receipt:no_errors')
    diagnostic_failure = read(BATCH / 'evidence/BEHAVIOR_DIAGNOSTIC_V1_FAILURE.json')
    verify([diagnostic_failure['failed_script']], 'diagnostic_failure_original_script')
    same(diagnostic_failure['solver_calls_added'], 0, 'diagnostic_failure:no_extra_solver_calls')
    same(diagnostic_failure['diagnostic_output_saved'], False, 'diagnostic_failure:no_failed_output_promoted')
    report = read_text(BATCH / 'REPORT_ZH.md')
    interpretation = read_text(BATCH / 'INTERPRETATION_ZH.md')
    read_first = read_text(BATCH / 'READ_FIRST.md')
    disclosures = report_disclosures(report + '\n' + read_first, interpretation)
    for key, label in [('multi_select_gold2_5', '多选 Gold2–5'), ('gold2', 'Gold2'), ('gold3', 'Gold3'), ('gold4', 'Gold4'), ('gold5', 'Gold5'), ('ordering', '排序'), ('gold6_control', 'Gold6 对照')]:
        a, b = sides['base']['metrics'][key], sides['trained']['metrics'][key]
        check(f"|{label}|{a['correct']}/{a['calls']}|{b['correct']}/{b['calls']}|" in report, 'report_accuracy_table:' + key)
    out = {'status': 'PASS' if not ERRORS else 'FAIL', 'reviewer': '/root/v113_14304_recheck',
           'review_kind': 'completed_delivery_independent_mechanical_recomputation_and_disclosure_check',
           'started_at_utc': started, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
           'audit_code_sha256': sha(__file__), 'roots': 4, 'scenes': 24, 'new_turns': 330,
           'questions': 28, 'multi_gold_counts': summary['multi_gold_counts'], 'selected_new_count': len(selected_ids),
           'raw_recomputed_sides': sides, 'dialogue_lengths_recomputed': lengths,
           'verified_hash_counts': {'execution_inputs': 32, 'original_tooling': 11, 'historical_files': 7249, 'path_addendum': len(path_freeze['files'])},
           'actual_author_files_archive': author_archive,
           'required_disclosures': disclosures, 'all_existing_reports_unmodified_by_reviewer': True,
           'new_training_or_merge_run_by_reviewer': False, 'root_files_modified_by_reviewer': False,
           'semantic_recertification': False, 'same_inherited_model_external_certification': False,
           'input_bindings': [{'path': p, 'sha256': digest} for p, digest in BINDINGS.items()], 'errors': ERRORS}
    with (BATCH / 'DELIVERY_REVIEW.json').open('x') as handle:
        json.dump(out, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps({k: out[k] for k in ('status', 'roots', 'scenes', 'new_turns', 'questions', 'selected_new_count', 'errors')}))
    if ERRORS:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
