"""Check completed v117 delivery only after parent authorizes final review.

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
PAIRED_REQUEST_BINDINGS = {}


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
    expected_calls = sum(18 if q['question_type'] == 'multi_select' else 12 for q in pool.values())
    run = BATCH / 'evaluation/run' / side
    manifest = read(run / 'manifest.json')
    complete = read(run / 'COMPLETE.json')
    wrapper = read(BATCH / 'evaluation' / (side.upper() + '_COMPLETE.json'))
    native = read(run / 'RESULTS.json')
    gold = read(run / 'gold.json')
    canonical = by_id(read(run / 'canonical_bindings.json')['candidates'])
    for relative, digest in manifest['input_sha256'].items():
        path = Path(manifest['pool_path']) if relative == 'pool' else run / relative
        same(sha(path), digest, side + ':manifest_input_hash:' + relative)
    for path, digest in manifest['code_sha256'].items():
        same(sha(path), digest, side + ':native_code_hash:' + path)
    dialogue_by_candidate = {}
    for cid, q in pool.items():
        source = read(q['source_path'])
        same(sha(q['source_path']), q['source_sha256'], side + ':pool_source_hash:' + cid)
        dialogue_by_candidate[cid] = '\n'.join('[' + t['turn_ref'] + '] ' + t['speaker'] + ': ' + t['text'] for t in source['turns'])
        field = 'options' if q['question_type'] == 'multi_select' else 'items'
        ids = {label: 'content_' + hashlib.sha256(text.strip().encode('utf-8')).hexdigest()[:16] for label, text in q[field].items()}
        same(gold[cid], [ids[label] for label in q['gold']], side + ':pool_label_to_canonical_Gold:' + cid)
        same(canonical[cid]['content_id_by_label'], ids, side + ':native_label_content_binding:' + cid)
        evidence_field = 'option_evidence' if field == 'options' else 'item_evidence'
        same(canonical[cid]['evidence_by_content_id'], {ids[label]: value for label, value in q[evidence_field].items()}, side + ':native_complete_evidence_binding:' + cid)
    same(manifest['request_count'], expected_calls, side + ':request_count')
    same(manifest['candidate_count'], len(pool), side + ':candidate_count')
    same(complete['status'], 'COMPLETE', side + ':complete')
    same(complete['actual_call_count'], expected_calls, side + ':actual_calls')
    same(wrapper['actual_calls'], expected_calls, side + ':wrapper_calls')
    same(wrapper['training_started'], False, side + ':no_training')
    same(native['call_set_errors'], [], side + ':call_set_errors')
    requests = {r['request_id']: r for r in manifest['requests']}
    same(len(requests), expected_calls, side + ':unique_requests')
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
        prompt_path = run / 'prompts' / (request_id + '.txt')
        prompt = read_text(prompt_path)
        prompt_hash = sha(prompt_path)
        same(prompt_hash, request['prompt_sha256'], side + ':actual_prompt_hash:' + request_id)
        check(dialogue_by_candidate[cid] in prompt, side + ':prompt_full_ordered_dialogue:' + request_id)
        check(q['question'] in prompt, side + ':prompt_public_stem:' + request_id)
        displayed = request['variant']['options' if q['question_type'] == 'multi_select' else 'items']
        for label, proposition in displayed.items():
            check(label + '. ' + proposition in prompt, side + ':prompt_displayed_option:' + request_id + ':' + label)
        paired_binding = {'prompt_sha256': prompt_hash, 'seed': request['seed'], 'variant': request['variant'],
                          'candidate_id': cid, 'root_id': request['root_id'], 'question_type': request['question_type'],
                          'mode': request['mode'], 'slot': request['slot']}
        if side == 'base':
            PAIRED_REQUEST_BINDINGS[request_id] = paired_binding
        else:
            same(paired_binding, PAIRED_REQUEST_BINDINGS.get(request_id), 'paired_actual_prompt_and_request:' + request_id)
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
    same(len(seen), expected_calls, side + ':response_count')
    comparisons = {}
    for name, value in metrics.items():
        actual = {'calls': value['calls'], 'correct': value['correct'], 'accuracy': value['correct'] / value['calls'], 'question_count': len(value['cids'])}
        same(final['solver'][side]['by_type'][name], actual, side + ':final_metric:' + name)
        comparisons[name] = actual
    for rid, groups in perroot.items():
        for name, v in groups.items():
            actual = {**v, 'accuracy': v['correct'] / v['calls']}
            same(final['solver'][side]['by_root'][rid][name], actual, side + ':final_root_metric:' + rid + ':' + name)
    same(final['solver'][side]['valid_calls'], expected_calls, side + ':final_valid_calls')
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


def report_disclosures(report, interpretation, limitations, source_audit):
    text = report + '\n' + interpretation + '\n' + limitations
    tests = {
        'screened_baselines_and_inherited_warnings': r'(?s)v116.{0,600}(?:WARNING|继承.*约束|基线.*筛)',
        'trained_roots_previously_exposed': r'(?:已暴露训练|已有训练曝光|训练.*暴露|训练.*曝光)',
        'family_and_repeated_calls_correlated': r'(?s)(?:family|事实族|同源|同一root|同一 root).{0,400}(?:相关|独立)',
        'role_separation_not_external_certification': r'(?s)(?:角色分离|作者与审核|同一.*模型).{0,250}(?:外部.*认证|外部.*审核)',
        'path_adapter_content_unchanged': r'(?s)(?:路径|source_path).{0,350}(?:不改|未改|不变|相同|无损)',
        'path_adapter_is_prospective': r'(?s)(?:路径|source_path|prospective).{0,400}(?:事前|首次.*验证|prospective|无.*失败|未.*失败)',
        'no_training_or_merge': r'(?s)(?:(?:未合并|没有合并|不合并).{0,150}(?:未启动训练|没有训练|未训练)|(?:没有启动新训练|未启动新训练|未启动训练).{0,150}(?:未合并|没有合并|不合并))',
        'tokens_not_actual_usage': r'(?:文本 tokens 不是实际调用消耗|文本.*不是.*usage|tokens.*非.*usage|tokens.*不.*账单)',
        'late_archive_not_contemporaneous': r'(?s)(?:LATE_ARCHIVE|后来.*归档|补充.*归档).{0,600}(?:未保存|未覆盖|未.*原|不完全统一|非统一)',
        'author_created_read_context_limitation': r'(?s)68549.{0,500}(?:误用|误.*审核者|作者.*创建).{0,500}(?:不能|不适合|不能当作|不作)',
        'extra_context_wrong_identity_not_evidence': r'(?s)75223.{0,500}(?:错误|误用).{0,500}(?:不作|不适合|不能)',
        'cross_review_assignment_change_disclosed': r'(?s)75223.{0,500}A2.{0,250}(?:调整|分工).{0,250}(?:14304|A3)',
        'constraints_receipt_field_variation_disclosed': r'(?s)68549.{0,350}read_files.{0,350}(?:字段|差异).{0,150}(?:未补|保留)',
        'main_author_failed_self_review_disclosed': r'(?s)(?:主代理|main).{0,300}(?:57679|A2).{0,250}(?:拒绝|断言).{0,250}(?:空日志|empty|未删除|没有删除)',
    }
    if source_audit.get('warnings'):
        tests['recorded_procedure_limitations_disclosed'] = r'(?s)(?:收据|read.context|读取上下文|读取记录).{0,700}(?:限制|缺失|误用|误启动|补充|不适合|不均一|非统一)'
    out = {}
    for name, pattern in tests.items():
        match = re.search(pattern, text, re.I)
        check(match is not None, 'report_disclosure:' + name)
        out[name] = {'present': match is not None, 'excerpt': match.group(0) if match else None}
    check(len(interpretation.strip()) > 200, 'interpretation_substantive')
    return out


def archive_and_procedure_supplements():
    archive = read(BATCH / 'evidence/main_author_actual_files/ARCHIVE_MANIFEST.json')
    same(archive['status'], 'LATE_ARCHIVE_OF_ACTUAL_EXISTING_FILES', 'actual_files_archive:late_status')
    same(archive['originals_modified'], False, 'actual_files_archive:originals_preserved')
    same(archive['receipts_modified'], False, 'actual_files_archive:receipts_preserved')
    verify(archive['files'], 'actual_files_archive:archived_hashes')
    originals_available = 0
    for row in archive['files']:
        if Path(row['original_path']).exists():
            originals_available += 1
            same(sha(row['original_path']), row['sha256'], 'actual_files_archive:original_byte_identity:' + row['original_path'])
    payload = read(BATCH / 'evidence/MAIN_PAYLOAD_ARCHIVE_RECEIPT.json')
    same(payload['status'], 'LATE_ARCHIVE_EXISTING_TMP_PAYLOADS', 'payload_archive:honest_late_status')
    same(payload['semantic_content_modified'], False, 'payload_archive:semantic_content_preserved')
    same(payload['earlier_receipts_modified'], False, 'payload_archive:earlier_receipts_preserved')
    same({r['scene'] for r in payload['rows']}, {'A_S02', 'A_S03', 'B_S01'}, 'payload_archive:actual_scene_coverage')
    for row in payload['rows']:
        same(row['archived_after_materialization'], True, 'payload_archive:honest_timing:' + row['scene'])
        same(sha(row['archived_path']), row['sha256'], 'payload_archive:archive_hash:' + row['scene'])
        if Path(row['original_path']).exists():
            same(sha(row['original_path']), row['sha256'], 'payload_archive:original_byte_identity:' + row['scene'])
    check('no separate original payload' in payload['A_S01'] and 'No replacement payload' in payload['A_S01'], 'payload_archive:no_constructed_A1_claim')
    for name in ('AUTHOR_PAYLOAD.json', 'AUTHOR_PAYLOAD_LATE_ARCHIVE.json'):
        check(not (BATCH / 'roots/root57679/scenes/A_S01' / name).exists(), 'payload_archive:no_fabricated_A1:' + name)
    supplement = read(BATCH / 'evidence/ROOT68549_A2_READ_PROVENANCE_SUPPLEMENT.json')
    verify([supplement['original_review_binding']], '68549_A2:immutable_original_review')
    same(supplement['immutable_originals_preserved'], True, '68549_A2:originals_preserved')
    reread = supplement['supplemental_independent_reread']; source = read(reread['source_path'])
    same(sha(reread['source_path']), reread['source_sha256'], '68549_A2:reread_source_hash')
    same(sha(reread['constraints_path']), reread['constraints_sha256'], '68549_A2:reread_constraints_hash')
    same(reread['turn_refs_read'], [t['turn_ref'] for t in source['turns']], '68549_A2:reread_exact_coverage')
    same(reread['all_constraints_read'], read(reread['constraints_path'])['continuation_constraints'], '68549_A2:reread_all_constraints')
    same(reread['decision_reaffirmed'], 'READY', '68549_A2:decision_reaffirmed')
    same(reread['full_source_read'], True, '68549_A2:full_reread_recorded')
    check(reread['started_at_utc'] <= reread['finished_at_utc'], '68549_A2:actual_reread_interval')
    draft = read(BATCH / 'roots/root68549/scenes/A_S02/EXPANSION_DRAFT.json')
    check(reread['reader'] != draft['author'], '68549_A2:actual_non_author_rereader')
    identity = read(BATCH / 'evidence/ROOT75223_A1_EXTRA_READ_IDENTITY_CLARIFICATION.json')
    same(identity['independent_review_evidence'], False, '75223_A1:extra_context_not_evidence')
    check(identity['actual_caller'] != identity['incorrect_reviewer_argument'], '75223_A1:actual_identity_disclosed')
    check('Failed FileExistsError' in identity['save_result'], '75223_A1:failed_extra_save_disclosed')
    misuse = read(BATCH / 'evidence/MAIN_REVIEW_READ_MISUSE_SUPPLEMENT.json')
    same(misuse['no_semantic_draft_or_review_replacement'], True, 'main_review_misuse:no_replacement')
    for event in misuse['events']:
        if 'temporary_log' in event:
            p = BATCH / event['temporary_log']
            same(p.stat().st_size, event['temporary_log_bytes'], 'main_review_misuse:retained_empty_log')
            same(event['used_as_review_evidence'], False, 'main_review_misuse:not_evidence')
            same(event['source_review_written'], False, 'main_review_misuse:no_authored_review')
    return {'actual_file_archive_count': len(archive['files']), 'original_files_available_for_byte_comparison': originals_available,
            'late_payload_archive_count': len(payload['rows']), 'A1_payload_gap_preserved': True,
            '68549_A2_additive_independent_reread_verified': True, '75223_A1_extra_context_not_evidence': True,
            'main_failed_review_log_retained_not_evidence': True}


def content_from_artifacts(cfg, summary):
    scenes = cfg['scene_sequence']
    same(scenes, ['R02:' + route + ':S0' + str(i) for route in 'AB' for i in (1, 2, 3)], 'six_scene_sequence')
    roots = []; total_attempted = total_accepted = ready_count = attempted_count = 0
    public = []; private = []; excluded = []; full_roots = 0
    for configured in cfg['roots']:
        rid = configured['root_id']; folder = BATCH / 'roots' / ('root' + rid.rsplit('-', 1)[1])
        baseline = read(configured['baseline_path'])
        previous = baseline['turns']; local_ready = 0; local_attempted = local_accepted = 0
        for scene in scenes:
            sf = folder / 'scenes' / scene.removeprefix('R02:').replace(':', '_')
            draft = read(sf / 'EXPANSION_DRAFT.json'); source = read(sf / 'CUMULATIVE_SOURCE.json'); review = read(sf / 'SOURCE_REVIEW.json')
            attempted_count += 1; local_attempted += len(draft['turns'])
            check(12 <= len(draft['turns']) <= 16, rid + scene + ':twelve_to_sixteen_new_turns')
            same(source['turns'], previous + draft['turns'], rid + scene + ':exact_cumulative_prefix')
            same(review['source_sha256'], sha(sf / 'CUMULATIVE_SOURCE.json'), rid + scene + ':review_source_binding')
            same(review['turn_refs_read'], [t['turn_ref'] for t in source['turns']], rid + scene + ':recorded_review_coverage')
            same(review['decision'], 'READY', rid + scene + ':recorded_READY')
            check(review['reviewer'] != draft['author'], rid + scene + ':non_author_reviewer')
            if review['decision'] == 'READY':
                ready_count += 1; local_ready += 1; local_accepted += len(draft['turns'])
            previous = source['turns']
        final_source = read(folder / 'CUMULATIVE_SOURCE.json')
        same(final_source['turns'], previous, rid + ':final_source_turns')
        same(sha(folder / 'CUMULATIVE_SOURCE.json'), sha(folder / 'scenes/B_S03/CUMULATIVE_SOURCE.json'), rid + ':final_source_bytecopy')
        same(sha(folder / 'SOURCE_REVIEW.json'), sha(folder / 'scenes/B_S03/SOURCE_REVIEW.json'), rid + ':final_review_bytecopy')
        if local_ready == 6: full_roots += 1
        qs = read(folder / 'PUBLIC_QUESTIONS.json')['questions']; gs = by_id(read(folder / 'PRIVATE_GOLD.json')['questions']); xs = by_id(read(folder / 'FINAL_CROSSCHECK.json')['results'])
        same(set(gs), {q['candidate_id'] for q in qs}, rid + ':public_gold_coverage')
        for q in qs:
            cid = q['candidate_id']; g = gs[cid]; x = xs[cid]
            same(x['original_gold_unchanged'], True, cid + ':original_gold_unchanged')
            same(x['source_derived_answer'], g['gold'], cid + ':source_derived_original_gold')
            if x['status'] == 'CONTENT_PASS': public.append(q); private.append(g)
            else: excluded.append(cid)
        total_attempted += local_attempted; total_accepted += local_accepted
        roots.append({'root_id': rid, 'new_turns': local_accepted, 'ready_scenes': local_ready, 'final_turns': len(previous)})
    fields = {'attempted_roots': len(cfg['roots']), 'complete_A3B3_roots': full_roots, 'expected_scene_count': len(cfg['roots']) * 6,
              'attempted_scene_count': attempted_count, 'ready_scene_count': ready_count, 'attempted_new_turns': total_attempted,
              'accepted_new_turns': total_accepted, 'new_questions': sum(g['role'] != 'control6' for g in private),
              'gold6_controls': sum(g['role'] == 'control6' for g in private), 'excluded_public_questions': len(excluded),
              'content_pass_type_counts': dict(Counter(q['question_type'] for q in public)),
              'multi_gold_counts': dict(Counter(str(len(g['gold'])) for q, g in zip(public, private) if q['question_type'] == 'multi_select'))}
    for key, value in fields.items(): same(summary[key], value, 'artifact_content:' + key)
    same(read(BATCH / 'CONTENT_PUBLIC_QUESTIONS.json')['questions'], public, 'aggregate_public_verbatim')
    same(read(BATCH / 'CONTENT_PRIVATE_GOLD.json')['questions'], private, 'aggregate_gold_verbatim')
    return fields, roots, by_id(public), by_id(private)


def prospective_path_receipt(pool, public, private):
    receipt = read(BATCH / 'evaluation/PREPARE_PATH_ADAPTER_RECEIPT.json')
    same(receipt['mode'], 'PROSPECTIVE_BEFORE_FIRST_NATIVE_VALIDATION', 'path_receipt:prospective_mode')
    same(receipt['prior_failed_prepare_claimed'], False, 'path_receipt:no_fictitious_failed_prepare')
    same(receipt['solver_calls'], 0, 'path_receipt:before_solver_calls')
    same(receipt['source_question_gold_evidence_scoring_changed'], False, 'path_receipt:no_content_change')
    same(receipt['changed_candidate_fields'], ['source_path'], 'path_receipt:only_path_field')
    verify([receipt['adapter_binding']], 'path_receipt:adapter_code')
    changes = {r['candidate_id']: r for r in receipt['changes']}
    same(len(changes), len(pool), 'path_receipt:unique_changes')
    same(receipt['candidate_count'], len(pool), 'path_receipt:candidate_count')
    project = BATCH.parent.parent
    for cid, q in pool.items():
        row = changes[cid]; original = public[cid]
        source = Path(original['source_path']); resolved = (source if source.is_absolute() else project / source).resolve()
        same(row['before'], original['source_path'], cid + ':path_before')
        same(row['after'], str(resolved), cid + ':path_after')
        same(q['source_path'], str(resolved), cid + ':pool_absolute_path')
        same(sha(resolved), original['source_sha256'], cid + ':source_text_hash')
        same(row['source_sha256'], original['source_sha256'], cid + ':receipt_source_hash')
        same(row['representation_changed'], row['before'] != row['after'], cid + ':path_change_truthful')
        for k, v in original.items():
            if k != 'source_path': same(q[k], v, cid + ':public_field_unchanged:' + k)
        same(q['gold'], private[cid]['gold'], cid + ':gold_unchanged_through_adapter')
    for side in ('base', 'trained'):
        started = read(BATCH / 'evaluation' / (side.upper() + '_STARTED.json'))
        check(receipt['created_at_utc'] <= started['started_at_utc'], 'path_receipt:predates_' + side)
    return {'candidate_count': len(changes), 'mode': receipt['mode'], 'prior_failed_prepare_claimed': False}


def main():
    required = ['FINAL_STATUS.json', 'REPORT_ZH.md', 'INTERPRETATION_ZH.md', 'READ_FIRST.md', 'BEHAVIOR_DIAGNOSTICS.json',
                'SELECTED_CANDIDATES.json', 'INDEPENDENT_SOLVER_AUDIT.json', 'SOURCE_PROVENANCE_AUDIT.json', 'EVIDENCE_EXPORT_AUDIT.json',
                'PROCEDURE_LIMITATIONS_ZH.md']
    missing = [name for name in required if not (BATCH / name).exists()]
    if missing: raise SystemExit('WAIT_FOR_PARENT_COMPLETED_DELIVERY: ' + ','.join(missing))
    started = datetime.now(timezone.utc).isoformat()
    final = read(BATCH / 'FINAL_STATUS.json'); summary = read(BATCH / 'CONTENT_SUMMARY.json'); cfg = read(BATCH / 'RUN_CONFIG.json')
    same(final['content'], summary, 'final_content_summary_verbatim')
    same(len(cfg['roots']), 4, 'fixed_root_denominator'); same(len(cfg['scene_sequence']) * len(cfg['roots']), 24, 'fixed_scene_denominator')
    fields, roots, public, private = content_from_artifacts(cfg, summary)
    same(final['status'], 'COMPLETE_SERIALIZED_A3B3_VALIDATION_NO_TRAINING', 'final_status')
    same(final['training_started'], False, 'final_no_training'); same(final['external_semantic_certification'], False, 'final_no_external_certification')
    counts = {}
    for key, name, expected in [('execution_inputs', 'EXECUTION_INPUTS_FROZEN.json', 36),
                                ('tooling', 'evidence/TOOLING_FREEZE.json', 6),
                                ('historical_files', 'evidence/HISTORICAL_PROTECTION.json', 9624),
                                ('preexecution_code', 'evidence/PREEXECUTION_CODE_BINDINGS.json', 4)]:
        doc = read(BATCH / name); same(len(doc['files']), expected, key + ':actual_frozen_binding_count'); verify(doc['files'], key)
        for row in doc['files']:
            if 'snapshot_path' in row:
                same(sha(row['snapshot_path']), row['snapshot_sha256'], key + ':snapshot_binding')
                same(row['snapshot_sha256'], row['sha256'], key + ':snapshot_original_equal')
        counts[key] = len(doc['files'])
    same(final['historical_files_verified_unchanged'], counts['historical_files'], 'final_history_count')
    source_audit = audit_bindings('SOURCE_PROVENANCE_AUDIT.json', 'input_bindings')
    evidence_audit = audit_bindings('EVIDENCE_EXPORT_AUDIT.json', 'checked_input_files')
    solver_audit = audit_bindings('INDEPENDENT_SOLVER_AUDIT.json', 'input_bindings')
    behavior = audit_bindings('BEHAVIOR_DIAGNOSTICS.json', 'input_bindings')
    analysis = read(BATCH / 'evidence/ANALYSIS_TOOL_BINDINGS.json'); verify(analysis['files'], 'analysis_tool_bindings')
    same(len(analysis['files']), 3, 'three_analysis_tools_recorded_before_solver')
    procedure_supplements = archive_and_procedure_supplements()
    for key, value in {'root_denominator': 4, 'scene_denominator': 24, 'full_ready_root_count': fields['complete_A3B3_roots'],
                       'ready_scene_count': fields['ready_scene_count'], 'total_new_turns': fields['accepted_new_turns']}.items():
        same(source_audit[key], value, 'source_audit:' + key)
    same(evidence_audit['candidate_count'], len(public), 'evidence_audit:candidate_count')
    same(sha(BATCH / 'evidence/audit_evidence_export_independent.py'), evidence_audit['audit_code_sha256'], 'evidence_audit:code_binding')
    adapter = read(BATCH / 'evaluation/ADAPTER_BINDINGS.json'); verify(adapter['files'], 'adapter_bindings')
    same(adapter['path_representation_adapter']['mode'], 'PROSPECTIVE', 'adapter:prospective_mode')
    same(adapter['path_representation_adapter']['prior_failed_prepare_claimed'], False, 'adapter:no_failed_prepare_claim')
    verify([adapter['path_representation_adapter']['receipt']], 'adapter:path_receipt_binding')
    same(adapter['native_generation_decoding_permutations_scoring_selection_changed'], False, 'adapter:native_protocol_unchanged')
    paired = read(BATCH / 'evaluation/PAIRED_PREFLIGHT.json'); verify(paired['files'], 'paired_preflight'); verify([paired['adapter_binding']], 'paired_adapter')
    same(paired['status'], 'PASS', 'paired_status')
    pool = by_id(read(BATCH / 'evaluation/EVALUATION_POOL.json')['questions']); same(set(pool), set(public), 'evaluation_content_pass_coverage')
    expected_calls = sum(18 if q['question_type'] == 'multi_select' else 12 for q in pool.values())
    same(paired['calls_per_model'], expected_calls, 'paired_calls'); same(adapter['calls_per_model'], expected_calls, 'adapter_calls')
    same(paired['candidate_count'], len(pool), 'paired_candidate_count'); same(adapter['candidate_count'], len(pool), 'adapter_candidate_count')
    path_check = prospective_path_receipt(pool, public, private)
    for side in ('base', 'trained'):
        same(solver_audit['results'][side], {'calls': expected_calls, 'candidates': len(pool), 'errors': 0}, 'solver_audit:' + side)
    for name in ('questions.json', 'gold.json'):
        same(read(BATCH / 'evaluation/run/base' / name), read(BATCH / 'evaluation/run/trained' / name), 'paired:' + name)
    sides = {side: raw_side(side, pool, final, behavior) for side in ('base', 'trained')}
    selected = read(BATCH / 'SELECTED_CANDIDATES.json'); selected_ids = set(by_id(selected['questions']))
    same(selected_ids, set(sides['trained']['selected_new_ids']), 'selection_complete_valid_rule_and_control_exclusion')
    check(all(q['role'] != 'control6' for q in selected['questions']), 'selection_no_Gold6')
    same(final['selected_new_question_count'], len(selected_ids), 'final_selected_count')
    same(final['selected_type_counts'], dict(Counter(q['question_type'] for q in selected['questions'])), 'final_selected_types')
    same(selected['training_started'], False, 'selected_no_training'); same(behavior['training_started'], False, 'behavior_no_training')
    same(behavior['pool_merge_performed'], False, 'behavior_no_merge')
    for q in selected['questions']: same(q, pool[q['candidate_id']], 'selected_question_verbatim:' + q['candidate_id'])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('/root/autodl-tmp/Qwen3-4B', local_files_only=True)
    lengths = []
    for root in summary['roots']:
        rid = root['root_id']; folder = BATCH / 'roots' / ('root' + rid.rsplit('-', 1)[1]); turns = read(folder / 'CUMULATIVE_SOURCE.json')['turns']
        new_turns = [t for t in turns if t.get('scene_ref') in cfg['scene_sequence']]
        plain = '\n'.join(t['speaker'] + ': ' + t['text'] for t in turns)
        refs = '\n'.join('[' + t['turn_ref'] + '] ' + t['speaker'] + ': ' + t['text'] for t in turns)
        new = '\n'.join(t['speaker'] + ': ' + t['text'] for t in new_turns)
        lengths.append({'root_id': rid, 'turns': len(turns), 'dialogue_tokens_with_refs_and_speakers': len(tok.encode(refs, add_special_tokens=False)),
                        'dialogue_tokens_speakers_and_body': len(tok.encode(plain, add_special_tokens=False)), 'new_turns': len(new_turns),
                        'new_dialogue_tokens_speakers_and_body': len(tok.encode(new, add_special_tokens=False))})
        check((folder / 'FULL_DIALOGUE.md').is_file() and (folder / 'QUESTIONS_AND_ANSWERS.md').is_file(), rid + ':human_readable_files')
    same(final['dialogue_lengths'], lengths, 'dialogue_tokens_independently_recomputed')
    same(final['mean_dialogue_tokens_speakers_and_body'], sum(r['dialogue_tokens_speakers_and_body'] for r in lengths) / len(lengths), 'mean_dialogue_tokens')
    same(sum(r['new_turns'] for r in lengths), fields['accepted_new_turns'], 'actual_new_turns')
    report = read_text(BATCH / 'REPORT_ZH.md'); interpretation = read_text(BATCH / 'INTERPRETATION_ZH.md'); read_first = read_text(BATCH / 'READ_FIRST.md')
    limitations = read_text(BATCH / 'PROCEDURE_LIMITATIONS_ZH.md')
    disclosures = report_disclosures(report + '\n' + read_first, interpretation, limitations, source_audit)
    for key, label in [('multi_select_gold2_5', '多选 Gold2–5'), ('gold2', 'Gold2'), ('gold3', 'Gold3'), ('gold4', 'Gold4'), ('gold5', 'Gold5'), ('ordering', '排序'), ('gold6_control', 'Gold6 对照')]:
        a, b = sides['base']['metrics'][key], sides['trained']['metrics'][key]
        check(f"|{label}|{a['correct']}/{a['calls']}|{b['correct']}/{b['calls']}|" in report, 'report_accuracy_table:' + key)
    out = {'status': 'PASS' if not ERRORS else 'FAIL', 'reviewer': '/root/v113_14304_recheck',
           'review_kind': 'completed_delivery_independent_mechanical_recomputation_and_disclosure_check_v117',
           'started_at_utc': started, 'finished_at_utc': datetime.now(timezone.utc).isoformat(), 'audit_code_sha256': sha(__file__),
           'roots': len(roots), 'scenes': fields['expected_scene_count'], 'new_turns': fields['accepted_new_turns'], 'questions': len(pool),
           'multi_gold_counts': fields['multi_gold_counts'], 'selected_new_count': len(selected_ids), 'raw_recomputed_sides': sides,
           'dialogue_lengths_recomputed': lengths, 'verified_hash_counts': counts, 'prospective_path_receipt': path_check,
           'source_procedure_warnings': source_audit.get('warnings', []), 'procedure_supplements_verified': procedure_supplements,
           'required_disclosures': disclosures,
           'all_existing_reports_unmodified_by_reviewer': True, 'new_training_or_merge_run_by_reviewer': False,
           'root_files_modified_by_reviewer': False, 'semantic_recertification': False, 'same_inherited_model_external_certification': False,
           'input_bindings': [{'path': p, 'sha256': digest} for p, digest in BINDINGS.items()], 'errors': ERRORS}
    with (BATCH / 'DELIVERY_REVIEW.json').open('x') as handle:
        json.dump(out, handle, ensure_ascii=False, indent=2); handle.write('\n')
    print(json.dumps({k: out[k] for k in ('status', 'roots', 'scenes', 'new_turns', 'questions', 'selected_new_count', 'errors')}))
    if ERRORS: raise SystemExit(1)


if __name__ == '__main__':
    main()
