"""Append-only private diagnostics; never provide these results to source-first reviewers.

Source-ID predicate snapshots separate independent uniqueness from agreement with program Gold.
Each stage is append-only and binds this helper.
Read frozen artifacts only. This helper neither reviews item semantics nor changes
production, source judgments, answers, or selection. Run generation, content, then
final snapshots after the corresponding authoritative COMPLETE/report exists.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import re

BATCH = Path(__file__).resolve().parents[1]
LABELS = {f'I{i}' for i in range(1, 7)}
CATEGORIES = ('schema', 'forbidden_gendered_pronoun', 'exact_whole_set_duplicate',
              'independent_source_semantics', 'fixed_anchor_containment',
              'unique_semantic_order', 'independent_order_program_gold_mismatch', 'program_gold_integrity', 'writer_null',
              'other_mechanical_or_review')


def read(path):
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError('duplicate_json_key:' + k)
            out[k] = v
        return out
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('invalid_constant:' + value)))


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def timestamp(value):
    return datetime.fromisoformat(value)


def category(reason):
    if 'fixed_anchor_disagrees_with_independent_semantic_anchor' in reason:
        return 'fixed_anchor_containment'
    if any(x in reason for x in ('supported_failed', 'actor_ok_failed', 'modality_ok_failed',
                                 'state_ok_failed', 'coherent_failed', 'neutral_stem_ok_failed')):
        return 'independent_source_semantics'
    if 'private_gold_disagrees_with_independent_source_order' in reason:
        return 'independent_order_program_gold_mismatch'
    if any(x in reason for x in ('unique_order', 'alternative_order', 'alternative_anchor_changes')):
        return 'unique_semantic_order'
    if any(x in reason for x in ('gendered_pronoun', 'forbidden_pronoun', 'actor_name_policy', 'forbidden_gender')):
        return 'forbidden_gendered_pronoun'
    if 'duplicate_prior_or_current_six_item_set' in reason:
        return 'exact_whole_set_duplicate'
    if 'writer_null_slot' in reason:
        return 'writer_null'
    if 'program_gold' in reason:
        return 'program_gold_integrity'
    if any(x in reason for x in ('schema', 'ordering_bound_writer_identity', 'ordering_bound_writer_items', 'ordering_writer_anchor_id_coverage', 'ordering_writer_record_text',
                                 'JSON', 'json', 'Expecting', 'duplicate_key', 'extra_fields', 'missing_fields')):
        return 'schema'
    return 'other_mechanical_or_review'


def normalized_set(row):
    values = row.get('items') if isinstance(row, dict) else None
    if not isinstance(values, dict) or len(values) != 6 or not all(isinstance(v, str) for v in values.values()):
        return None
    return tuple(sorted(' '.join(v.split()).casefold() for v in values.values()))


def parse_raw(path):
    if path is None:
        return None, 'pending_response'
    try:
        response = read(path)
        # A second strict decode catches duplicate keys in the writer's JSON text.
        def pairs(items):
            result = {}
            for k, v in items:
                if k in result:
                    raise ValueError('duplicate_json_key:' + k)
                result[k] = v
            return result
        obj = json.loads(response['text'], object_pairs_hook=pairs,
                         parse_constant=lambda value: (_ for _ in ()).throw(ValueError('invalid_constant:' + value)))
        if not isinstance(obj, dict) or set(obj) != {'questions'} or not isinstance(obj['questions'], list) or len(obj['questions']) != 2:
            return None, 'ordering_response_schema'
        return obj['questions'], None
    except (ValueError, KeyError, TypeError) as error:
        return None, str(error)


def snapshot(stage):
    if (BATCH / 'evidence/RUN_COMPLETE.json').exists():
        raise ValueError('batch_already_sealed')
    ordering = BATCH / 'ordering'
    required = {'generation': ordering / 'GENERATION.COMPLETE.json',
                'content': ordering / 'CONTENT_REVIEW.COMPLETE.json',
                'final': BATCH / 'MIXED_EVALUATION_REPORT.json'}[stage]
    if not required.exists():
        raise ValueError('stage_not_complete:' + str(required))
    out = BATCH / 'evidence' / ('ORDERING_DIAGNOSTIC_' + stage.upper() + '.json')
    markdown = out.with_suffix('.md')
    if out.exists() or markdown.exists():
        raise ValueError('snapshot_already_exists:' + str(out))
    files = {Path(__file__).resolve(), BATCH / 'CONFIG.json', ordering / 'GENERATION.COMPLETE.json'}
    config = read(BATCH / 'CONFIG.json')
    native_path = ordering / 'SOURCE_JUDGMENTS.json'
    content_path = ordering / 'CONTENT_REVIEW.json'
    native = {q['candidate_id']: q for q in read(native_path)['questions']} if native_path.exists() else {}
    content = {q['candidate_id']: q for q in read(content_path)['questions']} if content_path.exists() else {}
    for path in (native_path, content_path, ordering / 'CONTENT_REVIEW.COMPLETE.json'):
        if path.exists():
            files.add(path)
    report_path = BATCH / 'MIXED_EVALUATION_REPORT.json'
    report = read(report_path) if report_path.exists() else None
    selected = set(report['selected_candidate_ids']) if report is not None else None
    if report is not None:
        files.add(report_path)
    results = {}
    for side in ('base', 'trained'):
        path = BATCH / 'solver' / side / 'RESULTS.json'
        if path.exists():
            files.add(path)
            results[side] = {q['candidate_id']: q for q in read(path)['questions']}
    rows = []
    for slot in config['ordering_slots']:
        run = ordering / slot['slot_id']
        manifest_path, plan_path, gold_path = (run / name for name in ('MANIFEST.json', 'ANCHOR_PLAN.json', 'GOLD_PLAN.json'))
        manifest = read(manifest_path)
        plans = {q['candidate_id']: q for q in read(plan_path)['questions']}
        answers = {q['candidate_id']: q for q in read(gold_path)['questions']}
        gold_record = read(gold_path)
        source_path, prior_path = run / 'SOURCE.json', run / 'PRIOR.json'
        source, prior = read(source_path), read(prior_path)
        positions = {t['turn_ref']: i for i, t in enumerate(source['turns'])}
        files.update((manifest_path, plan_path, gold_path, source_path, prior_path, run / 'WRITER_PLAN.json'))
        completion_path, rejection_path, pool_path = (run / name for name in ('GENERATION.COMPLETE.json', 'REJECTIONS.json', 'POOL.json'))
        completed = completion_path.exists()
        rejections = {q['candidate_id']: q['reasons'] for q in read(rejection_path)} if rejection_path.exists() else {}
        pool = {q['candidate_id']: q for q in read(pool_path)['questions']} if pool_path.exists() else {}
        for path in (completion_path, rejection_path, pool_path, run / 'GENERATION.STARTED.json'):
            if path.exists():
                files.add(path)
        response_paths = sorted((run / 'requests').glob('*/RESPONSE.json'))
        if len(response_paths) > 1:
            raise ValueError('unexpected_multiple_responses:' + slot['slot_id'])
        response_path = response_paths[0] if response_paths else None
        if response_path:
            files.add(response_path)
        raw, parse_error = parse_raw(response_path)
        prior_by_set = {}
        for q in prior:
            key = normalized_set(q)
            if key is not None:
                prior_by_set.setdefault(key, []).append(q['candidate_id'])
        observed_current = {}
        for number, qid in enumerate(manifest['candidate_ids']):
            planned = plans[qid]
            answer = answers[qid]
            draft = raw[number] if raw is not None else None
            records = draft.get('records') if isinstance(draft,dict) else None
            schema_ok = bool(isinstance(draft, dict) and set(draft) == {'candidate_id', 'root_id', 'records'}
                             and draft.get('candidate_id') == qid and draft.get('root_id') == slot['root_id']
                             and isinstance(records,list) and len(records)==6
                             and all(isinstance(r,dict) and set(r)=={'anchor_id','predicate'}
                                 and isinstance(r['anchor_id'],str) and isinstance(r['predicate'],str)
                                 and bool(r['predicate']) and r['predicate']==r['predicate'].strip() for r in records))
            if schema_ok:
                schema_ok = (len({r['anchor_id'] for r in records})==6
                    and {r['anchor_id'] for r in records}=={a['turn_ref'] for a in planned['anchors'].values()})
            assembled = None
            if schema_ok:
                indexed={r['anchor_id']:r['predicate'] for r in records}
                assembled={'items':{label:anchor['speaker']+' '+indexed[anchor['turn_ref']]
                    for label,anchor in planned['anchors'].items()}}
            pronouns = []
            if isinstance(records,list):
                for record in records:
                    if isinstance(record,dict) and isinstance(record.get('predicate'),str):
                        matches = re.findall(r'\b(?:he|him|his|himself|she|her|hers|herself)\b', record['predicate'], re.I)
                        if matches:
                            pronouns.append({'anchor_id': record.get('anchor_id'), 'words': matches})
            key = normalized_set(assembled)
            duplicate_prior = prior_by_set.get(key, []) if key is not None else []
            duplicate_current = observed_current.get(key, []) if key is not None else []
            if key is not None:
                observed_current.setdefault(key, []).append(qid)
            if completed:
                if (qid in pool) == (qid in rejections):
                    raise ValueError('mechanical_outcome_not_exactly_one:' + qid)
                mechanical_status = 'PASS' if qid in pool else 'REJECTED'
            else:
                mechanical_status = 'PENDING'
            reasons = rejections.get(qid, [])
            question_native, question_content = native.get(qid), content.get(qid)
            content_reasons = question_content['reasons'] if question_content else []
            semantic_flags = []
            if question_native:
                for label, item in question_native['items'].items():
                    for flag in ('supported', 'actor_ok', 'modality_ok', 'state_ok', 'neutral_stem_ok'):
                        if item[flag] is False:
                            semantic_flags.append(label + ':' + flag + '_failed')
                for flag in ('coherent', 'neutral_stem_ok'):
                    if question_native[flag] is False:
                        semantic_flags.append(flag + '_failed')
            not_reviewed = mechanical_status == 'REJECTED'
            content_status = question_content['status'] if question_content else 'NOT_REVIEWED_MECHANICAL_REJECTION' if not_reviewed else 'PENDING'
            semantic_status = ('REJECTED' if semantic_flags else 'PASS') if question_native else 'NOT_REVIEWED_MECHANICAL_REJECTION' if not_reviewed else 'PENDING'
            anchor_errors = [r for r in content_reasons if category(r) == 'fixed_anchor_containment']
            order_errors = [r for r in content_reasons if category(r) == 'unique_semantic_order']
            answer_errors = [r for r in content_reasons if category(r) == 'independent_order_program_gold_mismatch']
            if question_native and question_native['unique_order'] is False and 'unique_order_failed' not in order_errors:
                order_errors.append('unique_order_failed')
            anchor_status = ('REJECTED' if anchor_errors else 'PASS') if question_content else 'NOT_REVIEWED_MECHANICAL_REJECTION' if not_reviewed else 'PENDING'
            order_status = ('REJECTED' if order_errors else 'PASS') if question_content else 'NOT_REVIEWED_MECHANICAL_REJECTION' if not_reviewed else 'PENDING'
            expected_order = sorted(planned['anchors'], key=lambda label: positions[planned['anchors'][label]['turn_ref']])
            created_before_prepared = timestamp(gold_record['created_at_utc']) <= timestamp(manifest['prepared_at_utc'])
            start_path = run / 'GENERATION.STARTED.json'
            prepared_before_generation = timestamp(manifest['prepared_at_utc']) <= timestamp(read(start_path)['started_at_utc']) if start_path.exists() else None
            program_ok = bool(answer['gold'] == expected_order and manifest['gold_plan'] == bind(gold_path)
                              and gold_record['source'] == bind(source_path) and gold_record['anchor_plan'] == bind(plan_path)
                              and created_before_prepared and prepared_before_generation is True
                              and (qid not in pool or pool[qid]['gold'] == answer['gold']))
            solver_models = {}
            for side, qs in results.items():
                if qid in qs:
                    solver_models[side] = {k: qs[qid][k] for k in ('full_exact_rate', 'full_partial_mean', 'selected_by_policy', 'selection_reason')}
            if not_reviewed or (question_content and question_content['status'] != 'USABLE'):
                solver_status = 'NOT_EVALUATED_REJECTED_BEFORE_SOLVER'
                selected_state = False
            elif question_content is not None and report is not None and all(qid in results.get(side, {}) for side in ('base', 'trained')):
                solver_status = 'SELECTED' if qid in selected else 'NOT_SELECTED_BY_DIFFICULTY_POLICY'
                selected_state = qid in selected
            else:
                solver_status = 'PENDING'
                selected_state = None
            rows.append({'candidate_id': qid, 'root_id': slot['root_id'], 'slot_number': number + 1,
                         'scene_id': planned['scene_id'],
                         'raw_observations': {'response_available': response_path is not None, 'parse_error': parse_error,
                           'writer_returned_null': raw is not None and draft is None,
                           'source_id_predicate_records_schema_observed': schema_ok if raw is not None else None,
                           'raw_field_names': sorted(draft) if isinstance(draft, dict) else [],
                           'assembled_items_before_semantic_review': assembled['items'] if assembled else None,
                           'forbidden_pronoun_occurrences': pronouns,
                           'exact_normalized_prior_duplicates': duplicate_prior,
                           'exact_normalized_earlier_raw_slot_duplicates': duplicate_current},
                         'program_gold': {'gold': answer['gold'], 'reconstructed_source_order': expected_order,
                           'pre_frozen_binding_and_order_check': 'PASS' if program_ok else 'FAIL',
                           'semantic_certification': False, 'gold_supplied_to_writer': gold_record['gold_supplied_to_writer']},
                         'mechanical': {'status': mechanical_status, 'official_reasons': reasons,
                           'official_failure_categories': sorted({category(r) for r in reasons})},
                         'independent_source_semantics': {'status': semantic_status, 'explicit_failed_flags': semantic_flags,
                           'reviewer_reason': question_native.get('reason') if question_native else None,
                           'note': 'Only saved independent source judgments determine semantics; a PASS here covers supported/actor/modality/state/coherence flags, not fixed-anchor containment or unique order.'},
                         'fixed_anchor_containment': {'status': anchor_status, 'reasons': anchor_errors,
                           'note': 'Saved private crosscheck requires each independent semantic anchor span inside the fixed quote. Span containment failure is separate from independently supported text.'},
                         'unique_semantic_order': {'status': order_status, 'reasons': order_errors,
                           'independent_unique_order_flag': question_native['unique_order'] if question_native else None,
                           'independent_computed_order': question_content.get('computed_order') if question_content else None},
                         'independent_order_program_gold_agreement': {'status': ('REJECTED' if answer_errors else 'PASS') if question_content else 'NOT_REVIEWED_MECHANICAL_REJECTION' if not_reviewed else 'PENDING', 'reasons': answer_errors},
                         'content': {'status': content_status, 'official_reasons': content_reasons,
                           'official_failure_categories': sorted({category(r) for r in content_reasons})},
                         'solver_selection': {'status': solver_status, 'selected': selected_state, 'models': solver_models}})
    if len(rows) != 8 or len({r['candidate_id'] for r in rows}) != 8:
        raise ValueError('expected_eight_slots')
    counts = {'slots': len(rows), 'mechanical_status': dict(Counter(r['mechanical']['status'] for r in rows)),
              'content_status': dict(Counter(r['content']['status'] for r in rows)),
              'solver_selection_status': dict(Counter(r['solver_selection']['status'] for r in rows)),
              'official_mechanical_failure_slots_by_category': {c: sum(c in r['mechanical']['official_failure_categories'] for r in rows) for c in CATEGORIES},
              'official_content_failure_slots_by_category': {c: sum(c in r['content']['official_failure_categories'] for r in rows) for c in CATEGORIES},
              'raw_forbidden_pronoun_slots_observed': sum(bool(r['raw_observations']['forbidden_pronoun_occurrences']) for r in rows),
              'raw_exact_prior_duplicate_slots_observed': sum(bool(r['raw_observations']['exact_normalized_prior_duplicates']) for r in rows),
              'program_gold_binding_and_order_passed_slots': sum(r['program_gold']['pre_frozen_binding_and_order_check'] == 'PASS' for r in rows)}
    document = {'version': 'v88_ordering_stage_diagnostic_v2', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'stage': stage, 'private_diagnostic_not_for_independent_reviewer': True,
                'authoritative_judgments_changed': False, 'source_or_answer_edits': [],
                'summary': counts, 'slots': rows, 'inputs': [bind(p) for p in sorted(files)],
                'limitations': ['This is artifact-based diagnosis, not a new semantic review or a mechanism to rescue rejected questions.',
                  'Mechanical validation short-circuits; official failure categories describe the recorded stopping condition. Raw observations may reveal additional style/exact-duplicate issues but do not replace that verdict.',
                  'Program Gold was derived from the frozen anchors before the response. A matching program answer does not establish correct item meaning or anchor binding.',
                  'Fixed-anchor containment can fail even when the independent reviewer supports the statement and derives a unique order; it remains a rejection under the frozen contract.',
                  'Every missing downstream stage is explicitly pending or not evaluated due to prior rejection. No selected status is inferred from absent results.',
                  'No private diagnostics may be passed to a reviewer who has not yet saved independent source judgments.']}
    with out.open('x') as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2); stream.write('\n')
    lines = [f'v88 排序诊断：{stage}', '', '此文件含私有程序 Gold 和审核交叉结果，不应提供给尚未保存独立判断的审核者。', '',
             '| Root / slot | 机械 | 独立语义 | 固定锚点包含 | 独立顺序唯一 | 顺序符合程序Gold | 内容结论 | Solver / 入选 |',
             '|---|---|---|---|---|---|---|---|']
    for row in rows:
        lines.append('| ' + ' | '.join([row['root_id'].split('-')[-1] + ' / ' + str(row['slot_number']),
                    row['mechanical']['status'], row['independent_source_semantics']['status'],
                    row['fixed_anchor_containment']['status'], row['unique_semantic_order']['status'],
                    row['independent_order_program_gold_agreement']['status'], row['content']['status'], row['solver_selection']['status']]) + ' |')
    lines.extend(['', '程序预先生成答案只验证冻结锚点的来源顺序，不证明模型陈述对应正确。机械拒绝原因是实际短路停点；后续未审核不记作通过。', '',
                  '逐项原始字段、禁用代词、重复、全部正式拒绝原因和来源绑定见同名 JSON。'])
    with markdown.open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    return {'stage': stage, 'summary': counts, 'json': bind(out), 'markdown': bind(markdown)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('generation','content','final'), required=True)
    args = parser.parse_args()
    print(json.dumps(snapshot(args.stage), ensure_ascii=False, indent=2))
