#!/usr/bin/env python3
"""Validate source-first reviews, bind them, then join frozen private scoring."""
import argparse, collections, datetime, hashlib, itertools, json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
WEEK = HERE.parent
JUDGMENTS = {'supported', 'contradicted', 'not_supported', 'ambiguous'}
SUPPORT = {'supports_item', 'refutes_item', 'unrelated', 'insufficient', 'missing_or_invalid'}

def read(path):
    return json.loads(pathlib.Path(path).read_text())

def bind(path):
    p = pathlib.Path(path)
    b = p.read_bytes()
    return {'path': str(p), 'sha256': hashlib.sha256(b).hexdigest(), 'bytes': len(b)}

def save(name, obj):
    p = HERE / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    return bind(p)

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def verify(b, errors, where):
    try:
        actual = bind(b['path'])
        if actual['sha256'] != b['sha256'] or actual['bytes'] != b.get('bytes', actual['bytes']):
            errors.append({'where': where, 'error': 'binding_changed', 'expected': b, 'actual': actual})
    except Exception as exc:
        errors.append({'where': where, 'error': 'binding_unreadable', 'detail': str(exc)})

def ambiguous(value):
    return value is True or isinstance(value, dict) and value.get('is_ambiguous') is True

def compatible_status(source, stated):
    if stated is None or source == 'ambiguous':
        return None
    if source == 'supported':
        return stated == 'supported'
    if source in {'contradicted', 'not_supported'}:
        return stated in {'contradicted', 'not_supported'}
    return None

def validate():
    start = read(HERE / 'REVIEW_START.json')
    errors, pending, warnings, all_records, inputs = [], [], [], [], []
    for b in start['source_and_response_inputs'] + start['packets']:
        verify(b, errors, 'start_input')
    for n in range(1, 4):
        packet_path = HERE / 'packets' / f'group_{n:02}.json'
        review_path = HERE / 'reviews' / f'group_{n:02}.json'
        packet = read(packet_path)
        records = {r['request_id']: r for r in packet['records']}
        if not review_path.exists():
            pending.append({'group': n, 'reason': 'review_not_saved'})
            continue
        try:
            rev = read(review_path)
        except Exception as exc:
            pending.append({'group': n, 'reason': 'review_write_in_progress', 'detail': str(exc)})
            continue
        if rev.get('status') != 'COMPLETE_WITH_UNCERTAINTY_ALLOWED' or len(rev.get('records', [])) != len(records):
            pending.append({'group': n, 'status': rev.get('status'), 'records': len(rev.get('records', [])), 'expected': len(records)})
            continue
        verify(rev['input_packet_binding'], errors, f'group_{n}_packet')
        inputs.append(bind(review_path))
        seen = set()
        for r in rev['records']:
            rid = r['request_id']
            if rid not in records or rid in seen:
                errors.append({'where': rid, 'error': 'unexpected_or_duplicate_review'})
                continue
            seen.add(rid)
            p = records[rid]
            for key in ['candidate_id', 'root_id', 'question_type']:
                if r.get(key) != p[key]: errors.append({'where': rid, 'error': 'identity_mismatch', 'field': key})
            if r.get('scope_fully_read') is not True:
                errors.append({'where': rid, 'error': 'scope_not_fully_read'})
            if r.get('source_binding') != p['source_binding']:
                errors.append({'where': rid, 'error': 'source_binding_mismatch'})
            verify(p['source_binding'], errors, rid + ':source')
            verify(p['response_binding'], errors, rid + ':response')
            turns = {t['turn_ref']: t for t in p['all_scoped_source_turns']}
            source = read(p['source_binding']['path'])
            expected_scope = [v for v in source['turns'] if v['turn_ref'].rsplit(':', 1)[0] in set(p['scope_scene_refs'])]
            if expected_scope != p['all_scoped_source_turns']:
                errors.append({'where': rid, 'error': 'packet_scope_incomplete_or_reordered'})
            for t in p['all_scoped_source_turns']:
                orig = [v for v in source['turns'] if v['turn_ref'] == t['turn_ref']]
                if len(orig) != 1 or orig[0]['text'] != t['text']:
                    errors.append({'where': rid, 'error': 'packet_source_mismatch', 'turn_ref': t['turn_ref']})
            items = {i['label']: i for i in p['displayed_items']}
            labels = [i.get('label') for i in r.get('items', [])]
            if len(labels) != 6 or set(labels) != set(items):
                errors.append({'where': rid, 'error': 'six_item_coverage'})
            gold = r.get('source_gold_display')
            if gold is not None and (len(set(gold)) != len(gold) or not set(gold) <= set(items)):
                errors.append({'where': rid, 'error': 'invalid_reconstructed_answer'})
            if p['question_type'] == 'ordering' and gold is not None and set(gold) != set(items):
                errors.append({'where': rid, 'error': 'ordering_not_permutation'})
            if p['question_type'] == 'multi_select' and gold is not None:
                supported = {i['label'] for i in r['items'] if i.get('source_judgment') == 'supported'}
                if supported != set(gold): errors.append({'where': rid, 'error': 'source_answer_judgment_disagreement'})
            for item in r.get('items', []):
                label = item['label']; loc = rid + ':' + label
                if label not in items: continue
                if item.get('source_judgment') not in JUDGMENTS: errors.append({'where': loc, 'error': 'source_enum'})
                if item.get('solver_citation_semantic_support') not in SUPPORT: errors.append({'where': loc, 'error': 'support_enum'})
                for f in ['solver_status_correct', 'final_answer_consistency']:
                    if item.get(f) is not None and type(item[f]) is not bool: errors.append({'where': loc, 'error': 'boolean_enum', 'field': f})
                for citation in item.get('source_citations', []):
                    ref, quote = citation.get('turn_ref'), citation.get('quote')
                    if ref not in turns or not quote or quote not in turns[ref]['text']:
                        errors.append({'where': loc, 'error': 'review_quote_not_verbatim_in_scope', 'citation': citation})
                if item.get('source_judgment') == 'not_supported' and (not r.get('scope_fully_read') or not item.get('reason_zh')):
                    errors.append({'where': loc, 'error': 'absence_search_scope_or_reason_missing'})
                if item.get('source_judgment') in {'supported', 'contradicted'} and not item.get('source_citations'):
                    errors.append({'where': loc, 'error': 'source_evidence_missing'})
                ev = items[label].get('solver_evidence')
                expected = compatible_status(item.get('source_judgment'), ev.get('status') if ev else None) if p['solver_output_valid'] else None
                if item.get('solver_status_correct') != expected:
                    errors.append({'where': loc, 'error': 'status_correctness_definition_mismatch', 'expected': expected, 'actual': item.get('solver_status_correct')})
                if p['question_type'] == 'multi_select' and p['solver_output_valid'] and ev:
                    consistency = (ev['status'] == 'supported') == (label in p['solver_answer'])
                    if item.get('final_answer_consistency') != consistency:
                        errors.append({'where': loc, 'error': 'mc_final_consistency_definition_mismatch', 'expected': consistency})
                if not p['solver_output_valid'] and any(item.get(f) is not None for f in ['solver_status_correct', 'final_answer_consistency']):
                    errors.append({'where': loc, 'error': 'invalid_response_invented_judgment'})
                if not isinstance(item.get('relation_tags'), list) or not item.get('reason_zh'):
                    errors.append({'where': loc, 'error': 'relation_or_reason_missing'})
            all_records.append((p, r, bind(review_path)))
        if seen != set(records): errors.append({'where': f'group_{n}', 'error': 'record_coverage'})
    ids = [p['request_id'] for p, _, _ in all_records]
    if len(ids) != len(set(ids)): errors.append({'error': 'cross_group_duplicate'})
    if not pending and (len(ids) != 39 or sum(len(r['items']) for _, r, _ in all_records) != 234):
        errors.append({'error': 'global_39_records_234_items_not_covered'})
    result = {'schema_version': 'day2_feedback_runtime_validation_v1', 'checked_at_utc': now(),
        'status': 'PENDING_REVIEWS' if pending else ('FAIL' if errors else 'PASS'),
        'completed_records': len(ids), 'expected_records': 39, 'completed_items': sum(len(r['items']) for _, r, _ in all_records),
        'expected_items': 234, 'review_bindings': inputs, 'pending': pending, 'errors': errors, 'warnings': warnings,
        'scope': 'Identity, source/response hashes, six-item coverage, exact source quotations, enum and consistency definitions. Semantic correctness remains reviewer judgment.',
        'new_solver_calls': 0, 'private_gold_opened_by_this_validation': False}
    save('audit/VALIDATION.json', result)
    return result, all_records

def build(validation, reviewed):
    if validation['status'] != 'PASS': return
    # Commit source-first reviews before opening private answer mappings or score gold.
    freeze = {'schema_version': 'source_review_freeze_v1', 'frozen_at_utc': now(),
        'reviews': validation['review_bindings'], 'validation': bind(HERE / 'audit/VALIDATION.json'),
        'parent_prior_gold_exposure': True, 'external_blind_review': False,
        'reviewers_prior_statistical_exposure_disclosed': True}
    save('SOURCE_REVIEW_FREEZE.json', freeze)
    mappings_path = WEEK / 'day2/solver/evidence_v2/private/DISPLAY_MAPPINGS.json'
    score_path = WEEK / 'day2/solver/evidence_v2/SCORES.json'
    obs_path = WEEK / 'day2/analysis/results_02/OBSERVATIONS.json'
    manifest_path = WEEK / 'day2/solver/evidence_v2/manifest.json'
    mappings, scores, obs, manifest = map(read, [mappings_path, score_path, obs_path, manifest_path])
    sm = {r['request_id']: r for r in scores['requests']}
    dq = {r['candidate_id']: r for r in obs['runs']['direct']['questions']}
    dr = collections.defaultdict(list)
    for r in obs['runs']['direct']['requests']: dr[r['candidate_id']].append(r)
    records, compare, counter = [], [], collections.Counter()
    for p, review, rb in reviewed:
        rid, cid, kind = p['request_id'], p['candidate_id'], p['question_type']
        mp = mappings['requests'][rid]['display_to_content']
        orig = mappings['public_to_canonical'][cid]
        inv_orig = {v: k for k, v in orig.items()}
        original_label = {k: inv_orig[v] for k, v in mp.items()}
        rec_gold = review.get('source_gold_display')
        source_ids = [mp[v] for v in rec_gold] if rec_gold is not None else None
        stored_ids = sm[rid]['score'].get('content_gold')
        agrees = None if source_ids is None else (set(source_ids) == set(stored_ids) if kind == 'multi_select' else source_ids == stored_ids)
        holds = agrees is not True or ambiguous(review.get('source_order_ambiguity')) or any(i['source_judgment'] == 'ambiguous' for i in review['items'])
        compare.append({'request_id': rid, 'candidate_id': cid, 'source_reconstructed_display': rec_gold,
            'stored_gold_display': [k for content in stored_ids for k, v in mp.items() if v == content],
            'source_matches_stored_gold': agrees, 'content_hold': holds, 'uncertainties': review.get('uncertainties', [])})
        labels = []
        errors = []
        for item in review['items']:
            label = item['label']; ol = original_label[label]
            frequency = {'valid_calls': sum(bool(r['valid']) for r in dr[cid])}
            if kind == 'multi_select':
                frequency.update({'false_positive_calls': sum(ol in r.get('false_positive_labels', []) for r in dr[cid]),
                    'false_negative_calls': sum(ol in r.get('false_negative_labels', []) for r in dr[cid])})
            else:
                pairs = [pair for r in dr[cid] for pair in r.get('ordering_pairs', []) if ol in (pair['earlier_gold_label'], pair['later_gold_label'])]
                frequency.update({'involving_pairs': len(pairs), 'inverted_pairs': sum(pair.get('correct') is False for pair in pairs)})
            displayed = next(i for i in p['displayed_items'] if i['label'] == label)
            label_record = dict(item, citation_mechanics=displayed['mechanical'], original_public_label=ol, canonical_content_id=mp[label], direct_repetition_frequency=frequency,
                displayed_text=next(i['text'] for i in p['displayed_items'] if i['label'] == label),
                solver_evidence=next(i.get('solver_evidence') for i in p['displayed_items'] if i['label'] == label))
            if kind == 'multi_select':
                direct_errors = frequency['false_positive_calls'] + frequency['false_negative_calls']
                diagnostic_wrong = p['solver_output_valid'] and item['solver_status_correct'] is False
                label_record['direct_vs_diagnostic'] = ('diagnostic_only_error_direct_item_all_correct' if diagnostic_wrong and direct_errors == 0 else 'both_direct_and_diagnostic_errors' if diagnostic_wrong else 'direct_error_diagnostic_item_correct' if direct_errors and p['solver_output_valid'] and item['solver_status_correct'] is True else 'direct_error_diagnostic_unavailable' if direct_errors and not p['solver_output_valid'] else 'no_observed_error_in_compared_conditions')
            else:
                label_record['direct_vs_diagnostic'] = 'ordering_answer_and_evidence_are_separate_levels'
            labels.append(label_record)
            counter['source:' + item['source_judgment']] += 1
            counter['citation:' + item['solver_citation_semantic_support']] += 1
            counter['status:' + str(item['solver_status_correct'])] += 1
            counter['consistency:' + str(item['final_answer_consistency'])] += 1
            if item['solver_status_correct'] is False or item['final_answer_consistency'] is False:
                errors.append({'label': label, 'source_judgment': item['source_judgment'], 'reviewed_relation_tags': item['relation_tags'],
                    'reason_zh': item['reason_zh'], 'source_citations': item['source_citations']})
        direct = dq[cid]
        if kind == 'multi_select':
            pair_frequencies = None
            outcome = {k: direct.get(k) for k in ['gold_count', 'planned_calls', 'valid_calls', 'exact_calls', 'exact_rate', 'tp', 'fp', 'fn', 'overselection_rate_valid_only', 'all_selection_rate_valid_only']}
        else:
            pair_frequencies = []
            pair_map = collections.defaultdict(list)
            for r in dr[cid]:
                for pair in r.get('ordering_pairs', []): pair_map[(pair['earlier_gold_label'], pair['later_gold_label'])].append(pair)
            for (earlier, later), rows in sorted(pair_map.items()):
                pair_frequencies.append({'earlier_original_label': earlier, 'later_original_label': later, 'calls': len(rows),
                    'inverted_calls': sum(v.get('correct') is False for v in rows), 'relation_scope': sorted({v['kind'] for v in rows})})
            outcome = {k: direct.get(k) for k in ['planned_calls', 'valid_calls', 'exact_calls', 'exact_rate', 'ordering_pair_correct', 'ordering_pair_denominator', 'pair_accuracy_fixed_denominator']}
        citation_pair_check = None
        if kind == 'ordering' and p['solver_output_valid']:
            positions = {}
            for item in labels:
                mech = item['citation_mechanics']
                if item['solver_citation_semantic_support'] == 'supports_item' and mech.get('declared_turn_exists') and mech.get('quote_verbatim_in_declared_turn'):
                    positions[item['label']] = mech['declared_source_turn_index']
            answer_pos = {label: i for i, label in enumerate(p['solver_answer'])}
            pairs = []
            for left, right in itertools.combinations(sorted(positions), 2):
                if positions[left] == positions[right]: continue
                earlier, later = (left, right) if positions[left] < positions[right] else (right, left)
                pairs.append({'earlier': earlier, 'later': later, 'consistent': answer_pos[earlier] < answer_pos[later]})
            citation_pair_check = {'policy': 'Only semantically supported citations with exact quote in declared scoped turn; compare cross-turn pairs, exclude ties/nonliteral/unsupported. This conservative measure is separate from reviewer per-item flags.', 'eligible_labels': sorted(positions), 'pair_count': len(pairs), 'contradictory_pairs': sum(not q['consistent'] for q in pairs), 'pairs': pairs}
        qualified = not holds and (bool(errors) or direct['exact_calls'] < direct['planned_calls'])
        source_exact = None if rec_gold is None or not p['solver_output_valid'] else (set(p['solver_answer']) == set(rec_gold) if kind == 'multi_select' else p['solver_answer'] == rec_gold)
        records.append({'schema_version': 'real_feedback_record_v1', 'feedback_id': 'D2FB-' + hashlib.sha256(cid.encode()).hexdigest()[:14],
            'namespace': 'train.feedback', 'candidate_id': cid, 'request_id': rid, 'root_id': p['root_id'], 'family_id': p['root_id'] + '|' + kind,
            'question_type': kind, 'family_definition': 'root_id|question_type, historical de-duplication key; not a claim of different structural question genres', 'solver_model_binding': {'name': manifest['solver_model'], 'model': manifest['model'], 'model_files': manifest['model_bindings'], 'manifest': bind(manifest_path)},
            'source_binding': p['source_binding'], 'response_binding': p['response_binding'], 'source_review_binding': rb,
            'source_review_freeze_binding': bind(HERE / 'SOURCE_REVIEW_FREEZE.json'), 'display_mapping_binding': bind(mappings_path),
            'display_to_original_label': original_label,
            'exposure_history': ['Train source; same candidate already evaluated in v137 and Day2 direct baseline', 'S0 is first-round M-clean; no new training', 'Internal AI source-first review; prior aggregate/source exposure disclosed; not external blinded review'],
            'source_gold_comparison': compare[-1], 'content_status': 'HOLD' if holds else 'SOURCE_RECONSTRUCTION_AGREES_WITH_STORED_GOLD',
            'training_reuse_gate': 'HOLD_NEUTRAL_WORDING_REWRITE_AND_REVIEW' if p['root_id'] == 'cc-train-episode-162896' and kind == 'ordering' else 'NOT_ADMITTED_BY_THIS_DIAGNOSTIC_AUDIT',
            'operational_case_usage': 'EVENT_ORDER_ONLY_GENDER_DIMENSION_EXCLUDED' if p['root_id'] == 'cc-train-episode-162896' and kind == 'ordering' else 'SOURCE_REVIEWED_OBSERVATION',
            'diagnostic_response': {'valid': p['solver_output_valid'], 'parse_error': p.get('parse_error'), 'answer': p['solver_answer'], 'source_exact_if_valid': source_exact, 'normalized_citation_pair_check': citation_pair_check, 'reviewer_consistency_flags_not_pooled_as_a_single_causal_metric': True, 'only_one_response_not_repeatable_mechanism_evidence': True},
            'direct_outcome': outcome, 'direct_permutation_observations': direct['permutation_observations'],
            'direct_pair_frequencies': pair_frequencies, 'items': labels, 'observed_error_evidence': errors,
            'answer_error': {'diagnostic_answer_wrong': None if source_exact is None else not source_exact, 'direct_wrong_calls': direct['planned_calls'] - direct['exact_calls'], 'direct_planned_calls': direct['planned_calls'], 'source_order_final_consistency_reviewed': review.get('source_order_final_consistency')},
            'evidence_validity_error': {'invalid_response': not p['solver_output_valid'], 'items_without_exact_declared_quote': [i['label'] for i in labels if not i['citation_mechanics'].get('quote_verbatim_in_declared_turn')], 'semantic_unrelated_or_insufficient': [i['label'] for i in labels if i['solver_citation_semantic_support'] in ['unrelated', 'insufficient']]},
            'reviewer_question_fields': {k: review[k] for k in ['source_order_final_consistency', 'verified_semantic_citation_pair_checks', 'operational_candidate'] if k in review},
            'content_warnings': [{'code': 'CONTENT_WARNING_NON_DECISIVE', 'detail_zh': 'I1的he未由六场scope建立；本记录只使用事件及唯一出现顺序，不推测性别。未来出题改用姓名；若目标依赖性别则HOLD。'}] if p['root_id'] == 'cc-train-episode-162896' and kind == 'ordering' else [],
            'error_interpretation_zh': review['error_interpretation_zh'], 'uncertainties': review.get('uncertainties', []),
            'trigger_relation': {'relation_status': 'supported_by_source_review', 'within_vs_across_scene': 'ACROSS_SCENES_ONLY' if kind == 'ordering' else 'ITEM_SPECIFIC', 'reviewed_error_items': errors, 'do_not_attribute_all_repeated_errors_to_one_cause': True},
            'hypothesis_status': 'SOURCE_REVIEWED_OBSERVATION_NOT_AUTOMATICALLY_REPRODUCED',
            'qualified_failure_for_candidate_cards': qualified,
            'counterexamples': {'direct_correct_calls': direct['exact_calls'], 'direct_calls': direct['planned_calls'], 'source_correct_diagnostic_answer': source_exact, 'per_item_nonerrors_retained': True},
            'competing_explanations': ['Output sampling variation and display remapping are confounded in across-mapping differences.',
                'Structured-evidence task changes the output contract; one diagnostic response cannot identify internal reasoning.',
                'Source wording and annotation uncertainty must remain separate from Solver weakness.'],
            'targeted_design': {'status': 'REQUIRES_CARD_SPECIFIC_PREDECLARATION_BEFORE_GENERATION',
                'retrievable_feedback_ids': [], 'source_scope': 'Six public R02 scenes; full_history remains production default',
                'allowed_relations': sorted({t for item in labels for t in item['relation_tags']}),
                'evidence': 'Use the exact source-audited item spans in this record; do not label every direct error as this semantic cause.',
                'distractor_or_ordering_design': 'Preserve supported proposition and change one source-verifiable attribute; or use cross-scene occurrence ordering with explicit unique source anchors.',
                'shortcut_controls': ['Shuffle displayed labels', 'Do not expose Gold count or expected failure to Solver', 'Review all initial drafts and retain rejects'],
                'falsification': 'No target relation in source, ambiguous anchor, or target error absent on newly frozen Train cases invalidates the proposed target realization.'},
            'generation_lineage': {'actual_challenger_read': False, 'candidate_generated_from_this_feedback': False, 'planned_pilot_only': True},
            'cost': {'new_solver_calls': 0, 'new_training_runs': 0, 'review_model_tokens': None, 'review_currency_cost': None}})
    summary = {'schema_version': 'day2_feedback_summary_v1', 'created_at_utc': now(), 'records': len(records),
        'roots': len({r['root_id'] for r in records}), 'families': len({r['family_id'] for r in records}),
        'types': dict(collections.Counter(r['question_type'] for r in records)),
        'diagnostic_by_type': {kind: {'questions': sum(r['question_type'] == kind for r in records), 'valid': sum(r['question_type'] == kind and r['diagnostic_response']['valid'] for r in records), 'source_exact': sum(r['question_type'] == kind and r['diagnostic_response']['source_exact_if_valid'] is True for r in records)} for kind in ['multi_select', 'ordering']},
        'items': sum(len(r['items']) for r in records), 'item_counts': dict(counter),
        'source_gold_agree': sum(c['source_matches_stored_gold'] is True for c in compare),
        'content_holds': sum(c['content_hold'] for c in compare),
        'valid_diagnostic_responses': sum(r['diagnostic_response']['valid'] for r in records),
        'qualified_failure_records': sum(r['qualified_failure_for_candidate_cards'] for r in records),
        'uncertainty_records': sum(bool(r['uncertainties']) for r in records),
        'uncertainty_field_includes_review_limit_disclosures_not_just_content_ambiguity': True,
        'mc_diagnostic_only_item_errors': sum(i['direct_vs_diagnostic'] == 'diagnostic_only_error_direct_item_all_correct' for r in records if r['question_type'] == 'multi_select' for i in r['items']),
        'mc_direct_error_diagnostic_correct_items': sum(i['direct_vs_diagnostic'] == 'direct_error_diagnostic_item_correct' for r in records if r['question_type'] == 'multi_select' for i in r['items']),
        'non_decisive_content_warnings': sum(len(r['content_warnings']) for r in records),
        'training_wording_holds': sum(r['training_reuse_gate'] == 'HOLD_NEUTRAL_WORDING_REWRITE_AND_REVIEW' for r in records),
        'operational_reproduced_count': None, 'admission_adjudication_pending': True,
        'actual_challenger_feedback_consumption': False, 'day2_all_goals_complete': False,
        'new_solver_calls': 0, 'new_generated_questions': 0, 'new_training_runs': 0,
        'private_gold_read_for_this_build_after_review_freeze_at_utc': now(),
        'prior_builder_gold_reads_disclosed': True,
        'prior_pre_final_group02_build': 'audit/superseded_pre_final_review',
        'input_bindings': [bind(p) for p in [mappings_path, score_path, obs_path, manifest_path]],
        'review_freeze_binding': bind(HERE / 'SOURCE_REVIEW_FREEZE.json')}
    save('SOURCE_GOLD_COMPARISON.json', {'records': compare, 'gold_modified': False, 'review_precedes_comparison': True})
    save('FEEDBACK_RECORDS.json', {'schema_version': 'real_feedback_collection_v1', 'records': records})
    save('SUMMARY.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action='store_true')
    args = parser.parse_args()
    validation, reviewed = validate()
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    if args.build: build(validation, reviewed)
    sys.exit(1 if validation['errors'] else 0)
