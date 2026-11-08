"""Recompute bounded v67 outcomes from frozen assets, reviews and native responses."""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import evaluate as ev

HERE = Path(__file__).resolve().parent


def display_module():
    key = 'v67_unchanged_display_export'
    if key not in sys.modules:
        path = HERE.parent / 'h4_option_composition_v63_qa/display_export.py'
        spec = importlib.util.spec_from_file_location(key, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def distribution(questions):
    multi = [q for q in questions if q['question_type'] == 'multi_select']
    counts = Counter(len(q['gold']) for q in multi)
    roots = sorted({q['root_id'] for q in questions})
    return {'questions':len(questions),'by_type':dict(Counter(q['question_type'] for q in questions)),
        'multiselect_gold_count_histogram':{str(n):counts[n] for n in range(2,7)},
        'missing_multiselect_gold_counts':[n for n in range(2,7) if not counts[n]],
        'five_or_six_fraction':str(Fraction(counts[5]+counts[6],len(multi))) if multi else None,
        'by_root':{root:{'questions':sum(q['root_id']==root for q in questions),
            'by_type':dict(Counter(q['question_type'] for q in questions if q['root_id']==root)),
            'multiselect_gold_count_histogram':{str(n):sum(q['root_id']==root and q['question_type']=='multi_select' and len(q['gold'])==n for q in questions) for n in range(2,7)}} for root in roots}}


def planned_questions(config):
    questions=[{'candidate_id':slot['slot_id'],'root_id':slot['root_id'],'question_type':'multi_select',
                'gold':list(range(slot['initial_target_count']))} for slot in config['slots']]
    target={str(key):value for key,value in config['target_distribution'].items()}
    ev.require(distribution(questions)['multiselect_gold_count_histogram']==target,
               'planned_slots_disagree_with_configured_target_distribution')
    return questions


def statement_key(q):
    field='options' if q['question_type']=='multi_select' else 'items'
    return q['root_id'],q['question_type'],tuple(sorted(' '.join(t.casefold().split()) for t in q[field].values()))


def canonical_gold(q):
    field='options' if q['question_type']=='multi_select' else 'items'
    values=[' '.join(q[field][label].casefold().split()) for label in q['gold']]
    return tuple(sorted(values) if field=='options' else values)


def merge_exact(baseline, new):
    seen, kept, duplicates = {}, [], []
    for q in baseline + new:
        key=statement_key(q)
        if key in seen:
            original=seen[key]
            duplicates.append({'candidate_id':q['candidate_id'],'retained_candidate_id':original['candidate_id'],
                'root_id':q['root_id'],'reason':'same_root_type_normalized_full_six_statement_set',
                'gold_content_disagreement':canonical_gold(q)!=canonical_gold(original)})
        else:
            seen[key]=q
            kept.append(q)
    return kept,duplicates


def aggregate(rows):
    calls=len(rows)
    exact=sum(r['exact'] for r in rows)
    partial=sum((Fraction(r['partial']) for r in rows),Fraction())
    return {'questions':len({r['candidate_id'] for r in rows}),'calls':calls,'exact_correct':exact,
        'exact_rate':str(Fraction(exact,calls)) if calls else None,
        'exact_rate_percent':round(100*exact/calls,2) if calls else None,
        'partial_mean':str(partial/calls) if calls else None,
        'partial_mean_percent':round(100*float(partial/calls),2) if calls else None}


def compare_pairs(pairs):
    base=aggregate([{'candidate_id':p['candidate_id'],**p['base']} for p in pairs])
    trained=aggregate([{'candidate_id':p['candidate_id'],**p['trained']} for p in pairs])
    return {'base':base,'trained':trained,
        'trained_minus_base_exact_percentage_points':round(100*(float(Fraction(trained['exact_rate']))-float(Fraction(base['exact_rate']))),2) if pairs else None,
        'trained_minus_base_partial_points':str(Fraction(trained['partial_mean'])-Fraction(base['partial_mean'])) if pairs else None,
        'paired_exact':{'both_correct':sum(p['base']['exact'] and p['trained']['exact'] for p in pairs),
            'trained_only_correct':sum(not p['base']['exact'] and p['trained']['exact'] for p in pairs),
            'base_only_correct':sum(p['base']['exact'] and not p['trained']['exact'] for p in pairs),
            'both_wrong':sum(not p['base']['exact'] and not p['trained']['exact'] for p in pairs)}}


def comparison(verification, usable):
    by_side={side:{r['request_id']:r for r in verification['models'][side]['responses']} for side in ('base','trained')}
    ev.require(set(by_side['base'])==set(by_side['trained']),'unpaired_responses')
    questions={q['candidate_id']:q for q in usable}
    selected_ids={q['candidate_id'] for q in verification['models']['trained']['questions'] if q['selected_by_policy']}
    pairs=[]
    for rid,left in by_side['base'].items():
        right=by_side['trained'][rid]
        q=questions[left['candidate_id']]
        ev.require(left['candidate_id']==right['candidate_id'],'paired_question_mismatch')
        pairs.append({'request_id':rid,'candidate_id':q['candidate_id'],'root_id':q['root_id'],'gold_count':len(q['gold']),
            'base':left,'trained':right,'trained_selected':q['candidate_id'] in selected_ids})
    result={'primary_scope':'all_content_usable_questions_frozen_before_solver','primary':compare_pairs(pairs),
        'by_root':{root:compare_pairs([p for p in pairs if p['root_id']==root]) for root in sorted({q['root_id'] for q in usable})},
        'by_correct_count':{str(n):compare_pairs([p for p in pairs if p['gold_count']==n]) for n in range(2,7)},
        'selected_secondary':compare_pairs([p for p in pairs if p['trained_selected']]),
        'questions':[{'candidate_id':qid,'root_id':q['root_id'],'gold_count':len(q['gold']),
            'trained_selected':qid in selected_ids,**compare_pairs([p for p in pairs if p['candidate_id']==qid])} for qid,q in questions.items()],
        'paired_inputs':verification['paired_inputs'],'training_started':False,
        'limitation':'same_five_roots_sources_focus_as_v66; task_repair_prior_and_allocation_changed; descriptive_not_randomized_control; not_second_round_training_effect'}
    return result,pairs


def blind_all_six(questions):
    n=ev.native()
    rows=[]
    for q in questions:
        if q['question_type']!='multi_select':
            continue
        variants,_=n._variants_for(q)
        variant=variants[0]
        mapping={text:cid for cid,text in variant['content_to_text'].items()}
        gold=[mapping[q['options'][label]] for label in q['gold']]
        question={**q,'gold_content_ids':gold}
        request={'question_type':'multi_select','variant':variant}
        score=n.score_response(question,request,'<box>A,B,C,D,E,F</box>')
        independent=ev.independent_score('<box>A,B,C,D,E,F</box>',request,gold)
        ev.require(score['partial_score_fraction']==independent['partial'] and score['exact_match']==independent['exact'],'blind_score_disagreement')
        rows.append({'candidate_id':q['candidate_id'],'root_id':q['root_id'],**independent})
    return {'scope':'multiselect_only_native_scorer_equal_question_weights','aggregate':aggregate(rows),
        'by_correct_count':{str(k):aggregate([r for r in rows if r['gold_count']==k]) for k in range(2,7)},
        'questions':rows,'formula':'TP/k - FP/(6-k); k=6 has no false-positive term',
        'all_six_partial_is_zero_for_gold_two_to_five':True}


def cost_summary(requests, usable_count=None):
    actual=[r for r in requests if r['remote_http_attempt']]
    known=[r for r in actual if r['token_usage_status']=='known']
    complete=len(known)==len(actual)
    inputs=sum(r['reported_input_tokens'] for r in known)
    outputs=sum(r['reported_output_tokens'] for r in known)
    return {'http_attempts':len(actual),'prepared_requests':len(requests),'known_usage_http_attempts':len(known),
        'unknown_usage_http_attempts':len(actual)-len(known),'usage_complete':complete,
        'input_tokens':inputs if complete else 'unknown','output_tokens':outputs if complete else 'unknown',
        'known_input_token_subtotal':inputs,'known_output_token_subtotal':outputs,
        'money_cost':'unknown_no_frozen_provider_pricing','standalone_planner_calls':0,
        'content_usable_questions':usable_count,
        'http_per_content_usable_question':str(Fraction(len(actual),usable_count)) if usable_count else None}


def fact_field_statistics(facts):
    """Count actual fields only; a nonempty reason is not a semantic endorsement."""
    rows = [fact for fact in facts if isinstance(fact, dict)]
    nulls = [fact for fact in rows if 'contradicted' in fact and fact['contradicted'] is None]
    reason_key = 'contradiction_unavailable_reason'
    reasons = [fact[reason_key] for fact in nulls
               if isinstance(fact.get(reason_key), str) and fact[reason_key].strip()]
    def reason_invalid(fact):
        if reason_key not in fact:
            return False
        reason = fact[reason_key]
        nonempty = isinstance(reason, str) and bool(reason.strip())
        if 'contradicted' in fact and fact['contradicted'] is None:
            return not nonempty
        if isinstance(fact.get('contradicted'), dict):
            return reason is not None
        return reason is not None and not nonempty
    return {'fact_entries': len(facts), 'object_facts': len(rows),
        'supported_versions': sum(isinstance(fact.get('supported'), dict) for fact in rows),
        'contradicted_versions': sum(isinstance(fact.get('contradicted'), dict) for fact in rows),
        'contradicted_explicit_null': len(nulls),
        'contradicted_field_missing': sum('contradicted' not in fact for fact in rows),
        'contradicted_invalid_nonnull': sum('contradicted' in fact and fact['contradicted'] is not None
                                          and not isinstance(fact['contradicted'], dict) for fact in rows),
        'reason_field_missing_all_facts': sum(reason_key not in fact for fact in rows),
        'reason_field_invalid_all_facts': sum(reason_invalid(fact) for fact in rows),
        'reason_condition_unknown_all_facts': sum('contradicted' not in fact or
            (fact['contradicted'] is not None and not isinstance(fact['contradicted'], dict)) for fact in rows),
        'reason_all_facts_definition': 'Missing counts absent fields on all object facts. Invalid counts present values violating null/nonempty-string type or the condition of a valid contradicted field. Unknown condition is recorded separately when contradicted is missing or invalid. Counts are mechanical, not semantic.',
        'null_reason_nonempty': len(reasons),
        'null_reason_field_missing': sum(reason_key not in fact for fact in nulls),
        'null_reason_empty_null_or_invalid': sum(reason_key in fact and
            not (isinstance(fact[reason_key], str) and fact[reason_key].strip()) for fact in nulls),
        'null_reason_verbatim_histogram': dict(sorted(Counter(reasons).items())),
        'null_reasons_semantically_verified': False,
        'unknown_or_missing_reason_is_not_counterevidence': True}


def evidence_statistics(home, config):
    """Separate all raw response attempts, accepted assets and independent reviews."""
    home = Path(home)
    raw_facts, accepted_facts, raw_rows, slot_rows, reviewed = [], [], [], [], []
    for slot in config['slots']:
        directory = home / 'assets' / slot['slot_id']
        local_raw = []
        for path in sorted((directory / 'requests').glob('*/RESPONSE.json')):
            response = ev.read(path)
            facts, status = [], 'UNPARSEABLE_JSON'
            try:
                value = json.loads(response.get('text', ''))
                if isinstance(value, dict) and isinstance(value.get('facts'), list):
                    facts, status = value['facts'], 'FACT_LIST_PRESENT'
                else:
                    status = 'FACT_LIST_MISSING_OR_INVALID'
            except (TypeError, ValueError):
                pass
            local_raw.extend(facts)
            raw_rows.append({'slot_id': slot['slot_id'], 'root_id': slot['root_id'],
                'response': ev.bind(path), 'parse_status': status,
                'field_statistics': fact_field_statistics(facts)})
        accepted_path = directory / 'ASSETS.json'
        facts = ev.read(accepted_path)['facts'] if accepted_path.exists() else []
        review_path = directory / 'ASSET_REVIEW.json'
        review = ev.read(review_path)['versions'] if review_path.exists() else []
        reviewed.extend(review)
        raw_facts.extend(local_raw)
        accepted_facts.extend(facts)
        slot_rows.append({'slot_id': slot['slot_id'], 'root_id': slot['root_id'],
            'accepted_assets_present': accepted_path.exists(),
            'raw_attempt_fields': fact_field_statistics(local_raw),
            'accepted_asset_fields': fact_field_statistics(facts),
            'usable_supported_versions': sum(v.get('usable') is True and v.get('relation') == 'supports' for v in review),
            'usable_contradicted_versions': sum(v.get('usable') is True and v.get('relation') == 'contradicts' for v in review),
            'reviewed_unknown_versions': sum(v.get('relation') == 'unknown' for v in review),
            'reviewed_unusable_versions': sum(v.get('usable') is not True for v in review)})
    return {'scope': 'raw_attempts_include_retries_and_rejected_responses; accepted_assets_and_reviews_count_each_slot_once',
        'raw_response_count': len(raw_rows),
        'raw_response_parse_statuses': dict(Counter(row['parse_status'] for row in raw_rows)),
        'raw_attempt_fields': fact_field_statistics(raw_facts),
        'accepted_asset_fields': fact_field_statistics(accepted_facts),
        'usable_supported_versions': sum(v.get('usable') is True and v.get('relation') == 'supports' for v in reviewed),
        'usable_contradicted_versions': sum(v.get('usable') is True and v.get('relation') == 'contradicts' for v in reviewed),
        'reviewed_unknown_versions': sum(v.get('relation') == 'unknown' for v in reviewed),
        'reviewed_unusable_versions': sum(v.get('usable') is not True for v in reviewed),
        'null_reason_interpretation': 'Verbatim writer explanations only; nonempty, unknown or missing reasons do not establish contradiction. Usability comes only from source-first content review.',
        'requests': raw_rows, 'slots': slot_rows}


def mechanical_repair_statistics(home, config):
    """Observe recorded repair prompts/outcomes; missing evidence stays unknown."""
    rows, repeated, groups = [], [], {}
    for slot in config['slots']:
        directory = Path(home) / 'assets' / slot['slot_id']
        prior = None
        for request_path in sorted((directory / 'requests').glob('*/REQUEST.json')):
            folder = request_path.parent
            request = ev.read(request_path)
            context, context_status = None, 'INPUT_JSON_UNAVAILABLE'
            try:
                tail = request['messages'][0]['content'].split('\nINPUT_JSON\n', 1)[1]
                context = json.JSONDecoder().raw_decode(tail)[0]
                context_status = 'PARSED' if isinstance(context, dict) else 'INPUT_JSON_NOT_OBJECT'
            except (KeyError, IndexError, TypeError, ValueError):
                pass
            repair = context.get('mechanical_repair') if isinstance(context, dict) else None
            repair_present = isinstance(repair, dict) if context_status == 'PARSED' else None
            response_path, validation_path = folder / 'RESPONSE.json', folder / 'VALIDATION.json'
            response = ev.read(response_path) if response_path.exists() else {}
            raw = response.get('text')
            validation = ev.read(validation_path) if validation_path.exists() else None
            errors = validation.get('mechanical_errors') if isinstance(validation, dict) else None
            errors = errors if isinstance(errors, list) else None
            outcome = 'PASSED' if errors == [] else 'FAILED' if errors else 'UNKNOWN_NO_VALIDATION'
            text_hash = hashlib.sha256(raw.encode()).hexdigest() if isinstance(raw, str) else None
            hints = repair.get('reference_hints') if repair_present else None
            hint_count = len(hints) if isinstance(hints, list) else 0 if repair_present and 'reference_hints' not in repair else None
            row = {'slot_id': slot['slot_id'], 'root_id': slot['root_id'], 'attempt_folder': folder.name,
                'request': ev.bind(request_path), 'context_status': context_status,
                'remote_http_attempt': (folder / 'HTTP_ATTEMPT.json').exists(),
                'repair_prompt_present': repair_present,
                'repair_errors': repair.get('errors') if repair_present else None,
                'reference_hint_count': hint_count,
                'reference_hints': hints if isinstance(hints, list) else None,
                'previous_response_matches_recorded_raw': repair.get('previous_response') == prior['raw']
                    if repair_present and prior is not None and isinstance(prior['raw'], str) else None,
                'prior_mechanical_errors': prior['errors'] if prior is not None else None,
                'repair_errors_match_prior_validation': repair.get('errors') == prior['errors']
                    if repair_present and prior is not None and prior['errors'] is not None else None,
                'validation_outcome': outcome, 'mechanical_errors': errors,
                'response': ev.bind(response_path) if response_path.exists() else None,
                'validation': ev.bind(validation_path) if validation_path.exists() else None,
                'raw_text_sha256': text_hash, 'transport_error_recorded': (folder / 'ERROR.json').exists()}
            rows.append(row)
            if isinstance(raw, str):
                groups.setdefault(text_hash, []).append({'slot_id': slot['slot_id'], 'attempt_folder': folder.name,
                    'response': ev.bind(response_path)})
                if prior is not None and isinstance(prior['raw'], str):
                    repeated.append({'slot_id': slot['slot_id'], 'first_attempt': prior['folder'],
                        'second_attempt': folder.name, 'raw_text_exactly_equal': raw == prior['raw'],
                        'first_raw_text_sha256': prior['text_hash'], 'second_raw_text_sha256': text_hash})
            prior = {'raw': raw, 'errors': errors, 'folder': folder.name, 'text_hash': text_hash}
    repair_rows = [row for row in rows if row['repair_prompt_present'] is True]
    remote_repairs = [row for row in repair_rows if row['remote_http_attempt']]
    return {'scope': 'recorded_INPUT_JSON.mechanical_repair_and_validation; exact_raw_text_comparison_only',
        'causal_repair_effect_claimed': False, 'prepared_requests': len(rows),
        'repair_prompt_requests': len(repair_rows), 'repair_http_attempts': len(remote_repairs),
        'repair_passed': sum(row['validation_outcome'] == 'PASSED' for row in remote_repairs),
        'repair_failed': sum(row['validation_outcome'] == 'FAILED' for row in remote_repairs),
        'repair_outcome_unknown': sum(row['validation_outcome'] == 'UNKNOWN_NO_VALIDATION' for row in remote_repairs),
        'repair_requests_with_reference_hints': sum(bool(row['reference_hint_count']) for row in repair_rows),
        'reference_hint_count': sum(row['reference_hint_count'] or 0 for row in repair_rows),
        'hinted_repair_passed': sum(bool(row['reference_hint_count']) and row['validation_outcome'] == 'PASSED' for row in remote_repairs),
        'comparable_within_slot_response_pairs': len(repeated),
        'exact_repeated_within_slot_response_pairs': sum(row['raw_text_exactly_equal'] for row in repeated),
        'exact_repeated_response_groups': [{'raw_text_sha256': key, 'responses': value}
            for key, value in sorted(groups.items()) if len(value) > 1],
        'requests': rows, 'within_slot_response_pairs': repeated,
        'interpretation': 'Pass/fail follows recorded mechanical validation after a repair request. Missing prompt, response or validation evidence is not counted as a successful repair. Exact repetition does not diagnose provider caching. These observations do not establish causal repair effectiveness.'}


def descriptive_previous_comparison(home, config, summary):
    previous_home = Path(config['baseline_pool']).parent
    old_config = ev.read(previous_home / 'CONFIG.json')
    previous = ev.read(previous_home / 'RESULT_SUMMARY.json')
    old_slots = {slot['slot_id']: slot for slot in old_config['slots']}
    same_slots = set(old_slots) == {slot['slot_id'] for slot in config['slots']}
    slot_checks = []
    prior_rows = []
    for slot in config['slots']:
        old = old_slots.get(slot['slot_id'], {})
        slot_checks.append({'slot_id': slot['slot_id'],
            'same_root_and_source_sha256': slot['root_id'] == old.get('root_id') and slot['history_sha256'] == old.get('history_sha256'),
            'same_focus': slot['focus'] == old.get('focus'),
            'same_initial_target_count': slot['initial_target_count'] == old.get('initial_target_count')})
        def prior_ids(paths):
            return {q['candidate_id'] for path in paths for q in ev.read(path)['questions']}
        old_ids, new_ids = prior_ids(old.get('prior_pool_paths', [])), prior_ids(slot['prior_pool_paths'])
        prior_rows.append({'slot_id': slot['slot_id'], 'previous_count': len(old_ids), 'current_count': len(new_ids),
            'added_candidate_ids': sorted(new_ids - old_ids), 'removed_candidate_ids': sorted(old_ids - new_ids),
            'previous_files': [ev.bind(path) for path in old.get('prior_pool_paths', [])],
            'current_files': [ev.bind(path) for path in slot['prior_pool_paths']]})
    old_evidence = evidence_statistics(previous_home, old_config)
    def metrics(result, evidence):
        return {**{key: result[key] for key in ('assembled_count', 'content_usable_count', 'known_issue_count',
                'selected_count', 'merged_count', 'calls_per_model')},
            'http_attempts': result['generation_cost']['http_attempts'],
            'raw_contradicted_versions': evidence['raw_attempt_fields']['contradicted_versions'],
            'accepted_contradicted_versions': evidence['accepted_asset_fields']['contradicted_versions'],
            'usable_contradicted_versions': evidence['usable_contradicted_versions'],
            'accepted_contradicted_explicit_null': evidence['accepted_asset_fields']['contradicted_explicit_null'],
            'accepted_null_reason_nonempty': evidence['accepted_asset_fields']['null_reason_nonempty'],
            'accepted_null_reason_field_missing': evidence['accepted_asset_fields']['null_reason_field_missing'],
            'raw_reason_field_missing_all_facts': evidence['raw_attempt_fields']['reason_field_missing_all_facts'],
            'raw_reason_field_invalid_all_facts': evidence['raw_attempt_fields']['reason_field_invalid_all_facts'],
            'accepted_reason_field_missing_all_facts': evidence['accepted_asset_fields']['reason_field_missing_all_facts'],
            'accepted_reason_field_invalid_all_facts': evidence['accepted_asset_fields']['reason_field_invalid_all_facts'],
            'distributions': result['distributions'],
            'base_trained_primary': result['base_trained_comparison']['primary']}
    before, after = metrics(previous, old_evidence), metrics(summary, summary['evidence_statistics'])
    numeric_keys = [key for key, value in before.items() if type(value) is int]
    return {'comparison_kind': 'descriptive_sequential_batches', 'strict_randomized_control': False,
        'causal_prompt_effect_claimed': False, 'same_slot_set': same_slots,
        'same_sources_and_focus': same_slots and all(row['same_root_and_source_sha256'] and row['same_focus'] for row in slot_checks),
        'slot_checks': slot_checks, 'historical_prior_changes': prior_rows,
        'previous_summary': ev.bind(previous_home / 'RESULT_SUMMARY.json'),
        'previous_config': ev.bind(previous_home / 'CONFIG.json'),
        'previous_evidence_statistics': old_evidence,
        'previous_mechanical_repair_statistics': mechanical_repair_statistics(previous_home, old_config),
        'allocation_config': {'previous_target_distribution': old_config.get('target_distribution'),
            'current_target_distribution': config.get('target_distribution'),
            'previous_priority_counts': old_config.get('allocation_priority_counts'),
            'current_priority_counts': config.get('allocation_priority_counts')},
        'v66': before, 'v67': after,
        'v67_minus_v66_counts': {key: after[key] - before[key] for key in numeric_keys},
        'limitation': 'Same five roots, frozen original-plus-R01 sources and focus; v67 changes task instructions, mechanical-repair prompts, historical priors and allocation priorities/targets. Different generated/reviewed sets and sequential sampling prevent a strict randomized or causal prompt-effect comparison; no second-round training occurred.'}


def generation_records(home, config):
    import asset_generation
    requests,slots=[],[]
    for slot in config['slots']:
        directory=Path(home)/'assets'/slot['slot_id']
        asset_generation.verify(directory,complete=True)
        complete=ev.read(directory/'GENERATION.COMPLETE.json')
        rows=[]
        for request_path in sorted((directory/'requests').glob('*/REQUEST.json')):
            folder=request_path.parent
            remote=(folder/'HTTP_ATTEMPT.json').exists()
            response=ev.read(folder/'RESPONSE.json') if (folder/'RESPONSE.json').exists() else {}
            metadata=response.get('metadata',{})
            usage=metadata.get('usage') or {}
            inputs=metadata.get('input_tokens',usage.get('prompt_tokens'))
            outputs=metadata.get('output_tokens',usage.get('completion_tokens'))
            error=ev.read(folder/'ERROR.json') if (folder/'ERROR.json').exists() else None
            row={'slot_id':slot['slot_id'],'root_id':slot['root_id'],'request':ev.bind(request_path),
                'remote_http_attempt':remote,'reported_input_tokens':inputs,'reported_output_tokens':outputs,
                'token_usage_status':('known' if type(inputs) is int and type(outputs) is int else 'unknown') if remote else 'not_called',
                'token_usage_source':'provider_response_metadata_no_estimation','error':error}
            rows.append(row)
        ev.require(len(rows)<=2 and len(rows)==complete['requests_prepared']
                   and sum(r['remote_http_attempt'] for r in rows)==complete['http_attempts'],
                   'asset_request_budget_or_count_mismatch')
        assets=ev.read(directory/'ASSETS.json') if (directory/'ASSETS.json').exists() else None
        facts=assets.get('facts',[]) if assets else []
        slots.append({'slot_id':slot['slot_id'],'root_id':slot['root_id'],'initial_target_count':slot['initial_target_count'],
            'generation_complete':complete,'requests':len(rows),'http_attempts':sum(r['remote_http_attempt'] for r in rows),
            'asset_fact_groups':len(facts),'asset_supported_versions':sum(bool(f.get('supported')) for f in facts),
            'asset_contradicted_versions':sum(bool(f.get('contradicted')) for f in facts),
            'asset_review':ev.read(directory/'ASSET_REVIEW.json') if (directory/'ASSET_REVIEW.json').exists() else None})
        requests.extend(rows)
    ev.require(sum(r['remote_http_attempt'] for r in requests)<=config['max_http'],'total_asset_http_budget')
    return requests,slots



def reference_normalization_summary(home, config):
    import asset_generation
    rows=[]
    for slot in config['slots']:
        run=Path(home)/'assets'/slot['slot_id']
        manifest=ev.read(run/'MANIFEST.json');source=ev.read(run/'SOURCE.json')
        for path in sorted((run/'requests').glob('*/REFERENCE_NORMALIZATION.json')):
            receipt=ev.read(path);response=ev.read(path.parent/'RESPONSE.json')
            _,strict_errors=asset_generation._validate_assets_exact(response['text'],manifest,source)
            rows.append({'slot_id':slot['slot_id'],'receipt':ev.bind(path),
                'strict_original_ref_errors':strict_errors,
                'strict_passed':not strict_errors,'normalized_passed':receipt['accepted'],
                'normalized_ref_count':receipt['normalized_ref_count'],
                'rescued_by_reference_normalization':bool(strict_errors) and receipt['accepted']})
    return {'requests':rows,'strict_passed':sum(r['strict_passed'] for r in rows),
        'normalized_passed':sum(r['normalized_passed'] for r in rows),
        'rescued_responses':sum(r['rescued_by_reference_normalization'] for r in rows),
        'normalized_ref_count':sum(r['normalized_ref_count'] for r in rows),
        'semantic_fields_modified':False}


def source_summary(home,config):
    inventory=ev.read(Path(home)/'ROOT_INVENTORY.json')
    roots={r['root_id']:r for r in inventory['roots']}
    out=[]
    for root in sorted({s['root_id'] for s in config['slots']}):
        row=roots[root]
        ev.verify_binding(row['history'])
        history=ev.read(row['history']['path'])
        ev.require(len(history['turns'])==row['full_history']['turns'],'source_turn_count_changed')
        out.append({'root_id':root,'history':row['history'],'full_history':row['full_history'],
            'formal_original_plus_R01':row['formal_original_plus_R01'],'retained_by_route':row['retained_by_route'],
            'body_token_definition':row['body_token_definition'],'new_expansion_tokens':0})
    return out


def export_questions(home,stem,questions,title,provenance):
    display=display_module()
    views=[display.display_view(q) for q in questions]
    for q,view in zip(questions,views): display.replay_view(q,view)
    ev.write(Path(home)/(stem+'_CANDIDATES.json'),{'questions':questions,'candidate_count':len(questions),
        'original_question_and_gold_unchanged':True,'training_allowed':False,'formal_training_admission':False,'provenance':provenance})
    ev.write(Path(home)/(stem+'_DISPLAY.json'),{'questions':views,'gold_blind_permutation_replayed':True})
    with (Path(home)/(stem+'_QUESTIONS.md')).open('x') as stream:
        stream.write(display.markdown(views,title=title)+'\n')


def report_markdown(summary):
    primary=summary['base_trained_comparison']['primary']
    b,t=primary['base'],primary['trained']
    evidence=summary['evidence_statistics']
    raw,accepted=evidence['raw_attempt_fields'],evidence['accepted_asset_fields']
    lines=['# v67 引用与机械修复试验结果','',
        f"{summary['planned_slots']} 个名额，组装 {summary['assembled_count']} 题，内容可用 {summary['content_usable_count']} 题，原难度规则入选 {summary['selected_count']} 题。与原{summary['baseline_count']}题合并后 {summary['merged_count']} 题。",
        '每题从六个不同事实中各取一个已核实命题版本，Gold由实际支持关系导出。源对话未扩充；当前只评估内容可用题。','',
        '| 分布口径 | 2项 | 3项 | 4项 | 5项 | 6项 |','|---|---:|---:|---:|---:|---:|']
    for label,key in [('起始构造目标','planned'),('已组装','assembled'),('内容可用新题','content_usable'),('难度筛选入选','selected'),('合并可用候选','merged')]:
        hist=summary['distributions'][key]['multiselect_gold_count_histogram']
        lines.append('| '+label+' | '+' | '.join(str(hist[str(n)]) for n in range(2,7))+' |')
    lines.extend(['',f"最终新增入池尚缺数量：{summary['distributions']['selected']['missing_multiselect_gold_counts']}；合并池尚缺：{summary['distributions']['merged']['missing_multiselect_gold_counts']}。",
        '', '| 内容可用集主比较 | Base | 已有训练检查点 |','|---|---:|---:|',
        f"| 完全正确 | {b['exact_correct']}/{b['calls']} | {t['exact_correct']}/{t['calls']} |",
        f"| 部分得分均值 | {b['partial_mean']} | {t['partial_mean']} |",'',
        '主比较使用评分前冻结的全部内容可用题。两模型输入、随机呈现、seed与评分一致；难度筛选结果单列，第二轮训练未启动。',
        f"生成 HTTP {summary['generation_cost']['http_attempts']} 次；每模型实际 {summary['calls_per_model']} 次。提供方输入/输出 tokens：{summary['generation_cost']['input_tokens']}/{summary['generation_cost']['output_tokens']}。",
        '原生部分得分为 TP/k − FP/(6−k)，k=6不计后一项；盲选六项对2–5真题得0分、全6真题得1分。',
        f"评估状态：{summary['evaluation_status']}。内容可用数为零时跳过两模型，调用数为零，比例记为未知。",'',
        '| 反述统计口径 | 原始响应全部尝试 | 接受的素材 | 内容审核可用 |','|---|---:|---:|---:|',
        f"| 非 null 反述版本 | {raw['contradicted_versions']} | {accepted['contradicted_versions']} | {evidence['usable_contradicted_versions']} |",
        f"| 显式 contradicted=null | {raw['contradicted_explicit_null']} | {accepted['contradicted_explicit_null']} | 不作为反述 |",
        f"| null 的非空理由 | {raw['null_reason_nonempty']} | {accepted['null_reason_nonempty']} | 不作语义认证 |",
        f"| null 的理由字段缺失 | {raw['null_reason_field_missing']} | {accepted['null_reason_field_missing']} | 不作为反证 |",'',
        f"所有事实的 reason 字段缺失：原始响应 {raw['reason_field_missing_all_facts']}，接受素材 {accepted['reason_field_missing_all_facts']}；字段已存在但无效：原始响应 {raw['reason_field_invalid_all_facts']}，接受素材 {accepted['reason_field_invalid_all_facts']}。这些计数独立于上表的 null-only 缺失数。",
        '原始响应计数包含失败与重试；接受素材按每slot最终文件计一次。理由只按实际 contradiction_unavailable_reason 字段记录原文频次，未知、缺失或非空理由均不等于已获反证。反述可用性只来自独立内容审核。',
        '', '| 与 v66 的描述性比较 | v66 | v67 |','|---|---:|---:|'])
    descriptive=summary['descriptive_v66_comparison']
    for title,key in [('接受的反述版本','accepted_contradicted_versions'),('可用反述版本','usable_contradicted_versions'),
                      ('接受素材的null数','accepted_contradicted_explicit_null'),('内容可用题','content_usable_count'),
                      ('原难度规则入选题','selected_count'),('生成HTTP次数','http_attempts')]:
        lines.append(f"| {title} | {descriptive['v66'][key]} | {descriptive['v67'][key]} |")
    lines.extend(['',
        f"机械repair实际HTTP {summary['mechanical_repair_statistics']['repair_http_attempts']} 次；后续校验通过 {summary['mechanical_repair_statistics']['repair_passed']}，失败 {summary['mechanical_repair_statistics']['repair_failed']}，未知 {summary['mechanical_repair_statistics']['repair_outcome_unknown']}。其中带唯一引用提示的请求 {summary['mechanical_repair_statistics']['repair_requests_with_reference_hints']}，提示共 {summary['mechanical_repair_statistics']['reference_hint_count']} 条。同slot可比较响应对 {summary['mechanical_repair_statistics']['comparable_within_slot_response_pairs']}，原文完全重复 {summary['mechanical_repair_statistics']['exact_repeated_within_slot_response_pairs']}。以上是记录观察，不证明repair因果效果或提供方缓存。",
        'v67沿用v66的五个root、原始＋R01源及focus；TASK、机械repair提示、历史prior及分配目标/优先级均有变化。本次是顺序批次的描述性比较，不能声称严格随机对照或将差异单独归因于提示词。当前批次结束，不自动追加补题，不表示第二轮训练收益。','',
        '[新增入选题](SELECTED_QUESTIONS.md) · [合并候选题](MERGED_USABLE_QUESTIONS.md) · [逐题模型比较](BASE_TRAINED_COMPARISON.json) · [实际覆盖](COVERAGE.json) · [反述和null统计](EVIDENCE_STATISTICS.json) · [机械修复记录](MECHANICAL_REPAIR_STATISTICS.json) · [v66描述性比较](DESCRIPTIVE_V66_COMPARISON.json)'])
    return '\n'.join(lines)+'\n'


def finalize(home=HERE):
    home=Path(home)
    ev.require(not (home/'RESULT_SUMMARY.json').exists(),'final_report_already_exists')
    verification=ev.verify(home)
    pool,usable,statuses,_=ev.reviewed_content(home)
    config=ev.read(home/'CONFIG.json')
    generation_complete=ev.read(home/'GENERATION.COMPLETE.json')
    ev.require(generation_complete['training_allowed'] is False and generation_complete['automatic_followup'] is False,
               'generation_completion_scope_changed')
    ev.verify_binding(generation_complete['plan'])
    for item in ev.read(generation_complete['plan']['path'])['files']: ev.verify_binding(item)
    assembled=pool['questions']
    selected_ids={q['candidate_id'] for q in verification['models']['trained']['questions'] if q['selected_by_policy']}
    selected=[q for q in usable if q['candidate_id'] in selected_ids]
    old=ev.read(config['baseline_pool'])
    baseline=old['questions']
    ev.require(len(baseline)==68 and len({q['candidate_id'] for q in baseline})==68,'baseline68_changed')
    previous_bindings=ev.read(config['baseline_result_bindings'])['files']
    for binding in previous_bindings: ev.verify_binding(binding)
    ev.require(ev.bind(config['baseline_pool']) in previous_bindings,'baseline_pool_not_previously_bound')
    merged,duplicates=merge_exact(baseline,selected)
    ev.require(not any(r['gold_content_disagreement'] for r in duplicates),'duplicate_gold_content_disagreement')
    compare,pairs=comparison(verification,usable)
    requests,slots=generation_records(home,config)
    ev.require(generation_complete['http_attempts']==sum(r['remote_http_attempt'] for r in requests),
               'aggregate_generation_http_count_mismatch')
    ev.require({s['slot_id'] for s in generation_complete['slots']}=={s['slot_id'] for s in config['slots']},
               'generation_completion_slot_coverage')
    for slot in generation_complete['slots']: ev.verify_binding(slot['completion'])
    # The asset/assembly implementation owns exact raw-response and review reconstruction.
    generation_audit={'status':'PASS','slots_verified':len(config['slots']),
        'exact_request_feedback_source_and_asset_projection_replayed':True,
        'models_called_by_audit':0}
    allocation=ev.read(home/'ALLOCATION.json')
    planned=planned_questions(config)
    distributions={name:distribution(rows) for name,rows in [('planned',planned),('assembled',assembled),
        ('content_usable',usable),('selected',selected),('baseline',baseline),('merged',merged)]}
    summary={'version':'v67_verified_composition_results_v1','planned_slots':len(config['slots']),
        'baseline_count':len(baseline),'target_distribution':config['target_distribution'],
        'allocation_priority_counts':config.get('allocation_priority_counts',[]),
        'assembled_count':len(assembled),'content_usable_count':len(usable),'known_issue_count':len(assembled)-len(usable),
        'selected_count':len(selected),'merged_count':len(merged),'exact_merge_duplicates':duplicates,
        'evaluation_status':verification['evaluation_status'],
        'calls_per_model':verification['calls_per_model'],'distributions':distributions,'allocation':allocation,
        'content_statuses':statuses,'generation_cost':cost_summary(requests,len(usable)),
        'generation_slots':slots,'base_trained_comparison':compare,'source_summaries':source_summary(home,config),
        'new_root_ids':sorted({s['root_id'] for s in config['slots']}),'source_scope':'full_original_plus_R01_no_new_AB',
        'root_only_causal_comparison':False,'reference_normalization':reference_normalization_summary(home,config),
        'new_expansion_tokens':0,'training_allowed':False,'training_started':False,'automatic_followup':False,
        'previous_bound_files_unchanged':len(previous_bindings),
        'generation_audit':generation_audit,'raw_native_evaluation_replay_status':verification['status'],
        'evidence_statistics':evidence_statistics(home,config),
        'mechanical_repair_statistics':mechanical_repair_statistics(home,config)}
    summary['descriptive_v66_comparison']=descriptive_previous_comparison(home,config,summary)
    provenance={'requested_model':pool['provenance']['requested_model'],'baseline':ev.bind(config['baseline_pool']),
        'assembly':ev.bind(home/'ASSEMBLY_RECEIPTS.json'),'content_review':ev.bind(home/'FINAL_CONTENT_REVIEW.COMPLETE.json')}
    for stem,rows,title in [('ASSEMBLED',assembled,'v67 全部组装题'),('CONTENT_USABLE',usable,'v67 内容可用题'),
                           ('SELECTED',selected,'v67 原难度规则入选题'),('MERGED_USABLE',merged,'累计可用候选题')]:
        export_questions(home,stem,rows,title,provenance)
    ev.write(home/'COVERAGE.json',{'distributions':distributions,'allocation':allocation,'target_is_not_actual_coverage':True})
    ev.write(home/'GENERATION_COST.json',{'summary':summary['generation_cost'],'requests':requests})
    ev.write(home/'EVIDENCE_STATISTICS.json',summary['evidence_statistics'])
    ev.write(home/'MECHANICAL_REPAIR_STATISTICS.json',summary['mechanical_repair_statistics'])
    ev.write(home/'DESCRIPTIVE_V66_COMPARISON.json',summary['descriptive_v66_comparison'])
    ev.write(home/'BASE_TRAINED_COMPARISON.json',compare)
    ev.write(home/'PAIRED_RAW_SCORES.json',{'pairs':pairs})
    ev.write(home/'BLIND_ALL_SIX_BASELINE.json',{name:blind_all_six(rows) for name,rows in
        [('baseline',baseline),('content_usable',usable),('selected',selected),('merged',merged)]})
    ev.write(home/'RAW_RECOMPUTATION_AUDIT.json',{'status':'PASS','audit_kind':'raw_recomputation_by_implementation_author',
        'models_called':0,'frozen_files_modified':0,'generation':generation_audit,
        'evaluation':verification,'previous_bound_files_unchanged':len(previous_bindings)})
    ev.write(home/'RESULT_SUMMARY.json',summary)
    with (home/'REPORT.md').open('x') as stream: stream.write(report_markdown(summary))
    files=[p for p in home.rglob('*') if p.is_file() and '__pycache__' not in p.parts
           and p.name!='RESULT_BINDINGS.json' and not p.name.endswith('.log')]
    files += [Path(config['baseline_pool']),Path(config['baseline_result_bindings'])]
    files += [Path(s['history_path']) for s in config['slots']]
    files += [Path(p) for s in config['slots'] for p in s['prior_pool_paths']]
    files += [Path(row['path']) for row in ev.read(home/'EVALUATION_PREPARATION.COMPLETE.json')['files']]
    ev.write(home/'RESULT_BINDINGS.json',{'scope':'all_v67_final_artifacts_and_direct_frozen_inputs_except_mutable_process_logs',
        'files':[ev.bind(p) for p in sorted(set(files))]})
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=HERE)
    args=parser.parse_args()
    result=finalize(args.report)
    print(json.dumps({k:result[k] for k in ('assembled_count','content_usable_count','selected_count','merged_count','calls_per_model')},indent=2))
