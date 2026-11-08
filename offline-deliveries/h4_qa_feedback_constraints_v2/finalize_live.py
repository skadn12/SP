"""Aggregate original responses and Codex offline diagnoses; no model calls."""
from collections import Counter
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
WORKTREE = Path('/root/autodl-tmp/gxy-selfplay/worktrees/h4-gate-downstream-v1')
sys.path.insert(0, str(WORKTREE))
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.qa_difficulty_feedback_run import check_freeze, ROOTS

RUN = BASE / 'prepared_run/live'
PREVIOUS = BASE.parent / 'h4_qa_difficulty_feedback_v1'
ERRORS = ('event_state', 'actor', 'event_identity', 'citation_support',
          'reversed_temporal_edge', 'unsupported_time_relation', 'answer_set_or_scope')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def counts(rows, key='question_type'):
    return dict(Counter(r[key] for r in rows))


def exact_key(q):
    items = q.get('items') or {n['item_id']: n['item_text'] for n in q.get('event_nodes', [])}
    return json.dumps([q['question_type'], q['question'],
                       sorted((q.get('options') or {}).values()),
                       sorted(items.values())], ensure_ascii=False)


def usage(records):
    result = {}
    for field in ('input_tokens', 'output_tokens', 'total_tokens', 'reasoning_tokens'):
        values = [r.get(field, 'UNKNOWN') for r in records]
        known = [v for v in values if type(v) is int]
        result[field] = sum(known) if len(known) == len(values) else 'UNKNOWN'
        result[field + '_known_subtotal'] = sum(known)
        result[field + '_unknown_calls'] = len(values) - len(known)
    result['verified_currency_cost'] = 'UNKNOWN'
    return result


def per_item(total, n):
    if not n:
        return {'value': None, 'reason': 'ZERO_ELIGIBLE_ITEMS'}
    return {'value': total / n if type(total) is int else 'UNKNOWN', 'denominator': n}


def error_counts(rows):
    return {kind: {'confirmed_rows': sum(r.get('error_flags', {}).get(kind) is True for r in rows),
                   'unknown_or_unclassified_rows': sum(r.get('error_flags', {}).get(kind, 'UNKNOWN') == 'UNKNOWN' for r in rows)}
            for kind in ERRORS}


def main():
    manifest = read(BASE / 'prepared_run/freeze_manifest.json')
    check_freeze(manifest)
    check_freeze(read(PREVIOUS / 'prepared_run/freeze_manifest.json'))
    execution = read(RUN / 'generation_summary.json')
    assert execution['status'] in ('FINISHED', 'STOPPED')
    audit = read(BASE / 'reports/EXECUTION_AUDIT.json')
    assert audit['status'] == 'PASS', 'execution audit must pass before finalization'
    ledger = read(RUN / 'execution/qa_call_ledger.json')
    assert ledger['http_attempt_count'] <= 21 and ledger['logical_call_count'] <= 21
    assert ledger['automatic_retry_count'] == 0
    previous = read(PREVIOUS / 'reports/FINAL_REPORT.json')
    roots, all_rows, final_rows_all, strict_all, new_all = [], [], [], [], []
    all_primary_quota, all_abstentions, exact_repeats = [], [], []
    checked_quotes = 0
    for root in ROOTS:
        code = root.rsplit('-', 1)[1]
        directory = RUN / 'roots' / root
        root_calls = [r for r in ledger['records'] if r['root_id'] == root]
        if not (directory / 'candidate_pool.json').exists():
            roots.append({'root_id': root, 'status': 'NOT_COMPLETED', 'calls': counts(root_calls, 'stage'),
                          'usage': usage(root_calls), 'content_evaluated': False})
            continue
        report = read(BASE / f'reports/CONTENT_{code}.json')
        rows = report['raw_candidate_reviews']
        pool = read(directory / 'candidate_pool.json')
        final = {q['candidate_id']: q for q in pool}
        assert len(final) == len(pool)
        turns = {t['turn_ref']: t for t in read(directory / 'qa_source.json')['turns']}
        expected = set()
        for record in read(directory / 'candidate_generation.json'):
            rid = record['request_id']
            raw = read(RUN / 'execution/calls' / root / rid / 'response.txt')
            candidates = raw.get('candidates', [])
            matching = {r['raw_row_index']: r for r in rows if r['request_id'] == rid}
            assert len(matching) == len(candidates), ('raw coverage', rid)
            for i, candidate in enumerate(candidates, 1):
                assert matching[i]['raw_candidate'] == candidate, (rid, i, 'raw modified')
                expected.add((rid, i))
            if '-generator-' in rid:
                type_count = counts(candidates)
                all_primary_quota.append({'request_id': rid, 'root_id': root, 'raw_counts': type_count,
                                          'two_single_choice_emitted': type_count.get('single_choice', 0) == 2})
            if raw.get('abstentions'):
                all_abstentions.append({'request_id': rid, 'root_id': root, 'raw_abstentions': raw['abstentions']})
        assert expected == {(r['request_id'], r['raw_row_index']) for r in rows}
        assert len(rows) == len(expected)
        for row in rows:
            assert row['verdict'] in ('PASS', 'HOLD', 'FAIL')
            assert row['historical_novelty']['verdict'] in ('NEW', 'EXACT_REPEAT', 'SEMANTIC_REPEAT', 'UNDETERMINED')
            for quote in row.get('diagnostic_evidence', []):
                original = turns[quote['turn_ref']]
                assert quote['text'] == original['text'], (row['candidate_id'], quote['turn_ref'], 'quote changed')
                assert quote['speaker'] == original['speaker']
                checked_quotes += 1
        retained = [r for r in rows if r['in_final_pool']]
        assert set(final) == {r['candidate_id'] for r in retained}
        assert all(r['final_candidate'] == final[r['candidate_id']] for r in retained)
        excluded = set(report.get('strict_pass_dedup_excluded_ids', []))
        assert excluded.issubset(final)
        strict = [r for r in retained if r['verdict'] == 'PASS' and r['candidate_id'] not in excluded]
        new = [r for r in strict if r['historical_novelty']['verdict'] == 'NEW']
        historical = read(BASE / 'feedback' / root / 'prompt_projection.json')['prompt_index']['entries']
        historical_keys = {exact_key(q): q['prior_occurrences'] for q in historical}
        root_exact = []
        for row in rows:
            key = exact_key(row['raw_candidate'])
            if key in historical_keys:
                # Exact matching ignores option labels, order, evidence and Gold.
                assert row['historical_novelty']['verdict'] == 'EXACT_REPEAT', row['candidate_id']
                root_exact.append({'candidate_id': row['candidate_id'], 'in_final_pool': row['in_final_pool'],
                                   'prior_occurrences': historical_keys[key]})
        source_need = lambda r: r.get('new_R02_information_necessary') in (True, 'YES')
        cost = usage(root_calls)
        stats = {'root_id': root, 'status': 'REVIEWED', 'raw_candidates': len(rows), 'raw_by_type': counts(rows),
                 'program_accepted': len(retained), 'program_accepted_by_type': counts(retained),
                 'content_verdicts': counts(retained, 'verdict'), 'raw_content_verdicts': counts(rows, 'verdict'),
                 'strict_pass_dedup': len(strict), 'strict_pass_dedup_by_type': counts(strict),
                 'strict_pass_dedup_excluded_ids': sorted(excluded),
                 'new_supported_dedup': len(new), 'new_supported_dedup_by_type': counts(new),
                 'history_novelty_final': dict(Counter(r['historical_novelty']['verdict'] for r in retained)),
                 'history_novelty_strict': dict(Counter(r['historical_novelty']['verdict'] for r in strict)),
                 'ordering_content_verdicts': counts([r for r in retained if r['question_type'] == 'ordering'], 'verdict'),
                 'ordering_whole_source_support': counts([r for r in retained if r['question_type'] == 'ordering'], 'whole_source_support'),
                 'new_supported_needing_R02': sum(source_need(r) for r in new),
                 'errors_final': error_counts(retained), 'errors_all_raw': error_counts(rows),
                 'calls': counts(root_calls, 'stage'), 'usage': cost,
                 'tokens_per_supported': per_item(cost['total_tokens'], len(strict)),
                 'tokens_per_new_supported': per_item(cost['total_tokens'], len(new)),
                 'tokens_per_valid_ordering': per_item(cost['total_tokens'], sum(r['question_type'] == 'ordering' for r in strict))}
        roots.append(stats)
        target = BASE / 'filtered' / root
        save(target / 'strict_pass_dedup.json', [r['final_candidate'] for r in strict])
        save(target / 'new_supported_dedup.json', [r['final_candidate'] for r in new])
        for verdict in ('PASS', 'HOLD', 'FAIL'):
            save(target / ('ordering_' + verdict + '.json'), [r['final_candidate'] for r in retained
                 if r['question_type'] == 'ordering' and r['verdict'] == verdict])
        save(target / 'FILTER_MANIFEST.json', {'reviewer': 'Codex', 'independent_human_certification': False,
             'original_questions_gold_and_evidence_modified': False, 'new_production_semantic_gate': False,
             'new_supported_policy': 'Content PASS, current-batch deduplicated, and historical_novelty NEW relative to supplied index only',
             'stats': stats})
        all_rows.extend(rows)
        final_rows_all.extend(retained)
        strict_all.extend(strict)
        new_all.extend(new)
        exact_repeats.extend(root_exact)
    cost = usage(ledger['records'])
    total = cost['total_tokens']
    cost.update(tokens_per_supported=per_item(total, len(strict_all)),
                tokens_per_new_supported=per_item(total, len(new_all)),
                tokens_per_valid_ordering=per_item(total, sum(r['question_type'] == 'ordering' for r in strict_all)))
    result = {'status': 'LIVE_COMPLETE_REVIEWED_STOPPED' if execution['status'] == 'FINISHED' else 'TECHNICAL_STOP_REVIEWED',
        'reviewer': 'Codex', 'independent_human_certification': False, 'roots': roots,
        'raw_candidates': len(all_rows), 'raw_by_type': counts(all_rows),
        'program_accepted': len(final_rows_all), 'program_accepted_by_type': counts(final_rows_all),
        'content_verdicts': counts(final_rows_all, 'verdict'), 'raw_content_verdicts': counts(all_rows, 'verdict'),
        'strict_pass_dedup': len(strict_all), 'strict_pass_dedup_by_type': counts(strict_all),
        'new_supported_dedup': len(new_all), 'new_supported_dedup_by_type': counts(new_all),
        'newness_scope': 'Supplied historical question index only; not a full historical-library claim',
        'historical_novelty_final': dict(Counter(r['historical_novelty']['verdict'] for r in final_rows_all)),
        'historical_novelty_strict': dict(Counter(r['historical_novelty']['verdict'] for r in strict_all)),
        'exact_historical_repeats': exact_repeats,
        'ordering_content_verdicts': counts([r for r in final_rows_all if r['question_type'] == 'ordering'], 'verdict'),
        'ordering_whole_source_support': counts([r for r in final_rows_all if r['question_type'] == 'ordering'], 'whole_source_support'),
        'errors_final': error_counts(final_rows_all), 'errors_all_raw': error_counts(all_rows),
        'primary_quota': all_primary_quota, 'raw_abstentions': all_abstentions,
        'calls': {'logical': ledger['logical_call_count'], 'http': ledger['http_attempt_count'],
                  'by_stage': counts(ledger['records'], 'stage'), 'automatic_retries': ledger['automatic_retry_count'],
                  'technical_failures': sum(r['status'] != 'COMPLETED' for r in ledger['records']),
                  'solver': 0, 'upstream_expansion': 0, 'memory_api': 0, 'training': 0, 'extra_audit_api': 0},
        'cost': cost, 'new_difficulty_measurement': 'NOT_RUN',
        'previous_run': {k: previous[k] for k in ('program_accepted', 'program_accepted_by_type',
                         'strict_pass_dedup', 'strict_pass_dedup_by_type', 'ordering_content_verdicts', 'calls', 'cost')},
        'comparison_limit': 'Descriptive known-case comparison of feedback rules and presentation together, not isolated causal attribution or generalization',
        'frozen_records_verified': len(manifest['files']), 'original_candidates_modified': False,
        'automatic_next_experiment': False, 'automatic_feedback_activation': False}
    save(BASE / 'reports/CONTENT_ALL_ROWS.json', all_rows)
    save(BASE / 'reports/FINAL_REPORT.json', result)
    save(BASE / 'reports/FINAL_VERIFICATION.json', {'status': 'PASS', 'raw_reviews': len(all_rows),
         'final_candidates_identical': len(final_rows_all), 'unmodified_supported_pool': len(strict_all),
         'unmodified_new_supported_pool': len(new_all), 'source_quotes_checked': checked_quotes,
         'frozen_files_unchanged': len(manifest['files']), 'previous_frozen_files_unchanged': 243,
         'budget_21_respected': True, 'new_audit_api_calls': 0, 'new_solver_calls': 0})
    print(json.dumps({k: result[k] for k in ('status', 'raw_candidates', 'program_accepted',
          'strict_pass_dedup', 'strict_pass_dedup_by_type', 'new_supported_dedup',
          'new_supported_dedup_by_type', 'historical_novelty_final', 'calls', 'cost')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
