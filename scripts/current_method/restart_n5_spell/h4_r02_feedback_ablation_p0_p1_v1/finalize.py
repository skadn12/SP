"""Additive reporting revision: join fixed metrics, blinded reviews and costs.

This module never changes generation, scoring, Gold, selection or admission.
Missing reviews prevent a COMPLETE final report; uncertain findings remain data.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import itertools
from pathlib import Path

from ..core import canonical_json, sha256_file
from .freeze import ARMS, ROOTS, DEFAULT, load, rows, source, write, write_rows, frozen_bytes, sha_obj

VERSION = 'p1_additive_reporting_audit_revision_v2'
TYPES = ('single_choice', 'multi_select', 'ordering')
GOLD_STATES = ('supported', 'contradicted', 'underdetermined', 'unsupported')
RELATION_STATES = ('supported', 'partial', 'absent', 'uncertain')
REVIEW_FIELDS = ('blind_id', 'root_id', 'question_type', 'full_transcript_read',
                 'gold_assessment', 'item_checks', 'ambiguity', 'duplicate_notes',
                 'feedback_relation_present', 'feedback_relation_evidence_refs', 'recommendation')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def mean(values):
    return sum(values) / len(values) if values and all(v is not None for v in values) else None


def record_revision(run: Path):
    """Freeze this reporting-only amendment after P0 and before Solver results."""
    path = run / 'audit/reporting_revision_v2.json'
    files = [Path(__file__), Path(__file__).with_name('analysis.py'), Path(__file__).with_name('test_finalize.py')]
    current = [source(p) for p in files]
    if path.is_file():
        existing = load(path)
        require(existing['current_code'] == current, 'reporting_revision_code_changed_after_recording')
        return existing
    p0 = load(run / 'p0_preflight_report.json')
    require(p0.get('pass') is True or p0.get('status') in ('PASS', 'PASS_COMPLETE', 'P0_PASS'), 'reporting_revision_requires_passed_p0')
    outcomes = [p for arm in ARMS for p in (run / arm / 'solver').glob('*/candidate_results.jsonl') if p.stat().st_size]
    require(not outcomes, 'reporting_revision_must_precede_solver_outcomes')
    old = load(run / 'execution_contract.json')
    previous = [x for x in old.get('runtime_code', []) if Path(x['path']).name == 'analysis.py']
    for p in files:
        frozen_bytes(run / 'audit/reporting_revision_v2_snapshots' / p.name, p.read_bytes())
    value = {'version': VERSION, 'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
             'timing': 'after P0; before any Solver candidate_results.jsonl outcomes',
             'p0_source': source(run / 'p0_preflight_report.json'), 'previous_analysis_identity': previous,
             'current_code': current, 'metrics_changed': False, 'review_selection_changed': False,
             'generation_or_scoring_changed': False, 'new_admission_gate': False,
             'changes': ['verify per-result dialogue and model solver-contract identities',
                         'require completed fixed blind-review and historical ordering review coverage for final report',
                         'join quality, relation observations, generation costs and existing diagnostics',
                         'render exploratory interpretation without changing frozen metric definitions'],
             'api_calls': 0, 'solver_outcome_files_present_at_revision': 0}
    write(path, value)
    return value


def expected_review_mapping(run: Path):
    # Import the frozen selector and payload builder, not a copied selection rule.
    from .analysis import qa_roots, selected_payload, blinded_case
    policy = load(run / 'audit/statistics_and_review_policy.json')['review']
    expected = []
    expected_cases = {}
    for arm in ARMS:
        inventory = list(qa_roots(run, arm))
        require(len(inventory) == 6 and {r for r, _ in inventory} == set(ROOTS), 'review_requires_six_roots_per_arm')
        for root, root_dir in inventory:
            require((root_dir / 'qa_completion_checkpoint.json').is_file(), f'review_root_generation_not_complete:{arm}:{root}')
            candidates = load(root_dir / 'candidates_valid.json')
            transcript = load(root_dir / 'expanded_dialogue_clean.json')
            require(transcript == load(run / 'inputs/shared' / root / 'expanded_dialogue_clean.json'), f'review_transcript_drift:{root}')
            history = load(run / 'inputs/feedback_on' / root / 'feedback_packet.json')['historical_qa']
            for qtype in TYPES:
                group = [c for c in candidates if c['question_type'] == qtype]
                chosen = sorted(group, key=lambda c: (sha_obj(selected_payload(c)), canonical_json(selected_payload(c))))[:2]
                for candidate in chosen:
                    digest = sha_obj(selected_payload(candidate))
                    blind_id = 'B' + digest
                    expected.append({'blind_id': blind_id, 'arm': arm, 'root_id': root,
                                     'candidate_id': candidate['candidate_id'], 'question_type': qtype,
                                     'selection_hash': digest, 'group_candidate_count': len(group)})
                    case = blinded_case(candidate, history, transcript, policy['checks'])
                    require(blind_id not in expected_cases or expected_cases[blind_id] == case, 'blind_id_collision')
                    expected_cases[blind_id] = case
    require(len(expected) <= 72, 'review_selection_exceeds_fixed_budget')
    return expected, expected_cases


def validate_review(review: dict, case: dict):
    blind_id = case['blind_id']
    require(all(k in review for k in REVIEW_FIELDS), f'blind_review_required_fields_missing:{blind_id}')
    require(review['blind_id'] == blind_id and review['root_id'] == case['root_id'] and
            review['question_type'] == case['candidate']['question_type'], f'blind_review_identity_drift:{blind_id}')
    require(review['full_transcript_read'] is True, f'blind_review_full_transcript_not_read:{blind_id}')
    require(review['gold_assessment'] in GOLD_STATES, f'blind_review_gold_assessment_invalid:{blind_id}')
    require(review['feedback_relation_present'] in RELATION_STATES, f'blind_review_relation_assessment_invalid:{blind_id}')
    require(isinstance(review['item_checks'], (list, dict)) and bool(review['item_checks']), f'blind_review_item_checks_missing:{blind_id}')
    require(isinstance(review['recommendation'], str) and bool(review['recommendation'].strip()), f'blind_review_recommendation_missing:{blind_id}')
    refs = review['feedback_relation_evidence_refs']
    valid_refs = {turn['turn_ref'] for turn in case['transcript']['turns']}
    require(isinstance(refs, list) and all(isinstance(ref, str) and ref in valid_refs for ref in refs), f'blind_review_relation_evidence_ref_invalid:{blind_id}')
    require(review['feedback_relation_present'] not in ('supported', 'partial') or bool(refs), f'blind_review_relation_support_requires_evidence:{blind_id}')
    # This is a review completeness check, not a programmatic semantic verdict.
    require(not {'arm', 'solver_scores', 'solver_observations', 'accuracy', 'candidate_id', 'generation_source', 'source_paths', 'feedback_status'} & set(review), f'blind_review_contains_unblinding_fields:{blind_id}')


def quality_groups(joined):
    groups = []
    scopes = {'main': ('single_choice', 'multi_select'), 'all_diagnostic': TYPES, **{t: (t,) for t in TYPES}}
    for arm, root, scope in itertools.product(ARMS, ('all',) + ROOTS, scopes):
        values = [r for r in joined if r['arm'] == arm and (root == 'all' or r['root_id'] == root) and r['question_type'] in scopes[scope]]
        count = len(values)
        gold = Counter(r['gold_assessment'] for r in values)
        relation = Counter(r['feedback_relation_present'] for r in values)
        expected_roots = ROOTS if root == 'all' else (root,)
        root_rates = {}
        for root_id in expected_roots:
            selected = [r for r in values if r['root_id'] == root_id]
            root_rates[root_id] = sum(r['gold_assessment'] == 'supported' for r in selected) / len(selected) if selected else None
        groups.append({'arm': arm, 'root_id': root, 'scope': scope, 'reviewed_candidate_instances': count,
                       'gold_assessment_counts': {s: gold[s] for s in GOLD_STATES},
                       'feedback_relation_counts': {s: relation[s] for s in RELATION_STATES},
                       'gold_supported_rate_of_reviewed': gold['supported'] / count if count else None,
                       'root_macro_gold_supported_rate': mean(list(root_rates.values())),
                       'root_supported_rates': root_rates,
                       'missing_review_sample_roots': [r for r, value in root_rates.items() if value is None]})
    return groups


def review_summary(run: Path):
    expected, expected_cases = expected_review_mapping(run)
    mapping_path = run / 'diagnostics/blind_review/arm_mapping_do_not_show_reviewers.jsonl'
    mapping = rows(mapping_path)
    key = lambda row: (row['arm'], row['root_id'], row['question_type'], row['candidate_id'])
    require(sorted(mapping, key=key) == sorted(expected, key=key), 'blind_review_mapping_differs_from_frozen_selection')
    review_root = run / 'diagnostics/blind_review/reviews'
    missing = sorted(bid for bid in expected_cases if not (review_root / (bid + '.json')).is_file())
    require(not missing, f'blind_review_coverage_incomplete:{len(missing)}:{missing[:3]}')
    reviews, files = {}, []
    for blind_id, expected_case in expected_cases.items():
        case_path = run / 'diagnostics/blind_review/cases' / (blind_id + '.json')
        require(load(case_path) == expected_case, f'blind_case_content_drift:{blind_id}')
        review_path = review_root / (blind_id + '.json')
        review = load(review_path)
        validate_review(review, expected_case)
        reviews[blind_id] = review
        files.append({'blind_id': blind_id, 'case_source': source(case_path), 'review_source': source(review_path)})
    joined = [{**m, **{k: reviews[m['blind_id']][k] for k in REVIEW_FIELDS}} for m in mapping]
    groups = quality_groups(joined)
    return {'coverage_complete': True, 'selected_candidate_instances': len(mapping),
            'unique_blind_cases': len(expected_cases), 'unique_reviews_completed': len(reviews),
            'all_full_transcripts_read': True, 'mapping_source': source(mapping_path), 'review_sources': files,
            'groups': groups, 'reviewed_candidate_rows': joined,
            'scope': 'fixed symmetric hash sample; at most 2 per root/arm/type; all available if fewer',
            'identical_payloads_share_review': True, 'blind_review_assertion': 'case files omit treatment and Solver scores; reviewer attestations are not independent human adjudication',
            'quality_is_a_diagnostic_only': True, 'gold_or_admission_changed': False,
            'relation_interpretation': 'Matching the historical relation is observational; it does not prove the generator causally used feedback.',
            'sampling_limit': 'Counts describe the fixed sample, not all candidates. Unknown/underdetermined judgments remain explicit.'}


def old_ordering_summary(run: Path):
    path = run / 'diagnostics/old_r02_ordering_review_summary.json'
    review_path = run / 'diagnostics/old_r02_ordering_review.jsonl'
    markdown_path = run / 'diagnostics/old_r02_ordering_review.md'
    require(path.is_file() and review_path.is_file() and markdown_path.is_file(), 'historical_52_ordering_diagnostics_missing')
    summary, reviewed = load(path), rows(review_path)
    coverage = summary['coverage']
    require(coverage['reviewed'] == coverage['expected'] == len(reviewed) == 52 and coverage['all_full_transcripts_read'] is True,
            'historical_ordering_review_coverage_incomplete')
    require(sha256_file(review_path) == summary['jsonl_sha256'], 'historical_ordering_review_hash_drift')
    inventory = summary['source_inventory']
    require(len(inventory) == 6 and {r['root_id'] for r in inventory} == set(ROOTS), 'historical_ordering_root_scope_mismatch')
    expected = set()
    for entry in inventory:
        cp, tp = Path(entry['candidate_source_path']), Path(entry['full_transcript_path'])
        require(sha256_file(cp) == entry['candidate_source_sha256'] and sha256_file(tp) == entry['full_transcript_sha256'], 'historical_ordering_source_hash_drift')
        candidates = [c for c in load(cp) if c['question_type'] == 'ordering']
        require(len(candidates) == entry['ordering_count'], 'historical_ordering_inventory_count_drift')
        for candidate in candidates:
            expected.add((entry['root_id'], candidate['candidate_id']))
            matches = [r for r in reviewed if r['root_id'] == entry['root_id'] and r['candidate_id'] == candidate['candidate_id']]
            require(len(matches) == 1, 'historical_ordering_duplicate_or_missing_review')
            review = matches[0]
            require(review['original_gold'] == candidate['gold'] and review['original_items'] == candidate['items'], 'historical_ordering_original_label_drift')
            require(review['applied_to_original_or_p1'] is False and bool(review['item_reviews']) and bool(review['ordering_basis']) and bool(review['order_uniqueness']), 'historical_ordering_review_incomplete_or_applied')
    require({(r['root_id'], r['candidate_id']) for r in reviewed} == expected, 'historical_ordering_candidate_scope_drift')
    require(dict(Counter(r['gold_assessment'] for r in reviewed)) == summary['gold_assessment_counts'], 'historical_ordering_summary_count_drift')
    return {'summary': summary, 'summary_source': source(path), 'review_source': source(review_path),
            'markdown_source': source(markdown_path), 'p1_inputs_or_labels_modified': False,
            'interpretation': 'Historical ordering findings explain diagnostic limitations; they do not re-enter active feedback or change P1 labels.'}


def generation_accounting(run: Path):
    summary_path = run / 'audit/qa_execution_summary.json'
    summary = load(summary_path)
    require(summary['status'] == 'completed' and summary['completed_root_arms'] == 12, 'qa_generation_not_complete_for_final_report')
    transport_path = run / 'audit/qa_transport_events.jsonl'
    transport = rows(transport_path)
    arms = {}
    for arm in ARMS:
        manifest = load(run / arm / 'qa/manifest.json')
        require(len(manifest['roots']) == 6 and {r['root_id'] for r in manifest['roots']} == set(ROOTS), 'qa_accounting_root_scope_mismatch')
        root_rows = []
        for entry in manifest['roots']:
            root = entry['root_id']
            rd = run / arm / 'qa/shards' / entry['shard'] / 'roots' / root
            checkpoint = load(rd / 'qa_completion_checkpoint.json')
            require(all(sha256_file(rd / name) == digest for name, digest in checkpoint['files'].items()), 'qa_completed_artifact_hash_drift')
            states = [load(p) for p in sorted((rd / 'ledger/requests').glob('*.json'))]
            require(len(states) <= 13, 'qa_root_arm_logical_budget_exceeded')
            require(all(s['request_metadata']['arm'] == arm and s['request_metadata']['root_id'] == root for s in states), 'qa_ledger_arm_root_identity_drift')
            counts = Counter()
            usage = Counter()
            for state in states:
                meta = state['request_metadata']
                counts['logical_calls'] += 1
                counts['miner_calls' if meta['mode'] == 'intent_miner' else 'repair_calls' if meta['attempt'] == 'repair' else 'writer_primary_calls'] += 1
                counts['completed_calls' if state['status'] == 'completed' else 'unknown_billing_status_calls'] += 1
                if state['status'] == 'completed':
                    require(sha256_file(Path(state['raw_response_path'])) == state['raw_response_sha256'], 'qa_raw_response_hash_drift')
                    usage.update({k: v for k, v in (state.get('response_metadata', {}).get('usage') or {}).items() if isinstance(v, (int, float))})
            records = load(rd / 'candidate_generation.json')
            pool = load(rd / 'candidate_pool.json')
            valid = load(rd / 'candidates_valid.json')
            report = load(rd / 'qa_generation_report.json')
            require(len(pool) == report['candidate_count'] and len(valid) == report['valid_candidate_count'], 'qa_candidate_report_count_drift')
            require(counts['miner_calls'] == 1 and counts['writer_primary_calls'] <= 6 and counts['repair_calls'] <= 6,
                    'qa_original_call_scope_drift')
            writer_attempts = [{'request_id': r['request_id'], 'mode': r['generation_source'],
                                'attempt': 'repair' if r.get('repair_of') else 'primary',
                                'returned_candidate_rows': len(r['validation'].get('candidates') or []),
                                'structurally_valid_counts': r['validation']['minimum_acceptance']['valid_counts'],
                                'minimum_batch_accepted': r['validation']['minimum_acceptance']['accepted']}
                               for r in records]
            root_rows.append({'arm': arm, 'root_id': root, 'status': report['status'],
                              **{k: counts[k] for k in ('logical_calls', 'miner_calls', 'writer_primary_calls', 'repair_calls', 'completed_calls', 'unknown_billing_status_calls')},
                              'completion_usage': dict(usage), 'valid_contract_count': report['valid_contract_count'],
                              'returned_candidate_rows_across_all_writer_attempts': sum(r['returned_candidate_rows'] for r in writer_attempts),
                              'candidate_pool_count_after_original_repair_replacement': len(pool),
                              'program_valid_candidate_count': len(valid), 'program_rejected_candidate_count': len(pool) - len(valid),
                              'valid_counts_by_type': dict(Counter(c['question_type'] for c in valid)),
                              'valid_counts_by_generation_source': dict(Counter(c['generation_source'] for c in valid)),
                              'valid_counts_by_type_and_source': [
                                  {'question_type': qtype, 'generation_source': mode,
                                   'count': sum(c['question_type'] == qtype and c['generation_source'] == mode for c in valid)}
                                  for qtype, mode in itertools.product(TYPES, ('contract_guided', 'free_exploration'))],
                              'writer_attempts': writer_attempts})
        outcomes = [r for r in transport if r['arm'] == arm and r['event'] == 'logical_call_transport_outcome']
        usage = Counter()
        for root in root_rows:
            usage.update(root['completion_usage'])
        arms[arm] = {k: sum(r[k] for r in root_rows) for k in (
            'logical_calls', 'miner_calls', 'writer_primary_calls', 'repair_calls', 'completed_calls', 'unknown_billing_status_calls',
            'returned_candidate_rows_across_all_writer_attempts', 'candidate_pool_count_after_original_repair_replacement',
            'program_valid_candidate_count', 'program_rejected_candidate_count')}
        arms[arm].update({'roots': root_rows, 'completion_usage': dict(usage),
                          'http_attempts': sum(r['arm'] == arm and r['event'] == 'http_attempt_started' for r in transport),
                          'http_retries': sum(max(0, r['transport_attempt_count'] - 1) for r in outcomes),
                          'transport_failures': sum(r['status'] == 'failed' for r in outcomes),
                          'timeout_calls': sum(bool(r.get('timeout')) for r in outcomes),
                          'monetary_cost': None, 'monetary_cost_known': False})
        require(arms[arm]['program_valid_candidate_count'] == summary['arms'][arm]['valid_candidate_count'], 'qa_arm_count_summary_drift')
    for metric in ('logical_calls', 'miner_calls', 'writer_primary_calls', 'repair_calls', 'completed_calls',
                   'unknown_billing_status_calls', 'http_attempts', 'http_retries', 'transport_failures', 'timeout_calls'):
        require(sum(a[metric] for a in arms.values()) == summary[metric], f'qa_global_accounting_drift:{metric}')
    total_usage = Counter()
    for arm in arms.values():
        total_usage.update(arm['completion_usage'])
    require(dict(total_usage) == summary['completion_usage'], 'qa_token_accounting_drift')
    require(summary['logical_calls'] <= 156, 'qa_experiment_budget_exceeded')
    return {'summary_source': source(summary_path), 'transport_source': source(transport_path),
            'original_scheduler_summary': summary, 'per_arm': arms,
            'budget': {'planned': 84, 'maximum_logical': 156, 'root_arm_planned': 7, 'root_arm_maximum': 13},
            'actual_counts_policy': 'No trimming of structurally valid over-quota returns; repaired primary rows are replaced according to the original generator.',
            'cost_limit': 'Actual completion tokens are recorded. Provider price and monetary cost are unknown; failed/unknown billing is reported separately.'}


def enrich_report(run: Path, summary: dict):
    require(summary.get('status') == 'COMPLETE', 'metric_report_not_complete')
    revision_path = run / 'audit/reporting_revision_v2.json'
    require(revision_path.is_file(), 'reporting_revision_not_frozen')
    revision = load(revision_path)
    require(all(sha256_file(Path(row['path'])) == row['sha256'] for row in revision['current_code']), 'reporting_revision_code_hash_drift')
    quality = review_summary(run)
    ordering = old_ordering_summary(run)
    accounting = generation_accounting(run)
    prompt_path = run / 'audit/prompt_parity_complete.json'
    prompt = load(prompt_path)
    require(prompt['status'] == 'PASS_COMPLETE' and prompt['completed_root_arms'] == 12 and not prompt['errors'], 'complete_live_prompt_parity_required')
    content_path = run / 'diagnostics/generation_content_diagnostics.json'
    duplicate_path = run / 'diagnostics/duplicate_diagnostics.json'
    content, duplicates = load(content_path), load(duplicate_path)
    require(len(content['rows']) == 12 and {(r['arm'], r['root_id']) for r in content['rows']} == set(itertools.product(ARMS, ROOTS)), 'generation_content_diagnostic_scope_incomplete')
    require(content['selection_count'] == quality['selected_candidate_instances'], 'content_review_selection_count_drift')
    for row in content['rows']:
        actual = next(r for r in accounting['per_arm'][row['arm']]['roots'] if r['root_id'] == row['root_id'])
        require(row['candidate_count'] == actual['program_valid_candidate_count'] and
                row['type_counts'] == actual['valid_counts_by_type'] and
                row['mode_counts'] == actual['valid_counts_by_generation_source'], 'generation_content_diagnostic_count_drift')
    require(content.get('admission_changed') is False and duplicates.get('admission_changed') is False, 'diagnostic_admission_changed')
    require(all(r['arm'] in ARMS and r['root_id'] in ROOTS for r in duplicates['groups']), 'duplicate_diagnostic_scope_drift')
    write(run / 'reports/blind_review_quality_summary.json', quality)
    write_rows(run / 'reports/blind_review_group_metrics.jsonl', quality['groups'])
    write(run / 'reports/qa_generation_accounting.json', accounting)
    return {**summary, 'reporting_revision': source(revision_path),
            'completion_checks': {'solver_denominators_complete': True, 'blind_review_coverage_complete': True,
                                  'historical_ordering_52_coverage_complete': True, 'qa_accounting_verified': True,
                                  'live_original_builder_prompt_parity_complete': True},
            'prompt_parity_source': source(prompt_path),
            'quality_review': quality, 'historical_ordering_diagnosis': ordering, 'generation_accounting': accounting,
            'generation_content_diagnostics': {'source': source(content_path), **content},
            'duplicate_diagnostics': {'source': source(duplicate_path), **duplicates,
                                      'comparison_scope': 'same arm and same root only',
                                      'cross_arm_or_cross_root_duplicates_computed': False},
            'solver_token_costs': {'source': source(run / 'reports/solver_token_costs.json'),
                                  'rows': load(run / 'reports/solver_token_costs.json')},
            'interpretation': {
                'finding_level': 'exploratory; six paired roots and one generation per arm',
                'difficulty': 'Negative on-minus-off means lower accuracy against unchanged Gold; it is not by itself a successful feedback treatment.',
                'quality': 'Interpret difficulty together with symmetric sampled Gold/ambiguity/item checks, R02 evidence, duplicate diagnostics and full-minus-no results.',
                'feedback_use': 'Historical-relation presence is an observed correspondence, not proof of causal use of feedback or improved reasoning.',
                'scope': 'This tests the existing appended QA-feedback mechanism on fixed transcripts that previously received old feedback.',
                'unknown': ['independent human adjudication', 'quality of every unreviewed candidate', 'causal mechanism of a relation match', 'training benefit', 'provider monetary cost'],
                'ordering': 'Ordering remains a separately scored diagnostic and never enters the primary difficulty conclusion.',
                'stop': 'P1 complete; no P2 interface, third arm, Memory write, expansion rerun or training.'}}


def render_report(summary: dict):
    quality, accounting = summary['quality_review'], summary['generation_accounting']
    lines = ['', f"内容抽查已完成 {quality['unique_reviews_completed']}/{quality['unique_blind_cases']} 个唯一盲化案例，覆盖 {quality['selected_candidate_instances']} 个按冻结哈希选中的 root/arm/type 候选实例；所有记录声明读取完整 transcript。",
             '同一内容跨臂出现时可共用盲审记录；下表分母仍按每臂实际被抽中实例计算。抽查不改 Gold，也不改变准入。', '',
             '| 臂 | 主范围抽查数 | Gold 有支持 | 矛盾 | 未定 | 无支持 | 历史关系 supported / partial / absent / uncertain |',
             '|---|---:|---:|---:|---:|---:|---|']
    for arm in ARMS:
        group = next(g for g in quality['groups'] if g['arm'] == arm and g['root_id'] == 'all' and g['scope'] == 'main')
        gold, relation = group['gold_assessment_counts'], group['feedback_relation_counts']
        lines.append(f"| {arm} | {group['reviewed_candidate_instances']} | {gold['supported']} | {gold['contradicted']} | {gold['underdetermined']} | {gold['unsupported']} | {' / '.join(str(relation[s]) for s in RELATION_STATES)} |")
    lines += ['', '完整对称抽查分组保存在 `blind_review_group_metrics.jsonl`，逐题判断、歧义和建议保存在 `blind_review_quality_summary.json`。历史关系相似只证明内容对应，不能独立证明生成器实际利用反馈。', '',
              '| 臂 | 程序有效题 | Miner | Writer primary | repair | 逻辑调用 | HTTP 尝试 | timeout | 付费状态未知 | prompt / completion tokens |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for arm in ARMS:
        a = accounting['per_arm'][arm]
        usage = a['completion_usage']
        lines.append(f"| {arm} | {a['program_valid_candidate_count']} | {a['miner_calls']} | {a['writer_primary_calls']} | {a['repair_calls']} | {a['logical_calls']} | {a['http_attempts']} | {a['timeout_calls']} | {a['unknown_billing_status_calls']} | {usage.get('prompt_tokens', 'unknown')} / {usage.get('completion_tokens', 'unknown')} |")
    old = summary['historical_ordering_diagnosis']['summary']
    lines += ['', '实际逐 root、题型、Guided/Free 返回量及 repair 批次见 `qa_generation_accounting.json`；实际超过请求配额而程序有效的题没有静默裁剪。API 未提供价格，金额成本未知；Solver 为本地推理，实际 token 保存在 `solver_token_costs.json`。', '',
              f"旧对齐 R02 排序题已核查 {old['coverage']['reviewed']}/{old['coverage']['expected']}，Gold 判断分布：{old['gold_assessment_counts']}。诊断未修改当前实验输入、排序生成方法、标签或评分。",
              'R02 引用与证据分布见 `generation_content_diagnostics`；重复检查范围仅为同臂、同 root，未计算跨臂或跨 root 重复。', '',
              '效果判断仍需同时考虑主指标、抽查中的 Gold 支持和歧义、full−no 的上下文依赖、关系对应及重复情况。准确率下降本身不能证明反馈成功；存在质量问题时，也可能只是题目或标签更难可靠解释。',
              '六个配对 root、每臂一次生成只能给探索性证据。未核查候选的完整质量、独立人工裁决、反馈使用的因果机制及新训练收益仍未知。本次交付在 P1 停止。']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT)
    parser.add_argument('--phase', choices=('record-revision',), required=True)
    args = parser.parse_args()
    print(record_revision(args.run.resolve())['version'])
