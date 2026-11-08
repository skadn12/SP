"""Read-only execution audit; writes only this delivery's audit reports.

No transport or model client is constructed. All responses, including failed
and rejected requests, remain in the accounting. Missing usage stays UNKNOWN.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

WORKTREE = Path('/root/autodl-tmp/gxy-selfplay/worktrees/h4-gate-downstream-v1')
sys.path.insert(0, str(WORKTREE))
from scripts.current_method.restart_n5_spell.core import canonical_json
from scripts.current_method.restart_n5_spell.h4_qa_mining_hybrid_generation_v1.contract_guided_generator import build_prompt
from scripts.current_method.restart_n5_spell.h4_writer_partial_memory_v1.admission import repair_required
from scripts.current_method.restart_n5_spell.h4_writer_partial_memory_v1.sources import render_dialogue

REPORTS = Path(__file__).resolve().parent
BASE = REPORTS.parent
RUN = BASE / 'prepared_run'
LIVE = RUN / 'live'
EXECUTION = LIVE / 'execution'
OLD_RUN = BASE.parent / 'h4_qa_difficulty_feedback_v1/prepared_run'
ROOTS = ['cc-train-episode-259192', 'cc-train-episode-96462', 'cc-train-episode-112812']
TOKEN_FIELDS = ('input_tokens', 'output_tokens', 'reasoning_tokens', 'total_tokens')
INDEX_OPEN = '[HISTORICAL_QUESTION_AVOIDANCE_INDEX]'
INDEX_CLOSE = '[/HISTORICAL_QUESTION_AVOIDANCE_INDEX]'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def known(value):
    return value if value is not None else 'UNKNOWN'


def usage_totals(records):
    result = {}
    for field in TOKEN_FIELDS:
        values = [row.get(field, 'UNKNOWN') for row in records]
        missing = sum(type(value) is not int for value in values)
        subtotal = sum(value for value in values if type(value) is int)
        result[field] = subtotal if not missing else 'UNKNOWN'
        result[field + '_known_subtotal'] = subtotal
        result[field + '_unknown_call_count'] = missing
    return result


def forbidden_index_keys(value):
    banned = {'gold', 'historical_gold', 'correct_option', 'solver_observations',
              'accuracy', 'verdict', 'content_diagnosis', 'audit_only', 'score', 'scores'}
    if isinstance(value, dict):
        result = set(value) & banned
        for child in value.values():
            result |= forbidden_index_keys(child)
        return result
    if isinstance(value, list):
        result = set()
        for child in value:
            result |= forbidden_index_keys(child)
        return result
    return set()


def main():
    ledger_path = EXECUTION / 'qa_call_ledger.json'
    ledger = read(ledger_path)
    summary_path = LIVE / 'generation_summary.json'
    if ledger['status'] not in ('FINISHED', 'STOPPED') or not summary_path.exists():
        print(json.dumps({'status': 'WAITING_FOR_TERMINAL_LEDGER_AND_SUMMARY',
                          'ledger_status': ledger['status'], 'audit_reports_written': False}))
        return
    summary = read(summary_path)
    records = ledger['records']
    manifest = read(RUN / 'freeze_manifest.json')
    old_manifest = read(OLD_RUN / 'freeze_manifest.json')
    profile = read(RUN / 'frozen/provider_profile.json')
    checks, request_checks, trigger_checks, usage_records, root_results = [], [], [], [], {}

    def check(name, passed, **details):
        checks.append({'check': name, 'pass': bool(passed), **details})

    for label, frozen, count in [('new', manifest, 265), ('previous', old_manifest, 243)]:
        changed = [row['path'] for row in frozen['files']
                   if not Path(row['path']).is_file() or sha(row['path']) != row['sha256']]
        check(label + '_frozen_files_unchanged', not changed and len(frozen['files']) == count,
              expected_file_count=count, actual_file_count=len(frozen['files']), changed_paths=changed)
    check('ledger_equals_complete_call_records', records == read(LIVE / 'complete_call_records.json'))
    check('fixed_three_roots', ledger['root_order'] == ROOTS)
    check('budget_and_accounting',
          ledger['maximum_http_attempts'] == ledger['maximum_logical_calls'] == 21
          and ledger['maximum_calls_per_root'] == 7
          and len(records) == ledger['logical_call_count'] == summary['logical_call_count']
          and sum(row['http_attempt_count'] for row in records)
          == ledger['http_attempt_count'] == summary['external_http_attempt_count']
          and ledger['logical_call_count'] <= 21 and ledger['http_attempt_count'] <= 21)
    check('no_automatic_retries_or_resume', ledger['automatic_retry_count'] == 0
          and ledger['automatic_resume'] is False
          and all(row['retry_count'] == 0 and row['http_attempt_count'] in (0, 1) for row in records))
    check('no_redirection_ssl_or_parameter_fallback', profile.get('http_retry_limit') == 0
          and all(profile.get(key) is False for key in ('redirects', 'ssl_fallback', 'parameter_fallback')))
    check('unique_logical_requests', len({row['request_id'] for row in records}) == len(records))
    check('sequential_reservations', [row['logical_call_index'] for row in records] == list(range(1, len(records) + 1)))
    root_indices = [ROOTS.index(row['root_id']) for row in records]
    check('fixed_root_call_order', root_indices == sorted(root_indices))
    check('only_miner_primary_repair', set(row['stage'] for row in records)
          <= {'miner', 'writer_primary', 'writer_repair'})
    by_id = {row['request_id']: row for row in records}
    rules = (BASE / 'writer_constraints.txt').read_text()
    index_occurrences, index_entries = [], 0

    for root in ROOTS:
        root_records = [row for row in records if row['root_id'] == root]
        check('per_root_budget:' + root, len(root_records) <= 7)
        frozen = RUN / 'frozen' / root
        directory = LIVE / 'roots' / root
        if not (directory / 'qa_source.json').exists():
            root_results[root] = {'logical_calls': len(root_records), 'status': 'NOT_STARTED'}
            continue
        source = read(directory / 'qa_source.json')
        context = read(frozen / 'qa_feedback_context.json')
        provenance = read(directory / 'qa_source_provenance.json')
        feedback_provenance = read(directory / 'qa_feedback_provenance.json')
        check('source_and_dates_unchanged:' + root, sha(directory / 'qa_source.json') == sha(frozen / 'qa_source.json'))
        check('feedback_context_frozen:' + root,
              sha(directory / 'qa_feedback_context.json') == sha(frozen / 'qa_feedback_context.json')
              and digest(context) == feedback_provenance['context_object_sha256'])
        check('feedback_separate_from_facts:' + root,
              feedback_provenance['qa_prompt_contains_feedback'] is True
              and feedback_provenance['qa_source_contains_feedback'] is False
              and feedback_provenance['memory_in_prompt'] is False
              and provenance['memory_or_feedback_in_qa_prompt'] is True
              and provenance['memory_in_prompt'] is False)
        check('three_identical_writer_suffixes:' + root, len(context['slot_suffixes']) == 3
              and len(set(context['slot_suffixes'])) == 1)
        suffix = context['slot_suffixes'][0]
        check('verbatim_writer_constraints:' + root, suffix.startswith(rules + '\n\n'))
        index = json.loads(suffix.split(INDEX_OPEN + '\n', 1)[1].split('\n' + INDEX_CLOSE, 1)[0])
        check('historical_index_has_no_answer_diagnosis_or_score_fields:' + root,
              not forbidden_index_keys(index) and index['root_id'] == root)
        index_entries += len(index['entries'])
        index_occurrences.extend((o['generation_batch'], o['candidate_id'])
                                 for item in index['entries'] for o in item['prior_occurrences'])

        registered_path = EXECUTION / 'registered_contracts' / (root + '.json')
        registered = read(registered_path) if registered_path.exists() else []
        contract_map = {row['intent_id']: row for row in registered}
        if registered:
            check('new_contracts_match_registration:' + root,
                  digest(registered) == ledger['registered_contracts'][root]['contracts_sha256']
                  and registered == read(directory / 'reasoning_contracts.json'))
        generation_path = directory / 'candidate_generation.json'
        progress_path = directory / 'qa_progress.json'
        generation = (read(generation_path) if generation_path.exists() else
                      read(progress_path).get('generation_records', []) if progress_path.exists() else [])
        generation_by_id = {row['request_id']: row for row in generation}

        for record in root_records:
            call_dir = EXECUTION / record['artifact_directory']
            request = read(call_dir / 'request.json')
            payload = request['payload']
            prompt = (call_dir / 'prompt.txt').read_text()
            stage = record['stage']
            slot_suffix = (context['intent_suffix'] if stage == 'miner' else
                           context['slot_suffixes'][list(contract_map).index(record['intent_id'])])
            own = {'request_id': record['request_id'], 'root_id': root, 'stage': stage, 'checks': {}}
            own_checks = own['checks']
            own_checks['payload_and_messages_hash'] = (
                digest(payload) == record['payload_sha256']
                and digest(payload['messages']) == record['messages_sha256'])
            own_checks['one_exact_user_message'] = payload['messages'] == [{'role': 'user', 'content': prompt}]
            own_checks['complete_transcript_no_truncation'] = render_dialogue(source, numbered=stage == 'miner') in prompt
            own_checks['corresponding_feedback_appended_once'] = (
                prompt.endswith('\n' + slot_suffix) and prompt.count(INDEX_OPEN) == 1)
            expected = {'model': 'tju-llm', 'temperature': 0.1,
                        'max_tokens': 4096 if stage == 'miner' else 8192,
                        'response_format': {'type': 'json_object'}}
            own_checks['only_approved_parameters'] = (
                set(payload) == set(expected) | {'messages'}
                and all(payload[key] == value for key, value in expected.items())
                and request['endpoint'] == 'https://ai.tju.edu.cn/api/v3/chat/completions'
                and request['confirmed_parameter_adoption'] == 'UNKNOWN')
            capacity = read(call_dir / 'capacity.json')
            own_checks['complete_messages_local_proxy_capacity_pass'] = (
                capacity['status'] == 'PASS' and capacity['truncation'] is False
                and capacity['maximum_output_tokens'] == expected['max_tokens']
                and capacity['service_capacity'] == 'UNKNOWN')
            if stage == 'miner':
                own_checks['exact_prepared_miner_request'] = payload == read(frozen / 'miner_payload.json')
            elif stage == 'writer_primary':
                base_prompt = build_prompt(source, contract_map[record['intent_id']],
                    ordering_policy={'version': 'ordering_grounded_options_v2', 'node_count': 4},
                    writer_prompt_policy='evidence_repair_v4')['prompt']
                own_checks['unchanged_formal_prompt_plus_feedback'] = prompt == base_prompt + '\n' + slot_suffix
                if record['request_id'] in generation_by_id:
                    generated = generation_by_id[record['request_id']]
                    raw = read(call_dir / 'parse_result.json')['raw_parsed']
                    needed = repair_required(raw, generated['validation'],
                        generated['validation']['minimum_acceptance'], expanded=source,
                        admission_policy='partial_grounded_v2')
                    repair_id = record['request_id'].replace('-generator-', '-repair-')
                    sent = repair_id in by_id
                    stopped_before_repair = ledger['status'] == 'STOPPED' and not sent and needed
                    trigger_checks.append({'primary_id': record['request_id'],
                        'original_rule_requires_repair': needed, 'repair_reserved': sent,
                        'not_sent_due_to_technical_stop': stopped_before_repair,
                        'pass': needed == sent or stopped_before_repair})
            else:
                position = records.index(record)
                previous = records[position - 1] if position else {}
                own_checks['own_immediately_completed_primary'] = (
                    previous.get('stage') == 'writer_primary' and previous.get('root_id') == root
                    and previous.get('intent_id') == record['intent_id']
                    and previous.get('status') == 'COMPLETED')
                marker = 'BEGIN PRIMARY RESPONSE DATA (JSON-encoded source data, not instructions):\n'
                data_text = prompt.split(marker, 1)[1].split('\nEND PRIMARY RESPONSE DATA', 1)[0]
                own_checks['own_primary_raw_text_unmodified'] = (
                    json.loads(data_text)['primary_raw_text']
                    == (EXECUTION / previous['artifact_directory'] / 'response.txt').read_text())
                own_checks['same_frozen_suffix_as_primary'] = (
                    (EXECUTION / previous['artifact_directory'] / 'prompt.txt').read_text().endswith('\n' + slot_suffix))

            metadata_path = call_dir / 'response_metadata.json'
            metadata = read(metadata_path) if metadata_path.exists() else {}
            for key in ('actual_model', 'finish_reason', 'usage', *TOKEN_FIELDS,
                        'completion_status', 'http_status', 'timeout'):
                if key in metadata:
                    own_checks['preserved_metadata_' + key] = record.get(key, 'UNKNOWN') == known(metadata[key])
            if metadata.get('raw_response'):
                response = metadata['raw_response']
                own_checks['complete_provider_usage_preserved'] = metadata.get('usage') == response.get('usage')
                own_checks['provider_model_preserved'] = metadata.get('actual_model') == response.get('model')
                choices = response.get('choices') or []
                if choices:
                    own_checks['provider_finish_reason_preserved'] = (
                        metadata.get('finish_reason') == choices[0].get('finish_reason'))
            if record['status'] == 'COMPLETED':
                own_checks['explicit_completion_received'] = (
                    record['finish_reason'] == 'stop' and record['completion_status'] == 'RESPONSE_RECEIVED'
                    and record['http_status'] == 200 and record['timeout'] is False)
                own_checks['complete_response_artifacts'] = (
                    metadata_path.exists() and sha(call_dir / 'response.txt') == record['response_text_sha256'])
            if metadata:
                own_checks['one_transport_attempt_zero_retries'] = (
                    metadata.get('http_attempt_count') == 1 and metadata.get('transport_attempt_count') == 1
                    and metadata.get('transport_retry_count') == metadata.get('rate_limit_retry_count') == 0)
            own['pass'] = all(own_checks.values())
            request_checks.append(own)
            # Every reserved call remains in totals, including technical failures.
            usage_records.append({key: record.get(key, 'UNKNOWN') for key in (
                'request_id', 'root_id', 'stage', 'status', 'http_attempt_count', 'actual_model',
                'finish_reason', 'completion_status', 'http_status', 'timeout', 'usage', *TOKEN_FIELDS)}
                | {'verified_cost': 'UNKNOWN'})

        root_summary = read(directory / 'qa_summary.json')
        pool_path = directory / 'candidate_pool.json'
        pool = read(pool_path) if pool_path.exists() else []
        root_results[root] = {'status': root_summary['status'], 'logical_calls': len(root_records),
            'http_attempts': sum(row['http_attempt_count'] for row in root_records),
            'stages': dict(Counter(row['stage'] for row in root_records)),
            'program_accepted_by_type': dict(Counter(row['question_type'] for row in pool)),
            'usage': usage_totals([row for row in usage_records if row['root_id'] == root]),
            'verified_cost': 'UNKNOWN'}
        if root_summary['status'] == 'MINER_INVALID':
            check('invalid_miner_no_writer_or_old_intent_fallback:' + root,
                  not registered and all(row['stage'] == 'miner' for row in root_records))

    check('request_identity_and_usage_preservation', all(row['pass'] for row in request_checks), count=len(request_checks))
    check('repairs_follow_original_conditions', all(row['pass'] for row in trigger_checks), checked_primary_count=len(trigger_checks))
    check('no_extra_request_artifacts', len(list((EXECUTION / 'calls').glob('*/*/request.json'))) == len(records))
    check('no_other_model_stages', summary['upstream_free_memory_solver_training_review_api_calls'] == 0
          and summary['automatic_next_experiment'] is False)
    check('terminal_state_consistent', summary['status'] == ledger['status'])
    incomplete = [row for row in records if row['status'] != 'COMPLETED']
    if incomplete:
        check('technical_failure_stopped_following_calls', ledger['status'] == 'STOPPED'
              and records.index(incomplete[0]) == len(records) - 1)
    else:
        check('no_pending_or_unknown_completion', all(row['completion_status'] == 'RESPONSE_RECEIVED' for row in records))
    # Complete index coverage is required whenever all three roots were reached.
    if len([root for root in ROOTS if (LIVE / 'roots' / root / 'qa_source.json').exists()]) == 3:
        check('all_forty_two_occurrences_in_thirty_five_index_entries',
              index_entries == 35 and len(index_occurrences) == len(set(index_occurrences)) == 42
              and Counter(batch for batch, _ in index_occurrences)
              == {'h4_gate_downstream_v1': 13, 'h4_qa_difficulty_feedback_v1': 29})
    usage = usage_totals(usage_records)
    result = {'audit_version': 'qa_feedback_constraints_v2_execution_audit_v1',
        'auditor': 'Codex independent execution-trace review, not independent human certification',
        'audited_at': datetime.now(timezone.utc).isoformat(),
        'status': 'PASS' if all(row['pass'] for row in checks) else 'FAIL',
        'scope': 'EXECUTION_INTEGRITY_AND_ACCOUNTING_NOT_CONTENT_OR_DIFFICULTY',
        'run_status': ledger['status'], 'stop_reason': ledger.get('stop_reason'),
        'logical_calls': ledger['logical_call_count'], 'http_attempts': ledger['http_attempt_count'],
        'authorized_logical_and_http_cap': 21,
        'stage_counts': dict(Counter(row['stage'] for row in records)),
        'automatic_retries': ledger['automatic_retry_count'],
        'technical_failure_count': len(incomplete), 'timeouts': sum(row['timeout'] is True for row in records),
        'by_root': root_results, 'usage': usage,
        'actual_models': sorted({row['actual_model'] for row in records}),
        'verified_cost': 'UNKNOWN', 'service_parameter_adoption': 'UNKNOWN',
        'service_capacity': 'UNKNOWN', 'service_reasoning_default': 'UNKNOWN',
        'checks': checks, 'request_checks': request_checks, 'repair_trigger_checks': trigger_checks,
        'complete_usage_by_call': usage_records,
        'sources': {label: {'path': str(path), 'sha256': sha(path)} for label, path in {
            'new_freeze_manifest': RUN / 'freeze_manifest.json',
            'previous_freeze_manifest': OLD_RUN / 'freeze_manifest.json',
            'ledger': ledger_path, 'summary': summary_path, 'audit_script': Path(__file__)}.items()},
        'new_external_calls_by_audit': 0, 'production_or_frozen_files_modified_by_audit': 0}
    (REPORTS / 'EXECUTION_AUDIT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = [f"执行与成本审计：{result['status']}。运行状态：{ledger['status']}。", '',
        f"共 {len(records)} 次逻辑调用、{ledger['http_attempt_count']} 次 HTTP；授权上限各 21。"
        f"阶段：{json.dumps(result['stage_counts'], ensure_ascii=False)}。重试 {result['automatic_retries']}，"
        f"超时 {result['timeouts']}，技术失败 {len(incomplete)}。", '',
        '核对新版 265 个冻结文件和上一轮 243 个冻结文件；完整请求逐一核对原文、反馈、参数、'
        '模板、原合同 repair、模型标识、finish_reason 和完整 usage。详细布尔检查保存在 JSON。', '',
        '| root | Miner | primary | repair | 输入 tokens | 输出 tokens | 总 tokens |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for root, entry in root_results.items():
        stages, totals = entry.get('stages', {}), entry.get('usage', {})
        lines.append(f"| {root} | {stages.get('miner', 0)} | {stages.get('writer_primary', 0)} | "
                     f"{stages.get('writer_repair', 0)} | {totals.get('input_tokens', 'UNKNOWN')} | "
                     f"{totals.get('output_tokens', 'UNKNOWN')} | {totals.get('total_tokens', 'UNKNOWN')} |")
    lines += ['', f"全部调用输入 {usage['input_tokens']}、输出 {usage['output_tokens']}、总 tokens {usage['total_tokens']}。"
        '包括拒收候选、被修复 primary、全部 repair 与技术失败调用；有缺失 usage 时总量记 UNKNOWN，'
        '已知小计另留 JSON，绝不将缺失记为零。', '',
        f"推理 tokens：{usage['reasoning_tokens']}。实际费用、服务参数采用、最大容量及默认 reasoning 均为 UNKNOWN。"
        f"实际返回模型：{', '.join(result['actual_models'])}。", '',
        '本报告不认证题目正确性、新颖性或难度效果；审计新增 API 调用 0，未修改生产代码、运行输入或冻结文件。']
    failed = [row['check'] for row in checks if not row['pass']]
    if failed:
        lines += ['', '未通过的检查：' + json.dumps(failed, ensure_ascii=False)]
    (REPORTS / 'EXECUTION_AUDIT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({key: result[key] for key in ('status', 'run_status', 'logical_calls',
        'http_attempts', 'stage_counts', 'usage', 'technical_failure_count', 'verified_cost')}, ensure_ascii=False))
    print(json.dumps({'failed_checks': failed}, ensure_ascii=False))


if __name__ == '__main__':
    main()
