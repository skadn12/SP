"""Read-only generation audit and separately attributed full-transcript review.

No model client is instantiated here. Semantic judgments are supplied by direct
transcript reading, never inferred from string similarity or candidate labels.
"""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path

from ..core import atomic_json, atomic_text, canonical_json, sha256_file, sha256_bytes
from ..h4_qa_mining_hybrid_generation_v1.contract_guided_generator import build_prompt as guided, build_repair_prompt as guided_repair
from ..h4_qa_mining_hybrid_generation_v1.free_exploration_generator import build_prompt as free, build_repair_prompt as free_repair
from ..h4_qa_mining_probe_v1.intent_miner import build_intent_prompt
from ..simple_append_v1.live_runtime import load_json_object_strict
from .runtime import FIRST_ROOTS, MODE, PLAN_TOTAL, LIMIT_TOTAL, LIMIT_PER_ROOT, root_path, state_paths, read, now
from .common_sources import P1

REPORT_VERSION = 'h4_r02_feedback_transfer_p2_2_report_v1'
FIELDS = ('TRANSFER_TRACEABLE', 'EVIDENCE_GROUNDED', 'WRITER_REALIZATION', 'NOVELTY', 'GOLD_SUPPORT', 'BINDING_INTENT_MISMATCH')
JUDGMENTS = {'PASS', 'FAIL', 'UNDETERMINED', 'NOT_APPLICABLE'}


def optional(path, default):
    return read(path) if path.is_file() else default


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def write_jsonl(path, rows):
    atomic_text(path, ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows))


def ref(path):
    return {'path': str(path), 'sha256': sha256_file(path)} if path.is_file() else {'path': str(path), 'status': 'NOT_RUN'}


def parse_response(root, request_id):
    path = root / 'raw/responses' / f'{request_id}.json.txt'
    if not path.is_file():
        return None
    try:
        return load_json_object_strict(path.read_text())
    except (ValueError, json.JSONDecodeError):
        return None


def mechanical_report(run):
    from .run import make_hooks, verify_frozen
    run = Path(run)
    states = [read(path) for path in state_paths(run)]
    completed = [state for state in states if state.get('status') == 'completed']
    transport = jsonl(run / 'audit/transport_events.jsonl')
    outcomes = [row for row in transport if row.get('event') == 'logical_call_transport_outcome']
    usage = Counter()
    for row in completed:
        usage.update({key: value for key, value in (row.get('response_metadata', {}).get('usage') or {}).items()
                      if isinstance(value, (float, int))})
    accounting = {'report_version': REPORT_VERSION, 'planned_logical_calls': PLAN_TOTAL,
        'maximum_logical_calls': LIMIT_TOTAL, 'logical_calls': len(states),
        'completed_calls': len(completed), 'failed_or_unresolved_calls': len(states) - len(completed),
        'unknown_billing_status_calls': len(states) - len(completed),
        'miner_calls': sum(row['request_metadata']['mode'] == 'intent_miner' for row in states),
        'writer_primary_calls': sum(row['request_metadata']['mode'] != 'intent_miner' and row['request_metadata']['attempt'] == 'primary' for row in states),
        'repair_calls': sum(row['request_metadata']['attempt'] == 'repair' for row in states),
        'http_attempts': sum(row.get('event') == 'http_attempt_started' for row in transport),
        'http_retries': sum(max(0, row['transport_attempt_count'] - 1) for row in outcomes),
        'transport_failures': sum(row['status'] == 'failed' for row in outcomes),
        'timeout_calls': sum(bool(row.get('timeout')) for row in outcomes),
        'completion_usage': dict(usage), 'finish_reason_counts': dict(Counter(row.get('response_metadata', {}).get('finish_reason') or 'unknown' for row in completed)),
        'monetary_cost': None, 'monetary_cost_note': 'No provider billing amount supplied; actual usage and unknown billable states are retained. No price inferred.',
        'solver_calls': 0, 'memory_calls': 0}
    roots, traces, batch_rows, prompt_errors = [], [], [], []
    requests_audit = []
    for root_id in FIRST_ROOTS:
        root = root_path(run, root_id)
        expanded_path = run / 'inputs' / root_id / 'expanded_dialogue_clean.json'
        if not expanded_path.is_file():
            roots.append({'root_id': root_id, 'status': 'BLOCKED', 'reason': 'SOURCE_DRIFT'})
            continue
        expanded = read(expanded_path)
        intents = optional(root / 'intents.json', {})
        contracts = optional(root / 'reasoning_contracts.json', [])
        bindings = optional(root / 'binding_validation.json', {})
        records = optional(root / 'candidate_generation.json', [])
        candidates = optional(root / 'candidates_valid.json', [])
        valid_ids = {row['candidate_id'] for row in candidates}
        root_states = [row for row in states if row['request_metadata']['root_id'] == root_id]
        cards = read(run / 'inputs' / root_id / 'observation_cards.json')
        hooks = make_hooks(run, root_id)
        # Rebuild prompts entirely from original builders plus deterministic hooks;
        # sidecars go in reports/reconstruction, never overwrite generated inputs.
        scratch = run / 'reports/reconstruction' / root_id
        miner = hooks.miner_prompt(build_intent_prompt(expanded), expanded=expanded, root_id=root_id, root_dir=scratch)
        raw_intents = parse_response(root, f'{root_id}-intent-miner')
        if (root / 'intents.json').is_file():
            hooks.after_intents(raw_intents, intents['validation'], contracts,
                                expanded=expanded, root_id=root_id, root_dir=scratch)
            if read(scratch / 'binding_validation.json') != bindings:
                prompt_errors.append(root_id + ':binding_validation_reconstruction_mismatch')
        expected_prompts = {f'{root_id}-intent-miner': miner['prompt']}
        for contract in contracts:
            for source in ('contract_guided', 'free_exploration'):
                suffix = hooks.writer_suffix(contract, source, expanded=expanded, root_id=root_id, root_dir=scratch)
                primary_id = f"{root_id}-{source}-generator-{contract['intent_id']}"
                repair_id = f"{root_id}-{source}-repair-{contract['intent_id']}"
                base = guided(expanded, contract) if source == 'contract_guided' else free(expanded, contract)
                expected_prompts[primary_id] = base['prompt'] + ('\n' + suffix if suffix else '')
                old_raw = parse_response(root, primary_id)
                base_repair = guided_repair(expanded, contract, old_raw, source) if source == 'contract_guided' else free_repair(expanded, contract, old_raw)
                expected_prompts[repair_id] = base_repair['prompt'] + ('\n' + suffix if suffix else '')
        for state in root_states:
            rid = state['request_id']
            path = root / 'raw/prompts' / f'{rid}.txt'
            expected = expected_prompts.get(rid)
            matches = path.is_file() and expected is not None and path.read_text() == expected
            if not matches:
                prompt_errors.append(rid + ':actual_prompt_vs_original_builder_and_brief_mismatch')
            request_audit = optional(root / 'writer_briefs/requests' / f'{rid}.json', {})
            if state['request_metadata']['mode'] != 'intent_miner' and not request_audit:
                prompt_errors.append(rid + ':missing_writer_request_sidecar')
            if request_audit:
                brief_path = Path(request_audit['brief_path'])
                brief = read(brief_path)
                if sha256_bytes(brief['rendered_suffix'].encode()) != request_audit['brief_sha256']:
                    prompt_errors.append(rid + ':brief_hash_mismatch')
                if request_audit['prompt_sha256'] != state['prompt_sha256']:
                    prompt_errors.append(rid + ':request_prompt_hash_mismatch')
            requests_audit.append({'root_id': root_id, 'request_id': rid, 'state': state['status'],
                'original_builder_plus_exact_brief': matches, 'sidecar': request_audit,
                'prompt': ref(path), 'raw': ref(root / 'raw/responses' / f'{rid}.json.txt')})
        replaced_ids = {row['repair_of'] for row in records if row.get('repair_of')}
        for record in records:
            rid = record['request_id']
            sidecar = optional(root / 'writer_briefs/requests' / f'{rid}.json', {})
            validation = record['validation']
            batch = {'root_id': root_id, 'request_id': rid, 'intent_id': record['intent_id'],
                'generation_source': record['generation_source'], 'attempt': 'repair' if record.get('repair_of') else 'primary',
                'repair_of': record.get('repair_of'), 'replaced_by_repair': rid in replaced_ids,
                'minimum_acceptance': validation.get('minimum_acceptance'),
                'normalizer_errors': validation.get('errors'), 'returned_candidate_count': len(validation.get('candidates') or []),
                'assigned_feedback_mode': MODE, 'effective_feedback_mode': sidecar.get('effective_feedback_mode'),
                'fallback_reason': sidecar.get('fallback_reason')}
            batch_rows.append(batch)
            for candidate in validation.get('candidates') or []:
                cid = f"{root_id}:{candidate['candidate_id']}"
                state = ('replaced' if rid in replaced_ids else 'accepted' if cid in valid_ids and candidate['schema_valid']
                         and (validation.get('minimum_acceptance') or {}).get('accepted') else 'rejected')
                traces.append({**batch, 'candidate_id': cid, 'source_row_index': candidate['source_row_index'],
                    'question_type': candidate['question_type'], 'question': candidate['question'], 'candidate_status': state,
                    'schema_valid': candidate['schema_valid'], 'schema_errors': candidate['schema_errors'],
                    'source_feedback_ids': sidecar.get('source_feedback_ids', []),
                    'brief_sha256': sidecar.get('brief_sha256'),
                    'source_provenance': ref(run / 'inputs' / root_id / 'provenance.json'),
                    'source_observations': ref(run / 'inputs' / root_id / 'observation_cards.json'),
                    'binding_validation': ref(root / 'binding_validation.json'),
                    'intent_contract': ref(root / 'reasoning_contracts.json'),
                    'actual_prompt': ref(root / 'raw/prompts' / f'{rid}.txt'),
                    'raw_response': ref(root / 'raw/responses' / f'{rid}.json.txt'),
                    'program_validation': ref(root / 'validity_audit.json'),
                    'final_candidate_pool': ref(root / 'candidates_valid.json')})
        effective = Counter()
        fallback = Counter()
        for contract in contracts:
            for source in ('contract_guided', 'free_exploration'):
                entry = optional(root / 'writer_briefs' / str(contract['intent_id']) / f'{source}.json', {})
                if entry:
                    effective[entry['effective_feedback_mode']] += 1
                    if entry.get('fallback_reason'):
                        fallback[entry['fallback_reason']] += 1
        negative_ok = True
        if root_id == FIRST_ROOTS[2]:
            negative_ok = cards['observation_cards'] == [] and cards['feedback_status'] == 'NO_ACTIVE_FEEDBACK'
            forbidden = ('"historical_gold"', '"solver_observations"', '"canonical_prediction_counts"', '"correct_count"', '"target_distinction"')
            # Whole Miner includes binding schema property names; only appended
            # root observations may contain none of the inactive data fields.
            for state in root_states:
                prompt = (root / 'raw/prompts' / f"{state['request_id']}.txt").read_text()
                if state['request_metadata']['mode'] == 'intent_miner':
                    prompt = prompt.split('ROOT OBSERVATIONS AND HISTORICAL TRACKING:\n', 1)[-1]
                if any(marker in prompt for marker in forbidden):
                    negative_ok = False
            if not negative_ok:
                prompt_errors.append(root_id + ':inactive_data_isolation_failed')
        roots.append({'root_id': root_id, 'status': optional(root / 'qa_generation_report.json', {}).get('status', 'NOT_RUN'),
            'valid_intents': len(contracts), 'valid_proposed_bindings': bindings.get('valid_proposed_binding_count', 0),
            'mechanically_valid_binding_rows': sum(row.get('mechanical_valid', False) for row in bindings.get('bindings', [])),
            'binding_errors': [row for row in bindings.get('bindings', []) if not row.get('mechanical_valid')],
            'envelope_errors': bindings.get('envelope_errors', []), 'effective_modes_per_primary_batch': dict(effective),
            'fallback_reasons_per_primary_batch': dict(fallback),
            'guided_candidates': sum(row['generation_source'] == 'contract_guided' for row in candidates),
            'free_candidates': sum(row['generation_source'] == 'free_exploration' for row in candidates),
            'question_types': dict(Counter(row['question_type'] for row in candidates)),
            'valid_candidates': len(candidates), 'repairs': sum(row['request_metadata']['attempt'] == 'repair' for row in root_states),
            'logical_calls': len(root_states), 'completed_calls': sum(row['status'] == 'completed' for row in root_states),
            'failures': len(root_states) - sum(row['status'] == 'completed' for row in root_states),
            'intent_errors': (intents.get('validation') or {}).get('errors', []),
            'no_inactive_feedback_data': negative_ok, 'feedback_status': cards['feedback_status']})
    accounting['roots'] = roots
    accounting['status'] = 'COMPLETED' if all((root_path(run, root_id) / 'qa_generation_report.json').is_file() for root_id in FIRST_ROOTS) else 'IN_PROGRESS'
    atomic_json(run / 'audit/call_accounting.json', accounting)
    atomic_json(run / 'reports/binding_validation_report.json', {'report_version': REPORT_VERSION, 'roots': roots,
        'mechanical_not_semantic': True, 'requests': requests_audit, 'errors': prompt_errors})
    atomic_json(run / 'reports/logical_batch_status.json', {'batches': batch_rows, 'logical_requests': requests_audit})
    write_jsonl(run / 'reports/source_to_writer_trace.jsonl', traces)
    return accounting


def p1_preservation(run):
    initial = read(run / 'audit/p1_preservation_initial.json')
    expected = {row['path']: row for row in initial['files']}
    paths = {str(path.relative_to(P1)): path for path in P1.rglob('*') if path.is_file()}
    missing, added = sorted(set(expected) - set(paths)), sorted(set(paths) - set(expected))
    changed = [name for name in sorted(set(expected) & set(paths))
        if paths[name].stat().st_size != expected[name]['size'] or sha256_file(paths[name]) != expected[name]['sha256']]
    result = {'status': 'PASS' if not (missing or added or changed) else 'FAIL', 'baseline_label': initial['baseline_label'],
        'checked_at_utc': now(), 'baseline_file_count': len(expected), 'current_file_count': len(paths),
        'missing': missing, 'added': added, 'changed': changed,
        'complete_byte_inventory_compared': True, 'initial_inventory': ref(run / 'audit/p1_preservation_initial.json')}
    atomic_json(run / 'audit/p1_preservation.json', result)
    return result


def final_report(run):
    from .run import verify_frozen
    run = Path(run)
    accounting = mechanical_report(run)
    preservation = p1_preservation(run)
    mechanics = read(run / 'reports/binding_validation_report.json')
    token = optional(run / 'audit/token_report.json', {'status': 'NOT_RUN', 'requests': []})
    tests = optional(run / 'audit/test_report.json', {'status': 'NOT_RUN'})
    preflight = optional(run / 'audit/preflight_report.json', {'status': 'NOT_RUN'})
    freeze_status, freeze_error = 'PASS', None
    try:
        verify_frozen(run)
    except BaseException as error:
        freeze_status, freeze_error = 'FAIL', str(error)
    token_states = {row['request_id']: row for row in token.get('requests', [])}
    actual_states = [read(path) for path in state_paths(run)]
    token_complete = all(row['request_id'] in token_states and token_states[row['request_id']]['status'] == 'PASS' for row in actual_states)
    checks = {'tests': tests['status'] == 'PASS', 'preflight': preflight['status'] == 'PASS',
        'all_three_roots_finished': accounting['status'] == 'COMPLETED',
        'frozen_identity': freeze_status == 'PASS', 'p1_preservation': preservation['status'] == 'PASS',
        'actual_prompts_briefs_isolation': not mechanics['errors'],
        'budget': accounting['logical_calls'] <= LIMIT_TOTAL and all(row.get('logical_calls', 0) <= LIMIT_PER_ROOT for row in accounting['roots']),
        'all_actual_requests_token_checked': token_complete, 'negative_isolation': accounting['roots'][-1].get('no_inactive_feedback_data', False),
        'no_orphan_reservations': all(row['status'] == 'completed' or (root_path(run, row['request_metadata']['root_id']) / 'ledger/observed_failures' / f"{row['request_id']}.json").is_file() for row in actual_states)}
    reviews, binding_reviews, coverage_errors = [], [], []
    per_positive = {}
    for root_id in FIRST_ROOTS:
        candidates = optional(root_path(run, root_id) / 'candidates_valid.json', [])
        review_path = run / 'reports/reviews' / f'{root_id}.json'
        review = optional(review_path, {})
        if candidates and (not review or not review.get('full_transcript_read') or review.get('reviewer') != 'Codex direct transcript review'):
            coverage_errors.append(root_id + ':missing_attributed_full_transcript_review')
        if review and review.get('transcript_sha256') != sha256_file(run / 'inputs' / root_id / 'expanded_dialogue_clean.json'):
            coverage_errors.append(root_id + ':review_transcript_sha_mismatch')
        actual_reviews = {row['candidate_id']: row for row in review.get('candidates', [])}
        if len(actual_reviews) != len(review.get('candidates', [])) or set(actual_reviews) != {row['candidate_id'] for row in candidates}:
            coverage_errors.append(root_id + ':review_candidate_coverage_mismatch')
        for candidate in candidates:
            row = actual_reviews.get(candidate['candidate_id'])
            if row is None:
                continue
            for field in FIELDS:
                judgment = row.get(field, {})
                if judgment.get('judgment') not in JUDGMENTS or not judgment.get('reason'):
                    coverage_errors.append(candidate['candidate_id'] + ':' + field + '_missing')
            if row.get('GOLD_SUPPORT', {}).get('classification') not in {'supported', 'contradicted', 'unsupported', 'underdetermined'}:
                coverage_errors.append(candidate['candidate_id'] + ':Gold_classification_missing')
            if candidate['question_type'] != 'ordering':
                option_ids = {entry['option_id'] for entry in row.get('options_review', [])}
                if option_ids != set(candidate['options']):
                    coverage_errors.append(candidate['candidate_id'] + ':option_review_coverage')
            else:
                item_ids = {entry['item_id'] for entry in row.get('ordering_item_review', [])}
                if item_ids != set(candidate['items']) or not row.get('ordering_relation_review'):
                    coverage_errors.append(candidate['candidate_id'] + ':ordering_item_or_relation_review_coverage')
            reviews.append({**row, 'root_id': root_id, 'question_type': candidate['question_type'],
                'reviewer': review.get('reviewer'), 'review_version': review.get('review_version'),
                'reviewed_at_utc': review.get('reviewed_at_utc'), 'review_source': ref(review_path),
                'historical_gold_or_admission_changed': False})
        binding_reviews.extend({**row, 'root_id': root_id, 'review_source': ref(review_path)} for row in review.get('bindings', []))
        if root_id in FIRST_ROOTS[:2]:
            grounded = {row['intent_id'] for row in review.get('bindings', []) if row.get('mechanical_valid')
                and row.get('EVIDENCE_GROUNDED', {}).get('judgment') == 'PASS'
                and row.get('BINDING_INTENT_MISMATCH', {}).get('judgment') == 'PASS'}
            realized = [row['candidate_id'] for row in actual_reviews.values()
                if row.get('generation_source') == 'contract_guided' and row.get('intent_id') in grounded
                and all(row.get(field, {}).get('judgment') == 'PASS' for field in ('TRANSFER_TRACEABLE', 'EVIDENCE_GROUNDED', 'WRITER_REALIZATION', 'NOVELTY'))
                and row.get('GOLD_SUPPORT', {}).get('classification') == 'supported']
            per_positive[root_id] = {'grounded_binding_intents': sorted(grounded), 'qualifying_guided_qa': realized,
                                    'minimum_local_feasibility_met': bool(grounded and realized)}
    write_jsonl(run / 'reports/candidate_content_review.jsonl', reviews)
    atomic_json(run / 'reports/binding_content_review.json', {'rows': binding_reviews})
    atomic_json(run / 'audit/content_review_coverage.json', {'status': 'PASS' if not coverage_errors else 'INCOMPLETE',
        'reviewed_candidates': len(reviews), 'expected_candidates': sum(row.get('valid_candidates', 0) for row in accounting['roots']), 'errors': coverage_errors})
    content = 'INCONCLUSIVE' if coverage_errors or accounting['status'] != 'COMPLETED' else 'PASS' if all(row['minimum_local_feasibility_met'] for row in per_positive.values()) else 'FAIL'
    engineering = 'PASS' if all(checks.values()) else 'FAIL'
    gold = dict(Counter(row['GOLD_SUPPORT']['classification'] for row in reviews))
    supplied_readiness = optional(run / 'reports/readiness_assessment.json', {})
    ready = 'READY' if engineering == content == 'PASS' and supplied_readiness.get('status') == 'READY' else 'NOT_READY' if engineering == 'FAIL' or content == 'FAIL' or supplied_readiness.get('status') == 'NOT_READY' else 'INCONCLUSIVE'
    result = {'status': 'COMPLETED' if accounting['status'] == 'COMPLETED' and not coverage_errors else 'INCOMPLETE',
        'report_version': REPORT_VERSION, 'engineering_acceptance': engineering, 'engineering_checks': checks,
        'freeze_error': freeze_error, 'content_feasibility': content, 'positive_roots': per_positive,
        'expansion_readiness': ready, 'readiness_assessment': supplied_readiness,
        'gold_diagnostics': gold, 'review_count': len(reviews), 'review_coverage_errors': coverage_errors,
        'call_accounting': accounting, 'causal_effect_or_difficulty_claim': False,
        'stopped_after_first_acceptance': True, 'solver_calls': 0, 'memory_calls': 0}
    atomic_json(run / 'reports/first_acceptance_report.json', result)
    lines = [f'# P2-2 首次三 root 验收（{REPORT_VERSION}）', '',
        f'工程验收：{engineering}；内容可行性：{content}；扩大就绪：{ready}。', '',
        '正式路径复用同一个原 QA generator；原文件仅 generation.py 增加默认关闭的 hooks。共同基础独立冻结来源资格与 Miner 无具象示例模板；P2-2 增加自产 binding、独立校验和 Guided/Free brief。', '',
        'Guided:Free 请求题目目标保持 1:2；两模式最低数量均为 2 SC + 1 MS + 1 ordering。只在低于原最低配额时整批 repair，结果替换 primary；未增语义或难度准入。', '',
        '| root | valid intents | proposed bindings | Guided / Free QA | repair | calls | failures |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in accounting['roots']:
        lines.append(f"| {row['root_id']} | {row.get('valid_intents', 0)} | {row.get('valid_proposed_bindings', 0)} | {row.get('guided_candidates', 0)} / {row.get('free_candidates', 0)} | {row.get('repairs', 0)} | {row.get('logical_calls', 0)} | {row.get('failures', 0)} |")
    lines += ['', f"实际逻辑调用 {accounting['logical_calls']}/39，计划 21；HTTP 尝试 {accounting['http_attempts']}，重试 {accounting['http_retries']}。未知付费状态 {accounting['unknown_billing_status_calls']}；completion usage：{json.dumps(accounting['completion_usage'])}。金额未由服务提供，未推算价格。", '',
        f"程序测试：{tests.get('tests_run')} 项，{tests['status']}。实际请求逐一计入 chat wrapper、原输出预算和固定 512 token 余量；token 检查：{token.get('status')}。服务身份和依据见 audit/token_identity.json。", '',
        '按 primary 批次的 effective mode / fallback：']
    lines.extend(f"- {row['root_id']}: {json.dumps(row.get('effective_modes_per_primary_batch', {}), ensure_ascii=False)} / {json.dumps(row.get('fallback_reasons_per_primary_batch', {}), ensure_ascii=False)}" for row in accounting['roots'])
    lines += ['', f"全部最终程序通过候选已审阅 {len(reviews)}；Gold 诊断：{json.dumps(gold, ensure_ascii=False)}。覆盖问题：{json.dumps(coverage_errors, ensure_ascii=False)}。", '',
        '评审为 Codex direct transcript review，使用完整冻结当前 transcript；不是独立人类盲审。逐多选选项、排序条目和关系检查保存在 reports/reviews/；这些诊断不修改 Gold、候选或准入。', '']
    for root_id, finding in per_positive.items():
        lines.append(f"{root_id}：有据绑定 {finding['grounded_binding_intents']}；满足局部内容条件的新 Guided QA {finding['qualifying_guided_qa']}。")
    lines += ['', '扩大判断依据：' + str(supplied_readiness.get('reason', '尚缺最终人工可审阅综合判断。')), '',
        '本次为单臂三 root 首次验收，只检验可追踪传递与局部内容可行性，不证明反馈因果有效或有效难度提升，不与 P1 直接作效果相减。', '',
        f"P1 完整 {preservation['baseline_file_count']} 文件字节/数量保全：{preservation['status']}。没有 Solver、Memory 或训练调用；完成本次后停止，不扩到六 root。"]
    atomic_text(run / 'reports/first_acceptance_report.md', '\n'.join(lines) + '\n')
    atomic_text(run / 'RUN_STATUS.md', f"# P2-2 首次验收状态\n\nUpdated: {now()}\n\n执行：{result['status']}\n工程：{engineering}\n内容：{content}\n扩大就绪：{ready}\n逻辑调用：{accounting['logical_calls']}/39；review：{len(reviews)}\n\n已停止；无 Solver、Memory、训练或六 root 自动扩展。\n")
    atomic_text(run / 'ARTIFACT_INDEX.md', '# P2-2 验收产物\n\n' + '\n'.join([
        '- reports/first_acceptance_report.md / .json：最终工程、内容与扩大判断。',
        '- implementation_manifest.json、execution_contract.json：版本、模式、范围和固定参数。',
        '- common_base/：六 seed 来源资格、共同 Miner 修正、两模式离线 Miner prompt。',
        '- input_path_audit.json、inputs/：三个 exact transcript、观察卡、history-only 和完整来源。',
        '- policy_snapshot/、audit/frozen_manifest.json：policy/schema/code/tokenizer 与冻结 SHA。',
        '- qa/shards/shard-00/roots/：原 generator 全量 prompt、raw、intents、contracts、binding、brief 和候选。',
        '- audit/test_report.json、hook_disabled_parity.json、preflight_report.json、token_identity.json、token_report.json：程序与服务长度依据。',
        '- audit/call_accounting.json、call_events.jsonl、transport_events.jsonl：逻辑调用、失败/缓存和 HTTP 账本。',
        '- reports/source_to_writer_trace.jsonl、logical_batch_status.json：含 rejected/replaced 的全部新 candidate 与批次。',
        '- reports/binding_validation_report.json、binding_content_review.json：机械与内容独立审核。',
        '- reports/candidate_content_review.jsonl、reports/reviews/、audit/content_review_coverage.json：全部最终候选的逐项阅读。',
        '- audit/p1_preservation.json：完整 P1 文件 inventory/hash 保全。',
    ]) + '\n')
    return result
