"""Read-only experiment audit; writes reports outside the frozen run."""
import hashlib
import importlib.util
import json
from pathlib import Path

DEST = Path(__file__).resolve().parent
RUN = DEST / 'prepared_run'
spec = importlib.util.spec_from_file_location('postcheck_audit_target', DEST / 'postcheck.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

checks = []
def check(name, okay, detail=None):
    checks.append({'name': name, 'status': 'PASS' if okay else 'FAIL', 'detail': detail})

for label, path in [('current_postcheck', RUN / 'freeze_manifest.json'),
                    ('original_downstream', p.SOURCE / 'freeze_manifest.json')]:
    manifest = read(path)
    drift = [f['path'] for f in manifest['files']
             if not Path(f['path']).is_file() or sha(f['path']) != f['sha256']]
    check(label + '_frozen_files', not drift, {'files': len(manifest['files']), 'drift': drift})

summary = read(RUN / 'generation_summary.json')
ledger = read(RUN / 'execution/development_call_ledger.json')
plan = read(RUN / 'frozen/plan.json')
slots = read(RUN / 'frozen/slots.json')
profile = read(RUN / 'frozen/provider_profile.json')
records = ledger['records']
check('completed_all_fixed_slots', summary['status'] == 'GENERATION_FINISHED'
      and summary['completed_review_count'] == 8 and len(records) == 8)
check('logical_and_http_budget', ledger['logical_call_count'] == 8
      and ledger['http_attempt_count'] == 8 and ledger['maximum_http_attempts'] == 8)
check('fixed_order_exact', [(r['root_id'], r['stage']) for r in records] ==
      [(s['root_id'], s['slot_id']) for s in p.SLOTS])
check('no_retry', ledger['automatic_retry_count'] == 0 and
      all(r['retry_count'] == 0 and r['http_attempt_count'] == 1 for r in records))
check('no_resume', ledger['automatic_resume'] is False and
      read(RUN / 'EXECUTION_STARTED.json')['automatic_resume'] is False)
check('no_extra_stage_artifacts', len(list((RUN / 'execution/calls').glob('*/*/request.json'))) == 8)
check('public_profile_identity', profile['endpoint'] == 'https://ai.tju.edu.cn/api/v3/chat/completions'
      and profile['model'] == 'tju-llm')
expected_parameters = {'temperature': 0.1, 'max_tokens': 8192, 'response_format': {'type': 'json_object'}}
check('explicit_parameters', profile['request_parameters'] == expected_parameters
      and profile['output_limit'] == 8192)
check('service_unknowns_retained', plan['reasoning'] == 'SERVICE_DEFAULT_UNKNOWN'
      and plan['actual_parameter_adoption'] == 'UNKNOWN' and plan['cost'] == 'UNKNOWN')

per_slot = []
for slot, record in zip(slots, records):
    sid = slot['slot_id']
    frozen = RUN / 'frozen' / sid
    call = RUN / 'execution' / record['artifact_directory']
    request = read(call / 'request.json')
    payload = request['payload']
    metadata = read(call / 'response_metadata.json')
    original = read(frozen / 'original_raw.json')
    source = read(frozen / 'source.json')
    messages = read(frozen / 'messages.json')
    parsed = read(call / 'parsed_response.json')
    outcome = read(RUN / 'results' / sid / 'outcome.json')
    check(sid + '_request_identity', payload == read(frozen / 'payload.json')
          and payload['messages'] == messages and payload['model'] == 'tju-llm'
          and {k: v for k, v in payload.items() if k not in {'messages', 'model'}} == expected_parameters)
    check(sid + '_hash_identity', p.digest(payload['messages']) == record['messages_sha256']
          and p.digest(payload) == record['payload_sha256'])
    check(sid + '_full_source_unchanged', (frozen / 'transcript.txt').read_bytes() ==
          (p.SOURCE / 'qa/roots' / slot['root_id'] / 'qa_transcript.txt').read_bytes()
          and source == read(p.SOURCE / 'qa/roots' / slot['root_id'] / 'qa_source.json'))
    check(sid + '_prompt_exact_no_evaluation_injection', messages == p.build_messages(slot, original,
          (frozen / 'transcript.txt').read_text()) and original.get('gold') is None)
    check(sid + '_original_raw_identity', original ==
          read(slot['original_raw_path'])['candidates'][slot['raw_row_index'] - 1])
    pool = read(slot['original_pool_path'])
    selected = next(c for c in pool if c['candidate_id'] == slot['candidate_id'])
    derivation_path = Path(slot['original_pool_path']).parent / selected['ordering_derivation']['path']
    derived_original = read(derivation_path)
    check(sid + '_original_derivation_identity', sha(derivation_path) == selected['ordering_derivation']['sha256']
          and derived_original['raw_row'] == original and
          derived_original['identity']['candidate_id'] == slot['candidate_id'].rsplit(':', 1)[0])
    cap = read(call / 'capacity.json')
    check(sid + '_capacity_proxy_honest', cap['status'] == 'PASS' and cap['truncation'] is False
          and cap['service_context_capacity'] == 'UNKNOWN'
          and cap['counting_tokenizer'] == 'FROZEN_QWEN3_14B_PROXY_NOT_TJU_EXACT')
    check(sid + '_one_complete_http', record['status'] == 'COMPLETED'
          and metadata['http_attempt_count'] == 1 and metadata['transport_retry_count'] == 0
          and metadata['rate_limit_retry_count'] == 0 and metadata['http_status'] == 200
          and metadata['finish_reason'] == 'stop' and metadata['timeout'] is False)
    check(sid + '_actual_model', metadata['actual_model'] == 'qwen3.6-35b-a3b'
          and metadata['requested_model'] == 'tju-llm')
    check(sid + '_response_and_usage_preserved', metadata['raw_response']['choices'][0]['message']['content'] ==
          (call / 'response.txt').read_text() and metadata['usage'] == record['usage']
          and metadata['raw_response']['usage'] == metadata['usage'])
    check(sid + '_usage_reconciles', metadata['input_tokens'] + metadata['output_tokens'] == metadata['total_tokens']
          and record['total_tokens'] == metadata['total_tokens']
          and record['reasoning_tokens'] == 'UNKNOWN')
    check(sid + '_parsed_review_replays', p.materialize(parsed, slot, original, source) == outcome)
    if outcome['program_acceptance']:
        rebound = outcome['rebound_contract']
        by_id = {n['item_id']: n for n in original['event_nodes']}
        immutable = ('item_id', 'item_text', 'actor', 'event_state')
        check(sid + '_displayed_event_identity_preserved', all(
            all(n.get(k) == by_id[n['item_id']].get(k) for k in immutable)
            for n in rebound['event_nodes']))
        catalog = {t['turn_ref']: t['text'] for t in source['turns']}
        check(sid + '_program_fetches_complete_turns', all(
            sp['quote'] == catalog[sp['turn_ref']]
            for entity in rebound['event_nodes'] + rebound['temporal_relations']
            for sp in entity['evidence_spans']))
        check(sid + '_evidence_and_quote_ids_equal', all(
            set(e['evidence_refs']) == {sp['turn_ref'] for sp in e['evidence_spans']}
            for e in rebound['event_nodes'] + rebound['temporal_relations']))
        check(sid + '_semantic_not_program_certified', outcome['semantic_support_status'] == 'NOT_ASSESSED_BY_PROGRAM')
    per_slot.append({'slot_id': sid, 'root_id': slot['root_id'], 'model_decision': parsed['decision'],
        'program_acceptance': outcome['program_acceptance'], 'input_tokens': record['input_tokens'],
        'output_tokens': record['output_tokens'], 'total_tokens': record['total_tokens'],
        'finish_reason': record['finish_reason'], 'actual_model': record['actual_model']})

totals = {k: sum(r[k] for r in records) for k in ('input_tokens', 'output_tokens', 'total_tokens')}
check('cost_summary_all_calls', all(summary['cost'][k] == v for k, v in totals.items())
      and summary['cost']['logical_calls'] == 8 and summary['cost']['http_attempts'] == 8
      and summary['cost']['retries'] == 0 and summary['cost']['verified_cost'] == 'UNKNOWN'
      and summary['cost']['reasoning_tokens'] == 'UNKNOWN')
check('final_ledger_matches_summary', records == summary['calls'] and ledger['status'] == 'FINISHED')
check('service_parameter_adoption_unverified', all(
    read(RUN / 'execution' / r['artifact_directory'] / 'request.json')['confirmed_parameter_adoption'] == 'UNKNOWN'
    for r in records))

report = {'status': 'PASS' if all(c['status'] == 'PASS' for c in checks) else 'FAIL',
    'audit_type': 'Codex offline execution/provenance audit, not semantic certification',
    'audit_new_external_requests': 0, 'check_count': len(checks),
    'failed_checks': [c for c in checks if c['status'] != 'PASS'],
    'cost': summary['cost'], 'checks': checks, 'per_slot': per_slot,
    'limits': [
        'No Gold array, previous Codex diagnosis, or corrected question entered the prompt. Original proposed temporal edges remained visible and imply the prior order: this is post-generation checking, not blind solving.',
        'Original item text/actor/state stay fixed. Minor question-stem changes are permitted but their semantic scope needs offline review.',
        'Model explanations/support labels can contradict each other even when the graph passes. The harness checks structure and exact ID existence, not semantic truth.',
        'School reasoning mode and actual parameter adoption remain UNKNOWN. Frozen Qwen tokenizer is a local proxy, not verified school service capacity.',
        'reasoning_tokens_known_sum=0 denotes no reported reasoning counts, not proof of zero reasoning tokens. Money remains UNKNOWN.',
        'One sampled postcheck for each of eight previously accepted ordering candidates from three roots does not establish a population error rate.'
    ]}
out = DEST / 'reports'
out.mkdir(exist_ok=True)
(out / 'HARNESS_AUDIT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
lines = ['# Qwen3.6 事后检查执行审计', '',
    f"状态：**{report['status']}**；{len(checks)} 项核对，失败 {len(report['failed_checks'])} 项。审计新增 API 调用 0。", '',
    '8 个固定原题槽位各发送一次，8 逻辑调用 / 8 HTTP，无重试、回退或额外采样。原生产 run 与当前冻结文件保持一致。', '',
    f"实际返回模型均为 `qwen3.6-35b-a3b`；全部 HTTP 200、finish_reason=stop。输入 {totals['input_tokens']:,}、输出 {totals['output_tokens']:,}、总计 {totals['total_tokens']:,} tokens；费用和 reasoning 分拆 UNKNOWN。", '',
    '| 槽位 | 模型决定 | 程序接收 | 输入 | 输出 | 总 tokens |',
    '|---|---|---:|---:|---:|---:|']
for r in per_slot:
    lines.append(f"| {r['slot_id']} | {r['model_decision']} | {r['program_acceptance']} | {r['input_tokens']} | {r['output_tokens']} | {r['total_tokens']} |")
lines += ['', '## 评估边界', '',
    '- 没有把先前 Codex 诊断、修正版或 Gold 数组发送给模型；但原时间边可推导旧顺序，因此并非对原答案完全盲。',
    '- 程序只回取完整原句、验证 ID 和图一致性；支持关系仍需离线语义判断。程序接收不能当作正确题数。',
    '- 四个选项及其人物、状态元数据未改；允许轻微题干澄清，需要检查是否真正轻微。',
    '- 学校上下文容量、参数实际采用、默认 reasoning 未验证；本地容量 PASS 只是 Qwen tokenizer 代理守卫。',
    '- reasoning 已知合计 0 只是没有返回该字段，不能解释成实际 reasoning 为 0。',
    '- 3 个 root、8 道已知题、每题一次检查，不能推出全库普遍错误率。', '',
    '完整逐项结果见 `HARNESS_AUDIT.json`。']
(out / 'HARNESS_AUDIT.md').write_text('\n'.join(lines) + '\n')
print(json.dumps({'status': report['status'], 'checks': len(checks), 'failures': report['failed_checks'], 'cost': summary['cost']}, ensure_ascii=False))
