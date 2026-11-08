"""Offline final-delivery audit; never invokes generation or compilation."""
import hashlib
import json
import re
from pathlib import Path

DEST = Path(__file__).resolve().parent
RUN = DEST / 'prepared_run'
REPORTS = DEST / 'reports'
SOURCE = Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_gate_downstream_v1/prepared_run')

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

checks = []
def check(name, okay, detail=None):
    checks.append({'name': name, 'status': 'PASS' if okay else 'FAIL', 'detail': detail})

slots = read(RUN / 'frozen/slots.json')
slot_map = {s['slot_id']: s for s in slots}
summary = read(RUN / 'generation_summary.json')
report = read(REPORTS / 'FINAL_REPORT.json')
rows = read(REPORTS / 'SLOT_REVIEW.json')
all_outputs = read(REPORTS / 'MODEL_POSTCHECK_ALL.json')
partitions = {k: read(REPORTS / f'MODEL_POSTCHECK_{k}.json') for k in ('KEEP', 'HOLD', 'EXCLUDED')}
all_map = {x['slot_id']: x for x in all_outputs}
row_map = {x['slot_id']: x for x in rows}
check('eight_unique_source_slots', len(slots) == len(slot_map) == 8)
check('eight_unique_delivered_rows', len(rows) == len(row_map) == 8 and set(row_map) == set(slot_map))
check('eight_unique_outputs', len(all_outputs) == len(all_map) == 8 and set(all_map) == set(slot_map))
partition_ids = {k: {r['slot_id'] for r in v} for k, v in partitions.items()}
check('partition_sizes', {k: len(v) for k, v in partitions.items()} == {'KEEP': 0, 'HOLD': 2, 'EXCLUDED': 6})
check('partitions_pairwise_disjoint', not(partition_ids['KEEP'] & partition_ids['HOLD'])
      and not(partition_ids['KEEP'] & partition_ids['EXCLUDED'])
      and not(partition_ids['HOLD'] & partition_ids['EXCLUDED']))
check('partitions_cover_all', set().union(*partition_ids.values()) == set(slot_map))
check('hold_identity', partition_ids['HOLD'] == {'Q04', 'Q05'})
check('partitions_no_duplicate_identity', all(len(v) == len({x['slot_id'] for x in v}) for v in partitions.values()))
for name, values in partitions.items():
    check(name + '_objects_identical_to_all', all(a == all_map[a['slot_id']] for a in values))

for sid, slot in slot_map.items():
    row, embedded = row_map[sid], all_map[sid]
    response_path = RUN / 'execution/calls' / slot['root_id'] / sid / 'parsed_response.json'
    response, outcome = read(response_path), read(RUN / 'results' / sid / 'outcome.json')
    check(sid + '_candidate_identity', row['candidate_id'] == embedded['candidate_id'] == slot['candidate_id'])
    check(sid + '_model_output_embedded_unmodified', embedded['model_review'] == response
          and embedded['model_output_modified_by_codex'] is False)
    check(sid + '_program_outcome_embedded_unmodified', embedded['program_outcome'] == outcome)
    check(sid + '_row_matches_actual', row['model_decision'] == response['decision']
          and row['program_acceptance'] == outcome['program_acceptance']
          and row['derived_gold'] == outcome.get('derived_gold')
          and row['review_disposition'] == embedded['disposition'])
    check(sid + '_all_paths_exist', all(Path(row[k]).is_file() for k in
          ('detail_report', 'response_path', 'transcript_path', 'outcome_path', 'original_raw_path', 'original_pool_path')))

accepted = [r for r in rows if r['program_acceptance']]
check('four_accepted_two_fail_two_hold', len(accepted) == 4
      and {r['slot_id'] for r in accepted if r['review_disposition'] == 'FAIL'} == {'Q01', 'Q08'}
      and {r['slot_id'] for r in accepted if r['review_disposition'] == 'HOLD'} == {'Q04', 'Q05'})
check('two_accepted_full_source_gold_supported', {r['slot_id'] for r in accepted
      if r['current_source_assessment'] == 'PASS'} == {'Q04', 'Q05'})
check('accepted_gold_unchanged', all(r['gold_changed'] is False for r in rows)
      and all(r['derived_gold'] == slot_map[r['slot_id']]['original_gold_not_sent'] for r in accepted))

# Whole-source support diagnosis is a fallible offline Codex reference.
# This audit reconciles the published node metric with actual labels and the
# detailed diagnosis, without recasting it as automated truth verification.
expected_node_errors = {'Q02': ['I3'], 'Q03': ['I4'], 'Q06': ['I3'], 'Q07': ['I2'], 'Q08': ['I3']}
actual_node_errors = {}
false_negatives = false_positives = node_count = 0
for sid, a in all_map.items():
    local = []
    for node in a['model_review']['item_checks']:
        node_count += 1
        reference = 'UNSUPPORTED' if (sid, node['item_id']) == ('Q08', 'I3') else 'SUPPORTED'
        if node['support'] != reference:
            local.append(node['item_id'])
            false_negatives += reference == 'SUPPORTED'
            false_positives += reference == 'UNSUPPORTED'
    if local:
        actual_node_errors[sid] = local
check('node_error_labels_reconcile', actual_node_errors == expected_node_errors
      and all(row_map[s]['node_fact_verdict_errors'] == expected_node_errors.get(s, []) for s in slot_map))
check('node_error_denominator', node_count == 32 and false_negatives == 4 and false_positives == 1)
metrics = report['metrics']
check('node_error_metric_defined', metrics['node_fact_verdict_errors']['numerator'] == 5
      and metrics['node_fact_verdict_errors']['denominator'] == 32
      and 'excludes local citation completeness' in metrics['node_fact_verdict_errors']['definition'])
check('program_wrong_release_metric', metrics['unsafe_among_program_accepted']['numerator'] == 2
      and metrics['unsafe_among_program_accepted']['denominator'] == 4
      and 'not counted as wrong answers' in metrics['unsafe_among_program_accepted']['definition'])
check('strict_zero_not_gold_zero', metrics['strict_complete_outputs']['numerator'] == 0
      and metrics['strict_complete_outputs']['denominator'] == 8
      and 'not wrong Gold' in metrics['strict_complete_outputs']['definition'])
check('preserved_two_full_source_orders', metrics['retained_full_source_correct_orders']['numerator'] == 2
      and metrics['retained_full_source_correct_orders']['denominator'] == 3)

occurrences = unknown = changed_nodes = added_nodes = 0
for sid, embedded in all_map.items():
    original = read(RUN / 'frozen' / sid / 'original_raw.json')
    source = read(RUN / 'frozen' / sid / 'source.json')
    review = embedded['model_review']
    valid = {t['turn_ref'] for t in source['turns']}
    old_union = {r for group in ('event_nodes', 'temporal_relations') for e in original[group] for r in e['evidence_refs']}
    returned = [r for group in ('item_checks', 'original_relation_checks', 'supported_relations')
                for e in review[group] for r in e['evidence_refs']]
    occurrences += len(returned)
    unknown += sum(r not in valid for r in returned)
    check(sid + '_no_new_ids_reconciles', not(set(returned) - old_union) and row_map[sid]['new_evidence_ids'] == [])
    by_id = {x['item_id']: x for x in review['item_checks']}
    for node in original['event_nodes']:
        old_set, new_set = set(node['evidence_refs']), set(by_id[node['item_id']]['evidence_refs'])
        changed_nodes += old_set != new_set
        added_nodes += bool(new_set - old_set)
check('citation_occurrence_denominator', occurrences == 132 and unknown == 0
      and metrics['citation_id_nonexistence']['denominator'] == 132
      and 'duplicates count repeatedly' in metrics['citation_id_nonexistence']['definition'])
check('node_rebinding_counts', changed_nodes == 3 and added_nodes == 0
      and metrics['nodes_with_added_evidence_ids']['denominator'] == 32)

check('reported_cost_matches_actual', report['cost'] == summary['cost']
      and summary['cost']['http_attempts'] == 8 and summary['cost']['logical_calls'] == 8
      and summary['cost']['retries'] == 0 and summary['cost']['total_tokens'] == 80866
      and summary['cost']['input_tokens'] == 70464 and summary['cost']['output_tokens'] == 10402)
check('money_and_reasoning_unknown', report['cost']['verified_cost'] == 'UNKNOWN'
      and report['cost']['reasoning_tokens'] == 'UNKNOWN')
check('root_cost_reconciles', sum(r['total_tokens'] for r in report['by_root']) == 80866
      and sum(r['reviews'] for r in report['by_root']) == 8)
check('no_finite_zero_output_unit_cost', report['cost_per_strict_usable_question'] == 'UNDEFINED_ZERO_STRICT_USABLE_OUTPUTS')
check('completed_stopped_scope', report['status'] == 'COMPLETED_STOPPED'
      and report['next_automatic_action'] == 'NONE'
      and report['additional_model_calls_allowed_after_completion'] is False
      and report['production_default_changed'] is False)

md = (REPORTS / 'FINAL_REPORT.md').read_text()
check('human_certification_not_claimed', '不是独立人工真值' in md
      and 'not human-certified ground truth' in report['method'])
check('population_error_rate_not_claimed', '不能推广为Qwen3.6在全库上的错误率' in md
      and any('not dataset-wide error rate' in x for x in report['limitations']))
check('zero_gold_accuracy_not_claimed', '此数不能称答案正确率0%' in md
      and 'Q04/Q05的全文Gold有据' in md)
check('original_edges_visibility_disclosed', '原关系可推旧顺序' in md
      and any('edges allow inference of original order' in x for x in report['limitations']))
check('no_internal_cause_overclaim', '不能断言模型内部原因' in md
      and any('one sample cannot establish its internal cause' in x for x in report['limitations']))

for name, path in [('postcheck158', RUN / 'freeze_manifest.json'),
                   ('downstream358', SOURCE / 'freeze_manifest.json')]:
    manifest = read(path)
    bad = [r['path'] for r in manifest['files'] if not Path(r['path']).is_file() or sha(r['path']) != r['sha256']]
    check(name + '_frozen_identity', not bad, {'files': len(manifest['files']), 'drift': bad})
check('execution_audit_present_pass', read(REPORTS / 'HARNESS_AUDIT.json')['status'] == 'PASS'
      and read(REPORTS / 'HARNESS_AUDIT.json')['check_count'] == 126)

missing = []
checked_links = 0
for path in [DEST / 'README.md', REPORTS / 'FINAL_REPORT.md']:
    for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
        if target.startswith(('http://', 'https://', '#')):
            continue
        target = target.split('#', 1)[0]
        resolved = Path(target) if target.startswith('/') else path.parent / target
        checked_links += 1
        if not resolved.is_file():
            missing.append({'file': str(path), 'target': target})
check('readme_and_report_links_exist', not missing, {'link_count': checked_links, 'missing': missing})
check('offline_rebuild_script_present', (DEST / 'compile_report.py').is_file())

result = {'status': 'PASS' if all(c['status'] == 'PASS' for c in checks) else 'FAIL',
    'method': 'Offline identity/accounting/report consistency audit; semantic labels remain Codex diagnosis, not independently certified truth.',
    'new_api_calls': 0, 'changed_prepared_run_files': 0, 'check_count': len(checks),
    'failed_checks': [c for c in checks if c['status'] == 'FAIL'], 'checks': checks,
    'partition_counts': {k: len(v) for k, v in partitions.items()},
    'cost': report['cost'], 'audited_report_sha256': sha(REPORTS / 'FINAL_REPORT.json'),
    'audited_all_outputs_sha256': sha(REPORTS / 'MODEL_POSTCHECK_ALL.json')}
(REPORTS / 'FINAL_DELIVERY_CHECK.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
lines = ['# 最终交付核对', '',
    f"**{result['status']}：{len(checks)} 项检查全部通过。**" if result['status'] == 'PASS' else f"FAIL：{len(result['failed_checks'])} 项失败。", '',
    '- 8 题身份完整，KEEP 0、HOLD 2、EXCLUDED 6，分区互斥；嵌入的模型原始返回与程序结果均未改写。',
    '- 4 道程序接收题对应 2 FAIL + 2 HOLD；Q04/Q05 全文 Gold 有据，报告没有将其解释成 4 个答案全错。',
    '- 5/32 节点支持标签错误明确为 4 假阴性 + 1 假阳性，分母不混入局部引文充分性和时间边；该语义参考仍为 Codex 离线诊断。',
    '- 0/132 未知引用以引用出现次数计数，不是 132 个独立证据或题目。29 个节点 ID 集合不变，3 个只删不增。',
    '- 8 HTTP、0 重试、80,866 tokens 与完整调用记录一致；费用和 reasoning 分拆 UNKNOWN。',
    '- 158 当前冻结文件和 358 原 downstream 冻结文件均一致；README 与最终报告链接全部存在。',
    '- 报告明确区分严格交付为 0 与答案正确率为 0，说明样本局限、Codex 诊断局限和原边可见，未宣称独立人工认证或全库错误率。', '',
    '此次仅写本终检报告及可重放审计脚本，未改任何原始 run 或已有报告，新增 API 调用 0。']
(REPORTS / 'FINAL_DELIVERY_CHECK.md').write_text('\n'.join(lines) + '\n')
print(json.dumps({'status': result['status'], 'checks': len(checks), 'failures': result['failed_checks']}, ensure_ascii=False))
