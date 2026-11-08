#!/usr/bin/env python3
"""Audit frozen expansion + QA counts without opening any private question pool.

Only report files are created. The QA engine and all input artifacts stay unchanged.
Refuses to inspect current QA review records until BATCH_REVIEW is FROZEN.
Gold distributions for usable questions come from saved source judgments whose
unchanged labels have already passed the engine's private cross-check.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


class AccountingError(RuntimeError):
    pass


def need(ok, message):
    if not ok:
        raise AccountingError(message)


class Reader:
    def __init__(self):
        self.files = {}

    def read(self, path):
        path = Path(path).resolve()
        need(path.name not in {
            'POOL.json', 'PARENT_POOL.json', 'GOLD_PLAN.json',
            'FAMILY_SIDECAR.json', 'ASSEMBLY_RECEIPTS.json',
            'EVALUATION_POOL.json', 'ASSETS.json', 'WRITER_PLAN.json',
            'RESPONSE.json', 'HTTP_RESPONSE.json',
        }, f'private_artifact_read_forbidden:{path}')
        data = path.read_bytes()
        binding = {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest()}
        self.files[str(path)] = binding
        return json.loads(data)

    def bound(self, binding):
        value = self.read(binding['path'])
        need(self.files[str(Path(binding['path']).resolve())]['sha256'] == binding['sha256'],
             f'binding_hash_mismatch:{binding["path"]}')
        return value


def indexed(rows, label):
    result = {r['candidate_id']: r for r in rows}
    need(len(result) == len(rows), f'duplicate_candidate_ids:{label}')
    return result


def source_refs(source):
    return [turn['turn_ref'] for turn in source['turns']]


def audit(delivery):
    reader = Reader()
    shard = delivery / 'shards/shard_01'
    # This guard runs before reading public QA, review records or candidate counts.
    frozen = reader.read(shard / 'BATCH_REVIEW.COMPLETE.json')
    need(frozen.get('status') == 'FROZEN', 'batch_not_frozen')
    need(frozen.get('before_any_solver') is True, 'freeze_not_before_solver')
    need(frozen.get('solver_scores_used') is False, 'solver_scores_used')
    need(frozen.get('training_allowed') is False, 'freeze_training_allowed')
    plan = reader.read(delivery / 'PILOT_PLAN.json')
    config = reader.read(shard / 'CONFIG.json')
    need(plan.get('training_allowed') is False and config.get('training_allowed') is False,
         'training_not_disabled')
    need(plan.get('solver_execution_in_this_batch') is False, 'solver_not_disabled')
    expansion = reader.bound(plan['upstream_expansion_run']['REPORT.json'])
    need(expansion.get('solver_calls') == 0, 'expansion_solver_calls')
    need(expansion.get('training_started') is False, 'expansion_training_started')
    routes = {r['root_id']: r for r in expansion['routes']}
    need(set(routes) == set(plan['candidate_roots']), 'expansion_root_coverage')
    materialization = reader.read(delivery / 'evidence/EXPANSION_MATERIALIZATION_COMPLETE.json')
    admission = reader.read(shard / 'SOURCE_ADMISSION_SUMMARY.json')
    ready = admission['ready_roots']
    need(ready == [r for r in plan['candidate_roots'] if r in ready], 'ready_order')
    need(set(ready) == set(config['source_consistency']['selected_roots']), 'config_ready_mismatch')
    sources = {}
    roots = []
    for root_id in plan['candidate_roots']:
        short = root_id.rsplit('-', 1)[-1]
        provenance_path = delivery / f'evidence/expansion_provenance/root{short}_PROVENANCE.json'
        if not provenance_path.exists():
            provenance_path = delivery / f'evidence/expansion_failures/root{short}_FAILURE.json'
        prov = reader.read(provenance_path)
        baseline = reader.bound(prov['input_source'])
        route = routes[root_id]
        baseline_count = len(baseline['turns'])
        need(baseline_count == route['input_turns'] == prov['historical_input_turns'],
             f'baseline_turn_count:{root_id}')
        new_count = prov['new_turn_count']
        need(new_count == route['added_turns'], f'new_turn_count:{root_id}')
        review = None
        if prov.get('source'):
            source = reader.bound(prov['source'])
            accepted = reader.bound(prov['accepted_output'])
            need(source == accepted, f'materialized_source_changed:{root_id}')
            need(source['turns'][:baseline_count] == baseline['turns'], f'baseline_prefix_changed:{root_id}')
            need(len(source['turns']) == baseline_count + new_count == prov['output_turns'],
                 f'cumulative_turn_count:{root_id}')
            need(source_refs(source)[baseline_count:] == prov['new_turn_refs'], f'suffix_ref_mismatch:{root_id}')
            sources[root_id] = source
            review = reader.read(delivery / f'inputs/root{short}_SOURCE_REVIEW.json')
            need(review['root_id'] == root_id, f'source_review_root:{root_id}')
            need(review['source'] == prov['source'], f'source_review_binding:{root_id}')
            need(review.get('full_source_read') is True, f'not_full_source_review:{root_id}')
            need(set(review['turn_refs_read']) == set(source_refs(source)), f'source_review_coverage:{root_id}')
            decision = review['decision']
        else:
            need(new_count == 0, f'no_source_with_new_turns:{root_id}')
            decision = 'NO_NEW_SOURCE'
        findings = review.get('findings', []) if review else []
        need((decision == 'READY') == (root_id in ready), f'ready_source_review_mismatch:{root_id}')
        roots.append({
            'root_id': root_id, 'baseline_turns': baseline_count,
            'model_accepted_new_turns': new_count, 'cumulative_turns': baseline_count + new_count,
            'model_accepted_scenes': route['accepted_scenes'],
            'source_decision': decision,
            'warning_findings': sum(f['severity'] == 'WARNING' for f in findings),
            'blocking_findings': sum(f['severity'] == 'BLOCKING' for f in findings),
            'findings': [{'finding_id': f['finding_id'], 'severity': f['severity'],
                          'kind': f.get('kind'), 'reason': f.get('reason')} for f in findings],
            'qa_admitted_new_turns': new_count if decision == 'READY' else 0,
            'expansion_failures': route.get('failures', []),
        })
    need(sum(r['model_accepted_new_turns'] > 0 for r in roots) == materialization['accepted_roots'],
         'materialized_root_partition')

    ms_public = reader.read(shard / 'multiselect/PUBLIC_QUESTIONS.json')
    ms_review = reader.read(shard / 'multiselect/FINAL_CONTENT_REVIEW.json')
    ms_receipt = reader.read(shard / 'multiselect/FINAL_CONTENT_REVIEW.COMPLETE.json')
    ms_generation = reader.read(shard / 'multiselect/GENERATION.COMPLETE.json')
    need(ms_receipt.get('status') == 'FROZEN', 'multiselect_not_frozen')
    need(ms_receipt.get('solver_scores_used') is False, 'multiselect_solver_scores_used')
    need(reader.bound(ms_receipt['review']) == ms_review, 'multiselect_review_receipt_binding')
    ms = indexed(ms_public['questions'], 'multiselect_public')
    ms_content = indexed(ms_review['questions'], 'multiselect_content')
    need(set(ms) == set(ms_content), 'multiselect_content_coverage')
    judgments = []
    for group in config['multiselect_config']['final_review_groups']:
        judgment_path = shard / f'multiselect/final_review/group_{group}_SOURCE_JUDGMENTS.json'
        saved = reader.read(judgment_path)
        need(saved['public_questions'] == reader.files[str((shard / 'multiselect/PUBLIC_QUESTIONS.json').resolve())],
             f'public_questions_changed:{group}')
        need(saved.get('child_gold_read_before_save') is False, f'source_first_claim:{group}')
        need(saved.get('current_pool_read_before_save') is False, f'pool_read_before_save:{group}')
        need(saved.get('solver_scores_used') is False, f'group_solver_scores:{group}')
        check = reader.read(shard / f'multiselect/final_review/group_{group}_AUDIT.json')
        binding = reader.files[str(judgment_path.resolve())]
        need(check['source_judgments_sha256'] == binding['sha256'], f'judgments_changed:{group}')
        need(check.get('question_or_gold_edits') == [], f'question_edits:{group}')
        judgments.extend(saved['questions'])
    ms_judgments = indexed(judgments, 'multiselect_source_judgments')
    need(set(ms_judgments) == set(ms), 'multiselect_source_judgment_coverage')
    parent_ids = {s['parent_candidate_id'] for s in config['multiselect_config']['slots']}
    ms_rows = []
    for candidate_id, question in ms.items():
        source_judgment = ms_judgments[candidate_id]
        need(question['root_id'] in ready, f'question_source_not_ready:{candidate_id}')
        source_binding = reader.files[str(Path(question['source_path']).resolve())]
        need(source_binding['sha256'] == question['source_sha256'], f'question_source_changed:{candidate_id}')
        need(source_judgment['root_id'] == question['root_id'], f'judgment_root:{candidate_id}')
        need(source_judgment['source_sha256'] == question['source_sha256'], f'judgment_source:{candidate_id}')
        labels = question['options']
        if isinstance(labels, list):
            labels = {o['label'] for o in labels}
        else:
            labels = set(labels)
        need(labels == set(source_judgment['options']), f'option_judgment_coverage:{candidate_id}')
        supported = sum(o['relation'] == 'supports' for o in source_judgment['options'].values())
        status = ms_content[candidate_id]['core_status']
        need(status in ['CORE_USABLE', 'KNOWN_SUBSTANTIVE_ISSUE'], f'unknown_content_status:{status}')
        control = candidate_id in parent_ids
        if status == 'CORE_USABLE':
            need(supported == 6 if control else supported in [2, 3, 4, 5], f'usable_gold_count:{candidate_id}')
        ms_rows.append({'candidate_id': candidate_id, 'root_id': question['root_id'],
                        'role': 'control6' if control else 'new_child',
                        'source_supported_count': supported, 'content_status': status,
                        'reason': ms_content[candidate_id].get('reason')})
    need(ms_receipt['candidate_count'] == len(ms_rows), 'multiselect_receipt_total')
    need(ms_receipt['usable_count'] == sum(r['content_status'] == 'CORE_USABLE' for r in ms_rows),
         'multiselect_receipt_usable')

    ordering_public = reader.read(shard / 'ordering/PUBLIC_QUESTIONS.json')
    ordering_generation = reader.read(shard / 'ordering/GENERATION.COMPLETE.json')
    ordering_review = reader.read(shard / 'ordering/CONTENT_REVIEW.json')
    ordering = indexed(ordering_public['questions'], 'ordering_public')
    ordering_content = indexed(ordering_review['questions'], 'ordering_content')
    need(set(ordering) == set(ordering_content), 'ordering_content_coverage')
    ordering_rows = []
    for candidate_id, question in ordering.items():
        record = ordering_content[candidate_id]
        need(record['status'] in ['USABLE', 'EXCLUDED'], f'ordering_content_status:{candidate_id}')
        ordering_rows.append({'candidate_id': candidate_id, 'root_id': question['root_id'],
                              'content_status': record['status'], 'reasons': record.get('reasons', [])})
    rejections = []
    for slot in config['ordering_slots']:
        path = shard / f'ordering/{slot["slot_id"]}/REJECTIONS.json'
        for reject in reader.read(path):
            rejections.append({'root_id': slot['root_id'], **reject})
    reject_ids = indexed(rejections, 'ordering_mechanical_rejections')
    need(not (set(ordering) & set(reject_ids)), 'ordering_rejected_and_public_overlap')
    need(ordering_generation['candidate_count'] == len(ordering_rows), 'ordering_generation_count')
    need(ordering_review['accepted_count'] == sum(r['content_status'] == 'USABLE' for r in ordering_rows),
         'ordering_accepted_count')
    need(ordering_review['rejected_count'] == sum(r['content_status'] == 'EXCLUDED' for r in ordering_rows),
         'ordering_rejected_count')
    for root in roots:
        root_id = root['root_id']
        multiselect_rows = [r for r in ms_rows if r['root_id'] == root_id]
        orders = [r for r in ordering_rows if r['root_id'] == root_id]
        rejects = [r for r in rejections if r['root_id'] == root_id]
        slots = [s for s in config['multiselect_config']['slots'] if s['root_id'] == root_id]
        planned_ms = sum(len(s['target_counts']) + 1 for s in slots)
        planned_order = sum(s['candidate_count'] for s in config['ordering_slots'] if s['root_id'] == root_id)
        need(len(multiselect_rows) <= planned_ms, f'multiselect_over_budget:{root_id}')
        need(len(orders) + len(rejects) <= planned_order, f'ordering_over_budget:{root_id}')
        usable_ms = [r for r in multiselect_rows if r['content_status'] == 'CORE_USABLE']
        root['qa'] = {
            'planned_candidate_slots': planned_ms + planned_order,
            'multiselect_public_candidates': len(multiselect_rows),
            'multiselect_uncreated_slots': planned_ms - len(multiselect_rows),
            'multiselect_content_pass': len(usable_ms),
            'multiselect_content_fail': len(multiselect_rows) - len(usable_ms),
            'usable_new_child_gold_distribution': {str(k): sum(r['role'] == 'new_child' and r['source_supported_count'] == k for r in usable_ms) for k in [2, 3, 4, 5]},
            'usable_control6': sum(r['role'] == 'control6' for r in usable_ms),
            'ordering_public_candidates': len(orders),
            'ordering_mechanical_rejections': len(rejects),
            'ordering_unreturned_slots': planned_order - len(orders) - len(rejects),
            'ordering_content_pass': sum(r['content_status'] == 'USABLE' for r in orders),
            'ordering_content_fail': sum(r['content_status'] == 'EXCLUDED' for r in orders),
        }
    totals = Counter()
    gold = Counter()
    for root in roots:
        for key, value in root['qa'].items():
            if isinstance(value, int):
                totals[key] += value
        gold.update(root['qa']['usable_new_child_gold_distribution'])
    totals = dict(totals)
    totals['usable_new_child_gold_distribution'] = dict(gold)
    totals['public_candidates'] = len(ms_rows) + len(ordering_rows)
    totals['content_pass_including_controls'] = totals['multiselect_content_pass'] + totals['ordering_content_pass']
    totals['content_fail'] = totals['multiselect_content_fail'] + totals['ordering_content_fail']
    totals['usable_new_questions_excluding_controls'] = totals['content_pass_including_controls'] - totals['usable_control6']
    need(totals['public_candidates'] == totals['content_pass_including_controls'] + totals['content_fail'],
         'public_candidate_partition')
    need(totals['planned_candidate_slots'] == totals['public_candidates'] + totals['multiselect_uncreated_slots']
         + totals['ordering_mechanical_rejections'] + totals['ordering_unreturned_slots'], 'planned_candidate_partition')
    need(frozen['by_type'] == {'multi_select': totals['multiselect_content_pass'],
                               'ordering': totals['ordering_content_pass']}, 'frozen_counts_mismatch')
    # Check saved freeze bindings for every accounting input that the freeze tracks.
    frozen_bindings = {str(Path(b['path']).resolve()): b for b in frozen.get('files', [])}
    checked_frozen_bindings = 0
    for path, binding in reader.files.items():
        if path in frozen_bindings:
            need(binding['sha256'] == frozen_bindings[path]['sha256'], f'frozen_input_changed:{path}')
            checked_frozen_bindings += 1
    need(checked_frozen_bindings > 0, 'freeze_has_no_matching_accounting_input_bindings')
    run_artifacts = []
    for base in [delivery, shard]:
        for pattern in ['solver_report', 'solver_runs', 'SOLVER*.json', 'TRAINING.STARTED*', 'TRAINING.COMPLETE*']:
            run_artifacts.extend(str(p) for p in base.glob(pattern))
    need(not run_artifacts, 'unexpected_solver_or_training_artifacts:' + repr(run_artifacts))
    return {
        'version': 'v103_frozen_expansion_qa_accounting_v1', 'status': 'PASS',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'delivery': str(delivery), 'training_allowed': False, 'training_started': False,
        'solver_execution': 'NOT_RUN_IN_THIS_BATCH_AS_RECORDED_BY_RUN_AND_FREEZE_RECEIPTS',
        'private_question_pools_read_by_this_audit': False,
        'root_count': len(roots), 'source_ready_roots': ready,
        'model_accepted_new_turns': sum(r['model_accepted_new_turns'] for r in roots),
        'qa_admitted_new_turns': sum(r['qa_admitted_new_turns'] for r in roots),
        'roots': roots, 'qa_totals': totals,
        'multiselect_questions': ms_rows, 'ordering_questions': ordering_rows,
        'ordering_mechanical_rejections': rejections,
        'http_attempts': {'main_expansion': expansion['http_attempts'],
                          'qa_multiselect': ms_generation['http_attempts'],
                          'qa_ordering': ordering_generation['http_attempts']},
        'limitations': [
            'Source READY certifies reviewed consistency only; warnings remain disclosed.',
            'Question content pass and frozen counts do not establish solver difficulty, accuracy or training value.',
            'Controls are reported separately and excluded from new-question totals.',
            'Same-family children and overlapping ordering items are related samples, not independent evidence.',
            'Usable Gold counts derive from saved public source relations after the engine cross-check; failed-question support counts are not claimed as correct private Gold.',
            'No private Pool, Gold plan, family sidecar, raw QA writer response or evaluation pool was opened by this audit.',
        ],
        'checked_input_files': sorted(reader.files.values(), key=lambda b: b['path']),
        'checks': {'source_hashes_and_prefixes': 'PASS', 'source_review_coverage': 'PASS',
                   'public_question_review_coverage': 'PASS', 'candidate_partitions': 'PASS',
                   'frozen_usable_counts': 'PASS', 'matching_freeze_input_hashes': checked_frozen_bindings,
                   'no_local_solver_training_artifacts': 'PASS'},
    }


def markdown(report):
    q = report['qa_totals']
    lines = ['# v103 扩充与出题冻结统计', '',
             '来源、候选、内容通过及控制题分别计数；本批未运行 solver，未训练。', '',
             '| Root | 原轮数 | 新增轮数 | 累计轮数 | 来源结论 | WARNING | 多选通过/失败 | 排序通过/内容失败/机械失败 |',
             '|---|---:|---:|---:|---|---:|---:|---:|']
    for r in report['roots']:
        a = r['qa']
        lines.append(f'| {r["root_id"].rsplit("-", 1)[-1]} | {r["baseline_turns"]} | {r["model_accepted_new_turns"]} | {r["cumulative_turns"]} | {r["source_decision"]} | {r["warning_findings"]} | {a["multiselect_content_pass"]}/{a["multiselect_content_fail"]} | {a["ordering_content_pass"]}/{a["ordering_content_fail"]}/{a["ordering_mechanical_rejections"]} |')
    lines += ['', f'模型接受新增 {report["model_accepted_new_turns"]} 轮；通过来源审核并用于出题的新增 {report["qa_admitted_new_turns"]} 轮。', '',
              f'公开候选 {q["public_candidates"]} 题，内容通过 {q["content_pass_including_controls"]} 题、失败 {q["content_fail"]} 题。', '',
              f'可用新题 {q["usable_new_questions_excluding_controls"]} 题，另有 Gold6 控制题 {q["usable_control6"]} 题。', '',
              '可用多选 Gold2/3/4/5：' + '/'.join(str(q['usable_new_child_gold_distribution'].get(str(k), 0)) for k in [2, 3, 4, 5]) + '。', '',
              f'排序机械失败 {q["ordering_mechanical_rejections"]} 题；多选未形成候选槽位 {q["multiselect_uncreated_slots"]}，排序未返回槽位 {q["ordering_unreturned_slots"]}。', '',
              '内容通过不表示难度已经验证；同一 family 的子题及重叠排序题不能作为独立样本计量。详细失败原因、WARNING 和文件哈希见配套 JSON。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delivery', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-json', type=Path)
    parser.add_argument('--output-md', type=Path)
    args = parser.parse_args()
    delivery = args.delivery.resolve()
    report = audit(delivery)
    outputs = [(args.output_json or delivery / 'evidence/FINAL_BATCH_ACCOUNTING.json',
                json.dumps(report, ensure_ascii=False, indent=2) + '\n'),
               (args.output_md or delivery / 'evidence/FINAL_BATCH_ACCOUNTING.md', markdown(report))]
    for path, _ in outputs:
        need(not path.exists(), f'output_already_exists:{path}')
    for path, content in outputs:
        with path.open('x') as handle:
            handle.write(content)
    print(json.dumps({'status': report['status'], 'qa_totals': report['qa_totals'],
                      'outputs': [str(path) for path, _ in outputs]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
