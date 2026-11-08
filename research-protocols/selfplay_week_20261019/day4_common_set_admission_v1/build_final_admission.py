"""Merge the finite, score-independent Day 4 foundation admission reviews."""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path


O = Path(__file__).resolve().parent
W = O.parent
REVIEW_FILES = [
    O / 'review/ROOT_REVIEW_INPUT_AUDIT.json',
    O / 'review/SAME_ROOT_SEMANTIC_ADJUDICATION_PART2.json',
    O / 'review/ROOT_REVIEW_2.json',
]


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(row) for row in Path(path).read_text().splitlines() if row.strip()]


def binding(path):
    p = Path(path)
    data = p.read_bytes()
    return {'path': str(p), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def digest(value):
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
    return hashlib.sha256(data).hexdigest()


def save(name, value):
    (O / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bindings_in(value):
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            yield value
        for child in value.values():
            yield from bindings_in(child)
    elif isinstance(value, list):
        for child in value:
            yield from bindings_in(child)


def compared_ids(record):
    if 'comparison' in record:
        return record['comparison']['new_candidate_ids_compared']
    if 'compared_new_candidate_ids' in record:
        return record['compared_new_candidate_ids']
    return (record['complete_same_type_new_candidate_ids_compared']
            + record['cross_type_candidate_ids_structurally_distinct'])


def rationale(record):
    if 'complete_task_rationale' in record:
        return record['complete_task_rationale']
    if 'comparison' in record:
        return record['comparison']['basis']
    return record['disposition_basis']


def merge():
    policy = read(O / 'REVIEW_POLICY.json')
    inventory = read(O / 'PACKET_INVENTORY.json')
    cross_path = O / 'review/CROSSROOT_REVIEW.json'
    cross = read(cross_path)
    inputs = {Path(b['path']).name: b for b in policy['frozen_inputs']}
    old = read(inputs['COMBINED_QUESTIONS.json']['path'])['questions']
    train = lines(inputs['TRAIN_NEW_ONLY.jsonl']['path'])
    lineage = lines(inputs['QUESTION_LINEAGE.jsonl']['path'])
    common_before = read(inputs['COMMON_OLD_BASE_SET.json']['path'])
    paired_ledger = read(inputs['PAIR88_ELIGIBILITY_AND_FAMILY_LEDGER.json']['path'])
    new = [x for x in paired_ledger['candidate_records'] if x['in88pair']]
    old_by = {x['candidate_id']: x for x in old}
    train_by = {x['candidate_id']: x for x in train}
    line_by = {x['candidate_id']: x for x in lineage}
    new_by = {x['candidate_id']: x for x in new}
    all_ids = set(old_by)
    require(len(old) == len(train) == len(lineage) == len(all_ids) == 363, 'Foundation counts/uniqueness')
    require(all_ids == set(train_by) == set(line_by), 'Identity join must use candidate_id')
    require(all_ids == set(common_before['foundation']['candidate_ids']), 'Frozen foundation identities differ')
    require(len(new) == len(new_by) == 176, 'Fixed paired new candidates changed')
    require(collections.Counter(x['arm'] for x in new) == {'A': 88, 'B': 88}, 'Paired arms must each have 88 candidates')
    new_roots = {x['root_id'] for x in new}
    overlap_ids = {x['candidate_id'] for x in old if x['root_id'] in new_roots}
    require(len(new_roots) == 20 and len(overlap_ids) == 102, 'Shared-root coverage changed')

    review_by = {}
    review_sources = {}
    review_payloads = []
    for path in REVIEW_FILES:
        payload = read(path)
        review_payloads.append(payload)
        for index, record in enumerate(payload['review_records']):
            cid = record['candidate_id']
            require(cid in overlap_ids and cid not in review_by, f'Duplicate/unexpected reviewed ID: {cid}')
            question = old_by[cid]
            require(record['root_id'] == question['root_id'], f'Review root mismatch: {cid}')
            require(record['question_type'] == question['question_type'], f'Review type mismatch: {cid}')
            require(record['decision'] in {'KEEP', 'EXCLUDE', 'HOLD'}, f'Invalid disposition: {cid}')
            group = record.get('dependence_group_id', record.get('local_dependency_links', {}).get('dependence_group_id')
                               if isinstance(record.get('local_dependency_links'), dict) else None)
            require(group == line_by[cid]['dependence_group_id'], f'Dependence group changed: {cid}')
            expected = {n['candidate_id'] for n in new if n['root_id'] == question['root_id']}
            compared = compared_ids(record)
            if 'comparison' in record:
                expected = {nid for nid in expected if new_by[nid]['question_type'] == question['question_type']}
            require(len(compared) == len(set(compared)) and set(compared) == expected,
                    f'Same-root review comparison set incomplete: {cid}')
            require(bool(rationale(record)), f'Missing complete-task rationale: {cid}')
            if 'pair_reviews' in record:
                require({p['new_candidate_id'] for p in record['pair_reviews']} == expected,
                        f'Pair review coverage mismatch: {cid}')
                require(all(p.get('whole_task_equivalent') is False for p in record['pair_reviews'])
                        if record['decision'] == 'KEEP' else True, f'KEEP contradicts duplicate evidence: {cid}')
            review_by[cid] = record
            review_sources[cid] = {'artifact': binding(path), 'json_pointer': f'/review_records/{index}'}
    require(set(review_by) == overlap_ids, 'Same-root review does not cover exactly all 102 shared-root foundation IDs')

    local_pairs = set()
    packets = []
    for item in inventory['packets']:
        packet = read(item['packet_binding']['path'])
        packets.append(packet)
        for b in packet['base_candidates']:
            for n in packet['new_candidates']:
                local_pairs.add((b['candidate_id'], n['candidate_id']))
    require(len(local_pairs) == 898, 'Root-local cartesian count changed')
    cross_by = collections.defaultdict(list)
    cross_keys = set()
    for index, pair in enumerate(cross['pairs']):
        cid, nid = pair['foundation_candidate_id'], pair['new_candidate_id']
        require(cid in old_by and nid in new_by, 'Unknown cross-root pair identity')
        require(old_by[cid]['root_id'] != new_by[nid]['root_id'], 'Same-root pair in cross-root review')
        require((cid, nid) not in cross_keys, 'Repeated cross-root pair')
        require(pair['decision'] in {'PASS', 'EXCLUDE', 'HOLD'}, 'Invalid cross-root disposition')
        cross_keys.add((cid, nid))
        confirmed = any(f.get('confirmed_same_necessary_assessed_fact') is True for f in pair['shared_facts'])
        cross_by[cid].append({
            'new_candidate_id': nid,
            'decision': pair['decision'],
            'semantic_class': pair['semantic_class'],
            'dependence_link_required': pair['dependence_link_required'],
            'confirmed_necessary_fact_overlap': confirmed,
            'evidence_json_pointer': f'/pairs/{index}',
        })
    require(len(cross_keys) == 840, 'Cross-root retrieved pair count changed')
    require(not cross['input_binding_issues'] and not cross['citation_binding_issues'], 'Cross-root review has unresolved input/citation issues')
    require(not cross['unresolved_semantic_review'], 'Cross-root review has unresolved semantic cases')

    # Verify every frozen file binding before using any admission decision.
    bound = {}
    payloads = [policy, inventory, cross, *packets, *review_payloads,
                read(O / 'review/SAME_ROOT_SEMANTIC_ADJUDICATION_PART1.json')]
    for payload in payloads:
        for b in bindings_in(payload):
            path = b['path']
            actual = bound.setdefault(path, binding(path))
            require(actual['sha256'] == b['sha256'], f'Input SHA-256 mismatch: {path}')
            if 'bytes' in b:
                require(actual['bytes'] == b['bytes'], f'Input length mismatch: {path}')
    for x in lineage:
        b = x['source_binding']
        actual = bound.setdefault(b['path'], binding(b['path']))
        require(actual['sha256'] == b['sha256'], f'Foundation source binding changed: {x["candidate_id"]}')

    records = []
    cross_binding = binding(cross_path)
    for cid in sorted(all_ids):
        question, row, lin = old_by[cid], train_by[cid], line_by[cid]
        require(row['root_id'] == lin['root_id'] == question['root_id'], f'Input join root mismatch: {cid}')
        require(row['question_type'] == lin['question_type'] == question['question_type'], f'Input join type mismatch: {cid}')
        require(row['dependence_group_id'] == lin['dependence_group_id'], f'Input join group mismatch: {cid}')
        local = review_by.get(cid)
        retrieved = cross_by[cid]
        local_decision = local['decision'] if local else 'KEEP'
        decisions = {local_decision, *(p['decision'] for p in retrieved)}
        decision = 'EXCLUDE' if 'EXCLUDE' in decisions else 'HOLD' if 'HOLD' in decisions else 'KEEP'
        if decision == 'EXCLUDE':
            require((local and (local.get('whole_task_duplicate_new_candidate_ids') or local.get('comparison', {}).get('full_task_equivalent_new_candidate_ids')))
                    or any(p['decision'] == 'EXCLUDE' for p in retrieved), 'EXCLUDE lacks complete-task duplicate evidence')
        same_root_nids = sorted(n['candidate_id'] for n in new if n['root_id'] == question['root_id'])
        record = {
            'candidate_id': cid,
            'root_id': question['root_id'],
            'question_type': question['question_type'],
            'dependence_group_id': lin['dependence_group_id'],
            'decision': decision,
            'eligibility_role': 'COMMON_ELIGIBLE_RESERVE_NOT_FINAL_TRAINING_SELECTION',
            'input_record_hashes': {'canonicalization': 'UTF-8 JSON, sorted keys, compact separators, ensure_ascii=False',
                                    'question_sha256': digest(question), 'training_record_sha256': digest(row),
                                    'lineage_record_sha256': digest(lin)},
            'source_binding': lin['source_binding'],
            'original_gold': question['gold'],
            'same_root_review': {
                'applicable': local is not None,
                'scope': 'COMPLETE_SAME_ROOT_TASK_REVIEW' if local else 'NO_NEW_CANDIDATE_SHARES_ROOT',
                'compared_new_candidate_ids': same_root_nids,
                'same_type_semantic_comparison_count': sum(new_by[nid]['question_type'] == question['question_type'] for nid in same_root_nids),
                'cross_type_structurally_distinct_count': sum(new_by[nid]['question_type'] != question['question_type'] for nid in same_root_nids),
                'decision': local_decision if local else 'NOT_APPLICABLE',
                'evidence_ref': review_sources.get(cid),
                'complete_task_rationale': rationale(local) if local else 'All 176 paired new candidates have different roots; cross-root finite retrieval review applies below.',
                'dependency_assessment': (local.get('dependence_links', local.get('local_dependency_links')) if local else []),
            },
            'cross_root_review': {
                'scope': 'FINITE_RETRIEVAL_SCREEN_AND_RETRIEVED_PAIR_ADJUDICATION_ONLY',
                'artifact_binding': cross_binding,
                'all_different_root_pairs_searched': 176 - len(same_root_nids),
                'retrieved_pairs_reviewed': len(retrieved),
                'unretrieved_pairs_not_individually_semantically_adjudicated': 176 - len(same_root_nids) - len(retrieved),
                'retrieved_comparisons': retrieved,
                'negative_result_is_exhaustive_semantic_proof': False,
                'interpretation': 'No confirmed whole-task duplicate in retrieved pairs. A negative retrieval result supports bounded reserve admission only, not semantic independence or exhaustive duplicate absence.',
            },
            'overall_scope': 'SHARED_ROOT_REVIEW_PLUS_BOUNDED_CROSS_ROOT_SCREEN' if local else 'BOUNDED_CROSS_ROOT_SCREEN_ONLY',
            'content_QA': 'Reuses frozen source/content attestations; this is not fresh full-option content QA.',
            'model_scores_used_for_admission': False,
            'final_training_selected_A': False,
            'final_training_selected_B': False,
        }
        records.append(record)
    counter = collections.Counter(r['decision'] for r in records)
    counts = {name: counter[name] for name in ('KEEP', 'EXCLUDE', 'HOLD')}
    eligible = sorted(r['candidate_id'] for r in records if r['decision'] == 'KEEP')
    excluded = sorted(r['candidate_id'] for r in records if r['decision'] == 'EXCLUDE')
    held = sorted(r['candidate_id'] for r in records if r['decision'] == 'HOLD')
    require(len(records) == 363, 'Final per-candidate ledger incomplete')
    save('ADMISSION_LEDGER.json', {
        'schema_version': 'day4_foundation_admission_ledger_v1',
        'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'status': 'FINITE_REVIEW_COMPLETE_FINAL_AUDIT_STATUS_IN_DECISION',
        'policy_binding': binding(O / 'REVIEW_POLICY.json'),
        'exclusion_rule': 'Only whole-task semantic equivalence excludes; partial overlap is retained with dependence.',
        'scope': {'foundation_candidates': 363, 'foundation_roots': 68, 'paired_new_candidates': 176,
                  'paired_new_per_arm': 88, 'shared_roots': 20, 'shared_root_foundation_candidates': 102,
                  'other_foundation_candidates': 261, 'same_root_pairs_reviewed': 898,
                  'different_root_pairs_searched': 62990, 'retrieved_cross_root_pairs_reviewed': 840,
                  'unretrieved_cross_root_pairs_not_individually_semantically_adjudicated': 62150},
        'input_bindings': [binding(O / 'PACKET_INVENTORY.json'), *policy['frozen_inputs'], cross_binding,
                           *[binding(p) for p in REVIEW_FILES],
                           binding(O / 'review/SAME_ROOT_SEMANTIC_ADJUDICATION_PART1.json')],
        'verified_unique_bound_files': len(bound),
        'summary': {'decisions': counts, 'eligible_common_reserve': len(eligible),
                    'excluded': len(excluded), 'hold': len(held),
                    'question_types': dict(collections.Counter(r['question_type'] for r in records)),
                    'crossroot_retrieval_hit_foundation_candidates': len({cid for cid in cross_by if cross_by[cid]}),
                    'confirmed_partial_crossroot_overlap_pairs_retained': sum(p['confirmed_necessary_fact_overlap'] for ps in cross_by.values() for p in ps)},
        'review_provenance': {'reviewer_kind': 'Codex semantic review plus deterministic input/coverage validation',
                              'human_review': False, 'technical_blinding': False,
                              'new_solver_calls': 0, 'new_model_inference': 0, 'training_runs': 0,
                              'scores_used_for_selection': False},
        'limitations': [
            'No claim of exhaustive cross-root semantic duplicate absence: 62,150 unretrieved pairs have no individual semantic adjudication.',
            'KEEP is common reserve eligibility under the disclosed bounded procedure, not factual independence.',
            'Root x type dependence groups are preserved; partial shared facts/events remain dependent.',
            'No fresh source-first content QA of all 363 questions, foundation-vs-foundation deduplication, or new-vs-new census.',
            'No actual mixed schedule, final selected training records, Solver inference, training, or causal-benefit evidence.',
        ],
        'records': records,
        'training_start_go': False,
    })
    save('COMMON_ELIGIBLE_BASE_SET.json', {
        'schema_version': 'day4_common_eligible_foundation_reserve_v1',
        'status': 'ELIGIBLE_RESERVE_NOT_SELECTED_TRAINING_POOL_FINAL_AUDIT_STATUS_IN_DECISION',
        'admission_ledger_binding': binding(O / 'ADMISSION_LEDGER.json'),
        'legacy_replay_candidate_ids': [], 'legacy_replay_selected_count': 0,
        'foundation_eligible_A': eligible, 'foundation_eligible_B': eligible,
        'same_ordered_common_set': True, 'common_set_identity_sha256': digest(eligible),
        'excluded_candidate_ids': excluded, 'hold_candidate_ids': held,
        'final_selected_foundation_A': [], 'final_selected_foundation_B': [],
        'final_training_record_counts': {'A': 0, 'B': 0},
        'mixture_frozen': False, 'q_final': None, 'training_start_go': False,
    })
    with (O / 'ADMISSION_TABLE.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['candidate_id', 'root_id', 'question_type', 'decision',
                                              'dependence_group_id', 'overall_scope', 'cross_root_retrieved_pairs'])
        writer.writeheader()
        for r in records:
            writer.writerow({**{key: r[key] for key in writer.fieldnames if key != 'cross_root_retrieved_pairs'},
                             'cross_root_retrieved_pairs': r['cross_root_review']['retrieved_pairs_reviewed']})
    save('BUILD_VALIDATION.json', {
        'schema_version': 'day4_admission_build_validation_v1', 'status': 'PASS',
        'candidate_id_join_used': True, 'per_candidate_coverage': 363,
        'shared_root_review_ids_unique_and_complete': 102,
        'same_root_cartesian_pairs': 898, 'cross_root_retrieved_pairs_unique': 840,
        'all_bound_input_files_verified': len(bound),
        'common_A_B_eligible_sets_and_order_identical': True,
        'partial_overlap_is_not_an_exclusion': True,
        'admission_ledger_binding': binding(O / 'ADMISSION_LEDGER.json'),
        'common_set_binding': binding(O / 'COMMON_ELIGIBLE_BASE_SET.json'),
        'errors': [], 'training_start_go': False,
    })
    print(json.dumps({'status': 'BUILT_PENDING_INDEPENDENT_FINAL_AUDIT', 'decisions': counts,
                      'common_eligible_reserve': len(eligible), 'verified_unique_files': len(bound)}))


def finalize():
    ledger = read(O / 'ADMISSION_LEDGER.json')
    common = read(O / 'COMMON_ELIGIBLE_BASE_SET.json')
    build = read(O / 'BUILD_VALIDATION.json')
    for b in ledger['input_bindings']:
        actual = binding(b['path'])
        require(actual['sha256'] == b['sha256'] and actual['bytes'] == b['bytes'],
                f'Admission input binding changed before seal: {b["path"]}')
    audits = [O / 'review/INDEPENDENT_FINAL_AUDIT.json', O / 'review/INDEPENDENT_COVERAGE_AUDIT.json']
    for path in audits:
        audit = read(path)
        require(audit['status'] == 'PASS', f'Independent audit did not pass: {path}')
        require(not audit.get('issues', []), f'Independent audit issues: {path}')
        require(audit['admission_ledger_binding']['sha256'] == binding(O / 'ADMISSION_LEDGER.json')['sha256'], 'Independent ledger audit stale')
        require(audit['common_set_binding']['sha256'] == binding(O / 'COMMON_ELIGIBLE_BASE_SET.json')['sha256'], 'Independent common-set audit stale')
    for key, filename in [('admission_ledger_binding', 'ADMISSION_LEDGER.json'), ('common_set_binding', 'COMMON_ELIGIBLE_BASE_SET.json')]:
        require(build[key]['sha256'] == binding(O / filename)['sha256'], 'Build validation stale')
    require(common['foundation_eligible_A'] == common['foundation_eligible_B'], 'A/B common eligible reserves differ')
    require(not common['final_selected_foundation_A'] and not common['final_selected_foundation_B'], 'Unexpected training selection')
    decision = {
        'schema_version': 'day4_sets_decision_v1',
        'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'status': 'D4_SETS_PASS_BOUNDED_COMMON_ELIGIBLE_RESERVE',
        'D4_SETS': 'PASS_WITH_DISCLOSED_FINITE_CROSSROOT_RETRIEVAL_BOUNDARY',
        'exclusion_rule': 'ONLY_WHOLE_TASK_SEMANTIC_EQUIVALENCE',
        'counts': ledger['summary'],
        'common_eligible_A': len(common['foundation_eligible_A']),
        'common_eligible_B': len(common['foundation_eligible_B']),
        'identical_common_eligible_sets': True,
        'common_set_identity_sha256': common['common_set_identity_sha256'],
        'legacy_replay_selected_count': 0,
        'final_training_record_counts': {'A': 0, 'B': 0},
        'D4_SCHEDULE': 'NOT_FROZEN',
        'q_final': None,
        'training_start_go': False,
        'new_solver_calls': 0, 'new_model_inference': 0, 'new_training_runs': 0,
        'day3_solver_calls_cumulative_unchanged': 3417,
        'independent_audits_pass': True,
        'input_bindings': [binding(O / name) for name in [
            'REVIEW_POLICY.json', 'ADMISSION_LEDGER.json', 'COMMON_ELIGIBLE_BASE_SET.json',
            'BUILD_VALIDATION.json']] + [binding(path) for path in audits],
        'limitations': ledger['limitations'],
        'next_work': 'D4-SCHEDULE: select and freeze actual symmetric A/B pools, supervised-token mixture, displays, equal steps, stopping/checkpoints and runtime; do not start training before that separate gate.',
    }
    save('DECISION.json', decision)
    save('SUMMARY.json', {k: v for k, v in decision.items() if k not in {'input_bindings', 'limitations'}})
    counts = ledger['summary']['decisions']
    report = f'''# Day 4 公共基础集合准入

按用户确认规则执行：只有整题任务等价才排除。多选比较完整命题集（包含错误选项）；排序比较完整事件集合及排序关系。局部事实、共享 turn、同 root 或少数排序边重叠均保留，并记录依赖。

## 本次结果

- 固定旧基础题 363 道，68 个 root；对照固定 A/B 新题 176 道，即每臂 88 道。
- 旧题逐题账本：KEEP {counts.get('KEEP', 0)}、EXCLUDE {counts.get('EXCLUDE', 0)}、HOLD {counts.get('HOLD', 0)}。
- 20 个共享 root 内，102 道旧题已覆盖全部 898 个 root 内候选对；其余 261 道旧题按跨 root 有限检索结果处理。
- 跨 root 对 62,990 个候选对执行有限检索，命中 840 对并裁决；未命中的 62,150 对未逐对做语义审核。未发现可确认的整题重复，不等于穷尽证明没有重复。
- 9 对明确的跨 root 局部必要属性重叠继续保留；另有保守的内容关联链接。所有原 dependence_group_id 保留，KEEP 不代表事实独立。
- A/B 共用同一份 {len(common['foundation_eligible_A'])} 道合格基础储备，ID、顺序和集合哈希一致。它还不是实际选入训练的基础题数量。
- 两份独立审计均通过；本次 Solver 0 次、模型调用 0 次、训练 0 次，第三天累计 Solver 仍为 3417 次。

## 查看文件

- `ADMISSION_LEDGER.json`：363 道逐题裁决、输入记录哈希、来源绑定、审核依据及依赖引用。
- `ADMISSION_TABLE.csv`：便于查阅的逐题简表。
- `COMMON_ELIGIBLE_BASE_SET.json`：两臂完全一致的合格基础储备；实际已选训练记录仍各 0。
- `DECISION.json`、`SUMMARY.json`：最终机器裁决与摘要。
- `review/INDEPENDENT_FINAL_AUDIT.json`、`review/INDEPENDENT_COVERAGE_AUDIT.json`：独立核验。
- `FINAL_MANIFEST.json`：本包输入、脚本、审核及结果文件哈希。

## 当前边界与下一步

D4-SETS 在披露的有限检索范围内通过。此次没有重新审核旧 legacy113，也没有重新认证所有旧题的内容质量；原始内容审核沿用冻结来源。全部保留不代表题目彼此独立，更不证明反馈、memory 或训练收益。

下一步执行 D4-SCHEDULE：冻结实际 A/B 训练池、新题/基础题的监督 token 比例、选项展示、相同步数、停止规则、checkpoint 与 runtime。当前 q_final 尚未冻结，训练 GO 仍为 false。历史回归及新增题族评估边界保持第三天裁决。
'''
    (O / 'REPORT_ZH.md').write_text(report)
    manifest_names = ['REVIEW_POLICY.json', 'PACKET_INVENTORY.json', 'prepare_admission.py',
                      'build_final_admission.py', 'refresh_navigation.py', 'ADMISSION_LEDGER.json', 'ADMISSION_TABLE.csv',
                      'COMMON_ELIGIBLE_BASE_SET.json', 'BUILD_VALIDATION.json', 'DECISION.json',
                      'SUMMARY.json', 'REPORT_ZH.md']
    manifest_paths = [O / name for name in manifest_names] + sorted((O / 'review').glob('*.json'))
    save('FINAL_MANIFEST.json', {
        'schema_version': 'day4_sets_final_manifest_v1',
        'status': 'SEALED_BOUNDED_D4_SETS',
        'input_bindings': ledger['input_bindings'],
        'artifacts': [binding(path) for path in manifest_paths],
        'self_hash_excluded': True, 'training_start_go': False,
    })
    print(json.dumps({'status': decision['status'], 'common_eligible_per_arm': len(common['foundation_eligible_A']),
                      'training_start_go': False, 'manifest_sha256': binding(O / 'FINAL_MANIFEST.json')['sha256']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--finalize', action='store_true')
    args = parser.parse_args()
    finalize() if args.finalize else merge()
