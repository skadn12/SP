"""Consolidate already sealed diagnosis evidence; no inference or Gold parsing."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

HERE = Path(__file__).resolve().parent
WEEK = HERE.parents[1]
CHECKS = []


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def check(name, value):
    if not value:
        raise AssertionError(name)
    CHECKS.append(name)


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def main():
    check('only_own_new_subdirectory', HERE == WEEK / 'day2_completion_v1/diagnosis')
    outputs = ['PROVENANCE.json', 'DIAGNOSIS_CLOSEOUT.json', 'REPORT_ZH.md', 'VALIDATION.json']
    check('exclusive_outputs', not any((HERE / name).exists() for name in outputs))
    seals = {
        'day1': '8918702665d1965e3dc54015239a871213df24c7d9d1db983d66f0bb013c30e8',
        'day2': '4411bdaedfbedc4c0b478f59ceda9e9a70b65f15c13e2a915b76c555a5dfafa7',
        'day2_order_probe_v1': '2b8a0d0565e094860f91d94f67e99a0bf8a3d566f47ecf630df659d41e24528b',
        'day2_frozen_evidence_probe_v1': 'b70f4b9fb655aebc3d08c14886dd33dff85552cb1e2aead475e9ebc2473c500d',
        'day2_frozen_evidence_probe_v2': '43a3759b35e98f2a208121efd9ade47f44c3e344d7ecb0db3c1eafe6cab461ef',
    }
    sealed_files = 0
    seal_rows = []
    for name, pinned in seals.items():
        directory = WEEK / name
        path = directory / 'COMPLETE_MANIFEST.json'
        check(name + ':seal_sha256', sha(path) == pinned)
        seal = read(path)
        check(name + ':file_count', len(seal['files']) == seal['file_count'])
        check(name + ':byte_count', sum(x['bytes'] for x in seal['files']) == seal['total_bytes'])
        expected = set()
        for row in seal['files']:
            rel = Path(row['path'])
            check(name + ':safe_path:' + str(rel), not rel.is_absolute() and '..' not in rel.parts)
            p = directory / rel
            check(name + ':unchanged:' + str(rel), p.is_file() and p.stat().st_size == row['bytes'] and sha(p) == row['sha256'])
            expected.add(p.resolve())
        actual = {p.resolve() for p in directory.rglob('*') if p.is_file() and p.name != 'COMPLETE_MANIFEST.json' and '__pycache__' not in p.parts and p.suffix != '.pyc'}
        check(name + ':exact_file_set', actual == expected)
        sealed_files += len(expected)
        seal_rows.append({**bind(path), 'status': seal['status'], 'files_verified': len(expected)})

    relatives = [
        'PLAN_ZH.md', 'day2/FINAL_STATUS.json', 'day2/CONTENT_ADMISSION.json',
        'day2/analysis/results_02/OBSERVATIONS.json', 'day2/analysis/results_02/REPORT_ZH.md',
        'day2/solver/direct/SCORES.json', 'day2/solver/direct/RAW_RESPONSE_FREEZE.json',
        'day2/solver/evidence_v2/SCORES.json', 'day2/solver/evidence_v2/RAW_RESPONSE_FREEZE.json',
        'day2/PROBE_SELECTION_POLICY.json', 'day2/probes/evidence_hint/SCORES.json',
        'day2/probes/evidence_hint/RAW_RESPONSE_FREEZE.json',
        'day2/diagnosis_exports/observed_six_v002/HINT_CONTRAST_CONTEXT.json',
        'day2/diagnosis_exports/observed_six_v002/MEMORY_B_S0_OBSERVED_v002.json',
        'day2/analysis/order_trace_consistency/ORDER_TRACE_CONSISTENCY.json',
        'day2/analysis/order_trace_consistency/SOURCE_REVIEW.json',
        'day2_order_probe_v1/PROTOCOL.json', 'day2_order_probe_v1/POLICY_FREEZE.json',
        'day2_order_probe_v1/run/SCORES.json', 'day2_order_probe_v1/run/RAW_RESPONSE_FREEZE.json',
        'day2_order_probe_v1/audit/RESULTS_AUDIT.json',
        'day2_frozen_evidence_probe_v1/COMPLETE_MANIFEST.json',
        'day2_frozen_evidence_probe_v2/PROTOCOL.json', 'day2_frozen_evidence_probe_v2/POLICY_FREEZE.json',
        'day2_frozen_evidence_probe_v2/run/SCORES.json', 'day2_frozen_evidence_probe_v2/run/RAW_RESPONSE_FREEZE.json',
        'day2_frozen_evidence_probe_v2/run/FROZEN_EVIDENCE.json',
        'day2_frozen_evidence_probe_v2/INDEPENDENT_RESULTS_AUDIT.json',
        'day2_frozen_evidence_probe_v2/SEMANTIC_REVIEW.json', 'day2_frozen_evidence_probe_v2/SEMANTIC_REVIEW_ZH.md',
    ]
    provenance = {'schema_version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                  'scope': 'Read-only consolidation of existing sealed diagnosis evidence; no new model call, private Gold JSON read, new semantic review or change to old memory.',
                  'sealed_snapshots': seal_rows, 'sealed_files_verified': sealed_files,
                  'inputs': [{**bind(WEEK / rel), 'relative_to_week': rel} for rel in relatives],
                  'builder': bind(__file__), 'new_solver_calls': 0, 'private_gold_json_read': False,
                  'sealed_directories_modified': False}
    obs = read(WEEK / 'day2/analysis/results_02/OBSERVATIONS.json')
    final = read(WEEK / 'day2/FINAL_STATUS.json')
    baseline = read(WEEK / 'day2/solver/direct/SCORES.json')
    hint = read(WEEK / 'day2/diagnosis_exports/observed_six_v002/HINT_CONTRAST_CONTEXT.json')
    memory = read(WEEK / 'day2/diagnosis_exports/observed_six_v002/MEMORY_B_S0_OBSERVED_v002.json')
    trace = read(WEEK / 'day2/analysis/order_trace_consistency/ORDER_TRACE_CONSISTENCY.json')
    order = read(WEEK / 'day2_order_probe_v1/run/SCORES.json')
    frozen = read(WEEK / 'day2_frozen_evidence_probe_v2/run/SCORES.json')
    semantics = read(WEEK / 'day2_frozen_evidence_probe_v2/SEMANTIC_REVIEW.json')
    order_audit = read(WEEK / 'day2_order_probe_v1/audit/RESULTS_AUDIT.json')
    frozen_audit = read(WEEK / 'day2_frozen_evidence_probe_v2/INDEPENDENT_RESULTS_AUDIT.json')
    check('both_controlled_results_independent_PASS', order_audit['status'] == frozen_audit['status'] == 'PASS' and order_audit['issues_count'] == frozen_audit['issues_count'] == 0)
    check('baseline_content_admitted', read(WEEK / 'day2/CONTENT_ADMISSION.json')['status'] == 'PASS')
    check('old_weakness_causes_empty', len(memory['causes']) == 0 and len(memory['probes']) == 0)
    check('first_gate_not_met', order['primary_endpoint']['met'] is False)
    check('frozen_effect_descriptive_only', frozen['primary_endpoint']['effect_only'] is True and frozen['primary_endpoint']['working_support_gate'] is None)
    check('no_training_cause_upgrade', all(not r['training_started'] and not r['cause_auto_published'] and not r['weakness_memory_upgraded'] for r in [order, frozen]))
    mc = [r for r in obs['runs']['direct']['requests'] if r['question_type'] == 'multi_select']
    mc_questions = [q for q in obs['runs']['direct']['questions'] if q['question_type'] == 'multi_select']
    ordering_questions = [q for q in obs['runs']['direct']['questions'] if q['question_type'] == 'ordering']
    pattern = {
        'multi_select': {'independent_questions': len(mc_questions), 'independent_roots': len({r['root_id'] for r in mc}),
                        'calls': len(mc), 'false_positive_calls': sum(r['fp'] > 0 for r in mc),
                        'false_positive_roots': sorted({r['root_id'] for r in mc if r['fp'] > 0}),
                        'false_negative_calls': sum(r['fn'] > 0 for r in mc),
                        'false_negative_roots': sorted({r['root_id'] for r in mc if r['fn'] > 0}),
                        'overselected_calls': sum(r['overselected'] for r in mc),
                        'underselected_calls': sum(r['selected_count'] < r['gold_count'] for r in mc),
                        'roots_with_any_content_failure': len({q['root_id'] for q in mc_questions if q['exact_calls'] < q['planned_calls']}),
                        'roots_all_calls_wrong': len({q['root_id'] for q in mc_questions if q['exact_calls'] == 0})},
        'ordering': {'independent_questions': len(ordering_questions),
                     'roots_with_any_content_failure': len({q['root_id'] for q in ordering_questions if q['exact_calls'] < q['planned_calls']}),
                     'roots_all_calls_wrong': len({q['root_id'] for q in ordering_questions if q['exact_calls'] == 0}),
                     'all_comparison_pairs_across_scenes': True, 'within_scene_evidence_count': 0},
    }
    experiments = [
        {'probe_id': 'reviewed_evidence_hint', 'record_status': 'EXECUTED_UNADJUDICATED_MECHANISM',
         'independent_roots': 6, 'questions': 6, 'new_calls': 90, 'calls_per_arm': 90,
         'baseline_reused': True, 'control_exact': 42, 'intervention_exact': 41,
         'multi_select_exact': {'control': 42, 'intervention': 41, 'denominator_per_arm': 54},
         'ordering_exact': {'control': 0, 'intervention': 0, 'denominator_per_arm': 36},
         'positive_roots': hint['strict_positive_distinct_roots'], 'working_threshold': 3, 'threshold_met': False,
         'interpretation': 'Reviewed source evidence hints did not show the predeclared cross-root improvement. This does not prove retrieval is irrelevant or hints generally ineffective.',
         'inputs': ['day2/PROBE_SELECTION_POLICY.json', 'day2/diagnosis_exports/observed_six_v002/HINT_CONTRAST_CONTEXT.json']},
        {'probe_id': 'global_citation_position_check_instruction', 'record_status': 'EXECUTED_AUDITED_GATE_NOT_MET',
         'independent_roots': 9, 'questions': 9, 'new_calls': 108, 'discovery_roots': 3, 'replication_roots': 6,
         'replication': order['by_cohort']['replication'], 'primary_root_equal_exact_difference': order['primary_replication_root_equal_exact_difference'],
         'predeclared_gate': order['primary_endpoint'],
         'interpretation': 'A global verification instruction gives local exact gains but does not meet the full predeclared exploratory gate. Citations can change between arms, so this is not isolated internal mechanism identification.',
         'inputs': ['day2_order_probe_v1/PROTOCOL.json', 'day2_order_probe_v1/run/SCORES.json', 'day2_order_probe_v1/audit/RESULTS_AUDIT.json']},
        {'probe_id': 'frozen_quotes_vs_verified_position_coordinates', 'record_status': 'EXECUTED_AUDITED_LIMITED_OPERATIONAL_EFFECT',
         'independent_roots': 5, 'questions': 5, 'selected_frozen_records': 11, 'candidate_control_records': 54,
         'excluded_control_records': 43, 'sampling_repeats': 4, 'matched_pairs': 44, 'new_calls': 88,
         'by_arm': frozen['by_arm'], 'root_equal_exact': frozen['root_equal_exact'],
         'root_exact_effect_counts': {'improved': 2, 'worsened': 1, 'unchanged': 2},
         'all_five_root_pair_counts_improved': True, 'citation_orders_matching_gold': 11,
         'stable_repair_demonstrated': False, 'working_support_gate': None,
         'interpretation': 'Verified positions improve some ordering relations on this selected subset; complete sorting is still wrong on 39/44 position-assisted calls. The intervention supplies the coordinates defining the mechanical target, not an inferred internal cause.',
         'inputs': ['day2_frozen_evidence_probe_v2/PROTOCOL.json', 'day2_frozen_evidence_probe_v2/run/SCORES.json', 'day2_frozen_evidence_probe_v2/INDEPENDENT_RESULTS_AUDIT.json']},
    ]
    observations = [
        {'observation_id': 'obs.mc.fp_fn_across_roots', 'status': 'REPRODUCED_OUTPUT_PATTERN',
         'claim': 'Content-admitted Train questions yield false positives in 10 roots and false negatives in 12 roots under fixed S0 direct evaluation.',
         'quantification': pattern['multi_select'], 'source_refs': ['day2/analysis/results_02/OBSERVATIONS.json#/runs/direct/requests'],
         'cause_status': 'NOT_IDENTIFIED'},
        {'observation_id': 'obs.order.low_complete_sequence_accuracy', 'status': 'REPRODUCED_OUTPUT_PATTERN',
         'claim': 'S0 direct ordering has 2/240 exact, 2472/3600 correct pairs across 20 roots; all 20 have a wrong call and 19 have no exact call.',
         'quantification': pattern['ordering'], 'source_refs': ['day2/analysis/results_02/OBSERVATIONS.json#/runs/direct/by_type/ordering'],
         'cause_status': 'NOT_IDENTIFIED'},
        {'observation_id': 'obs.order.answer_inconsistent_with_own_citation_positions', 'status': 'SOURCE_REVIEWED_EXTERNAL_INCONSISTENCY',
         'root_ids': [r['root_id'] for r in trace['results'] if r['group'] == 'EXACT_CITATIONS_UNIQUE_POSITIONS'],
         'claim': 'Three original evidence responses cite six source-review-supported locations whose order equals preserved Gold but their final answer differs. Self-reported citations are not assumed faithful internal computation traces.',
         'counts': trace['counts'], 'source_refs': ['day2/analysis/order_trace_consistency/ORDER_TRACE_CONSISTENCY.json', 'day2/analysis/order_trace_consistency/SOURCE_REVIEW.json'],
         'cause_status': 'NOT_IDENTIFIED'},
        {'observation_id': 'obs.order.literal_citations_incomplete_coverage', 'status': 'REPRODUCED_OUTPUT_PATTERN',
         'claim': '16/20 original evidence ordering questions include a nonverbatim quote, and one has tied locations; 43/54 later control records do not meet full literal unique untied eligibility. Unknown location is not automatically semantic falsity.',
         'source_refs': ['day2/analysis/order_trace_consistency/ORDER_TRACE_CONSISTENCY.json#/counts', 'day2_frozen_evidence_probe_v2/run/FROZEN_EVIDENCE.json'],
         'cause_status': 'NOT_IDENTIFIED'},
        {'observation_id': 'obs.order.positions_help_partially', 'status': 'LIMITED_CONTROLLED_OPERATIONAL_EFFECT',
         'claim': 'With the same frozen evidence, explicit verified coordinates improve root-equal exact by 0.0708333 and pooled pairs by 57/660, but do not stably repair full sorting.',
         'source_refs': ['day2_frozen_evidence_probe_v2/run/SCORES.json', 'day2_frozen_evidence_probe_v2/INDEPENDENT_RESULTS_AUDIT.json'],
         'cause_status': 'NOT_IDENTIFIED'},
    ]
    competing = [
        {'hypothesis_id': 'hyp.source_retrieval_or_binding', 'hypothesis': 'Failures can arise in locating the relevant source or binding a citation to the item.',
         'supporting_observations': ['obs.order.literal_citations_incomplete_coverage'],
         'qualification': 'Nonliteral typography, paraphrase, quote omission or citation reporting can explain some location failures. Exact citation inconsistency and residual frozen-position errors mean a retrieval-only account is insufficient for this observed subset.', 'status': 'UNCONFIRMED'},
        {'hypothesis_id': 'hyp.source_positions_to_sequence', 'hypothesis': 'The model has difficulty turning located source positions into a complete label sequence.',
         'supporting_observations': ['obs.order.answer_inconsistent_with_own_citation_positions', 'obs.order.positions_help_partially'],
         'qualification': 'The observable input-output conversion remains difficult even with coordinates, but this does not isolate numerical comparison, planning, attention, label binding, or decoding.', 'status': 'UNCONFIRMED_INTERNAL_CAUSE'},
        {'hypothesis_id': 'hyp.display_position_or_mapping_sensitivity', 'hypothesis': 'Displayed label position and label-to-content mapping contribute to errors.',
         'supporting_observations': ['Descriptive answer variation across display mappings in baseline observation file.'],
         'qualification': 'Independent sampling seeds confound between-mapping changes; ordering has one baseline sample per mapping. No causal position-sensitivity estimate.', 'status': 'UNCONFIRMED'},
        {'hypothesis_id': 'hyp.real_event_time_vs_appearance', 'hypothesis': 'The model sometimes uses real-world event time instead of appearance order.',
         'supporting_observations': [], 'qualification': 'Task and probes explicitly say source appearance; no probe separates an event-time heuristic from other errors.', 'status': 'UNTESTED'},
        {'hypothesis_id': 'hyp.semantic_or_gold_errors_explain_all', 'hypothesis': 'Question or author-Gold mistakes explain all observed ordering failure.',
         'supporting_observations': [], 'qualification': 'Prior content admission and reviewed cross-root examples contradict this as a complete explanation, but mechanical Gold monotonicity is not independent semantic certification of the entire pool.', 'status': 'NOT_SUPPORTED_AS_COMPLETE_EXPLANATION'},
        {'hypothesis_id': 'hyp.format_is_only_problem', 'hypothesis': 'All ordering failures are due to invalid output format.',
         'supporting_observations': [], 'qualification': 'Direct baseline is 240/240 valid and frozen-position probe 88/88 valid yet both have many wrong complete sequences. Format-only explanation is excluded for these runs.', 'status': 'CONTRADICTED_FOR_OBSERVED_RUNS'},
    ]
    closeout = {
        'schema_version': 'day2_diagnosis_closeout_v1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'DIAGNOSIS_WORK_COMPLETE_WITH_NEGATIVE_AND_LIMITED_RESULTS',
        'namespace': 'train.feedback', 'solver_model': 'S0_M_clean',
        'scope': 'Consolidation of original Day2 Solver diagnosis and two subsequent pre-frozen ordering interventions. This is a new closeout, not a rewrite of the sealed old Day2 status.',
        'completion_assessment': {
            's0_direct_and_evidence_branches_executed': True,
            'source_reviewed_cross_root_output_failures_recorded': True,
            'pre_frozen_controlled_tests_executed': True,
            'diagnosis_execution_and_hypothesis_testing_deliverable_complete': True,
            'confirmed_internal_mechanisms_count': 0,
            'positive_mechanism_required_to_close_experiment_phase': False,
            'justification': 'The Day2 task to run diagnosis, verify cross-root failures, and test candidate explanations has been executed and documented. Negative and limited results close this phase without inventing a positive cause; a stronger causal mechanism and transferable targeted-generation readiness remain unproven.',
            'readiness_for_day3_B_seventy_percent_verified_cause_targeting': False,
            'ready_to_publish_reproduced_causes': False,
            'fact_memory_54_pilot': 'OUTSIDE_THIS_SUBTASK_PARENT_HANDLES',
            'all_day2_goals_complete_asserted_by_this_artifact': False,
        },
        'baseline': {'direct': final['direct'], 'direct_by_type': baseline['by_type'], 'evidence_v2': final['evidence_v2'],
                     'question_count': 39, 'independent_roots': 20, 'reviewed_content_scope': 'Prior content-admission PASS reused; no full independent new semantic audit claimed.'},
        'confirmed_observations': observations, 'controlled_probe_registry': experiments,
        'unconfirmed_competing_causes': competing,
        'cause_admission': {'causes': [], 'confirmed_internal_causes': 0, 'memory_upgraded': False,
                            'original_memory_unchanged': True, 'existing_failure_cases': len(memory['failure_cases']),
                            'would_require_before_targeted_cause_generation': ['Predeclared new-root or new-family controlled replication of an operational mechanism.', 'Source-grounded content review and competing-explanation checks.', 'A separately frozen course-generation policy and new-question test; an instruction effect alone is insufficient.']},
        'semantic_review': {'scope': semantics['scope'], 'summary': semantics['summary'], 'reviewer': semantics['reviewer'],
                            'semantic_review_changes_selection': False, 'external_human_blind_gold_certification': False},
        'preserved_nonability_failures': [
            {'path': 'day2/solver/evidence', 'status': 'INFRASTRUCTURE_GRAMMAR_FAILURE', 'ability_interpretation': False, 'returned': 0, 'attempted_submissions': 8, 'not_submitted': 31},
            {'path': 'day2_frozen_evidence_probe_v1', 'status': 'ABORTED_BEFORE_INFERENCE', 'new_model_calls': 0, 'ability_interpretation': False, 'note': 'Day1 legacy seal schema compatibility bug in new auditor; v2 correction retained same experiment and inputs.'},
        ],
        'remaining_limits': [
            'All roots historically observed Train roots; no new held-out or final-evaluation claim.',
            'Repeated seeds and permutations are dependent measurements; report roots/questions separately from calls.',
            'All evaluated ordering pairs are cross-scene; no within-turn or within-scene generalization.',
            'Literal exactness identifies spans, not semantic entailment; 53/66 selected quotes need context and two use proxy offsets within their turns.',
            'The frozen probe is an output-selected 11/54 subset and its local gains cannot be projected to the excluded 43.',
            'Evidence-generating vs frozen-answer-only cross-run comparison also changes output format, decoder and budget.',
            'No statistical significance or stable repair established; no independent human blind semantic certification.',
            'No new questions or training in these probes; feedback, cause memory, fact memory and self-play training utility are unproven.',
        ],
        'next_actions_not_executed': [
            {'priority': 'day2_completion', 'action': 'Complete the separately assigned 6-root, three-arm, three-scene fact-memory trial and source review, or document a stopped/failed trial without calling it successful.', 'owner': 'parent'},
            {'priority': 'before_day3_B_targeting', 'action': 'Freeze independent Train-root or new-family replication and an end-to-end outcome rule if seeking a transferable targeted cause; keep current causes empty until evidence warrants admission.'},
            {'priority': 'week_protocol', 'action': 'If no mechanism qualifies, record B-targeted readiness failure explicitly and preserve comparable exploration-only alternatives rather than silently treating observed errors as confirmed causes.'},
        ],
        'new_solver_calls_by_closeout': 0, 'new_training_runs': 0, 'new_training_questions_generated': 0,
        'original_day2_final_status_preserved': bind(WEEK / 'day2/FINAL_STATUS.json'),
        'provenance_file': 'PROVENANCE.json',
    }
    save(HERE / 'PROVENANCE.json', provenance)
    save(HERE / 'DIAGNOSIS_CLOSEOUT.json', closeout)
    report = '''# 第2天诊断工作收口

S0正式作答、证据支路、跨root失败核查和两轮事前冻结的排序对照已经完成。诊断工作可按“已执行、结果为负或有限支持”收口；**已确认的内部错误机制仍为0**，不要求做出正结论才算完成实验。尚不能声称已准备好按已验证原因进行70%定向的B组生成。

这是新增的收口记录，不改写旧`day2/FINAL_STATUS.json`。旧文件封存时记录“诊断里程碑完成、Day2整体未完成”是当时的正确状态。6root事实memory三臂54场由主任务另行执行，本诊断文件不代替该实验，也不单独宣布Day2整体完成。

## 已确认的外部观察

| 范围 | 实际覆盖 | 可确认结果 | 不能据此确认 |
|---|---|---|---|
| S0直接作答 | 39题、20roots、582/582有效 | 多选197/342全对；排序2/240全对、2472/3600成对正确 | 不能把582次当582独立题；不能推定内部原因 |
| 多选错误分解 | 19题、19roots、342次 | 误选见于10roots，漏选见于12roots；98次有FP、56次有FN；94次过选、47次少选 | 误选/漏选是错误类型，未等同人物绑定、否定或阈值机制 |
| 排序跨root失败 | 20题、20roots | 20roots至少一次错；19roots所有12次均无全对；比较对全部跨场景 | 没有同场景、同turn排序结论 |
| 原证据支路核查 | 20排序题、120条引用 | 3roots的来源支持引用可唯一定位，其位置顺序等于原Gold，最终答案却不同 | 自报引用不是忠实内部计算轨迹；其余17题不能并为相同原因 |

原证据支路总体39次中38次严格有效、7次全对；20道排序0次全对、219/300成对正确。多选Gold2/3/4/5分层、FP/FN、同mapping与跨mapping变化均保存在旧观察文件。跨mapping同时改变独立采样seed，排序每mapping仅一次，不能直接称因果位置敏感。

## 受控检验得到什么

| 检验 | 独立覆盖及分母 | 对照→处理结果 | 判定 |
|---|---|---|---|
| 审核证据提示 | 6题、6roots；每臂90次；旧对照复用、新调用90次 | 总全对42→41；多选42/54→41/54；排序0/36→0/36；仅1root改善 | 预设3个改善root门槛未达到；旧记录未认定语义机制审核通过 |
| 增加引用位置核对指令 | 9题、9roots；3发现＋6复核；108新调用 | 复核1/36→4/36全对，408/540→398/540成对；位置一致性增加，但1次格式失效 | 原5项工作门槛未达到；局部全对提升不足以称稳定支持 |
| 固定引文，再额外给核验位置 | 5题、5roots；11条历史响应×4重复×2臂＝88新调用 | root等权全对2.92%→10.00%；汇总2/44→5/44；499/660→556/660成对 | 有限操作性效果；5roots成对数均升，整题2升1降2不变，给位置仍39/44整题错 |

第三个实验主要参照是固定引文的机械顺序，Gold为次要参照；本次11组引用排序恰好均等于Gold，所以数值相同。这是评分后事实，未用于选样。11条来自54条旧control中逐字唯一、无并列的子集；43条不满足机械条件均保留。4次重复没有把5roots变成44个独立样本。

提供坐标直接简化了当前机械任务，因此能支持“这一辅助形式在所选子集上改善部分关系”；不能认定改善由内部检索、数字比较、注意力、标签映射或解码中的哪一步导致。与前轮相比又同时改变了引用生成任务、输出格式、解码及预算，不能把跨轮变化全归因固定引文。

## 原因清单保持空

- **检索或引用绑定困难：未确认。** 许多引用无法严格定位，但排版差异、改写或报告错误也可能导致；已有正确引用和显式位置时仍错，使“只有检索问题”不足以解释当前子集。
- **从位置构造完整顺序困难：可作为可检验假设。** 位置与最终答案不一致是外部事实；现有干预仍没有隔离内部比较、标签绑定或输出机制。
- **展示位置敏感、真实事件时间替代原文出现顺序：未得到因果验证。** 不能把描述性变异或不完整轨迹升级原因。
- **全是格式问题：对本次direct与固定引文run可排除。** 两者分别240/240与88/88有效，仍有大量错误。
- **全是错题或Gold错误：不能作为完整解释。** 既有准入和来源核查给出多个可核验反例；同时不把机械锚点单调当作整池独立语义认证。

现有弱点memory保留6个训练侧失败case、0个cause，原文件不变。当前可以记录已复现的误选、漏选、错序和引用—答案不一致等**观察**，不能将其冒充已验证生成原因。冻结引文实验没有设新的通过门槛，不能事后套用或另造门槛宣称通过。

## 来源与语义限制

66条固定引文的既有Codex解释性审核在完整场景上下文下均支持条目；短quote自身仅13条充分，53条需上下文，2条没有覆盖目标子句而使用同turn代理位置。当前六项目标分处不同turn，未改变本题组相对顺序；不能外推到同turn项目。该审核不是外部人工盲Gold认证，也未改变选样与数值评分。本次收口未重新进行语义审核。

所有roots都是历史已观察Train roots；不是新TEST。证据schema的旧backend故障（8次尝试无返回）与固定引文v1的审核字段兼容中止（0次调用）是基础设施失败，均原样保留，不能计作模型能力0%。两轮有效排序对照分别经17,758项和15,781项独立复算检查通过；本收口另核验5个旧封存目录的完整字节与文件集合。

## 完成判断与尚未执行的工作

第2天要求的“固定S0正式/诊断作答、跨root核查、用事前冻结对照检验假设”已经执行，有可复核原始记录、否定/有限结果与原因清单。**执行完成不等于原因已确认，也不等于已修复弱点。** 为得到正结论而不停改提示不是本阶段的完成条件。

本子任务未运行memory、未新出题、未训练、未更新原因memory。主任务须单列事实memory54场及质量/成本判断，才可汇总Day2整体状态。后续若进入按原因定向的B组，应另冻结新Train root/题族复核、端到端成功规则与竞争解释检查；若没有机制达标，应明确记录B定向准备不足并保留公平的探索替代，不能悄悄将错误观察改名已确认cause。以上为建议，均未在本收口执行。

## 文件入口

- [结构化收口](DIAGNOSIS_CLOSEOUT.json)：观察、3类已执行检验、竞争解释、原因准入与剩余限制。
- [输入来源与SHA256](PROVENANCE.json)：本报告来源、五个旧封存快照与复核范围。
- [原正式/诊断观察](../../day2/analysis/results_02/REPORT_ZH.md)、[原引用核查](../../day2/analysis/order_trace_consistency/REPORT_ZH.md)。
- [核对指令对照](../../day2_order_probe_v1/REPORT_ZH.md)、[固定引文位置对照](../../day2_frozen_evidence_probe_v2/REPORT_ZH.md)、[66条引用审核](../../day2_frozen_evidence_probe_v2/SEMANTIC_REVIEW_ZH.md)。
- [本次构建校验](VALIDATION.json)。所有源文件保持原样；本报告不重写旧状态、不作任何反馈训练或self-play收益声明。
'''
    with (HERE / 'REPORT_ZH.md').open('x') as stream:
        stream.write(report)
    check('serialized_status', read(HERE / 'DIAGNOSIS_CLOSEOUT.json')['status'] == closeout['status'])
    check('no_cause_entries', read(HERE / 'DIAGNOSIS_CLOSEOUT.json')['cause_admission']['causes'] == [])
    check('explicit_no_day2_whole_claim', read(HERE / 'DIAGNOSIS_CLOSEOUT.json')['completion_assessment']['all_day2_goals_complete_asserted_by_this_artifact'] is False)
    for row in provenance['inputs']:
        check('input_still_unchanged:' + row['relative_to_week'], bind(row['path'])['sha256'] == row['sha256'])
    validation = {'status': 'PASS', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                  'checks_count': len(CHECKS), 'sealed_files_verified': sealed_files, 'checks': CHECKS,
                  'new_model_calls': 0, 'new_training_runs': 0, 'private_gold_json_read': False,
                  'semantic_review_reperformed': False, 'sealed_files_modified': False,
                  'outputs': [bind(HERE / name) for name in outputs if name != 'VALIDATION.json']}
    save(HERE / 'VALIDATION.json', validation)
    print(json.dumps({'status': closeout['status'], 'validation': validation['status'], 'checks': len(CHECKS), 'sealed_files_verified': sealed_files, 'causes': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
