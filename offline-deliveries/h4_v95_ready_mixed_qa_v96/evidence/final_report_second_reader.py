"""Second-reader report/export verification; no production calls or writes."""
import ast
from collections import Counter
from contextlib import redirect_stdout
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]
SHARD = HOME / 'shards/shard_01'
READS = {}


def binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def read(path):
    path = Path(path).resolve()
    READS[path] = binding(path)
    return json.loads(path.read_text())


def read_text(path):
    path = Path(path).resolve()
    READS[path] = binding(path)
    return path.read_text()


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def is_control(question):
    return question['question_type'] == 'multi_select' and len(question['gold']) == 6


def stats(rows):
    result = {}
    for model in ['base', 'trained']:
        exact = sum(row[model]['exact'] for row in rows)
        partial = sum((Fraction(row[model]['partial']) for row in rows), Fraction())
        result[model] = {'calls': len(rows), 'exact_correct': exact,
                         'exact_accuracy': str(Fraction(exact, len(rows))),
                         'partial_mean': str(partial / len(rows)),
                         'exact_percentage': f'{exact / len(rows) * 100:.2f}%'}
    return result


class CaptureCollectorWrites(ast.NodeTransformer):
    """Intercept collector outputs in memory, leaving input reads unchanged."""
    def visit_Assert(self, node):
        if ast.unparse(node.test) == 'not path.exists()':
            return None
        return self.generic_visit(node)

    def visit_Expr(self, node):
        call = node.value
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name) and call.func.value.id == 'path'
                and call.func.attr == 'write_text'):
            return ast.copy_location(ast.Expr(value=ast.Call(
                func=ast.Name(id='capture_markdown', ctx=ast.Load()),
                args=[call.func.value, *call.args], keywords=[])), node)
        return self.generic_visit(node)


def main():
    pool = read(SHARD / 'EVALUATION_POOL.json')['questions']
    selection = read(SHARD / 'SELECTED_CANDIDATES.json')['questions']
    exported = read(HOME / 'NEW_SELECTED_CANDIDATES.json')
    calls_record = read(SHARD / 'PAIRED_CALLS.json')
    calls = calls_record['calls']
    report = read(SHARD / 'MIXED_EVALUATION_REPORT.json')
    stage = read(HOME / 'evidence/COMPLETE_STAGE_RESULTS.json')
    independent = read(HOME / 'evidence/INDEPENDENT_RESULT_AUDIT.json')
    transport = read(HOME / 'evidence/ACTUAL_QA_TRANSPORT_PRIOR_AUDIT.json')
    markdown = read_text(HOME / 'COMPLETE_RESULTS_ZH.md')
    synthesis = read_text(HOME / 'evidence/QUALITY_SYNTHESIS_ZH.md')
    upstream = read_text(HOME.parent / 'h4_controlled_six_root_expansion_v95/COMPLETE_RESULTS_ZH.md')
    need(report['status'] == independent['status'] == 'PASS'
         and report['calls_per_model'] == 192 and len(calls) == 192
         and report['raw_scores_and_selection_replayed']
         and not report['training_started'] and not calls_record['training_started'],
         'completed_result_or_training_status_mismatch')
    by_id = {question['candidate_id']: question for question in pool}
    need(len(by_id) == len(pool) == 11 and len(selection) == 10
         and all(question == by_id[question['candidate_id']] for question in selection)
         and {question['candidate_id'] for question in selection} == set(report['selected_candidate_ids']),
         'source_selection_is_not_exact_pool_projection')
    expected_new = [question for question in selection if not is_control(question)]
    controls = [question for question in selection if is_control(question)]
    need(len(expected_new) == len(exported['questions']) == exported['question_count'] == 8
         and exported['questions'] == expected_new
         and exported['training_allowed'] is False and exported['formal_training_admission'] is False
         and exported['status'] == 'PROVISIONAL_REVIEWED_AND_SOLVER_SELECTED'
         and len(controls) == 2
         and {question['candidate_id'] for question in controls} == {
             'v96_root25652_fresh_parent', 'v96_root19708_fresh_parent'},
         'new_candidate_export_changed_content_order_or_controls')
    for key, path in [('source_pool', SHARD / 'EVALUATION_POOL.json'),
                      ('source_selection', SHARD / 'SELECTED_CANDIDATES.json'),
                      ('independent_result_audit', HOME / 'evidence/INDEPENDENT_RESULT_AUDIT.json')]:
        need(exported[key] == binding(path), 'new_export_source_sha_mismatch:' + key)
    distribution = dict(sorted(Counter(str(len(question['gold'])) for question in expected_new
                                       if question['question_type'] == 'multi_select').items()))
    need(distribution == {'2': 2, '3': 1, '4': 2, '5': 2}
         and sum(question['question_type'] == 'ordering' for question in expected_new) == 1,
         'selected_new_distribution_mismatch')
    computed = {}
    for label, predicate in [
        ('new_multiselect', lambda question: question['question_type'] == 'multi_select' and not is_control(question)),
        ('current_parent_gold6_control', is_control),
        ('ordering', lambda question: question['question_type'] == 'ordering'),
    ]:
        ids = {question['candidate_id'] for question in pool if predicate(question)}
        rows = [row for row in calls if row['candidate_id'] in ids]
        computed[label] = {'question_count': len(ids), 'models': stats(rows)}
        for model, model_stats in computed[label]['models'].items():
            for key in ['calls', 'exact_correct', 'exact_accuracy', 'partial_mean']:
                need(stage['new_vs_control_statistics'][label]['models'][model][key] == model_stats[key],
                     'stage_statistic_different_from_paired_calls:' + label + ':' + model + ':' + key)
    need(computed['new_multiselect']['question_count'] == 8
         and computed['new_multiselect']['models']['base']['exact_correct'] == 85
         and computed['new_multiselect']['models']['trained']['exact_correct'] == 92
         and computed['new_multiselect']['models']['base']['calls'] == 144
         and computed['ordering']['models']['base']['exact_correct'] == 8
         and computed['ordering']['models']['trained']['exact_correct'] == 7
         and computed['ordering']['models']['base']['calls'] == 12,
         'requested_headline_statistics_mismatch')
    root_summary = []
    for root, short, turns, added in [('cc-train-episode-25652', '25652', 105, 14),
                                     ('cc-train-episode-19708', '19708', 104, 16)]:
        root_pool = [question for question in pool if question['root_id'] == root]
        root_new = [question for question in expected_new if question['root_id'] == root]
        source = read(HOME / f'sources/root{short}_SOURCE.json')
        need(len(source['turns']) == turns
             and len([turn for turn in source['turns'] if turn['turn_ref'].startswith('R02:B:S01:')]) == added,
             'root_source_turn_count_mismatch')
        root_summary.append({'root_id': root, 'new_usable': len([question for question in root_pool if not is_control(question)]),
                             'new_selected': len(root_new), 'source_turns': turns, 'added_turns': added})
    need([row['new_selected'] for row in root_summary] == [4, 4], 'per_root_new_selection_mismatch')
    dropped = [question for question in pool if not is_control(question)
               and question['candidate_id'] not in {row['candidate_id'] for row in expected_new}]
    need(len(dropped) == 1 and dropped[0]['root_id'] == 'cc-train-episode-25652'
         and dropped[0]['question_type'] == 'multi_select' and len(dropped[0]['gold']) == 3,
         'unexpected_new_content_candidate_excluded_by_solver')
    dropped_calls = [row for row in calls if row['candidate_id'] == dropped[0]['candidate_id']]
    need(len(dropped_calls) == 18 and all(row['trained']['exact'] for row in dropped_calls),
         'gold3_perfect_exclusion_statement_mismatch')
    ordering_review = read(SHARD / 'ordering/CONTENT_REVIEW.json')
    need(ordering_review['accepted_count'] == ordering_review['rejected_count'] == 1
         and {row['candidate_id']: row['status'] for row in ordering_review['questions']} == {
            'cc-train-episode-25652-v96_shard_01-ordering-01': 'EXCLUDED',
            'cc-train-episode-25652-v96_shard_01-ordering-02': 'USABLE'},
         'ordering_content_disposition_mismatch')
    need(transport['actual_http_attempts'] == 4
         and transport['multiselect_complete_historical_prior_status'] == 'PASS'
         and all(row['actual_prior_rows'] == 0 for row in transport['ordering_prior_limitation']),
         'actual_transport_or_prior_disclosure_mismatch')
    need(stage['new_selected'] == 8 and stage['solver_selected_including_controls'] == 10
         and stage['selected_controls_excluded_from_new_count'] == 2
         and stage['new_content_usable'] == 9 and stage['content_usable_including_controls'] == 11
         and stage['public_questions_including_controls'] == 12
         and stage['candidate_slots_including_controls'] == 14
         and stage['new_selected_multiselect_by_gold'] == distribution,
         'stage_count_funnel_mismatch')
    for phrase in ['85/144 = 59.03%', '92/144 = 63.89%', '8/12 = 66.67%', '7/12 = 58.33%',
                   '另有 2 道全对对照入选，但不算新增训练题', '尚未训练或正式准入',
                   '不是本轮数据已经训练后的收益', '也不是第一轮旧题的同题复测',
                   '不能据此宣称整体难度提升、训练效果显著或全量生产稳定',
                   '144 次采样不能当成 144 道独立题', '排序实际题面 prior 为 []',
                   '不能声称排序 writer 看到了全部旧排序题文本', '非完全盲审']:
        need(phrase in markdown, 'missing_report_fact_or_required_limit:' + phrase)
    need('独立可用 2/6' in upstream and '模型流程接纳 3/6' in upstream
         and '42 次 HTTP' in upstream and '634–3126' in upstream
         and '9 次实际 planner 和 4 次 realizer' in upstream,
         'upstream_report_claims_mismatch')
    # Re-run only the export/report collector in memory with every output intercepted.
    # This creates no production files and invokes no model or batch operation.
    collector_path = HOME / 'evidence/collect_complete_results.py'
    tree = ast.parse(read_text(collector_path), filename=str(collector_path))
    tree = ast.fix_missing_locations(CaptureCollectorWrites().visit(tree))
    outputs = {}
    namespace = {'__name__': '_second_reader_collector', '__file__': str(collector_path)}
    exec(compile(tree, str(collector_path), 'exec'), namespace)
    namespace['write'] = lambda path, value: outputs.__setitem__(str(Path(path).resolve()), value)
    namespace['capture_markdown'] = lambda path, value: outputs.__setitem__(str(Path(path).resolve()), value)
    with redirect_stdout(io.StringIO()):
        namespace['main']()
    need(outputs[str(HOME / 'COMPLETE_RESULTS_ZH.md')] == markdown,
         'actual_markdown_not_exact_collector_output')
    need(outputs[str(HOME / 'NEW_SELECTED_CANDIDATES.json')] == exported,
         'actual_export_not_exact_collector_projection')
    regenerated_stage = outputs[str(HOME / 'evidence/COMPLETE_STAGE_RESULTS.json')]
    need({key: value for key, value in regenerated_stage.items() if key != 'completed_at_utc'} ==
         {key: value for key, value in stage.items() if key != 'completed_at_utc'},
         'stage_output_changed_beyond_completion_timestamp')
    result = {'version': 'v96_final_report_second_reader_v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'PASS',
        'reader': '/root/v92_runner_audit',
        'scope': 'Actual final Chinese report, exact new-only selection export, descriptive aggregate arithmetic and claim limits',
        'raw_solver_response_replay_repeated': False,
        'independent_raw_replay_reference': binding(HOME / 'evidence/INDEPENDENT_RESULT_AUDIT.json'),
        'actual_markdown_exactly_reproduced_from_collector_in_memory': True,
        'actual_new_export_exact_original_selection_projection_including_order': True,
        'current_fresh_parent_gold6_controls_excluded': [question['candidate_id'] for question in controls],
        'new_selected_count': 8, 'new_selected_by_type': {'multi_select': 7, 'ordering': 1},
        'new_selected_multiselect_by_gold': distribution,
        'candidate_count_funnel': {'planned_slots_with_controls': 14, 'public_questions_with_controls': 12,
                                  'content_usable_with_controls': 11, 'solver_selected_with_controls': 10,
                                  'new_content_usable': 9, 'new_selected': 8},
        'descriptive_statistics_recomputed_from_paired_calls': computed,
        'root_summary': root_summary,
        'report_does_not_claim_new_training_benefit_or_global_difficulty_increase': True,
        'report_discloses_current_checkpoints_same_new_questions_not_old_item_retest': True,
        'report_discloses_two_related_question_families_and_nonblind_limits': True,
        'report_discloses_actual_multiselect_prior_complete_and_ordering_text_prior_empty': True,
        'report_discloses_failed_generation_review_and_control_role_legacy_string': True,
        'observed_material_report_errors': [], 'corrections_performed': [],
        'model_or_batch_operations_performed': 0,
        'read_exposure': 'This second-reader verification reads current private evaluation/selection Gold and paired scored results, previous review/transport audit records and public/source reports. It is a mechanical reporting check, not a blind semantic review.',
        'input_files': [READS[path] for path in sorted(READS)],
        'audit_script': binding(__file__)}
    target = HOME / 'evidence/FINAL_REPORT_SECOND_READER.json'
    with target.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': 'PASS', 'new_selected': 8, 'new_multiselect_by_gold': distribution,
                      'exact_report_and_export_projection': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
