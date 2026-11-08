"""Add explicit generation-recovery reporting while preserving frozen v58 exporters."""
from pathlib import Path
import json
import finalize_report as frozen
import recover_generation
import run_collection as collection

HERE = Path(__file__).resolve().parent


def recovery_audit():
    receipt = recover_generation.verify_receipt()
    complete = collection.read(HERE / 'GENERATION_RECOVERY.COMPLETE.json')
    generation = collection.completion('generation')
    frozen.require(generation.get('ordinary_uninterrupted_execution') is False, 'generation_recovery_not_disclosed')
    collection.verify_binding(generation['explicit_recovery'])
    for key in ('preflight', 'script', 'original_error_archive', 'sealed'):
        collection.verify_binding(complete[key])
    sealed = collection.read(complete['sealed']['path'])
    for bound in sealed['empty_outputs']:
        collection.verify_binding(bound)
    archive = collection.binding(HERE / 'generation.ERROR.recovered.json')
    frozen.require(archive['sha256'] == receipt['original_error']['sha256'], 'original_error_archive_bytes_changed')
    frozen.require(complete['repeated_http_calls'] == 0 and complete['c01_http_calls_reused'] == 1
                   and complete['continued_chunks'] == [2,3,4,5,6], 'generation_recovery_call_scope_changed')
    frozen.require(complete['total_http_calls'] == generation['http_attempts'], 'recovery_total_calls_mismatch')
    first = generation['chunks'][0]
    frozen.require(first['http_attempts'] == 1 and first['mechanical_pass'] == first['ready_for_solver'] == 0,
                   'c01_incomplete_response_salvaged_or_retried')
    plan = collection.verify_plan()
    wrapper_binding = collection.binding(__file__)
    frozen_scripts = [r for r in plan['report_scripts'] if Path(r['path']).name in
        ('finalize_report.py', 'audit_provenance.py', 'comparison.py', 'run_paired_base.py')]
    for row in frozen_scripts:
        collection.verify_binding(row)
    return {'status': 'PASS', 'wrapper': wrapper_binding, 'frozen_reporting_scripts': frozen_scripts,
            'generation_recovery_complete': collection.binding(HERE / 'GENERATION_RECOVERY.COMPLETE.json'),
            'original_error_archive': archive, 'original_error_original_path': receipt['original_error']['path'],
            'c01_http_calls_reused': 1, 'c01_repeated_http_calls': 0, 'c01_generated_questions_admitted': 0,
            'remote_truncation_and_test_binding_interruption_are_separate_failures': True,
            'test_restored_byte_identically_before_resume': True,
            'generation_complete_is_explicitly_recovered_not_uninterrupted': True,
            'scoring_selection_questions_gold_and_inputs_unchanged': True,
            'model_calls_by_wrapper': 0}


def main():
    audit = recovery_audit()
    frozen.require((HERE / 'BASE_EVALUATION.COMPLETE.json').is_file(), 'both_models_must_complete_before_final_export')
    frozen.require(not (HERE / 'RESULT_SUMMARY.json').exists(), 'final_export_already_done')
    recovery_path = HERE / 'GENERATION_RECOVERY_AUDIT.json'
    frozen.write(recovery_path, audit)
    original_write = frozen.write
    original_markdown = frozen.report_markdown

    def write_with_recovery(path, value):
        if Path(path).name in ('RESULT_SUMMARY.json', 'VERIFICATION.json'):
            value = {**value, 'generation_recovery': audit}
        original_write(path, value)

    def markdown_with_recovery(summary):
        content = original_markdown(summary)
        totals = summary['totals']
        content += (f"\n\n出题/排序审核共 {totals['remote_error_requests']} 次远程响应失败，"
                    f"{totals['local_capacity_blocked_requests']} 次本地容量阻断；整套6声明重复剔除 {totals['duplicate_rejected_slots']} 个名额，其他机械拒绝 {totals['other_mechanical_rejected_slots']} 个。初稿数只统计完整可解析 JSON 中的题目，截断文本未作为草稿或可用题计入。")
        note = ('\n\n首批 c01 的生成响应超过输出上限而截断，原生流程将2个名额判为失败、产出0题；'
                '同时一次测试文件追加触发冻结校验中断。测试文件随后恢复原字节，保真归档原错误，'
                '仅补写已判空批次的导出文件，从 c02 继续。c01 的1次 HTTP 保留在总数中，未重试、未提取截断内容。'
                '\n\n[显式恢复核验](GENERATION_RECOVERY_AUDIT.json) · [原始错误归档](generation.ERROR.recovered.json)')
        return content + note

    frozen.write = write_with_recovery
    frozen.report_markdown = markdown_with_recovery
    frozen.INPUTS.add(HERE / 'GENERATION_RECOVERY.COMPLETE.json')
    frozen.INPUTS.add(HERE / 'generation.ERROR.recovered.json')
    frozen.main()


if __name__ == '__main__':
    main()
