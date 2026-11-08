#!/usr/bin/env python3
"""Record bounded Day3 progress without revising sealed production or policy."""
import hashlib
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

OUT = Path(__file__).resolve().parent
WEEK = OUT.parent


def binding(path):
    data = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    now = datetime.now(timezone.utc)
    material = WEEK / 'day3_material_feedback_v1'
    ms = json.loads((material / 'SUMMARY.json').read_text())
    ma = json.loads((material / 'audit/INDEPENDENT_FEEDBACK_CHECK.json').read_text())
    reconstruction = OUT / 'reconstruction/cc-train-episode-105961.json'
    rr = json.loads(reconstruction.read_text())
    audit_path = OUT / 'audit/FIRST_ROOT_RECONSTRUCTION_AUDIT.json'
    ra = json.loads(audit_path.read_text())
    assert str(ma['status']).startswith('PASS')
    assert str(ra['status']).startswith('PASS')
    assert ms['original_shortfall_rows'] == 42 and ms['merged_root_requests'] == 20
    assert rr['summary']['mapping_status_counts'] == {'MAPPED': 11, 'HOLD': 24}
    assert rr['summary']['normalized_fact_definitions'] == 91
    status = 'DAY3_MATERIAL_FEEDBACK_READY_COMMON_EVAL_SEMANTIC_FREEZE_PENDING'
    next_action = ('Prioritize60 source-supported common-evaluation slots in the original fixed order; '
                   'inventory actual families, resolve concrete related historical uncertainties and '
                   'complete all five exclusions before hash selection, question/Gold authorship, '
                   'independent source-first QA and full request freeze. Keep98 material HOLD and28 '
                   'shortfalls; next-version shared Train expansion consumes the prepared material '
                   'feedback under a separate execution/source freeze, not by rewriting current slots.')
    summary = {
        'schema_version': 'day3_material_feedback_and_bounded_semantic_progress_v1',
        'updated_at_utc': now.isoformat(), 'status': status,
        'material_feedback': {'shortfalls': 42, 'joint_relation': 40, 'explicit_revision': 2,
                              'paired_targets': 21, 'root_requests': 20,
                              'summary': binding(material / 'SUMMARY.json'),
                              'report': binding(material / 'REPORT_ZH.md'),
                              'independent_audit': binding(material / 'audit/INDEPENDENT_FEEDBACK_CHECK.json')},
        'evaluation_priority': {'source_supported_slots': 60, 'material_hold': 98,
                                'source_shortfall': 28,
                                'queue': binding(OUT / 'COMMON_EVAL_WORK_QUEUE.json')},
        'historical_support_mapping': {
            'reviewed_roots': 1, 'reviewed_records': 35, 'reviewed_items': 149,
            'mapped': 11, 'hold': 24, 'unique_exported_facts': 91,
            'exact_quote_occurrences': 464,
            'reconstruction': binding(reconstruction), 'independent_audit': binding(audit_path),
            'author_release': False,
            'limit': 'Historical support reconstruction only; no five-category new-family clearance.'},
        'actual_new_evaluation_questions': 0, 'actual_frozen_evaluation_requests': 0,
        'source_extensions_created': 0, 'new_solver_calls': 0, 'new_training_runs': 0,
        'full_day_complete': False, 'training_start_go': False,
        'remaining_checklist': binding(WEEK / 'DAY3_REMAINING_ZH.md'),
        'next_action': next_action,
    }
    write(OUT / 'SUMMARY.json', summary)
    (OUT / 'REPORT_ZH.md').write_text(
        '# 第三天素材反馈与共同评估排重进展\n\n'
        '第三天仍未完成。素材反馈包已准备并独立复核：42个原缺额，40个联合关系、2个显式修订，'
        '合并21个配对目标、20个root扩充请求。只面向下一版本共同来源，不修改当前封存数据。\n\n'
        '共同评估已按原顺序列出60个素材支持题位，另98个素材HOLD、28个短缺保持原记录。'
        '首root105961完整审核100turn、35条历史题、149实际条目，11条事实映射明确、24条HOLD；'
        '导出91个唯一事实指纹、464条精确引文实例。独立审核已通过限定范围；'
        '该映射仅供内部边界审核，不是新事实族排重通过或新评估题。\n\n'
        '下一步优先完成实际候选相关的五类语义排重，通过后选族、出题、独立全文重建Gold、'
        '检查完整prompt并冻结。993条旧格式缺口不作全局强制重跑门槛，相关未知映射仍HOLD。'
        '新增评估题0、实际扩充0、Solver调用0、新训练0，不能据此报告难度或反馈收益。\n\n'
        '详细收口要求见上级DAY3_REMAINING_ZH.md；证据及哈希见SUMMARY.json。\n')
    progress_ref = binding(OUT / 'SUMMARY.json')
    for filename in ('CURRENT_STATUS.json', 'DAY3_LATEST.json'):
        path = WEEK / filename
        data = path.read_bytes()
        backup = OUT / 'history' / (filename + '.BEFORE_BOUNDED_PROGRESS.' + hashlib.sha256(data).hexdigest()[:16])
        backup.parent.mkdir(exist_ok=True)
        if not backup.exists():
            backup.write_bytes(data)
        current = json.loads(data)
        current.update(status=status, updated_at_utc=now.isoformat(), next_action=next_action)
        if filename == 'CURRENT_STATUS.json':
            current['updated_at_beijing'] = now.astimezone(timezone(timedelta(hours=8))).isoformat()
            current['daily_planning']['day3'] = status
        current['day3_material_and_semantic_progress'] = progress_ref
        write(path, current)
    print(json.dumps({'status': status, 'summary': progress_ref,
                      'full_day_complete': False, 'new_evaluation_questions': 0}))


if __name__ == '__main__':
    main()
