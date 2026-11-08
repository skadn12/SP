"""Refresh the current navigation after the bounded D4-SETS decision is sealed."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path


O = Path(__file__).resolve().parent
W = O.parent


def binding(path):
    data = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    decision = json.loads((O / 'DECISION.json').read_text())
    assert decision['independent_audits_pass']
    assert decision['status'] == 'D4_SETS_PASS_BOUNDED_COMMON_ELIGIBLE_RESERVE'
    assert decision['training_start_go'] is False
    history = O / 'navigation_history'
    history.mkdir(exist_ok=True)
    for name in ('CURRENT_STATUS.json', 'README_ZH.md'):
        backup = history / name
        if not backup.exists():
            shutil.copy2(W / name, backup)
    now = dt.datetime.now(dt.timezone.utc)
    status = json.loads((W / 'CURRENT_STATUS.json').read_text())
    status['updated_at_utc'] = now.isoformat()
    status['updated_at_beijing'] = now.astimezone(dt.timezone(dt.timedelta(hours=8))).isoformat()
    status['status'] = decision['status']
    status['active_workstream'] = 'D4_SETS_CLOSED_BOUNDED_NEXT_D4_SCHEDULE'
    status['next_action'] = decision['next_work']
    status['mainline_next_action_before_diversity_pilot'] = decision['next_work']
    status['latest_iteration_solver_calls'] = 0
    status['training_start_go'] = False
    status['full_day_complete'] = False
    status['current_execution_summary'] = binding(O / 'SUMMARY.json')
    status['current_execution_report'] = binding(O / 'REPORT_ZH.md')
    status['training']['common_foundation_eligible_reserve_per_arm'] = 363
    status['training']['common_foundation_admission'] = decision['status']
    status['training']['new_AB_pools_frozen'] = False
    status['files']['day4_common_set_admission_report'] = 'day4_common_set_admission_v1/REPORT_ZH.md'
    status['files']['day4_common_set_admission_ledger'] = 'day4_common_set_admission_v1/ADMISSION_LEDGER.json'
    status['day4_common_set_admission_progress'] = {
        'status': decision['status'], 'decision': binding(O / 'DECISION.json'),
        'report': binding(O / 'REPORT_ZH.md'), 'summary': binding(O / 'SUMMARY.json'),
        'manifest': binding(O / 'FINAL_MANIFEST.json'),
        'admission_ledger': binding(O / 'ADMISSION_LEDGER.json'),
        'common_eligible_reserve': binding(O / 'COMMON_ELIGIBLE_BASE_SET.json'),
        'user_exclusion_rule': 'ONLY_WHOLE_TASK_SEMANTIC_EQUIVALENCE',
        'decisions': decision['counts']['decisions'],
        'eligible_common_reserve_per_arm': 363, 'same_root_reviewed_foundation': 102,
        'same_root_pairs': 898, 'cross_root_pairs_retrieved_and_reviewed': 840,
        'cross_root_pairs_not_individually_semantically_adjudicated': 62150,
        'bounded_cross_root_review_only': True,
        'partial_overlap_retained_with_dependencies': True,
        'identical_common_eligible_sets': True,
        'final_training_records_A': 0, 'final_training_records_B': 0,
        'D4_SCHEDULE': 'NOT_FROZEN', 'q_final': None,
        'new_solver_calls': 0, 'new_training_runs': 0, 'training_start_go': False,
    }
    save(W / 'CURRENT_STATUS.json', status)
    save(W / 'DAY4_LATEST.json', {
        'schema_version': 'selfplay_day4_latest_v1',
        'updated_at_utc': now.isoformat(),
        'status': decision['status'],
        'decision': binding(O / 'DECISION.json'), 'summary': binding(O / 'SUMMARY.json'),
        'report': binding(O / 'REPORT_ZH.md'), 'manifest': binding(O / 'FINAL_MANIFEST.json'),
        'next_action': decision['next_work'], 'training_start_go': False,
    })
    readme = W / 'README_ZH.md'
    marker = '<!-- DAY4_COMMON_SET_ADMISSION_LATEST -->'
    if marker not in readme.read_text():
        original = readme.read_text()
        title, body = original.split('\n', 1)
        update = '''
<!-- DAY4_COMMON_SET_ADMISSION_LATEST -->
最新完成[第四天公共基础集合准入](day4_common_set_admission_v1/REPORT_ZH.md)。按用户“仅整题完全重复才排除”的规则，固定363道基础题（238多选、125排序）均保留，A/B共用完全一致的合格储备。20个共享root内102道旧题已完成898个候选对的审查；跨root有限检索命中840对并裁决，未命中62150对没有逐对语义审核，故不宣称穷尽无重复或事实独立。部分重叠均保留依赖链接，两份独立审计通过。

D4-SETS在上述范围内完成；下一步是D4-SCHEDULE的实际训练池、监督token混合比例、等步数和runtime冻结。实际选入训练的记录仍各0，q_final未冻结，training_start_go=false；本轮Solver0次、训练0次，第三天累计仍3417次。见[逐题账本](day4_common_set_admission_v1/ADMISSION_LEDGER.json)、[最终裁决](day4_common_set_admission_v1/DECISION.json)与[第四天当前状态](DAY4_LATEST.json)。以下为历史快照，以本段及CURRENT_STATUS.json为准。
'''
        readme.write_text(title + '\n' + update + body)
    save(O / 'NAVIGATION_UPDATE.json', {
        'schema_version': 'day4_sets_navigation_update_v1',
        'decision_binding': binding(O / 'DECISION.json'),
        'updated_files': [binding(W / name) for name in ('CURRENT_STATUS.json', 'DAY4_LATEST.json', 'README_ZH.md')],
        'previous_navigation': [binding(history / name) for name in ('CURRENT_STATUS.json', 'README_ZH.md')],
        'training_start_go': False,
    })
    print(json.dumps({'status': 'NAVIGATION_UPDATED', 'day4_sets': decision['status'], 'training_start_go': False}))


if __name__ == '__main__':
    main()
