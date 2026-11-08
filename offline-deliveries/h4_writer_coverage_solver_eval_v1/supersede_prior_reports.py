"""Correct derived eligibility reports, preserving their previous bytes.

Never change Writer prompts, responses, candidates, provenance or call records.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

D = Path(__file__).resolve().parent
OLD = D.parent / 'h4_writer_coverage_v1'


def read(p):
    return json.loads(p.read_text())


def write(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n')


def main():
    assert all(read(D / 'run' / m / 'report.json')['status'] == 'COMPLETE'
               for m in ['M0', 'M_clean_policy', 'M_inclusive_policy'])
    assert (D / 'VERIFICATION.json').exists()
    snapshot = D / 'superseded_report_snapshots'
    assert not snapshot.exists()
    snapshot.mkdir()
    names = ['LIVE_RESULT.json', 'TRAINING_ELIGIBILITY.json', 'LIVE_CONTENT_DIAGNOSIS.md',
             'TRAINING_ELIGIBILITY.md', 'INDEX.md', 'REPORT.md']
    inventory = []
    for name in names:
        source = OLD / name
        target = snapshot / name
        shutil.copyfile(source, target)
        inventory.append({'path': str(source), 'snapshot': str(target),
                          'original_sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    result = read(OLD / 'LIVE_RESULT.json')
    result['codex_offline_diagnosis'] = {
        'status': 'SUPERSEDES_PRIOR_INCONSISTENT_ELIGIBILITY_DIAGNOSIS',
        'accepted_ordering_candidates': 7, 'duplicate_routed_out_ordering_candidates': 2,
        'routed_ordering_candidates_reviewed': 5, 'routed_original_gold_supported': 4,
        'routed_gold_corrected_before_solver': 1, 'routed_supported_after_correction': 5,
        'routed_hold': 0, 'routed_unusable_after_correction': 0,
        'raw_ordering_artifacts_including_repair': 11,
        'choice_semantic_gate': False, 'choice_evidence_completeness_gate': False,
        'review_is_not_independent_human_certification': True,
        'review_path': str(D / 'prepared/ORDERING_REVIEW.json'),
        'explanation': '语言交流上周末事件先于当前报告，原HOLD理由错误；寿司题首次建议问服务员在T004，修正I2绑定和方向后由程序重算Gold。重复路由不等于语义FAIL。'}
    result['subsequent_local_solver_evaluation'] = str(D / 'RESULTS.json')
    result['superseded_report_snapshot'] = str(snapshot / 'LIVE_RESULT.json')
    write(OLD / 'LIVE_RESULT.json', result)
    candidates = read(D / 'prepared/candidate_manifest.json')
    reviews = {r['candidate_id']: r for r in read(D / 'prepared/ORDERING_REVIEW.json')}
    rows = []
    for c in candidates:
        status = ('ORDERING_SUPPORTED_AFTER_CODEX_GOLD_CORRECTION' if reviews[c['candidate_id']]['status'] == 'CORRECTED_GOLD'
                  else 'ORDERING_ORIGINAL_GOLD_SUPPORTED') if c['question_type'] == 'ordering' else 'CHOICE_PROGRAM_ACCEPTED'
        rows.append({'candidate_id': c['candidate_id'], 'root_id': c['root_id'],
                     'question_type': c['question_type'], 'eligibility': status,
                     'evaluation_gold': c['gold'], 'training_started': False})
    write(OLD / 'TRAINING_ELIGIBILITY.json', {
        'version': 'h4_writer_coverage_eligibility_correction_v2',
        'status': 'SOLVER_EVALUATED_NOT_TRAINED', 'counts': dict(Counter(r['eligibility'] for r in rows)),
        'rules': {'choices': 'User policy: program structure acceptance, no semantic/evidence-completeness gate.',
                  'ordering': 'Codex offline source review: four original Gold supported, one corrected before Solver.',
                  'training': 'No new training performed. Solver evaluation does not establish subsequent training gain.',
                  'duplicates': 'Exact duplicates routed out of fresh pool are not semantic failures or automatic training negatives.'},
        'candidate_manifest': str(D / 'prepared/candidate_manifest.json'),
        'solver_results': str(D / 'RESULTS.json'),
        'superseded_report_snapshot': str(snapshot / 'TRAINING_ELIGIBILITY.json'), 'rows': rows})
    prefix = (f'> **2026-09-25 更正：下文为历史快照，排序准入结论与运行状态已被后续记录取代。**\n> '
              '本轮去重后28题（16单选、7多选、5排序）已完成本地Solver评估。排序为4道原Gold有据、1道修正后有据；'
              '旧“语言交流HOLD”判断撤回，寿司题可修Gold。选择题保持程序接收，不增加人工确认或证据门槛。'
              '重复不等于语义错误，失败题也不自动作为训练负例。\n> '
              f'[最新结果]({D}/RESULTS.md) · [与第一轮比较]({D}/COMPARISON.md) · '
              f'[逐题来源核对]({D}/prepared/ORDERING_REVIEW.json)\n\n---\n\n')
    for name in names:
        if name.endswith('.md'):
            (OLD / name).write_text(prefix + (snapshot / name).read_text())
    for r in inventory:
        r['updated_sha256'] = hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()
    write(D / 'PRIOR_REPORT_CORRECTIONS.json', {'status': 'COMPLETE', 'raw_writer_artifacts_modified': False,
                                             'files': inventory})
    print('Superseded six derived reports; raw Writer artifacts unchanged.')


if __name__ == '__main__':
    main()
