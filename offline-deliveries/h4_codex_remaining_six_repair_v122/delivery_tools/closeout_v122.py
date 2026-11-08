"""Create the six-root report and then an exclusive complete inventory."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
from state_v122 import B, bind, read


def save(path, value):
    with Path(path).open('x') as stream: json.dump(value, stream, ensure_ascii=False, indent=2); stream.write('\n')


def report():
    audit = read(B / 'delivery/POST_RUN_INTEGRITY_AUDIT_V1.json')
    if audit['pass'] is not True: raise ValueError('INTEGRITY_AUDIT_NOT_PASS')
    scope = read(B / 'REPAIR_SCOPE.json'); progress = read(B / 'delivery/PROGRESS_FINAL_V1.json')
    candidates = read(B / 'delivery/TRAINING_CANDIDATES_V1.json'); lengths = read(B / 'delivery/DIALOGUE_LENGTHS_ALL_ROOTS_V1.json')
    disposition = read(B / 'delivery/QUESTION_DISPOSITION_V1.json'); rows = audit['roots']
    complete = [r for r in rows if r.get('scene_state', {}).get('full_expansion_complete')]
    content = [r for r in rows if r['status'] == 'FINALIZED' and r.get('content_pass', 0)]
    per_root = Counter(q['root_id'] for q in candidates['questions'])
    lines = [f'本批固定补救剩余6个失败 root；此前历史28个已救回{scope["previously_recovered_count"]}个。', '',
             f'本批{len(complete)}个完成审核通过的六场对话，{len(content)}个产出内容合格题；历史累计救回{scope["previously_recovered_count"] + len(content)}个。', '',
             f'新增{progress["new_scenes_generated"]}场；另有{progress["carried_scene_reaudit_count"]}场使用历史原文重新审核。内容通过{progress["content_pass_questions"]}题，其中排序{progress["content_pass_ordering"]}题，多选Gold数量为{json.dumps(progress["content_pass_ms_gold_counts"], ensure_ascii=False)}。', '',
             f'Solver筛选后保留{candidates["candidate_count"]}道候选，来自{candidates["root_count"]}个root；Gold2–5分布为{json.dumps(candidates["multi_select_gold_counts"], ensure_ascii=False)}。Gold6仅为对照。尚未合并训练池或训练。', '',
             '| root | 来源 | 终态 | 内容通过 | 训练候选 | 记录 |', '|---|---|---|---:|---:|---|']
    for row in rows:
        short = row['root_id'].rsplit('-', 1)[-1]; path = B / 'delivery' / f'root{short}_DIALOGUE_AND_QUESTIONS.md'
        if not path.exists(): raise ValueError('READABLE_EXPORT_REQUIRED:' + short)
        lines.append(f'| {short} | {row["source_status"]} | {row["status"]} | {row.get("content_pass", 0)} | {per_root[row["root_id"]]} | [完整记录]({path}) |')
    mean = lengths['arithmetic_mean']
    if mean['full_tokens'] is not None:
        lines += ['', f'完整六场对话的平均累计长度为{mean["full_tokens"]:.1f} tokens、{mean["full_turns"]:.1f}条发言；六场扩充部分平均{mean["six_scope_tokens"]:.1f} tokens。使用本地Qwen3-4B tokenizer，仅统计speaker: text正文，非生成计费用量。']
    lines += ['', '原文未改与合成修订分别保留来源标签。历史作者、审核者和时间均保留；固定历史候选重审单列，不计作新生成。全部失败稿及本批终态见逐root记录与完整性审计。', '',
              '两模型使用相同冻结题目、排列、seed和原生解码。Gold2–5、Gold6、排序分别统计；重复排列、rollout及同root题族不能当作独立样本。所有同root原文、修订、扩充与题目必须保留在同一数据划分。', '',
              f'[两模型评估]({B / "delivery/PAIRED_SOLVER_REPORT_ZH_V1.md"}) · [候选]({B / "delivery/TRAINING_CANDIDATES_V1.json"}) · [全部题目处置]({B / "delivery/QUESTION_DISPOSITION_V1.json"}) · [完整性审计]({B / "delivery/POST_RUN_INTEGRITY_AUDIT_V1.json"}) · [审核质量记录]({B / "delivery/QUALITY_FINDINGS_V1.json"})', '',
              f'候选去向：{json.dumps(disposition["counts"], ensure_ascii=False)}。审核记录来自共享文件系统中的不同实际执行代理，不宣称外部独立认证或可证明的盲审。', '']
    path = B / 'delivery/BATCH_CLOSEOUT_ZH_V1.md'
    with path.open('x') as stream: stream.write('\n'.join(lines))
    save(B / 'delivery/CLOSEOUT_TOTALS_V1.json', {'created_at_utc': datetime.now(timezone.utc).isoformat(),
         'fixed_candidates': 6, 'previously_recovered_count': scope['previously_recovered_count'],
         'cumulative_recovered_content_roots': scope['previously_recovered_count'] + len(content),
         'terminal_statuses': audit['terminal_statuses'], 'complete_six_scene_roots': len(complete),
         'content_pass_roots': len(content), 'new_scenes_generated': progress['new_scenes_generated'],
         'carried_scene_reaudit_count': progress['carried_scene_reaudit_count'],
         'content_pass_questions': progress['content_pass_questions'], 'content_pass_ordering': progress['content_pass_ordering'],
         'content_pass_ms_gold_counts': progress['content_pass_ms_gold_counts'],
         'training_candidate_count': candidates['candidate_count'], 'training_candidate_roots': candidates['root_count'],
         'training_candidate_types': candidates['question_type_counts'], 'training_candidate_ms_gold_counts': candidates['multi_select_gold_counts'],
         'training_or_pool_merge': False, 'report_binding': bind(path)})
    print(str(path))


def seal():
    totals = read(B / 'delivery/CLOSEOUT_TOTALS_V1.json')
    if read(B / 'delivery/POST_RUN_INTEGRITY_AUDIT_V1.json')['pass'] is not True: raise ValueError('INTEGRITY_AUDIT_NOT_PASS')
    target = B / 'COMPLETE_MANIFEST.json'
    if target.exists(): raise ValueError('BATCH_ALREADY_SEALED')
    files = [bind(p) for p in sorted(B.rglob('*')) if p.is_file() and p != target]
    save(target, {'status': 'COMPLETE_V122_FIXED_REMAINING_SIX_REPAIR_AND_PAIRED_EVALUATION',
         'created_at_utc': datetime.now(timezone.utc).isoformat(), 'actual_executor': '/root', 'files': files,
         'excludes_only': [str(target)], 'totals': totals, 'closeout': bind(B / 'delivery/BATCH_CLOSEOUT_ZH_V1.md'),
         'training_or_pool_merge': False})
    print(json.dumps({'status': 'BATCH_SEALED', 'files': len(files), 'totals': totals}, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['report', 'seal']); args = p.parse_args(); globals()[args.action]()
