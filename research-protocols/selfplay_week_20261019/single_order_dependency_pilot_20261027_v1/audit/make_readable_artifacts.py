"""Render the frozen question display and result tables; does not alter inputs."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def pct(value):
    return '—' if value is None else f'{value * 100:.2f}%'


def main():
    candidates = read(HERE / 'pilot/CANDIDATES_FINAL.json')
    requests = read(HERE / 'execution/PUBLIC_REQUESTS.json')['requests']
    private = {r['request_id']: r for r in read(HERE / 'execution/PRIVATE_SCORING.json')['requests']}
    preview = ['# 单选与排序题目预览', '',
               '受测题展示第一种实际模型排列并附标准答案；HOLD题单独列稳定ID，未交模型。完整证据、删除组和版本保存在候选JSON；PASS为来源与语义审核结果。', '']
    for candidate in candidates:
        cid = candidate['candidate_id']
        row = next((r for r in requests if r['candidate_id'] == cid and r['condition'] == 'FULL' and r['variant_id'] == 'perm_00'), None)
        if row is None:
            preview += [f'## {cid}（未进入模型测试）', '', candidate['question'], '']
            preview += [f'- {key}. {text}' for key, text in candidate['options'].items()]
            preview += ['', f'作者 Gold（稳定 ID）：{" → ".join(candidate["gold"])}', '',
                        '本题保留用于审计，不计入合格样本。', '',
                        f'[完整来源]({candidate["source_binding"]["path"]})', '']
            continue
        info = private[row['request_id']]
        kind = '单选' if candidate['question_type'] == 'single_choice' else '排序'
        preview += [f'## {cid} · {kind} · {candidate["root_id"]}', '', candidate['question'], '']
        preview += [f'- {label}. {text}' for label, text in row['display'].items()]
        preview += ['', f'Gold：{" → ".join(info["presentation"]["display_gold"])}', '',
                    f'[完整来源]({candidate["source_binding"]["path"]})', '',
                    '作者关系说明：' + candidate['dependency_certificate']['joint_inference'], '']
    (HERE / 'pilot/QUESTIONS_PREVIEW.md').write_text('\n'.join(preview) + '\n')
    if not (HERE / 'execution/ANALYSIS_SUMMARY.json').exists():
        print(json.dumps({'preview': str(HERE / 'pilot/QUESTIONS_PREVIEW.md'), 'results': 'pending'}, ensure_ascii=False))
        return
    summary = read(HERE / 'execution/ANALYSIS_SUMMARY.json')
    tables = ['# 实际模型结果表', '',
              '仅按最终答案评分，引用合规另列。每题三个排列是重复测量。缺失上下文条件沿用 Full Gold，仅表示原答案保留。', '',
              '| 题型 | 条件 | 完全正确 | 排序 pairwise | 排序 LCS/n | 有效答案 |',
              '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in summary['summaries']:
        tables.append(f'| {row["question_type"]} | {row["condition"]} | {row["exact_count"]}/{row["requests"]} ({pct(row["exact_accuracy"])}) | {pct(row["pairwise_accuracy"])} | {pct(row["normalized_lcs"])} | {row["answer_valid"]}/{row["requests"]} |')
    tables += ['', '## 逐题结果', '',
               '| 题目 | 条件 | 完全正确 | pairwise | LCS/n | 目标判断保留 | 三排列同答 |',
               '| --- | --- | ---: | ---: | ---: | ---: | --- |']
    for row in summary['per_item']:
        tables.append(f'| {row["candidate_id"]} | {row["condition"]} | {row["exact_count"]}/{row["requests"]} | {pct(row["pairwise_accuracy"])} | {pct(row["normalized_lcs"])} | {pct(row["full_gold_target_retained"])} | {row["permutation_same_prediction"]} |')
    (HERE / 'audit/RESULT_TABLES_ZH.md').write_text('\n'.join(tables) + '\n')
    print(json.dumps({'preview': str(HERE / 'pilot/QUESTIONS_PREVIEW.md'), 'tables': str(HERE / 'audit/RESULT_TABLES_ZH.md')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
