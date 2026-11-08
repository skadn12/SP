#!/usr/bin/env python3
"""Render a small index from the independently verified fixed-DEV transitions."""
import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
data = json.loads((root / 'evidence/SECOND_ROUND_TRANSITIONS.json').read_text())
rows = data['dev_v127']['questions']
cols = ['candidate_id', 'root_id', 'question_type', 'gold_count', 'before_exact_correct',
        'after_exact_correct', 'planned_calls', 'transition', 'readable_path']
with (root / 'evidence/DEV_TRANSITIONS.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=cols)
    writer.writeheader()
    for row in rows:
        out = {k: row[k] for k in cols if k != 'readable_path'}
        out['readable_path'] = 'data/dev/readable/' + row['root_id'] + '.md'
        writer.writerow(out)
zero_full = [r for r in rows if r['transition'] == 'zero_to_full']
assert len(zero_full) == 12 and all(r['question_type'] == 'ordering' for r in zero_full)
lines = ['# DEV 从本批全错到本批全对的 12 道题', '',
         '对照为第二轮训练前 M_clean 与第二轮后 M_v125_newonly。这些是 DEV 题，不是 363 道实际入训题。', '',
         '每题均为 0/12 → 12/12；仅指计划的 12 次随机显示调用，不表示所有可能排列或未来调用均正确。12 题来自 7 个 root，不能当成 12 个独立故事。', '',
         '| 题目 ID | root 完整对话与题目 | 训练前 | 训练后 |',
         '|---|---|---:|---:|']
for r in zero_full:
    lines.append('| `' + r['candidate_id'] + '` | [' + r['root_id'] + '](../data/dev/readable/' + r['root_id'] + '.md) | 0/12 | 12/12 |')
lines += ['', '全部 48 道 DEV 的转移表见 [DEV_TRANSITIONS.csv](DEV_TRANSITIONS.csv)，完整逐调用依据见 [SECOND_ROUND_TRANSITIONS.json](SECOND_ROUND_TRANSITIONS.json)。多选没有零成功到非零成功的例子；有 27 道从部分调用正确变为本批全部正确，其中 21 道为 Gold2–5、6 道为 Gold6 控制题。', '']
(root / 'evidence/DEV_ZERO_TO_FULL_QUESTIONS.md').write_text('\n'.join(lines))
print(json.dumps({'status': 'PASS', 'indexed_dev_questions': len(rows), 'zero_to_full_questions': len(zero_full)}))
