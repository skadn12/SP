#!/usr/bin/env python3
"""Mechanically preserve reviewer responses and render frozen question variants."""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
SESSIONS = Path('/root/.codex/sessions/2026/11/06')
REVIEWERS = {
    'V1_REVIEWER_A': 'rollout-2026-11-06T00-01-08-01a1ac71-4b49-7090-b70a-96be4001c813.jsonl',
    'V1_REVIEWER_B': 'rollout-2026-11-06T00-01-54-01a1ac71-fe64-7f82-92c2-22f9a2f5cdc0.jsonl',
    'V2_REVIEWER_A': 'rollout-2026-11-06T00-11-39-01a1ac7a-eadb-7801-ae37-27973a35d1d0.jsonl',
    'V2_REVIEWER_B': 'rollout-2026-11-06T00-11-50-01a1ac7b-18ee-73a1-9962-ae5b397b8be8.jsonl',
}


def write_new(name, text):
    with (OUT / name).open('x', encoding='utf-8') as f:
        f.write(text)


def response(path):
    answers = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            p = record.get('payload', {})
            if (record.get('type') == 'response_item' and p.get('type') == 'message'
                    and p.get('role') == 'assistant' and p.get('phase') == 'final_answer'):
                answers.append('\n'.join(x.get('text', '') for x in p.get('content', [])))
    assert len(answers) == 1, (path, len(answers))
    return answers[0]


raw = {key: {'session_path': str(SESSIONS / name), 'final_response': response(SESSIONS / name)}
       for key, name in REVIEWERS.items()}
write_new('READABILITY_REVIEW_RAW.json', json.dumps(raw, ensure_ascii=False, indent=2) + '\n')

v1 = json.loads((OUT / 'QUESTION_DIFF.json').read_text(encoding='utf-8'))
v2 = {x['candidate_id']: x for x in json.loads((OUT / 'QUESTION_DIFF_V2.json').read_text(encoding='utf-8'))}
lines = ['# 题面试改对照', '',
         '以下为冻结的原题与派生试稿，选项、答案和证据没有改动；试稿不是训练数据。', '']
for item in v1:
    lines.extend([f"## {item['candidate_id']} ({item['question_type']})", '',
                  f"原题：{item['original_question']}", '', f"V1：{item['derived_question']}", ''])
    if item['candidate_id'] in v2:
        lines.extend([f"V2：{v2[item['candidate_id']]['v2_derived_question']}", ''])
write_new('QUESTION_COMPARISON_ZH.md', '\n'.join(lines))
