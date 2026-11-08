"""Freeze actual content counts after assembly and before content sealing/Solver."""
from collections import Counter
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
accounting = json.loads((HERE / 'FINAL_BATCH_ACCOUNTING.json').read_text())
questions = []
for name in ('NEW_PUBLIC_QUESTIONS.json', 'CONTROL_PUBLIC_QUESTIONS.json'):
    questions.extend(json.loads((HERE / name).read_text())['questions'])
private = []
for name in ('NEW_PRIVATE_GOLD.json', 'CONTROL_PRIVATE_GOLD.json'):
    private.extend(json.loads((HERE / name).read_text())['questions'])
gold = {q['candidate_id']: q['gold'] for q in private}
passed = accounting['new_questions'] + accounting['gold6_controls']
assert len(gold) == len(private) == len(questions) == passed
assert accounting['public_candidates'] == passed + accounting['excluded_final_candidates']
assert {q['candidate_id'] for q in questions} == set(gold)
types = Counter(q['question_type'] for q in questions)
counts = {'candidate_count': len(questions),
    'type_counts': {k: types.get(k, 0) for k in ('multi_select', 'ordering')},
    'multi_gold_counts': dict(sorted(Counter(str(len(gold[q['candidate_id']]))
        for q in questions if q['question_type'] == 'multi_select').items())),
    'calls_per_model': sum(18 if q['question_type'] == 'multi_select' else 12 for q in questions)}
plan = json.loads((HERE / 'PRECALL_PLAN.json').read_text())
assert 0 < counts['candidate_count'] <= plan['planned_new_questions_max'] + plan['planned_controls_max']
assert counts['calls_per_model'] <= plan['solver_calls_per_model_max']
assert len(accounting['roots']) == 12
with (HERE / 'EVALUATION_COUNTS.json').open('x', encoding='utf-8') as f:
    json.dump({'status': 'FIXED_AFTER_CONTENT_BEFORE_SOLVER', 'evaluation_counts': counts,
        'accounting_sha256': hashlib.sha256((HERE / 'FINAL_BATCH_ACCOUNTING.json').read_bytes()).hexdigest(),
        'precall_plan_sha256': hashlib.sha256((HERE / 'PRECALL_PLAN.json').read_bytes()).hexdigest(),
        'public_candidates_including_excluded': accounting['public_candidates'],
        'excluded_final_candidates': accounting['excluded_final_candidates'],
        'training_started': False}, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps(counts, ensure_ascii=False))
