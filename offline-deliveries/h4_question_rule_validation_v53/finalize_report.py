"""Read frozen live artifacts and export a reviewable v53 validation summary."""
from collections import Counter
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CODE = HERE.parent / 'h4_sequential_ab_v53_qa'
sys.path.insert(0, str(CODE))
import display_export

def read(path):
    return json.loads(path.read_text())

def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')

assert (HERE / 'evaluation.COMPLETE.json').exists()
rows, mechanical, selected, raw_rows = [], [], [], []
for short in ('135466', '210046'):
    run = CODE / 'runs' / f'root{short}_rules4_01_generation'
    evaluation = CODE / 'runs' / f'root{short}_rules4_01_evaluation'
    report = read(run / 'REPORT.json')
    pool_report = read(evaluation / 'training_pool/REPORT.json')
    drafts = []
    for request in sorted((run / 'requests').glob('*')):
        proof = read(request / 'PROOF.json')
        if proof['role'] != 'question_generator' or not (request / 'RESPONSE.json').exists():
            continue
        raw = read(request / 'RESPONSE.json')
        parsed = json.loads(raw['text'])
        for q in parsed['questions']:
            row = {'root': short, 'candidate_id': q['candidate_id'], 'question_type': proof['question_type'],
                   'repair': request.name.endswith('-2'), 'raw_response_path': str(request / 'RESPONSE.json'),
                   'raw_question': q}
            if proof['question_type'] == 'multi_select':
                row['supports_count'] = sum(x['relation'] == 'supports' for x in q['options'])
            drafts.append(row)
    raw_rows += drafts
    initial = [q for q in drafts if not q['repair']]
    kinds = {}
    for outcome in report['roots']:
        kind = outcome['question_type']
        rejected = outcome['mechanical_rejected']
        kinds[kind] = {'slots': outcome['candidate_slots'], 'mechanical_pass': outcome['mechanical_pass'],
                       'mechanical_rejected': rejected, 'abstentions': outcome['abstentions'],
                       'review_rejected': outcome['review_rejected'], 'ready': outcome['ready_for_solver']}
    qpool = read(run / 'MECHANICAL_POOL.json')['questions']
    mechanical += qpool
    chosen = [json.loads(line)['original_selected_question']
              for line in (evaluation / 'training_pool/TRAINING_POOL.jsonl').read_text().splitlines() if line.strip()]
    selected += chosen
    rows.append({'root': short, 'slots': report['candidate_slots'], 'initial_drafts': len(initial),
                 'repair_drafts': len(drafts) - len(initial), 'http_attempts': report['http_attempts'],
                 'mechanical_pass': report['mechanical_pass'], 'ready': report['ready_for_solver'],
                 'solver_calls': pool_report['full_solver_calls'], 'selected': pool_report['new_unique_added'],
                 'selected_by_type': pool_report['by_type'], 'by_type': kinds})

all_views = [display_export.display_view(q) for q in mechanical]
selected_views = [display_export.display_view(q) for q in selected]
for q, view in zip(mechanical, all_views):
    display_export.replay_view(q, view)
for q, view in zip(selected, selected_views):
    display_export.replay_view(q, view)
summary = {'status': 'LIVE_RULE_VALIDATION_COMPLETE', 'roots': rows,
           'totals': {key: sum(row[key] for row in rows)
                      for key in ('slots', 'initial_drafts', 'repair_drafts', 'http_attempts', 'mechanical_pass',
                                  'ready', 'solver_calls', 'selected')},
           'raw_initial_mc_support_count_histogram': dict(Counter(
               row['supports_count'] for row in raw_rows if row['question_type'] == 'multi_select' and not row['repair'])),
           'mechanical_display': display_export.display_stats(all_views),
           'selected_display': display_export.display_stats(selected_views),
           'machine_selection_is_not_semantic_certification': True,
           'four_slots_are_validation_size_not_final_per_root_quota': True,
           'training_calls': 0, 'training_allowed': False}
write(HERE / 'RESULT_SUMMARY.json', summary)
write(HERE / 'RAW_DRAFTS.json', raw_rows)
write(HERE / 'ALL_MECHANICAL_CANDIDATES.json', mechanical)
for filename, views in (('ALL_MECHANICAL_QUESTIONS.md', all_views), ('SELECTED_QUESTIONS.md', selected_views)):
    title = ('v53 机械通过题：包含独立内容检查发现问题的题目' if filename.startswith('ALL')
             else 'v53 机器筛选题：须结合独立内容检查阅读')
    with (HERE / filename).open('x') as stream:
        stream.write(display_export.markdown(views, title=title))
print(json.dumps({'totals': summary['totals'], 'selected_display': summary['selected_display']}, ensure_ascii=False))
