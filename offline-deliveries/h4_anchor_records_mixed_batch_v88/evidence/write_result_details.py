"""Summarize actual mixed outcomes without assuming any generation success."""
from collections import Counter
from fractions import Fraction
from pathlib import Path
import sys
HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOME))
from common import read, write, bind, now

pool = read(HOME / 'EVALUATION_POOL.json')['questions']
byid = {q['candidate_id']: q for q in pool}
report = read(HOME / 'MIXED_EVALUATION_REPORT.json')
pairs = read(HOME / 'PAIRED_CALLS.json')['calls']
newids = {q['candidate_id'] for q in pool if q['question_type'] == 'ordering' or len(q['gold']) < 6}
selected = set(report['selected_candidate_ids'])

def metric(rows):
    n = len(rows); correct = sum(r['exact'] for r in rows)
    return {'calls': n, 'exact_correct': correct,
            'exact_fraction': str(Fraction(correct, n)) if n else None,
            'exact_percent': 100 * correct / n if n else None,
            'partial_mean': str(sum((Fraction(r['partial']) for r in rows), Fraction()) / n) if n else None}

def accuracy(rows):
    m = metric(rows)
    return {'correct': m['exact_correct'], 'calls': m['calls'],
            'accuracy_fraction': m['exact_fraction'],
            'accuracy_percent': round(m['exact_percent'], 4) if m['calls'] else None,
            'all_six_selected_calls': sum(r.get('question_type') == 'multi_select' and r.get('selected_count') == 6 for r in rows),
            'all_six_selected_correct': sum(r.get('question_type') == 'multi_select' and r.get('selected_count') == 6 and r['exact'] for r in rows)}

def keep(pair, role):
    q = byid[pair['candidate_id']]; is_new = pair['candidate_id'] in newids
    return (role == 'all_questions' or
            role in ('new_child', 'new_candidates') and is_new or
            role == 'new_multiselect' and is_new and q['question_type'] == 'multi_select' or
            role == 'new_ordering' and is_new and q['question_type'] == 'ordering' or
            role == 'historical_control' and not is_new)

roles = ('new_child', 'new_candidates', 'new_multiselect', 'new_ordering', 'historical_control', 'all_questions')
metrics = {side: {role: metric([r[side] for r in pairs if keep(r, role)]) for role in roles}
           for side in ('trained', 'base')}
questions = []; perquestion = []
results = {side: {q['candidate_id']: q for q in read(HOME / 'solver' / side / 'RESULTS.json')['questions']}
           for side in ('trained', 'base')}
for q in pool:
    role = ('new_ordering' if q['question_type'] == 'ordering' else 'new_child') if q['candidate_id'] in newids else 'historical_control'
    identity = {'candidate_id': q['candidate_id'], 'root_id': q['root_id'], 'question_type': q['question_type'],
                'gold_count': len(q['gold']), 'role': role}
    rows = [r for r in pairs if r['candidate_id'] == q['candidate_id']]
    perquestion.append({**identity, 'selected': q['candidate_id'] in selected,
                        'models': {side: accuracy([r[side] for r in rows]) for side in ('base', 'trained')}})
    questions.append({**identity, 'models': {side: {k: results[side][q['candidate_id']][k]
                      for k in ('full_exact_rate', 'full_partial_mean', 'selected_by_policy', 'selection_reason')}
                      for side in ('base', 'trained')}})
aggregate = {label: {side: accuracy([r[side] for r in pairs if keep(r, role)]) for side in ('base', 'trained')}
             for label, role in [('new_children', 'new_candidates'), ('historical_controls', 'historical_control'), ('all', 'all_questions')]}
ordering_failures = []
for path in sorted((HOME / 'ordering').glob('*/REJECTIONS.json')):
    for row in read(path):
        ordering_failures.append({**row, 'record': bind(path)})
ordering_review = read(HOME / 'ordering/CONTENT_REVIEW.json')
ordering_content_failures = [{'candidate_id': q['candidate_id'], 'root_id': q['root_id'], 'reasons': q['reasons']}
                             for q in ordering_review['questions'] if q['status'] != 'USABLE']
multiselect_review = read(HOME / 'multiselect/FINAL_CONTENT_REVIEW.json')
multiselect_content_failures = [{'candidate_id': q['candidate_id'], 'root_id': q['root_id'],
                                'core_status': q['core_status'], 'reason': q['reason']}
                               for q in multiselect_review['questions'] if q['core_status'] != 'CORE_USABLE']
rounds = []
for path in sorted((HOME / 'multiselect/assets').glob('*/rounds/*/ASSET_REVIEW.json')):
    review = read(path)
    rounds.append({'family_id': review['family_id'], 'round_id': review['round_id'],
                   'usable_parents': sum(v['usable'] for v in review['parents']),
                   'usable_contradictions': sum(v['usable'] for v in review['versions']),
                   'rejected_versions': [v for v in review['versions'] if not v['usable']], 'record': bind(path)})
allocation = read(HOME / 'multiselect/ALLOCATION.json')
prior_comparisons = {}
for version, dirname in [('v81', 'h4_bound_scene_ordering_mixed_batch_v81'),
                         ('v82', 'h4_actor_reference_minfix_ordering_v82'),
                         ('v83', 'h4_name_policy_mixed_batch_v83'),
                         ('v84', 'h4_independent_roots_mixed_batch_v84'),
                         ('v85', 'h4_label_anchor_minfix_mixed_batch_v85'),
                         ('v86', 'h4_reason_before_gold_mixed_batch_v86'),
                          ('v87', 'h4_program_gold_mixed_batch_v87')]:
    oldhome = HOME.parent / dirname
    oldpool = {q['candidate_id']: q for q in read(oldhome / 'EVALUATION_POOL.json')['questions']}
    oldpairs = read(oldhome / 'PAIRED_CALLS.json')['calls']
    typed = {}
    for kind in ('multi_select', 'ordering'):
        ids = {qid for qid, q in oldpool.items() if q['question_type'] == kind and (kind == 'ordering' or len(q['gold']) < 6)}
        typed[kind] = {'question_count': len(ids), 'models': {side: metric([r[side] for r in oldpairs if r['candidate_id'] in ids])
                                                              for side in ('base', 'trained')}}
    prior_comparisons[version] = {'report': bind(oldhome / 'MIXED_EVALUATION_REPORT.json'),
                                  'paired_calls': bind(oldhome / 'PAIRED_CALLS.json'), 'new_questions_by_type': typed}
config = read(HOME / 'CONFIG.json')
source_root_ids = sorted({slot['root_id'] for slot in config['ordering_slots']})
details = {'created_at_utc': now(), 'status': 'ACTUAL_BATCH_RESULTS', 'report': bind(HOME / 'MIXED_EVALUATION_REPORT.json'),
           'source_root_ids': source_root_ids, 'source_root_count': len(source_root_ids),
           'source_split': 'known_historical_train_roots', 'independent_test_sources': False,
           'content_usable_new_count': len(newids), 'new_selected_count': len(newids & selected),
           'content_usable_new_by_type': dict(Counter(q['question_type'] for q in pool if q['candidate_id'] in newids)),
           'selected_new_by_type': dict(Counter(q['question_type'] for q in pool if q['candidate_id'] in newids & selected)),
           'content_usable_new_by_multi_gold': {str(g): sum(q['candidate_id'] in newids and q['question_type'] == 'multi_select' and len(q['gold']) == g for q in pool) for g in range(2, 6)},
           'selected_new_by_multi_gold': {str(g): sum(q['candidate_id'] in newids & selected and q['question_type'] == 'multi_select' and len(q['gold']) == g for q in pool) for g in range(2, 6)},
           'aggregate': aggregate, 'per_question': perquestion, 'metrics': metrics, 'questions': questions,
           'metric_role_definitions': {'new_child': 'Compatibility alias of new_candidates: all non-control questions, including ordering.',
                                       'new_candidates': 'All non-control multi_select and ordering questions.',
                                       'new_multiselect': 'Only non-control multi_select questions.',
                                       'new_ordering': 'Only ordering questions.'},
           'ordering_generation_failures': ordering_failures, 'ordering_content_failures': ordering_content_failures,
           'ordering_content_review': bind(HOME / 'ordering/CONTENT_REVIEW.json'),
           'multiselect_content_failures': multiselect_content_failures,
           'multiselect_content_review': bind(HOME / 'multiselect/FINAL_CONTENT_REVIEW.json'),
           'multiselect_asset_round_reviews': rounds, 'allocation': bind(HOME / 'multiselect/ALLOCATION.json'),
           'multiselect_unfilled_slots': [{'family_id': f['family_id'], 'root_id': f['root_id'], **shortfall}
                                          for f in allocation['families'] for shortfall in f['shortfalls']],
           'prior_comparisons': prior_comparisons,
           'selection_rule': '0 < trained mean partial < 1 with all responses valid; exact accuracy is descriptive and does not select questions.',
           'limitations': ['Current and prior batches contain different question mixtures; between-batch accuracy is not a controlled difficulty estimate.',
                          'v82 is a targeted repair sample, not fresh generation; comparisons retain this distinction.',
                          'Within-batch base/trained use identical prompts, permutations, seeds, decoder and scoring.',
                          'Actual source-review disclosures are bound in the freeze; no claim of fresh blind historical parents.',
                          'Ordering conclusions are limited to the actual generated, reviewed and evaluated candidate counts.',
                          f'Whole-question text novelty is not semantic novelty or independent new facts; this batch uses {len(source_root_ids)} different historical TRAIN roots: {", ".join(source_root_ids)}.',
                          'These sources are the same four historical TRAIN roots as v87; only current questions differ. They are not independent fixed test sources.'],
           'training_allowed': False, 'formal_training_admission': False}
write(HOME / 'evidence/RESULT_DETAILS.json', details)
print({'new_content': len(newids), 'new_selected': len(newids & selected), 'by_type': details['selected_new_by_type'], 'metrics': metrics})
