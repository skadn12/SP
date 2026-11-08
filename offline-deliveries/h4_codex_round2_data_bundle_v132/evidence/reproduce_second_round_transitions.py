"""Read existing raw responses; quantify second-round exact-success transitions.

No inference, training, question selection or writes to predecessor packages.
zero/partial/full describe frequency of fully correct answers across all fixed
calls. They do not describe within-answer Jaccard/pair/signed partial credit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DELIVERIES = REPO / 'offline-deliveries'
V127 = DELIVERIES / 'h4_codex_dev_eval_v127'
V131 = DELIVERIES / 'h4_codex_historical_frontier_eval_v131'
V125 = DELIVERIES / 'h4_codex_train_pool_v125'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state(correct, total):
    assert total > 0 and 0 <= correct <= total
    return 'zero' if correct == 0 else 'full' if correct == total else 'partial'


def independent_exact(raw, mapping, gold, qtype):
    match = re.fullmatch(r'\s*<box>\s*(.*?)\s*</box>\s*', raw, re.S)
    assert match is not None
    labels = [label.strip().upper() for label in match.group(1).split(',')]
    assert labels and len(labels) == len(set(labels)) and all(label in mapping for label in labels)
    prediction = [mapping[label] for label in labels]
    assert len(prediction) == len(set(prediction))
    if qtype == 'ordering': assert len(prediction) == len(gold) and set(prediction) == set(gold)
    if qtype == 'single_choice': assert len(prediction) == 1
    return int(set(prediction) == set(gold)) if qtype == 'multi_select' else int(prediction == gold)


def aggregate(rows):
    transitions = Counter(row['transition'] for row in rows)
    zero = [r for r in rows if r['before_state'] == 'zero']
    ids = lambda subset: [r['candidate_id'] for r in subset]
    return {
        'question_count': len(rows), 'root_count': len({r['root_id'] for r in rows}),
        'planned_calls_each_model': sum(r['planned_calls'] for r in rows),
        'transition_counts': dict(sorted(transitions.items())),
        'before_zero_count': len(zero),
        'before_zero_ids': ids(zero),
        'zero_to_nonzero_count': sum(r['after_exact_correct'] > 0 for r in zero),
        'zero_to_nonzero_ids': ids([r for r in zero if r['after_exact_correct'] > 0]),
        'zero_to_partial_count': sum(r['after_state'] == 'partial' for r in zero),
        'zero_to_partial_ids': ids([r for r in zero if r['after_state'] == 'partial']),
        'zero_to_full_count': sum(r['after_state'] == 'full' for r in zero),
        'zero_to_full_ids': ids([r for r in zero if r['after_state'] == 'full']),
        'new_full_from_not_full_count': sum(r['before_state'] != 'full' and r['after_state'] == 'full' for r in rows),
        'new_full_from_not_full_ids': ids([r for r in rows if r['before_state'] != 'full' and r['after_state'] == 'full']),
        'nonzero_to_zero_count': sum(r['before_state'] != 'zero' and r['after_state'] == 'zero' for r in rows),
        'nonzero_to_zero_ids': ids([r for r in rows if r['before_state'] != 'zero' and r['after_state'] == 'zero']),
        'before_full_count': sum(r['before_state'] == 'full' for r in rows),
        'after_full_count': sum(r['after_state'] == 'full' for r in rows),
        'before_any_exact_success_count': sum(r['before_exact_correct'] > 0 for r in rows),
        'after_any_exact_success_count': sum(r['after_exact_correct'] > 0 for r in rows),
    }


def strata(rows):
    groups = {'all': rows,
              **{qtype: [r for r in rows if r['question_type'] == qtype] for qtype in ['multi_select', 'ordering', 'single_choice']},
              'multi_select_gold2345': [r for r in rows if r['question_type'] == 'multi_select' and r['gold_count'] in [2, 3, 4, 5]],
              'multi_select_gold6_control': [r for r in rows if r['question_type'] == 'multi_select' and r['gold_count'] == 6]}
    groups.update({'multi_select_gold' + str(g): [r for r in rows if r['question_type'] == 'multi_select' and r['gold_count'] == g] for g in range(2, 7)})
    return {name: aggregate(members) for name, members in groups.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'SECOND_ROUND_TRANSITIONS.json')
    args = parser.parse_args()
    assert not args.output.exists(), 'OUTPUT_EXISTS_NO_SILENT_OVERWRITE'
    bindings = {}
    def read(path):
        path = Path(path).resolve(); bindings[str(path)] = sha(path)
        return json.loads(path.read_text())
    bindings[str(Path(__file__).resolve())] = sha(__file__)
    dev_before = read(V127 / 'run/baseline/manifest.json')
    dev_after = read(V127 / 'run/new/manifest.json')
    assert dev_before['requests'] == dev_after['requests'] and len(dev_before['requests']) == 780
    assert dev_before['solver_model'] == 'M_clean_policy' and dev_after['solver_model'] == 'M_v125_newonly'
    canonical = read(V127 / 'EVALUATION_POOL.json')['questions']
    questions = {q['candidate_id']: q for q in canonical}
    gold = read(V127 / 'run/baseline/gold.json')
    assert gold == read(V127 / 'run/new/gold.json')
    dev_published = {r['candidate_id']: r for r in read(V127 / 'PER_QUESTION.json')}
    dev_calls = {cid: {'before': [], 'after': []} for cid in questions}
    for request in dev_before['requests']:
        cid, rid = request['candidate_id'], request['request_id']
        for side, label in [('baseline', 'before'), ('new', 'after')]:
            path = V127 / 'run' / side / 'responses' / (rid + '.json')
            response = read(path)
            assert response['request_id'] == rid and response['evaluation_valid'] is True and not response['data_errors']
            correct = independent_exact(response['raw_output'], request['variant']['display_to_content'], gold[cid], request['question_type'])
            assert correct == int(response['score']['exact_match'])
            dev_calls[cid][label].append({'request_id': rid, 'correct': correct, 'response_path': str(path)})
    dev_rows = []
    for cid, q in questions.items():
        calls = dev_calls[cid]; n = 18 if q['question_type'] == 'multi_select' else 12
        assert len(calls['before']) == len(calls['after']) == n
        a, b = sum(r['correct'] for r in calls['before']), sum(r['correct'] for r in calls['after'])
        assert a == dev_published[cid]['baseline']['exact_correct'] and b == dev_published[cid]['new']['exact_correct']
        dev_rows.append({'candidate_id': cid, 'root_id': q['root_id'], 'question_type': q['question_type'],
                         'gold_count': len(q['gold']), 'planned_calls': n, 'before_exact_correct': a, 'after_exact_correct': b,
                         'before_state': state(a, n), 'after_state': state(b, n), 'transition': state(a, n) + '_to_' + state(b, n),
                         'question_source_path': str(V127 / 'EVALUATION_POOL.json'),
                         'original_dialogue_source_path': q['source_path'], 'calls': calls})
    frontier = read(V131 / 'manifest.json')
    assert len(frontier['requests']) == 1629
    frontier_questions = {q['candidate_id']: q for q in read(frontier['questions_path'])['candidates']}
    frontier_published = {r['candidate_id']: r for r in read(V131 / 'PER_QUESTION.json')}
    frontier_calls = {cid: {'before': [], 'after': []} for cid in frontier_questions}
    for request in frontier['requests']:
        cid, rid = request['candidate_id'], request['request_id']
        for side, label in [('live_clean', 'before'), ('live_new', 'after')]:
            path = V131 / 'responses' / side / (hashlib.sha256(rid.encode()).hexdigest() + '.json')
            response = read(path)
            assert response['request_id'] == rid and response['evaluation_valid'] is True
            assert response['solver_model'] == ('M_clean_policy' if label == 'before' else 'M_v125_newonly')
            correct = independent_exact(response['raw_output'], request['display_to_canonical'], request['canonical_gold'], request['question_type'])
            assert correct == int(response['score']['correct'])
            frontier_calls[cid][label].append({'request_id': rid, 'correct': correct, 'response_path': str(path)})
    frontier_rows = []
    for cid, q in frontier_questions.items():
        calls = frontier_calls[cid]; n = 15 if q['question_type'] == 'multi_select' else 12
        assert len(calls['before']) == len(calls['after']) == n
        a, b = sum(r['correct'] for r in calls['before']), sum(r['correct'] for r in calls['after'])
        published = frontier_published[cid]
        assert a == published['models']['live_clean']['exact_correct'] and b == published['models']['live_new']['exact_correct']
        historical_a = published['models']['historical_clean']['exact_correct']
        assert 0 < historical_a < n
        frontier_rows.append({'candidate_id': cid, 'root_id': q['root_id'], 'question_type': q['question_type'],
                              'gold_count': len(q.get('gold') or [q['correct_option']]), 'planned_calls': n,
                              'before_exact_correct': a, 'after_exact_correct': b,
                              'before_state': state(a, n), 'after_state': state(b, n), 'transition': state(a, n) + '_to_' + state(b, n),
                              'historical_clean_exact_correct_other_runtime': historical_a,
                              'historical_frontier_group': published['historical_frontier_group'],
                              'formal_feedback_seed': published['formal_feedback_seed'],
                              'gold_reliability': published['gold_reliability'],
                              'question_source_path': str(V131 / 'QUESTIONS.json'), 'calls': calls})
    train = read(V125 / 'data/COMBINED_QUESTIONS.json')['questions']
    train_ids = {q['candidate_id'] for q in train}
    assert len(train) == len(train_ids) == 363
    assert not train_ids & set(questions) and not train_ids & set(frontier_questions)
    train_coverage = {
        'second_round_actual_training_question_count': 363,
        'second_round_training_root_count': 68,
        'train_ids_intersect_v127_dev_ids': [], 'train_ids_intersect_v131_historical_frontier_ids': [],
        'full_363_question_second_round_pre_post_solver_pair_found': False,
        'actual_new_policy_evaluation_packages_inspected': [str(V127), str(V131)],
        'earlier_source_batch_pair_model_scope': 'M0 versus the first-round M_clean_policy, before second-round training. The word trained in those packages is not M_v125_newonly.',
        'source_batch_scope_evidence': [
            str(DELIVERIES / 'h4_codex_fullscale_ab_v118/evidence/CONTENT_CONTRACT.md'),
            str(DELIVERIES / 'h4_codex_scene_retry_v120/evidence/CONTENT_CONTRACT.md'),
            str(DELIVERIES / 'h4_codex_all_failed_repair_v121/evidence/CONTENT_CONTRACT.md'),
            str(DELIVERIES / 'h4_codex_remaining_six_repair_v122/evidence/CONTENT_CONTRACT.md')],
        'limitation': 'DEV and old Train-frontier questions are not the363 new training questions. No complete second-round pre/post solver claim for the363 is made from these existing evaluations.'}
    for path in train_coverage['source_batch_scope_evidence']:
        assert Path(path).is_file(); read_text = Path(path).read_text()
        assert 'M_clean_policy' in read_text and 'M0' in read_text
        bindings[str(Path(path).resolve())] = sha(path)
    for path, digest in bindings.items(): assert sha(path) == digest, 'INPUT_CHANGED_DURING_READ'
    result = {
        'schema_version': 'v132_second_round_fixed_exact_transition_v1', 'status': 'PASS',
        'before_model': 'M_clean_policy', 'after_model': 'M_v125_newonly',
        'definitions': {
            'zero': '0 fully correct answers across every planned call for that question',
            'partial': 'At least1 but fewer than all planned calls fully correct; intermittent complete success, not within-answer partial credit',
            'full': 'Every planned call fully correct',
            'can_answer_nonzero': 'At least1 fully correct call; weaker criterion than stable full correctness',
            'can_answer_full': 'Every planned call fully correct; does not establish competence outside these questions/permutations',
            'no_claim_zero_internal_knowledge': 'Zero exact success does not mean no correct individual options or ordering pairs.'},
        'dev_v127': {'scope': 'Fixed48 DEV questions from7 covered roots out of15 fixed DEV roots; all fully valid raw calls.',
                     'calls_per_question': {'multi_select': 18, 'ordering': 12}, 'strata': strata(dev_rows), 'questions': dev_rows,
                     'limitations': ['Sorting templates are uniform and rule retrieval can solve all14. Zero-to-full supports learning this fixed task, not general complex temporal reasoning.',
                                     'No direct363-training-question before/after claim. Gold6 controls separate from Gold2345 main multiselect.']},
        'historical_frontier_v131': {'scope': 'Fixed123 outcome-selected old questions on58 Train roots; same current-runtime clean versus new primary comparison.',
                                     'calls_per_question': {'multi_select': 15, 'ordering': 12, 'single_choice': 12},
                                     'strata': strata(frontier_rows), 'questions': frontier_rows,
                                     'historical_clean_other_runtime_strata': strata([{**r, 'before_exact_correct': r['historical_clean_exact_correct_other_runtime'],
                                         'before_state': state(r['historical_clean_exact_correct_other_runtime'], r['planned_calls']),
                                         'transition': state(r['historical_clean_exact_correct_other_runtime'], r['planned_calls']) + '_to_' + r['after_state']} for r in frontier_rows]),
                                     'limitations': ['All123 were historically M0-zero and M_clean-partial by selection. Current runtime has oneclean-zero and oneclean-full; preserve the bridge distinction.',
                                                     'Outcome selection and Train-root overlap prevent unbiased DEV/TEST training-gain claims.',
                                                     '121 Golds lack applicable semantic review;2 have same-content historicalaccept only. No Gold correction or semantic filtering performed.']},
        'new_training_question_evaluation_coverage': train_coverage,
        'source_report_paths': [str(V127 / 'REPORT_ZH.md'), str(V131 / 'REPORT_ZH.md')],
        'input_bindings': [{'path': path, 'sha256': digest} for path, digest in sorted(bindings.items())],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream: json.dump(result, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(json.dumps({'status': 'PASS', 'output': str(args.output),
                      'dev_zero_to_nonzero': result['dev_v127']['strata']['all']['zero_to_nonzero_count'],
                      'dev_zero_to_full': result['dev_v127']['strata']['all']['zero_to_full_count'],
                      'frontier_zero_to_nonzero': result['historical_frontier_v131']['strata']['all']['zero_to_nonzero_count']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
