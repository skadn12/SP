"""CPU-only paired comparison after all three fixed endpoints have scored DEV.

Reads committed private score packets; prints only aggregates. There is no
model loading, generation, training, checkpoint selection or Challenger export.
"""
from __future__ import annotations

import collections
from fractions import Fraction
from pathlib import Path
import statistics

import run

HERE = Path(__file__).resolve().parent


def independently_recalculate(role, scored, rawfreeze, requests, gold, maps, choices):
    """Check actual completed traces arithmetically without calling native scorer."""
    run.verify_bindings(rawfreeze['files'])
    stored_requests = {r['request_id']: r for r in scored['requests']}
    byquestion = collections.defaultdict(list)
    for request in requests:
        rid = request['request_id']
        raw = run.read(HERE / 'models' / role / 'responses' / (rid + '.json'))
        valid = raw['status'] == 'RETURNED' and raw['metadata']['finish_reason'] in ('stop', 'eos') and raw['raw_output'] in choices[request['constraint_key']]
        g = gold[request['candidate_id']]['gold_content_ids']
        mapping = maps[request['prompt_id']]['display_to_content']
        exact = False
        partial = Fraction(0)
        if valid:
            run.require(raw['seed'] == request['sampling']['seed'] and raw['metadata']['sampling'] == request['sampling'] and raw['prompt_sha256'] == request['prompt_sha256'], 'RAW_REQUEST_DRIFT_IN_INDEPENDENT_RECALCULATION')
            labels = raw['raw_output'][5:-6].split(',')
            prediction = [mapping[label] for label in labels]
            if request['question_type'] == 'multi_select':
                ps, gs = set(prediction), set(g)
                tp, fp = len(ps & gs), len(ps - gs)
                partial = Fraction(tp, len(gs))
                if len(gs) != len(mapping):
                    partial -= Fraction(fp, len(mapping) - len(gs))
                exact = ps == gs
            else:
                rank = {item: index for index, item in enumerate(prediction)}
                pairs = [(a, b) for i, a in enumerate(g) for b in g[i + 1:]]
                correct = sum(rank[a] < rank[b] for a, b in pairs)
                partial = Fraction(2 * correct - len(pairs), len(pairs))
                exact = prediction == g
        stored = stored_requests[rid]
        run.require(stored['valid'] == valid and stored['exact'] == exact and stored['partial_score'] == float(partial), 'NATIVE_SCORE_DISAGREES_WITH_INDEPENDENT_ARITHMETIC')
        byquestion[request['candidate_id']].append((exact, float(partial)))
    questions = {}
    for q in scored['questions']:
        data = byquestion[q['candidate_id']]
        exact_rate = statistics.mean(row[0] for row in data)
        partial = statistics.mean(row[1] for row in data)
        run.require(q['exact_rate'] == exact_rate and q['partial_score_mean'] == partial and q['planned_requests'] == len(data), 'QUESTION_AGGREGATE_DISAGREES_WITH_INDEPENDENT_RECALCULATION')
        questions[q['candidate_id']] = {**q, 'exact_rate': exact_rate, 'partial_score_mean': partial}
    return questions


def main():
    run.verify()
    # Check that every fixed endpoint has a committed trace before this process
    # opens the private DEV scoring packet.
    for role in ('S0', 'A1', 'B1'):
        aggregate = run.read(HERE / 'models' / role / 'AGGREGATE.json')
        run.require(aggregate['status'] == 'COMPLETE', 'PAIRED_COMPARISON_REQUIRES_COMPLETE_ENDPOINTS')
        run.verify_bindings([aggregate['raw_response_freeze']])
        rawfreeze = run.read(HERE / 'models' / role / 'RAW_RESPONSE_FREEZE.json')
        run.verify_bindings(rawfreeze['files'])
    requests = run.lines(HERE / 'DEV_REQUESTS.jsonl')
    gold = {q['candidate_id']: q for q in run.lines(HERE / 'private/DEV_GOLD.jsonl')}
    maps = run.read(HERE / 'DEV_DISPLAY_MAPS.json')
    choices = run.read(HERE / 'FINITE_CHOICES.json')
    models = {}
    bindings = [run.bindfile(__file__), run.bindfile(HERE / 'DEV_REQUEST_FREEZE.json')]
    for role in ('S0', 'A1', 'B1'):
        directory = HERE / 'models' / role
        aggregate = run.read(directory / 'AGGREGATE.json')
        run.require(aggregate['status'] == 'COMPLETE', 'PAIRED_COMPARISON_REQUIRES_COMPLETE_ENDPOINTS')
        run.verify_bindings([aggregate['execution_manifest'], aggregate['raw_response_freeze'], aggregate['private_scores']])
        manifest = run.read(directory / 'EXECUTION_MANIFEST.json')
        run.require(manifest['model_role'] == role and manifest['request_freeze']['sha256'] == run.sha(HERE / 'DEV_REQUEST_FREEZE.json'), 'ENDPOINT_REQUEST_FREEZE_MISMATCH')
        scored = run.read(directory / 'private/SCORED_REQUESTS.json')
        rawfreeze = run.read(directory / 'RAW_RESPONSE_FREEZE.json')
        questions = independently_recalculate(role, scored, rawfreeze, requests, gold, maps, choices)
        request_scores = {r['request_id']: r for r in scored['requests']}
        run.require(len(questions) == 42 and len(request_scores) == 660, 'PAIRED_DENOMINATOR_MISMATCH')
        models[role] = {'questions': questions, 'requests': request_scores}
        bindings.append(run.bindfile(directory / 'AGGREGATE.json'))
        bindings.extend([aggregate['execution_manifest'], aggregate['raw_response_freeze'], aggregate['private_scores']])
    run.require(set(models['S0']['questions']) == set(models['A1']['questions']) == set(models['B1']['questions']), 'PAIRED_QUESTION_IDS_DIFFER')
    run.require(set(models['S0']['requests']) == set(models['A1']['requests']) == set(models['B1']['requests']), 'PAIRED_REQUEST_IDS_DIFFER')
    allquestions = models['S0']['questions']
    comparisons = {}
    for left, right in [('B1', 'A1'), ('A1', 'S0'), ('B1', 'S0')]:
        groups = {}
        for cohort, reporting, required_n in [('historical115_DEV', 'HISTORICAL_REGRESSION_PRIMARY', 35), ('historical115_DEV', 'GOLD6_CONTROL_ONLY', 5), ('previous2MC_DEV', 'SEPARATE_PREVIOUS_MC_SUPPLEMENT', 2)]:
            group_ids = [cid for cid, q in allquestions.items() if q['cohort_id'] == cohort and q['reporting_role'] == reporting]
            run.require(len(group_ids) == required_n, 'PAIRED_GROUP_DENOMINATOR_CHANGED')
            for typ in ('ALL', 'multi_select', 'ordering'):
                ids = [cid for cid in group_ids if typ == 'ALL' or allquestions[cid]['question_type'] == typ]
                if not ids:
                    continue
                differences = []
                root_differences = collections.defaultdict(list)
                partial_differences = []
                for cid in ids:
                    a, b = models[left]['questions'][cid], models[right]['questions'][cid]
                    for key in ('cohort_id', 'root_id', 'question_type', 'reporting_role', 'planned_requests'):
                        run.require(a[key] == b[key] == allquestions[cid][key], 'PAIRED_QUESTION_METADATA_DRIFT')
                    delta = a['exact_rate'] - b['exact_rate']
                    differences.append(delta)
                    root_differences[a['root_id']].append(delta)
                    partial_differences.append(a['partial_score_mean'] - b['partial_score_mean'])
                root_means = [statistics.mean(values) for values in root_differences.values()]
                groups[cohort + '/' + reporting + '/' + typ] = {'questions': len(ids), 'roots': len(root_means), 'difference_direction': left + '_minus_' + right, 'question_equal_exact_difference': statistics.mean(differences), 'root_equal_exact_difference': statistics.mean(root_means), 'paired_root_cluster_CI': run.cluster_ci(root_means), 'question_equal_partial_difference': statistics.mean(partial_differences), 'improved_questions': sum(d > 0 for d in differences), 'worsened_questions': sum(d < 0 for d in differences), 'unchanged_questions': sum(d == 0 for d in differences)}
        comparisons[left + '_minus_' + right] = groups
    result = {'schema_version': 'day5_fixed_endpoint_paired_DEV_aggregate_v1', 'status': 'COMPLETE', 'created_at_utc': run.now(), 'primary_comparison': 'B1_minus_A1/historical115_DEV/HISTORICAL_REGRESSION_PRIMARY/ALL', 'primary_question_count': 35, 'paired_by': 'same frozen candidate IDs, request IDs, prompt bytes, permutations, samples and seeds', 'scorer_independent_recalculation': {'status': 'PASS', 'requests': 1980, 'question_endpoint_pairs': 126, 'native_scorer_called_by_comparison': False}, 'comparisons': comparisons, 'bindings': bindings, 'checkpoint_selected_by_DEV': False, 'challenger_release': False, 'TEST_requests': 0, 'historical_regression_only': True, 'causal_claim_limit': 'Single small matched training comparison on historically exposed held-out roots; descriptive paired differences do not establish broad feedback causality or unseen-root generalization'}
    run.save(HERE / 'PAIRED_AGGREGATE.json', result)
    print(__import__('json').dumps({key: result[key] for key in ('status', 'primary_question_count', 'comparisons')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
