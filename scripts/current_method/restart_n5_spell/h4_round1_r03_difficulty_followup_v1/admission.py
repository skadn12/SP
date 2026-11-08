"""Quality and novelty precede the existing first-round difficulty buckets."""
from ..h4_round0_b0_k3_difficulty_v1.difficulty import classify_difficulty_counts

MODELS = ('M0', 'M_clean_policy', 'M_inclusive_policy')


def difficulty_admission(review, measurements):
    blockers = [key for key in ('semantic_supported', 'novel_answer_target',
                               'independent_fact_combination', 'intent_relation_satisfied')
                if review.get(key) is not True]
    if blockers:
        return {'status': 'QUALITY_OR_NOVELTY_BLOCKED', 'blockers': blockers,
                'hard_pool_eligible': False, 'difficulty_increase_established': False}
    if any(model not in measurements for model in MODELS):
        return {'status': 'VALID_UNMEASURED', 'hard_pool_eligible': False,
                'difficulty_increase_established': False}
    buckets = {}
    for model in MODELS:
        row = measurements[model]
        buckets[model] = classify_difficulty_counts(
            full_correct=row['full_context_correct_count'],
            full_total=row['full_context_total_count'],
            no_correct=row['no_context_correct_count'],
            no_total=row['no_context_total_count'],
            technical=bool(row.get('technical_exclusion_reason')))
    review_buckets = {'technical_excluded', 'full_zero_review', 'full_context_worse'}
    if review_buckets.intersection(buckets.values()):
        status = 'MEASUREMENT_REVIEW_REQUIRED'
    elif any(buckets[m] == 'hard_candidate' for m in MODELS[1:]):
        status = 'TRAINED_SOLVER_HARD_CANDIDATE'
    else:
        status = 'NO_TRAINED_SOLVER_HARD_CANDIDATE'
    return {'status': status, 'model_buckets': buckets,
            'hard_pool_eligible': status == 'TRAINED_SOLVER_HARD_CANDIDATE',
            # New difficulty and a causal feedback improvement are different claims.
            'difficulty_increase_established': False}
