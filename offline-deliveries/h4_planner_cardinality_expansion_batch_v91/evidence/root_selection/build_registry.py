"""Freeze transparent v91 root selection; no historical writes or model calls."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


HOME = Path(__file__).resolve().parents[2]
PROJECT = Path('/root/autodl-tmp/gxy-selfplay')
DELIVERIES = PROJECT / 'offline-deliveries'
V89 = DELIVERIES / 'h4_scale_pilot_v89'
V90 = DELIVERIES / 'h4_end_to_end_expansion_mixed_v90'
QA = PROJECT / 'restart-n5-runs/h4_unexpanded122_qa_generation_only_v1'
EVAL = PROJECT / 'restart-n5-runs/h4_round0_trained_solver_train_reeval_v1_live_1'
NEW = (14304, 19053, 19708, 20432, 24719, 25652, 38601, 9557)
REGRESSION = (104814, 194259)
SCENARIOS = {
    14304: 'family planning, relationship communication, Florida travel and date night',
    19053: 'team project chairing, colleague collaboration and scheduled work meeting',
    19708: 'forgiveness, family support and a gift visit',
    20432: 'parent-child communication, working late and a family weekend plan',
    24719: 'longstanding friendship, career promotion and a celebration dinner',
    25652: 'department leadership, confidence and meeting a new neighbor',
    38601: 'friendship, movie nights and shared leisure activities',
    9557: 'contract negotiation, management, project rollout and external training',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def main():
    matrix_path = EVAL / 'candidate_transition_matrix.jsonl'
    exposure_path = EVAL / 'candidate_training_exposure_manifest.jsonl'
    matrix = [json.loads(line) for line in matrix_path.read_text().splitlines() if line]
    exposure_rows = [json.loads(line) for line in exposure_path.read_text().splitlines() if line]
    exposure = {row['candidate_id']: row for row in exposure_rows}
    assert len(exposure) == len(exposure_rows), 'duplicate exposure candidate'
    frozen_lock = read(DELIVERIES / 'h4_groundless_random_feedback_v1/feedback/compiled_v4/source_lock.json')
    assert sha(matrix_path) == frozen_lock['evaluation_sources']['candidate_transition_matrix']['sha256']
    assert sha(exposure_path) == frozen_lock['evaluation_sources']['candidate_training_exposure']['sha256']
    pools = {path.parent.name: path for path in QA.glob('shards/*/roots/*/candidate_pool.json')}
    prior_specs = {row['root_id']: row for row in read(V90 / 'expansion_engine/SOURCE_REGISTRY.json')['roots']}
    registry, evidence = [], []
    checks = []
    for number in NEW + REGRESSION:
        root = f'cc-train-episode-{number}'
        if number in NEW:
            review_path = V89 / f'source_reviews/root{number}_REVIEW.json'
            review = read(review_path)
            spec = {
                'root_id': root,
                'source_path': review['source']['path'],
                'source_sha256': review['source']['sha256'],
                'baseline_review_path': str(review_path),
                'baseline_review_sha256': sha(review_path),
                'baseline_review_reuse': 'historical_source_baseline_only_newly_appended_source_requires_fresh_full_review',
                'comparison_source_path': None,
                'comparison_source_status': 'no_historical_comparison_source',
            }
        else:
            spec = dict(prior_specs[root])
            review = read(spec['baseline_review_path'])
        source_path = Path(spec['source_path'])
        source = read(source_path)
        turns = source['turns']
        assert source['root_id'] == root
        assert sha(source_path) == spec['source_sha256']
        assert review['root_id'] == root and review['decision'] == 'READY'
        assert not any(row['severity'] == 'BLOCKING' for row in review['findings'])
        assert len({row['turn_ref'] for row in turns}) == len(turns)
        assert not any(row['turn_ref'].startswith('R02:B:') for row in turns)
        pool_path = pools[root]
        pool_rows = read(pool_path)
        pool = {row['candidate_id']: row for row in pool_rows}
        assert len(pool) == len(pool_rows), 'duplicate historical candidate'
        transcript_path = pool_path.parent / 'expanded_dialogue_clean.json'
        transcript = read(transcript_path)
        assert transcript['root_id'] == root
        historical = [row for row in matrix if row['root_id'] == root]
        assert historical and all(row['split'] == 'train' for row in historical)
        assert len({row['candidate_id'] for row in historical}) == len(historical)
        assert set(pool) >= {row['candidate_id'] for row in historical}
        assert all(row['candidate_id'] in exposure for row in historical)
        assert all(exposure[row['candidate_id']]['root_id'] == root for row in historical)
        assert all('M_clean_policy' in row['models'] for row in historical)
        by_type = Counter(row['question_type'] for row in historical)
        assert by_type['multi_select'] > 0 and by_type['ordering'] > 0
        projection = lambda rows: [(row['turn_ref'], row['speaker'], row['text']) for row in rows]
        prefix = projection(turns)[:len(transcript['turns'])] == projection(transcript['turns'])
        assert prefix, 'first-round transcript must be a complete prefix of expansion source'
        spec.update({
            'source_turns': len(turns),
            'cohort': 'batch_other_roots' if number in NEW else 'prior_planner_failure_regression',
            'historical_split': 'train',
            'historical_baseline_decision': 'READY',
            'historical_source_has_R02_B_suffix': False,
            'first_round_feedback': {
                'candidate_pool_path': str(pool_path),
                'candidate_pool_sha256': sha(pool_path),
                'transcript_path': str(transcript_path),
                'transcript_sha256': sha(transcript_path),
                'candidate_transition_matrix_path': str(matrix_path),
                'candidate_transition_matrix_sha256': sha(matrix_path),
                'candidate_training_exposure_path': str(exposure_path),
                'candidate_training_exposure_sha256': sha(exposure_path),
                'has_root_specific_feedback': True,
                'evaluation_run_id': EVAL.name,
                'model_name': 'M_clean_policy',
                'historical_candidate_count': len(historical),
                'historical_candidate_counts_by_type': dict(by_type),
                'historical_gold_semantic_certified': False,
            },
        })
        registry.append(spec)
        evidence.append({
            'root_id': root,
            'cohort': spec['cohort'],
            'scenario_summary_from_existing_source': SCENARIOS.get(number, 'v90 planner cardinality regression'),
            'input_turns': len(turns),
            'allowed_speakers': sorted({row['speaker'] for row in turns}),
            'baseline_review': artifact(spec['baseline_review_path']),
            'baseline_warning_kinds': dict(Counter(row['kind'] for row in review['findings'])),
            'first_round_transcript_turns': len(transcript['turns']),
            'first_round_transcript_matches_complete_source_prefix': prefix,
            'first_round_feedback': spec['first_round_feedback'],
            'recent_usage': ['v89 complete historical source review and mixed-question generation'] if number in NEW else ['v90 expansion attempted; two planner attempts exceeded 2-5 developments'],
            'claims_limit': 'historical TRAIN root; not unseen-test data; historical review is a baseline only, not a certificate for newly appended scenes',
        })
        checks.append({'root_id': root, 'status': 'PASS', 'source_hash_verified': True,
                       'complete_source_unique_turn_refs': True, 'no_new_suffix_ref_collision': True,
                       'historical_ready_no_blocking': True, 'first_round_feedback_candidate_join_complete': True})
    document = {
        'schema_version': 'h4.v91.expansion_source_registry.v1',
        'historical_prefix_policy': 'preserve_complete_exact_source_turn_sequence',
        'suffix_turn_ref_prefix': 'R02:B:',
        'roots': registry,
    }
    write_new(HOME / 'expansion_engine/SOURCE_REGISTRY.json', document)
    write_new(HOME / 'evidence/root_selection/ROOT_SELECTION.json', {
        'schema_version': 'h4.v91.root_selection.v1',
        'saved_at_utc': datetime.now(timezone.utc).isoformat(),
        'selection_method': 'eight other roots with existing complete READY baseline source, complete authentic first-round TRAIN feedback, varied source length and scenario; two prior failures isolated for regression',
        'selection_is_random_or_unbiased_sample': False,
        'source_length_range_other_roots': [min(row['source_turns'] for row in registry if row['cohort'] == 'batch_other_roots'), max(row['source_turns'] for row in registry if row['cohort'] == 'batch_other_roots')],
        'v90_four_roots_excluded_from_other_root_cohort': [103866, 104814, 135466, 194259],
        'other_root_count': len(NEW),
        'regression_root_count': len(REGRESSION),
        'historical_reviews_recertify_new_extensions': False,
        'all_selected_have_authentic_root_specific_first_round_feedback': True,
        'global_fallback_needed': False,
        'frozen_original_feedback_compiler_scope': read(DELIVERIES / 'h4_groundless_random_feedback_v1/feedback/compiled_v4/COMPILED_FEEDBACK.json')['root_order'],
        'raw_feedback_available_root_count': len({row['root_id'] for row in matrix}),
        'feedback_adaptation_required': 'recompile unchanged historical candidate metrics for explicit new root scope with true candidate-pool/transcript hashes; no fabricated root examples and no current-round results',
        'registry': artifact(HOME / 'expansion_engine/SOURCE_REGISTRY.json'),
        'raw_transition_matrix': artifact(matrix_path),
        'raw_training_exposure': artifact(exposure_path),
        'checks': checks,
        'roots': evidence,
    })
    print(json.dumps({'status': 'PASS', 'other_roots': list(NEW), 'regression_roots': list(REGRESSION),
                      'input_turns': sum(row['source_turns'] for row in registry)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
