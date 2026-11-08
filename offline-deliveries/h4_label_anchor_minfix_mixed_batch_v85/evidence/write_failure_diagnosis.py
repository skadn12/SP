"""Post-review diagnostic replay; does not certify semantics or change candidates."""
from pathlib import Path
from collections import Counter
import sys

HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOME))
from common import read, write, bind, now, need
from ordering_generation import whole_set
from actor_reference import name_policy_errors

need(not (HOME / 'evidence/RUN_COMPLETE.json').exists(), 'already_sealed')
batches = [('v84', HOME.parent / 'h4_independent_roots_mixed_batch_v84'), ('v85', HOME)]
rows = []
for version, home in batches:
    for run in sorted((home / 'ordering').glob('root*_ordering')):
        response = next((run / 'requests').glob('*/RESPONSE.json'))
        import json
        drafts = json.loads(read(response)['text'])['questions']
        plan_path = run / 'ANCHOR_PLAN.json'
        plans = {p['candidate_id']: p for p in read(plan_path)['questions']}
        source = read(run / 'SOURCE.json')
        positions = {t['turn_ref']: i for i, t in enumerate(source['turns'])}
        turns = {t['turn_ref']: t for t in source['turns']}
        rejects = {q['candidate_id']: q['reasons'] for q in read(run / 'REJECTIONS.json')}
        prior = read(run / 'PRIOR.json')
        for q in drafts:
            need(q is not None, 'unexpected_null_draft')
            anchors = plans[q['candidate_id']]['anchors']
            computed = sorted(anchors, key=lambda k: (positions[anchors[k]['turn_ref']], turns[anchors[k]['turn_ref']]['text'].index(anchors[k]['quote'])))
            # Root manually read every displayed item/anchor pair for both batches.
            # This narrow theme correspondence finding is NOT full semantic certification.
            theme_binding = version == 'v85' or q['root_id'].endswith('-103866')
            matches = [p['candidate_id'] for p in prior if whole_set(p) == whole_set(q)]
            rows.append({'batch': version, 'candidate_id': q['candidate_id'],
                'response': bind(response), 'plan': bind(plan_path),
                'formal_first_rejection': rejects.get(q['candidate_id'], []),
                'topic_corresponds_to_fixed_label_in_all_six_items': theme_binding,
                'topic_check_method': 'Manual reading of all six verbatim item/fixed-anchor pairs; not an independent full-source semantic answer review.',
                'pairs': {k: {'item': q['items'][k], 'fixed_anchor': anchors[k]} for k in anchors},
                'writer_gold': q['gold'], 'computed_anchor_order': computed,
                'gold_array_matches_anchor_order': q['gold'] == computed,
                'name_policy_errors': name_policy_errors(q['items']),
                'identical_prior_whole_sets': matches, 'writer_order_reason': q['order_reason']})
summaries = {}
for version, home in batches:
    r = [q for q in rows if q['batch'] == version]
    summaries[version] = {'draft_count': len(r),
        'mechanically_admitted': sum(not q['formal_first_rejection'] for q in r),
        'topic_label_misaligned_drafts': sum(not q['topic_corresponds_to_fixed_label_in_all_six_items'] for q in r),
        'wrong_gold_arrays_including_already_rejected_drafts': sum(not q['gold_array_matches_anchor_order'] for q in r),
        'name_policy_violating_drafts': sum(bool(q['name_policy_errors']) for q in r),
        'identical_prior_whole_sets_including_already_rejected_drafts': sum(bool(q['identical_prior_whole_sets']) for q in r)}
supplements = []
for root in ('103866', '135466'):
    run = HOME / 'multiselect/assets' / ('root' + root + '_family')
    initial = {r['fact_id']: r for r in read(run / 'rounds/initial/PUBLIC_ASSETS.json')['versions']}
    folder = run / 'rounds/supplement'
    manifest = read(folder / 'MANIFEST.json')
    review = read(folder / 'SOURCE_JUDGMENTS.json')
    judgments = {r['version_id']: r for r in review['versions']}
    request = next((folder / 'requests').glob('*/REQUEST.json'))
    content = read(request)['messages'][0]['content']
    feedback = manifest['content_feedback']
    for item in read(folder / 'PUBLIC_ASSETS.json')['versions']:
        f = next(v for v in feedback if v['fact_id'] == item['fact_id'])
        # JSON string escaping is allowed when locating the exact feedback reason.
        literal = json.dumps(f['reason'], ensure_ascii=False)[1:-1]
        need(f['reason'] in content or literal in content, 'feedback_reason_not_in_actual_payload')
        old = initial[item['fact_id']]['proposition']
        supplements.append({'family_id': run.name, 'fact_id': item['fact_id'],
            'initial_text': old, 'supplement_text': item['proposition'],
            'identical_text': old == item['proposition'],
            'identical_after_removing_commas': old.replace(',', '') == item['proposition'].replace(',', ''),
            'saved_relation': judgments[item['version_id']]['relation'],
            'feedback_reason_in_actual_request': True, 'feedback': f,
            'request': bind(request), 'judgments': bind(folder / 'SOURCE_JUDGMENTS.json')})
write(HOME / 'evidence/FAILURE_DIAGNOSIS_v85.json', {
    'created_at_utc': now(), 'status': 'POST_REVIEW_READ_ONLY_DIAGNOSIS',
    'reviewer': '/root', 'current_private_writer_and_anchor_material_read': True,
    'source_first_semantic_review_claimed': False, 'candidate_or_gold_edits': [],
    'ordering_summary': summaries, 'ordering_drafts': rows,
    'multiselect_supplements': supplements,
    'interpretation': [
        'Current label/topic correspondence improved from 2/8 to 8/8, while mechanical yield stayed 1/8. Theme alignment alone does not establish proposition fidelity.',
        'Formal first rejections are disjoint pipeline outcomes. Secondary Gold, pronoun and prior-duplicate findings overlap and must not be added.',
        'Both 103866 v85 reasons eventually state the correct source order after emitting a different Gold array; the immutable drafts were rejected without correcting Gold.',
        'All three supplemented public variants remain UNKNOWN. Exact saved content feedback was present; two variants repeated verbatim and the song variant only removed commas.',
        'These are the same historical TRAIN roots, with changed random IDs, prior inventory and prompt layout; this is not a controlled single-variable causal estimate or an independent test.'
    ],
    'next_small_experiment_not_executed': {
        'ordering': 'Ask the writer to emit order_reason before gold and to make the final gold array match the final sequence stated in that reason; retain all rejection gates and prohibit automatic Gold repair. This does not by itself solve pronoun or duplicate failures.',
        'multiselect_separate_experiment': 'Place the failed public sentence and its exact UNKNOWN scope reason beside each supplement output slot, requiring an explicitly scoped contradictory proposition or null; retain the current source-first review.'},
    'training_allowed': False, 'automatic_followup': False})
print({'ordering': summaries, 'supplement_unknown_count': len(supplements)})
