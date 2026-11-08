"""Post-generation diagnostic replay of immutable ordering drafts; no admission or answer repair.

Manual mappings below were recorded after reading every item/fixed-anchor pair
for both batches and every full selected v86 scene. They certify only narrow
topic correspondence, never full source support or unique semantic chronology.
"""
from collections import Counter
from pathlib import Path
import json
import re
import sys

BATCH = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(BATCH))
from common import bind, need, now, read, write
from ordering_generation import whole_set
from actor_reference import name_policy_errors

need(not (BATCH / 'evidence/RUN_COMPLETE.json').exists(), 'already_sealed')
OUTPUT = BATCH / 'evidence/FAILURE_DIAGNOSIS_v86.json'
need(not OUTPUT.exists(), 'refuse_overwrite_diagnosis')
LABELS = tuple('I' + str(i) for i in range(1, 7))
# Each value states the fixed anchor whose topic the item under I1..I6 describes.
# These judgments were made manually, not inferred by a semantic matching algorithm.
MANUAL_TOPIC_MAPS = {
    ('v86', '103866', 1): ('I1', 'I4', 'I3', 'I2', 'I5', 'I6'),
    ('v86', '103866', 2): LABELS,
    ('v86', '109164', 1): LABELS,
    ('v86', '109164', 2): LABELS,
    ('v86', '112812', 1): ('I1', 'I3', 'I6', 'I4', 'I2', 'I5'),
    ('v86', '112812', 2): LABELS,
    ('v86', '135466', 1): ('I4', 'I6', 'I3', 'I2', 'I5', 'I1'),
    ('v86', '135466', 2): LABELS,
    **{('v85', root, slot): LABELS for root in ('103866', '109164', '112812', '135466') for slot in (1, 2)},
}
PAIR_NOTES = {
    ('v86', '103866', 1, 'I2'): 'Camping-gear/emergency-packing topic belongs to fixed I4 (Emily T004), but the item also incorrectly attributes the statement to James. Topic mapping is not semantic endorsement.',
    ('v86', '103866', 1, 'I4'): 'Inexperience and a good experience for the kids belongs to fixed I2 (James T003).',
    ('v86', '112812', 1, 'I1'): 'Calendar topic matches I1, but notify the system reverses the source recipient: source says the system could notify us. This auxiliary fidelity finding is not a completed full-source review.',
    ('v85', '112812', 1, 'I1'): 'Calendar topic matches I1, but notify the system reverses the source recipient: source says the system could notify us. Historical topic alignment did not certify fidelity.',
    ('v86', '112812', 1, 'I2'): 'Manual-rescheduling frustration belongs to fixed I3 (Emily T002), not I2 appointment trial (James T007).',
    ('v86', '112812', 1, 'I3'): 'Phone/email sync belongs to fixed I6 (James T003), not I3 manual-rescheduling frustration.',
    ('v86', '112812', 1, 'I5'): 'Appointment trial and setup help belongs to fixed I2 (James T007), not I5 two-week review.',
    ('v86', '112812', 1, 'I6'): 'Two-week review belongs to fixed I5 (James T009), not I6 phone/email sync.',
    ('v86', '135466', 1, 'I1'): 'Knee pain belongs to fixed I4 (Morgan T001), not I1 explanation of scraping the knee.',
    ('v86', '135466', 1, 'I2'): 'Bandage offer belongs to fixed I6 (Quinn T006), not I2 offer to walk home.',
    ('v86', '135466', 1, 'I4'): 'Offer to walk home belongs to fixed I2 (Quinn T008), not I4 knee pain.',
    ('v86', '135466', 1, 'I6'): 'Bike/rock scraping explanation belongs to fixed I1 (Morgan T003), not I6 bandage offer.',
}
REASON_NOTES = {
    ('v86', '103866', 1): 'Explicit sequence is I1,I2,I3,I4,I5,I6. The prose assigns inexperience to I2 and camping gear to I4, agreeing with fixed topics but disagreeing with the swapped item texts. It also puts T005 before T004 and does not match the emitted Gold.',
    ('v86', '103866', 2): 'Explicit label sequence equals Gold, but it places coconut milk (T008) before rice (T007); the reason and array agree on an incorrect order.',
    ('v86', '109164', 1): 'One explicit final six-label sequence agrees with Gold and fixed-anchor order; pronoun rejection remains.',
    ('v86', '109164', 2): 'One explicit final six-label sequence agrees with Gold and fixed-anchor order; pronoun rejection remains.',
    ('v86', '112812', 1): 'Reason gives chronological turn numbers, not the requested six-label sequence. Mapping those turns through fixed anchors reproduces Gold, but four item topics use different labels. No semantic answer reliability follows.',
    ('v86', '112812', 2): 'Reason gives chronological turn numbers, not the requested six-label sequence. Their fixed-anchor projection agrees with Gold; pronoun rejection remains.',
    ('v86', '135466', 1): 'Reason lists T001,T006,T007,T008,T009,T003 respectively, which follows current item insertion order, not source chronology. It contains no final label permutation and does not support Gold.',
    ('v86', '135466', 2): 'Reason lists T012,T005,T017,T009,T015,T010 respectively, exactly the current item insertion mapping, not chronological order. It contains no final label permutation even though Gold itself matches fixed-anchor order.',
    ('v85', '103866', 1): 'The final Therefore Gold clause corrects the source order after an earlier mistaken sequence, but the Gold array was emitted earlier and differs.',
    ('v85', '103866', 2): 'The final Therefore Gold clause corrects the source order after an earlier mistaken sequence, but the Gold array was emitted earlier and differs.',
}

batches = [('v85', BATCH.parent / 'h4_label_anchor_minfix_mixed_batch_v85'), ('v86', BATCH)]
rows = []
for version, home in batches:
    for run in sorted((home / 'ordering').glob('root*_ordering')):
        responses = sorted((run / 'requests').glob('*/RESPONSE.json'))
        need(len(responses) == 1, 'diagnostic_expected_one_request')
        response = responses[0]
        raw_text = read(response)['text']
        drafts = json.loads(raw_text)['questions']
        plan_path = run / 'ANCHOR_PLAN.json'
        plans = {p['candidate_id']: p for p in read(plan_path)['questions']}
        source = read(run / 'SOURCE.json')
        positions = {t['turn_ref']: i for i, t in enumerate(source['turns'])}
        turns = {t['turn_ref']: t for t in source['turns']}
        rejects = {q['candidate_id']: q['reasons'] for q in read(run / 'REJECTIONS.json')}
        admitted = {q['candidate_id'] for q in read(run / 'POOL.json')['questions']}
        prior = read(run / 'PRIOR.json')
        for slot_number, q in enumerate(drafts, 1):
            need(isinstance(q, dict), 'unexpected_null_or_malformed_draft')
            key = (version, q['root_id'].split('-')[-1], slot_number)
            plan = plans[q['candidate_id']]
            anchors = plan['anchors']
            computed = sorted(anchors, key=lambda label: (positions[anchors[label]['turn_ref']], turns[anchors[label]['turn_ref']]['text'].index(anchors[label]['quote'])))
            topic_map = dict(zip(LABELS, MANUAL_TOPIC_MAPS[key]))
            need(set(q['items']) == set(topic_map) == set(anchors), 'manual_pair_label_coverage')
            pairs = {}
            for label in LABELS:
                matched = topic_map[label]
                pairs[label] = {'verbatim_item': q['items'][label], 'fixed_anchor': anchors[label],
                    'manual_topic_corresponding_fixed_label': matched,
                    'manual_topic_corresponding_anchor': anchors[matched],
                    'topic_corresponds_to_own_fixed_label': matched == label,
                    'manual_note': PAIR_NOTES.get((*key, label), 'Manual pair reading: item topic corresponds to its own fixed anchor; this does not certify full semantic fidelity.')}
            reason = q['order_reason']
            reason_tail = reason.rsplit('Therefore Gold:', 1)[-1] if 'Therefore Gold:' in reason else reason
            labels_in_reason = re.findall(r'\bI[1-6]\b', reason_tail)
            explicit_sequence = labels_in_reason if len(labels_in_reason) == 6 and set(labels_in_reason) == set(LABELS) else None
            turn_suffixes = re.findall(r'\bT\d{3}\b', reason_tail)
            suffix_to_label = {anchor['turn_ref'].rsplit(':', 1)[-1]: label for label, anchor in anchors.items()}
            turn_projection = ([suffix_to_label[t] for t in turn_suffixes]
                if len(turn_suffixes) == 6 and len(set(turn_suffixes)) == 6 and all(t in suffix_to_label for t in turn_suffixes) else None)
            row = {'batch': version, 'candidate_id': q['candidate_id'], 'root_id': q['root_id'], 'slot_number': slot_number,
                'response': bind(response), 'request': bind(response.parent / 'REQUEST.json'),
                'plan': bind(plan_path), 'source': bind(run / 'SOURCE.json'), 'prior': bind(run / 'PRIOR.json'),
                'formal_rejection_record': bind(run / 'REJECTIONS.json'),
                'formal_first_rejection': rejects.get(q['candidate_id'], []), 'mechanically_admitted': q['candidate_id'] in admitted,
                'observed_json_field_order': list(q),
                'items_then_order_reason_then_gold_last': list(q).index('items') < list(q).index('order_reason') < list(q).index('gold') == len(q) - 1,
                'order_reason_before_gold': list(q).index('order_reason') < list(q).index('gold'),
                'writer_gold': q['gold'], 'computed_anchor_order': computed,
                'gold_array_matches_anchor_order': q['gold'] == computed,
                'writer_order_reason': reason, 'reason_final_explicit_label_sequence': explicit_sequence,
                'reason_final_explicit_label_sequence_equals_gold': None if explicit_sequence is None else explicit_sequence == q['gold'],
                'reason_final_explicit_label_sequence_equals_anchor_order': None if explicit_sequence is None else explicit_sequence == computed,
                'reason_listed_turn_suffixes': turn_suffixes, 'reason_turn_sequence_projected_through_fixed_anchors': turn_projection,
                'reason_listed_turns_are_chronological': None if turn_projection is None else turn_projection == computed,
                'reason_turn_projection_equals_gold': None if turn_projection is None else turn_projection == q['gold'],
                'manual_reason_interpretation': REASON_NOTES.get(key, 'Read reason as written. Computed comparisons are diagnostic and do not certify the proposition semantics.'),
                'manual_topic_correspondence_method': 'Every verbatim item/fixed-anchor pair was manually read after accessing private writer output. Selected v86 scenes were read in full, but the full source outside those scenes was not rereviewed for this auxiliary diagnostic.',
                'topic_corresponds_to_fixed_label_in_all_six_items': all(label == matched for label, matched in topic_map.items()),
                'topic_misaligned_labels': [label for label, matched in topic_map.items() if label != matched],
                'pairs': pairs, 'name_policy_errors': name_policy_errors(q['items']),
                'identical_prior_whole_sets': [p['candidate_id'] for p in prior if whole_set(p) == whole_set(q)]}
            need(row['mechanically_admitted'] == (not row['formal_first_rejection']), 'formal_status_mismatch')
            rows.append(row)

summaries = {}
for version, home in batches:
    current = [row for row in rows if row['batch'] == version]
    first = Counter('accepted' if row['mechanically_admitted'] else
        'pronoun_policy' if all(x.startswith('ordering_actor_reference_policy_forbidden_pronoun:') for x in row['formal_first_rejection']) else
        'gold_anchor_mismatch' if row['formal_first_rejection'] == ['writer_gold_disagrees_with_source_anchors'] else
        'whole_set_duplicate' if row['formal_first_rejection'] == ['duplicate_prior_or_current_six_item_set'] else 'other'
        for row in current)
    explicit = [row for row in current if row['reason_final_explicit_label_sequence'] is not None]
    summaries[version] = {'draft_count': len(current), 'mechanically_admitted': sum(row['mechanically_admitted'] for row in current),
        'formal_first_stage_outcomes_disjoint': dict(first),
        'order_reason_before_gold_drafts': sum(row['order_reason_before_gold'] for row in current),
        'items_then_order_reason_then_gold_last_drafts': sum(row['items_then_order_reason_then_gold_last'] for row in current),
        'topic_label_misaligned_drafts': sum(not row['topic_corresponds_to_fixed_label_in_all_six_items'] for row in current),
        'topic_label_misaligned_item_count': sum(len(row['topic_misaligned_labels']) for row in current),
        'wrong_gold_arrays_including_already_rejected_drafts': sum(not row['gold_array_matches_anchor_order'] for row in current),
        'name_policy_violating_drafts': sum(bool(row['name_policy_errors']) for row in current),
        'identical_prior_whole_sets_including_already_rejected_drafts': sum(bool(row['identical_prior_whole_sets']) for row in current),
        'reason_has_explicit_final_six_label_sequence': len(explicit),
        'reason_missing_explicit_final_six_label_sequence': len(current) - len(explicit),
        'explicit_final_sequence_equals_gold': sum(row['reason_final_explicit_label_sequence_equals_gold'] for row in explicit),
        'explicit_final_sequence_equals_anchor_order': sum(row['reason_final_explicit_label_sequence_equals_anchor_order'] for row in explicit),
        'turn_only_reason_not_chronological': sum(row['reason_final_explicit_label_sequence'] is None and row['reason_listed_turns_are_chronological'] is False for row in current)}
need(summaries['v86']['draft_count'] == 8 and summaries['v86']['mechanically_admitted'] == 0, 'unexpected_actual_v86_totals')
need(summaries['v86']['formal_first_stage_outcomes_disjoint'] == {'pronoun_policy': 6, 'gold_anchor_mismatch': 1, 'whole_set_duplicate': 1}, 'unexpected_v86_first_rejections')
# Check again immediately before any diagnostic artifact is created.
need(not (BATCH / 'evidence/RUN_COMPLETE.json').exists(), 'already_sealed')
write(OUTPUT, {'version': 'v86_ordering_post_generation_diagnostic_v1', 'created_at_utc': now(),
    'status': 'POST_GENERATION_DIAGNOSIS_ONLY', 'reviewer': '/root/v86_helpers',
    'current_private_writer_and_gold_and_anchor_material_read': True, 'current_solver_results_read': False,
    'source_first_semantic_review_claimed': False, 'full_source_semantic_certification_claimed': False,
    'candidate_or_gold_edits': [], 'production_files_modified': False, 'new_http_calls': 0, 'new_solver_calls': 0,
    'ordering_summary': summaries, 'ordering_drafts': rows,
    'comparison_bindings': [bind(home / 'CONFIG.json') for _, home in batches]
        + [bind(home / 'ordering_generation.py') for _, home in batches]
        + [bind(BATCH / 'evidence/CAPACITY_LAYOUT_MINFIX.json'), bind(Path(__file__))],
    'interpretation': [
        'Observed serialized output order changed from 0/8 to 8/8 with reason before Gold. This says nothing about the model internal reasoning order.',
        'v86 still has 0/8 mechanically admitted. The final-output-order prompt was followed, but reason/Gold/source consistency and actor-name constraints were not reliably followed.',
        'Three v86 drafts have topic-label misalignment. In 112812-01 the fixed-anchor Gold happens to be correct while four item topics are under wrong labels; Gold checks alone cannot certify a usable ordering question.',
        'Four v86 reasons omit the requested final six-label permutation; the two 135466 reasons list item-associated turn numbers rather than a chronological sequence.',
        'Formal first rejection categories are disjoint pipeline outcomes. Secondary Gold, pronoun, label and prior-duplicate diagnostics overlap and must never be added as counts of failed questions.',
        'Topic correspondence is a narrow manual diagnostic. It does not certify actor identity, state, modality, entailment, absence of alternative matching occurrences, or unique semantic chronology.',
        'The four historical TRAIN roots and scenes are shared with v85, but question IDs/random label mappings, prior inventory, prompt order and compact prior metadata differ. This is not a controlled single-variable causal experiment or an independent test.',
        'All raw drafts remain rejected as originally recorded; no automatic Gold repair or question admission was performed.'
    ], 'training_allowed': False, 'automatic_followup': False})
print(json.dumps({'path': str(OUTPUT), 'ordering_summary': summaries}, ensure_ascii=False))
