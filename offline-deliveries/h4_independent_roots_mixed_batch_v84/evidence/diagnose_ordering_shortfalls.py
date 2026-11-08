"""Append a bound, read-only diagnosis of the eight actual v84 ordering drafts."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

BATCH = Path(__file__).resolve().parents[1]
LABELS = tuple(f'I{i}' for i in range(1, 7))
PRONOUNS = re.compile(r'\b(?:he|him|his|himself|she|her|hers|herself)\b', re.I)

# Human comparison of the unchanged writer proposition with the unchanged
# prebound excerpts. These are diagnostic topic matches, not repaired evidence
# and not a new semantic certification of the seven rejected drafts.
MATCHED_ANCHOR_LABELS = {
    ('103866', 1): ['I1', 'I2', 'I3', 'I4', 'I5', 'I6'],
    ('103866', 2): ['I1', 'I2', 'I3', 'I4', 'I5', 'I6'],
    ('109164', 1): ['I5', 'I6', 'I3', 'I1', 'I2', 'I4'],
    ('109164', 2): ['I1', 'I6', 'I3', 'I2', 'I5', 'I4'],
    ('112812', 1): ['I2', 'I6', 'I1', 'I4', 'I3', 'I5'],
    ('112812', 2): ['I1', 'I6', 'I2', 'I4', 'I5', 'I3'],
    ('135466', 1): ['I3', 'I6', 'I2', 'I5', 'I1', 'I4'],
    ('135466', 2): ['I5', 'I1', 'I4', 'I2', 'I6', 'I3'],
}
NOTES = {
    ('103866', 1): 'All six item topics retain their prebound labels. Writer puts I1 (T003) after I2 (T006), so this is a direct Gold permutation error.',
    ('103866', 2): 'All six item topics retain their prebound labels, names satisfy the output policy and writer Gold matches the prebound source order. The separate full-source review accepted this draft.',
    ('109164', 1): 'Writer items are numbered in source chronology instead of retaining the supplied shuffled labels. Gold duplicates I1 and omits I5. Gendered pronouns also occur, but schema rejects first.',
    ('109164', 2): 'Writer items are numbered in source chronology; I2/I4/I6 no longer describe their bound anchors. Gold happens to equal the bound anchor order, which does not validate the item-to-anchor relation. Pronoun gate rejects before Gold comparison.',
    ('112812', 1): 'Writer items are numbered in source chronology; five labels no longer describe their bound anchors. I4 includes she. I1 additionally changes notify us to notify the system; this observation is not a new formal gate reason or a complete semantic re-review.',
    ('112812', 2): 'Writer items are numbered in source chronology; I2/I3/I6 no longer describe their bound anchors. Gold equals the bound anchor order, but that agreement does not validate the mismatched propositions. I6 includes she/her.',
    ('135466', 1): 'Writer items and identity Gold follow their own chronological numbering rather than the frozen label-to-anchor mapping. Changing Gold alone cannot repair the mismatched bound evidence.',
    ('135466', 2): 'Writer items and identity Gold follow their own chronological numbering rather than the frozen label-to-anchor mapping. Changing Gold alone cannot repair the mismatched bound evidence.',
}


def read(path):
    return json.loads(path.read_text())


def binding(path):
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    output = BATCH / 'evidence/ORDERING_SHORTFALL_DIAGNOSIS.json'
    if output.exists():
        raise SystemExit('diagnosis_already_exists_no_overwrite')
    config = read(BATCH / 'CONFIG.json')
    ordering = BATCH / 'ordering'
    generation = read(ordering / 'GENERATION.COMPLETE.json')
    content = read(ordering / 'CONTENT_REVIEW.json')
    evaluation = read(ordering / 'EVALUATION_POOL.json')['questions']
    accepted = read(ordering / 'POOL.json')['questions']
    accepted_ids = {q['candidate_id'] for q in accepted}
    content_ids = {q['candidate_id'] for q in evaluation}
    paths = {Path(__file__).resolve(), BATCH / 'CONFIG.json', BATCH / 'SCENE_PLANS.json',
             BATCH / 'ACTOR_REFERENCE_POLICY.json', BATCH / 'CHARACTER_REGISTRY.json'}
    paths.update(BATCH.glob('*.py'))
    paths.update(ordering / name for name in (
        'GENERATION.COMPLETE.json', 'GENERATION.STARTED.json', 'PREPARATION.COMPLETE.json',
        'POOL.json', 'PUBLIC_QUESTIONS.json', 'SOURCE_JUDGMENTS.json',
        'CONTENT_REVIEW.json', 'CONTENT_REVIEW.COMPLETE.json', 'EVALUATION_POOL.json'))
    slots = []; rows = []; reasons = Counter(); checks = []

    def check(name, condition):
        checks.append({'name': name, 'status': 'PASS' if condition else 'FAIL'})

    for slot in config['ordering_slots']:
        run = ordering / slot['slot_id']
        root_number = slot['root_id'].rsplit('-', 1)[-1]
        source = read(run / 'SOURCE.json')
        plans = read(run / 'ANCHOR_PLAN.json')['questions']
        manifest = read(run / 'MANIFEST.json')
        response_paths = list((run / 'requests').glob('*/RESPONSE.json'))
        check(f'{root_number}_one_response', len(response_paths) == 1)
        if len(response_paths) != 1:
            raise SystemExit('expected_one_actual_response')
        response_path = response_paths[0]
        response = read(response_path); metadata = response['metadata']
        raw = json.loads(response['text'])['questions']
        rejected = read(run / 'REJECTIONS.json')
        rejection_by_id = {r['candidate_id']: r['reasons'] for r in rejected}
        turn_positions = {t['turn_ref']: i for i, t in enumerate(source['turns'])}
        turn_by_ref = {t['turn_ref']: t for t in source['turns']}
        attempts = list((run / 'requests').glob('*/HTTP_ATTEMPT.json'))
        check(f'{root_number}_one_http_no_retry', len(attempts) == 1 and read(attempts[0])['transport_retries'] == 0)
        check(f'{root_number}_complete_stop_response', metadata.get('http_status') == 200
              and metadata.get('returned_model') == 'Qwen/Qwen3.6-35B-A3B'
              and metadata.get('finish_reason') == 'stop')
        check(f'{root_number}_complete_two_slot_coverage', len(raw) == len(plans) == slot['candidate_count'] == 2
              and [r['candidate_id'] for r in raw] == manifest['candidate_ids']
              and [p['candidate_id'] for p in plans] == manifest['candidate_ids'])
        check(f'{root_number}_complete_source_sha', binding(run / 'SOURCE.json')['sha256'] == slot['source_sha256'])
        paths.update(run / name for name in ('SOURCE.json', 'ANCHOR_PLAN.json', 'MANIFEST.json',
                     'REJECTIONS.json', 'POOL.json', 'GENERATION.COMPLETE.json', 'ACTOR_REFERENCE_POLICY.json'))
        paths.update(p for p in response_path.parent.iterdir() if p.is_file())
        slots.append({'root_id': slot['root_id'], 'planned_drafts': slot['candidate_count'],
                      'returned_drafts': len(raw), 'http_attempts': len(attempts),
                      'http_status': metadata['http_status'], 'returned_model': metadata['returned_model'],
                      'finish_reason': metadata['finish_reason'], 'usage': metadata.get('usage'),
                      'mechanically_accepted': len(read(run / 'POOL.json')['questions']),
                      'formally_rejected': len(rejected)})
        for number, (plan, draft) in enumerate(zip(plans, raw), 1):
            qid = draft['candidate_id']; anchors = plan['anchors']
            source_order = sorted(LABELS, key=lambda label: (
                turn_positions[anchors[label]['turn_ref']],
                turn_by_ref[anchors[label]['turn_ref']]['text'].index(anchors[label]['quote'])))
            formal = rejection_by_id.get(qid, [])
            if not formal:
                category = 'accepted'
            elif formal == ['ordering_bound_writer_gold']:
                category = 'invalid_gold_permutation_schema'
            elif formal == ['writer_gold_disagrees_with_source_anchors']:
                category = 'writer_gold_disagrees_with_source_anchors'
            elif all(r.startswith('ordering_actor_reference_policy_forbidden_pronoun:') for r in formal):
                category = 'forbidden_gendered_pronouns'
            else:
                raise SystemExit('unrecognized_actual_formal_reason')
            reasons[category] += 1
            matched = dict(zip(LABELS, MATCHED_ANCHOR_LABELS[root_number, number]))
            pronoun_hits = [{'label': label, 'tokens': sorted({m.group().lower() for m in PRONOUNS.finditer(text)})}
                            for label, text in draft['items'].items() if PRONOUNS.search(text)]
            gold = draft['gold']
            permutation_ok = isinstance(gold, list) and len(gold) == 6 and set(gold) == set(LABELS)
            check(qid + '_accepted_or_formally_rejected_once', (qid in accepted_ids) != bool(formal))
            rows.append({'candidate_id': qid, 'root_id': slot['root_id'], 'scene_id': plan['scene_id'],
                         'original_formal_reasons': formal, 'first_rejection_category': category,
                         'mechanically_accepted': qid in accepted_ids, 'source_content_usable': qid in content_ids,
                         'unchanged_writer_gold': gold, 'gold_is_complete_label_permutation': permutation_ok,
                         'prebound_source_anchor_order': source_order, 'writer_gold_matches_prebound_order': gold == source_order,
                         'unchanged_items': draft['items'], 'unchanged_prebound_anchors': anchors,
                         'diagnostic_topic_mapping_writer_label_to_prebound_anchor_label': matched,
                         'diagnostic_labels_with_wrong_anchor_topic': [label for label in LABELS if matched[label] != label],
                         'diagnostic_forbidden_pronoun_occurrences': pronoun_hits,
                         'diagnostic_note': NOTES[root_number, number],
                         'rejected_draft_semantic_certification_performed': False,
                         'diagnostic_observations_are_additional_formal_reject_reasons': False})
    check('planned_and_returned_eight_drafts', sum(s['planned_drafts'] for s in slots) == len(rows) == 8)
    check('one_mechanical_seven_formal_rejects', len(accepted_ids) == 1 and reasons['accepted'] == 1
          and sum(v for k, v in reasons.items() if k != 'accepted') == 7 and generation['candidate_count'] == 1)
    check('one_full_source_content_usable', len(content_ids) == 1 and content['accepted_count'] == 1 and content['rejected_count'] == 0)
    check('formal_first_reason_counts', reasons == Counter({'accepted': 1,
          'writer_gold_disagrees_with_source_anchors': 3, 'invalid_gold_permutation_schema': 1,
          'forbidden_gendered_pronouns': 3}))
    check('four_actual_ordering_http', sum(s['http_attempts'] for s in slots) == generation['http_attempts'] == 4)
    check('all_eight_diagnostic_topic_mappings_covered', len(MATCHED_ANCHOR_LABELS) == len(rows))
    before = [binding(p) for p in sorted(paths)]
    check('all_inputs_unchanged_while_diagnosing', before == [binding(p) for p in sorted(paths)])
    if any(c['status'] != 'PASS' for c in checks):
        raise SystemExit(json.dumps({'status': 'FAIL', 'checks': checks}, ensure_ascii=False))
    document = {
        'version': 'v84_actual_ordering_shortfall_diagnosis_v1', 'status': 'PASS',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'reviewer': 'v84_source_gate_b', 'scope': 'Read-only diagnosis of the eight actual ordering drafts after source-first review was saved and private finish succeeded.',
        'actual_counts': {'planned_drafts': 8, 'returned_drafts': 8, 'http_attempts': 4,
                         'mechanically_accepted': 1, 'source_content_usable': 1, 'formally_rejected': 7,
                         'mechanical_and_content_yield': 0.125},
        'formal_first_rejection_counts': {k: v for k, v in reasons.items() if k != 'accepted'},
        'production_gate_stage_order': ['ordering_plan.project', 'actor_reference.name_policy_errors',
                                       'native_ordering_gate', 'writer_gold_vs_prebound_anchor_gold', 'prior_and_current_whole_set_dedup'],
        'overlapping_secondary_diagnostics': {
            'drafts_with_label_to_anchor_topic_mismatch': sum(bool(r['diagnostic_labels_with_wrong_anchor_topic']) for r in rows),
            'drafts_with_any_forbidden_pronoun': sum(bool(r['diagnostic_forbidden_pronoun_occurrences']) for r in rows),
            'drafts_with_invalid_gold_permutation': sum(not r['gold_is_complete_label_permutation'] for r in rows),
            'not_a_replacement_for_the_original_first_rejection_counts': True},
        'finding': 'The current low yield is chiefly a writer output-contract adherence problem: six drafts renumber the fixed shuffled item facts into source chronology. Gold-only validation misses this relation error. One additional draft retains the label mapping but returns the wrong Gold permutation. Gendered pronouns and one invalid permutation also reject drafts. All four HTTP responses completed normally; this is not a transport or truncation shortfall.',
        'diagnostic_limits': [
            'The seven rejected drafts did not undergo a new complete independent source-content certification; topic maps explain contract failures and do not certify all meanings, actors, modality or alternative occurrences.',
            'A draft rejected at schema or pronoun stage must not be described as having passed the later semantic, Gold or dedup gates.',
            'Unknown source gender was preserved. The output policy does not infer gender from names.',
            'Results refer to these four known training roots and are not an independent test or proof of stable large-batch yield.'],
        'proposed_next_batch_fix': {
            'implemented_in_current_batch': False,
            'observation': 'Keep the frozen source, anchor and Gold review. Have the writer check each supplied I-label against its own specified excerpt before returning the proposition; explicitly preserve that mapping instead of renumbering items into chronology. Retain complete permutation and pronoun checks. This is a proposal for a separately scoped next batch, not a rule change or repair of this batch.',
            'gold_or_pronoun_edits_alone_repair_label_anchor_mismatch': False},
        'actions': {'production_or_source_edits': 0, 'draft_gold_or_pronoun_edits': 0,
                    'new_generation_http': 0, 'new_model_calls': 0, 'additional_training': 0},
        'roots': slots, 'drafts': rows, 'inputs': before,
        'check_count': len(checks), 'failed_checks': [], 'checks': checks}
    with output.open('x') as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2); stream.write('\n')
    if before != [binding(p) for p in sorted(paths)]:
        raise SystemExit('inputs_changed_after_diagnosis_write')
    print(json.dumps({'status': 'PASS', 'path': str(output), 'check_count': len(checks),
                      'actual_counts': document['actual_counts'],
                      'formal_first_rejection_counts': document['formal_first_rejection_counts'],
                      'overlapping_secondary_diagnostics': document['overlapping_secondary_diagnostics']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
