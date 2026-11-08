#!/usr/bin/env python3
"""Mechanical admission of source-first independently reviewed real content."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTENT = HERE / 'content'
CHECKS = ('train_new_replay_basic_family', 'train_diagnostic_feedback_related_family',
          'historical_dev_test_v140_family', 'cross_root_cross_split_semantic_near_duplicate',
          'within_locked_eval_fact_overlap_and_revision')


def bind(path):
    data = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def main():
    selection_path = HERE / 'FAMILY_SELECTION.json'
    selection = json.loads(selection_path.read_text())
    qa_path = CONTENT / 'qa/FINAL_CONTENT_REVIEW.json'
    qa = json.loads(qa_path.read_text())
    reconstruction_path = Path(qa['source_first_reconstruction']['path'])
    seal_path = Path(qa['first_pass_seal']['path'])
    assert bind(reconstruction_path)['sha256'] == qa['source_first_reconstruction']['sha256']
    assert bind(seal_path)['sha256'] == qa['first_pass_seal']['sha256']
    reconstruction = json.loads(reconstruction_path.read_text())
    seal = json.loads(seal_path.read_text())
    assert seal['reconstruction_sha256'] == bind(reconstruction_path)['sha256']
    assert seal['chronology']['reconstruction_written_and_hash_verified_before_private_read'] is True
    assert seal['chronology']['reconstruction_modified_after_private_read'] is False
    assert (datetime.fromisoformat(reconstruction['created_at_utc'])
            <= datetime.fromisoformat(seal['chronology']['first_private_read_before_utc'])
            <= datetime.fromisoformat(qa['created_at_utc']))
    first_pass = {r['candidate_id']: r for r in reconstruction['items']}
    reviews = {r['candidate_id']: r for r in qa['items']}
    assert len(reviews) == len(qa['items']) == len(selection['selected']) == 4
    assert set(reviews) == {s['slot_id'] for s in selection['selected']}
    assert set(first_pass) == set(reviews) and len(first_pass) == len(reconstruction['items'])
    rows, errors, mechanical_checks = [], [], 0

    def check(ok, code, candidate_id):
        nonlocal mechanical_checks
        mechanical_checks += 1
        if not ok:
            errors.append({'candidate_id': candidate_id, 'code': code})

    for selected in selection['selected']:
        cid = selected['slot_id']
        public_path = CONTENT / 'public' / (cid + '.json')
        private_path = CONTENT / 'private' / (cid + '.json')
        public = json.loads(public_path.read_text())
        private = json.loads(private_path.read_text())
        review = reviews[cid]
        source = json.loads(Path(public['source_path']).read_text())
        turns = {t['turn_ref']: t for t in source['turns']}
        indices = {t['turn_ref']: i for i, t in enumerate(source['turns'], 1)}
        scenes = set(private['explicit_scope']['scene_refs'])
        check(public['candidate_id'] == private['candidate_id'] == cid, 'CONTENT_ID', cid)
        check(public['source_sha256'] == private['source_sha256'] == selected['source_sha256']
              == bind(Path(public['source_path']))['sha256'], 'SOURCE_SHA', cid)
        check(private['public_question_sha256'] == bind(public_path)['sha256'], 'PUBLIC_PRIVATE_BINDING', cid)
        check(private['candidate_selection_key'] == selected['selected_plan']['candidate_selection_key'], 'SELECTED_KEY', cid)
        check(private['explicit_scope'] == selected['selected_plan']['candidate_key_payload']['explicit_scope'], 'LOCKED_SCOPE', cid)
        check(private['mention_rule'] == selected['selected_plan']['candidate_key_payload']['mention_rule'], 'LOCKED_MENTION_RULE', cid)
        check(private['normalized_semantic_fact_tuples']
              == [r['canonical_tuple'] for r in private['canonical_fact_records']]
              == selected['selected_plan']['candidate_key_payload']['normalized_semantic_fact_tuples'], 'LOCKED_FACTS', cid)
        check(private['candidate_key_payload'] == selected['selected_plan']['candidate_key_payload'],
              'COMPLETE_SELECTED_PAYLOAD', cid)
        check(set(private['family_checks']) == set(CHECKS)
              and all(private['family_checks'][k]['status'] == 'PASS' for k in CHECKS), 'ALL_FIVE_FAMILY_CHECKS', cid)
        gold = private['gold']
        check(isinstance(gold, list) and len(gold) == len(set(gold)) == selected['design_target']['gold_count_target']
              and set(gold) <= set(public['options']), 'GOLD_SET_AND_DESIGN', cid)
        option_evidence = private['option_evidence']
        check(len(option_evidence) == len(public['options'])
              and {x['label'] for x in option_evidence} == set(public['options']), 'EVERY_OPTION_EVIDENCE', cid)
        for option in option_evidence:
            label = option['label']
            check(option['option_text'] == public['options'][label]
                  and option['is_gold'] == (label in gold), 'OPTION_TEXT_AND_GOLD:' + label, cid)
            spans = option['exact_evidence']
            check(bool(spans) and set(option['evidence_turn_refs']) == {s['turn_ref'] for s in spans},
                  'OPTION_EVIDENCE_REFS:' + label, cid)
            for span in spans:
                turn = turns.get(span['turn_ref'])
                check(turn is not None, 'EVIDENCE_TURN_EXISTS:' + label, cid)
                if turn is None:
                    continue
                start, end = span['char_start'], span['char_end']
                check(type(start) is int and type(end) is int and 0 <= start < end <= len(turn['text'])
                      and turn['text'][start:end] == span['quote'], 'EXACT_UNICODE_SPAN:' + label, cid)
                check(span['scene_ref'] == turn['scene_ref'] and turn['scene_ref'] in scenes
                      and span['speaker'] == turn['speaker']
                      and span['source_turn_index_1_based'] == indices[span['turn_ref']],
                      'EVIDENCE_LOCATION_AND_SCOPE:' + label, cid)
        # Source-first QA is required to explain how every option remains within
        # the selected family, including negative distractor transformations.
        check(review['status'] in ('CONTENT_ACCEPTED', 'CONTENT_HOLD'), 'KNOWN_QA_STATUS', cid)
        check(review['public_question_sha256'] == bind(public_path)['sha256'], 'QA_PUBLIC_BINDING', cid)
        check(review['author_private_sha256'] == bind(private_path)['sha256'], 'QA_PRIVATE_BINDING', cid)
        accepted = review['status'] == 'CONTENT_ACCEPTED'
        check(not accepted or review['family_scope_preserved'] == 'PASS', 'ALL_OPTION_FAMILY_SCOPE', cid)
        check(not accepted or (review['independent_gold_match'] is True and review['unique_answer'] is True),
              'INDEPENDENT_ANSWER_GATE', cid)
        check(not accepted or (set(first_pass[cid]['answer']) == set(gold)
              and first_pass[cid]['uniqueness']['unique_answer'] is True), 'SEALED_FIRST_PASS_ANSWER', cid)
        check(first_pass[cid]['public_question_sha256'] == bind(public_path)['sha256'],
              'SEALED_FIRST_PASS_PUBLIC_BINDING', cid)
        rows.append({'candidate_id': cid, 'root_id': selected['root_id'], 'original_split': selected['split'],
                     'question_type': selected['question_type'], 'status': review['status'],
                     'public_question': bind(public_path), 'private_author_record': bind(private_path),
                     'five_family_checks': 'PASS',
                     'independent_source_first_qa': 'PASS' if accepted else 'HOLD',
                     'independent_review': bind(qa_path)})
    assert not errors, errors
    report = {'schema_version': 'common_eval_actual_content_admission_v1',
              'created_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'CONTENT_ADMISSION_FINAL',
              'family_selection': bind(selection_path), 'independent_review': bind(qa_path),
              'source_first_reconstruction': bind(reconstruction_path), 'first_pass_seal': bind(seal_path),
              'chronology_limit': 'The reconstruction itself was written and its hash reported before private reads; the separate seal manifest was created afterwards and records that sequence. Exact first private-read tool timestamp was not captured.',
              'mechanical_checks': mechanical_checks, 'mechanical_errors': errors,
              'items': rows, 'historical_role': 'REGRESSION_ONLY',
              'accepted_count': sum(r['status'] == 'CONTENT_ACCEPTED' for r in rows),
              'held_count': sum(r['status'] != 'CONTENT_ACCEPTED' for r in rows),
              'scope': 'These four actual questions only. Sparse MC coverage does not establish readiness for a multi-type A/B effect comparison.',
              'new_solver_calls': 0, 'new_training_runs': 0}
    path = CONTENT / 'CONTENT_ADMISSION.json'
    assert not path.exists()
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['status', 'accepted_count', 'held_count', 'mechanical_checks']}))


if __name__ == '__main__':
    main()
