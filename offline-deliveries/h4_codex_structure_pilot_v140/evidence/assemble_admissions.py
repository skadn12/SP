"""Content-only admission after saved public reconstructions; never reads Solver results."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

B = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads(Path(path).read_text())

def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def rows(review):
    return review.get('question_reviews', review.get('questions', []))

def verdict(review):
    return review.get('verdict', review.get('status'))

def main():
    assert not (B / 'evaluation/RUN_STARTED.json').exists(), 'No admission after inference starts'
    assert not (B / 'evaluation/EVALUATION_POOL.json').exists(), 'No admission after evaluation freeze'
    plan = read(B / 'PLAN.json')
    originals = read(B / 'private/ALL_ORIGINAL_QUESTIONS.json')['questions']
    frozen = read(B / 'evidence/ORIGINAL_DRAFT_FREEZE.json')
    for bind in frozen['files']:
        assert sha(B / bind['path']) == bind['sha256'], 'Original input changed: ' + bind['path']
    reviews = {}
    for side in ['A', 'B']:
        for x in rows(read(B / f'reviews/REVIEW_{side}_RECONSTRUCTION_v1.json')):
            assert x['candidate_id'] not in reviews
            reviews[x['candidate_id']] = x
    assert len(originals) == len(reviews) == plan['planned_slots'] == 32
    revised, revision_reviews = {}, {}
    for p in sorted((B / 'drafts').glob('*_PRIVATE_v2.json')):
        for q in read(p)['questions']:
            sid = q['slot_id']
            assert sid not in revised, 'Only one revision permitted per slot'
            assert q['revises_candidate_id'] == sid and q['candidate_id'] == sid + '-r1'
            revised[sid] = q
    for side in ['A', 'B']:
        p = B / f'reviews/REVIEW_{side}_RECONSTRUCTION_v2.json'
        if p.exists():
            for x in rows(read(p)):
                assert x['candidate_id'] not in revision_reviews
                revision_reviews[x['candidate_id']] = x
    spec = importlib.util.spec_from_file_location('v140_admission_check', B / 'evidence/admission_gate.py')
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    admitted, decisions, comparisons = [], [], []
    for original in originals:
        sid = original['candidate_id']
        initial = reviews[sid]
        need_revision = verdict(initial) != 'PASS'
        assert (sid in revised) == need_revision, 'Revision must follow saved content finding: ' + sid
        chosen = copy.deepcopy(revised.get(sid, original))
        final_review = revision_reviews.get(chosen['candidate_id']) if need_revision else initial
        assert final_review is not None, 'Missing public-only revision review: ' + sid
        for key in ['gold', 'source_path', 'source_sha256', 'root_id', 'question_type', 'order_anchor_spans']:
            assert chosen.get(key) == original.get(key), 'No silent answer/anchor change:' + sid + ':' + key
        errors, facts = gate.validate(chosen)
        agree = final_review['predicted_gold'] == chosen['gold']
        passed = verdict(final_review) == 'PASS' and agree and not errors
        # A primary permutation alone is insufficient when alternatives remain.
        if final_review.get('predicted_gold_is_unique') is False:
            passed = False
        chosen['slot_id'] = sid
        chosen['split'] = 'dev'
        chosen['admission_status'] = 'PASS' if passed else 'REJECT'
        chosen['content_review_id'] = final_review['candidate_id']
        chosen['content_review_is_internal'] = True
        if passed:
            admitted.append(chosen)
        decisions.append({'slot_id': sid, 'root_id': chosen['root_id'],
                          'question_type': chosen['question_type'], 'pilot_arm': chosen['pilot_arm'],
                          'original_candidate_id': sid, 'original_review_verdict': verdict(initial),
                          'revision_attempts': int(need_revision), 'final_candidate_id': chosen['candidate_id'],
                          'final_review_verdict': verdict(final_review),
                          'independent_gold_agreement': agree, 'mechanical_errors': errors,
                          'admission_status': 'PASS' if passed else 'REJECT', 'facts': facts})
        comparisons.append({'slot_id': sid, 'candidate_id': chosen['candidate_id'],
                            'author_gold': chosen['gold'], 'reconstructed_gold': final_review['predicted_gold'],
                            'gold_agreement': agree, 'review_verdict': verdict(final_review)})
    # No performance data can enter this procedure. All content-pass items stay.
    validation = {'status': 'PASS', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                  'planned_slots': 32, 'admitted_count': len(admitted), 'rejected_count': 32 - len(admitted),
                  'planned_new_slots': 24, 'planned_old_control_slots': 8,
                  'original_pass_count': sum(verdict(r) == 'PASS' for r in reviews.values()),
                  'original_needs_revision_count': sum(verdict(r) != 'PASS' for r in reviews.values()),
                  'revised_slot_count': len(revised), 'max_revisions_per_slot': 1,
                  'no_answer_or_order_anchor_changes': True, 'solver_results_read': False,
                  'all_content_pass_retained': True, 'external_blind_review_claimed': False,
                  'admitted_by_arm': dict(Counter(q['pilot_arm'] for q in admitted)),
                  'admitted_by_type': dict(Counter(q['question_type'] for q in admitted)),
                  'decisions': decisions,
                  'limitations': ['Public reconstructions are separate internal tasks sharing model and filesystem.',
                                  'Some original wrong items have plausible alternative readings even though the primary reconstructed Gold agrees.',
                                  'New structure questions reuse source facts and some original anchors; these are correlated question families.',
                                  'A-author ordering items use direct quotations; within-scene comparisons can still reduce to lexical retrieval.',
                                  'Canonical ordering label patterns repeat, but both models receive Gold-blind randomized display labels.',
                                  'Author evidence-minimality claims are annotation analysis, not evidence-deletion model experiments.']}
    save(B / 'private/ADMITTED_QUESTIONS.json', {'questions': admitted})
    public_keys = ['candidate_id', 'root_id', 'question_type', 'question', 'options', 'items',
                   'source_path', 'source_sha256', 'chronology_kind', 'split']
    save(B / 'public/QUESTIONS.json', {'questions': [{k: q[k] for k in public_keys if k in q} for q in admitted]})
    save(B / 'reviews/FINAL_PRIVATE_COMPARISON.json', {'comparison_actor': '/root',
         'performed_after_public_reconstructions_saved': True, 'questions': comparisons})
    save(B / 'evidence/INPUT_VALIDATION.json', validation)
    print(json.dumps({k: validation[k] for k in ['status', 'planned_slots', 'admitted_count', 'rejected_count',
                      'original_pass_count', 'revised_slot_count', 'admitted_by_type']}, ensure_ascii=False))

if __name__ == '__main__':
    main()
