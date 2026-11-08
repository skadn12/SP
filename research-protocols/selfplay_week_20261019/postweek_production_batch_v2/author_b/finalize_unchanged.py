#!/usr/bin/env python3
"""Freeze the independently passed author draft without content changes."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
from scripts.current_method.production_question_gate import file_binding, fingerprint, validate_review


def main():
    draft_path = HERE / 'CANDIDATES_PRIVATE_DRAFT.json'
    final_path = HERE / 'CANDIDATES_PRIVATE.json'
    log_path = HERE / 'REPAIR_LOG.json'
    initial_path = HERE.parent / 'review_b/INITIAL_CONTENT_REVIEW_AI.json'
    selection_path = HERE.parent / 'SELECTION_FREEZE.json'
    initial = json.loads(initial_path.read_text())
    data = draft_path.read_bytes()
    drafts = json.loads(data)
    reviews = {r['candidate_id']: r for r in initial['reviews']}
    assert len(drafts) == len(reviews) == 33
    assert set(reviews) == {c['candidate_id'] for c in drafts}
    assert all(r['decision'] == 'PASS' for r in reviews.values())
    assert not final_path.exists() and not log_path.exists()
    candidate_hashes = []
    for candidate in drafts:
        review = reviews[candidate['candidate_id']]
        digest = fingerprint(candidate)
        assert review['candidate_sha256'] == digest
        assert validate_review(candidate, review, selection_path)['admitted']
        candidate_hashes.append({'candidate_id': candidate['candidate_id'],
                                 'before_candidate_sha256': digest,
                                 'after_candidate_sha256': digest,
                                 'gold_unchanged': True,
                                 'task_operation_unchanged': True})
    with final_path.open('xb') as stream:
        stream.write(data)
    final_path.chmod(0o600)
    assert final_path.read_bytes() == data
    final = json.loads(final_path.read_text())
    assert len(final) == len(drafts)
    assert all(fingerprint(a) == fingerprint(b) and a['gold'] == b['gold']
               and a['task_operation'] == b['task_operation']
               for a, b in zip(drafts, final))
    log = {'schema_version': 'production_author_single_preexecution_repair_v1',
           'created_at_utc': datetime.now(timezone.utc).isoformat(),
           'author_id': '/root/batch1_full_reason_review_a',
           'repair_round': 0, 'maximum_repair_rounds': 1,
           'repair_performed_before_solver': False,
           'reason': 'All 33 submitted candidates passed independent initial review; final file is a byte-identical copy of the preserved draft.',
           'initial_review_binding': file_binding(initial_path),
           'draft_binding': file_binding(draft_path),
           'final_binding': file_binding(final_path),
           'repairs': [], 'unchanged_candidate_count': 33,
           'unchanged_candidate_hashes': candidate_hashes,
           'byte_identical_draft_and_final': True,
           'original_draft_and_initial_review_unchanged': True,
           'post_solver_rewrite': False, 'new_solver_calls': 0,
           'independent_final_review_binding': 'PENDING_REVIEWER_FINALIZATION'}
    with log_path.open('x', encoding='utf-8') as stream:
        json.dump(log, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    log_path.chmod(0o600)
    print(json.dumps({'unchanged_candidates': 33, 'repairs': 0,
                      'byte_identical': True, 'final_binding': file_binding(final_path)}))


if __name__ == '__main__':
    main()
