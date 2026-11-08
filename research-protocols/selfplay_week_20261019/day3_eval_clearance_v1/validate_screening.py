#!/usr/bin/env python3
"""Verify screening provenance, concrete source witnesses and candidate accounting.

This cannot certify a semantic review, new-family eligibility or evaluation readiness.
"""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
WEEK = OUT.parent
PREP = WEEK / 'day3_common_evaluation_v1'
CHECKS = ('train_new_replay_basic_family', 'train_diagnostic_feedback_related_family',
          'historical_dev_test_v140_family', 'cross_root_cross_split_semantic_near_duplicate',
          'within_locked_eval_fact_overlap_and_revision')
KEY_FIELDS = {'source_sha256', 'normalized_semantic_fact_tuples', 'evidence_turn_refs',
              'explicit_scope', 'mention_rule'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def main():
    failures = []
    checks = 0

    def check(condition, code, detail):
        nonlocal checks
        checks += 1
        if not condition:
            failures.append({'code': code, 'detail': detail})

    manifest = json.loads((OUT / 'INPUT_MANIFEST.json').read_text())
    for ref in [manifest['policy'], manifest['source_lock'], manifest['exclusion_index'],
                manifest['exclusion_manifest']] + [g['packet'] for g in manifest['groups']]:
        check(digest(Path(ref['path'])) == ref['sha256'], 'INPUT_HASH', ref['path'])
    exclusions = {r['exclusion_record_id']: r for r in
                  map(json.loads, (PREP / 'author_release/EXCLUSION_INDEX.jsonl').read_text().splitlines())}
    # Local eval-family conflicts refer to actual eval plan IDs, not historical
    # exclusion IDs. Both namespaces are valid only for the fifth check.
    eval_plan_ids = set()
    for group in manifest['groups']:
        path = OUT / 'screening' / (group['group'] + '.json')
        if path.exists():
            eval_plan_ids.update(p['plan_id'] for s in json.loads(path.read_text())['slots']
                                 for p in s['plans'])
    slots_seen, plans_seen, groups, source_refs = [], [], [], {}
    pending_files = []
    for group in manifest['groups']:
        name = group['group']
        packet = json.loads(Path(group['packet']['path']).read_text())
        expected = {s['slot_id']: (root, s) for root in packet['roots']
                    for s in root['supported_slots']}
        for root in packet['roots']:
            source_refs[root['root_id']] = {t['turn_ref']: t for t in root['complete_source']['turns']}
        path = OUT / 'screening' / (name + '.json')
        if not path.exists():
            pending_files.append(str(path))
            continue
        data = json.loads(path.read_text())
        rows = data['slots']
        check(len(rows) == len(expected), 'GROUP_SLOT_COUNT', name)
        check({r['slot_id'] for r in rows} == set(expected), 'EXACT_GROUP_SLOT_SET', name)
        group_plan_count = 0
        for row in rows:
            sid = row['slot_id']
            slots_seen.append(sid)
            if sid not in expected:
                continue
            root, slot = expected[sid]
            source_hash = root['source_binding']['sha256']
            check(row['root_id'] == root['root_id'], 'SLOT_ROOT', sid)
            check(row['question_type'] == slot['question_type'], 'SLOT_TYPE', sid)
            check(row['source_sha256'] == source_hash, 'SLOT_SOURCE', sid)
            plans = row['plans']
            check(len(plans) <= 3, 'PLAN_LIMIT', sid)
            check(bool(plans) or bool(row.get('rationale')), 'EMPTY_PLAN_REASON', sid)
            for plan in plans:
                pid = plan['plan_id']
                plans_seen.append(pid)
                group_plan_count += 1
                payload = plan['candidate_key_payload']
                check(set(payload) == KEY_FIELDS, 'KEY_FIELDS', pid)
                check(payload['source_sha256'] == source_hash, 'PLAN_SOURCE', pid)
                check(hashlib.sha256(canonical(payload)).hexdigest() == plan['candidate_selection_key'],
                      'CANONICAL_CANDIDATE_SHA', pid)
                refs = payload['evidence_turn_refs']
                turns = source_refs[root['root_id']]
                check(bool(refs) and all(r in turns for r in refs), 'PLAN_REFS', pid)
                witnesses = plan['source_witnesses']
                check(bool(witnesses), 'SOURCE_WITNESSES_PRESENT', pid)
                for witness in witnesses:
                    ref = witness['turn_ref']
                    quote = witness['quote']
                    check(ref in turns and bool(quote) and quote in turns.get(ref, {}).get('text', ''),
                          'EXACT_SOURCE_QUOTE', {'plan_id': pid, 'ref': ref})
                check(set(plan['family_checks']) == set(CHECKS), 'FIVE_CHECK_FIELDS', pid)
                for category, review in plan['family_checks'].items():
                    check(isinstance(review, dict) and bool(review.get('status')),
                          'CHECK_STATUS', {'plan_id': pid, 'category': category})
                    for field in ('compared_record_ids', 'matched_or_related_record_ids'):
                        ids = review.get(field, [])
                        allowed = set(exclusions)
                        if category == 'within_locked_eval_fact_overlap_and_revision':
                            allowed |= eval_plan_ids
                        check(isinstance(ids, list) and all(r in allowed for r in ids),
                              'EXCLUSION_RECORD_IDS', {'plan_id': pid, 'category': category, 'field': field})
                    check(bool(review.get('rationale')),
                          'CHECK_RATIONALE', {'plan_id': pid, 'category': category})
                check('gold' not in plan and 'public_question' not in plan,
                      'NO_EARLY_PUBLIC_QUESTION_OR_GOLD', pid)
        groups.append({'group': name, 'slots': len(rows), 'plans': group_plan_count,
                       'path': str(path), 'sha256': digest(path)})
    if not pending_files:
        check(len(slots_seen) == len(set(slots_seen)) == 60, 'ALL60_UNIQUE_SLOTS', len(slots_seen))
        check(len(plans_seen) == len(set(plans_seen)), 'UNIQUE_PLAN_IDS', len(plans_seen))
    report = {
        'schema_version': 'finite_family_screening_mechanical_validation_v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'PENDING_FILES' if pending_files else
                  ('PASS_MECHANICAL_ONLY' if not failures else 'FAIL'),
        'checks': checks, 'failures': failures, 'pending_files': pending_files,
        'groups': groups, 'slots': len(slots_seen), 'plans': len(plans_seen),
        'scope': 'SHA/provenance, original slot/source identity, finite plan limit, exact quotes, concrete comparison IDs and complete status fields only. Semantic judgments and new-family eligibility are not mechanically certified.',
        'new_public_evaluation_questions': 0, 'new_solver_calls': 0, 'new_training_runs': 0,
    }
    (OUT / 'VALIDATION.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('groups', 'scope')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
