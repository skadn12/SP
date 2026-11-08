"""Apply the user's choice acceptance policy to existing program outputs.

No generation, answer correction, or choice semantic/evidence acceptance gate.
Existing within-batch groups and representatives remain stable. Newly restored
groups use the first program-accepted member. Historical exclusion uses only
the locked formal previous-round diagnosis, never current-round test history.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
OLD = BASE.parent / 'h4_qa_feedback_constraints_v2'
SCOPE = BASE.parent / 'h4_qa_feedback_scope_correction_v1'
GROUPS = {
    '259192': [
        ('EMILY_GYM_ARRANGEMENT', ['I002-C-Q01:primary', 'I003-C-Q01:primary'], 'I003-C-Q01:primary'),
        ('DAVID_GYM_OFFER', ['I002-C-Q02:primary', 'I003-C-Q02:primary'], 'I002-C-Q02:primary'),
        ('R02_FINALIZED_ARRANGEMENTS', ['I002-C-Q03:primary', 'I003-C-Q03:primary'], 'I003-C-Q03:primary'),
    ],
    '96462': [
        ('MATH_SMALL_GOALS', ['I001-C-Q01:repair', 'I002-C-Q01:primary', 'I003-C-Q01:primary'], 'I001-C-Q01:repair'),
        ('COURSE_LOCATION_TIME', ['I001-C-Q02:repair', 'I002-C-Q02:primary', 'I003-C-Q02:primary'], 'I001-C-Q02:repair'),
        ('PROSOCIAL_BEHAVIOR_SET', ['I003-C-Q01:repair'], 'I003-C-Q01:repair'),
    ],
    '112812': [
        ('NOTIFICATION_TEST', [f'I{i:03d}-C-Q01:primary' for i in range(1, 4)], 'I001-C-Q01:primary'),
        ('EMILY_COMMUNICATION_STRATEGY', [f'I{i:03d}-C-Q02:primary' for i in range(1, 4)], 'I001-C-Q02:primary'),
        ('BUDGET_IMPLEMENTATION_SET', [f'I{i:03d}-C-Q03:primary' for i in range(1, 4)], 'I001-C-Q03:primary'),
    ],
}


def read(path):
    return json.loads(path.read_text())


def artifact(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    roots = []
    for code, groups in GROUPS.items():
        root = 'cc-train-episode-' + code
        pool_path = OLD / 'prepared_run/live/roots' / root / 'candidate_pool.json'
        pool = read(pool_path)
        choices = [q for q in pool if q['question_type'] in ('single_choice', 'multi_select')]
        by_id = {q['candidate_id']: q for q in choices}
        coverage, representatives, decisions = [], [], []
        old_reps_path = OLD / 'filtered' / root / 'strict_pass_dedup.json'
        old_reps = {q['candidate_id'] for q in read(old_reps_path)}
        for key, suffixes, selected in groups:
            ids = [root + ':' + suffix for suffix in suffixes]
            representative = root + ':' + selected
            assert representative in ids and all(cid in by_id for cid in ids)
            assert len({by_id[cid]['question_type'] for cid in ids}) == 1
            coverage.extend(ids)
            representatives.append(deepcopy(by_id[representative]))
            decisions.append({'group': key, 'candidate_ids': ids, 'representative': representative,
                'selection_rule': 'PRESERVE_EXISTING_REPRESENTATIVE' if representative in old_reps else 'FIRST_PROGRAM_ACCEPTED_MEMBER_OF_NEWLY_RESTORED_GROUP',
                'excluded_as_duplicate': [cid for cid in ids if cid != representative],
                'choice_correctness_or_evidence_used_as_new_gate': False})
        assert Counter(coverage) == Counter(by_id.keys())
        novelty_path = SCOPE / 'reports' / ('NOVELTY_' + code + '.json')
        novelty_report = read(novelty_path)
        novelty = {r['candidate_id']: r['previous_formal_novelty'] for r in novelty_report.get(
            'candidate_reviews', novelty_report.get('raw_candidate_reviews'))}
        new = [q for q in representatives if novelty[q['candidate_id']]['verdict'] == 'NEW']
        directory = BASE / 'choices' / root
        write(directory / 'program_accepted.json', choices)
        write(directory / 'within_batch_deduplicated.json', representatives)
        write(directory / 'new_vs_previous_formal.json', new)
        write(directory / 'DEDUP_AUDIT.json', {'root_id': root, 'groups': decisions,
            'historical_scope': 'h4_unexpanded122_qa_generation_only_v1',
            'novelty_decisions': {q['candidate_id']: novelty[q['candidate_id']] for q in representatives},
            'question_gold_evidence_modified': False,
            'representative_selection_note': 'Existing representatives selected in the older review are kept stable; newly restored groups use their first program-accepted member. This is not a fresh semantic quality selection.'})
        roots.append({'root_id': root, 'program_choices': len(choices),
            'program_choice_types': dict(Counter(q['question_type'] for q in choices)),
            'within_batch_choices': len(representatives),
            'within_batch_types': dict(Counter(q['question_type'] for q in representatives)),
            'new_choices': len(new), 'new_choice_types': dict(Counter(q['question_type'] for q in new)),
            'excluded_within_batch': len(choices) - len(representatives),
            'excluded_previous_formal': len(representatives) - len(new),
            'source_artifacts': [artifact(pool_path), artifact(novelty_path), artifact(old_reps_path)]})
    write(BASE / 'CHOICE_SUMMARY.json', {'status': 'OFFLINE_RESELECTION_COMPLETE', 'roots': roots,
        'choice_semantic_gate': False, 'choice_evidence_completeness_gate': False,
        'scope': 'Previously program-accepted candidates only; existing program schema/ID rejections are not silently reversed.',
        'production_parser_or_prompts_modified': False,
        'historical_diagnoses_deleted': False, 'new_external_calls': 0})
    print(json.dumps({'program_choices': sum(r['program_choices'] for r in roots),
        'within_batch_choices': sum(r['within_batch_choices'] for r in roots),
        'new_choices': sum(r['new_choices'] for r in roots), 'new_external_calls': 0}))


if __name__ == '__main__':
    main()
