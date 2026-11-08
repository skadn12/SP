"""Export actual audited non-control selections and union with v83 inventory."""
from collections import Counter
from pathlib import Path
import sys
HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOME))
from common import read, write, bind, now
for name in ['RESULT_AUDIT.json', 'INPUT_AUDIT.json']:
    assert read(HOME / 'evidence' / name)['status'] == 'PASS', name
generation = read(HOME / 'evidence/GENERATION_AUDIT.json')
assert generation['status'] in ('PASS', 'PASS_WITH_GENERATION_AND_CONTENT_SHORTFALLS')
config = read(HOME / 'CONFIG.json')
import source_admission, source_subset
source_admission.verify(); source_subset.validate(config)
selected = read(HOME / 'SELECTED_CANDIDATES.json')['questions']
new = [q for q in selected if q['question_type'] == 'ordering' or len(q['gold']) < 6]
controls = [q for q in selected if q not in new]
def counts(rows):
    return {'by_type': dict(Counter(q['question_type'] for q in rows)),
            'type_distribution': dict(Counter(q['question_type'] for q in rows)),
            'multiselect_gold_distribution': {str(g): sum(q['question_type'] == 'multi_select' and len(q['gold']) == g for q in rows) for g in range(2, 6)}}
write(HOME / 'AUDITED_NEW_CANDIDATES.json', {
    'created_at_utc': now(), 'status': 'CONTENT_REVIEW_SOLVER_SELECTION_AND_INDEPENDENT_AUDIT_PASS',
    'questions': new, 'new_question_count': len(new), **counts(new),
    'gold_distribution': counts(new)['multiselect_gold_distribution'],
    'role': 'new_multiselect_combinations_and_fixed_scene_ordering_from_existing_sources',
    'content_review': bind(HOME / 'BATCH_REVIEW.COMPLETE.json'),
    'source_admission': bind(HOME / 'SOURCE_ADMISSION.COMPLETE.json'),
    'source_root_count': len(config['multiselect_config']['slots']),
    'source_split': 'first_round_known_train_roots', 'independent_test': False,
    'paired_report': bind(HOME / 'MIXED_EVALUATION_REPORT.json'),
    'independent_result_audit': bind(HOME / 'evidence/RESULT_AUDIT.json'),
    'independent_generation_audit': bind(HOME / 'evidence/GENERATION_AUDIT.json'),
    'character_registry': bind(HOME / 'CHARACTER_REGISTRY.json'),
    'actor_reference_policy': bind(HOME / 'ACTOR_REFERENCE_POLICY.json'),
    'historical_controls_excluded': [q['candidate_id'] for q in controls],
    'training_allowed': False, 'training_started': False, 'formal_training_admission': False, 'merged_into_baseline': False,
    'review_limitation': 'Reviewers save actual source judgments before current private Gold; actual historical exposure remains disclosed in bound source records.'})
previous = HOME.parent / 'h4_name_policy_mixed_batch_v83'
assert read(previous / 'evidence/RESULT_AUDIT.json')['status'] == 'PASS'
assert read(previous / 'evidence/UNION_AUDIT.json')['status'] == 'PASS'
pools = [previous / 'CUMULATIVE_AUDITED_NEW_CANDIDATES.json', HOME / 'AUDITED_NEW_CANDIDATES.json']
all_rows = []; members = []; seen = {}; duplicates = []
for source in pools:
    data = read(source); assert data['training_allowed'] is False
    for q in data['questions']:
        assert q['question_type'] == 'ordering' or len(q['gold']) < 6
        field = 'options' if q['question_type'] == 'multi_select' else 'items'
        key = (q['root_id'], q['question_type'], tuple(sorted(' '.join(t.casefold().split()) for t in q[field].values())))
        if key in seen:
            duplicates.append({'candidate_id': q['candidate_id'], 'matches': seen[key]}); continue
        seen[key] = q['candidate_id']; all_rows.append(q)
        members.append({'candidate_id': q['candidate_id'], 'source': str(source), 'root_id': q['root_id'], 'gold_count': len(q['gold'])})
write(HOME / 'CUMULATIVE_AUDITED_NEW_CANDIDATES.json', {
    'created_at_utc': now(), 'status': 'DEDUPLICATED_CANDIDATE_INVENTORY', 'questions': all_rows,
    'candidate_count': len(all_rows), **counts(all_rows),
    'gold_distribution': counts(all_rows)['multiselect_gold_distribution'],
    'root_distribution': dict(Counter(q['root_id'] for q in all_rows)), 'source_pools': [bind(p) for p in pools],
    'independent_batch_audits': [bind(previous / 'evidence/RESULT_AUDIT.json'), bind(HOME / 'evidence/RESULT_AUDIT.json')],
    'members': members, 'deduplication': 'root/type/sorted casefold-whitespace-normalized complete six-statement set; controls excluded before union',
    'duplicates_removed': duplicates, 'training_allowed': False, 'training_started': False,
    'formal_training_admission': False, 'merged_into_baseline': False,
    'source_root_count': len({q['root_id'] for q in all_rows}),
    'limitations': [f"Questions remain dependent on {len({q['root_id'] for q in all_rows})} source roots and existing fact families.",
                   'Whole-question text novelty does not imply semantic novelty or independent new facts.',
                   'Historical baseline and v83 inventory are preserved as previously audited; this union does not recertify them.',
                   'Actual current review exposure is explicitly disclosed in saved source judgments.']})
print({'new': len(new), 'cumulative': len(all_rows), **counts(all_rows)})
