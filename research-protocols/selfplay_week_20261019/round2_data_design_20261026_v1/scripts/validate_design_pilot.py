#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

def bind(p: Path):
    b = p.read_bytes(); return {'path': str(p.resolve()), 'sha256': hashlib.sha256(b).hexdigest(), 'bytes': len(b)}

def main():
    cal = json.loads((BASE/'calendar/SCENES_DRAFT.json').read_text())
    plan = json.loads((BASE/'calendar/CALENDAR_PLAN.json').read_text())
    assert cal['calendar_plan_binding'] == bind(BASE/'calendar/CALENDAR_PLAN.json')
    assert [s['scene_ref'] for s in cal['scenes']] == [s['scene_ref'] for s in plan['scenes']]
    all_refs = []
    for scene, planned in zip(cal['scenes'], plan['scenes']):
        assert scene['conversation_date'] == planned['conversation_date']
        assert scene['weekday'] == planned['weekday']
        assert len(scene['turns']) == 8
        all_refs.extend(t['turn_ref'] for t in scene['turns'])
        assert len(scene.get('temporal_checks', [])) >= 1
    assert len(all_refs) == len(set(all_refs)) == 24
    prototypes = json.loads((BASE/'question_design/TASK_ORIENTED_MC_PROTOTYPES.json').read_text())
    source_path = Path(prototypes['source_binding']['path'])
    source_bind = bind(source_path)
    assert all(prototypes['source_binding'][k] == source_bind[k] for k in ('path', 'sha256', 'bytes'))
    source = json.loads(source_path.read_text())
    turns_by_ref = {t['turn_ref']: t for t in source['turns']}
    assert prototypes['candidate_count'] == len(prototypes['candidates']) == 4
    pairs = []
    generic = re.compile(r'which statements are supported|which complete accounts', re.I)
    evidence_count = 0
    for c in prototypes['candidates']:
        options = c['solver_view']['options']; gold = c['gold']
        assert set(options) == set('ABCDEF')
        assert set(gold) <= set(options)
        assert len(gold) == c['answer_cardinality']
        assert c['operation'] in json.loads((BASE/'question_design/MC_ASSIGNMENT_MATRIX.json').read_text())['operations']
        assert not generic.search(c['solver_view']['question'])
        assert c['source_binding'] == prototypes['source_binding']
        for review in c['option_reviews']:
            assert review['option'] in options
            for ev in review['evidence']:
                evidence_count += 1
                assert ev['source_sha256'] == prototypes['source_binding']['sha256']
                assert ev['original_full_coordinate'].startswith(c['source_binding']['root_id'] + ':')
                assert ev['original_turn_ref'] in ev['original_full_coordinate']
                assert ev['quote'].strip()
                assert ev['original_turn_ref'] in turns_by_ref
                assert ev['quote'] in turns_by_ref[ev['original_turn_ref']]['text']
        pairs.append((c['operation'], c['answer_cardinality']))
    q = json.loads((BASE/'question_design/MC_ASSIGNMENT_MATRIX.json').read_text())
    assert q['pilot_coverage']
    assert len({card for op, card in pairs if op == 'quantitative_reconciliation'}) == 2
    # The form audit is descriptive only, but it must remain bound to the exact
    # source files used for the comparison so later edits cannot silently change
    # the conclusion about template convergence.
    old_audit_path = BASE/'question_design/OLD_MC_FORM_AUDIT.json'
    old_audit = json.loads(old_audit_path.read_text())
    assert old_audit['schema_version'] == 'old_mc_form_audit_v1'
    audit_sources = {s['label']: s for s in old_audit['sources']}
    assert audit_sources['current_19_root_mc']['multi_select_count'] == 19
    assert audit_sources['historical_reuse_113']['multi_select_count'] == 51
    assert 'does not prove imitation' in old_audit['interpretation']['cause_limit']
    short_path = BASE/'calendar/SHORT_RENDER.json'
    short = json.loads(short_path.read_text())
    assert short['source_draft_binding']['path'] == str((BASE/'calendar/SCENES_DRAFT.json').resolve())
    assert short['validation'] == {'unique_full_coordinates': True, 'turns': 24}
    assert short['token_counts']['short_render'] < short['token_counts']['full_coordinate_render']
    review_path = BASE/'validation/INDEPENDENT_REVIEW_20261026.json'
    review = json.loads(review_path.read_text())
    assert review['overall_decision'] == 'DESIGN_PILOT_PASS_DATA_ADMISSION_HOLD'
    assert review['results']['calendar']['status'] == 'PASS'
    assert review['results']['multiple_choice']['status'] == 'PASS'
    assert review['results']['data_admission']['status'] == 'HOLD'
    assert review['integrity_checks'] == {
        'model_calls': 0,
        'training': False,
        'model_evaluation': False,
        'frozen_experiment_modified': False,
        'legacy_roots_modified': False,
    }
    intake_path = BASE/'production/PRODUCTION_INTAKE_GATING_TEMPLATE.json'
    intake = json.loads(intake_path.read_text())
    assert intake['template_status'] == 'TEMPLATE_ONLY'
    assert intake['promotion_gate']['status'] == 'BLOCKED_UNTIL_GATES_PASS'
    assert intake['side_effect_guard']['model_calls'] == 0
    assert intake['side_effect_guard']['training_started'] is False
    out = {
        'schema_version':'round2_design_pilot_validation_v1',
        'status':'PASS',
        'calendar':{
            'scenes':len(cal['scenes']),
            'turns':len(all_refs),
            'plan_binding':bind(BASE/'calendar/CALENDAR_PLAN.json')
        },
        'short_render_binding':bind(short_path),
        'short_render_tokens':short['token_counts'],
        'mc':{
            'candidates':len(pairs),
            'evidence_quotes_checked':evidence_count,
            'operation_cardinality_pairs': [{'operation':op,'answer_cardinality':card} for op,card in pairs],
            'same_operation_multiple_cardinalities': True
        },
        'old_mc_form_audit_binding':bind(old_audit_path),
        'independent_review_binding':bind(review_path),
        'production_intake_template_binding':bind(intake_path),
        'data_admission':review['results']['data_admission']['status'],
        'model_calls':0,
        'training':False,
        'frozen_experiment_modified':False
    }
    (BASE/'validation/VALIDATION.json').write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(out, ensure_ascii=False))

if __name__ == '__main__': main()
