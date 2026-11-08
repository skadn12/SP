"""Build v96 mixed shards from saved fresh six-fact source judgments.

No authoring, network access, semantic repair, retries, or Solver calls occur.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

PILOT = Path(__file__).resolve().parents[1]
DELIVERIES = PILOT.parent
OLD = DELIVERIES / 'h4_anchor_records_mixed_batch_v88'
PYTHON = '/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python'
sys.dont_write_bytecode = True
sys.path.insert(0, str(PILOT / 'engine'))
import fresh_parent
import batch_scope
import expansion_binding
import ordering_plan


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write(chr(10))


def now():
    return datetime.now(timezone.utc).isoformat()


def check(spec_path):
    spec = read(spec_path)
    fresh_parent.need(spec.get('version') == 'v96_current_ready_shard_spec_v1', 'current_spec_version_required')
    fresh_parent.need(isinstance(spec.get('shard_id'), str)
                      and re.fullmatch('[A-Za-z0-9_]{1,96}', spec['shard_id']), 'safe_shard_id_required')
    batch_scope.validate_roots([row['root_id'] for row in spec['roots']])
    plan_path = Path(spec.get('plan_path', PILOT / 'PILOT_PLAN.json')).resolve()
    batch_scope.validate_plan(read(plan_path))
    plan_file = Path(spec['scene_plans']).resolve()
    plans = read(plan_file)['plans']
    fresh_parent.need(len(plans) == 4 and all(p['root_id'] in batch_scope.EXPECTED_ROOTS for p in plans),
                      'exact_current_ordering_plan_coverage')
    for row in spec['roots']:
        public, judgment, source = fresh_parent.validate_judgments(
            row['public_parent'], row['parent_judgments'], row['expanded_source_provenance'])
        fresh_parent.need(public['root_id'] == row['root_id'], 'spec_root_identity')
        lineage = expansion_binding.validate(row['root_id'], bind(public['source_path']),
                                              row['expanded_source_provenance'])
        new_refs = set(lineage['new_turn_refs'])
        fresh_parent.need(all(any(e['turn_ref'] in new_refs for e in item['required_evidence'])
                              for item in judgment['judgments']), 'parent_fact_requires_current_new_scene_evidence')
        review = fresh_parent.validate_source_review({'root_id': row['root_id'],
                 'source': bind(public['source_path']), 'review_path': row['source_review']})
        fresh_parent.need(review['decision'] == 'READY', 'source_not_ready:' + row['root_id'])
        selected = [p for p in plans if p['root_id'] == row['root_id']]
        fresh_parent.need(all(p['scene_id'] == 'R02:B:S01'
                              and all(a['turn_ref'] in new_refs for a in p['anchors']) for p in selected),
                          'ordering_plan_requires_current_new_scene')
        ordering_plan.build({'ordering_anchor_plan': bind(plan_file)},
                            {'root_id': row['root_id'], 'source_sha256': public['source_sha256']},
                            source, [row['root_id'] + '-check-' + str(i) for i in (1, 2)])
        for key in ('public_prior_paths', 'public_ordering_prior_paths'):
            fresh_parent.need(isinstance(row.get(key, []), list)
                              and all(Path(p).is_file() for p in row.get(key, [])), 'prior_paths_invalid:' + key)
    return spec


def build(spec_path):
    spec = check(spec_path)
    home = PILOT / 'shards' / spec['shard_id']
    assert not (home / 'BATCH_PREPARATION.STARTED.json').exists()
    subprocess.run([PYTHON, '-B', str(PILOT / 'engine/install_shard.py'), str(home)], check=True, stdout=subprocess.DEVNULL)
    (home / 'evidence').mkdir(exist_ok=True)
    admitted = []
    parents = []
    slots = []
    for row in spec['roots']:
        public_path = Path(row['public_parent']).resolve()
        judgment_path = Path(row['parent_judgments']).resolve()
        public, judgment = read(public_path), read(judgment_path)
        source_path = Path(public['source_path']).resolve()
        source = read(source_path)
        assert public['root_id'] == row['root_id'] == judgment['root_id'] == source['root_id']
        fresh_parent.validate_judgments(public_path, judgment_path, row['expanded_source_provenance'])
        review_path = Path(row['source_review']).resolve()
        review = fresh_parent.validate_source_review({'root_id': row['root_id'], 'source': bind(source_path),
                                                      'review_path': str(review_path)})
        assert review['decision'] == 'READY'
        admitted.append({'root_id': row['root_id'], 'split': 'train', 'source': bind(source_path), 'review': bind(review_path),
                         'expanded_source_provenance': bind(row['expanded_source_provenance'])})
        provenance_path = home / 'parent_review' / (row['root_id'] + '_PROVENANCE.json')
        parent, provenance = fresh_parent.construct_parent(public_path, judgment_path,
                                                           row['expanded_source_provenance'], provenance_path)
        parents.append(parent)
        root_number = row['root_id'].split('-')[-1]
        slot = dict(target_counts=[2, 3, 4, 5], control_count=6, split='train', scope_prefix='',
                    slot_id=f'root{root_number}_family', family_id=f'root{root_number}_family', root_id=row['root_id'],
                    parent_candidate_id=parent['candidate_id'], parent_pool_path=str(home / 'PARENT_POOL.json'),
                    source_path=str(source_path), source_sha256=public['source_sha256'],
                    history_path=str(source_path), history_sha256=public['source_sha256'],
                    prior_pool_paths=row.get('public_prior_paths', []), parent_review_path=str(judgment_path),
                    parent_review_sha256=bind(judgment_path)['sha256'], parent_provenance_path=str(provenance_path),
                    parent_provenance_sha256=bind(provenance_path)['sha256'], parent_origin=parent['parent_origin'],
                    parent_review_mode='saved_fresh_public_parent_full_source_review',
                    parent_gold_known_before_review=False, parent_gold_known_before_current_review=False,
                    fresh_blind_parent_review=judgment['fresh_blind_review'], current_parent_source_reread_claimed=True)
        slots.append(slot)
    write(home / 'ADMISSION_INPUT.json', {'version': 'v96_shard_admission_input_v1', 'shard_id': spec['shard_id'],
          'split': 'train', 'pilot_plan': bind(spec.get('plan_path', PILOT / 'PILOT_PLAN.json')),
          'source_gate': bind(DELIVERIES / 'h4_source_consistency_minfix_v77/source_gate.py'),
          'source_review_validator': bind(home / 'fresh_parent.py'),
          'source_review_contract': 'v96_full_source_v77_checks_truthful_exposure', 'roots': admitted})
    subprocess.run([PYTHON, '-B', 'source_admission.py', 'finalize'], cwd=home, check=True, stdout=subprocess.DEVNULL)
    write(home / 'PARENT_POOL.json', {'questions': parents, 'training_allowed': False, 'formal_training_admission': False})
    planned = read(spec['scene_plans'])
    plans = [p for p in planned['plans'] if p['root_id'] in {r['root_id'] for r in admitted}]
    assert len(plans) == 2 * len(admitted)
    write(home / 'SCENE_PLANS.json', {**planned, 'plans': plans, 'planning_record': bind(spec['scene_plans'])})
    registry = {'schema_version': 'evidence_backed_character_registry_v1', 'created_at_utc': now(),
                'purpose': 'Source speaker names only; no added QA premise or inferred gender.', 'source_bindings': [], 'characters': []}
    for row in admitted:
        source = read(row['source']['path'])
        names = sorted({t['speaker'] for t in source['turns']})
        registry['source_bindings'].append({**row['source'], 'root_id': row['root_id'], 'full_turn_count': len(source['turns']),
                                            'all_turns_reviewed': True, 'source_review': row['review'], 'speaker_names': names})
        registry['characters'].extend({'root_id': row['root_id'], 'canonical_name': name,
             'gender': {'value': None, 'status': 'not_inferred_or_used'},
             'third_person_pronouns': {'value': None, 'status': 'not_inferred_or_used'}} for name in names)
    write(home / 'CHARACTER_REGISTRY.json', registry)
    policy = read(OLD / 'ACTOR_REFERENCE_POLICY.json')
    policy['version'] = 'v96_source_names_actor_reference_v1'
    policy['registry'] = bind(home / 'CHARACTER_REGISTRY.json')
    write(home / 'ACTOR_REFERENCE_POLICY.json', policy)
    config = deepcopy(read(OLD / 'CONFIG.json'))
    config.update(version='v96_full_turn_mixed_shard_v1', batch_id='v96_' + spec['shard_id'])
    m = config['multiselect_config']
    n = len(slots)
    m.update(slots=slots, family_count=n, max_http=2*n, max_questions=5*n, max_calls_per_model=90*n,
             target_distribution={**{str(g): n for g in range(2, 6)}, '6': 0},
             new_child_target_distribution={str(g): n for g in range(2, 6)}, control_target_distribution={'6': n},
             final_review_groups=list('abcd')[:n])
    m['parent_semantic_review_mode'] = 'fresh_public_parent_saved_full_source_judgments_then_private_projection'
    m['control_provenance'] = 'fresh_parent_reviewed_on_current_expanded_source'
    config['ordering_slots'] = []
    rows_by_root = {row['root_id']: row for row in spec['roots']}
    for slot in slots:
        ordering = {k: slot[k] for k in ('root_id', 'source_path', 'source_sha256', 'history_path', 'history_sha256')}
        ordering.update(slot_id=slot['slot_id'].replace('_family', '_ordering'), candidate_count=2,
                        prior_pool_path=m['baseline_pool'],
                        prior_pool_paths=rows_by_root[slot['root_id']].get(
                            'public_ordering_prior_paths', slot['prior_pool_paths']))
        config['ordering_slots'].append(ordering)
    config.update(max_generation_http=3*n, max_candidates=7*n, max_solver_calls_per_model=114*n)
    for key, name in [('ordering_anchor_plan', 'SCENE_PLANS.json'), ('character_registry', 'CHARACTER_REGISTRY.json'),
                      ('actor_reference_policy', 'ACTOR_REFERENCE_POLICY.json')]:
        config[key] = bind(home / name)
    config['source_consistency'] = {'upstream_complete': bind(home / 'SOURCE_ADMISSION.COMPLETE.json'),
        'selected_roots': [r['root_id'] for r in admitted], 'excluded_held_roots': ['cc-train-episode-' + str(r) for r in read(spec.get('plan_path', PILOT / 'PILOT_PLAN.json'))['exclude_known_hold_roots']],
        'policy': 'complete_sources_fresh_review_ready_only', 'source_review_is_question_answer_certification': False}
    write(home / 'CONFIG.json', config)
    write(home / 'CONFIG_TEMPLATE.json', config)
    write(home / 'evidence/INPUT_BUILD_RECEIPT.json', {'created_at_utc': now(), 'spec': bind(spec_path),
        'config': bind(home/'CONFIG.json'), 'parent_count': n, 'training_allowed': False,
        'builder': bind(Path(__file__)),
        'fresh_parent_review_mode': 'saved_public_six_fact_full_source_review',
        'historical_parent_reuse': False, 'upstream_v79_strategy_changed': False})
    subprocess.run([PYTHON, '-B', 'source_subset.py', 'finalize'], cwd=home, check=True, stdout=subprocess.DEVNULL)
    print({'shard': str(home), 'roots': n, 'status': 'INPUTS_FROZEN_BEFORE_PREPARATION'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('spec', type=Path)
    parser.add_argument('--check', action='store_true', help='Validate public inputs without creating a shard or private parent.')
    args = parser.parse_args()
    if args.check:
        checked = check(args.spec)
        print(json.dumps({'status': 'PUBLIC_INPUTS_VALID_NO_WRITES',
                          'roots': [row['root_id'] for row in checked['roots']]}))
    else:
        build(args.spec)
