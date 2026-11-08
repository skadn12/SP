"""Build shard inputs only from saved source and six-option parent judgments."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

PILOT = Path(__file__).resolve().parents[1]
DELIVERIES = PILOT.parent
OLD = DELIVERIES / 'h4_anchor_records_mixed_batch_v88'
PYTHON = '/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python'


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


def build(spec_path):
    spec = read(spec_path)
    home = PILOT / 'shards' / spec['shard_id']
    assert 2 <= len(spec['roots']) <= 4
    assert not (home / 'BATCH_PREPARATION.STARTED.json').exists()
    subprocess.run([PYTHON, '-B', str(PILOT / 'engine/install_shard.py'), str(home)], check=True, stdout=subprocess.DEVNULL)
    (home / 'evidence').mkdir()
    admitted = []
    parents = []
    slots = []
    for row in spec['roots']:
        public_path = Path(row['public_parent'])
        judgment_path = Path(row['parent_judgments'])
        public, judgment = read(public_path), read(judgment_path)
        source_path = Path(public['source_path'])
        source = read(source_path)
        assert public['root_id'] == row['root_id'] == judgment['root_id'] == source['root_id']
        assert judgment['version'] == 'v89_parent_source_judgments_v1'
        assert judgment['public_question'] == public
        assert judgment['public_question_sha256'] == bind(public_path)['sha256']
        assert judgment['source_sha256'] == public['source_sha256'] == bind(source_path)['sha256']
        assert judgment['full_source_read'] is True and judgment['full_source_turn_count'] == len(source['turns'])
        assert judgment['current_parent_private_gold_read_before_judgments_saved'] is False
        assert judgment['current_parent_private_evidence_read_before_judgments_saved'] is False
        assert judgment['current_solver_results_read'] is False and judgment['fresh_blind_review'] is False
        assert judgment['known_old_parent_gold'] is True
        judged = {j['option']: j for j in judgment['judgments']}
        assert set(judged) == set(public['options']) == set('ABCDEF')
        assert all(j['source_judgment'] == 'supported' and j['entailed'] is True
                   and j['proposition'] == public['options'][label] for label, j in judged.items())
        assert judgment.get('content_status', 'CORE_USABLE') == 'CORE_USABLE'
        review_path = Path(row['source_review'])
        review = read(review_path)
        assert review['decision'] == 'READY' and review['source'] == bind(source_path)
        admitted.append({'root_id': row['root_id'], 'split': 'train', 'source': bind(source_path), 'review': bind(review_path)})
        turns = {t['turn_ref']: t for t in source['turns']}
        parent = deepcopy(public)
        parent['gold'] = list('ABCDEF')
        parent['option_evidence'] = {}
        for label, j in judged.items():
            assert j['required_evidence']
            for e in j['required_evidence']:
                assert e['speaker'] == turns[e['turn_ref']]['speaker'] and e['quote'] and e['quote'] in turns[e['turn_ref']]['text']
            parent['option_evidence'][label] = {**deepcopy(j['required_evidence'][0]), 'relation': 'supports',
                                                 'reason': j['reason'], 'rationale': j['reason']}
        parent.update(parent_origin='v89_historical_parent_current_full_source_review',
                      historical_parent_gold_known=True, current_parent_review=bind(judgment_path),
                      training_allowed=False, formal_training_admission=False)
        provenance_path = home / 'parent_review' / (row['root_id'] + '_PROVENANCE.json')
        write(provenance_path, {'version': 'v89_parent_private_construction_v2', 'created_at_utc': now(),
                               'source_judgments_saved_at_utc': judgment['created_at_utc'],
                               'new_parent_candidate_id': parent['candidate_id'],
                               'public_parent': bind(public_path), 'source_judgments': bind(judgment_path),
                               'source': bind(source_path), 'historical_parent_pool': bind(row['historical_parent_pool']),
                               'historical_parent_candidate_id': row['historical_parent_candidate_id'],
                               'public_preparation': bind(row['parent_preparation']),
                               'historical_gold_known': True, 'fresh_blind_review': False,
                               'current_private_parent_created_after_saved_source_judgments': True,
                               'current_solver_scores_read': False, 'training_allowed': False})
        parents.append(parent)
        root_number = row['root_id'].split('-')[-1]
        slot = dict(target_counts=[2, 3, 4, 5], control_count=6, split='train', scope_prefix='',
                    slot_id=f'root{root_number}_family', family_id=f'root{root_number}_family', root_id=row['root_id'],
                    parent_candidate_id=parent['candidate_id'], parent_pool_path=str(home / 'PARENT_POOL.json'),
                    source_path=str(source_path), source_sha256=public['source_sha256'],
                    history_path=str(source_path), history_sha256=public['source_sha256'],
                    prior_pool_paths=row['public_prior_paths'], parent_review_path=str(judgment_path),
                    parent_review_sha256=bind(judgment_path)['sha256'], parent_provenance_path=str(provenance_path),
                    parent_provenance_sha256=bind(provenance_path)['sha256'], parent_origin=parent['parent_origin'],
                    parent_review_mode='saved_pre_generation_full_source_parent_review_with_known_historical_gold',
                    parent_gold_known_before_review=True, parent_gold_known_before_current_review=True,
                    fresh_blind_parent_review=False, current_parent_source_reread_claimed=True,
                    previous_parent_candidate_id=row['historical_parent_candidate_id'])
        slots.append(slot)
    write(home / 'ADMISSION_INPUT.json', {'version': 'v89_shard_admission_input_v1', 'shard_id': spec['shard_id'],
          'split': 'train', 'pilot_plan': bind(PILOT / 'PILOT_PLAN.json'),
          'source_gate': bind(DELIVERIES / 'h4_source_consistency_minfix_v77/source_gate.py'), 'roots': admitted})
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
    policy['version'] = 'v89_source_names_actor_reference_v1'
    policy['registry'] = bind(home / 'CHARACTER_REGISTRY.json')
    write(home / 'ACTOR_REFERENCE_POLICY.json', policy)
    config = deepcopy(read(OLD / 'CONFIG.json'))
    config.update(version='v89_full_turn_mixed_shard_v1', batch_id='v89_' + spec['shard_id'])
    m = config['multiselect_config']
    n = len(slots)
    m.update(slots=slots, family_count=n, max_http=2*n, max_questions=5*n, max_calls_per_model=90*n,
             target_distribution={**{str(g): n for g in range(2, 6)}, '6': 0},
             new_child_target_distribution={str(g): n for g in range(2, 6)}, control_target_distribution={'6': n},
             final_review_groups=list('abcd')[:n])
    config['ordering_slots'] = []
    for slot in slots:
        ordering = {k: slot[k] for k in ('root_id', 'source_path', 'source_sha256', 'history_path', 'history_sha256')}
        ordering.update(slot_id=slot['slot_id'].replace('_family', '_ordering'), candidate_count=2,
                        prior_pool_path=m['baseline_pool'], prior_pool_paths=slot['prior_pool_paths'])
        config['ordering_slots'].append(ordering)
    config.update(max_generation_http=3*n, max_candidates=7*n, max_solver_calls_per_model=114*n)
    for key, name in [('ordering_anchor_plan', 'SCENE_PLANS.json'), ('character_registry', 'CHARACTER_REGISTRY.json'),
                      ('actor_reference_policy', 'ACTOR_REFERENCE_POLICY.json')]:
        config[key] = bind(home / name)
    config['source_consistency'] = {'upstream_complete': bind(home / 'SOURCE_ADMISSION.COMPLETE.json'),
        'selected_roots': [r['root_id'] for r in admitted], 'excluded_held_roots': ['cc-train-episode-' + str(r) for r in read(PILOT/'PILOT_PLAN.json')['exclude_known_hold_roots']],
        'policy': 'complete_sources_fresh_review_ready_only', 'source_review_is_question_answer_certification': False}
    write(home / 'CONFIG.json', config)
    write(home / 'CONFIG_TEMPLATE.json', config)
    write(home / 'evidence/INPUT_BUILD_RECEIPT.json', {'created_at_utc': now(), 'spec': bind(spec_path),
        'config': bind(home/'CONFIG.json'), 'parent_count': n, 'training_allowed': False})
    subprocess.run([PYTHON, '-B', 'source_subset.py', 'finalize'], cwd=home, check=True, stdout=subprocess.DEVNULL)
    print({'shard': str(home), 'roots': n, 'status': 'INPUTS_FROZEN_BEFORE_PREPARATION'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('spec', type=Path)
    build(parser.parse_args().spec)
