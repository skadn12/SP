"""Explicit READY-only subset derivation; preserve all upstream sources and rules."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from common import HERE, V75, bind, need, read, sha
V76 = HERE.parent / 'h4_mixed_question_batch_v76'
V77 = HERE.parent / 'h4_source_consistency_minfix_v77'
ROOTS = ('cc-train-episode-194259', 'cc-train-episode-130221')
PINS = {'/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_mixed_question_batch_v76/CONFIG.json': '6dd3f528a280b2c0384d9c18f2742f9f063b9f6552b24b27ff3ce864f353f592', '/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_source_consistency_minfix_v77/source_gate.py': '2c5d6b8bd0eeb50de96bb37800f0599ae6db9b5764a7a3e181b542e89c40e58e', '/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_source_consistency_minfix_v77/CONSISTENCY_REVIEW.COMPLETE.json': '54d292c33b6bd5fd56eb3747cbb783e55eedcca1257799d1fd33111a01a2842d'}


PINS['/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_ready_roots_mixed_batch_v78/multiselect/PUBLIC_QUESTIONS.json'] = '0defe10c2b436fac21befb22e0e6a67f40539ec03e28d956f5c8c6987b01c592'


PINS.update({'/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_output_contract_mixed_batch_v79/multiselect/PUBLIC_QUESTIONS.json': '38f10f0fb2c716cebcd8ec4fdb439aea2c7e3e2a08c5f6116755faf7384997c7', '/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_output_contract_mixed_batch_v79/ordering/PUBLIC_QUESTIONS.json': '3c5d4f5a9b73fa01d32a494ade0cbf610cefae75feffdd5e3ea4665517278a2b'})

PINS['/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_scoped_ordering_mixed_batch_v80/multiselect/PUBLIC_QUESTIONS.json'] = '7cb305f66717a6138dc1efb7e49ca46490f2805b25094a4e77a806bf205edf01'


PINS['/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_bound_scene_ordering_mixed_batch_v81/SCENE_PLANS.json'] = 'af5d0648650e16d95d4e81bfa2f415fd1b42ff651d213db6f9aed81749646b64'


def verified_summary():
    for path, expected in PINS.items():
        need(sha(path) == expected, 'upstream_binding_changed:' + path)
    spec = importlib.util.spec_from_file_location('_v78_frozen_source_gate', V77 / 'source_gate.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    summary = module.verify_frozen()
    by_root = {r['root_id']: r for r in summary['roots']}
    need(all(by_root[r]['decision'] == 'READY' for r in ROOTS), 'selected_source_not_ready')
    return summary


def expected_config(summary):
    value = deepcopy(read(V76 / 'CONFIG.json'))
    value['version'] = 'v81_bound_scene_ordering_mixed_batch_v1'
    value['batch_id'] = 'v81_bound_scene_mixed_01'
    value['multiselect_engine'] = str(V75)
    value['ordering_anchor_plan'] = bind(HERE / 'SCENE_PLANS.json')
    m = value['multiselect_config']
    m['version'] = 'v79_sentence_writer_program_bound_replacements_v1'
    m['slots'] = [s for s in m['slots'] if s['root_id'] in ROOTS]
    for slot in m['slots']:
        slot['prior_pool_paths'].extend([str(HERE.parent / 'h4_ready_roots_mixed_batch_v78/multiselect/PUBLIC_QUESTIONS.json'), str(HERE.parent / 'h4_output_contract_mixed_batch_v79/multiselect/PUBLIC_QUESTIONS.json'), str(HERE.parent / 'h4_scoped_ordering_mixed_batch_v80/multiselect/PUBLIC_QUESTIONS.json')])
    n = len(ROOTS)
    m.update(family_count=n, max_http=2*n, max_questions=5*n, max_calls_per_model=90*n,
             target_distribution={**{str(k):n for k in range(2,6)}, '6':0},
             new_child_target_distribution={str(k):n for k in range(2,6)}, control_target_distribution={'6':n})
    value['ordering_slots'] = [s for s in value['ordering_slots'] if s['root_id'] in ROOTS]
    for slot in value['ordering_slots']:
        slot['prior_pool_paths'] = [slot['prior_pool_path'], str(HERE.parent / 'h4_output_contract_mixed_batch_v79/ordering/PUBLIC_QUESTIONS.json')]
    value.update(max_generation_http=3*n, max_candidates=7*n, max_solver_calls_per_model=114*n)
    by_root = {r['root_id']:r for r in summary['roots']}
    need([s['root_id'] for s in m['slots']] == list(ROOTS), 'missing_frozen_subset')
    for s in m['slots']:
        need(by_root[s['root_id']]['source'] == {'path':s['source_path'], 'sha256':s['source_sha256']}, 'selected_source_binding_changed')
    value['source_consistency'] = {
        'upstream_complete':bind(V77 / 'CONSISTENCY_REVIEW.COMPLETE.json'),
        'selected_roots':list(ROOTS), 'excluded_held_roots':summary['held_roots'],
        'policy':'explicit_ready_subset_full_sources_unchanged',
        'source_review_is_question_answer_certification':False}
    return value


def validate(value):
    expected = expected_config(verified_summary())
    need(json.dumps(value, sort_keys=True) == json.dumps(expected, sort_keys=True), 'ready_subset_config_changed')
    return value
