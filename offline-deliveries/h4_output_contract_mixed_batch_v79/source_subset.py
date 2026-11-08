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
    value['version'] = 'v79_output_contract_mixed_batch_v1'
    value['batch_id'] = 'v79_contract_mixed_01'
    value['multiselect_engine'] = str(V75)
    m = value['multiselect_config']
    m['version'] = 'v79_sentence_writer_program_bound_replacements_v1'
    m['slots'] = [s for s in m['slots'] if s['root_id'] in ROOTS]
    for slot in m['slots']:
        slot['prior_pool_paths'].append(str(HERE.parent / 'h4_ready_roots_mixed_batch_v78/multiselect/PUBLIC_QUESTIONS.json'))
    n = len(ROOTS)
    m.update(family_count=n, max_http=2*n, max_questions=5*n, max_calls_per_model=90*n,
             target_distribution={**{str(k):n for k in range(2,6)}, '6':0},
             new_child_target_distribution={str(k):n for k in range(2,6)}, control_target_distribution={'6':n})
    value['ordering_slots'] = [s for s in value['ordering_slots'] if s['root_id'] in ROOTS]
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
