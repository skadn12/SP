"""Neutral soft allocation across the actually retained A/B scenes.

Allocation is frozen before generation. Missing scenes are reported rather than
replaced by invented evidence, and every request retains the complete history.
"""
from copy import deepcopy

CONTRACT = 'v52_soft_available_scene_topic_focus_v1'
REQUIREMENT = 'soft_preference_not_admission_gate'
TYPES = ('multi_select', 'ordering')
FIELDS = {'preferred_scene_refs', 'focus', 'requirement'}
CYCLE_RULE = 'cycle_existing_scenes_in_A1_B1_A2_B2_A3_B3_order_no_extra_generation_slots'
EXPLICIT_RULE = 'explicit_frozen_soft_focus_plan_no_extra_generation_slots'
REQUESTED_SCENES = tuple(f'CANDIDATE:{route}:S{number:02d}' for number in range(1, 4) for route in ('A', 'B'))
NEUTRAL_FOCI = (
    'Reported observations, what is established, and what remains uncertain.',
    'Questions, responses, decisions, and their stated qualifications.',
    'Earlier actions, current status, and intended next actions.',
    'Participant roles, objects, responsibilities, and stated limits.',
    'Suggestions, alternatives, follow-up intentions, and conditions.',
    'The exchange between initial concerns and subsequent responses.',
    'Explicit confirmations, disagreements, and unresolved details.',
    'Preparation, reported results, and later commitments.',
    'Constraints, reasons explicitly stated, and practical responses.',
    'Changes in plans, unchanged commitments, and anticipated next steps.',
    'The sequence of requests, clarifications, and replies in the dialogue.',
    'The sequence of reports, suggestions, and decisions in the dialogue.',
    'The sequence of intentions, qualifications, and confirmations in the dialogue.',
    'The sequence of concerns, information, and follow-up plans in the dialogue.',
    'The sequence of recalled actions, current observations, and proposed actions.',
    'The sequence of offers, questions, and responses in the dialogue.',
    'The sequence of stated constraints, alternatives, and commitments.',
    'The sequence of explanations, reported outcomes, and remaining questions.',
    'The sequence of requests for details, supplied details, and plans.',
    'The sequence of initial positions, later clarifications, and final intentions.',
)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def scene_refs(turns):
    return {turn.get('scene_ref') or turn['turn_ref'].rsplit(':', 1)[0] for turn in turns}


def candidate_ids(root, comparison_id, question_type, slots=2):
    require(question_type in TYPES, 'slot_focus_question_type_invalid')
    code = 'multi' if question_type == 'multi_select' else 'order'
    return [f'{root}-v52-{comparison_id}-merged-{code}-b01-{index:02d}' for index in range(1, slots + 1)]


def validate_map(mapping, expected_ids, turns, *, allow_empty=True):
    require(isinstance(mapping, dict), 'slot_focus_map_not_object')
    if not mapping and allow_empty:
        return
    require(set(mapping) == set(expected_ids), 'slot_focus_candidate_set_changed')
    available = scene_refs(turns)
    for focus in mapping.values():
        require(isinstance(focus, dict) and set(focus) == FIELDS, 'slot_focus_fields_changed')
        refs = focus['preferred_scene_refs']
        require(isinstance(refs, list) and bool(refs) and len(refs) == len(set(refs))
                and all(isinstance(ref, str) and ref in available for ref in refs), 'slot_focus_scene_not_in_full_source')
        require(isinstance(focus['focus'], str) and bool(focus['focus'].strip())
                and focus['requirement'] == REQUIREMENT, 'slot_focus_must_remain_soft')


def build_plan(root, batch_id, source_turns, slots=20):
    require(slots in (8, 20), 'slot_focus_slots_must_be_8_or_20')
    present = scene_refs(source_turns)
    available = [scene for scene in REQUESTED_SCENES if scene in present]
    require(available, 'slot_focus_no_retained_new_scene')
    chunks = []
    for index in range(1, slots // 4 + 1):
        comparison_id = batch_id + f'_c{index:02d}'
        mapping = {}
        for kind_index, kind in enumerate(TYPES):
            for slot, qid in enumerate(candidate_ids(root, comparison_id, kind)):
                global_slot = (index - 1) * 4 + kind_index * 2 + slot
                hint_index = kind_index * 10 + (index - 1) * 2 + slot
                mapping[qid] = {'preferred_scene_refs': [available[global_slot % len(available)]],
                    'focus': NEUTRAL_FOCI[hint_index], 'requirement': REQUIREMENT}
        chunks.append({'index': index, 'comparison_id': comparison_id, 'slot_focus_by_candidate_id': mapping})
    plan = {'contract': CONTRACT, 'root_id': root, 'batch_id': batch_id,
        'slots': slots, 'admission_gate': False, 'requested_scene_refs': list(REQUESTED_SCENES),
        'available_scene_refs': available, 'missing_scene_refs': [s for s in REQUESTED_SCENES if s not in present],
        'expansion_complete': all(s in present for s in REQUESTED_SCENES),
        'coverage_rule': CYCLE_RULE,
        'chunks': chunks}
    validate_plan(plan, root, batch_id, source_turns)
    return plan


def validate_plan(plan, root, batch_id, source_turns):
    required = {'contract','root_id','batch_id','slots','admission_gate','chunks','requested_scene_refs',
                'available_scene_refs','missing_scene_refs','expansion_complete','coverage_rule'}
    require(isinstance(plan, dict) and set(plan) == required, 'slot_focus_plan_fields_changed')
    require(plan['contract'] == CONTRACT and plan['root_id'] == root and plan['batch_id'] == batch_id
            and plan['slots'] in (8, 20) and plan['admission_gate'] is False, 'slot_focus_plan_contract_changed')
    present = scene_refs(source_turns)
    available = [s for s in REQUESTED_SCENES if s in present]
    require(available, 'slot_focus_no_retained_new_scene')
    require(plan['requested_scene_refs'] == list(REQUESTED_SCENES) and plan['available_scene_refs'] == available
            and plan['missing_scene_refs'] == [s for s in REQUESTED_SCENES if s not in present]
            and plan['expansion_complete'] is (len(available) == 6)
            and plan['coverage_rule'] in (CYCLE_RULE, EXPLICIT_RULE),
            'slot_focus_scene_availability_changed')
    chunks = plan['chunks']
    require(isinstance(chunks, list) and len(chunks) == plan['slots'] // 4, 'slot_focus_chunk_count_changed')
    hints = []
    for index, chunk in enumerate(chunks, 1):
        comparison_id = batch_id + f'_c{index:02d}'
        require(set(chunk) == {'index','comparison_id','slot_focus_by_candidate_id'} and chunk['index'] == index
                and chunk['comparison_id'] == comparison_id, 'slot_focus_chunk_identity_changed')
        ids = sum([candidate_ids(root, comparison_id, kind) for kind in TYPES], [])
        validate_map(chunk['slot_focus_by_candidate_id'], ids, source_turns, allow_empty=False)
        for offset, qid in enumerate(ids):
            focus = chunk['slot_focus_by_candidate_id'][qid]
            if plan['coverage_rule'] == CYCLE_RULE:
                require(focus['preferred_scene_refs'] == [available[((index - 1) * 4 + offset) % len(available)]],
                        'slot_focus_available_scene_cycle_changed')
            hints.append(focus['focus'])
    require(len(hints) == len(set(hints)) == plan['slots'], 'slot_focus_topic_hints_not_distinct')


def subset(mapping, ids):
    if not mapping:
        return {}
    require(all(qid in mapping for qid in ids), 'slot_focus_pending_id_missing')
    return {qid: deepcopy(mapping[qid]) for qid in ids}
