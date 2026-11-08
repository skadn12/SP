"""Soft, predeclared scene/topic preferences for 20 existing-schema slots.

Hints contain topic descriptions only. They neither supply propositions/Gold
nor impose an evidence-location or semantic-admission requirement.
"""
from copy import deepcopy

CONTRACT = 'v50_soft_per_candidate_scene_topic_focus_v1'
REQUIREMENT = 'soft_preference_not_admission_gate'
TYPES = ('multi_select', 'ordering')
FIELDS = {'preferred_scene_refs', 'focus', 'requirement'}

# Four different topical views per scene; all source turns remain available.
SCHEDULE = (
    ('R02:A:S01',
     ('Reported test evidence, consistency, quality, and reactions to the data.',
      'Communication plans, team participation, presentation preparation, and organization.'),
     ('The exchange about test evidence, confidence, and readiness for the next phase.',
      'The exchange about communicating results, preparing the team, and immediate next steps.')),
    ('CANDIDATE:B:S01',
     ('Family outing preferences, recommendation requests, location, and available activities.',
      'Information gaps, ways to check details, practical advice, and follow-up intentions.'),
     ('The exchange about outing preferences, possible activities, and recommendations.',
      'The exchange about practical questions, uncertain information, advice, and follow-up plans.')),
    ('R02:A:S02',
     ('Presentation of evidence, audience reactions, and questions about the transition.',
      'Learning support, participation, workload concerns, and possible next steps.'),
     ('The exchange about sharing results, reactions, and questions during the team discussion.',
      'The exchange about transition concerns, responses, agreement, and subsequent plans.')),
    ('CANDIDATE:B:S02',
     ('Document handling, backup choices, reported searches, and recollections of preparation.',
      'Suggested search locations, intended actions, contingencies, and future organization.'),
     ('The exchange about the missing documents, earlier handling, searches, and suggestions.',
      'The exchange about preparation recollections, suggested locations, intentions, and contingencies.')),
    ('R02:A:S03',
     ('Implementation arrangements, team reactions, preparation, and personal responses.',
      'Time away from work, delegation, instructions, communication boundaries, and well-being.'),
     ('The exchange about implementation arrangements, reactions, and personal planning.',
      'The exchange about work coverage, communication boundaries, rest, and later follow-up.')),
)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def scene_refs(turns):
    return {turn.get('scene_ref', turn['turn_ref'].rsplit(':', 1)[0]) for turn in turns}


def candidate_ids(root, comparison_id, question_type, slots=2):
    require(question_type in TYPES, 'slot_focus_question_type_invalid')
    code = 'multi' if question_type == 'multi_select' else 'order'
    return [f'{root}-v50-{comparison_id}-merged-{code}-b01-{index:02d}'
            for index in range(1, slots + 1)]


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
                and all(isinstance(ref, str) and ref in available for ref in refs),
                'slot_focus_scene_not_in_full_source')
        require(isinstance(focus['focus'], str) and bool(focus['focus'].strip())
                and focus['requirement'] == REQUIREMENT, 'slot_focus_must_remain_soft')


def build_plan(root, batch_id, source_turns):
    chunks = []
    for index, (scene, multi, ordering) in enumerate(SCHEDULE, 1):
        comparison_id = batch_id + f'_c{index:02d}'
        mapping = {}
        for kind, hints in (('multi_select', multi), ('ordering', ordering)):
            for qid, hint in zip(candidate_ids(root, comparison_id, kind), hints):
                mapping[qid] = {'preferred_scene_refs': [scene], 'focus': hint,
                                'requirement': REQUIREMENT}
        chunks.append({'index': index, 'comparison_id': comparison_id,
                       'slot_focus_by_candidate_id': mapping})
    plan = {'contract': CONTRACT, 'root_id': root, 'batch_id': batch_id,
            'slots': 20, 'admission_gate': False, 'chunks': chunks}
    validate_plan(plan, root, batch_id, source_turns)
    return plan


def validate_plan(plan, root, batch_id, source_turns):
    require(isinstance(plan, dict) and set(plan) == {'contract', 'root_id', 'batch_id', 'slots', 'admission_gate', 'chunks'},
            'slot_focus_plan_fields_changed')
    require(plan['contract'] == CONTRACT and plan['root_id'] == root and plan['batch_id'] == batch_id
            and plan['slots'] == 20 and plan['admission_gate'] is False, 'slot_focus_plan_contract_changed')
    chunks = plan['chunks']
    require(isinstance(chunks, list) and len(chunks) == 5, 'slot_focus_chunk_count_changed')
    hints = []
    for index, chunk in enumerate(chunks, 1):
        comparison_id = batch_id + f'_c{index:02d}'
        require(set(chunk) == {'index', 'comparison_id', 'slot_focus_by_candidate_id'}
                and chunk['index'] == index and chunk['comparison_id'] == comparison_id,
                'slot_focus_chunk_identity_changed')
        ids = sum([candidate_ids(root, comparison_id, kind) for kind in TYPES], [])
        validate_map(chunk['slot_focus_by_candidate_id'], ids, source_turns, allow_empty=False)
        hints += [entry['focus'] for entry in chunk['slot_focus_by_candidate_id'].values()]
    require(len(hints) == len(set(hints)) == 20, 'slot_focus_topic_hints_not_distinct')


def subset(mapping, ids):
    if not mapping:
        return {}
    require(all(qid in mapping for qid in ids), 'slot_focus_pending_id_missing')
    return {qid: deepcopy(mapping[qid]) for qid in ids}
