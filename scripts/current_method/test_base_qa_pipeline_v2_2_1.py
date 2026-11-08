"""Bounded offline V2.2.1 contracts, not admitted QA or model-quality claims.

Only synthetic values / controlled ledgers and read-only historical artifacts
are used.  These tests invoke no author, reviewer, API, Solver, or training.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

try:
    from . import base_qa_pipeline_v2_2_1 as qa
    from . import test_base_qa_pipeline_v2_2 as previous
except ImportError:
    import base_qa_pipeline_v2_2_1 as qa
    import test_base_qa_pipeline_v2_2 as previous


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'research-protocols/selfplay_week_20261019'


def query_value(value, query):
    return qa._query_answer(value, query)


def precedence_card(*, isolated=False):
    """The indirect A<C query needs both distinct-scene premises."""
    source = dict(root_id='v221-controlled-precedence', source_id='synthetic-fixture-only',
        source_status='CONTROLLED_REGRESSION_ONLY_NOT_ADMITTED', turns=[
            dict(turn_ref='T1', scene_ref='S1', speaker='Jo', text='For exhibition X, A must precede B.'),
            dict(turn_ref='T2', scene_ref='S2', speaker='Jo', text='For that exhibition X, B must precede C.'),
            dict(turn_ref='T3', scene_ref='S3', speaker='Jo', text='D is another stage with no order fixed.')])
    first = dict(precedence_edges=[['A', 'B']])
    if isolated:
        first['vertices'] = ['A', 'B', 'C', 'D']
        source['turns'][0]['text'] = 'For exhibition X, the stages are A, B, C, and D; A must precede B.'
    second = dict(precedence_edges=[['B', 'C']])
    nodes = [previous.fact(source, 'ab', 'T1', first), previous.fact(source, 'bc', 'T2', second)]
    node = dict(node_id='chain', type='operation', kind='constraint_chain', inputs=['ab', 'bc'],
        claim='A must precede C through the required intermediate B.',
        rationale='Compose the two explicit prerequisite links.', bridge=previous.bridge(source),
        rule=dict(type='precedence_closure', expected_result=None))
    node['rule']['expected_result'] = qa._relational_rule(node, [first, second], probe=True)
    nodes.append(node)
    card = dict(information_need='Must A finish before C for exhibition X?', natural_value='PASS',
        uncertainties=[], dependency_graph=dict(nodes=nodes, answer_node='chain'),
        query=dict(type='precedence', before='A', after='C'), path_audit=previous.path_audit(source))
    return source, card


def constraint_card():
    """A unique room does not require a unique global person assignment."""
    source = dict(root_id='v221-controlled-constraints', source_id='synthetic-fixture-only',
        source_status='CONTROLLED_REGRESSION_ONLY_NOT_ADMITTED', turns=[
            dict(turn_ref='T1', scene_ref='S1', speaker='Jo', text='For booking X, the person is Ann, Ben, or Cara, and the room Hall or Lab. Ben is unavailable.'),
            dict(turn_ref='T2', scene_ref='S2', speaker='Jo', text='For that booking X, Ann requires Hall, and Cara also requires Hall.')])
    values = [dict(domains=dict(person=['Ann', 'Ben', 'Cara'], room=['Hall', 'Lab']),
                   constraints=[{'not': {'person': 'Ben'}}]),
              dict(constraints=[{'if': {'person': 'Ann'}, 'then': {'room': 'Hall'}},
                                {'if': {'person': 'Cara'}, 'then': {'room': 'Hall'}}])]
    nodes = [previous.fact(source, 'availability', 'T1', values[0]),
             previous.fact(source, 'room_rules', 'T2', values[1])]
    node = dict(node_id='eligible', type='operation', kind='constraint_apply',
        inputs=['availability', 'room_rules'], claim='Either Ann or Cara can use Hall; the room is fixed.',
        rationale='Combine availability with the person-specific room requirements.',
        bridge=previous.bridge(source), rule=dict(type='constraint_filter', expected_result=None))
    node['rule']['expected_result'] = qa._relational_rule(node, values, probe=True)
    nodes.append(node)
    return source, dict(information_need='Which room is required for booking X?', natural_value='PASS',
        uncertainties=[], dependency_graph=dict(nodes=nodes, answer_node='eligible'),
        query=dict(type='assignment_field', field='room'), path_audit=previous.path_audit(source))


def irrelevant_coupling_card():
    """The room answer is explicit in A; B has unrelated coupled variables."""
    source = dict(root_id='v221-controlled-irrelevant-coupling', source_id='synthetic-fixture-only',
        source_status='CONTROLLED_REGRESSION_ONLY_NOT_ADMITTED', turns=[
            dict(turn_ref='T1', scene_ref='S1', speaker='Jo', text='For booking X the room is Hall or Lab, and Lab is excluded.'),
            dict(turn_ref='T2', scene_ref='S2', speaker='Jo', text='For that booking X the person is Ann or Ben and slot AM or PM. If Ann, use AM.')])
    values = [dict(domains=dict(room=['Hall', 'Lab']), constraints=[{'not': {'room': 'Lab'}}]),
              dict(domains=dict(person=['Ann', 'Ben'], slot=['AM', 'PM']),
                   constraints=[{'if': {'person': 'Ann'}, 'then': {'slot': 'AM'}}])]
    nodes = [previous.fact(source, 'direct_room', 'T1', values[0]),
             previous.fact(source, 'unrelated_coupling', 'T2', values[1])]
    node = dict(node_id='combined', type='operation', kind='constraint_apply',
        inputs=['direct_room', 'unrelated_coupling'], claim='Booking X uses Hall with a feasible person and slot.',
        rationale='Combine the declared finite constraints without inventing a room-person dependence.',
        bridge=previous.bridge(source), rule=dict(type='constraint_filter', expected_result=None))
    node['rule']['expected_result'] = qa._relational_rule(node, values, probe=True)
    nodes.append(node)
    return source, dict(information_need='Which room is required for booking X?', natural_value='PASS',
        uncertainties=[], dependency_graph=dict(nodes=nodes, answer_node='combined'),
        query=dict(type='assignment_field', field='room'), path_audit=previous.path_audit(source))


def independent_answer_bundle_card():
    """Both premises affect an answer set, but the coupling affects neither item."""
    source = dict(root_id='v221-controlled-independent-answer-bundle', source_id='synthetic-fixture-only',
        source_status='CONTROLLED_REGRESSION_ONLY_NOT_ADMITTED', turns=[
            dict(turn_ref='T1', scene_ref='S1', speaker='Jo', text='Booking X has room Hall or Lab, meal Veg or Meat, person Ann or Ben, and slot AM or PM. Exclude room Lab.'),
            dict(turn_ref='T2', scene_ref='S2', speaker='Jo', text='For that booking X, exclude meal Meat. If the person is Ann, the slot must be AM.')])
    values = [dict(domains=dict(room=['Hall', 'Lab'], meal=['Veg', 'Meat'], person=['Ann', 'Ben'], slot=['AM', 'PM']),
                   constraints=[{'not': {'room': 'Lab'}}]),
              dict(constraints=[{'not': {'meal': 'Meat'}}, {'if': {'person': 'Ann'}, 'then': {'slot': 'AM'}}])]
    nodes = [previous.fact(source, 'room_exclusion', 'T1', values[0]),
             previous.fact(source, 'meal_and_other_coupling', 'T2', values[1])]
    node = dict(node_id='combined', type='operation', kind='constraint_apply',
        inputs=['room_exclusion', 'meal_and_other_coupling'], claim='Hall and Veg are required independently of the person-slot relation.',
        rationale='Combine the declared constraints while preserving their independent scopes.', bridge=previous.bridge(source),
        rule=dict(type='constraint_filter', expected_result=None))
    node['rule']['expected_result'] = qa._relational_rule(node, values, probe=True)
    nodes.append(node)
    return source, dict(information_need='Which room and meal statements are required for booking X?',
        natural_value='PASS', uncertainties=[], dependency_graph=dict(nodes=nodes, answer_node='combined'),
        query=dict(type='entailed_choices', judgments=[
            dict(id='hall', where=dict(room='Hall'), quantifier='all'),
            dict(id='veg', where=dict(meal='Veg'), quantifier='all')]),
        path_audit=previous.path_audit(source))


def unqueried_transitive_edge_card():
    """A<C is new, but the requested A<B and D<E are explicit premises."""
    source = dict(root_id='v221-controlled-unqueried-transitivity', source_id='synthetic-fixture-only',
        source_status='CONTROLLED_REGRESSION_ONLY_NOT_ADMITTED', turns=[
            dict(turn_ref='T1', scene_ref='S1', speaker='Jo', text='For exhibition X, A must precede B and A must precede stage X.'),
            dict(turn_ref='T2', scene_ref='S2', speaker='Jo', text='For that exhibition X, stage X precedes C and D precedes E.')])
    values = [dict(precedence_edges=[['A', 'B'], ['A', 'X']]),
              dict(precedence_edges=[['X', 'C'], ['D', 'E']])]
    nodes = [previous.fact(source, 'ab_ax', 'T1', values[0]), previous.fact(source, 'xc_de', 'T2', values[1])]
    node = dict(node_id='closure', type='operation', kind='constraint_chain', inputs=['ab_ax', 'xc_de'],
        claim='A precedes C transitively; the requested A-B and D-E relations are direct.',
        rationale='Compute the closure without conflating queried and unqueried conclusions.',
        bridge=previous.bridge(source), rule=dict(type='precedence_closure', expected_result=None))
    node['rule']['expected_result'] = qa._relational_rule(node, values, probe=True)
    nodes.append(node)
    return source, dict(information_need='Which precedence relations hold among A, B, D, and E?',
        natural_value='PASS', uncertainties=[], dependency_graph=dict(nodes=nodes, answer_node='closure'),
        query=dict(type='partial_order', items=['A', 'B', 'D', 'E']), path_audit=previous.path_audit(source))


def state_card(l4=False):
    source, card = previous.state_card(l4)
    # Scalar status is deliberately not encoded as an internal answer list.
    for node in card['dependency_graph']['nodes']:
        value = node.get('symbolic_value', {})
        if 'state' in value:
            value['state']['values']['status'] = value['state']['values']['status'][0]
        if 'transition' in value:
            for key in ('when', 'set'):
                value['transition'][key]['status'] = value['transition'][key]['status'][0]
        result = node.get('rule', {}).get('expected_result', {})
        if 'state' in result:
            result['state']['values']['status'] = result['state']['values']['status'][0]
    card['query'] = dict(type='state_field', path=['state', 'values', 'status'], answer_kind='scalar')
    return source, card


def refresh_candidate(candidate, target, source):
    candidate['target_sha256'] = candidate['relation_sha256'] = qa.v2.fp(target)
    candidate['query_path_audit'] = dict(previous.path_audit(source),
        question_sha256=qa.fp(candidate['question']), options_sha256=qa.fp(candidate['options']),
        form_sha256=qa.candidate_form_fingerprint(candidate), critical_bridges_preserved=True,
        supplied_bridge_ids=[], minimum_route_level='L3')


def fixture(*, complete_sequence_single=False):
    """Explicitly migrate one old synthetic fixture; no real-source authoring."""
    source, discovery, target, candidate, scope = previous.fixture()
    card = discovery['relations'][0]
    values = {}
    for node in card['dependency_graph']['nodes']:
        if node['type'] == 'fact':
            values[node['node_id']] = copy.deepcopy(node.get('symbolic_value'))
        else:
            result = qa._relational_rule(node, [values[nid] for nid in node['inputs']], probe=True)
            node['rule']['expected_result'] = result
            values[node['node_id']] = result
    target['dependency_graph'] = copy.deepcopy(card['dependency_graph'])
    card['query'] = dict(type='full_order')
    discovery['schema_version'] = qa.VERSION + ':discovery'
    target['query'] = copy.deepcopy(card['query'])
    target['semantic_answer'] = dict(kind='sequence', value=target.pop('expected_answer'))
    target.pop('answer_projection')
    target['discovery_sha256'] = qa.v2.fp(discovery)
    target['selection_sha256'] = qa.fp(qa.rank_relations(discovery, source, scope))
    candidate['semantic_answer'] = copy.deepcopy(target['semantic_answer'])
    candidate['gold_answer'] = dict(kind='option_sequence', option_ids=candidate.pop('gold'))
    for analysis in candidate['item_analysis'].values():
        analysis['semantic_value'] = analysis.pop('answer_value')
    if complete_sequence_single:
        candidate['question_type'] = 'single_choice'
        candidate['question'] = 'Which complete sequence satisfies the exhibition requirements?'
        candidate['options'] = dict(abc='Installation, inspection, opening',
                                    acb='Installation, opening, inspection',
                                    bac='Inspection, installation, opening')
        candidate['gold_answer'] = dict(kind='single_option', option_id='abc')
        values = dict(abc=['installation', 'inspection', 'opening'],
                      acb=['installation', 'opening', 'inspection'],
                      bac=['inspection', 'installation', 'opening'])
        candidate['item_analysis'] = {key: dict(semantic_value=value,
            status='supported' if key == 'abc' else 'contradicted',
            evidence=previous.ev(source, 'T1', 'T2'), rationale='Check both required exhibition precedence links.',
            error_reason='This proposed complete sequence violates a required prerequisite.',
            error_path_type='reversed_prerequisite', error_path_turn_refs=['T1', 'T2'])
            for key, value in values.items()}
        candidate.pop('ordering_edges', None)
    refresh_candidate(candidate, target, source)
    return source, discovery, target, candidate, scope


def retarget(source, discovery, target, candidate, scope, query, information_need):
    card = discovery['relations'][0]
    card['query'] = copy.deepcopy(query)
    card['information_need'] = information_need
    target['query'] = copy.deepcopy(query)
    target['information_need'] = information_need
    symbolic = qa._graph_analysis(card['dependency_graph'], source)['symbolic_answer']
    target['semantic_answer'] = qa._query_answer(symbolic, query)
    target['discovery_sha256'] = qa.v2.fp(discovery)
    target['selection_sha256'] = qa.fp(qa.rank_relations(discovery, source, scope))
    candidate['semantic_answer'] = copy.deepcopy(target['semantic_answer'])


def plan_fixture(*, single=False):
    """Coherent multi-condition menu, with no source-admission claim."""
    source, card = constraint_card()
    # Equal-length names/rooms make this software fixture independent of the
    # inherited option-only surface-feature review, which remains unchanged.
    def rename(value):
        if isinstance(value, list): return [rename(item) for item in value]
        if isinstance(value, dict): return {key: rename(item) for key, item in value.items()}
        return {'Ann': 'Anna', 'Ben': 'Bert', 'Lab': 'Labs'}.get(value, value) if isinstance(value, str) else value
    source = rename(source)
    for turn in source['turns']:
        turn['text'] = turn['text'].replace('Ann', 'Anna').replace('Ben', 'Bert').replace('Lab', 'Labs')
    nodes = card['dependency_graph']['nodes']
    for node in nodes:
        if node['type'] == 'fact':
            node['symbolic_value'] = rename(node['symbolic_value'])
            refs = node['supports'][0]['turn_refs']
            node['supports'] = [previous.support(source, refs, 'Controlled booking constraint stated in this source turn.')]
        else:
            node['bridge'] = previous.bridge(source)
            node['rule']['expected_result'] = rename(node['rule']['expected_result'])
            node['claim'] = 'Either Anna or Cara can use Hall; the room is fixed.'
    plans = [dict(id='anna_hall', requirements=dict(person='Anna', room='Hall')),
             dict(id='cara_hall', requirements=dict(person='Cara', room='Hall')),
             dict(id='anna_labs', requirements=dict(person='Anna', room='Labs')),
             dict(id='bert_hall', requirements=dict(person='Bert', room='Hall'))]
    if single:
        plans = [plan for plan in plans if plan['id'] != 'cara_hall']
    question = 'Which booking arrangement can satisfy the recorded requirements?' if single else 'Which booking arrangements can satisfy the recorded requirements?'
    card.update(relation_id='controlled-plan-menu', endpoints=['X'], relation_type='joint_event_constraint',
        derivation_basis='Apply both availability and actor-specific room requirements to the offered complete arrangements.',
        decision='SELECT', evidence=previous.ev(source, 'T1', 'T2'), information_need=question,
        query=dict(type='feasible_plans', plans=plans, quantifier='any', require_unique=single),
        path_audit=previous.path_audit(source))
    discovery = dict(events=[dict(event_id='X', actors=['Jo'], action='book venue',
        objects=['person', 'room'], time=None, quantities=[],
        status_history=[dict(status='availability', turn_refs=['T1']), dict(status='room_rules', turn_refs=['T2'])],
        mentions=[dict(evidence=previous.ev(source, ref),
                       identity_basis='introduction' if ref == 'T1' else 'explicit_reference',
                       identity_rationale='Booking X is explicitly identified in the controlled sentence.')
                  for ref in ('T1', 'T2')])], relations=[card], source_scan=previous.scan(source))
    scope = {}
    previous.decorate(discovery, source, scope)
    discovery.update(schema_version=qa.VERSION + ':discovery', source_sha256=qa.fp(source), source_scope_sha256=qa.fp(scope))
    _, _, target, candidate, _ = fixture()
    target.update(relation_id=card['relation_id'], target='Feasible complete booking arrangements',
        relation_type=card['relation_type'], why_ask='Select an arrangement that satisfies the recorded booking constraints.',
        information_need=question, query=copy.deepcopy(card['query']),
        semantic_answer=qa._query_answer(nodes[-1]['rule']['expected_result'], card['query']),
        dependency_graph=copy.deepcopy(card['dependency_graph']), path_audit=copy.deepcopy(card['path_audit']))
    target['facts'] = [dict(fact_id=node['node_id'], claim=node['claim'], event_ids=['X'],
        evidence_turn_ids=list(node['supports'][0]['turn_refs']), evidence=copy.deepcopy(node['supports'][0]['evidence']),
        supports=copy.deepcopy(node['supports'])) for node in nodes if node['type'] == 'fact']
    target['inference_steps'] = [dict(step_id=node['node_id'], inputs=node['inputs'], conclusion=node['claim'])
                                 for node in nodes if node['type'] == 'operation']
    target['identity_links'] = [dict(event_ids=['X'], basis='explicit_reference', status='confirmed',
        evidence=previous.ev(source, 'T2'), rationale='That booking X explicitly refers to the previously named booking.')]
    target['necessity_groups'] = [dict(group_id=node['node_id'], fact_ids=[node['node_id']],
        turn_refs=node['supports'][0]['turn_refs'], why_necessary='This constraint excludes an otherwise feasible listed arrangement.',
        removal_effect='The answer set or uniqueness changes without this constraint.') for node in nodes if node['type'] == 'fact']
    target['evidence_certificate'] = dict(primary_turn_refs=['T1', 'T2'], all_minimal_support_sets=[['T1', 'T2']],
        necessity_claim='individual_turns', full_source_scan=[dict(turn_refs=[ref], classification='support',
            rationale='Controlled premise for feasible booking arrangements.') for ref in ('T1', 'T2')])
    target['discovery_sha256'] = qa.v2.fp(discovery)
    target['selection_sha256'] = qa.fp(qa.rank_relations(discovery, source, scope))
    option_text = dict(anna_hall='Anna books Hall.', cara_hall='Cara books Hall.',
                       anna_labs='Anna books Labs.', bert_hall='Bert books Hall.')
    candidate.update(root_id=source['root_id'], question_type='single_choice' if single else 'multi_select',
        question=question, semantic_answer=copy.deepcopy(target['semantic_answer']),
        options={plan['id']: option_text[plan['id']] for plan in plans},
        gold_answer=dict(kind='single_option', option_id='anna_hall') if single else
                    dict(kind='option_set', option_ids=['anna_hall', 'cara_hall']))
    good = {'anna_hall'} if single else {'anna_hall', 'cara_hall'}
    candidate['item_analysis'] = {plan['id']: dict(semantic_value=plan['id'],
        status='supported' if plan['id'] in good else 'contradicted',
        evidence=previous.ev(source, 'T1', 'T2'), rationale='Evaluate the complete arrangement against both recorded constraints.',
        error_reason='The proposed actor or room violates an explicitly recorded booking constraint.',
        error_path_type='actor_room_constraint_mismatch', error_path_turn_refs=['T1', 'T2']) for plan in plans}
    candidate.pop('ordering_edges', None)
    refresh_candidate(candidate, target, source)
    return source, discovery, target, candidate, scope


class V221QueryContracts(unittest.TestCase):
    def test_complete_order_is_typed_sequence_without_option_wrapper(self):
        answer = dict(order=['A', 'B', 'C'], vertices=['A', 'B', 'C'],
                      precedence_edges=[['A', 'B'], ['A', 'C'], ['B', 'C']])
        self.assertEqual(query_value(answer, {'type': 'full_order'}),
                         {'kind': 'sequence', 'value': ['A', 'B', 'C']})

    def test_partial_order_is_relation_set_not_a_forced_total_order(self):
        source, card = precedence_card(isolated=True)
        value = qa._graph_analysis(card['dependency_graph'], source)['symbolic_answer']
        self.assertIsNone(value['order'])
        self.assertEqual(query_value(value, dict(type='partial_order', items=['A', 'C', 'D'])),
                         dict(kind='relation_set', value=[['A', 'C']]))
        with self.assertRaises(qa.ValidationError):
            query_value(value, {'type': 'full_order'})

    def test_precedence_knows_false_only_from_reverse_entailment(self):
        value = dict(order=None, vertices=['A', 'B', 'C', 'D'],
                     precedence_edges=[['A', 'B'], ['B', 'C'], ['A', 'C']])
        self.assertEqual(query_value(value, dict(type='precedence', before='A', after='C')),
                         dict(kind='scalar', value=True))
        self.assertEqual(query_value(value, dict(type='precedence', before='C', after='A')),
                         dict(kind='scalar', value=False))
        with self.assertRaises(qa.ValidationError):
            query_value(value, dict(type='precedence', before='A', after='D'))

    def test_global_nonunique_assignment_has_unique_local_answer(self):
        source, card = constraint_card()
        symbolic = qa._graph_analysis(card['dependency_graph'], source)['symbolic_answer']
        self.assertEqual(len(symbolic['solutions']), 2)
        self.assertEqual(query_value(symbolic, card['query']), dict(kind='scalar', value='Hall'))
        with self.assertRaises(qa.ValidationError):
            query_value(symbolic, dict(type='assignment'))
        with self.assertRaises(qa.ValidationError):
            query_value(symbolic, dict(type='assignment_field', field='person'))

    def test_multi_condition_plan_quantifiers_have_distinct_meanings(self):
        symbolic = dict(solutions=[dict(person='Ann', room='Hall'), dict(person='Cara', room='Hall')])
        plans = [dict(id='hall', requirements=dict(room='Hall')),
                 dict(id='ann_hall', requirements=dict(person='Ann', room='Hall')),
                 dict(id='lab', requirements=dict(room='Lab'))]
        self.assertEqual(query_value(symbolic, dict(type='feasible_plans', plans=plans, quantifier='all', require_unique=True)),
                         dict(kind='scalar', value='hall'))
        self.assertEqual(query_value(symbolic, dict(type='feasible_plans', plans=plans, quantifier='any', require_unique=False)),
                         dict(kind='set', value=['ann_hall', 'hall']))
        with self.assertRaises(qa.ValidationError):
            query_value(symbolic, dict(type='feasible_plans', plans=plans, quantifier='any', require_unique=True))

    def test_entailed_choices_can_encode_more_than_one_local_judgment(self):
        symbolic = dict(solutions=[dict(person='Ann', room='Hall'), dict(person='Cara', room='Hall')])
        judgments = [dict(id='must_hall', where=dict(room='Hall'), quantifier='all'),
                     dict(id='may_ann', where=dict(person='Ann', room='Hall'), quantifier='any'),
                     dict(id='must_ann', where=dict(person='Ann'), quantifier='all')]
        self.assertEqual(query_value(symbolic, dict(type='entailed_choices', judgments=judgments)),
                         dict(kind='set', value=['may_ann', 'must_hall']))

    def test_scalar_state_is_not_forced_into_answer_list(self):
        self.assertEqual(query_value(dict(state=dict(values=dict(status='approved'))),
                         dict(type='state_field', path=['state', 'values', 'status'], answer_kind='scalar')),
                         dict(kind='scalar', value='approved'))

    def test_state_sequence_set_and_object_keep_distinct_semantics(self):
        symbolic = dict(state=dict(values=dict(sequence=['B', 'A'], rooms=['Hall', 'Lab'],
                                               booking=dict(status='approved', room='Hall'))))
        for key, kind, expected in [('sequence', 'sequence', ['B', 'A']),
                                    ('rooms', 'set', ['Hall', 'Lab']),
                                    ('booking', 'object', dict(status='approved', room='Hall'))]:
            self.assertEqual(query_value(symbolic, dict(type='state_field', path=['state', 'values', key], answer_kind=kind)),
                             dict(kind=kind, value=expected))

    def test_bad_query_fields_types_and_unknown_paths_fail_closed(self):
        value = dict(state=dict(values=dict(status='approved')))
        for query in ({}, {'type': 'invented'}, {'type': 'state_field', 'path': ['state', 'missing']},
                      {'type': 'state_field', 'path': ['state', 'values', 'status'], 'answer_kind': 'sequence'}):
            with self.subTest(query=query), self.assertRaises((qa.ValidationError, KeyError, TypeError)):
                query_value(value, query)

    def valid_query_shapes(self):
        return [dict(type='full_order'), dict(type='precedence', before='A', after='B'),
                dict(type='partial_order', items=['A', 'B']), dict(type='assignment'),
                dict(type='assignment_field', field='room'),
                dict(type='entailed_choices', judgments=[dict(id='hall', where=dict(room='Hall'), quantifier='all')]),
                dict(type='feasible_plans', plans=[dict(id='hall', requirements=dict(room='Hall'))], quantifier='all', require_unique=True),
                dict(type='state_field', path=['state', 'values', 'status'], answer_kind='scalar')]

    def query_shape_context(self):
        return dict(order=['A', 'B'], precedence_edges=[['A', 'B']],
                    solutions=[dict(person='Ann', room='Hall')], state=dict(values=dict(status='approved')))

    def test_query_schema_and_runtime_accept_all_eight_valid_shapes(self):
        import jsonschema
        schema = json.loads(qa.SCHEMA_PATH.read_text())
        contract = {'$defs': schema['$defs'], '$ref': '#/$defs/query'}
        for query in self.valid_query_shapes():
            with self.subTest(type=query['type']):
                jsonschema.validate(query, contract)
                self.assertIn('kind', query_value(self.query_shape_context(), query))

    def test_query_schema_and_runtime_reject_extra_and_missing_top_level_keys(self):
        import jsonschema
        schema = json.loads(qa.SCHEMA_PATH.read_text())
        contract = {'$defs': schema['$defs'], '$ref': '#/$defs/query'}
        for original in self.valid_query_shapes():
            for mode in ('extra', 'missing'):
                query = copy.deepcopy(original)
                if mode == 'extra': query['unexpected'] = 'must not be silently ignored'
                else: query.pop(next(key for key in reversed(query) if key != 'answer_kind'))
                with self.subTest(type=original['type'], mode=mode):
                    with self.assertRaises(jsonschema.ValidationError): jsonschema.validate(query, contract)
                    with self.assertRaises(qa.ValidationError): query_value(self.query_shape_context(), query)

    def test_query_schema_and_runtime_reject_extra_and_missing_nested_judgment_plan_keys(self):
        import jsonschema
        schema = json.loads(qa.SCHEMA_PATH.read_text())
        contract = {'$defs': schema['$defs'], '$ref': '#/$defs/query'}
        for original in self.valid_query_shapes()[5:7]:
            field = 'judgments' if original['type'] == 'entailed_choices' else 'plans'
            for mode in ('extra', 'missing'):
                query = copy.deepcopy(original)
                if mode == 'extra': query[field][0]['unexpected'] = 'ignored fields are forbidden'
                else: query[field][0].pop('id')
                with self.subTest(type=original['type'], mode=mode):
                    with self.assertRaises(jsonschema.ValidationError): jsonschema.validate(query, contract)
                    with self.assertRaises(qa.ValidationError): query_value(self.query_shape_context(), query)


class V221QueryConditionedDifficulty(unittest.TestCase):
    def test_local_transitive_query_retains_l3_despite_global_ambiguity(self):
        source, card = precedence_card(isolated=True)
        result = qa.classify_relation(card, source)
        self.assertTrue(result['eligible_for_hard'], result)
        self.assertEqual(result['level'], 'L3')
        self.assertTrue(result['query_dependency_audit'])
        self.assertFalse(result['semantic_quality_proved'])

    def test_direct_edge_cannot_borrow_unneeded_transitive_graph_for_l3(self):
        source, card = precedence_card()
        card['query'] = dict(type='precedence', before='A', after='B')
        self.assertFalse(qa.classify_relation(card, source)['eligible_for_hard'])

    def test_unknown_precedence_and_ambiguous_complete_order_hold(self):
        for query in (dict(type='precedence', before='A', after='D'), dict(type='full_order')):
            source, card = precedence_card(isolated=True)
            card['query'] = query
            self.assertFalse(qa.classify_relation(card, source)['eligible_for_hard'])

    def test_partial_order_query_retains_real_transitive_dependency(self):
        source, card = precedence_card(isolated=True)
        card['query'] = dict(type='partial_order', items=['A', 'C', 'D'])
        result = qa.classify_relation(card, source)
        self.assertTrue(result['eligible_for_hard'], result)
        self.assertEqual(result['level'], 'L3')

    def test_local_unique_constraint_answer_retains_joint_dependency(self):
        source, card = constraint_card()
        result = qa.classify_relation(card, source)
        self.assertTrue(result['eligible_for_hard'], result)
        self.assertEqual(result['level'], 'L3')
        self.assertTrue(result['query_dependency_audit'])
        self.assertEqual(result['graph_audit']['minimal_support_sets'], [['T1', 'T2']])

    def test_true_local_ambiguity_is_not_hidden_by_a_valid_constraint_graph(self):
        source, card = constraint_card()
        card['query'] = dict(type='assignment_field', field='person')
        self.assertFalse(qa.classify_relation(card, source)['eligible_for_hard'])

    def test_single_domain_probe_recovers_direct_answer_without_certifying_l3(self):
        source, card = irrelevant_coupling_card()
        graph = card['dependency_graph']
        only_direct = qa._relational_rule(graph['nodes'][-1],
            [graph['nodes'][0]['symbolic_value']], probe=True)
        self.assertEqual(qa._query_answer(only_direct, card['query']), dict(kind='scalar', value='Hall'))
        # The relaxed counterfactual is not a formally admissible high-order rule.
        with self.assertRaises(qa.ValidationError):
            qa._relational_rule(graph['nodes'][-1], [graph['nodes'][0]['symbolic_value']])

    def test_unrelated_coupling_cannot_promote_single_premise_local_room_answer(self):
        source, card = irrelevant_coupling_card()
        result = qa.classify_relation(card, source)
        self.assertFalse(result['eligible_for_hard'], result)
        self.assertTrue(any('query-irrelevant premise' in error for error in result['errors']), result)

    def test_unrelated_coupling_does_not_promote_judgment_or_plan_wrappers(self):
        queries = [dict(type='entailed_choices', judgments=[
                       dict(id='hall_required', where=dict(room='Hall'), quantifier='all')]),
                   dict(type='feasible_plans', plans=[dict(id='hall_plan', requirements=dict(room='Hall'))],
                        quantifier='all', require_unique=True)]
        for query in queries:
            source, card = irrelevant_coupling_card()
            card['query'] = query
            with self.subTest(query=query['type']):
                result = qa.classify_relation(card, source)
                self.assertFalse(result['eligible_for_hard'], result)
                self.assertTrue(any('query-irrelevant premise' in error for error in result['errors']), result)

    def test_fact_necessity_does_not_promote_independent_answer_bundle(self):
        source, card = independent_answer_bundle_card()
        graph = qa._graph_analysis(card['dependency_graph'], source)
        self.assertEqual(query_value(graph['symbolic_answer'], card['query']), dict(kind='set', value=['hall', 'veg']))
        result = qa.classify_relation(card, source)
        self.assertFalse(result['eligible_for_hard'], result)

    def test_partial_order_cannot_borrow_unqueried_transitive_conclusion(self):
        source, card = unqueried_transitive_edge_card()
        graph = qa._graph_analysis(card['dependency_graph'], source)
        self.assertIn(['A', 'C'], graph['symbolic_answer']['precedence_edges'])
        self.assertEqual(query_value(graph['symbolic_answer'], card['query']),
                         dict(kind='relation_set', value=[['A', 'B'], ['D', 'E']]))
        self.assertFalse(qa.classify_relation(card, source)['eligible_for_hard'])

    def test_scalar_state_update_and_coupled_state_chain_keep_l3_l4(self):
        for l4, expected in ((False, 'L3'), (True, 'L4')):
            source, card = state_card(l4)
            result = qa.classify_relation(card, source)
            self.assertTrue(result['eligible_for_hard'], result)
            self.assertEqual(result['level'], expected)

    def test_unaffected_state_field_cannot_inherit_status_update_difficulty(self):
        source, card = state_card()
        nodes = card['dependency_graph']['nodes']
        nodes[0]['symbolic_value']['state']['values']['room'] = 'Hall'
        nodes[-1]['rule']['expected_result']['state']['values']['room'] = 'Hall'
        card['query'] = dict(type='state_field', path=['state', 'values', 'room'], answer_kind='scalar')
        self.assertFalse(qa.classify_relation(card, source)['eligible_for_hard'])

    def test_sufficient_easy_and_unresolved_alternative_paths_remain_holds(self):
        for row in (dict(status='sufficient', level='L0', turn_refs=['T3'], rationale='Answer explicitly restated.'),
                    dict(status='unresolved', turn_refs=['T3'], rationale='Potential alternate identity not resolved.')):
            source, card = precedence_card()
            card['path_audit']['counterpaths'] = [row]
            self.assertFalse(qa.classify_relation(card, source)['eligible_for_hard'])

    def test_unmodeled_sufficient_hard_path_prevents_necessity_certification(self):
        source, card = precedence_card()
        card['path_audit']['counterpaths'] = [dict(status='sufficient', level='L3', turn_refs=['T2', 'T3'],
                                                  rationale='An alternative sufficient route remains unmodeled.')]
        self.assertFalse(qa.classify_relation(card, source)['eligible_for_hard'])

    def test_missing_query_does_not_default_to_full_order(self):
        source, card = precedence_card()
        card.pop('query')
        self.assertFalse(qa.classify_relation(card, source)['eligible_for_hard'])

    def test_historical_three_local_calculations_remain_rejected(self):
        run = BASE / 'base_qa_generation_v2_2_20261102/historical_cases'
        cases = [('109164_v2_61_minute_hold', BASE / 'feedback_replication_20root_resume_20261025_v1/sources/cc-train-episode-109164/FULL_SOURCE.json'),
                 ('103866_v21_arrival_gap', BASE / 'day3_production_v1/sources/cc-train-episode-103866.json'),
                 ('104814_v21_hold_endpoint', BASE / 'day3_production_v1/sources/cc-train-episode-104814.json')]
        for case, source_path in cases:
            with self.subTest(case=case):
                annotation = json.loads((run / (case + '.json')).read_text())
                target = json.loads(Path(annotation['target_binding']['path']).read_text())
                candidate = json.loads(Path(annotation['candidate_binding']['path']).read_text())
                result = qa.classify_legacy_target(target, json.loads(source_path.read_text()), candidate)
                self.assertEqual(result['level'], 'L1')
                self.assertFalse(result['eligible_for_hard'])


class V221TargetAndConstruction(unittest.TestCase):
    def assert_invalid(self, result, fragment=None):
        self.assertFalse(result['mechanical_pass'], result)
        if fragment:
            self.assertTrue(any(fragment in error for error in result['errors']), result)

    def test_existing_hard_order_fixture_keeps_target_and_ordering_valid(self):
        source, discovery, target, candidate, scope = fixture()
        for result in (qa.validate_discovery(discovery, source, scope),
                       qa.validate_target(target, discovery, source, scope),
                       qa.validate_candidate(candidate, target, discovery, source, scope)):
            self.assertTrue(result['mechanical_pass'], result)
        self.assertFalse(qa.validate_candidate(candidate, target, discovery, source, scope)['formal_qa_allowed'])

    def test_complete_semantic_sequence_can_be_a_single_choice_option(self):
        source, discovery, target, candidate, scope = fixture(complete_sequence_single=True)
        self.assertEqual(candidate['item_analysis']['abc']['semantic_value'], target['semantic_answer']['value'])
        self.assertTrue(qa.validate_candidate(candidate, target, discovery, source, scope)['mechanical_pass'])

    def test_partial_order_task_has_pairs_over_stable_option_ids(self):
        source, discovery, target, candidate, scope = fixture()
        retarget(source, discovery, target, candidate, scope,
                 dict(type='partial_order', items=['installation', 'inspection', 'opening']),
                 'Which prerequisite relations are required for exhibition X?')
        candidate.update(question_type='partial_order',
            question='Which prerequisite relations are required for exhibition X?',
            gold_answer=dict(kind='partial_order', pairs=[['i', 's'], ['s', 'o']]))
        refresh_candidate(candidate, target, source)
        self.assertTrue(qa.validate_candidate(candidate, target, discovery, source, scope)['mechanical_pass'])

    def test_multi_select_complete_plans_preserves_multi_condition_reasoning(self):
        source, discovery, target, candidate, scope = plan_fixture()
        result = qa.validate_candidate(candidate, target, discovery, source, scope)
        self.assertTrue(result['mechanical_pass'], result)
        self.assertEqual(target['semantic_answer'], dict(kind='set', value=['anna_hall', 'cara_hall']))
        self.assertEqual(candidate['gold_answer']['option_ids'], ['anna_hall', 'cara_hall'])

    def test_single_plan_menu_has_a_unique_answer_without_unique_global_assignment(self):
        source, discovery, target, candidate, scope = plan_fixture(single=True)
        self.assertEqual(len(target['dependency_graph']['nodes'][-1]['rule']['expected_result']['solutions']), 2)
        self.assertEqual(target['semantic_answer'], dict(kind='scalar', value='anna_hall'))
        result = qa.validate_candidate(candidate, target, discovery, source, scope)
        self.assertTrue(result['mechanical_pass'], result)

    def test_plan_options_cannot_change_the_frozen_answer_set(self):
        source, discovery, target, candidate, scope = plan_fixture()
        candidate['gold_answer']['option_ids'] = ['anna_hall', 'anna_labs']
        self.assert_invalid(qa.validate_candidate(candidate, target, discovery, source, scope))

    def test_transitive_local_boolean_query_can_use_single_choice(self):
        source, discovery, target, candidate, scope = fixture()
        retarget(source, discovery, target, candidate, scope,
                 dict(type='precedence', before='installation', after='opening'),
                 'Must installation finish before opening for exhibition X?')
        candidate.update(question_type='single_choice',
            question='Must installation finish before opening for exhibition X?',
            options=dict(yes='Installation must finish before opening.',
                         no='Opening must finish before installation.'),
            gold_answer=dict(kind='single_option', option_id='yes'))
        candidate['item_analysis'] = {key: dict(semantic_value=value,
            status='supported' if key == 'yes' else 'contradicted',
            evidence=previous.ev(source, 'T1', 'T2'), rationale='Both stated prerequisites establish the indirect relation.',
            error_reason='Reverses the required prerequisite relation.', error_path_type='reversed_prerequisite',
            error_path_turn_refs=['T1', 'T2']) for key, value in (('yes', True), ('no', False))}
        candidate.pop('ordering_edges', None)
        refresh_candidate(candidate, target, source)
        result = qa.validate_candidate(candidate, target, discovery, source, scope)
        self.assertTrue(result['mechanical_pass'], result)

    def test_target_semantic_gold_query_graph_and_bindings_fail_on_drift(self):
        for field in ('semantic_answer', 'query', 'dependency_graph', 'selection_sha256', 'discovery_sha256'):
            source, discovery, target, _, scope = fixture()
            if field == 'semantic_answer': target[field]['value'].reverse()
            elif field == 'query': target[field] = dict(type='precedence', before='installation', after='opening')
            elif field == 'dependency_graph': target[field]['nodes'][0]['claim'] += ' Drift.'
            else: target[field] = '0' * 64
            with self.subTest(field=field):
                self.assert_invalid(qa.validate_target(target, discovery, source, scope))

    def test_wrong_option_identity_or_sequence_gold_is_rejected(self):
        for mode in ('single', 'sequence'):
            source, discovery, target, candidate, scope = fixture(complete_sequence_single=mode == 'single')
            if mode == 'single': candidate['gold_answer']['option_id'] = 'acb'
            else: candidate['gold_answer']['option_ids'].reverse()
            self.assert_invalid(qa.validate_candidate(candidate, target, discovery, source, scope))

    def test_form_audit_binds_type_question_options_and_semantic_values(self):
        for field in ('question_type', 'question', 'options', 'semantic_value'):
            source, discovery, target, candidate, scope = fixture()
            if field == 'question_type': candidate[field] = 'multi_select'
            elif field == 'question': candidate[field] += ' Start at installation.'
            elif field == 'options': candidate[field]['i'] += ' stage'
            else: candidate['item_analysis']['i']['semantic_value'] = 'opening'
            with self.subTest(field=field):
                self.assert_invalid(qa.validate_candidate(candidate, target, discovery, source, scope))

    def test_bridge_leak_and_easy_or_unresolved_constructed_paths_remain_holds(self):
        for mode in ('bridge', 'easy', 'unresolved'):
            source, discovery, target, candidate, scope = fixture()
            if mode == 'bridge': candidate['query_path_audit']['supplied_bridge_ids'] = ['chain']
            else:
                candidate['query_path_audit']['counterpaths'] = [dict(
                    status='sufficient' if mode == 'easy' else 'unresolved', level='L1', turn_refs=['T1'],
                    rationale='Controlled construction-specific bypass for regression.')]
            self.assert_invalid(qa.validate_candidate(candidate, target, discovery, source, scope))

    def test_freeze_is_exclusive_and_reads_back_typed_answer(self):
        source, discovery, target, _, scope = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, 'TARGET.json')
            frozen = qa.freeze_target(path, target, discovery, source, scope)
            self.assertEqual(json.loads(path.read_text()), frozen)
            self.assertEqual(frozen['target']['semantic_answer'], target['semantic_answer'])
            with self.assertRaises(FileExistsError):
                qa.freeze_target(path, target, discovery, source, scope)

    def test_malformed_legacy_answer_field_cannot_override_typed_gold(self):
        source, discovery, target, candidate, scope = fixture(complete_sequence_single=True)
        candidate['gold'] = ['acb']
        self.assert_invalid(qa.validate_candidate(candidate, target, discovery, source, scope))

    def test_actual_pending_roots_stay_blocked(self):
        for root, ref in (('103866', 'R02:B:S03:T012'), ('104814', 'R02:B:S03:T016')):
            source = json.loads((BASE / f'day3_production_v1/sources/cc-train-episode-{root}.json').read_text())
            scope = json.loads((BASE / f'base_qa_v2_1_dual_root_real_20261101/cc-train-episode-{root}/SOURCE_SCOPE.json').read_text())
            gate = qa._formal_scope_gate(source, scope, [ref], 'Codex Challenger', True)
            self.assertFalse(gate['eligible'])
            self.assertIn('SOURCE_SCOPE_ADMISSION_INCOMPLETE', gate['errors'])


class V221ExactSemanticPathChecks(unittest.TestCase):
    def numeric_fixture(self):
        target = dict(semantic_answer=dict(kind='object', value=dict(
            allocation=dict(value='24', unit='count'), remaining=dict(value='18', unit='count'))),
            requires_deterministic_check=True, facts=[dict(fact_id='allocated'), dict(fact_id='reserved')])
        inputs = dict(allocation=dict(fact_id='allocated', value='24', unit='count'),
                      reservation=dict(fact_id='reserved', value='6', unit='count'))
        remaining = dict(operation='difference', inputs=inputs, operands=['allocation', 'reservation'],
            expected_value='18', unit='count', semantic_path=['remaining'])
        allocation = dict(operation='sum', inputs=dict(allocation=inputs['allocation']), operands=['allocation'],
            expected_value='24', unit='count', semantic_path=['allocation'])
        return target, dict(deterministic_checks=[allocation, remaining])

    def test_multiple_numeric_targets_bind_to_semantic_object_paths(self):
        target, candidate = self.numeric_fixture()
        checks = qa._deterministic_audit(candidate, target)
        self.assertEqual([check['result'] for check in checks], ['24', '18'])

    def test_every_numeric_path_requires_its_own_exact_check(self):
        target, candidate = self.numeric_fixture()
        candidate['deterministic_checks'] = candidate['deterministic_checks'][:1]
        with self.assertRaisesRegex(qa.ValidationError, 'every typed numeric answer'):
            qa._deterministic_audit(candidate, target)

    def test_legacy_answer_index_cannot_substitute_for_semantic_path(self):
        target, candidate = self.numeric_fixture()
        candidate['deterministic_checks'][0]['answer_index'] = 0
        with self.assertRaisesRegex(qa.ValidationError, 'semantic_path'):
            qa._deterministic_audit(candidate, target)

    def test_numeric_path_result_and_unit_mismatches_fail(self):
        for mode in ('path', 'result', 'unit', 'missing'):
            target, candidate = self.numeric_fixture()
            check = candidate['deterministic_checks'][1]
            if mode == 'path': check['semantic_path'] = ['allocation']
            elif mode == 'result': check['expected_value'] = '19'
            elif mode == 'unit': target['semantic_answer']['value']['remaining']['unit'] = 'minute'
            else: candidate['deterministic_checks'] = []
            with self.subTest(mode=mode), self.assertRaises((qa.ValidationError, ValueError)):
                qa._deterministic_audit(candidate, target)

    def test_numeric_sequence_paths_preserve_position_and_reject_boolean_index(self):
        target, candidate = self.numeric_fixture()
        target['semantic_answer'] = dict(kind='sequence', value=[dict(value='24', unit='count'), dict(value='18', unit='count')])
        for index, check in enumerate(candidate['deterministic_checks']): check['semantic_path'] = [index]
        self.assertEqual(len(qa._deterministic_audit(candidate, target)), 2)
        candidate['deterministic_checks'][0]['semantic_path'] = [False]
        with self.assertRaisesRegex(qa.ValidationError, 'index'):
            qa._deterministic_audit(candidate, target)


if __name__ == '__main__':
    unittest.main(verbosity=2)
