"""Bounded, offline software fixtures; no real QA/admission/model claims."""
from __future__ import annotations
import copy
from contextlib import redirect_stdout
from io import StringIO
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

try:
    from . import base_qa_pipeline_v2_2 as qa
    from .test_base_qa_pipeline_v2 import ordering_fixture, semantic_fixture, support, ev
except ImportError:
    import base_qa_pipeline_v2_2 as qa
    from test_base_qa_pipeline_v2 import ordering_fixture, semantic_fixture, support, ev


def scan(source):
    return [dict(turn_refs=[t['turn_ref']], rationale='Controlled source sentence examined.') for t in source['turns']]


def path_audit(source):
    return dict(source_scan=scan(source), counterpaths=[], natural_value_rationale='Required arrangement in the controlled source.', semantic_completeness_proved=False)


def decorate(discovery, source, scope):
    discovery.update(schema_version=qa.VERSION+':discovery', root_id=source['root_id'], status='READY',
                     source_sha256=qa.fp(source), source_scope_sha256=qa.fp(scope))
    discovery['coverage'] = dict(passes=list(qa.PASSES), claims_exhaustive=False, uncovered_ranges=[], checks=[
        dict(family=f, status='checked', scene_ids=sorted({r['scene_id'] for r in qa.source_index(source).values()}),
             event_ids=[e['event_id'] for e in discovery['events']], relation_ids=[r['relation_id'] for r in discovery['relations']],
             turn_refs=[t['turn_ref'] for t in source['turns']], rationale='Controlled inventory and relation family examined.') for f in qa.FAMILIES])


def query(candidate):
    candidate['query_path_audit'] = dict(path_audit(candidate.pop('_source')), question_sha256=qa.fp(candidate['question']),
        options_sha256=qa.fp(candidate['options']), critical_bridges_preserved=True,
        supplied_bridge_ids=[], minimum_route_level='L3')


def fixture(scope=None):
    source, discovery, target, candidate = ordering_fixture()
    scope = scope or {}
    nodes = [dict(node_id=f['fact_id'], type='fact', event_id='X', claim=f['claim'], supports=copy.deepcopy(f['supports']),
                  symbolic_value=dict(precedence_edges=[edge])) for f, edge in zip(target['facts'],
                    [['installation','inspection'], ['inspection','opening']])]
    expected = dict(order=target['expected_answer'], precedence_edges=sorted([
        ['installation','inspection'], ['inspection','opening'], ['installation','opening']]))
    step = target['inference_steps'][0]
    nodes.append(dict(node_id=step['step_id'], type='operation', kind='constraint_chain', inputs=step['inputs'],
        claim=step['conclusion'], rationale='Compose the two explicit required precedence constraints.',
        bridge=dict(identity_basis='explicit_reference', event_ids=['X'], identity_evidence=ev(source,'T2'),
                    rationale='That exhibition X continues the explicitly named exhibition.', change_or_constraint='Inspection links installation and opening.'),
        rule=dict(type='precedence_closure', expected_result=expected)))
    card = discovery['relations'][0]
    card.update(dependency_graph=dict(nodes=nodes, answer_node=step['step_id']), path_audit=path_audit(source))
    decorate(discovery, source, scope)
    target.update(discovery_sha256=qa.v2.fp(discovery), dependency_graph=copy.deepcopy(card['dependency_graph']),
                  path_audit=copy.deepcopy(card['path_audit']), answer_projection=dict(path=['order'], expected_answer=target['expected_answer']))
    target['selection_sha256'] = qa.fp(qa.rank_relations(discovery,source,scope))
    candidate.update(target_sha256=qa.v2.fp(target), relation_sha256=qa.v2.fp(target), _source=source)
    query(candidate)
    return source, discovery, target, candidate, scope


def fact(source, nid, ref, value, event='X'):
    return dict(node_id=nid, type='fact', event_id=event, claim='Controlled typed premise '+nid,
                supports=[support(source,[ref],'The controlled sentence supplies this typed premise.')], symbolic_value=value)


def bridge(source, ref='T2', events=None):
    return dict(identity_basis='explicit_reference', event_ids=events or ['X'], identity_evidence=ev(source,ref),
                rationale='Explicit identity/constraint link in controlled input.', change_or_constraint='Conditional update of the named event.')


def state_card(l4=False):
    source = dict(root_id='controlled-state-only', source_id='controlled', turns=[
        dict(turn_ref='T1',scene_ref='S1',speaker='Jo',text='Booking X begins in draft status.'),
        dict(turn_ref='T2',scene_ref='S2',speaker='Jo',text='If booking X is draft, mark it approved.'),
        dict(turn_ref='T3',scene_ref='S3',speaker='Jo',text='If booking X is approved, mark it booked.')])
    nodes=[fact(source,'initial','T1',dict(state=dict(event_id='X',values=dict(status=['draft'])))),
           fact(source,'change1','T2',dict(transition=dict(event_id='X',when=dict(status=['draft']),set=dict(status=['approved']))))]
    nodes.append(dict(node_id='update1',type='operation',kind='state_update',inputs=['initial','change1'],claim='X is approved.',
                      rationale='Apply the explicit conditional update.',bridge=bridge(source),
                      rule=dict(type='guarded_transition',expected_result=dict(state=dict(event_id='X',values=dict(status=['approved']))))))
    if l4:
        nodes += [fact(source,'change2','T3',dict(transition=dict(event_id='X',when=dict(status=['approved']),set=dict(status=['booked'])))),
                  dict(node_id='update2',type='operation',kind='state_update',inputs=['update1','change2'],claim='X is booked.',
                       rationale='The second update depends on the first update state.',bridge=bridge(source,'T3'),
                       rule=dict(type='guarded_transition',expected_result=dict(state=dict(event_id='X',values=dict(status=['booked'])))))]
    return source, dict(information_need='What is the current status of booking X?', natural_value='PASS', uncertainties=[],
                        dependency_graph=dict(nodes=nodes,answer_node='update2' if l4 else 'update1'),path_audit=path_audit(source))


def bind_file(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8')
    raw=path.read_bytes()
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def fixture_scope(tmp,source,author):
    """Fake receipt solely to test file contracts; never a production approval."""
    for t in source['turns']:
        t.update(provenance_type='original_root_dialogue',source_version='controlled-v1')
    binding=bind_file(Path(tmp,'SOURCE.json'),source)
    rows=[]
    for t in source['turns']:
        record=dict(schema_version='base_qa_turn_material_review_v2_1',source_sha256=binding['sha256'],
            source_id=source['source_id'],turn_id=t['turn_ref'],provenance_type='original_root_dialogue',
            source_version='controlled-v1',reviewer_id='fixture-material-reviewer',review_status='PASS',formal_qa_allowed=True)
        rb=bind_file(Path(tmp,t['turn_ref']+'.json'),record)
        rows.append(dict(turn_id=t['turn_ref'],source_id=source['source_id'],source_version='controlled-v1',
            provenance_type='original_root_dialogue',reviewer_id='fixture-material-reviewer',review_status='PASS',
            formal_qa_allowed=True,review_record=rb['path'],review_record_binding=rb))
    receipt=dict(schema_version=qa.v21.RECEIPT_SCHEMA,source_id=source['source_id'],source_sha256=binding['sha256'],
        review_scope='formal_qa_material_admission',reviewer_id='fixture-material-reviewer',decision='READY',turns=rows)
    rb=bind_file(Path(tmp,'ADMISSION.json'),receipt)
    return qa.v21.source_manifest(source,binding,receipt,rb,author)


def rebind(s,d,t,c,scope):
    decorate(d,s,scope)
    t.update(discovery_sha256=qa.v2.fp(d),selection_sha256=qa.fp(qa.rank_relations(d,s,scope)))
    c.update(target_sha256=qa.v2.fp(t),relation_sha256=qa.v2.fp(t),_source=s)
    query(c)


class V22CoreTests(unittest.TestCase):
    def assert_invalid(self,result,fragment=None):
        self.assertFalse(result['mechanical_pass'],result)
        if fragment:self.assertTrue(any(fragment in e for e in result['errors']),result)

    def test_staged_workplan_preserves_all_visible_context(self):
        s,d,_,_,scope=fixture()
        plan=qa.discovery_workplan(s,scope,d)
        self.assertEqual([x['stage'] for x in plan['stages']],['inventory','typed_sweep','composition_counterpath','ranking'])
        self.assertEqual(len(plan['stages'][1]['tasks']),10)
        self.assertEqual(set(plan['visible_turn_index']),{'T1','T2','T3'})
        self.assertTrue(all(qa.instruction(stage) for stage in ('inventory','typed_sweep','composition_counterpath','ranking','target','construction','postconstruction_audit')))

    def test_valid_order_path_reaches_target_freeze_and_construction(self):
        s,d,t,c,scope=fixture()
        self.assertTrue(qa.validate_discovery(d,s,scope)['mechanical_pass'])
        self.assertEqual(qa.rank_relations(d,s,scope)['eligible_relation_ids'],[t['relation_id']])
        for result in (qa.validate_target(t,d,s,scope),qa.validate_candidate(c,t,d,s,scope)):
            self.assertTrue(result['mechanical_pass'],result)
            self.assertFalse(result['semantic_quality_proved'])
            self.assertFalse(result['formal_qa_allowed'])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp,'TARGET.json'); frozen=qa.freeze_target(p,t,d,s,scope)
            self.assertEqual(json.loads(p.read_text()),frozen)
            with self.assertRaises(FileExistsError):qa.freeze_target(p,t,d,s,scope)

    def test_l0_l1_l2_are_retained_but_not_hard(self):
        s,d,_,_,_=fixture(); base=d['relations'][0]
        cases=[([fact(s,'one','T1',None)],'one','L0'),
               ([fact(s,'one','T1',None),fact(s,'two','T2',None),dict(node_id='a',type='operation',kind='arithmetic',inputs=['one','two'],claim='Simple scalar calculation.',rationale='Controlled subtraction.')],'a','L1'),
               ([fact(s,'one','T1',None),fact(s,'two','T2',None),dict(node_id='a',type='operation',kind='retrieve_bind',inputs=['one','two'],claim='Retrieve two facts.',rationale='No relational inference.')],'a','L2')]
        for nodes,answer,level in cases:
            card=copy.deepcopy(base);card['dependency_graph']=dict(nodes=nodes,answer_node=answer)
            r=qa.classify_relation(card,s)
            self.assertEqual(r['level'],level,r);self.assertFalse(r['eligible_for_hard'])

    def test_real_typed_state_update_and_coupled_l4(self):
        for l4,level in ((False,'L3'),(True,'L4')):
            s,card=state_card(l4);r=qa.classify_relation(card,s)
            self.assertEqual(r['level'],level,r);self.assertTrue(r['eligible_for_hard'])
            qa._projected_dependency(card['dependency_graph'],r['graph_audit'],dict(path=['state','values','status']))

    def test_independent_events_cannot_be_merged_by_transition(self):
        s,card=state_card();card['dependency_graph']['nodes'][1]['symbolic_value']['transition']['event_id']='Y'
        self.assertFalse(qa.classify_relation(card,s)['eligible_for_hard'])

    def test_symbolic_event_cannot_drift_from_source_bound_fact_event(self):
        s,card=state_card();nodes=card['dependency_graph']['nodes']
        nodes[0]['symbolic_value']['state']['event_id']='Y'
        nodes[1]['symbolic_value']['transition']['event_id']='Y'
        nodes[-1]['rule']['expected_result']['state']['event_id']='Y'
        r=qa.classify_relation(card,s)
        self.assertFalse(r['eligible_for_hard']);self.assertTrue(any('source-bound fact' in e for e in r['errors']))

    def test_wrong_state_guard_and_fabricated_rule_result_are_rejected(self):
        for mode in ('guard','result'):
            s,card=state_card()
            if mode=='guard':card['dependency_graph']['nodes'][1]['symbolic_value']['transition']['when']['status']=['other']
            else:card['dependency_graph']['nodes'][-1]['rule']['expected_result']['state']['values']['status']=['invented']
            self.assertFalse(qa.classify_relation(card,s)['eligible_for_hard'])

    def test_irrelevant_state_update_cannot_raise_query_difficulty(self):
        s,card=state_card()
        card['dependency_graph']['nodes'][0]['symbolic_value']['state']['values']['venue']=['Hall']
        card['dependency_graph']['nodes'][-1]['rule']['expected_result']['state']['values']['venue']=['Hall']
        r=qa.classify_relation(card,s)
        self.assertTrue(r['eligible_for_hard'],r)
        self.assertEqual(qa._answer_projection(r['graph_audit']['symbolic_answer'],dict(path=['state','values','venue'],expected_answer=['Hall'])),['Hall'])
        with self.assertRaisesRegex(qa.ValidationError,'does not affect'):
            qa._projected_dependency(card['dependency_graph'],r['graph_audit'],dict(path=['state','values','venue']))

    def test_bad_tag_cannot_promote_arithmetic(self):
        s,card=state_card();card['dependency_graph']['nodes'][-1].pop('rule')
        self.assertFalse(qa.classify_relation(card,s)['eligible_for_hard'])

    def test_executable_low_output_cannot_be_fabricated(self):
        s,card=state_card();nodes=card['dependency_graph']['nodes']
        nodes.insert(1,dict(node_id='fake',type='operation',kind='arithmetic',inputs=['initial'],
                           claim='Typed value without executed derivation.',rationale='Unverified value.',symbolic_value=nodes[0]['symbolic_value']))
        nodes[-1]['inputs'][0]='fake'
        self.assertFalse(qa.classify_relation(card,s)['eligible_for_hard'])

    def test_constraint_filter_requires_actual_coupling(self):
        node=dict(kind='constraint_apply',rule=dict(type='constraint_filter',expected_result=None))
        bad=[dict(domains=dict(person=['Ann'],room=['Hall']),constraints=[{'if':{'person':'Nobody'},'then':{'room':'Hall'}}])]
        with self.assertRaises(qa.ValidationError):qa._relational_rule(node,bad)
        bad[0]['constraints'][0]['if']['person']='Ann'
        with self.assertRaises(qa.ValidationError):qa._relational_rule(node,bad)

    def test_constraint_filter_positive_unique_assignment(self):
        values=[dict(domains=dict(person=['Ann','Ben'],room=['Hall','Lab']),constraints=[{'not':{'person':'Ben'}}]),
                dict(constraints=[{'if':{'person':'Ann'},'then':{'room':'Hall'}}])]
        expected=dict(domains=values[0]['domains'],constraints=values[0]['constraints']+values[1]['constraints'],solutions=[dict(person='Ann',room='Hall')])
        node=dict(kind='constraint_apply',rule=dict(type='constraint_filter',expected_result=expected))
        self.assertEqual(qa._relational_rule(node,values),expected)

    def test_symbolic_booleans_do_not_equal_integers(self):
        s,card=state_card()
        card['dependency_graph']['nodes'][0]['symbolic_value']['state']['values']['status']=True
        card['dependency_graph']['nodes'][1]['symbolic_value']['transition']['when']['status']=1
        self.assertFalse(qa.classify_relation(card,s)['eligible_for_hard'])
        self.assertFalse(qa._same(True,1));self.assertFalse(qa._same(False,0))

    def test_redundant_constraint_input_is_rejected(self):
        s,d,_,_,_=fixture();node=d['relations'][0]['dependency_graph']['nodes'][-1]
        values=[dict(precedence_edges=[['installation','inspection'],['inspection','opening']]),dict(precedence_edges=[['installation','inspection']])]
        with self.assertRaisesRegex(qa.ValidationError,'redundant'):
            qa._relational_rule(node,values)

    def test_cycles_unknown_refs_and_unused_nodes_fail_closed(self):
        for mode in ('cycle','ref','unused'):
            s,d,_,_,_=fixture();card=d['relations'][0];nodes=card['dependency_graph']['nodes']
            if mode=='cycle':nodes[-1]['inputs'][0]=nodes[-1]['node_id']
            elif mode=='ref':nodes[0]['supports'][0]['turn_refs']=['MISSING']
            else:nodes.insert(0,fact(s,'unused','T3',dict(value=9),event='Y'))
            self.assertFalse(qa.classify_relation(card,s)['eligible_for_hard'])

    def test_missing_scene_cannot_claim_cross_scene_hardness(self):
        s,d,_,_,_=fixture()
        for row in s['turns']:row.pop('scene_ref',None)
        r=qa.classify_relation(d['relations'][0],s)
        self.assertFalse(r['eligible_for_hard']);self.assertEqual(r['level'],'L2')

    def test_sufficient_context_only_summary_downgrades(self):
        s,d,_,_,_=fixture();card=d['relations'][0]
        card['path_audit']['counterpaths']=[dict(status='sufficient',level='L0',turn_refs=['T3'],rationale='Controlled directly given answer, even if context-only.')]
        r=qa.classify_relation(card,s,{'turns':[{'turn_id':'T3','formal_qa_allowed':False}]})
        self.assertEqual(r['level'],'L0');self.assertFalse(r['eligible_for_hard'])

    def test_unresolved_counterpath_and_ambiguous_identity_block(self):
        s,d,_,_,_=fixture();card=d['relations'][0]
        card['path_audit']['counterpaths']=[dict(status='unresolved',turn_refs=['T3'],rationale='Possible alternate event link remains unresolved.')]
        self.assertFalse(qa.classify_relation(card,s)['eligible_for_hard'])

    def test_sufficient_alternate_must_be_in_graph_before_necessity_claim(self):
        s,d,_,_,_=fixture();card=d['relations'][0]
        card['path_audit']['counterpaths']=[dict(status='sufficient',level='L3',turn_refs=['T2','T3'],rationale='Alternative sufficient support, not yet represented in graph.')]
        r=qa.classify_relation(card,s)
        self.assertFalse(r['eligible_for_hard']);self.assertTrue(r['unmodeled_sufficient_counterpaths'])

    def test_alternative_supports_and_deletion_sets_are_all_retained(self):
        s,d,_,_,_=fixture();card=d['relations'][0]
        s['turns'][2]['text']='For exhibition X, installation precedes inspection again.'
        card['dependency_graph']['nodes'][0]['supports'].append(support(s,['T3'],'Explicit restatement of the first premise.'))
        r=qa.classify_relation(card,s);a=r['graph_audit']
        self.assertEqual({frozenset(x) for x in a['minimal_support_sets']},{frozenset(['T1','T2']),frozenset(['T2','T3'])})
        deletions={x['removed_turn_ref']:x['remaining_paths'] for x in a['turn_deletions']}
        self.assertEqual(deletions['T1'],[['T2','T3']]);self.assertEqual(deletions['T2'],[])
        self.assertTrue(a['scene_deletions']);self.assertFalse(a['semantic_completeness_proved'])

    def test_independent_bridge_is_in_dependency_paths(self):
        s,d,_,_,_=fixture();card=d['relations'][0]
        card['dependency_graph']['nodes'][-1]['bridge']=bridge(s,'T3')
        r=qa.classify_relation(card,s)
        self.assertTrue(r['eligible_for_hard'],r)
        self.assertEqual(r['graph_audit']['minimal_support_sets'],[['T1','T2','T3']])

    def test_coverage_incomplete_is_explicit_and_prevents_selection(self):
        s,d,_,_,scope=fixture();d['coverage']['checks'][0]['status']='not_checked'
        self.assertTrue(qa.coverage_report(d,s)['errors'])
        d['coverage']['uncovered_ranges']=['Family not checked.']
        r=qa.rank_relations(d,s,scope)
        self.assertFalse(r['validation']['coverage']['operationally_complete'])
        self.assertEqual(r['status'],'INCOMPLETE_DISCOVERY_HOLD')

    def test_one_family_cannot_hide_unchecked_scenes_behind_other_families(self):
        s,d,_,_,_=fixture();row=d['coverage']['checks'][0]
        row.update(scene_ids=['S1'],turn_refs=['T1'])
        self.assertTrue(qa.coverage_report(d,s)['errors'])
        d['coverage']['uncovered_ranges']=['S2/S3 not checked for event identity.']
        r=qa.coverage_report(d,s)
        self.assertFalse(r['operationally_complete']);self.assertTrue(r['missing_by_family'])

    def test_unknown_coverage_family_event_scene_and_binding_fail(self):
        for field in ('family','event_ids','scene_ids','source_sha256','source_scope_sha256'):
            s,d,_,_,scope=fixture()
            if field.endswith('sha256'):d[field]='0'*64
            elif field=='family':d['coverage']['checks'][0][field]='invented'
            else:d['coverage']['checks'][0][field]=['invented']
            self.assert_invalid(qa.validate_discovery(d,s,scope))

    def test_low_and_unsupported_candidates_preserved_in_no_safe(self):
        s,d,_,_,scope=fixture();card=d['relations'][0]
        card.update(representation_status='UNSUPPORTED',representation_reason='Relation requires an unimplemented operator.',decision='HOLD')
        card['dependency_graph']='raw unsupported hypothesis retained verbatim'
        d.update(status='NO_SAFE_L3_L4',reason='No supported safe hard relationship; not a proof of absent supply.')
        r=qa.rank_relations(d,s,scope)
        self.assertEqual(r['status'],'NO_SAFE_L3_L4',r)
        self.assertEqual(r['relations'][0]['assessment']['status'],'UNSUPPORTED_RELATION_RETAINED')
        self.assertEqual(d['relations'][0]['dependency_graph'],'raw unsupported hypothesis retained verbatim')

    def test_low_relation_cannot_freeze_hard_target_and_is_not_discarded(self):
        s,d,t,_,scope=fixture();card=d['relations'][0]
        card['dependency_graph']['nodes'][-1].update(kind='arithmetic')
        card['decision']='HOLD';d.update(status='NO_SAFE_L3_L4',reason='Only local arithmetic remains.')
        r=qa.rank_relations(d,s,scope)
        self.assertEqual(r['status'],'NO_SAFE_L3_L4',r)
        self.assertEqual(r['relations'][0]['assessment']['level'],'L1')
        self.assert_invalid(qa.validate_target(t,d,s,scope),'TARGET_GATE')

    def test_actual_two_roots_pending_admission_remains_blocked(self):
        base=Path(__file__).resolve().parents[2]/'research-protocols/selfplay_week_20261019'
        for root,ref in (('103866','R02:B:S03:T012'),('104814','R02:B:S03:T016')):
            source=json.loads((base/f'day3_production_v1/sources/cc-train-episode-{root}.json').read_text())
            scope=json.loads((base/f'base_qa_v2_1_dual_root_real_20261101/cc-train-episode-{root}/SOURCE_SCOPE.json').read_text())
            r=qa._formal_scope_gate(source,scope,[ref],'Codex Challenger',True)
            self.assertFalse(r['eligible'],r)
            self.assertIn('SOURCE_SCOPE_ADMISSION_INCOMPLETE',r['errors'])

    def test_unsupported_cannot_be_selected_and_hold_is_not_selected(self):
        s,d,_,_,scope=fixture();card=d['relations'][0];card['decision']='HOLD'
        self.assertEqual(qa.rank_relations(d,s,scope)['eligible_relation_ids'],[])
        card.update(representation_status='UNSUPPORTED',representation_reason='Not modeled.',decision='SELECT')
        self.assert_invalid(qa.validate_discovery(d,s,scope))

    def test_target_gold_projection_and_graph_drift_fail(self):
        for mode in ('gold','projection','graph','selection','discovery'):
            s,d,t,_,scope=fixture()
            if mode=='gold':t['expected_answer']=list(reversed(t['expected_answer']))
            elif mode=='projection':t['answer_projection']['path']=['not_found']
            elif mode=='graph':t['dependency_graph']['nodes'][0]['claim']='Drifted premise.'
            elif mode=='selection':t['selection_sha256']='0'*64
            else:t['discovery_sha256']='0'*64
            self.assert_invalid(qa.validate_target(t,d,s,scope))

    def test_target_rejects_partial_order_projection(self):
        s,d,t,_,scope=fixture();t['answer_projection']=dict(path=['order',0],expected_answer='installation');t['expected_answer']='installation'
        self.assert_invalid(qa.validate_target(t,d,s,scope),'complete order')

    def test_query_audit_must_bind_wording_options_and_preserve_bridges(self):
        for mode in ('question','options','bridge','simple','unresolved','level'):
            s,d,t,c,scope=fixture()
            if mode=='question':c['question']+=' Installation comes first.'
            elif mode=='options':c['options']['i']='Installation (first)'
            elif mode=='bridge':c['query_path_audit']['supplied_bridge_ids']=['remaining']
            elif mode=='level':c['query_path_audit']['minimum_route_level']='L4'
            else:c['query_path_audit']['counterpaths']=[dict(status='sufficient' if mode=='simple' else 'unresolved',level='L1',turn_refs=['T1'],rationale='Controlled bypass.')]
            self.assert_invalid(qa.validate_candidate(c,t,d,s,scope))

    def test_pending_and_stale_scope_flags_never_authorize_formal_use(self):
        s,d,t,c,_=fixture();scope=dict(root_id=s['root_id'],source_sha256='0'*64,source_admission_complete=True,
            turns=[dict(turn_id=x['turn_ref'],formal_qa_allowed=True) for x in s['turns']])
        r=qa._formal_scope_gate(s,scope,['T1','T2'])
        self.assertFalse(r['eligible']);self.assertIn('SOURCE_SCOPE_VERSION_MISMATCH',r['errors'])
        scope['source_sha256']=qa.fp(s)
        self.assertFalse(qa._formal_scope_gate(s,scope,['T1','T2'],t['author_id'],True)['eligible'])

    def test_formal_facade_replays_receipts_and_freezes_fixture_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            s,d,t,c,_=fixture();scope=fixture_scope(tmp,s,t['author_id']);rebind(s,d,t,c,scope)
            p=Path(tmp,'TARGET.json');qa.freeze_target(p,t,d,s,scope)
            self.assertTrue(qa._formal_scope_gate(s,scope,['T1','T2'],t['author_id'],True)['eligible'])
            result=qa.prepare_frozen_review(Path(tmp,'freeze'),c,p,d,s,scope,reviewer_id='fixture-qa-reviewer',context_id='fresh-fixture',issuer_id='fixture-only',receipt_id='fixture-1')
            self.assertTrue(result)
            import importlib
            options=importlib.import_module(qa.__name__.rsplit('.',1)[0]+'.base_qa_option_freeze_v2_2') if '.' in qa.__name__ else importlib.import_module('base_qa_option_freeze_v2_2')
            self.assertEqual(set(options.frozen_mapping(Path(tmp,'freeze')).values()),set(c['options']))
            self.assertFalse(Path(tmp,'freeze','BLIND_RAW.bin').exists())

    def test_formal_facade_blocks_forged_scope_and_malformed_freeze(self):
        for mode in ('unapproved','extra','duplicate','selection'):
            with tempfile.TemporaryDirectory() as tmp:
                s,d,t,c,scope=fixture();p=Path(tmp,'TARGET.json');frozen=qa.freeze_target(p,t,d,s,scope)
                if mode=='extra':frozen['extra']=True;p.write_text(json.dumps(frozen))
                elif mode=='duplicate':p.write_text(json.dumps(frozen)[:-1]+',"version":"'+qa.VERSION+'"}')
                elif mode=='selection':frozen['selection_sha256']='0'*64;p.write_text(json.dumps(frozen))
                with self.assertRaises(ValueError):qa.prepare_frozen_review(Path(tmp,'freeze'),c,p,d,s,scope)
                self.assertFalse(Path(tmp,'freeze').exists())

    def test_schema_modeled_unsupported_and_projection_contracts(self):
        import jsonschema
        schema=json.loads(qa.SCHEMA_PATH.read_text());jsonschema.Draft202012Validator.check_schema(schema)
        s,d,t,_,_=fixture();jsonschema.validate(d,schema)
        jsonschema.validate(t,{'$defs':schema['$defs'],'$ref':'#/$defs/target'})
        d['relations'][0].update(representation_status='UNSUPPORTED',representation_reason='Retained unsupported hypothesis.',decision='HOLD')
        d['relations'][0].pop('dependency_graph');d['relations'][0].pop('path_audit');jsonschema.validate(d,schema)
        d['relations'][0]['decision']='SELECT'
        with self.assertRaises(jsonschema.ValidationError):jsonschema.validate(d,schema)

    def test_cli_workplan_rank_and_target_with_exclusive_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            s,d,t,_,scope=fixture()
            files={key:Path(tmp,key+'.json') for key in ('source','scope','discovery','target')}
            for key,value in (('source',s),('scope',scope),('discovery',d),('target',t)):files[key].write_text(json.dumps(value))
            common=['--source',str(files['source']),'--scope',str(files['scope'])]
            for stage,key in (('workplan','discovery'),('rank','discovery'),('target','target')):
                out=Path(tmp,stage+'_out.json');args=[stage,*common,'--input',str(files[key]),'--output',str(out)]
                if stage=='target':args+=['--discovery',str(files['discovery'])]
                with redirect_stdout(StringIO()):self.assertEqual(qa.main(args),0)
                self.assertTrue(json.loads(out.read_text()))
                with self.assertRaises(FileExistsError),redirect_stdout(StringIO()):qa.main(args)


if __name__=='__main__':unittest.main(verbosity=2)
