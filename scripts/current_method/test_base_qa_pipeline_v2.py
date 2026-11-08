"""Offline V2 regressions using explicit, controlled semantic propositions.

These fixtures are software regression cases, not historical admitted QA or
measurements of model quality. No model, API, Solver, or training is invoked.
"""
from __future__ import annotations

import copy
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

try:
    from . import base_qa_pipeline_v2 as qa
except ImportError:
    import base_qa_pipeline_v2 as qa


def admitted_turn(ref, text, scene='S1'):
    return dict(turn_ref=ref, speaker='Jo', text=text, scene_ref=scene,
                source_id='controlled-source', source_version='controlled-v1',
                provenance_type='original_dialogue', material_admission=dict(
                    review_status='PASS', formal_qa_allowed=True,
                    review_record='controlled-fixture-admission-only',
                    reviewer_id='fixture-material-reviewer'))


def ev(source, *refs):
    return qa.evidence(source, refs)


def support(source, refs, rationale):
    return dict(turn_refs=list(refs), evidence=ev(source, *refs),
                entailment_rationale=rationale)


def source_scan(source, relevant=('T1', 'T2')):
    return [dict(turn_refs=[t['turn_ref']],
                 classification='support' if t['turn_ref'] in relevant else 'irrelevant',
                 rationale=('Directly states a premise for consignment C4.'
                            if t['turn_ref'] in relevant else
                            'Concerns the separate C9 consignment; no transfer to C4.'))
            for t in source['turns']]


def semantic_fixture():
    source = dict(root_id='controlled-bottle-root', source_id='controlled-source',
                  warnings=['CONTROLLED FIXTURE: source admission is not a production claim.'],
                  source_status='CONTROLLED_REGRESSION_ONLY', turns=[
        admitted_turn('T1', 'For consignment C4, we allocated 24 bottles.'),
        admitted_turn('T2', 'Of those C4 bottles, reserve 6 for the speakers.', 'S2'),
        admitted_turn('T3', 'For the separate consignment C9, 3 boxes are ready.', 'S3')])
    discovery = dict(events=[
        dict(event_id='C4', actors=['Jo'], action='allocate bottles', objects=['bottles'],
             time=None, quantities=[dict(value=24, unit='bottles'), dict(value=6, unit='bottles')],
             status_history=[dict(status='allocated', turn_refs=['T1']),
                             dict(status='reserved_subset', turn_refs=['T2'])],
             mentions=[dict(evidence=ev(source, 'T1'), identity_basis='introduction',
                            identity_rationale='C4 is introduced by its explicit identity.'),
                       dict(evidence=ev(source, 'T2'), identity_basis='explicit_reference',
                            identity_rationale='Those C4 bottles explicitly refer to C4.')]),
        dict(event_id='C9', actors=['Jo'], action='prepare boxes', objects=['boxes'],
             time=None, quantities=[dict(value=3, unit='boxes')],
             status_history=[dict(status='ready', turn_refs=['T3'])],
             mentions=[dict(evidence=ev(source, 'T3'), identity_basis='introduction',
                            identity_rationale='Explicit separate C9 consignment.')])],
        relations=[dict(relation_id='remaining-c4', endpoints=['C4'],
                        relation_type='quantity_constraint',
                        derivation_basis='Subtract the explicitly reserved subset from the allocation.',
                        information_need='How many C4 bottles remain available for the audience?',
                        natural_value='PASS', uncertainties=[], decision='SELECT',
                        evidence=ev(source, 'T1', 'T2'))],
        source_scan=[dict(turn_refs=[t['turn_ref']], rationale=
                         'C4 quantity premise.' if t['turn_ref'] != 'T3' else
                         'Separate C9 event, not a C4 quantity premise.') for t in source['turns']])
    facts = [
        dict(fact_id='allocated', claim='C4 has 24 allocated bottles.', event_ids=['C4'],
             evidence_turn_ids=['T1'], evidence=ev(source, 'T1'),
             supports=[support(source, ['T1'], 'The sentence explicitly binds 24 bottles to C4.')]),
        dict(fact_id='reserved', claim='6 C4 bottles are reserved for speakers.', event_ids=['C4'],
             evidence_turn_ids=['T2'], evidence=ev(source, 'T2'),
             supports=[support(source, ['T2'], 'Those C4 bottles are an explicit subset of C4.')])]
    relation = dict(relation_id='remaining-c4', author_id='fixture-challenger',
        target='Audience-available bottle count for C4', relation_type='quantity_constraint',
        why_ask='The audience supply excludes the speaker reservation.',
        information_need=discovery['relations'][0]['information_need'],
        expected_answer=[dict(value='18', unit='count')], requires_deterministic_check=True,
        discovery_sha256=qa.fp(discovery), uncertainties=[],
        identity_requirement='same_event', identity_links=[dict(
            event_ids=['C4'], basis='explicit_reference', status='confirmed',
            evidence=ev(source, 'T2'), rationale='Those C4 bottles refers explicitly to C4.')],
        facts=facts, inference_steps=[dict(step_id='remaining', inputs=['allocated', 'reserved'],
                                         conclusion='24 minus 6 leaves 18 C4 bottles for the audience.')],
        necessity_groups=[
            dict(group_id='allocation', fact_ids=['allocated'], turn_refs=['T1'],
                 why_necessary='The starting allocation is required.',
                 removal_effect='The remaining count has no unique starting amount.'),
            dict(group_id='reservation', fact_ids=['reserved'], turn_refs=['T2'],
                 why_necessary='The reserved subset must be removed.',
                 removal_effect='The audience count has no unique reserved amount.')],
        evidence_certificate=dict(primary_turn_refs=['T1', 'T2'],
            all_minimal_support_sets=[['T1', 'T2']], necessity_claim='individual_turns',
            full_source_scan=source_scan(source)))
    values = {'c18': dict(value='18', unit='count'), 'c24': dict(value='24', unit='count'),
              'c6': dict(value='6', unit='count')}
    candidate = dict(candidate_id='controlled-c4-count', root_id=source['root_id'],
        author_id=relation['author_id'], relation_sha256=qa.fp(relation),
        target_sha256=qa.fp(relation), question_type='single_choice',
        question='How many C4 bottles remain available for the audience?',
        options={'c18': '18 bottles', 'c24': '24 bottles', 'c6': '6 bottles'}, gold=['c18'],
        item_analysis={}, deterministic_checks=[dict(operation='difference',
            inputs={'allocation': dict(fact_id='allocated', value='24', unit='count'),
                    'reservation': dict(fact_id='reserved', value='6', unit='count')},
            operands=['allocation', 'reservation'], expected_value='18', unit='count', answer_index=0)])
    for key, value in values.items():
        candidate['item_analysis'][key] = dict(answer_value=value,
            status='supported' if key == 'c18' else 'contradicted', evidence=ev(source, 'T1', 'T2'),
            rationale='Subtract the source-bound reservation from the C4 allocation.',
            error_reason='Confuses the remaining quantity with a stated component.',
            error_path_type='unsubtracted_allocation' if key == 'c24' else 'reserved_subset',
            error_path_turn_refs=['T1'] if key == 'c24' else ['T2'])
    return source, discovery, relation, candidate


def repeat_allocation_fixture(separate_fact=False):
    source, discovery, relation, candidate = semantic_fixture()
    source['turns'][2] = admitted_turn('T3', 'To confirm the allocation for C4: 24 bottles.', 'S3')
    discovery['source_scan'][2]['rationale'] = 'Repeated C4 allocation, an alternative source support.'
    discovery['events'][0]['mentions'].append(dict(evidence=ev(source, 'T3'),
        identity_basis='explicit_reference', identity_rationale='Explicit C4 allocation restatement.'))
    discovery['events'][1:] = []
    relation['discovery_sha256'] = qa.fp(discovery)
    relation['evidence_certificate'].update(all_minimal_support_sets=[['T1', 'T2'], ['T2', 'T3']],
        necessity_claim='premise_groups', full_source_scan=source_scan(source, ('T1','T2','T3')))
    relation['necessity_groups'][0]['turn_refs'] = ['T1', 'T3']
    if separate_fact:
        fact = copy.deepcopy(relation['facts'][0])
        fact.update(fact_id='allocation_repeat', evidence_turn_ids=['T3'], evidence=ev(source, 'T3'),
                    supports=[support(source, ['T3'], 'Explicit restatement of the same C4 allocation.')])
        relation['facts'].append(fact)
        relation['inference_steps'][0]['input_alternatives'] = [
            ['allocated', 'reserved'], ['allocation_repeat', 'reserved']]
        relation['necessity_groups'][0]['fact_ids'].append('allocation_repeat')
    else:
        relation['facts'][0]['supports'].append(support(source, ['T3'], 'Explicit C4 allocation restatement.'))
    candidate['relation_sha256'] = candidate['target_sha256'] = qa.fp(relation)
    return source, discovery, relation, candidate


def ordering_fixture():
    source, discovery, relation, candidate = semantic_fixture()
    source['root_id']='controlled-exhibition-root'
    source['turns'][0]['text'] = 'For exhibition X, installation I must finish before inspection S.'
    source['turns'][1]['text'] = 'For that exhibition X, inspection S must finish before opening O.'
    source['turns'][2]['text'] = 'A different exhibition Y has no confirmed timetable.'
    discovery['events'][0].update(event_id='X',action='prepare exhibition',
        objects=['installation','inspection','opening'],quantities=[],
        status_history=[dict(status='required_order',turn_refs=['T1','T2'])])
    discovery['events'][1].update(event_id='Y',action='prepare separate exhibition',
        objects=['timetable'],quantities=[],status_history=[dict(status='unconfirmed',turn_refs=['T3'])])
    for index, ref in enumerate(('T1','T2')):
        discovery['events'][0]['mentions'][index]['evidence'] = ev(source, ref)
        discovery['events'][0]['mentions'][index]['identity_rationale'] = (
            'Exhibition X is introduced explicitly.' if index==0 else 'That exhibition X explicitly refers to X.')
        relation['facts'][index]['evidence'] = ev(source, ref)
        relation['facts'][index]['event_ids'] = ['X']
        relation['facts'][index]['supports'] = [support(source, [ref], 'The source states this strict order edge.')]
    discovery['events'][1]['mentions'][0]['evidence'] = ev(source, 'T3')
    discovery['events'][1]['mentions'][0]['identity_rationale']='Y is explicitly described as a different exhibition.'
    discovery['relations'][0].update(endpoints=['X'],relation_type='temporal_order',
        evidence=ev(source,'T1','T2'),derivation_basis='Transitive order constraints for exhibition X.',
        information_need='What is the required order of installation, inspection, and opening?')
    for index,row in enumerate(discovery['source_scan']):
        row['rationale']='Required order constraint for X.' if index<2 else 'Separate exhibition Y has no constraint on X.'
    relation['identity_links'][0].update(event_ids=['X'],evidence=ev(source,'T2'),
        rationale='That exhibition X explicitly continues the same exhibition X.')
    relation.update(expected_answer=['installation','inspection','opening'],
                    requires_deterministic_check=False, discovery_sha256=qa.fp(discovery),
                    target='Required complete exhibition X order',relation_type='temporal_order',
                    why_ask='Opening depends on finishing prerequisite stages.',
                    information_need=discovery['relations'][0]['information_need'])
    relation['facts'][0]['claim'] = 'Installation precedes inspection.'
    relation['facts'][1]['claim'] = 'Inspection precedes opening.'
    relation['inference_steps'][0]['conclusion'] = 'Installation, inspection, opening is the unique strict order.'
    for group,necessity,effect in zip(relation['necessity_groups'],
            ['Installation-inspection edge is required.','Inspection-opening edge is required.'],
            ['Installation and inspection have no determined relative order.',
             'Inspection and opening have no determined relative order.']):
        group.update(why_necessary=necessity,removal_effect=effect)
    for index,row in enumerate(relation['evidence_certificate']['full_source_scan']):
        row['rationale']='Exhibition X order premise.' if index<2 else 'Separate exhibition Y provides no X order premise.'
    candidate.update(root_id=source['root_id'],question_type='ordering',question='What is the required order for exhibition X?',
        options={'i':'Installation','s':'Inspection','o':'Opening'}, gold=['i','s','o'],
        deterministic_checks=[], item_analysis={
            k:dict(answer_value=v, evidence=ev(source,'T1','T2'), rationale='A required exhibition stage.')
            for k,v in [('i','installation'),('s','inspection'),('o','opening')]},
        ordering_edges=[dict(before='i',after='s',evidence=ev(source,'T1')),
                        dict(before='s',after='o',evidence=ev(source,'T2'))])
    candidate['relation_sha256'] = candidate['target_sha256'] = qa.fp(relation)
    return source, discovery, relation, candidate


def invoke(*args):
    with redirect_stdout(StringIO()):
        return qa.main(list(map(str,args)))


class V2SemanticLedgerTests(unittest.TestCase):
    def test_positive_joint_quantity_target_and_candidate_are_retained(self):
        s,d,r,c = semantic_fixture()
        result = qa.validate_target(r,d,s,qa.source_manifest(s))
        self.assertTrue(result['mechanical_pass'], result)
        self.assertTrue(qa.v1.validate_candidate(c,r,s)['mechanical_pass'])
        self.assertFalse(result['semantic_quality_proved_by_code'])
        self.assertFalse(result['evidence_audit']['semantic_completeness_proved'])

    def test_event_updates_retain_explicit_same_event(self):
        s,d,_,_ = semantic_fixture()
        s['turns'][0]['text'] = 'Booking D17 is scheduled at 09:00 on 3 November.'
        s['turns'][1]['text'] = 'Move that D17 booking 45 minutes later; the date stays the same.'
        d['events'][0].update(event_id='D17', action='attend booking', quantities=[],
            status_history=[dict(status='proposed',turn_refs=['T1']),dict(status='modified',turn_refs=['T2'])])
        for index,ref in enumerate(('T1','T2')):d['events'][0]['mentions'][index]['evidence']=ev(s,ref)
        for index,mention in enumerate(d['events'][0]['mentions']):
            mention['identity_rationale']='D17 is introduced.' if index==0 else 'That D17 booking explicitly refers to D17.'
        d['relations'][0].update(endpoints=['D17'],relation_type='state_update',evidence=ev(s,'T1','T2'),
            derivation_basis='Apply the explicit 45-minute move to the same D17 booking.',
            information_need='What time should the person arrive for D17?')
        for index,row in enumerate(d['source_scan']):
            row['rationale']='D17 original time or its explicit update.' if index<2 else 'Separate consignment is irrelevant to D17.'
        self.assertEqual(qa.validate_discovery(d,s),[])

    def test_independent_events_stay_separate(self):
        s,d,_,_ = semantic_fixture()
        self.assertEqual([e['event_id'] for e in d['events']],['C4','C9'])
        self.assertEqual(qa.validate_discovery(d,s),[])

    def test_new_introduction_cannot_silently_merge_into_prior_event(self):
        s,d,_,_ = semantic_fixture()
        d['events'][0]['mentions'].append(copy.deepcopy(d['events'][1]['mentions'][0]))
        self.assertTrue(any('new introduction' in e for e in qa.validate_discovery(d,s)))

    def test_cross_scene_local_continuation_is_not_identity(self):
        s,d,_,_ = semantic_fixture()
        d['events'][0]['mentions'][1]['identity_basis']='local_continuation'
        self.assertTrue(any('cross-scene' in e for e in qa.validate_discovery(d,s)))

    def test_same_actor_or_similar_action_is_not_identity_link(self):
        s,d,r,_=semantic_fixture(); r['identity_links'][0]['basis']='same_actor'
        self.assertFalse(qa.validate_target(r,d,s,qa.source_manifest(s))['mechanical_pass'])

    def test_unknown_identity_status_cannot_support_target(self):
        s,d,r,_=semantic_fixture(); r['identity_links'][0]['status']='unknown'
        self.assertFalse(qa.validate_target(r,d,s,qa.source_manifest(s))['mechanical_pass'])

    def test_discovery_source_scan_requires_every_turn_exactly_once(self):
        for mode in ('omit','duplicate'):
            with self.subTest(mode=mode):
                s,d,_,_=semantic_fixture()
                if mode=='omit':d['source_scan'].pop()
                else:d['source_scan'].append(copy.deepcopy(d['source_scan'][0]))
                self.assertTrue(any('every source turn exactly once' in e for e in qa.validate_discovery(d,s)))

    def test_evidence_scan_requires_every_source_turn(self):
        s,_,r,_=semantic_fixture(); r['evidence_certificate']['full_source_scan'].pop()
        result=qa.evidence_audit(r,s,qa.source_manifest(s))
        self.assertFalse(result['mechanical_pass'])
        self.assertTrue(any('entire source' in e for e in result['errors']))

    def test_possible_alternative_with_unknown_identity_is_hold(self):
        s,_,r,_=semantic_fixture()
        r['evidence_certificate']['full_source_scan'][2].update(
            classification='possible_alternative',resolution='unknown',
            rationale='A similar event could suggest a transfer, but no explicit identity connects it.')
        result=qa.evidence_audit(r,s,qa.source_manifest(s))
        self.assertFalse(result['mechanical_pass'])
        self.assertTrue(any('unresolved alternative' in e for e in result['errors']))

    def test_excluding_possible_alternative_requires_exact_evidence(self):
        s,_,r,_=semantic_fixture()
        r['evidence_certificate']['full_source_scan'][2].update(
            classification='possible_alternative',resolution='excluded_with_evidence',resolution_evidence=[],
            rationale='C9 is explicitly separate from C4.')
        self.assertFalse(qa.evidence_audit(r,s,qa.source_manifest(s))['mechanical_pass'])

    def test_no_safe_relation_is_allowed_but_requires_reason(self):
        s,_,_,_=semantic_fixture()
        self.assertEqual(qa.validate_discovery(dict(status='NO_SAFE_RELATION',reason='No supported natural relation.'),s),[])
        self.assertTrue(qa.validate_discovery(dict(status='NO_SAFE_RELATION'),s))

    def test_mechanical_fact_bundle_cannot_be_selected_as_natural(self):
        s,d,_,_=semantic_fixture(); d['relations'][0]['natural_value']='FAIL'
        d['relations'][0]['information_need']='List a bottle quantity together with an unrelated box quantity.'
        self.assertTrue(qa.validate_discovery(d,s))

    def test_proof_paths_and_supports_are_joint_not_independent(self):
        _,_,r,_=semantic_fixture()
        self.assertEqual(qa.proof_paths(r),[frozenset(['T1','T2'])])
        r['facts'][0]['supports'][0]['turn_refs']=['T1','T3']
        self.assertEqual(qa.proof_paths(r),[frozenset(['T1','T2','T3'])])

    def test_or_supports_preserve_real_repetition_without_false_citation_necessity(self):
        s,d,r,_=repeat_allocation_fixture()
        result=qa.validate_target(r,d,s,qa.source_manifest(s))
        self.assertTrue(result['mechanical_pass'],result)
        audit=result['evidence_audit']
        self.assertEqual(audit['minimal_support_sets'],[['T1','T2'],['T2','T3']])
        self.assertEqual(audit['alternative_sufficient_path_count'],1)
        individual={v['removed_turn_ref']:v for v in audit['individual_deletions']}
        self.assertFalse(individual['T1']['necessary_in_ledger'])
        self.assertTrue(individual['T2']['necessary_in_ledger'])
        self.assertTrue(all(g['necessary_in_ledger'] for g in audit['premise_group_deletions']))

    def test_duplicate_support_cannot_claim_individual_necessity(self):
        s,_,r,_=repeat_allocation_fixture()
        r['evidence_certificate']['necessity_claim']='individual_turns'
        audit=qa.evidence_audit(r,s,qa.source_manifest(s))
        self.assertFalse(audit['mechanical_pass'])
        self.assertTrue(any('individual evidence necessity' in e for e in audit['errors']))

    def test_or_inference_alternatives_are_retained_by_target_validation(self):
        s,d,r,c=repeat_allocation_fixture(separate_fact=True)
        self.assertEqual(qa.proof_paths(r),[frozenset(['T1','T2']),frozenset(['T2','T3'])])
        result=qa.validate_target(r,d,s,qa.source_manifest(s))
        self.assertTrue(result['mechanical_pass'],result)
        candidate_audit=qa.validate_candidate(c,r,s)
        self.assertTrue(candidate_audit['mechanical_pass'],candidate_audit)

    def test_direct_answer_is_minimal_alternative_and_breaks_joint_necessity(self):
        s,_,r,_=semantic_fixture()
        s['turns'][2]['text']='The audience has 18 bottles available from C4.'
        r['facts'].append(dict(fact_id='direct_remaining',claim='C4 audience supply is 18 bottles.',
            event_ids=['C4'], evidence_turn_ids=['T3'],evidence=ev(s,'T3'),
            supports=[support(s,['T3'],'Direct statement of the requested remaining amount.')]))
        r['inference_steps'][0]['input_alternatives']=[['allocated','reserved'],['direct_remaining']]
        r['necessity_groups'].append(dict(group_id='direct',fact_ids=['direct_remaining'],turn_refs=['T3'],
            why_necessary='Test direct-answer bypass.',removal_effect='An arithmetic path remains.'))
        r['evidence_certificate'].update(all_minimal_support_sets=[['T3'],['T1','T2']],
                                        full_source_scan=source_scan(s,('T1','T2','T3')))
        audit=qa.evidence_audit(r,s,qa.source_manifest(s))
        self.assertEqual(audit['minimal_support_sets'],[['T3'],['T1','T2']])
        self.assertFalse(audit['mechanical_pass'])
        self.assertTrue(any(g['surviving_paths']==[['T3']] for g in audit['premise_group_deletions']))
        self.assertTrue(any(g['sufficient_alone_paths']==[['T3']] for g in audit['premise_group_deletions']))

    def test_primary_support_must_be_minimal_and_all_paths_disclosed(self):
        s,_,r,_=repeat_allocation_fixture()
        r['evidence_certificate'].update(primary_turn_refs=['T1','T2','T3'],
                                        all_minimal_support_sets=[['T1','T2']])
        result=qa.evidence_audit(r,s,qa.source_manifest(s))
        self.assertFalse(result['mechanical_pass'])
        self.assertTrue(any('not a minimal' in e for e in result['errors']))
        self.assertTrue(any('disclose every' in e for e in result['errors']))

    def test_exact_source_quotes_are_enforced_for_supports(self):
        s,_,r,_=semantic_fixture(); r['facts'][0]['supports'][0]['evidence'][0]['quote']='There are 25 bottles.'
        self.assertFalse(qa.evidence_audit(r,s,qa.source_manifest(s))['mechanical_pass'])

    def test_support_evidence_and_declared_refs_must_match(self):
        s,_,r,_=semantic_fixture(); r['facts'][0]['supports'][0]['turn_refs']=['T3']
        result=qa.evidence_audit(r,s,qa.source_manifest(s))
        self.assertFalse(result['mechanical_pass'])
        self.assertTrue(any('references must bind' in e for e in result['errors']))

    def test_or_branch_cannot_use_forward_unknown_or_self_node(self):
        for node in ('missing','remaining'):
            with self.subTest(node=node):
                s,_,r,_=semantic_fixture()
                r['inference_steps'][0]['input_alternatives']=[['allocated','reserved'],[node]]
                self.assertFalse(qa.evidence_audit(r,s,qa.source_manifest(s))['mechanical_pass'])

    def test_support_antichain_removes_supersets_without_losing_paths(self):
        _,_,r,_=semantic_fixture()
        r['facts'][0]['supports'].extend([dict(turn_refs=['T1','T3']),dict(turn_refs=['T3'])])
        self.assertEqual(qa.proof_paths(r),[frozenset(['T1','T2']),frozenset(['T2','T3'])])

    def test_path_limit_stops_instead_of_silently_truncating(self):
        with self.assertRaisesRegex(ValueError,'limit exceeded'):
            qa._minimal([{str(i)} for i in range(513)])

    def test_and_product_limit_stops_instead_of_silently_truncating(self):
        _,_,r,_=semantic_fixture()
        for n,fact in enumerate(r['facts']):
            fact['supports']=[dict(turn_refs=[str(n)+'-'+str(i)]) for i in range(65)]
        with self.assertRaisesRegex(ValueError,'product limit'):
            qa.proof_paths(r)

    def test_unknown_material_admission_is_research_only_not_formally_allowed(self):
        s,_,r,_=semantic_fixture(); s['turns'][1].pop('material_admission')
        audit=qa.evidence_audit(r,s,qa.source_manifest(s))
        self.assertTrue(audit['mechanical_pass'],audit)
        self.assertFalse(audit['formal_evidence_eligible'])
        self.assertEqual(audit['ineligible_turn_refs'],['T2'])

    def test_primary_fact_evidence_must_match_a_declared_support(self):
        s,_,r,_=semantic_fixture(); r['facts'][0]['evidence_turn_ids']=['T3']
        result=qa.evidence_audit(r,s,qa.source_manifest(s))
        self.assertFalse(result['mechanical_pass'])
        self.assertTrue(any('primary fact evidence' in e for e in result['errors']))

    def test_information_need_cannot_change_after_discovery(self):
        s,d,r,_=semantic_fixture(); r['information_need']='Which separate consignment has three boxes?'
        self.assertFalse(qa.validate_target(r,d,s,qa.source_manifest(s))['mechanical_pass'])

    def test_identity_requirement_must_be_explicit(self):
        s,d,r,_=semantic_fixture(); r['identity_requirement']='unnecessary'
        self.assertFalse(qa.validate_target(r,d,s,qa.source_manifest(s))['mechanical_pass'])

    def test_same_event_cannot_bind_two_event_ids(self):
        s,d,r,_=semantic_fixture(); r['facts'][1]['event_ids']=['C9']
        self.assertFalse(qa.validate_target(r,d,s,qa.source_manifest(s))['mechanical_pass'])

    def test_unadmitted_identity_or_exclusion_evidence_is_in_closure(self):
        for use in ('identity','exclusion'):
            with self.subTest(use=use):
                s,_,r,_=semantic_fixture(); s['turns'][2].pop('material_admission')
                if use=='identity':
                    r['identity_links'][0]['evidence']=ev(s,'T3')
                else:
                    r['evidence_certificate']['full_source_scan'][2].update(
                        classification='possible_alternative',resolution='excluded_with_evidence',
                        resolution_evidence=ev(s,'T3'),rationale='Explicitly separate consignment C9.')
                result=qa.evidence_audit(r,s,qa.source_manifest(s))
                self.assertFalse(result['formal_evidence_eligible'])
                self.assertIn('T3',result['ineligible_turn_refs'])

    def test_untraceable_or_unknown_provenance_admission_is_not_allowed(self):
        for field in ('review_record','reviewer_id','provenance_type'):
            with self.subTest(field=field):
                s,_,_,_=semantic_fixture()
                if field=='provenance_type':s['turns'][0][field]='unknown'
                else:s['turns'][0]['material_admission'].pop(field)
                self.assertFalse(qa.source_manifest(s)['turns'][0]['formal_qa_allowed'])

    def test_empty_or_nontext_provenance_cannot_be_admitted(self):
        for provenance in ('',None,'   '):
            with self.subTest(provenance=provenance):
                s,_,_,_=semantic_fixture(); s['turns'][0]['provenance_type']=provenance
                self.assertFalse(qa.source_manifest(s)['turns'][0]['formal_qa_allowed'])

    def test_formal_admission_preserves_synthetic_provenance(self):
        s,_,_,_=semantic_fixture(); s['turns'][0]['provenance_type']='audited_synthetic'
        row=qa.source_manifest(s)['turns'][0]
        self.assertTrue(row['formal_qa_allowed'])
        self.assertEqual(row['provenance_type'],'audited_synthetic')

    def test_v1_exact_calculation_is_reused_without_claiming_source_grounding(self):
        s,_,r,c=semantic_fixture(); result=qa.v1.validate_candidate(c,r,s)
        self.assertTrue(result['mechanical_pass'],result)
        checked=qa.v1.evaluate_check(c['deterministic_checks'][0],{'allocated','reserved'})
        self.assertEqual(checked['result'],'18')
        self.assertFalse(checked['source_value_grounding_verified'])
        c['deterministic_checks'][0]['expected_value']='19'
        self.assertFalse(qa.v1.validate_candidate(c,r,s)['mechanical_pass'])

    def test_v1_mismatched_units_do_not_pass_arithmetic(self):
        s,_,r,c=semantic_fixture(); c['deterministic_checks'][0]['inputs']['reservation']['unit']='minute'
        self.assertFalse(qa.v1.validate_candidate(c,r,s)['mechanical_pass'])

    def test_v1_unique_sorting_is_reused(self):
        s,_,r,c=ordering_fixture()
        self.assertTrue(qa.v1.validate_candidate(c,r,s)['mechanical_pass'])
        c['ordering_edges'].pop()
        self.assertFalse(qa.v1.validate_candidate(c,r,s)['mechanical_pass'])

    def test_v1_ordering_cycles_and_wrong_gold_are_rejected(self):
        for mode in ('cycle','wrong_gold'):
            with self.subTest(mode=mode):
                s,_,r,c=ordering_fixture()
                if mode=='cycle':c['ordering_edges'].append(dict(before='o',after='i',evidence=ev(s,'T1','T2')))
                else:c['gold']=['s','i','o']
                self.assertFalse(qa.v1.validate_candidate(c,r,s)['mechanical_pass'])

    def test_target_is_frozen_before_type_wording_or_options(self):
        for field,value in [('gold',['c18']),('options',{'c18':'18'}),('question','How many?'),('question_type','single_choice')]:
            with self.subTest(field=field):
                s,d,r,_=semantic_fixture(); r[field]=value
                self.assertFalse(qa.validate_target(r,d,s,qa.source_manifest(s))['mechanical_pass'])

    def test_discovery_hash_and_answer_semantics_cannot_drift(self):
        s,d,r,c=semantic_fixture(); r['discovery_sha256']='changed'
        self.assertFalse(qa.validate_target(r,d,s,qa.source_manifest(s))['mechanical_pass'])
        s,d,r,c=semantic_fixture(); c['item_analysis']['c18']['answer_value']=dict(value='19',unit='count')
        self.assertFalse(qa.v1.validate_candidate(c,r,s)['mechanical_pass'])

    def test_style_audit_flags_length_shortcut_without_claiming_semantics(self):
        _,_,_,c=semantic_fixture(); c['options']['c18']='There are exactly eighteen bottles in the carefully allocated audience supply'
        audit=qa.style_audit(c)
        self.assertIn('OPTION_LENGTH_IMBALANCE_REQUIRES_REVIEW',audit['flags'])
        self.assertFalse(audit['semantic_shortcuts_proved_absent'])

    def test_blind_packet_has_complete_source_and_warnings_without_author_fields(self):
        s,_,r,c=semantic_fixture(); manifest=qa.source_manifest(s)
        c['private_certificate']='AUTHOR_ONLY_SENTINEL'; c['gold_explanation']='AUTHOR_ONLY_SENTINEL'
        packet=qa.blind_packet(c,s,manifest)
        self.assertEqual(packet['source_scope']['warnings'],s['warnings'])
        self.assertEqual(len(packet['source']['turns']),len(s['turns']))
        self.assertEqual(packet['source']['turns'][0]['provenance']['source_version'],'controlled-v1')
        self.assertEqual(set(packet['options']),set(c['options']))
        self.assertNotIn('AUTHOR_ONLY_SENTINEL',json.dumps(packet))
        def keys(node):
            if isinstance(node,dict):
                for key,val in node.items():yield key; yield from keys(val)
            elif isinstance(node,list):
                for val in node:yield from keys(val)
        forbidden={'gold','expected_answer','facts','inference_steps','evidence_certificate',
                   'item_analysis','deterministic_checks','relation','discovery','target_sha256'}
        self.assertFalse(forbidden & set(keys(packet)))
        self.assertIn('same-model',packet['reviewer_mode'])


class V2FrozenWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name); self.run=self.base/'run'
        self.source,self.discovery,self.relation,self.candidate=semantic_fixture()
        self.source_path=self.base/'source.json'; self.dump(self.source_path,self.source)

    def dump(self,path,value):
        path.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8')

    def call_stage(self,stage,value):
        path=self.base/(stage+'.input.json'); self.dump(path,value)
        return invoke(stage,'--run-dir',self.run,'--input',path)

    def prepare(self):
        invoke('prepare','--source',self.source_path,'--run-dir',self.run)
        scope=qa.frozen(self.run,'INPUT')['source_scope']
        self.relation['source_scope_sha256']=qa.fp(scope)
        self.candidate['source_scope_sha256']=qa.fp(scope)
        self.candidate['target_sha256']=self.candidate['relation_sha256']=qa.fp(self.relation)

    def through_target(self):
        self.prepare(); self.call_stage('freeze-discovery',self.discovery)
        self.call_stage('freeze-target',self.relation)

    def through_candidate(self):
        self.through_target(); self.call_stage('freeze-candidate',self.candidate)

    def blind_answer(self):
        packet=qa.frozen(self.run,'BLIND_PACKET')
        manifest=qa.frozen(self.run,'INPUT')['source_scope']
        return dict(packet_sha256=qa.fp(packet),source_scope_sha256=qa.fp(manifest),
            reviewer_id='isolated-fixture-reviewer',model_id='offline controlled fixture; no model call',
            reviewer_mode='same_model_blind',decision='SOLVED',independent_gold=['c18'],ambiguities=[],
            isolation_receipt=dict(context_mode='fresh_context_no_history',author_material_visible=False,
                                   context_id='controlled-fixture-isolation-receipt'),
            evidence=ev(self.source,'T1','T2'),reasoning_steps=['24 allocated minus 6 reserved leaves 18.'])

    def comparison(self,blind):
        return dict(blind_sha256=qa.fp(blind),candidate_sha256=qa.fp(self.candidate),
            reviewer_id=blind['reviewer_id'],decision='PASS',
            checks={k:dict(decision='PASS',rationale='Controlled fixture semantic judgment; not production evidence.') for k in qa.CHECKS},
            necessity_review=[dict(group_id=g['group_id'],removed_turn_refs=g['turn_refs'],
                target_underdetermined=True,single_group_insufficient=True,no_remaining_bypass=True,
                rationale='Removing this premise makes the controlled remaining count indeterminate.')
                for g in self.relation['necessity_groups']])

    def test_freeze_order_rejects_target_before_discovery(self):
        self.prepare()
        with self.assertRaises((FileNotFoundError,ValueError)):
            self.call_stage('freeze-target',self.relation)
        self.assertFalse((self.run/'TARGET.json').exists())

    def test_freeze_order_rejects_candidate_before_target(self):
        self.prepare(); self.call_stage('freeze-discovery',self.discovery)
        with self.assertRaises((FileNotFoundError,ValueError)):
            self.call_stage('freeze-candidate',self.candidate)
        self.assertFalse((self.run/'CANDIDATE.json').exists())

    def test_gold_frozen_before_candidate_and_comparison_after_blind(self):
        self.through_target()
        self.assertEqual(qa.frozen(self.run,'TARGET')['expected_answer'],[dict(value='18',unit='count')])
        self.assertFalse((self.run/'CANDIDATE.json').exists())
        self.call_stage('freeze-candidate',self.candidate)
        self.assertFalse((self.run/'COMPARISON_PACKET.json').exists())
        blind=self.blind_answer(); self.call_stage('freeze-blind',blind)
        self.assertEqual(qa.frozen(self.run,'BLIND_SOLUTION'),blind)
        self.assertTrue((self.run/'COMPARISON_PACKET.json').exists())

    def test_frozen_artifacts_cannot_be_overwritten_and_drift_is_detected(self):
        self.through_target()
        with self.assertRaises(FileExistsError):self.call_stage('freeze-target',self.relation)
        modified=copy.deepcopy(self.relation); modified['expected_answer']=[dict(value='19',unit='count')]
        self.dump(self.run/'TARGET.json',modified)
        with self.assertRaisesRegex(ValueError,'frozen artifact drift'):qa.frozen(self.run,'TARGET')

    def test_source_content_drift_rejects_next_stage(self):
        self.prepare(); changed=copy.deepcopy(self.source); changed['turns'][0]['text']='Different source.'
        self.dump(self.source_path,changed)
        with self.assertRaisesRegex(ValueError,'source binding drift'):
            self.call_stage('freeze-discovery',self.discovery)

    def test_candidate_requires_target_fingerprint(self):
        self.through_target(); self.candidate['target_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'target freeze binding'):
            self.call_stage('freeze-candidate',self.candidate)

    def test_target_and_candidate_require_scope_binding(self):
        self.prepare(); self.call_stage('freeze-discovery',self.discovery)
        target=copy.deepcopy(self.relation); target.pop('source_scope_sha256')
        with self.assertRaisesRegex(ValueError,'target source scope'):
            self.call_stage('freeze-target',target)
        self.call_stage('freeze-target',self.relation)
        self.candidate.pop('source_scope_sha256')
        with self.assertRaisesRegex(ValueError,'candidate source scope'):
            self.call_stage('freeze-candidate',self.candidate)

    def test_no_safe_relation_command_stops_without_author_packet(self):
        self.prepare()
        self.assertEqual(self.call_stage('freeze-discovery',dict(
            status='NO_SAFE_RELATION',reason='No naturally useful supported relation.')),0)
        self.assertEqual(qa.read(self.run/'DECISION.json')['decision'],'NO_SAFE_RELATION')
        self.assertFalse((self.run/'TARGET_PACKET.json').exists())
        self.assertFalse((self.run/'AUTHOR_PACKET.json').exists())
        with self.assertRaisesRegex(ValueError,'NO_SAFE_RELATION has no target'):
            self.call_stage('freeze-target',self.relation)

    def test_logical_audit_tampering_is_detected_before_comparison(self):
        self.through_candidate(); changed=qa.frozen(self.run,'LOGICAL_AUDIT')
        changed['evidence_audit']['formal_evidence_eligible']=False
        self.dump(self.run/'LOGICAL_AUDIT.json',changed)
        with self.assertRaisesRegex(ValueError,'frozen artifact drift: LOGICAL_AUDIT'):
            self.call_stage('freeze-blind',self.blind_answer())

    def test_rehashed_logical_audit_is_recomputed_at_finalize(self):
        self.through_candidate(); blind=self.blind_answer(); self.call_stage('freeze-blind',blind)
        changed=qa.frozen(self.run,'LOGICAL_AUDIT')
        changed['evidence_audit']['alternative_sufficient_path_count']=99
        self.dump(self.run/'LOGICAL_AUDIT.json',changed)
        self.dump(self.run/'LOGICAL_AUDIT_FREEZE.json',dict(sha256=qa.fp(changed),version=qa.VERSION))
        self.assertEqual(self.call_stage('finalize',self.comparison(blind)),2)
        self.assertIn('logical audit drift',qa.read(self.run/'DECISION.json')['errors'])

    def test_distractor_error_path_cannot_reference_missing_source(self):
        self.through_target(); self.candidate['item_analysis']['c24']['error_path_turn_refs']=['MISSING_TURN']
        with self.assertRaises(ValueError):self.call_stage('freeze-candidate',self.candidate)

    def test_complete_positive_workflow_stays_static_and_without_solver(self):
        self.through_candidate(); blind=self.blind_answer(); self.call_stage('freeze-blind',blind)
        self.assertEqual(self.call_stage('finalize',self.comparison(blind)),0)
        result=qa.read(self.run/'DECISION.json')
        self.assertEqual(result['decision'],'PASS_STATIC_REVIEW')
        self.assertTrue(result['formal_qa_admitted'])
        self.assertFalse(result['independent_model_cross_validation'])
        self.assertFalse(result['solver_run'])
        self.assertFalse(result['empirical_quality_established'])

    def test_complete_positive_or_branch_workflow_is_retained(self):
        self.source,self.discovery,self.relation,self.candidate=repeat_allocation_fixture(separate_fact=True)
        self.dump(self.source_path,self.source)
        self.through_candidate(); blind=self.blind_answer(); self.call_stage('freeze-blind',blind)
        self.assertEqual(self.call_stage('finalize',self.comparison(blind)),0)
        self.assertEqual(qa.read(self.run/'DECISION.json')['decision'],'PASS_STATIC_REVIEW')

    def test_unadmitted_source_is_hold_even_with_controlled_review_pass(self):
        self.source['turns'][1].pop('material_admission'); self.dump(self.source_path,self.source)
        self.through_candidate(); blind=self.blind_answer(); self.call_stage('freeze-blind',blind)
        self.assertEqual(self.call_stage('finalize',self.comparison(blind)),2)
        result=qa.read(self.run/'DECISION.json')
        self.assertEqual(result['decision'],'HOLD')
        self.assertFalse(result['formal_qa_admitted'])
        self.assertTrue(any('not formally admitted' in e for e in result['errors']))

    def test_unadmitted_candidate_error_path_is_in_admission_closure(self):
        self.source['turns'][2].pop('material_admission'); self.dump(self.source_path,self.source)
        self.candidate['item_analysis']['c24']['error_path_turn_refs']=['T3']
        self.through_candidate(); blind=self.blind_answer(); self.call_stage('freeze-blind',blind)
        self.assertEqual(self.call_stage('finalize',self.comparison(blind)),2)
        self.assertTrue(any('candidate evidence or distractor path' in e for e in qa.read(self.run/'DECISION.json')['errors']))

    def test_missing_blind_isolation_receipt_cannot_admit(self):
        self.through_candidate(); blind=self.blind_answer(); blind.pop('isolation_receipt')
        self.call_stage('freeze-blind',blind)
        self.assertEqual(self.call_stage('finalize',self.comparison(blind)),2)
        self.assertTrue(any('isolation receipt' in e for e in qa.read(self.run/'DECISION.json')['errors']))

    def test_reviewer_scope_acknowledgement_and_same_model_label_are_required(self):
        self.through_candidate(); blind=self.blind_answer()
        blind.update(reviewer_mode='independent_model',source_scope_sha256='incorrect')
        self.call_stage('freeze-blind',blind)
        self.assertEqual(self.call_stage('finalize',self.comparison(blind)),2)
        result=qa.read(self.run/'DECISION.json')
        self.assertFalse(result['formal_qa_admitted'])
        self.assertTrue(any('review scope/mode' in e for e in result['errors']))

    def test_blind_hold_is_preserved_and_cannot_finalize_as_pass(self):
        self.through_candidate(); blind=self.blind_answer()
        blind.update(decision='HOLD',independent_gold=[],ambiguities=['Controlled unresolved answer.'])
        self.call_stage('freeze-blind',blind)
        self.assertEqual(qa.frozen(self.run,'BLIND_SOLUTION')['decision'],'HOLD')
        self.assertEqual(self.call_stage('finalize',self.comparison(blind)),2)
        self.assertEqual(qa.read(self.run/'DECISION.json')['decision'],'HOLD')

    def test_author_and_blind_reviewer_cannot_be_same_identity(self):
        self.through_candidate(); blind=self.blind_answer()
        blind['reviewer_id']=self.candidate['author_id']
        self.call_stage('freeze-blind',blind)
        self.assertEqual(self.call_stage('finalize',self.comparison(blind)),2)
        self.assertTrue(any('independent reviewer identity' in e
                            for e in qa.read(self.run/'DECISION.json')['errors']))

    def test_blind_disagreement_does_not_change_frozen_author_gold(self):
        self.through_candidate(); blind=self.blind_answer(); blind['independent_gold']=['c24']
        self.call_stage('freeze-blind',blind)
        self.assertEqual(self.call_stage('finalize',self.comparison(blind)),2)
        self.assertEqual(qa.frozen(self.run,'CANDIDATE')['gold'],['c18'])
        self.assertEqual(qa.frozen(self.run,'BLIND_SOLUTION')['independent_gold'],['c24'])
        self.assertTrue(any('Gold disagreement' in e for e in qa.read(self.run/'DECISION.json')['errors']))


if __name__=='__main__':
    unittest.main()
