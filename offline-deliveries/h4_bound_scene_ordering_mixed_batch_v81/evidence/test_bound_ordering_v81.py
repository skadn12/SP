"""Behavioral tests for fixed scene/evidence bindings and unchanged review boundary."""
from copy import deepcopy
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
import ordering_generation as generation
import ordering_plan
import ordering_review
from common import bind, load_ordering_gate


class BoundOrderingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='v81_bound_ordering_')
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        self.source = {'root_id':'test-root','turns':[
            {'turn_ref':f'CANDIDATE:A:S{scene:02d}:T{i:03d}', 'speaker':'Actor',
             'text':f'Actor reports fact {scene}-{i}.'}
            for scene in (1,2) for i in range(1,7)]}
        self.source_path = self.home/'SOURCE.json'
        self.source_path.write_text(json.dumps(self.source))
        self.selection_path = self.home/'SCENE_PLANS.json'
        self.plans = {'plans':[{'root_id':'test-root','slot_number':scene,
            'scene_id':f'CANDIDATE:A:S{scene:02d}',
            'anchors':[{'turn_ref':t['turn_ref'],'speaker':t['speaker'],'quote':t['text']}
                       for t in self.source['turns'] if t['turn_ref'].rsplit(':',1)[0]==f'CANDIDATE:A:S{scene:02d}']}
            for scene in (1,2)]}
        self.selection_path.write_text(json.dumps(self.plans))
        self.config = {'ordering_anchor_plan':bind(self.selection_path)}
        self.slot = {'root_id':'test-root','source_sha256':bind(self.source_path)['sha256']}
        self.ids = ['test-root-ordering-01','test-root-ordering-02']
        self.plan = ordering_plan.build(self.config,self.slot,self.source,self.ids)
        self.plan_path = self.home/'ANCHOR_PLAN.json'
        self.plan_path.write_text(json.dumps(self.plan))
        self.manifest = {'candidate_ids':self.ids,'root_id':'test-root','anchor_plan':bind(self.plan_path)}
        positions = {t['turn_ref']:i for i,t in enumerate(self.source['turns'])}
        self.rows = [{'candidate_id':p['candidate_id'],'root_id':'test-root',
            'items':{k:v['quote'] for k,v in p['anchors'].items()},
            'gold':sorted(p['anchors'],key=lambda k:positions[p['anchors'][k]['turn_ref']]),
            'order_reason':'Source appearance order.'} for p in self.plan['questions']]

    def validate(self, rows=None, prior=None):
        with patch('socket.create_connection',side_effect=AssertionError('network forbidden')):
            return generation.validate_raw(json.dumps({'questions':self.rows if rows is None else rows}),
                self.manifest,self.source,self.source_path,prior or [])

    def rewrite_selection(self):
        self.selection_path.write_text(json.dumps(self.plans))
        self.config['ordering_anchor_plan'] = bind(self.selection_path)

    def test_program_binds_exact_scene_evidence_and_keeps_writer_items(self):
        accepted,rejected = self.validate()
        self.assertFalse(rejected)
        self.assertEqual(len(accepted),2)
        for raw,plan,q in zip(self.rows,self.plan['questions'],accepted):
            self.assertEqual(q['question'],plan['question'])
            self.assertEqual(q['items'],raw['items'])
            self.assertEqual(q['gold'],raw['gold'])
            self.assertEqual(q['original_gold'],raw['gold'])
            scope=load_ordering_gate().ordering_match_scope(q,self.source)
            self.assertEqual(scope['scene_refs'],[plan['scene_id']])
            self.assertEqual(len(scope['context_turn_refs']),12)
            self.assertEqual({k:{n:v[n] for n in ('turn_ref','speaker','quote')}
                              for k,v in q['item_evidence'].items()},plan['anchors'])

    def test_writer_cannot_override_scene_question_or_evidence(self):
        for field,value in [('scene_id','imaginary'),('question','forged'),('item_evidence',{})]:
            rows=deepcopy(self.rows);rows[0][field]=value
            accepted,rejected=self.validate(rows)
            self.assertEqual([q['candidate_id'] for q in accepted],[self.ids[1]])
            self.assertEqual(rejected,[{'candidate_id':self.ids[0],'reasons':['ordering_bound_writer_schema']}])

    def test_wrong_gold_is_rejected_never_corrected(self):
        rows=deepcopy(self.rows);rows[0]['gold']=rows[0]['gold'][::-1]
        before=deepcopy(rows)
        accepted,rejected=self.validate(rows)
        self.assertEqual(rows,before)
        self.assertEqual([q['candidate_id'] for q in accepted],[self.ids[1]])
        self.assertEqual(rejected[0]['reasons'],['writer_gold_disagrees_with_source_anchors'])

    def test_post_plan_anchor_tampering_is_rejected_before_use(self):
        bad=deepcopy(self.plan);bad['questions'][0]['anchors']['I1']['quote']='forged'
        self.plan_path.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError,'ordering_anchor_plan_changed'):
            self.validate()

    def test_even_rebound_source_plan_must_use_six_distinct_exact_single_scene_anchors(self):
        original=deepcopy(self.plans)
        for mutation in ('wrong_scene','wrong_speaker','wrong_quote','duplicate_turn'):
            self.plans=deepcopy(original)
            a=self.plans['plans'][0]['anchors'][0]
            if mutation=='wrong_scene':self.plans['plans'][0]['scene_id']='CANDIDATE:A:S02'
            if mutation=='wrong_speaker':a['speaker']='SomeoneElse'
            if mutation=='wrong_quote':a['quote']='Missing quotation.'
            if mutation=='duplicate_turn':self.plans['plans'][0]['anchors'][0]=deepcopy(self.plans['plans'][0]['anchors'][1])
            self.rewrite_selection()
            with self.subTest(mutation=mutation),self.assertRaisesRegex(ValueError,'ordering_plan_'):
                ordering_plan.build(self.config,self.slot,self.source,self.ids)

    def test_prior_public_whole_set_remains_excluded(self):
        good,_=self.validate()
        accepted,rejected=self.validate(prior=[good[0]])
        self.assertEqual([q['candidate_id'] for q in accepted],[self.ids[1]])
        self.assertEqual(rejected[0]['reasons'],['duplicate_prior_or_current_six_item_set'])

    def test_exact_anchor_cannot_certify_unsupported_writer_proposition(self):
        rows=deepcopy(self.rows);rows[0]['items']['I1']='Actor completed an unsupported overseas journey.'
        accepted,rejected=self.validate(rows)
        self.assertFalse(rejected)  # Mechanics have no claim to semantic support.
        q=accepted[0];public=generation.public(q)
        row=deepcopy(public)
        row.update(complete_source_turns_read=len(self.source['turns']),coherent=True,
                   unique_order=True,neutral_stem_ok=True,alternative_orders=[])
        row['items']={label:{'proposition':text,**{k:True for k in ordering_review.ITEM_BOOLEANS},
            'order_anchor':{k:q['item_evidence'][label][k] for k in ('turn_ref','quote')},
            'support_refs':[q['item_evidence'][label]['turn_ref']],'alternative_matching_anchors':[]}
            for label,text in public['items'].items()}
        row['items']['I1']['supported']=False
        result=ordering_review._review_row(row,public,self.source,load_ordering_gate())
        self.assertTrue(any('supported' in r for r in result['reasons']))


if __name__=='__main__':unittest.main()
