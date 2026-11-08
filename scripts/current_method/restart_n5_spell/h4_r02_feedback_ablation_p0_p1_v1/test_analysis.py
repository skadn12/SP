"""Meaningful denominator, missing-root and immutable-source regression checks."""
from pathlib import Path
import copy
import tempfile
import unittest
from .analysis import (group_metrics,selected_payload,blinded_case,audit_candidate_requests,
                       formatted_rate,CONTEXTS)
from .freeze import frozen_bytes,sha_obj
from ..h4_round0_b0_k3_difficulty_v1.contract import position_variants,score_output
from ..h4_round0_b0_k3_difficulty_v1.io_utils import stable_seed

class AnalysisTests(unittest.TestCase):
    def test_request_weighting_is_not_candidate_or_root_weighting(self):
        values=[{'root_id':'r1','question_type':'single_choice','correct':12,'scored':12,'requests':12,'accuracy':1.0},
                {'root_id':'r1','question_type':'multi_select','correct':0,'scored':15,'requests':15,'accuracy':0.0},
                {'root_id':'r2','question_type':'single_choice','correct':3,'scored':12,'requests':12,'accuracy':.25}]
        got=group_metrics(values,('r1','r2'),('single_choice','multi_select'))
        self.assertEqual(got['correct'],15); self.assertEqual(got['scored_requests'],39)
        self.assertAlmostEqual(got['request_weighted_accuracy'],15/39)
        self.assertAlmostEqual(got['candidate_mean_accuracy'],1.25/3)
        self.assertEqual(got['root_macro_candidate_accuracy'],.375)
        self.assertEqual(got['question_type_equal_macro_candidate_accuracy'],.3125)
    def test_missing_root_is_null_and_unscored_is_excluded_from_accuracy(self):
        value={'root_id':'r1','question_type':'single_choice','correct':1,'scored':2,'requests':3,'accuracy':.5}
        got=group_metrics([value],('r1','r2'),('single_choice',))
        self.assertIsNone(got['root_macro_candidate_accuracy']); self.assertEqual(got['missing_roots'],['r2'])
        self.assertEqual(got['unscored_requests'],1);self.assertEqual(got['request_weighted_accuracy'],.5)
        empty=group_metrics([],('r1',),('single_choice',)); self.assertIsNone(empty['candidate_mean_accuracy'])
    def test_review_hash_ignores_treatment_and_scores(self):
        original={'root_id':'r','candidate_id':'c','question':'q','question_type':'ordering','gold':['I1']}
        self.assertEqual(sha_obj(selected_payload(original)),sha_obj(selected_payload({**original,'arm':'feedback_on','accuracy':0})))
    def test_blind_payload_strips_arm_scores_and_provenance_at_all_levels(self):
        candidate={'root_id':'r','candidate_id':'c','question':'q','question_type':'single_choice',
                   'options':{'A':'yes','B':'no','C':'maybe','D':'later'},'gold':['A'],
                   'arm':'feedback_on','accuracy':0,'generation_source':'contract_guided',
                   'intent_id':'I001','source_paths':['/arm-path']}
        history={**candidate,'solver_observations':{'M0':{'accuracy':0.1}}}
        transcript={'root_id':'r','arm':'feedback_on','source_paths':['/arm-path'],
                    'turns':[{'turn_ref':'r:T1','speaker':'Maya','text':'yes',
                              'arm':'feedback_on','solver_scores':{},'source_paths':['/path']}]}
        case=blinded_case(candidate,history,transcript,['support'])
        forbidden={'arm','accuracy','generation_source','intent_id','candidate_id','source_paths','solver_observations','solver_scores'}
        def keys(value):
            if isinstance(value,dict):
                return set(value)|set().union(*(keys(x) for x in value.values()))
            if isinstance(value,list):
                return set().union(*(keys(x) for x in value))
            return set()
        self.assertFalse(keys(case)&forbidden)
        self.assertEqual(case['transcript']['turns'][0]['text'],'yes')
        self.assertEqual(case['candidate']['gold'],['A'])
    def test_missing_metrics_render_as_na(self):
        self.assertEqual(formatted_rate(None),'NA')
        self.assertEqual(formatted_rate(0),'0.000000')
        value={'root_id':'r1','question_type':'single_choice','correct':0,'scored':0,'requests':12,'accuracy':None}
        got=group_metrics([value],('r1',),('single_choice',))
        self.assertIsNone(got['candidate_mean_accuracy'])
        self.assertIsNone(got['root_macro_candidate_accuracy'])
        self.assertEqual(got['unscored_requests'],12)
    def test_raw_request_schedule_denominator_and_original_scoring(self):
        candidate={'root_id':'r','candidate_id':'c','question_type':'single_choice','question':'q',
                   'options':{'A':'yes','B':'no','C':'maybe','D':'later'},'gold':['A'],'correct_option':'A'}
        requests=[]
        for variant in position_variants(candidate):
            for context in CONTEXTS:
                for slot in range(3):
                    raw='<box>'+variant['displayed_gold'][0]+'</box>'
                    requests.append({'root_id':'r','candidate_id':'c','question_type':'single_choice',
                        'request_id':f"c__{variant['variant_id']}__{context}__s{slot:02d}",
                        'context_mode':context,'variant_id':variant['variant_id'],'slot_id':slot,
                        'seed':stable_seed('r','c',slot),'raw_output':raw,
                        **{k:variant[k] for k in ('display_to_canonical','displayed_gold','canonical_gold')},
                        **score_output(raw,candidate,variant)})
        result={'root_id':'r','candidate_id':'c','question_type':'single_choice','request_count':24,'requests':requests}
        self.assertEqual(audit_candidate_requests(candidate,result),24)
        duplicate=copy.deepcopy(result);duplicate['requests'][1]=copy.deepcopy(duplicate['requests'][0])
        with self.assertRaises(AssertionError):audit_candidate_requests(candidate,duplicate)
        seed=copy.deepcopy(result);seed['requests'][0]['seed']+=1
        with self.assertRaises(AssertionError):audit_candidate_requests(candidate,seed)
        score=copy.deepcopy(result);score['requests'][0]['correct']=False
        with self.assertRaises(AssertionError):audit_candidate_requests(candidate,score)
    def test_frozen_output_reuse_and_drift_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'file'; frozen_bytes(path,b'original'); frozen_bytes(path,b'original')
            with self.assertRaises(RuntimeError):frozen_bytes(path,b'changed')
            self.assertEqual(path.read_bytes(),b'original')

if __name__=='__main__': unittest.main()
