import json
from pathlib import Path
import tempfile
import unittest
from copy import deepcopy
import generate

class SlotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/'source.json'
        self.source = {'root_id':'r','turns':[{'turn_ref':'T1','speaker':'A','text':'First fact. Second fact.'}]}
        self.path.write_text(json.dumps(self.source))
        self.q = {'candidate_id':'q1','topic':'T','question':'Which? Select all that apply.',
                  'options':[{'proposition':'statement '+str(i),'relation':'supports' if i<3 else 'contradicts',
                              'turn_ref':'missing','speaker':None,'quote':None} for i in range(6)]}
    def tearDown(self): self.temp.cleanup()
    def check(self, questions, abstentions, ids=('q1','q2')):
        return generate.validate_draft(json.dumps({'root_id':'r','questions':questions,'abstentions':abstentions}),
                                       list(ids),'r','multi_select',self.source,self.path)
    def test_duplicate_abstention_keeps_valid_sibling_and_raw_reasons(self):
        ok, bad, abst, err=self.check([self.q],[{'candidate_id':'q2','reason':'short'},{'candidate_id':'q2','reason':'long'}])
        self.assertEqual(len(ok),1);self.assertFalse(bad);self.assertIsNone(err)
        self.assertEqual(abst[0]['format_diagnostics']['duplicate_abstention_count'],1)
        self.assertEqual(len(abst[0]['format_diagnostics']['raw_abstention_rows']),2)
    def test_conflict_rejects_only_its_slot(self):
        q2=deepcopy(self.q);q2['candidate_id']='q2'
        ok,bad,abst,_=self.check([self.q,q2],[{'candidate_id':'q2','reason':'none'}])
        self.assertEqual([q['candidate_id'] for q in ok],['q1'])
        self.assertEqual(bad[0]['errors'],['conflicting_or_duplicate_slot'])
    def test_missing_and_unknown_rows_do_not_drop_complete_question(self):
        ok,bad,_,_=self.check([self.q,{'candidate_id':'unexpected'}],[])
        self.assertEqual(len(ok),1);self.assertEqual(bad[0]['errors'],['candidate_slot_missing'])
        self.assertTrue(ok[0]['generation_format_diagnostics']['unknown_or_unassigned_rows'])
    def test_two_question_versions_never_choose_arbitrarily(self):
        altered=deepcopy(self.q);altered['question']='changed'
        ok,bad,_,_=self.check([self.q,altered],[],('q1',))
        self.assertFalse(ok);self.assertEqual(bad[0]['errors'],['conflicting_or_duplicate_slot'])
    def test_truncated_json_never_recovered_into_success(self):
        ok,bad,_,error=generate.validate_draft('{"root_id":"r","questions":[', ['q1'],'r','multi_select',self.source,self.path)
        self.assertFalse(ok);self.assertTrue(bad);self.assertTrue(error)
    def test_dedup_prompt_drops_provenance_scores_collapses_same_fact_set(self):
        row={'candidate_id':'old','question_type':'multi_select','question':'Q','facts':['A','B'],
             'hard_dedup':False,'score':1,'source_path':'private/long','source_sha256':'abc'}
        selected={**row,'candidate_id':'selected','facts':['B','A'],'hard_dedup':True}
        out=generate.prompt_prior([row,selected]);self.assertEqual(len(out),1)
        self.assertEqual(out[0]['dedup_action'],'avoid_already_selected')
        self.assertNotIn('score',out[0]);self.assertNotIn('source_path',out[0])
    def test_repair_does_not_echo_whole_response_or_derived_metadata(self):
        raw=deepcopy(self.q);wrapped={**raw,'raw_model_candidate':raw,'source_path':'x','large_provenance':['x']*100}
        out=generate.repair_failures([{'candidate_id':'q1','errors':['e'],'draft':wrapped}])
        self.assertEqual(out[0]['draft'],raw)

if __name__=='__main__': unittest.main()
