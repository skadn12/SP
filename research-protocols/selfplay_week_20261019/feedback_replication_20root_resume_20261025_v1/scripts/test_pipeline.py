import unittest
from analysis_resume import root_comparisons,bootstrap,summary
from solver_resume import validate_initialization_recovery
class MissingnessTests(unittest.TestCase):
 def r(self,a,v,correct=True,valid=True):return {'root_id':'root','slot_id':'MC2','arm':a,'variant_id':v,'returned':True,'strict':{'exact':correct,'valid':valid}}
 def test_nonreturn_excludes_slot_instead_of_scoring_zero(self):
  r=[self.r('F',0),self.r('F',1),self.r('G',0)]
  result=root_comparisons(r,['root'],['MC2'])[0]
  self.assertIsNone(result['difference']);self.assertEqual(result['eligible_common_slots'],0)
 def test_invalid_return_still_in_paired_cohort(self):
  r=[self.r('F',0,False,False),self.r('F',1,False,False),self.r('G',0),self.r('G',1)]
  result=root_comparisons(r,['root'],['MC2'])[0]
  self.assertEqual(result['difference'],-1);self.assertEqual(result['eligible_common_slots'],1)
 def test_bootstrap_missing_roots_retained_as_missing(self):
  b=bootstrap([{'difference':None}]);self.assertEqual(b['eligible_roots'],0);self.assertIsNone(b['ci95'])
 def test_combined_answer_only_requires_entire_returned_cohort(self):
  s={'valid':True,'exact':True,'question_type':'multi_select','prediction':['A','B'],'gold':['A','B'],'false_positive':0,'false_negative':0}
  rows=[{'arm':'F','slot_id':'MC2','strict':s,'answer_only':{'exact':True}}, {'arm':'F','slot_id':'MC2','strict':s,'answer_only':None}]
  result=next(r for r in summary(rows) if r['arm']=='F' and r['group']=='PRIMARY')
  self.assertEqual(result['answer_only_presentations'],1);self.assertIsNone(result['answer_only_exact_accuracy']);self.assertEqual(result['strict_exact_accuracy'],1.)
 def test_initialization_recovery_never_retries_submitted_requests(self):
  with self.assertRaises(RuntimeError):validate_initialization_recovery({'returned_count':0},1,0)
  with self.assertRaises(RuntimeError):validate_initialization_recovery({'returned_count':0},0,1)
  with self.assertRaises(RuntimeError):validate_initialization_recovery({'returned_count':1},0,0)
if __name__=='__main__':unittest.main()
