import copy
import unittest
from .run import eligible
from .compile_feedback import measurement_id
from .admission import difficulty_admission

class EligibilityTests(unittest.TestCase):
    def test_bad_gold_low_scores_and_base_only_errors_are_not_progress(self):
        review=dict.fromkeys(('semantic_supported','novel_answer_target',
                              'independent_fact_combination','intent_relation_satisfied'), True)
        easy={'full_context_correct_count':12,'full_context_total_count':12,
              'no_context_correct_count':12,'no_context_total_count':12}
        hard={**easy,'full_context_correct_count':6,'no_context_correct_count':0}
        scores={'M0':hard,'M_clean_policy':easy,'M_inclusive_policy':easy}
        self.assertFalse(difficulty_admission(review,scores)['hard_pool_eligible'])
        scores['M_clean_policy']=hard
        self.assertTrue(difficulty_admission(review,scores)['hard_pool_eligible'])
        review['semantic_supported']=False
        self.assertFalse(difficulty_admission(review,scores)['hard_pool_eligible'])

    def test_measurement_join_preserves_same_source_id_in_distinct_cohorts(self):
        old={'candidate_id':'existing-id'}
        first={'root_id':'r','source_candidate_id':'same','evaluation_scope':'baseline_omitted'}
        second={**first,'evaluation_scope':'balanced_non_single'}
        self.assertEqual(measurement_id(old),'existing-id')
        self.assertNotEqual(measurement_id(first),measurement_id(second))
    def test_group_mismatch_does_not_replace_semantic_evidence_check(self):
        row={'status':'completed','semantic_status':'accept','answer_semantically_valid':True,
             'audit':{'evidence_sufficient':'sufficient'},
             'static_review':{'training_blockers':['evidence_group_flat_mismatch']}}
        self.assertTrue(eligible(row))
        unsupported=copy.deepcopy(row); unsupported['audit']['evidence_sufficient']='insufficient'
        self.assertFalse(eligible(unsupported))
        bad=copy.deepcopy(row); bad['static_review']['training_blockers'].append('alias_residue_in_evidence')
        self.assertFalse(eligible(bad))
        uncertain=copy.deepcopy(row); uncertain['semantic_status']='uncertain'
        self.assertFalse(eligible(uncertain))

if __name__=='__main__': unittest.main()
