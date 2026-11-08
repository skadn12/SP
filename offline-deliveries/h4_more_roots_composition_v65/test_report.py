"""Offline report checks for count coverage, native baseline and immutable merging."""
from copy import deepcopy
from fractions import Fraction
import unittest
import finalize_report as report


def question(qid,count,root='r'):
    return {'candidate_id':qid,'root_id':root,'question_type':'multi_select','question':'Which statements?',
        'options':{label:'Statement '+label for label in 'ABCDEF'},'gold':list('ABCDEF'[:count])}


class ReportTests(unittest.TestCase):
    def test_native_blind_all_six_scores_zero_for_two_to_five(self):
        qs=[question(str(n),n) for n in range(2,7)]
        result=report.blind_all_six(qs)
        self.assertEqual([r['partial'] for r in result['questions']],['0','0','0','0','1'])
        self.assertEqual(result['aggregate']['partial_mean'],'1/5')
        self.assertEqual(result['aggregate']['exact_correct'],1)

    def test_count_coverage_does_not_claim_plan_as_pool_success(self):
        planned=[question(str(n),n) for n in range(2,7)]
        retained=[question('two',2),question('five',5)]
        self.assertEqual(report.distribution(planned)['missing_multiselect_gold_counts'],[])
        self.assertEqual(report.distribution(retained)['missing_multiselect_gold_counts'],[3,4,6])

    def test_merge_only_exact_same_root_set_and_preserves_gold(self):
        old=question('old',2)
        duplicate=deepcopy(old);duplicate['candidate_id']='duplicate'
        new=question('new',3,'different-root')
        before=deepcopy([old,duplicate,new])
        merged,duplicates=report.merge_exact([old],[duplicate,new])
        self.assertEqual(merged,[old,new]);self.assertEqual(len(duplicates),1)
        self.assertEqual([old,duplicate,new],before)
        duplicate['gold']=list('ABC')
        self.assertTrue(report.merge_exact([old],[duplicate])[1][0]['gold_content_disagreement'])

    def test_unknown_provider_usage_is_not_zero(self):
        result=report.cost_summary([{'remote_http_attempt':True,'token_usage_status':'unknown',
            'reported_input_tokens':None,'reported_output_tokens':None}],1)
        self.assertEqual(result['input_tokens'],'unknown')
        self.assertEqual(result['unknown_usage_http_attempts'],1)
        self.assertFalse(result['usage_complete'])

    def test_empty_statistics_have_explicit_unknown_rates(self):
        self.assertIsNone(report.aggregate([])['exact_rate'])
        self.assertIsNone(report.compare_pairs([])['trained_minus_base_exact_percentage_points'])
        self.assertEqual(report.distribution([])['missing_multiselect_gold_counts'],[2,3,4,5,6])


if __name__=='__main__':unittest.main()
