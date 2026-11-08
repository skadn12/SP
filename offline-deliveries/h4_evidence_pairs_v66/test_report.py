"""Offline report checks for count coverage, native baseline and immutable merging."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import tempfile
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

    def test_null_field_statistics_preserve_unknown_and_missing_reasons(self):
        result=report.fact_field_statistics([
            {'supported':{},'contradicted':{},'contradiction_unavailable_reason':None},
            {'supported':{},'contradicted':None,'contradiction_unavailable_reason':'Unknown whether the plan was completed.'},
            {'contradicted':None},
            {'contradicted':None,'contradiction_unavailable_reason':' '},
            {},{'contradicted':'bad'},'invalid fact'])
        self.assertEqual(result['fact_entries'],7)
        self.assertEqual(result['contradicted_versions'],1)
        self.assertEqual(result['contradicted_explicit_null'],3)
        self.assertEqual(result['contradicted_field_missing'],1)
        self.assertEqual(result['contradicted_invalid_nonnull'],1)
        self.assertEqual(result['null_reason_nonempty'],1)
        self.assertEqual(result['null_reason_field_missing'],1)
        self.assertEqual(result['null_reason_empty_null_or_invalid'],1)
        self.assertEqual(result['null_reason_verbatim_histogram'],{'Unknown whether the plan was completed.':1})
        self.assertFalse(result['null_reasons_semantically_verified'])
        self.assertTrue(result['unknown_or_missing_reason_is_not_counterevidence'])

    def test_raw_retries_and_accepted_usable_versions_have_separate_counts(self):
        with tempfile.TemporaryDirectory() as td:
            home=Path(td);slot=home/'assets/s'
            facts=[{'supported':{},'contradicted':{},'contradiction_unavailable_reason':None},
                   {'supported':{},'contradicted':None,'contradiction_unavailable_reason':'No explicit alternative is stated.'}]
            for attempt in ['01','02']:
                report.ev.write(slot/'requests'/attempt/'RESPONSE.json',{'text':report.json.dumps({'facts':facts})})
            report.ev.write(slot/'requests/03/RESPONSE.json',{'text':'not json'})
            report.ev.write(slot/'ASSETS.json',{'facts':facts})
            report.ev.write(slot/'ASSET_REVIEW.json',{'versions':[{'usable':True,'relation':'supports'},
                {'usable':False,'relation':'unknown'},{'usable':True,'relation':'supports'}]})
            stats=report.evidence_statistics(home,{'slots':[{'slot_id':'s','root_id':'r'}]})
        self.assertEqual(stats['raw_response_count'],3)
        self.assertEqual(stats['raw_response_parse_statuses'],{'FACT_LIST_PRESENT':2,'UNPARSEABLE_JSON':1})
        self.assertEqual(stats['raw_attempt_fields']['contradicted_versions'],2)
        self.assertEqual(stats['accepted_asset_fields']['contradicted_versions'],1)
        self.assertEqual(stats['usable_contradicted_versions'],0)
        self.assertEqual(stats['reviewed_unknown_versions'],1)
        self.assertEqual(stats['accepted_asset_fields']['null_reason_nonempty'],1)

    def test_v65_comparison_records_changed_priors_without_causal_claim(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);old=base/'old';new=base/'new'
            slot={'slot_id':'s','root_id':'r','history_sha256':'source','focus':{'scene_refs':['R01:S01']},'initial_target_count':2}
            report.ev.write(old/'prior.json',{'questions':[{'candidate_id':'older'}]})
            report.ev.write(new/'prior.json',{'questions':[{'candidate_id':'older'},{'candidate_id':'v65-added'}]})
            old_config={'slots':[{**slot,'prior_pool_paths':[str(old/'prior.json')]}]}
            config={'slots':[{**slot,'prior_pool_paths':[str(new/'prior.json')]}],'baseline_pool':str(old/'MERGED_USABLE_CANDIDATES.json')}
            report.ev.write(old/'CONFIG.json',old_config)
            previous={key:0 for key in ['assembled_count','content_usable_count','known_issue_count','selected_count','calls_per_model']}
            previous.update(merged_count=65,generation_cost={'http_attempts':0},distributions={},
                base_trained_comparison={'primary':report.compare_pairs([])})
            report.ev.write(old/'RESULT_SUMMARY.json',previous)
            current={**previous,'evidence_statistics':report.evidence_statistics(new,config)}
            result=report.descriptive_previous_comparison(new,config,current)
        self.assertTrue(result['same_sources_and_focus'])
        self.assertEqual(result['historical_prior_changes'][0]['added_candidate_ids'],['v65-added'])
        self.assertFalse(result['strict_randomized_control'])
        self.assertFalse(result['causal_prompt_effect_claimed'])
        self.assertIsNone(result['v66']['base_trained_primary']['base']['exact_rate'])


if __name__=='__main__':unittest.main()
