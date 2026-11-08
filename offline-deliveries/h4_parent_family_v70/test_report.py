"""Family reporting checks preserve baseline admission and count dependent siblings correctly."""
from copy import deepcopy
from fractions import Fraction
import unittest
import finalize_report as r


def question(cid,k,root='root1',suffix=''):
    return {'candidate_id':cid,'root_id':root,'question_type':'multi_select','question':'Which?',
        'options':{label:label+suffix for label in 'ABCDEF'},'gold':list('ABCDEF'[:k])}

def member(q,role='child',family='f1'):
    return {'candidate_id':q['candidate_id'],'family_id':family,'root_id':q['root_id'],'train_root_id':q['root_id'],'split_group':q['root_id'],
        'parent_candidate_id':family+'-control','gold_count':len(q['gold']),'role':role,'new_candidate':role=='child','fixed_fact_ids':list('ABCDEF')}


class ReportTests(unittest.TestCase):
    def test_selected_control_never_added_even_when_outside_baseline(self):
        old=question('old',2);control=question('f1-control',6,suffix='control');child=question('child',3,suffix='child')
        inputs=deepcopy([old,control,child]);members={q['candidate_id']:member(q,'control6' if q is control else 'child') for q in (control,child)}
        merged,added,dupes,controls=r.merge_selected_children([old],[control,child],members)
        self.assertEqual(merged,[old,child]);self.assertEqual(added,[child]);self.assertFalse(controls[0]['already_in_baseline']);self.assertEqual(dupes,[])
        self.assertEqual([old,control,child],inputs)

    def test_failed_parent_reevaluation_does_not_delete_old_parent(self):
        control=question('f1-control',6);child=question('child',2,suffix='child');members={q['candidate_id']:member(q,'control6' if q is control else 'child') for q in (control,child)}
        merged,added,_,_=r.merge_selected_children([control],[child],members)
        self.assertEqual(merged,[control,child]);self.assertEqual(added,[child])
        cfg={'target_distribution':{str(k):2 for k in (2,3,4,5)}}
        stages=r.family_stage_report([control,child],[child],[child],[child],added,[control],members,cfg)
        self.assertFalse(stages['families'][0]['parent_control']['content_usable'])
        self.assertTrue(stages['families'][0]['parent_control']['retained_in_baseline'])

    def test_duplicate_baseline_children_not_readded_or_reordered(self):
        old=question('old',2);duplicate=deepcopy(old);duplicate['candidate_id']='child'
        members={'child':member(duplicate)}
        merged,added,dupes,_=r.merge_selected_children([old],[duplicate],members)
        self.assertEqual(merged,[old]);self.assertEqual(added,[]);self.assertEqual(len(dupes),1)
        duplicate['gold']=list('ABC')
        with self.assertRaisesRegex(ValueError,'gold_content_disagreement'):r.merge_selected_children([old],[duplicate],members)

    def test_family_stages_separate_plan_review_evaluation_and_admission(self):
        controls=[question(f'f{i}-control',6,'root'+str(i),'control') for i in (1,2)]
        children=[question(f'f{i}-child{k}',k,'root'+str(i),f'child{k}') for i in (1,2) for k in (2,3,4,5)]
        allq=controls+children
        members={q['candidate_id']:member(q,'control6' if len(q['gold'])==6 else 'child',q['candidate_id'].split('-')[0]) for q in allq}
        cfg={'target_distribution':{str(k):2 for k in (2,3,4,5)}}
        selected=[q for q in children if len(q['gold'])==4]
        result=r.family_stage_report(allq,allq,allq,selected,selected,controls,members,cfg)
        self.assertTrue(result['required_count_coverage']['assembled']['all_target_counts_met'])
        self.assertEqual(result['required_count_coverage']['selected']['missing_counts'],[2,3,5])
        self.assertEqual(result['children_stages']['assembled']['questions'],8)
        self.assertEqual(result['control_stages']['assembled']['questions'],2)
        self.assertEqual(result['independent_family_count'],2);self.assertFalse(result['siblings_are_independent_samples'])

    def test_sidecar_rejects_cross_root_sibling(self):
        control=question('f1-control',6);child=question('child',2,suffix='child');qs=[control,child]
        side={'members':[member(control,'control6'),member(child)]};r.sidecar_index(side,qs)
        side['members'][1]['split_group']='other'
        with self.assertRaisesRegex(ValueError,'family_root_or_count_changed'):r.sidecar_index(side,qs)

    def test_unknown_cost_not_zero_with_partial_usage(self):
        result=r.cost_summary([{'remote_http_attempt':True,'token_usage_status':'known','reported_input_tokens':10,'reported_output_tokens':20},
            {'remote_http_attempt':True,'token_usage_status':'unknown','reported_input_tokens':None,'reported_output_tokens':None}],1)
        self.assertEqual(result['input_tokens'],'unknown');self.assertEqual(result['known_input_token_subtotal'],10)

    def test_family_equal_weight_avoids_more_sibling_weight(self):
        qs=[question('f1-a',2),question('f1-b',3,suffix='b'),question('f2-a',2,'root2')]
        members={q['candidate_id']:member(q,family=q['candidate_id'].split('-')[0]) for q in qs}
        rows=[{'candidate_id':q['candidate_id'],'request_id':str(i),'exact':i==2,'partial':'1' if i==2 else '0'} for i,q in enumerate(qs)]
        v={'models':{side:{'responses':rows,'questions':[{'candidate_id':q['candidate_id'],'selected_by_policy':False} for q in qs]} for side in ('base','trained')},'paired_inputs':{}}
        compare,pairs=r.comparison(v,qs,members)
        self.assertEqual(compare['primary']['base']['partial_mean'],'1/3')
        self.assertEqual(compare['equal_family_weight_children']['base']['equal_family_weight_partial_mean'],'1/2')
        self.assertFalse(compare['siblings_are_independent_samples'])

    def test_native_blind_all_six_zero_children_one_controls(self):
        scores=r.blind_all_six([question(str(k),k) for k in (2,3,4,5,6)])
        self.assertEqual([s['partial'] for s in scores['questions']],['0','0','0','0','1'])

if __name__=='__main__':unittest.main()
