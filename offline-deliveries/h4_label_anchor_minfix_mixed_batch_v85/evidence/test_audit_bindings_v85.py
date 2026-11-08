"""Offline pure-AST fixtures; synthetic bindings make no semantic claims."""
import ast
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import unittest
HOME=Path(__file__).resolve().parents[1]
ROOTS=tuple('cc-train-episode-'+r for r in ('103866','109164','112812','135466'))

def functions(name):
    tree=ast.parse((HOME/'evidence'/name).read_text())
    wanted={'registry_contract_errors','validate_admission_contract'}
    body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
    namespace={'ROOTS':ROOTS,'datetime':datetime}
    exec(compile(ast.Module(body=body,type_ignores=[]),'pure_binding_fixture','exec'),namespace)
    return namespace


def fixture():
    home=Path('/tmp/synthetic_v85_audit_binding_fixture'); data={};slots=[];order=[];review_rows=[]
    def read(path):return deepcopy(data[str(path)])
    def bind(path):return {'path':str(path),'sha256':hashlib.sha256(json.dumps(data[str(path)],sort_keys=True).encode()).hexdigest()}
    registry={'source_bindings':[],'characters':[], 'policy':{'source_unchanged':True,'registry_is_planning_aid':True,
      'registry_is_extra_QA_evidence':False,'solver_context_unchanged':True}}
    sources={}
    for i,root in enumerate(ROOTS):
        suffix=root.split('-')[-1];sname='SyntheticActor'+str(i);sp=home/('source'+suffix+'.json')
        source={'root_id':root,'turns':[{'turn_ref':'P01:T001','speaker':sname,'text':'Synthetic fixture excerpt.'}]}
        data[str(sp)]=source;source_binding=bind(sp)
        slot={'root_id':root,'source_path':str(sp),'source_sha256':source_binding['sha256'],
          'history_path':str(sp),'history_sha256':source_binding['sha256'],'parent_candidate_id':root+'-v85-parent'}
        order.append({**slot,'candidate_count':2})
        directory=home/'parent_review';public_path=directory/('root'+suffix+'_PUBLIC.json');parent_path=directory/('root'+suffix+'_PARENT.json')
        judgment_path=directory/('root'+suffix+'_SOURCE_JUDGMENTS.json');provenance_path=directory/('root'+suffix+'_PROVENANCE.json')
        public={'root_id':root,'candidate_id':slot['parent_candidate_id'],'source_path':str(sp),'source_sha256':source_binding['sha256'],
          'options':{label:'Synthetic option '+label for label in 'ABCDEF'}}
        parent={**public,'gold':list('ABCDEF')};data[str(public_path)]=public;data[str(parent_path)]=parent
        judgment={'candidate_id':public['candidate_id'],'root_id':root,'source_path':str(sp),'source_sha256':source_binding['sha256'],
          'public_question':public,'public_question_path':str(public_path),'public_question_sha256':bind(public_path)['sha256'],
          'full_source_read':True,'full_source_turn_count':1,'content_status':'CORE_USABLE','known_old_parent_gold':True,
          'current_parent_private_gold_read_before_judgments_saved':False,'current_parent_private_evidence_read_before_judgments_saved':False,
          'created_at_utc':'2026-10-14T07:00:00+00:00','judgments':[{'option':label,'proposition':public['options'][label],
             'source_judgment':'supported','entailed':True,'reason':'Synthetic fixture; no semantic certification.',
             'required_evidence':[{'turn_ref':'P01:T001','speaker':sname,'quote':'Synthetic fixture excerpt.'}]} for label in 'ABCDEF']}
        data[str(judgment_path)]=judgment
        provenance={'source':source_binding,'public_question':bind(public_path),'parent':bind(parent_path),
          'parent_review_path':str(judgment_path),'parent_review_sha256':bind(judgment_path)['sha256'],
          'source_judgments_saved_at_utc':judgment['created_at_utc'],'current_private_parent_construction_at_utc':'2026-10-14T08:00:00+00:00',
          'historical_parent_gold_known':True,'fresh_blind_parent_review':False,'full_source_reread_this_batch':True,
          'source_bytes_unchanged':True,'current_solver_scores_read':False,'training_allowed':False}
        data[str(provenance_path)]=provenance
        slot.update(parent_review_path=str(judgment_path),parent_review_sha256=bind(judgment_path)['sha256'],
          parent_provenance_path=str(provenance_path),parent_provenance_sha256=bind(provenance_path)['sha256']);slots.append(slot)
        source_review_path=home/('source_review'+suffix+'.json')
        data[str(source_review_path)]={'decision':'READY','source':source_binding,'full_source_read':True,'turn_refs_read':['P01:T001']}
        review_rows.append({'root_id':root,'decision':'READY','source':source_binding,'review':bind(source_review_path),'blocking_findings':[]})
        registry['source_bindings'].append({'root_id':root,**source_binding,'full_turn_count':1,'all_turns_reviewed':True,'speaker_names':[sname]})
        registry['characters'].append({'root_id':root,'canonical_name':sname,
          'gender':{'value':None,'status':'unknown_no_explicit_source_evidence','evidence':[]},
          'third_person_pronouns':{'value':None,'status':'unknown_no_explicit_source_evidence','evidence':[]},
          'permitted_question_reference':{'kind':'canonical_name','value':sname,'allow_possessive_name':True,'inferred_gendered_pronouns_allowed':False}})
        sources[root]={'path':str(sp),'sha256':source_binding['sha256'],'source':source}
    summary={'roots':review_rows,'ready_roots':list(ROOTS),'held_roots':[],
      'historical_held_roots_unchanged':['cc-train-episode-231899','cc-train-episode-138343']}
    summary_path=home/'SOURCE_ADMISSION_SUMMARY.json';data[str(summary_path)]=summary
    admission_path=home/'SOURCE_ADMISSION.COMPLETE.json'
    data[str(admission_path)]={'status':'FROZEN','training_allowed':False,'ready_roots':list(ROOTS),'held_roots':[],
      'files':[bind(summary_path)]+[r['review'] for r in review_rows]+[r['source'] for r in review_rows]}
    data[str(home/'BATCH_PREPARATION.STARTED.json')]={'started_at_utc':'2026-10-14T09:00:00+00:00'}
    registry_path=home/'CHARACTER_REGISTRY.json';data[str(registry_path)]=registry
    policy_path=home/'ACTOR_REFERENCE_POLICY.json';data[str(policy_path)]={'registry':bind(registry_path),
      'output_style':'use_source_names_or_explicit_roles_for_people','forbidden_gendered_pronouns':['he','him','his','himself','she','her','hers','herself'],
      'new_source_facts':False,'automatic_repair':False}
    config={'multiselect_config':{'slots':slots},'ordering_slots':order,'character_registry':bind(registry_path),
      'actor_reference_policy':bind(policy_path),'source_consistency':{'upstream_complete':bind(admission_path),
      'selected_roots':list(ROOTS),'source_review_is_question_answer_certification':False,
      'excluded_held_roots':summary['historical_held_roots_unchanged']}}
    return home,data,read,bind,config,registry,sources


class BindingTests(unittest.TestCase):
    def run_contract(self,name,modify=None):
        home,data,read,bind,config,registry,sources=fixture()
        if modify:modify(home,data,config,registry)
        checks=[]
        result=functions(name)['validate_admission_contract'](config,home,read,bind,lambda ok,label,*detail:checks.append((ok,label)))
        return checks,result

    def test_both_auditors_use_identical_pure_admission_functions(self):
        texts=[]
        for name in ('audit_generation_v85.py','audit_results_v85.py'):
            tree=ast.parse((HOME/'evidence'/name).read_text())
            texts.append([ast.dump(n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'registry_contract_errors','validate_admission_contract'}])
        self.assertEqual(*texts)

    def test_four_root_fresh_binding_accepts_both_auditors(self):
        for name in ('audit_generation_v85.py','audit_results_v85.py'):
            checks,result=self.run_contract(name)
            self.assertTrue(all(ok for ok,_ in checks),[label for ok,label in checks if not ok])
            self.assertEqual(result['fresh_current_parent_review_count'],4)
            self.assertFalse(result['independent_test_claimed'])

    def test_registry_rejects_sha_name_unknown_and_extra_root_tampering(self):
        _,_,_,_,config,registry,sources=fixture();fn=functions('audit_generation_v85.py')['registry_contract_errors']
        self.assertEqual(fn(registry,config['multiselect_config']['slots'],sources),[])
        changes=[lambda r:r['source_bindings'][0].update(sha256='0'*64),
          lambda r:r['characters'][0].update(canonical_name='Invented'),
          lambda r:r['characters'][0]['gender'].update(value='invented'),
          lambda r:r['characters'][0]['third_person_pronouns'].update(status='known'),
          lambda r:r['characters'][0]['gender'].update(evidence=[{'invented':True}]),
          lambda r:r['characters'].append(deepcopy(r['characters'][0])),
          lambda r:r['characters'][0]['permitted_question_reference'].update(inferred_gendered_pronouns_allowed=True)]
        for change in changes:
            altered=deepcopy(registry);change(altered)
            self.assertTrue(fn(altered,config['multiselect_config']['slots'],sources))

    def test_current_source_hold_is_not_ready(self):
        def change(home,data,config,registry):data[str(home/'source_review103866.json')]['decision']='HOLD'
        checks,_=self.run_contract('audit_results_v85.py',change)
        self.assertTrue(any(not ok and label=='fresh_complete_source_review_binding:cc-train-episode-103866' for ok,label in checks))

    def test_non_supported_parent_cannot_become_six_gold(self):
        def change(home,data,config,registry):data[config['multiselect_config']['slots'][0]['parent_review_path']]['judgments'][0]['source_judgment']='unknown'
        checks,_=self.run_contract('audit_results_v85.py',change)
        self.assertTrue(any(not ok and label=='fresh_parent_supported_option:cc-train-episode-103866:A' for ok,label in checks))

    def test_private_before_judgments_or_after_preparation_rejected(self):
        for stamp in ['2026-10-14T06:00:00+00:00','2026-10-14T10:00:00+00:00']:
            def change(home,data,config,registry):data[config['multiselect_config']['slots'][0]['parent_provenance_path']]['current_private_parent_construction_at_utc']=stamp
            checks,_=self.run_contract('audit_results_v85.py',change)
            self.assertTrue(any(not ok and label=='fresh_parent_provenance_saved_before_current_private:cc-train-episode-103866' for ok,label in checks))

    def test_independent_result_ast_imports_no_native_score_or_models(self):
        tree=ast.parse((HOME/'evidence/audit_results_v85.py').read_text())
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        self.assertFalse(set(imports)&{'common','batch','source_admission','source_subset','mixed_evaluation','scoring','selection','protocol','vllm','torch'})

    def test_compact_prior_index_preserves_identity_without_private_answers(self):
        tree=ast.parse((HOME/'evidence/audit_generation_v85.py').read_text())
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='independent_ordering_prior_context']
        namespace={};exec(compile(ast.Module(body=nodes,type_ignores=[]),'pure_compact_prior_fixture','exec'),namespace)
        prior=[{'candidate_id':'synthetic-prior','root_id':ROOTS[0],'question_type':'ordering',
          'source_path':'synthetic-source.json','source_sha256':'0'*64,
          'items':{'I'+str(i):'Synthetic prior item '+str(i) for i in range(1,7)},
          'gold':['I1','I2','I3','I4','I5','I6'],'item_evidence':{'I1':'not_public'}}]
        context=namespace['independent_ordering_prior_context'](prior)
        self.assertEqual(context['question_index'],[{k:prior[0][k] for k in
            ('candidate_id','root_id','question_type','source_path','source_sha256')}])
        self.assertTrue({'gold','items','item_evidence'}.isdisjoint(context['question_index'][0]))
        self.assertIn('every distinct six-item set is listed there once',context['whole_set_text_location'])
        prior[0]['source_sha256']='1'*64
        self.assertNotEqual(context,namespace['independent_ordering_prior_context'](prior))

    def test_unknown_or_invalid_parent_cannot_enable_family_combinations(self):
        tree=ast.parse((HOME/'evidence/audit_generation_v85.py').read_text())
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='independent_parent_family_usable']
        namespace={};exec(compile(ast.Module(body=nodes,type_ignores=[]),'pure_parent_family_fixture','exec'),namespace)
        usable=namespace['independent_parent_family_usable']
        public={'parent_facts':[{'fact_id':'F'+str(i),'proposition':'Synthetic parent '+str(i)} for i in range(6)]}
        judgment={'stem_valid':True,'parents':[{**r,'relation':'supports','scope_valid':True,'fact_distinct':True}
            for r in public['parent_facts']]}
        self.assertTrue(usable(public,judgment))
        changes=[lambda j:j['parents'][0].update(relation='unknown'),
            lambda j:j['parents'][0].update(scope_valid=False),
            lambda j:j['parents'][0].update(fact_distinct=False),
            lambda j:j.update(stem_valid=False),lambda j:j['parents'].pop(),
            lambda j:j['parents'][0].update(proposition='changed'),
            lambda j:j['parents'][0].update(fact_id=j['parents'][1]['fact_id'])]
        for change in changes:
            altered=deepcopy(judgment);change(altered);self.assertFalse(usable(public,altered))


if __name__=='__main__':unittest.main(verbosity=2)
