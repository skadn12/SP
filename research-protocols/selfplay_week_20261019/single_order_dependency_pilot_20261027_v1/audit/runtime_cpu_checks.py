"""Focused synthetic checks only; never prepare, GPU-run or load real candidates."""
import copy, importlib.util, json, sys, tempfile
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('single_order_cpu_runner',BASE/'runner.py')
r=importlib.util.module_from_spec(spec);sys.modules[spec.name]=r;spec.loader.exec_module(r)
checks=[]
def ok(name, condition):
    assert condition, name
    checks.append({'name':name,'status':'PASS'})

def question(kind='ordering'):
    keys=[f'item_{i:02}' for i in range(1,5)]
    return {'candidate_id':'synthetic-'+kind,'root_id':'synthetic','question_type':kind,'question':'Synthetic task',
            'options':{x:'Synthetic text '+x for x in keys},'gold':keys if kind=='ordering' else keys[:1]}

def score(c, pred, refs=None, reason='brief', **raw_changes):
    view,p=r.quality.prepare_displays(c,[11,12,13])[0]
    labels=list(view['options'])
    reverse={v:k for k,v in p['display_to_canonical'].items()}
    ans=' '.join(reverse[x] if x in reverse else x for x in pred)
    row={'request_id':'syn','candidate_id':c['candidate_id'],'root_id':c['root_id'],'question_type':c['question_type'],
         'cohort':'new','condition':'FULL','variant_id':'perm_00','display':view['options'],'labels':labels}
    cert={'target_pair':c['gold'][:2]} if c['question_type']=='ordering' else {'target_option':c['gold'][0]}
    info={'presentation':p,'visible_refs':['R01:S01:T001'],'dependency_certificate':cert}
    out={'final_answer':'<box>'+ans+'</box>','reason':reason,'turn_refs':refs if refs is not None else ['R01:S01:T001']}
    raw={'request_id':'syn','status':'RETURNED','finish_reason':'stop','capture_errors':[],
         'raw_output':json.dumps(out),**raw_changes}
    return r.score_one(row,info,raw,r.output_schema())

c=question()
a=score(c,c['gold'])
ok('complete correct order exact/pairs/LCS',a['exact'] and a['pairwise_accuracy']==a['normalized_lcs']==1 and a['concordant_pairs']==6)
a=score(c,list(reversed(c['gold'])))
ok('reverse order zero pairs and LCS=1/n',not a['exact'] and a['pairwise_accuracy']==0 and a['normalized_lcs']==.25)
p=c['gold'][:];p[0],p[1]=p[1],p[0]
a=score(c,p)
ok('one inverted pair is 5/6 and LCS 3/4',a['pairwise_accuracy']==5/6 and a['normalized_lcs']==.75)
for name,pred in [('empty',[]),('missing',c['gold'][:-1]),('duplicate',c['gold'][:-1]+[c['gold'][0]]),('unknown',c['gold'][:-1]+['Z'])]:
    a=score(c,pred)
    ok(name+' invalid exact=false and partial unscored',not a['answer_valid'] and not a['exact'] and a['pairwise_accuracy'] is None)
for refs in [['hidden'],['R01:S01:T001']*2,123]:
    a=score(c,c['gold'],refs=refs)
    ok('bad refs preserve correct answer '+str(refs),a['exact'] and not a['reference_compliant'] and a['prediction']==c['gold'])
a=score(c,c['gold'],reason=0)
ok('reason/schema failure preserves final answer',a['exact'] and not a['schema_valid'])
s=question('single_choice')
a=score(s,s['gold'])
ok('single-choice correct with no J/ordering partial',a['exact'] and a['pairwise_accuracy'] is None and 'J' not in a)
a=score(s,[next(x for x in s['options'] if x not in s['gold'])])
ok('single-choice wrong but valid',a['answer_valid'] and not a['exact'])
a=score(s,[])
ok('single-choice abstention invalid',a['abstention'] and not a['answer_valid'])
a=score(s,list(s['options'])[:2])
ok('single-choice multi-label invalid',not a['answer_valid'])
views=r.quality.prepare_displays(c,[11,12,13]);sw=copy.deepcopy(c);sw['gold']=list(reversed(c['gold']))
ok('Gold swap preserves all public display mappings',[v[0] for v in views]==[v[0] for v in r.quality.prepare_displays(sw,[11,12,13])])
ok('three permutations distinct',len({tuple(v[1]['display_to_canonical'].values()) for v in views})==3)
ok('one seed independent of permutation and condition',len({r.sampling_seed(c['candidate_id']) for _ in range(27)})==1)
ok('review list/wrapper both supported',r.review_records([{}])==r.review_records({'reviews':[{}]}))
with tempfile.TemporaryDirectory() as temp:
    src=Path(temp)/'source.json'
    turns=[{'turn_ref':ref,'speaker':'Person','text':txt} for ref,txt in
           [('R01:S01:T001','premise one'),('R01:S01:T002','premise one repeated'),('R03:S02:T001','premise two')]]
    src.write_text(json.dumps({'root_id':'synthetic','turns':turns}))
    c['author_id']='author';c['source_binding']=r.bind(src)
    c['item_analysis']={key:{'should_select':True,'rationale':'synthetic rationale','evidence':[{'turn_ref':turns[0]['turn_ref'],'speaker':'Person','quote':'premise one'}]} for key in c['options']}
    c['dependency_certificate']={'target_pair':c['gold'][:2],'single_group_insufficient':True,'joint_inference':'synthetic joint inference',
        'redundant_turn_refs':['R01:S01:T002'],'premise_groups':[
            {'group_id':'g1','turn_refs':['R01:S01:T001','R01:S01:T002'],'premise':'one','why_necessary':'one needed','removal_effect':'unknown'},
            {'group_id':'g2','turn_refs':['R03:S02:T001'],'premise':'two','why_necessary':'two needed','removal_effect':'unknown'}]}
    review={'candidate_id':c['candidate_id'],'candidate_sha256':r.quality.fingerprint(c),'reviewer_id':'reviewer','gold_blind':True,
            'independent_gold':c['gold'],'decision':'PASS','checks':{k:{'decision':'PASS','rationale':'synthetic independent check'} for k in
             list(r.quality.REVIEW_CHECKS)+['ordering_total_order','item_style']},
             'ablation_review':[{'group_id':g['group_id'],'removed_turn_refs':g['turn_refs'],'target_underdetermined_after_removal':True,'rationale':'synthetic underdetermination'} for g in c['dependency_certificate']['premise_groups']]}
    ok('source-bound complete-order candidate and independent review admitted',r.quality.validate_review(c,review)['admitted'])
    rr=copy.deepcopy(review);rr['independent_gold']=list(reversed(c['gold']))
    ok('same-set wrong-order independent Gold rejected',not r.quality.validate_review(c,rr)['admitted'])
    cc=copy.deepcopy(c);cc['dependency_certificate']['premise_groups'][0]['turn_refs']=['R01:S01:T001']
    ok('redundant premise outside complete deletion rejected',not r.quality.validate_candidate(cc)['mechanical_pass'])
    cc=copy.deepcopy(c);cc['dependency_certificate']['target_pair']=list(reversed(cc['dependency_certificate']['target_pair']))
    ok('reversed certified target pair rejected',not r.quality.validate_candidate(cc)['mechanical_pass'])
    conds=dict(r.conditions(c,{'turns':turns}))
    ok('ablate complete repeated group and ONLY preserves source order',[t['turn_ref'] for t in conds['ABLATE_G1']]==['R03:S02:T001'] and
       [t['turn_ref'] for t in conds['ONLY_G1']]==['R01:S01:T001','R01:S01:T002'])
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(str(r.orig.TOKENIZER),local_files_only=True)
compile_receipt=r.cpu_schema_preflight(tok,r.output_schema())
ok('actual shared single-choice/ordering schema validates/compiles CPU xgrammar and guidance',compile_receipt['status']=='PASS_CPU_BOTH_BACKENDS_VALIDATED_AND_COMPILED')
result={'status':'PASS','checks':checks,'count':len(checks),'schema_compilation':compile_receipt,
        'prepare_called':False,'gpu_model_started':False,'real_candidate_content_read':False,
        'strict_raw':'Complete unchanged prior.check_raw delegated, plus exact returned batch_index check; legacy coverage retained.'}
(BASE/'audit/runtime_cpu_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'PASS','checks':len(checks),'schema_compilation':compile_receipt['status']}))
