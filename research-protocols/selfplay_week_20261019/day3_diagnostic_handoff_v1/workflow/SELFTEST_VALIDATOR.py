"""Synthetic integrity/negative-ledger tests. No Solver calls or real diagnostics."""
import importlib.util
import json
from pathlib import Path
import tempfile
spec=importlib.util.spec_from_file_location('handoff_validator',Path(__file__).with_name('validate_handoff.py'))
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
def run():
  with tempfile.TemporaryDirectory(prefix='handoff_validator_selftest_') as tmp:
    p=Path(tmp)
    def put(name,x):
      f=p/name;f.write_text(json.dumps(x)+'\n');return f
    def bind(f):return {'path':str(f),'sha256':v.sha(f)}
    source=put('source.json',{'turns':[{'turn_ref':'T1','text':'Red before blue.'}]})
    qs=put('questions.json',{'questions':[
      {'candidate_id':c,'root_id':'root','question_type':t,'source_path':str(source),'source_sha256':v.sha(source),field:{'A':'red','B':'blue'}}
      for c,t,field in [('Q1','multi_select','options'),('Q2','ordering','items')]]})
    freeze=put('freeze.json',{'status':'FROZEN'})
    base=put('base.json',{'calls':342,'unchanged':True,'bindings':[bind(freeze)]})
    plan=put('plan.json',{'max_solver_calls':2,'automatic_retries':False,'items':[
      {'candidate_id':c,'root_id':'root','fact_family_id':'family-'+c,'request_id':c+'-r','display_to_canonical':{'A':'A','B':'B'}} for c in ['Q1','Q2']]})
    raw1=put('raw1.json',{'raw_output':'wrong quote'});raw2=put('raw2.json',{'raw_output':'invalid'})
    responses=put('responses.json',{'responses':[
      {'request_id':'Q1-r','candidate_id':'Q1','status':'RETURNED','raw_output':'wrong quote','raw_response_binding':bind(raw1),'diagnostic_valid':True,'parsed':{'item_judgments':[
        {'label':'A','judgment':'supported','rationale':'red first','evidence':[{'turn':'T1','quote':'invented words'}]},
        {'label':'B','judgment':'not_supported','rationale':'no support','evidence':[],'evidence_status':'no_support_found'}],'final_answer':['A']}},
      {'request_id':'Q2-r','candidate_id':'Q2','status':'RETURNED','raw_output':'invalid','raw_response_binding':bind(raw2),'diagnostic_valid':False,'invalid_reason':'format','parsed':None}]})
    reviews=put('reviews.json',{'reviews':[
      {'candidate_id':'Q1','request_id':'Q1-r','review_status':'FAIL','quote_check':'FAIL','semantic_support_check':'FAIL','judgment_answer_consistency_check':'PASS','training_use':False,'limitations':[],'source_verified_error':True},
      {'candidate_id':'Q2','request_id':'Q2-r','review_status':'NOT_REVIEWABLE','quote_check':'NOT_APPLICABLE','semantic_support_check':'NOT_APPLICABLE','judgment_answer_consistency_check':'NOT_APPLICABLE','training_use':False,'limitations':['invalid']}]})
    fb=put('feedback.json',{'version':'v1','parent_version':None,'faithful_internal_trace_claim':False,'unverified_rationale_training_target':False,'entries':[
      {'feedback_id':'F1','version':'v1','status':'supported_negative','evidence_candidate_ids':['Q1'],'operational_claim':'invented quote observed','competing_explanations':['format error'],'falsification_condition':'exact quote used','source_verified_operational_feedback':True}],
      'question_dispositions':[
        {'candidate_id':'Q1','status':'negative_evidence','feedback_ids':['F1'],'reason':'audited false quotation','source_verified_operational_feedback':True},
        {'candidate_id':'Q2','status':'invalid_diagnostic','feedback_ids':[],'reason':'format invalid','excluded_from_usable_feedback':True}],
      'generation_handoff':{'feedback_loop_closed':False}})
    artifacts=dict(zip(v.REQUIRED,[qs,freeze,base,plan,responses,reviews,fb]))
    state=put('state.json',{'artifacts':{k:bind(f) for k,f in artifacts.items()},'expected_question_count':2,'expected_direct_calls':342,'prospective_all_question_diagnostic_rule':True,'question_admission_changed_by_diagnostic':False,'primary_evaluation':'direct'})
    good=v.validate(state)
    assert good['status']=='PASS' and good['gates']['local_feedback_ready_scope']==['Q1'] and not good['gates']['all_diagnostics_valid'] and not good['gates']['feedback_loop_closed'],good
    assert good['counts']['quote_mismatch_count']==1 and not good['failures'],good
    data=json.loads(reviews.read_text());data['reviews'][0]['quote_check']='PASS';reviews.write_text(json.dumps(data))
    s=json.loads(state.read_text());s['artifacts']['semantic_reviews']=bind(reviews);state.write_text(json.dumps(s))
    bad=v.validate(state)
    assert any(x['code']=='QUOTE_MISMATCH_MUST_BE_REVIEWED_FAIL' for x in bad['failures']),bad
    state.write_text(state.read_text()+' ')
    source.write_text('tamper')
    tampered=v.validate(state)
    assert any(x['code']=='BINDING_HASH_MISMATCH' for x in tampered['failures']),tampered
    return {'status':'PASS','synthetic_only':True,'solver_calls':0,'cases':['ordering-label coverage with invalid isolated','verified wrong quote remains negative usable feedback','false PASS on wrong quote rejected','source byte tampering rejected'], 'valid_case_checks':good['checks']}
if __name__=='__main__':
  result=run();out=Path(__file__).with_name('SELFTEST_RESULT.json');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
