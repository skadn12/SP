"""CPU regression of observation semantics; no model inference or real Gold edits."""
import pathlib, sys, tempfile, hashlib, json
import observe as o

def main():
    checks=[]
    def ck(c,name):
        if not c:raise AssertionError(name)
        checks.append(name)
    source={'root_id':'r','turns':[{'turn_ref':'S01:T001','speaker':'Alice','text':'The train leaves at noon.'},{'turn_ref':'S01:T002','speaker':'Alice','text':'The train leaves at noon. I saved the ticket.'},{'turn_ref':'S02:T001','speaker':'Bob','text':'I packed the bag.'}]}
    q={'candidate_id':'q','root_id':'r','question_type':'multi_select','option_evidence':{'A':{'relation':'supports','citations':[{'turn_ref':'S01:T001','speaker':'Alice','quote':'The train leaves at noon.'}]}},'gold':['A','B']}
    dm={'C':'contentA'};cm={'A':'contentA'}
    def cite(ref,quote,**kw):return o.validate_citation({'label':'C','status':'supported','turn_ref':ref,'quote':quote,**kw},source,q,dm,cm,'req',0)
    x=cite('S01:T002','The train leaves at noon.')
    ck(x['canonical_label']=='A','display_to_canonical_identity')
    ck(x['quote_exact_in_cited_turn'] and x['location_relation_to_author']=='DIFFERENT_TURN_SAME_SCENE_AND_SPEAKER','alternative_valid_turn_is_described')
    ck(x['semantic_judgment']=='NOT_DETERMINED_BY_LOCATION_OR_STRING_CHECK','alternative_turn_not_semantic_failure')
    ck(not x['quote_unique_across_source'] and len(x['quote_occurrence_turn_refs_in_complete_source'])==2,'repeated_quote_not_unique_anchor')
    ck(x['model_speaker_claim_status']=='NOT_REQUESTED_BY_SCHEMA' and x['model_speaker_matches_source'] is None,'missing_speaker_not_error')
    x=cite('S02:T001','The train leaves at noon.');ck(x['turn_exists'] and not x['quote_exact_in_cited_turn'],'quote_elsewhere_not_cited_turn')
    x=cite('S00:T001','The train leaves at noon.');ck(not x['turn_exists'] and not x['citation_mechanically_valid'],'missing_turn_is_provenance_failure')
    x=cite('S01:T001','The train leaves at noon.',speaker='Bob');ck(x['model_speaker_matches_source'] is False,'explicit_wrong_speaker_only_if_provided')
    x=o.validate_citation({'label':'C','status':'not_supported','turn_ref':'','quote':''},source,q,dm,cm,'req',0)
    ck(x['citation_mechanically_valid'] and x['empty_citation_status_compatible'],'allowed_empty_unsupported_citation')
    x=cite('S01:T001','leaves at noon.');ck(x['quote_exact_in_cited_turn'] and not x['exact_quote_equals_any_author_quote'],'shorter_quote_not_semantic_error')
    records=[{'variant_id':'p0','canonical_prediction':p,'valid':True} for p in [['A','B'],['A','B'],['A','C']]]+[
      {'variant_id':'p1','canonical_prediction':p,'valid':True} for p in [['A','B'],['A','B'],['A','B']]]
    p=o.permutation_observation(records,'multi_select')
    ck(p['within_mapping_sample_pairs']==6 and p['within_mapping_different_pairs']==2,'same_mapping_sampling_variation_count')
    ck(abs(p['within_mapping_pair_disagreement_valid_only']-1/3)<1e-10,'same_mapping_disagreement_rate')
    ck(abs(p['between_mapping_distribution_tv_mean_valid_only']-1/3)<1e-10,'between_mapping_distribution_separate')
    p=o.permutation_observation([{'variant_id':'p0','canonical_prediction':['I1','I2'],'valid':True},{'variant_id':'p1','canonical_prediction':['I2','I1'],'valid':True}],'ordering')
    ck(not p['within_mapping_variation_is_estimable'] and p['within_mapping_pair_disagreement_valid_only'] is None,'one_sample_cannot_estimate_sampling_noise')
    ck(p['ordering_single_sample_limitation'] and p['between_mapping_distribution_tv_mean_valid_only']==1,'ordering_cross_mapping_difference_descriptive')
    qs=[{'root_id':'r1','exact_rate':1},{'root_id':'r1','exact_rate':0},{'root_id':'r2','exact_rate':1}]
    s=o.summarize(qs,[]);ck(abs(s['question_equal_exact']-2/3)<1e-10 and abs(s['root_equal_exact']-.75)<1e-10,'root_equal_not_question_equal')
    row={'question_type':'multi_select','branch':'direct','display':{'options':dict.fromkeys('ABCDEF','x')}}
    a,e=o.parse_response({'status':'BACKEND_ERROR','raw_output':'<box>A,B</box>'},row);ck(a is None,'backend_failure_kept_invalid')
    a,e=o.parse_response({'status':'RETURNED','raw_output':'<box>A,B</box>','metadata':{'finish_reason':'length'}},row);ck(a is None,'truncated_even_parseable_is_invalid')
    a,e=o.parse_response({'status':'RETURNED','raw_output':'<box>A,A</box>','metadata':{'finish_reason':'stop'}},row);ck(a is None,'duplicate_answer_invalid')
    a,e=o.parse_response({'status':'RETURNED','raw_output':'<box>A,C</box>','metadata':{'finish_reason':'stop'}},row);ck(a==['A','C'] and e is None,'valid_direct_parse')
    # End-to-end synthetic frozen-artifact read: one successful and one failed call.
    with tempfile.TemporaryDirectory(prefix='day2_observation_cpu_') as td:
        root=pathlib.Path(td);run=root/'direct';run.mkdir()
        src={'root_id':'r','turns':source['turns']}
        sp=run/'sources/r.json';sp.parent.mkdir();sp.write_text(json.dumps(src))
        publicq={'candidate_id':'q1','root_id':'r','question_type':'multi_select','gold':['A','B'],'options':dict.fromkeys('ABCD','x'),
          'source_path':str(sp),'source_sha256':o.sha(sp),'option_evidence':{k:{'relation':'supports' if k in 'AB' else 'contradicts','citations':[{'turn_ref':'S01:T001','speaker':'Alice','quote':'The train leaves at noon.'}]} for k in 'ABCD'}}
        pp=root/'private.json';pp.write_text(json.dumps({'questions':[publicq]}))
        prompt='\n'.join(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" for t in src['turns'])
        requests=[];score_records=[];mappings={}
        for i,returned in enumerate([True,False]):
            rid=f'req{i}';p=run/'prompts'/f'{rid}.txt';p.parent.mkdir(exist_ok=True);p.write_text(prompt)
            requests.append({'request_id':rid,'candidate_id':'q1','root_id':'r','question_type':'multi_select','branch':'direct','variant_id':'p0','slot':i,
              'prompt_sha256':o.sha(p),'display':{'options':dict.fromkeys('ABCD','x')}})
            mappings[rid]={'display_to_content':{k:'c'+k for k in 'ABCD'}}
            raw={'request_id':rid,'candidate_id':'q1','root_id':'r','question_type':'multi_select','status':'RETURNED' if returned else 'BACKEND_ERROR',
              'raw_output':'<box>A,B</box>' if returned else None,'metadata':{'finish_reason':'stop'} if returned else {},'response_committed_at_utc':'2026-10-19T00:00:00Z'}
            o.save(run/'responses'/f'{rid}.json',raw);score_records.append({'request_id':rid,'valid':returned,'exact':returned})
        o.save(run/'private/DISPLAY_MAPPINGS.json',{'public_to_canonical':{'q1':{k:'c'+k for k in 'ABCD'}},'requests':mappings})
        o.save(run/'manifest.json',{'requests':requests,'private_answer_binding':o.bind(pp),'branch':'direct'})
        o.save(run/'RAW_RESPONSE_FREEZE.json',{'files':[o.bind(p) for p in sorted((run/'responses').glob('*.json'))],'manifest_binding':o.bind(run/'manifest.json')})
        o.save(run/'SCORES.json',{'status':'FAILED_WITH_FIXED_DENOMINATOR','raw_response_freeze_binding':o.bind(run/'RAW_RESPONSE_FREEZE.json'),'denominator_calls':2,
          'requests':score_records,'questions':[{'candidate_id':'q1','stratum':'occasionally_correct'}],'cost':{}})
        result=o.load_run(run);ck(result['summary']['planned_calls']==2 and result['summary']['valid_calls']==1 and result['summary']['question_equal_exact']==.5,'end_to_end_failed_call_retained_in_denominator')
        ck(result['summary']['multi_select_valid_only']['tp']==2 and result['summary']['multi_select_valid_only']['fp']==0,'end_to_end_valid_only_tp_fp_fn')
        ck(result['questions'][0]['permutation_observations']['within_mapping_sample_pairs']==0,'failed_sample_not_falsely_counted_as_semantic_prediction')
    obj={'status':'PASS','checks':len(checks),'check_names':checks,'cpu_only':True,'new_solver_calls':0,'tool_binding':o.bind(pathlib.Path(o.__file__)),'test_binding':o.bind(pathlib.Path(__file__))}
    target=pathlib.Path(sys.argv[1]) if len(sys.argv)>1 else o.HERE/'CPU_PRECHECK.json';o.save(target,obj);print(f'PASS: {len(checks)} meaningful observation checks')
if __name__=='__main__':main()
