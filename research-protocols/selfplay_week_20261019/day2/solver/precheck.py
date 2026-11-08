"""Meaningful CPU regression checks; does not create or call any model backend."""
import copy, importlib.util, itertools, json, math, pathlib, sys, tempfile
import adapter as a
import run

def main():
    checks=[]
    def check(c,name):
        if not c:raise AssertionError(name)
        checks.append(name)
    contract,solver=a.native();native=a.native_scoring()
    path=a.NATIVE/'protocol.py';spec=importlib.util.spec_from_file_location('precheck_original_v15',path);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    api,_=solver.detect_structured_output_api()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(run.TOKENIZER),local_files_only=True)
    source={'root_id':'synthetic-root','turns':[{'turn_ref':'S01:T001','session_ref':'S01','speaker':'Alice','text':'MARKER_FIRST Alice sent the revised agenda.'},{'turn_ref':'S02:T001','session_ref':'S02','speaker':'Bob','text':'MARKER_LAST Bob saved the telephone number.'}]}
    for typ,n in itertools.product(['multi_select','ordering'],[4,5,6]):
        field='options' if typ=='multi_select' else 'items';keys=a.labels(typ,n)
        q={'candidate_id':f'synthetic-{typ}-{n}','root_id':'synthetic-root','question_type':typ,'question':'Synthetic syntax and mapping check only.',field:{k:f'Unique statement {i}' for i,k in enumerate(keys)},'source_path':'/synthetic-source','source_sha256':'0'*64}
        a.public_question(q);vs=a.variants(q,2026101903,6 if typ=='multi_select' else 12)
        # Determinism and no Gold dependence are checked with poisoned metadata.
        poisoned=copy.deepcopy(q);poisoned.update(gold=list(reversed(keys)),option_evidence={'SECRET':'LEAK'},solver_score=1.0)
        check(vs==a.variants(poisoned,2026101903,len(vs)),f'{typ}{n}:gold_blind_rebind')
        if n==6:check(vs==old.make_variants(q,seed=2026101903,count=len(vs)),f'{typ}{n}:exact_native_v15_variants')
        for idx,v in enumerate(vs):
            check(len(set(v['display_to_content'].values()))==n,f'{typ}{n}:{idx}:bijection')
            prompt=a.direct_prompt(tok,q,v,source);poison_prompt=a.direct_prompt(tok,poisoned,v,source)
            check(prompt==poison_prompt and 'LEAK' not in prompt and 'SECRET' not in prompt,f'{typ}{n}:{idx}:prompt_metadata_invariant')
            check(all(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" in prompt for t in source['turns']),f'{typ}{n}:{idx}:full_source')
            choices=a.choices(typ,v[field]);check(len(choices)==(2**n-1 if typ=='multi_select' else math.factorial(n)),f'{typ}{n}:{idx}:finite_choice_count')
            row={'branch':'direct','question_type':typ,'display':{field:v[field]},'seed':123}
            params=a.sampling_params(row,api);check(params.max_tokens==32 and params.temperature==.8 and params.top_p==.9,f'{typ}{n}:{idx}:real_vllm_params')
            # Exhaustively roundtrip every admissible answer: catches label/mapping/order corruption.
            for text in choices:
                vals=native.parse_box(text)[0];gold=[v['display_to_content'][x] for x in vals]
                if len(gold)<2:continue # Native partial metric requires >=2 Gold; actual pool uses 2..5.
                s=native.score_response({'question_type':typ,'gold_content_ids':gold},{'variant':v},text)
                check(s['exact_match'] is True and s['partial_score']=='1',f'{typ}{n}:{idx}:roundtrip:{text}')
        ev=a.evidence_schema(q);row={'branch':'evidence','question_type':typ,'display':{field:vs[0][field]},'seed':999,'json_schema':ev}
        par=a.sampling_params(row,api);check(par.max_tokens==1024,f'{typ}{n}:json_schema_constructor')
        import jsonschema
        answer=list(keys) if typ=='ordering' else list(keys[:2]);obj={'evidence':[{'label':k,'status':'supported','turn_ref':'S01:T001','quote':'Alice sent'} for k in keys],'answer':answer}
        jsonschema.validate(obj,ev);check(True,f'{typ}{n}:json_roundtrip')
    for bad in ['<box>A,A</box>','Answer: <box>A,B</box>','<box>Z</box>','<box>I1,I2</box>']:
        # Check parser or allowed finite set rejects malformed/unsupported syntax.
        check(bad not in a.choices('multi_select',dict.fromkeys('ABCDEF','x')),f'invalid_answer_rejected:{bad}')
    q=copy.deepcopy(q);q['gold']=['I1','I2']
    try:a.public_question(q)
    except ValueError:check(True,'public_gold_whitelist_rejects')
    else:check(False,'public_gold_whitelist_rejects')
    # Fixed denominator failure behavior: missing/invalid is never a successful exact result.
    v=vs[0];g=list(v['display_to_content'].values())
    s=native.score_response({'question_type':'ordering','gold_content_ids':g},{'variant':v},None)
    check(s['scored'] is False and s['exact_match'] is None,'missing_response_is_not_exact')
    output={'status':'PASS','cpu_only':True,'model_backend_created':False,'new_solver_calls':0,'checks':len(checks),'versions':run.versions(),
      'dynamic_shapes':['multi_select4','multi_select5','multi_select6','ordering4','ordering5','ordering6'],
      'exhaustive_choice_roundtrips':True,'gold_blind_permutations':True,'native_six_item_exact_equivalence':True,'native_finite_choice_decoding':True,
      'test_code_binding':run.bind(pathlib.Path(__file__)),'implementation_bindings':[run.bind(p) for p in [pathlib.Path(a.__file__),pathlib.Path(run.__file__)]]}
    target=pathlib.Path(sys.argv[1]) if len(sys.argv)>1 else run.HERE/'CPU_PRECHECK.json'
    run.save(target,output);print(json.dumps(output,ensure_ascii=False))
if __name__=='__main__':main()
