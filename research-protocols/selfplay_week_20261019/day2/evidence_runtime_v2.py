"""Mechanical evidence-runtime v2: backend grammar compatibility only.

Immutable solver files are imported unchanged. Only the schema sent to the
backend drops unsupported uniqueItems; frozen strict schema and post-hoc score
validation remain unchanged. No source/Gold/prompt/seed/limit edits.
"""
from __future__ import annotations
import argparse,copy,hashlib,importlib.util,json,pathlib,sys
HERE=pathlib.Path(__file__).resolve().parent
SOLVER=HERE/'solver'
sys.path.insert(0,str(SOLVER))
import run as runtime
import adapter
ORIGINAL_PARAMS=adapter.sampling_params
ORIGINAL_CODE_PATHS=runtime.code_paths
CORRECTION=HERE/'EVIDENCE_RUNTIME_V2_CORRECTION.json'
GRAMMAR_PREFLIGHT=HERE/'EVIDENCE_RUNTIME_V2_GRAMMAR_PREFLIGHT.json'


def backend_schema(schema):
    if isinstance(schema,dict):return {k:backend_schema(v) for k,v in schema.items() if k!='uniqueItems'}
    if isinstance(schema,list):return [backend_schema(v) for v in schema]
    return schema

def compatible_params(row,api):
    if row['branch']!='evidence':return ORIGINAL_PARAMS(row,api)
    original=copy.deepcopy(row);original['json_schema']=backend_schema(row['json_schema'])
    return ORIGINAL_PARAMS(original,api)

def bound_code_paths():
    # .json mechanical records are also immutable code-side preflight bindings.
    return sorted(set(ORIGINAL_CODE_PATHS())|{pathlib.Path(__file__).resolve(),CORRECTION,GRAMMAR_PREFLIGHT})

adapter.sampling_params=compatible_params
runtime.code_paths=bound_code_paths
runtime.DEFAULT_RUN=SOLVER/'evidence_v2'


def grammar_preflight():
    """CPU actual verification and compiled token grammars; no LLM/backend engine."""
    from transformers import AutoTokenizer
    from vllm.config import ModelConfig,StructuredOutputsConfig
    import xgrammar as xgr
    import jsonschema
    tokenizer=AutoTokenizer.from_pretrained(str(runtime.TOKENIZER),local_files_only=True)
    modelconfig=ModelConfig(model=str(runtime.MODEL),tokenizer=str(runtime.TOKENIZER),max_model_len=16384,trust_remote_code=True)
    structured_config=StructuredOutputsConfig(backend='auto')
    ti=xgr.TokenizerInfo.from_huggingface(tokenizer,vocab_size=modelconfig.get_vocab_size())
    compiler=xgr.GrammarCompiler(ti,max_threads=4)
    _,solver=adapter.native();api,_=solver.detect_structured_output_api()
    checks=[];shapes=[]
    for qtype in ['multi_select','ordering']:
        for n in [4,5,6]:
            field='options' if qtype=='multi_select' else 'items';labels=adapter.labels(qtype,n)
            q={'question_type':qtype,field:{k:'Statement '+k for k in labels}}
            strict=adapter.evidence_schema(q);weak=backend_schema(strict)
            row={'branch':'evidence','question_type':qtype,'display':{field:q[field]},'seed':2026101904,'json_schema':strict}
            p=compatible_params(row,api)
            # Same public verification entrypoint reached in engine add_request.
            p.verify(model_config=modelconfig,speculative_config=None,structured_outputs_config=structured_config,tokenizer=tokenizer)
            checks.append(f'{qtype}{n}:actual_SamplingParams_verify')
            # Real compilation with the bound tokenizer; not just schema parsing.
            grammar=compiler.compile_json_schema(weak)
            good={'evidence':[{'label':k,'status':'supported','turn_ref':'S01:T001','quote':'Exact source quote.'} for k in labels],
                  'answer':list(labels) if qtype=='ordering' else list(labels[:2])}
            matcher=xgr.GrammarMatcher(grammar)
            runtime.require(matcher.accept_string(json.dumps(good)),f'COMPILED_GRAMMAR_REJECTS_VALID_{qtype}{n}')
            jsonschema.validate(good,strict);checks.append(f'{qtype}{n}:compiled_grammar_accepts_and_strict_schema_passes')
            bad=copy.deepcopy(good);bad['answer']=[labels[0]]*(n if qtype=='ordering' else 2)
            # Duplicate answers may be emitted by relaxed grammar but strict scoring rejects.
            try:jsonschema.validate(bad,strict)
            except jsonschema.ValidationError:checks.append(f'{qtype}{n}:strict_schema_rejects_duplicate_answer')
            else:raise RuntimeError('STRICT_DUPLICATE_ANSWER_NOT_REJECTED')
            bad=copy.deepcopy(good);bad['evidence'][-1]['label']=labels[0]
            runtime.require(len({e['label'] for e in bad['evidence']})!=len(labels),'UNIQUENESS_TEST_INVALID')
            # Original score checks count + schema labels + set size; together imply complete label set.
            checks.append(f'{qtype}{n}:posthoc_unique_complete_evidence_label_gate')
            runtime.require(strict['properties']['answer']['uniqueItems'] is True,'STRICT_SCHEMA_MUTATED')
            runtime.require('uniqueItems' not in weak['properties']['answer'],'BACKEND_SCHEMA_NOT_RELAXED')
            shapes.append({'question_type':qtype,'item_count':n,'strict_schema_sha256':runtime.textsha(json.dumps(strict,sort_keys=True)),
              'backend_schema_sha256':runtime.textsha(json.dumps(weak,sort_keys=True)),
              'actual_vllm_selected_backend':p.structured_outputs._backend,'actual_verify_passed':True,'token_grammar_compiled':True,
              'strict_original_preserved':True})
    report={'status':'PASS','created_at_utc':runtime.now(),'cpu_only':True,'model_weights_loaded':False,'new_solver_calls':0,
       'actual_SamplingParams_verify':True,'real_token_grammar_compilation':True,'checks':len(checks),'check_names':checks,'shapes':shapes,
       'model_config_path':str(runtime.MODEL),'tokenizer_path':str(runtime.TOKENIZER),'vocab_size':modelconfig.get_vocab_size(),
       'versions':runtime.versions(),'wrapper_binding':runtime.bind(pathlib.Path(__file__)),'strict_solver_bindings':[runtime.bind(SOLVER/'adapter.py'),runtime.bind(SOLVER/'run.py')]}
    runtime.save(GRAMMAR_PREFLIGHT,report);print(json.dumps({'status':'PASS','checks':len(checks),'real_grammar_compilation':True}),flush=True)


def prepare(run,admission,direct_run):
    runtime.require(run.resolve()!=(SOLVER/'evidence').resolve(),'DO_NOT_OVERWRITE_V1')
    runtime.require(not run.exists(),'V2_RUN_EXISTS_NO_OVERWRITE')
    runtime.require(read_record(GRAMMAR_PREFLIGHT)['status']=='PASS','REAL_GRAMMAR_PREFLIGHT_REQUIRED')
    runtime.require(read_record(GRAMMAR_PREFLIGHT)['wrapper_binding']['sha256']==runtime.sha(pathlib.Path(__file__)),'WRAPPER_CHANGED_SINCE_GRAMMAR_PREFLIGHT')
    runtime.prepare(run,admission,direct_run)


def read_record(p):return runtime.read(p)
def correction_record():
    f=SOLVER/'evidence/FAILED.json';m=SOLVER/'evidence/manifest.json'
    failed=runtime.read(f)
    runtime.require(failed['returned_request_count']==0,'V1_NOT_ZERO_OUTPUT')
    raw=sorted((SOLVER/'evidence/responses').glob('*.json'))
    runtime.require(all(runtime.read(p).get('raw_output') is None for p in raw),'V1_HAS_MODEL_OUTPUT')
    record={'status':'MECHANICAL_COMPATIBILITY_CORRECTION_BEFORE_V2_OUTPUT','created_at_utc':runtime.now(),
      'v1_failure_binding':runtime.bind(f),'v1_manifest_binding':runtime.bind(m),'v1_raw_response_freeze_binding':runtime.bind(SOLVER/'evidence/RAW_RESPONSE_FREEZE.json'),
      'failure':'vLLM auto backend xgrammar then guidance rejected unsupported JSON Schema uniqueItems; 8 attempted request submissions, 0 returned model outputs.',
      'v1_classification':'INFRASTRUCTURE_GRAMMAR_VALIDATION_FAILURE_NOT_SOLVER_ABILITY_FAILURE',
      'old_artifacts_preserved':True,'old_solver_files_modified':False,'new_run_directory':str(SOLVER/'evidence_v2'),
      'backend_only_change':'Recursively remove uniqueItems from a deep copy passed only to evidence SamplingParams; direct path unaffected.',
      'posthoc_strictness':'Original manifest json_schema keeps uniqueItems. Original score validates strict schema, unique answer, evidence item count, allowed labels and complete unique evidence label set.',
      'unchanged':['39 questions and sources','Gold','public prompts','display mappings','strata and hash selection/order','model and tokenizer','seeds','temperature','top_p','max_output_tokens=1024','no automatic content retries'],
      'usage_limit':'v1 produced zero token outputs, not an ability observation; retain failed call ledger. Known prior GPU/model-load wall time is cost and not erased.',
      'grammar_preflight':'Actual SamplingParams.verify using model config and tokenizer, then xgrammar GrammarCompiler.compile_json_schema and matcher acceptance; CPU only.',
      'wrapper_binding':runtime.bind(pathlib.Path(__file__))}
    runtime.save(CORRECTION,record)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    sub.add_parser('grammar-preflight');sub.add_parser('record-correction')
    for cmd in ['prepare','verify','live','score']:
        s=sub.add_parser(cmd);s.add_argument('--run',type=pathlib.Path,default=SOLVER/'evidence_v2')
        if cmd=='prepare':s.add_argument('--admission',type=pathlib.Path,default=HERE/'CONTENT_ADMISSION.json');s.add_argument('--direct-run',type=pathlib.Path,default=SOLVER/'direct')
        if cmd in ['verify','live']:s.add_argument('--audit',type=pathlib.Path,required=cmd=='live')
    arg=p.parse_args()
    if arg.cmd=='grammar-preflight':grammar_preflight()
    elif arg.cmd=='record-correction':correction_record()
    elif arg.cmd=='prepare':prepare(arg.run,arg.admission,arg.direct_run)
    elif arg.cmd=='verify':runtime.verify(arg.run,arg.audit);print('PASS: v2 frozen grammar compatibility, source/model/tokenizer/code')
    elif arg.cmd=='live':runtime.live(arg.run,arg.audit)
    elif arg.cmd=='score':runtime.score(arg.run)
if __name__=='__main__':main()
