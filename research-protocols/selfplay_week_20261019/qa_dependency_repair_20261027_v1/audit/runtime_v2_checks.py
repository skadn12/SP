import ast, copy, datetime, hashlib, json, sys, tempfile
from collections import Counter
from pathlib import Path
sys.path.insert(0, '/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/qa_dependency_repair_20261027_v1')
import pilot_runner_v2 as v2
v1=v2.load('read_only_attempt_one_runner_for_comparison',v2.HERE/'pilot_runner.py')
from vllm import SamplingParams, sampling_params
from vllm.v1.structured_output.backend_xgrammar import validate_xgrammar_grammar
from vllm.v1.structured_output.backend_guidance import validate_guidance_grammar
checks=[]
def verify(name,condition):
    assert condition,name
    checks.append({'name':name,'status':'PASS'})
def error(name,fn,marker=None):
    try:fn()
    except Exception as exc:
        if marker is not None:assert marker in str(exc),(marker,str(exc))
        checks.append({'name':name,'status':'PASS','observed_error':str(exc)})
        return
    raise AssertionError(name+' did not fail')
oldtree=ast.parse((v2.HERE/'pilot_runner.py').read_text())
newtree=ast.parse((v2.HERE/'pilot_runner_v2.py').read_text())
oldfunc={n.name:n for n in oldtree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
newfunc={n.name:n for n in newtree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
for name in ('make_prompt','conditions','sampling_seed','validate_private','summarize','review_records','checked_raw','run'):
    verify(name+' AST unchanged',ast.dump(oldfunc[name])==ast.dump(newfunc[name]))
newscore=copy.deepcopy(newfunc['score_one'])
for node in ast.walk(newscore):
    if isinstance(node,ast.For):
        node.body=[s for s in node.body if not (isinstance(s,ast.Expr) and isinstance(s.value,ast.Call) and any(isinstance(arg,ast.Constant) and arg.value=='DUPLICATE_TURN_REFERENCE' for arg in s.value.args))]
verify('score implementation unchanged after explicit post-generation duplicate-ref check',ast.dump(newscore)==ast.dump(oldfunc['score_one']))
expected=copy.deepcopy(v1.output_schema())
for item in expected['properties']['item_evidence']['properties'].values():del item['properties']['turn_refs']['uniqueItems']
verify('schema delta is exactly removal of six turn_refs uniqueItems',expected==v2.output_schema())
verify('all generation constants unchanged',(v1.MAX_TOKENS,v1.MODEL_LEN,v1.BATCH,v1.SAMPLING,v1.engine.STRUCTURED_CONFIG)==(v2.MAX_TOKENS,v2.MODEL_LEN,v2.BATCH,v2.SAMPLING,v2.engine.STRUCTURED_CONFIG))
params=SamplingParams(max_tokens=v2.MAX_TOKENS,structured_outputs=sampling_params.StructuredOutputsParams(json=v1.output_schema()))
error('frozen v1 xgrammar failure reproduced',lambda:validate_xgrammar_grammar(params),'features not supported')
error('frozen v1 guidance failure reproduced',lambda:validate_guidance_grammar(params))
fixture={'candidate_id':'CPU-only','root_id':'CPU-only','question_type':'multi_select','question':'CPU fixture',
         'options':{f'item{i}':f'Option {i}' for i in range(6)},'gold':['item0']}
pub,pres=v2.quality.prepare_display(fixture,2026102701)
row={k:pub[k] for k in ('candidate_id','root_id','question_type','question')}
row.update(request_id='CPU-only',display=pub['options'],cohort='new',condition='FULL',variant_id='perm_00')
info={'presentation':pres,'visible_refs':['R01:001']}
obj={'item_evidence':{k:{'judgment':'ambiguous','turn_refs':[],'reason':'CPU fixture'} for k in 'ABCDEF'},'answer_reason':'CPU fixture','final_answer':'<box></box>'}
obj['item_evidence']['A']['turn_refs']=['R01:001','R01:001']
raw={'request_id':'CPU-only','status':'RETURNED','finish_reason':'stop','capture_errors':[],'raw_output':json.dumps(obj)}
sc=v2.score_one(row,info,raw,v2.output_schema())
verify('duplicates still invalid J=None/exact=false after schema constraint removal',not sc['valid'] and sc['J'] is None and not sc['exact'] and 'DUPLICATE_TURN_REFERENCE' in sc['errors'])
obj['item_evidence']['A']['turn_refs']=['R01:001'];raw['raw_output']=json.dumps(obj)
sc=v2.score_one(row,info,raw,v2.output_schema())
verify('unique ref and empty selection remain valid with J=0',sc['valid'] and sc['J']==0 and not sc['exact'])
oldrows=v2.read(v2.V1_OUT/'PUBLIC_REQUESTS.json')['requests']
oldprivate=v2.read(v2.V1_OUT/'PRIVATE_SCORING.json')['requests']
oldtokens=v2.read(v2.V1_OUT/'PROMPT_TOKEN_IDS.json')
oldprep=v2.read(v2.V1_OUT/'PREPARATION.json')
with tempfile.TemporaryDirectory(prefix='qa-v2-equivalence-') as tmp:
    rows=copy.deepcopy(oldrows);d=Path(tmp)
    for rr,old in zip(rows,oldrows):
        pp=d/(rr['request_id']+'.txt');pp.symlink_to(old['prompt_path'])
        sp=d/(rr['request_id']+'.json');sp.write_text(json.dumps(v2.output_schema()))
        rr.update(prompt_path=str(pp),schema_path=str(sp))
    eq=v2.check_v1_equivalence(rows,oldprivate,oldtokens,oldprep['config_receipt'],oldprep['runtime_versions'])
verify('v1 equivalence guard accepts all 204 reference requests with only schema/path delta',eq['requests']==204)
freeze=v2.read(v2.V1_OUT/'RAW_RESPONSE_FREEZE.json')
counts=Counter(v2.read(v2.V1_OUT/'raw'/(rid+'.json'))['status'] for rid in freeze['request_ids'])
verify('failed v1 attempt preserved with zero returned outputs',freeze['returned']==0 and counts=={'BACKEND_ERROR':8,'NOT_SUBMITTED':196})
for bound in oldprep['bindings']+oldprep['private_bindings']:v2.check(bound)
v2.check(freeze['started_binding'])
for key in ('bindings','submission_bindings','batch_bindings'):
    for bound in freeze[key]:v2.check(bound)
verify('all frozen v1 preparation/private/raw/submission/batch/started bindings still match',True)
result={'status':'PASS','checks':checks,'check_count':len(checks),'reference_equivalence':eq,
        'reference_equivalence_scope':'204 frozen v1 rows rehomed in temporary fixtures; unchanged prompt-renderer AST checked. Actual newly prepared v2 requests must pass the same guard before v2 PREPARATION is written.',
        'failed_v1_status_counts':dict(counts),'gpu_model_started':False,'prepare_called':False}
Path('/tmp/qa_v2_runtime_check_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('checks','reference_equivalence')},ensure_ascii=False))
