"""Synthetic CPU-only end-to-end checks; never calls CUDA or loads weights."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('priority_runner_selfcheck', HERE / 'solver_runner.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
checks = []

def record(name, fn):
    fn()
    checks.append({'check': name, 'status': 'PASS'})

def rejects(name, fn, substring=None):
    try:
        fn()
    except (ValueError, AssertionError, RuntimeError) as error:
        if substring:
            assert substring in str(error), (name, str(error))
        checks.append({'check': name, 'status': 'PASS', 'rejected': str(error)})
    else:
        raise AssertionError('failed to reject: ' + name)

ast.parse((HERE / 'solver_runner.py').read_text())
checks.append({'check': 'runner syntax and imports', 'status': 'PASS'})
endpoint, config_receipt = m.endpoint_receipt()
assert config_receipt['effective_rope_theta'] == 1000000
checks.append({'check': 'real B1 and reference config bridge', 'status': 'PASS'})
config = SimpleNamespace(rope_parameters={'rope_type': 'default', 'rope_theta':1000000}, rope_theta=1000000)
record('actual loaded config gate correct theta', lambda: m.checked_runtime_config(config))
for name, bad in [
    ('wrong nested theta', SimpleNamespace(rope_parameters={'rope_type':'default','rope_theta':10000},rope_theta=1000000)),
    ('conflicting top theta', SimpleNamespace(rope_parameters={'rope_type':'default','rope_theta':1000000},rope_theta=10000)),
    ('missing explicit nested theta', SimpleNamespace(rope_theta=1000000)),
]:
    rejects(name, lambda bad=bad: m.checked_runtime_config(bad))
rejects('run requires explicit GPU flag', lambda: m.run(False), 'EXPLICIT_GPU_FLAG_REQUIRED')

with tempfile.TemporaryDirectory(prefix='runner_cpu_fixture_', dir=HERE) as tmp:
    root = Path(tmp)
    m.NEW, m.REVIEW, m.SELECTION = (root / x for x in ('CANDIDATES_PRIVATE.json','CONTENT_REVIEW_AI.json','SELECTION_FREEZE.json'))
    m.OPERATION_PATH, m.OUT = root / 'OPERATION_INSTRUCTIONS.json', root / 'execution'
    m.base.OUT = m.OUT
    source = {'root_id':'synthetic-train-root', 'turns':[
        {'turn_ref':f'turn_{i}', 'speaker':'Alex','text':f'The source assigns stage {i} the value {i*10}.'}
        for i in range(1,5)]}
    source_path = root / 'source.json'
    m.save(source_path, source)
    binding = m.bind(source_path)
    m.save(m.SELECTION, {'roots':[{'root_id':source['root_id'], 'split':'train', **binding}], 'planned_questions':2})
    evidence = lambda i: [{'turn_ref':f'turn_{i}','speaker':'Alex','quote':source['turns'][i-1]['text']}]
    common = dict(root_id=source['root_id'],author_id='synthetic_author',source_binding=binding)
    mc = dict(common, candidate_id='SYNTH-MC',question_type='multi_select',task_operation='MC_CONSTRAINT_CONJUNCTION',
        question='Which claims agree with the supplied source?',
        options={f'o{i}': f'The value assigned to stage {i if i < 3 else 1} is {i*10}.' for i in range(1,7)},gold=['o1','o2'])
    mc['item_analysis']={k:{'should_select':k in mc['gold'],'evidence':evidence(min(i,4))} for i,k in enumerate(mc['options'],1)}
    ordering = dict(common,candidate_id='SYNTH-ORDER',question_type='ordering',task_operation='ORDER_EXPLICIT_TIME',
        question='Order the four source values from low to high.',items={f'i{i}':f'Stage {i} value' for i in range(1,5)},gold=[f'i{i}' for i in range(1,5)])
    ordering['item_analysis']={f'i{i}':{'evidence':evidence(i)} for i in range(1,5)}
    ordering['ordering_analysis']={'operation_class':'clock_value','edges':[{'before':f'i{i}','after':f'i{i+1}','evidence':evidence(i)+evidence(i+1)} for i in range(1,4)]}
    candidates=[mc,ordering]
    reviews=[]
    for candidate in candidates:
        review=m.production_gate.build_review_packet(candidate,m.SELECTION)['review_template']
        review.update(reviewer_id='synthetic_reviewer',decision='PASS')
        review['checks']={k:{'decision':'PASS','rationale':'Synthetic mechanical fixture; no experimental admission.'} for k in review['checks']}
        reviews.append(review)
    m.save(m.NEW,candidates)
    m.save(m.REVIEW,{'reviews':reviews})
    for candidate in candidates:
        variants=list(m.displays(candidate))
        maps=[p[2]['display_to_canonical'] for p in variants]
        assert len(maps)==2 and maps[0]!=maps[1]
        assert all(list(p.values())!=list(candidate['options' if candidate['question_type']=='multi_select' else 'items']) for p in maps)
        changed=copy.deepcopy(candidate)
        changed['gold']=list(reversed(changed['gold'])) if changed['question_type']=='ordering' else ['o3','o4','o5']
        assert maps==[p[2]['display_to_canonical'] for p in m.displays(changed)]
    checks.append({'check':'both types have distinct nonidentity Gold-independent displays', 'status':'PASS'})
    m.prepare()
    prep,rows,tokens=m.public_preparation()
    assert len(rows)==8 and len({r['candidate_id'] for r in rows})==2
    assert {r['condition'] for r in rows}==set(m.CONDITIONS)
    checks.append({'check':'real tokenizer + xgrammar prepare full eight-request synthetic factorial', 'status':'PASS'})
    rejects('preparation cannot overwrite',m.prepare,'NO_PREPARATION_OVERWRITE')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(m.TOKENIZER),local_files_only=True)
    drift=copy.deepcopy(tokens)
    drift[rows[0]['request_id']][0]+=1
    rejects('same-length prompt token drift',lambda:m.checked_prompt_tokens(tokenizer,rows,drift),'PROMPT_TOKEN_CONTENT_DRIFT')
    real_read=m.read
    forbidden={p.resolve() for p in (m.NEW,m.REVIEW,m.SELECTION,m.OUT/'PRIVATE_SCORING.json')}
    reads=[]
    def guarded_read(path):
        assert Path(path).resolve() not in forbidden,'GENERATOR_OPENED_GOLD_INPUT'
        reads.append(str(path))
        return real_read(path)
    m.read=guarded_read
    m.base.read=guarded_read
    original_endpoint_receipt=m.endpoint_receipt
    m.endpoint_receipt=lambda: ({**endpoint,'model_bindings':[m.bind(m.MODEL/'config.json')],'tokenizer_bindings':[m.bind(m.TOKENIZER/'config.json')]}, config_receipt)
    fake=ModuleType('vllm')
    fake.SamplingParams=lambda **kwargs:SimpleNamespace(**kwargs)
    fake.sampling_params=SimpleNamespace(StructuredOutputsParams=lambda **kwargs:SimpleNamespace(**kwargs))
    class FakeLLM:
        def __init__(self,**kwargs):
            assert kwargs['hf_overrides']['rope_theta']==1000000
            self.llm_engine=SimpleNamespace(model_config=SimpleNamespace(model=str(m.MODEL),hf_config=config,hf_text_config=config),engine_core=SimpleNamespace(shutdown=lambda:None))
        def generate(self,prompts,params,use_tqdm=False):
            prompt=prompts[0]
            row=next(r for r in rows if Path(r['prompt_path']).read_text()==prompt)
            labels=row['labels']
            answer=labels[:2] if row['question_type']=='multi_select' else labels
            parsed={'item_evidence':{k:{'judgment':'supported' if k in answer else 'contradicted','turn_refs':['turn_1'] if row['condition']=='FULL' else [],'reason':'Synthetic concise reason.'} for k in labels},'answer_reason':'Synthetic brief answer reason.','final_answer':'<box>'+' '.join(answer)+'</box>'}
            return [SimpleNamespace(prompt=prompt,prompt_token_ids=tokens[row['request_id']],outputs=[SimpleNamespace(text=json.dumps(parsed),finish_reason='stop',stop_reason=None,token_ids=[1,2,3])])]
    fake.LLM=FakeLLM
    sys.modules['vllm']=fake
    m.run(True)
    checks.append({'check':'CPU fake backend run sealed all returns without opening private inputs', 'status':'PASS','guarded_reads':len(reads)})
    m.read=real_read
    m.base.read=real_read
    frozen=m.read(m.OUT/'RAW_RESPONSE_FREEZE.json')
    record('complete raw freeze',lambda:m.checked_raw_manifest(frozen,rows,m.OUT))
    for name,modify in [
        ('missing raw binding',lambda f:f['bindings'].pop()),
        ('missing validation binding',lambda f:f['validation_bindings'].pop()),
        ('missing submitted receipt binding',lambda f:f['submission_bindings'].pop()),
        ('incorrect returned count',lambda f:f.update(returned_requests=7)),
        ('returned request not submitted',lambda f:(f['submitted_request_ids'].pop(),f.update(submitted_requests=7))),
        ('missing model receipt',lambda f:f.update(model_load_receipt_binding=None)),
    ]:
        bad=copy.deepcopy(frozen)
        modify(bad)
        rejects(name,lambda bad=bad:m.checked_raw_manifest(bad,rows,m.OUT))
    raw_path=m.OUT/'raw'/(rows[0]['request_id']+'.json')
    original_bytes=raw_path.read_bytes()
    raw_path.write_bytes(original_bytes+b' ')
    rejects('raw response content tamper',lambda:m.checked_raw_manifest(frozen,rows,m.OUT))
    raw_path.write_bytes(original_bytes)
    row=next(r for r in rows if r['condition']=='QUESTION_ONLY')
    raw=m.read(m.OUT/'raw'/(row['request_id']+'.json'))
    parsed=json.loads(raw['raw_output'])
    schema=m.read(row['schema_path'])
    parsed['item_evidence']['A']['turn_refs']=['turn_1']
    try:
        m.base.validate_output(json.dumps(parsed),row,schema)
    except Exception:
        checks.append({'check':'QUESTION_ONLY invented citation rejected','status':'PASS'})
    else:
        raise AssertionError('QUESTION_ONLY invented citation accepted')
    parsed=json.loads(raw['raw_output'])
    parsed={'final_answer':parsed['final_answer'],'item_evidence':parsed['item_evidence'],'answer_reason':parsed['answer_reason']}
    rejects('answer before reasons rejected',lambda:m.base.validate_output(json.dumps(parsed),row,schema),'REASONS_MUST_PRECEDE_ANSWER')
    m.base.analyze()
    scores=m.read(m.OUT/'SCORES.json')['records']
    summary=m.read(m.OUT/'ANALYSIS_SUMMARY.json')
    assert len(scores)==summary['requests']==summary['valid']==8
    assert len(summary['variant_pairs'])==4
    assert not summary['feedback_eligible'] and not summary['training_admission']
    checks.append({'check':'analyze after full raw freeze scores all eight fixture responses', 'status':'PASS'})
    rejects('run cannot repeat',lambda:m.run(True),'NO_RERUN_OR_OVERWRITE')

report={'status':'PASS','checks':checks,'passed':len(checks),'runner_binding':m.bind(HERE/'solver_runner.py'),
    'test_binding':m.bind(Path(__file__)),'runtime_versions':m.versions(),
    'scope':'CPU-only synthetic sources/candidates/review/backend; real tokenizer/schema compilation and config preflight. No weights loaded, CUDA calls, training, or experimental Solver results.',
    'synthetic_fixtures_removed':True}
report_path=HERE/'RUNNER_CPU_CHECK.json'
if report_path.exists():
    raise RuntimeError('No report overwrite')
m.save(report_path,report)
print(json.dumps({'status':'PASS','passed':len(checks),'report':str(report_path)}))
