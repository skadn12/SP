"""Prepare a bounded new Train-only replication; never select on scores."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import importlib.metadata
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent
V11 = WEEK / 'day6_diagnostic_reliability_v11'
OLD = WEEK / 'day5_training_v1/train_diagnosis_prepare'
D3 = WEEK / 'day3_solver187_v1/runtime/direct_v2'
TOKENIZER = Path('/root/autodl-tmp/Qwen3-4B')
SEED = 'postweek-train-replication-v1'

def read(p):
    return json.loads(p.read_text())

def binding(p):
    p = p.resolve()
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''):
            h.update(block)
    return {'path':str(p), 'sha256':h.hexdigest(), 'bytes':p.stat().st_size}

def save(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')

def v11_module():
    spec=importlib.util.spec_from_file_location('frozen_v11_prepare',V11/'prepare.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def prepare():
    assert not (HERE/'PROTOCOL.json').exists(), 'NO_OVERWRITE'
    selection=read(OLD/'SELECTION.json')
    d4=read(WEEK/'day4_schedule_v1/SELECTION.json')
    public={q['candidate_id']:q for q in read(D3/'PUBLIC_QUESTIONS.json')['questions']}
    candidates=[]
    root_inventory=[]
    for root in selection['root_hash_rank']:
        if root in selection['selected_roots']:
            continue
        eligible=[c for c in d4['selected_new_B'] if public[c]['root_id']==root and public[c]['question_type']=='ordering']
        eligible.sort(key=lambda c:hashlib.sha256((SEED+'|'+c).encode()).hexdigest())
        root_inventory.append({'root_id':root,'eligible_candidates_hash_order':eligible})
        if eligible and len(candidates)<6:
            candidates.append(eligible[0])
    assert len(candidates)==len({public[c]['root_id'] for c in candidates})==6
    protocol={
      'schema_version':'postweek_train_replication_protocol_v1',
      'created_at_utc':datetime.now(timezone.utc).isoformat(),
      'scientific_question':'In six additional Train roots, how often does B1 supply source-sufficient first-report citations yet still order incorrectly, and does this reproduce under two fixed display permutations?',
      'scope':'NEW_PROSPECTIVE_TRAIN_ONLY_DIAGNOSTIC_REPLICATION_NOT_OLD_DAY6_PROBE',
      'selection':'Use the existing score-independent day5 root_hash_rank, exclude all six earlier diagnosis roots. Select the first six roots with D4 selected_new_B ordering candidates. Within each root select lowest SHA256(postweek-train-replication-v1|candidate_id). No Solver output, correctness, TEST or DEV used.',
      'roots': [public[c]['root_id'] for c in candidates],
      'candidate_ids': candidates,
      'budget':{'models':['B1'],'roots':6,'questions':6,'fixed_variants':['perm_00','perm_01'],'requests':12,'max_new_requests':12,'automatic_retries':0,'additional_requests_if_insufficient':0},
      'previous_budget':'Day5/6 narrow control reserve remains NO_GO_NOT_RELEASED_NOT_EXECUTED at0/24; this separate12-call diagnosis budget does not release or consume that reserve.',
      'interface':'Exact frozen v11 make_prompt and schema functions; full source, same JSON evidence-turn task; no author anchors or Gold in model input.',
      'decoder':{'temperature':0.8,'top_p':0.9,'max_tokens':1024,'max_model_len':16384,'batch_size':4,'enable_thinking':False,'structured_outputs':'StructuredOutputsParams(json=schema)','prefix_caching':False},
      'execution_env':{'VLLM_WORKER_MULTIPROC_METHOD':'spawn','VLLM_USE_FLASHINFER_SAMPLER':'0'},
      'model':'Fixed B1 Day5 step56 merged endpoint; no training or checkpoint selection.',
      'training_exposure':'All six are OWN_ARM_TRAINED_QUESTION; source/family already exposed; diagnostic replication only, no heldout generalization.',
      'source_review':'Before inference read complete source, establish first semantic report anchors for all6 items and compare source-derived order to canonical Gold. Any disagreement or ambiguity stops all inference without replacement.',
      'format_gate':'All12 terminal records retained; mechanical schema/labels/unique full permutation/turn refs/finish_reason must pass before a response may qualify. Invalids remain failures of the interface, never salvage partial answers.',
      'citation_rubric':'Review full question semantics. Each cited set must contain the true first full report of the item, preserving speaker/action state/objects/qualifiers; local antecedent context may resolve identity but cannot supply an uncited missing predicate. Plans and suggestions cannot establish completion. Empty or later-only citations fail. Additional irrelevant refs do not invalidate a sufficient core citation.',
      'qualification':'For each root and mapping separately, qualify only if all6 item judgments are source-correct, all6 citation sets are source-sufficient for first report and final order is incorrect. A replicated root requires qualification on BOTH perm_00 and perm_01; one mapping is a candidate only. Count at most one conservative fact family per root.',
      'diagnostic_readout_gate':{'minimum_new_replicated_roots':3,'minimum_new_replicated_fact_families':3,'historical_root_pooling':False,'controlled_support':'NOT_EXECUTED; even meeting3/3 only allows preparation of a separately audited control experiment, not formal B feedback or training.'},
      'stop':'Score and source-audit this fixed12-call population once; report insufficient or sufficient diagnostic replication, with no automatic append, retry, prompt revision or training.',
      'competing_explanations':['citation retrieval or first-occurrence errors','speaker/action-state/compound-fact interpretation','display permutation sensitivity','stochastic output variation','sequence organization after source-sufficient citations'],
      'forbidden':['DEV/TEST error feedback','Gold or reviewer anchors supplied to Solver','silently overwrite prior sealed records','declare internal reasoning mechanism from visible evidence','causal feedback/memory benefit claim','release Challenger feedback or start second-stage training'],
      'runtime_versions':{n:importlib.metadata.version(n) for n in ('torch','transformers','tokenizers','vllm','jsonschema')},
    }
    save(HERE/'PROTOCOL.json',protocol)
    save(HERE/'SELECTION.json',{'selection_seed':SEED,'excluded_prior_roots':selection['selected_roots'],'root_inventory':root_inventory,'selected_candidates':candidates,'source_admission_failure_policy':'STOP_NO_REPLACEMENT','score_files_read':False})
    maps=read(D3/'DISPLAY_MAPPINGS.json')['requests']
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True)
    v11=v11_module()
    requests=[]
    questions=[]
    for c in candidates:
        q=public[c]
        source=read(Path(q['source_path']))
        assert binding(Path(q['source_path']))['sha256']==q['source_sha256']
        refs=[t['turn_ref'] for t in source['turns']]
        assert len(refs)==len(set(refs))
        questions.append(q)
        for i in range(2):
            variant=f'perm_{i:02d}'
            oldid=c+'__'+variant+'__direct__s00'
            m=maps[oldid]
            display={'items':m['items']}
            assert sorted(display['items'].values())==sorted(q['items'].values())
            cid_by_text={text:label for label,text in q['items'].items()}
            display_to_canonical={label:cid_by_text[text] for label,text in display['items'].items()}
            rid=c+'__'+variant+'__train_replication_v1'
            prompt=v11.make_prompt(tokenizer,q,display,source)
            pp=HERE/'prompts'/(rid+'.txt'); pp.parent.mkdir(exist_ok=True)
            with pp.open('x') as f:f.write(prompt)
            schema=v11.schema(list(display['items']),refs,'ordering')
            sp=HERE/'schemas'/(rid+'.json');save(sp,schema)
            ids=tokenizer.encode(prompt,add_special_tokens=False)
            assert len(ids)+1024<=16384,'NO_TRUNCATION'
            requests.append({'request_id':rid,'candidate_id':c,'root_id':q['root_id'],'fact_family_id':q['root_id']+':conservative_root_family','question_type':'ordering','question':q['question'],'variant_id':variant,'seed':2026102300+len(requests),'display':display,'display_to_canonical':display_to_canonical,'labels':list(display['items']),'prompt_path':str(pp),'prompt_sha256':binding(pp)['sha256'],'prompt_tokens':len(ids),'prompt_token_ids_sha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),'schema_path':str(sp),'schema_sha256':binding(sp)['sha256'],'source_path':q['source_path'],'source_sha256':q['source_sha256'],'source_turn_refs':refs,'training_exposure':'OWN_ARM_TRAINED_QUESTION','max_output_tokens':1024})
    save(HERE/'PUBLIC_QUESTIONS.json',{'questions':questions})
    save(HERE/'REQUESTS.json',{'status':'FIXED_BEFORE_NEW_SOLVER','requests':requests})
    save(HERE/'SOURCE_REVIEW_PACKET.json',{'scope':'BLIND_SOURCE_FIRST_ORDER_REVIEW_BEFORE_INFERENCE','questions':questions,'instructions':'Read complete source. Establish earliest full semantic report per canonical item, distinguish plans/requests/completed reports, document earlier near matches and scene anchors. Do not read upstream Gold or Solver outputs. Return {candidate_id,source_sha256,source_order_canonical,items:{I1:{first_turn_ref,context_turn_refs,reason,earlier_near_matches:[...]},...},ambiguity:false/true}. If item not uniquely grounded mark HOLD. Root merges independent source order against original Gold; no item replacement.'})
    print(json.dumps({'status':'PREPARED_NOT_FROZEN_NOT_EXECUTED','roots':protocol['roots'],'requests':len(requests),'max_prompt_tokens':max(r['prompt_tokens'] for r in requests),'input_token_budget':sum(r['prompt_tokens'] for r in requests)}))

def freeze():
    assert not (HERE/'PREPARATION_FREEZE.json').exists()
    admission=read(HERE/'SOURCE_ADMISSION.json'); assert admission['status']=='PASS'
    ep=read(OLD/'models/B1/EXECUTION_MANIFEST.json')
    endpoint={k:ep[k] for k in ('model_directory','tokenizer_directory','model_bindings','tokenizer_bindings','code_bindings','endpoint_evidence_bindings')}
    backend=Path('/root/autodl-tmp/gxy-selfplay/worktrees/h4-gate-downstream-v1/scripts/current_method/restart_n5_spell/vllm_backends.py')
    for p in (backend,backend.with_name('backends.py'),backend.parents[1]/'__init__.py',backend.parent/'__init__.py'):
        if not any(Path(row['path']).resolve()==p for row in endpoint['code_bindings']):endpoint['code_bindings'].append(binding(p))
    inputs=[OLD/'SELECTION.json',OLD/'POLICY.json',OLD/'models/B1/EXECUTION_MANIFEST.json',WEEK/'day4_schedule_v1/SELECTION.json',D3/'PUBLIC_QUESTIONS.json',D3/'DISPLAY_MAPPINGS.json',WEEK/'day3_solver187_v1/runtime/direct_v2_private/PRIVATE_QUESTIONS.json',V11/'prepare.py',WEEK/'day6_narrow_probe_gate_v1/DECISION.json']
    inputs += [Path(q['source_path']) for q in read(HERE/'PUBLIC_QUESTIONS.json')['questions']]
    artifacts=[p for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
    save(HERE/'PREPARATION_FREEZE.json',{'schema_version':'postweek_train_replication_freeze_v1','created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'INPUTS_FROZEN_NOT_EXECUTED','artifacts':[binding(p) for p in artifacts],'inputs':[binding(p) for p in inputs],'endpoint':endpoint,'runtime_versions':read(HERE/'PROTOCOL.json')['runtime_versions'],'python_prefix':'/root/autodl-tmp/gxy-selfplay/.venv-local4b','new_requests_budget':12,'old_probe_released':False,'dev_test_used':False,'score_files_read_for_selection':False})
    print('FROZEN:12 prospective Train-only B1 diagnostic requests')

if __name__=='__main__':
    import sys
    {'prepare':prepare,'freeze':freeze}[sys.argv[1]]()
