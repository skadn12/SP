"""Frozen 11-question paired diagnostic interface study; no retries/truncation.

prepare/verify are CPU only. live requires independent input-bound preflight.
score reads private Gold only after all 22 terminal raw records are frozen.
Source semantic support is exported for review, never inferred from quote match.
V2 excludes private bytes from all prepare/verify/live input bindings. V1 is
retained after public preflight rejection; private verification is score-only.
"""
from __future__ import annotations
import argparse
import collections
import importlib.util
import json
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HOME = HERE.parent
WEEK = HOME.parent
OLD = WEEK / 'day3_feedback_realization_v1/runtime/eval_v1'
spec = importlib.util.spec_from_file_location('paired_previous_evaluator', OLD.parent / 'evaluate.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
c, evidence, legacy_score = previous.c, previous.evidence, previous.legacy_score
save = previous.save_new
ARMS = ('original', 'simplified')
SCHEDULE_SEED = 2026102103
SIMPLIFIED_TASK = (
    'For every displayed label, provide exactly one entry in the checks object, '
    'using that displayed label as its key. For multi_select, judge whether the '
    'displayed statement answers the precise question, rather than merely being '
    'related to the dialogue. Use judgment supported, contradicted, not_supported, '
    'or ambiguous. For ordering, cite the actual occurrence or explicit order '
    'relation used for that item. Each check must contain exactly judgment, '
    'turn_ref, quote, and reason. Give one short exact quotation of at most 200 '
    'characters and its turn_ref as printed above. If no source support exists, '
    'set both turn_ref and quote to empty strings; otherwise both must be nonempty. '
    'Give a brief reason of 1 to 160 characters, containing only the relevant '
    'evidence link or short calculation. Return only one JSON object with exactly '
    'checks and final_answer. checks must contain every displayed label exactly '
    'once and no other key. final_answer is an array of displayed labels: select '
    'all answers for multi_select, or list every item exactly once in the requested '
    'order for ordering.'
)

def code_paths():
    return sorted(set(previous.code_paths()) | {Path(__file__).resolve()})

def protocol_check(p):
    c.require(p['namespace'] == 'train.feedback.minimal_repair_control.v1', 'NAMESPACE')
    s, v = p['interface_study'], p['solver']
    c.require(s['questions'] == 11 and s['arms'] == list(ARMS) and s['total_calls'] == 22 and s['calls_each_arm'] == 11, 'FIXED_11_PAIRED')
    c.require(s['max_output_tokens_both'] == 3072, 'BOTH3072')
    c.require(v == {'name':'S0_M_clean','model':str(c.MODEL),'tokenizer':str(c.TOKENIZER),'temperature':.8,'top_p':.9,'enable_thinking':False,'max_model_len':16384,'batch_size':8,'gpu_memory_utilization':.8,'prefix_caching':False}, 'FIXED_SOLVER')
    c.require(p['new_roots'] == 0 and p['new_source_turns'] == 0 and p['limits']['max_solver_calls'] == 29 and p['limits']['automatic_retry'] is False and p['limits']['source_truncation'] is False, 'NO_RETRY_NO_EXTENSION')
    c.require(p['limits']['new_training_runs'] == 0 and p['limits']['dev_test_calls'] == 0, 'TRAIN_ONLY_NO_TRAINING')

def schema(base):
    labels = list(base['display']['options' if base['question_type'] == 'multi_select' else 'items'])
    check = {'type':'object','additionalProperties':False,'required':['judgment','turn_ref','quote','reason'],'properties':{
        'judgment':{'type':'string','enum':list(evidence.STATUS)},
        'turn_ref':{'type':'string'},'quote':{'type':'string','maxLength':200},
        'reason':{'type':'string','minLength':1,'maxLength':160}},
        'oneOf':[{'properties':{'turn_ref':{'const':''},'quote':{'const':''}}},
                 {'properties':{'turn_ref':{'minLength':1},'quote':{'minLength':1}}}]}
    return {'type':'object','additionalProperties':False,'required':['checks','final_answer'],'properties':{
        'checks':{'type':'object','additionalProperties':False,'required':labels,'properties':{label:check for label in labels}},
        'final_answer':base['json_schema']['properties']['final_answer']}}

def sampling(row):
    from vllm import SamplingParams
    return SamplingParams(temperature=.8, top_p=.9, max_tokens=3072, seed=row['seed'])

def input_bindings(p):
    s = p['interface_study']
    return [p['consumed_plan'], p['feedback_snapshot'], p['prior_manifest'],
            s['old_public'], s['old_manifest'], s['old_mappings']]

def build(p, tokenizer):
    protocol_check(p); c.verify_bindings(input_bindings(p))
    old = c.read(OLD / 'manifest.json'); public = c.read(OLD / 'PUBLIC_QUESTIONS.json')
    maps = c.read(OLD / 'DISPLAY_MAPPINGS.json'); oldtokens = c.read(OLD / 'PROMPT_TOKEN_IDS.json')
    bases = [r for r in old['requests'] if r['branch'] == 'evidence']
    c.require(len(bases) == len({r['candidate_id'] for r in bases}) == 11, 'ALL11_ORIGINAL_DIAGNOSTICS')
    c.require({r['candidate_id'] for r in bases} == {q['candidate_id'] for q in public['questions']}, 'ALL_ACCEPTED_NO_HOLD')
    c.require(collections.Counter(r['question_type'] for r in bases) == {'multi_select':7,'ordering':4}, 'OLD_TYPE_COUNTS')
    qby = {q['candidate_id']:q for q in public['questions']}
    rows, prompts, tokens, newmaps, sources, oldbindings = [], {}, {}, {}, {}, []
    for base in bases:
        q = qby[base['candidate_id']]; c.a.public_question(q)
        c.require(base['variant_id'] == 'perm_00' and base['slot'] == 0 and base['max_output_tokens'] == 3072, 'FIXED_BASE_DISPLAY_SAMPLE')
        srcpath = OLD / 'sources' / (base['root_id'] + '.json')
        source = c.rt.source_check(dict(q, source_path=str(srcpath)))
        sources[base['root_id']] = srcpath
        oldpath = OLD / 'prompts' / (base['request_id'] + '.txt')
        raw = oldpath.read_bytes(); original = raw.decode('utf-8')
        c.require(c.rt.textsha(original) == base['prompt_sha256'], 'OLD_PROMPT_HASH')
        c.require(original.count(evidence.TASK) == 1, 'ONE_TASK_SUFFIX')
        start = original.index(evidence.TASK); tail = original[start + len(evidence.TASK):]
        c.require(tail.startswith('<|im_end|>'), 'TASK_ONLY_AT_USER_END')
        simplified = original[:start] + SIMPLIFIED_TASK + tail
        c.require(original == evidence.prompt(tokenizer, q, maps['requests'][base['request_id']], source), 'OLD_BYTE_PROMPT_RECONSTRUCTION')
        oldbindings.extend([c.bind(oldpath),c.bind(srcpath)])
        for arm, text in [('original',original),('simplified',simplified)]:
            rid = base['request_id'] + '__' + arm
            ids = tokenizer.encode(text, add_special_tokens=False)
            c.require(len(ids) + 3072 <= 16384, 'FULL_CONTEXT_OVERFLOW_NO_TRUNCATION:' + rid)
            if arm == 'original':c.require(ids == oldtokens[base['request_id']] and text.encode('utf-8') == raw, 'ORIGINAL_BYTES_TOKENS_IDENTICAL')
            c.require(all(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" in text for t in source['turns']), 'FULL_SOURCE')
            row = {k:base[k] for k in ('candidate_id','root_id','question_type','variant_id','display','slot','seed','source_turns','source_sha256','direct_request_id')}
            row.update(request_id=rid, branch='paired_interface', arm=arm, original_request_id=base['request_id'], max_output_tokens=3072,
                       prompt_sha256=c.rt.textsha(text), prompt_tokens=len(ids), shared_prefix_sha256=c.rt.textsha(original[:start]),
                       shared_template_suffix_sha256=c.rt.textsha(tail),json_schema=base['json_schema'] if arm == 'original' else schema(base))
            rows.append(row);prompts[rid]=text;tokens[rid]=ids;newmaps[rid]=maps['requests'][base['request_id']]
    random.Random(SCHEDULE_SEED).shuffle(rows)
    for index,row in enumerate(rows):row.update(schedule_index=index,batch_id=f'batch{index//8:04d}')
    c.require(len(rows) == 22 and collections.Counter(r['arm'] for r in rows) == {'original':11,'simplified':11}, 'EXACT22')
    for offset in range(0,22,8):c.require({r['arm'] for r in rows[offset:offset+8]} == set(ARMS), 'EACH_BATCH_MIXED')
    return rows,prompts,tokens,{'requests':newmaps,'public_to_canonical':maps['public_to_canonical']},sources,public,oldbindings

def prepare(run, freeze):
    c.environment();run=Path(run).resolve();freeze=Path(freeze).resolve()
    c.require(c.below(run,HERE) and not run.exists(), 'NEW_LOCAL_RUN_NO_OVERWRITE')
    p=c.read(freeze);protocol_check(p)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(c.TOKENIZER),local_files_only=True)
    rows,prompts,tokens,maps,sources,public,oldbindings=build(p,tok)
    for row in rows:
        actual=sampling(row);c.require(getattr(actual,'structured_outputs',None) is None and getattr(actual,'guided_decoding',None) is None and not actual.stop,'NO_GRAMMAR_CHOICE_STOP')
    models=c.rt.models();run.mkdir(parents=True)
    for name,obj in [('PUBLIC_QUESTIONS.json',public),('PROTOCOL.json',p),('DISPLAY_MAPPINGS.json',maps),('PROMPT_TOKEN_IDS.json',tokens),('REQUEST_SCHEDULE.json',{'randomization_seed':SCHEDULE_SEED,'batch_size':8,'request_ids':[r['request_id'] for r in rows]}),('OUTPUT_SCHEMA_POLICY.json',{'policy':'posthoc_only_no_constraint','original_task':evidence.TASK,'simplified_task':SIMPLIFIED_TASK,'system':evidence.SYSTEM,'schemas':{r['request_id']:r['json_schema'] for r in rows}})]:save(run/name,obj)
    for root,path in sources.items():
        target=run/'sources'/(root+'.json');target.parent.mkdir(exist_ok=True);shutil.copy2(path,target);c.require(c.sha(path)==c.sha(target),'SOURCE_COPY')
    for rid,text in prompts.items():
        path=run/'prompts'/(rid+'.txt');path.parent.mkdir(exist_ok=True)
        with path.open('xb') as f:f.write(text.encode('utf-8'))
    limits={arm:{'min_prompt_tokens':min(r['prompt_tokens'] for r in rows if r['arm']==arm),'max_prompt_tokens':max(r['prompt_tokens'] for r in rows if r['arm']==arm),'min_context_headroom':min(16384-r['prompt_tokens']-3072 for r in rows if r['arm']==arm)} for arm in ARMS}
    m={'schema_version':'paired_interface_manifest_v1','status':'PREPARED_AWAITING_INDEPENDENT_PREFLIGHT','created_at_utc':c.now(),'namespace':p['namespace'],'question_count':11,'request_count':22,'request_counts':dict(collections.Counter(r['arm'] for r in rows)),'requests':rows,'solver':p['solver'],'model':str(c.MODEL),'tokenizer':str(c.TOKENIZER),'model_bindings':models,'code_bindings':[c.bind(x) for x in code_paths()],'protocol_binding':c.bind(freeze),'historical_input_bindings':input_bindings(p)+oldbindings,'public_input_bindings':[c.bind(x) for x in sorted(run.rglob('*')) if x.is_file()],'versions':c.rt.versions(),'python_prefix':sys.prefix,'limits':limits,'randomization_seed':SCHEDULE_SEED,'sampling':'both arms temperature0.8/top_p0.9/max_tokens3072; equal original per-question seed','constrained_decoding':False,'source_truncation':False,'automatic_retry':False,'gold_parsed_by_prepare':False,'old_private_binding':p['interface_study']['old_private'],'training_started':False}
    save(run/'manifest.json',m);save(run/'PREFLIGHT_FREEZE.json',{'status':m['status'],'created_at_utc':c.now(),'files':[c.bind(x) for x in sorted(run.rglob('*')) if x.is_file()],'planned_requests':22,'inference_started':False,'gold_json_parsed':False})
    print(json.dumps({'status':m['status'],'calls':22,'limits':limits,'gpu_started':False}))

def verify(run,audit=None,models=True):
    c.environment();run=Path(run).resolve();f=c.read(run/'PREFLIGHT_FREEZE.json')
    c.require(f['status']=='PREPARED_AWAITING_INDEPENDENT_PREFLIGHT' and f['planned_requests']==22,'INPUT_FREEZE')
    c.require(all(c.below(x['path'],run) for x in f['files']),'PUBLIC_INPUTS_LOCAL');c.verify_bindings(f['files'])
    m=c.read(run/'manifest.json');p=c.read(run/'PROTOCOL.json');protocol_check(p)
    c.verify_bindings(m['code_bindings']+m['historical_input_bindings']+m['public_input_bindings']+[m['protocol_binding']])
    if models:c.verify_bindings(m['model_bindings'])
    c.require(m['versions']==c.rt.versions() and m['python_prefix']==sys.prefix,'ENVIRONMENT_DRIFT')
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(c.TOKENIZER),local_files_only=True)
    rows,prompts,tokens,maps,sources,public,unused=build(p,tok)
    c.require(rows==m['requests'] and maps==c.read(run/'DISPLAY_MAPPINGS.json') and tokens==c.read(run/'PROMPT_TOKEN_IDS.json') and public==c.read(run/'PUBLIC_QUESTIONS.json'),'PLAN_RECONSTRUCTION')
    c.require(c.read(run/'REQUEST_SCHEDULE.json')=={'randomization_seed':SCHEDULE_SEED,'batch_size':8,'request_ids':[r['request_id'] for r in rows]},'SCHEDULE_RECONSTRUCTION')
    for rid,text in prompts.items():c.require((run/'prompts'/(rid+'.txt')).read_bytes()==text.encode('utf-8'),'PROMPT_BYTE_RECONSTRUCTION')
    for root,path in sources.items():c.require(c.sha(path)==c.sha(run/'sources'/(root+'.json')),'SOURCE_BYTE_RECONSTRUCTION')
    for row in rows:
        actual=sampling(row);c.require(getattr(actual,'structured_outputs',None) is None and getattr(actual,'guided_decoding',None) is None and not actual.stop,'NO_CONSTRAINTS')
    if audit:
        report=c.read(audit);bs=report.get('audited_files') or report.get('files') or []
        c.require(report['status']=='PASS' and any(Path(x['path']).resolve()==run/'PREFLIGHT_FREEZE.json' and x['sha256']==c.sha(run/'PREFLIGHT_FREEZE.json') for x in bs),'AUDIT_BOUND_TO_INPUT_FREEZE');c.verify_bindings(bs)
    return m

def terminal(row,status,**extra):
    return {k:row[k] for k in ('request_id','candidate_id','root_id','question_type','branch','arm','variant_id','slot','seed','prompt_sha256','original_request_id','schedule_index','batch_id')}|{'status':status,'raw_output':None,'response_committed_at_utc':c.now(),'model_path':str(c.MODEL),'gold_read_by_solver':False,'automatic_retry':False,'source_truncation':False,**extra}

def live(run,audit):
    run=Path(run).resolve();out=run/'live';c.require(not out.exists(),'NO_SECOND_START_OR_RETRY');m=verify(run,audit);rows=m['requests'];out.mkdir()
    save(out/'STARTED.json',{'started_at_utc':c.now(),'manifest':c.bind(run/'manifest.json'),'audit':c.bind(audit),'gold_opened':False,'max_calls':22})
    backend=None;submitted=set();done=set();failure=None;started=time.perf_counter();load_seconds=None
    try:
        from scripts.current_method.restart_n5_spell.vllm_backends import VLLMSolverBackend
        _,native=c.a.native();native.ensure_vllm_environment()
        backend=VLLMSolverBackend(model_path=c.MODEL,tokenizer_path=c.TOKENIZER,max_model_len=16384,temperature=.8,top_p=.9,max_tokens=3072,gpu_memory_utilization=.8)
        load_seconds=backend.load_latency_seconds;c.require(str(backend.model_identifier)==m['model'],'BACKEND_MODEL')
        tokens=c.read(run/'PROMPT_TOKEN_IDS.json')
        for offset in range(0,22,8):
            batch=rows[offset:offset+8];bid=f'batch{offset//8:04d}';prompts=[(run/'prompts'/(r['request_id']+'.txt')).read_text() for r in batch];params=[sampling(r) for r in batch]
            for row,text in zip(batch,prompts):c.require(c.rt.textsha(text)==row['prompt_sha256'],'PROMPT_CHANGED')
            at=c.now();bt=time.perf_counter();save(out/'batches'/(bid+'_SUBMITTED.json'),{'submitted_at_utc':at,'request_ids':[r['request_id'] for r in batch],'arms':[r['arm'] for r in batch],'sampling_params':[repr(x) for x in params],'automatic_retry':False});submitted.update(r['request_id'] for r in batch)
            outputs=backend.engine.generate(prompts,params,use_tqdm=False)
            for index,row in enumerate(batch):
                if index>=len(outputs):record=terminal(row,'BACKEND_ERROR',error='MISSING_OUTPUT',usage=None,usage_status='UNKNOWN_NOT_ZERO')
                else:
                    output=outputs[index];comps=list(output.outputs);primary=comps[0] if comps else None;errors=[]
                    if output.prompt!=prompts[index] or list(output.prompt_token_ids)!=tokens[row['request_id']]:errors.append('PROMPT_OR_TOKEN_DRIFT')
                    if len(comps)!=1:errors.append('COMPLETION_COUNT_MISMATCH')
                    record=terminal(row,'RETURNED',raw_output=primary.text if primary else None,request_submitted_at_utc=at,prompt_token_ids=list(output.prompt_token_ids),completion_token_ids=list(primary.token_ids) if primary else None,output_validation_errors=errors,all_completions=[{'text':x.text,'token_ids':list(x.token_ids),'finish_reason':x.finish_reason,'stop_reason':x.stop_reason} for x in comps],metadata={'prompt_token_count':len(output.prompt_token_ids),'completion_token_count':len(primary.token_ids) if primary else None,'finish_reason':primary.finish_reason if primary else None,'stop_reason':primary.stop_reason if primary else None,'temperature':.8,'top_p':.9,'max_tokens':3072,'enable_thinking':False,'structured_output_api':None,'constrained_decoding':False,'backend_request_id':output.request_id,'cached_tokens':getattr(output,'num_cached_tokens',None)})
                save(out/'responses'/(row['request_id']+'.json'),record);done.add(row['request_id'])
            if len(outputs)>len(batch):save(out/'batches'/(bid+'_EXTRA_OUTPUTS.json'),{'extra_outputs':[{'request_id':x.request_id,'prompt':x.prompt,'prompt_token_ids':list(x.prompt_token_ids),'completions':[{'text':y.text,'token_ids':list(y.token_ids),'finish_reason':y.finish_reason,'stop_reason':y.stop_reason} for y in x.outputs]} for x in outputs[len(batch):]]})
            c.require(len(outputs)==len(batch),'OUTPUT_COUNT_MISMATCH');save(out/'batches'/(bid+'_COMPLETE.json'),{'completed_at_utc':c.now(),'request_ids':[r['request_id'] for r in batch],'wall_seconds':time.perf_counter()-bt});print(json.dumps({'completed':len(done),'planned':22}),flush=True)
    except BaseException as error:failure={'error':str(error),'exception_type':type(error).__name__,'traceback':traceback.format_exc()}
    finally:
        if backend:
            try:backend.close()
            except BaseException as error:failure=failure or {'error':str(error),'exception_type':type(error).__name__,'traceback':traceback.format_exc()}
        if failure:
            save(out/'FAILED.json',{'status':'FAILED','failed_at_utc':c.now(),'actual_submitted_requests':len(submitted),'returned_requests':len(done),'wall_seconds':time.perf_counter()-started,'model_load_wall_seconds':load_seconds,**failure})
            for row in rows:
                if row['request_id'] not in done:save(out/'responses'/(row['request_id']+'.json'),terminal(row,'BACKEND_ERROR' if row['request_id'] in submitted else 'NOT_SUBMITTED',usage=None,usage_status='UNKNOWN_NOT_ZERO' if row['request_id'] in submitted else 'NO_REQUEST_SENT',**failure))
        else:save(out/'COMPLETE.json',{'status':'COMPLETE','completed_at_utc':c.now(),'actual_submitted_requests':len(submitted),'returned_requests':len(done),'wall_seconds':time.perf_counter()-started,'model_load_wall_seconds':load_seconds,'gold_opened':False})
        paths=[out/'responses'/(r['request_id']+'.json') for r in rows];c.require(all(x.is_file() for x in paths) and set((out/'responses').glob('*.json'))==set(paths),'EXACT22_TERMINAL_RECORDS')
        ledger=[x for x in sorted(out.rglob('*')) if x.is_file()]
        save(out/'RAW_RESPONSE_FREEZE.json',{'status':'COMMITTED','created_at_utc':c.now(),'planned_request_count':22,'manifest_binding':c.bind(run/'manifest.json'),'preflight_binding':c.bind(run/'PREFLIGHT_FREEZE.json'),'files':[c.bind(x) for x in paths],'execution_ledger_bindings':[c.bind(x) for x in ledger],'gold_opened_by_live':False,'automatic_retry':False,'source_truncation':False})
    if failure:raise RuntimeError('LIVE_FAILED_FIXED_DENOMINATOR:'+failure['error'])

def normal_items(arm,parsed):
    if not isinstance(parsed,dict):return []
    if arm=='original':return parsed.get('item_evidence') if isinstance(parsed.get('item_evidence'),list) else []
    checks=parsed.get('checks')
    if not isinstance(checks,dict):return []
    return [{'label':label,'judgment':x.get('judgment'),'reason':x.get('reason'),'citations':([{'turn_ref':x.get('turn_ref'),'quote':x.get('quote')}] if x.get('turn_ref') or x.get('quote') else [])} for label,x in checks.items() if isinstance(x,dict)]

def parse_one(row,raw,source):
    import jsonschema
    errors=[];parsed=None;truncated=raw.get('metadata',{}).get('finish_reason')=='length'
    if raw.get('status')!='RETURNED' or not isinstance(raw.get('raw_output'),str):errors.append('NO_RETURNED_TEXT')
    else:
        try:
            parsed=json.loads(raw['raw_output'].strip());c.require(isinstance(parsed,dict),'JSON_OBJECT_REQUIRED')
        except Exception as error:errors.append(type(error).__name__+':'+str(error)[:1500])
    labels=list(row['display']['options' if row['question_type']=='multi_select' else 'items'])
    items=normal_items(row['arm'],parsed);seen=[x.get('label') for x in items if isinstance(x,dict)]
    safe_seen=[x for x in seen if isinstance(x,str)]
    covered=set(safe_seen)&set(labels);coverage={'expected_items':len(labels),'observed_entries':len(items),'covered_displayed_labels':len(covered),'missing_labels':[x for x in labels if x not in covered],'duplicate_labels':[x for x in set(safe_seen) if safe_seen.count(x)>1],'extra_labels':[x for x in set(safe_seen) if x not in labels],'complete_unique':len(seen)==len(safe_seen)==len(labels) and set(safe_seen)==set(labels),'status':'STRUCTURAL_ONLY_NO_INVALID_SCORE_RESCUE'}
    if parsed is not None:
        try:
            jsonschema.validate(parsed,row['json_schema']);c.require(coverage['complete_unique'],'DUPLICATE_OR_MISSING_ITEM_EVIDENCE')
            if row['arm']=='simplified':c.require(all(bool(x['turn_ref'])==bool(x['quote']) for x in parsed['checks'].values()),'CITATION_FIELDS_BOTH_EMPTY_OR_PRESENT')
        except Exception as error:errors.append(type(error).__name__+':'+str(error)[:1500])
    if truncated:errors.append('OUTPUT_TRUNCATED')
    errors+=raw.get('output_validation_errors') or [];valid=not errors;turns={t['turn_ref']:t['text'] for t in source['turns']};fidelity=[];normalized=[]
    if valid:
        for item in items:
            citations=[]
            for cite in item['citations']:
                exact=cite['turn_ref'] in turns and cite['quote'] in turns[cite['turn_ref']]
                check={'label':item['label'],**cite,'turn_ref_exists':cite['turn_ref'] in turns,'exact_substring':exact};fidelity.append(check);citations.append(check)
            normalized.append({'label':item['label'],'judgment':item['judgment'],'reason':item['reason'],'citation_checks':citations,'has_citation':bool(citations),'has_exact_citation':any(x['exact_substring'] for x in citations),'all_citations_exact':bool(citations) and all(x['exact_substring'] for x in citations),'semantic_source_support':'PENDING','target_error':'PENDING'})
    return {'format_valid':valid,'invalid_reasons':errors,'truncated':truncated,'parsed_response':parsed,'item_coverage':coverage,'normalized_items':normalized,'citation_checks':fidelity,'citation_count':len(fidelity),'exact_quote_count':sum(x['exact_substring'] for x in fidelity),'items_with_exact_citation':sum(x['has_exact_citation'] for x in normalized),'semantic_review':'PENDING' if valid else 'NOT_REVIEWABLE','invalid_records_retained':True,'unverified_rationale_training_target':False}

def score(run):
    run=Path(run).resolve();m=verify(run,models=False);out=run/'live'
    c.require(not (run/'SCORES.json').exists() and not (run/'SCORING_STARTED.json').exists(),'NO_SCORE_OVERWRITE')
    f=c.read(out/'RAW_RESPONSE_FREEZE.json');c.require(f['status']=='COMMITTED' and f['planned_request_count']==22,'RAW_FREEZE_BEFORE_PRIVATE_JSON')
    expected={out/'responses'/(r['request_id']+'.json') for r in m['requests']}
    c.require({Path(x['path']).resolve() for x in f['files']}==expected and set((out/'responses').glob('*.json'))==expected,'FIXED_RAW_DENOMINATOR')
    c.verify_bindings(f['files']+f['execution_ledger_bindings']+[f['manifest_binding'],f['preflight_binding']])
    c.verify_bindings([m['old_private_binding']]);save(run/'SCORING_STARTED.json',{'gold_first_read_at_utc':c.now(),'raw_freeze_binding':c.bind(out/'RAW_RESPONSE_FREEZE.json'),'private_binding':m['old_private_binding'],'scope':'First private JSON read by this scoring command after22 raw terminal records frozen; prior author/reviewer exposure disclosed, no technical blinding.'})
    golds={q['candidate_id']:q for q in c.read(m['old_private_binding']['path'])['questions']};public=c.read(run/'PUBLIC_QUESTIONS.json')['questions'];qby={q['candidate_id']:q for q in public};maps=c.read(run/'DISPLAY_MAPPINGS.json')
    c.require(set(golds)==set(qby),'GOLD_PUBLIC_SET')
    for q in public:c.require({k:golds[q['candidate_id']][k] for k in c.a.PUBLIC_KEYS if k in golds[q['candidate_id']]}==q,'PRIVATE_PUBLIC_CONTENT')
    records=[];raws={}
    for row in m['requests']:
        rid=row['request_id'];path=out/'responses'/(rid+'.json');raw=c.read(path);raws[rid]=raw
        c.require(all(raw[k]==row[k] for k in ('request_id','candidate_id','root_id','question_type','arm','variant_id','slot','seed','prompt_sha256','schedule_index','batch_id')),'RAW_IDENTITY')
        c.require(c.timestamp(raw['response_committed_at_utc'])<=c.timestamp(f['created_at_utc']),'RAW_AFTER_FREEZE')
        rec={k:row[k] for k in ('request_id','candidate_id','root_id','question_type','arm','seed','original_request_id')}
        rec.update(parse_one(row,raw,c.read(run/'sources'/(row['root_id']+'.json'))));valid=rec['format_valid'];canonical=maps['public_to_canonical'][row['candidate_id']];reverse={v:k for k,v in canonical.items()};variant=maps['requests'][rid];gold=golds[row['candidate_id']]['gold']
        c.require(len(reverse)==len(canonical) and set(gold)<=set(canonical),'UNAMBIGUOUS_GOLD_DOMAIN')
        dp=rec['parsed_response']['final_answer'] if valid else None;cp=[reverse[variant['display_to_content'][x]] for x in dp] if valid else None
        rec.update(raw_response_binding=c.bind(path),display_prediction=dp,canonical_prediction=cp,canonical_gold=gold,exact=bool(valid and (set(cp)==set(gold) if row['question_type']=='multi_select' else cp==gold)))
        records.append(rec)
    arms={}
    for arm in ARMS:
        rs=[r for r in records if r['arm']==arm];expected_items=sum(x['item_coverage']['expected_items'] for x in rs);valid_items=sum(len(x['normalized_items']) for x in rs);citations=sum(x['citation_count'] for x in rs);exact_quotes=sum(x['exact_quote_count'] for x in rs)
        arms[arm]={'calls':11,'strict_valid':sum(x['format_valid'] for x in rs),'invalid':sum(not x['format_valid'] for x in rs),'truncated':sum(x['truncated'] for x in rs),'exact':sum(x['exact'] for x in rs),'exact_rate_invalid_retained':sum(x['exact'] for x in rs)/11,'expected_items':expected_items,'structurally_covered_items_including_invalid':sum(x['item_coverage']['covered_displayed_labels'] for x in rs),'responses_complete_unique_coverage':sum(x['item_coverage']['complete_unique'] for x in rs),'strict_reviewable_items':valid_items,'strict_reviewable_item_coverage':valid_items/expected_items,'items_with_exact_citation':sum(x['items_with_exact_citation'] for x in rs),'items_with_exact_citation_over_expected_items':sum(x['items_with_exact_citation'] for x in rs)/expected_items,'citation_count':citations,'exact_quote_count':exact_quotes,'citation_exact_rate':exact_quotes/citations if citations else None,'response_with_all_citations_exact':sum(x['format_valid'] and bool(x['citation_checks']) and all(y['exact_substring'] for y in x['citation_checks']) for x in rs),'semantic_source_support':'PENDING_SEPARATE_SOURCE_REVIEW','source_faithful_reviewable_coverage':None,'cost':legacy_score.cost(out,[x for x in m['requests'] if x['arm']==arm],{k:v for k,v in raws.items() if v['arm']==arm})}
    pairs=[]
    for q in public:
        a={arm:next(x for x in records if x['candidate_id']==q['candidate_id'] and x['arm']==arm) for arm in ARMS};pairs.append({'candidate_id':q['candidate_id'],'question_type':q['question_type'],'original_request_id':a['original']['request_id'],'simplified_request_id':a['simplified']['request_id'],'original_valid':a['original']['format_valid'],'simplified_valid':a['simplified']['format_valid'],'original_exact':a['original']['exact'],'simplified_exact':a['simplified']['exact'],'paired_same_seed':a['original']['seed']==a['simplified']['seed']})
    result={'schema_version':'paired_interface_scores_v1','status':'SCORED_FIXED22_SEMANTIC_REVIEW_PENDING','created_at_utc':c.now(),'manifest_binding':c.bind(run/'manifest.json'),'raw_freeze_binding':c.bind(out/'RAW_RESPONSE_FREEZE.json'),'private_binding':m['old_private_binding'],'request_count':22,'arms':arms,'pairs':pairs,'responses':records,'total_cost':legacy_score.cost(out,m['requests'],raws),'semantic_review_complete':False,'adoption_decision':'PENDING_SOURCE_SEMANTIC_REVIEW','invalid_policy':'All22 retained; invalid exact zero, no final answer extraction rescue; structural coverage on parseable invalid objects is descriptive only.','citation_denominator_policy':'Raw citation counts may differ because original permits0..2 and simplified0..1. Compare common response and expected-item denominators as well; exact quotation is not semantic correctness.','cost_policy':'Per-arm token sums are separable; shared batch/live/model-load wall times are repeated context and must not be summed across arms.','scope':'Exploratory paired11 one sample per arm; interface package effect, not isolated component attribution, feedback/no-feedback AB, or training result.','training_started':False,'new_training_runs':0,'dev_test_calls':0,'formal_reproduced':False,'matched_feedback_AB':False,'faithful_hidden_reasoning_claim':False}
    save(run/'SCORES.json',result);save(run/'SEMANTIC_REVIEW_PACKET.json',{'status':'PENDING_FULL_SOURCE_REVIEW','responses':records,'source_bindings':[c.bind(x) for x in sorted((run/'sources').glob('*.json'))],'raw_freeze_binding':c.bind(out/'RAW_RESPONSE_FREEZE.json'),'common_review_dimensions':['source truth of judgment','reason and calculation correctness','exact quotation at claimed turn','source supports stated inference','final answer alignment','target error class'],'review_granularity':['displayed item','response'],'invalid_response_policy':'NOT_REVIEWABLE; retain all expected items in total-denominator coverage'})
    print(json.dumps({'status':result['status'],'request_count':22,'arms':arms,'total_cost':result['total_cost']}))

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('prepare','verify','live','score'));ap.add_argument('--run',type=Path,default=HERE/'interface_v2');ap.add_argument('--freeze',type=Path,default=HOME/'GENERATION_FREEZE.json');ap.add_argument('--audit',type=Path);args=ap.parse_args()
    if args.command=='prepare':prepare(args.run,args.freeze)
    elif args.command=='verify':m=verify(args.run,args.audit);print(json.dumps({'status':'PASS','calls':m['request_count'],'limits':m['limits'],'gpu_started':False}))
    elif args.command=='live':c.require(args.audit,'INDEPENDENT_PREFLIGHT_REQUIRED');live(args.run,args.audit)
    else:score(args.run)
if __name__=='__main__':main()
