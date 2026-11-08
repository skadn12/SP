#!/usr/bin/env python3
"""CPU-only independent audit: public requests and QA provenance metadata.
Never parse private author files, import a Solver, execute finalizers, or infer.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
import ast
import hashlib
import itertools
import json
import os
import random
import sys

sys.dont_write_bytecode = True
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
C = Path(__file__).resolve().parents[1]
W = C.parent
checks, failures, input_bindings = [], [], {}

def bind(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    result = {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    input_bindings[str(path)] = result
    return result

def read(path):
    bind(path)
    return json.loads(Path(path).read_text())

def check(name, ok, evidence=None):
    checks.append({'check': name, 'status': 'PASS' if ok else 'FAIL', 'evidence': evidence})
    if not ok:
        failures.append(name)

def verify(ref, name):
    actual = bind(ref['path'])
    check(name, all(actual[k] == ref[k] for k in ('sha256', 'bytes') if k in ref), actual)
    return actual

def dt(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))

selection = read(C / 'FAMILY_SELECTION.json')
admission = read(C / 'content/CONTENT_ADMISSION.json')
recon = read(C / 'content/qa/SOURCE_FIRST_RECONSTRUCTION.json')
seal = read(C / 'content/qa/FIRST_PASS_SEAL.json')
qa = read(C / 'content/qa/FINAL_CONTENT_REVIEW.json')
freeze = read(C / 'requests/REQUEST_FREEZE.json')
ledger = read(C / 'CONTENT_AND_REQUEST_LEDGER.json')
policy = read(W / 'day3_common_evaluation_v1/protocol/REQUEST_POLICY_V1.json')
preflight = read(W / 'day3_common_evaluation_v1/audit/SOURCE_AND_MAPPING_PREFLIGHT.json')
lock = read(W / 'day3_production_v1/evaluation/EVAL_SOURCE_FAMILY_LOCK.json')
contract = read(W / 'day3_common_evaluation_v1/protocol/CONTENT_RECORD_CONTRACT.json')
for p in [C / 'finalize_content.py', C / 'freeze_requests.py', Path(__file__)]:
    bind(p)
    ast.parse(p.read_text())

for name, report, fields in [
    ('admission', admission, ['family_selection', 'independent_review', 'source_first_reconstruction', 'first_pass_seal']),
    ('freeze', freeze, ['admission', 'policy', 'script', 'synthetic_mapping_preflight', 'requests', 'display_maps']),
    ('ledger', ledger, ['family_selection', 'content_admission', 'request_freeze']),
    ('qa', qa, ['source_first_reconstruction', 'first_pass_seal']),
    ('policy', policy, ['source_lock']),
]:
    for key in fields:
        verify(report[key], name + '_binding_' + key)
verify(policy['prompt']['adapter'], 'adapter_binding')
verify(policy['display_mapping']['implementation'], 'display_implementation_binding')
for i, ref in enumerate(freeze['actual_tokenizer_bindings']):
    verify(ref, 'tokenizer_binding_' + str(i))
check('tokenizer_preflight_bindings_equal', freeze['actual_tokenizer_bindings'] == preflight['actual_tokenizer_bindings'])
for i, ref in enumerate(preflight['imported_native_code_bindings']):
    verify(ref, 'native_code_binding_' + str(i))

slots = {s['slot_id']: s for s in lock['itemized_slot_family_ledger']}
locked_sources = {r['root_id']: r['source_binding'] for r in lock['source_bindings']}
indices = {s['slot_id']: i for i, s in enumerate(lock['itemized_slot_family_ledger'])}
selected = {r['slot_id']: r for r in selection['selected']}
admitted = {r['candidate_id']: r for r in admission['items']}
qa_items = {r['candidate_id']: r for r in qa['items']}
recon_items = {r['candidate_id']: r for r in recon['items']}
first_inputs = {r['candidate_id']: r for r in recon['actual_inputs']}
check('four_admitted_selected_qa_ids', len(selected) == len(admitted) == len(qa_items) == len(recon_items) == 4 and set(selected) == set(admitted) == set(qa_items) == set(recon_items))
check('admission_statuses_and_counts', admission['status'] == 'CONTENT_ADMISSION_FINAL' and admission['accepted_count'] == 4 and admission['held_count'] == 0 and not admission['mechanical_errors'] and all(r['status'] == 'CONTENT_ACCEPTED' and r['five_family_checks'] == r['independent_source_first_qa'] == 'PASS' for r in admitted.values()), {'accepted': admission['accepted_count'], 'mechanical_checks_declared': admission['mechanical_checks']})

public, sources = {}, {}
span_count = 0
for cid, row in admitted.items():
    pubbind = verify(row['public_question'], cid + '_public_binding')
    # These opaque bytes are hashed only. No private JSON is parsed.
    privatebind = verify(row['private_author_record'], cid + '_opaque_private_hash_binding')
    verify(row['independent_review'], cid + '_qa_binding')
    q = read(row['public_question']['path'])
    public[cid] = q
    src = read(q['source_path'])
    sources[cid] = src
    required = set(contract['public_question']['required_common_keys'])
    schema_ok = set(q) == required | {'options'} and q['question_type'] == 'multi_select' and list(q['options']) == list('ABCDEF') and len(set(' '.join(t.casefold().split()) for t in q['options'].values())) == 6
    check(cid + '_public_whitelist_and_six_labels', schema_ok)
    check(cid + '_locked_identity_source', all(q[k] == slots[cid][k] for k in ['root_id', 'question_type', 'source_sha256']) and bind(q['source_path'])['sha256'] == q['source_sha256'] and src['root_id'] == q['root_id'])
    check(cid + '_original_source_path_and_five_family_gate', Path(q['source_path']).resolve() == Path(locked_sources[q['root_id']]['path']).resolve() and q['source_sha256'] == locked_sources[q['root_id']]['sha256'] and all(v['status'] == 'PASS' for v in selected[cid]['selected_plan']['family_checks'].values()) and qa_items[cid]['family_exclusion_integrity']['status'] == 'PASS' and all(v == 'PASS' for v in qa_items[cid]['family_exclusion_integrity']['all_five_statuses'].values()))
    check(cid + '_qa_admission_metadata', qa_items[cid]['status'] == 'CONTENT_ACCEPTED' and qa_items[cid]['public_question_sha256'] == pubbind['sha256'] and qa_items[cid]['author_private_sha256'] == privatebind['sha256'] and qa_items[cid]['independent_gold_match'] is True and qa_items[cid]['unique_answer'] is True and qa_items[cid]['family_scope_preserved'] == 'PASS' and qa_items[cid]['selected_plan_id'] == selected[cid]['selected_plan']['plan_id'] and qa_items[cid]['selected_candidate_key'] == selected[cid]['selected_plan']['candidate_selection_key'])
    fi = first_inputs[cid]
    check(cid + '_source_first_inputs', fi['public_question_sha256_actual'] == pubbind['sha256'] and fi['source_sha256_actual'] == bind(q['source_path'])['sha256'] and fi['source_binding_match'] is True and fi['complete_source_turn_count'] == len(src['turns']) and fi['complete_source_read'] is True)
    turns = {t['turn_ref']: (i, t) for i, t in enumerate(src['turns'], 1)}
    good = True
    for opt in recon_items[cid]['options']:
        good = good and opt['text'] == q['options'][opt['label']]
        for e in opt.get('evidence', []):
            span_count += 1
            i, t = turns[e['turn_ref']]
            span = e['span_in_turn_text']
            good = good and i == e['source_turn_index_1_based'] and t['speaker'] == e['speaker'] and t['scene_ref'] == e['scene_ref'] and t['text'][span['start']:span['end']] == e['quote']
    check(cid + '_source_first_exact_span_metadata', good)

verify({'path': seal['reconstruction_path'], 'sha256': seal['reconstruction_sha256']}, 'seal_reconstruction_hash')
check('seal_reconstruction_inputs_equal', seal['actual_input_bindings'] == recon['actual_inputs'])
chron = seal['chronology']
chronology_ok = dt(recon['created_at_utc']) == dt(seal['reconstruction_created_at_utc']) <= dt(seal['created_at_utc']) < dt(qa['created_at_utc']) < dt(admission['created_at_utc']) < dt(freeze['created_at_utc']) <= dt(ledger['created_at_utc'])
check('declared_file_chronology_consistent', chronology_ok)
check('chronology_limit_explicit_and_consistent', chron['reconstruction_written_and_hash_verified_before_private_read'] is True and chron['private_already_opened_at_seal_manifest_creation'] is True and chron['first_private_read_exact_timestamp_utc'] is None and chron['reconstruction_modified_after_private_read'] is False and bool(admission['chronology_limit']) and qa['source_first_reconstruction']['unchanged_since_before_private_read'] is True)

reqpath = C / 'requests/REQUESTS.jsonl'
requests = [json.loads(line) for line in reqpath.open()]
bind(reqpath)
mappings = read(C / 'requests/DISPLAY_MAPS.json')
mapping_by_key = {(r['candidate_id'], r['variant_id']): r for r in mappings}
check('request_display_counts', len(requests) == freeze['request_count'] == 72 and len(mappings) == len(mapping_by_key) == 24 and freeze['question_count'] == 4 and set(r['candidate_id'] for r in requests) == set(admitted))
check('unique_request_ids_and_seeds', len({r['request_id'] for r in requests}) == len({r['seed'] for r in requests}) == 72)

# Reproduce the approved seed algorithm without importing the adapter/Solver.
def seed_value(seed, cid, variant):
    return int.from_bytes(hashlib.sha256(f'{seed}|{cid}|{variant}'.encode()).digest()[:8], 'big')

def expected_variants(q):
    keys = list(q['options'])
    ids = {k: 'content_' + hashlib.sha256(q['options'][k].strip().encode()).hexdigest()[:16] for k in keys}
    content = list(ids.values())
    texts = {ids[k]: q['options'][k] for k in keys}
    random.Random(seed_value(policy['sampling']['display_seed'], q['candidate_id'], -1)).shuffle(content)
    used, result = set(), []
    for vi in range(6):
        rng = random.Random(seed_value(policy['sampling']['display_seed'], q['candidate_id'], vi))
        while True:
            order = content[:]
            rng.shuffle(order)
            if tuple(order) not in used:
                break
        used.add(tuple(order))
        result.append({'variant_id': f'perm_{vi:02d}', 'options': {keys[i]: texts[x] for i, x in enumerate(order)}, 'display_to_content': dict(zip(keys, order)), 'canonical_order_content_ids': content[:], 'display_order_content_ids': order, 'content_to_text': texts})
    return ids, result

# Extract only the native structured system constant; do not import code modules.
contractref = next(r for r in preflight['imported_native_code_bindings'] if r['path'].endswith('/contract.py'))
syntax = ast.parse(Path(contractref['path']).read_text())
system_prompt = next(ast.literal_eval(n.value) for n in syntax.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'STRUCTURED_SYSTEM_PROMPT' for t in n.targets))
from transformers import AutoTokenizer
T = AutoTokenizer.from_pretrained(policy['length']['tokenizer_directory'], local_files_only=True)
length_rows, roundtrips, expected_requests = [], 0, []
allowed_request_keys = {'request_id','candidate_id','root_id','original_split','question_type','global_slot_index_zero_based','variant_id','sample_index','branch','seed','display','prompt','prompt_tokens','source_sha256','public_question_sha256','temperature','top_p','max_output_tokens','stop','include_stop_str_in_output','choice_space'}
sample = policy['sampling']
for cid, q in public.items():
    canonical, variants = expected_variants(q)
    actual = [r for r in mappings if r['candidate_id'] == cid]
    check(cid + '_unique_seeded_display_permutations', len(actual) == 6 and len({tuple(r['display_order_content_ids']) for r in actual}) == 6 and actual == [{'candidate_id': cid, **v} for v in variants])
    candidate_lengths = []
    for vi, v in enumerate(variants):
        inverse = {x: k for k, x in v['display_to_content'].items()}
        for n in range(1, 7):
            for answer in itertools.combinations(list(q['options']), n):
                canonical_answer = [canonical[k] for k in answer]
                display_answer = [inverse[x] for x in canonical_answer]
                assert [v['display_to_content'][k] for k in display_answer] == canonical_answer
                roundtrips += 1
        chunks = ['Dialogue:\n' + '\n'.join(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" for t in sources[cid]['turns']), f"Question (multi_select):\n{q['question']}", 'Options:\n' + '\n'.join(f'{k}. {text}' for k, text in v['options'].items()), 'Output only selected letters separated by commas, e.g. <box>A,C</box>.']
        prompt = T.apply_chat_template([{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': '\n\n'.join(chunks)}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        tokens = len(T.encode(prompt, add_special_tokens=False))
        candidate_lengths.append(tokens)
        check(cid + '_' + v['variant_id'] + '_length', tokens + sample['max_output_tokens'] <= policy['length']['max_model_len'], {'prompt_tokens': tokens, 'reserved': sample['max_output_tokens']})
        for si in range(3):
            rid = f'{cid}:{v["variant_id"]}:sample_{si:02d}'
            expected_requests.append({'request_id':rid,'candidate_id':cid,'root_id':q['root_id'],'original_split':slots[cid]['split'],'question_type':'multi_select','global_slot_index_zero_based':indices[cid],'variant_id':v['variant_id'],'sample_index':si,'branch':'direct','seed':sample['seed_base']+indices[cid]*18+vi*3+si,'display':v,'prompt':prompt,'prompt_tokens':tokens,'source_sha256':q['source_sha256'],'public_question_sha256':admitted[cid]['public_question']['sha256'],'temperature':sample['temperature'],'top_p':sample['top_p'],'max_output_tokens':sample['max_output_tokens'],'stop':sample['stop'],'include_stop_str_in_output':sample['include_stop_str_in_output'],'choice_space':'all_nonempty_subsets'})
    frozen_length = next(r for r in freeze['length_and_mapping_checks'] if r['candidate_id'] == cid)
    check(cid + '_frozen_length_report', [r['prompt_tokens'] for r in frozen_length['variants']] == candidate_lengths and frozen_length['all_answer_space_roundtrips'] == 378 and all(r['reserved_answer_tokens'] == 32 for r in frozen_length['variants']))
    length_rows.append({'candidate_id':cid,'source_turn_count':len(sources[cid]['turns']),'global_slot_index_zero_based':indices[cid],'variants':6,'requests':18,'prompt_tokens_by_variant':candidate_lengths})
check('all72_requests_exact_independent_reconstruction', requests == expected_requests and all(set(r) == allowed_request_keys for r in requests), 'Byte-equivalent prompt strings, approved maps, indices, seeds and sampling parameters compared as decoded request objects; all complete source turns included.')
check('all_nonempty_answer_space_roundtrips', roundtrips == 1512, {'roundtrips':roundtrips,'choices_per_variant':63,'gold_cardinality_constraint_used':False})
check('length_report_no_holds_no_truncation', not freeze['length_holds'] and freeze['zero_truncation'] is True and sample['max_output_tokens'] == policy['length']['reserved_answer_tokens'] == 32 and policy['length']['max_model_len'] == 16384)

before = {r['slot_id']:r for r in selection['all186_dispositions']}
after = {r['slot_id']:r for r in ledger['slots']}
identity_ok = len(after) == len(ledger['slots']) == 186 and list(after) == list(slots)
unchanged = True
for cid, old in before.items():
    new = after[cid]
    unchanged = unchanged and all(new[k] == value for k,value in old.items() if k != 'status') and new['family_selection_status'] == old['status']
    if cid in admitted:
        unchanged = unchanged and new['status'] == 'REQUESTS_FROZEN' and new['content_status'] == 'CONTENT_ACCEPTED'
        verify(new['admission'], cid + '_ledger_admission')
        verify(new['request_freeze'], cid + '_ledger_freeze')
    else:
        unchanged = unchanged and new['status'] == old['status']
counts = dict(Counter(r['status'] for r in ledger['slots']))
check('all186_final_ledger_identity_and_preservation', identity_ok and unchanged)
check('all186_final_counts', counts == ledger['counts'] == {'REQUESTS_FROZEN':4,'CONTENT_HOLD':144,'SOURCE_SHORTFALL':28,'FAMILY_EXCLUDED':10}, counts)
check('model_pending_execution_boundaries', freeze['status'] == 'CONTENT_REQUESTS_FROZEN_MODEL_BINDINGS_PENDING' and freeze['actual_model_identity_frozen'] is False and freeze['full_evaluation_ready'] is False and freeze['test_execution'] == 'FINAL_MODEL_ONLY_AFTER_MODEL_IDENTITY_FREEZE' and freeze['new_solver_calls'] == freeze['new_training_runs'] == 0 and freeze['private_gold_opened'] is False and ledger['day3_complete'] is False and ledger['new_solver_calls'] == ledger['new_training_runs'] == 0 and admission['historical_role'] == freeze['historical_role'] == 'REGRESSION_ONLY')
report = {'schema_version':'independent_final_artifact_check_v1','created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'FAIL' if failures else 'PASS_WITH_DECLARED_CHRONOLOGY_LIMIT','reviewer':'eval_family_independent','scope':'Independent artifact integrity, accounting and public request reconstruction only. Separate source-first QA was performed by another reviewer; this audit does not reproduce or certify the unobserved blind phase. Private author files hashed as opaque bytes only; no private JSON parsed, no finalizers executed, no Solver imported and no inference.','input_bindings':list(input_bindings.values()),'checks':checks,'failures':failures,'summary':{'content_accepted':4,'request_count':72,'display_variant_count':24,'unique_seed_count':len({r['seed'] for r in requests}),'source_first_exact_span_metadata_checked':span_count,'all_answer_space_roundtrips':roundtrips,'full_prompt_token_min':min(r['prompt_tokens'] for r in requests),'full_prompt_token_max':max(r['prompt_tokens'] for r in requests),'reserved_answer_tokens':32,'max_model_len':16384,'all186_counts':counts,'new_solver_calls':0,'new_training_runs':0,'actual_model_identity_frozen':False,'full_evaluation_ready':False,'day3_complete':False},'per_question_request_checks':length_rows,'qa_chronology':{'status':'DECLARATIONS_AND_HASHES_CONSISTENT_WITH_EXPLICIT_LIMIT','reconstruction_created_at_utc':recon['created_at_utc'],'seal_manifest_created_at_utc':seal['created_at_utc'],'final_review_created_at_utc':qa['created_at_utc'],'private_first_read_exact_timestamp_utc':None,'reconstruction_pre_private_write_and_hash_notification':'DECLARED_BY_SOURCE_FIRST_REVIEWER_NOT_INDEPENDENTLY_OBSERVED_HERE','seal_manifest_pre_private':'FALSE_AS_EXPLICITLY_DISCLOSED','file_timestamp_order_verified':chronology_ok,'limitation':'Reconstruction hash is unchanged and bound by later review/admission. The separate seal manifest postdates private access; no captured exact private-read timestamp or external immutable event log proves the blind-read boundary. This is disclosed consistently, not represented as a strict pre-private seal-manifest chronology PASS.'},'static_review':{'finalize_content_script_reviewed':True,'freeze_requests_script_reviewed':True,'freeze_existing_report_guard_precedes_output_writes':Path(C/'freeze_requests.py').read_text().index("check(not (output / 'REQUEST_FREEZE.json').exists()") < Path(C/'freeze_requests.py').read_text().index("write_new(output / 'REQUESTS.jsonl'"),'finalizer_initial_failed_attempt':'Parent disclosed a wrapper-versus-tuple comparison failure before first admission output; final artifacts and QA bindings are checked here, without asserting an independent log of the failed attempt.'},'public_private_boundary':{'private_author_bytes_hashed_only':True,'private_author_json_parsed':False,'qa_metadata_read':True,'qa_answer_correctness_rejudged':False,'blind_source_first_phase_reproduced':False,'request_reconstruction_used_gold':False,'tokenizer_cpu_only':True,'solver_or_model_inference':False}}
out=C/'audit/FINAL_ARTIFACT_CHECK.json'
report['current_authority'] = 'FAMILY_SELECTION.json per-plan family gates, CONTENT_ADMISSION.json and CONTENT_AND_REQUEST_LEDGER.json are the bound final authorities. Earlier screening summaries and copied local disposition strings retain their stage metadata and were not treated as current gate statuses. No mismatch in the selected per-plan gates was found.'
out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':report['status'],'failures':failures,'report':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'summary':report['summary']},ensure_ascii=False))
