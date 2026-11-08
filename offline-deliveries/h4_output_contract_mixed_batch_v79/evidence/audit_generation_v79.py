"""Read-only generation and public whole-set dedup replay; no Solver access."""
from collections import Counter
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import hashlib
import json
import os
import sys

HOME = Path(__file__).resolve().parents[1]
OUT = HOME / 'evidence/GENERATION_AUDIT.json'
OPENED = set()
sys.dont_write_bytecode = True
sys.path.insert(0, str(HOME))


def guard(event, args):
    if event.startswith('socket.') or event in {'subprocess.Popen', 'os.system', 'os.posix_spawn'}:
        raise RuntimeError('audit_forbids_external:' + event)
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(args[0])).resolve()
        mode, flags = args[1:3]
        writing = ((isinstance(mode, str) and any(c in mode for c in 'wax+'))
                   or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
        if writing and path not in {OUT, Path('/dev/null')} and not path.is_relative_to(Path('/tmp')):
            raise RuntimeError('audit_forbids_write:' + str(path))
        if path.is_relative_to(HOME):
            if path.is_relative_to(HOME / 'solver'):
                raise RuntimeError('audit_forbids_current_final_or_solver:' + str(path))
            if not writing:
                OPENED.add(str(path))


sys.addaudithook(guard)
import common
import ordering_generation as ordering
import ordering_review
a = common.load_multi('asset_generation')
c = common.load_multi('compose')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}


def check(row):
    assert sha(row['path']) == row['sha256'], 'binding_changed:' + row['path']


config = read(HOME / 'CONFIG.json')
prep = read(HOME / 'BATCH_PREPARATION.COMPLETE.json')
for row in prep['files']:
    check(row)
for row in read(a.HERE / 'OFFLINE_TEST_RECEIPT.json')['files']:
    check(row)
for slot in config['multiselect_config']['slots']:
    a.verify(HOME / 'multiselect/assets' / slot['slot_id'], complete=True)
ordering_result = ordering.verify(HOME)
ordering_review_result = ordering_review.verify_review(HOME)
assembled = c.verify_assembly(HOME / 'multiselect')
final_review = common.load_multi('evaluate').verify_review(HOME / 'multiselect')
assert len(assembled['questions']) == final_review['usable_count'] == 6
_, _, annotate, capacity, dev = a.base.transport()
tokenizer = dev.tokenizer_from(str(a.base.TOKENIZER))
calls, multi_mechanics, order_mechanics = [], [], []
request_dirs = sorted(HOME.glob('multiselect/assets/*/rounds/*/requests/*')) + sorted(HOME.glob('ordering/*/requests/*'))
assert len(request_dirs) == 5
for request in request_dirs:
    is_multi = request.is_relative_to(HOME / 'multiselect')
    round_folder = request.parents[1]
    run = round_folder.parents[1] if is_multi else round_folder
    manifest = read(round_folder / 'MANIFEST.json')
    payload, proof = read(request / 'REQUEST.json'), read(request / 'PROOF.json')
    frozen_payload = read(round_folder / ('INITIAL_REQUEST.json' if is_multi else 'REQUEST.json'))
    frozen_proof = read(round_folder / ('INITIAL_PROOF.json' if is_multi else 'PROOF.json'))
    assert payload == frozen_payload
    assert {k:v for k,v in proof.items() if k != 'capacity'} == frozen_proof
    assert proof['payload_sha256'] == a.base.digest(payload)
    source = read(run / 'SOURCE.json')
    content = payload['messages'][0]['content']
    context, _ = json.JSONDecoder().raw_decode(content.split('INPUT_JSON\n', 1)[1])
    assert context['source_turns'] == source['turns']
    assert context['source_metadata'] == {k:v for k,v in source.items() if k != 'turns'}
    original = Path(manifest['source_history']['path'] if is_multi else manifest['original_source']['path'])
    assert (run / 'SOURCE.json').read_bytes() == original.read_bytes()
    attempt = read(request / 'HTTP_ATTEMPT.json')
    reservation, completion = read(request / 'RESERVATION.json'), read(request / 'HTTP_RESULT.json')
    assert attempt == {'actual_payload_equals_frozen':True, 'payload_sha256':a.base.digest(payload),
        'requested_model':a.legacy.MODEL, 'http_attempt_number':1, 'transport_retries':0}
    assert reservation['automatic_retries'] == 0 and reservation['request_number'] == 1
    assert completion['http_status'] == 200 and completion['outcome'] == 'response'
    assert not (request / 'ERROR.json').exists()
    response = read(request / 'RESPONSE.json'); metadata = response['metadata']
    assert json.loads(metadata['raw_response_text']) == metadata['raw_response']
    assert response['text'] == metadata['raw_response']['choices'][0]['message']['content']
    assert all(metadata[k] == a.legacy.MODEL for k in ['requested_model','actual_model','returned_model'])
    assert metadata['http_status'] == 200 and metadata['finish_reason'] == 'stop' and metadata['timeout'] is False
    assert metadata['transport_attempt_count'] == metadata['http_attempt_count'] == 1
    assert metadata['transport_retry_count'] == metadata['rate_limit_retry_count'] == 0
    profile = read(request / 'PROFILE.json')
    assert payload == {'model':profile['model'], 'messages':payload['messages'],
                       **{k:v for k,v in profile['request_parameters'].items() if v is not None}}
    assert profile['http_retry_limit'] == 0 and profile['parameter_fallback'] is False
    cap = annotate(capacity(profile, tokenizer, payload['messages'], payload['max_tokens']))
    assert cap == proof['capacity'] and cap['status'] == 'PASS' and cap['truncation'] is False
    entry = {'lane':'multi_select' if is_multi else 'ordering', 'root_id':manifest['root_id'],
        'round_id':manifest.get('round_id'), 'source_turn_count':len(source['turns']),
        'request':bind(request / 'REQUEST.json'), 'proof':bind(request / 'PROOF.json'),
        'response':bind(request / 'RESPONSE.json'), 'http_attempt':bind(request / 'HTTP_ATTEMPT.json'),
        'profile':bind(request / 'PROFILE.json'), 'source':bind(run / 'SOURCE.json'),
        'raw_service_content_equals_saved_text':True, 'complete_source_bytes_and_metadata_preserved':True,
        'payload_and_proof_replay':True, 'capacity':cap, 'service_reported_usage':metadata['usage'],
        'http_status':metadata['http_status'], 'finish_reason':metadata['finish_reason'], 'retry_count':0}
    if is_multi:
        targets = manifest['target_fact_ids']
        tail = a.render_task_tail(manifest, targets)
        assert content.endswith(tail)
        assert context['target_fact_ids'] == targets and proof['target_fact_ids'] == targets
        skeleton, end = json.JSONDecoder().raw_decode(tail.split('OUTPUT_JSON_SKELETON\n',1)[1])
        assert [r['fact_id'] for r in skeleton['replacements']] == targets
        raw = a.strict_json(response['text'])
        assert [r['fact_id'] for r in raw['replacements']] == targets
        if manifest['round_id'] == 'supplement':
            assert targets == ['F02','F03','F04','F05','F06'] and 'F01' not in tail
        data, errors = a.validate_assets(response['text'], manifest, source, read(run / 'PARENT.json'), targets)
        assert read(request / 'VALIDATION.json') == a.validation(data, errors)
        review = read(round_folder / 'SOURCE_JUDGMENTS.json')
        public = read(round_folder / 'PUBLIC_ASSETS.json')
        raw_by_fact = {r['fact_id']:r['replacement_proposition'] for r in raw['replacements']}
        assert all(v['proposition'] == raw_by_fact[v['fact_id']] for v in public['versions'])
        multi_mechanics.append({'root_id':manifest['root_id'], 'round_id':manifest['round_id'],
            'target_fact_ids':targets, 'raw_fact_ids':[r['fact_id'] for r in raw['replacements']],
            'exact_tail_skeleton':skeleton, 'mechanical_errors':errors,
            'rejected_raw_rows':[r for r in raw['replacements'] if any(e['fact_id']==r['fact_id'] for e in errors)],
            'public_version_count':len(public['versions']),
            'source_review_parent_relations':dict(Counter(r['relation'] for r in review['parents'])),
            'source_review_version_relations':dict(Counter(r['relation'] for r in review['versions'])),
            'source_review':bind(round_folder / 'SOURCE_JUDGMENTS.json'),
            'per_version_source_findings':review['versions'], 'raw_texts_unedited_in_public_projection':True})
    else:
        prior = read(run / 'PRIOR.json')
        accepted, rejected = ordering.validate_raw(response['text'],manifest,source,run/'SOURCE.json',prior)
        assert accepted == read(run/'POOL.json')['questions'] and rejected == read(run/'REJECTIONS.json')
        raw = a.strict_json(response['text']); accepted_ids = {q['candidate_id'] for q in accepted}
        rejected_ids = {q['candidate_id']:q for q in rejected}
        lookup = {t['turn_ref']:t for t in source['turns']}
        for row in raw['questions']:
            assert row['question'] == 'Order the following six statements as they APPEAR IN THE SUPPLIED DIALOGUE.'
            findings = []
            for error in rejected_ids.get(row['candidate_id'],{}).get('reasons',[]):
                label = error.split(':')[0]
                if label in row['item_evidence']:
                    ev = row['item_evidence'][label]
                    findings.append({'reason':error,'raw_evidence':ev,'actual_source_turn':lookup[ev['turn_ref']]})
            if row['candidate_id'] in accepted_ids:
                q = next(q for q in accepted if q['candidate_id']==row['candidate_id'])
                assert row['gold'] == q['gold'] and row['items'] == q['items']
            order_mechanics.append({'candidate_id':row['candidate_id'],'mechanically_accepted':row['candidate_id'] in accepted_ids,
                'rejection_reasons':rejected_ids.get(row['candidate_id'],{}).get('reasons',[]),
                'exact_quote_failure_details':findings,'raw_gold':row['gold'],
                'gold_or_items_rewritten':False,'prior_whole_set_matches':[q['candidate_id'] for q in prior if ordering.whole_set(q)==ordering.whole_set(row)]})
    calls.append(entry)

# Reconstruct only public six-statement sets from approved generation assets.
# Final source judgments are now saved; parent authorized full assembly replay.
multi = HOME / 'multiselect'; cfg = read(multi/'CONFIG.json')
prior_paths = {Path(cfg['baseline_pool'])} | {Path(p) for s in cfg['slots'] for p in s.get('prior_pool_paths',[])}
prior_index = {}
for path in sorted(prior_paths):
    for original in c._rows(read(path)):
        q = original.get('original_selected_question',original)
        if q.get('question_type') == 'multi_select':
            prior_index.setdefault(c.statement_key(q),[]).append({'candidate_id':q['candidate_id'],'prior':bind(path)})
current_public = read(multi/'PUBLIC_QUESTIONS.json')['questions']
members = read(multi/'FAMILY_SIDECAR.json')['members']
roles = {r['candidate_id']:r for r in members}
assert len(current_public) == len(members) == 6
assert sum(r['role']=='child' for r in members) == 4
assert all(r['new_candidate'] == (r['role']=='child') for r in members)
dedup, novel, repeated_facts = [], [], []
for slot in cfg['slots']:
    run = multi/'assets'/slot['slot_id']; public = read(run/'PUBLIC_ASSETS.json')
    variants = {p['fact_id']:[(p['proposition'],True)] for p in public['parent_facts']}
    for v in public['versions']: variants[v['fact_id']].append((v['proposition'],False))
    available = {k:[] for k in (2,3,4,5)}
    for combination in product(*(variants[fid] for fid in sorted(variants))):
        count = sum(supported for _,supported in combination)
        if count not in available: continue
        key = (slot['root_id'],'multi_select',tuple(sorted(c.norm(text) for text,_ in combination)))
        if len(set(key[2])) != 6: continue
        if key in prior_index:
            dedup.append({'root_id':slot['root_id'],'gold_count_from_reviewed_supports':count,
                'whole_set_digest':c.digest(key),'normalized_statement_set':list(key[2]),
                'reason':'duplicate_prior_six_statement_set','matched_prior_questions':prior_index[key],
                'excluded_from_new_count':True})
        else: available[count].append(key)
    children = [q for q in current_public if q['root_id']==slot['root_id'] and roles[q['candidate_id']]['role']=='child']
    for count, keys in available.items():
        matching = [q for q in children if roles[q['candidate_id']]['gold_count']==count]
        assert len(matching) == bool(keys)
        if not keys: continue
        chosen = min(keys,key=c.digest);q=matching[0]
        assert c.statement_key(q) == chosen and chosen not in prior_index
        assert q['candidate_id']==slot['root_id']+'-v79-'+slot['family_id']+'-'+c.digest(chosen)[:12]
        novel.append({'candidate_id':q['candidate_id'],'root_id':q['root_id'],'gold_count_from_reviewed_supports':count,
            'whole_set_digest':c.digest(chosen),'new_exact_six_option_combination':True,
            'new_independent_facts':False,'same_parent_and_source_family':slot['family_id']})
    old_path = HOME.parent/'h4_ready_roots_mixed_batch_v78/multiselect/assets'/slot['slot_id']/'rounds/initial/PUBLIC_ASSETS.json'
    old = read(old_path);new = read(run/'rounds/initial/PUBLIC_ASSETS.json')
    old_by_fact = {r['fact_id']:r['proposition'] for r in old['versions']}
    repeated_facts.append({'root_id':slot['root_id'],'old_public_assets':bind(old_path),
        'new_public_assets':bind(run/'rounds/initial/PUBLIC_ASSETS.json'),
        'identical_per_fact_replacements':[v['fact_id'] for v in new['versions'] if old_by_fact.get(v['fact_id'])==v['proposition']],
        'new_independent_fact_claim_permitted':False})
assert len(novel)==4 and Counter(r['gold_count_from_reviewed_supports'] for r in novel)==Counter({2:1,3:1,4:1,5:1})
root130dup = [r for r in dedup if r['root_id']=='cc-train-episode-130221']
assert len(root130dup)==1 and root130dup[0]['gold_count_from_reviewed_supports']==5
assert not any(q['root_id']=='cc-train-episode-130221' for q in novel)
review = read(HOME/'ordering/CONTENT_REVIEW.json'); review_sources = read(HOME/'ordering/SOURCE_JUDGMENTS.json')
assert review['accepted_count']==0 and review['rejected_count']==2 and review['question_or_gold_edits']==[]
review_findings = []
for q in review['questions']:
    source_row = next(r for r in review_sources['questions'] if r['candidate_id']==q['candidate_id'])
    review_findings.append({'candidate_id':q['candidate_id'],'status':q['status'],'reasons':q['reasons'],
        'complete_source_turns_read':source_row['complete_source_turns_read'],
        'source_review_reason':source_row['reason'],'alternative_orders':source_row['alternative_orders'],
        'items':source_row['items']})
initial_count=read(HOME/'BATCH_GENERATION.COMPLETE.json')['http_attempts']
assert initial_count==4 and read(multi/'GENERATION.COMPLETE.json')['http_attempts']==3
assert read(HOME/'ordering/GENERATION.COMPLETE.json')['http_attempts']==2
assembly_note = read(HOME/'evidence/AUDIT_ASSEMBLY_NOTE.json')
check(assembly_note['followup_entrypoint_rejected_missing_audits'])
assert assembly_note['status']=='PASS' and assembly_note['review_source_edits']==assembly_note['code_edits']==[]
assert assembly_note['model_retries']==0 and assembly_note['questions']==6
result={'version':'v79_independent_generation_and_dedup_audit_v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
    'status':'PASS_WITH_GENERATION_AND_CONTENT_SHORTFALLS',
    'disclosures':{'audit_generation_http_calls':0,'audit_solver_calls':0,'network_and_subprocess_blocked':True,
        'current_final_multiselect_private_pool_read':True,'current_final_multiselect_reviews_read':True,
        'final_private_access_authorization':'Parent confirmed both source-only groups saved and froze final review before authorizing this audit replay.',
        'current_solver_outputs_read':False,'generation_private_and_raw_read':True,'frozen_code_modified':False},
    'totals':{'actual_generation_http_calls':5,'initial_batch_snapshot_http_calls':4,'explicit_bounded_supplement_http_calls':1,
        'new_multiselect_six_option_combinations':4,'new_independent_fact_claim':False,'historical_controls':2,
        'ordering_raw_candidates':4,'ordering_mechanical_accepted':2,'ordering_mechanical_rejected':2,
        'ordering_content_usable':0,'ordering_content_rejected':2},
    'replay_checks':{'multiselect_complete_generation_verify':'PASS','ordering_generation_verify':'PASS',
        'ordering_content_review_verify':'PASS','all_payload_proof_source_capacity_receipt_raw_bindings':'PASS',
        'current_public_composition_reconstructed_from_generation_assets_and_prior':'PASS',
        'full_final_private_assembly_replay':'PASS; six questions and six usable frozen final reviews.',
        'root_and_engine_frozen_code_hashes':'PASS'},
    'calls':calls,'multiselect_rounds':multi_mechanics,'ordering_mechanical_findings':order_mechanics,
    'ordering_source_review_findings':review_findings,'deduplicated_combinations':dedup,
    'new_combinations':novel,'repeated_per_fact_replacement_texts':repeated_facts,
    'final_review_audit_assembly_note':{'record':bind(HOME/'evidence/AUDIT_ASSEMBLY_NOTE.json'),
        'saved_final_review_replay':'PASS','reported_source_edits':[],'reported_frozen_code_edits':[],
        'model_retries':0,'classification':'Completion of missing local audit records before final freeze; not a generation or Solver retry.'},
    'interpretation':['Root194259 repeats all six per-fact replacement sentences from v78. Four different complete six-option combinations pass the frozen exact-set dedup rule; they remain dependent siblings of one parent, not four independent new fact sets.',
        'Root130221 has only one approved false version. Its only Gold5 combination already exists in the v78 public prior, so it contributes zero new children; its historical control is retained with new_candidate=false.',
        'The immutable BATCH_GENERATION.COMPLETE.json is the initial four-call snapshot; the explicitly authorized bounded supplement raises actual total calls to five without an automatic retry.',
        'Capacity counts use the frozen Qwen3-14B proxy and local limits. Service token usage is reported separately and is not asserted to equal the proxy count.'],
    'evidence':[bind(HOME/'evidence/SUPPLEMENT_DECISION.json'),bind(multi/'PUBLIC_QUESTIONS.json'),
        bind(multi/'ALLOCATION.json'),bind(multi/'FAMILY_SIDECAR.json'),bind(HOME/'ordering/CONTENT_REVIEW.json'),
        bind(HOME/'ordering/SOURCE_JUDGMENTS.json'),bind(HOME/'BATCH_PREPARATION.COMPLETE.json')],
    'frozen_root_code_bindings':[r for r in prep['files'] if r['path'].endswith('.py')],
    'opened_current_batch_files':sorted(OPENED)}
with OUT.open('x') as stream:
    json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
print('GENERATION_AUDIT PASS_WITH_GENERATION_AND_CONTENT_SHORTFALLS; calls=5; new_combinations=4; controls=2; ordering_usable=0')
print('sha256',sha(OUT))
