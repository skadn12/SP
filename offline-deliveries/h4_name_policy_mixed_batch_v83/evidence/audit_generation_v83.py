"""Independent generation/assembly/dedup evidence audit; never invoke a model.

Production validators are replayed, then request receipts and public whole-set
composition are reconstructed separately. This is not blind semantic review.
"""
from collections import Counter
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import hashlib
import json
import os
import re
import sys
import traceback

HOME = Path(__file__).resolve().parents[1]
OUT = HOME / 'evidence/GENERATION_AUDIT.json'
OPENED = set()
CHECKS = []
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
                raise RuntimeError('audit_forbids_solver:' + str(path))
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


def require(condition, name, detail=None):
    record = {'check': name, 'status': 'PASS' if condition else 'FAIL'}
    if detail is not None:
        record['detail'] = detail
    CHECKS.append(record)
    if not condition:
        raise AssertionError(name)


def check_binding(row):
    require(sha(row['path']) == row['sha256'], 'binding:' + row['path'])


def normalized_set(q):
    field = 'options' if q['question_type'] == 'multi_select' else 'items'
    return (q['root_id'], q['question_type'], tuple(sorted(' '.join(t.split()).casefold() for t in q[field].values())))


PRONOUNS = ('he','him','his','himself','she','her','hers','herself')


def independent_name_errors(items):
    return ['ordering_actor_reference_policy_forbidden_pronoun:' + label + ':' + word
            for label, text in items.items()
            for word in sorted(set(re.findall(r'\b(?:he|him|his|himself|she|her|hers|herself)\b', text.casefold())))]


def independent_scope(q, source):
    """Recreate the old documented public scene boundary, without Gold input."""
    scenes = list(dict.fromkeys(t['turn_ref'].rsplit(':', 1)[0] for t in source['turns']))
    match = re.search(r'\bAPPEAR IN THE SUPPLIED DIALOGUE (?P<restriction>within the (?P<scenes>.+?) scenes?\.?)\s*$', q['question'], re.I)
    selected, fragment = [], None
    if match:
        entries = re.split(r'\s*,\s*(?:and\s+)?|\s+and\s+', match.group('scenes'))
        if entries and all(entry in scenes for entry in entries):
            selected = [scene for scene in scenes if scene in entries]
            fragment = match.group('restriction')
    return {'contract': 'explicit_public_stem_scene_scope_v52', 'candidate_id': q['candidate_id'],
            'kind': 'explicit_scene_union' if selected else 'full_dialogue',
            'public_stem_restriction': fragment, 'scene_refs': selected,
            'matching_turn_refs': [t['turn_ref'] for t in source['turns'] if not selected or t['turn_ref'].rsplit(':', 1)[0] in selected],
            'context_turn_refs': [t['turn_ref'] for t in source['turns']]}


def audit():
    config = read(HOME / 'CONFIG.json')
    prep = read(HOME / 'BATCH_PREPARATION.COMPLETE.json')
    require(config['max_generation_http']==6 and config['max_solver_calls_per_model']==228
            and config['max_candidates']==14, 'unchanged_mixed_generation_budgets')
    require(config['multiselect_engine'] == str(a.HERE), 'same_v79_multiselect_engine')
    check_binding(config['actor_reference_policy'])
    check_binding(config['character_registry'])
    actor_policy=read(config['actor_reference_policy']['path'])
    registry=read(config['character_registry']['path'])
    require(actor_policy['registry']==config['character_registry']
            and actor_policy['forbidden_gendered_pronouns']==list(PRONOUNS)
            and actor_policy['output_style']=='use_source_names_or_explicit_roles_for_people'
            and actor_policy['new_source_facts'] is False and actor_policy['automatic_repair'] is False,
            'frozen_actor_style_contract_adds_no_qa_facts_or_repairs')
    require(Path(config['character_registry']['path']).read_bytes()==
            (HOME.parent/'h4_actor_reference_minfix_ordering_v82/CHARACTER_REGISTRY.json').read_bytes(),
            'registry_unchanged_from_reviewed_v82')
    for row in prep['files']:
        check_binding(row)
    for row in read(a.HERE / 'OFFLINE_TEST_RECEIPT.json')['files']:
        check_binding(row)
    protected = read(HOME / 'PRESERVATION_BEFORE.json')
    require(len(protected['files'])==4939, 'all_4939_previous_files_protected')
    for path, digest in protected['files'].items():
        require(sha(path) == digest, 'protected_previous_file:' + path)
    for slot in config['multiselect_config']['slots']:
        a.verify(HOME / 'multiselect/assets' / slot['slot_id'], complete=True)
    ordering_result = ordering.verify(HOME)
    ordering_review.verify_review(HOME)
    assembled = c.verify_assembly(HOME / 'multiselect')
    final_review = common.load_multi('evaluate').verify_review(HOME / 'multiselect')
    require(final_review['usable_count'] <= len(assembled['questions']), 'final_review_usable_subset')
    _, _, annotate, capacity, dev = a.base.transport()
    tokenizer = dev.tokenizer_from(str(a.base.TOKENIZER))
    calls, multi_mechanics, order_mechanics = [], [], []
    request_dirs = sorted(HOME.glob('multiselect/assets/*/rounds/*/requests/*')) + sorted(HOME.glob('ordering/*/requests/*'))
    require(len(request_dirs) <= config['max_generation_http'], 'total_generation_request_budget')
    for request in request_dirs:
        is_multi = request.is_relative_to(HOME / 'multiselect')
        round_folder = request.parents[1]
        run = round_folder.parents[1] if is_multi else round_folder
        manifest = read(round_folder / 'MANIFEST.json')
        payload, proof = read(request / 'REQUEST.json'), read(request / 'PROOF.json')
        frozen_payload = read(round_folder / ('INITIAL_REQUEST.json' if is_multi else 'REQUEST.json'))
        frozen_proof = read(round_folder / ('INITIAL_PROOF.json' if is_multi else 'PROOF.json'))
        require(payload == frozen_payload and {k:v for k,v in proof.items() if k != 'capacity'} == frozen_proof,
                'request_frozen_payload_and_proof:' + str(request))
        require(proof['payload_sha256'] == a.base.digest(payload), 'request_payload_sha:' + str(request))
        source = read(run / 'SOURCE.json')
        content = payload['messages'][0]['content']
        context, _ = json.JSONDecoder().raw_decode(content.split('INPUT_JSON\n', 1)[1])
        require(context['source_turns'] == source['turns'] and context['source_metadata'] == {k:v for k,v in source.items() if k != 'turns'},
                'complete_source_context:' + str(request))
        original = Path(manifest['source_history']['path'] if is_multi else manifest['original_source']['path'])
        require((run / 'SOURCE.json').read_bytes() == original.read_bytes(), 'full_source_bytes:' + str(request))
        attempt = read(request / 'HTTP_ATTEMPT.json')
        reservation, completion = read(request / 'RESERVATION.json'), read(request / 'HTTP_RESULT.json')
        require(attempt == {'actual_payload_equals_frozen':True, 'payload_sha256':a.base.digest(payload),
            'requested_model':a.legacy.MODEL, 'http_attempt_number':1, 'transport_retries':0}, 'one_real_http_attempt:' + str(request))
        require(reservation['automatic_retries'] == 0 and reservation['request_number'] == 1, 'reservation_no_retry:' + str(request))
        require(completion['http_status'] == 200 and completion['outcome'] == 'response' and not (request / 'ERROR.json').exists(), 'successful_response_receipt:' + str(request))
        response = read(request / 'RESPONSE.json'); metadata = response['metadata']
        require(json.loads(metadata['raw_response_text']) == metadata['raw_response'] and response['text'] == metadata['raw_response']['choices'][0]['message']['content'],
                'raw_service_text_preserved:' + str(request))
        require(all(metadata[k] == a.legacy.MODEL for k in ['requested_model','actual_model','returned_model'])
                and metadata['http_status'] == 200 and metadata['finish_reason'] == 'stop' and metadata['timeout'] is False,
                'model_and_finish_reason:' + str(request))
        require(metadata['transport_attempt_count'] == metadata['http_attempt_count'] == 1
                and metadata['transport_retry_count'] == metadata['rate_limit_retry_count'] == 0,
                'transport_zero_retries:' + str(request))
        profile = read(request / 'PROFILE.json')
        require(payload == {'model':profile['model'], 'messages':payload['messages'],
                           **{k:v for k,v in profile['request_parameters'].items() if v is not None}}
                and profile['http_retry_limit'] == 0 and profile['parameter_fallback'] is False, 'profile_exact_request:' + str(request))
        cap = annotate(capacity(profile, tokenizer, payload['messages'], payload['max_tokens']))
        require(cap == proof['capacity'] and cap['status'] == 'PASS' and cap['truncation'] is False, 'capacity_no_truncation:' + str(request))
        entry = {'lane':'multi_select' if is_multi else 'ordering', 'root_id':manifest['root_id'],
            'round_id':manifest.get('round_id'), 'source_turn_count':len(source['turns']),
            'request':bind(request / 'REQUEST.json'), 'proof':bind(request / 'PROOF.json'),
            'response':bind(request / 'RESPONSE.json'), 'http_attempt':bind(request / 'HTTP_ATTEMPT.json'),
            'profile':bind(request / 'PROFILE.json'), 'source':bind(run / 'SOURCE.json'),
            'capacity':cap, 'service_reported_usage':metadata['usage'], 'retry_count':0}
        if is_multi:
            targets = manifest['target_fact_ids']; tail = a.render_task_tail(manifest, targets)
            require(content.endswith(tail) and context['target_fact_ids'] == targets and proof['target_fact_ids'] == targets,
                    'rendered_current_target_ids:' + str(request))
            skeleton, _ = json.JSONDecoder().raw_decode(tail.split('OUTPUT_JSON_SKELETON\n',1)[1])
            require([r['fact_id'] for r in skeleton['replacements']] == targets, 'exact_target_skeleton:' + str(request))
            data, errors = a.validate_assets(response['text'], manifest, source, read(run / 'PARENT.json'), targets)
            require(read(request / 'VALIDATION.json') == a.validation(data, errors), 'multiselect_mechanical_replay:' + str(request))
            review = read(round_folder / 'SOURCE_JUDGMENTS.json'); public = read(round_folder / 'PUBLIC_ASSETS.json')
            try:
                raw = a.strict_json(response['text'])
                raw_by_fact = {r['fact_id']:r['replacement_proposition'] for r in raw['replacements']}
                require(all(v['proposition'] == raw_by_fact[v['fact_id']] for v in public['versions']), 'public_asset_text_unedited:' + str(request))
            except (ValueError, TypeError, KeyError):
                require(not public['versions'] and bool(errors), 'malformed_writer_output_remains_excluded:' + str(request))
            multi_mechanics.append({'root_id':manifest['root_id'], 'round_id':manifest['round_id'],
                'target_fact_ids':targets, 'mechanical_errors':errors, 'public_version_count':len(public['versions']),
                'source_review_parent_relations':dict(Counter(r['relation'] for r in review['parents'])),
                'source_review_version_relations':dict(Counter(r['relation'] for r in review['versions'])),
                'source_review':bind(round_folder / 'SOURCE_JUDGMENTS.json'), 'per_version_source_findings':review['versions']})
        else:
            prior = read(run / 'PRIOR.json')
            accepted, rejected = ordering.validate_raw(response['text'],manifest,source,run/'SOURCE.json',prior)
            require(accepted == read(run/'POOL.json')['questions'] and rejected == read(run/'REJECTIONS.json'), 'ordering_mechanical_replay:' + str(request))
            tail = content.split('[ORDERING_BATCH_V83]',1)[1].split('[/ORDERING_BATCH_V83]',1)[0]
            policy=read(run/'ACTOR_REFERENCE_POLICY.json')
            require(manifest['actor_reference_policy']==proof['actor_reference_policy']==bind(run/'ACTOR_REFERENCE_POLICY.json'),
                    'actor_policy_bound_to_manifest_and_executed_request:' + str(request))
            characters=[r for r in registry['characters'] if r['root_id']==manifest['root_id']]
            names=sorted({turn['speaker'] for turn in source['turns']})
            require(sorted(r['canonical_name'] for r in characters)==names
                    and all(r['gender']['value'] is None and r['third_person_pronouns']['value'] is None for r in characters),
                    'actor_registry_root_names_and_unknown_pronouns:' + str(request))
            registry_source=next(r for r in registry['source_bindings'] if r['root_id']==manifest['root_id'])
            require(registry_source['sha256']==manifest['source_sha256']==sha(run/'SOURCE.json'),
                    'actor_registry_source_version:' + str(request))
            expected_policy={'version':actor_policy['version'],'root_id':manifest['root_id'],
                'canonical_source_speaker_names':names,'pronoun_status':'unknown_for_all_source_speakers',
                'output_style':'use_source_names_or_explicit_roles_for_people','forbidden_gendered_pronouns':list(PRONOUNS),
                'scope':'generated item propositions only; use source names or explicit roles for any person',
                'source_evidence_unchanged':True,'new_source_facts':False,'automatic_repair':False}
            listed_policy,_=json.JSONDecoder().raw_decode(tail.split('ACTOR_REFERENCE_POLICY:\n',1)[1])
            require(policy==expected_policy==context['actor_reference_policy']==listed_policy,
                    'independent_actor_policy_projection_in_context_and_tail:' + str(request))
            entry['actor_reference_policy']=bind(run/'ACTOR_REFERENCE_POLICY.json')
            scenes = list(dict.fromkeys(t['turn_ref'].rsplit(':', 1)[0] for t in source['turns']))
            scene_listing, _ = json.JSONDecoder().raw_decode(tail.split('available_scene_ids: ',1)[1])
            prior_listing, _ = json.JSONDecoder().raw_decode(tail.split('DISALLOWED PRIOR WHOLE SETS (public text, deduplication only):\n',1)[1])
            require(scene_listing == scenes and prior_listing['disallowed_prior_six_item_sets'] == [sorted(q['items'].values()) for q in prior], 'scene_and_public_dedup_payload:' + str(request))
            require('"gold"' not in json.dumps(prior_listing) and '"item_evidence"' not in json.dumps(prior_listing), 'ordering_prior_has_no_private_answers:' + str(request))
            plan = read(run/'ANCHOR_PLAN.json')
            require(manifest['anchor_plan'] == proof['anchor_plan'] == bind(run/'ANCHOR_PLAN.json'), 'frozen_anchor_plan_manifest_and_request_binding:' + str(request))
            check_binding(config['ordering_anchor_plan'])
            selected_plans = sorted([p for p in read(config['ordering_anchor_plan']['path'])['plans'] if p['root_id']==manifest['root_id']],key=lambda p:p['slot_number'])
            require(len(selected_plans)==len(manifest['candidate_ids'])==2 and [p['slot_number'] for p in selected_plans]==[1,2], 'independent_anchor_plan_two_slots:' + str(request))
            source_lookup = {t['turn_ref']: t for t in source['turns']}
            source_positions = {t['turn_ref']: i for i,t in enumerate(source['turns'])}
            expected_plans=[]
            for qid,p in zip(manifest['candidate_ids'], selected_plans):
                anchors=p['anchors']; scene=p['scene_id']
                require(len(anchors)==len({v['turn_ref'] for v in anchors})==6 and scene in scenes, 'six_distinct_planned_source_turns:' + qid)
                for anchor in anchors:
                    turn=source_lookup[anchor['turn_ref']]
                    require(anchor['turn_ref'].rsplit(':',1)[0]==scene and anchor['speaker']==turn['speaker']
                        and isinstance(anchor['quote'],str) and bool(anchor['quote'].strip()) and turn['text'].count(anchor['quote'])==1,
                        'independent_exact_single_scene_anchor:' + qid + ':' + anchor['turn_ref'])
                permutation=sorted(anchors,key=lambda v:hashlib.sha256((qid+'\0'+v['turn_ref']+'\0'+v['quote']).encode()).hexdigest())
                expected_plans.append({'candidate_id':qid,'root_id':manifest['root_id'],'scene_id':scene,
                    'question':'Order the statements, using left-to-right clause order within a turn, as they APPEAR IN THE SUPPLIED DIALOGUE within the '+scene+' scene.',
                    'anchors':{label:{k:v[k] for k in ('turn_ref','speaker','quote')} for label,v in zip(('I1','I2','I3','I4','I5','I6'),permutation)}})
            require(plan=={'version':'v81_fixed_single_scene_anchors_v1','plan_source':config['ordering_anchor_plan'],
                'root_id':manifest['root_id'],'source_sha256':manifest['source_sha256'],'questions':expected_plans,
                'semantic_certification':False,'gold_supplied_to_writer':False}, 'independent_plan_reconstruction:' + str(request))
            listed_plans,_=json.JSONDecoder().raw_decode(tail.split('FIXED_SINGLE_SCENE_ANCHOR_PLAN:\n',1)[1])
            require(listed_plans==context['fixed_single_scene_anchor_plan']==expected_plans, 'fixed_anchor_plan_in_complete_context_and_task_tail:' + str(request))
            require('"gold"' not in json.dumps(expected_plans), 'fixed_anchor_plan_has_no_supplied_gold:' + str(request))
            entry['anchor_plan']=bind(run/'ANCHOR_PLAN.json')
            try:
                raw = a.strict_json(response['text']); raw_rows = raw['questions']
            except (ValueError, TypeError, KeyError):
                raw_rows = []
            accepted_by_id = {q['candidate_id']:q for q in accepted}; rejected_by_id = {q['candidate_id']:q for q in rejected}
            for qid, row,planned in zip(manifest['candidate_ids'],raw_rows,expected_plans):
                findings = {'candidate_id':qid, 'writer_null':row is None, 'mechanically_accepted':qid in accepted_by_id,
                    'rejection_reasons':rejected_by_id.get(qid,{}).get('reasons',[]), 'gold_or_items_rewritten':False,
                    'public_scene_and_exact_evidence_bound_before_writer':True,'planned_scene_id':planned['scene_id']}
                if isinstance(row,dict) and isinstance(row.get('items'),dict) and all(isinstance(text,str) for text in row['items'].values()):
                    findings['program_bound_question'] = planned['question']
                    findings['raw_gold'] = row.get('gold')
                    findings['prior_whole_set_matches'] = [q['candidate_id'] for q in prior if ordering.whole_set(q)==ordering.whole_set(row)]
                    style_errors=independent_name_errors(row['items'])
                    findings['name_style_contract_errors']=style_errors
                    if style_errors:
                        require(qid not in accepted_by_id and bool(rejected_by_id[qid]['reasons']),
                                'independent_name_style_failure_is_rejected_without_edits:' + qid)
                    if qid in accepted_by_id:
                        q = accepted_by_id[qid]
                        require(set(row)=={'candidate_id','root_id','items','gold','order_reason'}, 'writer_contains_no_scope_or_evidence_overrides:' + qid)
                        require(row['gold'] == q['gold'] == q['original_gold'] and row['items'] == q['items'] and planned['question']==q['question'], 'ordering_raw_items_and_gold_unchanged_fixed_stem:' + qid)
                        require(not style_errors, 'accepted_ordering_items_follow_name_style_contract:' + qid)
                        require(q['item_evidence']=={label:{**anchor,'status':'statement',
                            'reason':'Program-bound source excerpt chosen before writer invocation; semantic support requires separate full-source review.'}
                            for label,anchor in planned['anchors'].items()}, 'independent_program_bound_exact_evidence:' + qid)
                        chronological=sorted(planned['anchors'],key=lambda label:source_positions[planned['anchors'][label]['turn_ref']])
                        require(row['gold']==chronological, 'independent_source_position_gold_match_not_repair:' + qid)
                        scope = independent_scope(q,source)
                        require(scope == common.load_ordering_gate().ordering_match_scope(q,source), 'independent_public_scope_parser:' + qid)
                        require(scope['kind']=='explicit_scene_union' and scope['scene_refs']==[planned['scene_id']]
                                and all(v['turn_ref'] in scope['matching_turn_refs'] for v in q['generated_order_anchor_spans'].values()), 'generated_anchors_in_fixed_public_single_scene:' + qid)
                        findings['ordering_match_scope'] = scope
                order_mechanics.append(findings)
        calls.append(entry)

    multi = HOME / 'multiselect'; cfg = read(multi/'CONFIG.json')
    prior_paths = {Path(cfg['baseline_pool'])} | {Path(p) for s in cfg['slots'] for p in s.get('prior_pool_paths',[])}
    prior_index = {}
    for path in sorted(prior_paths):
        for original in c._rows(read(path)):
            q = original.get('original_selected_question',original)
            if q.get('question_type') == 'multi_select':
                prior_index.setdefault(normalized_set(q),[]).append({'candidate_id':q['candidate_id'],'prior':bind(path)})
    current_public = read(multi/'PUBLIC_QUESTIONS.json')['questions']
    members = read(multi/'FAMILY_SIDECAR.json')['members']; roles = {r['candidate_id']:r for r in members}
    require(len(current_public) == len(members) and set(roles) == {q['candidate_id'] for q in current_public}, 'public_family_member_identity')
    require(all(r['new_candidate'] == (r['role']=='child') for r in members), 'control_newness_roles')
    dedup, novel, repeated_facts = [], [], []
    for slot in cfg['slots']:
        run = multi/'assets'/slot['slot_id']; public = read(run/'PUBLIC_ASSETS.json')
        variants = {p['fact_id']:[(p['proposition'],True)] for p in public['parent_facts']}
        for v in public['versions']:
            variants[v['fact_id']].append((v['proposition'],False))
        available = {k:[] for k in slot['target_counts']}
        for combination in product(*(variants[fid] for fid in sorted(variants))):
            count = sum(supported for _,supported in combination)
            if count not in available:
                continue
            key = (slot['root_id'],'multi_select',tuple(sorted(' '.join(text.split()).casefold() for text,_ in combination)))
            if len(set(key[2])) != 6:
                continue
            if key in prior_index:
                dedup.append({'root_id':slot['root_id'],'gold_count_from_reviewed_supports':count,
                    'whole_set_digest':c.digest(key),'normalized_statement_set':list(key[2]),
                    'matched_prior_questions':prior_index[key],'excluded_from_new_count':True})
            else:
                available[count].append(key)
        children = [q for q in current_public if q['root_id']==slot['root_id'] and roles[q['candidate_id']]['role']=='child']
        for count, keys in available.items():
            matching = [q for q in children if roles[q['candidate_id']]['gold_count']==count]
            require(len(matching) == bool(keys), 'one_novel_combination_per_available_gold:' + slot['root_id'] + ':' + str(count))
            if not keys:
                continue
            chosen = min(keys,key=c.digest); q=matching[0]
            require(normalized_set(q)==chosen and chosen not in prior_index, 'independent_dedup_min_digest_composition:' + q['candidate_id'])
            require(q['candidate_id']==slot['root_id']+'-v79-'+slot['family_id']+'-'+c.digest(chosen)[:12], 'reused_engine_candidate_identity:' + q['candidate_id'])
            novel.append({'candidate_id':q['candidate_id'],'root_id':q['root_id'],'gold_count_from_reviewed_supports':count,
                'whole_set_digest':c.digest(chosen),'new_exact_six_option_combination':True,'new_independent_facts':False,
                'same_parent_and_source_family':slot['family_id']})
        new = read(run/'rounds/initial/PUBLIC_ASSETS.json')
        for old_batch in ('h4_ready_roots_mixed_batch_v78','h4_output_contract_mixed_batch_v79','h4_scoped_ordering_mixed_batch_v80','h4_bound_scene_ordering_mixed_batch_v81'):
            old_path = HOME.parent/old_batch/'multiselect/assets'/slot['slot_id']/'rounds/initial/PUBLIC_ASSETS.json'
            old_by_fact = {r['fact_id']:r['proposition'] for r in read(old_path)['versions']}
            repeated_facts.append({'root_id':slot['root_id'],'old_public_assets':bind(old_path),
                'new_public_assets':bind(run/'rounds/initial/PUBLIC_ASSETS.json'),
                'identical_per_fact_replacements':[v['fact_id'] for v in new['versions'] if old_by_fact.get(v['fact_id'])==v['proposition']]})
    review = read(HOME/'ordering/CONTENT_REVIEW.json'); review_sources = read(HOME/'ordering/SOURCE_JUDGMENTS.json')
    require(review['accepted_count']+review['rejected_count']==ordering_result['candidate_count'] and review['question_or_gold_edits']==[], 'ordering_content_counts_and_no_edits')
    review_findings = []
    for q in review['questions']:
        source_row = next(r for r in review_sources['questions'] if r['candidate_id']==q['candidate_id'])
        review_findings.append({'candidate_id':q['candidate_id'],'status':q['status'],'reasons':q['reasons'],
            'complete_source_turns_read':source_row['complete_source_turns_read'], 'source_review_reason':source_row.get('reason'),
            'alternative_orders':source_row['alternative_orders'], 'items':source_row['items']})
    initial_count=read(HOME/'BATCH_GENERATION.COMPLETE.json')['http_attempts']
    expected_initial=len(config['multiselect_config']['slots'])+len(config['ordering_slots'])
    multi_calls=read(multi/'GENERATION.COMPLETE.json')['http_attempts']
    ordering_calls=read(HOME/'ordering/GENERATION.COMPLETE.json')['http_attempts']
    supplement_calls=sum(row['round_id']=='supplement' for row in calls)
    require(initial_count == expected_initial and multi_calls+ordering_calls==len(calls) and len(calls)-initial_count==supplement_calls,
            'initial_snapshot_plus_explicit_supplements')
    for slot in cfg['slots']:
        require(sum(r['root_id']==slot['root_id'] and r['round_id']=='supplement' for r in calls)<=1, 'supplement_family_budget:' + slot['root_id'])
    controls=sum(r['role']=='control6' for r in members)
    shortfalls=bool(len(novel)<sum(len(s['target_counts']) for s in cfg['slots']) or ordering_result['candidate_count']<sum(s['candidate_count'] for s in config['ordering_slots']) or review['rejected_count'] or final_review['usable_count']<len(assembled['questions']))
    return {'status':'PASS_WITH_GENERATION_AND_CONTENT_SHORTFALLS' if shortfalls else 'PASS',
        'totals':{'actual_generation_http_calls':len(calls),'initial_batch_snapshot_http_calls':initial_count,
            'explicit_bounded_supplement_http_calls':supplement_calls,'new_multiselect_six_option_combinations':len(novel),
            'historical_controls':controls,'multiselect_assembled':len(assembled['questions']),'multiselect_content_usable':final_review['usable_count'],
            'ordering_slot_count':sum(s['candidate_count'] for s in config['ordering_slots']),
            'ordering_mechanical_accepted':ordering_result['candidate_count'],'ordering_content_usable':review['accepted_count'],
            'ordering_content_rejected':review['rejected_count'],'protected_previous_file_count':len(protected['files'])},
        'calls':calls,'multiselect_rounds':multi_mechanics,'ordering_mechanical_findings':order_mechanics,
        'ordering_source_review_findings':review_findings,'deduplicated_combinations':dedup,'new_combinations':novel,
        'repeated_per_fact_replacement_texts':repeated_facts,
        'evidence':[bind(multi/'PUBLIC_QUESTIONS.json'),bind(multi/'ALLOCATION.json'),bind(multi/'FAMILY_SIDECAR.json'),
            bind(HOME/'ordering/CONTENT_REVIEW.json'),bind(HOME/'ordering/SOURCE_JUDGMENTS.json'),bind(HOME/'BATCH_PREPARATION.COMPLETE.json')],
        'limitations':['This audit replays records and exact-set composition, not source answer semantics or semantic novelty.',
            'The pronoun token rule is a generation style contract; passing it does not establish correct actor attribution, facts or order.',
            'New multiselect combinations retain parent/source dependencies; the audit does not certify independent facts.',
            'Fixed single-scene anchors prevent writer binding edits, but do not certify proposition support or unique semantic order.',
            'Explicit scene scope affects only matching occurrences; complete source remains in every generation request.',
            'Capacity uses the frozen local proxy; service token usage is reported separately.']}


def main():
    if OUT.exists():
        raise RuntimeError('refuse_overwrite:' + str(OUT))
    report = {'version':'v83_independent_generation_audit_v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'disclosures':{'audit_generation_http_calls':0,'audit_solver_calls':0,'network_and_subprocess_blocked':True,
            'generation_private_and_raw_read':True,'current_final_multiselect_private_pool_read':True,
            'current_solver_outputs_read':False,'frozen_code_modified':False,'blind_semantic_review_claimed':False}}
    try:
        report.update(audit())
    except BaseException as error:
        report['status']='FAIL'
        CHECKS.append({'check':'audit_execution_exception','status':'FAIL','detail':{'error':repr(error),'traceback':traceback.format_exc()}})
    report['checks']=CHECKS
    report['check_count']=len(CHECKS)
    report['failed_checks']=[row for row in CHECKS if row['status']=='FAIL']
    report['opened_current_batch_files']=sorted(OPENED)
    report['auditor']=bind(__file__)
    with OUT.open('x') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps({'path':str(OUT),'status':report['status'],'checks':len(CHECKS),'failed_checks':report['failed_checks']},ensure_ascii=False))
    if report['status']=='FAIL':
        raise SystemExit(1)


if __name__=='__main__':
    main()
