"""One full-source writer request per root; keep every rejected draft unchanged."""
from copy import deepcopy
from pathlib import Path
import json

from common import HERE, aware, bind, load_multi, load_ordering_gate, need, now, read, require_bindings, sha, verify_bindings, write

VERSION = 'v83_name_policy_ordering_writer_v1'
import ordering_plan
import actor_reference
LABELS = tuple(f'I{i}' for i in range(1, 7))
TASK = """Write the supplied two ordering slots from the COMPLETE dialogue and the
FIXED_SINGLE_SCENE_ANCHOR_PLAN below. Each slot already has one public scene,
six evidence anchors mapped to I1..I6, and an immutable question. Do not select
another scene, another anchor, or another six-item set. Read the full dialogue
for actor identity, intended meaning, modality and action state. For each label
write ONE faithful proposition corresponding to that label's fixed quote.
Keep the speaker name explicit where needed; preserve plans, questions,
suggestions and completed actions. Never add unsupported gender, activity,
motive, certainty or completion. Check all other appearances within the fixed
public scene; all plausible matching occurrences must preserve one strict order.
Use source names instead of unsupported gendered pronouns. Don't copy a whole
historical six-item set or paraphrase the same six statements. Return null if
the fixed anchors cannot support a reliable distinct question. Do not satisfy a
quota at the expense of fidelity. The anchors are not a semantic certificate.
Order statements by where they APPEAR IN THE SUPPLIED DIALOGUE, not by the times
of reported past events. Independently return gold as the six label permutation
in source appearance order. A wrong permutation will be rejected, never fixed.
Return JSON ONLY {"questions":[...]} in the supplied candidate_ids order.
Each slot is null or exactly {"candidate_id":"supplied id","root_id":"supplied root",
"items":{"I1":"...","I2":"...","I3":"...","I4":"...","I5":"...","I6":"..."},
"gold":[six item labels],"order_reason":"brief"}.
Do not return question, scene, evidence, source refs, quotes, annotations or
additional fields: those are bound by the program BEFORE this request.
The complete source and frozen previous-training-round feedback remain context.
Current Solver scores are absent. No automatic repair or retry.
ACTOR REFERENCE OUTPUT CONTRACT: Every generated item must refer to people by
their source names or explicit source roles. The source speakers' gender and
third-person pronouns are unknown in the frozen registry. Use natural clauses
with names, including possessive names where necessary. Do not use any of these
standalone words in item propositions: he, him, his, himself, she, her, hers,
herself. This is a writer style constraint, not a new source fact. Check all six
items before returning; a violating draft will be rejected without edits.
"""


def engine():
    return load_multi('asset_generation')


def whole_set(row):
    return tuple(sorted(' '.join(t.split()).casefold() for t in row['items'].values()))


def public(row):
    return {k: deepcopy(row[k]) for k in ('candidate_id', 'root_id', 'question_type',
            'chronology_kind', 'question', 'items', 'source_path', 'source_sha256')}


def candidate_ids(config, slot):
    return [slot['root_id'] + '-' + config['batch_id'] + f'-ordering-{i:02d}'
            for i in range(1, slot['candidate_count'] + 1)]


def payload_for(run):
    run = Path(run); m = read(run / 'MANIFEST.json')
    source = read(run / 'SOURCE.json')
    ctx = {'root_id': m['root_id'], 'candidate_ids': m['candidate_ids'],
           'fixed_single_scene_anchor_plan': read(run / 'ANCHOR_PLAN.json')['questions'],
           'actor_reference_policy': read(run / 'ACTOR_REFERENCE_POLICY.json'),
           'source_turns': source['turns'], 'source_metadata': {k:v for k,v in source.items() if k != 'turns'},
           'previous_public_questions_dedup_only': read(run / 'PRIOR.json'),
           'source_mode': 'complete_frozen_sources', 'training_allowed': False}
    e = engine()
    payload, receipt = e.legacy.legacy.payload_with_feedback(run, m['root_id'],
            'Read the complete frozen source and historical feedback.\n', ctx, m['max_tokens'])
    prior_sets = [sorted(q['items'].values()) for q in read(run / 'PRIOR.json')]
    scenes = list(dict.fromkeys(t['turn_ref'].rsplit(':', 1)[0] for t in source['turns']))
    tail = TASK + '\nFIXED_SINGLE_SCENE_ANCHOR_PLAN:\n' + json.dumps(read(run / 'ANCHOR_PLAN.json')['questions'], ensure_ascii=False) + '\navailable_scene_ids: ' + json.dumps(scenes, ensure_ascii=False) + '\nDISALLOWED PRIOR WHOLE SETS (public text, deduplication only):\n' + json.dumps({'disallowed_prior_six_item_sets': prior_sets}, ensure_ascii=False)
    tail += '\nACTOR_REFERENCE_POLICY:\n' + json.dumps(read(run / 'ACTOR_REFERENCE_POLICY.json'), ensure_ascii=False)
    payload['messages'][0]['content'] += '\n[ORDERING_BATCH_V83]\n' + tail + '\n[/ORDERING_BATCH_V83]'
    return payload, {'role': 'ordering_writer', 'source_sha256': m['source_sha256'],
            'candidate_ids': m['candidate_ids'], 'anchor_plan': bind(run / 'ANCHOR_PLAN.json'),
            'actor_reference_policy': bind(run / 'ACTOR_REFERENCE_POLICY.json'), 'feedback_receipt': receipt,
            'payload_sha256': e.base.digest(payload), 'current_solver_feedback': False}


def capacity_for(run, tokenizer):
    """Replay the transport's local capacity guard without constructing a caller."""
    run = Path(run); payload = read(run / 'REQUEST.json'); profile = read(run / 'provider_profile.json')
    profile['output_limit'] = payload['max_tokens']
    profile['request_parameters']['max_tokens'] = payload['max_tokens']
    expected = {'model': profile['model'], 'messages': payload['messages'],
                **{k: v for k, v in profile['request_parameters'].items() if v is not None}}
    need(payload == expected, 'ordering_request_profile_payload_mismatch')
    _, _, annotate, capacity, _ = engine().base.transport()
    return annotate(capacity(profile, tokenizer, payload['messages'], payload['max_tokens']))


def prepare(home=HERE):
    home = Path(home).resolve(); config = read(home / 'CONFIG.json'); root = home / 'ordering'
    slots = config['ordering_slots']
    need(slots and len({s['root_id'] for s in slots}) == len(slots)
         and len({s['slot_id'] for s in slots}) == len(slots)
         and all(s['candidate_count'] == 2 for s in slots), 'ordering_slots_invalid')
    need(not root.exists(), 'ordering_already_prepared')
    e = engine(); provider, dependencies = e._dependencies(); rows = []
    _, _, _, _, dev = e.base.transport()
    tokenizer = dev.tokenizer_from(str(e.base.TOKENIZER))
    for slot in slots:
        run = root / slot['slot_id']; source_path = Path(slot['source_path'])
        need(sha(source_path) == slot['source_sha256'] and sha(slot['history_path']) == slot['history_sha256'], 'ordering_source_changed')
        source = read(source_path)
        errors, _, _ = load_ordering_gate()._source(source)
        need(not errors and source['root_id'] == slot['root_id']
             and source['turns'] == read(slot['history_path'])['turns'], 'ordering_requires_complete_source')
        feedback = e.source_adapter.feedback_data(slot['root_id'])
        need(feedback['source_lock']['merge_current_generation_feedback'] is False
             and feedback['source_lock']['status'] == 'EXPLICIT_PREVIOUS_TRAINING_ROUND_ONLY', 'feedback_not_previous_round')
        feedback, files = e.source_adapter.freeze_feedback(slot['root_id'], run / 'feedback', data=feedback)
        prior_paths = slot.get('prior_pool_paths', [slot['prior_pool_path']])
        prior = [public(q) for path in prior_paths for q in read(path)['questions']
                 if q['root_id'] == slot['root_id'] and q['question_type'] == 'ordering']
        write(run / 'PRIOR.json', prior); write(run / 'provider_profile.json', provider)
        anchor_plan = ordering_plan.build(config, slot, source, candidate_ids(config, slot))
        write(run / 'ANCHOR_PLAN.json', anchor_plan)
        write(run / 'ACTOR_REFERENCE_POLICY.json', actor_reference.policy_for(config, slot, source))
        with (run / 'SOURCE.json').open('xb') as f:
            f.write(source_path.read_bytes())
        dependencies_for_slot = set(dependencies) | set(files) | set(HERE.glob('*.py')) | set(map(Path, prior_paths)) | {
            home / 'CONFIG.json', Path(config['ordering_anchor_plan']['path']), run / 'ANCHOR_PLAN.json',
            Path(config['actor_reference_policy']['path']), Path(config['character_registry']['path']), run / 'ACTOR_REFERENCE_POLICY.json',
            source_path, Path(slot['history_path']), Path(slot['prior_pool_path']),
            run / 'SOURCE.json', run / 'PRIOR.json', run / 'provider_profile.json'}
        m = {'version': VERSION, 'root_id': slot['root_id'], 'slot_id': slot['slot_id'],
             'candidate_ids': candidate_ids(config, slot), 'anchor_plan': bind(run / 'ANCHOR_PLAN.json'),
             'actor_reference_policy': bind(run / 'ACTOR_REFERENCE_POLICY.json'), 'prepared_at_utc': now(),
             'source_sha256': sha(run / 'SOURCE.json'), 'source': bind(run / 'SOURCE.json'),
             'original_source': bind(source_path), 'full_source_turn_count': len(source['turns']),
             'config': bind(home / 'CONFIG.json'), 'max_tokens': config['ordering_output_max_tokens'],
             'solver_model': e.legacy.TARGET, 'generator_model': e.legacy.MODEL,
             'budget': {'http_attempts_max': 1}, 'training_allowed': False,
             'current_solver_feedback': False, 'files': [bind(p) for p in sorted(dependencies_for_slot)]}
        write(run / 'MANIFEST.json', m)
        payload, proof = payload_for(run); write(run / 'REQUEST.json', payload); write(run / 'PROOF.json', proof)
        capacity = capacity_for(run, tokenizer); write(run / 'CAPACITY.json', capacity)
        need(capacity['status'] == 'PASS', 'ordering_capacity_blocked:' + slot['slot_id'])
        rows.append({'slot_id': slot['slot_id'], 'root_id': slot['root_id'],
                     'manifest': bind(run / 'MANIFEST.json'), 'capacity': bind(run / 'CAPACITY.json')})
    write(root / 'PREPARATION.COMPLETE.json', {'version': VERSION, 'completed_at_utc': now(), 'slots': rows, 'http_attempts': 0})
    verify(home); return rows


def validate_raw(raw, manifest, source, source_path, prior):
    """Never turn a writer Gold mismatch into a corrected question."""
    ids = manifest['candidate_ids']; accepted = []; rejected = []; seen = {whole_set(q) for q in prior}
    try:
        obj = engine().strict_json(raw)
        need(isinstance(obj, dict) and set(obj) == {'questions'} and isinstance(obj['questions'], list)
             and len(obj['questions']) == len(ids), 'ordering_response_schema')
        rows = obj['questions']
    except (ValueError, TypeError) as error:
        return [], [{'candidate_id': qid, 'reasons': [str(error)]} for qid in ids]
    plan_path = Path(source_path).parent / 'ANCHOR_PLAN.json'
    need(bind(plan_path) == manifest['anchor_plan'], 'ordering_anchor_plan_changed')
    plans = {p['candidate_id']: p for p in read(plan_path)['questions']}
    need(set(plans) == set(ids), 'ordering_anchor_plan_identity')
    for qid, row in zip(ids, rows):
        if row is None:
            rejected.append({'candidate_id': qid, 'reasons': ['writer_null_slot']}); continue
        try:
            row = ordering_plan.project(row, plans[qid])
        except (ValueError, TypeError, KeyError) as error:
            rejected.append({'candidate_id': qid, 'reasons': [str(error)]}); continue
        name_errors = actor_reference.name_policy_errors(row['items'])
        if name_errors:
            rejected.append({'candidate_id': qid, 'reasons': name_errors}); continue
        errors, candidate = load_ordering_gate().validate_generated_ordering(row, qid, manifest['root_id'], source, source_path)
        if not errors and candidate['gold'] != row['gold']:
            errors = ['writer_gold_disagrees_with_source_anchors']
        if not errors and whole_set(candidate) in seen:
            errors = ['duplicate_prior_or_current_six_item_set']
        if errors:
            rejected.append({'candidate_id': qid, 'reasons': errors}); continue
        seen.add(whole_set(candidate)); candidate['generator_model'] = engine().legacy.MODEL
        candidate['status'] = 'MECHANICAL_ONLY_REQUIRES_SOURCE_FIRST_REVIEW'
        accepted.append(candidate)
    return accepted, rejected


def verify(home=HERE):
    home = Path(home).resolve(); config = read(home / 'CONFIG.json'); root = home / 'ordering'
    expected = {s['slot_id'] for s in config['ordering_slots']}
    need({p.name for p in root.iterdir() if p.is_dir()} == expected, 'ordering_family_coverage')
    preparation = read(root / 'PREPARATION.COMPLETE.json')
    need(preparation['version'] == VERSION and preparation['http_attempts'] == 0
         and preparation['slots'] == [{'slot_id': s['slot_id'], 'root_id': s['root_id'],
              'manifest': bind(root / s['slot_id'] / 'MANIFEST.json'),
              'capacity': bind(root / s['slot_id'] / 'CAPACITY.json')} for s in config['ordering_slots']]
         and aware(preparation['completed_at_utc']) <= aware(now()), 'ordering_preparation_coverage')
    e = engine(); _, _, _, _, dev = e.base.transport()
    tokenizer = dev.tokenizer_from(str(e.base.TOKENIZER))
    rows = []; totals = 0
    for slot in config['ordering_slots']:
        run = root / slot['slot_id']; m = read(run / 'MANIFEST.json')
        need(m['version'] == VERSION and m['root_id'] == slot['root_id'] and m['slot_id'] == slot['slot_id']
             and m['candidate_ids'] == candidate_ids(config, slot) and m['budget'] == {'http_attempts_max': 1}
             and m['solver_model'] == e.legacy.TARGET and m['generator_model'] == e.legacy.MODEL
             and m['training_allowed'] is False and m['current_solver_feedback'] is False,
             'ordering_manifest_contract_changed')
        paths = [home / 'CONFIG.json', Path(config['ordering_anchor_plan']['path']), run / 'ANCHOR_PLAN.json',
                 Path(config['actor_reference_policy']['path']), Path(config['character_registry']['path']), run / 'ACTOR_REFERENCE_POLICY.json',
                 Path(slot['source_path']), Path(slot['history_path']), Path(slot['prior_pool_path']),
                 run / 'SOURCE.json', run / 'PRIOR.json', run / 'provider_profile.json'] + list(HERE.glob('*.py')) + list(map(Path, slot.get('prior_pool_paths', [slot['prior_pool_path']])))
        require_bindings(m['files'], set(paths))
        need(m['config'] == bind(home / 'CONFIG.json') and m['source'] == bind(run / 'SOURCE.json')
             and m['original_source'] == bind(slot['source_path']) and m['source_sha256'] == slot['source_sha256']
             and m['max_tokens'] == config['ordering_output_max_tokens']
             and m['full_source_turn_count'] == len(read(run / 'SOURCE.json')['turns'])
             and aware(m['prepared_at_utc']) <= aware(preparation['completed_at_utc']), 'ordering_manifest_binding')
        need(read(run / 'ANCHOR_PLAN.json') == ordering_plan.build(config, slot, read(run / 'SOURCE.json'), candidate_ids(config, slot))
             and m['anchor_plan'] == bind(run / 'ANCHOR_PLAN.json'), 'ordering_anchor_plan_changed')
        need(read(run / 'ACTOR_REFERENCE_POLICY.json') == actor_reference.policy_for(config, slot, read(run / 'SOURCE.json'))
             and m['actor_reference_policy'] == bind(run / 'ACTOR_REFERENCE_POLICY.json'), 'ordering_actor_reference_policy_changed')
        payload, proof = payload_for(run)
        need(read(run / 'REQUEST.json') == payload and read(run / 'PROOF.json') == proof, 'ordering_request_changed')
        capacity = read(run / 'CAPACITY.json')
        need(capacity['status'] == 'PASS' and capacity == capacity_for(run, tokenizer), 'ordering_capacity_changed')
        requests = list(run.glob('requests/*/REQUEST.json')); attempts = list(run.glob('requests/*/HTTP_ATTEMPT.json'))
        need(len(requests) <= 1 and len(attempts) <= 1, 'ordering_http_budget_exceeded'); totals += len(attempts)
        for request in requests:
            need(read(request) == payload, 'ordering_actual_request_changed')
            actual_proof = read(request.parent / 'PROOF.json')
            need(all(actual_proof.get(k) == v for k,v in proof.items()), 'ordering_actual_proof_changed')
        for attempt in attempts:
            a = read(attempt)
            need((attempt.parent / 'REQUEST.json').exists()
                 and a == {'actual_payload_equals_frozen': True, 'payload_sha256': engine().base.digest(payload),
                           'requested_model': engine().legacy.MODEL, 'http_attempt_number': 1, 'transport_retries': 0},
                 'ordering_http_receipt_changed')
        if (run / 'GENERATION.COMPLETE.json').exists():
            c = read(run / 'GENERATION.COMPLETE.json')
            paths = [run / name for name in ('MANIFEST.json', 'SOURCE.json', 'PRIOR.json', 'REQUEST.json', 'PROOF.json',
                     'CAPACITY.json', 'GENERATION.STARTED.json', 'POOL.json', 'PUBLIC_QUESTIONS.json', 'REJECTIONS.json')]
            paths += [p for p in (run / 'requests').rglob('*') if p.is_file()]
            require_bindings(c['files'], paths)
            need(c['version'] == VERSION and c['training_allowed'] is False and c['semantic_review_performed'] is False
                 and aware(preparation['completed_at_utc']) <= aware(read(run / 'GENERATION.STARTED.json')['started_at_utc'])
                 <= aware(c['completed_at_utc']) <= aware(now()), 'ordering_slot_completion_contract')
            need((run / 'GENERATION.STARTED.json').exists() and len(requests) == 1, 'ordering_not_executed')
            request_dir = requests[0].parent; raw = None
            if not (request_dir / 'ERROR.json').exists():
                response = read(request_dir / 'RESPONSE.json'); meta = response['metadata']; raw = response['text']
                need(len(attempts) == 1 and meta.get('returned_model') == engine().legacy.MODEL
                     and meta.get('finish_reason') == 'stop' and meta.get('http_status', 200) == 200, 'ordering_response_metadata')
            accepted, rejected = validate_raw(raw, m, read(run / 'SOURCE.json'), run / 'SOURCE.json', read(run / 'PRIOR.json'))
            need(read(run / 'POOL.json')['questions'] == accepted and read(run / 'REJECTIONS.json') == rejected
                 and read(run / 'PUBLIC_QUESTIONS.json')['questions'] == [public(q) for q in accepted]
                 and c['candidate_count'] == len(accepted) and c['http_attempts'] == len(attempts), 'ordering_generation_replay_changed')
            rows.extend(accepted)
    need(totals <= len(expected), 'ordering_global_http_budget')
    if (root / 'GENERATION.COMPLETE.json').exists():
        c = read(root / 'GENERATION.COMPLETE.json')
        paths = [root / name for name in ('PREPARATION.COMPLETE.json', 'GENERATION.STARTED.json', 'POOL.json', 'PUBLIC_QUESTIONS.json')]
        for sid in expected:
            child = read(root / sid / 'GENERATION.COMPLETE.json')
            paths += [root / sid / 'GENERATION.COMPLETE.json'] + [Path(r['path']) for r in child['files']]
        require_bindings(c['files'], paths)
        need(c['version'] == VERSION and c['status'] == 'GENERATED' and c['training_allowed'] is False,
             'ordering_global_completion_contract')
        need(all((root / s / 'GENERATION.COMPLETE.json').exists() for s in expected), 'ordering_incomplete_family')
        need(read(root / 'POOL.json')['questions'] == rows and read(root / 'PUBLIC_QUESTIONS.json')['questions'] == [public(q) for q in rows]
             and c['candidate_count'] == len(rows) and c['http_attempts'] == totals, 'ordering_global_projection_changed')
        need(aware(c['completed_at_utc']) >= max(aware(read(root / s / 'GENERATION.COMPLETE.json')['completed_at_utc']) for s in expected), 'ordering_completion_order')
    return {'candidate_count': len(rows), 'http_attempts': totals}


def generate(home=HERE):
    home = Path(home).resolve(); root = home / 'ordering'; verify(home)
    need(not (root / 'GENERATION.STARTED.json').exists(), 'ordering_no_automatic_replay')
    write(root / 'GENERATION.STARTED.json', {'started_at_utc': now(), 'automatic_retry': False})
    rows = []; e = engine()
    for slot in read(home / 'CONFIG.json')['ordering_slots']:
        run = root / slot['slot_id']; m = read(run / 'MANIFEST.json'); payload = read(run / 'REQUEST.json'); proof = read(run / 'PROOF.json')
        need(not (run / 'GENERATION.STARTED.json').exists(), 'ordering_slot_no_replay')
        write(run / 'GENERATION.STARTED.json', {'started_at_utc': now(), 'automatic_retry': False})
        with e.qa_transport.shared_lock():
            raw, request, _ = e.qa_transport.Caller(run).call(slot['slot_id'], payload, proof)
        accepted, rejected = validate_raw(raw, m, read(run / 'SOURCE.json'), run / 'SOURCE.json', read(run / 'PRIOR.json'))
        write(run / 'POOL.json', {'questions': accepted}); write(run / 'REJECTIONS.json', rejected)
        write(run / 'PUBLIC_QUESTIONS.json', {'questions': [public(q) for q in accepted]})
        write(run / 'GENERATION.COMPLETE.json', {'version': VERSION, 'completed_at_utc': now(), 'candidate_count': len(accepted),
             'http_attempts': len(list(run.glob('requests/*/HTTP_ATTEMPT.json'))), 'semantic_review_performed': False,
             'training_allowed': False, 'files': [bind(p) for p in sorted(run.rglob('*')) if p.is_file()]})
        rows.extend(accepted)
    provenance = {'requested_model': e.legacy.MODEL, 'writer': VERSION, 'semantic_review_performed': False}
    write(root / 'POOL.json', {'questions': rows, 'provenance': provenance, 'training_allowed': False})
    write(root / 'PUBLIC_QUESTIONS.json', {'questions': [public(q) for q in rows]})
    result = {'version': VERSION, 'status': 'GENERATED', 'completed_at_utc': now(), 'candidate_count': len(rows),
         'http_attempts': len(list(root.glob('*/requests/*/HTTP_ATTEMPT.json'))), 'training_allowed': False,
         'files': [bind(p) for p in sorted(root.rglob('*')) if p.is_file()]}
    write(root / 'GENERATION.COMPLETE.json', result); verify(home); return result
