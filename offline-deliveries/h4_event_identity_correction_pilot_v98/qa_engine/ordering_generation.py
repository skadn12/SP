"""One full-source writer request per root; keep every rejected draft unchanged."""
from copy import deepcopy
from pathlib import Path
import json

from common import HERE, aware, bind, load_multi, load_ordering_gate, need, now, read, require_bindings, require_shard, sha, verify_bindings, write

VERSION = 'v98_single_event_supported_records_ordering_writer_v1'
import ordering_plan
import actor_reference
import ordering_prior
LABELS = tuple(f'I{i}' for i in range(1, 7))
TASK = """Write a predicate for each supplied source anchor in the two question slots.
The COMPLETE dialogue and SOURCE_ID_ANCHOR_RECORDS below are immutable context.
Each record contains its original anchor_id, speaker, quote and statement_prefix.
The quote is the COMPLETE text of one original source turn, frozen before writing.
Each slot contains six distinct turns in one scene. Write one unambiguous statement
per turn; preserve the context and never borrow an event from another turn.
Select ONE event or speech act inside that complete turn, not a summary of every
clause. Return support_quote as one exact contiguous nonempty substring of the
supplied quote supporting that ONE event. Never splice separate clauses from
different turns into support_quote. The full original turn remains frozen and
is still the ordering anchor; the excerpt does not replace or shorten context.
Read the full dialogue to resolve actors, meaning, modality and action state.
For EACH anchor, write only the predicate that follows its statement_prefix.
The program assembles the complete item as statement_prefix + predicate.
Start the predicate with a verb appropriate to that named speaker: for example,
"suggests trying a new recipe with Morgan." is a predicate after "Alex ".
The example is grammar only, not a source fact. Do not repeat the initial name.
Every predicate must describe ONLY the ONE event in its OWN support_quote.
Do not merge different subjects, causes, benefits or action states. For example,
if a turn says a map helps navigation and the visitor center has updates, write
only that the speaker explains the map helps navigation OR only that the speaker
says the visitor center has updates. Never say the map provides both benefits.
This is an unrelated grammar illustration, not a current source fact. Repeat that exact
anchor_id beside its predicate. Never replace an anchor with a neighboring event.
Array position does not assign an item label. Display labels and the final answer
are private program data; do not create labels, number events or output an answer.
Preserve whether the speaker asks, suggests, promises, reports or completes an
action. Use a reporting predicate when a quote describes someone else's action.
Check the assembled statement for actor identity, negation, state and modality.
ACTOR NAME CONTRACT: The initial speaker name is already supplied by the program.
Inside the predicate use source names or explicit source roles for ALL people.
Repeat names where necessary; do not switch to a gendered pronoun. The standalone
words he, him, his, himself, she, her, hers, herself are forbidden in predicates.
Names and known source roles are allowed; never infer gender from a person's name.
For possessives write the relevant name's possessive, not his or her.
Resolve I/me/my to the named speaker, and you/your to a source-grounded
addressee name or explicit source role. For example a speaker Alex saying
"I appreciate your help" can become "thanks Morgan for Morgan's help."
Never use "says he appreciates her help". Names may repeat; fluency is secondary.
Before emitting JSON, scan the predicate token by token and remove every
forbidden pronoun by authoring a fully named statement. Do not emit an initial
draft or ask the program to replace pronouns. If a source-grounded actor cannot
be resolved, return null for the slot. Check each predicate before returning. Violations are
rejected without replacing or editing the words.
All six statements must correspond uniquely to their own source anchors within
that slot's public scene. Read other occurrences in that scene for ambiguity.
Don't copy or paraphrase a historical whole six-statement set. Return null for
an entire slot if the fixed anchors cannot support reliable distinct statements.
Keep every supplied anchor_id exactly once per non-null slot. No missing, duplicate
or invented IDs. Never move a predicate to another anchor_id. Record order is free.
Return JSON ONLY {"questions":[...]} in the supplied candidate_ids order.
Each slot is null or exactly {"candidate_id":"supplied id","root_id":"supplied root",
"records":[{"anchor_id":"exact supplied anchor_id","support_quote":"exact contiguous source substring","predicate":"one event, verb phrase with names only..."},...]}
with exactly six records. Do not return items, labels, Gold, order_reason, speaker,
statement_prefix, complete quotes, annotations, evidence or additional fields.
The only quote field allowed is the required support_quote inside each record.
Final preflight for every record: support_quote is exact inside its OWN full
quote; predicate describes one event with matching actor/state/modality; no
forbidden gendered pronouns; anchor_id unchanged. Otherwise return slot null.
The program binds each record by its anchor_id to labels fixed before the request.
It does not move statements or repair answers to rescue a meaning mismatch.
The full source and previous-training-round feedback remain context. Current
Solver scores are absent. No automatic repair or retry.
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
           'source_id_anchor_records': {'location': 'SOURCE_ID_ANCHOR_RECORDS in the task tail; complete immutable records appear there once.'},
           'actor_reference_policy': {'location': 'ACTOR_REFERENCE_POLICY in the task tail; complete frozen policy appears there once.'},
           'source_turns': source['turns'], 'source_metadata': {k:v for k,v in source.items() if k != 'turns'},
           'previous_public_questions_dedup_only': {
               'whole_set_text_location': 'PUBLIC HISTORICAL ORDERING QUESTIONS in the task tail; all historical stems and item texts appear, including sets with fewer than six items.',
               'question_index': [{k:q[k] for k in ('candidate_id','root_id','question_type')}
                                  for q in read(run / 'PRIOR.json')],
               'public_prior_coverage': {k: read(run / 'PRIOR_COVERAGE.json')[k] for k in (
                   'historical_candidate_count', 'historical_ordering_candidate_count',
                   'historical_ordering_candidate_ids', 'supplied_public_prior_rows',
                   'all_historical_ordering_rows_retained', 'historical_item_counts',
                   'historical_gold_or_evidence_supplied')}},
           'source_mode': 'complete_frozen_sources', 'training_allowed': False}
    e = engine()
    payload, receipt = e.legacy.legacy.payload_with_feedback(run, m['root_id'],
            'Read the complete frozen source and historical feedback.\n', ctx, m['max_tokens'])
    prior_public = [{k: q[k] for k in ('candidate_id', 'root_id', 'question_type', 'question', 'items')}
                    for q in read(run / 'PRIOR.json')]
    scenes = list(dict.fromkeys(t['turn_ref'].rsplit(':', 1)[0] for t in source['turns']))
    tail = TASK + '\nSOURCE_ID_ANCHOR_RECORDS:\n' + json.dumps(read(run / 'WRITER_PLAN.json')['questions'], ensure_ascii=False, separators=(',', ':')) + '\navailable_scene_ids: ' + json.dumps(scenes, ensure_ascii=False, separators=(',', ':')) + '\nPUBLIC HISTORICAL ORDERING QUESTIONS (deduplication only, never evidence):\n' + json.dumps({'public_ordering_questions': prior_public}, ensure_ascii=False, separators=(',', ':'))
    tail += '\nACTOR_REFERENCE_POLICY:\n' + json.dumps(read(run / 'ACTOR_REFERENCE_POLICY.json'), ensure_ascii=False, separators=(',', ':'))
    payload['messages'][0]['content'] += '\n[ORDERING_BATCH_V89]\n' + tail + '\n[/ORDERING_BATCH_V89]'
    return payload, {'role': 'ordering_writer', 'source_sha256': m['source_sha256'],
            'candidate_ids': m['candidate_ids'], 'writer_plan': bind(run / 'WRITER_PLAN.json'),
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
    home = require_shard(home); config = read(home / 'CONFIG.json'); root = home / 'ordering'
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
        prior_paths = ordering_prior.configured_paths(slot)
        compact, compact_files = e.source_adapter.prior_candidates(slot['root_id'], feedback,
            run / 'feedback' / slot['root_id'] / 'candidate_history.json', e.v15)
        prior, prior_receipt, prior_files = ordering_prior.collect(
            slot['root_id'], run / 'feedback' / slot['root_id'] / 'candidate_history.json', prior_paths,
            compact_rows=compact, compact_files=compact_files)
        write(run / 'PRIOR_COVERAGE.json', prior_receipt)
        write(run / 'PRIOR.json', prior); write(run / 'provider_profile.json', provider)
        anchor_plan = ordering_plan.build(config, slot, source, candidate_ids(config, slot))
        write(run / 'ANCHOR_PLAN.json', anchor_plan)
        write(run / 'WRITER_PLAN.json', ordering_plan.writer_plan(source, anchor_plan))
        write(run / 'ACTOR_REFERENCE_POLICY.json', actor_reference.policy_for(config, slot, source))
        with (run / 'SOURCE.json').open('xb') as f:
            f.write(source_path.read_bytes())
        write(run / 'GOLD_PLAN.json', {'version': 'v98_program_gold_plan_v1', 'created_at_utc': now(),
              'source': bind(run / 'SOURCE.json'), 'anchor_plan': bind(run / 'ANCHOR_PLAN.json'),
              'gold_origin': ordering_plan.GOLD_ORIGIN, 'semantic_certification': False,
              'gold_supplied_to_writer': False, 'questions': ordering_plan.gold_questions(source, anchor_plan)})
        dependencies_for_slot = set(dependencies) | set(files) | set(prior_files) | set(HERE.glob('*.py')) | set(map(Path, prior_paths)) | {
            home / 'CONFIG.json', Path(config['ordering_anchor_plan']['path']), run / 'ANCHOR_PLAN.json', run / 'GOLD_PLAN.json', run / 'WRITER_PLAN.json',
            Path(config['actor_reference_policy']['path']), Path(config['character_registry']['path']), run / 'ACTOR_REFERENCE_POLICY.json',
            source_path, Path(slot['history_path']), Path(slot['prior_pool_path']),
            run / 'SOURCE.json', run / 'PRIOR.json', run / 'PRIOR_COVERAGE.json', run / 'provider_profile.json'}
        m = {'version': VERSION, 'root_id': slot['root_id'], 'slot_id': slot['slot_id'],
             'candidate_ids': candidate_ids(config, slot), 'anchor_plan': bind(run / 'ANCHOR_PLAN.json'),
             'gold_plan': bind(run / 'GOLD_PLAN.json'), 'gold_origin': ordering_plan.GOLD_ORIGIN,
             'writer_plan': bind(run / 'WRITER_PLAN.json'),
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


def verify_writer_plan(run, manifest, source):
    run = Path(run)
    need(manifest.get('writer_plan') == bind(run / 'WRITER_PLAN.json'), 'writer_plan_binding_changed')
    need(bind(run / 'ANCHOR_PLAN.json') == manifest['anchor_plan'], 'ordering_anchor_plan_changed')
    expected = ordering_plan.writer_plan(source, read(run / 'ANCHOR_PLAN.json'))
    need(read(run / 'WRITER_PLAN.json') == expected, 'writer_plan_reconstruction_mismatch')
    return expected


def verify_gold_plan(run, manifest, source):
    run = Path(run)
    need(manifest.get('gold_origin') == ordering_plan.GOLD_ORIGIN
         and manifest.get('gold_plan') == bind(run / 'GOLD_PLAN.json'), 'program_gold_binding_changed')
    record = read(run / 'GOLD_PLAN.json')
    need(set(record) == {'version', 'created_at_utc', 'source', 'anchor_plan', 'gold_origin',
                         'semantic_certification', 'gold_supplied_to_writer', 'questions'}, 'program_gold_schema')
    need(record['version'] == 'v98_program_gold_plan_v1'
         and record['gold_origin'] == ordering_plan.GOLD_ORIGIN
         and record['semantic_certification'] is False and record['gold_supplied_to_writer'] is False,
         'program_gold_contract')
    need(record['source'] == bind(run / 'SOURCE.json') == manifest['source']
         and source == read(run / 'SOURCE.json')
         and record['anchor_plan'] == bind(run / 'ANCHOR_PLAN.json') == manifest['anchor_plan'],
         'program_gold_source_or_anchor_changed')
    need(record['questions'] == ordering_plan.gold_questions(source, read(run / 'ANCHOR_PLAN.json')),
         'program_gold_reconstruction_mismatch')
    need([q['candidate_id'] for q in record['questions']] == manifest['candidate_ids'], 'program_gold_identity')
    need(aware(record['created_at_utc']) <= aware(manifest['prepared_at_utc']) <= aware(now()),
         'program_gold_not_frozen_before_writer')
    for name in ('GENERATION.STARTED.json',):
        if (run / name).exists():
            need(aware(manifest['prepared_at_utc']) <= aware(read(run / name)['started_at_utc']),
                 'program_gold_after_generation_started')
    return {q['candidate_id']: q for q in record['questions']}


def validate_raw(raw, manifest, source, source_path, prior):
    """Accept statements only, then attach the immutable pre-request answer."""
    gold_rows = verify_gold_plan(Path(source_path).parent, manifest, source)
    verify_writer_plan(Path(source_path).parent, manifest, source)
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
            row = ordering_plan.project(row, plans[qid], gold_rows[qid])
        except (ValueError, TypeError, KeyError) as error:
            rejected.append({'candidate_id': qid, 'reasons': [str(error)]}); continue
        program_gold = row['gold'][:]
        name_errors = actor_reference.name_policy_errors(row['items'])
        if name_errors:
            rejected.append({'candidate_id': qid, 'reasons': name_errors}); continue
        errors, candidate = load_ordering_gate().validate_generated_ordering(row, qid, manifest['root_id'], source, source_path)
        if not errors and candidate['gold'] != program_gold:
            errors = ['program_gold_disagrees_with_source_anchors']
        if not errors and whole_set(candidate) in seen:
            errors = ['duplicate_prior_or_current_six_item_set']
        if errors:
            rejected.append({'candidate_id': qid, 'reasons': errors}); continue
        seen.add(whole_set(candidate)); candidate['generator_model'] = engine().legacy.MODEL
        candidate['status'] = 'MECHANICAL_ONLY_REQUIRES_SOURCE_FIRST_REVIEW'
        candidate['gold_origin'] = ordering_plan.GOLD_ORIGIN
        candidate['program_gold_plan'] = manifest['gold_plan']
        accepted.append(candidate)
    return accepted, rejected


def verify(home=HERE):
    home = require_shard(home); config = read(home / 'CONFIG.json'); root = home / 'ordering'
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
        paths = [home / 'CONFIG.json', Path(config['ordering_anchor_plan']['path']), run / 'ANCHOR_PLAN.json', run / 'GOLD_PLAN.json', run / 'WRITER_PLAN.json',
                 Path(config['actor_reference_policy']['path']), Path(config['character_registry']['path']), run / 'ACTOR_REFERENCE_POLICY.json',
                 Path(slot['source_path']), Path(slot['history_path']), Path(slot['prior_pool_path']),
                 run / 'SOURCE.json', run / 'PRIOR.json', run / 'PRIOR_COVERAGE.json', run / 'provider_profile.json'] + list(HERE.glob('*.py')) + list(map(Path, ordering_prior.configured_paths(slot)))
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
        frozen_feedback = e.source_adapter.feedback_data(slot['root_id'])
        compact, compact_files = e.source_adapter.prior_candidates(slot['root_id'], frozen_feedback,
            run / 'feedback' / slot['root_id'] / 'candidate_history.json', e.v15)
        priors, prior_receipt, _ = ordering_prior.collect(
            slot['root_id'], run / 'feedback' / slot['root_id'] / 'candidate_history.json',
            ordering_prior.configured_paths(slot), compact_rows=compact, compact_files=compact_files)
        need(read(run / 'PRIOR.json') == priors and read(run / 'PRIOR_COVERAGE.json') == prior_receipt,
             'ordering_prior_coverage_or_public_text_changed')
        verify_gold_plan(run, m, read(run / 'SOURCE.json'))
        verify_writer_plan(run, m, read(run / 'SOURCE.json'))
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
            paths = [run / name for name in ('MANIFEST.json', 'SOURCE.json', 'PRIOR.json', 'PRIOR_COVERAGE.json', 'REQUEST.json', 'PROOF.json',
                     'CAPACITY.json', 'ANCHOR_PLAN.json', 'GOLD_PLAN.json', 'WRITER_PLAN.json', 'GENERATION.STARTED.json', 'POOL.json', 'PUBLIC_QUESTIONS.json', 'REJECTIONS.json')]
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
    home = require_shard(home); root = home / 'ordering'; verify(home)
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
