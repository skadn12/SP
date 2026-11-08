"""Bounded v68 evidence generation with a replayable mechanical repair semantic lock."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import hmac
import importlib.util
import json
from pathlib import Path
import re
import secrets
import shutil
import sys
import source_adapter

HERE = Path(__file__).resolve().parent
V63 = HERE.parent / 'h4_option_composition_v63_qa'
_saved_path = sys.path[:]
sys.path.insert(0, str(V63))
_spec = importlib.util.spec_from_file_location('v64_readonly_v63_generation', V63 / 'generate.py')
legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legacy)
import qa_transport
sys.path[:] = _saved_path
base, v15 = legacy.base, legacy.v15
read, write, require = base.read, base.write, base.require
VERSION = 'v68_scope_locked_assets_v1'
CAP = 6144
START, END = '[CURRENT_ASSET_TASK_V64]', '[/CURRENT_ASSET_TASK_V64]'
INTRO = ('Read the complete supplied dialogue, public historical question inventory, and frozen previous-training-round feedback. '
         'The current output task and schema follow that feedback. The source is the exact frozen original dialogue plus R01, with no new expansion; no new training or external-world certification is implied.\n')
TASK = """
CURRENT TASK: CREATE EVIDENCE ASSETS, NOT FINAL QUESTIONS OR GOLD.
For the requested root and coherent soft focus, select 6–8 distinct source facts
(aim for eight). The entire source is available, including surrounding context,
corrections and later outcomes. Do not truncate it or treat the focus as evidence.
Prefer coherent facts within the soft focus whose interpretation needs the full
context to distinguish people, times, conditions and action states. Select only
the focus and context directly needed to understand it. Never use unrelated old
topics to fill six or eight facts; return fewer than six or [] if needed. Do not pad
the set with unrelated topics or favor mere mechanical negation over meaningful
source-grounded alternatives. The focus itself never proves a proposition.
Each fact needs a short exact contiguous quote, exact turn_ref and speaker, and
explicit actor/event/time/condition/action_state. Copy turn_ref verbatim from
the complete source. Preserve case and every
prefix: original turns use root_id:Pxx:Txxx (for example,
cc-train-episode-123456:P01:T001), while child turns use R01:Sxx:Txxx.
Never omit the original root prefix, invent a prefix, or lowercase R01/S/P/T.
For EACH fact, first write one
complete faithful supported proposition, then construct a complete contradicted
proposition with explicit counterevidence in the same source and scope. Do not
skip the contradicted field or default it to null. These two versions must be
generated before the later independent question-assembly step; do not assemble
questions or choose their answer counts here. The two versions are alternatives
for ONE fact, never two separate facts or two options for that fact in one question.
Seek useful explicit alternatives for distinct facts where available. Keep the
same actor, event, time, condition and action state; change only an attribute
whose incompatible value is explicitly fixed in that scope. Both alternatives
must be explicitly mutually exclusive for that fact; unknown is not false.
Both PUBLIC proposition strings must themselves explicitly name the same actor,
event, time, condition and action state. The private scope object cannot supply
missing public qualifiers. Anchor each proposition to the same particular event
and stage; do not rely on the broad stem or topic to imply that restriction.
For example: "In the reported booking update, Alex confirmed this appointment
for Tuesday at 3 PM" versus "In the reported booking update, Alex confirmed
this appointment for Wednesday at 3 PM." Do not contrast a generic never/already
claim with a differently timed state, or treat choosing one recipe/advice as
excluding an additional choice when the source does not fix a unique choice.
When nearby source context supplies a competing value for another actor, time
or condition, prefer that attribute as an alternative while explicitly keeping
the target scope, only if the target source fixes an incompatible value. Preserve
qualifiers such as approximately and proposed. Avoid invented extreme negations
or unsupported only/never claims; if the conditions fail, use the null procedure.
A plan, offer, condition or counterfactual must retain that status. A claim about
what a speaker confirmed differs from an assertion about external events.
Resolve pronouns from full surrounding turns. If both a source claim and your
alternative could be true, or a contradiction requires an unstated assumption,
use null only if no explicitly refuted, scope-preserving alternative can truly
be constructed. In that case set contradiction_unavailable_reason to a concrete
nonempty explanation identifying what this fact's source fixes and why that
evidence cannot refute an alternative in the same scope. With a non-null
contradicted proposition, contradiction_unavailable_reason must be null.
Include contradiction_unavailable_reason on every fact; a missing field is a
schema error even when contradicted is non-null. A null contradicted field needs
a specific nonempty explanation, never a missing, blank or null reason.
Unmentioned, unknown, unconfirmed, different-topic or out-of-scope facts are
never false merely for those reasons. Another actor or another time is not
counterevidence. A stated plan alone does not refute later completion; preserve
the planned action state in both versions. In the short rationale cite the
actual conflicting fact. Never relabel a faithful true statement contradicts.
Example: Alex says, "My appointment is scheduled for Tuesday at 3 PM."
Supported: "Alex's appointment is scheduled for Tuesday at 3 PM."
Contradicted: "Alex's appointment is scheduled for Wednesday at 3 PM."
The explicitly fixed Tuesday appointment contradicts Wednesday for the same
Alex, appointment and scheduled state; contradiction_unavailable_reason is null.
Do not choose or mention a target number of true options; do not write Gold.
Propose one neutral public_stem with Select all that apply, whose scope covers
all the facts. It must not reveal answer counts or facts. Prior whole-question
sets contain public statements only. Individual overlap and paraphrase are
allowed; seek combinations permitting a different entire six-statement set.
Return only JSON with exactly these fields and no reasoning trace:
{root_id,slot_id,public_stem,facts:[{fact_id,scope:{actor,event,time,condition,action_state},
anchor:{turn_ref,speaker,quote},supported:{proposition,relation:"supports",rationale},
contradicted:null|{proposition,relation:"contradicts",rationale},
contradiction_unavailable_reason:null|string}]}.
Use neutral fact IDs such as F01. Keep all scope values strings; use "unspecified"
when a field is genuinely unspecified. Do not invent facts to reach six or eight.
If fewer than six can be evidenced, return the available facts (possibly []);
this is an evidence shortage, not a request for another semantic attempt.
Follow this asset schema even if historical feedback discusses final questions.
"""
CONTRACT = {'version': VERSION, 'max_fact_groups': 8, 'preferred_fact_groups': 8,
    'minimum_for_six_option_assembly': 6, 'max_tokens': CAP, 'http_attempts_max_per_slot': 2,
    'mechanical_repair_max': 1, 'semantic_retries': 0, 'transport_retries': 0,
    'repair_semantic_guard_version': 'v68_mechanical_repair_semantic_lock_v1',
    'unlocked_repair_forbidden': True,
    'full_source_feedback_prior': True, 'current_solver_feedback': False,
    'target_count_in_writer_input': False, 'training_allowed': False}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def binding(path): return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def _dependencies():
    provider, paths = legacy._dependency_paths()
    return provider, set(paths) | set(HERE.glob('*.py')) | {Path(__file__).resolve()}


def get_slot(config, slot_id):
    slots = config.get('slots')
    require(isinstance(slots, list) and 1 <= len(slots) <= 10, 'invalid_asset_slots')
    require(len({s.get('slot_id') for s in slots}) == len(slots), 'duplicate_asset_slot_id')
    matches = [s for s in slots if s.get('slot_id') == slot_id]
    require(len(matches) == 1 and re.fullmatch(r'[A-Za-z0-9_-]{1,128}', slot_id), 'unknown_asset_slot')
    require(config.get('training_allowed') is False and config.get('max_http') == 20
            and config.get('max_questions') == 10 and config.get('max_calls_per_model') == 180,
            'asset_global_budget_or_training_changed')
    slot = deepcopy(matches[0])
    require(type(slot.get('initial_target_count')) is int and 2 <= slot['initial_target_count'] <= 6,
            'initial_target_count_invalid')
    focus = slot.get('focus')
    require(isinstance(focus, dict) and isinstance(focus.get('scene_refs'), list)
            and all(isinstance(x, str) for x in focus['scene_refs'])
            and isinstance(focus.get('description'), str) and focus['description'].strip(), 'asset_focus_invalid')
    return slot


def prepare(config_path, slot_id, *, run=None):
    config_path = Path(config_path).resolve(); config = read(config_path); slot = get_slot(config, slot_id)
    run = Path(run).resolve() if run is not None else HERE / 'assets' / slot_id
    require(not run.exists(), 'asset_run_already_exists')
    history = Path(slot['history_path']).resolve()
    require(not slot.get('history_sha256') or sha(history)==slot['history_sha256'], 'configured_history_sha_changed')
    checked = source_adapter.checked_source(history, slot['root_id'])
    available = {t['turn_ref'].rsplit(':', 1)[0] for t in checked['source']['turns']}
    require(set(slot['focus']['scene_refs']) <= available, 'asset_focus_scene_not_in_source')
    feedback = source_adapter.feedback_data(slot['root_id'])
    lock = feedback['source_lock']
    require(lock.get('status') == 'EXPLICIT_PREVIOUS_TRAINING_ROUND_ONLY'
        and lock.get('merge_current_generation_feedback') is False
        and lock['model_identity_source']['sha256'] == v15.evaluator.TRAINED_AUDIT_SHA256,
        'asset_previous_feedback_not_frozen')
    provider, frozen = _dependencies()
    frozen |= {config_path, history, *checked['files']}
    feedback, feedback_files = source_adapter.freeze_feedback(slot['root_id'], run / 'feedback', data=feedback)
    frozen.update(feedback_files)
    priors, prior_files = source_adapter.prior_candidates(slot['root_id'], feedback,
        run / 'feedback' / slot['root_id'] / 'candidate_history.json', v15)
    frozen.update(prior_files)
    for item in slot['prior_pool_paths']:
        path = Path(item).resolve(); priors += legacy._prior_pool_rows(path, slot['root_id']); frozen.add(path)
    for path, value in [(run / 'SOURCE.json', checked['source']), (run / 'PRIOR.json', priors),
                        (run / 'UPSTREAM_ARTIFACT_AUDIT.json', checked['audit']),
                        (run / 'provider_profile.json', provider)]:
        write(path, value); frozen.add(path)
    manifest = {'version': VERSION, 'prepared_at_utc': v15.now(), 'contract': CONTRACT,
        'slot': slot, 'config': binding(config_path), 'root_id': slot['root_id'], 'slot_id': slot_id,
        'source_path': str(run / 'SOURCE.json'), 'source_sha256': sha(run / 'SOURCE.json'),
        'source_history': binding(history), 'source_turn_count': len(checked['source']['turns']),
        'source_mode': source_adapter.SOURCE_MODE, 'source_admission_status': source_adapter.SOURCE_STATUS,
        'new_expansion_tokens': 0,
        'solver_model': legacy.TARGET, 'generator_model': legacy.MODEL,
        'budget': {'http_attempts_max': 2}, 'version_id_salt': secrets.token_hex(32),
        'task_sha256': base.digest(TASK.encode()), 'training_allowed': False,
        'current_solver_feedback': False, 'files': base.bindings(frozen)}
    write(run / 'MANIFEST.json', manifest)
    payload, receipt = make_payload(run)
    write(run / 'INITIAL_REQUEST.json', payload)
    write(run / 'INITIAL_PROOF.json', proof_for(manifest, payload, receipt, 1))
    verify(run)
    return {'slot_id': slot_id, 'run': str(run), 'source_turn_count': manifest['source_turn_count'], 'budget': manifest['budget']}


def context_for(run, repair=None):
    m = read(run / 'MANIFEST.json'); source = read(run / 'SOURCE.json')
    prior = legacy.prompt_prior(read(run / 'PRIOR.json'))
    context = {'root_id': m['root_id'], 'slot_id': m['slot_id'], 'source_turns': source['turns'],
        'soft_focus': m['slot']['focus'], 'previous_candidates_dedup_only': prior,
        'used_six_statement_sets': legacy.used_statement_sets(prior),
        'source_mode': source_adapter.SOURCE_MODE, 'new_expansion_tokens': 0,
        'source_training_pool_eligible': False, 'new_scene_evidence_required': False}
    if repair is not None: context['mechanical_repair'] = deepcopy(repair)
    return context


def make_payload(run, repair=None):
    run = Path(run); m = read(run / 'MANIFEST.json')
    payload, receipt = legacy.legacy.payload_with_feedback(run, m['root_id'], INTRO,
        context_for(run, repair), CAP)
    intermediate = payload['messages'][0]['content']
    tail = '\n\n' + START + TASK + END
    payload['messages'][0]['content'] += tail
    receipt['asset_task_tail_receipt'] = {'kind': 'evidence_assets', 'task_sha256': base.digest(tail.encode()),
        'intermediate_sha256': base.digest(intermediate.encode()), 'final_payload_sha256': base.digest(payload)}
    return payload, receipt


def proof_for(m, payload, receipt, attempt):
    return {'role': 'asset_generator', 'root_id': m['root_id'], 'slot_id': m['slot_id'],
            'source_sha256': m['source_sha256'], 'attempt': attempt,
            'feedback_receipt': receipt, 'payload_sha256': base.digest(payload)}


def verify_request(run, payload, proof, repair=None):
    m = read(run / 'MANIFEST.json'); expected, receipt = make_payload(run, repair)
    require(payload == expected and proof.get('feedback_receipt') == receipt, 'asset_request_replay_changed')
    require(proof.get('payload_sha256') == base.digest(payload) and proof.get('role') == 'asset_generator'
        and proof.get('root_id') == m['root_id'] and proof.get('slot_id') == m['slot_id']
        and proof.get('source_sha256') == m['source_sha256'] and proof.get('attempt') in (1, 2),
        'asset_request_proof_changed')
    return True


def _validate_assets_exact(raw, m, source):
    """Only schema/quoted-source errors trigger repair; relations are untrusted."""
    try: data = json.loads(raw)
    except (ValueError, TypeError): return None, ['invalid_json']
    errors = []
    def need(ok, msg):
        if not ok: errors.append(msg)
        return ok
    def obj(value, keys, path):
        if not need(isinstance(value, dict), path + ':expected_object'): return False
        missing, extra = sorted(set(keys) - set(value)), sorted(set(value) - set(keys))
        need(not missing, path + ':missing_keys:' + json.dumps(missing, separators=(',', ':')))
        need(not extra, path + ':extra_keys:' + json.dumps(extra, separators=(',', ':')))
        return not missing and not extra
    def text(value): return isinstance(value, str) and bool(value.strip())
    if not obj(data, ('root_id','slot_id','public_stem','facts'), 'assets'): return None, errors
    need(data['root_id'] == m['root_id'] and data['slot_id'] == m['slot_id'], 'asset_identity')
    need(text(data['public_stem']), 'public_stem_empty')
    if not need(isinstance(data['facts'], list) and len(data['facts']) <= 8, 'facts_schema_or_maximum'): return None, errors
    grouped = {}
    for turn in source['turns']:
        grouped.setdefault(turn['turn_ref'], []).append(turn)
    turns = {ref: values[0] for ref, values in grouped.items() if len(values) == 1}; seen = set()
    for i, fact in enumerate(data['facts']):
        path = f'facts[{i}]'
        if not obj(fact, ('fact_id','scope','anchor','supported','contradicted',
                          'contradiction_unavailable_reason'), path): continue
        fid = fact['fact_id']
        if need(isinstance(fid,str) and bool(re.fullmatch(r'[A-Za-z0-9_-]{1,64}',fid)), path+':fact_id'):
            need(fid not in seen, path+':duplicate_fact_id'); seen.add(fid)
        scope=fact['scope']
        if obj(scope, ('actor','event','time','condition','action_state'),path+':scope'):
            need(all(text(x) for x in scope.values()), path+':scope_values')
        anchor=fact['anchor']
        if obj(anchor, ('turn_ref','speaker','quote'), path+':anchor'):
            if need(all(text(x) for x in anchor.values()),path+':anchor_values'):
                turn=turns.get(anchor['turn_ref'])
                if need(turn is not None,path+':unknown_turn_ref:' + json.dumps(anchor['turn_ref'])
                        + ':copy_exact_SOURCE_turn_ref'):
                    need(anchor['speaker']==turn['speaker'],path+':speaker_mismatch')
                    need(anchor['quote'] in turn['text'],path+':quote_not_exact_contiguous')
        for key,relation in [('supported','supports'),('contradicted','contradicts')]:
            version=fact[key]
            if version is None and key=='contradicted':continue
            if obj(version, ('proposition','relation','rationale'),path+':'+key):
                need(text(version['proposition']) and text(version['rationale']),path+':'+key+':empty_text')
                need(version['relation']==relation,path+':'+key+':relation_enum')
        # Explanation content and contradiction quality remain source-first review decisions.
        reason = fact['contradiction_unavailable_reason']
        if fact['contradicted'] is None:
            need(text(reason), path+':contradiction_unavailable_reason_required')
        else:
            need(reason is None, path+':contradiction_unavailable_reason_must_be_null')
    return (None if errors else data), errors


def build_repair(raw, errors, m, source):
    """Give exact-source reference hints to the writer without normalizing by quote."""
    repair = {'instruction':
        'Repair only the semantic_lock.allowed_changes at their exact paths and fixed values; '
        'all other first-draft fields, ordering and values are immutable. Return the full asset schema. '
        'Only listed schema or exact-source binding errors are eligible. '
        'Preserve the existing propositions, fact scope, relations, rationales and source meaning; '
        'do not add facts, change semantic content, assemble questions or write Gold. '
        'Copy each turn_ref verbatim from SOURCE, including original root:P prefixes and R01:S case. '
        'Reference hints are candidates for you to verify and copy, never automatic replacements. '
        'Include contradiction_unavailable_reason on every fact: null for non-null contradicted; '
        'a specific nonempty string for null contradicted. Do not invent semantic reasons to repair schema.',
        'errors': deepcopy(errors), 'previous_response': raw, 'reference_hints': [],
        'semantic_lock': build_repair_guard(raw, errors, m, source)}
    try: data = json.loads(raw)
    except (ValueError, TypeError): return repair
    if (not isinstance(data, dict) or data.get('root_id') != m['root_id']
            or source.get('root_id') != m['root_id'] or not isinstance(data.get('facts'), list)):
        return repair
    unknown = {int(match.group(1)) for error in errors if isinstance(error, str)
               for match in [re.match(r'^facts\[(\d+)\]:unknown_turn_ref:', error)] if match}
    for i in sorted(unknown):
        if i >= len(data['facts']): continue
        fact = data['facts'][i]
        anchor = fact.get('anchor') if isinstance(fact, dict) else None
        if not isinstance(anchor, dict): continue
        speaker, quote = anchor.get('speaker'), anchor.get('quote')
        if not (isinstance(speaker, str) and speaker.strip()
                and isinstance(quote, str) and quote.strip()): continue
        matches = [turn for turn in source['turns']
                   if turn['speaker'] == speaker and quote in turn['text']]
        if len(matches) != 1: continue
        repair['reference_hints'].append({'fact_index': i, 'fact_id': fact.get('fact_id'),
            'original_turn_ref': anchor.get('turn_ref'), 'candidate_turn_ref': matches[0]['turn_ref'],
            'speaker': speaker, 'quote': quote})
    return repair


GUARD_VERSION = 'v68_mechanical_repair_semantic_lock_v1'


def _strict_asset_json(raw):
    """Ambiguous duplicate keys and non-JSON numbers cannot define a semantic lock."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result: raise ValueError('duplicate_json_key')
            result[key] = value
        return result
    def constant(value): raise ValueError('nonfinite_json_number')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def _guard_changes(data, errors, m, source):
    """Allow only deterministic schema removals/additions and exact-source bindings."""
    changes = []
    def change(path, operation, value=None):
        row = {'path': path, 'operation': operation}
        if operation == 'set': row['value'] = value
        if row not in changes: changes.append(row)
    def extra(obj, keys, path):
        if isinstance(obj, dict):
            for key in sorted(set(obj) - set(keys)): change(path + [key], 'delete')
    extra(data, ('root_id','slot_id','public_stem','facts'), [])
    if not isinstance(data, dict) or not isinstance(data.get('facts'), list): return changes
    grouped = {}
    for turn in source['turns']: grouped.setdefault(turn['turn_ref'], []).append(turn)
    for i, fact in enumerate(data['facts']):
        path = ['facts', i]
        if not isinstance(fact, dict): continue
        extra(fact, ('fact_id','scope','anchor','supported','contradicted','contradiction_unavailable_reason'), path)
        extra(fact.get('scope'), ('actor','event','time','condition','action_state'), path + ['scope'])
        for key in ('supported','contradicted'):
            extra(fact.get(key), ('proposition','relation','rationale'), path + [key])
        if 'contradiction_unavailable_reason' not in fact and isinstance(fact.get('contradicted'), dict):
            change(path + ['contradiction_unavailable_reason'], 'set', None)
        anchor = fact.get('anchor')
        extra(anchor, ('turn_ref','speaker','quote'), path + ['anchor'])
        if not isinstance(anchor, dict): continue
        ref = anchor.get('turn_ref'); direct = grouped.get(ref, []) if isinstance(ref,str) else []
        # This is the unchanged strip-only normalization's permitted exact alias.
        prefix = m['root_id'] + ':'
        if not direct and isinstance(ref,str) and ref.startswith(prefix) and source.get('root_id') == m['root_id']:
            matches = grouped.get(ref[len(prefix):], [])
            quote = anchor.get('quote')
            if (len(matches) == 1 and anchor.get('speaker') == matches[0]['speaker']
                    and isinstance(quote,str) and quote.strip() and quote in matches[0]['text']):
                change(path + ['anchor','turn_ref'], 'set', ref[len(prefix):]); direct = matches
        if len(direct) == 1:
            turn = direct[0]
            if anchor.get('speaker') != turn['speaker']:
                change(path + ['anchor','speaker'], 'set', turn['speaker'])
            quote = anchor.get('quote')
            if not (isinstance(quote,str) and quote.strip() and quote in turn['text']):
                change(path + ['anchor','quote'], 'set', turn['text'])
        elif source.get('root_id') == m['root_id']:
            speaker, quote = anchor.get('speaker'), anchor.get('quote')
            if isinstance(speaker,str) and speaker.strip() and isinstance(quote,str) and quote.strip():
                matches = [t for t in source['turns'] if t['speaker'] == speaker and quote in t['text']]
                if len(matches) == 1:
                    change(path + ['anchor','turn_ref'], 'set', matches[0]['turn_ref'])
    return changes


def _guard_apply(data, changes, *, mask=False):
    out = deepcopy(data)
    for row in changes:
        path = row['path']; parent = out
        for part in path[:-1]: parent = parent[part]
        if mask or row['operation'] == 'delete': parent.pop(path[-1], None)
        else: parent[path[-1]] = deepcopy(row['value'])
    return out


def _guard_diff(before, after, path=None):
    path = [] if path is None else path
    if type(before) is not type(after): return [path]
    if isinstance(before, dict):
        return [p for key in sorted(set(before) | set(after))
                for p in ([path + [key]] if key not in before or key not in after
                          else _guard_diff(before[key],after[key],path + [key]))]
    if isinstance(before, list):
        if len(before) != len(after): return [path]
        return [p for i,(left,right) in enumerate(zip(before,after))
                for p in _guard_diff(left,right,path + [i])]
    return [] if before == after else [path]


def build_repair_guard(baseline_raw, baseline_errors, m, source,
                       candidate_raw=None, candidate_errors=None):
    """Replayable lock over the full first-draft tree except explicit allowed paths."""
    digest_text = lambda raw: hashlib.sha256(raw.encode('utf-8')).hexdigest() if isinstance(raw,str) else None
    proof = {'version': GUARD_VERSION, 'attempt': 1 if candidate_raw is None else 2,
        'root_id': m['root_id'], 'slot_id': m['slot_id'], 'source_sha256': m.get('source_sha256'),
        'source_content_sha256': base.digest(source), 'baseline_raw_text_sha256': digest_text(baseline_raw),
        'candidate_raw_text_sha256': digest_text(candidate_raw), 'baseline_projection_sha256': None,
        'candidate_projection_sha256': None, 'allowed_changes': [], 'changed_paths': [],
        'unauthorized_changes': [], 'lock_errors': [], 'repair_allowed': False, 'accepted': False,
        'status': 'BASELINE_BLOCKED'}
    try: before = _strict_asset_json(baseline_raw)
    except (ValueError, TypeError):
        proof['lock_errors'] = ['baseline_json_not_unambiguous']; return proof
    changes = _guard_changes(before, baseline_errors, m, source)
    proof['allowed_changes'] = changes
    try:
        expected = _guard_apply(before, changes)
        _, remaining = validate_assets(json.dumps(expected,allow_nan=False),m,source)
        masked = _guard_apply(before, changes, mask=True)
    except (ValueError, TypeError, KeyError):
        proof['lock_errors'] = ['baseline_semantic_projection_unavailable']; return proof
    proof['baseline_projection_sha256'] = base.digest(masked)
    if remaining:
        proof['lock_errors'] = ['baseline_not_mechanically_lockable:' + e for e in remaining]; return proof
    _, actual_baseline_errors = validate_assets(baseline_raw,m,source)
    if baseline_errors != actual_baseline_errors:
        proof['lock_errors'] = ['baseline_diagnostics_changed']; return proof
    if not baseline_errors:
        # No repair is needed or authorized; semantic certification stays independent.
        if candidate_raw is not None:
            proof.update(status='REPAIR_BLOCKED',lock_errors=['repair_not_authorized'])
        else: proof.update(status='NOT_NEEDED', accepted=True)
        return proof
    proof.update(repair_allowed=True,status='REPAIR_ALLOWED')
    if candidate_raw is None: return proof
    proof['status'] = 'REPAIR_BLOCKED'
    try: after = _strict_asset_json(candidate_raw)
    except (ValueError,TypeError):
        proof['lock_errors'] = ['candidate_json_not_unambiguous']; return proof
    proof['changed_paths'] = _guard_diff(before,after)
    allowed = {tuple(c['path']):c for c in changes}
    for path in proof['changed_paths']:
        row = allowed.get(tuple(path)); valid = row is not None
        try:
            parent = after
            for part in path[:-1]: parent = parent[part]
            if valid:
                valid = (path[-1] not in parent if row['operation'] == 'delete'
                         else path[-1] in parent and type(parent[path[-1]]) is type(row['value'])
                         and parent[path[-1]] == row['value'])
        except (KeyError,IndexError,TypeError): valid = False
        if not valid: proof['unauthorized_changes'].append(path)
    try: projection = _guard_apply(after,changes,mask=True)
    except (KeyError,IndexError,TypeError):
        proof['lock_errors'].append('candidate_semantic_projection_unavailable'); return proof
    proof['candidate_projection_sha256'] = base.digest(projection)
    if proof['baseline_projection_sha256'] != proof['candidate_projection_sha256']:
        proof['lock_errors'].append('semantic_projection_changed')
    if proof['unauthorized_changes']: proof['lock_errors'].append('nonpermitted_repair_changes')
    _, actual_candidate_errors = validate_assets(candidate_raw,m,source)
    if actual_candidate_errors: proof['lock_errors'].append('candidate_mechanical_validation_failed')
    if candidate_errors is not None and candidate_errors != actual_candidate_errors:
        proof['lock_errors'].append('candidate_diagnostics_changed')
    if not proof['lock_errors']: proof.update(status='REPAIR_ACCEPTED',accepted=True)
    return proof


def replay_repair_guard(folder, baseline_raw, baseline_errors, m, source,
                        candidate_raw=None, candidate_errors=None):
    proof = build_repair_guard(baseline_raw,baseline_errors,m,source,candidate_raw,candidate_errors)
    require(read(Path(folder)/'REPAIR_GUARD.json') == proof,'asset_repair_guard_replay_changed')
    return proof


def normalize_assets(raw, m, source):
    """Resolve only an exact, current-root prefix alias; preserve every semantic field."""
    raw_hash = hashlib.sha256(raw.encode('utf-8')).hexdigest() if isinstance(raw, str) else None
    receipt = {'version': 'v65_exact_root_prefix_alias_v1', 'raw_text_sha256': raw_hash,
               'source_sha256': m.get('source_sha256'), 'source_content_sha256': base.digest(source),
               'root_id': m['root_id'], 'facts': [], 'normalized_ref_count': 0,
               'normalized_data_sha256': None, 'accepted': False, 'mechanical_errors': []}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        receipt['mechanical_errors'] = ['invalid_json']
        return None, ['invalid_json'], receipt
    grouped = {}
    for turn in source['turns']:
        grouped.setdefault(turn['turn_ref'], []).append(turn)
    prefix = m['root_id'] + ':'
    source_identity_matches = source.get('root_id') == m['root_id']
    facts = data.get('facts', []) if isinstance(data, dict) else []
    if isinstance(facts, list):
        for i, fact in enumerate(facts):
            anchor = fact.get('anchor') if isinstance(fact, dict) else None
            old = anchor.get('turn_ref') if isinstance(anchor, dict) else None
            row = {'fact_index': i, 'fact_id': fact.get('fact_id') if isinstance(fact, dict) else None,
                   'original_turn_ref': old, 'normalized_turn_ref': old, 'candidate_turn_ref': None,
                   'candidate_match_count': 0, 'speaker_matches': False, 'quote_matches': False,
                   'normalization_applied': False, 'resolution': 'invalid_anchor'}
            if isinstance(old, str):
                direct = grouped.get(old, [])
                candidate = old
                aliased = False
                if not direct and source_identity_matches and old.startswith(prefix):
                    candidate = old[len(prefix):]  # Strip exactly once, never search by quote.
                    aliased = True
                matches = grouped.get(candidate, [])
                row.update(candidate_turn_ref=candidate, candidate_match_count=len(matches))
                if len(matches) == 1 and isinstance(anchor, dict):
                    turn = matches[0]
                    row['speaker_matches'] = anchor.get('speaker') == turn['speaker']
                    quote = anchor.get('quote')
                    row['quote_matches'] = isinstance(quote, str) and bool(quote.strip()) and quote in turn['text']
                safe = len(matches) == 1 and row['speaker_matches'] and row['quote_matches']
                if aliased and safe:
                    anchor['turn_ref'] = candidate
                    row.update(normalized_turn_ref=candidate, normalization_applied=True,
                               resolution='current_root_prefix_removed_exact_source_match')
                    receipt['normalized_ref_count'] += 1
                else:
                    row['resolution'] = 'exact_ref_unchanged' if direct else 'alias_rejected_or_unresolved'
            receipt['facts'].append(row)
    receipt['normalized_data_sha256'] = base.digest(data)
    accepted, errors = _validate_assets_exact(json.dumps(data), m, source)
    receipt.update(accepted=accepted is not None, mechanical_errors=errors)
    return accepted, errors, receipt


def validate_assets(raw, m, source):
    """Backward-compatible validator; generation also persists the normalization receipt."""
    accepted, errors, _ = normalize_assets(raw, m, source)
    return accepted, errors


def replay_normalization(folder, raw, m, source):
    accepted, errors, receipt = normalize_assets(raw, m, source)
    require(read(Path(folder) / 'REFERENCE_NORMALIZATION.json') == receipt,
            'asset_reference_normalization_replay_changed')
    return accepted, errors


def projections(m, assets, assets_sha):
    public = {'root_id': m['root_id'], 'slot_id': m['slot_id'], 'source_path': m['source_path'],
        'source_sha256': m['source_sha256'], 'assets_sha256': assets_sha,
        'public_stem': assets['public_stem'], 'versions': [], 'contains_relations': False,
        'contains_solver_scores': False}
    mapping=[]
    for fact in assets['facts']:
        versions=[]
        for key,code in [('supported','s'),('contradicted','c')]:
            version=fact[key]
            if version is None:continue
            internal=fact['fact_id']+':'+code
            vid='v_'+hmac.new(bytes.fromhex(m['version_id_salt']),internal.encode(),hashlib.sha256).hexdigest()[:24]
            versions.append({'version_id':vid,'fact_id':fact['fact_id'],'proposition':version['proposition']})
            mapping.append({'version_id':vid,'fact_id':fact['fact_id'],'internal_version_id':internal,
                            'variant':key,'relation':version['relation']})
        public['versions'].extend(versions)
    public['versions'].sort(key=lambda x:x['version_id'])
    private={'source_sha256':m['source_sha256'],'assets_sha256':assets_sha,
             'versions':sorted(mapping,key=lambda x:x['version_id'])}
    return public,private


def export_public(run):
    run=Path(run);m=read(run/'MANIFEST.json');assets=read(run/'ASSETS.json')
    valid,errors=validate_assets(json.dumps(assets),m,read(run/'SOURCE.json'));require(not errors,'cannot_export_invalid_assets')
    public,mapping=projections(m,valid,sha(run/'ASSETS.json'))
    write(run/'PUBLIC_ASSETS.json',public)
    mapping['public_assets_sha256']=sha(run/'PUBLIC_ASSETS.json')
    write(run/'VERSION_MAP.json',mapping)
    return public


def verify(run, *, complete=False):
    run=Path(run);m=read(run/'MANIFEST.json')
    require(m['version']==VERSION and m['contract']==CONTRACT and m['task_sha256']==base.digest(TASK.encode())
        and m['budget']=={'http_attempts_max':2} and m['training_allowed'] is False
        and m['current_solver_feedback'] is False and m['solver_model']==legacy.TARGET
        and m['generator_model']==legacy.MODEL,'asset_manifest_contract_changed')
    base.verify_bindings(m['files'])
    require(sha(m['config']['path'])==m['config']['sha256']
        and get_slot(read(m['config']['path']),m['slot_id'])==m['slot'],'asset_config_slot_changed')
    require(sha(run/'SOURCE.json')==m['source_sha256'] and len(read(run/'SOURCE.json')['turns'])==m['source_turn_count'],
            'asset_full_source_changed')
    require(m['source_path']==str(run.resolve()/'SOURCE.json'),'asset_source_path_changed')
    verify_request(run,read(run/'INITIAL_REQUEST.json'),read(run/'INITIAL_PROOF.json'))
    requests=sorted((run/'requests').glob('*/REQUEST.json'))
    require(len(requests)<=2,'asset_request_budget_exceeded')
    previous=None;previous_raw=None;accepted=None;errors=[];baseline_raw=None;baseline_errors=None;guard=None
    for i,path in enumerate(requests,1):
        folder=path.parent;repair=None
        if i==2:
            require(previous is not None and previous.get('mechanical_errors') and guard is not None
                and guard['repair_allowed'],'asset_semantic_or_transport_retry_forbidden')
            repair=build_repair(previous_raw,previous['mechanical_errors'],m,read(run/'SOURCE.json'))
        proof=read(folder/'PROOF.json');require(proof.get('attempt')==i,'asset_attempt_order_changed')
        verify_request(run,read(path),proof,repair)
        if (folder/'HTTP_ATTEMPT.json').exists():
            http=read(folder/'HTTP_ATTEMPT.json')
            require(http['actual_payload_equals_frozen'] is True and http['payload_sha256']==base.digest(read(path)), 'asset_http_payload_changed')
        if (folder/'ERROR.json').exists():
            require(i==len(requests),'asset_transport_failure_retried');previous=None;accepted=None;errors=['transport_or_capacity_failure'];continue
        require((folder/'HTTP_ATTEMPT.json').exists(), 'asset_response_without_http_receipt')
        response=read(folder/'RESPONSE.json');previous_raw=response['text']
        metadata=response.get('metadata',{})
        require(metadata.get('returned_model')==legacy.MODEL and metadata.get('finish_reason')=='stop'
            and metadata.get('http_status',200)==200,'asset_response_transport_metadata_changed')
        accepted,errors=replay_normalization(folder,response['text'],m,read(run/'SOURCE.json'))
        if i==1:
            baseline_raw,baseline_errors=response['text'],deepcopy(errors)
            guard=replay_repair_guard(folder,baseline_raw,baseline_errors,m,read(run/'SOURCE.json'))
        else:
            guard=replay_repair_guard(folder,baseline_raw,baseline_errors,m,read(run/'SOURCE.json'),response['text'],errors)
        if not guard['accepted']: accepted=None
        previous=read(folder/'VALIDATION.json')
        require(previous=={'mechanical_errors':errors,'fact_count':len(accepted['facts']) if accepted else None,
                         'semantic_review_performed':False,'shortage_triggers_retry':False},'asset_validation_replay_changed')
        require(bool(errors) or i==len(requests),'valid_assets_retried')
    if complete or (run/'GENERATION.COMPLETE.json').exists():
        c=read(run/'GENERATION.COMPLETE.json')
        require(c['version']==VERSION and c['root_id']==m['root_id'] and c['slot_id']==m['slot_id']
            and len(requests)>=1 and (run/'GENERATION.STARTED.json').exists()
            and c['source_first_review_required'] is True
            and c['requests_prepared']==len(requests) and c['http_attempts']==len(list((run/'requests').glob('*/HTTP_ATTEMPT.json')))
            and c['training_allowed'] is False and c['semantic_retries']==0,'asset_completion_counts_changed')
        base.verify_bindings(c['files'])
        if accepted is not None:
            require(read(run/'ASSETS.json')==accepted,'asset_response_projection_changed')
            public,mapping=projections(m,accepted,sha(run/'ASSETS.json'))
            require(read(run/'PUBLIC_ASSETS.json')==public,'asset_public_projection_changed')
            mapping['public_assets_sha256']=sha(run/'PUBLIC_ASSETS.json')
            require(read(run/'VERSION_MAP.json')==mapping,'asset_private_mapping_changed')
            require(c['status']==('ASSETS_READY' if len(accepted['facts'])>=6 else 'INSUFFICIENT_FACTS')
                and c['fact_count']==len(accepted['facts']),'asset_completion_status_changed')
        else:
            require(c['status']=='NO_VALID_ASSETS' and c['fact_count']==0 and not (run/'ASSETS.json').exists(), 'invalid_asset_failure_completion')
    return m


def generate(run):
    run=Path(run)
    with qa_transport.shared_lock():
        m=verify(run);require(not (run/'GENERATION.STARTED.json').exists(),'asset_generation_already_started_no_replay')
        write(run/'GENERATION.STARTED.json',{'started_at_utc':v15.now(),'manifest':binding(run/'MANIFEST.json'),'automatic_retry':False})
        caller=qa_transport.Caller(run);accepted=None;repair=None;attempts=0;baseline_raw=None;baseline_errors=None
        for attempt in (1,2):
            payload,receipt=make_payload(run,repair);proof=proof_for(m,payload,receipt,attempt)
            verify_request(run,payload,proof,repair)
            raw,folder,metadata=caller.call(m['slot_id']+'-assets-'+str(attempt),payload,proof);attempts+=1
            if raw is None:break
            accepted,errors,normalization=normalize_assets(raw,m,read(run/'SOURCE.json'))
            write(folder/'REFERENCE_NORMALIZATION.json',normalization)
            if attempt==1:
                baseline_raw,baseline_errors=raw,deepcopy(errors)
                guard=build_repair_guard(raw,errors,m,read(run/'SOURCE.json'))
            else:
                guard=build_repair_guard(baseline_raw,baseline_errors,m,read(run/'SOURCE.json'),raw,errors)
            write(folder/'REPAIR_GUARD.json',guard)
            if not guard['accepted']: accepted=None
            write(folder/'VALIDATION.json',{'mechanical_errors':errors,'fact_count':len(accepted['facts']) if accepted else None,
                'semantic_review_performed':False,'shortage_triggers_retry':False})
            if not errors or attempt==2 or not guard['repair_allowed']:break
            repair=build_repair(raw,errors,m,read(run/'SOURCE.json'))
        if accepted is not None:
            write(run/'ASSETS.json',accepted);export_public(run)
        result={'version':VERSION,'completed_at_utc':v15.now(),'root_id':m['root_id'],'slot_id':m['slot_id'],
            'status':('ASSETS_READY' if len(accepted['facts'])>=6 else 'INSUFFICIENT_FACTS') if accepted is not None else 'NO_VALID_ASSETS',
            'fact_count':len(accepted['facts']) if accepted is not None else 0,'requests_prepared':attempts,
            'http_attempts':len(list((run/'requests').glob('*/HTTP_ATTEMPT.json'))),
            'semantic_retries':0,'training_allowed':False,'source_first_review_required':True,
            'files':base.bindings([p for p in run.rglob('*') if p.is_file()])}
        write(run/'GENERATION.COMPLETE.json',result);verify(run,complete=True)
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','generate','verify','export-public'))
    parser.add_argument('--config',type=Path,default=HERE/'CONFIG.json');parser.add_argument('--slot');parser.add_argument('--run',type=Path)
    args=parser.parse_args()
    require(not args.run or args.slot is not None,'explicit_run_requires_slot')
    slots=[args.slot] if args.slot else [s['slot_id'] for s in read(args.config)['slots']]
    for slot in slots:
        run=args.run or HERE/'assets'/slot
        if args.action=='prepare': result=prepare(args.config,slot,run=run)
        elif args.action=='generate': result=generate(run)
        elif args.action=='export-public': result=export_public(run)
        else:
            m=verify(run);result={'status':'PASS','slot_id':m['slot_id']}
        print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__': main()
