"""Bounded Realizer-only execution from an unchanged, Gate-complete source run.

No Planner, Gate, adherence, promotion, QA, Solver or Memory operation is exposed.
Prior structurally valid drafts provide local continuity, never promoted history.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from ..h4_round1_r03_incremental_plan_retention_v1 import planner_relations as relations
from ..h4_round1_r03_incremental_plan_retention_v1.planner_candidate_policy import candidate_limit
from ..h4_round1_r03_incremental_plan_retention_v1.planner_preparation import _validate_bundle
from ..h4_round1_r03_incremental_plan_retention_v1.planning import sha
from ..h4_round1_r03_incremental_plan_retention_v1.prompts import render_realizer
from ..h4_round1_r03_incremental_plan_retention_v1.shared_planner_contract import parse_output
from ..h4_round1_r03_incremental_plan_retention_v1.capacity import local_counter, measure
from ..h4_round1_r03_incremental_plan_retention_v1.budget import validate_budget
from ..h4_round1_r03_incremental_plan_retention_v1.run import SingleShotTransport, offline_network_guard
from ..h4_round1_r03_incremental_plan_retention_v1.transport_config import credential_environment
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.contracts import normalize_realizer_output

VERSION = 'h4_r03_realizer_only_sequential_draft_v1'
SCENE_QUOTA = 6
BATCH_SIZE = 2
MAX_REQUESTS = 3


def read(path):
    return json.loads(Path(path).read_text())


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def snapshot(directory):
    return {str(p): file_sha(p) for p in sorted(Path(directory).rglob('*')) if p.is_file()}


def project_scenes(compiled, decisions, sidecar):
    """Keep the existing compiler's six ranked slots and exact candidate identities."""
    prepared = compiled['prepared']
    registry = prepared['candidate_registry']
    refs = [row.get('candidate_ref') for row in decisions]
    if len(refs) != len(set(refs)) or set(refs) != set(registry):
        raise ValueError('gate_decisions_incomplete_or_duplicate')
    if any(row.get('candidate_id') != row['candidate_ref'] or row.get('decision') not in ('pass', 'reject') for row in decisions):
        raise ValueError('gate_decision_identity_or_status')
    passed = {row['candidate_ref'] for row in decisions if row['decision'] == 'pass'}
    selected = [str(scene['memory_event_id']) + ':C01' for scene in prepared['compiled']['scenes']]
    if len(selected) != SCENE_QUOTA or len(set(selected)) != SCENE_QUOTA:
        raise ValueError('compiler_scene_quota_not_six')
    if not set(selected) <= passed:
        raise ValueError('selected_candidate_rejected_no_automatic_substitution')
    active, omitted = [], []
    order = {ref: i for i, ref in enumerate(selected)}
    for relation in sidecar['relations']:
        row = deepcopy(relation)
        owner = row['owner_phase_id'].removeprefix('initial:')
        pred = row['predecessor_phase_id'].removeprefix('initial:')
        row.update(owner_candidate_ref=owner, predecessor_candidate_ref=pred)
        if owner not in selected:
            row['projection_status'] = 'OWNER_NOT_SELECTED'
            omitted.append(row)
        elif pred not in selected:
            row['projection_status'] = ('BLOCKED_BY_REJECTED_PREREQUISITE' if pred not in passed
                                        else 'PREDECESSOR_NOT_SELECTED')
            omitted.append(row)
            if row['relation_kind'] == relations.HARD:
                raise ValueError('selected_hard_prerequisite_unavailable_no_rebinding')
        else:
            if order[pred] >= order[owner]:
                raise ValueError('selected_relation_not_backward')
            if row['relation_kind'] == relations.HARD:
                # No semantic auditor is authorized in this Realizer-only stage.
                raise ValueError('hard_prerequisite_requires_actual_condition_review')
            row['projection_status'] = 'SELECTED_NATURAL_CONTINUITY'
            active.append(row)
    scenes = []
    for ref in selected:
        scene = deepcopy(registry[ref]['scene'])
        scene['phase_id'] = 'initial:' + ref
        scene['dependencies'] = [{
            'phase_id': row['predecessor_phase_id'],
            'predecessor_candidate_ref': row['predecessor_candidate_ref'],
            'relation_kind': row['relation_kind'], 'condition': row['condition'],
        } for row in active if row['owner_candidate_ref'] == ref]
        if scene['dependencies']:
            scene['dependency_interpretation'] = (
                'natural_continuity expresses a preferred natural sequence, not a required completed result. '
                'Use the actual earlier dialogue to continue; never infer achievement from a planned after_state.')
        scenes.append(scene)
    return scenes, {'selected_candidate_refs': selected, 'relations': active, 'omitted_relations': omitted,
                    'automatic_rebinding': False, 'formal_scene_ids_assigned_after_generation': True}


def prepare_source(source):
    source = Path(source).resolve()
    summary = read(source / 'run_summary.json')
    if summary['status'] != 'PLANNER_AND_GATE_COMPLETE':
        raise ValueError('source_gate_not_complete')
    if not read(source / 'planner/validation.json')['valid']:
        raise ValueError('source_planner_not_valid')
    identity = read(source / 'input_manifest.json')
    bundle_path = Path(identity['bundle_path']) / 'prepared/initial/bundle.json'
    if file_sha(bundle_path) != identity['bundle_sha256']:
        raise ValueError('source_bundle_file_changed')
    bundle = read(bundle_path)
    _validate_bundle(bundle)
    if bundle['identity']['root_id'] != summary['root_id']:
        raise ValueError('source_root_identity_mismatch')
    parsed = parse_output((source / 'planner/raw_response.json').read_text())
    if parsed != read(source / 'planner/parsed_plan.json'):
        raise ValueError('source_planner_raw_parsed_mismatch')
    if read(source / 'planner/lineage.json')['history_sha256'] != sha(bundle['parent']):
        raise ValueError('source_planner_history_mismatch')
    lineage = read(source / 'gate/lineage.json')
    for name, relpath in [('planner_parsed_plan_sha256', 'planner/parsed_plan.json'),
                          ('candidate_decisions_sha256', 'gate/candidate_decisions.json')]:
        if lineage[name] != file_sha(source / relpath):
            raise ValueError('source_gate_lineage_mismatch:' + name)
    compiled = relations.compile_initial_output(parsed, parent=bundle['parent'], candidates=bundle['candidates'],
        verified_tracks=bundle['verified_tracks'], max_new_candidates=candidate_limit(bundle))
    if not compiled['valid']:
        raise ValueError('source_planner_recompile_invalid:' + str(compiled['errors']))
    sidecar = read(source / 'planner/relation_metadata.json')
    if sidecar != compiled['relation_sidecar']:
        raise ValueError('source_relation_metadata_mismatch')
    decisions = read(source / 'gate/candidate_decisions.json')
    verified = {}
    for path in sorted((source / 'gate').glob('validation_*.json')):
        result = read(path)
        if not result['valid']:
            raise ValueError('source_gate_validation_invalid')
        for row in result['normalized_candidate_reviews']:
            ref = row['candidate_ref']
            if ref in verified:
                raise ValueError('duplicate_source_gate_validation')
            verified[ref] = row['decision']
    if verified != {row['candidate_ref']: row['decision'] for row in decisions}:
        raise ValueError('source_gate_validation_decision_mismatch')
    scenes, projection = project_scenes(compiled, decisions, sidecar)
    budget = deepcopy(read(source / 'budget.json'))
    budget['role_requests'] = {role: MAX_REQUESTS if role == 'realizer' else 0 for role in budget['role_requests']}
    budget['total_requests'] = MAX_REQUESTS
    budget['max_generated_tokens'] = MAX_REQUESTS * budget['max_output_tokens']['realizer']
    validate_budget(budget)
    model = read(source / 'model_config.json')
    counter = local_counter(model['tokenizer_path'], model['tokenizer_json_sha256'])
    prompt = render_realizer(parent=bundle['parent'], plan={'scenes': scenes[:2]}, mode='text_free')
    capacity = measure(prompt['messages'], counter=counter, budget=budget, role='realizer')
    if not capacity['fits']:
        raise ValueError('first_batch_input_exceeds_capacity')
    return {'root_id': summary['root_id'], 'source_run': str(source), 'bundle_file': str(bundle_path),
            'parent': bundle['parent'], 'scenes': scenes, 'relation_projection': projection,
            'relation_sidecar': sidecar, 'model': model, 'budget': budget,
            'first_batch_capacity': capacity, 'source_hashes': snapshot(source),
            'bundle_sha256': file_sha(bundle_path)}


def draft_turns(scenes):
    return [deepcopy(t) for scene in scenes for t in scene['dialogue_turns']]


def request_for(prepared, ordinal, generated):
    scenes = prepared['scenes'][(ordinal - 1) * 2:ordinal * 2]
    prefix = draft_turns(generated)
    built = render_realizer(parent=prepared['parent'], plan={'scenes': scenes},
                            mode='text_free', promoted_prefix=prefix)
    request = {'role': 'realizer', 'call_id': f'realizer-{ordinal:03d}', 'messages': built['messages'],
               'max_tokens': prepared['budget']['max_output_tokens']['realizer'],
               'prompt_version': built['prompt_version'], 'model': prepared['model']['model'],
               'temperature': prepared['model']['temperature'], 'thinking': False,
               'identity': {'root_id': prepared['root_id'], 'round_index': 3, 'batch_index': ordinal,
                            'candidate_refs': [s['selected_candidate_ref'] for s in scenes], 'run_version': VERSION}}
    return request, built, scenes, prefix


def execute(prepared, output, backend, *, before_request=lambda: None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, data in [('budget', prepared['budget']), ('model_config', prepared['model']),
                       ('selected_plan', {'scenes': prepared['scenes']}),
                       ('relation_projection', prepared['relation_projection'])]:
        write(output / (name + '.json'), data)
    write(output / 'source_manifest.json', {key: prepared[key] for key in
        ('source_run', 'root_id', 'bundle_file', 'bundle_sha256', 'source_hashes')})
    write(output / 'authorization_scope.json', {'run_version': VERSION, 'scope': 'other_gate_complete_roots_realizer_only',
        'authorized_max_calls': {'realizer': MAX_REQUESTS}, 'retry': 0,
        'forbidden_calls': ['planner', 'gate', 'supplement', 'adherence', 'qa', 'solver', 'training', 'memorax_write'],
        'formal_promotion': False, 'prefix_policy': 'earlier_structurally_valid_drafts_only'})
    generated, ledger = [], []
    error = None
    counter = local_counter(prepared['model']['tokenizer_path'], prepared['model']['tokenizer_json_sha256'])
    def finish(status):
        nonlocal error
        source_unchanged = None
        if status not in ('PREPARED_NOT_GENERATED', 'REALIZER_RUNNING'):
            source_unchanged = snapshot(prepared['source_run']) == prepared['source_hashes']
            write(output / 'source_unchanged.json', {'all_source_files_unchanged': source_unchanged})
            if not source_unchanged:
                error = {'type': 'SOURCE_FILES_CHANGED_DURING_EXECUTION', 'prior_error': error}
                status = 'SOURCE_INTEGRITY_CHANGED'
        write(output / 'generated_child.json', {'schema_version': 'memory_event_continuation_child_v1',
            'root_id': prepared['root_id'], 'round_index': 3, 'scenes': generated})
        write(output / 'generated_dialogue.json', {'root_id': prepared['root_id'], 'round_index': 3,
            'authority_status': 'GENERATED_DRAFT_NOT_PROMOTED', 'turn_count': len(draft_turns(generated)),
            'turns': draft_turns(generated)})
        summary = {'status': status, 'root_id': prepared['root_id'], 'run_version': VERSION,
            'realizer_real_calls': len(ledger), 'planner_real_calls': 0, 'gate_real_calls': 0, 'other_real_calls': 0,
            'successful_batches': len(generated) // 2, 'scene_count': len(generated),
            'turn_count': len(draft_turns(generated)), 'input_mode': 'text_free',
            'qa': 0, 'solver': 0, 'training': 0, 'memorax_write': 0, 'adherence': 0, 'promotion': 0,
            'semantic_validation': 'NOT_RUN', 'automatic_retry': False, 'error': error,
            'source_files_unchanged': source_unchanged}
        write(output / 'run_summary.json', summary)
        return summary
    finish('PREPARED_NOT_GENERATED')
    for ordinal in range(1, MAX_REQUESTS + 1):
        batch = output / 'batches' / f'{ordinal:03d}'
        request, built, scenes, prefix = request_for(prepared, ordinal, generated)
        write(batch / 'request.json', request)
        (batch / 'prompt.txt').write_text(built['prompt'])
        write(batch / 'plan.json', {'scenes': scenes})
        write(batch / 'request_manifest.json', {'request_sha256': sha(request), 'prompt_version': request['prompt_version'],
            'adapter_version': VERSION, 'prefix_turn_count': len(prefix), 'prefix_sha256': sha(prefix),
            'prefix_status': 'GENERATED_DRAFT_STRUCTURE_VALID_NOT_PROMOTED',
            'authoritative_history_included': False, 'qa_gold_solver_feedback_included': False,
            'candidate_refs': request['identity']['candidate_refs'], 'retry_count': 0})
        capacity = measure(request['messages'], counter=counter, budget=prepared['budget'], role='realizer')
        write(batch / 'capacity.json', capacity)
        if not capacity['fits']:
            error = {'batch': ordinal, 'type': 'INPUT_CAPACITY_EXCEEDED'}
            return finish('REALIZER_STOPPED_ON_CAPACITY')
        if len(ledger) >= prepared['budget']['role_requests']['realizer']:
            raise ValueError('realizer_budget_exhausted')
        backend.response_evidence_sink = lambda ev, path=batch: write(path / ('http_' + ev['evidence_event'] + '.json'), ev)
        before_request()
        entry = {'role': 'realizer', 'call_id': request['call_id'], 'request_sha256': sha(request),
                 'attempt_count': 1, 'started_at': datetime.now(timezone.utc).isoformat()}
        ledger.append(entry)
        write(output / 'call_attempts.json', ledger)
        finish('REALIZER_RUNNING')
        print(prepared['root_id'], request['call_id'], 'REQUEST_STARTED', flush=True)
        try:
            raw = backend.request(request)
            (batch / 'raw_response.json').write_text(raw)
            if backend.last_response_envelope is not None:
                write(batch / 'response_envelope.json', backend.last_response_envelope)
            parsed = parse_output(raw)
            write(batch / 'parsed_response.json', parsed)
            local_parent = deepcopy(prepared['parent'])
            local_parent['dialogue']['turns'].extend(prefix)
            result = normalize_realizer_output(parsed, plan={'scenes': scenes}, parent=local_parent,
                                               round_index=3, scene_index_offset=(ordinal - 1) * 2)
            write(batch / 'normalization_result.json', result)
            write(batch / 'validation.json', {k: v for k, v in result.items() if k != 'child'})
            if not result['valid']:
                error = {'batch': ordinal, 'type': 'STRUCTURE_OR_COPY_REJECT', 'errors': result['errors']}
                return finish('REALIZER_OUTPUT_INVALID_STOP')
            child = deepcopy(result['child'])
            changes = []
            for i, scene in enumerate(child['scenes']):
                changes.append({'path': f'/child/scenes/{i}/session_ref', 'before': scene['session_ref'],
                                'after': scene['scene_ref'], 'rule': 'existing_workflow_session_ref_equals_scene_ref'})
                scene['session_ref'] = scene['scene_ref']
            write(batch / 'normalization_diff.json', {'raw_sha256': sha(parsed),
                'normalized_child_sha256': sha(child), 'session_ref_changes': changes,
                'other_deterministic_normalization': 'existing normalizer: speaker/text trim; local scene/turn IDs; memory_event binding',
                'raw_and_pre_normalization_preserved': True})
            write(batch / 'normalized_child.json', child)
            generated.extend(child['scenes'])
            phase_map = {'initial:' + p['selected_candidate_ref']: g['scene_ref']
                         for p, g in zip(prepared['scenes'], generated)}
            write(output / 'actual_scene_relation_mapping.json', relations.map_formal_scene_ids(prepared['relation_sidecar'], phase_map))
            finish('REALIZER_RUNNING')
            print(prepared['root_id'], request['call_id'], 'STRUCTURE_VALID', len(draft_turns(child['scenes'])), 'turns', flush=True)
        except Exception as exc:
            error = {'batch': ordinal, 'type': type(exc).__name__, 'retry': False}
            write(batch / 'error.json', error)
            return finish('REALIZER_TECHNICAL_OR_PARSE_ERROR_STOP')
    return finish('REALIZER_GENERATION_COMPLETE_STRUCTURE_VALID')


class Pacer:
    def __init__(self, interval):
        self.interval, self.last = interval, None
    def __call__(self):
        if self.last is not None:
            remaining = self.interval - (time.monotonic() - self.last)
            if remaining > 0:
                print('COOLDOWN', round(remaining, 1), 'seconds', flush=True)
            while remaining > 0:
                time.sleep(min(remaining, 10))
                remaining = self.interval - (time.monotonic() - self.last)
        self.last = time.monotonic()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', action='append', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--credential-config', type=Path)
    p.add_argument('--request-start-interval', type=float, default=65)
    args = p.parse_args()
    if args.execute and not args.credential_config:
        p.error('execute requires explicit project credential source')
    prepared = []
    with offline_network_guard():
        for source in args.source_run:
            prepared.append(prepare_source(source))
    args.output.mkdir(parents=True, exist_ok=False)
    scope = {'run_version': VERSION, 'roots': [x['root_id'] for x in prepared],
             'per_root_realizer_max': MAX_REQUESTS, 'total_realizer_max': len(prepared) * MAX_REQUESTS,
             'other_role_call_max': 0, 'status': 'PREFLIGHT_PASS', 'execute': args.execute,
             'runner_file': str(Path(__file__).resolve()), 'runner_sha256': file_sha(__file__)}
    write(args.output / 'scope.json', scope)
    if not args.execute:
        for item in prepared:
            write(args.output / 'preflight' / (item['root_id'] + '.json'), item)
        print(json.dumps(scope), flush=True)
        return
    pacer = Pacer(args.request_start_interval)
    results = []
    credential_source = {'path': str(args.credential_config.resolve()), 'sha256': file_sha(args.credential_config)}
    for item in prepared:
        with credential_environment(item['model'], credential_source):
            backend = SingleShotTransport(item['model'], item['budget'])
            results.append(execute(item, args.output / 'roots' / item['root_id'], backend, before_request=pacer))
        write(args.output / 'run_summary.json', {'results': results,
              'realizer_real_calls': sum(x['realizer_real_calls'] for x in results),
              'other_real_calls': 0, 'remaining_roots': len(prepared) - len(results)})
        print(json.dumps(results[-1]), flush=True)


if __name__ == '__main__':
    main()
