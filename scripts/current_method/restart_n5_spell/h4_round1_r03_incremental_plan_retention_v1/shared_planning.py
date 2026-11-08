"""Compile one shared Planner contract into the existing immutable scene model."""
from copy import deepcopy

from .planning import prepare_original, sha
from .shared_planner_contract import VERSION, iter_candidates, strip_relations, validate_output, parse_output
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.contracts import (
    build_speaker_registry, materialize_planner_speaker_refs,
)

IMMUTABLE_CONTINUATION_FIELDS = (
    'verified_track_ref', 'new_people', 'goal', 'speaker_refs',
    'transition_from_previous', 'after_state', 'actor_bindings',
)
CONTINUATION_FIELDS = IMMUTABLE_CONTINUATION_FIELDS + ('required_transition', 'interaction_beats')


def _failure(raw, errors, **extra):
    return {'valid': False, 'errors': list(errors), 'planner_contract_version': VERSION,
            'shared_output': deepcopy(raw), **extra}


def _initial_pointer_refs(raw, verified_tracks):
    tracks = {row['verified_track_ref']: row for row in verified_tracks}
    refs = {}
    for pointer, row, collection in iter_candidates(raw):
        ordinal = int(pointer.rsplit('/', 1)[1]) + 1
        event = (tracks[row['verified_track_ref']]['memory_event_ref']
                 if collection == 'selected_continuations' else f'ND{ordinal:02d}')
        refs[pointer] = str(event) + ':C01'
    return refs


def _translate_dependencies(raw, phase_map, *, source_refs=()):
    earlier, translated, errors = set(), {}, []
    for pointer, row, _ in iter_candidates(raw):
        dependencies = []
        for dependency in row['dependencies']:
            source, output = dependency['source_candidate_ref'], dependency['output_ref']
            if source is not None:
                if source not in source_refs:
                    errors.append(pointer + ':unknown_or_forbidden_dependency_source:' + source)
                    continue
                target = 'initial:' + source
            else:
                if output not in earlier:
                    errors.append(pointer + ':unknown_self_or_forward_output_dependency:' + output)
                    continue
                if output not in phase_map:
                    errors.append(pointer + ':dependency_target_unavailable:' + output)
                    continue
                target = phase_map[output]
            dependencies.append({'phase_id': target, 'condition': dependency['condition']})
        translated[pointer] = dependencies
        earlier.add(pointer)
    return translated, errors


def _attach_initial_relations(prepared, raw, verified_tracks):
    result = deepcopy(prepared)
    pointer_refs = _initial_pointer_refs(raw, verified_tracks)
    phase_map = {pointer: 'initial:' + ref for pointer, ref in pointer_refs.items()
                 if ref in result['candidate_registry']}
    dependencies, errors = _translate_dependencies(raw, phase_map)
    for pointer, row, _ in iter_candidates(raw):
        ref = pointer_refs[pointer]
        if ref not in result['candidate_registry']:
            if row['dependencies']:
                errors.append(pointer + ':dependency_owner_unavailable')
            continue
        result['candidate_registry'][ref]['scene']['dependencies'] = dependencies[pointer]
    if errors:
        return _failure(raw, errors, prepared=result)
    result.update(planner_contract_version=VERSION, initial_output_phase_map=phase_map,
                  _initial_shared_output=deepcopy(raw))
    if 'source_scope' in result:
        result['source_scope']['candidate_registry_sha256'] = sha(result['candidate_registry'])
    return {'valid': True, 'errors': [], 'prepared': result, 'planner_contract_version': VERSION,
            'shared_output': deepcopy(raw), 'value': deepcopy(result['original'])}


def compile_initial_output(raw, *, parent, candidates, verified_tracks, max_new_candidates=8):
    """Validate initial relations before the legacy speaker and target checks."""
    if isinstance(raw, str):
        try:
            raw = parse_output(raw)
        except ValueError as exc:
            return _failure(raw, ['shared_initial_json_invalid:' + str(exc)])
    structural = validate_output(raw, max_new_candidates=max_new_candidates)
    if not structural['valid']:
        return _failure(raw, structural['errors'])
    errors = []
    if len(raw['selected_continuations']) > 3:
        errors.append('initial_continuations_max_3')
    tracks = {row['verified_track_ref']: row for row in verified_tracks}
    seen_tracks = set()
    for pointer, row, collection in iter_candidates(raw):
        if row['source_candidate_ref'] is not None:
            errors.append(pointer + ':initial_source_must_be_null')
        if collection == 'selected_continuations':
            track = row['verified_track_ref']
            if track is None or track not in tracks:
                errors.append(pointer + ':initial_verified_track_unknown')
            elif track in seen_tracks:
                errors.append(pointer + ':initial_verified_track_duplicate')
            seen_tracks.add(track)
    if errors:
        return _failure(raw, errors)
    pointer_refs = _initial_pointer_refs(raw, verified_tracks)
    _, errors = _translate_dependencies(raw, {pointer: 'initial:' + ref for pointer, ref in pointer_refs.items()})
    if errors:
        return _failure(raw, errors)
    materialized = materialize_planner_speaker_refs(strip_relations(raw), parent=parent)
    if not materialized['valid']:
        return _failure(raw, materialized['errors'])
    prepared = prepare_original(materialized['value'], parent=parent,
                                candidates=candidates, verified_tracks=verified_tracks,
                                max_new_candidates=max_new_candidates)
    if not prepared['valid_for_new_policy']:
        return _failure(raw, prepared['hard_errors'], value=materialized['value'], prepared=prepared)
    return _attach_initial_relations(prepared, raw, verified_tracks)


def apply_initial_relations(prepared, shared_output, *, parent, candidates, verified_tracks):
    """Replay compilation before attaching relations to a verified root registry."""
    compiled = compile_initial_output(shared_output, parent=parent,
                                      candidates=candidates, verified_tracks=verified_tracks)
    if not compiled['valid']:
        return compiled
    if compiled['value'] != prepared['original'] or sha(prepared['original']) != prepared['original_sha256']:
        return _failure(shared_output, ['initial_shared_output_canonical_mismatch'])
    expected = compiled['prepared']['candidate_registry']
    actual = deepcopy(prepared['candidate_registry'])
    for entry in expected.values():
        entry['scene'].pop('dependencies', None)
    for entry in actual.values():
        entry['scene'].pop('dependencies', None)
    if actual != expected:
        return _failure(shared_output, ['initial_shared_registry_mismatch'])
    return _attach_initial_relations(prepared, shared_output, verified_tracks)


def prepare_shared_original(bundle):
    """Bind frozen inputs before replaying shared initial relations."""
    from .root_planning import prepare_root_original
    prepared = prepare_root_original(bundle)
    if not prepared['valid_for_new_policy'] or 'initial_planner_output' not in bundle:
        return prepared
    replay = apply_initial_relations(prepared, bundle['initial_planner_output'],
        parent=bundle['parent'], candidates=bundle['candidates'],
        verified_tracks=bundle['verified_tracks'])
    if not replay['valid']:
        raise ValueError('shared_initial_output_invalid:' + ','.join(replay['errors']))
    return replay['prepared']


def _source_original_rows(prepared):
    original = prepared['original']
    for ref, entry in prepared['candidate_registry'].items():
        gate = entry['gate_scene']
        if gate.get('verified_track_ref') is not None:
            matches = [(index, row) for index, row in enumerate(original['selected_continuations'])
                       if row['verified_track_ref'] == gate['verified_track_ref']]
            if len(matches) != 1:
                raise ValueError('shared_source_track_identity_mismatch:' + ref)
            index, row = matches[0]
            pointer = f'/selected_continuations/{index}'
        else:
            index = int(gate['memory_event_id'].removeprefix('ND')) - 1
            row = original['new_development_candidates'][index]
            pointer = f'/new_development_candidates/{index}'
        yield ref, pointer, row


def source_continuation_rows(prepared, parent=None):
    """Return complete flat continuation templates in frozen registry order."""
    parent = parent or prepared.get('_root_materialization_inputs', {}).get('parent')
    if parent is None:
        raise ValueError('shared_source_parent_required')
    name_to_ref = {row['canonical_name']: row['speaker_ref']
                   for row in build_speaker_registry(parent)['speakers']}
    shared = prepared.get('_initial_shared_output')
    originals = {pointer: row for pointer, row, _ in iter_candidates(shared)} if shared else {}
    sources = {}
    for ref, pointer, original in _source_original_rows(prepared):
        if pointer in originals:
            raw = deepcopy(originals[pointer])
        else:
            raw = deepcopy(original)
            declared = {row['name'] for row in raw['new_people']}
            participants = raw.pop('participants')
            if any(name not in name_to_ref and name not in declared for name in participants):
                raise ValueError('shared_source_participant_not_declared:' + ref)
            raw['speaker_refs'] = [name_to_ref[name] for name in participants if name in name_to_ref]
            raw['actor_bindings'] = [{'actor_ref': name_to_ref[binding['actor']], 'binding': binding['binding']}
                                     for binding in raw['actor_bindings']]
        raw['verified_track_ref'] = prepared['candidate_registry'][ref]['gate_scene'].get('verified_track_ref')
        sources[ref] = {field: deepcopy(raw[field]) for field in CONTINUATION_FIELDS}
        sources[ref].update(source_candidate_ref=ref, dependencies=[])
    return list(sources.values())


def _check_source_materialization(raw, source, *, parent):
    # An existing ND declaration is not a newly introduced continuation person.
    # The neutral collection avoids the legacy continuation-only prohibition.
    fields = {field: deepcopy(raw[field]) for field in CONTINUATION_FIELDS if field != 'verified_track_ref'}
    fields.update(source_basis=['Frozen existing source identity.'], before_state='Frozen source navigation.')
    converted = materialize_planner_speaker_refs({'round_goal': 'Source identity check',
        'selected_continuations': [], 'new_development_candidates': [fields]}, parent=parent)
    if not converted['valid']:
        return converted['errors']
    canonical = converted['value']['new_development_candidates'][0]
    errors = []
    for field in ('goal', 'after_state', 'transition_from_previous', 'participants', 'actor_bindings'):
        if canonical[field] != source['scene'][field]:
            errors.append('shared_continuation_canonical_source_changed:' + field)
    declarations = [{key: row[key] for key in ('name', 'relationship_basis')}
                    for row in source['scene']['new_people']]
    if canonical['new_people'] != declarations:
        errors.append('shared_continuation_canonical_source_changed:new_people')
    return errors


def merge_shared_output(prepared, raw, *, allow_unconditional_continuations=False):
    """Adapt validated shared additions to the private legacy root patch."""
    if isinstance(raw, str):
        try:
            raw = parse_output(raw)
        except ValueError as exc:
            return _failure(raw, ['shared_append_json_invalid:' + str(exc)], raw_patch=raw)
    structural = validate_output(raw)
    if not structural['valid']:
        return _failure(raw, structural['errors'], raw_patch=deepcopy(raw))
    errors = []
    if raw['round_goal'] != prepared['original']['round_goal']:
        errors.append('shared_round_goal_changed')
    stored = prepared.get('_root_materialization_inputs') or {}
    if not stored.get('parent'):
        return _failure(raw, ['shared_root_materialization_inputs_required'], raw_patch=deepcopy(raw))
    try:
        sources = {row['source_candidate_ref']: row for row in source_continuation_rows(prepared)}
    except (KeyError, ValueError, IndexError) as exc:
        return _failure(raw, ['shared_source_view_invalid:' + str(exc)], raw_patch=deepcopy(raw))
    phase_map = {}
    for pointer, row, collection in iter_candidates(raw):
        index = int(pointer.rsplit('/', 1)[1]) + 1
        kind = 'continuation' if collection == 'selected_continuations' else 'development'
        phase_map[pointer] = f'phase:append-{kind}-{index}'
        source_ref = row['source_candidate_ref']
        if collection == 'selected_continuations':
            if source_ref not in sources:
                errors.append(pointer + ':unknown_continuation_source')
                continue
            if not row['dependencies'] and not allow_unconditional_continuations:
                errors.append(pointer + ':continuation_dependency_required')
            for field in IMMUTABLE_CONTINUATION_FIELDS:
                if row[field] != sources[source_ref][field]:
                    errors.append(pointer + ':immutable_source_field_changed:' + field)
            errors.extend(pointer + ':' + error for error in _check_source_materialization(
                row, prepared['candidate_registry'][source_ref], parent=stored['parent']))
        elif source_ref is not None:
            errors.append(pointer + ':independent_source_must_be_null')
    dependencies, dependency_errors = _translate_dependencies(raw, phase_map, source_refs=sources)
    errors.extend(dependency_errors)
    if errors:
        return _failure(raw, errors, raw_patch=deepcopy(raw))
    patch = {'status': 'PATCH' if phase_map else 'NO_SUPPLEMENT',
             'original_plan_sha256': prepared['original_sha256'], 'phases': [],
             'new_developments': [], 'reason': 'Compiled shared Planner additions; original plan preserved.'}
    stripped = strip_relations(raw)
    for pointer, row, collection in iter_candidates(raw):
        if collection == 'selected_continuations':
            patch['phases'].append({'phase_id': phase_map[pointer],
                'source_candidate_id': row['source_candidate_ref'], 'original_goal': row['goal'],
                'required_transition': row['required_transition'],
                'interaction_beats': deepcopy(row['interaction_beats']), 'dependencies': dependencies[pointer]})
        else:
            index = int(pointer.rsplit('/', 1)[1])
            patch['new_developments'].append({'phase_id': phase_map[pointer],
                'dependencies': dependencies[pointer],
                'candidate': deepcopy(stripped['new_development_candidates'][index])})
    from .root_planning import merge_root_patch
    before = sha(prepared)
    result = merge_root_patch(prepared, patch,
                              allow_unconditional_continuations=allow_unconditional_continuations)
    if sha(prepared) != before:
        raise ValueError('shared_merger_mutated_original_prepared')
    result.update(raw_patch=deepcopy(raw), compiled_patch=patch,
                  shared_output=deepcopy(raw), planner_contract_version=VERSION,
                  output_phase_map=phase_map)
    if result['valid']:
        if (result['merged']['original_plan'] != prepared['original']
                or result['merged']['original_sha256'] != prepared['original_sha256']):
            raise ValueError('shared_merge_original_changed')
        result['merged']['planner_contract_version'] = VERSION
        result['merged']['version_chain'][-1].update(patch_sha256=sha(raw), compiled_patch_sha256=sha(patch))
        result['merged_sha256'] = sha(result['merged'])
    return result


materialize_initial_output = compile_initial_output
continuation_source_view = source_continuation_rows
