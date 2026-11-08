"""Independent completed-run source, feedback lineage, and reservation checks."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import hashlib
import importlib.util
import json

def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def bind(path): return {'path': str(Path(path).resolve()), 'sha256': sha(path)}
def digest(x): return hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
def key(x): return (x['dataset'], x['split'], str(x['item_id']))
def owner(x): return (str(Path(x['run']).resolve()), x['root_id'], x['scene_id'])
def raw_context(text):
    position = text.index('INPUT_JSON\n') + len('INPUT_JSON\n')
    return json.JSONDecoder().raw_decode(text[position:])[0]

def check(run):
    run = Path(run).resolve()
    engine = run.parents[1]
    home = engine.parent
    manifest = read(run / 'MANIFEST.json')
    report = read(run / 'REPORT.json')
    roots = manifest['roots']
    selected = read(engine / 'SOURCE_SELECTION.json')
    assert roots == selected['roots']
    registry = {x['root_id']: x for x in read(engine / 'SOURCE_REGISTRY.json')['roots']}
    paths = {run / 'MANIFEST.json', run / 'REPORT.json', engine / 'SOURCE_SELECTION.json', engine / 'SOURCE_REGISTRY.json'}
    checks = []; failures = []
    def record(name, ok, **values):
        checks.append({'check': name, 'status': 'PASS' if ok else 'FAIL', **values})
        if not ok: failures.append(name)

    # Recompute recorded input bindings, then inspect original structured turns.
    for entry in manifest['files']:
        if isinstance(entry, dict) and 'path' in entry and 'sha256' in entry:
            record('manifest_binding:' + entry['path'], sha(entry['path']) == entry['sha256'])
            paths.add(Path(entry['path']))
    source_rows = []
    for root in roots:
        old = Path(registry[root]['source_path'])
        copied = run / 'sources' / (root + '.json')
        source = read(copied); original = read(old)
        record('original_history_hash:' + root, sha(old) == registry[root]['source_sha256'])
        record('complete_history_object_unchanged:' + root, source == original)
        record('complete_history_turn_count:' + root, len(source['turns']) == registry[root]['source_turns'])
        paths.update([old, copied])
        row = {'root_id': root, 'historical_turns': len(source['turns']), 'original': bind(old), 'run_source': bind(copied), 'cumulative_sources': []}
        for route in manifest['selected_routes']:
            letter = 'A' if route == 'old_event_continuation' else 'B'
            cumulative = run / 'roots' / root / letter / 'cumulative_dialogue.json'
            if cumulative.exists():
                cumulative_source = read(cumulative)
                turns = cumulative_source['turns']
                suffix = turns[len(source['turns']):]
                record('cumulative_frozen_prefix:' + root + ':' + route, turns[:len(source['turns'])] == source['turns'])
                record('cumulative_suffix_route:' + root + ':' + route, all(x['turn_ref'].startswith('R02:' + letter + ':') for x in suffix))
                record('cumulative_turn_refs_unique:' + root + ':' + route, len({x['turn_ref'] for x in turns}) == len(turns))
                result_path = cumulative.parent / 'RESULT.json'
                result = read(result_path)
                record('cumulative_report_hash:' + root + ':' + route, digest(turns) == result['output_history_sha256'])
                record('cumulative_added_turn_count:' + root + ':' + route, len(suffix) == result['added_turns'])
                paths.update([cumulative, result_path])
                row['cumulative_sources'].append({'route': route, 'total_turns': len(turns), 'added_turns': len(suffix), 'accepted_scenes': result['accepted_scenes'], 'source': bind(cumulative)})
        source_rows.append(row)

    writer_contexts = []
    for directory in sorted((run / 'requests').iterdir()):
        if not directory.is_dir(): continue
        proof = read(directory / 'PROOF.json')
        if proof['role'] not in ('planner', 'realizer'): continue
        request = read(directory / 'REQUEST.json')
        context = raw_context(request['messages'][0]['content'])
        frozen = read(run / 'sources' / (proof['root_id'] + '.json'))['turns']
        history = context['cumulative_history']
        ok = history[:len(frozen)] == frozen
        record('actual_writer_full_prefix:' + directory.name, ok)
        record('actual_writer_history_hash:' + directory.name, digest(history) == proof['history_sha256'])
        record('actual_writer_history_count:' + directory.name, len(history) == proof['history_turns'])
        record('actual_writer_no_other_branch:' + directory.name, proof['other_branch_in_generator_context'] is False)
        writer_contexts.append({'request': directory.name, 'root_id': proof['root_id'], 'role': proof['role'], 'historical_turns': len(frozen), 'actual_history_turns': len(history), 'complete_historical_prefix': ok})
        paths.update([directory / 'REQUEST.json', directory / 'PROOF.json'])

    compiled_path = run / 'feedback/COMPILED_FEEDBACK.json'
    lock_path = run / 'feedback/source_lock.json'
    compiled, lock = read(compiled_path), read(lock_path)
    historical = lock['historical_source_lock']
    record('feedback_selected_root_order', compiled['roots'] == roots)
    record('feedback_target_trained_model', compiled['target_solver_model'] == 'M_clean_policy')
    record('feedback_only_explicit_previous_round', historical['status'] == 'EXPLICIT_PREVIOUS_TRAINING_ROUND_ONLY')
    record('feedback_current_generation_merge_disabled', lock['merge_current_generation_feedback'] is False and historical['merge_current_generation_feedback'] is False)
    record('feedback_auto_latest_disabled', lock['automatic_latest_run_selection'] is False and historical['automatic_latest_run_selection'] is False)
    record('feedback_lock_hash', compiled['source_lock_sha256'] == sha(lock_path))
    lineage = []
    def lineage_walk(value):
        if isinstance(value, dict):
            if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
                p = Path(value['path']); record('feedback_lineage_binding:' + str(p), sha(p) == value['sha256'])
                lineage.append(bind(p)); paths.add(p)
            for child in value.values(): lineage_walk(child)
        elif isinstance(value, list):
            for child in value: lineage_walk(child)
    lineage_walk(lock)
    frozen_compiled = Path(lock['compiled_v4_dir'])
    for name, field in [('COMPILED_FEEDBACK.json', 'v4_manifest_sha256'), ('source_lock.json', 'v4_source_lock_sha256')]:
        p = frozen_compiled / name
        record('compiled_archive:' + name, sha(p) == lock[field]); paths.add(p)
    for row in compiled['records']:
        p = run / 'feedback' / row['path']
        record('compiled_feedback_suffix:' + row['path'], sha(p) == row['sha256'] and len(p.read_text()) == row['chars'])
        paths.add(p)
    paths.update([compiled_path, lock_path])

    inventory_before_path = run / 'INVENTORY_BEFORE.json'
    before = read(inventory_before_path)
    before_keys = {key(x) for x in before['accepted_uses']}
    reservation_path = run / 'SOURCE_RESERVATIONS.json'
    reservation = read(reservation_path)
    reserved_keys = {key(x) for x in reservation['reservations']}
    record('selected_roots_all_reserved_once', [x['root_id'] for x in reservation['reservations']] == roots)
    record('selected_sources_reserved_in_frozen_order', [x['source_id'] for x in reservation['reservations']] == selected['source_ids'])
    record('selected_sources_were_unused_before_reservation', not (reserved_keys & before_keys))
    for row in reservation['reservations']:
        allocation = next(x for x in selected['allocation'] if x['root_id'] == row['root_id'])
        record('reservation_source_hash:' + row['root_id'], row['source_id'] == allocation['source_id'] and row['source_item_sha256'] == allocation['source_item_sha256'])
    spec = importlib.util.spec_from_file_location('v97_actual_inventory_audit', engine / 'source_inventory.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    after = module.collect_prior()
    after_keys = {key(x) for x in after['accepted_uses']}
    record('permanent_inventory_delta_exact_selected_identities', after_keys == before_keys | reserved_keys)
    for row in reservation['reservations']:
        actual = next(x for x in after['accepted_uses'] if key(x) == key(row))
        owners = actual['historical_owners']
        record('reserved_identity_has_only_its_new_owner:' + row['root_id'], len(owners) == 1 and owner(owners[0]) == owner(row))
    record('historical_conflicts_not_silently_removed', len(before['historical_conflicts']) == len(after['historical_conflicts']))
    record('question_solver_training_not_in_expansion_run', report['question_calls'] == 0 and report['solver_calls'] == 0 and report['training_started'] is False)
    paths.update([inventory_before_path, reservation_path, engine / 'source_inventory.py'])
    return {'status': 'FAIL' if failures else 'PASS_STRUCTURAL_SOURCE_FEEDBACK_AND_RESERVATION_BINDINGS', 'checks': checks, 'failures': failures, 'source_roots': source_rows, 'actual_writer_full_history_checks': writer_contexts, 'feedback_lineage': lineage, 'historical_source_lock_evaluation_run': historical['evaluation_run_id'], 'feedback_current_generation_used': False, 'inventory_before_identities': len(before_keys), 'inventory_after_identities': len(after_keys), 'newly_permanently_reserved_identities': len(reserved_keys), 'historical_conflicts_preserved_count': len(after['historical_conflicts']), 'bindings': [bind(p) for p in sorted(paths)], 'semantic_certified': False, 'script': bind(__file__)}
