"""Append-only v89 typed export, independent replay, and final exact-fileset seal.

Specify every successful shard with --shard. closeout also requires every failed
shard with --failed-shard and its pre-existing FAILED_<SHARD>_SNAPSHOT.json.
--check-only is read-only. --verify-complete is read-only and needs only --home.
Do not redirect output into the pilot during or after closeout: the complete
manifest is the final write and every other pilot file is bound by it.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
DEFAULT_HOME = HERE.parent
EXPORT_NAMES = ('AUDITED_NEW_CANDIDATES.json', 'CUMULATIVE_AUDITED_NEW_CANDIDATES.json', 'evidence/EXPORT_AUDIT.json')
CLOSEOUT_NAMES = ('evidence/HISTORICAL_PRESERVATION_AUDIT.json', 'evidence/FAILED_SHARDS_AUDIT.json',
                  'evidence/SOURCE_SCREENING_FINAL.json', 'evidence/PILOT_EXECUTION_SUMMARY.json',
                  'REPORT.md', 'ARTIFACT_INDEX.md', 'PILOT_COMPLETE.json')


def need(value, message):
    if not value:
        raise ValueError('pilot_closeout:' + message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate_json_key:' + key); result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda value: need(False, 'nonfinite_json:' + value))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}


def check_bind(value):
    need(set(value) == {'path', 'sha256'} and Path(value['path']).is_absolute()
         and bind(value['path']) == value, 'binding_changed:' + str(value.get('path')))


def write(path, value):
    path = Path(path)
    need(path.parent.is_dir(), 'output_parent_missing:' + str(path.parent))
    with path.open('x') as stream:
        if isinstance(value, str):
            stream.write(value)
        else:
            json.dump(value, stream, ensure_ascii=False, indent=2); stream.write('\n')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def files_under(path):
    path = Path(path).resolve(); entries = list(path.rglob('*'))
    need(not any(p.is_symlink() for p in entries), 'symlink_not_allowed:' + str(path))
    return {str(p.resolve()) for p in entries if p.is_file()}


def exact_snapshot(snapshot, roots):
    expected = snapshot['files']; need(isinstance(expected, dict) and expected, 'empty_preservation_snapshot')
    actual = set().union(*(files_under(root) for root in roots))
    need(actual == set(expected), 'preservation_inventory_difference:' + json.dumps({
        'added': sorted(actual - set(expected)), 'missing': sorted(set(expected) - actual)}))
    for path, digest in expected.items():
        need(sha(path) == digest, 'preservation_bytes_changed:' + path)
    return {'status': 'PASS', 'file_count': len(expected), 'exact_file_inventory': True,
            'roots': [str(Path(p).resolve()) for p in roots], 'all_sha256_recomputed': True}


def historical_preservation(home):
    path = home / 'PRESERVATION_BEFORE.json'; snapshot = read(path)
    delivery = home.parent; roots = set()
    for name in snapshot['files']:
        absolute = Path(name)
        need(absolute.is_absolute() and absolute.resolve() == absolute, 'noncanonical_snapshot_path')
        relative = absolute.relative_to(delivery)
        need(len(relative.parts) >= 2 and relative.parts[0] != home.name, 'invalid_protected_directory')
        roots.add(delivery / relative.parts[0])
    result = exact_snapshot(snapshot, sorted(roots))
    if 'protected_count' in snapshot:
        need(snapshot['protected_count'] == result['file_count'], 'protected_count_disagrees')
    return result | {'snapshot': bind(path), 'training_performed': False}


def latest_pass(shard):
    candidates = []
    for path in (shard / 'evidence').glob('RESULT_AUDIT*.json'):
        row = read(path)
        if row.get('status') == 'PASS' and row.get('phase') == 'paired_results':
            when = datetime.fromisoformat(row['completed_at_utc'])
            need(when.utcoffset() is not None, 'naive_audit_timestamp')
            candidates.append((when, path))
    need(candidates, 'missing_result_audit_pass:' + str(shard))
    return bind(max(candidates, key=lambda row: (row[0], str(row[1])))[1])


def selected_shards(home, shards):
    need(shards and len(shards) == len(set(shards)), 'explicit_unique_successful_shards_required')
    result = []
    for shard in shards:
        need(shard.parent == home / 'shards' and shard.is_dir(), 'shard_must_be_installed_under_pilot')
        result.append({'home': str(shard), 'result_audit': latest_pass(shard)})
    return result


def fingerprint(q):
    field = 'options' if q['question_type'] == 'multi_select' else 'items'
    return q['root_id'], q['question_type'], tuple(sorted(' '.join(t.split()).casefold() for t in q[field].values()))


def deduplicate(rows):
    kept = []; seen = {}; ids = {}; removed = []
    for q in rows:
        identity = q['candidate_id']; key = fingerprint(q)
        need(identity not in ids or ids[identity] == q, 'candidate_id_content_collision:' + identity)
        ids[identity] = q
        if key in seen:
            removed.append({'candidate_id': identity, 'retained_candidate_id': seen[key],
                            'reason': 'same_root_type_all_six_normalized_statements'})
        else:
            seen[key] = identity; kept.append(q)
    return kept, removed


def pool(rows, status):
    typed = {kind: sum(q['question_type'] == kind for q in rows) for kind in ('multi_select', 'ordering')}
    gold = {str(i): sum(q['question_type'] == 'multi_select' and len(q['gold']) == i for q in rows) for i in range(2, 6)}
    roots = dict(sorted(Counter(q['root_id'] for q in rows).items()))
    return {'created_at_utc': now(), 'status': status, 'questions': rows, 'candidate_count': len(rows),
            'by_type': typed, 'type_distribution': typed, 'gold_distribution': gold,
            'multiselect_gold_distribution': gold, 'root_distribution': roots, 'source_root_count': len(roots),
            'source_split': 'train', 'independent_test': False, 'training_allowed': False,
            'training_started': False, 'formal_training_admission': False, 'merged_into_baseline': False,
            'deduplication': 'root_id + question_type + sorted whitespace-normalized casefolded all six statements',
            'review_limitation': 'Saved source/content judgments and paired Solver evidence support candidate admission. Shared source/parent families remain dependencies; no semantic guarantee or independent test claim.'}


def build_exports(home, baseline, entries):
    protection = read(home / 'PRESERVATION_BEFORE.json')['files']
    need(str(baseline) in protection and sha(baseline) == protection[str(baseline)], 'v88_baseline_not_protected_or_changed')
    need(baseline == home.parent / 'h4_anchor_records_mixed_batch_v88/CUMULATIVE_AUDITED_NEW_CANDIDATES.json', 'baseline_must_be_v88_cumulative')
    all_new = []; controls = []
    for entry in entries:
        shard = Path(entry['home']); audit = read(entry['result_audit']['path'])
        need(audit['status'] == 'PASS', 'result_pass_required')
        questions = read(shard / 'EVALUATION_POOL.json')['questions']
        selected = audit['selected_new_ids']; need(len(selected) == len(set(selected)), 'duplicate_selected_id')
        rows = [q for q in questions if q['candidate_id'] in selected]
        need([q['candidate_id'] for q in rows] == selected, 'selected_new_projection')
        need(all(q['question_type'] == 'ordering' or (q['question_type'] == 'multi_select' and 2 <= len(q['gold']) <= 5) for q in rows), 'typed_gold_projection')
        all_new.extend(rows); controls.extend(audit['controls'])
    need(len({q['candidate_id'] for q in all_new}) == len(all_new), 'cross_shard_duplicate_id')
    new, removed = deduplicate(all_new)
    new_pool = pool(new, 'CONTENT_REVIEW_SOLVER_SELECTION_AND_INDEPENDENT_AUDIT_PASS')
    new_pool.update(new_question_count=len(new), shards=entries, historical_controls_excluded=controls,
                    selected_new_ids_before_deduplication=[q['candidate_id'] for q in all_new], duplicates_removed=removed)
    old = read(baseline); cumulative, union_removed = deduplicate(old['questions'] + new)
    cumulative_pool = pool(cumulative, 'DEDUPLICATED_CANDIDATE_INVENTORY')
    cumulative_pool.update(baseline=bind(baseline), baseline_candidate_count=len(old['questions']),
                           added_candidate_count=len(cumulative) - len(old['questions']), duplicates_removed=union_removed,
                           shards=entries, source_pools=[bind(baseline)],
                           limitations=['Not a new training run or independent validation set.', 'All prior candidates are carried forward unchanged; no retrospective source-semantic recertification.'])
    return new_pool, cumulative_pool


def failed_shards(home, failed, successful):
    need(len(failed) == len(set(failed)) and not set(failed) & set(successful), 'shard_classification_overlap')
    actual = {p.resolve() for p in (home / 'shards').iterdir() if p.is_dir()}
    need(actual == set(failed) | set(successful), 'unclassified_or_missing_shard:' + str(sorted(actual - set(failed) - set(successful))))
    output = []
    for shard in failed:
        need(shard.parent == home / 'shards', 'failed_shard_location')
        path = home / 'evidence' / ('FAILED_' + shard.name.upper() + '_SNAPSHOT.json')
        snapshot = read(path); evidence = exact_snapshot(snapshot, [shard])
        attempts = list(shard.rglob('HTTP_ATTEMPT.json'))
        responses = list((shard / 'solver').glob('*/responses/*.json'))
        need(not attempts and not responses and snapshot['generation_http_calls'] == snapshot['solver_calls'] == 0,
             'failed_preparation_shard_must_have_zero_calls')
        need(not (shard / 'TRAINED_EVALUATION.STARTED.json').exists() and not (shard / 'BASE_EVALUATION.STARTED.json').exists(), 'failed_shard_solver_started')
        error = read(shard / 'BATCH_PREPARATION.ERROR.json')
        need(error['status'] == 'ERROR' and error['automatic_retry'] is False and error['training_allowed'] is False, 'failed_shard_error_record')
        output.append(evidence | {'home': str(shard), 'snapshot': bind(path), 'error': bind(shard / 'BATCH_PREPARATION.ERROR.json'),
                                  'generation_http_calls': 0, 'solver_calls': 0})
    return {'status': 'PASS', 'failed_shards': output, 'all_installed_shards_classified': True}


def source_screening(home, successful):
    inventory_path = home / 'inventory/CANDIDATE_ROOTS_20.json'; inventory = read(inventory_path)
    candidates = inventory['candidates']; need(len(candidates) == inventory['candidate_count'] == 20, 'twenty_source_inventory')
    need(len({r['root_id'] for r in candidates}) == len(candidates), 'duplicate_source_inventory_root')
    adapter_path = home / 'evidence/source_gate_disclosure_v89.py'
    adaptation = read(home / 'evidence/SOURCE_GATE_DISCLOSURE_ADAPTATION.json')
    check_bind(adaptation['upstream']); check_bind(adaptation['adapted'])
    need(adaptation['adapted'] == bind(adapter_path) and adaptation['source_decision_rules_changed'] is False
         and adaptation['upstream_modified'] is False, 'source_adapter_scope')
    gate = module(adapter_path, 'v89_screening_adapter_readonly')
    admitted = {}; lineage = []
    for shard in successful:
        config = read(shard / 'CONFIG.json')
        for slot in config['ordering_slots']:
            root = slot['root_id']; need(root not in admitted, 'production_root_reused_across_shards')
            admitted[root] = {'shard': str(shard), 'source': {'path': slot['source_path'], 'sha256': slot['source_sha256']}}
    rows = []; expected_reviews = set()
    for candidate in candidates:
        root = candidate['root_id']; path = home / 'source_reviews' / ('root' + root.rsplit('-', 1)[1] + '_REVIEW.json')
        expected_reviews.add(path.resolve()); review = read(path)
        check_bind(candidate['source']); check_bind(review['source'])
        need(review['source']['sha256'] == candidate['source']['sha256'], 'review_source_content_changed:' + root)
        checked = gate.validate_review({'root_id': root, 'source': review['source'], 'review_path': str(path)})
        if root in admitted:
            need(checked['decision'] == 'READY', 'production_source_hold:' + root)
            check_bind(admitted[root]['source'])
            need(admitted[root]['source']['sha256'] == candidate['source']['sha256'], 'production_source_content_changed:' + root)
            classification = 'PRODUCTION_SHARD_SOURCE_READY'
        elif checked['decision'] == 'HOLD':
            classification = 'SOURCE_HOLD'
        elif candidate['candidate_status'] != 'AUDITED_SIX_FACT_PARENT':
            classification = 'SOURCE_READY_MISSING_AUDITED_PARENT'
        else:
            classification = 'SOURCE_READY_PARENT_ADMISSION_PENDING'
        rows.append({'root_id': root, 'inventory_parent_status': candidate['candidate_status'],
                     'current_source_review_status': checked['decision'], 'production_admission': classification,
                     'inventory_source': candidate['source'], 'review_source': checked['source'], 'review': bind(path),
                     'production': admitted.get(root), 'blocking_findings': [f['finding_id'] for f in checked['findings'] if f['severity'] == 'BLOCKING'],
                     'source_ready_is_answer_certification': False})
        paths = {candidate['source']['path'], checked['source']['path']}
        if root in admitted:
            paths.add(admitted[root]['source']['path'])
        if len(paths) > 1:
            lineage.append({'root_id': root, 'paths': sorted(paths), 'sha256': candidate['source']['sha256'],
                            'classification': 'SAME_BYTES_DISTINCT_LINEAGE_PATHS', 'source_content_changed': False})
    need({p.resolve() for p in (home / 'source_reviews').glob('*.json')} == expected_reviews, 'unregistered_review_file')
    need(set(admitted) <= {r['root_id'] for r in candidates}, 'production_root_outside_inventory')
    return {'status': 'PASS', 'inventory': bind(inventory_path), 'review_adapter': bind(adapter_path), 'source_count': len(rows),
            'source_decision_counts': dict(Counter(r['current_source_review_status'] for r in rows)),
            'production_admission_counts': dict(Counter(r['production_admission'] for r in rows)),
            'production_root_count': len(admitted), 'roots': rows, 'source_lineage_aliases': lineage,
            'full_source_reviews_revalidated': True, 'source_hold_is_distinct_from_missing_parent': True,
            'source_semantics_recertified_by_validator': False, 'new_model_calls': 0}


def elapsed(start, end):
    value = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
    need(value >= 0, 'negative_execution_duration')
    return value


def execution_summary(successful):
    shards = []
    for shard in successful:
        requests = []
        for attempt in sorted(shard.rglob('HTTP_ATTEMPT.json')):
            folder = attempt.parent; result = read(folder / 'HTTP_RESULT.json'); timing = read(folder / 'THROTTLE.json')
            response_path = folder / 'RESPONSE.json'; error_path = folder / 'ERROR.json'
            response = read(response_path) if response_path.exists() else None
            metadata = response.get('metadata', {}) if response else {}
            usage = metadata.get('usage'); usage = usage if isinstance(usage, dict) else None
            requests.append({'attempt': bind(attempt), 'http_result': bind(folder / 'HTTP_RESULT.json'),
                             'outcome': result.get('outcome'), 'http_status': result.get('http_status'),
                             'response_present': response is not None, 'null_or_empty_response': response is None or not response.get('text'),
                             'response': bind(response_path) if response else None, 'error': bind(error_path) if error_path.exists() else None,
                             'usage': usage, 'usage_available': usage is not None,
                             'wall_seconds': elapsed(timing['http_started_at_utc'], result['http_completed_at_utc']),
                             'throttle_wait_seconds': timing['waited_seconds'],
                             'mechanical_validation': bind(folder / 'VALIDATION.json') if (folder / 'VALIDATION.json').exists() else None})
        report = read(shard / 'MIXED_EVALUATION_REPORT.json')
        layers = {}
        for kind in ('multi_select', 'ordering'):
            row = report['by_type'][kind]
            layers[kind] = {key: row[key] for key in ('mechanically_assembled', 'content_usable', 'content_excluded', 'solver_selected')}
            if 'candidate_capacity' in row:
                layers[kind]['candidate_capacity'] = row['candidate_capacity']
                layers[kind]['mechanically_unfilled_capacity'] = row['candidate_capacity'] - row['mechanically_assembled']
            layers[kind]['difficulty_not_selected'] = row['content_usable'] - row['solver_selected']
            layers[kind]['models'] = row['models']
        token_totals = {key: sum(r['usage'].get(key, 0) or 0 for r in requests if r['usage'])
                        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
        shards.append({'home': str(shard), 'report': bind(shard / 'MIXED_EVALUATION_REPORT.json'),
                       'generation_http_calls': len(requests), 'http_outcomes': dict(Counter(str(r['outcome']) for r in requests)),
                       'null_or_empty_response_count': sum(r['null_or_empty_response'] for r in requests),
                       'usage_reported_request_count': sum(r['usage_available'] for r in requests), 'reported_token_totals': token_totals,
                       'token_totals_complete': all(r['usage_available'] for r in requests),
                       'summed_http_wall_seconds': sum(r['wall_seconds'] for r in requests), 'requests': requests,
                       'typed_candidate_layers': layers, 'solver_calls_per_model': report['calls_per_model'],
                       'solver_wall_seconds': {side: elapsed(read(shard / (side + '_EVALUATION.STARTED.json'))['time'],
                                                           read(shard / (side + '_EVALUATION.COMPLETE.json'))['time'])
                                               for side in ('TRAINED', 'BASE')}, 'estimated_currency_cost': None,
                       'cost_limitation': 'No verified pricing was supplied. Missing provider usage is unknown, never zero.'})
    return {'status': 'PASS', 'shards': shards, 'training_performed': False, 'new_audit_model_calls': 0}


def verify_complete(home):
    path = home / 'PILOT_COMPLETE.json'; complete = read(path)
    need(complete['status'] == 'COMPLETE' and complete['home'] == str(home)
         and complete['seal_excludes_only'] == ['PILOT_COMPLETE.json'], 'complete_schema')
    expected = complete['files']; actual = files_under(home) - {str(path)}
    need(set(expected) == actual and complete['file_count'] == len(actual), 'complete_exact_fileset_changed')
    for filename, digest in expected.items():
        need(sha(filename) == digest, 'sealed_file_changed:' + filename)
    historical_preservation(home)
    need(complete['training_allowed'] is False and complete['training_performed'] is False, 'complete_training_policy')
    successful = [Path(r['home']) for r in complete['successful_shards']]
    failed_shards(home, [Path(p) for p in complete['failed_shards']], successful)
    source_screening(home, successful)
    replay = module(HERE / 'audit_export_v89.py', 'v89_sealed_export_replay')
    baseline = home.parent / 'h4_anchor_records_mixed_batch_v88/CUMULATIVE_AUDITED_NEW_CANDIDATES.json'
    findings = replay.audit_exports(home, baseline, complete['successful_shards'])
    need(findings['new_count'] == complete['new_count'] and findings['cumulative_count'] == complete['cumulative_count'], 'sealed_export_counts')
    return {'status': 'PASS', 'sealed_file_count': len(actual), 'exact_fileset_and_all_hashes_verified': True,
            'historical_preservation_verified': True, 'source_reviews_and_raw_export_replay_verified': True, 'read_only': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, default=DEFAULT_HOME)
    parser.add_argument('--shard', type=Path, action='append', default=[])
    parser.add_argument('--failed-shard', type=Path, action='append', default=[])
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--action', choices=('export', 'closeout'), default='export')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--verify-complete', action='store_true')
    args = parser.parse_args(); home = args.home.resolve()
    names = EXPORT_NAMES if args.action == 'export' else CLOSEOUT_NAMES
    allowed = set() if args.check_only or args.verify_complete else {str(home / name) for name in names}
    def guard(event, values):
        if event.startswith('socket.') or event in {'subprocess.Popen', 'os.system', 'os.posix_spawn'}:
            raise RuntimeError('closeout_forbids_external:' + event)
        if event in {'os.remove', 'os.rename', 'os.rmdir', 'os.mkdir', 'os.link', 'os.symlink', 'os.chmod', 'os.truncate', 'os.utime'}:
            raise RuntimeError('closeout_forbids_filesystem_mutation:' + event)
        if event == 'open' and isinstance(values[0], (str, bytes, os.PathLike)):
            path = str(Path(os.fsdecode(values[0])).resolve()); mode, flags = values[1:3]
            writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            if writing:
                need(path in allowed and not (home / 'PILOT_COMPLETE.json').exists(), 'write_forbidden:' + path)
    sys.addaudithook(guard)
    if args.verify_complete:
        print(json.dumps(verify_complete(home), ensure_ascii=False)); return
    need(not (home / 'PILOT_COMPLETE.json').exists(), 'pilot_already_sealed_use_verify_complete')
    successful = [p.resolve() for p in args.shard]; failed = [p.resolve() for p in args.failed_shard]
    entries = selected_shards(home, successful)
    baseline = (args.baseline or home.parent / 'h4_anchor_records_mixed_batch_v88/CUMULATIVE_AUDITED_NEW_CANDIDATES.json').resolve()
    replay = module(HERE / 'audit_export_v89.py', 'v89_independent_export_replay')
    if args.action == 'export':
        new_pool, cumulative = build_exports(home, baseline, entries)
        audit = replay.audit_exports(home, baseline, entries, new_pool, cumulative)
        if args.check_only:
            print(json.dumps({'status': 'PASS', 'action': 'export_check', 'new_count': audit['new_count'],
                              'cumulative_count': audit['cumulative_count'], 'read_only': True})); return
        need(not any((home / name).exists() for name in EXPORT_NAMES), 'export_output_already_exists_use_check_only')
        write(home / EXPORT_NAMES[0], new_pool); write(home / EXPORT_NAMES[1], cumulative)
        audit = replay.audit_exports(home, baseline, entries)
        audit.update(created_at_utc=now(), files=[bind(home / name) for name in EXPORT_NAMES[:2]],
                     implementation=[bind(Path(__file__)), bind(HERE / 'audit_export_v89.py')])
        write(home / EXPORT_NAMES[2], audit)
        print(json.dumps({'status': 'PASS', 'action': 'export', 'new_count': audit['new_count'], 'cumulative_count': audit['cumulative_count']})); return
    for name in EXPORT_NAMES:
        need((home / name).is_file(), 'audited_export_required_before_closeout:' + name)
    recorded = read(home / EXPORT_NAMES[2]); need(recorded['status'] == 'PASS', 'export_audit_not_pass')
    for value in recorded['files'] + recorded['implementation']:
        check_bind(value)
    audit = replay.audit_exports(home, baseline, entries)
    protection = historical_preservation(home); failed_record = failed_shards(home, failed, successful)
    screening = source_screening(home, successful); execution = execution_summary(successful)
    need(not any((home / name).exists() for name in CLOSEOUT_NAMES), 'closeout_outputs_already_exist')
    if args.check_only:
        print(json.dumps({'status': 'PASS', 'action': 'closeout_check', 'new_count': audit['new_count'],
                          'cumulative_count': audit['cumulative_count'], 'source_counts': screening['source_decision_counts'],
                          'protected_file_count': protection['file_count'], 'read_only': True}, ensure_ascii=False)); return
    for name, value in zip(CLOSEOUT_NAMES[:4], (protection, failed_record, screening, execution)):
        write(home / name, value | {'created_at_utc': now()})
    new_pool = read(home / EXPORT_NAMES[0]); cumulative = read(home / EXPORT_NAMES[1])
    report = ['# v89 mixed-question pilot', '',
              f"Audited new candidates: {new_pool['candidate_count']}; by type: {new_pool['by_type']}; multi-select Gold: {new_pool['gold_distribution']}.",
              f"Cumulative v88 + v89 inventory after whole-six-statement deduplication: {cumulative['candidate_count']}; by type: {cumulative['by_type']}.",
              f"Twenty full-source reviews: {screening['source_decision_counts']}; production-admission categories: {screening['production_admission_counts']}.",
              f"Actual successful production shards: {len(successful)}; unique production roots: {screening['production_root_count']}. Source READY does not certify question answers.", '',
              '| Shard | HTTP attempts | Outcomes | Null/empty response | Reported tokens | Solver calls/model |',
              '| --- | ---: | --- | ---: | --- | ---: |']
    for item in execution['shards']:
        report.append(f"| {Path(item['home']).name} | {item['generation_http_calls']} | {item['http_outcomes']} | {item['null_or_empty_response_count']} | {item['reported_token_totals']} ({item['usage_reported_request_count']} requests reported usage) | {item['solver_calls_per_model']} |")
        report += ['', f"{Path(item['home']).name} typed stages (historical multi Gold6 controls included in Solver/content counts, excluded from new exports): {json.dumps(item['typed_candidate_layers'], ensure_ascii=False)}.",
                   f"HTTP duration sum: {item['summed_http_wall_seconds']} seconds; Solver durations: {item['solver_wall_seconds']} seconds."]
    report += ['', 'Provider failures, null responses, mechanical assembly, content rejection, and difficulty selection are recorded separately in evidence/PILOT_EXECUTION_SUMMARY.json. Raw successful and failed requests remain in their shard directories.',
               'Token totals include only provider-reported usage; missing usage remains unknown. Currency cost is unavailable without verified pricing.',
               f"Failed preparation shards preserved: {[p.name for p in failed]}; every file matches its snapshot and generation HTTP/Solver counts are zero.",
               f"Historical preservation: {protection['file_count']} files across {len(protection['roots'])} directories; exact inventories and every SHA-256 pass.",
               f"Same-byte distinct-source paths: {json.dumps(screening['source_lineage_aliases'], ensure_ascii=False)}.",
               'No training, baseline mutation, automatic retries, or formal training admission. Raw Solver replay verifies score/selection correctness; semantic reliability remains limited by the saved source and content reviews. Related source/parent families are not independent test examples.', '',
               'PILOT_COMPLETE.json binds every pilot file except itself. No file may be written after sealing. Verify with --verify-complete, with output outside this directory.', '']
    write(home / 'REPORT.md', '\n'.join(report))
    paths = sorted(files_under(home) | {str(home / 'ARTIFACT_INDEX.md'), str(home / 'PILOT_COMPLETE.json')})
    index = ['# v89 artifact index', '', 'Every pilot artifact is listed below. PILOT_COMPLETE.json supplies exact byte hashes for every listed file except itself.', '']
    index += [f"- [{Path(path).relative_to(home).as_posix()}]({Path(path).relative_to(home).as_posix()})" for path in paths]
    write(home / 'ARTIFACT_INDEX.md', '\n'.join(index) + '\n')
    historical_preservation(home); failed_shards(home, failed, successful)
    sealed_files = {path: sha(path) for path in sorted(files_under(home))}
    complete = {'version': 'v89_pilot_exact_inventory_seal_v1', 'status': 'COMPLETE', 'home': str(home),
                'completed_at_utc': now(), 'files': sealed_files, 'file_count': len(sealed_files),
                'seal_excludes_only': ['PILOT_COMPLETE.json'], 'successful_shards': entries,
                'failed_shards': [str(p) for p in failed], 'training_allowed': False, 'training_performed': False,
                'new_count': audit['new_count'], 'cumulative_count': audit['cumulative_count'],
                'historical_preservation': bind(home / CLOSEOUT_NAMES[0]), 'no_further_pilot_writes': True}
    write(home / 'PILOT_COMPLETE.json', complete)
    # This is read-only; the complete manifest is the final filesystem write.
    print(json.dumps(verify_complete(home), ensure_ascii=False))


if __name__ == '__main__':
    main()
