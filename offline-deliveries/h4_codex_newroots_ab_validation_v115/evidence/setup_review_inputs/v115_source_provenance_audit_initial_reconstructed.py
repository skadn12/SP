"""Independent read-only provenance audit; writes only an exclusive report."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib
import json
import sys

BATCH = Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_codex_newroots_ab_validation_v115')
START = datetime.now(timezone.utc).isoformat()
ERRORS = []
BINDINGS = {}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    path = Path(path)
    value = json.loads(path.read_text())
    BINDINGS[str(path.resolve())] = sha(path)
    return value

def check(value, code):
    if not value:
        ERRORS.append(code)

def stamp(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('naive_timestamp')
    return result

def bound_table(path, expected_count):
    manifest = read(path)
    rows = manifest['files']
    check(len(rows) == expected_count, str(path) + ':count')
    mismatches = []
    for row in rows:
        try:
            if sha(row['path']) != row['sha256']:
                mismatches.append(row['path'])
            if 'snapshot_path' in row:
                if sha(row['snapshot_path']) != row['snapshot_sha256'] or row['snapshot_sha256'] != row['sha256']:
                    mismatches.append(row['snapshot_path'])
        except OSError:
            mismatches.append(row['path'])
    check(not mismatches, str(path) + ':hash_mismatch')
    return {'manifest_path': str(path), 'manifest_sha256': sha(path), 'expected': expected_count,
            'checked': len(rows), 'mismatches': mismatches}

cfg = read(BATCH / 'RUN_CONFIG.json')
roots = cfg['roots']
scenes = cfg['scene_sequence']
freeze = read(BATCH / 'EXECUTION_INPUTS_FROZEN.json')
check(len(roots) == freeze['root_denominator'] == 4, 'four_root_denominator')
check(len(roots) * len(scenes) == freeze['scene_denominator'] == 24, 'twentyfour_scene_denominator')
check(scenes == [f'R02:{route}:S{i:02d}' for route in 'AB' for i in range(1, 4)], 'scene_sequence')
manifests = {
    'frozen_inputs': bound_table(BATCH / 'EXECUTION_INPUTS_FROZEN.json', 32),
    'tooling': bound_table(BATCH / 'evidence/TOOLING_FREEZE.json', 11),
    'protected_history': bound_table(BATCH / 'evidence/HISTORICAL_PROTECTION.json', 7249),
}
reservations = read(BATCH / 'SOURCE_RESERVATIONS.json')
allocation = reservations['allocation']
check(len(allocation) == 12, 'twelve_reservations')
check(len({row['source_id'] for row in allocation}) == len(allocation), 'reserved_source_ids_unique')
check(not ({row['source_id'] for row in allocation} & set(reservations['consumed_source_ids_before_this_batch'])), 'reserved_sources_previously_consumed')
check(reservations['permanent_including_failed_and_unexecuted_slots'] is True, 'reservation_permanence')
check(sha(reservations['source_pool_path']) == reservations['source_pool_sha256'], 'native_source_pool_hash')
reservation_map = {(r['root_id'], r['scene_ref']): r for r in allocation}
expected_slots = {(r['root_id'], s) for r in roots for s in scenes if ':B:' in s}
check(set(reservation_map) == expected_slots, 'reservation_slots')
root_rows = []
scene_rows = []
event_rows = []
repair_rows = []
all_receipts = []

for root in roots:
    rid = root['root_id']
    short = rid.rsplit('-', 1)[-1]
    root_dir = BATCH / 'roots' / ('root' + short)
    baseline_path = Path(root['baseline_path'])
    baseline = read(baseline_path)
    check(sha(baseline_path) == root['baseline_sha256'] == sha(root['upstream_path']), rid + ':baseline_and_upstream_hash')
    check(len(baseline['turns']) == root['baseline_turn_count'], rid + ':baseline_turn_count')
    selection_path = BATCH / 'selection' / ('root' + short + '.json')
    selection = read(selection_path)
    check(selection['baseline_sha256'] == sha(baseline_path), rid + ':selection_baseline_hash')
    check(selection['full_baseline_read'] is True, rid + ':selection_full_read')
    check(selection['baseline_turn_refs_read'] == [t['turn_ref'] for t in baseline['turns']], rid + ':selection_coverage')
    pool_path = BATCH / 'pre_inputs' / ('root' + short + '_EVENT_POOL.json')
    partition = read(pool_path)
    chosen_events = {e['dialogue_id']: e for e in partition['events']}
    selected_map = {a['scene_ref']: a for a in selection['allocations']}
    for scene in scenes:
        if ':B:' not in scene:
            continue
        ar = reservation_map[(rid, scene)]
        selected = selected_map[scene]
        ep = Path(ar['source_item_path'])
        event = read(ep)
        check(ar['selection_sha256'] == sha(selection_path), rid + scene + ':reservation_selection_hash')
        check(ar['source_item_sha256'] == sha(ep), rid + scene + ':native_event_hash')
        for key in ('root_id', 'scene_ref', 'source_id', 'role_mapping', 'event_core_to_preserve', 'adaptation_limitation', 'baseline_continuity_constraints', 'native_turn_ids_read'):
            check(selected[key] == ar[key], rid + scene + ':reservation_semantic_field:' + key)
        check(event == chosen_events.get(event['dialogue_id']), rid + scene + ':native_event_is_full_partition_record')
        if 'native_event' in selected:
            check(selected['native_event'] == event, rid + scene + ':embedded_native_event')
        check(ar['source_id'] == 'multiwoz22:train:' + event['dialogue_id'], rid + scene + ':native_source_id')
        check(ar['native_turn_ids_read'] == [t['turn_id'] for t in event['turns']], rid + scene + ':native_turn_ids')
        check(stamp(selection['created_at_utc']) <= stamp(reservations['created_at_utc']), rid + scene + ':selection_before_reservation')
        event_rows.append({'root_id': rid, 'scene_ref': scene, 'source_id': ar['source_id'], 'native_event_sha256': sha(ep), 'native_turn_count': len(event['turns']), 'selection_embeds_native_event': 'native_event' in selected})

    previous_path = baseline_path
    previous = baseline
    predecessor_ready_at = None
    root_scene_rows = []
    for scene in scenes:
        row_errors_before = len(ERRORS)
        folder = root_dir / 'scenes' / scene.removeprefix('R02:').replace(':', '_')
        result = {'root_id': rid, 'scene_ref': scene}
        try:
            draft_path = folder / 'EXPANSION_DRAFT.json'
            source_path = folder / 'CUMULATIVE_SOURCE.json'
            review_path = folder / 'SOURCE_REVIEW.json'
            draft = read(draft_path)
            source = read(source_path)
            author = read(folder / 'AUTHOR_RECEIPT.json')
            review = read(review_path)
            payload_path = folder / 'AUTHOR_PAYLOAD.json'
            payload = read(payload_path)
            scene_receipts = [read(p) for p in sorted((folder / 'receipts').glob('*.json'))]
            all_receipts.extend(scene_receipts)
            structural = [r for r in scene_receipts if r['status'] == 'STRUCTURAL_PASS']
            accepted = [r for r in scene_receipts if r['status'] in ('REVIEW_READY', 'REVIEW_HOLD')]
            check(len(structural) == 1, rid + scene + ':one_structural_receipt')
            check(len(accepted) == 1, rid + scene + ':one_semantic_receipt')
            materialized = structural[0]
            ready = accepted[0]
            check(draft['input_source_sha256'] == source['input_source_sha256'] == sha(previous_path), rid + scene + ':input_hash')
            check(source['turns'] == previous['turns'] + draft['turns'], rid + scene + ':exact_prefix_plus_draft')
            check(source['baseline_sha256'] == sha(baseline_path), rid + scene + ':baseline_binding')
            check(source['draft_sha256'] == sha(draft_path), rid + scene + ':draft_binding')
            check(12 <= len(draft['turns']) <= 16, rid + scene + ':new_turn_count')
            check(source['new_turn_count'] == len(draft['turns']), rid + scene + ':recorded_new_turn_count')
            check(source['previous_turn_count'] == len(previous['turns']), rid + scene + ':recorded_previous_turn_count')
            check(source['cumulative_new_turn_count'] == len(source['turns']) - len(baseline['turns']), rid + scene + ':recorded_cumulative_turn_count')
            check(len(set(t['turn_ref'] for t in source['turns'])) == len(source['turns']), rid + scene + ':unique_turn_refs')
            check(all(t['speaker'] in root['baseline_speakers'] for t in draft['turns']), rid + scene + ':established_speakers')
            check(payload['messages'] == [{'speaker': t['speaker'], 'text': t['text']} for t in draft['turns']], rid + scene + ':payload_serialization')
            for key, path in [('draft_sha256', draft_path), ('source_sha256', source_path), ('input_source_sha256', previous_path)]:
                check(materialized[key] == sha(path), rid + scene + ':structural_' + key)
            check(materialized['run_config_sha256'] == sha(BATCH/'RUN_CONFIG.json'), rid + scene + ':structural_config')
            check(materialized['source_reservations_sha256'] == sha(BATCH/'SOURCE_RESERVATIONS.json'), rid + scene + ':structural_reservation')
            check(author['author'] == draft['author'] != review['reviewer'], rid + scene + ':different_author_reviewer')
            check(author['full_source_read'] is True and review['full_source_read'] is True, rid + scene + ':full_source_attestations')
            check(author['turn_refs_read'] == [t['turn_ref'] for t in previous['turns']], rid + scene + ':author_complete_prefix_coverage')
            check(review['turn_refs_read'] == [t['turn_ref'] for t in source['turns']], rid + scene + ':review_complete_cumulative_coverage')
            check(author['attempt'] == 1 and author.get('semantic_repairs', 0) == 0 and author.get('semantic_repair', False) is False and author.get('retries', 0) == 0, rid + scene + ':one_author_attempt_no_semantic_retry')
            check(author['draft_sha256'] == sha(draft_path), rid + scene + ':author_draft_binding')
            check(author.get('payload_sha256', author.get('author_payload_sha256')) == sha(payload_path), rid + scene + ':author_payload_binding')
            check(author['input_source_sha256'] == sha(previous_path), rid + scene + ':author_input_binding')
            input_files = author.get('read_files', author.get('input_files_read', []))
            for bound in input_files + author.get('rule_files_machine_validated', []):
                check(sha(bound['path']) == bound['sha256'], rid + scene + ':author_read_hash:' + bound['path'])
            check(sha(author['serializer_path']) == author['serializer_sha256'], rid + scene + ':serializer_binding')
            check(review['source_sha256'] == sha(source_path), rid + scene + ':review_source_binding')
            check(review['selection_sha256'] == sha(selection_path), rid + scene + ':review_selection_binding')
            check(ready['review_sha256'] == sha(review_path) and ready['source_sha256'] == sha(source_path), rid + scene + ':accepted_review_binding')
            check(ready['status'] == 'REVIEW_' + review['decision'], rid + scene + ':accepted_decision')
            ast, aend = stamp(author['started_at_utc']), stamp(author['finished_at_utc'])
            rst, rend = stamp(review['started_at_utc']), stamp(review['finished_at_utc'])
            mat, accepted_at = stamp(materialized['created_at_utc']), stamp(ready['created_at_utc'])
            check(ast <= aend, rid + scene + ':author_timestamp_order')
            check(ast >= stamp(freeze['created_at_utc']), rid + scene + ':author_after_inputs_frozen')
            if predecessor_ready_at:
                check(ast >= predecessor_ready_at, rid + scene + ':author_start_after_predecessor_READY')
            check(mat <= stamp(author.get('artifact_saved_at_utc', author['finished_at_utc'])), rid + scene + ':materialization_before_author_receipt_saved')
            check(mat <= rst <= rend <= accepted_at, rid + scene + ':materialization_review_acceptance_order')
            check(stamp(review['created_at_utc']) == rend, rid + scene + ':review_created_finish')
            if 'full_source_read_finished_at_utc' in author:
                check(ast <= stamp(author['full_source_read_finished_at_utc']) <= aend, rid + scene + ':full_read_finished_timing')
            check(set(review['checks']) >= {'time','location','actor_identity','plan_vs_completion','cross_scene_continuity','semantic_repetition','naturalness','event_fidelity'}, rid + scene + ':eight_review_checks')
            check(all(v.get('reason', '').strip() for v in review['checks'].values()), rid + scene + ':substantive_check_reasons')
            check(review['semantic_review_is_programmatic'] is False, rid + scene + ':nonprogrammatic_review_attestation')
            if ':B:' in scene:
                ar = reservation_map[(rid, scene)]
                ep = Path(ar['source_item_path']); event = read(ep)
                check(source['source_id'] == draft['source_id'] == review['source_id'] == ar['source_id'], rid + scene + ':B_source_id_binding')
                check(source['source_item_sha256'] == review['native_event_sha256'] == sha(ep), rid + scene + ':B_event_hash_binding')
                check(review['native_event_full_read'] is True and review['native_turn_ids_read'] == [t['turn_id'] for t in event['turns']], rid + scene + ':review_native_coverage')
                event_read = [x for x in input_files if Path(x['path']) == ep]
                check(len(event_read) == 1 and (event_read[0].get('full_native_utterances_read') is True or event_read[0].get('full_native_event_read') is True), rid + scene + ':author_native_read_attestation')
            else:
                check(not draft.get('source_id') and not source.get('source_id'), rid + scene + ':A_no_external_event')
            if review.get('schema_repair'):
                repair = read(folder/'SOURCE_REVIEW_SCHEMA_REPAIR_RECEIPT.json')
                invalid = read(folder/'SOURCE_REVIEW_INVALID_FIRST_SAVE.json')
                repaired_fields = {k:v for k,v in review.items() if k != 'schema_repair'}
                repaired_fields['findings'] = [v['reason'] for v in review['findings']]
                check(repaired_fields == invalid, rid + scene + ':schema_only_repair_equivalence')
                check(repair['invalid_first_save_sha256'] == sha(folder/'SOURCE_REVIEW_INVALID_FIRST_SAVE.json') and repair['repaired_review_sha256'] == sha(review_path), rid + scene + ':schema_repair_hashes')
                repair_rows.append(repair)
            result.update(decision=review['decision'], receipt_status=ready['status'], baseline_turns=len(baseline['turns']),
                          new_turns=len(draft['turns']), cumulative_turns=len(source['turns']), source_sha256=sha(source_path),
                          review_sha256=sha(review_path), author=author['author'], reviewer=review['reviewer'],
                          author_started_at_utc=author['started_at_utc'], author_finished_at_utc=author['finished_at_utc'],
                          materialized_at_utc=materialized['created_at_utc'], review_started_at_utc=review['started_at_utc'],
                          review_finished_at_utc=review['finished_at_utc'], accepted_at_utc=ready['created_at_utc'],
                          author_seconds=(aend-ast).total_seconds(), review_seconds=(rend-rst).total_seconds(),
                          inherited_findings=review['findings'])
            predecessor_ready_at = accepted_at if review['decision'] == 'READY' else None
            previous_path, previous = source_path, source
        except Exception as exc:
            ERRORS.append(rid + scene + ':' + type(exc).__name__ + ':' + str(exc))
            result['exception'] = type(exc).__name__ + ':' + str(exc)
        result['issues'] = ERRORS[row_errors_before:]
        result['mechanical_status'] = 'PASS' if not result['issues'] else 'FAIL'
        scene_rows.append(result)
        root_scene_rows.append(result)
    root_receipts = [read(p) for p in sorted((root_dir/'receipts').glob('*.json'))]
    all_receipts.extend(root_receipts)
    root_rows.append({'root_id': rid, 'baseline_turns': len(baseline['turns']), 'final_turns': len(previous['turns']),
                      'new_turns': len(previous['turns'])-len(baseline['turns']),
                      'ready_scenes': sum(x.get('decision') == 'READY' and not x['issues'] for x in root_scene_rows),
                      'all_six_ready': len(root_scene_rows)==6 and all(x.get('decision')=='READY' and not x['issues'] for x in root_scene_rows)})

report = {
    'audit_kind':'independent_mechanical_source_provenance_audit', 'reviewer':'/root/v113_setup',
    'started_at_utc':START,'finished_at_utc':datetime.now(timezone.utc).isoformat(),
    'status':'PASS' if not ERRORS else 'FAIL', 'scope':'Read-only artifact integrity and recorded procedure/timestamp checks; not a new semantic certification or proof of unobservable reading cognition.',
    'root_denominator':len(roots),'scene_denominator':len(roots)*len(scenes), 'event_denominator':len(expected_slots),
    'full_ready_root_count':sum(x['all_six_ready'] for x in root_rows),
    'ready_scene_count':sum(x.get('decision')=='READY' and not x['issues'] for x in scene_rows),
    'total_new_turns':sum(x['new_turns'] for x in root_rows),
    'manifests':manifests,'roots':root_rows,'scenes':scene_rows,'native_event_bindings':event_rows,
    'receipt_status_counts':dict(Counter(x['status'] for x in all_receipts)),
    'schema_record_failures':len(repair_rows),'schema_repairs':repair_rows,
    'terminal_failure_receipts':[x for x in all_receipts if x['status'] in {'STRUCTURAL_FAILURE','INTEGRITY_FAILURE','REVIEW_INVALID','REVIEW_HOLD'}],
    'semantic_retry_count':sum(x.get('semantic_retry_count',0) for x in repair_rows),
    'inherited_quality_limitations':'Read the scene findings. In particular 31457 retains a serious deceased-James/her versus living-James conflict and unresolved job chronology; procedural READY does not clean its baseline.',
    'timing_limits':'Recorded author intervals include cumulative rereading/composition/scheduling. Review intervals include cumulative reading and judgment; timestamps in JSON are UTC offsets. Not billed service latency. Token usage and cost are not measured.',
    'input_bindings':[{'path':p,'sha256':v} for p,v in sorted(BINDINGS.items())], 'issues':ERRORS,
    'training_started':False, 'pool_merge_performed':False,
}
output = Path(sys.argv[1]) if len(sys.argv)>1 else Path('/tmp/v115_SOURCE_PROVENANCE_AUDIT.json')
with output.open('x') as handle:
    json.dump(report, handle, ensure_ascii=False, indent=2);handle.write('\n')
print(json.dumps({k:report[k] for k in ('status','root_denominator','scene_denominator','full_ready_root_count','ready_scene_count','total_new_turns','receipt_status_counts','schema_record_failures','semantic_retry_count','issues')},ensure_ascii=False))
print(str(output),sha(output))
