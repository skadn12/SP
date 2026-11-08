"""Read-only artifact/procedure audit. Only an exclusive report is written.

Checks recorded evidence, never certifies unobservable reading cognition or new
semantic truth. Known read-context/receipt coverage limitations remain explicit.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
BATCH = Path(__file__).resolve().parent.parent
PROJECT = BATCH.parent.parent
sys.path.insert(0, str(PROJECT / 'scripts'))
from codex_sequential_ab import SCENES, load_config, inspect_root, expected_source

spec = importlib.util.spec_from_file_location('v117_source_review', BATCH / 'source_review.py')
REVIEW = importlib.util.module_from_spec(spec)
spec.loader.exec_module(REVIEW)
START = datetime.now(timezone.utc).isoformat()
ERRORS, WARNINGS, BINDINGS = [], [], {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    path = Path(path).resolve()
    BINDINGS[str(path)] = sha(path)
    return {'path': str(path), 'sha256': BINDINGS[str(path)]}


def read(path):
    bind(path)
    return json.loads(Path(path).read_text())


def check(value, code):
    if not value:
        ERRORS.append(code)


def stamp(value):
    out = datetime.fromisoformat(value.replace('Z', '+00:00'))
    assert out.tzinfo is not None, 'naive_timestamp'
    return out


def verify_table(path, count):
    data = read(path)
    check(len(data['files']) == count, str(path) + ':file_count')
    mismatches = []
    for row in data['files']:
        try:
            if sha(row['path']) != row['sha256']:
                mismatches.append(row['path'])
            if 'snapshot_path' in row:
                if sha(row['snapshot_path']) != row['snapshot_sha256'] or row['snapshot_sha256'] != row['sha256']:
                    mismatches.append(row['snapshot_path'])
        except OSError:
            mismatches.append(row['path'])
    check(not mismatches, str(path) + ':hash_mismatch')
    return {'manifest_binding': bind(path), 'expected': count, 'checked': len(data['files']), 'mismatches': mismatches}


def main():
    cfg = read(BATCH / 'RUN_CONFIG.json')
    load_config(BATCH)
    plan = read(BATCH / 'PRECALL_PLAN.json')
    freeze = read(BATCH / 'EXECUTION_INPUTS_FROZEN.json')
    gate_bindings = REVIEW.verify_batch_gate(cfg)
    for row in gate_bindings:
        bind(row['path'])
    check(len(gate_bindings) == 6, 'baseline_gate_six_bindings')
    check(len(cfg['roots']) == freeze['root_denominator'] == 4, 'four_root_denominator')
    check(cfg['scene_sequence'] == list(SCENES), 'scene_sequence')
    check(len(cfg['roots']) * len(SCENES) == freeze['scene_denominator'] == 24, 'twentyfour_scene_denominator')
    manifests = {
        'execution_inputs': verify_table(BATCH / 'EXECUTION_INPUTS_FROZEN.json', 36),
        'preexecution_code': verify_table(BATCH / 'evidence/PREEXECUTION_CODE_BINDINGS.json', 4),
        'tooling': verify_table(BATCH / 'evidence/TOOLING_FREEZE.json', 6),
        'protected_history': verify_table(BATCH / 'evidence/HISTORICAL_PROTECTION.json', 9624),
    }
    reservations = read(BATCH / 'SOURCE_RESERVATIONS.json')
    allocations = reservations['allocation']
    rmap = {(r['root_id'], r['scene_ref']): r for r in allocations}
    slots = {(r['root_id'], scene) for r in cfg['roots'] for scene in SCENES if ':B:' in scene}
    check(len(allocations) == len(rmap) == 12 and set(rmap) == slots, 'twelve_unique_reservation_slots')
    check(len({r['source_id'] for r in allocations}) == 12, 'native_event_disjointness')
    check(not ({r['source_id'] for r in allocations} & set(reservations['consumed_source_ids_before_this_batch'])), 'native_events_not_previously_consumed')
    check(reservations['permanent_including_failed_and_unexecuted_slots'] is True, 'permanent_reservations')
    check(sha(reservations['source_pool_path']) == reservations['source_pool_sha256'], 'source_pool_binding')
    check(cfg['training_or_pool_merge_authorized'] is False and plan['training_started'] is False, 'no_training_authorization')
    root_rows, scene_rows, event_rows, receipt_rows = [], [], [], []
    for root in cfg['roots']:
        rid = root['root_id']; short = rid.rsplit('-', 1)[-1]
        rd = BATCH / 'roots' / ('root' + short)
        bp = Path(root['baseline_path']); baseline = read(bp)
        cp, constraints = REVIEW.constraints_for(rid); read(cp)
        check(sha(bp) == root['baseline_sha256'] == sha(root['upstream_path']), rid + ':baseline_upstream')
        check(len(baseline['turns']) == root['baseline_turn_count'], rid + ':baseline_count')
        selection_path = BATCH / 'selection' / ('root' + short + '.json')
        selection = read(selection_path)
        check(selection['baseline_sha256'] == sha(bp) and selection['full_baseline_read'] is True, rid + ':selector_baseline_binding')
        check(selection['baseline_turn_refs_read'] == [t['turn_ref'] for t in baseline['turns']], rid + ':selector_baseline_coverage')
        if 'continuation_constraints' in selection:
            check(selection['continuation_constraints'] == constraints['continuation_constraints'], rid + ':selector_all_constraints')
        else:
            WARNINGS.append(rid + ':selection_has_no_literal_all_constraints_copy; frozen_root_constraints_and_scene_reviews_remain_bound')
        cb = selection.get('continuation_constraints_binding', {})
        selector_constraints_sha = selection.get('continuation_constraints_sha256', cb.get('sha256'))
        if selector_constraints_sha is not None:
            check(selector_constraints_sha == sha(cp), rid + ':selector_constraints_binding')
        else:
            WARNINGS.append(rid + ':selection_has_no_per_record_constraints_binding; no_claim_of_uniform_selector_receipt_coverage')
        partition = read(BATCH / 'pre_inputs' / ('root' + short + '_EVENT_POOL.json'))
        pool = {e['dialogue_id']: e for e in partition['events']}
        smap = {r['scene_ref']: r for r in selection['allocations']}
        for scene in SCENES[3:]:
            reserved, selected = rmap[(rid, scene)], smap[scene]
            ep = Path(reserved['source_item_path']); event = read(ep)
            check(reserved['selection_sha256'] == sha(selection_path), rid + scene + ':reserved_selection')
            check(reserved['source_item_sha256'] == sha(ep), rid + scene + ':reserved_native_hash')
            for key in ('root_id', 'scene_ref', 'source_id', 'role_mapping', 'event_core_to_preserve', 'adaptation_limitation', 'baseline_continuity_constraints', 'native_turn_ids_read'):
                if key in reserved or key in selected:
                    check(reserved.get(key) == selected.get(key), rid + scene + ':reservation_field:' + key)
            check(event == pool.get(event['dialogue_id']), rid + scene + ':full_partition_record')
            check(selected.get('native_event') == event, rid + scene + ':full_selected_native_record')
            check(reserved['source_id'] == 'multiwoz22:train:' + event['dialogue_id'], rid + scene + ':native_id')
            check(reserved['native_turn_ids_read'] == [t['turn_id'] for t in event['turns']], rid + scene + ':selector_native_coverage')
            check(stamp(selection.get('finished_at_utc', selection['created_at_utc'])) <= stamp(reservations['created_at_utc']), rid + scene + ':selection_before_reservation')
            event_rows.append({'root_id': rid, 'scene_ref': scene, 'source_id': reserved['source_id'], 'binding': bind(ep), 'native_turn_count': len(event['turns'])})
        previous_path, previous, accepted_at = bp, baseline, None
        local = []
        state = inspect_root(BATCH, cfg, rid, record_reviews=False)
        check(state['full_expansion_complete'] is True and state['ready_scene_count'] == 6, rid + ':controller_six_READY')
        for scene in SCENES:
            before = len(ERRORS); folder = rd / 'scenes' / scene.removeprefix('R02:').replace(':', '_')
            row = {'root_id': rid, 'scene_ref': scene}
            try:
                dp, sp, rp = [folder / n for n in ('EXPANSION_DRAFT.json', 'CUMULATIVE_SOURCE.json', 'SOURCE_REVIEW.json')]
                draft, source, review = read(dp), read(sp), read(rp)
                author = read(folder / 'AUTHOR_RECEIPT.json')
                receipts = [read(p) for p in sorted((folder / 'receipts').glob('*.json'))]
                receipt_rows.extend(receipts)
                structural = [x for x in receipts if x['status'] == 'STRUCTURAL_PASS']
                accepted = [x for x in receipts if x['status'] in {'REVIEW_READY', 'REVIEW_HOLD'}]
                check(len(structural) == len(accepted) == 1, rid + scene + ':one_structural_and_one_review_receipt')
                mat, ready = structural[0], accepted[0]
                check(source == expected_source(BATCH, cfg, rid, scene, previous_path, bp, baseline), rid + scene + ':exact_full_cumulative_metadata_prefix_draft')
                check(source['turns'] == previous['turns'] + draft['turns'], rid + scene + ':exact_prefix')
                check(source['baseline_turn_count'] == len(baseline['turns']), rid + scene + ':baseline_metadata')
                check(author['author'] == draft['author'] != review['reviewer'], rid + scene + ':non_author_review')
                check(author.get('full_source_read', author.get('full_input_source_read')) is True, rid + scene + ':author_full_read')
                check(author['turn_refs_read'] == [t['turn_ref'] for t in previous['turns']], rid + scene + ':author_complete_prefix_refs')
                check(review['full_source_read'] is True and review['turn_refs_read'] == [t['turn_ref'] for t in source['turns']], rid + scene + ':review_complete_source_refs')
                check(author['attempt'] == 1 and all(author.get(k, 0) == 0 for k in ('semantic_repairs', 'semantic_rewrites', 'retries')) and author.get('semantic_repair', False) is False, rid + scene + ':single_attempt_no_rewrite')
                check(author['input_source_sha256'] == draft['input_source_sha256'] == sha(previous_path), rid + scene + ':author_input_hash')
                check(author['draft_sha256'] == sha(dp), rid + scene + ':author_draft_hash')
                pp = folder / 'AUTHOR_PAYLOAD.json'
                if pp.exists():
                    payload = read(pp)
                    check(payload['messages'] == [{'speaker': t['speaker'], 'text': t['text']} for t in draft['turns']], rid + scene + ':payload_serialization')
                    if 'payload_sha256' in author or 'author_payload_sha256' in author:
                        check(author.get('payload_sha256', author.get('author_payload_sha256')) == sha(pp), rid + scene + ':author_payload_hash')
                else:
                    WARNINGS.append(rid + scene + ':no_contemporaneous_scene_AUTHOR_PAYLOAD; do_not_claim_uniform_serializer_receipt_coverage')
                for p in folder.glob('*LATE_ARCHIVE*.json'):
                    read(p)
                    WARNINGS.append(rid + scene + ':late_archive_is_supplement_not_contemporaneous_payload_evidence')
                if 'serializer_path' in author:
                    check(sha(author['serializer_path']) == author['serializer_sha256'], rid + scene + ':serializer_binding')
                else:
                    WARNINGS.append(rid + scene + ':author_receipt_has_no_per_scene_serializer_binding; global_frozen_tooling_only')
                files = author.get('read_files', author.get('input_files_read', []))
                for item in files + author.get('rule_files_machine_validated', []):
                    check(sha(item['path']) == item['sha256'], rid + scene + ':author_read_file_binding:' + item['path'])
                cc = author.get('continuation_constraints', author.get('all_continuation_constraints_read', author.get('continuation_constraints_read')))
                constraint_files = [v for v in files if Path(v['path']).resolve() == Path(cp).resolve()]
                bound_full_constraints = any(v['sha256'] == sha(cp) and (v.get('full_read') is True or v.get('all_constraints_read') is True) for v in constraint_files)
                check(cc is True or cc == constraints['continuation_constraints'] or bound_full_constraints, rid + scene + ':author_all_constraints')
                if cc is None and bound_full_constraints:
                    WARNINGS.append(rid + scene + ':constraints_full_read_bound_in_read_files; no_top_level_all_constraints_field')
                REVIEW.verify_scene_constraints(rid, scene, review)
                check(review['decision'] == 'READY' and review['semantic_review_is_programmatic'] is False, rid + scene + ':recorded_semantic_READY')
                check(set(review['checks']) >= REVIEW.CHECKS and all(v['status'] != 'FAIL' and v.get('reason', '').strip() for v in review['checks'].values()), rid + scene + ':eight_recorded_review_checks')
                check(not any(f.get('severity') == 'BLOCKING' for f in review['findings']), rid + scene + ':no_recorded_blocker')
                check(review['source_sha256'] == sha(sp) == mat['source_sha256'] == ready['source_sha256'], rid + scene + ':source_receipt_hashes')
                check(mat['draft_sha256'] == sha(dp) and mat['input_source_sha256'] == sha(previous_path), rid + scene + ':materialization_input_draft_hashes')
                check(mat['run_config_sha256'] == sha(BATCH / 'RUN_CONFIG.json') and mat['source_reservations_sha256'] == sha(BATCH / 'SOURCE_RESERVATIONS.json'), rid + scene + ':frozen_execution_bindings')
                check(ready['review_sha256'] == sha(rp) and ready['status'] == 'REVIEW_READY', rid + scene + ':accepted_review_binding')
                ast, aend = stamp(author['started_at_utc']), stamp(author['finished_at_utc'])
                rst, rend = stamp(review['started_at_utc']), stamp(review['finished_at_utc'])
                mt, at = stamp(mat['created_at_utc']), stamp(ready['created_at_utc'])
                check(stamp(freeze['created_at_utc']) <= ast <= aend, rid + scene + ':author_after_freeze')
                if accepted_at:
                    check(ast >= accepted_at, rid + scene + ':author_after_predecessor_READY')
                check(mt <= rst <= rend <= at and stamp(review['created_at_utc']) == rend, rid + scene + ':review_timestamp_chain')
                if ':B:' in scene:
                    reserved = rmap[(rid, scene)]; ep = Path(reserved['source_item_path']); event = read(ep)
                    check(draft['source_id'] == source['source_id'] == review['source_id'] == reserved['source_id'], rid + scene + ':native_source_id')
                    check(source['source_item_sha256'] == review['native_event_sha256'] == sha(ep), rid + scene + ':native_event_hash')
                    ids = [t['turn_id'] for t in event['turns']]
                    check(review['native_event_full_read'] is True and review['native_turn_ids_read'] == ids, rid + scene + ':review_native_full_coverage')
                    check(author.get('native_event_full_read') is True or author.get('native_turn_ids_read') == ids, rid + scene + ':author_native_full_read_evidence')
                    check(review['selection_sha256'] == sha(selection_path), rid + scene + ':review_selection_hash')
                else:
                    check(not draft.get('source_id') and not source.get('source_id'), rid + scene + ':A_without_native_source')
                row.update(decision=review['decision'], new_turns=len(draft['turns']), cumulative_turns=len(source['turns']), author=author['author'], reviewer=review['reviewer'], source_binding=bind(sp), review_binding=bind(rp), author_seconds=(aend-ast).total_seconds(), recorded_review_seconds=(rend-rst).total_seconds())
                accepted_at = at
                previous_path, previous = sp, source
            except Exception as exc:
                ERRORS.append(rid + scene + ':' + type(exc).__name__ + ':' + str(exc))
                if (folder / 'CUMULATIVE_SOURCE.json').exists():
                    previous_path = folder / 'CUMULATIVE_SOURCE.json'; previous = read(previous_path)
            row['errors'] = ERRORS[before:]; row['status'] = 'FAIL' if row['errors'] else 'PASS'
            scene_rows.append(row); local.append(row)
        for name in ('CUMULATIVE_SOURCE.json', 'SOURCE_REVIEW.json'):
            final = rd / name
            check(final.exists() and sha(final) == sha(rd / 'scenes/B_S03' / name), rid + ':final_root_copy:' + name)
            if final.exists(): read(final)
        root_rows.append({'root_id': rid, 'baseline_turns': len(baseline['turns']), 'final_turns': len(previous['turns']), 'new_turns': len(previous['turns'])-len(baseline['turns']), 'all_six_ready': len(local) == 6 and all(x['status'] == 'PASS' and x.get('decision') == 'READY' for x in local)})
    provenance_files = ['ROOT68549_A2_READ_PROVENANCE_SUPPLEMENT.json', 'ROOT75223_A1_EXTRA_READ_IDENTITY_CLARIFICATION.json']
    supplements = []
    for name in provenance_files:
        path = BATCH / 'evidence' / name
        supplement = read(path); supplements.append(bind(path))
        if name.startswith('ROOT68549_A2'):
            rr = supplement['supplemental_independent_reread']
            target = BATCH / 'roots/root68549/scenes/A_S02/CUMULATIVE_SOURCE.json'
            check(rr['source_sha256'] == sha(target) and rr['turn_refs_read'] == [t['turn_ref'] for t in read(target)['turns']], 'A2_supplement_full_read_binding')
            check(supplement['immutable_originals_preserved'] is True and rr['reader'] == '/root/v113_question_recheck', 'A2_supplement_reader')
            WARNINGS.append('68549 A2 original read context was author-created; original review start is unsuitable as independent read-start evidence. Actual full135-turn reread interval is additively recorded; immutable review retained.')
        else:
            check(supplement['independent_review_evidence'] is False, 'A1_extra_context_not_independent_review')
            WARNINGS.append('75223 A1 extra temporary read context used another reviewer name; save failed and legitimate parent review remains. Extra context is explicitly unusable as independent evidence.')
    # Complete required delivery coverage, including all receipts/supplements.
    for pattern in ('roots/*/scenes/**/*.json', 'roots/*/CUMULATIVE_SOURCE.json', 'roots/*/SOURCE_REVIEW.json', 'roots/*/ROOT_STOP_RECEIPT.json'):
        for path in BATCH.glob(pattern): bind(path)
    terminal = [x for x in receipt_rows if x['status'] in {'STRUCTURAL_FAILURE','INTEGRITY_FAILURE','REVIEW_INVALID','REVIEW_HOLD'}]
    check(not terminal, 'no_terminal_scene_failure_receipts')
    report = {'audit_kind':'mechanical_source_provenance_audit_v117','auditor':'/root/v113_question_recheck','started_at_utc':START,'finished_at_utc':datetime.now(timezone.utc).isoformat(),'status':'FAIL' if ERRORS else 'PASS','errors':ERRORS,'warnings':WARNINGS,'scope':'Artifact integrity and recorded procedure evidence only; does not prove unobservable reading cognition or certify semantic truth. Same inherited Codex model roles are not external model certification.','root_denominator':4,'scene_denominator':24,'native_event_denominator':12,'roots':root_rows,'scenes':scene_rows,'native_event_bindings':event_rows,'manifests':manifests,'baseline_gate_bindings':gate_bindings,'provenance_supplements':supplements,'ready_scene_count':sum(x.get('decision') == 'READY' and x['status'] == 'PASS' for x in scene_rows),'full_ready_root_count':sum(x['all_six_ready'] for x in root_rows),'total_new_turns':sum(x['new_turns'] for x in root_rows),'receipt_status_counts':dict(Counter(x['status'] for x in receipt_rows)),'terminal_failure_receipts':terminal,'input_bindings':[{'path':p,'sha256':h} for p,h in sorted(BINDINGS.items())],'external_semantic_certification':False,'training_started':False,'pool_merge_performed':False}
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else BATCH / 'SOURCE_PROVENANCE_AUDIT.json'
    with output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2);stream.write('\n')
    print(json.dumps({'status':report['status'],'errors':ERRORS,'warning_count':len(WARNINGS),'ready_scenes':report['ready_scene_count'],'full_roots':report['full_ready_root_count'],'new_turns':report['total_new_turns'],'output':str(output),'sha256':sha(output)},ensure_ascii=False))
    return 1 if ERRORS else 0


if __name__ == '__main__':
    raise SystemExit(main())
