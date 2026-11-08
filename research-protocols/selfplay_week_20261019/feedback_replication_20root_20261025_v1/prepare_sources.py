#!/usr/bin/env python3
"""Freeze Train parent sources and permanently reserve native events; no model calls."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[2]
PRIOR = BASE.parent / 'feedback_closed_loop_20261024_v1'
TRAIN = REPO / 'offline-deliveries/h4_codex_train_pool_v125'
SPLITS = REPO / 'restart-n5-runs/h4_round0_b0_k3_difficulty_v1_live_1/root_split_manifest.jsonl'
SEED = 'feedback_closed_loop_replication_20261025_v1'
OLD = ['218519', '222937', '225575', '229686']
EXPECTED = ['75241', '55788', '761', '77882', '45849', '194901', '68549', '25652',
            '179186', '109164', '14304', '24719', '238359', '136202', '246750', '156136']
SCENES = [f'R03:{route}:S{i:02}' for route in 'AB' for i in range(1, 4)]


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def bind(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def write_bytes(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == raw, ('IMMUTABLE_CONTENT_CONFLICT', str(path))
        return
    with path.open('xb') as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def write(path, value):
    write_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))


def import_inventory():
    path = BASE.parent / 'overnight_endtoend_20261024_v1/scripts/prepare_production.py'
    spec = importlib.util.spec_from_file_location('fgr20_inventory_helper', path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    # The old helper skips its own run. Rebind to this run so old OET1 reservations
    # are counted as historical use rather than accidentally omitted.
    helper.BASE = BASE
    return helper, bind(path)


def neutral_material_tasks():
    """Use only the original neutral source-author tasks; never consume F author cards."""
    rows, templates, seen = [], [], {}
    for old_root in OLD:
        path = PRIOR / f'inputs/cc-train-episode-{old_root}_SOURCE_AUTHOR_INPUT.json'
        packet = read(path)
        templates.append(bind(path))
        for task in packet['material_tasks']:
            payload = {k: copy.deepcopy(v) for k, v in task.items() if k != 'task_id'}
            key = json.dumps(payload, sort_keys=True, ensure_ascii=False)
            if key in seen:
                rows[seen[key]]['template_origins'].append({'path': str(path), 'task_id': task['task_id']})
                continue
            seen[key] = len(rows)
            rows.append({'task_id': f'FGR20-NEUTRAL-{len(rows)+1:02}-prospective-material',
                         **payload, 'template_origins': [{'path': str(path), 'task_id': task['task_id']}]})
    assert len(rows) == 8, len(rows)
    return rows, templates


def root_inventory():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/Qwen3-4B', local_files_only=True)
    manifest_path = TRAIN / 'data/ROOT_MANIFEST.json'
    manifest = read(manifest_path)
    authority = {row['root_id']: row for row in map(json.loads, SPLITS.read_text().splitlines())}
    assert len(manifest) == 68
    candidates = []
    for row in manifest:
        rid = row['root_id']
        path = TRAIN / row['packaged_source']
        source_binding = bind(path)
        assert source_binding['sha256'] == row['source_binding']['sha256']
        assert source_binding['bytes'] == row['source_binding']['size_bytes']
        assert Path(row['source_binding']['path']).read_bytes() == path.read_bytes()
        source = read(path)
        assert source['root_id'] == rid and source['scene_ref'] == 'R02:B:S03'
        assert row['split'] == authority[rid]['split'] == 'train'
        assert row['historical_parent_seen_by_M_clean'] is True
        turns = source['turns']
        assert len(turns) == source['baseline_turn_count'] + source['cumulative_new_turn_count']
        assert len({t['turn_ref'] for t in turns}) == len(turns)
        assert all(t['speaker'].strip() and t['text'].strip() for t in turns)
        rendered = '\n'.join(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" for t in turns)
        tokens = len(tokenizer(rendered, add_special_tokens=False)['input_ids'])
        previous = rid.rsplit('-', 1)[1] in OLD
        eligible = not previous and tokens <= 8192
        candidates.append({'root_id': rid, 'numeric_root_id': int(rid.rsplit('-', 1)[1]),
            'split': 'train', 'source_binding': source_binding,
            'original_source_binding': bind(row['source_binding']['path']),
            'source_final_scene_ref': source['scene_ref'], 'turn_count': len(turns),
            'source_status': 'GENERATED_EXTENSION_ON_SOURCE_GROUNDED_PARENT',
            'source_status_note': 'v125 has no explicit source_status key; this labels its generated R02 extension, not an original natural dialogue.',
            'speaker_names': list(dict.fromkeys(t['speaker'] for t in turns)),
            'historical_parent_seen_by_M_clean': True, 'unseen_root_claim_permitted': False,
            'authority_record': authority[rid], 'packaged_root_manifest_record': row,
            'full_source_tokens': tokens, 'full_source_tokens_lte8192': tokens <= 8192,
            'is_original_four': previous, 'eligible': eligible,
            'exclusion_reason': 'REUSE_OLD_FROZEN_RESULT' if previous else ('INITIAL_SOURCE_LENGTH_GT8192' if tokens > 8192 else None),
            'selection_hash': hashlib.sha256((SEED + '|' + rid).encode()).hexdigest()})
    ranked = sorted([r for r in candidates if not r['is_original_four']], key=lambda r: r['selection_hash'])
    selected = [r for r in ranked if r['eligible']][:16]
    assert [r['root_id'].rsplit('-', 1)[1] for r in selected] == EXPECTED
    for rank, row in enumerate(ranked, 1):
        row['nonoriginal_hash_rank'] = rank
        row['selected'] = row in selected
    return candidates, ranked, selected, bind(manifest_path), bind(SPLITS)


def main():
    receipt_path = BASE / 'SOURCE_PREPARATION_RECEIPT.json'
    if receipt_path.exists():
        receipt = read(receipt_path)
        for binding in receipt['artifacts']:
            assert bind(binding['path']) == binding
        print(json.dumps({'status': 'VERIFIED_ALREADY_PREPARED', 'receipt': bind(receipt_path)}))
        return
    started_path = BASE / 'inputs/SOURCE_PREPARATION_STARTED.json'
    if not started_path.exists():
        write(started_path, {'schema_version': 'fgr20_source_preparation_start_v1', 'created_at_utc': now(),
                            'purpose': 'Train-only source preparation, no generation/evaluation/training'})
    stamp = read(started_path)['created_at_utc']
    candidates, ranked, selected, manifest_binding, split_binding = root_inventory()
    tasks, templates = neutral_material_tasks()
    helper, helper_binding = import_inventory()
    records = helper.complete_records()
    reservations_path = BASE / 'inputs/SOURCE_RESERVATIONS.json'
    helper.LOCK.parent.mkdir(parents=True, exist_ok=True)
    with helper.LOCK.open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        acquired = now()
        if reservations_path.exists():
            reservation = read(reservations_path)
            assert len(reservation['reservations']) == 48
            for row in reservation['reservations']:
                assert bind(row['binding']['path']) == row['binding']
        else:
            blocked, evidence, coverage = helper.scan_history(set(records))
            trace, eligible = [], []
            for native_id in sorted(records):
                record = records[native_id][0]
                values, clock_evidence = helper.clocks(record)
                placeholder = bool(re.search(r'\[(?:value_)?time\]', ' '.join(t['utterance'] for t in record['turns']), re.I))
                okay = native_id not in blocked and len(values) >= 2 and not placeholder
                trace.append({'dialogue_id': native_id, 'prior_use_hit': native_id in blocked,
                    'distinct_explicit_HHMM_count': len(values), 'time_placeholder_found': placeholder, 'eligible': okay})
                if okay:
                    eligible.append(native_id)
            assert len(eligible) >= 48
            chosen = eligible[:48]
            reservations, groups = [], []
            for index, root in enumerate(selected):
                native_ids = chosen[index*3:index*3+3]
                clock_union = set()
                for slot, native_id in enumerate(native_ids, 1):
                    record, raw, begin, end = records[native_id]
                    values, clocks = helper.clocks(record)
                    clock_union.update(values)
                    path = BASE / 'inputs/native_events' / native_id
                    write_bytes(path, raw)
                    binding = bind(path)
                    reservations.append({'dataset': 'MultiWOZ_2.2', 'split': 'train',
                        'source_id': 'multiwoz22:train:' + native_id, 'item_id': native_id,
                        'pool_record_id': native_id, 'root_id': root['root_id'],
                        'scene_ref': f'R03:B:S{slot:02}', 'reserved_before_authoring': True,
                        'status': 'PERMANENTLY_RESERVED_BEFORE_AUTHORING', 'release_on_failure': False,
                        'release_on_nonexecution': False, 'binding': binding, 'pool_binding': bind(helper.POOL),
                        'pool_record_byte_span': {'start_inclusive': begin, 'end_exclusive': end},
                        'native_turn_count': len(record['turns']), 'distinct_explicit_clocks': values,
                        'clock_evidence': clocks, 'authorized_use': 'One shared R03 source for this root, consumed by paired F/G question arms only.'})
                assert len(clock_union) >= 4, ('CLOCK_UNION_LT4', root['root_id'], sorted(clock_union))
                groups.append({'root_id': root['root_id'], 'source_ids': ['multiwoz22:train:' + n for n in native_ids],
                               'distinct_explicit_clock_union': sorted(clock_union)})
            reservation = {'schema_version': 'h4.r02.source_reservations.v4', 'run': str(BASE),
                'created_at_utc': stamp, 'status': 'PERMANENT',
                'source_ids': [r['source_id'] for r in reservations], 'reservations': reservations,
                'policy': 'Reserve before authoring; no release on failure, rejection, cancellation, or nonexecution.',
                'selection_rule': 'Sort native dialogue_id; exclude every historical reservation/accepted use found by original mechanism; require >=2 literal HH:MM readings and no time placeholder; first 48; three per fixed root; >=4 distinct readings per group.'}
            write(reservations_path, reservation)
            write(BASE / 'inputs/RESERVATION_AUDIT.json', {
                'schema_version': 'fgr20_permanent_native_reservation_audit_v1', 'status': 'PASS',
                'created_at_utc': stamp, 'reservation_binding': bind(reservations_path),
                'global_lock': {'path': str(helper.LOCK), 'mechanism': 'fcntl.LOCK_EX', 'acquired_at_utc': acquired,
                    'inventory_and_reservation_under_same_lock': True},
                'pool_binding': bind(helper.POOL), 'pool_provenance_binding': bind(helper.PROVENANCE),
                'historical_scan_helper_binding': helper_binding,
                'helper_BASE_rebound_to_new_run': True, 'scan_coverage': coverage,
                'historically_blocked_count': len(blocked), 'remaining_before_reservation': len(records)-len(blocked),
                'eligible_native_count_before_reservation': len(eligible), 'logical_reservations': 48,
                'selected_prior_unused_within_scan_scope': True, 'root_groups': groups,
                'selection_trace': trace, 'inventory_evidence': evidence,
                'release_on_failure': False, 'no_solver_scores_used': True, 'old_files_modified': False})
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    packet_bindings = []
    task_path = BASE / 'inputs/NEUTRAL_MATERIAL_TASKS.json'
    write(task_path, {'schema_version': 'fgr20_neutral_source_material_tasks_v1', 'tasks': tasks,
        'template_bindings': templates, 'selection': 'Exact deduplicated union of original four neutral source-author material task objects, excluding only old task_id; no F weak-card content or Solver output consumed.',
        'same_for_all_new_roots': True, 'purpose': 'TEST_HYPOTHESIS'})
    template = read(PRIOR / 'inputs/cc-train-episode-218519_SOURCE_AUTHOR_INPUT.json')
    for root in selected:
        rid = root['root_id']
        parent_path = BASE / 'inputs/initial_sources' / f'{rid}.json'
        write_bytes(parent_path, Path(root['source_binding']['path']).read_bytes())
        root['frozen_initial_source_binding'] = bind(parent_path)
        native_events = [r for r in reservation['reservations'] if r['root_id'] == rid]
        assert len(native_events) == 3
        packet = copy.deepcopy(template)
        packet.update(root_id=rid, source_status=root['source_status'], parent_source_binding=bind(parent_path),
            source_round='R03', scene_sequence=SCENES, material_tasks=[{k:v for k,v in t.items() if k != 'template_origins'} for t in tasks],
            native_events=native_events)
        packet['output_contract']['shape']['root_id'] = rid
        packet['output_contract']['shape']['source_status'] = root['source_status']
        packet['output_contract']['shape']['scenes'][0]['turns'][0]['speaker'] = ' or '.join(root['speaker_names']) + ' as established'
        packet['replication_provenance'] = {'neutral_material_tasks_binding': bind(task_path),
            'source_packet_template_binding': templates[0], 'same_neutral_objectives_for_all_new_roots': True,
            'feedback_target_expected_error_solver_output_not_supplied': True,
            'new_root_diagnosis_not_performed': True, 'material_transfer_status': 'EXPLORATORY_TRANSFER_OF_ORIGINAL_NEUTRAL_SOURCE_OBJECTIVES'}
        path = BASE / 'inputs' / f'{rid}_SOURCE_AUTHOR_INPUT.json'
        write(path, packet)
        root['material_packet_binding'] = bind(path)
        packet_bindings.append(bind(path))
    parent_selection_path = BASE / 'PARENT_SOURCE_SELECTION.json'
    write(parent_selection_path, {'schema_version': 'fgr20_parent_source_selection_v1', 'split': 'train',
        'roots': selected, 'new_roots': EXPECTED, 'source_generation': 'v125 R02:B:S03',
        'final_R03_source_selection_is_not_this_file': True})
    original_manifest_path = PRIOR / 'FINAL_MANIFEST.json'
    original_manifest = read(original_manifest_path)
    original_bindings = {name: bind(PRIOR / name) for name in ['FINAL_MANIFEST.json', 'FINAL_MANIFEST.sha256',
        'CONFIG.json', 'SOURCE_SELECTION_FINAL.json', 'execution/SCORES.json', 'execution/PRIVATE_SCORING.json']}
    old_rows = read(PRIOR / 'SOURCE_SELECTION_FINAL.json')['roots']
    for row in old_rows:
        assert bind(row['path']) == {k: row[k] for k in ['path', 'sha256', 'bytes']}
    freeze_path = BASE / 'SELECTION_FREEZE.json'
    write(freeze_path, {'schema_version': 'fgr20_selection_freeze_v1', 'created_at_utc': stamp,
        'status': 'FROZEN_BEFORE_AUTHORING', 'selection_seed': SEED,
        'root_selection_rule': '68 v125 Train manifest roots; old four reused; reject only whole initial source >8192 tokens; sort remaining by SHA256(seed + vertical-bar + root_id); select first 16. No Solver correctness or generated quality used.',
        'source_length_definition': 'All source turns rendered [turn_ref] speaker: text, newline joined; Qwen3-4B tokenizer add_special_tokens=False; complete source, no truncation.',
        'source_manifest_binding': manifest_binding, 'train_split_authority_binding': split_binding,
        'all_68_candidates': candidates, 'all_64_nonoriginal_hash_ranked': ranked,
        'eligible_nonoriginal_count': sum(r['eligible'] for r in ranked),
        'original_roots': OLD, 'new_roots': EXPECTED, 'all_roots': OLD + EXPECTED,
        'new_root_count': 16, 'total_root_count': 20, 'roots': selected,
        'parent_source_selection_binding': bind(parent_selection_path),
        'original_four_reused_read_only': {'run': str(PRIOR), 'bindings': original_bindings,
            'sources': old_rows, 'manifest_sha256': bind(original_manifest_path)['sha256'],
            'semantic_questions': 64, 'solver_requests': 128, 'rerun': False},
        'new_planned_semantic_questions': 256, 'new_planned_solver_requests': 512,
        'combined_planned_semantic_questions': 320, 'combined_planned_solver_requests': 640,
        'native_event_reservation_binding': bind(reservations_path),
        'neutral_material_task_binding': bind(task_path),
        'no_solver_scores_used_for_selection': True, 'no_generation_or_model_calls': True,
        'no_unseen_root_claim': True, 'fixed_roots_no_replacement_after_failure': True,
        'user_authorization': '2026-10-25 新增16个，加在一起20个，挂上过夜实验。'})
    artifacts = [bind(Path(__file__)), bind(started_path), bind(parent_selection_path), bind(freeze_path),
        bind(task_path), bind(reservations_path), bind(BASE / 'inputs/RESERVATION_AUDIT.json')] + packet_bindings
    artifacts += [r['frozen_initial_source_binding'] for r in selected]
    artifacts += [r['binding'] for r in reservation['reservations']]
    receipt = {'schema_version': 'fgr20_source_preparation_receipt_v1', 'created_at_utc': now(),
        'status': 'PREPARED_AND_FROZEN_NOT_GENERATED', 'counts': {'new_roots':16, 'old_roots_reused':4,
            'total_roots':20, 'parent_sources_copied':16, 'neutral_tasks_shared':8, 'native_events_reserved':48,
            'source_author_packets':16, 'new_scenes_planned':96, 'new_turns_planned':768,
            'new_question_slots_planned':256, 'new_solver_requests_planned':512},
        'artifacts': artifacts, 'source_generation': 'v125 R02 parent -> one R03 A1-A3 then B1-B3 shared-source expansion',
        'source_budget': {'turns_per_scene':8, 'new_turns_per_root':48, 'average_english_words_per_turn_max':18,
            'complete_solver_prompt_tokens_max':10240},
        'old_files_modified':False, 'model_calls':0, 'training':False,
        'final_R03_sources_still_require_authoring_and_review':True,
        'limitations': ['Historical parent seen by M-clean; these are not unseen roots.',
            'Source objectives are transferred from original neutral templates, not newly diagnosed per-root weaknesses.',
            'Final full prompt must be retokenized after source and question authoring; failures retain missing slots, never replace roots.',
            'Historical usage certification is bounded by recorded repository ledgers; external/unlogged consumption cannot be certified.']}
    write(receipt_path, receipt)
    for binding in artifacts:
        assert bind(binding['path']) == binding
    print(json.dumps({'status':receipt['status'], 'receipt':bind(receipt_path), 'counts':receipt['counts']}))


if __name__ == '__main__':
    main()
