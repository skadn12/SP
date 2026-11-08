"""Independent read-only final delivery check, excluding its new evidence output."""
from pathlib import Path
from datetime import datetime, timezone
import importlib.util
import json
import re

B = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('frontier_dev_checks', B / 'evidence/independent_frontier_dev_audit.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
w = helper.w
read, sha, binding, stamp = helper.read, helper.sha, helper.binding, helper.stamp
WHO = '/root/frontier_quality'


def main():
    prior = {str(p): sha(p) for p in B.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    manifest = read(B / 'ROOT_MANIFEST.json')
    rows = read(B / 'ROOT_INDEX.json')['roots']
    assert len(rows) == manifest['fixed_roots'] == 7
    assert len({r['root_id'] for r in rows}) == 7
    assert [r['root_id'] for r in rows] == [r['root_id'] for r in manifest['roots']]
    freeze = read(B / 'MASTER_FREEZE.json')
    for row in freeze['files']:
        binding(row)
    for row in read(B / 'evidence/REPORTING_TOOLS_BINDING.json')['files']:
        binding(row)
    closure_path = B / 'DEV_PHASE_CLOSED.json'
    assert sha(closure_path) == '17907a9fc974a8bc8db70ed6054450899853b6041cc1a440503705d66853a1c1'
    closure = read(closure_path)
    assert closure['status'] == 'DEV_FIVE_ROOTS_TERMINAL_AUDITED_BEFORE_TEST_PROCESSING'
    assert closure['dev_complete_roots'] == 5 and closure['dev_ready_scenes'] == 30
    assert len(closure['frozen_dev_files']) == 479
    assert closure['policy_sha256'] == sha(B / 'PROTOCOL_ZH.md')
    for row in closure['frozen_dev_files'] + closure['audits']:
        binding(row, B)
    old_sealed = []
    for row in read(B / 'evidence/SEALED_DELIVERIES_BEFORE.json')['sealed_manifests']:
        p = Path(binding(row))
        payload = read(p)
        for item in payload['files']:
            binding(item, p.parent)
        old_sealed.append({'manifest_path': str(p), 'manifest_sha256': sha(p), 'payload_files_checked': len(payload['files']), 'status': 'PASS'})
    source_freeze = read(B / 'evidence/SOURCE_CONSUMPTION_FREEZE.json')
    binding(source_freeze['historical_pool_freeze'])
    for item in source_freeze['newer_v124_reservation_bindings']:
        binding(item)
    consumed = set(source_freeze['consumed_source_ids'])
    native_ids = []
    checked_roots, warnings = [], []
    full_turns, old_turns, added_turns = 0, 0, 0
    for row, root_binding in zip(rows, manifest['roots']):
        rid, split = row['root_id'], row['split']
        u = w.unit(rid)
        cfg = w.cfg(rid)
        assert cfg['author'] == root_binding['author'] and cfg['reviewer'] == root_binding['reviewer']
        assert cfg['author'] != cfg['reviewer'] and cfg['split'] == split
        assert root_binding['original_status'] == 'BASELINE_HOLD'
        assert cfg['legacy_holds_preserved'] and cfg['baseline_rewrite'] is False
        assert all(cfg[k] is False for k in ['solver_run', 'training_run', 'question_generation', 'score_based_selection'])
        for item in root_binding['input_bindings']:
            assert sha(item['original_path']) == sha(item['copied_path']) == item['sha256']
        constraints = read(u / 'inputs/HISTORICAL_CONTINUATION_CONSTRAINTS.json')
        assert constraints['decision'] == 'HOLD'
        for item in [constraints['baseline_binding']] + constraints['review_bindings'] + constraints['input_bindings']:
            binding(item)
        w.check_execution(rid)
        status = w.status(rid)
        assert status['full_expansion_complete'] and status['ready_scene_count'] == 6 and not status['root_stopped']
        base_path = u / 'inputs/BASELINE.json'
        base = read(base_path)
        plan, pr = read(u / 'SAFE_PLAN.json'), read(u / 'SAFE_PLAN_REVIEW.json')
        assert plan['author'] == cfg['author'] and pr['reviewer'] == cfg['reviewer']
        assert plan['baseline_conflicts_resolved'] is pr['baseline_conflicts_resolved'] is False
        assert pr['decision'] == 'READY' and pr['safe_plan_sha256'] == sha(u / 'SAFE_PLAN.json')
        helper.citations(plan, base)
        helper.citations(pr, base)
        reservations = read(u / 'SOURCE_RESERVATIONS.json')
        assert reservations['permanent_including_failed_and_unexecuted_slots'] is True
        assert [a['scene_ref'] for a in reservations['allocation']] == list(w.SCENES[3:])
        pool = {'multiwoz22:train:' + e['dialogue_id']: e for e in read(u / 'inputs/EVENT_POOL.json')['events']}
        assert len(pool) == 8
        for a in reservations['allocation']:
            native_ids.append(a['source_id'])
            assert a['source_id'] not in consumed
            assert sha(a['source_item_path']) == a['source_item_sha256']
            event = read(a['source_item_path'])
            assert event == pool[a['source_id']]
            assert a['native_turn_ids_read'] == [t['turn_id'] for t in event['turns']]
        previous_path, previous, ready_at = base_path, base['turns'], stamp(pr['created_at_utc'])
        counts, scene_bindings = [], []
        for scene in w.SCENES:
            folder = w.native.scene_folder(u, rid, scene)
            source_path = folder / 'CUMULATIVE_SOURCE.json'
            source, draft, review = read(source_path), read(folder / 'EXPANSION_DRAFT.json'), read(folder / 'SOURCE_REVIEW.json')
            assert source['turns'] == previous + draft['turns']
            assert source['turns'][:len(base['turns'])] == base['turns']
            assert source['input_source_sha256'] == draft['input_source_sha256'] == sha(previous_path)
            assert source['draft_sha256'] == sha(folder / 'EXPANSION_DRAFT.json')
            assert source['baseline_sha256'] == sha(base_path)
            assert 12 <= len(draft['turns']) <= 16
            assert draft['author'] == cfg['author'] and review['reviewer'] == cfg['reviewer']
            assert review['decision'] == 'READY' and review['source_sha256'] == sha(source_path)
            assert review['full_source_read'] is True and review['baseline_conflicts_resolved'] is False
            assert review['full_event_read'] == (':B:' in scene)
            assert review['turn_refs_read'] == [t['turn_ref'] for t in source['turns']]
            assert w.CHECKS <= review['checks'].keys()
            assert all(isinstance(x, dict) and x.get('severity') != 'BLOCKING' for x in review['findings'])
            assert all(x['status'] in {'PASS', 'PASS_WITH_WARNING', 'WARNING'} and x['reason'] and x['citations'] for x in review['checks'].values())
            helper.citations(review, source)
            for n, turn in enumerate(draft['turns'], 1):
                assert turn['turn_ref'] == f'{scene}:T{n:03d}'
                assert turn['speaker'] in {t['speaker'] for t in base['turns']}
                assert turn['scene_ref'] == turn['session_ref'] == scene
                assert turn['text'].strip() and not any('\u3400' <= c <= '\u9fff' for c in turn['text'])
            ar = helper.context_check(u / 'reads' / (scene.replace(':', '_') + '_author') / 'READ_CONTEXT.json', cfg['author'], 'author', scene, previous_path)
            rr = helper.context_check(u / 'reads' / (scene.replace(':', '_') + '_reviewer') / 'READ_CONTEXT.json', cfg['reviewer'], 'reviewer', scene, source_path)
            receipts = w.native.receipts(folder)
            assert len(receipts) == 2 and {x['status'] for x in receipts} == {'STRUCTURAL_PASS', 'REVIEW_READY'}
            structural = next(x for x in receipts if x['status'] == 'STRUCTURAL_PASS')
            accepted = next(x for x in receipts if x['status'] == 'REVIEW_READY')
            assert accepted['source_sha256'] == structural['source_sha256'] == sha(source_path)
            assert accepted['review_sha256'] == sha(folder / 'SOURCE_REVIEW.json')
            assert accepted['reviewer'] == cfg['reviewer']
            assert structural['run_config_sha256'] == sha(u / 'RUN_CONFIG.json')
            assert structural['source_reservations_sha256'] == sha(u / 'SOURCE_RESERVATIONS.json')
            assert ready_at <= stamp(ar['started_at_utc']) <= stamp(structural['created_at_utc'])
            assert stamp(structural['created_at_utc']) <= stamp(rr['started_at_utc']) <= stamp(review['created_at_utc']) <= stamp(accepted['created_at_utc'])
            assert stamp(reservations['created_at_utc']) <= stamp(ar['started_at_utc'])
            notes = {k: v for k, v in review['checks'].items() if v['status'] != 'PASS'}
            if notes or review['findings']:
                warnings.append({'root_id': rid, 'scene_ref': scene, 'checks': notes, 'findings': review['findings'], 'review_path': str((folder / 'SOURCE_REVIEW.json').relative_to(B))})
            counts.append(len(draft['turns']))
            scene_bindings.append({'scene_ref': scene, 'source_sha256': sha(source_path), 'review_sha256': sha(folder / 'SOURCE_REVIEW.json')})
            previous_path, previous, ready_at = source_path, source['turns'], stamp(accepted['created_at_utc'])
        assert len({t['turn_ref'] for t in previous}) == len(previous)
        final_source = w.native.root_folder(u, rid) / 'CUMULATIVE_SOURCE.json'
        assert final_source.read_bytes() == previous_path.read_bytes()
        full, new, text = B / row['dialogue_path'], B / row['new_turns_path'], B / row['readable_path']
        assert full.read_bytes() == final_source.read_bytes()
        new_obj = read(new)
        assert new_obj['root_id'] == rid and new_obj['status'] == row['status'] == 'EXPANSION_COMPLETE'
        assert new_obj['baseline_conflicts_resolved'] is False and new_obj['baseline_sha256'] == sha(base_path)
        assert new_obj['turns'] == previous[len(base['turns']):]
        assert row['baseline_turns'] == len(base['turns']) and row['new_turns'] == sum(counts) and row['total_turns'] == len(previous)
        assert row['ready_scenes'] == 6 and row['all_original_turns_preserved'] and row['baseline_conflicts_resolved'] is False
        readable = text.read_text()
        actual_lines = [line for line in readable.splitlines() if re.match(r'^\[[^\]]+\] \*\*', line)]
        expected_lines = [f"[{t['turn_ref']}] **{t['speaker']}**: {t['text']}" for t in previous]
        assert actual_lines == expected_lines
        assert len(actual_lines) == len(previous)
        assert readable.index('## 原始完整对话（保持原文）') < readable.index('## 新增 R02:A:S01')
        assert re.findall(r'^## 新增 (.+)$', readable, re.MULTILINE) == list(w.SCENES)
        for conflict in plan['quarantined_conflicts']:
            assert conflict['summary'] in readable
        for entry in plan['a_scene_plan']:
            assert entry['scene_ref'] + ': ' + entry['goal'] in readable
        if split == 'test':
            for p in (u / 'reads').rglob('READ_CONTEXT.json'):
                r = read(p)
                assert stamp(r['started_at_utc']) >= stamp(closure['closed_at_utc'])
        checked_roots.append({**{k: row[k] for k in ['root_id', 'split', 'baseline_turns', 'new_turns', 'total_turns']}, 'status': 'PASS', 'scene_turn_counts': counts, 'source_sha256': sha(final_source), 'delivery_dialogue_sha256': sha(full), 'delivery_new_turns_sha256': sha(new), 'delivery_readable_sha256': sha(text), 'scene_bindings': scene_bindings})
        old_turns += len(base['turns'])
        added_turns += sum(counts)
        full_turns += len(previous)
    assert len(native_ids) == len(set(native_ids)) == 21
    assert (old_turns, added_turns, full_turns) == (718, 570, 1288)
    schema = helper.schema_recovery()
    incident = read(B / 'units/root151581/author/SAFE_PLAN_SAVE_CONTEXT_RECEIPT.json')
    assert incident['initial_failure'] == 'AUTHOR_CONTEXT_REQUIRED'
    assert incident['semantic_payload_changed'] is False
    assert Path(incident['initial_full_context_path']).read_bytes() == Path(incident['accepted_full_context_path']).read_bytes()
    test_reports = []
    for short in (78460, 151581):
        p = B / 'evidence' / f'TEST{short}_INDEPENDENT_CHECK.json'
        result = read(p)
        assert result['status'] == 'PASS' and result['semantic_review_performed_by_this_check'] is False
        assert not result.get('failed_checks') and result.get('failure_count', 0) == 0
        rr = next(r for r in checked_roots if r['root_id'] == f'cc-train-episode-{short}')
        sc = result['roots'][0]['scenes'] if 'roots' in result else result['scenes']
        for current, checked in zip(rr['scene_bindings'], sc):
            assert all(current[k] == checked[k] for k in ['scene_ref', 'source_sha256', 'review_sha256'])
        test_reports.append({'path': str(p.relative_to(B)), 'sha256': sha(p), 'status': 'PASS', 'check_count': result['check_count']})
    quality = read(B / 'QUALITY_SUMMARY.json')
    assert quality['warnings'] == warnings and len(warnings) == quality['scenes_with_recorded_nonblocking_quality_notes'] == 8
    assert quality['baseline_turns'] == old_turns and quality['new_turns'] == added_turns and quality['total_turns'] == full_turns
    assert quality['unique_native_event_sources'] == 21 and quality['ready_scenes'] == 42 and quality['complete_roots'] == 7
    assert quality['historical_conflicts_resolved'] is quality['original_v124_success_claimed'] is quality['external_or_strict_blind_review_claimed'] is False
    assert len(quality['execution_incidents']) == 2 and all((B / x['evidence']).exists() for x in quality['execution_incidents'])
    assert all(quality[k] is False for k in ['question_generation', 'solver_run', 'training_run', 'merged_into_current_data_pools'])
    assert all(a['scene_turn_counts'] == b['scene_turn_counts'] for a, b in zip(checked_roots, quality['root_counts']))
    final = read(B / 'FINAL_STATUS.json')
    parent_check = read(B / 'evidence/FINAL_MECHANICAL_AUDIT.json')
    assert final == parent_check and final['status'] == 'PASS' and final['ready_scenes'] == 42 and final['complete_roots'] == 7
    assert all({k: v for k, v in row.items() if not k.endswith('_path')} == status for row, status in zip(rows, final['rows']))
    for name in ['README_ZH.md', 'QUALITY_NOTES_ZH.md']:
        doc = (B / name).read_text()
        assert '矛盾' in doc and ('同模型' in doc or '同一会话模型' in doc) and '盲审' in doc and 'TEST' in doc
        assert '未出题' in doc or '尚未出题' in doc
        for link in re.findall(r'\]\(([^)]+)\)', doc):
            if not link.startswith(('http:', 'https:', '#')):
                assert (B / link).exists()
    for p, old_sha in prior.items():
        assert sha(p) == old_sha, ('AUDIT_MUTATED_EXISTING_FILE', p)
    out = {'schema_version': 'v133_final_delivery_independent_check_v1', 'actual_checker': WHO, 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'PASS', 'scope': 'All seven exported retry roots and immutable acceptance/gate/sealed evidence; mechanical delivery verification only.', 'semantic_review_performed_by_this_check': False, 'external_or_strict_blind_review': False, 'existing_files_unchanged_during_check': len(prior), 'complete_roots': 7, 'ready_scenes': 42, 'baseline_turns': old_turns, 'new_turns': added_turns, 'total_turns': full_turns, 'unique_native_event_sources': 21, 'dev_closure_sha256': sha(closure_path), 'frozen_dev_bindings_checked': 479, 'dev_audit_bindings_checked': len(closure['audits']), 'master_frozen_bindings_checked': len(freeze['files']), 'reporting_tools_unchanged': True, 'delivery_json_exact_accepted_source_bytes': True, 'new_turns_exact_accepted_suffix': True, 'readable_dialogue_turns_exact_source_order_text': True, 'original_holds_and_conflicts_preserved': True, 'scene_gates_source_reservations_and_citations_passed': True, 'quality_summary_counts_and_all_eight_warnings_exactly_match_reviews': True, 'warning_scene_count': len(warnings), 'retained_schema_incident': schema, 'retained_plan_context_failure': {'path': 'units/root151581/author/SAFE_PLAN_SAVE_CONTEXT_RECEIPT.json', 'sha256': sha(B / 'units/root151581/author/SAFE_PLAN_SAVE_CONTEXT_RECEIPT.json'), 'identical_input_bytes': True, 'semantic_payload_unchanged': True}, 'old_sealed': old_sealed, 'old_sealed_payload_files_checked': sum(x['payload_files_checked'] for x in old_sealed), 'test_independent_reports': test_reports, 'roots': checked_roots, 'limitations': ['This verifies exact exported contents, provenance, immutable records and declared full-read coverage; it does not prove semantic truth or a reader mental process.', 'The checker authored no dialogue, was the semantic reviewer for three of the seven roots, and shares model family/filesystem with all agents.', 'New conflict-quarantine results do not repair historical contradictions or convert old v124 HOLD into success; TEST had prior exposure.', 'No QA, solver, training or existing-pool merge was performed.'], 'audit_script_sha256': sha(__file__), 'checked_top_level_artifacts': [{'path': name, 'sha256': sha(B / name)} for name in ['ROOT_INDEX.json', 'FINAL_STATUS.json', 'QUALITY_SUMMARY.json', 'QUALITY_NOTES_ZH.md', 'README_ZH.md', 'DEV_PHASE_CLOSED.json', 'MASTER_FREEZE.json']]}
    p = B / 'evidence/FINAL_DELIVERY_INDEPENDENT_CHECK.json'
    with p.open('x') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'status': out['status'], 'path': str(p), 'sha256': sha(p), 'complete_roots': 7, 'ready_scenes': 42, 'baseline_turns': old_turns, 'new_turns': added_turns, 'total_turns': full_turns, 'old_sealed_payload_files_checked': out['old_sealed_payload_files_checked']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
