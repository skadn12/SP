"""Prepare and verify the additional four-root run with all networking blocked."""
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1 import full_pipeline_planner_recovery_more as entry
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1 import full_pipeline_planner_recovery_more_inputs as inputs
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.development_validate import check_freeze
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.incremental_round import load_incremental_round_input

BASE = Path('/root/autodl-tmp/gxy-selfplay')
DELIVERY = BASE / 'offline-deliveries/h4_planner_recovery_more_roots_v1'
WORKTREE = BASE / 'worktrees/h4-planner-recovery-more-roots-v1'
PREVIOUS = BASE / 'worktrees/h4-planner-candidate-recovery-v1'
PACKAGE = Path('scripts/current_method/restart_n5_spell/simple_append_memorax_v1_memory_event_continuation_probe_v1')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    with patch('socket.socket.connect', side_effect=AssertionError('NETWORK_FORBIDDEN')) as network, \
         patch('socket.create_connection', side_effect=AssertionError('NETWORK_FORBIDDEN')) as connect, \
         patch.object(entry.shared, 'resolve_credential', side_effect=AssertionError('NO_CREDENTIAL_READ')) as credentials, \
         patch.object(entry.shared, 'ProfileClient', side_effect=AssertionError('NO_TRANSPORT')) as transport, \
         patch.object(entry.shared, 'QARuntime', side_effect=AssertionError('NO_QA_RUNTIME')) as qa_runtime:
        bundle = inputs.prepare_inputs()
        manifest = entry.prepare()
        for blocked in [network, connect, credentials, transport, qa_runtime]:
            blocked.assert_not_called()
    assert entry.HTTP_BUDGET == manifest['schedule']['maximum_http_attempts'] == 68
    check_freeze(manifest)
    check_freeze(bundle)
    previous_manifest = BASE / 'restart-n5-runs/h4_full_pipeline_planner_recovery_v1_live_1/freeze_manifest.json'
    check_freeze(read(previous_manifest))
    rows = []
    for row in bundle['roots']:
        root = row['root_id']
        source_input = Path(row['source_input_path'])
        _, original, _ = load_incremental_round_input(source_input, round_index=2)
        local = entry.DEFAULT_RUN / 'frozen' / root
        data = read(local / 'inputs.json')
        request = read(local / 'planner_request.json')
        plan = read(entry.DEFAULT_INPUTS / 'date_plans' / (root + '.json'))
        capacity = read(local / 'planner_capacity.json')
        assert data['prior_expanded'] == original
        assert data['snapshot'] == read(Path(row['source_memory_path']))
        assert data['feedback_record'] == read(Path(row['source_feedback_path']))['record']
        assert sha(local / 'inputs.json') == sha(Path(row['input_path']))
        prompt = '\n'.join(m['content'] for m in request['messages'])
        assert all(turn['text'] in prompt and turn['turn_ref'] in prompt for turn in original['turns'])
        for segment in data['conversation_date_segments']:
            header = f"[Conversation date: {segment['conversation_date']} | Session: {segment['segment_id']}]"
            assert prompt.count(header) == 1
        assert capacity['status'] == 'PASS' and capacity['truncation'] is False
        rows.append({'root_id': root, 'input_path': str(local / 'inputs.json'),
            'input_sha256': sha(local / 'inputs.json'), 'source_input_sha256': sha(source_input),
            'history_turn_count': len(original['turns']), 'segment_count': len(data['conversation_date_segments']),
            'date_range': row['date_range'], 'original_turns_refs_and_order_unchanged': True,
            'memory_unchanged': True, 'feedback_record_unchanged': True,
            'complete_transcript_and_date_headers_in_actual_messages': True,
            'date_validation': row['date_validation'], 'date_residual_notes': plan.get('residual_notes', []),
            'planner_request_path': str(local / 'planner_request.json'),
            'planner_request_sha256': sha(local / 'planner_request.json'), 'capacity': capacity})
    assert sum(row['history_turn_count'] for row in rows) == 380
    assert sum(row['segment_count'] for row in rows) == 32
    assert not (entry.DEFAULT_RUN / 'EXECUTION_STARTED.json').exists()

    unchanged = []
    old_changes = read(BASE / 'offline-deliveries/h4_planner_candidate_recovery_v1/CHANGED_FILES.json')
    for row in old_changes:
        if Path(row['path']).name == 'full_pipeline_newroots.py':
            continue
        assert sha(WORKTREE / row['path']) == row['sha256'], row['path']
        unchanged.append(row['path'])
    changed = []
    patch_parts = []
    for name in ['full_pipeline_newroots.py', 'full_pipeline_planner_recovery_more.py',
                 'full_pipeline_planner_recovery_more_inputs.py', 'test_full_pipeline_bundle_identity.py',
                 'test_full_pipeline_planner_recovery_more.py']:
        relative = PACKAGE / name
        path, old = WORKTREE / relative, PREVIOUS / relative
        current = path.read_text()
        previous = old.read_text() if old.exists() else ''
        patch_parts.extend(difflib.unified_diff(previous.splitlines(True), current.splitlines(True),
                           fromfile=str(relative) if old.exists() else '/dev/null', tofile=str(relative)))
        changed.append({'path': str(relative), 'new_file': not old.exists(), 'sha256': sha(path)})
    (DELIVERY / 'implementation.patch').write_text(''.join(patch_parts))
    write(DELIVERY / 'CHANGED_FILES.json', changed)
    plan = manifest['schedule']
    module = 'scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.full_pipeline_planner_recovery_more'
    command = f'python -m {module} execute --run-root {entry.DEFAULT_RUN} --approved-http-budget 68'
    report = {'status': 'PASS_OFFLINE_PREPARATION_WAITING_FOR_68_HTTP_BUDGET',
        'prepared_at': datetime.now(timezone.utc).isoformat(), 'external_calls': 0,
        'credential_reads': 0, 'transport_constructions': 0, 'network_attempts': 0,
        'workdir': str(WORKTREE), 'run_root': str(entry.DEFAULT_RUN), 'root_order': list(entry.ROOT_ORDER),
        'input_bundle': str(entry.DEFAULT_INPUTS), 'input_identity_checks': rows,
        'freeze_file_count': len(manifest['files']), 'input_bundle_file_count': len(bundle['files']),
        'freeze_manifest_sha256': sha(entry.DEFAULT_RUN / 'freeze_manifest.json'),
        'input_manifest_sha256': sha(entry.DEFAULT_INPUTS / 'manifest.json'),
        'previous_run_freeze_unchanged': True, 'previous_recovery_implementation_unchanged_files': unchanged,
        'total_history_turns': 380, 'total_date_segments': 32,
        'schedule': plan, 'initial_planner_input_token_estimate_sum': sum(r['capacity']['input_tokens'] for r in rows),
        'future_execute_command': command, 'execution_marker_created': False,
        'authorization': 'New 68-HTTP budget confirmation needed under the user-established per-run budget boundary; no previous allowance reused.',
        'quality_status': 'NOT_EVALUATED_NO_NEW_OUTPUTS',
        'service_context_reasoning_money': 'UNKNOWN; local tokenizer capacity is only a proxy guard.'}
    write(DELIVERY / 'PREPARATION_VERIFICATION.json', report)
    write(DELIVERY / 'STATUS.json', {'status': report['status'], 'external_calls': 0,
        'run_root': str(entry.DEFAULT_RUN), 'http_budget_to_confirm': 68})
    print(json.dumps({'status': report['status'], 'roots': list(entry.ROOT_ORDER), 'turns': 380,
        'segments': 32, 'http_budget': 68, 'initial_planner_input_tokens': report['initial_planner_input_token_estimate_sum'],
        'frozen_files': len(manifest['files']), 'external_calls': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
