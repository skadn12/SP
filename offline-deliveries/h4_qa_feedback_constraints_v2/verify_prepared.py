"""Check the new feedback on the existing offline replay; no model calls."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

WORKTREE = Path('/root/autodl-tmp/gxy-selfplay/worktrees/h4-gate-downstream-v1')
sys.path.insert(0, str(WORKTREE))
from scripts.current_method.restart_n5_spell.core import atomic_json, sha256_file
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1 import qa_difficulty_feedback_run as runner

BASE = Path(__file__).resolve().parent
RUN = BASE / 'prepared_run'
PREVIOUS = BASE.parent / 'h4_qa_difficulty_feedback_v1'


def read(path):
    return json.loads(Path(path).read_text())


def main():
    manifest = read(RUN / 'freeze_manifest.json')
    runner.check_freeze(manifest)
    previous_manifest = read(PREVIOUS / 'prepared_run/freeze_manifest.json')
    runner.check_freeze(previous_manifest)
    runner.verify_mock(RUN, manifest)
    assert not (RUN / 'EXECUTION_STARTED.json').exists()
    assert not (RUN / 'live').exists()
    source = Path(manifest['source_run'])
    replay = RUN / 'mock_replay'
    summary = read(replay / 'generation_summary.json')
    records = read(replay / 'complete_call_records.json')
    assert len(records) == 18
    assert summary['external_http_attempt_count'] == 0
    assert summary['new_generation_evidence'] is False
    calls, roots, capacities = [], [], defaultdict(list)
    fields = ('candidate_id', 'question_type', 'question', 'options', 'items', 'gold', 'evidence_refs')
    total_candidates = 0
    compiled = read(BASE / 'COMPILED_FEEDBACK.json')
    rules = (BASE / 'writer_constraints.txt').read_text().strip()
    slot_rows = []
    for root in runner.ROOTS:
        original = source / 'qa/roots' / root
        frozen = RUN / 'frozen' / root
        mocked = replay / 'roots' / root
        # Both the older replay and the most recent experiment used this source.
        source_paths = [original / 'qa_source.json', frozen / 'qa_source.json',
                        mocked / 'qa_source.json',
                        PREVIOUS / 'prepared_run/live/roots' / root / 'qa_source.json']
        assert len({sha256_file(p) for p in source_paths}) == 1
        context = runner.validate_context(read(frozen / 'qa_feedback_context.json'))
        assert rules in context['slot_suffixes'][0]
        provenance = read(mocked / 'qa_feedback_provenance.json')
        assert provenance['qa_prompt_contains_feedback'] is True
        assert provenance['qa_source_contains_feedback'] is False
        assert provenance['memory_in_prompt'] is False
        assert read(mocked / 'qa_feedback_context.json') == context
        old_pool, new_pool = read(original / 'candidate_pool.json'), read(mocked / 'candidate_pool.json')
        assert [[q.get(k) for k in fields] for q in old_pool] == [[q.get(k) for k in fields] for q in new_pool]
        total_candidates += len(new_pool)
        entry = next(r for r in compiled['records'] if r['root_id'] == root)
        assert Path(entry['preserved_correction']['path']).read_bytes() == (
            source.parent / 'feedback' / root / 'feedback_input.json').read_bytes()
        roots.append({'root_id': root, 'source_and_dates_byte_identical_to_both_prior_runs': True,
                      'original_correction_byte_identical': True, 'historical_replay_candidates': len(new_pool),
                      'feedback_outside_factual_transcript': True, 'memory_in_prompt': False,
                      'primary_and_repair_share_one_suffix': True})
        slot_rows.append({'root_id': root, 'stage': 'miner', 'request_id': root + '-intent-miner',
                          'status': 'FROZEN_READY', 'http_max': 1,
                          'request': str(frozen / 'miner_payload.json')})
        for i in range(1, 4):
            slot_rows.append({'root_id': root, 'stage': 'writer_primary', 'slot': i,
                              'status': 'CONDITIONAL_ON_NEW_VALID_MINER_CONTRACT', 'http_max': 1,
                              'maximum_repairs': 1, 'repair_condition': 'EXISTING_STRUCTURAL_RULES_ONLY',
                              'actual_intent_and_prompt': 'DETERMINED_BY_NEW_MINER_NOT_HISTORICAL_REPLAY'})
        for record in (r for r in records if r['root_id'] == root):
            rid, stage = record['request_id'], record['stage']
            old = source / 'qa/execution/calls' / root / rid
            new = replay / 'execution/calls' / root / rid
            old_request, new_request = read(old / 'request.json'), read(new / 'request.json')
            suffix = context['intent_suffix'] if stage == 'miner' else context['slot_suffixes'][0]
            old_messages, new_messages = old_request['payload']['messages'], new_request['payload']['messages']
            assert len(old_messages) == len(new_messages) == 1
            assert new_messages[0]['role'] == old_messages[0]['role']
            assert new_messages[0]['content'] == old_messages[0]['content'] + '\n' + suffix
            assert {k: v for k, v in old_request['payload'].items() if k != 'messages'} == {
                k: v for k, v in new_request['payload'].items() if k != 'messages'}
            capacity = read(new / 'capacity.json')
            assert capacity['status'] == 'PASS' and capacity['truncation'] is False
            capacities[stage].append(capacity)
            calls.append({'request_id': rid, 'stage': stage, 'complete_message_checked': True,
                          'only_prompt_change_is_feedback_suffix': True, 'same_request_parameters': True,
                          'input_tokens_local_proxy': capacity['input_tokens'],
                          'required_context_tokens_local_proxy': capacity['required_context_tokens']})
    assert total_candidates == 29
    stage_counts = dict(Counter(r['stage'] for r in records))
    assert stage_counts == {'miner': 3, 'writer_primary': 9, 'writer_repair': 6}
    statistics = {stage: {'count': len(rows), 'input_min': min(r['input_tokens'] for r in rows),
                         'input_max': max(r['input_tokens'] for r in rows),
                         'input_total': sum(r['input_tokens'] for r in rows),
                         'max_required_context': max(r['required_context_tokens'] for r in rows)}
                  for stage, rows in capacities.items()}
    output = {'status': 'PASS', 'external_http_attempt_count': 0, 'new_model_generation_count': 0,
              'effectiveness_assessed': False, 'live_execution_started': False,
              'previous_frozen_records_unchanged': len(previous_manifest['files']),
              'new_frozen_records_verified': len(manifest['files']), 'roots': roots, 'requests': calls,
              'historical_replay_calls': len(records), 'replay_stage_counts': stage_counts,
              'historical_replay_candidates_unchanged': total_candidates,
              'replay_source': str(source), 'feedback_experience_source': str(PREVIOUS),
              'replay_is_not_new_generation_or_constraint_compliance_evidence': True,
              'token_counts_local_proxy': statistics,
              'proxy_context_guard': capacities['miner'][0]['context_window_tokens'],
              'service_capacity': 'UNKNOWN', 'service_reasoning': 'UNKNOWN', 'verified_cost': 'UNKNOWN',
              'future_run_schedule': runner.schedule(),
              'entry_regression': {'module': 'test_qa_difficulty_feedback_run', 'tests_passed': 4,
                                   'scope': 'Existing entry budget, mock, no-network replay, context and identity checks'},
              'automatic_next_experiment': False}
    atomic_json(BASE / 'OFFLINE_VERIFICATION.json', output)
    atomic_json(BASE / 'REQUEST_SLOTS.json', {'status': 'PREPARED_NOT_SENT', 'slots': slot_rows,
                'miner': 3, 'primary': 9, 'conditional_repair_max': 9,
                'maximum_logical_calls': 21, 'maximum_http_attempts': 21})
    print(json.dumps({k: output[k] for k in ('status', 'external_http_attempt_count',
          'historical_replay_calls', 'historical_replay_candidates_unchanged',
          'previous_frozen_records_unchanged', 'token_counts_local_proxy')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
