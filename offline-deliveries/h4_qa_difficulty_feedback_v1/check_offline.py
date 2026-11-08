"""Read-only checks of the used source, feedback, and historical response replay."""
import json
from pathlib import Path
import sys

WORKTREE = Path('/root/autodl-tmp/gxy-selfplay/worktrees/h4-gate-downstream-v1')
sys.path.insert(0, str(WORKTREE))
from scripts.current_method.restart_n5_spell.core import atomic_json, sha256_file
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1 import qa_difficulty_feedback_run as runner
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1 import qa_difficulty_feedback as compiler

BASE = Path(__file__).resolve().parent
RUN = BASE / 'prepared_run'


def read(path):
    return json.loads(Path(path).read_text())


def main():
    manifest = read(RUN / 'freeze_manifest.json')
    runner.check_freeze(manifest)
    assert not (RUN / 'EXECUTION_STARTED.json').exists()
    assert not (RUN / 'live').exists()
    source = Path(manifest['source_run'])
    replay = RUN / 'mock_replay'
    summary = read(replay / 'generation_summary.json')
    assert summary['status'] == 'FINISHED'
    assert summary['all_replay_slots_consumed'] is True
    assert summary['external_http_attempt_count'] == 0
    assert summary['new_generation_evidence'] is False
    records = read(replay / 'complete_call_records.json')
    assert len(records) == 18
    prompt_checks, source_checks, capacities = [], [], []
    fields = ('candidate_id', 'question_type', 'question', 'options', 'items', 'gold', 'evidence_refs')
    total_candidates = 0
    for root in runner.ROOTS:
        original = source / 'qa/roots' / root
        frozen = RUN / 'frozen' / root
        mocked = replay / 'roots' / root
        assert sha256_file(original / 'qa_source.json') == sha256_file(frozen / 'qa_source.json')
        assert sha256_file(original / 'qa_source.json') == sha256_file(mocked / 'qa_source.json')
        context = read(frozen / 'qa_feedback_context.json')
        provenance = read(mocked / 'qa_feedback_provenance.json')
        assert provenance['qa_prompt_contains_feedback'] is True
        assert provenance['qa_source_contains_feedback'] is False
        assert provenance['memory_in_prompt'] is False
        assert read(mocked / 'qa_feedback_context.json') == context
        old_pool, new_pool = read(original / 'candidate_pool.json'), read(mocked / 'candidate_pool.json')
        assert [[q.get(k) for k in fields] for q in old_pool] == [[q.get(k) for k in fields] for q in new_pool]
        total_candidates += len(new_pool)
        preserved = BASE / 'feedback' / root / 'fact_feedback_preserved.json'
        original_feedback = source.parent / 'feedback' / root / 'feedback_input.json'
        assert preserved.read_bytes() == original_feedback.read_bytes()
        difficulty = read(BASE / 'feedback' / root / 'difficulty_feedback.json')
        compiler._verify_sources(difficulty['source_provenance'])
        regenerated = compiler.project_feedback(root_id=root, difficulty=difficulty,
            correction=read(preserved), qa_source=read(original / 'qa_source.json'))
        assert compiler.compile_context(projection=regenerated, role_guidance=difficulty['role_guidance']) == context
        source_checks.append({'root_id': root, 'source_and_dates_byte_identical': True,
                              'original_correction_byte_identical': True, 'candidate_count': len(new_pool)})
        for record in (r for r in records if r['root_id'] == root):
            rid = record['request_id']
            old = source / 'qa/execution/calls' / root / rid
            new = replay / 'execution/calls' / root / rid
            old_request, new_request = read(old / 'request.json'), read(new / 'request.json')
            suffix = context['intent_suffix'] if record['stage'] == 'miner' else context['slot_suffixes'][0]
            assert new_request['payload']['messages'][0]['content'] == old_request['payload']['messages'][0]['content'] + '\n' + suffix
            assert len(new_request['payload']['messages']) == 1
            stripped_old = {k: v for k, v in old_request['payload'].items() if k != 'messages'}
            stripped_new = {k: v for k, v in new_request['payload'].items() if k != 'messages'}
            assert stripped_old == stripped_new
            capacity = read(new / 'capacity.json')
            assert capacity['status'] == 'PASS' and capacity['truncation'] is False
            capacities.append(capacity)
            prompt_checks.append({'request_id': rid, 'stage': record['stage'],
                                  'only_prompt_change_is_feedback_suffix': True,
                                  'same_request_parameters': True})
    assert total_candidates == 29
    output = {'status': 'PASS', 'external_http_attempt_count': 0,
        'new_model_generation_count': 0, 'new_solver_calls': 0,
        'effectiveness_assessed': False, 'live_execution_started': False,
        'source_checks': source_checks, 'request_checks': prompt_checks,
        'historical_replay_calls': len(records), 'historical_candidates_unchanged': total_candidates,
        'replay_prompt_tokens_local_proxy': sum(c['input_tokens'] for c in capacities),
        'maximum_replay_required_context_tokens': max(c['required_context_tokens'] for c in capacities),
        'proxy_context_guard': capacities[0]['context_window_tokens'],
        'real_service_capacity': 'UNKNOWN', 'real_reasoning_configuration': 'UNKNOWN',
        'real_cost': 'UNKNOWN', 'next_run_budget': runner.schedule()}
    atomic_json(BASE / 'OFFLINE_VERIFICATION.json', output)
    print(json.dumps({k: output[k] for k in ('status', 'external_http_attempt_count',
        'historical_replay_calls', 'historical_candidates_unchanged')}))


if __name__ == '__main__':
    main()
