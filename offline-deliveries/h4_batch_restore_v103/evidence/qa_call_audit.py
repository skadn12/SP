"""Independent stdlib audit of actual QA calls, complete inputs and token usage.

Reads actual request payloads, proof/transport records and response metadata.
Does not inspect response text, private question Gold or private candidate pools.
Source judgments and final content counts have separate owners and receipts.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
SHARD = BASE / 'shards/shard_01'


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()
def bind(path): return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def main():
    output = BASE / 'evidence/QA_CALL_ACCOUNTING.json'
    if output.exists(): raise SystemExit('refuse to overwrite ' + str(output))
    failures = []
    def check(condition, code, **details):
        if not condition: failures.append({'code': code, **details})
    rows = []; roles = Counter(); usage = Counter(); finishes = Counter()
    attempts = 0
    for path in sorted(SHARD.rglob('requests/*/REQUEST.json')):
        folder = path.parent
        payload = read(path); proof = read(folder / 'PROOF.json')
        role = proof['role']; roles[role] += 1
        text = payload['messages'][0]['content']
        ctx = json.JSONDecoder().raw_decode(text.split('INPUT_JSON\n', 1)[1])[0]
        root = ctx['root_id']; short = root.rsplit('-', 1)[-1]
        source_path = BASE / 'sources' / ('root' + short + '_SOURCE.json')
        source = read(source_path)
        check(short in ['24719', '38601'] and ctx['source_turns'] == source['turns']
              and proof['source_sha256'] == sha(source_path),
              'actual_qa_complete_source_changed', request=str(folder))
        check(payload['max_tokens'] == 6144 and payload['temperature'] == .25
              and payload['model'] == 'Qwen/Qwen3.6-35B-A3B'
              and digest(payload) == proof['payload_sha256'],
              'actual_qa_payload_sampling_or_model_changed', request=str(folder))
        check(proof['capacity']['status'] == 'PASS' and proof['capacity']['truncation'] is False,
              'actual_qa_capacity_failed_or_truncated', request=str(folder))
        receipt = proof['feedback_receipt']
        start = text.index('[PREVIOUS_TRAINING_ROUND_FEEDBACK_V4]')
        end = text.index('[/PREVIOUS_TRAINING_ROUND_FEEDBACK_V4]') + len('[/PREVIOUS_TRAINING_ROUND_FEEDBACK_V4]')
        suffix = '\n' + text[start:end] + '\n'
        check(hashlib.sha256(suffix.encode()).hexdigest() == receipt['suffix_sha256']
              and receipt['feedback_appended'] is True and proof['current_solver_feedback'] is False,
              'actual_qa_previous_training_feedback_missing_or_changed', request=str(folder))
        check('gold' not in {k.casefold() for k in ctx}, 'qa_private_gold_input_field', request=str(folder))
        transport = sorted(folder.glob('HTTP_ATTEMPT*.json'))
        check(len(transport) == 1, 'qa_unplanned_transport_retry', request=str(folder))
        for path in transport:
            attempt = read(path); attempts += 1
            check(attempt['actual_payload_equals_frozen'] is True
                  and attempt['payload_sha256'] == digest(payload) and attempt['transport_retries'] == 0,
                  'actual_qa_transport_payload_changed', request=str(folder))
        result = read(folder / 'HTTP_RESULT.json')
        check(result['http_status'] == 200 and result['outcome'] == 'response',
              'qa_transport_not_successful', request=str(folder))
        metadata = read(folder / 'RESPONSE.json')['metadata']
        usage.update({k: v for k, v in metadata['usage'].items() if type(v) is int})
        finishes[str(metadata['finish_reason'])] += 1
        check(metadata['requested_model'] == payload['model'] == metadata['returned_model']
              and metadata['http_attempt_count'] == 1 and metadata['transport_retry_count'] == 0,
              'qa_response_model_or_transport_count_changed', request=str(folder))
        rows.append({'role': role, 'root_id': root, 'source_turn_count': len(source['turns']),
                     'request': bind(folder / 'REQUEST.json'), 'proof': bind(folder / 'PROOF.json'),
                     'response': bind(folder / 'RESPONSE.json'), 'http_attempts': len(transport),
                     'source': bind(source_path), 'usage': metadata['usage'],
                     'finish_reason': metadata['finish_reason'], 'complete_history_exactly_preserved': True,
                     'previous_training_feedback_exact_suffix_preserved': True})
    generation = read(SHARD / 'BATCH_GENERATION.COMPLETE.json')
    check(len(rows) == attempts == generation['http_attempts'] == 4 <= 6,
          'qa_global_count_or_budget_changed')
    check(roles == {'asset_generator': 2, 'ordering_writer': 2}, 'qa_role_counts_changed')
    check(finishes == {'stop': 4}, 'qa_non_stop_finish')
    prior = read(BASE / 'evidence/EXPANSION_STAGE_AND_PROBE_ACCOUNTING.json')
    before = prior['combined_before_qa']
    total_usage = {k: before['usage'].get(k, 0) + usage.get(k, 0)
                   for k in ['prompt_tokens', 'completion_tokens', 'total_tokens']}
    rejected = []
    for path in sorted((SHARD / 'ordering').glob('*/REJECTIONS.json')):
        rejected.extend(read(path))
    check(len(rejected) == 1 and rejected[0]['candidate_id'].endswith('24719-v98_shard_01-ordering-02')
          and rejected[0]['reasons'] == ['ordering_writer_forbidden_gendered_pronoun'],
          'ordering_generation_shortfall_changed')
    preserved = read(BASE / 'PRESERVATION_BEFORE.json')['files']
    changes = [row['path'] for row in preserved
               if not Path(row['path']).is_file() or sha(row['path']) != row['sha256']]
    check(not changes, 'historical_preserved_files_changed_after_qa', paths=changes)
    result = {'version': 'v103_independent_actual_qa_call_accounting_v1',
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'FAIL' if failures else 'PASS', 'failures': failures,
              'qa_generation': {'logical_requests': len(rows), 'http_attempts': attempts,
                  'http_max': 6, 'roles': roles, 'usage': usage, 'finish_reasons': finishes},
              'all_v103_model_work': {'logical_requests': before['logical_requests'] + len(rows),
                  'http_attempts': before['http_attempts'] + attempts, 'usage': total_usage},
              'requests': rows, 'ordering_planned_slots': 4,
              'ordering_mechanically_generated_public_questions': 3,
              'ordering_generation_rejections': rejected,
              'ordering_semantic_admission_is_separate': True,
              'historical_protected_files_rechecked_after_qa': len(preserved),
              'prior_expansion_and_probe_accounting': bind(BASE / 'evidence/EXPANSION_STAGE_AND_PROBE_ACCOUNTING.json'),
              'qa_generation_receipt': bind(SHARD / 'BATCH_GENERATION.COMPLETE.json'),
              'script_sha256': sha(__file__), 'auditor_model_calls': 0, 'auditor_http_calls': 0,
              'private_question_gold_or_pool_read': False, 'raw_response_text_inspected': False,
              'semantic_certified': False,
              'limitation': 'Counts and exact actual input bindings do not certify question meaning, difficulty or training admission.'}
    with output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(json.dumps({'status': result['status'], 'out': str(output), 'sha256': sha(output),
                      'failures': failures, 'qa_usage': usage, 'all_usage': total_usage}, ensure_ascii=False, indent=2))
    raise SystemExit(bool(failures))


if __name__ == '__main__': main()
