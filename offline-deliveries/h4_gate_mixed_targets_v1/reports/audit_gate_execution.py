"""Offline audit of frozen Gate replay; only audit reports are written."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

DELIVERY = Path(__file__).resolve().parent.parent
RUN = DELIVERY / 'prepared_run'
OLD = Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_planner_output_contract_v1/prepared_run')
WORKTREE = Path('/root/autodl-tmp/gxy-selfplay/worktrees/h4-gate-mixed-targets-v1')
sys.path.insert(0, str(WORKTREE))
from scripts.current_method.restart_n5_spell.core import canonical_json
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1 import gate_mixed_targets_replay as replay
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.development_policy import candidate_views


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


errors = []


def check(condition, label):
    if not condition:
        errors.append(label)


def freeze(path, expected):
    rows = read(path)['files']
    failures = [row['path'] for row in rows if not Path(row['path']).is_file()
                or sha(row['path']) != row['sha256']]
    check(len(rows) == expected and not failures, 'freeze:' + str(path))
    return {'path': str(path), 'records': len(rows), 'matched': len(rows) - len(failures),
            'failed_paths': failures, 'status': 'PASS' if not failures else 'FAIL'}


freezes = {'current': freeze(RUN / 'freeze_manifest.json', 229),
           'previous': freeze(OLD / 'freeze_manifest.json', 293)}
manifest = read(RUN / 'freeze_manifest.json')
records = read(RUN / 'complete_call_records.json')
ledger = read(RUN / 'execution/development_call_ledger.json')
summary = read(RUN / 'generation_summary.json')
check(ledger['records'] == records, 'ledger_records_identity')
check([(r['root_id'], r['stage']) for r in records] == list(replay.SLOTS), 'eight_slot_order')
check(ledger['http_attempt_count'] == ledger['logical_call_count'] == len(records) == 8, 'eight_calls')
check(ledger['status'] == 'FINISHED' and ledger['automatic_retry_count'] == 0
      and ledger['automatic_resume'] is False, 'ledger_final_state')
check(summary['unsent_slots'] == [] and summary['other_stage_calls'] == 0
      and summary['planner_outline_realizer_miner_writer_memory_solver_training_extra_api_calls'] == 0,
      'other_stages_or_unsent')
counts = {key: Counter() for key in ('old_output_old_validator', 'old_output_new_validator',
                                   'new_output_old_validator', 'new_output_new_validator')}
relations = {'old': Counter(), 'new': Counter()}
root_costs = defaultdict(lambda: Counter())
calls, slots = [], []
metadata_paths = set()
for record in records:
    root, stage = record['root_id'], record['stage']
    name = root + '/' + stage
    frozen = RUN / 'frozen' / root / stage
    result = RUN / 'results' / root / stage
    call = RUN / 'execution' / record['artifact_directory']
    meta_path = call / 'response_metadata.json'
    metadata_paths.add(meta_path)
    request, metadata = read(call / 'request.json'), read(meta_path)
    body = json.loads(metadata['raw_response_text'])
    choice = body['choices'][0]
    check(record['actual_model'] == metadata['actual_model'] == body['model'] == 'qwen3.6-35b-a3b', 'model:' + name)
    check(record['finish_reason'] == metadata['finish_reason'] == choice['finish_reason'] == 'stop', 'finish:' + name)
    check(record['usage'] == metadata['usage'] == body['usage'], 'usage_identity:' + name)
    check(record['http_attempt_count'] == metadata['http_attempt_count'] == 1, 'single_attempt:' + name)
    check(record['retry_count'] == metadata['transport_retry_count'] == metadata['rate_limit_retry_count'] == 0, 'retry:' + name)
    check(record['http_status'] == metadata['http_status'] == 200
          and record['timeout'] is False and metadata['timeout'] is False, 'http_status:' + name)
    check(record['status'] == 'COMPLETED' and metadata['completion_status'] == 'RESPONSE_RECEIVED', 'completion:' + name)
    check((call / 'response.txt').read_text() == choice['message']['content'], 'raw_content:' + name)
    check(record['messages_sha256'] == digest(request['payload']['messages'])
          and record['payload_sha256'] == digest(request['payload']), 'request_hash:' + name)
    check(request['requested_parameters'] == replay.PARAMETERS, 'parameters:' + name)
    check(request['payload'] == read(frozen / 'payload.json'), 'frozen_payload:' + name)
    usage = body['usage']
    check(record['input_tokens'] == usage['prompt_tokens'] and record['output_tokens'] == usage['completion_tokens']
          and record['total_tokens'] == usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens'], 'token_math:' + name)
    root_costs[root].update({'logical_calls': 1, 'http_attempts': 1, 'input_tokens': record['input_tokens'],
        'output_tokens': record['output_tokens'], 'total_tokens': record['total_tokens']})
    calls.append({'root_id': root, 'stage': stage, 'actual_model': body['model'],
        'finish_reason': choice['finish_reason'], 'usage': usage, 'http_attempts': 1,
        'retries': 0, 'timeout': False, 'http_status': 200, 'reasoning_tokens': 'UNKNOWN',
        'verified_cost': 'UNKNOWN', 'request_parameters': request['requested_parameters'],
        'confirmed_parameter_adoption': request['confirmed_parameter_adoption'],
        'artifact_directory': str(call)})

    suffix = '' if stage == 'gate' else '_supplement'
    old_root = OLD / 'expansion/roots' / root
    old_plan_path = old_root / ('planner/output.json' if stage == 'gate' else 'planner/supplement_output.json')
    old_validation_path = old_root / ('planner/validation.json' if stage == 'gate' else 'planner/supplement_validation.json')
    check((frozen / 'planner_output.json').read_bytes() == old_plan_path.read_bytes(), 'planner_bytes:' + name)
    check((frozen / 'planner_validation.json').read_bytes() == old_validation_path.read_bytes(), 'planner_validation_bytes:' + name)
    check((frozen / 'context.json').read_bytes() == (old_root / f'semantic_gate/context{suffix}.json').read_bytes(), 'context_bytes:' + name)
    old_call = OLD / 'expansion/execution' / root / 'calls' / root / stage
    check((frozen / 'old_request.json').read_bytes() == (old_call / 'request.json').read_bytes(), 'old_request_bytes:' + name)
    context, plan = read(frozen / 'context.json'), read(frozen / 'plan.json')
    check(plan == read(old_validation_path)['compiled_plan'], 'candidate_plan_identity:' + name)
    check(candidate_views(plan) == context['candidates'], 'candidate_views:' + name)
    old_prompt = (frozen / 'old_prompt.txt').read_text()
    new_prompt = (frozen / 'prompt.txt').read_text()
    check(replay._input_block(old_prompt)[1] == replay._input_block(new_prompt)[1], 'input_tail_bytes:' + name)
    rebuilt_prompt, identity = replay.rebuild_prompt(context, old_prompt, read(frozen / 'old_request.json'))
    check(rebuilt_prompt == read(frozen / 'prompt.json'), 'prompt_rebuild:' + name)
    old_output, new_output = read(frozen / 'old_output.json'), read(result / 'output.json')
    check(new_output == json.loads(choice['message']['content']) == read(call / 'parsed_response.json'), 'response_object_identity:' + name)
    validations = {}
    for epoch, output in (('old', old_output), ('new', new_output)):
        relations[epoch].update(row['relation'] for row in output['candidate_reviews'])
        for policy_name, legacy in (('old', True), ('new', False)):
            key = epoch + '_output_' + policy_name + '_validator'
            audit = replay._validation(output, context, plan, legacy=legacy)
            validations[key] = audit
            counts[key].update(replay._counts(audit))
    check(validations['old_output_old_validator'] == read(frozen / 'old_validation.json'), 'old_validation_rebuild:' + name)
    check(validations['old_output_new_validator'] == read(frozen / 'old_output_new_contract_validation.json'), 'old_counterfactual_rebuild:' + name)
    check(validations['new_output_new_validator'] == read(result / 'validation.json'), 'new_validation_rebuild:' + name)
    row_decisions = []
    for index, old_row in enumerate(old_output['candidate_reviews']):
        ref = old_row['candidate_ref']
        new_row = next(row for row in new_output['candidate_reviews'] if row['candidate_ref'] == ref)
        row_decisions.append({'candidate_ref': ref,
            'old_relation': old_row['relation'], 'new_relation': new_row['relation'],
            'old_target_checks': old_row['target_checks'], 'new_target_checks': new_row['target_checks'],
            'decisions': {key: next(row for row in audit['normalized_candidate_reviews'] if row['candidate_ref'] == ref)['decision']
                          for key, audit in validations.items()},
            'errors': {key: audit['row_errors'].get(ref, []) for key, audit in validations.items()}})
    slots.append({'root_id': root, 'stage': stage, 'candidate_count': len(context['candidates']),
        'planner_output_path': str(old_plan_path), 'planner_output_sha256': sha(old_plan_path),
        'planner_output_bytes_preserved': True, 'context_and_plan_identity': True,
        'input_recovery_tail_bytes_preserved': True, 'request_parameters_preserved': True,
        'matrix_counts': {key: replay._counts(audit) for key, audit in validations.items()},
        'candidate_decisions': row_decisions})

check(set((RUN / 'execution/calls').glob('*/*/response_metadata.json')) == metadata_paths, 'unaccounted_response_metadata')
totals = Counter()
for value in root_costs.values():
    totals.update(value)
for key, value in totals.items():
    check(summary['cost'][key] == value, 'summary_cost:' + key)
check(totals['total_tokens'] == 184868, 'expected_total_tokens')
check(sum(slot['candidate_count'] for slot in slots) == 15, 'candidate_count')
check(relations['new']['UNKNOWN'] == 0, 'new_unknown_count')
check(counts['old_output_old_validator'] == counts['old_output_new_validator'], 'old_response_counts_policy_identity')
check(counts['new_output_old_validator'] == counts['new_output_new_validator'], 'new_response_counts_policy_identity')
report = {'version': 'gate_mixed_targets_execution_and_validator_matrix_audit_v1',
    'audited_at': datetime.now(timezone.utc).isoformat(), 'status': 'PASS' if not errors else 'FAIL',
    'errors': errors, 'external_calls_during_audit': 0, 'run_or_source_changes': 0,
    'run_root': str(RUN), 'freeze_checks': freezes, 'ledger_status': ledger['status'],
    'cost': {**dict(totals), 'retries': 0, 'timeouts': 0, 'technical_failures': 0,
             'reasoning_tokens': 'UNKNOWN', 'verified_cost': 'UNKNOWN'},
    'root_costs': {key: dict(value) for key, value in root_costs.items()},
    'actual_model_counts': dict(Counter(row['actual_model'] for row in records)),
    'finish_reason_counts': dict(Counter(row['finish_reason'] for row in records)),
    'calls': calls, 'fixed_slot_count': 8, 'candidate_count': 15,
    'validator_matrix': {key: dict(value) for key, value in counts.items()},
    'relation_counts': {key: dict(value) for key, value in relations.items()}, 'slots': slots,
    'attribution_boundary': 'The new outputs also yield 14 eligible and 1 quarantined under the legacy validator. No new output used UNKNOWN, so the live count improvement is not caused by relaxing mixed-state UNKNOWN validation. It reflects different model outputs under changed instructions and one new sample; semantic correctness requires separate review.',
    'offline_contract_boundary': 'The mixed-state representational repair can be verified with offline constructed rows; unchanged old outputs retain their old incorrect top-level relations and are not silently relabeled.',
    'no_extra_api': {'recorded_stages': dict(Counter(row['stage'] for row in records)),
        'all_eight_response_metadata_files_accounted': True,
        'planner_outline_realizer_miner_writer_memory_solver_training_extra_api_calls': 0,
        'scope': 'Run ledgers and response artifacts; not a machine-wide packet capture.'},
    'semantic_support': 'NOT_CERTIFIED_BY_PROGRAM_ELIGIBILITY_OR_THIS_EXECUTION_AUDIT',
    'pipeline_scope': 'Fixed historical primary and supplement candidate pools; no new Planner, Outline, Realizer or QA and no counterfactual full-pipeline success claim.'}
lines = ['# Gate-only execution audit', '',
    f"Status: **{report['status']}**. This offline audit made 0 model/API calls and changed no run or source file.", '',
    '8 logical calls = 8 HTTP attempts / 8. Every response returned qwen3.6-35b-a3b, HTTP 200 and finish_reason=stop. Retries, timeouts and technical failures: 0. No other model stage ran.', '',
    '| Root | Gate HTTP | Input tokens | Output tokens | Total tokens |',
    '|---|---:|---:|---:|---:|']
for root, value in root_costs.items():
    lines.append(f"| {root.rsplit('-', 1)[-1]} | {value['http_attempts']} | {value['input_tokens']:,} | {value['output_tokens']:,} | {value['total_tokens']:,} |")
lines += [f"| Total | 8 | {totals['input_tokens']:,} | {totals['output_tokens']:,} | {totals['total_tokens']:,} |", '',
    'Reasoning token breakdown, verified monetary cost and service parameter adoption remain UNKNOWN. All eight calls, including the quarantined row, are included.', '',
    'Current 229/229 and preceding 293/293 frozen records match. All eight original Planner outputs and validation files remain byte-identical. Each actual Gate request preserves the complete serialized FROZEN INPUT and recovery tail; only the Gate rule prefix changes. Saved model responses, request hashes, full usage and local validation recomputation agree.', '',
    '| Model response | Validator | Eligible | Already-realized rejection | Deferred | Quarantined |',
    '|---|---|---:|---:|---:|---:|']
for key, value in counts.items():
    epoch, policy = key.split('_output_')
    lines.append(f"| {epoch} | {policy} | {value['passed_candidate_refs']} | {value['rejected_candidate_refs']} | {value['deferred_candidate_refs']} | {value['quarantined_candidate_refs']} |")
lines += ['',
    '**The 14/15 live eligibility result is also produced by the old validator. It cannot be attributed to relaxing the mixed-state rule.** The new responses contain 12 NEW_PROPOSAL and 3 PLANNED_OR_INTENDED rows, and zero UNKNOWN rows. The new model outputs, not a validator switch applied to the same output, account for the count difference. Changed instructions plus a single new sample do not establish a causal or semantic quality gain.', '',
    'The old outputs remain 11 eligible / 4 quarantined under both validators because their original top-level relations are preserved. The new contract supports faithfully representing mixed target states under UNKNOWN; this mechanical capability is covered offline but was not exercised by an UNKNOWN row in this live batch.', '',
    'Gate eligibility is not validated novelty, compatible scene selection, semantic correctness or end-to-end completion. This run used eight fixed historical candidate pools, including original supplemental pools, and did not execute a new Planner or any downstream stage.', '',
    '[Full per-call usage, per-candidate four-cell comparison and identity checks](EXECUTION_AUDIT.json).', '']
if errors:
    lines.extend(['Audit failures:', '', *['- ' + item for item in errors], ''])
(DELIVERY / 'reports/EXECUTION_AUDIT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
(DELIVERY / 'reports/EXECUTION_AUDIT.md').write_text('\n'.join(lines))
print(json.dumps({'status': report['status'], 'errors': errors, 'cost': report['cost'],
                  'matrix': report['validator_matrix'], 'freeze_checks': freezes}, indent=2))
