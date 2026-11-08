"""Offline regression and formal preparation for the staged Planner contract."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import difflib
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

BASE = Path('/root/autodl-tmp/gxy-selfplay')
DEST = Path(__file__).resolve().parent
SOURCE = BASE / 'worktrees/h4-realizer-state-contract-v1'
WORKTREE = BASE / 'worktrees/h4-planner-output-contract-v1'
OLD_RUN = BASE / 'restart-n5-runs/h4_full_pipeline_realizer_state_contract_v1_live_1'
INPUTS = BASE / 'offline-deliveries/h4_codex_qa_feedback_filter_v1/inputs'
PREPARED = DEST / 'prepared_run'
PACKAGE = 'scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1'
sys.path.insert(0, str(WORKTREE))

from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1 import full_pipeline_planner_recovery_more as entry
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.development_validate import check_freeze
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.planner_candidate_recovery import build_supplement_prompt, validate_candidate_batch
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.contracts import build_planner_outline_pairs


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def file_id(path):
    return {'path': str(path), 'sha256': sha(path)}


def optional(path):
    return read(path) if path.is_file() else {}


def main():
    if (PREPARED / 'EXECUTION_STARTED.json').exists():
        raise FileExistsError('cannot_reverify_executed_preparation')
    source_manifest = read(OLD_RUN / 'freeze_manifest.json')
    check_freeze(source_manifest)
    check_freeze(read(INPUTS / 'manifest.json'))
    prior_report = read(BASE / 'offline-deliveries/h4_realizer_state_contract_v1/reports/PREPARATION.json')
    modules = list(prior_report['tests']['modules']) + [
        'test_planner_output_contract','test_planner_output_entry','test_planner_output_workflow']
    test_log = io.StringIO()
    with patch('socket.socket.connect', side_effect=AssertionError('NETWORK_FORBIDDEN')), \
         patch('socket.socket.connect_ex', side_effect=AssertionError('NETWORK_FORBIDDEN')), \
         patch('socket.create_connection', side_effect=AssertionError('NETWORK_FORBIDDEN')), \
         redirect_stdout(test_log), redirect_stderr(test_log):
        suite = unittest.defaultTestLoader.loadTestsFromNames([PACKAGE+'.'+name for name in modules])
        result = unittest.TextTestRunner(stream=test_log, verbosity=2).run(suite)
    (DEST / 'test_output.txt').write_text(test_log.getvalue())
    test_summary = dict(tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                        skipped=len(result.skipped), modules=modules, external_calls=0,
                        network_forbidden=True, scope='Offline contracts and mocked formal workflow; not live quality evidence')
    write(DEST / 'TEST_RESULTS.json', test_summary)
    if not result.wasSuccessful():
        raise AssertionError('offline_tests_failed_see_test_output')
    options = dict(planner_method='v2', review_contract='scoped-v1',
                   realizer_contract='keyed-state-v1', planner_output_contract='staged-v1')
    checks, history = [], []
    with patch('socket.socket.connect', side_effect=AssertionError('NETWORK_FORBIDDEN')) as network, \
         patch('socket.create_connection', side_effect=AssertionError('NETWORK_FORBIDDEN')) as connect, \
         patch.object(entry.shared, 'resolve_credential', side_effect=AssertionError('NO_CREDENTIAL_READ')) as credentials, \
         patch.object(entry.shared, 'ProfileClient', side_effect=AssertionError('NO_TRANSPORT')) as transport, \
         patch.object(entry.shared, 'QARuntime', side_effect=AssertionError('NO_QA_RUNTIME')) as qa:
        if PREPARED.exists():
            prepared = read(PREPARED / 'freeze_manifest.json')
            check_freeze(prepared)
            assert prepared['execution_profile'] == entry.spec_for(**options).execution_profile.to_dict()
            assert prepared['input_bundle_manifest_path'] == str(INPUTS / 'manifest.json')
        else:
            prepared = entry.prepare(PREPARED, input_bundle=INPUTS, **options)
        tokenizer = entry.shared.tokenizer_from(source_manifest['tokenizer_path'])
        counter = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
        class Boundary(Exception):
            pass
        for root in entry.ROOT_ORDER:
            holder = {}
            def boundary(stage, prompt, maximum):
                assert stage == 'planner'
                holder.update(deepcopy(prompt))
                raise Boundary()
            try:
                entry.shared.run_round(read(OLD_RUN / 'frozen' / root / 'inputs.json'), boundary,
                    lambda name,value: None, counter,
                    execution_profile=entry.spec_for('v2','scoped-v1','keyed-state-v1').execution_profile)
            except Boundary:
                pass
            assert holder == read(OLD_RUN / 'frozen' / root / 'planner_prompt.json'), 'legacy_prompt_changed:'+root
            frozen = PREPARED / 'frozen' / root
            new = read(frozen / 'planner_prompt.json')
            inputs = read(frozen / 'inputs.json')
            assert inputs == read(INPUTS / root / 'inputs.json')
            previous_input = read(OLD_RUN / 'frozen' / root / 'inputs.json')
            for key in ('parent','prior_expanded','snapshot','feedback_dialogue','conversation_date_segments'):
                assert inputs[key] == previous_input[key], key
            new_request = read(frozen / 'planner_request.json')
            old_request = read(OLD_RUN / 'frozen' / root / 'planner_request.json')
            assert {k:v for k,v in new_request.items() if k!='messages'} == {k:v for k,v in old_request.items() if k!='messages'}
            shared = new['planner_shared_prompt']
            for turn in inputs['prior_expanded']['turns']:
                assert turn['turn_ref'] in shared and turn['text'] in shared
            assert 'Output only this JSON object.' not in shared
            assert 'BOTH primary and supplement' not in shared
            assert 'exactly one complete plan' not in shared
            assert '"name": "Elena", "relationship_basis": "new neighbor' not in shared
            root_dir = OLD_RUN / 'expansion/roots' / root
            validation = read(root_dir / 'planner/validation.json')
            old_supplement = read(root_dir / 'planner/supplement_prompt_manifest.json')
            compiled = validation['compiled_plan']
            rejection = {
                'structural': [{'candidate_ref': row['candidate_ref'], 'errors': row['errors']}
                               for row in validation['candidate_records'] if not row['valid']],
                'gate': read(root_dir / 'semantic_gate/merged_validation.json'),
                'outline': optional(root_dir / 'planner/outline_semantic_validation.json'),
                'selection': optional(root_dir / 'planner/outline_semantic_selection.json'),
                'gate_raw': read(root_dir / 'semantic_gate/output.json'),
                'outline_raw': optional(root_dir / 'planner/outline_semantic_output.json'),
                'pair_menu': [{'pair_ref': p['pair_ref'], 'left_candidate_ref':p['left']['candidate_ref'],
                               'right_candidate_ref':p['right']['candidate_ref']} for p in build_planner_outline_pairs(compiled)],
            }
            supplement = build_supplement_prompt(new,
                candidate_records=validation['candidate_records'],
                qualified_candidate_refs=old_supplement['qualified_candidate_refs'],
                rejection_reasons=rejection, planner_method_policy='planner_method_v2',
                planner_output_policy='staged_contract_v1')
            assert supplement['planner_shared_prompt'] == shared
            assert new['planner_output_contract']['text'] not in supplement['prompt']
            assert supplement['maximum_added_candidates'] == old_supplement['maximum_added_candidates']
            assert supplement['candidate_recovery_context']['original_candidates'] == old_supplement['candidate_recovery_context']['original_candidates']
            assert supplement['candidate_recovery_context']['pair_relations'] == old_supplement['candidate_recovery_context']['pair_relations']
            dest = DEST / 'supplement_previews' / root
            supplement['preview_only'] = 'Historical candidates and reviews used only to exercise current prompt construction; not a response to newly generated primary output.'
            write(dest / 'prompt_manifest.json', {k:v for k,v in supplement.items() if k!='prompt'})
            (dest / 'prompt.txt').write_text(supplement['prompt'])
            payload = entry.shared.build_payload(prepared['public_profile'], entry.shared.messages(supplement))
            capacity = entry.shared.annotate_school_capacity(entry.shared.capacity_check(
                prepared['public_profile'], tokenizer, payload['messages'], entry.spec_for(**options).stage_budgets['planner_supplement']), prepared['public_profile'])
            assert capacity['status']=='PASS' and capacity['truncation'] is False
            write(dest / 'request.json', payload)
            write(dest / 'capacity.json', capacity)
            # Existing raw responses remain failures where they failed; this is
            # compatibility replay, not a new sample or evidence of improvement.
            actual_parent = deepcopy(inputs['parent'])
            actual_parent['dialogue']['turns'] = [dict(turn_id=t['turn_ref'], speaker=t['speaker'], text=t['text'])
                                                 for t in inputs['prior_expanded']['turns']]
            candidates = read(root_dir / 'memory/event_candidates.json')['candidates']
            tracks = read(root_dir / 'event_verifier/materialized_tracks.json')['verified_tracks']
            replay = validate_candidate_batch(read(root_dir / 'planner/output.json'), parent=actual_parent,
                candidates=candidates, verified_tracks=tracks, speaker_registry=read(root_dir / 'planner/speaker_registry.json'),
                planner_method_policy='planner_method_v2', planner_output_policy='staged_contract_v1')
            assert [r['candidate_ref'] for r in replay['retained_records']] == [r['candidate_ref'] for r in validation['retained_records']]
            write(DEST / 'historical_replay' / root / 'primary_validation.json', replay)
            history.append(dict(root_id=root, raw_count=len(replay['candidate_records']),
                                retained_count=len(replay['retained_records']),
                                acceptance_unchanged=True, model_call_performed=False))
            checks.append(dict(root_id=root, legacy_prompt_byte_identity=True,
                complete_transcript_preserved=True, memory_dates_unchanged=True,
                source_feedback_input=str(INPUTS / root / 'feedback_input.json'),
                primary_contract=new['planner_output_contract']['top_level_fields'],
                supplement_contract=supplement['planner_output_contract']['top_level_fields'],
                supplement_shared_text_exact=True, locks_reviews_candidates_preserved=True,
                previous_request_parameters_unchanged=True,
                primary_capacity=read(frozen/'planner_capacity.json'), supplement_capacity=capacity,
                service_support='Unchanged service assumptions; no new strict-schema request or capacity probe'))
        for blocked in (network,connect,credentials,transport,qa):
            blocked.assert_not_called()
    check_freeze(source_manifest)
    check_freeze(read(INPUTS / 'manifest.json'))
    check_freeze(prepared)
    changes, diff = [], []
    for path in sorted((WORKTREE / PACKAGE.replace('.','/')).glob('*.py')):
        old = SOURCE / path.relative_to(WORKTREE)
        if old.exists() and path.read_bytes() == old.read_bytes():
            continue
        changes.append(dict(path=str(path), baseline_path=str(old) if old.exists() else None,
                            before_sha256=sha(old) if old.exists() else None, after_sha256=sha(path)))
        diff.extend(difflib.unified_diff(old.read_text().splitlines(True) if old.exists() else [],
                                       path.read_text().splitlines(True),
                                       fromfile=str(old) if old.exists() else '/dev/null', tofile=str(path)))
    (DEST / 'changes.patch').write_text(''.join(diff))
    write(DEST / 'CHANGES.json', changes)
    write(DEST / 'VERIFICATION.json', dict(status='PASS_OFFLINE_PREPARED_NOT_EXECUTED',
        external_calls=0, credential_reads=0, transport_constructions=0,
        source_run_freeze_entries_unchanged=len(source_manifest['files']),
        tests=test_summary, root_checks=checks, historical_replay=history,
        prepared_run=str(PREPARED), changed_python_files=len(changes),
        intended_live_http_cap=68, new_live_budget_authorized=False,
        semantic_quality='NOT_MEASURED_NO_NEW_GENERATIONS',
        scope='Staged Planner output contract plus previously prepared Codex feedback; Gate/Outline/Realizer/Writer semantics and call schedule unchanged'))
    print(json.dumps(dict(status='PASS_OFFLINE_PREPARED_NOT_EXECUTED', tests=result.testsRun,
                         roots=len(checks), external_calls=0, prepared_run=str(PREPARED)),ensure_ascii=False))


if __name__ == '__main__':
    main()
