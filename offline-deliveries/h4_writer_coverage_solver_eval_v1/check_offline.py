"""Exercise every frozen request with mock answers; never create a model engine."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import evaluate as e
from transformers import AutoTokenizer
from scripts.current_method.restart_n5_spell.backends import BackendResponse
from scripts.current_method.restart_n5_spell.h4_round0_b0_k3_difficulty_v1.io_utils import stable_seed


def expect_error(thunk, text):
    try:
        thunk()
    except RuntimeError as error:
        assert text in str(error), str(error)
    else:
        raise AssertionError('expected:' + text)


def main(run):
    e.verify_sha_sidecar(run / 'preflight.json')
    pre = e.load_json(run / 'preflight.json')
    protocol = e.load_json(run / 'protocol.json')
    assert e.verify_sha_sidecar(run / 'protocol.json') == pre['protocol_sha256']
    e.verify_records(protocol['sources'])
    prepared = Path(protocol['prepared'])
    rows, gates, dialogues = e.load_inputs(prepared)
    tokenizer = AutoTokenizer.from_pretrained(str(e.TOKENIZER), local_files_only=True)
    limit = pre['per_model_requests']
    e.solver.validate_gate_display_policy(protocol, gates)
    checked_prompts = 0
    for row, g in zip(rows, gates):
        dialogue = e.contract.dialogue_rows(dialogues[row['root_id']])
        variants = e.contract.position_variants(row, ordering_display_policy=e.RELABEL_POLICY)
        assert e.contract.audit_rebinding(row, variants)['pass']
        with e.dated_context(dialogues):
            current = e.gate.gate_candidate(row, dialogues[row['root_id']], tokenizer,
                                           root_id=row['root_id'], ordering_display_policy=e.RELABEL_POLICY)
            assert current == g
            for variant in variants:
                no = e.solver.build_prompt(tokenizer, row, variant, dialogue, context_mode='no_context')
                assert no == e.contract.build_prompt(tokenizer, row, variant, dialogue, context_mode='no_context')
                full = e.solver.build_prompt(tokenizer, row, variant, dialogue, context_mode='full_context')
                assert e.writer_sources.render_dialogue(dialogues[row['root_id']], numbered=False) in full
                assert row['question'] in no and row['question'] in full
                gold = variant['displayed_gold']
                assert e.contract.score_output('<box>' + ','.join(gold) + '</box>', row, variant)['correct']
                if row['question_type'] == 'single_choice':
                    wrong = [next(k for k in variant['options'] if k != gold[0])]
                elif row['question_type'] == 'multi_select':
                    wrong = list(gold[1:]) if len(gold) > 1 else [next(k for k in variant['options'] if k not in gold)]
                else:
                    wrong = list(reversed(gold))
                wrong_score = e.contract.score_output('<box>' + ','.join(wrong) + '</box>', row, variant)
                assert wrong_score['scored'] and not wrong_score['correct']
                checked_prompts += 2

    class FakeBackend:
        model_identifier = 'OFFLINE_MOCK_NO_MODEL'
        tokenizer_path = e.TOKENIZER
        max_model_len = e.MAX_MODEL_LEN

        def __init__(self, mode='correct'):
            self.count, self.mode = 0, mode

        def generate(self, request):
            self.count += 1
            if self.mode == 'failure':
                raise ValueError('intentional offline failure')
            answer = 'MALFORMED' if self.mode == 'parse' else '<box>' + request.canonical_answer_text + '</box>'
            finish = {'truncated': 'length', 'unknown': None}.get(self.mode, 'stop')
            return BackendResponse(answer, self.model_identifier, {
                'finish_reason': finish,
                'prompt_token_count': len(tokenizer.encode(request.prompt, add_special_tokens=False)),
                'completion_token_count': len(tokenizer.encode(answer, add_special_tokens=False))})

        def generate_structured(self, request, *, question_type):
            return self.generate(request)

    def scoped(row, g):
        return {**g, 'candidate_pool_sha256': e.sha256_file(prepared / 'candidate_manifest.json'),
                'dialogue_sha256': e.sha256_file(prepared / 'dialogues' / (row['root_id'] + '.json'))}

    failures = []
    with tempfile.TemporaryDirectory(prefix='h4_coverage_solver_mock_') as temp:
        temp = Path(temp)
        ledger = e.NoRetryLedger(temp / 'all_correct', solver_contract_sha256='offline_mock', limit=limit)
        backend = FakeBackend()
        with e.dated_context(dialogues):
            for row, g in zip(rows, gates):
                result = e.solver.evaluate_candidate(candidate=row, expanded=dialogues[row['root_id']],
                    gate_row=scoped(row, g), tokenizer=tokenizer, backend=backend,
                    ledger=ledger, solver_contract_sha256='offline_mock')
                assert all(q['correct'] and q['scored'] for q in result['requests'])
                for q in result['requests']:
                    assert q['seed'] == stable_seed(row['root_id'], row['candidate_id'], q['slot_id'])
                    if row['question_type'] == 'ordering':
                        assert q['canonical_prediction'] == row['gold']
        assert backend.count == ledger.used == limit
        assert ledger.summary()['retried_generation_count'] == 0
        first_state = ledger.states()[0]
        request_contract = first_state['request_contract']
        prompt = Path(first_state['prompt_path']).read_text()
        bad = e.NoRetryLedger(temp / 'no_resend', solver_contract_sha256='offline_mock', limit=1)
        calls = [0]

        def fail():
            calls[0] += 1
            raise ValueError('intentional offline failure')

        expect_error(lambda: bad.invoke(contract=request_contract, prompt=prompt, call=fail),
                     'technical_failure_stop_without_retry')
        expect_error(lambda: bad.invoke(contract=request_contract, prompt=prompt, call=fail),
                     'noncompleted_request_cannot_be_resent')
        assert calls == [1] and bad.summary()['retried_generation_count'] == 0
        for kind in ('single_choice', 'multi_select', 'ordering'):
            index = next((i for i, r in enumerate(rows) if r['question_type'] == kind), None)
            if index is None:
                continue
            row, g = rows[index], gates[index]
            for mode, expected in [('failure', 'technical_failure_stop_without_retry'),
                                   ('parse', 'unscorable_output_stop_without_retry'),
                                   ('truncated', 'nonexplicit_or_truncated_completion_stop'),
                                   ('unknown', 'nonexplicit_or_truncated_completion_stop')]:
                case = kind + '_' + mode
                bad = e.NoRetryLedger(temp / case, solver_contract_sha256='offline_mock', limit=limit)
                fake = FakeBackend(mode)
                with e.dated_context(dialogues):
                    expect_error(lambda: e.solver.evaluate_candidate(candidate=row, expanded=dialogues[row['root_id']],
                        gate_row=scoped(row, g), tokenizer=tokenizer, backend=fake,
                        ledger=bad, solver_contract_sha256='offline_mock'), expected)
                assert fake.count == bad.used == 1
                assert bad.summary()['retried_generation_count'] == 0
                failures.append({'case': case, 'attempts': 1, 'status': 'PASS'})
        budget = e.NoRetryLedger(temp / 'budget', solver_contract_sha256='offline_mock', limit=0)
        expect_error(lambda: budget.invoke(contract=request_contract, prompt=prompt, call=fail),
                     'local_generation_budget_exhausted')
        assert calls == [1] and budget.used == 0
    e.verify_records(protocol['sources'])
    report = {'status': 'PASS', 'candidate_count': len(rows),
        'full_and_no_context_prompt_checks': checked_prompts,
        'mock_responses_scored': limit, 'real_model_calls': 0, 'external_api_calls': 0,
        'all_correct_and_incorrect_answers_checked': True, 'display_gold_rebinding_checked': True,
        'paired_seed_policy_checked': True, 'failure_not_retried': True,
        'shared_loop_failure_cases': failures, 'budget_rejection_checked': True,
        'source_files_unchanged': True, 'protocol_sha256': e.sha256_file(run / 'protocol.json')}
    e.write_once_json(run.parent / 'OFFLINE_CHECKS.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=e.RUN)
    main(parser.parse_args().run.resolve())
