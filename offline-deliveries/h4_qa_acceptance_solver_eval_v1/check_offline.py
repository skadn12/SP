"""No-model verification of date adaptation, scoring, ledger and frozen inputs."""
import json
from pathlib import Path
import tempfile

import evaluate as e
from transformers import AutoTokenizer
from scripts.current_method.restart_n5_spell.backends import BackendResponse


def main():
    pre = e.load_preflight()
    tokenizer = AutoTokenizer.from_pretrained(str(e.BASE_MODEL), local_files_only=True)
    rows = e.load_jsonl(e.RUN / 'candidate_manifest.jsonl')
    gates = {r['candidate_id']: r for r in e.load_jsonl(e.RUN / 'program_gate_results.jsonl')}
    dialogues = {r: e.load_json(e.RUN / 'dialogues' / (r + '.json')) for r in e.ROOTS}
    e.install_date_adapter(dialogues)
    original = []
    for root in e.ROOTS:
        original.extend(e.load_json(e.SOURCE / 'final' / root / 'new_vs_previous_formal.json'))
    assert original == rows
    checked_prompts = 0
    for row in rows:
        dialogue = e.contract.dialogue_rows(dialogues[row['root_id']])
        for variant in e.contract.position_variants(row):
            no = e.dated_build_prompt(tokenizer, row, variant, dialogue, context_mode='no_context')
            assert no == e.ORIGINAL_BUILD_PROMPT(tokenizer, row, variant, dialogue, context_mode='no_context')
            full = e.dated_build_prompt(tokenizer, row, variant, dialogue, context_mode='full_context')
            assert e.render_dialogue(dialogues[row['root_id']]) in full
            assert '[Conversation date: UNKNOWN | Session: R02:S01]' in full
            assert row['question'] in full and row['question'] in no
            checked_prompts += 2

    class FakeBackend:
        model_identifier = 'OFFLINE_FAKE_SOLVER_NO_MODEL'
        tokenizer_path = e.BASE_MODEL
        max_model_len = e.MAX_MODEL_LEN
        count = 0

        def generate(self, request):
            self.count += 1
            answer = '<box>' + request.canonical_answer_text + '</box>'
            return BackendResponse(answer, self.model_identifier, {
                'finish_reason': 'stop',
                'prompt_token_count': len(tokenizer.encode(request.prompt, add_special_tokens=False)),
                'completion_token_count': len(tokenizer.encode(answer, add_special_tokens=False)),
            })

        def generate_structured(self, request, *, question_type):
            return self.generate(request)

    with tempfile.TemporaryDirectory(prefix='h4_solver_offline_') as td:
        ledger = e.NoRetryLedger(Path(td) / 'good', solver_contract_sha256='mock_contract', limit=234)
        backend = FakeBackend()
        for row in rows:
            result = e.solver.evaluate_candidate(candidate=row, expanded=dialogues[row['root_id']],
                gate_row=gates[row['candidate_id']], tokenizer=tokenizer, backend=backend,
                ledger=ledger, solver_contract_sha256='mock_contract')
            assert all(r['correct'] and r['scored'] for r in result['requests'])
            summary = e.summarize_candidate_result(result)
            assert summary['full_context_accuracy'] == summary['no_context_accuracy'] == 1.0
        assert backend.count == 234
        assert ledger.summary()['retried_generation_count'] == 0
        state = ledger.states()[0]
        contract = state['request_contract']
        prompt = Path(state['prompt_path']).read_text()
        calls = [0]

        def fail():
            calls[0] += 1
            raise ValueError('intentional offline technical failure')

        bad = e.NoRetryLedger(Path(td) / 'bad', solver_contract_sha256='mock_contract', limit=1)
        for _ in range(2):
            try:
                bad.invoke(contract=contract, prompt=prompt, call=fail)
            except RuntimeError:
                pass
            else:
                raise AssertionError('technical_failure_was_not_stopped')
        assert calls == [1]
        assert bad.summary()['actual_solver_generation_count'] == 1
        assert bad.summary()['retried_generation_count'] == 0
    e.verify_records(pre['sources'])
    report = {'status': 'PASS', 'candidates_unchanged': 9,
              'full_and_no_context_prompt_checks': checked_prompts,
              'mock_responses_scored': 234, 'real_model_calls': 0,
              'external_api_calls': 0, 'failure_not_retried': True,
              'source_files_unchanged': True}
    e.write_once_json(e.HERE / 'OFFLINE_CHECKS.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
