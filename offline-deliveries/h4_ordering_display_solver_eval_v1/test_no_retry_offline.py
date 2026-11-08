"""Offline safety checks. Fake responses only: never construct a model backend."""
from __future__ import annotations
import copy
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import socket
import sys
import types

sys.dont_write_bytecode = True
WORKTREE = Path('/root/autodl-tmp/gxy-selfplay/worktrees/h4-ordering-display-v2')
sys.path.insert(0, str(WORKTREE))
from scripts.current_method.restart_n5_spell.h4_ordering_display_v2.evaluate import NoRetryLedger, dated_context, LIMIT, MODELS
from scripts.current_method.restart_n5_spell.h4_round0_b0_k3_difficulty_v1 import solver
from scripts.current_method.restart_n5_spell.h4_round0_b0_k3_difficulty_v1.request_ledger import request_cache_key
from scripts.current_method.restart_n5_spell.backends import BackendResponse
from scripts.current_method.restart_n5_spell.core import atomic_json, sha256_bytes

socket.create_connection = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('MOCK_ONLY_NETWORK_FORBIDDEN'))
HERE = Path(__file__).resolve().parent
CASES = HERE / 'offline_mock_cases'
if CASES.exists():
    raise RuntimeError('mock_cases_exist_do_not_overwrite')
CASES.mkdir()
CHECKS = []
READ = lambda p: json.loads(Path(p).read_text())
SHA = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
CONTRACT_SHA = 'f' * 64
PROMPT = 'OFFLINE MOCK PROMPT'


def contract(n):
    return {'solver_contract_sha256': CONTRACT_SHA, 'root_id': 'MOCK_ROOT', 'case': n,
            'prompt_sha256': sha256_bytes(PROMPT.encode())}


def ledger(name, limit=120):
    return NoRetryLedger(CASES / name, solver_contract_sha256=CONTRACT_SHA, limit=limit)


def expect_error(label, thunk, expected):
    try:
        thunk()
    except BaseException as error:
        assert expected in str(error), (label, type(error).__name__, str(error))
        return str(error)
    raise AssertionError('expected exception: ' + label)


class CounterCall:
    def __init__(self, finish='stop', error=None, raw='<box>I1,I2,I3,I4</box>'):
        self.count = 0
        self.finish, self.error, self.raw = finish, error, raw
    def __call__(self):
        self.count += 1
        if self.error is not None:
            raise self.error
        meta = {} if self.finish is None else {'finish_reason': self.finish}
        return BackendResponse(self.raw, 'OFFLINE_FAKE_MODEL', meta)


failed = ledger('technical_failure')
call = CounterCall(error=RuntimeError('mock transport failure'))
c = contract('technical_failure')
expect_error('technical', lambda: failed.invoke(contract=c, prompt=PROMPT, call=call), 'technical_failure_stop_without_retry')
assert call.count == failed.used == 1
assert failed.summary()['failed_request_count'] == 1
expect_error('repeat failed', lambda: failed.invoke(contract=c, prompt=PROMPT, call=call), 'noncompleted_request_cannot_be_resent')
assert call.count == 1 and failed.summary()['retried_generation_count'] == 0
CHECKS.append({'name': 'technical_failure_single_attempt_and_no_resend', 'status': 'PASS', 'attempts': 1})

for status in [None, 'length', 'unknown', 'abort']:
    name = 'finish_' + str(status)
    l = ledger(name); cb = CounterCall(finish=status); c = contract(name)
    expect_error(name, lambda: l.invoke(contract=c, prompt=PROMPT, call=cb), 'nonexplicit_or_truncated_completion_stop')
    assert cb.count == l.used == 1
    assert l.summary()['completed_request_count'] == 1
    expect_error(name + ' cache', lambda: l.invoke(contract=c, prompt=PROMPT, call=cb), 'nonexplicit_or_truncated_completion_stop')
    assert cb.count == 1
    CHECKS.append({'name': name, 'status': 'PASS', 'attempts': 1,
                   'raw_transport_state': 'completed', 'acceptance': 'STOPPED_NOT_SCORED', 'cache_did_not_regenerate': True})

for status in ['reserved', 'interrupted', 'retryable', 'failed']:
    l = ledger('blocked_' + status); c = contract(status); key = request_cache_key(c)
    atomic_json(l._state_path('MOCK_ROOT', key), {'status': status, 'attempts': [{'status': status, 'attempt_id': 1}]})
    cb = CounterCall()
    expect_error(status, lambda: l.invoke(contract=c, prompt=PROMPT, call=cb), 'noncompleted_request_cannot_be_resent')
    assert cb.count == 0
    CHECKS.append({'name': 'existing_' + status + '_cannot_resend', 'status': 'PASS', 'new_attempts': 0})

budget = ledger('budget_and_cache', limit=2); cb = CounterCall()
for n in [1, 2]:
    response, reused = budget.invoke(contract=contract(n), prompt=PROMPT, call=cb)
    assert not reused
assert cb.count == budget.used == 2
expect_error('budget', lambda: budget.invoke(contract=contract(3), prompt=PROMPT, call=cb), 'local_generation_budget_exhausted')
response, reused = budget.invoke(contract=contract(1), prompt=PROMPT, call=cb)
assert reused and response.metadata['reused'] is True and cb.count == budget.used == 2
reloaded = NoRetryLedger(CASES / 'budget_and_cache', solver_contract_sha256=CONTRACT_SHA, limit=2)
assert reloaded.used == 2
expect_error('reloaded budget', lambda: reloaded.invoke(contract=contract(3), prompt=PROMPT, call=cb), 'local_generation_budget_exhausted')
CHECKS.append({'name': 'budget_counts_attempts_and_cache_reuses_without_generation', 'status': 'PASS', 'limit': 2, 'calls': 2, 'reloaded_used': 2})

invalid = ledger('parse_stop'); cb = CounterCall(raw='malformed'); c = contract('parse_stop')
invalid.invoke(contract=c, prompt=PROMPT, call=cb)
decision = {'scored': False, 'correct': None, 'failure_type': 'parse_failure'}
expect_error('invalid parse', lambda: invalid.record_decision('MOCK_ROOT', request_cache_key(c), decision), 'unscorable_output_stop_without_retry')
assert invalid.read('MOCK_ROOT', request_cache_key(c))['decision'] == decision
CHECKS.append({'name': 'unscorable_decision_persisted_then_stops', 'status': 'PASS', 'calls': 1})

# Exercise the real shared evaluate_candidate loop with frozen prompt strings.
PREPARED = Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_ordering_display_v2/prepared')
candidate = next(c for c in READ(PREPARED / 'candidate_manifest.json') if c['question_type'] == 'ordering')
gate_row = next(c for c in READ(PREPARED / 'program_gate_results.json') if c['candidate_id'] == candidate['candidate_id'])
gate_row = {**gate_row, 'candidate_pool_sha256': SHA(PREPARED / 'candidate_manifest.json'),
            'dialogue_sha256': SHA(PREPARED / 'dialogues' / f"{candidate['root_id']}.json")}
expanded = READ(PREPARED / 'dialogues' / f"{candidate['root_id']}.json")
prompt_tokens = {p['sha256']: p['input_tokens'] for p in READ(PREPARED / 'prompt_manifest.json')}


class FrozenTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {'tokenize': False, 'add_generation_prompt': True, 'enable_thinking': False}
        return ''.join('<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' for m in messages) + '<|im_start|>assistant\n<think>\n\n</think>\n\n'
    def encode(self, text, **kwargs):
        return range(prompt_tokens[sha256_bytes(text.encode())])


class FakeStructuredBackend:
    model_identifier = 'OFFLINE_FAKE_MODEL'
    def __init__(self, mode='correct'):
        self.mode, self.calls = mode, []
        variants = solver.position_variants(candidate, ordering_display_policy='ordering_relabel_balanced_v2')
        self.answers = {v['variant_id']: v['displayed_gold'] for v in variants}
    def generate_structured(self, request, *, question_type):
        self.calls.append(request)
        assert question_type == 'ordering'
        for forbidden in ['canonical_gold', 'display_to_canonical', 'ordered_pair', 'displayed_gold']:
            assert forbidden not in request.prompt
        if self.mode == 'failure':
            raise RuntimeError('mock first failure')
        if self.mode == 'parse':
            return BackendResponse('MALFORMED', self.model_identifier, {'finish_reason': 'stop'})
        finish = {'unknown': None, 'truncated': 'length'}.get(self.mode, 'stop')
        text = '<box>' + ','.join(self.answers[request.order_id]) + '</box>'
        return BackendResponse(text, self.model_identifier, {'finish_reason': finish,
            'prompt_token_count': len(FrozenTokenizer().encode(request.prompt)), 'completion_token_count': 11})


old_detect = solver.detect_structured_output_api
solver.detect_structured_output_api = lambda: ('structured_outputs=mock', 'OFFLINE_MOCK')
try:
    for mode, expected in [('failure', 'technical_failure_stop_without_retry'),
                           ('unknown', 'nonexplicit_or_truncated_completion_stop'),
                           ('truncated', 'nonexplicit_or_truncated_completion_stop'),
                           ('parse', 'unscorable_output_stop_without_retry')]:
        l = ledger('shared_loop_' + mode); backend = FakeStructuredBackend(mode)
        with dated_context({candidate['root_id']: expanded}):
            expect_error(mode, lambda: solver.evaluate_candidate(candidate=candidate, expanded=expanded,
                gate_row=gate_row, tokenizer=FrozenTokenizer(), backend=backend, ledger=l,
                solver_contract_sha256=CONTRACT_SHA), expected)
        assert len(backend.calls) == l.used == 1
        assert l.summary()['retried_generation_count'] == 0
        CHECKS.append({'name': 'shared_evaluate_candidate_' + mode + '_stops_after_first_call', 'status': 'PASS', 'calls': 1})
    l = ledger('shared_loop_complete', limit=24); backend = FakeStructuredBackend()
    with dated_context({candidate['root_id']: expanded}):
        result = solver.evaluate_candidate(candidate=candidate, expanded=expanded, gate_row=gate_row,
            tokenizer=FrozenTokenizer(), backend=backend, ledger=l, solver_contract_sha256=CONTRACT_SHA)
        reused = solver.evaluate_candidate(candidate=candidate, expanded=expanded, gate_row=gate_row,
            tokenizer=FrozenTokenizer(), backend=backend, ledger=l, solver_contract_sha256=CONTRACT_SHA)
    assert len(backend.calls) == l.used == 24
    assert result == reused and len(result['requests']) == 24
    assert all(q['correct'] and q['canonical_prediction'] == candidate['gold'] for q in result['requests'])
    CHECKS.append({'name': 'shared_loop_24_requests_ordered_rebinding_and_cache_no_extra_generation', 'status': 'PASS', 'calls': 24})

    # Exercise actual structured transport with a fake vllm module and engine.
    # It must receive all 24 legal orders, never a Gold-selected answer list.
    class SamplingParams:
        def __init__(self, **kw): self.kw = kw
    class StructParams:
        def __init__(self, **kw): self.kw = kw
    fake_vllm = types.SimpleNamespace(SamplingParams=SamplingParams,
        sampling_params=types.SimpleNamespace(StructuredOutputsParams=StructParams))
    previous_vllm = sys.modules.get('vllm')
    sys.modules['vllm'] = fake_vllm
    class FakeEngine:
        def __init__(self): self.requests=[]
        def generate(self, prompts, params, **kw):
            self.requests.append((prompts, params.kw))
            return [types.SimpleNamespace(prompt_token_ids=[1,2], outputs=[types.SimpleNamespace(text='<box>I1,I2,I3,I4</box>',token_ids=[1],finish_reason='stop')])]
    transport_backend = types.SimpleNamespace(engine=FakeEngine(), model_identifier='OFFLINE_FAKE_MODEL')
    req = backend.calls[0]
    solver._structured_generate(transport_backend, req, question_type='ordering')
    sent_prompts, params = transport_backend.engine.requests[0]
    choices = params['structured_outputs'].kw['choice']
    expected = {'<box>' + ','.join(order) + '</box>' for order in itertools.permutations(req.options)}
    assert set(choices) == expected and len(choices) == 24 and sent_prompts == [req.prompt]
    assert 'gold' not in params and 'canonical_answer_text' not in params
    CHECKS.append({'name': 'actual_structured_transport_passes_only_prompt_and_all_24_allowed_orders', 'status': 'PASS', 'fake_engine_calls': 1})
    if previous_vllm is None: del sys.modules['vllm']
    else: sys.modules['vllm'] = previous_vllm
finally:
    solver.detect_structured_output_api = old_detect

assert LIMIT == 120 and MODELS == ('M0', 'M_clean_policy')
PROD = WORKTREE / 'scripts/current_method/restart_n5_spell/h4_ordering_display_v2/evaluate.py'
report = {'status': 'PASS', 'kind': 'OFFLINE_FAKE_RESPONSE_CHECK_NOT_MODEL_EVALUATION',
          'checks': CHECKS, 'count': len(CHECKS), 'external_calls': 0,
          'gpu_model_inference_calls': 0, 'actual_model_generations': 0,
          'new_entry_source': {'path': str(PROD), 'sha256': SHA(PROD)},
          'authorized_live_scope_checked': {'models': list(MODELS), 'per_model_limit': LIMIT, 'total_limit': LIMIT*len(MODELS)},
          'technical_finish_state_note': 'Unknown/truncated responses have a saved transport-completed envelope but are rejected before scoring; the outer live catch writes STOPPED and STARTED prevents resume. This does not restore their attempt budget.'}
(HERE / 'OFFLINE_LEDGER_REVIEW.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status': report['status'], 'checks': len(CHECKS), 'external_calls': 0, 'gpu_calls': 0},ensure_ascii=False))
