"""Synthetic-only mixed Writer and downstream demonstrations; no real clients."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ..core import atomic_json, atomic_text
from ..h4_qa_multiroot_k3_t08_probe_v1 import generation, solver_adapter
from ..h4_qa_multiroot_k3_t08_probe_v1.permutation import position_variants
from .contract import VERSION, ordering_schema
from .dependencies import forbid_network
from .guidance import synthetic_example


class FakeClient:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []
        self.config = SimpleNamespace(model='Qwen/Qwen3-14B', base_url='https://api.siliconflow.cn/v1')

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        raw = next(self.replies)
        return SimpleNamespace(raw_text=raw if isinstance(raw, str) else json.dumps(raw), metadata={})


class FakeLedger:
    def invoke(self, **kwargs):
        return kwargs['call']()


class PlainTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return '\n'.join(message['content'] for message in messages)


def fixture(n=4, source='contract_guided'):
    bundle = synthetic_example(n)
    refs = [row['turn_ref'] for row in bundle['expanded']['turns']]
    contract = {'intent_id': 'intent_1', 'intent_type': 'temporal_evolution',
                'discriminative_variable': 'Example chronology',
                'required_evidence_groups': {'before': refs[:1], 'trigger': refs[1:2], 'after': refs[2:]}}
    counts = (2, 1, 1) if source == 'contract_guided' else (4, 2, 2)
    rows = []
    for kind, count in zip(('single_choice', 'multi_select'), counts[:2]):
        for index in range(count):
            row = {'question_type': kind, 'question': f'Synthetic {kind} {index}: which stated event occurred?',
                   'options': {key: f'Example distinct option {key}' for key in ('ABCD' if kind == 'single_choice' else 'ABCDE')},
                   'evidence_refs': refs, 'evidence_groups': {'event_evidence': refs},
                   'discriminative_variable': 'Example event facts',
                   'reasoning_groups_used': ['event_evidence'], 'hypothesized_reasoning_type': 'temporal',
                   'intent_reference': 'intent_1'}
            row.update({'correct_option': 'A'} if kind == 'single_choice' else {'gold': ['A', 'B']})
            rows.append(row)
    for index in range(counts[2]):
        row = copy.deepcopy(bundle['raw']['candidates'][0])
        row['intent_reference'] = 'intent_1'
        row['question'] += f' (Synthetic example {index})'
        rows.append(row)
    return bundle['expanded'], contract, {'candidates': rows}


def exercise(run):
    run = Path(run)
    outcomes = []
    with forbid_network(), patch.object(generation, 'OpenAICompatibleChallengerClient',
            side_effect=AssertionError('REAL_CLIENT_CONSTRUCTION_FORBIDDEN')):
        for n in (4, 5, 6):
            atomic_json(run / 'policy_snapshot/schema' / f'ordering_{n}.json', ordering_schema(n))
            atomic_json(run / 'policy_snapshot/examples' / f'ordering_{n}.json', synthetic_example(n))
            for source in ('contract_guided', 'free_exploration'):
                expanded, contract, raw = fixture(n, source)
                # Force the actual replacement path with an underconstrained primary graph.
                broken = copy.deepcopy(raw)
                for row in broken['candidates']:
                    if row['question_type'] == 'ordering':
                        row['temporal_relations'] = []
                root = run / 'audit/offline_examples' / source / f'n{n}'
                policy = {'version': VERSION, 'node_count': n}
                context = {'run_id': 'SYNTHETIC_OFFLINE_ONLY', 'arm_id': 'FIXTURE', 'root_id': 'fixture-root'}
                client = FakeClient([broken, raw])
                result = generation.execute_writer_batch(expanded=expanded, contract=contract,
                    root_dir=root, root_id='fixture-root', ledger=FakeLedger(), client=client,
                    source=source, ordering_policy=policy, ordering_context=context)
                assert len(client.calls) == 2
                expected = 4 if source == 'contract_guided' else 8
                assert len(result['candidates']) == expected
                assert result['generation_records'][0]['validation']['minimum_acceptance']['missing_minimums'] == {
                    'ordering': 1}
                assert client.calls[1]['prompt'].endswith(json.dumps(broken, ensure_ascii=False))
                summary = generation.finalize_writer_candidates(root_dir=root, root_id='fixture-root',
                    candidates=result['candidates'], generation_records=result['generation_records'],
                    expanded=expanded, ordering_policy=policy, ordering_context=context)
                assert summary['valid_candidate_count'] == expected and summary['gold_rebinding_errors'] == 0
                views = []
                for candidate in result['candidates']:
                    if candidate['question_type'] != 'ordering':
                        continue
                    for variant in position_variants(candidate):
                        prompt = solver_adapter._structured_prompt(PlainTokenizer(), candidate, variant,
                            solver_adapter._dialogue_rows(expanded))
                        # Renderer output type is kept by the existing Solver API.
                        views.append({'candidate_id': candidate['candidate_id'], 'variant': variant['variant'],
                                      'prompt': prompt})
                atomic_json(root / 'solver_views.json', views)
                atomic_json(root / 'expanded_dialogue_clean.json', expanded)
                atomic_json(root / 'raw_fixture.json', raw)
                atomic_json(root / 'result.json', result)
                outcomes.append({'source': source, 'node_count': n, 'status': 'PASS',
                    'synthetic_primary': 1, 'synthetic_repair': 1,
                    'target': dict(zip(('single_choice', 'multi_select', 'ordering'),
                                       (2, 1, 1) if source == 'contract_guided' else (4, 2, 2))),
                    'minima': generation.MINIMUM_COUNTS, 'final_candidates': expected,
                    'path': str(root), 'external_calls': 0})
    from .dependencies import loaded_repository_files
    report = {'status': 'PASS', 'synthetic_only': True, 'model_success_claim': False,
              'repository_modules': sorted(map(str, loaded_repository_files())),
              'external_calls': 0, 'scenarios': outcomes}
    atomic_json(run / 'audit/offline_integration_report.json', report)
    return report
