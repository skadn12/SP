"""Actual Gate tasks through final messages, JSON parsing and unchanged rules."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest

from . import resume, run
from .fixtures import FixtureBackend
from .prompts import render_gate
from .test_task_selection_entry import ActualPatchHarness, select_args
from .test_readiness import Response
from .test_root_entry import RootCliHarness
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1 import contracts, prompts
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.gate_output_contract import (
    FULL_PROMPT_VERSION, FULL_CONTRACT_VERSION,
)

ACTUAL = Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round1_r03_same_task_selection_resume_v1_live_1')


def read(path):
    return json.loads(Path(path).read_text())


def prompt_parts(messages):
    text = messages[0]['content']
    decoder = json.JSONDecoder()
    context = decoder.raw_decode(text.split('\nGATE INPUT\n', 1)[1].lstrip())[0]
    examples = [decoder.raw_decode(text.split(marker, 1)[1].lstrip())[0] for marker in (
        'OUTPUT ONLY THIS JSON OBJECT, WITH NO EXTRA KEYS',
        'ALTERNATIVE REALIZED/REJECT SHAPE (NOT A FACT CLAIM)')]
    return context, examples


def frozen_module(name):
    relative = 'scripts/current_method/restart_n5_spell/simple_append_memorax_v1_memory_event_continuation_probe_v1/'
    qualified = contracts.__package__ + '._frozen_gate_' + name
    spec = importlib.util.spec_from_file_location(qualified, ACTUAL / 'method_snapshot' / relative / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PromptDrivenGateHarness(ActualPatchHarness):
    """Private formal CLI/HTTP boundary; response shapes come from actual prompts."""
    def __enter__(self):
        super().__enter__()
        self.bundle = read(ACTUAL / 'execution/input_freeze.json')
        self.bundle['identity']['branch_id'] = 'MOCK-GATE-CONTRACT-ALIGNMENT-NOT-LIVE'
        self.budget = read(ACTUAL / 'execution/execution_budget.json')
        self.model = read(ACTUAL / 'execution/model_config.json')
        self.fixture = FixtureBackend(self.bundle)
        self.pin_fixture_approval()
        return self

    def http_open(self, request, **kwargs):
        actual = self.requests[-1]
        if actual['role'] != 'gate':
            if actual['role'] == 'supplement' and self.case == 'independent_only':
                return RootCliHarness.http_open(self, request, **kwargs)
            return super().http_open(request, **kwargs)
        payload = json.loads(request.data)
        self.http_payloads.append(payload)
        assert payload['messages'] == actual['messages']
        assert 'response_format' not in payload and 'json_schema' not in payload
        context, examples = prompt_parts(payload['messages'])
        identity = actual['gate_contract_identity']
        assert identity['input_mode'] == 'authoritative_full_history'
        assert context['memory_only_fast'] is False
        original = self.bundle['parent']['dialogue']['turns']
        history = context['authoritative_complete_history']
        assert history[:len(original)] == original
        assert all(turn['turn_id'].startswith('R03:') and 'OFFLINE MOCK' in turn['text']
                   for turn in history[len(original):])
        assert identity['examples'] == examples
        raw = deepcopy(examples[1] if self.case == 'reject_all_from_example' else examples[0])
        # IDs/refs and every field originate in the final example, not a parallel mock schema.
        if self.case != 'reject_all_from_example':
            for row in raw['candidate_reviews']:
                row['evidence_turn_refs'] = [context['authoritative_complete_history'][0]['turn_id']]
        self.prompt_derived_results.append(deepcopy(raw))
        return Response(json.dumps({'choices': [{'finish_reason': 'stop', 'message': {
            'content': json.dumps(raw)}}]}).encode())


def exercise_formal_gate_chain(case):
    with PromptDrivenGateHarness() as harness:
        harness.prompt_derived_results = []
        manifest, args = select_args(harness)
        before = Path(manifest['ledger_path']).read_bytes()
        harness.case = case
        result = resume.main(args)
        assert Path(manifest['ledger_path']).read_bytes().startswith(before)
        requests = deepcopy(harness.requests[1:])
        assert all(row['role'] not in {'planner', 'supplement'} for row in requests)
        files = {str(path.relative_to(harness.root / 'MOCK_resume')): read(path)
                 for path in (harness.root / 'MOCK_resume/checks').glob('*.json')}
        return {'notice': 'OFFLINE MOCK ONLY; private grant/ledger; not reusable execution state',
                'case': case, 'requests': requests, 'parsed_gate_responses': harness.prompt_derived_results,
                'report_MOCK_ONLY': result, 'checks': files, 'real_calls': 0,
                'temporary_supplement_setup_mock_count': 1}


def exercise_independent_development_fixture():
    with PromptDrivenGateHarness() as harness:
        harness.prompt_derived_results = []
        result = harness.invoke(case='independent_only')
        return {'notice': 'ISOLATED SYNTHETIC independent-development patch, not actual selected queue',
                'requests': deepcopy(harness.requests[1:]), 'report_MOCK_ONLY': result,
                'real_calls': 0}


class GateContractAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = read(ACTUAL / 'execution/input_freeze.json')
        cls.queue = read(ACTUAL / 'selected_execution_plan.json')
        cls.parent = cls.bundle['parent']
        cls.old_contracts = frozen_module('contracts')
        cls.old_prompts = frozen_module('prompts')

    def render(self, entry):
        return render_gate(parent=self.parent, gate_scene=entry['gate_scene'], candidates=self.bundle['candidates'])

    def check(self, value, entry, old=False):
        validator = self.old_contracts if old else contracts
        return validator.validate_semantic_gate_output(value, plan={'scenes': [entry['gate_scene']]},
            parent=self.parent, memory_only_fast=False, minimum_pass_candidates=1)

    def test_all_twelve_actual_entries_have_matching_examples_and_schema(self):
        for entry in self.queue:
            with self.subTest(phase=entry['scene']['phase_id']):
                built = self.render(entry)
                context, examples = prompt_parts(built['messages'])
                self.assertEqual(context['authoritative_complete_history'], self.parent['dialogue']['turns'])
                self.assertEqual(built['gate_contract_identity']['examples'], examples)
                self.assertEqual(built['gate_contract_identity']['output_contract_version'], FULL_CONTRACT_VERSION)
                schema = built['gate_contract_identity']['structural_schema']['properties']['candidate_reviews']['items']
                for raw in examples:
                    for row in raw['candidate_reviews']:
                        self.assertEqual(set(row), set(schema['required']))
                        self.assertNotIn('supporting_memory_refs', row)
                    result = self.check(raw, entry)
                    self.assertTrue(result['valid'], result)
                    self.assertEqual(result, self.check(raw, entry, old=True))
                self.assertEqual(self.check(examples[1], entry)['overall_decision'], 'reject')
                instructions = built['prompt'].split('\nGATE INPUT\n', 1)[0]
                self.assertNotIn('supporting_memory_refs', instructions)
                self.assertIn('reason is optional', instructions)
                self.assertIn('empty evidence list is allowed', instructions)
                self.assertIn('ONGOING', instructions)

    def test_old_eight_failures_unchanged_and_alias_only_candidate_views(self):
        for path in sorted((ACTUAL / 'execution/requests').glob('*.json')):
            raw = json.loads(read(ACTUAL / 'execution/responses' / path.name))
            ref = raw['candidate_reviews'][0]['candidate_ref']
            entry = next(e for e in self.queue if e['scene']['phase_id'] == 'initial:' + ref)
            current, old = self.check(raw, entry), self.check(raw, entry, old=True)
            self.assertEqual(current, old)
            self.assertEqual(current['errors'], ['gate_review_1_shape', 'gate_review_1_evidence_shape'])
            isolated = deepcopy(raw)
            isolated['candidate_reviews'][0]['evidence_turn_refs'] = isolated['candidate_reviews'][0].pop('supporting_memory_refs')
            self.assertTrue(self.check(isolated, entry)['valid'])
            self.assertEqual(self.check(isolated, entry), self.check(isolated, entry, old=True))

    def test_nonfast_negative_refs_shapes_and_decisions_match_frozen_validator(self):
        entry = self.queue[0]
        examples = prompt_parts(self.render(entry)['messages'])[1]
        base = examples[0]
        variants = []
        def changed(**fields):
            value = deepcopy(base)
            value['candidate_reviews'][0].update(fields)
            return value
        missing = deepcopy(base)
        del missing['candidate_reviews'][0]['evidence_turn_refs']
        variants.append(missing)
        for invalid in ('GM:E001:C001', 'NO_SUCH_TURN', 'cc-train-episode-OTHER:T001', 'R03:S01:T01'):
            variants.append(changed(evidence_turn_refs=[invalid]))
        variants += [changed(supporting_memory_refs=['GM:E001:C001']), changed(decision='fail'),
                     changed(parent_relation='NOT_REALIZED'), changed(duplicate_with_parent=True),
                     changed(candidate_ref='ND09:C01'), changed(evidence_turn_refs=None),
                     changed(evidence_turn_refs=[self.parent['dialogue']['turns'][0]['turn_id']] * 2)]
        missing_required = deepcopy(base)
        del missing_required['candidate_reviews'][0]['decision']
        variants.append(missing_required)
        realized_without_refs = deepcopy(examples[1])
        realized_without_refs['candidate_reviews'][0]['evidence_turn_refs'] = []
        variants.append(realized_without_refs)
        for raw in variants:
            with self.subTest(raw=raw):
                result = self.check(raw, entry)
                self.assertFalse(result['valid'])
                self.assertEqual(result, self.check(raw, entry, old=True))

    def test_foreign_reference_alias_collision_is_not_provenance_authentication(self):
        entry = self.queue[0]
        raw = prompt_parts(self.render(entry)['messages'])[1][0]
        # The schema contains local refs, not root/snapshot-qualified output IDs.
        raw['candidate_reviews'][0]['evidence_turn_refs'] = [self.parent['dialogue']['turns'][0]['turn_id']]
        self.assertTrue(self.check(raw, entry)['valid'])
        self.assertEqual(self.check(raw, entry), self.check(raw, entry, old=True))
        # Same-spelled IDs from other snapshots are indistinguishable here;
        # provenance must be pinned by request identity, not invented in validator.

    def test_memory_only_old_prompt_bytes_and_rules_remain_unchanged(self):
        entry = self.queue[0]
        memory = {'buckets': {'events': [{'memory_facts': [{'memory_ref': 'GM:E001:C001', 'text': 'MOCK fact'}]}]}}
        context = contracts.build_semantic_gate_context(parent=self.parent, candidates=self.bundle['candidates'],
            plan={'scenes': [entry['gate_scene']]}, memory_only_fast=True,
            global_memory_state_view=memory, minimum_pass_candidates=1)
        self.assertEqual(prompts.build_semantic_continuation_gate_prompt(gate_context=context),
                         self.old_prompts.build_semantic_continuation_gate_prompt(gate_context=context))
        raw = {'candidate_reviews': [{'candidate_ref': context['expected_candidate_refs'][0],
            'decision': 'reject', 'parent_relation': 'REALIZED', 'duplicate_with_parent': True,
            'supporting_memory_refs': ['GM:E001:C001'], 'reason': 'MOCK shape check only'}]}
        options = {'plan': {'scenes': [entry['gate_scene']]}, 'parent': self.parent, 'memory_only_fast': True,
                   'global_memory_state_view': memory, 'minimum_pass_candidates': 1}
        self.assertTrue(contracts.validate_semantic_gate_output(raw, **options)['valid'])
        self.assertEqual(contracts.validate_semantic_gate_output(raw, **options),
                         self.old_contracts.validate_semantic_gate_output(raw, **options))
        with self.assertRaises(ValueError):
            prompts.build_semantic_continuation_gate_prompt(gate_context=context, prompt_version=FULL_PROMPT_VERSION)

    def test_legacy_full_prompt_is_still_explicitly_reproducible(self):
        context = prompt_parts(self.render(self.queue[0])['messages'])[0]
        self.assertEqual(prompts.build_semantic_continuation_gate_prompt(gate_context=context),
                         self.old_prompts.build_semantic_continuation_gate_prompt(gate_context=context))

    def test_formal_saved_patch_workflow_covers_actual_eight_gate_tasks(self):
        result = exercise_formal_gate_chain('reject_all_from_example')
        gates = [r for r in result['requests'] if r['role'] == 'gate']
        self.assertEqual(len(gates), 8)
        refs = [prompt_parts(r['messages'])[0]['expected_candidate_refs'][0] for r in gates]
        self.assertEqual(refs, ['ME08:C01', 'ME07:C01', 'ME03:C01', 'ND01:C01',
                                'ND02:C01', 'ND03:C01', 'ND04:C01', 'ND05:C01'])
        self.assertEqual(result['report_MOCK_ONLY']['attempted_request_counts']['realizer'], 0)
        for check in result['checks'].values():
            if 'overall_decision' in check:
                self.assertTrue(check['valid'])
                self.assertEqual(check['overall_decision'], 'reject')

    def test_formal_pass_example_parsing_reaches_mock_realizer_and_adherence(self):
        result = exercise_formal_gate_chain('pass_from_example')
        self.assertEqual(result['report_MOCK_ONLY']['status'], 'EXECUTION_ENDED_GOALS_RECORDED_SEPARATELY')
        roles = [row['role'] for row in result['requests']]
        self.assertIn('realizer', roles)
        self.assertIn('adherence', roles)
        for request in result['requests']:
            self.assertNotIn('ND09', json.dumps(request['messages']))
            if request['role'] == 'realizer':
                self.assertEqual(request['prompt_version'], 'incremental_retention_v1.realizer.successful_prompt_alignment_v3.text_free')
                self.assertNotIn('authoritative_history', request['messages'][0]['content'])
        self.assertEqual(result['report_MOCK_ONLY']['experiment_total_attempted_counts']['supplement'], 1)

    def test_independent_new_development_uses_same_contract_through_root_workflow(self):
        result = exercise_independent_development_fixture()
        gates = [r for r in result['requests'] if r['role'] == 'gate']
        self.assertTrue(any(prompt_parts(r['messages'])[0]['expected_candidate_refs'] == ['ND09:C01'] for r in gates))
        self.assertTrue(all(r['gate_contract_identity']['output_contract_version'] == FULL_CONTRACT_VERSION for r in gates))
        self.assertGreater(result['report_MOCK_ONLY']['retained_scene_count'], 0)


if __name__ == '__main__':
    with run.offline_network_guard():
        unittest.main()
