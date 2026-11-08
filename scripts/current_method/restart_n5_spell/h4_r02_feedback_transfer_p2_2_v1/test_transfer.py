"""Offline transfer-contract tests; no client or API is imported or called."""
from copy import deepcopy
import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ..h4_qa_mining_probe_v1.intent_miner import build_intent_prompt
from ..h4_qa_mining_probe_v1.intent_contract import canonicalize_intents
from ..h4_qa_mining_probe_v1.reasoning_contract import build_reasoning_contract
from .transfer import (
    TransferHooks, validate_bindings, compile_free_brief, compile_guided_brief,
    revise_miner_prompt, model_cards, model_history, record_semantic_diagnostic,
)

ROOT = 'root-alpha'
EXPANDED = {'turns': [{'turn_ref': f'T{index}', 'speaker': 'Speaker', 'text': f'Actual utterance {index}'} for index in range(1, 11)]}
CARD = {'root_id': ROOT, 'feedback_id': 'feedback-alpha', 'source_candidate_id': ROOT + ':OLD',
    'source_question_type': 'single_choice', 'source_quality_status': 'ACTIVE_GUIDANCE_ELIGIBLE',
    'source_question': 'Historical question', 'options': [{'label': 'A', 'text': 'Historical option'}],
    'historical_gold': 'A', 'solver_observations': {'model': {'by_context': {'full': {'correct_count': 4}}}},
    'provenance': {'secret_path': '/never/model-visible'}}


def fixtures():
    intents = [{'intent_id': f'I{index}', 'intent_type': 'state_transition',
        'reasoning_chain': {'before_state': [1], 'trigger': [2], 'after_state': [3]},
        'discriminative_variable': 'Distinction ' + str(index),
        'candidate_question_types': ['single_choice', 'multi_select', 'ordering']} for index in range(1, 4)]
    bindings = [{'intent_id': f'I{index}', 'status': 'PROPOSED_TRANSFER', 'source_feedback_ids': ['feedback-alpha'],
        'target_distinction': 'Proposed distinction ' + str(index), 'evidence_refs': ['T1', 'T3'],
        'scope': 'Observed scope', 'change_from_source': 'Changed relation'} for index in range(1, 4)]
    raw = {'intents': intents, 'feedback_bindings': bindings}
    result = canonicalize_intents({'intents': intents}, expanded_dialogue=EXPANDED)
    contracts = [build_reasoning_contract(row) for row in result['intents'] if row['valid']]
    return raw, result, contracts


def evaluate(raw, *, cards=None, recanonicalize=True):
    result = canonicalize_intents({'intents': raw.get('intents')} if isinstance(raw, dict) else raw, expanded_dialogue=EXPANDED)
    contracts = [build_reasoning_contract(row) for row in result['intents'] if row['valid']]
    return validate_bindings(raw, result, contracts, expanded=EXPANDED, root_id=ROOT, cards=[CARD] if cards is None else cards)


class BindingTests(unittest.TestCase):
    def test_valid_proposal_exact_refs_and_counts(self):
        raw, result, contracts = fixtures()
        audit = evaluate(raw)
        self.assertEqual(audit['valid_proposed_binding_count'], 3)
        self.assertFalse(audit['envelope_errors'])
        brief = compile_guided_brief(root_id=ROOT, contract=contracts[0], validation=audit, expanded=EXPANDED, cards=[CARD])
        self.assertEqual(brief['current_evidence'], [EXPANDED['turns'][0], EXPANDED['turns'][2]])
        self.assertEqual(brief['observation_cards'][0]['solver_observations'], CARD['solver_observations'])
        self.assertNotIn('provenance', brief['observation_cards'][0])

    def test_explicit_no_supported_transfer_keeps_intent(self):
        raw, result, contracts = fixtures()
        raw['feedback_bindings'][0].update(status='NO_SUPPORTED_TRANSFER', source_feedback_ids=[], evidence_refs=[], target_distinction='', scope='', change_from_source='')
        audit = evaluate(raw)
        self.assertEqual(audit['by_intent']['I1']['fallback_reason'], 'NO_SUPPORTED_TRANSFER')
        self.assertTrue(audit['by_intent']['I1']['mechanical_valid'])
        self.assertEqual(len(contracts), 3)

    def test_empty_active_and_inactive_cross_root_unknown_sources(self):
        raw, _, _ = fixtures()
        for cards in ([], [{**CARD, 'root_id': 'other-root'}], [{**CARD, 'source_quality_status': 'AMBIGUOUS_HISTORY_ONLY'}], [{**CARD, 'source_question_type': 'ordering'}]):
            with self.subTest(cards=cards):
                audit = evaluate(raw, cards=cards)
                self.assertEqual(audit['feedback_status'], 'NO_ACTIVE_FEEDBACK')
                self.assertEqual(audit['by_intent']['I1']['fallback_reason'], 'NO_ACTIVE_FEEDBACK')
                self.assertIn('SOURCE_NOT_ELIGIBLE_FOR_ROOT', audit['bindings'][0]['errors'])
        raw['feedback_bindings'][0]['source_feedback_ids'] = ['unknown']
        self.assertIn('SOURCE_NOT_ELIGIBLE_FOR_ROOT', evaluate(raw)['bindings'][0]['errors'])

    def test_unknown_ref_and_existing_ref_outside_intent(self):
        for ref, expected in [('NO_SUCH_REF', 'UNKNOWN_CURRENT_REF'), ('T10', 'REF_OUTSIDE_INTENT_CHAIN')]:
            raw, _, _ = fixtures()
            raw['feedback_bindings'][0]['evidence_refs'] = [ref]
            errors = evaluate(raw)['bindings'][0]['errors']
            self.assertIn(expected, errors)

    def test_duplicate_binding_invalidates_all_not_first(self):
        raw, _, _ = fixtures()
        raw['feedback_bindings'].append(deepcopy(raw['feedback_bindings'][0]))
        audit = evaluate(raw)
        self.assertEqual(audit['by_intent']['I1']['fallback_reason'], 'BINDING_INVALID')
        self.assertEqual(audit['by_intent']['I1']['binding_indices'], [0, 3])
        self.assertIn('DUPLICATE_BINDING_INTENT_ID', audit['bindings'][0]['errors'])
        self.assertIn('DUPLICATE_BINDING_INTENT_ID', audit['bindings'][3]['errors'])

    def test_unknown_envelope_preserves_valid_original_intents(self):
        raw, result, contracts = fixtures()
        raw['arbitrary_extra'] = True
        audit = evaluate(raw)
        self.assertEqual(len(audit['by_intent']), 3)
        self.assertEqual({x['fallback_reason'] for x in audit['by_intent'].values()}, {'ENVELOPE_INVALID'})
        hooks = TransferHooks('P2_2_CONCRETE_TRANSFER', [CARD], [], 'generic')
        projected = hooks.project_miner_response(raw, expanded=EXPANDED, root_id=ROOT, root_dir=Path('/unused'))
        self.assertEqual(set(projected), {'intents'})
        self.assertTrue(canonicalize_intents(projected, expanded_dialogue=EXPANDED)['valid'])

    def test_missing_and_nonarray_bindings_fallback(self):
        raw, _, _ = fixtures()
        del raw['feedback_bindings']
        self.assertEqual(evaluate(raw)['by_intent']['I1']['fallback_reason'], 'BINDING_MISSING')
        raw['feedback_bindings'] = {}
        audit = evaluate(raw)
        self.assertIn('BINDINGS_NOT_ARRAY', audit['envelope_errors'])
        self.assertEqual(audit['by_intent']['I1']['fallback_reason'], 'BINDING_INVALID')

    def test_raw_json_failure_does_not_guess(self):
        hooks = TransferHooks('P2_2_CONCRETE_TRANSFER', [CARD], [], 'generic')
        self.assertIsNone(hooks.project_miner_response(None, expanded=EXPANDED, root_id=ROOT, root_dir=Path('/unused')))
        audit = evaluate(None)
        self.assertEqual(audit['by_intent'], {})
        self.assertIn('ENVELOPE_NOT_OBJECT_OR_JSON_PARSE_FAILED', audit['envelope_errors'])

    def test_filtered_intent_does_not_shift_later_binding(self):
        raw, _, _ = fixtures()
        raw['intents'][0]['reasoning_chain']['trigger'] = []
        audit = evaluate(raw)
        self.assertNotIn('I1', audit['by_intent'])
        self.assertIn('INTENT_FILTERED', audit['bindings'][0]['errors'])
        self.assertEqual(audit['by_intent']['I2']['binding']['target_distinction'], 'Proposed distinction 2')
        self.assertEqual(audit['by_intent']['I3']['binding']['target_distinction'], 'Proposed distinction 3')

    def test_no_imputed_or_duplicate_raw_intent_id(self):
        raw, _, _ = fixtures()
        del raw['intents'][0]['intent_id']
        raw['feedback_bindings'][0]['intent_id'] = 'I001'
        audit = evaluate(raw)
        self.assertIn('INTENT_ID_NOT_EXPLICIT', audit['bindings'][0]['errors'])
        self.assertEqual(audit['by_intent']['I001']['fallback_reason'], 'BINDING_INVALID')
        raw, _, _ = fixtures()
        raw['intents'][1]['intent_id'] = 'I1'
        self.assertIn('DUPLICATE_RAW_INTENT_ID', evaluate(raw)['bindings'][0]['errors'])

    def test_strict_binding_fields_and_values(self):
        mutations = [({'extra': 'not allowed'}, 'BINDING_FIELDS_INVALID'),
            ({'status': 'SEMANTIC_PASS'}, 'BINDING_STATUS_INVALID'),
            ({'source_feedback_ids': ['feedback-alpha', 'feedback-alpha']}, 'SOURCE_IDS_INVALID'),
            ({'evidence_refs': ['T1', 'T1']}, 'EVIDENCE_REFS_INVALID'),
            ({'scope': '   '}, 'SCOPE_MISSING'),
            ({'status': 'NO_SUPPORTED_TRANSFER'}, 'NO_SUPPORTED_TRANSFER_NONEMPTY_FIELDS')]
        for mutation, expected in mutations:
            with self.subTest(mutation=mutation):
                raw, _, _ = fixtures()
                raw['feedback_bindings'][0].update(mutation)
                self.assertIn(expected, evaluate(raw)['bindings'][0]['errors'])

    def test_json_valid_malformed_binding_fields_never_crash(self):
        malformed = [None, False, 0, 1.5, [], {}, [None], [{"nested": []}], ""]
        for field in ("intent_id", "status", "source_feedback_ids", "target_distinction", "evidence_refs", "scope", "change_from_source"):
            for value in malformed:
                with self.subTest(field=field, value=value):
                    raw, _, _ = fixtures()
                    raw["feedback_bindings"][0][field] = value
                    audit = evaluate(raw)
                    self.assertFalse(audit["bindings"][0]["mechanical_valid"])
                    self.assertIsNotNone(audit["by_intent"]["I1"]["fallback_reason"])

    def test_exact_contract_union_is_independently_required(self):
        raw, result, contracts = fixtures()
        contracts[0]["required_evidence_groups"]["after_state"] = ["T9"]
        audit = validate_bindings(raw, result, contracts, expanded=EXPANDED, root_id=ROOT, cards=[CARD])
        self.assertIn("REF_OUTSIDE_REASONING_CONTRACT", audit["bindings"][0]["errors"])

    def test_unknown_and_nonobject_rows_are_audited(self):
        raw, _, _ = fixtures()
        raw['feedback_bindings'].append({'intent_id': 'absent', 'status': 'unknown'})
        raw['feedback_bindings'].append(None)
        audit = evaluate(raw)
        self.assertIn('UNKNOWN_INTENT_ID', audit['bindings'][3]['errors'])
        self.assertEqual(audit['bindings'][4]['errors'], ['BINDING_NOT_OBJECT'])
        self.assertEqual(audit['valid_proposed_binding_count'], 3)


class BriefAndTemplateTests(unittest.TestCase):
    def test_free_data_dependency_and_unique_sentinel_isolation(self):
        raw, result, contracts = fixtures()
        sentinel = 'SENTINEL_BINDING_ONLY_9ff8c17'
        contract = deepcopy(contracts[0])
        contract.update(target_distinction=sentinel, scope=sentinel, change_from_source=sentinel,
            discriminative_variable=sentinel, binding={'evidence_excerpt': sentinel})
        first = compile_free_brief(root_id=ROOT, intent_label=contracts[0], cards=[CARD], history_only=[])
        second = compile_free_brief(root_id=ROOT, intent_label=contract, cards=[CARD], history_only=[])
        self.assertEqual(first, second)
        self.assertNotIn(sentinel, json.dumps(second))
        self.assertNotIn('binding', inspect.signature(compile_free_brief).parameters)
        for name in ('target_distinction', 'scope', 'change_from_source', 'required_evidence_groups'):
            self.assertNotIn(name, inspect.getsource(compile_free_brief))
        for binding in raw['feedback_bindings']:
            binding.update(target_distinction=sentinel, scope=sentinel, change_from_source=sentinel)
        with TemporaryDirectory() as directory:
            hooks = TransferHooks('P2_2_CONCRETE_TRANSFER', [CARD], [], 'generic')
            hooks.after_intents(raw, result, contracts, expanded=EXPANDED, root_id=ROOT, root_dir=Path(directory))
            free = hooks.writer_suffix(contract, 'free_exploration', expanded=EXPANDED, root_id=ROOT, root_dir=Path(directory))
            guided = hooks.writer_suffix(contract, 'contract_guided', expanded=EXPANDED, root_id=ROOT, root_dir=Path(directory))
            self.assertNotIn(sentinel, free)
            self.assertIn(sentinel, guided)

    def test_guided_fallback_exact_generic_and_repair_byte_reuse(self):
        raw, result, contracts = fixtures()
        raw['feedback_bindings'] = []
        generic = 'FROZEN GENERIC WRITER\nwith exact bytes\n'
        with TemporaryDirectory() as directory:
            path = Path(directory)
            hooks = TransferHooks('P2_2_CONCRETE_TRANSFER', [CARD], [], generic)
            hooks.after_intents(raw, result, contracts, expanded=EXPANDED, root_id=ROOT, root_dir=path)
            suffix = hooks.writer_suffix(contracts[0], 'contract_guided', expanded=EXPANDED, root_id=ROOT, root_dir=path)
            self.assertEqual(suffix, generic)
            hooks.generic_writer_suffix = 'MUTATED AFTER COMPILE'
            self.assertEqual(hooks.writer_suffix(contracts[0], 'contract_guided', expanded=EXPANDED, root_id=ROOT, root_dir=path), suffix)
            for attempt, request in [('primary', 'first'), ('repair', 'second')]:
                hooks.writer_request({'prompt': attempt + '\n' + suffix}, contracts[0], 'contract_guided', attempt, request,
                    expanded=EXPANDED, root_id=ROOT, root_dir=path)
            records = [json.loads((path/'writer_briefs'/'requests'/f'{request}.json').read_text()) for request in ('first', 'second')]
            self.assertEqual(records[0]['brief_sha256'], records[1]['brief_sha256'])
            self.assertEqual(records[0]['fallback_reason'], 'BINDING_MISSING')
            with self.assertRaisesRegex(RuntimeError, 'WRITER_BRIEF_PROMPT_MISMATCH'):
                hooks.writer_request({'prompt': 'missing suffix'}, contracts[0], 'contract_guided', 'repair', 'third', expanded=EXPANDED, root_id=ROOT, root_dir=path)

    def test_no_active_free_and_history_safe_projection(self):
        sentinel = 'INACTIVE_SCORE_SECRET_29f22'
        inactive = {**CARD, 'source_quality_status': 'AMBIGUOUS_HISTORY_ONLY', 'historical_gold': sentinel}
        history = {**inactive, 'stem': 'Safe tracked old stem', 'correct_count': sentinel}
        brief = compile_free_brief(root_id=ROOT, intent_label=fixtures()[2][0], cards=[inactive], history_only=[history])
        self.assertEqual(brief['observation_cards'], [])
        self.assertEqual(brief['effective_feedback_mode'], 'NO_ACTIVE_FEEDBACK')
        self.assertNotIn(sentinel, json.dumps(brief))
        self.assertEqual(brief['historical_tracking'][0]['stem'], 'Safe tracked old stem')

    def test_common_shape_only_revision_no_new_concrete_example(self):
        original = build_intent_prompt(EXPANDED)
        revised = revise_miner_prompt(original)
        self.assertIn('workload sustainability', original['prompt'])
        self.assertNotIn('workload sustainability', revised['prompt'])
        for literal in ('[12]', '[31]', '[42]'):
            self.assertNotIn(literal, revised['prompt'])
        self.assertNotIn('feedback_bindings', revised['prompt'])
        prefix = original['prompt'].split('Return exactly one JSON object with this shape:')[0]
        self.assertTrue(revised['prompt'].startswith(prefix))
        self.assertEqual(original['prompt'].split('NUMBERED AUTHORITATIVE DIALOGUE:')[1], revised['prompt'].split('NUMBERED AUTHORITATIVE DIALOGUE:')[1])
        self.assertIn('feedback_bindings', revise_miner_prompt(original, envelope=True)['prompt'])
        with self.assertRaisesRegex(ValueError, 'MINER_EXAMPLE_SOURCE_DRIFT'):
            revise_miner_prompt({**original, 'prompt': original['prompt'].replace('workload sustainability', 'different theme')})

    def test_legacy_common_base_no_new_response_fields(self):
        raw, result, contracts = fixtures()
        with TemporaryDirectory() as directory:
            hooks = TransferHooks('LEGACY_COMMON_BASE', [CARD], [], 'generic frozen', 'safe old feedback suffix')
            spec = hooks.miner_prompt(build_intent_prompt(EXPANDED), expanded=EXPANDED, root_id=ROOT, root_dir=Path(directory))
            self.assertNotIn('feedback_bindings', spec['prompt'])
            self.assertIn('safe old feedback suffix', spec['prompt'])
            self.assertNotIn('workload sustainability', spec['prompt'])
            self.assertIs(hooks.project_miner_response(raw, expanded=EXPANDED, root_id=ROOT, root_dir=Path(directory)), raw)
            for source in ('contract_guided', 'free_exploration'):
                self.assertEqual(hooks.writer_suffix(contracts[0], source, expanded=EXPANDED, root_id=ROOT, root_dir=Path(directory)), 'generic frozen')
        with self.assertRaises(ValueError):
            TransferHooks('ORIGINAL_HOOK_DISABLED', [], [], '')

    def test_semantic_mismatch_records_supplied_verdict_without_gate(self):
        raw, result, contracts = fixtures()
        before = deepcopy((raw, result, contracts))
        review = record_semantic_diagnostic(intent_id='I1', judgment='FAIL', reason='Direct review finds a different tested distinction.', evidence_refs=['T1'])
        self.assertEqual(review['diagnostic_code'], 'BINDING_INTENT_MISMATCH')
        self.assertFalse(review['affects_candidate_admission'])
        self.assertEqual((raw, result, contracts), before)

    def test_public_brief_schemas_validate_compiled_payloads(self):
        import jsonschema
        from .transfer import binding_schema
        raw, _, contracts = fixtures()
        jsonschema.validate(raw['feedback_bindings'][0], binding_schema())
        directory = Path(__file__).with_name('schemas')
        audit = evaluate(raw)
        guided = compile_guided_brief(root_id=ROOT, contract=contracts[0], validation=audit, expanded=EXPANDED, cards=[CARD])
        free = compile_free_brief(root_id=ROOT, intent_label=contracts[0], cards=[CARD], history_only=[])
        jsonschema.validate(guided, json.loads((directory/'guided_brief_schema.json').read_text()))
        jsonschema.validate(free, json.loads((directory/'free_brief_schema.json').read_text()))
        audit = evaluate(raw, cards=[])
        fallback = compile_guided_brief(root_id=ROOT, contract=contracts[0], validation=audit, expanded=EXPANDED, cards=[])
        jsonschema.validate(fallback, json.loads((directory/'guided_brief_schema.json').read_text()))


if __name__ == '__main__':
    unittest.main()
