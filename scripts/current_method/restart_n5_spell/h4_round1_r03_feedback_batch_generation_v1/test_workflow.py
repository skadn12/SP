import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from .contract import validate_intents
from .inputs import observations
from .prompts import generator_prompt
from .run import filter_candidates, generate_root, save
from ..backends import BackendResponse


def fixture():
    refs = ['P:T001', 'R02:S01:T001', 'R03:S01:T001']
    dialogue = {'turns': [{'turn_ref': r, 'speaker': 'Taylor', 'text': t} for r, t in zip(refs, [
        'The Friday meeting was planned for the cafe.',
        'The cafe cancelled our reservation; I will ask the library.',
        'The library has confirmed our Friday meeting; the cafe plan is cancelled.'])]}
    packet = {'eligible_feedback': [{'feedback_id': 'F001', 'solver_observations': {'accuracy': 'PRIVATE_SCORE'}}],
              'coverage_gaps': [{'gap_id': 'G001', 'uncited_turn_refs': refs[1:]}],
              'history': [], 'dialogue_sha256': 'fixture'}
    intents = []
    for i, slot in enumerate(['SUCCESSOR_FRONTIER', 'COVERAGE_GAP', 'FRESH_R03'], 1):
        intents.append({'intent_id': f'I{i:03}', 'intent_type': slot, 'status': 'READY',
                        'strategy': 'TEMPORAL_RELATION', 'target_relation': 'Planned venue versus confirmed replacement',
                        'source_feedback_ids': ['F001'] if i == 1 else [],
                        'coverage_gap_ids': ['G001'] if i == 2 else [],
                        'reasoning_chain': dict(zip(['before_state', 'trigger', 'after_state'], [[r] for r in refs])),
                        'required_r03_evidence': refs[2:], 'no_safe_reason': None})
    return dialogue, packet, {'intents': intents}


def no_safe(intent):
    return {**intent, 'status': 'NO_SAFE_EVOLUTION', 'strategy': 'NO_SAFE_EVOLUTION',
            'target_relation': None, 'source_feedback_ids': [], 'coverage_gap_ids': [],
            'reasoning_chain': {k: [] for k in intent['reasoning_chain']},
            'required_r03_evidence': [], 'no_safe_reason': 'Insufficient distinct supported relation.'}


def candidate():
    return {'candidate_id': 'new', 'question_type': 'single_choice', 'question': 'Where is the confirmed meeting?',
            'options': dict(zip('ABCD', ['Library', 'Cafe', 'Park', 'Office'])), 'items': {},
            'correct_option': 'A', 'gold': ['A'], 'evidence_refs': ['R03:S01:T001'],
            'reasoning_groups_used': ['after_state'], 'slot_type': 'FRESH_R03',
            'intent_anchors': ['R03:S01:T001'], 'required_r03_evidence': ['R03:S01:T001']}


class WorkflowTests(unittest.TestCase):
    def test_ref_and_feedback_identity_are_enforced(self):
        d, p, raw = fixture()
        self.assertTrue(validate_intents(raw, d, p)['valid'])
        raw['intents'][0]['source_feedback_ids'] = ['invented']
        raw['intents'][2]['required_r03_evidence'] = ['R03:invented']
        result = validate_intents(raw, d, p)
        self.assertFalse(result['slots'][0]['generation_allowed'])
        self.assertFalse(result['slots'][2]['generation_allowed'])
        self.assertTrue(result['slots'][1]['generation_allowed'])

    def test_model_cannot_grant_permission_or_add_strategy(self):
        d, p, raw = fixture()
        raw['intents'][0]['generation_allowed'] = True
        self.assertFalse(validate_intents(raw, d, p)['valid'])
        del raw['intents'][0]['generation_allowed']
        raw['intents'][0]['strategy'] = 'MODEL_WEAKNESS'
        self.assertFalse(validate_intents(raw, d, p)['valid'])

    def test_no_safe_slots_make_no_generator_calls(self):
        d, p, raw = fixture()
        raw['intents'] = [no_safe(i) for i in raw['intents']]
        calls = []
        def complete(**kwargs):
            calls.append(kwargs)
            return BackendResponse(json.dumps(raw), 'fixture', {})
        client = SimpleNamespace(config=SimpleNamespace(model='fixture', base_url='fixture'), complete=complete)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            rd = out / 'fixture-root'
            save(rd / 'dialogue_source.json', d)
            save(rd / 'feedback_packet.json', p)
            (rd / 'intent_prompt.txt').write_text('fixture')
            report = generate_root(out, 'fixture-root', {'test': True}, client, backend_mode='mock')
            self.assertEqual(report['scheduled_intent_count'], 0)
            self.assertEqual(len(calls), 1)
            self.assertEqual(report['candidate_count'], 0)

    def test_repair_occurs_once_per_missing_root_type(self):
        d, p, raw = fixture()
        calls = []
        row = candidate()
        row.update(evidence_groups={'after_state': ['R03:S01:T001']}, discriminative_variable='confirmed venue')
        def complete(**kwargs):
            calls.append(kwargs['prompt'])
            value = raw if len(calls) == 1 else {'candidates': [row]}
            if 'Generate ONLY question_type' in kwargs['prompt']:
                value = {'candidates': []}
            return BackendResponse(json.dumps(value), 'fixture', {})
        client = SimpleNamespace(config=SimpleNamespace(model='fixture', base_url='fixture'), complete=complete)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            rd = out / 'fixture-root'
            save(rd / 'dialogue_source.json', d)
            save(rd / 'feedback_packet.json', p)
            (rd / 'intent_prompt.txt').write_text('fixture')
            report = generate_root(out, 'fixture-root', {'test': True}, client, backend_mode='mock')
            self.assertEqual(len(calls), 9)  # miner + 6 primaries + multi/ordering repairs
            self.assertEqual(report['structurally_accepted_count'], 1)
            self.assertEqual(sum('Generate ONLY question_type' in c for c in calls), 2)
            self.assertEqual(report['ledger']['paid_api_generation_count'], 0)

    def test_writer_does_not_receive_solver_observations(self):
        d, p, raw = fixture()
        for mode in ('contract_guided', 'free_exploration'):
            prompt = generator_prompt(d, raw['intents'][0], p['history'], mode)
            self.assertNotIn('PRIVATE_SCORE', prompt)
            self.assertNotIn('solver_observations', prompt)
            self.assertNotIn('Generate exactly', prompt)
            self.assertIn('Target type counts', prompt)
            self.assertIn('APPROVED_INTENT', prompt)
            self.assertIn('R03:S01:T001', prompt)

    def test_type_focus_makes_only_two_batches_and_rejects_single_fallback(self):
        d, p, raw = fixture()
        calls = []
        row = candidate()
        row.update(evidence_groups={'after_state': ['R03:S01:T001']}, discriminative_variable='venue')
        def complete(**kwargs):
            calls.append(kwargs['prompt'])
            return BackendResponse(json.dumps(raw if len(calls)==1 else {'candidates':[row]}), 'fixture', {})
        client = SimpleNamespace(config=SimpleNamespace(model='fixture',base_url='fixture'),complete=complete)
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder); rd=out/'fixture-root'
            save(rd/'dialogue_source.json',d); save(rd/'feedback_packet.json',p)
            (rd/'intent_prompt.txt').write_text('fixture')
            report=generate_root(out,'fixture-root',{'test':True},client,backend_mode='mock',
                                 balanced_type_minimums={'multi_select':2,'ordering':2},type_focus_only=True)
            self.assertEqual(len(calls),3)
            self.assertEqual(report['structurally_accepted_count'],0)
            self.assertEqual(report['type_shortfalls'],{'multi_select':2,'ordering':2})
            self.assertIn('Generate ONLY question_type multi_select; target 2',calls[1])
            self.assertIn('Generate ONLY question_type ordering; target 2',calls[2])

    def test_fewer_candidates_and_bad_sibling_do_not_discard_valid_row(self):
        d, p, _ = fixture()
        good = candidate()
        bad = copy.deepcopy(good)
        bad.update(candidate_id='bad', question='Bad?', evidence_refs=['invented'])
        accepted, audits, _ = filter_candidates([good, bad], d, [])
        self.assertEqual([r['candidate_id'] for r in accepted], ['new'])
        self.assertIn('evidence_ref_unknown', audits[1]['filter_reasons'])
        accepted, audits, _ = filter_candidates([good], d, [copy.deepcopy(good)])
        self.assertEqual(accepted, [])
        self.assertIn('duplicate_content_in_root_bank', audits[0]['filter_reasons'])

    def test_independent_real_r03_anchor_and_neutral_defaults(self):
        d, p, raw = fixture()
        d['turns'].append({'turn_ref': 'R03:S01:T002', 'speaker': 'Taylor', 'text': 'I have the confirmation.'})
        raw['intents'][0]['required_r03_evidence'] = ['R03:S01:T002']
        del raw['intents'][0]['no_safe_reason']
        self.assertTrue(validate_intents(raw, d, p)['slots'][0]['generation_allowed'])

    def test_same_stem_different_options_is_review_not_auto_duplicate(self):
        d, _, _ = fixture()
        a = candidate()
        b = copy.deepcopy(a)
        b['options']['B'] = 'Restaurant'
        accepted, audits, _ = filter_candidates([b], d, [a])
        self.assertEqual(len(accepted), 1)
        self.assertTrue(audits[0]['lexical_similarity_review'])

    def test_invalid_no_safe_slot_does_not_discard_ready_siblings(self):
        d, p, raw = fixture()
        raw['intents'][1] = no_safe(raw['intents'][1])
        raw['intents'][1]['coverage_gap_ids'] = ['G001']
        result = validate_intents(raw, d, p)
        self.assertFalse(result['valid'])
        self.assertEqual([x['generation_allowed'] for x in result['slots']], [True, False, True])

    def test_citation_does_not_claim_semantic_admission(self):
        d, _, _ = fixture()
        accepted, audits, _ = filter_candidates([candidate()], d, [])
        self.assertEqual(len(accepted), 1)
        self.assertEqual(audits[0]['semantic_evidence_support'], 'NOT_VALIDATED')
        self.assertEqual(audits[0]['r03_necessity'], 'UNKNOWN')

    def test_diagnoses_and_strategy_recommendations_are_not_compiled(self):
        result = observations({'M_clean': {'full_context_accuracy': .2,
            'difficulty_bucket': 'hard', 'weakness': 'temporal', 'recommended_strategy': 'TEMPORAL'}})
        self.assertEqual(result, {'M_clean': {'full_context_accuracy': .2}})


if __name__ == '__main__':
    unittest.main()
