"""Audit accepted/rejected/replaced candidates without changing generation."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ..core import atomic_json
from ..h4_qa_multiroot_k3_t08_probe_v1.generation import generate_root_qa
from .run import make_hooks, token_components
from .runtime import FIRST_ROOTS, root_path, BoundedLedger
from .reporting import mechanical_report, jsonl
from .test_generation_hooks import expanded_fixture, intents_fixture, batch_fixture, FakeClient


class ReportingTests(unittest.TestCase):
    def test_all_returned_candidates_preserved_and_prompts_rebuilt(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            for root_id in FIRST_ROOTS:
                expanded = expanded_fixture(); expanded['root_id'] = root_id
                inputs = run / 'inputs' / root_id
                atomic_json(inputs / 'expanded_dialogue_clean.json', expanded)
                atomic_json(inputs / 'observation_cards.json', {'root_id': root_id, 'feedback_status': 'NO_ACTIVE_FEEDBACK', 'observation_cards': []})
                atomic_json(inputs / 'history_only.json', {'root_id': root_id, 'records': []})
                atomic_json(inputs / 'provenance.json', {})
                miner = intents_fixture()
                miner['feedback_bindings'] = [{'intent_id': row['intent_id'], 'status': 'NO_SUPPORTED_TRANSFER',
                    'source_feedback_ids': [], 'evidence_refs': [], 'target_distinction': '', 'scope': '', 'change_from_source': ''} for row in miner['intents']]
                # Three primary rows are replaced; one malformed extra in repair
                # is rejected; original structural minima still retain four.
                repair = batch_fixture('repair')
                repair['candidates'].append({'question_type': 'unknown', 'question': 'invalid'})
                replies = [miner, batch_fixture('replaced', (2, 1, 0)), repair] + [batch_fixture()] * 5
                root = root_path(run, root_id)
                client = FakeClient(replies)
                ledger = BoundedLedger(root, 'fake', run, root_id, backend_mode='mock')
                result = generate_root_qa(expanded=expanded, root_dir=root, root_id=root_id,
                    api_config=Path('/unused'), ledger=ledger, client=client, hooks=make_hooks(run, root_id))
                atomic_json(root / 'qa_generation_report.json', result)
            result = mechanical_report(run)
            self.assertEqual(result['logical_calls'], 24)
            self.assertEqual(result['repair_calls'], 3)
            self.assertEqual(sum(row['valid_candidates'] for row in result['roots']), 72)
            trace = jsonl(run / 'reports/source_to_writer_trace.jsonl')
            self.assertEqual(sum(row['candidate_status'] == 'accepted' for row in trace), 72)
            self.assertEqual(sum(row['candidate_status'] == 'replaced' for row in trace), 9)
            self.assertEqual(sum(row['candidate_status'] == 'rejected' for row in trace), 3)
            audit = json.loads((run / 'reports/binding_validation_report.json').read_text())
            self.assertEqual(audit['errors'], [])
            root_id = FIRST_ROOTS[0]
            request_id = f'{root_id}-contract_guided-generator-I001'
            prompt = (root_path(run, root_id) / 'raw/prompts' / f'{request_id}.txt').read_text()
            components = token_components(run, root_id, prompt, expanded, request_id)
            self.assertEqual(components['feedback'], '')
            self.assertEqual(components['binding'], '')
            path = root_path(run, root_id) / 'raw/prompts' / f'{request_id}.txt'
            path.write_text(prompt + 'tamper')
            mechanical_report(run)
            audit = json.loads((run / 'reports/binding_validation_report.json').read_text())
            self.assertTrue(audit['errors'])


if __name__ == '__main__':
    unittest.main()
