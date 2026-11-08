"""Mechanical tests of retry allocation boundaries; no generated samples."""
import copy
import unittest
import prepare_v120 as p


class RetryMatching(unittest.TestCase):
    def setUp(self):
        self.history = {'a_continuations': [{'scene_ref': s, 'thread': 'inherited thread'} for s in p.SCENES[:3]]}
        self.row = {'new_b_scene_refs': [p.SCENES[-1]]}
        self.pool = {'multiwoz22:train:fresh.json': {'turns': []}}
        self.raw = {'read_attestation': {k: True for k in ['full_source_read', 'all_continuation_constraints_read', 'all_candidate_events_fully_read', 'retained_native_events_fully_read']},
                    'a_continuations': copy.deepcopy(self.history['a_continuations']),
                    'allocations': [{'scene_ref': p.SCENES[-1], 'source_id': 'multiwoz22:train:fresh.json', 'compatibility_reason': 'Compatible without new speakers.', 'main_purpose': 'New purpose.'}]}

    def check(self, consumed=()):
        return p.validate_payload(self.raw, self.row, self.history, self.pool, set(consumed))

    def test_exact_remaining_slot_accepted(self):
        self.assertEqual(len(self.check()), 1)

    def test_historical_a_cannot_be_changed(self):
        self.raw['a_continuations'][1]['thread'] = 'replacement'
        with self.assertRaisesRegex(ValueError, 'INHERITED_A_PLAN_CHANGED'): self.check()

    def test_retained_slot_cannot_be_newly_allocated(self):
        self.raw['allocations'][0]['scene_ref'] = p.SCENES[3]
        with self.assertRaisesRegex(ValueError, 'ONLY_EXACT_NEW_B_SLOTS'): self.check()

    def test_missing_slot_rejected(self):
        self.raw['allocations'] = []
        with self.assertRaisesRegex(ValueError, 'ONLY_EXACT_NEW_B_SLOTS'): self.check()

    def test_consumed_including_unexecuted_cannot_be_reused(self):
        with self.assertRaisesRegex(ValueError, 'SOURCE_UNAVAILABLE_OR_CONSUMED'):
            self.check(['multiwoz22:train:fresh.json'])

    def test_undeclared_candidate_rejected(self):
        self.raw['allocations'][0]['source_id'] = 'multiwoz22:train:other.json'
        with self.assertRaisesRegex(ValueError, 'SOURCE_UNAVAILABLE_OR_CONSUMED'): self.check()

    def test_full_event_read_attestation_required(self):
        self.raw['read_attestation']['retained_native_events_fully_read'] = False
        with self.assertRaisesRegex(ValueError, 'MATCHING_READ_ATTESTATION'): self.check()

    def test_extra_private_payload_field_rejected(self):
        self.raw['answer_key'] = []
        with self.assertRaisesRegex(ValueError, 'MATCHING_PAYLOAD_FIELDS'): self.check()




class RealMatchingChain(unittest.TestCase):
    """Actual read/save I/O against an isolated, openly synthetic fixture."""
    def setUp(self):
        import tempfile
        from pathlib import Path
        from contextlib import redirect_stdout
        import io
        self.tmp = tempfile.TemporaryDirectory()
        self.old_b = p.B
        p.B = Path(self.tmp.name) / 'batch'
        self.root = 'cc-train-episode-1'; self.author = '/root/synthetic_author'
        self.unit = p.B / 'units/root1'
        base = self.unit / 'inputs/BASELINE.json'
        p.save(base, {'root_id': self.root, 'turns': [{'turn_ref':'R01:S01:T001','speaker':'A','text':'Synthetic source only.'}]})
        constraints = self.unit / 'inputs/CONTINUATION_CONSTRAINTS.json'
        p.save(constraints, {'root_id': self.root, 'decision':'PASS','continuation_constraints':['Synthetic only.'],
                            'baseline_binding':p.bind(base),'review_bindings':[]})
        p.save(self.unit / 'PREFIX_IMPORT.json', {'synthetic_fixture':True})
        p.save(self.unit / 'HISTORICAL_SELECTION.json', {'a_continuations':[{'scene_ref':s,'thread':'Inherited synthetic thread.'} for s in p.SCENES[:3]]})
        old = p.B / 'history/inputs/CONTINUATION_CONSTRAINTS.json'
        p.save(old, {'synthetic_fixture':True})
        p.save(old.parents[1] / 'SOURCE_RESERVATIONS.json', {'allocation':[]})
        p.save(self.unit / 'CONSTRAINTS_PATH_ADAPTATION.json', {'original_binding':p.bind(old)})
        self.row = {'root_id':self.root,'prefix_reuse_approved':True,'unit_path':str(self.unit),
                    'baseline_binding':p.bind(base),'prefix_import_binding':p.bind(self.unit/'PREFIX_IMPORT.json'),
                    'candidate_prefix_binding':p.bind(base),'retained_scene_refs':[],
                    'new_b_scene_refs':list(p.SCENES[3:]),'retry_start':p.SCENES[0]}
        p.save(p.B/'MANIFEST.json', {'roots':[self.row]})
        p.save(p.B/'ROLE_ASSIGNMENTS.json', {'roots':[{'root_id':self.root,'author':self.author,'reviewer':'/root/synthetic_reviewer'}]})
        p.save(self.unit/'inputs/EVENT_POOL.json', {'events':[{'dialogue_id':f'event{i}.json','turns':[{'turn_id':'0','utterance':'Synthetic native event.'}]} for i in range(6)]})
        p.save(p.B/'evidence/SOURCE_POOL_FREEZE.json', {'consumed_source_ids':[]})
        p.save(p.B/'evidence/RETRY_EXECUTION_POLICY.json', {'synthetic_fixture':True})
        (p.B/'evidence/CONTENT_CONTRACT.md').write_text('SYNTHETIC TEST ONLY\n')
        files=[q for q in p.B.rglob('*') if q.is_file()]
        p.save(p.B/'MASTER_FREEZE.json', {'status':'FROZEN_BEFORE_V120_FORMAL_AUTHOR_MATCHING_AND_GENERATION','files':[p.bind(q) for q in files]})
        self.payload = p.B/'raw_payload.json'
        p.save(self.payload, {'read_attestation':{k:True for k in ['full_source_read','all_continuation_constraints_read','all_candidate_events_fully_read','retained_native_events_fully_read']},
                             'a_continuations':p.read(self.unit/'HISTORICAL_SELECTION.json')['a_continuations'],
                             'allocations':[{'scene_ref':s,'source_id':f'multiwoz22:train:event{i}.json','compatibility_reason':'Synthetic only.','main_purpose':'Synthetic only.'} for i,s in enumerate(p.SCENES[3:])]})
        with redirect_stdout(io.StringIO()): p.match_read(self.root,self.author)

    def tearDown(self):
        p.B=self.old_b; self.tmp.cleanup()

    def save_match(self):
        from contextlib import redirect_stdout
        import io
        with redirect_stdout(io.StringIO()): p.match_save(self.root,self.author,self.payload)

    def test_actual_read_save_produces_three_new_reservations(self):
        self.save_match()
        self.assertEqual(p.read(self.unit/'SOURCE_RESERVATIONS.json')['new_consumption_count'],3)
        self.assertTrue((self.unit/'EXECUTION_INPUTS_FROZEN.json').is_file())

    def test_input_binding_omission_rejected_before_outputs(self):
        import json
        q=self.unit/'evidence/MATCHING/READ_CONTEXT.json'; obj=p.read(q); obj['input_bindings']=[]; q.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError,'MATCHING_EXACT_INPUT_SET'): self.save_match()
        self.assertFalse((self.unit/'selection.json').exists())

    def test_consistently_rehashed_incomplete_render_rejected(self):
        import json
        folder=self.unit/'evidence/MATCHING'; render=folder/'FULL_INPUT.txt'; render.write_text('Synthetic incomplete render\n')
        q=folder/'READ_CONTEXT.json'; obj=p.read(q); obj['input_render_binding']=p.bind(render); q.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError,'MATCHING_RENDER_CHANGED_OR_INCOMPLETE'): self.save_match()
        self.assertFalse((self.unit/'selection.json').exists())


if __name__ == '__main__': unittest.main(verbosity=2)
